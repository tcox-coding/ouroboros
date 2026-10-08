"""Vision-LLM backends for the judge. Each takes instructions, an ordered list of
parts ({"text": str} or {"image": Path | PIL.Image}), and a JSON schema, and returns
(parsed JSON, cost in USD, prompt tokens used). `context_window` is the model's
context size in tokens, so callers can keep prompts well inside it.

Pick one with judge.backend in config.json: "deepinfra" (hosted models), "ollama"
(a model on the LAN) or "openai".
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
import math
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

from .sizes import to_rgb
from .usage import charge


def encode_jpeg(image: Path | Image.Image, max_side: int) -> str:
    img = to_rgb(Image.open(image) if isinstance(image, Path) else image)
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


class OllamaBackend:
    """Ollama /api/chat with structured outputs (format = JSON schema).

    Needs a vision model (llama3.2-vision, qwen3-vl, gemma3, llava, ...). Models that
    accept only one image per message (llama3.2-vision and its fine-tunes) need
    judge.contact_sheet = true: the reference and candidates go as one labelled grid.
    """

    def __init__(self, cfg: dict):
        self.url = cfg["url"].rstrip("/")
        self.cfg = cfg

    @property
    def context_window(self) -> int:
        return int(self.cfg.get("num_ctx") or 16384)

    def complete(self, instructions: str, parts: list[dict], schema: dict, name: str, max_side: int):
        texts, images = [], []
        for part in parts:
            if "image" in part:
                images.append(encode_jpeg(part["image"], part.get("max_side", max_side)))
                texts.append(f"(attached image {len(images)})")
            else:
                texts.append(part["text"])
        if len(images) > 1:
            texts.insert(0, f"{len(images)} images are attached, in the order of the (attached image N) markers below.")
        body = {
            "model": self.cfg["model"],
            "messages": [
                # Compact schema text grounds the field meanings for about half the tokens of
                # spaced JSON; format= below is what actually enforces it.
                {"role": "system", "content": instructions + "\nJSON schema:\n"
                                              + json.dumps(schema, separators=(",", ":"))},
                {"role": "user", "content": "\n\n".join(texts), "images": images},
            ],
            "format": schema,
            "stream": False,
            "keep_alive": self.cfg.get("keep_alive", "30m"),  # stay loaded between rounds
            # Unset options keep the model's own defaults. Don't force temperature 0 on
            # thinking models: greedy decoding makes them loop until they run out of tokens.
            "options": {k: self.cfg[k] for k in ("temperature", "top_p", "top_k", "num_ctx", "num_predict")
                        if self.cfg.get(k) is not None},
        }
        if self.cfg.get("think") is not None:
            body["think"] = self.cfg["think"]

        # Sampling is random, so a retry usually succeeds where one answer ran out of
        # tokens (a thinking loop) or came back as broken JSON.
        attempts = 1 + self.cfg.get("retries", 1)
        problem = ""
        for attempt in range(attempts):
            try:
                r = requests.post(f"{self.url}/api/chat", json=body, timeout=self.cfg.get("timeout", 600))
            except requests.RequestException as e:
                # A dropped connection (the LAN machine sleeping, Ollama restarting, a
                # router closing a long-lived socket) shouldn't cost the whole job.
                problem = f"connection failed: {str(e)[:200]}"
                if attempt + 1 < attempts:
                    time.sleep(5)
                continue
            if r.status_code != 200:
                raise RuntimeError(f"Ollama error {r.status_code}: {r.text[:1000]}")
            data = r.json()
            # Ollama's own timings (nanoseconds), for speed comparisons (judge_eval.py).
            self.last_stats = {k: data.get(k) for k in ("total_duration", "load_duration", "prompt_eval_count",
                                                         "prompt_eval_duration", "eval_count", "eval_duration")}
            if data.get("done_reason") == "length":
                problem = ("ran out of tokens (num_predict); the model is probably looping while thinking. "
                           "Try another model or raise num_predict.")
                continue
            message = data["message"]
            content = message.get("content") or ""
            if not content.strip():
                # Some models (qwen3-vl with think=false) come back with the structured
                # answer in the "thinking" field and an empty content.
                content = message.get("thinking") or ""
            try:
                return json.loads(content), 0.0, int(data.get("prompt_eval_count") or 0)
            except json.JSONDecodeError:
                problem = f"returned invalid JSON: {content[:500]!r}"
        raise RuntimeError(f"Ollama {problem} ({attempts} attempts)")


def ollama_models(url: str) -> list[str]:
    r = requests.get(f"{url.rstrip('/')}/api/tags", timeout=5)
    r.raise_for_status()
    return sorted(m["name"] for m in r.json().get("models", []))


class OpenAIBackend:
    """OpenAI Responses API with strict structured output. Needs an OpenAI API key
    (Settings -> API keys, or OPENAI_API_KEY)."""

    def __init__(self, cfg: dict):
        from openai import OpenAI  # imported lazily so Ollama-only setups don't need it

        from . import keys
        self.client = OpenAI(api_key=keys.get("openai") or None)
        self.cfg = cfg

    @property
    def context_window(self) -> int:
        return int(self.cfg.get("context_tokens") or 128000)

    def complete(self, instructions: str, parts: list[dict], schema: dict, name: str, max_side: int):
        detail = self.cfg.get("image_detail", "low")
        content = []
        for part in parts:
            if "image" in part:
                url = "data:image/jpeg;base64," + encode_jpeg(part["image"], part.get("max_side", max_side))
                content.append({"type": "input_image", "image_url": url, "detail": detail})
            else:
                content.append({"type": "input_text", "text": part["text"]})
        kwargs = {}
        if self.cfg.get("reasoning_effort"):
            kwargs["reasoning"] = {"effort": self.cfg["reasoning_effort"]}
        resp = self.client.responses.create(
            model=self.cfg["model"],
            instructions=instructions,
            input=[{"role": "user", "content": content}],
            text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
            **kwargs,
        )
        cost = self._cost(resp.usage)
        charge(cost)
        try:
            return json.loads(resp.output_text), cost, int(resp.usage.input_tokens or 0)
        except (ValueError, TypeError) as e:
            raise CompletionError(str(e), cost, int(resp.usage.input_tokens or 0)) from e

    def _cost(self, usage) -> float:
        p = self.cfg["price_per_mtok"]
        cached = getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0
        fresh = usage.input_tokens - cached
        return (fresh * p["input"] + cached * p["cached_input"] + usage.output_tokens * p["output"]) / 1e6


_IMAGE_LIMITS: dict[str, int] = {}  # model -> images per request, from its "Too many images" error
MAX_OUTPUT_TOKENS = 32768  # ceiling when a truncated answer is retried with more room


class capped_output:
    """`with capped_output(backend, tokens, penalty):` caps a call's answer and adds a mild
    repetition penalty, for tasks with a short answer. A prompt is 500-650 tokens; Qwen3-VL at
    temperature 0 sometimes repeats a tag ("hairline at toes, ...") to its 4096-token limit,
    which took ~125 s, past the 120 s read timeout, so the reply never came back to be retried
    (2 of 4 "Write prompts" with LoRA notes, 2026-10-06). Capped, the loop ends in ~45 s and is
    retried nudged off it; a reasoning model that runs out is still given more room."""

    def __init__(self, backend, tokens: int, penalty: float = 0.3):
        self.cfg = getattr(backend, "cfg", None)
        self.set = {}
        if isinstance(self.cfg, dict) and isinstance(backend, DeepInfraBackend):
            self.set = {"max_tokens": min(int(self.cfg.get("max_tokens") or tokens), tokens),
                        "frequency_penalty": max(float(self.cfg.get("frequency_penalty") or 0), penalty)}
        elif isinstance(self.cfg, dict) and isinstance(backend, OllamaBackend):
            self.set = {"num_predict": min(int(self.cfg.get("num_predict") or tokens), tokens)}
        self.saved = {}

    def __enter__(self):
        for k, v in self.set.items():
            self.saved[k] = self.cfg.get(k, _MISSING)
            self.cfg[k] = v
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is _MISSING:
                self.cfg.pop(k, None)
            else:
                self.cfg[k] = v


_MISSING = object()


class CompletionError(RuntimeError):
    """A failed completion can still have billed attempts."""
    def __init__(self, message, cost_usd=0.0, prompt_tokens=0):
        super().__init__(message)
        self.cost_usd, self.prompt_tokens = cost_usd, prompt_tokens


def labelled_sheet(parts: list[dict], cell_side: int) -> tuple[Image.Image, list[str]]:
    images = [p for p in parts if "image" in p]
    cols = math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / cols)
    side = max(128, cell_side)
    sheet = Image.new("RGB", (cols * side, rows * (side + 32)), "#202020")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    labels, previous = [], ""
    i = 0
    for part in parts:
        if "image" not in part:
            previous = part.get("text", "")
            continue
        label = f"Image {i + 1}"
        labels.append(f"{label}: {previous[-160:]}" if previous else label)
        source = part["image"]
        if isinstance(source, Path):
            with Image.open(source) as im:
                img = to_rgb(im).copy()
        else:
            img = to_rgb(source).copy()
        img.thumbnail((side, side))
        x, y = (i % cols) * side, (i // cols) * (side + 32)
        sheet.paste(img, (x + (side - img.width) // 2, y + 32 + (side - img.height) // 2))
        draw.text((x + 8, y + 5), label, fill="white", font=font)
        i += 1
    return sheet, labels


def _decode_json(text: str):
    """Parse the JSON object out of a reply. Hosted models sometimes put their answer in
    a code fence, after a <think> block, or with a sentence around it, even when a schema
    was given."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in the reply")
    return json.JSONDecoder().raw_decode(text, start)[0]


class DeepInfraBackend:
    """DeepInfra's hosted models, through their OpenAI-compatible chat API.

    Any model the catalogue tags "vision" can judge (deepseek-ai/DeepSeek-V4.1-Flash by
    default): images go as data URLs and the answer is constrained by the same JSON
    schema the local backend uses, so nothing else in the loop changes.

    The API key comes from Settings -> API keys (or the DEEPINFRA_API_KEY environment
    variable); see keys.py.

    Unlike Ollama there is no single GPU to keep free, so several calls can be in flight
    at once (queue.llm_parallel).
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.url = (cfg.get("url") or "https://api.deepinfra.com/v1/openai").rstrip("/")
        self.last_stats: dict = {}

    @property
    def api_key(self) -> str:
        from . import keys
        key = keys.get("deepinfra")
        if not key:
            raise RuntimeError("No DeepInfra API key. Add it in Settings -> API keys "
                               "(or set the DEEPINFRA_API_KEY environment variable).")
        return key

    @property
    def context_window(self) -> int:
        return int(self.cfg.get("context_tokens") or 163840)

    def _content(self, parts: list[dict], max_side: int) -> list[dict]:
        """Preserve every image, using numbered panels when the request exceeds a limit."""
        content, n = [], 0
        limit = self.cfg.get("max_images") or _IMAGE_LIMITS.get(self.cfg["model"])
        if limit and sum("image" in p for p in parts) > limit:
            sheet, labels = labelled_sheet(parts, max_side)
            content.append({"type": "text", "text": "All attached images are in one contact sheet. "
                            "Image numbers identify panels, in original order.\n" + "\n".join(labels)})
            content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," +
                            encode_jpeg(sheet, max(sheet.size))}})
            for part in parts:
                if "image" in part:
                    n += 1
                    content.append({"type": "text", "text": f"(see Image {n} in the contact sheet)"})
                else:
                    content.append({"type": "text", "text": part["text"]})
            return content
        for part in parts:
            if "image" in part:
                n += 1
                url = "data:image/jpeg;base64," + encode_jpeg(part["image"], part.get("max_side", max_side))
                content.append({"type": "image_url", "image_url": {"url": url}})
            else:
                content.append({"type": "text", "text": part["text"]})
        return content

    def complete(self, instructions: str, parts: list[dict], schema: dict, name: str, max_side: int):
        content = self._content(parts, max_side)
        body = {
            "model": self.cfg["model"],
            # The schema is also spelled out in the system message: it is what keeps the
            # answer well-formed if the plain-JSON fallback below has to be used.
            "messages": [
                {"role": "system", "content": instructions + "\nJSON schema:\n"
                                              + json.dumps(schema, separators=(",", ":"))},
                {"role": "user", "content": content},
            ],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": name, "schema": schema, "strict": True}},
            "stream": False,
        }
        # seed is passed through when set, but don't count on it: with
        # DeepSeek-V4.1-Flash, three calls at seed 42 still gave two different answers
        # (hosted MoE inference isn't deterministic). Left in for models that do honour it.
        for k in ("temperature", "top_p", "max_tokens", "reasoning_effort", "seed", "frequency_penalty"):
            if self.cfg.get(k) is not None:
                body[k] = self.cfg[k]

        attempts = 1 + int(self.cfg.get("retries", 2))
        problem = ""
        total_cost, total_prompt = 0.0, 0
        self.last_stats = {}
        schema_500s = 0
        attempt = -1
        while (attempt := attempt + 1) < attempts:
            t0 = time.monotonic()
            try:
                r = requests.post(f"{self.url}/chat/completions", json=body,
                                  headers={"Authorization": f"Bearer {self.api_key}"},
                                  timeout=self.cfg.get("timeout", 600))
            except requests.RequestException as e:
                problem = f"connection failed: {str(e)[:200]}"
                if attempt + 1 < attempts:
                    time.sleep(5)
                continue
            if r.status_code in (400, 405, 500) and "response_format" in body and (
                    "json_schema" in r.text or "response_format" in r.text):
                # Not every hosted model takes a schema; plain JSON mode plus the schema
                # in the system message is the fallback, and no format at all after that
                # (gemma-4-31B-it-Ultra answers 405, ByteDance Seed 500 InvalidParameter,
                # and Seed-2.0-code takes neither).
                if body["response_format"]["type"] == "json_schema":
                    body["response_format"] = {"type": "json_object"}
                else:
                    del body["response_format"]
                attempts += 1
                continue
            if r.status_code == 400 and (m := re.search(r"Too many images in request: \d+ > (\d+)", r.text)):
                # MiMo-V2.6-Flash takes at most 4 images, and LoRA picking sends 6.
                limit = int(m.group(1))
                if limit < 1 or _IMAGE_LIMITS.get(self.cfg["model"]) == limit:
                    raise CompletionError("Model cannot accept the contact sheet", total_cost, total_prompt)
                _IMAGE_LIMITS[self.cfg["model"]] = limit
                body["messages"][1]["content"] = self._content(parts, max_side)
                attempts += 1
                continue
            if r.status_code == 500 and (body.get("response_format") or {}).get("type") == "json_schema":
                # ByteDance Seed-2.0 answers a schema with a bare "InternalServiceError"
                # every time: retry once as is (it may be a passing fault), then step down.
                schema_500s += 1
                if schema_500s >= 2:
                    body["response_format"] = {"type": "json_object"}
                    attempts += 1
                    continue
                if attempt + 1 >= attempts:
                    attempts += 1
            if r.status_code in (408, 429, 500, 502, 503, 504):
                problem = f"HTTP {r.status_code}: {r.text[:300]}"
                if attempt + 1 < attempts:
                    time.sleep(5)
                continue
            if r.status_code != 200:
                raise CompletionError(f"DeepInfra error {r.status_code}: {r.text[:1000]}", total_cost, total_prompt)
            data = r.json()
            usage = data.get("usage") or {}
            billed = self._cost(usage)
            charge(billed)
            total_cost += billed
            total_prompt += int(usage.get("prompt_tokens") or 0)
            ns = int((time.monotonic() - t0) * 1e9)
            # Same shape as Ollama's timings, so judge_eval can report tokens per second.
            self.last_stats = {"total_duration": ns, "load_duration": 0,
                               "prompt_eval_count": usage.get("prompt_tokens"), "prompt_eval_duration": 0,
                               "eval_count": usage.get("completion_tokens"), "eval_duration": ns,
                               "attempts": attempt + 1, "cost_usd": total_cost,
                               "billed_prompt_tokens": total_prompt}
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            if choice.get("finish_reason") == "length" and not (
                    (message.get("reasoning_content") or "").strip()
                    or int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)):
                # No reasoning: the answer itself ran on, a model repeating itself (Qwen3-VL at
                # temperature 0 repeats prompt tags). More room only makes it loop longer: the
                # retries doubled max_tokens to 32768 and each one hit the 600 s read timeout,
                # a 40-minute "Write prompts" (2026-10-05). Retry nudged off the loop instead.
                tail = (message.get("content") or "")[-120:]
                problem = f"kept repeating itself until max_tokens ({body.get('max_tokens', '?')}): ...{tail!r}"
                if float(body.get("frequency_penalty") or 0) < 0.5:
                    body["frequency_penalty"] = 0.5
                    body["temperature"] = max(float(body.get("temperature") or 0), 0.4)
                    attempts += 1  # the nudged retry shouldn't count
                continue
            if choice.get("finish_reason") == "length":
                # Reasoning counts against max_tokens, and an open-ended task (writing a
                # prompt from a reference image) can spend the lot on thinking. Retrying
                # the same request would just hit the same wall, so give the next attempt
                # more room instead.
                room = int(body.get("max_tokens") or 8192)
                problem = (f"ran out of tokens ({room}); the model is probably looping while reasoning. "
                           "Raise judge.deepinfra.max_tokens or pick another model.")
                ceiling = min(MAX_OUTPUT_TOKENS, int(self.cfg.get("max_output_tokens") or MAX_OUTPUT_TOKENS))
                if room < ceiling:
                    body["max_tokens"] = min(ceiling, room * 2)
                    attempts += 1  # the retry that just gained headroom shouldn't count
                continue
            # Reasoning models put the answer in content and their thinking in
            # reasoning_content; a few return only the latter.
            text = (message.get("content") or "").strip() or (message.get("reasoning_content") or "")
            try:
                # Callers use this token count as the size of one context, not billed usage.
                return _decode_json(text), total_cost, int(usage.get("prompt_tokens") or 0)
            except (json.JSONDecodeError, ValueError):
                problem = f"returned invalid JSON: {text[:500]!r}"
        raise CompletionError(f"DeepInfra {problem} ({attempts} attempts)", total_cost, total_prompt)

    def _cost(self, usage: dict) -> float:
        if usage.get("estimated_cost") is not None:
            return float(usage["estimated_cost"])
        p = self.cfg.get("price_per_mtok") or {}
        cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
        fresh = (usage.get("prompt_tokens") or 0) - cached
        return (fresh * p.get("input", 0.0) + cached * p.get("cached_input", p.get("input", 0.0))
                + (usage.get("completion_tokens") or 0) * p.get("output", 0.0)) / 1e6


def deepinfra_models(url: str = "", vision_only: bool = True) -> list[str]:
    """The catalogue, fetched fresh (it changes often). Only models tagged "vision" can
    judge images, so by default only those are offered."""
    base = (url or "https://api.deepinfra.com/v1/openai").rstrip("/")
    r = requests.get(f"{base}/models", timeout=15)
    r.raise_for_status()
    out = []
    for m in r.json().get("data", []):
        tags = (m.get("metadata") or {}).get("tags") or []
        if "vision" in tags or not vision_only:
            out.append(m["id"])
    return sorted(out)


BACKENDS = {"ollama": OllamaBackend, "openai": OpenAIBackend, "deepinfra": DeepInfraBackend}


def make_backend(judge_cfg: dict, role: str = "judge"):
    from .model_profiles import role_config
    judge_cfg = role_config(judge_cfg, role)
    name = judge_cfg.get("backend", "ollama")
    if name not in BACKENDS:
        raise ValueError(f"Unknown judge backend '{name}' (choose from {', '.join(BACKENDS)})")
    return BACKENDS[name](judge_cfg[name])
