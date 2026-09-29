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
from pathlib import Path

import requests
from PIL import Image

from .sizes import to_rgb


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
        return json.loads(resp.output_text), self._cost(resp.usage), int(resp.usage.input_tokens or 0)

    def _cost(self, usage) -> float:
        p = self.cfg["price_per_mtok"]
        cached = getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0
        fresh = usage.input_tokens - cached
        return (fresh * p["input"] + cached * p["cached_input"] + usage.output_tokens * p["output"]) / 1e6


MAX_OUTPUT_TOKENS = 32768  # ceiling when a truncated answer is retried with more room


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

    def complete(self, instructions: str, parts: list[dict], schema: dict, name: str, max_side: int):
        content = []
        for part in parts:
            if "image" in part:
                url = "data:image/jpeg;base64," + encode_jpeg(part["image"], part.get("max_side", max_side))
                content.append({"type": "image_url", "image_url": {"url": url}})
            else:
                content.append({"type": "text", "text": part["text"]})
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
        for k in ("temperature", "top_p", "max_tokens", "reasoning_effort", "seed"):
            if self.cfg.get(k) is not None:
                body[k] = self.cfg[k]

        attempts = 1 + int(self.cfg.get("retries", 2))
        problem = ""
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
            if r.status_code == 400 and "json_schema" in r.text and body["response_format"]["type"] != "json_object":
                # Not every hosted model takes a schema; plain JSON mode plus the schema
                # in the system message is the fallback.
                body["response_format"] = {"type": "json_object"}
                continue
            if r.status_code in (408, 429, 500, 502, 503, 504):
                problem = f"HTTP {r.status_code}: {r.text[:300]}"
                if attempt + 1 < attempts:
                    time.sleep(5)
                continue
            if r.status_code != 200:
                raise RuntimeError(f"DeepInfra error {r.status_code}: {r.text[:1000]}")
            data = r.json()
            usage = data.get("usage") or {}
            ns = int((time.monotonic() - t0) * 1e9)
            # Same shape as Ollama's timings, so judge_eval can report tokens per second.
            self.last_stats = {"total_duration": ns, "load_duration": 0,
                               "prompt_eval_count": usage.get("prompt_tokens"), "prompt_eval_duration": 0,
                               "eval_count": usage.get("completion_tokens"), "eval_duration": ns}
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            if choice.get("finish_reason") == "length":
                # Reasoning counts against max_tokens, and an open-ended task (writing a
                # prompt from a reference image) can spend the lot on thinking. Retrying
                # the same request would just hit the same wall, so give the next attempt
                # more room instead.
                room = int(body.get("max_tokens") or 8192)
                problem = (f"ran out of tokens ({room}); the model is probably looping while reasoning. "
                           "Raise judge.deepinfra.max_tokens or pick another model.")
                if room < MAX_OUTPUT_TOKENS:
                    body["max_tokens"] = min(MAX_OUTPUT_TOKENS, room * 2)
                    attempts += 1  # the retry that just gained headroom shouldn't count
                continue
            # Reasoning models put the answer in content and their thinking in
            # reasoning_content; a few return only the latter.
            text = (message.get("content") or "").strip() or (message.get("reasoning_content") or "")
            try:
                return _decode_json(text), self._cost(usage), int(usage.get("prompt_tokens") or 0)
            except (json.JSONDecodeError, ValueError):
                problem = f"returned invalid JSON: {text[:500]!r}"
        raise RuntimeError(f"DeepInfra {problem} ({attempts} attempts)")

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


def make_backend(judge_cfg: dict):
    name = judge_cfg.get("backend", "ollama")
    if name not in BACKENDS:
        raise ValueError(f"Unknown judge backend '{name}' (choose from {', '.join(BACKENDS)})")
    return BACKENDS[name](judge_cfg[name])
