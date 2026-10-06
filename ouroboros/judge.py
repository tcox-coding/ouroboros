"""One vision-LLM call per round: score every candidate against the reference with a
fixed rubric AND propose the next edit. Structured output means no parsing
failures or retries.

Also: a fresh-eyes confirmation of a passing image (no history, so the judge can't
talk itself into "that issue is fixed"), and a summary call that compresses older
rounds so the prompt stays short (see loop.JudgeMemory).

Each candidate's answer lists its differences from the reference BEFORE its scores.
Tested on qwen3-vl:8b-instruct: without that, a single image shown alone got a perfect
100 ("exact match") three times out of three; with it, 80-82 and the real differences.
Showing the confirmed image among other candidates instead made scores erratic.

That list is capped at 8, because its length is what makes a score wobble: a model that
enumerates every difference it can find decides afresh each time whether the background
shade is worth a line, and the score follows. Judging 12 fixed images five times each
with DeepSeek V4.1 Flash, the cap took the spread between looks of the SAME image from
9.5 points to 5.0 (worst case 16.5 -> 11.8) with pair accuracy and flaw recall both
still 100%, and made each call cheaper for having less to write. Scoring by an explicit
deduction rule (-1 a major difference, -1 per two minor ones) was also tried: it spread
the good and flawed bands further apart but left the wobble at 8.2, because an uncapped
list feeds the rule.

The model behind it is a backend (see backends.py): a hosted model on DeepInfra or
OpenAI, or Ollama on the LAN.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .backends import make_backend, CompletionError
from .params import MODES, PHASES
from .sizes import to_rgb
from .targets import LABELS, Targets, judge_parts, target_images

INSTRUCTIONS = """You steer a local Stable Diffusion XL (ComfyUI) pipeline toward a reference image.

Each round you get the reference image, the parameters of the current round, notes on
earlier rounds, and several candidate images rendered this round.

1. For EVERY candidate, first list each visible difference from the reference (hair,
   face and chin shape, expression, head direction and eye gaze direction (separately:
   a turned head can still look the other way), each garment and accessory, colours,
   pose, hands, proportions, background). List at most 8 differences, the most visible first, and ignore anything you
   would have to zoom in to see; an empty list means it is indistinguishable. Then score it on each rubric criterion,
   0-10, using the anchors given; each difference should cost points somewhere.
   Judge only what you can see in the images now; notes on earlier rounds may be wrong.
   Be consistent: the same image must get the same score.
2. Pick the best candidate.
3. Diagnose the most important remaining difference(s) from the reference.
4. Propose ONE edit, applied to the best candidate's parameters, most likely to fix them.

Editing guidance:
- Work coarse to fine. phase "explore": composition, pose, character identity (prompt
  tags, mode, new seeds). "refine": lock the seed (keep_seed=true), tune cfg/denoise/
  sampler. "repair": fix one local detail.
- LOCAL problems (an extra or missing emblem, strap, accessory, a bad hand, wrong eye
  colour) are fixed by masked repainting: set mask_target to a short name of that ONE
  region (e.g. "left sleeve", "belt", "right hand") and mode inpaint_best, usually with
  prompt/negative edits for it. A segmentation model builds the mask from that text and
  only that region changes. img2img at low denoise will NOT remove or add details.
  Say "left"/"right" as seen by the viewer (image left/right).
- img2img_reference starts from the reference image (higher denoise = freer; see RULES
  for its limit). img2img_best refines the current best image (denoise ~0.3-0.5); use it
  for global issues like colour or shading, not for local details.
- ONE AXIS PER EDIT: set "focus" to what the edit changes, and change only that, so the
  next round shows what it did:
  "prompt": prompt/negative tags (optionally with a mode/denoise change or a masked
  repaint); loras = null. "lora_weights": only the strengths of the LoRAs in use; no
  prompt edits. "loras": a different set of LoRAs (swap one, add one, or an empty list
  for none); no prompt edits. "settings": mode, cfg, denoise, sampler, mask only.
  Use LoRAs for the LOOK (drawing style, line work, shading, colour treatment, rendering)
  and the prompt for CONTENT (character, outfit, hair, pose, background). If the last edit
  on one axis didn't help, try the other.
- LoRAs (when a LORAS section is given): each comes with a brief of what it does, what it
  suits, what it's weak at, and how its effect builds with strength. Early in a job (the
  section says whether switching is still open) a wrong look is best fixed by switching:
  try the LoRA whose brief best matches the reference's style, or drop one that fights
  it. No LoRA at all is a valid choice when none suits the image. Later, tune strengths
  (about 0.1-0.3 at a time): lower when its look is overbaked or overrides the
  character, higher when its look doesn't show.
- Prefer small, targeted prompt edits. prompt_remove takes exact existing tags from the
  prompt; to get rid of something the image shows but the prompt doesn't ask for, put
  it in negative_add. Don't re-add tags the prompt already has.
- Weights: keep them at or below 1.5. If the notes show a stronger weight didn't fix
  something, don't raise it again: change the approach (a LoRA strength, the mode, a
  masked repaint of that region, or a negative tag for what shows instead).
- Use null for any setting you don't want to change. Don't repeat an edit the notes
  show didn't help.

Reply with JSON only, matching the given schema.
"""

CONFIRM_NOTE = ("This is an independent check of a candidate that passed an earlier review. Look closely "
                "and list every visible difference from the reference; score strictly by the anchors. "
                "If differences remain, they are usually local details (an added stripe, pocket, buckle): "
                "fix the most visible ONE with mode inpaint_best and a mask_target naming its region, plus "
                "prompt/negative edits for it. Low-denoise img2img will not remove it.")

SUMMARY_INSTRUCTIONS = """You keep the working notes for an image-refinement loop that steers Stable Diffusion
toward a reference image. Merge the previous summary and the new round notes into ONE
short summary for the next reviewer: what the best result so far is (its parameters and
score), which edits helped, which didn't (so they aren't repeated), and what still
differs from the reference. Facts only, no advice beyond what the notes support.
At most 150 words. Reply with JSON only."""

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}


@dataclass
class Review:
    scores: list[float]          # 0-100 per candidate, computed from the rubric
    best_index: int
    diagnosis: str
    edit: dict
    raw: dict
    cost_usd: float
    prompt_tokens: int = 0


def model_options(model: str, backend: str = "ollama") -> dict:
    """Sampling settings for a model from model_presets.json (so a second-opinion model
    gets its own temperature, context and token budget), for the backend in use."""
    from .model_profiles import preset
    p = preset(model, backend)
    return ((p or {}).get("settings") or {}).get("judge", {}).get(backend, {})


def _nullable(t: str) -> dict:
    return {"type": [t, "null"]}


def build_schema(criteria: list[str], samplers: list[str], schedulers: list[str], n_candidates: int,
                 modes: list[str] | None = None, lora_stems: list[str] | None = None, max_loras: int = 3,
                 lora_switch: bool = False) -> dict:
    score_obj = {
        "type": "object",
        "properties": {c: {"type": "integer"} for c in criteria},
        "required": criteria,
        "additionalProperties": False,
    }
    str_list = {"type": "array", "items": {"type": "string"}}
    edit = {
        "type": "object",
        "properties": {
            "phase": {"type": "string", "enum": PHASES},
            "focus": {"type": "string", "enum": ["prompt", "settings"] + (
                (["lora_weights", "loras"] if lora_switch else ["lora_weights"]) if lora_stems else [])},
            "mode": {"type": "string", "enum": modes or list(MODES)},
            "keep_seed": {"type": "boolean"},
            "prompt_add": str_list,
            "prompt_remove": str_list,
            "negative_add": str_list,
            "negative_remove": str_list,
            "steps": _nullable("integer"),
            "cfg": _nullable("number"),
            "denoise": _nullable("number"),
            "sampler_name": {"anyOf": [{"type": "string", "enum": samplers}, {"type": "null"}]},
            "scheduler": {"anyOf": [{"type": "string", "enum": schedulers}, {"type": "null"}]},
            "mask_target": _nullable("string"),
            "reason": {"type": "string"},
        },
        "additionalProperties": False,
    }
    if lora_stems:
        # null = keep the current LoRAs; a list = the new full set of style LoRAs.
        edit["properties"]["loras"] = {"anyOf": [{"type": "null"}, {
            "type": "array", "maxItems": max_loras,
            "items": {"type": "object",
                      "properties": {"lora": {"type": "string", "enum": lora_stems},
                                     "strength": {"type": "number"}},
                      "required": ["lora", "strength"], "additionalProperties": False}}]}
    edit["required"] = list(edit["properties"])
    return {
        "type": "object",
        "properties": {
            # Exactly one entry per candidate: small local models otherwise tend to
            # score the reference as an extra candidate and shift every index.
            "candidates": {
                "type": "array",
                "minItems": n_candidates,
                "maxItems": n_candidates,
                "items": {
                    "type": "object",
                    # "differences" comes before "scores" so the model has to look
                    # and list what differs before it writes numbers.
                    "properties": {"index": {"type": "integer", "enum": list(range(n_candidates))},
                                   "differences": {"type": "array", "items": {"type": "string"}},
                                   "scores": score_obj, "notes": {"type": "string"}},
                    "required": ["index", "differences", "scores", "notes"],
                    "additionalProperties": False,
                },
            },
            "best_index": {"type": "integer", "enum": list(range(n_candidates))},
            "diagnosis": {"type": "string"},
            "edit": edit,
        },
        "required": ["candidates", "best_index", "diagnosis", "edit"],
        "additionalProperties": False,
    }


def contact_sheet(reference: Path, candidates: list[Path], size: int,
                  labels: list[str] | None = None) -> Image.Image:
    """Reference + candidates as one labelled grid, at most size x size. For vision
    models that accept a single image per message (e.g. llama3.2-vision). Tiles keep
    the images' aspect ratio, and the column count that gives the largest tiles is
    used (tall portraits usually end up side by side in one row)."""
    labels = labels or ["REFERENCE"] + [f"CANDIDATE {i}" for i in range(len(candidates))]
    tiles = list(zip(labels, [reference] + list(candidates)))
    images = [to_rgb(Image.open(p)) for _, p in tiles]
    w, h = images[0].size
    label_h, gap = 30, 6

    def tile_height(cols: int) -> float:
        rows = -(-len(tiles) // cols)
        by_width = (size - gap * (cols - 1)) / cols * h / w
        by_height = (size - rows * label_h - gap * (rows - 1)) / rows
        return min(by_width, by_height)

    cols = max(range(1, len(tiles) + 1), key=tile_height)
    rows = -(-len(tiles) // cols)
    th = int(tile_height(cols))
    tw = int(th * w / h)
    sheet = Image.new("RGB", (cols * tw + gap * (cols - 1), rows * (th + label_h) + gap * (rows - 1)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arialbd.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    for n, ((label, _), img) in enumerate(zip(tiles, images)):
        x, y = (n % cols) * (tw + gap), (n // cols) * (th + label_h + gap)
        draw.text((x + 4, y + 5), label, fill="black", font=font)
        img.thumbnail((tw, th))
        sheet.paste(img, (x + (tw - img.width) // 2, y + label_h))
    return sheet


class Judge:
    def __init__(self, cfg: dict, samplers: list[str], schedulers: list[str]):
        self.cfg = cfg
        self.backend = make_backend(cfg)
        self.prompt_backend = make_backend(cfg, "prompt") if cfg.get("prompt_model") else self.backend
        # Optional second opinion for the pass check only (judge.confirm_model): a
        # careful, slower model decides when a job may stop, while the fast one ranks
        # candidates every round. In testing, a 30B thinking model ranked every labelled
        # pair right but took 110 s a call against 14 s (python -m ouroboros eval).
        self.confirm_backend = None
        bkey = cfg.get("backend", "ollama")
        name = (cfg.get("confirm_model") or "").strip()
        if name and (name != cfg.get(bkey, {}).get("model") or cfg.get("confirm_options")):
            self.confirm_backend = make_backend(cfg, "confirm")
        self.base_rubric: dict[str, dict] = cfg["rubric"]  # name -> {"weight", "anchors"}
        self.optional_rubric: dict[str, dict] = cfg.get("rubric_optional", {})
        self.samplers, self.schedulers = samplers, schedulers

    @property
    def context_window(self) -> int:
        return self.backend.context_window

    def rubric(self, penalize_extras: bool) -> dict[str, dict]:
        """The base rubric, plus the optional "extras" criterion (items the reference
        doesn't have) when this job penalizes additions."""
        r = dict(self.base_rubric)
        if penalize_extras and "extras" in self.optional_rubric:
            r["extras"] = self.optional_rubric["extras"]
        return r

    @staticmethod
    def score(crit_scores: dict, rubric: dict[str, dict]) -> float:
        total_w = sum(r["weight"] for r in rubric.values())
        def num(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0
        s = sum(max(0, min(10, num(crit_scores.get(n, 0)))) * r["weight"] for n, r in rubric.items())
        return round(10 * s / total_w, 1)

    def review(self, reference: Path, job_goal: str, current: str, notes: str,
               candidates: list[Path], modes: list[str] | None = None, rules: str = "",
               rubric: dict[str, dict] | None = None, extra_note: str = "",
               loras: dict | None = None, pose: dict | None = None, backend=None,
               targets: Targets | None = None) -> Review:
        """loras: {"menu": text listing the job's LoRA shortlist, "stems": [names the
        judge may use], "max": int}, or None when LoRAs aren't being tuned.
        pose: {"image": skeleton path, "description": str} when the pose comes from a
        separate pose image (the pose library), not from the reference.
        targets: separate style/subject/pose targets (targets.py). When they say more than
        the one reference would (targets.split), each criterion is judged against its own
        target instead of everything against `reference`."""
        split = targets is not None and (targets.split or bool(pose))
        rubric = rubric or self.base_rubric
        side = self.cfg.get("image_max_side", 512)
        sheet = self.cfg.get("contact_sheet_size", 1120)  # llama3.2-vision reads up to 1120x1120
        rubric_text = "\n".join(f"- {name} (weight {r['weight']}): {r['anchors']}" for name, r in rubric.items())
        # The stable, most important content goes first (rubric, goal, rules, reference):
        # models attend best to the start of the context, and it's what OpenAI caches.
        parts: list[dict] = [
            {"text": f"RUBRIC\n{rubric_text}\n\nGOAL (the prompt being refined)\n{job_goal}"
                     + (f"\n\nRULES\n{rules}" if rules else "")
                     + (f"\n\nLORAS (at most {loras['max']} at once; "
                        + ("switching is OPEN this round: any of these, or none" if loras.get("switch", True)
                           else "switching is CLOSED: keep the LoRAs in use and tune their strengths, or edit the prompt")
                        + f")\n{loras['menu']}" if loras else "")},
        ]
        round_text = (f"NOTES ON EARLIER ROUNDS\n{notes or 'none'}"
                      + f"\n\nPARAMETERS THE EDIT APPLIES TO\n{current}"
                      + (f"\n\n{extra_note}" if extra_note else ""))
        if self.cfg.get("contact_sheet") and split:
            seen, tiles = set(), []
            for role, image, _text in target_images(targets, reference, pose):
                if image is not None and str(Path(image).resolve()) not in seen:
                    seen.add(str(Path(image).resolve()))
                    tiles.append((LABELS[role], image))
            texts = [p["text"] for p in judge_parts(targets, reference, pose) if "text" in p]
            labels = [lbl for lbl, _ in tiles] + [f"CANDIDATE {i}" for i in range(len(candidates))]
            parts += [{"text": "\n".join(texts)}, {"text": round_text},
                      {"text": f"The image is a grid: the targets first ({', '.join(l for l, _ in tiles)}), then "
                               f"CANDIDATE 0 to {len(candidates) - 1}, each labelled above its tile."},
                      {"image": contact_sheet(tiles[0][1], [i for _, i in tiles[1:]] + list(candidates), sheet, labels)
                       if tiles else contact_sheet(candidates[0], candidates[1:], sheet, labels),
                       "max_side": sheet}]
        elif self.cfg.get("contact_sheet"):
            parts += [
                {"text": round_text},
                {"text": f"The image is a grid: REFERENCE first, then CANDIDATE 0 to {len(candidates) - 1}, "
                         "each labelled above its tile."},
                {"image": contact_sheet(reference, candidates, sheet), "max_side": sheet},
            ]
        elif split:
            n = len(candidates)
            parts += judge_parts(targets, reference, pose)
            parts += [{"text": round_text},
                      {"text": f"Now {n} candidate(s), numbered 0 to {n - 1}:"}]
            for i, path in enumerate(candidates):
                parts += [{"text": f"CANDIDATE {i}:"}, {"image": path}]
        else:
            n = len(candidates)
            parts += [
                {"text": f"There are {n + 1} images: the REFERENCE, then {n} candidate(s) numbered 0 to {n - 1}. "
                         "The reference is NOT a candidate; never score it."},
                {"text": "REFERENCE IMAGE (target, not a candidate):"}, {"image": reference},
            ]
            if pose:
                parts += [{"text": "POSE (a skeleton, not a candidate): the pose and framing every candidate should "
                                   "have instead of the reference's. Judge composition against this pose"
                                   + (f" ({pose['description']})" if pose.get("description") else "")
                                   + "; judge character, outfit, style and colour against the reference."},
                          {"image": pose["image"]}]
            parts += [{"text": round_text}]
            for i, path in enumerate(candidates):
                parts += [{"text": f"CANDIDATE {i}:"}, {"image": path}]
        # Models also attend well to the very end of the prompt: restate the task there.
        parts.append({"text": "TASK: for each candidate list its differences from "
                              + ("its TARGETS (each criterion against its own target)" if split else "the REFERENCE")
                              + ", then score it against the rubric anchors; pick the best; diagnose; propose ONE "
                                "edit. JSON only."})

        schema = build_schema(list(rubric), self.samplers, self.schedulers, len(candidates), modes,
                              loras["stems"] if loras else None, loras["max"] if loras else 3,
                              loras.get("switch", True) if loras else False)
        # A model that can't take the schema answers in plain JSON mode, where nothing
        # enforces the shape (ByteDance Seed once sent candidates as strings): drop
        # malformed entries, and ask once more if no candidate was scored at all.
        cost = 0.0
        for attempt in range(2):
            try:
                data, c, tokens = (backend or self.backend).complete(INSTRUCTIONS, parts, schema, "round_review", side)
            except Exception as e:
                raise CompletionError(str(e), cost + float(getattr(e, "cost_usd", 0.0))) from e
            cost += c or 0.0
            data = data if isinstance(data, dict) else {}
            data["candidates"] = [c for c in data.get("candidates") or []
                                  if isinstance(c, dict) and isinstance(c.get("index"), int)
                                  and 0 <= c["index"] < len(candidates) and isinstance(c.get("scores"), dict)]
            if not isinstance(data.get("edit"), dict):
                data["edit"] = {}
            if data["candidates"]:
                break
        if not data["candidates"]:
            raise CompletionError("the judge's answer scored no candidate (twice); try another model", cost)

        scores = [0.0] * len(candidates)
        for c in data["candidates"]:
            scores[c["index"]] = self.score(c["scores"], rubric)
        best = max(range(len(scores)), key=scores.__getitem__)  # trust the numbers, not best_index
        return Review(scores, best, data.get("diagnosis", ""), data.get("edit", {}), data, cost, tokens)

    def review_each(self, reference: Path, job_goal: str, current: str | list[str], notes: str,
                    candidates: list[Path], modes: list[str] | None = None, rules: str = "",
                    rubric: dict[str, dict] | None = None, extra_note: str = "",
                    loras: dict | None = None, pose: dict | None = None, targets: Targets | None = None) -> Review:
        """Score each candidate in its own call (reference + that one image) and combine.

        A position test (the same 3 images in all 6 orders, qwen3-vl 8B) showed that
        scoring several candidates in one call makes the model copy scores between
        neighbours (10 of 12 calls had ties; a 92-point image and a 71-point one averaged
        64.9 vs 64.4), favour the first image on ties, and dock the last ~4 points. One
        image per call avoids all three. The best is the highest score; ties go to fewer
        listed differences. The next edit comes from the best candidate's own call.

        current: the base parameters, or one line per candidate. Candidates in a round
        differ (seed, cfg, denoise, LoRA set), and each edit is applied to its own
        candidate's parameters, so each call is told that candidate's."""
        from .llm_queue import current_label, set_label
        rubric = rubric or self.base_rubric
        currents = current if isinstance(current, list) else [current] * len(candidates)
        label = current_label()  # the job these calls are for, shown in the LLM queue

        def one(args):
            c, cur = args
            set_label(label)
            return self.review(reference, job_goal, cur, notes, [c], modes, rules, rubric, extra_note, loras, pose,
                               targets=targets)
        # Side by side: a big early round would otherwise wait for each call in turn. The
        # backend's gate (queue.llm_parallel) still decides how many really run at once.
        workers = max(1, min(len(candidates), int(getattr(self, "parallel", 1))))
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(workers) as pool:
                from contextvars import copy_context
                futures = [pool.submit(copy_context().run, one, pair) for pair in zip(candidates, currents)]
                reviews, failures = [], []
                for future in futures:
                    try:
                        reviews.append(future.result())
                    except Exception as e:
                        failures.append(e)
                if failures:
                    cost = sum(r.cost_usd for r in reviews) + sum(float(getattr(e, "cost_usd", 0.0)) for e in failures)
                    raise CompletionError(str(failures[0]), cost) from failures[0]
        else:
            reviews = []
            for pair in zip(candidates, currents):
                try:
                    reviews.append(one(pair))
                except Exception as e:
                    raise CompletionError(str(e), sum(r.cost_usd for r in reviews) + float(getattr(e, "cost_usd", 0))) from e
        scores = [r.scores[0] for r in reviews]
        diffs = [len((r.raw.get("candidates") or [{}])[0].get("differences") or []) for r in reviews]
        best = max(range(len(candidates)), key=lambda i: (scores[i], -diffs[i]))
        entries = []
        for i, r in enumerate(reviews):
            c = dict((r.raw.get("candidates") or [{}])[0])
            c["index"] = i
            entries.append(c)
        raw = {"candidates": entries, "best_index": best, "diagnosis": reviews[best].diagnosis,
               "edit": reviews[best].edit, "per_candidate": [{"diagnosis": r.diagnosis, "edit": r.edit}
                                                             for r in reviews]}
        return Review(scores, best, reviews[best].diagnosis, reviews[best].edit, raw,
                      sum(r.cost_usd for r in reviews), max(r.prompt_tokens for r in reviews))

    def confirm(self, reference: Path, job_goal: str, current: str, image: Path,
                modes: list[str] | None, rules: str, rubric: dict[str, dict],
                loras: dict | None = None, pose: dict | None = None, targets: Targets | None = None) -> Review:
        """Re-score one passing image in a fresh context: no notes from earlier rounds,
        so a claim like "the emblem is gone" can't carry over. Its edit is used to
        continue if the image doesn't pass again."""
        return self.review(reference, job_goal, current, "", [image], modes, rules, rubric, CONFIRM_NOTE, loras,
                           pose, self.confirm_backend, targets=targets)

    def summarize(self, previous: str, entries: list[str]) -> tuple[str, float]:
        parts = [{"text": f"PREVIOUS SUMMARY\n{previous or 'none'}\n\nNEW ROUND NOTES\n" + "\n".join(entries)}]
        data, cost, _ = self.backend.complete(SUMMARY_INSTRUCTIONS, parts, SUMMARY_SCHEMA, "summary", 0)
        return data.get("summary", "").strip(), cost
