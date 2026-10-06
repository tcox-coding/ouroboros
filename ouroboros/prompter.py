"""Write a job's starting prompt from a plain-language description.

Two cases:
  - description only: the LLM writes positive and negative prompts, in the tag style of
    the workflow's saved prompts (quality tags, style/LoRA trigger words), but none of
    their subject matter.
  - description + your positive/negative: the LLM merges the description into them.
    Your tags are kept unless the description contradicts them; anything the model
    drops without listing it in "dropped" is put back.

The reference image is shown too (loop.prompt_sees_reference), so the first round
already starts close to it: fewer rounds.
"""

from __future__ import annotations

import re
from pathlib import Path

from .params import drop_negative_conflicts, norm_tag, split_tags

INSTRUCTIONS = """You write prompts for a Stable Diffusion XL (Pony-family) model: comma-separated
tags and short phrases, one section per line with a blank line between sections, in this
order: quality and source tags; drawing style; character (build, skin, face and chin
shape, eyes, expression); hair; clothing; pose and framing; background. Describe only
what should be visible; no sentences, no explanations, no section headers.

Be detailed and specific, like a carefully tuned prompt (typically 40-80 tags): hair
(colour, length, cut, bangs, how it falls), face, eyes and expression, build, every
garment (colour, cut, fit, how it's worn), accessories, pose (arms, hands, head
direction, and eye gaze direction separately, as seen by the viewer), framing and camera
angle, background, rendering style. Put weights
(tag:1.2) to (tag:1.4) on the few tags that most define the character (hair, main
garment colours, pose). Where the example or user prompts are long and weighted, match
their length and weighting style.

Negative prompt: things to avoid (quality problems, wrong styles, and anything that
contradicts the description, e.g. other hair colours, extra people, props).

Reply with JSON only."""


SCHEMA = {
    "type": "object",
    "properties": {
        "positive": {"type": "string"},
        "negative": {"type": "string"},
        "dropped": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
    },
    "required": ["positive", "negative", "dropped", "notes"],
    "additionalProperties": False,
}


def _restore(original: str, result: str, dropped: list[str]) -> str:
    """Put back tags from the user's prompt that the model silently lost."""
    have = {norm_tag(t) for t in split_tags(result)}
    # Models often return "dropped" as one comma-joined string rather than a list.
    drop = {norm_tag(t) for d in dropped for t in split_tags(d)}
    missing = [t for t in split_tags(original) if norm_tag(t) not in have and norm_tag(t) not in drop]
    if not missing:
        return result.strip()
    return result.strip().rstrip(",") + ",\n" + ", ".join(missing)


NEGATIVE_ONLY = re.compile(r"^\(?(score_[1-6]|source_(anime|furry|pony)|worst quality|low quality|lowres|"
                           r"bad anatomy|bad hands|watermark|signature)(:[\d.]+\)?)?$", re.I)


def _problems(positive: str, negative: str) -> list[str]:
    """What's structurally wrong with a written prompt pair (it happened in testing: a
    model put the whole negative into the positive and left the negative empty, and the
    renders scored ~30 instead of ~45)."""
    out = []
    if not negative.strip():
        out.append("the negative prompt is empty; it must hold the things to avoid")
    leaked = [t for t in split_tags(positive) if NEGATIVE_ONLY.match(t.strip())]
    if leaked:
        out.append("the positive contains negative-prompt tags (" + ", ".join(leaked[:6]) + "); they belong "
                   "only in the negative")
    if not positive.strip():
        out.append("the positive prompt is empty")
    return out


def write_prompt(backend, description: str, positive: str = "", negative: str = "",
                 style_positive: str = "", style_negative: str = "",
                 reference: Path | None = None, max_side: int = 512, lora_notes: str = "",
                 pose_note: str = "", targets: list[dict] | None = None) -> dict:
    """Returns {"positive", "negative", "notes", "dropped"}.
    targets: targets.prompt_parts(): separate STYLE / SUBJECT / POSE texts and images, each
    saying what to take from it (then `reference` is usually None)."""
    from_image = not description.strip() and not positive.strip() and not targets
    parts: list[dict] = [{"text": f"DESCRIPTION\n{description.strip()}" if not from_image else
                          "DESCRIPTION\n(none: write the prompt from the REFERENCE image alone, describing "
                          "the character, outfit, pose, framing, background and drawing style you see)"}]
    if targets:
        parts.append({"text": "TARGETS: the render has a separate style, subject and pose. Take each part of the "
                              "prompt from its own target only, as each says:"})
        parts += targets
        if not description.strip():
            parts[0] = {"text": "DESCRIPTION\n(none: write the prompt from the TARGETS below)"}
    if lora_notes:
        parts.append({"text": "LORAS that will be active (put each one's \"trigger word\" in the positive prompt "
                              "exactly as written, and only those; write the prompt so it works with what each "
                              "LoRA does - its style, character, pose or concept - but describe the reference's "
                              "look, not the LoRA's; never write LoRA names or <lora:...> tags):\n" + lora_notes})
    if pose_note:
        parts.append({"text": "POSE for this job (a ControlNet imposes it; it replaces the reference's pose, so "
                              "describe this pose and framing, not the reference's):\n" + pose_note})
    if positive.strip():
        parts.append({"text": (
            "MERGE the description into the user's prompts below. Keep every user tag (quality, style, "
            "LoRA trigger words, layout) unless the description contradicts it; list removed tags in "
            "\"dropped\". Add the tags the description needs.\n\n"
            f"USER POSITIVE PROMPT\n{positive.strip()}\n\n"
            f"USER NEGATIVE PROMPT\n{negative.strip() or '(empty)'}")})
    else:
        # A user negative with no positive still needs the workflow's style examples for
        # the positive: merging alone left the score tags and trigger words out.
        parts.append({"text": (
            "Write new prompts. Match the TAG STYLE of these example prompts from the user's workflow: "
            "copy their quality and style tags (e.g. score tags, style/LoRA trigger words, rendering style) "
            "but NOT their subject, outfit or pose. From the example negative, copy only general quality and "
            "style negatives; it was written for a different subject, so leave out its clothing, pose and "
            "background tags, and never negate anything the description asks for.\n\n"
            f"EXAMPLE POSITIVE\n{style_positive.strip() or '(none)'}\n\n"
            f"EXAMPLE NEGATIVE\n{style_negative.strip() or '(none)'}")
            + (f"\n\nUSER NEGATIVE PROMPT (keep every tag of it in the negative; list any you remove in "
               f"\"dropped\")\n{negative.strip()}" if negative.strip() else "")})
    if reference is not None:
        parts += [{"text": "REFERENCE IMAGE the result should resemble" + (
                       ":" if from_image else " (use it for details the description leaves out; the "
                                              "description wins where they differ):")},
                  {"image": reference}]

    data, _cost, _tokens = backend.complete(INSTRUCTIONS, parts, SCHEMA, "prompt", max_side)
    problems = _problems(data.get("positive", ""), data.get("negative", ""))
    if problems:  # one more try, told what was wrong
        retry = parts + [{"text": "Your previous answer had these problems; fix them and answer again:\n- "
                                  + "\n- ".join(problems) + "\nPREVIOUS POSITIVE\n" + data.get("positive", "")[:1500]
                                  + "\nPREVIOUS NEGATIVE\n" + data.get("negative", "")[:800]}]
        again, _cost, _tokens = backend.complete(INSTRUCTIONS, retry, SCHEMA, "prompt", max_side)
        if len(_problems(again.get("positive", ""), again.get("negative", ""))) < len(problems):
            data = again
    dropped = [t for d in data.get("dropped") or [] for t in split_tags(d)]
    out_pos, out_neg = data.get("positive", "").strip(), data.get("negative", "").strip()
    leaked = [t for t in split_tags(out_pos) if NEGATIVE_ONLY.match(t.strip())]
    if leaked:  # still there after the retry: move them where they belong
        gone = {norm_tag(t) for t in leaked}
        out_pos = "\n".join(", ".join(t.strip() for t in line.split(",") if t.strip() and norm_tag(t) not in gone)
                             for line in out_pos.splitlines()).strip()
        out_neg = ", ".join(leaked) + ("\n" + out_neg if out_neg else "")
    if positive.strip():
        out_pos = _restore(positive, out_pos, dropped)
    if negative.strip():
        out_neg = _restore(negative, out_neg, dropped)
    out_neg, conflicts = drop_negative_conflicts(out_pos, out_neg)
    notes = data.get("notes", "")
    if conflicts:
        notes = (notes + " " if notes else "") + "Removed from the negative because the positive asks for them: " \
                + ", ".join(conflicts) + "."
    return {"positive": out_pos, "negative": out_neg, "dropped": dropped, "notes": notes}


# ---- does a description ask for an art style? ----------------------------------------------
# An automatic run with a character image and a description but no style target: a style the
# description names ("in Incase style, flat colours") is what the run aims for, and with none
# named the run keeps the character image's own look (loop.run_job).
STYLE_CHECK = """Does the DESCRIPTION ask for a particular art style? That is a named style, artist,
game or show whose look it means; a medium (watercolour, oil painting, 3D render, pixel art,
photo); or a way of drawing (flat colours, cel shading, thick outlines, sketchy lines, soft
gradients). Words about the character, outfit, pose, expression, setting or mood are not an
art style. JSON only."""

STYLE_CHECK_SCHEMA = {
    "type": "object",
    "properties": {"names_style": {"type": "boolean"}, "style": {"type": "string"}},
    "required": ["names_style", "style"],
    "additionalProperties": False,
}


def names_style(backend, description: str) -> tuple[bool, str, float]:
    """(whether the description asks for an art style, the words that do, cost)."""
    data, cost, _ = backend.complete(STYLE_CHECK, [{"text": f"DESCRIPTION\n{description.strip()}"}],
                                     STYLE_CHECK_SCHEMA, "style_check", 0)
    data = data if isinstance(data, dict) else {}
    named = data.get("names_style")
    if isinstance(named, str):  # plain-JSON fallbacks sometimes send "true"/"false"
        named = named.strip().lower() == "true"
    return bool(named), str(data.get("style") or "").strip(), cost or 0.0
