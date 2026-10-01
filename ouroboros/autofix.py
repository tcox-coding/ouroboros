"""Auto-fix: find what looks wrong in a finished image and repair it.

The likeness loop compares an image with a reference. This looks at an image on its own,
the way a person would before keeping it: six fingers, an elbow bending the wrong way, a
waist pinched impossibly thin, a body twisted in a way no body can, a face that melts,
a strap that fuses into skin. Each round:

1. Inspect: a vision LLM lists the flaws it sees, each with the region it's in ("left
   hand", "waist"), how bad it is (1 barely visible .. 3 glaring), and how to fix it.
   It sees the prompt too, so it doesn't "fix" what was asked for (an unusual pose,
   a stylised build).
2. Fix the worst few, one after another, each starting from the previous result:
   - "inpaint": the region is found by CLIPSeg from its name and repainted on its own
     (the workflow's masked path), with prompt tags for what it should look like;
   - "hands": the hand refiner (DWPose fits a five-finger skeleton, the pose ControlNet
     repaints each hand to it), or a masked repaint when no hand skeleton is found;
   - "img2img": a light pass over the whole image, for problems that aren't in one place.
3. Review: the LLM sees the image before and after and says whether it got better. A
   worse result is thrown away, and the next round is told what didn't work. Its list of
   what's still wrong becomes the next round's to-do list, so each round after the first
   costs one LLM call.

Nothing replaces the original: the result is saved next to it (<name>_fixed.png) and
every step is logged in the run folder's autofix.json.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import replace
from pathlib import Path

from PIL import Image

from . import masks, pose
from .comfy import Cancelled
from .params import GenParams, edit_prompt

KINDS = ["hands", "anatomy", "proportions", "pose", "face", "clothing", "object", "artifact", "background"]
METHODS = ["inpaint", "hands", "img2img"]

INSPECT_INSTRUCTIONS = """You inspect a finished Stable Diffusion XL image for flaws, the way an artist checks
a drawing before keeping it. Look at the image on its own: there is no reference to match.

Report what looks WRONG or UNINTENDED:
- hands: count the fingers on every visible hand (five, including the thumb); fused,
  missing, extra or bent-backwards fingers; a hand that is a blob; two left hands;
- anatomy: extra or missing limbs, joints bending the wrong way, limbs that don't connect
  to the body, a body twisted or contorted in a way a real body can't be;
- proportions: waist impossibly thin, neck too long, head too small or large for the body,
  limbs of different lengths, breasts or hips distorted beyond the drawing style;
- face: melted or asymmetric features, mismatched or misaligned eyes, extra teeth;
- clothing and objects: straps or garments fusing into skin, objects melting into each
  other, half-formed props, duplicated items;
- artifacts: garbled text, smears, broken lines, doubled outlines, background bleeding in.

Do NOT report:
- anything the PROMPT asks for (an intentionally unusual pose, a stylised build or drawing
  style, a requested object or expression);
- style choices, composition, or anything you would need to zoom in to see.

For each flaw give: what is wrong (specific: "left hand has six fingers"), the region it
is in as a short name a segmentation model can find ("left hand", "waist", "face",
"right leg", "belt"; say left/right as seen by the viewer), its kind, severity (1 = barely
noticeable, 2 = noticeable, 3 = glaring), and how to fix it: method "hands" for hands,
"inpaint" for anything in one region, "img2img" only for problems spread over the whole
image. Also give prompt tags that describe how that region SHOULD look ("add") and tags
for what to avoid ("negative_add"). List at most 6 flaws, worst first. An empty list is
the right answer for a clean image.

Reply with JSON only."""

REVIEW_INSTRUCTIONS = """You check whether a repair of a Stable Diffusion XL image worked. You get the image
BEFORE and AFTER the repair, the flaws the repair targeted, and the prompt.

1. For each targeted flaw: is it fixed in AFTER?
2. Did the repair break anything (a new flaw, a changed face or outfit, a region that no
   longer matches the rest)?
3. Is AFTER better than BEFORE overall? Only say yes if it is clearly better or the
   targeted flaws are fixed without new damage.
4. List the flaws still visible in AFTER, in the same form as an inspection (at most 6,
   worst first; don't report what the prompt asks for; empty if clean).

Reply with JSON only."""

_ISSUE = {
    "type": "object",
    "properties": {
        "what": {"type": "string"},
        "region": {"type": "string"},
        "kind": {"type": "string", "enum": KINDS},
        "severity": {"type": "integer", "minimum": 1, "maximum": 3},
        "method": {"type": "string", "enum": METHODS},
        "add": {"type": "array", "items": {"type": "string"}},
        "negative_add": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["what", "region", "kind", "severity", "method", "add", "negative_add"],
    "additionalProperties": False,
}
INSPECT_SCHEMA = {
    "type": "object",
    "properties": {"issues": {"type": "array", "maxItems": 6, "items": _ISSUE}, "notes": {"type": "string"}},
    "required": ["issues", "notes"],
    "additionalProperties": False,
}
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "fixed": {"type": "array", "items": {"type": "boolean"}},
        "new_problems": {"type": "string"},
        "better": {"type": "boolean"},
        "issues": {"type": "array", "maxItems": 6, "items": _ISSUE},
        "notes": {"type": "string"},
    },
    "required": ["fixed", "new_problems", "better", "issues", "notes"],
    "additionalProperties": False,
}

DEFAULTS = {"max_rounds": 2, "fixes_per_round": 2, "min_severity": 2, "denoise": 0.6,
            "img2img_denoise": 0.4, "image_max_side": 1024, "keep_margin": 2}


def settings(cfg: dict) -> dict:
    """autofix.* from the config over the defaults (a field left empty in Settings is null)."""
    return {**DEFAULTS, **{k: v for k, v in (cfg.get("autofix") or {}).items() if v is not None and v != ""}}


def _prompt_part(positive: str) -> dict:
    return {"text": "PROMPT the image was made from (what it is meant to show):\n" + (positive.strip() or "(unknown)")}


def inspect(backend, image: Path, positive: str, max_side: int, tried: list[str] = ()) -> tuple[list[dict], float]:
    parts = [_prompt_part(positive)]
    if tried:
        parts.append({"text": "ALREADY TRIED without success (don't propose the same fix again; try another "
                              "method or region, or leave it):\n" + "\n".join(f"- {t}" for t in tried)})
    parts += [{"text": "IMAGE to inspect:"}, {"image": image}]
    data, cost, _ = backend.complete(INSPECT_INSTRUCTIONS, parts, INSPECT_SCHEMA, "autofix_inspect", max_side)
    return _clean(data.get("issues")), cost


def review(backend, before: Path, after: Path, targeted: list[dict], positive: str,
           max_side: int) -> tuple[dict, float]:
    parts = [_prompt_part(positive),
             {"text": "FLAWS the repair targeted:\n" + "\n".join(f"{i + 1}. {t['what']} ({t['region']})"
                                                               for i, t in enumerate(targeted))},
             {"text": "BEFORE:"}, {"image": before}, {"text": "AFTER:"}, {"image": after}]
    data, cost, _ = backend.complete(REVIEW_INSTRUCTIONS, parts, REVIEW_SCHEMA, "autofix_review", max_side)
    data["issues"] = _clean(data.get("issues"))
    return data, cost


def _clean(issues) -> list[dict]:
    out = []
    for i in issues or []:
        if not (i.get("what") or "").strip():
            continue
        out.append({**i, "severity": max(1, min(3, int(i.get("severity") or 1))),
                    "region": (i.get("region") or "").strip() or "body",
                    "method": i.get("method") if i.get("method") in METHODS else "inpaint"})
    return sorted(out, key=lambda i: -i["severity"])


def weight(issues: list[dict]) -> int:
    """How much is wrong, all told: the sum of the severities."""
    return sum(i["severity"] for i in issues)


def _region(issue: dict) -> str:
    return re.sub(r"\(.*?\)|[^a-z ]", "", issue["region"].lower()).strip()


def carry_forward(after: list[dict], before: list[dict], targeted: list[dict]) -> list[dict]:
    """The reviewer's list of what's still wrong, plus anything from before that this
    round didn't touch and the list forgot. An untouched flaw can't have gone away, but
    a model listing what it sees will drop some each time it looks."""
    seen = {_region(i) for i in after}
    touched = {id(i) for i in targeted}
    kept = [i for i in before if id(i) not in touched
            and not any(r and (r in _region(i) or _region(i) in r) for r in seen)]
    return sorted(after + kept, key=lambda i: -i["severity"])


def _next_try(issue: dict, tried: list[str]) -> dict | None:
    """The issue as it should be attempted next, or None when every way was tried: the
    hand refiner first for hands, then a masked repaint that follows the hand's shape."""
    if _attempt_key(issue) not in tried:
        return issue
    if issue["method"] == "hands":
        alt = {**issue, "method": "inpaint"}
        return alt if _attempt_key(alt) not in tried else None
    return None


def _attempt_key(issue: dict) -> str:
    """A repair already tried: its method and region. Not the wording of the flaw, which
    the reviewer rephrases every round ("six fingers", "an extra finger")."""
    return f"{issue['method']} on '{_region(issue)}'"


def autofix(image: Path, params: GenParams, *, backend, comfy, flows, cfg: dict, checkpoint: str | None,
            positive: str, out_dir: Path, upload, log=print, tag: str = "fix",
            stage=lambda text: None, should_stop=lambda: False) -> dict:
    """Inspect `image` and repair what's wrong, for up to autofix.max_rounds rounds.

    params: the settings the image was made with (sampler, steps, cfg, LoRAs, negative);
    positive: the prompt as rendered. Returns {"image": the fixed image's path or None if
    nothing was kept, "rounds": [...], "issues_found": [...], "issues_left": [...],
    "cost_usd": ...}. Work files go to out_dir/<tag>_*.png."""
    s = settings(cfg)
    side = int(s["image_max_side"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cost = 0.0
    tried: list[str] = []

    def check():
        if should_stop():
            raise Cancelled()

    check()  # stopped already: don't pay for a look
    stage("inspecting the image")
    issues, c = inspect(backend, image, positive, side)
    cost += c
    found = list(issues)
    log(f"auto-fix: {len(issues)} issue(s) found" + (": " + "; ".join(i["what"] for i in issues) if issues else ""))
    current, rounds = image, []
    for rnd in range(1, int(s["max_rounds"]) + 1):
        check()
        todo = [t for t in (_next_try(i, tried) for i in issues if i["severity"] >= int(s["min_severity"])) if t]
        todo = todo[:int(s["fixes_per_round"])]
        if not todo:
            break
        candidate, applied = current, []
        hands_done = False
        for n, issue in enumerate(todo, 1):
            check()
            if issue["method"] == "hands" and hands_done:
                applied.append(issue)  # the hand refiner already repainted every hand this round
                continue
            stage(f"round {rnd}: fixing {issue['what'][:60]}")
            try:
                fixed = _apply(issue, candidate, params, s, comfy=comfy, flows=flows, cfg=cfg, checkpoint=checkpoint,
                               positive=positive, out_dir=out_dir, upload=upload, stem=f"{tag}_r{rnd}_{n}",
                               log=log, should_stop=should_stop)
            except masks.EmptyMask as e:
                log(f"auto-fix: {e}; skipping that one")
                tried.append(_attempt_key(issue))
                continue
            if fixed:
                candidate = fixed
                applied.append(issue)
                hands_done = hands_done or issue["method"] == "hands"
        if not applied:
            break
        check()
        stage(f"round {rnd}: checking the result")
        verdict, c = review(backend, current, candidate, applied, positive, side)
        cost += c
        # Kept only if it is better by the reviewer's word AND by the numbers: what's still
        # wrong must weigh less than before. A model asked "better?" says yes too easily,
        # even while describing the new damage the repair did.
        after = carry_forward(verdict["issues"], issues, [i for i in issues if any(
            _region(i) == _region(a) and i["what"] == a["what"] for a in applied)])
        kept = bool(verdict.get("better")) and weight(after) < weight(issues)
        rounds.append({"round": rnd, "targeted": applied, "image": candidate.name, "before": current.name,
                       "kept": kept, "fixed": verdict.get("fixed"), "new_problems": verdict.get("new_problems", ""),
                       "weight_before": weight(issues), "weight_after": weight(after),
                       "notes": verdict.get("notes", "")})
        log(f"auto-fix round {rnd}: " + ("kept" if kept else "discarded") + f" ({verdict.get('notes', '')[:160]})")
        if kept:
            current, issues = candidate, after
        else:
            tried += [_attempt_key(i) for i in applied]
    return {"image": current if current != image else None, "rounds": rounds, "issues_found": found,
            "issues_left": issues, "cost_usd": round(cost, 5)}


def _apply(issue: dict, image: Path, params: GenParams, s: dict, *, comfy, flows, cfg, checkpoint, positive,
           out_dir: Path, upload, stem: str, log, should_stop) -> Path | None:
    """One repair; returns the new image (or None if nothing came back)."""
    size = Image.open(image).size
    pos = edit_prompt(positive, list(issue.get("add") or []), [])
    neg = edit_prompt(params.negative, list(issue.get("negative_add") or []), [])
    denoise = float(s["denoise"]) + (0.1 if issue["severity"] >= 3 else 0.0)
    if issue["method"] == "hands" and pose.available():
        from .handfix import refine_hands
        # Same strength rule as the other repairs: too gentle, and the old extra finger
        # shows through the new hand as a ghost.
        hcfg = {**cfg, "hands": {**(cfg.get("hands") or {}), "denoise": min(0.85, denoise)}}
        res = refine_hands(image, replace(params, negative=neg), comfy=comfy, flows=flows, cfg=hcfg,
                           checkpoint=checkpoint, positive=pos, out_dir=out_dir, upload=upload, log=log, tag=stem)
        if res.get("image"):
            return res["image"]
        log("auto-fix: no hand skeleton found; repainting the region instead")
    if issue["method"] == "img2img":
        p = replace(params, mode="img2img_best", mask_target=None, negative=neg,
                    denoise=float(s["img2img_denoise"]), seed=random.randrange(2**32))
        name = upload(image)
    else:
        target = issue["region"] or "hands"
        masked = masks.masked_image(comfy, image, upload(image), target, out_dir / f"{stem}_mask.png")
        p = replace(params, mode="inpaint_best", mask_target=target, negative=neg,
                    denoise=min(0.85, denoise), seed=random.randrange(2**32))
        name = upload(masked)
    graph = flows.build(p, name, 1, checkpoint, pos, size)
    (out_dir / f"{stem}_graph.json").write_text(json.dumps(graph), encoding="utf-8")
    pid = comfy.queue(graph)
    data = comfy.fetch_images(comfy.wait([pid], should_stop=should_stop)[pid], flows.output_node)
    if not data:
        return None
    out = out_dir / f"{stem}.png"
    out.write_bytes(data[0])
    return out


# ---- results kept per run folder ------------------------------------------------------------

def results_file(run_dir: Path) -> Path:
    return run_dir / "autofix.json"


def load_results(run_dir: Path) -> dict:
    """{source image name: {"state", "image" (fixed file name or None), "rounds", ...}}"""
    try:
        return json.loads(results_file(run_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_result(run_dir: Path, source: str, entry: dict) -> None:
    data = load_results(run_dir)
    data[source] = {**data.get(source, {}), **entry}
    tmp = results_file(run_dir).with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(results_file(run_dir))


def run_and_record(source: Path, params: GenParams, *, run_dir: Path, backend, comfy, flows, cfg, checkpoint,
                   positive: str, upload, log=print, stage=lambda t: None, should_stop=lambda: False) -> dict:
    """autofix() on one image of a run, keeping the result as <stem>_fixed.png and the
    record in the run's autofix.json (what History shows)."""
    save_result(run_dir, source.name, {"state": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                                       "error": None, "image": None})
    try:
        res = autofix(source, params, backend=backend, comfy=comfy, flows=flows, cfg=cfg, checkpoint=checkpoint,
                      positive=positive, out_dir=run_dir / "autofix", upload=upload, log=log,
                      tag=source.stem, stage=stage, should_stop=should_stop)
    except Cancelled:
        save_result(run_dir, source.name, {"state": "cancelled", "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
        raise
    except Exception as e:
        save_result(run_dir, source.name, {"state": "error", "error": f"{type(e).__name__}: {e}"[:500],
                                           "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
        raise
    fixed = None
    if res["image"]:
        fixed = run_dir / f"{source.stem}_fixed.png"
        fixed.write_bytes(Path(res["image"]).read_bytes())
    save_result(run_dir, source.name, {
        "state": "done", "image": fixed.name if fixed else None, "rounds": res["rounds"],
        "issues_found": res["issues_found"], "issues_left": res["issues_left"], "cost_usd": res["cost_usd"],
        "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    return {**res, "fixed": fixed}
