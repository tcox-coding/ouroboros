"""Designer: one character, changed one thing at a time, everything else kept.

A design is a character image (usually a Home render, with the settings that made it) and a
catalog of versions of it the person chose to keep. An edit changes exactly one of:

  style     the drawing style; character, outfit, pose and framing stay
  pose      the pose; character, outfit, style and background stay
  features  what the person asks for in words ("shorter hair", "no necklace"); the rest stays

and is optimised in rounds, toward the base image everywhere except the change:

  1. plan (one LLM call): the base image's prompt edited only where the change is, what must
     change, and a short list of the base image's details that must not;
  2. each round renders a few candidates with the settings (the "knobs") for this kind of edit:
       style:    img2img from the base image + its edges (Canny) and pose as ControlNets, the
                 new style in the prompt, a style LoRA in place of the old style's when one
                 fits, and a style IP-Adapter for a style image;
       pose:     img2img from the base image with the new pose's skeleton (framed like the
                 base image) + the character through an IP-Adapter (padded, background
                 removed); once the pose is right, img2img from the best candidate;
       features: the region it concerns repainted (CLIPSeg mask) from the base image, or for a
                 change to the whole figure img2img + edges + IP-Adapter;
  3. the judge (one LLM call per candidate) compares each with the base image: how fully the
     change is made, and how well the character, outfit, style, pose, framing and background
     are kept, apart from the change. score = 35% change + 50% kept + 15% quality, capped
     below a pass while the change isn't made (so the unchanged image never wins) or the
     face or outfit isn't kept;
  4. the best candidate's verdict moves the knobs (change missing: more freedom; drift: less)
     and adds the details it lost to the prompt.

It stops at the threshold (confirmed by a second look) or after max_rounds; "one more round"
runs one more from where it left off. Kept candidates go into the design's catalog or start a
new design.

designs/<id>/design.json, original.png, catalog/<vid>.png, sessions/<sid>/session.json + images.
"""

from __future__ import annotations

import json
import random
import re
import shutil
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path

from PIL import Image

from .comfy import Cancelled
from .params import GenParams, edit_prompt, lora_stem, norm_tag, split_tags
from .sizes import closest_size, fit_to

KINDS = ("style", "pose", "features")
# subject_ip: the character IP-Adapter's weight in pose and feature edits. 0: off. Fed the base
# image it washed out the art style (muted colours, soft lines, a grey bodysuit) at 0.6 and
# still at 0.3, while img2img from the base image alone kept the style and the face
# (2026-10-05); kept as a setting for characters img2img can't hold.
DEFAULTS = {"threshold": 85, "max_rounds": 6, "batch": 3, "subject_ip": 0.0}
_lock = threading.Lock()


def settings(cfg: dict) -> dict:
    return {**DEFAULTS, **(cfg.get("designer") or {})}


# ---- storage --------------------------------------------------------------------------------

def designs_root(root: Path) -> Path:
    return Path(root) / "designs"


def design_dir(root: Path, design_id: str) -> Path:
    base = designs_root(root).resolve()
    d = (base / str(design_id)).resolve()
    if d.parent != base or design_id.startswith("_") or not (d / "design.json").is_file():
        raise FileNotFoundError(f"no design {design_id}")
    return d


def load(root: Path, design_id: str) -> dict:
    return json.loads((design_dir(root, design_id) / "design.json").read_text(encoding="utf-8"))


def _write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(path)


def save(root: Path, design: dict) -> None:
    _write(design_dir(root, design["id"]) / "design.json", design)


def file_of(root: Path, design_id: str, rel: str) -> Path:
    """A file inside a design's folder (rel as stored: "original.png", "catalog/x.png",
    "sessions/s/r1_1.png"); raises for anything outside it."""
    d = design_dir(root, design_id)
    p = (d / rel).resolve()
    if not p.is_relative_to(d) or not p.is_file():
        raise FileNotFoundError(rel)
    return p


def list_designs(root: Path) -> list[dict]:
    base = designs_root(root)
    out = []
    for d in sorted(base.glob("*/design.json"), key=lambda p: p.stat().st_mtime, reverse=True) if base.is_dir() else []:
        if d.parent.name.startswith("_"):
            continue
        try:
            out.append(json.loads(d.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out


def create(root: Path, image: Path, *, name: str = "", params: dict | None = None, source: dict | None = None,
           description: str = "") -> dict:
    """A new design from `image` (copied as original.png, as an RGB PNG). params: what made it
    ({"positive", "negative", "checkpoint", "loras", "steps", "cfg", "sampler_name",
    "scheduler", "seed"}), when known."""
    design_id = time.strftime("%Y%m%d-%H%M%S") + "_" + uuid.uuid4().hex[:6]
    d = designs_root(root) / design_id
    d.mkdir(parents=True)
    try:
        with Image.open(image) as im:
            im.convert("RGB").save(d / "original.png")
        design = {"id": design_id, "name": name.strip()[:80] or "Character " + time.strftime("%b %d %H:%M"),
                  "created": time.strftime("%Y-%m-%d %H:%M:%S"), "cover": "original.png",
                  "source": source or {}, "params": params or {}, "description": description, "catalog": [],
                  "sessions": []}
        _write(d / "design.json", design)
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise
    return design


def update(root: Path, design_id: str, **fields) -> dict:
    with _lock:
        design = load(root, design_id)
        if "name" in fields and str(fields["name"] or "").strip():
            design["name"] = str(fields["name"]).strip()[:80]
        if fields.get("cover"):
            file_of(root, design_id, fields["cover"])  # must be one of its images
            design["cover"] = fields["cover"]
        save(root, design)
    return design


def remove(root: Path, design_id: str) -> None:
    d = design_dir(root, design_id)
    trash = designs_root(root) / "_removed"
    trash.mkdir(exist_ok=True)
    dest, n = trash / d.name, 2
    while dest.exists():
        dest, n = trash / f"{d.name}_{n}", n + 1
    shutil.move(str(d), str(dest))


def keep(root: Path, design_id: str, session_id: str, image: str) -> dict:
    """A candidate of an edit, into the design's catalog (with what it changed and the
    settings that made it). Returns the catalog entry."""
    with _lock:
        design = load(root, design_id)
        sess = load_session(root, design_id, session_id)
        cand = _candidate(sess, image)
        src = file_of(root, design_id, f"sessions/{session_id}/{image}")
        vid = uuid.uuid4().hex[:8]
        (design_dir(root, design_id) / "catalog").mkdir(exist_ok=True)
        shutil.copy2(src, design_dir(root, design_id) / "catalog" / f"{vid}.png")
        entry = {"id": vid, "image": f"catalog/{vid}.png", "kind": sess["kind"], "change": change_text(sess),
                 "score": cand.get("score"), "from": sess["base"], "session": session_id, "candidate": image,
                 "created": time.strftime("%Y-%m-%d %H:%M:%S"), "params": cand.get("params") or {}}
        entry.update({k: cand[k] for k in ("first_score", "recheck", "judge_model", "confirm_model",
                                           "confirm_threshold", "confirmed") if k in cand})
        design["catalog"].append(entry)
        save(root, design)
    return entry


def new_from_candidate(root: Path, design_id: str, session_id: str, image: str, name: str = "") -> dict:
    """A candidate of an edit as a design of its own (its settings come along)."""
    design = load(root, design_id)
    sess = load_session(root, design_id, session_id)
    cand = _candidate(sess, image)
    src = file_of(root, design_id, f"sessions/{session_id}/{image}")
    return create(root, src, name=name or f"{design['name']} ({change_text(sess)[:40]})",
                  params={**design.get("params", {}), **(cand.get("params") or {})},
                  source={"design": design_id, "session": session_id, "image": image})


def rename_version(root: Path, design_id: str, image: str, name: str) -> dict:
    """Name one of a character's versions ("original.png" or a kept one); an empty name goes
    back to the one made from what changed."""
    name = str(name or "").strip()[:80]
    with _lock:
        design = load(root, design_id)
        if image == "original.png":
            design["original_name"] = name
        else:
            entry = next((e for e in design.get("catalog", []) if e["image"] == image), None)
            if not entry:
                raise FileNotFoundError(image)
            entry["name"] = name
        save(root, design)
    return design


def remove_from_catalog(root: Path, design_id: str, vid: str) -> None:
    with _lock:
        design = load(root, design_id)
        entry = next((e for e in design["catalog"] if e["id"] == vid), None)
        if not entry:
            raise FileNotFoundError(vid)
        design["catalog"].remove(entry)
        if design.get("cover") == entry["image"]:
            design["cover"] = "original.png"
        save(root, design)
        (design_dir(root, design_id) / entry["image"]).unlink(missing_ok=True)


def remove_session(root: Path, design_id: str, session_id: str) -> None:
    """An edit and all its renders deleted, to keep a character's folder small. Versions kept
    from it stay (the catalog has its own copies)."""
    with _lock:
        design = load(root, design_id)
        sdir = session_dir(root, design_id, session_id)
        if session_id not in design.get("sessions", []) and not sdir.is_dir():
            raise FileNotFoundError(session_id)
        design["sessions"] = [s for s in design.get("sessions", []) if s != session_id]
        save(root, design)
        shutil.rmtree(sdir, ignore_errors=True)


def merge(root: Path, into_id: str, from_id: str) -> dict:
    """Character `from_id` is really `into_id`: its image, its kept versions and its edits move
    into `into_id` (its image becomes a kept version there), and `from_id` goes to the bin.
    Each moved version keeps the settings that made it in full, since an edit reads a version's
    settings over its character's, and the two characters' differ."""
    if into_id == from_id:
        raise ValueError("a character can't be merged into itself")
    with _lock:
        into, frm = load(root, into_id), load(root, from_id)
        di, df = design_dir(root, into_id), design_dir(root, from_id)
        (di / "catalog").mkdir(exist_ok=True)
        (di / "sessions").mkdir(exist_ok=True)
        base_params = {"from_noise": False, **(frm.get("params") or {})}  # unknown origin: not redrawable
        moved = {}  # from's image paths -> into's

        def add_version(src_rel: str, entry: dict) -> None:
            vid = uuid.uuid4().hex[:8]
            shutil.copy2(df / src_rel, di / "catalog" / f"{vid}.png")
            moved[src_rel] = f"catalog/{vid}.png"
            into["catalog"].append({**entry, "id": vid, "image": f"catalog/{vid}.png"})

        add_version("original.png", {"kind": "merged", "change": f"merged · {frm['name']}", "score": None, "from": None,
                                     "created": time.strftime("%Y-%m-%d %H:%M:%S"), "params": base_params,
                                     "merged_from": {"id": from_id, "name": frm["name"], "source": frm.get("source")}})
        for e in frm.get("catalog", []):
            if (df / e["image"]).is_file():
                add_version(e["image"], {**e, "params": {**base_params, **(e.get("params") or {})},
                                         "merged_from": {"id": from_id, "name": frm["name"]}})
        for e in into["catalog"][-len(moved):]:
            if e.get("from") in moved:
                e["from"] = moved[e["from"]]
        sessions = []
        for sid in frm.get("sessions", []):
            src = df / "sessions" / sid
            if not (src / "session.json").is_file():
                continue
            new, n = sid, 2
            while (di / "sessions" / new).exists():
                new, n = f"{sid}_{n}", n + 1
            shutil.move(str(src), str(di / "sessions" / new))
            sess = json.loads((di / "sessions" / new / "session.json").read_text(encoding="utf-8"))
            sess.update(id=new, base=moved.get(sess.get("base"), sess.get("base")))
            _write(di / "sessions" / new / "session.json", sess)
            for e in into["catalog"]:
                if e.get("merged_from", {}).get("id") == from_id and e.get("session") == sid:
                    e["session"] = new
            sessions.append(new)
        into["sessions"] = into.get("sessions", []) + sessions
        save(root, into)
        frm["sessions"] = []  # they live in `into` now
        save(root, frm)
    remove(root, from_id)
    return into


def _candidate(sess: dict, image: str) -> dict:
    for r in sess.get("rounds", []):
        for c in r["candidates"]:
            if c["image"] == image:
                return c
    raise FileNotFoundError(image)


# ---- sessions (one edit each) -----------------------------------------------------------------

def session_dir(root: Path, design_id: str, session_id: str) -> Path:
    d = (design_dir(root, design_id) / "sessions" / session_id).resolve()
    if d.parent != (design_dir(root, design_id) / "sessions").resolve():
        raise FileNotFoundError(session_id)
    return d


def load_session(root: Path, design_id: str, session_id: str) -> dict:
    return json.loads((session_dir(root, design_id, session_id) / "session.json").read_text(encoding="utf-8"))


def save_session(root: Path, design_id: str, sess: dict) -> None:
    _write(session_dir(root, design_id, sess["id"]) / "session.json", sess)


def change_text(sess: dict) -> str:
    c = sess.get("change") or {}
    named = c.get("style") or c.get("pose") or ""
    return " · ".join(x for x in (f"{sess['kind']}: {named}" if named else sess["kind"], c.get("text", "")) if x)


def new_session(root: Path, design_id: str, *, base: str, kind: str, change: dict, threshold: float | None = None,
                max_rounds: int | None = None, target_image: Path | None = None) -> dict:
    """An edit of `base` (one of the design's images) changing only `kind`. change: {"text",
    and "style" / "pose": a saved one's name}. target_image: an uploaded style or pose image."""
    if kind not in KINDS:
        raise ValueError(f"choose one of {', '.join(KINDS)}")
    file_of(root, design_id, base)
    if kind == "features" and not (change.get("text") or "").strip():
        raise ValueError("say what to change")
    if kind != "features" and not ((change.get("text") or "").strip() or change.get(kind) or target_image):
        raise ValueError(f"choose a {kind}, add an image of it, or describe it")
    sid = time.strftime("%Y%m%d-%H%M%S") + "_" + uuid.uuid4().hex[:4]
    d = design_dir(root, design_id) / "sessions" / sid
    d.mkdir(parents=True)
    if target_image is not None:
        with Image.open(target_image) as im:
            im.convert("RGB").save(d / "target.png")
    sess = {"id": sid, "design": design_id, "base": base, "kind": kind,
            "change": {"text": (change.get("text") or "").strip(), "style": change.get("style") or "",
                       "pose": change.get("pose") or "", "image": "target.png" if target_image is not None else ""},
            "threshold": threshold, "max_rounds": max_rounds, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "queued", "error": None, "plan": None, "knobs": None, "rounds": [], "best": None,
            "confirmed": False, "cost_usd": 0.0, "log": []}
    _write(d / "session.json", sess)
    with _lock:
        design = load(root, design_id)
        design.setdefault("sessions", []).append(sid)
        save(root, design)
    return sess


# ---- planning --------------------------------------------------------------------------------

DESCRIBE_INSTRUCTIONS = """Describe this character image exactly as it is, for someone who must redraw it
without seeing it. Only what you see, with exact colours; nothing you can't see.

- face: age look, face shape, expression, eye colour and shape, brows, freckles, blush,
  marks, makeup.
- hair: colour, length, cut, how it's worn (bun, ponytail, loose), loose strands, fringe.
- outfit: every garment and armour piece, one per entry, each with its exact colours and
  shape ("olive-gold breastplate with a green gem in the centre", "black undersuit at the
  neck and arms", "dark gloves"). Accessories, weapons and gems only if visible.
- background: its colour and anything in it.
- framing: shot size (close-up, bust, cowboy shot, full body) and view angle.
- pose: stance, each arm and hand, head turn, gaze.
- style: line work, shading, palette.
JSON only."""

DESCRIBE_SCHEMA = {
    "type": "object",
    "properties": {"face": {"type": "string"}, "hair": {"type": "string"},
                   "outfit": {"type": "array", "items": {"type": "string"}},
                   "background": {"type": "string"}, "framing": {"type": "string"}, "pose": {"type": "string"},
                   "style": {"type": "string"}},
    "required": ["face", "hair", "outfit", "background", "framing", "pose", "style"],
    "additionalProperties": False,
}


def describe(backend, image: Path, max_side: int) -> tuple[dict, float]:
    """What the image shows, from the image alone. The planner saw the base image's prompt
    beside it and copied it ("silver body armor", "green sword on back") over gold-olive
    armour and no sword; shown the image alone the same model said "gold and black armor
    with green accents, beige background" (Qwen3-VL-30B, 2026-10-05)."""
    data, cost, _t = backend.complete(DESCRIBE_INSTRUCTIONS, [{"text": "IMAGE:"}, {"image": image},
                                                              {"text": "Describe it. JSON only."}],
                                      DESCRIBE_SCHEMA, "design_describe", max_side)
    out = {}
    for k in DESCRIBE_SCHEMA["properties"]:
        v = data.get(k)
        out[k] = "; ".join(str(x).strip() for x in v if str(x).strip()) if isinstance(v, list) else str(v or "").strip()
    return {k: v for k, v in out.items() if v}, cost


PLAN_INSTRUCTIONS = """You plan ONE edit of an existing character image for Stable Diffusion XL (Pony family).
The person wants the SAME image - same character, face, hair, outfit, colours, accessories,
art style, pose, framing and background - with only the requested change.

You get the BASE image, the KIND of change (style, pose or features), the request, and the
prompt that made the base image when it is known.

THE BASE IMAGE IS THE TRUTH, NOT ITS PROMPT. Prompts are often wrong about their own image:
one said "silver body armor", "plain background" and "green sword on back" for an image of
gold-olive armour on a tan background with no sword. You also get SEEN, a description made
from the image alone: where SEEN and the original prompt disagree, SEEN is right.

- seen: SEEN, as given (correct it only where the image clearly shows otherwise).
- style_tags: the prompt's quality and drawing-style tags. For a STYLE change, the STYLE TAGS
  given with the new style in place of the old; otherwise the STYLE TAGS given, as they are.
- content: the tags for everything else, written from SEEN and the image: the character
  (1girl or similar, build, face shape, age look, expression, eyes, brows, freckles, blush,
  hair colour, length, style, loose strands), every garment and armour piece with its exact
  colour, accessories (only those visible), framing and view, pose, and the background with
  its colour ("light beige background"). Weight the defining details like (x:1.2). Apply the
  change: pose, arm, hand, leg, head and gaze tags for a pose change; the requested features
  for a feature change. Nothing that isn't in the image or the change. (Used only when the
  base image has no prompt of its own; otherwise its own tags are kept, see change_tags.)
- change_tags: the tags that make the change and nothing else: for a pose change the new
  stance, arm, hand, leg, head and gaze tags ("arms crossed", "weight on left leg"); for a
  feature change the requested features ("hair down", "shoulder-length hair"); for a style
  change none (the new style is in style_tags).
- negative_add: a few tags against the old version coming back (the old pose or style) or
  against likely mistakes (wrong colours seen in the prompt, e.g. "silver armor").
- drop_loras: for a STYLE change only, the LoRAs listed that carry the OLD style; otherwise
  none (the LoRAs are part of how the character looks).
- region (features only): the part of the image to repaint, as a short noun phrase an image
  segmenter can find ("hair", "necklace", "left hand", "face"). Cover where the changed part
  is now AND where it will be ("hair down" -> "hair and shoulders"). "" when the change
  concerns the whole figure and can't be done in one region.
- protect (features only): a part next to the region that must not be repainted ("face"
  when changing the hair, a hat or earrings); "" for none or when the face is the change.
- summary: the change in one short line.
- must_change: 1-4 checks that the change is made, each one visible fact.
- must_keep: 8-14 details from `seen` that must not change, each with its aspect: identity
  (face, eyes, hair, skin, build), outfit (each garment, piece, colour, gem, accessory),
  style, pose, framing, background. Concrete and checkable ("olive-gold breastplate", "tan
  background", "stern expression"), never prompt tags like "1girl" or "masterpiece", never
  the thing being changed. tag: the short prompt tag that draws it ("freckles", "beige
  background", "olive-gold breastplate", "loose hair strands framing face"); every one of
  them ends up in the prompt.
- For a POSE change the framing stays the base image's (a cowboy shot stays one; never add
  "full body" from the pose image); must_change names only body, limb, head and gaze facts.
- For a STYLE change describe the new style in drawing terms (line work, shading, palette,
  brushwork, medium); a painted illustration is not "photorealistic".

Reply with JSON only."""

ASPECTS = ("identity", "outfit", "style", "pose", "framing", "background")

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "seen": {"type": "object", "properties": {k: {"type": "string"} for k in
                                                  ("face", "hair", "outfit", "background", "framing", "pose", "style")},
                 "required": ["face", "hair", "outfit", "background", "framing", "pose", "style"],
                 "additionalProperties": False},
        "style_tags": {"type": "string"},
        "content": {"type": "string"},
        "change_tags": {"type": "array", "items": {"type": "string"}},
        "negative_add": {"type": "array", "items": {"type": "string"}},
        "drop_loras": {"type": "array", "items": {"type": "string"}},
        "region": {"type": "string"},
        "protect": {"type": "string"},
        "summary": {"type": "string"},
        "must_change": {"type": "array", "items": {"type": "string"}},
        "must_keep": {"type": "array", "items": {
            "type": "object", "properties": {"detail": {"type": "string"}, "aspect": {"type": "string", "enum": list(ASPECTS)},
                                             "tag": {"type": "string"}},
            "required": ["detail", "aspect", "tag"], "additionalProperties": False}},
    },
    "required": ["seen", "style_tags", "content", "change_tags", "negative_add", "drop_loras", "region", "protect",
                 "summary", "must_change", "must_keep"],
    "additionalProperties": False,
}


def keep_items(the_plan: dict, kind: str) -> list[dict]:
    """The plan's must_keep as [{"detail", "aspect"}], without the aspect being changed (an
    older plan's plain strings count as outfit)."""
    out = []
    for x in the_plan.get("must_keep") or []:
        item = x if isinstance(x, dict) else {"detail": str(x), "aspect": "outfit"}
        detail, aspect = str(item.get("detail") or "").strip(), str(item.get("aspect") or "outfit").strip().lower()
        if detail and aspect != kind:
            out.append({"detail": detail, "aspect": aspect if aspect in ASPECTS else "outfit",
                        "tag": str(item.get("tag") or "").strip()})
    return out[:14]


def target_parts(kind: str, change: dict, target_image: Path | None, target_tags: str) -> list[dict]:
    """What the change is, for the planner and the judge: the words, the saved item's tags,
    and its image (a style or pose image), each saying what to take from it."""
    parts = [{"text": f"KIND OF CHANGE: {kind}\nREQUEST: {(change.get('text') or '').strip() or '(none in words)'}"}]
    if target_tags.strip():
        parts.append({"text": f"THE NEW {kind.upper()} (saved {kind} '{change.get(kind)}'): {target_tags.strip()}"})
    if target_image is not None:
        what = {"style": "only its drawing style; ignore who and what it shows",
                "pose": "only its body pose: limbs, torso, head and gaze; ignore its framing, who is in it "
                        "and how it is drawn"}.get(kind, "")
        parts += [{"text": f"THE NEW {kind.upper()} ({what}):"}, {"image": target_image}]
    return parts


def plan(backend, base_image: Path, kind: str, change: dict, *, target_image: Path | None, target_tags: str,
         original_positive: str, example_prompt: str, loras: list[tuple[str, str]], max_side: int,
         seen: dict | None = None) -> tuple[dict, float]:
    """The edit's prompt, guards and checks (see PLAN_INSTRUCTIONS). loras: [(stem, type)];
    seen: describe()'s description of the base image."""
    parts = [{"text": "BASE IMAGE (keep everything but the change):"}, {"image": base_image},
             *target_parts(kind, change, target_image, target_tags)]
    if seen:
        parts.append({"text": "SEEN (described from the base image alone; trust it over the prompt):\n"
                              + "\n".join(f"{k}: {v}" for k, v in seen.items())})
    # Only the original prompt's style tags are shown: shown its description of the character,
    # the planner kept "(green sword on back:1.3)" and "plain background" over an image with
    # no sword and a beige backdrop, even with a correct description beside it (2026-10-05).
    style_part, _content = split_style(original_positive)
    if style_part:
        parts.append({"text": f"STYLE TAGS (the base image's quality and drawing-style tags):\n{style_part}"})
    if example_prompt.strip():
        parts.append({"text": f"EXAMPLE PROMPT (match its tag style):\n{split_style(example_prompt)[0][:400] or example_prompt.strip()[:400]}"})
    if loras:
        parts.append({"text": "LORAS the base image was made with:\n" + "\n".join(f"- {s} [{t or 'unknown type'}]"
                                                                               for s, t in loras)})
    parts.append({"text": f"Plan the {kind} change. JSON only."})
    data, cost, _t = backend.complete(PLAN_INSTRUCTIONS, parts, PLAN_SCHEMA, "design_plan", max_side)
    cost = float(cost or 0)
    out = {k: data.get(k) for k in PLAN_SCHEMA["properties"]}
    style_part = split_style(original_positive)[0]
    out["seen"] = {k: str(v).strip() for k, v in (out.get("seen") or {}).items() if str(v).strip()} \
        if isinstance(out.get("seen"), dict) else {}
    out["seen"] = {**out["seen"], **(seen or {})}  # the image-only look wins
    style_tags = style_part if (kind != "style" and style_part) else str(out.get("style_tags") or style_part).strip()
    change_tags = [str(x).strip() for x in out.get("change_tags") or [] if str(x).strip()][:10]
    own = split_style(original_positive)[1]
    if own:
        # The base image's own tags drew its face and look: rewritten from a description, the
        # face drifted ("serious expression" became a frown) where the original's "cute face,
        # soft jawline" kept it (2026-10-05). So they stay, less those the image contradicts,
        # those it doesn't show and those the change replaces; the change's tags are added.
        verdicts = {}
        reviewed, removed, c = review_tags(backend, own, out["seen"], kind, change, out.get("summary") or "", max_side,
                                           verdicts=verdicts)
        cost += c
        content = edit_prompt(reviewed, change_tags, [])
        out["removed_tags"] = removed
        # From the image's own noise (see regen_source), the original prompt with only what the
        # change replaces swapped: every other tag, seen or not, is part of how that noise drew
        # this character.
        replaced = [t for t, v in verdicts.items() if v == "changes"]
        out["regen_positive"] = swap_tags(original_positive, replaced, change_tags) if kind != "style" else \
            "\n\n".join(x for x in (style_tags, swap_tags(own, replaced, change_tags)) if x)
    else:
        content = edit_prompt(str(out.get("content") or data.get("positive") or "").strip(), change_tags, [])
    out["positive"] = "\n\n".join(x for x in (style_tags, content) if x) or original_positive.strip()
    out["change_tags"] = change_tags
    out.pop("style_tags", None)
    out.pop("content", None)
    for k in ("negative_add", "drop_loras", "must_change"):
        out[k] = [str(x).strip() for x in out.get(k) or [] if str(x).strip()][:12]
    if kind != "style":
        out["drop_loras"] = []  # the LoRAs are part of how the character looks
    derived = keep_from_seen(out["seen"], kind)
    planned = [x for x in keep_items(out, kind) if specific(x["detail"])]
    # The planner's list varies run to run (once it was just "face", "hair", "outfit"...): the
    # details come from the image's description, the planner's only when it gives more of them.
    out["must_keep"] = planned if len(planned) > len(derived) else derived
    out["positive"] = with_keep_tags(out["positive"], out["must_keep"], own_prompt=bool(own))
    out["region"] = (out.get("region") or "").strip() if kind == "features" else ""
    out["protect"] = (out.get("protect") or "").strip() if out["region"] else ""
    if out["protect"] and out["protect"].lower() in out["region"].lower():
        out["protect"] = ""  # the change is in it
    out["summary"] = (out.get("summary") or change.get("text") or kind).strip()
    return out, cost


REVIEW_INSTRUCTIONS = """You check a Stable Diffusion prompt against a description of the image it should
draw. For EACH numbered tag answer one verdict:
- "keep": agrees with the description, or the description says nothing against it (face
  and body descriptors like "cute face", "soft jawline", quality and framing words stay).
- "contradicted": the description says otherwise (the tag says silver, the image is gold;
  the tag says plain background, the image's is beige). These stay in the prompt (it drew
  the image); the verdict is only noted.
- "not visible": an object, garment, weapon or accessory the description doesn't show (it
  would be added to the image).
- "changes": the tag describes what the requested change replaces (for a pose change: the
  old stance, arms, hands, legs, head turn and gaze; for a feature change: the old version of
  those features; for a style change: words for the old drawing style, medium, line work or
  shading).
JSON only."""

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {"tags": {"type": "array", "items": {
        "type": "object", "properties": {"n": {"type": "integer"}, "tag": {"type": "string"},
                                         "verdict": {"type": "string",
                                                     "enum": ["keep", "contradicted", "not visible", "changes"]}},
        "required": ["n", "tag", "verdict"], "additionalProperties": False}}},
    "required": ["tags"],
    "additionalProperties": False,
}


def review_tags(backend, content: str, seen: dict, kind: str, change: dict, summary: str,
                max_side: int, verdicts: dict | None = None) -> tuple[str, list[str], float]:
    """`content` (the base image's own non-style tags, lines kept) less the tags the image's
    description contradicts or doesn't show and those the change replaces, by a text-only call
    (shown the image next to its prompt, the planner believed the prompt). Unanswered tags stay,
    and so do the ones it calls "contradicted": the tag drew the image as it is. "(silver body
    armor:1.3)" drew the original's muted, pale gold plate; dropped as "the image is gold", every
    render's armour turned saturated, ornate gold (2026-10-05). Returns (content, removed tags, cost)."""
    tags = split_tags(content)
    if not tags:
        return content, [], 0.0
    parts = [{"text": "IMAGE DESCRIPTION (true):\n" + "\n".join(f"{k}: {v}" for k, v in (seen or {}).items())
                      + f"\n\nREQUESTED CHANGE ({kind}): {summary or (change.get('text') or '')}"
                      + "\n\nTAGS:\n" + "\n".join(f"{i}. {t}" for i, t in enumerate(tags, 1))
                      + "\n\nA verdict for every tag. JSON only."}]
    data, cost, _t = backend.complete(REVIEW_INSTRUCTIONS, parts, REVIEW_SCHEMA, "design_review", max_side)
    drop = []
    for x in data.get("tags") or []:
        if not isinstance(x, dict):
            continue
        verdict = str(x.get("verdict") or "keep").strip().lower()
        n = x.get("n")
        tag = tags[n - 1] if isinstance(n, int) and 1 <= n <= len(tags) else str(x.get("tag") or "")
        if verdicts is not None and tag in tags:
            verdicts[tag] = verdict
        if verdict in ("not visible", "changes") and tag in tags:
            drop.append(tag)
    return (edit_prompt(content, [], drop) if drop else content), drop, cost


def swap_tags(prompt: str, replaced: list[str], new: list[str]) -> str:
    """`prompt` with `replaced` taken out and `new` put where the first of them was (same line,
    same place), the way the hand edits that made a set of matching characters did ("arms at
    sides" -> "arms crossed"). With nothing replaced, `new` goes where it fits (edit_prompt)."""
    drop = {norm_tag(t) for t in replaced}
    have = {norm_tag(t) for t in split_tags(prompt)}
    new = [t for t in dict.fromkeys(x.strip() for x in new if x.strip()) if norm_tag(t) not in have - drop]
    lines, placed = [], False
    for line in prompt.split("\n"):
        tags = [x.strip() for x in line.split(",") if x.strip()]
        if not tags:
            lines.append(line)
            continue
        out = []
        for t in tags:
            if norm_tag(t) in drop:
                if not placed:
                    out += new
                    placed = True
            else:
                out.append(t)
        if out:
            lines.append(", ".join(out))
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return text if placed else edit_prompt(text, new, [])


def regen_source(made: dict) -> dict | None:
    """How to draw the image again from its own noise, when it was drawn from noise alone
    (txt2img, denoise 1, no reference image, ControlNet or IP-Adapter): {"seed", "index" (its
    0-based place in its batch), "of" (the batch size), "size"}. A batch gives each image its
    own slice of the seed's noise, so image 8 of 8 is that seed AND that place. Five
    characters made this way, one prompt tag changed each time, came out the same character
    (2026-10-05); an image whose making isn't known gets None."""
    if not made.get("from_noise") or made.get("seed") is None or made.get("batch_index") is None:
        return None
    try:
        index, of = int(made["batch_index"]), int(made.get("batch_of") or int(made["batch_index"]) + 1)
        size = [int(x) for x in made["size"]] if made.get("size") else None
    except (TypeError, ValueError):
        return None
    return {"seed": int(made["seed"]), "index": index, "of": max(of, index + 1), "size": size}


SUBJECT_TAGS = re.compile(r"^(\d+(girl|boy|other)s?|solo|female|male|woman|man|girl|boy|multiple girls|multiple boys)$")


def split_style(prompt: str) -> tuple[str, str]:
    """(quality and style tags, the rest): everything before the first subject tag ("1girl",
    "solo", "female"...) is the style part, the way Pony prompts are written; with no subject
    tag, only the score_/source_/rating_ tags are. Lines are kept."""
    if not (prompt or "").strip():
        return "", ""
    lines = [[t.strip() for t in ln.split(",") if t.strip()] for ln in prompt.strip().split("\n")]
    flat = [(i, t) for i, ln in enumerate(lines) for t in ln]
    cut = next((n for n, (_i, t) in enumerate(flat) if SUBJECT_TAGS.match(norm_tag(t))), None)
    if cut is None:
        head = [(i, t) for i, t in flat if norm_tag(t).startswith(("score_", "source_", "rating_"))]
        rest = [(i, t) for i, t in flat if (i, t) not in head]
    else:
        head, rest = flat[:cut], flat[cut:]

    def join(pairs):
        out = {}
        for i, t in pairs:
            out.setdefault(i, []).append(t)
        return "\n".join(", ".join(v) for _k, v in sorted(out.items()))
    return join(head), join(rest)


GENERIC = {"face", "hair", "outfit", "background", "framing", "style", "pose", "build", "expression", "eyes", "brows",
           "freckles", "blush", "skin", "armor", "armour", "clothes", "clothing", "colors", "colours", "body"}
ASPECT_OF_FIELD = {"face": "identity", "hair": "identity", "outfit": "outfit", "background": "background",
                   "framing": "framing", "pose": "pose", "style": "style"}


def specific(detail: str) -> bool:
    """A checkable detail ("olive-gold breastplate"), not a heading ("outfit")."""
    words = norm_tag(detail).split()
    return bool(words) and not (len(words) == 1 and words[0] in GENERIC)


def backdrop_words(text: str) -> str:
    """The base image's backdrop as words for a "<words> background" tag: "Light tan or beige
    background." -> "light tan" (it made "(light tan or beige background. background:1.2)",
    2026-10-05)."""
    t = re.split(r"[.;,]|\s+or\s+|\s+with\s+", str(text or "").strip())[0]
    return re.sub(r"(?i)^(a\s+|an\s+)?(solid|plain|simple)\s+|\s*(color\s+|colou?red\s+)?background$", "",
                  t.strip()).strip().lower()


def keep_from_seen(seen: dict, kind: str) -> list[dict]:
    """The details to keep, straight from the image's description: each comma/semicolon part
    of each field is one detail of that field's aspect (a lone word gets its field: "brown" ->
    "brown hair"); the changed aspect and negations ("no visible makeup") are left out."""
    out = []
    for field, aspect in ASPECT_OF_FIELD.items():
        if aspect == kind or not seen.get(field):
            continue
        parts = re.split(r";|,(?![^(]*\))", str(seen[field])) if field != "background" else [str(seen[field])]
        for part in parts:
            d = re.sub(r"(?i)^(and|with|plus)\s+", "", part.strip().strip(".").strip())
            if not d or norm_tag(d).startswith(("no ", "without ", "none")):
                continue
            if len(d.split()) == 1 or field == "background":
                d = f"{d} {field}" if field in ("hair", "background") and field not in d.lower() else d
            if specific(d):
                out.append({"detail": d, "aspect": aspect, "tag": d if len(d) <= 60 else ""})
    return out[:16]


MARKS = ("freckle", "scar", "mole", "tattoo", "strand", "beauty mark", "piercing", "earring", "makeup", "eyeliner")


GARMENT_HEAD_END = re.compile(r"\s+(with|on|at|over|under|covering|across|around|framing|visible|in|of)\s+.*$")


def garment_head(tag: str) -> str:
    """The thing a description tag is about: "gold-colored segmented armor covering torso" ->
    "armor", "gold gauntlets with green accents" -> "gauntlet"."""
    words = GARMENT_HEAD_END.sub("", norm_tag(tag)).split()
    return _noun(words[-1]) if words else ""


def _noun(word: str) -> str:
    """One spelling of a noun: singular, American ("armours" -> "armor")."""
    return re.sub(r"our$", "or", re.sub(r"s$", "", word))


def with_keep_tags(positive: str, items: list[dict], own_prompt: bool = False) -> str:
    """The prompt with a tag for every detail to keep that it doesn't draw yet: the plan listed
    "light freckles" and "beige background" to keep but its prompt kept "cute face" and "solid
    color background", and the renders lost both (2026-10-05)."""
    have = [set(norm_tag(t).split()) for t in split_tags(positive)]
    add = []
    for x in items:
        tag = (x.get("tag") or "").strip()
        words = set(norm_tag(tag).split())
        # With the image's own prompt, its face tags drew the face: description tags for it
        # ("serious expression", "thin arched brows") moved it, so only marks are added.
        face_ok = not own_prompt or any(m in tag.lower() for m in MARKS)
        # Nor are garments it already names, in other words: "gold-colored segmented armor" next
        # to its "(silver body armor:1.3)" made the plate saturated, ornate gold (2026-10-05).
        head = garment_head(tag)
        named = own_prompt and x["aspect"] == "outfit" and head and any(
            head in {_noun(w) for w in h} for h in have)
        if (tag and x["aspect"] in ("outfit", "background") and not named
                or (x["aspect"] == "identity" and face_ok and tag)) \
                and not any(words <= h for h in have):
            add.append(tag)
    return edit_prompt(positive, add, []) if add else positive


# ---- judging ---------------------------------------------------------------------------------
# The judge fills in a checklist, and the scores are worked out here from its answers. Asked
# for 0-10 scores directly, a judge (Qwen3-VL-30B) gave identity 10 and outfit 10 to a
# candidate whose armour had turned from olive-gold to silver, while listing the very
# differences that should have cost it, and "change 10" to arms that weren't crossed: every
# edit passed in round 1 and nothing was optimised (2026-10-05).

JUDGE_INSTRUCTIONS = """You check one edit of a character image. The person asked for ONE change to the BASE
image; everything else should be the same. Compare the CANDIDATE with the BASE, detail by
detail. Be strict: you are the only check; a lenient answer ships a wrong image.

Look at the BASE image for what each detail really is (the list may be wrong about it; the
base image wins), then find it in the CANDIDATE and write what the candidate shows BEFORE
the verdict:
- keep: one entry per detail listed, in the same order. in_candidate: what the candidate
  shows for it, with colours. verdict: "same" (you'd not notice), "slightly different"
  (shade, size or shape a little off), "different" (another colour, shape or design), or
  "missing".
- change: one entry per check listed. seen: what the candidate shows. verdict: "done",
  "partly" or "not done". Look at the actual limbs, head, hair, colours: don't assume.
- other_differences: anything else that differs from the BASE and isn't the change (a
  colour, a garment, the background, the face, the framing, a repainting seam or halo),
  each with its aspect and "minor" or "major".
- quality 0-10: anatomy and artifacts (hands, extra limbs, melted details, seams).
- missing: what of the change isn't there yet, "" if none.
- prompt_add: up to 4 tags that would bring back what was lost or complete the change (for a
  style change never style words); prompt_remove: things that crept in, as tags ("sword",
  "silver armor"); negative_add: up to 3 tags against what crept in (e.g. a wrong colour).
For a pose change the framing (shot size, crop) stays the BASE's, even when the pose image
is full body. JSON only."""

KEEP_VERDICTS = {"same": 10.0, "slightly different": 7.0, "different": 2.0, "missing": 0.0}
CHANGE_VERDICTS = {"done": 10.0, "partly": 5.0, "not done": 0.0}
SEVERITY_COST = {"major": 3.0, "minor": 1.0}

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "keep": {"type": "array", "items": {
            "type": "object", "properties": {"detail": {"type": "string"}, "in_candidate": {"type": "string"},
                                             "verdict": {"type": "string", "enum": list(KEEP_VERDICTS)}},
            "required": ["detail", "in_candidate", "verdict"], "additionalProperties": False}},
        "change": {"type": "array", "items": {
            "type": "object", "properties": {"check": {"type": "string"}, "seen": {"type": "string"},
                                             "verdict": {"type": "string", "enum": list(CHANGE_VERDICTS)}},
            "required": ["check", "seen", "verdict"], "additionalProperties": False}},
        "other_differences": {"type": "array", "items": {
            "type": "object", "properties": {"what": {"type": "string"},
                                             "aspect": {"type": "string", "enum": list(ASPECTS)},
                                             "severity": {"type": "string", "enum": list(SEVERITY_COST)}},
            "required": ["what", "aspect", "severity"], "additionalProperties": False}},
        "quality": {"type": "number"},
        "missing": {"type": "string"},
        "prompt_add": {"type": "array", "items": {"type": "string"}},
        "prompt_remove": {"type": "array", "items": {"type": "string"}},
        "negative_add": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["keep", "change", "other_differences", "quality", "missing", "prompt_add", "prompt_remove",
                 "negative_add"],
    "additionalProperties": False,
}


def _verdict(text, table: dict) -> float | None:
    t = str(text or "").strip().lower().replace("_", " ")
    for k in sorted(table, key=len, reverse=True):  # "slightly different" before "different"
        if t.startswith(k) or k in t:
            return table[k]
    return None


def checklist_scores(data: dict, items: list[dict], checks: list[str], kind: str) -> dict:
    """0-10 per aspect, the change and quality, from the judge's checklist: each kept detail
    scores by its verdict (same 10, slightly different 7, different 2, missing 0) in its
    aspect, each other difference costs its aspect 1 (minor) or 3 (major), the change is the
    mean of its checks. Details are matched to the plan's by position (by text when the judge
    dropped some); an aspect with nothing listed is 10 less its other differences."""
    got = [x for x in data.get("keep") or [] if isinstance(x, dict)]
    per: dict[str, list[float]] = {a: [] for a in ASPECTS}
    diffs = []
    for i, item in enumerate(items):
        ans = got[i] if len(got) == len(items) else next(
            (g for g in got if str(g.get("detail", "")).strip().lower() == item["detail"].lower()), None)
        val = _verdict((ans or {}).get("verdict"), KEEP_VERDICTS)
        if val is None:
            val = 5.0  # not answered: neither kept nor lost
        per[item["aspect"]].append(val)
        if val < 10:
            seen = str((ans or {}).get("in_candidate") or "").strip()
            diffs.append(f"{item['detail']}: {seen}" if seen else f"{item['detail']} ({(ans or {}).get('verdict', '?')})")
    cost = {a: 0.0 for a in ASPECTS}
    for d in data.get("other_differences") or []:
        if not isinstance(d, dict):
            continue
        aspect = str(d.get("aspect") or "outfit").lower()
        aspect = aspect if aspect in ASPECTS else "outfit"
        cost[aspect] += SEVERITY_COST.get(str(d.get("severity") or "minor").lower(), 1.0)
        if str(d.get("what") or "").strip():
            diffs.append(str(d["what"]).strip())
    out = {a: round(max(0.0, (sum(per[a]) / len(per[a]) if per[a] else 10.0) - cost[a]), 2) for a in ASPECTS}
    out[kind] = 10.0 if kind in out else None  # the changed aspect is scored as the change
    vals = [_verdict((c or {}).get("verdict"), CHANGE_VERDICTS) for c in data.get("change") or [] if isinstance(c, dict)]
    vals = [v for v in vals if v is not None]
    out["change"] = round(sum(vals) / len(vals), 2) if vals else 0.0
    if len(vals) < len(checks):  # a check left unanswered isn't done
        out["change"] = round(sum(vals) / len(checks), 2)
    out["quality"] = max(0.0, min(10.0, float(data.get("quality") or 0)))
    out["differences"] = diffs
    return {k: v for k, v in out.items() if v is not None}


def kept_aspects(kind: str) -> tuple[str, ...]:
    """The aspects that must stay for this kind of change (the changed one is the change)."""
    return tuple(a for a in ASPECTS if a != kind)


# What's kept, weighted: the same character is first the same face, then the same clothes.
KEPT_WEIGHTS = {"identity": 2.0, "outfit": 1.5, "style": 1.0, "pose": 1.0, "framing": 0.5, "background": 1.0}


def kept_score(v: dict, kind: str) -> float:
    num = lambda k: max(0.0, min(10.0, float(v.get(k) if v.get(k) is not None else 10)))  # noqa: E731
    w = {a: KEPT_WEIGHTS[a] for a in kept_aspects(kind)}
    return round(sum(num(a) * x for a, x in w.items()) / sum(w.values()), 2)


def combined(v: dict, kind: str) -> float:
    """0-100: 35% the change, 50% what's kept (the face counting most), 15% quality.
    While the change is barely made the score stays below any sensible threshold, however
    well the rest is kept (the base image itself would otherwise win an edit), and one kept
    aspect clearly broken (below 7: a changed face, a different outfit) can't pass either,
    however good the rest: a face judged 6 next to 10s elsewhere scored 89.5 (2026-10-04)."""
    num = lambda k: max(0.0, min(10.0, float(v.get(k) if v.get(k) is not None else 10)))  # noqa: E731
    score = 10 * (0.35 * num("change") + 0.5 * kept_score(v, kind) + 0.15 * num("quality"))
    if num("change") < 6:
        score = min(score, 50 + 3 * num("change"))
    worst = min(num(a) for a in kept_aspects(kind))
    if worst < 7:
        score = min(score, 70 + 2 * worst)
    # The same character: its face and outfit need 8 to pass the default 85. A pose edit
    # whose armour had become another suit (outfit 7) passed at 94.4 (2026-10-04).
    same = min(num("identity"), num("outfit"))
    if same < 8:
        score = min(score, 76 + same)
    return round(score, 1)


def _lab(rgb) -> "np.ndarray":
    import numpy as np
    c = np.asarray(rgb, dtype=float) / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz = xyz / np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])])


def backdrop_colour(image: Path):
    """The median colour of the image's top edge and the top half of its sides (a cowboy shot's
    legs reach the bottom edge, so it isn't used)."""
    import numpy as np
    with Image.open(image) as im:
        a = np.asarray(im.convert("RGB").resize((128, 128)))
    edge = np.concatenate([a[:4].reshape(-1, 3), a[:64, :4].reshape(-1, 3), a[:64, -4:].reshape(-1, 3)])
    return np.median(edge, axis=0)


def backdrop_shift(base: Path, candidate: Path) -> float:
    """How far the backdrop's colour moved, in CIE76 delta E (2-3 just noticeable; beige to grey
    is ~15). Measured, because a judge called a grey backdrop for a beige one "same"."""
    import numpy as np
    return float(np.linalg.norm(_lab(backdrop_colour(base)) - _lab(backdrop_colour(candidate))))


def face_crop(image: Path, out: Path) -> Path | None:
    """A square close-up of the main character's face (DWPose face points), or None. At the
    judge's 512 px a cowboy shot's face is ~40 px: freckles and loose strands can't be seen."""
    from . import pose as posemod
    if out.is_file():
        return out
    try:
        if not posemod.available():
            return None
        found = posemod.detect(image)
    except Exception:
        return None
    pts = [p for p in (found or {}).get("face") or [] if p] or \
          [p for i, p in enumerate((found or {}).get("body") or []) if p and i in (0, 14, 15, 16, 17)]
    if len(pts) < 3:
        return None
    with Image.open(image) as im:
        w, h = im.size
        xs, ys = [p[0] * w for p in pts], [p[1] * h for p in pts]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2 - 0.15 * (max(ys) - min(ys))
        side = max(max(xs) - min(xs), max(ys) - min(ys), 32) * 2.2  # forehead, hair and chin too
        box = [int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)]
        im.convert("RGB").crop(box).resize((384, 384)).save(out)
    return out


def judge(backend, base_image: Path, candidate: Path, kind: str, the_plan: dict, *, change: dict,
          target_image: Path | None, target_tags: str, positive: str, max_side: int,
          work_dir: Path | None = None, target_pose: dict | None = None) -> tuple[dict, float]:
    """One candidate against the base image: the checklist (see JUDGE_INSTRUCTIONS) answered
    with face close-ups of both and a description of the candidate made blind (by a call that
    hasn't seen the checklist, so it can't just agree with it), then the backdrop's colour
    measured, and for a pose edit with a skeleton (`target_pose`, framed as rendered) the
    pose. positive is kept for callers; the judge isn't shown it."""
    items, checks = keep_items(the_plan, kind), list(the_plan.get("must_change") or [])
    cost = 0.0
    blind, c = describe(backend, candidate, max_side)
    cost += c
    faces = []
    if work_dir is not None:
        bf = face_crop(base_image, work_dir / "base_face.png")
        cf = face_crop(candidate, work_dir / f"{candidate.stem}_face.png")
        if bf is not None and cf is not None:
            faces = [{"text": "BASE FACE (close-up):"}, {"image": bf}, {"text": "CANDIDATE FACE (close-up):"},
                     {"image": cf}]
    seen = the_plan.get("seen") or {}
    parts = [{"text": f"THE CHANGE: {the_plan.get('summary', '')}\n"
                      + ("CHANGE CHECKS (answer each, in order):\n" + "\n".join(f"{i}. {c}" for i, c in
                                                                                 enumerate(checks, 1)) + "\n"
                         if checks else "")
                      + ("DETAILS TO KEEP (answer each, in order):\n" + "\n".join(
                          f"{i}. [{x['aspect']}] {x['detail']}" for i, x in enumerate(items, 1)) if items else "")},
             *target_parts(kind, change, target_image, target_tags)[1:],
             {"text": "BASE IMAGE:"}, {"image": base_image},
             {"text": "CANDIDATE:"}, {"image": candidate}, *faces,
             {"text": "BASE, described from the image alone:\n" + "\n".join(f"{k}: {v}" for k, v in seen.items())
                      + "\n\nCANDIDATE, described from the image alone (by someone who hadn't seen the list):\n"
                      + "\n".join(f"{k}: {v}" for k, v in blind.items())
                      + "\n\nWhere the two descriptions differ on a listed detail, look again: it is probably "
                        "not the same. Fill in the checklist. JSON only."}]
    data, c, _t = backend.complete(JUDGE_INSTRUCTIONS, parts, JUDGE_SCHEMA, "design_judge", max_side)
    cost += c
    v = checklist_scores(data, items, checks, kind)
    if kind != "style":  # a new style may tint the backdrop; nothing else should
        shift = backdrop_shift(base_image, candidate)
        v["backdrop_shift"] = round(shift, 1)
        if shift > 6:
            v["background"] = round(min(v.get("background", 10), max(0.0, 10 - (shift - 6) / 2)), 2)
            v["differences"] = [f"backdrop colour moved (delta E {shift:.0f})"] + v["differences"]
    if kind == "pose" and target_pose:
        # Measured, the pose can't be talked into: Qwen3-VL-30B gave "change 8.3, pose 10" to
        # arms still at the sides, it passed at 85, and two rounds refined it (2026-10-05).
        from . import pose as posemod
        with Image.open(candidate) as im:
            found_size = im.size
        match, off = posemod.limb_match(target_pose, posemod.detect(candidate), found_size)
        if match is not None:
            v["pose_match"] = match
            if match < v.get("change", 10):
                v["change"] = match
            if off:
                v["differences"] = [f"pose measured: {', '.join(off)}"] + v["differences"]
    for k in ("prompt_add", "prompt_remove", "negative_add"):
        v[k] = [str(x).strip() for x in data.get(k) or [] if str(x).strip()][:6]
    v["differences"] = v["differences"][:10]
    v["missing"] = str(data.get("missing") or "").strip()
    v["blind"] = blind
    v["score"] = combined(v, kind)
    v["kept"] = round(kept_score(v, kind), 1)
    return v, cost


# ---- the knobs: what each kind of edit renders with, and how a verdict moves them ---------------

def initial_knobs(kind: str, the_plan: dict, *, has_skeleton: bool, has_style_image: bool,
                  has_style_lora: bool = False, subject_ip: float = 0.0, regen: bool = False) -> dict:
    k = {"positive": the_plan["positive"], "negative_add": list(the_plan.get("negative_add") or []), "stage": "edit"}
    if kind == "style":
        # No character IP-Adapter: fed the base image it carries the old style too (three rounds
        # up to denoise 0.81 stayed in the old style, 2026-10-04). img2img + edges keep the character.
        # The style image's adapter also carries what it shows (an Arcane portrait's bare skin
        # turned armour into a bodysuit at 0.6, 2026-10-04): lighter when a style LoRA brings the look.
        k.update(denoise=0.65, edge=0.45, edge_end=0.6, pose=0.5 if has_skeleton else 0.0, subject_ip=0.0,
                 style_ip=(0.35 if has_style_lora else 0.55) if has_style_image else 0.0)
    elif kind == "pose":
        # From the character's own image (img2img), not from noise: the pose ControlNet moves the
        # limbs while the armour's shapes and colours carry over where the body stays. From noise
        # with only an IP-Adapter, plate armour came back as a bodysuit in every round (2026-10-04).
        # denoise is the knob between following the pose (up) and keeping the outfit (down).
        # 0.8 is as low as it goes: at 0.7 the arms stayed at her sides (2026-10-05).
        k.update(denoise=0.8 if has_skeleton else 0.75, pose=0.85 if has_skeleton else 0.0, pose_end=0.9,
                 subject_ip=subject_ip, refine_denoise=0.45)
    else:
        local = bool(the_plan.get("region"))
        # Repainting a region at 0.7 kept the old shape (a bun stayed a bun for two rounds,
        # 2026-10-04): the mask already keeps the rest, so the region starts nearly free.
        k.update(local=local, denoise=0.85 if local else 0.5, grow=24, edge=0.0 if local else 0.45,
                 subject_ip=0.0 if local else subject_ip)
    if regen and the_plan.get("regen_positive"):
        # First from the image's own noise with the prompt edited (see regen_source); the
        # settings above are the fallback when that can't make the change.
        k.update(stage="regen", regen_positive=the_plan["regen_positive"], regen_weight=1.0, regen_misses=0,
                 regen_change=list(the_plan.get("change_tags") or []))
    return k


def prompt_tags_matching(prompt: str, phrases: list[str]) -> list[str]:
    """The prompt's tags that hold every word of one of `phrases`: the judge, which no longer
    sees the prompt, says "sword" and the prompt has "(green sword on back:1.3)"."""
    out = []
    for phrase in phrases:
        words = set(norm_tag(phrase).split())
        if words:
            out += [t for t in split_tags(prompt) if words <= set(norm_tag(t).split())]
    return list(dict.fromkeys(out))


def _step(value: float, delta: float, lo: float, hi: float) -> float:
    return round(max(lo, min(hi, value + delta)), 2)


def in_base(tag: str, base_text: str) -> bool:
    """Every content word of `tag` (longer than 3 letters, plural "s" ignored) is in the base
    image's description."""
    text = {w.rstrip("s") for w in re.findall(r"[a-z]+", base_text.lower())}
    words = [w.rstrip("s") for w in re.findall(r"[a-z]+", norm_tag(tag)) if len(w) > 3]
    return bool(words) and all(w in text for w in words)


def adjust(kind: str, knobs: dict, verdict: dict, protected: list[str], backdrop: str = "",
           base_text: str = "") -> tuple[dict, list[str]]:
    """The next round's knobs from the round's best verdict: the change not made -> more
    freedom toward it; the rest drifting -> hold it closer. Its prompt tips go into the prompt
    (never removing the change's own tags). Returns (knobs, notes)."""
    k, notes = dict(knobs), []
    change, kept = float(verdict.get("change") or 0), float(verdict.get("kept") or 0)
    ident = min(float(verdict.get("identity") or 0), float(verdict.get("outfit") or 0))
    if k.get("stage") == "regen":
        # The best of the round's variants sets how strongly the change's tags are weighted.
        if verdict.get("_regen_weight"):
            k["regen_weight"] = float(verdict["_regen_weight"])
        if change < 6:
            k["regen_misses"] = int(k.get("regen_misses") or 0) + 1
            k["regen_weight"] = _step(float(k.get("regen_weight") or 1.0), 0.2, 1.0, 1.6)
            if k["regen_misses"] >= 2:
                k["stage"] = "edit"
                notes.append("two rounds from the image's own noise didn't make the change: from the image instead")
            else:
                notes.append("the change isn't there from the image's own noise yet: its tags stronger")
        elif kept < 8:
            k["regen_weight"] = _step(float(k.get("regen_weight") or 1.0), -0.1, 1.0, 1.6)
            notes.append("drifting from the character: the change's tags lighter")
    elif kind == "style":
        if change < 7:
            k["denoise"] = _step(k["denoise"], 0.08, 0.35, 0.9)
            k["edge"] = _step(k["edge"], -0.1, 0.15, 0.9)
            if k.get("style_ip"):
                k["style_ip"] = _step(k["style_ip"], 0.1, 0.0, 0.9)
            notes.append("style not there yet: more denoise, looser edges")
        elif kept < 8:
            k["edge"] = _step(k["edge"], 0.1, 0.2, 0.9)
            k["denoise"] = _step(k["denoise"], -0.05, 0.35, 0.9)
            if ident < 8:
                k["edge_end"] = _step(k["edge_end"], 0.1, 0.4, 1.0)
                if k.get("style_ip"):
                    k["style_ip"] = _step(k["style_ip"], -0.1, 0.2, 0.9)
            notes.append("drifting from the base: tighter edges, less denoise"
                         + (", a lighter style image" if ident < 8 and k.get("style_ip") else ""))
        elif change < 9:  # nearly there and the rest well kept: a small step on (a round with
            # nothing to fix moved nothing and scored the same, 2026-10-04)
            k["denoise"] = _step(k["denoise"], 0.04, 0.35, 0.9)
            if k.get("style_ip"):
                k["style_ip"] = _step(k["style_ip"], 0.05, 0.0, 0.9)
            notes.append("style nearly there: a little more denoise")
    elif kind == "pose":
        # With the skeleton measured as matched, a low change is the head or gaze, which more
        # denoise and ControlNet don't fix: gemma-4-26B held "change 6.7" (gaze) on candidates
        # measured 8.4-9.3, the knobs climbed to denoise 0.9 and ControlNet 1.0, and the face
        # and armour drifted every round (2026-10-05).
        skeleton_ok = verdict.get("pose_match") is not None and float(verdict["pose_match"]) >= 8
        if change < 7 and skeleton_ok:
            notes.append(f"the skeleton matches (measured {float(verdict['pose_match']):g}): the rest of the "
                         "change is left to the prompt")
        if change < 7 and not skeleton_ok:
            if k.get("pose"):
                k["pose"] = _step(k["pose"], 0.1, 0.0, 1.0)
                k["pose_end"] = _step(k["pose_end"], 0.1, 0.5, 1.0)
            k["denoise"] = _step(k.get("denoise", 0.8), 0.05, 0.6, 1.0)
            notes.append("pose not matched: stronger pose ControlNet, more denoise")
        elif ident < 8:  # not below 0.78: at 0.75 the arms stayed down in 5 of 6 (2026-10-05)
            k["denoise"] = _step(k.get("denoise", 0.8), -0.03, 0.78, 1.0)
        elif change < 9 and k.get("pose") and not skeleton_ok:
            k["pose"] = _step(k["pose"], 0.05, 0.0, 1.0)
            notes.append("pose nearly matched: a little stronger pose ControlNet")
        if ident < 8:
            if k.get("subject_ip"):  # only when one is set (designer.subject_ip): it costs the style
                k["subject_ip"] = _step(k["subject_ip"], 0.05, 0.0, 0.8)
            notes.append("character drifting: " + ("a stronger character IP-Adapter, " if k.get("subject_ip") else "")
                         + ("less denoise" if change >= 7 or skeleton_ok else "holding denoise"))
        # Refining polishes the best candidate, mistakes and all: three rounds refined one with
        # a sword and a grey backdrop and kept both (2026-10-05). So only a candidate that has
        # the pose AND keeps the rest is refined; otherwise fresh renders from the base image.
        if change >= 8 and kept >= 8.5 and k.get("stage") == "edit":
            k["stage"] = "refine"
            notes.append("pose right and the rest kept: refining the best candidate")
        elif k.get("stage") == "refine" and (change < 7 or kept < 8):
            k["stage"] = "edit"
            notes.append("refining isn't holding: back to fresh renders from the base image")
    else:
        if change < 7:
            k["denoise"] = _step(k["denoise"], 0.1, 0.35, 0.95)
            if k.get("local"):
                k["grow"] = min(48, int(k.get("grow", 16)) + 8)
            else:
                k["edge"] = _step(k.get("edge", 0.45), -0.1, 0.0, 0.8)
            notes.append("change not made yet: more freedom where it is")
        elif kept < 8:
            k["denoise"] = _step(k["denoise"], -0.05, 0.35, 0.95)
            if not k.get("local"):
                k["edge"] = _step(k.get("edge", 0.45), 0.1, 0.0, 0.8)
            notes.append("drifting from the base: less denoise")
        elif change < 9:
            k["denoise"] = _step(k["denoise"], 0.04, 0.35, 0.95)
            notes.append("change nearly made: a little more denoise")
    keep_tags = {norm_tag(t) for t in protected}
    removes = [t for t in prompt_tags_matching(k["positive"], verdict.get("prompt_remove") or [])
               if norm_tag(t) not in keep_tags]
    # Only tips that bring back what the base image has: Qwen3-VL-30B's "+black skirt" and
    # "+black cape with green lining" described the candidate and pushed it further off (2026-10-05).
    adds = [a for a in verdict.get("prompt_add") or [] if in_base(a, base_text)][:4] if base_text else \
        list(verdict.get("prompt_add") or [])[:4]
    # "+no gloves" draws gloves: a prompt can't say no.
    adds = [a for a in adds if not re.match(r"(?i)\s*\(?(no|without|not|remove)\b", a)]
    if adds or removes:
        k["positive"] = edit_prompt(k["positive"], adds, removes)
        if k.get("regen_positive"):
            k["regen_positive"] = edit_prompt(k["regen_positive"], adds, removes)
        notes.append("prompt: " + ", ".join([f"+{a}" for a in adds] + [f"-{r}" for r in removes]))
    neg_extra = list(verdict.get("negative_add") or [])[:3]
    if kind != "style" and float(verdict.get("backdrop_shift") or 0) > 8 and backdrop:
        # the backdrop moved (measured): name the base image's, weighted, and the new one as negative
        # No more than 1.2: at 1.4-1.5 "(light beige background)" bled into the whole image, the
        # brown hair turned blonde and the armour beige, and the backdrop moved no less (2026-10-05).
        k["backdrop_weight"] = round(min(1.2, float(k.get("backdrop_weight", 1.1)) + 0.1), 2)
        tag = f"{backdrop} background"
        k["positive"] = edit_prompt(k["positive"], [], prompt_tags_matching(k["positive"], [tag]))
        k["positive"] = edit_prompt(k["positive"], [f"({tag}:{k['backdrop_weight']:g})"], [])
        seen_bg = str((verdict.get("blind") or {}).get("background") or "").strip()
        if seen_bg:
            neg_extra.append(f"{seen_bg} background")
        notes.append(f"backdrop moved: ({tag}:{k['backdrop_weight']:g}) in the prompt")
    k["negative_add"] = list(dict.fromkeys(k.get("negative_add", []) + neg_extra))[:12]
    return k, notes


# ---- one round of renders --------------------------------------------------------------------

def _subject_image(base: Path, *, comfy, cfg, root: Path, upload) -> Path:
    """The base image for the character IP-Adapter: background removed when possible, padded
    to a square (its encoder centre-crops; see nobg.square_for_ip)."""
    from . import nobg
    img = base
    try:
        img = nobg.cutout(base, comfy=comfy, cfg=cfg, cache_dir=root / "cache" / "cutouts", upload=upload)
    except Exception:
        pass  # no background remover: the image as it is
    return nobg.square_for_ip(img, root / "cache" / "cutouts")


def render_round(kind: str, knobs: dict, *, sdir: Path, rnd: int, base_input: Path, best: Path | None,
                 skeleton: Path | None, style_image: Path | None, subject_ip_image: Path | None, region: str,
                 protect: str,
                 params: GenParams, size: tuple[int, int], batch: int, seed: int | None, comfy, flows, cfg: dict,
                 checkpoint: str | None, upload, should_stop, regen: dict | None = None) -> tuple[list[Path], dict]:
    """Render `batch` candidates; returns their paths and the settings used (with "per": each
    candidate's own settings, when they differ)."""
    if knobs.get("stage") == "regen" and regen:
        return _render_regen(kind, knobs, sdir=sdir, rnd=rnd, base_input=base_input, skeleton=skeleton,
                             params=params, size=size, batch=batch, comfy=comfy, flows=flows, cfg=cfg,
                             checkpoint=checkpoint, upload=upload, should_stop=should_stop, regen=regen)
    from . import masks
    from .targets import cap_ip_weights, max_combined, subject_preset
    cn = cfg.get("controlnet", {})
    model = cn.get("model", "xinsir_union_sdxl_promax.safetensors")
    ip_cfg = cfg.get("ipadapter", {})
    subject = subject_preset(ip_cfg)
    control, ipa = [], []
    neg = edit_prompt(params.negative, knobs.get("negative_add") or [], [])
    seed = seed if seed is not None else random.randrange(2**32)
    p = replace(params, positive=knobs["positive"], negative=neg, seed=seed, mask_target=None)
    image_name = upload(base_input)
    if kind == "style":
        p = replace(p, mode="img2img_reference", denoise=knobs["denoise"])
        control.append({"model": model, "image": image_name, "type": "canny/lineart/anime_lineart/mlsd",
                        "preprocess": "canny", "strength": knobs["edge"], "start": 0.0, "end": knobs["edge_end"]})
        if skeleton is not None and knobs.get("pose"):
            control.append({"model": model, "image": upload(skeleton), "type": "openpose",
                            "strength": knobs["pose"], "start": 0.0, "end": 0.8})
        if subject_ip_image is not None and knobs.get("subject_ip"):
            ipa.append({"image": upload(subject_ip_image), "role": "subject", "preset": subject,
                        "weight": knobs["subject_ip"], "weight_type": "linear", "start": 0.0, "end": 1.0})
        if style_image is not None and knobs.get("style_ip"):
            ipa.append({"image": upload(style_image), "role": "style",
                        "preset": ip_cfg.get("preset", "PLUS (high strength)"), "weight": knobs["style_ip"],
                        "weight_type": ip_cfg.get("style_weight_type", "style transfer"), "start": 0.0, "end": 1.0})
    elif kind == "pose":
        if knobs.get("stage") == "refine" and best is not None:
            refine_src = sdir / f"r{rnd}_source.png"
            fit_to(best, size, refine_src)
            image_name = upload(refine_src)
            p = replace(p, mode="img2img_reference", denoise=knobs["refine_denoise"])
        elif knobs.get("denoise", 1.0) < 0.99:
            p = replace(p, mode="img2img_reference", denoise=knobs["denoise"])
        else:
            p = replace(p, mode="txt2img", denoise=1.0)
        if skeleton is not None and knobs.get("pose"):
            control.append({"model": model, "image": upload(skeleton), "type": "openpose",
                            "strength": knobs["pose"], "start": 0.0, "end": knobs["pose_end"]})
        if subject_ip_image is not None:
            ipa.append({"image": upload(subject_ip_image), "role": "subject", "preset": subject,
                        "weight": knobs["subject_ip"], "weight_type": "linear", "start": 0.0, "end": 1.0})
    else:
        if knobs.get("local") and region:
            masked = masks.masked_image(comfy, base_input, image_name, region, sdir / f"r{rnd}_mask.png",
                                        grow_px=int(knobs.get("grow", 16)), exclude=protect)
            image_name = upload(masked)
            p = replace(p, mode="inpaint_reference", denoise=max(0.5, knobs["denoise"]), mask_target=region)
        else:
            p = replace(p, mode="img2img_reference", denoise=knobs["denoise"])
            if knobs.get("edge"):
                control.append({"model": model, "image": image_name, "type": "canny/lineart/anime_lineart/mlsd",
                                "preprocess": "canny", "strength": knobs["edge"], "start": 0.0, "end": 0.5})
            if subject_ip_image is not None and knobs.get("subject_ip"):
                ipa.append({"image": upload(subject_ip_image), "role": "subject",
                            "preset": subject, "weight": knobs["subject_ip"],
                            "weight_type": "linear", "start": 0.0, "end": 1.0})
    cap_ip_weights(ipa, max_combined(ip_cfg))  # two adapters on one image wash it out
    graph = flows.build(p, image_name, batch, checkpoint, p.positive, size, control=control or None,
                        ipadapter=ipa or None)
    (sdir / f"r{rnd}_graph.json").write_text(json.dumps(graph), encoding="utf-8")
    pid = comfy.queue(graph)
    data = comfy.fetch_images(comfy.wait([pid], should_stop=should_stop)[pid], flows.output_node)
    out = []
    for i, blob in enumerate(data[:batch], 1):
        path = sdir / f"r{rnd}_{i}.png"
        path.write_bytes(blob)
        out.append(path)
    used = {"mode": p.mode, "denoise": p.denoise, "seed": seed, "steps": p.steps, "cfg": p.cfg,
            "sampler_name": p.sampler_name, "scheduler": p.scheduler, "loras": [list(x) for x in p.loras],
            "positive": p.positive, "negative": p.negative, "checkpoint": checkpoint,
            "control": [{k: v for k, v in c.items() if k != "image"} for c in control],
            "ipadapter": [{k: v for k, v in a.items() if k != "image"} for a in ipa], "size": list(size),
            "from_noise": False}  # an edit of an image: its seed alone doesn't draw it again
    return out, used


def weight_tags(prompt: str, tags: list[str], weight: float) -> str:
    """`prompt` with each of `tags` weighted (tag:weight); 1.0 leaves the prompt as it is."""
    if abs(weight - 1.0) < 1e-6 or not tags:
        return prompt
    want = {norm_tag(t) for t in tags}
    return "\n".join(", ".join(f"({norm_tag(t)}:{weight:g})" if norm_tag(t) in want else t
                               for t in (x.strip() for x in line.split(",")) if t) if line.strip() else line
                     for line in prompt.split("\n"))


def _render_regen(kind: str, knobs: dict, *, sdir: Path, rnd: int, base_input: Path, skeleton: Path | None,
                  params: GenParams, size: tuple[int, int], batch: int, comfy, flows, cfg: dict,
                  checkpoint: str | None, upload, should_stop, regen: dict) -> tuple[list[Path], dict]:
    """The candidates drawn again from the base image's own noise (its seed and its place in
    its batch, at its size), the original prompt edited: the change's tags at the round's
    weight, a stronger one, and for a pose with a skeleton the pose ControlNet as well (a
    prompt alone may not move the arms), each rendered alone."""
    w = float(knobs.get("regen_weight") or 1.0)
    variants = [{"regen_weight": w}, {"regen_weight": round(min(1.8, w + 0.2), 2)}]
    variants.append({"regen_weight": w, "regen_pose": 0.6} if kind == "pose" and skeleton is not None
                    else {"regen_weight": round(min(1.8, w + 0.4), 2)})
    while len(variants) < batch:
        variants.append({"regen_weight": round(min(1.8, w + 0.2 * len(variants)), 2)})
    variants = variants[:max(1, batch)]
    out_size = tuple(regen["size"]) if regen.get("size") else size
    model = cfg.get("controlnet", {}).get("model", "xinsir_union_sdxl_promax.safetensors")
    neg = edit_prompt(params.negative, knobs.get("negative_add") or [], [])
    image_name = upload(base_input)  # unused from noise, but the workflow's image input wants one
    out, per = [], []
    for i, v in enumerate(variants, 1):
        if should_stop():
            break
        positive = weight_tags(knobs["regen_positive"], knobs.get("regen_change") or [], v["regen_weight"])
        p = replace(params, positive=positive, negative=neg, seed=regen["seed"], mode="txt2img", denoise=1.0,
                    mask_target=None)
        control = [{"model": model, "image": upload(skeleton), "type": "openpose", "strength": v["regen_pose"],
                    "start": 0.0, "end": 0.8}] if v.get("regen_pose") else None
        graph = flows.build(p, image_name, regen["of"], checkpoint, positive, out_size, control=control,
                            pick=regen["index"])
        (sdir / f"r{rnd}_{i}_graph.json").write_text(json.dumps(graph), encoding="utf-8")
        pid = comfy.queue(graph)
        data = comfy.fetch_images(comfy.wait([pid], should_stop=should_stop)[pid], flows.output_node)
        if not data:
            continue
        path = sdir / f"r{rnd}_{i}.png"
        path.write_bytes(data[0])
        out.append(path)
        per.append({**v, "positive": positive, "control": [{k: x for k, x in c.items() if k != "image"}
                                                            for c in control or []],
                    # what a kept version needs to be drawn again the same way
                    "from_noise": not control, "batch_index": regen["index"], "batch_of": regen["of"]})
    used = {"mode": "txt2img", "denoise": 1.0, "seed": regen["seed"], "steps": params.steps, "cfg": params.cfg,
            "sampler_name": params.sampler_name, "scheduler": params.scheduler,
            "loras": [list(x) for x in params.loras], "negative": neg, "checkpoint": checkpoint,
            "ipadapter": [], "size": list(out_size), "stage": "regen", "per": per}
    return out, used


def _default(flows, role: str) -> str:
    try:
        return str(flows.default(role) or "")
    except (KeyError, AttributeError):
        return ""


# ---- the loop ---------------------------------------------------------------------------------

def edit_loras(kind: str, loras, the_plan: dict, types: dict, *, change: dict, target_image: Path | None,
               target_tags: str, cards, checkpoint: str | None, backend, max_side: int, sess: dict, log) -> list:
    """The LoRAs an edit renders with: the base image's, less those the plan drops. A style
    edit always drops the old style's LoRAs (a style LoRA at 0.9 held the old look through
    three rounds at denoise up to 0.74, 2026-10-04) and asks the LoRA chooser for one in the
    new style, when one in the library fits."""
    drop = {d.lower() for d in the_plan.get("drop_loras") or []} if kind == "style" else set()
    if kind == "style":
        drop |= {lora_stem(n).lower() for n, _ in loras if types.get(n) == "style"}
    kept = [[n, w] for n, w in loras if lora_stem(n).lower() not in drop]
    if len(kept) < len(loras):
        log("dropped " + ", ".join(lora_stem(n) for n, _ in loras if lora_stem(n).lower() in drop))
    if kind != "style" or cards is None:
        return kept
    try:
        from .loras import checkpoint_base, compatible
        from .lora_picker import suggest_loras
        base = checkpoint_base(checkpoint) if checkpoint else None
        menu = [c for c in cards() if c.get("type") == "style"
                and (base is None or compatible(c.get("base_model"), base) is not False)]
        wanted = (change.get("text") or "").strip() or target_tags or "the style of the style image"
        out = suggest_loras(backend, menu, the_plan["positive"], "", f"ONLY a new drawing style: {wanted}. "
                            "Pick a style LoRA only if it clearly gives this look; none is better than a different look.",
                            None, 1, max_side, references=[("style", target_image, target_tags)] if target_image else None)
        sess["cost_usd"] = sess.get("cost_usd", 0.0) + float(out.get("cost_usd") or 0)
        for pick in out.get("picks") or []:
            kept.append([pick["comfy_name"], pick["strength"]])
            sess["style_lora"] = pick["comfy_name"]
            trig = pick.get("triggers") or []
            sess["lora_triggers"] = list(trig[:1]) if isinstance(trig, list) else []
            log(f"style LoRA: {lora_stem(pick['comfy_name'])} at {pick['strength']:g}"
                + (f" ({pick['why']})" if pick.get("why") else ""))
        if not out.get("picks"):
            log("no style LoRA in the library fits; the style comes from the prompt"
                + (" and the style image" if target_image else ""))
    except Exception as e:  # no catalog, or the call failed: the prompt (and image) carry it
        log(f"warning: no style LoRA chosen ({type(e).__name__}: {str(e)[:120]})")
    return kept


def confirm_backend_for(judge_cfg: dict):
    """The judge.confirm_model backend (the second opinion on a pass), or None when unset or
    the same as the judge's model; see judge.Judge."""
    from .backends import make_backend
    bkey = judge_cfg.get("backend", "ollama")
    name = (judge_cfg.get("confirm_model") or "").strip()
    if not name or (name == judge_cfg.get(bkey, {}).get("model") and not judge_cfg.get("confirm_options")):
        return None
    return make_backend(judge_cfg, "confirm")


from . import usage


# ---- manual edits: your own prompt and settings, no judge ----------------------------------------

MANUAL_MODES = ("noise", "same_noise", "img2img")


def made_settings(root: Path, design_id: str, image: str) -> dict:
    """The settings that made one of a character's images (its own over the character's)."""
    design = load(root, design_id)
    entry = next((e for e in design.get("catalog", []) if e["image"] == image), None)
    return {**(design.get("params") or {}), **((entry or {}).get("params") or {})}


def new_manual_session(root: Path, design_id: str, *, base: str, request: dict, style_image: Path | None = None,
                       pose_image: Path | None = None) -> dict:
    """A manual edit: renders with the prompts and settings given (request: positive,
    negative, steps, cfg, sampler_name, scheduler, seed, random_seed, batch_size, size,
    checkpoint, loras [[name, strength]], mode (MANUAL_MODES), denoise; and as on Generate:
    subject_ip / subject_ip_weight (the character IP-Adapter, from the image you start from),
    style_ip / style_ip_weight with style_library or an uploaded style_image, pose_control with
    pose_library, an uploaded pose_image or pose_own (the image's own pose), pose_strength,
    pose_end), nothing judged."""
    file_of(root, design_id, base)
    q = dict(request)
    if not str(q.get("positive") or "").strip():
        raise ValueError("write a positive prompt")
    q["mode"] = q.get("mode") if q.get("mode") in MANUAL_MODES else "noise"
    if q["mode"] == "same_noise" and not regen_source(made_settings(root, design_id, base)):
        raise ValueError("this image wasn't drawn from noise alone (or how isn't known): "
                         "use a new seed or start from the image")
    sid = time.strftime("%Y%m%d-%H%M%S") + "_" + uuid.uuid4().hex[:4]
    d = design_dir(root, design_id) / "sessions" / sid
    d.mkdir(parents=True)
    for img, name, key in ((style_image, "style.png", "style_image"), (pose_image, "pose.png", "pose_image")):
        q.pop(key, None)
        if img is not None:
            with Image.open(img) as im:
                im.convert("RGB").save(d / name)
            q[key] = name
    if q.get("style_ip") and not (q.get("style_library") or q.get("style_image")):
        q["style_ip"] = False  # nothing to take the style from
    sess = {"id": sid, "design": design_id, "base": base, "kind": "manual",
            "change": {"text": _manual_summary(q), "style": "", "pose": "", "image": ""},
            "manual": q, "threshold": None, "max_rounds": None, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "queued", "error": None, "plan": None, "knobs": None, "rounds": [], "best": None,
            "confirmed": False, "cost_usd": 0.0, "log": []}
    _write(d / "session.json", sess)
    with _lock:
        design = load(root, design_id)
        design.setdefault("sessions", []).append(sid)
        save(root, design)
    return sess


def _manual_summary(q: dict) -> str:
    how = {"noise": "new noise", "same_noise": "its own noise", "img2img": f"from the image at {q.get('denoise')}"}
    return " · ".join(x for x in (how.get(q["mode"], q["mode"]), (q.get("note") or "").strip()[:80]) if x)


def _manual_guides(q: dict, *, sdir: Path, base: Path, size: tuple[int, int], cfg: dict, root: Path, comfy,
                   upload, stage) -> tuple[list, list]:
    """The pose ControlNet and IP-Adapters a manual edit asked for, as on Generate."""
    from . import pose as posemod
    from .targets import cap_ip_weights, max_combined, subject_preset
    control, ipa = [], []
    cn, ip_cfg = cfg.get("controlnet", {}), cfg.get("ipadapter", {})
    if q.get("pose_control") and posemod.available():
        stage("reading the pose")
        found, src_size = None, size
        if q.get("pose_library"):
            found = posemod.PoseLibrary(root / "poses").get(q["pose_library"])
            src_size = tuple(found.get("size") or size)
        elif q.get("pose_image") or q.get("pose_own"):
            src = sdir / q["pose_image"] if q.get("pose_image") else base
            found = posemod.detect(src)
            with Image.open(src) as im:
                src_size = im.size
        if found and posemod.has_body(found):
            drawn = None
            if not q.get("pose_own"):  # a new pose, framed the way the character's image is
                base_pose = posemod.detect(base)
                with Image.open(base) as im:
                    base_size = im.size
                drawn = posemod.match_framing(found, src_size, base_pose, base_size, size) if base_pose else None
            skeleton = sdir / "pose_control.png"
            posemod.render(drawn or posemod.fit(found, src_size, size), size, hands=cn.get("pose_hands", True),
                           face=cn.get("pose_face", False)).save(skeleton)
            control.append({"model": cn.get("model", "xinsir_union_sdxl_promax.safetensors"), "image": upload(skeleton),
                            "type": "openpose", "strength": float(q.get("pose_strength") or cn.get("pose_strength", 0.6)),
                            "start": 0.0, "end": float(q.get("pose_end") or cn.get("pose_end", 0.8))})
    if q.get("subject_ip") and float(q.get("subject_ip_weight") or 0) > 0:
        stage("preparing the character image")
        ipa.append({"image": upload(_subject_image(base, comfy=comfy, cfg=cfg, root=root, upload=upload)),
                    "role": "subject", "preset": subject_preset(ip_cfg), "weight": float(q["subject_ip_weight"]),
                    "weight_type": ip_cfg.get("weight_type", "linear"), "start": 0.0, "end": 1.0})
    if q.get("style_ip") and float(q.get("style_ip_weight") or 0) > 0:
        style = sdir / q["style_image"] if q.get("style_image") else None
        if style is None and q.get("style_library"):
            from .reflib import RefLibrary
            style = Path(RefLibrary(root, "style").get(q["style_library"])["source"])
        if style is not None:
            ipa.append({"image": upload(style), "role": "style", "preset": ip_cfg.get("preset", "PLUS (high strength)"),
                        "weight": float(q["style_ip_weight"]),
                        "weight_type": ip_cfg.get("style_weight_type", "style transfer"), "start": 0.0, "end": 1.0})
    cap_ip_weights(ipa, max_combined(ip_cfg))  # two adapters on one image wash it out
    return control, ipa


def run_manual(root: Path, design_id: str, session_id: str, *, cfg: dict, comfy, flows, upload,
               should_stop=lambda: False, stage=lambda text: None) -> dict:
    """One round of a manual edit with its settings (a random seed is new each round)."""
    sess = load_session(root, design_id, session_id)
    sdir = session_dir(root, design_id, session_id)
    q, base = sess["manual"], file_of(root, design_id, sess["base"])
    sess.update(status="running", error=None)
    save_session(root, design_id, sess)
    try:
        made = made_settings(root, design_id, sess["base"])
        with Image.open(base) as im:
            native = im.size
        size = tuple(int(x) for x in q["size"]) if q.get("size") else closest_size(*native)
        batch = max(1, min(8, int(q.get("batch_size") or 1)))
        regen = regen_source(made) if q["mode"] == "same_noise" else None
        seed = regen["seed"] if regen else (random.randrange(2**31) if q.get("random_seed") else int(q.get("seed") or 0))
        p = GenParams(mode="img2img_reference" if q["mode"] == "img2img" else "txt2img",
                      positive=str(q["positive"]), negative=str(q.get("negative") or ""), seed=seed,
                      steps=int(q.get("steps") or 30), cfg=float(q.get("cfg") or 5.5),
                      sampler_name=q.get("sampler_name") or "dpmpp_2m", scheduler=q.get("scheduler") or "karras",
                      denoise=float(q.get("denoise") or 0.6) if q["mode"] == "img2img" else 1.0,
                      loras=tuple((str(n), float(w)) for n, w in q.get("loras") or []))
        if regen:
            size, batch = tuple(regen["size"]) if regen.get("size") else size, 1
        base_input = fit_to(base, size, sdir / "base_input.png")
        rnd = len(sess["rounds"]) + 1
        control, ipa = _manual_guides(q, sdir=sdir, base=base, size=size, cfg=cfg, root=root, comfy=comfy,
                                      upload=upload, stage=stage)
        stage(f"rendering {batch}")
        graph = flows.build(p, upload(base_input), regen["of"] if regen else batch, q.get("checkpoint") or None,
                            p.positive, size, control=control or None, ipadapter=ipa or None,
                            pick=regen["index"] if regen else None)
        (sdir / f"r{rnd}_graph.json").write_text(json.dumps(graph), encoding="utf-8")
        pid = comfy.queue(graph)
        blobs = comfy.fetch_images(comfy.wait([pid], should_stop=should_stop)[pid], flows.output_node)
        cands = []
        for i, blob in enumerate(blobs[:batch], 1):
            path = sdir / f"r{rnd}_{i}.png"
            path.write_bytes(blob)
            # What a kept one needs to be drawn again: from noise, image i of this batch.
            cands.append({"image": path.name, "score": None, "scores": {}, "differences": [], "missing": "",
                          "params": {"mode": p.mode, "denoise": p.denoise, "seed": seed, "steps": p.steps,
                                     "cfg": p.cfg, "sampler_name": p.sampler_name, "scheduler": p.scheduler,
                                     "loras": [list(x) for x in p.loras], "positive": p.positive,
                                     "negative": p.negative, "checkpoint": q.get("checkpoint") or None,
                                     "size": list(size),
                                     # a ControlNet or IP-Adapter is part of what drew it: not redrawable from noise
                                     "from_noise": p.mode == "txt2img" and not control and not ipa,
                                     "control": [{k: v for k, v in c.items() if k != "image"} for c in control],
                                     "ipadapter": [{k: v for k, v in a.items() if k != "image"} for a in ipa],
                                     "batch_index": regen["index"] if regen else i - 1,
                                     "batch_of": regen["of"] if regen else batch}})
        if not cands:
            raise RuntimeError("the render returned no images")
        sess["rounds"].append({"round": rnd, "settings": {k: v for k, v in q.items() if k not in ("positive", "negative")}
                               | {"seed": seed}, "candidates": cands, "best": 0, "notes": []})
        sess["best"] = sess.get("best") or {"round": rnd, "image": cands[0]["image"], "score": None}
        sess["status"] = "finished"
        sess["log"] = (sess.get("log") or [])[-60:] + [f"{time.strftime('%H:%M:%S')} round {rnd}: {len(cands)} "
                                                       f"rendered, seed {seed}"]
        save_session(root, design_id, sess)
        return sess
    except Cancelled:
        sess["status"] = "stopped"
        save_session(root, design_id, sess)
        raise
    except Exception as e:
        sess.update(status="error", error=f"{type(e).__name__}: {str(e)[:300]}")
        save_session(root, design_id, sess)
        raise


@usage.scoped
def run_session(root: Path, design_id: str, session_id: str, *, cfg: dict, comfy, flows, backend, upload,
                rounds: int | None = None, should_stop=lambda: False, stage=lambda text: None,
                lora_cards=None, confirm_backend=None, prompt_backend=None) -> dict:
    """Run an edit for up to `rounds` rounds (default: its max_rounds, less those done), until
    a candidate passes the threshold and a second look agrees. Saves session.json after
    every step, so the page can follow it."""
    from . import pose as posemod
    root = Path(root)
    st = settings(cfg)
    side = int(cfg.get("judge", {}).get("image_max_side", 768))
    design = load(root, design_id)
    sess = load_session(root, design_id, session_id)
    sdir = session_dir(root, design_id, session_id)
    threshold = float(sess.get("threshold") or st["threshold"])
    from .model_profiles import confirmation_threshold
    jc = cfg.get("judge", {})
    bk = jc.get("backend", "ollama")
    judge_model = jc.get(bk, {}).get("model", "unknown")
    confirm_model = jc.get("confirm_model") or judge_model
    confirm_threshold = confirmation_threshold(jc, threshold, "designer")
    prompt_backend = prompt_backend or backend
    sess.update(judge_model=judge_model, confirm_model=confirm_model, confirm_threshold=confirm_threshold)
    max_rounds = int(sess.get("max_rounds") or st["max_rounds"])
    batch = max(1, int(st["batch"]))
    base = file_of(root, design_id, sess["base"])
    kind, change = sess["kind"], sess["change"]

    initial_cost = sess.get("cost_usd", 0.0)

    def log(text: str) -> None:
        sess["cost_usd"] = usage.total_or(sess["cost_usd"], initial_cost)
        sess["log"] = (sess.get("log") or [])[-60:] + [f"{time.strftime('%H:%M:%S')} {text}"]
        save_session(root, design_id, sess)

    def check():
        if should_stop():
            raise Cancelled()

    sess.update(status="running", error=None)
    save_session(root, design_id, sess)
    try:
        # What it was made with: the base image's own settings (a kept version's, else the design's).
        entry = next((e for e in design.get("catalog", []) if e["image"] == sess["base"]), None)
        made = {**design.get("params", {}), **((entry or {}).get("params") or {})}
        params = GenParams(positive=made.get("positive", ""),
                           negative=made.get("negative") or _default(flows, "negative"),
                           steps=int(made.get("steps") or 30), cfg=float(made.get("cfg") or 5.5),
                           sampler_name=made.get("sampler_name") or "dpmpp_2m",
                           scheduler=made.get("scheduler") or "karras",
                           loras=tuple((str(n), float(w)) for n, w in made.get("loras") or []))
        checkpoint = made.get("checkpoint") or None
        with Image.open(base) as im:
            size = closest_size(*im.size)
        base_input = fit_to(base, size, sdir / "base_input.png")

        # The new style or pose, when it is a saved one or an image.
        target_image = sdir / change["image"] if change.get("image") else None
        target_tags = ""
        if kind == "style" and change.get("style"):
            from .reflib import RefLibrary
            item = RefLibrary(root, "style").get(change["style"])
            target_image, target_tags = target_image or item["source"], item["description"]
        if kind == "pose" and change.get("pose"):
            lib = posemod.PoseLibrary(root / "poses")
            target_tags = lib.get(change["pose"]).get("description", "")
            target_image = target_image or lib.source(change["pose"])

        lora_types = {}
        if params.loras and sess.get("loras") is None:
            try:
                from .runner import lora_library
                lora_types = {n: r.get("type", "") for n, r in lora_library(cfg).index().items()}
            except Exception:
                pass
        if not sess.get("plan"):
            check()
            loras = [(lora_stem(n), lora_types.get(n, "")) for n, _ in params.loras]
            stage("looking at the image")
            seen, c = describe(backend, base, side)
            sess["cost_usd"] += c
            stage("planning the edit")
            the_plan, c = plan(prompt_backend, base, kind, change, target_image=target_image, target_tags=target_tags, seen=seen,
                               original_positive=params.positive, example_prompt=_default(flows, "positive"),
                               loras=loras, max_side=side)
            sess["cost_usd"] += c
            sess["plan"] = the_plan
            log(f"plan: {the_plan['summary']}"
                + (f" (repainting '{the_plan['region']}'" + (f", not the {the_plan['protect']}" if the_plan.get("protect") else "")
                   + ")" if the_plan.get("region") else ""))
        the_plan = sess["plan"]
        if sess.get("loras") is None:  # decided once per edit
            sess["loras"] = edit_loras(kind, params.loras, the_plan, lora_types, change=change, target_image=target_image,
                                       target_tags=target_tags, cards=lora_cards, checkpoint=checkpoint,
                                       backend=backend, max_side=side, sess=sess, log=log)
        params = replace(params, loras=tuple((n, float(w)) for n, w in sess["loras"]))
        if kind == "style":
            for t in sess.get("lora_triggers") or []:
                if norm_tag(t) not in {norm_tag(x) for x in split_tags(the_plan["positive"])}:
                    the_plan["positive"] = f"{t}, {the_plan['positive']}"

        # Control images, made once.
        skeleton = target_pose = None
        if posemod.available():
            stage("reading the pose")
            src = target_image if kind == "pose" else base
            if kind == "pose" and change.get("pose"):
                found = posemod.PoseLibrary(root / "poses").get(change["pose"])
                src_size = tuple(found.get("size") or size)
            else:
                found = posemod.detect(src) if src is not None else None
                src_size = size
                if src is not None:
                    with Image.open(src) as im:
                        src_size = im.size
            if found and posemod.has_body(found):
                skeleton = sdir / "pose_control.png"
                drawn = None
                if kind == "pose":  # the new pose, framed like the character's image
                    base_pose = posemod.detect(base)
                    with Image.open(base) as im:
                        base_size = im.size
                    drawn = posemod.match_framing(found, src_size, base_pose, base_size, size) if base_pose else None
                    if drawn is None:
                        log("the new pose is drawn as framed in its own image (no neck and hip found in both)")
                drawn = drawn or posemod.fit(found, src_size, size)
                target_pose = drawn if kind == "pose" else None
                posemod.render(drawn, size,
                               hands=cfg.get("controlnet", {}).get("pose_hands", True),
                               face=cfg.get("controlnet", {}).get("pose_face", False)).save(skeleton)
        if kind == "pose" and skeleton is None:
            log("warning: no pose skeleton (" + ("none found in the pose image, or rtmlib missing" if target_image
                is not None else "the pose is in words only") + "): the pose comes from the prompt alone and is much "
                "less exact; a saved pose or a pose image works better")
        subject_ip_image = None
        if kind != "style" and float(st.get("subject_ip") or 0) > 0 and not the_plan.get("region"):
            stage("preparing the character image")
            subject_ip_image = _subject_image(base, comfy=comfy, cfg=cfg, root=root, upload=upload)
        style_image = target_image if kind == "style" else None

        regen = regen_source(made) if the_plan.get("regen_positive") else None
        knobs = sess.get("knobs") or initial_knobs(kind, the_plan, has_skeleton=skeleton is not None,
                                                   has_style_image=style_image is not None,
                                                   has_style_lora=bool(sess.get("style_lora")),
                                                   subject_ip=float(st.get("subject_ip") or 0),
                                                   regen=regen is not None)
        done = len(sess["rounds"])
        todo = rounds if rounds is not None else max(0, max_rounds - done)
        protected = [t for t in split_tags(the_plan["positive"])
                     if any(w in norm_tag(t) for w in " ".join(the_plan.get("must_change") or []).lower().split()
                            if len(w) > 3)] + list(sess.get("lora_triggers") or [])
        if kind != "style":
            # The image's own tags drew it; the judge's tips may add to them but not take them
            # away: gemma-4-26B's "-(silver body armor:1.3), +gold armor" turned the muted plate
            # into shiny gold and then a bikini (2026-10-05).
            protected += [t for t in split_tags(params.positive) if norm_tag(t) in
                          {norm_tag(x) for x in split_tags(the_plan["positive"])}]
        # Nor the tags of the details to keep: Qwen's "-light freckles across nose and cheeks"
        # in round 1 cost every later candidate its freckles (2026-10-05).
        protected += [x["tag"] for x in the_plan.get("must_keep") or [] if isinstance(x, dict) and x.get("tag")]
        sess["status"] = "running"
        for rnd in range(done + 1, done + todo + 1):
            check()
            stage(f"round {rnd}: rendering {batch}")
            best_path = sdir / sess["best"]["image"] if sess.get("best") else None
            seed = int(made["seed"]) if rnd == 1 and kind == "pose" and made.get("seed") is not None else None
            paths, used = render_round(kind, knobs, sdir=sdir, rnd=rnd, base_input=base_input, best=best_path,
                                       skeleton=skeleton, style_image=style_image,
                                       subject_ip_image=subject_ip_image, region=the_plan.get("region", ""),
                                       protect=the_plan.get("protect", ""),
                                       params=params, size=size, batch=batch, seed=seed, comfy=comfy, flows=flows,
                                       cfg=cfg, checkpoint=checkpoint, upload=upload, should_stop=should_stop,
                                       regen=regen)
            cands = []
            for i, path in enumerate(paths, 1):
                check()
                stage(f"round {rnd}: judging {i} of {len(paths)}")
                v, c = judge(backend, base, path, kind, the_plan, change=change, target_image=target_image,
                             target_tags=target_tags, positive=knobs["positive"], max_side=side, work_dir=sdir,
                             target_pose=target_pose)
                sess["cost_usd"] += c
                cands.append({"image": path.name, "score": v["score"], "first_score": v["score"], "judge_model": judge_model, "scores": {k: v.get(k) for k in
                              ("change", *ASPECTS, "quality", "kept", "backdrop_shift", "pose_match")
                              if v.get(k) is not None}, "differences": v["differences"],
                              "missing": v["missing"], "_v": v,
                              "params": {**{k: x for k, x in used.items() if k != "per"}, "batch_index": i - 1,
                                         **((used.get("per") or [])[i - 1] if i <= len(used.get("per") or []) else {})}})
            if not cands:
                raise RuntimeError("the render returned no images")
            bi = max(range(len(cands)), key=lambda i: cands[i]["score"])
            passed = False
            # Each candidate at the threshold gets the second look, best first, until one passes:
            # turned down, the next one up stood at 87.5 as the session's best unchecked (2026-10-05).
            while not passed and cands[bi]["score"] >= threshold and cands[bi].get("recheck") is None:
                check()
                stage(f"round {rnd}: confirming with a second look")
                v2, c = judge(confirm_backend or backend, base, sdir / cands[bi]["image"], kind, the_plan,
                              change=change, target_image=target_image, target_tags=target_tags,
                              positive=knobs["positive"], max_side=side, work_dir=sdir, target_pose=target_pose)
                sess["cost_usd"] += c
                cands[bi]["recheck"] = v2["score"]
                # Confirmation must independently pass its calibrated threshold.
                second = v2["score"]
                passed = second >= confirm_threshold
                cands[bi].update(first_score=cands[bi]["score"], judge_model=judge_model,
                                 confirm_model=confirm_model, confirm_threshold=confirm_threshold,
                                 confirmed=passed)
                if not passed:
                    # Turned down, it scores what the second look gave and the next round is
                    # steered by that verdict: Gemma's 61.6 on a false 85 was logged, and the next
                    # two rounds refined the false pass all the same (2026-10-05).
                    cands[bi]["first_score"], cands[bi]["score"] = cands[bi]["score"], (min(round(second, 1), threshold - 0.1) if confirm_model == judge_model
                                                                       else min(cands[bi]["score"], threshold - 0.1))
                    cands[bi]["_v"] = v2
                    bi = max(range(len(cands)), key=lambda i: cands[i]["score"])
            verdict = cands[bi].pop("_v")
            if cands[bi]["params"].get("regen_weight"):
                verdict["_regen_weight"] = cands[bi]["params"]["regen_weight"]
            for c in cands:
                c.pop("_v", None)
            knobs_before = dict(knobs)
            knobs, notes = adjust(kind, knobs, verdict, protected,
                                  base_text=" ".join(str(v) for v in (the_plan.get("seen") or {}).values()),
                                  backdrop=backdrop_words((the_plan.get("seen") or {}).get("background", "")))
            sess["rounds"].append({"round": rnd, "settings": {k: v for k, v in knobs_before.items()
                                                               if k not in ("positive",)},
                                   "candidates": cands, "best": bi, "notes": notes})
            if not sess.get("best") or cands[bi]["score"] > sess["best"]["score"]:
                sess["best"] = {"round": rnd, "image": cands[bi]["image"], "score": cands[bi]["score"]}
            sess["knobs"] = knobs
            log(f"round {rnd}: best {cands[bi]['score']} (change {verdict.get('change')}, kept {verdict.get('kept')})"
                + (f"; {'; '.join(notes)}" if notes else ""))
            for c in cands:
                if c.get("recheck") is not None:
                    log(f"{'passed' if passed and c is cands[bi] else 'not confirmed'}: {c['image']} scored "
                        f"{c.get('first_score', c['score'])}, a second look {c['recheck']}")
            if passed:
                sess.update(status="passed", confirmed=True,
                            best={"round": rnd, "image": cands[bi]["image"], "score": cands[bi]["score"]})
                break
            save_session(root, design_id, sess)
        if sess["status"] != "passed":
            sess["status"] = "finished"
            log(f"stopped after {len(sess['rounds'])} rounds without passing {threshold:g}; best "
                f"{sess['best']['score'] if sess.get('best') else '-'}")
        sess["cost_usd"] = usage.total_or(sess["cost_usd"], initial_cost)
        save_session(root, design_id, sess)
        return sess
    except Cancelled:
        sess["status"] = "stopped"
        log("stopped")
        raise
    except Exception as e:
        sess["cost_usd"] += float(getattr(e, "cost_usd", 0.0))
        sess.update(status="error", error=f"{type(e).__name__}: {str(e)[:300]}")
        log(f"error: {sess['error']}")
        raise
