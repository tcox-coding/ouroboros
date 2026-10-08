"""Local web UI: python -m ouroboros serve  ->  http://127.0.0.1:8765

Standard library only. The page (static/index.html) polls /api/state once a second.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import shutil
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import requests

from . import generate as generate_mod
from . import autofix, designer, favorites, keys, organize, thumbs
from . import nobg as nobg_mod
from . import upscale as upscale_mod
from .comfy import combo_options
from .backends import deepinfra_models, ollama_models
from .ckpt_info import checkpoint_note, prompt_setup
from .jobs import IMAGE_EXTS, STATUSES, Queue, load_job
from .loras import checkpoint_base, compatible, lora_folders_for
from .params import lora_stem
from .targets import max_combined
from .runner import ROOT, Runner, comfy_launcher, load_config, lora_library, rel_url, save_config

STATIC = ROOT / "static"
FILE_DIRS = ("jobs", "runs", "cache/lora_images", "cache/reruns", "poses", "styles", "characters",
             "designs")  # what /files/ serves
REF_ROUTES = {"/api/styles": "style", "/api/characters": "character"}  # the saved style / character libraries
runner = Runner()
queue = Queue(ROOT / "jobs")


def library_root() -> Path:
    return (ROOT / load_config().get("prompt_library", "../prompt-generator/output")).resolve()


def job_folder(status: str, name: str) -> Path:
    """jobs/<status>/<name>, refusing names that point anywhere else ("..", slashes)."""
    base = (ROOT / "jobs" / status).resolve()
    folder = (base / name).resolve()
    if not name or folder.parent != base:
        raise FileNotFoundError(name)
    return folder


def job_info(folder: Path, status: str) -> dict:
    info = {"name": folder.name, "status": status, "reference": None, "prompt": "", "error": None,
            "running": Queue.is_running(folder)}
    try:
        job = load_job(folder)
        info.update(reference=rel_url(job.reference), prompt=job.positive[:300],
                    description=job.description[:300], settings=job.settings, overrides=job.overrides)
    except Exception as e:  # show broken jobs instead of hiding them
        info["error"] = str(e)
    return info


def library_items() -> list[dict]:
    root = library_root()
    if not root.is_dir():
        return []
    items = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            job = load_job(folder)
        except Exception:
            continue
        items.append({"name": folder.name, "prompt": job.positive[:300],
                      "reference": f"/library/{folder.name}/{job.reference.name}"})
    return items


def library_item(name: str) -> dict:
    """Full prompts and settings of one saved prompt, for viewing and copying."""
    root = library_root()
    folder = (root / name).resolve()
    if folder.parent != root or not folder.is_dir():
        raise FileNotFoundError(name)
    job = load_job(folder)
    loras = folder / "loras.txt"
    return {"name": folder.name, "positive": job.positive, "negative": job.negative,
            "settings": job.settings, "reference": f"/library/{folder.name}/{job.reference.name}",
            "loras": loras.read_text(encoding="utf-8").strip() if loras.exists() else ""}


def selected_lora_notes(cfg: dict, loras) -> str:
    """Prompt-writer notes for LoRAs picked in the UI ([{"name", "strength"}, ...])."""
    from .lora_picker import prompt_notes
    picks = tuple((l["name"], float(l.get("strength", 0.8))) for l in loras or [] if l.get("name"))
    return prompt_notes(lora_library(cfg), picks) if picks else ""


def preview_prompt(data: dict) -> dict:
    """Write a prompt from a description (as a job would at its start) without queueing.
    LoRAs already selected (data["loras"]) are written for: their trigger words go in."""
    from .backends import make_backend
    from .prompter import write_prompt
    from .workflow import Workflows

    cfg = load_config()
    # The page sends its checkpoint ("" = as saved in the workflow); other callers get Settings'.
    ckpt = (data["checkpoint"] if "checkpoint" in data else cfg.get("defaults", {}).get("checkpoint")) or ""
    try:
        flows = Workflows(ROOT / "workflows")
        style_pos, style_neg = flows.default("positive"), flows.default("negative")
        ckpt = ckpt or flows.default("checkpoint")
    except Exception:
        style_pos = style_neg = ""
    import io

    from PIL import Image

    from .targets import Target, Targets, prompt_parts

    def image(b64):
        return Image.open(io.BytesIO(base64.b64decode(b64.split(",", 1)[-1]))) if b64 else None
    reference = None
    if data.get("image_b64") and cfg["loop"].get("prompt_sees_reference", True):
        reference = image(data["image_b64"])
    # Separate style / subject / pose targets ({role: {"image_b64", "text"}}): each part of the
    # prompt from its own target, as a generation would write it (one image and no text is
    # the classic single reference).
    from .reflib import resolve
    raw = data.get("targets") or {}
    for r, t in raw.items():  # a saved style / character stands in for an uploaded image
        lib = (t or {}).get("library") if isinstance(t, dict) else None
        if lib and lib != "auto" and not t.get("image_b64") and (got := resolve(ROOT, r, lib)):
            t["image_b64"] = base64.b64encode(Path(got["image"]).read_bytes()).decode()
            t["text"] = (t.get("text") or "").strip() or got["description"]
    pose_choice = data.get("pose_library") or ""
    pose_t = raw.setdefault("pose", {}) if pose_choice and pose_choice != "auto" else None
    if pose_t is not None and not pose_t.get("image_b64"):  # a saved pose: its photo and description
        try:
            pose_t["image_b64"] = base64.b64encode(pose_library().source(pose_choice).read_bytes()).decode()
            pose_t["text"] = (pose_t.get("text") or "").strip() or pose_library().get(pose_choice).get("description", "")
        except (OSError, ValueError):
            pass
    tg = Targets(**{r: Target(image(t.get("image_b64")), (t.get("text") or "").strip())
                    for r, t in raw.items() if r in ("style", "subject", "pose") and isinstance(t, dict)})
    texts = any(t.text for _, t in tg.items())
    images = [t.image for _, t in tg.items() if t.image is not None]
    split = texts or (tg.style.image is not None and tg.subject.image is not None)
    if not split and images and reference is None and cfg["loop"].get("prompt_sees_reference", True):
        reference = tg.subject.image or tg.style.image or images[0]
    return write_prompt(make_backend(cfg["judge"], "prompt"), data.get("description", ""), data.get("prompt", ""),
                        data.get("negative", ""), style_pos, style_neg, None if split else reference,
                        cfg["judge"].get("image_max_side", 512), selected_lora_notes(cfg, data.get("loras")),
                        targets=prompt_parts(tg) if split else None, **prompt_setup(cfg, ckpt))


def runs_list(limit: int = 60) -> list[dict]:
    """History: loop runs (runs/<run>/summary.json) and Home tab generations
    (runs/manual/<run>/run.json, one entry per generation however large the batch),
    newest first. Both kinds of folder are named <timestamp>_..., so they sort together."""
    found = []
    runs = ROOT / "runs"
    for d in (p for p in runs.iterdir() if p.is_dir() and not p.name.startswith("_") and p.name != "manual"):
        if (d / "summary.json").exists():
            found.append((d.name, "loop", d))
    if (runs / "manual").is_dir():
        for d in (runs / "manual").iterdir():
            if d.is_dir() and ((d / "run.json").exists() or (d / "error.txt").exists() or any(d.glob("image_*.png"))):
                found.append((d.name, "manual", d))
    out = []
    for _, kind, d in sorted(found, key=lambda x: x[0], reverse=True)[:limit]:
        try:
            out.append(_loop_entry(d) if kind == "loop" else _manual_entry(d))
        except (OSError, ValueError):
            continue
    return out


def favorites_list() -> list[dict]:
    """Favourite History entries, newest favourite first: each run's History entry (however
    old it is) with its favourite name ("fav_name") and when it was added."""
    out = []
    for run, fav in sorted(favorites.load(ROOT).items(), key=lambda kv: kv[1].get("added", ""), reverse=True):
        try:
            d = favorites.run_dir(ROOT, run)
            entry = _manual_entry(d) if run.startswith("manual/") else _loop_entry(d)
        except (OSError, ValueError):
            entry = {"run": run, "kind": "missing", "status": "missing"}
        out.append({**entry, "fav_name": fav.get("name", ""), "fav_added": fav.get("added")})
    return out


def _loop_entry(d: Path) -> dict:
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    s["kind"] = "loop"
    s["run"] = d.name
    s["best_url"] = rel_url(d / "best.png") if (d / "best.png").exists() else None
    s["best_thumb"] = thumbs.url(ROOT, d / "best.png") if s["best_url"] else None
    s["autofix"] = autofix_view(d)
    s["upscale"] = upscale_view(d)
    s["nobg"] = nobg_view(d)
    return s


def _manual_entry(d: Path) -> dict:
    try:
        rec = json.loads((d / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        rec = {}
    files = rec.get("images") or sorted(f.name for f in d.glob("image_*.png"))
    files = [d / f for f in files if (d / f).exists()]
    images = [rel_url(f) for f in files]
    status = rec.get("status") or ("done" if images else "error")
    if status == "running" and d.name not in generate_mod.active_runs():
        status = "stopped"  # the server went away mid-render
    error = rec.get("error")
    if not error and (d / "error.txt").exists():
        lines = (d / "error.txt").read_text(encoding="utf-8").strip().splitlines()
        error = lines[-1] if lines else "failed"
        status = "error"
    started = time.strftime("%Y-%m-%d %H:%M:%S", time.strptime(d.name[:15], "%Y%m%d-%H%M%S"))
    ref = d / (rec.get("reference") or "reference.png")
    pose = d / rec["pose_reference"] if rec.get("pose_reference") else None
    return {"kind": "manual", "run": f"manual/{d.name}", "status": status, "error": error,
            "warning": rec.get("warning"),
            "started": rec.get("started") or started, "finished": rec.get("finished"),
            "images": images, "thumbs": [thumbs.url(ROOT, f) for f in files],
            "best_url": images[0] if images else None,
            "reference_url": rel_url(ref) if ref.exists() else None,
            "pose_url": rel_url(pose) if pose and pose.exists() else None,
            "subject_url": rel_url(d / rec["subject_reference"]) if rec.get("subject_reference")
            and (d / rec["subject_reference"]).exists() else None,
            "targets": rec.get("targets"), "pose_pick": rec.get("pose_pick"),
            "positive": rec.get("positive", ""), "negative": rec.get("negative", ""),
            "params": rec.get("params", ""), "size": rec.get("size"),
            "loras": rec.get("loras") or [], "timing": rec.get("timing"),
            "ipadapter": bool(rec.get("ipadapter")), "control": bool(rec.get("control")),
            "request": rec.get("request") or {}, "autofix": autofix_view(d), "upscale": upscale_view(d),
            "nobg": nobg_view(d), "image_names": [f.name for f in files]}


def run_detail(name: str) -> dict:
    d = (ROOT / "runs" / name).resolve()
    if d.parent != (ROOT / "runs").resolve() or not d.is_dir():
        raise FileNotFoundError(name)
    rounds = []
    log = d / "log.jsonl"
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            kind = r.get("type", "round")
            if kind == "confirm" and rounds:
                rounds[-1].setdefault("confirms", []).append({
                    "image": rel_url(d / r["image"]), "first": r["first"], "recheck": r["recheck"],
                    "judge_model": r.get("judge_model"), "confirm_model": r.get("confirm_model"),
                    "confirm_threshold": r.get("confirm_threshold"), "passed": r["passed"], "diagnosis": r["review"].get("diagnosis", "")})
            elif kind == "memory" and rounds:
                rounds[-1]["memory"] = r["summary"]
            elif kind == "round":
                n = len(r["params"])
                diffs = [[] for _ in range(n)]
                for pos, i in enumerate(r.get("sent", [])):
                    c = next((c for c in r["review"].get("candidates", []) if c.get("index") == pos), {})
                    diffs[i] = c.get("differences") or []
                rounds.append({
                    "round": r["round"], "phase": r["phase"], "judge_model": r.get("judge_model"),
                    "images": [rel_url(d / f"r{r['round']:02d}_c{i}.png") for i in range(n)],
                    "scores": r["scores"], "best": r["best"], "score": r["scores"][r["best"]],
                    "params": [f"{p['mode']} seed={p['seed']} cfg={p['cfg']:g} denoise={p['denoise']:g}"
                               + (f" mask='{p['mask_target']}'" if p.get("mask_target") else "") for p in r["params"]],
                    "diagnosis": r["review"].get("diagnosis", ""), "edit": r["review"].get("edit", {}),
                    "prompt_tokens": r.get("prompt_tokens"), "differences": diffs,
                })
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8")) if (d / "summary.json").exists() else {}
    prompt = json.loads((d / "prompt.json").read_text(encoding="utf-8")) if (d / "prompt.json").exists() else None
    info = json.loads((d / "run.json").read_text(encoding="utf-8")) if (d / "run.json").exists() else {}
    ref = next((p for p in d.iterdir() if p.stem == "reference"), None)
    return {"run": name, "summary": summary, "reference": rel_url(ref) if ref else None, "rounds": rounds,
            "threshold": info.get("threshold"),
            "prompt": prompt, "reproduce": reproduce_info(d, summary, info),
            "hands": hands_results(d, summary),
            "setup": {**{k: info.get(k) for k in ("checkpoint", "checkpoint_base", "judge_backend", "judge_model",
                                                  "lora_mode", "start_loras",
                                                  "lora_pick", "incompatible_loras", "size", "size_setting", "pose")},
                      "pose_url": rel_url(d / info["pose"]["control"]) if (info.get("pose") or {}).get("control")
                      else None} if info else None}


def reproduce_info(d: Path, summary: dict, info: dict) -> dict | None:
    """Everything needed to render the best image again. Newer runs store it in
    summary.json; older ones are rebuilt from log.jsonl (without checkpoint/LoRAs)."""
    best = summary.get("best")
    if not best and summary.get("best_image") and (d / "log.jsonl").exists():
        m = re.match(r"r(\d+)_c(\d+)", summary["best_image"])
        if m:
            for line in (d / "log.jsonl").read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                if r.get("type", "round") == "round" and r["round"] == int(m.group(1)):
                    p = r["params"][int(m.group(2))]
                    best = {"image": summary["best_image"], "round": r["round"], "params": p, "legacy": True,
                            "rendered_positive": (r.get("rendered_positive") or [None] * 9)[int(m.group(2))]
                            or p["positive"], "source": r.get("source")}
    if not best:
        return None
    src = best.get("source") or {}
    graph = d / best["graph"] if best.get("graph") else None
    return {**best, "checkpoint": best.get("checkpoint") or info.get("checkpoint"),
            "workflow": info.get("workflow"), "fixed_inputs": info.get("fixed_inputs"),
            "judge_model": info.get("judge_model"), "lora_mode": info.get("lora_mode"),
            "lora_pick": info.get("lora_pick"), "description": info.get("description"),
            "graph_url": rel_url(graph) if graph and graph.exists() else None,
            "image_url": rel_url(d / "best.png") if (d / "best.png").exists() else None,
            "source_url": rel_url(d / src["image"]) if src.get("image") and src["image"] != "reference" else None,
            "mask_url": rel_url(d / src["mask"]) if src.get("mask") else None}


# ---- poses and the hand refiner ------------------------------------------------------------
POSE_DESCRIBE = """Describe only the POSE and FRAMING of the main character in the image, as Stable
Diffusion tags: framing (e.g. cowboy shot, full body, upper body), body orientation,
stance and weight, arm and hand positions, head direction, and eye gaze direction (as
seen by the viewer: looking to the viewer's left/right, at the viewer). No clothing,
hair, colours, character or background. 8-20 short comma-separated tags.

Also give the pose a short name (2-5 lowercase words joined by underscores, e.g.
hands_on_hips_three_quarter, sitting_cross_legged, over_shoulder_glance) naming its most
distinctive feature so it can be told apart in a list of poses. Only the pose, never the
character or outfit. JSON only."""


def pose_library():
    from .pose import PoseLibrary
    return PoseLibrary(ROOT / "poses")


def poses_list() -> dict:
    from . import pose
    return {"available": pose.available(),
            **organize.annotate(pose_library().root, [
                {"name": p["name"], "description": p["description"], "size": p["size"],
                 "preview": rel_url(p["preview"]), "source": rel_url(p["source"]),
                 "thumb": thumbs.url(ROOT, p["source"])}
                for p in pose_library().list()])}


def _decode_image(image_b64: str):
    import io
    from PIL import Image
    img = Image.open(io.BytesIO(base64.b64decode(image_b64.split(",", 1)[-1])))
    img.load()
    return img


def target_references(targets, pose_choice=None) -> list[tuple]:
    """Home's style / subject / pose as (role, image, tags) for the LoRA chooser: an uploaded
    image, else the saved style, character or pose chosen. "AI picks" has no image yet."""
    from PIL import Image

    from .reflib import resolve
    out = []
    for role in ("style", "subject", "pose"):
        t = (targets or {}).get(role) or {}
        lib = (pose_choice if role == "pose" else t.get("library")) or ""
        try:
            if t.get("image_b64"):
                out.append((role, _decode_image(t["image_b64"]), ""))
            elif lib and lib != "auto" and role == "pose":
                with Image.open(pose_library().source(lib)) as im:
                    out.append((role, im.copy(), pose_library().get(lib).get("description", "")))
            elif lib and lib != "auto" and (got := resolve(ROOT, role, lib)):
                with Image.open(got["image"]) as im:
                    out.append((role, im.copy(), got["description"]))
        except (OSError, ValueError):
            continue  # a saved one that has gone missing: choose without it
    return out


def add_pose(name: str, image_b64: str, fallback: str = "", img=None) -> str:
    """Save a pose. With no name, the LLM names it from the pose (else `fallback`, the file name)."""
    from .backends import make_backend

    img = img if img is not None else _decode_image(image_b64)
    cfg = load_config()

    def describe(image) -> tuple[str, str]:
        """(pose tags, suggested name); the name is only asked for when none was given."""
        props = {"pose": {"type": "string"}, **({} if name.strip() else {"name": {"type": "string"}})}
        try:
            data, _c, _t = make_backend(cfg["judge"]).complete(
                POSE_DESCRIBE, [{"text": "IMAGE:"}, {"image": image}],
                {"type": "object", "properties": props, "required": list(props),
                 "additionalProperties": False}, "pose", cfg["judge"].get("image_max_side", 512))
            return data.get("pose", "").strip(), str(data.get("name") or "").strip().lower().replace(" ", "_")
        except Exception:  # the LLM is optional here: the skeleton is what matters
            return "", ""
    return pose_library().add(name.strip(), img, describe, fallback=fallback or "pose")


def ref_library(kind: str):
    from .reflib import RefLibrary
    return RefLibrary(ROOT, kind)


def ref_list(kind: str) -> dict:
    return organize.annotate(ref_library(kind).root, [
        {"name": p["name"], "description": p["description"], "size": p["size"],
         "source": rel_url(p["source"]), "thumb": thumbs.url(ROOT, p["source"])}
        for p in ref_library(kind).list()])


LIBRARY_ROUTES = {"/api/poses/organize": "pose", "/api/styles/organize": "style",
                  "/api/characters/organize": "character"}


def library_dir(kind: str) -> Path:
    return pose_library().root if kind == "pose" else ref_library(kind).root


def organize_library(kind: str, data: dict) -> dict:
    """Folders and favourites of a saved library (see organize.py). One of:
    {"names", "folder"?, "favorite"?}, {"new_folder"}, {"move_folder", "to"}."""
    d = library_dir(kind)
    if data.get("new_folder") is not None:
        organize.add_folder(d, data["new_folder"])
    elif data.get("move_folder") is not None:
        organize.move_folder(d, data["move_folder"], data.get("to") or "")
    else:
        names = [str(n) for n in data.get("names") or []]
        if not names:
            raise ValueError("choose something to file")
        organize.update(d, names, folder=data.get("folder"), favorite=data.get("favorite"))
    return poses_list() if kind == "pose" else ref_list(kind)


def add_ref(kind: str, name: str, image_b64: str, fallback: str = "", img=None) -> str:
    """Save a style or character. The LLM writes its tags and, with no name, names it."""
    from .backends import make_backend
    from .reflib import KINDS

    img = img if img is not None else _decode_image(image_b64)
    cfg = load_config()

    def describe(image) -> tuple[str, str]:
        props = {"tags": {"type": "string"}, **({} if name.strip() else {"name": {"type": "string"}})}
        try:
            data, _c, _t = make_backend(cfg["judge"]).complete(
                KINDS[kind]["describe"], [{"text": "IMAGE:"}, {"image": image}],
                {"type": "object", "properties": props, "required": list(props), "additionalProperties": False},
                kind, cfg["judge"].get("image_max_side", 512))
            return data.get("tags", "").strip(), str(data.get("name") or "").strip().lower().replace(" ", "_")
        except Exception:  # the LLM is optional here: the image is what matters
            return "", ""
    return ref_library(kind).add(name.strip(), img, describe, fallback=fallback or kind)



LIBRARY_KINDS = ("character", "style", "pose")


def save_run_image(run: str, image: str, save: dict) -> dict:
    """Save one image of a History entry to the character, style and/or pose library.
    save: {kind: name}; an empty name has the LLM name it. The kinds are saved side by
    side (each is one LLM call). Returns {kind: {"name"} or {"error"}}."""
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image

    d = (ROOT / "runs" / run).resolve()
    if not d.is_relative_to((ROOT / "runs").resolve()) or not d.is_dir():
        raise FileNotFoundError(f"no History entry {run}")
    names = generate_mod._image_names(d, [image])
    if not names:
        raise FileNotFoundError(f"no image {image} in {run}")
    kinds = {k: str(v or "").strip() for k, v in (save or {}).items() if k in LIBRARY_KINDS}
    if not kinds:
        raise ValueError("choose character, style and/or pose")
    with Image.open(d / names[0]) as im:
        img = im.copy()
    fallback = d.name

    def one(kind: str) -> tuple[str, dict]:
        try:
            if kind == "pose":
                return kind, {"name": add_pose(kinds[kind], "", fallback, img=img.copy())}
            return kind, {"name": add_ref(kind, kinds[kind], "", fallback, img=img.copy())}
        except Exception as e:
            return kind, {"error": str(e)}
    with ThreadPoolExecutor(len(kinds)) as pool:
        return dict(pool.map(one, kinds))

# ---- Designer -------------------------------------------------------------------------------

def _project_file(url: str) -> Path:
    """The file behind a /files/ URL of a History image, saved character or design image."""
    rel = unquote(urlparse(url or "").path).removeprefix("/files/")
    target = (ROOT / rel).resolve()
    if not any(target.is_relative_to((ROOT / d).resolve()) for d in ("runs", "characters", "styles", "designs")):
        raise FileNotFoundError(url)
    if not target.is_file() or target.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
        raise FileNotFoundError(url)
    return target


def _made_with(image: Path) -> tuple[dict, dict]:
    """(params, source) of a History image: the settings that made it, as Designer keeps
    them ({} for an image without recorded settings, e.g. a saved character)."""
    runs = (ROOT / "runs").resolve()
    if not image.is_relative_to(runs):
        return {}, {"image": _files_url(str(image))}
    rel = image.relative_to(runs).parts
    run = "/".join(rel[:2]) if rel[0] == "manual" else rel[0]
    try:
        t = fix_target(run)
    except Exception:
        return {}, {"run": run, "image": image.name}
    p = t["params"]
    params = {"positive": t["positive"] or p.positive, "negative": p.negative, "seed": p.seed, "steps": p.steps,
              "cfg": p.cfg, "sampler_name": p.sampler_name, "scheduler": p.scheduler,
              "loras": [list(x) for x in p.loras], "checkpoint": t.get("checkpoint")}
    params.update(_noise_origin(runs / run, image.name))
    return params, {"run": run, "image": image.name}


def _noise_origin(run_dir: Path, name: str) -> dict:
    """Where a History image sits in the noise that drew it (designer.regen_source): its place
    in its batch ("image_08.png" -> 7; a remade single image keeps the place it had), the batch
    size, the size, and whether it was drawn from noise alone (txt2img at denoise 1, no
    reference image, ControlNet or IP-Adapter), which is when that seed and place draw it again."""
    try:
        rec = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    m = re.fullmatch(r"image_(\d+)\.png", name)
    if not m or not isinstance(rec.get("request"), dict):
        return {}
    q = rec["request"]
    pick = generate_mod.batch_pick(q)
    index = int(m.group(1)) - 1
    of = pick[1] if pick else int(q.get("batch_size") or rec.get("batch") or index + 1)
    mode = (rec.get("params") or "").split(" ", 1)[0] if isinstance(rec.get("params"), str) else ""
    from_noise = (mode in ("", "txt2img") and float(q.get("denoise") or 1.0) >= 0.999 and not rec.get("control")
                  and not rec.get("ipadapter") and not rec.get("reference") and not rec.get("subject_reference")
                  and not rec.get("pose_reference"))
    return {"batch_index": index, "batch_of": max(of, index + 1), "size": rec.get("size"),
            "from_noise": bool(from_noise)}


def _design_urls(design: dict) -> dict:
    d = designer.design_dir(ROOT, design["id"])
    url = lambda rel: rel_url(d / rel)  # noqa: E731
    thumb = lambda rel: thumbs.url(ROOT, d / rel) or url(rel)  # noqa: E731
    return {**design, "cover_url": url(design.get("cover") or "original.png"),
            "cover_thumb": thumb(design.get("cover") or "original.png"),
            "original_url": url("original.png"), "original_thumb": thumb("original.png"),
            "catalog": [{**e, "url": url(e["image"]), "thumb": thumb(e["image"])} for e in design.get("catalog", [])]}


def designs_list() -> list[dict]:
    busy = {b.split("/")[0] for b in generate_mod.design_busy()}
    return [{**{k: v for k, v in _design_urls(x).items() if k not in ("params", "description")},
             "busy": x["id"] in busy} for x in designer.list_designs(ROOT)]


def design_detail(design_id: str) -> dict:
    _backfill_noise_origin(design_id)  # the manual editor offers "its own noise" when it's known
    design = _design_urls(designer.load(ROOT, design_id))
    busy = generate_mod.design_busy()
    jobs = {(j.get("design"), j.get("session")): j for j in generate_mod.queue_status() if j.get("kind") == "design"}
    sessions = []
    for sid in reversed(design.get("sessions", [])):
        try:
            s = designer.load_session(ROOT, design_id, sid)
        except (OSError, ValueError):
            continue
        sdir = designer.session_dir(ROOT, design_id, sid)
        job = jobs.get((design_id, sid))
        for r in s.get("rounds", []):
            for c in r["candidates"]:
                c["url"] = rel_url(sdir / c["image"])
                c["thumb"] = thumbs.url(ROOT, sdir / c["image"]) or c["url"]
        sessions.append({**{k: v for k, v in s.items() if k not in ("knobs",)},
                         "summary": designer.change_text(s),
                         "base_url": rel_url(designer.file_of(ROOT, design_id, s["base"]))
                         if (designer.design_dir(ROOT, design_id) / s["base"]).is_file() else None,
                         "target_url": rel_url(sdir / "target.png") if (sdir / "target.png").is_file() else None,
                         "busy": f"{design_id}/{sid}" in busy,
                         "size_mb": round(sum(f.stat().st_size for f in sdir.rglob("*") if f.is_file()) / 1e6, 1),
                         "job": {"id": job["id"], "status": job["status"], "state": job.get("state")} if job else None})
    # Candidates already kept, so the page can mark them: "<session>/<image>".
    kept = sorted(f"{e.get('session')}/{e.get('candidate', '')}" for e in design["catalog"])
    return {**design, "sessions": sessions, "kept": kept, "settings": designer.settings(load_config()),
            "source_description": _source_description(design)}


def _source_description(design: dict) -> str:
    """The description (Generate's "describe it" box, which the LLM writes prompts from) of the
    History run a character was added from; "" when it came from elsewhere or had none."""
    run = (design.get("source") or {}).get("run")
    if design.get("description"):
        return design["description"]
    runs = (ROOT / "runs").resolve()
    try:
        rec_file = (runs / run / "run.json").resolve() if run else None
        if rec_file is None or runs not in rec_file.parents:
            return ""
        return str((json.loads(rec_file.read_text(encoding="utf-8")).get("request") or {}).get("description") or "")
    except (OSError, ValueError):
        return ""


def create_design(data: dict) -> dict:
    if data.get("image_b64"):
        tmp = ROOT / "designs" / "_upload.png"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        _decode_image(data["image_b64"]).convert("RGB").save(tmp)
        try:
            design = designer.create(ROOT, tmp, name=data.get("name") or "", source={"upload": True})
        finally:
            tmp.unlink(missing_ok=True)
        return design
    image = _project_file(data.get("image") or "")
    params, source = _made_with(image)
    if not params and data.get("design"):  # an image of another design: its settings come along
        try:
            params = designer.load(ROOT, data["design"]).get("params") or {}
        except (OSError, ValueError):
            pass
    return designer.create(ROOT, image, name=data.get("name") or "", params=params, source=source)


def _backfill_noise_origin(design_id: str) -> None:
    """A character added before its noise origin was recorded gets it from its History run."""
    with designer._lock:
        design = designer.load(ROOT, design_id)
        src, params = design.get("source") or {}, design.get("params") or {}
        if "from_noise" in params or not src.get("run") or not src.get("image"):
            return
        origin = _noise_origin((ROOT / "runs" / src["run"]).resolve(), src["image"])
        if origin and (ROOT / "runs").resolve() in (ROOT / "runs" / src["run"]).resolve().parents:
            design["params"] = {**params, **origin}
            designer.save(ROOT, design)


def design_edit(data: dict) -> dict:
    target = None
    if data.get("image_b64"):
        target = ROOT / "designs" / "_target.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        _decode_image(data["image_b64"]).convert("RGB").save(target)
    _backfill_noise_origin(data["id"])
    if data.get("manual"):  # your own prompts and settings: rendered, not judged
        q, files = dict(data["manual"]), {}
        try:
            for key in ("style", "pose"):
                b64 = q.pop(f"{key}_b64", None)
                if b64:
                    files[key] = ROOT / "designs" / f"_{key}_upload.png"
                    files[key].parent.mkdir(parents=True, exist_ok=True)
                    _decode_image(b64).convert("RGB").save(files[key])
            sess = designer.new_manual_session(ROOT, data["id"], base=data.get("base") or "original.png",
                                               request=q, style_image=files.get("style"), pose_image=files.get("pose"))
        finally:
            for f in files.values():
                f.unlink(missing_ok=True)
        name = designer.load(ROOT, data["id"])["name"]
        gen_id = generate_mod.start_design(data["id"], sess["id"], label=f"{name}: manual edit"[:140])
        return {"session": sess["id"], "id": gen_id}
    try:
        num = lambda k, lo, hi: max(lo, min(hi, float(data[k]))) if data.get(k) not in (None, "") else None  # noqa: E731
        sess = designer.new_session(ROOT, data["id"], base=data.get("base") or "original.png", kind=data.get("kind"),
                                    change={"text": data.get("text") or "", "style": data.get("style") or "",
                                            "pose": data.get("pose") or ""},
                                    threshold=num("threshold", 50, 100),
                                    max_rounds=int(num("max_rounds", 1, 20)) if num("max_rounds", 1, 20) else None,
                                    target_image=target)
    finally:
        if target is not None:
            target.unlink(missing_ok=True)
    name = designer.load(ROOT, data["id"])["name"]
    gen_id = generate_mod.start_design(data["id"], sess["id"], label=f"{name}: {designer.change_text(sess)}"[:140])
    return {"session": sess["id"], "id": gen_id}


def design_continue(data: dict) -> dict:
    key = f"{data['id']}/{data['session']}"
    if key in generate_mod.design_busy():
        raise ValueError("that edit is still running")
    sess = designer.load_session(ROOT, data["id"], data["session"])
    rounds = max(1, min(10, int(data.get("rounds") or 1)))
    sess.update(status="queued", confirmed=False)
    designer.save_session(ROOT, data["id"], sess)
    name = designer.load(ROOT, data["id"])["name"]
    gen_id = generate_mod.start_design(data["id"], data["session"], rounds=rounds,
                                       label=f"{name}: one more round of {designer.change_text(sess)}"[:140])
    return {"id": gen_id}


def design_keep(data: dict) -> dict:
    if data.get("to") == "new":
        return {"design": designer.new_from_candidate(ROOT, data["id"], data["session"], data["image"],
                                                      data.get("name") or "")}
    entry = designer.keep(ROOT, data["id"], data["session"], data["image"])
    if data.get("cover"):
        designer.update(ROOT, data["id"], cover=entry["image"])
    return {"entry": entry}


_hand_jobs: dict[str, threading.Thread] = {}


def refine_hands_run(name: str) -> dict:
    """Repaint the hands of a finished run's final image (best.png), in the background.
    Each attempt is kept in the run folder (manualN_hands_*.png) and listed in
    hands_manual.json; nothing replaces best.png."""
    d = (ROOT / "runs" / name).resolve()
    if d.parent != (ROOT / "runs").resolve() or not (d / "best.png").exists():
        raise FileNotFoundError(name)
    if name in _hand_jobs and _hand_jobs[name].is_alive():
        raise RuntimeError("hands are already being refined for this run")
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    info = json.loads((d / "run.json").read_text(encoding="utf-8")) if (d / "run.json").exists() else {}
    rep = reproduce_info(d, summary, info)
    if not rep or not rep.get("params"):
        raise RuntimeError("this run has no recorded settings to repaint with")
    log_file = d / "hands_manual.json"
    attempts = json.loads(log_file.read_text(encoding="utf-8")) if log_file.exists() else []
    entry = {"n": len(attempts) + 1, "state": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
             "source": "best.png"}
    attempts.append(entry)
    log_file.write_text(json.dumps(attempts, indent=2), encoding="utf-8")

    def work():
        from .comfy import ComfyClient
        from .handfix import refine_hands
        from .params import GenParams
        from .workflow import Workflows
        messages = []
        try:
            cfg = load_config()
            launcher = comfy_launcher(cfg)
            if cfg.get("comfyui", {}).get("autostart", True) and not launcher.ensure_running(lambda m: None):
                raise RuntimeError(launcher.message)
            comfy = ComfyClient(cfg["comfy_url"])
            p = dict(rep["params"])
            params = GenParams(**{**p, "loras": tuple(tuple(x) for x in (rep.get("loras") and
                                  [(l["name"], l["strength"]) for l in rep["loras"]] or p.get("loras") or []))})
            res = refine_hands(d / "best.png", params, comfy=comfy, flows=Workflows(ROOT / "workflows"), cfg=cfg,
                               checkpoint=rep.get("checkpoint"), positive=rep.get("rendered_positive") or p["positive"],
                               out_dir=d, upload=lambda path: comfy.upload_image(path, f"ouroboros/{d.name}"),
                               log=messages.append, tag=f"manual{entry['n']}_hands")
            entry.update(state="done" if res["image"] else "nothing to do", image=res["image"].name if res["image"] else None,
                         hands=res["hands"], found=res["found"], steps=res["steps"])
        except Exception as e:
            entry.update(state="failed", error=str(e)[:500])
        entry["messages"] = messages
        entry["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        current = json.loads(log_file.read_text(encoding="utf-8"))
        current[entry["n"] - 1] = entry
        log_file.write_text(json.dumps(current, indent=2), encoding="utf-8")

    _hand_jobs[name] = threading.Thread(target=work, daemon=True)
    _hand_jobs[name].start()
    return entry


def hands_results(d: Path, summary: dict) -> dict:
    auto = (summary.get("best") or {}).get("hands")
    if auto:
        auto = {**auto, "image_url": rel_url(d / auto["image"]), "before_url": rel_url(d / auto["before"])}
    manual = []
    if (d / "hands_manual.json").exists():
        for a in json.loads((d / "hands_manual.json").read_text(encoding="utf-8")):
            manual.append({**a, "image_url": rel_url(d / a["image"]) if a.get("image") else None,
                           "before_url": rel_url(d / "best.png")})
    return {"auto": auto, "manual": manual,
            "busy": d.name in _hand_jobs and _hand_jobs[d.name].is_alive()}


def remove_run(name: str) -> None:
    """Move a run to runs/_removed/ (hidden from History; delete that folder to free space).
    Home tab generations are named manual/<run>."""
    manual = name.startswith("manual/")
    base = ROOT / "runs" / ("manual" if manual else "")
    leaf = name[len("manual/"):] if manual else name
    d = (base / leaf).resolve()
    if d.parent != base.resolve() or not d.is_dir() or leaf.startswith("_") or leaf == "manual":
        raise FileNotFoundError(name)
    if name in runner.active_runs() or (manual and leaf in generate_mod.active_runs()):
        raise RuntimeError("that run is in progress")
    if name in favorites.load(ROOT):
        raise RuntimeError("that entry is a favorite; take it off the Favorites page first")
    if name in generate_mod.busy_targets():
        raise RuntimeError("an auto-fix, upscale or background removal of this entry is queued or running; "
                           "remove it from the queue first")
    trash = ROOT / "runs" / "_removed"
    trash.mkdir(exist_ok=True)
    dest = trash / (f"manual_{leaf}" if manual else leaf)
    if dest.exists():
        dest = trash / f"{dest.name}_{int(time.time())}"
    shutil.move(str(d), str(dest))


BIN = ROOT / "runs" / "_removed"
_bin_cache: dict = {}


def bin_info() -> dict:
    """What the History recycle bin (runs/_removed/) holds. The size walk is cached until
    the folder changes, since /api/state is polled every few seconds."""
    try:
        entries = [p for p in BIN.iterdir()]
        stamp = (BIN.stat().st_mtime, len(entries))
    except OSError:
        return {"count": 0, "bytes": 0}
    if _bin_cache.get("stamp") != stamp:
        size = 0
        for entry in entries:
            files = entry.rglob("*") if entry.is_dir() else [entry]
            size += sum(f.stat().st_size for f in files if f.is_file() and not f.is_symlink())
        _bin_cache.update(stamp=stamp, info={"count": len(entries), "bytes": size})
    return _bin_cache["info"]


def empty_bin() -> dict:
    """Permanently delete everything removed from History (runs/_removed/)."""
    before = bin_info()
    if BIN.is_dir():
        for entry in BIN.iterdir():
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()
    _bin_cache.clear()
    thumbs.prune(ROOT)
    return before


# ---- LoRAs, checkpoints, model presets ------------------------------------------------------
_lora_refresh: dict = {"thread": None, "error": None}


def _files_url(path: str | None) -> str | None:
    """A /files/ URL for a file under the project folder, or None for anything outside it."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return "/files/" + p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return None


def loras_state() -> dict:
    cfg = load_config()
    lib = lora_library(cfg)
    ckpt = cfg.get("defaults", {}).get("checkpoint") or default_checkpoint()
    base = checkpoint_base(ckpt or "", cfg.get("checkpoint_bases"))
    workflow_loras = dict(_workflow_loras())
    items = []
    for name, r in sorted(lib.index().items(), key=lambda kv: kv[1]["title"].lower()):
        items.append({
            "name": name, "stem": lora_stem(name), "title": r["title"], "base_model": r.get("base_model"),
            "compatible": compatible(r.get("base_model"), base), "trigger_words": r["trigger_words"],
            "typical_weight": r.get("typical_weight"), "tags": r.get("tags", [])[:8],
            "description": r.get("description", "")[:300], "civitai_url": r.get("civitai_url"),
            "source": r.get("source"), "error": r.get("error"), "stats": lib.stats_line(name),
            "in_workflow": workflow_loras.get(name),
            # Catalog example renders live outside the project, where /files/ can't reach
            # them; the picker still uses the paths, the UI just gets no thumbnail.
            "examples": [{"thumb": _files_url(e.get("thumb")),
                          "prompt": e.get("prompt", "")[:300], "weight": e.get("weight")}
                         for e in r.get("examples", [])[:4]],
        })
    running = bool(_lora_refresh["thread"] and _lora_refresh["thread"].is_alive())
    return {"items": items, "checkpoint": ckpt, "checkpoint_base": base, "files": len(lib.files()),
            "dirs": [str(p) for p in lib.dirs], "refreshing": running, "status": lib.status,
            "error": _lora_refresh["error"]}


_status_lib = {}


def refresh_loras() -> bool:
    if _lora_refresh["thread"] and _lora_refresh["thread"].is_alive():
        return False
    lib = lora_library(load_config())

    def work():
        _lora_refresh["error"] = None
        try:
            lib.refresh(online=True)
        except Exception as e:
            _lora_refresh["error"] = str(e)

    _lora_refresh["thread"] = threading.Thread(target=work, daemon=True)
    _lora_refresh["thread"].start()
    return True


def _workflow_loras() -> tuple:
    try:
        from .workflow import Workflows
        return Workflows(ROOT / "workflows").default_loras()
    except Exception:
        return ()


def default_checkpoint() -> str | None:
    try:
        from .workflow import Workflows
        return Workflows(ROOT / "workflows").default("checkpoint")
    except Exception:
        return None


def folder_checkpoints(folder: str) -> list[str]:
    """Checkpoints under the folder, named as ComfyUI names them (relative path, subfolders too)."""
    if not (folder or "").strip() or not Path(folder).is_dir():
        return []
    folder = Path(folder)
    return sorted(str(p.relative_to(folder)) for p in folder.rglob("*")
                  if p.suffix.lower() in (".safetensors", ".ckpt") and p.is_file())


def checkpoints() -> dict:
    cfg = load_config()
    folder_str = (cfg.get("checkpoints_dir") or "").strip()
    folder = Path(folder_str or "/nonexistent")
    in_folder = folder_checkpoints(folder_str)
    names, comfy_up = [], False
    try:
        info = requests.get(f"{cfg['comfy_url']}/object_info/CheckpointLoaderSimple", timeout=5).json()
        names = combo_options(info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"])
        comfy_up = True
    except Exception:
        names = in_folder
    # Checkpoints in the chosen folder that the running ComfyUI doesn't offer: it was started
    # before the folder was chosen (or outside Ouroboros), so it needs a restart to load them.
    not_loaded = [n for n in in_folder if n not in set(names)] if comfy_up else []
    out = []
    from .ckpt_info import lookup
    for n in names:
        base = checkpoint_base(n, cfg.get("checkpoint_bases"))
        info = lookup(n, cfg)
        f = folder / n
        out.append({"name": n, "base": base, "size_gb": round(f.stat().st_size / 1e9, 1) if f.exists() else None,
                    "usable": base != "Flux", "model": info["model"],
                    "vpred": (info["prediction"] or "").startswith("v-"),
                    "lora_folders": lora_folders_for(n, cfg)})
    return {"items": out, "default": default_checkpoint(), "selected": cfg.get("defaults", {}).get("checkpoint") or "",
            "folder": cfg.get("checkpoints_dir", ""), "folder_found": folder.is_dir() if folder_str else None,
            "not_loaded": not_loaded}


_di_models: tuple[float, list[str]] = (0.0, [])


def deepinfra_model_list(url: str = "") -> list[str]:
    """DeepInfra's vision models, cached for five minutes (the status pill asks often)."""
    global _di_models
    if not _di_models[1] or time.time() - _di_models[0] > 300:
        _di_models = (time.time(), deepinfra_models(url))
    return _di_models[1]


def catalog_root() -> Path:
    """Where the lora-classifier writes its output (loras.catalog_dir)."""
    return Path(load_config().get("loras", {}).get("catalog_dir", "")).expanduser()


def lora_cards(refresh: bool = False) -> list[dict]:
    """The classifier's cards, plus basic ones for folders it hasn't classified yet."""
    from .lora_catalog import catalog, folder_card
    root = catalog_root()
    if not (root / "state").is_dir():
        return []
    cards = catalog(root, ROOT / "cache", refresh)
    have = {c["comfy_name"] for c in cards}
    extra = [folder_card(r) for n, r in lora_library(load_config()).unclassified().items() if n not in have]
    return cards + sorted(extra, key=lambda c: c["title"].lower())


def lora_thumb(lora_id: str) -> Path | None:
    from .lora_catalog import thumbnail
    cfg = load_config()
    entry = next((e for e in lora_cards() if e["id"] == lora_id), None)
    if entry is None:
        return None
    return thumbnail(catalog_root(), ROOT / "cache", entry,
                     int(cfg.get("loras", {}).get("civitai_max_nsfw_level", 16)))


def model_preset(model: str, backend: str = "ollama") -> dict | None:
    from .model_profiles import preset
    return preset(model, backend)


def service_status() -> dict:
    cfg = load_config()
    status = {"backend": cfg["judge"]["backend"],
              "model": cfg["judge"].get(cfg["judge"]["backend"], {}).get("model") or "",
              "confirm_model": cfg["judge"].get("confirm_model") or "",
              "prompt_model": cfg["judge"].get("prompt_model") or "",
              "ip_max_combined": max_combined(cfg.get("ipadapter", {}))}
    launcher = comfy_launcher(cfg).status()
    status["comfyui"] = {k: launcher[k] for k in ("state", "message")}
    try:
        requests.get(f"{cfg['comfy_url']}/system_stats", timeout=2).raise_for_status()
        status["comfy"] = "ok"
        ks = requests.get(f"{cfg['comfy_url']}/object_info/KSampler", timeout=5).json()["KSampler"]["input"]["required"]
        status["samplers"], status["schedulers"] = combo_options(ks["sampler_name"]), combo_options(ks["scheduler"])
        try:
            up = requests.get(f"{cfg['comfy_url']}/object_info/UpscaleModelLoader", timeout=5).json()
            status["upscale_models"] = combo_options(up["UpscaleModelLoader"]["input"]["required"]["model_name"])
        except Exception:
            status["upscale_models"] = []
    except Exception as e:
        status["comfy"] = ("starting in the background" if launcher["state"] == "starting"
                           else f"not running ({launcher['message'] or type(e).__name__})")
    if cfg["judge"]["backend"] == "ollama":
        o = cfg["judge"]["ollama"]
        try:
            models = ollama_models(o["url"])
            status["judge"] = "ok" if o["model"] in models else f"model '{o['model']}' not found on server"
        except Exception as e:
            status["judge"] = f"unreachable ({type(e).__name__})"
    elif cfg["judge"]["backend"] == "deepinfra":
        from .backends import DeepInfraBackend
        d = cfg["judge"]["deepinfra"]
        try:
            DeepInfraBackend(d).api_key
        except RuntimeError:
            status["judge"] = "no DeepInfra API key (Settings -> API keys)"
        else:
            try:
                status["judge"] = ("ok" if d["model"] in deepinfra_model_list(d.get("url", ""))
                                   else f"'{d['model']}' is not a vision model on DeepInfra")
            except Exception as e:
                status["judge"] = f"unreachable ({type(e).__name__})"
    else:
        status["judge"] = "ok" if keys.get("openai") else "no OpenAI API key (Settings -> API keys)"
    from .workflow import spec_path
    status["workflows"] = "ok" if spec_path(ROOT / "workflows") else "workflows/nodes.json missing"
    return status


def public_settings() -> dict:
    cfg = load_config()
    j = cfg["judge"]
    from .model_profiles import role_config
    role_fields = {"temperature", "top_p", "top_k", "num_ctx", "num_predict", "think", "max_tokens",
                   "context_tokens", "reasoning_effort", "image_detail"}
    effective = {role: {k: v for k, v in role_config(j, role).get(j["backend"], {}).items() if k in role_fields}
                 for role in ("prompt", "confirm")}
    return {
        "comfy_url": cfg["comfy_url"],
        "prompt_library": cfg.get("prompt_library", ""),
        "judge": {
            "prompt_model": j.get("prompt_model", ""),
            "prompt_options": j.get("prompt_options", {}), "confirm_options": j.get("confirm_options", {}),
            "effective_options": effective,
            "confirm_thresholds": j.get("confirm_thresholds", {}),
            "backend": j["backend"], "contact_sheet": j.get("contact_sheet", False),
            "image_max_side": j.get("image_max_side", 512), "confirm_model": j.get("confirm_model", ""),
            "ollama": {k: j["ollama"].get(k) for k in ("url", "model", "num_ctx", "num_predict", "temperature",
                                                        "top_p", "top_k", "keep_alive", "think")},
            "openai": {k: j["openai"].get(k) for k in ("model", "reasoning_effort", "image_detail")},
            "deepinfra": {k: j["deepinfra"].get(k) for k in ("model", "temperature", "top_p", "max_tokens",
                                                             "reasoning_effort", "context_tokens", "max_images")},
            "deepinfra_key_set": bool(keys.get("deepinfra")),
        },
        "loop": cfg["loop"], "designer": designer.settings(cfg),
        "defaults": cfg["defaults"],
        "checkpoints_dir": cfg.get("checkpoints_dir", ""),
        "comfyui": cfg.get("comfyui", {}),
        "queue": cfg.get("queue", {}),
        "controlnet": cfg.get("controlnet", {}),
        "hands": cfg.get("hands", {}),
        "autofix": autofix.settings(cfg),
        "upscale": upscale_mod.settings(cfg),
        "loras": {k: v for k, v in cfg.get("loras", {}).items() if k != "civitai_api_key"},
        # Keys never leave the server; Settings only learns whether and where each is set.
        "api_keys": keys.status(),
        "preset": model_preset(j.get(j["backend"], {}).get("model", ""), j["backend"]),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "Ouroboros"

    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    # --- helpers ---------------------------------------------------------------
    def send_json(self, data, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, allowed_root: Path, cache_seconds: int = 0) -> None:
        try:
            path = path.resolve()
            path.relative_to(allowed_root.resolve())
        except ValueError:
            return self.send_json({"error": "forbidden"}, 403)
        if not path.is_file():
            return self.send_json({"error": "not found"}, 404)
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        if cache_seconds:
            self.send_header("Cache-Control", f"max-age={cache_seconds}")
        self.end_headers()
        self.wfile.write(body)

    def body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    # --- routes ----------------------------------------------------------------
    def do_GET(self):
        url = urlparse(self.path)
        path, q = unquote(url.path), parse_qs(url.query)
        try:
            if path in ("/", "/index.html"):
                return self.send_file(STATIC / "index.html", STATIC)
            if path.startswith("/static/"):
                return self.send_file(STATIC / path[len("/static/"):], STATIC)
            if path.startswith("/thumbs/"):
                thumb = thumbs.get(ROOT, path[len("/thumbs/"):])
                if thumb is None:
                    return self.send_json({"error": "not found"}, 404)
                # The URL carries the image's mtime, so the browser can keep it.
                return self.send_file(thumb, thumbs.cache_dir(ROOT), cache_seconds=30 * 86400)
            if path.startswith("/files/"):
                # Checked after resolving: "runs/../config.json" starts with "runs/" too.
                target = (ROOT / path[len("/files/"):]).resolve()
                base = next((b for b in (ROOT / d for d in FILE_DIRS) if target.is_relative_to(b.resolve())), None)
                if base is None:
                    return self.send_json({"error": "forbidden"}, 403)
                return self.send_file(target, base)
            if path.startswith("/library/"):
                return self.send_file(library_root() / path[len("/library/"):], library_root())
            if path == "/api/state":
                return self.send_json({
                    **runner.snapshot(),
                    "queue": [job_info(f, "pending") for f in queue.pending()],
                    "runs": runs_list(),
                    "favorites": favorites_list(),
                    "bin": bin_info(),
                    "generations": generate_mod.queue_status(),
                })
            if path == "/api/status":
                return self.send_json(service_status())
            if path == "/api/library":
                return self.send_json(library_items())
            if path == "/api/library/item":
                return self.send_json(library_item(q["name"][0]))
            if path == "/api/run":
                return self.send_json(run_detail(q["name"][0]))
            if path == "/api/settings":
                return self.send_json(public_settings())
            if path == "/api/comfyui":
                return self.send_json(comfy_launcher(load_config()).status())
            if path == "/api/preprompts":
                from . import preprompts
                return self.send_json({"models": preprompts.listing(load_config())})
            if path == "/api/loras/catalog":
                items = lora_cards("refresh" in q)
                from .lora_catalog import warm
                cfg = load_config()  # start filling the thumbnail cache behind the picker
                warm(catalog_root(), ROOT / "cache",
                     int(cfg.get("loras", {}).get("civitai_max_nsfw_level", 16)))
                return self.send_json({"items": items})
            if path == "/api/loras/thumb":
                thumb = lora_thumb(q.get("id", [""])[0])
                if thumb is None:
                    return self.send_json({"error": "no image"}, 404)
                return self.send_file(thumb, ROOT / "cache")
            if path == "/api/loras/example":
                # The classifier's own example renders, for the LoRA detail pane.
                lora_id, n = q.get("id", [""])[0], q.get("n", ["0"])[0]
                entry = next((e for e in lora_cards() if e["id"] == lora_id), None)
                examples = (entry or {}).get("examples") or []
                if not n.isdigit() or int(n) >= len(examples):
                    return self.send_json({"error": "no image"}, 404)
                source = Path(examples[int(n)]).resolve()
                try:
                    source.relative_to(catalog_root().resolve())
                except ValueError:
                    return self.send_json({"error": "forbidden"}, 403)
                if "thumb" in q:  # the small copy for the detail pane; the full one opens on click
                    thumb = thumbs.lora_example(ROOT, source, lora_id)
                    if thumb is None:
                        return self.send_json({"error": "no image"}, 404)
                    return self.send_file(thumb, thumbs.cache_dir(ROOT), cache_seconds=7 * 86400)
                return self.send_file(source, catalog_root())
            if path == "/api/generate/estimate":
                est = generate_mod.estimate(ROOT)
                cfg = load_config()
                est["comfyui"] = comfy_launcher(cfg).status()["state"]
                est["autofix"] = autofix.settings(cfg)
                est["upscale"] = upscale_mod.settings(cfg)
                return self.send_json(est)
            if path == "/api/generate/status":
                from .generate import status as gen_status
                state = gen_status(q.get("id", [""])[0])
                return self.send_json(state or {"error": "unknown generation"}, 200 if state else 404)
            if path == "/api/loras":
                return self.send_json(loras_state())
            if path == "/api/checkpoints":
                return self.send_json(checkpoints())
            if path == "/api/poses":
                return self.send_json(poses_list())
            if path == "/api/designs":
                return self.send_json({"designs": designs_list()})
            if path == "/api/design":
                return self.send_json(design_detail(q.get("id", [""])[0]))
            if path in REF_ROUTES:
                return self.send_json(ref_list(REF_ROUTES[path]))
            if path == "/api/model-preset":
                return self.send_json(model_preset(q.get("model", [""])[0],
                                                   q.get("backend", ["ollama"])[0]))
            if path == "/api/ollama/models":
                return self.send_json(ollama_models(q["url"][0]))
            if path == "/api/deepinfra/models":
                return self.send_json(deepinfra_model_list(q.get("url", [""])[0]))
            return self.send_json({"error": "not found"}, 404)
        except FileNotFoundError:
            return self.send_json({"error": "not found"}, 404)
        except Exception as e:
            return self.send_json({"error": str(e)}, 500)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            data = self.body()
            if path == "/api/start":
                started = runner.start(once=bool(data.get("once")))
                return self.send_json({"started": started})
            if path == "/api/stop":
                runner.stop()
                return self.send_json({"ok": True})
            if path == "/api/jobs":
                # A reference image, and/or separate style / subject / pose targets
                # ({role: {"image_b64", "image_name", "text"}}; see targets.py).
                images, target_spec, auto_libs = {}, {}, {}
                for role, t in (data.get("targets") or {}).items():
                    if role not in ("style", "subject", "pose") or not isinstance(t, dict):
                        continue
                    if t.get("image_b64"):
                        if Path(t.get("image_name") or "x.png").suffix.lower() not in IMAGE_EXTS:
                            return self.send_json({"error": f"the {role} image must be png/jpg/webp"}, 400)
                        images[role] = (t.get("image_name") or f"{role}.png",
                                        base64.b64decode(t["image_b64"].split(",", 1)[-1]))
                    if (t.get("text") or "").strip():
                        target_spec[role] = {"text": t["text"].strip()}
                    lib = (t.get("library") or "").strip()
                    if lib == "auto":  # the loop picks a saved one when the run starts
                        auto_libs[role] = "auto"
                    elif lib and role in ("style", "subject") and role not in images:
                        from .reflib import resolve
                        got = resolve(ROOT, role, lib)
                        if not got:
                            return self.send_json({"error": f"no saved {role} named {lib}"}, 400)
                        images[role] = ("source.png", Path(got["image"]).read_bytes())
                        target_spec.setdefault(role, {"text": got["description"]})
                        target_spec[role]["name"] = got["name"]
                image = data["image_b64"].split(",", 1)[-1] if data.get("image_b64") else None
                if image and Path(data.get("image_name") or "").suffix.lower() not in IMAGE_EXTS:
                    return self.send_json({"error": "reference must be png/jpg/webp"}, 400)
                if not image and not images:
                    return self.send_json({"error": "an automatic run needs at least one image: style, subject "
                                                    "or pose"}, 400)
                description = (data.get("description") or "").strip()
                spec = {"prompt": data.get("prompt", ""), "negative": data.get("negative", "")}
                if description:
                    spec["description"] = description
                spec.update({k: v for k, v in (data.get("overrides") or {}).items() if v not in (None, "")})
                settings = {k: v for k, v in (data.get("settings") or {}).items() if v not in (None, "")}
                if settings:
                    spec["settings"] = settings
                spec.update(target_spec)
                if auto_libs:
                    spec.setdefault("settings", {}).update({f"{r}_library": v for r, v in auto_libs.items()})
                folder = queue.add(data.get("name") or "job", data.get("image_name"),
                                   base64.b64decode(image) if image else None, spec, images)
                return self.send_json({"name": folder.name})
            if path == "/api/loras/suggest":
                from .backends import make_backend
                from .lora_picker import suggest_loras
                cfg = load_config()
                cards = lora_cards()
                # Only the LoRA folders for the checkpoint being used (NoobAI-XL for a NoobAI one).
                from .loras import in_folders, lora_folders_for
                folders = lora_folders_for(data.get("checkpoint") or default_checkpoint(), cfg)
                keep_names = {l.get("name") for l in data.get("keep") or []}
                cards = [c for c in cards if in_folders(c["comfy_name"], folders) or c["comfy_name"] in keep_names]
                if not cards:
                    return self.send_json({"error": "no classified LoRAs found (loras.catalog_dir)"}, 400)
                reference = _decode_image(data["image_b64"]) if data.get("image_b64") else None
                keep = {l.get("name") for l in data.get("keep") or []}
                out = suggest_loras(make_backend(cfg["judge"]), cards,
                                    data.get("positive", ""), data.get("negative", ""),
                                    data.get("description", ""), reference,
                                    int(data.get("max") or cfg.get("loras", {}).get("max_loras", 3)),
                                    cfg["judge"].get("image_max_side", 512),
                                    [c for c in cards if c["comfy_name"] in keep],
                                    references=target_references(data.get("targets"), data.get("pose_library")))
                return self.send_json(out)
            if path == "/api/sampler/suggest":
                from . import settings_advisor as advisor
                from .backends import make_backend
                cfg = load_config()
                current = data.get("current") or {}
                if not all(current.get(k) not in (None, "") for k in advisor.FIELDS):
                    return self.send_json({"error": "steps, cfg, sampler and scheduler are required"}, 400)
                # The page sends the lists it shows (from ComfyUI); keep the current values if it has none.
                samplers = [str(s) for s in data.get("samplers") or []] or [current["sampler_name"]]
                schedulers = [str(s) for s in data.get("schedulers") or []] or [current["scheduler"]]
                try:
                    from .workflow import Workflows
                    ckpt = data.get("checkpoint") or Workflows(ROOT / "workflows").default("checkpoint")
                except Exception:
                    ckpt = data.get("checkpoint") or ""
                loras = tuple((l["name"], float(l.get("strength", 0.8))) for l in data.get("loras") or [])
                size = data.get("size")
                out = advisor.suggest_settings(
                    make_backend(cfg["judge"]), checkpoint=ckpt,
                    checkpoint_base=checkpoint_base(ckpt, cfg.get("checkpoint_bases")),
                    checkpoint_note=checkpoint_note(cfg, ckpt),
                    samplers=samplers, schedulers=schedulers,
                    current={"steps": int(current["steps"]), "cfg": float(current["cfg"]),
                             "sampler_name": current["sampler_name"], "scheduler": current["scheduler"]},
                    positive=data.get("positive", ""), negative=data.get("negative", ""),
                    description=data.get("description", ""), mode=data.get("mode") or "txt2img",
                    denoise=float(data.get("denoise") or 1.0),
                    size=tuple(size) if isinstance(size, list) and len(size) == 2 else None,
                    lora_notes=advisor.lora_lines(lora_library(cfg), loras) if loras else "",
                    max_side=cfg["judge"].get("image_max_side", 512))
                return self.send_json(out)
            if path == "/api/generate":
                # Queued, not run here: the page is free again at once (ComfyUI is started,
                # if needed, when the task's turn comes).
                if data.get("rerun"):  # a History entry, reproduced exactly and not saved again
                    data = {**generate_mod.rerun_request(ROOT, data["rerun"]), "estimate": data.get("estimate")}
                gen_id = generate_mod.start(data)
                return self.send_json({"id": gen_id, **(generate_mod.status(gen_id) or {})})
            if path == "/api/library/save-image":
                return self.send_json(save_run_image(data["run"], data["image"], data.get("save") or {}))
            if path == "/api/favorites":
                return self.send_json(favorites.set(ROOT, data["run"], data.get("name")))
            if path == "/api/favorites/remove":
                favorites.remove(ROOT, data["run"])
                return self.send_json({"ok": True})
            if path == "/api/remove-bg":
                check_nobg_model()
                return self.send_json({"id": generate_mod.start_nobg(data["run"], data.get("images") or [])})
            if path == "/api/upscale":
                gen_id = generate_mod.start_upscale(data["run"], data.get("images") or [])
                return self.send_json({"id": gen_id})
            if path == "/api/autofix":
                gen_id = generate_mod.start_autofix(data["run"], data.get("images") or [])
                return self.send_json({"id": gen_id})
            if path == "/api/generations/remove":
                if generate_mod.status(data["id"]) is None:
                    return self.send_json({"error": "that task is no longer in the queue"}, 404)
                return self.send_json(generate_mod.remove(data["id"]))
            if path == "/api/generations/clear":
                generate_mod.clear_finished()
                return self.send_json({"ok": True})
            if path == "/api/prompt/preview":
                # A description or a reference image is enough: with only an image the
                # prompter describes what it sees.
                if not (data.get("description") or "").strip() and not data.get("image_b64") and not any(
                        (t or {}).get("image_b64") or ((t or {}).get("text") or "").strip()
                        for t in (data.get("targets") or {}).values()):
                    return self.send_json({"error": "enter a description, a style, subject or pose, or add an image"}, 400)
                return self.send_json(preview_prompt(data))
            if path == "/api/jobs/import":
                src = (library_root() / data["name"]).resolve()
                if src.parent != library_root():
                    return self.send_json({"error": "forbidden"}, 403)
                return self.send_json({"name": queue.import_folder(src).name})
            if path == "/api/jobs/remove":
                folder = job_folder("pending", data["name"])
                if not folder.is_dir():
                    return self.send_json({"error": "job not in the queue"}, 404)
                if queue.is_running(folder):
                    return self.send_json({"error": "job is running; stop first"}, 409)
                queue.move(folder, "removed")
                return self.send_json({"ok": True})
            if path == "/api/jobs/requeue":
                for status in STATUSES[1:]:
                    folder = job_folder(status, data["name"])
                    if folder.is_dir():
                        queue.move(folder, "pending")
                        return self.send_json({"ok": True})
                return self.send_json({"error": "job folder not found"}, 404)
            if path == "/api/settings":
                (data.get("loras") or {}).pop("civitai_api_key", None)  # keys go through /api/keys
                save_config(data)
                return self.send_json(public_settings())
            if path == "/api/preprompts":
                # {key, text}: the prompt writer's guidelines for one model; text that is empty or
                # the default goes back to the default.
                from . import preprompts
                key = data.get("key") or ""
                if key not in {k for k, _ in preprompts.MODELS}:
                    return self.send_json({"error": "unknown model"}, 400)
                text = (data.get("text") or "").strip()
                keep = text if text and text != preprompts.default(key).strip() else None
                save_config({"prompt_writer": {"preprompts": {key: keep}}})
                return self.send_json({"models": preprompts.listing(load_config())})
            if path == "/api/workflow/checkpoint":
                name = (data.get("name") or "").strip()
                if not name:
                    return self.send_json({"error": "no checkpoint given"}, 400)
                from .workflow import Workflows
                Workflows.set_default(ROOT / "workflows", "checkpoint", name)
                return self.send_json(checkpoints())
            if path == "/api/keys":
                keys.save(data["name"], data.get("value", ""))
                return self.send_json({"api_keys": keys.status()})
            if path == "/api/comfyui/start":
                comfy_launcher(load_config()).start_async()
                return self.send_json({"ok": True})
            if path == "/api/comfyui/stop":
                if runner.running:
                    return self.send_json({"error": "the queue is running; stop it first"}, 409)
                if any(g["status"] == "running" for g in generate_mod.queue_status()):
                    return self.send_json({"error": "a generation is rendering; cancel it on the Queue tab first"}, 409)
                return self.send_json({"stopped": comfy_launcher(load_config()).stop()})
            if path == "/api/loras/refresh":
                return self.send_json({"started": refresh_loras()})
            if path == "/api/poses":
                name = add_pose(data.get("name") or "", data["image_b64"], data.get("fallback") or "")
                if organize.clean_folder(data.get("folder")):
                    organize.update(library_dir("pose"), [name], folder=data["folder"])
                return self.send_json({"name": name})
            if path == "/api/poses/remove":
                pose_library().remove(data["name"])
                organize.forget(library_dir("pose"), data["name"])
                return self.send_json({"ok": True})
            if path in REF_ROUTES:
                kind = REF_ROUTES[path]
                name = add_ref(kind, data.get("name") or "", data["image_b64"], data.get("fallback") or "")
                if organize.clean_folder(data.get("folder")):
                    organize.update(library_dir(kind), [name], folder=data["folder"])
                return self.send_json({"name": name})
            if path.removesuffix("/remove") in REF_ROUTES and path.endswith("/remove"):
                kind = REF_ROUTES[path.removesuffix("/remove")]
                ref_library(kind).remove(data["name"])
                organize.forget(library_dir(kind), data["name"])
                return self.send_json({"ok": True})
            if path in LIBRARY_ROUTES:
                try:
                    return self.send_json(organize_library(LIBRARY_ROUTES[path], data))
                except ValueError as e:
                    return self.send_json({"error": str(e)}, 400)
            if path in DESIGN_ROUTES:
                return self.send_json(DESIGN_ROUTES[path](data))
            if path == "/api/runs/refine-hands":
                return self.send_json(refine_hands_run(data["run"]))
            if path == "/api/runs/remove":
                remove_run(data["run"])
                return self.send_json({"ok": True})
            if path == "/api/runs/empty-bin":
                return self.send_json({"ok": True, "deleted": empty_bin()})
            return self.send_json({"error": "not found"}, 404)
        except KeyError as e:
            return self.send_json({"error": f"missing field {e}"}, 400)
        except Exception as e:
            return self.send_json({"error": str(e)}, 500)


DESIGN_ROUTES = {
    "/api/designs": lambda d: {"design": create_design(d)},
    "/api/designs/update": lambda d: {"design": designer.update(ROOT, d["id"], name=d.get("name"), cover=d.get("cover"))},
    "/api/designs/remove": lambda d: designer.remove(ROOT, d["id"]) or {"ok": True},
    "/api/designs/edit": design_edit,
    "/api/designs/continue": design_continue,
    "/api/designs/keep": design_keep,
    "/api/designs/catalog/remove": lambda d: designer.remove_from_catalog(ROOT, d["id"], d["vid"]) or {"ok": True},
    "/api/designs/session/remove": lambda d: design_session_remove(d),
    "/api/designs/version/rename": lambda d: designer.rename_version(ROOT, d["id"], d["image"], d.get("name")) and {"ok": True},
    "/api/designs/merge": lambda d: design_merge(d),
}


def design_session_remove(data: dict) -> dict:
    if f"{data['id']}/{data['session']}" in generate_mod.design_busy():
        raise ValueError("that edit is still running or queued: stop it first")
    designer.remove_session(ROOT, data["id"], data["session"])
    return {"ok": True}


def design_merge(data: dict) -> dict:
    """Character data["from"] moved into data["into"] (see designer.merge)."""
    busy = {b.split("/")[0] for b in generate_mod.design_busy()}
    if data["from"] in busy or data["into"] in busy:
        raise ValueError("one of them has an edit running or queued: wait for it or stop it first")
    _backfill_noise_origin(data["from"])  # its image keeps how it was drawn
    return {"design": designer.merge(ROOT, data["into"], data["from"])}


def prepare_comfy() -> None:
    """Make sure ComfyUI answers before a queued generation runs (starting it if allowed)."""
    cfg = load_config()
    launcher = comfy_launcher(cfg)
    if cfg.get("comfyui", {}).get("autostart", True):
        if not launcher.ensure_running(lambda m: None):
            raise RuntimeError(f"ComfyUI is not running: {launcher.message}")
    elif not launcher.reachable():
        raise RuntimeError("ComfyUI is not running (and starting it is switched off in Settings)")


def fix_target(run: str) -> dict:
    """What auto-fix needs for a History entry: a Home generation (manual/<run>) or an
    automatic run (its best.png, with the settings that made it)."""
    if run.startswith("manual/"):
        return generate_mod.manual_fix_target(ROOT, run)
    from .params import GenParams
    d = (ROOT / "runs" / run).resolve()
    if d.parent != (ROOT / "runs").resolve() or not (d / "best.png").exists():
        raise FileNotFoundError(run)
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    info = json.loads((d / "run.json").read_text(encoding="utf-8")) if (d / "run.json").exists() else {}
    rep = reproduce_info(d, summary, info)
    if not rep or not rep.get("params"):
        raise RuntimeError("this run has no recorded settings to repaint with")
    p = dict(rep["params"])
    loras = [(l["name"], l["strength"]) for l in rep.get("loras") or []] or p.get("loras") or []
    params = GenParams(**{**p, "loras": tuple(tuple(x) for x in loras)})
    return {"dir": d, "params": params, "positive": rep.get("rendered_positive") or p["positive"],
            "checkpoint": rep.get("checkpoint"), "images": ["best.png"]}


def autofix_view(d: Path) -> dict:
    """A run's auto-fix results for History: {source image: {state, fixed URL, ...}}."""
    out = {}
    for name, r in autofix.load_results(d).items():
        out[name] = {"state": r.get("state"), "error": r.get("error"),
                     "fixed_url": rel_url(d / r["image"]) if r.get("image") and (d / r["image"]).exists() else None,
                     "fixed_thumb": thumbs.url(ROOT, d / r["image"]) if r.get("image") and (d / r["image"]).exists() else None,
                     "issues_found": r.get("issues_found") or [], "issues_left": r.get("issues_left") or [],
                     "rounds": r.get("rounds") or [], "finished": r.get("finished")}
    return out


def upscale_view(d: Path) -> dict:
    """A run's upscale results for History: {image name: {state, upscaled URL, size, ...}}."""
    out = {}
    for name, r in upscale_mod.load_results(d).items():
        f = d / r["image"] if r.get("image") else None
        ok = bool(f and f.exists())
        out[name] = {"state": r.get("state"), "error": r.get("error"), "source": r.get("source"),
                     "upscaled_url": rel_url(f) if ok else None, "upscaled_thumb": thumbs.url(ROOT, f) if ok else None,
                     "size": r.get("size"), "from_size": r.get("from_size"), "model": r.get("model"),
                     "seconds": r.get("seconds"), "finished": r.get("finished")}
    return out


def nobg_view(d: Path) -> dict:
    """A run's background removals for History: {image name: {state, URL, source, ...}}."""
    out = {}
    for name, r in nobg_mod.load_results(d).items():
        f = d / r["image"] if r.get("image") else None
        ok = bool(f and f.exists())
        out[name] = {"state": r.get("state"), "error": r.get("error"), "source": r.get("source"),
                     "nobg_url": rel_url(f) if ok else None, "nobg_thumb": thumbs.url(ROOT, f) if ok else None,
                     "seconds": r.get("seconds"), "finished": r.get("finished")}
    return out


def check_nobg_model() -> None:
    """Fail at once, rather than in the queue, when ComfyUI is up but has no remover model."""
    from .comfy import ComfyClient
    cfg = load_config()
    comfy = ComfyClient(cfg["comfy_url"])
    try:
        requests.get(f"{cfg['comfy_url']}/system_stats", timeout=2).raise_for_status()
    except requests.RequestException:
        return  # not running yet: the queued task starts it and checks then
    nobg_mod.pick_model(cfg, nobg_mod.available_models(comfy))


generate_mod.configure(root=ROOT, rel_url=rel_url, load_config=load_config, prepare=prepare_comfy,
                       fix_target=fix_target, lora_cards=lora_cards)


def open_browser_quietly(url: str) -> None:
    """Open the page without the browser writing into this terminal. A browser started
    from here inherits its output, so e.g. LibreWolf/Firefox from Flatpak printed
    "Sandbox: CanCreateUserNamespace() clone() failure: EPERM" (harmless: Flatpak doesn't
    allow nested user namespaces, so it uses Flatpak's own sandbox instead)."""
    import shutil
    import subprocess
    import sys
    opener = shutil.which("xdg-open") if sys.platform.startswith("linux") else None
    if opener:
        try:
            subprocess.Popen([opener, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             stdin=subprocess.DEVNULL, start_new_session=True)
            return
        except OSError:
            pass
    webbrowser.open(url)


def serve(open_browser: bool = True) -> None:
    web = load_config().get("web", {})
    host = web.get("host", "127.0.0.1")
    port = int(os.environ.get("OUROBOROS_PORT") or web.get("port", 8765))  # env: run a second copy
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}"
    print(f"Ouroboros UI at {url}  (Ctrl+C to quit)")
    cfg = load_config()
    if cfg.get("comfyui", {}).get("autostart", True):
        # Warm ComfyUI up now (hidden) so it's ready when the queue starts.
        comfy_launcher(cfg).start_async(lambda m: print("[ComfyUI]", m))
    if open_browser:
        threading.Timer(0.5, open_browser_quietly, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
