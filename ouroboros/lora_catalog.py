"""The classified LoRA library, as cards for the picker.

Reads the lora-classifier's output (loras.catalog_dir): one state file per LoRA holding
the name ComfyUI loads it by, what it does, its trigger words and the weight range its
own test renders supported. That is everything a card needs, so nothing here re-derives
it from the .safetensors files.

Only entries the classifier finished and cleared (status "done") are offered. The ones it
marked "caution" or left "pending" are the ones its age screener flagged, whose images it
deleted; they have no usable preview or description and are not shown.

Card images come from Civitai, fetched on first view and cached under cache/lora_thumbs/
(nothing is downloaded until a card is actually looked at). Images flagged `minor`, or
above loras.civitai_max_nsfw_level, are skipped; when a LoRA has no usable Civitai image
the classifier's own example render is used instead.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

CACHE_NAME = "lora_catalog.json"
THUMBS = "lora_thumbs"
# Bump when an entry gains or changes a field, so the cache is rebuilt instead of
# silently serving records that are missing it.
SCHEMA = 2


def _leaf(path: list) -> str:
    return path[-1] if path else ""


def _entry(state: dict) -> dict | None:
    meta = state.get("meta") or {}
    comfy = meta.get("comfy_name")
    if state.get("status") != "done" or not comfy:
        return None
    # The evaluation stage saw the example renders; the analysis stage only the probes.
    ev = (state.get("stages") or {}).get("evaluation") or {}
    an = (state.get("stages") or {}).get("analysis") or {}
    best = ev if ev.get("summary") else an
    civitai = meta.get("civitai") or {}
    weight = best.get("recommended_weight") or {}
    cats = best.get("category_path") or an.get("category_path") or []
    triggers = [t for t in (best.get("trigger_words") or meta.get("triggers") or []) if t]
    tags = [t for t in (_leaf(cats), an.get("lora_type"), best.get("nsfw_level")) if t]
    return {
        "id": meta["id"],
        "comfy_name": comfy,
        "file": meta.get("filename", ""),
        "title": civitai.get("model_name") or meta.get("title") or meta.get("filename", ""),
        "summary": (best.get("summary") or "").strip(),
        "category": "/".join(cats),
        "type": an.get("lora_type") or "",
        "tags": tags,
        "triggers": triggers,
        "weight": {"min": weight.get("min", 0.2), "default": weight.get("default", 0.8),
                   "max": weight.get("max", 1.2)},
        "scores": best.get("scores") or {},
        "nsfw": best.get("nsfw_level") or "",
        "sha256": meta.get("sha256") or "",
        "base_model": civitai.get("base_model") or meta.get("base_model") or "",
        "example_prompt": next((i.get("prompt", "") for i in (civitai.get("images") or []) if i.get("prompt")), ""),
        "civitai_url": (f"https://civitai.com/models/{civitai['model_id']}"
                        if civitai.get("model_id") else ""),
    }


def _examples(root: Path, lora_id: str) -> list[str]:
    d = root / "catalog" / "examples" / lora_id
    if not d.is_dir():
        return []
    return sorted(str(p) for p in d.glob("example_*.png"))[:3]


def catalog(root: Path, cache_dir: Path, refresh: bool = False) -> list[dict]:
    """Every usable LoRA, newest classification first. Cached: rebuilt only when a state
    file has changed, since this reads ~570 JSON files."""
    states = sorted((root / "state").glob("*.json"))
    cache = cache_dir / CACHE_NAME
    newest = max((p.stat().st_mtime for p in states), default=0.0)
    if cache.exists() and not refresh:
        try:
            saved = json.loads(cache.read_text(encoding="utf-8"))
            if (saved.get("schema") == SCHEMA and saved.get("newest") == newest
                    and saved.get("count") == len(states)):
                return _with_briefs(saved["items"], cache_dir, root)
        except (OSError, ValueError, KeyError):
            pass
    items = []
    for p in states:
        try:
            e = _entry(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
        if e:
            e["examples"] = _examples(root, e["id"])
            items.append(e)
    items.sort(key=lambda e: e["title"].lower())
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"schema": SCHEMA, "newest": newest, "count": len(states),
                                 "built": time.time(), "items": items}, indent=1), encoding="utf-8")
    return _with_briefs(items, cache_dir, root)


_files: dict = {"root": None, "at": 0.0, "names": {}}


def installed_names(loras_root: Path) -> dict[str, str]:
    """file name (lower case) -> the name ComfyUI knows it by (its path below the loras
    root, backslash-separated like the classifier writes it). Rescanned every minute."""
    if _files["root"] != loras_root or time.time() - _files["at"] > 60:
        names = {}
        for f in loras_root.rglob("*.safetensors") if loras_root.is_dir() else []:
            names.setdefault(f.name.lower(), str(f.relative_to(loras_root)).replace("/", "\\"))
        _files.update(root=loras_root, at=time.time(), names=names)
    return _files["names"]


def current_name(name: str, loras_root: Path) -> str:
    """Where a LoRA is now. The classifier moves LoRAs as it re-sorts the library (each
    now in its own folder), and its records, saved workflows and older runs keep the old
    path; ComfyUI silently skips a LoRA it can't find. File names are unique, so the
    file name finds it."""
    fname = name.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return installed_names(loras_root).get(fname, name)


def _with_briefs(items: list[dict], cache_dir: Path, root: Path | None = None) -> list[dict]:
    """Each LoRA at its current path, with its short description for LLMs
    (lora_briefs.py) if one was written. A LoRA whose brief says it's made to depict
    minors is left out entirely."""
    from .lora_briefs import load
    briefs = load(cache_dir)
    out = []
    for e in items:
        b = briefs.get(e["id"])
        if b and b.get("minors"):
            continue
        if root is not None:
            e = {**e, "comfy_name": current_name(e["comfy_name"], root / "loras")}
        out.append({**e, "brief": b["text"]} if b and b.get("text") else e)
    return out


def by_name(root: Path, cache_dir: Path) -> dict[str, dict]:
    """ComfyUI lora name -> card, for showing what a run used."""
    return {e["comfy_name"]: e for e in catalog(root, cache_dir)}


def records(root: Path, cache_dir: Path) -> dict[str, dict]:
    """The catalog in the shape LoraLibrary.index() returns.

    The loop's picker, the judge's LoRA menu and the prompt writer all read the library
    through that shape, so producing it here is what lets them use the classified
    catalog - one library instead of two that disagree about what is installed.
    """
    out = {}
    for e in catalog(root, cache_dir):
        examples = [{"thumb": p, "path": p, "prompt": e.get("example_prompt", "")}
                    for p in e.get("examples") or []]
        out[e["comfy_name"]] = {
            "name": e["comfy_name"], "title": e["title"], "base_model": e.get("base_model") or None,
            "trigger_words": list(e["triggers"]), "training_tags": [], "tags": list(e["tags"]),
            "description": e["summary"], "examples": examples,
            "typical_weight": (e["weight"] or {}).get("default"),
            "civitai_url": e.get("civitai_url"), "source": "catalog", "error": None,
            "weight_range": e.get("weight"), "type": e.get("type", ""), "category": e.get("category", ""),
            "scores": e.get("scores") or {}, "id": e["id"], "brief": e.get("brief", ""),
        }
    return out


_warming = False


def warm(root: Path, cache_dir: Path, max_nsfw: int = 16, workers: int = 4) -> None:
    """Fetch the card images that aren't cached yet, in the background.

    Without this the picker asks for ~500 images at once on first open, each needing a
    round trip to Civitai; the browser runs out of connections and most cards stay blank.
    Warming once makes every later open instant.
    """
    global _warming
    if _warming:
        return
    _warming = True

    def run():
        global _warming
        try:
            from concurrent.futures import ThreadPoolExecutor
            todo = [e for e in catalog(root, cache_dir)
                    if not (cache_dir / THUMBS / f"{e['id']}.jpg").exists()]
            with ThreadPoolExecutor(workers) as pool:
                list(pool.map(lambda e: thumbnail(root, cache_dir, e, max_nsfw), todo))
        except Exception:
            pass
        finally:
            _warming = False

    import threading

    threading.Thread(target=run, daemon=True, name="lora-thumbs").start()


def _civitai_image(root: Path, sha: str, max_nsfw: int) -> str | None:
    p = root / "cache" / "civitai" / f"{sha}.json"
    if not sha or not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for img in data.get("images") or []:
        if (img.get("type") == "image" and img.get("url") and not img.get("minor")
                and (img.get("nsfwLevel") or 0) <= max_nsfw):
            return img["url"]
    return None


def thumbnail(root: Path, cache_dir: Path, entry: dict, max_nsfw: int = 16,
              max_side: int = 320) -> Path | None:
    """The card image for one LoRA, downloading it the first time it is asked for.

    Civitai's showcase image when one passes the filters, otherwise the classifier's own
    example render. Returns None when the LoRA has neither.
    """
    from PIL import Image

    from .sizes import to_rgb

    out = cache_dir / THUMBS / f"{entry['id']}.jpg"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    raw: bytes | None = None
    url = _civitai_image(root, entry.get("sha256", ""), max_nsfw)
    if url:
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                raw = r.content
        except requests.RequestException:
            raw = None
    if raw is None:
        local = root / "catalog" / "examples" / entry["id"] / "example_01.png"
        if not local.exists():
            return None
        raw = local.read_bytes()
    try:
        import io

        img = to_rgb(Image.open(io.BytesIO(raw)))
        img.thumbnail((max_side, max_side))
        img.save(out, "JPEG", quality=85)
    except Exception:
        return None
    return out
