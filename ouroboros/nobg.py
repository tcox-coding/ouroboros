"""Remove the background of a finished image, keeping the character on a transparent PNG.

ComfyUI does the work with its built-in background remover (LoadBackgroundRemovalModel ->
RemoveBackground, ComfyUI 0.37+), which needs a model in its models/background_removal
folder: BiRefNet (birefnet.safetensors, from huggingface.co/Comfy-Org/BiRefNet). The graph
returns the foreground mask as an image; it becomes the alpha channel of the source here.

The source is the image's latest version (upscaled, else auto-fixed, else the original).
Nothing is replaced: the result is saved next to the image as <stem>_nobg.png and recorded
in the run folder's nobg.json, which History shows.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

from PIL import Image

from .comfy import Cancelled, combo_options
from .sizes import to_rgb

DEFAULT_MODEL = "birefnet.safetensors"
MISSING = ("Background removal needs a model in ComfyUI's models/background_removal folder: "
           "birefnet.safetensors from huggingface.co/Comfy-Org/BiRefNet (444 MB).")


def model_name(cfg: dict) -> str:
    return ((cfg.get("background_removal") or {}).get("model") or DEFAULT_MODEL).strip()


def available_models(comfy) -> list[str]:
    """Models ComfyUI can load for background removal ([] if its version has no remover)."""
    import requests
    try:
        info = requests.get(f"{comfy.url}/object_info/LoadBackgroundRemovalModel", timeout=10).json()
        return combo_options(info["LoadBackgroundRemovalModel"]["input"]["required"]["bg_removal_name"])
    except (requests.RequestException, KeyError, ValueError):
        return []


def pick_model(cfg: dict, models: list[str]) -> str:
    """The configured model if ComfyUI has it, else the only one it has; else raise."""
    want = model_name(cfg)
    if want in models:
        return want
    if len(models) == 1:
        return models[0]
    raise RuntimeError(MISSING if not models else f"No background removal model named {want}; "
                       f"ComfyUI has {', '.join(models)} (set background_removal.model in config.json).")


def graph(image_name: str, model: str) -> tuple[dict, str]:
    """load -> remover -> foreground mask -> as an image -> preview. Returns (graph, output node)."""
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": model}},
        "3": {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["2", 0], "image": ["1", 0]}},
        "4": {"class_type": "MaskToImage", "inputs": {"mask": ["3", 0]}},
        "5": {"class_type": "PreviewImage", "inputs": {"images": ["4", 0]}},
    }, "5"


def apply_mask(source: Path, mask_png: bytes, out: Path) -> Path:
    """source with the mask (white = keep) as its alpha channel, saved as an RGBA PNG."""
    with Image.open(source) as im:
        rgb = to_rgb(im)
    with Image.open(io.BytesIO(mask_png)) as m:
        alpha = m.convert("L")
    if alpha.size != rgb.size:
        alpha = alpha.resize(rgb.size, Image.LANCZOS)
    rgba = rgb.convert("RGBA")
    rgba.putalpha(alpha)
    rgba.save(out)
    return out


def latest_version(run_dir: Path, name: str) -> Path:
    """The newest version of <name>: upscaled, else auto-fixed, else itself."""
    from . import autofix, upscale
    for results in (upscale.load_results(run_dir), autofix.load_results(run_dir)):
        f = (results.get(name) or {}).get("image")
        if f and (run_dir / f).is_file():
            return run_dir / f
    return run_dir / name


# ---- results kept per run folder ------------------------------------------------------------

def results_file(run_dir: Path) -> Path:
    return run_dir / "nobg.json"


def load_results(run_dir: Path) -> dict:
    """{image name: {"state", "image" (the _nobg file name or None), "source", ...}}"""
    try:
        return json.loads(results_file(run_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_result(run_dir: Path, name: str, entry: dict) -> None:
    data = load_results(run_dir)
    data[name] = {**data.get(name, {}), **entry}
    tmp = results_file(run_dir).with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(results_file(run_dir))


def run_and_record(name: str, *, run_dir: Path, comfy, model: str, upload, should_stop=lambda: False) -> dict:
    """Remove the background of <name> (its latest version), keeping <stem>_nobg.png."""
    source = latest_version(run_dir, name)
    now = lambda: time.strftime("%Y-%m-%d %H:%M:%S")  # noqa: E731
    save_result(run_dir, name, {"state": "running", "started": now(), "error": None, "image": None,
                                "source": source.name})
    t0 = time.time()
    try:
        g, node = graph(upload(source), model)
        pid = comfy.queue(g)
        data = comfy.fetch_images(comfy.wait([pid], should_stop=should_stop)[pid], node)
        if not data:
            raise RuntimeError("the background remover returned no mask")
        out = apply_mask(source, data[0], run_dir / f"{Path(name).stem}_nobg.png")
    except Cancelled:
        save_result(run_dir, name, {"state": "cancelled", "finished": now()})
        raise
    except Exception as e:
        save_result(run_dir, name, {"state": "error", "error": f"{type(e).__name__}: {e}"[:500], "finished": now()})
        raise
    entry = {"state": "done", "image": out.name, "model": model, "seconds": round(time.time() - t0, 1),
             "finished": now()}
    save_result(run_dir, name, entry)
    return {**entry, "final": out, "source": source.name}


# ---- a background-free copy of a character image, for its IP-Adapter ------------------------
# An IP-Adapter carries the whole image, background included: a character on a flat beige
# backdrop tinted every render peach. The same image cut out onto neutral grey kept the
# character and outfit and lost the cast (same seed, 2026-10-04).
CUTOUT_GREY = (128, 128, 128)


def cutout(image: Path, *, comfy, cfg: dict, cache_dir: Path, upload) -> Path:
    """`image` with its background replaced by neutral grey, cached by the image's content
    in cache_dir (made once per image). Raises if ComfyUI has no background remover."""
    import hashlib
    key = hashlib.sha1(Path(image).read_bytes()).hexdigest()[:20]
    out = Path(cache_dir) / f"{key}.png"
    if out.is_file():
        return out
    model = pick_model(cfg, available_models(comfy))
    g, node = graph(upload(Path(image)), model)
    pid = comfy.queue(g)
    data = comfy.fetch_images(comfy.wait([pid])[pid], node)
    if not data:
        raise RuntimeError("the background remover returned no mask")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + "_rgba.png")
    apply_mask(Path(image), data[0], tmp)
    with Image.open(tmp) as cut:
        flat = Image.new("RGBA", cut.size, CUTOUT_GREY + (255,))
        flat.alpha_composite(cut)
        flat.convert("RGB").save(out)
    tmp.unlink(missing_ok=True)
    return out


# ---- a square copy of a character image, for its IP-Adapter ---------------------------------
# The IP-Adapter's image encoder sees a 224 x 224 square: a portrait is centre-cropped to its
# middle, so a full-body character lost her head and legs. Renders kept the armour (the middle)
# and drifted in face and hair (2026-10-04). Padded to a square, the whole character is seen.

def square_for_ip(image: Path, cache_dir: Path) -> Path:
    """`image` padded to a square with its own backdrop colour (the corners' median; neutral
    grey for a cut-out), cached by content in cache_dir. A square image is returned as is."""
    import hashlib
    from statistics import median
    with Image.open(image) as im:
        rgb = to_rgb(im)
    w, h = rgb.size
    if abs(w - h) <= 2:
        return Path(image)
    key = hashlib.sha1(Path(image).read_bytes()).hexdigest()[:20]
    out = Path(cache_dir) / f"{key}_square.png"
    if out.is_file():
        return out
    px = rgb.load()
    corners = [px[x, y] for x, y in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1))]
    fill = tuple(int(median(c[i] for c in corners)) for i in range(3))
    side = max(w, h)
    sq = Image.new("RGB", (side, side), fill)
    sq.paste(rgb, ((side - w) // 2, (side - h) // 2))
    out.parent.mkdir(parents=True, exist_ok=True)
    sq.save(out)
    return out
