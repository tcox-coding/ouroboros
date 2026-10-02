"""Upscale: make a finished image larger and sharper with the checkpoint that made it.

Two steps, both in ComfyUI:

1. Enlarge the pixels: an upscale model (e.g. 4x-AnimeSharp, set in upscale.model) and then
   a Lanczos resize to the target size, or only the Lanczos resize when no model is set.
2. Add detail: a light img2img pass at the new size through the user's own workflow, with
   the image's own checkpoint, LoRAs, prompts, sampler and seed (denoise upscale.denoise,
   ~0.3-0.45). This is what "hires fix" does: the enlarged image is redrawn just enough to
   add real detail at the higher resolution, while the LoRAs keep the style.

The workflow's img2img path encodes the loaded image as it is, so the output has the
upscaled image's size. The original is never replaced: the result is saved next to it
(<name>_upscaled.png) and recorded in the run folder's upscale.json (what History shows).
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

from PIL import Image

from .comfy import Cancelled
from .params import GenParams

DEFAULTS = {"model": "", "scale": 1.5, "denoise": 0.35, "max_side": 2560}
SCALE_RANGE = (1.0, 4.0)


def settings(cfg: dict) -> dict:
    return {**DEFAULTS, **(cfg.get("upscale") or {})}


def target_size(size: tuple[int, int], scale: float, max_side: int) -> tuple[int, int]:
    """size x scale, capped so the longest side is at most max_side, in multiples of 8 (the
    latent grid). Never smaller than the original."""
    scale = min(SCALE_RANGE[1], max(SCALE_RANGE[0], float(scale)))
    w, h = size
    scale = min(scale, max(1.0, max_side / max(w, h)))
    return max(8, round(w * scale / 8) * 8), max(8, round(h * scale / 8) * 8)


def model_graph(image_name: str, model: str, size: tuple[int, int]) -> tuple[dict, str]:
    """ComfyUI graph: load -> upscale model -> Lanczos to the exact size -> preview.
    Returns (graph, output node id). PreviewImage writes to ComfyUI's temp folder only."""
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": model}},
        "3": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]}},
        "4": {"class_type": "ImageScale", "inputs": {"image": ["3", 0], "upscale_method": "lanczos",
                                                      "width": size[0], "height": size[1], "crop": "disabled"}},
        "5": {"class_type": "PreviewImage", "inputs": {"images": ["4", 0]}},
    }, "5"


def _run_graph(comfy, graph: dict, node: str, should_stop) -> bytes | None:
    pid = comfy.queue(graph)
    data = comfy.fetch_images(comfy.wait([pid], should_stop=should_stop)[pid], node)
    return data[0] if data else None


def upscale(image: Path, params: GenParams, *, comfy, flows, cfg: dict, checkpoint: str | None, positive: str,
            out_dir: Path, upload, tag: str = "", stage=lambda t: None, should_stop=lambda: False) -> dict:
    """Returns {"image": Path, "size", "from_size", "model", "denoise", "seconds"}."""
    s = settings(cfg)
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = tag or image.stem
    from_size = Image.open(image).size
    size = target_size(from_size, s["scale"], int(s["max_side"]))
    model = (s.get("model") or "").strip()

    def check():
        if should_stop():
            raise Cancelled()

    enlarged = out_dir / f"{stem}_enlarged.png"
    if model:
        stage(f"enlarging with {Path(model).stem}")
        graph, node = model_graph(upload(image), model, size)
        data = _run_graph(comfy, graph, node, should_stop)
        if not data:
            raise RuntimeError("the upscale model returned no image")
        enlarged.write_bytes(data)
    else:
        with Image.open(image) as im:
            im.convert("RGB").resize(size, Image.LANCZOS).save(enlarged)
    check()
    denoise = float(s["denoise"])
    out = out_dir / f"{stem}_upscaled.png"
    if denoise <= 0:  # pixels only, no detail pass
        out.write_bytes(enlarged.read_bytes())
    else:
        stage(f"adding detail at {size[0]}x{size[1]}")
        p = replace(params, mode="img2img_best", mask_target=None, denoise=min(1.0, denoise))
        graph = flows.build(p, upload(enlarged), 1, checkpoint, positive, size)
        (out_dir / f"{stem}_upscale_graph.json").write_text(json.dumps(graph), encoding="utf-8")
        data = _run_graph(comfy, graph, flows.output_node, should_stop)
        if not data:
            raise RuntimeError("the detail pass returned no image")
        out.write_bytes(data)
    return {"image": out, "size": list(size), "from_size": list(from_size), "model": model or None,
            "denoise": denoise, "seconds": round(time.time() - t0, 1)}


# ---- results kept per run folder ------------------------------------------------------------

def results_file(run_dir: Path) -> Path:
    return run_dir / "upscale.json"


def load_results(run_dir: Path) -> dict:
    """{image name: {"state", "image" (upscaled file name or None), "source", "size", ...}}"""
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


def best_version(run_dir: Path, name: str) -> Path:
    """The image to upscale for <name>: its auto-fixed version if one was kept, else itself."""
    from . import autofix
    fixed = (autofix.load_results(run_dir).get(name) or {}).get("image")
    return run_dir / fixed if fixed and (run_dir / fixed).is_file() else run_dir / name


def run_and_record(name: str, params: GenParams, *, run_dir: Path, comfy, flows, cfg, checkpoint, positive: str,
                   upload, stage=lambda t: None, should_stop=lambda: False) -> dict:
    """upscale() on one image of a run (its auto-fixed version if there is one), keeping the
    result as <stem>_upscaled.png next to it and the record in the run's upscale.json."""
    source = best_version(run_dir, name)
    save_result(run_dir, name, {"state": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "error": None, "image": None, "source": source.name})
    try:
        res = upscale(source, params, comfy=comfy, flows=flows, cfg=cfg, checkpoint=checkpoint, positive=positive,
                      out_dir=run_dir / "upscale", upload=upload, tag=source.stem, stage=stage,
                      should_stop=should_stop)
    except Cancelled:
        save_result(run_dir, name, {"state": "cancelled", "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
        raise
    except Exception as e:
        save_result(run_dir, name, {"state": "error", "error": f"{type(e).__name__}: {e}"[:500],
                                    "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
        raise
    final = run_dir / f"{source.stem}_upscaled.png"
    final.write_bytes(Path(res["image"]).read_bytes())
    save_result(run_dir, name, {"state": "done", "image": final.name, "size": res["size"],
                                "from_size": res["from_size"], "model": res["model"], "denoise": res["denoise"],
                                "seconds": res["seconds"], "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    return {**res, "final": final}
