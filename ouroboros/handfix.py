"""Hand refiner: repaint each hand, guided by a clean hand skeleton.

The HandRefiner ControlNet itself is SD 1.5 only (a ControlNet works only with the model
family it was trained on), so this does the same thing natively in SDXL:

1. DWPose finds each hand and fits it with a 21-point hand skeleton, which always has
   five fingers, even when the drawn hand has six or fused ones.
2. The hand's box, padded for the wrist and some context, is made transparent. The
   workflow's masked path (Inpaint Crop -> sample at 1024x1024 -> Inpaint Stitch)
   repaints only that region, at a far higher working resolution than the hand has in
   the full image.
3. The Union ControlNet in openpose mode follows that hand's skeleton, drawn at full
   size and cropped with the same mask, so the new hand has the fitted finger layout.

Hands are repainted one at a time, each pass starting from the previous result.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from . import pose
from .params import GenParams, edit_prompt
from .sizes import to_rgb

HAND_POSITIVE = ["detailed hands", "five fingers", "well-drawn fingers"]
HAND_NEGATIVE = ["extra fingers", "fused fingers", "missing fingers", "malformed hands", "bad hands",
                 "extra digits"]


def hand_mask_image(image: Path, box: tuple, out: Path, feather: int = 6) -> Path:
    """`image` with an ellipse over the box made transparent (the workflow repaints
    transparent pixels). An ellipse follows a hand better than a rectangle does."""
    img = to_rgb(Image.open(image)).convert("RGBA")
    alpha = Image.new("L", img.size, 255)
    ImageDraw.Draw(alpha).ellipse(box, fill=0)
    if feather:
        alpha = alpha.filter(ImageFilter.GaussianBlur(feather)).point(lambda v: 0 if v < 128 else 255)
    img.putalpha(alpha)
    img.save(out)
    return out


def refine_hands(image: Path, params: GenParams, *, comfy, flows, cfg: dict, checkpoint: str | None,
                 positive: str, out_dir: Path, upload, log=print, tag: str = "hands") -> dict:
    """Repaint every confidently found hand in `image`. Returns {"image": final path or
    None, "hands": number repainted, "found": number found, "steps": [...]}. `positive`
    is the prompt as rendered (with LoRA triggers); `upload(path)` -> ComfyUI name."""
    hc = cfg.get("hands", {})
    cn = cfg.get("controlnet", {})
    size = Image.open(image).size
    found = pose.detect(image)
    boxes = pose.hand_boxes(found, size, int(hc.get("min_points", 12)), float(hc.get("pad", 0.3)),
                            float(hc.get("min_confidence", pose.HAND_MIN_MEAN))) if found else {}
    result = {"found": len(boxes), "hands": 0, "image": None, "steps": [], "source": image.name}
    if not boxes:
        log("hand refine: no hands found clearly enough to repaint")
        return result
    p = replace(params, mode="inpaint_best", mask_target="hands", loras=params.loras,
                denoise=float(hc.get("denoise", 0.6)), seed=random.randrange(2**32),
                negative=edit_prompt(params.negative, HAND_NEGATIVE, []))
    pos = edit_prompt(positive, HAND_POSITIVE, [])
    current = image
    for n, (hand, box) in enumerate(sorted(boxes.items()), 1):
        stem = f"{tag}_{n}_{hand}"
        masked = hand_mask_image(current, box, out_dir / f"{stem}_mask.png")
        skeleton = out_dir / f"{stem}_skeleton.png"
        pose.render(found, size, only_hand=hand).save(skeleton)
        control = {"model": cn.get("model", "xinsir_union_sdxl_promax.safetensors"), "image": upload(skeleton),
                   "type": "openpose", "strength": float(hc.get("control_strength", 0.8)),
                   "start": 0.0, "end": float(hc.get("control_end", 1.0))}
        graph = flows.build(p, upload(masked), 1, checkpoint, pos, size, control)
        (out_dir / f"{stem}_graph.json").write_text(json.dumps(graph), encoding="utf-8")
        t0 = time.monotonic()
        pid = comfy.queue(graph)
        data = comfy.fetch_images(comfy.wait([pid])[pid], flows.output_node)
        if not data:
            log(f"hand refine: no image back for the {hand.replace('_', ' ')}")
            continue
        current = out_dir / f"{stem}.png"
        current.write_bytes(data[0])
        result["hands"] += 1
        result["steps"].append({"hand": hand, "box": list(box), "image": current.name, "mask": masked.name,
                                "skeleton": skeleton.name, "seconds": round(time.monotonic() - t0, 1)})
        log(f"hand refine: repainted the {hand.replace('_', ' ')} (box {box[2] - box[0]}x{box[3] - box[1]} px)")
    if result["hands"]:
        result["image"] = current
        result["params"] = {**p.to_dict(), "positive": pos}
    return result
