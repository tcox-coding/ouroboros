"""Text -> inpaint mask, in the format the workflow already uses.

The workflow takes its inpaint mask from the LoadImage alpha channel (transparent =
repaint), like a mask painted in ComfyUI's mask editor. So: run CLIPSeg in ComfyUI on
the image for the planner's mask_target ("left hand"), threshold and grow the mask
here, and save the image with that region made transparent.
"""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .comfy import ComfyClient


class EmptyMask(RuntimeError):
    pass


def clipseg_mask(comfy: ComfyClient, image_name: str, text: str) -> Image.Image:
    graph = {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "CLIPSeg Masking", "inputs": {"image": ["1", 0], "text": text}},
        "3": {"class_type": "PreviewImage", "inputs": {"images": ["2", 1]}},
    }
    pid = comfy.queue(graph)
    data = comfy.fetch_images(comfy.wait([pid])[pid], "3")
    if not data:
        raise RuntimeError("CLIPSeg returned no mask image")
    return Image.open(BytesIO(data[0])).convert("L")


def keep_side(mask: Image.Image, text: str) -> Image.Image:
    """CLIPSeg can't tell left from right: "left hand" finds both hands. If the
    target names a side (as seen by the viewer), keep only that side of the mask,
    split at the widest empty column gap. With no gap there's one region (e.g. CLIPSeg
    found only one hand), and it's kept whole: halving it would repaint half a hand."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    side = "left" if "left" in words else "right" if "right" in words else None
    if side is None or not mask.getbbox():
        return mask
    m = np.array(mask) > 0
    cols = np.flatnonzero(m.any(axis=0))
    gaps = np.flatnonzero(np.diff(cols) > 1)
    if not len(gaps):
        return mask
    widest = gaps[np.argmax(np.diff(cols)[gaps])]
    split = (cols[widest] + cols[widest + 1]) // 2
    if side == "left":
        m[:, split:] = False
    else:
        m[:, :split] = False
    return Image.fromarray((m * 255).astype(np.uint8), "L")


def masked_image(comfy: ComfyClient, source: Path, source_name: str, text: str, out: Path,
                 threshold: float = 0.35, grow_px: int = 12) -> Path:
    """Write source with the region matching `text` made transparent; returns out."""
    img = Image.open(source).convert("RGBA")
    mask = clipseg_mask(comfy, source_name, text).resize(img.size, Image.BILINEAR)
    mask = mask.point(lambda v: 255 if v >= threshold * 255 else 0)
    mask = keep_side(mask, text)
    if grow_px > 0:
        mask = mask.filter(ImageFilter.MaxFilter(grow_px * 2 + 1))
    if not mask.getbbox():
        raise EmptyMask(f"CLIPSeg found nothing for '{text}'")
    alpha = Image.eval(mask, lambda v: 255 - v)
    img.putalpha(alpha)
    img.save(out)
    return out
