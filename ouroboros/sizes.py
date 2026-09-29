"""Output size: the SDXL resolution closest to the reference image's shape.

SDXL is trained on about one megapixel in these aspect-ratio buckets; other sizes
work less well. The reference is centre-cropped to the chosen bucket's exact shape
and resized to it, so txt2img, img2img (which encodes the reference itself) and
masked repaints all come out the same size.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image

# ~1 megapixel each, all multiples of 64: square, 4:3, 3:2, 16:9 and 21:9, wide and tall.
SDXL_SIZES = [(1024, 1024), (1152, 896), (896, 1152), (1216, 832), (832, 1216),
              (1344, 768), (768, 1344), (1536, 640), (640, 1536)]


def check_size(size: tuple[int, int]) -> tuple[tuple[int, int], str | None]:
    """A custom size rounded to multiples of 8 (the latent is 1/8 scale), plus a
    warning when it's far from ~1 megapixel (anatomy problems, doubled subjects)."""
    w, h = (max(8, round(v / 8) * 8) for v in size)
    mp = w * h / 1e6
    warn = None
    if not 0.75 <= mp <= 1.3:
        warn = (f"{w}x{h} is {mp:.2f} megapixels; SDXL works best near 1 megapixel "
                f"(closest standard size: {'x'.join(map(str, closest_size(w, h)))})")
    return (w, h), warn


def closest_size(width: int, height: int, sizes=SDXL_SIZES) -> tuple[int, int]:
    """The size whose aspect ratio is nearest (on a log scale, so 2:1 and 1:2 are
    equally far from square)."""
    target = math.log(width / height)
    return min(sizes, key=lambda s: abs(math.log(s[0] / s[1]) - target))


def parse_size(value: str | None) -> tuple[int, int] | None:
    """ "1344x768" -> (1344, 768); "auto", "workflow" or blank -> None."""
    if not value or not isinstance(value, str) or "x" not in value.lower():
        return None
    w, h = value.lower().split("x", 1)
    return int(w), int(h)


def output_size(setting: str | None, reference: Path, workflow_size: tuple[int, int] | None) -> tuple[int, int]:
    """setting: "auto" (match the reference's shape), "workflow" (as saved in the
    workflow), or "WxH"."""
    explicit = parse_size(setting)
    if explicit:
        return check_size(explicit)[0]
    if setting == "workflow" and workflow_size:
        return workflow_size
    with Image.open(reference) as im:
        return closest_size(*im.size)


def to_rgb(img: Image.Image, background: str = "white") -> Image.Image:
    """RGB copy with any transparency flattened onto `background`. A plain
    convert("RGB") shows whatever colour transparent pixels happen to hold (often
    black), so a drawing on a transparent background would turn into one on black."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        flat = Image.new("RGB", rgba.size, background)
        flat.paste(rgba, mask=rgba.getchannel("A"))
        return flat
    return img.convert("RGB")


def fit_to(source: Path, size: tuple[int, int], out: Path) -> Path:
    """Centre-crop `source` to size's aspect ratio, resize to size, save as an RGB PNG.
    Transparency is flattened onto white: the workflow reads a LoadImage alpha channel
    as an inpaint mask, and img2img would otherwise start from black."""
    w, h = size
    with Image.open(source) as im:
        im.load()
        sw, sh = im.size
        if sw * h > sh * w:  # too wide: trim the sides
            nw = round(sh * w / h)
            box = ((sw - nw) // 2, 0, (sw - nw) // 2 + nw, sh)
        else:  # too tall: trim top and bottom
            nh = round(sw * h / w)
            box = (0, (sh - nh) // 2, sw, (sh - nh) // 2 + nh)
        to_rgb(im).crop(box).resize((w, h), Image.LANCZOS).save(out)
    return out
