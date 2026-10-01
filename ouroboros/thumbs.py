"""Small JPEG copies of run images (for History) and of the LoRA classifier's example
renders (for the LoRA detail pane).

A History card shows up to four full-size PNGs (1-2 MB each), so a page of runs used to
pull tens of megabytes. Each image is shrunk once, the first time it is asked for, and
kept in cache/thumbs/ under the same relative path (runs/manual/<run>/image_01.png ->
cache/thumbs/runs/manual/<run>/image_01.jpg). A thumbnail older than its image (a run's
best.png is rewritten while it improves) is made again.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from PIL import Image

from .sizes import to_rgb

MAX_SIDE = 384  # History cards are ~190 px wide; twice that stays sharp on hi-dpi screens
QUALITY = 82
EXAMPLES = "lora_examples"
_locks: dict[Path, threading.Lock] = {}
_locks_guard = threading.Lock()


def url(root: Path, image: Path) -> str | None:
    """/thumbs/... URL for an image under runs/, versioned by the image's mtime so the
    browser can cache it for good and still sees a rewritten best.png."""
    try:
        rel = image.resolve().relative_to(root.resolve()).as_posix()
        return f"/thumbs/{rel}?v={int(image.stat().st_mtime)}"
    except (OSError, ValueError):
        return None


def get(root: Path, rel: str) -> Path | None:
    """The cached thumbnail for runs/<...> (made now if missing or stale), or None."""
    source = (root / rel).resolve()
    try:
        source.relative_to((root / "runs").resolve())
    except ValueError:
        return None
    if not source.is_file():
        return None
    # Named after the resolved path, not `rel`: "runs/../../x.png" must not write outside the cache.
    return _make(source, cache_dir(root) / source.relative_to(root.resolve()).with_suffix(".jpg"))


def lora_example(root: Path, source: Path, lora_id: str) -> Path | None:
    """Thumbnail of one of a LoRA's example renders (they live in the lora-classifier's
    output, outside this project), kept under cache/thumbs/lora_examples/<lora id>/."""
    if not source.is_file() or not lora_id or "/" in lora_id or lora_id.startswith("."):
        return None
    return _make(source, cache_dir(root) / EXAMPLES / lora_id / (source.stem + ".jpg"))


def _make(source: Path, out: Path) -> Path:
    """Shrink source into out, unless out is already there and newer than source."""
    with _lock_for(out):
        if out.exists() and out.stat().st_mtime >= source.stat().st_mtime:
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as img:
            img = to_rgb(img)
            img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
            tmp = out.with_name(out.name + ".tmp")
            img.save(tmp, "JPEG", quality=QUALITY, optimize=True)
        os.replace(tmp, out)  # never serve a half-written file
    return out


def prune(root: Path) -> int:
    """Delete thumbnails whose image is gone (e.g. after emptying the recycle bin)."""
    base = cache_dir(root)
    removed = 0
    for thumb in base.rglob("*.jpg") if base.is_dir() else []:
        rel = thumb.relative_to(base)
        if rel.parts[0] == EXAMPLES:
            continue  # their sources are outside the project; the catalog decides those
        if not any((root / rel.with_suffix(ext)).exists() for ext in (".png", ".jpg", ".jpeg", ".webp")):
            thumb.unlink(missing_ok=True)
            removed += 1
    for d in sorted((p for p in base.rglob("*") if p.is_dir()), reverse=True) if base.is_dir() else []:
        try:
            d.rmdir()  # only succeeds when empty
        except OSError:
            pass
    return removed


def cache_dir(root: Path) -> Path:
    return root / "cache" / "thumbs"


def _lock_for(path: Path) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(path, threading.Lock())
