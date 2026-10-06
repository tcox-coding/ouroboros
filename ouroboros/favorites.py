"""Favourite History entries: {run: {"name", "added"}} in <root>/favorites.json.

A favourite points at its run folder (runs/<run> or runs/manual/<run>), which holds every
setting, prompt, reference and image of the run, so nothing is copied. A favourite can't be
deleted from History until it is taken off the Favorites page.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

_lock = threading.Lock()


def _file(root: Path) -> Path:
    return Path(root) / "favorites.json"


def load(root: Path) -> dict:
    try:
        data = json.loads(_file(root).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(root: Path, data: dict) -> None:
    tmp = _file(root).with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(_file(root))


def run_dir(root: Path, run: str) -> Path:
    """The folder of a History entry ("manual/<run>" or a loop run), or FileNotFoundError."""
    runs = (Path(root) / "runs").resolve()
    manual = run.startswith("manual/")
    leaf = run.removeprefix("manual/")
    d = (runs / ("manual" if manual else "") / leaf).resolve()
    if (d.parent != (runs / "manual" if manual else runs) or not d.is_dir() or not leaf
            or leaf.startswith("_") or leaf == "manual"):
        raise FileNotFoundError(f"no History entry {run}")
    return d


def set(root: Path, run: str, name: str | None = None) -> dict:  # noqa: A001
    """Favourite `run`, or rename it (None keeps the current name; "" clears it, so the
    page shows the run's own title)."""
    run_dir(root, run)
    with _lock:
        data = load(root)
        entry = data.get(run) or {"name": "", "added": time.strftime("%Y-%m-%d %H:%M:%S")}
        if name is not None:
            entry["name"] = name.strip()[:120]
        data[run] = entry
        _save(root, data)
        return entry


def remove(root: Path, run: str) -> None:
    with _lock:
        data = load(root)
        if data.pop(run, None) is not None:
            _save(root, data)
