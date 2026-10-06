"""Folders and favourites for the saved poses, styles and characters.

Kept in <library folder>/_library.json, beside the items, so an item's name (what jobs,
History and the pickers refer to) never changes when it is filed somewhere else:

    {"items": {name: {"folder": "anime/female", "favorite": true}},
     "folders": ["anime", "anime/female", "empty folder"]}

A folder is a path of names joined by "/"; "" is the top level. "folders" holds the folders
made empty (by New folder, or by moving everything out); folders with items in them exist
anyway.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

FILE = "_library.json"
_lock = threading.Lock()


def clean_folder(folder) -> str:
    """'a / b\\c' -> 'a/b/c'; drops empty, '.', '..' and hidden ('_x', '.x') parts."""
    parts = [p.strip() for p in str(folder or "").replace("\\", "/").split("/")]
    return "/".join(p for p in parts if p and p not in (".", "..") and p[0] not in "._")[:200]


def load(lib_dir: Path) -> dict:
    try:
        data = json.loads((Path(lib_dir) / FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    items = data.get("items") if isinstance(data.get("items"), dict) else {}
    folders = [clean_folder(f) for f in data.get("folders") or [] if isinstance(f, str)]
    return {"items": {k: v for k, v in items.items() if isinstance(v, dict)}, "folders": [f for f in folders if f]}


def _save(lib_dir: Path, data: dict) -> None:
    lib_dir = Path(lib_dir)
    lib_dir.mkdir(parents=True, exist_ok=True)
    data = {"items": {k: v for k, v in data["items"].items() if v.get("folder") or v.get("favorite")},
            "folders": sorted(set(data["folders"]))}
    tmp = lib_dir / (FILE + ".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(lib_dir / FILE)


def annotate(lib_dir: Path, items: list[dict]) -> dict:
    """{"items": items with "folder" and "favorite", "folders": every folder, parents included}."""
    data = load(lib_dir)
    out = []
    for it in items:
        o = data["items"].get(it["name"], {})
        out.append({**it, "folder": clean_folder(o.get("folder")), "favorite": bool(o.get("favorite"))})
    folders = set(data["folders"]) | {it["folder"] for it in out if it["folder"]}
    for f in list(folders):  # and every parent of each
        parts = f.split("/")
        folders.update("/".join(parts[:i]) for i in range(1, len(parts)))
    return {"items": out, "folders": sorted(folders, key=str.lower)}


def update(lib_dir: Path, names: list[str], *, folder=None, favorite=None) -> None:
    """File `names` under `folder` ("" = top level) and/or set their favourite flag;
    None leaves that one as it is."""
    with _lock:
        data = load(lib_dir)
        for name in names:
            o = dict(data["items"].get(name, {}))
            if folder is not None:
                o["folder"] = clean_folder(folder)
            if favorite is not None:
                o["favorite"] = bool(favorite)
            data["items"][name] = o
        if folder is not None and clean_folder(folder):
            data["folders"].append(clean_folder(folder))  # stays after its items move on
        _save(lib_dir, data)


def add_folder(lib_dir: Path, folder: str) -> str:
    folder = clean_folder(folder)
    if not folder:
        raise ValueError("name the folder")
    with _lock:
        data = load(lib_dir)
        data["folders"].append(folder)
        _save(lib_dir, data)
    return folder


def move_folder(lib_dir: Path, old: str, new: str) -> None:
    """Rename folder `old` to `new` with everything in and under it. new = its parent
    removes the folder, its contents moving up a level."""
    old, new = clean_folder(old), clean_folder(new)
    if not old:
        raise ValueError("choose a folder")
    if new == old or new.startswith(old + "/"):
        raise ValueError("a folder can't go inside itself")
    moved = lambda f: new + f[len(old):] if f == old or f.startswith(old + "/") else f  # noqa: E731
    with _lock:
        data = load(lib_dir)
        for o in data["items"].values():
            o["folder"] = clean_folder(moved(clean_folder(o.get("folder"))))
        data["folders"] = [clean_folder(moved(f)) for f in data["folders"]]
        data["folders"] = [f for f in data["folders"] if f]
        _save(lib_dir, data)


def forget(lib_dir: Path, name: str) -> None:
    """A deleted item: drop its folder and favourite (its folder stays)."""
    with _lock:
        data = load(lib_dir)
        o = data["items"].pop(name, None)
        if o is None:
            return
        if clean_folder(o.get("folder")):
            data["folders"].append(clean_folder(o.get("folder")))
        _save(lib_dir, data)
