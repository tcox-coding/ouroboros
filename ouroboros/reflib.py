"""Saved styles and characters: the style and subject targets' counterpart of the pose library.

<root>/<kind folder>/<name>/{source.png, meta.json}; meta.json holds the tags the LLM wrote
when the item was added (the art style's, or the character's) and the image size. Picking a
saved item for a target uses its image (and its tags as the target's words when none are
given). Like poses, an item added without a name is named by the LLM, deleting moves it to
<kind folder>/_removed/, and "AI picks" chooses one from the tags (pose_picker.pick_item).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image

from .pose import slugify
from .sizes import to_rgb

KINDS = {
    "style": {"folder": "styles", "describe": """Describe only the ART STYLE of the image, as Stable Diffusion tags: medium, line work,
shading, colouring, palette, lighting, texture and rendering, and the era or school it
looks like. No subject, character, clothing, pose or background content. 8-20 short
comma-separated tags.

Also give the style a short name (2-5 lowercase words joined by underscores, e.g.
flat_cel_shading, thick_ink_watercolour, glossy_3d_render) naming what sets it apart from
other styles. JSON only."""},
    "character": {"folder": "characters", "describe": """Describe only the main CHARACTER of the image, as Stable Diffusion tags: build, skin,
face, eyes, hair (colour, length, cut, bangs), every garment and accessory with its colour
and cut, and any distinctive mark. No art style, pose, framing or background. 10-30 short
comma-separated tags.

Also give the character a short name (2-5 lowercase words joined by underscores, e.g.
red_haired_knight, silver_fox_mage) from their most distinctive features. JSON only."""},
}


class RefLibrary:
    def __init__(self, root: Path, kind: str):
        if kind not in KINDS:
            raise ValueError(kind)
        self.kind, self.root = kind, Path(root) / KINDS[kind]["folder"]

    def list(self) -> list[dict]:
        if not self.root.is_dir():
            return []
        out = []
        for d in sorted(p for p in self.root.iterdir() if p.is_dir() and not p.name.startswith("_")):
            if not (d / "source.png").is_file():
                continue
            try:
                meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                meta = {}
            out.append({"name": d.name, "description": meta.get("description", ""), "size": meta.get("size"),
                        "source": d / "source.png"})
        return out

    def _dir(self, name: str) -> Path:
        d = (self.root / name).resolve()
        if d.parent != self.root.resolve() or not (d / "source.png").is_file() or name.startswith("_"):
            raise FileNotFoundError(name)
        return d

    def get(self, name: str) -> dict:
        d = self._dir(name)
        try:
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        return {"name": d.name, "description": meta.get("description", ""), "size": meta.get("size"),
                "source": d / "source.png"}

    def source(self, name: str) -> Path:
        return self._dir(name) / "source.png"

    def add(self, name: str, image: Image.Image, describe=None, fallback: str = "") -> str:
        """Save `image`. describe(image) -> tags, or (tags, suggested name) (optional). With no
        `name`, the suggested name is used, else `fallback`, else the kind. Returns the name."""
        img = to_rgb(image)
        described = describe(img) if describe else ""
        description, suggested = described if isinstance(described, tuple) else (described, "")
        base = (slugify(name) or slugify(suggested) or slugify(fallback) or self.kind)[:60].strip(" ._-") or self.kind
        self.root.mkdir(parents=True, exist_ok=True)
        slug, n = base, 2
        while (self.root / slug).exists():
            slug, n = f"{base}_{n}", n + 1
        d = self.root / slug
        d.mkdir()
        try:
            img.save(d / "source.png")
            (d / "meta.json").write_text(json.dumps({"description": description, "size": list(img.size)}),
                                         encoding="utf-8")
        except Exception:
            shutil.rmtree(d, ignore_errors=True)
            raise
        return slug

    def remove(self, name: str) -> None:
        d = self._dir(name)
        trash = self.root / "_removed"
        trash.mkdir(exist_ok=True)
        dest, n = trash / d.name, 2
        while dest.exists():
            dest, n = trash / f"{d.name}_{n}", n + 1
        shutil.move(str(d), str(dest))

    def menu(self) -> list[dict]:
        """[{"name", "description"}] for pose_picker.pick_item."""
        return [{"name": p["name"], "description": p["description"]} for p in self.list()]


ROLE_KIND = {"style": "style", "subject": "character"}  # target role -> library


def resolve(root: Path, role: str, value: str, *, backend=None, description: str = "", positive: str = "",
            max_side: int = 512) -> dict | None:
    """A target's saved choice: a name, or "auto" (the LLM picks one that fits; needs
    `backend`). Returns {"name", "image", "description", "pick"} or None (no choice, none
    fits, or the name isn't saved). "pick" is the LLM's answer when it chose."""
    value = (value or "").strip()
    if not value or role not in ROLE_KIND:
        return None
    lib = RefLibrary(root, ROLE_KIND[role])
    pick = None
    if value == "auto":
        if backend is None:
            return None
        from .pose_picker import pick_item
        pick = pick_item(backend, lib.menu(), ROLE_KIND[role], description=description, positive=positive,
                         max_side=max_side)
        if not pick["pick"]:
            return {"name": None, "image": None, "description": "", "pick": pick}
        value = pick["pick"]
    try:
        item = lib.get(value)
    except FileNotFoundError:
        return None
    return {"name": item["name"], "image": item["source"], "description": item["description"], "pick": pick}
