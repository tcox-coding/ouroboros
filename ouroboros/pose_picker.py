""""AI picks": choose the saved pose (poses/), style or character (reflib.py) that best fits
what's being made, or none.

The LLM gets the description and/or prompt (and, when there's neither, the reference image)
and the library as a menu: each pose's name and the pose tags written when it was added.
It must answer with one of those names or "none"; "none" is the right answer when no pose
fits, so a pose is never forced on a render it doesn't suit. Large libraries are cut to a
text shortlist first, so the call stays small.
"""

from __future__ import annotations

import re

NONE = "none"
MAX_MENU = 40

INSTRUCTIONS = """You choose a pose for a Stable Diffusion render from a library of saved poses. The chosen
pose is imposed with a pose ControlNet (body, arms, hands and head follow it exactly), while
the character, outfit and style come from the description, prompt and reference.

Pick the pose that best matches what the description or prompt asks for: framing (full body,
cowboy shot, upper body), stance (standing, sitting, kneeling, lying), what the arms and hands
do, where the head and eyes point, and the mood. Prefer an exact match of what is asked for
over a merely pleasant pose. If the request names or implies a pose that none of the saved
poses has, or the poses would contradict it (e.g. sitting when it asks for running), answer
"none". If the request says nothing about the pose, choose one that suits the subject and
framing, or "none" if none does.

Answer with the pose's exact name from the menu (or "none") and one short sentence why.
Reply with JSON only."""


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", (text or "").lower()) if len(w) > 2}


def shortlist(poses: list[dict], text: str, n: int = MAX_MENU) -> list[dict]:
    """The n poses whose name and tags share the most words with the request (all of them
    when the library is small). Keeps library order among equals."""
    if len(poses) <= n:
        return list(poses)
    want = _words(text)
    score = lambda p: len(want & _words(f"{p['name'].replace('_', ' ')} {p.get('description', '')}"))
    return sorted(poses, key=score, reverse=True)[:n]


def schema(names: list[str]) -> dict:
    return {"type": "object",
            "properties": {"pose": {"type": "string", "enum": [*names, NONE]}, "reason": {"type": "string"}},
            "required": ["pose", "reason"], "additionalProperties": False}


ITEM_INSTRUCTIONS = {
    "style": """You choose an art style for a Stable Diffusion render from a library of saved styles. The
chosen style's image is applied with an IP-Adapter in style-transfer mode, so the render
takes on its line work, shading, palette and rendering, while the content comes from the
description and prompt.

Pick the saved style that best matches the look the description or prompt asks for
(medium, line work, shading, colours, era). If it names a style none of them has, or they
would contradict it, answer "none". If it says nothing about the style, choose one that
suits the subject, or "none".

Answer with the style's exact name from the menu (or "none") and one short sentence why.
Reply with JSON only.""",
    "character": """You choose a character for a Stable Diffusion render from a library of saved characters.
The chosen character's image is carried into the render with an IP-Adapter (face, hair,
build, outfit), while the style and pose come from elsewhere.

Pick the saved character the description or prompt asks for: by name, or by matching
features (hair, eyes, build, outfit, species). Answer "none" unless one clearly matches:
never substitute a different character for the one described.

Answer with the character's exact name from the menu (or "none") and one short sentence why.
Reply with JSON only.""",
}


def pick_item(backend, items: list[dict], kind: str, *, description: str = "", positive: str = "",
              max_side: int = 512) -> dict:
    """A saved style or character for the request ("style" / "character"); items:
    [{"name", "description"}]. Returns {"pick": name or None, "reason", "cost", "considered"}."""
    if not items:
        return {"pick": None, "reason": f"the {kind} library is empty", "cost": 0.0, "considered": 0}
    request = "\n\n".join(x for x in (f"DESCRIPTION\n{description.strip()}" if description.strip() else "",
                                      f"PROMPT\n{positive.strip()}" if positive.strip() else "") if x)
    if not request:
        return {"pick": None, "reason": "no description or prompt to choose from", "cost": 0.0, "considered": 0}
    menu = shortlist(items, f"{description} {positive}")
    parts = [{"text": request + f"\n\nSAVED {kind.upper()}S (name: tags)\n"
              + "\n".join(f"- {p['name']}: {p.get('description') or '(no tags)'}" for p in menu)}]
    sch = schema([p["name"] for p in menu])
    sch["properties"] = {"pick": sch["properties"].pop("pose"), "reason": sch["properties"]["reason"]}
    sch["required"] = ["pick", "reason"]
    data, cost, _tokens = backend.complete(ITEM_INSTRUCTIONS[kind], parts, sch, f"{kind}_pick", max_side)
    name = str(data.get("pick") or "").strip()
    return {"pick": name if name in {p["name"] for p in menu} else None,
            "reason": str(data.get("reason") or "").strip(), "cost": cost or 0.0, "considered": len(menu)}


def pick_pose(backend, poses: list[dict], *, description: str = "", positive: str = "", reference=None,
              max_side: int = 512) -> dict:
    """poses: [{"name", "description"}]. Returns {"pose": name or None, "reason", "cost", "considered"}."""
    if not poses:
        return {"pose": None, "reason": "the pose library is empty", "cost": 0.0, "considered": 0}
    request = "\n\n".join(x for x in (f"DESCRIPTION\n{description.strip()}" if description.strip() else "",
                                      f"PROMPT\n{positive.strip()}" if positive.strip() else "") if x)
    menu = shortlist(poses, f"{description} {positive}")
    parts = [{"text": (request or "No description or prompt: choose from the REFERENCE image below.")
              + "\n\nSAVED POSES (name: pose tags)\n"
              + "\n".join(f"- {p['name']}: {p.get('description') or '(no tags)'}" for p in menu)}]
    if reference is not None and not request:
        parts += [{"text": "REFERENCE IMAGE (choose the pose closest to what it shows, or one that suits it):"},
                  {"image": reference}]
    data, cost, _tokens = backend.complete(INSTRUCTIONS, parts, schema([p["name"] for p in menu]), "pose_pick",
                                           max_side)
    name = str(data.get("pose") or "").strip()
    known = {p["name"] for p in menu}
    return {"pose": name if name in known else None, "reason": str(data.get("reason") or "").strip(),
            "cost": cost or 0.0, "considered": len(menu)}


def library_menu(library) -> list[dict]:
    """[{"name", "description"}] from a PoseLibrary."""
    return [{"name": p["name"], "description": p.get("description", "")} for p in library.list()]
