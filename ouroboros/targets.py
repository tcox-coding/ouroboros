"""The three things a render aims at: its STYLE, its SUBJECT and its POSE.

Each is a Target: an image, a short text, or both ("90s anime cel shading", "red-haired
knight in silver armour", "kneeling, hands together"). A job used to have one reference
image standing for all three; an empty target still falls back to that image (the job's
`reference`), so a one-image job behaves exactly as before.

Who reads which:
  - the judge scores each rubric criterion against its own target (CRITERION_TARGET);
  - the prompt writer describes the subject from SUBJECT, the drawing style from STYLE
    and the pose from POSE;
  - an IP-Adapter carries the subject image (linear) and the style image (style transfer);
  - the pose ControlNet follows the pose image (or a saved pose);
  - the LoRA picker compares LoRA examples with the style image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROLES = ("style", "subject", "pose")
# Which target each rubric criterion is judged against (anything else: the whole image).
CRITERION_TARGET = {"identity": "subject", "color": "subject", "extras": "subject",
                    "style": "style", "composition": "pose"}
LABELS = {"style": "STYLE", "subject": "SUBJECT", "pose": "POSE"}
HINTS = {
    "style": "the art style to match: line work, shading, rendering, colour treatment, medium. "
             "Not its subject or pose.",
    "subject": "the character to match: face, hair, body type, outfit, accessories and their colours. "
               "Not its drawing style or pose.",
    "pose": "the pose and framing to match: stance, arms and hands, head and gaze direction, camera angle. "
            "Not its character or style.",
}


@dataclass
class Target:
    image: Path | None = None
    text: str = ""
    name: str = ""        # a saved library item it came from (pose / style / character), if any

    @property
    def empty(self) -> bool:
        return self.image is None and not self.text.strip()

    def to_dict(self) -> dict:
        return {"image": self.image.name if self.image else None, "text": self.text, "name": self.name}


@dataclass
class Targets:
    style: Target = field(default_factory=Target)
    subject: Target = field(default_factory=Target)
    pose: Target = field(default_factory=Target)

    def get(self, role: str) -> Target:
        return getattr(self, role)

    def items(self):
        return [(r, self.get(r)) for r in ROLES]

    @property
    def split(self) -> bool:
        """True when the targets say more than one reference image would: any text, or
        images that aren't all the same file. False means the classic one-reference job."""
        if any(t.text.strip() for _, t in self.items()):
            return True
        images = {str(t.image.resolve()) for _, t in self.items() if t.image}
        return len(images) > 1

    def image_for(self, role: str, fallback: Path | None) -> Path | None:
        """The image for a role: its own, else the job's main reference."""
        return self.get(role).image or (fallback if self.get(role).text.strip() == "" else None)

    def primary(self) -> Path | None:
        """The main image of a job: the subject's, else the style's, else the pose's."""
        return self.subject.image or self.style.image or self.pose.image

    def to_dict(self) -> dict:
        return {r: t.to_dict() for r, t in self.items()}


def from_spec(spec: dict, folder: Path) -> Targets:
    """Targets from a job.json: {"style": {"image": "style.png", "text": "..."}, "subject": ..., "pose": ...}
    (a bare string is a text). Image names are files in the job folder."""
    out = Targets()
    for role in ROLES:
        v = spec.get(role)
        if isinstance(v, str):
            v = {"text": v}
        if not isinstance(v, dict):
            continue
        img = folder / v["image"] if v.get("image") else None
        if img is not None and not img.is_file():
            raise FileNotFoundError(f"{role} image {v['image']} is missing from {folder}")
        setattr(out, role, Target(image=img, text=str(v.get("text") or "").strip(), name=str(v.get("name") or "")))
    return out


def judge_parts(targets: Targets, fallback: Path | None, pose_skeleton: dict | None = None) -> list[dict]:
    """The target section of a judge request: each target's image and/or text, labelled with
    what it stands for and which criteria are judged against it. pose_skeleton:
    {"image", "description"} when the pose comes from a skeleton (saved pose / pose image)."""
    crit = {r: [c for c, t in CRITERION_TARGET.items() if t == r] for r in ROLES}
    parts: list[dict] = [{"text": "TARGETS: this render has three separate targets. Judge each rubric criterion "
                                  "against ITS target only: " + "; ".join(
                                      f"{', '.join(crit[r])} against {LABELS[r]}" for r in ROLES)
                                  + "; quality on the candidate alone. Differences are differences from the "
                                    "relevant target. None of these is a candidate; never score them."}]
    shown: dict[str, str] = {}  # image file -> the target it was first shown for (each image is sent once)

    def image_part(role: str, image: Path) -> list[dict]:
        key = str(Path(image).resolve())
        if key in shown:
            return [{"text": f"(the same image as the {shown[key]} target above)"}]
        shown[key] = LABELS[role]
        return [{"image": image}]

    for role, image, text in target_images(targets, fallback, pose_skeleton):
        head = f"{LABELS[role]} TARGET ({HINTS[role]})"
        if role == "pose" and pose_skeleton:
            parts += [{"text": f"{head}: a pose skeleton" + (f" ({text})" if text else "") + ":"},
                      *image_part(role, image)]
        elif text and image:
            parts += [{"text": f"{head}, described: {text} Image:"}, *image_part(role, image)]
        elif text:
            parts.append({"text": f"{head}, described: {text}"})
        elif image:
            parts += [{"text": f"{head}:"}, *image_part(role, image)]
        else:
            parts.append({"text": f"{head}: none given; judge it against the GOAL."})
    return parts


def target_images(targets: Targets, fallback: Path | None, pose_skeleton: dict | None = None):
    """[(role, image or None, text)] as the judge sees them: an empty target falls back to
    the job's reference image; a pose skeleton stands in for the pose image."""
    out = []
    for role in ROLES:
        t = targets.get(role)
        if role == "pose" and pose_skeleton:
            out.append((role, pose_skeleton["image"], pose_skeleton.get("description") or t.text.strip()))
        else:
            out.append((role, t.image or (fallback if not t.text.strip() else None), t.text.strip()))
    return out


def prompt_parts(targets: Targets) -> list[dict]:
    """For the prompt writer: what each given target says, images included."""
    parts: list[dict] = []
    for role in ROLES:
        t = targets.get(role)
        if t.empty:
            continue
        what = {"style": "write the drawing-style and rendering tags from this, not its subject",
                "subject": "describe this character (build, face, hair, every garment and colour), not its style or pose",
                "pose": "describe this pose and framing, not its character"}[role]
        parts.append({"text": f"{LABELS[role]} ({what})" + (f": {t.text.strip()}" if t.text.strip() else "")
                      + (" Image:" if t.image else "")})
        if t.image:
            parts.append({"image": t.image})
    return parts
