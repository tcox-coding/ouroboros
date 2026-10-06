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
    # Set for a job with a description that gives some targets but not all ("this character,
    # arms crossed, in Incase style"): a missing target is then judged against the GOAL, not
    # the job's reference image, which would pull the pose and style back to that image's.
    goal_for_missing: bool = False

    def get(self, role: str) -> Target:
        return getattr(self, role)

    def items(self):
        return [(r, self.get(r)) for r in ROLES]

    @property
    def split(self) -> bool:
        """True when the targets say more than one reference image would: any text, or
        images that aren't all the same file. False means the classic one-reference job."""
        if self.goal_for_missing or any(t.text.strip() for _, t in self.items()):
            return True
        images = {str(t.image.resolve()) for _, t in self.items() if t.image}
        return len(images) > 1

    def image_for(self, role: str, fallback: Path | None) -> Path | None:
        """The image for a role: its own, else the job's main reference (not with goal_for_missing)."""
        if self.goal_for_missing:
            return self.get(role).image
        return self.get(role).image or (fallback if self.get(role).text.strip() == "" else None)

    def missing(self) -> list[str]:
        """Roles with neither an image nor words."""
        return [r for r, t in self.items() if t.empty]

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
            out.append((role, targets.image_for(role, fallback), t.text.strip()))
    return out


def prompt_parts(targets: Targets) -> list[dict]:
    """For the prompt writer: what each given target says, images included (an image that
    stands for two targets, e.g. the character's own look as the style, is sent once)."""
    parts: list[dict] = []
    shown: dict[str, str] = {}
    for role in ROLES:
        t = targets.get(role)
        if t.empty:
            continue
        what = {"style": "write the drawing-style and rendering tags from this, not its subject",
                "subject": "describe this character (build, face, hair, every garment and colour), not its style or pose",
                "pose": "describe this pose and framing, not its character"}[role]
        # A file (jobs, Home renders) or an image in memory (Home's Write prompts)
        key = None if t.image is None else str(Path(t.image).resolve()) if isinstance(t.image, (str, Path)) \
            else f"id{id(t.image)}"
        same = shown.get(key) if key else None
        parts.append({"text": f"{LABELS[role]} ({what})" + (f": {t.text.strip()}" if t.text.strip() else "")
                      + (f" Image: the same as the {same} image." if same else " Image:" if t.image else "")})
        if t.image and not same:
            shown[key] = LABELS[role]
            parts.append({"image": t.image})
    return parts


# Two IP-Adapters on the same character (the saved style often comes from the same artwork)
# add up: subject 0.6 + style 0.5 PLUS washed a render out to flat cream with blown
# highlights, while 0.3 + 0.3 of the same pair rendered clean (same seed, 2026-10-04).
DEFAULT_MAX_COMBINED = 0.6


def max_combined(ip_cfg: dict) -> float:
    return float(ip_cfg.get("max_combined_weight", DEFAULT_MAX_COMBINED))


def cap_ip_weights(adapters: list[dict], cap: float) -> str | None:
    """Scale the weights of two or more IP-Adapters down in proportion (in place) so they
    add up to at most `cap`. One adapter is left as set. Returns a note when scaled."""
    if len(adapters) < 2 or cap <= 0:
        return None
    total = sum(float(a["weight"]) for a in adapters)
    if total <= cap + 1e-9:
        return None
    before = ", ".join(f"{a.get('role', 'image')} {float(a['weight']):g}" for a in adapters)
    for a in adapters:
        a["weight"] = round(float(a["weight"]) * cap / total, 2)
    after = ", ".join(f"{a.get('role', 'image')} {a['weight']:g}" for a in adapters)
    return (f"IP-Adapter weights added up to {total:g} (above {cap:g}, where renders wash out); "
            f"scaled from {before} to {after}")


# The character's IP-Adapter: STANDARD kept Cassandra's black undersuit under the armour and
# her build, where PLUS made the render glossy and pushed the image's colours into it (same
# seed, 2026-10-04). The style's IP-Adapter keeps ipadapter.preset (PLUS).
def subject_preset(ip_cfg: dict) -> str:
    return ip_cfg.get("subject_preset") or "STANDARD (medium strength)"


def subject_cutout_default(ip_cfg: dict) -> bool:
    """Remove the character image's background before its IP-Adapter (nobg.cutout)."""
    return bool(ip_cfg.get("subject_cutout", True))
