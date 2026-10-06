"""Pose skeletons for the pose ControlNet, and a library of saved poses.

DWPose (RTMW whole-body keypoints, via rtmlib and onnxruntime, run here rather than in
ComfyUI) finds the main character's body, hand and face keypoints. Its person detector
is trained on HumanArt, so it handles drawings and game art. The skeleton is drawn the
way ControlNet openpose models were trained on (controlnet_aux's DWPose renderer:
coloured limb ellipses, HSV-coloured hand bones, white face dots on black), at the
output size, and fed to the Union ControlNet in openpose mode.

A pose is stored as normalized keypoints (0-1 of its image), so it can be drawn at any
output size: poses/<name>/pose.json, plus source.png (the image it came from) and
preview.png (the skeleton the pose picker shows).
"""

from __future__ import annotations

import colorsys
import json
import math
import re
import shutil
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .sizes import to_rgb

# OpenPose body order: 0 nose, 1 neck, 2-4 right arm, 5-7 left arm, 8-10 right leg,
# 11-13 left leg, 14/15 eyes, 16/17 ears ("right"/"left" are the character's).
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
         (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
          (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
          (170, 0, 255), (255, 0, 255), (255, 0, 85), (255, 0, 0)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11),
              (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
# rtmlib (to_openpose=True) keypoint slices per person.
BODY, FACE, LEFT_HAND, RIGHT_HAND = slice(0, 18), slice(24, 92), slice(92, 113), slice(113, 134)
# RTMW scores are SimCC peak values, not probabilities (visible joints ~3-8).
MIN_SCORE = 2.0
EDGE = 0.03  # leg points this close to the frame edge are out-of-frame guesses

_lock = threading.Lock()


def available() -> bool:
    try:
        import rtmlib  # noqa: F401
        return True
    except ImportError:
        return False


@lru_cache(maxsize=1)
def _model():
    from rtmlib import Wholebody
    # "performance": the 384x288 RTMW-x pose model; ~0.6 s per image on the CPU.
    # rtmlib keeps only the extracted .onnx files, then downloads the zips again in every
    # new process (~300 MB); the cached files are passed directly instead.
    mode = Wholebody.MODE["performance"]
    cache = Path.home() / ".cache" / "rtmlib" / "hub" / "checkpoints"

    def local(url: str) -> str:
        f = cache / (url.rsplit("/", 1)[-1].rsplit(".", 1)[0] + ".onnx")
        return str(f) if f.exists() else url
    return Wholebody(det=local(mode["det"]), det_input_size=mode["det_input_size"], pose=local(mode["pose"]),
                     pose_input_size=mode["pose_input_size"], to_openpose=True, backend="onnxruntime", device="cpu")


def detect(image: Path | Image.Image, min_score: float = MIN_SCORE) -> dict | None:
    """Keypoints of the main (largest) person, normalized to 0-1 of the image:
    {"body": [[x, y] or None] * 18, "left_hand": [...] * 21, "right_hand": [...] * 21,
     "face": [...] * 68, "scores": {...}}, or None if nobody is found."""
    import cv2

    img = to_rgb(Image.open(image) if isinstance(image, Path) else image)
    w, h = img.size
    arr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    scale = 1024 / max(w, h)  # small screenshots detect much better upscaled
    if abs(scale - 1) > 0.05:
        arr = cv2.resize(arr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    with _lock:
        kps, scores = _model()(arr)
    if len(kps) == 0:
        return None
    H, W = arr.shape[:2]

    def extent(i):
        pts = kps[i][BODY][scores[i][BODY] >= min_score]
        return 0 if len(pts) < 2 else float(np.ptp(pts[:, 0]) * np.ptp(pts[:, 1]) + 1)

    main = max(range(len(kps)), key=extent)
    k, s = kps[main], scores[main]

    def part(sl):
        return [[round(float(x) / W, 4), round(float(y) / H, 4)] if sc >= min_score and 0 <= x < W and 0 <= y < H
                else None for (x, y), sc in zip(k[sl], s[sl])]

    body = part(BODY)
    # In a cowboy shot or bust the model still "finds" the knees and ankles, pinned to
    # the frame edge; drawn, they'd tell the ControlNet the legs end there.
    for i in (9, 10, 12, 13):
        p = body[i]
        if p is not None and (p[1] > 1 - EDGE or p[0] < EDGE or p[0] > 1 - EDGE):
            body[i] = None
    for knee, ankle in ((9, 10), (12, 13)):  # an ankle without its knee is noise
        if body[knee] is None:
            body[ankle] = None
    return {"body": body, "left_hand": part(LEFT_HAND), "right_hand": part(RIGHT_HAND),
            "face": part(FACE), "size": [w, h],
            "scores": {"left_hand": round(float(s[LEFT_HAND].mean()), 2),
                       "right_hand": round(float(s[RIGHT_HAND].mean()), 2)}}


HAND_MIN_MEAN = 3.5  # mean keypoint score of a hand worth repainting; hidden hands score ~1.5-3


def fit(pose: dict, src_size: tuple[int, int], dst_size: tuple[int, int]) -> dict:
    """The pose as it lands after sizes.fit_to (centre-crop to dst's shape): keypoints
    re-normalized to the cropped frame; points cropped away are dropped."""
    sw, sh = src_size
    w, h = dst_size
    if sw * h > sh * w:  # too wide: sides trimmed
        nw = sh * w / h
        x0, y0, cw, ch = (sw - nw) / 2 / sw, 0.0, nw / sw, 1.0
    else:
        nh = sw * h / w
        x0, y0, cw, ch = 0.0, (sh - nh) / 2 / sh, 1.0, nh / sh

    def move(pts):
        out = []
        for p in pts or []:
            if p is None:
                out.append(None)
                continue
            x, y = (p[0] - x0) / cw, (p[1] - y0) / ch
            out.append([x, y] if 0 <= x <= 1 and 0 <= y <= 1 else None)
        return out

    return {**pose, **{k: move(pose.get(k)) for k in ("body", "left_hand", "right_hand", "face")},
            "size": list(dst_size)}


def _torso(pose: dict, size: tuple[int, int]):
    """(neck, mid-hip) in pixels of `size`, or None without a neck and a hip."""
    body = pose.get("body") or []
    neck = body[1] if len(body) > 1 else None
    hips = [body[i] for i in (8, 11) if len(body) > i and body[i] is not None]
    if neck is None or not hips:
        return None
    w, h = size
    return ((neck[0] * w, neck[1] * h),
            (sum(p[0] for p in hips) / len(hips) * w, sum(p[1] for p in hips) / len(hips) * h))


def match_framing(target: dict, target_size: tuple[int, int], base: dict, base_size: tuple[int, int],
                  size: tuple[int, int]) -> dict | None:
    """`target` drawn into a `size` frame the way `base` is framed: scaled so its torso is as
    long as base's and moved so its neck is where base's is. A new pose for a character keeps
    the character's framing (a cowboy shot stays one; points that leave the frame are
    dropped) instead of the pose image's: a full-body pose put a cowboy-shot character at half
    the size, and its armour lost its detail (2026-10-04). None when either lacks a neck or hip."""
    b = fit(base, base_size, size)
    tb, bb = _torso(target, target_size), _torso(b, size)
    if tb is None or bb is None:
        return None
    t_len = math.dist(*tb)
    if t_len < 1:
        return None
    scale = math.dist(*bb) / t_len
    (tnx, tny), (bnx, bny) = tb[0], bb[0]
    tw, th = target_size
    w, h = size

    def move(pts):
        out = []
        for p in pts or []:
            if p is None:
                out.append(None)
                continue
            x = ((p[0] * tw - tnx) * scale + bnx) / w
            y = ((p[1] * th - tny) * scale + bny) / h
            out.append([round(x, 4), round(y, 4)] if 0 <= x <= 1 and 0 <= y <= 1 else None)
        return out

    return {**target, **{k: move(target.get(k)) for k in ("body", "left_hand", "right_hand", "face")},
            "size": list(size)}


def render(pose: dict, size: tuple[int, int], body: bool = True, hands: bool = True, face: bool = False,
           only_hand: str | None = None) -> Image.Image:
    """The skeleton as ControlNet openpose models expect it, on black."""
    import cv2

    w, h = size
    canvas = np.zeros((h, w, 3), np.uint8)
    unit = max(w, h) / 1024  # line widths as DWPose draws them at ~1 megapixel
    stick = max(2, round(4 * unit))

    def xy(p):
        return p[0] * w, p[1] * h

    if body and not only_hand:
        pts = pose.get("body") or []
        for (a, b), color in zip(LIMBS, COLORS):
            if a >= len(pts) or b >= len(pts) or pts[a] is None or pts[b] is None:
                continue
            (x1, y1), (x2, y2) = xy(pts[a]), xy(pts[b])
            length = math.hypot(x1 - x2, y1 - y2)
            angle = math.degrees(math.atan2(y1 - y2, x1 - x2))
            poly = cv2.ellipse2Poly((int((x1 + x2) / 2), int((y1 + y2) / 2)), (int(length / 2), stick),
                                    int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, poly, [int(c * 0.6) for c in color])
        for p, color in zip(pts, COLORS):
            if p is not None:
                cv2.circle(canvas, tuple(int(v) for v in xy(p)), stick, color, -1)
    if hands or only_hand:
        for key in ([only_hand] if only_hand else ["left_hand", "right_hand"]):
            pts = pose.get(key) or []
            for i, (a, b) in enumerate(HAND_EDGES):
                if a < len(pts) and b < len(pts) and pts[a] is not None and pts[b] is not None:
                    color = [int(c * 255) for c in colorsys.hsv_to_rgb(i / len(HAND_EDGES), 1.0, 1.0)]
                    cv2.line(canvas, tuple(int(v) for v in xy(pts[a])), tuple(int(v) for v in xy(pts[b])),
                             color, max(1, round(2 * unit)))
            for p in pts:
                if p is not None:
                    cv2.circle(canvas, tuple(int(v) for v in xy(p)), max(2, round(4 * unit)), (0, 0, 255), -1)
    if face and not only_hand:
        for p in pose.get("face") or []:
            if p is not None:
                cv2.circle(canvas, tuple(int(v) for v in xy(p)), max(1, round(3 * unit)), (255, 255, 255), -1)
    return Image.fromarray(canvas)


def has_body(pose: dict | None, min_points: int = 4) -> bool:
    return bool(pose) and sum(p is not None for p in pose.get("body") or []) >= min_points


# The limbs a pose is told apart by: arms, legs and the head's lean (shoulders and hips sit
# the same in almost any pose and would only dilute the match).
MATCH_LIMBS = {(2, 3): "right upper arm", (3, 4): "right forearm", (5, 6): "left upper arm",
               (6, 7): "left forearm", (8, 9): "right thigh", (9, 10): "right shin", (11, 12): "left thigh",
               (12, 13): "left shin", (1, 0): "head"}
MATCH_TOLERANCE = 60  # degrees off at which a limb counts for nothing


def limb_match(target: dict, found: dict | None, size: tuple[int, int]) -> tuple[float | None, list[str]]:
    """How closely `found`'s limbs point the way `target`'s do (both normalized to a `size`
    frame): 0-10, the mean over the limbs both show of 1 - angle off / MATCH_TOLERANCE, but no
    more than 10 * (1 - the worst limb's angle off / 90): one arm clearly elsewhere isn't the
    pose, however well the rest matches. Also the limbs 25 or more degrees off. (None, []) when fewer than two limbs can be compared.
    Angles, not positions: they don't care where the character stands or how big it is.
    Arms left at the sides came out 100-128 degrees off crossed arms, crossed ones within
    21 (2026-10-05)."""
    if not found:
        return None, []
    w, h = size
    tb, fb = target.get("body") or [], found.get("body") or []
    scores, off, worst = [], [], 0.0
    for (i, j), name in MATCH_LIMBS.items():
        pts = [b[k] if k < len(b) else None for b in (tb, fb) for k in (i, j)]
        if None in pts:
            continue
        a = math.atan2((pts[1][1] - pts[0][1]) * h, (pts[1][0] - pts[0][0]) * w)
        b = math.atan2((pts[3][1] - pts[2][1]) * h, (pts[3][0] - pts[2][0]) * w)
        d = abs(math.degrees(a - b)) % 360
        d = min(d, 360 - d)
        scores.append(max(0.0, 1 - d / MATCH_TOLERANCE))
        worst = max(worst, d)
        if d >= 25:
            off.append(f"{name} {d:.0f} degrees off")
    if len(scores) < 2:
        return None, []
    return round(10 * min(sum(scores) / len(scores), max(0.0, 1 - worst / 90)), 2), off


def hand_boxes(pose: dict, size: tuple[int, int], min_points: int = 12, pad: float = 0.3,
               min_mean: float = HAND_MIN_MEAN) -> dict[str, tuple]:
    """Pixel boxes (x0, y0, x1, y1) around each confidently found hand, padded by `pad`
    of the hand's size on each side (the repaint needs the wrist and some context).
    A hand behind the body still gets a (low-confidence) skeleton; min_mean skips it."""
    w, h = size
    out = {}
    for key in ("left_hand", "right_hand"):
        pts = [p for p in pose.get(key) or [] if p is not None]
        if len(pts) < min_points or (pose.get("scores") or {}).get(key, 99) < min_mean:
            continue
        xs, ys = [p[0] * w for p in pts], [p[1] * h for p in pts]
        side = max(max(xs) - min(xs), max(ys) - min(ys), 24)
        cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        r = side / 2 * (1 + 2 * pad)
        out[key] = (max(0, int(cx - r)), max(0, int(cy - r)), min(w, int(cx + r)), min(h, int(cy + r)))
    return out


def slugify(name: str) -> str:
    """A folder-safe pose name: letters, digits, spaces, _ . - (others become _)."""
    return re.sub(r"[^A-Za-z0-9 _.-]+", "_", name or "").strip(" ._")


class PoseLibrary:
    """poses/<name>/{source.png, pose.json, preview.png}."""

    def __init__(self, root: Path):
        self.root = root

    def list(self) -> list[dict]:
        if not self.root.is_dir():
            return []
        out = []
        for d in sorted(p for p in self.root.iterdir() if p.is_dir()):
            try:
                meta = json.loads((d / "pose.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append({"name": d.name, "description": meta.get("description", ""), "size": meta.get("size"),
                        "preview": d / "preview.png", "source": d / "source.png"})
        return out

    def get(self, name: str) -> dict:
        d = self._dir(name)
        return json.loads((d / "pose.json").read_text(encoding="utf-8"))

    def source(self, name: str) -> Path:
        return self._dir(name) / "source.png"

    def _dir(self, name: str) -> Path:
        d = (self.root / name).resolve()
        if d.parent != self.root.resolve() or not (d / "pose.json").exists():
            raise FileNotFoundError(name)
        return d

    def add(self, name: str, image: Image.Image, describe=None, fallback: str = "pose") -> str:
        """Detect the pose in `image` and save it. describe(image) -> short pose tags, or
        (tags, suggested name) (optional; the prompt writer and judge use the tags). With no
        `name`, the suggested name is used, else `fallback`. Returns the saved name."""
        pose = detect(image)
        if not has_body(pose):
            raise ValueError("no person found in that image")
        img = to_rgb(image)
        described = describe(img) if describe else ""
        description, suggested = described if isinstance(described, tuple) else (described, "")
        base = (slugify(name) or slugify(suggested) or slugify(fallback) or "pose")[:60].strip(" ._-") or "pose"
        self.root.mkdir(parents=True, exist_ok=True)
        slug, n = base, 2
        while (self.root / slug).exists():
            slug, n = f"{base}_{n}", n + 1
        d = self.root / slug
        d.mkdir()
        try:
            img.save(d / "source.png")
            pose["description"] = description
            (d / "pose.json").write_text(json.dumps(pose), encoding="utf-8")
            w, h = img.size
            s = 512 / max(w, h)
            render(pose, (max(1, round(w * s)), max(1, round(h * s))), face=True).save(d / "preview.png")
        except Exception:
            shutil.rmtree(d, ignore_errors=True)
            raise
        return slug

    def remove(self, name: str) -> None:
        d = self._dir(name)
        trash = self.root / "_removed"
        trash.mkdir(exist_ok=True)
        dest = trash / d.name
        n = 2
        while dest.exists():
            dest, n = trash / f"{d.name}_{n}", n + 1
        shutil.move(str(d), str(dest))
