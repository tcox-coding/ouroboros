"""Generation parameters, their bounds, and applying the planner's edits."""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass, replace

# Modes the planner can choose. Each sets the workflow's switches and which image
# goes into its image input: (use_reference, masked, source image).
MODES = {
    "txt2img": (False, False, "reference"),           # image is loaded but unused
    "img2img_reference": (True, False, "reference"),  # "use the reference image"
    "img2img_best": (True, False, "best"),            # refine the current best result
    "inpaint_reference": (True, True, "reference"),   # repair a named region of the reference
    "inpaint_best": (True, True, "best"),             # repair a named region of the best result
}
INPAINT_MODES = {m for m, (_, masked, _) in MODES.items() if masked}
# Denoise a mode starts at when the planner switches to it without giving one.
DEFAULT_DENOISE = {"txt2img": 1.0, "img2img_reference": 0.75, "img2img_best": 0.45,
                   "inpaint_reference": 0.75, "inpaint_best": 0.75}

BOUNDS = {"steps": (10, 60), "cfg": (1.0, 15.0), "denoise": (0.15, 1.0)}
MIN_INPAINT_DENOISE = 0.5

# Planner phases, coarse to fine. Later phases change fewer things.
PHASES = ["explore", "refine", "repair"]


@dataclass
class GenParams:
    mode: str = "txt2img"
    positive: str = ""
    negative: str = ""
    seed: int = 0
    steps: int = 30
    cfg: float = 5.5
    sampler_name: str = "dpmpp_2m"
    scheduler: str = "karras"
    denoise: float = 1.0
    mask_target: str | None = None  # text for CLIPSeg/SAM3, e.g. "left hand"
    loras: tuple = ()               # ((ComfyUI lora name, strength), ...): the full active set

    def to_dict(self) -> dict:
        return asdict(self)

    def short(self) -> str:
        """One-line summary for the compact history sent to the planner."""
        s = (f"{self.mode} seed={self.seed} steps={self.steps} cfg={self.cfg:g} "
             f"{self.sampler_name}/{self.scheduler} denoise={self.denoise:g}")
        s += f" mask='{self.mask_target}'" if self.mask_target else ""
        return s + (" loras=" + ",".join(f"{lora_stem(n)}:{w:g}" for n, w in self.loras) if self.loras else "")


def lora_stem(name: str) -> str:
    return name.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]


def clamp(name: str, value):
    lo, hi = BOUNDS[name]
    return type(lo)(min(hi, max(lo, value)))


def split_tags(prompt: str) -> list[str]:
    return [t.strip() for t in prompt.replace("\n", ",").split(",") if t.strip()]


def norm_tag(tag: str) -> str:
    """Compare tags without case, weight brackets, weights or escapes:
    "(Red Shirt:1.3)" == "red shirt", "1990s \\(style\\)" == "1990s (style)". Brackets
    inside a tag are kept: stripping them made "1990s (style)" into "1990s (style"."""
    t = tag.strip().replace("\\", "")
    m = re.fullmatch(r"[(\[]+(.*?)(?::\s*[\d.]+)?\s*[)\]]+", t)  # (tag), ((tag)), (tag:1.2), [tag]
    if m and _balanced(m.group(1)):  # not "(a) b (c)", whose outer brackets aren't a pair
        t = m.group(1)
    return re.sub(r":\s*[\d.]+\s*$", "", t).strip().lower()


def _balanced(text: str) -> bool:
    depth = 0
    for ch in text:
        depth += {"(": 1, "[": 1, ")": -1, "]": -1}.get(ch, 0)
        if depth < 0:
            return False
    return depth == 0


def _words(tag: str) -> set[str]:
    # Exact words only: stripping plurals would make "shorts" match "short-sleeve".
    return set(re.findall(r"[a-z0-9]+", norm_tag(tag)))


def drop_negative_conflicts(positive: str, negative: str) -> tuple[str, list[str]]:
    """Remove negative tags that negate what the positive asks for: a negative tag
    whose words all appear in one positive tag ("belt" vs "thin dark belt",
    "shirt" vs "red work shirt"). Keeps the negative's line layout."""
    pos_words = [_words(t) for t in split_tags(positive)]
    removed: list[str] = []
    lines = []
    for line in negative.splitlines():
        kept = []
        for tag in [t.strip() for t in line.split(",") if t.strip()]:
            w = _words(tag)
            if w and any(w <= p for p in pos_words):
                removed.append(tag)
            else:
                kept.append(tag)
        lines.append(", ".join(kept))
    return "\n".join(l for l in lines if l.strip()), removed


NEGATION = re.compile(r"^\(?\s*(no|without|not|remove)\s+", re.I)


def split_negations(tags: list[str]) -> tuple[list[str], list[str]]:
    """Stable Diffusion reads the positive tag "no chest pocket" as "chest pocket".
    Returns (positive tags, tags to move to the negative with the "no" removed)."""
    keep, negate = [], []
    for t in tags:
        m = NEGATION.match(t.strip())
        if m:
            negate.append(t.strip()[m.end():].strip(" )"))
        else:
            keep.append(t)
    return keep, [n for n in negate if n]


STOPWORDS = {"a", "an", "the", "and", "with", "of", "on", "in", "to", "at", "for", "from", "her", "his", "is"}
MAX_WEIGHT = 1.5


def cap_weight(tag: str) -> str:
    """(tag:1.8) -> (tag:1.5): past ~1.5 a weight distorts the image more than it helps."""
    return re.sub(r":\s*([\d.]+)\s*\)\s*$", lambda m: f":{min(float(m.group(1)), MAX_WEIGHT):g})", tag)


def _best_line(lines: list[str], tag: str) -> int:
    """The prompt line a new tag belongs to: the one sharing most words with it (a
    clothing tag joins the clothing line, a gaze tag the pose line). Prompts are
    grouped by subject, one line per group, and a tag appended at the very end carries
    the least weight. Falls back to the last line."""
    words = {w for w in _words(tag) if w not in STOPWORDS and len(w) > 2}
    filled = [i for i, l in enumerate(lines) if l.strip()]
    if not filled:
        return 0
    best, score = filled[-1], 0
    for i in filled:
        overlap = len(words & set().union(*(_words(x) for x in lines[i].split(",") if x.strip())))
        if overlap > score:
            best, score = i, overlap
    return best


def edit_prompt(prompt: str, add: list[str], remove: list[str]) -> str:
    """Remove tags, then add new ones, keeping the prompt's line layout. Tags match
    without case, brackets or weights, so removing "red shirt" also removes
    "(red shirt:1.3)". Each new tag goes into the line it fits best (see _best_line)."""
    drop = {norm_tag(r) for r in remove if r.strip()}
    lines = prompt.split("\n")
    if drop:
        kept_lines = []
        for line in lines:
            tags = [x.strip() for x in line.split(",") if x.strip()]
            kept = [x for x in tags if norm_tag(x) not in drop]
            if kept or not tags:  # blank lines separate sections: keep them
                kept_lines.append(", ".join(kept) if tags else line)
        lines = kept_lines
    existing = {norm_tag(x) for x in split_tags("\n".join(lines))}
    extra = list(dict.fromkeys(cap_weight(a.strip()) for a in add if a.strip() and norm_tag(a) not in existing))
    for tag in extra:
        i = _best_line(lines, tag)
        if not lines:
            lines = [""]
        lines[i] = lines[i].rstrip().rstrip(",") + (", " if lines[i].strip() else "") + tag
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def allowed_modes(lc: dict) -> list[str]:
    """Modes the planner may pick. Inpainting the reference keeps everything outside
    the mask identical to it, so it's off unless loop.allow_inpaint_reference is set.
    With a pose from the pose library (lc["pose_from_library"]), nothing may start from
    the reference: img2img would bring the reference's pose back."""
    out = [m for m in MODES if m != "inpaint_reference" or lc.get("allow_inpaint_reference", False)]
    if lc.get("pose_from_library"):
        out = [m for m in out if MODES[m][2] != "reference" or m == "txt2img"]
    return out


def enforce_reference_rules(p: GenParams, lc: dict) -> GenParams:
    """Keep results from being near-copies of the reference: img2img from the
    reference needs at least loop.min_reference_denoise, and a disallowed
    inpaint_reference becomes img2img_reference."""
    modes = allowed_modes(lc)
    if p.mode not in modes:
        p = (replace(p, mode="img2img_reference", mask_target=None) if "img2img_reference" in modes
             else replace(p, mode="txt2img", mask_target=None, denoise=1.0))
    min_dn = lc.get("min_reference_denoise") or 0
    if p.mode == "img2img_reference" and p.denoise < min_dn:
        p = replace(p, denoise=min_dn)
    return p


def reseed_duplicates(batch: list[GenParams], rendered: set | None = None) -> list[GenParams]:
    """Give a new seed to any candidate identical to another in the batch (clamping
    can do that) or to one already rendered in an earlier round (the same settings and
    seed give the same image, so it would only be re-scored). `rendered` is updated."""
    rendered = rendered if rendered is not None else set()
    out: list[GenParams] = []
    for p in batch:
        while p in out or _key(p) in rendered:
            p = replace(p, seed=random.randrange(2**32))
        out.append(p)
        rendered.add(_key(p))
    return out


def _key(p: GenParams) -> tuple:
    return tuple(sorted(p.to_dict().items()))


LORA_STRENGTH = (0.0, 1.5)


def apply_loras(p: GenParams, chosen, allowed: dict[str, str], max_loras: int) -> GenParams:
    """Replace the managed LoRAs in p.loras with `chosen` ([{"lora": stem, "strength"}]).
    `allowed` maps the stems the judge may use to ComfyUI names; LoRAs outside it
    (e.g. a hand-fix LoRA saved in the workflow) are left as they are."""
    if chosen is None:
        return p
    managed = set(allowed.values())
    keep = [(n, w) for n, w in p.loras if n not in managed]
    picked = []
    for c in chosen:
        name = allowed.get(str(c.get("lora", "")))
        if name and name not in {n for n, _ in picked}:
            s = c.get("strength")
            # 0 means "switch it off": `or 0.6` would have turned it into 0.6.
            w = min(LORA_STRENGTH[1], max(LORA_STRENGTH[0], float(0.6 if s is None else s)))
            if w > 0:
                picked.append((name, round(w, 2)))
    return replace(p, loras=tuple(keep + picked[:max_loras]))


def apply_edit(base: GenParams, edit: dict, samplers: list[str], schedulers: list[str],
               lora_choices: dict[str, str] | None = None, max_loras: int = 3) -> GenParams:
    """Apply a planner edit (see judge.EDIT_SCHEMA). Unknown or out-of-range values
    are clamped or ignored, so a bad suggestion can't break the workflow."""
    p = replace(base)
    if lora_choices:
        p = apply_loras(p, edit.get("loras"), lora_choices, max_loras)
    add, neg_add = split_negations(edit.get("prompt_add") or [])
    remove = [r for r in edit.get("prompt_remove") or [] if r.strip()]
    # The judge often "removes" what it sees in the image but the prompt never asked
    # for ("green highlights in hair"); that's a no-op, and in testing the same
    # difference came back round after round. It means "avoid this": negate it, unless
    # that would ban something the prompt (after this edit) asks for.
    present = {norm_tag(t) for t in split_tags(p.positive)}
    p.positive = edit_prompt(p.positive, add, remove)
    pos_words = [_words(t) for t in split_tags(p.positive)]
    neg_add += [r.strip() for r in remove if norm_tag(r) not in present
                and not any(_words(r) <= w for w in pos_words)]
    p.negative = edit_prompt(p.negative, (edit.get("negative_add") or []) + neg_add, edit.get("negative_remove") or [])
    added = ", ".join(add)
    if added:  # a tag the judge adds shouldn't stay banned by the negative
        p.negative, _ = drop_negative_conflicts(added, p.negative)
    if edit.get("mode") in MODES and edit["mode"] != p.mode:
        p.mode = edit["mode"]
        p.denoise = DEFAULT_DENOISE[p.mode]  # txt2img's 1.0 would discard the image entirely
    for name in ("steps", "cfg", "denoise"):
        if edit.get(name) is not None:
            setattr(p, name, clamp(name, edit[name]))
    if edit.get("sampler_name") in samplers:
        p.sampler_name = edit["sampler_name"]
    if edit.get("scheduler") in schedulers:
        p.scheduler = edit["scheduler"]
    p.mask_target = (edit.get("mask_target") or "").strip() or None
    if p.mask_target and p.mode not in INPAINT_MODES:
        # A named region means a local fix, and low-denoise img2img can't add or
        # remove details: repaint just that region of the best result instead.
        p.mode = "inpaint_best"
        if edit.get("denoise") is None:
            p.denoise = 0.75
    if p.mode == "txt2img":
        p.denoise = 1.0
    if p.mode in INPAINT_MODES and not p.mask_target:
        p.mode = p.mode.replace("inpaint", "img2img")
        if edit.get("denoise") is None:  # an inpaint denoise (0.75) would redraw most of the image
            p.denoise = DEFAULT_DENOISE[p.mode]
    if p.mode in INPAINT_MODES and p.denoise < MIN_INPAINT_DENOISE:
        p.denoise = MIN_INPAINT_DENOISE  # below this a repaint can't change a colour or remove a detail
    if not edit.get("keep_seed", False):
        p.seed = random.randrange(2**32)
    return p


def variants(p: GenParams, n: int, phase: str) -> list[GenParams]:
    """The candidates rendered in one round. GPU time is cheap next to an API round
    trip, so each round explores several points around the planner's choice:
    new seeds while exploring, a cfg/denoise spread around a locked seed later."""
    if phase == "explore" or p.mode == "img2img_best" or p.mode in INPAINT_MODES:
        # Reworking an existing image: with the same seed, nearby denoise values give
        # near-identical results, so vary the seed (and denoise a little) instead.
        out = [p]
        for i in range(1, n):
            dn = p.denoise if phase == "explore" else clamp("denoise", round(p.denoise + (0.05 if i % 2 else -0.05) * ((i + 1) // 2), 2))
            out.append(replace(p, seed=random.randrange(2**32), denoise=dn))
        return out
    out = [p]
    offsets = [-1.0, 1.0, -2.0, 2.0, -0.5, 0.5]
    for off in offsets[: n - 1]:
        if p.mode == "txt2img":
            v = replace(p, cfg=clamp("cfg", p.cfg + off))
        else:
            v = replace(p, denoise=clamp("denoise", round(p.denoise + off * 0.08, 2)))
        if v in out:  # hit a bound; spend the slot on a new seed instead
            v = replace(v, seed=random.randrange(2**32))
        out.append(v)
    return out
