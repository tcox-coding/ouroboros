"""Run one job: (write prompt) -> render -> pre-filter -> judge+plan -> confirm or
edit -> repeat.

Progress goes out through report(event_dict), which the CLI prints and the web UI
shows live. should_stop() is checked between rounds.
"""

from __future__ import annotations

import json
import random
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from . import masks, scoring
from . import pose as posemod
from . import pose_picker
from . import autofix as autofix_mod
from . import nobg as nobg_mod
from . import settings_advisor as advisor
from . import upscale as upscale_mod
from .handfix import refine_hands
from .comfy import Cancelled, ComfyClient
from .jobs import Job
from .judge import Judge
from .lora_picker import pick_loras, prompt_notes
from .loras import LoraLibrary, checkpoint_base, compatible, lora_folders_for, with_triggers
from .params import (DEFAULT_DENOISE, MODES, GenParams, allowed_modes, apply_edit, enforce_reference_rules,
                     lora_stem, norm_tag, reseed_duplicates, split_tags, variants)
from .prompter import names_style, write_prompt
from .sizes import check_size, fit_to, output_size
from .targets import Target, cap_ip_weights, max_combined, prompt_parts, subject_cutout_default, subject_preset
from .workflow import Workflows

Report = Callable[[dict], None]


def print_report(event: dict) -> None:
    if event["type"] == "log":
        print(event["text"])


@dataclass
class Result:
    status: str          # "done" (threshold), "review" (budget/plateau), "stopped" (user)
    best_score: float
    best_image: Path | None
    rounds: int
    cost_usd: float
    run_dir: Path


class JudgeMemory:
    """What the judge is told about earlier rounds.

    Every call starts a fresh context (nothing accumulates like a chat), and models
    work best on what's near the start of it. So notes are kept short: the latest
    rounds verbatim, everything older folded into a summary the LLM writes. The
    summary is refreshed when there are more than `keep_rounds` verbatim notes, or
    when the last judge prompt used more than `token_budget` tokens. The budget is
    absolute, not a share of the window: accuracy drops with prompt length long
    before a large window is full.
    """

    def __init__(self, judge: Judge, keep_rounds: int, token_budget: int):
        self.judge, self.keep_rounds, self.token_budget = judge, max(1, keep_rounds), token_budget
        self.summary = ""
        self.entries: list[str] = []

    def text(self) -> str:
        parts = [f"Summary of earlier rounds: {self.summary}"] if self.summary else []
        return "\n".join(parts + self.entries)

    def add(self, entry: str) -> None:
        self.entries.append(entry)

    def compact_if_needed(self, prompt_tokens: int) -> tuple[str | None, float]:
        """Returns (new summary or None, cost)."""
        too_many = len(self.entries) > self.keep_rounds
        too_full = prompt_tokens > self.token_budget
        if not (too_many or too_full) or len(self.entries) < 2:
            return None, 0.0
        old, self.entries = self.entries[:-1], self.entries[-1:]
        self.summary, cost = self.judge.summarize(self.summary, old)
        return self.summary, cost


FINE_TUNING_MODES = {"img2img_best", "inpaint_best", "inpaint_reference"}


def criteria_of(raw: dict, index: int) -> dict:
    """Per-criterion scores (0-10) the judge gave candidate `index`."""
    for c in raw.get("candidates", []):
        if c.get("index") == index:
            return c.get("scores", {})
    return {}


def differences_of(raw: dict, index: int) -> list[str]:
    for c in raw.get("candidates", []):
        if c.get("index") == index:
            return [str(d) for d in c.get("differences") or []]
    return []


def gate_passed(criteria: dict, lc: dict) -> bool:
    gate = lc.get("refine_gate") or {}
    return bool(criteria) and bool(gate) and all(criteria.get(k, 0) >= v for k, v in gate.items())


def gate_fine_tuning(edit: dict, criteria: dict, lc: dict, modes: list[str], log, stalled: bool = False) -> dict:
    """Coarse to fine, enforced: reworking or repainting the best image only helps once
    it's close. While identity or composition are below loop.refine_gate, a refine or
    repair edit becomes an explore edit: img2img from the reference if composition is
    off (it borrows the reference's layout), else txt2img, with the judge's prompt
    edits and new seeds."""
    gate = lc.get("refine_gate") or {}
    low = {k: criteria[k] for k, need in gate.items() if k in criteria and criteria[k] < need}
    if low and stalled and edit.get("mode") == "txt2img" and "img2img_reference" in modes:
        # In testing, more prompt tags on txt2img rarely fixed a wrong character; img2img
        # from the reference (at >= the minimum denoise) jumped from ~50 to ~80 each time.
        log("txt2img didn't improve (" + ", ".join(f"{k} {v}/10" for k, v in low.items())
            + "): trying img2img from the reference")
        return {**edit, "mode": "img2img_reference", "phase": "explore", "mask_target": None,
                "keep_seed": False, "denoise": None}
    if not low or edit.get("mode") not in FINE_TUNING_MODES:
        return edit
    mode = "img2img_reference" if "composition" in low and "img2img_reference" in modes else "txt2img"
    log("big differences remain (" + ", ".join(f"{k} {v}/10" for k, v in low.items())
        + f"): exploring with {mode} instead of {edit.get('mode')}")
    return {**edit, "mode": mode, "phase": "explore", "mask_target": None, "keep_seed": False, "denoise": None}


def lora_sweep(p: GenParams, alternatives: list[tuple[str, float]], managed: set[str], n: int,
               max_loras: int, baseline: tuple = ()) -> list[GenParams]:
    """Round 1 when LoRAs were picked: the picked set, the workflow's own set (the
    baseline the picks have to beat), no LoRAs at all (a pick has to earn its place),
    then sets with one alternative swapped in, all on the same seed so the judge compares
    the LoRAs, not the seeds. Round 1's batch is the largest, so this is where the most
    LoRA options are tried."""
    out = [p]
    if baseline and tuple(baseline) != tuple(p.loras):
        out.append(replace(p, loras=tuple(baseline)))
    picked = [(name, w) for name, w in p.loras if name in managed]
    pinned = [(name, w) for name, w in p.loras if name not in managed]
    if picked and len(out) < n:
        out.append(replace(p, loras=tuple(pinned)))
    for alt in alternatives:
        if len(out) >= n:
            break
        if alt[0] in {m for m, _ in picked}:
            continue
        if picked:
            weakest = min(range(len(picked)), key=lambda i: picked[i][1])
            new = picked[:weakest] + picked[weakest + 1:] + [alt]
        else:
            new = [alt]
        out.append(replace(p, loras=tuple(pinned + new[:max_loras])))
    while len(out) < n:
        out.append(replace(p, seed=random.randrange(2**32)))
    return out


# ---- how many candidates a round renders -------------------------------------------------
MAX_BATCH = 16
FOCUSES = ("prompt", "lora_weights", "loras", "settings")


def batch_for_round(rnd: int, phase: str, best: float | None, first: float | None, threshold: float,
                    lc: dict, fixed: int | None = None) -> int:
    """Wide early, narrow late. Round 1 renders loop.batch_start candidates (up to 16),
    while nothing is known and the LoRAs and prompt are still being chosen; the batch
    then shrinks toward loop.batch_end (2-4) as the job gets tuned. "Tuned" is whichever
    is furthest along: rounds done (30% fewer each round), the phase (refine is most of
    the way, repair all of it), or the score's progress from round 1 to the threshold.
    A job that sets candidates_per_round itself keeps that fixed size."""
    if fixed:
        return max(1, min(MAX_BATCH, int(fixed)))
    start = max(2, min(MAX_BATCH, int(lc.get("batch_start") or 12)))  # empty/0 in Settings = the default
    end = max(1, min(start, int(lc.get("batch_end") or 3)))
    progress = 1 - 0.7 ** (rnd - 1)
    progress = max(progress, {"explore": 0.0, "refine": 0.6, "repair": 1.0}.get(phase, 0.0))
    if best is not None and first is not None and threshold > first:
        progress = max(progress, min(1.0, max(0.0, (best - first) / (threshold - first))))
    return max(end, min(start, round(start - (start - end) * progress)))


# ---- one axis per edit: the prompt or the LoRAs, not both ----------------------------------

def enforce_focus(edit: dict, current: dict[str, float], allowed: dict[str, str], switch_ok: bool) -> dict:
    """Make an edit change one axis, so the next round shows what that change did.

    focus "prompt": prompt/negative tags (with settings or a masked repaint); LoRAs stay.
    focus "lora_weights": only the strengths of the LoRAs in use; the prompt stays.
    focus "loras": a different set of LoRAs (a swap, an addition, or none); prompt stays.
    focus "settings": mode, cfg, denoise, sampler, mask; neither prompt nor LoRAs.
    current: stem -> strength of the managed LoRAs in use. switch_ok: whether the set
    may still change (only early in a job); otherwise a switch becomes a weight change."""
    e = dict(edit)
    loras = e.get("loras")
    prompt_keys = ("prompt_add", "prompt_remove", "negative_add", "negative_remove")
    has_prompt = any(e.get(k) for k in prompt_keys)
    focus = e.get("focus") if e.get("focus") in FOCUSES else None
    if focus is None:  # infer it from what the edit touches
        if loras is not None and {str(l.get("lora")) for l in loras} != set(current):
            focus = "loras"
        elif loras is not None:
            focus = "lora_weights"
        else:
            focus = "prompt" if has_prompt else "settings"
    if focus == "loras" and not switch_ok:
        focus = "lora_weights"
    if focus == "lora_weights" and not current:
        focus = "prompt" if has_prompt else "settings"
    if focus in ("prompt", "settings"):
        e["loras"] = None
    if focus == "settings":
        for k in prompt_keys:
            e[k] = []
    if focus in ("lora_weights", "loras"):
        for k in prompt_keys:
            e[k] = []
    if focus == "lora_weights":
        # The same LoRAs, only new strengths: any LoRA the edit didn't mention keeps its own.
        asked = {str(l.get("lora")): l.get("strength") for l in loras or [] if str(l.get("lora")) in current}
        e["loras"] = [{"lora": stem, "strength": asked.get(stem, w) if asked.get(stem) is not None else w}
                      for stem, w in current.items()]
    if focus == "loras" and loras is not None:
        e["loras"] = [l for l in loras if str(l.get("lora")) in allowed]
    e["focus"] = focus
    return e


def weight_variants(p: GenParams, n: int, managed: set[str], ranges: dict[str, tuple[float, float]]) -> list[GenParams]:
    """A round that tunes LoRA strengths: the proposed strengths plus a sweep around
    them, one LoRA at a time, all on the same seed so only the strength differs."""
    out = [p]
    steps = (-0.15, 0.15, -0.3, 0.3, -0.45, 0.45)
    for d in steps:
        for i, (name, w) in enumerate(p.loras):
            if len(out) >= n:
                return out
            if name not in managed:
                continue
            lo, hi = ranges.get(name, (0.1, 1.5))
            nw = round(min(hi, max(lo, w + d)), 2)
            loras = tuple((m, nw if j == i else x) for j, (m, x) in enumerate(p.loras))
            v = replace(p, loras=loras)
            if v not in out:
                out.append(v)
    while len(out) < n:
        out.append(replace(p, seed=random.randrange(2**32)))
    return out


def set_variants(p: GenParams, previous: GenParams, n: int, managed: set[str], options: list[tuple[str, float]],
                 max_loras: int) -> list[GenParams]:
    """A round that switches LoRAs: the proposed set, the set it replaces, no LoRAs at
    all, then the proposed set with its weakest LoRA swapped for each other option -
    all on the same seed, so the judge compares LoRAs, not seeds."""
    pinned = tuple((m, w) for m, w in p.loras if m not in managed)
    chosen = [(m, w) for m, w in p.loras if m in managed]
    out = [p]

    def add(loras):
        v = replace(p, loras=tuple(pinned) + tuple(loras[:max_loras]))
        if v not in out and len(out) < n:
            out.append(v)
    add([(m, w) for m, w in previous.loras if m in managed])
    add([])
    for alt in options:
        if alt[0] in {m for m, _ in chosen}:
            continue
        if chosen:
            weakest = min(range(len(chosen)), key=lambda i: chosen[i][1])
            add(chosen[:weakest] + chosen[weakest + 1:] + [alt])
        else:
            add([alt])
    while len(out) < n:
        out.append(replace(p, seed=random.randrange(2**32)))
    return out


def resolve_loras(loras: tuple, index: dict, installed: list[str] | None, log, what: str = "workflow") -> tuple:
    """The workflow's saved LoRAs, pointed at where they are now. A workflow saved
    before the library was reorganised (or on another machine) names LoRAs by an old
    path, e.g. Pony\\styles\\X.safetensors for what is now Pony/styles/artists/x/X.safetensors;
    ComfyUI skips a LoRA it can't find without a word. Matched by file name: the
    library's entry first (so the LoRA can be compared, re-weighted or dropped like any
    other), else ComfyUI's own path; one that isn't installed at all is left out."""
    norm = lambda n: n.replace("\\", "/").lower()
    by_stem: dict[str, str] = {}
    for n in index:
        by_stem.setdefault(lora_stem(n).lower(), n)
    have = {norm(n): n for n in installed or []}
    for n in installed or []:
        by_stem.setdefault(lora_stem(n).lower(), n)
    out, seen = [], set()
    for name, w in loras:
        found = name if name in index or (installed is not None and norm(name) in have) else by_stem.get(lora_stem(name).lower())
        if found is None and installed is None:
            found = name  # can't check: keep it as saved
        if found is None:
            log(f"{what} LoRA {lora_stem(name)} isn't installed; leaving it out")
            continue
        if found != name:
            log(f"{what} LoRA {lora_stem(name)} found at {found.replace(chr(92), '/')}")
        if norm(found) not in seen:
            seen.add(norm(found))
            out.append((found, w))
    return tuple(out)


def _flag(value, default: bool) -> bool:
    """A job setting that is a bool, or a string from a settings.txt ("false", "0", "off")."""
    if value is None or value == "":
        return default
    return str(value).strip().lower() not in ("false", "0", "off", "no")


def _num(value, default) -> float:
    try:
        return float(value) if value not in (None, "") else float(default)
    except (TypeError, ValueError):
        return float(default)


def initial_params(job: Job, defaults: dict) -> GenParams:
    s = {**defaults, **job.settings}
    mode = s.get("mode", "txt2img")
    return GenParams(
        mode=mode,
        positive=job.positive,
        negative=job.negative,
        seed=int(s["seed"]) if s.get("seed") not in (None, "") else random.randrange(2**32),
        steps=int(s.get("steps", 30)),
        cfg=float(s.get("cfg", 5.5)),
        sampler_name=s.get("sampler_name", "dpmpp_2m"),
        scheduler=s.get("scheduler", "karras"),
        denoise=float(s.get("denoise", 1.0 if mode == "txt2img" else 0.65)),
    )


def lora_context(job, tg, style_ref, poses_dir: Path, lc: dict, cfg: dict) -> list[tuple]:
    """The job's subject and pose images for the LoRA picker, each (role, image, tags), beside
    the style reference (`style_ref`, not repeated). The pose: the job's pose image, else the
    saved pose it names; "auto" hasn't been picked yet and "reference" is the main image."""
    def same(a, b) -> bool:
        try:
            return a is not None and b is not None and Path(a).resolve() == Path(b).resolve()
        except (TypeError, OSError):
            return a is b
    out = []
    if tg.subject.image is not None and not same(tg.subject.image, style_ref):
        out.append(("subject", tg.subject.image, tg.subject.text))
    pose_img, pose_tags = tg.pose.image, tg.pose.text
    setting = str(job.settings.get("pose") or lc.get("pose") or cfg.get("defaults", {}).get("pose") or "off")
    if pose_img is None and setting not in ("off", "auto", "reference"):
        try:
            lib = posemod.PoseLibrary(poses_dir)
            pose_img, pose_tags = lib.source(setting), lib.get(setting).get("description", "")
        except (OSError, ValueError):
            pose_img = None
    if pose_img is not None and not same(pose_img, style_ref) and not same(pose_img, tg.subject.image):
        out.append(("pose", pose_img, pose_tags))
    return out


from . import usage


@usage.scoped
def run_job(job: Job, cfg: dict, comfy: ComfyClient, flows: Workflows, judge: Judge,
            samplers: list[str], schedulers: list[str], runs_dir: Path,
            report: Report = print_report, should_stop: Callable[[], bool] = lambda: False,
            library: LoraLibrary | None = None) -> Result:
    def log(text: str) -> None:
        report({"type": "log", "text": f"[{job.name}] {text}"})

    def record(entry: dict) -> None:
        with open(out_dir / "log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    lc = {**cfg["loop"], **job.overrides}
    threshold, max_rounds, max_cost = lc["threshold"], lc["max_rounds"], lc["max_cost_usd"]
    # Optional two-stage goal: once the best score reaches stretch.from, the job gets
    # stretch.extra_rounds more rounds (from that round) to reach the threshold, and the
    # plateau rule no longer ends it early. milestones records the round each was reached.
    stretch = lc.get("stretch") or {}
    milestones: dict[str, int] = {}
    out_dir = runs_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{job.name}"
    out_dir.mkdir(parents=True)
    reference = out_dir / f"reference{job.reference.suffix}"
    shutil.copy2(job.reference, reference)
    report({"type": "job_start", "job": job.name, "run_dir": out_dir, "reference": reference,
            "threshold": threshold, "max_rounds": max_rounds, "context_window": judge.context_window})
    cost = 0.0
    prompt_backend = getattr(judge, "prompt_backend", judge.backend)
    from .model_profiles import confirmation_threshold
    confirm_threshold = confirmation_threshold(cfg["judge"], threshold)
    bk = cfg["judge"]["backend"]
    judge_model = cfg["judge"].get(bk, {}).get("model", "")
    confirm_model = cfg["judge"].get("confirm_model") or judge_model

    # 0. Saved style / character "AI picks" (reflib): the chosen item becomes that target.
    for role in ("style", "subject"):
        if str(job.settings.get(f"{role}_library") or "") != "auto":
            continue
        report({"type": "stage", "round": 0, "phase": "explore", "stage": f"choosing a saved {role}"})
        try:
            from . import reflib
            got = reflib.resolve(runs_dir.parent, role, "auto", backend=judge.backend, description=job.description,
                                 positive=job.positive, max_side=cfg["judge"].get("image_max_side", 512))
            pick = (got or {}).get("pick") or {}
            cost += pick.get("cost", 0.0)
            if got and got.get("image"):
                old = job.targets.get(role)
                setattr(job.targets, role, Target(image=Path(got["image"]), name=got["name"],
                                                  text=old.text or got["description"]))
                log(f"{role} picked: {got['name']}" + (f" ({pick['reason']})" if pick.get("reason") else ""))
            else:
                log(f"no saved {role} fits" + (f" ({pick['reason']})" if pick.get("reason") else ""))
        except Cancelled:
            raise
        except Exception as e:
            cost += float(getattr(e, "cost_usd", 0.0))
            log(f"warning: choosing a saved {role} failed ({str(e)[:200]})")
    tg = job.targets
    style_from, style_words = None, ""
    if job.description.strip() and tg.missing() and len(tg.missing()) < 3:
        # Some targets given, others left to the description ("this character, arms crossed,
        # in flat colours"): judge the missing ones against the goal, not the reference image.
        tg = replace(tg, goal_for_missing=True)
        if "style" in tg.missing() and tg.subject.image is not None:
            # No style given: a style the description names is the aim; with none named, the
            # run keeps the character's own look, so the character image is the style target too.
            report({"type": "stage", "round": 0, "phase": "explore", "stage": "reading the description's style"})
            try:
                named, words, c = names_style(prompt_backend, job.description)
                cost += c
            except Cancelled:
                raise
            except Exception as e:  # keep judging the style against the description, as it says
                cost += float(getattr(e, "cost_usd", 0.0))
                named, words = True, ""
                log(f"warning: couldn't tell whether the description names an art style ({str(e)[:160]}); "
                    "judging the style against the description")
            if named:
                style_from, style_words = "description", words
                log("style: from the description" + (f" ({words})" if words else ""))
            else:
                style_from = "character image"
                tg = replace(tg, style=Target(image=tg.subject.image, name=tg.subject.name))
                log("style: the description names none, so the character image's own look")
    # 0a. Checkpoint and LoRAs. "auto": the LLM picks style LoRAs from the managed folders
    # (Settings -> LoRAs) by comparing the reference with each LoRA's Civitai examples;
    # LoRAs outside those folders stay as the workflow has them. "workflow": the saved
    # workflow's LoRAs, strengths tunable. "fixed": exactly the LoRAs chosen by hand
    # (settings.loras, e.g. from Home) for the whole job: strengths tunable, never swapped,
    # added or dropped. "off": none.
    lcfg = cfg.get("loras", {})
    checkpoint = job.settings.get("checkpoint") or cfg.get("defaults", {}).get("checkpoint") or None
    ckpt_name = checkpoint or flows.default("checkpoint")
    ckpt_base = checkpoint_base(ckpt_name, cfg.get("checkpoint_bases"))
    from .ckpt_info import checkpoint_note, prompt_setup
    ckpt_text = checkpoint_note(cfg, ckpt_name)  # the model and its tag conventions, for the LLMs
    lora_mode = lc.get("lora_mode") or lcfg.get("mode", "workflow")
    max_loras = int(lcfg.get("max_loras", 3))
    index = library.index() if library else {}
    try:
        installed = comfy.choices("LoraLoader", "lora_name")
    except Exception:
        installed = None
    workflow_loras = resolve_loras(flows.default_loras(), index, installed, log)
    start_loras: tuple = () if lora_mode == "off" else workflow_loras
    fixed_loras = lora_mode == "fixed"
    if fixed_loras:
        chosen = tuple((str(l["name"]), float(l.get("strength") if l.get("strength") is not None else 0.8))
                       for l in job.settings.get("loras") or [] if isinstance(l, dict) and l.get("name"))
        start_loras = resolve_loras(chosen, index, installed, log, "selected")
        log("LoRAs: only the selected ones for the whole job: "
            + (", ".join(f"{lora_stem(n)} {w:g}" for n, w in start_loras) or "none"))
        max_loras = max(max_loras, len(start_loras))
    alternatives: list[tuple[str, float]] = []
    lora_pick: dict | None = None
    if lora_mode == "auto" and index:
        report({"type": "stage", "round": 0, "phase": "explore", "stage": "choosing LoRAs"})
        pinned = tuple((n, w) for n, w in workflow_loras if n not in index)
        try:
            # The style to match: the style image, else the description's style when it names
            # one (no image then), else the main image (the character's own look).
            style_ref = None if style_from == "description" else tg.style.image or job.reference
            context = lora_context(job, tg, style_ref, runs_dir.parent / "poses", lc, cfg)
            if context:
                log("LoRA choice also sees the " + " and ".join(f"{r} image" for r, _, _ in context))
            lora_pick = pick_loras(judge.backend, library, style_ref,
                                   job.description or job.positive or flows.default("positive"), ckpt_base,
                                   max_loras, {**cfg["judge"], **lcfg},
                                   style_text=(style_words or job.description) if style_from == "description" else "",
                                   context=context, folders=lora_folders_for(ckpt_name, cfg))
            start_loras = pinned + tuple((n, w) for n, w, _ in lora_pick["picks"])
            alternatives = lora_pick["alternatives"]
            log("LoRAs picked: " + (", ".join(f"{lora_stem(n)} {w:g}" for n, w, _ in lora_pick["picks"]) or "none")
                + (f"; alternatives to test: {', '.join(lora_stem(n) for n, _ in alternatives)}" if alternatives else ""))
        except Exception as e:  # a failed pick shouldn't cost the job: fall back to the workflow's LoRAs
            cost += float(getattr(e, "cost_usd", 0.0))
            log(f"warning: LoRA picking failed ({str(e)[:200]}); using the workflow's LoRAs")
    elif lora_mode == "auto":
        log("LoRA mode is auto but the LoRA index is empty (Settings -> LoRAs -> Refresh); using the workflow's")
    active_managed = [n for n, _ in start_loras if n in index]
    # The judge may switch between the picked LoRAs, the alternatives and the workflow's.
    shortlisted = [n for n in (lora_pick or {}).get("shortlist", []) if n in index]
    lora_choices = {lora_stem(n): n for n in
                    dict.fromkeys(active_managed + [n for n, _ in alternatives] + shortlisted
                                  + [n for n, _ in workflow_loras if n in index])} if lora_mode != "off" else {}
    if fixed_loras:  # only the selected ones, so their strengths can be tuned
        lora_choices = {lora_stem(n): n for n in active_managed}
    managed_set = set(lora_choices.values())

    def usual_weight(name: str) -> float:
        w = (index.get(name) or {}).get("weight_range") or {}
        return float(w.get("default") or index.get(name, {}).get("typical_weight") or 0.8)

    def weight_range(name: str) -> tuple[float, float]:
        w = (index.get(name) or {}).get("weight_range") or {}
        return float(w.get("min", 0.1)), float(w.get("max", 1.5))
    # Round 1 tries the shortlisted LoRAs that weren't picked too, at their usual weight.
    alternatives = alternatives + [(n, usual_weight(n)) for n in shortlisted
                                   if n not in {a for a, _ in alternatives} | set(active_managed)]
    switch_rounds = 0 if fixed_loras else int(lc.get("lora_switch_rounds") or 3)  # 0: the set never changes
    lora_menu = None
    if lora_choices:
        lora_menu = {"menu": "\n".join("- " + library.card(index[n], detail=False) for n in lora_choices.values()),
                     "stems": list(lora_choices), "max": max_loras, "switch": True}
    # 0a'. Pose ControlNet (optional). "reference": every render follows the reference's
    # pose. A pose library name: the character comes from the reference and the pose from
    # that pose image; nothing may start from the reference then (img2img would bring its
    # pose back), the output size follows the pose image's shape, and the judge scores
    # composition against the pose.
    cn = cfg.get("controlnet", {})
    pose_setting = str(job.settings.get("pose") or lc.get("pose") or cfg.get("defaults", {}).get("pose") or "off")
    pose_data, pose_source, pose_desc = None, None, ""
    pose_pick = None
    if pose_setting == "auto":  # the LLM picks a saved pose that fits, or none (pose_picker.py)
        pose_setting = "off"
        if posemod.available():
            report({"type": "stage", "round": 0, "phase": "explore", "stage": "choosing a pose"})
            try:
                from_image = not job.description.strip() and not job.positive.strip()
                pose_pick = pose_picker.pick_pose(
                    judge.backend, pose_picker.library_menu(posemod.PoseLibrary(runs_dir.parent / "poses")),
                    description=job.description, positive=job.positive,
                    reference=job.reference if from_image else None, max_side=cfg["judge"].get("image_max_side", 512))
                cost += pose_pick["cost"]
                pose_setting = pose_pick["pose"] or "off"
                log((f"pose picked: {pose_pick['pose']}" if pose_pick["pose"] else "no saved pose fits; continuing without one")
                    + (f" ({pose_pick['reason']})" if pose_pick["reason"] else ""))
            except Cancelled:
                raise
            except Exception as e:  # optional: the job goes on without a pose
                cost += float(getattr(e, "cost_usd", 0.0))
                log(f"warning: choosing a pose failed ({str(e)[:200]}); continuing without one")
    if tg.pose.image and pose_setting in ("off", "reference") and \
            str(job.settings.get("pose_control", True)).lower() not in ("false", "0", "off"):
        pose_setting = "image"  # the job's own pose image (a pose target), through the ControlNet
    if pose_setting != "off":
        if not posemod.available():
            log("warning: the pose ControlNet needs rtmlib (pip install rtmlib); continuing without a pose")
        elif pose_setting == "reference":
            pose_data, pose_source = posemod.detect(job.reference), job.reference
        elif pose_setting == "image":
            pose_data, pose_source, pose_desc = posemod.detect(tg.pose.image), tg.pose.image, tg.pose.text
        else:
            try:
                library_poses = posemod.PoseLibrary(runs_dir.parent / "poses")
                pose_data, pose_source = library_poses.get(pose_setting), library_poses.source(pose_setting)
                pose_desc = pose_data.get("description", "")
            except FileNotFoundError:
                log(f"warning: pose '{pose_setting}' isn't in the pose library; continuing without a pose")
        if posemod.available() and pose_source and not posemod.has_body(pose_data):
            log("warning: no person found in the pose image; continuing without a pose")
            pose_data = None
        if pose_data and pose_setting != "reference":
            lc["pose_from_library"] = True
    if tg.goal_for_missing and "pose" in tg.missing() and not lc.get("pose_from_library"):
        lc["pose_from_goal"] = True  # the pose is only in the description
        log("the pose comes from the description: judged against the goal, never by repainting the reference")
    run_info = {"job": job.name, "checkpoint": ckpt_name, "checkpoint_base": ckpt_base,
                "workflow": flows.spec.get("file"), "fixed_inputs": flows.fixed_inputs(),
                "judge_backend": cfg["judge"]["backend"],
                "judge_model": judge_model, "confirm_model": confirm_model,
                "prompt_model": cfg["judge"].get("prompt_model") or judge_model,
                "confirm_threshold": confirm_threshold,
                "threshold": threshold, "lora_mode": lora_mode, "workflow_loras": workflow_loras,
                "start_loras": start_loras, "lora_pick": lora_pick,
                "pose": {"setting": pose_setting, "active": bool(pose_data), "description": pose_desc,
                         "picked_by_llm": pose_pick},
                "targets": tg.to_dict(), "style_from": style_from,
                "incompatible_loras": [lora_stem(n) for n, _ in start_loras
                                       if compatible(index.get(n, {}).get("base_model"), ckpt_base) is False]}
    if run_info["incompatible_loras"]:
        log(f"warning: {', '.join(run_info['incompatible_loras'])} may not suit a {ckpt_base} checkpoint")
    report({"type": "setup", **run_info})

    user_tags = {norm_tag(t) for t in split_tags(job.positive)}  # the user's own words stay
    # 0b. Starting prompt: written from the description (merged with any prompt given),
    # or from the reference image alone when the job has neither, or the job's prompt.
    from_image = not job.description.strip() and not job.positive.strip()
    if job.description or from_image:
        # Separate targets: each part of the prompt from its own target (targets.prompt_parts).
        # A saved pose (or pose image) the ControlNet imposes is the prompt's POSE target, so the
        # pose words come from it and not from the character image, whose own pose the prompt
        # writer otherwise copies ("arms at sides" under a crossed-arms skeleton, 2026-10-04).
        ptg = tg
        if lc.get("pose_from_library") and pose_desc.strip() and tg.pose.empty:
            ptg = replace(tg, pose=Target(text=pose_desc.strip(), name=pose_setting))
        report({"type": "stage", "round": 0, "phase": "explore",
                "stage": "writing prompt from " + ("the style, subject and pose" if ptg.split
                                                   else "the reference image" if from_image else "description")})
        written = write_prompt(prompt_backend, job.description, job.positive, job.negative,
                               flows.default("positive"), flows.default("negative"),
                               None if ptg.split else
                               job.reference if from_image or lc.get("prompt_sees_reference", True) else None,
                               cfg["judge"].get("image_max_side", 512),
                               prompt_notes(library, start_loras) if library else "",
                               pose_desc if lc.get("pose_from_library") else "",
                               targets=prompt_parts(ptg) if ptg.split else None, **prompt_setup(cfg, ckpt_name))
        prompt_info = {"description": job.description or "(none: written from the reference image)",
                       "input_positive": job.positive,
                       "input_negative": job.negative, "merged": bool(job.positive or job.negative), **written}
        (out_dir / "prompt.json").write_text(json.dumps(prompt_info, indent=2), encoding="utf-8")
        report({"type": "prompt", **prompt_info})
        log("prompt written from " + ("the reference image" if from_image else "description")
            + (" (merged with your prompt)" if prompt_info["merged"] else ""))
        job.positive, job.negative = written["positive"], written["negative"]
    job.positive = job.positive or flows.default("positive")
    job.negative = job.negative or flows.default("negative")
    goal = (f"Description: {job.description}\n\nPrompt: {job.positive}" if job.description else job.positive)
    run_info.update(positive=job.positive, negative=job.negative, description=job.description)
    (out_dir / "run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    (out_dir / "graphs").mkdir()

    def rendered_positive(p: GenParams) -> str:
        return with_triggers(p.positive, p.loras, library, split_tags, norm_tag, user_tags)

    # Output size follows the reference's shape (or the job's / Settings' choice). The
    # reference itself is fitted to that size for rendering, since img2img encodes it
    # as-is; the judge still compares against the original.
    size_setting = job.settings.get("size") or cfg.get("defaults", {}).get("size") or "auto"
    size = output_size(size_setting, pose_source if lc.get("pose_from_library") else job.reference,
                       flows.workflow_size())
    render_ref = fit_to(job.reference, size, out_dir / "reference_input.png")
    run_info["size"] = list(size)
    run_info["size_setting"] = size_setting
    (out_dir / "run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    log(f"output size {size[0]}x{size[1]} ({size_setting})")
    if check_size(size)[1]:
        log("warning: " + check_size(size)[1])
    report({"type": "size", "size": list(size), "size_setting": size_setting})
    def upload(path: Path) -> str:
        return comfy.upload_image(path, f"ouroboros/{out_dir.name}")

    control, pose_judge = None, None
    if pose_data:
        skeleton = out_dir / "pose_control.png"
        posemod.render(posemod.fit(pose_data, tuple(pose_data["size"]), size), size,
                       hands=cn.get("pose_hands", True), face=cn.get("pose_face", False)).save(skeleton)
        control = {"model": cn.get("model", "xinsir_union_sdxl_promax.safetensors"), "image": upload(skeleton),
                   "type": "openpose", "strength": float(cn.get("pose_strength", 0.6)), "start": 0.0,
                   "end": float(cn.get("pose_end", 0.8))}
        if lc.get("pose_from_library"):
            pose_judge = {"image": skeleton, "description": pose_desc}
        run_info["pose"].update(control=skeleton.name, strength=control["strength"], end=control["end"])
        (out_dir / "run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
        report({"type": "setup", **run_info})
        log(f"pose ControlNet on ({'the reference' if pose_setting == 'reference' else pose_setting}, "
            f"strength {control['strength']:g})")

    local = {"reference": render_ref}                # source role -> local file
    uploaded = {"reference": upload(render_ref)}     # source role -> ComfyUI name
    # IP-Adapters: the subject image carries the character (linear), the style image the look
    # (style transfer). Only for images given as targets; a one-reference job renders as before.
    # A job from Home -> Run automatically carries the form's settings (on/off, weights, the
    # character's preset, background removal); anything it doesn't set comes from config.json.
    ipa = []
    ip_cfg = cfg.get("ipadapter", {})
    js = job.settings
    if ip_cfg.get("enabled", True) and lc.get("ipadapter", True):
        for role, weight, wtype in (("subject", _num(js.get("subject_ip_weight"), ip_cfg.get("subject_weight", 0.6)),
                                     ip_cfg.get("weight_type", "linear")),
                                    ("style", _num(js.get("style_ip_weight"), ip_cfg.get("style_weight", 0.5)),
                                     ip_cfg.get("style_weight_type", "style transfer"))):
            img = tg.get(role).image
            if img is None or float(weight or 0) <= 0 or (role == "style" and img == tg.subject.image) \
                    or not _flag(js.get(f"{role}_ip"), True):
                continue
            ip_img = img
            if role == "subject" and _flag(js.get("subject_cutout"), subject_cutout_default(ip_cfg)):
                try:  # its background tints every render otherwise (nobg.cutout)
                    ip_img = nobg_mod.cutout(img, comfy=comfy, cfg=cfg, cache_dir=runs_dir.parent / "cache" / "cutouts",
                                             upload=upload)
                    run_info["subject_cutout"] = True
                    log("IP-Adapter: the character image with its background removed")
                except Exception as e:
                    cost += float(getattr(e, "cost_usd", 0.0))
                    log(f"warning: couldn't remove the character image's background ({str(e)[:160]}); using it as it is")
            if role == "subject":  # the encoder sees a centre square: pad, or the head is cut off
                ip_img = nobg_mod.square_for_ip(ip_img, runs_dir.parent / "cache" / "cutouts")
            ipa.append({"image": upload(ip_img),
                        "preset": (js.get("ip_preset") or subject_preset(ip_cfg)) if role == "subject"
                        else ip_cfg.get("preset", "PLUS (high strength)"),
                        "weight": float(weight), "weight_type": wtype, "start": 0.0,
                        "end": float(ip_cfg.get("end", 1.0)), "role": role})
            shutil.copy2(img, out_dir / f"target_{role}{img.suffix}")
        scaled = cap_ip_weights(ipa, max_combined(ip_cfg))
        if scaled:
            log(scaled)
            run_info["ip_scaled"] = scaled
        if ipa:
            log("IP-Adapter: " + ", ".join(f"{a['role']} image ({a['weight_type']}, {a['weight']:g})" for a in ipa))
    run_info["ipadapter"] = [{k: v for k, v in a.items() if k != "image"} for a in ipa]
    (out_dir / "run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    params = replace(enforce_reference_rules(initial_params(job, cfg.get("defaults", {})), lc), loras=start_loras)
    # 0c. Starting sampler settings from the LLM (optional; the judge tunes them from round 2).
    if lc.get("ai_settings"):
        report({"type": "stage", "round": 0, "phase": "explore", "stage": "choosing sampler settings"})
        try:
            chosen = advisor.suggest_settings(
                judge.backend, checkpoint=ckpt_name, checkpoint_base=ckpt_base, checkpoint_note=ckpt_text,
                samplers=samplers,
                schedulers=schedulers, current={k: getattr(params, k) for k in advisor.FIELDS},
                positive=rendered_positive(params), negative=params.negative, description=job.description,
                mode=params.mode, denoise=params.denoise, size=size,
                lora_notes=advisor.lora_lines(library, params.loras),
                max_side=cfg["judge"].get("image_max_side", 512))
            cost += chosen["cost"]
            params = replace(params, **{k: chosen[k] for k in advisor.FIELDS})
            run_info["ai_settings"] = {k: chosen[k] for k in (*advisor.FIELDS, "notes", "changed")}
            (out_dir / "run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
            log(f"sampler settings from the LLM: {advisor.summary(chosen)}"
                + (f" ({chosen['notes']})" if chosen["notes"] else ""))
        except Cancelled:
            raise
        except Exception as e:  # keep the job's own settings rather than lose the job
            cost += float(getattr(e, "cost_usd", 0.0))
            log(f"warning: choosing sampler settings failed ({str(e)[:200]}); using {advisor.summary(vars(params))}")
    modes = allowed_modes(lc)
    min_dn = lc.get("min_reference_denoise") or 0
    rules = (("The result must be a NEW image of the same character in the art style the GOAL describes, not a "
              "copy of the reference. " if style_from == "description" else
              "The result must be a NEW image of the same character and style, not a copy of the reference. ")
             + (f"img2img_reference is limited to denoise >= {min_dn:g}. " if min_dn else "")
             + ("" if "inpaint_reference" in modes else "Inpainting the reference is not available; "
                "inpaint_best repairs the best result instead. ")
             + ("Reworking or repainting the best image (img2img_best, inpaint_best) is only allowed once "
                + " and ".join(f"{k} scores at least {v}" for k, v in (lc.get("refine_gate") or {}).items())
                + "; until then fix identity, outfit and pose with prompt edits and new images. "
                if lc.get("refine_gate") else "")
             + ("The pose comes from a separate POSE image through a ControlNet: judge composition against "
                "that pose, not the reference's. Starting from the reference image is disabled, since it "
                "would bring the reference's pose back. " if lc.get("pose_from_library") else
                "The pose and framing come from the GOAL, not from any target image: judge composition against "
                "the GOAL's pose. Starting from the reference image is disabled, since it would bring its pose "
                "back. " if lc.get("pose_from_goal") else
                "A pose ControlNet holds the reference's pose in every render. " if control else "")
             + ("Items the reference doesn't have (accessories, emblems, straps, props) count against a "
                "candidate (the 'extras' criterion)." if lc.get("penalize_extras", True)
                else "Extra items the reference doesn't have are acceptable; don't penalize them.")
             + (f"\nCHECKPOINT (prompt edits must follow its tag conventions): {ckpt_text}" if ckpt_text else ""))
    rubric = judge.rubric(lc.get("penalize_extras", True))
    if lc.get("pose_from_library") and "composition" in rubric:
        # In testing the judge kept marking composition down for "not matching the
        # reference's pose" despite the rules; the criterion itself has to say it.
        rubric = {**rubric, "composition": {**rubric["composition"], "anchors": (
            "Pose and framing compared with the POSE skeleton image, NOT the reference: 10 = the skeleton's "
            "pose and framing; 5 = similar framing, different arms or legs; 0 = unrelated.")}}
    if lc.get("pose_from_goal") and "composition" in rubric:
        rubric = {**rubric, "composition": {**rubric["composition"], "anchors": (
            "Pose and framing compared with the pose the GOAL describes, NOT any target image's pose: 10 = "
            "the GOAL's pose and framing; 5 = similar framing, different arms or legs; 0 = unrelated.")}}
    memory = JudgeMemory(judge, lc.get("history_keep_rounds", 2), int(lc.get("prompt_token_budget", 8000)))
    confirm = lc.get("confirm_pass", True)

    phase = "explore"
    scored: dict[Path, list[float]] = {}   # every judged image -> its scores (re-checks add more)
    params_of: dict[Path, GenParams] = {}
    criteria_by: dict[Path, dict] = {}     # every judged image -> its latest per-criterion scores
    rendered: set = set()                  # settings already rendered (never render twice)
    best_criteria: dict = {}
    close_note = ("The best image so far is close in identity and composition. If what still differs are "
                  "small details, fix the most visible ONE with mode inpaint_best and a mask_target naming "
                  "its region (plus prompt edits for it) instead of generating new images.")
    best_image: Path | None = None
    best_score, best_round = -1.0, 0
    use_local = lc.get("local_prefilter", True) and scoring.available()
    status, rnd = "review", 0
    first_score: float | None = None
    focus: str | None = None              # what the last edit changed (see enforce_focus)
    before_edit: GenParams | None = None  # the parameters that edit was applied to
    fixed_batch = job.overrides.get("candidates_per_round")

    def mean(path: Path) -> float:
        return round(sum(scored[path]) / len(scored[path]), 1)

    def update_best(rnd: int) -> None:
        nonlocal best_image, best_score, best_round
        top = max(scored, key=mean)
        if mean(top) > best_score:
            best_round = rnd
        if top != best_image:
            local["best"] = top
            uploaded["best"] = upload(top)
        best_image, best_score = top, mean(top)

    source_used: dict[int, dict] = {}  # round -> {"image": file name, "mask": file name or None}

    def source_image(p: GenParams, rnd: int) -> str:
        """ComfyUI name of the image this round loads; for inpaint modes, a copy with
        the mask_target region made transparent. Falls back to img2img if CLIPSeg
        finds nothing."""
        _, masked, source = MODES[p.mode]
        source_used[rnd] = {"image": Path(local[source]).name, "mask": None}
        if not masked:
            return uploaded[source]
        report({"type": "stage", "round": rnd, "phase": phase, "stage": f"masking '{p.mask_target}'"})
        out = out_dir / f"r{rnd:02d}_mask_{source}.png"
        try:
            masks.masked_image(comfy, local[source], uploaded[source], p.mask_target, out,
                               lc.get("mask_threshold", 0.35), lc.get("mask_grow_px", 12))
        except masks.EmptyMask as e:
            log(f"{e}; using img2img instead")
            # At img2img's own strength: an inpaint denoise (0.75) would redraw most of the image.
            p.mode = p.mode.replace("inpaint", "img2img")
            p.denoise, p.mask_target = min(p.denoise, DEFAULT_DENOISE[p.mode]), None
            return uploaded[source]
        source_used[rnd]["mask"] = out.name
        return upload(out)

    rnd = 0
    while rnd < max_rounds:
        rnd += 1
        if should_stop():
            status = "stopped"
            log("stopped by user")
            rnd -= 1
            break

        # 1. Render a batch locally. Queue all first so ComfyUI never idles between them.
        image_name = source_image(params, rnd)  # may fall back from inpaint to img2img
        n = batch_for_round(rnd, phase, best_score if rnd > 1 else None, first_score, threshold, lc, fixed_batch)
        if lora_menu:
            lora_menu["switch"] = phase == "explore" and rnd < switch_rounds
        if rnd == 1 and lora_pick and (alternatives or workflow_loras) and lcfg.get("sweep", True):
            found = lora_sweep(params, alternatives, set(index), n, max_loras, workflow_loras)
            log(f"round 1 compares LoRA sets on one seed ({n} candidates)")
        elif focus == "lora_weights" and any(m in managed_set for m, _ in params.loras):
            found = weight_variants(params, n, managed_set, {m: weight_range(m) for m, _ in params.loras})
            log(f"round {rnd} tries LoRA strengths around the proposed ones ({n} candidates, one seed)")
        elif focus == "loras" and before_edit is not None:
            options = [(m, usual_weight(m)) for m in lora_choices.values()]
            found = set_variants(params, before_edit, n, managed_set, options, max_loras)
            log(f"round {rnd} compares LoRA sets ({n} candidates, one seed)")
        else:
            found = variants(params, n, phase)
        batch = reseed_duplicates([enforce_reference_rules(p, lc) for p in found], rendered)
        report({"type": "stage", "round": rnd, "phase": phase, "stage": f"rendering {len(batch)} candidates"})
        # One prompt per candidate (batch size 1) so every candidate is reproducible
        # from its seed alone; batch items can't be re-rendered individually.
        graphs = [flows.build(p, image_name, 1, checkpoint, rendered_positive(p), size, control, ipa or None)
                  for p in batch]
        for i, g in enumerate(graphs):
            (out_dir / "graphs" / f"r{rnd:02d}_c{i}.json").write_text(json.dumps(g), encoding="utf-8")
        ids = [comfy.queue(g) for g in graphs]
        hist = comfy.wait(ids)
        paths: list[Path] = []
        for i, (pid, p) in enumerate(zip(ids, batch)):
            data = comfy.fetch_images(hist[pid], flows.output_node)
            if not data:
                raise RuntimeError(f"No image from output node for {p.mode}")
            path = out_dir / f"r{rnd:02d}_c{i}.png"
            path.write_bytes(data[0])
            paths.append(path)
            params_of[path] = p

        # 2. Free local pre-filter: only the top-k go to the judge.
        order = list(range(len(batch)))
        sweep = any(p.loras != batch[0].loras for p in batch)
        if use_local and len(batch) > lc["send_top_k"] and not sweep:  # a LoRA sweep is judged in full
            report({"type": "stage", "round": rnd, "phase": phase, "stage": "local pre-filter"})
            try:
                sims = scoring.similarities(job.reference, paths)
                order = sorted(order, key=lambda i: -sims[i])[: lc["send_top_k"]]
            except Exception as e:  # an optional speed-up must never fail the job
                cost += float(getattr(e, "cost_usd", 0.0))
                use_local = False
                log(f"local pre-filter unavailable ({str(e).strip().splitlines()[0][:150]}); judging every candidate")
        sent = [paths[i] for i in order]

        # 3. One LLM call: score all sent candidates + plan the next edit.
        report({"type": "stage", "round": rnd, "phase": phase, "stage": f"judging {len(sent)} candidates"})
        extra = close_note if gate_passed(best_criteria, lc) else ""
        if cfg["judge"].get("scoring", "per_candidate") == "per_candidate":
            review = judge.review_each(job.reference, goal, [batch[i].short() for i in order], memory.text(), sent,
                                       modes, rules, rubric, extra, lora_menu, pose_judge, targets=tg)
        else:
            review = judge.review(job.reference, goal, params.short(), memory.text(), sent, modes, rules, rubric,
                                  extra, lora_menu, pose_judge, targets=tg)
        cost += review.cost_usd
        round_best = order[review.best_index]
        round_score = review.scores[review.best_index]
        for pos, i in enumerate(order):
            scored[paths[i]] = [review.scores[pos]]
            criteria_by[paths[i]] = criteria_of(review.raw, pos)
        update_best(rnd)
        if first_score is None:
            first_score = round_score
        if library and index:  # local test results, used when picking LoRAs for later jobs
            library.record_results([(tuple((n, w) for n, w in batch[i].loras if n in index),
                                     criteria_of(review.raw, pos), review.scores[pos])
                                    for pos, i in enumerate(order)])

        edit = review.edit
        memory.add(f"r{rnd}: {batch[round_best].short()} -> {round_score:g}. Issue: {review.diagnosis[:200]}. "
                   f"Next: {edit.get('reason', '')[:160]}")
        scores_by_candidate: list[float | None] = [None] * len(batch)
        diffs_by_candidate: list[list[str]] = [[] for _ in batch]
        for pos, i in enumerate(order):
            scores_by_candidate[i] = review.scores[pos]
            diffs_by_candidate[i] = differences_of(review.raw, pos)
        record({"type": "round", "round": rnd, "phase": phase, "judge_model": judge_model, "params": [p.to_dict() for p in batch],
                "rendered_positive": [rendered_positive(p) for p in batch], "source": source_used.get(rnd),
                "sent": order, "scores": scores_by_candidate, "best": round_best,
                "review": review.raw, "cost_usd": review.cost_usd, "prompt_tokens": review.prompt_tokens})
        report({"type": "round", "round": rnd, "phase": phase, "judge_model": judge_model, "images": paths, "scores": scores_by_candidate,
                "params": [p.short() for p in batch], "round_best": round_best, "round_score": round_score,
                "differences": diffs_by_candidate,
                "best_score": best_score, "best_image": best_image, "diagnosis": review.diagnosis,
                "edit": edit, "cost_usd": cost, "prompt_tokens": review.prompt_tokens})
        log(f"round {rnd}: best {round_score:g} (overall {best_score:g}), ${cost:.3f}. {review.diagnosis[:120]}")

        # 4. A pass must hold up to a second, fresh-context look before the job stops.
        # Every passing candidate is checked, best first, until one holds up. The fresh
        # look's score replaces the first one: in testing it was the more careful of the
        # two (it caught a trouser stripe the round judge scored as "no extras").
        passing = sorted(((review.scores[pos], i) for pos, i in enumerate(order)
                          if review.scores[pos] >= threshold), reverse=True)
        if passing and not confirm:
            status = "done"
            break
        first_failed = None
        for first, i in passing:
            candidate = paths[i]
            report({"type": "stage", "round": rnd, "phase": phase, "stage": "confirming pass with a fresh look"})
            check = judge.confirm(job.reference, goal, batch[i].short(), candidate, modes, rules, rubric,
                                  lora_menu, pose_judge, targets=tg)
            cost += check.cost_usd
            recheck = check.scores[0]
            passed = recheck >= confirm_threshold
            # Keep ranking on the first model's scale; rejection removes pass eligibility.
            ranked = (recheck if passed else min(recheck, threshold - 0.1)) if confirm_model == judge_model else (first if passed else min(first, threshold - 0.1))
            scored[candidate] = [ranked]
            criteria_by[candidate] = criteria_of(check.raw, 0)
            update_best(rnd)
            passed = recheck >= confirm_threshold
            record({"type": "confirm", "round": rnd, "image": candidate.name, "first": first,
                    "recheck": recheck, "passed": passed, "judge_model": judge_model,
                    "confirm_model": confirm_model, "confirm_threshold": confirm_threshold, "review": check.raw, "cost_usd": check.cost_usd})
            report({"type": "confirm", "round": rnd, "image": candidate, "first": first, "recheck": recheck,
                    "passed": passed, "judge_model": judge_model, "confirm_model": confirm_model,
                    "confirm_threshold": confirm_threshold, "diagnosis": check.diagnosis, "best_score": best_score,
                    "best_image": best_image, "cost_usd": cost})
            if passed:
                log(f"pass confirmed: {first:g} then {recheck:g}")
                milestones["goal"] = rnd
                status = "done"
                # The job ends on the image that passed both looks, not on a higher
                # first-look score in the same round that was never re-checked.
                best_image, best_score = candidate, ranked
                break
            log(f"pass not confirmed ({first:g} then {recheck:g}): {check.diagnosis[:120]}")
            memory.add(f"r{rnd} re-check of {candidate.name}: {recheck:g} (first {first:g}). "
                       f"Issue: {check.diagnosis[:200]}")
            if first_failed is None:
                first_failed = (i, check)
        if status == "done":
            break
        if first_failed:
            # Continue from the top candidate with the fresh look's fix for it.
            round_best, check = first_failed
            edit = check.edit

        # 5. Stop rules.
        cost = usage.total_or(cost)
        if cost >= max_cost:
            log("cost budget reached")
            break
        if stretch and best_score >= stretch["from"] and "stretch" not in milestones:
            milestones["stretch"] = rnd
            max_rounds = max(max_rounds, rnd + int(stretch.get("extra_rounds", 10)))
            log(f"reached {stretch['from']:g} in round {rnd}: up to {max_rounds - rnd} more rounds to reach {threshold:g}")
        if rnd - best_round >= lc["plateau_rounds"] and "stretch" not in milestones:
            log(f"no improvement for {lc['plateau_rounds']} rounds")
            break

        # 6. Keep the judge's notes short, then apply the edit and continue.
        summary, summary_cost = memory.compact_if_needed(review.prompt_tokens)
        cost += summary_cost
        if summary:
            record({"type": "memory", "round": rnd, "summary": summary, "prompt_tokens": review.prompt_tokens})
            report({"type": "memory", "round": rnd, "summary": summary})
            log("judge notes summarized to keep its prompt short")
        if review.prompt_tokens > 0.85 * judge.context_window:
            log(f"judge prompt used {review.prompt_tokens} of {judge.context_window} tokens; "
                "lower the image size or candidates per round")
        # The next round builds on the best image so far. When this round didn't beat
        # it, the judge's new advice is applied to the best image's settings rather than
        # to a worse image's: img2img_best / inpaint_best already start from the best
        # image, and its settings belong with it.
        base = paths[round_best] if best_image in paths else best_image
        if base != paths[round_best]:
            log(f"round {rnd} didn't beat {best_image.name} ({best_score:g}); the next edit starts from it")
        best_criteria = criteria_by.get(best_image, {})
        edit = gate_fine_tuning(edit, best_criteria, lc, modes, log,
                                stalled=best_round < rnd and params.mode == "txt2img")
        # One axis per edit: the prompt or the LoRAs. LoRA sets may still change only in
        # the first rounds (loop.lora_switch_rounds), while the job is exploring.
        current_loras = {lora_stem(m): w for m, w in params_of[base].loras if m in managed_set}
        edit = enforce_focus(edit, current_loras, lora_choices,
                             bool(lora_choices) and phase == "explore" and rnd + 1 <= switch_rounds)
        focus = edit["focus"]
        before_edit = params_of[base]
        log(f"next edit changes the {focus.replace('_', ' ')}"
            + (": " + ", ".join(f"{l['lora']} {l['strength']:g}" for l in edit["loras"]) if edit.get("loras") is not None
               and focus.startswith("lora") else ""))
        phase = edit.get("phase", phase)
        params = enforce_reference_rules(apply_edit(params_of[base], edit, samplers, schedulers,
                                                    lora_choices, max_loras), lc)
        if fixed_loras and {n for n, _ in params.loras} != {n for n, _ in params_of[base].loras}:
            log("the edit would have dropped a selected LoRA; keeping them all at their strengths")
            params = replace(params, loras=params_of[base].loras)
        if not edit.get("keep_seed") and lc.get("controlled_seed", True):
            # The first candidate keeps the seed of the image the edit builds on, so its
            # score shows what the edit itself did (a controlled comparison); the others
            # still explore new seeds. Duplicates of earlier renders get reseeded.
            params = replace(params, seed=params_of[base].seed)

    hands_info = None
    if best_image and lc.get("hand_refine") and status != "stopped" and posemod.available():
        # Repaint the hands of the final image, then keep whichever of the two a fresh
        # look scores higher (both are scored the same way, so they're comparable).
        report({"type": "stage", "round": rnd, "phase": "repair", "stage": "refining hands"})
        try:
            bp0 = params_of[best_image]
            res = refine_hands(best_image, bp0, comfy=comfy, flows=flows, cfg=cfg, checkpoint=checkpoint,
                               positive=rendered_positive(bp0), out_dir=out_dir, upload=upload, log=log)
            if res["image"]:
                before = judge.confirm(job.reference, goal, bp0.short(), best_image, modes, rules, rubric,
                                       lora_menu, pose_judge, targets=tg)
                after = judge.confirm(job.reference, goal, bp0.short(), res["image"], modes, rules, rubric,
                                      lora_menu, pose_judge, targets=tg)
                cost += before.cost_usd + after.cost_usd
                kept = after.scores[0] >= before.scores[0]
                hands_info = {"image": res["image"].name, "before": best_image.name, "hands": res["hands"],
                              "score_before": before.scores[0], "score_after": after.scores[0], "kept": kept,
                              "steps": res["steps"], "params": res.get("params")}
                record({"type": "hands", **hands_info})
                report({"type": "hands", **hands_info, "image": res["image"], "before": best_image})
                log(f"hands refined: {before.scores[0]:g} before, {after.scores[0]:g} after; "
                    + ("keeping the refined image" if kept else "keeping the original"))
        except Exception as e:  # an optional finishing pass must never cost the job its result
            cost += float(getattr(e, "cost_usd", 0.0))
            log(f"warning: hand refine failed ({str(e)[:200]})")

    fix_info = None
    if best_image and lc.get("auto_fix") and status != "stopped":
        # Look for flaws (hands, anatomy, proportions...) in the final image and repair
        # them. The likeness to the reference must survive: a fix is kept only if a fresh
        # look scores it no more than autofix.keep_margin below the image it started from.
        src = out_dir / hands_info["image"] if hands_info and hands_info["kept"] else best_image
        bp0 = params_of[best_image]
        report({"type": "stage", "round": rnd, "phase": "repair", "stage": "auto-fixing the result"})
        try:
            res = autofix_mod.autofix(
                src, bp0, backend=judge.backend, comfy=comfy, flows=flows, cfg=cfg, checkpoint=checkpoint,
                positive=rendered_positive(bp0), out_dir=out_dir / "autofix", upload=upload, log=log, tag="best",
                stage=lambda t: report({"type": "stage", "round": rnd, "phase": "repair", "stage": "auto-fix: " + t}),
                should_stop=should_stop)
            cost += res["cost_usd"]
            fix_info = {"before": src.name, "image": None, "kept": False, "issues_found": res["issues_found"],
                        "issues_left": res["issues_left"], "rounds": res["rounds"]}
            if res["image"]:
                before = judge.confirm(job.reference, goal, bp0.short(), src, modes, rules, rubric,
                                       lora_menu, pose_judge, targets=tg)
                after = judge.confirm(job.reference, goal, bp0.short(), res["image"], modes, rules, rubric,
                                      lora_menu, pose_judge, targets=tg)
                cost += before.cost_usd + after.cost_usd
                margin = float(autofix_mod.settings(cfg).get("keep_margin", 2))
                kept = after.scores[0] >= before.scores[0] - margin
                fix_info.update(image=str(Path(res["image"]).relative_to(out_dir)), kept=kept,
                                score_before=before.scores[0], score_after=after.scores[0])
                log(f"auto-fix: likeness {before.scores[0]:g} before, {after.scores[0]:g} after; "
                    + ("keeping the fixed image" if kept else "keeping the original"))
            else:
                log("auto-fix: nothing needed fixing" if not res["issues_found"] else "auto-fix: no fix was kept")
            record({"type": "autofix", **fix_info})
        except Cancelled:
            log("auto-fix stopped")
        except Exception as e:  # an optional finishing pass must never cost the job its result
            cost += float(getattr(e, "cost_usd", 0.0))
            log(f"warning: auto-fix failed ({str(e)[:200]})")

    best = None
    if best_image:
        shutil.copy2(best_image, out_dir / "best.png")
        if hands_info and hands_info["kept"]:
            shutil.copy2(best_image, out_dir / "best_before_hands.png")
            shutil.copy2(out_dir / hands_info["image"], out_dir / "best.png")
        if fix_info and fix_info["kept"]:
            shutil.copy2(out_dir / "best.png", out_dir / "best_before_autofix.png")
            shutil.copy2(out_dir / fix_info["image"], out_dir / "best.png")
        bp = params_of[best_image]
        brnd = int(best_image.stem[1:].split("_")[0])  # r07_c2 -> 7 (r107_c2 past round 99)
        best = {"image": best_image.name, "round": brnd, "params": bp.to_dict(),
                "rendered_positive": rendered_positive(bp), "checkpoint": ckpt_name, "size": list(size),
                "loras": [{"name": n, "strength": w, "title": index.get(n, {}).get("title"),
                           "trigger_words": index.get(n, {}).get("trigger_words", [])} for n, w in bp.loras],
                "source": source_used.get(brnd), "graph": f"graphs/{best_image.stem}.json",
                "confirmed": status == "done" and confirm, "hands": hands_info, "autofix": fix_info,
                "pose": run_info.get("pose")}
        if lc.get("upscale") and status != "stopped":
            # A larger copy of the final image (best_upscaled.png); best.png stays as judged.
            report({"type": "stage", "round": rnd, "phase": "repair", "stage": "upscaling the result"})
            try:
                res = upscale_mod.run_and_record(
                    "best.png", bp, run_dir=out_dir, comfy=comfy, flows=flows, cfg=cfg, checkpoint=checkpoint,
                    positive=rendered_positive(bp), upload=upload,
                    stage=lambda t: report({"type": "stage", "round": rnd, "phase": "repair", "stage": "upscale: " + t}),
                    should_stop=should_stop)
                best["upscaled"] = {"image": res["final"].name, "size": res["size"]}
                log(f"upscaled to {res['size'][0]}x{res['size'][1]} ({res['final'].name})")
            except Cancelled:
                log("upscale stopped")
            except Exception as e:  # an optional finishing pass must never cost the job its result
                cost += float(getattr(e, "cost_usd", 0.0))
                log(f"warning: upscale failed ({str(e)[:200]})")
    cost = usage.total_or(cost)
    (out_dir / "summary.json").write_text(json.dumps({
        "job": job.name, "status": status, "best_score": best_score, "milestones": milestones,
        "best_image": best_image.name if best_image else None, "rounds": rnd, "cost_usd": round(cost, 4),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "best": best,
    }, indent=2), encoding="utf-8")
    return Result(status, best_score, best_image, rnd, cost, out_dir)
