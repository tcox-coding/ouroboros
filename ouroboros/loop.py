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
from .handfix import refine_hands
from .comfy import ComfyClient
from .jobs import Job
from .judge import Judge
from .lora_picker import pick_loras, prompt_notes
from .loras import LoraLibrary, checkpoint_base, compatible, with_triggers
from .params import (MODES, GenParams, allowed_modes, apply_edit, enforce_reference_rules, lora_stem,
                     norm_tag, reseed_duplicates, split_tags, variants)
from .prompter import write_prompt
from .sizes import check_size, fit_to, output_size
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
    baseline the picks have to beat), then sets with one alternative swapped in, all on
    the same seed so the judge compares the LoRAs, not the seeds."""
    out = [p]
    if baseline and tuple(baseline) != tuple(p.loras):
        out.append(replace(p, loras=tuple(baseline)))
    picked = [(name, w) for name, w in p.loras if name in managed]
    pinned = [(name, w) for name, w in p.loras if name not in managed]
    for alt in alternatives:
        if len(out) >= n:
            break
        if picked:
            weakest = min(range(len(picked)), key=lambda i: picked[i][1])
            new = picked[:weakest] + picked[weakest + 1:] + [alt]
        else:
            new = [alt]
        out.append(replace(p, loras=tuple(pinned + new[:max_loras])))
    while len(out) < n:
        out.append(replace(p, seed=random.randrange(2**32)))
    return out


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

    # 0a. Checkpoint and LoRAs. "auto": the LLM picks style LoRAs from the managed folders
    # (Settings -> LoRAs) by comparing the reference with each LoRA's Civitai examples;
    # LoRAs outside those folders stay as the workflow has them. "workflow": the saved
    # workflow's LoRAs, strengths tunable. "off": none.
    lcfg = cfg.get("loras", {})
    checkpoint = job.settings.get("checkpoint") or cfg.get("defaults", {}).get("checkpoint") or None
    ckpt_name = checkpoint or flows.default("checkpoint")
    ckpt_base = checkpoint_base(ckpt_name, cfg.get("checkpoint_bases"))
    lora_mode = lc.get("lora_mode") or lcfg.get("mode", "workflow")
    max_loras = int(lcfg.get("max_loras", 3))
    index = library.index() if library else {}
    workflow_loras = flows.default_loras()
    start_loras: tuple = () if lora_mode == "off" else workflow_loras
    alternatives: list[tuple[str, float]] = []
    lora_pick: dict | None = None
    if lora_mode == "auto" and index:
        report({"type": "stage", "round": 0, "phase": "explore", "stage": "choosing LoRAs"})
        pinned = tuple((n, w) for n, w in workflow_loras if n not in index)
        try:
            lora_pick = pick_loras(judge.backend, library, job.reference,
                                   job.description or job.positive or flows.default("positive"), ckpt_base,
                                   max_loras, {**cfg["judge"], **lcfg})
            start_loras = pinned + tuple((n, w) for n, w, _ in lora_pick["picks"])
            alternatives = lora_pick["alternatives"]
            log("LoRAs picked: " + (", ".join(f"{lora_stem(n)} {w:g}" for n, w, _ in lora_pick["picks"]) or "none")
                + (f"; alternatives to test: {', '.join(lora_stem(n) for n, _ in alternatives)}" if alternatives else ""))
        except Exception as e:  # a failed pick shouldn't cost the job: fall back to the workflow's LoRAs
            log(f"warning: LoRA picking failed ({str(e)[:200]}); using the workflow's LoRAs")
    elif lora_mode == "auto":
        log("LoRA mode is auto but the LoRA index is empty (Settings -> LoRAs -> Refresh); using the workflow's")
    active_managed = [n for n, _ in start_loras if n in index]
    # The judge may switch between the picked LoRAs, the alternatives and the workflow's.
    lora_choices = {lora_stem(n): n for n in
                    dict.fromkeys(active_managed + [n for n, _ in alternatives]
                                  + [n for n, _ in workflow_loras if n in index])} if lora_mode != "off" else {}
    lora_menu = None
    if lora_choices:
        lora_menu = {"menu": "\n".join("- " + library.card(index[n], detail=False) for n in lora_choices.values()),
                     "stems": list(lora_choices), "max": max_loras}
    # 0a'. Pose ControlNet (optional). "reference": every render follows the reference's
    # pose. A pose library name: the character comes from the reference and the pose from
    # that pose image; nothing may start from the reference then (img2img would bring its
    # pose back), the output size follows the pose image's shape, and the judge scores
    # composition against the pose.
    cn = cfg.get("controlnet", {})
    pose_setting = str(job.settings.get("pose") or lc.get("pose") or cfg.get("defaults", {}).get("pose") or "off")
    pose_data, pose_source, pose_desc = None, None, ""
    if pose_setting != "off":
        if not posemod.available():
            log("warning: the pose ControlNet needs rtmlib (pip install rtmlib); continuing without a pose")
        elif pose_setting == "reference":
            pose_data, pose_source = posemod.detect(job.reference), job.reference
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
    run_info = {"job": job.name, "checkpoint": ckpt_name, "checkpoint_base": ckpt_base,
                "workflow": flows.spec.get("file"), "fixed_inputs": flows.fixed_inputs(),
                "judge_backend": cfg["judge"]["backend"],
                "judge_model": cfg["judge"].get(cfg["judge"]["backend"], {}).get("model"),
                "threshold": threshold, "lora_mode": lora_mode, "workflow_loras": workflow_loras,
                "start_loras": start_loras, "lora_pick": lora_pick,
                "pose": {"setting": pose_setting, "active": bool(pose_data), "description": pose_desc},
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
        report({"type": "stage", "round": 0, "phase": "explore",
                "stage": "writing prompt from " + ("the reference image" if from_image else "description")})
        written = write_prompt(judge.backend, job.description, job.positive, job.negative,
                               flows.default("positive"), flows.default("negative"),
                               job.reference if from_image or lc.get("prompt_sees_reference", True) else None,
                               cfg["judge"].get("image_max_side", 512),
                               prompt_notes(library, start_loras) if library else "",
                               pose_desc if lc.get("pose_from_library") else "")
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
    params = replace(enforce_reference_rules(initial_params(job, cfg.get("defaults", {})), lc), loras=start_loras)
    modes = allowed_modes(lc)
    min_dn = lc.get("min_reference_denoise") or 0
    rules = ("The result must be a NEW image of the same character and style, not a copy of the reference. "
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
                "A pose ControlNet holds the reference's pose in every render. " if control else "")
             + ("Items the reference doesn't have (accessories, emblems, straps, props) count against a "
                "candidate (the 'extras' criterion)." if lc.get("penalize_extras", True)
                else "Extra items the reference doesn't have are acceptable; don't penalize them."))
    rubric = judge.rubric(lc.get("penalize_extras", True))
    if lc.get("pose_from_library") and "composition" in rubric:
        # In testing the judge kept marking composition down for "not matching the
        # reference's pose" despite the rules; the criterion itself has to say it.
        rubric = {**rubric, "composition": {**rubric["composition"], "anchors": (
            "Pose and framing compared with the POSE skeleton image, NOT the reference: 10 = the skeleton's "
            "pose and framing; 5 = similar framing, different arms or legs; 0 = unrelated.")}}
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
            p.mode = p.mode.replace("inpaint", "img2img")
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
        if rnd == 1 and lora_pick and (alternatives or workflow_loras) and lcfg.get("sweep", True):
            found = lora_sweep(params, alternatives, set(index), lc["candidates_per_round"], max_loras,
                               workflow_loras)
            log("round 1 compares LoRA sets on one seed")
        else:
            found = variants(params, lc["candidates_per_round"], phase)
        batch = reseed_duplicates([enforce_reference_rules(p, lc) for p in found], rendered)
        report({"type": "stage", "round": rnd, "phase": phase, "stage": f"rendering {len(batch)} candidates"})
        # One prompt per candidate (batch size 1) so every candidate is reproducible
        # from its seed alone; batch items can't be re-rendered individually.
        graphs = [flows.build(p, image_name, 1, checkpoint, rendered_positive(p), size, control) for p in batch]
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
        sweep = rnd == 1 and any(p.loras != batch[0].loras for p in batch)
        if use_local and len(batch) > lc["send_top_k"] and not sweep:  # a LoRA sweep is judged in full
            report({"type": "stage", "round": rnd, "phase": phase, "stage": "local pre-filter"})
            try:
                sims = scoring.similarities(job.reference, paths)
                order = sorted(order, key=lambda i: -sims[i])[: lc["send_top_k"]]
            except Exception as e:  # an optional speed-up must never fail the job
                use_local = False
                log(f"local pre-filter unavailable ({str(e).strip().splitlines()[0][:150]}); judging every candidate")
        sent = [paths[i] for i in order]

        # 3. One LLM call: score all sent candidates + plan the next edit.
        report({"type": "stage", "round": rnd, "phase": phase, "stage": f"judging {len(sent)} candidates"})
        extra = close_note if gate_passed(best_criteria, lc) else ""
        if cfg["judge"].get("scoring", "per_candidate") == "per_candidate":
            review = judge.review_each(job.reference, goal, [batch[i].short() for i in order], memory.text(), sent,
                                       modes, rules, rubric, extra, lora_menu, pose_judge)
        else:
            review = judge.review(job.reference, goal, params.short(), memory.text(), sent, modes, rules, rubric,
                                  extra, lora_menu, pose_judge)
        cost += review.cost_usd
        round_best = order[review.best_index]
        round_score = review.scores[review.best_index]
        for pos, i in enumerate(order):
            scored[paths[i]] = [review.scores[pos]]
            criteria_by[paths[i]] = criteria_of(review.raw, pos)
        update_best(rnd)
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
        record({"type": "round", "round": rnd, "phase": phase, "params": [p.to_dict() for p in batch],
                "rendered_positive": [rendered_positive(p) for p in batch], "source": source_used.get(rnd),
                "sent": order, "scores": scores_by_candidate, "best": round_best,
                "review": review.raw, "cost_usd": review.cost_usd, "prompt_tokens": review.prompt_tokens})
        report({"type": "round", "round": rnd, "phase": phase, "images": paths, "scores": scores_by_candidate,
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
                                  lora_menu, pose_judge)
            cost += check.cost_usd
            recheck = check.scores[0]
            scored[candidate] = [recheck]
            criteria_by[candidate] = criteria_of(check.raw, 0)
            update_best(rnd)
            passed = recheck >= threshold
            record({"type": "confirm", "round": rnd, "image": candidate.name, "first": first,
                    "recheck": recheck, "passed": passed, "review": check.raw, "cost_usd": check.cost_usd})
            report({"type": "confirm", "round": rnd, "image": candidate, "first": first, "recheck": recheck,
                    "passed": passed, "diagnosis": check.diagnosis, "best_score": best_score,
                    "best_image": best_image, "cost_usd": cost})
            if passed:
                log(f"pass confirmed: {first:g} then {recheck:g}")
                milestones["goal"] = rnd
                status = "done"
                # The job ends on the image that passed both looks, not on a higher
                # first-look score in the same round that was never re-checked.
                best_image, best_score = candidate, recheck
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
        phase = edit.get("phase", phase)
        params = enforce_reference_rules(apply_edit(params_of[base], edit, samplers, schedulers,
                                                    lora_choices, max_loras), lc)
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
                                       lora_menu, pose_judge)
                after = judge.confirm(job.reference, goal, bp0.short(), res["image"], modes, rules, rubric,
                                      lora_menu, pose_judge)
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
            log(f"warning: hand refine failed ({str(e)[:200]})")

    best = None
    if best_image:
        shutil.copy2(best_image, out_dir / "best.png")
        if hands_info and hands_info["kept"]:
            shutil.copy2(best_image, out_dir / "best_before_hands.png")
            shutil.copy2(out_dir / hands_info["image"], out_dir / "best.png")
        bp = params_of[best_image]
        brnd = int(best_image.stem[1:3])
        best = {"image": best_image.name, "round": brnd, "params": bp.to_dict(),
                "rendered_positive": rendered_positive(bp), "checkpoint": ckpt_name, "size": list(size),
                "loras": [{"name": n, "strength": w, "title": index.get(n, {}).get("title"),
                           "trigger_words": index.get(n, {}).get("trigger_words", [])} for n, w in bp.loras],
                "source": source_used.get(brnd), "graph": f"graphs/{best_image.stem}.json",
                "confirmed": status == "done" and confirm, "hands": hands_info, "pose": run_info.get("pose")}
    (out_dir / "summary.json").write_text(json.dumps({
        "job": job.name, "status": status, "best_score": best_score, "milestones": milestones,
        "best_image": best_image.name if best_image else None, "rounds": rnd, "cost_usd": round(cost, 4),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "best": best,
    }, indent=2), encoding="utf-8")
    return Result(status, best_score, best_image, rnd, cost, out_dir)
