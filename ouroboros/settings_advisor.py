"""Ask the LLM for starting sampler settings: steps, CFG, sampler and scheduler.

The refinement loop's judge already tunes these from round 2 on (judge.EDIT_SCHEMA); this
picks the values round 1 (or a Home-tab render) starts from, instead of fixed defaults.
It sees the checkpoint and its model family, the mode, the output size, the prompt and
what each active LoRA does, and must choose from the samplers and schedulers this
ComfyUI actually has. Its answer is clamped like any planner edit, so a bad suggestion
can't break the workflow; on any failure the caller keeps its current values.

One text-only call (no images), so it's cheap and fast.
"""

from __future__ import annotations

from .params import clamp, lora_stem

FIELDS = ("steps", "cfg", "sampler_name", "scheduler")

INSTRUCTIONS = """You choose sampler settings for one Stable Diffusion render in ComfyUI.

Pick values that suit the checkpoint's model family and this render:
- steps: enough to converge with the chosen sampler (ancestral and SDE samplers want more
  than deterministic ones; more steps rarely help past ~40).
- cfg: how strongly the prompt is enforced. Pony and Illustrious models usually work best
  around 5-7; SDXL base 5-8; Flux and distilled/turbo/lightning/LCM models need a very
  low cfg (about 1-2) and few steps. Long, heavily weighted prompts and stacked LoRAs
  tend to overbake at high cfg; lower it a little for them.
- sampler_name and scheduler: only from the lists given. Match the scheduler to the
  sampler (e.g. karras or exponential for dpmpp samplers, sgm_uniform or simple for LCM
  and turbo-style models, normal for euler).
- img2img and inpainting run only part of the steps (steps x denoise), so give enough
  total steps that the sampled part is still ~15 or more.

Respect what the checkpoint name says about itself (e.g. "lightning", "turbo", "8step",
"lcm", "hyper", "dmd"). Explain the choice in one or two short sentences in notes.

Reply with JSON only."""


def schema(samplers: list[str], schedulers: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "steps": {"type": "integer"},
            "cfg": {"type": "number"},
            "sampler_name": {"type": "string", "enum": samplers} if samplers else {"type": "string"},
            "scheduler": {"type": "string", "enum": schedulers} if schedulers else {"type": "string"},
            "notes": {"type": "string"},
        },
        "required": [*FIELDS, "notes"],
        "additionalProperties": False,
    }


def suggest_settings(backend, *, checkpoint: str, checkpoint_base: str, samplers: list[str],
                     schedulers: list[str], current: dict, positive: str = "", negative: str = "",
                     description: str = "", mode: str = "txt2img", denoise: float = 1.0,
                     size: tuple[int, int] | None = None, lora_notes: str = "", max_side: int = 512,
                     checkpoint_note: str = "") -> dict:
    """Returns {"steps", "cfg", "sampler_name", "scheduler", "notes", "changed", "cost"}.
    checkpoint_note: ckpt_info.note(), which model it is and whether it is v-prediction.
    `current` holds the values in use now (the fallback for anything the model gets wrong);
    "changed" lists the fields the suggestion actually changes."""
    lines = [f"CHECKPOINT {checkpoint or '(the workflow default)'} (model family: {checkpoint_base})"
             + (f"\n{checkpoint_note.splitlines()[0]}" if checkpoint_note else ""),
             f"MODE {mode}" + (f", denoise {denoise:g}" if mode != "txt2img" else ""),
             f"SIZE {size[0]}x{size[1]}" if size else "",
             "CURRENT SETTINGS " + ", ".join(f"{k}={current.get(k)}" for k in FIELDS),
             f"AVAILABLE SAMPLERS {', '.join(samplers)}" if samplers else "",
             f"AVAILABLE SCHEDULERS {', '.join(schedulers)}" if schedulers else ""]
    if description.strip():
        lines.append(f"DESCRIPTION\n{description.strip()}")
    lines.append(f"POSITIVE PROMPT\n{positive.strip() or '(not written yet)'}")
    if negative.strip():
        lines.append(f"NEGATIVE PROMPT\n{negative.strip()}")
    lines.append(f"ACTIVE LORAS\n{lora_notes.strip() or '(none)'}")
    data, cost, _tokens = backend.complete(INSTRUCTIONS, [{"text": "\n\n".join(x for x in lines if x)}],
                                           schema(samplers, schedulers), "settings", max_side)
    out = {k: current.get(k) for k in FIELDS}
    for name in ("steps", "cfg"):
        try:
            out[name] = clamp(name, float(data.get(name)))
        except (TypeError, ValueError):
            pass
    out["steps"] = int(round(out["steps"])) if out["steps"] is not None else None
    if out["cfg"] is not None:
        out["cfg"] = round(float(out["cfg"]), 1)
    for name, allowed in (("sampler_name", samplers), ("scheduler", schedulers)):
        v = data.get(name)
        if isinstance(v, str) and v and (not allowed or v in allowed):
            out[name] = v
    out["notes"] = str(data.get("notes") or "").strip()
    out["changed"] = [k for k in FIELDS if out[k] != current.get(k)]
    out["cost"] = cost or 0.0
    return out


def lora_lines(library, loras) -> str:
    """One line per active LoRA: its card (what it does, usual weight) and the strength used."""
    index = library.index() if library else {}
    return "\n".join(f"- {library.card(index[n], detail=False) if n in index else lora_stem(n)} "
                     f"(used at strength {w:g})" for n, w in loras)


def summary(s: dict) -> str:
    return f"steps {s['steps']}, cfg {s['cfg']:g}, {s['sampler_name']}/{s['scheduler']}"

