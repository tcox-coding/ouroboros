"""What a checkpoint is: its model (Pony Diffusion V6, Illustrious-XL, NoobAI-XL, ...), its
family (for LoRA compatibility), whether it is v-prediction, and the tag conventions the
prompt writer and the judge should follow for it.

Read from the file name and the safetensors header: the `__metadata__` block
(modelspec.title, modelspec.merged_from, modelspec.prediction_type, kohya's ss_* keys) and
the marker tensors a v-prediction checkpoint carries (`v_pred`, `ztsnr`, which ComfyUI
reads too). Many checkpoints carry no metadata at all (WAI-Illustrious, most Civitai
merges), so the name is the first clue; metadata fills in what the name doesn't say, such
as a merge listing Pony Diffusion in merged_from.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

# In the conventions of every model that isn't Pony-based; the prompt writer looks for it to
# strip the workflow's Pony score_/source_ tags the LLM copies anyway (prompter.write_prompt).
NO_PONY_TAGS = "Pony score_ and source_ tags mean nothing to it"

PONY = ("Pony Diffusion V6 XL conventions: the positive starts with the score tags "
        "\"score_9, score_8_up, score_7_up\", then a source tag (source_anime, source_cartoon, source_furry or "
        "source_pony) and optionally a rating tag (rating_safe, rating_questionable, rating_explicit); "
        "Danbooru and e621 tags. Low score tags (score_4, score_5, score_6) and unwanted sources go in the negative.")
ILLUSTRIOUS = ("Illustrious-XL conventions: Danbooru tags; quality tags \"masterpiece, best quality, amazing "
               "quality, absurdres\" in the positive and \"worst quality, low quality, bad quality, lowres\" in the "
               "negative. Pony score_ and source_ tags mean nothing to it: don't use them.")
NOOBAI = ("NoobAI-XL conventions: Danbooru-style tags; quality tags \"masterpiece, best quality, very awa, "
          "newest\" (optionally highres, absurdres) in the positive and \"worst quality, low quality, lowres, bad "
          "anatomy, bad hands\" in the negative. Pony score_ and source_ tags mean nothing to it: don't use them.")
ANIMAGINE = ("Animagine XL conventions: Danbooru tags; quality tags \"masterpiece, best quality, very aesthetic, "
             "absurdres\" in the positive and \"lowres, worst quality, low quality, jpeg artifacts\" in the "
             "negative. Pony score_ and source_ tags mean nothing to it: don't use them.")
SDXL = ("A general SDXL model: tags and short phrases, with quality words such as \"masterpiece, best quality, "
        "highly detailed\". Pony score_ tags only help Pony-based models; use them only if the example prompts do.")
FLUX = "A Flux model: it reads natural-language sentences better than tag lists."

# (words in the name or metadata, model, family, conventions); the first match wins, so
# NoobAI (trained on Illustrious) comes before Illustrious, and merges are named by their base.
MODELS = (
    (("noobai", "noob"), "NoobAI-XL", "Illustrious", NOOBAI),
    (("illustrious", "illust"), "Illustrious-XL", "Illustrious", ILLUSTRIOUS),
    (("pony", "autismmix"), "Pony Diffusion V6 XL", "Pony", PONY),
    (("animagine",), "Animagine XL", "SDXL", ANIMAGINE),
    (("flux",), "Flux", "Flux", FLUX),
)
META_KEYS = ("modelspec.title", "modelspec.merged_from", "modelspec.description", "modelspec.tags",
             "ss_base_model_version", "ss_sd_model_name", "ss_output_name", "name", "description")

_cache: dict[tuple, dict] = {}


def read_header(path: Path) -> tuple[dict, set[str]]:
    """(metadata, names of the non-weight marker tensors) from a .safetensors file."""
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        if size > 100_000_000:
            raise ValueError("not a safetensors header")
        header = json.loads(f.read(size))
    meta = header.pop("__metadata__", None) or {}
    markers = {k for k in header if k in ("v_pred", "ztsnr")}
    return {k: str(v) for k, v in meta.items()}, markers


def _match(text: str):
    t = text.lower()
    return next((m for m in MODELS if any(w in t for w in m[0])), None)


def folder_model(folder: str) -> tuple[str, str] | None:
    """(model, family) a LoRA folder is for, from its name ("NoobAI-XL", "Pony"), or None."""
    hit = _match(folder)
    return (hit[1], hit[2]) if hit else None


def describe(name: str, path: Path | None = None, overrides: dict | None = None) -> dict:
    """{"name", "model", "family", "prediction", "conventions", "source", "title", "merged_from"}.
    overrides: config checkpoint_bases ({name part: family}), which wins for the family."""
    meta, markers = {}, set()
    if path is not None and path.suffix.lower() == ".safetensors":
        try:
            st = path.stat()
            key = (str(path), st.st_mtime, st.st_size)
            if key not in _cache:
                _cache[key] = read_header(path)
            meta, markers = _cache[key]
        except (OSError, ValueError, struct.error):
            pass
    stem = Path(name).stem
    hit, source = _match(stem), "name"
    if hit is None:
        hit = _match(" ".join(meta.get(k, "") for k in META_KEYS))
        source = "metadata" if hit else "unknown"
    model, family, conventions = (hit[1], hit[2], hit[3]) if hit else ("an SDXL model", "SDXL", SDXL)
    for pattern, base in (overrides or {}).items():
        if pattern.lower() in name.lower():
            family = base
            if not hit or hit[2] != base:  # the override names a family the name didn't show
                fam = next((m for m in MODELS if m[2] == base), None)
                model, conventions = (fam[1] + " (from config)", fam[3]) if fam else (model, conventions)
            source = "config"
            break
    pred_meta = meta.get("modelspec.prediction_type", "").lower()
    if "v_pred" in markers or pred_meta.startswith("v") or re.search(r"v[-_ ]?pred", stem, re.I):
        prediction = "v-prediction" + (" with zero terminal SNR" if "ztsnr" in markers else "")
    elif pred_meta == "epsilon":
        prediction = "epsilon"
    else:
        prediction = None
    return {"name": stem, "model": model, "family": family, "prediction": prediction,
            "conventions": conventions, "source": source,
            "title": meta.get("modelspec.title", ""), "merged_from": meta.get("modelspec.merged_from", "")}


def note(info: dict, conventions: bool = True) -> str:
    """The CHECKPOINT text for an LLM: which model renders the prompt and how to tag for it.
    conventions False: the model has prompt-writing guidelines of its own (preprompts.py),
    so only the NO_PONY_TAGS marker stays (prompter still strips Pony tags by it)."""
    how = {"name": "from its file name", "metadata": "from its file's metadata", "config": "set in config",
           "unknown": "not recognised from its name or metadata"}[info["source"]]
    bits = [f"{info['name']}: {info['model']} ({how})"]
    if info["family"] not in info["model"]:
        bits.append(f"{info['family']} family")
    if info["prediction"] and info["prediction"] != "epsilon":
        bits.append(info["prediction"])
    if info.get("merged_from"):
        bits.append("merged from " + info["merged_from"][:200])
    if not conventions:
        return f"{'; '.join(bits)}." + (f"\n{NO_PONY_TAGS}." if NO_PONY_TAGS in info["conventions"] else "")
    return f"{'; '.join(bits)}.\n{info['conventions']}"


def checkpoint_dirs(cfg: dict) -> list[Path]:
    """Where checkpoints live: Settings' checkpoints folder, then Comfy Desktop's own."""
    dirs = [Path(d) for d in [cfg.get("checkpoints_dir") or ""] if d.strip()]
    try:
        from .comfy_launcher import detect_desktop_install
        args = detect_desktop_install().get("args", [])
        if "--extra-model-paths-config" in args:
            text = Path(args[args.index("--extra-model-paths-config") + 1]).read_text(encoding="utf-8")
            # The file Comfy Desktop writes: base_path, then 'checkpoints': 'checkpoints/' per block.
            for block in re.split(r"\n(?=\S)", text):
                base = re.search(r"base_path:\s*'?([^'\n]+)'?", block)
                sub = re.search(r"'?checkpoints'?:\s*'?([^'\n|]+)'?", block)
                if base and sub:
                    dirs.append(Path(base.group(1).strip()) / sub.group(1).strip())
    except Exception:
        pass
    return list(dict.fromkeys(dirs))


def lookup(name: str, cfg: dict) -> dict:
    """describe() for a checkpoint as ComfyUI names it, finding its file in the checkpoint folders."""
    path = next((d / name for d in checkpoint_dirs(cfg) if (d / name).is_file()), None)
    return describe(name, path, cfg.get("checkpoint_bases"))


def checkpoint_note(cfg: dict, name: str | None, conventions: bool = True) -> str:
    """note() for a checkpoint name, or "" when there is none or it can't be read."""
    try:
        return note(lookup(name, cfg), conventions) if name else ""
    except Exception:
        return ""


def prompt_setup(cfg: dict, name: str | None) -> dict:
    """write_prompt's checkpoint_note and guidelines for the checkpoint that renders the prompt."""
    from .preprompts import for_checkpoint
    guide, own = for_checkpoint(cfg, name)
    return {"checkpoint_note": checkpoint_note(cfg, name, conventions=not own), "guidelines": guide}
