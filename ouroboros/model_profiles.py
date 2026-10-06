"""Built-in recommendations, overlaid by exact-model, locally measured profiles."""
from __future__ import annotations

import copy
import json
import os
import re
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "eval" / "model_profiles.json"
_lock = threading.Lock()
OPTION_KEYS = {"temperature", "top_p", "top_k", "num_ctx", "num_predict", "think", "keep_alive",
               "max_tokens", "context_tokens", "reasoning_effort", "image_detail", "max_images", "seed"}


def preset(model: str, backend: str = "ollama") -> dict | None:
    try:
        cached = json.loads(CACHE.read_text(encoding="utf-8")).get("profiles", {}).get(f"{backend}:{model}")
    except (OSError, ValueError):
        cached = None
    if cached:
        return copy.deepcopy(cached)
    builtins = json.loads((Path(__file__).parent / "model_presets.json").read_text(encoding="utf-8"))["presets"]
    return next((p for p in builtins if backend in p.get("backends", [backend])
                 and re.search(p["match"], model or "", re.I)), None)


def save_profile(model: str, backend: str, profile: dict) -> None:
    with _lock:
        try:
            data = json.loads(CACHE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {"version": 1, "profiles": {}}
        data["profiles"][f"{backend}:{model}"] = profile
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(tmp, CACHE)


def role_config(judge: dict, role: str = "judge") -> dict:
    """Overrides share the selected provider and credentials, with their own options."""
    cfg = copy.deepcopy(judge)
    bk = cfg.get("backend", "ollama")
    if role == "judge":
        return cfg
    name = (cfg.get(f"{role}_model") or "").strip()
    if not name:
        return cfg
    p = preset(name, bk) or {}
    options = p.get("settings", {}).get("judge", {}).get(bk, {})
    overrides = {k: v for k, v in cfg.get(f"{role}_options", {}).items() if k in OPTION_KEYS}
    cfg[bk] = {**cfg.get(bk, {}), **options, **overrides, "model": name}
    return cfg


def confirmation_threshold(judge: dict, fallback: float, task: str = "loop") -> float:
    explicit = judge.get("confirm_thresholds", {}).get(task)
    if explicit is not None:
        return float(explicit)
    bk = judge.get("backend", "ollama")
    model = judge.get("confirm_model") or judge.get(bk, {}).get("model", "")
    p = preset(model, bk) or {}
    # Never transfer a loop rubric's calibration to Designer's checklist score.
    value = p.get("thresholds", {}).get(task)
    if value is None and task == "loop" and judge.get("confirm_model"):
        value = p.get("settings", {}).get("loop", {}).get("threshold")
    return float(fallback if value is None else value)
