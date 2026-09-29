"""API keys for the hosted services, managed from Settings -> API keys.

Keys are kept in api_keys.json next to config.json (readable only by you, never
committed). An environment variable wins when set, so a key can also come from the
shell that starts the server. Older setups are still read: deepinfra_key.txt and a
Civitai key saved in config.json (loras.civitai_api_key).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "api_keys.json"

PLATFORMS = {
    "deepinfra": {"label": "DeepInfra", "env": "DEEPINFRA_API_KEY", "used_for": "the judge and prompt writer (DeepInfra backend)",
                  "url": "https://deepinfra.com/dash/api_keys"},
    "openai": {"label": "OpenAI", "env": "OPENAI_API_KEY", "used_for": "the judge and prompt writer (OpenAI backend)",
               "url": "https://platform.openai.com/api-keys"},
    "civitai": {"label": "Civitai", "env": "CIVITAI_API_KEY", "used_for": "LoRA lookups and images (optional; needed for restricted models)",
                "url": "https://civitai.com/user/account"},
}


def _saved() -> dict:
    try:
        return json.loads(FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _legacy(name: str) -> tuple[str, str] | None:
    if name == "deepinfra":
        f = ROOT / "deepinfra_key.txt"
        key = f.read_text(encoding="utf-8").strip() if f.exists() else ""
        if key:
            return key, "deepinfra_key.txt"
    if name in ("deepinfra", "civitai"):
        try:
            cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cfg = {}
        key = (cfg.get("judge", {}).get("deepinfra", {}).get("api_key") if name == "deepinfra"
               else cfg.get("loras", {}).get("civitai_api_key")) or ""
        if key.strip():
            return key.strip(), "config.json"
    return None


def lookup(name: str) -> tuple[str, str | None]:
    """(key, where it came from): environment, settings, or a legacy file. ("", None) if unset."""
    env = os.environ.get(PLATFORMS[name]["env"], "").strip()
    if env:
        return env, "environment"
    saved = (_saved().get(name) or "").strip()
    if saved:
        return saved, "settings"
    return _legacy(name) or ("", None)


def get(name: str) -> str:
    return lookup(name)[0]


def save(name: str, value: str) -> None:
    """Store a key from Settings; an empty value removes the saved one."""
    if name not in PLATFORMS:
        raise KeyError(name)
    keys = _saved()
    value = (value or "").strip()
    if value:
        keys[name] = value
    else:
        keys.pop(name, None)
        _clear_legacy(name)  # clearing means clearing: an older copy must not take over
    tmp = FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(keys, indent=1), encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, FILE)


def _clear_legacy(name: str) -> None:
    if name not in ("deepinfra", "civitai"):
        return  # OpenAI never had a legacy location
    if name == "deepinfra":
        (ROOT / "deepinfra_key.txt").unlink(missing_ok=True)
    cfg_file = ROOT / "config.json"
    try:
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    section = cfg.get("judge", {}).get("deepinfra", {}) if name == "deepinfra" else cfg.get("loras", {})
    field = "api_key" if name == "deepinfra" else "civitai_api_key"
    if section.get(field):
        section[field] = ""
        cfg_file.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def status() -> dict:
    """What Settings shows: never the key itself, only whether and where it is set."""
    out = {}
    for name, p in PLATFORMS.items():
        key, source = lookup(name)
        out[name] = {"label": p["label"], "env": p["env"], "used_for": p["used_for"], "url": p["url"],
                     "set": bool(key), "source": source, "hint": ("…" + key[-4:]) if len(key) >= 8 else ""}
    return out
