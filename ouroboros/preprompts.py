"""The prompt writer's guidelines for each model (Settings > Prompt writer guidelines).

The prompt writer's system text is these guidelines followed by prompter.CONTRACT (the JSON
reply the app reads, and how LoRAs are handled), which stays fixed whatever is written here.
A model's guidelines are, in order: what Settings saved for it
(config prompt_writer.preprompts), the file shipped for it in preprompts/<model>.md, or the
general guidelines (prompter.GUIDE) that every model used before.

When a model has guidelines of its own, they carry its tag conventions, so the short
conventions line from ckpt_info is left out of the CHECKPOINT section (they would disagree,
e.g. "very awa" against "very aesthetic").
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).parent / "preprompts"

# Settings' model selector: (key, label). The keys are ckpt_info's model names; "SDXL" is
# every checkpoint it doesn't recognise.
MODELS = (
    ("NoobAI-XL", "NoobAI-XL"),
    ("Pony Diffusion V6 XL", "Pony Diffusion V6 XL (and Pony merges such as AutismMix)"),
    ("Illustrious-XL", "Illustrious-XL"),
    ("Animagine XL", "Animagine XL"),
    ("SDXL", "Other SDXL models"),
    ("Flux", "Flux"),
)


def model_key(model: str) -> str:
    """ckpt_info's model name ("NoobAI-XL (from config)", "an SDXL model") as a MODELS key."""
    model = model.replace(" (from config)", "")
    return model if any(model == k for k, _ in MODELS) else "SDXL"


def default(key: str) -> str:
    from .prompter import GUIDE
    f = HERE / f"{key}.md"
    try:
        return f.read_text(encoding="utf-8").strip() if f.is_file() else GUIDE
    except OSError:
        return GUIDE


def saved(cfg: dict) -> dict:
    return {k: v for k, v in ((cfg.get("prompt_writer") or {}).get("preprompts") or {}).items()
            if isinstance(v, str) and v.strip()}


def get(cfg: dict, key: str) -> str:
    return saved(cfg).get(key) or default(key)


def own(cfg: dict, key: str) -> bool:
    """Whether the model has guidelines of its own rather than the general ones."""
    from .prompter import GUIDE
    return get(cfg, key).strip() != GUIDE.strip()


def listing(cfg: dict) -> list[dict]:
    keep = saved(cfg)
    return [{"key": k, "label": label, "text": get(cfg, k), "default": default(k), "edited": k in keep}
            for k, label in MODELS]


def for_checkpoint(cfg: dict, name: str | None) -> tuple[str, bool]:
    """(guidelines, own) for the checkpoint that will render the prompt."""
    try:
        from .ckpt_info import lookup
        key = model_key(lookup(name, cfg)["model"]) if name else "SDXL"
    except Exception:
        key = "SDXL"
    return get(cfg, key), own(cfg, key)
