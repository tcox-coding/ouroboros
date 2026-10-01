"""Short descriptions ("briefs") of each LoRA, for the LLM that chooses LoRAs.

The LoRA classifier writes a long description of every LoRA it tested
(<catalog_dir>/catalog/descriptions/<category>/<name>.json): what it does, its style
metrics, which subjects and framings it suits, its trigger words, the weight range
that works and how the effect builds up, its strengths and weaknesses. That's far too
much to show for hundreds of LoRAs at once, and a bare title ("Jab Style PNY V1.5")
says nothing. So the configured judge model condenses each description into a brief:
a few fixed fields, rendered as one line of ~60 words, e.g.

  [style] semi-realistic painterly look, sculpted athletic-curvy women, glossy skin |
  look: muted warm palette, strong highlights, soft cinematic light | best for: solo
  adult women, any framing | weak: crowded scenes | trigger: Jabstyle |
  weight 0.5-1.2, usually 0.9 (visible from 0.6, overbaked past 1.2) | suggestive

Every place that offers LoRAs to a model (Home's "choose LoRAs", the loop's shortlist,
the judge's LoRA menu) shows the brief. Briefs are cached in cache/lora_briefs.json
and only rewritten when a description changes:

    python -m ouroboros briefs            # write the missing ones
    python -m ouroboros briefs --force    # rewrite all

Only LoRAs the classifier finished ("done") are described. A brief also says whether
the LoRA is made to depict minors; one that is gets no brief and is never offered.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

KINDS = ["style", "character", "concept", "pose", "clothing", "body", "effect", "quality", "slider",
         "background", "other"]
RATINGS = ["sfw", "suggestive", "explicit"]

INSTRUCTIONS = """You condense the description of a Stable Diffusion XL LoRA into a short brief for
another model that will choose LoRAs for a render. It has to know, at a glance, what the
LoRA does to an image, when it helps, when it hurts, and how to use it.

Fill every field with the most decision-relevant facts from the description, in plain
neutral words, as terse as possible (fragments, no full sentences, no filler):
- kind: what sort of LoRA it is.
- effect: what it changes in the image (<= 20 words) - the look, subject, pose or concept
  it imposes.
- look: its visual signature (<= 15 words): line, shading, palette, lighting, anatomy.
  Empty if it isn't a style.
- best_for: subjects, framings and uses it works well with (<= 12 words).
- weak: what it does badly, fights, or overrides (<= 12 words). Empty if nothing notable.
- triggers: its trigger words exactly as written (at most 4).
- weight: min, usual and max strength that work; weight_note: how the effect builds up
  with strength (<= 10 words, e.g. "visible from 0.6, overbaked past 1.2").
- rating: how explicit its typical output is.
- minors: true only if the LoRA is made to depict children or childlike characters
  (by its subject, training or purpose), false otherwise.

Reply with JSON only."""

SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": KINDS},
        "effect": {"type": "string"},
        "look": {"type": "string"},
        "best_for": {"type": "string"},
        "weak": {"type": "string"},
        "triggers": {"type": "array", "maxItems": 4, "items": {"type": "string"}},
        "weight_min": {"type": "number"},
        "weight_usual": {"type": "number"},
        "weight_max": {"type": "number"},
        "weight_note": {"type": "string"},
        "rating": {"type": "string", "enum": RATINGS},
        "minors": {"type": "boolean"},
    },
    "required": ["kind", "effect", "look", "best_for", "weak", "triggers", "weight_min", "weight_usual",
                 "weight_max", "weight_note", "rating", "minors"],
    "additionalProperties": False,
}

# Fields of a classifier description worth condensing; the rest (example images, probe
# renders, file paths, render settings, cost) is bookkeeping.
KEEP = ["name", "lora_type", "category", "summary", "description", "trigger_words", "recommended_weight",
        "nsfw_level", "metrics", "extra_metrics", "scores", "strengths", "weaknesses", "usage_tips"]


def cache_file(cache_dir: Path) -> Path:
    return cache_dir / "lora_briefs.json"


_loaded: dict = {"mtime": None, "data": {}}
_lock = threading.Lock()


def load(cache_dir: Path) -> dict:
    """{lora id: {"text", "fields", "minors", ...}}, re-read only when the file changes."""
    f = cache_file(cache_dir)
    try:
        mtime = f.stat().st_mtime
    except OSError:
        return {}
    with _lock:
        if _loaded["mtime"] != mtime:
            try:
                _loaded["data"] = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                _loaded["data"] = {}
            _loaded["mtime"] = mtime
        return _loaded["data"]


def render(b: dict) -> str:
    """The one-line brief shown to the model that picks LoRAs."""
    w = f"weight {b['weight_min']:g}-{b['weight_max']:g}, usually {b['weight_usual']:g}"
    if b.get("weight_note"):
        w += f" ({b['weight_note']})"
    bits = [f"[{b['kind']}] {b['effect']}"]
    for label, key in (("look", "look"), ("best for", "best_for"), ("weak", "weak")):
        if (b.get(key) or "").strip():
            bits.append(f"{label}: {b[key].strip()}")
    if b.get("triggers"):
        bits.append("trigger: " + ", ".join(b["triggers"]))
    bits += [w, b["rating"]]
    return " | ".join(bits)


def descriptions(catalog_root: Path) -> dict[str, Path]:
    """Classifier description files by LoRA id (only finished LoRAs)."""
    base = catalog_root / "catalog" / "descriptions"
    out = {}
    for f in base.rglob("*.json") if base.is_dir() else []:
        if "caution loras" in f.relative_to(base).parts:
            continue  # flagged by the classifier's safety checks; never described or offered
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if d.get("id") and d.get("status") == "done":
            out[d["id"]] = f
    return out


def source_text(desc: dict) -> str:
    """The parts of a classifier description worth condensing, as compact JSON."""
    keep = {k: desc[k] for k in KEEP if desc.get(k) not in (None, "", [], {})}
    if (desc.get("civitai") or {}).get("model"):
        keep["civitai_title"] = desc["civitai"]["model"]
    return json.dumps(keep, ensure_ascii=False, separators=(",", ":"))


def write_brief(backend, text: str) -> tuple[dict, float]:
    data, cost, _ = backend.complete(INSTRUCTIONS, [{"text": "LORA DESCRIPTION\n" + text}], SCHEMA,
                                     "lora_brief", 512)
    lo, hi = sorted((float(data["weight_min"]), float(data["weight_max"])))
    data.update(weight_min=round(lo, 2), weight_max=round(hi, 2),
                weight_usual=round(min(hi, max(lo, float(data["weight_usual"]))), 2),
                triggers=[t.strip() for t in data.get("triggers") or [] if t.strip()][:4])
    for k in ("effect", "look", "best_for", "weak", "weight_note"):
        data[k] = re.sub(r"\s+", " ", (data.get(k) or "")).strip().rstrip(".")
    return data, cost


def build(cfg: dict, catalog_root: Path, cache_dir: Path, *, force: bool = False, only: str = "",
          limit: int = 0, workers: int = 4, log=print) -> dict:
    """Write the briefs that are missing or whose description changed. Returns counts."""
    from .backends import make_backend
    backend = make_backend(cfg["judge"])
    model = cfg["judge"].get(cfg["judge"]["backend"], {}).get("model", "")
    files = descriptions(catalog_root)
    current = dict(load(cache_dir))
    todo = []
    for lora_id, f in sorted(files.items()):
        desc = json.loads(f.read_text(encoding="utf-8"))
        if only and only.lower() not in (desc.get("category", "") + " " + desc.get("name", "")).lower():
            continue
        text = source_text(desc)
        digest = hashlib.sha1(text.encode()).hexdigest()
        if not force and current.get(lora_id, {}).get("source_hash") == digest:
            continue
        todo.append((lora_id, desc.get("name", lora_id), text, digest))
    if limit:
        todo = todo[:limit]
    log(f"{len(files)} described LoRAs; {len(todo)} brief(s) to write with {model or cfg['judge']['backend']}")
    done = failed = 0
    cost = 0.0
    save_lock = threading.Lock()

    def one(item):
        lora_id, name, text, digest = item
        fields, c = write_brief(backend, text)
        return lora_id, name, digest, fields, c

    with ThreadPoolExecutor(max(1, workers)) as pool:
        futures = [pool.submit(one, item) for item in todo]
        for fut in as_completed(futures):
            try:
                lora_id, name, digest, fields, c = fut.result()
            except Exception as e:
                failed += 1
                log(f"  failed: {str(e)[:160]}")
                continue
            cost += c or 0.0
            entry = {"name": name, "fields": fields, "minors": bool(fields.get("minors")),
                     "text": "" if fields.get("minors") else render(fields), "source": "classifier",
                     "source_hash": digest, "model": model, "written": time.strftime("%Y-%m-%d %H:%M:%S")}
            with save_lock:
                current[lora_id] = entry
                done += 1
                if done % 10 == 0 or done == len(todo):
                    _save(cache_dir, current)
            log(f"  [{done}/{len(todo)}] {name}: " + ("not offered (made to depict minors)" if entry["minors"]
                                                     else entry["text"][:140]))
    _save(cache_dir, current)
    return {"written": done, "failed": failed, "total": len(files), "cost_usd": round(cost, 4)}


def _save(cache_dir: Path, data: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_file(cache_dir)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(f)


def main(argv=None) -> None:
    import argparse
    from .runner import ROOT, load_config
    ap = argparse.ArgumentParser(prog="ouroboros briefs",
                                 description="Write short LoRA descriptions for the LLM that chooses LoRAs.")
    ap.add_argument("--force", action="store_true", help="rewrite every brief, not just missing or changed ones")
    ap.add_argument("--only", default="", help="only LoRAs whose category or name contains this text")
    ap.add_argument("--limit", type=int, default=0, help="write at most this many")
    ap.add_argument("--workers", type=int, default=0, help="LLM calls at once (default: queue.llm_parallel)")
    args = ap.parse_args(argv)
    cfg = load_config()
    root = Path(cfg.get("loras", {}).get("catalog_dir", "")).expanduser()
    if not (root / "catalog" / "descriptions").is_dir():
        raise SystemExit(f"No classifier descriptions under {root} (Settings -> loras.catalog_dir).")
    res = build(cfg, root, ROOT / "cache", force=args.force, only=args.only, limit=args.limit,
                workers=args.workers or int(cfg.get("queue", {}).get("llm_parallel", 4)))
    print(f"done: {res['written']} written, {res['failed']} failed, ${res['cost_usd']}")
