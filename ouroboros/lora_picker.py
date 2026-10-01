"""Choose style LoRAs for a job: two vision-LLM calls: a text shortlist, then the reference next
to each shortlisted LoRA's Civitai example image, its trigger words, usual weight, and
how it has scored here. Returns the set to start with plus alternatives, which the
first round renders side by side (same seed) so the judge compares them directly.
"""

from __future__ import annotations

from pathlib import Path

from .judge import contact_sheet
from .loras import LoraLibrary, compatible
from .params import LORA_STRENGTH, lora_stem

INSTRUCTIONS = """You choose style LoRAs for a Stable Diffusion XL (Pony-family) model so that new
images match the REFERENCE image's art style: line work, shading, colour palette,
proportions and overall look. You get the reference image and, for each available
LoRA, one example image made with it (from Civitai), its trigger words, its usual
weight, and how it scored in earlier local tests (scores are 0-100 likeness to other
references; style is 0-10).

- Pick the LoRAs that together best reproduce the reference's style. Fewer is fine,
  and none is the right answer when no LoRA fits: a render without LoRAs is always
  tested alongside your picks. Don't stack LoRAs that pull toward different looks.
- Strength: near the usual weight for one LoRA; lower (0.3-0.6 each) when combining.
- Prefer LoRAs whose local results are good, but trust what you see over a few tests.
- Also list a few alternatives worth testing against your picks.
- Judge by style only; the subject of an example image doesn't matter.

Reply with JSON only."""


SHORTLIST_INSTRUCTIONS = """You shortlist style LoRAs for a Stable Diffusion XL (Pony-family) model. From the
REFERENCE image and each LoRA's description, tags, trigger words, the prompt of one of
its example images and its local test results, choose the LoRAs most likely to
reproduce the reference's art style (line work, shading, colour palette, proportions).
Only their example images will be compared next. Reply with JSON only."""


def pick_loras(backend, library: LoraLibrary, reference: Path, goal: str, ckpt_base: str,
               max_loras: int, cfg: dict) -> dict:
    """Returns {"picks": [(name, strength, why)], "alternatives": [(name, strength)],
    "notes": str, "considered": [names], "shortlist": [names]}.

    Two calls, to stay inside the judge's prompt budget: a vision model's image costs
    about as much as ~1,100 tokens of text in Ollama's qwen3-vl however small it is,
    so 15 example images (~19K tokens) don't fit. First a text shortlist (reference +
    LoRA cards), then the visual comparison with only the shortlisted examples."""
    index = library.index()
    cands = [r for r in index.values() if compatible(r.get("base_model"), ckpt_base) is not False]
    cands.sort(key=lambda r: (compatible(r.get("base_model"), ckpt_base) is not True, not r.get("examples")))
    # With the classified catalog there are hundreds of candidates, and the shortlist
    # stage below is what narrows them: one compact line each is cheap enough to show all
    # of them (~30 tokens a line), where the detailed cards were not. loras.max_candidates
    # only caps the detailed path - truncating the catalog to its first 14 entries would
    # hide almost everything before the model ever saw it.
    compact = len(cands) > int(cfg.get("max_detailed_candidates", 20))
    if not compact:
        cands = cands[: int(cfg.get("max_candidates", 14))]
    if not cands:
        return {"picks": [], "alternatives": [], "notes": "no compatible LoRAs indexed", "considered": []}
    considered = [r["name"] for r in cands]
    side = cfg.get("image_max_side", 384)
    visual_n = int(cfg.get("pick_visual_candidates", 5))
    shortlist_note = ""
    if len(cands) > visual_n:
        stems_all = {lora_stem(r["name"]): r for r in cands}
        cards = []
        for r in cands:
            if compact:
                cards.append("- " + _short_card(r))
                continue
            ex = next((e["prompt"] for e in r["examples"] if e.get("prompt")), "")
            cards.append("- " + library.card(r) + (f"\n  example prompt: {ex[:180]}" if ex else ""))
        parts = [{"text": f"GOAL (what the images show)\n{goal[:400]}\n\nLORAS\n" + "\n".join(cards)},
                 {"text": "REFERENCE IMAGE (the style to match):"}, {"image": reference},
                 {"text": f"Shortlist the {visual_n} LoRAs most likely to match the REFERENCE's style."}]
        schema = {"type": "object",
                  "properties": {"shortlist": {"type": "array", "minItems": min(visual_n, len(cands)),
                                               "maxItems": visual_n,
                                               "items": {"type": "string", "enum": list(stems_all)}},
                                 "why": {"type": "string"}},
                  "required": ["shortlist", "why"], "additionalProperties": False}
        data, _c, _t = backend.complete(SHORTLIST_INSTRUCTIONS, parts, schema, "lora_shortlist", side)
        keep = list(dict.fromkeys(s for s in data.get("shortlist", []) if s in stems_all))[:visual_n]
        if keep:
            cands = [stems_all[s] for s in keep]
            shortlist_note = data.get("why", "")
    stems = {lora_stem(r["name"]): r["name"] for r in cands}

    parts: list[dict] = [{"text": f"GOAL (what the images show)\n{goal[:600]}\n\nMax LoRAs to use together: {max_loras}."}]
    thumbs = [(r, next((e["thumb"] for e in r["examples"] if e.get("thumb") and Path(e["thumb"]).exists()), None))
              for r in cands]
    if cfg.get("contact_sheet"):
        tiles = [("REFERENCE", reference)] + [(f"LORA {lora_stem(r['name'])}", Path(t)) for r, t in thumbs if t]
        parts += [{"text": "LORAS\n" + "\n".join("- " + library.card(r) for r, _ in thumbs)},
                  {"text": "The image is a grid: the REFERENCE, then one example per LoRA, each labelled."},
                  {"image": contact_sheet_tiles(tiles, cfg.get("contact_sheet_size", 1120)),
                   "max_side": cfg.get("contact_sheet_size", 1120)}]
    else:
        parts += [{"text": "REFERENCE IMAGE (the style to match):"}, {"image": reference}]
        for r, t in thumbs:
            ex = next((e for e in r["examples"] if e.get("thumb") == t), None)
            text = f"LORA {library.card(r)}"
            if ex and ex.get("prompt"):
                text += f"\n  example prompt: {ex['prompt'][:220]}"
            parts.append({"text": text + ("\n  example image:" if t else "\n  (no example image)")})
            if t:
                parts.append({"image": Path(t), "max_side": side})
    parts.append({"text": "Now pick the LoRAs (and alternatives) that best match the REFERENCE's style."})

    enum = list(stems)
    item = {"type": "object", "properties": {"lora": {"type": "string", "enum": enum},
                                             "strength": {"type": "number"}, "why": {"type": "string"}},
            "required": ["lora", "strength", "why"], "additionalProperties": False}
    schema = {"type": "object",
              "properties": {"picks": {"type": "array", "maxItems": max_loras, "items": item},
                             "alternatives": {"type": "array", "maxItems": 3, "items": item},
                             "notes": {"type": "string"}},
              "required": ["picks", "alternatives", "notes"], "additionalProperties": False}
    data, _cost, _tokens = backend.complete(INSTRUCTIONS, parts, schema, "lora_pick", side)

    def clean(items, limit):
        out, seen = [], set()
        for it in items or []:
            name = stems.get(it.get("lora"))
            if name and name not in seen:
                seen.add(name)
                w = min(LORA_STRENGTH[1], max(0.1, float(it.get("strength") or 0.6)))
                out.append((name, round(w, 2), it.get("why", "")))
        return out[:limit]

    picks = clean(data.get("picks"), max_loras)
    alts = [a for a in clean(data.get("alternatives"), 3) if a[0] not in {p[0] for p in picks}]
    return {"picks": picks, "alternatives": [(n, w) for n, w, _ in alts], "alternative_reasons": [a[2] for a in alts],
            "notes": data.get("notes", ""), "considered": considered, "shortlist": [r["name"] for r in cands],
            "shortlist_reason": shortlist_note}


def _short_card(rec: dict) -> str:
    """One line per LoRA for the shortlist stage, when the whole catalog is on offer."""
    if rec.get("brief"):
        return f"{lora_stem(rec['name'])}: {rec['title'][:52]} {rec['brief']}"
    bits = [f"{lora_stem(rec['name'])}: {rec['title'][:52]}"]
    kind = " / ".join(x for x in (rec.get("type"), (rec.get("category") or "").split("/")[-1]) if x)
    if kind:
        bits.append(f"[{kind}]")
    if rec.get("description"):
        bits.append(rec["description"][:120])
    if rec.get("typical_weight") is not None:
        bits.append(f"usual weight {rec['typical_weight']:g}")
    return " ".join(bits)


def contact_sheet_tiles(tiles: list[tuple[str, Path]], size: int):
    """contact_sheet() for arbitrary labelled tiles."""
    (first_label, first), rest = tiles[0], tiles[1:]
    sheet = contact_sheet(first, [p for _, p in rest], size, labels=[first_label] + [l for l, _ in rest])
    return sheet


def prompt_notes(library: LoraLibrary, loras: tuple) -> str:
    """For the prompt writer: what each active managed LoRA does, its trigger words, and
    how its example prompts describe the style."""
    index = library.index()
    lines = []
    for name, w in loras:
        rec = index.get(name)
        if not rec:
            continue
        kind = f" ({rec['type']})" if rec.get("type") else ""
        what = f" - {rec['description'][:160]}" if rec.get("description") else ""
        line = (f"- {rec.get('title', lora_stem(name))[:60]}{kind} at strength {w:g}{what}; "
                f"trigger words: {', '.join(rec['trigger_words'][:6]) or 'none'}")
        ex = next((e["prompt"] for e in rec["examples"] if e.get("prompt")), "")
        if ex:
            line += f"; an example prompt made with it: {ex[:260]}"
        lines.append(line)
    return "\n".join(lines)


SUGGEST_INSTRUCTIONS = """You choose LoRAs for a Stable Diffusion XL (Pony-family) render.

You get what the person is trying to make - their prompt, and often a reference image -
and a numbered menu of every LoRA installed here, each with its type, category and a
one-line description of what it does.

- Pick only LoRAs that clearly serve THIS prompt. Most renders need none or one; three is
  a lot. An unnecessary LoRA costs quality.
- A style LoRA changes the drawing style; a character LoRA changes who is depicted; a
  pose or concept LoRA forces a specific pose, act or object into the image. Don't pick
  two that fight each other (two styles, two poses).
- When a reference image is given, match its drawing style and subject, not just the words.
- strength: stay inside the range given for that LoRA; use its default unless you have a
  reason, and go lower when stacking several.
- Say in one short sentence why each pick helps this prompt.
- Picking nothing is a valid answer when nothing fits.

Reply with JSON only."""


def _menu_line(n: int, e: dict) -> str:
    if e.get("brief"):
        return f"{n}. {e['title'][:58]}: {e['brief']}"
    bits = [f"{n}. {e['title'][:58]}"]
    kind = " / ".join(x for x in (e.get("type"), e.get("category", "").split("/")[-1]) if x)
    if kind:
        bits.append(f"[{kind}]")
    if e.get("summary"):
        bits.append(e["summary"][:130])
    w = e.get("weight") or {}
    bits.append(f"(weight {w.get('min', 0.2):g}-{w.get('max', 1.2):g}, usually {w.get('default', 0.8):g})")
    return " ".join(bits)


def suggest_loras(backend, cards: list[dict], positive: str = "", negative: str = "",
                  description: str = "", reference=None, max_loras: int = 3,
                  max_side: int = 512, selected: list[dict] | None = None) -> dict:
    """Ask the model which of the installed LoRAs suit this prompt.

    One call with the whole menu: each LoRA is one line (~30 tokens), so ~500 of them
    cost well under a cent and sit inside a fraction of the model's window. The model
    answers with menu numbers rather than names, which keeps the schema small and makes
    a wrong answer easy to drop.

    `selected` are LoRAs the person already picked: they stay, count toward max_loras,
    aren't offered again, and the model is told not to add anything that fights them.
    """
    selected = selected or []
    kept = {e.get("comfy_name") for e in selected}
    cards = [e for e in cards if e.get("comfy_name") not in kept]
    max_loras = max_loras - len(selected)
    if not cards or max_loras <= 0:
        return {"picks": [], "considered": len(cards), "cost_usd": 0.0,
                "notes": "no LoRAs indexed" if not cards else
                         f"{len(selected)} already selected, which is the limit; none added."}
    menu = "\n".join(_menu_line(i + 1, e) for i, e in enumerate(cards))
    wanted = ("WHAT THEY WANT TO MAKE\n"
              + "\n".join(x for x in (f"prompt: {positive.strip()}" if positive.strip() else "",
                                      f"negative prompt: {negative.strip()}" if negative.strip() else "",
                                      f"description: {description.strip()}" if description.strip() else "")
                          if x) or "WHAT THEY WANT TO MAKE\n(only the reference image)")
    parts = [{"text": wanted}]
    if reference is not None:
        parts += [{"text": "REFERENCE IMAGE (match its style and subject):"}, {"image": reference}]
    if selected:
        parts.append({"text": "ALREADY SELECTED by the person (these stay; add only what they lack, and "
                              "nothing that fights them, e.g. a second style or pose):\n"
                              + "\n".join(f"- {e.get('title', '')[:58]} [{e.get('type', '')}] "
                                           f"{(e.get('summary') or '')[:130]}" for e in selected)})
    parts += [{"text": f"AVAILABLE LORAS\n{menu}"},
              {"text": f"TASK: choose at most {max_loras} LoRAs from the menu that best serve this render, "
                       "by their numbers. Fewer is better. JSON only."}]
    schema = {
        "type": "object",
        "properties": {
            "picks": {
                "type": "array", "maxItems": max_loras,
                "items": {"type": "object",
                          "properties": {"n": {"type": "integer"},
                                         "strength": {"type": "number"},
                                         "why": {"type": "string"}},
                          "required": ["n", "strength", "why"], "additionalProperties": False},
            },
            "notes": {"type": "string"},
        },
        "required": ["picks", "notes"], "additionalProperties": False,
    }
    data, cost, _tokens = backend.complete(SUGGEST_INSTRUCTIONS, parts, schema, "lora_suggest", max_side)
    picks, seen = [], set()
    for p in data.get("picks") or []:
        i = int(p.get("n", 0)) - 1
        if not (0 <= i < len(cards)) or i in seen or len(picks) >= max_loras:
            continue  # a number it made up, one picked twice, or past the limit
        seen.add(i)
        e = cards[i]
        w = e.get("weight") or {}
        strength = float(p.get("strength", w.get("default", 0.8)))
        strength = max(float(w.get("min", 0.1)), min(float(w.get("max", 1.5)), strength))
        picks.append({**e, "strength": round(strength, 2), "why": (p.get("why") or "").strip()})
    return {"picks": picks, "notes": (data.get("notes") or "").strip(), "cost_usd": cost,
            "considered": len(cards)}
