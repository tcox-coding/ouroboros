"""Compare judge models on quality AND speed: python -m ouroboros eval MODEL [MODEL ...]

Every model judges the same images the way the loop does (one candidate per call, the
same rubric, instructions and schema), each image twice. The set (eval/judge_set.json)
holds references, images and hand labels:
  - pairs: (better, worse) images whose order is clear to a person;
  - flaws: for a flawed image, words that should appear in the judge's differences
    (e.g. "crown" for an image with a crown the reference doesn't have).

Quality (0-100) = 50% pair accuracy + 30% flaw recall + 20% consistency (the two scores
of one image agreeing; 10+ points apart counts as 0).
Speed: seconds per judge call as seen by the loop (wall clock, including model load and
retries), and the model's own output tokens per second.
Efficiency = quality points per minute of LLM time (quality / minutes per call). The
loop spends 3-4 calls per round, so a model twice as slow must be clearly better to win.
Also reported: calls that failed or needed a retry.

Results go to eval/results/<time>.json and a table is printed.
"""

from __future__ import annotations

import json
import statistics
import sys
import time

from .judge import Judge
from .runner import ROOT, load_config

SET_FILE = ROOT / "eval" / "judge_set.json"


from .judge import model_options as _preset


def evaluate(model: str, repeats: int = 2, log=print, temperature: float | None = None,
             backend: str | None = None, options: dict | None = None, confirm: bool = False,
             image_ids: set[str] | None = None) -> dict:
    cfg = load_config()
    jc = json.loads(json.dumps(cfg["judge"]))
    # A hosted model and a local one are compared the same way; --backend picks which
    # section of the judge config the model name belongs to.
    bk = backend or ("deepinfra" if "/" in model and ":" not in model else cfg["judge"]["backend"])
    jc["backend"] = bk
    jc["confirm_model"] = ""
    jc["prompt_model"] = ""
    jc[bk].update({k: v for k, v in _preset(model, bk).items()})
    jc[bk].update(options or {})
    jc[bk]["model"] = model
    if temperature is not None:  # --temp: same model, different sampling (consistency test)
        jc[bk]["temperature"] = temperature
    judge = Judge(jc, ["dpmpp_2m", "euler"], ["karras"])
    data = json.loads(SET_FILE.read_text(encoding="utf-8"))
    rubric = judge.rubric(True)
    results: dict[str, dict] = {}
    calls = []
    for group in data["groups"]:
        ref = ROOT / group["reference"]
        for img in group["images"]:
            if image_ids is not None and img["id"] not in image_ids:
                continue
            path = ROOT / img["path"]
            entry = results.setdefault(img["id"], {"scores": [], "differences": [], "flaw_hits": []})
            for _ in range(repeats):
                t0 = time.monotonic()
                billed, error = 0.0, None
                try:
                    if confirm:
                        r = judge.confirm(ref, group["goal"], "txt2img seed=1 steps=30 cfg=5.5 dpmpp_2m/karras denoise=1",
                                          path, None, "", rubric)
                    else:
                        r = judge.review(ref, group["goal"], "txt2img seed=1 steps=30 cfg=5.5 dpmpp_2m/karras denoise=1",
                                         "", [path], None, "", rubric)
                    billed = r.cost_usd
                    ok = True
                except Exception as e:
                    log(f"  {model} {img['id']}: failed ({str(e)[:120]})")
                    ok, r = False, None
                    billed, error = float(getattr(e, "cost_usd", 0.0)), str(e)[:300]
                wall = time.monotonic() - t0
                st = getattr(judge.backend, "last_stats", {}) or {}
                calls.append({"image": img["id"], "ok": ok, "seconds": round(wall, 1),
                              "cost_usd": billed, "error": error, "attempts": st.get("attempts"),
                              "eval_count": st.get("eval_count"), "eval_s": (st.get("eval_duration") or 0) / 1e9,
                              "load_s": (st.get("load_duration") or 0) / 1e9})
                if not ok:
                    continue
                diffs = [str(d) for d in (r.raw.get("candidates") or [{}])[0].get("differences") or []]
                text = " ".join(diffs).lower()
                entry["scores"].append(r.scores[0])
                entry["differences"].append(diffs)
                if img.get("flaws"):
                    entry["flaw_hits"].append(any(w.lower() in text for w in img["flaws"]))
                log(f"  {model} {img['id']}: {r.scores[0]:g} in {wall:.0f}s")
    mean = {k: statistics.mean(v["scores"]) for k, v in results.items() if v["scores"]}
    pairs = [(a, b) for g in data["groups"] for a, b in g["pairs"] if a in mean and b in mean]
    pair_acc = sum(mean[a] > mean[b] for a, b in pairs) / len(pairs) if pairs else 0.0
    hits = [h for v in results.values() for h in v["flaw_hits"]]
    flaw_recall = sum(hits) / len(hits) if hits else 0.0
    spreads = [abs(v["scores"][0] - v["scores"][1]) for v in results.values() if len(v["scores"]) >= 2]
    spread = statistics.mean(spreads) if spreads else None
    if spread is None:  # one pass per image: consistency can't be measured, so it's left out
        quality = round(100 * (0.5 * pair_acc + 0.3 * flaw_recall) / 0.8, 1)
    else:
        quality = round(100 * (0.5 * pair_acc + 0.3 * flaw_recall + 0.2 * max(0.0, 1 - spread / 10)), 1)
    good = [c for c in calls if c["ok"]]
    sec = statistics.mean(c["seconds"] for c in calls) if calls else 0.0
    toks = [c["eval_count"] / c["eval_s"] for c in good if c["eval_count"] and c["eval_s"]]
    expected_pairs = [(a, b) for g in data["groups"] for a, b in g["pairs"]
                      if image_ids is None or (a in image_ids and b in image_ids)]
    good_ids = {a for a, _ in expected_pairs}
    bad_ids = {b for _, b in expected_pairs}
    good_scores = [s for i in good_ids for s in results.get(i, {}).get("scores", [])]
    bad_scores = [s for i in bad_ids for s in results.get(i, {}).get("scores", [])]
    calibration = None
    if good_scores and bad_scores and all(results.get(i, {}).get("scores") for i in good_ids | bad_ids):
        low_good, high_bad = min(good_scores), max(bad_scores)
        threshold = min(100.0, round((low_good + high_bad) / 2 if low_good > high_bad else high_bad + 0.1, 1))
        calibration = {"threshold": threshold, "margin": round(low_good - high_bad, 2),
                       "false_pass_rate": sum(s >= threshold for s in bad_scores) / len(bad_scores),
                       "good_pass_rate": sum(s >= threshold for s in good_scores) / len(good_scores),
                       "provisional": True}
    return {"cost_usd": sum(c["cost_usd"] for c in calls), "options": jc[bk],
            "confirmation": confirm, "calibration": calibration,
            "expected_pairs": len(expected_pairs),
            "model": model + (f" (temp {temperature})" if temperature is not None else ""), "quality": quality, "pair_accuracy": round(pair_acc, 3), "pairs": len(pairs),
            "flaw_recall": round(flaw_recall, 3), "flaw_checks": len(hits),
            "score_spread": round(spread, 1) if spread is not None else None, "repeats": repeats,
            "seconds_per_call": round(sec, 1), "tokens_per_s": round(statistics.mean(toks), 1) if toks else None,
            "failed_calls": sum(not c["ok"] for c in calls), "calls": len(calls),
            "efficiency": round(quality / (sec / 60), 1) if good else 0.0,
            "mean_scores": {k: round(v, 1) for k, v in mean.items()}, "per_call": calls,
            "per_image": results}


def main(models: list[str], repeats: int = 2, temperature: float | None = None,
         backend: str | None = None) -> None:
    out = ROOT / "eval" / "results"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in models:
        print(f"== {m}")
        rows.append(evaluate(m, repeats, temperature=temperature, backend=backend))
        (out / f"{time.strftime('%Y%m%d-%H%M%S')}_{m.replace('/', '_').replace(':', '_')}.json").write_text(
            json.dumps(rows[-1], indent=1), encoding="utf-8")
    print(f"\n{'model':55} {'quality':>7} {'pairs':>6} {'flaws':>6} {'spread':>6} {'s/call':>7} {'tok/s':>6} {'fail':>5} {'effic.':>7}")
    for r in rows:
        print(f"{r['model']:55} {r['quality']:7.1f} {r['pair_accuracy']:6.0%} {r['flaw_recall']:6.0%} "
              f"{r['score_spread'] if r['score_spread'] is not None else '-':>6} "
              f"{r['seconds_per_call']:7.1f} {r['tokens_per_s'] or 0:6.1f} "
              f"{r['failed_calls']:5d} {r['efficiency']:7.1f}")


if __name__ == "__main__":
    temp = next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--temp=")), None)
    main([a for a in sys.argv[1:] if not a.startswith("-")],
         int(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--repeats=")), 2)),
         float(temp) if temp is not None else None,
         next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--backend=")), None))
