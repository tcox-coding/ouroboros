"""Repeatable DeepInfra configuration trials; no ComfyUI generation or config changes.

Run with python -m ouroboros.model_benchmark. Raw evidence is kept in eval/results.
The best *tested* options are cached, with provisional calibration clearly labelled.
"""
from __future__ import annotations

import concurrent.futures
import copy
import json
import threading
import time
from pathlib import Path

from . import backends, judge_eval, model_profiles
from .runner import ROOT, load_config

MODELS = [
    "XiaomiMiMo/MiMo-V2.6-Pro", "moonshotai/Kimi-K2.6",
    "Qwen/Qwen3.5-397B-A17B", "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp", "moonshotai/Kimi-K3",
    "google/gemma-4-26B-A4B-it", "google/gemma-4-31B-it-turbo", "Qwen/Qwen3-VL-30B-A3B-Instruct",
]
CONTEXT = [1048576, 262144, 262144, 1048576, 1048576, 262144, 262144, 262144]


def choose_trial(trials):
    # Missing calls cannot win by leaving hard examples out of the denominator.
    return max(trials, key=lambda r: (-r["failed_calls"], r["quality"], -r["cost_usd"], -r["seconds_per_call"]))


def main():
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = ROOT / "eval" / "results" / f"profiles-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    spent = 0.0
    original = backends.DeepInfraBackend.complete

    def measured(self, *args, **kwargs):
        nonlocal spent
        with lock:
            if spent >= 15:
                raise RuntimeError("Benchmark's $15 spending limit reached")
        cost = 0.0
        try:
            result = original(self, *args, **kwargs)
            cost = result[1]
            return result
        except Exception as e:
            cost = float(getattr(e, "cost_usd", 0.0))
            raise
        finally:
            with lock:
                spent += cost

    backends.DeepInfraBackend.complete = measured

    def run(model, context):
        folder = out / model.replace("/", "_")
        folder.mkdir()
        def save(name, data):
            (folder / f"{name}.json").write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
        def log(msg):
            with (folder / "progress.log").open("a") as f:
                f.write(msg + "\n")
        print(f"START {model}", flush=True)
        trials = []
        efforts = [None, "low"] if "gemma" in model or "Instruct" in model else ["none", "low"]
        for effort in efforts:
            options = {"temperature": 0, "top_p": None, "max_tokens": 4096 if effort != "low" else 8192,
                       "max_output_tokens": 8192, "context_tokens": context, "reasoning_effort": effort,
                       "timeout": 120, "retries": 0}
            row = judge_eval.evaluate(model, 1, log=log, backend="deepinfra", options=options,
                                      image_ids={"s_good_a", "s_crown", "b_good", "b_dress"})
            trials.append(row)
            save(f"pilot-{effort or 'default'}", row)
            print(f"PILOT {model} {effort}: quality={row['quality']} failed={row['failed_calls']} ${row['cost_usd']:.3f}", flush=True)
        winner = choose_trial(trials)
        if winner["failed_calls"] == winner["calls"] and None not in efforts:
            options.update(reasoning_effort=None, max_tokens=8192)
            row = judge_eval.evaluate(model, 1, log=log, backend="deepinfra", options=options,
                                      image_ids={"s_good_a", "s_crown", "b_good", "b_dress"})
            trials.append(row)
            save("pilot-default", row)
            winner = choose_trial(trials)
        if winner["failed_calls"] == winner["calls"]:
            save("unavailable", {"model": model, "trials": trials})
            return {"model": model, "error": "All pilot calls failed", "cost_usd": sum(t["cost_usd"] for t in trials)}
        options = winner["options"]
        review = judge_eval.evaluate(model, 2, log=log, backend="deepinfra", options=options)
        save("review", review)
        print(f"REVIEW {model}: quality={review['quality']} failed={review['failed_calls']}", flush=True)
        confirm = judge_eval.evaluate(model, 2, log=log, backend="deepinfra", options=options, confirm=True)
        save("confirmation", confirm)
        # Prompt writing is exercised separately with a clothed character description.
        cfg = copy.deepcopy(load_config()["judge"])
        cfg.update(backend="deepinfra", prompt_model="", confirm_model="")
        cfg["deepinfra"] = {**options, "model": model}
        from .prompter import write_prompt
        started = time.monotonic()
        try:
            prompt = write_prompt(backends.make_backend(cfg),
                                  "An adult woman in a burgundy military uniform, black bob haircut, green eyes, "
                                  "standing with one hand on her hip, flat cel shading.", "", "", "", "",
                                  ROOT / "eval/images/soldier_reference.png", 512)
            prompt_test = {"ok": bool(prompt.get("positive")), "seconds": time.monotonic() - started, "result": prompt}
        except Exception as e:
            prompt_test = {"ok": False, "seconds": time.monotonic() - started, "error": str(e)[:400]}
        save("prompt", prompt_test)
        # Exercise the image-limit fallback with the same approved evaluation images.
        names = ["soldier_reference", "s_good_a", "s_crown", "s_tabs", "d_wrong"]
        parts = [p for i, name in enumerate(names) for p in
                 ({"text": f"IMAGE {i + 1}: {name}"}, {"image": ROOT / f"eval/images/{name}.png"})]
        try:
            answer, cost, _ = backends.make_backend(cfg).complete(
                "Report how many distinct numbered images or panels you received. JSON only.", parts,
                {"type": "object", "properties": {"count": {"type": "integer"}},
                 "required": ["count"], "additionalProperties": False}, "image_limit_probe", 512)
            image_test = {"ok": answer.get("count") == 5, "answer": answer, "cost_usd": cost,
                          "max_images": backends._IMAGE_LIMITS.get(model)}
        except Exception as e:
            image_test = {"ok": False, "error": str(e)[:300]}
        save("images", image_test)
        settings = {"judge": {"backend": "deepinfra", "deepinfra": {k: v for k, v in options.items()
                     if k not in ("model", "url", "price_per_mtok", "timeout", "retries", "max_output_tokens")},
                     "image_max_side": 512, "contact_sheet": False},
                    "loop": {"history_keep_rounds": 2, "prompt_token_budget": 8000}}
        if image_test.get("max_images"):
            settings["judge"]["deepinfra"]["max_images"] = image_test["max_images"]
        cal = review.get("calibration")
        if cal and not review["failed_calls"]:
            settings["loop"]["threshold"] = cal["threshold"]
        thresholds = {}
        if confirm.get("calibration") and not confirm["failed_calls"]:
            thresholds["loop"] = confirm["calibration"]["threshold"]
        profile = {"label": model + " · measured " + stamp[:8], "source": "local benchmark",
                   "tested_at": stamp, "settings": settings, "thresholds": thresholds,
                   "notes": "Best of two tested configurations on the local labelled image set. "
                            "Thresholds are provisional and fitted to this set; Designer uses a separate rubric. "
                            f"Review quality {review['quality']}/100, {review['seconds_per_call']} s/call; "
                            f"{review['failed_calls']} review failures, {confirm['failed_calls']} confirmation failures. "
                            f"Prompt smoke test: {'passed' if prompt_test['ok'] else 'failed'}.",
                   "evidence": str(folder.relative_to(ROOT)),
                   "metrics": {"review": {k: v for k, v in review.items() if k not in ('per_call', 'per_image', 'options')},
                               "confirmation": {k: v for k, v in confirm.items() if k not in ('per_call', 'per_image', 'options')},
                               "prompt_ok": prompt_test["ok"], "image_probe_ok": image_test["ok"]}}
        # Avoid silently promoting an unreliable configuration.
        if not review["failed_calls"] and not confirm["failed_calls"]:
            model_profiles.save_profile(model, "deepinfra", profile)
        save("profile", profile)
        print(f"DONE {model}: review={review['quality']} confirm={confirm['quality']} prompt={prompt_test['ok']}", flush=True)
        return profile

    rows = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(run, m, c): m for m, c in zip(MODELS, CONTEXT)}
            for future in concurrent.futures.as_completed(futures):
                try:
                    rows.append(future.result())
                except Exception as e:
                    rows.append({"model": futures[future], "error": str(e)[:500]})
                (out / "summary.json").write_text(json.dumps({"cost_usd": spent, "results": rows}, indent=2))
    finally:
        backends.DeepInfraBackend.complete = original
    print(f"COMPLETE ${spent:.4f}; evidence: {out}", flush=True)


if __name__ == "__main__":
    main()
