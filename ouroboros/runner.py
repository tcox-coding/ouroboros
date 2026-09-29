"""Processes the job queue (several jobs at once) and keeps live state for the web UI."""

from __future__ import annotations

import copy
import json
import shutil
import threading
import time
import traceback
from collections import deque
from pathlib import Path

from .comfy import ComfyClient
from .comfy_launcher import ComfyLauncher
from .jobs import Queue, load_job
from .llm_queue import GatedBackend, LLMGate, set_label
from .judge import Judge
from .loop import run_job
from .loras import LoraLibrary
from .workflow import Workflows

ROOT = Path(__file__).resolve().parent.parent


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config() -> dict:
    """config.example.json overlaid with config.json, so new keys get defaults."""
    cfg = json.loads((ROOT / "config.example.json").read_text(encoding="utf-8"))
    user = ROOT / "config.json"
    if user.exists():
        cfg = _merge(cfg, json.loads(user.read_text(encoding="utf-8")))
    return cfg


def save_config(changes: dict) -> dict:
    user = ROOT / "config.json"
    current = json.loads(user.read_text(encoding="utf-8")) if user.exists() else {}
    user.write_text(json.dumps(_merge(current, changes), indent=2), encoding="utf-8")
    return load_config()


_launchers: dict[str, ComfyLauncher] = {}


def comfy_launcher(cfg: dict) -> ComfyLauncher:
    """One launcher per ComfyUI URL, shared by the web server and the queue runner."""
    url = cfg["comfy_url"]
    if url not in _launchers:
        _launchers[url] = ComfyLauncher(cfg.get("comfyui", {}), url, ROOT / "logs")
    else:
        _launchers[url].cfg = cfg.get("comfyui", {})
    return _launchers[url]


def lora_library(cfg: dict) -> LoraLibrary:
    return LoraLibrary({**cfg.get("loras", {}), "image_max_side": 384}, ROOT / "cache")


def rel_url(path: Path | str | None) -> str | None:
    """/files/... URL for a file under the project folder."""
    if path is None:
        return None
    return "/files/" + Path(path).resolve().relative_to(ROOT).as_posix()


class Runner:
    """Runs queued jobs, up to queue.parallel_jobs at once, each on its own thread.

    Jobs overlap where it helps: ComfyUI queues their renders, and their LLM calls take
    turns at one gate (llm_queue.LLMGate), so while one job waits for the LLM another's
    images render. Each job's own loop (loop.run_job) is unchanged."""

    FINISHED_KEEP = 6  # finished jobs still shown in the Run tab

    def __init__(self):
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.stop_requested = False
        self.log: deque[dict] = deque(maxlen=600)
        self.session_cost = 0.0
        self.jobs: dict[str, dict] = {}      # job folder name -> live state (see _report)
        self.order: list[str] = []           # job keys, most recently started last
        self.run_dirs: dict[str, Path] = {}
        self.last_error: str | None = None
        self.gate = LLMGate()

    # --- control -------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, once: bool = False) -> bool:
        with self.lock:
            if self.running:
                return False
            self.stop_requested = False
            self.last_error = None
            self.thread = threading.Thread(target=self._run, args=(once,), daemon=True)
            self.thread.start()
            return True

    def stop(self) -> None:
        """Finish the current rounds, then stop. The jobs stay in the queue."""
        self.stop_requested = True
        self._log("Stop requested; finishing the current rounds.")

    def run_blocking(self, once: bool = False) -> None:
        self._run(once, echo=True)

    # --- state ---------------------------------------------------------------
    def snapshot(self) -> dict:
        with self.lock:
            jobs = [copy.deepcopy(self.jobs[k]) for k in self.order if k in self.jobs]
        active = [j for j in jobs if j.get("active")]
        return {
            "running": self.running,
            "stopping": self.running and self.stop_requested,
            "session_cost": round(self.session_cost, 4),
            "jobs": jobs,
            # The job the Run tab opens on: the most recently started one still running.
            "current": (active or jobs or [None])[-1],
            "llm": self.gate.status(),
            "log": list(self.log)[-150:],
            "error": self.last_error,
        }

    @property
    def current(self) -> dict | None:
        return self.snapshot()["current"]

    def active_runs(self) -> set[str]:
        with self.lock:
            return {j["run"] for j in self.jobs.values() if j.get("active")}

    def _log(self, text: str) -> None:
        with self.lock:
            self.log.append({"t": time.strftime("%H:%M:%S"), "text": text})
        if self._echo:
            print(text)

    _echo = False

    def _report(self, e: dict, key: str) -> None:
        kind = e["type"]
        if kind == "log":
            self._log(e["text"])
            return
        with self.lock:
            cur = self.jobs.get(key)
            if kind == "job_start":
                self.run_dirs[key] = Path(e["run_dir"])
                self.jobs[key] = {
                    "key": key, "active": True,
                    "job": e["job"], "run": Path(e["run_dir"]).name, "reference": rel_url(e["reference"]),
                    "threshold": e["threshold"], "max_rounds": e["max_rounds"], "round": 0,
                    "phase": "explore", "stage": "starting", "rounds": [], "best_score": None,
                    "best_image": None, "cost_usd": 0.0, "context_window": e.get("context_window"),
                    "prompt": None,
                }
                if key in self.order:
                    self.order.remove(key)
                self.order.append(key)
                done = [k for k in self.order if not self.jobs.get(k, {}).get("active")]
                for k in done[:-self.FINISHED_KEEP]:
                    self.order.remove(k)
                    self.jobs.pop(k, None)
                return
            if cur is None:
                return
            if kind == "stage":
                cur.update(round=e["round"], phase=e["phase"], stage=e["stage"])
            elif kind == "setup":
                cur["setup"] = {k: e.get(k) for k in ("checkpoint", "checkpoint_base", "lora_mode",
                                                      "start_loras", "lora_pick", "incompatible_loras",
                                                      "pose")}
            elif kind == "size" and cur.get("setup") is not None:
                cur["setup"].update(size=e["size"], size_setting=e["size_setting"])
            elif kind == "prompt":
                cur["prompt"] = {k: e.get(k) for k in
                                 ("description", "positive", "negative", "merged", "dropped", "notes")}
            elif kind == "round":
                cur["rounds"].append({
                    "round": e["round"], "phase": e["phase"], "images": [rel_url(p) for p in e["images"]],
                    "scores": e["scores"], "params": e["params"], "best": e["round_best"],
                    "score": e["round_score"], "diagnosis": e["diagnosis"], "edit": e["edit"],
                    "prompt_tokens": e.get("prompt_tokens"), "differences": e.get("differences"),
                })
                cur.update(best_score=e["best_score"], best_image=rel_url(e["best_image"]),
                           cost_usd=round(e["cost_usd"], 4), stage="planning next round")
            elif kind == "confirm" and cur["rounds"]:
                cur["rounds"][-1].setdefault("confirms", []).append({
                    "image": rel_url(e["image"]), "first": e["first"], "recheck": e["recheck"],
                    "passed": e["passed"], "diagnosis": e["diagnosis"]})
                cur.update(best_score=e["best_score"], best_image=rel_url(e["best_image"]),
                           cost_usd=round(e["cost_usd"], 4))
            elif kind == "memory" and cur["rounds"]:
                cur["rounds"][-1]["memory"] = e["summary"]
            elif kind == "hands":  # the automatic hand refine after the loop
                auto = {k: v for k, v in e.items() if k not in ("type", "image", "before")}
                auto.update(image_url=rel_url(e["image"]), before_url=rel_url(e["before"]))
                cur["hands"] = {"auto": auto, "manual": [], "busy": False}

    def _finish_state(self, key: str, stage: str) -> None:
        with self.lock:
            if key in self.jobs:
                self.jobs[key].update(active=False, stage=stage)

    def _record_failure(self, key: str, error: str) -> None:
        """A run that crashed still gets a summary.json (status "failed", the error and
        its best image so far), so it shows in History and can be viewed or removed."""
        d = self.run_dirs.get(key)
        if not d or not d.is_dir() or (d / "summary.json").exists():
            return
        cur = self.jobs.get(key) or {}
        best = (cur.get("best_image") or "").rsplit("/", 1)[-1] or None
        if best and (d / best).exists():
            shutil.copy2(d / best, d / "best.png")
        (d / "summary.json").write_text(json.dumps({
            "job": cur.get("job") or d.name, "status": "failed", "error": error,
            "best_score": cur.get("best_score"), "best_image": best if best and (d / best).exists() else None,
            "rounds": len(cur.get("rounds") or []), "cost_usd": cur.get("cost_usd", 0.0),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "best": None,
        }, indent=2), encoding="utf-8")

    # --- worker --------------------------------------------------------------
    def _run(self, once: bool, echo: bool = False) -> None:
        self._echo = echo
        try:
            cfg = load_config()
            launcher = comfy_launcher(cfg)
            if cfg.get("comfyui", {}).get("autostart", True):
                if not launcher.ensure_running(self._log):
                    raise RuntimeError(launcher.message)
            comfy = ComfyClient(cfg["comfy_url"])
            samplers = comfy.choices("KSampler", "sampler_name")
            schedulers = comfy.choices("KSampler", "scheduler")
            flows = Workflows(ROOT / "workflows")
            judge = Judge(cfg["judge"], samplers, schedulers)
            judge.backend = GatedBackend(judge.backend, self.gate)  # every job's LLM calls take turns
            if judge.confirm_backend is not None:
                judge.confirm_backend = GatedBackend(judge.confirm_backend, self.gate)
            library = lora_library(cfg)
        except Exception as e:
            self.last_error = f"Could not start: {e}"
            self._log(self.last_error)
            return

        queue = Queue(ROOT / "jobs")
        parallel = 1 if once else max(1, int(cfg.get("queue", {}).get("parallel_jobs", 2)))
        # A hosted judge answers several calls at once; one local GPU does not.
        self.gate.set_capacity(int(cfg.get("queue", {}).get("llm_parallel", 1)))
        self._log(f"Started (judge: {cfg['judge']['backend']}, up to {parallel} job(s) at once, "
                  f"{self.gate.status()['capacity']} LLM call(s) at once).")
        workers: dict[str, threading.Thread] = {}
        started = 0
        while True:
            for k in [k for k, t in workers.items() if not t.is_alive()]:
                del workers[k]
            if not self.stop_requested and not (once and started):
                for folder in queue.pending():
                    if len(workers) >= parallel:
                        break
                    if folder.name in workers or queue.is_running(folder):
                        continue
                    queue.mark_running(folder, True)
                    t = threading.Thread(target=self._job, daemon=True, name=f"job-{folder.name}",
                                         args=(folder, queue, cfg, comfy, flows, judge, samplers, schedulers,
                                               library))
                    workers[folder.name] = t
                    t.start()
                    started += 1
                    if once:
                        break
            if not workers and (self.stop_requested or (once and started)
                                or not [f for f in queue.pending() if not queue.is_running(f)]):
                break
            time.sleep(1)
        if started and not self.stop_requested:
            self._log("Queue empty.")
        self._log("Stopped." if self.stop_requested else "Idle.")

    def _job(self, folder: Path, queue: Queue, cfg, comfy, flows, judge, samplers, schedulers, library) -> None:
        key = folder.name
        set_label(key)
        try:
            job = load_job(folder)
            res = run_job(job, cfg, comfy, flows, judge, samplers, schedulers, ROOT / "runs",
                          lambda e: self._report(e, key), lambda: self.stop_requested, library)
            with self.lock:
                self.session_cost += res.cost_usd
            if res.status == "stopped":
                queue.mark_running(folder, False)
            elif queue.finish(folder, res.status) is None:
                self._log(f"{key}: job folder disappeared during the run; results are in runs/")
            self._log(f"{job.name}: {res.status}, best {res.best_score:g} in {res.rounds} rounds, "
                      f"${res.cost_usd:.3f}")
            self._finish_state(key, "stopped" if res.status == "stopped" else f"finished ({res.status})")
        except Exception as e:
            self._log(traceback.format_exc())
            self.last_error = f"{key}: {e}"
            self._record_failure(key, str(e))
            queue.finish(folder, "failed")
            self._finish_state(key, "failed")
        finally:
            set_label(None)
