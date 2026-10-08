"""Run ComfyUI headless in the background, so the Comfy Desktop app doesn't have to be open.

It uses the ComfyUI that Comfy Desktop installed: the same Python environment, custom
nodes, model folders, input/output folders and launch arguments (read from Comfy
Desktop's installations.json / settings.json), started as a hidden process with no
window and no browser. Its output goes to logs/comfyui.log.

If something is already answering on the ComfyUI URL (e.g. the desktop app is open), that
one is used and nothing is started. A ComfyUI started here is stopped when Ouroboros
exits (comfyui.stop_on_exit).
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

if sys.platform == "win32":
    DESKTOP_CONFIG = Path(os.environ.get("APPDATA", "")) / "Comfy Desktop"
    DESKTOP_DATA = DESKTOP_CONFIG
else:
    # Comfy Desktop 2 on Linux: settings in ~/.config, installations in ~/.local/share.
    _xdg_config = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    _xdg_data = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    DESKTOP_CONFIG = next((d for d in (_xdg_config / "comfyui-desktop-2", _xdg_config / "Comfy Desktop")
                           if (d / "settings.json").exists()), _xdg_config / "comfyui-desktop-2")
    DESKTOP_DATA = next((d for d in (_xdg_data / "comfyui-desktop-2", DESKTOP_CONFIG)
                         if (d / "installations.json").exists()), DESKTOP_CONFIG)
_job_handle = None


def _tie_to_this_process(proc: subprocess.Popen) -> None:
    """Windows: put the child in a job object that is killed when this process ends,
    however it ends (closing the console window skips Python's exit handlers)."""
    global _job_handle
    if sys.platform.startswith("linux"):
        return  # handled by _linux_parent_death_signal in the child
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount",
                    "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.OpenProcess.restype = wintypes.HANDLE
    if _job_handle is None:
        _job_handle = k32.CreateJobObjectW(None, None)
        info = EXTENDED()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        k32.SetInformationJobObject(_job_handle, 9, ctypes.byref(info), ctypes.sizeof(info))
    handle = k32.OpenProcess(0x1F0FFF, False, proc.pid)  # PROCESS_ALL_ACCESS
    if handle:
        k32.AssignProcessToJobObject(_job_handle, handle)
        k32.CloseHandle(handle)


def _linux_parent_death_signal() -> None:
    """Linux (runs in the child before exec): SIGTERM ComfyUI when this process dies,
    however it dies (a closed terminal skips Python's exit handlers)."""
    import ctypes
    import signal
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


_spawn_requests: queue.Queue = queue.Queue()
_spawner: threading.Thread | None = None
_spawner_lock = threading.Lock()


def _spawn_forever() -> None:
    while True:
        args, kwargs, done = _spawn_requests.get()
        try:
            done["proc"] = subprocess.Popen(*args, **kwargs)
        except BaseException as e:  # handed back to the caller
            done["error"] = e
        done["event"].set()


def _popen_tied(*args, **kwargs) -> subprocess.Popen:
    """Popen with PR_SET_PDEATHSIG. The kernel sends that signal when the *thread* that
    forked the child exits, not the process, so every child is forked from one thread
    that lives as long as Ouroboros (the callers are short-lived worker threads)."""
    global _spawner
    with _spawner_lock:
        if _spawner is None:
            _spawner = threading.Thread(target=_spawn_forever, name="comfyui-spawner", daemon=True)
            _spawner.start()
    done = {"event": threading.Event()}
    _spawn_requests.put((args, {**kwargs, "preexec_fn": _linux_parent_death_signal}, done))
    done["event"].wait()
    if "error" in done:
        raise done["error"]
    return done["proc"]


def detect_desktop_install() -> dict:
    """What Comfy Desktop runs: {"install", "python", "main", "args", "name"} or {} if not found."""
    try:
        installs = json.loads((DESKTOP_DATA / "installations.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    try:
        last = json.loads((DESKTOP_CONFIG / "last-session.json").read_text(encoding="utf-8")).get("installationId")
    except (OSError, ValueError):
        last = None
    try:
        settings = json.loads((DESKTOP_CONFIG / "settings.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        settings = {}
    installs = [i for i in installs if i.get("installPath")]  # skips e.g. Comfy Cloud entries
    inst = next((i for i in installs if i.get("id") == last), None) or next(
        (i for i in installs if i.get("status") == "installed"), None)
    if not inst:
        return {}
    root = Path(inst["installPath"])
    python = next((p for p in (root / "ComfyUI" / ".venv" / "Scripts" / "python.exe",
                               root / "standalone-env" / "python.exe",
                               root / "ComfyUI" / ".venv" / "bin" / "python3",
                               root / "standalone-env" / "bin" / "python3") if p.exists()), None)
    main = root / "ComfyUI" / "main.py"
    if not python or not main.exists():
        return {}
    args = (inst.get("launchArgs") or "").split()
    paths = DESKTOP_DATA / "instance-model-paths" / f"{inst['id']}.yaml"
    if paths.exists():
        args += ["--extra-model-paths-config", str(paths)]
    if settings.get("inputDir"):
        args += ["--input-directory", settings["inputDir"]]
    if settings.get("outputDir"):
        args += ["--output-directory", settings["outputDir"]]
    return {"install": str(root), "python": str(python), "main": str(main), "args": args,
            "name": inst.get("name", root.name)}


class ComfyLauncher:
    def __init__(self, cfg: dict, comfy_url: str, log_dir: Path):
        self.cfg = cfg
        self.url = comfy_url.rstrip("/")
        self.log_file = log_dir / "comfyui.log"
        self.proc: subprocess.Popen | None = None
        self.state = "stopped"  # stopped | starting | running | external | failed
        self.message = ""
        self._lock = threading.Lock()
        atexit.register(self._atexit)

    # --- status ------------------------------------------------------------------
    def reachable(self, timeout: float = 2) -> bool:
        try:
            return requests.get(f"{self.url}/system_stats", timeout=timeout).ok
        except requests.RequestException:
            return False

    def launch_command(self) -> tuple[list[str], str] | None:
        """(command, working dir) from config, or detected from Comfy Desktop."""
        det = detect_desktop_install()
        python = self.cfg.get("python") or det.get("python")
        main = self.cfg.get("main") or det.get("main")
        if not python or not main:
            return None
        args = self.cfg.get("args") if self.cfg.get("args") is not None else det.get("args", [])
        u = urlparse(self.url)
        cmd = [python, main, "--listen", u.hostname or "127.0.0.1", "--port", str(u.port or 8188),
               "--disable-auto-launch", *args, *self.cfg.get("extra_args", [])]
        extra = self.model_paths_file()
        if extra:
            cmd += ["--extra-model-paths-config", str(extra)]
        return cmd, str(Path(main).parent.parent)

    def model_paths_file(self) -> Path | None:
        """A model-paths file adding the checkpoints folder chosen in Settings (checkpoints_dir)
        and the loras root (loras.comfy_root), so ComfyUI loads checkpoints and LoRAs from them
        as well as from its own folders: a LoRA folder added there for another model family
        (e.g. NoobAI-XL next to Pony) is then loadable without linking it into ComfyUI's.
        ComfyUI takes several of these files; Comfy Desktop's own one stays as it is."""
        lines = []
        for kind, folder in (("checkpoints", self.cfg.get("checkpoints_dir")), ("loras", self.cfg.get("loras_root"))):
            folder = (folder or "").strip()
            if folder and Path(folder).is_dir():
                lines.append(f"  {kind}: '" + str(Path(folder).resolve()).replace("'", "''") + "'")
        if not lines:
            return None
        path = self.log_file.parent / "comfy_model_paths.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Written by Ouroboros from Settings (Checkpoints folder, ComfyUI loras folder).\n"
                        "ouroboros:\n" + "\n".join(lines) + "\n", encoding="utf-8")
        return path

    def status(self) -> dict:
        if self.proc and self.proc.poll() is None:
            state = self.state
        elif self.reachable(1):
            state = "external" if not self.proc or self.proc.poll() is not None else "running"
        elif self.state == "starting":
            state = "starting"
        else:
            state = "failed" if self.state == "failed" else "stopped"
        lc = self.launch_command()
        return {"state": state, "message": self.message, "pid": self.proc.pid if self.proc and self.proc.poll() is None else None,
                "command": " ".join(f'"{c}"' if " " in c else c for c in lc[0]) if lc else None,
                "autostart": bool(self.cfg.get("autostart", True)), "log": self.tail()}

    def tail(self, lines: int = 25) -> str:
        try:
            text = self.log_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(l for l in text.splitlines()[-lines:] if "it/s]" not in l)

    # --- control -----------------------------------------------------------------
    def ensure_running(self, log=print, timeout: float | None = None) -> bool:
        """Start ComfyUI hidden unless something already answers on the URL. Blocks
        until it's ready (or fails / times out). Returns True when reachable."""
        with self._lock:
            if self.reachable():
                if not (self.proc and self.proc.poll() is None):
                    self.state, self.message = "external", "using the ComfyUI that's already running"
                else:
                    self.state = "running"
                return True
            if not (self.proc and self.proc.poll() is None):
                lc = self.launch_command()
                if not lc:
                    self.state = "failed"
                    self.message = ("ComfyUI isn't running and no install was found to start. Set "
                                    "comfyui.python and comfyui.main in config.json, or open Comfy Desktop.")
                    return False
                cmd, cwd = lc
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                logf = open(self.log_file, "w", encoding="utf-8", errors="replace")
                flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
                log("starting ComfyUI in the background (no window): " + " ".join(cmd[:2]))
                popen = (_popen_tied if sys.platform.startswith("linux")
                         and self.cfg.get("stop_on_exit", True) else subprocess.Popen)
                self.proc = popen(cmd, cwd=cwd, stdout=logf, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, creationflags=flags, env=env)
                if self.cfg.get("stop_on_exit", True):
                    try:
                        _tie_to_this_process(self.proc)
                    except Exception as e:  # not fatal: atexit still stops it on a normal exit
                        log(f"couldn't tie ComfyUI to this process ({e})")
                self.state, self.message = "starting", "starting ComfyUI (loading custom nodes)"
            deadline = time.monotonic() + (timeout or self.cfg.get("start_timeout", 300))
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    self.state = "failed"
                    self.message = f"ComfyUI exited with code {self.proc.returncode}; see logs/comfyui.log"
                    log(self.message)
                    return False
                if self.reachable(2):
                    self.state, self.message = "running", f"started in the background (pid {self.proc.pid})"
                    log("ComfyUI is ready")
                    return True
                time.sleep(2)
            self.state, self.message = "failed", "ComfyUI didn't answer in time; see logs/comfyui.log"
            return False

    def start_async(self, log=print) -> None:
        threading.Thread(target=self.ensure_running, args=(log,), daemon=True).start()

    def stop(self) -> bool:
        """Stop a ComfyUI started here (an external one is left alone)."""
        if not (self.proc and self.proc.poll() is None):
            return False
        self.proc.terminate()
        try:
            self.proc.wait(20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.state, self.message = "stopped", "stopped"
        return True

    def _atexit(self) -> None:
        if self.cfg.get("stop_on_exit", True):
            self.stop()
