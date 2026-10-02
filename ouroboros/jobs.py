"""File-based job queue.

A job is a folder in jobs/pending/. Either:
  - a prompt-generator saved prompt (positive.txt, negative.txt, settings.txt,
    reference image), dropped in as-is, or
  - reference.png + job.json {"prompt", "negative", "description", "threshold", ...}

With a "description" (or description.txt), the prompt is written from it when the job
starts; any prompt/negative given as well is merged with it (see prompter.py).

Finished jobs move to jobs/done/ (threshold reached) or jobs/review/ (best effort,
needs a human look). Crashes move to jobs/failed/. Jobs removed from the queue in
the web UI move to jobs/removed/ (nothing is deleted).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from .targets import ROLES, Targets, from_spec

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class Job:
    name: str
    folder: Path
    reference: Path
    positive: str
    negative: str
    description: str = ""
    settings: dict = field(default_factory=dict)   # seed/steps/cfg/sampler_name/scheduler/denoise/mode
    overrides: dict = field(default_factory=dict)  # threshold, max_rounds, max_cost_usd, ...
    targets: Targets = field(default_factory=Targets)  # style / subject / pose (see targets.py)


def _pairs(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
    return out


def load_job(folder: Path) -> Job:
    spec = json.loads((folder / "job.json").read_text(encoding="utf-8")) if (folder / "job.json").exists() else {}
    settings = _pairs(folder / "settings.txt")

    targets = from_spec(spec, folder)
    ref_name = spec.get("reference") or settings.pop("reference_image", None)
    reference = folder / ref_name if ref_name else targets.primary() or next(
        (p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXTS), None)
    if not reference or not reference.exists():
        raise FileNotFoundError(f"No reference image in {folder} (an automatic run needs at least one image: "
                                "style, subject or pose)")

    def text(key: str, file: str) -> str:
        if key in spec:
            return spec[key]
        f = folder / file
        return f.read_text(encoding="utf-8").strip() if f.exists() else ""

    settings.update(spec.get("settings", {}))
    overrides = {k: v for k, v in spec.items()
                 if k not in ("prompt", "negative", "description", "reference", "settings", *ROLES)}
    return Job(folder.name, folder, reference, text("prompt", "positive.txt"), text("negative", "negative.txt"),
               text("description", "description.txt"), settings, overrides, targets)


def _pid_alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        return True  # can't tell: assume the other process is still working on it


QUEUED_FILE = "queued.txt"  # enqueue timestamp, keeps the queue first-in first-out
RUNNING_FILE = "running.txt"  # present while a runner (any process) works on the job
STATUSES = ("pending", "done", "review", "failed", "removed")


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9 _.-]+", "_", name).strip(" ._")
    return cleaned or "job"


class Queue:
    def __init__(self, root: Path):
        self.root = root
        for sub in STATUSES:
            (root / sub).mkdir(parents=True, exist_ok=True)

    def list(self, status: str = "pending") -> list[Path]:
        folders = [p for p in (self.root / status).iterdir() if p.is_dir()]

        def key(p: Path) -> float:
            try:
                return float((p / QUEUED_FILE).read_text())
            except (OSError, ValueError):
                return p.stat().st_mtime
        return sorted(folders, key=key)

    def pending(self) -> list[Path]:
        return self.list("pending")

    def _new_folder(self, name: str) -> Path:
        base = safe_name(name)
        taken = {p.name for s in STATUSES for p in (self.root / s).iterdir()}
        n, candidate = 2, base
        while candidate in taken:
            candidate, n = f"{base}_{n}", n + 1
        folder = self.root / "pending" / candidate
        folder.mkdir()
        return folder

    def _stamp(self, folder: Path) -> Path:
        (folder / QUEUED_FILE).write_text(str(time.time()))
        return folder

    def add(self, name: str, reference_name: str | None, reference_bytes: bytes | None, spec: dict,
            target_images: dict[str, tuple[str, bytes]] | None = None) -> Path:
        """target_images: {role: (file name, bytes)} for the style/subject/pose images; each
        is saved as <role><ext> and named in spec[role]["image"]. With no reference image
        given, the subject's (else style's, else pose's) is the job's reference."""
        folder = self._new_folder(name)
        spec = dict(spec)
        for role, (fname, data) in (target_images or {}).items():
            if role not in ROLES:
                continue
            f = f"{role}{Path(fname).suffix.lower() or '.png'}"
            (folder / f).write_bytes(data)
            spec[role] = {**(spec.get(role) or {}), "image": f}
        if reference_bytes is not None:
            ref = safe_name(Path(reference_name or "reference.png").name) or "reference.png"
            (folder / ref).write_bytes(reference_bytes)
            spec["reference"] = ref
        else:
            ref = next((spec[r]["image"] for r in ("subject", "style", "pose")
                        if isinstance(spec.get(r), dict) and spec[r].get("image")), None)
            if not ref:
                shutil.rmtree(folder, ignore_errors=True)
                raise ValueError("an automatic run needs at least one image: style, subject or pose")
            spec["reference"] = ref
        (folder / "job.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        return self._stamp(folder)

    def import_folder(self, source: Path) -> Path:
        """Queue a prompt-generator saved prompt as-is (copy)."""
        load_job(source)  # validate first: has a reference image
        folder = self._new_folder(source.name)
        shutil.copytree(source, folder, dirs_exist_ok=True)
        return self._stamp(folder)

    def move(self, folder: Path, status: str) -> Path:
        dest = self.root / status / folder.name
        n = 2
        while dest.exists():  # e.g. a folder dropped into pending/ under a finished job's name
            dest, n = self.root / status / f"{folder.name}_{n}", n + 1
        dest = Path(shutil.move(str(folder), str(dest)))
        return self._stamp(dest) if status == "pending" else dest

    def finish(self, folder: Path, status: str) -> Path | None:
        """Move a processed job out of the queue; None if it's already gone."""
        (folder / RUNNING_FILE).unlink(missing_ok=True)
        if not folder.is_dir():
            return None
        return self.move(folder, status)

    @staticmethod
    def is_running(folder: Path) -> bool:
        """True while a live Ouroboros process works on the job. A marker left by a
        process that has since exited (a crash, a closed window) doesn't count."""
        marker = folder / RUNNING_FILE
        try:
            pid = int(marker.read_text().strip() or 0)
        except (OSError, ValueError):
            return False
        return pid == os.getpid() or _pid_alive(pid)

    @staticmethod
    def mark_running(folder: Path, running: bool) -> None:
        marker = folder / RUNNING_FILE
        if running:
            marker.write_text(str(os.getpid()))
        else:
            marker.unlink(missing_ok=True)
