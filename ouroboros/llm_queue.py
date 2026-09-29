"""One queue for every LLM call, so several jobs can be in progress at once.

A local LLM answers one request at a time, and a large model takes a while. Rather than
leaving the GPU idle while a job waits for its answer, several jobs run side by side
(runner, queue.parallel_jobs): while one job's candidates are being judged, another's
render in ComfyUI. Their LLM calls (judging, prompt writing, LoRA picking, summaries)
all go through this gate, first come first served, so nobody jumps the line. status()
shows who is being served and who is waiting, for the UI.

How many calls the gate lets through at once is queue.llm_parallel: 1 for a single
local GPU, which can only answer one request at a time, and more for a hosted API
(DeepInfra), where waiting in line just adds latency for no reason. Admission stays
first come first served whatever the number.
"""

from __future__ import annotations

import threading
import time

_local = threading.local()


def set_label(label: str | None) -> None:
    """Name the job the current thread works for (shown in the LLM queue status)."""
    _local.label = label


class LLMGate:
    def __init__(self, capacity: int = 1):
        self._cv = threading.Condition()
        self._next_ticket = 0
        self._admit = 0                       # the next ticket allowed to start
        self._waiting: dict[int, str] = {}
        self._busy: dict[int, tuple[str, float]] = {}
        self._capacity = max(1, int(capacity))
        self.calls = 0
        self.busy_seconds = 0.0

    def set_capacity(self, capacity: int) -> None:
        """Set how many calls may be in flight (read from the config at each start)."""
        with self._cv:
            self._capacity = max(1, int(capacity))
            self._cv.notify_all()

    def run(self, fn, label: str | None = None):
        label = label or getattr(_local, "label", None) or "?"
        with self._cv:
            ticket = self._next_ticket
            self._next_ticket += 1
            self._waiting[ticket] = label
            while ticket != self._admit or len(self._busy) >= self._capacity:
                self._cv.wait()
            del self._waiting[ticket]
            self._admit += 1
            self._busy[ticket] = (label, time.monotonic())
            self._cv.notify_all()             # the next in line may fit too
        try:
            return fn()
        finally:
            with self._cv:
                self.calls += 1
                self.busy_seconds += time.monotonic() - self._busy.pop(ticket)[1]
                self._cv.notify_all()

    def status(self) -> dict:
        with self._cv:
            busy = [self._busy[t] for t in sorted(self._busy)]
            return {"busy": ", ".join(label for label, _ in busy) or None,
                    "busy_for": round(time.monotonic() - min(s for _, s in busy)) if busy else 0,
                    "in_flight": len(busy), "capacity": self._capacity,
                    "waiting": [self._waiting[t] for t in sorted(self._waiting)],
                    "calls": self.calls, "busy_seconds": round(self.busy_seconds)}


class GatedBackend:
    """A judge backend whose complete() waits its turn at the gate."""

    def __init__(self, backend, gate: LLMGate):
        self._backend, self._gate = backend, gate

    def complete(self, *args, **kwargs):
        return self._gate.run(lambda: self._backend.complete(*args, **kwargs))

    def __getattr__(self, name):
        return getattr(self._backend, name)
