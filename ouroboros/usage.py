"""Per-operation billed usage, including retries and calls that never return an answer."""
from contextvars import ContextVar
from functools import wraps
import threading

_current = ContextVar("llm_usage", default=None)


class Meter:
    def __init__(self):
        self.cost, self.calls = 0.0, 0
        self.lock = threading.Lock()

    def add(self, cost):
        with self.lock:
            self.cost += cost
            self.calls += 1


def charge(cost):
    meter = _current.get()
    if meter is not None:
        meter.add(cost)


def total_or(fallback, initial=0.0):
    meter = _current.get()
    return initial + meter.cost if meter is not None and meter.calls else fallback


def scoped(fn):
    @wraps(fn)
    def run(*args, **kwargs):
        token = _current.set(Meter())
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            meter = _current.get()
            if meter.calls:
                e.cost_usd = meter.cost
                e.cost_is_total = True
            raise
        finally:
            _current.reset(token)
    return run
