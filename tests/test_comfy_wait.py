"""ComfyClient.wait: a long render isn't given up on while ComfyUI still has it."""
import pytest

from ouroboros import comfy


class Resp:
    def __init__(self, data):
        self.data = data

    def json(self):
        return self.data


def fake_comfy(monkeypatch, finishes_after: int, in_queue: bool):
    calls = {"history": 0}

    def get(url, timeout=None):
        if "/history/" in url:
            calls["history"] += 1
            done = calls["history"] > finishes_after
            return Resp({"p1": {"status": {"completed": True}, "outputs": {}}} if done else {})
        if url.endswith("/queue"):
            return Resp({"queue_running": [[0, "p1", {}]] if in_queue else [], "queue_pending": []})
        raise AssertionError(url)
    monkeypatch.setattr(comfy.requests, "get", get)
    clock = {"t": 0.0}
    monkeypatch.setattr(comfy.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(comfy.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    return calls


def test_a_render_still_running_past_the_timeout_is_waited_for(monkeypatch):
    fake_comfy(monkeypatch, finishes_after=50, in_queue=True)  # 50 s of polling against a 10 s timeout
    c = comfy.ComfyClient("http://x", timeout=10)
    assert c.wait(["p1"], poll=1.0)["p1"]["status"]["completed"]


def test_a_job_comfyui_no_longer_has_still_times_out(monkeypatch):
    fake_comfy(monkeypatch, finishes_after=10**6, in_queue=False)
    c = comfy.ComfyClient("http://x", timeout=10)
    with pytest.raises(comfy.ComfyError, match="Timed out"):
        c.wait(["p1"], poll=1.0)
