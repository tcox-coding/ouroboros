import json
import threading
import time

import pytest

import ouroboros.generate as gen
from ouroboros.comfy import Cancelled


@pytest.fixture
def queue(tmp_path, monkeypatch):
    """The generation queue with _run replaced: each task blocks until released. One
    worker serves every test, as in the app; each test starts and ends with it idle."""
    with gen._LOCK:
        gen.JOBS.clear(); gen._REQS.clear(); gen._ORDER.clear()
    started, release, order = {}, {}, []

    def fake_run(gen_id, cfg, req, root, rel_url, stop):
        order.append(req["tag"])
        started[req["tag"]].set()
        while not release[req["tag"]].wait(0.05):
            if stop():
                raise Cancelled()
        gen._set(gen_id, status="done", state="done", finished=time.time())

    monkeypatch.setattr(gen, "_run", fake_run)
    gen.configure(root=tmp_path, rel_url=lambda p: str(p), load_config=lambda: {}, prepare=lambda: None,
                  fix_target=lambda run: {"dir": tmp_path})

    def add(tag):
        started[tag], release[tag] = threading.Event(), threading.Event()
        return gen.start({"tag": tag, "positive": tag})
    yield add, started, release, order
    for e in release.values():
        e.set()
    assert wait_for(lambda: not gen._ORDER), "the queue didn't drain"


def wait_for(cond, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def status(i):
    return gen.status(i)["status"]


def test_runs_in_order_one_at_a_time(queue):
    add, started, release, order = queue
    add("a"); b = add("b")
    assert started["a"].wait(2) and status(b) == "waiting"
    assert [t["position"] for t in gen.queue_status() if t["status"] == "waiting"] == [1]
    release["a"].set()
    assert started["b"].wait(2)
    release["b"].set()
    assert wait_for(lambda: status(b) == "done") and order == ["a", "b"]


def test_removing_a_waiting_task_skips_it(queue):
    add, started, release, order = queue
    add("a"); b, c = add("b"), add("c")
    assert started["a"].wait(2)
    assert gen.remove(b)["status"] == "cancelled"
    release["a"].set(); release["c"].set()
    assert wait_for(lambda: status(c) == "done") and order == ["a", "c"]


def test_cancelling_the_running_task(queue):
    add, started, release, order = queue
    a = add("a")
    assert started["a"].wait(2)
    gen.remove(a)
    assert wait_for(lambda: status(a) == "cancelled")


def test_clear_finished_hides_them(queue):
    add, started, release, order = queue
    a = add("a"); release["a"].set()
    assert wait_for(lambda: status(a) == "done")
    gen.clear_finished()
    assert all(t["id"] != a for t in gen.queue_status())


def test_unknown_task_cannot_be_removed(queue):
    with pytest.raises(KeyError):
        gen.remove("nope")


def manual_run(tmp_path, name="20260101-000000_abc", **rec):
    d = tmp_path / "runs" / "manual" / name
    d.mkdir(parents=True)
    (d / "reference.png").write_bytes(b"\x89PNG fake")
    (d / "run.json").write_text(json.dumps({"positive": "1girl, red", "negative": "lowres", "size": [832, 1216],
                                            "loras": [["Pony\\X.safetensors", 0.7]], "images": ["image_01.png"],
                                            "request": {"steps": 20, "cfg": 6, "seed": 42, "size": "auto",
                                                        "description": "a girl"}, **rec}))
    return d


def test_rerun_request_reproduces_the_entry(tmp_path):
    manual_run(tmp_path)
    req = gen.rerun_request(tmp_path, "manual/20260101-000000_abc")
    assert req["positive"] == "1girl, red" and req["description"] == "" and req["seed"] == 42
    assert req["size"] == [832, 1216] and req["loras"] == [{"name": "Pony\\X.safetensors", "strength": 0.7}]
    assert req["style_b64"].startswith("data:image/png;base64,") and req["_rerun"]
    with pytest.raises(FileNotFoundError):
        gen.rerun_request(tmp_path, "manual/../../etc")


def test_manual_fix_target_reads_the_saved_settings(tmp_path):
    manual_run(tmp_path)
    t = gen.manual_fix_target(tmp_path, "manual/20260101-000000_abc")
    assert t["params"].steps == 20 and t["params"].loras == (("Pony\\X.safetensors", 0.7),)


def test_estimate_learns_from_recorded_timings(tmp_path):
    manual_run(tmp_path, timing={"render_s": 4.0 + 0.5 * 20 * 1.0 * 2, "steps": 20, "megapixels": 1.0, "images": 2})
    est = gen.estimate(tmp_path)
    assert est["samples"] == 1 and abs(est["rate"] - 0.5) < 0.01
    assert gen.estimate(tmp_path / "empty")["rate"] == gen.DEFAULT_RATE


def test_one_failed_autofix_does_not_stop_the_others(tmp_path, monkeypatch):
    import ouroboros.autofix as af
    calls = []

    def flaky(src, *a, **k):
        calls.append(src.name)
        if src.name == "image_01.png":
            raise RuntimeError("LLM timed out")
        return {"fixed": None, "issues_found": [], "issues_left": [], "rounds": []}
    monkeypatch.setattr(af, "run_and_record", flaky)
    import ouroboros.backends as backends
    monkeypatch.setattr(backends, "make_backend", lambda cfg: None)
    gen.configure(root=tmp_path, rel_url=lambda p: str(p), load_config=lambda: {}, prepare=lambda: None,
                  fix_target=lambda run: {})
    gen.JOBS["t"] = {"id": "t"}
    out = gen._autofix_images("t", {"judge": {}}, None, None, tmp_path, [tmp_path / "image_01.png", tmp_path / "image_02.png"],
                              None, "", None, lambda: False)
    assert calls == ["image_01.png", "image_02.png"]
    assert "timed out" in out[0]["error"] and "error" not in out[1]


def test_history_entry_with_a_queued_autofix_cannot_be_deleted(monkeypatch):
    from ouroboros import server
    monkeypatch.setattr(gen, "busy_targets", lambda: {"manual/x"})
    monkeypatch.setattr(server, "ROOT", server.ROOT)
    import pytest as _p
    d = server.ROOT / "runs" / "manual" / "x"
    created = not d.exists()
    d.mkdir(parents=True, exist_ok=True)
    try:
        with _p.raises(RuntimeError, match="auto-fix"):
            server.remove_run("manual/x")
        assert d.exists()
    finally:
        if created:
            d.rmdir()


class FakeComfy:
    """Records uploads and graphs; every render returns one small PNG."""
    uploads: list = []
    graphs: list = []

    def __init__(self, url):
        pass

    def upload_image(self, path, subfolder="x"):
        from PIL import Image
        with Image.open(path) as im:
            FakeComfy.uploads.append((path.name, im.size, im.mode))
        return f"{subfolder}/{path.name}"

    def queue(self, graph):
        FakeComfy.graphs.append(graph)
        return "pid"

    def wait(self, ids, should_stop=None):
        return {"pid": {"outputs": {}}}

    def fetch_images(self, hist, node):
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, "PNG")
        return [buf.getvalue()]


@pytest.fixture
def home_run(tmp_path, monkeypatch):
    """_run against the example workflow and a fake ComfyUI."""
    import shutil
    from pathlib import Path
    import ouroboros.workflow as wf
    src = Path(__file__).resolve().parent.parent / "workflows"
    (tmp_path / "workflows").mkdir()
    for f in ("example_workflow.json", "nodes.example.json"):
        shutil.copy(src / f, tmp_path / "workflows" / f)
    monkeypatch.setattr(wf, "_lora_root", lambda: None)
    monkeypatch.setattr(gen, "ComfyClient", FakeComfy)
    FakeComfy.uploads, FakeComfy.graphs = [], []

    def run(**req):
        gen.JOBS["h"] = {"id": "h"}
        gen._run("h", {"comfy_url": "x", "judge": {}}, {"positive": "1girl", "steps": 20, **req}, tmp_path,
                 lambda p: str(p), lambda: False)
        assert gen.JOBS["h"]["status"] == "done", gen.JOBS["h"].get("error")
        return gen.JOBS["h"]
    return run


def b64_png(size, mode="RGBA"):
    import base64
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new(mode, size, (255, 0, 0, 0) if mode == "RGBA" else "red").save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_img2img_loads_the_reference_fitted_to_the_output_size(home_run):
    home_run(style_b64=b64_png((3000, 2000)), size=[832, 1216], mode="img2img_reference")
    name, size, mode = FakeComfy.uploads[0]
    assert name == "reference_input.png" and size == (832, 1216) and mode == "RGB"


def test_img2img_without_a_reference_renders_txt2img(home_run):
    job = home_run(mode="img2img_reference")
    assert job["params"].startswith("txt2img")


def test_library_pose_is_cropped_not_stretched_and_sets_auto_size(home_run, tmp_path, monkeypatch):
    import ouroboros.pose as pm
    d = tmp_path / "poses" / "wide"
    d.mkdir(parents=True)
    from PIL import Image
    Image.new("RGB", (1216, 832)).save(d / "source.png")
    body = [[0.5, 0.5], [0.05, 0.5]] + [None] * 16  # the centre, and a point near the left edge
    (d / "pose.json").write_text(json.dumps({"body": body, "size": [1216, 832]}))
    monkeypatch.setattr(pm, "has_body", lambda p, min_points=4: True)
    drawn = {}
    monkeypatch.setattr(pm, "render", lambda pose, size, **k: drawn.update(pose=pose, size=size) or Image.new("RGB", size))
    job = home_run(pose_library="wide", pose_control=True, size="auto")
    assert drawn["size"] == (1216, 832) and "1216x832" in job["params"]  # auto follows the pose's shape
    home_run(pose_library="wide", pose_control=True, size=[832, 1216])
    assert drawn["pose"]["body"][0] == pytest.approx([0.5, 0.5]) and drawn["pose"]["body"][1] is None  # cropped
