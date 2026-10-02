"""The judge's answer format, and the web server's API (read-only endpoints and path safety,
against this project folder; nothing is changed)."""
import json
import socket
import threading
import urllib.error
import urllib.request

import pytest

from ouroboros.judge import build_schema


def focus_enum(**kw):
    return build_schema(["identity"], ["euler"], ["karras"], 1, **kw)["properties"]["edit"]["properties"]["focus"]["enum"]


def test_focus_choices_follow_lora_switching():
    assert focus_enum() == ["prompt", "settings"]
    assert focus_enum(lora_stems=["a"], lora_switch=False) == ["prompt", "settings", "lora_weights"]
    assert focus_enum(lora_stems=["a"], lora_switch=True) == ["prompt", "settings", "lora_weights", "loras"]
    edit = build_schema(["identity"], ["euler"], ["karras"], 1, lora_stems=["a"])["properties"]["edit"]
    assert "focus" in edit["required"] and "loras" in edit["required"]


def test_review_each_runs_calls_in_parallel_and_keeps_order(monkeypatch):
    import time
    from ouroboros.judge import Judge, Review
    j = Judge.__new__(Judge)
    j.parallel, j.base_rubric = 4, {"identity": {"weight": 1, "anchors": ""}}
    seen = []

    def fake_review(reference, goal, cur, notes, cands, *a, **k):
        time.sleep(0.2)
        seen.append(cur)
        return Review([float(cur)], 0, "", {}, {"candidates": [{"index": 0, "differences": []}]}, 0.0)
    monkeypatch.setattr(j, "review", fake_review, raising=False)
    t = time.time()
    r = j.review_each("ref", "goal", ["10", "30", "20", "5"], "", ["a", "b", "c", "d"])
    assert time.time() - t < 0.6 and r.scores == [10.0, 30.0, 20.0, 5.0] and r.best_index == 1


@pytest.fixture(scope="module")
def server():
    from ouroboros import server as srv
    from http.server import ThreadingHTTPServer
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    httpd = ThreadingHTTPServer(("127.0.0.1", port), srv.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()


def get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.mark.parametrize("path", ["/", "/api/state", "/api/settings", "/api/generate/estimate", "/api/poses",
                                  "/api/styles", "/api/characters", "/api/loras/catalog", "/static/icon.png"])
def test_pages_and_read_only_endpoints_answer(server, path):
    code, body = get(server, path)
    assert code == 200, body[:200]


def test_state_lists_history_queue_and_bin(server):
    state = json.loads(get(server, "/api/state")[1])
    assert {"runs", "generations", "bin", "queue"} <= set(state)


def test_settings_never_contain_keys(server):
    s = json.loads(get(server, "/api/settings")[1])
    assert "api_key" not in json.dumps(s).replace("api_keys", "") and "civitai_api_key" not in s["loras"]


@pytest.mark.parametrize("path", ["/files/config.json", "/files/../config.json", "/files/api_keys.json",
                                  "/thumbs/config.json", "/thumbs/runs/../api_keys.json", "/static/../config.json",
                                  "/api/loras/example?id=../x&n=0"])
def test_private_files_are_never_served(server, path):
    code, body = get(server, path)
    assert code in (403, 404) and b"api" not in body.lower().replace(b"api_keys", b"")


def test_bad_requests_are_refused_without_side_effects(server):
    assert post(server, "/api/generations/remove", {"id": "nope"})[0] >= 400
    assert post(server, "/api/runs/remove", {"run": "../config.json"})[0] >= 400
    assert post(server, "/api/runs/remove", {"run": "manual/../../config.json"})[0] >= 400
    assert post(server, "/api/autofix", {"run": "manual/does-not-exist", "images": ["image_01.png"]})[0] >= 400
    assert post(server, "/api/keys", {"name": "not-a-platform", "value": "x"})[0] >= 400
    assert post(server, "/api/styles/remove", {"name": "../config.json"})[0] >= 400
    assert post(server, "/api/characters/remove", {"name": "_removed"})[0] >= 400
    code, body = post(server, "/api/jobs", {"targets": {"style": {"library": "no_such_style_saved"}}})
    assert code == 400 and "no saved style" in body["error"]


@pytest.mark.parametrize("path", ["/files/runs/../config.json", "/files/runs/%2e%2e/api_keys.json",
                                  "/files/poses/../deepinfra_key.txt", "/files/cache/reruns/../../config.json"])
def test_dot_dot_inside_an_allowed_folder_is_refused(server, path):
    # Sent as-is: urllib and browsers would tidy "a/../b" away before it reached the server.
    import http.client
    host, port = server.removeprefix("http://").split(":")
    c = http.client.HTTPConnection(host, int(port), timeout=10)
    c.request("GET", path)
    r = c.getresponse()
    assert r.status == 403 and b"api" not in r.read().lower()


def test_removing_an_unknown_task_says_so(server):
    code, body = post(server, "/api/generations/remove", {"id": "nope"})
    assert code == 404 and "no longer" in body["error"]


def test_parallel_reviews_keep_the_jobs_llm_queue_label(monkeypatch):
    from ouroboros.judge import Judge, Review
    from ouroboros.llm_queue import current_label, set_label
    j = Judge.__new__(Judge)
    j.parallel, j.base_rubric = 3, {"identity": {"weight": 1, "anchors": ""}}
    seen = []

    def fake_review(reference, goal, cur, notes, cands, *a, **k):
        seen.append(current_label())
        return Review([1.0], 0, "", {}, {"candidates": [{"index": 0, "differences": []}]}, 0.0)
    monkeypatch.setattr(j, "review", fake_review, raising=False)
    set_label("job-a")
    try:
        j.review_each("ref", "goal", "p", "", ["a", "b", "c"])
    finally:
        set_label(None)
    assert seen == ["job-a"] * 3
