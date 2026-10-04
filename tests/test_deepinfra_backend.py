"""DeepInfra models differ in which JSON modes they take; the backend steps down until one works."""
import json

from ouroboros import backends


class Resp:
    def __init__(self, status, text="", data=None):
        self.status_code, self.text, self._data = status, text, data

    def json(self):
        return self._data


def ok(answer):
    return Resp(200, data={"choices": [{"message": {"content": json.dumps(answer)}, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 10, "completion_tokens": 5, "estimated_cost": 0.0001}})


def run(monkeypatch, replies):
    sent = []

    def post(url, json=None, headers=None, timeout=None):
        sent.append(json.get("response_format", {}).get("type"))
        return replies.pop(0)
    monkeypatch.setattr(backends.requests, "post", post)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")
    b = backends.DeepInfraBackend({"model": "x/y", "retries": 0})
    data, cost, _ = b.complete("Reply with JSON.", [{"text": "hi"}], {"type": "object"}, "t", 512)
    return data, sent


def test_a_405_for_the_schema_falls_back_to_json_mode(monkeypatch):
    data, sent = run(monkeypatch, [Resp(405, '{"error":"json_schema response format is not supported"}'),
                                   ok({"a": 1})])
    assert data == {"a": 1} and sent == ["json_schema", "json_object"]


def test_a_model_that_takes_neither_mode_is_asked_without_one(monkeypatch):
    bad = '{"error":"The parameter `response_format.type` specified in the request are not valid"}'
    data, sent = run(monkeypatch, [Resp(500, bad), Resp(500, bad), ok({"a": 2})])
    assert data == {"a": 2} and sent == ["json_schema", "json_object", None]


def test_a_bare_500_twice_for_a_schema_steps_down_to_json_mode(monkeypatch):
    monkeypatch.setattr(backends.time, "sleep", lambda s: None)
    err = '{"error":{"code":"InternalServiceError","message":"The service encountered an unexpected internal error."}}'
    data, sent = run(monkeypatch, [Resp(500, err), Resp(500, err), ok({"a": 3})])
    assert data == {"a": 3} and sent == ["json_schema", "json_schema", "json_object"]


def test_too_many_images_keeps_the_first_and_notes_the_rest(monkeypatch, tmp_path):
    from PIL import Image
    paths = []
    for i in range(3):
        paths.append(tmp_path / f"{i}.png")
        Image.new("RGB", (8, 8)).save(paths[-1])
    sent = []

    def post(url, json=None, headers=None, timeout=None):
        content = json["messages"][1]["content"]
        sent.append(sum(c["type"] == "image_url" for c in content))
        return (Resp(400, '{"error":{"message":"Too many images in request: 3 > 2"}}') if len(sent) == 1
                else ok({"a": 4}))
    monkeypatch.setattr(backends.requests, "post", post)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")
    monkeypatch.setattr(backends, "_IMAGE_LIMITS", {})
    b = backends.DeepInfraBackend({"model": "lim/two", "retries": 0})
    data, _c, _t = b.complete("x", [{"image": p} for p in paths], {"type": "object"}, "t", 64)
    assert data == {"a": 4} and sent == [3, 2]
