"""DeepInfra models differ in which JSON modes they take; the backend steps down until one works."""
import json
import pytest

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


def test_too_many_images_uses_a_sheet_instead_of_dropping_images(monkeypatch, tmp_path):
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
    assert data == {"a": 4} and sent == [3, 1]


def test_sheet_contains_every_panel_and_keeps_images_unmodified():
    from PIL import Image
    colors = [(230, 20, 20), (20, 230, 20), (20, 20, 230)]
    images = [Image.new("RGB", (64, 96), color) for color in colors]
    parts = [part for i, image in enumerate(images) for part in ({"text": f"Candidate {i}"}, {"image": image})]
    sheet, labels = backends.labelled_sheet(parts, 128)
    assert labels == [f"Image {i + 1}: Candidate {i}" for i in range(3)]
    assert [sheet.getpixel(((i % 2) * 128 + 64, (i // 2) * 160 + 96)) for i in range(3)] == colors
    assert all(im.size == (64, 96) for im in images)


def test_billed_retries_are_included_even_when_the_final_attempt_fails(monkeypatch):
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")
    bad = ok({})
    bad._data["choices"][0]["message"]["content"] = "not JSON"
    replies = [bad, ok({"ok": True})]
    monkeypatch.setattr(backends.requests, "post", lambda *a, **k: replies.pop(0))
    b = backends.DeepInfraBackend({"model": "test", "retries": 1})
    assert b.complete("", [], {}, "t", 512)[1] == pytest.approx(0.0002)
    replies[:] = [bad, bad]
    with pytest.raises(backends.CompletionError) as exc:
        b.complete("", [], {}, "t", 512)
    assert exc.value.cost_usd == pytest.approx(0.0002)
    assert exc.value.prompt_tokens == 20


def test_truncated_attempts_count_towards_cost(monkeypatch):
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")
    truncated = ok({})
    truncated._data["choices"][0]["finish_reason"] = "length"
    replies = [truncated, ok({"ok": True})]
    monkeypatch.setattr(backends.requests, "post", lambda *a, **k: replies.pop(0))
    b = backends.DeepInfraBackend({"model": "test", "retries": 0, "max_tokens": 1024})
    assert b.complete("", [], {}, "t", 512)[1] == pytest.approx(0.0002)


def test_an_answer_cut_off_without_reasoning_is_a_loop_retried_with_a_penalty_not_more_room(monkeypatch):
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")
    looping = ok({})
    looping._data["choices"][0].update(finish_reason="length", message={"content": "masterpiece, " * 50})
    sent = []

    def post(url, json=None, headers=None, timeout=None):
        sent.append(dict(json))
        return replies.pop(0)
    monkeypatch.setattr(backends.requests, "post", post)
    replies = [looping, ok({"ok": True})]
    b = backends.DeepInfraBackend({"model": "test", "retries": 0, "max_tokens": 4096, "temperature": 0})
    assert b.complete("", [], {}, "t", 512)[0] == {"ok": True}
    assert sent[1]["max_tokens"] == 4096 and sent[1]["frequency_penalty"] == 0.5 and sent[1]["temperature"] == 0.4
    replies[:] = [looping, looping]  # still looping: one nudged retry, then the error says so
    with pytest.raises(backends.CompletionError, match="repeating itself"):
        b.complete("", [], {}, "t", 512)


def test_an_answer_cut_off_while_reasoning_gets_more_room(monkeypatch):
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")
    thinking = ok({})
    thinking._data["choices"][0].update(finish_reason="length", message={"content": "", "reasoning_content": "hmm"})
    sent = []

    def post(url, json=None, headers=None, timeout=None):
        sent.append(dict(json))
        return replies.pop(0)
    monkeypatch.setattr(backends.requests, "post", post)
    replies = [thinking, ok({"ok": True})]
    b = backends.DeepInfraBackend({"model": "test", "retries": 0, "max_tokens": 4096})
    b.complete("", [], {}, "t", 512)
    assert sent[1]["max_tokens"] == 8192 and "frequency_penalty" not in sent[1]


def test_a_capped_call_has_a_short_answer_limit_and_a_mild_penalty_and_the_settings_come_back(monkeypatch):
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")
    sent = []

    def post(url, json=None, headers=None, timeout=None):
        sent.append(dict(json))
        return ok({"ok": True})
    monkeypatch.setattr(backends.requests, "post", post)
    b = backends.DeepInfraBackend({"model": "x", "retries": 0, "max_tokens": 4096, "temperature": 0})
    with backends.capped_output(b, 1536):
        b.complete("", [], {}, "t", 512)
    assert sent[0]["max_tokens"] == 1536 and sent[0]["frequency_penalty"] == 0.3
    b.complete("", [], {}, "t", 512)
    assert sent[1]["max_tokens"] == 4096 and "frequency_penalty" not in sent[1]  # restored after
