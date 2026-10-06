"""Answers in plain JSON mode aren't held to the schema: the judge must cope with bad shapes."""
from pathlib import Path

import pytest

from ouroboros.judge import Judge
from ouroboros.runner import load_config

IMG = Path(__file__).resolve().parents[1] / "eval" / "images" / "s_good_a.png"


class Backend:
    context_window = 100000

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), 0

    def complete(self, instructions, parts, schema, name, max_side):
        self.calls += 1
        return self.answers.pop(0), 0.001, 10


def judge(backend):
    cfg = {**load_config()["judge"], "backend": "ollama", "ollama": {"url": "http://x", "model": "m"},
           "confirm_model": ""}
    j = Judge(cfg, ["euler"], ["karras"])
    j.backend = backend
    return j


GOOD = {"candidates": [{"index": 0, "scores": {"identity": "8", "composition": 7}, "differences": []}],
        "best_index": 0, "diagnosis": "ok", "edit": {"focus": "prompt"}}


def test_string_candidates_are_asked_again():
    b = Backend({"candidates": ["looks fine"], "edit": "none"}, GOOD)
    rv = judge(b).review(IMG, "goal", "txt2img", "", [IMG])
    assert b.calls == 2 and rv.scores[0] > 0 and rv.cost_usd == pytest.approx(0.002)


def test_two_unusable_answers_fail_clearly():
    b = Backend({"candidates": []}, {"oops": 1})
    with pytest.raises(RuntimeError, match="scored no candidate") as exc:
        judge(b).review(IMG, "goal", "txt2img", "", [IMG])
    assert exc.value.cost_usd == pytest.approx(0.002)
