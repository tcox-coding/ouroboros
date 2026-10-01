import pytest
from PIL import Image

import ouroboros.autofix as af
from ouroboros.comfy import Cancelled
from ouroboros.params import GenParams


def issue(what, region, sev, method="inpaint"):
    return {"what": what, "region": region, "kind": "anatomy", "severity": sev, "method": method,
            "add": [], "negative_add": []}


def test_settings_ignore_empty_values():
    s = af.settings({"autofix": {"max_rounds": None, "denoise": "", "fixes_per_round": 3}})
    assert s["max_rounds"] == af.DEFAULTS["max_rounds"] and s["denoise"] == af.DEFAULTS["denoise"]
    assert s["fixes_per_round"] == 3


def test_carry_forward_keeps_untouched_flaws_the_reviewer_forgot():
    hands, badge = issue("six fingers", "right hand (viewer's right)", 3), issue("garbled text", "chest badge", 1)
    after = carry_forward_result = af.carry_forward([issue("simplified badge", "chest badges", 1)], [hands, badge], [badge])
    regions = [i["region"] for i in carry_forward_result]
    assert "right hand (viewer's right)" in regions and "chest badge" not in regions
    assert after[0]["severity"] == 3


def test_failed_repairs_are_remembered_by_method_and_region():
    a = issue("six fingers", "left hand", 3, "hands")
    b = issue("an extra finger", "left hand (viewer's left)", 3, "hands")  # same flaw, reworded
    tried = [af._attempt_key(a)]
    alt = af._next_try(b, tried)
    assert alt["method"] == "inpaint"
    assert af._next_try(b, tried + [af._attempt_key(alt)]) is None


class FakeBackend:
    """Inspect returns `found`; each review returns the next verdict."""

    def __init__(self, found, verdicts):
        self.found, self.verdicts, self.calls = found, list(verdicts), []

    def complete(self, instructions, parts, schema, name, max_side):
        self.calls.append(name)
        if name == "autofix_inspect":
            return {"issues": self.found, "notes": ""}, 0.001, 0
        return self.verdicts.pop(0), 0.001, 0


@pytest.fixture
def image(tmp_path):
    p = tmp_path / "img.png"
    Image.new("RGB", (64, 64), "white").save(p)
    return p


def run(image, backend, monkeypatch, tmp_path, **kw):
    n = {"i": 0}

    def fake_apply(issue, img, *a, **k):
        n["i"] += 1
        out = tmp_path / f"fix{n['i']}.png"
        Image.new("RGB", (64, 64), "gray").save(out)
        return out
    monkeypatch.setattr(af, "_apply", fake_apply)
    return af.autofix(image, GenParams(positive="1girl"), backend=backend, comfy=None, flows=None,
                      cfg={"autofix": {"max_rounds": 2}}, checkpoint=None, positive="1girl", out_dir=tmp_path / "out",
                      upload=lambda p: p.name, log=lambda m: None, **kw)


def test_clean_image_changes_nothing(image, monkeypatch, tmp_path):
    res = run(image, FakeBackend([], []), monkeypatch, tmp_path)
    assert res["image"] is None and res["rounds"] == []


def test_better_review_with_less_wrong_is_kept(image, monkeypatch, tmp_path):
    found = [issue("six fingers", "left hand", 3)]
    res = run(image, FakeBackend(found, [{"fixed": [True], "new_problems": "", "better": True, "issues": [], "notes": ""}]),
              monkeypatch, tmp_path)
    assert res["image"] is not None and res["rounds"][0]["kept"]


def test_better_but_heavier_review_is_discarded(image, monkeypatch, tmp_path):
    found = [issue("thin waist", "waist", 2)]
    worse = {"fixed": [True], "new_problems": "ghost hand", "better": True,
             "issues": [issue("ghosted hand", "left hand", 3)], "notes": ""}
    res = run(image, FakeBackend(found, [worse, worse]), monkeypatch, tmp_path)
    assert res["image"] is None and not any(r["kept"] for r in res["rounds"])


def test_cancel_stops_it(image, monkeypatch, tmp_path):
    with pytest.raises(Cancelled):
        run(image, FakeBackend([issue("x", "hand", 3)], []), monkeypatch, tmp_path, should_stop=lambda: True)
