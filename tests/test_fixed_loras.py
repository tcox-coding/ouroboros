"""LoRAs chosen by hand (Home -> Run automatically): the loop uses exactly those, every round."""
import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from fakes import FakeComfy
from ouroboros.jobs import Queue, load_job
from ouroboros.judge import Review

ROOT = Path(__file__).resolve().parent.parent
JAB = "Pony\\styles\\artists\\jab\\Jabstyle_PNYV1.5.safetensors"
POSE_A = "Pony\\concepts\\a\\poseA.safetensors"
FLAT = "Pony\\styles\\flat\\FlatColor.safetensors"


class Comfy(FakeComfy):
    def choices(self, node_type, input_name):
        return [JAB, POSE_A, FLAT, "Pony\\concepts\\b\\poseB.safetensors"]


class Judge:
    """Each round's edit tries to change the LoRA set: swap one in, then drop one."""
    context_window = 8000
    backend = None

    def __init__(self):
        self.calls, self.menus = 0, []

    def rubric(self, penalize_extras):
        return {"identity": {"weight": 1, "anchors": "x"}}

    def _review(self, cands, loras):
        self.calls += 1
        self.menus.append(loras)
        edits = [{"focus": "loras", "loras": [{"lora": "FlatColor", "strength": 0.7}], "diagnosis": "x"},
                 {"focus": "lora_weights", "loras": [{"lora": "Jabstyle_PNYV1.5", "strength": 0},
                                                     {"lora": "poseA", "strength": 0.9}]}]
        return Review([40.0] * len(cands), 0, "needs work", edits[(self.calls - 1) % 2], {}, 0.0)

    def review_each(self, reference, goal, current, notes, cands, modes, rules, rubric, extra, loras=None, *a, **k):
        return self._review(cands, loras)

    def review(self, reference, goal, current, notes, cands, modes, rules, rubric, extra, loras=None, *a, **k):
        return self._review(cands, loras)

    def confirm(self, *a, **k):
        return Review([40.0], 0, "", {}, {}, 0.0)

    def summarize(self, previous, entries):
        return previous, 0.0


@pytest.fixture
def flows(tmp_path, monkeypatch):
    import ouroboros.workflow as wf
    (tmp_path / "workflows").mkdir()
    for f in ("example_workflow.json", "nodes.example.json"):
        shutil.copy(ROOT / "workflows" / f, tmp_path / "workflows" / f)
    monkeypatch.setattr(wf, "_lora_root", lambda: None)
    return wf.Workflows(tmp_path / "workflows")


class Library:
    def __init__(self, index):
        self._index = index

    def index(self):
        return self._index

    def card(self, rec, detail=True):
        return rec.get("title", "")

    def record_results(self, results):
        pass


def run(tmp_path, flows, settings, overrides):
    from ouroboros.loop import run_job
    cfg = json.loads((ROOT / "config.example.json").read_text())
    cfg["loop"].update(max_rounds=3, threshold=99, hand_refine=False, auto_fix=False, upscale=False,
                       ai_settings=False, batch_start=2, batch_end=2, plateau_rounds=10)
    cfg["loras"]["mode"] = "auto"  # what the job overrides
    Image.new("RGB", (64, 96), "red").save(tmp_path / "me.png")
    folder = Queue(tmp_path / "jobs").add("x", None, None, {"prompt": "1girl", "settings": settings, **overrides},
                                          {"subject": ("me.png", (tmp_path / "me.png").read_bytes())})
    FakeComfy.uploads, FakeComfy.graphs = [], []
    index = {n: {"trigger_words": [], "title": Path(n).stem, "examples": []} for n in (JAB, POSE_A, FLAT)}
    judge = Judge()
    run_job(load_job(folder), cfg, Comfy("x"), flows, judge, ["euler"], ["normal"], tmp_path / "runs",
            report=lambda e: None, library=Library(index))
    # ComfyUI is sent "/" paths (workflow.build); the library names use "\\".
    sets = [sorted(v["lora"].replace("/", "\\") for k, v in g["23:5"]["inputs"].items() if k.startswith("lora_") and v.get("on"))
            for g in FakeComfy.graphs]
    return sets, judge


def test_selected_loras_are_the_only_ones_in_every_round(tmp_path, flows, monkeypatch):
    import ouroboros.loop as loop
    monkeypatch.setattr(loop, "pick_loras", lambda *a, **k: pytest.fail("fixed LoRAs must not be re-picked"))
    sets, judge = run(tmp_path, flows, {"loras": [{"name": JAB, "strength": 0.8}, {"name": POSE_A, "strength": 0.6}]},
                      {"lora_mode": "fixed"})
    assert judge.calls >= 2 and sets                       # several rounds, each tried to change the set
    assert all(s == sorted([JAB, POSE_A]) for s in sets), sets
    assert all(m is None or (m["switch"] is False and set(m["stems"]) == {"Jabstyle_PNYV1.5", "poseA"})
               for m in judge.menus)


def test_strengths_of_selected_loras_may_still_be_tuned(tmp_path, flows):
    sets, _ = run(tmp_path, flows, {"loras": [{"name": POSE_A, "strength": 0.6}]}, {"lora_mode": "fixed"})
    strengths = {v["strength"] for g in FakeComfy.graphs for k, v in g["23:5"]["inputs"].items()
                 if k.startswith("lora_") and v.get("on")}
    assert all(s == [POSE_A] for s in sets) and len(strengths) > 1


def test_without_fixed_mode_the_judge_may_switch(tmp_path, flows, monkeypatch):
    import ouroboros.loop as loop
    monkeypatch.setattr(loop, "pick_loras", lambda *a, **k: {"picks": [(JAB, 0.8, "")], "alternatives": [],
                                                             "shortlist": [JAB, FLAT]})
    sets, _ = run(tmp_path, flows, {}, {})
    assert any(FLAT in s for s in sets)  # the control: in "auto" the swap does happen


def test_write_prompts_is_told_about_the_selected_loras(monkeypatch):
    import ouroboros.backends as backends
    import ouroboros.runner as runner
    from ouroboros import server
    seen = {}

    class B:
        def complete(self, instructions, parts, schema, name, max_side):
            seen["text"] = " ".join(p.get("text", "") for p in parts)
            return {"positive": "jabstyle, 1girl", "negative": "", "dropped": [], "notes": ""}, 0.0, 0
    lib = Library({JAB: {"trigger_words": ["Jabstyle"], "title": "Jab Style", "examples": [], "type": "style",
                         "description": "thick ink lines"}})
    monkeypatch.setattr(backends, "make_backend", lambda cfg: B())
    monkeypatch.setattr(server, "lora_library", lambda cfg: lib)
    monkeypatch.setattr(runner, "lora_library", lambda cfg: lib, raising=False)
    server.preview_prompt({"description": "a knight", "loras": [{"name": JAB, "strength": 0.7}]})
    assert "Jab Style" in seen["text"] and "trigger word: Jabstyle" in seen["text"] and "strength 0.7" in seen["text"]
