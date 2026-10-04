"""The back and forth between Ouroboros and the real judge (see conftest.py; run with
OUROBOROS_LIVE=1). Each test sends what the app sends and checks that the answer is one the
app can use: the right shape, values within range, references to things that exist."""
import json
import shutil
from pathlib import Path

import pytest

from livecfg import EVAL, ROOT, SAMPLERS, SCHEDULERS

REF = EVAL / "soldier_reference.png"
GOOD, FLAWED, WRONG = EVAL / "s_good_a.png", EVAL / "s_tabs.png", EVAL / "d_wrong.png"
GOAL = "a stern woman with a short black bob in a red collared shirt and red trousers, flat colour cartoon"


# ---- the connection itself ------------------------------------------------------------

def test_a_structured_text_call_comes_back_as_json(backend):
    schema = {"type": "object", "properties": {"word": {"type": "string"}, "n": {"type": "integer"}},
              "required": ["word", "n"], "additionalProperties": False}
    data, cost, _tokens = backend.complete("Reply with JSON only.", [{"text": "Give any word and the number 7."}],
                                           schema, "live_text", 512)
    assert isinstance(data.get("word"), str) and data.get("n") == 7
    assert cost is None or cost >= 0


def test_the_model_accepts_images(backend):
    """Every judge call sends images: a text-only model fails here, with the provider's reason."""
    schema = {"type": "object", "properties": {"hair": {"type": "string"}}, "required": ["hair"],
              "additionalProperties": False}
    data, _cost, _tokens = backend.complete("Reply with JSON only.",
                                            [{"text": "What colour is the woman's hair?"}, {"image": REF}],
                                            schema, "live_vision", 512)
    assert "black" in data["hair"].lower()


# ---- a round: review, review_each, confirm ----------------------------------------------

def _check_review(rv, n, rubric):
    from ouroboros.params import MODES, PHASES
    assert len(rv.scores) == n and all(0 <= s <= 100 for s in rv.scores)
    assert 0 <= rv.best_index < n
    for c in rv.raw["candidates"]:
        assert set(c["scores"]) == set(rubric) and all(0 <= v <= 10 for v in c["scores"].values())
        assert isinstance(c["differences"], list)
    e = rv.edit
    assert e.get("focus") in ("prompt", "settings", "lora_weights", "loras")
    assert e.get("mode") in MODES and e.get("phase") in PHASES
    assert e.get("sampler_name") in (None, *SAMPLERS) and e.get("scheduler") in (None, *SCHEDULERS)
    assert rv.diagnosis.strip()


def test_a_round_review_scores_a_candidate_and_proposes_an_edit(judge):
    rubric = judge.rubric(True)
    rv = judge.review(REF, GOAL, "txt2img seed=1 steps=30 cfg=5.5", "", [GOOD], None, "", rubric)
    _check_review(rv, 1, rubric)


def test_one_call_per_candidate_ranks_a_good_image_over_a_wrong_one(judge):
    rubric = judge.rubric(True)
    rv = judge.review_each(REF, GOAL, ["txt2img seed=1", "txt2img seed=2"], "", [WRONG, GOOD], None, "", rubric)
    _check_review(rv, 2, rubric)
    assert rv.best_index == 1 and rv.scores[1] > rv.scores[0] + 5, rv.scores


def test_the_fresh_look_finds_an_added_item(judge):
    rubric = judge.rubric(True)
    rv = judge.confirm(REF, GOAL, "txt2img seed=1", FLAWED, None, "", rubric)
    _check_review(rv, 1, rubric)
    text = " ".join(rv.raw["candidates"][0]["differences"]).lower()
    assert any(w in text for w in ("shoulder", "epaulet", "emblem", "badge", "insignia", "fist")), text


def test_the_judges_edit_applies_to_real_parameters(judge):
    """What comes back is fed straight into apply_edit: it must give valid parameters."""
    from ouroboros.params import BOUNDS, MODES, GenParams, apply_edit
    rubric = judge.rubric(True)
    rv = judge.review(REF, GOAL, "txt2img seed=1 steps=30 cfg=5.5", "", [FLAWED], None, "", rubric)
    p = apply_edit(GenParams(positive="score_9, 1girl, red shirt", negative="lowres"), rv.edit, SAMPLERS, SCHEDULERS)
    assert p.mode in MODES and p.sampler_name in SAMPLERS and p.scheduler in SCHEDULERS
    for k in ("steps", "cfg", "denoise"):
        lo, hi = BOUNDS[k]
        assert lo <= getattr(p, k) <= hi
    assert p.positive.strip()


def test_loras_offered_are_the_only_ones_it_may_pick(judge):
    rubric = judge.rubric(True)
    menu = {"menu": "- FlatColor: flat colour cel shading\n- Jabstyle: painterly semi-realism",
            "stems": ["FlatColor", "Jabstyle"], "max": 2, "switch": True}
    rv = judge.review(REF, GOAL, "txt2img loras=FlatColor:0.8", "", [GOOD], None, "", rubric, "", menu)
    for l in rv.edit.get("loras") or []:
        assert l["lora"] in menu["stems"] and isinstance(l["strength"], (int, float))


def test_notes_are_summarized_short(judge):
    entries = [f"r{i}: best {50 + i}; edit added 'red collared shirt'; the shirt stayed white" for i in range(1, 6)]
    summary, cost = judge.summarize("", entries)
    assert summary and len(summary.split()) <= 220 and cost >= 0


# ---- the start: prompt, settings, picks ---------------------------------------------------

def test_a_prompt_written_from_the_reference_is_usable(backend):
    from ouroboros.prompter import _problems, write_prompt
    out = write_prompt(backend, "", "", "", "score_9, score_8_up, score_7_up, source_cartoon, flat color, 1girl",
                       "score_4, score_5, lowres", REF, 512, "- FlatColor (style) at strength 0.8; trigger word: flat color")
    assert not _problems(out["positive"], out["negative"])
    pos = out["positive"].lower()
    assert "score_9" in pos and "flat color" in pos and "black" in pos and "red" in pos
    assert "<lora:" not in pos


def test_a_prompt_written_from_words_alone_is_usable(backend):
    from ouroboros.prompter import _problems, write_prompt
    out = write_prompt(backend, GOAL, "", "", "score_9, score_8_up, source_cartoon, 1girl", "score_4, lowres")
    assert not _problems(out["positive"], out["negative"]) and "red" in out["positive"].lower()


def test_starting_settings_stay_in_range(backend):
    from ouroboros.params import BOUNDS
    from ouroboros.settings_advisor import suggest_settings
    out = suggest_settings(backend, checkpoint="autismmixSDXL_autismmixConfetti.safetensors", checkpoint_base="Pony",
                           samplers=SAMPLERS, schedulers=SCHEDULERS,
                           current={"steps": 30, "cfg": 5.5, "sampler_name": "dpmpp_2m", "scheduler": "karras"},
                           positive="score_9, 1girl, red shirt", description=GOAL, size=(832, 1216))
    assert out["sampler_name"] in SAMPLERS and out["scheduler"] in SCHEDULERS
    assert BOUNDS["steps"][0] <= out["steps"] <= BOUNDS["steps"][1] and BOUNDS["cfg"][0] <= out["cfg"] <= BOUNDS["cfg"][1]


def test_ai_picks_answer_from_the_menu(backend):
    from ouroboros.pose_picker import pick_item, pick_pose
    poses = [{"name": "hand_on_hip", "description": "standing, one hand on hip, other arm at side, cowboy shot"},
             {"name": "sitting_crosslegged", "description": "sitting cross-legged on the floor, hands in lap"}]
    p = pick_pose(backend, poses, description="standing with one hand on her hip")
    assert p["pose"] == "hand_on_hip", p
    styles = [{"name": "flat_cel", "description": "flat colour, bold black outlines, cel shading"},
              {"name": "oil_paint", "description": "thick oil paint, visible brush strokes"}]
    s = pick_item(backend, styles, "style", description="flat colour cartoon with bold outlines")
    assert s["pick"] == "flat_cel", s
    c = pick_item(backend, [{"name": "red_knight", "description": "red hair, silver armour"}], "character",
                  description="a blue-haired mage in robes")
    assert c["pick"] is None, c  # never a different character


# ---- the end: auto-fix's inspection and review ---------------------------------------------

def test_inspection_reports_flaws_the_repair_can_use(backend):
    from ouroboros import autofix
    issues, cost = autofix.inspect(backend, FLAWED, "score_9, 1girl, red shirt, red pants", 512)
    for i in issues:
        assert i["kind"] in autofix.KINDS and i["method"] in autofix.METHODS and 1 <= i["severity"] <= 3
        assert i["region"].strip()


def test_a_repair_review_compares_before_and_after(backend):
    from ouroboros import autofix
    targeted = [{"what": "added shoulder tabs", "region": "shoulders", "kind": "clothing", "severity": 2,
                 "method": "inpaint", "add": [], "negative_add": ["epaulettes"]}]
    data, cost = autofix.review(backend, FLAWED, GOOD, targeted, "score_9, 1girl, red shirt", 512)
    assert isinstance(data["better"], bool) and len(data["fixed"]) >= 1 and isinstance(data["issues"], list)
    # GOOD has no shoulder tabs, but it isn't a repaired copy of FLAWED (its arm and face
    # differ too): the tabs must count as fixed, and the other changes must be noticed.
    assert data["fixed"][0] is True, data
    assert data["new_problems"].strip() or data["issues"], data


# ---- a whole run: the loop and the judge talking for two rounds ---------------------------

class EvalComfy:
    """ComfyUI stand-in: each render 'comes out' as the next labelled eval image."""
    order = [GOOD, FLAWED]

    def __init__(self, url):
        self.n = 0

    def choices(self, node, name):
        raise RuntimeError("offline")

    def upload_image(self, path, subfolder="x"):
        return f"{subfolder}/{Path(path).name}"

    def queue(self, graph):
        self.n += 1
        return f"p{self.n}"

    def wait(self, ids, should_stop=None):
        return {i: {} for i in ids}

    def fetch_images(self, hist, node):
        img = self.order[(self.n - 1) % len(self.order)]
        return [img.read_bytes()]


@pytest.mark.parametrize("threshold, want", [(99, "review"), (1, "done")])
def test_a_two_round_job_with_the_real_judge(tmp_path, judge, judge_cfg, threshold, want):
    from ouroboros.jobs import Queue, load_job
    from ouroboros.loop import run_job
    from ouroboros.workflow import Workflows
    import ouroboros.workflow as wf
    (tmp_path / "workflows").mkdir()
    for f in ("example_workflow.json", "nodes.example.json"):
        shutil.copy(ROOT / "workflows" / f, tmp_path / "workflows" / f)
    wf._lora_root = lambda: None
    cfg = json.loads(json.dumps(judge_cfg))
    cfg["loop"].update(max_rounds=2, threshold=threshold, hand_refine=False, auto_fix=False, upscale=False,
                       ai_settings=False, batch_start=2, batch_end=2, plateau_rounds=5, local_prefilter=False)
    cfg["loras"]["mode"] = "off"
    folder = Queue(tmp_path / "jobs").add("live", "ref.png", REF.read_bytes(), {})  # no prompt: written from the image
    events = []
    result = run_job(load_job(folder), cfg, EvalComfy("x"), Workflows(tmp_path / "workflows"), judge,
                     SAMPLERS, SCHEDULERS, tmp_path / "runs", report=events.append)
    rounds = [e for e in events if e["type"] == "round"]
    prompt = next(e for e in events if e["type"] == "prompt")
    assert prompt["positive"].strip() and prompt["negative"].strip()
    assert rounds and all(isinstance(s, (int, float)) for r in rounds for s in r["scores"] if s is not None)
    assert result.status == want, [e.get("text") for e in events if e["type"] == "log"][-6:]
    if want == "done":
        assert any(e["type"] == "confirm" and e["passed"] for e in events)
    else:
        assert len(rounds) == 2  # round 2 rendered from the judge's edit
