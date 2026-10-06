"""Designer: a character changed one thing at a time (designer.py), its queue task and routes."""

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from ouroboros import designer
from ouroboros import pose as pose_mod


def png(path, colour=(200, 40, 40), size=(64, 96)):
    Image.new("RGB", size, colour).save(path)
    return path


class FakeFlows:
    output_node = "out"

    def __init__(self):
        self.built = []

    def build(self, p, image_name, batch, checkpoint, positive, size, control=None, ipadapter=None, pick=None):
        self.built.append({"p": p, "image": image_name, "batch": batch, "size": size, "control": control or [],
                           "ipadapter": ipadapter or [], "pick": pick})
        return {"graph": len(self.built)}

    def default(self, role):
        return {"positive": "score_9, 1girl", "negative": "lowres"}[role]


class FakeComfy:
    def queue(self, graph):
        return "pid"

    def wait(self, pids, should_stop=None):
        return {"pid": {}}

    def fetch_images(self, hist, node):
        out = []
        for c in ((10, 10, 10), (20, 20, 20), (30, 30, 30)):
            b = io.BytesIO()
            Image.new("RGB", (64, 96), c).save(b, "PNG")
            out.append(b.getvalue())
        return out


class FakeJudge:
    """Plans, then scores each candidate from a list of verdicts (the last one repeats)."""

    def __init__(self, verdicts, plan=None):
        self.verdicts, self.calls = list(verdicts), []
        self.plan = plan or {"style_tags": "score_9, (watercolor:1.2)", "content": "1girl, red hair",
                             "change_tags": [], "negative_add": ["3d"],
                             "drop_loras": ["oldstyle"], "region": "", "protect": "", "summary": "watercolor style",
                             "must_change": ["watercolor washes"], "must_keep": ["red hair", "green cloak"]}

    def complete(self, instructions, parts, schema, name, max_side):
        self.calls.append((name, parts))
        if name == "design_plan":
            return dict(self.plan), 0.01, 0
        if name == "design_review":  # every tag kept, but an old style word
            tags = [ln.split(". ", 1)[1] for ln in parts[0]["text"].splitlines() if ln[:1].isdigit()]
            return {"tags": [{"n": i, "tag": t, "verdict": "changes" if "style" in t else "keep"}
                             for i, t in enumerate(tags, 1)]}, 0.0, 0
        if name == "design_describe":
            return {"face": "stern", "hair": "red", "outfit": ["green cloak"], "background": "tan", "framing": "cowboy shot",
                    "pose": "arms down", "style": "cel"}, 0.001, 0
        v = self.verdicts.pop(0) if len(self.verdicts) > 1 else self.verdicts[0]
        listed = [ln.split("] ", 1)[1] for ln in parts[0]["text"].splitlines() if "] " in ln and ln[:1].isdigit()]
        return {**checklist_answer(v), "keep": [{"detail": x, "in_candidate": x, "verdict": "same"} for x in listed]}, 0.001, 0


def checklist_answer(v: dict) -> dict:
    """The judge's checklist for wanted scores: one change check (done 10 / partly 5 / not
    done 0, nearest), every kept detail "same", and each aspect brought down to its score by
    other differences (major -3, minor -1). Aspects default to 9, quality to 9."""
    want = {"identity": 9, "outfit": 9, "style": 9, "pose": 9, "framing": 9, "background": 9, **v}
    ch = v.get("change", 0)
    others = []
    for aspect in ("identity", "outfit", "style", "pose", "framing", "background"):
        gap = 10 - int(want[aspect])
        others += [{"what": f"{aspect} off", "aspect": aspect, "severity": "major"}] * (gap // 3)
        others += [{"what": f"{aspect} a little off", "aspect": aspect, "severity": "minor"}] * (gap % 3)
    return {"keep": [], "change": [{"check": "the change", "seen": "-",
                                    "verdict": "done" if ch >= 8 else "partly" if ch >= 4 else "not done"}],
            "other_differences": others, "quality": v.get("quality", 9), "missing": v.get("missing", ""),
            "prompt_add": v.get("prompt_add", []), "prompt_remove": v.get("prompt_remove", []),
            "negative_add": v.get("negative_add", [])}


@pytest.fixture
def design(tmp_path, monkeypatch):
    monkeypatch.setattr(pose_mod, "available", lambda: False)
    src = png(tmp_path / "src.png")
    d = designer.create(tmp_path, src, name="Ava", params={"positive": "score_9, 1girl, red hair, anime style",
                                                         "negative": "lowres", "seed": 7, "steps": 20, "cfg": 5,
                                                         "loras": [["styles/oldstyle.safetensors", 0.8],
                                                                   ["chars/ava.safetensors", 1.0]]})
    return tmp_path, d


def run(root, d, sess, judge, rounds=None, cfg=None):
    flows = FakeFlows()
    out = designer.run_session(root, d["id"], sess["id"], cfg=cfg or {"designer": {"batch": 3}}, comfy=FakeComfy(),
                               flows=flows, backend=judge, upload=lambda p: p.name, rounds=rounds)
    return out, flows


def test_same_model_confirmation_must_pass_independently(design, monkeypatch):
    root, d = design
    sess = designer.new_session(root, d["id"], base="original.png", kind="style",
                                change={"text": "watercolor"}, threshold=80, max_rounds=1)
    scores = iter([95, 40, 40, 65])  # three candidates, then the failing confirmation
    def verdict(*a, **kw):
        return {"score": next(scores), "differences": [], "missing": "", "change": 10,
                "kept": 10, "quality": 10, "prompt_add": [], "prompt_remove": [], "negative_add": []}, 0.01
    monkeypatch.setattr(designer, "judge", verdict)
    out, _ = run(root, d, sess, FakeJudge([{}]))
    assert out["status"] == "finished" and not out["confirmed"]
    candidate = out["rounds"][0]["candidates"][0]
    assert candidate["first_score"] == 95 and candidate["recheck"] == 65
    assert not candidate["confirmed"]


def test_confirmation_uses_its_own_threshold_and_checks_remaining_candidates(design, monkeypatch):
    root, d = design
    sess = designer.new_session(root, d["id"], base="original.png", kind="style",
                                change={"text": "watercolor"}, threshold=80, max_rounds=1)
    scores = iter([95, 90, 40, 85, 92])
    def verdict(*a, **kw):
        return {"score": next(scores), "differences": [], "missing": "", "change": 10,
                "kept": 10, "quality": 10, "prompt_add": [], "prompt_remove": [], "negative_add": []}, 0.01
    monkeypatch.setattr(designer, "judge", verdict)
    out, _ = run(root, d, sess, FakeJudge([{}]), cfg={"designer": {"batch": 3},
                  "judge": {"confirm_thresholds": {"designer": 90}}})
    first, second, _ = out["rounds"][0]["candidates"]
    assert not first["confirmed"] and first["score"] < 80
    assert second["confirmed"] and second["confirm_threshold"] == 90
    assert out["status"] == "passed" and out["best"]["image"] == second["image"]


def test_a_design_keeps_its_image_and_settings_and_lists_newest_first(design):
    root, d = design
    assert (designer.design_dir(root, d["id"]) / "original.png").is_file()
    assert designer.load(root, d["id"])["params"]["seed"] == 7
    other = designer.create(root, png(root / "b.png"), name="Bo")
    assert [x["name"] for x in designer.list_designs(root)][0] in ("Bo", "Ava")
    assert {x["id"] for x in designer.list_designs(root)} == {d["id"], other["id"]}
    with pytest.raises(FileNotFoundError):
        designer.load(root, "../" + d["id"])
    with pytest.raises(FileNotFoundError):
        designer.file_of(root, d["id"], "../../src.png")


def test_an_edit_changes_exactly_one_thing_and_needs_something_to_change_to(design):
    root, d = design
    with pytest.raises(ValueError):
        designer.new_session(root, d["id"], base="original.png", kind="hair", change={"text": "x"})
    with pytest.raises(ValueError):
        designer.new_session(root, d["id"], base="original.png", kind="features", change={"text": " "})
    with pytest.raises(ValueError):
        designer.new_session(root, d["id"], base="original.png", kind="pose", change={})
    s = designer.new_session(root, d["id"], base="original.png", kind="features", change={"text": "shorter hair"})
    assert designer.load(root, d["id"])["sessions"] == [s["id"]] and s["status"] == "queued"


def test_the_unchanged_image_can_never_pass_however_well_it_keeps_the_rest():
    same = {"change": 0, "identity": 10, "outfit": 10, "style": 10, "pose": 10, "background": 10, "quality": 10}
    assert designer.combined(same, "style") < 60
    done = {**same, "change": 9, "identity": 9, "outfit": 9}
    assert designer.combined(done, "style") >= 85
    drifted = {**done, "identity": 4, "outfit": 4, "background": 5}
    assert designer.combined(drifted, "style") < 85
    # one kept aspect clearly broken can't pass, however good the rest (a changed face)
    assert designer.combined({**done, "identity": 6, "outfit": 10, "background": 10, "change": 10}, "style") < 85
    assert designer.combined({**done, "outfit": 7, "change": 10, "quality": 10}, "style") < 85
    # the changed aspect doesn't count against keeping the rest
    assert designer.combined({**done, "style": 0}, "style") == designer.combined(done, "style")


def test_the_knobs_move_toward_the_change_when_it_is_missing_and_back_when_the_rest_drifts():
    k = designer.initial_knobs("style", {"positive": "a, b"}, has_skeleton=True, has_style_image=True)
    more, notes = designer.adjust("style", k, {"change": 3, "kept": 9}, [])
    assert more["denoise"] > k["denoise"] and more["edge"] < k["edge"] and notes
    less, _ = designer.adjust("style", k, {"change": 9, "kept": 6, "identity": 5, "outfit": 9}, [])
    assert less["denoise"] < k["denoise"] and less["edge"] > k["edge"] and less["edge_end"] > k["edge_end"]
    assert k["subject_ip"] == 0 and k["style_ip"] > 0  # the base image's adapter would carry its old style
    near, notes = designer.adjust("style", k, {"change": 7, "kept": 9, "identity": 9, "outfit": 9}, [])
    assert near["denoise"] > k["denoise"] and notes  # nearly there: a round must still move something
    p = designer.initial_knobs("pose", {"positive": "a"}, has_skeleton=True, has_style_image=False)
    p2, _ = designer.adjust("pose", p, {"change": 4, "kept": 9, "identity": 9, "outfit": 9}, [])
    assert p2["pose"] > p["pose"] and p2["denoise"] > p["denoise"] and p2["stage"] == "edit"
    p4, _ = designer.adjust("pose", p, {"change": 7, "kept": 8, "identity": 9, "outfit": 6}, [])
    assert p4["denoise"] < p["denoise"] and p4["subject_ip"] == 0  # the outfit drifts: hold it, no IP-Adapter
    assert p4["denoise"] >= 0.75  # below that the pose isn't followed
    p3, _ = designer.adjust("pose", p, {"change": 9, "kept": 9, "identity": 9, "outfit": 9}, [])
    assert p3["stage"] == "refine"
    p5, _ = designer.adjust("pose", p, {"change": 9, "kept": 7, "identity": 9, "outfit": 9}, [])
    assert p5["stage"] == "edit"  # the pose is right but something else isn't: don't polish its mistakes


def test_the_judges_tips_go_into_the_prompt_but_never_remove_the_change_itself():
    k = {"positive": "1girl, watercolor, red hair", "negative_add": [], "stage": "edit", "denoise": 0.6, "edge": 0.5,
         "subject_ip": 0.3, "style_ip": 0}
    out, _ = designer.adjust("style", k, {"change": 8, "kept": 9, "prompt_add": ["green cloak"],
                                          "prompt_remove": ["watercolor", "red hair"], "negative_add": ["blue cloak"]},
                             ["watercolor"])
    assert "watercolor" in out["positive"] and "green cloak" in out["positive"] and "red hair" not in out["positive"]
    assert out["negative_add"] == ["blue cloak"]


def test_a_style_edit_renders_from_the_base_with_its_edges_and_stops_when_a_second_look_agrees(design):
    root, d = design
    s = designer.new_session(root, d["id"], base="original.png", kind="style", change={"text": "watercolor"})
    judge = FakeJudge([{"change": 3}, {"change": 4}, {"change": 2},          # round 1: the style isn't there
                       {"change": 9}, {"change": 5}, {"change": 5}, {"change": 9}])  # round 2 passes; recheck 9
    out, flows = run(root, d, s, judge)
    assert out["status"] == "passed" and out["confirmed"] and len(out["rounds"]) == 2
    assert out["best"]["round"] == 2 and out["best"]["image"] == "r2_1.png"
    first, second = flows.built
    assert first["p"].mode == "img2img_reference" and first["image"] == "base_input.png"
    canny = [c for c in first["control"] if c.get("preprocess") == "canny"]
    assert canny and canny[0]["image"] == "base_input.png"
    assert second["p"].denoise > first["p"].denoise  # the style was missing: more freedom
    # the old style's LoRA is dropped, the character's kept; the plan's prompt and guard are used
    assert [n for n, _ in first["p"].loras] == ["chars/ava.safetensors"]
    assert "watercolor" in first["p"].positive and "3d" in first["p"].negative
    assert "anime style" not in first["p"].positive and "red hair" in first["p"].positive
    names = [c[0] for c in judge.calls]
    # the base described from the image alone, planned, then each candidate described blind and judged
    # the base described from the image alone, planned, its own tags reviewed against the
    # description, then each candidate described blind and judged
    assert names == ["design_describe", "design_plan", "design_review"] + ["design_describe", "design_judge"] * 7
    plan_parts = judge.calls[1][1]
    assert any("SEEN" in (x.get("text") or "") and "green cloak" in x["text"] for x in plan_parts)
    judged = judge.calls[4][1]
    assert not any("CURRENT PROMPT" in (x.get("text") or "") for x in judged)  # the judge never sees it
    assert any("described from the image alone" in (x.get("text") or "") for x in judged)
    saved = designer.load_session(root, d["id"], s["id"])
    assert saved["status"] == "passed" and saved["rounds"][1]["candidates"][0]["recheck"] >= 85


def test_without_passing_it_stops_at_its_maximum_and_one_more_round_continues_from_there(design):
    root, d = design
    s = designer.new_session(root, d["id"], base="original.png", kind="features", change={"text": "shorter hair"},
                             max_rounds=2)
    judge = FakeJudge([{"change": 5}], plan={**FakeJudge([]).plan, "region": "hair and shoulders", "protect": "face"})
    import ouroboros.masks as masks
    calls = []

    def fake_mask(comfy, source, name, text, out, threshold=0.35, grow_px=12, exclude=""):
        calls.append((text, grow_px, exclude))
        Image.new("RGBA", (64, 96)).save(out)
        return out
    orig = masks.masked_image
    masks.masked_image = fake_mask
    try:
        out, flows = run(root, d, s, judge)
        assert out["status"] == "finished" and len(out["rounds"]) == 2
        assert [b["p"].mode for b in flows.built] == ["inpaint_reference"] * 2
        assert calls[0][0] == "hair and shoulders" and calls[1][1] > calls[0][1]  # it grows while the change is missing
        assert calls[0][2] == "face"  # the face it frames isn't repainted
        more, _ = run(root, d, s, judge, rounds=1)
        assert len(more["rounds"]) == 3 and more["rounds"][-1]["round"] == 3
        assert len(judge.calls) == 3 + 2 * 9  # described, planned, reviewed once; not again for the extra round
    finally:
        masks.masked_image = orig


def test_a_pose_edit_draws_the_new_pose_over_the_characters_own_image_without_an_ip_adapter(design, monkeypatch):
    root, d = design
    monkeypatch.setattr(pose_mod, "available", lambda: True)
    found = {"body": [[0.5, 0.1, 1]] * 18, "size": [64, 96]}
    monkeypatch.setattr(pose_mod, "detect", lambda img: found)
    monkeypatch.setattr(pose_mod, "has_body", lambda p: True)
    monkeypatch.setattr(pose_mod, "fit", lambda p, a, b: p)
    monkeypatch.setattr(pose_mod, "render", lambda p, size, hands=True, face=False: Image.new("RGB", size))
    target = png(root / "pose.png", (0, 0, 0))
    s = designer.new_session(root, d["id"], base="original.png", kind="pose", change={"text": "arms crossed"},
                             target_image=target, max_rounds=3)
    judge = FakeJudge([{"change": 9, "identity": 4}])  # the pose is right, the face drifted
    _, flows = run(root, d, s, judge)
    first = flows.built[0]
    # from the character's own image, the new pose moving its limbs
    assert first["p"].mode == "img2img_reference" and first["image"] == "base_input.png"
    assert first["p"].denoise == 0.8 and first["p"].seed == 7  # the original's seed first
    assert [c["type"] for c in first["control"]] == ["openpose"]
    assert first["ipadapter"] == []  # it washed out the art style; img2img carries the character
    assert flows.built[1]["p"].mode == "img2img_reference"  # pose right: refine the best one


def test_kept_candidates_go_to_the_catalog_or_become_a_design_of_their_own(design):
    root, d = design
    s = designer.new_session(root, d["id"], base="original.png", kind="style", change={"text": "watercolor"},
                             max_rounds=1)
    run(root, d, s, FakeJudge([{"change": 7}]))
    entry = designer.keep(root, d["id"], s["id"], "r1_2.png")
    design_now = designer.load(root, d["id"])
    assert design_now["catalog"][0]["image"] == entry["image"] and entry["kind"] == "style"
    assert (designer.design_dir(root, d["id"]) / entry["image"]).is_file()
    assert entry["first_score"] == entry["score"] and entry["judge_model"] == "unknown"
    designer.update(root, d["id"], cover=entry["image"])
    designer.remove_from_catalog(root, d["id"], entry["id"])
    assert designer.load(root, d["id"])["cover"] == "original.png"
    new = designer.new_from_candidate(root, d["id"], s["id"], "r1_1.png")
    assert new["params"]["denoise"] and new["source"]["design"] == d["id"] and "Ava" in new["name"]
    with pytest.raises(FileNotFoundError):
        designer.keep(root, d["id"], s["id"], "../../original.png")
    # a kept version can be the base of the next edit, with the settings that made it
    s2 = designer.new_session(root, d["id"], base="original.png", kind="features", change={"text": "x"})
    assert s2["base"] == "original.png"


def test_the_server_makes_a_design_from_a_history_image_with_its_settings(tmp_path, monkeypatch):
    from ouroboros import generate, runner, server
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(server, "load_config", lambda: {})
    run_dir = tmp_path / "runs" / "manual" / "r1"
    run_dir.mkdir(parents=True)
    png(run_dir / "image_01.png")
    (run_dir / "run.json").write_text(json.dumps({
        "positive": "1girl, red hair", "negative": "lowres", "images": ["image_01.png"],
        "loras": [["a.safetensors", 0.7]], "request": {"seed": 5, "steps": 25, "checkpoint": "ck.safetensors"}}))
    out = server.create_design({"image": "/files/runs/manual/r1/image_01.png"})
    p = out["params"]
    assert (p["positive"], p["seed"], p["checkpoint"], p["loras"]) == ("1girl, red hair", 5, "ck.safetensors",
                                                                       [["a.safetensors", 0.7]])
    assert out["source"] == {"run": "manual/r1", "image": "image_01.png"}
    with pytest.raises(FileNotFoundError):
        server.create_design({"image": "/files/../config.json"})
    listed = server.designs_list()
    assert listed[0]["id"] == out["id"] and listed[0]["cover_url"].endswith("original.png")
    queued = []
    monkeypatch.setattr(generate, "start_design", lambda *a, **k: queued.append((a, k)) or "job1")
    res = server.design_edit({"id": out["id"], "kind": "features", "text": "blue eyes", "threshold": 120})
    assert res["id"] == "job1" and queued[0][0] == (out["id"], res["session"])
    detail = server.design_detail(out["id"])
    assert detail["sessions"][0]["summary"] == "features · blue eyes"
    assert designer.load_session(tmp_path, out["id"], res["session"])["threshold"] == 100


def test_a_protected_region_is_cut_out_of_the_mask(tmp_path, monkeypatch):
    from ouroboros import masks
    hair = Image.new("L", (40, 40), 0)
    hair.paste(255, (0, 0, 40, 20))   # the hair, top half
    face = Image.new("L", (40, 40), 0)
    face.paste(255, (10, 10, 30, 20))  # the face, inside it
    monkeypatch.setattr(masks, "clipseg_mask", lambda comfy, name, text: {"hair": hair, "face": face}[text])
    out = masks.masked_image(None, png(tmp_path / "s.png", size=(40, 40)), "s.png", "hair", tmp_path / "m.png",
                             grow_px=0, exclude="face")
    alpha = Image.open(out).getchannel("A")
    assert alpha.getpixel((2, 5)) == 0 and alpha.getpixel((20, 15)) == 255 and alpha.getpixel((20, 30)) == 255


def test_a_style_edit_swaps_the_old_style_lora_for_one_the_chooser_picks(monkeypatch):
    from ouroboros import lora_picker
    seen = {}

    def fake_suggest(backend, cards, positive, negative, description, reference, max_loras, max_side, references=None):
        seen.update(cards=[c["comfy_name"] for c in cards], description=description, max=max_loras)
        return {"picks": [{**cards[0], "strength": 0.8, "why": "watercolor look"}], "cost_usd": 0.001}
    monkeypatch.setattr(lora_picker, "suggest_loras", fake_suggest)
    cards = [{"comfy_name": "s/water.safetensors", "type": "style", "triggers": ["wtrcolor"]},
             {"comfy_name": "c/hat.safetensors", "type": "clothing"}]
    sess, logs = {}, []
    out = designer.edit_loras("style", [("s/old.safetensors", 0.9), ("c/ava.safetensors", 1.0)], {"positive": "1girl"},
                              {"s/old.safetensors": "style", "c/ava.safetensors": "character"},
                              change={"text": "watercolor"}, target_image=None, target_tags="", cards=lambda: cards,
                              checkpoint=None, backend=None, max_side=512, sess=sess, log=logs.append)
    assert out == [["c/ava.safetensors", 1.0], ["s/water.safetensors", 0.8]]
    assert seen["cards"] == ["s/water.safetensors"] and seen["max"] == 1 and "watercolor" in seen["description"]
    assert sess["lora_triggers"] == ["wtrcolor"] and any("old" in x for x in logs)
    # other kinds keep the base image's LoRAs (less any the plan drops) and pick none
    same = designer.edit_loras("pose", [("s/old.safetensors", 0.9)], {"positive": ""}, {"s/old.safetensors": "style"},
                               change={}, target_image=None, target_tags="", cards=lambda: cards, checkpoint=None,
                               backend=None, max_side=512, sess={}, log=logs.append)
    assert same == [["s/old.safetensors", 0.9]]


def test_a_new_pose_keeps_the_characters_framing():
    body = lambda pts: {"body": [pts.get(i) for i in range(18)]}  # noqa: E731
    # the pose image: a full figure, neck at 20% height, hips at 50%, ankles at 95%
    target = body({1: [0.5, 0.2], 8: [0.45, 0.5], 11: [0.55, 0.5], 10: [0.45, 0.95], 4: [0.3, 0.45]})
    # the character: a cowboy shot, neck at 25%, hips at 85% (the torso is twice as long)
    base = body({1: [0.5, 0.25], 8: [0.45, 0.85], 11: [0.55, 0.85]})
    out = pose_mod.match_framing(target, (100, 100), base, (100, 100), (100, 100))
    b = out["body"]
    assert b[1] == [0.5, 0.25] and abs(b[8][1] - 0.85) < 1e-6  # neck and hips land where the character's are
    assert b[10] is None  # the ankles fall out of the cowboy shot
    assert abs(b[4][1] - (0.25 + (0.45 - 0.2) * 2)) < 1e-6  # the hand scaled with the torso
    assert pose_mod.match_framing(target, (100, 100), body({1: [0.5, 0.2]}), (100, 100), (100, 100)) is None


def test_the_score_comes_from_the_checklist_not_from_the_judges_own_numbers():
    items = [{"detail": "olive-gold breastplate", "aspect": "outfit"}, {"detail": "tan background", "aspect": "background"},
             {"detail": "stern face", "aspect": "identity"}]
    lenient = {"keep": [{"detail": "olive-gold breastplate", "in_candidate": "silver breastplate", "verdict": "different"},
                        {"detail": "tan background", "in_candidate": "grey", "verdict": "different"},
                        {"detail": "stern face", "in_candidate": "stern face", "verdict": "same"}],
               "change": [{"check": "arms crossed", "seen": "arms hang down", "verdict": "not done"}],
               "other_differences": [{"what": "sword added", "aspect": "outfit", "severity": "major"}],
               "quality": 10, "identity": 10, "outfit": 10}  # its own 10s are ignored
    v = designer.checklist_scores(lenient, items, ["arms crossed"], "pose")
    assert v["outfit"] == 0 and v["background"] == 2 and v["identity"] == 10 and v["change"] == 0
    assert "olive-gold breastplate: silver breastplate" in v["differences"] and "sword added" in v["differences"]
    assert designer.combined(v, "pose") < 60
    # unanswered details and checks count as not kept / not done
    v = designer.checklist_scores({"keep": [], "change": [], "quality": 9}, items, ["arms crossed"], "pose")
    assert v["outfit"] == 5 and v["change"] == 0


def test_only_a_style_edit_may_drop_the_base_images_loras():
    class Planner:
        def complete(self, *a):
            return {"seen": {"outfit": "olive-gold armour"}, "positive": "x", "negative_add": [],
                    "drop_loras": ["FlatColor"], "region": "", "protect": "", "summary": "s", "must_change": ["c"],
                    "must_keep": [{"detail": "olive-gold armour", "aspect": "outfit"},
                                  {"detail": "arms at sides", "aspect": "pose"}]}, 0, 0
    out, _ = designer.plan(Planner(), png(Path(__import__("tempfile").mkdtemp()) / "b.png"), "pose", {"text": "arms crossed"},
                           target_image=None, target_tags="", original_positive="", example_prompt="", loras=[],
                           max_side=512)
    assert out["drop_loras"] == [] and out["seen"]["outfit"] == "olive-gold armour"
    assert out["must_keep"] == [{"detail": "olive-gold armour", "aspect": "outfit", "tag": "olive-gold armour"}]
    logs = []
    kept = designer.edit_loras("pose", [("FlatColor.safetensors", 0.8)], {"drop_loras": ["FlatColor"]}, {},
                               change={}, target_image=None, target_tags="", cards=None, checkpoint=None, backend=None,
                               max_side=512, sess={}, log=logs.append)
    assert kept == [["FlatColor.safetensors", 0.8]]


def test_the_judges_removals_find_the_prompts_own_tags():
    prompt = "1girl, (green sword on back:1.3), hilt visible over shoulder, (silver body armor:1.3), sword art style"
    assert designer.prompt_tags_matching(prompt, ["sword", "silver armor"]) == [
        "(green sword on back:1.3)", "sword art style", "(silver body armor:1.3)"]
    k = {"positive": prompt, "negative_add": [], "stage": "edit", "pose": 0.8, "pose_end": 0.9, "subject_ip": 0.6,
         "denoise": 0.8}
    out, _ = designer.adjust("pose", k, {"change": 9, "kept": 9, "identity": 9, "outfit": 9,
                                         "prompt_remove": ["silver armor"]}, [])
    assert "silver" not in out["positive"] and "green sword on back" in out["positive"]


def test_every_detail_to_keep_is_drawn_by_the_prompt():
    items = [{"detail": "light freckles", "aspect": "identity", "tag": "freckles"},
             {"detail": "beige backdrop", "aspect": "background", "tag": "beige background"},
             {"detail": "gold armour", "aspect": "outfit", "tag": "gold armor"},
             {"detail": "flat colours", "aspect": "style", "tag": "flat color"}]
    out = designer.with_keep_tags("1girl, cute face\n(gold armor:1.3), breastplate\nsolid color background", items)
    assert "freckles" in out and "beige background" in out
    assert out.count("gold armor") == 1  # already drawn, weight and all
    assert "flat color" not in out  # style tags are the style's business


def test_a_moved_backdrop_is_measured_not_judged(tmp_path):
    beige, grey = png(tmp_path / "b.png", (214, 190, 160)), png(tmp_path / "g.png", (110, 104, 108))
    same = png(tmp_path / "s.png", (212, 191, 158))
    assert designer.backdrop_shift(beige, same) < 3 and designer.backdrop_shift(beige, grey) > 12


def test_the_prompt_keeps_its_style_tags_and_takes_the_rest_from_the_image():
    original = ("score_9, score_8_up, source_cartoon, flat color\nwestern animation, bold outlines\n"
                "1girl, cute face, (silver body armor:1.3)\n(green sword on back:1.3), plain background")
    style, rest = designer.split_style(original)
    assert style == "score_9, score_8_up, source_cartoon, flat color\nwestern animation, bold outlines"
    assert rest.startswith("1girl") and "sword" in rest

    class Planner:
        def __init__(self):
            self.parts = None

        def complete(self, instr, parts, schema, name, *a):
            if name == "design_review":  # the base image's own tags, checked against the description
                tags = [ln.split(". ", 1)[1] for ln in parts[0]["text"].splitlines() if ln[:1].isdigit()]
                self.reviewed = tags
                verdict = lambda t: ("not visible" if "sword" in t else "contradicted" if "silver" in t  # noqa: E731
                                     else "keep")
                return {"tags": [{"n": i, "tag": t, "verdict": verdict(t)} for i, t in enumerate(tags, 1)]}, 0, 0
            self.parts = parts
            return {"seen": {}, "style_tags": "watercolor", "content": "1girl, (olive-gold armor:1.3)",
                    "change_tags": ["arms crossed"], "negative_add": [], "drop_loras": [], "region": "", "protect": "",
                    "summary": "s", "must_change": ["arms crossed"],
                    "must_keep": [{"detail": "olive-gold armour", "aspect": "outfit", "tag": "olive-gold armor"}]}, 0, 0
    pl = Planner()
    out, _ = designer.plan(pl, png(Path(__import__("tempfile").mkdtemp()) / "b.png"), "pose", {"text": "arms crossed"},
                           target_image=None, target_tags="", original_positive=original, example_prompt="", loras=[],
                           max_side=512, seen={"outfit": "olive-gold armour"})
    assert out["positive"].startswith("score_9, score_8_up, source_cartoon, flat color\nwestern animation")
    # The tag the description contradicts stays (it drew the image), and the armour it names
    # isn't named again in the description's words.
    assert "sword" not in out["positive"] and "silver body armor" in out["positive"]
    assert "olive-gold" not in out["positive"]
    assert "cute face" in out["positive"] and "arms crossed" in out["positive"]  # its own face tags stay
    assert not any("sword" in (x.get("text") or "") for x in pl.parts)  # the planner never sees the old description
    assert "(green sword on back:1.3)" in out["removed_tags"]
    out, _ = designer.plan(pl, png(Path(__import__("tempfile").mkdtemp()) / "b.png"), "style", {"text": "watercolor"},
                           target_image=None, target_tags="", original_positive=original, example_prompt="", loras=[],
                           max_side=512)
    assert out["positive"].startswith("watercolor\n\n1girl, cute face")  # a style edit writes new style tags


def test_a_moved_backdrop_puts_the_base_images_backdrop_in_the_prompt():
    k = {"positive": "1girl, solid color background", "negative_add": [], "stage": "edit", "pose": 0.8,
         "pose_end": 0.9, "subject_ip": 0.6, "denoise": 0.8}
    v = {"change": 9, "kept": 9, "identity": 9, "outfit": 9, "backdrop_shift": 28, "blind": {"background": "grey"}}
    out, notes = designer.adjust("pose", k, v, [], backdrop="light beige")
    assert "(light beige background:1.2)" in out["positive"] and "grey background" in out["negative_add"]
    again, _ = designer.adjust("pose", out, v, [], backdrop="light beige")
    # no stronger than 1.2: heavier, the beige bled into the hair and armour
    assert "(light beige background:1.2)" in again["positive"] and again["positive"].count("beige") == 1


def test_the_details_to_keep_come_from_the_images_description():
    seen = {"face": "young adult female, oval face, light freckles across nose, no visible makeup",
            "hair": "brown, high bun with loose strands framing the face",
            "outfit": "gold segmented armor; black undersuit; green gem in the centre of the chest plate",
            "background": "solid light beige", "framing": "cowboy shot", "pose": "arms at sides", "style": "flat colour"}
    items = designer.keep_from_seen(seen, "pose")
    details = [x["detail"] for x in items]
    assert "brown hair" in details and "solid light beige background" in details and "light freckles across nose" in details
    assert "gold segmented armor" in details and "green gem in the centre of the chest plate" in details
    assert not any("makeup" in d for d in details) and "arms at sides" not in details  # negations, the change
    assert not designer.specific("outfit") and designer.specific("olive-gold breastplate")
    # with the image's own prompt only marks are added for the face; outfit and backdrop always
    out = designer.with_keep_tags("1girl, cute face", items, own_prompt=True)
    assert "light freckles across nose" in out and "loose strands framing the face" in out
    assert "oval face" not in out and "solid light beige background" in out and "gold segmented armor" in out


def test_the_judges_tips_must_name_something_the_base_image_has():
    base = "gold segmented armor, black undersuit, black gloves, green gems on the gauntlets, light beige"
    assert designer.in_base("black gloves", base) and designer.in_base("green gem on gauntlets", base)
    assert not designer.in_base("black skirt", base) and not designer.in_base("black cape with green lining", base)
    k = {"positive": "1girl", "negative_add": [], "stage": "edit", "pose": 0.8, "pose_end": 0.9, "subject_ip": 0.0,
         "denoise": 0.8}
    out, _ = designer.adjust("pose", k, {"change": 9, "kept": 9, "identity": 9, "outfit": 9,
                                         "prompt_add": ["black skirt", "black gloves"]}, [], base_text=base)
    assert "black gloves" in out["positive"] and "skirt" not in out["positive"]


def test_the_pose_is_measured_limb_by_limb():
    # neck, right arm (2-4), left arm (5-7): crossed forearms point across the body
    def body(rw, lw):
        b = [None] * 18
        b[0], b[1] = [0.5, 0.1], [0.5, 0.2]
        b[2], b[3], b[4] = [0.4, 0.2], [0.38, 0.35], rw
        b[5], b[6], b[7] = [0.6, 0.2], [0.62, 0.35], lw
        return {"body": b}
    crossed, down = body([0.6, 0.33], [0.4, 0.33]), body([0.37, 0.5], [0.63, 0.5])
    assert pose_mod.limb_match(crossed, crossed, (64, 96)) == (10.0, [])
    score, off = pose_mod.limb_match(crossed, down, (64, 96))
    assert score < 3 and "right forearm" in off[0] and len(off) == 2
    assert pose_mod.limb_match(crossed, None, (64, 96)) == (None, [])
    assert pose_mod.limb_match(crossed, {"body": [None] * 18}, (64, 96)) == (None, [])


def test_a_pose_the_judge_calls_done_but_the_skeleton_doesnt_match_isnt_done(tmp_path, monkeypatch):
    base, cand = png(tmp_path / "b.png"), png(tmp_path / "c.png")
    monkeypatch.setattr(pose_mod, "detect", lambda img: {"body": "measured"})
    monkeypatch.setattr(pose_mod, "limb_match", lambda t, f, size: (2.5, ["left forearm 100 degrees off"]))
    the_plan = {"summary": "arms crossed", "must_change": ["arms crossed"], "seen": {}, "must_keep": []}
    v, _ = designer.judge(FakeJudge([{"change": 10}]), base, cand, "pose", the_plan, change={}, target_image=None,
                          target_tags="", positive="", max_side=512, target_pose={"body": []})
    assert v["change"] == 2.5 and v["pose_match"] == 2.5 and v["score"] < 60
    assert v["differences"][0] == "pose measured: left forearm 100 degrees off"


def test_a_pass_the_second_opinion_turns_down_scores_what_it_gave_and_the_next_one_up_is_checked(design):
    root, d = design
    s = designer.new_session(root, d["id"], base="original.png", kind="style", change={"text": "watercolor"},
                             max_rounds=1)
    perfect = {"change": 10, "identity": 10, "outfit": 10, "style": 10, "pose": 10, "framing": 10,
               "background": 10, "quality": 10}
    sure = FakeJudge([perfect])
    second = FakeJudge([{"change": 3}, perfect])  # turns the first down, agrees on the second

    def go(confirm):
        return designer.run_session(root, d["id"], s["id"], cfg={"designer": {"batch": 2}}, comfy=FakeComfy(),
                                    flows=FakeFlows(), backend=sure, upload=lambda p: p.name, confirm_backend=confirm)
    out = go(second)
    first, other = out["rounds"][0]["candidates"]
    assert first["first_score"] >= 85 and first["score"] == first["recheck"] < 85
    assert out["status"] == "passed" and out["best"]["image"] == "r1_2.png" and other["recheck"] >= 85
    assert any("not confirmed: r1_1.png" in x for x in out["log"])
    # Both turned down: neither stands as the best at a passing score.
    s2 = designer.new_session(root, d["id"], base="original.png", kind="style", change={"text": "watercolor"},
                              max_rounds=1)
    s = s2
    out = go(FakeJudge([{"change": 3}]))
    assert out["status"] == "finished" and out["best"]["score"] < 85


def test_a_measured_skeleton_match_stops_the_knobs_chasing_the_gaze():
    k = {"positive": "1girl", "negative_add": [], "stage": "edit", "pose": 0.85, "pose_end": 0.9, "denoise": 0.8}
    v = {"change": 6.67, "pose_match": 9.1, "kept": 7, "identity": 7, "outfit": 9}
    out, notes = designer.adjust("pose", k, v, [])
    assert out["pose"] == 0.85 and out["denoise"] < 0.8 and "the skeleton matches" in notes[0]
    off, _ = designer.adjust("pose", k, {**v, "pose_match": 3.0}, [])
    assert off["pose"] > 0.85 and off["denoise"] > 0.8  # the arms really are elsewhere


def test_the_backdrop_tag_is_made_from_plain_words():
    assert designer.backdrop_words("Light tan or beige background.") == "light tan"
    assert designer.backdrop_words("plain light beige background") == "light beige"
    assert designer.backdrop_words("A plain grey background with a vignette") == "grey"


def test_a_tip_that_says_no_isnt_put_in_the_prompt():
    k = {"positive": "1girl", "negative_add": [], "stage": "edit", "pose": 0.85, "pose_end": 0.9, "denoise": 0.8}
    v = {"change": 9, "kept": 9, "identity": 9, "outfit": 9, "prompt_add": ["no gloves", "black gloves"]}
    out, _ = designer.adjust("pose", k, v, [], base_text="black gloves")
    assert "black gloves" in out["positive"] and "no gloves" not in out["positive"]



def test_the_change_swaps_the_tags_it_replaces_in_place_and_leaves_every_other_tag():
    original = "score_9, flat color\n\n1girl, cute face, looking left\n\n(cowboy shot:1.2), arms at sides, (green sword on back:1.3)"
    out = designer.swap_tags(original, ["arms at sides", "looking left"], ["arms crossed", "looking at viewer"])
    assert out == ("score_9, flat color\n\n1girl, cute face, arms crossed, looking at viewer\n\n"
                   "(cowboy shot:1.2), (green sword on back:1.3)")
    # nothing replaced: the new tags go where they fit, nothing else moves
    assert designer.swap_tags("1girl, cute face", [], ["angry face"]) == "1girl, cute face, angry face"
    assert designer.weight_tags("1girl, arms crossed\nx", ["arms crossed"], 1.2) == "1girl, (arms crossed:1.2)\nx"
    assert designer.weight_tags("1girl, arms crossed", ["arms crossed"], 1.0) == "1girl, arms crossed"


def test_an_image_drawn_from_noise_can_be_drawn_again_from_its_seed_and_place_in_the_batch():
    made = {"seed": 965599748, "batch_index": 7, "batch_of": 8, "size": [768, 1344], "from_noise": True}
    assert designer.regen_source(made) == {"seed": 965599748, "index": 7, "of": 8, "size": [768, 1344]}
    assert designer.regen_source({**made, "from_noise": False}) is None  # an img2img edit, a ControlNet...
    assert designer.regen_source({"seed": 1}) is None


def test_an_edit_of_a_character_drawn_from_noise_draws_it_again_from_that_noise(design):
    root, d = design
    data = designer.load(root, d["id"])
    data["params"].update(batch_index=5, batch_of=8, size=[64, 96], from_noise=True)
    designer.save(root, data)
    s = designer.new_session(root, d["id"], base="original.png", kind="features", change={"text": "angry face"},
                             max_rounds=3)
    plan = {**FakeJudge([]).plan, "style_tags": "", "change_tags": ["angry face"], "summary": "angry face",
            "must_change": ["angry face"]}
    judge = FakeJudge([{"change": 3}, {"change": 4}, {"change": 2}], plan=plan)  # never made: falls back
    out, flows = run(root, d, s, judge)
    regen = [b for b in flows.built if b["pick"] is not None]
    assert len(regen) == 6 and all(b["pick"] == 5 and b["batch"] == 8 and b["p"].seed == 7 for b in regen)
    assert all(b["p"].mode == "txt2img" and b["p"].denoise == 1.0 for b in regen)
    # the original prompt, the replaced tag swapped ("anime style" is the review's "changes")
    first = regen[0]["p"].positive
    assert "angry face" in first and "anime style" not in first and "red hair" in first
    assert "(angry face:1.2)" in regen[1]["p"].positive  # a stronger variant
    c = out["rounds"][0]["candidates"][0]["params"]
    assert c["from_noise"] and c["batch_index"] == 5 and c["batch_of"] == 8  # a kept one can be drawn again
    # two rounds without the change: the third edits the image instead
    assert out["rounds"][2]["settings"]["stage"] == "edit" and flows.built[-1]["pick"] is None


def test_an_edit_can_be_deleted_and_what_was_kept_from_it_stays(design):
    root, d = design
    s = designer.new_session(root, d["id"], base="original.png", kind="style", change={"text": "watercolor"},
                             max_rounds=1)
    run(root, d, s, FakeJudge([{"change": 5}]))
    entry = designer.keep(root, d["id"], s["id"], "r1_1.png")
    designer.remove_session(root, d["id"], s["id"])
    after = designer.load(root, d["id"])
    assert s["id"] not in after["sessions"] and not (designer.design_dir(root, d["id"]) / "sessions" / s["id"]).exists()
    assert designer.file_of(root, d["id"], entry["image"]).is_file()


def test_a_character_that_is_really_another_moves_into_it_with_its_versions_and_edits(design):
    root, d = design
    other = designer.create(root, png(root / "o.png", (200, 0, 0)), name="Bea",
                            params={"positive": "1girl, blue hair", "seed": 3, "from_noise": True, "batch_index": 2,
                                    "batch_of": 4, "size": [64, 96]})
    s = designer.new_session(root, other["id"], base="original.png", kind="style", change={"text": "ink"}, max_rounds=1)
    run(root, other, s, FakeJudge([{"change": 5}]))
    kept = designer.keep(root, other["id"], s["id"], "r1_2.png")
    into = designer.merge(root, d["id"], other["id"])
    with pytest.raises(FileNotFoundError):
        designer.load(root, other["id"])  # in the bin
    first, second = into["catalog"][-2:]
    assert first["kind"] == "merged" and first["merged_from"]["name"] == "Bea"
    assert first["params"]["seed"] == 3 and first["params"]["from_noise"]  # still drawn again from its noise
    assert second["params"]["positive"] and second["from"] == first["image"]  # from Bea's image, as moved
    assert s["id"] in into["sessions"]
    moved = designer.load_session(root, d["id"], s["id"])
    assert moved["base"] == first["image"] and designer.file_of(root, d["id"], moved["base"]).is_file()
    assert second["session"] == s["id"] and second["candidate"] == kept["candidate"]
    with pytest.raises(ValueError):
        designer.merge(root, d["id"], d["id"])


def test_a_manual_edit_renders_your_prompt_and_settings_and_judges_nothing(design):
    root, d = design
    data = designer.load(root, d["id"])
    data["params"].update(batch_index=7, batch_of=8, size=[64, 96], from_noise=True)
    designer.save(root, data)
    q = {"positive": "score_9, 1girl, red hair, angry face", "negative": "lowres", "mode": "same_noise",
         "steps": 20, "cfg": 5, "loras": [["chars/ava.safetensors", 1.0]], "batch_size": 4}
    s = designer.new_manual_session(root, d["id"], base="original.png", request=q)
    flows = FakeFlows()
    out = designer.run_manual(root, d["id"], s["id"], cfg={}, comfy=FakeComfy(), flows=flows, upload=lambda p: p.name)
    b = flows.built[0]
    assert b["pick"] == 7 and b["batch"] == 8 and b["p"].seed == 7 and b["p"].mode == "txt2img"  # its own noise
    assert b["p"].positive == q["positive"] and b["p"].loras == (("chars/ava.safetensors", 1.0),)
    c = out["rounds"][0]["candidates"]
    assert len(c) == 1 and c[0]["params"]["batch_index"] == 7 and c[0]["score"] is None
    # new noise, a batch: each one is image i of that batch
    s2 = designer.new_manual_session(root, d["id"], base="original.png",
                                     request={**q, "mode": "noise", "seed": 11, "batch_size": 3})
    out = designer.run_manual(root, d["id"], s2["id"], cfg={}, comfy=FakeComfy(), flows=flows, upload=lambda p: p.name)
    assert flows.built[1]["pick"] is None and flows.built[1]["batch"] == 3
    assert [x["params"]["batch_index"] for x in out["rounds"][0]["candidates"]] == [0, 1, 2]
    s3 = designer.new_manual_session(root, d["id"], base="original.png", request={**q, "mode": "img2img", "denoise": 0.4})
    designer.run_manual(root, d["id"], s3["id"], cfg={}, comfy=FakeComfy(), flows=flows, upload=lambda p: p.name)
    assert flows.built[2]["p"].mode == "img2img_reference" and flows.built[2]["p"].denoise == 0.4
    with pytest.raises(ValueError):
        designer.new_manual_session(root, d["id"], base="original.png", request={**q, "positive": " "})
    kept = designer.keep(root, d["id"], s2["id"], "r1_2.png")
    assert designer.regen_source(designer.made_settings(root, d["id"], kept["image"]))["index"] == 1


def test_versions_can_be_renamed_and_go_back_to_their_default_name(design):
    root, d = design
    s = designer.new_session(root, d["id"], base="original.png", kind="style", change={"text": "ink"}, max_rounds=1)
    run(root, d, s, FakeJudge([{"change": 5}]))
    entry = designer.keep(root, d["id"], s["id"], "r1_1.png")
    designer.rename_version(root, d["id"], entry["image"], "  Ink test  ")
    designer.rename_version(root, d["id"], "original.png", "Armour")
    got = designer.load(root, d["id"])
    assert got["catalog"][0]["name"] == "Ink test" and got["original_name"] == "Armour"
    designer.rename_version(root, d["id"], entry["image"], "")
    assert designer.load(root, d["id"])["catalog"][0]["name"] == ""
    with pytest.raises(FileNotFoundError):
        designer.rename_version(root, d["id"], "catalog/nope.png", "x")


def test_a_manual_edit_can_follow_a_pose_and_carry_the_character_and_a_style(design, monkeypatch):
    root, d = design
    monkeypatch.setattr(pose_mod, "available", lambda: True)
    monkeypatch.setattr(pose_mod, "detect", lambda img: {"body": [[0.5, 0.1]] * 18, "size": [64, 96]})
    monkeypatch.setattr(pose_mod, "has_body", lambda p: True)
    monkeypatch.setattr(pose_mod, "render", lambda p, size, hands=True, face=False: Image.new("RGB", size))
    style = png(root / "style.png", (0, 0, 200))
    q = {"positive": "1girl, red hair", "mode": "noise", "seed": 3, "pose_control": True, "pose_own": True,
         "pose_strength": 0.7, "pose_end": 0.9, "subject_ip": True, "subject_ip_weight": 0.3,
         "style_ip": True, "style_ip_weight": 0.5, "style_image": "../../../../etc/passwd"}  # a path is never taken
    s = designer.new_manual_session(root, d["id"], base="original.png", request=q, style_image=style)
    assert s["manual"]["style_image"] == "style.png"
    flows = FakeFlows()
    out = designer.run_manual(root, d["id"], s["id"], cfg={}, comfy=FakeComfy(), flows=flows,
                              upload=lambda p: p.name)
    b = flows.built[0]
    assert [c["type"] for c in b["control"]] == ["openpose"] and b["control"][0]["strength"] == 0.7
    assert [a["role"] for a in b["ipadapter"]] == ["subject", "style"]
    c = out["rounds"][0]["candidates"][0]["params"]
    assert c["from_noise"] is False and c["control"] and c["ipadapter"]  # guided: not redrawable from noise alone
    # a style IP-Adapter with nothing to take the style from is off
    s2 = designer.new_manual_session(root, d["id"], base="original.png", request={**q, "style_image": None})
    assert s2["manual"]["style_ip"] is False


def test_versions_stay_whole_and_usable_after_every_edit_they_came_from_is_deleted(design, monkeypatch):
    from ouroboros import runner, server
    root, d = design
    monkeypatch.setattr(server, "ROOT", root)
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(server, "load_config", lambda: {})
    data = designer.load(root, d["id"])
    data["params"].update(batch_index=1, batch_of=4, size=[64, 96], from_noise=True)
    designer.save(root, data)
    auto = designer.new_session(root, d["id"], base="original.png", kind="style", change={"text": "ink"}, max_rounds=1)
    run(root, d, auto, FakeJudge([{"change": 5}]))
    manual = designer.new_manual_session(root, d["id"], base="original.png",
                                         request={"positive": "1girl, red hair", "mode": "noise", "seed": 4,
                                                  "batch_size": 2})
    designer.run_manual(root, d["id"], manual["id"], cfg={}, comfy=FakeComfy(), flows=FakeFlows(),
                        upload=lambda p: p.name)
    v1 = designer.keep(root, d["id"], auto["id"], "r1_1.png")
    v2 = designer.keep(root, d["id"], manual["id"], "r1_2.png")
    card = designer.new_from_candidate(root, d["id"], manual["id"], "r1_1.png")
    designer.update(root, d["id"], cover=v2["image"])
    for sid in (auto["id"], manual["id"]):
        designer.remove_session(root, d["id"], sid)
    assert not (designer.design_dir(root, d["id"]) / "sessions").exists() or \
        not any((designer.design_dir(root, d["id"]) / "sessions").iterdir())
    detail = server.design_detail(d["id"])  # the page still lists them, with their images
    assert [e["image"] for e in detail["catalog"]] == [v1["image"], v2["image"]] and detail["sessions"] == []
    assert detail["cover"] == v2["image"]
    for e in detail["catalog"]:
        assert designer.file_of(root, d["id"], e["image"]).is_file()
    assert designer.file_of(root, card["id"], "original.png").is_file()  # the new card has its own copy
    # and each can still be edited, the manual one still from its own noise
    assert designer.regen_source(designer.made_settings(root, d["id"], v2["image"]))["index"] == 1
    again = designer.new_session(root, d["id"], base=v1["image"], kind="style", change={"text": "ink"},
                                 max_rounds=1)
    out, _ = run(root, d, again, FakeJudge([{"change": 5}]))
    assert out["status"] == "finished" and out["rounds"]
