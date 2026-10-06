"""Separate style / subject / pose targets: jobs, the judge's request, the prompt writer and
the chained IP-Adapters."""
import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from ouroboros import targets as tmod
from ouroboros.jobs import Queue, load_job
from ouroboros.targets import Target, Targets


def img(path: Path, color="red") -> Path:
    Image.new("RGB", (64, 96), color).save(path)
    return path


def test_a_one_image_job_is_not_split(tmp_path):
    ref = img(tmp_path / "ref.png")
    assert not Targets().split
    assert not Targets(subject=Target(ref), style=Target(ref)).split  # one image for everything
    assert Targets(style=Target(text="cel shading")).split
    assert Targets(subject=Target(ref), style=Target(img(tmp_path / "s.png", "blue"))).split


def test_queue_add_saves_target_images_and_the_subject_is_the_reference(tmp_path):
    q = Queue(tmp_path / "jobs")
    folder = q.add("knight", None, None, {"description": "a knight", "style": {"text": "90s anime"}},
                   {"subject": ("me.jpg", (img(tmp_path / "a.png")).read_bytes()),
                    "pose": ("p.png", img(tmp_path / "b.png", "blue").read_bytes())})
    job = load_job(folder)
    assert job.reference.name == "subject.jpg" and job.targets.subject.image.name == "subject.jpg"
    assert job.targets.pose.image.name == "pose.png" and job.targets.style.text == "90s anime"
    assert "style" not in job.overrides and "subject" not in job.overrides


def test_queue_add_needs_an_image(tmp_path):
    with pytest.raises(ValueError):
        Queue(tmp_path / "jobs").add("x", None, None, {"subject": {"text": "a knight"}})
    assert not any((tmp_path / "jobs" / "pending").iterdir())  # nothing left behind


def test_judge_parts_label_each_target_and_send_each_image_once(tmp_path):
    ref = img(tmp_path / "ref.png")
    tg = Targets(style=Target(text="thick ink lines"), subject=Target(ref))
    parts = tmod.judge_parts(tg, ref)
    text = " ".join(p.get("text", "") for p in parts)
    assert "identity, color, extras against SUBJECT" in text and "composition against POSE" in text
    assert "STYLE TARGET" in text and "described: thick ink lines" in text
    assert "(the same image as the SUBJECT target above)" in text  # pose falls back to the same image
    assert sum(1 for p in parts if "image" in p) == 1


def test_judge_uses_targets_when_split(tmp_path):
    from ouroboros.judge import Judge
    ref, cand = img(tmp_path / "ref.png"), img(tmp_path / "c.png", "green")
    seen = {}

    class B:
        context_window = 8000

        def complete(self, instructions, parts, schema, name, max_side):
            seen["parts"] = parts
            return {"candidates": [{"index": 0, "differences": [], "scores": {"identity": 8}, "notes": ""}],
                    "best_index": 0, "diagnosis": "", "edit": {}}, 0.0, 10

    j = Judge.__new__(Judge)
    j.cfg, j.backend, j.base_rubric, j.samplers, j.schedulers = {}, B(), {"identity": {"weight": 1, "anchors": "x"}}, [], []
    j.review(ref, "goal", "p", "", [cand], targets=Targets(subject=Target(ref), style=Target(text="watercolour")))
    text = " ".join(p.get("text", "") for p in seen["parts"])
    assert "TARGETS:" in text and "REFERENCE IMAGE" not in text and "its TARGETS" in text
    j.review(ref, "goal", "p", "", [cand], targets=Targets())  # not split: the classic request
    assert any(p.get("text", "").startswith("REFERENCE IMAGE") for p in seen["parts"])


def test_prompt_writer_gets_each_target(tmp_path):
    from ouroboros.prompter import write_prompt
    seen = {}

    class B:
        def complete(self, instructions, parts, schema, name, max_side):
            seen["parts"] = parts
            return {"positive": "1girl", "negative": "", "dropped": [], "notes": ""}, 0, 0
    subj = img(tmp_path / "s.png")
    write_prompt(B(), "", targets=tmod.prompt_parts(Targets(subject=Target(subj), style=Target(text="ink wash"))))
    text = " ".join(p.get("text", "") for p in seen["parts"])
    assert "write the prompt from the TARGETS" in text and "STYLE (write the drawing-style" in text
    assert "ink wash" in text and any(p.get("image") == subj for p in seen["parts"])


@pytest.fixture
def flows(tmp_path, monkeypatch):
    import ouroboros.workflow as wf
    src = Path(__file__).resolve().parent.parent / "workflows"
    (tmp_path / "workflows").mkdir()
    for f in ("example_workflow.json", "nodes.example.json"):
        shutil.copy(src / f, tmp_path / "workflows" / f)
    monkeypatch.setattr(wf, "_lora_root", lambda: None)
    return wf.Workflows(tmp_path / "workflows")


def test_two_ip_adapters_chain_and_share_the_loaded_models(flows):
    from ouroboros.params import GenParams
    ips = [{"image": "subject.png", "weight": 0.6, "weight_type": "linear"},
           {"image": "style.png", "weight": 0.5, "weight_type": "style transfer"}]
    g = flows.build(GenParams(), "x.png", ipadapter=ips)
    ks = g[flows.spec["roles"]["seed"][0]]["inputs"]
    assert ks["model"] == ["ip6", 0]                       # the sampler gets the second adapter's model
    assert g["ip5"]["inputs"]["model"] == ["ip3", 0]       # which applies on top of the first
    assert g["ip5"]["inputs"]["ipadapter"] == ["ip2", 1]   # reusing the first loader's models
    assert g["ip6"]["inputs"]["weight_type"] == "style transfer" and g["ip4"]["inputs"]["image"] == "style.png"
    one = flows.build(GenParams(), "x.png", ipadapter=ips[0])  # a single dict still works
    assert "ip4" not in one and "ip3" in one


def test_home_subject_and_style_targets(home_run_targets):
    job, rec, comfy = home_run_targets
    ips = [n for n in comfy.graphs[-1].values() if n["class_type"] == "IPAdapterAdvanced"]
    assert [n["inputs"]["weight_type"] for n in ips] == ["linear", "style transfer"]
    assert rec["targets"]["style"]["text"] == "ink wash" and rec["subject_reference"] == "subject.png"
    assert job["params"].startswith("txt2img")  # IP-Adapters on: no img2img from the reference


@pytest.fixture
def home_run_targets(tmp_path, monkeypatch):
    import base64
    import io
    import ouroboros.generate as gen
    import ouroboros.workflow as wf
    from fakes import FakeComfy
    src = Path(__file__).resolve().parent.parent / "workflows"
    (tmp_path / "workflows").mkdir()
    for f in ("example_workflow.json", "nodes.example.json"):
        shutil.copy(src / f, tmp_path / "workflows" / f)
    monkeypatch.setattr(wf, "_lora_root", lambda: None)
    monkeypatch.setattr(gen, "ComfyClient", FakeComfy)
    FakeComfy.uploads, FakeComfy.graphs = [], []

    def b64(color):
        buf = io.BytesIO()
        Image.new("RGB", (64, 96), color).save(buf, "PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    gen.JOBS["t"] = {"id": "t"}
    gen._run("t", {"comfy_url": "x", "judge": {}}, {
        "positive": "1girl", "steps": 20, "subject_b64": b64("red"), "style_b64": b64("blue"),
        "style_text": "ink wash", "subject_ip": True, "style_ip": True, "size": [832, 1216]},
        tmp_path, lambda p: str(p), lambda: False)
    assert gen.JOBS["t"]["status"] == "done", gen.JOBS["t"].get("error")
    rec = json.loads(next((tmp_path / "runs" / "manual").glob("*/run.json")).read_text())
    return gen.JOBS["t"], rec, FakeComfy


def test_two_ip_adapters_are_scaled_to_the_cap():
    from ouroboros.targets import cap_ip_weights
    ipa = [{"role": "subject", "weight": 0.6}, {"role": "style", "weight": 0.5}]
    note = cap_ip_weights(ipa, 0.6)
    assert [a["weight"] for a in ipa] == [0.33, 0.27] and "scaled" in note and "1.1" in note


def test_one_adapter_or_a_total_under_the_cap_is_left_alone():
    from ouroboros.targets import cap_ip_weights
    one = [{"role": "subject", "weight": 0.9}]
    pair = [{"role": "subject", "weight": 0.3}, {"role": "style", "weight": 0.3}]
    assert cap_ip_weights(one, 0.6) is None and one[0]["weight"] == 0.9
    assert cap_ip_weights(pair, 0.6) is None and [a["weight"] for a in pair] == [0.3, 0.3]
    assert cap_ip_weights([{"weight": 0.6}, {"weight": 0.5}], 0) is None  # 0 switches the cap off


def test_a_home_render_with_both_adapters_is_scaled_and_says_so(home_run_targets):
    job, rec, comfy = home_run_targets
    ips = [n for n in comfy.graphs[-1].values() if n["class_type"] == "IPAdapterAdvanced"]
    assert sorted(n["inputs"]["weight"] for n in ips) == [0.27, 0.33]  # 0.6 + 0.5 -> 0.6 in all
    assert "scaled" in rec["ip_scaled"] and rec["ip_scaled"] in job.get("warning")
    assert [a["weight"] for a in rec["ipadapter"]] == [0.33, 0.27]


def test_missing_targets_are_judged_against_the_goal_not_the_reference(tmp_path):
    ref = img(tmp_path / "cassandra.png")
    tg = Targets(subject=Target(ref), goal_for_missing=True)
    assert tg.split and tg.missing() == ["style", "pose"]
    texts = [p.get("text", "[image]") for p in tmod.judge_parts(tg, ref)]
    assert sum(t == "[image]" for t in texts) == 1                     # the character image, once
    assert any(t.startswith("STYLE TARGET") and "judge it against the GOAL" in t for t in texts)
    assert any(t.startswith("POSE TARGET") and "judge it against the GOAL" in t for t in texts)
    # without the flag, a one-image job is still the classic one-reference job
    assert not Targets(subject=Target(ref)).split


def test_a_pose_from_the_description_never_starts_from_the_reference():
    from ouroboros.params import allowed_modes
    assert "img2img_reference" in allowed_modes({})
    assert not [m for m in allowed_modes({"pose_from_goal": True}) if "reference" in m]


def test_a_character_job_with_a_new_pose_in_its_description(tmp_path, monkeypatch):
    """Home -> Run automatically with a saved character and "arms crossed": the judge must see
    the character as SUBJECT only, with style and pose taken from the goal."""
    import ouroboros.loop as loop
    import ouroboros.workflow as wf
    from fakes import FakeComfy
    from ouroboros.judge import Review
    from ouroboros.loop import run_job
    root = Path(__file__).resolve().parent.parent
    (tmp_path / "workflows").mkdir()
    for f in ("example_workflow.json", "nodes.example.json"):
        shutil.copy(root / "workflows" / f, tmp_path / "workflows" / f)
    monkeypatch.setattr(wf, "_lora_root", lambda: None)
    monkeypatch.setattr(loop, "write_prompt", lambda *a, **k: {
        "positive": "score_9, 1girl, gold armour, arms crossed", "negative": "lowres", "dropped": [], "notes": ""})
    seen = []

    class J:
        context_window, backend = 8000, None

        def rubric(self, penalize_extras):
            return {"composition": {"weight": 1, "anchors": "x"}}

        def review_each(self, *a, **k):
            seen.append({"targets": k.get("targets"), "modes": a[5], "rules": a[6], "rubric": a[7]})
            return Review([40.0] * len(a[4]), 0, "x", {"focus": "prompt", "mode": "img2img_reference"}, {}, 0.0)
        review = review_each

        def confirm(self, *a, **k):
            return Review([40.0], 0, "", {}, {}, 0.0)

        def summarize(self, previous, entries):
            return previous, 0.0
    cfg = json.loads((root / "config.example.json").read_text())
    cfg["loop"].update(max_rounds=2, threshold=99, hand_refine=False, auto_fix=False, upscale=False,
                       ai_settings=False, batch_start=2, batch_end=2, plateau_rounds=10, local_prefilter=False)
    cfg["loras"]["mode"] = "off"
    folder = Queue(tmp_path / "jobs").add("cass", None, None, {"description": "this character, arms crossed"},
                                          {"subject": ("me.png", img(tmp_path / "me.png").read_bytes())})
    FakeComfy.uploads, FakeComfy.graphs = [], []
    run_job(load_job(folder), cfg, FakeComfy("x"), wf.Workflows(tmp_path / "workflows"), J(),
            ["euler"], ["normal"], tmp_path / "runs", report=lambda e: None)
    assert seen and all(s["targets"].goal_for_missing for s in seen)
    assert not [m for s in seen for m in s["modes"] if "reference" in m]
    assert "GOAL" in seen[0]["rubric"]["composition"]["anchors"] and "from the GOAL" in seen[0]["rules"]
    rows = [json.loads(l) for f in (tmp_path / "runs").glob("*/log.jsonl") for l in f.read_text().splitlines()]
    modes = [p["mode"] for r in rows if r.get("type") == "round" for p in r["params"]]
    assert len(modes) >= 4 and "img2img_reference" not in modes, modes  # though the judge asked for it


def test_the_character_adapter_is_standard_and_cut_out_the_style_plus(home_run_targets):
    job, rec, comfy = home_run_targets
    loaders = {n["inputs"]["preset"] for n in comfy.graphs[-1].values() if n["class_type"] == "IPAdapterUnifiedLoader"}
    assert [a["preset"] for a in rec["ipadapter"]] == ["STANDARD (medium strength)", "PLUS (high strength)"]
    assert loaders == {"STANDARD (medium strength)", "PLUS (high strength)"}
    # the fake ComfyUI has no remover: the image is used as it is, and the page is told why
    assert not rec.get("subject_cutout") and "background" in job["warning"] and "scaled" in job["warning"]


def run_character_job(tmp_path, monkeypatch, description, settings=None, named=None, style_image=False,
                      overrides=None):
    """Home -> Run automatically with a character image (and maybe a style image): two rounds
    against fakes. Returns (what the judge was given per call, the IP-Adapter nodes of the last
    render, the images cut out, the run's run.json)."""
    import ouroboros.loop as loop
    import ouroboros.workflow as wf
    from fakes import FakeComfy
    from ouroboros.judge import Review
    root = Path(__file__).resolve().parent.parent
    (tmp_path / "workflows").mkdir(exist_ok=True)
    for f in ("example_workflow.json", "nodes.example.json"):
        shutil.copy(root / "workflows" / f, tmp_path / "workflows" / f)
    monkeypatch.setattr(wf, "_lora_root", lambda: None)
    written = []
    monkeypatch.setattr(loop, "write_prompt", lambda *a, **k: written.append(k.get("targets")) or {
        "positive": "score_9, 1girl", "negative": "lowres", "dropped": [], "notes": ""})
    if named is None:
        def fail(backend, description):
            raise RuntimeError("LLM down")
        monkeypatch.setattr(loop, "names_style", fail)
    else:
        monkeypatch.setattr(loop, "names_style", lambda b, d: (named, "flat colours" if named else "", 0.0))
    cut = []
    monkeypatch.setattr(loop.nobg_mod, "cutout", lambda image, **k: cut.append(Path(image).name) or image)
    seen = []

    class J:
        context_window, backend = 8000, None

        def rubric(self, penalize_extras):
            return {"style": {"weight": 1, "anchors": "x"}}

        def review_each(self, *a, **k):
            seen.append({"targets": k.get("targets"), "rules": a[6]})
            return Review([40.0] * len(a[4]), 0, "x", {"focus": "prompt"}, {}, 0.0)
        review = review_each

        def confirm(self, *a, **k):
            return Review([40.0], 0, "", {}, {}, 0.0)

        def summarize(self, previous, entries):
            return previous, 0.0
    cfg = json.loads((root / "config.example.json").read_text())
    cfg["loop"].update(max_rounds=1, threshold=99, hand_refine=False, auto_fix=False, upscale=False,
                       ai_settings=False, batch_start=1, batch_end=1, local_prefilter=False)
    cfg["loras"]["mode"] = "off"
    images = {"subject": ("me.png", img(tmp_path / "me.png").read_bytes())}
    if style_image:
        images["style"] = ("look.png", img(tmp_path / "look.png", "blue").read_bytes())
    folder = Queue(tmp_path / "jobs").add("cass", None, None, {"description": description,
                                                               "settings": settings or {}, **(overrides or {})},
                                          images)
    FakeComfy.uploads, FakeComfy.graphs = [], []
    res = loop.run_job(load_job(folder), cfg, FakeComfy("x"), wf.Workflows(tmp_path / "workflows"), J(),
                       ["euler"], ["normal"], tmp_path / "runs", report=lambda e: None)
    ipa = [n for n in FakeComfy.graphs[-1].values() if n["class_type"] in ("IPAdapterAdvanced", "IPAdapterUnifiedLoader")]
    return seen, ipa, cut, json.loads((res.run_dir / "run.json").read_text()), written


def test_with_no_style_named_the_character_image_is_the_style_target(tmp_path, monkeypatch):
    seen, ipa, _, info, written = run_character_job(tmp_path, monkeypatch, "this character, arms crossed", named=False)
    tg = seen[0]["targets"]
    assert tg.style.image == tg.subject.image and tg.missing() == ["pose"] and info["style_from"] == "character image"
    # one IP-Adapter (the character's): the same image isn't carried twice
    assert sum(n["class_type"] == "IPAdapterAdvanced" for n in ipa) == 1
    assert sum("image" in p for p in written[0]) == 1 and "same character and style" in seen[0]["rules"]


def test_a_style_the_description_names_is_the_aim(tmp_path, monkeypatch):
    seen, _, _, info, _ = run_character_job(tmp_path, monkeypatch, "this character in flat colours", named=True)
    tg = seen[0]["targets"]
    assert tg.style.image is None and "style" in tg.missing() and info["style_from"] == "description"
    assert "art style the GOAL describes" in seen[0]["rules"]


def test_if_the_style_check_fails_the_description_still_decides(tmp_path, monkeypatch):
    seen, _, _, info, _ = run_character_job(tmp_path, monkeypatch, "this character, arms crossed", named=None)
    assert seen[0]["targets"].style.image is None and info["style_from"] == "description"


def test_run_automatically_uses_the_home_forms_ip_adapter_settings(tmp_path, monkeypatch):
    settings = {"subject_ip": True, "subject_ip_weight": 0.45, "ip_preset": "PLUS FACE (portraits)",
                "subject_cutout": False, "style_ip": False, "style_ip_weight": 0.5}
    _, ipa, cut, info, _ = run_character_job(tmp_path, monkeypatch, "this character", settings, True,
                                             style_image=True)
    assert [n["inputs"]["weight"] for n in ipa if n["class_type"] == "IPAdapterAdvanced"] == [0.45]  # no style one
    assert [n["inputs"]["preset"] for n in ipa if n["class_type"] == "IPAdapterUnifiedLoader"] == ["PLUS FACE (portraits)"]
    assert cut == [] and not info.get("subject_cutout")


def test_without_home_settings_a_job_uses_the_config(tmp_path, monkeypatch):
    _, ipa, cut, info, _ = run_character_job(tmp_path, monkeypatch, "this character", None, True, style_image=True)
    assert [n["inputs"]["preset"] for n in ipa if n["class_type"] == "IPAdapterUnifiedLoader"] == [
        "STANDARD (medium strength)", "PLUS (high strength)"]
    assert cut == ["subject.png"] and info["subject_cutout"] and info["ip_scaled"]  # 0.6 + 0.5 > 0.6


def test_the_lora_picker_matches_a_described_style_without_the_character_image(tmp_path):
    from fakes import FakeLibrary
    from ouroboros.lora_picker import pick_loras
    ex = img(tmp_path / "ex.png", "green")
    index = {f"L{i}.safetensors": {"name": f"L{i}.safetensors", "title": f"L{i}", "base_model": "Pony",
                                   "trigger_words": [], "examples": [{"thumb": str(ex), "prompt": "p"}]}
             for i in range(3)}
    lib = FakeLibrary(index)
    lib.card = lambda r, detail=True: r["title"]
    calls = []

    class B:
        def complete(self, instructions, parts, schema, name, side):
            calls.append(parts)
            return {"picks": [{"lora": "L1", "strength": 0.8, "why": "flat"}], "alternatives": [], "notes": ""}, 0, 0
    for sheet in (False, True):
        calls.clear()
        out = pick_loras(B(), lib, None, "a knight", "Pony", 2, {"contact_sheet": sheet}, style_text="Incase style")
        texts = [p.get("text", "") for p in calls[-1]]
        assert out["picks"][0][0] == "L1.safetensors" and any("Incase style" in t for t in texts)
        assert not any(p.get("image") == ex.parent / "me.png" for p in calls[-1])
        assert not any("REFERENCE IMAGE" in t for t in texts)


@pytest.mark.parametrize("named", [True, False])
def test_auto_lora_picks_follow_the_style_rule(tmp_path, monkeypatch, named):
    import ouroboros.loop as loop
    got = {}

    def fake_pick(backend, library, reference, goal, base, n, cfg, style_text="", context=None):
        got.update(reference=reference, style_text=style_text, context=context)
        raise RuntimeError("stop here")  # the loop falls back to the workflow's LoRAs
    monkeypatch.setattr(loop, "pick_loras", fake_pick)
    from fakes import FakeLibrary
    lib = FakeLibrary({"L.safetensors": {"name": "L.safetensors", "title": "L", "trigger_words": []}})
    lib.record_results = lambda results: None
    lib.card = lambda r, detail=True: r["title"]
    real_run = loop.run_job
    monkeypatch.setattr(loop, "run_job", lambda *a, **k: real_run(*a, **{**k, "library": lib}))
    # run_character_job switches LoRAs off in the config; the job's own lora_mode wins
    run_character_job(tmp_path, monkeypatch, "this character in flat colours", {"_": 1}, named,
                      overrides={"lora_mode": "auto"})
    if named:  # the character image still goes along, labelled as the subject, not the style
        assert got["reference"] is None and got["style_text"] == "flat colours"
        assert [(r, Path(i).name) for r, i, _ in got["context"]] == [("subject", "subject.png")]
    else:  # it is the style reference already: not sent twice
        assert got["reference"].name == "subject.png" and got["style_text"] == "" and got["context"] == []


def test_the_loop_lora_picker_sees_the_subject_and_pose_images(tmp_path):
    from fakes import FakeLibrary
    from ouroboros.lora_picker import pick_loras
    ex, style, subj, pose = (img(tmp_path / f"{n}.png", c) for n, c in
                             (("ex", "green"), ("style", "red"), ("subj", "blue"), ("pose", "white")))
    index = {f"L{i}.safetensors": {"name": f"L{i}.safetensors", "title": f"L{i}", "base_model": "Pony",
                                   "trigger_words": [], "examples": [{"thumb": str(ex), "prompt": "p"}]}
             for i in range(8)}  # more than the visual shortlist: two calls
    lib = FakeLibrary(index)
    lib.card = lambda r, detail=True: r["title"]
    calls = []

    class B:
        def complete(self, instructions, parts, schema, name, side):
            calls.append(parts)
            if name == "lora_shortlist":
                return {"shortlist": ["L1", "L2"], "why": ""}, 0, 0
            return {"picks": [{"lora": "L1", "strength": 0.8, "why": ""}], "alternatives": [], "notes": ""}, 0, 0
    context = [("subject", subj, "red hair"), ("pose", pose, "kneeling")]
    pick_loras(B(), lib, style, "a knight", "Pony", 2, {}, context=context)
    for parts in calls:  # shortlist and pick: both labelled images
        images = [p["image"] for p in parts if "image" in p]
        texts = " ".join(p.get("text", "") for p in parts)
        assert style in images and subj in images and pose in images
        assert "SUBJECT IMAGE" in texts and "red hair" in texts and "POSE IMAGE" in texts
    calls.clear()  # one image per message: the grid carries them, the shortlist only their tags
    pick_loras(B(), lib, style, "a knight", "Pony", 2, {"contact_sheet": True}, context=context)
    assert all(sum("image" in p for p in parts) == 1 for parts in calls)
    assert "the REFERENCE, the SUBJECT, the POSE, then" in " ".join(p.get("text", "") for p in calls[-1])
