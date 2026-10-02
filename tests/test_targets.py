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
    from conftest import FakeComfy
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
