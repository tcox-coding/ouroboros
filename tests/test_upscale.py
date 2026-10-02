"""upscale: size maths, the two ComfyUI steps (model or Lanczos, then the detail pass through
the workflow), and the per-run record History reads. ComfyUI is faked."""
import io
import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from ouroboros import upscale as up
from ouroboros.params import GenParams


def png(size, color="red") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


class Comfy:
    """Returns an image of the size the graph asks for; records what it was sent."""

    def __init__(self):
        self.graphs, self.uploads = [], []

    def upload_image(self, path, subfolder="x"):
        with Image.open(path) as im:
            self.uploads.append((Path(path).name, im.size))
        return f"{subfolder}/{Path(path).name}"

    def queue(self, graph):
        self.graphs.append(graph)
        return "pid"

    def wait(self, ids, should_stop=None):
        return {"pid": {}}

    def fetch_images(self, hist, node):
        g = self.graphs[-1]
        if "4" in g and g["4"]["class_type"] == "ImageScale":  # the upscale-model graph
            return [png((g["4"]["inputs"]["width"], g["4"]["inputs"]["height"]), "blue")]
        return [png(self.uploads[-1][1], "green")]  # img2img keeps the loaded image's size


@pytest.fixture
def flows(tmp_path, monkeypatch):
    import ouroboros.workflow as wf
    src = Path(__file__).resolve().parent.parent / "workflows"
    (tmp_path / "workflows").mkdir()
    for f in ("example_workflow.json", "nodes.example.json"):
        shutil.copy(src / f, tmp_path / "workflows" / f)
    monkeypatch.setattr(wf, "_lora_root", lambda: None)
    return wf.Workflows(tmp_path / "workflows")


PARAMS = GenParams(positive="1girl", negative="lowres", seed=7, steps=30, cfg=6.0, loras=(("Pony\\x.safetensors", 0.8),))


@pytest.mark.parametrize("size,scale,cap,want", [
    ((1024, 1024), 1.5, 2560, (1536, 1536)),
    ((832, 1216), 2.0, 2560, (1664, 2432)),
    ((832, 1216), 3.0, 2048, (1400, 2048)),   # capped by the longest side
    ((1024, 1024), 0.5, 2560, (1024, 1024)),  # never smaller
    ((2048, 2048), 2.0, 1536, (2048, 2048)),  # already past the cap: unchanged, not shrunk
])
def test_target_size(size, scale, cap, want):
    w, h = up.target_size(size, scale, cap)
    assert (w, h) == want and w % 8 == 0 and h % 8 == 0


def test_without_a_model_lanczos_then_a_detail_pass_through_the_workflow(tmp_path, flows):
    src = tmp_path / "image_01.png"
    src.write_bytes(png((832, 1216)))
    comfy = Comfy()
    res = up.upscale(src, PARAMS, comfy=comfy, flows=flows, cfg={"upscale": {"scale": 1.5, "denoise": 0.3}},
                     checkpoint=None, positive="1girl, rendered", out_dir=tmp_path / "out",
                     upload=lambda p: comfy.upload_image(p))
    assert res["size"] == [1248, 1824] and res["from_size"] == [832, 1216] and res["model"] is None
    assert comfy.uploads == [("image_01_enlarged.png", (1248, 1824))]  # resized locally, then uploaded once
    g = comfy.graphs[-1]
    ks = next(n["inputs"] for n in g.values() if n["class_type"] == "KSampler")
    assert ks["denoise"] == 0.3 and ks["seed"] == 7 and ks["cfg"] == 6.0  # the image's own settings
    with Image.open(res["image"]) as im:
        assert im.size == (1248, 1824)


def test_with_a_model_comfyui_enlarges_first(tmp_path, flows):
    src = tmp_path / "best.png"
    src.write_bytes(png((1024, 1024)))
    comfy = Comfy()
    res = up.upscale(src, PARAMS, comfy=comfy, flows=flows,
                     cfg={"upscale": {"model": "4x-AnimeSharp.pth", "scale": 2, "denoise": 0.35}},
                     checkpoint=None, positive="1girl", out_dir=tmp_path / "out", upload=lambda p: comfy.upload_image(p))
    model_graph = comfy.graphs[0]
    assert model_graph["2"]["inputs"]["model_name"] == "4x-AnimeSharp.pth"
    assert (model_graph["4"]["inputs"]["width"], model_graph["4"]["inputs"]["height"]) == (2048, 2048)
    assert len(comfy.graphs) == 2 and res["model"] == "4x-AnimeSharp.pth"


def test_zero_denoise_skips_the_detail_pass(tmp_path, flows):
    src = tmp_path / "a.png"
    src.write_bytes(png((512, 512)))
    comfy = Comfy()
    res = up.upscale(src, PARAMS, comfy=comfy, flows=flows, cfg={"upscale": {"scale": 2, "denoise": 0}},
                     checkpoint=None, positive="x", out_dir=tmp_path / "out", upload=lambda p: comfy.upload_image(p))
    assert comfy.graphs == [] and Image.open(res["image"]).size == (1024, 1024)


def test_run_and_record_upscales_the_auto_fixed_version_and_keeps_the_original(tmp_path, flows):
    (tmp_path / "image_01.png").write_bytes(png((512, 512)))
    (tmp_path / "image_01_fixed.png").write_bytes(png((512, 512), "white"))
    (tmp_path / "autofix.json").write_text(json.dumps({"image_01.png": {"state": "done", "image": "image_01_fixed.png"}}))
    comfy = Comfy()
    res = up.run_and_record("image_01.png", PARAMS, run_dir=tmp_path, comfy=comfy, flows=flows,
                            cfg={"upscale": {"scale": 2}}, checkpoint=None, positive="x",
                            upload=lambda p: comfy.upload_image(p))
    assert res["final"].name == "image_01_fixed_upscaled.png" and res["final"].exists()
    rec = up.load_results(tmp_path)["image_01.png"]
    assert rec["state"] == "done" and rec["source"] == "image_01_fixed.png" and rec["size"] == [1024, 1024]
    assert Image.open(tmp_path / "image_01.png").size == (512, 512)  # untouched


def test_a_failure_is_recorded(tmp_path, flows):
    (tmp_path / "a.png").write_bytes(png((512, 512)))

    class Broken(Comfy):
        def fetch_images(self, hist, node):
            return []
    comfy = Broken()
    with pytest.raises(RuntimeError):
        up.run_and_record("a.png", PARAMS, run_dir=tmp_path, comfy=comfy, flows=flows, cfg={},
                          checkpoint=None, positive="x", upload=lambda p: comfy.upload_image(p))
    assert up.load_results(tmp_path)["a.png"]["state"] == "error"
