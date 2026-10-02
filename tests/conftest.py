"""Shared helpers. The tests need neither ComfyUI nor an LLM: those are replaced by small
fakes, and anything that writes files does so under pytest's tmp_path."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class FakeLibrary:
    """Stands in for LoraLibrary: index() is all with_triggers and the pickers read."""

    def __init__(self, index: dict):
        self._index = index

    def index(self) -> dict:
        return self._index


@pytest.fixture
def library():
    return FakeLibrary({
        "Pony\\styles\\artists\\jab\\Jabstyle_PNYV1.5.safetensors": {"trigger_words": ["Jabstyle"], "title": "Jab Style"},
        "Pony\\styles\\flat\\FlatColor.safetensors": {"trigger_words": ["flat color"], "title": "FlatColor"},
        "Pony\\concepts\\a\\poseA.safetensors": {"trigger_words": ["1girl", "standing", "p0seA"], "title": "Pose A"},
        "Pony\\concepts\\b\\poseB.safetensors": {"trigger_words": ["1girl", "full body"], "title": "Standing"},
    })


class FakeComfy:
    """Records uploads and graphs; every render returns one small PNG."""
    uploads: list = []
    graphs: list = []

    def __init__(self, url):
        pass

    def upload_image(self, path, subfolder="x"):
        from PIL import Image
        with Image.open(path) as im:
            FakeComfy.uploads.append((path.name, im.size, im.mode))
        return f"{subfolder}/{path.name}"

    def queue(self, graph):
        FakeComfy.graphs.append(graph)
        return "pid"

    def wait(self, ids, should_stop=None):
        return {"pid": {"outputs": {}}}

    def fetch_images(self, hist, node):
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, "PNG")
        return [buf.getvalue()]


@pytest.fixture
def home_run(tmp_path, monkeypatch):
    """_run against the example workflow and a fake ComfyUI."""
    import shutil
    import ouroboros.generate as gen
    import ouroboros.workflow as wf
    src = ROOT / "workflows"
    (tmp_path / "workflows").mkdir()
    for f in ("example_workflow.json", "nodes.example.json"):
        shutil.copy(src / f, tmp_path / "workflows" / f)
    monkeypatch.setattr(wf, "_lora_root", lambda: None)
    monkeypatch.setattr(gen, "ComfyClient", FakeComfy)
    FakeComfy.uploads, FakeComfy.graphs = [], []

    def run(**req):
        gen.JOBS["h"] = {"id": "h"}
        gen._run("h", {"comfy_url": "x", "judge": {}}, {"positive": "1girl", "steps": 20, **req}, tmp_path,
                 lambda p: str(p), lambda: False)
        assert gen.JOBS["h"]["status"] == "done", gen.JOBS["h"].get("error")
        return gen.JOBS["h"]
    return run
