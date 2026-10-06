"""Background removal of History images, and favourite runs."""
import io
import json

import pytest
from PIL import Image

from ouroboros import favorites, nobg

RUN = "20261004-120000_abc123"


def png(size=(32, 48), color="red", mode="RGB"):
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, "PNG")
    return buf.getvalue()


class Comfy:
    """Answers the remover graph with a mask: left half white (keep), right half black."""
    url = "http://x"

    def __init__(self):
        self.graphs = []

    def queue(self, graph):
        self.graphs.append(graph)
        return "p1"

    def wait(self, ids, should_stop=None):
        return {"p1": {}}

    def fetch_images(self, hist, node):
        m = Image.new("L", (16, 24), 0)
        m.paste(255, (0, 0, 8, 24))
        buf = io.BytesIO()
        m.convert("RGB").save(buf, "PNG")
        return [buf.getvalue()]


def run_folder(tmp_path):
    d = tmp_path / "runs" / "manual" / RUN
    d.mkdir(parents=True)
    (d / "image_01.png").write_bytes(png())
    return d


def test_the_mask_becomes_the_alpha_channel_at_full_size(tmp_path):
    d = run_folder(tmp_path)
    res = nobg.run_and_record("image_01.png", run_dir=d, comfy=Comfy(), model="birefnet.safetensors",
                              upload=lambda p: p.name)
    out = Image.open(d / "image_01_nobg.png")
    assert out.mode == "RGBA" and out.size == (32, 48)
    assert out.getpixel((2, 10))[3] == 255 and out.getpixel((30, 10))[3] == 0
    assert nobg.load_results(d)["image_01.png"]["state"] == "done" and res["source"] == "image_01.png"


def test_the_latest_version_is_used(tmp_path):
    d = run_folder(tmp_path)
    (d / "image_01_upscaled.png").write_bytes(png((64, 96)))
    (d / "upscale.json").write_text(json.dumps({"image_01.png": {"image": "image_01_upscaled.png"}}))
    nobg.run_and_record("image_01.png", run_dir=d, comfy=Comfy(), model="m", upload=lambda p: p.name)
    assert Image.open(d / "image_01_nobg.png").size == (64, 96)
    assert nobg.load_results(d)["image_01.png"]["source"] == "image_01_upscaled.png"


def test_graph_uses_comfys_remover():
    g, node = nobg.graph("x.png", "birefnet.safetensors")
    kinds = [n["class_type"] for n in g.values()]
    assert kinds == ["LoadImage", "LoadBackgroundRemovalModel", "RemoveBackground", "MaskToImage", "PreviewImage"]
    assert g[node]["class_type"] == "PreviewImage"


def test_a_missing_model_says_what_to_get():
    with pytest.raises(RuntimeError, match="birefnet.safetensors"):
        nobg.pick_model({}, [])
    assert nobg.pick_model({}, ["birefnet.safetensors", "other.safetensors"]) == "birefnet.safetensors"
    assert nobg.pick_model({}, ["only_one.safetensors"]) == "only_one.safetensors"


def test_favourites_name_rename_and_remove(tmp_path):
    run_folder(tmp_path)
    favorites.set(tmp_path, "manual/" + RUN)
    assert favorites.load(tmp_path)["manual/" + RUN]["name"] == ""
    favorites.set(tmp_path, "manual/" + RUN, "Gold knight, final")
    favorites.set(tmp_path, "manual/" + RUN)  # starring again keeps the name
    assert favorites.load(tmp_path)["manual/" + RUN]["name"] == "Gold knight, final"
    favorites.set(tmp_path, "manual/" + RUN, "")  # cleared: back to the run's own title
    assert favorites.load(tmp_path)["manual/" + RUN]["name"] == ""
    favorites.remove(tmp_path, "manual/" + RUN)
    assert favorites.load(tmp_path) == {}


@pytest.mark.parametrize("run", ["manual/../../x", "../x", "manual/missing", "_removed", "manual", ""])
def test_only_history_entries_can_be_favourites(tmp_path, run):
    run_folder(tmp_path)
    (tmp_path / "runs" / "_removed").mkdir()
    with pytest.raises(FileNotFoundError):
        favorites.set(tmp_path, run)


def test_a_favourite_cant_be_deleted_and_is_listed_with_its_settings(tmp_path, monkeypatch):
    from ouroboros import runner, server
    d = run_folder(tmp_path)
    (d / "run.json").write_text(json.dumps({"images": ["image_01.png"], "positive": "score_9, knight",
                                            "request": {"steps": 30}, "status": "done"}))
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "ROOT", tmp_path)  # rel_url
    favorites.set(tmp_path, "manual/" + RUN, "Knight")
    with pytest.raises(RuntimeError, match="favorite"):
        server.remove_run("manual/" + RUN)
    fav = server.favorites_list()[0]
    assert fav["fav_name"] == "Knight" and fav["positive"] == "score_9, knight" and fav["request"] == {"steps": 30}
    favorites.remove(tmp_path, "manual/" + RUN)
    server.remove_run("manual/" + RUN)
    assert not d.exists()


def test_a_cutout_is_on_grey_and_made_once_per_image(tmp_path, monkeypatch):
    monkeypatch.setattr(nobg, "available_models", lambda comfy: ["birefnet.safetensors"])
    src = tmp_path / "char.png"
    Image.new("RGB", (32, 48), (230, 200, 170)).save(src)  # a beige backdrop everywhere
    comfy = Comfy()
    a = nobg.cutout(src, comfy=comfy, cfg={}, cache_dir=tmp_path / "cut", upload=lambda p: p.name)
    b = nobg.cutout(src, comfy=comfy, cfg={}, cache_dir=tmp_path / "cut", upload=lambda p: p.name)
    im = Image.open(a)
    assert a == b and len(comfy.graphs) == 1 and im.mode == "RGB" and im.size == (32, 48)
    assert im.getpixel((2, 10)) == (230, 200, 170) and im.getpixel((30, 10)) == nobg.CUTOUT_GREY


def test_background_removal_needs_no_recorded_settings(tmp_path, monkeypatch):
    """A failed automatic run has a best.png but nothing auto-fix or upscale could reuse."""
    from ouroboros import generate as gen
    d = tmp_path / "runs" / RUN
    d.mkdir(parents=True)
    (d / "best.png").write_bytes(png())

    def no_settings(run):
        raise RuntimeError("this run has no recorded settings to repaint with")
    monkeypatch.setattr(gen, "_CTX", dict(gen._CTX))  # the server's settings come back afterwards
    gen.configure(root=tmp_path, rel_url=str, load_config=dict, prepare=lambda: None, fix_target=no_settings)
    queued = []
    monkeypatch.setattr(gen, "_enqueue", lambda kind, req, **f: queued.append((kind, req)) or "id")
    assert gen.start_nobg(RUN, ["best.png", "../x.png"]) == "id"
    assert queued == [("nobg", {"run": RUN, "images": ["best.png"]})]
    with pytest.raises(FileNotFoundError):
        gen.start_nobg("../" + RUN, ["best.png"])


def test_warnings_add_up_without_repeating():
    from ouroboros import generate as gen
    gen._set("w1", status="running")
    gen._warn("w1", "pose control skipped")
    gen._warn("w1", "IP-Adapter weights scaled")
    gen._warn("w1", "pose control skipped")
    assert gen.status("w1")["warning"] == "pose control skipped · IP-Adapter weights scaled"
    with gen._LOCK:
        gen.JOBS.pop("w1")


def test_a_rerun_uses_the_ip_adapters_as_they_rendered(tmp_path):
    """Weights after the cap, each role's preset, and whether the background was removed:
    the defaults changed since (STANDARD, the 0.6 cap, the cut-out), the entry didn't."""
    from ouroboros import generate as gen
    d = run_folder(tmp_path)
    (d / "subject.png").write_bytes(png())
    (d / "run.json").write_text(json.dumps({
        "positive": "knight", "request": {"subject_ip": True, "style_ip": True, "subject_ip_weight": 0.6,
                                          "style_ip_weight": 0.5, "ip_preset": "PLUS (high strength)",
                                          "subject_cutout": True},
        "subject_reference": "subject.png",
        "ipadapter": [{"role": "subject", "preset": "PLUS (high strength)", "weight": 0.33},
                      {"role": "style", "preset": "PLUS (high strength)", "weight": 0.27}]}))
    req = gen.rerun_request(tmp_path, "manual/" + RUN)
    assert (req["subject_ip_weight"], req["style_ip_weight"]) == (0.33, 0.27)
    assert req["ip_preset"] == req["style_ip_preset"] == "PLUS (high strength)"
    assert req["_ip_as_recorded"] and req["subject_cutout"] is False  # asked for, but it didn't happen


def test_a_character_image_is_padded_to_a_square_in_its_own_backdrop_colour(tmp_path):
    from PIL import Image
    from ouroboros.nobg import square_for_ip
    tall = tmp_path / "tall.png"
    im = Image.new("RGB", (100, 300), (128, 128, 128))
    im.paste((255, 0, 0), (40, 0, 60, 300))  # the character, top to bottom
    im.save(tall)
    out = square_for_ip(tall, tmp_path / "cache")
    with Image.open(out) as sq:
        assert sq.size == (300, 300) and sq.getpixel((5, 150)) == (128, 128, 128)
        assert sq.getpixel((150, 2)) == (255, 0, 0) and sq.getpixel((150, 297)) == (255, 0, 0)  # head and feet kept
    assert square_for_ip(tall, tmp_path / "cache") == out  # cached
    square = tmp_path / "sq.png"
    Image.new("RGB", (64, 64)).save(square)
    assert square_for_ip(square, tmp_path / "cache") == square
