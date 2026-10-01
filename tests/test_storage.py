"""API keys, thumbnails, LoRA paths and briefs: everything that reads or writes files."""
import json
import os
import time

import pytest
from PIL import Image

import ouroboros.keys as keys
import ouroboros.lora_briefs as briefs
import ouroboros.lora_catalog as catalog
import ouroboros.thumbs as thumbs


@pytest.fixture
def key_root(tmp_path, monkeypatch):
    monkeypatch.setattr(keys, "ROOT", tmp_path)
    monkeypatch.setattr(keys, "FILE", tmp_path / "api_keys.json")
    for p in keys.PLATFORMS.values():
        monkeypatch.delenv(p["env"], raising=False)
    return tmp_path


def test_keys_save_clear_and_never_return_the_key(key_root):
    keys.save("civitai", "abcdefgh1234")
    assert keys.lookup("civitai") == ("abcdefgh1234", "settings")
    st = keys.status()["civitai"]
    assert st["set"] and st["hint"] == "…1234" and "abcdefgh" not in json.dumps(st)
    assert oct(os.stat(keys.FILE).st_mode & 0o777) == "0o600"
    keys.save("civitai", "")
    assert keys.lookup("civitai") == ("", None)


def test_environment_wins_and_legacy_file_is_read_and_cleared(key_root, monkeypatch):
    (key_root / "deepinfra_key.txt").write_text("legacykey123\n")
    assert keys.lookup("deepinfra") == ("legacykey123", "deepinfra_key.txt")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "envkey")
    assert keys.lookup("deepinfra") == ("envkey", "environment")
    monkeypatch.delenv("DEEPINFRA_API_KEY")
    keys.save("deepinfra", "")
    assert not (key_root / "deepinfra_key.txt").exists()


def test_clearing_openai_never_touches_the_civitai_key_in_config(key_root):
    (key_root / "config.json").write_text(json.dumps({"loras": {"civitai_api_key": "civ123456"}}))
    keys.save("openai", "")
    assert keys.lookup("civitai") == ("civ123456", "config.json")


def make_png(path, color="red"):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (800, 1200), color).save(path)
    return path


def test_thumbnail_is_cached_regenerated_and_pruned(tmp_path):
    src = make_png(tmp_path / "runs" / "manual" / "r1" / "image_01.png")
    t = thumbs.get(tmp_path, "runs/manual/r1/image_01.png")
    assert t and max(Image.open(t).size) == thumbs.MAX_SIDE
    first = t.stat().st_mtime
    assert thumbs.get(tmp_path, "runs/manual/r1/image_01.png").stat().st_mtime == first
    time.sleep(1.1); make_png(src, "blue")
    assert thumbs.get(tmp_path, "runs/manual/r1/image_01.png").stat().st_mtime > first
    src.unlink()
    assert thumbs.prune(tmp_path) == 1 and not t.exists()


@pytest.mark.parametrize("rel", ["config.json", "runs/../config.json", "../outside.png", "cache/x.png"])
def test_thumbnails_only_for_images_under_runs(tmp_path, rel):
    (tmp_path / "config.json").write_text("{}")
    assert thumbs.get(tmp_path, rel) is None


def test_lora_example_thumbnails_reject_odd_ids(tmp_path):
    src = make_png(tmp_path / "ex.png")
    assert thumbs.lora_example(tmp_path, src, "../evil") is None
    assert thumbs.lora_example(tmp_path, src, "good_id").exists()


def test_current_name_finds_moved_loras(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "_files", {"root": None, "at": 0.0, "names": {}})
    f = tmp_path / "Pony" / "styles" / "artists" / "jab" / "Jabstyle.safetensors"
    f.parent.mkdir(parents=True); f.write_bytes(b"")
    assert catalog.current_name("Pony\\styles\\Jabstyle.safetensors", tmp_path) == "Pony\\styles\\artists\\jab\\Jabstyle.safetensors"
    assert catalog.current_name("Pony\\Missing.safetensors", tmp_path) == "Pony\\Missing.safetensors"


def test_briefs_attach_and_exclude_loras_made_for_minors(tmp_path, monkeypatch):
    monkeypatch.setattr(briefs, "_loaded", {"mtime": None, "data": {}})
    (tmp_path / "lora_briefs.json").write_text(json.dumps({
        "a": {"text": "[style] nice", "minors": False}, "b": {"text": "", "minors": True}}))
    items = [{"id": "a", "comfy_name": "A"}, {"id": "b", "comfy_name": "B"}, {"id": "c", "comfy_name": "C"}]
    out = catalog._with_briefs(items, tmp_path)
    assert [e["id"] for e in out] == ["a", "c"] and out[0]["brief"] == "[style] nice" and "brief" not in out[1]


def brief_fields(**kw):
    return {"kind": "style", "effect": "flat look.", "look": "", "best_for": "anything", "weak": "", "triggers": [" x ", ""],
            "weight_min": 1.2, "weight_usual": 2.0, "weight_max": 0.5, "weight_note": "", "rating": "sfw",
            "minors": False, **kw}


class BriefBackend:
    def __init__(self, fields):
        self.fields = fields

    def complete(self, *a, **k):
        return dict(self.fields), 0.0003, 0


def test_write_brief_normalizes_weights_and_text():
    f, _ = briefs.write_brief(BriefBackend(brief_fields()), "{}")
    assert (f["weight_min"], f["weight_usual"], f["weight_max"]) == (0.5, 1.2, 1.2)
    assert f["triggers"] == ["x"] and f["effect"] == "flat look"
    text = briefs.render(f)
    assert text.startswith("[style] flat look") and "trigger: x" in text and "weight 0.5-1.2" in text


def test_build_skips_caution_loras_and_unchanged_descriptions(tmp_path, monkeypatch):
    monkeypatch.setattr(briefs, "_loaded", {"mtime": None, "data": {}})
    import ouroboros.backends as backends
    monkeypatch.setattr(backends, "make_backend", lambda cfg: BriefBackend(brief_fields()))
    d = tmp_path / "catalog" / "descriptions"
    for rel, lid, st in (("styles/a/A.json", "a", "done"), ("caution loras/x/X.json", "x", "caution"),
                         ("styles/b/B.json", "b", "pending")):
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(json.dumps({"id": lid, "status": st, "name": lid, "summary": "s"}))
    cfg = {"judge": {"backend": "deepinfra", "deepinfra": {"model": "m"}}}
    res = briefs.build(cfg, tmp_path, tmp_path / "cache", workers=1, log=lambda m: None)
    assert res["written"] == 1 and set(json.loads((tmp_path / "cache" / "lora_briefs.json").read_text())) == {"a"}
    assert briefs.build(cfg, tmp_path, tmp_path / "cache", workers=1, log=lambda m: None)["written"] == 0


def test_thumbnail_of_a_dot_dot_path_stays_in_the_cache(tmp_path):
    root = tmp_path / "proj"
    make_png(root / "runs" / "manual" / "r1" / "image_01.png")
    t = thumbs.get(root, f"runs/../../{root.name}/runs/manual/r1/image_01.png")
    assert t is not None and t.resolve().is_relative_to(thumbs.cache_dir(root).resolve())
    assert not (root / "cache" / root.name).exists()


def test_suggested_loras_are_unique_and_within_the_limit():
    from ouroboros.lora_picker import suggest_loras

    class Backend:
        def complete(self, *a, **k):
            return {"picks": [{"n": 1, "strength": 0.8, "why": ""}, {"n": 1, "strength": 0.7, "why": ""},
                              {"n": 2, "strength": 0.6, "why": ""}, {"n": 3, "strength": 0.6, "why": ""}],
                    "notes": ""}, 0.0, 0
    cards = [{"comfy_name": c, "title": c, "weight": {"min": 0.2, "max": 1.2, "default": 0.8}} for c in "ABC"]
    out = suggest_loras(Backend(), cards, positive="1girl", max_loras=2)
    assert [p["comfy_name"] for p in out["picks"]] == ["A", "B"]
