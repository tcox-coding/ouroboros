"""Saved styles and characters (reflib), their "AI picks", and Home runs that use them."""
import json

import pytest
from PIL import Image

from ouroboros import reflib
from ouroboros.pose_picker import pick_item
from fakes import FakeComfy


class Backend:
    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def complete(self, instructions, parts, schema, name, max_side):
        self.calls.append((instructions, parts, schema, name))
        return self.answer, 0.01, 10


def test_add_names_tags_lists_and_removes(tmp_path):
    lib = reflib.RefLibrary(tmp_path, "style")
    name = lib.add("", Image.new("RGBA", (40, 60)), describe=lambda im: ("cel shading, flat colour", "Flat Cel"))
    assert name == "Flat Cel" and (tmp_path / "styles" / name / "source.png").is_file()
    assert lib.add("", Image.new("RGB", (8, 8)), fallback="my file?") == "my file"   # no LLM name: the file's
    assert lib.add("Flat Cel", Image.new("RGB", (8, 8))) == "Flat Cel_2"             # a typed name wins, numbered
    assert [p["name"] for p in lib.list()] == ["Flat Cel", "Flat Cel_2", "my file"]
    assert lib.get("Flat Cel")["description"] == "cel shading, flat colour"
    lib.remove("Flat Cel")
    assert (tmp_path / "styles" / "_removed" / "Flat Cel").is_dir() and len(lib.list()) == 2


@pytest.mark.parametrize("name", ["../x", "_removed", "nope", ""])
def test_names_outside_the_library_are_refused(tmp_path, name):
    lib = reflib.RefLibrary(tmp_path, "character")
    lib.add("knight", Image.new("RGB", (8, 8)))
    (tmp_path / "characters" / "_removed").mkdir()
    with pytest.raises(FileNotFoundError):
        lib.get(name)


def test_pick_item_answers_only_from_the_menu():
    items = [{"name": "red_knight", "description": "red hair, silver armour"},
             {"name": "fox_mage", "description": "fox ears, robe"}]
    b = Backend({"pick": "red_knight", "reason": "red hair"})
    got = pick_item(b, items, "character", description="a red-haired knight")
    assert got["pick"] == "red_knight" and got["considered"] == 2
    assert b.calls[0][2]["properties"]["pick"]["enum"] == ["red_knight", "fox_mage", "none"]
    assert "never substitute" in b.calls[0][0]
    assert pick_item(Backend({"pick": "made_up", "reason": ""}), items, "character", description="x")["pick"] is None
    assert pick_item(Backend({}), [], "style", description="x")["pick"] is None          # empty library: no call
    assert pick_item(Backend({}), items, "style")["reason"].startswith("no description")  # nothing to go on


def test_resolve_by_name_auto_and_none(tmp_path):
    reflib.RefLibrary(tmp_path, "character").add("knight", Image.new("RGB", (8, 8)), describe=lambda im: "red hair")
    got = reflib.resolve(tmp_path, "subject", "knight")
    assert got["image"].name == "source.png" and got["description"] == "red hair" and got["pick"] is None
    assert reflib.resolve(tmp_path, "subject", "") is None
    assert reflib.resolve(tmp_path, "subject", "missing") is None
    assert reflib.resolve(tmp_path, "pose", "knight") is None                    # poses have their own library
    assert reflib.resolve(tmp_path, "subject", "auto") is None                   # "auto" needs an LLM
    got = reflib.resolve(tmp_path, "subject", "auto", backend=Backend({"pick": "knight", "reason": "r"}),
                         description="a knight")
    assert got["name"] == "knight" and got["pick"]["reason"] == "r"
    none = reflib.resolve(tmp_path, "subject", "auto", backend=Backend({"pick": "none", "reason": "no match"}),
                          description="a wizard")
    assert none["image"] is None and none["pick"]["reason"] == "no match"


def _record(tmp_path):
    return json.loads(next((tmp_path / "runs" / "manual").glob("*/run.json")).read_text())


def test_home_run_with_a_saved_style_uses_its_image_and_tags(home_run, tmp_path):
    reflib.RefLibrary(tmp_path, "style").add("ink", Image.new("RGB", (64, 96), "blue"), describe=lambda im: "ink wash")
    home_run(style_library="ink", style_ip=True, size=[832, 1216])
    ips = [n for n in FakeComfy.graphs[-1].values() if n["class_type"] == "IPAdapterAdvanced"]
    assert [n["inputs"]["weight_type"] for n in ips] == ["style transfer"]
    rec = _record(tmp_path)
    assert rec["request"]["style_library"] == "ink" and rec["targets"]["style"]["text"] == "ink wash"
    assert "style_text" not in rec["request"] or not rec["request"]["style_text"]  # a rerun resolves the name again


def test_home_ai_picks_a_character_and_records_it(home_run, tmp_path, monkeypatch):
    import ouroboros.backends as backends
    reflib.RefLibrary(tmp_path, "character").add("knight", Image.new("RGB", (64, 96)), describe=lambda im: "red hair")
    monkeypatch.setattr(backends, "make_backend", lambda cfg: Backend({"pick": "knight", "reason": "red hair"}))
    job = home_run(subject_library="auto", subject_ip=True, description="a red-haired knight", size=[832, 1216])
    assert "knight" in job["library_notes"]
    rec = _record(tmp_path)
    assert rec["request"]["subject_library"] == "knight" and rec["subject_pick"]["reason"] == "red hair"
    assert rec["subject_reference"] == "subject.png"


def test_home_ai_finding_nothing_renders_without_it(home_run, tmp_path, monkeypatch):
    import ouroboros.backends as backends
    reflib.RefLibrary(tmp_path, "character").add("knight", Image.new("RGB", (64, 96)))
    monkeypatch.setattr(backends, "make_backend", lambda cfg: Backend({"pick": "none", "reason": "no match"}))
    job = home_run(subject_library="auto", subject_ip=True, description="a wizard", size=[832, 1216])
    assert "none fits" in job["library_notes"]
    assert not [n for n in FakeComfy.graphs[-1].values() if n["class_type"] == "IPAdapterAdvanced"]
    assert _record(tmp_path)["request"]["subject_library"] == ""
