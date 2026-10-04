import json
import shutil
from pathlib import Path

import pytest

import ouroboros.workflow as wf
from ouroboros.loras import with_triggers
from ouroboros.params import GenParams, norm_tag, split_tags

from fakes import FakeLibrary

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def flows(tmp_path, monkeypatch):
    """The shipped example workflow, loaded through the nodes.example.json fallback."""
    for f in ("example_workflow.json", "nodes.example.json"):
        shutil.copy(ROOT / "workflows" / f, tmp_path / f)
    monkeypatch.setattr(wf, "_lora_root", lambda: None)
    return wf.Workflows(tmp_path)


def test_example_workflow_loads_without_nodes_json(flows):
    assert flows.spec["file"] == "example_workflow.json" and flows.default_loras() == ()


def test_safety_negative_is_always_added_once(flows):
    g = flows.build(GenParams(positive="1girl", negative="lowres, (loli:1.2)"), "x.png")
    neg = g[flows.spec["roles"]["negative"][0]]["inputs"]["text"]
    tags = [norm_tag(t) for t in split_tags(neg)]
    for t in wf.SAFETY_NEGATIVE:
        assert tags.count(t) == 1
    assert wf.with_safety_negative("") == ", ".join(wf.SAFETY_NEGATIVE)


def test_lora_names_and_embeddings_use_this_os_separator(flows):
    g = flows.build(GenParams(positive="1girl", negative="embedding:Pony\\easyneg, bad \\(style\\)",
                              loras=(("Pony\\styles\\X.safetensors", 0.8),)), "x.png", checkpoint="Pony\\ck.safetensors")
    text = json.dumps(g)
    assert "Pony/styles/X.safetensors" in text and "embedding:Pony/easyneg" in text
    assert "bad \\\\(style\\\\)" in text  # prompt escapes untouched


def test_lora_paths_resolve_to_where_the_file_is(flows, tmp_path, monkeypatch):
    root = tmp_path / "loras"
    (root / "Pony" / "styles" / "x").mkdir(parents=True)
    (root / "Pony" / "styles" / "x" / "X.safetensors").write_bytes(b"")
    monkeypatch.setattr(wf, "_lora_root", lambda: root)
    import ouroboros.lora_catalog as lc
    monkeypatch.setattr(lc, "_files", {"root": None, "at": 0.0, "names": {}})
    g = flows.build(GenParams(positive="1girl", loras=(("Pony\\X.safetensors", 0.8),)), "x.png")
    assert "Pony/styles/x/X.safetensors" in json.dumps(g)


P = "score_9, 1girl, solo, standing, full body, looking at viewer, red dress, flat color, p0seA, Jabstyle_PNYV1.5"


def test_ordinary_tags_listed_as_triggers_are_kept(library):
    out = [norm_tag(t) for t in split_tags(with_triggers(P, (), library, split_tags, norm_tag, set()))]
    for t in ("1girl", "solo", "standing", "full body", "flat color"):
        assert t in out


def test_lora_specific_tokens_and_file_names_are_removed_when_off(library):
    out = [norm_tag(t) for t in split_tags(with_triggers(P, (), library, split_tags, norm_tag, set()))]
    assert norm_tag("p0seA") not in out and norm_tag("Jabstyle_PNYV1.5") not in out


def test_active_lora_trigger_is_added_and_user_tags_protected(library):
    jab = next(n for n in library.index() if "Jabstyle" in n)
    out = with_triggers("1girl, p0seA", ((jab, 0.8),), library, split_tags, norm_tag, {norm_tag("p0seA")})
    assert out.startswith("Jabstyle") and "p0seA" in out


def test_once_listed_ordinary_tags_and_plain_file_names_are_kept():
    lib = FakeLibrary({
        "Pony\\c\\Handjob.safetensors": {"trigger_words": ["Kiss", "69", "4girls", "1980s \\(style\\)", "score_9"]},
        "Pony\\c\\other.safetensors": {"trigger_words": ["wh0r3", "PuffyNips", "melkor_style"]},
    })
    p = "score_9, 1girl, kiss, 69, 4girls, 1980s \\(style\\), handjob, wh0r3, PuffyNips, melkor_style"
    out = [norm_tag(t) for t in split_tags(with_triggers(p, (), lib, split_tags, norm_tag, set()))]
    for t in ("score_9", "kiss", "69", "4girls", "1980s (style)", "handjob"):
        assert t in out
    for t in ("wh0r3", "puffynips", "melkor_style"):
        assert t not in out


@pytest.mark.parametrize("word,token", [("p0seA", True), ("RSV1.2", True), ("Jabstyle_PNYV1.5", True),
                                        ("PuffyNips", True), ("Kiss", False), ("4girls", False), ("69", False),
                                        ("score_8_up", False), ("source_anime", False), ("flat chest", False)])
def test_made_up_token(word, token):
    from ouroboros.loras import made_up_token
    assert made_up_token(word) is token
