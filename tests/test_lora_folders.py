"""Each checkpoint gets the LoRA folder named for its model, and a folder the classifier
hasn't catalogued yet is still in the library."""

import json
from pathlib import Path

from ouroboros import comfy_launcher, loras
from ouroboros.loras import LoraLibrary, compatible, in_folders, lora_folders_for


def _root(tmp_path):
    root = tmp_path / "loras"
    for d in ("Pony/styles", "NoobAI-XL", "Inbox"):
        (root / d).mkdir(parents=True)
    return root


def _cfg(root, **loras_cfg):
    return {"loras": {"comfy_root": str(root), **loras_cfg}, "checkpoint_bases": {}}


def test_folder_by_model_then_family(tmp_path):
    cfg = _cfg(_root(tmp_path))
    assert lora_folders_for("noobaiXL_vPred10.safetensors", cfg) == ["NoobAI-XL"]
    assert lora_folders_for("ponyDiffusionV6XL.safetensors", cfg) == ["Pony"]
    assert lora_folders_for("autismmixSDXL.safetensors", cfg) == ["Pony"]
    # no Illustrious folder: the NoobAI one, NoobAI being Illustrious-based
    assert lora_folders_for("waiIllustriousSDXL.safetensors", cfg) == ["NoobAI-XL"]
    assert lora_folders_for("mystery.safetensors", cfg) is None
    assert lora_folders_for("", cfg) is None and lora_folders_for("x", {"loras": {}}) is None


def test_config_override(tmp_path):
    cfg = _cfg(_root(tmp_path), checkpoint_folders={"mystery": ["Inbox", "Pony"]})
    assert lora_folders_for("mystery_v2.safetensors", cfg) == ["Inbox", "Pony"]


def test_in_folders_and_noob_compatibility():
    assert in_folders("NoobAI-XL\\a.safetensors", ["NoobAI-XL"])
    assert not in_folders("Pony\\styles\\a.safetensors", ["NoobAI-XL"])
    assert in_folders("anything.safetensors", None)
    assert compatible("NoobAI", "Illustrious") is True and compatible("NoobAI", "Pony") is False


def _state(cat, name, status="done"):
    (cat / "state").mkdir(parents=True, exist_ok=True)
    (cat / "state" / f"{Path(name).stem}.json").write_text(json.dumps(
        {"status": status, "meta": {"filename": Path(name.replace("\\", "/")).name, "comfy_name": name}}))


def test_unclassified_folder_is_in_the_library(tmp_path, monkeypatch):
    monkeypatch.setattr(loras, "_offline", {})
    root, cat = _root(tmp_path), tmp_path / "catalog"
    for f in ("Pony/styles/a.safetensors", "Pony/caution loras/c.safetensors", "NoobAI-XL/n.safetensors"):
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_bytes(b"\x02\x00\x00\x00\x00\x00\x00\x00{}")
    _state(cat, "Pony\\styles\\a.safetensors")
    _state(cat, "Pony\\caution loras\\c.safetensors", status="caution")
    lib = LoraLibrary({"comfy_root": str(root), "catalog_dir": str(cat)}, tmp_path / "cache")
    assert [d.name for d in lib.unclassified_dirs()] == ["Inbox", "NoobAI-XL"]
    extra = lib.unclassified()
    assert list(extra) == ["NoobAI-XL\\n.safetensors"]  # never the caution LoRA
    assert [p.name for p in lib.files()] == ["n.safetensors"]  # a refresh reads only these


def test_comfyui_gets_the_loras_root(tmp_path, monkeypatch):
    monkeypatch.setattr(comfy_launcher, "detect_desktop_install", lambda: {})
    root = _root(tmp_path)
    launcher = comfy_launcher.ComfyLauncher({"python": "py", "main": str(tmp_path / "m.py"), "args": [],
                                             "loras_root": str(root)}, "http://127.0.0.1:8188", tmp_path / "logs")
    cmd, _ = launcher.launch_command()
    text = Path(cmd[cmd.index("--extra-model-paths-config") + 1]).read_text()
    assert f"loras: '{root.resolve()}'" in text and "checkpoints" not in text
