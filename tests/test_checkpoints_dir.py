"""The checkpoints folder chosen in Settings: ComfyUI is started with it, and the list says
which of its checkpoints a running ComfyUI hasn't loaded."""

import requests

from ouroboros import comfy_launcher, runner, server


def _launcher(tmp_path, folder):
    launcher = comfy_launcher.ComfyLauncher({"python": "py", "main": str(tmp_path / "c" / "main.py"),
                                             "args": [], "checkpoints_dir": folder},
                                            "http://127.0.0.1:8188", tmp_path / "logs")
    return launcher


def test_launch_adds_the_checkpoints_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(comfy_launcher, "detect_desktop_install", lambda: {})
    ckpts = tmp_path / "it's ckpts"
    ckpts.mkdir()
    cmd, _ = _launcher(tmp_path, str(ckpts)).launch_command()
    i = cmd.index("--extra-model-paths-config")
    text = (tmp_path / "logs" / "comfy_model_paths.yaml").read_text()
    assert cmd[i + 1] == str(tmp_path / "logs" / "comfy_model_paths.yaml")
    assert "checkpoints: '" + str(ckpts.resolve()).replace("'", "''") + "'" in text


def test_launch_without_a_folder_adds_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(comfy_launcher, "detect_desktop_install", lambda: {})
    for folder in ("", str(tmp_path / "missing")):
        cmd, _ = _launcher(tmp_path, folder).launch_command()
        assert "--extra-model-paths-config" not in cmd


def test_factory_passes_the_folder(monkeypatch):
    monkeypatch.setattr(runner, "_launchers", {})
    cfg = {"comfy_url": "http://x:1", "comfyui": {"autostart": False}, "checkpoints_dir": "/a"}
    assert runner.comfy_launcher(cfg).cfg == {"autostart": False, "checkpoints_dir": "/a", "loras_root": ""}
    cfg["checkpoints_dir"] = "/b"
    assert runner.comfy_launcher(cfg).cfg["checkpoints_dir"] == "/b"


def _cfg(folder):
    return {"comfy_url": "http://127.0.0.1:1", "checkpoints_dir": folder, "defaults": {}, "checkpoint_bases": {}}


def test_list_from_the_folder_when_comfy_is_down(tmp_path, monkeypatch):
    (tmp_path / "sub").mkdir()
    for name in ("a.safetensors", "sub/b.safetensors", "notes.txt"):
        (tmp_path / name).write_text("x")
    monkeypatch.setattr(server, "load_config", lambda: _cfg(str(tmp_path)))
    monkeypatch.setattr(server, "default_checkpoint", lambda: None)
    monkeypatch.setattr(server.requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError()))
    out = server.checkpoints()
    assert [c["name"] for c in out["items"]] == ["a.safetensors", "sub/b.safetensors"]
    assert out["folder_found"] is True and out["not_loaded"] == []


def test_not_loaded_lists_what_comfy_lacks(tmp_path, monkeypatch):
    for name in ("a.safetensors", "new.safetensors"):
        (tmp_path / name).write_text("x")

    class Reply:
        def json(self):
            return {"CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["a.safetensors", "z.safetensors"]]}}}}

    monkeypatch.setattr(server, "load_config", lambda: _cfg(str(tmp_path)))
    monkeypatch.setattr(server, "default_checkpoint", lambda: None)
    monkeypatch.setattr(server.requests, "get", lambda *a, **k: Reply())
    out = server.checkpoints()
    assert [c["name"] for c in out["items"]] == ["a.safetensors", "z.safetensors"]
    assert out["not_loaded"] == ["new.safetensors"]


def test_empty_or_missing_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "default_checkpoint", lambda: None)
    monkeypatch.setattr(server.requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError()))
    monkeypatch.setattr(server, "load_config", lambda: _cfg(""))
    assert server.checkpoints()["items"] == [] and server.checkpoints()["folder_found"] is None
    monkeypatch.setattr(server, "load_config", lambda: _cfg(str(tmp_path / "gone")))
    assert server.checkpoints()["folder_found"] is False


def test_set_workflow_checkpoint(tmp_path):
    import json
    from ouroboros.workflow import Workflows
    graph = {"4": {"inputs": {"ckpt_name": "old.safetensors"}, "class_type": "CheckpointLoaderSimple"},
             "6": {"inputs": {"text": "naïve"}, "class_type": "CLIPTextEncode"}}
    original = json.dumps(graph, indent=2, ensure_ascii=False)
    (tmp_path / "wf.json").write_text(original, encoding="utf-8")
    (tmp_path / "nodes.json").write_text(json.dumps({"file": "wf.json", "roles": {"checkpoint": ["4", "ckpt_name"]}}))
    Workflows.set_default(tmp_path, "checkpoint", "new.safetensors")
    Workflows.set_default(tmp_path, "checkpoint", "newer.safetensors")
    text = (tmp_path / "wf.json").read_text(encoding="utf-8")
    assert json.loads(text)["4"]["inputs"]["ckpt_name"] == "newer.safetensors"
    assert "naïve" in text and not text.endswith("\n")
    assert (tmp_path / "wf.orig.json").read_text(encoding="utf-8") == original  # the first original is kept
