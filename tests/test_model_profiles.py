import json

from ouroboros import model_profiles as profiles


def test_measured_settings_override_builtins_only_for_exact_model(tmp_path, monkeypatch):
    monkeypatch.setattr(profiles, "CACHE", tmp_path / "profiles.json")
    profile = {"label": "Measured", "settings": {"judge": {"deepinfra": {"max_tokens": 1234}}},
               "thresholds": {"loop": 72}}
    profiles.save_profile("Qwen/test", "deepinfra", profile)
    assert profiles.preset("Qwen/test", "deepinfra") == profile
    assert profiles.preset("Qwen/test-other", "deepinfra") != profile
    assert json.loads(profiles.CACHE.read_text())["version"] == 1


def test_roles_get_independent_options_and_do_not_mutate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(profiles, "CACHE", tmp_path / "profiles.json")
    profiles.save_profile("writer", "deepinfra", {"settings": {"judge": {"deepinfra": {"max_tokens": 8192}}}})
    cfg = {"backend": "deepinfra", "deepinfra": {"model": "judge", "max_tokens": 4096},
           "prompt_model": "writer", "prompt_options": {"temperature": 0.2}}
    resolved = profiles.role_config(cfg, "prompt")
    assert resolved["deepinfra"] == {"model": "writer", "max_tokens": 8192, "temperature": 0.2}
    assert cfg["deepinfra"]["model"] == "judge"


def test_confirmation_calibration_does_not_cross_rubrics(tmp_path, monkeypatch):
    monkeypatch.setattr(profiles, "CACHE", tmp_path / "profiles.json")
    profiles.save_profile("checker", "deepinfra", {"thresholds": {"loop": 68}})
    cfg = {"backend": "deepinfra", "confirm_model": "checker"}
    assert profiles.confirmation_threshold(cfg, 80) == 68
    assert profiles.confirmation_threshold(cfg, 85, "designer") == 85
    cfg["confirm_thresholds"] = {"loop": 73, "designer": 90}
    assert profiles.confirmation_threshold(cfg, 80) == 73
    assert profiles.confirmation_threshold(cfg, 85, "designer") == 90


def test_clearing_role_options_does_not_keep_stale_values(tmp_path, monkeypatch):
    from ouroboros import runner
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    (tmp_path / "config.example.json").write_text('{}')
    (tmp_path / "config.json").write_text(json.dumps({"judge": {"prompt_options": {"reasoning_effort": "low"}}}))
    assert runner.save_config({"judge": {"prompt_options": {}}})["judge"]["prompt_options"] == {}
