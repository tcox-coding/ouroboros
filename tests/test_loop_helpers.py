from ouroboros.loop import (MAX_BATCH, batch_for_round, enforce_focus, lora_sweep, resolve_loras, set_variants,
                            weight_variants)
from ouroboros.params import GenParams

LC = {"batch_start": 12, "batch_end": 3}


def test_batch_shrinks_round_by_round_within_bounds():
    sizes = [batch_for_round(r, "explore", 50, 50, 80, LC) for r in range(1, 12)]
    assert sizes[0] == 12 and sizes[-1] >= 3
    assert all(a >= b for a, b in zip(sizes, sizes[1:]))


def test_batch_follows_phase_and_score_and_fixed_override():
    assert batch_for_round(2, "repair", 60, 50, 80, LC) == 3
    assert batch_for_round(2, "explore", 79, 50, 80, LC) <= 4
    assert batch_for_round(1, "explore", None, None, 80, LC, fixed=5) == 5
    assert batch_for_round(1, "explore", None, None, 80, {"batch_start": 99}) == MAX_BATCH


def test_batch_empty_settings_mean_defaults():
    assert batch_for_round(1, "explore", None, None, 80, {"batch_start": None, "batch_end": None}) == 12
    assert batch_for_round(1, "explore", None, None, 80, {"batch_start": 0, "batch_end": 0}) == 12


CUR = {"jab": 0.9}
ALLOWED = {"jab": "J", "flat": "F"}


def test_focus_prompt_drops_lora_changes():
    e = enforce_focus({"focus": "prompt", "prompt_add": ["hat"], "loras": [{"lora": "flat", "strength": 1}]}, CUR, ALLOWED, True)
    assert e["loras"] is None and e["prompt_add"] == ["hat"]


def test_focus_loras_drops_prompt_changes_and_allows_none():
    e = enforce_focus({"focus": "loras", "prompt_add": ["hat"], "loras": [{"lora": "flat", "strength": 0.7}]}, CUR, ALLOWED, True)
    assert e["prompt_add"] == [] and e["loras"] == [{"lora": "flat", "strength": 0.7}]
    assert enforce_focus({"focus": "loras", "loras": []}, CUR, ALLOWED, True)["loras"] == []


def test_late_switch_becomes_a_weight_change_of_the_current_set():
    e = enforce_focus({"focus": "loras", "loras": [{"lora": "flat", "strength": 0.7}]}, CUR, ALLOWED, False)
    assert e["focus"] == "lora_weights" and e["loras"] == [{"lora": "jab", "strength": 0.9}]


def test_weights_focus_keeps_unmentioned_loras():
    e = enforce_focus({"focus": "lora_weights", "loras": [{"lora": "jab", "strength": 0.6}]}, {"jab": 0.9, "flat": 0.5},
                      ALLOWED, False)
    assert {(l["lora"], l["strength"]) for l in e["loras"]} == {("jab", 0.6), ("flat", 0.5)}


def test_focus_is_inferred_and_weights_without_loras_fall_back():
    assert enforce_focus({"prompt_add": ["x"]}, CUR, ALLOWED, True)["focus"] == "prompt"
    assert enforce_focus({}, CUR, ALLOWED, True)["focus"] == "settings"
    assert enforce_focus({"focus": "lora_weights", "prompt_add": ["x"]}, {}, ALLOWED, True)["focus"] == "prompt"


def test_weight_variants_same_seed_within_range():
    p = GenParams(seed=7, loras=(("J", 0.9), ("pin", 1.0)))
    vs = weight_variants(p, 6, {"J"}, {"J": (0.5, 1.2)})
    assert len(vs) == 6 and vs[0] == p
    swept = [v for v in vs if v.seed == 7]
    assert all(0.5 <= dict(v.loras)["J"] <= 1.2 and dict(v.loras)["pin"] == 1.0 for v in swept)
    assert len({dict(v.loras)["J"] for v in swept}) == len(swept)


def test_set_variants_compare_previous_and_none_on_one_seed():
    p = GenParams(seed=3, loras=(("J", 0.9), ("pin", 1.0)))
    prev = GenParams(seed=3, loras=(("F", 0.7), ("pin", 1.0)))
    vs = set_variants(p, prev, 4, {"J", "F", "K"}, [("K", 0.8)], 3)
    sets = [tuple(sorted(n for n, _ in v.loras)) for v in vs]
    assert ("F", "pin") in sets and ("pin",) in sets and all(v.seed == 3 for v in vs)


def test_round_one_sweep_includes_no_loras_and_skips_picked_alternatives():
    p = GenParams(seed=1, loras=(("pin", 1.0), ("J", 0.8)))
    out = lora_sweep(p, [("J", 0.8), ("K", 0.6)], {"J", "K"}, 5, 3, baseline=(("W", 0.5),))
    sets = [tuple(n for n, _ in v.loras) for v in out]
    assert ("pin",) in sets and ("W",) in sets and ("pin", "K") in sets
    assert sum(1 for s in sets if s == ("pin", "J")) >= 1


def test_resolve_loras_by_file_name():
    index = {"Pony\\styles\\a\\X.safetensors": {}}
    installed = ["Pony/styles/a/X.safetensors", "Other/Y.safetensors"]
    logs = []
    out = resolve_loras((("Pony\\X.safetensors", 0.5), ("Y.safetensors", 0.4), ("Gone.safetensors", 1.0),
                         ("Pony\\styles\\a\\X.safetensors", 0.6)), index, installed, logs.append)
    assert out == (("Pony\\styles\\a\\X.safetensors", 0.5), ("Other/Y.safetensors", 0.4))
    assert any("Gone" in m for m in logs)
    assert resolve_loras((("Unknown.safetensors", 1.0),), {}, None, logs.append) == (("Unknown.safetensors", 1.0),)
