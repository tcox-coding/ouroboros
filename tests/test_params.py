from ouroboros.params import (GenParams, apply_edit, apply_loras, drop_negative_conflicts, edit_prompt, lora_stem,
                              norm_tag, reseed_duplicates, split_tags, variants)

SAMPLERS, SCHEDULERS = ["euler", "dpmpp_2m"], ["karras", "normal"]


def test_lora_stem_handles_both_separators():
    assert lora_stem("Pony\\styles\\x\\Jabstyle_PNYV1.5.safetensors") == "Jabstyle_PNYV1.5"
    assert lora_stem("Pony/styles/x/FlatColor.safetensors") == "FlatColor"


def test_norm_tag_ignores_case_brackets_and_weights():
    assert norm_tag("(Red Shirt:1.3)") == norm_tag("red shirt")
    assert norm_tag("1990s \\(style\\)") == norm_tag("1990s (style)")


def test_edit_prompt_adds_and_removes_by_normalized_tag():
    out = edit_prompt("1girl, (red shirt:1.2), smile\n\noutdoors", ["blue hat"], ["red shirt"])
    tags = [norm_tag(t) for t in split_tags(out)]
    assert "red shirt" not in tags and "blue hat" in tags and "smile" in tags and "outdoors" in tags
    assert "\n\n" in out  # section layout kept


def test_negative_conflicts_are_dropped():
    neg, dropped = drop_negative_conflicts("1girl, thin dark belt", "lowres, belt, bad hands")
    assert "belt" not in [norm_tag(t) for t in split_tags(neg)]
    assert dropped


def test_apply_edit_turns_removing_an_unprompted_tag_into_a_negative():
    base = GenParams(positive="1girl, red dress", negative="lowres", seed=1)
    p = apply_edit(base, {"prompt_remove": ["green highlights"], "keep_seed": True}, SAMPLERS, SCHEDULERS)
    assert "green highlights" in p.negative and p.seed == 1


def test_apply_edit_mask_target_means_inpaint():
    p = apply_edit(GenParams(mode="txt2img"), {"mask_target": "left hand"}, SAMPLERS, SCHEDULERS)
    assert p.mode == "inpaint_best" and p.mask_target == "left hand"


def test_apply_loras_respects_allowed_pinned_and_zero_strength():
    allowed = {"jab": "J.safetensors", "flat": "F.safetensors"}
    p = GenParams(loras=(("pinned.safetensors", 1.0), ("J.safetensors", 0.9)))
    out = apply_loras(p, [{"lora": "flat", "strength": 0.7}, {"lora": "jab", "strength": 0}, {"lora": "nope", "strength": 1}],
                      allowed, 3)
    assert out.loras == (("pinned.safetensors", 1.0), ("F.safetensors", 0.7))
    assert apply_loras(p, None, allowed, 3) is p          # null = keep
    assert apply_loras(p, [], allowed, 3).loras == (("pinned.safetensors", 1.0),)  # [] = none


def test_variants_count_and_reseed_duplicates():
    p = GenParams(seed=5)
    v = variants(p, 6, "explore")
    assert len(v) == 6
    batch = reseed_duplicates([p, p, p], set())
    assert len({b.seed for b in batch}) == 3


def test_refine_variants_fill_the_batch():
    p = GenParams(mode="txt2img", seed=5, cfg=5.5)
    assert len(variants(p, 10, "refine")) == 10
    assert len({(v.seed, v.cfg) for v in variants(p, 10, "refine")}) == 10
