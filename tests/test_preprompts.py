"""Per-model prompt-writer guidelines: NoobAI ships its own, the rest use the general ones,
Settings can replace any of them, and the app's reply format always follows."""

from ouroboros import ckpt_info, preprompts
from ouroboros.prompter import CONTRACT, GUIDE, write_prompt

NOOB = "noobaiXLNAIXL_vPred10Version.safetensors"
PONY = "ponyDiffusionV6XL_v6StartWithThisOne.safetensors"


def test_defaults():
    assert preprompts.default("NoobAI-XL").startswith("# NoobAI XL Prompt Generation Guidelines")
    assert preprompts.default("Pony Diffusion V6 XL") == GUIDE
    assert preprompts.model_key("an SDXL model") == "SDXL"
    assert preprompts.model_key("NoobAI-XL (from config)") == "NoobAI-XL"


def test_for_checkpoint_and_settings_override():
    guide, own = preprompts.for_checkpoint({}, NOOB)
    assert own and "very awa" in guide
    assert preprompts.for_checkpoint({}, PONY) == (GUIDE, False)
    cfg = {"prompt_writer": {"preprompts": {"Pony Diffusion V6 XL": "Pony rules.", "NoobAI-XL": None}}}
    assert preprompts.for_checkpoint(cfg, PONY) == ("Pony rules.", True)
    assert preprompts.for_checkpoint(cfg, NOOB)[0] == preprompts.default("NoobAI-XL")  # None = default
    listed = {m["key"]: m for m in preprompts.listing(cfg)}
    assert listed["Pony Diffusion V6 XL"]["edited"] and not listed["NoobAI-XL"]["edited"]


def test_own_guidelines_drop_the_short_conventions_but_keep_the_marker():
    setup = ckpt_info.prompt_setup({}, NOOB)
    assert "NoobAI-XL" in setup["checkpoint_note"] and "quality tags" not in setup["checkpoint_note"]
    assert ckpt_info.NO_PONY_TAGS in setup["checkpoint_note"]
    pony = ckpt_info.prompt_setup({}, PONY)
    assert "score_9" in pony["checkpoint_note"] and pony["guidelines"] == GUIDE


def test_writer_gets_guidelines_then_contract():
    seen = []

    class Backend:
        def complete(self, instructions, parts, schema, name, max_side):
            seen.append(instructions)
            return {"positive": "masterpiece, very awa, score_9, 1girl", "negative": "worst quality, score_4",
                    "dropped": [], "notes": ""}, 0, 0

    out = write_prompt(Backend(), "a knight", **ckpt_info.prompt_setup({}, NOOB))
    assert seen[0].startswith("# NoobAI XL") and seen[0].endswith(CONTRACT)
    assert out["positive"] == "masterpiece, very awa, 1girl" and out["negative"] == "worst quality"
    write_prompt(Backend(), "a knight")
    assert seen[1] == GUIDE + "\n\n" + CONTRACT
