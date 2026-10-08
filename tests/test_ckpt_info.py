"""Which model a checkpoint is, from its name and safetensors header, and that the prompt
writer is told."""

import json
import struct

from ouroboros import ckpt_info
from ouroboros.prompter import write_prompt


def fake_safetensors(path, metadata=None, markers=()):
    header = {"model.diffusion_model.x": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}
    for m in markers:
        header[m] = {"dtype": "F32", "shape": [0], "data_offsets": [2, 2]}
    if metadata:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0\0")
    return path


def test_known_names():
    assert ckpt_info.describe("ponyDiffusionV6XL_v6.safetensors")["model"] == "Pony Diffusion V6 XL"
    assert ckpt_info.describe("autismmixSDXL_confetti.safetensors")["family"] == "Pony"
    wai = ckpt_info.describe("waiIllustriousSDXL_v120.safetensors")
    assert (wai["model"], wai["family"]) == ("Illustrious-XL", "Illustrious")
    noob = ckpt_info.describe("noobaiXLNAIXL_vPred10Version.safetensors")
    assert (noob["model"], noob["family"], noob["prediction"]) == ("NoobAI-XL", "Illustrious", "v-prediction")
    assert ckpt_info.describe("mystery.safetensors")["source"] == "unknown"


def test_metadata_names_a_merge(tmp_path):
    f = fake_safetensors(tmp_path / "myMerge.safetensors",
                         {"modelspec.title": "MyMerge", "modelspec.merged_from": "base, Pony Diffusion V6 XL",
                          "modelspec.prediction_type": "epsilon"})
    info = ckpt_info.describe(f.name, f)
    assert (info["model"], info["source"], info["prediction"]) == ("Pony Diffusion V6 XL", "metadata", "epsilon")
    assert "merged from base, Pony Diffusion V6 XL" in ckpt_info.note(info)


def test_vpred_markers(tmp_path):
    f = fake_safetensors(tmp_path / "someNoob.safetensors", markers=("v_pred", "ztsnr"))
    info = ckpt_info.describe(f.name, f)
    assert info["prediction"] == "v-prediction with zero terminal SNR"
    assert "v-prediction" in ckpt_info.note(info)


def test_config_override_wins():
    info = ckpt_info.describe("fancyMix.safetensors", overrides={"fancy": "Illustrious"})
    assert (info["family"], info["source"]) == ("Illustrious", "config")
    assert "score_" in info["conventions"] and "don't use them" in info["conventions"]


def test_lookup_finds_the_file(tmp_path):
    fake_safetensors(tmp_path / "plain.safetensors", {"modelspec.title": "Illustrious merge"})
    info = ckpt_info.lookup("plain.safetensors", {"checkpoints_dir": str(tmp_path)})
    assert (info["family"], info["source"]) == ("Illustrious", "metadata")
    assert ckpt_info.checkpoint_note({}, "") == ""


def test_prompt_writer_is_told():
    seen = []

    class Backend:
        def complete(self, instructions, parts, schema, name, max_side):
            seen.append(parts)
            return {"positive": "masterpiece, 1girl", "negative": "worst quality", "dropped": [], "notes": ""}, 0, 0

    note = ckpt_info.note(ckpt_info.describe("noobaiXL_vPred.safetensors"))
    write_prompt(Backend(), "a knight", checkpoint_note=note)
    text = "\n".join(p.get("text", "") for p in seen[0])
    assert "CHECKPOINT" in text and "NoobAI-XL" in text and "newest" in text


def test_pony_tags_stripped_for_other_models():
    class Backend:
        def complete(self, instructions, parts, schema, name, max_side):
            return {"positive": "masterpiece, newest, score_9, 1girl\n\nsource_anime, knight",
                    "negative": "worst quality, score_4, score_5, (source_pony:1.2)", "dropped": [], "notes": ""}, 0, 0

    noob = ckpt_info.note(ckpt_info.describe("noobaiXL.safetensors"))
    out = write_prompt(Backend(), "a knight", checkpoint_note=noob)
    assert out["positive"] == "masterpiece, newest, 1girl\n\nknight"
    assert out["negative"] == "worst quality"
    assert {"score_9", "source_anime", "score_4", "(source_pony:1.2)"} <= set(out["dropped"])
    # a tag the user wrote themselves is kept, as with any other tag of theirs
    out = write_prompt(Backend(), "a knight", positive="score_9, 1girl", checkpoint_note=noob)
    assert "score_9" in out["positive"]
    # Pony keeps them
    pony = ckpt_info.note(ckpt_info.describe("ponyDiffusionV6XL.safetensors"))
    assert "score_9" in write_prompt(Backend(), "a knight", checkpoint_note=pony)["positive"]
