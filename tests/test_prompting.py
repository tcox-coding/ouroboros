"""The start of a job: which LoRA trigger words reach the prompt, and the prompt writer's
safety nets (found by rendering written prompts and judging them; see prompter.py)."""
import pytest

from ouroboros import prompter
from ouroboros.loras import lora_trigger, with_triggers
from ouroboros.params import norm_tag, split_tags


@pytest.mark.parametrize("name, rec, want", [
    ("Pony\\styles\\Jabstyle_PNYV1.5.safetensors", {"trigger_words": ["Jabstyle"], "title": "Jab Style"}, "Jabstyle"),
    ("Pony\\styles\\FlatColor.safetensors", {"trigger_words": ["flat color"], "title": "FlatColor"}, "flat color"),
    ("x\\tendertroupe_v0.1-pony.safetensors", {"trigger_words": ["long_hair"], "title": "Cartoon style"}, None),
    ("x\\incase_style_v3-1_ponyxl.safetensors", {"trigger_words": ["female"], "title": "Incase Style [PonyXL]"}, None),
    ("x\\poseA.safetensors", {"trigger_words": ["1girl", "p0seA"], "title": "Pose A"}, "p0seA"),
    ("x\\lbh.safetensors", {"trigger_words": ["legsbehindhead"], "title": "Legs Behind Head"}, "legsbehindhead"),
    ("x\\helix.safetensors", {"trigger_words": ["helixart", "vectorized"], "title": "helix"}, "helixart"),
    ("x\\wrong.safetensors", {"trigger_words": ["WHF"], "title": "Wrong Hole Expression"}, "WHF"),
    ("x\\style.safetensors", {"trigger_words": ["score_9", "rating_safe"], "title": "x"}, None),
    ("x\\none.safetensors", {"trigger_words": [], "title": "x"}, None),
])
def test_only_a_loras_own_trigger_counts(name, rec, want):
    assert lora_trigger(rec, name) == want


class Lib:
    def __init__(self, index):
        self._index = index

    def index(self):
        return self._index


def test_renders_get_the_loras_own_trigger_not_a_caption_tag():
    cartoon, flat = "x\\tendertroupe_v0.1-pony.safetensors", "x\\FlatColor.safetensors"
    lib = Lib({cartoon: {"trigger_words": ["long_hair"], "title": "Cartoon style"},
               flat: {"trigger_words": ["flat color"], "title": "FlatColor"}})
    out = with_triggers("1girl, short hair, bob cut", ((cartoon, 0.5), (flat, 0.8)), lib, split_tags, norm_tag)
    assert "flat color" in out and "long_hair" not in out and "long hair" not in out


class Writer:
    """Answers each call from a list, and records what it was sent."""
    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    def complete(self, instructions, parts, schema, name, max_side):
        self.calls.append(" ".join(p.get("text", "") for p in parts))
        return self.answers.pop(0), 0.0, 0


def answer(pos, neg):
    return {"positive": pos, "negative": neg, "dropped": [], "notes": ""}


def test_a_negative_leaked_into_the_positive_is_retried_once():
    bad = answer("score_9, 1girl, white hair\n\nscore_4, score_5, worst quality, blonde hair", "")
    good = answer("score_9, 1girl, white hair", "score_4, score_5, worst quality, blonde hair")
    w = Writer(bad, good)
    out = prompter.write_prompt(w, "a white-haired girl")
    assert len(w.calls) == 2 and "negative prompt is empty" in w.calls[1]
    assert out["positive"] == "score_9, 1girl, white hair" and "score_4" in out["negative"]


def test_a_retry_that_is_no_better_is_repaired_instead():
    bad = answer("score_9, 1girl\nscore_4, lowres", "")
    w = Writer(bad, bad)
    out = prompter.write_prompt(w, "a girl")
    assert "score_4" not in out["positive"] and "lowres" not in out["positive"]
    assert out["negative"].startswith("score_4, lowres")


def test_a_clean_answer_needs_one_call():
    w = Writer(answer("score_9, 1girl", "lowres"))
    prompter.write_prompt(w, "a girl")
    assert len(w.calls) == 1


def test_a_negative_alone_still_gets_the_workflows_style_examples():
    w = Writer(answer("score_9, Jabstyle, 1girl", "lowres, text"))
    prompter.write_prompt(w, "a girl", "", "text, watermark", style_positive="score_9, Jabstyle, 1girl, red dress",
                          style_negative="lowres")
    sent = w.calls[0]
    assert "EXAMPLE POSITIVE" in sent and "score_9, Jabstyle" in sent and "MERGE" not in sent
    assert "USER NEGATIVE PROMPT" in sent and "text, watermark" in sent


def test_a_user_negative_is_kept():
    w = Writer(answer("score_9, 1girl", "lowres"))
    out = prompter.write_prompt(w, "a girl", "", "text, watermark")
    assert "text" in out["negative"] and "watermark" in out["negative"]
