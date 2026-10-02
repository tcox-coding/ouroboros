"""settings_advisor: the LLM's starting sampler settings are clamped, restricted to what
ComfyUI has, and fall back to the current values when the answer is unusable."""
import pytest

from conftest import FakeLibrary
from ouroboros import settings_advisor as advisor

CURRENT = {"steps": 30, "cfg": 5.5, "sampler_name": "dpmpp_2m", "scheduler": "karras"}
SAMPLERS = ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde"]
SCHEDULERS = ["normal", "karras", "exponential", "sgm_uniform"]


class Backend:
    def __init__(self, answer):
        self.answer, self.parts, self.schema = answer, None, None

    def complete(self, instructions, parts, schema, name, max_side):
        self.parts, self.schema = parts, schema
        return self.answer, 0.001, 100


def ask(answer, **kw):
    b = Backend(answer)
    out = advisor.suggest_settings(b, checkpoint="ponyDiffusionV6XL.safetensors", checkpoint_base="Pony",
                                   samplers=SAMPLERS, schedulers=SCHEDULERS, current=CURRENT, **kw)
    return out, b


def test_a_valid_answer_is_used_and_changes_are_listed():
    out, b = ask({"steps": 28, "cfg": 6.5, "sampler_name": "euler_ancestral", "scheduler": "normal", "notes": "Pony"})
    assert (out["steps"], out["cfg"], out["sampler_name"], out["scheduler"]) == (28, 6.5, "euler_ancestral", "normal")
    assert out["changed"] == ["steps", "cfg", "sampler_name", "scheduler"] and out["cost"] == 0.001
    assert b.schema["properties"]["sampler_name"]["enum"] == SAMPLERS  # only what ComfyUI has


def test_out_of_range_values_are_clamped():
    out, _ = ask({"steps": 500, "cfg": 0.2, "sampler_name": "dpmpp_2m", "scheduler": "karras", "notes": ""})
    assert out["steps"] == 60 and out["cfg"] == 1.0
    assert out["changed"] == ["steps", "cfg"]


def test_unknown_or_missing_values_keep_the_current_settings():
    out, _ = ask({"steps": "lots", "cfg": None, "sampler_name": "made_up", "scheduler": "", "notes": ""})
    assert {k: out[k] for k in advisor.FIELDS} == CURRENT and out["changed"] == []


def test_the_request_describes_checkpoint_mode_size_prompt_and_loras():
    lib = FakeLibrary({"Pony\\styles\\x.safetensors": {"title": "X Style"}})
    lib.card = lambda rec, detail=True: f"x: {rec['title']} | weight 0.5-1"
    _, b = ask({**CURRENT, "notes": ""}, positive="1girl, red hair", mode="img2img_reference", denoise=0.6,
               size=(832, 1216), lora_notes=advisor.lora_lines(lib, (("Pony\\styles\\x.safetensors", 0.8),
                                                                     ("Pony\\other\\unknown.safetensors", 1.0))))
    text = b.parts[0]["text"]
    for want in ("model family: Pony", "MODE img2img_reference, denoise 0.6", "SIZE 832x1216", "1girl, red hair",
                 "x: X Style | weight 0.5-1 (used at strength 0.8)", "unknown (used at strength 1)",
                 "AVAILABLE SAMPLERS euler, euler_ancestral"):
        assert want in text


@pytest.mark.parametrize("cfg", [6.0, 7])
def test_summary(cfg):
    assert advisor.summary({**CURRENT, "cfg": cfg}).startswith("steps 30, cfg ")
