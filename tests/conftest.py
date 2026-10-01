"""Shared helpers. The tests need neither ComfyUI nor an LLM: those are replaced by small
fakes, and anything that writes files does so under pytest's tmp_path."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class FakeLibrary:
    """Stands in for LoraLibrary: index() is all with_triggers and the pickers read."""

    def __init__(self, index: dict):
        self._index = index

    def index(self) -> dict:
        return self._index


@pytest.fixture
def library():
    return FakeLibrary({
        "Pony\\styles\\artists\\jab\\Jabstyle_PNYV1.5.safetensors": {"trigger_words": ["Jabstyle"], "title": "Jab Style"},
        "Pony\\styles\\flat\\FlatColor.safetensors": {"trigger_words": ["flat color"], "title": "FlatColor"},
        "Pony\\concepts\\a\\poseA.safetensors": {"trigger_words": ["1girl", "standing", "p0seA"], "title": "Pose A"},
        "Pony\\concepts\\b\\poseB.safetensors": {"trigger_words": ["1girl", "full body"], "title": "Standing"},
    })
