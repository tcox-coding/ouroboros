"""Live tests: real calls to the LLM in config.json (the judge backend and model), checking
the back and forth between Ouroboros and the judge: what we send it is accepted, and what
it sends back is something the loop can use. They cost a little (a few US cents a run on
a hosted model) and need the network, so they only run when asked:

    OUROBOROS_LIVE=1 .venv/bin/python -m pytest tests/live -v

OUROBOROS_LIVE_MODEL=<model> tries another model on the same backend without editing
config.json. Rendering is never needed: ComfyUI is replaced by a fake that hands back
labelled images from eval/images, so the judge sees real pictures.
"""
import copy
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from livecfg import SAMPLERS, SCHEDULERS  # noqa: E402

LIVE = os.environ.get("OUROBOROS_LIVE") == "1"


def pytest_collection_modifyitems(config, items):
    if LIVE:
        return
    skip = pytest.mark.skip(reason="live LLM test: set OUROBOROS_LIVE=1 to run")
    for item in items:
        if "tests/live" in str(item.fspath).replace("\\", "/"):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def judge_cfg():
    from ouroboros.runner import load_config
    cfg = copy.deepcopy(load_config())
    model = os.environ.get("OUROBOROS_LIVE_MODEL")
    if model:
        cfg["judge"][cfg["judge"]["backend"]]["model"] = model
    return cfg


@pytest.fixture(scope="session")
def judge(judge_cfg):
    from ouroboros.judge import Judge
    return Judge(judge_cfg["judge"], SAMPLERS, SCHEDULERS)


@pytest.fixture(scope="session")
def backend(judge):
    return judge.backend

