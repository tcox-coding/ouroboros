"""Shared constants for the live tests."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "eval" / "images"
SAMPLERS = ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde"]
SCHEDULERS = ["normal", "karras", "exponential"]
