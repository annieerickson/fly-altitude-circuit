"""Locate and load repository YAML configuration shared across datasets."""

import yaml
from functools import lru_cache
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[2] / "config"

@lru_cache
def load(name):
    return yaml.safe_load((CONFIG / f"{name}.yaml").read_text())

