# src/config/loader.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg
