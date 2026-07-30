from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    required = {"docs_root", "documents", "chunking", "embedding", "vector_store"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing configuration keys: {sorted(missing)}")
    return config
