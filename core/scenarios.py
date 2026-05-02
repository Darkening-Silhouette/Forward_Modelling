from __future__ import annotations

from pathlib import Path
import yaml


def load_scenario(path: str | Path) -> dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"Scenario must contain a YAML dictionary: {path}")

    return data