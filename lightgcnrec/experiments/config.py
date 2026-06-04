"""Configuration loading for experiment entrypoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON or YAML config files.

    YAML support is optional so the package can still import in minimal
    environments. Install PyYAML for normal experiment execution.
    """

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install PyYAML to load YAML experiment configs.") from exc
    return yaml.safe_load(text)
