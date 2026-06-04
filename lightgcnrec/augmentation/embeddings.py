"""Semantic embedding cache helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def load_embedding_cache(path: str | Path) -> dict[str, list[float]]:
    """Load JSON embeddings saved as `{external_id: [float, ...]}`."""

    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Embedding cache must be a JSON object.")
    return {str(key): [float(value) for value in values] for key, values in payload.items()}


def align_embedding_cache(cache: Mapping[str, list[float]], mapping: Mapping[object, int]) -> list[list[float] | None]:
    """Align external-ID embeddings to contiguous model IDs."""

    aligned: list[list[float] | None] = [None] * len(mapping)
    for external_id, model_id in mapping.items():
        aligned[model_id] = cache.get(str(external_id))
    return aligned
