"""Graph augmentation helpers for prompt-derived user-item links."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from lightgcnrec.data import Edge


@dataclass(frozen=True)
class AugmentedEdge:
    """A prompt-derived recommendation edge before integer mapping."""

    user_id: object
    item_id: object
    source: str
    score: float | None = None


def build_augmented_edges(
    records: Iterable[Mapping[str, object]],
    user_mapping: Mapping[object, int],
    item_mapping: Mapping[object, int],
    user_col: str = "user_id",
    item_col: str = "item_id",
    min_score: float | None = None,
    score_col: str = "score",
) -> list[Edge]:
    """Convert prompt-derived records into mapped user-item edges."""

    edges: list[Edge] = []
    for record in records:
        if min_score is not None and score_col in record and float(record[score_col]) < min_score:
            continue
        user = record[user_col]
        item = record[item_col]
        if user in user_mapping and item in item_mapping:
            edges.append((user_mapping[user], item_mapping[item]))
    return edges


def merge_edges(base_edges: Iterable[Edge], augmented_edges: Iterable[Edge]) -> list[Edge]:
    """Merge base and augmented edge sets while preserving first-seen order."""

    merged: list[Edge] = []
    seen: set[Edge] = set()
    for edge in list(base_edges) + list(augmented_edges):
        if edge not in seen:
            merged.append(edge)
            seen.add(edge)
    return merged
