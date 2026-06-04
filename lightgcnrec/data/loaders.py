"""Small, dependency-light data utilities for recommendation experiments."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import random
from pathlib import Path
from typing import Iterable, Mapping, Sequence

Edge = tuple[int, int]


@dataclass(frozen=True)
class DatasetConfig:
    """Column and path contract for a user-item interaction dataset."""

    name: str
    interactions_path: Path
    user_col: str
    item_col: str
    rating_col: str | None = None
    item_metadata_path: Path | None = None
    rating_threshold: float | None = None


def load_csv_records(path: str | Path) -> list[dict[str, str]]:
    """Load a CSV file as dictionaries without requiring pandas."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_columns(records: Sequence[Mapping[str, object]], required: Iterable[str]) -> None:
    """Raise a useful error if required columns are absent."""

    if not records:
        raise ValueError("Cannot validate columns for an empty record set.")
    available = set(records[0].keys())
    missing = [column for column in required if column not in available]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def build_id_mapping(values: Iterable[object]) -> dict[object, int]:
    """Map stable external IDs to contiguous integer IDs in first-seen order."""

    mapping: dict[object, int] = {}
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping)
    return mapping


def edges_from_interactions(
    records: Sequence[Mapping[str, object]],
    user_col: str,
    item_col: str,
    rating_col: str | None = None,
    rating_threshold: float | None = None,
) -> tuple[list[Edge], dict[object, int], dict[object, int]]:
    """Build integer user-item edges and ID mappings from interaction records."""

    required = [user_col, item_col]
    if rating_col:
        required.append(rating_col)
    validate_columns(records, required)

    filtered: list[Mapping[str, object]] = []
    for record in records:
        if rating_col and rating_threshold is not None:
            try:
                if float(record[rating_col]) < rating_threshold:
                    continue
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid rating value {record[rating_col]!r}") from exc
        filtered.append(record)

    user_mapping = build_id_mapping(record[user_col] for record in filtered)
    item_mapping = build_id_mapping(record[item_col] for record in filtered)
    edges = [(user_mapping[record[user_col]], item_mapping[record[item_col]]) for record in filtered]
    return edges, user_mapping, item_mapping


def split_edges(
    edges: Sequence[Edge],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[Edge], list[Edge], list[Edge]]:
    """Shuffle and split edges into train, validation, and test partitions."""

    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to less than 1.")
    shuffled = list(edges)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    total = len(shuffled)
    test_size = int(total * test_ratio)
    val_size = int(total * val_ratio)
    test = shuffled[:test_size]
    val = shuffled[test_size : test_size + val_size]
    train = shuffled[test_size + val_size :]
    return train, val, test


def user_positive_items(edges: Iterable[Edge]) -> dict[int, set[int]]:
    """Group positive items by user from a user-item edge list."""

    positives: dict[int, set[int]] = {}
    for user_id, item_id in edges:
        positives.setdefault(user_id, set()).add(item_id)
    return positives
