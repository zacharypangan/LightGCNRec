"""Ranking and coverage metrics for top-k recommendation lists."""

from __future__ import annotations

from collections import Counter
import math
from typing import Iterable, Mapping, Sequence


def _top_k(items: Sequence[int], k: int) -> list[int]:
    return list(items[:k])


def recall_at_k(recommendations: Mapping[int, Sequence[int]], ground_truth: Mapping[int, set[int]], k: int) -> float:
    """Mean Recall@K across users with at least one relevant item."""

    scores = []
    for user, relevant in ground_truth.items():
        if not relevant:
            continue
        hits = len(set(_top_k(recommendations.get(user, []), k)) & relevant)
        scores.append(hits / len(relevant))
    return sum(scores) / len(scores) if scores else 0.0


def precision_at_k(recommendations: Mapping[int, Sequence[int]], ground_truth: Mapping[int, set[int]], k: int) -> float:
    """Mean Precision@K across users."""

    if k <= 0:
        raise ValueError("k must be positive.")
    scores = []
    for user, recommended in recommendations.items():
        relevant = ground_truth.get(user, set())
        hits = len(set(_top_k(recommended, k)) & relevant)
        scores.append(hits / k)
    return sum(scores) / len(scores) if scores else 0.0


def ndcg_at_k(recommendations: Mapping[int, Sequence[int]], ground_truth: Mapping[int, set[int]], k: int) -> float:
    """Mean binary NDCG@K across users."""

    scores = []
    for user, recommended in recommendations.items():
        relevant = ground_truth.get(user, set())
        if not relevant:
            continue
        dcg = sum(1 / math.log2(rank + 2) for rank, item in enumerate(_top_k(recommended, k)) if item in relevant)
        ideal_hits = min(len(relevant), k)
        idcg = sum(1 / math.log2(rank + 2) for rank in range(ideal_hits))
        scores.append(dcg / idcg if idcg else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def item_coverage_at_k(recommendations: Mapping[int, Sequence[int]], total_items: int, k: int) -> float:
    """Fraction of catalog items appearing in any top-k list."""

    if total_items <= 0:
        raise ValueError("total_items must be positive.")
    covered = {item for recs in recommendations.values() for item in _top_k(recs, k)}
    return len(covered) / total_items


def category_coverage_at_k(
    recommendations: Mapping[int, Sequence[int]],
    item_categories: Mapping[int, Iterable[str]],
    total_categories: int | None,
    k: int,
) -> float:
    """Fraction of categories represented in any top-k list."""

    covered = {
        category
        for recs in recommendations.values()
        for item in _top_k(recs, k)
        for category in item_categories.get(item, [])
    }
    denominator = total_categories or len({category for values in item_categories.values() for category in values})
    return len(covered) / denominator if denominator else 0.0


def dcc_at_k(recommendations: Mapping[int, Sequence[int]], item_categories: Mapping[int, Iterable[str]], k: int) -> float:
    """Discounted Category Coverage averaged over users."""

    scores = []
    for recs in recommendations.values():
        seen: set[str] = set()
        score = 0.0
        for rank, item in enumerate(_top_k(recs, k)):
            new_categories = set(item_categories.get(item, [])) - seen
            if new_categories:
                score += len(new_categories) / math.log2(rank + 2)
                seen.update(new_categories)
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


def fadcc_at_k(
    recommendations: Mapping[int, Sequence[int]],
    item_categories: Mapping[int, Iterable[str]],
    category_frequency: Mapping[str, int],
    k: int,
) -> float:
    """Frequency-aware DCC where rarer categories receive higher weight."""

    total_frequency = sum(category_frequency.values()) or 1
    scores = []
    for recs in recommendations.values():
        seen: set[str] = set()
        score = 0.0
        for rank, item in enumerate(_top_k(recs, k)):
            for category in set(item_categories.get(item, [])) - seen:
                rarity = math.log1p(total_frequency / max(category_frequency.get(category, 1), 1))
                score += rarity / math.log2(rank + 2)
                seen.add(category)
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


def system_personal_diversity(recommendations: Mapping[int, Sequence[int]], k: int) -> dict[str, float]:
    """Return simple system-level list diversity diagnostics."""

    lists = [_top_k(recs, k) for recs in recommendations.values()]
    if not lists:
        return {"unique_items": 0.0, "mean_duplicate_rate": 0.0}
    counts = Counter(item for recs in lists for item in recs)
    duplicate_rates = [1 - (len(set(recs)) / len(recs)) for recs in lists if recs]
    return {
        "unique_items": float(len(counts)),
        "mean_duplicate_rate": sum(duplicate_rates) / len(duplicate_rates) if duplicate_rates else 0.0,
    }
