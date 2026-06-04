"""Evaluate recommendation outputs against held-out interactions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lightgcnrec.metrics import ndcg_at_k, precision_at_k, recall_at_k


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recommendations", required=True, help="JSON mapping user IDs to ranked item IDs.")
    parser.add_argument("--ground-truth", required=True, help="JSON mapping user IDs to relevant item IDs.")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--checkpoint", help="Accepted for interface compatibility; not required for JSON evaluation.")
    args = parser.parse_args()
    recommendations = _load_rankings(args.recommendations)
    ground_truth = {int(user): set(items) for user, items in _load_rankings(args.ground_truth).items()}
    print(
        json.dumps(
            {
                f"recall@{args.k}": recall_at_k(recommendations, ground_truth, args.k),
                f"precision@{args.k}": precision_at_k(recommendations, ground_truth, args.k),
                f"ndcg@{args.k}": ndcg_at_k(recommendations, ground_truth, args.k),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _load_rankings(path: str | Path) -> dict[int, list[int]]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return {int(user): [int(item) for item in items] for user, items in payload.items()}


if __name__ == "__main__":
    main()
