from lightgcnrec.metrics import (
    category_coverage_at_k,
    dcc_at_k,
    fadcc_at_k,
    item_coverage_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_accuracy_metrics_on_tiny_rankings():
    recs = {0: [1, 2, 3], 1: [2, 4, 5]}
    truth = {0: {1, 3}, 1: {5}}

    assert recall_at_k(recs, truth, 2) == 0.25
    assert precision_at_k(recs, truth, 2) == 0.25
    assert round(ndcg_at_k(recs, truth, 3), 4) == 0.7099


def test_coverage_metrics_on_tiny_rankings():
    recs = {0: [1, 2, 3], 1: [2, 4, 5]}
    categories = {1: ["Action"], 2: ["Drama"], 3: ["Action"], 4: ["Comedy"], 5: ["Drama"]}

    assert item_coverage_at_k(recs, total_items=6, k=2) == 0.5
    assert category_coverage_at_k(recs, categories, total_categories=3, k=2) == 1.0
    assert dcc_at_k(recs, categories, k=2) > 0
    assert fadcc_at_k(recs, categories, {"Action": 10, "Drama": 5, "Comedy": 1}, k=2) > 0
