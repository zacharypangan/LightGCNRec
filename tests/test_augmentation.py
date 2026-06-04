from lightgcnrec.augmentation import PromptTemplate, build_augmented_edges, merge_edges, render_prompt


def test_build_augmented_edges_maps_known_records_and_filters_scores():
    records = [
        {"user_id": "u1", "item_id": "i1", "score": 0.9},
        {"user_id": "u1", "item_id": "i2", "score": 0.1},
        {"user_id": "unknown", "item_id": "i1", "score": 0.9},
    ]

    edges = build_augmented_edges(records, {"u1": 0}, {"i1": 0, "i2": 1}, min_score=0.5)

    assert edges == [(0, 0)]


def test_merge_edges_deduplicates_preserving_order():
    assert merge_edges([(0, 1), (0, 2)], [(0, 1), (1, 2)]) == [(0, 1), (0, 2), (1, 2)]


def test_render_prompt_uses_safe_substitution():
    prompt = PromptTemplate("p", "user", "test", "Hello $name from $missing")

    assert render_prompt(prompt, {"name": "Zac"}) == "Hello Zac from $missing"
