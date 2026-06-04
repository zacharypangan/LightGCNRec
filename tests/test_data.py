from lightgcnrec.data import build_id_mapping, edges_from_interactions, split_edges, user_positive_items


def test_edges_from_interactions_filters_and_maps_in_first_seen_order():
    records = [
        {"userId": "u2", "movieId": "m9", "rating": "5"},
        {"userId": "u1", "movieId": "m8", "rating": "3"},
        {"userId": "u2", "movieId": "m7", "rating": "4"},
    ]

    edges, users, items = edges_from_interactions(records, "userId", "movieId", "rating", 4)

    assert users == {"u2": 0}
    assert items == {"m9": 0, "m7": 1}
    assert edges == [(0, 0), (0, 1)]


def test_split_edges_is_deterministic_and_complete():
    edges = [(i, i + 1) for i in range(10)]

    train, val, test = split_edges(edges, val_ratio=0.2, test_ratio=0.2, seed=7)

    assert (train, val, test) == split_edges(edges, val_ratio=0.2, test_ratio=0.2, seed=7)
    assert sorted(train + val + test) == sorted(edges)
    assert len(train) == 6
    assert len(val) == 2
    assert len(test) == 2


def test_user_positive_items_groups_edges():
    assert user_positive_items([(1, 2), (1, 3), (2, 3)]) == {1: {2, 3}, 2: {3}}


def test_build_id_mapping_keeps_first_seen_order():
    assert build_id_mapping(["b", "a", "b"]) == {"b": 0, "a": 1}
