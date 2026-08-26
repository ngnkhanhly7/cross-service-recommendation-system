import numpy as np

from src.serve import RecommendationEngine


def _base_artifact() -> dict:
    return {
        "model_type": "test",
        "user_factors": np.array([[1.0, 0.0]], dtype=np.float32),
        "item_factors": np.array([[1.0, 0.0], [0.9, 0.0], [0.8, 0.0]], dtype=np.float32),
        "user_to_idx": {"u1": 0},
        "idx_to_item": ["seen", "hotel", "shop"],
        "item_categories": ["xanh_sm", "vinpearl", "shopping"],
        "train_seen": {"u1": ["seen"]},
        "user_category_history": {"u1": {"xanh_sm": 3}},
        "category_transitions": {"xanh_sm": {"vinpearl": 68.0}},
        "item_popularity": {"seen": 5.0, "hotel": 12.0, "shop": 3.0},
    }


def test_engine_filters_seen_and_target_category():
    engine = RecommendationEngine(_base_artifact())
    outcome = engine.recommend("u1", context="xanh_sm_trip_to_vinpearl", top_k=5)
    assert outcome["personalized"] is True
    assert [row["item_id"] for row in outcome["results"]] == ["hotel"]
    assert "68.0%" in outcome["results"][0]["reason"]


def test_unknown_user_with_context_gets_popular_fallback_in_target_category():
    engine = RecommendationEngine(_base_artifact())
    outcome = engine.recommend("new_user", context="xanh_sm_trip_to_vinpearl", top_k=5)
    assert outcome["personalized"] is False
    assert [row["item_id"] for row in outcome["results"]] == ["hotel"]
    assert all(row["category"] == "vinpearl" for row in outcome["results"])


def test_unknown_user_without_context_gets_global_popular_fallback():
    engine = RecommendationEngine(_base_artifact())
    outcome = engine.recommend("new_user", top_k=5)
    assert outcome["personalized"] is False
    assert [row["item_id"] for row in outcome["results"]] == ["hotel", "seen", "shop"]

