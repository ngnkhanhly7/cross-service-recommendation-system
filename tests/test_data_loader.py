import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from src.common import SCHEMA, validate_interactions
from src.data_loader import (
    generate_demo_amazon,
    generate_multi_service_simulation,
    iterative_sparsity_filter,
    summarize,
)


def test_demo_obeys_schema_and_is_multi_category():
    frame = generate_demo_amazon(n_users=30, n_items_per_category=12, seed=7)
    assert list(frame.columns) == SCHEMA
    assert is_datetime64_any_dtype(frame["timestamp"])
    assert frame["timestamp"].dt.tz is not None
    assert summarize(frame)["multi_category_user_pct"] > 90


def test_k_core_thresholds_hold():
    frame = generate_demo_amazon(n_users=50, n_items_per_category=8, seed=3)
    filtered = iterative_sparsity_filter(
        frame, min_user_interactions=5, min_item_interactions=2, min_user_categories=2
    )
    assert filtered.groupby("user_id").size().min() >= 5
    assert filtered.groupby("item_id").size().min() >= 2
    assert filtered.groupby("user_id")["category"].nunique().min() >= 2


def test_multi_service_simulation_is_labeled_and_cross_service():
    frame = generate_multi_service_simulation(n_users=40, seed=11)
    assert set(frame["category"]) == {"xanh_sm", "vinpearl", "vinmec", "shopping"}
    assert frame["item_id"].str.startswith("sim:").all()
    assert frame.groupby("user_id")["category"].nunique().mean() >= 2


def test_schema_rejects_nonpositive_strength():
    bad = pd.DataFrame(
        [["u", "i", "c", "2024-01-01", 0]], columns=SCHEMA
    )
    try:
        validate_interactions(bad)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("nonpositive strength should fail validation")
