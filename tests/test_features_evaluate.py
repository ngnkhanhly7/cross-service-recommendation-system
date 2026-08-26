import numpy as np

from src.data_loader import generate_demo_amazon
from src.evaluate import evaluate_rankings
from src.features import encode_train_test, temporal_leave_one_out


def test_temporal_split_has_no_user_target_leakage():
    frame = generate_demo_amazon(n_users=25, n_items_per_category=16, seed=12)
    train, test = temporal_leave_one_out(frame)
    joined = train.merge(test, on=["user_id", "item_id"])
    assert joined.empty
    latest_train = train.groupby("user_id")["timestamp"].max()
    test_time = test.set_index("user_id")["timestamp"]
    eligible_latest_train = latest_train.loc[test_time.index]
    assert (test_time >= eligible_latest_train).all()


def test_perfect_rankings_score_one():
    frame = generate_demo_amazon(n_users=20, n_items_per_category=12, seed=5)
    train, test = temporal_leave_one_out(frame)
    encoded = encode_train_test(train, test)
    rankings = np.full((len(encoded.idx_to_user), 10), -1, dtype=int)
    for user_id, user_idx in encoded.user_to_idx.items():
        target = test.set_index("user_id").loc[user_id, "item_id"]
        rankings[user_idx, 0] = encoded.item_to_idx[target]
    metrics = evaluate_rankings(rankings, encoded, k=10)
    assert metrics["overall"]["recall@10"] == 1.0
    assert metrics["overall"]["ndcg@10"] == 1.0
