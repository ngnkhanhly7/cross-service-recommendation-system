from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.common import dump_json, read_interactions
from src.features import encode_train_test, temporal_leave_one_out


def top_k_from_factors(
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    seen_matrix,
    k: int = 10,
) -> np.ndarray:
    """Exact top-K retrieval used for offline evaluation and small catalogs."""

    n_users, n_items = user_factors.shape[0], item_factors.shape[0]
    result = np.full((n_users, min(k, n_items)), -1, dtype=np.int64)
    for user_idx in range(n_users):
        scores = user_factors[user_idx] @ item_factors.T
        seen = seen_matrix.indices[
            seen_matrix.indptr[user_idx] : seen_matrix.indptr[user_idx + 1]
        ]
        scores[seen] = -np.inf
        valid_k = min(k, n_items - len(seen))
        if valid_k <= 0:
            continue
        candidate = np.argpartition(scores, -valid_k)[-valid_k:]
        ordered = candidate[np.argsort(scores[candidate])[::-1]]
        result[user_idx, :valid_k] = ordered
    return result


def _segment_metrics(ranks: list[int | None], k: int) -> dict[str, float | int]:
    if not ranks:
        return {"users": 0, f"recall@{k}": 0.0, f"ndcg@{k}": 0.0}
    recalls = [1.0 if rank is not None and rank < k else 0.0 for rank in ranks]
    ndcgs = [
        1.0 / math.log2(rank + 2) if rank is not None and rank < k else 0.0
        for rank in ranks
    ]
    return {
        "users": len(ranks),
        f"recall@{k}": round(float(np.mean(recalls)), 6),
        f"ndcg@{k}": round(float(np.mean(ndcgs)), 6),
    }


def evaluate_rankings(
    rankings: np.ndarray,
    encoded,
    *,
    k: int = 10,
) -> dict[str, Any]:
    test_by_user = encoded.test.set_index("user_id")
    dominant = (
        encoded.train.groupby(["user_id", "category"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["user_id", "count", "category"], ascending=[True, False, True])
        .drop_duplicates("user_id")
        .set_index("user_id")["category"]
    )
    all_ranks: list[int | None] = []
    cross_ranks: list[int | None] = []
    same_ranks: list[int | None] = []
    per_user: list[dict[str, Any]] = []
    for user_id, user_idx in encoded.user_to_idx.items():
        if user_id not in test_by_user.index:
            continue
        test_row = test_by_user.loc[user_id]
        if isinstance(test_row, pd.DataFrame):
            test_row = test_row.iloc[-1]
        target_idx = encoded.item_to_idx.get(str(test_row["item_id"]))
        positions = np.flatnonzero(rankings[user_idx] == target_idx)
        rank = int(positions[0]) if len(positions) else None
        is_cross = str(test_row["category"]) != str(dominant.get(user_id, ""))
        all_ranks.append(rank)
        (cross_ranks if is_cross else same_ranks).append(rank)
        per_user.append(
            {
                "user_id": user_id,
                "target_item": str(test_row["item_id"]),
                "target_category": str(test_row["category"]),
                "dominant_history_category": str(dominant.get(user_id, "")),
                "is_cross_category": is_cross,
                "rank": rank,
            }
        )
    return {
        "k": k,
        "overall": _segment_metrics(all_ranks, k),
        "cross_category": _segment_metrics(cross_ranks, k),
        "same_category": _segment_metrics(same_ranks, k),
        "cross_category_definition": (
            "held-out item's category differs from the user's most frequent training category"
        ),
        "per_user": per_user,
    }


def evaluate_artifact(
    artifact: dict[str, Any], interactions_path: str | Path, k: int = 10
) -> dict[str, Any]:
    df = read_interactions(interactions_path)
    train, test = temporal_leave_one_out(df)
    encoded = encode_train_test(train, test)
    if artifact["idx_to_user"] != encoded.idx_to_user or artifact["idx_to_item"] != encoded.idx_to_item:
        raise ValueError(
            "Artifact mappings do not match this data split. Evaluate with the same input file used for training."
        )
    rankings = top_k_from_factors(
        artifact["user_factors"], artifact["item_factors"], encoded.train_matrix, k
    )
    return evaluate_rankings(rankings, encoded, k=k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained factor artifact")
    parser.add_argument("--data", default="data/processed/interactions.csv")
    parser.add_argument("--model", default="models/als_v1.pkl")
    parser.add_argument("--output", default="reports/evaluation_metrics.json")
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()
    artifact = joblib.load(args.model)
    metrics = evaluate_artifact(artifact, args.data, args.k)
    dump_json({key: value for key, value in metrics.items() if key != "per_user"}, args.output)
    print(json.dumps({key: value for key, value in metrics.items() if key != "per_user"}, indent=2))


if __name__ == "__main__":
    main()

