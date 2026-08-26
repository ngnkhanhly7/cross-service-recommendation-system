from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD

from src.common import (
    SYNTHETIC_CAUTION,
    dump_json,
    read_interactions,
    read_provenance_sidecar,
    seed_everything,
)
from src.evaluate import evaluate_rankings, top_k_from_factors
from src.features import category_transition_stats, encode_train_test, temporal_leave_one_out


def train_als(
    train_matrix,
    *,
    factors: int = 64,
    regularization: float = 0.05,
    iterations: int = 20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from implicit.als import AlternatingLeastSquares
    except ImportError as exc:
        raise RuntimeError(
            "ALS backend needs `implicit`. Install requirements.txt or use --backend svd for a smoke test."
        ) from exc
    # implicit parallelizes itself; nested OpenBLAS threads can make ALS much slower.
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1, user_api="blas"):
        model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=seed,
        )
        model.fit(train_matrix)
    return model.user_factors.astype("float32"), model.item_factors.astype("float32")


def train_svd(train_matrix, factors: int = 64, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Dependency-light smoke-test backend; ALS remains the project baseline."""

    max_factors = max(1, min(factors, min(train_matrix.shape) - 1))
    model = TruncatedSVD(n_components=max_factors, random_state=seed)
    user_factors = model.fit_transform(train_matrix).astype("float32")
    item_factors = model.components_.T.astype("float32")
    return user_factors, item_factors


def main() -> None:
    parser = argparse.ArgumentParser(description="Train implicit-feedback collaborative filtering")
    parser.add_argument("--data", default="data/processed/interactions.csv")
    parser.add_argument("--model-output", default="models/als_v1.pkl")
    parser.add_argument("--metrics-output", default="reports/baseline_metrics.json")
    parser.add_argument("--backend", choices=["als", "svd"], default="als")
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--regularization", type=float, default=0.05)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)

    interactions = read_interactions(args.data)
    provenance = read_provenance_sidecar(args.data)
    train, test = temporal_leave_one_out(interactions)
    encoded = encode_train_test(train, test)
    if args.backend == "als":
        user_factors, item_factors = train_als(
            encoded.train_matrix,
            factors=args.factors,
            regularization=args.regularization,
            iterations=args.iterations,
            seed=args.seed,
        )
    else:
        user_factors, item_factors = train_svd(encoded.train_matrix, args.factors, args.seed)

    rankings = top_k_from_factors(user_factors, item_factors, encoded.train_matrix, args.k)
    metrics = evaluate_rankings(rankings, encoded, k=args.k)
    artifact = {
        "artifact_version": 1,
        "model_type": "als" if args.backend == "als" else "svd_smoke_test",
        "data_provenance": provenance["data_provenance"],
        "data_source": provenance["source"],
        "user_factors": user_factors,
        "item_factors": item_factors,
        "idx_to_user": encoded.idx_to_user,
        "idx_to_item": encoded.idx_to_item,
        "item_categories": encoded.item_categories,
        "user_to_idx": encoded.user_to_idx,
        "item_to_idx": encoded.item_to_idx,
        "category_transitions": category_transition_stats(train),
        "item_popularity": train.groupby("item_id")["interaction_strength"].sum().to_dict(),
        "user_category_history": {
            user_id: train.loc[train["user_id"] == user_id, "category"]
            .value_counts()
            .to_dict()
            for user_id in encoded.idx_to_user
        },
        "train_seen": {
            user_id: train.loc[train["user_id"] == user_id, "item_id"].unique().tolist()
            for user_id in encoded.idx_to_user
        },
        "metrics": {key: value for key, value in metrics.items() if key != "per_user"},
    }
    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    report = {key: value for key, value in metrics.items() if key != "per_user"}
    report.update(
        {
            "model_type": artifact["model_type"],
            "factors": int(user_factors.shape[1]),
            "train_interactions": int(encoded.train_matrix.nnz),
            "data_provenance": provenance["data_provenance"],
            "data_source": provenance["source"],
        }
    )
    if provenance["data_provenance"] == "synthetic_controlled":
        report["caution"] = SYNTHETIC_CAUTION
    dump_json(report, args.metrics_output)
    print(json.dumps(report, indent=2))
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
