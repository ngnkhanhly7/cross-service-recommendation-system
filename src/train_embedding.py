from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from src.common import (
    SYNTHETIC_CAUTION,
    dump_json,
    read_interactions,
    read_provenance_sidecar,
    seed_everything,
)
from src.evaluate import evaluate_rankings, top_k_from_factors
from src.features import (
    category_history_matrix,
    category_transition_stats,
    encode_train_test,
    temporal_leave_one_out,
)


def train_two_tower(
    encoded,
    *,
    embedding_dim: int = 64,
    epochs: int = 15,
    batch_size: int = 2048,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    category_weight: float = 0.2,
    device_name: str = "auto",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("Two-Tower training needs PyTorch: pip install torch") from exc

    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else
        "cpu" if device_name == "auto" else device_name
    )
    history, categories = category_history_matrix(encoded)
    category_to_idx = {category: index for index, category in enumerate(categories)}
    item_category_indices = np.array(
        [category_to_idx[category] for category in encoded.item_categories], dtype=np.int64
    )

    class TwoTower(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.user_embedding = nn.Embedding(len(encoded.idx_to_user), embedding_dim)
            self.item_embedding = nn.Embedding(len(encoded.idx_to_item), embedding_dim)
            self.category_embedding = nn.Embedding(len(categories), embedding_dim)
            # A direct retrieval space is more stable than a deep MLP for sparse IDs.
            # Category vectors are shared by both towers, which lets a user's mixed
            # category history influence candidates from another category.
            nn.init.normal_(self.user_embedding.weight, std=0.05)
            nn.init.normal_(self.item_embedding.weight, std=0.05)
            nn.init.normal_(self.category_embedding.weight, std=0.05)

        def user_tower(self, users, category_history):
            category_context = category_history @ self.category_embedding.weight
            return self.user_embedding(users) + category_weight * category_context

        def item_tower(self, items, item_categories):
            return self.item_embedding(items) + category_weight * self.category_embedding(item_categories)

    model = TwoTower().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    rows, cols = encoded.train_matrix.nonzero()
    positive_users = torch.as_tensor(rows, dtype=torch.long)
    positive_items = torch.as_tensor(cols, dtype=torch.long)
    history_tensor = torch.as_tensor(history, dtype=torch.float32, device=device)
    item_category_tensor = torch.as_tensor(item_category_indices, dtype=torch.long, device=device)
    generator = torch.Generator().manual_seed(seed)
    losses: list[float] = []

    for epoch in range(epochs):
        permutation = torch.randperm(len(positive_users), generator=generator)
        epoch_loss = 0.0
        examples = 0
        model.train()
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            users = positive_users[indices].to(device)
            positives = positive_items[indices].to(device)
            negatives = torch.randint(
                0, len(encoded.idx_to_item), positives.shape, generator=generator
            ).to(device)
            collision = negatives == positives
            negatives[collision] = (negatives[collision] + 1) % len(encoded.idx_to_item)

            user_vectors = model.user_tower(users, history_tensor[users])
            positive_vectors = model.item_tower(
                positives, item_category_tensor[positives]
            )
            negative_vectors = model.item_tower(
                negatives, item_category_tensor[negatives]
            )
            positive_scores = (user_vectors * positive_vectors).sum(dim=1)
            negative_scores = (user_vectors * negative_vectors).sum(dim=1)
            loss = -functional.logsigmoid(positive_scores - negative_scores).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(indices)
            examples += len(indices)
        mean_loss = epoch_loss / max(examples, 1)
        losses.append(mean_loss)
        print(f"epoch={epoch + 1:02d}/{epochs} bpr_loss={mean_loss:.5f} device={device}")

    model.eval()
    with torch.no_grad():
        all_users = torch.arange(len(encoded.idx_to_user), device=device)
        all_items = torch.arange(len(encoded.idx_to_item), device=device)
        user_factors = model.user_tower(all_users, history_tensor).cpu().numpy().astype("float32")
        item_factors = model.item_tower(
            all_items, item_category_tensor
        ).cpu().numpy().astype("float32")
    return user_factors, item_factors, losses


def save_faiss_index(item_factors: np.ndarray, output: str | Path) -> bool:
    try:
        import faiss
    except ImportError:
        print("FAISS is unavailable; serving will use exact NumPy retrieval.")
        return False
    index = faiss.IndexFlatIP(item_factors.shape[1])
    index.add(np.ascontiguousarray(item_factors.astype("float32")))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output))
    return True


def write_comparison(
    baseline_metrics_path: str, two_tower_metrics: dict, output: str, *, data_provenance: str
) -> None:
    try:
        baseline = json.loads(Path(baseline_metrics_path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        baseline = None
    k = two_tower_metrics["k"]
    lines = [
        "# Model comparison",
        "",
        f"| Model | Recall@{k} overall | NDCG@{k} overall | Recall@{k} cross-category | NDCG@{k} cross-category |",
        "|---|---:|---:|---:|---:|",
    ]
    if baseline:
        lines.append(
            f"| ALS | {baseline['overall'][f'recall@{k}']:.4f} | "
            f"{baseline['overall'][f'ndcg@{k}']:.4f} | "
            f"{baseline['cross_category'][f'recall@{k}']:.4f} | "
            f"{baseline['cross_category'][f'ndcg@{k}']:.4f} |"
        )
    lines.append(
        f"| Two-Tower | {two_tower_metrics['overall'][f'recall@{k}']:.4f} | "
        f"{two_tower_metrics['overall'][f'ndcg@{k}']:.4f} | "
        f"{two_tower_metrics['cross_category'][f'recall@{k}']:.4f} | "
        f"{two_tower_metrics['cross_category'][f'ndcg@{k}']:.4f} |"
    )
    lines += [
        "",
        "The cross-category test contains only users whose held-out item category differs "
        "from their dominant training-history category.",
        "",
        f"**Data provenance:** `{data_provenance}`.",
    ]
    if data_provenance == "synthetic_controlled":
        lines.append(f"**Caution:** {SYNTHETIC_CAUTION}")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a category-aware Two-Tower model")
    parser.add_argument("--data", default="data/processed/interactions.csv")
    parser.add_argument("--model-output", default="models/two_tower_v2.pkl")
    parser.add_argument("--index-output", default="models/two_tower_v2.faiss")
    parser.add_argument("--metrics-output", default="reports/two_tower_metrics.json")
    parser.add_argument("--comparison-output", default="reports/model_comparison.md")
    parser.add_argument("--baseline-metrics", default="reports/baseline_metrics.json")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--category-weight",
        type=float,
        default=0.2,
        help="Strength of category side-information relative to ID embeddings",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)

    interactions = read_interactions(args.data)
    provenance = read_provenance_sidecar(args.data)
    train, test = temporal_leave_one_out(interactions)
    encoded = encode_train_test(train, test)
    user_factors, item_factors, losses = train_two_tower(
        encoded,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        category_weight=args.category_weight,
        device_name=args.device,
        seed=args.seed,
    )
    rankings = top_k_from_factors(user_factors, item_factors, encoded.train_matrix, args.k)
    metrics = evaluate_rankings(rankings, encoded, k=args.k)
    artifact = {
        "artifact_version": 1,
        "model_type": "two_tower",
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
        "training_loss": losses,
        "category_weight": args.category_weight,
    }
    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    faiss_saved = save_faiss_index(item_factors, args.index_output)
    report = {key: value for key, value in metrics.items() if key != "per_user"}
    report.update(
        {
            "model_type": "two_tower",
            "embedding_dim": int(item_factors.shape[1]),
            "epochs": args.epochs,
            "category_weight": args.category_weight,
            "final_bpr_loss": round(losses[-1], 6),
            "faiss_index_saved": faiss_saved,
            "data_provenance": provenance["data_provenance"],
            "data_source": provenance["source"],
        }
    )
    if provenance["data_provenance"] == "synthetic_controlled":
        report["caution"] = SYNTHETIC_CAUTION
    dump_json(report, args.metrics_output)
    write_comparison(
        args.baseline_metrics,
        report,
        args.comparison_output,
        data_provenance=provenance["data_provenance"],
    )
    print(json.dumps(report, indent=2))
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
