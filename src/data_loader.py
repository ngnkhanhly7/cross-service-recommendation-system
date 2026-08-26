"""Load every source into one canonical interaction schema.

The module deliberately keeps source-specific logic here. Training, evaluation and
serving only consume the five-column schema defined in :mod:`src.common`.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import SCHEMA, validate_interactions, write_interactions, write_provenance_sidecar

AMAZON_DEFAULT_CATEGORIES = [
    "All_Beauty",
    "Electronics",
    "Home_and_Kitchen",
    "Grocery_and_Gourmet_Food",
]


def _records_to_frame(records: Iterable[dict], category: str) -> pd.DataFrame:
    rows = []
    for row in records:
        user_id = row.get("user_id")
        item_id = row.get("parent_asin") or row.get("asin")
        timestamp = row.get("timestamp")
        rating = row.get("rating", 1.0)
        if user_id is None or item_id is None or timestamp is None:
            continue
        # Amazon timestamps are milliseconds since epoch; ISO strings also work.
        if isinstance(timestamp, (int, float, np.integer, np.floating)):
            timestamp = pd.to_datetime(timestamp, unit="ms", utc=True)
        rows.append(
            {
                "user_id": str(user_id),
                "item_id": f"amazon:{category}:{item_id}",
                "category": category,
                "timestamp": timestamp,
                "interaction_strength": float(rating),
            }
        )
    return pd.DataFrame(rows, columns=SCHEMA)


def load_amazon_reviews(
    categories: list[str] | None = None,
    *,
    max_rows_per_category: int = 250_000,
    streaming: bool = True,
) -> pd.DataFrame:
    """Stream selected Amazon Reviews 2023 categories from Hugging Face.

    ``max_rows_per_category`` is intentionally required for a laptop/Colab-friendly
    first run. Raise it gradually after the pipeline succeeds. The dataset config is
    named ``raw_review_<category>``.
    """

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional data dependency first: pip install datasets"
        ) from exc

    from tqdm import tqdm

    categories = categories or AMAZON_DEFAULT_CATEGORIES
    frames: list[pd.DataFrame] = []
    for category in categories:
        config = f"raw_review_{category}"
        print(f"Streaming {config} (up to {max_rows_per_category:,} rows)...", flush=True)
        dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            config,
            split="full",
            streaming=streaming,
            trust_remote_code=True,
        )
        iterator: Iterator[dict] = iter(dataset)
        records = []
        for index, row in enumerate(
            tqdm(iterator, total=max_rows_per_category, desc=category, unit="row")
        ):
            if index >= max_rows_per_category:
                break
            records.append(row)
        frames.append(_records_to_frame(records, category))
    if not frames:
        raise ValueError("At least one Amazon category is required")
    return validate_interactions(pd.concat(frames, ignore_index=True))


def iterative_sparsity_filter(
    df: pd.DataFrame,
    *,
    min_user_interactions: int = 5,
    min_item_interactions: int = 3,
    min_user_categories: int = 2,
) -> pd.DataFrame:
    """Apply a stable user-item k-core, then retain cross-category users.

    User and item filtering is repeated because removing one side can make the other
    side fall below its threshold.
    """

    result = validate_interactions(df)
    while True:
        before = len(result)
        user_count = result.groupby("user_id")["item_id"].transform("size")
        result = result[user_count >= min_user_interactions]
        item_count = result.groupby("item_id")["user_id"].transform("size")
        result = result[item_count >= min_item_interactions]
        if len(result) == before:
            break
        if result.empty:
            raise ValueError(
                "Sparsity filtering removed every row. Lower thresholds or sample "
                "more source data."
            )

    if min_user_categories > 1:
        category_count = result.groupby("user_id")["category"].transform("nunique")
        result = result[category_count >= min_user_categories]
    if result.empty:
        raise ValueError(
            "No multi-category users remain. Amazon category files have few shared "
            "users in this sample; increase max_rows_per_category."
        )
    return validate_interactions(result)


def summarize(df: pd.DataFrame) -> dict[str, float | int]:
    df = validate_interactions(df)
    category_counts = df.groupby("user_id")["category"].nunique()
    return {
        "interactions": int(len(df)),
        "users": int(df["user_id"].nunique()),
        "items": int(df["item_id"].nunique()),
        "categories": int(df["category"].nunique()),
        "multi_category_users": int((category_counts >= 2).sum()),
        "multi_category_user_pct": round(float((category_counts >= 2).mean() * 100), 2),
    }


def generate_demo_amazon(
    n_users: int = 250,
    n_items_per_category: int = 40,
    seed: int = 42,
) -> pd.DataFrame:
    """Small reproducible dataset with learnable cross-category affinity."""

    rng = np.random.default_rng(seed)
    categories = ["Electronics", "All_Beauty", "Home_and_Kitchen", "Grocery"]
    # Personas induce cross-category signal rather than independent random noise.
    persona_weights = np.array(
        [
            [0.45, 0.08, 0.37, 0.10],
            [0.08, 0.43, 0.12, 0.37],
            [0.18, 0.17, 0.40, 0.25],
        ]
    )
    base = pd.Timestamp("2023-01-01", tz="UTC")
    rows: list[dict] = []
    for user_index in range(n_users):
        persona = int(rng.integers(0, len(persona_weights)))
        n_events = int(rng.integers(10, 25))
        days = np.sort(rng.integers(0, 365, size=n_events))
        user_taste = int(rng.integers(0, 5))
        for event_index, day in enumerate(days):
            category_index = int(rng.choice(len(categories), p=persona_weights[persona]))
            category = categories[category_index]
            # Shared taste bucket makes item collaborative structure learnable.
            local_item = (user_taste * 8 + int(rng.integers(0, 8))) % n_items_per_category
            rows.append(
                {
                    "user_id": f"amazon_user_{user_index:05d}",
                    "item_id": f"amazon:{category}:item_{local_item:04d}",
                    "category": category,
                    "timestamp": base + pd.Timedelta(days=int(day), hours=event_index % 24),
                    "interaction_strength": float(rng.integers(3, 6)),
                }
            )
    return validate_interactions(pd.DataFrame(rows))


def generate_multi_service_simulation(n_users: int = 5_000, seed: int = 42) -> pd.DataFrame:
    """Generate explicitly synthetic, persona/time/region-constrained service events.

    This is a controlled simulation for algorithm and architecture validation. Journey
    patterns (e.g. mobility followed by hospitality) are injected by this generator, so
    a model recovering them proves the pipeline works, not that any real customer base
    behaves this way. It must not be presented as real customer data or business
    evidence for any specific company.
    """

    rng = np.random.default_rng(seed)
    services = ["xanh_sm", "vinpearl", "vinmec", "shopping"]
    regions = ["hanoi", "hochiminh", "nhatrang", "phuquoc"]
    persona_probs = {
        "weekend_traveler": np.array([0.42, 0.38, 0.05, 0.15]),
        "local_resident": np.array([0.28, 0.05, 0.34, 0.33]),
        "family": np.array([0.25, 0.24, 0.22, 0.29]),
    }
    persona_names = list(persona_probs)
    item_counts = {"xanh_sm": 50, "vinpearl": 80, "vinmec": 60, "shopping": 100}
    start = pd.Timestamp("2024-01-01", tz="UTC")
    rows: list[dict] = []

    for user_index in range(n_users):
        persona = str(rng.choice(persona_names, p=[0.34, 0.43, 0.23]))
        home_region = str(rng.choice(regions, p=[0.33, 0.35, 0.17, 0.15]))
        n_sessions = int(rng.integers(3, 9))
        user_taste = int(rng.integers(0, 10))
        session_starts = np.sort(rng.integers(0, 540, size=n_sessions))
        for session_index, day in enumerate(session_starts):
            n_events = int(rng.integers(2, 5))
            chosen = rng.choice(
                services,
                size=n_events,
                replace=True,
                p=persona_probs[persona],
            ).tolist()
            # Encode a meaningful journey: travel commonly starts with mobility and
            # reaches hospitality within three days in the same region.
            if persona == "weekend_traveler" and session_index % 2 == 0:
                chosen[:2] = ["xanh_sm", "vinpearl"]
            if persona == "local_resident" and session_index % 3 == 0:
                chosen[:2] = ["vinmec", "shopping"]
            for offset, service in enumerate(chosen):
                region = home_region
                item_count = item_counts[service]
                local_item = (user_taste * max(1, item_count // 10) + int(rng.integers(0, 8))) % item_count
                timestamp = start + pd.Timedelta(
                    days=int(day + min(offset, 3)), hours=int(rng.integers(7, 22))
                )
                rows.append(
                    {
                        "user_id": f"sim_user_{user_index:06d}",
                        "item_id": f"sim:{service}:{region}:item_{local_item:04d}",
                        "category": service,
                        "timestamp": timestamp,
                        "interaction_strength": float(rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1])),
                    }
                )
    return validate_interactions(pd.DataFrame(rows))


def normalize_service_csv(
    path: str | Path,
    *,
    category: str,
    user_column: str,
    item_column: str,
    timestamp_column: str,
    strength_column: str | None = None,
    item_prefix: str | None = None,
) -> pd.DataFrame:
    """Normalize one real service CSV after its identity-mapping policy is defined."""

    raw = pd.read_csv(path)
    required = {user_column, item_column, timestamp_column}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing source columns in {path}: {sorted(missing)}")
    prefix = item_prefix or category
    strength = raw[strength_column] if strength_column else 1.0
    normalized = pd.DataFrame(
        {
            "user_id": raw[user_column].astype(str),
            "item_id": prefix + ":" + raw[item_column].astype(str),
            "category": category,
            "timestamp": raw[timestamp_column],
            "interaction_strength": strength,
        }
    )
    return validate_interactions(normalized)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["demo-amazon", "amazon-hf", "multi-service-sim"],
        default="demo-amazon",
    )
    parser.add_argument("--categories", nargs="+", default=AMAZON_DEFAULT_CATEGORIES)
    parser.add_argument("--max-rows-per-category", type=int, default=250_000)
    parser.add_argument("--n-users", type=int, default=None)
    parser.add_argument("--min-user-interactions", type=int, default=5)
    parser.add_argument("--min-item-interactions", type=int, default=3)
    parser.add_argument("--min-user-categories", type=int, default=2)
    parser.add_argument("--output", default="data/processed/interactions.csv")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.source == "amazon-hf":
        raw = load_amazon_reviews(
            args.categories,
            max_rows_per_category=args.max_rows_per_category,
        )
    elif args.source == "multi-service-sim":
        raw = generate_multi_service_simulation(args.n_users or 5_000, args.seed)
    else:
        raw = generate_demo_amazon(args.n_users or 250, seed=args.seed)

    raw_summary = summarize(raw)
    filtered = iterative_sparsity_filter(
        raw,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
        min_user_categories=args.min_user_categories,
    )
    output = write_interactions(filtered, args.output)
    data_provenance = "public_dataset" if args.source == "amazon-hf" else "synthetic_controlled"
    sidecar = write_provenance_sidecar(output, source=args.source, data_provenance=data_provenance)
    print("Before filtering:", raw_summary)
    print("After filtering: ", summarize(filtered))
    print(f"Saved canonical interactions to {output}")
    print(f"Saved data provenance ({data_provenance}) to {sidecar}")


if __name__ == "__main__":
    main()
