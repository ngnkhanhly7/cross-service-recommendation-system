from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from src.common import validate_interactions


@dataclass(frozen=True)
class EncodedData:
    train: pd.DataFrame
    test: pd.DataFrame
    user_to_idx: dict[str, int]
    item_to_idx: dict[str, int]
    idx_to_user: list[str]
    idx_to_item: list[str]
    item_categories: list[str]
    train_matrix: sparse.csr_matrix


def temporal_leave_one_out(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out each user's most recent unique item.

    Repeated user-item events are aggregated before splitting. This prevents a held
    out item from also appearing in that user's training history, which would make
    filtered ranking evaluation ill-defined.
    """

    clean = validate_interactions(df)
    unique = (
        clean.groupby(["user_id", "item_id", "category"], as_index=False)
        .agg(timestamp=("timestamp", "max"), interaction_strength=("interaction_strength", "sum"))
        .sort_values(["user_id", "timestamp", "item_id"])
    )
    eligible = unique.groupby("user_id")["item_id"].transform("size") >= 2
    unique = unique[eligible].copy()
    if unique.empty:
        raise ValueError("Temporal split requires users with at least two unique items")
    test_indices = unique.groupby("user_id", sort=False).tail(1).index
    test = unique.loc[test_indices].copy()
    train = unique.drop(index=test_indices).copy()
    # Retrieval models cannot score a genuinely unseen item ID. Keep the offline
    # protocol focused on warm-item ranking; cold-item evaluation is a separate task.
    warm_items = set(train["item_id"])
    test = test[test["item_id"].isin(warm_items)].copy()
    if test.empty:
        raise ValueError("No warm held-out items remain after the temporal split")
    return validate_interactions(train), validate_interactions(test)


def encode_train_test(train: pd.DataFrame, test: pd.DataFrame) -> EncodedData:
    train = validate_interactions(train)
    test = validate_interactions(test)
    # Include test items in the item vocabulary. They may have been observed from
    # other users even though they are held out for this user.
    idx_to_user = sorted(set(train["user_id"]) & set(test["user_id"]))
    catalog = pd.concat(
        [train[["item_id", "category"]], test[["item_id", "category"]]], ignore_index=True
    ).drop_duplicates("item_id", keep="last")
    catalog = catalog.sort_values("item_id")
    idx_to_item = catalog["item_id"].tolist()
    item_categories = catalog["category"].tolist()
    user_to_idx = {value: index for index, value in enumerate(idx_to_user)}
    item_to_idx = {value: index for index, value in enumerate(idx_to_item)}

    encoded = train[train["user_id"].isin(user_to_idx)].copy()
    rows = encoded["user_id"].map(user_to_idx).to_numpy()
    cols = encoded["item_id"].map(item_to_idx).to_numpy()
    values = encoded["interaction_strength"].astype("float32").to_numpy()
    matrix = sparse.coo_matrix(
        (values, (rows, cols)), shape=(len(idx_to_user), len(idx_to_item)), dtype=np.float32
    ).tocsr()
    return EncodedData(
        train=train,
        test=test,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        idx_to_user=idx_to_user,
        idx_to_item=idx_to_item,
        item_categories=item_categories,
        train_matrix=matrix,
    )


def category_history_matrix(encoded: EncodedData) -> tuple[np.ndarray, list[str]]:
    categories = sorted(set(encoded.item_categories))
    category_to_idx = {category: index for index, category in enumerate(categories)}
    history = np.zeros((len(encoded.idx_to_user), len(categories)), dtype=np.float32)
    for row in encoded.train.itertuples(index=False):
        user_idx = encoded.user_to_idx.get(row.user_id)
        if user_idx is not None:
            history[user_idx, category_to_idx[row.category]] += float(row.interaction_strength)
    totals = history.sum(axis=1, keepdims=True)
    history = np.divide(history, totals, out=np.zeros_like(history), where=totals > 0)
    return history, categories


def category_transition_stats(train: pd.DataFrame) -> dict[str, dict[str, float]]:
    """P(target category present | source category present), in percent."""

    user_categories = train.groupby("user_id")["category"].agg(lambda x: set(x))
    categories = sorted(train["category"].unique())
    result: dict[str, dict[str, float]] = {}
    for source in categories:
        source_users = user_categories[
            user_categories.map(lambda values, source=source: source in values)
        ]
        denominator = len(source_users)
        result[source] = {}
        for target in categories:
            numerator = int(
                source_users.map(lambda values, target=target: target in values).sum()
            )
            result[source][target] = round(100.0 * numerator / denominator, 1) if denominator else 0.0
    return result
