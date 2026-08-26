from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.common import read_interactions


def category_cooccurrence(df: pd.DataFrame) -> pd.DataFrame:
    presence = pd.crosstab(df["user_id"], df["category"]).gt(0).astype(int)
    cooccurrence = presence.T @ presence
    return cooccurrence


def run_eda(data_path: str, output_dir: str) -> None:
    df = read_interactions(data_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    counts = df["category"].value_counts().sort_values(ascending=False)
    counts.rename_axis("category").rename("interactions").to_csv(
        output / "category_interactions.csv"
    )
    fig, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(x=counts.values, y=counts.index, ax=axis, color="#2f80ed")
    axis.set(title="Interactions by category", xlabel="Interactions", ylabel="Category")
    fig.tight_layout()
    fig.savefig(output / "category_interactions.png", dpi=160)
    plt.close(fig)

    cooccurrence = category_cooccurrence(df)
    cooccurrence.to_csv(output / "category_cooccurrence.csv")
    fig, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(cooccurrence, annot=True, fmt="g", cmap="Blues", ax=axis)
    axis.set_title("Users shared by category pair")
    fig.tight_layout()
    fig.savefig(output / "category_cooccurrence.png", dpi=160)
    plt.close(fig)
    print(f"Saved EDA tables and figures to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-category exploratory analysis")
    parser.add_argument("--data", default="data/processed/interactions.csv")
    parser.add_argument("--output-dir", default="reports/eda")
    args = parser.parse_args()
    run_eda(args.data, args.output_dir)


if __name__ == "__main__":
    main()
