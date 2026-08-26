from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib

CONTEXT_TARGETS = {
    "xanh_sm_trip_to_vinpearl": "vinpearl",
    "vinmec_appointment": "shopping",
    "electronics_purchase": "Home_and_Kitchen",
    "beauty_purchase": "Grocery",
}


class RecommendationEngine:
    def __init__(self, artifact: dict[str, Any]):
        required = {
            "user_factors",
            "item_factors",
            "user_to_idx",
            "idx_to_item",
            "item_categories",
            "train_seen",
        }
        missing = required - set(artifact)
        if missing:
            raise ValueError(f"Invalid model artifact; missing {sorted(missing)}")
        self.artifact = artifact

    @classmethod
    def load(cls, model_path: str | Path) -> RecommendationEngine:
        return cls(joblib.load(model_path))

    def recommend(
        self,
        user_id: str,
        *,
        top_k: int = 10,
        context: str | None = None,
        target_category: str | None = None,
        cross_category_only: bool = False,
    ) -> dict[str, Any]:
        """Return {"personalized": bool, "results": [...]}.

        An unknown user is not an error: it is the cross-sell moment the whole system
        exists for (a user seen in one service but with no trained factor yet). Callers
        get a popularity fallback instead of a hard failure, tagged ``personalized``
        so they can distinguish it from a model-backed ranking.
        """

        target_category = target_category or CONTEXT_TARGETS.get(context or "")
        user_to_idx = self.artifact["user_to_idx"]
        if user_id not in user_to_idx:
            return {
                "personalized": False,
                "results": self._popularity_fallback(top_k=top_k, target_category=target_category),
            }
        return {
            "personalized": True,
            "results": self._personalized(
                user_id,
                top_k=top_k,
                target_category=target_category,
                cross_category_only=cross_category_only,
            ),
        }

    def _personalized(
        self,
        user_id: str,
        *,
        top_k: int,
        target_category: str | None,
        cross_category_only: bool,
    ) -> list[dict[str, Any]]:
        user_idx = self.artifact["user_to_idx"][user_id]
        scores = (
            self.artifact["user_factors"][user_idx]
            @ self.artifact["item_factors"].T
        ).astype(float)
        seen = set(self.artifact["train_seen"].get(user_id, []))
        history = self.artifact.get("user_category_history", {}).get(user_id, {})
        dominant = max(history, key=history.get) if history else None
        candidates: list[int] = []
        for item_idx, (item_id, category) in enumerate(
            zip(self.artifact["idx_to_item"], self.artifact["item_categories"], strict=True)
        ):
            if item_id in seen:
                continue
            if target_category and category != target_category:
                continue
            if cross_category_only and dominant and category == dominant:
                continue
            candidates.append(item_idx)
        candidates.sort(key=lambda item_idx: scores[item_idx], reverse=True)
        results: list[dict[str, Any]] = []
        transitions = self.artifact.get("category_transitions", {})
        for item_idx in candidates[:top_k]:
            category = self.artifact["item_categories"][item_idx]
            probability = transitions.get(dominant, {}).get(category, 0.0)
            reason = (
                f"Your strongest history is {dominant}; {probability:.1f}% of users active "
                f"in {dominant} also interacted with {category}."
                if dominant
                else f"Popular candidate in {category}."
            )
            results.append(
                {
                    "rank": len(results) + 1,
                    "item_id": self.artifact["idx_to_item"][item_idx],
                    "category": category,
                    "score": round(float(scores[item_idx]), 6),
                    "reason": reason,
                }
            )
        return results

    def _popularity_fallback(
        self, *, top_k: int, target_category: str | None
    ) -> list[dict[str, Any]]:
        popularity: dict[str, float] = self.artifact.get("item_popularity", {})
        candidates = [
            (item_id, category)
            for item_id, category in zip(
                self.artifact["idx_to_item"], self.artifact["item_categories"], strict=True
            )
            if not target_category or category == target_category
        ]
        candidates.sort(key=lambda pair: popularity.get(pair[0], 0.0), reverse=True)
        results: list[dict[str, Any]] = []
        for item_id, category in candidates[:top_k]:
            results.append(
                {
                    "rank": len(results) + 1,
                    "item_id": item_id,
                    "category": category,
                    "score": round(float(popularity.get(item_id, 0.0)), 6),
                    "reason": (
                        f"No interaction history yet for this user; showing a popular "
                        f"item in {category} as a starting point."
                        if target_category
                        else "No interaction history yet for this user; showing a "
                        "popular item overall."
                    ),
                }
            )
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend items for one known user")
    parser.add_argument("user_id")
    parser.add_argument("--model", default="models/two_tower_v2.pkl")
    parser.add_argument("--context")
    parser.add_argument("--target-category")
    parser.add_argument("--cross-category-only", action="store_true")
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()
    engine = RecommendationEngine.load(args.model)
    outcome = engine.recommend(
        args.user_id,
        top_k=args.k,
        context=args.context,
        target_category=args.target_category,
        cross_category_only=args.cross_category_only,
    )
    print(json.dumps(outcome, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

