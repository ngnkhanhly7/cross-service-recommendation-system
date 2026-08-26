from __future__ import annotations

import logging
import os
import time
import uuid
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from pydantic import BaseModel

from src.common import SYNTHETIC_CAUTION
from src.serve import RecommendationEngine

logger = logging.getLogger("cross_service_rec.api")

app = FastAPI(
    title="Cross-service Recommendation API",
    version="0.1.0",
    description="Top-K recommendations with optional context/category constraints.",
)


class Recommendation(BaseModel):
    rank: int
    item_id: str
    category: str
    score: float
    reason: str


class RecommendationResponse(BaseModel):
    user_id: str
    model_type: str
    context: str | None
    personalized: bool
    data_provenance: str
    caution: str | None
    recommendations: list[Recommendation]


class ReloadResponse(BaseModel):
    reloaded: bool
    model_path: str
    model_type: str


def resolve_model_path() -> Path:
    configured = os.getenv("MODEL_PATH")
    if configured:
        return Path(configured)
    two_tower = Path("models/two_tower_v2.pkl")
    return two_tower if two_tower.exists() else Path("models/als_v1.pkl")


@lru_cache(maxsize=1)
def get_engine() -> RecommendationEngine:
    model_path = resolve_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained artifact at {model_path}. Train ALS or Two-Tower before serving."
        )
    return RecommendationEngine.load(model_path)


# --- Minimal operational guardrails -----------------------------------------
#
# This is intentionally dependency-light (no slowapi/redis): the goal is to close the
# most obvious gaps for a small internal deployment, not to build a full API gateway.
# Anything above single-instance rate limiting should move to a proper gateway.


def _api_key_configured() -> str | None:
    return os.getenv("API_KEY")


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    configured = _api_key_configured()
    if configured is None:
        return
    if x_api_key != configured:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")


_RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
_request_log: dict[str, deque[float]] = defaultdict(deque)


def enforce_rate_limit(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    if _RATE_LIMIT_PER_MINUTE <= 0:
        return
    client_key = x_api_key or (request.client.host if request.client else "unknown")
    now = time.monotonic()
    window = _request_log[client_key]
    while window and now - window[0] > 60.0:
        window.popleft()
    if len(window) >= _RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    window.append(now)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get(
    "/recommend/{user_id}",
    response_model=RecommendationResponse,
    dependencies=[Depends(enforce_rate_limit)],
)
def recommend(
    user_id: str,
    context: str | None = None,
    target_category: str | None = None,
    cross_category_only: bool = False,
    top_k: int = Query(default=10, ge=1, le=100),
) -> RecommendationResponse:
    request_id = uuid.uuid4().hex[:12]
    started = time.monotonic()
    try:
        engine = get_engine()
        outcome = engine.recommend(
            user_id,
            top_k=top_k,
            context=context,
            target_category=target_category,
            cross_category_only=cross_category_only,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    data_provenance = engine.artifact.get("data_provenance", "unknown")
    model_type = engine.artifact.get("model_type", "unknown")
    latency_ms = round((time.monotonic() - started) * 1000, 2)
    logger.info(
        "recommend request_id=%s user_id=%s model_type=%s personalized=%s "
        "data_provenance=%s latency_ms=%s",
        request_id,
        user_id,
        model_type,
        outcome["personalized"],
        data_provenance,
        latency_ms,
    )
    return RecommendationResponse(
        user_id=user_id,
        model_type=model_type,
        context=context,
        personalized=outcome["personalized"],
        data_provenance=data_provenance,
        caution=SYNTHETIC_CAUTION if data_provenance == "synthetic_controlled" else None,
        recommendations=outcome["results"],
    )


@app.post(
    "/reload-model",
    response_model=ReloadResponse,
    dependencies=[Depends(require_api_key)],
)
def reload_model() -> ReloadResponse:
    """Pick up a new artifact from MODEL_PATH without restarting the process.

    Does not remove any user from an already-served artifact (see
    docs/DATA_GOVERNANCE.md section 3) - it only swaps which trained file is served.
    """

    if hasattr(get_engine, "cache_clear"):
        get_engine.cache_clear()
    try:
        engine = get_engine()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    logger.info("model reloaded path=%s model_type=%s", resolve_model_path(), engine.artifact.get("model_type"))
    return ReloadResponse(
        reloaded=True,
        model_path=str(resolve_model_path()),
        model_type=engine.artifact.get("model_type", "unknown"),
    )
