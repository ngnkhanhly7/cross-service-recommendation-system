import pytest
from fastapi.testclient import TestClient

from src.api import main


@pytest.fixture(autouse=True)
def _reset_rate_limit_state():
    main._request_log.clear()
    yield
    main._request_log.clear()


class _FakeEngine:
    def __init__(self, *, personalized: bool, data_provenance: str = "synthetic_controlled"):
        self._personalized = personalized
        self.artifact = {"model_type": "fake", "data_provenance": data_provenance}

    def recommend(self, user_id, *, top_k, context, target_category, cross_category_only):
        results = [
            {
                "rank": 1,
                "item_id": "item_0001",
                "category": "vinpearl",
                "score": 0.9,
                "reason": "test reason",
            }
        ]
        return {"personalized": self._personalized, "results": results}


def test_health():
    client = TestClient(main.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommend_known_user_is_personalized_and_flags_synthetic_data(monkeypatch):
    monkeypatch.setattr(main, "get_engine", lambda: _FakeEngine(personalized=True))
    client = TestClient(main.app)
    response = client.get("/recommend/u1")
    assert response.status_code == 200
    body = response.json()
    assert body["personalized"] is True
    assert body["data_provenance"] == "synthetic_controlled"
    assert body["caution"] is not None
    assert body["recommendations"][0]["item_id"] == "item_0001"


def test_recommend_unknown_user_returns_200_not_404(monkeypatch):
    monkeypatch.setattr(main, "get_engine", lambda: _FakeEngine(personalized=False))
    client = TestClient(main.app)
    response = client.get("/recommend/never_seen_user?context=xanh_sm_trip_to_vinpearl")
    assert response.status_code == 200
    assert response.json()["personalized"] is False


def test_recommend_real_data_has_no_caution(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_engine",
        lambda: _FakeEngine(personalized=True, data_provenance="public_dataset"),
    )
    client = TestClient(main.app)
    response = client.get("/recommend/u1")
    assert response.json()["caution"] is None


def test_recommend_missing_model_returns_503(monkeypatch):
    def _raise():
        raise FileNotFoundError("no model on disk")

    monkeypatch.setattr(main, "get_engine", _raise)
    client = TestClient(main.app)
    response = client.get("/recommend/u1")
    assert response.status_code == 503


def test_recommend_rejects_out_of_range_top_k(monkeypatch):
    monkeypatch.setattr(main, "get_engine", lambda: _FakeEngine(personalized=True))
    client = TestClient(main.app)
    response = client.get("/recommend/u1?top_k=1000")
    assert response.status_code == 422


def test_reload_model_requires_api_key_when_configured(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    client = TestClient(main.app)
    response = client.post("/reload-model")
    assert response.status_code == 401


def test_reload_model_succeeds_with_correct_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    monkeypatch.setattr(main, "get_engine", lambda: _FakeEngine(personalized=True))
    client = TestClient(main.app)
    response = client.post("/reload-model", headers={"X-API-Key": "secret123"})
    assert response.status_code == 200
    assert response.json()["reloaded"] is True


def test_reload_model_open_when_no_api_key_configured(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setattr(main, "get_engine", lambda: _FakeEngine(personalized=True))
    client = TestClient(main.app)
    response = client.post("/reload-model")
    assert response.status_code == 200


def test_rate_limit_blocks_after_threshold(monkeypatch):
    monkeypatch.setattr(main, "_RATE_LIMIT_PER_MINUTE", 2)
    monkeypatch.setattr(main, "get_engine", lambda: _FakeEngine(personalized=True))
    client = TestClient(main.app)
    assert client.get("/recommend/u1").status_code == 200
    assert client.get("/recommend/u1").status_code == 200
    assert client.get("/recommend/u1").status_code == 429
