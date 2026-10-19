"""Tests for the Financial Sentiment API."""

from fastapi.testclient import TestClient
from src.api import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"text": "The company reported higher profits."},
        )

    assert response.status_code == 200

    result = response.json()
    assert result["sentiment"] in {"negative", "neutral", "positive"}
    assert 0 <= result["confidence"] <= 1
