"""Tests for the FastAPI inference service."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parents[1]))


# ── Patch the predictor before importing the app ────────────────────────────
@pytest.fixture(autouse=True)
def mock_predictor():
    """Replace the real predictor with a deterministic mock."""
    mock = MagicMock()
    mock.predict.return_value = ("positive", 0.93)
    with patch("api.main.get_predictor", return_value=mock):
        yield mock


@pytest.fixture
def client(mock_predictor):
    from api.main import app
    return TestClient(app)


# ── /health ──────────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_schema(client):
    data = client.get("/health").json()
    assert "status" in data
    assert "model_loaded" in data


# ── /predict ─────────────────────────────────────────────────────────────────

def test_predict_positive(client):
    response = client.post("/predict", json={"review": "This was an amazing book!"})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "positive"
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_returns_model_type(client):
    data = client.post("/predict", json={"review": "Great story overall."}).json()
    assert "model_type" in data


def test_predict_rejects_short_review(client):
    response = client.post("/predict", json={"review": "ok"})
    assert response.status_code == 422


def test_predict_rejects_empty_review(client):
    response = client.post("/predict", json={"review": ""})
    assert response.status_code == 422


def test_predict_strips_whitespace(client, mock_predictor):
    client.post("/predict", json={"review": "   great book   "})
    args, _ = mock_predictor.predict.call_args
    assert args[0] == "great book"


# ── /batch ───────────────────────────────────────────────────────────────────

def test_batch_predict_multiple(client):
    reviews = ["Great read!", "Terrible book.", "Decent story."]
    response = client.post("/batch", json={"reviews": reviews})
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == len(reviews)


def test_batch_predict_empty_list_rejected(client):
    response = client.post("/batch", json={"reviews": []})
    assert response.status_code == 422


# ── /model-info ──────────────────────────────────────────────────────────────

def test_model_info(client):
    data = client.get("/model-info").json()
    assert "model_type" in data
    assert "artifacts_dir" in data
