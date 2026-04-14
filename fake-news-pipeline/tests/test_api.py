"""Tests for the FastAPI application."""

import os
import pytest
import joblib
from fastapi.testclient import TestClient

from src.data_pipeline import load_config, run_pipeline
from src.train import train_model


@pytest.fixture(scope="module", autouse=True)
def setup_model():
    """Ensure model exists before testing API."""
    config = load_config()
    if not os.path.exists(config["model"]["save_path"]):
        run_pipeline()
        train_model()

    # Also ensure the api.app module has the model loaded
    import api.app as api_module
    api_module.model_pipeline = joblib.load(config["model"]["save_path"])


@pytest.fixture
def client():
    from api.app import app
    return TestClient(app)


class TestAPI:
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_predict_real_news(self, client):
        response = client.post(
            "/predict",
            json={"title": "Government announces new economic policy"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in ["Real News", "Fake News"]
        assert 0 <= data["confidence"] <= 1
        assert "probabilities" in data

    def test_predict_fake_news(self, client):
        response = client.post(
            "/predict",
            json={"title": "Aliens land in New York confirmed by NASA"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in ["Real News", "Fake News"]

    def test_predict_empty_title_fails(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_response_structure(self, client):
        response = client.post(
            "/predict",
            json={"title": "Test headline"},
        )
        data = response.json()
        assert "title" in data
        assert "prediction" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "fake" in data["probabilities"]
        assert "real" in data["probabilities"]
