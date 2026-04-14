"""Tests for the model training module."""

import os
import pytest

from src.data_pipeline import load_config, run_pipeline
from src.train import train_model


@pytest.fixture(scope="module")
def trained_result():
    """Run the full pipeline once for all tests."""
    run_pipeline()
    pipeline, metrics = train_model()
    return pipeline, metrics


class TestTraining:
    def test_model_file_created(self, trained_result):
        config = load_config()
        assert os.path.exists(config["model"]["save_path"])

    def test_accuracy_above_threshold(self, trained_result):
        _, metrics = trained_result
        assert metrics["accuracy"] >= 0.70, (
            f"Accuracy {metrics['accuracy']} is below threshold 0.70"
        )

    def test_metrics_complete(self, trained_result):
        _, metrics = trained_result
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

    def test_model_can_predict(self, trained_result):
        pipeline, _ = trained_result
        prediction = pipeline.predict(["Breaking news reveals truth"])
        assert prediction[0] in [0, 1]

    def test_metrics_file_created(self, trained_result):
        config = load_config()
        assert os.path.exists(config["reports"]["metrics"])
