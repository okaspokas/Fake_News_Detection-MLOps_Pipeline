"""
Model Training Module
Trains a TF-IDF + Logistic Regression pipeline for fake news detection.
"""

import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from src.data_pipeline import load_config


def load_processed_data(filepath):
    """Load the cleaned/processed dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Processed data not found at: {filepath}. "
            "Run the data pipeline first: python -m src.data_pipeline"
        )
    df = pd.read_csv(filepath)
    print(f"[TRAIN] Loaded {len(df)} rows from {filepath}")
    return df


def build_pipeline(config):
    """Build the sklearn pipeline."""
    tfidf = TfidfVectorizer(
        max_features=config["model"]["tfidf_max_features"],
        stop_words=config["model"]["tfidf_stop_words"],
    )
    model = LogisticRegression(max_iter=1000, random_state=config["model"]["random_state"])
    pipeline = Pipeline([("tfidf", tfidf), ("model", model)])
    return pipeline


def train_model(config_path="config.yaml"):
    """Execute the full training pipeline."""
    config = load_config(config_path)

    print("=" * 50)
    print("MODEL TRAINING")
    print("=" * 50)

    df = load_processed_data(config["data"]["processed_path"])

    X = df["title"]
    y = df["real"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
    )
    print(f"[TRAIN] Train set: {len(X_train)} | Test set: {len(X_test)}")

    pipeline = build_pipeline(config)
    print("[TRAIN] Training TF-IDF + Logistic Regression pipeline...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average="weighted"), 4),
        "recall": round(recall_score(y_test, y_pred, average="weighted"), 4),
        "f1_score": round(f1_score(y_test, y_pred, average="weighted"), 4),
    }

    print(f"[TRAIN] Accuracy:  {metrics['accuracy']}")
    print(f"[TRAIN] Precision: {metrics['precision']}")
    print(f"[TRAIN] Recall:    {metrics['recall']}")
    print(f"[TRAIN] F1 Score:  {metrics['f1_score']}")

    print("\n[TRAIN] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

    model_path = config["model"]["save_path"]
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"[TRAIN] Model saved to {model_path}")

    reports_dir = config["reports"]["dir"]
    os.makedirs(reports_dir, exist_ok=True)
    metrics_path = config["reports"]["metrics"]
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[TRAIN] Metrics saved to {metrics_path}")

    return pipeline, metrics


if __name__ == "__main__":
    train_model()
