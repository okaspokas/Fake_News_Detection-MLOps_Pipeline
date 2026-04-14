"""
Model Evaluation Module
Generates confusion matrix, classification report, and evaluation metrics.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.data_pipeline import load_config


def load_model(model_path):
    """Load a saved model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at: {model_path}. "
            "Run training first: python -m src.train"
        )
    return joblib.load(model_path)


def generate_confusion_matrix(y_test, y_pred, save_path):
    """Generate and save confusion matrix plot."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fake", "Real"],
                yticklabels=["Fake", "Real"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[EVAL] Confusion matrix saved to {save_path}")


def evaluate_model(config_path="config.yaml"):
    """Run full evaluation on the test set."""
    config = load_config(config_path)

    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    pipeline = load_model(config["model"]["save_path"])

    df = pd.read_csv(config["data"]["processed_path"])
    X = df["title"]
    y = df["real"]

    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
    )

    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average="weighted"), 4),
        "recall": round(recall_score(y_test, y_pred, average="weighted"), 4),
        "f1_score": round(f1_score(y_test, y_pred, average="weighted"), 4),
    }

    print(f"[EVAL] Accuracy:  {metrics['accuracy']}")
    print(f"[EVAL] Precision: {metrics['precision']}")
    print(f"[EVAL] Recall:    {metrics['recall']}")
    print(f"[EVAL] F1 Score:  {metrics['f1_score']}")

    report = classification_report(
        y_test, y_pred,
        target_names=["Fake", "Real"],
        output_dict=True,
    )
    print("\n[EVAL] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

    reports_dir = config["reports"]["dir"]
    os.makedirs(reports_dir, exist_ok=True)

    report_path = config["reports"]["classification_report"]
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[EVAL] Classification report saved to {report_path}")

    metrics_path = config["reports"]["metrics"]
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[EVAL] Metrics saved to {metrics_path}")

    cm_path = config["reports"]["confusion_matrix"]
    generate_confusion_matrix(y_test, y_pred, cm_path)

    print("[EVAL] Evaluation complete!")
    return metrics


if __name__ == "__main__":
    evaluate_model()
