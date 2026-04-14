"""
Prediction Module
CLI tool to predict whether a news headline is real or fake.
"""

import sys
import joblib
import numpy as np

from src.data_pipeline import load_config


def load_model(model_path):
    """Load trained model pipeline."""
    return joblib.load(model_path)


def predict(pipeline, title):
    """Predict whether a title is fake or real news."""
    prediction = pipeline.predict([title])[0]
    probabilities = pipeline.predict_proba([title])[0]
    confidence = float(np.max(probabilities))

    label = "Real News" if prediction == 1 else "Fake News"
    return {
        "title": title,
        "prediction": label,
        "confidence": round(confidence, 4),
        "probabilities": {
            "fake": round(float(probabilities[0]), 4),
            "real": round(float(probabilities[1]), 4),
        },
    }


def main():
    """CLI entrypoint."""
    config = load_config()
    pipeline = load_model(config["model"]["save_path"])

    if len(sys.argv) < 2:
        print("Usage: python -m src.predict \"Your news headline here\"")
        sys.exit(1)

    title = " ".join(sys.argv[1:])
    result = predict(pipeline, title)

    print("=" * 50)
    print("FAKE NEWS PREDICTION")
    print("=" * 50)
    print(f"Title:      {result['title']}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"P(Fake):    {result['probabilities']['fake']:.2%}")
    print(f"P(Real):    {result['probabilities']['real']:.2%}")


if __name__ == "__main__":
    main()
