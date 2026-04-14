"""
FastAPI Application
Serves the trained fake news detection model via REST API.
"""

import os
import json
import joblib
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    ProbabilityDetail,
    HealthResponse,
    MetricsResponse,
)
from src.predict import predict as run_prediction
from src.data_pipeline import load_config


config = load_config()
model_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model_pipeline
    model_path = config["model"]["save_path"]
    if os.path.exists(model_path):
        model_pipeline = joblib.load(model_path)
        print(f"[API] Model loaded from {model_path}")
    else:
        print(f"[API] WARNING: No model found at {model_path}")
    yield
    print("[API] Shutting down...")


app = FastAPI(
    title="Fake News Detection API",
    description="End-to-end ML pipeline for detecting fake news from headlines.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API and model are operational."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_pipeline is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict whether a news headline is real or fake."""
    if model_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first.",
        )

    result = run_prediction(model_pipeline, request.title)

    return PredictionResponse(
        title=result["title"],
        prediction=result["prediction"],
        confidence=result["confidence"],
        probabilities=ProbabilityDetail(
            fake=result["probabilities"]["fake"],
            real=result["probabilities"]["real"],
        ),
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Return the latest model evaluation metrics."""
    metrics_path = config["reports"]["metrics"]
    if not os.path.exists(metrics_path):
        raise HTTPException(
            status_code=404,
            detail="No metrics found. Run evaluation first.",
        )

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    return MetricsResponse(**metrics)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host=config["api"]["host"],
        port=config["api"]["port"],
        reload=True,
    )
