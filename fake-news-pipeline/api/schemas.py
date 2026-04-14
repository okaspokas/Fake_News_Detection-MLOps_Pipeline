"""
Pydantic schemas for the FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from typing import Dict


class PredictionRequest(BaseModel):
    """Request body for /predict endpoint."""
    title: str = Field(
        ...,
        description="The news headline to classify",
        examples=["Breaking: Scientists discover new planet"],
    )


class ProbabilityDetail(BaseModel):
    """Probability breakdown for each class."""
    fake: float = Field(..., description="Probability of being fake news")
    real: float = Field(..., description="Probability of being real news")


class PredictionResponse(BaseModel):
    """Response body for /predict endpoint."""
    title: str = Field(..., description="The input headline")
    prediction: str = Field(..., description="Predicted label: 'Real News' or 'Fake News'")
    confidence: float = Field(..., description="Confidence score (0-1)")
    probabilities: ProbabilityDetail = Field(..., description="Class probabilities")


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""
    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")


class MetricsResponse(BaseModel):
    """Response body for /metrics endpoint."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
