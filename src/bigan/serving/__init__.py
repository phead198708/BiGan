"""Serving API contracts and fail-closed inference helpers."""

from .contracts import (
    API_ENDPOINTS,
    ErrorResponse,
    HealthResponse,
    LatestPredictionResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    ServingStatus,
    api_contract,
    fixed_error,
)

__all__ = [
    "API_ENDPOINTS",
    "ErrorResponse",
    "HealthResponse",
    "LatestPredictionResponse",
    "ModelInfoResponse",
    "PredictRequest",
    "PredictResponse",
    "ServingStatus",
    "api_contract",
    "fixed_error",
]
