"""Online inference API contract models (issue #43)."""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

API_ENDPOINTS: dict[str, dict[str, str]] = {
    "health": {"method": "GET", "path": "/health"},
    "model_info": {"method": "GET", "path": "/model-info"},
    "predict": {"method": "POST", "path": "/predict"},
    "latest_prediction": {"method": "GET", "path": "/latest-prediction"},
}

ServingStatus = Literal["ok", "degraded", "unhealthy"]


def _now_ms() -> int:
    return int(time.time() * 1000)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErrorResponse(_StrictModel):
    """Stable error response envelope."""

    error_code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None


class HealthResponse(_StrictModel):
    """Response for GET /health."""

    status: ServingStatus
    model_version: str | None = None
    checks: dict[str, bool]
    server_time_ms: int = Field(default_factory=_now_ms)


class ModelInfoResponse(_StrictModel):
    """Response for GET /model-info."""

    model_version: str
    model_family: str
    feature_version: str
    dataset_version: str | None = None
    calibration_method: str | None = None
    status: str
    artifact_uri: str
    loaded_at: int


class PredictRequest(_StrictModel):
    """Request body for POST /predict."""

    source: str = "polymarket"
    source_symbol: str
    feature_version: str
    features: dict[str, float]
    request_id: str | None = None

    @field_validator("features")
    @classmethod
    def _features_are_non_empty(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("features must be non-empty")
        return value


class PredictResponse(_StrictModel):
    """Response body for POST /predict."""

    prob_up_15m: float = Field(ge=0.0, le=1.0)
    model_version: str
    feature_version: str
    confidence_bucket: str
    top_features_json: str
    inference_ts: int
    serving_latency_ms: float = Field(ge=0.0)
    request_id: str | None = None
    event_id: str | None = None


class LatestPredictionResponse(PredictResponse):
    """Response for GET /latest-prediction."""

    source: str
    source_symbol: str
    prediction_ts: int


def fixed_error(
    error_code: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
    request_id: str | None = None,
) -> ErrorResponse:
    """Build the fixed API error envelope."""

    return ErrorResponse(
        error_code=error_code,
        message=message,
        details={} if details is None else details,
        request_id=request_id,
    )


def api_contract() -> dict[str, Any]:
    """Return a lightweight, framework-neutral API schema contract."""

    return {
        "endpoints": API_ENDPOINTS,
        "schemas": {
            "ErrorResponse": ErrorResponse.model_json_schema(),
            "HealthResponse": HealthResponse.model_json_schema(),
            "ModelInfoResponse": ModelInfoResponse.model_json_schema(),
            "PredictRequest": PredictRequest.model_json_schema(),
            "PredictResponse": PredictResponse.model_json_schema(),
            "LatestPredictionResponse": LatestPredictionResponse.model_json_schema(),
        },
        "error_format": {
            "error_code": "string",
            "message": "string",
            "details": "object",
            "request_id": "string|null",
        },
    }
