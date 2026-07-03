"""Online inference API contract models (issue #43)."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bigan.monitoring import PredictionEvent

API_ENDPOINTS: dict[str, dict[str, str]] = {
    "health": {"method": "GET", "path": "/health"},
    "model_info": {"method": "GET", "path": "/model-info"},
    "predict": {"method": "POST", "path": "/predict"},
    "latest_prediction": {"method": "GET", "path": "/latest-prediction"},
}

ServingStatus = Literal["ok", "degraded", "unhealthy"]
DEFAULT_PROBABILITY_CLIP_LOWER = 0.05
DEFAULT_PROBABILITY_CLIP_UPPER = 0.95


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
    p_up: float | None = Field(default=None, ge=0.0, le=1.0)
    p_down: float | None = Field(default=None, ge=0.0, le=1.0)
    p_neutral: float | None = Field(default=None, ge=0.0, le=1.0)
    p_vol_up: float | None = Field(default=None, ge=0.0, le=1.0)
    p_vol_down: float | None = Field(default=None, ge=0.0, le=1.0)
    model_version: str
    feature_version: str
    confidence_bucket: str
    top_features_json: str
    inference_ts: int
    serving_latency_ms: float = Field(ge=0.0)
    request_id: str | None = None
    event_id: str | None = None

    @field_validator("prob_up_15m", mode="after")
    @classmethod
    def _clip_served_probability(cls, value: float) -> float:
        return clip_probability(value)

    @model_validator(mode="after")
    def _v6_payload_is_explicit(self) -> Self:
        settlement_values = (self.p_up, self.p_down, self.p_neutral)
        volatility_values = (self.p_vol_up, self.p_vol_down)
        has_v6_fields = any(value is not None for value in (*settlement_values, *volatility_values))
        is_v6 = self.model_version == "xgboost-v6" or self.model_version.startswith("xgboost-v6:")
        if not is_v6 and not has_v6_fields:
            return self
        if any(value is None for value in settlement_values):
            raise ValueError("v6 prediction payload must include p_up, p_down, and p_neutral")
        if any(value is None for value in volatility_values):
            raise ValueError("v6 prediction payload must include p_vol_up and p_vol_down")
        assert self.p_up is not None
        assert self.p_down is not None
        assert self.p_neutral is not None
        probability_sum = self.p_up + self.p_down + self.p_neutral
        if abs(probability_sum - 1.0) > 1e-6:
            raise ValueError("v6 settlement probabilities must sum to 1")
        if abs(self.prob_up_15m - clip_probability(self.p_up)) > 1e-9:
            raise ValueError("prob_up_15m must be the clipped legacy alias for explicit p_up")
        return self


class LatestPredictionResponse(PredictResponse):
    """Response for GET /latest-prediction."""

    source: str
    source_symbol: str
    prediction_ts: int


def prediction_event_from_contract(
    request: PredictRequest,
    response: PredictResponse,
) -> PredictionEvent:
    """Build a monitoring event from a serving predict request/response pair."""

    feature_snapshot_json = json.dumps(
        {
            "source": request.source,
            "source_symbol": request.source_symbol,
            "request_id": request.request_id,
            "market_implied_prob": request.features.get("market_implied_prob"),
            "features": request.features,
        },
        sort_keys=True,
    )
    event_id = response.event_id or _contract_event_id(request, response)
    return PredictionEvent(
        event_id=event_id,
        ts=response.inference_ts,
        model_version=response.model_version,
        feature_version=response.feature_version,
        prob_up_15m=response.prob_up_15m,
        confidence_bucket=response.confidence_bucket,
        top_features_json=response.top_features_json,
        feature_hash=_stable_hash(feature_snapshot_json),
        feature_snapshot_json=feature_snapshot_json,
        serving_latency_ms=response.serving_latency_ms,
    )


def clip_probability(
    probability: float,
    *,
    lower: float = DEFAULT_PROBABILITY_CLIP_LOWER,
    upper: float = DEFAULT_PROBABILITY_CLIP_UPPER,
) -> float:
    """Clip served probabilities to the configured hotfix interval."""

    value = float(probability)
    if lower < 0.0 or upper > 1.0 or lower >= upper:
        raise ValueError("probability clip bounds must satisfy 0 <= lower < upper <= 1")
    return min(upper, max(lower, value))


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
        "probability_postprocessing": {
            "prob_up_15m_clip_lower": DEFAULT_PROBABILITY_CLIP_LOWER,
            "prob_up_15m_clip_upper": DEFAULT_PROBABILITY_CLIP_UPPER,
            "scope": "serving_contract_only",
        },
        "v6_prediction_payload": {
            "settlement_probabilities": ["p_up", "p_down", "p_neutral"],
            "volatility_probabilities": ["p_vol_up", "p_vol_down"],
            "legacy_alias": "prob_up_15m is clipped p_up only",
            "down_probability_rule": "read explicit p_down; never derive from 1 - p_up",
        },
    }


def _contract_event_id(request: PredictRequest, response: PredictResponse) -> str:
    raw = (
        f"{response.model_version}:{request.source}:"
        f"{request.source_symbol}:{response.inference_ts}:{request.request_id or ''}"
    )
    return f"pred-{_stable_hash(raw)[:24]}"


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
