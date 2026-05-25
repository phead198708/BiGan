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
    clip_probability,
    fixed_error,
    prediction_event_from_contract,
)
from .schema_validation import (
    FeatureSchemaArtifact,
    FeatureSchemaMismatch,
    build_feature_schema_artifact,
    load_feature_schema_artifact,
    validate_features_fail_closed,
    write_feature_schema_artifact,
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
    "FeatureSchemaArtifact",
    "FeatureSchemaMismatch",
    "api_contract",
    "build_feature_schema_artifact",
    "clip_probability",
    "fixed_error",
    "load_feature_schema_artifact",
    "prediction_event_from_contract",
    "validate_features_fail_closed",
    "write_feature_schema_artifact",
]
