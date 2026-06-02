"""Serving API contract tests for issue #43."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bigan.serving import (
    API_ENDPOINTS,
    ErrorResponse,
    HealthResponse,
    LatestPredictionResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    api_contract,
    clip_probability,
    fixed_error,
    prediction_event_from_contract,
)


def test_endpoint_contract_contains_required_routes_and_methods() -> None:
    assert API_ENDPOINTS == {
        "health": {"method": "GET", "path": "/health"},
        "model_info": {"method": "GET", "path": "/model-info"},
        "predict": {"method": "POST", "path": "/predict"},
        "latest_prediction": {"method": "GET", "path": "/latest-prediction"},
    }
    contract = api_contract()
    assert set(contract["schemas"]) == {
        "ErrorResponse",
        "HealthResponse",
        "ModelInfoResponse",
        "PredictRequest",
        "PredictResponse",
        "LatestPredictionResponse",
    }


def test_health_and_model_info_contract_support_probe_and_introspection() -> None:
    health = HealthResponse(status="ok", model_version="xgb-v1", checks={"model_loaded": True})
    assert health.status == "ok"
    assert health.checks["model_loaded"] is True

    info = ModelInfoResponse(
        model_version="xgb-v1",
        model_family="btc-updown-15m",
        feature_version="bigan-mvp-v1.0.0",
        dataset_version="bigan-training-15m-v1.0.0",
        calibration_method="platt",
        status="champion",
        artifact_uri="models/btc-updown-15m/xgb-v1/model.json",
        loaded_at=1_000,
    )
    assert info.model_version == "xgb-v1"
    assert info.calibration_method == "platt"


def test_predict_request_and_response_fields_are_fixed() -> None:
    request = PredictRequest(
        source_symbol="token-1",
        feature_version="bigan-mvp-v1.0.0",
        features={"ret_15m": 0.01, "spread": 0.02},
        request_id="req-1",
    )
    assert request.source == "polymarket"

    response = PredictResponse(
        prob_up_15m=0.62,
        model_version="xgb-v1",
        feature_version=request.feature_version,
        confidence_bucket="medium_up",
        top_features_json="[]",
        inference_ts=1_000,
        serving_latency_ms=4.2,
        request_id="req-1",
    )
    assert response.prob_up_15m == pytest.approx(0.62)

    with pytest.raises(ValidationError):
        PredictResponse(
            prob_up_15m=1.2,
            model_version="xgb-v1",
            feature_version=request.feature_version,
            confidence_bucket="bad",
            top_features_json="[]",
            inference_ts=1_000,
            serving_latency_ms=4.2,
        )


def test_predict_response_clips_served_probability_hotfix() -> None:
    response = PredictResponse(
        prob_up_15m=0.997,
        model_version="xgboost-v4",
        feature_version="bigan-mvp-v1.0.0",
        confidence_bucket="high_up",
        top_features_json="[]",
        inference_ts=1_000,
        serving_latency_ms=4.2,
    )

    assert response.prob_up_15m == pytest.approx(0.95)
    assert clip_probability(0.002, lower=0.10, upper=0.90) == pytest.approx(0.10)
    assert api_contract()["probability_postprocessing"] == {
        "prob_up_15m_clip_lower": 0.05,
        "prob_up_15m_clip_upper": 0.95,
        "scope": "serving_contract_only",
    }


def test_predict_response_requires_explicit_v6_three_way_and_volatility_payload() -> None:
    response = PredictResponse(
        prob_up_15m=0.55,
        p_up=0.55,
        p_down=0.30,
        p_neutral=0.15,
        p_vol_up=0.64,
        p_vol_down=0.41,
        model_version="xgboost-v6",
        feature_version="bigan-mvp-v1.0.0",
        confidence_bucket="medium_up",
        top_features_json="[]",
        inference_ts=1_000,
        serving_latency_ms=4.2,
    )

    assert response.p_down == pytest.approx(0.30)
    assert response.p_down != pytest.approx(1.0 - response.p_up)
    assert api_contract()["v6_prediction_payload"] == {
        "settlement_probabilities": ["p_up", "p_down", "p_neutral"],
        "volatility_probabilities": ["p_vol_up", "p_vol_down"],
        "legacy_alias": "prob_up_15m is clipped p_up only",
        "down_probability_rule": "read explicit p_down; never derive from 1 - p_up",
    }

    with pytest.raises(ValidationError, match="p_up, p_down, and p_neutral"):
        PredictResponse(
            prob_up_15m=0.55,
            p_up=0.55,
            p_neutral=0.45,
            p_vol_up=0.64,
            p_vol_down=0.41,
            model_version="xgboost-v6",
            feature_version="bigan-mvp-v1.0.0",
            confidence_bucket="bad",
            top_features_json="[]",
            inference_ts=1_000,
            serving_latency_ms=4.2,
        )

    with pytest.raises(ValidationError, match="sum to 1"):
        PredictResponse(
            prob_up_15m=0.55,
            p_up=0.55,
            p_down=0.30,
            p_neutral=0.30,
            p_vol_up=0.64,
            p_vol_down=0.41,
            model_version="xgboost-v6",
            feature_version="bigan-mvp-v1.0.0",
            confidence_bucket="bad",
            top_features_json="[]",
            inference_ts=1_000,
            serving_latency_ms=4.2,
        )


def test_latest_prediction_extends_predict_response_with_identity() -> None:
    latest = LatestPredictionResponse(
        prob_up_15m=0.51,
        model_version="xgb-v1",
        feature_version="bigan-mvp-v1.0.0",
        confidence_bucket="neutral",
        top_features_json="[]",
        inference_ts=1_000,
        serving_latency_ms=1.0,
        source="polymarket",
        source_symbol="token-1",
        prediction_ts=1_000,
    )
    assert latest.source_symbol == "token-1"


def test_predict_contract_can_emit_monitoring_event() -> None:
    request = PredictRequest(
        source_symbol="token-1",
        feature_version="bigan-mvp-v1.0.0",
        features={"market_implied_prob": 0.40, "ret_15m": 0.01},
        request_id="req-1",
    )
    response = PredictResponse(
        prob_up_15m=0.72,
        model_version="xgboost-v3",
        feature_version=request.feature_version,
        confidence_bucket="high_up",
        top_features_json="[]",
        inference_ts=1_000,
        serving_latency_ms=4.2,
        request_id=request.request_id,
    )

    event = prediction_event_from_contract(request, response)

    assert event.event_id.startswith("pred-")
    assert event.model_version == "xgboost-v3"
    assert event.prob_up_15m == pytest.approx(0.72)
    assert event.serving_latency_ms == pytest.approx(4.2)
    assert '"market_implied_prob": 0.4' in event.feature_snapshot_json


def test_error_response_format_is_stable_and_strict() -> None:
    err = fixed_error(
        "schema_mismatch",
        "Input feature schema does not match training schema.",
        details={"missing": ["ret_15m"]},
        request_id="req-1",
    )
    assert isinstance(err, ErrorResponse)
    assert err.model_dump() == {
        "error_code": "schema_mismatch",
        "message": "Input feature schema does not match training schema.",
        "details": {"missing": ["ret_15m"]},
        "request_id": "req-1",
    }

    with pytest.raises(ValidationError):
        ErrorResponse(error_code="bad", message="bad", surprise=True)
