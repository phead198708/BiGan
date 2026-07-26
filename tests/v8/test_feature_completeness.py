from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from bigan.v8.polymarket.corpus.contracts import (
    PolymarketCorpusMarket,
    PolymarketCorpusTrade,
)
from bigan.v8.polymarket.feature_completeness import (
    FeatureCompletenessError,
    TradeTapeCoverageStatus,
    build_provider_health_diagnostics,
    build_trade_volume_feature_bundle,
    derive_trade_tape_coverage_status,
    validate_feature_missingness_contract,
    validate_feature_missingness_runtime_schema,
    validate_trade_volume_feature_bundle,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "examples/v8/polymarket_configs/feature_missingness_contract.json"
CONTRACT_SHA = CONTRACT.with_suffix(".sha256")
RUNTIME_SCHEMA = (
    ROOT / "examples/v8/polymarket_configs/feature_missingness_runtime.schema.json"
)
RUNTIME_SCHEMA_SHA = RUNTIME_SCHEMA.with_suffix(".sha256")


def _status(**overrides) -> TradeTapeCoverageStatus:
    values = {
        "market_id": "market-1",
        "decision_ts": 200_000,
        "provider_source": "polymarket_data_api",
        "collection_mode": "rest",
        "collection_started_ts": 199_900,
        "collection_completed_ts": 199_950,
        "observation_window_start_ts": 140_000,
        "observation_window_end_ts": 200_000,
        "max_causal_input_ts": 199_000,
        "available_at_ts": 199_950,
        "observed_trade_count": 0,
        "provider_timeout": False,
        "truncated": False,
        "censored": False,
        "coverage_complete": True,
        "missingness_reason": None,
        "provider_health_score": 1.0,
        "historical_backfill": False,
    }
    values.update(overrides)
    return TradeTapeCoverageStatus(**values)


def _trade(outcome: str, size: float, ts: int = 180_000) -> PolymarketCorpusTrade:
    return PolymarketCorpusTrade(
        market_id="market-1",
        token_id=f"token-{outcome}",
        outcome=outcome,
        ts=ts,
        available_at_ts=ts,
        price=0.5,
        size=size,
        side="BUY",
    )


def _market(**overrides) -> PolymarketCorpusMarket:
    values = {
        "market_id": "market-1",
        "condition_id": "condition-1",
        "slug": "btc-updown-5m-140",
        "market_family": "btc_updown_5m",
        "horizon_ms": 300_000,
        "market_start_ts": 140_000,
        "market_end_ts": 440_000,
        "settlement_ts": 440_000,
        "up_token_id": "token-UP",
        "down_token_id": "token-DOWN",
        "reference_price_source": "chainlink",
        "settlement_rule": "UP if close >= open",
        "raw_market_sha256": "a" * 64,
        "trade_collection_mode": (
            "clob_websocket_plus_data_api_paginated_reconciliation"
        ),
        "trade_stream_started_at_ts": 140_000,
        "trade_stream_ended_at_ts": 440_000,
        "trade_stream_continuity_passed": True,
        "trade_stream_timestamp_causality_violation_count": 0,
        "trade_api_collection_ts": 440_000,
        "trade_api_request_failed": False,
        "trade_rest_rows_truncated": False,
        "trade_full_round_coverage_complete": True,
        "trade_tape_censored": False,
        "trade_collection_reason_codes": (),
    }
    values.update(overrides)
    return PolymarketCorpusMarket(**values)


def test_contract_and_runtime_schema_are_hash_pinned_and_closed() -> None:
    for path, sha_path in (
        (CONTRACT, CONTRACT_SHA),
        (RUNTIME_SCHEMA, RUNTIME_SCHEMA_SHA),
    ):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha_path.read_text().strip()
    contract = json.loads(CONTRACT.read_text())
    assert all(value is False for value in contract["safety"].values())
    validate_feature_missingness_contract(contract)
    validate_feature_missingness_runtime_schema(
        json.loads(RUNTIME_SCHEMA.read_text())
    )


def test_semantic_governance_validation_rejects_repinnable_weakening() -> None:
    contract = json.loads(CONTRACT.read_text())
    weakened_contract = copy.deepcopy(contract)
    weakened_contract["rules"]["missing_encoded_as_zero"] = True
    with pytest.raises(FeatureCompletenessError, match="rules"):
        validate_feature_missingness_contract(weakened_contract)

    schema = json.loads(RUNTIME_SCHEMA.read_text())
    weakened_schema = copy.deepcopy(schema)
    weakened_schema["properties"]["provider_health_score"]["maximum"] = 2
    with pytest.raises(FeatureCompletenessError, match="properties"):
        validate_feature_missingness_runtime_schema(weakened_schema)


def test_runtime_schema_validates_complete_and_missing_feature_bundles() -> None:
    schema = json.loads(RUNTIME_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    complete, _ = build_trade_volume_feature_bundle(
        trades=(),
        status=_status(),
    )
    missing, _ = build_trade_volume_feature_bundle(
        trades=(),
        status=_status(
            collection_mode="timeout",
            provider_timeout=True,
            coverage_complete=False,
            missingness_reason="provider_timeout",
            provider_health_score=0.0,
        ),
    )

    validator.validate(complete)
    validator.validate(missing)


def test_true_zero_requires_complete_coverage() -> None:
    features, _ = build_trade_volume_feature_bundle(trades=(), status=_status())
    assert features["recent_up_trade_volume"] == 0
    assert features["recent_down_trade_volume"] == 0
    assert features["recent_up_trade_volume_missing"] == 0
    assert features["recent_up_trade_volume_coverage_complete"] == 1


def test_complete_tape_sums_only_causally_available_window_trades() -> None:
    features, provenance = build_trade_volume_feature_bundle(
        trades=(
            _trade("UP", 2.5),
            _trade("DOWN", 1.25),
            _trade("UP", 99, ts=139_999),
            _trade("DOWN", 99, ts=200_001),
        ),
        status=_status(observed_trade_count=2),
    )
    assert features["recent_up_trade_volume"] == 2.5
    assert features["recent_down_trade_volume"] == 1.25
    assert provenance["recent_up_trade_volume"]["available_at_ts"] <= 200_000


def test_captured_trade_tape_proof_derives_complete_decision_metadata() -> None:
    status = derive_trade_tape_coverage_status(
        market=_market(),
        trades=(_trade("UP", 2.5),),
        decision_ts=200_000,
    )

    assert status is not None
    assert status.collection_mode == "websocket"
    assert status.coverage_complete is True
    assert status.available_at_ts <= status.decision_ts
    features, _ = build_trade_volume_feature_bundle(
        trades=(_trade("UP", 2.5),),
        status=status,
    )
    assert features["recent_up_trade_volume"] == 2.5
    assert features["recent_down_trade_volume"] == 0


def test_captured_backfill_or_censored_tape_derives_missing_not_zero() -> None:
    status = derive_trade_tape_coverage_status(
        market=_market(
            trade_stream_started_at_ts=None,
            trade_stream_ended_at_ts=None,
            trade_stream_continuity_passed=False,
            trade_rest_rows_truncated=True,
            trade_full_round_coverage_complete=False,
            trade_tape_censored=True,
            trade_collection_reason_codes=(
                "trade_data_api_pagination_truncated",
            ),
        ),
        trades=(_trade("UP", 2.5),),
        decision_ts=200_000,
    )

    assert status is not None
    assert status.collection_mode == "backfill"
    assert status.coverage_complete is False
    features, _ = build_trade_volume_feature_bundle(
        trades=(_trade("UP", 2.5),),
        status=status,
    )
    assert features["recent_up_trade_volume"] is None
    assert features["recent_up_trade_volume_missing"] == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "collection_mode": "timeout",
            "provider_timeout": True,
            "coverage_complete": False,
            "missingness_reason": "provider_timeout",
            "provider_health_score": 0.0,
        },
        {
            "truncated": True,
            "coverage_complete": False,
            "missingness_reason": "provider_limit_truncated",
            "provider_health_score": 0.4,
        },
        {
            "censored": True,
            "coverage_complete": False,
            "missingness_reason": "websocket_gap_censored",
            "provider_health_score": 0.3,
            "collection_mode": "websocket",
        },
        {
            "collection_mode": "backfill",
            "historical_backfill": True,
            "coverage_complete": False,
            "missingness_reason": "historical_backfill_not_causal",
            "provider_health_score": 0.0,
        },
    ],
)
def test_timeout_truncation_censoring_and_backfill_are_null_not_zero(overrides) -> None:
    features, _ = build_trade_volume_feature_bundle(
        trades=(_trade("UP", 4.0),),
        status=_status(**overrides),
    )
    assert features["recent_up_trade_volume"] is None
    assert features["recent_down_trade_volume"] is None
    assert features["recent_up_trade_volume_missing"] == 1
    assert features["recent_up_trade_volume_coverage_complete"] == 0


def test_incomplete_observation_window_is_missing() -> None:
    features, _ = build_trade_volume_feature_bundle(
        trades=(),
        status=_status(observation_window_start_ts=150_000),
    )
    assert features["recent_up_trade_volume"] is None
    assert features["trade_tape_missingness_reason"] == "observation_window_incomplete"


def test_provider_metadata_after_decision_fails_closed() -> None:
    with pytest.raises(FeatureCompletenessError, match="decision_ts"):
        _status(available_at_ts=200_001)


def test_future_candidate_rejects_missing_or_incomplete_metadata() -> None:
    with pytest.raises(FeatureCompletenessError, match="metadata missing"):
        validate_trade_volume_feature_bundle(
            {"recent_up_trade_volume": 0},
            decision_ts=200_000,
            require_complete=True,
        )
    features, _ = build_trade_volume_feature_bundle(
        trades=(),
        status=_status(
            truncated=True,
            coverage_complete=False,
            missingness_reason="truncated",
        ),
    )
    with pytest.raises(FeatureCompletenessError, match="requires complete"):
        validate_trade_volume_feature_bundle(
            features,
            decision_ts=200_000,
            require_complete=True,
        )


def test_provider_health_diagnostics_attribute_primary_fallback_and_no_trade() -> None:
    healthy, _ = build_trade_volume_feature_bundle(trades=(), status=_status())
    timeout, _ = build_trade_volume_feature_bundle(
        trades=(),
        status=_status(
            market_id="market-2",
            collection_mode="timeout",
            provider_timeout=True,
            coverage_complete=False,
            missingness_reason="timeout",
            provider_health_score=0,
        ),
    )
    diagnostics = build_provider_health_diagnostics(
        feature_rows=[
            {"market_id": "market-1", "decision_ts": 200_000, "features": healthy},
            {"market_id": "market-2", "decision_ts": 200_000, "features": timeout},
        ],
        decision_rows=[
            {
                "market_id": "market-1",
                "decision_ts": 200_000,
                "executed_action": "BUY_UP_HOLD_TO_SETTLEMENT",
                "selected_side": "UP",
                "decision_origin": "primary",
                "execution_guard_order_allowed": True,
            },
            {
                "market_id": "market-2",
                "decision_ts": 200_000,
                "executed_action": "NO_TRADE",
                "selected_side": "NONE",
                "decision_origin": "fallback",
                "execution_guard_order_allowed": False,
            },
        ],
    )
    assert diagnostics["fallback_provider_health_association_report"]["healthy"]["primary"] == 1
    assert diagnostics["fallback_provider_health_association_report"]["timeout"]["no_trade"] == 1
    assert diagnostics["fallback_provider_health_association_report"]["timeout"][
        "execution_guard_rejected"
    ] == 1
    assert diagnostics["outcomes_settlement_pnl_or_future_information_used"] is False
