"""Phase 1 hard-gate tests for pure policy learning."""

from __future__ import annotations

import math
from copy import deepcopy
from datetime import UTC, datetime

import pytest

from bigan.v8.phase0 import (
    CostModelConfig,
    FeatureProvenance,
    FeatureVector,
    Label,
    Phase0ArtifactError,
    Phase0Pipeline,
    Phase0PipelineConfig,
    ValidationConfig,
)
from bigan.v8.phase1 import (
    PolicyAcceptanceConfig,
    PolicyDatasetConfig,
    PolicyPrediction,
    XGBoostPolicyConfig,
    build_policy_dataset,
    build_policy_dataset_from_phase0,
    policy_dataset_hash,
    train_xgboost_policy,
    validate_policy_acceptance,
)

MINUTE_MS = 60_000


def _ts_at(year: int, month: int, day: int, hour: int, minute: int) -> int:
    return int(datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp() * 1000)


def _market_rows(count: int = 90) -> list[dict]:
    t0 = _ts_at(2026, 6, 1, 12, 0)
    rows = []
    for idx in range(count):
        ts = t0 + idx * MINUTE_MS
        price = 100.0 + 0.22 * math.sin(idx * 0.41) + 0.15 * math.cos(idx * 0.17)
        spread = 0.02 + 0.002 * (idx % 4)
        rows.append(
            {
                "ts": ts,
                "available_at_ts": ts,
                "source": "polymarket",
                "instrument_id": "btc-updown-15m:UP",
                "bid_price": price - spread / 2.0,
                "ask_price": price + spread / 2.0,
                "volume": 10.0 + idx % 7,
                "trade_count": 1 + idx % 3,
                "bid_size": 80.0 + idx % 11,
                "ask_size": 75.0 + (idx * 2) % 13,
                "liquidity_depth": 180.0 + idx % 17,
            }
        )
    return rows


def _pipeline(*, fail_on_validation_error: bool = True) -> Phase0Pipeline:
    return Phase0Pipeline(
        Phase0PipelineConfig(
            cost_config=CostModelConfig(
                fee_bps=5.0,
                base_slippage_bps=1.0,
                volatility_slippage_factor=0.10,
                liquidity_impact_factor=0.002,
            ),
            validation_config=ValidationConfig(
                max_abs_feature_future_corr=0.995,
                min_correlation_rows=25,
            ),
            fail_on_validation_error=fail_on_validation_error,
        )
    )


def _causal_policy_rows(row_count: int = 80) -> tuple[list[FeatureVector], list[Label]]:
    t0 = _ts_at(2026, 6, 1, 12, 0)
    features: list[FeatureVector] = []
    labels: list[Label] = []
    for idx in range(row_count):
        decision_ts = t0 + idx * MINUTE_MS
        signal = -1.0 if idx % 4 in (0, 1) else 1.0
        net_return = 0.020 if signal > 0.0 else -0.015
        feature_values = {
            "signal": signal,
            "volatility_5m": 0.01 + (idx % 5) * 0.001,
            "volatility_15m": 0.015 + (idx % 7) * 0.001,
            "spread_bps": 2.0 + idx % 3,
            "liquidity_depth": 100.0 + idx % 11,
        }
        provenance = {
            name: FeatureProvenance(
                feature_name=name,
                input_start_ts=decision_ts,
                input_end_ts=decision_ts,
                available_at_ts=decision_ts,
                lookback_ms=0,
            )
            for name in feature_values
        }
        features.append(
            FeatureVector(
                decision_ts=decision_ts,
                feature_cutoff_ts=decision_ts,
                lookback_start_ts=decision_ts,
                max_input_ts=decision_ts,
                source="polymarket",
                instrument_id="btc-updown-15m:UP",
                features=feature_values,
                provenance=provenance,
            )
        )
        labels.append(
            Label(
                decision_ts=decision_ts,
                label_ts=decision_ts + MINUTE_MS,
                horizon_ms=MINUTE_MS,
                source="polymarket",
                instrument_id="btc-updown-15m:UP",
                entry_price=100.0,
                exit_price=100.0 * (1.0 + net_return),
                gross_return=net_return,
                spread_cost=0.0,
                fee_cost=0.0,
                slippage_cost=0.0,
                liquidity_impact_cost=0.0,
                total_cost=0.0,
                net_return=net_return,
                is_positive=net_return > 0.0,
            )
        )
    return features, labels


def _causal_policy_dataset(row_count: int = 80):
    features, labels = _causal_policy_rows(row_count)
    return build_policy_dataset(
        features=features,
        labels=labels,
        phase0_dataset_hash="phase0-hash-for-policy-test",
        phase0_dataset_version="bigan-v8-phase0-v1.0.0",
        config=PolicyDatasetConfig(
            horizon_ms=MINUTE_MS,
            feature_columns=(
                "signal",
                "volatility_5m",
                "volatility_15m",
                "spread_bps",
                "liquidity_depth",
            ),
        ),
    )


def test_phase1_dataset_adapter_requires_ready_phase0_and_is_deterministic() -> None:
    phase0_dataset = _pipeline().build(_market_rows())

    first = build_policy_dataset_from_phase0(phase0_dataset)
    second = build_policy_dataset_from_phase0(phase0_dataset)

    assert first.policy_dataset_hash == second.policy_dataset_hash
    assert first.phase0_dataset_hash == phase0_dataset.manifest["dataset_hash"]
    assert first.phase0_dataset_version == phase0_dataset.manifest["dataset_version"]
    assert first.to_dict()["row_count"] == len(first.examples)
    assert "net_return" not in first.feature_columns
    assert "total_cost" not in first.feature_columns
    assert all(
        example.target_action in {0.0, first.config.max_position_size}
        for example in first.examples
    )
    assert first.policy_dataset_hash == policy_dataset_hash(
        examples=first.examples,
        feature_columns=first.feature_columns,
        phase0_dataset_hash=first.phase0_dataset_hash,
        phase0_dataset_version=first.phase0_dataset_version,
        config=first.config,
    )

    failed_manifest = deepcopy(phase0_dataset.manifest)
    failed_manifest["validation"]["acceptance_criteria"]["zero_detectable_leakage"] = False
    unsafe_phase0 = phase0_dataset
    unsafe_phase0.manifest = failed_manifest
    with pytest.raises(Phase0ArtifactError, match="zero_detectable_leakage"):
        build_policy_dataset_from_phase0(unsafe_phase0)


def test_phase1_dataset_rejects_label_or_cost_columns_as_features() -> None:
    features, labels = _causal_policy_rows(row_count=8)
    with pytest.raises(ValueError, match="label/cost columns"):
        build_policy_dataset(
            features=features,
            labels=labels,
            phase0_dataset_hash="phase0-hash-for-policy-test",
            phase0_dataset_version="bigan-v8-phase0-v1.0.0",
            config=PolicyDatasetConfig(feature_columns=("net_return",)),
        )
    with pytest.raises(ValueError, match="no usable values"):
        build_policy_dataset(
            features=features,
            labels=labels,
            phase0_dataset_hash="phase0-hash-for-policy-test",
            phase0_dataset_version="bigan-v8-phase0-v1.0.0",
            config=PolicyDatasetConfig(feature_columns=("missing_signal",)),
        )


def test_xgboost_policy_rejects_direct_pnl_optimization_knobs() -> None:
    with pytest.raises(ValueError, match="PnL/profit"):
        XGBoostPolicyConfig(selection_metric="shadow_sharpe")
    with pytest.raises(ValueError, match="PnL/profit"):
        XGBoostPolicyConfig(eval_metric="profit_factor")
    with pytest.raises(ValueError):
        XGBoostPolicyConfig(objective="pnl:max")  # type: ignore[arg-type]


def test_xgboost_policy_trains_without_pnl_objective_and_outputs_policy_fields() -> None:
    dataset = _causal_policy_dataset()
    config = XGBoostPolicyConfig(
        num_boost_round=8,
        max_depth=2,
        learning_rate=0.2,
        action_activation_threshold=0.40,
    )

    model = train_xgboost_policy(dataset, config)
    predictions = model.predict_dataset(dataset)

    assert model.training_manifest["direct_pnl_optimization"] is False
    assert model.training_manifest["pnl_usage"] == "shadow_acceptance_after_inference_only"
    assert model.training_manifest["policy_dataset_hash"] == dataset.policy_dataset_hash
    assert len(predictions) == len(dataset.examples)
    assert all(0.0 <= prediction.action <= config.max_position_size for prediction in predictions)
    assert all(0.0 <= prediction.confidence <= 1.0 for prediction in predictions)
    assert all(
        len(prediction.regime_embedding) == len(config.regime_feature_names)
        for prediction in predictions
    )
    assert max(prediction.action for prediction in predictions) > min(
        prediction.action for prediction in predictions
    )


def test_xgboost_policy_supports_pairwise_ranking_loss_without_pnl_magnitude() -> None:
    dataset = _causal_policy_dataset()
    config = XGBoostPolicyConfig(
        objective="rank:pairwise",
        eval_metric="ndcg",
        selection_metric="ndcg",
        num_boost_round=4,
        max_depth=2,
    )

    model = train_xgboost_policy(dataset, config)
    predictions = model.predict_dataset(dataset)

    assert model.training_manifest["objective_type"] == "pairwise_ranking_policy"
    assert (
        model.training_manifest["target_encoding"]
        == "dense_relevance_rank_from_cost_aware_net_return"
    )
    assert model.training_manifest["direct_pnl_optimization"] is False
    assert len(predictions) == len(dataset.examples)
    assert all(0.0 <= prediction.action <= config.max_position_size for prediction in predictions)


def test_policy_acceptance_validates_shadow_sharpe_distribution_and_buckets() -> None:
    dataset = _causal_policy_dataset(row_count=12)
    actions = (0.0, 0.0, 0.50, 0.50, 1.0, 1.0, 0.0, 0.50, 1.0, 0.0, 0.50, 1.0)
    examples = tuple(
        example.__class__(
            decision_ts=example.decision_ts,
            source=example.source,
            instrument_id=example.instrument_id,
            features=example.features,
            target_action=example.target_action,
            target_score=example.target_score,
            net_return=(
                -0.010
                if action == 0.0
                else 0.015
                if action == 0.50
                else 0.030
            ),
            horizon_ms=example.horizon_ms,
            regime_key=example.regime_key,
        )
        for example, action in zip(dataset.examples, actions, strict=True)
    )
    predictions = tuple(
        PolicyPrediction(
            decision_ts=example.decision_ts,
            source=example.source,
            instrument_id=example.instrument_id,
            action=action,
            confidence=0.75,
            regime_embedding=(0.01, 0.02, 2.0, 100.0),
            score=action,
        )
        for example, action in zip(examples, actions, strict=True)
    )

    report = validate_policy_acceptance(
        examples,
        predictions,
        PolicyAcceptanceConfig(max_dominant_bucket_ratio=0.50),
    )

    assert report.passed
    assert report.acceptance_criteria["shadow_sharpe_positive"]
    assert report.acceptance_criteria["stable_action_distribution"]
    assert report.acceptance_criteria["monotonic_pnl_bucket_behavior"]
    assert report.acceptance_criteria["no_direct_pnl_optimization"]
    assert report.metrics["shadow_sharpe"] > 0.0

    collapsed_predictions = tuple(
        PolicyPrediction(
            decision_ts=example.decision_ts,
            source=example.source,
            instrument_id=example.instrument_id,
            action=0.0,
            confidence=0.5,
            regime_embedding=(0.0, 0.0, 0.0, 0.0),
            score=0.0,
        )
        for example in examples
    )
    collapsed_report = validate_policy_acceptance(examples, collapsed_predictions)

    assert not collapsed_report.passed
    assert collapsed_report.acceptance_criteria["stable_action_distribution"] is False
    assert collapsed_report.acceptance_criteria["monotonic_pnl_bucket_behavior"] is False
    assert "unstable_action_distribution" in {
        failure.code for failure in collapsed_report.failures
    }
