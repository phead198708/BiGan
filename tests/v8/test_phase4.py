"""Phase 4 online adaptive-system tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.v8.phase0.costs import CostModelConfig
from bigan.v8.phase1 import PolicyPrediction, PolicyTrainingExample
from bigan.v8.phase4 import (
    PHASE4_ADAPTIVE_SYSTEM_PHASE,
    ExecutionAdaptationConfig,
    LambdaControllerConfig,
    Phase4AdaptiveError,
    Phase4AdaptiveSystemConfig,
    RegimeDetectorConfig,
    run_phase4_adaptive_system,
)


def _example(
    index: int,
    *,
    signal: float,
    volatility: float,
    liquidity_depth: float,
    spread_bps: float,
    shadow_net_return: float,
) -> PolicyTrainingExample:
    return PolicyTrainingExample(
        decision_ts=1_800_000 + index * 60_000,
        source="polymarket",
        instrument_id="btc-up",
        features={
            "mid_price": 100.0,
            "signal": signal,
            "return_5m": signal * 0.01,
            "volatility_5m": volatility,
            "volatility_15m": volatility * 0.8,
            "spread_bps": spread_bps,
            "liquidity_depth": liquidity_depth,
            "volume_1m": 100.0,
            "trade_count_1m": 10,
        },
        target_label=1.0 if shadow_net_return > 0.0 else 0.0,
        shadow_net_return=shadow_net_return,
        horizon_ms=300_000,
        regime_key="synthetic",
    )


def _prediction(example: PolicyTrainingExample, *, action: float = 0.82) -> PolicyPrediction:
    return PolicyPrediction(
        decision_ts=example.decision_ts,
        source=example.source,
        instrument_id=example.instrument_id,
        action=action,
        confidence=0.86,
        regime_embedding=(0.0, 0.0, 0.0, 0.0),
        score=0.80,
    )


def _adaptive_examples() -> tuple[PolicyTrainingExample, ...]:
    rows: list[PolicyTrainingExample] = []
    for _ in range(24):
        rows.append(
            _example(
                len(rows),
                signal=0.88,
                volatility=0.012,
                liquidity_depth=500.0,
                spread_bps=5.0,
                shadow_net_return=0.012,
            )
        )
    for _ in range(24):
        rows.append(
            _example(
                len(rows),
                signal=0.10,
                volatility=0.12,
                liquidity_depth=500.0,
                spread_bps=18.0,
                shadow_net_return=-0.035,
            )
        )
    for _ in range(12):
        rows.append(
            _example(
                len(rows),
                signal=0.20,
                volatility=0.020,
                liquidity_depth=4.0,
                spread_bps=90.0,
                shadow_net_return=-0.018,
            )
        )
    for _ in range(24):
        rows.append(
            _example(
                len(rows),
                signal=0.92,
                volatility=0.010,
                liquidity_depth=600.0,
                spread_bps=5.0,
                shadow_net_return=0.013,
            )
        )
    return tuple(rows)


def _adaptive_predictions(
    examples: tuple[PolicyTrainingExample, ...],
) -> tuple[PolicyPrediction, ...]:
    return tuple(_prediction(example) for example in examples)


def _cost_config() -> CostModelConfig:
    return CostModelConfig(
        fee_bps=0.1,
        base_slippage_bps=0.1,
        volatility_slippage_factor=0.0,
        liquidity_impact_factor=0.0,
    )


def _phase4_config(output_dir: Path | None = None) -> Phase4AdaptiveSystemConfig:
    return Phase4AdaptiveSystemConfig(
        detector_config=RegimeDetectorConfig(
            trend_score_threshold=0.50,
            high_volatility_threshold=0.060,
            liquidity_stress_threshold=15.0,
            high_cost_spread_bps_threshold=60.0,
            transition_confirmation_count=2,
        ),
        lambda_config=LambdaControllerConfig(
            base_lambda=0.30,
            min_lambda=0.02,
            max_lambda=0.90,
            trend_multiplier=1.55,
            range_multiplier=1.00,
            high_volatility_multiplier=0.30,
            liquidity_stress_multiplier=0.45,
            volatility_sensitivity=0.50,
            drawdown_sensitivity=2.0,
            smoothing_alpha=0.80,
            max_step_change=0.12,
        ),
        execution_config=ExecutionAdaptationConfig(
            cost_model_config=_cost_config(),
            trend_multiplier=1.15,
            range_multiplier=0.90,
            high_volatility_multiplier=0.30,
            liquidity_stress_multiplier=0.35,
            cost_sensitivity=0.30,
            smoothing_alpha=0.70,
            max_step_change=0.15,
            risk_penalty_factor=0.0,
        ),
        min_regime_stability_ratio=0.90,
        max_accepted_lambda_step=0.13,
        max_accepted_aggressiveness_step=0.16,
        min_tail_loss_reduction_ratio=0.10,
        stress_drawdown_shock=0.02,
        output_dir=output_dir,
        created_at="2026-06-21T04:00:00Z",
    )


def test_phase4_adapts_lambda_execution_and_writes_report(tmp_path: Path) -> None:
    examples = _adaptive_examples()
    predictions = _adaptive_predictions(examples)

    result = run_phase4_adaptive_system(
        examples=examples,
        predictions=predictions,
        config=_phase4_config(output_dir=tmp_path),
    )

    assert result.passed
    assert result.report.phase == PHASE4_ADAPTIVE_SYSTEM_PHASE
    assert result.report.row_count == len(examples)
    assert result.report.acceptance_criteria["regime_transitions_stable"] is True
    assert result.report.acceptance_criteria["lambda_stability_under_stress"] is True
    assert result.report.acceptance_criteria["tail_risk_performance_improved"] is True
    assert set(result.report.stress_metrics) == {"1.2", "1.5", "2"}
    regime_lambdas = result.report.adaptive_metrics["mean_lambda_by_regime"]
    assert regime_lambdas["trend"] > regime_lambdas["high_volatility"]
    regime_actions = result.report.adaptive_metrics["mean_filled_action_by_regime"]
    assert regime_actions["trend"] > regime_actions["high_volatility"]
    assert result.report.comparison_metrics["tail_loss_reduction_ratio"] > 0.10
    assert result.report_path is not None
    saved = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert saved["phase"] == PHASE4_ADAPTIVE_SYSTEM_PHASE
    assert saved["passed"] is True
    assert saved["acceptance_criteria"] == result.report.acceptance_criteria


def test_phase4_rejects_prediction_key_mismatch_before_adaptation() -> None:
    examples = _adaptive_examples()
    predictions = list(_adaptive_predictions(examples))
    first = predictions[0]
    predictions[0] = PolicyPrediction(
        decision_ts=first.decision_ts,
        source="wrong-source",
        instrument_id=first.instrument_id,
        action=first.action,
        confidence=first.confidence,
        regime_embedding=first.regime_embedding,
        score=first.score,
    )

    with pytest.raises(Phase4AdaptiveError, match="prediction keys"):
        run_phase4_adaptive_system(
            examples=examples,
            predictions=tuple(predictions),
            config=_phase4_config(),
        )


def test_phase4_rejects_non_causal_timestamp_order() -> None:
    examples = tuple(reversed(_adaptive_examples()))
    predictions = _adaptive_predictions(examples)

    with pytest.raises(Phase4AdaptiveError, match="non-decreasing decision_ts"):
        run_phase4_adaptive_system(
            examples=examples,
            predictions=predictions,
            config=_phase4_config(),
        )


def test_phase4_tail_risk_gate_fails_when_overlay_cannot_reduce_exposure() -> None:
    examples = _adaptive_examples()
    predictions = _adaptive_predictions(examples)
    neutral_config = Phase4AdaptiveSystemConfig(
        detector_config=RegimeDetectorConfig(
            trend_score_threshold=0.50,
            high_volatility_threshold=0.060,
            liquidity_stress_threshold=0.0,
            high_cost_spread_bps_threshold=60.0,
            transition_confirmation_count=2,
        ),
        lambda_config=LambdaControllerConfig(
            base_lambda=0.30,
            min_lambda=0.299999,
            max_lambda=0.300001,
            trend_multiplier=1.0,
            range_multiplier=1.0,
            high_volatility_multiplier=1.0,
            liquidity_stress_multiplier=1.0,
            volatility_sensitivity=0.0,
            drawdown_sensitivity=0.0,
            smoothing_alpha=1.0,
            max_step_change=1.0,
        ),
        execution_config=ExecutionAdaptationConfig(
            cost_model_config=_cost_config(),
            trend_multiplier=1.0,
            range_multiplier=1.0,
            high_volatility_multiplier=1.0,
            liquidity_stress_multiplier=1.0,
            cost_sensitivity=0.0,
            liquidity_sensitivity=0.0,
            smoothing_alpha=1.0,
            max_step_change=1.0,
            risk_penalty_factor=0.0,
        ),
        min_tail_loss_reduction_ratio=0.10,
        created_at="2026-06-21T04:01:00Z",
    )

    result = run_phase4_adaptive_system(
        examples=examples,
        predictions=predictions,
        config=neutral_config,
    )

    assert not result.passed
    assert result.report.acceptance_criteria["tail_risk_performance_improved"] is False
    assert result.report.comparison_metrics["tail_loss_reduction_ratio"] == 0.0


def test_phase4_lambda_stress_gate_fails_for_oscillating_controller() -> None:
    examples = _adaptive_examples()
    predictions = _adaptive_predictions(examples)
    config = Phase4AdaptiveSystemConfig(
        detector_config=_phase4_config().detector_config,
        lambda_config=LambdaControllerConfig(
            base_lambda=0.30,
            min_lambda=0.02,
            max_lambda=0.90,
            trend_multiplier=1.80,
            range_multiplier=1.0,
            high_volatility_multiplier=0.20,
            liquidity_stress_multiplier=0.30,
            volatility_sensitivity=0.0,
            drawdown_sensitivity=0.0,
            smoothing_alpha=1.0,
            max_step_change=0.80,
        ),
        execution_config=_phase4_config().execution_config,
        max_accepted_lambda_step=0.05,
        min_tail_loss_reduction_ratio=0.10,
        created_at="2026-06-21T04:02:00Z",
    )

    result = run_phase4_adaptive_system(
        examples=examples,
        predictions=predictions,
        config=config,
    )

    assert not result.passed
    assert result.report.acceptance_criteria["lambda_stability_under_stress"] is False
    assert result.report.adaptive_metrics["max_lambda_step"] > 0.05


def test_phase4_regime_confirmation_smooths_noisy_raw_regimes() -> None:
    examples = tuple(
        _example(
            index,
            signal=0.85,
            volatility=0.070 if index % 2 else 0.012,
            liquidity_depth=500.0,
            spread_bps=5.0,
            shadow_net_return=0.007,
        )
        for index in range(40)
    )
    predictions = _adaptive_predictions(examples)
    config = Phase4AdaptiveSystemConfig(
        detector_config=RegimeDetectorConfig(
            trend_score_threshold=0.50,
            high_volatility_threshold=0.060,
            liquidity_stress_threshold=15.0,
            high_cost_spread_bps_threshold=60.0,
            transition_confirmation_count=3,
        ),
        lambda_config=_phase4_config().lambda_config,
        execution_config=_phase4_config().execution_config,
        min_regime_stability_ratio=0.95,
        max_accepted_lambda_step=0.13,
        max_accepted_aggressiveness_step=0.16,
        min_tail_loss_reduction_ratio=0.0,
        created_at="2026-06-21T04:03:00Z",
    )

    result = run_phase4_adaptive_system(
        examples=examples,
        predictions=predictions,
        config=config,
    )

    assert result.passed
    assert result.report.adaptive_metrics["transition_count"] == 0
    assert result.report.adaptive_metrics["regime_stability_ratio"] == 1.0
