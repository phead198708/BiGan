"""Phase 5 safety-layer tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.v8.phase4 import AdaptiveDecision
from bigan.v8.phase5 import (
    PHASE5_SAFETY_LAYER_PHASE,
    LiveExecutionObservation,
    Phase5SafetyError,
    SafetyLayerConfig,
    StableModelSnapshot,
    run_phase5_safety_layer,
)


def _decision(
    index: int,
    *,
    net_return: float,
    total_cost: float = 0.001,
    regime: str = "trend",
    filled_action: float = 0.70,
) -> AdaptiveDecision:
    return AdaptiveDecision(
        decision_ts=2_000_000 + index * 60_000,
        source="polymarket",
        instrument_id="btc-up",
        raw_action=filled_action,
        adapted_action=filled_action,
        filled_action=filled_action,
        confidence=0.82,
        score=0.78,
        regime=regime,
        raw_regime=regime,
        pending_regime_active=False,
        transitioned=False,
        lambda_value=0.30,
        execution_aggressiveness=0.90,
        fill_probability=1.0,
        turnover=0.02,
        shadow_net_return=net_return,
        gross_return=net_return + total_cost,
        spread_cost=total_cost * 0.25,
        fee_cost=total_cost * 0.25,
        slippage_cost=total_cost * 0.25,
        liquidity_impact_cost=total_cost * 0.25,
        total_execution_cost=total_cost,
        risk_penalty=0.0,
        turnover_penalty=0.0,
        net_return=net_return,
        baseline_net_return=net_return,
        drawdown=0.0,
    )


def _live(
    shadow: AdaptiveDecision,
    *,
    live_net_return: float,
    live_total_execution_cost: float = 0.0011,
    live_regime: str | None = None,
) -> LiveExecutionObservation:
    return LiveExecutionObservation(
        decision_ts=shadow.decision_ts,
        source=shadow.source,
        instrument_id=shadow.instrument_id,
        live_filled_action=shadow.filled_action,
        live_net_return=live_net_return,
        live_total_execution_cost=live_total_execution_cost,
        live_regime=live_regime or shadow.regime,
        capital_at_risk=True,
    )


def _stable_model() -> StableModelSnapshot:
    return StableModelSnapshot(
        model_id="stable-phase4-model",
        model_sha256="a" * 64,
        policy_dataset_hash="b" * 64,
        split_hash="c" * 64,
        safe_parameter_sha256="d" * 64,
        safe_parameters={
            "max_position_size": 0.10,
            "risk_mode": "safe",
        },
    )


def _config(output_dir: Path | None = None) -> SafetyLayerConfig:
    return SafetyLayerConfig(
        detection_window_size=4,
        min_shadow_live_correlation=0.70,
        max_mean_pnl_drift=0.007,
        max_cost_drift_ratio=0.50,
        max_regime_mismatch_rate=0.25,
        max_live_drawdown=0.05,
        output_dir=output_dir,
        created_at="2026-06-21T05:00:00Z",
    )


def test_phase5_shadow_mode_stays_safe_without_degradation(tmp_path: Path) -> None:
    shadow_returns = (0.010, 0.012, 0.009, 0.013, 0.011, 0.014, 0.010, 0.012)
    shadow = tuple(
        _decision(index, net_return=value)
        for index, value in enumerate(shadow_returns)
    )
    live = tuple(
        _live(decision, live_net_return=decision.net_return - 0.0004)
        for decision in shadow
    )

    result = run_phase5_safety_layer(
        shadow_decisions=shadow,
        live_observations=live,
        stable_model=_stable_model(),
        config=_config(output_dir=tmp_path),
    )

    assert result.passed
    assert result.report.phase == PHASE5_SAFETY_LAYER_PHASE
    assert result.report.shadow_mode_metrics["parallel_streams_aligned"] is True
    assert result.report.shadow_mode_metrics["shadow_capital_risk_free"] is True
    assert result.report.drift_metrics["degradation_detected"] is False
    assert result.report.safety_action["kill_switch_triggered"] is False
    assert result.report.acceptance_criteria["safe_when_no_degradation"] is True
    assert result.report_path is not None
    saved = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert saved["phase"] == PHASE5_SAFETY_LAYER_PHASE
    assert saved["passed"] is True
    assert saved["safety_action"]["kill_switch_triggered"] is False


def test_phase5_detects_degradation_before_drawdown_and_rolls_back() -> None:
    shadow_returns = (
        0.010,
        0.012,
        0.009,
        0.013,
        0.011,
        0.014,
        0.010,
        0.012,
        0.011,
        0.013,
        0.010,
        0.012,
    )
    shadow = tuple(
        _decision(index, net_return=value, regime="trend")
        for index, value in enumerate(shadow_returns)
    )
    live_returns = (
        0.009,
        0.011,
        0.008,
        0.012,
        -0.012,
        -0.014,
        -0.013,
        -0.015,
        -0.014,
        -0.016,
        -0.015,
        -0.017,
    )
    live = tuple(
        _live(
            decision,
            live_net_return=live_returns[index],
            live_total_execution_cost=0.004 if index >= 4 else 0.0011,
            live_regime="high_volatility" if index >= 4 else "trend",
        )
        for index, decision in enumerate(shadow)
    )
    stable_model = _stable_model()

    result = run_phase5_safety_layer(
        shadow_decisions=shadow,
        live_observations=live,
        stable_model=stable_model,
        config=_config(),
    )

    assert result.passed
    assert result.report.drift_metrics["degradation_detected"] is True
    assert "pnl_drift" in result.report.drift_metrics["reason_codes"]
    assert "cost_drift" in result.report.drift_metrics["reason_codes"]
    assert "regime_mismatch" in result.report.drift_metrics["reason_codes"]
    assert result.report.live_risk_metrics["drawdown_breach_detected"] is True
    assert result.report.drift_metrics["first_degradation_ts"] <= (
        result.report.live_risk_metrics["first_drawdown_breach_ts"]
    )
    action = result.report.safety_action
    assert action["kill_switch_triggered"] is True
    assert action["stop_trading"] is True
    assert action["flatten_positions"] is True
    assert action["freeze_model_updates"] is True
    assert action["rollback_model_id"] == stable_model.model_id
    assert action["rollback_model_sha256"] == stable_model.model_sha256
    assert action["restored_safe_parameters"] == dict(stable_model.safe_parameters)
    assert result.report.acceptance_criteria["degradation_detected_before_drawdown"] is True
    assert result.report.acceptance_criteria["rollback_executes_reliably"] is True


def test_phase5_correlation_break_triggers_kill_switch() -> None:
    shadow_returns = (0.010, 0.012, 0.009, 0.013, 0.011, 0.014, 0.010, 0.012)
    shadow = tuple(
        _decision(index, net_return=value)
        for index, value in enumerate(shadow_returns)
    )
    live_returns = tuple(reversed(shadow_returns))
    live = tuple(
        _live(decision, live_net_return=live_returns[index])
        for index, decision in enumerate(shadow)
    )

    result = run_phase5_safety_layer(
        shadow_decisions=shadow,
        live_observations=live,
        stable_model=_stable_model(),
        config=_config(),
    )

    assert result.passed
    assert result.report.drift_metrics["correlation_break_detected"] is True
    assert result.report.safety_action["kill_switch_triggered"] is True
    assert (
        result.report.acceptance_criteria[
            "shadow_live_correlation_stable_or_kill_triggered"
        ]
        is True
    )


def test_phase5_rejects_misaligned_live_stream_before_monitoring() -> None:
    shadow = tuple(
        _decision(index, net_return=0.010 + index * 0.001)
        for index in range(5)
    )
    live = [
        _live(decision, live_net_return=decision.net_return)
        for decision in shadow
    ]
    bad = live[0]
    live[0] = LiveExecutionObservation(
        decision_ts=bad.decision_ts,
        source="wrong-source",
        instrument_id=bad.instrument_id,
        live_filled_action=bad.live_filled_action,
        live_net_return=bad.live_net_return,
        live_total_execution_cost=bad.live_total_execution_cost,
        live_regime=bad.live_regime,
        capital_at_risk=bad.capital_at_risk,
    )

    with pytest.raises(Phase5SafetyError, match="shadow and live keys"):
        run_phase5_safety_layer(
            shadow_decisions=shadow,
            live_observations=tuple(live),
            stable_model=_stable_model(),
            config=_config(),
        )
