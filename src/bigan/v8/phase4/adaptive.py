"""Phase 4 online adaptive-system replay runner."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.phase0.contracts import MarketData
from bigan.v8.phase0.costs import CostBreakdown, TradingCostModel
from bigan.v8.phase1.contracts import PolicyPrediction, PolicyTrainingExample
from bigan.v8.phase4.contracts import (
    PHASE4_ADAPTIVE_SYSTEM_PHASE,
    AdaptiveDecision,
    ExecutionAdaptationConfig,
    LambdaControllerConfig,
    Phase4AdaptiveError,
    Phase4AdaptiveSystemConfig,
    Phase4AdaptiveSystemReport,
    Phase4InputProvenance,
    RegimeClassification,
    RegimeDetectorConfig,
    RegimeName,
)


@dataclass(frozen=True, slots=True)
class Phase4AdaptiveSystemResult:
    """Result of a Phase 4 adaptive-system replay."""

    decisions: tuple[AdaptiveDecision, ...]
    report: Phase4AdaptiveSystemReport
    report_path: Path | None = None

    @property
    def passed(self) -> bool:
        return self.report.passed


def run_phase4_adaptive_system(
    *,
    examples: tuple[PolicyTrainingExample, ...],
    predictions: tuple[PolicyPrediction, ...],
    provenance: Phase4InputProvenance | None = None,
    config: Phase4AdaptiveSystemConfig | None = None,
) -> Phase4AdaptiveSystemResult:
    """Replay an online adaptive overlay over frozen policy predictions.

    Regime, lambda, and execution-aggressiveness decisions are based only on the
    current row and prior adaptive state. ``shadow_net_return`` is consumed only
    after the action has been chosen, for offline acceptance measurement.
    """

    resolved_config = config or Phase4AdaptiveSystemConfig()
    _assert_safe_stream(examples, predictions)
    example_stream_sha256 = compute_phase4_example_stream_sha256(examples)
    prediction_stream_sha256 = compute_phase4_prediction_stream_sha256(predictions)
    input_provenance_verified = _input_provenance_verified(
        provenance=provenance,
        example_stream_sha256=example_stream_sha256,
        prediction_stream_sha256=prediction_stream_sha256,
    )
    decisions = _simulate_decisions(
        examples=examples,
        predictions=predictions,
        config=resolved_config,
        volatility_multiplier=1.0,
        drawdown_shock=0.0,
    )
    adaptive_metrics = _adaptive_metrics(
        decisions,
        tail_quantile=resolved_config.tail_quantile,
    )
    baseline_metrics = _baseline_metrics(
        decisions,
        tail_quantile=resolved_config.tail_quantile,
    )
    comparison_metrics = _comparison_metrics(
        adaptive_metrics=adaptive_metrics,
        baseline_metrics=baseline_metrics,
    )
    stress_metrics = _stress_metrics(
        examples=examples,
        predictions=predictions,
        config=resolved_config,
    )
    report = Phase4AdaptiveSystemReport(
        phase=PHASE4_ADAPTIVE_SYSTEM_PHASE,
        row_count=len(decisions),
        candidate_run_id=None if provenance is None else provenance.candidate_run_id,
        policy_dataset_hash=(
            None if provenance is None else provenance.policy_dataset_hash
        ),
        split_hash=None if provenance is None else provenance.split_hash,
        model_sha256=None if provenance is None else provenance.model_sha256,
        phase2_report_sha256=(
            None if provenance is None else provenance.phase2_report_sha256
        ),
        phase3_report_sha256=(
            None if provenance is None else provenance.phase3_report_sha256
        ),
        example_stream_sha256=example_stream_sha256,
        prediction_stream_sha256=prediction_stream_sha256,
        input_provenance_verified=input_provenance_verified,
        baseline_type="non_adaptive_frozen_policy_execution",
        baseline_execution_config_sha256=_canonical_payload_sha256(
            resolved_config.execution_config.to_dict()
        ),
        decision_trace_sha256=compute_phase4_decision_trace_sha256(decisions),
        decision_count=len(decisions),
        first_decision_ts=decisions[0].decision_ts if decisions else None,
        last_decision_ts=decisions[-1].decision_ts if decisions else None,
        adaptive_metrics=adaptive_metrics,
        baseline_metrics=baseline_metrics,
        comparison_metrics=comparison_metrics,
        stress_metrics=stress_metrics,
        regime_counts=_regime_counts(decisions),
        acceptance_criteria=_acceptance_criteria(
            adaptive_metrics=adaptive_metrics,
            comparison_metrics=comparison_metrics,
            stress_metrics=stress_metrics,
            input_provenance_verified=input_provenance_verified,
            config=resolved_config,
        ),
        config=resolved_config.to_dict(),
        created_at=resolved_config.created_at,
    )
    report_path = None
    if resolved_config.output_dir is not None:
        output_dir = Path(resolved_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "phase4_adaptive_system_report.json"
        report_path.write_text(
            json.dumps(
                _json_ready(report.to_dict()),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
    return Phase4AdaptiveSystemResult(
        decisions=decisions,
        report=report,
        report_path=report_path,
    )


def build_phase4_input_provenance(
    *,
    examples: tuple[PolicyTrainingExample, ...],
    predictions: tuple[PolicyPrediction, ...],
    candidate_run_id: str,
    policy_dataset_hash: str,
    split_hash: str,
    model_sha256: str,
    phase2_report_sha256: str | None = None,
    phase3_report_sha256: str | None = None,
) -> Phase4InputProvenance:
    """Build deterministic input provenance for a Phase 4 replay."""

    return Phase4InputProvenance(
        candidate_run_id=candidate_run_id,
        policy_dataset_hash=policy_dataset_hash,
        split_hash=split_hash,
        model_sha256=model_sha256,
        phase2_report_sha256=phase2_report_sha256,
        phase3_report_sha256=phase3_report_sha256,
        example_stream_sha256=compute_phase4_example_stream_sha256(examples),
        prediction_stream_sha256=compute_phase4_prediction_stream_sha256(predictions),
    )


def compute_phase4_example_stream_sha256(
    examples: tuple[PolicyTrainingExample, ...],
) -> str:
    """Hash the exact Phase 4 example stream in replay order."""

    return _canonical_payload_sha256([example.to_dict() for example in examples])


def compute_phase4_prediction_stream_sha256(
    predictions: tuple[PolicyPrediction, ...],
) -> str:
    """Hash the exact Phase 4 prediction stream in replay order."""

    return _canonical_payload_sha256([prediction.to_dict() for prediction in predictions])


def compute_phase4_decision_trace_sha256(
    decisions: tuple[AdaptiveDecision, ...],
) -> str:
    """Hash the in-memory Phase 4 adaptive decision trace."""

    return _canonical_payload_sha256([decision.to_dict() for decision in decisions])


@dataclass(slots=True)
class _AdaptiveState:
    regime: RegimeName | None = None
    pending_regime: RegimeName | None = None
    pending_count: int = 0
    smoothed_volatility: float | None = None
    lambda_value: float | None = None
    execution_aggressiveness: float | None = None
    previous_filled_action: float = 0.0
    previous_baseline_filled_action: float = 0.0
    cumulative_return: float = 0.0
    peak_cumulative_return: float = 0.0


def _simulate_decisions(
    *,
    examples: tuple[PolicyTrainingExample, ...],
    predictions: tuple[PolicyPrediction, ...],
    config: Phase4AdaptiveSystemConfig,
    volatility_multiplier: float,
    drawdown_shock: float,
) -> tuple[AdaptiveDecision, ...]:
    cost_model = TradingCostModel(config.execution_config.cost_model_config)
    state_by_key: dict[tuple[str, str], _AdaptiveState] = {}
    decisions: list[AdaptiveDecision] = []
    for example, prediction in zip(examples, predictions, strict=True):
        key = (example.source, example.instrument_id)
        state = state_by_key.setdefault(key, _AdaptiveState())
        classification = _classify_regime(
            example=example,
            state=state,
            config=config.detector_config,
            volatility_multiplier=volatility_multiplier,
        )
        drawdown_before = max(
            0.0,
            state.peak_cumulative_return - state.cumulative_return + drawdown_shock,
        )
        lambda_value = _next_lambda_value(
            classification=classification,
            previous=state.lambda_value,
            drawdown=drawdown_before,
            config=config.lambda_config,
        )
        execution_aggressiveness = _next_execution_aggressiveness(
            classification=classification,
            previous=state.execution_aggressiveness,
            detector_config=config.detector_config,
            config=config.execution_config,
        )
        adapted_action = _bounded(
            prediction.action
            * (lambda_value / config.lambda_config.base_lambda)
            * execution_aggressiveness,
            0.0,
            1.0,
        )
        fill_probability = _fill_probability(
            example=example,
            action=adapted_action,
            config=config.execution_config,
        )
        filled_action = adapted_action * fill_probability
        turnover = abs(filled_action - state.previous_filled_action)
        scaled_costs = _scaled_execution_costs(
            example=example,
            turnover=turnover,
            config=config.execution_config,
            cost_model=cost_model,
            volatility_multiplier=volatility_multiplier,
        )
        risk_penalty = _risk_penalty(
            action=filled_action,
            example=example,
            config=config.execution_config,
            volatility_multiplier=volatility_multiplier,
        )
        turnover_penalty = config.execution_config.turnover_penalty_factor * turnover
        gross_return = filled_action * example.shadow_net_return
        net_return = gross_return - scaled_costs.total_cost - risk_penalty - turnover_penalty
        baseline_net_return, baseline_filled_action = _baseline_net_return(
            example=example,
            prediction=prediction,
            previous_filled_action=state.previous_baseline_filled_action,
            config=config.execution_config,
            cost_model=cost_model,
            volatility_multiplier=volatility_multiplier,
        )
        state.cumulative_return += net_return
        state.peak_cumulative_return = max(
            state.peak_cumulative_return,
            state.cumulative_return,
        )
        decisions.append(
            AdaptiveDecision(
                decision_ts=example.decision_ts,
                source=example.source,
                instrument_id=example.instrument_id,
                raw_action=prediction.action,
                adapted_action=adapted_action,
                filled_action=filled_action,
                confidence=prediction.confidence,
                score=prediction.score,
                regime=classification.regime,
                raw_regime=classification.raw_regime,
                pending_regime_active=classification.pending_regime_active,
                transitioned=classification.transitioned,
                lambda_value=lambda_value,
                execution_aggressiveness=execution_aggressiveness,
                fill_probability=fill_probability,
                turnover=turnover,
                shadow_net_return=example.shadow_net_return,
                gross_return=gross_return,
                spread_cost=scaled_costs.spread_cost,
                fee_cost=scaled_costs.fee_cost,
                slippage_cost=scaled_costs.slippage_cost,
                liquidity_impact_cost=scaled_costs.liquidity_impact_cost,
                total_execution_cost=scaled_costs.total_cost,
                risk_penalty=risk_penalty,
                turnover_penalty=turnover_penalty,
                net_return=net_return,
                baseline_net_return=baseline_net_return,
                drawdown=max(
                    0.0,
                    state.peak_cumulative_return - state.cumulative_return,
                ),
            )
        )
        state.lambda_value = lambda_value
        state.execution_aggressiveness = execution_aggressiveness
        state.previous_filled_action = filled_action
        state.previous_baseline_filled_action = baseline_filled_action
    return tuple(decisions)


def _assert_safe_stream(
    examples: tuple[PolicyTrainingExample, ...],
    predictions: tuple[PolicyPrediction, ...],
) -> None:
    if not examples:
        raise Phase4AdaptiveError("Phase 4 requires at least one example")
    if len(examples) != len(predictions):
        raise Phase4AdaptiveError("examples and predictions must have the same length")
    previous_ts: int | None = None
    for example, prediction in zip(examples, predictions, strict=True):
        if previous_ts is not None and example.decision_ts < previous_ts:
            raise Phase4AdaptiveError(
                "examples must be ordered by non-decreasing decision_ts"
            )
        previous_ts = example.decision_ts
        if (
            example.decision_ts != prediction.decision_ts
            or example.source != prediction.source
            or example.instrument_id != prediction.instrument_id
        ):
            raise Phase4AdaptiveError("prediction keys must match policy examples")


def _classify_regime(
    *,
    example: PolicyTrainingExample,
    state: _AdaptiveState,
    config: RegimeDetectorConfig,
    volatility_multiplier: float,
) -> RegimeClassification:
    trend_score = _trend_score(example)
    raw_volatility = _volatility(example) * volatility_multiplier
    if state.smoothed_volatility is None:
        volatility = raw_volatility
    else:
        volatility = state.smoothed_volatility + config.volatility_smoothing_alpha * (
            raw_volatility - state.smoothed_volatility
        )
    state.smoothed_volatility = volatility
    liquidity_depth = max(0.0, _feature_float(example, "liquidity_depth") or 0.0)
    spread_bps = _spread_bps(example)
    raw_regime = _raw_regime(
        trend_score=trend_score,
        volatility=raw_volatility,
        liquidity_depth=liquidity_depth,
        spread_bps=spread_bps,
        config=config,
    )
    confirmation_count = config.transition_confirmation_count
    transitioned = False
    if state.regime is None:
        state.regime = raw_regime
        state.pending_regime = None
        state.pending_count = 0
    elif raw_regime == state.regime:
        state.pending_regime = None
        state.pending_count = 0
    else:
        if state.pending_regime == raw_regime:
            state.pending_count += 1
        else:
            state.pending_regime = raw_regime
            state.pending_count = 1
        confirmation_count = state.pending_count
        if state.pending_count >= config.transition_confirmation_count:
            state.regime = raw_regime
            state.pending_regime = None
            state.pending_count = 0
            transitioned = True
            confirmation_count = config.transition_confirmation_count
    if state.regime is None:
        raise Phase4AdaptiveError("regime detector failed to emit a regime")
    return RegimeClassification(
        raw_regime=raw_regime,
        regime=state.regime,
        trend_score=trend_score,
        volatility=volatility,
        liquidity_depth=liquidity_depth,
        spread_bps=spread_bps,
        confirmation_count=confirmation_count,
        pending_regime_active=state.pending_regime is not None,
        transitioned=transitioned,
    )


def _raw_regime(
    *,
    trend_score: float,
    volatility: float,
    liquidity_depth: float,
    spread_bps: float,
    config: RegimeDetectorConfig,
) -> RegimeName:
    if volatility >= config.high_volatility_threshold:
        return "high_volatility"
    if (
        liquidity_depth <= config.liquidity_stress_threshold
        or spread_bps >= config.high_cost_spread_bps_threshold
    ):
        return "liquidity_stress"
    if trend_score >= config.trend_score_threshold:
        return "trend"
    return "range"


def _next_lambda_value(
    *,
    classification: RegimeClassification,
    previous: float | None,
    drawdown: float,
    config: LambdaControllerConfig,
) -> float:
    previous_value = config.base_lambda if previous is None else previous
    target = config.base_lambda * _regime_lambda_multiplier(classification.regime, config)
    volatility_dampener = 1.0 / (
        1.0 + config.volatility_sensitivity * max(0.0, classification.volatility)
    )
    drawdown_dampener = max(
        config.min_drawdown_multiplier,
        1.0 - config.drawdown_sensitivity * max(0.0, drawdown),
    )
    target = _bounded(
        target * volatility_dampener * drawdown_dampener,
        config.min_lambda,
        config.max_lambda,
    )
    smoothed = previous_value + config.smoothing_alpha * (target - previous_value)
    return _bounded(
        _bounded_step(
            previous=previous_value,
            current=smoothed,
            max_step_change=config.max_step_change,
        ),
        config.min_lambda,
        config.max_lambda,
    )


def _next_execution_aggressiveness(
    *,
    classification: RegimeClassification,
    previous: float | None,
    detector_config: RegimeDetectorConfig,
    config: ExecutionAdaptationConfig,
) -> float:
    previous_value = config.base_aggressiveness if previous is None else previous
    target = config.base_aggressiveness * _regime_aggressiveness_multiplier(
        classification.regime,
        config,
    )
    cost_score = (
        classification.spread_bps / detector_config.high_cost_spread_bps_threshold
        + classification.volatility / detector_config.high_volatility_threshold
    )
    cost_multiplier = 1.0 / (1.0 + config.cost_sensitivity * max(0.0, cost_score))
    if detector_config.liquidity_stress_threshold <= 0.0:
        liquidity_multiplier = 1.0
    else:
        liquidity_ratio = (
            classification.liquidity_depth / detector_config.liquidity_stress_threshold
        )
        liquidity_multiplier = _bounded(liquidity_ratio, 0.0, 1.0)
        if config.liquidity_sensitivity != 1.0:
            liquidity_multiplier = liquidity_multiplier**config.liquidity_sensitivity
    target = _bounded(
        target * cost_multiplier * liquidity_multiplier,
        config.min_aggressiveness,
        config.max_aggressiveness,
    )
    smoothed = previous_value + config.smoothing_alpha * (target - previous_value)
    return _bounded(
        _bounded_step(
            previous=previous_value,
            current=smoothed,
            max_step_change=config.max_step_change,
        ),
        config.min_aggressiveness,
        config.max_aggressiveness,
    )


def _baseline_net_return(
    *,
    example: PolicyTrainingExample,
    prediction: PolicyPrediction,
    previous_filled_action: float,
    config: ExecutionAdaptationConfig,
    cost_model: TradingCostModel,
    volatility_multiplier: float,
) -> tuple[float, float]:
    fill_probability = _fill_probability(
        example=example,
        action=prediction.action,
        config=config,
    )
    filled_action = prediction.action * fill_probability
    turnover = abs(filled_action - previous_filled_action)
    scaled_costs = _scaled_execution_costs(
        example=example,
        turnover=turnover,
        config=config,
        cost_model=cost_model,
        volatility_multiplier=volatility_multiplier,
    )
    risk_penalty = _risk_penalty(
        action=filled_action,
        example=example,
        config=config,
        volatility_multiplier=volatility_multiplier,
    )
    net_return = (
        filled_action * example.shadow_net_return
        - scaled_costs.total_cost
        - risk_penalty
        - config.turnover_penalty_factor * turnover
    )
    return net_return, filled_action


def _scaled_execution_costs(
    *,
    example: PolicyTrainingExample,
    turnover: float,
    config: ExecutionAdaptationConfig,
    cost_model: TradingCostModel,
    volatility_multiplier: float,
) -> CostBreakdown:
    bounded_turnover = max(0.0, turnover)
    cost = cost_model.estimate(
        entry=_market_row_from_example(example),
        order_size=max(config.min_order_size, config.order_size * bounded_turnover),
        volatility=_volatility(example) * volatility_multiplier,
        slippage_multiplier=config.slippage_multiplier,
    )
    return CostBreakdown(
        spread_cost=cost.spread_cost * bounded_turnover,
        fee_cost=cost.fee_cost * bounded_turnover,
        slippage_cost=cost.slippage_cost * bounded_turnover,
        liquidity_impact_cost=cost.liquidity_impact_cost * bounded_turnover,
    )


def _market_row_from_example(example: PolicyTrainingExample) -> MarketData:
    mid_price = _feature_float(example, "mid_price") or 100.0
    spread = _feature_float(example, "spread")
    if spread is None:
        spread = _spread_bps(example) / 10_000.0 * mid_price
    spread = max(0.0, min(spread, mid_price * 0.99))
    return MarketData(
        ts=example.decision_ts,
        available_at_ts=example.decision_ts,
        source=example.source,
        instrument_id=example.instrument_id,
        bid_price=max(1e-12, mid_price - spread / 2.0),
        ask_price=max(1e-12, mid_price + spread / 2.0),
        volume=_feature_float(example, "volume_1m") or 0.0,
        trade_count=int(_feature_float(example, "trade_count_1m") or 0),
        liquidity_depth=_feature_float(example, "liquidity_depth"),
    )


def _fill_probability(
    *,
    example: PolicyTrainingExample,
    action: float,
    config: ExecutionAdaptationConfig,
) -> float:
    if action <= config.active_action_epsilon:
        return 0.0
    liquidity = max(0.0, _feature_float(example, "liquidity_depth") or 0.0)
    if liquidity <= 0.0:
        return config.min_fill_probability
    requested_size = config.order_size * action
    probability = liquidity / (liquidity + requested_size)
    return max(config.min_fill_probability, min(1.0, probability))


def _risk_penalty(
    *,
    action: float,
    example: PolicyTrainingExample,
    config: ExecutionAdaptationConfig,
    volatility_multiplier: float,
) -> float:
    return (
        config.risk_penalty_factor
        * action
        * action
        * max(0.0, _volatility(example) * volatility_multiplier)
    )


def _stress_metrics(
    *,
    examples: tuple[PolicyTrainingExample, ...],
    predictions: tuple[PolicyPrediction, ...],
    config: Phase4AdaptiveSystemConfig,
) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for multiplier in config.stress_volatility_multipliers:
        stressed_decisions = _simulate_decisions(
            examples=examples,
            predictions=predictions,
            config=config,
            volatility_multiplier=multiplier,
            drawdown_shock=config.stress_drawdown_shock,
        )
        stress_adaptive = _adaptive_metrics(
            stressed_decisions,
            tail_quantile=config.tail_quantile,
        )
        stress_baseline = _baseline_metrics(
            stressed_decisions,
            tail_quantile=config.tail_quantile,
        )
        stress_comparison = _comparison_metrics(
            adaptive_metrics=stress_adaptive,
            baseline_metrics=stress_baseline,
        )
        metrics[f"{multiplier:g}"] = {
            **stress_adaptive,
            "baseline_tail_loss": stress_baseline["tail_loss"],
            "tail_loss_reduction_ratio": (
                stress_comparison["tail_loss_reduction_ratio"]
            ),
        }
    return metrics


def _adaptive_metrics(
    decisions: tuple[AdaptiveDecision, ...],
    *,
    tail_quantile: float,
) -> dict[str, Any]:
    net_returns = [decision.net_return for decision in decisions]
    lambdas = [decision.lambda_value for decision in decisions]
    aggressiveness = [decision.execution_aggressiveness for decision in decisions]
    confirmed_transition_count = sum(1 for decision in decisions if decision.transitioned)
    raw_transition_count = sum(
        1
        for previous, current in zip(decisions, decisions[1:], strict=False)
        if previous.raw_regime != current.raw_regime
    )
    pending_regime_count = sum(1 for decision in decisions if decision.pending_regime_active)
    transition_denominator = max(len(decisions) - 1, 1)
    raw_transition_rate = raw_transition_count / transition_denominator
    confirmed_transition_rate = confirmed_transition_count / transition_denominator
    pending_regime_rate = pending_regime_count / len(decisions) if decisions else 0.0
    suppression_ratio = (
        (raw_transition_count - confirmed_transition_count) / raw_transition_count
        if raw_transition_count > 0
        else 0.0
    )
    lambda_by_regime: dict[str, list[float]] = {}
    action_by_regime: dict[str, list[float]] = {}
    for decision in decisions:
        lambda_by_regime.setdefault(decision.regime, []).append(decision.lambda_value)
        action_by_regime.setdefault(decision.regime, []).append(decision.filled_action)
    return {
        "row_count": len(decisions),
        "mean_net_return": _mean(net_returns),
        "adaptive_sharpe": _sharpe(net_returns),
        "tail_loss": _tail_loss(net_returns, tail_quantile),
        "max_drawdown": max((decision.drawdown for decision in decisions), default=0.0),
        "mean_lambda": _mean(lambdas),
        "min_lambda": min(lambdas) if lambdas else 0.0,
        "max_lambda": max(lambdas) if lambdas else 0.0,
        "max_lambda_step": max(_absolute_steps(lambdas), default=0.0),
        "lambda_oscillation_count": _oscillation_count(lambdas),
        "lambda_oscillation_rate": _oscillation_rate(lambdas),
        "mean_execution_aggressiveness": _mean(aggressiveness),
        "max_aggressiveness_step": max(_absolute_steps(aggressiveness), default=0.0),
        "mean_filled_action": _mean([decision.filled_action for decision in decisions]),
        "mean_adapted_action": _mean([decision.adapted_action for decision in decisions]),
        "mean_fill_probability": _mean(
            [decision.fill_probability for decision in decisions]
        ),
        "mean_execution_cost": _mean(
            [decision.total_execution_cost for decision in decisions]
        ),
        "mean_risk_penalty": _mean([decision.risk_penalty for decision in decisions]),
        "mean_abs_turnover": _mean([decision.turnover for decision in decisions]),
        "raw_regime_transition_count": raw_transition_count,
        "raw_regime_transition_rate": raw_transition_rate,
        "confirmed_regime_transition_count": confirmed_transition_count,
        "confirmed_regime_transition_rate": confirmed_transition_rate,
        "pending_regime_count": pending_regime_count,
        "pending_regime_rate": pending_regime_rate,
        "raw_to_confirmed_transition_suppression_ratio": suppression_ratio,
        "transition_count": confirmed_transition_count,
        "transition_rate": confirmed_transition_rate,
        "regime_stability_ratio": 1.0 - confirmed_transition_rate,
        "mean_lambda_by_regime": {
            regime: _mean(values) for regime, values in sorted(lambda_by_regime.items())
        },
        "mean_filled_action_by_regime": {
            regime: _mean(values) for regime, values in sorted(action_by_regime.items())
        },
    }


def _baseline_metrics(
    decisions: tuple[AdaptiveDecision, ...],
    *,
    tail_quantile: float,
) -> dict[str, Any]:
    returns = [decision.baseline_net_return for decision in decisions]
    return {
        "row_count": len(decisions),
        "mean_net_return": _mean(returns),
        "baseline_sharpe": _sharpe(returns),
        "tail_loss": _tail_loss(returns, tail_quantile),
        "max_drawdown": _max_drawdown(returns),
    }


def _comparison_metrics(
    *,
    adaptive_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    baseline_tail_loss = float(baseline_metrics["tail_loss"])
    adaptive_tail_loss = float(adaptive_metrics["tail_loss"])
    if baseline_tail_loss <= 1e-12:
        tail_loss_reduction_ratio = 0.0 if adaptive_tail_loss <= 1e-12 else -1.0
    else:
        tail_loss_reduction_ratio = (
            baseline_tail_loss - adaptive_tail_loss
        ) / baseline_tail_loss
    return {
        "mean_net_return_delta": (
            float(adaptive_metrics["mean_net_return"])
            - float(baseline_metrics["mean_net_return"])
        ),
        "sharpe_delta": (
            float(adaptive_metrics["adaptive_sharpe"])
            - float(baseline_metrics["baseline_sharpe"])
        ),
        "adaptive_tail_loss": adaptive_tail_loss,
        "baseline_tail_loss": baseline_tail_loss,
        "tail_loss_delta": baseline_tail_loss - adaptive_tail_loss,
        "tail_loss_reduction_ratio": tail_loss_reduction_ratio,
        "max_drawdown_delta": (
            float(baseline_metrics["max_drawdown"])
            - float(adaptive_metrics["max_drawdown"])
        ),
    }


def _acceptance_criteria(
    *,
    adaptive_metrics: dict[str, Any],
    comparison_metrics: dict[str, Any],
    stress_metrics: dict[str, dict[str, Any]],
    input_provenance_verified: bool,
    config: Phase4AdaptiveSystemConfig,
) -> dict[str, bool]:
    stress_lambda_stable = bool(stress_metrics) and all(
        metrics["max_lambda_step"] <= config.max_accepted_lambda_step + 1e-12
        and metrics["lambda_oscillation_rate"]
        <= config.max_accepted_lambda_oscillation_rate + 1e-12
        and config.lambda_config.min_lambda - 1e-12 <= metrics["min_lambda"]
        and metrics["max_lambda"] <= config.lambda_config.max_lambda + 1e-12
        for metrics in stress_metrics.values()
    )
    return {
        "causal_stream_ordered": True,
        "prediction_keys_match_examples": True,
        "input_provenance_verified": input_provenance_verified,
        "regime_detector_active": adaptive_metrics["row_count"] > 0,
        "raw_regime_flicker_bounded": (
            adaptive_metrics["raw_regime_transition_rate"]
            <= config.max_raw_regime_transition_rate + 1e-12
            and adaptive_metrics["pending_regime_rate"]
            <= config.max_pending_regime_rate + 1e-12
        ),
        "confirmed_regime_transitions_stable": (
            adaptive_metrics["regime_stability_ratio"]
            >= config.min_regime_stability_ratio
        ),
        "regime_transitions_stable": (
            adaptive_metrics["regime_stability_ratio"]
            >= config.min_regime_stability_ratio
        ),
        "lambda_values_bounded": (
            config.lambda_config.min_lambda - 1e-12
            <= adaptive_metrics["min_lambda"]
            and adaptive_metrics["max_lambda"]
            <= config.lambda_config.max_lambda + 1e-12
        ),
        "lambda_stability": (
            adaptive_metrics["max_lambda_step"]
            <= config.max_accepted_lambda_step + 1e-12
            and adaptive_metrics["lambda_oscillation_rate"]
            <= config.max_accepted_lambda_oscillation_rate + 1e-12
        ),
        "lambda_stability_under_stress": stress_lambda_stable,
        "execution_adaptation_stable": (
            adaptive_metrics["max_aggressiveness_step"]
            <= config.max_accepted_aggressiveness_step + 1e-12
        ),
        "tail_risk_performance_improved": (
            comparison_metrics["tail_loss_reduction_ratio"]
            >= config.min_tail_loss_reduction_ratio
            and comparison_metrics["adaptive_tail_loss"]
            <= comparison_metrics["baseline_tail_loss"] + 1e-12
        ),
        "adaptive_metrics_finite": _metrics_are_finite(adaptive_metrics),
        "comparison_metrics_finite": _metrics_are_finite(comparison_metrics),
        "stress_metrics_finite": _metrics_are_finite(stress_metrics),
    }


def _regime_counts(decisions: tuple[AdaptiveDecision, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision.regime] = counts.get(decision.regime, 0) + 1
    return dict(sorted(counts.items()))


def _regime_lambda_multiplier(
    regime: RegimeName,
    config: LambdaControllerConfig,
) -> float:
    if regime == "trend":
        return config.trend_multiplier
    if regime == "high_volatility":
        return config.high_volatility_multiplier
    if regime == "liquidity_stress":
        return config.liquidity_stress_multiplier
    return config.range_multiplier


def _regime_aggressiveness_multiplier(
    regime: RegimeName,
    config: ExecutionAdaptationConfig,
) -> float:
    if regime == "trend":
        return config.trend_multiplier
    if regime == "high_volatility":
        return config.high_volatility_multiplier
    if regime == "liquidity_stress":
        return config.liquidity_stress_multiplier
    return config.range_multiplier


def _trend_score(example: PolicyTrainingExample) -> float:
    for name in ("trend_score", "trend_strength", "signal"):
        value = _feature_float(example, name)
        if value is not None:
            return _bounded(abs(value), 0.0, 1.0)
    momentum_values = [
        abs(value)
        for name in ("momentum_5m", "return_5m", "return_15m")
        if (value := _feature_float(example, name)) is not None
    ]
    return _bounded(max(momentum_values, default=0.0), 0.0, 1.0)


def _volatility(example: PolicyTrainingExample) -> float:
    values = [
        value
        for name in ("volatility_5m", "volatility_15m")
        if (value := _feature_float(example, name)) is not None
    ]
    return max(0.0, max(values, default=0.0))


def _spread_bps(example: PolicyTrainingExample) -> float:
    explicit = _feature_float(example, "spread_bps")
    if explicit is not None:
        return max(0.0, explicit)
    spread = _feature_float(example, "spread")
    mid_price = _feature_float(example, "mid_price")
    if spread is None or mid_price is None or mid_price <= 0.0:
        return 0.0
    return max(0.0, spread / mid_price * 10_000.0)


def _feature_float(example: PolicyTrainingExample, name: str) -> float | None:
    value = example.features.get(name)
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _absolute_steps(values: list[float]) -> list[float]:
    return [
        abs(current - previous)
        for previous, current in zip(values, values[1:], strict=False)
    ]


def _oscillation_count(values: list[float]) -> int:
    deltas = [
        current - previous
        for previous, current in zip(values, values[1:], strict=False)
        if abs(current - previous) > 0.005
    ]
    return sum(
        1
        for previous, current in zip(deltas, deltas[1:], strict=False)
        if previous * current < 0.0
    )


def _oscillation_rate(values: list[float]) -> float:
    possible_turns = max(len(values) - 2, 1)
    return _oscillation_count(values) / possible_turns


def _bounded(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _bounded_step(
    *,
    previous: float,
    current: float,
    max_step_change: float,
) -> float:
    return previous + _bounded(
        current - previous,
        -max_step_change,
        max_step_change,
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sharpe(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean / math.sqrt(variance + 1e-12) * math.sqrt(float(len(values)))


def _tail_loss(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int(math.floor(quantile * (len(ordered) - 1)))),
    )
    return max(0.0, -ordered[index])


def _max_drawdown(returns: list[float]) -> float:
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in returns:
        cumulative += value
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
    return max_drawdown


def _metrics_are_finite(value: Any) -> bool:
    if isinstance(value, bool | str):
        return True
    if isinstance(value, int | float):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_metrics_are_finite(item) for item in value.values())
    if isinstance(value, list | tuple):
        return all(_metrics_are_finite(item) for item in value)
    return True


def _input_provenance_verified(
    *,
    provenance: Phase4InputProvenance | None,
    example_stream_sha256: str,
    prediction_stream_sha256: str,
) -> bool:
    return (
        provenance is not None
        and provenance.example_stream_sha256 == example_stream_sha256
        and provenance.prediction_stream_sha256 == prediction_stream_sha256
    )


def _canonical_payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        _json_ready(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value
