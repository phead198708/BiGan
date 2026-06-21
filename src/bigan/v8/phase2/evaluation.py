"""Phase 2 execution-consistent PnL evaluation runner."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.phase0.contracts import MarketData
from bigan.v8.phase0.costs import CostBreakdown, TradingCostModel
from bigan.v8.phase1.contracts import (
    PolicyPrediction,
    PolicyTrainingExample,
    PolicyTrainShadowSplit,
)
from bigan.v8.phase2.artifacts import Phase15CandidateArtifact, load_phase15_candidate
from bigan.v8.phase2.contracts import (
    PHASE2_EVALUATION_PHASE,
    ExecutionFill,
    ExecutionSimulationConfig,
    Phase2ArtifactError,
    Phase2EvaluationConfig,
    Phase2EvaluationReport,
)


@dataclass(frozen=True, slots=True)
class Phase2EvaluationResult:
    """Result of Phase 2 evaluation."""

    candidate: Phase15CandidateArtifact
    split: PolicyTrainShadowSplit
    predictions: tuple[PolicyPrediction, ...]
    fills: tuple[ExecutionFill, ...]
    report: Phase2EvaluationReport
    report_path: Path | None = None

    @property
    def passed(self) -> bool:
        return self.report.passed


def run_phase2_evaluation(
    candidate_artifact_dir: Path | str,
    split: PolicyTrainShadowSplit,
    config: Phase2EvaluationConfig | None = None,
) -> Phase2EvaluationResult:
    """Run Phase 2 without retraining the Phase 1.5 policy model."""

    resolved_config = config or Phase2EvaluationConfig()
    candidate = load_phase15_candidate(candidate_artifact_dir)
    _assert_split_matches_candidate(candidate, split)
    predictions = candidate.model.predict_examples(split.shadow_examples)
    fills = simulate_execution(
        examples=split.shadow_examples,
        predictions=predictions,
        config=resolved_config.execution_config,
    )
    report = build_phase2_report(
        candidate=candidate,
        fills=fills,
        config=resolved_config,
    )
    report_path = None
    if resolved_config.output_dir is not None:
        output_dir = Path(resolved_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"{candidate.run_id}_phase2_report.json"
        report_path.write_text(
            json.dumps(_json_ready(report.to_dict()), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
    return Phase2EvaluationResult(
        candidate=candidate,
        split=split,
        predictions=predictions,
        fills=fills,
        report=report,
        report_path=report_path,
    )


def simulate_execution(
    *,
    examples: tuple[PolicyTrainingExample, ...],
    predictions: tuple[PolicyPrediction, ...],
    config: ExecutionSimulationConfig | None = None,
) -> tuple[ExecutionFill, ...]:
    """Apply a deterministic cost/risk/turnover execution overlay."""

    resolved_config = config or ExecutionSimulationConfig()
    if len(examples) != len(predictions):
        raise ValueError("examples and predictions must have the same length")
    cost_model = TradingCostModel(resolved_config.cost_model_config)
    previous_action_by_key: dict[tuple[str, str], float] = {}
    fills: list[ExecutionFill] = []
    for example, prediction in zip(examples, predictions, strict=True):
        _assert_prediction_matches_example(example, prediction)
        key = (example.source, example.instrument_id)
        previous_action = previous_action_by_key.get(key, 0.0)
        raw_action = prediction.action
        raw_turnover = abs(raw_action - previous_action)
        estimated_cost = _estimate_cost(
            example=example,
            turnover=raw_turnover,
            config=resolved_config,
            cost_model=cost_model,
        )
        estimated_total_cost = raw_turnover * estimated_cost.total_cost
        estimated_risk = _risk_penalty(raw_action, example, resolved_config)
        estimated_friction = estimated_total_cost + estimated_risk
        estimated_policy_edge = raw_action * prediction.confidence * resolved_config.policy_edge_scale
        expected_net_edge = estimated_policy_edge - estimated_friction
        low_ev_filtered = (
            resolved_config.apply_cost_aware_filter
            and raw_action > resolved_config.active_action_epsilon
            and expected_net_edge < resolved_config.min_expected_net_edge
        )
        adjusted_action = 0.0 if low_ev_filtered else raw_action
        fill_probability = _fill_probability(example, adjusted_action, resolved_config)
        filled_action = adjusted_action * fill_probability
        turnover = abs(filled_action - previous_action)
        final_cost = _estimate_cost(
            example=example,
            turnover=turnover,
            config=resolved_config,
            cost_model=cost_model,
        )
        scaled_costs = _scale_costs(final_cost, turnover)
        total_execution_cost = scaled_costs.total_cost
        risk_penalty = _risk_penalty(filled_action, example, resolved_config)
        turnover_penalty = resolved_config.turnover_penalty_factor * turnover
        gross_policy_return = filled_action * example.shadow_net_return
        net_execution_return = (
            gross_policy_return
            - total_execution_cost
            - risk_penalty
            - turnover_penalty
        )
        fills.append(
            ExecutionFill(
                decision_ts=example.decision_ts,
                source=example.source,
                instrument_id=example.instrument_id,
                raw_action=raw_action,
                adjusted_action=adjusted_action,
                fill_probability=fill_probability,
                filled_action=filled_action,
                confidence=prediction.confidence,
                score=prediction.score,
                shadow_net_return=example.shadow_net_return,
                gross_policy_return=gross_policy_return,
                spread_cost=scaled_costs.spread_cost,
                fee_cost=scaled_costs.fee_cost,
                slippage_cost=scaled_costs.slippage_cost,
                liquidity_impact_cost=scaled_costs.liquidity_impact_cost,
                total_execution_cost=total_execution_cost,
                risk_penalty=risk_penalty,
                turnover_penalty=turnover_penalty,
                net_execution_return=net_execution_return,
                turnover=turnover,
                estimated_policy_edge=estimated_policy_edge,
                estimated_friction=estimated_friction,
                expected_net_edge=expected_net_edge,
                low_ev_filtered=low_ev_filtered,
            )
        )
        previous_action_by_key[key] = filled_action
    return tuple(fills)


def build_phase2_report(
    *,
    candidate: Phase15CandidateArtifact,
    fills: tuple[ExecutionFill, ...],
    config: Phase2EvaluationConfig,
) -> Phase2EvaluationReport:
    """Build an auditable Phase 2 report from simulated fills."""

    execution_metrics = _execution_metrics(fills, config.execution_config)
    phase1_metrics = _phase1_5_shadow_metrics(candidate)
    comparison_metrics = _comparison_metrics(
        phase1_5_metrics=phase1_metrics,
        execution_metrics=execution_metrics,
    )
    criteria = {
        "phase1_5_candidate_verified": True,
        "execution_adjusted_pnl_reported": "mean_net_execution_return" in execution_metrics,
        "execution_metrics_finite": _metrics_are_finite(execution_metrics),
        "sharpe_improvement_ge_min": (
            comparison_metrics["sharpe_improvement_ratio"]
            >= config.min_sharpe_improvement_ratio
        ),
        "reduced_turnover": (
            comparison_metrics["turnover_reduction_ratio"]
            >= config.min_turnover_reduction_ratio
        ),
        "cost_aware_behavior_emerged": (
            execution_metrics["filtered_low_ev_trade_count"] > 0
            or execution_metrics["mean_execution_cost"] > 0.0
        ),
    }
    return Phase2EvaluationReport(
        phase=PHASE2_EVALUATION_PHASE,
        candidate_run_id=candidate.run_id,
        candidate_artifact_dir=str(candidate.artifact_dir),
        phase1_5_hashes=candidate.phase1_5_hashes(),
        phase1_5_shadow_acceptance_metrics=phase1_metrics,
        execution_metrics=execution_metrics,
        comparison_metrics=comparison_metrics,
        acceptance_criteria=criteria,
        config=config.to_dict(),
        created_at=config.created_at,
    )


def _assert_split_matches_candidate(
    candidate: Phase15CandidateArtifact,
    split: PolicyTrainShadowSplit,
) -> None:
    mismatches: list[str] = []
    if split.split_hash != candidate.split_hash:
        mismatches.append("split_hash")
    if split.train_dataset_hash != candidate.train_dataset_hash:
        mismatches.append("train_dataset_hash")
    if split.shadow_dataset_hash != candidate.shadow_dataset_hash:
        mismatches.append("shadow_dataset_hash")
    if mismatches:
        raise Phase2ArtifactError(
            "Phase 2 split does not match Phase 1.5 candidate provenance: "
            + ", ".join(mismatches)
        )


def _assert_prediction_matches_example(
    example: PolicyTrainingExample,
    prediction: PolicyPrediction,
) -> None:
    if (
        example.decision_ts != prediction.decision_ts
        or example.source != prediction.source
        or example.instrument_id != prediction.instrument_id
    ):
        raise ValueError("prediction keys must match policy examples")


def _estimate_cost(
    *,
    example: PolicyTrainingExample,
    turnover: float,
    config: ExecutionSimulationConfig,
    cost_model: TradingCostModel,
) -> CostBreakdown:
    entry = _market_row_from_example(example)
    order_size = max(config.min_order_size, config.order_size * max(turnover, 0.0))
    return cost_model.estimate(
        entry=entry,
        order_size=order_size,
        volatility=_feature_float(example, "volatility_5m")
        or _feature_float(example, "volatility_15m"),
        slippage_multiplier=config.slippage_multiplier,
    )


def _market_row_from_example(example: PolicyTrainingExample) -> MarketData:
    mid_price = _feature_float(example, "mid_price") or 100.0
    spread = _feature_float(example, "spread")
    if spread is None:
        spread_bps = _feature_float(example, "spread_bps") or 0.0
        spread = max(0.0, spread_bps / 10_000.0 * mid_price)
    spread = max(0.0, spread)
    liquidity_depth = _feature_float(example, "liquidity_depth")
    return MarketData(
        ts=example.decision_ts,
        available_at_ts=example.decision_ts,
        source=example.source,
        instrument_id=example.instrument_id,
        bid_price=max(1e-12, mid_price - spread / 2.0),
        ask_price=max(1e-12, mid_price + spread / 2.0),
        volume=_feature_float(example, "volume_1m") or 0.0,
        trade_count=int(_feature_float(example, "trade_count_1m") or 0),
        liquidity_depth=liquidity_depth,
    )


def _scale_costs(costs: CostBreakdown, scale: float) -> CostBreakdown:
    bounded_scale = max(0.0, scale)
    return CostBreakdown(
        spread_cost=costs.spread_cost * bounded_scale,
        fee_cost=costs.fee_cost * bounded_scale,
        slippage_cost=costs.slippage_cost * bounded_scale,
        liquidity_impact_cost=costs.liquidity_impact_cost * bounded_scale,
    )


def _risk_penalty(
    action: float,
    example: PolicyTrainingExample,
    config: ExecutionSimulationConfig,
) -> float:
    volatility = _feature_float(example, "volatility_5m") or _feature_float(
        example,
        "volatility_15m",
    )
    return config.risk_penalty_factor * action * action * max(0.0, volatility or 0.0)


def _fill_probability(
    example: PolicyTrainingExample,
    action: float,
    config: ExecutionSimulationConfig,
) -> float:
    if action <= config.active_action_epsilon:
        return 0.0
    liquidity = max(0.0, _feature_float(example, "liquidity_depth") or 0.0)
    if liquidity <= 0.0:
        return config.min_fill_probability
    requested_size = config.order_size * action
    probability = liquidity / (liquidity + requested_size)
    return max(config.min_fill_probability, min(1.0, probability))


def _feature_float(example: PolicyTrainingExample, name: str) -> float | None:
    value = example.features.get(name)
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _execution_metrics(
    fills: tuple[ExecutionFill, ...],
    config: ExecutionSimulationConfig,
) -> dict[str, Any]:
    net_returns = [fill.net_execution_return for fill in fills]
    gross_returns = [fill.gross_policy_return for fill in fills]
    costs = [fill.total_execution_cost for fill in fills]
    risk_penalties = [fill.risk_penalty for fill in fills]
    turnover_penalties = [fill.turnover_penalty for fill in fills]
    turnovers = [fill.turnover for fill in fills]
    filled_actions = [fill.filled_action for fill in fills]
    active_count = sum(1 for action in filled_actions if action > config.active_action_epsilon)
    filtered_count = sum(1 for fill in fills if fill.low_ev_filtered)
    mean_confidence = _mean([fill.confidence for fill in fills])
    hybrid_scores = [
        fill.confidence + config.pnl_lambda * fill.net_execution_return
        for fill in fills
    ]
    return {
        "row_count": len(fills),
        "mean_gross_policy_return": _mean(gross_returns),
        "mean_execution_cost": _mean(costs),
        "mean_risk_penalty": _mean(risk_penalties),
        "mean_turnover_penalty": _mean(turnover_penalties),
        "mean_net_execution_return": _mean(net_returns),
        "execution_sharpe": _sharpe(net_returns),
        "total_turnover": sum(turnovers),
        "mean_abs_turnover": _mean(turnovers),
        "active_rate": active_count / len(fills) if fills else 0.0,
        "mean_filled_action": _mean(filled_actions),
        "mean_fill_probability": _mean([fill.fill_probability for fill in fills]),
        "filtered_low_ev_trade_count": filtered_count,
        "filtered_low_ev_trade_rate": filtered_count / len(fills) if fills else 0.0,
        "mean_estimated_friction": _mean([fill.estimated_friction for fill in fills]),
        "mean_expected_net_edge": _mean([fill.expected_net_edge for fill in fills]),
        "mean_policy_confidence": mean_confidence,
        "hybrid_score_mean": _mean(hybrid_scores),
        "pnl_lambda": config.pnl_lambda,
        "cost_to_abs_gross_return_ratio": (
            sum(costs) / max(sum(abs(value) for value in gross_returns), 1e-12)
        ),
    }


def _phase1_5_shadow_metrics(candidate: Phase15CandidateArtifact) -> dict[str, Any]:
    metrics = candidate.shadow_acceptance_report.get("metrics", {})
    if not isinstance(metrics, dict):
        raise Phase2ArtifactError("Phase 1.5 shadow acceptance metrics are required")
    action_distribution = metrics.get("action_distribution", {})
    if not isinstance(action_distribution, dict):
        raise Phase2ArtifactError("Phase 1.5 action_distribution metrics are required")
    return {
        "shadow_sharpe": _required_finite_float(
            metrics,
            "shadow_sharpe",
            "Phase 1.5 shadow acceptance metrics",
        ),
        "mean_shadow_return": _required_finite_float(
            metrics,
            "mean_shadow_return",
            "Phase 1.5 shadow acceptance metrics",
        ),
        "mean_abs_turnover": _required_finite_float(
            action_distribution,
            "mean_abs_turnover",
            "Phase 1.5 action_distribution metrics",
        ),
        "active_rate": _required_finite_float(
            action_distribution,
            "active_rate",
            "Phase 1.5 action_distribution metrics",
        ),
        "row_count": _required_positive_int(
            metrics,
            "row_count",
            "Phase 1.5 shadow acceptance metrics",
        ),
    }


def _comparison_metrics(
    *,
    phase1_5_metrics: dict[str, Any],
    execution_metrics: dict[str, Any],
) -> dict[str, Any]:
    baseline_sharpe = float(phase1_5_metrics["shadow_sharpe"])
    execution_sharpe = float(execution_metrics["execution_sharpe"])
    baseline_turnover = float(phase1_5_metrics["mean_abs_turnover"])
    execution_turnover = float(execution_metrics["mean_abs_turnover"])
    return {
        "phase1_5_shadow_sharpe": baseline_sharpe,
        "phase2_execution_sharpe": execution_sharpe,
        "sharpe_delta": execution_sharpe - baseline_sharpe,
        "sharpe_improvement_ratio": _relative_change(execution_sharpe, baseline_sharpe),
        "phase1_5_mean_abs_turnover": baseline_turnover,
        "phase2_mean_abs_turnover": execution_turnover,
        "turnover_delta": execution_turnover - baseline_turnover,
        "turnover_reduction_ratio": (
            (baseline_turnover - execution_turnover) / max(abs(baseline_turnover), 1e-12)
            if baseline_turnover > 0.0
            else 0.0 if execution_turnover <= 1e-12 else -math.inf
        ),
    }


def _relative_change(current: float, baseline: float) -> float:
    if abs(baseline) <= 1e-12:
        if current > 0.0:
            return math.inf
        if current < 0.0:
            return -math.inf
        return 0.0
    return (current - baseline) / abs(baseline)


def _metrics_are_finite(metrics: dict[str, Any]) -> bool:
    for value in metrics.values():
        if isinstance(value, int | bool | str):
            continue
        if isinstance(value, float) and not math.isfinite(value):
            return False
    return True


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _required_finite_float(metrics: dict[str, Any], key: str, context: str) -> float:
    if key not in metrics:
        raise Phase2ArtifactError(f"{context} missing required field: {key}")
    try:
        value = float(metrics[key])
    except (TypeError, ValueError) as exc:
        raise Phase2ArtifactError(f"{context} field must be numeric: {key}") from exc
    if not math.isfinite(value):
        raise Phase2ArtifactError(f"{context} field must be finite: {key}")
    return value


def _required_positive_int(metrics: dict[str, Any], key: str, context: str) -> int:
    if key not in metrics:
        raise Phase2ArtifactError(f"{context} missing required field: {key}")
    try:
        value = int(metrics[key])
    except (TypeError, ValueError) as exc:
        raise Phase2ArtifactError(f"{context} field must be an integer: {key}") from exc
    if value <= 0:
        raise Phase2ArtifactError(f"{context} field must be positive: {key}")
    return value


def _sharpe(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    if len(values) < 2:
        return math.inf if mean > 0.0 else 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    std = math.sqrt(variance)
    if std <= 1e-12:
        return math.inf if mean > 0.0 else 0.0
    return mean / std * math.sqrt(float(len(values)))


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
