"""Phase 3 differentiable PnL optimization runner."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bigan.v8.phase1.contracts import (
    PolicyPrediction,
    PolicyTrainingExample,
    PolicyTrainShadowSplit,
)
from bigan.v8.phase2 import (
    PHASE2_EVALUATION_PHASE,
    Phase2EvaluationConfig,
    Phase15CandidateArtifact,
    build_phase2_report,
    load_phase15_candidate,
    simulate_execution,
)
from bigan.v8.phase2.contracts import ExecutionSimulationConfig
from bigan.v8.phase3.contracts import (
    PHASE3_DIFFERENTIABLE_PNL_PHASE,
    PHASE3_PARAMETER_NAMES,
    DifferentiableExecutionConfig,
    DifferentiablePnlOptimizationConfig,
    Phase3OptimizationError,
    Phase3OptimizationReport,
)


@dataclass(frozen=True, slots=True)
class Phase3OptimizationResult:
    """Result of a Phase 3 optimization run."""

    candidate: Phase15CandidateArtifact
    split: PolicyTrainShadowSplit
    train_predictions: tuple[PolicyPrediction, ...]
    oos_predictions: tuple[PolicyPrediction, ...]
    report: Phase3OptimizationReport
    report_path: Path | None = None

    @property
    def passed(self) -> bool:
        return self.report.passed


def run_phase3_optimization(
    candidate_artifact_dir: Path | str,
    split: PolicyTrainShadowSplit,
    config: DifferentiablePnlOptimizationConfig | None = None,
    *,
    phase2_report_path: Path | str | None = None,
) -> Phase3OptimizationResult:
    """Optimize a differentiable cost-aware action head over a Phase 1.5 policy."""

    resolved_config = config or DifferentiablePnlOptimizationConfig()
    candidate = load_phase15_candidate(candidate_artifact_dir)
    _assert_split_matches_candidate(candidate, split)
    expected_phase2_execution_config = _phase2_baseline_execution_config(resolved_config)
    frozen_phase2_baseline = (
        _load_frozen_phase2_baseline(
            phase2_report_path=phase2_report_path,
            candidate=candidate,
            expected_execution_config=expected_phase2_execution_config,
        )
        if phase2_report_path is not None
        else None
    )
    train_predictions = candidate.model.predict_examples(split.train_examples)
    oos_predictions = candidate.model.predict_examples(split.shadow_examples)
    if len(train_predictions) != len(split.train_examples):
        raise Phase3OptimizationError("train prediction count mismatch")
    if len(oos_predictions) != len(split.shadow_examples):
        raise Phase3OptimizationError("OOS prediction count mismatch")

    phase2_baseline = frozen_phase2_baseline or _diagnostic_phase2_baseline(
        candidate=candidate,
        examples=split.shadow_examples,
        predictions=oos_predictions,
        config=resolved_config,
    )

    optimized_parameters, optimization_trace = _optimize_parameters(
        examples=split.train_examples,
        predictions=train_predictions,
        config=resolved_config,
    )
    train_eval = _evaluate_differentiable_policy(
        examples=split.train_examples,
        predictions=train_predictions,
        parameters=optimized_parameters,
        execution_config=resolved_config.execution_config,
        return_variance_penalty=resolved_config.return_variance_penalty,
    )
    oos_eval = _evaluate_differentiable_policy(
        examples=split.shadow_examples,
        predictions=oos_predictions,
        parameters=optimized_parameters,
        execution_config=resolved_config.execution_config,
        return_variance_penalty=resolved_config.return_variance_penalty,
    )
    stress_metrics = _cost_stress_metrics(
        examples=split.shadow_examples,
        predictions=oos_predictions,
        parameters=optimized_parameters,
        config=resolved_config,
    )
    comparison_metrics = _comparison_metrics(
        phase2_metrics=phase2_baseline.metrics,
        phase3_metrics=oos_eval.metrics,
        stress_metrics=stress_metrics,
    )
    report = Phase3OptimizationReport(
        phase=PHASE3_DIFFERENTIABLE_PNL_PHASE,
        candidate_run_id=candidate.run_id,
        candidate_artifact_dir=str(candidate.artifact_dir),
        phase1_5_hashes=candidate.phase1_5_hashes(),
        phase2_report_path=phase2_baseline.report_path,
        phase2_report_sha256=phase2_baseline.report_sha256,
        phase2_baseline_source=phase2_baseline.source,
        phase2_execution_config_sha256=phase2_baseline.execution_config_sha256,
        phase2_execution_config_verified=phase2_baseline.execution_config_verified,
        phase2_baseline_metrics=phase2_baseline.metrics,
        train_metrics=train_eval.metrics,
        oos_metrics=oos_eval.metrics,
        comparison_metrics=comparison_metrics,
        cost_stress_metrics=stress_metrics,
        optimization_trace=optimization_trace,
        optimized_parameters=_parameters_to_dict(optimized_parameters),
        acceptance_criteria=_acceptance_criteria(
            optimization_trace=optimization_trace,
            phase2_baseline_source=phase2_baseline.source,
            phase2_execution_config_verified=phase2_baseline.execution_config_verified,
            phase2_metrics=phase2_baseline.metrics,
            phase3_train_metrics=train_eval.metrics,
            phase3_oos_metrics=oos_eval.metrics,
            comparison_metrics=comparison_metrics,
            stress_metrics=stress_metrics,
            config=resolved_config,
        ),
        config=resolved_config.to_dict(),
        created_at=resolved_config.created_at,
    )
    report_path = None
    if resolved_config.output_dir is not None:
        output_dir = Path(resolved_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"{candidate.run_id}_phase3_report.json"
        report_path.write_text(
            json.dumps(_json_ready(report.to_dict()), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
    return Phase3OptimizationResult(
        candidate=candidate,
        split=split,
        train_predictions=train_predictions,
        oos_predictions=oos_predictions,
        report=report,
        report_path=report_path,
    )


@dataclass(frozen=True, slots=True)
class _DifferentiableEvaluation:
    metrics: dict[str, Any]
    loss: float


@dataclass(frozen=True, slots=True)
class _Phase2Baseline:
    metrics: dict[str, Any]
    source: str
    report_path: str | None
    report_sha256: str | None
    execution_config_sha256: str
    execution_config_verified: bool


def _load_frozen_phase2_baseline(
    *,
    phase2_report_path: Path | str,
    candidate: Phase15CandidateArtifact,
    expected_execution_config: ExecutionSimulationConfig,
) -> _Phase2Baseline:
    path = Path(phase2_report_path)
    payload = _read_json(path)
    if payload.get("phase") != PHASE2_EVALUATION_PHASE:
        raise Phase3OptimizationError("Phase 2 report phase mismatch")
    if payload.get("passed") is not True:
        raise Phase3OptimizationError("Phase 3 requires a passed frozen Phase 2 report")
    if payload.get("candidate_run_id") != candidate.run_id:
        raise Phase3OptimizationError("Phase 2 report candidate_run_id mismatch")
    report_hashes = payload.get("phase1_5_hashes")
    if not isinstance(report_hashes, dict):
        raise Phase3OptimizationError("Phase 2 report phase1_5_hashes are required")
    candidate_hashes = candidate.phase1_5_hashes()
    for field_name in (
        "policy_dataset_hash",
        "split_hash",
        "model_sha256",
    ):
        if report_hashes.get(field_name) != candidate_hashes.get(field_name):
            raise Phase3OptimizationError(f"Phase 2 report {field_name} mismatch")
    metrics = payload.get("execution_metrics")
    if not isinstance(metrics, dict) or int(metrics.get("row_count", 0)) <= 0:
        raise Phase3OptimizationError("Phase 2 report execution_metrics are required")
    report_config = payload.get("config")
    if not isinstance(report_config, dict):
        raise Phase3OptimizationError("Phase 2 report config is required")
    actual_execution_config = report_config.get("execution_config")
    if not isinstance(actual_execution_config, dict):
        raise Phase3OptimizationError("Phase 2 report config.execution_config is required")
    expected_execution_config_hash = _canonical_payload_sha256(
        expected_execution_config.to_dict()
    )
    actual_execution_config_hash = _canonical_payload_sha256(actual_execution_config)
    if actual_execution_config_hash != expected_execution_config_hash:
        raise Phase3OptimizationError("Phase 2 report execution_config mismatch")
    return _Phase2Baseline(
        metrics=metrics,
        source="frozen_phase2_report",
        report_path=str(path),
        report_sha256=_sha256_file(path),
        execution_config_sha256=actual_execution_config_hash,
        execution_config_verified=True,
    )


def _diagnostic_phase2_baseline(
    *,
    candidate: Phase15CandidateArtifact,
    examples: tuple[PolicyTrainingExample, ...],
    predictions: tuple[PolicyPrediction, ...],
    config: DifferentiablePnlOptimizationConfig,
) -> _Phase2Baseline:
    baseline_execution_config = _phase2_baseline_execution_config(config)
    phase2_fills = simulate_execution(
        examples=examples,
        predictions=predictions,
        config=baseline_execution_config,
    )
    phase2_report = build_phase2_report(
        candidate=candidate,
        fills=phase2_fills,
        config=Phase2EvaluationConfig(
            execution_config=baseline_execution_config,
            min_sharpe_improvement_ratio=-1_000_000_000.0,
            min_turnover_reduction_ratio=-1_000_000_000.0,
            created_at=config.created_at,
        ),
    )
    return _Phase2Baseline(
        metrics=phase2_report.execution_metrics,
        source="diagnostic_recomputed_phase2_baseline",
        report_path=None,
        report_sha256=None,
        execution_config_sha256=_canonical_payload_sha256(
            baseline_execution_config.to_dict()
        ),
        execution_config_verified=False,
    )


def _optimize_parameters(
    *,
    examples: tuple[PolicyTrainingExample, ...],
    predictions: tuple[PolicyPrediction, ...],
    config: DifferentiablePnlOptimizationConfig,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    params = np.asarray(config.initial_parameters, dtype=np.float64)
    trace: list[dict[str, float | int]] = []
    for step in range(config.max_steps + 1):
        evaluation = _evaluate_differentiable_policy(
            examples=examples,
            predictions=predictions,
            parameters=params,
            execution_config=config.execution_config,
            return_variance_penalty=config.return_variance_penalty,
        )
        gradient = _finite_difference_gradient(
            examples=examples,
            predictions=predictions,
            parameters=params,
            config=config,
        )
        gradient_norm = float(np.linalg.norm(gradient))
        trace.append(
            {
                "step": step,
                "loss": evaluation.loss,
                "mean_net_return": float(evaluation.metrics["mean_net_return"]),
                "differentiable_sharpe": float(evaluation.metrics["differentiable_sharpe"]),
                "gradient_norm": gradient_norm,
            }
        )
        if step == config.max_steps:
            break
        if not np.all(np.isfinite(gradient)):
            raise Phase3OptimizationError("non-finite differentiable PnL gradient")
        params = params - config.learning_rate * gradient
        params = np.clip(params, -config.max_abs_parameter, config.max_abs_parameter)
    return params, trace


def _finite_difference_gradient(
    *,
    examples: tuple[PolicyTrainingExample, ...],
    predictions: tuple[PolicyPrediction, ...],
    parameters: np.ndarray,
    config: DifferentiablePnlOptimizationConfig,
) -> np.ndarray:
    gradient = np.zeros_like(parameters)
    epsilon = config.finite_difference_epsilon
    for idx in range(parameters.size):
        offset = np.zeros_like(parameters)
        offset[idx] = epsilon
        plus = _evaluate_differentiable_policy(
            examples=examples,
            predictions=predictions,
            parameters=parameters + offset,
            execution_config=config.execution_config,
            return_variance_penalty=config.return_variance_penalty,
        ).loss
        minus = _evaluate_differentiable_policy(
            examples=examples,
            predictions=predictions,
            parameters=parameters - offset,
            execution_config=config.execution_config,
            return_variance_penalty=config.return_variance_penalty,
        ).loss
        gradient[idx] = (plus - minus) / (2.0 * epsilon)
    return gradient


def _evaluate_differentiable_policy(
    *,
    examples: tuple[PolicyTrainingExample, ...],
    predictions: tuple[PolicyPrediction, ...],
    parameters: np.ndarray,
    execution_config: DifferentiableExecutionConfig,
    return_variance_penalty: float,
) -> _DifferentiableEvaluation:
    if len(examples) != len(predictions):
        raise Phase3OptimizationError("examples and predictions must have the same length")
    previous_action_by_key: dict[tuple[str, str], float] = {}
    net_returns: list[float] = []
    gross_returns: list[float] = []
    costs: list[float] = []
    risks: list[float] = []
    turnover_penalties: list[float] = []
    turnovers: list[float] = []
    actions: list[float] = []
    fill_probabilities: list[float] = []
    for example, prediction in zip(examples, predictions, strict=True):
        _assert_prediction_matches_example(example, prediction)
        key = (example.source, example.instrument_id)
        previous_action = previous_action_by_key.get(key, 0.0)
        raw_action = _differentiable_action(prediction, parameters, execution_config)
        fill_probability = _smooth_fill_probability(example, raw_action, execution_config)
        filled_action = raw_action * fill_probability
        turnover = _smooth_abs(
            filled_action - previous_action,
            execution_config.smooth_abs_epsilon,
        )
        total_cost = _differentiable_execution_cost(
            example=example,
            turnover=turnover,
            config=execution_config,
        )
        risk_penalty = _risk_penalty(filled_action, example, execution_config)
        turnover_penalty = execution_config.turnover_penalty_factor * turnover
        gross_return = filled_action * example.shadow_net_return
        net_return = gross_return - total_cost - risk_penalty - turnover_penalty
        net_returns.append(net_return)
        gross_returns.append(gross_return)
        costs.append(total_cost)
        risks.append(risk_penalty)
        turnover_penalties.append(turnover_penalty)
        turnovers.append(turnover)
        actions.append(filled_action)
        fill_probabilities.append(fill_probability)
        previous_action_by_key[key] = filled_action

    mean_net_return = _mean(net_returns)
    variance = _variance(net_returns)
    loss = -mean_net_return + return_variance_penalty * variance
    metrics = {
        "row_count": len(examples),
        "loss": loss,
        "mean_gross_policy_return": _mean(gross_returns),
        "mean_execution_cost": _mean(costs),
        "mean_risk_penalty": _mean(risks),
        "mean_turnover_penalty": _mean(turnover_penalties),
        "mean_net_return": mean_net_return,
        "return_variance": variance,
        "differentiable_sharpe": _smooth_sharpe(net_returns),
        "total_turnover": sum(turnovers),
        "mean_abs_turnover": _mean(turnovers),
        "active_rate": (
            sum(1 for action in actions if action > execution_config.active_action_epsilon)
            / len(actions)
            if actions
            else 0.0
        ),
        "mean_filled_action": _mean(actions),
        "mean_fill_probability": _mean(fill_probabilities),
        "cost_to_abs_gross_return_ratio": (
            sum(costs) / max(sum(abs(value) for value in gross_returns), 1e-12)
        ),
    }
    return _DifferentiableEvaluation(metrics=metrics, loss=loss)


def _differentiable_action(
    prediction: PolicyPrediction,
    parameters: np.ndarray,
    config: DifferentiableExecutionConfig,
) -> float:
    score_signal = 2.0 * _bounded_score(prediction.score) - 1.0
    logit = (
        parameters[0]
        + parameters[1] * score_signal
        + parameters[2] * prediction.confidence
        + parameters[3] * prediction.action
    )
    return config.max_position_size * _sigmoid(logit / config.action_temperature)


def _differentiable_execution_cost(
    *,
    example: PolicyTrainingExample,
    turnover: float,
    config: DifferentiableExecutionConfig,
) -> float:
    cost_config = config.cost_model_config
    volatility = _feature_float(example, "volatility_5m") or _feature_float(
        example,
        "volatility_15m",
    )
    liquidity = max(
        cost_config.minimum_liquidity,
        _feature_float(example, "liquidity_depth") or 0.0,
    )
    order_size = max(config.min_order_size, config.order_size * max(turnover, 0.0))
    spread_cost = _spread_fraction(example)
    fee_cost = cost_config.fee_bps / 10_000.0
    slippage_cost = (
        cost_config.base_slippage_bps / 10_000.0
        + max(0.0, volatility or 0.0) * cost_config.volatility_slippage_factor
    ) * config.slippage_multiplier
    liquidity_impact_cost = cost_config.liquidity_impact_factor * math.sqrt(
        order_size / liquidity
    )
    return turnover * (
        spread_cost
        + fee_cost
        + slippage_cost
        + liquidity_impact_cost
    )


def _cost_stress_metrics(
    *,
    examples: tuple[PolicyTrainingExample, ...],
    predictions: tuple[PolicyPrediction, ...],
    parameters: np.ndarray,
    config: DifferentiablePnlOptimizationConfig,
) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for multiplier in config.cost_stress_multipliers:
        stressed_execution_config = DifferentiableExecutionConfig(
            cost_model_config=config.execution_config.cost_model_config,
            slippage_multiplier=(
                config.execution_config.slippage_multiplier * multiplier
            ),
            order_size=config.execution_config.order_size,
            min_order_size=config.execution_config.min_order_size,
            min_fill_probability=config.execution_config.min_fill_probability,
            risk_penalty_factor=config.execution_config.risk_penalty_factor,
            turnover_penalty_factor=config.execution_config.turnover_penalty_factor,
            max_position_size=config.execution_config.max_position_size,
            active_action_epsilon=config.execution_config.active_action_epsilon,
            smooth_abs_epsilon=config.execution_config.smooth_abs_epsilon,
            action_temperature=config.execution_config.action_temperature,
        )
        evaluation = _evaluate_differentiable_policy(
            examples=examples,
            predictions=predictions,
            parameters=parameters,
            execution_config=stressed_execution_config,
            return_variance_penalty=config.return_variance_penalty,
        )
        metrics[f"{multiplier:g}"] = evaluation.metrics
    return metrics


def _comparison_metrics(
    *,
    phase2_metrics: dict[str, Any],
    phase3_metrics: dict[str, Any],
    stress_metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    phase2_sharpe = float(phase2_metrics["execution_sharpe"])
    phase3_sharpe = float(phase3_metrics["differentiable_sharpe"])
    stress_sharpes = [
        float(metrics["differentiable_sharpe"])
        for metrics in stress_metrics.values()
    ]
    worst_stress_sharpe = min(stress_sharpes) if stress_sharpes else phase3_sharpe
    return {
        "phase2_execution_sharpe": phase2_sharpe,
        "phase3_oos_differentiable_sharpe": phase3_sharpe,
        "sharpe_delta_over_phase2": phase3_sharpe - phase2_sharpe,
        "sharpe_improvement_ratio_over_phase2": _relative_change(
            phase3_sharpe,
            phase2_sharpe,
        ),
        "phase2_mean_net_execution_return": float(
            phase2_metrics["mean_net_execution_return"]
        ),
        "phase3_oos_mean_net_return": float(phase3_metrics["mean_net_return"]),
        "mean_net_return_delta_over_phase2": (
            float(phase3_metrics["mean_net_return"])
            - float(phase2_metrics["mean_net_execution_return"])
        ),
        "phase2_mean_abs_turnover": float(phase2_metrics["mean_abs_turnover"]),
        "phase3_oos_mean_abs_turnover": float(phase3_metrics["mean_abs_turnover"]),
        "worst_cost_stress_sharpe": worst_stress_sharpe,
        "cost_stress_sharpe_drop_ratio": _stress_drop_ratio(
            phase3_sharpe,
            worst_stress_sharpe,
        ),
    }


def _acceptance_criteria(
    *,
    optimization_trace: list[dict[str, float | int]],
    phase2_baseline_source: str,
    phase2_execution_config_verified: bool,
    phase2_metrics: dict[str, Any],
    phase3_train_metrics: dict[str, Any],
    phase3_oos_metrics: dict[str, Any],
    comparison_metrics: dict[str, Any],
    stress_metrics: dict[str, dict[str, Any]],
    config: DifferentiablePnlOptimizationConfig,
) -> dict[str, bool]:
    initial_loss = float(optimization_trace[0]["loss"])
    final_loss = float(optimization_trace[-1]["loss"])
    gradient_norms = [float(row["gradient_norm"]) for row in optimization_trace]
    return {
        "phase1_5_candidate_verified": True,
        "phase2_baseline_reported": int(phase2_metrics.get("row_count", 0)) > 0,
        "frozen_phase2_report_verified": phase2_baseline_source == "frozen_phase2_report",
        "phase2_execution_config_verified": phase2_execution_config_verified,
        "direct_pnl_optimization": True,
        "gradient_flow_verified": max(gradient_norms) >= config.min_gradient_norm,
        "gradient_norms_finite": all(math.isfinite(value) for value in gradient_norms),
        "gradient_norms_below_limit": max(gradient_norms) <= config.max_gradient_norm,
        "optimization_loss_decreased": (
            initial_loss - final_loss >= config.min_loss_improvement
        ),
        "train_metrics_finite": _metrics_are_finite(phase3_train_metrics),
        "oos_metrics_finite": _metrics_are_finite(phase3_oos_metrics),
        "sharpe_improvement_over_phase2": (
            comparison_metrics["sharpe_improvement_ratio_over_phase2"]
            >= config.min_sharpe_improvement_ratio_over_phase2
        ),
        "stable_oos_performance": (
            phase3_oos_metrics["row_count"] > 0
            and phase3_oos_metrics["differentiable_sharpe"] >= config.min_oos_sharpe
        ),
        "cost_perturbation_robust": (
            bool(stress_metrics)
            and _stress_metrics_are_finite(stress_metrics)
            and comparison_metrics["cost_stress_sharpe_drop_ratio"]
            <= config.max_cost_stress_sharpe_drop_ratio
        ),
    }


def _phase2_baseline_execution_config(
    config: DifferentiablePnlOptimizationConfig,
) -> ExecutionSimulationConfig:
    if config.phase2_baseline_execution_config is not None:
        return config.phase2_baseline_execution_config
    return config.execution_config.to_phase2_execution_config()


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
        raise Phase3OptimizationError(
            "Phase 3 split does not match Phase 1.5 candidate provenance: "
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
        raise Phase3OptimizationError("prediction keys must match policy examples")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Phase3OptimizationError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise Phase3OptimizationError(f"JSON artifact must contain an object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        _json_ready(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _risk_penalty(
    action: float,
    example: PolicyTrainingExample,
    config: DifferentiableExecutionConfig,
) -> float:
    volatility = _feature_float(example, "volatility_5m") or _feature_float(
        example,
        "volatility_15m",
    )
    return config.risk_penalty_factor * action * action * max(0.0, volatility or 0.0)


def _smooth_fill_probability(
    example: PolicyTrainingExample,
    action: float,
    config: DifferentiableExecutionConfig,
) -> float:
    if action <= config.active_action_epsilon:
        return 0.0
    liquidity = max(0.0, _feature_float(example, "liquidity_depth") or 0.0)
    if liquidity <= 0.0:
        return config.min_fill_probability
    requested_size = config.order_size * action
    probability = liquidity / (liquidity + requested_size)
    return max(config.min_fill_probability, min(1.0, probability))


def _smooth_abs(value: float, epsilon: float) -> float:
    return math.sqrt(value * value + epsilon)


def _spread_fraction(example: PolicyTrainingExample) -> float:
    mid_price = _feature_float(example, "mid_price") or 100.0
    spread = _feature_float(example, "spread")
    if spread is not None:
        return max(0.0, spread / max(mid_price, 1e-12))
    spread_bps = _feature_float(example, "spread_bps") or 0.0
    return max(0.0, spread_bps / 10_000.0)


def _feature_float(example: PolicyTrainingExample, name: str) -> float | None:
    value = example.features.get(name)
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _bounded_score(score: float) -> float:
    if not math.isfinite(score):
        return 0.5
    if 0.0 <= score <= 1.0:
        return score
    return _sigmoid(score)


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _smooth_sharpe(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    variance = _variance(values)
    return mean / math.sqrt(variance + 1e-12) * math.sqrt(float(len(values)))


def _relative_change(current: float, baseline: float) -> float:
    if abs(baseline) <= 1e-12:
        if current > 0.0:
            return math.inf
        if current < 0.0:
            return -math.inf
        return 0.0
    return (current - baseline) / abs(baseline)


def _stress_drop_ratio(base_sharpe: float, stressed_sharpe: float) -> float:
    if not math.isfinite(base_sharpe) or not math.isfinite(stressed_sharpe):
        return math.inf
    if base_sharpe <= 1e-12:
        return 0.0 if stressed_sharpe >= base_sharpe else math.inf
    return max(0.0, base_sharpe - stressed_sharpe) / max(abs(base_sharpe), 1e-12)


def _metrics_are_finite(metrics: dict[str, Any]) -> bool:
    for value in metrics.values():
        if isinstance(value, int | bool | str):
            continue
        if isinstance(value, float) and not math.isfinite(value):
            return False
    return True


def _stress_metrics_are_finite(metrics_by_multiplier: dict[str, dict[str, Any]]) -> bool:
    return all(_metrics_are_finite(metrics) for metrics in metrics_by_multiplier.values())


def _parameters_to_dict(parameters: np.ndarray) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(PHASE3_PARAMETER_NAMES, parameters.tolist(), strict=True)
    }


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
