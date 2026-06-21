"""Phase 5 safety-layer runner."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.phase4.contracts import AdaptiveDecision
from bigan.v8.phase5.contracts import (
    PHASE5_SAFETY_LAYER_PHASE,
    LiveExecutionObservation,
    Phase5SafetyError,
    Phase5SafetyLayerReport,
    SafetyAction,
    SafetyLayerConfig,
    ShadowLiveRecord,
    StableModelSnapshot,
)


@dataclass(frozen=True, slots=True)
class Phase5SafetyLayerResult:
    """Result of a Phase 5 safety-layer replay."""

    records: tuple[ShadowLiveRecord, ...]
    report: Phase5SafetyLayerReport
    report_path: Path | None = None

    @property
    def passed(self) -> bool:
        return self.report.passed


def run_phase5_safety_layer(
    *,
    shadow_decisions: tuple[AdaptiveDecision, ...],
    live_observations: tuple[LiveExecutionObservation, ...],
    stable_model: StableModelSnapshot,
    config: SafetyLayerConfig | None = None,
) -> Phase5SafetyLayerResult:
    """Monitor shadow-vs-live degradation and fail closed when needed."""

    resolved_config = config or SafetyLayerConfig()
    _assert_streams_safe(shadow_decisions, live_observations)
    records = _build_shadow_live_records(shadow_decisions, live_observations)
    rolling_diagnostics = _rolling_diagnostics(records, resolved_config)
    drift_metrics = _drift_metrics(
        records=records,
        rolling_diagnostics=rolling_diagnostics,
        config=resolved_config,
    )
    live_risk_metrics = _live_risk_metrics(records, resolved_config)
    safety_action = _safety_action(
        drift_metrics=drift_metrics,
        live_risk_metrics=live_risk_metrics,
        stable_model=stable_model,
        config=resolved_config,
    )
    shadow_mode_metrics = _shadow_mode_metrics(records)
    acceptance_criteria = _acceptance_criteria(
        records=records,
        shadow_mode_metrics=shadow_mode_metrics,
        drift_metrics=drift_metrics,
        live_risk_metrics=live_risk_metrics,
        safety_action=safety_action,
        stable_model=stable_model,
        config=resolved_config,
    )
    report = Phase5SafetyLayerReport(
        phase=PHASE5_SAFETY_LAYER_PHASE,
        row_count=len(records),
        shadow_mode_metrics=shadow_mode_metrics,
        drift_metrics=drift_metrics,
        live_risk_metrics=live_risk_metrics,
        rolling_diagnostics=rolling_diagnostics,
        safety_action=safety_action.to_dict(),
        rollback_snapshot=stable_model.to_dict(),
        acceptance_criteria=acceptance_criteria,
        config=resolved_config.to_dict(),
        created_at=resolved_config.created_at,
    )
    report_path = None
    if resolved_config.output_dir is not None:
        output_dir = Path(resolved_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "phase5_safety_layer_report.json"
        report_path.write_text(
            json.dumps(
                _json_ready(report.to_dict()),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
    return Phase5SafetyLayerResult(
        records=records,
        report=report,
        report_path=report_path,
    )


def _assert_streams_safe(
    shadow_decisions: tuple[AdaptiveDecision, ...],
    live_observations: tuple[LiveExecutionObservation, ...],
) -> None:
    if not shadow_decisions:
        raise Phase5SafetyError("Phase 5 requires at least one shadow decision")
    if len(shadow_decisions) != len(live_observations):
        raise Phase5SafetyError("shadow and live streams must have the same length")
    previous_ts: int | None = None
    for shadow, live in zip(shadow_decisions, live_observations, strict=True):
        if previous_ts is not None and shadow.decision_ts < previous_ts:
            raise Phase5SafetyError(
                "shadow decisions must be ordered by non-decreasing decision_ts"
            )
        previous_ts = shadow.decision_ts
        if (
            shadow.decision_ts != live.decision_ts
            or shadow.source != live.source
            or shadow.instrument_id != live.instrument_id
        ):
            raise Phase5SafetyError("shadow and live keys must match")


def _build_shadow_live_records(
    shadow_decisions: tuple[AdaptiveDecision, ...],
    live_observations: tuple[LiveExecutionObservation, ...],
) -> tuple[ShadowLiveRecord, ...]:
    records: list[ShadowLiveRecord] = []
    for shadow, live in zip(shadow_decisions, live_observations, strict=True):
        records.append(
            ShadowLiveRecord(
                decision_ts=shadow.decision_ts,
                source=shadow.source,
                instrument_id=shadow.instrument_id,
                shadow_net_return=shadow.net_return,
                live_net_return=live.live_net_return,
                shadow_total_execution_cost=shadow.total_execution_cost,
                live_total_execution_cost=live.live_total_execution_cost,
                shadow_regime=shadow.regime,
                live_regime=live.live_regime,
                shadow_filled_action=shadow.filled_action,
                live_filled_action=live.live_filled_action,
                shadow_capital_at_risk=False,
                live_capital_at_risk=live.capital_at_risk,
            )
        )
    return tuple(records)


def _rolling_diagnostics(
    records: tuple[ShadowLiveRecord, ...],
    config: SafetyLayerConfig,
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    window_size = min(config.detection_window_size, len(records))
    for end_index in range(window_size, len(records) + 1):
        window = records[end_index - window_size : end_index]
        metrics = _window_metrics(window, config)
        diagnostics.append(
            {
                "start_ts": window[0].decision_ts,
                "end_ts": window[-1].decision_ts,
                "row_count": len(window),
                **metrics,
                "degradation_detected": _window_degraded(metrics, config),
            }
        )
    return diagnostics


def _window_metrics(
    records: tuple[ShadowLiveRecord, ...],
    config: SafetyLayerConfig,
) -> dict[str, Any]:
    shadow_returns = [record.shadow_net_return for record in records]
    live_returns = [record.live_net_return for record in records]
    shadow_costs = [record.shadow_total_execution_cost for record in records]
    live_costs = [record.live_total_execution_cost for record in records]
    mean_shadow_cost = _mean(shadow_costs)
    mean_live_cost = _mean(live_costs)
    pnl_drift = _mean(
        [
            shadow_return - live_return
            for shadow_return, live_return in zip(shadow_returns, live_returns, strict=True)
        ]
    )
    cost_drift_ratio = (
        mean_live_cost - mean_shadow_cost
    ) / max(abs(mean_shadow_cost), config.cost_drift_floor)
    regime_mismatch_rate = (
        sum(1 for record in records if record.shadow_regime != record.live_regime)
        / len(records)
    )
    shadow_live_correlation = _correlation(shadow_returns, live_returns)
    return {
        "mean_shadow_return": _mean(shadow_returns),
        "mean_live_return": _mean(live_returns),
        "mean_pnl_drift": pnl_drift,
        "abs_mean_pnl_drift": abs(pnl_drift),
        "mean_shadow_cost": mean_shadow_cost,
        "mean_live_cost": mean_live_cost,
        "cost_drift_ratio": cost_drift_ratio,
        "regime_mismatch_rate": regime_mismatch_rate,
        "shadow_live_correlation": shadow_live_correlation,
        "shadow_live_correlation_stable": (
            shadow_live_correlation >= config.min_shadow_live_correlation
        ),
    }


def _drift_metrics(
    *,
    records: tuple[ShadowLiveRecord, ...],
    rolling_diagnostics: list[dict[str, Any]],
    config: SafetyLayerConfig,
) -> dict[str, Any]:
    aggregate = _window_metrics(records, config)
    first_degradation = next(
        (
            diagnostic
            for diagnostic in rolling_diagnostics
            if bool(diagnostic["degradation_detected"])
        ),
        None,
    )
    reason_codes = _reason_codes(aggregate, config)
    return {
        **aggregate,
        "pnl_drift_detected": (
            aggregate["abs_mean_pnl_drift"] > config.max_mean_pnl_drift
        ),
        "cost_drift_detected": (
            aggregate["cost_drift_ratio"] > config.max_cost_drift_ratio
        ),
        "regime_mismatch_detected": (
            aggregate["regime_mismatch_rate"] > config.max_regime_mismatch_rate
        ),
        "correlation_break_detected": (
            aggregate["shadow_live_correlation"] < config.min_shadow_live_correlation
        ),
        "degradation_detected": first_degradation is not None or bool(reason_codes),
        "first_degradation_ts": (
            None if first_degradation is None else int(first_degradation["end_ts"])
        ),
        "reason_codes": reason_codes,
    }


def _live_risk_metrics(
    records: tuple[ShadowLiveRecord, ...],
    config: SafetyLayerConfig,
) -> dict[str, Any]:
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    first_breach_ts: int | None = None
    for record in records:
        cumulative += record.live_net_return
        peak = max(peak, cumulative)
        drawdown = peak - cumulative
        max_drawdown = max(max_drawdown, drawdown)
        if drawdown >= config.max_live_drawdown and first_breach_ts is None:
            first_breach_ts = record.decision_ts
    return {
        "cumulative_live_return": cumulative,
        "max_live_drawdown": max_drawdown,
        "drawdown_breach_detected": first_breach_ts is not None,
        "first_drawdown_breach_ts": first_breach_ts,
    }


def _safety_action(
    *,
    drift_metrics: dict[str, Any],
    live_risk_metrics: dict[str, Any],
    stable_model: StableModelSnapshot,
    config: SafetyLayerConfig,
) -> SafetyAction:
    kill_required = bool(drift_metrics["degradation_detected"]) or bool(
        live_risk_metrics["drawdown_breach_detected"]
    )
    reason_codes = list(drift_metrics["reason_codes"])
    if live_risk_metrics["drawdown_breach_detected"]:
        reason_codes.append("live_drawdown_breach")
    triggered_at_ts = _earliest_ts(
        drift_metrics.get("first_degradation_ts"),
        live_risk_metrics.get("first_drawdown_breach_ts"),
    )
    if not kill_required:
        return SafetyAction(
            kill_switch_triggered=False,
            stop_trading=False,
            flatten_positions=False,
            freeze_model_updates=False,
            rollback_model_id=None,
            rollback_model_sha256=None,
            restored_safe_parameters={},
            reason_codes=(),
            triggered_at_ts=None,
        )
    return SafetyAction(
        kill_switch_triggered=True,
        stop_trading=True,
        flatten_positions=config.flatten_positions_on_kill,
        freeze_model_updates=config.freeze_model_updates_on_kill,
        rollback_model_id=stable_model.model_id,
        rollback_model_sha256=stable_model.model_sha256,
        restored_safe_parameters=dict(stable_model.safe_parameters),
        reason_codes=tuple(sorted(set(reason_codes))),
        triggered_at_ts=triggered_at_ts,
    )


def _shadow_mode_metrics(records: tuple[ShadowLiveRecord, ...]) -> dict[str, Any]:
    return {
        "row_count": len(records),
        "parallel_streams_aligned": True,
        "shadow_mode_enabled": True,
        "shadow_capital_at_risk": any(record.shadow_capital_at_risk for record in records),
        "shadow_capital_risk_free": all(
            not record.shadow_capital_at_risk for record in records
        ),
        "live_capital_at_risk": any(record.live_capital_at_risk for record in records),
        "full_simulation_pipeline": all(
            math.isfinite(record.shadow_net_return)
            and math.isfinite(record.shadow_total_execution_cost)
            and record.shadow_regime
            for record in records
        ),
    }


def _acceptance_criteria(
    *,
    records: tuple[ShadowLiveRecord, ...],
    shadow_mode_metrics: dict[str, Any],
    drift_metrics: dict[str, Any],
    live_risk_metrics: dict[str, Any],
    safety_action: SafetyAction,
    stable_model: StableModelSnapshot,
    config: SafetyLayerConfig,
) -> dict[str, bool]:
    degradation_detected = bool(drift_metrics["degradation_detected"])
    drawdown_breach_detected = bool(live_risk_metrics["drawdown_breach_detected"])
    kill_required = degradation_detected or drawdown_breach_detected
    first_degradation_ts = drift_metrics.get("first_degradation_ts")
    first_drawdown_breach_ts = live_risk_metrics.get("first_drawdown_breach_ts")
    if first_drawdown_breach_ts is None:
        degradation_before_drawdown = True
    elif first_degradation_ts is None:
        degradation_before_drawdown = False
    else:
        degradation_before_drawdown = first_degradation_ts <= first_drawdown_breach_ts
    return {
        "shadow_mode_parallel": bool(shadow_mode_metrics["parallel_streams_aligned"]),
        "shadow_mode_no_capital_risk": bool(
            shadow_mode_metrics["shadow_capital_risk_free"]
        ),
        "full_simulation_pipeline": bool(
            shadow_mode_metrics["full_simulation_pipeline"]
        ),
        "pnl_drift_monitored": math.isfinite(float(drift_metrics["mean_pnl_drift"])),
        "cost_drift_monitored": math.isfinite(float(drift_metrics["cost_drift_ratio"])),
        "regime_mismatch_monitored": 0.0
        <= float(drift_metrics["regime_mismatch_rate"])
        <= 1.0,
        "shadow_live_correlation_stable_or_kill_triggered": (
            drift_metrics["shadow_live_correlation"]
            >= config.min_shadow_live_correlation
            or safety_action.kill_switch_triggered
        ),
        "degradation_detected_before_drawdown": degradation_before_drawdown,
        "kill_switch_reliable": (
            safety_action.kill_switch_triggered == kill_required
            and (not kill_required or safety_action.stop_trading)
            and (
                not kill_required
                or safety_action.freeze_model_updates
                == config.freeze_model_updates_on_kill
            )
            and (
                not kill_required
                or safety_action.flatten_positions == config.flatten_positions_on_kill
            )
        ),
        "rollback_executes_reliably": (
            not kill_required
            or (
                safety_action.rollback_model_id == stable_model.model_id
                and safety_action.rollback_model_sha256 == stable_model.model_sha256
                and safety_action.restored_safe_parameters
                == dict(stable_model.safe_parameters)
            )
        ),
        "safe_when_no_degradation": (
            kill_required
            or (
                not safety_action.kill_switch_triggered
                and not live_risk_metrics["drawdown_breach_detected"]
            )
        ),
        "metrics_finite": _metrics_are_finite(
            {
                "drift_metrics": drift_metrics,
                "live_risk_metrics": live_risk_metrics,
                "shadow_mode_metrics": shadow_mode_metrics,
                "row_count": len(records),
            }
        ),
    }


def _window_degraded(metrics: dict[str, Any], config: SafetyLayerConfig) -> bool:
    return bool(_reason_codes(metrics, config))


def _reason_codes(metrics: dict[str, Any], config: SafetyLayerConfig) -> tuple[str, ...]:
    reasons: list[str] = []
    if metrics["abs_mean_pnl_drift"] > config.max_mean_pnl_drift:
        reasons.append("pnl_drift")
    if metrics["cost_drift_ratio"] > config.max_cost_drift_ratio:
        reasons.append("cost_drift")
    if metrics["regime_mismatch_rate"] > config.max_regime_mismatch_rate:
        reasons.append("regime_mismatch")
    if metrics["shadow_live_correlation"] < config.min_shadow_live_correlation:
        reasons.append("shadow_live_correlation_break")
    return tuple(reasons)


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("correlation inputs must have equal lengths")
    if not left:
        return 0.0
    left_mean = _mean(left)
    right_mean = _mean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    left_norm = math.sqrt(sum(value * value for value in left_centered))
    right_norm = math.sqrt(sum(value * value for value in right_centered))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 1.0 if all(abs(a - b) <= 1e-12 for a, b in zip(left, right, strict=True)) else 0.0
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left_centered, right_centered, strict=True)
    ) / (left_norm * right_norm)


def _earliest_ts(*values: Any) -> int | None:
    numeric = [int(value) for value in values if value is not None]
    return min(numeric) if numeric else None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _metrics_are_finite(value: Any) -> bool:
    if isinstance(value, bool | str) or value is None:
        return True
    if isinstance(value, int | float):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_metrics_are_finite(item) for item in value.values())
    if isinstance(value, list | tuple):
        return all(_metrics_are_finite(item) for item in value)
    return True


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
