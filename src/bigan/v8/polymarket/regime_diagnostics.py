"""Preregistered causal regime and side diagnostics for v8 future gates."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import mean, median
from typing import Any

from bigan.v8.canonical_payload import canonical_payload_sha256

REGIME_DEFINITION_SCHEMA_VERSION = "bigan-v8-regime-definition-contract-v1"
REGIME_ASSIGNMENT_SCHEMA_VERSION = "bigan-v8-regime-assignment-v1"
REGIME_REPORT_SCHEMA_VERSION = "bigan-v8-regime-stratified-pnl-report-v1"
DIMENSION_BUCKETS = {
    "selected_side": ("UP", "DOWN", "NONE"),
    "market_direction_bucket": ("negative", "flat", "positive", "unknown"),
    "regime": ("bearish", "sideways", "bullish", "unknown"),
    "realized_volatility_bucket": ("low", "medium", "high", "unknown"),
    "spread_liquidity_bucket": (
        "tight_high",
        "tight_medium",
        "tight_low",
        "normal_high",
        "normal_medium",
        "normal_low",
        "wide_high",
        "wide_medium",
        "wide_low",
        "unknown",
    ),
    "time_of_day_bucket": ("utc_00_05", "utc_06_11", "utc_12_17", "utc_18_23"),
    "provider_health_bucket": (
        "healthy",
        "degraded",
        "unhealthy",
        "timeout",
        "truncated",
        "censored",
        "backfill",
        "incomplete",
        "unknown",
    ),
    "decision_origin": ("primary", "fallback", "abstention", "unknown"),
    "action_family": (
        "HOLD_TO_SETTLEMENT",
        "SELL_BEFORE_CLOSE",
        "NO_TRADE",
        "UNKNOWN",
    ),
}


class RegimeDiagnosticError(ValueError):
    """Raised when a regime assignment is not causal or contract-compliant."""


def validate_regime_definition_contract(contract: dict[str, Any]) -> None:
    blockers: list[str] = []
    if contract.get("schema_version") != REGIME_DEFINITION_SCHEMA_VERSION:
        blockers.append("schema_version")
    if contract.get("classification_inputs_available_by_decision_time") is not True:
        blockers.append("causal_inputs")
    if contract.get("diagnostic_only") is not True:
        blockers.append("diagnostic_only")
    if contract.get("fixed_side_quota_enabled") is not False:
        blockers.append("side_quota")
    if contract.get("post_outcome_boundary_change_allowed") is not False:
        blockers.append("post_outcome_boundary_change")
    if any(contract.get("safety", {}).values()):
        blockers.append("safety")
    if blockers:
        raise RegimeDiagnosticError(
            "regime definition contract invalid: " + ", ".join(sorted(blockers))
        )


def assign_regime(
    *,
    decision: dict[str, Any],
    causal_context: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Assign immutable buckets using only decision-time available context."""

    validate_regime_definition_contract(contract)
    decision_ts = int(decision["decision_ts"])
    available_at_ts = int(causal_context["available_at_ts"])
    max_input_ts = int(causal_context["max_input_ts"])
    if available_at_ts > decision_ts or max_input_ts > decision_ts:
        raise RegimeDiagnosticError("regime input is not available by decision_ts")
    forbidden = {
        key
        for key in causal_context
        if any(
            token in str(key).lower()
            for token in ("outcome", "settlement", "pnl", "profit", "future_return", "label")
        )
    }
    if forbidden:
        raise RegimeDiagnosticError(
            "target fields cannot classify regime: " + ", ".join(sorted(forbidden))
        )
    boundaries = contract["boundaries"]
    reference_return = _optional_finite(causal_context.get("reference_return"))
    realized_volatility = _optional_finite(
        causal_context.get("realized_volatility")
    )
    spread_bps = _optional_finite(causal_context.get("combined_spread_bps"))
    liquidity = _optional_finite(causal_context.get("liquidity_depth"))
    assignment = {
        "schema_version": REGIME_ASSIGNMENT_SCHEMA_VERSION,
        "market_id": str(decision["market_id"]),
        "decision_ts": decision_ts,
        "available_at_ts": available_at_ts,
        "max_input_ts": max_input_ts,
        "selected_side": _selected_side(decision),
        "market_direction_bucket": _direction_bucket(
            reference_return,
            flat_abs_max=float(boundaries["market_direction_flat_abs_max"]),
        ),
        "regime": _regime_bucket(
            reference_return,
            bearish_max=float(boundaries["bearish_return_max"]),
            bullish_min=float(boundaries["bullish_return_min"]),
        ),
        "realized_volatility_bucket": _volatility_bucket(
            realized_volatility,
            low_max=float(boundaries["volatility_low_max_exclusive"]),
            medium_max=float(boundaries["volatility_medium_max_exclusive"]),
        ),
        "spread_liquidity_bucket": _spread_liquidity_bucket(
            spread_bps,
            liquidity,
            tight_max=float(boundaries["spread_tight_max_bps"]),
            normal_max=float(boundaries["spread_normal_max_bps"]),
            low_liquidity_max=float(boundaries["liquidity_low_max_exclusive"]),
            medium_liquidity_max=float(
                boundaries["liquidity_medium_max_exclusive"]
            ),
        ),
        "time_of_day_bucket": _time_bucket(decision_ts),
        "provider_health_bucket": _provider_health_bucket(causal_context),
        "decision_origin": _decision_origin(decision),
        "action_family": _action_family(decision),
        "classification_contract_sha256": canonical_payload_sha256(
            contract,
            payload_schema_version=REGIME_DEFINITION_SCHEMA_VERSION,
        ),
        "outcome_settlement_pnl_or_future_information_used": False,
    }
    assignment["regime_assignment_sha256"] = canonical_payload_sha256(
        assignment,
        payload_schema_version=REGIME_ASSIGNMENT_SCHEMA_VERSION,
    )
    return assignment


def build_regime_stratified_diagnostics(
    *,
    assignments: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Report complete/empty strata and reconcile every partition to aggregate PnL."""

    validate_regime_definition_contract(contract)
    assignment_by_key = {_key(row): row for row in assignments}
    candidate_by_key = {_key(row): row for row in candidate_rows}
    baseline_by_key = {_key(row): row for row in baseline_rows}
    if (
        len(assignment_by_key) != len(assignments)
        or len(candidate_by_key) != len(candidate_rows)
        or len(baseline_by_key) != len(baseline_rows)
        or set(assignment_by_key) != set(candidate_by_key)
        or set(candidate_by_key) != set(baseline_by_key)
    ):
        raise RegimeDiagnosticError("assignment/candidate/baseline decision grids differ")
    joined = []
    for key in sorted(candidate_by_key):
        assignment = assignment_by_key[key]
        candidate = candidate_by_key[key]
        baseline = baseline_by_key[key]
        joined.append(
            {
                **assignment,
                "after_cost_pnl": float(candidate["after_cost_pnl"]),
                "baseline_after_cost_pnl": float(baseline["after_cost_pnl"]),
                "candidate_minus_baseline": float(candidate["after_cost_pnl"])
                - float(baseline["after_cost_pnl"]),
                "accepted_bet": bool(candidate.get("execution_guard_order_allowed"))
                and candidate.get("executed_action") != "NO_TRADE",
            }
        )
    reporting = contract["reporting"]
    aggregate = _stratum_metrics(
        joined,
        minimum_support=int(reporting["minimum_supported_stratum_count"]),
        bootstrap_resample_count=int(reporting["bootstrap_resample_count"]),
        bootstrap_seed=int(reporting["bootstrap_seed"]),
        confidence_level=float(reporting["confidence_level"]),
        aggregate_total=None,
    )
    dimension_reports: dict[str, dict[str, dict[str, Any]]] = {}
    reconciliation: dict[str, bool] = {}
    aggregate_total = float(aggregate["after_cost_pnl"])
    for dimension, buckets in DIMENSION_BUCKETS.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in joined:
            grouped[str(row[dimension])].append(row)
        report = {
            bucket: _stratum_metrics(
                grouped.get(bucket, []),
                minimum_support=int(reporting["minimum_supported_stratum_count"]),
                bootstrap_resample_count=int(reporting["bootstrap_resample_count"]),
                bootstrap_seed=int(reporting["bootstrap_seed"])
                + list(DIMENSION_BUCKETS).index(dimension) * 100
                + index,
                confidence_level=float(reporting["confidence_level"]),
                aggregate_total=aggregate_total,
            )
            for index, bucket in enumerate(buckets)
        }
        dimension_reports[dimension] = report
        reconciliation[dimension] = (
            math.isclose(
                math.fsum(
                    float(metrics["after_cost_pnl"]) for metrics in report.values()
                ),
                aggregate_total,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and sum(int(metrics["support"]) for metrics in report.values())
            == len(joined)
        )
    joint_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        joint_key = "|".join(str(row[dimension]) for dimension in DIMENSION_BUCKETS)
        joint_groups[joint_key].append(row)
    joint_report = {
        key: _stratum_metrics(
            rows,
            minimum_support=int(reporting["minimum_supported_stratum_count"]),
            bootstrap_resample_count=int(reporting["bootstrap_resample_count"]),
            bootstrap_seed=int(reporting["bootstrap_seed"]) + 1000 + index,
            confidence_level=float(reporting["confidence_level"]),
            aggregate_total=aggregate_total,
        )
        for index, (key, rows) in enumerate(sorted(joint_groups.items()))
    }
    report = {
        "schema_version": REGIME_REPORT_SCHEMA_VERSION,
        "aggregate": aggregate,
        "dimensions": dimension_reports,
        "joint_strata": joint_report,
        "reconciliation": reconciliation,
        "all_dimension_partitions_reconcile": all(reconciliation.values()),
        "diagnostic_only": True,
        "aggregate_hard_gate_changed": False,
        "stratified_metrics_are_eligibility_blockers": False,
        "fixed_side_quota_enabled": False,
        "paper_candidate_unlocked": False,
        "promotion_unlocked": False,
        "live_unlocked": False,
        "capital_at_risk": False,
    }
    report["report_sha256"] = canonical_payload_sha256(
        report,
        payload_schema_version=REGIME_REPORT_SCHEMA_VERSION,
    )
    return {
        "regime_assignment": assignments,
        "regime_stratified_pnl_report": report,
        "regime_bootstrap_report": {
            "schema_version": "bigan-v8-regime-bootstrap-report-v1",
            "confidence_level": reporting["confidence_level"],
            "bootstrap_unit": "market_id",
            "aggregate_candidate_minus_baseline_lcb": aggregate[
                "candidate_minus_baseline_bootstrap_lcb"
            ],
            "aggregate_candidate_minus_baseline_ucb": aggregate[
                "candidate_minus_baseline_bootstrap_ucb"
            ],
            "dimension_intervals": {
                dimension: {
                    bucket: {
                        "status": metrics["status"],
                        "lcb": metrics["candidate_minus_baseline_bootstrap_lcb"],
                        "ucb": metrics["candidate_minus_baseline_bootstrap_ucb"],
                    }
                    for bucket, metrics in strata.items()
                }
                for dimension, strata in dimension_reports.items()
            },
        },
        "side_action_attribution_report": {
            "schema_version": "bigan-v8-side-action-attribution-v1",
            "selected_side": dimension_reports["selected_side"],
            "action_family": dimension_reports["action_family"],
            "decision_origin": dimension_reports["decision_origin"],
            "diagnostic_only": True,
        },
    }


def regime_diagnostics_markdown(artifacts: dict[str, Any]) -> str:
    report = artifacts["regime_stratified_pnl_report"]
    aggregate = report["aggregate"]
    lines = [
        "# Regime-Stratified Future-Gate Diagnostics",
        "",
        f"- support: `{aggregate['support']}`",
        f"- after-cost PnL: `{aggregate['after_cost_pnl']}`",
        (
            "- candidate-minus-baseline bootstrap interval: "
            f"`[{aggregate['candidate_minus_baseline_bootstrap_lcb']}, "
            f"{aggregate['candidate_minus_baseline_bootstrap_ucb']}]`"
        ),
        f"- all partitions reconcile: `{report['all_dimension_partitions_reconcile']}`",
        "- diagnostic only: `true`",
        "- aggregate hard gate changed: `false`",
        "- promotion unlocked: `false`",
    ]
    return "\n".join(lines) + "\n"


def _stratum_metrics(
    rows: list[dict[str, Any]],
    *,
    minimum_support: int,
    bootstrap_resample_count: int,
    bootstrap_seed: int,
    confidence_level: float,
    aggregate_total: float | None,
) -> dict[str, Any]:
    support = len(rows)
    candidate_values = [float(row["after_cost_pnl"]) for row in rows]
    deltas = [float(row["candidate_minus_baseline"]) for row in rows]
    if not rows:
        status = "empty"
    elif support < minimum_support:
        status = "insufficient_support"
    else:
        status = "reported"
    if rows:
        delta_by_market: dict[str, float] = defaultdict(float)
        for row in rows:
            delta_by_market[str(row["market_id"])] += float(
                row["candidate_minus_baseline"]
            )
        lcb, ucb = _bootstrap_interval(
            dict(delta_by_market),
            confidence_level=confidence_level,
            resample_count=bootstrap_resample_count,
            seed=bootstrap_seed,
        )
    else:
        lcb = None
        ucb = None
    positive = sum(1 for value in candidate_values if value > 0)
    negative = sum(1 for value in candidate_values if value < 0)
    absolute_total = sum(abs(value) for value in candidate_values)
    pnl_total = math.fsum(candidate_values)
    return {
        "status": status,
        "support": support,
        "unique_market_count": len({str(row["market_id"]) for row in rows}),
        "after_cost_pnl": pnl_total,
        "mean_return": mean(candidate_values) if rows else None,
        "median_return": median(candidate_values) if rows else None,
        "candidate_minus_baseline_delta": math.fsum(deltas),
        "candidate_minus_baseline_bootstrap_lcb": lcb,
        "candidate_minus_baseline_bootstrap_ucb": ucb,
        "largest_winner_removed_after_cost_pnl": (
            math.fsum(
                (
                    pnl_total,
                    -max([value for value in candidate_values if value > 0], default=0.0),
                )
            )
        ),
        "fallback_share": (
            sum(1 for row in rows if row["decision_origin"] == "fallback") / support
            if support
            else None
        ),
        "win_count": positive,
        "loss_count": negative,
        "max_absolute_pnl_concentration": (
            max((abs(value) for value in candidate_values), default=0.0) / absolute_total
            if absolute_total
            else 0.0
        ),
        "contribution_to_total_pnl": (
            pnl_total / aggregate_total if aggregate_total not in {None, 0.0} else None
        ),
        "uncertainty_available": bool(rows),
    }


def _bootstrap_interval(
    values_by_market: dict[str, float],
    *,
    confidence_level: float,
    resample_count: int,
    seed: int,
) -> tuple[float, float]:
    values = list(values_by_market.values())
    rng = random.Random(seed)
    samples = sorted(
        sum(rng.choice(values) for _ in values) for _ in range(resample_count)
    )
    alpha = (1.0 - confidence_level) / 2.0
    lower = max(0, min(len(samples) - 1, int(alpha * len(samples))))
    upper = max(
        0,
        min(len(samples) - 1, int((1.0 - alpha) * len(samples)) - 1),
    )
    return samples[lower], samples[upper]


def _optional_finite(value: Any) -> float | None:
    if value is None:
        return None
    converted = float(value)
    if converted != converted or converted in {float("inf"), float("-inf")}:
        raise RegimeDiagnosticError("regime numeric input must be finite")
    return converted


def _direction_bucket(value: float | None, *, flat_abs_max: float) -> str:
    if value is None:
        return "unknown"
    if value < -flat_abs_max:
        return "negative"
    if value > flat_abs_max:
        return "positive"
    return "flat"


def _regime_bucket(
    value: float | None,
    *,
    bearish_max: float,
    bullish_min: float,
) -> str:
    if value is None:
        return "unknown"
    if value <= bearish_max:
        return "bearish"
    if value >= bullish_min:
        return "bullish"
    return "sideways"


def _volatility_bucket(
    value: float | None,
    *,
    low_max: float,
    medium_max: float,
) -> str:
    if value is None:
        return "unknown"
    if value < low_max:
        return "low"
    if value < medium_max:
        return "medium"
    return "high"


def _spread_liquidity_bucket(
    spread_bps: float | None,
    liquidity: float | None,
    *,
    tight_max: float,
    normal_max: float,
    low_liquidity_max: float,
    medium_liquidity_max: float,
) -> str:
    if spread_bps is None or liquidity is None:
        return "unknown"
    spread = "tight" if spread_bps <= tight_max else "normal" if spread_bps <= normal_max else "wide"
    depth = (
        "low"
        if liquidity < low_liquidity_max
        else "medium"
        if liquidity < medium_liquidity_max
        else "high"
    )
    return f"{spread}_{depth}"


def _time_bucket(decision_ts: int) -> str:
    hour = (decision_ts // 3_600_000) % 24
    if hour < 6:
        return "utc_00_05"
    if hour < 12:
        return "utc_06_11"
    if hour < 18:
        return "utc_12_17"
    return "utc_18_23"


def _provider_health_bucket(context: dict[str, Any]) -> str:
    if context.get("trade_tape_provider_timeout") == 1:
        return "timeout"
    if context.get("trade_tape_truncated") == 1:
        return "truncated"
    if context.get("trade_tape_censored") == 1:
        return "censored"
    if context.get("trade_tape_historical_backfill") == 1:
        return "backfill"
    if context.get("provider_coverage_complete") == 0:
        return "incomplete"
    score = _optional_finite(context.get("provider_health_score"))
    if score is None:
        return "unknown"
    if score >= 0.9:
        return "healthy"
    if score >= 0.5:
        return "degraded"
    return "unhealthy"


def _selected_side(decision: dict[str, Any]) -> str:
    side = str(decision.get("selected_side") or "NONE").upper()
    return side if side in {"UP", "DOWN", "NONE"} else "NONE"


def _decision_origin(decision: dict[str, Any]) -> str:
    if decision.get("executed_action") == "NO_TRADE":
        return "abstention"
    origin = str(decision.get("decision_origin") or "").lower()
    if "fallback" in origin:
        return "fallback"
    return "primary" if origin else "unknown"


def _action_family(decision: dict[str, Any]) -> str:
    action = str(decision.get("executed_action") or "")
    if action == "NO_TRADE":
        return "NO_TRADE"
    if "SELL_BEFORE_CLOSE" in action:
        return "SELL_BEFORE_CLOSE"
    if "HOLD_TO_SETTLEMENT" in action:
        return "HOLD_TO_SETTLEMENT"
    return "UNKNOWN"


def _key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["market_id"]), int(row["decision_ts"])


__all__ = [
    "DIMENSION_BUCKETS",
    "REGIME_ASSIGNMENT_SCHEMA_VERSION",
    "REGIME_DEFINITION_SCHEMA_VERSION",
    "REGIME_REPORT_SCHEMA_VERSION",
    "RegimeDiagnosticError",
    "assign_regime",
    "build_regime_stratified_diagnostics",
    "regime_diagnostics_markdown",
    "validate_regime_definition_contract",
]
