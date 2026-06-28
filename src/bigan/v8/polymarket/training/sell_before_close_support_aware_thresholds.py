"""Validation-fitted thresholds for support-aware SELL_BEFORE_CLOSE candidates."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from itertools import product
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.execution_ev import run_polymarket_policy_replay
from bigan.v8.polymarket.training.contracts import (
    PolymarketPolicyPrediction,
    PolymarketPolicyTrainingConfig,
    compact_safety_fields,
)
from bigan.v8.polymarket.training.sell_before_close_exit_reliability import (
    build_sell_before_close_exit_reliability_guard_decisions,
)
from bigan.v8.polymarket.training.sell_before_close_promotion_support import (
    evaluate_sell_before_close_promotion_support,
)
from bigan.v8.polymarket.training.sell_before_close_source_candidates import (
    SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_EXIT_POLICY,
    SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_THRESHOLDS,
    SELL_BEFORE_CLOSE_SUPPORT_AWARE_P_UP_ALIGNED_CANDIDATE_NAME,
    SELL_BEFORE_CLOSE_SUPPORT_AWARE_THRESHOLD_SELECTION_GRID,
)

SELL_BEFORE_CLOSE_SUPPORT_AWARE_THRESHOLD_SELECTION_SCHEMA_VERSION = (
    "bigan-v8-polymarket-sell-before-close-support-aware-threshold-selection-v1"
)
SELL_BEFORE_CLOSE_SUPPORT_AWARE_VALIDATION_GATES = {
    "min_entry_count": 10,
    "min_unique_market_count": 5,
    "min_side_count": 2,
    "min_sell_count": 10,
    "max_residual_count": 0,
    "min_residual_settlement_drag": 0.0,
    "min_total_pnl_exclusive": 0.0,
    "min_mean_pnl_per_entry_exclusive": 0.0,
    "max_p_up_disagreement_rate": 0.50,
}


def build_sell_before_close_support_aware_threshold_selection_report(
    *,
    dataset: Any,
    validation_predictions: tuple[PolymarketPolicyPrediction, ...],
    shadow_predictions: tuple[PolymarketPolicyPrediction, ...],
    config: PolymarketPolicyTrainingConfig,
    calibration_error: float,
    calibration_split: str,
    validation_split: str = "validation",
    shadow_split: str = "shadow",
) -> dict[str, Any]:
    """Fit support-aware guard thresholds on validation and evaluate on shadow."""

    validation_rows = []
    grid_items = list(SELL_BEFORE_CLOSE_SUPPORT_AWARE_THRESHOLD_SELECTION_GRID.items())
    for values in product(*(item[1] for item in grid_items)):
        thresholds = _thresholds(
            {name: float(value) for (name, _), value in zip(grid_items, values, strict=True)}
        )
        validation_rows.append(
            evaluate_support_aware_threshold_row(
                dataset=dataset,
                predictions=validation_predictions,
                config=config,
                calibration_error=calibration_error,
                calibration_split=calibration_split,
                replay_split=validation_split,
                thresholds=thresholds,
                split_name=validation_split,
            )
        )
    passing_rows = [
        row for row in validation_rows if row["validation_support_gate_passed"]
    ]
    selected_validation_row = _select_validation_row(passing_rows)
    selected_thresholds = (
        dict(selected_validation_row["thresholds"]) if selected_validation_row else {}
    )
    shadow_row = None
    reason_codes = []
    if selected_validation_row is None:
        reason_codes.append("support_aware_threshold_selection_failed")
    else:
        shadow_row = evaluate_support_aware_threshold_row(
            dataset=dataset,
            predictions=shadow_predictions,
            config=config,
            calibration_error=calibration_error,
            calibration_split=calibration_split,
            replay_split=shadow_split,
            thresholds=selected_thresholds,
            split_name=shadow_split,
        )
    report = {
        "schema_version": SELL_BEFORE_CLOSE_SUPPORT_AWARE_THRESHOLD_SELECTION_SCHEMA_VERSION,
        "candidate_name": SELL_BEFORE_CLOSE_SUPPORT_AWARE_P_UP_ALIGNED_CANDIDATE_NAME,
        "threshold_selection_method": "validation_fitted_support_aware_thresholds",
        "threshold_selection_fit_split": validation_split,
        "threshold_selection_evaluation_split": shadow_split,
        "uses_shadow_for_fit": False,
        "shadow_sweep_not_used_for_threshold_fit": True,
        "diagnostic_only": False,
        "threshold_grid": SELL_BEFORE_CLOSE_SUPPORT_AWARE_THRESHOLD_SELECTION_GRID,
        "validation_gates": SELL_BEFORE_CLOSE_SUPPORT_AWARE_VALIDATION_GATES,
        "selected_thresholds": selected_thresholds,
        "validation_row_count": len(validation_rows),
        "validation_passing_row_count": len(passing_rows),
        "selected_validation_row": selected_validation_row,
        "shadow_evaluation_row": shadow_row,
        "selection_reason_codes": reason_codes,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    report["sell_before_close_support_aware_threshold_selection_report_id"] = (
        canonical_json_sha256(report)
    )
    return report


def evaluate_support_aware_threshold_row(
    *,
    dataset: Any,
    predictions: tuple[PolymarketPolicyPrediction, ...],
    config: PolymarketPolicyTrainingConfig,
    calibration_error: float,
    calibration_split: str,
    replay_split: str,
    thresholds: dict[str, float],
    split_name: str,
) -> dict[str, Any]:
    """Replay one threshold row and return support-aware gate evidence."""

    replay_config = replace(config, ev_threshold=float(config.ev_threshold))
    decisions, guard_summary = build_sell_before_close_exit_reliability_guard_decisions(
        predictions=predictions,
        config=replay_config,
        thresholds=thresholds,
        exit_policy=SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_EXIT_POLICY,
        candidate_name=SELL_BEFORE_CLOSE_SUPPORT_AWARE_P_UP_ALIGNED_CANDIDATE_NAME,
        p_up_side_alignment_filter_enabled=True,
    )
    prefilter = _decision_support_prefilter(
        decisions=[decision.to_dict() for decision in decisions],
        guard_summary=guard_summary,
    )
    if prefilter["validation_support_gate_reason_codes"]:
        return _threshold_row_payload(
            split_name=split_name,
            thresholds=thresholds,
            guard_summary=guard_summary,
            support_counts=prefilter,
            replay_report=None,
            residual_drag=0.0,
            residual_count=0,
            validation_reason_codes=prefilter[
                "validation_support_gate_reason_codes"
            ],
            support_gate_passed=False,
            support_gate_reason_codes=prefilter[
                "validation_support_gate_reason_codes"
            ],
            promotion_support_eligible=False,
        )
    replay_report = run_polymarket_policy_replay(
        dataset=dataset,
        decisions=decisions,
        config=replay_config,
        calibration_error=calibration_error,
        calibration_split=calibration_split,
        replay_split=replay_split,
        prediction_count=len(predictions),
    )
    support = evaluate_sell_before_close_promotion_support(
        candidate_name=SELL_BEFORE_CLOSE_SUPPORT_AWARE_P_UP_ALIGNED_CANDIDATE_NAME,
        decisions=[decision.to_dict() for decision in decisions],
        replay_report=replay_report,
        exit_reliability_summary=guard_summary,
    )
    residual_drag = min(0.0, float(replay_report["settlement_pnl"]))
    residual_count = int(replay_report["settled_position_count"])
    gate_reasons = _validation_gate_reason_codes(
        entry_count=int(support["entry_decision_count"]),
        unique_market_count=int(support["unique_market_count"]),
        side_count=int(support["side_count"]),
        sell_count=int(support["sell_decision_count"]),
        residual_count=residual_count,
        residual_drag=residual_drag,
        total_pnl=float(replay_report["total_polymarket_pnl"]),
        mean_pnl_per_entry=float(support["mean_pnl_per_entry"]),
        p_up_disagreement_rate=float(
            guard_summary["candidate_scoped_p_up_action_disagreement_rate"]
        ),
    )
    return _threshold_row_payload(
        split_name=split_name,
        thresholds=thresholds,
        guard_summary=guard_summary,
        support_counts=support,
        replay_report=replay_report,
        residual_drag=residual_drag,
        residual_count=residual_count,
        validation_reason_codes=gate_reasons,
        support_gate_passed=bool(support["support_gate_passed"]),
        support_gate_reason_codes=support["support_gate_reason_codes"],
        promotion_support_eligible=bool(support["promotion_support_eligible"]),
    )


def sell_before_close_support_aware_threshold_selection_summary(
    report: dict[str, Any],
) -> dict[str, Any]:
    """Return compact support-aware threshold selection evidence."""

    selected = report.get("selected_validation_row") or {}
    shadow = report.get("shadow_evaluation_row") or {}
    return {
        "schema_version": report["schema_version"],
        "candidate_name": report["candidate_name"],
        "threshold_selection_method": report["threshold_selection_method"],
        "threshold_selection_fit_split": report["threshold_selection_fit_split"],
        "threshold_selection_evaluation_split": report[
            "threshold_selection_evaluation_split"
        ],
        "uses_shadow_for_fit": report["uses_shadow_for_fit"],
        "shadow_sweep_not_used_for_threshold_fit": report[
            "shadow_sweep_not_used_for_threshold_fit"
        ],
        "selected_thresholds": report["selected_thresholds"],
        "validation_row_count": report["validation_row_count"],
        "validation_passing_row_count": report["validation_passing_row_count"],
        "selection_reason_codes": report["selection_reason_codes"],
        "validation_entry_count": selected.get("entry_count"),
        "validation_total_pnl": selected.get("total_pnl"),
        "validation_support_gate_passed": selected.get(
            "validation_support_gate_passed",
            False,
        ),
        "shadow_entry_count": shadow.get("entry_count"),
        "shadow_total_pnl": shadow.get("total_pnl"),
        "shadow_support_gate_passed": shadow.get("support_gate_passed", False),
        "promotion_evidence_eligible": report["promotion_evidence_eligible"],
        "paper_run_resume_allowed": report["paper_run_resume_allowed"],
    }


def sell_before_close_support_aware_threshold_selection_markdown(
    report: dict[str, Any],
) -> str:
    """Render support-aware threshold selection report markdown."""

    selected = report.get("selected_validation_row") or {}
    shadow = report.get("shadow_evaluation_row") or {}
    lines = [
        "# SELL_BEFORE_CLOSE Support-Aware Threshold Selection",
        "",
        f"- candidate_name: `{report['candidate_name']}`",
        f"- threshold_selection_method: `{report['threshold_selection_method']}`",
        f"- threshold_selection_fit_split: `{report['threshold_selection_fit_split']}`",
        f"- threshold_selection_evaluation_split: `{report['threshold_selection_evaluation_split']}`",
        f"- uses_shadow_for_fit: `{str(report['uses_shadow_for_fit']).lower()}`",
        "- shadow_sweep_not_used_for_threshold_fit: "
        f"`{str(report['shadow_sweep_not_used_for_threshold_fit']).lower()}`",
        f"- selected_thresholds: `{json.dumps(report['selected_thresholds'], sort_keys=True)}`",
        f"- validation_row_count: `{report['validation_row_count']}`",
        f"- validation_passing_row_count: `{report['validation_passing_row_count']}`",
        "- selection_reason_codes: "
        f"`{json.dumps(report['selection_reason_codes'])}`",
        f"- promotion_evidence_eligible: `{str(report['promotion_evidence_eligible']).lower()}`",
        f"- paper_run_resume_allowed: `{str(report['paper_run_resume_allowed']).lower()}`",
        "",
        "## Selected Validation Row",
        "",
        _row_markdown(selected),
        "",
        "## Shadow Evaluation Row",
        "",
        _row_markdown(shadow),
        "",
        "- paper_only: true",
        "- capital_at_risk: false",
        "- polymarket_write_enabled: false",
        "- wallet_signing_enabled: false",
        "",
    ]
    return "\n".join(lines)


def _thresholds(values: dict[str, float]) -> dict[str, float]:
    merged = dict(SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_THRESHOLDS)
    merged.update(values)
    return dict(sorted(merged.items()))


def _decision_support_prefilter(
    *,
    decisions: list[dict[str, Any]],
    guard_summary: dict[str, Any],
) -> dict[str, Any]:
    entry_decisions = [
        decision
        for decision in decisions
        if str(decision.get("action")) in {"BUY_UP", "BUY_DOWN"}
        and str(decision.get("intended_exit_policy")) == "sell_before_close"
    ]
    sell_decisions = [
        decision
        for decision in decisions
        if str(decision.get("action")) in {"SELL_UP", "SELL_DOWN"}
    ]
    side_distribution = Counter(_entry_side(decision) for decision in entry_decisions)
    side_distribution = Counter(
        {side: count for side, count in side_distribution.items() if side}
    )
    entry_count = len(entry_decisions)
    sell_count = len(sell_decisions)
    market_count = len({str(decision.get("market_id")) for decision in entry_decisions})
    side_count = len(side_distribution)
    p_up_rate = float(
        guard_summary["candidate_scoped_p_up_action_disagreement_rate"]
    )
    reasons = _validation_gate_reason_codes(
        entry_count=entry_count,
        unique_market_count=market_count,
        side_count=side_count,
        sell_count=sell_count,
        residual_count=0,
        residual_drag=0.0,
        total_pnl=1.0,
        mean_pnl_per_entry=1.0,
        p_up_disagreement_rate=p_up_rate,
    )
    return {
        "entry_decision_count": entry_count,
        "sell_decision_count": sell_count,
        "unique_market_count": market_count,
        "side_count": side_count,
        "side_distribution": dict(sorted(side_distribution.items())),
        "mean_pnl_per_entry": 0.0,
        "validation_support_gate_reason_codes": reasons,
    }


def _threshold_row_payload(
    *,
    split_name: str,
    thresholds: dict[str, float],
    guard_summary: dict[str, Any],
    support_counts: dict[str, Any],
    replay_report: dict[str, Any] | None,
    residual_drag: float,
    residual_count: int,
    validation_reason_codes: list[str],
    support_gate_passed: bool,
    support_gate_reason_codes: list[str],
    promotion_support_eligible: bool,
) -> dict[str, Any]:
    return {
        "candidate_name": SELL_BEFORE_CLOSE_SUPPORT_AWARE_P_UP_ALIGNED_CANDIDATE_NAME,
        "split_name": split_name,
        "thresholds": dict(sorted(thresholds.items())),
        "entry_count": int(support_counts["entry_decision_count"]),
        "sell_count": int(support_counts["sell_decision_count"]),
        "unique_market_count": int(support_counts["unique_market_count"]),
        "side_count": int(support_counts["side_count"]),
        "side_distribution": support_counts["side_distribution"],
        "positions_opened_count": int(support_counts["entry_decision_count"]),
        "positions_closed_before_settlement_count": int(
            support_counts["sell_decision_count"]
        ),
        "positions_opened_but_not_closed_before_settlement": residual_count,
        "residual_count": residual_count,
        "realized_trade_pnl": 0.0
        if replay_report is None
        else float(replay_report["realized_trade_pnl"]),
        "settlement_pnl": 0.0
        if replay_report is None
        else float(replay_report["settlement_pnl"]),
        "total_pnl": 0.0
        if replay_report is None
        else float(replay_report["total_polymarket_pnl"]),
        "mean_pnl_per_entry": float(support_counts["mean_pnl_per_entry"]),
        "max_drawdown": 0.0
        if replay_report is None
        else float(replay_report["max_drawdown"]),
        "replay_residual_settlement_drag": residual_drag,
        "candidate_scoped_p_up_action_disagreement_count": int(
            guard_summary["candidate_scoped_p_up_action_disagreement_count"]
        ),
        "candidate_scoped_p_up_action_disagreement_denominator": int(
            guard_summary["candidate_scoped_p_up_action_disagreement_denominator"]
        ),
        "candidate_scoped_p_up_action_disagreement_rate": float(
            guard_summary["candidate_scoped_p_up_action_disagreement_rate"]
        ),
        "candidate_scoped_p_up_action_disagreement_within_limit": bool(
            guard_summary[
                "candidate_scoped_p_up_action_disagreement_within_limit"
            ]
        ),
        "entry_decision_count_before_guard": int(
            guard_summary["entry_decision_count_before_guard"]
        ),
        "entry_decision_count_after_exit_guard": int(
            guard_summary["entry_decision_count_after_exit_guard"]
        ),
        "entry_decision_count_after_p_up_alignment": int(
            guard_summary["entry_decision_count_after_p_up_alignment"]
        ),
        "entry_decision_count_after_guard": int(
            guard_summary["entry_decision_count_after_guard"]
        ),
        "entry_filter_blocked_count": int(guard_summary["entry_filter_blocked_count"]),
        "entry_filter_blocked_by_p_up_alignment_count": int(
            guard_summary["entry_filter_blocked_by_p_up_alignment_count"]
        ),
        "entry_filter_blocked_by_quality_count": int(
            guard_summary["entry_filter_blocked_by_quality_count"]
        ),
        "reentry_blocked_count": int(guard_summary["reentry_blocked_count"]),
        "validation_support_gate_passed": not validation_reason_codes,
        "validation_support_gate_reason_codes": validation_reason_codes,
        "support_gate_passed": support_gate_passed,
        "support_gate_reason_codes": support_gate_reason_codes,
        "promotion_support_eligible": promotion_support_eligible,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        **compact_safety_fields(),
    }


def _entry_side(decision: dict[str, Any]) -> str:
    selected = str(decision.get("selected_outcome") or "")
    if selected in {"UP", "DOWN"}:
        return selected
    action = str(decision.get("action") or "")
    if action.endswith("_UP"):
        return "UP"
    if action.endswith("_DOWN"):
        return "DOWN"
    return ""


def _validation_gate_reason_codes(
    *,
    entry_count: int,
    unique_market_count: int,
    side_count: int,
    sell_count: int,
    residual_count: int,
    residual_drag: float,
    total_pnl: float,
    mean_pnl_per_entry: float,
    p_up_disagreement_rate: float,
) -> list[str]:
    gates = SELL_BEFORE_CLOSE_SUPPORT_AWARE_VALIDATION_GATES
    reasons = []
    if entry_count < int(gates["min_entry_count"]):
        reasons.append("support_aware_validation_entry_support_insufficient")
    if unique_market_count < int(gates["min_unique_market_count"]):
        reasons.append("support_aware_validation_market_support_insufficient")
    if side_count < int(gates["min_side_count"]):
        reasons.append("support_aware_validation_side_coverage_insufficient")
    if sell_count < int(gates["min_sell_count"]):
        reasons.append("support_aware_validation_sell_support_insufficient")
    if residual_count > int(gates["max_residual_count"]):
        reasons.append("support_aware_validation_residual_positions_remaining")
    if residual_drag < float(gates["min_residual_settlement_drag"]):
        reasons.append("support_aware_validation_residual_settlement_drag_negative")
    if total_pnl <= float(gates["min_total_pnl_exclusive"]):
        reasons.append("support_aware_validation_total_pnl_not_positive")
    if mean_pnl_per_entry <= float(gates["min_mean_pnl_per_entry_exclusive"]):
        reasons.append("support_aware_validation_mean_pnl_per_entry_not_positive")
    if p_up_disagreement_rate > float(gates["max_p_up_disagreement_rate"]):
        reasons.append("support_aware_validation_p_up_disagreement_excessive")
    return sorted(reasons)


def _select_validation_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return sorted(
        rows,
        key=lambda row: (
            -float(row["total_pnl"]),
            -int(row["entry_count"]),
            float(row["max_drawdown"]),
            float(row["candidate_scoped_p_up_action_disagreement_rate"]),
            tuple(sorted(row["thresholds"].items())),
        ),
    )[0]


def _row_markdown(row: dict[str, Any]) -> str:
    if not row:
        return "- none"
    return "\n".join(
        [
            f"- entry_count: `{row['entry_count']}`",
            f"- unique_market_count: `{row['unique_market_count']}`",
            f"- side_count: `{row['side_count']}`",
            f"- sell_count: `{row['sell_count']}`",
            f"- total_pnl: `{row['total_pnl']}`",
            f"- mean_pnl_per_entry: `{row['mean_pnl_per_entry']}`",
            f"- max_drawdown: `{row['max_drawdown']}`",
            f"- residual_count: `{row['residual_count']}`",
            "- replay_residual_settlement_drag: "
            f"`{row['replay_residual_settlement_drag']}`",
            "- p_up_disagreement_rate: "
            f"`{row['candidate_scoped_p_up_action_disagreement_rate']}`",
            f"- support_gate_passed: `{str(row['support_gate_passed']).lower()}`",
            "- support_gate_reason_codes: "
            f"`{json.dumps(row['support_gate_reason_codes'])}`",
        ]
    )
