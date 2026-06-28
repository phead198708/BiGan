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
SELL_BEFORE_CLOSE_SUPPORT_AWARE_THRESHOLD_FAILURE_ATTRIBUTION_SCHEMA_VERSION = (
    "bigan-v8-polymarket-sell-before-close-support-aware-threshold-failure-attribution-v1"
)
SELL_BEFORE_CLOSE_VALIDATION_FAILURE_DRILLDOWN_SCHEMA_VERSION = (
    "bigan-v8-polymarket-sell-before-close-validation-failure-drilldown-v1"
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
VALIDATION_GATE_REASON_CODES = {
    "entry_support_gate": "validation_entry_support_insufficient",
    "market_support_gate": "validation_market_support_insufficient",
    "side_coverage_gate": "validation_side_coverage_insufficient",
    "sell_support_gate": "validation_sell_support_insufficient",
    "residual_count_gate": "validation_residual_positions_remaining",
    "residual_settlement_drag_gate": "validation_residual_settlement_drag_negative",
    "total_pnl_gate": "validation_total_pnl_not_positive",
    "mean_pnl_per_entry_gate": "validation_mean_pnl_per_entry_not_positive",
    "p_up_disagreement_gate": "validation_p_up_disagreement_excessive",
}
SUPPORT_VS_PNL_BUCKETS = (
    "support_passed_pnl_passed",
    "support_passed_pnl_failed",
    "support_failed_pnl_passed",
    "support_failed_pnl_failed",
)
ENTRY_QUALITY_THRESHOLD_PARAMETERS = (
    "p_up_alignment_min",
    "min_calibrated_action_score",
    "min_best_action_margin",
    "min_queue_fill_probability_proxy",
    "min_seconds_to_close",
    "max_entries_per_market",
    "min_reentry_cooldown_seconds",
)
FAILURE_INTERPRETATIONS_ALLOWED = {
    "support_too_sparse",
    "one_sided_support",
    "positive_pnl_only_at_low_support",
    "support_adequate_pnl_negative",
    "pnl_not_positive_under_support",
    "p_up_alignment_over_filters",
    "p_up_alignment_not_primary_blocker",
    "exit_reliability_failure",
    "action_value_ranking_failure_likely",
    "collect_more_data_needed",
    "insufficient_evidence",
    "mixed_threshold_failure",
}
RECOMMENDED_NEXT_ACTIONS_ALLOWED = {
    "collect_more_real_corpus",
    "expand_validation_grid_without_shadow_fit",
    "revise_action_value_ranking_model",
    "revise_p_up_auxiliary_calibration",
    "add_side_balancing_candidate",
    "add_support_aware_ranking_candidate",
    "keep_blocked",
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
    attribution_report = (
        build_sell_before_close_support_aware_threshold_failure_attribution_report(
            validation_rows=validation_rows,
            selection_reason_codes=reason_codes,
            split_name=validation_split,
        )
    )
    drilldown_report = build_sell_before_close_validation_failure_drilldown_report(
        validation_rows=validation_rows,
        selection_reason_codes=reason_codes,
        split_name=validation_split,
    )
    drilldown_summary = sell_before_close_validation_failure_drilldown_summary(
        drilldown_report
    )
    attribution_report["sell_before_close_validation_failure_drilldown_summary"] = (
        drilldown_summary
    )
    attribution_report[
        "sell_before_close_support_aware_threshold_failure_attribution_report_id"
    ] = canonical_json_sha256(attribution_report)
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
        "threshold_selection_passed": not reason_codes,
        "threshold_selection_failed": bool(reason_codes),
        "threshold_selection_failure_reason_codes": reason_codes,
        "selection_reason_codes": reason_codes,
        "failure_attribution_report": attribution_report,
        "validation_failure_drilldown_report": drilldown_report,
        "sell_before_close_validation_failure_drilldown_summary": (
            drilldown_summary
        ),
        "threshold_selection_failure_interpretation": attribution_report[
            "threshold_selection_failure_interpretation"
        ],
        "recommended_next_action": attribution_report["recommended_next_action"],
        "top_failed_gates": attribution_report["top_failed_gates"],
        "best_near_miss_rows": attribution_report["best_near_miss_rows"],
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
    return _threshold_row_payload(
        split_name=split_name,
        thresholds=thresholds,
        guard_summary=guard_summary,
        support_counts=support,
        replay_report=replay_report,
        residual_drag=residual_drag,
        residual_count=residual_count,
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
        "threshold_selection_passed": report["threshold_selection_passed"],
        "threshold_selection_failed": report["threshold_selection_failed"],
        "threshold_selection_failure_reason_codes": report[
            "threshold_selection_failure_reason_codes"
        ],
        "selection_reason_codes": report["selection_reason_codes"],
        "failure_attribution_report_path": report.get(
            "failure_attribution_report_path"
        ),
        "failure_attribution_report_sha256": report.get(
            "failure_attribution_report_sha256"
        ),
        "validation_failure_drilldown_report_path": report.get(
            "validation_failure_drilldown_report_path"
        ),
        "validation_failure_drilldown_report_sha256": report.get(
            "validation_failure_drilldown_report_sha256"
        ),
        "sell_before_close_validation_failure_drilldown_summary": report.get(
            "sell_before_close_validation_failure_drilldown_summary",
        ),
        "threshold_selection_failure_interpretation": report[
            "threshold_selection_failure_interpretation"
        ],
        "recommended_next_action": report["recommended_next_action"],
        "top_failed_gates": report["top_failed_gates"],
        "best_near_miss_rows": report["best_near_miss_rows"],
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
        f"- threshold_selection_passed: `{str(report['threshold_selection_passed']).lower()}`",
        f"- threshold_selection_failed: `{str(report['threshold_selection_failed']).lower()}`",
        "- threshold_selection_failure_reason_codes: "
        f"`{json.dumps(report['threshold_selection_failure_reason_codes'])}`",
        "- selection_reason_codes: "
        f"`{json.dumps(report['selection_reason_codes'])}`",
        "- threshold_selection_failure_interpretation: "
        f"`{report['threshold_selection_failure_interpretation']}`",
        f"- recommended_next_action: `{report['recommended_next_action']}`",
        "- failure_attribution_report_path: "
        f"`{report.get('failure_attribution_report_path')}`",
        "- failure_attribution_report_sha256: "
        f"`{report.get('failure_attribution_report_sha256')}`",
        "- validation_failure_drilldown_report_path: "
        f"`{report.get('validation_failure_drilldown_report_path')}`",
        "- validation_failure_drilldown_report_sha256: "
        f"`{report.get('validation_failure_drilldown_report_sha256')}`",
        f"- promotion_evidence_eligible: `{str(report['promotion_evidence_eligible']).lower()}`",
        f"- paper_run_resume_allowed: `{str(report['paper_run_resume_allowed']).lower()}`",
        "",
        "## Top Failed Gates",
        "",
        "| gate | failed_rows | fail_rate |",
        "|---|---:|---:|",
    ]
    for row in report.get("top_failed_gates", [])[:10]:
        lines.append(
            "| {gate} | {failed} | {rate:.4f} |".format(
                gate=row["gate_name"],
                failed=row["failed_row_count"],
                rate=row["fail_rate"],
            )
        )
    lines.extend(
        [
            "",
            "## Best Near-Miss Rows",
            "",
            "| gate | total_pnl | entries | markets | sides | reasons |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in report.get("best_near_miss_rows", [])[:10]:
        lines.append(
            "| {gate} | {total:.6f} | {entries} | {markets} | {sides} | {reasons} |".format(
                gate=row["gate_name"],
                total=row["total_pnl"],
                entries=row["entry_count"],
                markets=row["unique_market_count"],
                sides=row["side_count"],
                reasons=", ".join(row["failed_reason_codes"]) or "none",
            )
        )
    lines.extend(
        [
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
    )
    return "\n".join(lines)


def build_sell_before_close_support_aware_threshold_failure_attribution_report(
    *,
    validation_rows: list[dict[str, Any]],
    selection_reason_codes: list[str],
    split_name: str,
) -> dict[str, Any]:
    """Attribute why validation-fitted support-aware thresholds failed."""

    row_count = len(validation_rows)
    passing_rows = [
        row for row in validation_rows if row["validation_support_gate_passed"]
    ]
    gate_attribution = _gate_level_attribution(validation_rows)
    reason_attribution = _reason_code_attribution(validation_rows)
    combination_attribution = _combination_attribution(validation_rows)
    near_miss_rows = _near_miss_rows(validation_rows)
    best_rows = _best_rows_by_objective(validation_rows)
    interpretation = _failure_interpretation(
        validation_rows=validation_rows,
        gate_attribution=gate_attribution,
        reason_attribution=reason_attribution,
        passing_row_count=len(passing_rows),
    )
    recommended_next_action = _recommended_next_action(interpretation)
    report = {
        "schema_version": (
            SELL_BEFORE_CLOSE_SUPPORT_AWARE_THRESHOLD_FAILURE_ATTRIBUTION_SCHEMA_VERSION
        ),
        "candidate_name": SELL_BEFORE_CLOSE_SUPPORT_AWARE_P_UP_ALIGNED_CANDIDATE_NAME,
        "split_name": split_name,
        "diagnostic_only": True,
        "uses_shadow_for_fit": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        "row_count": row_count,
        "validation_row_count": row_count,
        "validation_passing_row_count": len(passing_rows),
        "validation_failed_row_count": row_count - len(passing_rows),
        "selection_reason_codes": selection_reason_codes,
        "threshold_selection_passed": not selection_reason_codes,
        "threshold_selection_failed": bool(selection_reason_codes),
        "threshold_selection_failure_reason_codes": selection_reason_codes,
        "validation_gates": SELL_BEFORE_CLOSE_SUPPORT_AWARE_VALIDATION_GATES,
        "gate_level_attribution": gate_attribution,
        "reason_code_attribution": reason_attribution,
        "cumulative_blocker_combinations": combination_attribution,
        "best_near_miss_rows": near_miss_rows,
        "best_rows_by_objective": best_rows,
        "top_failed_gates": sorted(
            gate_attribution,
            key=lambda row: (
                -int(row["failed_row_count"]),
                str(row["gate_name"]),
            ),
        )[:5],
        "top_reason_codes": sorted(
            reason_attribution,
            key=lambda row: (-int(row["row_count"]), str(row["reason_code"])),
        )[:10],
        "threshold_selection_failure_interpretation": interpretation,
        "recommended_next_action": recommended_next_action,
        "all_validation_rows_have_failed_gate_attribution": all(
            bool(row["failed_reason_codes"]) for row in validation_rows
        ),
        "validation_threshold_rows": [
            _attribution_row(row) for row in validation_rows
        ],
        **compact_safety_fields(),
    }
    report[
        "sell_before_close_support_aware_threshold_failure_attribution_report_id"
    ] = canonical_json_sha256(report)
    return report


def sell_before_close_support_aware_threshold_failure_attribution_summary(
    report: dict[str, Any],
) -> dict[str, Any]:
    """Return compact failure-attribution evidence for embedding."""

    return {
        "schema_version": report["schema_version"],
        "candidate_name": report["candidate_name"],
        "split_name": report["split_name"],
        "diagnostic_only": report["diagnostic_only"],
        "uses_shadow_for_fit": report["uses_shadow_for_fit"],
        "row_count": report["row_count"],
        "validation_row_count": report["validation_row_count"],
        "validation_passing_row_count": report["validation_passing_row_count"],
        "validation_failed_row_count": report["validation_failed_row_count"],
        "threshold_selection_passed": report["threshold_selection_passed"],
        "threshold_selection_failed": report["threshold_selection_failed"],
        "threshold_selection_failure_reason_codes": report[
            "threshold_selection_failure_reason_codes"
        ],
        "top_failed_gates": report["top_failed_gates"],
        "top_reason_codes": report["top_reason_codes"],
        "best_near_miss_rows": report["best_near_miss_rows"][:10],
        "best_rows_by_objective": report["best_rows_by_objective"],
        "threshold_selection_failure_interpretation": report[
            "threshold_selection_failure_interpretation"
        ],
        "recommended_next_action": report["recommended_next_action"],
        "validation_failure_drilldown_report_path": report.get(
            "validation_failure_drilldown_report_path"
        ),
        "validation_failure_drilldown_report_sha256": report.get(
            "validation_failure_drilldown_report_sha256"
        ),
        "sell_before_close_validation_failure_drilldown_summary": report.get(
            "sell_before_close_validation_failure_drilldown_summary",
        ),
        "promotion_evidence_eligible": report["promotion_evidence_eligible"],
        "paper_run_resume_allowed": report["paper_run_resume_allowed"],
    }


def sell_before_close_support_aware_threshold_failure_attribution_markdown(
    report: dict[str, Any],
) -> str:
    """Render failure attribution markdown."""

    lines = [
        "# SELL_BEFORE_CLOSE Support-Aware Threshold Failure Attribution",
        "",
        f"- candidate_name: `{report['candidate_name']}`",
        f"- split_name: `{report['split_name']}`",
        f"- diagnostic_only: `{str(report['diagnostic_only']).lower()}`",
        f"- uses_shadow_for_fit: `{str(report['uses_shadow_for_fit']).lower()}`",
        f"- row_count: `{report['row_count']}`",
        f"- validation_passing_row_count: `{report['validation_passing_row_count']}`",
        "- threshold_selection_failure_reason_codes: "
        f"`{json.dumps(report['threshold_selection_failure_reason_codes'])}`",
        "- threshold_selection_failure_interpretation: "
        f"`{report['threshold_selection_failure_interpretation']}`",
        f"- recommended_next_action: `{report['recommended_next_action']}`",
        "- validation_failure_drilldown_report_path: "
        f"`{report.get('validation_failure_drilldown_report_path')}`",
        "- validation_failure_drilldown_report_sha256: "
        f"`{report.get('validation_failure_drilldown_report_sha256')}`",
        f"- promotion_evidence_eligible: `{str(report['promotion_evidence_eligible']).lower()}`",
        f"- paper_run_resume_allowed: `{str(report['paper_run_resume_allowed']).lower()}`",
        "",
        "## Gate Attribution",
        "",
        "| gate | passed | failed | pass_rate | fail_rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["gate_level_attribution"]:
        lines.append(
            "| {gate} | {passed} | {failed} | {pass_rate:.4f} | {fail_rate:.4f} |".format(
                gate=row["gate_name"],
                passed=row["passed_row_count"],
                failed=row["failed_row_count"],
                pass_rate=row["pass_rate"],
                fail_rate=row["fail_rate"],
            )
        )
    lines.extend(
        [
            "",
            "## Reason Codes",
            "",
            "| reason_code | rows | row_rate |",
            "|---|---:|---:|",
        ]
    )
    for row in report["reason_code_attribution"]:
        lines.append(
            "| {reason} | {count} | {rate:.4f} |".format(
                reason=row["reason_code"],
                count=row["row_count"],
                rate=row["row_rate"],
            )
        )
    lines.extend(
        [
            "",
            "## Best Near Misses",
            "",
            "| gate | total_pnl | entries | markets | sides | reasons |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["best_near_miss_rows"][:10]:
        lines.append(
            "| {gate} | {total:.6f} | {entries} | {markets} | {sides} | {reasons} |".format(
                gate=row["gate_name"],
                total=row["total_pnl"],
                entries=row["entry_count"],
                markets=row["unique_market_count"],
                sides=row["side_count"],
                reasons=", ".join(row["failed_reason_codes"]) or "none",
            )
        )
    lines.extend(
        [
            "",
            "- paper_only: true",
            "- capital_at_risk: false",
            "- polymarket_write_enabled: false",
            "- wallet_signing_enabled: false",
            "",
        ]
    )
    return "\n".join(lines)


def build_sell_before_close_validation_failure_drilldown_report(
    *,
    validation_rows: list[dict[str, Any]],
    selection_reason_codes: list[str],
    split_name: str,
) -> dict[str, Any]:
    """Build diagnostic drilldown for failed validation threshold selection."""

    row_count = len(validation_rows)
    quadrant = _support_vs_pnl_quadrants(validation_rows)
    support_drilldown = _support_adequate_drilldown(validation_rows)
    positive_pnl_drilldown = _positive_pnl_drilldown(validation_rows)
    side_drilldown = _side_coverage_drilldown(validation_rows)
    p_up_drilldown = _p_up_alignment_drilldown(validation_rows)
    threshold_drilldown = _entry_quality_threshold_drilldown(validation_rows)
    action_side_drilldown = _action_side_market_drilldown(validation_rows)
    model_failure_clues = _model_failure_clues(validation_rows)
    interpretations = _failure_interpretations(
        row_count=row_count,
        support_drilldown=support_drilldown,
        positive_pnl_drilldown=positive_pnl_drilldown,
        side_drilldown=side_drilldown,
        p_up_drilldown=p_up_drilldown,
        model_failure_clues=model_failure_clues,
    )
    primary_interpretation = _primary_failure_interpretation(interpretations)
    recommendations = _recommended_next_actions(
        interpretations=interpretations,
        positive_pnl_only_at_low_support=positive_pnl_drilldown[
            "positive_pnl_only_at_low_support"
        ],
        side_coverage_all_rows_failed=model_failure_clues[
            "side_coverage_all_rows_failed"
        ],
        all_rows_total_pnl_failed=model_failure_clues["all_rows_total_pnl_failed"],
        all_rows_mean_pnl_failed=model_failure_clues["all_rows_mean_pnl_failed"],
    )
    report = {
        "schema_version": SELL_BEFORE_CLOSE_VALIDATION_FAILURE_DRILLDOWN_SCHEMA_VERSION,
        "candidate_name": SELL_BEFORE_CLOSE_SUPPORT_AWARE_P_UP_ALIGNED_CANDIDATE_NAME,
        "split_name": split_name,
        "diagnostic_only": True,
        "uses_shadow_for_fit": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        "selection_reason_codes": selection_reason_codes,
        "validation_row_count": row_count,
        "validation_passing_row_count": sum(
            1 for row in validation_rows if row["validation_support_gate_passed"]
        ),
        "support_vs_pnl_quadrants": quadrant,
        "support_adequate_drilldown": support_drilldown,
        "positive_pnl_drilldown": positive_pnl_drilldown,
        "side_coverage_drilldown": side_drilldown,
        "p_up_alignment_drilldown": p_up_drilldown,
        "entry_quality_threshold_drilldown": threshold_drilldown,
        "action_side_market_drilldown": action_side_drilldown,
        "model_ranking_failure_clues": model_failure_clues,
        "primary_failure_interpretation": primary_interpretation,
        "failure_interpretations": interpretations,
        "recommended_next_actions": recommendations,
        **compact_safety_fields(),
    }
    report["sell_before_close_validation_failure_drilldown_report_id"] = (
        canonical_json_sha256(report)
    )
    return report


def sell_before_close_validation_failure_drilldown_summary(
    report: dict[str, Any],
) -> dict[str, Any]:
    """Return compact validation failure drilldown evidence for embedding."""

    quadrants = report["support_vs_pnl_quadrants"]
    support = report["support_adequate_drilldown"]
    positive = report["positive_pnl_drilldown"]
    side = report["side_coverage_drilldown"]
    return {
        "schema_version": report["schema_version"],
        "candidate_name": report["candidate_name"],
        "split_name": report["split_name"],
        "diagnostic_only": report["diagnostic_only"],
        "uses_shadow_for_fit": report["uses_shadow_for_fit"],
        "promotion_evidence_eligible": report["promotion_evidence_eligible"],
        "paper_run_resume_allowed": report["paper_run_resume_allowed"],
        "#146_start_allowed": report["#146_start_allowed"],
        "#134_resume_allowed": report["#134_resume_allowed"],
        "validation_row_count": report["validation_row_count"],
        "validation_passing_row_count": report["validation_passing_row_count"],
        "support_passed_pnl_passed_count": quadrants[
            "support_passed_pnl_passed"
        ]["row_count"],
        "support_passed_pnl_failed_count": quadrants[
            "support_passed_pnl_failed"
        ]["row_count"],
        "support_failed_pnl_passed_count": quadrants[
            "support_failed_pnl_passed"
        ]["row_count"],
        "support_failed_pnl_failed_count": quadrants[
            "support_failed_pnl_failed"
        ]["row_count"],
        "support_adequate_row_count": support["support_adequate_row_count"],
        "support_adequate_positive_pnl_row_count": support[
            "support_adequate_positive_pnl_row_count"
        ],
        "support_adequate_best_total_pnl": support[
            "support_adequate_best_total_pnl"
        ],
        "positive_pnl_row_count": positive["positive_pnl_row_count"],
        "positive_pnl_support_passed_count": positive[
            "positive_pnl_support_passed_count"
        ],
        "positive_pnl_only_at_low_support": positive[
            "positive_pnl_only_at_low_support"
        ],
        "side_coverage_failure_rate": side["side_coverage_failure_rate"],
        "rows_with_both_sides": side["rows_with_both_sides"],
        "positive_pnl_rows_with_both_sides": side[
            "positive_pnl_rows_with_both_sides"
        ],
        "side_coverage_interpretation": side["side_coverage_interpretation"],
        "primary_failure_interpretation": report[
            "primary_failure_interpretation"
        ],
        "failure_interpretations": report["failure_interpretations"],
        "recommended_next_actions": report["recommended_next_actions"],
    }


def sell_before_close_validation_failure_drilldown_markdown(
    report: dict[str, Any],
) -> str:
    """Render validation failure drilldown report markdown."""

    summary = sell_before_close_validation_failure_drilldown_summary(report)
    lines = [
        "# SELL_BEFORE_CLOSE Validation Failure Drilldown",
        "",
        f"- candidate_name: `{report['candidate_name']}`",
        f"- split_name: `{report['split_name']}`",
        f"- diagnostic_only: `{str(report['diagnostic_only']).lower()}`",
        f"- uses_shadow_for_fit: `{str(report['uses_shadow_for_fit']).lower()}`",
        f"- validation_row_count: `{report['validation_row_count']}`",
        f"- validation_passing_row_count: `{report['validation_passing_row_count']}`",
        "- primary_failure_interpretation: "
        f"`{report['primary_failure_interpretation']}`",
        "- failure_interpretations: "
        f"`{json.dumps(report['failure_interpretations'])}`",
        "- recommended_next_actions: "
        f"`{json.dumps(report['recommended_next_actions'])}`",
        f"- promotion_evidence_eligible: `{str(report['promotion_evidence_eligible']).lower()}`",
        f"- paper_run_resume_allowed: `{str(report['paper_run_resume_allowed']).lower()}`",
        "",
        "## Support vs PnL Quadrants",
        "",
        "| bucket | rows | rate | best_total_pnl | best_mean_pnl | entries | markets | sides | reasons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for bucket in SUPPORT_VS_PNL_BUCKETS:
        row = report["support_vs_pnl_quadrants"][bucket]
        lines.append(
            "| {bucket} | {count} | {rate:.4f} | {total:.6f} | {mean:.6f} | "
            "{entries} | {markets} | {sides} | {reasons} |".format(
                bucket=bucket,
                count=row["row_count"],
                rate=row["row_rate"],
                total=row["best_total_pnl"],
                mean=row["best_mean_pnl_per_entry"],
                entries=row["best_entry_count"],
                markets=row["best_unique_market_count"],
                sides=row["best_side_count"],
                reasons=", ".join(row["dominant_failed_reason_codes"]) or "none",
            )
        )
    lines.extend(
        [
            "",
            "## Drilldown Summary",
            "",
            f"- support_adequate_row_count: `{summary['support_adequate_row_count']}`",
            "- support_adequate_positive_pnl_row_count: "
            f"`{summary['support_adequate_positive_pnl_row_count']}`",
            f"- positive_pnl_row_count: `{summary['positive_pnl_row_count']}`",
            "- positive_pnl_support_passed_count: "
            f"`{summary['positive_pnl_support_passed_count']}`",
            "- side_coverage_failure_rate: "
            f"`{summary['side_coverage_failure_rate']}`",
            f"- rows_with_both_sides: `{summary['rows_with_both_sides']}`",
            "- positive_pnl_rows_with_both_sides: "
            f"`{summary['positive_pnl_rows_with_both_sides']}`",
            "- side_coverage_interpretation: "
            f"`{summary['side_coverage_interpretation']}`",
            "",
            "## Entry Quality Thresholds",
            "",
        ]
    )
    for parameter, rows in report["entry_quality_threshold_drilldown"].items():
        lines.extend(
            [
                f"### `{parameter}`",
                "",
                "| value | rows | mean_entries | max_entries | mean_total_pnl | max_total_pnl | positive_pnl_rate | support_pass_rate | side_pass_rate |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                "| {value} | {count} | {mean_entries:.4f} | {max_entries} | "
                "{mean_total:.6f} | {max_total:.6f} | {pos:.4f} | "
                "{support:.4f} | {side:.4f} |".format(
                    value=row["threshold_value"],
                    count=row["row_count"],
                    mean_entries=row["mean_entry_count"],
                    max_entries=row["max_entry_count"],
                    mean_total=row["mean_total_pnl"],
                    max_total=row["max_total_pnl"],
                    pos=row["positive_pnl_rate"],
                    support=row["support_pass_rate"],
                    side=row["side_coverage_pass_rate"],
                )
            )
        lines.append("")
    lines.extend(
        [
            "- paper_only: true",
            "- capital_at_risk: false",
            "- polymarket_write_enabled: false",
            "- wallet_signing_enabled: false",
            "",
        ]
    )
    return "\n".join(lines)


def _support_vs_pnl_quadrants(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped = {bucket: [] for bucket in SUPPORT_VS_PNL_BUCKETS}
    for row in rows:
        support = _support_gates_passed(row)
        pnl = _pnl_gates_passed(row)
        if support and pnl:
            bucket = "support_passed_pnl_passed"
        elif support:
            bucket = "support_passed_pnl_failed"
        elif pnl:
            bucket = "support_failed_pnl_passed"
        else:
            bucket = "support_failed_pnl_failed"
        grouped[bucket].append(row)
    row_count = len(rows)
    return {
        bucket: _bucket_summary(bucket_rows, row_count)
        for bucket, bucket_rows in grouped.items()
    }


def _bucket_summary(
    rows: list[dict[str, Any]],
    total_row_count: int,
) -> dict[str, Any]:
    best = _rank_rows(rows)[0] if rows else None
    return {
        "row_count": len(rows),
        "row_rate": 0.0 if total_row_count == 0 else len(rows) / total_row_count,
        "best_total_pnl": 0.0 if best is None else float(best["total_pnl"]),
        "best_mean_pnl_per_entry": max(
            (float(row["mean_pnl_per_entry"]) for row in rows),
            default=0.0,
        ),
        "best_entry_count": 0 if best is None else int(best["entry_count"]),
        "best_unique_market_count": 0
        if best is None
        else int(best["unique_market_count"]),
        "best_side_count": 0 if best is None else int(best["side_count"]),
        "best_thresholds": {} if best is None else best["thresholds"],
        "dominant_failed_reason_codes": _dominant_reason_codes(rows),
    }


def _support_adequate_drilldown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    support_rows = [row for row in rows if _support_gates_passed(row)]
    positive_rows = [row for row in support_rows if float(row["total_pnl"]) > 0.0]
    best = _rank_rows(support_rows)[0] if support_rows else None
    return {
        "support_adequate_row_count": len(support_rows),
        "support_adequate_positive_pnl_row_count": len(positive_rows),
        "support_adequate_best_total_pnl": 0.0
        if best is None
        else float(best["total_pnl"]),
        "support_adequate_best_mean_pnl_per_entry": 0.0
        if best is None
        else float(best["mean_pnl_per_entry"]),
        "support_adequate_best_thresholds": {} if best is None else best["thresholds"],
        "support_adequate_failure_reason_codes": _dominant_reason_codes(
            support_rows,
        ),
        "support_adequate_pnl_distribution": _distribution(
            [float(row["total_pnl"]) for row in support_rows]
        ),
        "support_adequate_pnl_failure": bool(support_rows)
        and not any(_pnl_gates_passed(row) for row in support_rows),
    }


def _positive_pnl_drilldown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive_rows = [row for row in rows if float(row["total_pnl"]) > 0.0]
    support_passed = [row for row in positive_rows if _support_gates_passed(row)]
    best = _rank_rows(positive_rows)[0] if positive_rows else None
    return {
        "positive_pnl_row_count": len(positive_rows),
        "positive_pnl_support_passed_count": len(support_passed),
        "positive_pnl_best_entry_count": 0
        if best is None
        else int(best["entry_count"]),
        "positive_pnl_best_unique_market_count": 0
        if best is None
        else int(best["unique_market_count"]),
        "positive_pnl_best_side_count": 0 if best is None else int(best["side_count"]),
        "positive_pnl_best_thresholds": {} if best is None else best["thresholds"],
        "positive_pnl_dominant_failed_gates": _dominant_failed_gates(
            positive_rows,
        ),
        "positive_pnl_only_at_low_support": bool(positive_rows)
        and not support_passed,
    }


def _side_coverage_drilldown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row_count = len(rows)
    side_failed = [
        row
        for row in rows
        if not row["validation_gate_results"]["side_coverage_gate"]["passed"]
    ]
    best = _best_side_coverage_row(rows)
    rows_with_up_only = [
        row for row in rows if _up_entry_count(row) > 0 and _down_entry_count(row) == 0
    ]
    rows_with_down_only = [
        row for row in rows if _down_entry_count(row) > 0 and _up_entry_count(row) == 0
    ]
    rows_with_both = [
        row for row in rows if _up_entry_count(row) > 0 and _down_entry_count(row) > 0
    ]
    positive_with_both = [
        row for row in rows_with_both if float(row["total_pnl"]) > 0.0
    ]
    support_with_both = [row for row in rows_with_both if _support_gates_passed(row)]
    total_up = sum(_up_entry_count(row) for row in rows)
    total_down = sum(_down_entry_count(row) for row in rows)
    p_up_alignment_blocked = sum(
        int(row.get("entry_filter_blocked_by_p_up_alignment_count", 0))
        for row in rows
    )
    return {
        "side_coverage_failure_rate": 0.0
        if row_count == 0
        else len(side_failed) / row_count,
        "best_side_distribution": {}
        if best is None
        else dict(best["side_distribution"]),
        "best_up_entry_count": 0 if best is None else _up_entry_count(best),
        "best_down_entry_count": 0 if best is None else _down_entry_count(best),
        "rows_with_up_only": len(rows_with_up_only),
        "rows_with_down_only": len(rows_with_down_only),
        "rows_with_both_sides": len(rows_with_both),
        "positive_pnl_rows_with_both_sides": len(positive_with_both),
        "support_adequate_rows_with_both_sides": len(support_with_both),
        "aggregate_up_entry_count": total_up,
        "aggregate_down_entry_count": total_down,
        "side_coverage_interpretation": _side_coverage_interpretation(
            row_count=row_count,
            rows_with_both_count=len(rows_with_both),
            total_up=total_up,
            total_down=total_down,
            p_up_alignment_blocked_count=p_up_alignment_blocked,
        ),
    }


def _p_up_alignment_drilldown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    zero_disagreement = [
        row
        for row in rows
        if float(row["candidate_scoped_p_up_action_disagreement_rate"]) == 0.0
    ]
    blocked_distribution = _counter_distribution(
        int(row.get("entry_filter_blocked_by_p_up_alignment_count", 0))
        for row in rows
    )
    best_rows = _best_rows_by_objective(rows)
    p_up_by_best = {
        name: None if row is None else row["thresholds"].get("p_up_alignment_min")
        for name, row in best_rows.items()
    }
    pass_rate = _gate_pass_rate(rows, "p_up_disagreement_gate")
    blocked_positive = [
        row
        for row in rows
        if int(row.get("entry_filter_blocked_by_p_up_alignment_count", 0)) > 0
        and float(row["total_pnl"]) > 0.0
    ]
    return {
        "p_up_alignment_blocked_count_distribution": blocked_distribution,
        "rows_with_zero_p_up_disagreement": len(zero_disagreement),
        "rows_with_zero_p_up_disagreement_and_positive_pnl": sum(
            1 for row in zero_disagreement if float(row["total_pnl"]) > 0.0
        ),
        "rows_with_zero_p_up_disagreement_and_support_passed": sum(
            1 for row in zero_disagreement if _support_gates_passed(row)
        ),
        "p_up_alignment_min_by_best_rows": p_up_by_best,
        "p_up_alignment_min_failure_pattern": _threshold_grouping(
            rows,
            "p_up_alignment_min",
        ),
        "p_up_disagreement_gate_pass_rate": pass_rate,
        "p_up_alignment_interpretation": _p_up_alignment_interpretation(
            pass_rate=pass_rate,
            blocked_positive_count=len(blocked_positive),
            blocked_distribution=blocked_distribution,
        ),
    }


def _entry_quality_threshold_drilldown(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        parameter: _threshold_grouping(rows, parameter)
        for parameter in ENTRY_QUALITY_THRESHOLD_PARAMETERS
    }


def _action_side_market_drilldown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "row_level_replay_decisions_retained": False,
        "compact_summary_from_threshold_rows": True,
        "action_side_pnl_is_row_level_proxy": True,
        "best_markets_by_total_pnl": [],
        "worst_markets_by_total_pnl": [],
        "market_count_distribution": _counter_distribution(
            int(row["unique_market_count"]) for row in rows
        ),
        "side_distribution_by_threshold_family": (
            _side_distribution_by_threshold_family(rows)
        ),
        "BUY_UP_SELL_BEFORE_CLOSE": _action_side_summary(rows, "UP"),
        "BUY_DOWN_SELL_BEFORE_CLOSE": _action_side_summary(rows, "DOWN"),
    }


def _model_failure_clues(rows: list[dict[str, Any]]) -> dict[str, Any]:
    support_rows = [row for row in rows if _support_gates_passed(row)]
    best_total = _rank_rows(rows)[0] if rows else None
    return {
        "all_rows_total_pnl_failed": bool(rows)
        and all(float(row["total_pnl"]) <= 0.0 for row in rows),
        "all_rows_mean_pnl_failed": bool(rows)
        and all(float(row["mean_pnl_per_entry"]) <= 0.0 for row in rows),
        "support_adequate_rows_negative_pnl": bool(support_rows)
        and all(float(row["total_pnl"]) <= 0.0 for row in support_rows),
        "best_total_pnl_row_support_failed": bool(best_total)
        and not _support_gates_passed(best_total),
        "side_coverage_all_rows_failed": bool(rows)
        and all(
            not row["validation_gate_results"]["side_coverage_gate"]["passed"]
            for row in rows
        ),
        "p_up_disagreement_gate_pass_rate": _gate_pass_rate(
            rows,
            "p_up_disagreement_gate",
        ),
        "exit_reliability_gate_pass_rate": _exit_reliability_gate_pass_rate(rows),
    }


def _failure_interpretations(
    *,
    row_count: int,
    support_drilldown: dict[str, Any],
    positive_pnl_drilldown: dict[str, Any],
    side_drilldown: dict[str, Any],
    p_up_drilldown: dict[str, Any],
    model_failure_clues: dict[str, Any],
) -> list[str]:
    if row_count == 0:
        return ["insufficient_evidence"]
    labels = []
    if support_drilldown["support_adequate_row_count"] == 0:
        labels.append("support_too_sparse")
    if side_drilldown["side_coverage_failure_rate"] >= 1.0:
        labels.append("one_sided_support")
    if positive_pnl_drilldown["positive_pnl_only_at_low_support"]:
        labels.append("positive_pnl_only_at_low_support")
    if support_drilldown["support_adequate_pnl_failure"]:
        labels.append("support_adequate_pnl_negative")
    if (
        model_failure_clues["all_rows_total_pnl_failed"]
        or model_failure_clues["all_rows_mean_pnl_failed"]
    ):
        labels.append("pnl_not_positive_under_support")
        labels.append("action_value_ranking_failure_likely")
    if p_up_drilldown["p_up_alignment_interpretation"] == "p_up_alignment_over_filters":
        labels.append("p_up_alignment_over_filters")
    elif p_up_drilldown["p_up_alignment_interpretation"] == (
        "p_up_alignment_not_primary_blocker"
    ):
        labels.append("p_up_alignment_not_primary_blocker")
    if model_failure_clues["exit_reliability_gate_pass_rate"] < 0.5:
        labels.append("exit_reliability_failure")
    if "support_too_sparse" in labels or "one_sided_support" in labels:
        labels.append("collect_more_data_needed")
    if not labels:
        labels.append("mixed_threshold_failure")
    return _ordered_unique(
        [label for label in labels if label in FAILURE_INTERPRETATIONS_ALLOWED]
    )


def _primary_failure_interpretation(interpretations: list[str]) -> str:
    for label in (
        "one_sided_support",
        "action_value_ranking_failure_likely",
        "support_too_sparse",
        "support_adequate_pnl_negative",
        "positive_pnl_only_at_low_support",
        "p_up_alignment_over_filters",
        "exit_reliability_failure",
    ):
        if label in interpretations:
            return label
    return interpretations[0] if interpretations else "insufficient_evidence"


def _recommended_next_actions(
    *,
    interpretations: list[str],
    positive_pnl_only_at_low_support: bool,
    side_coverage_all_rows_failed: bool,
    all_rows_total_pnl_failed: bool,
    all_rows_mean_pnl_failed: bool,
) -> list[str]:
    actions = []
    if side_coverage_all_rows_failed:
        actions.append("add_side_balancing_candidate")
    if all_rows_total_pnl_failed or all_rows_mean_pnl_failed:
        actions.append("revise_action_value_ranking_model")
    if positive_pnl_only_at_low_support or {
        "support_too_sparse",
        "one_sided_support",
        "collect_more_data_needed",
    }.intersection(interpretations):
        actions.append("collect_more_real_corpus")
    if "p_up_alignment_over_filters" in interpretations:
        actions.append("revise_p_up_auxiliary_calibration")
    if "mixed_threshold_failure" in interpretations:
        actions.append("expand_validation_grid_without_shadow_fit")
    actions.append("keep_blocked")
    return _ordered_unique(
        [action for action in actions if action in RECOMMENDED_NEXT_ACTIONS_ALLOWED]
    )


def _support_gates_passed(row: dict[str, Any]) -> bool:
    gates = row["validation_gate_results"]
    return all(
        bool(gates[gate_name]["passed"])
        for gate_name in (
            "entry_support_gate",
            "market_support_gate",
            "side_coverage_gate",
            "sell_support_gate",
        )
    )


def _pnl_gates_passed(row: dict[str, Any]) -> bool:
    return (
        float(row["total_pnl"]) > 0.0
        and float(row["mean_pnl_per_entry"]) > 0.0
        and int(row["residual_count"]) == 0
        and float(row["replay_residual_settlement_drag"]) >= 0.0
    )


def _dominant_reason_codes(
    rows: list[dict[str, Any]],
    limit: int = 5,
) -> list[str]:
    counts = Counter(reason for row in rows for reason in row["failed_reason_codes"])
    return [
        reason
        for reason, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
            :limit
        ]
    ]


def _dominant_failed_gates(
    rows: list[dict[str, Any]],
    limit: int = 5,
) -> list[str]:
    counts = Counter(gate for row in rows for gate in row["failed_gates"])
    return [
        gate
        for gate, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
            :limit
        ]
    ]


def _distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
            "mean": None,
        }
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p25": _quantile(ordered, 0.25),
        "median": _quantile(ordered, 0.50),
        "p75": _quantile(ordered, 0.75),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def _quantile(ordered: list[float], q: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _best_side_coverage_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return sorted(
        rows,
        key=lambda row: (
            -int(row["side_count"]),
            -int(row["entry_count"]),
            -float(row["total_pnl"]),
            tuple(sorted(row["thresholds"].items())),
        ),
    )[0]


def _up_entry_count(row: dict[str, Any]) -> int:
    return int(row.get("side_distribution", {}).get("UP", 0))


def _down_entry_count(row: dict[str, Any]) -> int:
    return int(row.get("side_distribution", {}).get("DOWN", 0))


def _side_coverage_interpretation(
    *,
    row_count: int,
    rows_with_both_count: int,
    total_up: int,
    total_down: int,
    p_up_alignment_blocked_count: int,
) -> str:
    if row_count == 0:
        return "insufficient_two_sided_opportunities"
    if rows_with_both_count > 0:
        return "insufficient_two_sided_opportunities"
    if p_up_alignment_blocked_count > 0 and (total_up == 0 or total_down == 0):
        return "p_up_alignment_blocks_one_side"
    if total_up == 0 or total_down == 0:
        return "one_sided_model_selection"
    return "insufficient_two_sided_opportunities"


def _counter_distribution(values: Any) -> dict[str, int]:
    counts = Counter(values)
    return {
        str(value): count
        for value, count in sorted(counts.items(), key=lambda item: (item[0], item[1]))
    }


def _gate_pass_rate(rows: list[dict[str, Any]], gate_name: str) -> float:
    if not rows:
        return 0.0
    return sum(
        1 for row in rows if row["validation_gate_results"][gate_name]["passed"]
    ) / len(rows)


def _threshold_grouping(
    rows: list[dict[str, Any]],
    parameter: str,
) -> list[dict[str, Any]]:
    grouped: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        value = float(row["thresholds"].get(parameter, 0.0))
        grouped.setdefault(value, []).append(row)
    payloads = []
    for value, group_rows in grouped.items():
        best = _rank_rows(group_rows)[0]
        payloads.append(
            {
                "threshold_parameter": parameter,
                "threshold_value": value,
                "row_count": len(group_rows),
                "mean_entry_count": sum(
                    int(row["entry_count"]) for row in group_rows
                )
                / len(group_rows),
                "max_entry_count": max(int(row["entry_count"]) for row in group_rows),
                "mean_total_pnl": sum(
                    float(row["total_pnl"]) for row in group_rows
                )
                / len(group_rows),
                "max_total_pnl": max(float(row["total_pnl"]) for row in group_rows),
                "positive_pnl_rate": sum(
                    1 for row in group_rows if float(row["total_pnl"]) > 0.0
                )
                / len(group_rows),
                "support_pass_rate": sum(
                    1 for row in group_rows if _support_gates_passed(row)
                )
                / len(group_rows),
                "side_coverage_pass_rate": sum(
                    1
                    for row in group_rows
                    if row["validation_gate_results"]["side_coverage_gate"][
                        "passed"
                    ]
                )
                / len(group_rows),
                "best_thresholds_for_value": best["thresholds"],
            }
        )
    return sorted(payloads, key=lambda row: float(row["threshold_value"]))


def _p_up_alignment_interpretation(
    *,
    pass_rate: float,
    blocked_positive_count: int,
    blocked_distribution: dict[str, int],
) -> str:
    blocked_rows = sum(
        count for value, count in blocked_distribution.items() if int(value) > 0
    )
    total_rows = sum(blocked_distribution.values())
    blocked_rate = 0.0 if total_rows == 0 else blocked_rows / total_rows
    if pass_rate >= 0.95 and blocked_rate < 0.50:
        return "p_up_alignment_not_primary_blocker"
    if blocked_rate >= 0.50 and blocked_positive_count > 0:
        return "p_up_alignment_over_filters"
    if pass_rate >= 0.95:
        return "p_up_alignment_helpful"
    if total_rows == 0:
        return "insufficient_evidence"
    return "p_up_alignment_not_primary_blocker"


def _side_distribution_by_threshold_family(
    rows: list[dict[str, Any]],
    limit: int = 20,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        thresholds = row["thresholds"]
        key = (
            float(thresholds.get("p_up_alignment_min", 0.0)),
            float(thresholds.get("min_calibrated_action_score", 0.0)),
            float(thresholds.get("min_best_action_margin", 0.0)),
        )
        grouped.setdefault(key, []).append(row)
    payloads = []
    for key, group_rows in grouped.items():
        total_up = sum(_up_entry_count(row) for row in group_rows)
        total_down = sum(_down_entry_count(row) for row in group_rows)
        payloads.append(
            {
                "p_up_alignment_min": key[0],
                "min_calibrated_action_score": key[1],
                "min_best_action_margin": key[2],
                "row_count": len(group_rows),
                "up_entry_count": total_up,
                "down_entry_count": total_down,
                "rows_with_both_sides": sum(
                    1
                    for row in group_rows
                    if _up_entry_count(row) > 0 and _down_entry_count(row) > 0
                ),
                "mean_total_pnl": sum(
                    float(row["total_pnl"]) for row in group_rows
                )
                / len(group_rows),
                "max_total_pnl": max(float(row["total_pnl"]) for row in group_rows),
            }
        )
    return sorted(
        payloads,
        key=lambda row: (
            -int(row["row_count"]),
            -float(row["max_total_pnl"]),
            float(row["p_up_alignment_min"]),
            float(row["min_calibrated_action_score"]),
            float(row["min_best_action_margin"]),
        ),
    )[:limit]


def _action_side_summary(
    rows: list[dict[str, Any]],
    side: str,
) -> dict[str, Any]:
    count_fn = _up_entry_count if side == "UP" else _down_entry_count
    rows_with_action = [row for row in rows if count_fn(row) > 0]
    return {
        "row_count_with_action": len(rows_with_action),
        "entry_count": sum(count_fn(row) for row in rows),
        "row_level_proxy_total_pnl_sum": sum(
            float(row["total_pnl"]) for row in rows_with_action
        ),
        "row_level_proxy_mean_pnl": 0.0
        if not rows_with_action
        else sum(float(row["total_pnl"]) for row in rows_with_action)
        / len(rows_with_action),
        "row_level_proxy_max_pnl": max(
            (float(row["total_pnl"]) for row in rows_with_action),
            default=0.0,
        ),
        "positive_pnl_row_count": sum(
            1 for row in rows_with_action if float(row["total_pnl"]) > 0.0
        ),
    }


def _exit_reliability_gate_pass_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    passed = sum(
        1
        for row in rows
        if row["validation_gate_results"]["residual_count_gate"]["passed"]
        and row["validation_gate_results"]["residual_settlement_drag_gate"]["passed"]
    )
    return passed / len(rows)


def _ordered_unique(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _gate_level_attribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    row_count = len(rows)
    if not rows:
        return []
    gate_names = list(rows[0]["validation_gate_results"])
    attribution = []
    for gate_name in gate_names:
        passed = sum(
            1 for row in rows if row["validation_gate_results"][gate_name]["passed"]
        )
        failed = row_count - passed
        attribution.append(
            {
                "gate_name": gate_name,
                "passed_row_count": passed,
                "failed_row_count": failed,
                "pass_rate": 0.0 if row_count == 0 else passed / row_count,
                "fail_rate": 0.0 if row_count == 0 else failed / row_count,
            }
        )
    return attribution


def _reason_code_attribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    row_count = len(rows)
    counts = Counter(
        reason for row in rows for reason in row["failed_reason_codes"]
    )
    return [
        {
            "reason_code": reason,
            "row_count": count,
            "row_rate": 0.0 if row_count == 0 else count / row_count,
        }
        for reason, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def _combination_attribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = "|".join(row["failed_reason_codes"]) or "none"
        grouped.setdefault(key, []).append(row)
    payloads = []
    for key, group_rows in grouped.items():
        best = _rank_rows(group_rows)[0]
        payloads.append(
            {
                "reason_code_combination": key,
                "row_count": len(group_rows),
                "best_total_pnl": best["total_pnl"],
                "_total_pnl": best["total_pnl"],
                "best_entry_count": best["entry_count"],
                "best_unique_market_count": best["unique_market_count"],
                "best_side_count": best["side_count"],
                "best_thresholds": best["thresholds"],
            }
        )
    return sorted(
        payloads,
        key=lambda row: (-int(row["row_count"]), str(row["reason_code_combination"])),
    )


def _near_miss_rows(rows: list[dict[str, Any]], limit_per_gate: int = 3) -> list[dict[str, Any]]:
    near_misses = []
    if not rows:
        return near_misses
    gate_names = list(rows[0]["validation_gate_results"])
    for gate_name in gate_names:
        candidates = [
            row for row in rows if not row["validation_gate_results"][gate_name]["passed"]
        ]
        for row in _rank_rows(candidates)[:limit_per_gate]:
            near_misses.append(
                {
                    "gate_name": gate_name,
                    **_row_core(row),
                    "near_miss_score": _near_miss_score(row),
                }
            )
    return near_misses


def _best_rows_by_objective(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
    positive_rows = [row for row in rows if float(row["total_pnl"]) > 0.0]
    zero_residual_positive_rows = [
        row
        for row in positive_rows
        if int(row["residual_count"]) == 0
        and float(row["replay_residual_settlement_drag"]) >= 0.0
    ]
    p_up_agreement_positive_rows = [
        row
        for row in positive_rows
        if bool(row["candidate_scoped_p_up_action_disagreement_within_limit"])
    ]
    return {
        "best_by_total_pnl": _best_or_none(rows, "highest_total_pnl"),
        "best_by_entry_count_with_positive_pnl": _best_or_none(
            positive_rows,
            "highest_entry_count_with_positive_pnl",
        ),
        "best_by_market_coverage_with_positive_pnl": _best_or_none(
            positive_rows,
            "highest_market_coverage_with_positive_pnl",
        ),
        "best_by_side_coverage_with_positive_pnl": _best_or_none(
            positive_rows,
            "highest_side_coverage_with_positive_pnl",
        ),
        "best_by_zero_residual_positive_pnl": _best_or_none(
            zero_residual_positive_rows,
            "zero_residual_positive_pnl",
        ),
        "best_by_p_up_agreement_positive_pnl": _best_or_none(
            p_up_agreement_positive_rows,
            "p_up_agreement_positive_pnl",
        ),
        "least_bad_overall_row": _best_or_none(rows, "least_bad_overall"),
    }


def _best_or_none(rows: list[dict[str, Any]], explanation: str) -> dict[str, Any] | None:
    if not rows:
        return None
    row = _rank_rows(rows)[0]
    return {**_row_core(row), "explanation": explanation}


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -float(row["total_pnl"]),
            -int(row["entry_count"]),
            -int(row["unique_market_count"]),
            -int(row["side_count"]),
            int(row["residual_count"]),
            float(row["candidate_scoped_p_up_action_disagreement_rate"]),
            tuple(sorted(row["thresholds"].items())),
        ),
    )


def _row_core(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "thresholds": row["thresholds"],
        "entry_count": row["entry_count"],
        "unique_market_count": row["unique_market_count"],
        "side_count": row["side_count"],
        "sell_count": row["sell_count"],
        "side_distribution": row["side_distribution"],
        "residual_count": row["residual_count"],
        "residual_settlement_drag": row["replay_residual_settlement_drag"],
        "total_pnl": row["total_pnl"],
        "mean_pnl_per_entry": row["mean_pnl_per_entry"],
        "max_drawdown": row["max_drawdown"],
        "p_up_disagreement_rate": row[
            "candidate_scoped_p_up_action_disagreement_rate"
        ],
        "failed_gates": row["failed_gates"],
        "failed_reason_codes": row["failed_reason_codes"],
    }


def _attribution_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **_row_core(row),
        "validation_gate_results": row["validation_gate_results"],
        "validation_support_gate_passed": row["validation_support_gate_passed"],
    }


def _near_miss_score(row: dict[str, Any]) -> float:
    passed_gate_count = sum(
        1 for gate in row["validation_gate_results"].values() if gate["passed"]
    )
    return (
        passed_gate_count
        + float(row["total_pnl"])
        + int(row["entry_count"]) * 0.001
        + int(row["unique_market_count"]) * 0.001
        + int(row["side_count"]) * 0.001
    )


def _failure_interpretation(
    *,
    validation_rows: list[dict[str, Any]],
    gate_attribution: list[dict[str, Any]],
    reason_attribution: list[dict[str, Any]],
    passing_row_count: int,
) -> str:
    if not validation_rows:
        return "insufficient_evidence"
    if passing_row_count > 0:
        return "mixed_threshold_failure"
    reason_counts = {
        row["reason_code"]: int(row["row_count"]) for row in reason_attribution
    }
    row_count = len(validation_rows)
    positive_rows = [row for row in validation_rows if float(row["total_pnl"]) > 0.0]
    support_reason_total = sum(
        reason_counts.get(reason, 0)
        for reason in (
            "validation_entry_support_insufficient",
            "validation_market_support_insufficient",
            "validation_sell_support_insufficient",
        )
    )
    if support_reason_total >= row_count:
        return "support_too_sparse"
    if reason_counts.get("validation_side_coverage_insufficient", 0) >= row_count * 0.5:
        return "one_sided_support"
    if positive_rows and all(
        any(
            reason in row["failed_reason_codes"]
            for reason in (
                "validation_entry_support_insufficient",
                "validation_market_support_insufficient",
                "validation_sell_support_insufficient",
                "validation_side_coverage_insufficient",
            )
        )
        for row in positive_rows
    ):
        return "positive_pnl_only_at_low_support"
    if reason_counts.get("validation_residual_positions_remaining", 0) >= row_count * 0.5:
        return "exit_reliability_failure"
    if reason_counts.get("validation_p_up_disagreement_excessive", 0) >= row_count * 0.5:
        return "p_up_alignment_over_filters"
    if reason_counts.get("validation_total_pnl_not_positive", 0) >= row_count * 0.5:
        return "pnl_not_positive_under_support"
    return "mixed_threshold_failure"


def _recommended_next_action(interpretation: str) -> str:
    if interpretation in {"support_too_sparse", "one_sided_support"}:
        return "collect_more_real_corpus"
    if interpretation in {
        "positive_pnl_only_at_low_support",
        "mixed_threshold_failure",
        "p_up_alignment_over_filters",
    }:
        return "expand_validation_grid_without_shadow_fit"
    if interpretation == "exit_reliability_failure":
        return "relax_candidate_defaults_only_after_validation_evidence"
    if interpretation == "pnl_not_positive_under_support":
        return "revise_action_value_ranking_model"
    return "keep_blocked"


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
    support_gate_passed: bool,
    support_gate_reason_codes: list[str],
    promotion_support_eligible: bool,
) -> dict[str, Any]:
    row = {
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
        "support_gate_passed": support_gate_passed,
        "support_gate_reason_codes": support_gate_reason_codes,
        "promotion_support_eligible": promotion_support_eligible,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        **compact_safety_fields(),
    }
    gate_results = _validation_gate_results(
        entry_count=row["entry_count"],
        unique_market_count=row["unique_market_count"],
        side_count=row["side_count"],
        sell_count=row["sell_count"],
        residual_count=row["residual_count"],
        residual_drag=row["replay_residual_settlement_drag"],
        total_pnl=row["total_pnl"],
        mean_pnl_per_entry=row["mean_pnl_per_entry"],
        p_up_disagreement_rate=row[
            "candidate_scoped_p_up_action_disagreement_rate"
        ],
    )
    gate_reason_codes = _reason_codes_from_gate_results(gate_results)
    row["validation_gate_results"] = gate_results
    row["failed_gates"] = [
        name for name, gate in gate_results.items() if not gate["passed"]
    ]
    row["failed_reason_codes"] = gate_reason_codes
    row["validation_support_gate_passed"] = not gate_reason_codes
    row["validation_support_gate_reason_codes"] = gate_reason_codes
    return row


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
    return _reason_codes_from_gate_results(
        _validation_gate_results(
            entry_count=entry_count,
            unique_market_count=unique_market_count,
            side_count=side_count,
            sell_count=sell_count,
            residual_count=residual_count,
            residual_drag=residual_drag,
            total_pnl=total_pnl,
            mean_pnl_per_entry=mean_pnl_per_entry,
            p_up_disagreement_rate=p_up_disagreement_rate,
        )
    )


def _validation_gate_results(
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
) -> dict[str, dict[str, Any]]:
    gates = SELL_BEFORE_CLOSE_SUPPORT_AWARE_VALIDATION_GATES
    return {
        "entry_support_gate": _gate(
            actual=entry_count,
            required=f">={int(gates['min_entry_count'])}",
            passed=entry_count >= int(gates["min_entry_count"]),
            reason_code=VALIDATION_GATE_REASON_CODES["entry_support_gate"],
        ),
        "market_support_gate": _gate(
            actual=unique_market_count,
            required=f">={int(gates['min_unique_market_count'])}",
            passed=unique_market_count >= int(gates["min_unique_market_count"]),
            reason_code=VALIDATION_GATE_REASON_CODES["market_support_gate"],
        ),
        "side_coverage_gate": _gate(
            actual=side_count,
            required=f">={int(gates['min_side_count'])}",
            passed=side_count >= int(gates["min_side_count"]),
            reason_code=VALIDATION_GATE_REASON_CODES["side_coverage_gate"],
        ),
        "sell_support_gate": _gate(
            actual=sell_count,
            required=f">={int(gates['min_sell_count'])}",
            passed=sell_count >= int(gates["min_sell_count"]),
            reason_code=VALIDATION_GATE_REASON_CODES["sell_support_gate"],
        ),
        "residual_count_gate": _gate(
            actual=residual_count,
            required=f"<={int(gates['max_residual_count'])}",
            passed=residual_count <= int(gates["max_residual_count"]),
            reason_code=VALIDATION_GATE_REASON_CODES["residual_count_gate"],
        ),
        "residual_settlement_drag_gate": _gate(
            actual=residual_drag,
            required=f">={float(gates['min_residual_settlement_drag'])}",
            passed=residual_drag >= float(gates["min_residual_settlement_drag"]),
            reason_code=VALIDATION_GATE_REASON_CODES[
                "residual_settlement_drag_gate"
            ],
        ),
        "total_pnl_gate": _gate(
            actual=total_pnl,
            required=f">{float(gates['min_total_pnl_exclusive'])}",
            passed=total_pnl > float(gates["min_total_pnl_exclusive"]),
            reason_code=VALIDATION_GATE_REASON_CODES["total_pnl_gate"],
        ),
        "mean_pnl_per_entry_gate": _gate(
            actual=mean_pnl_per_entry,
            required=f">{float(gates['min_mean_pnl_per_entry_exclusive'])}",
            passed=mean_pnl_per_entry
            > float(gates["min_mean_pnl_per_entry_exclusive"]),
            reason_code=VALIDATION_GATE_REASON_CODES["mean_pnl_per_entry_gate"],
        ),
        "p_up_disagreement_gate": _gate(
            actual=p_up_disagreement_rate,
            required=f"<={float(gates['max_p_up_disagreement_rate'])}",
            passed=p_up_disagreement_rate
            <= float(gates["max_p_up_disagreement_rate"]),
            reason_code=VALIDATION_GATE_REASON_CODES["p_up_disagreement_gate"],
        ),
    }


def _gate(
    *,
    actual: int | float,
    required: str,
    passed: bool,
    reason_code: str,
) -> dict[str, Any]:
    return {
        "actual": actual,
        "required": required,
        "passed": passed,
        "reason_code": None if passed else reason_code,
    }


def _reason_codes_from_gate_results(
    gate_results: dict[str, dict[str, Any]],
) -> list[str]:
    return sorted(
        gate["reason_code"]
        for gate in gate_results.values()
        if not gate["passed"] and gate["reason_code"]
    )


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
