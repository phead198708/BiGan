"""Ranking diagnostics and fail-closed source eligibility reports."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Any

from bigan.v8.polymarket.action_value_guards import (
    action_value_action_family,
    action_value_bucket_payload,
    action_value_fine_action_family,
)
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.action_family_eligibility import (
    build_action_family_eligibility_report,
)
from bigan.v8.polymarket.training.action_value_calibration import (
    ACTION_VALUE_HIGH_SCORE_MIN_SUPPORT,
    ACTION_VALUE_QUALITY_MAE_TOLERANCE,
)
from bigan.v8.polymarket.training.contracts import (
    ACTION_VALUE_LABEL_ACTIONS,
    POLYMARKET_POLICY_SCHEMA_VERSION,
    POLYMARKET_POLICY_TRAINING_PHASE,
    PolymarketPolicyExample,
    PolymarketPolicyPrediction,
    compact_safety_fields,
)
from bigan.v8.polymarket.training.sell_before_close_source_candidates import (
    SELL_BEFORE_CLOSE_DISABLED_SOURCE_CANDIDATE_ACTIONS,
    SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_CANDIDATE_NAME,
    SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_EXIT_POLICY,
    SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_THRESHOLDS,
    SELL_BEFORE_CLOSE_ONLY_SOURCE_CANDIDATE_ACTIONS,
    SELL_BEFORE_CLOSE_ONLY_SOURCE_CANDIDATE_NAME,
    SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_CANDIDATE_NAME,
    SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_THRESHOLDS,
    SELL_BEFORE_CLOSE_SUPPORT_AWARE_P_UP_ALIGNED_CANDIDATE_NAME,
)

MODEL_RANKING_ERROR_SCHEMA_VERSION = (
    "bigan-v8-polymarket-model-ranking-error-v1"
)
MODEL_RANKING_CANDIDATE_COMPARISON_SCHEMA_VERSION = (
    "bigan-v8-polymarket-model-ranking-candidate-comparison-v1"
)
SOURCE_MODEL_ELIGIBILITY_SCHEMA_VERSION = (
    "bigan-v8-polymarket-source-model-eligibility-v1"
)
RANKING_OVERLAY_ZERO_ENTRY_DIAGNOSTIC_SCHEMA_VERSION = (
    "bigan-v8-polymarket-ranking-overlay-zero-entry-diagnostic-v1"
)
ACTION_REPRESENTATION_DIAGNOSTIC_SCHEMA_VERSION = (
    "bigan-v8-polymarket-action-representation-diagnostic-v1"
)
BEST_ACTION_CONCENTRATION_FAIL_THRESHOLD = 0.95
P_UP_ACTION_DISAGREEMENT_FAIL_THRESHOLD = 0.50
P_UP_MATERIAL_DISAGREEMENT_THRESHOLD = 0.55
RANKING_OVERLAY_G_MIN_BUCKET_SUPPORT = 10
RANKING_OVERLAY_H_MIN_BUCKET_SUPPORT = 3
RANKING_OVERLAY_MIN_FAMILY_SUPPORT = 10
RANKING_OVERLAY_SHRINKAGE_PRIOR_SUPPORT = 10
RANKING_OVERLAY_SHRINKAGE_PRIOR_MEAN = 0.0
RANKING_OVERLAY_H_BUCKET_EVIDENCE_WEIGHT = 0.90
RANKING_OVERLAY_H_MODEL_SCORE_WEIGHT = 0.10
RANKING_OVERLAY_G_MODEL_SCORE_TIEBREAKER_WEIGHT = 0.001
RANKING_OVERLAY_DIAGNOSTIC_MIN_BUCKET_SUPPORT_VALUES = (3, 5, 10)
RANKING_OVERLAY_DIAGNOSTIC_SHRINKAGE_PRIOR_SUPPORT_VALUES = (5, 10, 20)
RANKING_OVERLAY_DIAGNOSTIC_BUFFER_MULTIPLIERS = (0.0, 0.5, 1.0)
ACTION_REPRESENTATION_MIN_BUCKET_SUPPORT = 10
def build_model_ranking_error_report(
    *,
    validation_examples: tuple[PolymarketPolicyExample, ...],
    validation_predictions: tuple[PolymarketPolicyPrediction, ...],
    shadow_examples: tuple[PolymarketPolicyExample, ...],
    shadow_predictions: tuple[PolymarketPolicyPrediction, ...],
) -> dict[str, Any]:
    """Build split-level ranking error diagnostics for calibrated action scores."""

    validation = _ranking_split_report(
        split_name="validation",
        examples=validation_examples,
        predictions=validation_predictions,
    )
    shadow = _ranking_split_report(
        split_name="shadow",
        examples=shadow_examples,
        predictions=shadow_predictions,
    )
    report = {
        "schema_version": MODEL_RANKING_ERROR_SCHEMA_VERSION,
        "policy_schema_version": POLYMARKET_POLICY_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "diagnostic_splits": ["validation", "shadow"],
        "calibrated_score_field": "calibrated_expected_pnl_per_notional_by_action",
        "realized_target_field": "action_return_targets",
        "validation": validation,
        "shadow": shadow,
        "required_breakdowns": [
            "action_family",
            "side",
            "price_bucket",
            "time_to_close_bucket",
            "raw_score_bucket",
            "market_family",
        ],
        **compact_safety_fields(),
    }
    report["model_ranking_error_report_id"] = canonical_json_sha256(report)
    return report


def build_model_ranking_candidate_comparison(
    *,
    validation_examples: tuple[PolymarketPolicyExample, ...],
    raw_validation_predictions: tuple[PolymarketPolicyPrediction, ...],
    calibrated_validation_predictions: tuple[PolymarketPolicyPrediction, ...],
    shadow_examples: tuple[PolymarketPolicyExample, ...],
    raw_shadow_predictions: tuple[PolymarketPolicyPrediction, ...],
    calibrated_shadow_predictions: tuple[PolymarketPolicyPrediction, ...],
    execution_buffer: float,
) -> dict[str, Any]:
    """Evaluate deterministic post-model ranking/calibration candidates."""

    _validate_aligned(validation_examples, raw_validation_predictions)
    _validate_aligned(validation_examples, calibrated_validation_predictions)
    _validate_aligned(shadow_examples, raw_shadow_predictions)
    _validate_aligned(shadow_examples, calibrated_shadow_predictions)

    candidate_specs = _candidate_specs(
        validation_examples=validation_examples,
        raw_validation_predictions=raw_validation_predictions,
        calibrated_validation_predictions=calibrated_validation_predictions,
        execution_buffer=execution_buffer,
    )
    candidates = []
    for spec in candidate_specs:
        validation_predictions = _apply_candidate_spec(
            predictions=raw_validation_predictions,
            fallback_predictions=calibrated_validation_predictions,
            spec=spec,
        )
        shadow_predictions = _apply_candidate_spec(
            predictions=raw_shadow_predictions,
            fallback_predictions=calibrated_shadow_predictions,
            spec=spec,
        )
        candidates.append(
            _candidate_report(
                spec=spec,
                validation_examples=validation_examples,
                validation_predictions=validation_predictions,
                shadow_examples=shadow_examples,
                raw_shadow_predictions=raw_shadow_predictions,
                shadow_predictions=shadow_predictions,
                execution_buffer=execution_buffer,
            )
        )
    eligible_candidates = [
        candidate
        for candidate in candidates
        if candidate["source_model_candidate_eligible"]
    ]
    source_model_candidate_eligible = bool(eligible_candidates)
    report = {
        "schema_version": MODEL_RANKING_CANDIDATE_COMPARISON_SCHEMA_VERSION,
        "policy_schema_version": POLYMARKET_POLICY_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible_candidates),
        "candidate_names": [candidate["candidate_name"] for candidate in candidates],
        "best_candidate_name": _best_candidate(candidates)["candidate_name"],
        "best_candidate_source_model_eligible": _best_candidate(candidates)[
            "source_model_candidate_eligible"
        ],
        "source_model_candidate_eligible": source_model_candidate_eligible,
        "requires_promotion_replay_gate": True,
        "paper_run_resume_allowed": False,
        "paper_run_resume_blocked_reason": "promotion_replay_gate_required",
        "no_candidate_eligible": not eligible_candidates,
        "no_candidate_eligible_reason_codes": sorted(
            {
                reason
                for candidate in candidates
                for reason in candidate["ineligible_reason_codes"]
            }
        ),
        "candidates": candidates,
        **compact_safety_fields(),
    }
    report["model_ranking_candidate_comparison_id"] = canonical_json_sha256(report)
    return report


def build_ranking_overlay_zero_entry_diagnostic_report(
    *,
    validation_examples: tuple[PolymarketPolicyExample, ...],
    raw_validation_predictions: tuple[PolymarketPolicyPrediction, ...],
    calibrated_validation_predictions: tuple[PolymarketPolicyPrediction, ...],
    shadow_examples: tuple[PolymarketPolicyExample, ...],
    raw_shadow_predictions: tuple[PolymarketPolicyPrediction, ...],
    calibrated_shadow_predictions: tuple[PolymarketPolicyPrediction, ...],
    execution_buffer: float,
) -> dict[str, Any]:
    """Explain why validation-fitted G/H overlays produce zero entries."""

    _validate_aligned(validation_examples, raw_validation_predictions)
    _validate_aligned(validation_examples, calibrated_validation_predictions)
    _validate_aligned(shadow_examples, raw_shadow_predictions)
    _validate_aligned(shadow_examples, calibrated_shadow_predictions)

    overlay_specs = [
        spec
        for spec in _candidate_specs(
            validation_examples=validation_examples,
            raw_validation_predictions=raw_validation_predictions,
            calibrated_validation_predictions=calibrated_validation_predictions,
            execution_buffer=execution_buffer,
        )
        if spec.get("ranking_overlay") is not None
    ]
    candidates = [
        _zero_entry_candidate_report(
            spec=spec,
            raw_shadow_predictions=raw_shadow_predictions,
            calibrated_shadow_predictions=calibrated_shadow_predictions,
        )
        for spec in overlay_specs
    ]
    diagnostic_sweeps = _overlay_diagnostic_sweeps(
        validation_examples=validation_examples,
        calibrated_validation_predictions=calibrated_validation_predictions,
        shadow_examples=shadow_examples,
        raw_shadow_predictions=raw_shadow_predictions,
        calibrated_shadow_predictions=calibrated_shadow_predictions,
        execution_buffer=execution_buffer,
    )
    report = {
        "schema_version": RANKING_OVERLAY_ZERO_ENTRY_DIAGNOSTIC_SCHEMA_VERSION,
        "policy_schema_version": POLYMARKET_POLICY_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "diagnostic_only": True,
        "promotion_evidence_eligible": False,
        "source_model_candidate_eligible": False,
        "requires_promotion_replay_gate": True,
        "paper_run_resume_allowed": False,
        "paper_run_resume_blocked_reason": "promotion_replay_gate_required",
        "fit_split": "validation",
        "evaluation_split": "shadow",
        "uses_shadow_for_fit": False,
        "execution_buffer": execution_buffer,
        "candidate_count": len(candidates),
        "candidate_names": [candidate["candidate_name"] for candidate in candidates],
        "candidates": candidates,
        "diagnostic_sweeps": diagnostic_sweeps,
        "diagnostic_sweep_settings": {
            "min_bucket_support_values": list(
                RANKING_OVERLAY_DIAGNOSTIC_MIN_BUCKET_SUPPORT_VALUES
            ),
            "shrinkage_prior_support_values": list(
                RANKING_OVERLAY_DIAGNOSTIC_SHRINKAGE_PRIOR_SUPPORT_VALUES
            ),
            "buffer_multipliers": list(RANKING_OVERLAY_DIAGNOSTIC_BUFFER_MULTIPLIERS),
        },
        "notes": [
            "diagnostic-only artifact; never authorizes #134",
            "sweeps are sensitivity checks and cannot promote a source model",
        ],
        **compact_safety_fields(),
    }
    report["ranking_overlay_zero_entry_diagnostic_report_id"] = (
        canonical_json_sha256(report)
    )
    return report


def build_action_representation_diagnostic_report(
    *,
    validation_examples: tuple[PolymarketPolicyExample, ...],
    validation_predictions: tuple[PolymarketPolicyPrediction, ...],
    shadow_examples: tuple[PolymarketPolicyExample, ...],
    shadow_predictions: tuple[PolymarketPolicyPrediction, ...],
    execution_buffer: float,
) -> dict[str, Any]:
    """Diagnose fine action-family, label, and feature representation quality."""

    _validate_aligned(validation_examples, validation_predictions)
    _validate_aligned(shadow_examples, shadow_predictions)
    split_reports = {
        "validation": _action_representation_split_report(
            split_name="validation",
            examples=validation_examples,
            predictions=validation_predictions,
            execution_buffer=execution_buffer,
        ),
        "shadow": _action_representation_split_report(
            split_name="shadow",
            examples=shadow_examples,
            predictions=shadow_predictions,
            execution_buffer=execution_buffer,
        ),
    }
    validation_sell = split_reports["validation"]["sell_before_close_summary"]
    shadow_sell = split_reports["shadow"]["sell_before_close_summary"]
    label_exit_path_assessment = _sell_before_close_label_exit_path_assessment(
        examples=(*validation_examples, *shadow_examples),
    )
    report = {
        "schema_version": ACTION_REPRESENTATION_DIAGNOSTIC_SCHEMA_VERSION,
        "policy_schema_version": POLYMARKET_POLICY_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "diagnostic_only": True,
        "promotion_evidence_eligible": False,
        "source_model_candidate_eligible": False,
        "requires_promotion_replay_gate": True,
        "paper_run_resume_allowed": False,
        "paper_run_resume_blocked_reason": "promotion_replay_gate_required",
        "execution_buffer": execution_buffer,
        "min_positive_bucket_support": ACTION_REPRESENTATION_MIN_BUCKET_SUPPORT,
        "fine_action_family_definition": (
            "side|intended_exit_policy|price_bucket|time_to_close_bucket"
        ),
        "label_exit_path_assessment": label_exit_path_assessment,
        "validation": split_reports["validation"],
        "shadow": split_reports["shadow"],
        "sell_before_close_overall": {
            "validation_mean": validation_sell["realized_return_mean"],
            "validation_sum": validation_sell["realized_return_sum"],
            "validation_support": validation_sell["support_count"],
            "shadow_mean": shadow_sell["realized_return_mean"],
            "shadow_sum": shadow_sell["realized_return_sum"],
            "shadow_support": shadow_sell["support_count"],
        },
        "needs_more_sell_before_close_positive_bucket_data": (
            split_reports["validation"][
                "supported_positive_sell_before_close_bucket_count"
            ]
            == 0
            or split_reports["shadow"][
                "supported_positive_sell_before_close_bucket_count"
            ]
            == 0
        ),
        "notes": [
            "diagnostic-only artifact; does not change production gates",
            "fine families are causal feature buckets and use no future data",
            "label review compares theoretical terminal bid and executable exit path",
        ],
        **compact_safety_fields(),
    }
    report["action_representation_diagnostic_report_id"] = canonical_json_sha256(
        report
    )
    return report


def build_source_model_eligibility_report(
    *,
    signal_sanity: dict[str, Any],
    action_value_calibration: dict[str, Any],
    action_family_eligibility: dict[str, Any],
    model_ranking_candidate_comparison: dict[str, Any],
) -> dict[str, Any]:
    """Summarize strict source-model eligibility without relaxing hard gates."""

    source_model_eligible = bool(signal_sanity["action_value_paper_decision_eligible"])
    candidate_eligible = [
        candidate
        for candidate in model_ranking_candidate_comparison["candidates"]
        if candidate["source_model_candidate_eligible"]
    ]
    report = {
        "schema_version": SOURCE_MODEL_ELIGIBILITY_SCHEMA_VERSION,
        "policy_schema_version": POLYMARKET_POLICY_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "source_model_eligible": source_model_eligible,
        "source_model_candidate_eligible": bool(candidate_eligible),
        "requires_promotion_replay_gate": True,
        "source_model_ineligible_reason_codes": signal_sanity[
            "action_value_paper_decision_ineligible_reasons"
        ],
        "hard_gates": {
            "calibration_support_passed": signal_sanity[
                "calibration_support_passed"
            ],
            "calibration_quality_passed": signal_sanity[
                "calibration_quality_passed"
            ],
            "action_family_paper_decision_eligible": signal_sanity[
                "action_family_paper_decision_eligible"
            ],
            "best_action_concentration_passed": signal_sanity[
                "best_action_concentration_passed"
            ],
            "p_up_action_disagreement_within_limit": signal_sanity[
                "p_up_action_disagreement_within_limit"
            ],
            "action_value_paper_decision_eligible": signal_sanity[
                "action_value_paper_decision_eligible"
            ],
        },
        "shadow_raw_mae": action_value_calibration["shadow_mae_comparison"][
            "raw_mae"
        ],
        "shadow_calibrated_mae": action_value_calibration["shadow_mae_comparison"][
            "bucketed_calibrated_mae"
        ],
        "high_score_support_count": action_family_eligibility[
            "high_score_support_count"
        ],
        "high_score_realized_return_mean": action_family_eligibility[
            "high_score_realized_return_mean"
        ],
        "high_score_realized_return_sum": action_family_eligibility[
            "high_score_realized_return_sum"
        ],
        "action_family_gates": action_family_eligibility[
            "action_family_gate_results"
        ],
        "candidate_count": model_ranking_candidate_comparison["candidate_count"],
        "eligible_candidate_count": len(candidate_eligible),
        "candidate_source_model_eligible": bool(candidate_eligible),
        "candidate_names": model_ranking_candidate_comparison["candidate_names"],
        "candidate_scoped_eligibility_summary": [
            _source_candidate_summary(candidate)
            for candidate in model_ranking_candidate_comparison["candidates"]
        ],
        "best_candidate_name": model_ranking_candidate_comparison[
            "best_candidate_name"
        ],
        "best_candidate_source_model_eligible": (
            model_ranking_candidate_comparison["best_candidate_source_model_eligible"]
        ),
        "no_candidate_eligible": model_ranking_candidate_comparison[
            "no_candidate_eligible"
        ],
        "no_candidate_eligible_reason_codes": (
            model_ranking_candidate_comparison[
                "no_candidate_eligible_reason_codes"
            ]
        ),
        "paper_run_resume_allowed": False,
        "paper_run_resume_blocked_reason": "promotion_replay_gate_required",
        "candidate_artifacts": [],
        **compact_safety_fields(),
    }
    report["source_model_eligibility_report_id"] = canonical_json_sha256(report)
    return report


def _source_candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "candidate_name",
        "enabled_action_families",
        "disabled_action_families",
        "enabled_actions",
        "disabled_actions",
        "exit_reliability_guard_enabled",
        "p_up_side_alignment_filter_enabled",
        "exit_policy",
        "entry_filter_thresholds",
        "threshold_selection_method",
        "threshold_selection_fit_split",
        "threshold_selection_evaluation_split",
        "uses_shadow_for_fit",
        "shadow_sweep_not_used_for_threshold_fit",
        "support_aware_threshold_selection_failed",
        "support_aware_threshold_selection_reason_codes",
        "entry_decision_count_before_guard",
        "entry_decision_count_after_exit_guard",
        "entry_decision_count_after_p_up_alignment",
        "entry_decision_count_after_guard",
        "entry_filter_blocked_count",
        "entry_filter_blocked_by_p_up_alignment_count",
        "entry_filter_blocked_by_quality_count",
        "reentry_cooldown_seconds",
        "reentry_blocked_count",
        "entries_per_market_distribution",
        "positions_opened_count",
        "positions_closed_before_settlement_count",
        "positions_opened_but_not_closed_before_settlement",
        "replay_realized_trade_pnl",
        "replay_settlement_pnl",
        "replay_total_polymarket_pnl",
        "replay_residual_settlement_drag",
        "replay_total_pnl_improved_vs_i_candidate",
        "promotion_support_eligible",
        "promotion_support_gate_passed",
        "promotion_support_reason_codes",
        "promotion_support_thresholds",
        "promotion_replay_entry_decision_count",
        "promotion_replay_sell_decision_count",
        "promotion_replay_unique_market_count",
        "promotion_replay_side_count",
        "promotion_replay_side_distribution",
        "promotion_replay_mean_pnl_per_entry",
        "candidate_scoped_p_up_action_disagreement_rate",
        "candidate_scoped_p_up_action_disagreement_within_limit",
        "candidate_scoped_action_family_gate_results",
        "candidate_scoped_high_score_support_count",
        "candidate_scoped_high_score_realized_return_mean",
        "candidate_scoped_high_score_realized_return_sum",
        "source_model_candidate_eligible",
        "ineligible_reason_codes",
    )
    return {field: candidate.get(field) for field in fields}


def model_ranking_error_markdown(report: dict[str, Any]) -> str:
    """Render a compact ranking-error report."""

    lines = [
        "# Model Ranking Error Report",
        "",
        f"- schema_version: `{report['schema_version']}`",
        f"- paper_only: `{str(report['paper_only']).lower()}`",
        "",
        "| split | rows | top1 | top2 | top3 | mean_regret | selected_return | oracle_return |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name in ("validation", "shadow"):
        split = report[split_name]
        lines.append(
            "| {split} | {rows} | {top1:.4f} | {top2:.4f} | {top3:.4f} | "
            "{regret:.6f} | {selected:.6f} | {oracle:.6f} |".format(
                split=split_name,
                rows=split["sample_count"],
                top1=split["top_1_action_hit_rate"],
                top2=split["top_2_action_hit_rate"],
                top3=split["top_3_action_hit_rate"],
                regret=split["mean_regret"],
                selected=split["mean_selected_action_realized_return"],
                oracle=split["mean_oracle_best_action_realized_return"],
            )
        )
    return "\n".join(lines) + "\n"


def model_ranking_candidate_comparison_markdown(report: dict[str, Any]) -> str:
    """Render candidate comparison summary markdown."""

    lines = [
        "# Model Ranking Candidate Comparison",
        "",
        f"- schema_version: `{report['schema_version']}`",
        f"- candidate_count: `{report['candidate_count']}`",
        f"- eligible_candidate_count: `{report['eligible_candidate_count']}`",
        f"- best_candidate_name: `{report['best_candidate_name']}`",
        f"- no_candidate_eligible: `{str(report['no_candidate_eligible']).lower()}`",
        "",
        "| candidate | source_eligible | enabled_families | scoped_p_up_disagreement | high_score_support | high_score_mean | high_score_sum | reasons |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for candidate in report["candidates"]:
        reasons = ", ".join(candidate["ineligible_reason_codes"]) or "none"
        lines.append(
            "| {name} | {eligible} | {families} | {p_up:.6f} | {support} | "
            "{mean:.6f} | {total:.6f} | {reasons} |".format(
                name=candidate["candidate_name"],
                eligible=str(candidate["source_model_eligible"]).lower(),
                families=", ".join(candidate["enabled_action_families"]) or "none",
                p_up=candidate["candidate_scoped_p_up_action_disagreement_rate"],
                support=candidate["high_score_support_count"],
                mean=candidate["high_score_realized_return_mean"],
                total=candidate["high_score_realized_return_sum"],
                reasons=reasons,
            )
        )
    return "\n".join(lines) + "\n"


def ranking_overlay_zero_entry_diagnostic_markdown(report: dict[str, Any]) -> str:
    """Render compact zero-entry diagnostics for G/H overlays."""

    lines = [
        "# Ranking Overlay Zero-Entry Diagnostic",
        "",
        f"- schema_version: `{report['schema_version']}`",
        f"- diagnostic_only: `{str(report['diagnostic_only']).lower()}`",
        f"- promotion_evidence_eligible: `{str(report['promotion_evidence_eligible']).lower()}`",
        f"- source_model_candidate_eligible: `{str(report['source_model_candidate_eligible']).lower()}`",
        f"- paper_run_resume_allowed: `{str(report['paper_run_resume_allowed']).lower()}`",
        "",
        "| candidate | predictions | actions | no_trade_selected | selected_non_no_trade | passed_actions | bucket_support_failed | family_support_failed | bucket_metric_failed | family_metric_failed | bucket_sum_failed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in report["candidates"]:
        lines.append(
            "| {name} | {predictions} | {actions} | {no_trade} | {selected} | "
            "{passed} | {bucket_support} | {family_support} | "
            "{bucket_metric} | {family_metric} | {bucket_sum} |".format(
                name=candidate["candidate_name"],
                predictions=candidate["prediction_count"],
                actions=candidate["action_count_considered"],
                no_trade=candidate["no_trade_selected_count"],
                selected=candidate["selected_non_no_trade_count"],
                passed=candidate["passed_bucket_and_family_count"],
                bucket_support=candidate["bucket_support_failed_count"],
                family_support=candidate["family_support_failed_count"],
                bucket_metric=candidate["bucket_lcb_or_mean_failed_count"],
                family_metric=candidate["family_lcb_or_mean_failed_count"],
                bucket_sum=candidate["bucket_sum_failed_count"],
            )
        )
    lines.extend(
        [
            "",
            "Diagnostic sweeps are non-promotion sensitivity checks only.",
            "",
            "| candidate | rows | max_selected_non_no_trade | max_high_score_support | best_high_score_mean |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for candidate_name in report["candidate_names"]:
        rows = [
            row
            for row in report["diagnostic_sweeps"]
            if row["candidate_name"] == candidate_name
        ]
        lines.append(
            "| {name} | {rows} | {selected} | {support} | {mean:.6f} |".format(
                name=candidate_name,
                rows=len(rows),
                selected=max(
                    (row["selected_non_no_trade_count"] for row in rows),
                    default=0,
                ),
                support=max(
                    (row["shadow_high_score_support"] for row in rows),
                    default=0,
                ),
                mean=max(
                    (row["shadow_high_score_mean"] for row in rows),
                    default=0.0,
                ),
            )
        )
    return "\n".join(lines) + "\n"


def action_representation_diagnostic_markdown(report: dict[str, Any]) -> str:
    """Render compact action representation diagnostics."""

    sell = report["sell_before_close_overall"]
    lines = [
        "# Action Representation Diagnostic",
        "",
        f"- schema_version: `{report['schema_version']}`",
        f"- diagnostic_only: `{str(report['diagnostic_only']).lower()}`",
        f"- source_model_candidate_eligible: `{str(report['source_model_candidate_eligible']).lower()}`",
        f"- paper_run_resume_allowed: `{str(report['paper_run_resume_allowed']).lower()}`",
        f"- fine_action_family_definition: `{report['fine_action_family_definition']}`",
        "- sell_before_close_exit_path_coarse: "
        f"`{str(report['label_exit_path_assessment']['sell_before_close_exit_path_coarse']).lower()}`",
        "",
        "## SELL_BEFORE_CLOSE Overall",
        "",
        "| split | support | mean | sum | positive_supported_buckets |",
        "|---|---:|---:|---:|---:|",
        "| validation | {support} | {mean:.6f} | {total:.6f} | {buckets} |".format(
            support=sell["validation_support"],
            mean=sell["validation_mean"],
            total=sell["validation_sum"],
            buckets=report["validation"][
                "supported_positive_sell_before_close_bucket_count"
            ],
        ),
        "| shadow | {support} | {mean:.6f} | {total:.6f} | {buckets} |".format(
            support=sell["shadow_support"],
            mean=sell["shadow_mean"],
            total=sell["shadow_sum"],
            buckets=report["shadow"][
                "supported_positive_sell_before_close_bucket_count"
            ],
        ),
        "",
        "## Top Negative SELL_BEFORE_CLOSE Buckets",
        "",
    ]
    for split_name in ("validation", "shadow"):
        lines.append(f"### {split_name}")
        for row in report[split_name]["sell_before_close_negative_contributors"][:5]:
            lines.append(
                "- "
                f"{row['fine_action_family']}: support={row['support_count']} "
                f"markets={row['unique_market_count']} "
                f"mean={row['realized_return_mean']} "
                f"sum={row['realized_return_sum']}"
            )
        lines.append("")
        lines.append("Top negative high-score examples:")
        for row in report[split_name][
            "top_negative_high_score_sell_before_close_examples"
        ][:5]:
            lines.append(
                "- "
                f"{row['decision_ts']} {row['action']} "
                f"{row['fine_action_family']} "
                f"score={row['calibrated_score']} "
                f"return={row['realized_return']}"
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


def source_model_eligibility_markdown(report: dict[str, Any]) -> str:
    """Render strict source eligibility report markdown."""

    reasons = ", ".join(report["source_model_ineligible_reason_codes"]) or "none"
    candidate_reasons = ", ".join(report["no_candidate_eligible_reason_codes"]) or "none"
    lines = [
        "# Source Model Eligibility Report",
        "",
        f"- schema_version: `{report['schema_version']}`",
        f"- source_model_eligible: `{str(report['source_model_eligible']).lower()}`",
        f"- source_model_candidate_eligible: `{str(report['source_model_candidate_eligible']).lower()}`",
        f"- requires_promotion_replay_gate: `{str(report['requires_promotion_replay_gate']).lower()}`",
        f"- source_model_ineligible_reason_codes: `{reasons}`",
        f"- eligible_candidate_count: `{report['eligible_candidate_count']}`",
        f"- no_candidate_eligible: `{str(report['no_candidate_eligible']).lower()}`",
        f"- no_candidate_eligible_reason_codes: `{candidate_reasons}`",
        f"- paper_run_resume_allowed: `{str(report['paper_run_resume_allowed']).lower()}`",
        "",
        "| gate | passed |",
        "|---|---|",
    ]
    for gate, passed in report["hard_gates"].items():
        lines.append(f"| {gate} | {str(passed).lower()} |")
    lines.extend(
        [
            "",
            "## Candidate-Scoped Eligibility",
            "",
            "| candidate | eligible | enabled_families | disabled_families | scoped_p_up_disagreement | support | mean | sum | reasons |",
            "|---|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for candidate in report["candidate_scoped_eligibility_summary"]:
        lines.append(
            "| {name} | {eligible} | {enabled} | {disabled} | {p_up:.6f} | "
            "{support} | {mean:.6f} | {total:.6f} | {reasons} |".format(
                name=candidate["candidate_name"],
                eligible=str(candidate["source_model_candidate_eligible"]).lower(),
                enabled=", ".join(candidate["enabled_action_families"]) or "none",
                disabled=", ".join(candidate["disabled_action_families"]) or "none",
                p_up=candidate["candidate_scoped_p_up_action_disagreement_rate"],
                support=candidate["candidate_scoped_high_score_support_count"],
                mean=candidate[
                    "candidate_scoped_high_score_realized_return_mean"
                ],
                total=candidate[
                    "candidate_scoped_high_score_realized_return_sum"
                ],
                reasons=", ".join(candidate["ineligible_reason_codes"]) or "none",
            )
        )
    return "\n".join(lines) + "\n"


def _ranking_split_report(
    *,
    split_name: str,
    examples: tuple[PolymarketPolicyExample, ...],
    predictions: tuple[PolymarketPolicyPrediction, ...],
) -> dict[str, Any]:
    _validate_aligned(examples, predictions)
    rows = [
        _ranking_row(example=example, prediction=prediction)
        for example, prediction in zip(examples, predictions, strict=True)
    ]
    metrics = _ranking_metrics(rows)
    return {
        "split_name": split_name,
        **metrics,
        "selected_action_distribution": dict(
            sorted(Counter(row["calibrated_best_policy_action"] for row in rows).items())
        ),
        "realized_best_action_distribution": dict(
            sorted(Counter(row["realized_best_action"] for row in rows).items())
        ),
        "realized_best_rank_distribution": dict(
            sorted(
                Counter(
                    str(row["rank_of_realized_best_action_under_calibrated_scores"])
                    for row in rows
                ).items()
            )
        ),
        "breakdowns": _ranking_breakdowns(rows),
        "rows": rows,
        "top_regret_examples": sorted(
            rows,
            key=lambda row: (-float(row["regret"]), row["decision_ts"], row["market_id"]),
        )[:20],
    }


def _ranking_row(
    *,
    example: PolymarketPolicyExample,
    prediction: PolymarketPolicyPrediction,
) -> dict[str, Any]:
    scores = _score_map(prediction)
    selected_action = _selected_action(prediction)
    ranked_actions = _ranked_actions(scores)
    realized_best_action, oracle_return = _realized_best_action(example)
    realized_rank = ranked_actions.index(realized_best_action) + 1
    selected_return = float(example.action_return_targets[selected_action])
    score_spread = float(scores[selected_action]) - float(scores[realized_best_action])
    raw_score = float(prediction.expected_return_by_action[selected_action])
    bucket = action_value_bucket_payload(
        action=selected_action,
        features=prediction.features,
        raw_score=raw_score,
    )
    return {
        "market_id": example.market_id,
        "condition_id": example.condition_id,
        "slug": example.slug,
        "market_family": example.market_family,
        "decision_ts": int(example.decision_ts),
        "calibrated_best_policy_action": selected_action,
        "realized_best_action": realized_best_action,
        "rank_of_realized_best_action_under_calibrated_scores": realized_rank,
        "top_1_action_hit": realized_rank <= 1,
        "top_2_action_hit": realized_rank <= 2,
        "top_3_action_hit": realized_rank <= 3,
        "score_spread_selected_minus_realized_best": score_spread,
        "selected_action_realized_return": selected_return,
        "oracle_best_action_realized_return": oracle_return,
        "regret": oracle_return - selected_return,
        "selected_action_calibrated_score": float(scores[selected_action]),
        "realized_best_action_calibrated_score": float(scores[realized_best_action]),
        **bucket,
    }


def _candidate_specs(
    *,
    validation_examples: tuple[PolymarketPolicyExample, ...],
    raw_validation_predictions: tuple[PolymarketPolicyPrediction, ...],
    calibrated_validation_predictions: tuple[PolymarketPolicyPrediction, ...],
    execution_buffer: float,
) -> tuple[dict[str, Any], ...]:
    family_corrections = _fit_group_corrections(
        examples=validation_examples,
        predictions=raw_validation_predictions,
        group_fn=action_value_action_family,
    )
    action_corrections = _fit_group_corrections(
        examples=validation_examples,
        predictions=raw_validation_predictions,
        group_fn=lambda action: action,
    )
    pairwise_corrections = _fit_pairwise_rank_corrections(validation_examples)
    validation_action_corrected = _apply_candidate_spec(
        predictions=raw_validation_predictions,
        fallback_predictions=calibrated_validation_predictions,
        spec={
            "candidate_name": "C_action_specific_calibration_with_family_gates",
            "score_source": "raw",
            "corrections": action_corrections,
            "correction_group": "action",
            "eligible_families": None,
            "notes": [],
        },
    )
    validation_family_report = build_action_family_eligibility_report(
        examples=validation_examples,
        predictions=validation_action_corrected,
        execution_buffer=execution_buffer,
    )
    eligible_families = tuple(validation_family_report["eligible_action_families"])
    prior_penalties = _fit_family_prior_penalties(
        examples=validation_examples,
        predictions=calibrated_validation_predictions,
        execution_buffer=execution_buffer,
    )
    lcb_overlay = _fit_bucket_overlay(
        examples=validation_examples,
        predictions=calibrated_validation_predictions,
        execution_buffer=execution_buffer,
        method="bucketed_lcb_rank_selector",
        min_bucket_support=RANKING_OVERLAY_G_MIN_BUCKET_SUPPORT,
        min_family_support=RANKING_OVERLAY_MIN_FAMILY_SUPPORT,
        require_lcb_over_buffer=True,
        require_positive_only=False,
    )
    positive_overlay = _fit_bucket_overlay(
        examples=validation_examples,
        predictions=calibrated_validation_predictions,
        execution_buffer=execution_buffer,
        method="positive_bucket_rank_selector",
        min_bucket_support=RANKING_OVERLAY_H_MIN_BUCKET_SUPPORT,
        min_family_support=RANKING_OVERLAY_MIN_FAMILY_SUPPORT,
        require_lcb_over_buffer=False,
        require_positive_only=True,
    )
    return (
        {
            "candidate_name": "A_current_model_baseline",
            "candidate_type": "current_model_baseline",
            "score_source": "fallback",
            "corrections": {},
            "correction_group": "none",
            "eligible_families": None,
            "notes": ["current calibrated model scores"],
        },
        {
            "candidate_name": "B_family_specific_calibration_only",
            "candidate_type": "family_specific_calibration",
            "score_source": "raw",
            "corrections": family_corrections,
            "correction_group": "action_family",
            "eligible_families": None,
            "notes": ["validation residual correction grouped by action family"],
        },
        {
            "candidate_name": "C_action_specific_calibration_with_family_gates",
            "candidate_type": "action_specific_calibration_with_family_gates",
            "score_source": "raw",
            "corrections": action_corrections,
            "correction_group": "action",
            "eligible_families": eligible_families,
            "notes": [
                "validation residual correction grouped by action",
                "families failing validation high-score gates are blocked to NO_TRADE",
            ],
        },
        {
            "candidate_name": "D_pairwise_rank_correction",
            "candidate_type": "pairwise_rank_correction",
            "score_source": "raw",
            "corrections": pairwise_corrections,
            "correction_group": "action",
            "eligible_families": None,
            "notes": ["deterministic pairwise oracle-frequency ranking correction"],
        },
        {
            "candidate_name": "E_action_family_prior_penalty",
            "candidate_type": "action_family_prior_penalty",
            "score_source": "fallback",
            "corrections": prior_penalties,
            "correction_group": "action_family",
            "eligible_families": None,
            "notes": [
                "penalize validation high-score families below execution buffer"
            ],
        },
        {
            "candidate_name": "F_live_eligible_feature_subset_retrain_proxy",
            "candidate_type": "live_eligible_feature_subset_retrain_proxy",
            "score_source": "fallback",
            "corrections": {},
            "correction_group": "none",
            "eligible_families": None,
            "notes": [
                "diagnostic proxy only; no model-family retrain is promoted by #145"
            ],
        },
        {
            "candidate_name": "G_bucketed_lcb_rank_selector",
            "candidate_type": "bucketed_lcb_rank_selector",
            "score_source": "fallback",
            "corrections": {},
            "correction_group": "none",
            "eligible_families": None,
            "ranking_overlay": lcb_overlay,
            "notes": [
                "validation-fitted shrunk bucket/family lower-confidence-bound selector",
                "shadow labels are not used for fit",
            ],
        },
        {
            "candidate_name": "H_positive_bucket_rank_selector",
            "candidate_type": "positive_bucket_rank_selector",
            "score_source": "fallback",
            "corrections": {},
            "correction_group": "none",
            "eligible_families": None,
            "ranking_overlay": positive_overlay,
            "notes": [
                "validation-fitted buffer-positive bucket/family selector",
                "shadow labels are not used for fit",
            ],
        },
        {
            "candidate_name": SELL_BEFORE_CLOSE_ONLY_SOURCE_CANDIDATE_NAME,
            "candidate_type": "sell_before_close_only_source_candidate",
            "score_source": "fallback",
            "corrections": {},
            "correction_group": "none",
            "eligible_families": ("SELL_BEFORE_CLOSE",),
            "enabled_actions": SELL_BEFORE_CLOSE_ONLY_SOURCE_CANDIDATE_ACTIONS,
            "disabled_actions": SELL_BEFORE_CLOSE_DISABLED_SOURCE_CANDIDATE_ACTIONS,
            "notes": [
                "source eligibility candidate scoped to SELL_BEFORE_CLOSE actions",
                "HOLD_TO_SETTLEMENT is disabled and remains diagnostic-only",
            ],
        },
        {
            "candidate_name": SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_CANDIDATE_NAME,
            "candidate_type": "sell_before_close_exit_reliability_guard_candidate",
            "score_source": "fallback",
            "corrections": {},
            "correction_group": "none",
            "eligible_families": ("SELL_BEFORE_CLOSE",),
            "enabled_actions": SELL_BEFORE_CLOSE_ONLY_SOURCE_CANDIDATE_ACTIONS,
            "disabled_actions": SELL_BEFORE_CLOSE_DISABLED_SOURCE_CANDIDATE_ACTIONS,
            "exit_reliability_guard_enabled": True,
            "exit_policy": SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_EXIT_POLICY,
            "entry_filter_thresholds": dict(
                SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_THRESHOLDS
            ),
            "notes": [
                "source eligibility candidate scoped to SELL_BEFORE_CLOSE actions",
                "entry requires causal exit-reliability guard evidence",
                "replay enforces deterministic pre-settlement exit attempts",
                "HOLD_TO_SETTLEMENT is disabled and remains diagnostic-only",
            ],
        },
        {
            "candidate_name": SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_CANDIDATE_NAME,
            "candidate_type": "sell_before_close_exit_reliability_p_up_aligned_candidate",
            "score_source": "fallback",
            "corrections": {},
            "correction_group": "none",
            "eligible_families": ("SELL_BEFORE_CLOSE",),
            "enabled_actions": SELL_BEFORE_CLOSE_ONLY_SOURCE_CANDIDATE_ACTIONS,
            "disabled_actions": SELL_BEFORE_CLOSE_DISABLED_SOURCE_CANDIDATE_ACTIONS,
            "exit_reliability_guard_enabled": True,
            "p_up_side_alignment_filter_enabled": True,
            "exit_policy": SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_EXIT_POLICY,
            "entry_filter_thresholds": dict(
                SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_THRESHOLDS
            ),
            "notes": [
                "source eligibility candidate scoped to SELL_BEFORE_CLOSE actions",
                "inherits the stateful exit-reliability guard",
                "requires causal p_up/action side alignment before entry",
                "prevents same-market churn with re-entry controls",
                "HOLD_TO_SETTLEMENT is disabled and remains diagnostic-only",
            ],
        },
        {
            "candidate_name": (
                SELL_BEFORE_CLOSE_SUPPORT_AWARE_P_UP_ALIGNED_CANDIDATE_NAME
            ),
            "candidate_type": (
                "sell_before_close_support_aware_p_up_aligned_candidate"
            ),
            "score_source": "fallback",
            "corrections": {},
            "correction_group": "none",
            "eligible_families": ("SELL_BEFORE_CLOSE",),
            "enabled_actions": SELL_BEFORE_CLOSE_ONLY_SOURCE_CANDIDATE_ACTIONS,
            "disabled_actions": SELL_BEFORE_CLOSE_DISABLED_SOURCE_CANDIDATE_ACTIONS,
            "exit_reliability_guard_enabled": True,
            "p_up_side_alignment_filter_enabled": True,
            "exit_policy": SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_EXIT_POLICY,
            "entry_filter_thresholds": {},
            "threshold_selection_method": (
                "validation_fitted_support_aware_thresholds"
            ),
            "threshold_selection_fit_split": "validation",
            "threshold_selection_evaluation_split": "shadow",
            "uses_shadow_for_fit": False,
            "notes": [
                "source eligibility candidate scoped to SELL_BEFORE_CLOSE actions",
                "inherits the stateful exit-reliability guard",
                "uses validation-fitted support-aware entry thresholds",
                "requires causal p_up/action side alignment before entry",
                "HOLD_TO_SETTLEMENT is disabled and remains diagnostic-only",
            ],
        },
    )


def _candidate_report(
    *,
    spec: dict[str, Any],
    validation_examples: tuple[PolymarketPolicyExample, ...],
    validation_predictions: tuple[PolymarketPolicyPrediction, ...],
    shadow_examples: tuple[PolymarketPolicyExample, ...],
    raw_shadow_predictions: tuple[PolymarketPolicyPrediction, ...],
    shadow_predictions: tuple[PolymarketPolicyPrediction, ...],
    execution_buffer: float,
) -> dict[str, Any]:
    enabled_actions = _candidate_enabled_actions(spec)
    disabled_actions = _candidate_disabled_actions(spec, enabled_actions=enabled_actions)
    enabled_action_families = _families_for_actions(enabled_actions)
    disabled_action_families = _families_for_actions(disabled_actions)
    ranking = _ranking_split_report(
        split_name="shadow",
        examples=shadow_examples,
        predictions=shadow_predictions,
    )
    raw_mae = _mae(
        examples=shadow_examples,
        predictions=raw_shadow_predictions,
        score_getter=lambda prediction: prediction.expected_return_by_action,
        actions=enabled_actions,
    )
    calibrated_mae = _mae(
        examples=shadow_examples,
        predictions=shadow_predictions,
        score_getter=_score_map,
        actions=enabled_actions,
    )
    family_report = build_action_family_eligibility_report(
        examples=shadow_examples,
        predictions=shadow_predictions,
        execution_buffer=execution_buffer,
    )
    candidate_scoped_family_report = _candidate_scoped_family_report(
        family_report=family_report,
        enabled_action_families=enabled_action_families,
        disabled_action_families=disabled_action_families,
    )
    calibration_support_passed = len(validation_examples) >= 3
    high_score_support_passed = (
        int(family_report["high_score_support_count"])
        >= ACTION_VALUE_HIGH_SCORE_MIN_SUPPORT
    )
    high_score_mean_passed = (
        high_score_support_passed
        and float(family_report["high_score_realized_return_mean"]) > execution_buffer
    )
    calibration_quality_passed = (
        calibrated_mae <= raw_mae + ACTION_VALUE_QUALITY_MAE_TOLERANCE
        and high_score_support_passed
        and high_score_mean_passed
    )
    action_counts = Counter(
        _selected_action(prediction)
        for prediction in (*validation_predictions, *shadow_predictions)
    )
    sample_count = sum(action_counts.values())
    max_action, max_action_count = (
        ("", 0) if sample_count == 0 else action_counts.most_common(1)[0]
    )
    max_action_ratio = 0.0 if sample_count == 0 else max_action_count / sample_count
    concentration_passed = max_action_ratio <= BEST_ACTION_CONCENTRATION_FAIL_THRESHOLD
    candidate_scoped_disagreement_rows = [
        prediction
        for prediction in shadow_predictions
        if _selected_action(prediction) in set(enabled_actions)
        and _selected_action(prediction) != "NO_TRADE"
    ]
    entry_decision_count_before_guard = sum(
        1
        for prediction in shadow_predictions
        if _selected_action(prediction)
        in {"BUY_UP_SELL_BEFORE_CLOSE", "BUY_DOWN_SELL_BEFORE_CLOSE"}
    )
    disagreement_count = sum(
        _p_up_action_disagrees(prediction)
        for prediction in candidate_scoped_disagreement_rows
    )
    disagreement_rate = (
        0.0
        if not candidate_scoped_disagreement_rows
        else disagreement_count / len(candidate_scoped_disagreement_rows)
    )
    disagreement_passed = disagreement_rate <= P_UP_ACTION_DISAGREEMENT_FAIL_THRESHOLD
    ineligible_reasons = set()
    if not calibration_support_passed:
        ineligible_reasons.add("action_value_calibration_support_insufficient")
    if not calibration_quality_passed:
        ineligible_reasons.add("action_value_calibration_quality_failed")
    if not concentration_passed:
        ineligible_reasons.add("action_value_policy_collapse")
    if not disagreement_passed:
        ineligible_reasons.add("p_up_action_disagreement_excessive")
    if not family_report["action_family_paper_decision_eligible"]:
        ineligible_reasons.update(
            family_report["action_family_paper_decision_ineligible_reasons"]
        )
    high_score_sum_positive = (
        float(family_report["high_score_realized_return_sum"]) > 0.0
    )
    if not high_score_sum_positive:
        ineligible_reasons.add("action_value_high_score_return_sum_not_positive")
    source_model_eligible = (
        calibration_support_passed
        and calibration_quality_passed
        and bool(family_report["action_family_paper_decision_eligible"])
        and concentration_passed
        and disagreement_passed
        and high_score_sum_positive
    )
    overlay = spec.get("ranking_overlay")
    ranking_overlay_used = overlay is not None
    return {
        "candidate_name": spec["candidate_name"],
        "candidate_type": spec["candidate_type"],
        "score_source": spec["score_source"],
        "correction_group": spec["correction_group"],
        "notes": list(spec["notes"]),
        "exit_reliability_guard_enabled": bool(
            spec.get("exit_reliability_guard_enabled", False)
        ),
        "p_up_side_alignment_filter_enabled": bool(
            spec.get("p_up_side_alignment_filter_enabled", False)
        ),
        "exit_policy": spec.get("exit_policy"),
        "entry_filter_thresholds": dict(spec.get("entry_filter_thresholds", {})),
        "threshold_selection_method": spec.get("threshold_selection_method"),
        "threshold_selection_fit_split": spec.get("threshold_selection_fit_split"),
        "threshold_selection_evaluation_split": spec.get(
            "threshold_selection_evaluation_split"
        ),
        "uses_shadow_for_fit": spec.get("uses_shadow_for_fit"),
        "shadow_sweep_not_used_for_threshold_fit": spec.get(
            "shadow_sweep_not_used_for_threshold_fit",
            True if spec.get("threshold_selection_method") else None,
        ),
        "support_aware_threshold_selection_failed": False,
        "support_aware_threshold_selection_reason_codes": [],
        "entry_decision_count_before_guard": entry_decision_count_before_guard,
        "entry_decision_count_after_exit_guard": entry_decision_count_before_guard,
        "entry_decision_count_after_p_up_alignment": entry_decision_count_before_guard,
        "entry_decision_count_after_guard": entry_decision_count_before_guard,
        "entry_filter_blocked_count": 0,
        "entry_filter_blocked_by_p_up_alignment_count": 0,
        "entry_filter_blocked_by_quality_count": 0,
        "reentry_cooldown_seconds": None,
        "reentry_blocked_count": 0,
        "entries_per_market_distribution": {},
        "positions_opened_count": None,
        "positions_closed_before_settlement_count": None,
        "positions_opened_but_not_closed_before_settlement": None,
        "replay_realized_trade_pnl": None,
        "replay_settlement_pnl": None,
        "replay_total_polymarket_pnl": None,
        "replay_residual_settlement_drag": None,
        "replay_total_pnl_improved_vs_i_candidate": None,
        "promotion_support_eligible": False,
        "promotion_support_gate_passed": False,
        "promotion_support_reason_codes": [],
        "promotion_support_thresholds": {},
        "promotion_replay_entry_decision_count": None,
        "promotion_replay_sell_decision_count": None,
        "promotion_replay_unique_market_count": None,
        "promotion_replay_side_count": None,
        "promotion_replay_side_distribution": {},
        "promotion_replay_mean_pnl_per_entry": None,
        "ranking_overlay_used": ranking_overlay_used,
        "ranking_overlay_method": None if overlay is None else overlay["method"],
        "ranking_overlay_fit_split": None if overlay is None else overlay["fit_split"],
        "ranking_overlay_evaluation_split": None
        if overlay is None
        else overlay["evaluation_split"],
        "ranking_overlay_uses_shadow_split": False
        if overlay is not None
        else None,
        "ranking_overlay_min_bucket_support": None
        if overlay is None
        else overlay["min_bucket_support"],
        "ranking_overlay_min_family_support": None
        if overlay is None
        else overlay["min_family_support"],
        "ranking_overlay_shrinkage_prior_support": None
        if overlay is None
        else overlay["shrinkage_prior_support"],
        "ranking_overlay_shrinkage_prior_mean": None
        if overlay is None
        else overlay["shrinkage_prior_mean"],
        "ranking_overlay_score_combination": None
        if overlay is None
        else overlay["score_combination"],
        "ranking_overlay_bucket_evidence_weight": None
        if overlay is None
        else overlay["bucket_evidence_weight"],
        "ranking_overlay_model_score_weight": None
        if overlay is None
        else overlay["model_score_weight"],
        "shadow_raw_mae": raw_mae,
        "shadow_calibrated_mae": calibrated_mae,
        "shadow_top_1_action_hit_rate": ranking["top_1_action_hit_rate"],
        "shadow_top_2_action_hit_rate": ranking["top_2_action_hit_rate"],
        "shadow_top_3_action_hit_rate": ranking["top_3_action_hit_rate"],
        "shadow_mean_regret": ranking["mean_regret"],
        "high_score_support_count": family_report["high_score_support_count"],
        "high_score_realized_return_mean": family_report[
            "high_score_realized_return_mean"
        ],
        "high_score_realized_return_sum": family_report[
            "high_score_realized_return_sum"
        ],
        "high_score_realized_return_sum_positive": high_score_sum_positive,
        "enabled_action_families": enabled_action_families,
        "disabled_action_families": disabled_action_families,
        "enabled_actions": enabled_actions,
        "disabled_actions": disabled_actions,
        "candidate_scoped_p_up_action_disagreement_count": disagreement_count,
        "candidate_scoped_p_up_action_disagreement_denominator": (
            len(candidate_scoped_disagreement_rows)
        ),
        "candidate_scoped_p_up_action_disagreement_rate": disagreement_rate,
        "candidate_scoped_p_up_action_disagreement_within_limit": (
            disagreement_passed
        ),
        "candidate_scoped_action_family_gate_results": (
            candidate_scoped_family_report["enabled_action_family_gate_results"]
        ),
        "candidate_scoped_disabled_action_family_gate_results": (
            candidate_scoped_family_report["disabled_action_family_gate_results"]
        ),
        "candidate_scoped_action_gate_results": (
            candidate_scoped_family_report["enabled_action_gate_results"]
        ),
        "candidate_scoped_disabled_action_gate_results": (
            candidate_scoped_family_report["disabled_action_gate_results"]
        ),
        "candidate_scoped_high_score_support_count": family_report[
            "high_score_support_count"
        ],
        "candidate_scoped_high_score_realized_return_mean": family_report[
            "high_score_realized_return_mean"
        ],
        "candidate_scoped_high_score_realized_return_sum": family_report[
            "high_score_realized_return_sum"
        ],
        "action_family_gates": family_report["action_family_gate_results"],
        "action_family_paper_decision_eligible": family_report[
            "action_family_paper_decision_eligible"
        ],
        "action_family_paper_decision_ineligible_reasons": family_report[
            "action_family_paper_decision_ineligible_reasons"
        ],
        "calibration_support_passed": calibration_support_passed,
        "calibration_quality_passed": calibration_quality_passed,
        "best_action_concentration_passed": concentration_passed,
        "best_action_max_action": max_action or None,
        "best_action_max_ratio": max_action_ratio,
        "p_up_action_disagreement_within_limit": disagreement_passed,
        "p_up_action_disagreement_rate": disagreement_rate,
        "source_model_eligible": source_model_eligible,
        "source_model_candidate_eligible": source_model_eligible,
        "requires_promotion_replay_gate": True,
        "paper_run_resume_allowed": False,
        "ineligible_reason_codes": sorted(ineligible_reasons),
        "selected_action_distribution": dict(sorted(action_counts.items())),
        "ranking_summary": {
            key: ranking[key]
            for key in (
                "sample_count",
                "top_1_action_hit_rate",
                "top_2_action_hit_rate",
                "top_3_action_hit_rate",
                "mean_regret",
                "mean_selected_action_realized_return",
                "mean_oracle_best_action_realized_return",
            )
        },
        "candidate_artifact_required": (
            source_model_eligible
            or ranking_overlay_used
            or spec["candidate_name"].startswith("D_")
        ),
        "candidate_artifact_reason": (
            "source_model_candidate_eligible"
            if source_model_eligible
            else "ranking_overlay_candidate"
            if ranking_overlay_used
            else "best_near_eligible_diagnostic"
            if spec["candidate_name"].startswith("D_")
            else "not_exported"
        ),
        "candidate_predictions": [
            prediction.to_dict() for prediction in shadow_predictions
        ],
        "candidate_manifest": _candidate_manifest(
            candidate_name=spec["candidate_name"],
            candidate_type=spec["candidate_type"],
            ranking_overlay=overlay,
            source_model_eligible=source_model_eligible,
            action_family_paper_decision_eligible=family_report[
                "action_family_paper_decision_eligible"
            ],
            calibration_quality_passed=calibration_quality_passed,
            best_action_concentration_passed=concentration_passed,
            p_up_action_disagreement_within_limit=disagreement_passed,
            high_score_support_count=family_report["high_score_support_count"],
            high_score_realized_return_mean=family_report[
                "high_score_realized_return_mean"
            ],
            high_score_realized_return_sum=family_report[
                "high_score_realized_return_sum"
            ],
            ineligible_reason_codes=sorted(ineligible_reasons),
            enabled_action_families=enabled_action_families,
            disabled_action_families=disabled_action_families,
            enabled_actions=enabled_actions,
            disabled_actions=disabled_actions,
            candidate_scoped_p_up_action_disagreement_rate=disagreement_rate,
            candidate_scoped_action_family_gate_results=(
                candidate_scoped_family_report["enabled_action_family_gate_results"]
            ),
        ),
        "ranking_overlay": _candidate_overlay_payload(spec),
        **compact_safety_fields(),
    }


def _action_representation_split_report(
    *,
    split_name: str,
    examples: tuple[PolymarketPolicyExample, ...],
    predictions: tuple[PolymarketPolicyPrediction, ...],
    execution_buffer: float,
) -> dict[str, Any]:
    rows = [
        _action_representation_row(
            split_name=split_name,
            example=example,
            prediction=prediction,
            action=action,
        )
        for example, prediction in zip(examples, predictions, strict=True)
        for action in ACTION_VALUE_LABEL_ACTIONS
        if action != "NO_TRADE"
    ]
    sell_rows = [
        row
        for row in rows
        if row["intended_exit_policy"] == "sell_before_close"
    ]
    sell_groups = _action_representation_group_summaries(
        rows=sell_rows,
        group_fields=(
            "fine_action_family",
            "side",
            "intended_exit_policy",
            "price_bucket",
            "time_to_close_bucket",
        ),
        execution_buffer=execution_buffer,
    )
    positive_sell_groups = [
        row
        for row in sell_groups
        if row["support_count"] >= ACTION_REPRESENTATION_MIN_BUCKET_SUPPORT
        and row["realized_return_mean"] > execution_buffer
        and row["realized_return_sum"] > 0.0
    ]
    return {
        "split": split_name,
        "example_count": len(examples),
        "action_count": len(rows),
        "sell_before_close_summary": _action_representation_metrics(
            sell_rows,
            execution_buffer=execution_buffer,
        ),
        "action_family_summary": _action_representation_group_summaries(
            rows=rows,
            group_fields=("action_family",),
            execution_buffer=execution_buffer,
        ),
        "fine_action_family_summary": _action_representation_group_summaries(
            rows=rows,
            group_fields=("fine_action_family",),
            execution_buffer=execution_buffer,
        ),
        "side_exit_policy_price_time_summary": _action_representation_group_summaries(
            rows=rows,
            group_fields=(
                "side",
                "intended_exit_policy",
                "price_bucket",
                "time_to_close_bucket",
            ),
            execution_buffer=execution_buffer,
        ),
        "sell_before_close_negative_contributors": [
            row for row in sell_groups if row["realized_return_sum"] < 0.0
        ][:10],
        "sell_before_close_positive_supported_buckets": positive_sell_groups[:10],
        "top_negative_high_score_sell_before_close_examples": (
            _action_representation_top_negative_examples(sell_rows)
        ),
        "supported_positive_sell_before_close_bucket_count": len(
            positive_sell_groups
        ),
        "sell_before_close_unique_market_count": len(
            {row["market_id"] for row in sell_rows}
        ),
        "min_positive_bucket_support": ACTION_REPRESENTATION_MIN_BUCKET_SUPPORT,
        "execution_buffer": execution_buffer,
    }


def _sell_before_close_label_exit_path_assessment(
    *,
    examples: tuple[PolymarketPolicyExample, ...],
) -> dict[str, Any]:
    sell_actions = (
        "BUY_UP_SELL_BEFORE_CLOSE",
        "BUY_DOWN_SELL_BEFORE_CLOSE",
    )
    class_counts: Counter[str] = Counter()
    executable_path_count = 0
    missing_metadata_count = 0
    theoretical_positive_non_executable_count = 0
    for example in examples:
        for action in sell_actions:
            execution_class = example.sell_before_close_execution_class_targets.get(
                action,
            )
            if not execution_class:
                missing_metadata_count += 1
                continue
            class_counts[str(execution_class)] += 1
            if example.sell_before_close_label_uses_executable_exit_path_targets.get(
                action,
                False,
            ):
                executable_path_count += 1
            theoretical_return = float(
                example.sell_before_close_theoretical_return_targets.get(action, 0.0)
            )
            if theoretical_return > 0.0 and execution_class != (
                "realizable_sell_before_close"
            ):
                theoretical_positive_non_executable_count += 1
    redesigned_metadata_present = missing_metadata_count == 0 and bool(class_counts)
    if not redesigned_metadata_present:
        return {
            "sell_before_close_entry_price": "entry ask at decision_ts",
            "sell_before_close_exit_price": "last available bid at market_end_ts - 1",
            "sell_before_close_exit_path_is_fixed_terminal_bid": True,
            "sell_before_close_exit_path_coarse": True,
            "uses_intraround_exit_opportunity_model": False,
            "uses_queue_fill_probability_model": False,
            "compares_theoretical_vs_executable_exit_return": False,
            "missing_executable_exit_metadata_count": missing_metadata_count,
            "coarse_exit_path_risk_codes": [
                "single_terminal_exit_bid_path",
                "no_intraround_exit_optimization",
                "no_queue_fill_probability_model",
            ],
        }
    reason_codes = []
    if theoretical_positive_non_executable_count:
        reason_codes.append("positive_theoretical_return_without_executable_exit")
    return {
        "sell_before_close_entry_price": "entry ask at decision_ts",
        "sell_before_close_exit_price": "best executable intraround bid path",
        "sell_before_close_theoretical_reference": "terminal bid is diagnostic only",
        "sell_before_close_exit_path_is_fixed_terminal_bid": False,
        "sell_before_close_exit_path_coarse": False,
        "uses_intraround_exit_opportunity_model": True,
        "uses_queue_fill_probability_model": True,
        "compares_theoretical_vs_executable_exit_return": True,
        "sell_before_close_execution_class_distribution": dict(
            sorted(class_counts.items())
        ),
        "label_uses_executable_exit_path_count": executable_path_count,
        "theoretical_positive_non_executable_count": (
            theoretical_positive_non_executable_count
        ),
        "coarse_exit_path_risk_codes": reason_codes,
    }


def _action_representation_row(
    *,
    split_name: str,
    example: PolymarketPolicyExample,
    prediction: PolymarketPolicyPrediction,
    action: str,
) -> dict[str, Any]:
    raw_score = float(prediction.expected_return_by_action[action])
    scores = _score_map(prediction)
    bucket = action_value_bucket_payload(
        action=action,
        features=prediction.features,
        raw_score=raw_score,
    )
    execution_class = example.sell_before_close_execution_class_targets.get(
        action,
        "not_applicable",
    )
    return {
        "split": split_name,
        "market_id": example.market_id,
        "condition_id": example.condition_id,
        "slug": example.slug,
        "market_family": example.market_family,
        "decision_ts": int(example.decision_ts),
        "action": action,
        "action_family": bucket["action_family"],
        "fine_action_family": bucket["fine_action_family"],
        "side": bucket["side"],
        "intended_exit_policy": bucket["intended_exit_policy"],
        "price_bucket": bucket["price_bucket"],
        "time_to_close_bucket": bucket["time_to_close_bucket"],
        "raw_score_bucket": bucket["raw_score_bucket"],
        "raw_score": raw_score,
        "calibrated_score": float(scores[action]),
        "realized_return": float(example.action_return_targets[action]),
        "realized_trade_return": float(
            example.realized_trade_return_targets.get(action, 0.0)
        ),
        "settlement_return": float(
            example.settlement_return_targets.get(action, 0.0)
        ),
        "is_positive": bool(example.action_is_positive_targets.get(action, False)),
        "sell_before_close_execution_class": execution_class,
        "label_uses_executable_exit_path": bool(
            example.sell_before_close_label_uses_executable_exit_path_targets.get(
                action,
                False,
            )
        ),
        "theoretical_terminal_bid_return": float(
            example.sell_before_close_theoretical_return_targets.get(action, 0.0)
        ),
        "realized_executable_sell_before_close_return": float(
            example.sell_before_close_executable_return_targets.get(action, 0.0)
        ),
        "execution_gap_return": float(
            example.sell_before_close_execution_gap_targets.get(action, 0.0)
        ),
        "queue_fill_probability_estimate": float(
            example.sell_before_close_queue_fill_probability_targets.get(action, 0.0)
        ),
        "entry_up_ask": prediction.features.get("up_ask"),
        "entry_down_ask": prediction.features.get("down_ask"),
        "entry_up_bid": prediction.features.get("up_bid"),
        "entry_down_bid": prediction.features.get("down_bid"),
        "time_to_close_seconds": prediction.features.get("time_to_close_seconds"),
        "market_age_seconds": prediction.features.get("market_age_seconds"),
    }


def _action_representation_top_negative_examples(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    negative_high_score_rows = [
        row
        for row in rows
        if row["realized_return"] < 0.0 and row["calibrated_score"] > 0.0
    ]
    ranked = sorted(
        negative_high_score_rows,
        key=lambda row: (
            -float(row["calibrated_score"]),
            float(row["realized_return"]),
            str(row["market_id"]),
            int(row["decision_ts"]),
            str(row["action"]),
        ),
    )
    fields = (
        "market_id",
        "condition_id",
        "slug",
        "decision_ts",
        "action",
        "fine_action_family",
        "side",
        "price_bucket",
        "time_to_close_bucket",
        "raw_score",
        "calibrated_score",
        "realized_return",
        "realized_trade_return",
        "settlement_return",
        "entry_up_ask",
        "entry_down_ask",
        "entry_up_bid",
        "entry_down_bid",
        "time_to_close_seconds",
        "market_age_seconds",
    )
    return [{field: row[field] for field in fields} for row in ranked[:10]]


def _action_representation_group_summaries(
    *,
    rows: list[dict[str, Any]],
    group_fields: tuple[str, ...],
    execution_buffer: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[field]) for field in group_fields)].append(row)
    summaries = []
    for key, group_rows in grouped.items():
        payload = {field: key[index] for index, field in enumerate(group_fields)}
        payload.update(
            _action_representation_metrics(
                group_rows,
                execution_buffer=execution_buffer,
            )
        )
        summaries.append(payload)
    return sorted(
        summaries,
        key=lambda row: (
            float(row["realized_return_sum"]),
            -int(row["support_count"]),
            tuple(str(row[field]) for field in group_fields),
        ),
    )


def _action_representation_metrics(
    rows: list[dict[str, Any]],
    *,
    execution_buffer: float,
) -> dict[str, Any]:
    realized_returns = [float(row["realized_return"]) for row in rows]
    trade_returns = [float(row["realized_trade_return"]) for row in rows]
    settlement_returns = [float(row["settlement_return"]) for row in rows]
    theoretical_returns = [float(row["theoretical_terminal_bid_return"]) for row in rows]
    executable_returns = [
        float(row["realized_executable_sell_before_close_return"]) for row in rows
    ]
    execution_gaps = [float(row["execution_gap_return"]) for row in rows]
    queue_fill_probabilities = [
        float(row["queue_fill_probability_estimate"]) for row in rows
    ]
    calibrated_scores = [float(row["calibrated_score"]) for row in rows]
    support_count = len(rows)
    unique_market_count = len({row["market_id"] for row in rows})
    positive_count = sum(1 for value in realized_returns if value > 0.0)
    realized_sum = sum(realized_returns)
    realized_mean = _mean(realized_returns)
    execution_class_counts = Counter(
        str(row["sell_before_close_execution_class"]) for row in rows
    )
    executable_path_count = sum(
        1 for row in rows if row["label_uses_executable_exit_path"]
    )
    return {
        "support_count": support_count,
        "unique_market_count": unique_market_count,
        "realized_return_mean": realized_mean,
        "realized_return_sum": realized_sum,
        "realized_trade_return_mean": _mean(trade_returns),
        "settlement_return_mean": _mean(settlement_returns),
        "theoretical_terminal_bid_return_mean": _mean(theoretical_returns),
        "realized_executable_sell_before_close_return_mean": _mean(
            executable_returns
        ),
        "execution_gap_return_mean": _mean(execution_gaps),
        "queue_fill_probability_mean": _mean(queue_fill_probabilities),
        "sell_before_close_execution_class_distribution": dict(
            sorted(execution_class_counts.items())
        ),
        "label_uses_executable_exit_path_count": executable_path_count,
        "calibrated_score_mean": _mean(calibrated_scores),
        "positive_count": positive_count,
        "positive_rate": 0.0 if support_count == 0 else positive_count / support_count,
        "mean_exceeds_execution_buffer": realized_mean > execution_buffer,
        "sum_positive": realized_sum > 0.0,
        "supported_positive_bucket": (
            support_count >= ACTION_REPRESENTATION_MIN_BUCKET_SUPPORT
            and realized_mean > execution_buffer
            and realized_sum > 0.0
        ),
    }


def _zero_entry_candidate_report(
    *,
    spec: dict[str, Any],
    raw_shadow_predictions: tuple[PolymarketPolicyPrediction, ...],
    calibrated_shadow_predictions: tuple[PolymarketPolicyPrediction, ...],
) -> dict[str, Any]:
    overlay = spec["ranking_overlay"]
    overlay_predictions = _apply_candidate_spec(
        predictions=raw_shadow_predictions,
        fallback_predictions=calibrated_shadow_predictions,
        spec=spec,
    )
    selected_actions = [_selected_action(prediction) for prediction in overlay_predictions]
    rows = [
        _zero_entry_action_row(
            action=action,
            prediction=prediction,
            selected_action=selected_action,
            overlay=overlay,
        )
        for prediction, selected_action in zip(
            calibrated_shadow_predictions,
            selected_actions,
            strict=True,
        )
        for action in ACTION_VALUE_LABEL_ACTIONS
        if action != "NO_TRADE"
    ]
    metric_field = _overlay_metric_field(overlay)
    stage_counts = _zero_entry_stage_counts(
        rows=rows,
        selected_actions=selected_actions,
    )
    grouped_summaries = {
        field: _zero_entry_group_summaries(rows=rows, field=field)
        for field in (
            "action",
            "action_family",
            "fine_action_family",
            "intended_exit_policy",
            "side",
            "price_bucket",
            "time_to_close_bucket",
            "raw_score_bucket",
            "market_family",
        )
    }
    return {
        "candidate_name": spec["candidate_name"],
        "ranking_overlay_method": overlay["method"],
        "diagnostic_only": True,
        "fit_split": overlay["fit_split"],
        "evaluation_split": overlay["evaluation_split"],
        "uses_shadow_for_fit": overlay["uses_shadow_for_fit"],
        "min_bucket_support": overlay["min_bucket_support"],
        "min_family_support": overlay["min_family_support"],
        "shrinkage_prior_support": overlay["shrinkage_prior_support"],
        "shrinkage_prior_mean": overlay["shrinkage_prior_mean"],
        "execution_buffer": overlay["execution_buffer"],
        "metric_field": metric_field,
        "blocking_stage_counts_are_non_exclusive": True,
        **stage_counts,
        "grouped_summaries": grouped_summaries,
        "top_near_pass_buckets": _near_pass_evidence(
            evidence=overlay["bucket_evidence"],
            metric_field=metric_field,
            limit=10,
        ),
        "top_near_pass_families": _near_pass_evidence(
            evidence=overlay["family_evidence"],
            metric_field=metric_field,
            limit=10,
        ),
        "source_model_candidate_eligible": False,
        "promotion_eligible": False,
        "paper_run_resume_allowed": False,
        **compact_safety_fields(),
    }


def _zero_entry_action_row(
    *,
    action: str,
    prediction: PolymarketPolicyPrediction,
    selected_action: str,
    overlay: dict[str, Any],
) -> dict[str, Any]:
    family = action_value_action_family(action)
    fine_family = action_value_fine_action_family(
        action=action,
        features=prediction.features,
    )
    bucket = action_value_bucket_payload(
        action=action,
        features=prediction.features,
        raw_score=float(prediction.expected_return_by_action[action]),
    )
    bucket_key = _overlay_bucket_key(action=action, prediction=prediction)
    bucket_evidence = overlay["bucket_evidence"].get(bucket_key)
    family_evidence = overlay["family_evidence"].get(fine_family)
    bucket_missing = bucket_evidence is None
    family_missing = family_evidence is None
    bucket_support_failed = bool(
        bucket_evidence is not None and not bucket_evidence["support_passed"]
    )
    family_support_failed = bool(
        family_evidence is not None and not family_evidence["support_passed"]
    )
    bucket_metric_failed = bool(
        bucket_evidence is not None
        and bucket_evidence["support_passed"]
        and not _overlay_metric_passed(bucket_evidence, overlay)
    )
    family_metric_failed = bool(
        family_evidence is not None
        and family_evidence["support_passed"]
        and not _overlay_metric_passed(family_evidence, overlay)
    )
    bucket_sum_failed = bool(
        bucket_evidence is not None
        and bucket_evidence["support_passed"]
        and not bucket_evidence["sum_positive_passed"]
    )
    family_sum_failed = bool(
        family_evidence is not None
        and family_evidence["support_passed"]
        and not family_evidence["sum_positive_passed"]
    )
    passed_bucket = bool(
        bucket_evidence is not None and bucket_evidence["evidence_passed"]
    )
    passed_family = bool(
        family_evidence is not None and family_evidence["evidence_passed"]
    )
    return {
        "market_id": prediction.market_id,
        "decision_ts": prediction.decision_ts,
        "action": action,
        "action_family": family,
        "fine_action_family": fine_family,
        "intended_exit_policy": bucket["intended_exit_policy"],
        "side": bucket["side"],
        "price_bucket": bucket["price_bucket"],
        "time_to_close_bucket": bucket["time_to_close_bucket"],
        "raw_score_bucket": bucket["raw_score_bucket"],
        "market_family": prediction.market_family,
        "selected_action": selected_action,
        "candidate_selected_this_action": selected_action == action,
        "bucket_missing": bucket_missing,
        "family_missing": family_missing,
        "bucket_support_failed": bucket_support_failed,
        "family_support_failed": family_support_failed,
        "bucket_lcb_or_mean_failed": bucket_metric_failed,
        "family_lcb_or_mean_failed": family_metric_failed,
        "bucket_sum_failed": bucket_sum_failed,
        "family_sum_failed": family_sum_failed,
        "passed_bucket_and_family": passed_bucket and passed_family,
        "primary_blocking_stage": _primary_blocking_stage(
            bucket_missing=bucket_missing,
            family_missing=family_missing,
            bucket_support_failed=bucket_support_failed,
            family_support_failed=family_support_failed,
            bucket_metric_failed=bucket_metric_failed,
            family_metric_failed=family_metric_failed,
            bucket_sum_failed=bucket_sum_failed,
            family_sum_failed=family_sum_failed,
            passed_bucket_and_family=passed_bucket and passed_family,
        ),
    }


def _zero_entry_stage_counts(
    *,
    rows: list[dict[str, Any]],
    selected_actions: list[str],
) -> dict[str, Any]:
    return {
        "prediction_count": len(selected_actions),
        "action_count_considered": len(rows),
        "non_no_trade_candidate_count": len(rows),
        "no_trade_selected_count": sum(
            1 for action in selected_actions if action == "NO_TRADE"
        ),
        "selected_non_no_trade_count": sum(
            1 for action in selected_actions if action != "NO_TRADE"
        ),
        "bucket_missing_count": _bool_count(rows, "bucket_missing"),
        "family_missing_count": _bool_count(rows, "family_missing"),
        "bucket_support_failed_count": _bool_count(rows, "bucket_support_failed"),
        "family_support_failed_count": _bool_count(rows, "family_support_failed"),
        "bucket_lcb_or_mean_failed_count": _bool_count(
            rows,
            "bucket_lcb_or_mean_failed",
        ),
        "family_lcb_or_mean_failed_count": _bool_count(
            rows,
            "family_lcb_or_mean_failed",
        ),
        "bucket_sum_failed_count": _bool_count(rows, "bucket_sum_failed"),
        "family_sum_failed_count": _bool_count(rows, "family_sum_failed"),
        "passed_bucket_and_family_count": _bool_count(
            rows,
            "passed_bucket_and_family",
        ),
        "primary_blocking_stage_counts": dict(
            sorted(Counter(row["primary_blocking_stage"] for row in rows).items())
        ),
    }


def _zero_entry_group_summaries(
    *,
    rows: list[dict[str, Any]],
    field: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return [
        {
            "name": key,
            "action_count_considered": len(group_rows),
            "selected_action_count": _bool_count(
                group_rows,
                "candidate_selected_this_action",
            ),
            "bucket_missing_count": _bool_count(group_rows, "bucket_missing"),
            "family_missing_count": _bool_count(group_rows, "family_missing"),
            "bucket_support_failed_count": _bool_count(
                group_rows,
                "bucket_support_failed",
            ),
            "family_support_failed_count": _bool_count(
                group_rows,
                "family_support_failed",
            ),
            "bucket_lcb_or_mean_failed_count": _bool_count(
                group_rows,
                "bucket_lcb_or_mean_failed",
            ),
            "family_lcb_or_mean_failed_count": _bool_count(
                group_rows,
                "family_lcb_or_mean_failed",
            ),
            "bucket_sum_failed_count": _bool_count(group_rows, "bucket_sum_failed"),
            "family_sum_failed_count": _bool_count(group_rows, "family_sum_failed"),
            "passed_bucket_and_family_count": _bool_count(
                group_rows,
                "passed_bucket_and_family",
            ),
            "primary_blocking_stage_counts": dict(
                sorted(
                    Counter(
                        row["primary_blocking_stage"] for row in group_rows
                    ).items()
                )
            ),
        }
        for key, group_rows in sorted(grouped.items())
    ]


def _near_pass_evidence(
    *,
    evidence: dict[str, dict[str, Any]],
    metric_field: str,
    limit: int,
) -> list[dict[str, Any]]:
    rows = []
    for name, row in evidence.items():
        if row["evidence_passed"]:
            continue
        support_deficit = max(int(row["min_support"]) - int(row["support_count"]), 0)
        metric_deficit = max(
            float(row["execution_buffer"]) - float(row[metric_field]),
            0.0,
        )
        sum_deficit = max(-float(row["realized_return_sum"]), 0.0)
        rows.append(
            {
                "name": name,
                "support_count": row["support_count"],
                "min_support": row["min_support"],
                "support_deficit": support_deficit,
                "realized_return_mean": row["realized_return_mean"],
                "realized_return_sum": row["realized_return_sum"],
                "risk_adjusted_lcb": row["risk_adjusted_lcb"],
                "shrunk_risk_adjusted_lcb": row["shrunk_risk_adjusted_lcb"],
                "metric_field": metric_field,
                "metric_value": row[metric_field],
                "metric_deficit": metric_deficit,
                "sum_deficit": sum_deficit,
                "support_passed": row["support_passed"],
                "sum_positive_passed": row["sum_positive_passed"],
                "mean_exceeds_execution_buffer": row[
                    "mean_exceeds_execution_buffer"
                ],
                "lcb_exceeds_execution_buffer": row["lcb_exceeds_execution_buffer"],
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["support_deficit"],
            row["metric_deficit"],
            row["sum_deficit"],
            -int(row["support_count"]),
            row["name"],
        ),
    )[:limit]


def _overlay_diagnostic_sweeps(
    *,
    validation_examples: tuple[PolymarketPolicyExample, ...],
    calibrated_validation_predictions: tuple[PolymarketPolicyPrediction, ...],
    shadow_examples: tuple[PolymarketPolicyExample, ...],
    raw_shadow_predictions: tuple[PolymarketPolicyPrediction, ...],
    calibrated_shadow_predictions: tuple[PolymarketPolicyPrediction, ...],
    execution_buffer: float,
) -> list[dict[str, Any]]:
    rows = []
    for candidate in (
        {
            "candidate_name": "G_bucketed_lcb_rank_selector",
            "candidate_type": "bucketed_lcb_rank_selector",
            "method": "bucketed_lcb_rank_selector",
            "require_lcb_over_buffer": True,
            "require_positive_only": False,
        },
        {
            "candidate_name": "H_positive_bucket_rank_selector",
            "candidate_type": "positive_bucket_rank_selector",
            "method": "positive_bucket_rank_selector",
            "require_lcb_over_buffer": False,
            "require_positive_only": True,
        },
    ):
        for min_bucket_support in RANKING_OVERLAY_DIAGNOSTIC_MIN_BUCKET_SUPPORT_VALUES:
            for prior_support in (
                RANKING_OVERLAY_DIAGNOSTIC_SHRINKAGE_PRIOR_SUPPORT_VALUES
            ):
                for multiplier in RANKING_OVERLAY_DIAGNOSTIC_BUFFER_MULTIPLIERS:
                    sweep_buffer = float(execution_buffer) * float(multiplier)
                    overlay = _fit_bucket_overlay(
                        examples=validation_examples,
                        predictions=calibrated_validation_predictions,
                        execution_buffer=sweep_buffer,
                        method=candidate["method"],
                        min_bucket_support=min_bucket_support,
                        min_family_support=RANKING_OVERLAY_MIN_FAMILY_SUPPORT,
                        require_lcb_over_buffer=candidate[
                            "require_lcb_over_buffer"
                        ],
                        require_positive_only=candidate["require_positive_only"],
                        shrinkage_prior_support=prior_support,
                    )
                    spec = {
                        "candidate_name": candidate["candidate_name"],
                        "candidate_type": candidate["candidate_type"],
                        "score_source": "fallback",
                        "corrections": {},
                        "correction_group": "none",
                        "eligible_families": None,
                        "ranking_overlay": overlay,
                        "notes": ["diagnostic-only sweep"],
                    }
                    predictions = _apply_candidate_spec(
                        predictions=raw_shadow_predictions,
                        fallback_predictions=calibrated_shadow_predictions,
                        spec=spec,
                    )
                    family_report = build_action_family_eligibility_report(
                        examples=shadow_examples,
                        predictions=predictions,
                        execution_buffer=sweep_buffer,
                    )
                    selected_non_no_trade_count = sum(
                        1 for prediction in predictions if _selected_action(prediction) != "NO_TRADE"
                    )
                    rows.append(
                        {
                            "candidate_name": candidate["candidate_name"],
                            "diagnostic_only": True,
                            "settings": {
                                "min_bucket_support": min_bucket_support,
                                "min_family_support": RANKING_OVERLAY_MIN_FAMILY_SUPPORT,
                                "shrinkage_prior_support": prior_support,
                                "shrinkage_prior_mean": (
                                    RANKING_OVERLAY_SHRINKAGE_PRIOR_MEAN
                                ),
                                "buffer_multiplier": multiplier,
                                "execution_buffer": sweep_buffer,
                            },
                            "passed_bucket_count": sum(
                                1
                                for item in overlay["bucket_evidence"].values()
                                if item["evidence_passed"]
                            ),
                            "passed_family_count": sum(
                                1
                                for item in overlay["family_evidence"].values()
                                if item["evidence_passed"]
                            ),
                            "selected_non_no_trade_count": (
                                selected_non_no_trade_count
                            ),
                            "shadow_high_score_support": family_report[
                                "high_score_support_count"
                            ],
                            "shadow_high_score_mean": family_report[
                                "high_score_realized_return_mean"
                            ],
                            "shadow_high_score_sum": family_report[
                                "high_score_realized_return_sum"
                            ],
                            "source_model_candidate_eligible": False,
                            "promotion_eligible": False,
                            "paper_run_resume_allowed": False,
                            **compact_safety_fields(),
                        }
                    )
    return rows


def _primary_blocking_stage(
    *,
    bucket_missing: bool,
    family_missing: bool,
    bucket_support_failed: bool,
    family_support_failed: bool,
    bucket_metric_failed: bool,
    family_metric_failed: bool,
    bucket_sum_failed: bool,
    family_sum_failed: bool,
    passed_bucket_and_family: bool,
) -> str:
    if passed_bucket_and_family:
        return "passed_bucket_and_family"
    if bucket_missing:
        return "bucket_missing"
    if family_missing:
        return "family_missing"
    if bucket_support_failed:
        return "bucket_support_failed"
    if family_support_failed:
        return "family_support_failed"
    if bucket_metric_failed:
        return "bucket_lcb_or_mean_failed"
    if family_metric_failed:
        return "family_lcb_or_mean_failed"
    if bucket_sum_failed:
        return "bucket_sum_failed"
    if family_sum_failed:
        return "family_sum_failed"
    return "unknown_blocked"


def _overlay_metric_field(overlay: dict[str, Any]) -> str:
    if overlay["method"] == "bucketed_lcb_rank_selector":
        return "shrunk_risk_adjusted_lcb"
    return "realized_return_mean"


def _overlay_metric_passed(
    evidence: dict[str, Any],
    overlay: dict[str, Any],
) -> bool:
    if overlay["method"] == "bucketed_lcb_rank_selector":
        return bool(evidence["lcb_exceeds_execution_buffer"])
    return bool(evidence["mean_exceeds_execution_buffer"])


def _bool_count(rows: list[dict[str, Any]], field: str) -> int:
    return sum(1 for row in rows if bool(row[field]))


def _apply_candidate_spec(
    *,
    predictions: tuple[PolymarketPolicyPrediction, ...],
    fallback_predictions: tuple[PolymarketPolicyPrediction, ...],
    spec: dict[str, Any],
) -> tuple[PolymarketPolicyPrediction, ...]:
    calibrated = []
    enabled_actions = set(_candidate_enabled_actions(spec))
    for prediction, fallback in zip(predictions, fallback_predictions, strict=True):
        if spec["score_source"] == "fallback":
            scores = _score_map(fallback)
        else:
            scores = dict(prediction.expected_return_by_action)
        for action in ACTION_VALUE_LABEL_ACTIONS:
            if spec["correction_group"] == "action_family":
                key = action_value_action_family(action)
            elif spec["correction_group"] == "action":
                key = action
            else:
                key = None
            if key is not None:
                scores[action] = float(scores[action]) + float(
                    spec["corrections"].get(key, 0.0)
                )
            eligible_families = spec.get("eligible_families")
            if (
                eligible_families is not None
                and action != "NO_TRADE"
                and action_value_action_family(action) not in set(eligible_families)
            ):
                scores[action] = -1_000_000.0
            overlay = spec.get("ranking_overlay")
            if overlay is not None:
                scores[action] = _overlay_score(
                    action=action,
                    prediction=fallback,
                    score=float(scores[action]),
                    overlay=overlay,
                )
            if action not in enabled_actions:
                scores[action] = -1_000_000.0
        calibrated.append(
            _prediction_with_scores(
                prediction=fallback,
                scores=scores,
                calibration_id=canonical_json_sha256(
                    {
                        "candidate_name": spec["candidate_name"],
                        "scores": scores,
                        "market_id": prediction.market_id,
                        "decision_ts": prediction.decision_ts,
                    }
                ),
            )
        )
    return tuple(calibrated)


def _candidate_enabled_actions(spec: dict[str, Any]) -> list[str]:
    actions = tuple(spec.get("enabled_actions") or ACTION_VALUE_LABEL_ACTIONS)
    enabled = [action for action in ACTION_VALUE_LABEL_ACTIONS if action in set(actions)]
    if "NO_TRADE" not in enabled:
        enabled.append("NO_TRADE")
    return enabled


def _candidate_disabled_actions(
    spec: dict[str, Any],
    *,
    enabled_actions: list[str],
) -> list[str]:
    configured = spec.get("disabled_actions")
    if configured is not None:
        disabled = [
            action for action in ACTION_VALUE_LABEL_ACTIONS if action in set(configured)
        ]
    else:
        disabled = [
            action for action in ACTION_VALUE_LABEL_ACTIONS if action not in enabled_actions
        ]
    return [action for action in disabled if action != "NO_TRADE"]


def _families_for_actions(actions: list[str]) -> list[str]:
    return sorted(
        {
            action_value_action_family(action)
            for action in actions
            if action != "NO_TRADE"
        }
    )


def _candidate_scoped_family_report(
    *,
    family_report: dict[str, Any],
    enabled_action_families: list[str],
    disabled_action_families: list[str],
) -> dict[str, Any]:
    enabled_families = set(enabled_action_families)
    disabled_families = set(disabled_action_families)
    enabled_actions = {
        action
        for action in ACTION_VALUE_LABEL_ACTIONS
        if action_value_action_family(action) in enabled_families
    }
    disabled_actions = {
        action
        for action in ACTION_VALUE_LABEL_ACTIONS
        if action_value_action_family(action) in disabled_families
    }
    return {
        "enabled_action_family_gate_results": {
            family: gate
            for family, gate in family_report["action_family_gate_results"].items()
            if family in enabled_families
        },
        "disabled_action_family_gate_results": {
            family: family_report["action_family_gate_results"].get(
                family,
                _disabled_gate_placeholder(family),
            )
            for family in sorted(disabled_families)
        },
        "enabled_action_gate_results": {
            action: gate
            for action, gate in family_report["action_gate_results"].items()
            if action in enabled_actions
        },
        "disabled_action_gate_results": {
            action: family_report["action_gate_results"].get(
                action,
                _disabled_gate_placeholder(action),
            )
            for action in sorted(disabled_actions)
        },
    }


def _disabled_gate_placeholder(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "support_count": 0,
        "min_support": ACTION_VALUE_HIGH_SCORE_MIN_SUPPORT,
        "support_passed": False,
        "realized_return_mean": 0.0,
        "realized_return_sum": 0.0,
        "realized_return_mean_exceeds_execution_buffer": False,
        "realized_return_sum_positive": False,
        "execution_buffer": None,
        "gate_passed": False,
        "disabled_for_candidate": True,
        "diagnostic_only": True,
    }


def _prediction_with_scores(
    *,
    prediction: PolymarketPolicyPrediction,
    scores: dict[str, float],
    calibration_id: str,
) -> PolymarketPolicyPrediction:
    ranked = _ranked_actions(scores)
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else best
    best_value = float(scores[best])
    second_value = float(scores[second])
    return replace(
        prediction,
        calibrated_expected_pnl_per_notional_by_action={
            action: float(scores[action]) for action in ACTION_VALUE_LABEL_ACTIONS
        },
        calibrated_best_policy_action=best,
        calibrated_expected_pnl_per_notional=best_value,
        calibrated_second_best_expected_pnl_per_notional=second_value,
        calibrated_action_margin=best_value - second_value,
        action_value_calibration_applied=True,
        action_value_calibration_id=calibration_id,
        calibration_support_count=prediction.calibration_support_count or 0,
        calibration_bucket_count=prediction.calibration_bucket_count
        or len(ACTION_VALUE_LABEL_ACTIONS),
    )


def _fit_group_corrections(
    *,
    examples: tuple[PolymarketPolicyExample, ...],
    predictions: tuple[PolymarketPolicyPrediction, ...],
    group_fn: Any,
) -> dict[str, float]:
    residuals: dict[str, list[float]] = defaultdict(list)
    for example, prediction in zip(examples, predictions, strict=True):
        for action in ACTION_VALUE_LABEL_ACTIONS:
            residuals[str(group_fn(action))].append(
                float(example.action_return_targets[action])
                - float(prediction.expected_return_by_action[action])
            )
    return {
        key: _clamp(_mean(values), -0.50, 0.50)
        for key, values in sorted(residuals.items())
    }


def _fit_pairwise_rank_corrections(
    examples: tuple[PolymarketPolicyExample, ...],
) -> dict[str, float]:
    wins = Counter()
    totals = Counter()
    for example in examples:
        targets = example.action_return_targets
        for action in ACTION_VALUE_LABEL_ACTIONS:
            for other in ACTION_VALUE_LABEL_ACTIONS:
                if action == other:
                    continue
                totals[action] += 1
                if float(targets[action]) > float(targets[other]):
                    wins[action] += 1
    return {
        action: _clamp(((wins[action] / totals[action]) - 0.50) * 0.20, -0.20, 0.20)
        if totals[action]
        else 0.0
        for action in ACTION_VALUE_LABEL_ACTIONS
    }


def _fit_family_prior_penalties(
    *,
    examples: tuple[PolymarketPolicyExample, ...],
    predictions: tuple[PolymarketPolicyPrediction, ...],
    execution_buffer: float,
) -> dict[str, float]:
    rows = []
    for example, prediction in zip(examples, predictions, strict=True):
        action = _selected_action(prediction)
        if action == "NO_TRADE":
            continue
        score = float(_score_map(prediction)[action])
        if score < 0.0:
            continue
        rows.append(
            {
                "family": action_value_action_family(action),
                "realized_return": float(example.action_return_targets[action]),
            }
        )
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row["realized_return"])
    penalties = dict.fromkeys(
        {action_value_action_family(action) for action in ACTION_VALUE_LABEL_ACTIONS},
        0.0,
    )
    for family, returns in grouped.items():
        mean_return = _mean(returns)
        if mean_return <= execution_buffer:
            penalties[family] = -_clamp(execution_buffer - mean_return + 0.01, 0.0, 0.50)
    return penalties


def _fit_bucket_overlay(
    *,
    examples: tuple[PolymarketPolicyExample, ...],
    predictions: tuple[PolymarketPolicyPrediction, ...],
    execution_buffer: float,
    method: str,
    min_bucket_support: int,
    min_family_support: int,
    require_lcb_over_buffer: bool,
    require_positive_only: bool,
    shrinkage_prior_support: int = RANKING_OVERLAY_SHRINKAGE_PRIOR_SUPPORT,
    shrinkage_prior_mean: float = RANKING_OVERLAY_SHRINKAGE_PRIOR_MEAN,
) -> dict[str, Any]:
    bucket_rows: dict[str, list[float]] = defaultdict(list)
    family_rows: dict[str, list[float]] = defaultdict(list)
    for example, prediction in zip(examples, predictions, strict=True):
        for action in ACTION_VALUE_LABEL_ACTIONS:
            if action == "NO_TRADE":
                continue
            bucket_key = _overlay_bucket_key(action=action, prediction=prediction)
            realized_return = float(example.action_return_targets[action])
            bucket_rows[bucket_key].append(realized_return)
            family_rows[
                action_value_fine_action_family(
                    action=action,
                    features=prediction.features,
                )
            ].append(realized_return)
    bucket_evidence = {
        key: _overlay_evidence(
            name=key,
            returns=returns,
            min_support=min_bucket_support,
            execution_buffer=execution_buffer,
            require_lcb_over_buffer=require_lcb_over_buffer,
            require_positive_only=require_positive_only,
            shrinkage_prior_support=shrinkage_prior_support,
            shrinkage_prior_mean=shrinkage_prior_mean,
        )
        for key, returns in sorted(bucket_rows.items())
    }
    family_evidence = {
        key: _overlay_evidence(
            name=key,
            returns=returns,
            min_support=min_family_support,
            execution_buffer=execution_buffer,
            require_lcb_over_buffer=require_lcb_over_buffer,
            require_positive_only=require_positive_only,
            shrinkage_prior_support=shrinkage_prior_support,
            shrinkage_prior_mean=shrinkage_prior_mean,
        )
        for key, returns in sorted(family_rows.items())
    }
    return {
        "schema_version": "bigan-v8-polymarket-ranking-overlay-v1",
        "method": method,
        "fit_split": "validation",
        "evaluation_split": "shadow",
        "uses_shadow_for_fit": False,
        "min_bucket_support": min_bucket_support,
        "min_family_support": min_family_support,
        "execution_buffer": execution_buffer,
        "shrinkage_prior_support": shrinkage_prior_support,
        "shrinkage_prior_mean": shrinkage_prior_mean,
        "require_lcb_over_buffer": require_lcb_over_buffer,
        "require_positive_only": require_positive_only,
        "unsupported_action_score": -1.0,
        "score_combination": (
            "shrunk_lcb_evidence_primary_model_score_tiebreaker"
            if require_lcb_over_buffer
            else "bucket_evidence_dominant_model_score_secondary"
        ),
        "bucket_evidence_weight": 1.0
        if require_lcb_over_buffer
        else RANKING_OVERLAY_H_BUCKET_EVIDENCE_WEIGHT,
        "model_score_weight": RANKING_OVERLAY_G_MODEL_SCORE_TIEBREAKER_WEIGHT
        if require_lcb_over_buffer
        else RANKING_OVERLAY_H_MODEL_SCORE_WEIGHT,
        "bucket_evidence": bucket_evidence,
        "family_evidence": family_evidence,
        **compact_safety_fields(),
    }


def _overlay_evidence(
    *,
    name: str,
    returns: list[float],
    min_support: int,
    execution_buffer: float,
    require_lcb_over_buffer: bool,
    require_positive_only: bool,
    shrinkage_prior_support: int,
    shrinkage_prior_mean: float,
) -> dict[str, Any]:
    support_count = len(returns)
    mean_return = _mean(returns)
    return_sum = sum(float(value) for value in returns)
    stdev = _stdev(returns)
    lcb = mean_return - (stdev / math.sqrt(support_count)) if support_count else 0.0
    shrunk_mean = _shrunk_mean(
        return_sum=return_sum,
        support_count=support_count,
        prior_support=shrinkage_prior_support,
        prior_mean=shrinkage_prior_mean,
    )
    shrunk_lcb = (
        shrunk_mean - (stdev / math.sqrt(support_count)) if support_count else 0.0
    )
    support_passed = support_count >= min_support
    positive_passed = mean_return > 0.0
    sum_positive_passed = return_sum > 0.0
    lcb_passed = shrunk_lcb > execution_buffer
    buffer_passed = mean_return > execution_buffer
    if require_lcb_over_buffer:
        passed = support_passed and lcb_passed and sum_positive_passed
    elif require_positive_only:
        passed = support_passed and positive_passed and buffer_passed and sum_positive_passed
    else:
        passed = support_passed and buffer_passed and sum_positive_passed
    return {
        "name": name,
        "support_count": support_count,
        "min_support": min_support,
        "support_passed": support_passed,
        "realized_return_mean": mean_return,
        "realized_return_sum": return_sum,
        "realized_return_stdev": stdev,
        "risk_adjusted_lcb": lcb,
        "shrunk_realized_return_mean": shrunk_mean,
        "shrunk_risk_adjusted_lcb": shrunk_lcb,
        "shrinkage_prior_support": shrinkage_prior_support,
        "shrinkage_prior_mean": shrinkage_prior_mean,
        "execution_buffer": execution_buffer,
        "positive_passed": positive_passed,
        "sum_positive_passed": sum_positive_passed,
        "mean_exceeds_execution_buffer": buffer_passed,
        "lcb_exceeds_execution_buffer": lcb_passed,
        "evidence_passed": passed,
    }


def _overlay_score(
    *,
    action: str,
    prediction: PolymarketPolicyPrediction,
    score: float,
    overlay: dict[str, Any],
) -> float:
    if action == "NO_TRADE":
        return score
    family = action_value_fine_action_family(
        action=action,
        features=prediction.features,
    )
    bucket_key = _overlay_bucket_key(action=action, prediction=prediction)
    family_evidence = overlay["family_evidence"].get(family)
    bucket_evidence = overlay["bucket_evidence"].get(bucket_key)
    if not family_evidence or not bucket_evidence:
        return float(overlay["unsupported_action_score"])
    if not family_evidence["evidence_passed"] or not bucket_evidence["evidence_passed"]:
        return float(overlay["unsupported_action_score"])
    if overlay["method"] == "bucketed_lcb_rank_selector":
        return float(bucket_evidence["shrunk_risk_adjusted_lcb"]) + (
            float(overlay["model_score_weight"]) * score
        )
    return (
        float(overlay["bucket_evidence_weight"])
        * float(bucket_evidence["realized_return_mean"])
    ) + (float(overlay["model_score_weight"]) * score)


def _shrunk_mean(
    *,
    return_sum: float,
    support_count: int,
    prior_support: int,
    prior_mean: float,
) -> float:
    denominator = support_count + prior_support
    if denominator <= 0:
        return 0.0
    numerator = return_sum + (prior_support * prior_mean)
    return float(numerator / denominator)


def _overlay_bucket_key(
    *,
    action: str,
    prediction: PolymarketPolicyPrediction,
) -> str:
    bucket = action_value_bucket_payload(
        action=action,
        features=prediction.features,
        raw_score=float(prediction.expected_return_by_action[action]),
    )
    return "|".join(
        [
            action,
            str(bucket["action_family"]),
            str(bucket["side"]),
            str(bucket["price_bucket"]),
            str(bucket["time_to_close_bucket"]),
            str(bucket["raw_score_bucket"]),
        ]
    )


def _candidate_manifest(
    *,
    candidate_name: str,
    candidate_type: str,
    ranking_overlay: dict[str, Any] | None,
    source_model_eligible: bool,
    action_family_paper_decision_eligible: bool,
    calibration_quality_passed: bool,
    best_action_concentration_passed: bool,
    p_up_action_disagreement_within_limit: bool,
    high_score_support_count: int,
    high_score_realized_return_mean: float,
    high_score_realized_return_sum: float,
    ineligible_reason_codes: list[str],
    enabled_action_families: list[str],
    disabled_action_families: list[str],
    enabled_actions: list[str],
    disabled_actions: list[str],
    candidate_scoped_p_up_action_disagreement_rate: float,
    candidate_scoped_action_family_gate_results: dict[str, Any],
) -> dict[str, Any]:
    ranking_overlay_used = ranking_overlay is not None
    return {
        "schema_version": "bigan-v8-polymarket-policy-candidate-v1",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": candidate_name,
        "candidate_type": candidate_type,
        "source_model_candidate_eligible": source_model_eligible,
        "ranking_overlay_used": ranking_overlay_used,
        "ranking_overlay_method": None
        if ranking_overlay is None
        else ranking_overlay["method"],
        "ranking_overlay_fit_split": None
        if ranking_overlay is None
        else ranking_overlay["fit_split"],
        "ranking_overlay_evaluation_split": None
        if ranking_overlay is None
        else ranking_overlay["evaluation_split"],
        "ranking_overlay_uses_shadow_split": False
        if ranking_overlay is not None
        else None,
        "ranking_overlay_min_bucket_support": None
        if ranking_overlay is None
        else ranking_overlay["min_bucket_support"],
        "ranking_overlay_min_family_support": None
        if ranking_overlay is None
        else ranking_overlay["min_family_support"],
        "ranking_overlay_shrinkage_prior_support": None
        if ranking_overlay is None
        else ranking_overlay["shrinkage_prior_support"],
        "ranking_overlay_shrinkage_prior_mean": None
        if ranking_overlay is None
        else ranking_overlay["shrinkage_prior_mean"],
        "ranking_overlay_score_combination": None
        if ranking_overlay is None
        else ranking_overlay["score_combination"],
        "ranking_overlay_bucket_evidence_weight": None
        if ranking_overlay is None
        else ranking_overlay["bucket_evidence_weight"],
        "ranking_overlay_model_score_weight": None
        if ranking_overlay is None
        else ranking_overlay["model_score_weight"],
        "action_value_paper_decision_eligible": source_model_eligible,
        "enabled_action_families": enabled_action_families,
        "disabled_action_families": disabled_action_families,
        "enabled_actions": enabled_actions,
        "disabled_actions": disabled_actions,
        "candidate_scoped_p_up_action_disagreement_rate": (
            candidate_scoped_p_up_action_disagreement_rate
        ),
        "candidate_scoped_action_family_gate_results": (
            candidate_scoped_action_family_gate_results
        ),
        "action_family_paper_decision_eligible": action_family_paper_decision_eligible,
        "calibration_quality_passed": calibration_quality_passed,
        "best_action_concentration_passed": best_action_concentration_passed,
        "p_up_action_disagreement_within_limit": (
            p_up_action_disagreement_within_limit
        ),
        "requires_promotion_replay_gate": True,
        "paper_run_resume_allowed": False,
        "paper_run_resume_blocked_reason": "promotion_replay_gate_required",
        "high_score_support_count": high_score_support_count,
        "high_score_realized_return_mean": high_score_realized_return_mean,
        "high_score_realized_return_sum": high_score_realized_return_sum,
        "ineligible_reason_codes": ineligible_reason_codes,
        **compact_safety_fields(),
    }


def _candidate_overlay_payload(spec: dict[str, Any]) -> dict[str, Any]:
    overlay = spec.get("ranking_overlay")
    if overlay is None:
        return {
            "schema_version": "bigan-v8-polymarket-ranking-overlay-v1",
            "candidate_name": spec["candidate_name"],
            "ranking_overlay_used": False,
            "fit_split": "validation",
            "evaluation_split": "shadow",
            "uses_shadow_for_fit": False,
            **compact_safety_fields(),
        }
    payload = dict(overlay)
    payload["candidate_name"] = spec["candidate_name"]
    payload["ranking_overlay_used"] = True
    return payload


def _ranking_breakdowns(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    fields = (
        "action_family",
        "side",
        "price_bucket",
        "time_to_close_bucket",
        "raw_score_bucket",
        "market_family",
    )
    return {field: _group_breakdown(rows, field) for field in fields}


def _group_breakdown(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return [
        {"name": key, **_ranking_metrics(group_rows)}
        for key, group_rows in sorted(grouped.items())
    ]


def _ranking_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sample_count = len(rows)
    return {
        "sample_count": sample_count,
        "top_1_action_hit_rate": _rate(rows, "top_1_action_hit"),
        "top_2_action_hit_rate": _rate(rows, "top_2_action_hit"),
        "top_3_action_hit_rate": _rate(rows, "top_3_action_hit"),
        "mean_score_spread_selected_minus_realized_best": _mean(
            [row["score_spread_selected_minus_realized_best"] for row in rows]
        ),
        "mean_selected_action_realized_return": _mean(
            [row["selected_action_realized_return"] for row in rows]
        ),
        "mean_oracle_best_action_realized_return": _mean(
            [row["oracle_best_action_realized_return"] for row in rows]
        ),
        "mean_regret": _mean([row["regret"] for row in rows]),
    }


def _score_map(prediction: PolymarketPolicyPrediction) -> dict[str, float]:
    if prediction.calibrated_expected_pnl_per_notional_by_action:
        return {
            action: float(prediction.calibrated_expected_pnl_per_notional_by_action[action])
            for action in ACTION_VALUE_LABEL_ACTIONS
        }
    return {
        action: float(prediction.expected_return_by_action[action])
        for action in ACTION_VALUE_LABEL_ACTIONS
    }


def _selected_action(prediction: PolymarketPolicyPrediction) -> str:
    if prediction.calibrated_best_policy_action is not None:
        return str(prediction.calibrated_best_policy_action)
    return str(prediction.best_policy_action)


def _ranked_actions(scores: dict[str, float]) -> list[str]:
    return [
        action
        for action, _ in sorted(
            scores.items(),
            key=lambda item: (-float(item[1]), item[0]),
        )
    ]


def _realized_best_action(example: PolymarketPolicyExample) -> tuple[str, float]:
    action, value = sorted(
        example.action_return_targets.items(),
        key=lambda item: (-float(item[1]), item[0]),
    )[0]
    return action, float(value)


def _mae(
    *,
    examples: tuple[PolymarketPolicyExample, ...],
    predictions: tuple[PolymarketPolicyPrediction, ...],
    score_getter: Any,
    actions: list[str] | tuple[str, ...] = ACTION_VALUE_LABEL_ACTIONS,
) -> float:
    errors = []
    selected_actions = tuple(actions)
    for example, prediction in zip(examples, predictions, strict=True):
        scores = score_getter(prediction)
        for action in selected_actions:
            errors.append(
                abs(
                    float(scores[action])
                    - float(example.action_return_targets[action])
                )
            )
    return _mean(errors)


def _best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        candidates,
        key=lambda candidate: (
            not bool(candidate["source_model_eligible"]),
            -float(candidate["high_score_realized_return_mean"]),
            float(candidate["shadow_mean_regret"]),
            candidate["candidate_name"],
        ),
    )[0]


def _p_up_action_disagrees(prediction: PolymarketPolicyPrediction) -> bool:
    action = _selected_action(prediction)
    p_up = prediction.p_up_auxiliary
    if p_up is None:
        p_up = prediction.estimated_up_probability
    if action.startswith("BUY_DOWN_"):
        return float(p_up) >= P_UP_MATERIAL_DISAGREEMENT_THRESHOLD
    if action.startswith("BUY_UP_"):
        return float(p_up) <= 1.0 - P_UP_MATERIAL_DISAGREEMENT_THRESHOLD
    return False


def _validate_aligned(
    examples: tuple[PolymarketPolicyExample, ...],
    predictions: tuple[PolymarketPolicyPrediction, ...],
) -> None:
    if len(examples) != len(predictions):
        raise ValueError("examples and predictions must have the same length")
    for example, prediction in zip(examples, predictions, strict=True):
        if example.market_id != prediction.market_id:
            raise ValueError("example/prediction market_id mismatch")
        if int(example.decision_ts) != int(prediction.decision_ts):
            raise ValueError("example/prediction decision_ts mismatch")


def _rate(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if bool(row[field])) / len(rows)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)


def _stdev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    variance = sum((float(value) - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)
