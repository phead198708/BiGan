"""Ranking diagnostics and fail-closed source eligibility reports."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Any

from bigan.v8.polymarket.action_value_guards import (
    action_value_action_family,
    action_value_bucket_payload,
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

MODEL_RANKING_ERROR_SCHEMA_VERSION = (
    "bigan-v8-polymarket-model-ranking-error-v1"
)
MODEL_RANKING_CANDIDATE_COMPARISON_SCHEMA_VERSION = (
    "bigan-v8-polymarket-model-ranking-candidate-comparison-v1"
)
SOURCE_MODEL_ELIGIBILITY_SCHEMA_VERSION = (
    "bigan-v8-polymarket-source-model-eligibility-v1"
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
        "| candidate | source_eligible | shadow_raw_mae | shadow_calibrated_mae | high_score_support | high_score_mean | high_score_sum | reasons |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for candidate in report["candidates"]:
        reasons = ", ".join(candidate["ineligible_reason_codes"]) or "none"
        lines.append(
            "| {name} | {eligible} | {raw:.6f} | {cal:.6f} | {support} | "
            "{mean:.6f} | {total:.6f} | {reasons} |".format(
                name=candidate["candidate_name"],
                eligible=str(candidate["source_model_eligible"]).lower(),
                raw=candidate["shadow_raw_mae"],
                cal=candidate["shadow_calibrated_mae"],
                support=candidate["high_score_support_count"],
                mean=candidate["high_score_realized_return_mean"],
                total=candidate["high_score_realized_return_sum"],
                reasons=reasons,
            )
        )
    return "\n".join(lines) + "\n"


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
    ranking = _ranking_split_report(
        split_name="shadow",
        examples=shadow_examples,
        predictions=shadow_predictions,
    )
    raw_mae = _mae(
        examples=shadow_examples,
        predictions=raw_shadow_predictions,
        score_getter=lambda prediction: prediction.expected_return_by_action,
    )
    calibrated_mae = _mae(
        examples=shadow_examples,
        predictions=shadow_predictions,
        score_getter=_score_map,
    )
    family_report = build_action_family_eligibility_report(
        examples=shadow_examples,
        predictions=shadow_predictions,
        execution_buffer=execution_buffer,
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
    disagreement_count = sum(
        _p_up_action_disagrees(prediction) for prediction in shadow_predictions
    )
    disagreement_rate = (
        0.0 if not shadow_predictions else disagreement_count / len(shadow_predictions)
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
        ),
        "ranking_overlay": _candidate_overlay_payload(spec),
        **compact_safety_fields(),
    }


def _apply_candidate_spec(
    *,
    predictions: tuple[PolymarketPolicyPrediction, ...],
    fallback_predictions: tuple[PolymarketPolicyPrediction, ...],
    spec: dict[str, Any],
) -> tuple[PolymarketPolicyPrediction, ...]:
    calibrated = []
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
            family_rows[action_value_action_family(action)].append(realized_return)
    bucket_evidence = {
        key: _overlay_evidence(
            name=key,
            returns=returns,
            min_support=min_bucket_support,
            execution_buffer=execution_buffer,
            require_lcb_over_buffer=require_lcb_over_buffer,
            require_positive_only=require_positive_only,
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
        "shrinkage_prior_support": RANKING_OVERLAY_SHRINKAGE_PRIOR_SUPPORT,
        "shrinkage_prior_mean": RANKING_OVERLAY_SHRINKAGE_PRIOR_MEAN,
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
) -> dict[str, Any]:
    support_count = len(returns)
    mean_return = _mean(returns)
    return_sum = sum(float(value) for value in returns)
    stdev = _stdev(returns)
    lcb = mean_return - (stdev / math.sqrt(support_count)) if support_count else 0.0
    shrunk_mean = _shrunk_mean(
        return_sum=return_sum,
        support_count=support_count,
        prior_support=RANKING_OVERLAY_SHRINKAGE_PRIOR_SUPPORT,
        prior_mean=RANKING_OVERLAY_SHRINKAGE_PRIOR_MEAN,
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
        "shrinkage_prior_support": RANKING_OVERLAY_SHRINKAGE_PRIOR_SUPPORT,
        "shrinkage_prior_mean": RANKING_OVERLAY_SHRINKAGE_PRIOR_MEAN,
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
    family = action_value_action_family(action)
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
) -> float:
    errors = []
    for example, prediction in zip(examples, predictions, strict=True):
        scores = score_getter(prediction)
        for action in ACTION_VALUE_LABEL_ACTIONS:
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
