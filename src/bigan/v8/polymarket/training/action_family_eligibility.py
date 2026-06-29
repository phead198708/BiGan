"""Action-family eligibility diagnostics for calibrated Polymarket policies."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Any

from bigan.v8.polymarket.action_value_guards import (
    ACTION_FAMILY_HIGH_SCORE_UNPROFITABLE,
    ACTION_FAMILY_HOLD_TO_SETTLEMENT,
    ACTION_FAMILY_INELIGIBLE,
    ACTION_FAMILY_NO_TRADE,
    ACTION_FAMILY_SELL_BEFORE_CLOSE,
    BUY_DOWN_HOLD_TO_SETTLEMENT_UNPROFITABLE,
    BUY_UP_HOLD_TO_SETTLEMENT_UNPROFITABLE,
    HOLD_TO_SETTLEMENT_HIGH_SCORE_UNPROFITABLE,
    HOLD_TO_SETTLEMENT_LONGSHOT_GUARD,
    LONGSHOT_GUARD_PRICE_BUCKETS,
    LONGSHOT_GUARD_RAW_SCORE_BUCKETS,
    LONGSHOT_GUARD_TIME_TO_CLOSE_BUCKETS,
    action_value_action_family,
    action_value_bucket_payload,
)
from bigan.v8.polymarket.training.action_value_calibration import (
    ACTION_VALUE_HIGH_SCORE_MIN_SUPPORT,
    ACTION_VALUE_HIGH_SCORE_THRESHOLD,
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
    SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_CANDIDATE_NAME,
    SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_EXIT_POLICY,
    SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_THRESHOLDS,
    SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_CANDIDATE_NAME,
    SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_THRESHOLDS,
    SELL_BEFORE_CLOSE_P_UP_DIAGNOSTIC_ALIGNMENT_MIN,
    SELL_BEFORE_CLOSE_SIDE_BALANCE_THRESHOLDS,
    SELL_BEFORE_CLOSE_SIDE_BALANCED_ENTRY_GUARD_THRESHOLDS,
    SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
    SELL_BEFORE_CLOSE_SUPPORT_AWARE_P_UP_ALIGNED_CANDIDATE_NAME,
)

ACTION_FAMILY_ELIGIBILITY_SCHEMA_VERSION = (
    "bigan-v8-polymarket-action-family-eligibility-v1"
)
HOLD_TO_SETTLEMENT_LONGSHOT_GUARD_SCHEMA_VERSION = (
    "bigan-v8-polymarket-hold-to-settlement-longshot-guard-v1"
)
ACTION_FAMILY_REPLAY_VARIANTS_SCHEMA_VERSION = (
    "bigan-v8-polymarket-action-family-replay-variants-v1"
)
ACTION_FAMILY_COUNTERFACTUAL_REPLAY_SCHEMA_VERSION = (
    "bigan-v8-polymarket-action-family-counterfactual-replay-v1"
)
ACTION_FAMILY_MIN_HIGH_SCORE_SUPPORT = ACTION_VALUE_HIGH_SCORE_MIN_SUPPORT
P_UP_MATERIAL_DISAGREEMENT_THRESHOLD = 0.55
M_POSITION_STATE_MAX_PAPER_NOTIONAL = 0.20
M_EXECUTION_PNL_AWARE_MODEL_SCORE_WEIGHT = 0.20
M_EXECUTION_PNL_AWARE_IMMEDIATE_EXIT_RETURN_WEIGHT = 8.0
M_EXECUTION_PNL_AWARE_MARGIN_WEIGHT = 0.10
M_EXECUTION_PNL_AWARE_QUALITY_WEIGHT = 1.0
M_EXECUTION_PNL_AWARE_GAP_PENALTY_WEIGHT = 0.05


def build_action_family_eligibility_report(
    *,
    examples: tuple[PolymarketPolicyExample, ...],
    predictions: tuple[PolymarketPolicyPrediction, ...],
    execution_buffer: float,
    min_support: int = ACTION_FAMILY_MIN_HIGH_SCORE_SUPPORT,
    high_score_threshold: float = ACTION_VALUE_HIGH_SCORE_THRESHOLD,
) -> dict[str, Any]:
    """Build out-of-sample high-score action-family gate diagnostics."""

    rows = _high_score_rows(
        examples=examples,
        predictions=predictions,
        high_score_threshold=high_score_threshold,
    )
    family_gate_results = _gate_results(
        rows=rows,
        group_field="action_family",
        execution_buffer=execution_buffer,
        min_support=min_support,
    )
    action_gate_results = _gate_results(
        rows=rows,
        group_field="action",
        execution_buffer=execution_buffer,
        min_support=min_support,
    )
    fine_family_gate_results = _gate_results(
        rows=rows,
        group_field="fine_action_family",
        execution_buffer=execution_buffer,
        min_support=min_support,
    )
    enabled_families = sorted(
        family
        for family, gate in family_gate_results.items()
        if int(gate["support_count"]) > 0
    )
    eligible_families = sorted(
        family
        for family, gate in family_gate_results.items()
        if int(gate["support_count"]) > 0 and bool(gate["gate_passed"])
    )
    ineligible_families = sorted(set(enabled_families) - set(eligible_families))
    reason_codes = _eligibility_reason_codes(
        enabled_families=enabled_families,
        ineligible_families=ineligible_families,
        action_gate_results=action_gate_results,
    )
    paper_decision_eligible = bool(enabled_families) and not reason_codes
    return {
        "schema_version": ACTION_FAMILY_ELIGIBILITY_SCHEMA_VERSION,
        "policy_schema_version": POLYMARKET_POLICY_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "replay_split": "shadow",
        "out_of_sample_replay": True,
        "high_score_threshold": high_score_threshold,
        "min_family_high_score_support": min_support,
        "family_high_score_execution_buffer": execution_buffer,
        "high_score_support_count": len(rows),
        "high_score_realized_return_mean": _mean(
            [row["realized_return"] for row in rows]
        ),
        "high_score_realized_return_sum": _sum(
            [row["realized_return"] for row in rows]
        ),
        "high_score_calibrated_score_mean": _mean(
            [row["calibrated_score"] for row in rows]
        ),
        "enabled_action_families": enabled_families,
        "eligible_action_families": eligible_families,
        "ineligible_action_families": ineligible_families,
        "action_family_gate_results": family_gate_results,
        "action_gate_results": action_gate_results,
        "fine_action_family_gate_results": fine_family_gate_results,
        "action_family_paper_decision_eligible": paper_decision_eligible,
        "action_family_paper_decision_ineligible_reasons": reason_codes,
        "reason_codes": reason_codes,
        "high_score_by_action": _group_summaries(rows, ("action",)),
        "high_score_by_action_family": _group_summaries(rows, ("action_family",)),
        "high_score_by_fine_action_family": _group_summaries(
            rows,
            ("fine_action_family",),
        ),
        "high_score_by_side": _group_summaries(rows, ("side",)),
        "high_score_by_price_bucket": _group_summaries(rows, ("price_bucket",)),
        "high_score_by_time_to_close_bucket": _group_summaries(
            rows,
            ("time_to_close_bucket",),
        ),
        "high_score_by_side_exit_policy_price_time_bucket": _group_summaries(
            rows,
            ("side", "intended_exit_policy", "price_bucket", "time_to_close_bucket"),
        ),
        "high_score_by_raw_score_bucket": _group_summaries(
            rows,
            ("raw_score_bucket",),
        ),
        "high_score_by_action_family_side_price_time_raw_bucket": _group_summaries(
            rows,
            (
                "action_family",
                "side",
                "price_bucket",
                "time_to_close_bucket",
                "raw_score_bucket",
            ),
        ),
        "negative_high_score_examples": _negative_examples(rows),
        **compact_safety_fields(),
    }


def build_hold_to_settlement_longshot_guard_report(
    *,
    examples: tuple[PolymarketPolicyExample, ...],
    predictions: tuple[PolymarketPolicyPrediction, ...],
    execution_buffer: float,
    high_score_threshold: float = ACTION_VALUE_HIGH_SCORE_THRESHOLD,
) -> dict[str, Any]:
    """Build diagnostics for the initial HOLD_TO_SETTLEMENT long-shot guard."""

    rows = _high_score_rows(
        examples=examples,
        predictions=predictions,
        high_score_threshold=high_score_threshold,
    )
    guarded_rows = [
        row for row in rows if row["hold_to_settlement_longshot_guard_applies"]
    ]
    return {
        "schema_version": HOLD_TO_SETTLEMENT_LONGSHOT_GUARD_SCHEMA_VERSION,
        "policy_schema_version": POLYMARKET_POLICY_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "replay_split": "shadow",
        "out_of_sample_replay": True,
        "guard_enabled": True,
        "guard_mode": "block_to_no_trade",
        "guard_reason_codes": [
            HOLD_TO_SETTLEMENT_LONGSHOT_GUARD,
            ACTION_FAMILY_INELIGIBLE,
        ],
        "price_buckets": list(LONGSHOT_GUARD_PRICE_BUCKETS),
        "time_to_close_buckets": list(LONGSHOT_GUARD_TIME_TO_CLOSE_BUCKETS),
        "raw_score_buckets": list(LONGSHOT_GUARD_RAW_SCORE_BUCKETS),
        "high_score_threshold": high_score_threshold,
        "execution_buffer": execution_buffer,
        "high_score_support_count": len(rows),
        "guarded_high_score_count": len(guarded_rows),
        "guarded_high_score_realized_return_mean": _mean(
            [row["realized_return"] for row in guarded_rows]
        ),
        "guarded_high_score_realized_return_sum": _sum(
            [row["realized_return"] for row in guarded_rows]
        ),
        "guarded_by_action": _group_summaries(guarded_rows, ("action",)),
        "guarded_by_side": _group_summaries(guarded_rows, ("side",)),
        "guarded_by_price_time_raw_bucket": _group_summaries(
            guarded_rows,
            ("price_bucket", "time_to_close_bucket", "raw_score_bucket"),
        ),
        "negative_guarded_examples": _negative_examples(guarded_rows),
        **compact_safety_fields(),
    }


def build_action_family_replay_variants_report(
    *,
    examples: tuple[PolymarketPolicyExample, ...],
    predictions: tuple[PolymarketPolicyPrediction, ...],
    execution_buffer: float,
    thresholds: tuple[float, ...] = (0.0, 0.03, 0.05),
    min_support: int = ACTION_FAMILY_MIN_HIGH_SCORE_SUPPORT,
) -> dict[str, Any]:
    """Estimate before/after policy variants on the same shadow evidence set."""

    baseline_rows = _high_score_rows(
        examples=examples,
        predictions=predictions,
        high_score_threshold=ACTION_VALUE_HIGH_SCORE_THRESHOLD,
    )
    baseline_gates = _gate_results(
        rows=baseline_rows,
        group_field="action_family",
        execution_buffer=execution_buffer,
        min_support=min_support,
    )
    passed_baseline_families = {
        family for family, gate in baseline_gates.items() if gate["gate_passed"]
    }
    passed_bucket_keys = _passed_bucket_keys(
        rows=baseline_rows,
        execution_buffer=execution_buffer,
        min_support=min_support,
    )
    variants = [
        _variant_report(
            variant="A_baseline_current_calibrated_policy_blocked",
            rows=baseline_rows,
            candidate_rows=baseline_rows,
            threshold=ACTION_VALUE_HIGH_SCORE_THRESHOLD,
            gate_mode="no_action_family_filter",
            execution_buffer=execution_buffer,
            min_support=min_support,
            blocked=True,
            reason_codes=[ACTION_FAMILY_HIGH_SCORE_UNPROFITABLE],
        ),
        _variant_report(
            variant="B_hold_to_settlement_disabled",
            rows=[
                row
                for row in baseline_rows
                if row["action_family"] != ACTION_FAMILY_HOLD_TO_SETTLEMENT
            ],
            candidate_rows=baseline_rows,
            threshold=ACTION_VALUE_HIGH_SCORE_THRESHOLD,
            gate_mode="hold_to_settlement_disabled",
            execution_buffer=execution_buffer,
            min_support=min_support,
        ),
        _variant_report(
            variant="C_sell_before_close_only",
            rows=[
                row
                for row in baseline_rows
                if row["action_family"] == ACTION_FAMILY_SELL_BEFORE_CLOSE
            ],
            candidate_rows=baseline_rows,
            threshold=ACTION_VALUE_HIGH_SCORE_THRESHOLD,
            gate_mode="sell_before_close_only",
            execution_buffer=execution_buffer,
            min_support=min_support,
        ),
        _variant_report(
            variant="D_hold_to_settlement_allowed_only_for_passed_buckets",
            rows=[
                row
                for row in baseline_rows
                if (
                    row["action_family"] != ACTION_FAMILY_HOLD_TO_SETTLEMENT
                    and row["action_family"] in passed_baseline_families
                )
                or (
                    row["action_family"] == ACTION_FAMILY_HOLD_TO_SETTLEMENT
                    and row["action_family"] in passed_baseline_families
                    and _bucket_key(row) in passed_bucket_keys
                )
            ],
            candidate_rows=baseline_rows,
            threshold=ACTION_VALUE_HIGH_SCORE_THRESHOLD,
            gate_mode="passed_family_and_bucket_only",
            execution_buffer=execution_buffer,
            min_support=min_support,
        ),
    ]
    threshold_sweep = []
    for threshold in thresholds:
        candidate_rows = _high_score_rows(
            examples=examples,
            predictions=predictions,
            high_score_threshold=threshold,
        )
        family_gates = _gate_results(
            rows=candidate_rows,
            group_field="action_family",
            execution_buffer=execution_buffer,
            min_support=min_support,
        )
        passed_families = {
            family for family, gate in family_gates.items() if gate["gate_passed"]
        }
        selected_rows = [
            row for row in candidate_rows if row["action_family"] in passed_families
        ]
        threshold_sweep.append(
            _variant_report(
                variant=f"E_threshold_{threshold:.2f}_action_family_gates_enabled",
                rows=selected_rows,
                candidate_rows=candidate_rows,
                threshold=threshold,
                gate_mode="action_family_gates_enabled",
                execution_buffer=execution_buffer,
                min_support=min_support,
                family_gate_results=family_gates,
                eligible_action_families=sorted(passed_families),
                reason_codes=(
                    []
                    if passed_families
                    else [ACTION_FAMILY_HIGH_SCORE_UNPROFITABLE]
                ),
            )
        )
    return {
        "schema_version": ACTION_FAMILY_REPLAY_VARIANTS_SCHEMA_VERSION,
        "policy_schema_version": POLYMARKET_POLICY_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "report_mode": "filtered_high_score_estimate",
        "promotion_evidence_eligible": False,
        "counterfactual_replay_required_for_promotion": True,
        "replay_split": "shadow",
        "out_of_sample_replay": True,
        "execution_buffer": execution_buffer,
        "min_family_high_score_support": min_support,
        "variants": variants,
        "threshold_sweep_with_action_family_gates": threshold_sweep,
        **compact_safety_fields(),
    }


def build_action_family_counterfactual_prediction_sets(
    *,
    examples: tuple[PolymarketPolicyExample, ...],
    predictions: tuple[PolymarketPolicyPrediction, ...],
    execution_buffer: float,
    support_aware_thresholds: dict[str, float] | None = None,
    support_aware_threshold_selection_report: dict[str, Any] | None = None,
    thresholds: tuple[float, ...] = (0.0, 0.03, 0.05),
    min_support: int = ACTION_FAMILY_MIN_HIGH_SCORE_SUPPORT,
) -> tuple[dict[str, Any], ...]:
    """Build re-ranked counterfactual prediction sets for replay."""

    _validate_aligned(examples, predictions)
    baseline_rows = _high_score_rows(
        examples=examples,
        predictions=predictions,
        high_score_threshold=ACTION_VALUE_HIGH_SCORE_THRESHOLD,
    )
    baseline_gates = _gate_results(
        rows=baseline_rows,
        group_field="action_family",
        execution_buffer=execution_buffer,
        min_support=min_support,
    )
    passed_baseline_families = {
        family for family, gate in baseline_gates.items() if gate["gate_passed"]
    }
    passed_bucket_keys = _passed_bucket_keys(
        rows=baseline_rows,
        execution_buffer=execution_buffer,
        min_support=min_support,
    )
    variants = [
        _counterfactual_variant(
            variant="A_baseline_current_policy_with_runtime_guards",
            predictions=predictions,
            ev_threshold=execution_buffer,
            allowed_mode="baseline",
            family_gate_results=baseline_gates,
            eligible_action_families=sorted(passed_baseline_families),
            description="baseline calibrated policy replay using runtime guards",
        ),
        _counterfactual_variant(
            variant="B_hold_to_settlement_disabled_reranked",
            predictions=predictions,
            ev_threshold=execution_buffer,
            allowed_mode="hold_to_settlement_disabled",
            description="disable HOLD_TO_SETTLEMENT, then re-rank remaining calibrated actions",
        ),
        _counterfactual_variant(
            variant="C_sell_before_close_only_reranked",
            predictions=predictions,
            ev_threshold=execution_buffer,
            allowed_mode="sell_before_close_only",
            description="allow SELL_BEFORE_CLOSE actions only, then re-rank",
        ),
        _counterfactual_variant(
            variant="I_sell_before_close_only_source_candidate",
            predictions=predictions,
            ev_threshold=execution_buffer,
            allowed_mode="sell_before_close_only",
            description=(
                "source-model candidate replay scoped to SELL_BEFORE_CLOSE actions only"
            ),
        ),
        _counterfactual_variant(
            variant=SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_CANDIDATE_NAME,
            predictions=predictions,
            ev_threshold=execution_buffer,
            allowed_mode="sell_before_close_exit_reliability_guard",
            description=(
                "SELL_BEFORE_CLOSE source candidate with causal entry guard "
                "and policy-constrained pre-settlement exits"
            ),
            exit_reliability_guard_enabled=True,
            exit_policy=SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_EXIT_POLICY,
            entry_filter_thresholds=dict(
                SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_THRESHOLDS
            ),
        ),
        _counterfactual_variant(
            variant=SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_CANDIDATE_NAME,
            predictions=predictions,
            ev_threshold=execution_buffer,
            allowed_mode="sell_before_close_exit_reliability_p_up_aligned",
            description=(
                "SELL_BEFORE_CLOSE source candidate with causal p_up-aligned "
                "entry guard and policy-constrained exits"
            ),
            exit_reliability_guard_enabled=True,
            p_up_side_alignment_filter_enabled=True,
            exit_policy=SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_EXIT_POLICY,
            entry_filter_thresholds=dict(
                SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_THRESHOLDS
            ),
        ),
        build_sell_before_close_support_aware_prediction_set(
            predictions=predictions,
            execution_buffer=execution_buffer,
            entry_filter_thresholds=support_aware_thresholds,
            threshold_selection_report=support_aware_threshold_selection_report,
        ),
        build_sell_before_close_side_balanced_prediction_set(
            predictions=predictions,
            execution_buffer=execution_buffer,
        ),
        _counterfactual_variant(
            variant="D_hold_to_settlement_allowed_only_for_passed_buckets_reranked",
            predictions=predictions,
            ev_threshold=execution_buffer,
            allowed_mode="passed_family_and_bucket_only",
            family_gate_results=baseline_gates,
            eligible_action_families=sorted(passed_baseline_families),
            passed_bucket_keys=passed_bucket_keys,
            description="allow actions only when family and HOLD bucket gates pass",
        ),
    ]
    for threshold in thresholds:
        candidate_rows = _high_score_rows(
            examples=examples,
            predictions=predictions,
            high_score_threshold=threshold,
        )
        family_gates = _gate_results(
            rows=candidate_rows,
            group_field="action_family",
            execution_buffer=execution_buffer,
            min_support=min_support,
        )
        passed_families = {
            family for family, gate in family_gates.items() if gate["gate_passed"]
        }
        variants.append(
            _counterfactual_variant(
                variant=f"E_threshold_{threshold:.2f}_action_family_gates_reranked",
                predictions=predictions,
                ev_threshold=threshold,
                allowed_mode="action_family_gates_enabled",
                family_gate_results=family_gates,
                eligible_action_families=sorted(passed_families),
                description=(
                    "re-rank using only action families that pass gates at the "
                    f"{threshold:.2f} high-score threshold"
                ),
            )
        )
    return tuple(variants)


def build_sell_before_close_support_aware_prediction_set(
    *,
    predictions: tuple[PolymarketPolicyPrediction, ...],
    execution_buffer: float,
    entry_filter_thresholds: dict[str, float] | None,
    threshold_selection_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the L support-aware p_up-aligned counterfactual prediction set."""

    selection_failed = not entry_filter_thresholds
    variant = _counterfactual_variant(
        variant=SELL_BEFORE_CLOSE_SUPPORT_AWARE_P_UP_ALIGNED_CANDIDATE_NAME,
        predictions=predictions,
        ev_threshold=execution_buffer,
        allowed_mode="support_aware_selection_failed_no_trade"
        if selection_failed
        else "sell_before_close_support_aware_p_up_aligned",
        description=(
            "support-aware SELL_BEFORE_CLOSE source candidate disabled because "
            "validation threshold selection failed"
            if selection_failed
            else "support-aware SELL_BEFORE_CLOSE source candidate with "
            "validation-fitted entry thresholds and causal p_up-aligned exits"
        ),
        exit_reliability_guard_enabled=not selection_failed,
        p_up_side_alignment_filter_enabled=True,
        exit_policy=SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_EXIT_POLICY,
        entry_filter_thresholds=dict(entry_filter_thresholds or {}),
    )
    variant["threshold_selection_failed"] = selection_failed
    variant["threshold_selection_method"] = (
        "validation_fitted_support_aware_thresholds"
    )
    variant["threshold_selection_fit_split"] = "validation"
    variant["threshold_selection_evaluation_split"] = "shadow"
    variant["uses_shadow_for_fit"] = False
    variant["shadow_sweep_not_used_for_threshold_fit"] = True
    if threshold_selection_report is not None:
        variant["support_aware_threshold_selection_summary"] = {
            "selected_thresholds": threshold_selection_report.get(
                "selected_thresholds",
                {},
            ),
            "validation_row_count": threshold_selection_report.get(
                "validation_row_count",
                0,
            ),
            "validation_passing_row_count": threshold_selection_report.get(
                "validation_passing_row_count",
                0,
            ),
            "selection_reason_codes": threshold_selection_report.get(
                "selection_reason_codes",
                [],
            ),
        }
    return variant


def build_sell_before_close_side_balanced_prediction_set(
    *,
    predictions: tuple[PolymarketPolicyPrediction, ...],
    execution_buffer: float,
    side_balance_thresholds: dict[str, Any] | None = None,
    guard_thresholds: dict[str, float] | None = None,
    enforce_p_up_alignment: bool = False,
) -> dict[str, Any]:
    """Build the M side-balanced SELL_BEFORE_CLOSE ranking prediction set."""

    thresholds = dict(SELL_BEFORE_CLOSE_SIDE_BALANCE_THRESHOLDS)
    if side_balance_thresholds:
        thresholds.update(side_balance_thresholds)
    entry_filter_thresholds = dict(SELL_BEFORE_CLOSE_SIDE_BALANCED_ENTRY_GUARD_THRESHOLDS)
    if guard_thresholds:
        entry_filter_thresholds.update(guard_thresholds)
    reranked = [
        _rerank_counterfactual_prediction(
            prediction=prediction,
            allowed_mode="sell_before_close_only",
            eligible_action_families=(),
            passed_bucket_keys=set(),
        )
        for prediction in predictions
    ]
    candidate_rows = _side_balance_candidate_rows(
        predictions=tuple(reranked),
        execution_buffer=execution_buffer,
        guard_thresholds=entry_filter_thresholds,
        enforce_p_up_alignment=enforce_p_up_alignment,
    )
    _annotate_position_state_fresh_entry_candidates(
        rows=candidate_rows,
        predictions=tuple(reranked),
        guard_thresholds=entry_filter_thresholds,
    )
    selection_pool_rows = [
        row
        for row in candidate_rows
        if bool(row["side_balance_guard_compatible_entry"])
        and bool(row["position_state_fresh_entry_compatible"])
    ]
    selected_keys = _side_quota_selected_keys(
        rows=selection_pool_rows,
        thresholds=thresholds,
    )
    replay_predictions = tuple(
        _side_balance_prediction(
            prediction=prediction,
            selected_keys=selected_keys,
        )
        for prediction in reranked
    )
    ranked_rows = _side_balance_ranked_rows(
        rows=candidate_rows,
        selected_keys=selected_keys,
        thresholds=thresholds,
    )
    summary = _side_balance_selection_summary(
        rows=ranked_rows,
        thresholds=thresholds,
    )
    return {
        "variant": SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
        "description": (
            "side-balanced SELL_BEFORE_CLOSE source candidate with deterministic "
            "position-state-aware side quota ranking and causal guarded exits"
        ),
        "counterfactual_replay_mode": "re_ranked_counterfactual_policy_replay",
        "allowed_mode": "sell_before_close_side_balanced_ranking",
        "ev_threshold": execution_buffer,
        "eligible_action_families": ["SELL_BEFORE_CLOSE"],
        "family_gate_results": {},
        "prediction_count": len(replay_predictions),
        "predictions": replay_predictions,
        "exit_reliability_guard_enabled": True,
        "p_up_side_alignment_filter_enabled": bool(enforce_p_up_alignment),
        "p_up_side_alignment_diagnostic_enabled": True,
        "p_up_side_alignment_diagnostic_threshold": (
            SELL_BEFORE_CLOSE_P_UP_DIAGNOSTIC_ALIGNMENT_MIN
        ),
        "exit_policy": SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_EXIT_POLICY,
        "entry_filter_thresholds": dict(entry_filter_thresholds),
        "position_state_aware_selection_enabled": True,
        "execution_pnl_aware_ranking_enabled": True,
        "rank_score_components": (
            "0.20*calibrated_action_score + 0.10*best_action_margin + "
            "entry_exit_quality_score + 8.00*immediate_exit_return - "
            "0.05*model_vs_immediate_exit_pnl_gap_estimate"
        ),
        "side_balance_required": True,
        "side_balance_thresholds": thresholds,
        "side_balance_candidate_entries": ranked_rows,
        "side_balance_selection_summary": summary,
        "side_balance_selection_fit_split": "validation",
        "side_balance_selection_evaluation_split": "shadow",
        "uses_shadow_for_fit": False,
        "shadow_sweep_not_used_for_fit": True,
        **compact_safety_fields(),
    }


def _side_balance_candidate_rows(
    *,
    predictions: tuple[PolymarketPolicyPrediction, ...],
    execution_buffer: float,
    guard_thresholds: dict[str, float],
    enforce_p_up_alignment: bool,
) -> list[dict[str, Any]]:
    rows = []
    for prediction in predictions:
        action = str(prediction.calibrated_best_policy_action)
        if action not in {
            "BUY_UP_SELL_BEFORE_CLOSE",
            "BUY_DOWN_SELL_BEFORE_CLOSE",
        }:
            continue
        score = float(prediction.calibrated_expected_pnl_per_notional or 0.0)
        if score < execution_buffer:
            continue
        side = "UP" if action.startswith("BUY_UP_") else "DOWN"
        margin = float(prediction.calibrated_action_margin or 0.0)
        p_up = _p_up(prediction)
        entry_exit_quality = _side_balance_entry_exit_quality_score(
            prediction=prediction,
            side=side,
            guard_thresholds=guard_thresholds,
            model_score=score,
        )
        rank_components = _side_balance_execution_pnl_aware_rank_components(
            model_score=score,
            margin=margin,
            entry_exit_quality=entry_exit_quality,
        )
        guard = evaluate_sell_before_close_guard_compatibility(
            prediction=prediction,
            action=action,
            execution_buffer=execution_buffer,
            thresholds=guard_thresholds,
            enforce_p_up_alignment=enforce_p_up_alignment,
        )
        rows.append(
            {
                "market_id": prediction.market_id,
                "decision_ts": int(prediction.decision_ts),
                "action": action,
                "selected_side": side,
                "side_balance_bucket": side,
                "candidate_rank_score": rank_components["candidate_rank_score"],
                "execution_pnl_aware_ranking_enabled": True,
                "raw_calibrated_action_score": score,
                "best_action_margin": margin,
                **entry_exit_quality,
                **rank_components,
                "p_up": p_up,
                "side_balance_guard_compatible_entry": guard["passed"],
                "exit_reliability_guard_passed": guard[
                    "exit_reliability_guard_passed"
                ],
                "p_up_side_alignment_passed": guard["p_up_side_alignment_passed"],
                "p_up_side_alignment_filter_enabled": guard[
                    "p_up_side_alignment_filter_enabled"
                ],
                "p_up_side_alignment_diagnostic_only": guard[
                    "p_up_side_alignment_diagnostic_only"
                ],
                "side_balance_guard_reason_codes": guard["reason_codes"],
                "position_state_aware_selection_enabled": True,
                "position_state_selection_evaluated": False,
                "position_state_fresh_entry_compatible": False,
                "position_state_replay_action_if_selected": None,
                "position_state_open_side_before_decision": None,
                "position_state_reason_codes": [],
                "side_quota_rank": None,
                "side_quota_selected": False,
                "side_balance_reason_codes": [],
            }
        )
    return rows


def _side_balance_entry_exit_quality_score(
    *,
    prediction: PolymarketPolicyPrediction,
    side: str,
    guard_thresholds: dict[str, float],
    model_score: float,
) -> dict[str, Any]:
    features = prediction.features
    bid = _side_balance_side_feature(features, side, "bid")
    ask = _side_balance_side_feature(features, side, "ask")
    liquidity = _side_balance_side_feature(
        features,
        side,
        "executable_bid_notional",
    )
    queue_fill = _side_balance_side_feature(
        features,
        side,
        "queue_fill_probability_proxy",
    )
    spread = _side_balance_side_feature(features, side, "spread_bps")
    staleness = _side_balance_side_feature(features, side, "book_staleness_ms")
    if staleness is None:
        staleness = _side_balance_side_feature(features, side, "book_update_lag_ms")
    time_to_close = float(features.get("time_to_close_seconds", 0.0))
    max_notional = float(
        guard_thresholds.get(
            "min_executable_bid_notional",
            M_POSITION_STATE_MAX_PAPER_NOTIONAL,
        )
    )
    exit_edge = 0.0 if bid is None or ask is None else float(bid) - float(ask)
    liquidity_ratio = 0.0 if liquidity is None else min(2.0, float(liquidity) / max_notional)
    queue_component = 0.0 if queue_fill is None else max(0.0, min(1.0, queue_fill)) * 0.05
    liquidity_component = liquidity_ratio * 0.02
    spread_penalty = 0.0 if spread is None else min(0.05, max(0.0, spread) / 10_000.0)
    staleness_penalty = (
        0.0
        if staleness is None
        else min(
            0.05,
            max(0.0, staleness)
            / max(1.0, float(guard_thresholds["max_book_staleness_ms"]))
            * 0.01,
        )
    )
    time_component = min(0.02, max(0.0, time_to_close) / 300.0 * 0.01)
    quality_score = (
        exit_edge * 0.25
        + queue_component
        + liquidity_component
        + time_component
        - spread_penalty
        - staleness_penalty
    )
    paper_notional = _m_position_state_paper_notional(
        score=model_score,
        guard_thresholds=guard_thresholds,
    )
    entry_shares = 0.0 if ask is None or ask <= 0.0 else paper_notional / float(ask)
    immediate_exit_proceeds = entry_shares * float(bid or 0.0)
    immediate_exit_pnl = immediate_exit_proceeds - paper_notional
    immediate_exit_return = (
        0.0 if paper_notional <= 0.0 else immediate_exit_pnl / paper_notional
    )
    model_expected_pnl = model_score * paper_notional
    model_vs_immediate_exit_pnl_gap = max(
        0.0,
        model_expected_pnl - immediate_exit_pnl,
    )
    return {
        "entry_exit_quality_score": quality_score,
        "exit_quality_bid": bid,
        "entry_quality_ask": ask,
        "entry_exit_quality_edge": exit_edge,
        "entry_exit_quality_liquidity_ratio": liquidity_ratio,
        "entry_exit_quality_queue_fill": queue_fill,
        "entry_exit_quality_spread_bps": spread,
        "entry_exit_quality_book_staleness_ms": staleness,
        "entry_exit_quality_time_to_close_seconds": time_to_close,
        "execution_pnl_entry_notional": paper_notional,
        "execution_pnl_entry_share_qty": entry_shares,
        "execution_pnl_immediate_exit_proceeds": immediate_exit_proceeds,
        "execution_pnl_immediate_exit_pnl": immediate_exit_pnl,
        "execution_pnl_immediate_exit_return": immediate_exit_return,
        "execution_pnl_model_expected_pnl": model_expected_pnl,
        "execution_pnl_model_vs_immediate_exit_pnl_gap_estimate": (
            model_vs_immediate_exit_pnl_gap
        ),
    }


def _side_balance_execution_pnl_aware_rank_components(
    *,
    model_score: float,
    margin: float,
    entry_exit_quality: dict[str, Any],
) -> dict[str, float]:
    model_score_component = model_score * M_EXECUTION_PNL_AWARE_MODEL_SCORE_WEIGHT
    margin_component = margin * M_EXECUTION_PNL_AWARE_MARGIN_WEIGHT
    entry_exit_quality_component = (
        float(entry_exit_quality["entry_exit_quality_score"])
        * M_EXECUTION_PNL_AWARE_QUALITY_WEIGHT
    )
    immediate_exit_component = (
        float(entry_exit_quality["execution_pnl_immediate_exit_return"])
        * M_EXECUTION_PNL_AWARE_IMMEDIATE_EXIT_RETURN_WEIGHT
    )
    gap_penalty = (
        float(entry_exit_quality["execution_pnl_model_vs_immediate_exit_pnl_gap_estimate"])
        * M_EXECUTION_PNL_AWARE_GAP_PENALTY_WEIGHT
    )
    rank_score = (
        model_score_component
        + margin_component
        + entry_exit_quality_component
        + immediate_exit_component
        - gap_penalty
    )
    return {
        "candidate_rank_score": rank_score,
        "execution_pnl_aware_model_score_component": model_score_component,
        "execution_pnl_aware_margin_component": margin_component,
        "execution_pnl_aware_entry_exit_quality_component": (
            entry_exit_quality_component
        ),
        "execution_pnl_aware_immediate_exit_return_component": (
            immediate_exit_component
        ),
        "execution_pnl_aware_gap_penalty_component": gap_penalty,
        "execution_pnl_aware_rank_score": rank_score,
    }


def evaluate_sell_before_close_guard_compatibility(
    *,
    prediction: PolymarketPolicyPrediction,
    action: str,
    execution_buffer: float,
    thresholds: dict[str, float],
    enforce_p_up_alignment: bool = True,
) -> dict[str, Any]:
    side = "UP" if action.startswith("BUY_UP") else "DOWN"
    features = prediction.features
    time_to_close = float(features.get("time_to_close_seconds", 0.0))
    executable_bid_notional = _side_balance_side_feature(
        features,
        side,
        "executable_bid_notional",
    )
    queue_fill = _side_balance_side_feature(
        features,
        side,
        "queue_fill_probability_proxy",
    )
    spread = _side_balance_side_feature(features, side, "spread_bps")
    staleness = _side_balance_side_feature(features, side, "book_staleness_ms")
    if staleness is None:
        staleness = _side_balance_side_feature(
            features,
            side,
            "book_update_lag_ms",
        )
    recent_updates = _side_balance_side_feature(
        features,
        side,
        "recent_book_update_count_1m",
    )
    best_margin = float(prediction.calibrated_action_margin or 0.0)
    best_score = float(prediction.calibrated_expected_pnl_per_notional or 0.0)
    exit_reasons = []
    if time_to_close < float(thresholds["min_seconds_to_close"]):
        exit_reasons.append("entry_blocked_too_close_to_close")
    if (
        executable_bid_notional is None
        or executable_bid_notional < float(thresholds["min_executable_bid_notional"])
    ):
        exit_reasons.append("entry_blocked_insufficient_executable_bid_notional")
    if (
        queue_fill is None
        or queue_fill < float(thresholds["min_queue_fill_probability_proxy"])
    ):
        exit_reasons.append("entry_blocked_low_queue_fill_probability")
    if spread is None or spread > float(thresholds["max_spread"]):
        exit_reasons.append("entry_blocked_spread_too_wide")
    if staleness is None or staleness > float(thresholds["max_book_staleness_ms"]):
        exit_reasons.append("entry_blocked_stale_book")
    if (
        recent_updates is None
        or recent_updates < float(thresholds["min_recent_book_update_count_1m"])
    ):
        exit_reasons.append("entry_blocked_stale_book")
    if best_margin < float(thresholds["min_best_action_margin"]):
        exit_reasons.append("entry_blocked_exit_reliability_guard")
    min_score = max(
        float(thresholds["min_calibrated_action_score"]),
        float(execution_buffer),
    )
    if best_score < min_score:
        exit_reasons.append("entry_blocked_exit_reliability_guard")
    exit_passed = not exit_reasons
    p_up_alignment_min = float(
        thresholds.get(
            "p_up_alignment_min",
            SELL_BEFORE_CLOSE_P_UP_DIAGNOSTIC_ALIGNMENT_MIN,
        )
    )
    p_up = _p_up(prediction)
    if action.startswith("BUY_UP"):
        p_up_passed = p_up >= p_up_alignment_min
    elif action.startswith("BUY_DOWN"):
        p_up_passed = p_up <= 1.0 - p_up_alignment_min
    else:
        p_up_passed = True
    if p_up_passed:
        p_up_reasons = ("p_up_side_alignment_passed",)
    elif enforce_p_up_alignment:
        p_up_reasons = (
            "entry_blocked_p_up_action_disagreement",
            "entry_blocked_p_up_side_alignment_failed",
        )
    else:
        p_up_reasons = ("p_up_side_alignment_disagreed_diagnostic",)
    exit_reason_codes = (
        ("exit_reliability_guard_thresholds_passed",)
        if exit_passed
        else tuple(dict.fromkeys(("entry_blocked_exit_reliability_guard", *exit_reasons)))
    )
    passed = exit_passed and (p_up_passed or not enforce_p_up_alignment)
    reason_codes = (
        (
            "side_balance_guard_compatible_entry",
            *exit_reason_codes,
            *p_up_reasons,
        )
        if passed
        else tuple(dict.fromkeys((*exit_reason_codes, *p_up_reasons)))
    )
    return {
        "passed": passed,
        "exit_reliability_guard_passed": exit_passed,
        "p_up_side_alignment_passed": p_up_passed,
        "p_up_side_alignment_filter_enabled": bool(enforce_p_up_alignment),
        "p_up_side_alignment_diagnostic_only": not enforce_p_up_alignment,
        "reason_codes": tuple(reason_codes),
    }


def _side_balance_side_feature(
    features: dict[str, Any],
    side: str,
    field: str,
) -> float | None:
    prefix = "up" if side == "UP" else "down"
    value = features.get(f"{prefix}_{field}")
    return None if value is None else float(value)


def _side_quota_selected_keys(
    *,
    rows: list[dict[str, Any]],
    thresholds: dict[str, Any],
) -> set[tuple[str, int]]:
    quota = int(float(thresholds.get("side_quota_per_side", 10.0)))
    selected = set()
    for side in ("UP", "DOWN"):
        side_rows = sorted(
            [row for row in rows if row["selected_side"] == side],
            key=lambda row: (
                -float(row["candidate_rank_score"]),
                -float(row["raw_calibrated_action_score"]),
                -float(row["best_action_margin"]),
                int(row["decision_ts"]),
                str(row["market_id"]),
            ),
        )
        for rank, row in enumerate(side_rows, start=1):
            row["side_quota_rank"] = rank
            if rank <= quota:
                selected.add((str(row["market_id"]), int(row["decision_ts"])))
    return selected


def _annotate_position_state_fresh_entry_candidates(
    *,
    rows: list[dict[str, Any]],
    predictions: tuple[PolymarketPolicyPrediction, ...],
    guard_thresholds: dict[str, float],
) -> None:
    rows_by_key = {
        (str(row["market_id"]), int(row["decision_ts"])): row for row in rows
    }
    positions: dict[str, dict[str, Any]] = {}
    for prediction in sorted(
        predictions,
        key=lambda row: (int(row.decision_ts), str(row.market_id)),
    ):
        key = (str(prediction.market_id), int(prediction.decision_ts))
        row = rows_by_key.get(key)
        position = positions.setdefault(
            str(prediction.market_id),
            _empty_m_position_state(),
        )
        open_side = _open_m_position_side(position)
        if open_side is not None:
            exit_assessment = _m_position_exit_assessment(
                prediction=prediction,
                position=position,
                side=open_side,
            )
            if row is not None:
                row.update(
                    {
                        "position_state_selection_evaluated": True,
                        "position_state_fresh_entry_compatible": False,
                        "position_state_replay_action_if_selected": (
                            f"SELL_{open_side}"
                            if exit_assessment["executable"]
                            else "HOLD"
                        ),
                        "position_state_open_side_before_decision": open_side,
                        "position_state_reason_codes": list(
                            dict.fromkeys(
                                (
                                    "entry_blocked_position_state_not_fresh_entry",
                                    "entry_blocked_existing_position_would_take_precedence",
                                    *exit_assessment["reason_codes"],
                                )
                            )
                        ),
                    }
                )
            if exit_assessment["executable"]:
                positions[str(prediction.market_id)] = _empty_m_position_state()
            continue
        if row is None or not bool(row.get("side_balance_guard_compatible_entry", False)):
            continue
        action = str(prediction.calibrated_best_policy_action)
        if action not in {
            "BUY_UP_SELL_BEFORE_CLOSE",
            "BUY_DOWN_SELL_BEFORE_CLOSE",
        }:
            continue
        side = "UP" if action.startswith("BUY_UP") else "DOWN"
        row.update(
            {
                "position_state_selection_evaluated": True,
                "position_state_fresh_entry_compatible": True,
                "position_state_replay_action_if_selected": _m_entry_action(prediction),
                "position_state_open_side_before_decision": None,
                "position_state_reason_codes": ["position_state_fresh_entry_passed"],
            }
        )
        _open_m_position_state(
            position=position,
            prediction=prediction,
            side=side,
            guard_thresholds=guard_thresholds,
        )
    for row in rows:
        if not bool(row.get("position_state_selection_evaluated", False)):
            row.update(
                {
                    "position_state_selection_evaluated": True,
                    "position_state_fresh_entry_compatible": False,
                    "position_state_replay_action_if_selected": None,
                    "position_state_open_side_before_decision": None,
                    "position_state_reason_codes": [
                        "entry_blocked_position_state_not_guard_compatible"
                    ],
                }
            )


def _empty_m_position_state() -> dict[str, Any]:
    return {
        "side": None,
        "entry_notional": 0.0,
        "entry_qty": 0.0,
    }


def _open_m_position_side(position: dict[str, Any]) -> str | None:
    side = position.get("side")
    if side in {"UP", "DOWN"} and float(position.get("entry_qty", 0.0)) > 0.0:
        return str(side)
    return None


def _open_m_position_state(
    *,
    position: dict[str, Any],
    prediction: PolymarketPolicyPrediction,
    side: str,
    guard_thresholds: dict[str, float],
) -> None:
    ask = _side_balance_side_feature(prediction.features, side, "ask") or 0.0
    score = float(prediction.calibrated_expected_pnl_per_notional or 0.0)
    notional = _m_position_state_paper_notional(
        score=score,
        guard_thresholds=guard_thresholds,
    )
    position["side"] = side
    position["entry_notional"] = notional
    position["entry_qty"] = 0.0 if ask <= 0.0 else notional / ask


def _m_position_state_paper_notional(
    *,
    score: float,
    guard_thresholds: dict[str, float],
) -> float:
    max_notional = float(
        guard_thresholds.get(
            "min_executable_bid_notional",
            M_POSITION_STATE_MAX_PAPER_NOTIONAL,
        )
    )
    return min(max_notional, max(0.01, score * max_notional * 5.0))


def _m_position_exit_assessment(
    *,
    prediction: PolymarketPolicyPrediction,
    position: dict[str, Any],
    side: str,
) -> dict[str, Any]:
    bid = _side_balance_side_feature(prediction.features, side, "bid") or 0.0
    liquidity = _side_balance_side_feature(
        prediction.features,
        side,
        "executable_bid_notional",
    )
    entry_notional = float(position.get("entry_notional", 0.0))
    executable = (
        bid > 0.0
        and liquidity is not None
        and float(liquidity) + 1e-12 >= entry_notional
    )
    if executable:
        reason_codes = [
            "position_state_existing_position_would_exit",
            "position_state_exit_liquidity_available",
        ]
    else:
        reason_codes = [
            "position_state_existing_position_would_hold",
            "position_state_exit_liquidity_unavailable",
        ]
    return {
        "executable": executable,
        "bid": bid,
        "liquidity": liquidity,
        "entry_notional": entry_notional,
        "reason_codes": reason_codes,
    }


def _m_entry_action(prediction: PolymarketPolicyPrediction) -> str:
    action = str(prediction.calibrated_best_policy_action)
    return "BUY_UP" if action.startswith("BUY_UP") else "BUY_DOWN"


def _side_balance_prediction(
    *,
    prediction: PolymarketPolicyPrediction,
    selected_keys: set[tuple[str, int]],
) -> PolymarketPolicyPrediction:
    key = (prediction.market_id, int(prediction.decision_ts))
    action = str(prediction.calibrated_best_policy_action)
    if key in selected_keys and action in {
        "BUY_UP_SELL_BEFORE_CLOSE",
        "BUY_DOWN_SELL_BEFORE_CLOSE",
    }:
        return prediction
    calibrated_returns = dict(prediction.calibrated_expected_pnl_per_notional_by_action)
    calibrated_returns["NO_TRADE"] = max(0.0, float(calibrated_returns.get("NO_TRADE", 0.0)))
    return replace(
        prediction,
        best_policy_action="NO_TRADE",
        best_action_expected_return=float(
            prediction.expected_return_by_action.get("NO_TRADE", 0.0)
        ),
        second_best_action_expected_return=float(
            prediction.expected_return_by_action.get("NO_TRADE", 0.0)
        ),
        best_action_margin=0.0,
        calibrated_best_policy_action="NO_TRADE",
        calibrated_expected_pnl_per_notional=float(calibrated_returns["NO_TRADE"]),
        calibrated_second_best_expected_pnl_per_notional=float(
            calibrated_returns["NO_TRADE"]
        ),
        calibrated_action_margin=0.0,
        calibrated_expected_pnl_per_notional_by_action=calibrated_returns,
    )


def _side_balance_ranked_rows(
    *,
    rows: list[dict[str, Any]],
    selected_keys: set[tuple[str, int]],
    thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    payloads = []
    side_counts = Counter(
        row["selected_side"]
        for row in rows
        if (row["market_id"], int(row["decision_ts"])) in selected_keys
    )
    max_ratio = float(thresholds.get("max_side_entry_ratio", 0.75))
    total_selected = sum(side_counts.values())
    for row in rows:
        key = (row["market_id"], int(row["decision_ts"]))
        selected = key in selected_keys
        reason_codes = ["side_balance_candidate_selected"] if selected else []
        if not selected:
            if bool(row.get("side_balance_guard_compatible_entry", False)):
                if bool(row.get("position_state_selection_evaluated", False)) and not bool(
                    row.get("position_state_fresh_entry_compatible", False)
                ):
                    reason_codes.extend(row.get("position_state_reason_codes", ()))
                else:
                    reason_codes.append("entry_blocked_side_quota_full")
            else:
                reason_codes.append(
                    "entry_blocked_side_balance_guard_compatibility_failed"
                )
                reason_codes.extend(row.get("side_balance_guard_reason_codes", ()))
        if selected and total_selected > 0:
            ratio = side_counts[row["selected_side"]] / total_selected
            if ratio > max_ratio:
                reason_codes.append("entry_blocked_side_ratio_limit")
        if selected:
            reason_codes.append("side_balance_guard_compatible_entry")
        payloads.append(
            {
                **row,
                "side_quota_selected": selected,
                "side_balance_reason_codes": sorted(set(reason_codes)),
            }
        )
    return sorted(
        payloads,
        key=lambda row: (
            str(row["selected_side"]),
            int(row["side_quota_rank"] or 999_999),
            int(row["decision_ts"]),
            str(row["market_id"]),
        ),
    )


def _side_balance_selection_summary(
    *,
    rows: list[dict[str, Any]],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    selected = [row for row in rows if row["side_quota_selected"]]
    guard_compatible = [
        row for row in rows if bool(row.get("side_balance_guard_compatible_entry", False))
    ]
    fresh_entry_compatible = [
        row
        for row in guard_compatible
        if bool(row.get("position_state_fresh_entry_compatible", False))
    ]
    side_distribution = Counter(row["selected_side"] for row in selected)
    guard_side_distribution = Counter(row["selected_side"] for row in guard_compatible)
    fresh_side_distribution = Counter(
        row["selected_side"] for row in fresh_entry_compatible
    )
    market_by_side: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        market_by_side[row["selected_side"]].add(str(row["market_id"]))
    guard_market_by_side: dict[str, set[str]] = defaultdict(set)
    for row in guard_compatible:
        guard_market_by_side[row["selected_side"]].add(str(row["market_id"]))
    fresh_market_by_side: dict[str, set[str]] = defaultdict(set)
    for row in fresh_entry_compatible:
        fresh_market_by_side[row["selected_side"]].add(str(row["market_id"]))
    total = len(selected)
    max_side = max(side_distribution.values(), default=0)
    ratio = 0.0 if total == 0 else max_side / total
    guard_total = len(guard_compatible)
    guard_max_side = max(guard_side_distribution.values(), default=0)
    guard_ratio = 0.0 if guard_total == 0 else guard_max_side / guard_total
    return {
        "candidate_count": len(rows),
        "pre_guard_candidate_count": len(rows),
        "selection_pool": "position_state_aware_guard_compatible_fresh_entry_rows",
        "position_state_aware_selection_enabled": True,
        "execution_pnl_aware_ranking_enabled": True,
        "guard_compatible_candidate_count": guard_total,
        "guard_compatible_up_entry_count": int(guard_side_distribution.get("UP", 0)),
        "guard_compatible_down_entry_count": int(
            guard_side_distribution.get("DOWN", 0)
        ),
        "guard_compatible_up_market_count": len(
            guard_market_by_side.get("UP", set())
        ),
        "guard_compatible_down_market_count": len(
            guard_market_by_side.get("DOWN", set())
        ),
        "guard_compatible_side_count": len(guard_side_distribution),
        "guard_compatible_side_entry_ratio": guard_ratio,
        "guard_compatible_two_sided_entry_set_exists": (
            len(guard_side_distribution) >= 2
        ),
        "position_state_fresh_entry_candidate_count": len(fresh_entry_compatible),
        "position_state_fresh_up_entry_count": int(
            fresh_side_distribution.get("UP", 0)
        ),
        "position_state_fresh_down_entry_count": int(
            fresh_side_distribution.get("DOWN", 0)
        ),
        "position_state_fresh_up_market_count": len(
            fresh_market_by_side.get("UP", set())
        ),
        "position_state_fresh_down_market_count": len(
            fresh_market_by_side.get("DOWN", set())
        ),
        "position_state_blocked_count": sum(
            1
            for row in guard_compatible
            if bool(row.get("position_state_selection_evaluated", False))
            and not bool(row.get("position_state_fresh_entry_compatible", False))
        ),
        "position_state_blocked_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for row in guard_compatible
                    if bool(row.get("position_state_selection_evaluated", False))
                    and not bool(row.get("position_state_fresh_entry_compatible", False))
                    for reason in row.get("position_state_reason_codes", ())
                ).items()
            )
        ),
        "guard_compatible_side_balance_gate_passed": _side_balance_gate_passed(
            up_count=int(guard_side_distribution.get("UP", 0)),
            down_count=int(guard_side_distribution.get("DOWN", 0)),
            up_market_count=len(guard_market_by_side.get("UP", set())),
            down_market_count=len(guard_market_by_side.get("DOWN", set())),
            side_count=len(guard_side_distribution),
            side_entry_ratio=guard_ratio,
            thresholds=thresholds,
        ),
        "guard_compatibility_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for row in rows
                    if not bool(row.get("side_balance_guard_compatible_entry", False))
                    for reason in row.get("side_balance_guard_reason_codes", ())
                ).items()
            )
        ),
        "selected_entry_count": total,
        "selected_execution_pnl_immediate_exit_pnl_sum": _sum_row_float(
            rows=selected,
            field="execution_pnl_immediate_exit_pnl",
        ),
        "selected_execution_pnl_immediate_exit_return_mean": _mean_row_float(
            rows=selected,
            field="execution_pnl_immediate_exit_return",
        ),
        "selected_execution_pnl_model_expected_pnl_sum": _sum_row_float(
            rows=selected,
            field="execution_pnl_model_expected_pnl",
        ),
        "selected_execution_pnl_model_vs_immediate_exit_pnl_gap_estimate_sum": (
            _sum_row_float(
                rows=selected,
                field="execution_pnl_model_vs_immediate_exit_pnl_gap_estimate",
            )
        ),
        "up_entry_count": int(side_distribution.get("UP", 0)),
        "down_entry_count": int(side_distribution.get("DOWN", 0)),
        "up_market_count": len(market_by_side.get("UP", set())),
        "down_market_count": len(market_by_side.get("DOWN", set())),
        "side_count": len(side_distribution),
        "side_entry_ratio": ratio,
        "side_balance_thresholds": dict(thresholds),
        "side_balance_gate_passed": _side_balance_gate_passed(
            up_count=int(side_distribution.get("UP", 0)),
            down_count=int(side_distribution.get("DOWN", 0)),
            up_market_count=len(market_by_side.get("UP", set())),
            down_market_count=len(market_by_side.get("DOWN", set())),
            side_count=len(side_distribution),
            side_entry_ratio=ratio,
            thresholds=thresholds,
        ),
    }


def _sum_row_float(*, rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row.get(field) or 0.0) for row in rows)


def _mean_row_float(*, rows: list[dict[str, Any]], field: str) -> float | None:
    if not rows:
        return None
    return _sum_row_float(rows=rows, field=field) / len(rows)


def _side_balance_gate_passed(
    *,
    up_count: int,
    down_count: int,
    up_market_count: int,
    down_market_count: int,
    side_count: int,
    side_entry_ratio: float,
    thresholds: dict[str, Any],
) -> bool:
    return (
        side_count >= int(float(thresholds["min_side_count"]))
        and up_count >= int(float(thresholds["min_per_side_entry_count"]))
        and down_count >= int(float(thresholds["min_per_side_entry_count"]))
        and up_market_count >= int(float(thresholds["min_per_side_market_count"]))
        and down_market_count >= int(float(thresholds["min_per_side_market_count"]))
        and side_entry_ratio <= float(thresholds["max_side_entry_ratio"])
    )


def action_family_eligibility_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown summary for issue evidence."""

    lines = [
        "# Polymarket Action-Family Eligibility Report",
        "",
        f"- high_score_support_count: {report['high_score_support_count']}",
        "- high_score_realized_return_mean: "
        f"{report['high_score_realized_return_mean']}",
        f"- high_score_realized_return_sum: {report['high_score_realized_return_sum']}",
        f"- execution_buffer: {report['family_high_score_execution_buffer']}",
        f"- min_family_high_score_support: {report['min_family_high_score_support']}",
        "- action_family_paper_decision_eligible: "
        f"{str(report['action_family_paper_decision_eligible']).lower()}",
        "- action_family_paper_decision_ineligible_reasons: "
        f"{json.dumps(report['action_family_paper_decision_ineligible_reasons'])}",
        "- enabled_action_families: "
        f"{json.dumps(report['enabled_action_families'])}",
        "- eligible_action_families: "
        f"{json.dumps(report['eligible_action_families'])}",
        "",
        "## Family Gates",
        "",
    ]
    for family, gate in sorted(report["action_family_gate_results"].items()):
        lines.append(
            "- "
            f"{family}: support={gate['support_count']} "
            f"mean={gate['realized_return_mean']} "
            f"sum={gate['realized_return_sum']} "
            f"passed={str(gate['gate_passed']).lower()}"
        )
    lines.extend(["", "## Action Gates", ""])
    for action, gate in sorted(report["action_gate_results"].items()):
        lines.append(
            "- "
            f"{action}: support={gate['support_count']} "
            f"mean={gate['realized_return_mean']} "
            f"sum={gate['realized_return_sum']} "
            f"passed={str(gate['gate_passed']).lower()}"
        )
    lines.extend(["", "## Fine Family Gates", ""])
    for family, gate in sorted(report["fine_action_family_gate_results"].items()):
        lines.append(
            "- "
            f"{family}: support={gate['support_count']} "
            f"mean={gate['realized_return_mean']} "
            f"sum={gate['realized_return_sum']} "
            f"passed={str(gate['gate_passed']).lower()}"
        )
    lines.extend(
        [
            "",
            "## Negative Examples",
            "",
        ]
    )
    for row in report["negative_high_score_examples"][:10]:
        lines.append(
            "- "
            f"{row['action']} market_id={row['market_id']} "
            f"ts={row['decision_ts']} "
            f"realized={row['realized_return']} "
            f"score={row['calibrated_score']} "
            f"price={row['price_bucket']} "
            f"time={row['time_to_close_bucket']} "
            f"raw={row['raw_score_bucket']}"
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


def hold_to_settlement_longshot_guard_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown summary for the long-shot guard."""

    lines = [
        "# HOLD_TO_SETTLEMENT Long-Shot Guard Report",
        "",
        f"- guard_enabled: {str(report['guard_enabled']).lower()}",
        f"- guard_mode: {report['guard_mode']}",
        f"- guard_reason_codes: {json.dumps(report['guard_reason_codes'])}",
        f"- high_score_support_count: {report['high_score_support_count']}",
        f"- guarded_high_score_count: {report['guarded_high_score_count']}",
        "- guarded_high_score_realized_return_mean: "
        f"{report['guarded_high_score_realized_return_mean']}",
        "- guarded_high_score_realized_return_sum: "
        f"{report['guarded_high_score_realized_return_sum']}",
        f"- price_buckets: {json.dumps(report['price_buckets'])}",
        f"- time_to_close_buckets: {json.dumps(report['time_to_close_buckets'])}",
        f"- raw_score_buckets: {json.dumps(report['raw_score_buckets'])}",
        "",
        "## Guarded By Action",
        "",
    ]
    for row in report["guarded_by_action"]:
        lines.append(
            "- "
            f"{row['action']}: support={row['support_count']} "
            f"mean={row['realized_return_mean']} "
            f"sum={row['realized_return_sum']}"
        )
    lines.extend(["", "## Negative Guarded Examples", ""])
    for row in report["negative_guarded_examples"][:10]:
        lines.append(
            "- "
            f"{row['action']} market_id={row['market_id']} "
            f"ts={row['decision_ts']} "
            f"realized={row['realized_return']} "
            f"score={row['calibrated_score']} "
            f"price={row['price_bucket']} "
            f"time={row['time_to_close_bucket']} "
            f"raw={row['raw_score_bucket']}"
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


def action_family_replay_variants_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown summary for the replay variants."""

    lines = [
        "# Action-Family Replay Variants",
        "",
        f"- execution_buffer: {report['execution_buffer']}",
        f"- min_family_high_score_support: {report['min_family_high_score_support']}",
        "",
        "## Variants",
        "",
    ]
    for variant in report["variants"]:
        lines.append(_variant_markdown_line(variant))
    lines.extend(["", "## Threshold Sweep With Family Gates", ""])
    for variant in report["threshold_sweep_with_action_family_gates"]:
        lines.append(_variant_markdown_line(variant))
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


def _high_score_rows(
    *,
    examples: tuple[PolymarketPolicyExample, ...],
    predictions: tuple[PolymarketPolicyPrediction, ...],
    high_score_threshold: float,
) -> list[dict[str, Any]]:
    _validate_aligned(examples, predictions)
    rows = []
    for example, prediction in zip(examples, predictions, strict=True):
        action = _execution_policy_action(prediction)
        if action == "NO_TRADE":
            continue
        score = _execution_score(prediction, action)
        if score < high_score_threshold:
            continue
        raw_score = float(prediction.expected_return_by_action[action])
        bucket = action_value_bucket_payload(
            action=action,
            features=prediction.features,
            raw_score=raw_score,
        )
        p_up = _p_up(prediction)
        rows.append(
            {
                "market_id": example.market_id,
                "condition_id": example.condition_id,
                "slug": example.slug,
                "market_family": example.market_family,
                "decision_ts": int(example.decision_ts),
                "action": action,
                "calibrated_score": score,
                "raw_score": raw_score,
                "realized_return": float(example.action_return_targets[action]),
                "p_up_auxiliary": p_up,
                "estimated_up_probability": float(prediction.estimated_up_probability),
                "p_up_action_disagreement": _p_up_action_disagrees(
                    action=action,
                    p_up=p_up,
                ),
                **bucket,
            }
        )
    return rows


def _gate_results(
    *,
    rows: list[dict[str, Any]],
    group_field: str,
    execution_buffer: float,
    min_support: int,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_field])].append(row)
    return {
        key: _gate_payload(
            key=key,
            rows=group_rows,
            execution_buffer=execution_buffer,
            min_support=min_support,
        )
        for key, group_rows in sorted(grouped.items())
    }


def _gate_payload(
    *,
    key: str,
    rows: list[dict[str, Any]],
    execution_buffer: float,
    min_support: int,
) -> dict[str, Any]:
    realized_returns = [row["realized_return"] for row in rows]
    support_count = len(rows)
    realized_return_mean = _mean(realized_returns)
    realized_return_sum = _sum(realized_returns)
    support_passed = support_count >= min_support
    mean_exceeds_buffer = support_passed and realized_return_mean > execution_buffer
    sum_positive = realized_return_sum > 0.0
    gate_passed = support_passed and mean_exceeds_buffer and sum_positive
    return {
        "name": key,
        "support_count": support_count,
        "min_support": min_support,
        "support_passed": support_passed,
        "realized_return_mean": realized_return_mean,
        "realized_return_sum": realized_return_sum,
        "realized_return_mean_exceeds_execution_buffer": mean_exceeds_buffer,
        "realized_return_sum_positive": sum_positive,
        "execution_buffer": execution_buffer,
        "gate_passed": gate_passed,
    }


def _eligibility_reason_codes(
    *,
    enabled_families: list[str],
    ineligible_families: list[str],
    action_gate_results: dict[str, dict[str, Any]],
) -> list[str]:
    reasons = set()
    if not enabled_families or ineligible_families:
        reasons.add(ACTION_FAMILY_HIGH_SCORE_UNPROFITABLE)
    if ACTION_FAMILY_HOLD_TO_SETTLEMENT in ineligible_families:
        reasons.add(HOLD_TO_SETTLEMENT_HIGH_SCORE_UNPROFITABLE)
    for action, reason_code in (
        ("BUY_UP_HOLD_TO_SETTLEMENT", BUY_UP_HOLD_TO_SETTLEMENT_UNPROFITABLE),
        ("BUY_DOWN_HOLD_TO_SETTLEMENT", BUY_DOWN_HOLD_TO_SETTLEMENT_UNPROFITABLE),
    ):
        gate = action_gate_results.get(action)
        if gate is not None and int(gate["support_count"]) > 0 and not gate["gate_passed"]:
            reasons.add(reason_code)
    return sorted(reasons)


def _group_summaries(
    rows: list[dict[str, Any]],
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[field]) for field in group_fields)].append(row)
    summaries = []
    for key, group_rows in grouped.items():
        payload = {field: key[index] for index, field in enumerate(group_fields)}
        payload.update(_row_metrics(group_rows))
        summaries.append(payload)
    return sorted(
        summaries,
        key=lambda row: (-int(row["support_count"]), tuple(str(row[field]) for field in group_fields)),
    )


def _row_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    realized_returns = [row["realized_return"] for row in rows]
    calibrated_scores = [row["calibrated_score"] for row in rows]
    action_counts = Counter(str(row["action"]) for row in rows)
    family_counts = Counter(str(row["action_family"]) for row in rows)
    fine_family_counts = Counter(str(row["fine_action_family"]) for row in rows)
    side_counts = Counter(str(row["side"]) for row in rows)
    unique_market_count = len({str(row["market_id"]) for row in rows})
    support_count = len(rows)
    max_side_count = max(side_counts.values(), default=0)
    return {
        "support_count": support_count,
        "realized_return_mean": _mean(realized_returns),
        "realized_return_sum": _sum(realized_returns),
        "calibrated_score_mean": _mean(calibrated_scores),
        "action_distribution": dict(sorted(action_counts.items())),
        "action_family_distribution": dict(sorted(family_counts.items())),
        "fine_action_family_distribution": dict(sorted(fine_family_counts.items())),
        "side_distribution": dict(sorted(side_counts.items())),
        "paper_decision_count_estimate": support_count,
        "unique_market_count": unique_market_count,
        "churn_repeated_decision_estimate": max(0, support_count - unique_market_count),
        "side_concentration": 0.0 if support_count == 0 else max_side_count / support_count,
        "p_up_action_disagreement_count": sum(
            bool(row["p_up_action_disagreement"]) for row in rows
        ),
        "p_up_action_disagreement_rate": (
            0.0
            if support_count == 0
            else sum(bool(row["p_up_action_disagreement"]) for row in rows)
            / support_count
        ),
    }


def _variant_report(
    *,
    variant: str,
    rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    threshold: float,
    gate_mode: str,
    execution_buffer: float,
    min_support: int,
    blocked: bool = False,
    reason_codes: list[str] | None = None,
    family_gate_results: dict[str, dict[str, Any]] | None = None,
    eligible_action_families: list[str] | None = None,
) -> dict[str, Any]:
    metrics = _row_metrics(rows)
    candidate_metrics = _row_metrics(candidate_rows)
    selected_reason_codes = [] if reason_codes is None else sorted(set(reason_codes))
    support_passed = metrics["support_count"] >= min_support
    mean_exceeds_buffer = (
        support_passed and metrics["realized_return_mean"] > execution_buffer
    )
    sum_positive = metrics["realized_return_sum"] > 0.0
    execution_buffer_gate_passed = support_passed and mean_exceeds_buffer and sum_positive
    blocked = blocked or not execution_buffer_gate_passed
    if not selected_reason_codes and blocked:
        selected_reason_codes = [ACTION_FAMILY_HIGH_SCORE_UNPROFITABLE]
    return {
        "variant": variant,
        "threshold": threshold,
        "gate_mode": gate_mode,
        "blocked": blocked or not rows,
        "reason_codes": selected_reason_codes,
        "min_support": min_support,
        "support_passed": support_passed,
        "execution_buffer": execution_buffer,
        "realized_return_mean_exceeds_execution_buffer": mean_exceeds_buffer,
        "realized_return_sum_positive": sum_positive,
        "execution_buffer_gate_passed": execution_buffer_gate_passed,
        "candidate_high_score_support_count": candidate_metrics["support_count"],
        "candidate_high_score_realized_return_mean": candidate_metrics[
            "realized_return_mean"
        ],
        "candidate_high_score_realized_return_sum": candidate_metrics[
            "realized_return_sum"
        ],
        "high_score_support_count": metrics["support_count"],
        "high_score_realized_return_mean": metrics["realized_return_mean"],
        "high_score_realized_return_sum": metrics["realized_return_sum"],
        "high_score_calibrated_score_mean": metrics["calibrated_score_mean"],
        "action_distribution": metrics["action_distribution"],
        "action_family_distribution": metrics["action_family_distribution"],
        "side_distribution": metrics["side_distribution"],
        "paper_decision_count_estimate": metrics["paper_decision_count_estimate"],
        "churn_repeated_decision_estimate": metrics["churn_repeated_decision_estimate"],
        "side_concentration": metrics["side_concentration"],
        "p_up_action_disagreement_count": metrics["p_up_action_disagreement_count"],
        "p_up_action_disagreement_rate": metrics["p_up_action_disagreement_rate"],
        "family_gate_results": family_gate_results or {},
        "eligible_action_families": eligible_action_families or [],
    }


def _passed_bucket_keys(
    *,
    rows: list[dict[str, Any]],
    execution_buffer: float,
    min_support: int,
) -> set[tuple[str, str, str, str]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_bucket_key(row)].append(row)
    passed = set()
    for key, group_rows in grouped.items():
        gate = _gate_payload(
            key="|".join(key),
            rows=group_rows,
            execution_buffer=execution_buffer,
            min_support=min_support,
        )
        if gate["gate_passed"]:
            passed.add(key)
    return passed


def _counterfactual_variant(
    *,
    variant: str,
    predictions: tuple[PolymarketPolicyPrediction, ...],
    ev_threshold: float,
    allowed_mode: str,
    description: str,
    family_gate_results: dict[str, dict[str, Any]] | None = None,
    eligible_action_families: list[str] | None = None,
    passed_bucket_keys: set[tuple[str, str, str, str]] | None = None,
    exit_reliability_guard_enabled: bool = False,
    p_up_side_alignment_filter_enabled: bool = False,
    exit_policy: str | None = None,
    entry_filter_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    eligible_families = tuple(eligible_action_families or ())
    bucket_keys = passed_bucket_keys or set()
    if allowed_mode == "baseline":
        replay_predictions = predictions
    else:
        replay_predictions = tuple(
            _rerank_counterfactual_prediction(
                prediction=prediction,
                allowed_mode=allowed_mode,
                eligible_action_families=eligible_families,
                passed_bucket_keys=bucket_keys,
            )
            for prediction in predictions
        )
    return {
        "variant": variant,
        "description": description,
        "counterfactual_replay_mode": "re_ranked_counterfactual_policy_replay",
        "allowed_mode": allowed_mode,
        "ev_threshold": ev_threshold,
        "eligible_action_families": list(eligible_families),
        "family_gate_results": family_gate_results or {},
        "prediction_count": len(replay_predictions),
        "predictions": replay_predictions,
        "exit_reliability_guard_enabled": exit_reliability_guard_enabled,
        "p_up_side_alignment_filter_enabled": p_up_side_alignment_filter_enabled,
        "exit_policy": exit_policy,
        "entry_filter_thresholds": dict(entry_filter_thresholds or {}),
        **compact_safety_fields(),
    }


def _rerank_counterfactual_prediction(
    *,
    prediction: PolymarketPolicyPrediction,
    allowed_mode: str,
    eligible_action_families: tuple[str, ...],
    passed_bucket_keys: set[tuple[str, str, str, str]],
) -> PolymarketPolicyPrediction:
    allowed_actions = [
        action
        for action in ACTION_VALUE_LABEL_ACTIONS
        if _counterfactual_action_allowed(
            action=action,
            prediction=prediction,
            allowed_mode=allowed_mode,
            eligible_action_families=eligible_action_families,
            passed_bucket_keys=passed_bucket_keys,
        )
    ]
    if "NO_TRADE" not in allowed_actions:
        allowed_actions.append("NO_TRADE")
    calibrated_returns = prediction.calibrated_expected_pnl_per_notional_by_action
    if not calibrated_returns:
        calibrated_returns = prediction.expected_return_by_action
    calibrated_best, calibrated_best_return, calibrated_second, calibrated_margin = (
        _rank_allowed_actions(
            returns={action: float(calibrated_returns[action]) for action in allowed_actions},
        )
    )
    raw_best, raw_best_return, raw_second, raw_margin = _rank_allowed_actions(
        returns={
            action: float(prediction.expected_return_by_action[action])
            for action in allowed_actions
        },
    )
    return replace(
        prediction,
        best_policy_action=raw_best,
        best_action_expected_return=raw_best_return,
        second_best_action_expected_return=raw_second,
        best_action_margin=raw_margin,
        calibrated_best_policy_action=calibrated_best,
        calibrated_expected_pnl_per_notional=calibrated_best_return,
        calibrated_second_best_expected_pnl_per_notional=calibrated_second,
        calibrated_action_margin=calibrated_margin,
    )


def _counterfactual_action_allowed(
    *,
    action: str,
    prediction: PolymarketPolicyPrediction,
    allowed_mode: str,
    eligible_action_families: tuple[str, ...],
    passed_bucket_keys: set[tuple[str, str, str, str]],
) -> bool:
    family = action_value_action_family(action)
    if family == ACTION_FAMILY_NO_TRADE:
        return True
    if allowed_mode == "hold_to_settlement_disabled":
        return family != ACTION_FAMILY_HOLD_TO_SETTLEMENT
    if allowed_mode == "sell_before_close_only":
        return family == ACTION_FAMILY_SELL_BEFORE_CLOSE
    if allowed_mode == "sell_before_close_exit_reliability_guard":
        return family == ACTION_FAMILY_SELL_BEFORE_CLOSE
    if allowed_mode == "sell_before_close_exit_reliability_p_up_aligned":
        return family == ACTION_FAMILY_SELL_BEFORE_CLOSE
    if allowed_mode == "sell_before_close_support_aware_p_up_aligned":
        return family == ACTION_FAMILY_SELL_BEFORE_CLOSE
    if allowed_mode == "support_aware_selection_failed_no_trade":
        return False
    if allowed_mode == "passed_family_and_bucket_only":
        if family not in eligible_action_families:
            return False
        if family != ACTION_FAMILY_HOLD_TO_SETTLEMENT:
            return True
        return _prediction_bucket_key(action=action, prediction=prediction) in passed_bucket_keys
    if allowed_mode == "action_family_gates_enabled":
        return family in eligible_action_families
    raise ValueError(f"unsupported counterfactual allowed_mode: {allowed_mode}")


def _prediction_bucket_key(
    *,
    action: str,
    prediction: PolymarketPolicyPrediction,
) -> tuple[str, str, str, str]:
    raw_score = float(prediction.expected_return_by_action[action])
    bucket = action_value_bucket_payload(
        action=action,
        features=prediction.features,
        raw_score=raw_score,
    )
    return (
        action,
        str(bucket["price_bucket"]),
        str(bucket["time_to_close_bucket"]),
        str(bucket["raw_score_bucket"]),
    )


def _rank_allowed_actions(returns: dict[str, float]) -> tuple[str, float, float, float]:
    ranked = sorted(returns.items(), key=lambda item: (-item[1], item[0]))
    best_action, best_return = ranked[0]
    second_return = ranked[1][1] if len(ranked) > 1 else best_return
    return best_action, best_return, second_return, best_return - second_return


def _bucket_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["action"]),
        str(row["price_bucket"]),
        str(row["time_to_close_bucket"]),
        str(row["raw_score_bucket"]),
    )


def _negative_examples(rows: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    keys = (
        "market_id",
        "decision_ts",
        "action",
        "action_family",
        "side",
        "realized_return",
        "calibrated_score",
        "raw_score",
        "price_bucket",
        "time_to_close_bucket",
        "raw_score_bucket",
        "p_up_auxiliary",
        "p_up_action_disagreement",
        "hold_to_settlement_longshot_guard_applies",
    )
    return [
        {key: row[key] for key in keys}
        for row in sorted(rows, key=lambda item: (item["realized_return"], item["decision_ts"]))[
            :limit
        ]
    ]


def _execution_policy_action(prediction: PolymarketPolicyPrediction) -> str:
    calibrated_action = prediction.calibrated_best_policy_action
    if calibrated_action is not None:
        return str(calibrated_action)
    return str(prediction.best_policy_action)


def _execution_score(prediction: PolymarketPolicyPrediction, action: str) -> float:
    if prediction.calibrated_expected_pnl_per_notional is not None:
        return float(prediction.calibrated_expected_pnl_per_notional)
    return float(prediction.expected_return_by_action[action])


def _p_up(prediction: PolymarketPolicyPrediction) -> float:
    if prediction.p_up_auxiliary is not None:
        return float(prediction.p_up_auxiliary)
    return float(prediction.estimated_up_probability)


def _p_up_action_disagrees(*, action: str, p_up: float) -> bool:
    if action.startswith("BUY_DOWN_"):
        return p_up >= P_UP_MATERIAL_DISAGREEMENT_THRESHOLD
    if action.startswith("BUY_UP_"):
        return p_up <= 1.0 - P_UP_MATERIAL_DISAGREEMENT_THRESHOLD
    return False


def _validate_aligned(
    examples: tuple[PolymarketPolicyExample, ...],
    predictions: tuple[PolymarketPolicyPrediction, ...],
) -> None:
    if len(examples) != len(predictions):
        raise ValueError("action-family examples/predictions length mismatch")
    for example, prediction in zip(examples, predictions, strict=True):
        if (example.market_id, example.decision_ts) != (
            prediction.market_id,
            prediction.decision_ts,
        ):
            raise ValueError("action-family examples/predictions misaligned")
        missing_targets = set(ACTION_VALUE_LABEL_ACTIONS) - set(
            example.action_return_targets
        )
        if missing_targets:
            raise ValueError(
                "action-family example missing targets: "
                + ", ".join(sorted(missing_targets))
            )
        missing_predictions = set(ACTION_VALUE_LABEL_ACTIONS) - set(
            prediction.expected_return_by_action
        )
        if missing_predictions:
            raise ValueError(
                "action-family prediction missing actions: "
                + ", ".join(sorted(missing_predictions))
            )


def _variant_markdown_line(variant: dict[str, Any]) -> str:
    return (
        "- "
        f"{variant['variant']}: "
        f"support={variant['high_score_support_count']} "
        f"mean={variant['high_score_realized_return_mean']} "
        f"sum={variant['high_score_realized_return_sum']} "
        f"paper_decisions={variant['paper_decision_count_estimate']} "
        f"churn={variant['churn_repeated_decision_estimate']} "
        f"side_concentration={variant['side_concentration']} "
        f"p_up_disagreement={variant['p_up_action_disagreement_rate']} "
        f"actions={json.dumps(variant['action_distribution'], sort_keys=True)} "
        f"blocked={str(variant['blocked']).lower()} "
        f"reasons={json.dumps(variant['reason_codes'])}"
    )


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return _sum(values) / len(values)


def _sum(values: list[float]) -> float:
    return float(sum(values))
