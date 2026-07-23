"""Frozen future-unseen holdout contract for issue #246 v8.1."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_abstention_aware_expected_net_pnl_v7_0 import (
    _v7_0_blocked_safety_fields,
)
from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7 import (
    SBC_ACTIONS,
    SIDES,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    _find_nonempty_fields,
    _require_sha256,
)
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
    _require_git_sha,
)

SCHEMA_PREFIX = "bigan-v8-adaptive-support-controller-v8-1-future-holdout"
PLAN_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-plan-v1"
CANDIDATE_NAME = "adaptive_support_controller_v8_1"
BASELINE_NAME = "p_up_semantic_compatibility_v6_7"
ISSUE_NUMBER = 246
EXACT_MARKET_COUNT = 120
SCAN_CAP = 180
BOUNDED_BATCH_MARKET_COUNT = 12
MINIMUM_GUARD_ACCEPTED_MARKET_COUNT = 40
STRICTLY_LATER_MINIMUM_MARKET_START_TS_EXCLUSIVE = 1_784_819_024_000
COMPLETE_CANARY_BATCH_LATEST_MARKET_CLOSE_TS = 1_784_818_500_000
FROZEN_PLAN_SHA256 = (
    "c2121f20bfc6aefcd62a72781d81053913137ed99706abc380075b07d0d0f100"
)
FIVE_ACTIONS = frozenset(
    {
        "BUY_UP_SELL_BEFORE_CLOSE",
        "BUY_DOWN_SELL_BEFORE_CLOSE",
        "BUY_UP_HOLD_TO_SETTLEMENT",
        "BUY_DOWN_HOLD_TO_SETTLEMENT",
        "NO_TRADE",
    }
)
FORBIDDEN_TARGET_FIELDS = frozenset(
    {
        "outcome",
        "resolved_outcome",
        "resolution",
        "winner",
        "settlement_pnl",
        "settlement_price",
        "runtime_policy_after_cost_net_pnl_per_contract",
        "runtime_policy_after_cost_net_pnl_at_frozen_size",
        "realized_pnl",
        "realized_return",
        "future_return",
        "label",
        "oracle_action",
    }
)

LINEAGE_HASH_FIELDS = frozenset(
    {
        "historical_manifest_sha256",
        "historical_model_sha256",
        "candidate_profile_sha256",
        "candidate_decision_policy_source_sha256",
        "target_free_canary_plan_sha256",
        "target_free_canary_manifest_sha256",
        "target_free_canary_report_sha256",
        "target_free_canary_batch_index_sha256",
        "target_free_canary_batch_last_entry_sha256",
        "target_free_canary_development_manifest_sha256",
        "target_free_canary_v6_2_manifest_sha256",
        "v6_2_source_candidate_manifest_sha256",
        "v6_7_profile_sha256",
        "v7_0_training_profile_sha256",
        "runtime_policy_profile_sha256",
        "collector_protocol_sha256",
        "feature_contract_sha256",
    }
)


def validate_adaptive_support_controller_v8_1_future_holdout_plan(
    plan: dict[str, Any],
) -> None:
    """Reject lineage, collection, decision, gate, or safety drift."""

    lineage = dict(plan.get("lineage") or {})
    collection = dict(plan.get("collection") or {})
    freeze = dict(plan.get("target_free_decision_freeze") or {})
    gate = dict(plan.get("single_use_future_pnl_gate") or {})
    scope = dict(plan.get("result_scope") or {})
    if set(lineage) != LINEAGE_HASH_FIELDS | {"implementation_commit"}:
        raise ValueError("#246 future holdout lineage field set drifted")
    _require_git_sha(str(lineage["implementation_commit"]))
    for name in LINEAGE_HASH_FIELDS:
        _require_sha256(str(lineage[name]), name=name)

    checks = {
        "schema": plan.get("schema_version") == PLAN_SCHEMA_VERSION,
        "issue": plan.get("issue_number") == ISSUE_NUMBER,
        "candidate": plan.get("candidate_name") == CANDIDATE_NAME,
        "baseline": plan.get("baseline_name") == BASELINE_NAME,
        "created_before_boundary": int(plan.get("plan_created_ts") or 0)
        == STRICTLY_LATER_MINIMUM_MARKET_START_TS_EXCLUSIVE,
        "frozen": plan.get("frozen") is True,
        "preregistered": plan.get("preregistered_before_collection") is True,
        "collection": collection
        == {
            "mode": "bounded_candidate_agnostic_outcome_blind_raw_collection",
            "selection_method": (
                "earliest_quality_valid_strictly_later_disjoint_markets"
            ),
            "exact_quality_valid_market_count": EXACT_MARKET_COUNT,
            "maximum_attempted_market_count": SCAN_CAP,
            "bounded_batch_market_count": BOUNDED_BATCH_MARKET_COUNT,
            "strictly_later_minimum_market_start_ts_exclusive": (
                STRICTLY_LATER_MINIMUM_MARKET_START_TS_EXCLUSIVE
            ),
            "complete_target_free_canary_batch_latest_market_close_ts": (
                COMPLETE_CANARY_BATCH_LATEST_MARKET_CLOSE_TS
            ),
            "boundary_source": (
                "max_plan_created_ts_and_complete_target_free_canary_batch_"
                "latest_market_close_ts"
            ),
            "market_id_disjointness_required": True,
            "slug_disjointness_required": True,
            "decision_id_disjointness_required": True,
            "source_row_hash_disjointness_required": True,
            "raw_artifact_hash_verification_required": True,
            "feature_timestamp_causality_required": True,
            "outcomes_resolution_labels_or_pnl_opened": False,
            "candidate_model_scoring_during_collection_allowed": False,
            "partial_authoritative_window_freeze_allowed": False,
        },
        "target_free_freeze": freeze
        == {
            "exact_market_count": EXACT_MARKET_COUNT,
            "minimum_candidate_guard_accepted_unique_market_count": (
                MINIMUM_GUARD_ACCEPTED_MARKET_COUNT
            ),
            "one_frozen_decision_per_policy_per_market": True,
            "controller_initial_state_source": (
                "frozen_historical_model_final_rank_state"
            ),
            "controller_state_advances_chronologically_and_only_after_each_"
            "decision_full_guard_freeze": True,
            "full_v6_7_execution_guard_unchanged": True,
            "fixed_position_size_unchanged": True,
            "source_score_mutation_allowed": False,
            "model_score_controller_or_threshold_tuning_allowed": False,
            "side_quota_enabled": False,
            "side_action_and_family_metrics_diagnostic_only": True,
            "all_selected_markets_closed_before_target_access": True,
            "outcomes_resolution_labels_or_pnl_opened": False,
        },
        "single_use_gate": gate
        == {
            "official_read_only_settlement_on_quarantine_copies": True,
            "complete_settlement_required": True,
            "candidate_total_after_cost_pnl_minimum_exclusive": 0.0,
            "candidate_minus_v6_7_total_after_cost_pnl_minimum_inclusive": 0.0,
            "candidate_minus_v6_7_largest_winner_removed_after_cost_pnl_"
            "minimum_inclusive": 0.0,
            "comparison_operator": "greater_than_or_equal",
            "equality_passes_noninferiority": True,
            "legacy_or_no_bet_market_pnl": 0.0,
            "side_action_and_family_metrics_diagnostic_only": True,
            "outcomes_used_for_selection_or_tuning": False,
            "result_selected_rerun_allowed": False,
            "result_selected_extension_allowed": False,
        },
        "scope": scope
        == {
            "future_unseen_pnl_evidence_for_promotion_discussion_only": True,
            "automatic_paper_or_live_unlock_allowed": False,
            "separate_promotion_review_required": True,
        },
        "safety": plan.get("safety") == _v7_0_blocked_safety_fields(),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"#246 future holdout plan drifted: {failed}")


def select_adaptive_support_controller_v8_1_future_holdout_window(
    index_rows: list[dict[str, Any]],
    *,
    plan: dict[str, Any],
    prior_market_ids: set[str],
    prior_slugs: set[str],
    prior_decision_ids: set[str],
    prior_source_row_hashes: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Select the earliest exact-120 eligible rows within the frozen scan cap."""

    validate_adaptive_support_controller_v8_1_future_holdout_plan(plan)
    collection = dict(plan["collection"])
    boundary = int(collection["strictly_later_minimum_market_start_ts_exclusive"])
    ordered = sorted(
        index_rows,
        key=lambda row: (
            int(row.get("scheduled_round_start_ts") or 0),
            int(row.get("sequence") or 0),
            str(row.get("attempt_id") or row.get("run_id") or ""),
        ),
    )
    attempted = ordered[: int(collection["maximum_attempted_market_count"])]
    eligible: list[dict[str, Any]] = []
    exclusion_reasons: Counter[str] = Counter()
    for row in attempted:
        reasons = []
        if row.get("capture_quality_valid") is not True:
            reasons.append("capture_quality_invalid")
        if int(row.get("scheduled_round_start_ts") or 0) <= boundary:
            reasons.append("scheduled_round_not_strictly_later")
        if int(row.get("market_start_ts") or 0) <= boundary:
            reasons.append("market_start_not_strictly_later")
        if str(row.get("market_id") or "") in prior_market_ids:
            reasons.append("prior_market_id_overlap")
        if str(row.get("slug") or "") in prior_slugs:
            reasons.append("prior_slug_overlap")
        if str(row.get("decision_id") or "") in prior_decision_ids:
            reasons.append("prior_decision_id_overlap")
        if str(row.get("source_row_hash") or "") in prior_source_row_hashes:
            reasons.append("prior_source_row_hash_overlap")
        if reasons:
            exclusion_reasons.update(reasons)
            continue
        eligible.append(row)
        if len(eligible) == EXACT_MARKET_COUNT:
            break
    selected = eligible[:EXACT_MARKET_COUNT]
    selected_ids = [str(row.get("market_id") or "") for row in selected]
    summary = {
        "attempted_scan_count": len(attempted),
        "eligible_market_count": len(eligible),
        "selected_market_count": len(selected),
        "selected_sequence_start": int(selected[0]["sequence"]) if selected else None,
        "selected_sequence_end": int(selected[-1]["sequence"]) if selected else None,
        "selected_market_ids_sha256": canonical_json_sha256(selected_ids),
        "exclusion_reason_distribution": dict(sorted(exclusion_reasons.items())),
        "strictly_later_time_violation_count": sum(
            int(row.get("market_start_ts") or 0) <= boundary for row in selected
        ),
        "selected_identity_duplicate_count": len(selected_ids) - len(set(selected_ids)),
        "exact_window_ready": (
            len(selected) == EXACT_MARKET_COUNT
            and "" not in selected_ids
            and len(set(selected_ids)) == EXACT_MARKET_COUNT
        ),
    }
    return selected, attempted, summary


def build_adaptive_support_controller_v8_1_target_free_freeze_report(
    selected_rows: list[dict[str, Any]],
    *,
    attempted_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    candidate_guard_rows: list[dict[str, Any]],
    baseline_guard_rows: list[dict[str, Any]],
    selection_summary: dict[str, Any],
    plan: dict[str, Any],
    stage_started_ts: int,
    collector_index_sha256: str,
) -> dict[str, Any]:
    """Freeze candidate and baseline decisions before target access."""

    validate_adaptive_support_controller_v8_1_future_holdout_plan(plan)
    _require_sha256(collector_index_sha256, name="collector_index_sha256")
    selected_ids = [str(row.get("market_id") or "") for row in selected_rows]
    selected_set = set(selected_ids)
    action_rows = [
        row
        for row in action_rows
        if str(row.get("market_id") or "") in selected_set
    ]
    candidate_guard_rows = [
        row
        for row in candidate_guard_rows
        if str(row.get("market_id") or "") in selected_set
    ]
    baseline_guard_rows = [
        row
        for row in baseline_guard_rows
        if str(row.get("market_id") or "") in selected_set
    ]
    candidate_by_market = _one_guard_row_per_market(candidate_guard_rows)
    baseline_by_market = _one_guard_row_per_market(baseline_guard_rows)
    forbidden = sorted(
        set(_find_nonempty_fields(action_rows, FORBIDDEN_TARGET_FIELDS))
        | set(_find_nonempty_fields(candidate_guard_rows, FORBIDDEN_TARGET_FIELDS))
        | set(_find_nonempty_fields(baseline_guard_rows, FORBIDDEN_TARGET_FIELDS))
    )
    causality_violations = sum(
        int(row.get("max_input_ts") or 0) > int(row.get("decision_ts") or 0)
        for row in action_rows
    )
    candidate_accepted = [
        row
        for row in candidate_guard_rows
        if row.get("execution_guard_order_allowed") is True
    ]
    baseline_accepted = [
        row
        for row in baseline_guard_rows
        if row.get("execution_guard_order_allowed") is True
    ]
    checks = {
        "exact_120_selected_markets": (
            selection_summary.get("exact_window_ready") is True
            and len(selected_rows) == EXACT_MARKET_COUNT
            and len(selected_set) == EXACT_MARKET_COUNT
        ),
        "attempted_scan_cap_respected": len(attempted_rows) <= SCAN_CAP,
        "all_selected_markets_closed_before_target_access": (
            bool(selected_rows)
            and stage_started_ts
            > max(int(row.get("market_end_ts") or 0) for row in selected_rows)
        ),
        "complete_five_action_grid": _complete_five_action_grid(
            action_rows, selected_market_ids=selected_set
        ),
        "candidate_complete_decision_coverage": set(candidate_by_market)
        == selected_set,
        "baseline_complete_decision_coverage": set(baseline_by_market)
        == selected_set,
        "minimum_candidate_guard_accepted_market_support": len(candidate_accepted)
        >= MINIMUM_GUARD_ACCEPTED_MARKET_COUNT,
        "feature_timestamp_causality": causality_violations == 0,
        "forbidden_target_fields_absent": not forbidden,
        "candidate_source_scores_unchanged": all(
            row.get("source_score_mutated") is False
            for row in candidate_guard_rows
        ),
        "baseline_source_scores_unchanged": all(
            row.get("source_score_mutated") is False
            for row in baseline_guard_rows
        ),
        "outcomes_resolution_labels_or_pnl_sealed": all(
            row.get("labels_outcomes_or_pnl_opened") is False
            for row in candidate_guard_rows + baseline_guard_rows
        ),
        "controller_state_advanced_post_guard_only": all(
            row.get("current_guard_result_used_for_own_controller_decision") is False
            and row.get("current_guard_result_added_after_decision_freeze") is True
            for row in candidate_guard_rows
        ),
    }
    reason_map = {
        "exact_120_selected_markets": "target_free_exact_120_window_not_ready",
        "attempted_scan_cap_respected": "target_free_scan_cap_exceeded",
        "all_selected_markets_closed_before_target_access": (
            "target_free_markets_not_all_closed"
        ),
        "complete_five_action_grid": "target_free_five_action_grid_incomplete",
        "candidate_complete_decision_coverage": (
            "target_free_v8_1_decision_coverage_incomplete"
        ),
        "baseline_complete_decision_coverage": (
            "target_free_v6_7_decision_coverage_incomplete"
        ),
        "minimum_candidate_guard_accepted_market_support": (
            "target_free_v8_1_guard_accepted_support_insufficient"
        ),
        "feature_timestamp_causality": "target_free_feature_causality_violation",
        "forbidden_target_fields_absent": (
            "target_free_forbidden_target_field_present"
        ),
        "candidate_source_scores_unchanged": (
            "target_free_v8_1_source_score_mutated"
        ),
        "baseline_source_scores_unchanged": (
            "target_free_v6_7_source_score_mutated"
        ),
        "outcomes_resolution_labels_or_pnl_sealed": (
            "target_free_outcome_or_pnl_access_detected"
        ),
        "controller_state_advanced_post_guard_only": (
            "target_free_controller_state_advance_order_invalid"
        ),
    }
    blockers = [reason_map[name] for name, passed in checks.items() if not passed]
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-target-free-freeze-report-v1",
        "candidate_name": CANDIDATE_NAME,
        "baseline_name": BASELINE_NAME,
        "collector_index_sha256": collector_index_sha256,
        "selected_market_count": len(selected_rows),
        "attempted_market_count": len(attempted_rows),
        "selection_summary": selection_summary,
        "candidate_guard_accepted_market_count": len(candidate_accepted),
        "v6_7_guard_accepted_market_count": len(baseline_accepted),
        "candidate_guard_accepted_side_distribution_diagnostic": dict(
            sorted(Counter(_guard_side(row) for row in candidate_accepted).items())
        ),
        "v6_7_guard_accepted_side_distribution_diagnostic": dict(
            sorted(Counter(_guard_side(row) for row in baseline_accepted).items())
        ),
        "side_quota_enabled": False,
        "side_action_and_family_metrics_diagnostic_only": True,
        "feature_causality_violation_count": causality_violations,
        "forbidden_target_fields": forbidden,
        "target_free_checks": checks,
        "target_free_freeze_passed": not blockers,
        "target_free_blocking_reason_codes": blockers,
        "future_target_access_allowed": not blockers,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "settlement_provider_called": False,
        "threshold_model_or_controller_tuning_performed": False,
        "source_scores_mutated": False,
        "promotion_evidence": False,
        **_v7_0_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def materialize_adaptive_support_controller_v8_1_runtime_decisions(
    guard_rows: list[dict[str, Any]],
    *,
    action_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Bind guard-accepted actions to their immutable decision-time source."""

    sources: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in action_rows:
        action = str(row.get("action") or "")
        if action in SBC_ACTIONS:
            sources[
                (
                    str(row.get("market_id") or ""),
                    int(row.get("decision_ts") or 0),
                    action,
                )
            ].append(row)
    output = []
    seen: set[str] = set()
    for guard in guard_rows:
        if guard.get("execution_guard_order_allowed") is not True:
            continue
        market_id = str(guard.get("market_id") or "")
        decision_ts = int(guard.get("decision_ts") or 0)
        action = str(guard.get("selected_action") or guard.get("action") or "")
        side = str(guard.get("selected_side") or guard.get("side") or "")
        matches = sources.get((market_id, decision_ts, action), [])
        if (
            not market_id
            or market_id in seen
            or decision_ts <= 0
            or action not in SBC_ACTIONS
            or side not in SIDES
            or len(matches) != 1
        ):
            raise ValueError("#246 guard-accepted runtime source identity invalid")
        source = matches[0]
        max_input_ts = int(source.get("max_input_ts") or 0)
        if max_input_ts > decision_ts:
            raise ValueError("#246 guard-accepted runtime source causality invalid")
        seen.add(market_id)
        output.append(
            {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "max_input_ts": max_input_ts,
                "market_close_ts": int(source.get("market_close_ts") or 0),
                "side": side,
                "action": action,
                "microstructure_snapshot": dict(
                    source.get("microstructure_snapshot") or {}
                ),
                "source_decision_id": str(source.get("decision_id") or ""),
                "source_score_mutated": False,
                "labels_outcomes_resolution_or_pnl_opened": False,
                "target_used_as_decision_time_input": False,
                **_v7_0_blocked_safety_fields(),
            }
        )
    return sorted(output, key=lambda row: (int(row["decision_ts"]), row["market_id"]))


def build_adaptive_support_controller_v8_1_future_pnl_gate(
    candidate_rows: list[dict[str, Any]],
    *,
    baseline_rows: list[dict[str, Any]],
    evaluation_market_ids: list[str],
    settled_market_ids: list[str],
    plan: dict[str, Any],
    target_free_freeze_sha256: str,
) -> dict[str, Any]:
    """Compare frozen v8.1 and v6.7 decisions on the exact same markets."""

    validate_adaptive_support_controller_v8_1_future_holdout_plan(plan)
    _require_sha256(target_free_freeze_sha256, name="target_free_freeze_sha256")
    market_ids = list(dict.fromkeys(str(value) for value in evaluation_market_ids))
    settled_ids = list(dict.fromkeys(str(value) for value in settled_market_ids))
    if (
        len(market_ids) != EXACT_MARKET_COUNT
        or "" in market_ids
        or len(settled_ids) != EXACT_MARKET_COUNT
        or set(settled_ids) != set(market_ids)
    ):
        raise ValueError("#246 exact settled evaluation market identity invalid")
    _validate_runtime_target_rows(candidate_rows, market_ids=market_ids)
    _validate_runtime_target_rows(baseline_rows, market_ids=market_ids)

    candidate_by_market = dict.fromkeys(market_ids, 0.0)
    baseline_by_market = dict.fromkeys(market_ids, 0.0)
    for row in candidate_rows:
        candidate_by_market[str(row["market_id"])] += float(
            row["runtime_policy_after_cost_net_pnl_at_frozen_size"]
        )
    for row in baseline_rows:
        baseline_by_market[str(row["market_id"])] += float(
            row["runtime_policy_after_cost_net_pnl_at_frozen_size"]
        )
    candidate_total = float(sum(candidate_by_market.values()))
    baseline_total = float(sum(baseline_by_market.values()))
    candidate_largest_winner = max(candidate_by_market.values(), default=0.0)
    baseline_largest_winner = max(baseline_by_market.values(), default=0.0)
    candidate_lwr = candidate_total - max(candidate_largest_winner, 0.0)
    baseline_lwr = baseline_total - max(baseline_largest_winner, 0.0)
    total_delta = candidate_total - baseline_total
    lwr_delta = candidate_lwr - baseline_lwr
    gate = dict(plan["single_use_future_pnl_gate"])
    target_isolation = all(
        row.get("target_available_only_post_exit_or_official_resolution") is True
        and row.get("target_used_as_decision_time_input") is False
        and int(row["max_input_ts"]) <= int(row["decision_ts"])
        for row in candidate_rows + baseline_rows
    )
    checks = {
        "exact_120_settled_market_reconciliation": set(settled_ids)
        == set(market_ids),
        "minimum_candidate_guard_accepted_unique_market_support": len(
            candidate_rows
        )
        >= MINIMUM_GUARD_ACCEPTED_MARKET_COUNT,
        "candidate_total_after_cost_pnl_positive": candidate_total
        > float(gate["candidate_total_after_cost_pnl_minimum_exclusive"]),
        "candidate_noninferior_to_v6_7_total_pnl": total_delta
        >= float(
            gate["candidate_minus_v6_7_total_after_cost_pnl_minimum_inclusive"]
        ),
        "candidate_noninferior_to_v6_7_largest_winner_removed": lwr_delta
        >= float(
            gate[
                "candidate_minus_v6_7_largest_winner_removed_after_cost_pnl_"
                "minimum_inclusive"
            ]
        ),
        "settlement_causality_and_target_isolation": target_isolation,
    }
    reason_map = {
        "exact_120_settled_market_reconciliation": (
            "exact_120_settlement_reconciliation_failed"
        ),
        "minimum_candidate_guard_accepted_unique_market_support": (
            "insufficient_v8_1_guard_accepted_unique_market_support"
        ),
        "candidate_total_after_cost_pnl_positive": (
            "candidate_total_after_cost_pnl_not_positive"
        ),
        "candidate_noninferior_to_v6_7_total_pnl": (
            "candidate_total_pnl_inferior_to_v6_7"
        ),
        "candidate_noninferior_to_v6_7_largest_winner_removed": (
            "candidate_largest_winner_removed_pnl_inferior_to_v6_7"
        ),
        "settlement_causality_and_target_isolation": (
            "settlement_causality_or_target_isolation_failed"
        ),
    }
    blockers = [reason_map[name] for name, passed in checks.items() if not passed]
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-pnl-noninferiority-gate-report-v1",
        "candidate_name": CANDIDATE_NAME,
        "baseline_name": BASELINE_NAME,
        "target_free_freeze_sha256": target_free_freeze_sha256,
        "evaluation_market_count": len(market_ids),
        "settled_market_count": len(settled_ids),
        "candidate_guard_accepted_unique_market_count": len(candidate_rows),
        "v6_7_guard_accepted_unique_market_count": len(baseline_rows),
        "candidate_after_cost_pnl": candidate_total,
        "v6_7_after_cost_pnl": baseline_total,
        "candidate_minus_v6_7_after_cost_pnl": total_delta,
        "candidate_largest_winner_after_cost_pnl": candidate_largest_winner,
        "candidate_largest_winner_removed_after_cost_pnl": candidate_lwr,
        "v6_7_largest_winner_after_cost_pnl": baseline_largest_winner,
        "v6_7_largest_winner_removed_after_cost_pnl": baseline_lwr,
        "candidate_minus_v6_7_largest_winner_removed_after_cost_pnl": lwr_delta,
        "candidate_side_distribution_diagnostic": dict(
            sorted(Counter(str(row["side"]) for row in candidate_rows).items())
        ),
        "v6_7_side_distribution_diagnostic": dict(
            sorted(Counter(str(row["side"]) for row in baseline_rows).items())
        ),
        "noninferiority_comparison_operator": "greater_than_or_equal",
        "equality_passes_noninferiority": True,
        "future_noninferiority_gate_passed": (
            checks["candidate_noninferior_to_v6_7_total_pnl"]
            and checks["candidate_noninferior_to_v6_7_largest_winner_removed"]
        ),
        "model_improvement_demonstrated": total_delta > 0.0 and lwr_delta > 0.0,
        "side_quota_enabled": False,
        "side_action_and_family_metrics_diagnostic_only": True,
        "future_pnl_gate_checks": checks,
        "future_pnl_gate_passed": not blockers,
        "future_pnl_gate_blocking_reason_codes": blockers,
        "promotion_discussion_evidence_available": not blockers,
        "automatic_paper_or_live_unlock_allowed": False,
        "future_outcomes_used_for_model_threshold_cost_sizing_or_guard_tuning": False,
        "single_use_future_gate": True,
        "result_selected_rerun_allowed": False,
        **_v7_0_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def _one_guard_row_per_market(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        market_id = str(row.get("market_id") or "")
        if not market_id or market_id in output:
            raise ValueError("#246 guard replay market identity is missing or duplicated")
        output[market_id] = row
    return output


def _complete_five_action_grid(
    rows: list[dict[str, Any]], *, selected_market_ids: set[str]
) -> bool:
    groups: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows:
        market_id = str(row.get("market_id") or "")
        decision_ts = int(row.get("decision_ts") or 0)
        action = str(row.get("action") or "")
        if (
            market_id not in selected_market_ids
            or decision_ts <= 0
            or action not in FIVE_ACTIONS
            or int(row.get("max_input_ts") or 0) > decision_ts
        ):
            return False
        groups[(market_id, decision_ts)].add(action)
    return (
        {market_id for market_id, _ in groups} == selected_market_ids
        and len(selected_market_ids) == EXACT_MARKET_COUNT
        and all(actions == FIVE_ACTIONS for actions in groups.values())
    )


def _guard_side(row: dict[str, Any]) -> str:
    return str(row.get("selected_side") or row.get("side") or "NONE")


def _validate_runtime_target_rows(
    rows: list[dict[str, Any]], *, market_ids: list[str]
) -> None:
    allowed = set(market_ids)
    seen: set[str] = set()
    for row in rows:
        market_id = str(row.get("market_id") or "")
        side = str(row.get("side") or row.get("selected_side") or "")
        action = str(row.get("action") or row.get("executed_action") or "")
        if (
            not market_id
            or market_id not in allowed
            or market_id in seen
            or side not in SIDES
            or action not in SBC_ACTIONS
        ):
            raise ValueError("#246 runtime target identity invalid")
        seen.add(market_id)
        value = float(row["runtime_policy_after_cost_net_pnl_at_frozen_size"])
        if not math.isfinite(value):
            raise ValueError("#246 runtime target PnL is non-finite")
