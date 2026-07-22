"""Frozen #241 future-unseen holdout contract for v7.7."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7 import (
    SBC_ACTIONS,
    SIDES,
)
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
    _blocked_safety_fields,
    _require_git_sha,
    _require_sha256,
)

SCHEMA_PREFIX = (
    "bigan-v8-rolling-origin-drift-adaptive-action-value-v7-7-future-holdout"
)
PLAN_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-plan-v1"
CANDIDATE_NAME = "rolling_origin_drift_adaptive_action_value_v7_7"
BASELINE_NAME = "p_up_semantic_compatibility_v6_7"
EXACT_MARKET_COUNT = 120
SCAN_CAP = 180
BOUNDED_BATCH_MARKET_COUNT = 12
MINIMUM_GUARD_ACCEPTED_MARKET_COUNT = 40
STRICTLY_LATER_MINIMUM_MARKET_START_TS_EXCLUSIVE = 1_784_760_900_000
FROZEN_PLAN_SHA256 = "9750317ff18d698f5130489a871ffb9c71812bb79bf172e25b00ce4dc9382602"


def _safety_fields() -> dict[str, Any]:
    return {
        **_blocked_safety_fields(),
        "paper_candidate_allowed": False,
        "live_trading_enabled": False,
    }


def validate_v7_7_future_holdout_plan(plan: dict[str, Any]) -> None:
    """Reject lineage, sampling, gate, target-access, or safety drift."""

    lineage = dict(plan.get("lineage") or {})
    collection = dict(plan.get("collection") or {})
    freeze = dict(plan.get("target_free_decision_freeze") or {})
    gate = dict(plan.get("single_use_future_pnl_gate") or {})
    scope = dict(plan.get("result_scope") or {})

    _require_git_sha(str(lineage.get("implementation_commit") or ""))
    expected_lineage_fields = {
        "historical_manifest_sha256",
        "historical_model_sha256",
        "v7_7_profile_sha256",
        "target_free_canary_plan_sha256",
        "target_free_canary_manifest_sha256",
        "target_free_canary_report_sha256",
        "target_free_canary_batch_index_sha256",
        "target_free_canary_batch_last_entry_sha256",
        "v6_2_source_candidate_manifest_sha256",
        "v6_7_profile_sha256",
        "runtime_policy_profile_sha256",
        "collector_protocol_sha256",
        "feature_contract_sha256",
    }
    if set(lineage) != expected_lineage_fields | {"implementation_commit"}:
        raise ValueError("#241 frozen lineage field set drifted")
    for name in expected_lineage_fields:
        _require_sha256(str(lineage[name]), name=name)

    checks = {
        "schema": plan.get("schema_version") == PLAN_SCHEMA_VERSION,
        "issue": plan.get("issue_number") == 241,
        "candidate": plan.get("candidate_name") == CANDIDATE_NAME,
        "baseline": plan.get("baseline_name") == BASELINE_NAME,
        "frozen": plan.get("frozen") is True,
        "preregistered": plan.get("preregistered_before_collection") is True,
        "collection": collection
        == {
            "mode": "bounded_candidate_agnostic_outcome_blind_raw_collection",
            "selection_method": "earliest_quality_valid_strictly_later_disjoint_markets",
            "exact_quality_valid_market_count": EXACT_MARKET_COUNT,
            "maximum_attempted_market_count": SCAN_CAP,
            "bounded_batch_market_count": BOUNDED_BATCH_MARKET_COUNT,
            "strictly_later_minimum_market_start_ts_exclusive": STRICTLY_LATER_MINIMUM_MARKET_START_TS_EXCLUSIVE,
            "boundary_source": "complete_target_free_canary_batch_latest_market_close_ts",
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
            "minimum_v7_7_guard_accepted_unique_market_count": MINIMUM_GUARD_ACCEPTED_MARKET_COUNT,
            "one_frozen_decision_per_policy_per_market": True,
            "full_v6_7_execution_guard_unchanged": True,
            "fixed_position_size_unchanged": True,
            "source_score_mutation_allowed": False,
            "model_score_or_threshold_tuning_allowed": False,
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
            "candidate_minus_v6_7_largest_winner_removed_after_cost_pnl_minimum_inclusive": 0.0,
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
        "safety": plan.get("safety") == _safety_fields(),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"#241 frozen holdout plan drifted: {failed}")


def build_v7_7_future_pnl_noninferiority_gate(
    candidate_rows: list[dict[str, Any]],
    *,
    baseline_rows: list[dict[str, Any]],
    evaluation_market_ids: list[str],
    settled_market_ids: list[str],
    plan: dict[str, Any],
    target_free_freeze_sha256: str,
) -> dict[str, Any]:
    """Compare distinct v7.7/v6.7 frozen decisions on the exact same markets."""

    validate_v7_7_future_holdout_plan(plan)
    _require_sha256(target_free_freeze_sha256, name="target_free_freeze_sha256")
    market_ids = list(dict.fromkeys(str(value) for value in evaluation_market_ids))
    settled_ids = list(dict.fromkeys(str(value) for value in settled_market_ids))
    if (
        len(market_ids) != EXACT_MARKET_COUNT
        or "" in market_ids
        or len(settled_ids) != EXACT_MARKET_COUNT
        or set(settled_ids) != set(market_ids)
    ):
        raise ValueError("#241 exact settled evaluation market identity invalid")
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
    delta_by_market = {
        market_id: candidate_by_market[market_id] - baseline_by_market[market_id]
        for market_id in market_ids
    }
    gate = dict(plan["single_use_future_pnl_gate"])
    freeze = dict(plan["target_free_decision_freeze"])
    target_isolation = all(
        row.get("target_available_only_post_exit_or_official_resolution") is True
        and row.get("target_used_as_decision_time_input") is False
        and int(row["max_input_ts"]) <= int(row["decision_ts"])
        for row in candidate_rows + baseline_rows
    )
    checks = {
        "exact_120_settled_market_reconciliation": set(settled_ids) == set(market_ids),
        "minimum_v7_7_guard_accepted_unique_market_support": len(candidate_rows)
        >= int(freeze["minimum_v7_7_guard_accepted_unique_market_count"]),
        "candidate_total_after_cost_pnl_positive": candidate_total
        > float(gate["candidate_total_after_cost_pnl_minimum_exclusive"]),
        "candidate_noninferior_to_v6_7_total_pnl": total_delta
        >= float(gate["candidate_minus_v6_7_total_after_cost_pnl_minimum_inclusive"]),
        "candidate_noninferior_to_v6_7_largest_winner_removed": lwr_delta
        >= float(
            gate[
                "candidate_minus_v6_7_largest_winner_removed_after_cost_pnl_minimum_inclusive"
            ]
        ),
        "settlement_causality_and_target_isolation": target_isolation,
    }
    reason_map = {
        "exact_120_settled_market_reconciliation": "exact_120_settlement_reconciliation_failed",
        "minimum_v7_7_guard_accepted_unique_market_support": "insufficient_v7_7_guard_accepted_unique_market_support",
        "candidate_total_after_cost_pnl_positive": "candidate_total_after_cost_pnl_not_positive",
        "candidate_noninferior_to_v6_7_total_pnl": "candidate_total_pnl_inferior_to_v6_7",
        "candidate_noninferior_to_v6_7_largest_winner_removed": "candidate_largest_winner_removed_pnl_inferior_to_v6_7",
        "settlement_causality_and_target_isolation": "settlement_causality_or_target_isolation_failed",
    }
    blockers = [reason_map[name] for name, passed in checks.items() if not passed]
    candidate_by_side: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidate_by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    baseline_by_side: dict[str, list[dict[str, Any]]] = defaultdict(list)
    baseline_by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        candidate_by_side[str(row["side"])].append(row)
        candidate_by_action[str(row["action"])].append(row)
    for row in baseline_rows:
        baseline_by_side[str(row["side"])].append(row)
        baseline_by_action[str(row["action"])].append(row)
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
        "candidate_minus_v6_7_after_cost_pnl_by_market": delta_by_market,
        "candidate_side_distribution_diagnostic": dict(
            sorted(Counter(str(row["side"]) for row in candidate_rows).items())
        ),
        "v6_7_side_distribution_diagnostic": dict(
            sorted(Counter(str(row["side"]) for row in baseline_rows).items())
        ),
        "candidate_side_metrics_diagnostic": {
            side: _group_metrics(rows) for side, rows in sorted(candidate_by_side.items())
        },
        "v6_7_side_metrics_diagnostic": {
            side: _group_metrics(rows) for side, rows in sorted(baseline_by_side.items())
        },
        "candidate_action_metrics_diagnostic": {
            action: _group_metrics(rows)
            for action, rows in sorted(candidate_by_action.items())
        },
        "v6_7_action_metrics_diagnostic": {
            action: _group_metrics(rows)
            for action, rows in sorted(baseline_by_action.items())
        },
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
        **_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


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
            raise ValueError("#241 runtime target identity invalid")
        seen.add(market_id)
        value = float(row["runtime_policy_after_cost_net_pnl_at_frozen_size"])
        if not math.isfinite(value):
            raise ValueError("#241 runtime target PnL is non-finite")


def _group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["runtime_policy_after_cost_net_pnl_at_frozen_size"]) for row in rows]
    return {
        "accepted_bet_count": len(rows),
        "accepted_unique_market_count": len({str(row["market_id"]) for row in rows}),
        "after_cost_pnl_sum": float(sum(values)),
        "after_cost_pnl_mean": float(sum(values) / len(values)) if values else 0.0,
        "win_rate": float(sum(value > 0.0 for value in values) / len(values)) if values else 0.0,
        "diagnostic_only": True,
    }
