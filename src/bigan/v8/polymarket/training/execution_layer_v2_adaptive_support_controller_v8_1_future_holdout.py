"""Frozen future-unseen holdout contract for issue #246 v8.1."""

from __future__ import annotations

from typing import Any

from bigan.v8.polymarket.training.execution_layer_v2_abstention_aware_expected_net_pnl_v7_0 import (
    _v7_0_blocked_safety_fields,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
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
