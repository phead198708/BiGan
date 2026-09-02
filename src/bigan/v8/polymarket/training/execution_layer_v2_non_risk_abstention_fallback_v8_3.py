"""Non-risk abstention fallback for issue #248."""

from __future__ import annotations

import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training import (
    execution_layer_v2_adaptive_support_controller_v8_1 as v81,
)
from bigan.v8.polymarket.training import (
    execution_layer_v2_adaptive_support_controller_v8_1_canary as v81_canary,
)
from bigan.v8.polymarket.training import (
    execution_layer_v2_support_preserving_overlay_v8_2 as v82,
)
from bigan.v8.polymarket.training.execution_layer_v2_abstention_aware_expected_net_pnl_v7_0 import (
    _v7_0_blocked_safety_fields,
)
from bigan.v8.polymarket.training.execution_layer_v2_nested_boosted_action_value_v7_4_canary import (
    _canonicalize_target_free_sbc_rows,
)
from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7 import (
    _microstructure_blocking_reasons,
    build_v6_7_target_free_candidate_rows,
    select_v6_7_target_free_rows,
    validate_p_up_semantic_compatibility_v6_7_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_rolling_origin_drift_adaptive_action_value_v7_7_canary import (
    _action_key,
    _earliest_market_ids,
)
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
    _descriptor,
    _load_json,
    _load_jsonl,
    _sha256_file,
    _verify_pin,
    _write_json,
    _write_jsonl,
    _write_text,
)
from bigan.v8.polymarket.training.execution_layer_v2_v6_7_relative_safe_policy_v7_2 import (
    FORBIDDEN_INFERENCE_FIELDS,
)

CANDIDATE_NAME = "non_risk_abstention_fallback_v8_3"
PROFILE_SCHEMA_VERSION = (
    "bigan-v8-non-risk-abstention-fallback-v8-3-profile-v1"
)
DECISION_SCHEMA_VERSION = (
    "bigan-v8-non-risk-abstention-fallback-v8-3-decision-v1"
)
HISTORICAL_REPORT_SCHEMA_VERSION = (
    "bigan-v8-non-risk-abstention-fallback-v8-3-historical-report-v1"
)
CANARY_REPORT_SCHEMA_VERSION = (
    "bigan-v8-non-risk-abstention-fallback-v8-3-canary-report-v1"
)
MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-non-risk-abstention-fallback-v8-3-manifest-v1"
)
FUTURE_PLAN_SCHEMA_VERSION = (
    "bigan-v8-non-risk-abstention-fallback-v8-3-future-holdout-plan-v1"
)
FUTURE_PLAN_V2_SCHEMA_VERSION = (
    "bigan-v8-non-risk-abstention-fallback-v8-3-future-holdout-plan-v2"
)
FUTURE_SCHEMA_PREFIX = (
    "bigan-v8-non-risk-abstention-fallback-v8-3-future-holdout"
)
FUTURE_EXACT_MARKET_COUNT = 120
FUTURE_SCAN_CAP = 180
FUTURE_MINIMUM_GUARD_ACCEPTED_MARKET_COUNT = 40
POLICY_LEVEL_ABSTENTION_REASON_CODES = {
    "policy_selected_no_trade",
    "v6_7_no_positive_guard_compatible_action",
    "v8_1_veto_to_no_trade",
}


@dataclass(frozen=True, slots=True)
class NonRiskAbstentionFallbackV83HistoricalConfig:
    run_id: str
    output_dir: str
    profile_path: str
    expected_profile_sha256: str
    historical_manifest_path: str
    expected_historical_manifest_sha256: str
    implementation_commit: str
    evaluation_started_ts: int
    overwrite_existing: bool = False


@dataclass(frozen=True, slots=True)
class NonRiskAbstentionFallbackV83CanaryConfig:
    run_id: str
    output_dir: str
    profile_path: str
    expected_profile_sha256: str
    historical_gate_manifest_path: str
    expected_historical_gate_manifest_sha256: str
    issue246_target_free_manifest_path: str
    expected_issue246_target_free_manifest_sha256: str
    implementation_commit: str
    canary_started_ts: int
    overwrite_existing: bool = False


@dataclass(frozen=True, slots=True)
class NonRiskAbstentionFallbackV83BatchConfig:
    run_id: str
    output_dir: str
    profile_path: str
    expected_profile_sha256: str
    future_plan_path: str
    expected_future_plan_sha256: str
    development_batch_manifest_path: str
    expected_development_batch_manifest_sha256: str
    v6_2_batch_manifest_path: str
    expected_v6_2_batch_manifest_sha256: str
    v8_1_historical_manifest_path: str
    expected_v8_1_historical_manifest_sha256: str
    implementation_commit: str
    diagnostic_started_ts: int
    overwrite_existing: bool = False


def validate_non_risk_abstention_fallback_v8_3_profile(
    profile: dict[str, Any],
) -> None:
    checks = {
        "identity": profile.get("schema_version") == PROFILE_SCHEMA_VERSION
        and profile.get("issue_number") == 248
        and profile.get("candidate_name") == CANDIDATE_NAME
        and profile.get(
            "preregistered_before_implementation_and_historical_target_access"
        )
        is True,
        "policy": profile.get("policy_contract")
        == {
            "candidate_primary": "adaptive_support_controller_v8_1",
            "explicit_execution_risk_blocker_bypass_allowed": False,
            "fallback_baseline": "p_up_semantic_compatibility_v6_7",
            "fallback_requires_independent_full_guard_pass": True,
            "fallback_trigger": "v8_1_policy_level_non_risk_abstention",
            "full_execution_guard_unchanged": True,
            "model_threshold_quantile_cost_sizing_or_guard_changed": False,
            "policy_level_abstention_reason_codes": sorted(
                POLICY_LEVEL_ABSTENTION_REASON_CODES
            ),
            "source_or_o_score_mutation_allowed": False,
        },
        "historical_gate": profile.get("historical_gate")
        == {
            "candidate_minus_v6_7_largest_winner_removed_after_cost_pnl_minimum_inclusive": 0.0,
            "candidate_minus_v6_7_total_after_cost_pnl_minimum_inclusive": 0.0,
            "candidate_support_not_below_v6_7": True,
            "equality_passes_noninferiority": True,
            "side_quota_enabled": False,
        },
        "canary_gate": profile.get("target_free_canary_gate")
        == {
            "exact_market_count": 120,
            "minimum_guard_accepted_market_count": 40,
            "outcomes_labels_resolution_or_pnl_opened": False,
            "side_quota_enabled": False,
        },
        "lineage": profile.get("lineage", {}).get(
            "issue246_outcomes_allowed_for_v8_3"
        )
        is False,
        "safety": profile.get("safety") == _expected_safety(),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("#248 v8.3 profile invalid: " + ", ".join(blockers))


def validate_non_risk_abstention_fallback_v8_3_future_plan(
    plan: dict[str, Any],
) -> None:
    collection = plan.get("collection", {})
    diagnostic = plan.get("per_batch_target_free_diagnostic", {})
    freeze = plan.get("target_free_decision_freeze", {})
    future_gate = plan.get("single_use_future_pnl_gate", {})
    lineage = plan.get("lineage", {})
    schema_version = plan.get("schema_version")
    issue_number = plan.get("issue_number")
    v1 = schema_version == FUTURE_PLAN_SCHEMA_VERSION and issue_number == 249
    v2 = schema_version == FUTURE_PLAN_V2_SCHEMA_VERSION and issue_number == 250
    prior_latest_market_end_ts = lineage.get(
        "latest_prior_selected_market_end_ts",
        lineage.get("issue246_latest_selected_market_end_ts", 0),
    )
    lineage_valid = (
        lineage.get("candidate_profile_sha256")
        == "84c6bc06db0c2d25d342ecda23f5c06a4d9809c39db94a8eca1e550a4f822088"
        and lineage.get("historical_gate_manifest_sha256")
        == "adb930dc8bde72d89ae8b7520907ad88bb29d54f9c3b22317f9f4635ce5e015d"
        and lineage.get("issue246_target_free_canary_manifest_sha256")
        == "bf4cff80df1bf92a3980b7e79c772f8bfe55f34f575cb8d2ad0e52235376a18b"
    )
    if v2:
        lineage_valid = lineage_valid and {
            "candidate_implementation_commit": lineage.get(
                "candidate_implementation_commit"
            ),
            "candidate_decision_policy_source_sha256": lineage.get(
                "candidate_decision_policy_source_sha256"
            ),
            "settlement_fallback_source_sha256": lineage.get(
                "settlement_fallback_source_sha256"
            ),
            "issue249_consumed_freeze_manifest_sha256": lineage.get(
                "issue249_consumed_freeze_manifest_sha256"
            ),
            "issue249_target_access_claim_sha256": lineage.get(
                "issue249_target_access_claim_sha256"
            ),
            "issue249_terminal_settlement_report_sha256": lineage.get(
                "issue249_terminal_settlement_report_sha256"
            ),
            "frozen_feature_artifact_canary_manifest_sha256": lineage.get(
                "frozen_feature_artifact_canary_manifest_sha256"
            ),
            "frozen_feature_artifact_canary_report_sha256": lineage.get(
                "frozen_feature_artifact_canary_report_sha256"
            ),
            "frozen_feature_rows_sha256": lineage.get(
                "frozen_feature_rows_sha256"
            ),
        } == {
            "candidate_implementation_commit": (
                "9eff4026f4fc6e1eeff70e0cef3685feb542a7d9"
            ),
            "candidate_decision_policy_source_sha256": (
                "6622e5bf8c349f58bb9977bde1007cf69450225dda5abe0a9d14dbb5848469cf"
            ),
            "settlement_fallback_source_sha256": (
                "6f0670c041e4258c1451a3332c6e67dfad2f4ab57313ce6fc48ea24619c9749c"
            ),
            "issue249_consumed_freeze_manifest_sha256": (
                "e2e70d0bac83e2fac1bafee4e4c913aeaf1595272aaa73d62da7ac9a8ff5b499"
            ),
            "issue249_target_access_claim_sha256": (
                "15c73d459dda3045e673c89d00ea8d1255d832a14415b5a819767a444d144f53"
            ),
            "issue249_terminal_settlement_report_sha256": (
                "e8f426604395dd0cff27985035a7a3a1db341a8bdcca7f7ac8df000cf6a83fac"
            ),
            "frozen_feature_artifact_canary_manifest_sha256": (
                "2cef83685dde40fcfa9b6c2be81496a5dc41f3fc798ede6dad2d587668b1ccb7"
            ),
            "frozen_feature_artifact_canary_report_sha256": (
                "9cf9e84395f8609fd909f33f3728016114ad5bdcd91138ed2f9e6dc1b65e67ef"
            ),
            "frozen_feature_rows_sha256": (
                "ba806b6926bdcbe6d0f3344f1a7aab921563740cfddd7c180adf826c70beff66"
            ),
        }
    hardening = plan.get("settlement_hardening")
    checks = {
        "identity": (v1 or v2)
        and plan.get("candidate_name") == CANDIDATE_NAME
        and plan.get("frozen") is True
        and plan.get("preregistered_before_collection") is True,
        "collection": collection
        == {
            "bounded_batch_market_count": 12,
            "candidate_scoring_during_raw_capture_allowed": False,
            "decision_id_disjointness_required": True,
            "exact_quality_valid_market_count": 120,
            "feature_timestamp_causality_required": True,
            "market_id_disjointness_required": True,
            "maximum_attempted_market_count": 180,
            "mode": "bounded_candidate_agnostic_outcome_blind_raw_collection",
            "outcomes_resolution_labels_or_pnl_opened": False,
            "partial_authoritative_window_freeze_allowed": False,
            "raw_artifact_hash_verification_required": True,
            "selection_method": (
                "earliest_quality_valid_strictly_later_disjoint_markets"
            ),
            "slug_disjointness_required": True,
            "source_row_hash_disjointness_required": True,
            "strictly_later_minimum_market_start_ts_exclusive": (
                plan.get("plan_created_ts")
            ),
        },
        "strict_boundary": isinstance(plan.get("plan_created_ts"), int)
        and plan["plan_created_ts"]
        > prior_latest_market_end_ts,
        "batch_diagnostic": diagnostic
        == {
            "action_and_fallback_distribution_reported": True,
            "candidate_scoring_after_batch_seal_allowed": True,
            "consecutive_zero_support_batches_before_stop": 2,
            "full_guard_blocker_distribution_reported": True,
            "model_threshold_cost_sizing_guard_or_gate_tuning_allowed": False,
            "outcomes_resolution_labels_or_pnl_opened": False,
            "provider_causality_or_hash_violation_stops_fail_closed": True,
            "result_scope": "engineering_and_actionability_only",
        },
        "freeze": freeze
        == {
            "exact_market_count": 120,
            "full_v6_7_execution_guard_unchanged": True,
            "minimum_candidate_guard_accepted_unique_market_count": 40,
            "model_score_threshold_fallback_cost_sizing_or_guard_tuning_allowed": False,
            "one_frozen_decision_per_policy_per_market": True,
            "outcomes_resolution_labels_or_pnl_opened": False,
            "side_quota_enabled": False,
            "source_score_mutation_allowed": False,
        },
        "future_gate": future_gate
        == {
            "candidate_minus_v6_7_largest_winner_removed_after_cost_pnl_minimum_inclusive": 0.0,
            "candidate_minus_v6_7_total_after_cost_pnl_minimum_inclusive": 0.0,
            "candidate_total_after_cost_pnl_minimum_exclusive": 0.0,
            "complete_settlement_required": True,
            "equality_passes_comparative_noninferiority": True,
            "official_read_only_settlement_on_quarantine_copies": True,
            "outcomes_used_for_selection_or_tuning": False,
            "result_selected_extension_allowed": False,
            "result_selected_rerun_allowed": False,
            "side_action_and_family_metrics_diagnostic_only": True,
        },
        "lineage": lineage_valid,
        "settlement_hardening": (
            hardening is None
            if v1
            else hardening
            == {
                "additional_finalization_blockers_allowed": False,
                "direct_training_or_export_eligibility_relaxed": False,
                "evaluation_only_fallback_requires_exact_frozen_feature_payload_match": True,
                "evaluation_only_fallback_requires_official_resolution": True,
                "evaluation_only_fallback_requires_zero_timestamp_violations": True,
                "frozen_feature_market_coverage_required": True,
                "frozen_feature_rows_hash_pinned_before_target_access": True,
                "legacy_freeze_without_frozen_features_fails_closed": True,
            }
        ),
        "safety": plan.get("safety") == _expected_safety(),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError(
            "#249 v8.3 future plan invalid: " + ", ".join(blockers)
        )


def select_non_risk_abstention_fallback_v8_3_future_window(
    index_rows: list[dict[str, Any]],
    *,
    plan: dict[str, Any],
    prior_market_ids: set[str],
    prior_slugs: set[str],
    prior_decision_ids: set[str],
    prior_source_row_hashes: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Select the earliest exact-120 eligible #249 markets."""

    validate_non_risk_abstention_fallback_v8_3_future_plan(plan)
    collection = dict(plan["collection"])
    boundary = int(
        collection["strictly_later_minimum_market_start_ts_exclusive"]
    )
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
        reasons: list[str] = []
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
        if len(eligible) == FUTURE_EXACT_MARKET_COUNT:
            break
    selected = eligible[:FUTURE_EXACT_MARKET_COUNT]
    selected_ids = [str(row.get("market_id") or "") for row in selected]
    summary = {
        "attempted_scan_count": len(attempted),
        "eligible_market_count": len(eligible),
        "selected_market_count": len(selected),
        "selected_sequence_start": (
            int(selected[0]["sequence"]) if selected else None
        ),
        "selected_sequence_end": (
            int(selected[-1]["sequence"]) if selected else None
        ),
        "selected_market_ids_sha256": canonical_json_sha256(selected_ids),
        "exclusion_reason_distribution": dict(
            sorted(exclusion_reasons.items())
        ),
        "strictly_later_time_violation_count": sum(
            int(row.get("market_start_ts") or 0) <= boundary
            for row in selected
        ),
        "selected_identity_duplicate_count": len(selected_ids)
        - len(set(selected_ids)),
        "exact_window_ready": (
            len(selected) == FUTURE_EXACT_MARKET_COUNT
            and "" not in selected_ids
            and len(set(selected_ids)) == FUTURE_EXACT_MARKET_COUNT
        ),
    }
    return selected, attempted, summary


def build_non_risk_abstention_fallback_v8_3_target_free_freeze_report(
    selected_rows: list[dict[str, Any]],
    *,
    attempted_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    overlay_decisions: list[dict[str, Any]],
    baseline_guard_rows: list[dict[str, Any]],
    selection_summary: dict[str, Any],
    plan: dict[str, Any],
    stage_started_ts: int,
    collector_index_sha256: str,
) -> dict[str, Any]:
    """Validate the authoritative v8.3/v6.7 decision freeze."""

    validate_non_risk_abstention_fallback_v8_3_future_plan(plan)
    selected_ids = [str(row.get("market_id") or "") for row in selected_rows]
    selected_set = set(selected_ids)
    filtered_actions = [
        row
        for row in action_rows
        if str(row.get("market_id") or "") in selected_set
    ]
    filtered_features = [
        row
        for row in feature_rows
        if str(row.get("market_id") or "") in selected_set
    ]
    overlay_by_market = _one_row_per_market(
        overlay_decisions, label="v8.3 overlay"
    )
    baseline_by_market = _one_row_per_market(
        baseline_guard_rows, label="v6.7 guard"
    )
    forbidden = sorted(
        set(
            v81_canary._find_nonempty_fields(
                filtered_actions, FORBIDDEN_INFERENCE_FIELDS
            )
        )
        | set(
            v81_canary._find_nonempty_fields(
                filtered_features, FORBIDDEN_INFERENCE_FIELDS
            )
        )
        | set(
            v81_canary._find_nonempty_fields(
                overlay_decisions, FORBIDDEN_INFERENCE_FIELDS
            )
        )
        | set(
            v81_canary._find_nonempty_fields(
                baseline_guard_rows, FORBIDDEN_INFERENCE_FIELDS
            )
        )
    )
    causality_violations = sum(
        int(row.get("max_input_ts") or 0)
        > int(row.get("decision_ts") or 0)
        for row in filtered_actions + filtered_features
    )
    feature_market_ids = {
        str(row.get("market_id") or "") for row in filtered_features
    } - {""}
    overlay_accepted = [
        row
        for row in overlay_decisions
        if row.get("execution_guard_order_allowed") is True
    ]
    baseline_accepted = [
        row
        for row in baseline_guard_rows
        if row.get("execution_guard_order_allowed") is True
    ]
    five_action_groups: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in filtered_actions:
        five_action_groups[
            (
                str(row.get("market_id") or ""),
                int(row.get("decision_ts") or 0),
            )
        ].add(str(row.get("action") or ""))
    expected_actions = {
        "BUY_UP_SELL_BEFORE_CLOSE",
        "BUY_DOWN_SELL_BEFORE_CLOSE",
        "BUY_UP_HOLD_TO_SETTLEMENT",
        "BUY_DOWN_HOLD_TO_SETTLEMENT",
        "NO_TRADE",
    }
    complete_grid = (
        {market_id for market_id, _ in five_action_groups} == selected_set
        and all(actions == expected_actions for actions in five_action_groups.values())
    )
    checks = {
        "exact_120_selected_markets": (
            selection_summary.get("exact_window_ready") is True
            and len(selected_rows) == FUTURE_EXACT_MARKET_COUNT
            and len(selected_set) == FUTURE_EXACT_MARKET_COUNT
        ),
        "attempted_scan_cap_respected": len(attempted_rows) <= FUTURE_SCAN_CAP,
        "all_selected_markets_closed_before_target_access": (
            bool(selected_rows)
            and stage_started_ts
            > max(int(row.get("market_end_ts") or 0) for row in selected_rows)
        ),
        "complete_five_action_grid": complete_grid,
        "complete_frozen_feature_coverage": feature_market_ids == selected_set,
        "candidate_complete_decision_coverage": set(overlay_by_market)
        == selected_set,
        "baseline_complete_decision_coverage": set(baseline_by_market)
        == selected_set,
        "minimum_candidate_guard_accepted_market_support": len(
            overlay_accepted
        )
        >= FUTURE_MINIMUM_GUARD_ACCEPTED_MARKET_COUNT,
        "feature_timestamp_causality": causality_violations == 0,
        "forbidden_target_fields_absent": not forbidden,
        "source_scores_unchanged": all(
            row.get("source_score_mutated") is False
            for row in overlay_decisions + baseline_guard_rows
        ),
        "risk_blocker_bypass_absent": all(
            row.get("explicit_execution_risk_blocker_bypass_used") is False
            for row in overlay_decisions
        ),
        "outcomes_resolution_labels_or_pnl_sealed": all(
            row.get("target_or_outcome_used_for_selection") is not True
            and row.get("labels_outcomes_resolution_or_pnl_opened") is not True
            and row.get("labels_outcomes_or_pnl_opened") is not True
            for row in overlay_decisions + baseline_guard_rows
        ),
    }
    reason_map = {
        "exact_120_selected_markets": "target_free_exact_120_window_not_ready",
        "attempted_scan_cap_respected": "target_free_scan_cap_exceeded",
        "all_selected_markets_closed_before_target_access": (
            "target_free_markets_not_all_closed"
        ),
        "complete_five_action_grid": "target_free_five_action_grid_incomplete",
        "complete_frozen_feature_coverage": (
            "target_free_feature_coverage_incomplete"
        ),
        "candidate_complete_decision_coverage": (
            "target_free_v8_3_decision_coverage_incomplete"
        ),
        "baseline_complete_decision_coverage": (
            "target_free_v6_7_decision_coverage_incomplete"
        ),
        "minimum_candidate_guard_accepted_market_support": (
            "target_free_v8_3_guard_accepted_support_insufficient"
        ),
        "feature_timestamp_causality": "target_free_feature_causality_violation",
        "forbidden_target_fields_absent": (
            "target_free_forbidden_target_field_present"
        ),
        "source_scores_unchanged": "target_free_source_score_mutated",
        "risk_blocker_bypass_absent": "execution_risk_blocker_bypass_detected",
        "outcomes_resolution_labels_or_pnl_sealed": (
            "target_free_outcome_or_pnl_access_detected"
        ),
    }
    blockers = [
        reason_map[name] for name, passed in checks.items() if not passed
    ]
    report = {
        "schema_version": f"{FUTURE_SCHEMA_PREFIX}-target-free-freeze-report-v1",
        "candidate_name": CANDIDATE_NAME,
        "baseline_name": "p_up_semantic_compatibility_v6_7",
        "collector_index_sha256": collector_index_sha256,
        "selected_market_count": len(selected_rows),
        "attempted_market_count": len(attempted_rows),
        "selection_summary": selection_summary,
        "candidate_guard_accepted_market_count": len(overlay_accepted),
        "v6_7_guard_accepted_market_count": len(baseline_accepted),
        "selection_source_distribution": _distribution(
            overlay_decisions, "selection_source"
        ),
        "selected_action_distribution": _distribution(
            overlay_decisions, "selected_action"
        ),
        "selected_side_distribution_diagnostic": _distribution(
            [
                row
                for row in overlay_accepted
                if row.get("selected_side") != "NONE"
            ],
            "selected_side",
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
        "threshold_model_cost_sizing_guard_or_gate_tuning_performed": False,
        "source_scores_mutated": False,
        "promotion_evidence": False,
        **_v7_0_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def build_non_risk_abstention_fallback_v8_3_future_pnl_gate(
    candidate_rows: list[dict[str, Any]],
    *,
    baseline_rows: list[dict[str, Any]],
    evaluation_market_ids: list[str],
    settled_market_ids: list[str],
    plan: dict[str, Any],
    target_free_freeze_sha256: str,
) -> dict[str, Any]:
    """Run the preregistered single-use #249 PnL comparison."""

    validate_non_risk_abstention_fallback_v8_3_future_plan(plan)
    market_ids = list(dict.fromkeys(str(value) for value in evaluation_market_ids))
    settled_ids = list(dict.fromkeys(str(value) for value in settled_market_ids))
    if (
        len(market_ids) != FUTURE_EXACT_MARKET_COUNT
        or "" in market_ids
        or set(settled_ids) != set(market_ids)
    ):
        raise ValueError("#249 exact settled evaluation market identity invalid")
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
    candidate_largest = max(candidate_by_market.values(), default=0.0)
    baseline_largest = max(baseline_by_market.values(), default=0.0)
    candidate_lwr = candidate_total - max(candidate_largest, 0.0)
    baseline_lwr = baseline_total - max(baseline_largest, 0.0)
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
        >= FUTURE_MINIMUM_GUARD_ACCEPTED_MARKET_COUNT,
        "candidate_total_after_cost_pnl_positive": candidate_total
        > float(gate["candidate_total_after_cost_pnl_minimum_exclusive"]),
        "candidate_noninferior_to_v6_7_total_pnl": total_delta
        >= float(
            gate[
                "candidate_minus_v6_7_total_after_cost_pnl_minimum_inclusive"
            ]
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
            "insufficient_v8_3_guard_accepted_unique_market_support"
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
    blockers = [
        reason_map[name] for name, passed in checks.items() if not passed
    ]
    report = {
        "schema_version": f"{FUTURE_SCHEMA_PREFIX}-pnl-gate-report-v1",
        "candidate_name": CANDIDATE_NAME,
        "baseline_name": "p_up_semantic_compatibility_v6_7",
        "target_free_freeze_sha256": target_free_freeze_sha256,
        "evaluation_market_count": len(market_ids),
        "settled_market_count": len(settled_ids),
        "candidate_guard_accepted_unique_market_count": len(candidate_rows),
        "v6_7_guard_accepted_unique_market_count": len(baseline_rows),
        "candidate_after_cost_pnl": candidate_total,
        "v6_7_after_cost_pnl": baseline_total,
        "candidate_minus_v6_7_after_cost_pnl": total_delta,
        "candidate_largest_winner_after_cost_pnl": candidate_largest,
        "candidate_largest_winner_removed_after_cost_pnl": candidate_lwr,
        "v6_7_largest_winner_after_cost_pnl": baseline_largest,
        "v6_7_largest_winner_removed_after_cost_pnl": baseline_lwr,
        "candidate_minus_v6_7_largest_winner_removed_after_cost_pnl": (
            lwr_delta
        ),
        "candidate_side_distribution_diagnostic": dict(
            sorted(Counter(str(row["side"]) for row in candidate_rows).items())
        ),
        "v6_7_side_distribution_diagnostic": dict(
            sorted(Counter(str(row["side"]) for row in baseline_rows).items())
        ),
        "noninferiority_comparison_operator": "greater_than_or_equal",
        "equality_passes_noninferiority": True,
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


def select_non_risk_abstention_fallback_v8_3_decision(
    *,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    v82._validate_decision_input(candidate, label="v8.1")
    v82._validate_decision_input(baseline, label="v6.7")
    if candidate["market_id"] != baseline["market_id"]:
        raise ValueError("#248 candidate/baseline market mismatch")
    candidate_action = str(candidate["selected_action"])
    baseline_action = str(baseline["selected_action"])
    candidate_allowed = candidate["execution_guard_order_allowed"] is True
    baseline_allowed = baseline["execution_guard_order_allowed"] is True
    candidate_blockers = set(candidate["execution_blocking_reason_codes"])
    policy_abstention_only = (
        candidate_action == "NO_TRADE"
        and candidate_allowed is False
        and bool(candidate_blockers)
        and candidate_blockers <= POLICY_LEVEL_ABSTENTION_REASON_CODES
    )

    if candidate_allowed and candidate_action in v82.TRADE_ACTIONS:
        action = candidate_action
        side = str(candidate["selected_side"])
        source = "v8_1_primary"
        allowed = True
        reasons = ["v8_1_primary_full_guard_passed"]
    elif (
        policy_abstention_only
        and baseline_allowed
        and baseline_action in v82.TRADE_ACTIONS
    ):
        action = baseline_action
        side = str(baseline["selected_side"])
        source = "v6_7_non_risk_abstention_fallback"
        allowed = True
        reasons = [
            "v8_1_policy_level_non_risk_abstention",
            "v6_7_independent_full_guard_passed",
        ]
    else:
        action = "NO_TRADE"
        side = "NONE"
        source = "fail_closed_no_trade"
        allowed = False
        reasons = _no_trade_reasons(
            candidate=candidate,
            baseline=baseline,
            policy_abstention_only=policy_abstention_only,
        )
    decision = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "market_id": candidate["market_id"],
        "decision_ts": candidate.get("decision_ts"),
        "selected_action": action,
        "selected_side": side,
        "execution_guard_order_allowed": allowed,
        "selection_source": source,
        "selection_reason_codes": reasons,
        "original_v8_1_action": candidate_action,
        "original_v8_1_side": candidate["selected_side"],
        "original_v8_1_guard_allowed": candidate_allowed,
        "original_v8_1_blocking_reason_codes": sorted(candidate_blockers),
        "original_v8_1_rank_abstention_passed": candidate.get(
            "rank_abstention_passed"
        ),
        "original_v8_1_point_selected_action": candidate.get(
            "point_selected_action"
        ),
        "original_v6_7_action": baseline_action,
        "original_v6_7_side": baseline["selected_side"],
        "original_v6_7_guard_allowed": baseline_allowed,
        "original_v6_7_blocking_reason_codes": sorted(
            baseline["execution_blocking_reason_codes"]
        ),
        "fallback_applied": source == "v6_7_non_risk_abstention_fallback",
        "fallback_requires_independent_full_guard_pass": True,
        "explicit_execution_risk_blocker_bypass_used": False,
        "full_execution_guard_unchanged": True,
        "target_or_outcome_used_for_selection": False,
        "source_score_mutated": False,
        **_v7_0_blocked_safety_fields(),
    }
    v82._assert_target_free_decision(decision)
    decision["overlay_decision_id"] = canonical_json_sha256(decision)
    return decision


def build_non_risk_abstention_fallback_v8_3_historical(
    rows: list[dict[str, Any]],
    *,
    profile: dict[str, Any],
) -> dict[str, Any]:
    validate_non_risk_abstention_fallback_v8_3_profile(profile)
    rows_by_market: dict[str, dict[str, Any]] = {}
    decisions: list[dict[str, Any]] = []
    for row in rows:
        market_id = str(row["market_id"])
        if market_id in rows_by_market:
            raise ValueError(f"#248 duplicate historical market: {market_id}")
        rows_by_market[market_id] = row
        decisions.append(
            select_non_risk_abstention_fallback_v8_3_decision(
                candidate=v82._historical_candidate_projection(row),
                baseline=v82._historical_baseline_projection(row),
            )
        )
    evaluation = _evaluate_historical_decisions(
        decisions=decisions,
        rows_by_market=rows_by_market,
    )
    return {"decisions": decisions, **evaluation}


def build_non_risk_abstention_fallback_v8_3_canary(
    *,
    candidate_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    profile: dict[str, Any],
) -> dict[str, Any]:
    validate_non_risk_abstention_fallback_v8_3_profile(profile)
    baseline_by_market = {str(row["market_id"]): row for row in baseline_rows}
    if len(baseline_by_market) != len(baseline_rows):
        raise ValueError("#248 duplicate v6.7 canary market")
    decisions: list[dict[str, Any]] = []
    for row in candidate_rows:
        market_id = str(row["market_id"])
        baseline = baseline_by_market.get(market_id)
        if baseline is None:
            raise ValueError(f"#248 missing v6.7 canary row: {market_id}")
        decisions.append(
            select_non_risk_abstention_fallback_v8_3_decision(
                candidate=_future_candidate_projection(row),
                baseline=_future_baseline_projection(baseline),
            )
        )
    support = sum(
        row["execution_guard_order_allowed"] is True for row in decisions
    )
    checks = {
        "exact_120_markets": len(decisions) == 120,
        "minimum_guard_accepted_support": support >= 40,
        "targets_outcomes_resolution_or_pnl_sealed": True,
        "source_scores_unchanged": all(
            row["source_score_mutated"] is False for row in decisions
        ),
        "risk_blocker_bypass_absent": all(
            row["explicit_execution_risk_blocker_bypass_used"] is False
            for row in decisions
        ),
    }
    reason_map = {
        "exact_120_markets": "target_free_exact_market_count_not_met",
        "minimum_guard_accepted_support": (
            "target_free_guard_accepted_support_insufficient"
        ),
        "targets_outcomes_resolution_or_pnl_sealed": (
            "target_free_outcome_access_detected"
        ),
        "source_scores_unchanged": "source_score_mutation_detected",
        "risk_blocker_bypass_absent": "execution_risk_blocker_bypass_detected",
    }
    blockers = [
        reason_map[name] for name, passed in checks.items() if not passed
    ]
    report = {
        "schema_version": CANARY_REPORT_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "market_count": len(decisions),
        "guard_accepted_market_count": support,
        "selection_source_distribution": _distribution(
            decisions, "selection_source"
        ),
        "selected_action_distribution": _distribution(
            decisions, "selected_action"
        ),
        "selected_side_distribution_diagnostic": _distribution(
            [
                row
                for row in decisions
                if row["selected_side"] != "NONE"
            ],
            "selected_side",
        ),
        "side_quota_enabled": False,
        "issue246_outcomes_opened": False,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "threshold_model_quantile_cost_sizing_or_guard_changed": False,
        "checks": checks,
        "target_free_canary_passed": not blockers,
        "target_free_canary_blocking_reason_codes": blockers,
        "new_future_holdout_collection_allowed": not blockers,
        **_v7_0_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return {"decisions": decisions, "report": report}


def run_non_risk_abstention_fallback_v8_3_historical_gate(
    config: NonRiskAbstentionFallbackV83HistoricalConfig,
) -> dict[str, Any]:
    profile_path = Path(config.profile_path).resolve()
    input_manifest_path = Path(config.historical_manifest_path).resolve()
    _verify_pin(profile_path, config.expected_profile_sha256, "#248 profile")
    _verify_pin(
        input_manifest_path,
        config.expected_historical_manifest_sha256,
        "#248 v8.1 historical manifest",
    )
    profile = _load_json(profile_path)
    validate_non_risk_abstention_fallback_v8_3_profile(profile)
    if (
        profile["lineage"]["v8_1_historical_manifest_sha256"]
        != config.expected_historical_manifest_sha256
    ):
        raise ValueError("#248 historical manifest lineage mismatch")
    input_manifest = _load_json(input_manifest_path)
    rows_path = _verified_descriptor_path(
        input_manifest["prequential_rows"],
        label="#248 historical prequential rows",
    )
    result = build_non_risk_abstention_fallback_v8_3_historical(
        _load_jsonl(rows_path),
        profile=profile,
    )
    report = result["report"]
    report.update(
        {
            "run_id": config.run_id,
            "implementation_commit": config.implementation_commit,
            "evaluation_started_ts": config.evaluation_started_ts,
        }
    )
    report["report_id"] = canonical_json_sha256(report)
    return _write_historical_outputs(
        config=config,
        profile_path=profile_path,
        input_manifest_path=input_manifest_path,
        result={**result, "report": report},
    )


def run_non_risk_abstention_fallback_v8_3_canary(
    config: NonRiskAbstentionFallbackV83CanaryConfig,
) -> dict[str, Any]:
    profile_path = Path(config.profile_path).resolve()
    historical_manifest_path = Path(
        config.historical_gate_manifest_path
    ).resolve()
    issue246_manifest_path = Path(
        config.issue246_target_free_manifest_path
    ).resolve()
    _verify_pin(profile_path, config.expected_profile_sha256, "#248 profile")
    _verify_pin(
        historical_manifest_path,
        config.expected_historical_gate_manifest_sha256,
        "#248 historical gate manifest",
    )
    _verify_pin(
        issue246_manifest_path,
        config.expected_issue246_target_free_manifest_sha256,
        "#248 issue246 target-free manifest",
    )
    profile = _load_json(profile_path)
    validate_non_risk_abstention_fallback_v8_3_profile(profile)
    historical_manifest = _load_json(historical_manifest_path)
    if historical_manifest.get("historical_noninferiority_gate_passed") is not True:
        raise ValueError("#248 historical gate did not pass")
    issue246 = _load_json(issue246_manifest_path)
    if (
        issue246.get("labels_outcomes_resolution_or_pnl_opened") is not False
        or issue246.get("settlement_provider_called") is not False
        or issue246.get("source_scores_mutated") is not False
    ):
        raise ValueError("#248 issue246 input is not target-free")
    candidate_path = _verified_descriptor_path(
        issue246["candidate_guard"], label="#248 issue246 candidate guard"
    )
    baseline_path = _verified_descriptor_path(
        issue246["baseline_guard"], label="#248 issue246 v6.7 guard"
    )
    result = build_non_risk_abstention_fallback_v8_3_canary(
        candidate_rows=_load_jsonl(candidate_path),
        baseline_rows=_load_jsonl(baseline_path),
        profile=profile,
    )
    report = result["report"]
    report.update(
        {
            "run_id": config.run_id,
            "implementation_commit": config.implementation_commit,
            "canary_started_ts": config.canary_started_ts,
            "historical_gate_manifest_sha256": (
                config.expected_historical_gate_manifest_sha256
            ),
            "issue246_target_free_manifest_sha256": (
                config.expected_issue246_target_free_manifest_sha256
            ),
        }
    )
    report["report_id"] = canonical_json_sha256(report)
    return _write_canary_outputs(
        config=config,
        profile_path=profile_path,
        historical_manifest_path=historical_manifest_path,
        issue246_manifest_path=issue246_manifest_path,
        result={**result, "report": report},
    )


def run_non_risk_abstention_fallback_v8_3_batch_diagnostic(
    config: NonRiskAbstentionFallbackV83BatchConfig,
) -> dict[str, Any]:
    paths = {
        "profile": Path(config.profile_path).resolve(),
        "plan": Path(config.future_plan_path).resolve(),
        "development": Path(config.development_batch_manifest_path).resolve(),
        "v6_2": Path(config.v6_2_batch_manifest_path).resolve(),
        "v8_1_historical": Path(
            config.v8_1_historical_manifest_path
        ).resolve(),
    }
    pins = {
        "profile": config.expected_profile_sha256,
        "plan": config.expected_future_plan_sha256,
        "development": config.expected_development_batch_manifest_sha256,
        "v6_2": config.expected_v6_2_batch_manifest_sha256,
        "v8_1_historical": (
            config.expected_v8_1_historical_manifest_sha256
        ),
    }
    for name, path in paths.items():
        _verify_pin(path, pins[name], f"#249 batch {name}")
    profile = _load_json(paths["profile"])
    plan = _load_json(paths["plan"])
    development = _load_json(paths["development"])
    v6_2 = _load_json(paths["v6_2"])
    historical = _load_json(paths["v8_1_historical"])
    validate_non_risk_abstention_fallback_v8_3_profile(profile)
    validate_non_risk_abstention_fallback_v8_3_future_plan(plan)
    if (
        plan["lineage"]["candidate_profile_sha256"]
        != pins["profile"].lower()
        or plan["lineage"]["historical_gate_manifest_sha256"]
        != "adb930dc8bde72d89ae8b7520907ad88bb29d54f9c3b22317f9f4635ce5e015d"
    ):
        raise ValueError("#249 batch frozen lineage mismatch")
    if (
        development.get("development_data_canary_passed") is not True
        or development.get("labels_outcomes_or_pnl_opened") is not False
        or v6_2.get("labels_outcomes_or_pnl_opened") is not False
    ):
        raise ValueError("#249 batch input is not sealed outcome-free evidence")
    model_path = _verified_descriptor_path(
        historical["model"], label="#249 v8.1 model"
    )
    v8_1_profile_path = _verified_descriptor_path(
        historical["profile"], label="#249 v8.1 profile"
    )
    v6_7_profile_path = _verified_descriptor_path(
        historical["v6_7_candidate_profile"], label="#249 v6.7 profile"
    )
    v7_0_profile_path = _verified_descriptor_path(
        historical["v7_0_training_profile"], label="#249 v7.0 profile"
    )
    action_path = _verified_descriptor_path(
        development["five_action_grid"], label="#249 five-action grid"
    )
    scored_path = _verified_descriptor_path(
        v6_2["mean_ev_scored_rows"], label="#249 v6.2 scored rows"
    )
    model = _load_json(model_path)
    v8_1_profile = _load_json(v8_1_profile_path)
    v6_7_profile = _load_json(v6_7_profile_path)
    v7_0_profile = _load_json(v7_0_profile_path)
    v81.validate_adaptive_support_controller_v8_1_profile(v8_1_profile)
    validate_p_up_semantic_compatibility_v6_7_profile(v6_7_profile)
    action_rows = _load_jsonl(action_path)
    scored_rows = _load_jsonl(scored_path)
    forbidden = sorted(
        set(v81_canary._find_nonempty_fields(action_rows, FORBIDDEN_INFERENCE_FIELDS))
        | set(
            v81_canary._find_nonempty_fields(
                scored_rows, FORBIDDEN_INFERENCE_FIELDS
            )
        )
    )
    if forbidden:
        raise ValueError(
            "#249 batch forbidden inference fields: " + ",".join(forbidden)
        )
    available_markets = _earliest_market_ids(
        action_rows,
        target=len({str(row["market_id"]) for row in action_rows}),
    )
    selected_set = set(available_markets)
    action_rows = [
        row for row in action_rows if str(row["market_id"]) in selected_set
    ]
    scored_rows = [
        row for row in scored_rows if str(row["market_id"]) in selected_set
    ]
    candidate_rows, _ = build_v6_7_target_free_candidate_rows(
        scored_rows,
        action_rows=action_rows,
        profile=v6_7_profile,
    )
    baseline_rows = select_v6_7_target_free_rows(
        candidate_rows, profile=v6_7_profile
    )
    canonical_rows, canonical_summary = _canonicalize_target_free_sbc_rows(
        scored_rows,
        action_rows=action_rows,
        v6_7_profile=v6_7_profile,
        v7_0_profile=v7_0_profile,
    )
    _, candidate_guard_rows, _ = v81_canary._score_window(
        available_markets,
        canonical_rows=canonical_rows,
        baseline_rows=baseline_rows,
        action_rows=action_rows,
        model=model,
        v6_7_profile=v6_7_profile,
    )
    baseline_guard_rows = _baseline_guard_rows(
        available_markets,
        baseline_rows=baseline_rows,
        action_rows=action_rows,
        v6_7_profile=v6_7_profile,
    )
    overlay = build_non_risk_abstention_fallback_v8_3_canary(
        candidate_rows=candidate_guard_rows,
        baseline_rows=baseline_guard_rows,
        profile=profile,
    )
    decisions = overlay["decisions"]
    support = sum(
        row["execution_guard_order_allowed"] is True for row in decisions
    )
    causality_violations = sum(
        int(row.get("max_input_ts") or 0) > int(row.get("decision_ts") or 0)
        for row in action_rows
    )
    minimum_start = int(
        plan["collection"][
            "strictly_later_minimum_market_start_ts_exclusive"
        ]
    )
    time_violations = sum(
        int(row.get("market_close_ts") or 0) - 300_000 <= minimum_start
        for row in action_rows
    )
    checks = {
        "development_data_canary_passed": True,
        "v6_2_target_free_scoring_sealed": True,
        "forbidden_target_fields_absent": not forbidden,
        "complete_decision_coverage": len(decisions) == len(available_markets),
        "canonical_sbc_mapping_complete": canonical_summary[
            "missing_scored_or_source_action_row_count"
        ]
        == 0,
        "feature_timestamp_causality": causality_violations == 0,
        "strictly_later": time_violations == 0,
        "source_scores_unchanged": all(
            row["source_score_mutated"] is False for row in decisions
        ),
        "risk_blocker_bypass_absent": all(
            row["explicit_execution_risk_blocker_bypass_used"] is False
            for row in decisions
        ),
    }
    reason_map = {
        "development_data_canary_passed": "development_data_canary_not_passed",
        "v6_2_target_free_scoring_sealed": "v6_2_scoring_not_sealed",
        "forbidden_target_fields_absent": "forbidden_target_field_present",
        "complete_decision_coverage": "decision_coverage_incomplete",
        "canonical_sbc_mapping_complete": "canonical_sbc_mapping_incomplete",
        "feature_timestamp_causality": "feature_timestamp_causality_violation",
        "strictly_later": "batch_market_not_strictly_later",
        "source_scores_unchanged": "source_score_mutation_detected",
        "risk_blocker_bypass_absent": "execution_risk_blocker_bypass_detected",
    }
    blockers = [
        reason_map[name] for name, passed in checks.items() if not passed
    ]
    report = {
        "schema_version": (
            "bigan-v8-non-risk-abstention-fallback-v8-3-batch-report-v1"
        ),
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        "diagnostic_started_ts": config.diagnostic_started_ts,
        "batch_market_count": len(available_markets),
        "guard_accepted_market_count": support,
        "zero_guard_accepted_support": support == 0,
        "selection_source_distribution": _distribution(
            decisions, "selection_source"
        ),
        "selected_action_distribution": _distribution(
            decisions, "selected_action"
        ),
        "selected_side_distribution_diagnostic": _distribution(
            [
                row
                for row in decisions
                if row["selected_side"] != "NONE"
            ],
            "selected_side",
        ),
        "execution_blocking_reason_distribution": dict(
            sorted(
                Counter(
                    reason
                    for row in decisions
                    if row["execution_guard_order_allowed"] is False
                    for reason in row["selection_reason_codes"]
                ).items()
            )
        ),
        "feature_causality_violation_count": causality_violations,
        "strictly_later_violation_count": time_violations,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "threshold_model_cost_sizing_guard_or_gate_tuning_performed": False,
        "checks": checks,
        "batch_target_free_diagnostic_passed": not blockers,
        "batch_target_free_blocking_reason_codes": blockers,
        **_v7_0_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    run_dir = _fresh_run_dir(
        output_dir=config.output_dir,
        run_id=config.run_id,
        overwrite_existing=config.overwrite_existing,
    )
    outputs = {
        "report": run_dir / "v8_3_batch_target_free_report.json",
        "report_markdown": run_dir / "v8_3_batch_target_free_report.md",
        "decision_rows": run_dir / "v8_3_batch_target_free_decisions.jsonl",
    }
    _write_json(outputs["report"], report)
    _write_text(outputs["report_markdown"], _markdown(report))
    _write_jsonl(outputs["decision_rows"], decisions)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "stage": "post_seal_batch_target_free_diagnostic",
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        **{name: _descriptor(path) for name, path in paths.items()},
        **{name: _descriptor(path) for name, path in outputs.items()},
        "batch_target_free_diagnostic_passed": report[
            "batch_target_free_diagnostic_passed"
        ],
        "zero_guard_accepted_support": report["zero_guard_accepted_support"],
        "labels_outcomes_resolution_or_pnl_opened": False,
        "new_future_holdout_collection_allowed": False,
        **_v7_0_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v8_3_batch_target_free_manifest.json"
    _write_json(manifest_path, manifest)
    return _run_result(
        run_dir=run_dir,
        report=report,
        manifest=manifest,
        manifest_path=manifest_path,
        outputs=outputs,
    )


def _evaluate_historical_decisions(
    *,
    decisions: list[dict[str, Any]],
    rows_by_market: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidate_selected: list[dict[str, Any]] = []
    baseline_selected: list[dict[str, Any]] = []
    candidate_by_market: dict[str, float] = {}
    baseline_by_market: dict[str, float] = {}
    for decision in decisions:
        row = rows_by_market[str(decision["market_id"])]
        source = str(decision["selection_source"])
        if source == "v8_1_primary":
            candidate_target = float(
                row["selected_target_after_cost_net_pnl_per_contract"]
            )
        elif source == "v6_7_non_risk_abstention_fallback":
            candidate_target = float(
                row["baseline_target_after_cost_net_pnl_per_contract"]
            )
        else:
            candidate_target = 0.0
        baseline_target = (
            float(row["baseline_target_after_cost_net_pnl_per_contract"])
            if row["baseline_execution_guard_order_allowed"] is True
            else 0.0
        )
        candidate_pnl = candidate_target * 0.2
        baseline_pnl = baseline_target * 0.2
        market_id = str(decision["market_id"])
        candidate_by_market[market_id] = candidate_pnl
        baseline_by_market[market_id] = baseline_pnl
        if decision["execution_guard_order_allowed"] is True:
            candidate_selected.append(
                v82._evaluation_row(
                    decision=decision,
                    target=candidate_target,
                    pnl=candidate_pnl,
                    source=source,
                )
            )
        if row["baseline_execution_guard_order_allowed"] is True:
            baseline_selected.append(
                {
                    "market_id": market_id,
                    "action": row["baseline_action"],
                    "side": row["baseline_side"],
                    "target_after_cost_net_pnl_per_contract": baseline_target,
                    "fixed_position_size": 0.2,
                    "after_cost_net_pnl_at_frozen_size": baseline_pnl,
                    "target_used_as_decision_time_input": False,
                    "target_opened_only_after_overlay_decision_freeze": True,
                }
            )
    candidate_total = sum(candidate_by_market.values())
    baseline_total = sum(baseline_by_market.values())
    candidate_largest = max(candidate_by_market.values(), default=0.0)
    baseline_largest = max(baseline_by_market.values(), default=0.0)
    candidate_lwr = candidate_total - max(candidate_largest, 0.0)
    baseline_lwr = baseline_total - max(baseline_largest, 0.0)
    total_delta = candidate_total - baseline_total
    lwr_delta = candidate_lwr - baseline_lwr
    support_delta = len(candidate_selected) - len(baseline_selected)
    checks = {
        "candidate_total_pnl_noninferior_to_v6_7": total_delta >= 0.0,
        "candidate_largest_winner_removed_pnl_noninferior_to_v6_7": (
            lwr_delta >= 0.0
        ),
        "candidate_guard_accepted_support_not_below_v6_7": support_delta >= 0,
        "decisions_frozen_before_historical_target_access": all(
            row["target_or_outcome_used_for_selection"] is False
            for row in decisions
        ),
        "risk_blocker_bypass_absent": all(
            row["explicit_execution_risk_blocker_bypass_used"] is False
            for row in decisions
        ),
        "source_scores_unchanged": all(
            row["source_score_mutated"] is False for row in decisions
        ),
    }
    reason_map = {
        "candidate_total_pnl_noninferior_to_v6_7": (
            "historical_total_after_cost_pnl_inferior_to_v6_7"
        ),
        "candidate_largest_winner_removed_pnl_noninferior_to_v6_7": (
            "historical_largest_winner_removed_pnl_inferior_to_v6_7"
        ),
        "candidate_guard_accepted_support_not_below_v6_7": (
            "historical_guard_accepted_support_below_v6_7"
        ),
        "decisions_frozen_before_historical_target_access": (
            "historical_target_accessed_before_decision_freeze"
        ),
        "risk_blocker_bypass_absent": "execution_risk_blocker_bypass_detected",
        "source_scores_unchanged": "source_score_mutation_detected",
    }
    blockers = [
        reason_map[name] for name, passed in checks.items() if not passed
    ]
    report = {
        "schema_version": HISTORICAL_REPORT_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "historical_market_count": len(decisions),
        "candidate_guard_accepted_market_count": len(candidate_selected),
        "v6_7_guard_accepted_market_count": len(baseline_selected),
        "candidate_minus_v6_7_guard_accepted_market_count": support_delta,
        "candidate_total_after_cost_net_pnl_at_frozen_size": candidate_total,
        "v6_7_total_after_cost_net_pnl_at_frozen_size": baseline_total,
        "candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size": (
            total_delta
        ),
        "candidate_largest_winner_after_cost_net_pnl_at_frozen_size": (
            candidate_largest
        ),
        "v6_7_largest_winner_after_cost_net_pnl_at_frozen_size": baseline_largest,
        "candidate_largest_winner_removed_after_cost_net_pnl_at_frozen_size": (
            candidate_lwr
        ),
        "v6_7_largest_winner_removed_after_cost_net_pnl_at_frozen_size": (
            baseline_lwr
        ),
        "candidate_minus_v6_7_largest_winner_removed_after_cost_net_pnl_at_frozen_size": (
            lwr_delta
        ),
        "selection_source_distribution": _distribution(
            decisions, "selection_source"
        ),
        "selected_action_distribution": _distribution(
            decisions, "selected_action"
        ),
        "selected_side_distribution_diagnostic": _distribution(
            [
                row
                for row in decisions
                if row["selected_side"] != "NONE"
            ],
            "selected_side",
        ),
        "side_quota_enabled": False,
        "historical_targets_opened_only_after_overlay_decision_freeze": True,
        "issue246_outcomes_opened": False,
        "checks": checks,
        "historical_noninferiority_gate_passed": not blockers,
        "historical_gate_blocking_reason_codes": blockers,
        "target_free_canary_allowed": not blockers,
        "future_holdout_collection_allowed": False,
        **_v7_0_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return {
        "candidate_selected_rows": candidate_selected,
        "baseline_selected_rows": baseline_selected,
        "report": report,
    }


def _future_candidate_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "market_id": row["market_id"],
        "decision_ts": row.get("decision_ts"),
        "selected_action": row["selected_action"],
        "selected_side": row["selected_side"],
        "execution_guard_order_allowed": row["execution_guard_order_allowed"],
        "execution_blocking_reason_codes": row[
            "execution_blocking_reason_codes"
        ],
        "rank_abstention_passed": row.get("rank_abstention_passed"),
        "point_selected_action": row.get("point_selected_action"),
    }


def _future_baseline_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "market_id": row["market_id"],
        "decision_ts": row.get("decision_ts"),
        "selected_action": row["selected_action"],
        "selected_side": row["selected_side"],
        "execution_guard_order_allowed": row["execution_guard_order_allowed"],
        "execution_blocking_reason_codes": row[
            "execution_blocking_reason_codes"
        ],
        "rank_abstention_passed": None,
        "point_selected_action": row["selected_action"],
    }


def _baseline_guard_rows(
    market_ids: list[str],
    *,
    baseline_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    v6_7_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline_by_market = {str(row["market_id"]): row for row in baseline_rows}
    source_by_key = {_action_key(row): row for row in action_rows}
    rows: list[dict[str, Any]] = []
    for market_id in market_ids:
        baseline = baseline_by_market.get(market_id)
        source = source_by_key.get(_action_key(baseline)) if baseline else None
        reasons = (
            ["v6_7_no_positive_guard_compatible_action"]
            if baseline is None
            else ["selected_action_source_row_missing"]
            if source is None
            else _microstructure_blocking_reasons(
                source, guard=v6_7_profile["hard_execution_safety"]
            )
        )
        action = str(baseline["action"]) if baseline else "NO_TRADE"
        side = (
            str(baseline["side"])
            if baseline and baseline.get("side")
            else "UP"
            if "UP" in action
            else "DOWN"
            if "DOWN" in action
            else "NONE"
        )
        rows.append(
            {
                "market_id": market_id,
                "decision_ts": int(source.get("decision_ts") or 0)
                if source
                else 0,
                "selected_action": action,
                "selected_side": side,
                "execution_guard_order_allowed": source is not None
                and not reasons,
                "execution_blocking_reason_codes": reasons,
                "labels_outcomes_resolution_or_pnl_opened": False,
                "source_score_mutated": False,
            }
        )
    return rows


def _no_trade_reasons(
    *,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    policy_abstention_only: bool,
) -> list[str]:
    reasons = list(candidate["execution_blocking_reason_codes"])
    if not policy_abstention_only:
        reasons.append("v8_1_not_policy_level_non_risk_abstention")
    if baseline["execution_guard_order_allowed"] is False:
        reasons.extend(
            f"v6_7_{code}"
            for code in baseline["execution_blocking_reason_codes"]
        )
        reasons.append("v6_7_independent_full_guard_failed")
    if baseline["selected_action"] not in v82.TRADE_ACTIONS:
        reasons.append("v6_7_trade_action_unavailable")
    return sorted(set(reasons or {"non_risk_abstention_overlay_no_trade"}))


def _verified_descriptor_path(
    descriptor: dict[str, Any],
    *,
    label: str,
) -> Path:
    path = Path(descriptor["path"])
    _verify_pin(path, descriptor["sha256"], label)
    return path


def _distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[key]) for row in rows).items()))


def _one_row_per_market(
    rows: list[dict[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        market_id = str(row.get("market_id") or "")
        if not market_id or market_id in output:
            raise ValueError(f"#249 {label} market identity missing or duplicated")
        output[market_id] = row
    return output


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
            or side not in {"UP", "DOWN"}
            or action
            not in {
                "BUY_UP_SELL_BEFORE_CLOSE",
                "BUY_DOWN_SELL_BEFORE_CLOSE",
            }
        ):
            raise ValueError("#249 runtime target identity invalid")
        seen.add(market_id)
        value = float(row["runtime_policy_after_cost_net_pnl_at_frozen_size"])
        if not math.isfinite(value):
            raise ValueError("#249 runtime target PnL is non-finite")


def _expected_safety() -> dict[str, bool]:
    safety = _v7_0_blocked_safety_fields()
    safety["paper_only"] = True
    return safety


def _write_historical_outputs(
    *,
    config: NonRiskAbstentionFallbackV83HistoricalConfig,
    profile_path: Path,
    input_manifest_path: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    run_dir = _fresh_run_dir(
        output_dir=config.output_dir,
        run_id=config.run_id,
        overwrite_existing=config.overwrite_existing,
    )
    outputs = {
        "report": run_dir / "v8_3_historical_noninferiority_report.json",
        "report_markdown": (
            run_dir / "v8_3_historical_noninferiority_report.md"
        ),
        "decision_rows": run_dir / "v8_3_historical_frozen_decisions.jsonl",
        "candidate_selected_rows": (
            run_dir / "v8_3_historical_candidate_selected_rows.jsonl"
        ),
        "v6_7_baseline_selected_rows": (
            run_dir / "v8_3_historical_v6_7_selected_rows.jsonl"
        ),
    }
    _write_json(outputs["report"], result["report"])
    _write_text(outputs["report_markdown"], _markdown(result["report"]))
    _write_jsonl(outputs["decision_rows"], result["decisions"])
    _write_jsonl(
        outputs["candidate_selected_rows"],
        result["candidate_selected_rows"],
    )
    _write_jsonl(
        outputs["v6_7_baseline_selected_rows"],
        result["baseline_selected_rows"],
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "stage": "historical_noninferiority",
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        "profile": _descriptor(profile_path),
        "v8_1_historical_manifest": _descriptor(input_manifest_path),
        **{name: _descriptor(path) for name, path in outputs.items()},
        "historical_noninferiority_gate_passed": result["report"][
            "historical_noninferiority_gate_passed"
        ],
        "historical_gate_blocking_reason_codes": result["report"][
            "historical_gate_blocking_reason_codes"
        ],
        "target_free_canary_allowed": result["report"][
            "target_free_canary_allowed"
        ],
        "future_holdout_collection_allowed": False,
        "issue246_outcomes_opened": False,
        **_v7_0_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v8_3_historical_noninferiority_manifest.json"
    _write_json(manifest_path, manifest)
    return _run_result(
        run_dir=run_dir,
        report=result["report"],
        manifest=manifest,
        manifest_path=manifest_path,
        outputs=outputs,
    )


def _write_canary_outputs(
    *,
    config: NonRiskAbstentionFallbackV83CanaryConfig,
    profile_path: Path,
    historical_manifest_path: Path,
    issue246_manifest_path: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    run_dir = _fresh_run_dir(
        output_dir=config.output_dir,
        run_id=config.run_id,
        overwrite_existing=config.overwrite_existing,
    )
    outputs = {
        "report": run_dir / "v8_3_target_free_canary_report.json",
        "report_markdown": run_dir / "v8_3_target_free_canary_report.md",
        "decision_rows": run_dir / "v8_3_target_free_canary_decisions.jsonl",
    }
    _write_json(outputs["report"], result["report"])
    _write_text(outputs["report_markdown"], _markdown(result["report"]))
    _write_jsonl(outputs["decision_rows"], result["decisions"])
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "stage": "target_free_canary",
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        "profile": _descriptor(profile_path),
        "historical_gate_manifest": _descriptor(historical_manifest_path),
        "issue246_target_free_manifest": _descriptor(issue246_manifest_path),
        **{name: _descriptor(path) for name, path in outputs.items()},
        "target_free_canary_passed": result["report"][
            "target_free_canary_passed"
        ],
        "target_free_canary_blocking_reason_codes": result["report"][
            "target_free_canary_blocking_reason_codes"
        ],
        "new_future_holdout_collection_allowed": result["report"][
            "new_future_holdout_collection_allowed"
        ],
        "issue246_outcomes_opened": False,
        **_v7_0_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v8_3_target_free_canary_manifest.json"
    _write_json(manifest_path, manifest)
    return _run_result(
        run_dir=run_dir,
        report=result["report"],
        manifest=manifest,
        manifest_path=manifest_path,
        outputs=outputs,
    )


def _fresh_run_dir(
    *,
    output_dir: str,
    run_id: str,
    overwrite_existing: bool,
) -> Path:
    run_dir = Path(output_dir).resolve() / run_id
    if run_dir.exists():
        if not overwrite_existing:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    return run_dir


def _run_result(
    *,
    run_dir: Path,
    report: dict[str, Any],
    manifest: dict[str, Any],
    manifest_path: Path,
    outputs: dict[str, Path],
) -> dict[str, Any]:
    return {
        "run_dir": run_dir,
        "report": report,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "outputs": outputs,
    }


def _markdown(report: dict[str, Any]) -> str:
    if "batch_target_free_diagnostic_passed" in report:
        lines = [
            "# Execution Layer v2 v8.3 Post-Seal Batch Diagnostic",
            "",
            f"- markets: `{report['batch_market_count']}`",
            f"- guard accepted: `{report['guard_accepted_market_count']}`",
            f"- zero support: `{str(report['zero_guard_accepted_support']).lower()}`",
            f"- selection sources: `{report['selection_source_distribution']}`",
            f"- diagnostic passed: `{str(report['batch_target_free_diagnostic_passed']).lower()}`",
        ]
    elif "historical_noninferiority_gate_passed" in report:
        lines = [
            "# Execution Layer v2 v8.3 Historical Non-Inferiority",
            "",
            f"- candidate support: `{report['candidate_guard_accepted_market_count']}`",
            f"- v6.7 support: `{report['v6_7_guard_accepted_market_count']}`",
            f"- candidate PnL: `{report['candidate_total_after_cost_net_pnl_at_frozen_size']}`",
            f"- v6.7 PnL: `{report['v6_7_total_after_cost_net_pnl_at_frozen_size']}`",
            f"- total delta: `{report['candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size']}`",
            f"- LWR delta: `{report['candidate_minus_v6_7_largest_winner_removed_after_cost_net_pnl_at_frozen_size']}`",
            f"- gate passed: `{str(report['historical_noninferiority_gate_passed']).lower()}`",
        ]
    else:
        lines = [
            "# Execution Layer v2 v8.3 Target-Free Canary",
            "",
            f"- markets: `{report['market_count']}`",
            f"- guard accepted: `{report['guard_accepted_market_count']}`",
            f"- selection sources: `{report['selection_source_distribution']}`",
            f"- canary passed: `{str(report['target_free_canary_passed']).lower()}`",
        ]
    return "\n".join(
        [
            *lines,
            f"- blockers: `{report.get('historical_gate_blocking_reason_codes', report.get('target_free_canary_blocking_reason_codes'))}`",
            "- issue #246 outcomes opened: `false`",
            "- paper/live/write/wallet/capital remain blocked.",
            "",
        ]
    )


__all__ = [
    "FUTURE_EXACT_MARKET_COUNT",
    "FUTURE_MINIMUM_GUARD_ACCEPTED_MARKET_COUNT",
    "FUTURE_SCAN_CAP",
    "FUTURE_SCHEMA_PREFIX",
    "NonRiskAbstentionFallbackV83CanaryConfig",
    "NonRiskAbstentionFallbackV83HistoricalConfig",
    "NonRiskAbstentionFallbackV83BatchConfig",
    "build_non_risk_abstention_fallback_v8_3_canary",
    "build_non_risk_abstention_fallback_v8_3_future_pnl_gate",
    "build_non_risk_abstention_fallback_v8_3_historical",
    "build_non_risk_abstention_fallback_v8_3_target_free_freeze_report",
    "run_non_risk_abstention_fallback_v8_3_canary",
    "run_non_risk_abstention_fallback_v8_3_batch_diagnostic",
    "run_non_risk_abstention_fallback_v8_3_historical_gate",
    "select_non_risk_abstention_fallback_v8_3_decision",
    "select_non_risk_abstention_fallback_v8_3_future_window",
    "validate_non_risk_abstention_fallback_v8_3_profile",
    "validate_non_risk_abstention_fallback_v8_3_future_plan",
]
