"""Regime-emergent calibration and execution-PnL gates for #229 v6.8."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_direct_advantage_estimand_audit import (
    _market_bootstrap_interval,
)
from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7 import (
    SBC_ACTIONS,
    SIDES,
)
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
    _blocked_safety_fields,
    _require_sha256,
)

CANDIDATE_NAME = "regime_emergent_side_composition_v6_8"
PROFILE_SCHEMA_VERSION = "bigan-v8-regime-emergent-pnl-v6-8-evaluation-profile-v1"
CALIBRATION_ARTIFACT_SCHEMA_VERSION = (
    "bigan-v8-regime-emergent-pnl-v6-8-pooled-residual-calibration-artifact-v1"
)
CONFIRMATORY_GATE_SCHEMA_VERSION = (
    "bigan-v8-regime-emergent-pnl-v6-8-confirmatory-gate-report-v1"
)
FROZEN_LINEAGE = {
    "parent_v6_7_prediction_freeze_manifest_sha256": (
        "1ae8a95700fc2b4808483becd3893ab2a77bc666261b3e3cfd623a4db26db431"
    ),
    "parent_v6_7_decision_freeze_sha256": (
        "f6010116541bff3b530672d036f36ba4201612763f2411f1f0ae4de481769735"
    ),
    "parent_v6_7_selected_decisions_sha256": (
        "9d3ee6913122516782c56ff7d85963ac3824ad6353652d33204eaae5ddc26edb"
    ),
    "parent_v6_7_selected_index_rows_sha256": (
        "80cfa188086c14f6e4c5e136ed5297daadfdfffd3853a81d64dec22cc42f4fc1"
    ),
    "collector_index_sha256": (
        "eeec98814f9f5cc95a7dd99c886e7f17b8d41bb31c6887ad0e917011c927b6a2"
    ),
    "candidate_freeze_manifest_sha256": (
        "2c9d0bb52ba2e59f960648845c8c9a4a574dd402684f3505d146ca4ca6e12493"
    ),
    "v6_7_evaluation_profile_sha256": (
        "900dba0b3d1e280271ff2489e0d0320f1eca150787bf2be30b8b751a3a993c3e"
    ),
    "runtime_policy_profile_sha256": (
        "1306f6b6f7a6c1216b23413352ff66f4061ec62a9751b0de51eded256ca51264"
    ),
}
FORBIDDEN_TARGET_FIELDS = {
    "resolved_outcome",
    "settlement_pnl",
    "runtime_policy_after_cost_net_pnl_per_contract",
    "runtime_policy_after_cost_net_pnl_at_frozen_size",
    "future_return",
    "label",
    "oracle_action",
}


def validate_regime_emergent_pnl_v6_8_profile(profile: dict[str, Any]) -> None:
    """Reject score, side-quota, calibration, PnL-gate, or lineage drift."""

    calibration = dict(profile.get("fresh_calibration") or {})
    confirmatory = dict(profile.get("future_confirmatory") or {})
    checks = {
        "schema": profile.get("schema_version") == PROFILE_SCHEMA_VERSION,
        "issue": profile.get("issue_number") == 229,
        "candidate": profile.get("candidate_name") == CANDIDATE_NAME,
        "frozen": profile.get("frozen") is True,
        "lineage": profile.get("lineage") == FROZEN_LINEAGE,
        "sealed_window": profile.get("sealed_calibration_window")
        == {
            "quality_valid_market_count": 60,
            "attempted_index_row_count": 66,
            "selection_order": "append_only_chronological_earliest_quality_valid",
            "one_selected_decision_per_market": True,
            "selected_side_composition": {"DOWN": 45, "UP": 15},
            "side_composition_is_regime_emergent": True,
            "side_quota_enforced": False,
            "side_count_hard_gate_enabled": False,
            "labels_outcomes_resolution_or_pnl_opened_before_profile_freeze": False,
        },
        "scoring": profile.get("candidate_scoring")
        == {
            "base_score_field": "v6_7_base_score",
            "base_score_source": "frozen_v6_2_market_clustered_mean_ev_lcb",
            "base_entry_threshold": 0.0,
            "threshold_operator": "strictly_greater_than",
            "source_score_mutation_allowed": False,
            "decision_reselection_allowed": False,
            "side_balancing_or_quota_allowed": False,
            "execution_guard_threshold_mutation_allowed": False,
        },
        "calibration": calibration
        == {
            "method": (
                "pooled_market_bootstrap_upper_confidence_bound_of_base_score_"
                "minus_runtime_after_cost_pnl"
            ),
            "target": "runtime_policy_after_cost_net_pnl_per_contract",
            "target_used_as_decision_time_input": False,
            "confidence_level": 0.95,
            "bootstrap_resample_count": 5000,
            "bootstrap_seed": 2292026,
            "minimum_selected_unique_market_count_total": 60,
            "minimum_selected_unique_market_count_per_side": None,
            "minimum_positive_calibrated_lcb_unique_market_count_per_side": None,
            "maximum_absolute_pooled_residual_upper_confidence_bound": 2.0,
            "calibrated_score": (
                "base_score_minus_pooled_residual_upper_confidence_bound"
            ),
            "calibrated_entry_threshold": 0.0,
            "threshold_search_enabled": False,
            "feature_or_model_search_enabled": False,
            "result_selected_rerun_allowed": False,
        },
        "confirmatory": confirmatory
        == {
            "quality_valid_market_count": 120,
            "maximum_attempted_market_count": 180,
            "strictly_after_calibration": True,
            "market_identity_disjointness_required": True,
            "minimum_guard_accepted_unique_market_count_total": 40,
            "minimum_supported_side_unique_market_count": None,
            "required_supported_sides": [],
            "accepted_total_after_cost_pnl_minimum_exclusive": 0.0,
            "candidate_minus_matched_legacy_pnl_minimum_exclusive": 0.0,
            "candidate_minus_matched_legacy_bootstrap_lcb_minimum_exclusive": 0.0,
            "largest_winner_removed_after_cost_pnl_minimum_exclusive": 0.0,
            "bootstrap_unit": "market_id",
            "bootstrap_confidence_level": 0.95,
            "bootstrap_resample_count": 5000,
            "bootstrap_seed": 2292027,
            "side_count_and_side_pnl_diagnostic_only": True,
            "action_and_action_family_pnl_diagnostic_only": True,
            "single_use_confirmatory": True,
            "result_selected_extension_allowed": False,
            "result_selected_rerun_allowed": False,
        },
        "access": profile.get("access_sequence")
        == {
            "sealed_target_free_decision_adoption_first": True,
            "official_read_only_calibration_settlement_on_quarantine_copies_second": True,
            "pooled_calibration_artifact_freeze_third": True,
            "strictly_later_target_free_confirmatory_decision_freeze_fourth": True,
            "official_read_only_confirmatory_settlement_on_quarantine_copies_fifth": True,
            "execution_pnl_hard_gate_last": True,
            "outcomes_used_for_model_feature_threshold_cost_sizing_guard_or_side_quota_tuning": False,
        },
        "safety": profile.get("safety") == _blocked_safety_fields(),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("#229 evaluation profile invalid: " + ", ".join(blockers))


def build_regime_emergent_target_free_support(
    rows: list[dict[str, Any]],
    *,
    exact_window_market_count: int,
    required_total_market_count: int,
    score_field: str,
    expected_window_market_count: int | None = None,
) -> dict[str, Any]:
    """Validate total target-free support without imposing a side composition."""

    side_count = Counter(str(row.get("side") or "") for row in rows)
    expected_window = (
        required_total_market_count
        if expected_window_market_count is None
        else expected_window_market_count
    )
    checks = {
        "exact_window_market_count": exact_window_market_count
        == expected_window,
        "one_selected_row_per_market": len(
            {str(row.get("market_id") or "") for row in rows}
        )
        == len(rows),
        "minimum_selected_market_support_total": len(rows)
        >= required_total_market_count,
        "selected_side_values_valid": all(
            str(row.get("side") or "") in SIDES for row in rows
        ),
        "all_scores_positive": all(float(row[score_field]) > 0.0 for row in rows),
        "feature_causality": all(
            int(row["max_input_ts"]) <= int(row["decision_ts"]) for row in rows
        ),
        "targets_sealed": all(
            row.get("labels_outcomes_resolution_or_pnl_opened") is False
            and not FORBIDDEN_TARGET_FIELDS.intersection(row)
            for row in rows
        ),
        "source_scores_unchanged": all(
            row.get("source_score_mutated") is False for row in rows
        ),
    }
    blockers = [f"{name}_gate_failed" for name, passed in checks.items() if not passed]
    return {
        "selected_market_count": len(rows),
        "count_by_side": {side: side_count[side] for side in SIDES},
        "minimum_total_required": required_total_market_count,
        "expected_window_market_count": expected_window,
        "minimum_per_side_required": None,
        "side_count_hard_gate_enabled": False,
        "side_composition_is_regime_emergent": True,
        "checks": checks,
        "target_free_support_gate_passed": not blockers,
        "blocking_reason_codes": blockers,
    }


def build_v6_8_pooled_residual_calibration(
    joined_rows: list[dict[str, Any]],
    *,
    profile: dict[str, Any],
    decision_freeze_descriptor: dict[str, Any],
    settled_index_descriptor: dict[str, Any],
    runtime_policy_profile_descriptor: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fit the single preregistered pooled residual UCB on 60 markets."""

    validate_regime_emergent_pnl_v6_8_profile(profile)
    config = dict(profile["fresh_calibration"])
    _validate_calibration_rows(joined_rows)
    residuals = [
        float(row["v6_7_base_score"])
        - float(row["runtime_policy_after_cost_net_pnl_per_contract"])
        for row in joined_rows
    ]
    pooled = _market_bootstrap_interval(
        residuals,
        resample_count=int(config["bootstrap_resample_count"]),
        confidence_level=float(config["confidence_level"]),
        seed=int(config["bootstrap_seed"]),
    )
    correction = float(pooled["upper_confidence_bound"])
    calibrated_rows = []
    for row in joined_rows:
        calibrated = {
            **row,
            "pooled_residual_upper_confidence_bound": correction,
            "v6_8_calibrated_runtime_pnl_lcb": float(row["v6_7_base_score"])
            - correction,
            "calibration_target_used_as_decision_time_input": False,
            "side_quota_applied": False,
        }
        calibrated["calibrated_row_id"] = canonical_json_sha256(calibrated)
        calibrated_rows.append(calibrated)

    count_by_side = Counter(str(row["side"]) for row in joined_rows)
    positive_by_side = {
        side: sum(
            row["side"] == side
            and float(row["v6_8_calibrated_runtime_pnl_lcb"]) > 0.0
            for row in calibrated_rows
        )
        for side in SIDES
    }
    targets = np.asarray(
        [
            float(row["runtime_policy_after_cost_net_pnl_per_contract"])
            for row in joined_rows
        ]
    )
    base_scores = np.asarray([float(row["v6_7_base_score"]) for row in joined_rows])
    mean_corrected = np.asarray(
        [float(row["v6_7_base_score"]) - float(pooled["point_estimate"]) for row in joined_rows]
    )
    conservative = np.asarray(
        [float(row["v6_8_calibrated_runtime_pnl_lcb"]) for row in calibrated_rows]
    )
    maximum_correction = float(
        config["maximum_absolute_pooled_residual_upper_confidence_bound"]
    )
    checks = {
        "exact_fresh_calibration_market_count": len(joined_rows)
        == int(config["minimum_selected_unique_market_count_total"]),
        "one_row_per_market": len({str(row["market_id"]) for row in joined_rows})
        == len(joined_rows),
        "pooled_residual_ucb_finite_and_bounded": math.isfinite(correction)
        and abs(correction) <= maximum_correction,
        "feature_timestamp_causality": all(
            int(row["max_input_ts"]) <= int(row["decision_ts"])
            for row in joined_rows
        ),
        "targets_post_freeze_only": all(
            row.get("target_available_only_post_exit_or_official_resolution") is True
            and row.get("target_used_as_decision_time_input") is False
            for row in joined_rows
        ),
        "side_quota_disabled": config[
            "minimum_selected_unique_market_count_per_side"
        ]
        is None
        and config["minimum_positive_calibrated_lcb_unique_market_count_per_side"]
        is None,
        "no_threshold_search": config["threshold_search_enabled"] is False,
        "no_feature_or_model_search": config["feature_or_model_search_enabled"]
        is False,
        "no_result_selected_rerun": config["result_selected_rerun_allowed"]
        is False,
    }
    reasons = [f"{name}_gate_failed" for name, passed in checks.items() if not passed]
    artifact = {
        "schema_version": CALIBRATION_ARTIFACT_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "method": config["method"],
        "decision_freeze": decision_freeze_descriptor,
        "settled_corpus_index": settled_index_descriptor,
        "runtime_policy_profile": runtime_policy_profile_descriptor,
        "selected_market_count": len(joined_rows),
        "selected_market_count_by_side_diagnostic": {
            side: count_by_side[side] for side in SIDES
        },
        "pooled_residual_calibration": pooled,
        "positive_calibrated_lcb_unique_market_count_by_side_diagnostic": (
            positive_by_side
        ),
        "error_metrics": {
            "frozen_v6_7_base_score": _metrics(targets, base_scores),
            "pooled_mean_residual_corrected_report_only": _metrics(
                targets, mean_corrected
            ),
            "pooled_residual_ucb_lcb_report_only": _metrics(targets, conservative),
        },
        "calibrated_entry_threshold": 0.0,
        "calibrated_threshold_operator": "strictly_greater_than",
        "side_composition_is_regime_emergent": True,
        "side_count_hard_gate_enabled": False,
        "calibration_gate_checks": checks,
        "calibration_gate_passed": not reasons,
        "calibration_gate_blocking_reason_codes": reasons,
        "calibration_outcomes_used_for_pooled_residual_only": True,
        "calibration_outcomes_used_as_model_or_decision_inputs": False,
        "calibration_outcomes_used_for_threshold_guard_or_side_quota_search": False,
        "strictly_later_single_use_confirmatory_required": True,
        **_blocked_safety_fields(),
    }
    artifact["calibration_artifact_id"] = canonical_json_sha256(artifact)
    return artifact, calibrated_rows


def apply_v6_8_pooled_residual_calibration(
    target_free_rows: list[dict[str, Any]],
    *,
    calibration_artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply one pooled correction without inspecting future targets."""

    if (
        calibration_artifact.get("schema_version")
        != CALIBRATION_ARTIFACT_SCHEMA_VERSION
        or calibration_artifact.get("calibration_gate_passed") is not True
        or calibration_artifact.get("calibration_gate_blocking_reason_codes") != []
        or calibration_artifact.get("side_count_hard_gate_enabled") is not False
    ):
        raise ValueError("#229 calibration artifact is not confirmatory eligible")
    pooled = dict(calibration_artifact.get("pooled_residual_calibration") or {})
    correction = float(pooled.get("upper_confidence_bound"))
    if not math.isfinite(correction):
        raise ValueError("#229 pooled residual correction is non-finite")
    output = []
    seen_markets: set[str] = set()
    for row in target_free_rows:
        if FORBIDDEN_TARGET_FIELDS.intersection(row):
            raise ValueError("#229 confirmatory row contains target fields")
        market_id = str(row.get("market_id") or "")
        side = str(row.get("side") or "")
        action = str(row.get("action") or "")
        if (
            not market_id
            or market_id in seen_markets
            or side not in SIDES
            or action not in SBC_ACTIONS
        ):
            raise ValueError("#229 confirmatory selected-row identity invalid")
        if int(row["max_input_ts"]) > int(row["decision_ts"]):
            raise ValueError("#229 confirmatory feature causality violation")
        calibrated_score = float(row["v6_7_base_score"]) - correction
        if calibrated_score <= 0.0:
            continue
        updated = {
            **row,
            "v6_8_calibrated_runtime_pnl_lcb": calibrated_score,
            "pooled_residual_upper_confidence_bound": correction,
            "calibration_applied_without_future_target_access": True,
            "side_quota_applied": False,
            "source_score_mutated": False,
            "labels_outcomes_resolution_or_pnl_opened": False,
        }
        updated["confirmatory_selected_row_id"] = canonical_json_sha256(updated)
        output.append(updated)
        seen_markets.add(market_id)
    return sorted(output, key=lambda row: (int(row["decision_ts"]), row["market_id"]))


def build_v6_8_regime_emergent_confirmatory_gate(
    candidate_rows: list[dict[str, Any]],
    *,
    matched_legacy_rows: list[dict[str, Any]],
    evaluation_market_ids: list[str],
    profile: dict[str, Any],
    decision_freeze_sha256: str,
) -> dict[str, Any]:
    """Evaluate after-cost PnL without imposing a directional composition."""

    validate_regime_emergent_pnl_v6_8_profile(profile)
    _require_sha256(decision_freeze_sha256, name="decision_freeze_sha256")
    gates = dict(profile["future_confirmatory"])
    market_ids = list(dict.fromkeys(str(value) for value in evaluation_market_ids))
    if "" in market_ids or len(market_ids) != len(evaluation_market_ids):
        raise ValueError("#229 confirmatory evaluation market identity invalid")
    _validate_evaluation_rows(candidate_rows, market_ids=market_ids)
    _validate_evaluation_rows(matched_legacy_rows, market_ids=market_ids)

    candidate_by_market = dict.fromkeys(market_ids, 0.0)
    legacy_by_market = dict.fromkeys(market_ids, 0.0)
    for row in candidate_rows:
        candidate_by_market[str(row["market_id"])] += float(
            row["runtime_policy_after_cost_net_pnl_at_frozen_size"]
        )
    for row in matched_legacy_rows:
        legacy_by_market[str(row["market_id"])] += float(
            row["runtime_policy_after_cost_net_pnl_at_frozen_size"]
        )
    delta_by_market = {
        market_id: candidate_by_market[market_id] - legacy_by_market[market_id]
        for market_id in market_ids
    }
    bootstrap = _market_bootstrap_interval(
        list(delta_by_market.values()),
        resample_count=int(gates["bootstrap_resample_count"]),
        confidence_level=float(gates["bootstrap_confidence_level"]),
        seed=int(gates["bootstrap_seed"]),
    )
    by_side: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        by_side[str(row["side"])].append(row)
        by_action[str(row["action"])].append(row)
    side_metrics = {
        side: _group_metrics(rows, diagnostic_only=True)
        for side, rows in sorted(by_side.items())
    }
    action_metrics = {
        action: _group_metrics(rows, diagnostic_only=True)
        for action, rows in sorted(by_action.items())
    }
    candidate_total = float(sum(candidate_by_market.values()))
    legacy_total = float(sum(legacy_by_market.values()))
    delta_total = candidate_total - legacy_total
    largest_winner = max(candidate_by_market.values(), default=0.0)
    largest_winner_removed = candidate_total - max(largest_winner, 0.0)
    checks = {
        "minimum_guard_accepted_unique_market_support_total": len(candidate_rows)
        >= int(gates["minimum_guard_accepted_unique_market_count_total"]),
        "accepted_total_after_cost_pnl_positive": candidate_total
        > float(gates["accepted_total_after_cost_pnl_minimum_exclusive"]),
        "candidate_exceeds_matched_legacy": delta_total
        > float(gates["candidate_minus_matched_legacy_pnl_minimum_exclusive"]),
        "candidate_minus_legacy_bootstrap_lcb_positive": float(
            bootstrap["lower_confidence_bound"]
        )
        > float(
            gates[
                "candidate_minus_matched_legacy_bootstrap_lcb_minimum_exclusive"
            ]
        ),
        "largest_winner_removed_pnl_positive": largest_winner_removed
        > float(gates["largest_winner_removed_after_cost_pnl_minimum_exclusive"]),
        "settlement_causality_and_target_isolation": all(
            row.get("target_available_only_post_exit_or_official_resolution") is True
            and row.get("target_used_as_decision_time_input") is False
            and int(row["max_input_ts"]) <= int(row["decision_ts"])
            for row in candidate_rows + matched_legacy_rows
        ),
    }
    reason_map = {
        "minimum_guard_accepted_unique_market_support_total": (
            "insufficient_guard_accepted_unique_market_support_total"
        ),
        "accepted_total_after_cost_pnl_positive": (
            "accepted_total_after_cost_pnl_not_positive"
        ),
        "candidate_exceeds_matched_legacy": "candidate_does_not_exceed_matched_legacy",
        "candidate_minus_legacy_bootstrap_lcb_positive": (
            "candidate_minus_matched_legacy_bootstrap_lcb_not_positive"
        ),
        "largest_winner_removed_pnl_positive": (
            "largest_winner_removed_after_cost_pnl_not_positive"
        ),
        "settlement_causality_and_target_isolation": (
            "settlement_causality_or_target_isolation_failed"
        ),
    }
    blockers = [reason_map[name] for name, passed in checks.items() if not passed]
    report = {
        "schema_version": CONFIRMATORY_GATE_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "decision_freeze_sha256": decision_freeze_sha256,
        "evaluation_market_count": len(market_ids),
        "accepted_unique_market_count": len(candidate_rows),
        "accepted_side_distribution_diagnostic": dict(
            sorted(Counter(str(row["side"]) for row in candidate_rows).items())
        ),
        "accepted_side_metrics_diagnostic": side_metrics,
        "accepted_action_metrics_diagnostic": action_metrics,
        "accepted_action_family_metrics_diagnostic": {
            "SELL_BEFORE_CLOSE": _group_metrics(
                candidate_rows, diagnostic_only=True
            )
        },
        "candidate_after_cost_pnl": candidate_total,
        "matched_legacy_after_cost_pnl": legacy_total,
        "candidate_minus_matched_legacy_after_cost_pnl": delta_total,
        "candidate_minus_matched_legacy_market_bootstrap": bootstrap,
        "largest_winner_after_cost_pnl": largest_winner,
        "largest_winner_removed_after_cost_pnl": largest_winner_removed,
        "side_count_hard_gate_enabled": False,
        "side_pnl_hard_gate_enabled": False,
        "side_composition_is_regime_emergent": True,
        "execution_pnl_gate_checks": checks,
        "confirmatory_execution_pnl_gate_passed": not blockers,
        "confirmatory_execution_pnl_gate_blocking_reason_codes": blockers,
        "future_outcomes_used_for_model_threshold_cost_sizing_guard_or_side_quota_tuning": False,
        "single_use_confirmatory": True,
        "result_selected_rerun_allowed": False,
        "manual_approval_does_not_bypass_execution_pnl_gate": True,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def _validate_calibration_rows(rows: list[dict[str, Any]]) -> None:
    seen: set[tuple[str, int, str]] = set()
    for row in rows:
        key = (
            str(row.get("market_id") or ""),
            int(row.get("decision_ts") or 0),
            str(row.get("action") or ""),
        )
        if (
            not key[0]
            or key[1] <= 0
            or key[2] not in SBC_ACTIONS
            or str(row.get("side") or "") not in SIDES
            or key in seen
        ):
            raise ValueError("#229 calibration row identity invalid")
        seen.add(key)
        for field in (
            "v6_7_base_score",
            "runtime_policy_after_cost_net_pnl_per_contract",
        ):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"#229 calibration field non-finite: {field}")
        if int(row["max_input_ts"]) > int(row["decision_ts"]):
            raise ValueError("#229 calibration feature causality violation")
        if (
            row.get("target_available_only_post_exit_or_official_resolution") is not True
            or row.get("target_used_as_decision_time_input") is not False
        ):
            raise ValueError("#229 calibration target isolation invalid")


def _validate_evaluation_rows(
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
            raise ValueError("#229 confirmatory evaluation row identity invalid")
        seen.add(market_id)
        if not math.isfinite(
            float(row["runtime_policy_after_cost_net_pnl_at_frozen_size"])
        ):
            raise ValueError("#229 confirmatory PnL is non-finite")


def _group_metrics(
    rows: list[dict[str, Any]], *, diagnostic_only: bool
) -> dict[str, Any]:
    values = [
        float(row["runtime_policy_after_cost_net_pnl_at_frozen_size"])
        for row in rows
    ]
    return {
        "accepted_bet_count": len(rows),
        "accepted_unique_market_count": len(
            {str(row["market_id"]) for row in rows}
        ),
        "after_cost_pnl_sum": float(sum(values)),
        "after_cost_pnl_mean": float(np.mean(values)) if values else 0.0,
        "win_rate": (
            float(sum(value > 0.0 for value in values) / len(values))
            if values
            else 0.0
        ),
        "diagnostic_only": diagnostic_only,
    }


def _metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    errors = predictions - targets
    return {
        "mae": float(np.mean(np.abs(errors))),
        "mse": float(np.mean(np.square(errors))),
        "bias": float(np.mean(errors)),
    }


__all__ = [
    "CALIBRATION_ARTIFACT_SCHEMA_VERSION",
    "CONFIRMATORY_GATE_SCHEMA_VERSION",
    "PROFILE_SCHEMA_VERSION",
    "apply_v6_8_pooled_residual_calibration",
    "build_regime_emergent_target_free_support",
    "build_v6_8_pooled_residual_calibration",
    "build_v6_8_regime_emergent_confirmatory_gate",
    "validate_regime_emergent_pnl_v6_8_profile",
]
