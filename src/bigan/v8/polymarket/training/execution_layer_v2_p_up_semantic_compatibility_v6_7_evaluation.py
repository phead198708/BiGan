"""Preregistered calibration and side-only future gate for #227 v6.7."""

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
    CANDIDATE_NAME,
    SBC_ACTIONS,
    SIDES,
)
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
    _blocked_safety_fields,
    _require_sha256,
)

PROFILE_SCHEMA_VERSION = (
    "bigan-v8-p-up-semantic-execution-compatibility-v6-7-evaluation-profile-v1"
)
CALIBRATION_ARTIFACT_SCHEMA_VERSION = (
    "bigan-v8-p-up-semantic-execution-compatibility-v6-7-side-residual-"
    "calibration-artifact-v1"
)
CONFIRMATORY_GATE_SCHEMA_VERSION = (
    "bigan-v8-p-up-semantic-execution-compatibility-v6-7-side-only-"
    "confirmatory-gate-report-v1"
)
FROZEN_LINEAGE = {
    "candidate_freeze_manifest_sha256": (
        "2c9d0bb52ba2e59f960648845c8c9a4a574dd402684f3505d146ca4ca6e12493"
    ),
    "candidate_profile_sha256": (
        "cec55d243acd6bbf60a5e8474545b487086ddcd4d18073682ae7f2d4660d2248"
    ),
    "collection_plan_sha256": (
        "6d3dae149d12113f7735fedbbd67db39d1e082bcf8bb9ee35d06ab38277544a9"
    ),
    "collection_plan_correction_sha256": (
        "c3162eaa39917ae099c05a8aaf24ca37bc11ba51a3c3afdf60ecfa66f381daba"
    ),
    "collector_protocol_sha256": (
        "2343f8247b2c1441e694b2975bccec7ae2448db5e5a5c916c3a02def49d44843"
    ),
    "v6_2_candidate_manifest_sha256": (
        "b9441b04fb595a927cbf9af9311612b037c36fc8c623ac8a92b6f4cb8ece84b9"
    ),
    "v6_2_source_model_sha256": (
        "7e292852673fe2072017effc2d40fce000be81734f0c8c3d6950c02e957bcf0c"
    ),
    "v6_2_calibration_sha256": (
        "dc82ddebc51e95e46477894f2a0ba7bd8fa2f6845b22ced43402822b66b68e43"
    ),
    "feature_contract_sha256": (
        "a4819ad6beec8d72612aa25ef2af751c357e807d514dcf1d2c94b37eba07c959"
    ),
    "runtime_policy_profile_sha256": (
        "1306f6b6f7a6c1216b23413352ff66f4061ec62a9751b0de51eded256ca51264"
    ),
}


def validate_v6_7_evaluation_profile(profile: dict[str, Any]) -> None:
    """Reject any post-registration sizing, score, or hard-gate drift."""

    windows = dict(profile.get("collection_windows") or {})
    scoring = dict(profile.get("candidate_scoring") or {})
    calibration = dict(profile.get("fresh_calibration") or {})
    confirmatory = dict(profile.get("confirmatory_side_only_pnl_gate") or {})
    access = dict(profile.get("access_sequence") or {})
    checks = {
        "schema": profile.get("schema_version") == PROFILE_SCHEMA_VERSION,
        "issue": profile.get("issue_number") == 227,
        "candidate": profile.get("candidate_name") == CANDIDATE_NAME,
        "frozen": profile.get("frozen") is True,
        "lineage": profile.get("lineage") == FROZEN_LINEAGE,
        "windows": windows
        == {
            "future_collection_minimum_created_ts_exclusive": 1784591891531,
            "selection_order": "append_only_chronological_earliest_quality_valid",
            "fresh_calibration_quality_valid_market_count": 60,
            "fresh_calibration_maximum_attempted_market_count": 90,
            "confirmatory_quality_valid_market_count": 120,
            "confirmatory_maximum_attempted_market_count": 180,
            "confirmatory_strictly_after_calibration": True,
            "market_identity_disjointness_required": True,
            "result_selected_extension_allowed": False,
        },
        "scoring": scoring
        == {
            "base_score_field": "mean_ev_lower_confidence_bound",
            "base_score_source": "frozen_v6_2_market_clustered_mean_ev_lcb",
            "eligible_actions": sorted(SBC_ACTIONS),
            "base_entry_threshold": 0.0,
            "threshold_operator": "strictly_greater_than",
            "one_row_per_market": True,
            "p_up_side_alignment_filter_enabled": False,
            "p_up_action_disagreement_diagnostic_only": True,
            "source_score_mutation_allowed": False,
            "execution_guard_threshold_mutation_allowed": False,
        },
        "calibration": calibration
        == {
            "method": (
                "side_specific_market_bootstrap_upper_confidence_bound_of_"
                "base_score_minus_runtime_after_cost_pnl"
            ),
            "target": "runtime_policy_after_cost_net_pnl_per_contract",
            "target_used_as_decision_time_input": False,
            "confidence_level": 0.95,
            "bootstrap_resample_count": 5000,
            "bootstrap_seed": 2272026,
            "minimum_selected_unique_market_count_per_side": 20,
            "minimum_positive_calibrated_lcb_unique_market_count_per_side": 10,
            "maximum_absolute_side_residual_upper_confidence_bound": 2.0,
            "calibrated_score": (
                "base_score_minus_side_residual_upper_confidence_bound"
            ),
            "calibrated_entry_threshold": 0.0,
            "threshold_search_enabled": False,
            "feature_or_model_search_enabled": False,
            "result_selected_rerun_allowed": False,
        },
        "confirmatory": confirmatory
        == {
            "minimum_guard_accepted_unique_market_count": 40,
            "minimum_supported_side_unique_market_count": 20,
            "required_supported_sides": ["UP", "DOWN"],
            "accepted_total_after_cost_pnl_minimum_exclusive": 0.0,
            "supported_side_after_cost_pnl_minimum_exclusive": 0.0,
            "candidate_minus_matched_legacy_pnl_minimum_exclusive": 0.0,
            "candidate_minus_matched_legacy_bootstrap_lcb_minimum_exclusive": 0.0,
            "largest_winner_removed_after_cost_pnl_minimum_exclusive": 0.0,
            "bootstrap_unit": "market_id",
            "bootstrap_confidence_level": 0.95,
            "bootstrap_resample_count": 5000,
            "bootstrap_seed": 2272027,
            "pnl_hard_gate_aggregation": "selected_side_buy_up_buy_down_only",
            "action_and_action_family_pnl_diagnostic_only": True,
            "single_use_confirmatory": True,
            "result_selected_rerun_allowed": False,
        },
        "access": access
        == {
            "target_free_calibration_decision_freeze_first": True,
            "official_read_only_calibration_settlement_on_quarantine_copies_second": True,
            "calibration_artifact_freeze_third": True,
            "target_free_confirmatory_decision_freeze_fourth": True,
            "official_read_only_confirmatory_settlement_on_quarantine_copies_fifth": True,
            "single_side_only_pnl_gate_last": True,
            "labels_outcomes_resolution_or_pnl_opened_during_collection": False,
            "outcomes_used_for_model_feature_threshold_cost_sizing_or_guard_tuning": False,
        },
        "safety": profile.get("safety") == _blocked_safety_fields(),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("#227 evaluation profile invalid: " + ", ".join(blockers))


def build_v6_7_side_residual_calibration(
    joined_rows: list[dict[str, Any]],
    *,
    profile: dict[str, Any],
    decision_freeze_descriptor: dict[str, Any],
    settled_index_descriptor: dict[str, Any],
    runtime_policy_profile_descriptor: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fit only the preregistered side residual UCB on the fresh 60 markets."""

    validate_v6_7_evaluation_profile(profile)
    config = dict(profile["fresh_calibration"])
    _validate_calibration_rows(joined_rows)
    count_by_side = Counter(str(row["side"]) for row in joined_rows)
    side_calibration: dict[str, dict[str, float]] = {}
    for side in SIDES:
        residuals = [
            float(row["v6_7_base_score"])
            - float(row["runtime_policy_after_cost_net_pnl_per_contract"])
            for row in joined_rows
            if row["side"] == side
        ]
        side_calibration[side] = _market_bootstrap_interval(
            residuals,
            resample_count=int(config["bootstrap_resample_count"]),
            confidence_level=float(config["confidence_level"]),
            seed=int(config["bootstrap_seed"]),
        )

    calibrated_rows = []
    for row in joined_rows:
        side = str(row["side"])
        correction = float(side_calibration[side]["upper_confidence_bound"])
        calibrated = {
            **row,
            "side_residual_upper_confidence_bound": correction,
            "v6_7_calibrated_runtime_pnl_lcb": float(row["v6_7_base_score"])
            - correction,
            "calibration_target_used_as_decision_time_input": False,
        }
        calibrated["calibrated_row_id"] = canonical_json_sha256(calibrated)
        calibrated_rows.append(calibrated)

    positive_by_side = {
        side: sum(
            row["side"] == side
            and float(row["v6_7_calibrated_runtime_pnl_lcb"]) > 0.0
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
        [
                float(row["v6_7_base_score"])
                - float(side_calibration[str(row["side"])]["point_estimate"])
            for row in joined_rows
        ]
    )
    conservative = np.asarray(
        [float(row["v6_7_calibrated_runtime_pnl_lcb"]) for row in calibrated_rows]
    )
    minimum_side = int(config["minimum_selected_unique_market_count_per_side"])
    minimum_positive = int(
        config["minimum_positive_calibrated_lcb_unique_market_count_per_side"]
    )
    maximum_correction = float(
        config["maximum_absolute_side_residual_upper_confidence_bound"]
    )
    checks = {
        "exact_fresh_calibration_market_count": len(joined_rows) == 60,
        "one_row_per_market": len({str(row["market_id"]) for row in joined_rows})
        == len(joined_rows),
        "selected_side_support": all(
            count_by_side[side] >= minimum_side for side in SIDES
        ),
        "positive_calibrated_lcb_support": all(
            positive_by_side[side] >= minimum_positive for side in SIDES
        ),
        "side_residual_ucb_finite_and_bounded": all(
            math.isfinite(float(side_calibration[side]["upper_confidence_bound"]))
            and abs(float(side_calibration[side]["upper_confidence_bound"]))
            <= maximum_correction
            for side in SIDES
        ),
        "feature_timestamp_causality": all(
            int(row["max_input_ts"]) <= int(row["decision_ts"])
            for row in joined_rows
        ),
        "targets_post_freeze_only": all(
            row.get("target_available_only_post_exit_or_official_resolution") is True
            and row.get("target_used_as_decision_time_input") is False
            for row in joined_rows
        ),
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
        "selected_market_count_by_side": {side: count_by_side[side] for side in SIDES},
        "side_calibration": side_calibration,
        "positive_calibrated_lcb_unique_market_count_by_side": positive_by_side,
        "error_metrics": {
            "frozen_v6_7_base_score": _metrics(targets, base_scores),
            "side_mean_residual_corrected_report_only": _metrics(
                targets, mean_corrected
            ),
            "side_residual_ucb_lcb_report_only": _metrics(targets, conservative),
        },
        "calibrated_entry_threshold": 0.0,
        "calibrated_threshold_operator": "strictly_greater_than",
        "calibration_gate_checks": checks,
        "calibration_gate_passed": not reasons,
        "calibration_gate_blocking_reason_codes": reasons,
        "calibration_outcomes_used_for_side_residual_only": True,
        "calibration_outcomes_used_as_model_or_decision_inputs": False,
        "calibration_outcomes_used_for_threshold_or_guard_search": False,
        "strictly_later_single_use_confirmatory_required": True,
        **_blocked_safety_fields(),
    }
    artifact["calibration_artifact_id"] = canonical_json_sha256(artifact)
    return artifact, calibrated_rows


def apply_v6_7_side_residual_calibration(
    target_free_rows: list[dict[str, Any]],
    *,
    calibration_artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply the frozen side UCB without opening any future target."""

    if (
        calibration_artifact.get("schema_version")
        != CALIBRATION_ARTIFACT_SCHEMA_VERSION
        or calibration_artifact.get("calibration_gate_passed") is not True
        or calibration_artifact.get("calibration_gate_blocking_reason_codes") != []
    ):
        raise ValueError("#227 calibration artifact is not confirmatory eligible")
    side_calibration = dict(calibration_artifact.get("side_calibration") or {})
    if set(side_calibration) != set(SIDES):
        raise ValueError("#227 calibration artifact side coverage invalid")
    output = []
    seen_markets: set[str] = set()
    for row in target_free_rows:
        forbidden = {
            "resolved_outcome",
            "settlement_pnl",
            "runtime_policy_after_cost_net_pnl_per_contract",
            "runtime_policy_after_cost_net_pnl_at_frozen_size",
            "future_return",
            "label",
        }.intersection(row)
        if forbidden:
            raise ValueError("#227 confirmatory target-free row contains target fields")
        market_id = str(row.get("market_id") or "")
        side = str(row.get("side") or "")
        action = str(row.get("action") or "")
        if not market_id or market_id in seen_markets or side not in SIDES or action not in SBC_ACTIONS:
            raise ValueError("#227 confirmatory selected-row identity invalid")
        if int(row["max_input_ts"]) > int(row["decision_ts"]):
            raise ValueError("#227 confirmatory feature causality violation")
        correction = float(side_calibration[side]["upper_confidence_bound"])
        score = float(row["v6_7_base_score"])
        calibrated_score = score - correction
        if calibrated_score <= 0.0:
            continue
        updated = {
            **row,
            "v6_7_calibrated_runtime_pnl_lcb": calibrated_score,
            "side_residual_upper_confidence_bound": correction,
            "calibration_applied_without_future_target_access": True,
            "source_score_mutated": False,
            "labels_outcomes_resolution_or_pnl_opened": False,
        }
        updated["confirmatory_selected_row_id"] = canonical_json_sha256(updated)
        output.append(updated)
        seen_markets.add(market_id)
    output.sort(key=lambda row: (int(row["decision_ts"]), str(row["market_id"])))
    return output


def build_v6_7_side_only_confirmatory_gate(
    candidate_rows: list[dict[str, Any]],
    *,
    matched_legacy_rows: list[dict[str, Any]],
    evaluation_market_ids: list[str],
    profile: dict[str, Any],
    decision_freeze_sha256: str,
) -> dict[str, Any]:
    """Evaluate the single-use future evidence by BUY_UP/BUY_DOWN only."""

    validate_v6_7_evaluation_profile(profile)
    _require_sha256(decision_freeze_sha256, name="decision_freeze_sha256")
    gates = dict(profile["confirmatory_side_only_pnl_gate"])
    market_ids = list(dict.fromkeys(str(value) for value in evaluation_market_ids))
    if "" in market_ids or len(market_ids) != len(evaluation_market_ids):
        raise ValueError("#227 confirmatory evaluation market identity invalid")
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
        side: _group_metrics(rows, diagnostic_only=False)
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
    minimum_side = int(gates["minimum_supported_side_unique_market_count"])
    minimum_side_pnl = float(
        gates["supported_side_after_cost_pnl_minimum_exclusive"]
    )
    required_sides = list(gates["required_supported_sides"])
    checks = {
        "minimum_guard_accepted_unique_market_support": len(candidate_rows)
        >= int(gates["minimum_guard_accepted_unique_market_count"]),
        "buy_up_buy_down_side_support_and_pnl": all(
            side in side_metrics
            and side_metrics[side]["accepted_unique_market_count"] >= minimum_side
            and side_metrics[side]["after_cost_pnl_sum"] > minimum_side_pnl
            for side in required_sides
        ),
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
        "minimum_guard_accepted_unique_market_support": (
            "insufficient_guard_accepted_unique_market_support"
        ),
        "buy_up_buy_down_side_support_and_pnl": (
            "supported_side_post_cost_pnl_gate_failed"
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
        "accepted_side_distribution": dict(
            sorted(Counter(str(row["side"]) for row in candidate_rows).items())
        ),
        "accepted_side_metrics": side_metrics,
        "accepted_action_metrics": action_metrics,
        "accepted_action_family_metrics": {
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
        "pnl_hard_gate_aggregation": "selected_side_buy_up_buy_down_only",
        "action_and_action_family_pnl_diagnostic_only": True,
        "side_only_gate_checks": checks,
        "confirmatory_side_only_pnl_gate_passed": not blockers,
        "confirmatory_side_only_pnl_gate_blocking_reason_codes": blockers,
        "future_outcomes_used_for_model_threshold_cost_sizing_or_guard_tuning": False,
        "single_use_confirmatory": True,
        "result_selected_rerun_allowed": False,
        "manual_approval_does_not_bypass_side_only_pnl_gate": True,
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
            raise ValueError("#227 calibration row identity invalid")
        seen.add(key)
        for field in (
            "v6_7_base_score",
            "runtime_policy_after_cost_net_pnl_per_contract",
        ):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"#227 calibration field non-finite: {field}")
        if int(row["max_input_ts"]) > int(row["decision_ts"]):
            raise ValueError("#227 calibration feature causality violation")
        if (
            row.get("target_available_only_post_exit_or_official_resolution") is not True
            or row.get("target_used_as_decision_time_input") is not False
        ):
            raise ValueError("#227 calibration target isolation invalid")


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
            raise ValueError("#227 confirmatory evaluation row identity invalid")
        seen.add(market_id)
        if not math.isfinite(
            float(row["runtime_policy_after_cost_net_pnl_at_frozen_size"])
        ):
            raise ValueError("#227 confirmatory PnL is non-finite")


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
    "apply_v6_7_side_residual_calibration",
    "build_v6_7_side_only_confirmatory_gate",
    "build_v6_7_side_residual_calibration",
    "validate_v6_7_evaluation_profile",
]
