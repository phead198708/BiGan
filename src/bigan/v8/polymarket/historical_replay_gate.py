"""Strict pre-collection historical replay gate for the v8.5 challenger."""

from __future__ import annotations

import math
import string
from collections import Counter
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.parallel_future_gate import _bootstrap_sum_interval

HISTORICAL_REPLAY_CONTRACT_SCHEMA_VERSION = (
    "bigan-v8-challenge-historical-replay-contract-v1"
)
HISTORICAL_REPLAY_REPORT_SCHEMA_VERSION = (
    "bigan-v8-challenge-historical-replay-report-v1"
)
SUPPORTED_CANDIDATE_ID = "v8_1_primary_no_fallback"
SUPPORTED_BASELINE_ID = "matched_frozen_v6_7"


class HistoricalReplayGateError(ValueError):
    """Raised when historical replay evidence is malformed or contaminated."""


def validate_exact_historical_model_binding(
    *,
    candidate_contract: dict[str, Any],
    frozen_model_binding: dict[str, Any],
    frozen_model_artifact: dict[str, Any],
    source_manifest: dict[str, Any],
    expected_binding_sha256: str,
    expected_model_artifact_sha256: str,
    expected_source_manifest_sha256: str,
    expected_candidate_profile_sha256: str,
) -> None:
    """Bind replay evidence to exact model bytes and controller state."""

    state = dict(frozen_model_binding.get("initial_controller_state") or {})
    model_state = dict(frozen_model_artifact.get("final_rank_state") or {})
    weighted_model = dict(
        frozen_model_artifact.get("final_weighted_model") or {}
    )
    manifest_model = dict(source_manifest.get("model") or {})
    manifest_profile = dict(source_manifest.get("profile") or {})
    safety_false_fields = (
        "source_model_candidate_eligible",
        "freeze_ready",
        "promotion_evidence_eligible",
        "paper_candidate_allowed",
        "v8_execution_handoff_allowed",
        "capital_at_risk",
        "polymarket_write_enabled",
        "wallet_signing_enabled",
        "#134_resume_allowed",
        "#146_start_allowed",
    )
    checks = {
        "binding_sha256": (
            _valid_sha256(expected_binding_sha256)
            and candidate_contract.get("frozen_model_binding_sha256")
            == expected_binding_sha256
        ),
        "model_artifact_sha256": (
            _valid_sha256(expected_model_artifact_sha256)
            and candidate_contract.get("frozen_model_artifact_sha256")
            == expected_model_artifact_sha256
            and frozen_model_binding.get("frozen_model_artifact_sha256")
            == expected_model_artifact_sha256
            and manifest_model.get("sha256")
            == expected_model_artifact_sha256
        ),
        "model_artifact_id": (
            _valid_sha256(frozen_model_artifact.get("model_artifact_id"))
            and candidate_contract.get("frozen_model_artifact_id")
            == frozen_model_artifact.get("model_artifact_id")
            and frozen_model_binding.get("frozen_model_artifact_id")
            == frozen_model_artifact.get("model_artifact_id")
        ),
        "historical_manifest_sha256": (
            _valid_sha256(expected_source_manifest_sha256)
            and frozen_model_binding.get("historical_fit_manifest_sha256")
            == expected_source_manifest_sha256
        ),
        "candidate_profile_sha256": (
            _valid_sha256(expected_candidate_profile_sha256)
            and candidate_contract.get("profile_sha256")
            == expected_candidate_profile_sha256
            and frozen_model_binding.get("frozen_profile_sha256")
            == expected_candidate_profile_sha256
            and manifest_profile.get("sha256")
            == expected_candidate_profile_sha256
        ),
        "candidate_identity": (
            frozen_model_artifact.get("candidate_name")
            == "adaptive_support_controller_v8_1"
            and frozen_model_binding.get("candidate_name")
            == frozen_model_artifact.get("candidate_name")
        ),
        "booster_sha256": (
            _valid_sha256(weighted_model.get("booster_sha256"))
            and frozen_model_binding.get("frozen_booster_sha256")
            == weighted_model.get("booster_sha256")
        ),
        "rank_state_id": (
            _valid_sha256(model_state.get("rank_state_id"))
            and candidate_contract.get("initial_controller_state_id")
            == model_state.get("rank_state_id")
            and state.get("rank_state_id") == model_state.get("rank_state_id")
        ),
        "rank_lineage_hash": (
            _valid_sha256(model_state.get("rank_lineage_hash"))
            and state.get("rank_lineage_hash")
            == model_state.get("rank_lineage_hash")
        ),
        "eligible_prediction_scores_hash": (
            _valid_sha256(
                model_state.get("eligible_prediction_scores_hash")
            )
            and state.get("eligible_prediction_scores_hash")
            == model_state.get("eligible_prediction_scores_hash")
        ),
        "controller_guard_acceptance_history_hash": (
            _valid_sha256(
                model_state.get(
                    "controller_guard_acceptance_history_hash"
                )
            )
            and state.get("controller_guard_acceptance_history_hash")
            == model_state.get(
                "controller_guard_acceptance_history_hash"
            )
        ),
        "historical_gate_passed": (
            frozen_model_artifact.get("historical_hard_gate_passed") is True
            and not frozen_model_artifact.get(
                "historical_gate_blocking_reason_codes"
            )
        ),
        "target_free_controller": (
            state.get("controller_state_uses_target_outcome_label_or_pnl")
            is False
            and state.get(
                "future_controller_updates_use_strictly_prior_guard_results_only"
            )
            is True
            and model_state.get(
                "controller_target_outcome_label_or_pnl_free"
            )
            is True
        ),
        "all_safety_unlocks_remain_false": all(
            frozen_model_artifact.get(field) is False
            for field in safety_false_fields
        ),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise HistoricalReplayGateError(
            "exact historical model binding invalid: "
            + ", ".join(sorted(blockers))
        )


def validate_historical_replay_contract(contract: dict[str, Any]) -> None:
    """Validate the immutable, point-estimate pre-collection gate contract."""

    gate = dict(contract.get("gate") or {})
    bootstrap = dict(contract.get("paired_market_bootstrap") or {})
    checks = {
        "schema_version": contract.get("schema_version")
        == HISTORICAL_REPLAY_CONTRACT_SCHEMA_VERSION,
        "candidate_id": contract.get("candidate_id") == SUPPORTED_CANDIDATE_ID,
        "baseline_id": contract.get("baseline_id") == SUPPORTED_BASELINE_ID,
        "historical_role": contract.get("historical_role")
        == "development_screening_only_before_new_collection",
        "promotion_evidence": contract.get("promotion_evidence_eligible") is False,
        "exact_market_count": int(gate.get("exact_market_count") or 0) == 120,
        "minimum_support": int(gate.get("minimum_candidate_support") or 0) >= 40,
        "behavioral_difference": int(
            gate.get("minimum_policy_difference_market_count") or 0
        )
        >= 1,
        "strict_total_delta": float(
            gate.get(
                "candidate_minus_champion_total_pnl_minimum_exclusive",
                -1.0,
            )
        )
        == 0.0,
        "strict_candidate_lwr": float(
            gate.get(
                "candidate_largest_winner_removed_pnl_minimum_exclusive",
                -1.0,
            )
        )
        == 0.0,
        "strict_lwr_delta": float(
            gate.get(
                "candidate_minus_champion_largest_winner_removed_pnl_minimum_exclusive",
                -1.0,
            )
        )
        == 0.0,
        "strict_delta_lwr": float(
            gate.get(
                "paired_delta_largest_winner_removed_pnl_minimum_exclusive",
                -1.0,
            )
        )
        == 0.0,
        "same_contract": gate.get(
            "identical_market_cost_sizing_position_management_and_guard_required"
        )
        is True,
        "no_tuning": gate.get(
            "historical_pnl_feature_parameter_or_threshold_tuning_allowed"
        )
        is False,
        "bootstrap_method": bootstrap.get("method")
        == "paired_market_cluster_percentile",
        "bootstrap_alpha": float(bootstrap.get("alpha") or 0.0) == 0.0125,
        "bootstrap_count": int(bootstrap.get("resample_count") or 0) == 10_000,
        "bootstrap_diagnostic": bootstrap.get("diagnostic_only") is True,
        "bootstrap_not_blocking": bootstrap.get("eligibility_blocker") is False,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise HistoricalReplayGateError(
            "historical replay contract invalid: " + ", ".join(sorted(blockers))
        )


def audit_historical_replay_superiority(
    *,
    gate_contract: dict[str, Any],
    candidate_contract: dict[str, Any],
    source_report: dict[str, Any],
    leakage_audit: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    lineage_sha256s: dict[str, str],
    evaluation_completed_ts: int,
    exact_model_binding_verified: bool,
) -> dict[str, Any]:
    """Recompute the matched replay and enforce strict pre-collection superiority."""

    validate_historical_replay_contract(gate_contract)
    if evaluation_completed_ts <= 0:
        raise HistoricalReplayGateError("evaluation_completed_ts must be positive")
    _validate_row_sets(candidate_rows, baseline_rows)

    replay = dict(source_report.get("historical_prequential_hard_gate") or {})
    candidate_source = dict(replay.get("candidate") or {})
    baseline_source = dict(replay.get("v6_7_baseline") or {})
    baseline_by_key = {
        _row_key(row): float(row["after_cost_net_pnl_at_frozen_size"])
        for row in baseline_rows
    }
    candidate_by_key = {
        _row_key(row): float(row["after_cost_net_pnl_at_frozen_size"])
        for row in candidate_rows
    }
    market_keys = sorted(baseline_by_key)
    candidate_values = [candidate_by_key.get(key, 0.0) for key in market_keys]
    baseline_values = [baseline_by_key[key] for key in market_keys]
    delta_values = [
        candidate - baseline
        for candidate, baseline in zip(
            candidate_values, baseline_values, strict=True
        )
    ]

    candidate_total = sum(candidate_values)
    baseline_total = sum(baseline_values)
    total_delta = sum(delta_values)
    candidate_lwr = _largest_winner_removed(candidate_values)
    baseline_lwr = _largest_winner_removed(baseline_values)
    lwr_delta = candidate_lwr - baseline_lwr
    paired_delta_lwr = _largest_winner_removed(delta_values)
    gate = gate_contract["gate"]
    policy_difference_count = int(
        source_report.get("historical_policy_difference_market_count") or 0
    )
    market_ids = sorted(key[0] for key in market_keys)
    expected_market_hash = canonical_json_sha256(market_ids)
    fixed_sizes = {
        float(row["fixed_position_size"])
        for row in candidate_rows + baseline_rows
    }

    source_metrics_reconciled = all(
        (
            _close(
                candidate_total,
                candidate_source.get(
                    "total_after_cost_net_pnl_at_frozen_size"
                ),
            ),
            _close(
                baseline_total,
                baseline_source.get(
                    "total_after_cost_net_pnl_at_frozen_size"
                ),
            ),
            _close(
                total_delta,
                replay.get(
                    "candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size"
                ),
            ),
            _close(
                candidate_lwr,
                candidate_source.get(
                    "largest_winner_removed_after_cost_net_pnl_at_frozen_size"
                ),
            ),
            _close(
                baseline_lwr,
                baseline_source.get(
                    "largest_winner_removed_after_cost_net_pnl_at_frozen_size"
                ),
            ),
            _close(
                lwr_delta,
                replay.get(
                    "candidate_minus_v6_7_largest_winner_removed_after_cost_net_pnl_at_frozen_size"
                ),
            ),
        )
    )
    safety_false_fields = (
        "source_model_candidate_eligible",
        "freeze_ready",
        "promotion_evidence_eligible",
        "paper_candidate_allowed",
        "v8_execution_handoff_allowed",
        "capital_at_risk",
        "polymarket_write_enabled",
        "wallet_signing_enabled",
        "#134_resume_allowed",
        "#146_start_allowed",
    )
    checks = {
        "candidate_contract_is_frozen_v8_1_no_fallback": (
            candidate_contract.get("candidate_id") == SUPPORTED_CANDIDATE_ID
            and candidate_contract.get("source_model_hash_role")
            == "source_training_rows_sha256"
            and _valid_sha256(
                candidate_contract.get("frozen_model_binding_sha256")
            )
            and _valid_sha256(
                candidate_contract.get("frozen_model_artifact_sha256")
            )
            and _valid_sha256(
                candidate_contract.get("frozen_model_artifact_id")
            )
            and _valid_sha256(
                candidate_contract.get("initial_controller_state_id")
            )
            and float(
                candidate_contract.get("paper_position_size") or 0.0
            )
            == 0.2
            and candidate_contract.get("fallback_enabled") is False
            and candidate_contract.get("v6_7_routing_on_abstention_allowed")
            is False
            and candidate_contract.get("contract_frozen_before_target_access")
            is True
            and candidate_contract.get(
                "outcomes_labels_settlement_returns_or_pnl_opened"
            )
            is False
        ),
        "exact_same_historical_market_grid": (
            len(market_keys) == int(gate["exact_market_count"])
            and int(replay.get("evaluation_market_count") or 0)
            == len(market_keys)
            and replay.get("evaluation_market_ids_hash")
            == expected_market_hash
            and set(candidate_by_key).issubset(baseline_by_key)
        ),
        "identical_cost_sizing_position_management_and_guard": (
            fixed_sizes == {float(replay.get("fixed_position_size") or -1.0)}
            and replay.get("same_runtime_aligned_target_and_cost_contract")
            is True
            and replay.get("same_position_management_and_guard_contract")
            is True
            and replay.get("full_execution_guard_applied_to_candidate_and_baseline")
            is True
            and replay.get("common_selected_row_filter_applied") is False
            and float(replay.get("no_bet_market_pnl") or 0.0) == 0.0
        ),
        "source_report_metrics_reconciled": source_metrics_reconciled,
        "candidate_support_sufficient": len(candidate_rows)
        >= int(gate["minimum_candidate_support"]),
        "candidate_behaviorally_distinct": policy_difference_count
        >= int(gate["minimum_policy_difference_market_count"]),
        "candidate_total_pnl_strictly_better_than_champion": total_delta
        > float(
            gate["candidate_minus_champion_total_pnl_minimum_exclusive"]
        ),
        "candidate_largest_winner_removed_pnl_positive": candidate_lwr
        > float(
            gate["candidate_largest_winner_removed_pnl_minimum_exclusive"]
        ),
        "candidate_largest_winner_removed_strictly_better_than_champion": lwr_delta
        > float(
            gate[
                "candidate_minus_champion_largest_winner_removed_pnl_minimum_exclusive"
            ]
        ),
        "paired_delta_largest_winner_removed_positive": paired_delta_lwr
        > float(
            gate[
                "paired_delta_largest_winner_removed_pnl_minimum_exclusive"
            ]
        ),
        "prediction_and_rank_state_are_strictly_prequential": (
            source_report.get("historical_hard_gate_passed") is True
            and source_report.get("fit_leakage_audit_passed") is True
            and not source_report.get("historical_gate_blocking_reason_codes")
        ),
        "historical_targets_not_used_for_tuning_or_inference": (
            replay.get(
                "historical_oof_or_validation_pnl_used_for_feature_hyperparameter_or_threshold_tuning"
            )
            is False
            and replay.get("historical_pnl_used_for_pre_collection_screening_only")
            is True
            and all(
                row.get("target_used_as_decision_time_input") is False
                for row in candidate_rows + baseline_rows
            )
            and leakage_audit.get(
                "issue243_pnl_targets_winners_or_losers_used_for_threshold_selection"
            )
            is False
            and leakage_audit.get(
                "issue244_pnl_targets_winners_or_losers_used_for_controller_design"
            )
            is False
        ),
        "leakage_audit_passed": (
            leakage_audit.get("fit_leakage_audit_passed") is True
            and not leakage_audit.get("fit_leakage_blocking_reason_codes")
            and all(
                leakage_audit.get("fit_leakage_checks", {}).values()
            )
        ),
        "lineage_hashes_complete": bool(lineage_sha256s)
        and all(_valid_sha256(value) for value in lineage_sha256s.values()),
        "exact_frozen_model_binding_verified": (
            exact_model_binding_verified is True
        ),
        "all_safety_unlocks_remain_false": (
            all(source_report.get(field) is False for field in safety_false_fields)
            and all(leakage_audit.get(field) is False for field in safety_false_fields)
            and not any(candidate_contract.get("safety", {}).values())
        ),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    bootstrap = gate_contract["paired_market_bootstrap"]
    bootstrap_lcb, bootstrap_ucb = _bootstrap_sum_interval(
        {
            market_id: delta
            for (market_id, _decision_ts), delta in zip(
                market_keys, delta_values, strict=True
            )
        },
        alpha=float(bootstrap["alpha"]),
        resample_count=int(bootstrap["resample_count"]),
        seed=int(bootstrap["seed"]),
    )
    passed = not blockers
    report = {
        "schema_version": HISTORICAL_REPLAY_REPORT_SCHEMA_VERSION,
        "candidate_id": SUPPORTED_CANDIDATE_ID,
        "candidate_name": source_report.get("candidate_name"),
        "baseline_id": SUPPORTED_BASELINE_ID,
        "historical_role": gate_contract["historical_role"],
        "evaluation_completed_ts": evaluation_completed_ts,
        "lineage_sha256s": dict(sorted(lineage_sha256s.items())),
        "evaluation_market_count": len(market_keys),
        "evaluation_market_ids_hash": expected_market_hash,
        "candidate_accepted_market_count": len(candidate_rows),
        "champion_accepted_market_count": len(baseline_rows),
        "policy_difference_market_count": policy_difference_count,
        "candidate_action_distribution": dict(
            sorted(Counter(str(row["action"]) for row in candidate_rows).items())
        ),
        "candidate_side_distribution": dict(
            sorted(Counter(str(row["side"]) for row in candidate_rows).items())
        ),
        "metrics": {
            "candidate_total_after_cost_pnl": candidate_total,
            "champion_total_after_cost_pnl": baseline_total,
            "candidate_minus_champion_total_after_cost_pnl": total_delta,
            "candidate_largest_winner_removed_after_cost_pnl": candidate_lwr,
            "champion_largest_winner_removed_after_cost_pnl": baseline_lwr,
            "candidate_minus_champion_largest_winner_removed_after_cost_pnl": lwr_delta,
            "paired_delta_largest_winner_removed_after_cost_pnl": paired_delta_lwr,
        },
        "paired_market_bootstrap_diagnostic": {
            "method": bootstrap["method"],
            "alpha": float(bootstrap["alpha"]),
            "resample_count": int(bootstrap["resample_count"]),
            "seed": int(bootstrap["seed"]),
            "point_estimate": total_delta,
            "lower_confidence_bound": bootstrap_lcb,
            "upper_confidence_bound": bootstrap_ucb,
            "diagnostic_only": True,
            "eligibility_blocker": False,
        },
        "checks": checks,
        "blocking_reason_codes": blockers,
        "historical_superiority_gate_passed": passed,
        "future_collection_prerequisite_satisfied": passed,
        "historical_replay_is_promotion_evidence": False,
        "paper_candidate_allowed": False,
        "promotion_unlocked": False,
        "live_unlocked": False,
        "write_enabled": False,
        "wallet_enabled": False,
        "capital_at_risk": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def _validate_row_sets(
    candidate_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> None:
    if not candidate_rows or not baseline_rows:
        raise HistoricalReplayGateError("historical replay rows must not be empty")
    for role, rows in (
        ("candidate", candidate_rows),
        ("baseline", baseline_rows),
    ):
        keys = [_row_key(row) for row in rows]
        if len(keys) != len(set(keys)):
            raise HistoricalReplayGateError(
                f"{role} replay rows contain duplicate market decisions"
            )
        market_ids = [key[0] for key in keys]
        if len(market_ids) != len(set(market_ids)):
            raise HistoricalReplayGateError(
                f"{role} replay rows violate one-bet-per-market"
            )
        for row in rows:
            for field in (
                "market_id",
                "decision_ts",
                "action",
                "side",
                "fixed_position_size",
                "after_cost_net_pnl_at_frozen_size",
                "target_used_as_decision_time_input",
            ):
                if field not in row:
                    raise HistoricalReplayGateError(
                        f"{role} replay row missing {field}"
                    )
            if row["target_used_as_decision_time_input"] is not False:
                raise HistoricalReplayGateError(
                    f"{role} replay target contaminated a decision"
                )
            for field in (
                "fixed_position_size",
                "after_cost_net_pnl_at_frozen_size",
            ):
                value = float(row[field])
                if not math.isfinite(value):
                    raise HistoricalReplayGateError(
                        f"{role} replay row has non-finite {field}"
                    )


def _row_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["market_id"]), int(row["decision_ts"])


def _largest_winner_removed(values: list[float]) -> float:
    return sum(values) - max(
        [value for value in values if value > 0.0],
        default=0.0,
    )


def _close(actual: float, expected: Any) -> bool:
    try:
        return math.isclose(
            actual,
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    except (TypeError, ValueError):
        return False


def _valid_sha256(value: Any) -> bool:
    candidate = str(value or "").lower()
    return len(candidate) == 64 and all(
        character in string.hexdigits.lower() for character in candidate
    )


__all__ = [
    "HISTORICAL_REPLAY_CONTRACT_SCHEMA_VERSION",
    "HISTORICAL_REPLAY_REPORT_SCHEMA_VERSION",
    "HistoricalReplayGateError",
    "audit_historical_replay_superiority",
    "validate_exact_historical_model_binding",
    "validate_historical_replay_contract",
]
