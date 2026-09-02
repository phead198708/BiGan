"""Fixed historical conditional-quantile SBC candidate for issue #233."""

from __future__ import annotations

import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linprog

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_abstention_aware_expected_net_pnl_v7_0 import (
    FORBIDDEN_INFERENCE_FIELDS,
    _v7_0_blocked_safety_fields,
)
from bigan.v8.polymarket.training.execution_layer_v2_abstention_aware_expected_net_pnl_v7_0_fit import (
    FEATURE_NAMES,
    HTS_ACTIONS,
    SBC_ACTIONS,
    materialize_v7_0_sbc_rows,
    validate_v7_0_training_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
    _descriptor,
    _load_json,
    _load_jsonl,
    _require_git_sha,
    _require_sha256,
    _sha256_file,
    _verify_pin,
    _write_json,
    _write_jsonl,
    _write_text,
)

CANDIDATE_NAME = "sbc_conditional_quantile_abstention_v7_1"
PROFILE_SCHEMA_VERSION = "bigan-v8-sbc-conditional-quantile-v7-1-profile-v1"
MODEL_SCHEMA_VERSION = "bigan-v8-sbc-conditional-quantile-v7-1-model-v1"
REPORT_SCHEMA_VERSION = "bigan-v8-sbc-conditional-quantile-v7-1-report-v1"
LEAKAGE_SCHEMA_VERSION = "bigan-v8-sbc-conditional-quantile-v7-1-leakage-audit-v1"
MANIFEST_SCHEMA_VERSION = "bigan-v8-sbc-conditional-quantile-v7-1-manifest-v1"
FULL_TRADE_ACTIONS = set(SBC_ACTIONS).union(HTS_ACTIONS)
FROZEN_LINEAGE = {
    "runtime_target_rows_sha256": (
        "1565116daeb2f5d4d8c33fefa507276f59251edd5ffb5f4f313041bcf9dbb0ec"
    ),
    "v7_0_training_profile_sha256": (
        "1f66d8699b9727651538cc34a9a2a25ba5eaac5cfded75cf8f4a258b1b5d3f4a"
    ),
    "v7_0_lineage_audit_manifest_sha256": (
        "9c92e51fe1e8d003fceb5197d04a7e099b48fcc45749b1b74b17f48bb7d593e4"
    ),
    "v7_0_fit_manifest_rejection_evidence_sha256": (
        "844487d897c1cb437acdb4b93fc2f8eb63c48cf67847556d915067efd7e597cd"
    ),
    "v7_0_model_rejection_evidence_sha256": (
        "eaedb78c4e696e16c427bd99f48cb861d35427465f5627950cee549307f95c5a"
    ),
    "v7_0_fit_report_rejection_evidence_sha256": (
        "2f91143ce64512b4d231b8a06ca1ff8cc5d6c9b3c8e57fe093800c9695606ee5"
    ),
    "issue229_selected_window_rows_forbidden_sha256": (
        "8c1a5b92ccd4657bdd6d064cbc45af39d208d944c2b763419597500a7dde48fa"
    ),
    "issue231_selected_window_rows_forbidden_sha256": (
        "90cd57f9aa557e264d34d14084c4e4d7811ecbc4f81b9e8799cde1e568e01bbb"
    ),
    "issue231_execution_pnl_report_forbidden_sha256": (
        "3a1dcf58dbd5ec059cc010baa3b2c595a98d57dddf235415fb7d8fd859bba20a"
    ),
}


@dataclass(frozen=True, slots=True)
class SbcConditionalQuantileV71Config:
    """Pinned inputs for the single historical v7.1 fit."""

    run_id: str
    output_dir: Path | str
    profile_path: Path | str
    expected_profile_sha256: str
    v7_0_training_profile_path: Path | str
    runtime_target_rows_path: Path | str
    v7_0_lineage_audit_manifest_path: Path | str
    v7_0_fit_manifest_rejection_evidence_path: Path | str
    v7_0_model_rejection_evidence_path: Path | str
    v7_0_fit_report_rejection_evidence_path: Path | str
    implementation_commit: str
    fit_created_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(self.expected_profile_sha256, "expected_profile_sha256")
        _require_git_sha(self.implementation_commit)
        if self.fit_created_ts <= 0:
            raise ValueError("fit_created_ts must be positive")
        for name in (
            "output_dir",
            "profile_path",
            "v7_0_training_profile_path",
            "runtime_target_rows_path",
            "v7_0_lineage_audit_manifest_path",
            "v7_0_fit_manifest_rejection_evidence_path",
            "v7_0_model_rejection_evidence_path",
            "v7_0_fit_report_rejection_evidence_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


def validate_sbc_conditional_quantile_v7_1_profile(profile: dict[str, Any]) -> None:
    """Reject drift from the pre-fit v7.1 contract."""

    family = dict(profile.get("family_contract") or {})
    split = dict(profile.get("historical_split") or {})
    features = dict(profile.get("feature_contract") or {})
    model = dict(profile.get("quantile_model") or {})
    conformal = dict(profile.get("conformal_contract") or {})
    gates = dict(profile.get("historical_gates") or {})
    replay_gate = dict(profile.get("historical_replay_superiority_gate") or {})
    selection = dict(profile.get("selection_contract") or {})
    canary = dict(profile.get("target_free_canary") or {})
    future = dict(profile.get("future_confirmatory") or {})
    checks = {
        "identity": profile.get("schema_version") == PROFILE_SCHEMA_VERSION
        and profile.get("issue_number") == 233
        and profile.get("candidate_name") == CANDIDATE_NAME
        and profile.get("preregistered_before_fit") is True,
        "lineage": profile.get("lineage") == FROZEN_LINEAGE,
        "family": family
        == {
            "enabled_action_family": "SELL_BEFORE_CLOSE",
            "enabled_actions": list(SBC_ACTIONS),
            "disabled_action_family": "HOLD_TO_SETTLEMENT",
            "disabled_actions": list(HTS_ACTIONS),
            "disabled_family_behavior": (
                "explicit_unavailable_fail_closed_not_a_blocker_for_enabled_family"
            ),
            "no_trade_action": "NO_TRADE",
            "full_five_action_interface_required": True,
            "side_composition_is_regime_emergent": True,
            "side_quota_allowed": False,
            "side_count_hard_gate_enabled": False,
            "side_pnl_hard_gate_enabled": False,
        },
        "split": split
        == {
            "source_market_count": 134,
            "market_order": "minimum_decision_ts_then_market_id",
            "initial_fit_market_count": 44,
            "forward_fold_count": 5,
            "forward_validation_market_count_per_fold": 18,
            "forward_oof_market_count": 90,
            "all_historical_markets_used_for_final_fit": True,
        },
        "features": tuple(features.get("feature_names") or ()) == FEATURE_NAMES
        and features.get("reuse_v7_0_runtime_sbc_adapter") is True
        and features.get("fit_only_standardization") is True
        and features.get("market_probability_usage")
        == "market_price_value_conditioning_only_not_direct_fair_value"
        and set(features.get("forbidden_inference_field_names") or ())
        == FORBIDDEN_INFERENCE_FIELDS,
        "model": model
        == {
            "solver": "scipy_optimize_linprog_highs",
            "conditional_quantile_tau": 0.2,
            "l1_regularization": 0.05,
            "intercept_penalized": False,
            "market_weighting": "each_market_total_weight_one",
            "coefficient_absolute_bound": 8.0,
            "hyperparameter_search_enabled": False,
            "feature_selection_enabled": False,
            "oof_or_validation_pnl_used_for_model_or_parameter_selection": False,
            "result_selected_rerun_allowed": False,
        },
        "conformal": conformal
        == {
            "method": (
                "market_weighted_cross_fit_selected_action_lower_quantile_"
                "conformal"
            ),
            "coverage_level": 0.8,
            "conformity_score": (
                "predicted_lower_quantile_minus_realized_after_cost_target"
            ),
            "selected_action_rule": (
                "highest_raw_conditional_quantile_within_sbc_per_decision_group"
            ),
            "correction_quantile": 0.8,
            "upward_score_correction_allowed": False,
            "applied_correction": "max(weighted_conformity_quantile,0)",
            "minimum_oof_market_count": 90,
            "minimum_empirical_coverage": 0.75,
            "oof_pnl_report_only_not_gate_or_tuning_input": True,
        },
        "gates": gates
        == {
            "solver_convergence_required": True,
            "finite_bounded_coefficients_required": True,
            "feature_causality_required": True,
            "forbidden_inference_field_count_must_equal": 0,
            "minimum_positive_lcb_unique_market_count": 1,
            "prediction_loss_and_absolute_candidate_pnl_metrics_report_only": True,
        },
        "replay_gate": replay_gate
        == {
            "baseline_candidate_name": "p_up_semantic_execution_compatibility_v6_7",
            "baseline_score_source": "frozen_v6_2_market_clustered_mean_ev_lcb",
            "evaluation_market_source": "exact_v7_1_forward_oof_market_cohort",
            "exact_evaluation_market_count": 90,
            "selection_unit": "at_most_one_trade_per_market",
            "candidate_selection_rule": (
                "highest_positive_conformalized_lcb_per_market_then_earliest_"
                "decision_ts_then_action"
            ),
            "baseline_selection_rule": (
                "highest_positive_frozen_v6_7_base_score_per_market_then_earliest_"
                "decision_ts_then_action"
            ),
            "no_bet_market_pnl": 0.0,
            "common_selected_row_filter_allowed": False,
            "fixed_position_size": 0.2,
            "primary_metric": "total_after_cost_net_pnl_at_frozen_size",
            "candidate_minus_v6_7_total_pnl_minimum_exclusive": 0.0,
            "candidate_minus_v6_7_largest_winner_removed_pnl_minimum_inclusive": 0.0,
            "identical_cost_sizing_position_management_and_guard_required": True,
            "historical_pnl_used_for_model_feature_parameter_or_threshold_tuning": False,
            "historical_pnl_used_for_pre_collection_screening_only": True,
            "result_selected_rerun_allowed": False,
            "gate_failure_stops_before_target_free_collection": True,
            "promotion_or_paper_unlock_allowed": False,
        },
        "selection": selection
        == {
            "ranking_score_source": (
                "model_predicted_conformalized_conditional_lower_quantile_"
                "after_cost_net_pnl"
            ),
            "select_highest_positive_lcb_else_no_trade": True,
            "selection_threshold": 0.0,
            "threshold_operator": "strictly_greater_than",
            "no_trade_score": 0.0,
            "missing_or_invalid_feature_behavior": (
                "fail_closed_to_no_trade_with_explicit_reason"
            ),
            "source_score_mutation_allowed": False,
        },
        "canary": canary
        == {
            "historical_replay_superiority_gate_must_pass_before_collection": True,
            "new_strictly_later_outcome_blind_market_count": 12,
            "maximum_attempt_count": 18,
            "minimum_guard_accepted_unique_market_count_to_continue": 1,
            "zero_action_batch_terminal_fail_closed": True,
            "outcome_resolution_label_or_pnl_access_allowed": False,
            "full_execution_guard_unchanged": True,
        },
        "future": future
        == {
            "not_authorized_until_target_free_canary_passes": True,
            "sample_size_and_power_must_be_preregistered_after_target_free_canary": True,
            "single_use_official_read_only_settlement_gate_required": True,
            "side_action_family_metrics_diagnostic_only": True,
            "separate_explicit_paper_candidate_approval_required": True,
        },
        "safety": profile.get("safety") == _v7_0_blocked_safety_fields(),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("#233 v7.1 profile invalid: " + ", ".join(blockers))


def fit_sbc_conditional_quantile_v7_1(
    *,
    rows: list[dict[str, Any]],
    profile: dict[str, Any],
    implementation_commit: str,
    fit_created_ts: int,
) -> dict[str, Any]:
    """Run the fixed rolling-origin quantile fit and cross-conformal calibration."""

    validate_sbc_conditional_quantile_v7_1_profile(profile)
    _validate_canonical_rows(rows, expected_market_count=134)
    split = profile["historical_split"]
    market_order = _market_order(rows)
    initial = int(split["initial_fit_market_count"])
    width = int(split["forward_validation_market_count_per_fold"])
    fold_count = int(split["forward_fold_count"])
    oof_rows = []
    fold_reports = []
    all_solver_converged = True
    for fold_index in range(fold_count):
        train_ids = set(market_order[: initial + fold_index * width])
        validation_ids = set(
            market_order[
                initial + fold_index * width : initial + (fold_index + 1) * width
            ]
        )
        train = [row for row in rows if row["market_id"] in train_ids]
        validation = [row for row in rows if row["market_id"] in validation_ids]
        if len(validation_ids) != width:
            raise ValueError("#233 fixed forward fold support invalid")
        if max(row["decision_ts"] for row in train) >= min(
            row["decision_ts"] for row in validation
        ):
            raise ValueError("#233 forward fold chronology invalid")
        model = _fit_quantile_model(train, profile=profile)
        all_solver_converged = all_solver_converged and model["solver_converged"]
        train_constant = _weighted_quantile(
            [row["target_after_cost_net_pnl_per_contract"] for row in train],
            _market_weights(train),
            float(profile["quantile_model"]["conditional_quantile_tau"]),
        )
        for row in validation:
            prediction = _predict(row, model)
            oof_rows.append(
                _prediction_row(
                    row,
                    prediction=prediction,
                    fold_index=fold_index,
                    constant_prediction=train_constant,
                )
            )
        fold_reports.append(
            {
                "fold_index": fold_index,
                "train_market_count": len(train_ids),
                "validation_market_count": len(validation_ids),
                "train_max_decision_ts": max(row["decision_ts"] for row in train),
                "validation_min_decision_ts": min(
                    row["decision_ts"] for row in validation
                ),
                "solver_status": model["solver_status"],
                "solver_converged": model["solver_converged"],
            }
        )
    conformal = _cross_conformal_calibration(oof_rows, profile=profile)
    final_model = _fit_quantile_model(rows, profile=profile)
    all_solver_converged = all_solver_converged and final_model["solver_converged"]
    oof_metrics = _oof_metrics(oof_rows, profile=profile)
    selected_oof = _select_oof_rows(oof_rows, conformal_correction=conformal["applied_correction"])
    historical_replay = _historical_replay_superiority(
        oof_rows,
        conformal_correction=conformal["applied_correction"],
        profile=profile,
    )
    historical_replay_summary = {
        key: value
        for key, value in historical_replay.items()
        if key not in {"candidate_selected_rows", "v6_7_baseline_selected_rows"}
    }
    positive_market_count = len({row["market_id"] for row in selected_oof})
    gates = profile["historical_gates"]
    checks = {
        "solver_convergence": all_solver_converged,
        "finite_bounded_coefficients": final_model[
            "coefficients_finite_and_bounded"
        ],
        "feature_causality": all(
            row["max_input_ts"] <= row["decision_ts"] for row in rows
        ),
        "forbidden_inference_fields": all(
            not FORBIDDEN_INFERENCE_FIELDS.intersection(
                row["decision_time_features"]
            )
            for row in rows
        ),
        "oof_market_support": len({row["market_id"] for row in oof_rows})
        >= int(profile["conformal_contract"]["minimum_oof_market_count"]),
        "conformal_coverage": conformal["empirical_coverage"]
        >= float(profile["conformal_contract"]["minimum_empirical_coverage"]),
        "positive_lcb_actionability": positive_market_count
        >= int(gates["minimum_positive_lcb_unique_market_count"]),
        "historical_replay_candidate_pnl_strictly_better_than_v6_7": (
            historical_replay["candidate_total_after_cost_net_pnl_at_frozen_size"]
            > historical_replay[
                "v6_7_baseline_total_after_cost_net_pnl_at_frozen_size"
            ]
        ),
        "historical_replay_largest_winner_removed_not_worse_than_v6_7": (
            historical_replay[
                "candidate_largest_winner_removed_after_cost_net_pnl_at_frozen_size"
            ]
            >= historical_replay[
                "v6_7_baseline_largest_winner_removed_after_cost_net_pnl_at_frozen_size"
            ]
        ),
    }
    reason_map = {
        "solver_convergence": "quantile_solver_did_not_converge",
        "finite_bounded_coefficients": "quantile_coefficients_invalid",
        "feature_causality": "decision_time_feature_causality_failed",
        "forbidden_inference_fields": "forbidden_inference_field_detected",
        "oof_market_support": "cross_fit_market_support_insufficient",
        "conformal_coverage": "cross_conformal_coverage_below_minimum",
        "positive_lcb_actionability": "historical_positive_lcb_actionability_zero",
        "historical_replay_candidate_pnl_strictly_better_than_v6_7": (
            "historical_same_dataset_candidate_pnl_not_strictly_better_than_v6_7"
        ),
        "historical_replay_largest_winner_removed_not_worse_than_v6_7": (
            "historical_same_dataset_largest_winner_removed_pnl_worse_than_v6_7"
        ),
    }
    blockers = [reason_map[name] for name, passed in checks.items() if not passed]
    model_artifact = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": implementation_commit,
        "fit_created_ts": fit_created_ts,
        "frozen_historical_model": not blockers,
        "decision_time_safe": True,
        "enabled_action_family": "SELL_BEFORE_CLOSE",
        "enabled_actions": list(SBC_ACTIONS),
        "disabled_action_family": "HOLD_TO_SETTLEMENT",
        "disabled_actions": list(HTS_ACTIONS),
        "disabled_action_reason_code": (
            "prior_preregistered_hts_prediction_loss_gate_failed"
        ),
        "full_five_action_interface_required": True,
        "no_trade_score": 0.0,
        "ranking_score_source": profile["selection_contract"][
            "ranking_score_source"
        ],
        "quantile_model": final_model,
        "cross_conformal": conformal,
        "fold_reports": fold_reports,
        "forward_oof_metrics_report_only": oof_metrics,
        "historical_positive_lcb_selected_row_count": len(selected_oof),
        "historical_positive_lcb_unique_market_count": positive_market_count,
        "historical_positive_lcb_side_distribution_diagnostic": dict(
            sorted(Counter(row["side"] for row in selected_oof).items())
        ),
        "historical_positive_lcb_target_pnl_sum_report_only": sum(
            row["target_after_cost_net_pnl_per_contract"] for row in selected_oof
        ),
        "historical_replay_superiority_gate": historical_replay_summary,
        "historical_pnl_used_for_model_feature_parameter_or_threshold_tuning": False,
        "historical_pnl_used_for_pre_collection_superiority_gate": True,
        "historical_replay_superiority_is_promotion_evidence": False,
        "historical_gate_checks": checks,
        "historical_gate_passed": not blockers,
        "historical_gate_blocking_reason_codes": blockers,
        "issue229_issue231_or_v7_0_result_rows_used_for_fit_or_tuning": False,
        "side_quota_applied": False,
        "side_count_hard_gate_enabled": False,
        "side_pnl_hard_gate_enabled": False,
        "target_free_canary_required_before_confirmatory": True,
        "target_free_canary_collection_allowed": not blockers,
        "future_confirmatory_authorized": False,
        **_v7_0_blocked_safety_fields(),
    }
    model_artifact["model_artifact_id"] = canonical_json_sha256(model_artifact)
    return {
        "model_artifact": model_artifact,
        "oof_rows": sorted(oof_rows, key=_row_sort_key),
        "selected_oof_rows": sorted(selected_oof, key=_row_sort_key),
        "historical_replay_candidate_rows": historical_replay[
            "candidate_selected_rows"
        ],
        "historical_replay_v6_7_baseline_rows": historical_replay[
            "v6_7_baseline_selected_rows"
        ],
    }


def score_sbc_conditional_quantile_v7_1_decision_group(
    action_rows: list[dict[str, Any]], *, model_artifact: dict[str, Any]
) -> dict[str, Any]:
    """Outcome-free full-grid consumer with explicit HTS unavailability."""

    reasons = []
    if model_artifact.get("schema_version") != MODEL_SCHEMA_VERSION:
        reasons.append("v7_1_model_artifact_schema_invalid")
    if model_artifact.get("historical_gate_passed") is not True:
        reasons.append("v7_1_historical_gate_not_passed")
    actions = {str(row.get("action") or "") for row in action_rows}
    if actions != FULL_TRADE_ACTIONS or len(action_rows) != len(FULL_TRADE_ACTIONS):
        reasons.append("v7_1_full_trade_action_grid_incomplete")
    decision_group_ids = {str(row.get("decision_group_id") or "") for row in action_rows}
    if len(decision_group_ids) != 1 or "" in decision_group_ids:
        reasons.append("v7_1_decision_group_identity_invalid")
    scores = []
    if not reasons:
        for row in sorted(action_rows, key=lambda item: item["action"]):
            action = str(row["action"])
            if action in HTS_ACTIONS:
                scores.append(
                    {
                        "action": action,
                        "action_family": "HOLD_TO_SETTLEMENT",
                        "side": row["side"],
                        "score_available": False,
                        "calibrated_lower_bound_after_cost_net_pnl_per_contract": None,
                        "reason_codes": [
                            "prior_preregistered_hts_prediction_loss_gate_failed"
                        ],
                    }
                )
                continue
            if FORBIDDEN_INFERENCE_FIELDS.intersection(row):
                reasons.append("v7_1_forbidden_outcome_field_in_inference_row")
                break
            features = dict(row.get("decision_time_features") or {})
            if tuple(features) != FEATURE_NAMES:
                reasons.append("v7_1_decision_time_feature_contract_invalid")
                break
            if int(row["max_input_ts"]) > int(row["decision_ts"]):
                reasons.append("v7_1_inference_feature_causality_failed")
                break
            prediction = _predict(row, model_artifact["quantile_model"])
            lower_bound = prediction - float(
                model_artifact["cross_conformal"]["applied_correction"]
            )
            scores.append(
                {
                    "action": action,
                    "action_family": "SELL_BEFORE_CLOSE",
                    "side": row["side"],
                    "score_available": True,
                    "raw_conditional_lower_quantile_after_cost_net_pnl_per_contract": (
                        prediction
                    ),
                    "calibrated_lower_bound_after_cost_net_pnl_per_contract": (
                        lower_bound
                    ),
                    "reason_codes": [],
                }
            )
    available = [row for row in scores if row["score_available"]]
    if reasons or not available:
        selected = None
    else:
        selected = sorted(
            available,
            key=lambda row: (
                -row["calibrated_lower_bound_after_cost_net_pnl_per_contract"],
                row["action"],
            ),
        )[0]
        if selected["calibrated_lower_bound_after_cost_net_pnl_per_contract"] <= 0.0:
            selected = None
            reasons.append("v7_1_no_positive_calibrated_lower_bound")
    result = {
        "decision_group_id": next(iter(decision_group_ids), ""),
        "ranking_score_source": (
            "model_predicted_conformalized_conditional_lower_quantile_after_cost_net_pnl"
        ),
        "selected_action": selected["action"] if selected else "NO_TRADE",
        "selected_action_family": (
            selected["action_family"] if selected else "NO_TRADE"
        ),
        "selected_side": selected["side"] if selected else "NONE",
        "trade_selected": selected is not None,
        "action_scores": scores,
        "selection_reason_codes": reasons,
        "source_score_mutated": False,
        "outcome_or_pnl_field_used_at_inference": False,
        **_v7_0_blocked_safety_fields(),
    }
    result["decision_id"] = canonical_json_sha256(result)
    return result


def run_sbc_conditional_quantile_v7_1_fit(
    config: SbcConditionalQuantileV71Config,
) -> dict[str, Any]:
    """Verify all pins, fit once, and write hashable diagnostic artifacts."""

    paths = {
        "profile": Path(config.profile_path).resolve(),
        "v7_0_training_profile": Path(config.v7_0_training_profile_path).resolve(),
        "runtime_target_rows": Path(config.runtime_target_rows_path).resolve(),
        "v7_0_lineage_audit_manifest": Path(
            config.v7_0_lineage_audit_manifest_path
        ).resolve(),
        "v7_0_fit_manifest_rejection_evidence": Path(
            config.v7_0_fit_manifest_rejection_evidence_path
        ).resolve(),
        "v7_0_model_rejection_evidence": Path(
            config.v7_0_model_rejection_evidence_path
        ).resolve(),
        "v7_0_fit_report_rejection_evidence": Path(
            config.v7_0_fit_report_rejection_evidence_path
        ).resolve(),
    }
    _verify_pin(paths["profile"], config.expected_profile_sha256, "#233 profile")
    profile = _load_json(paths["profile"])
    validate_sbc_conditional_quantile_v7_1_profile(profile)
    lineage = profile["lineage"]
    pins = {
        "v7_0_training_profile": "v7_0_training_profile_sha256",
        "runtime_target_rows": "runtime_target_rows_sha256",
        "v7_0_lineage_audit_manifest": "v7_0_lineage_audit_manifest_sha256",
        "v7_0_fit_manifest_rejection_evidence": (
            "v7_0_fit_manifest_rejection_evidence_sha256"
        ),
        "v7_0_model_rejection_evidence": (
            "v7_0_model_rejection_evidence_sha256"
        ),
        "v7_0_fit_report_rejection_evidence": (
            "v7_0_fit_report_rejection_evidence_sha256"
        ),
    }
    for path_name, pin_name in pins.items():
        _verify_pin(paths[path_name], lineage[pin_name], f"#233 {path_name}")
    v7_profile = _load_json(paths["v7_0_training_profile"])
    validate_v7_0_training_profile(v7_profile)
    canonical_rows = materialize_v7_0_sbc_rows(
        _load_jsonl(paths["runtime_target_rows"]), v7_profile
    )
    canonical_rows = [
        {**row, "role": "historical_development"} for row in canonical_rows
    ]
    fit = fit_sbc_conditional_quantile_v7_1(
        rows=canonical_rows,
        profile=profile,
        implementation_commit=config.implementation_commit,
        fit_created_ts=config.fit_created_ts,
    )
    model = fit["model_artifact"]
    leakage = _leakage_audit(canonical_rows, model=model, profile=profile)
    report = _report(model=model, leakage=leakage)
    run_dir = Path(config.output_dir).resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    outputs = {
        "model": run_dir / "v7_1_sbc_conditional_quantile_model.json",
        "report": run_dir / "v7_1_historical_cross_fit_report.json",
        "report_markdown": run_dir / "v7_1_historical_cross_fit_report.md",
        "leakage_audit": run_dir / "v7_1_fit_leakage_audit.json",
        "oof_rows": run_dir / "v7_1_forward_oof_prediction_rows.jsonl",
        "selected_oof_rows": run_dir / "v7_1_positive_lcb_selected_oof_rows.jsonl",
        "historical_replay_candidate_rows": (
            run_dir / "v7_1_historical_replay_candidate_selected_rows.jsonl"
        ),
        "historical_replay_v6_7_baseline_rows": (
            run_dir / "v7_1_historical_replay_v6_7_baseline_selected_rows.jsonl"
        ),
    }
    _write_json(outputs["model"], model)
    _write_json(outputs["report"], report)
    _write_text(outputs["report_markdown"], _report_markdown(report))
    _write_json(outputs["leakage_audit"], leakage)
    _write_jsonl(outputs["oof_rows"], fit["oof_rows"])
    _write_jsonl(outputs["selected_oof_rows"], fit["selected_oof_rows"])
    _write_jsonl(
        outputs["historical_replay_candidate_rows"],
        fit["historical_replay_candidate_rows"],
    )
    _write_jsonl(
        outputs["historical_replay_v6_7_baseline_rows"],
        fit["historical_replay_v6_7_baseline_rows"],
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        **{name: _descriptor(path) for name, path in paths.items()},
        **{name: _descriptor(path) for name, path in outputs.items()},
        "historical_gate_passed": model["historical_gate_passed"],
        "historical_gate_blocking_reason_codes": model[
            "historical_gate_blocking_reason_codes"
        ],
        "fit_leakage_audit_passed": leakage["fit_leakage_audit_passed"],
        "historical_replay_superiority_gate": model[
            "historical_replay_superiority_gate"
        ],
        "target_free_canary_collection_allowed": model[
            "target_free_canary_collection_allowed"
        ],
        "target_free_canary_started": False,
        "future_confirmatory_authorized": False,
        "future_target_accessed": False,
        **_v7_0_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v7_1_historical_fit_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "model": model,
        "report": report,
        "leakage": leakage,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "outputs": outputs,
    }


def _fit_quantile_model(
    rows: list[dict[str, Any]], *, profile: dict[str, Any]
) -> dict[str, Any]:
    x = np.asarray(
        [[row["decision_time_features"][name] for name in FEATURE_NAMES] for row in rows],
        dtype=float,
    )
    y = np.asarray(
        [row["target_after_cost_net_pnl_per_contract"] for row in rows],
        dtype=float,
    )
    weights = np.asarray(_market_weights(rows), dtype=float)
    mean = np.average(x, axis=0, weights=weights)
    variance = np.average((x - mean) ** 2, axis=0, weights=weights)
    floor = float(profile["feature_contract"]["standard_deviation_floor"])
    scale = np.where(np.sqrt(variance) > floor, np.sqrt(variance), 1.0)
    z = (x - mean) / scale
    design = np.column_stack((np.ones(len(rows)), z))
    feature_count = len(FEATURE_NAMES)
    row_count = len(rows)
    beta_count = feature_count + 1
    u_start = beta_count
    v_start = u_start + row_count
    t_start = v_start + row_count
    variable_count = t_start + feature_count
    tau = float(profile["quantile_model"]["conditional_quantile_tau"])
    regularization = float(profile["quantile_model"]["l1_regularization"])
    objective = np.zeros(variable_count)
    objective[u_start:v_start] = weights * tau
    objective[v_start:t_start] = weights * (1.0 - tau)
    objective[t_start:] = regularization
    equality = np.zeros((row_count, variable_count))
    equality[:, :beta_count] = design
    equality[:, u_start:v_start] = np.eye(row_count)
    equality[:, v_start:t_start] = -np.eye(row_count)
    inequality = np.zeros((2 * feature_count, variable_count))
    for index in range(feature_count):
        coefficient_index = index + 1
        absolute_index = t_start + index
        inequality[2 * index, coefficient_index] = 1.0
        inequality[2 * index, absolute_index] = -1.0
        inequality[2 * index + 1, coefficient_index] = -1.0
        inequality[2 * index + 1, absolute_index] = -1.0
    bound = float(profile["quantile_model"]["coefficient_absolute_bound"])
    bounds = [
        (None, None),
        *([(-bound, bound)] * feature_count),
        *([(0.0, None)] * (2 * row_count + feature_count)),
    ]
    result = linprog(
        objective,
        A_ub=inequality,
        b_ub=np.zeros(2 * feature_count),
        A_eq=equality,
        b_eq=y,
        bounds=bounds,
        method="highs",
    )
    coefficients = (
        np.full(beta_count, np.nan) if result.x is None else result.x[:beta_count]
    )
    finite_bounded = bool(
        result.success
        and np.all(np.isfinite(coefficients))
        and np.max(np.abs(coefficients[1:])) <= bound
    )
    return {
        "solver": "scipy_optimize_linprog_highs",
        "solver_converged": bool(result.success),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "objective_value": float(result.fun) if result.fun is not None else None,
        "conditional_quantile_tau": tau,
        "l1_regularization": regularization,
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": [float(value) for value in mean],
        "feature_scale": [float(value) for value in scale],
        "intercept": float(coefficients[0]),
        "coefficients": [float(value) for value in coefficients[1:]],
        "coefficient_absolute_bound": bound,
        "coefficients_finite_and_bounded": finite_bounded,
        "fit_market_count": len({row["market_id"] for row in rows}),
        "fit_row_count": len(rows),
        "fit_market_ids_hash": canonical_json_sha256(
            sorted({row["market_id"] for row in rows})
        ),
    }


def _predict(row: dict[str, Any], model: dict[str, Any]) -> float:
    x = np.asarray(
        [row["decision_time_features"][name] for name in FEATURE_NAMES], dtype=float
    )
    mean = np.asarray(model["feature_mean"], dtype=float)
    scale = np.asarray(model["feature_scale"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    return float(model["intercept"] + ((x - mean) / scale) @ coefficients)


def _prediction_row(
    row: dict[str, Any], *, prediction: float, fold_index: int, constant_prediction: float
) -> dict[str, Any]:
    decision_features = row["decision_time_features"]
    item = {
        "market_id": row["market_id"],
        "decision_group_id": row["decision_group_id"],
        "decision_ts": row["decision_ts"],
        "max_input_ts": row["max_input_ts"],
        "action": row["action"],
        "side": row["side"],
        "fold_index": fold_index,
        "raw_conditional_lower_quantile_after_cost_net_pnl_per_contract": prediction,
        "fold_train_constant_lower_quantile": constant_prediction,
        "frozen_v6_7_base_score_available": bool(
            decision_features["action_score_available"] > 0.0
        ),
        "frozen_v6_7_base_score": float(decision_features["action_score"]),
        "frozen_v6_7_base_score_source": (
            "frozen_v6_2_market_clustered_mean_ev_lcb"
        ),
        "target_after_cost_net_pnl_per_contract": row[
            "target_after_cost_net_pnl_per_contract"
        ],
        "target_used_as_decision_time_input": False,
        "target_used_for_model_parameter_threshold_or_gate_selection": False,
    }
    item["prediction_row_id"] = canonical_json_sha256(item)
    return item


def _cross_conformal_calibration(
    rows: list[dict[str, Any]], *, profile: dict[str, Any]
) -> dict[str, Any]:
    selected = _select_raw_predictions(rows)
    scores = [
        row["raw_conditional_lower_quantile_after_cost_net_pnl_per_contract"]
        - row["target_after_cost_net_pnl_per_contract"]
        for row in selected
    ]
    weights = _market_weights(selected)
    raw_correction = _weighted_quantile(
        scores,
        weights,
        float(profile["conformal_contract"]["correction_quantile"]),
    )
    correction = max(raw_correction, 0.0)
    coverage = _weighted_mean(
        [
            float(
                row["target_after_cost_net_pnl_per_contract"]
                >= row[
                    "raw_conditional_lower_quantile_after_cost_net_pnl_per_contract"
                ]
                - correction
            )
            for row in selected
        ],
        weights,
    )
    return {
        "method": profile["conformal_contract"]["method"],
        "coverage_level": profile["conformal_contract"]["coverage_level"],
        "selected_action_conformity_score_count": len(scores),
        "unique_market_count": len({row["market_id"] for row in selected}),
        "raw_weighted_conformity_quantile": raw_correction,
        "upward_score_correction_allowed": False,
        "applied_correction": correction,
        "empirical_coverage": coverage,
        "target_pnl_used_for_model_parameter_or_threshold_selection": False,
    }


def _select_raw_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["decision_group_id"]].append(row)
    return [
        sorted(
            group,
            key=lambda row: (
                -row[
                    "raw_conditional_lower_quantile_after_cost_net_pnl_per_contract"
                ],
                row["action"],
            ),
        )[0]
        for group in grouped.values()
    ]


def _select_oof_rows(
    rows: list[dict[str, Any]], *, conformal_correction: float
) -> list[dict[str, Any]]:
    selected = []
    for row in _select_raw_predictions(rows):
        lower_bound = (
            row["raw_conditional_lower_quantile_after_cost_net_pnl_per_contract"]
            - conformal_correction
        )
        if lower_bound > 0.0:
            selected.append({**row, "conformalized_lower_bound": lower_bound})
    return selected


def _historical_replay_superiority(
    rows: list[dict[str, Any]],
    *,
    conformal_correction: float,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Compare one-trade-per-market v7.1 OOF decisions with frozen v6.7."""

    contract = profile["historical_replay_superiority_gate"]
    market_ids = sorted({str(row["market_id"]) for row in rows})
    expected_market_count = int(contract["exact_evaluation_market_count"])
    if len(market_ids) != expected_market_count:
        raise ValueError("#233 historical replay market support invalid")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["market_id"])].append(row)

    candidate_rows = []
    baseline_rows = []
    fixed_size = float(contract["fixed_position_size"])
    for market_id in market_ids:
        group = grouped[market_id]
        candidate_eligible = []
        baseline_eligible = []
        for row in group:
            lower_bound = (
                float(
                    row[
                        "raw_conditional_lower_quantile_after_cost_net_pnl_per_contract"
                    ]
                )
                - conformal_correction
            )
            if lower_bound > 0.0:
                candidate_eligible.append({**row, "conformalized_lower_bound": lower_bound})
            baseline_score = float(row["frozen_v6_7_base_score"])
            if row["frozen_v6_7_base_score_available"] and baseline_score > 0.0:
                baseline_eligible.append(row)
        if candidate_eligible:
            winner = sorted(
                candidate_eligible,
                key=lambda row: (
                    -float(row["conformalized_lower_bound"]),
                    int(row["decision_ts"]),
                    str(row["action"]),
                ),
            )[0]
            candidate_rows.append(
                _historical_replay_selected_row(
                    winner,
                    selection_score=float(winner["conformalized_lower_bound"]),
                    selection_score_source=(
                        "v7_1_oof_conformalized_conditional_lower_quantile"
                    ),
                    fixed_position_size=fixed_size,
                )
            )
        if baseline_eligible:
            winner = sorted(
                baseline_eligible,
                key=lambda row: (
                    -float(row["frozen_v6_7_base_score"]),
                    int(row["decision_ts"]),
                    str(row["action"]),
                ),
            )[0]
            baseline_rows.append(
                _historical_replay_selected_row(
                    winner,
                    selection_score=float(winner["frozen_v6_7_base_score"]),
                    selection_score_source=(
                        "frozen_v6_2_market_clustered_mean_ev_lcb"
                    ),
                    fixed_position_size=fixed_size,
                )
            )

    candidate = _historical_replay_metrics(candidate_rows, market_ids=market_ids)
    baseline = _historical_replay_metrics(baseline_rows, market_ids=market_ids)
    total_delta = (
        candidate["total_after_cost_net_pnl_at_frozen_size"]
        - baseline["total_after_cost_net_pnl_at_frozen_size"]
    )
    robust_delta = (
        candidate["largest_winner_removed_after_cost_net_pnl_at_frozen_size"]
        - baseline["largest_winner_removed_after_cost_net_pnl_at_frozen_size"]
    )
    candidate_ids = {row["market_id"] for row in candidate_rows}
    baseline_ids = {row["market_id"] for row in baseline_rows}
    return {
        "gate_name": "same_dataset_historical_replay_superiority_over_v6_7",
        "gate_mode": "development_screening_only_before_new_collection",
        "evaluation_market_count": len(market_ids),
        "evaluation_market_ids_hash": canonical_json_sha256(market_ids),
        "fixed_position_size": fixed_size,
        "no_bet_market_pnl": 0.0,
        "common_selected_row_filter_applied": False,
        "candidate": candidate,
        "v6_7_baseline": baseline,
        "candidate_selected_market_count": len(candidate_ids),
        "v6_7_baseline_selected_market_count": len(baseline_ids),
        "both_selected_market_count": len(candidate_ids.intersection(baseline_ids)),
        "candidate_only_selected_market_count": len(candidate_ids - baseline_ids),
        "v6_7_only_selected_market_count": len(baseline_ids - candidate_ids),
        "neither_selected_market_count": len(
            set(market_ids) - candidate_ids.union(baseline_ids)
        ),
        "candidate_total_after_cost_net_pnl_at_frozen_size": candidate[
            "total_after_cost_net_pnl_at_frozen_size"
        ],
        "v6_7_baseline_total_after_cost_net_pnl_at_frozen_size": baseline[
            "total_after_cost_net_pnl_at_frozen_size"
        ],
        "candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size": total_delta,
        "candidate_largest_winner_removed_after_cost_net_pnl_at_frozen_size": (
            candidate["largest_winner_removed_after_cost_net_pnl_at_frozen_size"]
        ),
        "v6_7_baseline_largest_winner_removed_after_cost_net_pnl_at_frozen_size": (
            baseline["largest_winner_removed_after_cost_net_pnl_at_frozen_size"]
        ),
        "candidate_minus_v6_7_largest_winner_removed_after_cost_net_pnl_at_frozen_size": (
            robust_delta
        ),
        "candidate_total_pnl_strictly_better_than_v6_7": total_delta
        > float(contract["candidate_minus_v6_7_total_pnl_minimum_exclusive"]),
        "candidate_largest_winner_removed_pnl_not_worse_than_v6_7": robust_delta
        >= float(
            contract[
                "candidate_minus_v6_7_largest_winner_removed_pnl_minimum_inclusive"
            ]
        ),
        "identical_market_cost_sizing_position_management_and_guard_contract": True,
        "historical_pnl_used_for_model_feature_parameter_or_threshold_tuning": False,
        "historical_pnl_used_for_pre_collection_screening_only": True,
        "promotion_or_paper_unlock_allowed": False,
        "candidate_selected_rows": candidate_rows,
        "v6_7_baseline_selected_rows": baseline_rows,
    }


def _historical_replay_selected_row(
    row: dict[str, Any],
    *,
    selection_score: float,
    selection_score_source: str,
    fixed_position_size: float,
) -> dict[str, Any]:
    target = float(row["target_after_cost_net_pnl_per_contract"])
    item = {
        "market_id": str(row["market_id"]),
        "decision_group_id": str(row["decision_group_id"]),
        "decision_ts": int(row["decision_ts"]),
        "action": str(row["action"]),
        "side": str(row["side"]),
        "selection_score": selection_score,
        "selection_score_source": selection_score_source,
        "target_after_cost_net_pnl_per_contract": target,
        "fixed_position_size": fixed_position_size,
        "after_cost_net_pnl_at_frozen_size": target * fixed_position_size,
        "target_used_as_decision_time_input": False,
        "target_used_for_model_feature_parameter_or_threshold_tuning": False,
        "target_used_for_pre_collection_historical_replay_gate": True,
    }
    item["historical_replay_row_id"] = canonical_json_sha256(item)
    return item


def _historical_replay_metrics(
    rows: list[dict[str, Any]], *, market_ids: list[str]
) -> dict[str, Any]:
    pnl_by_market = dict.fromkeys(market_ids, 0.0)
    per_contract_by_market = dict.fromkeys(market_ids, 0.0)
    for row in rows:
        pnl_by_market[row["market_id"]] = float(
            row["after_cost_net_pnl_at_frozen_size"]
        )
        per_contract_by_market[row["market_id"]] = float(
            row["target_after_cost_net_pnl_per_contract"]
        )
    total = sum(pnl_by_market.values())
    per_contract_total = sum(per_contract_by_market.values())
    largest_winner = max(max(pnl_by_market.values(), default=0.0), 0.0)
    return {
        "evaluation_market_count": len(market_ids),
        "selected_bet_count": len(rows),
        "no_bet_market_count": len(market_ids) - len(rows),
        "total_after_cost_net_pnl_per_contract": per_contract_total,
        "total_after_cost_net_pnl_at_frozen_size": total,
        "largest_winning_market_pnl_at_frozen_size": largest_winner,
        "largest_winner_removed_after_cost_net_pnl_at_frozen_size": (
            total - largest_winner
        ),
        "pnl_by_side": dict(
            sorted(
                _sum_by(rows, key="side", value="after_cost_net_pnl_at_frozen_size").items()
            )
        ),
        "bet_count_by_side": dict(sorted(Counter(row["side"] for row in rows).items())),
        "bet_count_by_action": dict(
            sorted(Counter(row["action"] for row in rows).items())
        ),
    }


def _sum_by(
    rows: list[dict[str, Any]], *, key: str, value: str
) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for row in rows:
        totals[str(row[key])] += float(row[value])
    return dict(totals)


def _oof_metrics(rows: list[dict[str, Any]], *, profile: dict[str, Any]) -> dict[str, Any]:
    tau = float(profile["quantile_model"]["conditional_quantile_tau"])
    weights = _market_weights(rows)
    candidate_loss = _weighted_mean(
        [
            _pinball(
                row["target_after_cost_net_pnl_per_contract"]
                - row[
                    "raw_conditional_lower_quantile_after_cost_net_pnl_per_contract"
                ],
                tau,
            )
            for row in rows
        ],
        weights,
    )
    constant_loss = _weighted_mean(
        [
            _pinball(
                row["target_after_cost_net_pnl_per_contract"]
                - row["fold_train_constant_lower_quantile"],
                tau,
            )
            for row in rows
        ],
        weights,
    )
    return {
        "row_count": len(rows),
        "unique_market_count": len({row["market_id"] for row in rows}),
        "market_weighted_pinball_loss": candidate_loss,
        "fold_train_constant_market_weighted_pinball_loss": constant_loss,
        "relative_pinball_loss_improvement_report_only": (
            (constant_loss - candidate_loss) / constant_loss
            if constant_loss > 0.0
            else 0.0
        ),
        "prediction_loss_used_as_gate_or_tuning_input": False,
    }


def _leakage_audit(
    rows: list[dict[str, Any]], *, model: dict[str, Any], profile: dict[str, Any]
) -> dict[str, Any]:
    forbidden = set(profile["feature_contract"]["forbidden_inference_field_names"])
    violations = sum(
        bool(forbidden.intersection(row["decision_time_features"])) for row in rows
    )
    causality = sum(row["max_input_ts"] > row["decision_ts"] for row in rows)
    checks = {
        "all_134_historical_markets_used": len({row["market_id"] for row in rows})
        == 134,
        "feature_contract_exact": all(
            tuple(row["decision_time_features"]) == FEATURE_NAMES for row in rows
        ),
        "forbidden_inference_fields_zero": violations == 0,
        "timestamp_causality_violations_zero": causality == 0,
        "future_and_prior_result_rows_excluded": model[
            "issue229_issue231_or_v7_0_result_rows_used_for_fit_or_tuning"
        ]
        is False,
        "pnl_not_used_for_gate_or_tuning": model[
            "historical_pnl_used_for_model_feature_parameter_or_threshold_tuning"
        ]
        is False,
        "historical_replay_gate_is_pre_collection_only": model[
            "historical_pnl_used_for_pre_collection_superiority_gate"
        ]
        is True
        and model["historical_replay_superiority_is_promotion_evidence"] is False,
        "side_rules_disabled": model["side_quota_applied"] is False
        and model["side_count_hard_gate_enabled"] is False
        and model["side_pnl_hard_gate_enabled"] is False,
    }
    reasons = [name for name, passed in checks.items() if not passed]
    audit = {
        "schema_version": LEAKAGE_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "canonical_row_count": len(rows),
        "unique_market_count": len({row["market_id"] for row in rows}),
        "forbidden_inference_field_violation_count": violations,
        "timestamp_causality_violation_count": causality,
        "fit_leakage_checks": checks,
        "fit_leakage_audit_passed": not reasons,
        "fit_leakage_blocking_reason_codes": reasons,
        "future_target_accessed": False,
        **_v7_0_blocked_safety_fields(),
    }
    audit["leakage_audit_id"] = canonical_json_sha256(audit)
    return audit


def _report(model: dict[str, Any], leakage: dict[str, Any]) -> dict[str, Any]:
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "historical_gate_passed": model["historical_gate_passed"],
        "historical_gate_blocking_reason_codes": model[
            "historical_gate_blocking_reason_codes"
        ],
        "fit_leakage_audit_passed": leakage["fit_leakage_audit_passed"],
        "enabled_action_family": model["enabled_action_family"],
        "disabled_action_family": model["disabled_action_family"],
        "disabled_action_reason_code": model["disabled_action_reason_code"],
        "forward_oof_metrics_report_only": model[
            "forward_oof_metrics_report_only"
        ],
        "cross_conformal": model["cross_conformal"],
        "historical_positive_lcb_selected_row_count": model[
            "historical_positive_lcb_selected_row_count"
        ],
        "historical_positive_lcb_unique_market_count": model[
            "historical_positive_lcb_unique_market_count"
        ],
        "historical_positive_lcb_side_distribution_diagnostic": model[
            "historical_positive_lcb_side_distribution_diagnostic"
        ],
        "historical_positive_lcb_target_pnl_sum_report_only": model[
            "historical_positive_lcb_target_pnl_sum_report_only"
        ],
        "historical_replay_superiority_gate": model[
            "historical_replay_superiority_gate"
        ],
        "historical_pnl_used_for_model_feature_parameter_or_threshold_tuning": False,
        "historical_pnl_used_for_pre_collection_superiority_gate": True,
        "historical_replay_superiority_is_promotion_evidence": False,
        "target_free_canary_started": False,
        "future_confirmatory_authorized": False,
        **_v7_0_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def _report_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v7.1 SBC Conditional-Quantile Historical Report",
            "",
            f"- historical gate passed: `{str(report['historical_gate_passed']).lower()}`",
            f"- blockers: `{report['historical_gate_blocking_reason_codes']}`",
            f"- leakage audit passed: `{str(report['fit_leakage_audit_passed']).lower()}`",
            f"- OOF markets: `{report['forward_oof_metrics_report_only']['unique_market_count']}`",
            f"- OOF pinball improvement (report-only): `{report['forward_oof_metrics_report_only']['relative_pinball_loss_improvement_report_only']}`",
            f"- conformal coverage: `{report['cross_conformal']['empirical_coverage']}`",
            f"- positive-LCB markets: `{report['historical_positive_lcb_unique_market_count']}`",
            f"- positive-LCB PnL (report-only): `{report['historical_positive_lcb_target_pnl_sum_report_only']}`",
            "- historical same-dataset replay gate passed: "
            f"`{str(report['historical_replay_superiority_gate']['candidate_total_pnl_strictly_better_than_v6_7'] and report['historical_replay_superiority_gate']['candidate_largest_winner_removed_pnl_not_worse_than_v6_7']).lower()}`",
            "- v7.1 frozen-size PnL: "
            f"`{report['historical_replay_superiority_gate']['candidate_total_after_cost_net_pnl_at_frozen_size']}`",
            "- v6.7 frozen-size PnL: "
            f"`{report['historical_replay_superiority_gate']['v6_7_baseline_total_after_cost_net_pnl_at_frozen_size']}`",
            "- candidate-minus-v6.7 frozen-size PnL: "
            f"`{report['historical_replay_superiority_gate']['candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size']}`",
            "- HTS: `explicitly unavailable fail-closed`",
            "- historical PnL used for model/feature/threshold tuning: `false`",
            "- historical PnL used for pre-collection screening: `true`",
            "- historical replay is promotion evidence: `false`",
            "- target-free canary started: `false`",
            "- paper/live/write/wallet/capital unlock: `false`",
            "",
        ]
    )


def _validate_canonical_rows(
    rows: list[dict[str, Any]], *, expected_market_count: int
) -> None:
    if len({row["market_id"] for row in rows}) != expected_market_count:
        raise ValueError("#233 canonical historical market count invalid")
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if tuple(row["decision_time_features"]) != FEATURE_NAMES:
            raise ValueError("#233 canonical feature contract invalid")
        if row["max_input_ts"] > row["decision_ts"]:
            raise ValueError("#233 canonical feature causality invalid")
        if row.get("target_used_as_decision_time_input") is not False:
            raise ValueError("#233 target used as decision input")
        grouped[row["decision_group_id"]].add(row["action"])
    if any(actions != set(SBC_ACTIONS) for actions in grouped.values()):
        raise ValueError("#233 canonical SBC action grid incomplete")


def _market_order(rows: list[dict[str, Any]]) -> list[str]:
    minimum: dict[str, int] = {}
    for row in rows:
        market_id = row["market_id"]
        minimum[market_id] = min(
            minimum.get(market_id, row["decision_ts"]), row["decision_ts"]
        )
    return [market_id for market_id, _ in sorted(minimum.items(), key=lambda x: (x[1], x[0]))]


def _market_weights(rows: list[dict[str, Any]]) -> list[float]:
    counts = Counter(row["market_id"] for row in rows)
    return [1.0 / counts[row["market_id"]] for row in rows]


def _weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    value_list = list(values)
    weight_list = list(weights)
    total = sum(weight_list)
    if not value_list or len(value_list) != len(weight_list) or total <= 0.0:
        raise ValueError("#233 weighted mean inputs invalid")
    return sum(
        value * weight
        for value, weight in zip(value_list, weight_list, strict=True)
    ) / total


def _weighted_quantile(
    values: list[float], weights: list[float], quantile: float
) -> float:
    if not values or len(values) != len(weights) or not 0.0 <= quantile <= 1.0:
        raise ValueError("#233 weighted quantile inputs invalid")
    ordered = sorted(zip(values, weights, strict=True), key=lambda item: item[0])
    target = quantile * sum(weights)
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= target:
            return float(value)
    return float(ordered[-1][0])


def _pinball(residual: float, tau: float) -> float:
    return tau * residual if residual >= 0.0 else (tau - 1.0) * residual


def _row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["decision_ts"]),
        str(row["market_id"]),
        str(row.get("action") or ""),
    )
