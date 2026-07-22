"""Historical-only fitting for the issue #232 abstention-aware v7.0 ranker."""

from __future__ import annotations

import math
import shutil
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_abstention_aware_expected_net_pnl_v7_0 import (
    CANDIDATE_NAME,
    FORBIDDEN_INFERENCE_FIELDS,
    FULL_ACTION_GRID,
    _v7_0_blocked_safety_fields,
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

TRAINING_PROFILE_SCHEMA_VERSION = (
    "bigan-v8-abstention-aware-expected-net-pnl-v7-0-training-profile-v1"
)
MODEL_SCHEMA_VERSION = "bigan-v8-abstention-aware-expected-net-pnl-v7-0-model-v1"
REPORT_SCHEMA_VERSION = "bigan-v8-abstention-aware-expected-net-pnl-v7-0-fit-report-v1"
LEAKAGE_SCHEMA_VERSION = "bigan-v8-abstention-aware-v7-0-fit-leakage-audit-v1"
MANIFEST_SCHEMA_VERSION = "bigan-v8-abstention-aware-v7-0-fit-manifest-v1"
FAMILIES = ("SELL_BEFORE_CLOSE", "HOLD_TO_SETTLEMENT")
SBC_ACTIONS = ("BUY_UP_SELL_BEFORE_CLOSE", "BUY_DOWN_SELL_BEFORE_CLOSE")
HTS_ACTIONS = ("BUY_UP_HOLD_TO_SETTLEMENT", "BUY_DOWN_HOLD_TO_SETTLEMENT")
FEATURE_NAMES = (
    "action_score_available",
    "action_score",
    "action_score_margin",
    "btc_anchor_direction",
    "selected_side_probability",
    "execution_price",
    "selected_side_probability_minus_execution_price",
    "log1p_spread_bps",
    "queue_fill_shortfall",
    "log1p_book_staleness_ms",
    "late_window_pressure",
    "pre_entry_market_exposure",
    "same_side_prior_entry",
    "side_flip_prior_entry",
    "side_is_up",
)
FROZEN_TRAINING_LINEAGE = {
    "lineage_profile_sha256": (
        "7ff6e5d95f7502b0b07c188354cdbb09798e115cdf68d148ebc1e02f231db808"
    ),
    "lineage_audit_manifest_sha256": (
        "9c92e51fe1e8d003fceb5197d04a7e099b48fcc45749b1b74b17f48bb7d593e4"
    ),
    "runtime_target_rows_sha256": (
        "1565116daeb2f5d4d8c33fefa507276f59251edd5ffb5f4f313041bcf9dbb0ec"
    ),
    "full_action_grid_rows_sha256": (
        "fc20e07801743d7f62640bb3a99942ea43cf19e7f4b16770b80053886ae6043a"
    ),
    "issue229_selected_window_rows_forbidden_from_fit_sha256": (
        "8c1a5b92ccd4657bdd6d064cbc45af39d208d944c2b763419597500a7dde48fa"
    ),
    "issue231_selected_window_rows_forbidden_from_fit_sha256": (
        "90cd57f9aa557e264d34d14084c4e4d7811ecbc4f81b9e8799cde1e568e01bbb"
    ),
    "issue231_settled_index_forbidden_sha256": (
        "fac02081f288b58b36a871b231ce186bb8d442b879bb6f614b8a31d495acf826"
    ),
    "issue231_execution_pnl_report_forbidden_sha256": (
        "3a1dcf58dbd5ec059cc010baa3b2c595a98d57dddf235415fb7d8fd859bba20a"
    ),
}


@dataclass(frozen=True, slots=True)
class AbstentionAwareV70FitConfig:
    """Pinned historical-only fit inputs."""

    run_id: str
    output_dir: Path | str
    training_profile_path: Path | str
    expected_training_profile_sha256: str
    lineage_audit_manifest_path: Path | str
    runtime_target_rows_path: Path | str
    full_action_grid_rows_path: Path | str
    implementation_commit: str
    fit_created_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(
            self.expected_training_profile_sha256,
            "expected_training_profile_sha256",
        )
        _require_git_sha(self.implementation_commit)
        if self.fit_created_ts <= 0:
            raise ValueError("fit_created_ts must be positive")
        for name in (
            "output_dir",
            "training_profile_path",
            "lineage_audit_manifest_path",
            "runtime_target_rows_path",
            "full_action_grid_rows_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


def validate_v7_0_training_profile(profile: dict[str, Any]) -> None:
    """Enforce the pre-fit model, split, calibration, and safety contract."""

    splits = dict(profile.get("family_splits") or {})
    sbc = dict(splits.get("SELL_BEFORE_CLOSE") or {})
    hts = dict(splits.get("HOLD_TO_SETTLEMENT") or {})
    no_trade = dict(splits.get("NO_TRADE") or {})
    features = dict(profile.get("feature_contract") or {})
    model = dict(profile.get("model_contract") or {})
    calibration = dict(profile.get("calibration_contract") or {})
    gates = dict(profile.get("historical_fit_gates") or {})
    selection = dict(profile.get("selection_contract") or {})
    future = dict(profile.get("future_contract") or {})
    checks = {
        "identity": profile.get("schema_version") == TRAINING_PROFILE_SCHEMA_VERSION
        and profile.get("issue_number") == 232
        and profile.get("candidate_name") == CANDIDATE_NAME
        and profile.get("preregistered_before_model_fit") is True,
        "lineage": profile.get("lineage") == FROZEN_TRAINING_LINEAGE,
        "sbc_split": sbc
        == {
            "source": "runtime_aligned_after_cost_exit_policy_target",
            "fit_role": "development_train",
            "fit_market_count": 89,
            "calibration_role": "development_calibration",
            "calibration_market_count": 45,
            "forward_oof_initial_fit_market_count": 44,
            "forward_oof_validation_market_count_per_fold": 15,
            "forward_oof_fold_count": 3,
        },
        "hts_split": hts
        == {
            "source": "historical_full_action_grid_after_cost_target",
            "split_method": (
                "market_grouped_chronological_by_min_decision_ts_then_market_id"
            ),
            "total_market_count": 65,
            "fit_market_count": 44,
            "calibration_market_count": 21,
            "forward_oof_initial_fit_market_count": 22,
            "forward_oof_validation_market_count_per_fold": 11,
            "forward_oof_fold_count": 2,
        },
        "no_trade": no_trade
        == {"source": "deterministic_zero_target", "score": 0.0},
        "features": tuple(features.get("feature_names") or ()) == FEATURE_NAMES
        and features.get("fit_only_standardization") is True
        and features.get("market_probability_usage")
        == "market_price_value_conditioning_only_not_direct_fair_value"
        and set(features.get("forbidden_inference_field_names") or ())
        == FORBIDDEN_INFERENCE_FIELDS,
        "model": model
        == {
            "model_family": (
                "market_weighted_family_specific_ridge_with_unpenalized_intercept"
            ),
            "ridge_alpha": 100.0,
            "coefficient_absolute_bound": 8.0,
            "market_weighting": "each_market_total_weight_one_within_family",
            "hyperparameter_search_enabled": False,
            "feature_selection_enabled": False,
            "validation_labels_used_for_model_fit": False,
            "validation_labels_used_for_hyperparameter_or_feature_selection": False,
            "current_oof_or_validation_pnl_used_for_tuning": False,
            "issue229_or_issue231_rows_used_for_fit_or_tuning": False,
            "result_selected_rerun_allowed": False,
        },
        "calibration": calibration
        == {
            "method": (
                "market_weighted_one_sided_selected_action_within_family_"
                "residual_quantile"
            ),
            "coverage_level": 0.8,
            "residual_quantile": 0.2,
            "selection_threshold": 0.0,
            "threshold_operator": "strictly_greater_than",
            "minimum_calibration_market_count_per_family": 20,
            "minimum_empirical_selected_action_coverage": 0.75,
            "calibration_labels_used_for_uncertainty_calibration_only": True,
            "calibration_pnl_report_only_not_gate_or_tuning_input": True,
        },
        "gates": gates
        == {
            "forward_oof_relative_mae_improvement_over_fold_train_mean_must_exceed": 0.0,
            "forward_oof_relative_mse_improvement_over_fold_train_mean_must_exceed": 0.0,
            "coefficient_stability_method": "leave_one_market_out",
            "maximum_leave_one_market_out_coefficient_absolute_delta": 4.0,
            "all_coefficients_finite_and_bounded": True,
            "all_family_calibration_coverage_gates_required": True,
            "historical_selected_pnl_is_report_only": True,
        },
        "selection": selection
        == {
            "decision_grid": "full_five_action_grid",
            "select_highest_positive_calibrated_lower_bound_else_no_trade": True,
            "no_trade_score": 0.0,
            "side_quota_allowed": False,
            "side_count_hard_gate_enabled": False,
            "side_pnl_hard_gate_enabled": False,
            "action_family_pnl_hard_gate_enabled": False,
            "side_composition_is_regime_emergent": True,
            "unsupported_or_missing_feature_behavior": (
                "fail_closed_to_no_trade_with_explicit_reason"
            ),
        },
        "future": future
        == {
            "new_strictly_later_disjoint_outcome_blind_window_required": True,
            "minimum_target_free_guard_accepted_unique_market_count_total": 40,
            "minimum_target_free_guard_accepted_unique_market_count_per_side": None,
            "side_metrics_diagnostic_only": True,
            "single_use_official_read_only_settlement_gate_required": True,
            "separate_explicit_paper_candidate_approval_required": True,
        },
        "safety": profile.get("safety") == _v7_0_blocked_safety_fields(),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("#232 v7.0 training profile invalid: " + ", ".join(blockers))


def materialize_v7_0_sbc_rows(
    source_rows: list[dict[str, Any]], profile: dict[str, Any]
) -> list[dict[str, Any]]:
    """Map all runtime-aligned SBC rows, including missing-score examples."""

    validate_v7_0_training_profile(profile)
    if not source_rows:
        raise ValueError("#232 runtime-aligned SBC rows are empty")
    split = profile["family_splits"]["SELL_BEFORE_CLOSE"]
    expected_roles = {split["fit_role"], split["calibration_role"]}
    output = []
    for row in source_rows:
        role = str(row.get("role") or "")
        action = str(row.get("action") or "")
        side = str(row.get("side") or "")
        if role not in expected_roles or action not in SBC_ACTIONS:
            raise ValueError("#232 runtime-aligned SBC identity invalid")
        if side not in {"UP", "DOWN"} or f"BUY_{side}_" not in action:
            raise ValueError("#232 runtime-aligned SBC side/action mismatch")
        if int(row["max_input_ts"]) > int(row["decision_ts"]):
            raise ValueError("#232 runtime-aligned SBC feature causality violation")
        if row.get("target_used_as_decision_time_input") is not False:
            raise ValueError("#232 runtime-aligned target used as input")
        if row.get("target_available_only_post_exit_or_official_resolution") is not True:
            raise ValueError("#232 runtime-aligned target provenance invalid")
        item = _canonical_training_row(
            source="runtime_aligned_sbc",
            market_id=str(row["market_id"]),
            decision_ts=int(row["decision_ts"]),
            max_input_ts=int(row["max_input_ts"]),
            role=role,
            family="SELL_BEFORE_CLOSE",
            action=action,
            side=side,
            values=_sbc_feature_values(row, profile),
            target=float(row["runtime_policy_after_cost_net_pnl_per_contract"]),
        )
        output.append(item)
    _validate_materialized_family_grid(output, actions=set(SBC_ACTIONS))
    role_market_counts = {
        role: len({row["market_id"] for row in output if row["role"] == role})
        for role in expected_roles
    }
    if role_market_counts != {
        split["fit_role"]: split["fit_market_count"],
        split["calibration_role"]: split["calibration_market_count"],
    }:
        raise ValueError("#232 runtime-aligned SBC role market counts invalid")
    return sorted(output, key=_row_sort_key)


def materialize_v7_0_hts_rows(
    source_rows: list[dict[str, Any]], profile: dict[str, Any]
) -> list[dict[str, Any]]:
    """Materialize both HTS alternatives from every complete historical grid."""

    validate_v7_0_training_profile(profile)
    if not source_rows:
        raise ValueError("#232 full action-grid rows are empty")
    split = profile["family_splits"]["HOLD_TO_SETTLEMENT"]
    market_order = _chronological_market_order(source_rows)
    if len(market_order) != int(split["total_market_count"]):
        raise ValueError("#232 full action-grid market count invalid")
    fit_ids = set(market_order[: int(split["fit_market_count"])])
    calibration_ids = set(market_order[int(split["fit_market_count"]) :])
    if len(calibration_ids) != int(split["calibration_market_count"]):
        raise ValueError("#232 HTS chronological calibration support invalid")
    output = []
    for row in source_rows:
        market_id = str(row.get("market_id") or "")
        decision_ts = int(row["decision_ts"])
        if int(row["max_input_ts"]) > decision_ts:
            raise ValueError("#232 full action-grid feature causality violation")
        targets = dict(row.get("evaluation_target_net_pnl_per_contract_by_action") or {})
        if set(targets) != FULL_ACTION_GRID:
            raise ValueError("#232 historical five-action target grid incomplete")
        if row.get("target_outcome_available_only_post_resolution") is not True:
            raise ValueError("#232 historical HTS target provenance invalid")
        if row.get("target_provenance", {}).get("outcome_used_as_decision_input") is not False:
            raise ValueError("#232 historical outcome used as decision input")
        ranking = _ranking_by_action(row)
        role = "development_train" if market_id in fit_ids else "development_calibration"
        for action in HTS_ACTIONS:
            side = "UP" if "_UP_" in action else "DOWN"
            item = _canonical_training_row(
                source="historical_full_action_grid",
                market_id=market_id,
                decision_ts=decision_ts,
                max_input_ts=int(row["max_input_ts"]),
                role=role,
                family="HOLD_TO_SETTLEMENT",
                action=action,
                side=side,
                values=_hts_feature_values(row, ranking=ranking, action=action, profile=profile),
                target=float(targets[action]),
            )
            output.append(item)
    _validate_materialized_family_grid(output, actions=set(HTS_ACTIONS))
    return sorted(output, key=_row_sort_key)


def fit_abstention_aware_v7_0(
    *,
    sbc_rows: list[dict[str, Any]],
    hts_rows: list[dict[str, Any]],
    profile: dict[str, Any],
    implementation_commit: str,
    fit_created_ts: int,
) -> dict[str, Any]:
    """Fit fixed family models, forward OOF diagnostics, and uncertainty bounds."""

    validate_v7_0_training_profile(profile)
    rows_by_family = {
        "SELL_BEFORE_CLOSE": sbc_rows,
        "HOLD_TO_SETTLEMENT": hts_rows,
    }
    family_artifacts: dict[str, Any] = {}
    oof_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    family_gate_results = {}
    for family in FAMILIES:
        split = profile["family_splits"][family]
        family_rows = rows_by_family[family]
        fit_rows = [row for row in family_rows if row["role"] == "development_train"]
        calibration_source_rows = [
            row for row in family_rows if row["role"] == "development_calibration"
        ]
        family_oof, oof_metrics = _forward_oof(fit_rows, split=split, profile=profile)
        final_model = _fit_family_model(fit_rows, profile=profile)
        stability = _coefficient_stability(
            fit_rows,
            final_model=final_model,
            profile=profile,
        )
        calibrated, calibration = _calibrate_family(
            calibration_source_rows,
            model=final_model,
            profile=profile,
        )
        gates = profile["historical_fit_gates"]
        calibration_contract = profile["calibration_contract"]
        checks = {
            "forward_oof_mae_improved": oof_metrics[
                "relative_mae_improvement_over_fold_train_mean"
            ]
            > float(
                gates[
                    "forward_oof_relative_mae_improvement_over_fold_train_mean_must_exceed"
                ]
            ),
            "forward_oof_mse_improved": oof_metrics[
                "relative_mse_improvement_over_fold_train_mean"
            ]
            > float(
                gates[
                    "forward_oof_relative_mse_improvement_over_fold_train_mean_must_exceed"
                ]
            ),
            "coefficients_finite_and_bounded": final_model[
                "coefficients_finite_and_bounded"
            ],
            "coefficient_stability_passed": stability[
                "maximum_coefficient_absolute_delta"
            ]
            <= float(
                gates["maximum_leave_one_market_out_coefficient_absolute_delta"]
            ),
            "calibration_market_support_passed": calibration[
                "unique_market_count"
            ]
            >= int(
                calibration_contract["minimum_calibration_market_count_per_family"]
            ),
            "calibration_coverage_passed": calibration[
                "empirical_selected_action_lower_bound_coverage"
            ]
            >= float(
                calibration_contract["minimum_empirical_selected_action_coverage"]
            ),
        }
        reasons = [
            f"{family.lower()}_{name}_failed"
            for name, passed in checks.items()
            if not passed
        ]
        family_artifacts[family] = {
            "family": family,
            "fit_market_count": len({row["market_id"] for row in fit_rows}),
            "fit_row_count": len(fit_rows),
            "calibration_market_count": calibration["unique_market_count"],
            "calibration_row_count": len(calibration_source_rows),
            "feature_names": list(FEATURE_NAMES),
            "model": final_model,
            "forward_oof_metrics": oof_metrics,
            "coefficient_stability": stability,
            "calibration": calibration,
            "historical_fit_gate_checks": checks,
            "historical_fit_gate_passed": not reasons,
            "historical_fit_gate_blocking_reason_codes": reasons,
        }
        family_gate_results[family] = {
            "passed": not reasons,
            "reason_codes": reasons,
        }
        oof_rows.extend(family_oof)
        calibration_rows.extend(calibrated)
    blockers = [
        reason
        for family in FAMILIES
        for reason in family_gate_results[family]["reason_codes"]
    ]
    model_artifact = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": implementation_commit,
        "fit_created_ts": fit_created_ts,
        "frozen_historical_model": not blockers,
        "decision_time_safe": True,
        "ranking_score_source": "model_predicted_after_cost_net_pnl_lower_bound",
        "full_five_action_grid_required": True,
        "no_trade_score": 0.0,
        "family_models": family_artifacts,
        "family_gate_results": family_gate_results,
        "historical_development_gate_passed": not blockers,
        "historical_development_blocking_reason_codes": blockers,
        "validation_or_oof_pnl_used_for_tuning_or_gate": False,
        "issue229_or_issue231_rows_used_for_fit_or_tuning": False,
        "issue229_or_issue231_outcomes_opened": False,
        "side_quota_applied": False,
        "side_count_hard_gate_enabled": False,
        "side_pnl_hard_gate_enabled": False,
        "new_strictly_later_future_confirmatory_required": True,
        **_v7_0_blocked_safety_fields(),
    }
    model_artifact["model_artifact_id"] = canonical_json_sha256(model_artifact)
    return {
        "model_artifact": model_artifact,
        "oof_rows": sorted(oof_rows, key=_row_sort_key),
        "calibration_rows": sorted(calibration_rows, key=_row_sort_key),
    }


def score_v7_0_decision_group(
    action_rows: list[dict[str, Any]], *, model_artifact: dict[str, Any]
) -> dict[str, Any]:
    """Score four trade alternatives and abstain unless the best LCB is positive."""

    reason_codes = []
    if model_artifact.get("schema_version") != MODEL_SCHEMA_VERSION:
        reason_codes.append("v7_0_model_artifact_schema_invalid")
    if model_artifact.get("historical_development_gate_passed") is not True:
        reason_codes.append("v7_0_historical_development_gate_not_passed")
    actions = {str(row.get("action") or "") for row in action_rows}
    expected_actions = set(SBC_ACTIONS).union(HTS_ACTIONS)
    if actions != expected_actions or len(action_rows) != len(expected_actions):
        reason_codes.append("v7_0_trade_action_grid_incomplete")
    decision_group_ids = {str(row.get("decision_group_id") or "") for row in action_rows}
    if len(decision_group_ids) != 1 or "" in decision_group_ids:
        reason_codes.append("v7_0_decision_group_identity_invalid")
    scored = []
    if not reason_codes:
        for row in action_rows:
            if FORBIDDEN_INFERENCE_FIELDS.intersection(row):
                reason_codes.append("v7_0_forbidden_outcome_field_in_inference_row")
                break
            features = dict(row.get("decision_time_features") or {})
            if tuple(features) != FEATURE_NAMES:
                reason_codes.append("v7_0_decision_time_feature_contract_invalid")
                break
            if int(row["max_input_ts"]) > int(row["decision_ts"]):
                reason_codes.append("v7_0_inference_feature_causality_failed")
                break
            family = str(row.get("action_family") or "")
            family_artifact = dict(
                (model_artifact.get("family_models") or {}).get(family) or {}
            )
            model = dict(family_artifact.get("model") or {})
            calibration = dict(family_artifact.get("calibration") or {})
            if not model or "residual_quantile" not in calibration:
                reason_codes.append("v7_0_family_model_or_calibrator_missing")
                break
            prediction = _predict_model(row, model)
            lower_bound = prediction + float(calibration["residual_quantile"])
            scored.append(
                {
                    "action": row["action"],
                    "action_family": family,
                    "side": row["side"],
                    "model_predicted_after_cost_net_pnl_per_contract": prediction,
                    "calibrated_lower_bound_after_cost_net_pnl_per_contract": (
                        lower_bound
                    ),
                }
            )
    if reason_codes:
        selected_action = "NO_TRADE"
        selected_family = "NO_TRADE"
        selected_side = "NONE"
    else:
        best = sorted(
            scored,
            key=lambda row: (
                -row["calibrated_lower_bound_after_cost_net_pnl_per_contract"],
                row["action"],
            ),
        )[0]
        if best["calibrated_lower_bound_after_cost_net_pnl_per_contract"] > 0.0:
            selected_action = best["action"]
            selected_family = best["action_family"]
            selected_side = best["side"]
        else:
            selected_action = "NO_TRADE"
            selected_family = "NO_TRADE"
            selected_side = "NONE"
            reason_codes.append("v7_0_no_positive_calibrated_lower_bound")
    result = {
        "decision_group_id": next(iter(decision_group_ids), ""),
        "ranking_score_source": "model_predicted_after_cost_net_pnl_lower_bound",
        "selected_action": selected_action,
        "selected_action_family": selected_family,
        "selected_side": selected_side,
        "trade_selected": selected_action != "NO_TRADE",
        "action_scores": scored,
        "selection_reason_codes": reason_codes,
        "source_score_mutated": False,
        "outcome_or_pnl_field_used_at_inference": False,
        **_v7_0_blocked_safety_fields(),
    }
    result["decision_id"] = canonical_json_sha256(result)
    return result


def run_abstention_aware_v7_0_fit(config: AbstentionAwareV70FitConfig) -> dict[str, Any]:
    """Verify pins, fit once, and emit model, audit, report, and manifest artifacts."""

    paths = {
        "training_profile": Path(config.training_profile_path).resolve(),
        "lineage_audit_manifest": Path(config.lineage_audit_manifest_path).resolve(),
        "runtime_target_rows": Path(config.runtime_target_rows_path).resolve(),
        "full_action_grid_rows": Path(config.full_action_grid_rows_path).resolve(),
    }
    _verify_pin(
        paths["training_profile"],
        config.expected_training_profile_sha256,
        "#232 v7.0 training profile",
    )
    profile = _load_json(paths["training_profile"])
    validate_v7_0_training_profile(profile)
    lineage = profile["lineage"]
    _verify_pin(
        paths["lineage_audit_manifest"],
        lineage["lineage_audit_manifest_sha256"],
        "#232 lineage audit manifest",
    )
    _verify_pin(
        paths["runtime_target_rows"],
        lineage["runtime_target_rows_sha256"],
        "#232 runtime target rows",
    )
    _verify_pin(
        paths["full_action_grid_rows"],
        lineage["full_action_grid_rows_sha256"],
        "#232 full action-grid rows",
    )
    lineage_manifest = _load_json(paths["lineage_audit_manifest"])
    if lineage_manifest.get("lineage_audit_passed") is not True:
        raise ValueError("#232 lineage audit did not pass")
    if lineage_manifest.get("forbidden_future_outcome_artifacts_opened") is not False:
        raise ValueError("#232 forbidden future outcome artifact was opened")

    sbc_rows = materialize_v7_0_sbc_rows(
        _load_jsonl(paths["runtime_target_rows"]), profile
    )
    hts_rows = materialize_v7_0_hts_rows(
        _load_jsonl(paths["full_action_grid_rows"]), profile
    )
    result = fit_abstention_aware_v7_0(
        sbc_rows=sbc_rows,
        hts_rows=hts_rows,
        profile=profile,
        implementation_commit=config.implementation_commit,
        fit_created_ts=config.fit_created_ts,
    )
    model = result["model_artifact"]
    leakage_audit = _fit_leakage_audit(
        sbc_rows=sbc_rows,
        hts_rows=hts_rows,
        profile=profile,
        model=model,
    )
    report = _fit_report(
        sbc_rows=sbc_rows,
        hts_rows=hts_rows,
        model=model,
        leakage_audit=leakage_audit,
        implementation_commit=config.implementation_commit,
        fit_created_ts=config.fit_created_ts,
    )

    run_dir = Path(config.output_dir).resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    output_paths = {
        "model": run_dir / "v7_0_abstention_aware_expected_net_pnl_model.json",
        "report": run_dir / "v7_0_historical_fit_and_calibration_report.json",
        "report_markdown": run_dir / "v7_0_historical_fit_and_calibration_report.md",
        "leakage_audit": run_dir / "v7_0_fit_leakage_audit.json",
        "oof_rows": run_dir / "v7_0_forward_oof_prediction_rows.jsonl",
        "calibration_rows": run_dir / "v7_0_calibration_prediction_rows.jsonl",
    }
    _write_json(output_paths["model"], model)
    _write_json(output_paths["report"], report)
    _write_text(output_paths["report_markdown"], _fit_report_markdown(report))
    _write_json(output_paths["leakage_audit"], leakage_audit)
    _write_jsonl(output_paths["oof_rows"], result["oof_rows"])
    _write_jsonl(output_paths["calibration_rows"], result["calibration_rows"])
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        **{name: _descriptor(path) for name, path in paths.items()},
        **{name: _descriptor(path) for name, path in output_paths.items()},
        "historical_development_gate_passed": model[
            "historical_development_gate_passed"
        ],
        "historical_development_blocking_reason_codes": model[
            "historical_development_blocking_reason_codes"
        ],
        "fit_leakage_audit_passed": leakage_audit["fit_leakage_audit_passed"],
        "future_target_accessed": False,
        "future_confirmatory_started": False,
        **_v7_0_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v7_0_historical_fit_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "model": model,
        "report": report,
        "leakage_audit": leakage_audit,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "output_paths": output_paths,
    }


def _canonical_training_row(
    *,
    source: str,
    market_id: str,
    decision_ts: int,
    max_input_ts: int,
    role: str,
    family: str,
    action: str,
    side: str,
    values: dict[str, float],
    target: float,
) -> dict[str, Any]:
    if set(values) != set(FEATURE_NAMES):
        raise ValueError("#232 canonical feature set mismatch")
    if not market_id or not all(math.isfinite(value) for value in values.values()):
        raise ValueError("#232 canonical feature value invalid")
    if not math.isfinite(target):
        raise ValueError("#232 canonical target invalid")
    item = {
        "source": source,
        "market_id": market_id,
        "decision_group_id": f"{market_id}|{decision_ts}",
        "decision_ts": decision_ts,
        "max_input_ts": max_input_ts,
        "role": role,
        "action_family": family,
        "action": action,
        "side": side,
        "decision_time_features": values,
        "target_after_cost_net_pnl_per_contract": target,
        "target_used_as_decision_time_input": False,
        "target_available_only_post_exit_or_official_resolution": True,
    }
    item["canonical_training_row_id"] = canonical_json_sha256(item)
    return item


def _sbc_feature_values(row: dict[str, Any], profile: dict[str, Any]) -> dict[str, float]:
    features = dict(row.get("features") or {})
    score = _finite(features.get("canonical_v6_2_score"), "canonical_v6_2_score")
    margin = _finite(features.get("action_score_margin"), "action_score_margin")
    sentinel = float(profile["feature_contract"]["source_score_missing_sentinel_maximum"])
    available = score > sentinel
    if not available:
        score = float(profile["feature_contract"]["source_score_missing_replacement"])
        margin = float(
            profile["feature_contract"]["source_score_margin_missing_replacement"]
        )
    side = str(row["side"])
    anchor = _side_anchor(
        side,
        [
            features.get("btc_return_30s"),
            features.get("btc_return_1m"),
            features.get("reference_price_to_beat_distance_at_decision"),
        ],
    )
    return _common_feature_values(
        action_score_available=float(available),
        action_score=score,
        action_score_margin=margin,
        btc_anchor_direction=anchor,
        selected_side_probability=features.get("selected_side_probability"),
        execution_price=features.get("execution_price"),
        spread_bps=features.get("spread_bps"),
        queue_fill=features.get("queue_fill_probability_proxy"),
        book_staleness_ms=features.get("book_staleness_ms"),
        time_to_close_seconds=features.get("time_to_close_seconds"),
        pre_entry_market_exposure=features.get("pre_entry_market_exposure"),
        same_side_prior_entry=features.get("same_side_prior_entry"),
        side_flip_prior_entry=features.get("side_flip_prior_entry"),
        side=side,
        profile=profile,
    )


def _hts_feature_values(
    row: dict[str, Any],
    *,
    ranking: dict[str, dict[str, Any]],
    action: str,
    profile: dict[str, Any],
) -> dict[str, float]:
    rank_row = ranking[action]
    source_features = dict(row.get("decision_time_features") or {})
    context = dict(row.get("execution_handoff_context") or {})
    side = "UP" if "_UP_" in action else "DOWN"
    score = _finite(rank_row.get("corrected_model_score"), "corrected_model_score")
    other_scores = [
        _finite(value.get("corrected_model_score"), "corrected_model_score")
        for name, value in ranking.items()
        if name != action
    ]
    margin = score - max(other_scores)
    micro = dict(rank_row.get("microstructure_snapshot") or {})
    probability = context.get("p_up") if side == "UP" else context.get("p_down")
    anchor = _side_anchor(
        side,
        [
            source_features.get("chainlink_momentum_30s"),
            source_features.get("chainlink_momentum_60s"),
            source_features.get("chainlink_momentum_120s"),
            source_features.get("reference_price_to_beat_distance_at_decision"),
        ],
    )
    return _common_feature_values(
        action_score_available=1.0,
        action_score=score,
        action_score_margin=margin,
        btc_anchor_direction=anchor,
        selected_side_probability=probability,
        execution_price=micro.get("entry_ask"),
        spread_bps=micro.get("spread_bps"),
        queue_fill=micro.get("queue_fill_proxy"),
        book_staleness_ms=micro.get("book_staleness_ms"),
        time_to_close_seconds=micro.get("time_to_close_seconds"),
        pre_entry_market_exposure=source_features.get(
            "cumulative_market_exposure_before_entry"
        ),
        same_side_prior_entry=source_features.get("same_side_reentry"),
        side_flip_prior_entry=source_features.get("side_flip"),
        side=side,
        profile=profile,
    )


def _common_feature_values(
    *,
    action_score_available: Any,
    action_score: Any,
    action_score_margin: Any,
    btc_anchor_direction: Any,
    selected_side_probability: Any,
    execution_price: Any,
    spread_bps: Any,
    queue_fill: Any,
    book_staleness_ms: Any,
    time_to_close_seconds: Any,
    pre_entry_market_exposure: Any,
    same_side_prior_entry: Any,
    side_flip_prior_entry: Any,
    side: str,
    profile: dict[str, Any],
) -> dict[str, float]:
    probability = _bounded(selected_side_probability, "selected_side_probability")
    price = _bounded(execution_price, "execution_price")
    spread = _nonnegative(spread_bps, "spread_bps")
    queue = _bounded(queue_fill, "queue_fill_probability")
    staleness = _nonnegative(book_staleness_ms, "book_staleness_ms")
    time_to_close = _nonnegative(time_to_close_seconds, "time_to_close_seconds")
    late_window_seconds = float(profile["feature_contract"]["late_window_seconds"])
    values = {
        "action_score_available": _binary(
            action_score_available, "action_score_available"
        ),
        "action_score": _finite(action_score, "action_score"),
        "action_score_margin": _finite(action_score_margin, "action_score_margin"),
        "btc_anchor_direction": _finite(
            btc_anchor_direction, "btc_anchor_direction"
        ),
        "selected_side_probability": probability,
        "execution_price": price,
        "selected_side_probability_minus_execution_price": probability - price,
        "log1p_spread_bps": math.log1p(spread),
        "queue_fill_shortfall": 1.0 - queue,
        "log1p_book_staleness_ms": math.log1p(staleness),
        "late_window_pressure": max(0.0, 1.0 - time_to_close / late_window_seconds),
        "pre_entry_market_exposure": _nonnegative(
            pre_entry_market_exposure, "pre_entry_market_exposure"
        ),
        "same_side_prior_entry": _binary(
            same_side_prior_entry, "same_side_prior_entry"
        ),
        "side_flip_prior_entry": _binary(
            side_flip_prior_entry, "side_flip_prior_entry"
        ),
        "side_is_up": 1.0 if side == "UP" else 0.0,
    }
    return values


def _ranking_by_action(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    ranking = list(
        (row.get("execution_handoff_context") or {}).get("full_5_action_ranking")
        or []
    )
    result = {str(item.get("selected_action") or ""): dict(item) for item in ranking}
    if set(result) != FULL_ACTION_GRID:
        raise ValueError("#232 full five-action ranking incomplete")
    return result


def _chronological_market_order(rows: list[dict[str, Any]]) -> list[str]:
    minimum_ts: dict[str, int] = {}
    for row in rows:
        market_id = str(row.get("market_id") or "")
        if not market_id:
            raise ValueError("#232 historical full-grid market id missing")
        minimum_ts[market_id] = min(
            minimum_ts.get(market_id, int(row["decision_ts"])),
            int(row["decision_ts"]),
        )
    return [market_id for market_id, _ in sorted(minimum_ts.items(), key=lambda x: (x[1], x[0]))]


def _validate_materialized_family_grid(
    rows: list[dict[str, Any]], *, actions: set[str]
) -> None:
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        grouped[row["decision_group_id"]].add(row["action"])
        if FORBIDDEN_INFERENCE_FIELDS.intersection(row["decision_time_features"]):
            raise ValueError("#232 forbidden field entered canonical feature map")
    if any(group_actions != actions for group_actions in grouped.values()):
        raise ValueError("#232 materialized family action grid incomplete")


def _forward_oof(
    rows: list[dict[str, Any]], *, split: dict[str, Any], profile: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    market_order = _chronological_market_order(rows)
    initial = int(split["forward_oof_initial_fit_market_count"])
    width = int(split["forward_oof_validation_market_count_per_fold"])
    fold_count = int(split["forward_oof_fold_count"])
    predictions: list[dict[str, Any]] = []
    fold_summaries = []
    for fold in range(fold_count):
        train_ids = set(market_order[: initial + fold * width])
        validation_ids = set(market_order[initial + fold * width : initial + (fold + 1) * width])
        if not train_ids or len(validation_ids) != width:
            raise ValueError("#232 fixed forward OOF split support invalid")
        train = [row for row in rows if row["market_id"] in train_ids]
        validation = [row for row in rows if row["market_id"] in validation_ids]
        if max(row["decision_ts"] for row in train) >= min(
            row["decision_ts"] for row in validation
        ):
            raise ValueError("#232 forward OOF chronology invalid")
        model = _fit_family_model(train, profile=profile)
        train_mean = _market_weighted_mean(
            train, [row["target_after_cost_net_pnl_per_contract"] for row in train]
        )
        fold_rows = []
        for row in validation:
            prediction = _predict_model(row, model)
            item = _prediction_row(
                row,
                prediction=prediction,
                lower_bound=None,
                stage="forward_oof",
                fold_index=fold,
                baseline_prediction=train_mean,
            )
            fold_rows.append(item)
        predictions.extend(fold_rows)
        fold_summaries.append(
            {
                "fold_index": fold,
                "train_market_count": len(train_ids),
                "validation_market_count": len(validation_ids),
                "train_max_decision_ts": max(row["decision_ts"] for row in train),
                "validation_min_decision_ts": min(
                    row["decision_ts"] for row in validation
                ),
            }
        )
    metrics = _prediction_metrics(predictions)
    metrics["fold_summaries"] = fold_summaries
    return predictions, metrics


def _fit_family_model(
    rows: list[dict[str, Any]],
    *,
    profile: dict[str, Any],
    fixed_standardization: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("#232 family fit rows are empty")
    x = np.asarray(
        [[float(row["decision_time_features"][name]) for name in FEATURE_NAMES] for row in rows],
        dtype=float,
    )
    y = np.asarray(
        [float(row["target_after_cost_net_pnl_per_contract"]) for row in rows],
        dtype=float,
    )
    weights = np.asarray(_market_weights(rows), dtype=float)
    if fixed_standardization is None:
        mean = np.average(x, axis=0, weights=weights)
        variance = np.average((x - mean) ** 2, axis=0, weights=weights)
        scale = np.sqrt(variance)
        floor = float(profile["feature_contract"]["standard_deviation_floor"])
        scale = np.where(scale > floor, scale, 1.0)
    else:
        mean, scale = fixed_standardization
    standardized = (x - mean) / scale
    design = np.column_stack((np.ones(len(rows)), standardized))
    root_weights = np.sqrt(weights)
    weighted_design = design * root_weights[:, None]
    weighted_target = y * root_weights
    alpha = float(profile["model_contract"]["ridge_alpha"])
    penalty = np.diag([0.0, *([alpha] * len(FEATURE_NAMES))])
    coefficients = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_target,
    )
    bound = float(profile["model_contract"]["coefficient_absolute_bound"])
    finite_bounded = bool(
        np.all(np.isfinite(coefficients)) and np.max(np.abs(coefficients)) <= bound
    )
    return {
        "model_family": profile["model_contract"]["model_family"],
        "ridge_alpha": alpha,
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


def _predict_model(row: dict[str, Any], model: dict[str, Any]) -> float:
    x = np.asarray(
        [float(row["decision_time_features"][name]) for name in FEATURE_NAMES],
        dtype=float,
    )
    mean = np.asarray(model["feature_mean"], dtype=float)
    scale = np.asarray(model["feature_scale"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    return float(model["intercept"] + ((x - mean) / scale) @ coefficients)


def _coefficient_stability(
    rows: list[dict[str, Any]], *, final_model: dict[str, Any], profile: dict[str, Any]
) -> dict[str, Any]:
    markets = sorted({row["market_id"] for row in rows})
    reference = np.asarray(
        [final_model["intercept"], *final_model["coefficients"]], dtype=float
    )
    mean = np.asarray(final_model["feature_mean"], dtype=float)
    scale = np.asarray(final_model["feature_scale"], dtype=float)
    maximum_delta = 0.0
    per_market = []
    for market_id in markets:
        subset = [row for row in rows if row["market_id"] != market_id]
        model = _fit_family_model(
            subset,
            profile=profile,
            fixed_standardization=(mean, scale),
        )
        candidate = np.asarray([model["intercept"], *model["coefficients"]])
        delta = float(np.max(np.abs(candidate - reference)))
        maximum_delta = max(maximum_delta, delta)
        per_market.append({"excluded_market_id": market_id, "max_absolute_delta": delta})
    return {
        "method": "leave_one_market_out_with_final_fit_standardization",
        "market_count": len(markets),
        "maximum_coefficient_absolute_delta": maximum_delta,
        "median_market_max_absolute_delta": statistics.median(
            item["max_absolute_delta"] for item in per_market
        ),
        "worst_market_ids": [
            item["excluded_market_id"]
            for item in sorted(
                per_market,
                key=lambda item: (-item["max_absolute_delta"], item["excluded_market_id"]),
            )[:5]
        ],
    }


def _calibrate_family(
    rows: list[dict[str, Any]], *, model: dict[str, Any], profile: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predicted = [(row, _predict_model(row, model)) for row in rows]
    selected = []
    grouped: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, prediction in predicted:
        grouped[row["decision_group_id"]].append((row, prediction))
    for group in grouped.values():
        selected.append(
            sorted(group, key=lambda item: (-item[1], item[0]["action"]))[0]
        )
    residuals = [
        float(row["target_after_cost_net_pnl_per_contract"]) - prediction
        for row, prediction in selected
    ]
    weights = _market_weights([row for row, _ in selected])
    quantile = _weighted_quantile(
        residuals,
        weights,
        float(profile["calibration_contract"]["residual_quantile"]),
    )
    output = []
    for row, prediction in predicted:
        output.append(
            _prediction_row(
                row,
                prediction=prediction,
                lower_bound=prediction + quantile,
                stage="calibration",
                fold_index=None,
                baseline_prediction=None,
            )
        )
    coverage = _weighted_mean(
        [float(residual >= quantile) for residual in residuals], weights
    )
    selected_by_lcb = _select_rows_by_lower_bound(output)
    pnl_sum = sum(
        float(row["target_after_cost_net_pnl_per_contract"])
        for row in selected_by_lcb
    )
    return output, {
        "method": profile["calibration_contract"]["method"],
        "coverage_level": profile["calibration_contract"]["coverage_level"],
        "residual_quantile": quantile,
        "selected_action_residual_count": len(residuals),
        "unique_market_count": len({row["market_id"] for row in rows}),
        "empirical_selected_action_lower_bound_coverage": coverage,
        "lower_bound_positive_selected_row_count": len(selected_by_lcb),
        "lower_bound_positive_selected_unique_market_count": len(
            {row["market_id"] for row in selected_by_lcb}
        ),
        "lower_bound_positive_selected_target_pnl_sum_report_only": pnl_sum,
        "pnl_used_for_calibration_or_gate": False,
    }


def _prediction_row(
    row: dict[str, Any],
    *,
    prediction: float,
    lower_bound: float | None,
    stage: str,
    fold_index: int | None,
    baseline_prediction: float | None,
) -> dict[str, Any]:
    item = {
        "market_id": row["market_id"],
        "decision_group_id": row["decision_group_id"],
        "decision_ts": row["decision_ts"],
        "max_input_ts": row["max_input_ts"],
        "role": row["role"],
        "action_family": row["action_family"],
        "action": row["action"],
        "side": row["side"],
        "stage": stage,
        "fold_index": fold_index,
        "model_predicted_after_cost_net_pnl_per_contract": prediction,
        "calibrated_lower_bound_after_cost_net_pnl_per_contract": lower_bound,
        "fold_train_mean_baseline_prediction": baseline_prediction,
        "target_after_cost_net_pnl_per_contract": row[
            "target_after_cost_net_pnl_per_contract"
        ],
        "target_used_as_decision_time_input": False,
        "target_used_for_threshold_or_model_selection": False,
    }
    item["prediction_row_id"] = canonical_json_sha256(item)
    return item


def _prediction_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    targets = [float(row["target_after_cost_net_pnl_per_contract"]) for row in rows]
    predictions = [
        float(row["model_predicted_after_cost_net_pnl_per_contract"]) for row in rows
    ]
    baselines = [float(row["fold_train_mean_baseline_prediction"]) for row in rows]
    weights = _market_weights(rows)
    mae = _weighted_mean(
        [abs(target - prediction) for target, prediction in zip(targets, predictions, strict=True)],
        weights,
    )
    mse = _weighted_mean(
        [(target - prediction) ** 2 for target, prediction in zip(targets, predictions, strict=True)],
        weights,
    )
    baseline_mae = _weighted_mean(
        [abs(target - baseline) for target, baseline in zip(targets, baselines, strict=True)],
        weights,
    )
    baseline_mse = _weighted_mean(
        [(target - baseline) ** 2 for target, baseline in zip(targets, baselines, strict=True)],
        weights,
    )
    return {
        "row_count": len(rows),
        "unique_market_count": len({row["market_id"] for row in rows}),
        "market_weighted_mae": mae,
        "market_weighted_mse": mse,
        "fold_train_mean_baseline_market_weighted_mae": baseline_mae,
        "fold_train_mean_baseline_market_weighted_mse": baseline_mse,
        "relative_mae_improvement_over_fold_train_mean": _relative_improvement(
            baseline_mae, mae
        ),
        "relative_mse_improvement_over_fold_train_mean": _relative_improvement(
            baseline_mse, mse
        ),
    }


def _fit_leakage_audit(
    *,
    sbc_rows: list[dict[str, Any]],
    hts_rows: list[dict[str, Any]],
    profile: dict[str, Any],
    model: dict[str, Any],
) -> dict[str, Any]:
    rows = [*sbc_rows, *hts_rows]
    forbidden = set(profile["feature_contract"]["forbidden_inference_field_names"])
    violations = sum(
        bool(forbidden.intersection(row["decision_time_features"])) for row in rows
    )
    causality = sum(row["max_input_ts"] > row["decision_ts"] for row in rows)
    checks = {
        "decision_time_feature_set_exact": all(
            tuple(row["decision_time_features"]) == FEATURE_NAMES for row in rows
        ),
        "forbidden_inference_field_violation_count_zero": violations == 0,
        "feature_timestamp_causality_violation_count_zero": causality == 0,
        "targets_not_used_as_decision_inputs": all(
            row["target_used_as_decision_time_input"] is False for row in rows
        ),
        "validation_or_oof_pnl_not_used_for_tuning_or_gate": model[
            "validation_or_oof_pnl_used_for_tuning_or_gate"
        ]
        is False,
        "issue229_issue231_excluded": model[
            "issue229_or_issue231_rows_used_for_fit_or_tuning"
        ]
        is False,
        "side_rules_disabled": model["side_quota_applied"] is False
        and model["side_count_hard_gate_enabled"] is False
        and model["side_pnl_hard_gate_enabled"] is False,
    }
    reasons = [name for name, passed in checks.items() if not passed]
    audit = {
        "schema_version": LEAKAGE_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "canonical_row_count": len(rows),
        "canonical_unique_market_count": len({row["market_id"] for row in rows}),
        "forbidden_inference_field_violation_count": violations,
        "feature_timestamp_causality_violation_count": causality,
        "fit_leakage_checks": checks,
        "fit_leakage_audit_passed": not reasons,
        "fit_leakage_blocking_reason_codes": reasons,
        "future_target_accessed": False,
        **_v7_0_blocked_safety_fields(),
    }
    audit["leakage_audit_id"] = canonical_json_sha256(audit)
    return audit


def _fit_report(
    *,
    sbc_rows: list[dict[str, Any]],
    hts_rows: list[dict[str, Any]],
    model: dict[str, Any],
    leakage_audit: dict[str, Any],
    implementation_commit: str,
    fit_created_ts: int,
) -> dict[str, Any]:
    family_summary = {}
    for family in FAMILIES:
        artifact = model["family_models"][family]
        family_summary[family] = {
            "fit_market_count": artifact["fit_market_count"],
            "fit_row_count": artifact["fit_row_count"],
            "calibration_market_count": artifact["calibration_market_count"],
            "calibration_row_count": artifact["calibration_row_count"],
            "forward_oof_metrics": artifact["forward_oof_metrics"],
            "coefficient_stability": artifact["coefficient_stability"],
            "calibration": artifact["calibration"],
            "historical_fit_gate_passed": artifact["historical_fit_gate_passed"],
            "historical_fit_gate_blocking_reason_codes": artifact[
                "historical_fit_gate_blocking_reason_codes"
            ],
        }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": implementation_commit,
        "fit_created_ts": fit_created_ts,
        "historical_unique_market_count": len(
            {row["market_id"] for row in [*sbc_rows, *hts_rows]}
        ),
        "runtime_aligned_sbc_market_count": len(
            {row["market_id"] for row in sbc_rows}
        ),
        "full_action_grid_market_count": len(
            {row["market_id"] for row in hts_rows}
        ),
        "family_summary": family_summary,
        "historical_development_gate_passed": model[
            "historical_development_gate_passed"
        ],
        "historical_development_blocking_reason_codes": model[
            "historical_development_blocking_reason_codes"
        ],
        "fit_leakage_audit_passed": leakage_audit["fit_leakage_audit_passed"],
        "historical_selected_pnl_report_only": True,
        "validation_or_oof_pnl_used_for_tuning_or_gate": False,
        "issue229_or_issue231_outcomes_opened": False,
        "future_confirmatory_started": False,
        "new_strictly_later_future_confirmatory_required": True,
        **_v7_0_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def _fit_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# v7.0 Historical Fit and Calibration Report",
        "",
        f"- historical markets: `{report['historical_unique_market_count']}`",
        f"- development gate passed: `{str(report['historical_development_gate_passed']).lower()}`",
        f"- blockers: `{report['historical_development_blocking_reason_codes']}`",
        f"- leakage audit passed: `{str(report['fit_leakage_audit_passed']).lower()}`",
        "- validation/OOF PnL used for tuning or gate: `false`",
        "- #229/#231 outcomes opened: `false`",
        "- future confirmatory started: `false`",
        "",
        "## Family Diagnostics",
        "",
    ]
    for family, summary in report["family_summary"].items():
        lines.extend(
            [
                f"### {family}",
                "",
                f"- fit/calibration markets: `{summary['fit_market_count']} / {summary['calibration_market_count']}`",
                f"- OOF relative MAE improvement: `{summary['forward_oof_metrics']['relative_mae_improvement_over_fold_train_mean']}`",
                f"- OOF relative MSE improvement: `{summary['forward_oof_metrics']['relative_mse_improvement_over_fold_train_mean']}`",
                f"- calibration coverage: `{summary['calibration']['empirical_selected_action_lower_bound_coverage']}`",
                f"- report-only selected PnL: `{summary['calibration']['lower_bound_positive_selected_target_pnl_sum_report_only']}`",
                f"- gate passed: `{str(summary['historical_fit_gate_passed']).lower()}`",
                f"- blockers: `{summary['historical_fit_gate_blocking_reason_codes']}`",
                "",
            ]
        )
    lines.extend(
        [
            "No paper/live/write/wallet/capital/handoff/source/freeze/promotion unlock.",
            "",
        ]
    )
    return "\n".join(lines)


def _select_rows_by_lower_bound(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["decision_group_id"]].append(row)
    selected = []
    for group in grouped.values():
        best = sorted(
            group,
            key=lambda row: (
                -float(row["calibrated_lower_bound_after_cost_net_pnl_per_contract"]),
                row["action"],
            ),
        )[0]
        if float(best["calibrated_lower_bound_after_cost_net_pnl_per_contract"]) > 0.0:
            selected.append(best)
    return selected


def _market_weights(rows: list[dict[str, Any]]) -> list[float]:
    counts = Counter(str(row["market_id"]) for row in rows)
    return [1.0 / counts[str(row["market_id"])] for row in rows]


def _market_weighted_mean(rows: list[dict[str, Any]], values: list[float]) -> float:
    return _weighted_mean(values, _market_weights(rows))


def _weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    value_list = list(values)
    weight_list = list(weights)
    total = sum(weight_list)
    if not value_list or len(value_list) != len(weight_list) or total <= 0.0:
        raise ValueError("#232 weighted mean inputs invalid")
    return sum(value * weight for value, weight in zip(value_list, weight_list, strict=True)) / total


def _weighted_quantile(
    values: list[float], weights: list[float], quantile: float
) -> float:
    if not values or len(values) != len(weights) or not 0.0 <= quantile <= 1.0:
        raise ValueError("#232 weighted quantile inputs invalid")
    ordered = sorted(zip(values, weights, strict=True), key=lambda item: item[0])
    target = quantile * sum(weights)
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= target:
            return float(value)
    return float(ordered[-1][0])


def _relative_improvement(baseline: float, candidate: float) -> float:
    if baseline <= 0.0:
        return 0.0
    return (baseline - candidate) / baseline


def _side_anchor(side: str, values: list[Any]) -> float:
    available = [
        _finite(value, "btc_anchor_source") for value in values if value is not None
    ]
    if not available:
        raise ValueError("#232 BTC anchor source unavailable")
    sign = 1.0 if side == "UP" else -1.0
    return sign * statistics.median(available)


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"#232 {name} is not numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"#232 {name} is not finite")
    return result


def _bounded(value: Any, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"#232 {name} outside [0,1]")
    return result


def _nonnegative(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"#232 {name} is negative")
    return result


def _binary(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result not in {0.0, 1.0}:
        raise ValueError(f"#232 {name} is not binary")
    return result


def _row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["decision_ts"]),
        str(row["market_id"]),
        str(row.get("action_family") or ""),
        str(row.get("action") or ""),
    )
