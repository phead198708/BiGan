"""Calibration-scale alignment and target-free liveness for issue #231."""

from __future__ import annotations

import math
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_regime_emergent_pnl_v6_8 import (
    CALIBRATION_ARTIFACT_SCHEMA_VERSION as V6_8_CALIBRATION_SCHEMA_VERSION,
)
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
    _blocked_safety_fields,
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

CANDIDATE_NAME = "calibration_scale_aligned_runtime_pnl_v6_9"
PROFILE_SCHEMA_VERSION = "bigan-v8-calibration-scale-aligned-runtime-pnl-v6-9-profile-v1"
SCALE_AUDIT_SCHEMA_VERSION = "bigan-v8-v6-9-calibration-scale-contract-audit-v1"
MAPPING_SCHEMA_VERSION = "bigan-v8-v6-9-score-runtime-pnl-mapping-v1"
LIVENESS_REPORT_SCHEMA_VERSION = "bigan-v8-v6-9-target-free-liveness-report-v1"
MANIFEST_SCHEMA_VERSION = "bigan-v8-v6-9-candidate-freeze-manifest-v1"
SIDES = ("UP", "DOWN")
SBC_ACTIONS = {
    "BUY_UP_SELL_BEFORE_CLOSE",
    "BUY_DOWN_SELL_BEFORE_CLOSE",
}
FORBIDDEN_TARGET_FIELDS = {
    "resolved_outcome",
    "winning_outcome",
    "settlement_pnl",
    "realized_trade_pnl",
    "total_polymarket_pnl",
    "runtime_policy_after_cost_net_pnl_per_contract",
    "runtime_policy_after_cost_net_pnl_at_frozen_size",
    "future_return",
    "label",
    "oracle_action",
}
FROZEN_LINEAGE = {
    "v6_4_runtime_target_manifest_sha256": (
        "e08872532724ace0f84829342092f7cea721e4b6ae91e2b4d8e2974f55fdaab9"
    ),
    "v6_4_runtime_target_rows_sha256": (
        "1565116daeb2f5d4d8c33fefa507276f59251edd5ffb5f4f313041bcf9dbb0ec"
    ),
    "v6_2_mean_risk_calibration_sha256": (
        "dc82ddebc51e95e46477894f2a0ba7bd8fa2f6845b22ced43402822b66b68e43"
    ),
    "v6_8_failed_calibration_artifact_sha256": (
        "15acb580de9c1e30193211d4117093929e699a8631afa89d7744cb09162af8fd"
    ),
    "v6_8_evaluation_profile_sha256": (
        "d885b5a81fc217175eefac8a27c53eadd8044fd7731148624396709db5167dfe"
    ),
    "v6_7_evaluation_profile_sha256": (
        "900dba0b3d1e280271ff2489e0d0320f1eca150787bf2be30b8b751a3a993c3e"
    ),
    "issue229_target_free_freeze_manifest_sha256": (
        "186f82099ac2075e9c7f34411c68ef1655bddc7510e92426fb6b2ff82214dd80"
    ),
    "issue229_v6_7_base_selected_rows_sha256": (
        "ef2e66a0e7577e1230f7871d7a713dbdbff1c315dec29536409d93d583d00cb1"
    ),
}


@dataclass(frozen=True, slots=True)
class CalibrationScaleAlignedV69Config:
    """Pinned inputs for the one offline v6.9 fit and liveness freeze."""

    run_id: str
    output_dir: Path | str
    profile_path: Path | str
    expected_profile_sha256: str
    runtime_target_manifest_path: Path | str
    runtime_target_rows_path: Path | str
    failed_v6_8_calibration_artifact_path: Path | str
    issue229_target_free_freeze_manifest_path: Path | str
    issue229_v6_7_base_selected_rows_path: Path | str
    implementation_commit: str
    candidate_freeze_created_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(self.expected_profile_sha256, "expected_profile_sha256")
        _require_git_sha(self.implementation_commit)
        if self.candidate_freeze_created_ts <= 0:
            raise ValueError("candidate_freeze_created_ts must be positive")
        for name in (
            "output_dir",
            "profile_path",
            "runtime_target_manifest_path",
            "runtime_target_rows_path",
            "failed_v6_8_calibration_artifact_path",
            "issue229_target_free_freeze_manifest_path",
            "issue229_v6_7_base_selected_rows_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


def validate_calibration_scale_aligned_v6_9_profile(profile: dict[str, Any]) -> None:
    """Reject lineage, scale, mapping, liveness, or safety drift."""

    scale = dict(profile.get("calibration_scale_contract") or {})
    mapping = dict(profile.get("historical_mapping") or {})
    liveness = dict(profile.get("target_free_liveness") or {})
    confirmatory = dict(profile.get("future_confirmatory") or {})
    checks = {
        "schema": profile.get("schema_version") == PROFILE_SCHEMA_VERSION,
        "issue": profile.get("issue_number") == 231,
        "candidate": profile.get("candidate_name") == CANDIDATE_NAME,
        "frozen": profile.get("frozen") is True,
        "lineage": profile.get("lineage") == FROZEN_LINEAGE,
        "scale": scale
        == {
            "source_score_field": "v6_7_base_score",
            "source_score_semantic": "frozen_v6_2_market_clustered_mean_ev_lcb",
            "source_score_nominal_unit": "after_cost_net_pnl_per_contract",
            "target_field": "runtime_policy_after_cost_net_pnl_per_contract",
            "target_semantic": "frozen_runtime_exit_policy_after_cost_net_pnl",
            "target_nominal_unit": "after_cost_net_pnl_per_contract",
            "nominal_unit_equality_is_sufficient_for_additive_correction": False,
            "estimand_semantics_must_match_for_additive_correction": True,
            "unconditional_additive_ucb_correction_allowed": False,
            "replacement": "bounded_regularized_score_to_runtime_pnl_mapping",
        },
        "mapping": mapping
        == {
            "fit_role": "development_train",
            "fit_market_count": 89,
            "validation_role": "development_calibration",
            "validation_market_count": 45,
            "source_row_selection": (
                "highest_positive_canonical_v6_2_score_per_market_then_earliest_"
                "decision_ts_then_lexicographic_action"
            ),
            "mapping_family": "univariate_l2_ridge_with_unpenalized_intercept",
            "mapping_input": "canonical_v6_2_score",
            "mapping_output": "runtime_policy_after_cost_net_pnl_per_contract",
            "ridge_alpha": 100.0,
            "coefficient_absolute_bound": 8.0,
            "minimum_selected_fit_market_count": 40,
            "minimum_selected_validation_market_count": 20,
            "minimum_relative_validation_mae_improvement_over_train_mean_constant_exclusive": 0.0,
            "minimum_relative_validation_mse_improvement_over_train_mean_constant_exclusive": 0.0,
            "entry_threshold": 0.0,
            "threshold_operator": "strictly_greater_than",
            "validation_labels_used_for_model_fit": False,
            "validation_labels_used_for_threshold_selection": False,
            "validation_labels_used_for_fixed_mapping_validation": True,
            "hyperparameter_search_enabled": False,
            "result_selected_rerun_allowed": False,
        },
        "liveness": liveness
        == {
            "source": "issue229_target_free_features_only_outcomes_permanently_sealed",
            "exact_market_count": 120,
            "minimum_positive_mapped_score_unique_market_count_total": 40,
            "minimum_guard_accepted_unique_market_count_total": 40,
            "minimum_unique_market_count_per_side": None,
            "required_sides": [],
            "side_count_hard_gate_enabled": False,
            "side_composition_is_regime_emergent": True,
            "labels_outcomes_resolution_or_pnl_access_allowed": False,
        },
        "confirmatory": confirmatory
        == {
            "new_strictly_later_disjoint_outcome_blind_window_required": True,
            "issue229_market_ids_excluded": True,
            "issue229_outcomes_must_remain_sealed": True,
            "side_only_after_cost_pnl_hard_gate": True,
            "action_and_family_pnl_diagnostic_only": True,
            "side_quota_allowed": False,
            "result_selected_extension_or_rerun_allowed": False,
        },
        "safety": profile.get("safety") == _blocked_safety_fields(),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("#231 v6.9 profile invalid: " + ", ".join(blockers))


def build_v6_9_scale_contract_audit(
    source_rows: list[dict[str, Any]],
    *,
    failed_v6_8_artifact: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Prove why the old unconditional additive correction must stay disabled."""

    validate_calibration_scale_aligned_v6_9_profile(profile)
    _validate_target_free_rows(source_rows, exact_market_count=120)
    if failed_v6_8_artifact.get("schema_version") != V6_8_CALIBRATION_SCHEMA_VERSION:
        raise ValueError("#231 source v6.8 calibration artifact schema mismatch")
    pooled = dict(failed_v6_8_artifact.get("pooled_residual_calibration") or {})
    correction = _finite_float(pooled.get("upper_confidence_bound"), "v6.8 correction")
    scores = np.asarray([float(row["v6_7_base_score"]) for row in source_rows])
    score_summary = _distribution(scores)
    semantic_match = False
    checks = {
        "nominal_units_match": profile["calibration_scale_contract"][
            "source_score_nominal_unit"
        ]
        == profile["calibration_scale_contract"]["target_nominal_unit"],
        "estimand_semantics_match": semantic_match,
        "correction_inside_observed_source_score_support": float(scores.min())
        <= correction
        <= float(scores.max()),
        "additive_contract_complete": False,
    }
    audit = {
        "schema_version": SCALE_AUDIT_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "source_score_field": "v6_7_base_score",
        "source_score_semantic": profile["calibration_scale_contract"][
            "source_score_semantic"
        ],
        "runtime_target_field": profile["calibration_scale_contract"]["target_field"],
        "runtime_target_semantic": profile["calibration_scale_contract"][
            "target_semantic"
        ],
        "source_score_distribution": score_summary,
        "failed_v6_8_pooled_residual_upper_confidence_bound": correction,
        "correction_to_source_score_max_ratio": correction / float(scores.max()),
        "positive_source_score_count_before_correction": int(np.sum(scores > 0.0)),
        "positive_source_score_count_after_failed_correction": int(
            np.sum(scores - correction > 0.0)
        ),
        "scale_contract_checks": checks,
        "direct_additive_scale_contract_passed": False,
        "unconditional_additive_ucb_correction_allowed": False,
        "scale_contract_blocking_reason_codes": [
            "source_score_and_runtime_target_estimand_semantics_not_proven_equivalent",
            "pooled_correction_outside_observed_target_free_source_score_support",
            "unconditional_additive_correction_would_zero_target_free_actions",
        ],
        "required_replacement": profile["calibration_scale_contract"]["replacement"],
        "issue229_outcomes_opened": False,
        **_blocked_safety_fields(),
    }
    audit["audit_id"] = canonical_json_sha256(audit)
    return audit


def fit_v6_9_score_to_runtime_pnl_mapping(
    runtime_rows: list[dict[str, Any]],
    *,
    issue229_market_ids: set[str],
    profile: dict[str, Any],
    runtime_target_rows_descriptor: dict[str, Any],
    runtime_target_manifest_descriptor: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit one fixed ridge mapping on historical fit targets only."""

    validate_calibration_scale_aligned_v6_9_profile(profile)
    mapping = dict(profile["historical_mapping"])
    split = _validate_runtime_rows(runtime_rows, profile=profile)
    fit_rows = _select_mapping_rows(runtime_rows, role=mapping["fit_role"])
    validation_rows = _select_mapping_rows(
        runtime_rows, role=mapping["validation_role"]
    )
    historical_ids = {str(row["market_id"]) for row in runtime_rows}
    overlap = historical_ids.intersection(issue229_market_ids)

    x = np.asarray([float(row["features"]["canonical_v6_2_score"]) for row in fit_rows])
    y = np.asarray(
        [float(row["runtime_policy_after_cost_net_pnl_per_contract"]) for row in fit_rows]
    )
    mean = float(x.mean())
    scale = float(x.std())
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("#231 mapping source-score scale is invalid")
    z = (x - mean) / scale
    design = np.column_stack((np.ones(len(z)), z))
    penalty = np.diag((0.0, float(mapping["ridge_alpha"])))
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ y,
    )
    intercept = float(coefficients[0])
    standardized_slope = float(coefficients[1])
    raw_slope = standardized_slope / scale
    raw_intercept = intercept - standardized_slope * mean / scale
    fit_predictions = _mapping_predictions(
        fit_rows,
        intercept=intercept,
        standardized_slope=standardized_slope,
        source_mean=mean,
        source_scale=scale,
    )
    validation_predictions = _mapping_predictions(
        validation_rows,
        intercept=intercept,
        standardized_slope=standardized_slope,
        source_mean=mean,
        source_scale=scale,
    )
    train_mean = float(y.mean())
    fit_metrics = _mapping_metrics(fit_rows, fit_predictions, baseline=train_mean)
    validation_metrics = _mapping_metrics(
        validation_rows, validation_predictions, baseline=train_mean
    )
    bound = float(mapping["coefficient_absolute_bound"])
    checks = {
        "historical_roles_complete": split["roles_complete"],
        "chronological_market_disjoint_split": split["chronological"]
        and split["market_disjoint"],
        "fit_selected_market_support": len(fit_rows)
        >= int(mapping["minimum_selected_fit_market_count"]),
        "validation_selected_market_support": len(validation_rows)
        >= int(mapping["minimum_selected_validation_market_count"]),
        "issue229_market_disjoint": not overlap,
        "coefficients_finite_and_bounded": all(
            math.isfinite(value) and abs(value) <= bound
            for value in (intercept, standardized_slope, raw_slope, raw_intercept)
        ),
        "positive_monotone_source_score_slope": raw_slope > 0.0,
        "validation_relative_mae_improved": validation_metrics[
            "relative_mae_improvement_over_train_mean_constant"
        ]
        > float(
            mapping[
                "minimum_relative_validation_mae_improvement_over_train_mean_constant_exclusive"
            ]
        ),
        "validation_relative_mse_improved": validation_metrics[
            "relative_mse_improvement_over_train_mean_constant"
        ]
        > float(
            mapping[
                "minimum_relative_validation_mse_improvement_over_train_mean_constant_exclusive"
            ]
        ),
        "validation_labels_not_used_for_fit_or_threshold": mapping[
            "validation_labels_used_for_model_fit"
        ]
        is False
        and mapping["validation_labels_used_for_threshold_selection"] is False,
        "no_search_or_result_selected_rerun": mapping[
            "hyperparameter_search_enabled"
        ]
        is False
        and mapping["result_selected_rerun_allowed"] is False,
    }
    reason_map = {
        "historical_roles_complete": "historical_runtime_role_coverage_failed",
        "chronological_market_disjoint_split": "historical_runtime_split_invalid",
        "fit_selected_market_support": "mapping_fit_support_failed",
        "validation_selected_market_support": "mapping_validation_support_failed",
        "issue229_market_disjoint": "issue229_market_overlap_with_mapping_lineage",
        "coefficients_finite_and_bounded": "mapping_coefficients_invalid",
        "positive_monotone_source_score_slope": "mapping_source_score_slope_not_positive",
        "validation_relative_mae_improved": "mapping_validation_mae_gate_failed",
        "validation_relative_mse_improved": "mapping_validation_mse_gate_failed",
        "validation_labels_not_used_for_fit_or_threshold": (
            "mapping_validation_label_usage_contract_failed"
        ),
        "no_search_or_result_selected_rerun": "mapping_search_or_rerun_enabled",
    }
    reasons = [reason_map[name] for name, passed in checks.items() if not passed]
    artifact = {
        "schema_version": MAPPING_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "frozen": not reasons,
        "decision_time_safe_at_inference": True,
        "mapping_family": mapping["mapping_family"],
        "mapping_input": mapping["mapping_input"],
        "mapping_output": mapping["mapping_output"],
        "ridge_alpha": float(mapping["ridge_alpha"]),
        "source_score_mean": mean,
        "source_score_scale": scale,
        "intercept": intercept,
        "standardized_source_score_coefficient": standardized_slope,
        "raw_source_score_slope": raw_slope,
        "raw_intercept": raw_intercept,
        "entry_threshold": float(mapping["entry_threshold"]),
        "threshold_operator": mapping["threshold_operator"],
        "fit_market_count": len(fit_rows),
        "validation_market_count": len(validation_rows),
        "fit_side_distribution_diagnostic": _side_distribution(fit_rows),
        "validation_side_distribution_diagnostic": _side_distribution(
            validation_rows
        ),
        "fit_metrics": fit_metrics,
        "validation_metrics": validation_metrics,
        "mapping_gate_checks": checks,
        "mapping_gate_passed": not reasons,
        "mapping_gate_blocking_reason_codes": reasons,
        "runtime_target_rows": runtime_target_rows_descriptor,
        "runtime_target_manifest": runtime_target_manifest_descriptor,
        "historical_split_hash": split["split_hash"],
        "issue229_excluded_market_count": len(issue229_market_ids),
        "issue229_market_overlap_count": len(overlap),
        "issue229_outcomes_opened": False,
        "validation_labels_used_for_model_fit": False,
        "validation_labels_used_for_threshold_selection": False,
        "validation_labels_used_for_fixed_mapping_validation": True,
        "target_used_as_decision_time_input": False,
        "side_count_hard_gate_enabled": False,
        "side_composition_is_regime_emergent": True,
        "strictly_later_new_confirmatory_required": True,
        **_blocked_safety_fields(),
    }
    artifact["mapping_artifact_id"] = canonical_json_sha256(artifact)
    return artifact, fit_predictions, validation_predictions


def apply_v6_9_score_to_runtime_pnl_mapping(
    target_free_rows: list[dict[str, Any]],
    *,
    mapping_artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply the frozen mapping without reading any target or outcome field."""

    _validate_mapping_artifact(mapping_artifact)
    _validate_target_free_rows(target_free_rows, exact_market_count=len(target_free_rows))
    output = []
    for row in target_free_rows:
        score = float(row["v6_7_base_score"])
        mapped = _apply_mapping_value(score, mapping_artifact)
        if mapped <= float(mapping_artifact["entry_threshold"]):
            continue
        updated = {
            **row,
            "v6_9_calibrated_runtime_expected_pnl_per_contract": mapped,
            "v6_9_mapping_artifact_id": mapping_artifact["mapping_artifact_id"],
            "v6_9_mapping_applied_without_target_access": True,
            "source_score_mutated": False,
            "side_quota_applied": False,
            "labels_outcomes_resolution_or_pnl_opened": False,
        }
        updated["v6_9_selected_row_id"] = canonical_json_sha256(updated)
        output.append(updated)
    return sorted(output, key=lambda row: (int(row["decision_ts"]), row["market_id"]))


def build_v6_9_target_free_liveness_report(
    source_rows: list[dict[str, Any]],
    mapped_rows: list[dict[str, Any]],
    *,
    mapping_artifact: dict[str, Any],
    scale_audit: dict[str, Any],
    profile: dict[str, Any],
    implementation_commit: str,
    candidate_freeze_created_ts: int,
) -> dict[str, Any]:
    """Require total target-free action support without directional quotas."""

    validate_calibration_scale_aligned_v6_9_profile(profile)
    _validate_mapping_artifact(mapping_artifact)
    liveness = dict(profile["target_free_liveness"])
    exact = int(liveness["exact_market_count"])
    _validate_target_free_rows(source_rows, exact_market_count=exact)
    source_ids = {str(row["market_id"]) for row in source_rows}
    mapped_ids = {str(row["market_id"]) for row in mapped_rows}
    guard_accepted = [
        row
        for row in mapped_rows
        if row.get("microstructure_safety_passed") is True
        and row.get("hard_execution_safety_thresholds_unchanged") is True
        and row.get("exposure_duplicate_position_and_sizing_guards_unchanged") is True
    ]
    guard_ids = {str(row["market_id"]) for row in guard_accepted}
    checks = {
        "scale_mapping_required": scale_audit[
            "unconditional_additive_ucb_correction_allowed"
        ]
        is False,
        "mapping_gate": mapping_artifact["mapping_gate_passed"] is True,
        "exact_target_free_market_count": len(source_ids) == exact,
        "positive_mapped_score_total_support": len(mapped_ids)
        >= int(liveness["minimum_positive_mapped_score_unique_market_count_total"]),
        "guard_accepted_total_support": len(guard_ids)
        >= int(liveness["minimum_guard_accepted_unique_market_count_total"]),
        "mapped_rows_subset_of_source": mapped_ids.issubset(source_ids),
        "side_quota_disabled": liveness["side_count_hard_gate_enabled"] is False
        and liveness["minimum_unique_market_count_per_side"] is None
        and liveness["required_sides"] == [],
        "feature_causality": all(
            int(row["max_input_ts"]) <= int(row["decision_ts"])
            for row in source_rows
        ),
        "source_scores_unchanged": all(
            row.get("source_score_mutated") is False for row in mapped_rows
        ),
        "targets_remained_sealed": all(
            not FORBIDDEN_TARGET_FIELDS.intersection(row)
            and row.get("labels_outcomes_resolution_or_pnl_opened") is False
            for row in source_rows
        ),
        "freeze_after_decisions": candidate_freeze_created_ts
        > max(int(row["decision_ts"]) for row in source_rows),
    }
    blockers = [f"{name}_gate_failed" for name, passed in checks.items() if not passed]
    report = {
        "schema_version": LIVENESS_REPORT_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": implementation_commit,
        "candidate_freeze_created_ts": candidate_freeze_created_ts,
        "source_target_free_market_count": len(source_ids),
        "source_target_free_side_distribution_diagnostic": _side_distribution(
            source_rows
        ),
        "positive_mapped_score_unique_market_count": len(mapped_ids),
        "positive_mapped_score_side_distribution_diagnostic": _side_distribution(
            mapped_rows
        ),
        "guard_accepted_unique_market_count": len(guard_ids),
        "guard_accepted_side_distribution_diagnostic": _side_distribution(
            guard_accepted
        ),
        "minimum_total_support_required": int(
            liveness["minimum_guard_accepted_unique_market_count_total"]
        ),
        "minimum_per_side_support_required": None,
        "side_count_hard_gate_enabled": False,
        "side_composition_is_regime_emergent": True,
        "target_free_liveness_gate_checks": checks,
        "target_free_liveness_gate_passed": not blockers,
        "target_free_liveness_blocking_reason_codes": blockers,
        "candidate_scoring_frozen": not blockers,
        "strictly_later_outcome_blind_collection_allowed": not blockers,
        "current_issue229_window_eligible_for_confirmatory": False,
        "current_issue229_outcomes_opened": False,
        "new_strictly_later_disjoint_confirmatory_required": True,
        "future_target_access_allowed": False,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def run_calibration_scale_aligned_v6_9(
    config: CalibrationScaleAlignedV69Config,
) -> dict[str, Any]:
    """Run and freeze the one issue #231 offline mapping/liveness candidate."""

    paths = {
        "profile": Path(config.profile_path).resolve(),
        "runtime_target_manifest": Path(config.runtime_target_manifest_path).resolve(),
        "runtime_target_rows": Path(config.runtime_target_rows_path).resolve(),
        "failed_v6_8_calibration_artifact": Path(
            config.failed_v6_8_calibration_artifact_path
        ).resolve(),
        "issue229_target_free_freeze_manifest": Path(
            config.issue229_target_free_freeze_manifest_path
        ).resolve(),
        "issue229_v6_7_base_selected_rows": Path(
            config.issue229_v6_7_base_selected_rows_path
        ).resolve(),
    }
    _verify_pin(paths["profile"], config.expected_profile_sha256, "#231 profile")
    profile = _load_json(paths["profile"])
    validate_calibration_scale_aligned_v6_9_profile(profile)
    lineage = dict(profile["lineage"])
    pin_names = {
        "runtime_target_manifest": "v6_4_runtime_target_manifest_sha256",
        "runtime_target_rows": "v6_4_runtime_target_rows_sha256",
        "failed_v6_8_calibration_artifact": (
            "v6_8_failed_calibration_artifact_sha256"
        ),
        "issue229_target_free_freeze_manifest": (
            "issue229_target_free_freeze_manifest_sha256"
        ),
        "issue229_v6_7_base_selected_rows": (
            "issue229_v6_7_base_selected_rows_sha256"
        ),
    }
    for name, lineage_name in pin_names.items():
        _verify_pin(paths[name], lineage[lineage_name], f"#231 {name}")

    target_manifest = _load_json(paths["runtime_target_manifest"])
    manifest_rows = dict(target_manifest.get("runtime_aligned_rows") or {})
    if manifest_rows != _descriptor(paths["runtime_target_rows"]):
        raise ValueError("#231 runtime target manifest row descriptor mismatch")
    issue229_manifest = _load_json(paths["issue229_target_free_freeze_manifest"])
    base_descriptor = dict(issue229_manifest.get("v6_7_base_selected_rows") or {})
    if base_descriptor != _descriptor(paths["issue229_v6_7_base_selected_rows"]):
        raise ValueError("#231 issue229 base-row descriptor mismatch")
    if issue229_manifest.get("labels_outcomes_resolution_or_pnl_opened") is not False:
        raise ValueError("#231 issue229 target sealing is invalid")

    runtime_rows = _load_jsonl(paths["runtime_target_rows"])
    source_rows = _load_jsonl(paths["issue229_v6_7_base_selected_rows"])
    failed_artifact = _load_json(paths["failed_v6_8_calibration_artifact"])
    issue229_market_ids = {str(row["market_id"]) for row in source_rows}
    scale_audit = build_v6_9_scale_contract_audit(
        source_rows,
        failed_v6_8_artifact=failed_artifact,
        profile=profile,
    )
    mapping_artifact, fit_predictions, validation_predictions = (
        fit_v6_9_score_to_runtime_pnl_mapping(
            runtime_rows,
            issue229_market_ids=issue229_market_ids,
            profile=profile,
            runtime_target_rows_descriptor=_descriptor(paths["runtime_target_rows"]),
            runtime_target_manifest_descriptor=_descriptor(
                paths["runtime_target_manifest"]
            ),
        )
    )
    if mapping_artifact["mapping_gate_passed"] is not True:
        mapped_rows: list[dict[str, Any]] = []
    else:
        mapped_rows = apply_v6_9_score_to_runtime_pnl_mapping(
            source_rows,
            mapping_artifact=mapping_artifact,
        )
    report = build_v6_9_target_free_liveness_report(
        source_rows,
        mapped_rows,
        mapping_artifact=mapping_artifact,
        scale_audit=scale_audit,
        profile=profile,
        implementation_commit=config.implementation_commit,
        candidate_freeze_created_ts=config.candidate_freeze_created_ts,
    )

    run_dir = Path(config.output_dir).resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    artifact_paths = {
        "scale_contract_audit": run_dir / "v6_9_calibration_scale_contract_audit.json",
        "mapping_artifact": run_dir / "v6_9_score_to_runtime_pnl_mapping_artifact.json",
        "fit_predictions": run_dir / "v6_9_historical_fit_mapping_predictions.jsonl",
        "validation_predictions": (
            run_dir / "v6_9_historical_validation_mapping_predictions.jsonl"
        ),
        "mapped_liveness_rows": run_dir / "v6_9_target_free_mapped_rows.jsonl",
        "liveness_report": run_dir / "v6_9_target_free_liveness_report.json",
    }
    _write_json(artifact_paths["scale_contract_audit"], scale_audit)
    _write_text(
        artifact_paths["scale_contract_audit"].with_suffix(".md"),
        _scale_audit_markdown(scale_audit),
    )
    _write_json(artifact_paths["mapping_artifact"], mapping_artifact)
    _write_jsonl(artifact_paths["fit_predictions"], fit_predictions)
    _write_jsonl(artifact_paths["validation_predictions"], validation_predictions)
    _write_jsonl(artifact_paths["mapped_liveness_rows"], mapped_rows)
    _write_json(artifact_paths["liveness_report"], report)
    _write_text(
        artifact_paths["liveness_report"].with_suffix(".md"),
        _liveness_markdown(report, mapping_artifact),
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "issue_number": 231,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        "candidate_freeze_created_ts": config.candidate_freeze_created_ts,
        "profile": _descriptor(paths["profile"]),
        "runtime_target_manifest": _descriptor(paths["runtime_target_manifest"]),
        "runtime_target_rows": _descriptor(paths["runtime_target_rows"]),
        "failed_v6_8_calibration_artifact": _descriptor(
            paths["failed_v6_8_calibration_artifact"]
        ),
        "issue229_target_free_freeze_manifest": _descriptor(
            paths["issue229_target_free_freeze_manifest"]
        ),
        "issue229_v6_7_base_selected_rows": _descriptor(
            paths["issue229_v6_7_base_selected_rows"]
        ),
        **{name: _descriptor(path) for name, path in artifact_paths.items()},
        "direct_additive_scale_contract_passed": False,
        "mapping_gate_passed": mapping_artifact["mapping_gate_passed"],
        "mapping_gate_blocking_reason_codes": mapping_artifact[
            "mapping_gate_blocking_reason_codes"
        ],
        "target_free_liveness_gate_passed": report[
            "target_free_liveness_gate_passed"
        ],
        "target_free_liveness_blocking_reason_codes": report[
            "target_free_liveness_blocking_reason_codes"
        ],
        "candidate_scoring_frozen": report["candidate_scoring_frozen"],
        "strictly_later_outcome_blind_collection_allowed": report[
            "strictly_later_outcome_blind_collection_allowed"
        ],
        "current_issue229_window_eligible_for_confirmatory": False,
        "current_issue229_outcomes_opened": False,
        "new_strictly_later_disjoint_confirmatory_required": True,
        "future_target_access_allowed": False,
        **_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v6_9_candidate_freeze_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "scale_audit": scale_audit,
        "mapping_artifact": mapping_artifact,
        "report": report,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def _validate_runtime_rows(
    rows: list[dict[str, Any]], *, profile: dict[str, Any]
) -> dict[str, Any]:
    mapping = profile["historical_mapping"]
    roles = {mapping["fit_role"], mapping["validation_role"]}
    counts = Counter(str(row.get("role") or "") for row in rows)
    market_ids = {
        role: {str(row["market_id"]) for row in rows if row.get("role") == role}
        for role in roles
    }
    fit_ids = market_ids[mapping["fit_role"]]
    validation_ids = market_ids[mapping["validation_role"]]
    if any(str(row.get("side") or "") not in SIDES for row in rows):
        raise ValueError("#231 runtime row side invalid")
    if any(str(row.get("action") or "") not in SBC_ACTIONS for row in rows):
        raise ValueError("#231 runtime row action invalid")
    if any(int(row["max_input_ts"]) > int(row["decision_ts"]) for row in rows):
        raise ValueError("#231 runtime row feature causality violation")
    if any(row.get("target_used_as_decision_time_input") is not False for row in rows):
        raise ValueError("#231 runtime target was used as a decision input")
    if any(
        row.get("target_available_only_post_exit_or_official_resolution") is not True
        for row in rows
    ):
        raise ValueError("#231 runtime target availability provenance invalid")
    expected_market_counts = {
        mapping["fit_role"]: int(mapping["fit_market_count"]),
        mapping["validation_role"]: int(mapping["validation_market_count"]),
    }
    roles_complete = set(counts) == roles and all(
        len(market_ids[role]) == expected_market_counts[role] for role in roles
    )
    chronological = max(
        int(row["decision_ts"])
        for row in rows
        if row["role"] == mapping["fit_role"]
    ) < min(
        int(row["decision_ts"])
        for row in rows
        if row["role"] == mapping["validation_role"]
    )
    market_disjoint = not fit_ids.intersection(validation_ids)
    split = {
        "fit_market_ids": sorted(fit_ids),
        "validation_market_ids": sorted(validation_ids),
        "fit_role": mapping["fit_role"],
        "validation_role": mapping["validation_role"],
    }
    return {
        "roles_complete": roles_complete,
        "chronological": chronological,
        "market_disjoint": market_disjoint,
        "split_hash": canonical_json_sha256(split),
    }


def _select_mapping_rows(
    rows: list[dict[str, Any]], *, role: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("role") != role:
            continue
        features = dict(row.get("features") or {})
        score = _finite_float(
            features.get("canonical_v6_2_score"), "canonical_v6_2_score"
        )
        if score <= 0.0:
            continue
        grouped.setdefault(str(row["market_id"]), []).append(row)
    selected = [
        sorted(
            group,
            key=lambda row: (
                -float(row["features"]["canonical_v6_2_score"]),
                int(row["decision_ts"]),
                str(row["action"]),
            ),
        )[0]
        for group in grouped.values()
    ]
    return sorted(selected, key=lambda row: (int(row["decision_ts"]), row["market_id"]))


def _mapping_predictions(
    rows: list[dict[str, Any]],
    *,
    intercept: float,
    standardized_slope: float,
    source_mean: float,
    source_scale: float,
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        score = float(row["features"]["canonical_v6_2_score"])
        prediction = intercept + standardized_slope * (score - source_mean) / source_scale
        item = {
            "market_id": str(row["market_id"]),
            "decision_ts": int(row["decision_ts"]),
            "max_input_ts": int(row["max_input_ts"]),
            "role": str(row["role"]),
            "side": str(row["side"]),
            "action": str(row["action"]),
            "canonical_v6_2_score": score,
            "mapped_runtime_expected_pnl_per_contract": float(prediction),
            "runtime_policy_after_cost_net_pnl_per_contract": float(
                row["runtime_policy_after_cost_net_pnl_per_contract"]
            ),
            "target_used_as_decision_time_input": False,
            "issue229_outcomes_used": False,
        }
        item["prediction_row_id"] = canonical_json_sha256(item)
        output.append(item)
    return output


def _mapping_metrics(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    baseline: float,
) -> dict[str, Any]:
    targets = np.asarray(
        [float(row["runtime_policy_after_cost_net_pnl_per_contract"]) for row in rows]
    )
    predicted = np.asarray(
        [float(row["mapped_runtime_expected_pnl_per_contract"]) for row in predictions]
    )
    constant = np.full(len(targets), baseline)
    mae = float(np.mean(np.abs(predicted - targets)))
    mse = float(np.mean(np.square(predicted - targets)))
    constant_mae = float(np.mean(np.abs(constant - targets)))
    constant_mse = float(np.mean(np.square(constant - targets)))
    positive = predicted > 0.0
    return {
        "row_count": len(rows),
        "unique_market_count": len({str(row["market_id"]) for row in rows}),
        "mae": mae,
        "mse": mse,
        "train_mean_constant": baseline,
        "train_mean_constant_mae": constant_mae,
        "train_mean_constant_mse": constant_mse,
        "relative_mae_improvement_over_train_mean_constant": (
            (constant_mae - mae) / constant_mae if constant_mae else 0.0
        ),
        "relative_mse_improvement_over_train_mean_constant": (
            (constant_mse - mse) / constant_mse if constant_mse else 0.0
        ),
        "positive_prediction_count": int(np.sum(positive)),
        "positive_prediction_target_pnl_sum_diagnostic": float(
            targets[positive].sum()
        ),
    }


def _validate_mapping_artifact(artifact: dict[str, Any]) -> None:
    if (
        artifact.get("schema_version") != MAPPING_SCHEMA_VERSION
        or artifact.get("frozen") is not True
        or artifact.get("mapping_gate_passed") is not True
        or artifact.get("mapping_gate_blocking_reason_codes") != []
        or artifact.get("side_count_hard_gate_enabled") is not False
        or artifact.get("issue229_outcomes_opened") is not False
    ):
        raise ValueError("#231 v6.9 mapping artifact is not liveness eligible")
    for field in (
        "source_score_mean",
        "source_score_scale",
        "intercept",
        "standardized_source_score_coefficient",
        "entry_threshold",
    ):
        _finite_float(artifact.get(field), field)


def _validate_target_free_rows(
    rows: list[dict[str, Any]], *, exact_market_count: int
) -> None:
    market_ids = [str(row.get("market_id") or "") for row in rows]
    if (
        len(rows) != exact_market_count
        or len(set(market_ids)) != exact_market_count
        or "" in market_ids
    ):
        raise ValueError("#231 target-free rows do not have exact unique-market support")
    for row in rows:
        if FORBIDDEN_TARGET_FIELDS.intersection(row):
            raise ValueError("#231 target-free row contains forbidden target fields")
        if str(row.get("side") or "") not in SIDES:
            raise ValueError("#231 target-free row side invalid")
        if str(row.get("action") or "") not in SBC_ACTIONS:
            raise ValueError("#231 target-free row action invalid")
        if int(row["max_input_ts"]) > int(row["decision_ts"]):
            raise ValueError("#231 target-free feature causality violation")
        if row.get("labels_outcomes_resolution_or_pnl_opened") is not False:
            raise ValueError("#231 target-free target sealing invalid")
        _finite_float(row.get("v6_7_base_score"), "v6_7_base_score")


def _apply_mapping_value(score: float, artifact: dict[str, Any]) -> float:
    return float(artifact["intercept"]) + float(
        artifact["standardized_source_score_coefficient"]
    ) * (score - float(artifact["source_score_mean"])) / float(
        artifact["source_score_scale"]
    )


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(values.size),
        "minimum": float(values.min()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "maximum": float(values.max()),
    }


def _side_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row["side"]) for row in rows)
    return {side: counts[side] for side in SIDES}


def _finite_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _scale_audit_markdown(audit: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6.9 Calibration Scale Contract Audit",
            "",
            f"- additive scale contract passed: `{str(audit['direct_additive_scale_contract_passed']).lower()}`",
            f"- v6.8 correction: `{audit['failed_v6_8_pooled_residual_upper_confidence_bound']}`",
            f"- source score distribution: `{audit['source_score_distribution']}`",
            f"- actions after failed correction: `{audit['positive_source_score_count_after_failed_correction']}`",
            f"- blockers: `{audit['scale_contract_blocking_reason_codes']}`",
            f"- required replacement: `{audit['required_replacement']}`",
            "- issue #229 outcomes opened: `false`",
            "- paper/live/write/wallet/capital unlock: `false`",
            "",
        ]
    )


def _liveness_markdown(
    report: dict[str, Any], mapping_artifact: dict[str, Any]
) -> str:
    return "\n".join(
        [
            "# v6.9 Target-Free Action Liveness",
            "",
            f"- mapping gate passed: `{str(mapping_artifact['mapping_gate_passed']).lower()}`",
            f"- validation metrics: `{mapping_artifact['validation_metrics']}`",
            f"- source markets: `{report['source_target_free_market_count']}`",
            f"- positive mapped markets: `{report['positive_mapped_score_unique_market_count']}`",
            f"- guard-accepted markets: `{report['guard_accepted_unique_market_count']}`",
            f"- accepted side distribution diagnostic: `{report['guard_accepted_side_distribution_diagnostic']}`",
            f"- liveness passed: `{str(report['target_free_liveness_gate_passed']).lower()}`",
            f"- blockers: `{report['target_free_liveness_blocking_reason_codes']}`",
            "- side quota: `disabled`",
            "- issue #229 outcomes opened: `false`",
            "- issue #229 window eligible for confirmatory: `false`",
            "- new strictly-later disjoint confirmatory required: `true`",
            "- paper/live/write/wallet/capital unlock: `false`",
            "",
        ]
    )


__all__ = [
    "CalibrationScaleAlignedV69Config",
    "apply_v6_9_score_to_runtime_pnl_mapping",
    "build_v6_9_scale_contract_audit",
    "build_v6_9_target_free_liveness_report",
    "fit_v6_9_score_to_runtime_pnl_mapping",
    "run_calibration_scale_aligned_v6_9",
    "validate_calibration_scale_aligned_v6_9_profile",
]
