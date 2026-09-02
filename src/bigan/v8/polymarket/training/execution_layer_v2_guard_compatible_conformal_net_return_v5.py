"""Freeze the #203 guard-compatible split-conformal net-return v5 candidate."""

from __future__ import annotations

import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_guard_compatible_direct_net_return_v4 import (
    TARGET_FIELDS,
    _blocked_safety_fields,
    _is_sha1,
    _is_sha256,
    _predict_regressor,
    _row_key,
    _row_sort_key,
    _strip_target_fields,
    _train_regressor,
    _verify_file_hash,
    _write_json_fsync,
    _write_jsonl_fsync,
    _write_text_fsync,
)
from bigan.v8.polymarket.training.execution_layer_v2_p_up_aligned_action_value_support import (
    build_execution_compatible_action_universe,
)
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb import (
    validate_pairwise_action_advantage_lcb_feature_contract,
)
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb_fit import (
    _descriptor,
    _find_fields,
    _load_json,
    _load_jsonl,
    _materialize_role_action_rows,
    _require_sha256,
    _sha256_file,
    _verified_descriptor,
)
from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_action_value import (
    REQUIRED_ACTIONS,
)
from bigan.v8.polymarket.training.post_freeze_o_replay_aligned_source_ranking import (
    _v8_execution_guard_config,
)

PROFILE_SCHEMA_VERSION = "bigan-v8-guard-compatible-conformal-net-return-v5-fit-profile-v1"
SCHEMA_PREFIX = "bigan-v8-guard-compatible-conformal-net-return-v5"
CANDIDATE_NAME = "guard_compatible_conformal_net_return_v5"
FIT_ROLES = ("development_train", "development_calibration")
CALIBRATION_ROLE = "confirmatory_validation"
MODEL_FILENAME = "guard_compatible_conformal_net_return_v5.xgb.json"
TRADE_ACTIONS = tuple(action for action in REQUIRED_ACTIONS if action != "NO_TRADE")


@dataclass(frozen=True, slots=True)
class GuardCompatibleConformalNetReturnV5Config:
    """Pinned train/calibration inputs for the prospective #203 candidate."""

    run_id: str
    output_dir: Path | str
    fit_profile_path: Path | str
    expected_fit_profile_sha256: str
    role_assignment_manifest_path: Path | str
    expected_role_assignment_manifest_sha256: str
    accepted_bet_power_manifest_path: Path | str
    expected_accepted_bet_power_manifest_sha256: str
    accepted_bet_power_report_path: Path | str
    expected_accepted_bet_power_report_sha256: str
    issue201_manifest_path: Path | str
    expected_issue201_manifest_sha256: str
    implementation_commit: str
    candidate_freeze_created_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for name in (
            "expected_fit_profile_sha256",
            "expected_role_assignment_manifest_sha256",
            "expected_accepted_bet_power_manifest_sha256",
            "expected_accepted_bet_power_report_sha256",
            "expected_issue201_manifest_sha256",
        ):
            _require_sha256(str(getattr(self, name)), name=name)
        if not _is_sha1(self.implementation_commit):
            raise ValueError("implementation_commit must be a Git SHA-1")
        if int(self.candidate_freeze_created_ts) <= 0:
            raise ValueError("candidate_freeze_created_ts must be positive")
        for name in (
            "output_dir",
            "fit_profile_path",
            "role_assignment_manifest_path",
            "accepted_bet_power_manifest_path",
            "accepted_bet_power_report_path",
            "issue201_manifest_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


def validate_guard_compatible_conformal_net_return_v5_profile(
    profile: dict[str, Any],
) -> None:
    """Reject drift from the prospective train/calibration-only v5 contract."""

    roles = dict(profile.get("roles") or {})
    model = dict(profile.get("model") or {})
    calibration = dict(profile.get("conformal_calibration") or {})
    decision = dict(profile.get("decision_rule") or {})
    future = dict(profile.get("future_evaluation") or {})
    access = dict(profile.get("access_sequence") or {})
    output = dict(profile.get("output_contract") or {})
    hashes = (
        "parent_issue_201_manifest_sha256",
        "role_assignment_manifest_sha256",
        "role_assignment_rows_sha256",
        "feature_contract_sha256",
        "accepted_bet_power_manifest_sha256",
        "accepted_bet_power_report_sha256",
        "execution_guard_config_sha256",
    )
    checks = {
        "schema": profile.get("schema_version") == PROFILE_SCHEMA_VERSION,
        "candidate": profile.get("candidate_name") == CANDIDATE_NAME,
        "frozen": profile.get("frozen") is True,
        "parent_commit": _is_sha1(str(profile.get("parent_issue_202_commit") or "")),
        "hashes": all(_is_sha256(str(profile.get(name) or "")) for name in hashes),
        "roles": roles
        == {
            "fit": list(FIT_ROLES),
            "fit_market_count": 135,
            "calibration": CALIBRATION_ROLE,
            "calibration_market_count": 60,
            "calibration_role_is_not_policy_validation": True,
            "future_policy_evaluation": "issue_192_strictly_later_persistent_window_only",
        },
        "actions": profile.get("required_actions") == list(REQUIRED_ACTIONS),
        "fixed_model": model.get("objective") == "reg:squarederror"
        and model.get("eval_metric") == "rmse"
        and model.get("target") == "target_net_pnl_per_contract"
        and model.get("num_boost_round") == 120
        and model.get("max_depth") == 3
        and math.isclose(float(model.get("eta") or 0.0), 0.03)
        and math.isclose(float(model.get("reg_alpha") or 0.0), 1.0)
        and math.isclose(float(model.get("reg_lambda") or 0.0), 10.0)
        and model.get("nthread") == 1
        and model.get("hyperparameter_search_enabled") is False,
        "calibration": calibration.get("method")
        == "market_grouped_split_conformal_one_sided_lower_prediction_bound"
        and math.isclose(float(calibration.get("alpha") or 0.0), 0.1)
        and calibration.get("nonconformity_score")
        == "maximum_raw_prediction_minus_target_per_market_and_group"
        and calibration.get("finite_sample_quantile_rank")
        == "ceil((market_count + 1) * (1 - alpha))"
        and calibration.get("quantile_interpolation") == "higher_order_statistic_no_interpolation"
        and calibration.get("minimum_action_calibration_market_count") == 30
        and calibration.get("minimum_family_calibration_market_count") == 30
        and calibration.get("minimum_global_calibration_market_count") == 30
        and calibration.get("fallback_order") == ["action", "action_family", "all_trade_actions"]
        and math.isclose(
            float(calibration.get("maximum_absolute_calibration_penalty") or 0.0),
            2.0,
        )
        and calibration.get("no_trade_prediction") == 0.0
        and calibration.get("no_trade_calibration_penalty") == 0.0
        and calibration.get("policy_pnl_computed_on_calibration") is False
        and calibration.get("calibration_threshold_search_enabled") is False
        and calibration.get("candidate_comparison_enabled") is False,
        "decision": decision.get("method")
        == "guard_compatible_mask_before_conformal_net_return_lcb_argmax"
        and decision.get("score_field") == "conformal_net_return_lower_bound"
        and decision.get("no_trade_score") == 0.0
        and decision.get("minimum_selected_lower_bound_exclusive") == 0.0
        and decision.get("minimum_runner_up_margin_exclusive") == 0.0
        and decision.get("p_up_side_alignment_required") is True
        and decision.get("frozen_execution_quality_required") is True
        and decision.get("mask_score") == -1_000_000.0
        and all(
            decision.get(name) is False
            for name in (
                "model_score_mutation_allowed",
                "execution_guard_mutation_allowed",
                "cost_model_mutation_allowed",
                "order_sizing_mutation_allowed",
                "exposure_policy_mutation_allowed",
            )
        ),
        "future": future.get("eligible_collection")
        == "issue_192_strictly_later_persistent_window_only"
        and future.get("issue_190_collection_eligible") is False
        and future.get("minimum_quality_valid_market_count") == 220
        and future.get("minimum_guard_accepted_unique_market_count") == 88
        and future.get("minimum_supported_side_market_count") == 10
        and future.get("pnl_hard_gate_aggregation") == "selected_side_buy_up_buy_down_only"
        and future.get("action_and_family_pnl_diagnostic_only") is True
        and future.get("accepted_bet_total_post_cost_pnl_minimum_exclusive") == 0.0
        and future.get("all_market_policy_pnl_bootstrap_lcb_minimum_exclusive") == 0.0
        and future.get("largest_winner_removed_pnl_minimum_exclusive") == 0.0
        and future.get("bootstrap_unit") == "market_id"
        and future.get("bootstrap_confidence_level") == 0.95
        and future.get("bootstrap_resample_count") == 5000
        and future.get("result_dependent_extension_allowed") is False
        and future.get("single_use_holdout") is True,
        "access": access.get("pre_label_audit_required") is True
        and access.get("role_assignment_metadata_may_be_opened_before_audit") is True
        and access.get("development_train_labels_may_be_opened_after_audit") is True
        and access.get("development_calibration_labels_may_be_opened_for_fit_after_audit") is True
        and access.get(
            "confirmatory_validation_labels_may_be_opened_for_conformal_calibration_after_audit"
        )
        is True
        and access.get("calibration_labels_used_only_for_nonconformity_quantiles") is True
        and all(
            access.get(name) is False
            for name in (
                "issue_202_oof_or_gate_artifacts_may_be_opened",
                "issue_190_or_192_future_files_may_be_opened_during_fit",
                "calibration_policy_pnl_may_be_computed",
                "future_accepted_bet_pnl_may_be_opened_during_fit",
            )
        ),
        "output": output.get("research_candidate_only") is True
        and output.get("candidate_freeze_must_precede_issue_192_collection") is True
        and output.get("issue_190_collection_started_before_candidate_freeze_and_is_not_eligible")
        is True
        and output.get("strictly_later_persistent_window_required") is True
        and output.get("future_evaluation_requires_calibration_gate") is True
        and output.get("promotion_evidence_created") is False,
        "safety": dict(profile.get("safety") or {}) == _blocked_safety_fields(),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("invalid #203 fit profile: " + ", ".join(failed))


def fit_guard_compatible_conformal_net_return_v5(
    config: GuardCompatibleConformalNetReturnV5Config,
) -> dict[str, Any]:
    """Fit on train, calibrate uncertainty only, and freeze before #192."""

    paths = {
        "profile": config.fit_profile_path.resolve(),
        "role_manifest": config.role_assignment_manifest_path.resolve(),
        "power_manifest": config.accepted_bet_power_manifest_path.resolve(),
        "power_report": config.accepted_bet_power_report_path.resolve(),
        "issue201_manifest": config.issue201_manifest_path.resolve(),
    }
    expected = {
        "profile": config.expected_fit_profile_sha256,
        "role_manifest": config.expected_role_assignment_manifest_sha256,
        "power_manifest": config.expected_accepted_bet_power_manifest_sha256,
        "power_report": config.expected_accepted_bet_power_report_sha256,
        "issue201_manifest": config.expected_issue201_manifest_sha256,
    }
    for name, path in paths.items():
        _verify_file_hash(path, expected[name], name=name)
    profile = {**_load_json(paths["profile"]), "fit_profile_sha256": expected["profile"]}
    validate_guard_compatible_conformal_net_return_v5_profile(profile)
    _validate_expected_hashes(profile, expected=expected)

    role_manifest = _load_json(paths["role_manifest"])
    power_manifest = _load_json(paths["power_manifest"])
    power_report = _load_json(paths["power_report"])
    lineage = _validate_prelabel_lineage(
        profile=profile,
        role_manifest=role_manifest,
        power_manifest=power_manifest,
        power_report=power_report,
    )
    run_dir = Path(config.output_dir) / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    pre_label = {
        "schema_version": f"{SCHEMA_PREFIX}-pre-label-access-audit-v1",
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "fit_profile": _descriptor(paths["profile"]),
        "role_assignment_manifest": _descriptor(paths["role_manifest"]),
        "role_assignment_rows": lineage["role_rows_descriptor"],
        "feature_contract": lineage["feature_contract_descriptor"],
        "accepted_bet_power_manifest": _descriptor(paths["power_manifest"]),
        "accepted_bet_power_report": _descriptor(paths["power_report"]),
        "issue201_manifest_hash_verified": True,
        "issue201_manifest_content_opened": False,
        "issue202_oof_or_gate_artifacts_opened": False,
        "role_assignment_metadata_opened": True,
        "feature_label_resolution_or_pnl_files_opened_before_audit": False,
        "development_train_or_calibration_target_hashes_read_before_audit": False,
        "future_files_opened": False,
        "fit_and_calibration_rules_frozen_before_target_access": True,
        "role_market_counts": role_manifest["role_market_counts"],
        "pre_label_access_validation_passed": True,
        **_blocked_safety_fields(),
    }
    pre_label["audit_id"] = canonical_json_sha256(pre_label)
    pre_label_path = run_dir / "pre_label_access_lineage_audit.json"
    _write_json_fsync(pre_label_path, pre_label)
    _write_text_fsync(
        run_dir / "pre_label_access_lineage_audit.md",
        _pre_label_markdown(pre_label),
    )

    action_rows_by_role, corpus_audits = _materialize_role_action_rows(
        lineage["role_rows"],
        feature_columns=lineage["feature_columns"],
        roles=(*FIT_ROLES, CALIBRATION_ROLE),
    )
    train_rows = sorted(
        [row for role in FIT_ROLES for row in action_rows_by_role[role]],
        key=_row_sort_key,
    )
    calibration_rows = action_rows_by_role[CALIBRATION_ROLE]
    _validate_action_rows(train_rows, role="combined_development_fit", expected_market_count=135)
    _validate_action_rows(
        calibration_rows,
        role=CALIBRATION_ROLE,
        expected_market_count=60,
    )
    if any(audit["blocking_reason_codes"] for audit in corpus_audits):
        raise ValueError("development corpus integrity failed")
    train_max_ts = max(int(row["decision_ts"]) for row in train_rows)
    calibration_min_ts = min(int(row["decision_ts"]) for row in calibration_rows)
    if train_max_ts >= calibration_min_ts:
        raise ValueError("development_train does not strictly precede calibration")

    train_rows_path = run_dir / "conformal_v5_development_train_action_rows.jsonl"
    calibration_rows_path = run_dir / "conformal_v5_development_calibration_action_rows.jsonl"
    _write_jsonl_fsync(train_rows_path, train_rows)
    _write_jsonl_fsync(calibration_rows_path, calibration_rows)

    booster = _train_regressor(
        train_rows,
        feature_columns=lineage["feature_columns"],
        model_config=dict(profile["model"]),
    )
    model_path = run_dir / MODEL_FILENAME
    booster.save_model(model_path)

    train_predictions = _raw_target_stripped_predictions(
        booster,
        train_rows,
        feature_columns=lineage["feature_columns"],
    )
    calibration_predictions = _raw_target_stripped_predictions(
        booster,
        calibration_rows,
        feature_columns=lineage["feature_columns"],
    )
    train_prediction_path = run_dir / "conformal_v5_target_stripped_train_predictions.jsonl"
    calibration_prediction_path = (
        run_dir / "conformal_v5_target_stripped_calibration_predictions.jsonl"
    )
    _write_jsonl_fsync(train_prediction_path, train_predictions)
    _write_jsonl_fsync(calibration_prediction_path, calibration_predictions)
    _validate_target_stripped_rows(train_predictions, expected_count=2700)
    _validate_target_stripped_rows(calibration_predictions, expected_count=1200)

    calibration_artifact = build_market_grouped_conformal_artifact(
        calibration_predictions,
        target_rows=calibration_rows,
        profile=profile,
        feature_contract_sha256=profile["feature_contract_sha256"],
    )
    calibration_artifact_path = run_dir / "conformal_v5_calibration_artifact.json"
    _write_json_fsync(calibration_artifact_path, calibration_artifact)
    scored_calibration_rows = apply_conformal_scores(
        calibration_predictions,
        calibration_artifact=calibration_artifact,
        profile=profile,
    )
    scored_calibration_path = run_dir / "conformal_v5_target_stripped_calibration_scored_rows.jsonl"
    _write_jsonl_fsync(scored_calibration_path, scored_calibration_rows)
    _validate_target_stripped_rows(scored_calibration_rows, expected_count=1200)

    calibration_gate = _build_calibration_gate(
        profile=profile,
        train_rows=train_rows,
        calibration_rows=calibration_rows,
        calibration_predictions=calibration_predictions,
        calibration_artifact=calibration_artifact,
        corpus_audits=corpus_audits,
        train_max_ts=train_max_ts,
        calibration_min_ts=calibration_min_ts,
    )
    calibration_report = {
        "schema_version": f"{SCHEMA_PREFIX}-calibration-report-v1",
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "source_split": "development_calibration_only",
        "method": calibration_artifact["method"],
        "calibration_market_count": 60,
        "calibration_action_row_count": len(calibration_rows),
        "action_calibration": calibration_artifact["actions"],
        "calibration_gate_checks": calibration_gate["checks"],
        "calibration_gate_passed": calibration_gate["passed"],
        "calibration_gate_blocking_reason_codes": calibration_gate["reason_codes"],
        "policy_pnl_computed_on_calibration": False,
        "calibration_threshold_search_enabled": False,
        "candidate_comparison_enabled": False,
        "issue202_oof_or_gate_artifacts_opened": False,
        "uses_current_oof_validation_confirmatory_or_future_pnl_for_tuning": False,
        "future_labels_opened": False,
        **_blocked_safety_fields(),
    }
    calibration_report["report_id"] = canonical_json_sha256(calibration_report)
    calibration_report_path = run_dir / "conformal_v5_calibration_report.json"
    _write_json_fsync(calibration_report_path, calibration_report)
    _write_text_fsync(
        run_dir / "conformal_v5_calibration_report.md",
        _calibration_markdown(calibration_report),
    )

    future_protocol = _build_future_evaluation_protocol(
        run_id=config.run_id,
        profile=profile,
        candidate_freeze_created_ts=int(config.candidate_freeze_created_ts),
        calibration_gate_passed=calibration_gate["passed"],
        power_report=power_report,
    )
    future_protocol_path = run_dir / "conformal_v5_future_evaluation_protocol.json"
    _write_json_fsync(future_protocol_path, future_protocol)
    _write_text_fsync(
        run_dir / "conformal_v5_future_evaluation_protocol.md",
        _future_protocol_markdown(future_protocol),
    )

    training_report = {
        "schema_version": f"{SCHEMA_PREFIX}-training-report-v1",
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "primary_policy_target": "action_expected_net_return",
        "action_value_model_family": "guard_compatible_split_conformal_direct_net_return_model",
        "feature_conditioned_action_value_model_enabled": True,
        "model_objective": profile["model"]["objective"],
        "model_target": profile["model"]["target"],
        "training_target_includes_costs": True,
        "fit_roles": list(FIT_ROLES),
        "fit_market_count": 135,
        "fit_action_row_count": len(train_rows),
        "calibration_role": CALIBRATION_ROLE,
        "calibration_role_is_not_policy_validation": True,
        "calibration_market_count": 60,
        "calibration_action_row_count": len(calibration_rows),
        "train_max_decision_ts": train_max_ts,
        "calibration_min_decision_ts": calibration_min_ts,
        "train_strictly_precedes_calibration": True,
        "hyperparameter_search_enabled": False,
        "calibration_threshold_search_enabled": False,
        "model": _descriptor(model_path),
        "calibration_artifact": _descriptor(calibration_artifact_path),
        "calibration_report": _descriptor(calibration_report_path),
        "calibration_gate_passed": calibration_gate["passed"],
        "issue202_oof_or_gate_artifacts_opened": False,
        "future_files_opened": False,
        "uses_current_oof_validation_confirmatory_or_future_pnl_for_tuning": False,
        **_blocked_safety_fields(),
    }
    training_report["report_id"] = canonical_json_sha256(training_report)
    training_report_path = run_dir / "conformal_v5_training_report.json"
    _write_json_fsync(training_report_path, training_report)

    candidate_allowed = bool(calibration_gate["passed"])
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-research-candidate-freeze-manifest-v1",
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        "candidate_freeze_created_ts": int(config.candidate_freeze_created_ts),
        "research_candidate_frozen": candidate_allowed,
        "research_candidate_only": True,
        "fit_profile": _descriptor(paths["profile"]),
        "pre_label_access_audit": _descriptor(pre_label_path),
        "role_assignment_manifest": _descriptor(paths["role_manifest"]),
        "accepted_bet_power_manifest": _descriptor(paths["power_manifest"]),
        "accepted_bet_power_report": _descriptor(paths["power_report"]),
        "development_train_action_rows": _descriptor(train_rows_path),
        "development_calibration_action_rows": _descriptor(calibration_rows_path),
        "target_stripped_train_predictions": _descriptor(train_prediction_path),
        "target_stripped_calibration_predictions": _descriptor(calibration_prediction_path),
        "target_stripped_calibration_scored_rows": _descriptor(scored_calibration_path),
        "model": _descriptor(model_path),
        "calibration_artifact": _descriptor(calibration_artifact_path),
        "calibration_report": _descriptor(calibration_report_path),
        "training_report": _descriptor(training_report_path),
        "future_evaluation_protocol": _descriptor(future_protocol_path),
        "model_sha256": _sha256_file(model_path),
        "policy_dataset_hash": _sha256_file(train_rows_path),
        "split_hash": canonical_json_sha256(
            {
                "role_assignment_rows_sha256": profile["role_assignment_rows_sha256"],
                "fit_roles": list(FIT_ROLES),
                "calibration_role": CALIBRATION_ROLE,
            }
        ),
        "calibration_gate_passed": calibration_gate["passed"],
        "candidate_specific_future_evaluation_allowed": candidate_allowed,
        "candidate_specific_future_evaluation_blocking_reason_codes": calibration_gate[
            "reason_codes"
        ],
        "eligible_future_collection": (
            "issue_192_strictly_later_persistent_window_only" if candidate_allowed else None
        ),
        "issue_190_collection_eligible_for_this_candidate": False,
        "issue_192_collection_must_start_after_candidate_freeze": True,
        "issue202_oof_or_gate_artifacts_opened": False,
        "future_files_opened": False,
        "calibration_policy_pnl_computed": False,
        "uses_current_oof_validation_confirmatory_or_future_pnl_for_tuning": False,
        "result_driven_rerun_or_parameter_change_allowed": False,
        "guard_cost_threshold_sizing_or_exposure_mutated": False,
        **_blocked_safety_fields(),
    }
    manifest["research_candidate_hash"] = canonical_json_sha256(
        {
            "candidate_name": CANDIDATE_NAME,
            "implementation_commit": config.implementation_commit,
            "fit_profile_sha256": expected["profile"],
            "model_sha256": manifest["model_sha256"],
            "calibration_artifact_sha256": manifest["calibration_artifact"]["sha256"],
            "policy_dataset_hash": manifest["policy_dataset_hash"],
            "split_hash": manifest["split_hash"],
            "future_protocol_sha256": manifest["future_evaluation_protocol"]["sha256"],
        }
    )
    manifest_path = run_dir / "conformal_v5_research_candidate_freeze_manifest.json"
    _write_json_fsync(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "calibration_report": calibration_report,
        "future_protocol": future_protocol,
    }


def build_market_grouped_conformal_artifact(
    calibration_predictions: list[dict[str, Any]],
    *,
    target_rows: list[dict[str, Any]],
    profile: dict[str, Any],
    feature_contract_sha256: str,
) -> dict[str, Any]:
    """Build fixed one-sided market-grouped residual bounds without policy PnL."""

    calibration = dict(profile["conformal_calibration"])
    alpha = float(calibration["alpha"])
    targets = {_row_key(row): float(row["target_net_pnl_per_contract"]) for row in target_rows}
    if set(targets) != {_row_key(row) for row in calibration_predictions}:
        raise ValueError("calibration prediction and target identity mismatch")
    no_trade_targets = [
        targets[_row_key(row)] for row in calibration_predictions if row["action"] == "NO_TRADE"
    ]
    if not no_trade_targets or any(abs(value) > 1e-12 for value in no_trade_targets):
        raise ValueError("NO_TRADE calibration target must be the exact zero anchor")
    residual_rows = [
        {
            "market_id": str(row["market_id"]),
            "action": str(row["action"]),
            "action_family": str(row["action_family"]),
            "residual": float(row["raw_direct_predicted_net_return"]) - targets[_row_key(row)],
        }
        for row in calibration_predictions
        if row["action"] != "NO_TRADE"
    ]
    action_groups = {
        action: _group_quantile(
            [row for row in residual_rows if row["action"] == action],
            alpha=alpha,
            group_name=f"action:{action}",
        )
        for action in TRADE_ACTIONS
    }
    families = sorted({str(row["action_family"]) for row in residual_rows})
    family_groups = {
        family: _group_quantile(
            [row for row in residual_rows if row["action_family"] == family],
            alpha=alpha,
            group_name=f"family:{family}",
        )
        for family in families
    }
    global_group = _group_quantile(
        residual_rows,
        alpha=alpha,
        group_name="all_trade_actions",
    )
    actions: dict[str, Any] = {}
    for action in REQUIRED_ACTIONS:
        if action == "NO_TRADE":
            actions[action] = {
                "calibration_source": "frozen_no_trade_zero_anchor",
                "calibration_penalty": 0.0,
                "calibration_market_count": len(
                    {row["market_id"] for row in calibration_predictions}
                ),
                "empirical_market_simultaneous_coverage": 1.0,
                "support_passed": True,
            }
            continue
        family = _action_family(action)
        action_group = action_groups[action]
        family_group = family_groups[family]
        if action_group["market_count"] >= int(
            calibration["minimum_action_calibration_market_count"]
        ):
            selected_group = action_group
            source = "action"
        elif family_group["market_count"] >= int(
            calibration["minimum_family_calibration_market_count"]
        ):
            selected_group = family_group
            source = "action_family"
        elif global_group["market_count"] >= int(
            calibration["minimum_global_calibration_market_count"]
        ):
            selected_group = global_group
            source = "all_trade_actions"
        else:
            selected_group = {
                **global_group,
                "quantile": float("nan"),
            }
            source = "insufficient_support_fail_closed"
        penalty = float(selected_group["quantile"])
        actions[action] = {
            "calibration_source": source,
            "calibration_group_name": selected_group["group_name"],
            "calibration_penalty": penalty,
            "calibration_market_count": selected_group["market_count"],
            "quantile_rank": selected_group["quantile_rank"],
            "one_sided_alpha": alpha,
            "empirical_market_simultaneous_coverage": selected_group[
                "empirical_market_simultaneous_coverage"
            ],
            "support_passed": source != "insufficient_support_fail_closed",
        }
    artifact = {
        "schema_version": f"{SCHEMA_PREFIX}-calibration-artifact-v1",
        "candidate_name": CANDIDATE_NAME,
        "source_split": "development_calibration_only",
        "method": calibration["method"],
        "nonconformity_score": calibration["nonconformity_score"],
        "finite_sample_quantile_rank": calibration["finite_sample_quantile_rank"],
        "quantile_interpolation": calibration["quantile_interpolation"],
        "decision_score_formula": ("raw_direct_predicted_net_return - action_calibration_penalty"),
        "feature_contract_sha256": feature_contract_sha256,
        "action_groups": action_groups,
        "family_groups": family_groups,
        "global_group": global_group,
        "actions": actions,
        "policy_pnl_computed_on_calibration": False,
        "calibration_threshold_search_enabled": False,
        "candidate_comparison_enabled": False,
        "uses_current_oof_validation_confirmatory_or_future_pnl_for_tuning": False,
        **_blocked_safety_fields(),
    }
    artifact["calibration_artifact_id"] = canonical_json_sha256(artifact)
    return artifact


def apply_conformal_scores(
    predictions: list[dict[str, Any]],
    *,
    calibration_artifact: dict[str, Any],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply frozen conformal penalties and compatibility masks to target-free rows."""

    if _find_fields({"rows": predictions}, set(TARGET_FIELDS)):
        raise ValueError("conformal inference rows contain target fields")
    decision_rows = sorted(predictions, key=_row_sort_key)
    compatibility_rows = build_execution_compatible_action_universe(decision_rows)
    compatibility = {
        _row_key(row): bool(row["p_up_alignment_passed"] and row["execution_quality_only_passed"])
        for row in compatibility_rows
    }
    mask_score = float(profile["decision_rule"]["mask_score"])
    output = []
    for row in decision_rows:
        action = str(row["action"])
        raw = 0.0 if action == "NO_TRADE" else float(row["raw_direct_predicted_net_return"])
        calibration_row = dict(calibration_artifact["actions"][action])
        penalty = float(calibration_row["calibration_penalty"])
        compatible = action == "NO_TRADE" or compatibility[_row_key(row)]
        lower_bound = raw - penalty
        if action == "NO_TRADE":
            selection_score = 0.0
            score_source = "frozen_no_trade_zero_anchor"
        elif not compatible:
            selection_score = mask_score
            score_source = "masked_by_frozen_execution_compatibility"
        elif not math.isfinite(lower_bound):
            selection_score = mask_score
            score_source = "masked_by_invalid_conformal_bound"
        else:
            selection_score = lower_bound
            score_source = "market_grouped_split_conformal_net_return_lcb"
        updated = {
            **row,
            "conformal_calibration_source": calibration_row["calibration_source"],
            "conformal_calibration_penalty": penalty,
            "conformal_net_return_lower_bound": lower_bound,
            "guard_compatible_before_ranking": compatible,
            "guard_compatibility_mask_applied_before_argmax": True,
            "action_selection_score": selection_score,
            "action_advantage_lcb_net_return": selection_score,
            "calibrated_action_expected_net_return": raw,
            "ranking_score_source": score_source,
            "target_used_as_decision_input": False,
            "outcome_fields_used_as_decision_input": False,
            "target_or_outcome_fields_used": False,
        }
        updated["v5_prediction_row_sha256"] = canonical_json_sha256(updated)
        output.append(updated)
    return output


def _raw_target_stripped_predictions(
    booster: xgb.Booster,
    rows: list[dict[str, Any]],
    *,
    feature_columns: tuple[str, ...],
) -> list[dict[str, Any]]:
    decision_rows = sorted((_strip_target_fields(row) for row in rows), key=_row_sort_key)
    values = _predict_regressor(booster, decision_rows, feature_columns=feature_columns)
    output = []
    for row, value in zip(decision_rows, values, strict=True):
        updated = {
            **row,
            "raw_model_prediction": value,
            "raw_direct_predicted_net_return": 0.0 if row["action"] == "NO_TRADE" else value,
            "raw_prediction_source": (
                "frozen_no_trade_zero_anchor"
                if row["action"] == "NO_TRADE"
                else "train_only_direct_net_return_model"
            ),
            "target_used_as_decision_input": False,
            "outcome_fields_used_as_decision_input": False,
            "target_or_outcome_fields_used": False,
        }
        updated["v5_raw_prediction_row_sha256"] = canonical_json_sha256(updated)
        output.append(updated)
    return output


def _group_quantile(
    rows: list[dict[str, Any]],
    *,
    alpha: float,
    group_name: str,
) -> dict[str, Any]:
    residuals_by_market: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        residual = float(row["residual"])
        if not math.isfinite(residual):
            raise ValueError("calibration residual must be finite")
        residuals_by_market[str(row["market_id"])].append(residual)
    market_maxima = [
        max(residuals_by_market[market_id]) for market_id in sorted(residuals_by_market)
    ]
    market_count = len(market_maxima)
    if market_count == 0:
        return {
            "group_name": group_name,
            "market_count": 0,
            "quantile_rank": 0,
            "quantile": float("nan"),
            "empirical_market_simultaneous_coverage": 0.0,
        }
    rank = min(market_count, math.ceil((market_count + 1) * (1.0 - alpha)))
    quantile = float(sorted(market_maxima)[rank - 1])
    coverage = float(np.mean(np.asarray(market_maxima) <= quantile))
    return {
        "group_name": group_name,
        "market_count": market_count,
        "quantile_rank": rank,
        "quantile": quantile,
        "empirical_market_simultaneous_coverage": coverage,
    }


def _build_calibration_gate(
    *,
    profile: dict[str, Any],
    train_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    calibration_predictions: list[dict[str, Any]],
    calibration_artifact: dict[str, Any],
    corpus_audits: list[dict[str, Any]],
    train_max_ts: int,
    calibration_min_ts: int,
) -> dict[str, Any]:
    calibration = dict(profile["conformal_calibration"])
    trade_rows = [calibration_artifact["actions"][action] for action in TRADE_ACTIONS]
    checks = {
        "fit_market_support": len({row["market_id"] for row in train_rows}) == 135,
        "calibration_market_support": len({row["market_id"] for row in calibration_rows}) == 60,
        "train_strictly_precedes_calibration": train_max_ts < calibration_min_ts,
        "complete_five_action_grid": len(train_rows) == 2700 and len(calibration_rows) == 1200,
        "feature_causality": all(
            int(row["max_input_ts"]) <= int(row["decision_ts"])
            for row in train_rows + calibration_rows
        ),
        "target_stripped_calibration_predictions": not _find_fields(
            {"rows": calibration_predictions}, set(TARGET_FIELDS)
        ),
        "action_calibration_support": all(row["support_passed"] for row in trade_rows),
        "finite_bounded_calibration_penalties": all(
            math.isfinite(float(row["calibration_penalty"]))
            and abs(float(row["calibration_penalty"]))
            <= float(calibration["maximum_absolute_calibration_penalty"])
            for row in trade_rows
        ),
        "nominal_empirical_market_coverage": all(
            float(row["empirical_market_simultaneous_coverage"])
            >= 1.0 - float(calibration["alpha"])
            for row in trade_rows
        ),
        "corpus_integrity": all(not audit["blocking_reason_codes"] for audit in corpus_audits),
        "no_policy_pnl_or_candidate_selection_on_calibration": (
            calibration_artifact["policy_pnl_computed_on_calibration"] is False
            and calibration_artifact["calibration_threshold_search_enabled"] is False
            and calibration_artifact["candidate_comparison_enabled"] is False
        ),
        "no_current_oof_validation_confirmatory_or_future_pnl_tuning": (
            calibration_artifact[
                "uses_current_oof_validation_confirmatory_or_future_pnl_for_tuning"
            ]
            is False
        ),
    }
    reason_map = {
        "fit_market_support": "fit_market_support_failed",
        "calibration_market_support": "calibration_market_support_failed",
        "train_strictly_precedes_calibration": "train_calibration_chronology_failed",
        "complete_five_action_grid": "five_action_grid_incomplete",
        "feature_causality": "feature_causality_violation",
        "target_stripped_calibration_predictions": "target_present_in_calibration_prediction",
        "action_calibration_support": "action_calibration_support_failed",
        "finite_bounded_calibration_penalties": "calibration_penalty_invalid_or_unbounded",
        "nominal_empirical_market_coverage": "nominal_empirical_market_coverage_failed",
        "corpus_integrity": "corpus_integrity_failed",
        "no_policy_pnl_or_candidate_selection_on_calibration": (
            "calibration_policy_pnl_or_selection_present"
        ),
        "no_current_oof_validation_confirmatory_or_future_pnl_tuning": (
            "current_or_future_pnl_used_for_tuning"
        ),
    }
    blockers = [reason_map[name] for name, passed in checks.items() if not passed]
    return {"checks": checks, "passed": not blockers, "reason_codes": blockers}


def _validate_expected_hashes(
    profile: dict[str, Any],
    *,
    expected: dict[str, str],
) -> None:
    pairs = {
        "role_assignment_manifest_sha256": expected["role_manifest"],
        "accepted_bet_power_manifest_sha256": expected["power_manifest"],
        "accepted_bet_power_report_sha256": expected["power_report"],
        "parent_issue_201_manifest_sha256": expected["issue201_manifest"],
    }
    mismatches = [name for name, value in pairs.items() if profile[name] != value]
    if mismatches:
        raise ValueError("profile input hash mismatch: " + ", ".join(mismatches))


def _validate_prelabel_lineage(
    *,
    profile: dict[str, Any],
    role_manifest: dict[str, Any],
    power_manifest: dict[str, Any],
    power_report: dict[str, Any],
) -> dict[str, Any]:
    if role_manifest.get("role_assignment_ready") is not True:
        raise ValueError("role assignment is not ready")
    if role_manifest.get("blocking_reason_codes"):
        raise ValueError("role assignment has blockers")
    if role_manifest.get("labels_or_outcomes_opened_for_role_assignment") is not False:
        raise ValueError("role assignment opened labels or outcomes")
    if role_manifest.get("role_market_counts") != {
        "development_train": 90,
        "development_calibration": 45,
        "confirmatory_validation": 60,
    }:
        raise ValueError("role market counts mismatch")
    role_rows_descriptor = _verified_descriptor(
        role_manifest.get("selected_rows"), name="role assignment rows"
    )
    if role_rows_descriptor["sha256"] != profile["role_assignment_rows_sha256"]:
        raise ValueError("role assignment rows hash mismatch")
    feature_contract_descriptor = _verified_descriptor(
        role_manifest.get("feature_contract"), name="feature contract"
    )
    if feature_contract_descriptor["sha256"] != profile["feature_contract_sha256"]:
        raise ValueError("feature contract hash mismatch")
    feature_contract = _load_json(Path(feature_contract_descriptor["path"]))
    validate_pairwise_action_advantage_lcb_feature_contract(
        feature_contract,
        expected_parent_protocol_sha256=feature_contract["parent_protocol_sha256"],
    )
    if (
        canonical_json_sha256(_v8_execution_guard_config())
        != profile["execution_guard_config_sha256"]
    ):
        raise ValueError("execution guard config hash mismatch")
    if power_manifest.get("power_analysis_ready") is not True:
        raise ValueError("accepted-bet power manifest is not ready")
    if power_report.get("power_analysis_ready") is not True:
        raise ValueError("accepted-bet power report is not ready")
    if power_report.get("uses_current_oof_validation_or_confirmatory_pnl") is not False:
        raise ValueError("power design used current outcome evidence")
    if power_report.get("uses_realized_candidate_pnl_for_design") is not False:
        raise ValueError("power design used realized candidate PnL")
    if power_report.get("recommended_required_accepted_unique_market_count") != 88:
        raise ValueError("accepted-market power target mismatch")
    if power_report.get("recommended_quality_valid_market_count") != 220:
        raise ValueError("quality-valid market power target mismatch")
    role_rows = _load_jsonl(Path(role_rows_descriptor["path"]))
    if _find_fields({"rows": role_rows}, set(TARGET_FIELDS)):
        raise ValueError("role rows contain target fields")
    return {
        "role_rows": role_rows,
        "role_rows_descriptor": role_rows_descriptor,
        "feature_contract_descriptor": feature_contract_descriptor,
        "feature_columns": tuple(str(value) for value in feature_contract["feature_columns"]),
    }


def _validate_action_rows(
    rows: list[dict[str, Any]],
    *,
    role: str,
    expected_market_count: int,
) -> None:
    expected_count = expected_market_count * 4 * len(REQUIRED_ACTIONS)
    if len(rows) != expected_count:
        raise ValueError(f"{role} action row count mismatch")
    if len({str(row["market_id"]) for row in rows}) != expected_market_count:
        raise ValueError(f"{role} market count mismatch")
    groups: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows:
        groups[(str(row["market_id"]), int(row["decision_ts"]))].add(str(row["action"]))
        if not math.isfinite(float(row["target_net_pnl_per_contract"])):
            raise ValueError(f"{role} target is not finite")
        if int(row["max_input_ts"]) > int(row["decision_ts"]):
            raise ValueError(f"{role} feature causality violation")
    if len(groups) != expected_market_count * 4:
        raise ValueError(f"{role} decision group count mismatch")
    if any(actions != set(REQUIRED_ACTIONS) for actions in groups.values()):
        raise ValueError(f"{role} five-action grid is incomplete")


def _validate_target_stripped_rows(rows: list[dict[str, Any]], *, expected_count: int) -> None:
    if len(rows) != expected_count:
        raise ValueError("target-stripped prediction count mismatch")
    found = _find_fields({"rows": rows}, set(TARGET_FIELDS))
    if found:
        raise ValueError("target-stripped predictions contain target fields")
    if any(int(row["max_input_ts"]) > int(row["decision_ts"]) for row in rows):
        raise ValueError("target-stripped prediction causality violation")


def _build_future_evaluation_protocol(
    *,
    run_id: str,
    profile: dict[str, Any],
    candidate_freeze_created_ts: int,
    calibration_gate_passed: bool,
    power_report: dict[str, Any],
) -> dict[str, Any]:
    future = dict(profile["future_evaluation"])
    protocol = {
        "schema_version": f"{SCHEMA_PREFIX}-future-evaluation-protocol-v1",
        "run_id": run_id,
        "candidate_name": CANDIDATE_NAME,
        "candidate_freeze_created_ts": candidate_freeze_created_ts,
        "eligible_collection": future["eligible_collection"],
        "issue_190_collection_eligible": False,
        "issue_192_collection_start_must_be_strictly_after_candidate_freeze": True,
        "single_use_holdout": True,
        "result_dependent_extension_allowed": False,
        "required_quality_valid_market_count": power_report[
            "recommended_quality_valid_market_count"
        ],
        "required_guard_accepted_unique_market_count": power_report[
            "recommended_required_accepted_unique_market_count"
        ],
        "required_checks": {
            "candidate_and_window_hash_binding_before_outcome_access": True,
            "strict_time_and_market_disjointness": True,
            "zero_feature_causality_violations": True,
            "zero_provenance_violations": True,
            "frozen_model_calibration_guard_cost_sizing_and_exposure": True,
            "minimum_quality_valid_market_count": future["minimum_quality_valid_market_count"],
            "minimum_guard_accepted_unique_market_count": future[
                "minimum_guard_accepted_unique_market_count"
            ],
            "minimum_supported_side_market_count": future["minimum_supported_side_market_count"],
            "pnl_hard_gate_aggregation": future["pnl_hard_gate_aggregation"],
            "action_and_family_pnl_diagnostic_only": future[
                "action_and_family_pnl_diagnostic_only"
            ],
            "accepted_bet_total_post_cost_pnl_positive": True,
            "all_market_policy_pnl_bootstrap_lcb_positive": True,
            "largest_winner_removed_pnl_positive": True,
        },
        "calibration_gate_passed": calibration_gate_passed,
        "future_evaluation_allowed": calibration_gate_passed,
        "outcomes_or_pnl_opened_during_fit": False,
        "paper_or_live_execution_allowed": False,
        **_blocked_safety_fields(),
    }
    protocol["protocol_id"] = canonical_json_sha256(protocol)
    return protocol


def _action_family(action: str) -> str:
    if action.endswith("HOLD_TO_SETTLEMENT"):
        return "HOLD_TO_SETTLEMENT"
    if action.endswith("SELL_BEFORE_CLOSE"):
        return "SELL_BEFORE_CLOSE"
    if action == "NO_TRADE":
        return "NO_TRADE"
    raise ValueError(f"unknown action: {action}")


def _pre_label_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# #203 pre-label access audit",
            "",
            f"- passed: `{report['pre_label_access_validation_passed']}`",
            "- #202 OOF/gate artifacts opened: `false`",
            "- targets opened before audit: `false`",
            "- future files opened: `false`",
            "- paper/live/promotion unlock: `false`",
            "",
        ]
    )


def _calibration_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# #203 split-conformal calibration",
        "",
        f"- calibration gate passed: `{report['calibration_gate_passed']}`",
        "- source split: `development_calibration_only`",
        "- calibration policy PnL computed: `false`",
        "- threshold/candidate search: `false/false`",
        "",
        "| Action | Source | Markets | Penalty | Coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for action, row in report["action_calibration"].items():
        lines.append(
            "| {action} | {source} | {markets} | {penalty:.8f} | {coverage:.6f} |".format(
                action=action,
                source=row["calibration_source"],
                markets=row["calibration_market_count"],
                penalty=float(row["calibration_penalty"]),
                coverage=float(row["empirical_market_simultaneous_coverage"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _future_protocol_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# #203 future evaluation protocol",
            "",
            f"- future evaluation allowed: `{report['future_evaluation_allowed']}`",
            f"- eligible collection: `{report['eligible_collection']}`",
            "- #190 eligible: `false`",
            f"- required quality-valid markets: `{report['required_quality_valid_market_count']}`",
            "- required accepted unique markets: "
            f"`{report['required_guard_accepted_unique_market_count']}`",
            "- result-dependent extension: `false`",
            "- paper/live/promotion unlock: `false`",
            "",
        ]
    )
