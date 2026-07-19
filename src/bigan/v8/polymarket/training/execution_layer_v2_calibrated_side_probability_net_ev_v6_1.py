"""Fit the #210 calibrated side-probability net-EV v6.1 candidate."""

from __future__ import annotations

import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from scipy.optimize import minimize

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_prediction_freeze import (
    _materialize_future_action_rows,
    _materialize_selected_window_features,
)
from bigan.v8.polymarket.training.execution_layer_v2_guard_compatible_direct_net_return_v4 import (
    TARGET_FIELDS as V4_TARGET_FIELDS,
)
from bigan.v8.polymarket.training.execution_layer_v2_guard_compatible_direct_net_return_v4 import (
    _row_sort_key,
    _strip_target_fields,
)
from bigan.v8.polymarket.training.execution_layer_v2_outcome_blind_acceptance_viability import (
    _outcome_blind_acceptance_replay,
)
from bigan.v8.polymarket.training.execution_layer_v2_persistent_outcome_blind_collector import (
    load_and_validate_persistent_outcome_blind_index,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    REQUIRED_ACTIONS,
    SIDES,
    _blocked_safety_fields,
    _descriptor,
    _find_nonempty_fields,
    _load_json,
    _load_jsonl,
    _require_git_sha,
    _require_sha256,
    _sha256_file,
    _verified_descriptor,
    _verify_pin,
    _write_json,
    _write_jsonl,
    _write_text,
    validate_policy_selected_conformal_v6_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6_fit import (
    CALIBRATION_ROLE,
    CHECK_ROLE,
    FIT_ROLE,
    apply_policy_selected_conformal_scores,
    attach_frozen_execution_compatibility,
    build_policy_selected_conformal_artifact,
    select_sequential_policy_rows,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6_fit import (
    TARGET_FIELDS as V6_TARGET_FIELDS,
)
from bigan.v8.polymarket.training.execution_layer_v2_v6_on_v5_target_free_diagnostic import (
    _normalize_v5_labeled_rows,
    _select_target_free_rows,
    _validate_preregistration_lineage,
)

PROFILE_SCHEMA_VERSION = "bigan-v8-calibrated-side-probability-net-ev-v6-1-profile-v1"
REPORT_SCHEMA_VERSION = "bigan-v8-calibrated-side-probability-net-ev-v6-1-report-v1"
MANIFEST_SCHEMA_VERSION = "bigan-v8-calibrated-side-probability-net-ev-v6-1-manifest-v1"
CANDIDATE_NAME = "calibrated_side_probability_net_ev_v6_1"
PROBABILITY_CALIBRATION_ROLE = "probability_calibration"
HTS_ACTIONS = frozenset({"BUY_UP_HOLD_TO_SETTLEMENT", "BUY_DOWN_HOLD_TO_SETTLEMENT"})
SBC_ACTIONS = frozenset({"BUY_UP_SELL_BEFORE_CLOSE", "BUY_DOWN_SELL_BEFORE_CLOSE"})
ALL_TARGET_FIELDS = frozenset(V4_TARGET_FIELDS | V6_TARGET_FIELDS)


@dataclass(frozen=True, slots=True)
class CalibratedSideProbabilityNetEVV61Config:
    """Pinned inputs for one #210 model redesign run."""

    run_id: str
    output_dir: Path | str
    profile_path: Path | str
    expected_profile_sha256: str
    v5_freeze_manifest_path: Path | str
    v6_profile_path: Path | str
    v6_preregistration_manifest_path: Path | str
    collector_index_path: Path | str
    feature_contract_path: Path | str
    implementation_commit: str
    freeze_created_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(self.expected_profile_sha256, name="expected_profile_sha256")
        _require_git_sha(self.implementation_commit)
        if self.freeze_created_ts <= 0:
            raise ValueError("freeze_created_ts must be positive")
        for name in (
            "output_dir",
            "profile_path",
            "v5_freeze_manifest_path",
            "v6_profile_path",
            "v6_preregistration_manifest_path",
            "collector_index_path",
            "feature_contract_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


def validate_calibrated_side_probability_net_ev_v6_1_profile(
    profile: dict[str, Any],
) -> None:
    """Reject any drift from the pre-outcome #210 contract."""

    roles = dict(profile.get("chronological_roles") or {})
    check = dict(profile.get("target_free_check") or {})
    model = dict(profile.get("side_probability_model") or {})
    probability = dict(profile.get("probability_calibration") or {})
    ev = dict(profile.get("probability_to_net_ev") or {})
    conformal = dict(profile.get("policy_selected_conformal_calibration") or {})
    collection = dict(profile.get("collection_policy") or {})
    expected_safety = _blocked_safety_fields() | {"paper_candidate_allowed": False}
    checks = {
        "schema": profile.get("schema_version") == PROFILE_SCHEMA_VERSION,
        "candidate": profile.get("candidate_name") == CANDIDATE_NAME,
        "issue": profile.get("issue_number") == 210,
        "frozen": profile.get("frozen") is True,
        "roles": roles
        == {
            "side_probability_fit_market_count": 90,
            "probability_calibration_market_count": 45,
            "net_return_conformal_market_count": 60,
            "target_free_check_market_count": 50,
            "assignment": (
                "v5_development_train_then_development_calibration_then_confirmatory_then_"
                "post_issue204_target_free"
            ),
        },
        "target_free_boundary": check.get("minimum_eligible_index_sequence") == 237
        and check.get("maximum_index_scan_count") == 100
        and check.get("target_free_check_market_count") is None,
        "target_free_support": check.get("minimum_positive_lcb_market_count") == 10
        and check.get("minimum_positive_lcb_market_count_per_side") == 5
        and check.get("minimum_full_guard_accepted_market_count") == 10
        and check.get("minimum_full_guard_accepted_market_count_per_side") == 5,
        "target_free_sealed": check.get("result_dependent_extension_allowed") is False
        and check.get("labels_outcomes_settlement_targets_or_pnl_opened") is False,
        "binary_model": model.get("objective") == "binary:logistic"
        and model.get("target") == "target_resolved_outcome_is_up"
        and model.get("one_training_row_per_decision_group") is True
        and model.get("hyperparameter_search_enabled") is False,
        "probability_calibration": probability.get("method") == "bounded_platt_logit_scaling"
        and probability.get("threshold_search_enabled") is False,
        "ev_contract": set(ev.get("enabled_actions") or []) == HTS_ACTIONS
        and set(ev.get("disabled_actions") or []) == SBC_ACTIONS
        and ev.get("market_implied_probability_used_as_conditioning_feature") is True
        and ev.get("market_implied_probability_used_as_direct_fair_value_ev") is False
        and ev.get("sell_before_close_exit_value_fabricated") is False
        and ev.get("cost_model_mutation_allowed") is False,
        "conformal_support": conformal.get("minimum_selected_calibration_market_count") == 50
        and conformal.get("minimum_selected_calibration_market_count_per_side") == 20
        and conformal.get("calibration_threshold_search_enabled") is False,
        "collection": collection.get("persistent_collector_paused_before_target_free_check") is True
        and collection.get(
            "resume_only_if_all_target_free_action_and_full_guard_support_gates_pass"
        )
        is True
        and collection.get("new_strictly_later_future_holdout_required_after_candidate_freeze")
        is True,
        "prohibited": profile.get("prohibited_inputs")
        == {
            "uses_204_outcomes_for_fitting": False,
            "uses_204_pnl_for_tuning": False,
            "uses_target_free_check_labels_for_tuning": False,
            "uses_current_oof_validation_or_confirmatory_pnl_for_tuning": False,
            "result_driven_model_threshold_penalty_or_gate_change_allowed": False,
        },
        "safety": profile.get("safety") == expected_safety,
    }
    lineage = dict(profile.get("source_lineage") or {})
    checks["lineage_hashes"] = bool(lineage) and all(
        _is_sha256(str(value)) for value in lineage.values()
    )
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("#210 profile invalid: " + ", ".join(blockers))


def run_calibrated_side_probability_net_ev_v6_1(
    config: CalibratedSideProbabilityNetEVV61Config,
) -> dict[str, Any]:
    """Fit on frozen v5 roles and evaluate one sealed target-free canary."""

    profile_path = config.profile_path.resolve()
    v5_manifest_path = config.v5_freeze_manifest_path.resolve()
    v6_profile_path = config.v6_profile_path.resolve()
    prereg_path = config.v6_preregistration_manifest_path.resolve()
    index_path = config.collector_index_path.resolve()
    feature_contract_path = config.feature_contract_path.resolve()
    _verify_pin(profile_path, config.expected_profile_sha256, "#210 profile")
    profile = _load_json(profile_path)
    validate_calibrated_side_probability_net_ev_v6_1_profile(profile)
    lineage = dict(profile["source_lineage"])
    _verify_pin(v5_manifest_path, lineage["v5_freeze_manifest_sha256"], "v5 manifest")
    _verify_pin(v6_profile_path, lineage["v6_preregistration_profile_sha256"], "v6 profile")
    _verify_pin(
        prereg_path,
        lineage["v6_preregistration_manifest_sha256"],
        "v6 preregistration",
    )
    _verify_pin(feature_contract_path, lineage["feature_contract_sha256"], "feature contract")
    v6_profile = _load_json(v6_profile_path)
    validate_policy_selected_conformal_v6_profile(v6_profile)
    prereg = _load_json(prereg_path)
    _validate_preregistration_lineage(
        prereg,
        expected_v6_profile_sha256=lineage["v6_preregistration_profile_sha256"],
    )
    feature_contract = _load_json(feature_contract_path)
    action_feature_columns = tuple(str(value) for value in feature_contract["feature_columns"])

    v5_manifest = _load_json(v5_manifest_path)
    fit_descriptor = _verified_descriptor(
        v5_manifest.get("development_train_action_rows"), "v5 fit action rows"
    )
    conformal_descriptor = _verified_descriptor(
        v5_manifest.get("development_calibration_action_rows"),
        "v5 conformal action rows",
    )
    role_descriptor = _verified_descriptor(
        v5_manifest.get("role_assignment_manifest"), "v5 role manifest"
    )
    role_manifest = _load_json(Path(role_descriptor["path"]))
    role_rows_descriptor = _verified_descriptor(role_manifest.get("selected_rows"), "v5 role rows")
    _verify_source_lineage(
        lineage,
        fit_descriptor=fit_descriptor,
        conformal_descriptor=conformal_descriptor,
        role_descriptor=role_descriptor,
        role_rows_descriptor=role_rows_descriptor,
    )

    observed_index_hash = _sha256_file(index_path)
    index_rows = load_and_validate_persistent_outcome_blind_index(index_path)
    if _sha256_file(index_path) != observed_index_hash:
        raise ValueError("collector index changed while freezing #210 target-free prefix")
    selected_rows, scanned_count, rejection_distribution = _select_target_free_rows(
        index_rows,
        prereg=prereg,
        v6_profile=v6_profile,
        diagnostic_profile=profile,
    )
    if len(selected_rows) != 50:
        raise ValueError("#210 target-free check does not have exactly 50 valid markets")
    if config.freeze_created_ts <= max(int(row["market_end_ts"]) for row in selected_rows):
        raise ValueError("#210 target-free markets are not all closed before freeze")

    run_dir = config.output_dir.resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    selected_path = run_dir / "v6_1_target_free_selected_rows.jsonl"
    _write_jsonl(selected_path, selected_rows)
    feature_rows, raw_descriptors = _materialize_selected_window_features(selected_rows)
    feature_rows = [
        {
            **row,
            "role": CHECK_ROLE,
            "development_role": CHECK_ROLE,
            "target_used_as_decision_input": False,
            "outcome_fields_used_as_decision_input": False,
        }
        for row in feature_rows
    ]
    action_rows = _materialize_future_action_rows(
        feature_rows,
        selected_rows=selected_rows,
        feature_columns=action_feature_columns,
    )
    action_rows = [
        {
            **row,
            "role": CHECK_ROLE,
            "development_role": CHECK_ROLE,
            "target_used_as_decision_input": False,
            "outcome_fields_used_as_decision_input": False,
        }
        for row in action_rows
    ]
    if _find_nonempty_fields(action_rows, ALL_TARGET_FIELDS):
        raise ValueError("#210 target-free action rows contain forbidden targets")
    feature_rows_path = run_dir / "v6_1_target_free_feature_rows.jsonl"
    action_rows_path = run_dir / "v6_1_target_free_five_action_rows.jsonl"
    _write_jsonl(feature_rows_path, feature_rows)
    _write_jsonl(action_rows_path, action_rows)

    pre_label = {
        "schema_version": "bigan-v8-v6-1-pre-label-lineage-audit-v1",
        "run_id": config.run_id,
        "implementation_commit": config.implementation_commit,
        "profile": _descriptor(profile_path),
        "v5_freeze_manifest": _descriptor(v5_manifest_path),
        "v5_fit_action_rows": fit_descriptor,
        "v5_conformal_action_rows": conformal_descriptor,
        "v5_role_assignment_manifest": role_descriptor,
        "v5_role_assignment_rows": role_rows_descriptor,
        "v6_profile": _descriptor(v6_profile_path),
        "v6_preregistration_manifest": _descriptor(prereg_path),
        "feature_contract": _descriptor(feature_contract_path),
        "collector_index_observed_sha256": observed_index_hash,
        "selected_target_free_rows": _descriptor(selected_path),
        "selected_sequence_start": int(selected_rows[0]["sequence"]),
        "selected_sequence_end": int(selected_rows[-1]["sequence"]),
        "selected_market_count": len(selected_rows),
        "scanned_post_boundary_row_count": scanned_count,
        "rejected_reason_distribution": rejection_distribution,
        "target_free_feature_rows": _descriptor(feature_rows_path),
        "target_free_five_action_rows": _descriptor(action_rows_path),
        "target_free_feature_causality_violation_count": sum(
            int(row["max_input_ts"]) > int(row["decision_ts"]) for row in feature_rows
        ),
        "v5_labels_opened_before_audit": False,
        "issue204_or_target_free_outcome_settlement_target_or_pnl_files_opened": False,
        "pre_label_access_validation_passed": True,
        **_blocked_safety_fields(),
        "paper_candidate_allowed": False,
    }
    pre_label["audit_id"] = canonical_json_sha256(pre_label)
    pre_label_path = run_dir / "v6_1_pre_label_lineage_audit.json"
    _write_json(pre_label_path, pre_label)

    all_fit_source = _load_jsonl(Path(fit_descriptor["path"]))
    train_rows = _normalize_v5_labeled_rows(
        [row for row in all_fit_source if row.get("role") == "development_train"],
        role=FIT_ROLE,
        expected_source_roles={"development_train"},
        feature_columns=action_feature_columns,
    )
    probability_calibration_rows = _normalize_v5_labeled_rows(
        [row for row in all_fit_source if row.get("role") == "development_calibration"],
        role=PROBABILITY_CALIBRATION_ROLE,
        expected_source_roles={"development_calibration"},
        feature_columns=action_feature_columns,
    )
    conformal_rows = _normalize_v5_labeled_rows(
        _load_jsonl(Path(conformal_descriptor["path"])),
        role=CALIBRATION_ROLE,
        expected_source_roles={"confirmatory_validation"},
        feature_columns=action_feature_columns,
    )
    _validate_role_market_counts(
        train_rows,
        probability_calibration_rows,
        conformal_rows,
        profile=profile,
    )
    if max(int(row["decision_ts"]) for row in train_rows) >= min(
        int(row["decision_ts"]) for row in probability_calibration_rows
    ):
        raise ValueError("side-probability fit does not strictly precede probability calibration")
    if max(int(row["decision_ts"]) for row in probability_calibration_rows) >= min(
        int(row["decision_ts"]) for row in conformal_rows
    ):
        raise ValueError("probability calibration does not strictly precede conformal calibration")
    if max(int(row["decision_ts"]) for row in conformal_rows) >= min(
        int(row["decision_ts"]) for row in action_rows
    ):
        raise ValueError("conformal calibration does not strictly precede target-free canary")

    train_probability_rows = build_side_probability_rows(
        train_rows, profile=profile, include_target=True
    )
    probability_calibration_probability_rows = build_side_probability_rows(
        probability_calibration_rows, profile=profile, include_target=True
    )
    conformal_probability_rows = build_side_probability_rows(
        conformal_rows, profile=profile, include_target=True
    )
    target_free_probability_rows = build_side_probability_rows(
        action_rows, profile=profile, include_target=False
    )
    model_feature_columns = tuple(train_probability_rows[0]["model_features"])
    booster = _train_side_probability_model(
        train_probability_rows,
        feature_columns=model_feature_columns,
        model_config=dict(profile["side_probability_model"]),
    )
    model_path = run_dir / "v6_1_side_probability_model.xgb.json"
    booster.save_model(model_path)

    train_raw = _predict_side_probability(
        booster, train_probability_rows, feature_columns=model_feature_columns
    )
    probability_calibration_raw = _predict_side_probability(
        booster,
        probability_calibration_probability_rows,
        feature_columns=model_feature_columns,
    )
    calibration_artifact = fit_probability_calibration(
        probability_calibration_probability_rows,
        probability_calibration_raw,
        profile=profile,
        model_sha256=_sha256_file(model_path),
    )
    probability_artifact_path = run_dir / "v6_1_probability_calibration_artifact.json"
    _write_json(probability_artifact_path, calibration_artifact)
    train_calibrated = apply_probability_calibration(train_raw, calibration_artifact)
    probability_calibration_calibrated = apply_probability_calibration(
        probability_calibration_raw, calibration_artifact
    )
    conformal_raw = _predict_side_probability(
        booster, conformal_probability_rows, feature_columns=model_feature_columns
    )
    conformal_calibrated = apply_probability_calibration(conformal_raw, calibration_artifact)
    target_free_raw = _predict_side_probability(
        booster, target_free_probability_rows, feature_columns=model_feature_columns
    )
    target_free_calibrated = apply_probability_calibration(target_free_raw, calibration_artifact)

    probability_report = _probability_calibration_report(
        train_probability_rows=train_probability_rows,
        train_raw=train_raw,
        train_calibrated=train_calibrated,
        calibration_rows=probability_calibration_probability_rows,
        calibration_raw=probability_calibration_raw,
        calibration_calibrated=probability_calibration_calibrated,
        calibration_artifact=calibration_artifact,
    )
    probability_report_path = run_dir / "v6_1_probability_calibration_report.json"
    _write_json(probability_report_path, probability_report)
    _write_text(
        probability_report_path.with_suffix(".md"), _probability_markdown(probability_report)
    )

    conformal_predictions = score_action_rows_from_probability(
        conformal_rows,
        probability_rows=conformal_probability_rows,
        calibrated_p_up=conformal_calibrated,
        profile=profile,
    )
    target_free_predictions = score_action_rows_from_probability(
        action_rows,
        probability_rows=target_free_probability_rows,
        calibrated_p_up=target_free_calibrated,
        profile=profile,
    )
    conformal_artifact = build_policy_selected_conformal_artifact(
        conformal_predictions,
        target_rows=conformal_rows,
        profile=profile,
        feature_contract_sha256=lineage["feature_contract_sha256"],
    )
    conformal_artifact = {
        **conformal_artifact,
        "candidate_name": CANDIDATE_NAME,
        "raw_score_source": "calibrated_side_probability_minus_execution_and_frozen_costs",
        "probability_calibration_artifact": _descriptor(probability_artifact_path),
        "uses_probability_calibration_labels_for_threshold_selection": False,
        "uses_conformal_labels_for_policy_pnl_or_threshold_selection": False,
        "uses_target_free_check_labels_for_tuning": False,
        "diagnostic_only": True,
    }
    conformal_artifact["v6_1_conformal_artifact_id"] = canonical_json_sha256(conformal_artifact)
    conformal_artifact_path = run_dir / "v6_1_policy_selected_conformal_artifact.json"
    _write_json(conformal_artifact_path, conformal_artifact)

    scored_check = apply_policy_selected_conformal_scores(
        target_free_predictions,
        calibration_artifact=conformal_artifact,
        profile=profile,
    )
    scored_check = [_attach_rank_compatibility(row) for row in scored_check]
    static_selected = select_sequential_policy_rows(
        scored_check,
        score_field="conformal_net_return_lower_bound",
        require_positive=True,
    )
    calibration_gate = _calibration_and_target_free_gate(
        conformal_artifact=conformal_artifact,
        static_selected=static_selected,
        profile=profile,
    )
    replay = _outcome_blind_acceptance_replay(
        scored_check,
        entry_threshold=0.0,
        runner_up_advantage_threshold=0.0,
    )
    accepted = [row for row in replay if row["execution_guard_order_allowed"]]
    viability = _viability_summary(
        calibration_gate=calibration_gate,
        static_selected=static_selected,
        accepted=accepted,
        profile=profile,
    )

    train_probability_path = run_dir / "v6_1_side_probability_fit_rows.jsonl"
    probability_calibration_path = run_dir / "v6_1_probability_calibration_rows.jsonl"
    conformal_probability_path = run_dir / "v6_1_conformal_probability_rows.jsonl"
    target_free_probability_path = run_dir / "v6_1_target_free_probability_rows.jsonl"
    conformal_predictions_path = run_dir / "v6_1_conformal_target_stripped_predictions.jsonl"
    target_free_predictions_path = run_dir / "v6_1_target_free_raw_ev_predictions.jsonl"
    scored_check_path = run_dir / "v6_1_target_free_conformal_scored_rows.jsonl"
    replay_path = run_dir / "v6_1_target_free_full_guard_replay.jsonl"
    accepted_path = run_dir / "v6_1_target_free_guard_accepted_bets.jsonl"
    _write_jsonl(train_probability_path, train_probability_rows)
    _write_jsonl(probability_calibration_path, probability_calibration_probability_rows)
    _write_jsonl(conformal_probability_path, conformal_probability_rows)
    _write_jsonl(target_free_probability_path, target_free_probability_rows)
    _write_jsonl(conformal_predictions_path, conformal_predictions)
    _write_jsonl(target_free_predictions_path, target_free_predictions)
    _write_jsonl(scored_check_path, scored_check)
    _write_jsonl(replay_path, replay)
    _write_jsonl(accepted_path, accepted)

    report = _build_report(
        config=config,
        profile=profile,
        selected_rows=selected_rows,
        feature_rows=feature_rows,
        train_probability_rows=train_probability_rows,
        probability_calibration_probability_rows=probability_calibration_probability_rows,
        conformal_probability_rows=conformal_probability_rows,
        target_free_probability_rows=target_free_probability_rows,
        probability_report=probability_report,
        conformal_artifact=conformal_artifact,
        scored_check=scored_check,
        static_selected=static_selected,
        replay=replay,
        accepted=accepted,
        calibration_gate=calibration_gate,
        viability=viability,
    )
    report_path = run_dir / "v6_1_target_free_action_viability_report.json"
    _write_json(report_path, report)
    _write_text(report_path.with_suffix(".md"), _report_markdown(report))
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        "profile": _descriptor(profile_path),
        "pre_label_lineage_audit": _descriptor(pre_label_path),
        "side_probability_model": _descriptor(model_path),
        "probability_calibration_artifact": _descriptor(probability_artifact_path),
        "probability_calibration_report": _descriptor(probability_report_path),
        "policy_selected_conformal_artifact": _descriptor(conformal_artifact_path),
        "target_free_selected_rows": _descriptor(selected_path),
        "target_free_feature_rows": _descriptor(feature_rows_path),
        "target_free_five_action_rows": _descriptor(action_rows_path),
        "target_free_raw_ev_predictions": _descriptor(target_free_predictions_path),
        "target_free_conformal_scored_rows": _descriptor(scored_check_path),
        "target_free_full_guard_replay": _descriptor(replay_path),
        "target_free_guard_accepted_bets": _descriptor(accepted_path),
        "viability_report": _descriptor(report_path),
        "opened_raw_feature_artifacts": raw_descriptors,
        "target_free_action_viability_passed": viability["passed"],
        "target_free_action_viability_blocking_reason_codes": viability["reason_codes"],
        "collector_resume_allowed": viability["passed"],
        "new_strictly_later_future_holdout_required": True,
        "promotion_evidence": False,
        "target_free_labels_outcomes_settlement_targets_or_pnl_opened": False,
        **_blocked_safety_fields(),
        "paper_candidate_allowed": False,
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v6_1_candidate_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "report": report,
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def build_side_probability_rows(
    action_rows: list[dict[str, Any]],
    *,
    profile: dict[str, Any],
    include_target: bool,
) -> list[dict[str, Any]]:
    """Collapse each complete five-action group into one decision-time feature row."""

    feature_contract = dict(profile["side_probability_features"])
    common_names = tuple(str(value) for value in feature_contract["common_feature_names"])
    side_names = tuple(str(value) for value in feature_contract["side_feature_names"])
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in action_rows:
        grouped[(str(row["market_id"]), int(row["decision_ts"]))].append(row)
    output = []
    for (market_id, decision_ts), rows in sorted(
        grouped.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        by_action = {str(row["action"]): row for row in rows}
        if set(by_action) != set(REQUIRED_ACTIONS):
            raise ValueError("side-probability decision group is not a complete five-action grid")
        up = by_action["BUY_UP_HOLD_TO_SETTLEMENT"]
        down = by_action["BUY_DOWN_HOLD_TO_SETTLEMENT"]
        if int(up["max_input_ts"]) > decision_ts or int(down["max_input_ts"]) > decision_ts:
            raise ValueError("side-probability feature causality violation")
        up_features = dict(up["decision_time_features"])
        down_features = dict(down["decision_time_features"])
        model_features: dict[str, float] = {}
        for name in common_names:
            up_value = _finite_feature(up_features, name)
            down_value = _finite_feature(down_features, name)
            if not math.isclose(up_value, down_value, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"common side-probability feature differs by side: {name}")
            model_features[name] = up_value
        for prefix, source in (("up", up_features), ("down", down_features)):
            for name in side_names:
                model_features[f"{prefix}_{name}"] = _finite_feature(source, name)
        row = {
            "market_id": market_id,
            "decision_ts": decision_ts,
            "max_input_ts": max(int(up["max_input_ts"]), int(down["max_input_ts"])),
            "model_features": model_features,
            "model_feature_columns": list(model_features),
            "decision_time_features_only": True,
            "market_implied_probability_used_as_conditioning_feature": True,
            "market_implied_probability_used_as_direct_fair_value_ev": False,
            "target_used_as_decision_input": False,
            "outcome_fields_used_as_decision_input": False,
        }
        if include_target:
            outcomes = {str(value.get("target_resolved_outcome") or "") for value in rows}
            if len(outcomes) != 1 or next(iter(outcomes)) not in SIDES:
                raise ValueError("side-probability target outcome is missing or inconsistent")
            outcome = next(iter(outcomes))
            row["target_resolved_outcome"] = outcome
            row["target_resolved_outcome_is_up"] = 1 if outcome == "UP" else 0
        elif _find_nonempty_fields(rows, ALL_TARGET_FIELDS):
            raise ValueError("target-free side-probability source rows contain targets")
        row["side_probability_row_sha256"] = canonical_json_sha256(row)
        output.append(row)
    return output


def fit_probability_calibration(
    rows: list[dict[str, Any]],
    raw_probabilities: list[float],
    *,
    profile: dict[str, Any],
    model_sha256: str,
) -> dict[str, Any]:
    """Fit one bounded Platt transform on the frozen probability-calibration role."""

    if len(rows) != len(raw_probabilities) or not rows:
        raise ValueError("probability calibration coverage mismatch")
    calibration = dict(profile["probability_calibration"])
    clip = float(calibration["probability_clip"])
    logits = np.asarray([_logit(value, clip=clip) for value in raw_probabilities], dtype=float)
    labels = np.asarray([int(row["target_resolved_outcome_is_up"]) for row in rows], dtype=float)
    l2 = float(calibration["l2_strength"])

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        slope, intercept = float(theta[0]), float(theta[1])
        values = slope * logits + intercept
        probabilities = _sigmoid_array(values)
        loss = float(np.mean(np.logaddexp(0.0, values) - labels * values))
        loss += l2 * ((slope - 1.0) ** 2 + intercept**2)
        error = probabilities - labels
        gradient = np.asarray(
            [
                float(np.mean(error * logits)) + 2.0 * l2 * (slope - 1.0),
                float(np.mean(error)) + 2.0 * l2 * intercept,
            ],
            dtype=float,
        )
        return loss, gradient

    result = minimize(
        lambda theta: objective(theta)[0],
        x0=np.asarray([1.0, 0.0]),
        jac=lambda theta: objective(theta)[1],
        bounds=[
            tuple(float(value) for value in calibration["slope_bounds"]),
            tuple(float(value) for value in calibration["intercept_bounds"]),
        ],
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-10, "maxls": 50},
    )
    if not result.success or not all(math.isfinite(float(value)) for value in result.x):
        raise ValueError("bounded Platt probability calibration failed")
    artifact = {
        "schema_version": "bigan-v8-v6-1-probability-calibration-artifact-v1",
        "candidate_name": CANDIDATE_NAME,
        "frozen": True,
        "decision_time_safe": True,
        "method": calibration["method"],
        "model_sha256": model_sha256,
        "calibration_market_count": len({str(row["market_id"]) for row in rows}),
        "calibration_decision_count": len(rows),
        "slope": float(result.x[0]),
        "intercept": float(result.x[1]),
        "probability_clip": clip,
        "optimizer_success": True,
        "optimizer_iteration_count": int(result.nit),
        "uses_probability_calibration_labels_for_threshold_selection": False,
        "uses_probability_calibration_policy_pnl": False,
        "market_implied_probability_used_as_conditioning_feature": True,
        "market_implied_probability_used_as_direct_fair_value_ev": False,
        "uses_204_outcomes_for_fitting": False,
        "uses_target_free_check_labels_for_tuning": False,
        **_blocked_safety_fields(),
        "paper_candidate_allowed": False,
    }
    artifact["artifact_id"] = canonical_json_sha256(artifact)
    validate_probability_calibration_artifact(artifact)
    return artifact


def validate_probability_calibration_artifact(artifact: dict[str, Any]) -> None:
    """Fail closed on incomplete or semantically unsafe probability calibration."""

    blockers = []
    if artifact.get("schema_version") != "bigan-v8-v6-1-probability-calibration-artifact-v1":
        blockers.append("schema")
    if artifact.get("frozen") is not True or artifact.get("decision_time_safe") is not True:
        blockers.append("freeze_or_causality")
    if not _is_sha256(str(artifact.get("model_sha256") or "")):
        blockers.append("model_hash")
    for name in ("slope", "intercept", "probability_clip"):
        value = artifact.get(name)
        if not isinstance(value, int | float) or not math.isfinite(float(value)):
            blockers.append(name)
    if artifact.get("market_implied_probability_used_as_conditioning_feature") is not True:
        blockers.append("market_probability_conditioning")
    if artifact.get("market_implied_probability_used_as_direct_fair_value_ev") is not False:
        blockers.append("market_probability_direct_ev")
    if artifact.get("uses_target_free_check_labels_for_tuning") is not False:
        blockers.append("target_free_tuning")
    if blockers:
        raise ValueError("probability calibration artifact invalid: " + ", ".join(blockers))


def apply_probability_calibration(
    raw_probabilities: list[float], artifact: dict[str, Any]
) -> list[float]:
    validate_probability_calibration_artifact(artifact)
    slope = float(artifact["slope"])
    intercept = float(artifact["intercept"])
    clip = float(artifact["probability_clip"])
    return [
        float(_sigmoid(slope * _logit(value, clip=clip) + intercept)) for value in raw_probabilities
    ]


def score_action_rows_from_probability(
    action_rows: list[dict[str, Any]],
    *,
    probability_rows: list[dict[str, Any]],
    calibrated_p_up: list[float],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Translate calibrated fair side probability into cost-aware HTS action EV."""

    if len(probability_rows) != len(calibrated_p_up):
        raise ValueError("side-probability prediction coverage mismatch")
    by_group = {
        (str(row["market_id"]), int(row["decision_ts"])): float(value)
        for row, value in zip(probability_rows, calibrated_p_up, strict=True)
    }
    ev = dict(profile["probability_to_net_ev"])
    output = []
    for source in sorted(action_rows, key=_row_sort_key):
        row = _strip_target_fields(source)
        action = str(row["action"])
        fair_p_up = by_group[(str(row["market_id"]), int(row["decision_ts"]))]
        side = str(row["side"])
        fair_side = fair_p_up if side == "UP" else 1.0 - fair_p_up if side == "DOWN" else 0.0
        if action == "NO_TRADE":
            execution_price = fees = slippage = liquidity_impact = 0.0
            raw_ev = 0.0
            available = True
            source_name = "frozen_no_trade_zero_anchor"
            unavailable_reasons: list[str] = []
        else:
            features = dict(row["decision_time_features"])
            microstructure = dict(row.get("microstructure_snapshot") or {})
            execution_price = _finite_feature(features, "execution_price")
            execution_bid = _finite_number(microstructure.get("entry_bid"), "entry_bid")
            if not 0.0 <= execution_bid <= execution_price <= 1.0:
                raise ValueError("invalid decision-time executable bid/ask")
            fees = float(ev["fees_per_contract"])
            slippage = max(0.0001, (execution_price - execution_bid) / 2.0)
            depth = _finite_feature(features, "selected_side_liquidity_depth")
            liquidity_impact = float(
                ev[
                    "liquidity_impact_when_depth_positive"
                    if depth > 0.0
                    else "liquidity_impact_when_depth_non_positive"
                ]
            )
            raw_ev = fair_side - execution_price - fees - slippage - liquidity_impact
            available = action in HTS_ACTIONS
            source_name = (
                "calibrated_side_probability_minus_execution_and_frozen_costs"
                if available
                else "sell_before_close_exit_value_unavailable_fail_closed"
            )
            unavailable_reasons = [] if available else ["sell_before_close_ev_model_unavailable"]
            if source.get("target_cost_components") is not None:
                expected_costs = dict(source["target_cost_components"])
                observed = (fees, slippage, liquidity_impact)
                expected = (
                    float(expected_costs["fees"]),
                    float(expected_costs["slippage"]),
                    float(expected_costs["liquidity_impact"]),
                )
                if any(
                    not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
                    for left, right in zip(observed, expected, strict=True)
                ):
                    raise ValueError(
                        "decision-time cost reconstruction differs from v5 label contract"
                    )
        selection_ev = raw_ev if available else -1_000_000.0
        updated = {
            **row,
            "raw_model_prediction": selection_ev,
            "raw_direct_predicted_net_return": selection_ev,
            "calibrated_action_expected_net_return": selection_ev,
            "raw_prediction_source": source_name,
            "calibrated_model_fair_value_up": fair_p_up,
            "calibrated_model_fair_value_selected_side": fair_side,
            "execution_price": execution_price,
            "decision_time_cost_components": {
                "fees": fees,
                "slippage": slippage,
                "liquidity_impact": liquidity_impact,
            },
            "probability_to_net_ev_action_available": available,
            "probability_to_net_ev_ineligible_reason_codes": unavailable_reasons,
            "market_implied_probability_used_as_conditioning_feature": True,
            "market_implied_probability_used_as_direct_fair_value_ev": False,
            "target_used_as_decision_input": False,
            "outcome_fields_used_as_decision_input": False,
            "target_or_outcome_fields_used": False,
        }
        updated["v6_1_raw_prediction_row_sha256"] = canonical_json_sha256(updated)
        output.append(updated)
    compatible = attach_frozen_execution_compatibility(output)
    result = []
    for row in compatible:
        if row["action"] in SBC_ACTIONS:
            row = {
                **row,
                "guard_compatible_before_ranking": False,
                "guard_compatibility_mask_applied_before_argmax": True,
            }
        row["v6_1_compatible_prediction_row_sha256"] = canonical_json_sha256(row)
        result.append(row)
    if _find_nonempty_fields(result, ALL_TARGET_FIELDS):
        raise ValueError("v6.1 prediction rows contain target fields")
    return result


def _train_side_probability_model(
    rows: list[dict[str, Any]],
    *,
    feature_columns: tuple[str, ...],
    model_config: dict[str, Any],
) -> xgb.Booster:
    matrix = xgb.DMatrix(
        np.asarray(
            [[float(row["model_features"][name]) for name in feature_columns] for row in rows],
            dtype=np.float32,
        ),
        label=np.asarray([float(row["target_resolved_outcome_is_up"]) for row in rows]),
        feature_names=list(feature_columns),
    )
    parameters = {
        key: value
        for key, value in model_config.items()
        if key
        not in {
            "target",
            "grouping_unit",
            "one_training_row_per_decision_group",
            "num_boost_round",
            "hyperparameter_search_enabled",
        }
    }
    return xgb.train(
        parameters,
        matrix,
        num_boost_round=int(model_config["num_boost_round"]),
    )


def _predict_side_probability(
    booster: xgb.Booster,
    rows: list[dict[str, Any]],
    *,
    feature_columns: tuple[str, ...],
) -> list[float]:
    matrix = xgb.DMatrix(
        np.asarray(
            [[float(row["model_features"][name]) for name in feature_columns] for row in rows],
            dtype=np.float32,
        ),
        feature_names=list(feature_columns),
    )
    probabilities = [float(value) for value in booster.predict(matrix)]
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError("side-probability model produced invalid probability")
    return probabilities


def _calibration_and_target_free_gate(
    *,
    conformal_artifact: dict[str, Any],
    static_selected: list[dict[str, Any]],
    profile: dict[str, Any],
) -> dict[str, Any]:
    calibration = dict(profile["policy_selected_conformal_calibration"])
    selected_side = Counter(str(row["side"]) for row in static_selected)
    calibration_side = {
        side: int(conformal_artifact["selected_side_distribution"].get(side, 0)) for side in SIDES
    }
    checks = {
        "selected_calibration_market_support": int(
            conformal_artifact["selected_calibration_market_count"]
        )
        >= int(calibration["minimum_selected_calibration_market_count"]),
        "selected_calibration_side_support": all(
            calibration_side[side]
            >= int(calibration["minimum_selected_calibration_market_count_per_side"])
            for side in SIDES
        ),
        "finite_bounded_conformal_penalties": conformal_artifact["finite_bounded_penalties"]
        is True,
        "nominal_one_sided_coverage": all(
            float(conformal_artifact["sides"][side]["empirical_one_sided_coverage"]) >= 0.9
            for side in SIDES
        ),
        "target_free_positive_lcb_total_support": len(static_selected)
        >= int(profile["target_free_check"]["minimum_positive_lcb_market_count"]),
        "target_free_positive_lcb_side_support": all(
            selected_side[side]
            >= int(profile["target_free_check"]["minimum_positive_lcb_market_count_per_side"])
            for side in SIDES
        ),
        "no_calibration_policy_pnl": conformal_artifact["policy_pnl_computed_on_calibration"]
        is False,
        "no_threshold_search": conformal_artifact["calibration_threshold_search_enabled"] is False,
    }
    reason_map = {
        "selected_calibration_market_support": "selected_calibration_market_support_failed",
        "selected_calibration_side_support": "selected_calibration_side_support_failed",
        "finite_bounded_conformal_penalties": "conformal_penalty_invalid_or_unbounded",
        "nominal_one_sided_coverage": "nominal_one_sided_coverage_failed",
        "target_free_positive_lcb_total_support": "target_free_positive_lcb_total_support_failed",
        "target_free_positive_lcb_side_support": "target_free_positive_lcb_side_support_failed",
        "no_calibration_policy_pnl": "calibration_policy_pnl_was_computed",
        "no_threshold_search": "calibration_threshold_search_enabled",
    }
    reasons = [reason_map[name] for name, passed in checks.items() if not passed]
    return {
        "passed": not reasons,
        "checks": checks,
        "reason_codes": reasons,
        "selected_calibration_side_distribution": calibration_side,
        "target_free_positive_lcb_side_distribution": {side: selected_side[side] for side in SIDES},
    }


def _viability_summary(
    *,
    calibration_gate: dict[str, Any],
    static_selected: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    profile: dict[str, Any],
) -> dict[str, Any]:
    check = dict(profile["target_free_check"])
    accepted_markets = {str(row["market_id"]) for row in accepted}
    accepted_side_markets = {
        side: {
            str(row["market_id"]) for row in accepted if str(row.get("selected_side") or "") == side
        }
        for side in SIDES
    }
    checks = {
        "calibration_and_positive_lcb_support": calibration_gate["passed"] is True,
        "full_guard_total_support": len(accepted_markets)
        >= int(check["minimum_full_guard_accepted_market_count"]),
        "full_guard_side_support": all(
            len(accepted_side_markets[side])
            >= int(check["minimum_full_guard_accepted_market_count_per_side"])
            for side in SIDES
        ),
        "some_positive_lcb_action": bool(static_selected),
        "some_full_guard_accepted_bet": bool(accepted),
    }
    reason_map = {
        "calibration_and_positive_lcb_support": "calibration_or_positive_lcb_support_failed",
        "full_guard_total_support": "target_free_full_guard_total_support_failed",
        "full_guard_side_support": "target_free_full_guard_side_support_failed",
        "some_positive_lcb_action": "target_free_all_actions_no_trade_after_lcb",
        "some_full_guard_accepted_bet": "target_free_all_actions_no_trade_or_guard_blocked",
    }
    reasons = list(calibration_gate["reason_codes"])
    reasons.extend(reason_map[name] for name, passed in checks.items() if not passed)
    return {
        "passed": not reasons,
        "checks": checks,
        "reason_codes": sorted(set(reasons)),
        "full_guard_accepted_unique_market_count": len(accepted_markets),
        "full_guard_accepted_side_market_count": {
            side: len(accepted_side_markets[side]) for side in SIDES
        },
    }


def _probability_calibration_report(
    *,
    train_probability_rows: list[dict[str, Any]],
    train_raw: list[float],
    train_calibrated: list[float],
    calibration_rows: list[dict[str, Any]],
    calibration_raw: list[float],
    calibration_calibrated: list[float],
    calibration_artifact: dict[str, Any],
) -> dict[str, Any]:
    report = {
        "schema_version": "bigan-v8-v6-1-probability-calibration-report-v1",
        "candidate_name": CANDIDATE_NAME,
        "fit": _probability_metrics(train_probability_rows, train_raw, train_calibrated),
        "probability_calibration": _probability_metrics(
            calibration_rows, calibration_raw, calibration_calibrated
        ),
        "calibration_artifact_id": calibration_artifact["artifact_id"],
        "uses_probability_calibration_labels_for_threshold_selection": False,
        "uses_probability_calibration_policy_pnl": False,
        "market_implied_probability_used_as_direct_fair_value_ev": False,
        **_blocked_safety_fields(),
        "paper_candidate_allowed": False,
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def _probability_metrics(
    rows: list[dict[str, Any]], raw: list[float], calibrated: list[float]
) -> dict[str, Any]:
    labels = [int(row["target_resolved_outcome_is_up"]) for row in rows]
    return {
        "market_count": len({str(row["market_id"]) for row in rows}),
        "decision_count": len(rows),
        "resolved_outcome_distribution": dict(
            sorted(Counter("UP" if value == 1 else "DOWN" for value in labels).items())
        ),
        "raw_brier_score": _brier(labels, raw),
        "calibrated_brier_score": _brier(labels, calibrated),
        "raw_log_loss": _log_loss(labels, raw),
        "calibrated_log_loss": _log_loss(labels, calibrated),
    }


def _build_report(
    *,
    config: CalibratedSideProbabilityNetEVV61Config,
    profile: dict[str, Any],
    selected_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    train_probability_rows: list[dict[str, Any]],
    probability_calibration_probability_rows: list[dict[str, Any]],
    conformal_probability_rows: list[dict[str, Any]],
    target_free_probability_rows: list[dict[str, Any]],
    probability_report: dict[str, Any],
    conformal_artifact: dict[str, Any],
    scored_check: list[dict[str, Any]],
    static_selected: list[dict[str, Any]],
    replay: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    calibration_gate: dict[str, Any],
    viability: dict[str, Any],
) -> dict[str, Any]:
    raw_positive = [
        row
        for row in scored_check
        if row["action"] in HTS_ACTIONS
        and row["guard_compatible_before_ranking"]
        and float(row["raw_direct_predicted_net_return"]) > 0.0
    ]
    static_sides = Counter(str(row["side"]) for row in static_selected)
    accepted_actions = Counter(str(row["executed_action"]) for row in accepted)
    blockers = Counter(
        str(reason) for row in replay for reason in row["execution_blocking_reason_codes"]
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_id": None,
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        "fit_market_count": len({str(row["market_id"]) for row in train_probability_rows}),
        "fit_decision_count": len(train_probability_rows),
        "probability_calibration_market_count": len(
            {str(row["market_id"]) for row in probability_calibration_probability_rows}
        ),
        "probability_calibration_decision_count": len(probability_calibration_probability_rows),
        "conformal_market_count": len(
            {str(row["market_id"]) for row in conformal_probability_rows}
        ),
        "conformal_decision_count": len(conformal_probability_rows),
        "target_free_check_market_count": len(selected_rows),
        "target_free_feature_row_count": len(feature_rows),
        "target_free_probability_decision_count": len(target_free_probability_rows),
        "target_free_feature_causality_violation_count": sum(
            int(row["max_input_ts"]) > int(row["decision_ts"]) for row in feature_rows
        ),
        "probability_calibration_metrics": probability_report,
        "market_implied_probability_used_as_conditioning_feature": True,
        "market_implied_probability_used_as_direct_fair_value_ev": False,
        "enabled_actions": sorted(HTS_ACTIONS),
        "disabled_actions": sorted(SBC_ACTIONS),
        "sell_before_close_exit_value_fabricated": False,
        "cost_model": profile["probability_to_net_ev"],
        "policy_selected_calibration_market_count": conformal_artifact[
            "selected_calibration_market_count"
        ],
        "policy_selected_calibration_side_distribution": conformal_artifact[
            "selected_side_distribution"
        ],
        "conformal_penalty_by_side": {
            side: float(conformal_artifact["sides"][side]["calibration_penalty"]) for side in SIDES
        },
        "calibration_and_target_free_gate": calibration_gate,
        "raw_positive_guard_compatible_hts_row_count": len(raw_positive),
        "raw_positive_guard_compatible_market_count": len(
            {str(row["market_id"]) for row in raw_positive}
        ),
        "positive_lcb_selected_market_count": len(static_selected),
        "positive_lcb_selected_side_distribution": {side: static_sides[side] for side in SIDES},
        "full_guard_accepted_bet_count": len(accepted),
        "full_guard_accepted_unique_market_count": viability[
            "full_guard_accepted_unique_market_count"
        ],
        "full_guard_accepted_side_market_count": viability["full_guard_accepted_side_market_count"],
        "full_guard_accepted_action_distribution": dict(sorted(accepted_actions.items())),
        "full_guard_blocking_reason_distribution": dict(sorted(blockers.items())),
        "policy_selected_no_trade_decision_count": sum(
            row["source_selected_action"] == "NO_TRADE" for row in replay
        ),
        "target_free_action_viability_passed": viability["passed"],
        "target_free_action_viability_checks": viability["checks"],
        "target_free_action_viability_blocking_reason_codes": viability["reason_codes"],
        "collector_resume_allowed": viability["passed"],
        "new_strictly_later_future_holdout_required": True,
        "target_free_labels_outcomes_settlement_targets_or_pnl_opened": False,
        "uses_target_free_check_labels_for_tuning": False,
        "uses_current_oof_validation_or_confirmatory_pnl_for_tuning": False,
        "hyperparameter_search_enabled": False,
        "threshold_search_enabled": False,
        "promotion_evidence": False,
        **_blocked_safety_fields(),
        "paper_candidate_allowed": False,
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def _verify_source_lineage(
    lineage: dict[str, str],
    *,
    fit_descriptor: dict[str, str],
    conformal_descriptor: dict[str, str],
    role_descriptor: dict[str, str],
    role_rows_descriptor: dict[str, str],
) -> None:
    expected = {
        "v5_fit_action_rows_sha256": fit_descriptor["sha256"],
        "v5_conformal_action_rows_sha256": conformal_descriptor["sha256"],
        "v5_role_assignment_manifest_sha256": role_descriptor["sha256"],
        "v5_role_rows_sha256": role_rows_descriptor["sha256"],
    }
    for name, value in expected.items():
        if lineage[name] != value:
            raise ValueError(f"#210 source lineage mismatch: {name}")


def _validate_role_market_counts(
    train_rows: list[dict[str, Any]],
    probability_rows: list[dict[str, Any]],
    conformal_rows: list[dict[str, Any]],
    *,
    profile: dict[str, Any],
) -> None:
    roles = dict(profile["chronological_roles"])
    expected = (
        (train_rows, FIT_ROLE, int(roles["side_probability_fit_market_count"])),
        (
            probability_rows,
            PROBABILITY_CALIBRATION_ROLE,
            int(roles["probability_calibration_market_count"]),
        ),
        (conformal_rows, CALIBRATION_ROLE, int(roles["net_return_conformal_market_count"])),
    )
    for rows, role, market_count in expected:
        if len({str(row["market_id"]) for row in rows}) != market_count:
            raise ValueError(f"{role} market count differs from #210 profile")
        if {str(row["role"]) for row in rows} != {role}:
            raise ValueError(f"{role} role mapping invalid")


def _attach_rank_compatibility(row: dict[str, Any]) -> dict[str, Any]:
    raw = float(row["raw_direct_predicted_net_return"])
    return {
        **row,
        "raw_pairwise_rank_score": raw,
        "pairwise_group_normalized_rank_score": raw,
        "action_advantage_lcb_score_bucket": "calibrated_side_probability_net_ev_v6_1",
        "action_advantage_lcb_estimate_source": row["ranking_score_source"],
    }


def _finite_feature(features: dict[str, Any], name: str) -> float:
    if name not in features:
        raise ValueError(f"missing decision-time feature: {name}")
    return _finite_number(features[name], name)


def _finite_number(value: Any, name: str) -> float:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise ValueError(f"non-finite decision-time value: {name}")
    return float(value)


def _logit(value: float, *, clip: float) -> float:
    bounded = min(1.0 - clip, max(clip, float(value)))
    return math.log(bounded / (1.0 - bounded))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _sigmoid_array(values: np.ndarray) -> np.ndarray:
    return np.where(
        values >= 0.0, 1.0 / (1.0 + np.exp(-values)), np.exp(values) / (1.0 + np.exp(values))
    )


def _brier(labels: list[int], probabilities: list[float]) -> float:
    return sum(
        (float(probability) - label) ** 2
        for label, probability in zip(labels, probabilities, strict=True)
    ) / len(labels)


def _log_loss(labels: list[int], probabilities: list[float]) -> float:
    epsilon = 1e-12
    return -sum(
        label * math.log(min(1.0 - epsilon, max(epsilon, probability)))
        + (1 - label) * math.log(min(1.0 - epsilon, max(epsilon, 1.0 - probability)))
        for label, probability in zip(labels, probabilities, strict=True)
    ) / len(labels)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _probability_markdown(report: dict[str, Any]) -> str:
    calibration = report["probability_calibration"]
    return "\n".join(
        [
            "# v6.1 probability calibration",
            "",
            f"- calibration markets: `{calibration['market_count']}`",
            f"- raw Brier: `{calibration['raw_brier_score']:.8f}`",
            f"- calibrated Brier: `{calibration['calibrated_brier_score']:.8f}`",
            "- probability-calibration policy PnL computed: `false`",
            "- threshold search enabled: `false`",
            "",
        ]
    )


def _report_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# #210 calibrated side-probability net-EV v6.1",
            "",
            f"- target-free viability passed: `{report['target_free_action_viability_passed']}`",
            f"- blockers: `{report['target_free_action_viability_blocking_reason_codes']}`",
            f"- conformal selected markets: `{report['policy_selected_calibration_market_count']}`",
            f"- positive-LCB markets: `{report['positive_lcb_selected_market_count']}`",
            f"- full-guard accepted unique markets: `{report['full_guard_accepted_unique_market_count']}`",
            f"- accepted sides: `{report['full_guard_accepted_side_market_count']}`",
            f"- collector resume allowed: `{report['collector_resume_allowed']}`",
            "- target-free outcomes/labels/PnL opened: `false`",
            "- promotion evidence: `false`",
            "",
        ]
    )
