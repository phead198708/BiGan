"""Run the #209 v6-on-v5 target-free viability diagnostic."""

from __future__ import annotations

import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_evaluation import (
    FORBIDDEN_TARGET_FIELDS,
)
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_prediction_freeze import (
    _materialize_future_action_rows,
    _materialize_selected_window_features,
)
from bigan.v8.polymarket.training.execution_layer_v2_guard_compatible_conformal_net_return_v5 import (
    _raw_target_stripped_predictions,
)
from bigan.v8.polymarket.training.execution_layer_v2_guard_compatible_direct_net_return_v4 import (
    _row_sort_key,
    _train_regressor,
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
    _development_row_rejection_reasons,
    _find_nonempty_fields,
    _load_json,
    _load_jsonl,
    _prior_exclusion_summary,
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
    TARGET_FIELDS,
    _calibration_gate,
    _target_free_check_support,
    _validate_labeled_role_rows,
    _validate_target_free_check_rows,
    _xgb_model_config,
    attach_frozen_execution_compatibility,
    build_policy_selected_conformal_artifact,
    select_sequential_policy_rows,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6_future_prediction import (
    _candidate_predictions,
)

PROFILE_SCHEMA_VERSION = "bigan-v8-v6-on-v5-target-free-diagnostic-profile-v1"
REPORT_SCHEMA_VERSION = "bigan-v8-v6-on-v5-target-free-viability-report-v1"
MANIFEST_SCHEMA_VERSION = "bigan-v8-v6-on-v5-target-free-diagnostic-manifest-v1"
CANDIDATE_NAME = "v6_on_v5_policy_selected_conformal_target_free_diagnostic"


@dataclass(frozen=True, slots=True)
class V6OnV5TargetFreeDiagnosticConfig:
    """Pinned inputs for one non-promotional #209 diagnostic run."""

    run_id: str
    output_dir: Path | str
    diagnostic_profile_path: Path | str
    expected_diagnostic_profile_sha256: str
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
        _require_sha256(
            self.expected_diagnostic_profile_sha256,
            name="expected_diagnostic_profile_sha256",
        )
        _require_git_sha(self.implementation_commit)
        if self.freeze_created_ts <= 0:
            raise ValueError("freeze_created_ts must be positive")
        for field in (
            "output_dir",
            "diagnostic_profile_path",
            "v5_freeze_manifest_path",
            "v6_profile_path",
            "v6_preregistration_manifest_path",
            "collector_index_path",
            "feature_contract_path",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)))


def validate_v6_on_v5_diagnostic_profile(profile: dict[str, Any]) -> None:
    """Reject drift from the issue #209 pre-registered diagnostic contract."""

    lineage = dict(profile.get("source_lineage") or {})
    roles = dict(profile.get("chronological_roles") or {})
    check = dict(profile.get("target_free_check") or {})
    model = dict(profile.get("model_and_policy_contract") or {})
    prohibited = dict(profile.get("prohibited_inputs") or {})
    expected_safety = _blocked_safety_fields() | {"paper_candidate_allowed": False}
    checks = {
        "schema": profile.get("schema_version") == PROFILE_SCHEMA_VERSION,
        "candidate": profile.get("candidate_name") == CANDIDATE_NAME,
        "issue": profile.get("issue_number") == 209,
        "frozen_diagnostic": profile.get("frozen") is True
        and profile.get("purpose") == "retrospective_development_diagnostic_only",
        "lineage_hashes": set(lineage)
        == {
            "v5_freeze_manifest_sha256",
            "v5_fit_action_rows_sha256",
            "v5_calibration_action_rows_sha256",
            "v5_role_assignment_manifest_sha256",
            "v5_role_rows_sha256",
            "v6_preregistration_profile_sha256",
            "v6_preregistration_manifest_sha256",
            "feature_contract_sha256",
        }
        and all(_is_sha256(value) for value in lineage.values()),
        "roles": roles
        == {
            "point_model_fit_market_count": 135,
            "policy_selected_conformal_market_count": 60,
            "target_free_check_market_count": 50,
            "assignment": ("v5_fit_then_v5_calibration_then_strictly_later_post_issue204_check"),
        },
        "target_free_boundary": check.get("minimum_eligible_index_sequence") == 237
        and check.get("maximum_index_scan_count") == 100
        and check.get("selection_method") == "earliest_quality_valid_post_issue204_disjoint_rows",
        "target_free_support": check.get("minimum_full_guard_accepted_market_count_per_side") == 5,
        "target_free_sealed": check.get("result_dependent_extension_allowed") is False
        and check.get("labels_outcomes_settlement_targets_or_pnl_opened") is False,
        "fixed_model": model.get("point_model_config_source") == "issue207_frozen_v6_profile"
        and model.get("feature_set_source") == "issue207_frozen_feature_contract"
        and model.get("conformal_method_source") == "issue207_policy_selected_conformal_contract"
        and model.get("execution_guard_source") == "issue207_unchanged_full_execution_guard"
        and model.get("hyperparameter_search_enabled") is False
        and model.get("threshold_search_enabled") is False,
        "no_policy_mutation": model.get("cost_model_mutation_allowed") is False
        and model.get("execution_guard_mutation_allowed") is False
        and model.get("sizing_or_exposure_mutation_allowed") is False,
        "prohibited_inputs": prohibited
        == {
            "uses_204_outcomes_for_fitting": False,
            "uses_204_pnl_for_tuning": False,
            "uses_target_free_check_labels_for_tuning": False,
            "uses_current_oof_validation_or_confirmatory_pnl_for_tuning": False,
            "result_driven_model_threshold_penalty_or_gate_change_allowed": False,
        },
        "safety": profile.get("safety") == expected_safety,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("#209 diagnostic profile invalid: " + ", ".join(blockers))


def run_v6_on_v5_target_free_diagnostic(
    config: V6OnV5TargetFreeDiagnosticConfig,
) -> dict[str, Any]:
    """Fit on v5 development rows and evaluate only target-free post-#204 support."""

    profile_path = config.diagnostic_profile_path.resolve()
    v5_manifest_path = config.v5_freeze_manifest_path.resolve()
    v6_profile_path = config.v6_profile_path.resolve()
    prereg_path = config.v6_preregistration_manifest_path.resolve()
    index_path = config.collector_index_path.resolve()
    feature_contract_path = config.feature_contract_path.resolve()
    _verify_pin(profile_path, config.expected_diagnostic_profile_sha256, "#209 profile")
    diagnostic_profile = _load_json(profile_path)
    validate_v6_on_v5_diagnostic_profile(diagnostic_profile)
    lineage = dict(diagnostic_profile["source_lineage"])
    _verify_pin(v5_manifest_path, lineage["v5_freeze_manifest_sha256"], "v5 freeze manifest")
    _verify_pin(v6_profile_path, lineage["v6_preregistration_profile_sha256"], "v6 profile")
    _verify_pin(prereg_path, lineage["v6_preregistration_manifest_sha256"], "v6 prereg")
    _verify_pin(feature_contract_path, lineage["feature_contract_sha256"], "feature contract")
    v6_profile = _load_json(v6_profile_path)
    validate_policy_selected_conformal_v6_profile(v6_profile)
    prereg = _load_json(prereg_path)
    _validate_preregistration_lineage(
        prereg,
        expected_v6_profile_sha256=lineage["v6_preregistration_profile_sha256"],
    )
    feature_contract = _load_json(feature_contract_path)
    feature_columns = tuple(str(value) for value in feature_contract["feature_columns"])

    v5_manifest = _load_json(v5_manifest_path)
    fit_descriptor = _verified_descriptor(
        v5_manifest.get("development_train_action_rows"), "v5 fit action rows"
    )
    calibration_descriptor = _verified_descriptor(
        v5_manifest.get("development_calibration_action_rows"),
        "v5 calibration action rows",
    )
    if fit_descriptor["sha256"] != lineage["v5_fit_action_rows_sha256"]:
        raise ValueError("v5 fit action-row lineage mismatch")
    if calibration_descriptor["sha256"] != lineage["v5_calibration_action_rows_sha256"]:
        raise ValueError("v5 calibration action-row lineage mismatch")
    role_descriptor = _verified_descriptor(
        v5_manifest.get("role_assignment_manifest"), "v5 role assignment manifest"
    )
    if role_descriptor["sha256"] != lineage["v5_role_assignment_manifest_sha256"]:
        raise ValueError("v5 role-assignment lineage mismatch")
    role_manifest = _load_json(Path(role_descriptor["path"]))
    role_rows_descriptor = _verified_descriptor(
        role_manifest.get("selected_rows"), "v5 role assignment rows"
    )
    if role_rows_descriptor["sha256"] != lineage["v5_role_rows_sha256"]:
        raise ValueError("v5 role-row lineage mismatch")

    observed_index_hash = _sha256_file(index_path)
    index_rows = load_and_validate_persistent_outcome_blind_index(index_path)
    if _sha256_file(index_path) != observed_index_hash:
        raise ValueError("collector index changed while freezing #209 prefix")
    selected_rows, scanned_count, rejection_distribution = _select_target_free_rows(
        index_rows,
        prereg=prereg,
        v6_profile=v6_profile,
        diagnostic_profile=diagnostic_profile,
    )
    target_count = int(diagnostic_profile["chronological_roles"]["target_free_check_market_count"])
    if len(selected_rows) != target_count:
        raise ValueError("#209 target-free check does not have exactly 50 valid markets")
    if config.freeze_created_ts <= max(int(row["market_end_ts"]) for row in selected_rows):
        raise ValueError("#209 target-free markets are not all closed before freeze")

    run_dir = config.output_dir.resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    selected_path = run_dir / "v6_on_v5_target_free_selected_rows.jsonl"
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
        feature_columns=feature_columns,
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
    _validate_target_free_check_rows(action_rows, feature_columns=feature_columns)
    feature_rows_path = run_dir / "v6_on_v5_target_free_feature_rows.jsonl"
    action_rows_path = run_dir / "v6_on_v5_target_free_five_action_rows.jsonl"
    _write_jsonl(feature_rows_path, feature_rows)
    _write_jsonl(action_rows_path, action_rows)

    pre_label = {
        "schema_version": "bigan-v8-v6-on-v5-pre-label-lineage-audit-v1",
        "run_id": config.run_id,
        "implementation_commit": config.implementation_commit,
        "diagnostic_profile": _descriptor(profile_path),
        "v5_freeze_manifest": _descriptor(v5_manifest_path),
        "v5_fit_action_rows": fit_descriptor,
        "v5_calibration_action_rows": calibration_descriptor,
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
        "v5_fit_or_calibration_target_rows_opened_before_audit": False,
        "issue204_outcome_settlement_target_or_pnl_files_opened": False,
        "target_free_check_labels_outcomes_settlement_targets_or_pnl_opened": False,
        "raw_resolution_artifact_opened": False,
        "pre_label_access_validation_passed": True,
        **_blocked_safety_fields(),
        "paper_candidate_allowed": False,
    }
    pre_label["audit_id"] = canonical_json_sha256(pre_label)
    pre_label_path = run_dir / "v6_on_v5_pre_label_lineage_audit.json"
    _write_json(pre_label_path, pre_label)
    _write_text(pre_label_path.with_suffix(".md"), _pre_label_markdown(pre_label))

    fit_rows = _normalize_v5_labeled_rows(
        _load_jsonl(Path(fit_descriptor["path"])),
        role=FIT_ROLE,
        expected_source_roles={"development_train", "development_calibration"},
        feature_columns=feature_columns,
    )
    calibration_rows = _normalize_v5_labeled_rows(
        _load_jsonl(Path(calibration_descriptor["path"])),
        role=CALIBRATION_ROLE,
        expected_source_roles={"confirmatory_validation"},
        feature_columns=feature_columns,
    )
    _validate_labeled_role_rows(fit_rows, expected_market_count=135, role=FIT_ROLE)
    _validate_labeled_role_rows(
        calibration_rows,
        expected_market_count=60,
        role=CALIBRATION_ROLE,
    )
    if max(int(row["decision_ts"]) for row in fit_rows) >= min(
        int(row["decision_ts"]) for row in calibration_rows
    ):
        raise ValueError("v5 fit rows do not strictly precede v5 calibration rows")
    if max(int(row["decision_ts"]) for row in calibration_rows) >= min(
        int(row["decision_ts"]) for row in action_rows
    ):
        raise ValueError("v5 calibration rows do not strictly precede target-free check rows")

    booster = _train_regressor(
        fit_rows,
        feature_columns=feature_columns,
        model_config=_xgb_model_config(dict(v6_profile["point_model"])),
    )
    model_path = run_dir / "v6_on_v5_policy_selected_conformal_model.xgb.json"
    booster.save_model(model_path)
    fit_predictions = attach_frozen_execution_compatibility(
        _raw_target_stripped_predictions(booster, fit_rows, feature_columns=feature_columns)
    )
    calibration_predictions = attach_frozen_execution_compatibility(
        _raw_target_stripped_predictions(
            booster,
            calibration_rows,
            feature_columns=feature_columns,
        )
    )
    calibration_artifact = build_policy_selected_conformal_artifact(
        calibration_predictions,
        target_rows=calibration_rows,
        profile=v6_profile,
        feature_contract_sha256=lineage["feature_contract_sha256"],
    )
    calibration_artifact = {
        **calibration_artifact,
        "diagnostic_candidate_name": CANDIDATE_NAME,
        "source_labeled_corpus": "frozen_issue203_v5_development_only",
        "target_free_check_source": "earliest_50_post_issue204_outcome_blind_markets",
        "uses_target_free_check_labels_for_tuning": False,
        "diagnostic_only": True,
    }
    calibration_artifact["diagnostic_calibration_artifact_id"] = canonical_json_sha256(
        calibration_artifact
    )
    calibration_path = run_dir / "v6_on_v5_policy_selected_conformal_artifact.json"
    _write_json(calibration_path, calibration_artifact)

    scored_check = _candidate_predictions(
        action_rows,
        model_descriptor=_descriptor(model_path),
        calibration_artifact=calibration_artifact,
        profile=v6_profile,
        feature_columns=feature_columns,
    )
    static_selected = select_sequential_policy_rows(
        scored_check,
        score_field="conformal_net_return_lower_bound",
        require_positive=True,
    )
    static_support = _target_free_check_support(static_selected, profile=v6_profile)
    calibration_gate = _calibration_gate(
        calibration_artifact=calibration_artifact,
        check_support=static_support,
        corpus_audits=[{"blocking_reason_codes": []}],
    )
    replay = _outcome_blind_acceptance_replay(
        scored_check,
        entry_threshold=0.0,
        runner_up_advantage_threshold=0.0,
    )
    accepted = [row for row in replay if row["execution_guard_order_allowed"]]
    accepted_side_counts = Counter(str(row["selected_side"]) for row in accepted)
    minimum_per_side = int(
        diagnostic_profile["target_free_check"]["minimum_full_guard_accepted_market_count_per_side"]
    )
    full_guard_support_passed = all(
        accepted_side_counts.get(side, 0) >= minimum_per_side for side in SIDES
    )
    viability_reasons = []
    if not calibration_gate["passed"]:
        viability_reasons.extend(calibration_gate["reason_codes"])
    if not full_guard_support_passed:
        viability_reasons.append("target_free_full_guard_side_support_failed")
    if not accepted:
        viability_reasons.append("target_free_all_actions_no_trade_or_guard_blocked")
    viability_reasons = sorted(set(viability_reasons))

    fit_predictions_path = run_dir / "v6_on_v5_target_stripped_fit_predictions.jsonl"
    calibration_predictions_path = (
        run_dir / "v6_on_v5_target_stripped_calibration_predictions.jsonl"
    )
    scored_check_path = run_dir / "v6_on_v5_target_free_scored_rows.jsonl"
    replay_path = run_dir / "v6_on_v5_target_free_full_guard_replay.jsonl"
    accepted_path = run_dir / "v6_on_v5_target_free_guard_accepted_bets.jsonl"
    _write_jsonl(fit_predictions_path, fit_predictions)
    _write_jsonl(calibration_predictions_path, calibration_predictions)
    _write_jsonl(scored_check_path, scored_check)
    _write_jsonl(replay_path, replay)
    _write_jsonl(accepted_path, accepted)

    report = _build_report(
        config=config,
        selected_rows=selected_rows,
        feature_rows=feature_rows,
        action_rows=action_rows,
        fit_rows=fit_rows,
        calibration_rows=calibration_rows,
        calibration_artifact=calibration_artifact,
        scored_check=scored_check,
        static_selected=static_selected,
        static_support=static_support,
        calibration_gate=calibration_gate,
        replay=replay,
        accepted=accepted,
        minimum_per_side=minimum_per_side,
        viability_reasons=viability_reasons,
    )
    report_path = run_dir / "v6_on_v5_target_free_viability_report.json"
    _write_json(report_path, report)
    _write_text(report_path.with_suffix(".md"), _report_markdown(report))
    training_report = {
        "schema_version": "bigan-v8-v6-on-v5-training-report-v1",
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "model": _descriptor(model_path),
        "model_target": "target_net_pnl_per_contract",
        "training_target_includes_costs": True,
        "fit_market_count": 135,
        "fit_action_row_count": len(fit_rows),
        "calibration_market_count": 60,
        "calibration_action_row_count": len(calibration_rows),
        "hyperparameter_search_enabled": False,
        "uses_204_outcomes_for_fitting": False,
        "uses_204_pnl_for_tuning": False,
        "uses_target_free_check_labels_for_tuning": False,
        "diagnostic_only": True,
        **_blocked_safety_fields(),
        "paper_candidate_allowed": False,
    }
    training_report["report_id"] = canonical_json_sha256(training_report)
    training_report_path = run_dir / "v6_on_v5_training_report.json"
    _write_json(training_report_path, training_report)
    _write_text(training_report_path.with_suffix(".md"), _training_markdown(training_report))
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        "diagnostic_profile": _descriptor(profile_path),
        "pre_label_lineage_audit": _descriptor(pre_label_path),
        "training_report": _descriptor(training_report_path),
        "model": _descriptor(model_path),
        "calibration_artifact": _descriptor(calibration_path),
        "selected_target_free_rows": _descriptor(selected_path),
        "target_free_feature_rows": _descriptor(feature_rows_path),
        "target_free_five_action_rows": _descriptor(action_rows_path),
        "target_free_scored_rows": _descriptor(scored_check_path),
        "target_free_full_guard_replay": _descriptor(replay_path),
        "target_free_guard_accepted_bets": _descriptor(accepted_path),
        "viability_report": _descriptor(report_path),
        "opened_raw_feature_artifacts": raw_descriptors,
        "diagnostic_viability_passed": report["diagnostic_viability_passed"],
        "diagnostic_viability_blocking_reason_codes": viability_reasons,
        "future_pnl_evaluation_required": True,
        "promotion_evidence": False,
        "issue207_authoritative_protocol_mutated": False,
        **_blocked_safety_fields(),
        "paper_candidate_allowed": False,
    }
    manifest["diagnostic_manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v6_on_v5_diagnostic_manifest.json"
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


def _select_target_free_rows(
    index_rows: list[dict[str, Any]],
    *,
    prereg: dict[str, Any],
    v6_profile: dict[str, Any],
    diagnostic_profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
    prefix = _load_jsonl(
        Path(_verified_descriptor(prereg["collector_index_prefix"], "collector prefix")["path"])
    )
    if len(index_rows) < len(prefix) or index_rows[: len(prefix)] != prefix:
        raise ValueError("collector index does not preserve #207 preregistration prefix")
    issue204_window = _load_json(
        Path(_verified_descriptor(prereg["issue204_window_manifest"], "#204 window")["path"])
    )
    issue204_rows = _load_jsonl(
        Path(_verified_descriptor(issue204_window["selected_rows"], "#204 selected rows")["path"])
    )
    exclusion = _prior_exclusion_summary(issue204_rows)
    check = diagnostic_profile["target_free_check"]
    minimum_sequence = int(check["minimum_eligible_index_sequence"])
    scan_cap = int(check["maximum_index_scan_count"])
    target_count = int(diagnostic_profile["chronological_roles"]["target_free_check_market_count"])
    selected: list[dict[str, Any]] = []
    reasons = Counter()
    seen_markets: set[str] = set()
    seen_slugs: set[str] = set()
    seen_source_hashes: set[str] = set()
    scanned = 0
    for row in [value for value in index_rows if int(value["sequence"]) >= minimum_sequence][
        :scan_cap
    ]:
        scanned += 1
        row_reasons = _development_row_rejection_reasons(
            row,
            profile=v6_profile,
            exclusion=exclusion,
            seen_markets=seen_markets,
            seen_slugs=seen_slugs,
            seen_source_hashes=seen_source_hashes,
        )
        if row_reasons:
            reasons.update(row_reasons)
            continue
        selected.append(row)
        seen_markets.add(str(row["market_id"]))
        seen_slugs.add(str(row["slug"]))
        seen_source_hashes.add(str(row["source_row_hash"]))
        if len(selected) == target_count:
            break
    if _find_nonempty_fields(selected, FORBIDDEN_TARGET_FIELDS):
        raise ValueError("#209 selected target-free rows contain forbidden targets")
    if any(row.get("labels_outcomes_or_pnl_opened") is not False for row in selected):
        raise ValueError("#209 selected target-free row sealing is invalid")
    return selected, scanned, dict(sorted(reasons.items()))


def _normalize_v5_labeled_rows(
    rows: list[dict[str, Any]],
    *,
    role: str,
    expected_source_roles: set[str],
    feature_columns: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Adapt immutable v5 action rows to v6 role names without changing payloads."""

    output = []
    for row in rows:
        if str(row.get("role") or "") not in expected_source_roles:
            raise ValueError("v5 source role does not match #209 frozen mapping")
        decision_ts = int(row.get("decision_ts") or 0)
        if decision_ts <= 0 or int(row.get("max_input_ts") or 0) > decision_ts:
            raise ValueError("v5 labeled row has feature causality violation")
        features = dict(row.get("decision_time_features") or {})
        if set(feature_columns) - set(features):
            raise ValueError("v5 labeled row is missing a decision-time feature")
        if any(
            not isinstance(features[name], int | float) or not math.isfinite(float(features[name]))
            for name in feature_columns
        ):
            raise ValueError("v5 labeled row has non-finite decision-time feature")
        if any(_nonempty(features.get(name)) for name in TARGET_FIELDS):
            raise ValueError("v5 target leaked into decision-time feature inputs")
        target = row.get("target_net_pnl_per_contract")
        if not isinstance(target, int | float) or not math.isfinite(float(target)):
            raise ValueError("v5 labeled row target is missing or invalid")
        normalized = {
            **row,
            "source_v5_role": row["role"],
            "role": role,
            "development_role": role,
            "target_used_as_decision_input": False,
            "outcome_fields_used_as_decision_input": False,
        }
        normalized["v6_on_v5_labeled_row_sha256"] = canonical_json_sha256(normalized)
        output.append(normalized)
    output.sort(key=_row_sort_key)
    groups: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in output:
        groups[(str(row["market_id"]), int(row["decision_ts"]))].add(str(row["action"]))
    if any(actions != set(REQUIRED_ACTIONS) for actions in groups.values()):
        raise ValueError("v5 labeled decision group is not a complete five-action grid")
    return output


def _validate_preregistration_lineage(
    prereg: dict[str, Any], *, expected_v6_profile_sha256: str
) -> None:
    if prereg.get("preregistration_passed") is not True:
        raise ValueError("#207 preregistration did not pass")
    if prereg.get("new_development_target_accessed") is not False:
        raise ValueError("#207 preregistration target-access state invalid")
    profile = _verified_descriptor(prereg.get("profile"), "#207 v6 profile")
    if profile["sha256"] != expected_v6_profile_sha256:
        raise ValueError("#207 preregistration does not bind the provided v6 profile")
    for key, expected in _blocked_safety_fields().items():
        if prereg.get(key) != expected:
            raise ValueError(f"#207 preregistration safety invalid: {key}")


def _build_report(
    *,
    config: V6OnV5TargetFreeDiagnosticConfig,
    selected_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    fit_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    calibration_artifact: dict[str, Any],
    scored_check: list[dict[str, Any]],
    static_selected: list[dict[str, Any]],
    static_support: dict[str, Any],
    calibration_gate: dict[str, Any],
    replay: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    minimum_per_side: int,
    viability_reasons: list[str],
) -> dict[str, Any]:
    positive_raw = [
        row
        for row in scored_check
        if row["action"] != "NO_TRADE"
        and row["guard_compatible_before_ranking"]
        and float(row["raw_direct_predicted_net_return"]) > 0.0
    ]
    positive_lcb = [
        row
        for row in scored_check
        if row["action"] != "NO_TRADE"
        and row["guard_compatible_before_ranking"]
        and float(row["conformal_net_return_lower_bound"]) > 0.0
    ]
    accepted_sides = Counter(str(row["selected_side"]) for row in accepted)
    accepted_actions = Counter(str(row["executed_action"]) for row in accepted)
    no_trade_count = sum(row["source_selected_action"] == "NO_TRADE" for row in replay)
    blocking = Counter(
        str(reason) for row in replay for reason in row["execution_blocking_reason_codes"]
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_id": None,
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        "diagnostic_only": True,
        "fit_market_count": len({str(row["market_id"]) for row in fit_rows}),
        "fit_action_row_count": len(fit_rows),
        "conformal_market_count": len({str(row["market_id"]) for row in calibration_rows}),
        "conformal_action_row_count": len(calibration_rows),
        "target_free_check_market_count": len(selected_rows),
        "target_free_feature_row_count": len(feature_rows),
        "target_free_five_action_row_count": len(action_rows),
        "target_free_decision_count": len(replay),
        "target_free_feature_causality_violation_count": sum(
            int(row["max_input_ts"]) > int(row["decision_ts"]) for row in feature_rows
        ),
        "policy_selected_calibration_market_count": calibration_artifact[
            "selected_calibration_market_count"
        ],
        "policy_selected_calibration_side_distribution": calibration_artifact[
            "selected_side_distribution"
        ],
        "conformal_penalty_by_side": {
            side: float(calibration_artifact["sides"][side]["calibration_penalty"])
            for side in SIDES
        },
        "calibration_source_by_side": {
            side: calibration_artifact["sides"][side]["calibration_source"] for side in SIDES
        },
        "calibration_gate_passed": calibration_gate["passed"],
        "calibration_gate_checks": calibration_gate["checks"],
        "calibration_gate_blocking_reason_codes": calibration_gate["reason_codes"],
        "raw_positive_guard_compatible_trade_row_count": len(positive_raw),
        "raw_positive_guard_compatible_market_count": len(
            {str(row["market_id"]) for row in positive_raw}
        ),
        "positive_calibrated_lcb_trade_row_count": len(positive_lcb),
        "positive_calibrated_lcb_market_count": len(
            {str(row["market_id"]) for row in positive_lcb}
        ),
        "static_policy_selected_market_count": len(static_selected),
        "static_policy_selected_side_distribution": dict(
            sorted(Counter(str(row["side"]) for row in static_selected).items())
        ),
        "static_target_free_support": static_support,
        "full_guard_accepted_bet_count": len(accepted),
        "full_guard_accepted_unique_market_count": len({str(row["market_id"]) for row in accepted}),
        "full_guard_accepted_side_distribution": {
            side: accepted_sides.get(side, 0) for side in SIDES
        },
        "full_guard_accepted_action_distribution": dict(sorted(accepted_actions.items())),
        "minimum_full_guard_accepted_market_count_per_side": minimum_per_side,
        "full_guard_side_support_passed": all(
            accepted_sides.get(side, 0) >= minimum_per_side for side in SIDES
        ),
        "policy_selected_no_trade_decision_count": no_trade_count,
        "full_guard_blocking_reason_distribution": dict(sorted(blocking.items())),
        "diagnostic_viability_passed": not viability_reasons,
        "diagnostic_viability_blocking_reason_codes": viability_reasons,
        "issue207_authoritative_protocol_mutated": False,
        "future_pnl_evaluation_required": True,
        "promotion_evidence": False,
        "issue204_outcome_settlement_target_or_pnl_files_opened": False,
        "target_free_check_labels_outcomes_settlement_targets_or_pnl_opened": False,
        "uses_204_outcomes_for_fitting": False,
        "uses_204_pnl_for_tuning": False,
        "uses_target_free_check_labels_for_tuning": False,
        "model_threshold_penalty_guard_cost_sizing_or_exposure_tuned": False,
        **_blocked_safety_fields(),
        "paper_candidate_allowed": False,
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def _report_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6-on-v5 target-free viability diagnostic",
            "",
            f"- fit / conformal / target-free markets: `{report['fit_market_count']} / {report['conformal_market_count']} / {report['target_free_check_market_count']}`",
            f"- policy-selected conformal penalties: `{report['conformal_penalty_by_side']}`",
            f"- raw-positive guard-compatible markets: `{report['raw_positive_guard_compatible_market_count']}`",
            f"- positive calibrated-LCB markets: `{report['positive_calibrated_lcb_market_count']}`",
            f"- full-guard accepted bets: `{report['full_guard_accepted_bet_count']}`",
            f"- full-guard accepted sides: `{report['full_guard_accepted_side_distribution']}`",
            f"- selected NO_TRADE decisions: `{report['policy_selected_no_trade_decision_count']}`",
            f"- viability passed: `{str(report['diagnostic_viability_passed']).lower()}`",
            f"- blockers: `{report['diagnostic_viability_blocking_reason_codes']}`",
            "- #204 outcomes/PnL opened: `false`",
            "- promotion evidence: `false`",
            "- future PnL evaluation still required: `true`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )


def _pre_label_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6-on-v5 pre-label lineage audit",
            "",
            f"- target-free markets: `{report['selected_market_count']}`",
            f"- sequence range: `{report['selected_sequence_start']}..{report['selected_sequence_end']}`",
            f"- feature causality violations: `{report['target_free_feature_causality_violation_count']}`",
            "- v5 targets opened before audit: `false`",
            "- #204 outcome/settlement/target/PnL opened: `false`",
            "- target-free labels/outcomes/PnL opened: `false`",
            "- audit passed: `true`",
            "",
        ]
    )


def _training_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6-on-v5 training report",
            "",
            f"- fit markets / rows: `{report['fit_market_count']} / {report['fit_action_row_count']}`",
            f"- conformal markets / rows: `{report['calibration_market_count']} / {report['calibration_action_row_count']}`",
            f"- model SHA-256: `{report['model']['sha256']}`",
            "- hyperparameter search: `false`",
            "- #204 outcomes/PnL used: `false`",
            "- diagnostic only: `true`",
            "",
        ]
    )


def _nonempty(value: Any) -> bool:
    return value not in (None, "", [], {})


def _is_sha256(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


__all__ = [
    "V6OnV5TargetFreeDiagnosticConfig",
    "run_v6_on_v5_target_free_diagnostic",
    "validate_v6_on_v5_diagnostic_profile",
]
