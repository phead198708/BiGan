"""Frozen #175 action-advantage LCB model and one-shot confirmatory gate."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.contracts import compact_safety_fields
from bigan.v8.polymarket.training.execution_layer_v2_hierarchical_action_value import (
    _accepted_bet_metrics,
    _market_robustness,
    _regression_metrics,
    _release_closed_positions,
)
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb import (
    CANDIDATE_NAME,
    validate_pairwise_action_advantage_lcb_feature_contract,
    validate_pairwise_action_advantage_lcb_protocol,
)
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_future_unseen_holdout import (
    load_and_validate_pairwise_future_unseen_holdout_pre_registration,
)
from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_action_value import (
    REQUIRED_ACTIONS,
)
from bigan.v8.polymarket.training.post_freeze_o_replay_aligned_source_ranking import (
    _v8_apply_simulated_order_to_state,
    _v8_execution_guard_config,
    _v8_execution_guard_decision,
    _v8_initial_runtime_state,
)

SCHEMA_PREFIX = "bigan-v8-execution-layer-v2-pairwise-action-advantage-lcb"
TRADE_FAMILIES = ("HOLD_TO_SETTLEMENT", "SELL_BEFORE_CLOSE")
ROLE_NAMES = (
    "development_train",
    "development_calibration",
    "confirmatory_validation",
)
ROLE_MARKET_COUNTS = {
    "development_train": 90,
    "development_calibration": 45,
    "confirmatory_validation": 60,
}
TARGET_MARKET_COUNT = sum(ROLE_MARKET_COUNTS.values())
SUPPLEMENTAL_SUPPORT_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-pairwise-supplemental-support-gate-manifest-v1"
)
SUPPLEMENTAL_SUPPORT_REPORT_SCHEMA_VERSION = (
    "bigan-v8-pairwise-supplemental-support-gate-report-v1"
)
CORE_SUPPORT_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-pairwise-precollection-continuation-manifest-v1"
)
CORE_SUPPORT_REPORT_SCHEMA_VERSION = (
    "bigan-v8-pairwise-precollection-support-gate-v1"
)
RANK_MODEL_FILENAME = "pairwise_action_advantage_ranker.xgb.json"
FORBIDDEN_DECISION_FIELDS = {
    "future_return",
    "oracle_action",
    "realized_pnl",
    "resolved_outcome",
    "settlement_pnl",
    "settlement_return",
    "target_net_return_after_cost",
    "total_net_pnl_per_notional",
    "total_net_return",
}


@dataclass(frozen=True, slots=True)
class PairwiseActionAdvantageLCBFitConfig:
    """Immutable inputs for fitting and one untouched confirmatory evaluation."""

    run_id: str
    output_dir: Path | str
    support_gate_manifest_path: Path | str
    expected_support_gate_manifest_sha256: str
    role_assignment_manifest_path: Path | str
    expected_role_assignment_manifest_sha256: str
    feature_contract_path: Path | str
    expected_feature_contract_sha256: str
    future_holdout_pre_registration_manifest_path: Path | str
    expected_future_holdout_pre_registration_manifest_sha256: str

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(
            self.expected_support_gate_manifest_sha256,
            name="support gate manifest SHA-256",
        )
        _require_sha256(
            self.expected_role_assignment_manifest_sha256,
            name="role assignment manifest SHA-256",
        )
        _require_sha256(
            self.expected_feature_contract_sha256,
            name="feature contract SHA-256",
        )
        _require_sha256(
            self.expected_future_holdout_pre_registration_manifest_sha256,
            name="future holdout pre-registration manifest SHA-256",
        )
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(
            self,
            "support_gate_manifest_path",
            Path(self.support_gate_manifest_path),
        )
        object.__setattr__(
            self,
            "role_assignment_manifest_path",
            Path(self.role_assignment_manifest_path),
        )
        object.__setattr__(self, "feature_contract_path", Path(self.feature_contract_path))
        object.__setattr__(
            self,
            "future_holdout_pre_registration_manifest_path",
            Path(self.future_holdout_pre_registration_manifest_path),
        )


def fit_pairwise_action_advantage_lcb(
    config: PairwiseActionAdvantageLCBFitConfig,
) -> dict[str, Any]:
    """Fit on 90 markets, calibrate on 45, and evaluate once on 60."""

    support_manifest_path = config.support_gate_manifest_path.resolve()
    _verify_pin(
        support_manifest_path,
        config.expected_support_gate_manifest_sha256,
        name="supplemental support gate manifest",
    )
    support_manifest = _load_json(support_manifest_path)

    role_manifest_path = config.role_assignment_manifest_path.resolve()
    _verify_pin(
        role_manifest_path,
        config.expected_role_assignment_manifest_sha256,
        name="role assignment manifest",
    )
    role_manifest = _load_json(role_manifest_path)
    support_lineage_audit = _validate_support_gate_lineage(
        support_manifest=support_manifest,
        support_manifest_path=support_manifest_path,
        role_manifest=role_manifest,
        role_manifest_path=role_manifest_path,
        expected_role_manifest_sha256=(
            config.expected_role_assignment_manifest_sha256
        ),
    )
    if role_manifest.get("role_assignment_ready") is not True:
        raise ValueError("role assignment is not ready")
    if role_manifest.get("labels_or_outcomes_opened_for_role_assignment") is not False:
        raise ValueError("role assignment did not preserve outcome blindness")
    protocol_descriptor = _verified_descriptor(
        role_manifest.get("protocol"), name="cross-fitted family protocol"
    )
    protocol = _load_json(Path(protocol_descriptor["path"]))
    validate_pairwise_action_advantage_lcb_protocol(protocol)
    compatibility_descriptor = _verified_descriptor(
        role_manifest.get("execution_compatible_feature_coverage_report"),
        name="execution-compatible feature coverage report",
    )
    compatibility_report = _load_json(Path(compatibility_descriptor["path"]))
    _validate_execution_compatibility_report(compatibility_report)

    frozen_feature_descriptor = _verified_descriptor(
        role_manifest.get("feature_contract"), name="frozen feature contract"
    )

    feature_contract_path = config.feature_contract_path.resolve()
    _verify_pin(
        feature_contract_path,
        config.expected_feature_contract_sha256,
        name="feature contract",
    )
    if str(feature_contract_path) != frozen_feature_descriptor["path"]:
        raise ValueError("feature contract path does not match precollection freeze")
    if config.expected_feature_contract_sha256.lower() != frozen_feature_descriptor["sha256"]:
        raise ValueError("feature contract SHA-256 does not match precollection freeze")
    feature_contract = _load_json(feature_contract_path)
    validate_pairwise_action_advantage_lcb_feature_contract(
        feature_contract,
        expected_parent_protocol_sha256=protocol_descriptor["sha256"],
    )
    future_holdout_pre_registration_path = (
        config.future_holdout_pre_registration_manifest_path.resolve()
    )
    _, future_holdout_lineage_audit = (
        load_and_validate_pairwise_future_unseen_holdout_pre_registration(
            future_holdout_pre_registration_path,
            config.expected_future_holdout_pre_registration_manifest_sha256,
            expected_candidate_protocol_path=Path(protocol_descriptor["path"]),
            expected_candidate_protocol_sha256=protocol_descriptor["sha256"],
            expected_feature_contract_path=feature_contract_path,
            expected_feature_contract_sha256=(
                config.expected_feature_contract_sha256
            ),
        )
    )
    selected_descriptor = _verified_descriptor(
        role_manifest.get("selected_rows"), name="role assignment rows"
    )
    role_rows = _load_jsonl(Path(selected_descriptor["path"]))
    _validate_role_rows(role_rows)
    selected_market_ids = {str(row["market_id"]) for row in role_rows}
    if canonical_json_sha256(sorted(selected_market_ids)) != str(
        role_manifest.get("selected_market_ids_sha256") or ""
    ):
        raise ValueError("role assignment selected-market hash mismatch")
    exclusion_descriptor = _verified_descriptor(
        role_manifest.get("prior_evidence_exclusion_registry"),
        name="prior evidence exclusion registry",
    )
    exclusion_registry = _load_json(Path(exclusion_descriptor["path"]))
    prior_market_ids = {str(value) for value in exclusion_registry.get("prior_market_ids") or []}
    prior_market_overlap = selected_market_ids & prior_market_ids
    if prior_market_overlap:
        raise ValueError("role assignment overlaps prior evidence")

    run_dir = config.output_dir / config.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    pre_label_audit = {
        "schema_version": f"{SCHEMA_PREFIX}-pre-label-access-lineage-audit-v1",
        "run_id": config.run_id,
        "candidate_name": protocol["candidate_name"],
        **support_lineage_audit,
        **future_holdout_lineage_audit,
        "role_assignment_manifest": _descriptor(role_manifest_path),
        "role_assignment_selected_rows": selected_descriptor,
        "execution_compatible_feature_coverage_report": (
            compatibility_descriptor
        ),
        "target_market_count": TARGET_MARKET_COUNT,
        "role_market_counts": dict(ROLE_MARKET_COUNTS),
        "selected_market_count": len(selected_market_ids),
        "role_assignment_completed_before_label_access": True,
        "execution_compatibility_validated_before_label_access": True,
        "label_artifacts_opened_before_pre_label_audit": False,
        "settlement_outcomes_opened_before_pre_label_audit": False,
        "pnl_fields_opened_before_pre_label_audit": False,
        "prediction_attempted_before_pre_label_audit": False,
        "pre_label_access_validation_passed": True,
        **_blocked_safety_fields(),
    }
    pre_label_audit_path = run_dir / "pre_label_access_lineage_audit.json"
    _write_json(pre_label_audit_path, pre_label_audit)
    _write_text(
        run_dir / "pre_label_access_lineage_audit.md",
        _pre_label_access_markdown(pre_label_audit),
    )
    feature_columns = tuple(feature_contract["feature_columns"])
    development_roles = ("development_train", "development_calibration")
    action_rows_by_role, corpus_audits = _materialize_role_action_rows(
        role_rows,
        feature_columns=feature_columns,
        roles=development_roles,
    )
    action_row_paths: dict[str, Path] = {}
    for role, rows in action_rows_by_role.items():
        path = run_dir / f"pairwise_action_advantage_lcb_{role}_action_rows.jsonl"
        _write_jsonl(path, rows)
        action_row_paths[role] = path

    cross_fit = _cross_fit_training_predictions(
        action_rows_by_role["development_train"],
        feature_columns=feature_columns,
        model_protocol=dict(protocol["cross_fit_protocol"]),
    )
    oof_predictions = list(cross_fit.pop("oof_predictions"))
    oof_path = run_dir / "pairwise_action_advantage_lcb_train_oof_predictions.jsonl"
    _write_jsonl(oof_path, oof_predictions)
    train_rows = action_rows_by_role["development_train"]
    final_booster = _train_pairwise_ranker(
        train_rows,
        feature_columns=feature_columns,
        model_protocol=_xgb_model_protocol(dict(protocol["cross_fit_protocol"])),
    )
    model_path = run_dir / RANK_MODEL_FILENAME
    final_booster.save_model(model_path)

    calibration_predictions = _predict_role_rows(
        action_rows_by_role["development_calibration"],
        booster=final_booster,
        feature_columns=feature_columns,
    )
    lcb_artifact = _action_advantage_lcb_artifact(
        calibration_predictions,
        train_oof_predictions=oof_predictions,
        protocol=protocol,
        feature_contract_sha256=config.expected_feature_contract_sha256,
    )
    lcb_path = run_dir / "action_advantage_lcb_calibration_artifact.json"
    _write_json(lcb_path, lcb_artifact)
    calibration_report = {
        "schema_version": f"{SCHEMA_PREFIX}-calibration-report-v1",
        "run_id": config.run_id,
        "candidate_name": protocol["candidate_name"],
        "source_split": "development_calibration_only",
        "method": lcb_artifact["method"],
        "estimand": lcb_artifact["estimand"],
        "train_oof_predictions": _descriptor(oof_path),
        "actions": lcb_artifact["actions"],
        "calibration_groups": lcb_artifact["calibration_groups"],
        "calibration_artifact": _descriptor(lcb_path),
        "confirmatory_labels_opened_before_calibration_freeze": False,
        "uses_confirmatory_validation_labels_for_tuning": False,
        "uses_prior_or_future_evidence_for_tuning": False,
        **_blocked_safety_fields(),
    }
    calibration_report_path = run_dir / "action_advantage_lcb_calibration_report.json"
    _write_json(calibration_report_path, calibration_report)
    _write_text(
        run_dir / "action_advantage_lcb_calibration_report.md",
        _calibration_markdown(calibration_report),
    )

    development_fit_freeze = {
        "schema_version": f"{SCHEMA_PREFIX}-development-fit-freeze-v1",
        "run_id": config.run_id,
        "candidate_name": protocol["candidate_name"],
        "protocol": protocol_descriptor,
        "support_gate_manifest": _descriptor(support_manifest_path),
        "future_holdout_pre_registration_manifest": _descriptor(
            future_holdout_pre_registration_path
        ),
        "pre_label_access_lineage_audit": _descriptor(pre_label_audit_path),
        "feature_contract": _descriptor(feature_contract_path),
        "role_assignment_manifest": _descriptor(role_manifest_path),
        "development_action_rows": {
            role: _descriptor(action_row_paths[role]) for role in development_roles
        },
        "train_oof_predictions": _descriptor(oof_path),
        "model": _descriptor(model_path),
        "action_advantage_lcb_calibration_artifact": _descriptor(lcb_path),
        "action_advantage_lcb_calibration_report": _descriptor(calibration_report_path),
        "confirmatory_labels_opened_before_this_freeze": False,
        "uses_confirmatory_validation_labels_for_tuning": False,
        "uses_prior_or_future_evidence_for_tuning": False,
        **_blocked_safety_fields(),
    }
    development_fit_freeze["development_fit_freeze_id"] = canonical_json_sha256(
        development_fit_freeze
    )
    development_fit_freeze_path = (
        run_dir / "pairwise_action_advantage_lcb_development_fit_freeze_manifest.json"
    )
    _write_json(development_fit_freeze_path, development_fit_freeze)

    development_predictions = _apply_action_advantage_lcb_scores(
        calibration_predictions,
        lcb_artifact=lcb_artifact,
    )
    development_prediction_path = (
        run_dir / "pairwise_action_advantage_lcb_development_calibration_predictions.jsonl"
    )
    _write_jsonl(development_prediction_path, development_predictions)
    entry_threshold = float(protocol["frozen_execution_contract"]["entry_edge_threshold"])
    runner_up_threshold = float(
        protocol["frozen_execution_contract"]["runner_up_advantage_threshold"]
    )
    development_candidate_replay = _run_policy_replay(
        development_predictions,
        score_field="action_advantage_lcb_net_return",
        policy_name=CANDIDATE_NAME,
        entry_threshold=entry_threshold,
        runner_up_advantage_threshold=runner_up_threshold,
    )
    development_baseline_replay = _run_policy_replay(
        development_predictions,
        score_field="calibrated_action_expected_net_return",
        policy_name="uncertainty_unadjusted_pairwise_ranker_same_threshold_and_guard",
        entry_threshold=entry_threshold,
        runner_up_advantage_threshold=runner_up_threshold,
    )
    development_candidate_path = (
        run_dir / "pairwise_action_advantage_lcb_development_candidate_replay.jsonl"
    )
    development_baseline_path = (
        run_dir / "pairwise_action_advantage_lcb_development_baseline_replay.jsonl"
    )
    _write_jsonl(development_candidate_path, development_candidate_replay)
    _write_jsonl(development_baseline_path, development_baseline_replay)
    development_candidate_metrics = _accepted_bet_metrics(development_candidate_replay)
    development_baseline_metrics = _accepted_bet_metrics(development_baseline_replay)
    development_robustness = _market_robustness(
        development_candidate_replay,
        development_baseline_replay,
    )
    development_gate = _development_freeze_gate(
        protocol=protocol,
        action_rows=action_rows_by_role["development_calibration"],
        candidate_replay=development_candidate_replay,
        candidate_metrics=development_candidate_metrics,
        baseline_metrics=development_baseline_metrics,
        robustness=development_robustness,
    )
    development_gate_report = {
        "schema_version": f"{SCHEMA_PREFIX}-development-freeze-gate-report-v1",
        "run_id": config.run_id,
        "candidate_name": protocol["candidate_name"],
        "candidate_metrics": development_candidate_metrics,
        "baseline_metrics": development_baseline_metrics,
        "candidate_minus_baseline_net_pnl": (
            development_candidate_metrics["net_pnl_sum"]
            - development_baseline_metrics["net_pnl_sum"]
        ),
        "market_robustness_diagnostics": development_robustness,
        "development_freeze_gate_checks": development_gate["checks"],
        "development_freeze_gate_passed": development_gate["passed"],
        "development_freeze_blocking_reason_codes": development_gate["reason_codes"],
        "confirmatory_labels_opened": False,
        "uses_confirmatory_validation_labels_for_tuning": False,
        "uses_issue174_confirmatory_labels_for_tuning": False,
        **_blocked_safety_fields(),
    }
    development_gate_report_path = run_dir / "development_freeze_gate_report.json"
    _write_json(development_gate_report_path, development_gate_report)
    development_fit_freeze.update(
        {
            "development_predictions": _descriptor(development_prediction_path),
            "development_candidate_replay": _descriptor(development_candidate_path),
            "development_baseline_replay": _descriptor(development_baseline_path),
            "development_freeze_gate_report": _descriptor(development_gate_report_path),
            "development_freeze_gate_passed": development_gate["passed"],
            "development_freeze_blocking_reason_codes": development_gate["reason_codes"],
            "confirmatory_labels_opened_before_this_freeze": False,
        }
    )
    development_fit_freeze["development_fit_freeze_id"] = canonical_json_sha256(
        {
            key: value
            for key, value in development_fit_freeze.items()
            if key != "development_fit_freeze_id"
        }
    )
    _write_json(development_fit_freeze_path, development_fit_freeze)
    training_report_path = run_dir / "cross_fit_training_report.json"
    training_report = {
        "schema_version": f"{SCHEMA_PREFIX}-training-report-v1",
        "run_id": config.run_id,
        "candidate_name": protocol["candidate_name"],
        "support_gate_manifest": _descriptor(support_manifest_path),
        "future_holdout_pre_registration_manifest": _descriptor(
            future_holdout_pre_registration_path
        ),
        "pre_label_access_lineage_audit": _descriptor(pre_label_audit_path),
        "feature_columns": list(feature_columns),
        "cross_fit": cross_fit,
        "model": _descriptor(model_path),
        "training_market_count": ROLE_MARKET_COUNTS["development_train"],
        "calibration_market_count": ROLE_MARKET_COUNTS["development_calibration"],
        "complete_five_action_decision_grid_required": True,
        "development_fit_freeze_manifest": _descriptor(development_fit_freeze_path),
        "confirmatory_labels_opened_before_model_and_lcb_freeze": False,
        "uses_confirmatory_validation_labels_for_tuning": False,
        "uses_issue174_confirmatory_labels_for_tuning": False,
        "uses_prior_or_future_evidence_for_tuning": False,
        **_blocked_safety_fields(),
    }
    _write_json(training_report_path, training_report)
    _write_text(
        run_dir / "cross_fit_training_report.md",
        _training_markdown(training_report),
    )
    development_forbidden_violation_count = sum(
        int(
            row["target_used_as_decision_input"] is not False
            or row["outcome_fields_used_as_decision_input"] is not False
        )
        for role in ("development_train", "development_calibration")
        for row in action_rows_by_role[role]
    )
    development_leakage_audit = {
        "schema_version": f"{SCHEMA_PREFIX}-development-leakage-role-audit-v1",
        "run_id": config.run_id,
        "candidate_name": protocol["candidate_name"],
        "feature_causality_violation_count": sum(
            int(audit["feature_causality_violation_count"]) for audit in corpus_audits
        ),
        "forbidden_inference_field_violation_count": (development_forbidden_violation_count),
        "complete_decision_group_count": sum(
            len(action_rows_by_role[role]) // len(REQUIRED_ACTIONS)
            for role in ("development_train", "development_calibration")
        ),
        "incomplete_decision_group_count": 0,
        "issue174_confirmatory_artifacts_opened_for_tuning": False,
        "uses_issue174_confirmatory_labels_for_tuning": False,
        "confirmatory_labels_opened": False,
        "leakage_and_role_audit_passed": (
            development_forbidden_violation_count == 0
            and all(int(audit["feature_causality_violation_count"]) == 0 for audit in corpus_audits)
        ),
        **_blocked_safety_fields(),
    }
    development_leakage_audit_path = run_dir / "development_leakage_and_role_audit.json"
    _write_json(development_leakage_audit_path, development_leakage_audit)

    if not development_gate["passed"]:
        blocked_manifest = {
            "schema_version": f"{SCHEMA_PREFIX}-candidate-freeze-manifest-v1",
            "run_id": config.run_id,
            "candidate_name": protocol["candidate_name"],
            "protocol": protocol_descriptor,
            "support_gate_manifest": _descriptor(support_manifest_path),
            "future_holdout_pre_registration_manifest": _descriptor(
                future_holdout_pre_registration_path
            ),
            "pre_label_access_lineage_audit": _descriptor(pre_label_audit_path),
            "feature_contract": _descriptor(feature_contract_path),
            "role_assignment_manifest": _descriptor(role_manifest_path),
            "model": _descriptor(model_path),
            "action_advantage_lcb_calibration_artifact": _descriptor(lcb_path),
            "development_fit_freeze_manifest": _descriptor(development_fit_freeze_path),
            "development_freeze_gate_report": _descriptor(development_gate_report_path),
            "training_report": _descriptor(training_report_path),
            "development_leakage_and_role_audit": _descriptor(development_leakage_audit_path),
            "development_freeze_gate_passed": False,
            "development_freeze_blocking_reason_codes": development_gate["reason_codes"],
            "confirmatory_evaluation_started": False,
            "confirmatory_labels_opened": False,
            "candidate_frozen_for_future_evaluation": False,
            "candidate_agnostic_future_raw_collection_allowed": True,
            "candidate_specific_future_evaluation_allowed": False,
            "uses_confirmatory_validation_labels_for_tuning": False,
            "uses_issue174_confirmatory_labels_for_tuning": False,
            **_blocked_safety_fields(),
        }
        blocked_manifest["research_candidate_hash"] = canonical_json_sha256(blocked_manifest)
        blocked_manifest_path = run_dir / "candidate_freeze_manifest.json"
        _write_json(blocked_manifest_path, blocked_manifest)
        return {
            "run_dir": run_dir,
            "pre_label_access_lineage_audit_path": pre_label_audit_path,
            "development_fit_freeze_manifest_path": development_fit_freeze_path,
            "development_gate_report_path": development_gate_report_path,
            "freeze_manifest_path": blocked_manifest_path,
            "freeze_manifest_sha256": _sha256_file(blocked_manifest_path),
            "validation_report": development_gate_report,
            "freeze_manifest": blocked_manifest,
        }

    confirmatory_rows, confirmatory_audits = _materialize_role_action_rows(
        role_rows,
        feature_columns=feature_columns,
        roles=("confirmatory_validation",),
    )
    action_rows_by_role.update(confirmatory_rows)
    corpus_audits.extend(confirmatory_audits)
    confirmatory_action_path = (
        run_dir / "pairwise_action_advantage_lcb_confirmatory_validation_action_rows.jsonl"
    )
    _write_jsonl(
        confirmatory_action_path,
        action_rows_by_role["confirmatory_validation"],
    )
    action_row_paths["confirmatory_validation"] = confirmatory_action_path
    split_manifest = _split_manifest(
        run_id=config.run_id,
        role_manifest_path=role_manifest_path,
        protocol_descriptor=protocol_descriptor,
        feature_contract_path=feature_contract_path,
        action_rows_by_role=action_rows_by_role,
        action_row_paths=action_row_paths,
        corpus_audits=corpus_audits,
    )
    split_manifest_path = run_dir / "pairwise_action_advantage_lcb_split_manifest.json"
    _write_json(split_manifest_path, split_manifest)
    forbidden_inference_field_violation_count = sum(
        int(
            row["target_used_as_decision_input"] is not False
            or row["outcome_fields_used_as_decision_input"] is not False
        )
        for rows in action_rows_by_role.values()
        for row in rows
    )
    leakage_and_role_audit = {
        "schema_version": f"{SCHEMA_PREFIX}-leakage-role-audit-v1",
        "run_id": config.run_id,
        "candidate_name": protocol["candidate_name"],
        "role_market_counts": {
            role: split_manifest["roles"][role]["market_count"] for role in ROLE_NAMES
        },
        "role_market_overlap_count": split_manifest["role_market_overlap_count"],
        "prior_market_overlap_count": len(prior_market_overlap),
        "chronology_validation_passed": split_manifest["chronology_validation_passed"],
        "feature_causality_violation_count": split_manifest["feature_causality_violation_count"],
        "forbidden_inference_field_violation_count": (forbidden_inference_field_violation_count),
        "role_assignment_completed_before_label_access": True,
        "execution_compatibility_validated_before_label_access": True,
        "execution_compatible_feature_coverage_report": compatibility_descriptor,
        "development_fit_frozen_before_confirmatory_label_access": True,
        "confirmatory_labels_used_for_tuning": False,
        "prior_or_future_evidence_used_for_tuning": False,
        "leakage_and_role_audit_passed": (
            split_manifest["role_market_overlap_count"] == 0
            and not prior_market_overlap
            and split_manifest["chronology_validation_passed"] is True
            and split_manifest["feature_causality_violation_count"] == 0
            and forbidden_inference_field_violation_count == 0
            and compatibility_report["selected_market_failure_count"] == 0
        ),
        **_blocked_safety_fields(),
    }
    leakage_audit_path = run_dir / "leakage_and_role_audit.json"
    _write_json(leakage_audit_path, leakage_and_role_audit)
    _write_text(
        run_dir / "leakage_and_role_audit.md",
        _leakage_markdown(leakage_and_role_audit),
    )

    confirmatory_predictions = _predict_role_rows(
        action_rows_by_role["confirmatory_validation"],
        booster=final_booster,
        feature_columns=feature_columns,
    )
    confirmatory_predictions = _apply_action_advantage_lcb_scores(
        confirmatory_predictions,
        lcb_artifact=lcb_artifact,
    )
    prediction_path = run_dir / "pairwise_action_advantage_lcb_confirmatory_predictions.jsonl"
    _write_jsonl(prediction_path, confirmatory_predictions)
    entry_threshold = float(protocol["frozen_execution_contract"]["entry_edge_threshold"])
    candidate_replay = _run_policy_replay(
        confirmatory_predictions,
        score_field="action_advantage_lcb_net_return",
        policy_name=CANDIDATE_NAME,
        entry_threshold=entry_threshold,
        runner_up_advantage_threshold=float(
            protocol["frozen_execution_contract"]["runner_up_advantage_threshold"]
        ),
    )
    baseline_replay = _run_policy_replay(
        confirmatory_predictions,
        score_field="calibrated_action_expected_net_return",
        policy_name="uncertainty_unadjusted_pairwise_ranker_same_threshold_and_guard",
        entry_threshold=entry_threshold,
        runner_up_advantage_threshold=0.0,
    )
    candidate_path = run_dir / "pairwise_action_advantage_lcb_confirmatory_replay.jsonl"
    baseline_path = run_dir / "pairwise_action_advantage_lcb_confirmatory_baseline_replay.jsonl"
    _write_jsonl(candidate_path, candidate_replay)
    _write_jsonl(baseline_path, baseline_replay)
    candidate_metrics = _accepted_bet_metrics(candidate_replay)
    baseline_metrics = _accepted_bet_metrics(baseline_replay)
    robustness = _market_robustness(candidate_replay, baseline_replay)
    confirmatory_gate = _confirmatory_gate(
        protocol=protocol,
        action_rows=action_rows_by_role["confirmatory_validation"],
        candidate_replay=candidate_replay,
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
        robustness=robustness,
    )

    training_report = {
        "schema_version": f"{SCHEMA_PREFIX}-training-report-v1",
        "run_id": config.run_id,
        "candidate_name": protocol["candidate_name"],
        "feature_columns": list(feature_columns),
        "cross_fit": cross_fit,
        "model": _descriptor(model_path),
        "training_market_count": ROLE_MARKET_COUNTS["development_train"],
        "calibration_market_count": ROLE_MARKET_COUNTS["development_calibration"],
        "complete_five_action_decision_grid_required": True,
        "development_freeze_gate_report": _descriptor(development_gate_report_path),
        "development_freeze_gate_passed": development_gate["passed"],
        "development_fit_freeze_manifest": _descriptor(development_fit_freeze_path),
        "confirmatory_labels_opened_before_model_and_lcb_freeze": False,
        "uses_confirmatory_validation_labels_for_tuning": False,
        "uses_issue174_confirmatory_labels_for_tuning": False,
        "uses_prior_or_future_evidence_for_tuning": False,
        **_blocked_safety_fields(),
    }
    training_report_path = run_dir / "cross_fit_training_report.json"
    _write_json(training_report_path, training_report)
    _write_text(
        run_dir / "cross_fit_training_report.md",
        _training_markdown(training_report),
    )
    validation_report = {
        "schema_version": f"{SCHEMA_PREFIX}-confirmatory-validation-report-v1",
        "run_id": config.run_id,
        "candidate_name": protocol["candidate_name"],
        "candidate_metrics": candidate_metrics,
        "baseline_metrics": baseline_metrics,
        "candidate_minus_baseline_net_pnl": candidate_metrics["net_pnl_sum"]
        - baseline_metrics["net_pnl_sum"],
        "market_robustness_diagnostics": robustness,
        "confirmatory_gate_checks": confirmatory_gate["checks"],
        "confirmatory_gate_passed": confirmatory_gate["passed"],
        "confirmatory_gate_blocking_reason_codes": confirmatory_gate["reason_codes"],
        "confirmatory_labels_used_for_report_only": True,
        "confirmatory_labels_used_for_tuning": False,
        "candidate_frozen_for_future_evaluation": confirmatory_gate["passed"],
        "candidate_agnostic_future_raw_collection_allowed": True,
        "candidate_specific_future_evaluation_allowed": confirmatory_gate["passed"],
        **_blocked_safety_fields(),
    }
    validation_report_path = run_dir / "confirmatory_validation_report.json"
    _write_json(validation_report_path, validation_report)
    _write_text(
        run_dir / "confirmatory_validation_report.md",
        _validation_markdown(validation_report),
    )
    accepted_bet_pnl_report = {
        "schema_version": f"{SCHEMA_PREFIX}-confirmatory-accepted-bet-pnl-report-v1",
        "run_id": config.run_id,
        "candidate_name": protocol["candidate_name"],
        "evaluation_scope": (
            "frozen_execution_guard_accepted_bets_after_full_costs"
        ),
        "candidate_metrics": candidate_metrics,
        "candidate_accepted_bet_diagnostics": _accepted_bet_diagnostics(
            candidate_replay
        ),
        "baseline_metrics": baseline_metrics,
        "baseline_accepted_bet_diagnostics": _accepted_bet_diagnostics(
            baseline_replay
        ),
        "candidate_minus_baseline_net_pnl": (
            candidate_metrics["net_pnl_sum"] - baseline_metrics["net_pnl_sum"]
        ),
        "market_robustness_diagnostics": robustness,
        "confirmatory_gate_passed": confirmatory_gate["passed"],
        "confirmatory_gate_blocking_reason_codes": confirmatory_gate[
            "reason_codes"
        ],
        "full_cost_aware_target_used_for_report_only": True,
        "confirmatory_labels_used_for_tuning": False,
        "execution_guard_mutated": False,
        "order_sizing_mutated": False,
        "cost_model_mutated": False,
        "future_unseen_holdout_required": True,
        **_blocked_safety_fields(),
    }
    accepted_bet_pnl_report_path = (
        run_dir / "confirmatory_accepted_bet_pnl_report.json"
    )
    _write_json(accepted_bet_pnl_report_path, accepted_bet_pnl_report)
    _write_text(
        run_dir / "confirmatory_accepted_bet_pnl_report.md",
        _accepted_bet_pnl_markdown(accepted_bet_pnl_report),
    )

    freeze_manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-candidate-freeze-manifest-v1",
        "run_id": config.run_id,
        "candidate_name": protocol["candidate_name"],
        "protocol": protocol_descriptor,
        "support_gate_manifest": _descriptor(support_manifest_path),
        "future_holdout_pre_registration_manifest": _descriptor(
            future_holdout_pre_registration_path
        ),
        "pre_label_access_lineage_audit": _descriptor(pre_label_audit_path),
        "feature_contract": _descriptor(feature_contract_path),
        "role_assignment_manifest": _descriptor(role_manifest_path),
        "split_manifest": _descriptor(split_manifest_path),
        "model": _descriptor(model_path),
        "action_advantage_lcb_calibration_artifact": _descriptor(lcb_path),
        "action_advantage_lcb_calibration_report": _descriptor(calibration_report_path),
        "development_fit_freeze_manifest": _descriptor(development_fit_freeze_path),
        "development_freeze_gate_report": _descriptor(development_gate_report_path),
        "development_freeze_gate_passed": development_gate["passed"],
        "training_report": _descriptor(training_report_path),
        "leakage_and_role_audit": _descriptor(leakage_audit_path),
        "confirmatory_validation_report": _descriptor(validation_report_path),
        "confirmatory_accepted_bet_pnl_report": _descriptor(
            accepted_bet_pnl_report_path
        ),
        "candidate_frozen_for_future_evaluation": confirmatory_gate["passed"],
        "candidate_agnostic_future_raw_collection_allowed": True,
        "candidate_specific_future_evaluation_allowed": confirmatory_gate["passed"],
        "future_unseen_evaluation_required": True,
        "confirmatory_gate_passed": confirmatory_gate["passed"],
        "confirmatory_gate_blocking_reason_codes": confirmatory_gate["reason_codes"],
        "uses_confirmatory_validation_labels_for_tuning": False,
        "uses_issue174_confirmatory_labels_for_tuning": False,
        "uses_prior_or_future_evidence_for_tuning": False,
        **_blocked_safety_fields(),
    }
    freeze_manifest["research_candidate_hash"] = canonical_json_sha256(
        {
            "support_gate": freeze_manifest["support_gate_manifest"]["sha256"],
            "future_holdout_pre_registration": freeze_manifest[
                "future_holdout_pre_registration_manifest"
            ]["sha256"],
            "protocol": protocol_descriptor["sha256"],
            "feature_contract": freeze_manifest["feature_contract"]["sha256"],
            "role_assignment": freeze_manifest["role_assignment_manifest"]["sha256"],
            "split": freeze_manifest["split_manifest"]["sha256"],
            "model": freeze_manifest["model"]["sha256"],
            "lcb": freeze_manifest["action_advantage_lcb_calibration_artifact"]["sha256"],
        }
    )
    freeze_manifest_path = run_dir / "candidate_freeze_manifest.json"
    _write_json(freeze_manifest_path, freeze_manifest)
    return {
        "run_dir": run_dir,
        "pre_label_access_lineage_audit_path": pre_label_audit_path,
        "split_manifest_path": split_manifest_path,
        "development_fit_freeze_manifest_path": development_fit_freeze_path,
        "training_report_path": training_report_path,
        "calibration_report_path": calibration_report_path,
        "leakage_audit_path": leakage_audit_path,
        "validation_report_path": validation_report_path,
        "accepted_bet_pnl_report_path": accepted_bet_pnl_report_path,
        "freeze_manifest_path": freeze_manifest_path,
        "freeze_manifest_sha256": _sha256_file(freeze_manifest_path),
        "validation_report": validation_report,
        "freeze_manifest": freeze_manifest,
    }


def _validate_role_rows(rows: list[dict[str, Any]]) -> None:
    if len(rows) != TARGET_MARKET_COUNT:
        raise ValueError(
            f"role assignment must contain exactly {TARGET_MARKET_COUNT} markets"
        )
    market_ids = [str(row.get("market_id") or "") for row in rows]
    if any(not value for value in market_ids) or len(market_ids) != len(set(market_ids)):
        raise ValueError("role assignment market identities are incomplete")
    if [int(row.get("selection_rank") or 0) for row in rows] != list(
        range(1, TARGET_MARKET_COUNT + 1)
    ):
        raise ValueError("role assignment selection ranks are incomplete")
    counts = Counter(str(row.get("role") or "") for row in rows)
    if dict(counts) != ROLE_MARKET_COUNTS:
        raise ValueError("role assignment market counts do not match 90/45/60")
    expected_roles = [
        (
            "development_train"
            if rank <= ROLE_MARKET_COUNTS["development_train"]
            else "development_calibration"
            if rank
            <= ROLE_MARKET_COUNTS["development_train"]
            + ROLE_MARKET_COUNTS["development_calibration"]
            else "confirmatory_validation"
        )
        for rank in range(1, TARGET_MARKET_COUNT + 1)
    ]
    if [str(row.get("role") or "") for row in rows] != expected_roles:
        raise ValueError("role assignment does not follow frozen rank boundaries")
    minimum_decision_timestamps = [
        int(row.get("minimum_decision_ts") or 0) for row in rows
    ]
    if any(value <= 0 for value in minimum_decision_timestamps) or any(
        current <= previous
        for previous, current in zip(
            minimum_decision_timestamps,
            minimum_decision_timestamps[1:],
            strict=False,
        )
    ):
        raise ValueError("role assignment chronology is incomplete or non-increasing")
    if any(
        int(row.get("maximum_decision_ts") or 0)
        < int(row.get("minimum_decision_ts") or 0)
        for row in rows
    ):
        raise ValueError("role assignment decision window is invalid")
    if any(
        row.get("execution_compatibility_validated_before_label_access") is not True for row in rows
    ):
        raise ValueError("role assignment did not validate execution compatibility")
    if any(row.get("labels_or_outcomes_opened_for_role_assignment") is not False for row in rows):
        raise ValueError("role assignment opened labels or outcomes")


def _validate_execution_compatibility_report(report: dict[str, Any]) -> None:
    if (
        report.get("execution_compatibility_validated_before_label_access") is not True
        or int(report.get("selected_market_failure_count", -1)) != 0
        or int(report.get("selected_market_count") or 0) != TARGET_MARKET_COUNT
        or report.get("labels_or_outcomes_opened") is not False
    ):
        raise ValueError("execution compatibility did not pass before label access")


def _validate_support_gate_lineage(
    *,
    support_manifest: dict[str, Any],
    support_manifest_path: Path,
    role_manifest: dict[str, Any],
    role_manifest_path: Path,
    expected_role_manifest_sha256: str,
) -> dict[str, Any]:
    """Prove #188 support and role lineage before any target file is opened."""

    support_report_descriptor = _verified_descriptor(
        support_manifest.get("report"),
        name="supplemental support gate report",
    )
    support_report = _load_json(Path(support_report_descriptor["path"]))
    core_manifest_descriptor = _verified_descriptor(
        support_manifest.get("core_support_gate_manifest"),
        name="core support gate manifest",
    )
    core_manifest = _load_json(Path(core_manifest_descriptor["path"]))
    core_report_descriptor = _verified_descriptor(
        core_manifest.get("support_gate_report"),
        name="core support gate report",
    )
    core_report = _load_json(Path(core_report_descriptor["path"]))
    linked_role_descriptor = _verified_descriptor(
        core_manifest.get("role_assignment_manifest"),
        name="support-linked role assignment manifest",
    )
    role_report_descriptor = _verified_descriptor(
        role_manifest.get("report"),
        name="role assignment report",
    )
    role_report = _load_json(Path(role_report_descriptor["path"]))

    blockers: list[str] = []
    if support_manifest.get("schema_version") != (
        SUPPLEMENTAL_SUPPORT_MANIFEST_SCHEMA_VERSION
    ):
        blockers.append("supplemental_support_manifest_schema_invalid")
    if support_report.get("schema_version") != (
        SUPPLEMENTAL_SUPPORT_REPORT_SCHEMA_VERSION
    ):
        blockers.append("supplemental_support_report_schema_invalid")
    if core_manifest.get("schema_version") != CORE_SUPPORT_MANIFEST_SCHEMA_VERSION:
        blockers.append("core_support_manifest_schema_invalid")
    if core_report.get("schema_version") != CORE_SUPPORT_REPORT_SCHEMA_VERSION:
        blockers.append("core_support_report_schema_invalid")
    if support_manifest.get("supplemental_support_target_ready") is not True:
        blockers.append("supplemental_support_target_not_ready")
    if support_manifest.get("blocking_reason_codes") not in ([], None):
        blockers.append("supplemental_support_manifest_has_blockers")
    if int(support_manifest.get("selected_market_count") or 0) != TARGET_MARKET_COUNT:
        blockers.append("supplemental_support_market_count_mismatch")
    if dict(support_manifest.get("role_market_counts") or {}) != ROLE_MARKET_COUNTS:
        blockers.append("supplemental_support_role_counts_mismatch")
    if support_manifest.get("continuation_allowed") is not False:
        blockers.append("supplemental_support_continuation_not_closed")
    if support_manifest.get("labels_or_outcomes_opened_for_support_gate") is not False:
        blockers.append("supplemental_support_opened_labels_or_outcomes")
    if support_report.get("status") != (
        "OUTCOME_BLIND_SUPPLEMENTAL_SUPPORT_TARGET_READY"
    ):
        blockers.append("supplemental_support_report_not_ready")
    if support_report.get("supplemental_support_target_ready") is not True:
        blockers.append("supplemental_support_report_target_not_ready")
    if support_report.get("blocking_reason_codes") not in ([], None):
        blockers.append("supplemental_support_report_has_blockers")
    if int(support_report.get("selected_market_count") or 0) != TARGET_MARKET_COUNT:
        blockers.append("supplemental_support_report_market_count_mismatch")
    if dict(support_report.get("role_market_counts") or {}) != ROLE_MARKET_COUNTS:
        blockers.append("supplemental_support_report_role_counts_mismatch")
    if core_report.get("status") != "OUTCOME_BLIND_SUPPORT_TARGET_READY":
        blockers.append("core_support_gate_not_ready")
    if int(core_report.get("selected_market_count") or 0) != TARGET_MARKET_COUNT:
        blockers.append("core_support_market_count_mismatch")
    if core_report.get("blocking_reason_codes") not in ([], None):
        blockers.append("core_support_gate_has_blockers")
    if core_report.get("continuation_allowed") is not False:
        blockers.append("core_support_continuation_not_closed")
    if core_report.get("labels_or_outcomes_opened_for_continuation") is not False:
        blockers.append("core_support_opened_labels_or_outcomes")
    if (
        Path(linked_role_descriptor["path"]).resolve()
        != role_manifest_path.resolve()
        or linked_role_descriptor["sha256"]
        != expected_role_manifest_sha256.lower()
    ):
        blockers.append("support_gate_role_manifest_lineage_mismatch")
    if role_manifest.get("role_assignment_ready") is not True:
        blockers.append("role_assignment_not_ready")
    if role_manifest.get("blocking_reason_codes") not in ([], None):
        blockers.append("role_assignment_has_blockers")
    if dict(role_manifest.get("role_market_counts") or {}) != ROLE_MARKET_COUNTS:
        blockers.append("role_assignment_role_counts_mismatch")
    if role_manifest.get("labels_or_outcomes_opened_for_role_assignment") is not False:
        blockers.append("role_assignment_opened_labels_or_outcomes")
    if role_report.get("role_assignment_ready") is not True:
        blockers.append("role_assignment_report_not_ready")
    if role_report.get("blocking_reason_codes") not in ([], None):
        blockers.append("role_assignment_report_has_blockers")
    if int(role_report.get("selected_market_count") or 0) != TARGET_MARKET_COUNT:
        blockers.append("role_assignment_report_market_count_mismatch")
    if dict(role_report.get("role_market_counts") or {}) != ROLE_MARKET_COUNTS:
        blockers.append("role_assignment_report_role_counts_mismatch")
    if int(role_report.get("role_market_overlap_count") or 0) != 0:
        blockers.append("role_assignment_report_role_overlap")
    if int(role_report.get("prior_market_overlap_count") or 0) != 0:
        blockers.append("role_assignment_report_prior_overlap")
    if role_report.get("labels_or_outcomes_opened_for_role_assignment") is not False:
        blockers.append("role_assignment_report_opened_labels_or_outcomes")
    if (
        role_report.get("execution_compatibility_validated_before_label_access")
        is not True
    ):
        blockers.append("role_assignment_report_execution_compatibility_invalid")
    for name, payload in (
        ("supplemental_support_manifest", support_manifest),
        ("supplemental_support_report", support_report),
        ("core_support_manifest", core_manifest),
        ("core_support_report", core_report),
        ("role_assignment_manifest", role_manifest),
        ("role_assignment_report", role_report),
    ):
        if not _safety_contract_blocked(payload):
            blockers.append(f"{name}_safety_contract_failed")
    if blockers:
        raise ValueError(
            "pre-label support lineage validation failed: "
            + ", ".join(sorted(set(blockers)))
        )
    return {
        "support_gate_manifest": _descriptor(support_manifest_path),
        "supplemental_support_gate_report": support_report_descriptor,
        "core_support_gate_manifest": core_manifest_descriptor,
        "core_support_gate_report": core_report_descriptor,
        "support_gate_role_assignment_manifest": linked_role_descriptor,
        "role_assignment_report": role_report_descriptor,
        "support_gate_lineage_hash_verified": True,
        "supplemental_support_target_ready": True,
        "support_gate_blocking_reason_codes": [],
        "labels_or_outcomes_opened_for_support_or_role_assignment": False,
    }


def _safety_contract_blocked(payload: dict[str, Any]) -> bool:
    return (
        payload.get("paper_only") is True
        and payload.get("capital_at_risk") is False
        and payload.get("polymarket_write_enabled", False) is False
        and payload.get("wallet_signing_enabled", False) is False
        and payload.get("source_model_candidate_eligible", False) is False
        and payload.get("freeze_ready", False) is False
        and payload.get("promotion_evidence_eligible", False) is False
        and payload.get("v8_execution_handoff_allowed", False) is False
        and payload.get("#134_resume_allowed", False) is False
        and payload.get("#146_start_allowed", False) is False
    )


def _materialize_role_action_rows(
    role_rows: list[dict[str, Any]],
    *,
    feature_columns: tuple[str, ...],
    roles: tuple[str, ...],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    if not roles or any(role not in ROLE_NAMES for role in roles):
        raise ValueError("requested materialization roles are invalid")
    output: dict[str, list[dict[str, Any]]] = {role: [] for role in roles}
    audits: list[dict[str, Any]] = []
    for role_row in role_rows:
        role = str(role_row["role"])
        if role not in output:
            continue
        corpus_dir = Path(str(role_row["source_corpus_dir"])).resolve()
        rows, audit = _load_corpus_action_rows(
            corpus_dir,
            role_row=role_row,
            feature_columns=feature_columns,
        )
        if audit["blocking_reason_codes"]:
            raise ValueError(
                f"corpus action-row materialization failed for {corpus_dir}: "
                + ", ".join(audit["blocking_reason_codes"])
            )
        output[role].extend(rows)
        audits.append(audit)
    for role in roles:
        output[role].sort(
            key=lambda row: (
                int(row["decision_ts"]),
                str(row["market_id"]),
                str(row["action"]),
            )
        )
        markets = {str(row["market_id"]) for row in output[role]}
        if len(markets) != ROLE_MARKET_COUNTS[role]:
            raise ValueError(f"{role} action rows have incomplete market coverage")
    return output, audits


def _load_corpus_action_rows(
    corpus_dir: Path,
    *,
    role_row: dict[str, Any],
    feature_columns: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    filenames = {
        "manifest": "polymarket_corpus_manifest.json",
        "features": "polymarket_feature_rows.jsonl",
        "labels": "polymarket_label_rows.jsonl",
        "metadata": "polymarket_market_metadata.jsonl",
        "resolutions": "polymarket_resolution_events.jsonl",
    }
    missing = sorted(
        filename for filename in filenames.values() if not (corpus_dir / filename).is_file()
    )
    if missing:
        raise ValueError(f"required corpus artifacts are missing: {missing}")
    manifest_path = corpus_dir / filenames["manifest"]
    manifest = _load_json(manifest_path)
    if _sha256_file(manifest_path) != role_row["corpus_manifest"]["sha256"]:
        raise ValueError("role-assigned corpus manifest SHA-256 mismatch")
    normalized_hashes = dict(manifest.get("normalized_artifact_hashes") or {})
    hash_keys = {
        "features": "feature_rows",
        "labels": "label_rows",
        "metadata": "market_metadata",
        "resolutions": "resolution_events",
    }
    for name, hash_key in hash_keys.items():
        path = corpus_dir / filenames[name]
        if normalized_hashes.get(hash_key) != _sha256_file(path):
            raise ValueError(f"normalized artifact SHA-256 mismatch: {path.name}")

    features = _load_jsonl(corpus_dir / filenames["features"])
    labels = _load_jsonl(corpus_dir / filenames["labels"])
    metadata_rows = _load_jsonl(corpus_dir / filenames["metadata"])
    resolution_rows = _load_jsonl(corpus_dir / filenames["resolutions"])
    market_id = str(role_row["market_id"])
    blockers: list[str] = []
    if {str(row.get("market_id") or "") for row in features} != {market_id}:
        blockers.append("feature_market_identity_mismatch")
    if {str(row.get("market_id") or "") for row in labels} != {market_id}:
        blockers.append("label_market_identity_mismatch")
    metadata_by_market = {str(row.get("market_id") or ""): row for row in metadata_rows}
    resolution_by_market = {str(row.get("market_id") or ""): row for row in resolution_rows}
    metadata = metadata_by_market.get(market_id)
    resolution = resolution_by_market.get(market_id)
    if metadata is None:
        blockers.append("market_metadata_missing")
    if resolution is None or resolution.get("resolved_outcome") not in {"UP", "DOWN"}:
        blockers.append("official_resolution_missing")
    labels_by_decision: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for label in labels:
        labels_by_decision[int(label.get("decision_ts") or 0)].append(label)
    action_rows: list[dict[str, Any]] = []
    causality_violations = 0
    cost_component_violations = 0
    incomplete_grids = 0
    for feature_row in features:
        decision_ts = int(feature_row.get("decision_ts") or 0)
        max_input_ts = int(feature_row.get("max_input_ts") or 0)
        if max_input_ts > decision_ts or decision_ts <= 0:
            causality_violations += 1
            continue
        if _find_fields(feature_row, FORBIDDEN_DECISION_FIELDS):
            causality_violations += 1
            continue
        decision_labels = labels_by_decision.get(decision_ts, [])
        label_actions = {str(row.get("action") or "") for row in decision_labels}
        if label_actions != set(REQUIRED_ACTIONS):
            incomplete_grids += 1
            continue
        label_by_action = {str(row["action"]): row for row in decision_labels}
        if any(not _cost_aware_label_valid(label_by_action[action]) for action in REQUIRED_ACTIONS):
            cost_component_violations += 1
            continue
        assert metadata is not None
        for action in REQUIRED_ACTIONS:
            action_rows.append(
                _action_row(
                    feature_row=feature_row,
                    label=label_by_action[action],
                    metadata=metadata,
                    resolution=resolution or {},
                    role=str(role_row["role"]),
                    selection_rank=int(role_row["selection_rank"]),
                    source_corpus_dir=corpus_dir,
                    source_manifest_sha256=_sha256_file(manifest_path),
                    feature_columns=feature_columns,
                )
            )
    if causality_violations:
        blockers.append("feature_timestamp_or_field_causality_violation")
    if incomplete_grids:
        blockers.append("incomplete_5_action_label_grid")
    if cost_component_violations:
        blockers.append("cost_aware_label_contract_violation")
    expected_rows = len(features) * len(REQUIRED_ACTIONS)
    if len(action_rows) != expected_rows:
        blockers.append("materialized_action_row_count_mismatch")
    return action_rows, {
        "market_id": market_id,
        "role": role_row["role"],
        "source_corpus_dir": str(corpus_dir),
        "source_corpus_manifest_sha256": _sha256_file(manifest_path),
        "feature_row_count": len(features),
        "label_row_count": len(labels),
        "materialized_action_row_count": len(action_rows),
        "feature_causality_violation_count": causality_violations,
        "incomplete_action_grid_count": incomplete_grids,
        "cost_component_violation_count": cost_component_violations,
        "blocking_reason_codes": sorted(set(blockers)),
        "role_assignment_completed_before_label_access": True,
        "outcomes_used_as_training_or_evaluation_targets_only": True,
        "outcomes_used_as_decision_inputs": False,
    }


def _cost_aware_label_valid(row: dict[str, Any]) -> bool:
    if row.get("paper_only") is not True or row.get("capital_at_risk") is not False:
        return False
    for field in ("fees", "slippage", "liquidity_impact", "total_net_pnl_per_notional"):
        value = row.get(field)
        if not isinstance(value, int | float) or not math.isfinite(float(value)):
            return False
    return all(float(row[field]) >= 0.0 for field in ("fees", "slippage", "liquidity_impact"))


def _action_row(
    *,
    feature_row: dict[str, Any],
    label: dict[str, Any],
    metadata: dict[str, Any],
    resolution: dict[str, Any],
    role: str,
    selection_rank: int,
    source_corpus_dir: Path,
    source_manifest_sha256: str,
    feature_columns: tuple[str, ...],
) -> dict[str, Any]:
    action = str(label["action"])
    side = "UP" if "BUY_UP" in action else "DOWN" if "BUY_DOWN" in action else "NONE"
    family = (
        "HOLD_TO_SETTLEMENT"
        if "HOLD_TO_SETTLEMENT" in action
        else "SELL_BEFORE_CLOSE"
        if "SELL_BEFORE_CLOSE" in action
        else "NO_TRADE"
    )
    raw = dict(feature_row.get("features") or {})
    decision_features = _decision_features(raw, action=action, side=side, family=family)
    missing = sorted(name for name in feature_columns if name not in decision_features)
    if missing:
        raise ValueError(f"decision-time features are missing: {missing}")
    values = {name: float(decision_features[name]) for name in feature_columns}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("decision-time features must be finite")
    decision_ts = int(feature_row["decision_ts"])
    max_input_ts = int(feature_row["max_input_ts"])
    feature_provenance = dict(feature_row.get("feature_provenance") or {})
    reference_provenance = dict(
        feature_provenance.get("reference_price_to_beat_distance_at_decision") or {}
    )
    reference_max_input_ts = int(reference_provenance.get("max_input_ts") or 0)
    reference_valid = bool(
        reference_provenance.get("provenance_valid") is True
        and reference_max_input_ts <= decision_ts
    )
    if action != "NO_TRADE" and not reference_valid:
        raise ValueError("reference price feature provenance is invalid")
    p_up = _p_up(raw)
    p_down = 1.0 - p_up
    selected_probability = p_up if side == "UP" else p_down if side == "DOWN" else 0.0
    side_prefix = side.lower()
    microstructure = {
        "entry_bid": float(raw.get(f"{side_prefix}_bid") or 0.0),
        "entry_ask": float(raw.get(f"{side_prefix}_ask") or 0.0),
        "spread_bps": float(raw.get(f"{side_prefix}_spread_bps") or 0.0),
        "book_staleness_ms": float(raw.get(f"{side_prefix}_book_staleness_ms") or 0.0),
        "queue_fill_proxy": float(raw.get(f"{side_prefix}_queue_fill_probability_proxy") or 0.0),
        "time_to_close_seconds": float(raw.get("time_to_close_seconds") or 0.0),
    }
    row = {
        "market_id": str(feature_row["market_id"]),
        "condition_id": str(metadata.get("condition_id") or feature_row["market_id"]),
        "market_slug": str(metadata.get("slug") or ""),
        "decision_ts": decision_ts,
        "market_close_ts": int(metadata.get("market_end_ts") or 0),
        "max_input_ts": max_input_ts,
        "role": role,
        "market_selection_rank": selection_rank,
        "action": action,
        "side": side,
        "action_family": family,
        "decision_time_features": values,
        "p_up": p_up,
        "p_down": p_down,
        "selected_side_probability": selected_probability,
        "microstructure_snapshot": microstructure,
        "reference_price_feature_provenance": {
            **reference_provenance,
            "provenance_valid": reference_valid,
        },
        "p_up_action_disagreement": bool(
            (side == "UP" and p_up < 0.5) or (side == "DOWN" and p_up > 0.5)
        ),
        "target_net_pnl_per_contract": float(label["total_net_pnl_per_notional"]),
        "target_cost_components": {
            "fees": float(label["fees"]),
            "slippage": float(label["slippage"]),
            "liquidity_impact": float(label["liquidity_impact"]),
        },
        "target_resolved_outcome": resolution.get("resolved_outcome"),
        "target_used_as_decision_input": False,
        "outcome_fields_used_as_decision_input": False,
        "source_corpus_dir": str(source_corpus_dir),
        "source_corpus_manifest_sha256": source_manifest_sha256,
        "paper_only": True,
        "capital_at_risk": False,
    }
    row["action_row_sha256"] = canonical_json_sha256(row)
    return row


def _decision_features(
    raw: dict[str, Any],
    *,
    action: str,
    side: str,
    family: str,
) -> dict[str, float]:
    p_up = _p_up(raw)
    p_down = 1.0 - p_up
    if side == "NONE":
        selected = "up"
        opposite = "down"
        selected_probability = 0.0
        execution_price = 0.0
    else:
        selected = side.lower()
        opposite = "down" if selected == "up" else "up"
        selected_probability = p_up if side == "UP" else p_down
        execution_price = float(raw.get(f"{selected}_ask") or 0.0)
    return {
        "btc_return_10s": float(raw.get("btc_return_10s") or 0.0),
        "btc_return_30s": float(raw.get("btc_return_30s") or 0.0),
        "btc_return_1m": float(raw.get("btc_return_1m") or 0.0),
        "btc_return_5m": float(raw.get("btc_return_5m") or 0.0),
        "btc_return_15m": float(raw.get("btc_return_15m") or 0.0),
        "btc_volatility_1m": float(raw.get("btc_volatility_1m") or 0.0),
        "btc_volatility_5m": float(raw.get("btc_volatility_5m") or 0.0),
        "btc_volatility_15m": float(raw.get("btc_volatility_15m") or 0.0),
        "reference_price_to_beat_distance_at_decision": float(
            raw.get("reference_price_to_beat_distance_at_decision") or 0.0
        ),
        "time_to_close_seconds": float(raw.get("time_to_close_seconds") or 0.0),
        "market_age_seconds": float(raw.get("market_age_seconds") or 0.0),
        "combined_spread_bps": float(raw.get("combined_spread_bps") or 0.0),
        "liquidity_imbalance": float(raw.get("liquidity_imbalance") or 0.0),
        "recent_selected_side_trade_volume": float(
            raw.get(f"recent_{selected}_trade_volume") or 0.0
        ),
        "recent_opposite_side_trade_volume": float(
            raw.get(f"recent_{opposite}_trade_volume") or 0.0
        ),
        "selected_side_probability": selected_probability,
        "execution_price": execution_price,
        "selected_side_probability_minus_execution_price": (selected_probability - execution_price),
        "selected_side_spread_bps": float(raw.get(f"{selected}_spread_bps") or 0.0),
        "selected_side_queue_fill_probability_proxy": float(
            raw.get(f"{selected}_queue_fill_probability_proxy") or 0.0
        ),
        "selected_side_book_staleness_ms": float(raw.get(f"{selected}_book_staleness_ms") or 0.0),
        "selected_side_liquidity_depth": float(raw.get(f"{selected}_liquidity_depth") or 0.0),
        "selected_side_executable_ask_notional": float(
            raw.get(f"{selected}_executable_ask_notional") or 0.0
        ),
        "selected_side_executable_bid_notional": float(
            raw.get(f"{selected}_executable_bid_notional") or 0.0
        ),
        "selected_side_recent_book_update_count_1m": float(
            raw.get(f"{selected}_recent_book_update_count_1m") or 0.0
        ),
        "selected_side_recent_spread_stability_1m": float(
            raw.get(f"{selected}_recent_spread_stability_1m") or 0.0
        ),
        "selected_side_recent_bid_depth_volatility_1m": float(
            raw.get(f"{selected}_recent_bid_depth_volatility_1m") or 0.0
        ),
        "action_buy_up": float("BUY_UP" in action),
        "action_buy_down": float("BUY_DOWN" in action),
        "action_hold_to_settlement": float(family == "HOLD_TO_SETTLEMENT"),
        "action_sell_before_close": float(family == "SELL_BEFORE_CLOSE"),
        "action_no_trade": float(action == "NO_TRADE"),
    }


def _p_up(raw: dict[str, Any]) -> float:
    up_mid = float(
        raw.get("up_mid")
        or (float(raw.get("up_bid") or 0.0) + float(raw.get("up_ask") or 0.0)) / 2.0
    )
    down_mid = float(
        raw.get("down_mid")
        or (float(raw.get("down_bid") or 0.0) + float(raw.get("down_ask") or 0.0)) / 2.0
    )
    denominator = up_mid + down_mid
    if denominator <= 0.0:
        raise ValueError("market-implied probability inputs are invalid")
    value = up_mid / denominator
    if not 0.0 <= value <= 1.0:
        raise ValueError("market-implied probability is outside [0, 1]")
    return value


def _split_manifest(
    *,
    run_id: str,
    role_manifest_path: Path,
    protocol_descriptor: dict[str, str],
    feature_contract_path: Path,
    action_rows_by_role: dict[str, list[dict[str, Any]]],
    action_row_paths: dict[str, Path],
    corpus_audits: list[dict[str, Any]],
) -> dict[str, Any]:
    market_sets = {
        role: {str(row["market_id"]) for row in rows} for role, rows in action_rows_by_role.items()
    }
    overlap = (
        (market_sets[ROLE_NAMES[0]] & market_sets[ROLE_NAMES[1]])
        | (market_sets[ROLE_NAMES[0]] & market_sets[ROLE_NAMES[2]])
        | (market_sets[ROLE_NAMES[1]] & market_sets[ROLE_NAMES[2]])
    )
    if overlap:
        raise ValueError("role action-row market overlap detected")
    previous_max: int | None = None
    role_summaries: dict[str, Any] = {}
    for role in ROLE_NAMES:
        rows = action_rows_by_role[role]
        minimum = min(int(row["decision_ts"]) for row in rows)
        maximum = max(int(row["decision_ts"]) for row in rows)
        if previous_max is not None and minimum <= previous_max:
            raise ValueError("role chronology overlaps")
        previous_max = maximum
        role_summaries[role] = {
            "market_count": len(market_sets[role]),
            "market_ids": sorted(market_sets[role]),
            "market_ids_sha256": canonical_json_sha256(sorted(market_sets[role])),
            "decision_count": len(rows) // len(REQUIRED_ACTIONS),
            "action_row_count": len(rows),
            "minimum_decision_ts": minimum,
            "maximum_decision_ts": maximum,
            "support_by_family_side": dict(
                sorted(
                    Counter(
                        f"{row['action_family']}|{row['side']}"
                        for row in rows
                        if row["action_family"] in TRADE_FAMILIES
                    ).items()
                )
            ),
        }
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-split-manifest-v1",
        "run_id": run_id,
        "role_assignment_manifest": _descriptor(role_manifest_path),
        "protocol": protocol_descriptor,
        "feature_contract": _descriptor(feature_contract_path),
        "roles": role_summaries,
        "action_rows": {role: _descriptor(path) for role, path in action_row_paths.items()},
        "corpus_audits": corpus_audits,
        "role_market_overlap_count": 0,
        "chronology_validation_passed": True,
        "feature_causality_violation_count": sum(
            int(audit["feature_causality_violation_count"]) for audit in corpus_audits
        ),
        "role_assignment_completed_before_label_access": True,
        "confirmatory_labels_used_for_tuning": False,
        **_blocked_safety_fields(),
    }
    manifest["split_hash"] = canonical_json_sha256(
        {
            "role_assignment": manifest["role_assignment_manifest"]["sha256"],
            "protocol": protocol_descriptor["sha256"],
            "feature_contract": manifest["feature_contract"]["sha256"],
            "roles": role_summaries,
            "action_rows": {
                role: descriptor["sha256"] for role, descriptor in manifest["action_rows"].items()
            },
        }
    )
    return manifest


def _cross_fit_training_predictions(
    rows: list[dict[str, Any]],
    *,
    feature_columns: tuple[str, ...],
    model_protocol: dict[str, Any],
) -> dict[str, Any]:
    _validate_complete_decision_groups(rows)
    market_first_ts: dict[str, int] = {}
    for row in rows:
        market_id = str(row["market_id"])
        market_first_ts[market_id] = min(
            market_first_ts.get(market_id, int(row["decision_ts"])),
            int(row["decision_ts"]),
        )
    ordered_markets = sorted(market_first_ts, key=lambda value: (market_first_ts[value], value))
    fold_count = int(model_protocol["fold_count"])
    initial_training_market_count = int(model_protocol["initial_training_market_count"])
    validation_market_count_per_fold = int(model_protocol["validation_market_count_per_fold"])
    expected_oof_market_count = int(model_protocol["expected_oof_market_count"])
    if len(ordered_markets) != ROLE_MARKET_COUNTS["development_train"] or fold_count != 5:
        raise ValueError("cross-fit requires exactly 90 markets and five folds")
    if model_protocol.get("fold_assignment") != (
        "chronological_expanding_window_prior_markets_only"
    ):
        raise ValueError("cross-fit requires chronological expanding-window folds")
    if model_protocol.get("future_market_labels_excluded_from_each_fold") is not True:
        raise ValueError("cross-fit must exclude future market labels from every fold")
    if (
        initial_training_market_count + fold_count * validation_market_count_per_fold
        != len(ordered_markets)
        or fold_count * validation_market_count_per_fold != expected_oof_market_count
    ):
        raise ValueError("cross-fit warmup and validation windows do not cover 90 markets")
    fold_market_groups = [
        ordered_markets[
            initial_training_market_count
            + index * validation_market_count_per_fold : initial_training_market_count
            + (index + 1) * validation_market_count_per_fold
        ]
        for index in range(fold_count)
    ]
    oof_rows: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    xgb_protocol = _xgb_model_protocol(model_protocol)
    for fold_index, validation_markets in enumerate(fold_market_groups, start=1):
        validation_start_index = (
            initial_training_market_count + (fold_index - 1) * validation_market_count_per_fold
        )
        validation_set = set(validation_markets)
        training_market_list = ordered_markets[:validation_start_index]
        training_markets = set(training_market_list)
        fit_rows = [row for row in rows if str(row["market_id"]) in training_markets]
        validation_rows = [row for row in rows if str(row["market_id"]) in validation_set]
        training_max_decision_ts = max(int(row["decision_ts"]) for row in fit_rows)
        validation_min_decision_ts = min(int(row["decision_ts"]) for row in validation_rows)
        if training_max_decision_ts >= validation_min_decision_ts:
            raise ValueError("cross-fit training markets are not strictly earlier than validation")
        booster = _train_pairwise_ranker(
            fit_rows,
            feature_columns=feature_columns,
            model_protocol=xgb_protocol,
        )
        predictions = _predict_ranker(
            booster,
            validation_rows,
            feature_columns=feature_columns,
        )
        fold_oof_rows = _attach_group_normalized_rank_features(
            [
                {
                    "fold_index": fold_index,
                    "market_id": row["market_id"],
                    "decision_ts": row["decision_ts"],
                    "action": row["action"],
                    "action_family": row["action_family"],
                    "side": row["side"],
                    "action_row_sha256": row["action_row_sha256"],
                    "oof_raw_prediction": prediction,
                    "target_net_pnl_per_contract": row["target_net_pnl_per_contract"],
                }
                for row, prediction in zip(validation_rows, predictions, strict=True)
            ],
            score_field="oof_raw_prediction",
        )
        oof_rows.extend(fold_oof_rows)
        fold_reports.append(
            {
                "fold_index": fold_index,
                "training_market_count": len(training_markets),
                "validation_market_count": len(validation_markets),
                "training_market_ids_sha256": canonical_json_sha256(training_market_list),
                "validation_market_ids": validation_markets,
                "validation_market_ids_sha256": canonical_json_sha256(validation_markets),
                "market_overlap_count": 0,
                "training_max_decision_ts": training_max_decision_ts,
                "validation_min_decision_ts": validation_min_decision_ts,
                "training_strictly_precedes_validation": True,
                "future_market_label_access_count": 0,
                "ranking_metrics": _decision_group_ranking_metrics(
                    validation_rows,
                    predictions,
                ),
            }
        )
    expected_oof_action_row_count = expected_oof_market_count * len(REQUIRED_ACTIONS)
    if len(oof_rows) != expected_oof_action_row_count:
        raise ValueError("cross-fit OOF prediction coverage is incomplete")
    if len({str(row["action_row_sha256"]) for row in oof_rows}) != len(oof_rows):
        raise ValueError("cross-fit OOF prediction identities are duplicated")
    oof_predictions_by_row = {
        str(row["action_row_sha256"]): float(row["oof_raw_prediction"]) for row in oof_rows
    }
    oof_source_rows = [
        row for row in rows if str(row["action_row_sha256"]) in oof_predictions_by_row
    ]
    return {
        "method": "five_fold_chronological_expanding_window_market_grouped_oof",
        "objective": "rank:pairwise",
        "decision_group_key": "market_id_decision_ts",
        "fold_count": fold_count,
        "market_count": len(ordered_markets),
        "decision_group_count": len(rows) // len(REQUIRED_ACTIONS),
        "initial_training_only_market_count": initial_training_market_count,
        "oof_market_count": expected_oof_market_count,
        "oof_decision_group_count": len(oof_source_rows) // len(REQUIRED_ACTIONS),
        "action_row_count": len(rows),
        "oof_prediction_count": len(oof_rows),
        "oof_prediction_coverage_complete": True,
        "all_development_train_markets_have_oof_predictions": False,
        "initial_training_markets_excluded_from_oof": True,
        "future_market_label_access_violation_count": 0,
        "training_window": "expanding_prior_markets_only",
        "fold_reports": fold_reports,
        "ranking_metrics": _decision_group_ranking_metrics(
            oof_source_rows,
            [oof_predictions_by_row[str(row["action_row_sha256"])] for row in oof_source_rows],
        ),
        "oof_predictions": oof_rows,
        "uses_development_calibration_labels": False,
        "uses_confirmatory_validation_labels": False,
        "uses_issue174_confirmatory_labels": False,
    }


def _xgb_model_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "objective",
        "eval_metric",
        "num_boost_round",
        "max_depth",
        "eta",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "lambda",
        "alpha",
        "seed",
        "nthread",
        "verbosity",
    )
    return {field: protocol[field] for field in fields}


def _train_pairwise_ranker(
    rows: list[dict[str, Any]],
    *,
    feature_columns: tuple[str, ...],
    model_protocol: dict[str, Any],
) -> xgb.Booster:
    ordered_rows, group_sizes = _ordered_decision_group_rows(rows)
    matrix = xgb.DMatrix(
        np.asarray(
            [
                [float(row["decision_time_features"][name]) for name in feature_columns]
                for row in ordered_rows
            ],
            dtype=np.float32,
        ),
        label=np.asarray(_pairwise_relevance_labels(ordered_rows), dtype=np.float32),
        feature_names=list(feature_columns),
    )
    matrix.set_group(group_sizes)
    parameters = {key: value for key, value in model_protocol.items() if key != "num_boost_round"}
    return xgb.train(
        parameters,
        matrix,
        num_boost_round=int(model_protocol["num_boost_round"]),
    )


def _predict_ranker(
    booster: xgb.Booster,
    rows: list[dict[str, Any]],
    *,
    feature_columns: tuple[str, ...],
) -> list[float]:
    matrix = xgb.DMatrix(
        np.asarray(
            [
                [float(row["decision_time_features"][name]) for name in feature_columns]
                for row in rows
            ],
            dtype=np.float32,
        ),
        feature_names=list(feature_columns),
    )
    return [float(value) for value in booster.predict(matrix)]


def _ordered_decision_group_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[int]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["market_id"]), int(row["decision_ts"]))].append(row)
    ordered_rows: list[dict[str, Any]] = []
    group_sizes: list[int] = []
    for key in sorted(groups, key=lambda value: (value[1], value[0])):
        group = sorted(groups[key], key=lambda row: str(row["action"]))
        actions = {str(row["action"]) for row in group}
        if actions != set(REQUIRED_ACTIONS) or len(group) != len(REQUIRED_ACTIONS):
            raise ValueError("pairwise ranker requires a complete five-action grid")
        ordered_rows.extend(group)
        group_sizes.append(len(group))
    return ordered_rows, group_sizes


def _validate_complete_decision_groups(rows: list[dict[str, Any]]) -> None:
    _, group_sizes = _ordered_decision_group_rows(rows)
    if not group_sizes or any(size != len(REQUIRED_ACTIONS) for size in group_sizes):
        raise ValueError("decision-group action coverage is incomplete")


def _pairwise_relevance_labels(rows: list[dict[str, Any]]) -> list[float]:
    labels: list[float] = []
    for start in range(0, len(rows), len(REQUIRED_ACTIONS)):
        group = rows[start : start + len(REQUIRED_ACTIONS)]
        ranked = sorted(
            group,
            key=lambda row: (
                float(row["target_net_pnl_per_contract"]),
                str(row["action"]),
            ),
        )
        relevance = {str(row["action_row_sha256"]): float(rank) for rank, row in enumerate(ranked)}
        labels.extend(relevance[str(row["action_row_sha256"])] for row in group)
    return labels


def _decision_group_ranking_metrics(
    rows: list[dict[str, Any]],
    scores: list[float],
) -> dict[str, Any]:
    if len(rows) != len(scores):
        raise ValueError("ranking metric rows and scores must align")
    grouped: dict[tuple[str, int], list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, score in zip(rows, scores, strict=True):
        grouped[(str(row["market_id"]), int(row["decision_ts"]))].append((row, float(score)))
    top1_hits = 0
    regrets: list[float] = []
    for group in grouped.values():
        selected = max(group, key=lambda item: (item[1], str(item[0]["action"])))
        oracle = max(
            group,
            key=lambda item: (
                float(item[0]["target_net_pnl_per_contract"]),
                str(item[0]["action"]),
            ),
        )
        top1_hits += int(selected[0]["action"] == oracle[0]["action"])
        regrets.append(
            float(oracle[0]["target_net_pnl_per_contract"])
            - float(selected[0]["target_net_pnl_per_contract"])
        )
    return {
        "decision_group_count": len(grouped),
        "top1_realized_best_action_hit_rate": (top1_hits / len(grouped) if grouped else 0.0),
        "mean_regret": float(np.mean(regrets)) if regrets else 0.0,
        "maximum_regret": max(regrets, default=0.0),
    }


def _predict_role_rows(
    rows: list[dict[str, Any]],
    *,
    booster: xgb.Booster,
    feature_columns: tuple[str, ...],
) -> list[dict[str, Any]]:
    _validate_complete_decision_groups(rows)
    scores = _predict_ranker(booster, rows, feature_columns=feature_columns)
    output = []
    for row, raw_prediction in zip(rows, scores, strict=True):
        output.append(
            {
                **row,
                "raw_pairwise_rank_score": raw_prediction,
                "ranking_score_source": "model_predicted_pairwise_rank_score",
                "target_used_as_decision_input": False,
                "outcome_fields_used_as_decision_input": False,
            }
        )
    normalized = _attach_group_normalized_rank_features(
        output,
        score_field="raw_pairwise_rank_score",
    )
    for prediction in normalized:
        prediction["prediction_sha256"] = canonical_json_sha256(prediction)
    return normalized


def _attach_group_normalized_rank_features(
    rows: list[dict[str, Any]],
    *,
    score_field: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["market_id"]), int(row["decision_ts"]))].append(row)
    output: list[dict[str, Any]] = []
    for group_key in sorted(grouped, key=lambda value: (value[1], value[0])):
        group = grouped[group_key]
        if {str(row["action"]) for row in group} != set(REQUIRED_ACTIONS):
            raise ValueError("group-normalized rank features require a complete five-action grid")
        scored = [(row, float(row[score_field])) for row in group]
        if not all(math.isfinite(score) for _, score in scored):
            raise ValueError("pairwise rank scores must be finite")
        ranked = sorted(scored, key=lambda item: (-item[1], str(item[0]["action"])))
        score_min = min(score for _, score in scored)
        score_max = max(score for _, score in scored)
        score_range = score_max - score_min
        denominator = score_range if score_range > 1e-12 else 1.0
        no_trade_score = next(score for row, score in scored if str(row["action"]) == "NO_TRADE")
        rank_by_action = {str(row["action"]): rank for rank, (row, _) in enumerate(ranked, start=1)}
        score_by_action = {str(row["action"]): score for row, score in scored}
        for row, score in scored:
            action = str(row["action"])
            best_alternative_score = max(
                candidate_score
                for candidate_action, candidate_score in score_by_action.items()
                if candidate_action != action
            )
            action_rank = rank_by_action[action]
            output.append(
                {
                    **row,
                    "pairwise_action_rank": action_rank,
                    "pairwise_rank_percentile": (
                        (len(REQUIRED_ACTIONS) - action_rank) / (len(REQUIRED_ACTIONS) - 1)
                    ),
                    "pairwise_group_normalized_rank_score": (
                        (score - score_min) / denominator if score_range > 1e-12 else 0.0
                    ),
                    "pairwise_group_score_range": score_range,
                    "pairwise_normalized_margin_vs_no_trade": (
                        (score - no_trade_score) / denominator if score_range > 1e-12 else 0.0
                    ),
                    "pairwise_normalized_margin_vs_best_alternative": (
                        (score - best_alternative_score) / denominator
                        if score_range > 1e-12
                        else 0.0
                    ),
                    "pairwise_rank_normalization_scope": "market_id_decision_ts_five_action_group",
                    "raw_rank_score_cross_model_comparison_allowed": False,
                }
            )
    return output


def _action_advantage_lcb_artifact(
    calibration_predictions: list[dict[str, Any]],
    *,
    train_oof_predictions: list[dict[str, Any]],
    protocol: dict[str, Any],
    feature_contract_sha256: str,
) -> dict[str, Any]:
    lcb_protocol = dict(protocol["action_advantage_lcb_protocol"])
    confidence_level = float(lcb_protocol["confidence_level"])
    bootstrap_resample_count = int(lcb_protocol["bootstrap_resample_count"])
    bootstrap_seed = int(lcb_protocol["bootstrap_seed"])
    minimum_group_markets = int(lcb_protocol["minimum_calibration_unique_markets_per_group"])
    minimum_action_markets = int(lcb_protocol["minimum_calibration_unique_markets_per_action"])
    shrinkage_prior_markets = int(lcb_protocol["shrinkage_prior_market_count"])
    actions: dict[str, Any] = {}
    calibration_groups: dict[str, Any] = {}
    for action_index, action in enumerate(REQUIRED_ACTIONS):
        oof_scores = sorted(
            float(row["pairwise_group_normalized_rank_score"])
            for row in train_oof_predictions
            if row["action"] == action
        )
        if not oof_scores:
            raise ValueError(f"train OOF score coverage is missing for {action}")
        boundaries = [
            float(np.quantile(oof_scores, 1.0 / 3.0, method="linear")),
            float(np.quantile(oof_scores, 2.0 / 3.0, method="linear")),
        ]
        rows = [row for row in calibration_predictions if row["action"] == action]
        action_stats = _market_grouped_target_mean_lcb(
            rows,
            confidence_level=confidence_level,
            bootstrap_resample_count=bootstrap_resample_count,
            seed=bootstrap_seed + action_index * 10_000,
        )
        if action_stats["unique_market_count"] < minimum_action_markets:
            raise ValueError(f"insufficient calibration market support for {action}")
        actions[action] = {
            "calibration_row_count": len(rows),
            "calibration_unique_market_count": action_stats["unique_market_count"],
            "train_oof_group_normalized_score_tertile_boundaries": boundaries,
            "score_bucket_boundaries_source": (
                "development_train_oof_group_normalized_rank_scores_only"
            ),
            "target_mean": action_stats["target_mean"],
            "target_mean_lower_confidence_bound": action_stats[
                "target_mean_lower_confidence_bound"
            ],
            "market_grouped_bootstrap": action_stats,
            "group_normalized_rank_metrics": _regression_metrics(
                [float(row["target_net_pnl_per_contract"]) for row in rows],
                [float(row["pairwise_group_normalized_rank_score"]) for row in rows],
            ),
        }
        for bucket_index, bucket_name in enumerate(("low", "middle", "high")):
            bucket_rows = [
                row
                for row in rows
                if _score_bucket(
                    float(row["pairwise_group_normalized_rank_score"]),
                    boundaries,
                )
                == bucket_name
            ]
            group_stats = _market_grouped_target_mean_lcb(
                bucket_rows,
                confidence_level=confidence_level,
                bootstrap_resample_count=bootstrap_resample_count,
                seed=bootstrap_seed + action_index * 10_000 + bucket_index + 1,
            )
            support_passed = group_stats["unique_market_count"] >= minimum_group_markets
            if support_passed:
                weight = group_stats["unique_market_count"] / (
                    group_stats["unique_market_count"] + shrinkage_prior_markets
                )
                calibrated_expected_net_return = weight * float(group_stats["target_mean"]) + (
                    1.0 - weight
                ) * float(action_stats["target_mean"])
                action_return_lcb = weight * float(
                    group_stats["target_mean_lower_confidence_bound"]
                ) + (1.0 - weight) * float(action_stats["target_mean_lower_confidence_bound"])
                estimate_source = "shrunken_group_and_action_target_mean_lcb"
            else:
                weight = 0.0
                calibrated_expected_net_return = float(action_stats["target_mean"])
                action_return_lcb = float(action_stats["target_mean_lower_confidence_bound"])
                estimate_source = "action_level_target_mean_lcb_fallback"
            if not all(
                math.isfinite(value)
                for value in (
                    calibrated_expected_net_return,
                    action_return_lcb,
                )
            ):
                raise ValueError("action-advantage target estimates must be finite")
            key = f"{action}|{bucket_name}"
            calibration_groups[key] = {
                "action": action,
                "score_bucket": bucket_name,
                "calibration_row_count": len(bucket_rows),
                "calibration_unique_market_count": group_stats["unique_market_count"],
                "minimum_required_unique_markets": minimum_group_markets,
                "group_support_passed": support_passed,
                "group_target_mean": group_stats["target_mean"],
                "group_target_mean_lower_confidence_bound": group_stats[
                    "target_mean_lower_confidence_bound"
                ],
                "action_target_mean": action_stats["target_mean"],
                "action_target_mean_lower_confidence_bound": action_stats[
                    "target_mean_lower_confidence_bound"
                ],
                "shrinkage_group_weight": weight,
                "calibrated_action_expected_net_return": calibrated_expected_net_return,
                "action_return_lower_confidence_bound": action_return_lcb,
                "estimate_source": estimate_source,
                "market_grouped_bootstrap": group_stats,
            }
    artifact = {
        "schema_version": f"{SCHEMA_PREFIX}-calibration-artifact-v1",
        "candidate_name": protocol["candidate_name"],
        "source_split": "development_calibration_only",
        "estimand": "conditional_cost_aware_action_advantage",
        "method": "market_grouped_bootstrap_conditional_action_return_lcb",
        "decision_score_formula": (
            "action_x_oof_group_normalized_rank_score_bucket_target_mean_lcb"
        ),
        "advantage_against_no_trade_required": True,
        "advantage_against_runner_up_required": True,
        "confidence_level": confidence_level,
        "bootstrap_unit": "market_id",
        "bootstrap_resample_count": bootstrap_resample_count,
        "bootstrap_seed": bootstrap_seed,
        "score_bucket_boundaries_source": (
            "development_train_oof_group_normalized_rank_scores_only"
        ),
        "raw_rank_score_cross_model_comparison_allowed": False,
        "shrinkage_formula": lcb_protocol["shrinkage_formula"],
        "insufficient_group_support_fallback": lcb_protocol["insufficient_group_support_fallback"],
        "individual_outcome_quantile_subtraction_enabled": False,
        "affine_calibration_enabled": False,
        "actions": actions,
        "calibration_groups": calibration_groups,
        "feature_contract_sha256": feature_contract_sha256,
        "uses_confirmatory_validation_labels_for_tuning": False,
        "uses_issue174_confirmatory_labels_for_tuning": False,
        "uses_prior_or_future_evidence_for_tuning": False,
        **_blocked_safety_fields(),
    }
    artifact["calibration_artifact_id"] = canonical_json_sha256(artifact)
    return artifact


def _market_grouped_target_mean_lcb(
    rows: list[dict[str, Any]],
    *,
    confidence_level: float,
    bootstrap_resample_count: int,
    seed: int,
) -> dict[str, Any]:
    targets_by_market: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        targets_by_market[str(row["market_id"])].append(float(row["target_net_pnl_per_contract"]))
    market_means = np.asarray(
        [float(np.mean(targets_by_market[market_id])) for market_id in sorted(targets_by_market)],
        dtype=np.float64,
    )
    if market_means.size == 0:
        return {
            "reported": False,
            "unique_market_count": 0,
            "target_mean": None,
            "target_mean_lower_confidence_bound": None,
            "bootstrap_resample_count": bootstrap_resample_count,
            "bootstrap_seed": seed,
        }
    rng = np.random.default_rng(seed)
    sampled = rng.choice(
        market_means,
        size=(bootstrap_resample_count, market_means.size),
        replace=True,
    )
    bootstrap_means = sampled.mean(axis=1)
    lower = float(
        np.quantile(
            bootstrap_means,
            1.0 - confidence_level,
            method="lower",
        )
    )
    target_mean = float(market_means.mean())
    if not math.isfinite(target_mean) or not math.isfinite(lower):
        raise ValueError("market-grouped target mean LCB is not finite")
    return {
        "reported": True,
        "unique_market_count": int(market_means.size),
        "target_mean": target_mean,
        "target_mean_lower_confidence_bound": lower,
        "confidence_level": confidence_level,
        "bootstrap_resample_count": bootstrap_resample_count,
        "bootstrap_seed": seed,
    }


def _score_bucket(score: float, boundaries: list[float]) -> str:
    if len(boundaries) != 2 or not all(math.isfinite(value) for value in boundaries):
        raise ValueError("score tertile boundaries are invalid")
    if score <= boundaries[0]:
        return "low"
    if score <= boundaries[1]:
        return "middle"
    return "high"


def _apply_action_advantage_lcb_scores(
    predictions: list[dict[str, Any]],
    *,
    lcb_artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    output = []
    for row in predictions:
        action = str(row["action"])
        normalized_score = float(row["pairwise_group_normalized_rank_score"])
        boundaries = list(
            lcb_artifact["actions"][action]["train_oof_group_normalized_score_tertile_boundaries"]
        )
        bucket = _score_bucket(normalized_score, boundaries)
        group = lcb_artifact["calibration_groups"][f"{action}|{bucket}"]
        calibrated_score = float(group["calibrated_action_expected_net_return"])
        lcb_score = float(group["action_return_lower_confidence_bound"])
        estimate_source = str(group["estimate_source"])
        if action == "NO_TRADE":
            calibrated_score = 0.0
            lcb_score = 0.0
            estimate_source = "frozen_no_trade_zero_anchor"
        updated = {
            **row,
            "action_advantage_lcb_score_bucket": bucket,
            "action_advantage_lcb_estimate_source": estimate_source,
            "calibrated_action_expected_net_return": calibrated_score,
            "action_advantage_lcb_net_return": lcb_score,
            "ranking_score_source": (
                "model_predicted_pairwise_rank_score_calibrated_to_action_advantage_lcb"
            ),
            "market_implied_probability_used_as_direct_fair_value_ev": False,
        }
        updated["prediction_sha256"] = canonical_json_sha256(updated)
        output.append(updated)
    return output


def _run_policy_replay(
    predictions: list[dict[str, Any]],
    *,
    score_field: str,
    policy_name: str,
    entry_threshold: float,
    runner_up_advantage_threshold: float,
) -> list[dict[str, Any]]:
    by_decision: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_decision[(str(row["market_id"]), int(row["decision_ts"]))].append(row)
    guard_config = _v8_execution_guard_config()
    state = _v8_initial_runtime_state(guard_config)
    closes: dict[str, int] = {}
    replay: list[dict[str, Any]] = []
    for index, ((market_id, decision_ts), action_rows) in enumerate(
        sorted(by_decision.items(), key=lambda item: (item[0][1], item[0][0])),
        start=1,
    ):
        _release_closed_positions(
            state=state,
            market_close_by_open_position=closes,
            decision_ts=decision_ts,
        )
        if {str(row["action"]) for row in action_rows} != set(REQUIRED_ACTIONS):
            raise ValueError("confirmatory action grid is incomplete")
        ranked = sorted(
            action_rows,
            key=lambda row: (-float(row[score_field]), str(row["action"])),
        )
        full_ranking = [
            {
                "rank": rank,
                "selected_action": row["action"],
                "selected_side": row["side"],
                "selected_action_family": row["action_family"],
                "corrected_model_score": float(row[score_field]),
                "raw_model_score": float(row["raw_pairwise_rank_score"]),
                "high_score_flag": float(row[score_field]) >= entry_threshold,
                "p_up_action_disagreement": row["p_up_action_disagreement"],
                "microstructure_snapshot": row["microstructure_snapshot"],
            }
            for rank, row in enumerate(ranked, start=1)
        ]
        selected = ranked[0]
        runner_up = ranked[1]
        selected_action = str(selected["action"])
        decision_score = float(selected[score_field])
        runner_up_score = float(runner_up[score_field])
        runner_up_advantage = decision_score - runner_up_score
        blockers: list[str] = []
        guard_result: dict[str, Any] | None = None
        if selected_action == "NO_TRADE":
            blockers.append("policy_selected_no_trade")
        elif decision_score < entry_threshold:
            blockers.append("expected_net_return_below_frozen_entry_threshold")
        elif runner_up_advantage <= runner_up_advantage_threshold:
            blockers.append("selected_vs_runner_up_advantage_not_positive")
        else:
            guard_context = {
                "decision_group_id": canonical_json_sha256(
                    {"market_id": market_id, "decision_ts": decision_ts}
                ),
                "market_id": market_id,
                "decision_ts": decision_ts,
                "selected_action": selected_action,
                "selected_side": selected["side"],
                "selected_action_family": selected["action_family"],
                "corrected_model_score": decision_score,
                "raw_model_score": selected["raw_pairwise_rank_score"],
                "high_score_flag": decision_score >= entry_threshold,
                "p_up": selected["p_up"],
                "p_down": selected["p_down"],
                "p_up_action_disagreement": selected["p_up_action_disagreement"],
                "microstructure_snapshot": selected["microstructure_snapshot"],
                "reference_price_feature_provenance": selected[
                    "reference_price_feature_provenance"
                ],
                "decision_time_feature_max_input_ts": selected["max_input_ts"],
                "full_5_action_ranking": full_ranking,
            }
            guard_result = _v8_execution_guard_decision(
                guard_context,
                guard_config=guard_config,
                runtime_state=state,
                runtime_mode="simulated_runtime_state",
            )
            blockers.extend(guard_result["execution_blocking_reason_codes"])
        accepted = bool(guard_result and guard_result["order_allowed"])
        executed_action = (
            str(guard_result["execution_guarded_action"]) if accepted else selected_action
        )
        executed = next(row for row in action_rows if str(row["action"]) == executed_action)
        size = float(guard_result["proposed_order_size"]) if accepted else 0.0
        if accepted:
            order_id = f"{policy_name}-confirmatory-{index:06d}"
            _v8_apply_simulated_order_to_state(
                state=state,
                decision=guard_result,
                simulated_order_id=order_id,
            )
            closes[market_id] = int(executed["market_close_ts"])
        execution_price = float(executed["decision_time_features"]["execution_price"])
        target = float(executed["target_net_pnl_per_contract"])
        replay_row = {
            "policy_name": policy_name,
            "market_id": market_id,
            "decision_ts": decision_ts,
            "source_selected_action": selected_action,
            "selected_action": executed_action,
            "selected_side": executed["side"],
            "selected_action_family": executed["action_family"],
            "decision_score": decision_score,
            "runner_up_action": runner_up["action"],
            "runner_up_score": runner_up_score,
            "selected_vs_runner_up_advantage": runner_up_advantage,
            "frozen_runner_up_advantage_threshold": (runner_up_advantage_threshold),
            "score_field": score_field,
            "frozen_entry_threshold": entry_threshold,
            "execution_guard_order_allowed": accepted,
            "guard_action_remapped": accepted and executed_action != selected_action,
            "proposed_order_size": size,
            "accepted_bet_cost_basis": execution_price * size,
            "accepted_bet_net_pnl": target * size if accepted else 0.0,
            "target_cost_components": executed["target_cost_components"],
            "evaluation_target_used_after_selection_for_report_only": True,
            "settlement_resolved_for_report_only": executed["target_resolved_outcome"]
            in {"UP", "DOWN"},
            "execution_blocking_reason_codes": sorted(set(blockers)),
            "required_runtime_fields_present": bool(
                guard_result is None or guard_result["required_runtime_fields_present"]
            ),
            "reference_provenance_valid": executed["reference_price_feature_provenance"].get(
                "provenance_valid"
            )
            is True,
            "paper_only": True,
            "capital_at_risk": False,
        }
        replay_row["replay_row_sha256"] = canonical_json_sha256(replay_row)
        replay.append(replay_row)
    return replay


def _development_freeze_gate(
    *,
    protocol: dict[str, Any],
    action_rows: list[dict[str, Any]],
    candidate_replay: list[dict[str, Any]],
    candidate_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    robustness: dict[str, Any],
) -> dict[str, Any]:
    gates = dict(protocol["development_freeze_gates"])
    market_ids = {str(row["market_id"]) for row in action_rows}
    side_counts = dict(candidate_metrics["accepted_bet_count_by_side"])
    family_counts = dict(candidate_metrics["accepted_bet_count_by_family"])
    bootstrap = robustness["market_bootstrap_interval_95"]
    leave_one_out = robustness["leave_one_market_out"]
    largest_winner = robustness["largest_winner_removal"]
    checks = {
        "calibration_unique_market_support": len(market_ids)
        == int(gates["required_calibration_market_count"]),
        "accepted_bet_support": candidate_metrics["accepted_bet_count"]
        >= int(gates["minimum_accepted_bet_count"]),
        "accepted_unique_market_support": candidate_metrics["accepted_unique_market_count"]
        >= int(gates["minimum_accepted_unique_market_count"]),
        "accepted_side_support": all(
            int(side_counts.get(side, 0)) >= int(gates["minimum_accepted_bet_count_per_side"])
            for side in ("UP", "DOWN")
        ),
        "accepted_family_support": all(
            int(family_counts.get(family, 0)) >= int(gates["minimum_accepted_bet_count_per_family"])
            for family in TRADE_FAMILIES
        ),
        "candidate_net_pnl_positive": candidate_metrics["net_pnl_sum"] > 0.0,
        "candidate_roi_positive": candidate_metrics["roi"] > 0.0,
        "candidate_better_than_frozen_baseline": candidate_metrics["net_pnl_sum"]
        > baseline_metrics["net_pnl_sum"],
        "candidate_minus_baseline_bootstrap_lower_bound_positive": (
            bootstrap.get("reported") is True and float(bootstrap.get("lower") or 0.0) > 0.0
        ),
        "largest_winner_removed_candidate_pnl_positive": (
            largest_winner.get("reported") is True
            and float(largest_winner.get("candidate_net_pnl_after_removal") or 0.0) > 0.0
        ),
        "leave_one_market_out_candidate_minus_baseline_positive": (
            leave_one_out.get("reported") is True
            and leave_one_out.get("all_scenarios_positive") is True
        ),
        "zero_feature_causality_violations": all(
            int(row["max_input_ts"]) <= int(row["decision_ts"]) for row in action_rows
        ),
        "zero_forbidden_evidence_access_violations": all(
            row["target_used_as_decision_input"] is False
            and row["outcome_fields_used_as_decision_input"] is False
            for row in action_rows
        ),
        "all_accepted_runtime_fields_present": all(
            row["required_runtime_fields_present"] is True for row in candidate_replay
        ),
    }
    reason_map = {
        "calibration_unique_market_support": "development_calibration_market_support_failed",
        "accepted_bet_support": "development_accepted_bet_support_failed",
        "accepted_unique_market_support": "development_accepted_unique_market_support_failed",
        "accepted_side_support": "development_side_support_failed",
        "accepted_family_support": "development_family_support_failed",
        "candidate_net_pnl_positive": "development_candidate_net_pnl_not_positive",
        "candidate_roi_positive": "development_candidate_roi_not_positive",
        "candidate_better_than_frozen_baseline": "development_candidate_not_better_than_baseline",
        "candidate_minus_baseline_bootstrap_lower_bound_positive": "development_candidate_minus_baseline_bootstrap_lower_bound_not_positive",
        "largest_winner_removed_candidate_pnl_positive": "development_largest_winner_removed_pnl_not_positive",
        "leave_one_market_out_candidate_minus_baseline_positive": "development_leave_one_market_out_not_all_positive",
        "zero_feature_causality_violations": "development_feature_causality_violation",
        "zero_forbidden_evidence_access_violations": "development_forbidden_evidence_access_violation",
        "all_accepted_runtime_fields_present": "development_runtime_fields_missing",
    }
    reasons = [reason_map[name] for name, passed in checks.items() if not passed]
    return {"passed": not reasons, "checks": checks, "reason_codes": reasons}


def _confirmatory_gate(
    *,
    protocol: dict[str, Any],
    action_rows: list[dict[str, Any]],
    candidate_replay: list[dict[str, Any]],
    candidate_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    robustness: dict[str, Any],
) -> dict[str, Any]:
    gates = dict(protocol["confirmatory_validation_gates"])
    market_ids = {str(row["market_id"]) for row in action_rows}
    side_counts = dict(candidate_metrics["accepted_bet_count_by_side"])
    family_counts = dict(candidate_metrics["accepted_bet_count_by_family"])
    bootstrap = robustness["market_bootstrap_interval_95"]
    leave_one_out = robustness["leave_one_market_out"]
    largest_winner = robustness["largest_winner_removal"]
    checks = {
        "confirmatory_unique_market_support": len(market_ids)
        == int(gates["required_unique_market_count"]),
        "accepted_bet_support": candidate_metrics["accepted_bet_count"]
        >= int(gates["minimum_accepted_bet_count"]),
        "accepted_unique_market_support": candidate_metrics["accepted_unique_market_count"]
        >= int(gates["minimum_accepted_unique_market_count"]),
        "accepted_side_support": all(
            int(side_counts.get(side, 0)) >= int(gates["minimum_accepted_bet_count_per_side"])
            for side in ("UP", "DOWN")
        ),
        "accepted_family_support": all(
            int(family_counts.get(family, 0)) >= int(gates["minimum_accepted_bet_count_per_family"])
            for family in TRADE_FAMILIES
        ),
        "candidate_net_pnl_positive": candidate_metrics["net_pnl_sum"] > 0.0,
        "candidate_roi_positive": candidate_metrics["roi"] > 0.0,
        "candidate_better_than_frozen_baseline": candidate_metrics["net_pnl_sum"]
        > baseline_metrics["net_pnl_sum"],
        "all_accepted_bets_settled": all(
            row["settlement_resolved_for_report_only"] is True
            for row in candidate_replay
            if row["execution_guard_order_allowed"] is True
        ),
        "zero_missing_runtime_fields": all(
            row["required_runtime_fields_present"] is True for row in candidate_replay
        ),
        "zero_provenance_violations": all(
            int(row["max_input_ts"]) <= int(row["decision_ts"])
            and row["reference_price_feature_provenance"].get("provenance_valid") is True
            for row in action_rows
            if row["action"] != "NO_TRADE"
        ),
        "zero_forbidden_inference_field_violations": all(
            row["target_used_as_decision_input"] is False
            and row["outcome_fields_used_as_decision_input"] is False
            for row in action_rows
        ),
        "candidate_minus_baseline_bootstrap_lower_bound_positive": (
            bootstrap.get("reported") is True and float(bootstrap.get("lower") or 0.0) > 0.0
        ),
        "leave_one_market_out_candidate_minus_baseline_positive": (
            leave_one_out.get("reported") is True
            and leave_one_out.get("all_scenarios_positive") is True
        ),
        "largest_winner_removed_candidate_pnl_positive": (
            largest_winner.get("reported") is True
            and float(largest_winner.get("candidate_net_pnl_after_removal") or 0.0) > 0.0
        ),
    }
    reason_map = {
        "confirmatory_unique_market_support": "insufficient_confirmatory_unique_market_support",
        "accepted_bet_support": "insufficient_confirmatory_accepted_bet_support",
        "accepted_unique_market_support": "insufficient_confirmatory_accepted_unique_market_support",
        "accepted_side_support": "insufficient_confirmatory_side_support",
        "accepted_family_support": "insufficient_confirmatory_family_support",
        "candidate_net_pnl_positive": "confirmatory_candidate_net_pnl_not_positive",
        "candidate_roi_positive": "confirmatory_candidate_roi_not_positive",
        "candidate_better_than_frozen_baseline": "confirmatory_candidate_not_better_than_baseline",
        "all_accepted_bets_settled": "confirmatory_accepted_bet_settlement_incomplete",
        "zero_missing_runtime_fields": "confirmatory_runtime_fields_missing",
        "zero_provenance_violations": "confirmatory_provenance_violation",
        "zero_forbidden_inference_field_violations": "confirmatory_forbidden_inference_field_violation",
        "candidate_minus_baseline_bootstrap_lower_bound_positive": "confirmatory_candidate_minus_baseline_bootstrap_lower_bound_not_positive",
        "leave_one_market_out_candidate_minus_baseline_positive": "confirmatory_leave_one_market_out_not_all_positive",
        "largest_winner_removed_candidate_pnl_positive": "confirmatory_largest_winner_removed_pnl_not_positive",
    }
    reasons = [reason_map[name] for name, passed in checks.items() if not passed]
    return {"passed": not reasons, "checks": checks, "reason_codes": reasons}


def _accepted_bet_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row["execution_guard_order_allowed"]]
    chronological = sorted(
        accepted,
        key=lambda row: (
            int(row.get("decision_ts") or 0),
            str(row.get("market_id") or ""),
            str(row.get("selected_action") or ""),
        ),
    )
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for row in chronological:
        cumulative += float(row["accepted_bet_net_pnl"])
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)

    def _pnl_by(field: str) -> dict[str, float]:
        values: dict[str, float] = defaultdict(float)
        for row in accepted:
            values[str(row.get(field) or "UNKNOWN")] += float(
                row["accepted_bet_net_pnl"]
            )
        return dict(sorted(values.items()))

    market_pnl = _pnl_by("market_id")
    return {
        "accepted_bet_count": len(accepted),
        "accepted_unique_market_count": len(market_pnl),
        "net_pnl_by_side": _pnl_by("selected_side"),
        "net_pnl_by_family": _pnl_by("selected_action_family"),
        "net_pnl_by_action": _pnl_by("selected_action"),
        "net_pnl_by_market": market_pnl,
        "chronological_sort_fields": [
            "decision_ts",
            "market_id",
            "selected_action",
        ],
        "max_drawdown_ordering": "chronological_decision_time",
        "max_drawdown": max_drawdown,
        "terminal_cumulative_net_pnl": cumulative,
    }


def _training_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# #175 Pairwise Action-Advantage LCB Training",
            "",
            f"- candidate: `{report['candidate_name']}`",
            f"- training markets: `{report['training_market_count']}`",
            f"- calibration markets: `{report['calibration_market_count']}`",
            f"- cross-fit folds: `{report['cross_fit']['fold_count']}`",
            "- confirmatory labels used for tuning: `false`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )


def _pre_label_access_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# #189 Pairwise Pre-Label Access Lineage Audit",
            "",
            f"- audit passed: `{str(report['pre_label_access_validation_passed']).lower()}`",
            f"- selected markets: `{report['selected_market_count']}`",
            f"- role counts: `{report['role_market_counts']}`",
            f"- support target ready: `{str(report['supplemental_support_target_ready']).lower()}`",
            f"- support lineage hash verified: `{str(report['support_gate_lineage_hash_verified']).lower()}`",
            f"- future holdout pre-registration ready: `{str(report['future_holdout_pre_registration_ready']).lower()}`",
            f"- future quality-valid target / max attempts: `{report['future_holdout_target_valid_market_count']}/{report['future_holdout_maximum_capture_attempt_count']}`",
            "- future collection controlled by outcomes/model/PnL: `false`",
            "- labels/outcomes/PnL opened before audit: `false`",
            "- prediction attempted before audit: `false`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )


def _calibration_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# #175 Conformal Family LCB Calibration",
            "",
            f"- source split: `{report['source_split']}`",
            f"- method: `{report['method']}`",
            "- confirmatory labels opened before calibration freeze: `false`",
            "- confirmatory labels used for tuning: `false`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )


def _leakage_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# #175 Leakage And Role Audit",
            "",
            f"- audit passed: `{str(report['leakage_and_role_audit_passed']).lower()}`",
            f"- role overlap count: `{report['role_market_overlap_count']}`",
            f"- prior overlap count: `{report['prior_market_overlap_count']}`",
            f"- feature causality violations: `{report['feature_causality_violation_count']}`",
            f"- forbidden inference fields: `{report['forbidden_inference_field_violation_count']}`",
            "- confirmatory labels used for tuning: `false`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )


def _validation_markdown(report: dict[str, Any]) -> str:
    candidate = report["candidate_metrics"]
    baseline = report["baseline_metrics"]
    return "\n".join(
        [
            "# #175 Untouched Confirmatory Validation",
            "",
            f"- gate passed: `{str(report['confirmatory_gate_passed']).lower()}`",
            f"- accepted bets: `{candidate['accepted_bet_count']}`",
            f"- candidate net PnL: `{candidate['net_pnl_sum']:.12f}`",
            f"- baseline net PnL: `{baseline['net_pnl_sum']:.12f}`",
            f"- blockers: `{report['confirmatory_gate_blocking_reason_codes']}`",
            "- confirmatory labels used for tuning: `false`",
            "- future unseen holdout required: `true`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )


def _accepted_bet_pnl_markdown(report: dict[str, Any]) -> str:
    candidate = report["candidate_metrics"]
    diagnostics = report["candidate_accepted_bet_diagnostics"]
    return "\n".join(
        [
            "# #189 Confirmatory Accepted-Bet Cost-Aware PnL",
            "",
            f"- evaluation scope: `{report['evaluation_scope']}`",
            f"- accepted bets: `{candidate['accepted_bet_count']}`",
            f"- accepted markets: `{candidate['accepted_unique_market_count']}`",
            f"- cost basis: `{candidate['cost_basis_sum']:.12f}`",
            f"- net PnL: `{candidate['net_pnl_sum']:.12f}`",
            f"- ROI: `{candidate['roi']:.12f}`",
            f"- max drawdown: `{diagnostics['max_drawdown']:.12f}`",
            f"- PnL by side: `{diagnostics['net_pnl_by_side']}`",
            f"- PnL by family: `{diagnostics['net_pnl_by_family']}`",
            "- confirmatory labels used for tuning: `false`",
            "- future unseen holdout required: `true`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )


def _blocked_safety_fields() -> dict[str, Any]:
    return {
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }


def _find_fields(payload: Any, forbidden: set[str], prefix: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key) in forbidden:
                found.add(path)
            found.update(_find_fields(value, forbidden, path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.update(_find_fields(value, forbidden, f"{prefix}[{index}]"))
    return found


def _verified_descriptor(payload: Any, *, name: str) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise ValueError(f"{name} descriptor is missing")
    path = Path(str(payload.get("path") or "")).resolve()
    expected_sha256 = str(payload.get("sha256") or "")
    _verify_pin(path, expected_sha256, name=name)
    return {"path": str(path), "sha256": expected_sha256.lower()}


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def _verify_pin(path: Path, expected_sha256: str, *, name: str) -> None:
    if not path.is_file():
        raise ValueError(f"{name} is missing: {path}")
    _require_sha256(expected_sha256, name=f"{name} SHA-256")
    if _sha256_file(path) != expected_sha256.lower():
        raise ValueError(f"{name} SHA-256 mismatch")


def _require_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise ValueError(f"{name} must be a SHA-256 digest")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")
