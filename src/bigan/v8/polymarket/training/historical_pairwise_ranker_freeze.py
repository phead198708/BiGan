"""Freeze a historical-train-only pairwise action ranker."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.contracts import compact_safety_fields
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb import (
    ROLE_MARKET_COUNTS,
    validate_pairwise_action_advantage_lcb_feature_contract,
    validate_pairwise_action_advantage_lcb_protocol,
)
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb_fit import (
    FORBIDDEN_DECISION_FIELDS,
    _cross_fit_training_predictions,
    _materialize_role_action_rows,
    _train_pairwise_ranker,
    _xgb_model_protocol,
)
from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_action_value import (
    REQUIRED_ACTIONS,
)

SCHEMA_PREFIX = "bigan-v8-historical-pairwise-ranker"
CANDIDATE_LINEAGE = (
    "historical_train_fresh_calibration_pairwise_action_advantage_lcb_v1"
)
MODEL_FILENAME = "historical_pairwise_ranker.xgb.json"
REQUIRED_HISTORICAL_MARKET_COUNT = 90


@dataclass(frozen=True, slots=True)
class HistoricalPairwiseRankerFreezeConfig:
    """Immutable inputs for historical pairwise ranker fitting."""

    run_id: str
    output_dir: Path | str
    registry_descriptor_path: Path | str
    expected_registry_descriptor_sha256: str
    registry_manifest_path: Path | str
    expected_registry_manifest_sha256: str
    registry_report_path: Path | str
    expected_registry_report_sha256: str
    registry_rows_path: Path | str
    expected_registry_rows_sha256: str
    protocol_path: Path | str
    expected_protocol_sha256: str
    feature_contract_path: Path | str
    expected_feature_contract_sha256: str
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for name, value in (
            ("registry descriptor SHA-256", self.expected_registry_descriptor_sha256),
            ("registry manifest SHA-256", self.expected_registry_manifest_sha256),
            ("registry report SHA-256", self.expected_registry_report_sha256),
            ("registry rows SHA-256", self.expected_registry_rows_sha256),
            ("protocol SHA-256", self.expected_protocol_sha256),
            ("feature contract SHA-256", self.expected_feature_contract_sha256),
        ):
            _require_sha256(value, name=name)
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(
            self, "registry_descriptor_path", Path(self.registry_descriptor_path)
        )
        object.__setattr__(self, "registry_manifest_path", Path(self.registry_manifest_path))
        object.__setattr__(self, "registry_report_path", Path(self.registry_report_path))
        object.__setattr__(self, "registry_rows_path", Path(self.registry_rows_path))
        object.__setattr__(self, "protocol_path", Path(self.protocol_path))
        object.__setattr__(self, "feature_contract_path", Path(self.feature_contract_path))


def freeze_historical_pairwise_ranker(
    config: HistoricalPairwiseRankerFreezeConfig,
) -> dict[str, Any]:
    """Fit and freeze a ranker without calibration, replay, or execution."""

    descriptor_path = config.registry_descriptor_path.resolve()
    registry_manifest_path = config.registry_manifest_path.resolve()
    registry_report_path = config.registry_report_path.resolve()
    registry_rows_path = config.registry_rows_path.resolve()
    protocol_path = config.protocol_path.resolve()
    feature_contract_path = config.feature_contract_path.resolve()
    for path, expected, name in (
        (
            descriptor_path,
            config.expected_registry_descriptor_sha256,
            "registry descriptor",
        ),
        (
            registry_manifest_path,
            config.expected_registry_manifest_sha256,
            "registry manifest",
        ),
        (registry_report_path, config.expected_registry_report_sha256, "registry report"),
        (registry_rows_path, config.expected_registry_rows_sha256, "registry rows"),
        (protocol_path, config.expected_protocol_sha256, "pairwise protocol"),
        (
            feature_contract_path,
            config.expected_feature_contract_sha256,
            "pairwise feature contract",
        ),
    ):
        _verify_pin(path, expected, name=name)
    registry_descriptor = _load_json(descriptor_path)
    registry_manifest = _load_json(registry_manifest_path)
    registry_report = _load_json(registry_report_path)
    registry_rows = _load_jsonl(registry_rows_path)
    _validate_registry_inputs(
        descriptor=registry_descriptor,
        manifest=registry_manifest,
        report=registry_report,
        rows=registry_rows,
        manifest_path=registry_manifest_path,
        report_path=registry_report_path,
        rows_path=registry_rows_path,
    )

    protocol = _load_json(protocol_path)
    validate_pairwise_action_advantage_lcb_protocol(protocol)
    feature_contract = _load_json(feature_contract_path)
    validate_pairwise_action_advantage_lcb_feature_contract(
        feature_contract,
        expected_parent_protocol_sha256=config.expected_protocol_sha256,
    )
    feature_columns = tuple(str(value) for value in feature_contract["feature_columns"])
    role_rows = [_registry_to_role_row(row) for row in registry_rows]
    action_rows_by_role, corpus_audits = _materialize_role_action_rows(
        role_rows,
        feature_columns=feature_columns,
        roles=("development_train",),
    )
    action_rows = action_rows_by_role["development_train"]
    _validate_historical_action_rows(action_rows)

    run_dir = (config.output_dir / config.run_id).resolve()
    if run_dir.exists() and not config.overwrite_existing:
        raise ValueError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    action_rows_path = run_dir / "historical_pairwise_action_rows.jsonl"
    _write_jsonl(action_rows_path, action_rows)

    cross_fit_result = _cross_fit_training_predictions(
        action_rows,
        feature_columns=feature_columns,
        model_protocol=dict(protocol["cross_fit_protocol"]),
    )
    oof_predictions = list(cross_fit_result.pop("oof_predictions"))
    oof_path = run_dir / "historical_pairwise_train_oof_predictions.jsonl"
    _write_jsonl(oof_path, oof_predictions)
    booster = _train_pairwise_ranker(
        action_rows,
        feature_columns=feature_columns,
        model_protocol=_xgb_model_protocol(dict(protocol["cross_fit_protocol"])),
    )
    model_path = run_dir / MODEL_FILENAME
    booster.save_model(model_path)

    dataset_hash = canonical_json_sha256(action_rows)
    oof_dataset_hash = canonical_json_sha256(oof_predictions)
    split_identity = {
        "registry_selected_market_ids_sha256": registry_manifest[
            "selected_market_ids_sha256"
        ],
        "ordered_market_ids": [
            str(row["market_id"])
            for row in sorted(registry_rows, key=lambda row: int(row["selection_rank"]))
        ],
        "folds": [
            {
                "fold_index": fold["fold_index"],
                "training_market_ids_sha256": fold["training_market_ids_sha256"],
                "validation_market_ids_sha256": fold["validation_market_ids_sha256"],
                "training_max_decision_ts": fold["training_max_decision_ts"],
                "validation_min_decision_ts": fold["validation_min_decision_ts"],
            }
            for fold in cross_fit_result["fold_reports"]
        ],
    }
    split_hash = canonical_json_sha256(split_identity)
    model_config_hash = canonical_json_sha256(
        _xgb_model_protocol(dict(protocol["cross_fit_protocol"]))
    )
    training_report = {
        "schema_version": f"{SCHEMA_PREFIX}-cross-fit-report-v1",
        "run_id": config.run_id,
        "candidate_lineage": CANDIDATE_LINEAGE,
        "training_role": "historical_development_train",
        "training_market_count": REQUIRED_HISTORICAL_MARKET_COUNT,
        "decision_group_count": len(action_rows) // len(REQUIRED_ACTIONS),
        "action_row_count": len(action_rows),
        "feature_columns": list(feature_columns),
        "protocol": _descriptor(protocol_path),
        "feature_contract": _descriptor(feature_contract_path),
        "registry_descriptor": _descriptor(descriptor_path),
        "registry_manifest": _descriptor(registry_manifest_path),
        "registry_report": _descriptor(registry_report_path),
        "registry_rows": _descriptor(registry_rows_path),
        "historical_action_rows": _descriptor(action_rows_path),
        "train_oof_predictions": _descriptor(oof_path),
        "model": _descriptor(model_path),
        "dataset_hash": dataset_hash,
        "oof_dataset_hash": oof_dataset_hash,
        "split_hash": split_hash,
        "model_config_hash": model_config_hash,
        "cross_fit": cross_fit_result,
        "historical_outcomes_used_as_training_targets_only": True,
        "oof_metrics_report_only": True,
        "oof_metrics_used_for_model_or_threshold_tuning": False,
        "fresh_calibration_labels_loaded": False,
        "confirmatory_labels_loaded": False,
        "current_issue175_labels_loaded": False,
        "action_advantage_calibration_attempted": False,
        "policy_replay_attempted": False,
        "accepted_bet_evaluation_attempted": False,
        "fresh_calibration_required": True,
        "rank_scores_execution_eligible": False,
        **_blocked_safety_fields(),
    }
    training_report["report_id"] = canonical_json_sha256(training_report)
    training_report_path = run_dir / "historical_pairwise_cross_fit_report.json"
    _write_json(training_report_path, training_report)
    training_markdown_path = run_dir / "historical_pairwise_cross_fit_report.md"
    training_markdown_path.write_text(
        _training_markdown(training_report),
        encoding="utf-8",
    )

    leakage_audit = _leakage_audit(
        run_id=config.run_id,
        action_rows=action_rows,
        corpus_audits=corpus_audits,
        cross_fit=cross_fit_result,
        registry_report=registry_report,
    )
    leakage_path = run_dir / "historical_pairwise_leakage_audit.json"
    _write_json(leakage_path, leakage_audit)
    leakage_markdown_path = run_dir / "historical_pairwise_leakage_audit.md"
    leakage_markdown_path.write_text(
        _leakage_markdown(leakage_audit),
        encoding="utf-8",
    )
    if leakage_audit["leakage_audit_passed"] is not True:
        raise ValueError("historical pairwise leakage audit failed")

    freeze_manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-freeze-manifest-v1",
        "run_id": config.run_id,
        "candidate_lineage": CANDIDATE_LINEAGE,
        "freeze_status": "historical_ranker_frozen_awaiting_fresh_calibration",
        "registry_descriptor": _descriptor(descriptor_path),
        "registry_manifest": _descriptor(registry_manifest_path),
        "registry_report": _descriptor(registry_report_path),
        "registry_rows": _descriptor(registry_rows_path),
        "protocol": _descriptor(protocol_path),
        "feature_contract": _descriptor(feature_contract_path),
        "historical_action_rows": _descriptor(action_rows_path),
        "train_oof_predictions": _descriptor(oof_path),
        "cross_fit_report": _descriptor(training_report_path),
        "leakage_audit": _descriptor(leakage_path),
        "model": _descriptor(model_path),
        "model_sha256": _sha256_file(model_path),
        "dataset_hash": dataset_hash,
        "oof_dataset_hash": oof_dataset_hash,
        "split_hash": split_hash,
        "model_config_hash": model_config_hash,
        "training_market_count": REQUIRED_HISTORICAL_MARKET_COUNT,
        "oof_market_count": int(cross_fit_result["oof_market_count"]),
        "historical_training_complete": True,
        "historical_outcomes_used_as_training_targets_only": True,
        "fresh_calibration_required": True,
        "fresh_calibration_market_count_required": 45,
        "fresh_confirmatory_market_count_required": 60,
        "rank_scores_execution_eligible": False,
        "action_advantage_lcb_artifact_created": False,
        "calibrated_expected_net_return_available": False,
        "policy_replay_attempted": False,
        "accepted_bet_evaluation_attempted": False,
        "current_issue175_labels_loaded": False,
        "oof_metrics_used_for_tuning": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    freeze_manifest["freeze_id"] = canonical_json_sha256(freeze_manifest)
    freeze_manifest_path = run_dir / "historical_pairwise_ranker_freeze_manifest.json"
    _write_json(freeze_manifest_path, freeze_manifest)
    descriptor = {
        "schema_version": f"{SCHEMA_PREFIX}-freeze-descriptor-v1",
        "run_id": config.run_id,
        "candidate_lineage": CANDIDATE_LINEAGE,
        "freeze_id": freeze_manifest["freeze_id"],
        "freeze_manifest": _descriptor(freeze_manifest_path),
        "model": _descriptor(model_path),
        "historical_action_rows": _descriptor(action_rows_path),
        "train_oof_predictions": _descriptor(oof_path),
        "cross_fit_report": _descriptor(training_report_path),
        "leakage_audit": _descriptor(leakage_path),
        "model_sha256": freeze_manifest["model_sha256"],
        "dataset_hash": dataset_hash,
        "split_hash": split_hash,
        "model_config_hash": model_config_hash,
        "fresh_calibration_required": True,
        "rank_scores_execution_eligible": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        **compact_safety_fields(),
    }
    descriptor["descriptor_id"] = canonical_json_sha256(descriptor)
    descriptor_path = run_dir / "historical_pairwise_ranker_freeze_descriptor.json"
    _write_json(descriptor_path, descriptor)
    return {
        "run_dir": run_dir,
        "action_rows_path": action_rows_path,
        "oof_path": oof_path,
        "model_path": model_path,
        "training_report_path": training_report_path,
        "leakage_audit_path": leakage_path,
        "freeze_manifest_path": freeze_manifest_path,
        "descriptor_path": descriptor_path,
        "training_report": training_report,
        "leakage_audit": leakage_audit,
        "freeze_manifest": freeze_manifest,
        "descriptor": descriptor,
    }


def _validate_registry_inputs(
    *,
    descriptor: dict[str, Any],
    manifest: dict[str, Any],
    report: dict[str, Any],
    rows: list[dict[str, Any]],
    manifest_path: Path,
    report_path: Path,
    rows_path: Path,
) -> None:
    expected_descriptor_id = canonical_json_sha256(
        {key: value for key, value in descriptor.items() if key != "descriptor_id"}
    )
    if str(descriptor.get("descriptor_id") or "") != expected_descriptor_id:
        raise ValueError("registry descriptor id mismatch")
    expected_manifest_id = canonical_json_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_id"}
    )
    if str(manifest.get("manifest_id") or "") != expected_manifest_id:
        raise ValueError("registry manifest id mismatch")
    expected_report_id = canonical_json_sha256(
        {key: value for key, value in report.items() if key != "report_id"}
    )
    if str(report.get("report_id") or "") != expected_report_id:
        raise ValueError("registry report id mismatch")
    descriptor_manifest = {
        "path": str(Path(str(descriptor.get("manifest_path") or "")).resolve()),
        "sha256": str(descriptor.get("manifest_sha256") or ""),
    }
    if descriptor_manifest != _descriptor(manifest_path):
        raise ValueError("registry descriptor manifest mismatch")
    if str(descriptor.get("registry_rows_sha256") or "") != _sha256_file(rows_path):
        raise ValueError("registry descriptor rows mismatch")
    if str(descriptor.get("registry_report_sha256") or "") != _sha256_file(
        report_path
    ):
        raise ValueError("registry descriptor report mismatch")
    if _verified_descriptor(manifest.get("registry_rows"), name="manifest rows") != _descriptor(
        rows_path
    ):
        raise ValueError("registry manifest rows mismatch")
    if _verified_descriptor(
        manifest.get("registry_report"), name="manifest report"
    ) != _descriptor(report_path):
        raise ValueError("registry manifest report mismatch")
    if len(rows) != REQUIRED_HISTORICAL_MARKET_COUNT:
        raise ValueError("historical registry must contain exactly 90 markets")
    if int(manifest.get("selected_market_count") or 0) != len(rows):
        raise ValueError("registry manifest selected market count mismatch")
    if int(report.get("selected_market_count") or 0) != len(rows):
        raise ValueError("registry report selected market count mismatch")
    if report.get("all_selected_strictly_before_boundary") is not True:
        raise ValueError("registry contains post-boundary markets")
    if int(report.get("duplicate_selected_market_count", -1)) != 0:
        raise ValueError("registry report contains duplicate market identities")
    access = dict(report.get("forbidden_evidence_access_audit") or {})
    if access.get("selection_uses_only_compatibility_time_and_identity") is not True:
        raise ValueError("registry selection evidence is not outcome blind")
    for field in (
        "label_rows_semantic_content_parsed",
        "resolution_rows_semantic_content_parsed",
        "outcome_values_loaded",
        "pnl_values_loaded",
        "oracle_values_loaded",
        "oof_metrics_loaded",
        "validation_metrics_loaded",
        "confirmatory_metrics_loaded",
    ):
        if access.get(field) is not False:
            raise ValueError("registry selection opened forbidden evidence")
    ordered_rows = sorted(rows, key=lambda row: int(row.get("selection_rank") or 0))
    if [int(row.get("selection_rank") or 0) for row in ordered_rows] != list(
        range(1, REQUIRED_HISTORICAL_MARKET_COUNT + 1)
    ):
        raise ValueError("registry selection ranks are incomplete")
    market_ids = [str(row.get("market_id") or "") for row in ordered_rows]
    if any(not value for value in market_ids) or len(market_ids) != len(set(market_ids)):
        raise ValueError("registry market identities are incomplete or duplicated")
    market_ids_hash = canonical_json_sha256(market_ids)
    if market_ids_hash != str(manifest.get("selected_market_ids_sha256") or ""):
        raise ValueError("registry selected market hash mismatch")
    if market_ids_hash != str(descriptor.get("selected_market_ids_sha256") or ""):
        raise ValueError("registry descriptor selected market hash mismatch")
    for row in ordered_rows:
        expected_row_id = canonical_json_sha256(
            {key: value for key, value in row.items() if key != "registry_row_id"}
        )
        if str(row.get("registry_row_id") or "") != expected_row_id:
            raise ValueError("registry row id mismatch")
        if row.get("role") != "historical_development_train":
            raise ValueError("registry role is invalid")
        if row.get("strictly_before_boundary") is not True:
            raise ValueError("registry row is not strictly before boundary")
        if row.get("labels_or_outcomes_used_for_selection") is not False:
            raise ValueError("registry row used labels or outcomes for selection")
        if row.get("fresh_calibration_eligible") is not False:
            raise ValueError("historical registry row claims fresh calibration eligibility")
        if row.get("fresh_confirmatory_eligible") is not False:
            raise ValueError("historical registry row claims fresh confirmatory eligibility")


def _registry_to_role_row(row: dict[str, Any]) -> dict[str, Any]:
    artifact_pins = dict(row.get("artifact_pins") or {})
    manifest_pin = artifact_pins.get("polymarket_corpus_manifest.json")
    if not isinstance(manifest_pin, dict):
        raise ValueError("registry corpus manifest pin is missing")
    return {
        "role": "development_train",
        "selection_rank": int(row["selection_rank"]),
        "market_id": str(row["market_id"]),
        "source_corpus_dir": str(row["corpus_dir"]),
        "corpus_manifest": {
            "path": str(Path(str(manifest_pin["path"])).resolve()),
            "sha256": str(manifest_pin["sha256"]),
        },
        "execution_compatibility_validated_before_label_access": True,
        "labels_or_outcomes_opened_for_role_assignment": False,
    }


def _validate_historical_action_rows(rows: list[dict[str, Any]]) -> None:
    market_ids = {str(row.get("market_id") or "") for row in rows}
    if len(market_ids) != ROLE_MARKET_COUNTS["development_train"]:
        raise ValueError("historical action rows do not cover exactly 90 markets")
    if len(rows) % len(REQUIRED_ACTIONS):
        raise ValueError("historical action rows are not complete decision groups")
    if any(int(row["max_input_ts"]) > int(row["decision_ts"]) for row in rows):
        raise ValueError("historical action-row timestamp causality failed")
    if any(row.get("target_used_as_decision_input") is not False for row in rows):
        raise ValueError("historical target was used as a decision input")
    if any(row.get("outcome_fields_used_as_decision_input") is not False for row in rows):
        raise ValueError("historical outcome was used as a decision input")


def _leakage_audit(
    *,
    run_id: str,
    action_rows: list[dict[str, Any]],
    corpus_audits: list[dict[str, Any]],
    cross_fit: dict[str, Any],
    registry_report: dict[str, Any],
) -> dict[str, Any]:
    timestamp_violations = sum(
        int(int(row["max_input_ts"]) > int(row["decision_ts"])) for row in action_rows
    )
    forbidden_feature_violations = sum(
        int(bool(_find_fields(row.get("decision_time_features") or {}, FORBIDDEN_DECISION_FIELDS)))
        for row in action_rows
    )
    target_input_violations = sum(
        int(
            row.get("target_used_as_decision_input") is not False
            or row.get("outcome_fields_used_as_decision_input") is not False
        )
        for row in action_rows
    )
    corpus_causality_violations = sum(
        int(audit.get("feature_causality_violation_count") or 0)
        for audit in corpus_audits
    )
    future_label_violations = int(
        cross_fit.get("future_market_label_access_violation_count") or 0
    )
    checks = {
        "exact_historical_market_count": len(
            {str(row["market_id"]) for row in action_rows}
        )
        == REQUIRED_HISTORICAL_MARKET_COUNT,
        "complete_five_action_groups": len(action_rows) % len(REQUIRED_ACTIONS) == 0,
        "zero_timestamp_causality_violations": timestamp_violations == 0
        and corpus_causality_violations == 0,
        "zero_forbidden_decision_feature_violations": forbidden_feature_violations == 0,
        "zero_target_or_outcome_decision_input_violations": target_input_violations == 0,
        "strict_chronological_oof": all(
            fold.get("training_strictly_precedes_validation") is True
            and int(fold["training_max_decision_ts"])
            < int(fold["validation_min_decision_ts"])
            for fold in cross_fit["fold_reports"]
        ),
        "zero_future_market_label_access": future_label_violations == 0,
        "registry_strictly_pre_boundary": registry_report.get(
            "all_selected_strictly_before_boundary"
        )
        is True,
        "no_fresh_or_confirmatory_labels_loaded": True,
        "no_oof_metric_driven_tuning": True,
    }
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-leakage-audit-v1",
        "run_id": run_id,
        "candidate_lineage": CANDIDATE_LINEAGE,
        "checks": checks,
        "timestamp_causality_violation_count": timestamp_violations,
        "corpus_feature_causality_violation_count": corpus_causality_violations,
        "forbidden_decision_feature_violation_count": forbidden_feature_violations,
        "target_or_outcome_decision_input_violation_count": target_input_violations,
        "future_market_label_access_violation_count": future_label_violations,
        "historical_outcomes_used_as_training_targets_only": True,
        "fresh_calibration_labels_loaded": False,
        "confirmatory_labels_loaded": False,
        "current_issue175_labels_loaded": False,
        "oof_metrics_used_for_tuning": False,
        "leakage_audit_passed": all(checks.values()),
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def _training_markdown(report: dict[str, Any]) -> str:
    cross_fit = report["cross_fit"]
    return "\n".join(
        [
            "# Historical Pairwise Cross-Fit",
            "",
            f"- run id: `{report['run_id']}`",
            f"- candidate lineage: `{report['candidate_lineage']}`",
            f"- training markets: `{report['training_market_count']}`",
            f"- decision groups: `{report['decision_group_count']}`",
            f"- action rows: `{report['action_row_count']}`",
            f"- OOF markets: `{cross_fit['oof_market_count']}`",
            f"- OOF predictions: `{cross_fit['oof_prediction_count']}`",
            f"- model SHA-256: `{report['model']['sha256']}`",
            f"- dataset hash: `{report['dataset_hash']}`",
            f"- split hash: `{report['split_hash']}`",
            "- historical outcomes used as training targets only: `true`",
            "- OOF metrics used for tuning: `false`",
            "- fresh calibration required: `true`",
            "- rank scores execution eligible: `false`",
            "- policy replay attempted: `false`",
            "",
        ]
    )


def _leakage_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Historical Pairwise Leakage Audit",
            "",
            f"- run id: `{report['run_id']}`",
            f"- passed: `{str(report['leakage_audit_passed']).lower()}`",
            (
                "- timestamp causality violations: "
                f"`{report['timestamp_causality_violation_count']}`"
            ),
            (
                "- forbidden decision feature violations: "
                f"`{report['forbidden_decision_feature_violation_count']}`"
            ),
            (
                "- target/outcome decision-input violations: "
                f"`{report['target_or_outcome_decision_input_violation_count']}`"
            ),
            (
                "- future-market label access violations: "
                f"`{report['future_market_label_access_violation_count']}`"
            ),
            "- fresh calibration labels loaded: `false`",
            "- confirmatory labels loaded: `false`",
            "- current #175 labels loaded: `false`",
            "- OOF metrics used for tuning: `false`",
            "",
        ]
    )


def _blocked_safety_fields() -> dict[str, Any]:
    return {
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }


def _verified_descriptor(value: Any, *, name: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} descriptor is missing")
    path = Path(str(value.get("path") or "")).resolve()
    sha256 = str(value.get("sha256") or "")
    _require_sha256(sha256, name=f"{name} SHA-256")
    _verify_pin(path, sha256, name=name)
    return {"path": str(path), "sha256": sha256}


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def _verify_pin(path: Path, expected_sha256: str, *, name: str) -> None:
    if not path.is_file():
        raise ValueError(f"{name} does not exist: {path}")
    if _sha256_file(path) != expected_sha256.lower():
        raise ValueError(f"{name} SHA-256 mismatch")


def _require_sha256(value: str, *, name: str) -> None:
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a SHA-256 hex digest")


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
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"expected JSON object row: {path}")
        rows.append(payload)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _find_fields(payload: Any, forbidden: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key) in forbidden:
                found.add(str(key))
            found.update(_find_fields(value, forbidden))
    elif isinstance(payload, list):
        for value in payload:
            found.update(_find_fields(value, forbidden))
    return found
