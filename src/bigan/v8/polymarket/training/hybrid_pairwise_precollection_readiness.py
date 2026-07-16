"""Fail-closed readiness gate for hybrid fresh calibration collection."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb import (
    validate_pairwise_action_advantage_lcb_feature_contract,
    validate_pairwise_action_advantage_lcb_protocol,
)

SCHEMA_PREFIX = "bigan-v8-hybrid-pairwise-precollection"
PROTOCOL_SCHEMA_VERSION = (
    "bigan-v8-execution-layer-v2-hybrid-pairwise-fresh-calibration-protocol-v1"
)
FORBIDDEN_READINESS_FIELDS = {
    "confirmatory_gate_passed",
    "future_return",
    "net_pnl",
    "oracle_action",
    "outcome",
    "realized_pnl",
    "resolved_outcome",
    "settlement_pnl",
    "target_net_return_after_cost",
    "total_net_pnl_per_notional",
}
COMPLETE_ACTIVE_LINEAGE_STATUSES = {
    "completed",
    "issue179_role_assignment_ready",
    "prior_lineage_complete",
}


@dataclass(frozen=True, slots=True)
class HybridPairwisePrecollectionReadinessConfig:
    """Immutable inputs for readiness evaluation and optional freeze."""

    run_id: str
    output_dir: Path | str
    hybrid_protocol_path: Path | str
    expected_hybrid_protocol_sha256: str
    source_pairwise_protocol_path: Path | str
    expected_source_pairwise_protocol_sha256: str
    source_feature_contract_path: Path | str
    expected_source_feature_contract_sha256: str
    historical_registry_descriptor_path: Path | str
    expected_historical_registry_descriptor_sha256: str
    historical_ranker_descriptor_path: Path | str
    expected_historical_ranker_descriptor_sha256: str
    historical_ranker_manifest_path: Path | str
    expected_historical_ranker_manifest_sha256: str
    freeze_created_at_ts: int
    active_lineage_state_paths: tuple[Path | str, ...] = ()
    final_prior_quarantine_path: Path | str | None = None
    expected_final_prior_quarantine_sha256: str | None = None
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if self.freeze_created_at_ts <= 0:
            raise ValueError("freeze_created_at_ts must be positive")
        for name, value in (
            ("hybrid protocol SHA-256", self.expected_hybrid_protocol_sha256),
            (
                "source pairwise protocol SHA-256",
                self.expected_source_pairwise_protocol_sha256,
            ),
            (
                "source feature contract SHA-256",
                self.expected_source_feature_contract_sha256,
            ),
            (
                "historical registry descriptor SHA-256",
                self.expected_historical_registry_descriptor_sha256,
            ),
            (
                "historical ranker descriptor SHA-256",
                self.expected_historical_ranker_descriptor_sha256,
            ),
            (
                "historical ranker manifest SHA-256",
                self.expected_historical_ranker_manifest_sha256,
            ),
        ):
            _require_sha256(value, name=name)
        if (self.final_prior_quarantine_path is None) != (
            self.expected_final_prior_quarantine_sha256 is None
        ):
            raise ValueError(
                "final prior quarantine path and SHA-256 must be provided together"
            )
        if self.expected_final_prior_quarantine_sha256 is not None:
            _require_sha256(
                self.expected_final_prior_quarantine_sha256,
                name="final prior quarantine SHA-256",
            )
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "hybrid_protocol_path", Path(self.hybrid_protocol_path))
        object.__setattr__(
            self, "source_pairwise_protocol_path", Path(self.source_pairwise_protocol_path)
        )
        object.__setattr__(
            self, "source_feature_contract_path", Path(self.source_feature_contract_path)
        )
        object.__setattr__(
            self,
            "historical_registry_descriptor_path",
            Path(self.historical_registry_descriptor_path),
        )
        object.__setattr__(
            self,
            "historical_ranker_descriptor_path",
            Path(self.historical_ranker_descriptor_path),
        )
        object.__setattr__(
            self,
            "historical_ranker_manifest_path",
            Path(self.historical_ranker_manifest_path),
        )
        object.__setattr__(
            self,
            "active_lineage_state_paths",
            tuple(Path(value) for value in self.active_lineage_state_paths),
        )
        if self.final_prior_quarantine_path is not None:
            object.__setattr__(
                self,
                "final_prior_quarantine_path",
                Path(self.final_prior_quarantine_path),
            )


def evaluate_hybrid_pairwise_precollection_readiness(
    config: HybridPairwisePrecollectionReadinessConfig,
) -> dict[str, Any]:
    """Write readiness evidence and freeze only when prior lineage is complete."""

    hybrid_protocol_path = config.hybrid_protocol_path.resolve()
    source_protocol_path = config.source_pairwise_protocol_path.resolve()
    feature_contract_path = config.source_feature_contract_path.resolve()
    registry_descriptor_path = config.historical_registry_descriptor_path.resolve()
    ranker_descriptor_path = config.historical_ranker_descriptor_path.resolve()
    ranker_manifest_path = config.historical_ranker_manifest_path.resolve()
    for path, expected, name in (
        (
            hybrid_protocol_path,
            config.expected_hybrid_protocol_sha256,
            "hybrid protocol",
        ),
        (
            source_protocol_path,
            config.expected_source_pairwise_protocol_sha256,
            "source pairwise protocol",
        ),
        (
            feature_contract_path,
            config.expected_source_feature_contract_sha256,
            "source feature contract",
        ),
        (
            registry_descriptor_path,
            config.expected_historical_registry_descriptor_sha256,
            "historical registry descriptor",
        ),
        (
            ranker_descriptor_path,
            config.expected_historical_ranker_descriptor_sha256,
            "historical ranker descriptor",
        ),
        (
            ranker_manifest_path,
            config.expected_historical_ranker_manifest_sha256,
            "historical ranker manifest",
        ),
    ):
        _verify_pin(path, expected, name=name)

    hybrid_protocol = _load_json(hybrid_protocol_path)
    source_protocol = _load_json(source_protocol_path)
    feature_contract = _load_json(feature_contract_path)
    registry_descriptor = _load_json(registry_descriptor_path)
    ranker_descriptor = _load_json(ranker_descriptor_path)
    ranker_manifest = _load_json(ranker_manifest_path)
    validate_pairwise_action_advantage_lcb_protocol(source_protocol)
    validate_pairwise_action_advantage_lcb_feature_contract(
        feature_contract,
        expected_parent_protocol_sha256=config.expected_source_pairwise_protocol_sha256,
    )
    validate_hybrid_pairwise_fresh_calibration_protocol(
        hybrid_protocol,
        source_protocol=source_protocol,
        source_protocol_sha256=config.expected_source_pairwise_protocol_sha256,
        feature_contract_sha256=config.expected_source_feature_contract_sha256,
        registry_descriptor=registry_descriptor,
        registry_descriptor_sha256=config.expected_historical_registry_descriptor_sha256,
        ranker_descriptor=ranker_descriptor,
        ranker_descriptor_sha256=config.expected_historical_ranker_descriptor_sha256,
        ranker_manifest=ranker_manifest,
        ranker_manifest_path=ranker_manifest_path,
    )

    active_snapshots = [
        _active_lineage_snapshot(path.resolve())
        for path in config.active_lineage_state_paths
    ]
    active_lineage_complete = bool(active_snapshots) and all(
        snapshot["lineage_complete"] is True for snapshot in active_snapshots
    )
    blocking_reasons: list[str] = []
    if not active_lineage_complete:
        blocking_reasons.append("active_prior_lineage_incomplete")

    quarantine: dict[str, Any] | None = None
    quarantine_descriptor: dict[str, str] | None = None
    quarantine_checks: dict[str, bool] = {
        "provided": False,
        "final": False,
        "active_lineage_complete": False,
        "includes_issue175_through_issue179": False,
        "historical_training_markets_quarantined": False,
        "outcome_blind": False,
        "safety": False,
        "chronology": False,
    }
    if config.final_prior_quarantine_path is None:
        blocking_reasons.append("final_prior_lineage_quarantine_missing")
    else:
        quarantine_path = Path(config.final_prior_quarantine_path).resolve()
        assert config.expected_final_prior_quarantine_sha256 is not None
        _verify_pin(
            quarantine_path,
            config.expected_final_prior_quarantine_sha256,
            name="final prior quarantine",
        )
        quarantine = _load_json(quarantine_path)
        quarantine_descriptor = _descriptor(quarantine_path)
        quarantine_checks = _quarantine_checks(
            quarantine=quarantine,
            protocol=hybrid_protocol,
            freeze_created_at_ts=config.freeze_created_at_ts,
        )
        blocking_reasons.extend(
            f"final_prior_quarantine_{name}_failed"
            for name, passed in quarantine_checks.items()
            if not passed
        )

    checks = {
        "hybrid_protocol_valid": True,
        "historical_ranker_identity_verified": True,
        "historical_registry_identity_verified": True,
        "active_prior_lineage_complete": active_lineage_complete,
        "final_prior_lineage_quarantine_valid": all(quarantine_checks.values()),
        "no_labels_outcomes_or_pnl_used_for_readiness": True,
        "ranker_retraining_or_score_mutation_disabled": True,
        "rank_scores_execution_ineligible_before_calibration": True,
        "collection_start_command_not_generated": True,
    }
    readiness_passed = all(checks.values()) and not blocking_reasons
    run_dir = (config.output_dir / config.run_id).resolve()
    if run_dir.exists():
        if not config.overwrite_existing:
            raise ValueError(f"run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    freeze_manifest_path: Path | None = None
    freeze_manifest: dict[str, Any] | None = None
    if readiness_passed:
        assert quarantine is not None
        minimum_future_decision_ts = max(
            config.freeze_created_at_ts + 1,
            int(quarantine["maximum_prior_decision_ts"]) + 1,
        )
        freeze_manifest = _precollection_freeze_manifest(
            config=config,
            protocol=hybrid_protocol,
            hybrid_protocol_path=hybrid_protocol_path,
            source_protocol_path=source_protocol_path,
            feature_contract_path=feature_contract_path,
            registry_descriptor_path=registry_descriptor_path,
            ranker_descriptor_path=ranker_descriptor_path,
            ranker_manifest_path=ranker_manifest_path,
            quarantine_descriptor=quarantine_descriptor,
            minimum_future_decision_ts=minimum_future_decision_ts,
        )
        freeze_manifest_path = (
            run_dir / "hybrid_pairwise_precollection_freeze_manifest.json"
        )
        _write_json(freeze_manifest_path, freeze_manifest)
        (
            run_dir / "hybrid_pairwise_precollection_freeze_manifest.md"
        ).write_text(_freeze_markdown(freeze_manifest), encoding="utf-8")

    report = {
        "schema_version": f"{SCHEMA_PREFIX}-readiness-report-v1",
        "run_id": config.run_id,
        "readiness_status": (
            "ready_for_separate_future_collection_freeze"
            if readiness_passed
            else "blocked_fail_closed"
        ),
        "hybrid_protocol": _descriptor(hybrid_protocol_path),
        "source_pairwise_protocol": _descriptor(source_protocol_path),
        "source_feature_contract": _descriptor(feature_contract_path),
        "historical_registry_descriptor": _descriptor(registry_descriptor_path),
        "historical_ranker_descriptor": _descriptor(ranker_descriptor_path),
        "historical_ranker_manifest": _descriptor(ranker_manifest_path),
        "active_lineage_snapshots": active_snapshots,
        "active_prior_lineage_complete": active_lineage_complete,
        "final_prior_quarantine": quarantine_descriptor,
        "final_prior_quarantine_checks": quarantine_checks,
        "readiness_checks": checks,
        "precollection_readiness_passed": readiness_passed,
        "precollection_freeze_created": freeze_manifest_path is not None,
        "precollection_freeze_manifest": (
            _descriptor(freeze_manifest_path) if freeze_manifest_path else None
        ),
        "collection_start_allowed": False,
        "collection_start_command_generated": False,
        "blocking_reason_codes": sorted(set(blocking_reasons)),
        "labels_or_outcomes_opened_for_readiness": False,
        "oof_or_validation_metrics_used_for_tuning": False,
        "ranker_retraining_attempted": False,
        "ranker_score_mutation_attempted": False,
        "rank_scores_execution_eligible": False,
        "fresh_calibration_required": True,
        "fresh_confirmatory_required": True,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "hybrid_pairwise_precollection_readiness_report.json"
    _write_json(report_path, report)
    markdown_path = run_dir / "hybrid_pairwise_precollection_readiness_report.md"
    markdown_path.write_text(_readiness_markdown(report), encoding="utf-8")
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-readiness-manifest-v1",
        "run_id": config.run_id,
        "readiness_report": _descriptor(report_path),
        "readiness_report_markdown": _descriptor(markdown_path),
        "hybrid_protocol": _descriptor(hybrid_protocol_path),
        "historical_ranker_descriptor": _descriptor(ranker_descriptor_path),
        "active_lineage_snapshots": active_snapshots,
        "final_prior_quarantine": quarantine_descriptor,
        "precollection_readiness_passed": readiness_passed,
        "precollection_freeze_created": freeze_manifest_path is not None,
        "precollection_freeze_manifest": (
            _descriptor(freeze_manifest_path) if freeze_manifest_path else None
        ),
        "collection_start_allowed": False,
        "collection_start_command_generated": False,
        "blocking_reason_codes": report["blocking_reason_codes"],
        **_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "hybrid_pairwise_precollection_readiness_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "report_path": report_path,
        "markdown_path": markdown_path,
        "manifest_path": manifest_path,
        "freeze_manifest_path": freeze_manifest_path,
        "report": report,
        "manifest": manifest,
        "freeze_manifest": freeze_manifest,
    }


def validate_hybrid_pairwise_fresh_calibration_protocol(
    protocol: dict[str, Any],
    *,
    source_protocol: dict[str, Any],
    source_protocol_sha256: str,
    feature_contract_sha256: str,
    registry_descriptor: dict[str, Any],
    registry_descriptor_sha256: str,
    ranker_descriptor: dict[str, Any],
    ranker_descriptor_sha256: str,
    ranker_manifest: dict[str, Any],
    ranker_manifest_path: Path,
) -> None:
    """Fail closed on role, model, contract, or safety drift."""

    roles = list(protocol.get("fresh_role_plan") or [])
    collection = dict(protocol.get("collection_plan") or {})
    calibration = dict(protocol.get("calibration_contract") or {})
    confirmatory = dict(protocol.get("confirmatory_contract") or {})
    readiness = dict(protocol.get("readiness_contract") or {})
    safety = dict(protocol.get("safety") or {})
    source_hashes = dict(protocol.get("source_contract_hashes") or {})
    registry = dict(protocol.get("historical_development_registry") or {})
    ranker = dict(protocol.get("historical_ranker_freeze") or {})
    checks = {
        "schema_version": protocol.get("schema_version") == PROTOCOL_SCHEMA_VERSION,
        "candidate_lineage": protocol.get("candidate_lineage")
        == "historical_train_fresh_calibration_pairwise_action_advantage_lcb_v1",
        "frozen": protocol.get("frozen") is True,
        "decision_time_safe": protocol.get("decision_time_safe") is True,
        "source_protocol": protocol.get("source_pairwise_protocol_sha256")
        == source_protocol_sha256,
        "feature_contract": protocol.get("source_feature_contract_sha256")
        == feature_contract_sha256,
        "source_contract_hashes": source_hashes
        == {
            "collector_contract_sha256": canonical_json_sha256(
                source_protocol["collector_contract"]
            ),
            "action_advantage_lcb_protocol_sha256": canonical_json_sha256(
                source_protocol["action_advantage_lcb_protocol"]
            ),
            "development_freeze_gates_sha256": canonical_json_sha256(
                source_protocol["development_freeze_gates"]
            ),
            "confirmatory_validation_gates_sha256": canonical_json_sha256(
                source_protocol["confirmatory_validation_gates"]
            ),
            "frozen_execution_contract_sha256": canonical_json_sha256(
                source_protocol["frozen_execution_contract"]
            ),
        },
        "historical_registry": registry.get("descriptor_sha256")
        == registry_descriptor_sha256
        and int(registry.get("selected_market_count") or 0) == 90
        and registry.get("selected_market_ids_sha256")
        == registry_descriptor.get("selected_market_ids_sha256"),
        "historical_ranker": ranker.get("descriptor_sha256")
        == ranker_descriptor_sha256
        and ranker.get("freeze_id") == ranker_descriptor.get("freeze_id")
        and ranker.get("model_sha256") == ranker_descriptor.get("model_sha256")
        and ranker_descriptor.get("model", {}).get("sha256")
        == ranker.get("model_sha256")
        and ranker.get("dataset_hash") == ranker_descriptor.get("dataset_hash")
        and ranker.get("split_hash") == ranker_descriptor.get("split_hash")
        and ranker.get("model_config_hash")
        == ranker_descriptor.get("model_config_hash")
        and ranker.get("fresh_calibration_required") is True
        and ranker.get("rank_scores_execution_eligible") is False,
        "ranker_manifest": _sha256_file(ranker_manifest_path)
        == str(ranker_descriptor["freeze_manifest"]["sha256"])
        and ranker_manifest.get("freeze_id") == ranker.get("freeze_id")
        and ranker_manifest.get("model_sha256") == ranker.get("model_sha256")
        and ranker_manifest.get("dataset_hash") == ranker.get("dataset_hash")
        and ranker_manifest.get("oof_dataset_hash") == ranker.get("oof_dataset_hash")
        and ranker_manifest.get("split_hash") == ranker.get("split_hash")
        and ranker_manifest.get("model_config_hash")
        == ranker.get("model_config_hash")
        and ranker_manifest.get("rank_scores_execution_eligible") is False,
        "roles": roles
        == [
            {
                "role": "fresh_development_calibration",
                "valid_market_rank_start": 1,
                "valid_market_rank_end": 45,
            },
            {
                "role": "fresh_confirmatory_validation",
                "valid_market_rank_start": 46,
                "valid_market_rank_end": 105,
            },
        ],
        "collection": collection.get("target_valid_unique_market_count") == 105
        and collection.get("initial_capture_attempt_count") == 120
        and collection.get("maximum_total_capture_attempt_count") == 150
        and collection.get("replacement_only_for_pre_label_capture_quality_failure")
        is True
        and collection.get("bounded_continuation_only_for_support_shortfall") is True,
        "collection_lineage": collection.get("role_assignment")
        == "earliest_quality_valid_unique_markets_chronological_outcome_blind"
        and collection.get(
            "active_issue175_through_issue179_markets_must_be_quarantined"
        )
        is True,
        "calibration": calibration.get("ranker_retraining_allowed") is False
        and calibration.get("ranker_score_mutation_allowed") is False
        and calibration.get("fresh_calibration_market_count") == 45
        and calibration.get("action_advantage_lcb_only") is True
        and calibration.get(
            "development_gate_must_pass_before_confirmatory_label_access"
        )
        is True
        and calibration.get("uses_confirmatory_labels_for_tuning") is False
        and calibration.get("uses_current_oof_or_validation_metrics_for_tuning")
        is False,
        "confirmatory": confirmatory.get("fresh_confirmatory_market_count") == 60
        and confirmatory.get("one_shot") is True
        and confirmatory.get("retry_or_reselection_from_results_allowed") is False
        and confirmatory.get("threshold_or_gate_change_from_results_allowed") is False,
        "readiness": readiness.get("final_prior_lineage_quarantine_required") is True
        and readiness.get("active_prior_lineage_must_be_complete") is True
        and readiness.get("historical_training_markets_must_be_in_prior_quarantine")
        is True
        and readiness.get(
            "future_decision_ts_strictly_after_model_and_prior_lineage_freeze"
        )
        is True
        and readiness.get("labels_or_outcomes_allowed_for_readiness") is False
        and readiness.get("collection_start_command_allowed_from_readiness_report")
        is False,
        "safety": safety.get("paper_only") is True
        and safety.get("capital_at_risk") is False
        and safety.get("polymarket_write_enabled") is False
        and safety.get("wallet_signing_enabled") is False
        and safety.get("source_model_candidate_eligible") is False
        and safety.get("freeze_ready") is False
        and safety.get("promotion_evidence_eligible") is False
        and safety.get("v8_execution_handoff_allowed") is False
        and safety.get("#134_resume_allowed") is False
        and safety.get("#146_start_allowed") is False,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("invalid hybrid pairwise protocol: " + ", ".join(failed))


def _active_lineage_snapshot(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    forbidden = sorted(_find_fields(payload, FORBIDDEN_READINESS_FIELDS))
    status = str(payload.get("status") or "")
    if status:
        complete = status in COMPLETE_ACTIVE_LINEAGE_STATUSES
    else:
        complete = (
            payload.get("collection_complete") is True
            and int(payload.get("pending_resolution_count") or 0) == 0
            and int(payload.get("error_count") or 0) == 0
        )
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "status": status or None,
        "capture_count": int(payload.get("capture_count") or 0),
        "exported_round_count": int(payload.get("exported_round_count") or 0),
        "pending_resolution_count": int(payload.get("pending_resolution_count") or 0),
        "error_count": int(
            payload.get("error_count")
            or payload.get("collector_error_count")
            or 0
        ),
        "lineage_complete": complete and not forbidden,
        "forbidden_field_paths": forbidden,
    }


def _quarantine_checks(
    *,
    quarantine: dict[str, Any],
    protocol: dict[str, Any],
    freeze_created_at_ts: int,
) -> dict[str, bool]:
    forbidden = _find_fields(quarantine, FORBIDDEN_READINESS_FIELDS)
    maximum_prior_decision_ts = int(quarantine.get("maximum_prior_decision_ts") or 0)
    registry = protocol["historical_development_registry"]
    safety = dict(quarantine.get("safety") or {})
    return {
        "provided": True,
        "final": quarantine.get("final") is True,
        "active_lineage_complete": quarantine.get("active_prior_lineage_complete")
        is True,
        "includes_issue175_through_issue179": quarantine.get(
            "includes_issue175_through_issue179"
        )
        is True,
        "historical_training_markets_quarantined": quarantine.get(
            "historical_development_market_ids_sha256"
        )
        == registry["selected_market_ids_sha256"],
        "outcome_blind": not forbidden
        and quarantine.get("outcome_label_or_pnl_artifacts_opened") is False
        and quarantine.get("resolution_artifacts_opened") is False,
        "safety": safety.get("paper_only") is True
        and safety.get("capital_at_risk") is False
        and safety.get("polymarket_write_enabled") is False
        and safety.get("wallet_signing_enabled") is False,
        "chronology": maximum_prior_decision_ts > 0
        and freeze_created_at_ts > maximum_prior_decision_ts,
    }


def _precollection_freeze_manifest(
    *,
    config: HybridPairwisePrecollectionReadinessConfig,
    protocol: dict[str, Any],
    hybrid_protocol_path: Path,
    source_protocol_path: Path,
    feature_contract_path: Path,
    registry_descriptor_path: Path,
    ranker_descriptor_path: Path,
    ranker_manifest_path: Path,
    quarantine_descriptor: dict[str, str] | None,
    minimum_future_decision_ts: int,
) -> dict[str, Any]:
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-freeze-manifest-v1",
        "run_id": config.run_id,
        "candidate_lineage": protocol["candidate_lineage"],
        "freeze_created_at_ts": config.freeze_created_at_ts,
        "minimum_collection_decision_ts": minimum_future_decision_ts,
        "hybrid_protocol": _descriptor(hybrid_protocol_path),
        "source_pairwise_protocol": _descriptor(source_protocol_path),
        "source_feature_contract": _descriptor(feature_contract_path),
        "historical_registry_descriptor": _descriptor(registry_descriptor_path),
        "historical_ranker_descriptor": _descriptor(ranker_descriptor_path),
        "historical_ranker_manifest": _descriptor(ranker_manifest_path),
        "final_prior_lineage_quarantine": quarantine_descriptor,
        "fresh_role_plan": protocol["fresh_role_plan"],
        "collection_plan": protocol["collection_plan"],
        "ranker_retraining_allowed": False,
        "ranker_score_mutation_allowed": False,
        "labels_or_outcomes_opened_for_role_assignment": False,
        "confirmatory_labels_opened": False,
        "collection_started": False,
        "collection_start_allowed": False,
        "collection_start_command_generated": False,
        "fresh_calibration_required": True,
        "rank_scores_execution_eligible": False,
        **_blocked_safety_fields(),
    }
    manifest["precollection_freeze_id"] = canonical_json_sha256(manifest)
    return manifest


def _readiness_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Hybrid Pairwise Precollection Readiness",
            "",
            f"- run id: `{report['run_id']}`",
            f"- status: `{report['readiness_status']}`",
            (
                "- readiness passed: "
                f"`{str(report['precollection_readiness_passed']).lower()}`"
            ),
            (
                "- precollection freeze created: "
                f"`{str(report['precollection_freeze_created']).lower()}`"
            ),
            "- collection start allowed: `false`",
            "- collection start command generated: `false`",
            f"- blocking reasons: `{report['blocking_reason_codes']}`",
            "- rank scores execution eligible: `false`",
            "- labels/outcomes opened for readiness: `false`",
            "",
        ]
    )


def _freeze_markdown(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Hybrid Pairwise Precollection Freeze",
            "",
            f"- run id: `{manifest['run_id']}`",
            (
                "- minimum collection decision timestamp: "
                f"`{manifest['minimum_collection_decision_ts']}`"
            ),
            "- historical training markets: `90`",
            "- fresh calibration markets: `45`",
            "- fresh confirmatory markets: `60`",
            "- initial capture attempts: `120`",
            "- maximum capture attempts: `150`",
            "- collection started: `false`",
            "- rank scores execution eligible: `false`",
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
