"""Authorized fresh collection and outcome-blind 45/60 role assignment."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb import (
    FORBIDDEN_REGISTRY_FIELDS,
    _blocked_safety_fields,
    _capture_quality_audit,
    _descriptor,
    _execution_compatibility_audit,
    _finalization_quality_reasons,
    _find_fields,
    _load_json,
    _load_jsonl,
    _outcome_blind_corpus_role_audit,
    _require_sha256,
    _sha256_file,
    _verify_pin,
    _write_json,
    _write_jsonl,
    _write_text,
    validate_pairwise_action_advantage_lcb_feature_contract,
    validate_pairwise_action_advantage_lcb_protocol,
)
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_precollection_support import (
    _batch_preflight,
)
from bigan.v8.polymarket.training.hybrid_pairwise_frozen_ranker_calibration import (
    CALIBRATION_MARKET_COUNT,
    CALIBRATION_ROLE,
    CONFIRMATORY_MARKET_COUNT,
    CONFIRMATORY_ROLE,
    ROLE_ASSIGNMENT_SCHEMA_VERSION,
    TOTAL_FRESH_MARKET_COUNT,
)
from bigan.v8.polymarket.training.hybrid_pairwise_precollection_readiness import (
    PROTOCOL_SCHEMA_VERSION,
)

SCHEMA_PREFIX = "bigan-v8-hybrid-pairwise-fresh-collection"
START_GATE_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-start-gate-report-v1"
START_GATE_MANIFEST_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-start-gate-manifest-v1"
AUTHORIZATION_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-authorization-v1"
LAUNCH_PLAN_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-launch-plan-v1"
SUPPORT_REPORT_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-support-gate-report-v1"
SUPPORT_MANIFEST_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-support-gate-manifest-v1"
ROLE_REPORT_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-role-assignment-report-v1"
EXECUTION_REPORT_SCHEMA_VERSION = (
    f"{SCHEMA_PREFIX}-execution-compatibility-report-v1"
)
PRECOLLECTION_FREEZE_SCHEMA_VERSION = (
    "bigan-v8-hybrid-pairwise-precollection-freeze-manifest-v1"
)
READINESS_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-hybrid-pairwise-precollection-readiness-manifest-v1"
)
FINAL_QUARANTINE_SCHEMA_VERSION = (
    "bigan-v8-hybrid-pairwise-prior-lineage-quarantine-v1"
)
INITIAL_CAPTURE_ATTEMPT_COUNT = 120
MAXIMUM_CAPTURE_ATTEMPT_COUNT = 150
MAXIMUM_CONTINUATION_ATTEMPT_COUNT = 30
SUPPORT_ONLY_ROLE_BLOCKERS = frozenset(
    {
        "insufficient_quality_valid_unique_market_support",
        "role_market_count_mismatch",
    }
)


@dataclass(frozen=True, slots=True)
class HybridFreshCollectionStartGateConfig:
    """Pinned inputs for explicit, separately authorized collection start."""

    run_id: str
    output_dir: Path | str
    readiness_manifest_path: Path | str
    expected_readiness_manifest_sha256: str
    collector_script_path: Path | str
    expected_collector_script_sha256: str
    collector_git_commit: str
    precollection_freeze_manifest_path: Path | str | None = None
    expected_precollection_freeze_manifest_sha256: str | None = None
    final_prior_quarantine_path: Path | str | None = None
    expected_final_prior_quarantine_sha256: str | None = None
    authorization_path: Path | str | None = None
    expected_authorization_sha256: str | None = None
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(
            self.expected_readiness_manifest_sha256,
            name="readiness manifest SHA-256",
        )
        _require_sha256(
            self.expected_collector_script_sha256,
            name="collector script SHA-256",
        )
        if len(self.collector_git_commit) != 40 or any(
            character not in "0123456789abcdef"
            for character in self.collector_git_commit.lower()
        ):
            raise ValueError("collector_git_commit must be a 40-character hex digest")
        for name, path_value, digest in (
            (
                "precollection freeze manifest",
                self.precollection_freeze_manifest_path,
                self.expected_precollection_freeze_manifest_sha256,
            ),
            (
                "final prior quarantine",
                self.final_prior_quarantine_path,
                self.expected_final_prior_quarantine_sha256,
            ),
            (
                "collection authorization",
                self.authorization_path,
                self.expected_authorization_sha256,
            ),
        ):
            if (path_value is None) != (digest is None):
                raise ValueError(f"{name} path and SHA-256 must be provided together")
            if digest is not None:
                _require_sha256(digest, name=f"{name} SHA-256")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(
            self,
            "readiness_manifest_path",
            Path(self.readiness_manifest_path),
        )
        object.__setattr__(
            self,
            "collector_script_path",
            Path(self.collector_script_path),
        )
        for name in (
            "precollection_freeze_manifest_path",
            "final_prior_quarantine_path",
            "authorization_path",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))


@dataclass(frozen=True, slots=True)
class HybridFreshCollectionSupportGateConfig:
    """Pinned collector evidence for outcome-blind 45/60 role assignment."""

    run_id: str
    output_dir: Path | str
    collection_launch_plan_path: Path | str
    expected_collection_launch_plan_sha256: str
    precollection_freeze_manifest_path: Path | str
    expected_precollection_freeze_manifest_sha256: str
    batch_progress_pins: tuple[tuple[Path | str, str], ...]
    training_corpus_root: Path | str = Path("/Volumes/PHILIPS/v8")
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(
            self.expected_collection_launch_plan_sha256,
            name="collection launch plan SHA-256",
        )
        _require_sha256(
            self.expected_precollection_freeze_manifest_sha256,
            name="precollection freeze manifest SHA-256",
        )
        normalized_pins: list[tuple[Path, str]] = []
        for path, digest in self.batch_progress_pins:
            _require_sha256(digest, name="batch progress SHA-256")
            normalized_pins.append((Path(path), digest.lower()))
        if not normalized_pins:
            raise ValueError("at least one batch progress pin is required")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(
            self,
            "collection_launch_plan_path",
            Path(self.collection_launch_plan_path),
        )
        object.__setattr__(
            self,
            "precollection_freeze_manifest_path",
            Path(self.precollection_freeze_manifest_path),
        )
        object.__setattr__(
            self,
            "batch_progress_pins",
            tuple(normalized_pins),
        )
        object.__setattr__(
            self,
            "training_corpus_root",
            Path(self.training_corpus_root),
        )


def evaluate_hybrid_fresh_collection_start_gate(
    config: HybridFreshCollectionStartGateConfig,
) -> dict[str, Any]:
    """Write a launch plan only after terminal freeze and explicit authorization."""

    readiness_path = config.readiness_manifest_path.resolve()
    collector_script_path = config.collector_script_path.resolve()
    _verify_pin(
        readiness_path,
        config.expected_readiness_manifest_sha256,
        name="readiness manifest",
    )
    _verify_pin(
        collector_script_path,
        config.expected_collector_script_sha256,
        name="collector script",
    )
    readiness = _load_json(readiness_path)
    blockers: list[str] = []
    if _find_fields(readiness, FORBIDDEN_REGISTRY_FIELDS):
        blockers.append("readiness_manifest_forbidden_outcome_fields_present")
    if readiness.get("schema_version") != READINESS_MANIFEST_SCHEMA_VERSION:
        blockers.append("readiness_manifest_schema_mismatch")
    if readiness.get("precollection_readiness_passed") is not True:
        blockers.append("issue183_terminal_readiness_not_passed")
    if readiness.get("precollection_freeze_created") is not True:
        blockers.append("issue183_terminal_precollection_freeze_missing")
    if not _blocked_safety_valid(readiness):
        blockers.append("readiness_manifest_safety_contract_failed")

    freeze: dict[str, Any] | None = None
    freeze_descriptor: dict[str, str] | None = None
    source_protocol: dict[str, Any] | None = None
    hybrid_protocol: dict[str, Any] | None = None
    final_quarantine: dict[str, Any] | None = None
    quarantine_descriptor: dict[str, str] | None = None
    collector_contract: dict[str, Any] | None = None
    if config.precollection_freeze_manifest_path is None:
        blockers.append("issue183_terminal_precollection_freeze_not_provided")
    else:
        freeze_path = config.precollection_freeze_manifest_path.resolve()
        assert config.expected_precollection_freeze_manifest_sha256 is not None
        _verify_pin(
            freeze_path,
            config.expected_precollection_freeze_manifest_sha256,
            name="precollection freeze manifest",
        )
        freeze = _load_json(freeze_path)
        freeze_descriptor = _descriptor(freeze_path)
        blockers.extend(
            _freeze_validation_blockers(
                freeze=freeze,
                freeze_descriptor=freeze_descriptor,
                readiness=readiness,
            )
        )
        if not blockers:
            hybrid_descriptor = _verified_descriptor(
                freeze.get("hybrid_protocol"),
                name="hybrid protocol",
            )
            source_descriptor = _verified_descriptor(
                freeze.get("source_pairwise_protocol"),
                name="source pairwise protocol",
            )
            feature_descriptor = _verified_descriptor(
                freeze.get("source_feature_contract"),
                name="source feature contract",
            )
            _verified_descriptor(
                freeze.get("historical_registry_descriptor"),
                name="historical registry descriptor",
            )
            ranker_descriptor_info = _verified_descriptor(
                freeze.get("historical_ranker_descriptor"),
                name="historical ranker descriptor",
            )
            ranker_manifest_info = _verified_descriptor(
                freeze.get("historical_ranker_manifest"),
                name="historical ranker manifest",
            )
            hybrid_protocol = _load_json(Path(hybrid_descriptor["path"]))
            source_protocol = _load_json(Path(source_descriptor["path"]))
            ranker_descriptor = _load_json(
                Path(ranker_descriptor_info["path"])
            )
            ranker_manifest = _load_json(Path(ranker_manifest_info["path"]))
            model_descriptor = _verified_descriptor(
                ranker_descriptor.get("model"),
                name="historical ranker model",
            )
            validate_pairwise_action_advantage_lcb_protocol(source_protocol)
            validate_pairwise_action_advantage_lcb_feature_contract(
                _load_json(Path(feature_descriptor["path"])),
                expected_parent_protocol_sha256=source_descriptor["sha256"],
            )
            collector_contract = dict(source_protocol["collector_contract"])
            frozen_ranker = dict(
                hybrid_protocol.get("historical_ranker_freeze") or {}
            )
            if (
                hybrid_protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION
                or hybrid_protocol.get("source_pairwise_protocol_sha256")
                != source_descriptor["sha256"]
                or hybrid_protocol.get("source_feature_contract_sha256")
                != feature_descriptor["sha256"]
                or hybrid_protocol.get("collection_plan")
                != freeze.get("collection_plan")
                or hybrid_protocol.get("fresh_role_plan")
                != freeze.get("fresh_role_plan")
                or canonical_json_sha256(collector_contract)
                != (
                    hybrid_protocol.get("source_contract_hashes") or {}
                ).get("collector_contract_sha256")
                or frozen_ranker.get("descriptor_sha256")
                != ranker_descriptor_info["sha256"]
                or frozen_ranker.get("model_sha256")
                != model_descriptor["sha256"]
                or frozen_ranker.get("model_sha256")
                != ranker_descriptor.get("model_sha256")
                or frozen_ranker.get("model_sha256")
                != ranker_manifest.get("model_sha256")
                or frozen_ranker.get("dataset_hash")
                != ranker_descriptor.get("dataset_hash")
                or frozen_ranker.get("dataset_hash")
                != ranker_manifest.get("dataset_hash")
                or frozen_ranker.get("oof_dataset_hash")
                != ranker_manifest.get("oof_dataset_hash")
                or frozen_ranker.get("split_hash")
                != ranker_descriptor.get("split_hash")
                or frozen_ranker.get("split_hash")
                != ranker_manifest.get("split_hash")
                or frozen_ranker.get("model_config_hash")
                != ranker_descriptor.get("model_config_hash")
                or frozen_ranker.get("model_config_hash")
                != ranker_manifest.get("model_config_hash")
            ):
                blockers.append("frozen_collector_or_protocol_hash_drift")

    if config.final_prior_quarantine_path is None:
        blockers.append("final_prior_lineage_quarantine_not_provided")
    else:
        quarantine_path = config.final_prior_quarantine_path.resolve()
        assert config.expected_final_prior_quarantine_sha256 is not None
        _verify_pin(
            quarantine_path,
            config.expected_final_prior_quarantine_sha256,
            name="final prior quarantine",
        )
        final_quarantine = _load_json(quarantine_path)
        quarantine_descriptor = _descriptor(quarantine_path)
        blockers.extend(
            _quarantine_validation_blockers(
                quarantine=final_quarantine,
                descriptor=quarantine_descriptor,
                freeze=freeze,
            )
        )

    authorization: dict[str, Any] | None = None
    authorization_descriptor: dict[str, str] | None = None
    if config.authorization_path is None:
        blockers.append("explicit_issue185_collection_authorization_missing")
    else:
        authorization_path = config.authorization_path.resolve()
        assert config.expected_authorization_sha256 is not None
        _verify_pin(
            authorization_path,
            config.expected_authorization_sha256,
            name="collection authorization",
        )
        authorization = _load_json(authorization_path)
        authorization_descriptor = _descriptor(authorization_path)
        blockers.extend(
            _authorization_validation_blockers(
                authorization=authorization,
                freeze=freeze,
                freeze_descriptor=freeze_descriptor,
                quarantine_descriptor=quarantine_descriptor,
                collector_script_sha256=config.expected_collector_script_sha256,
                collector_git_commit=config.collector_git_commit,
            )
        )

    blockers = sorted(set(blockers))
    collection_start_allowed = not blockers
    run_dir = _prepare_run_dir(
        config.output_dir,
        config.run_id,
        overwrite=config.overwrite_existing,
    )
    launch_plan: dict[str, Any] | None = None
    launch_plan_path: Path | None = None
    if collection_start_allowed:
        assert freeze is not None
        assert freeze_descriptor is not None
        assert quarantine_descriptor is not None
        assert authorization is not None
        assert authorization_descriptor is not None
        assert collector_contract is not None
        launch_plan = _collection_launch_plan(
            run_id=config.run_id,
            freeze=freeze,
            freeze_descriptor=freeze_descriptor,
            quarantine_descriptor=quarantine_descriptor,
            authorization=authorization,
            authorization_descriptor=authorization_descriptor,
            collector_contract=collector_contract,
            collector_script_path=collector_script_path,
            collector_script_sha256=config.expected_collector_script_sha256,
            collector_git_commit=config.collector_git_commit,
        )
        launch_plan_path = run_dir / "hybrid_pairwise_fresh_collection_launch_plan.json"
        _write_json(launch_plan_path, launch_plan)

    report = {
        "schema_version": START_GATE_SCHEMA_VERSION,
        "run_id": config.run_id,
        "status": (
            "authorized_collection_launch_plan_ready"
            if collection_start_allowed
            else "blocked_fail_closed"
        ),
        "readiness_manifest": _descriptor(readiness_path),
        "hybrid_precollection_freeze": freeze_descriptor,
        "final_prior_lineage_quarantine": quarantine_descriptor,
        "collection_authorization": authorization_descriptor,
        "collector_script": _descriptor(collector_script_path),
        "collector_git_commit": config.collector_git_commit,
        "collection_start_allowed": collection_start_allowed,
        "collection_start_command_generated": launch_plan is not None,
        "collector_execution_attempted": False,
        "launch_plan": (
            _descriptor(launch_plan_path) if launch_plan_path is not None else None
        ),
        "initial_capture_attempt_count": INITIAL_CAPTURE_ATTEMPT_COUNT,
        "maximum_total_capture_attempt_count": MAXIMUM_CAPTURE_ATTEMPT_COUNT,
        "blocking_reason_codes": blockers,
        "labels_or_outcomes_opened_for_start_gate": False,
        "oof_validation_or_pnl_used_for_start_gate": False,
        "model_training_or_prediction_attempted": False,
        "source_scores_mutated": False,
        "execution_thresholds_mutated": False,
        **_hybrid_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "hybrid_pairwise_fresh_collection_start_gate_report.json"
    markdown_path = run_dir / "hybrid_pairwise_fresh_collection_start_gate_report.md"
    _write_json(report_path, report)
    _write_text(markdown_path, _start_gate_markdown(report))
    manifest = {
        "schema_version": START_GATE_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "start_gate_report": _descriptor(report_path),
        "start_gate_markdown": _descriptor(markdown_path),
        "launch_plan": (
            _descriptor(launch_plan_path) if launch_plan_path is not None else None
        ),
        "collection_start_allowed": collection_start_allowed,
        "collection_start_command_generated": launch_plan is not None,
        "collector_execution_attempted": False,
        "blocking_reason_codes": blockers,
        "labels_or_outcomes_opened_for_start_gate": False,
        **_hybrid_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "hybrid_pairwise_fresh_collection_start_gate_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "markdown_path": markdown_path,
        "markdown_sha256": _sha256_file(markdown_path),
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "launch_plan_path": launch_plan_path,
        "launch_plan_sha256": (
            _sha256_file(launch_plan_path) if launch_plan_path is not None else None
        ),
        "report": report,
        "manifest": manifest,
        "launch_plan": launch_plan,
    }


def run_hybrid_fresh_collection_support_gate(
    config: HybridFreshCollectionSupportGateConfig,
) -> dict[str, Any]:
    """Evaluate 120/150 support and assign earliest valid 45/60 markets."""

    launch_plan_path = config.collection_launch_plan_path.resolve()
    freeze_path = config.precollection_freeze_manifest_path.resolve()
    _verify_pin(
        launch_plan_path,
        config.expected_collection_launch_plan_sha256,
        name="collection launch plan",
    )
    _verify_pin(
        freeze_path,
        config.expected_precollection_freeze_manifest_sha256,
        name="precollection freeze manifest",
    )
    launch_plan = _load_json(launch_plan_path)
    freeze = _load_json(freeze_path)
    launch_blockers = _launch_plan_validation_blockers(
        launch_plan=launch_plan,
        freeze=freeze,
        freeze_descriptor=_descriptor(freeze_path),
    )
    source_protocol_descriptor = _verified_descriptor(
        freeze.get("source_pairwise_protocol"),
        name="source pairwise protocol",
    )
    source_protocol = _load_json(Path(source_protocol_descriptor["path"]))
    validate_pairwise_action_advantage_lcb_protocol(source_protocol)
    collector_contract = dict(source_protocol["collector_contract"])
    training_root = config.training_corpus_root.expanduser().resolve()
    if str(training_root) != str(
        Path(str(collector_contract["training_corpus_root"])).expanduser().resolve()
    ):
        launch_blockers.append("training_corpus_root_drift")

    (
        unique_batch_pins,
        duplicate_excluded_inputs,
        batch_preflight,
    ) = _batch_preflight(config.batch_progress_pins)
    attempted_capture_count = int(batch_preflight["attempted_capture_count"])
    preflight_blockers = list(batch_preflight["blocking_reason_codes"])
    preflight_blockers.extend(launch_blockers)
    if attempted_capture_count > MAXIMUM_CAPTURE_ATTEMPT_COUNT:
        preflight_blockers.append("frozen_maximum_capture_attempt_count_exceeded")
    preflight_blockers.extend(
        _batch_launch_plan_blockers(
            pins=unique_batch_pins,
            launch_plan=launch_plan,
        )
    )
    preflight_blockers = sorted(set(preflight_blockers))

    run_dir = _prepare_run_dir(
        config.output_dir,
        config.run_id,
        overwrite=config.overwrite_existing,
    )
    role_result: dict[str, Any] | None = None
    if (
        attempted_capture_count >= INITIAL_CAPTURE_ATTEMPT_COUNT
        and not preflight_blockers
    ):
        role_result = _assign_hybrid_fresh_roles(
            run_id=config.run_id,
            run_dir=run_dir,
            freeze=freeze,
            freeze_path=freeze_path,
            launch_plan=launch_plan,
            launch_plan_path=launch_plan_path,
            batch_progress_pins=unique_batch_pins,
            collector_contract=collector_contract,
            training_root=training_root,
        )

    decision = _support_decision(
        attempted_capture_count=attempted_capture_count,
        preflight_blockers=preflight_blockers,
        role_report=(role_result or {}).get("report"),
    )
    report = {
        "schema_version": SUPPORT_REPORT_SCHEMA_VERSION,
        "run_id": config.run_id,
        "status": decision["status"],
        "target_valid_unique_market_count": TOTAL_FRESH_MARKET_COUNT,
        "initial_capture_attempt_count": INITIAL_CAPTURE_ATTEMPT_COUNT,
        "maximum_total_capture_attempt_count": MAXIMUM_CAPTURE_ATTEMPT_COUNT,
        "attempted_capture_count": attempted_capture_count,
        "remaining_initial_capture_attempt_count": max(
            0,
            INITIAL_CAPTURE_ATTEMPT_COUNT - attempted_capture_count,
        ),
        "remaining_frozen_capture_attempt_count": max(
            0,
            MAXIMUM_CAPTURE_ATTEMPT_COUNT - attempted_capture_count,
        ),
        "role_assignment_attempted": role_result is not None,
        "role_assignment_ready": bool(
            role_result and role_result["report"]["role_assignment_ready"]
        ),
        "selected_market_count": int(
            ((role_result or {}).get("report") or {}).get("selected_market_count")
            or 0
        ),
        "role_market_counts": dict(
            ((role_result or {}).get("report") or {}).get("role_market_counts")
            or {}
        ),
        "role_assignment_blocking_reason_codes": list(
            ((role_result or {}).get("report") or {}).get("blocking_reason_codes")
            or []
        ),
        "support_only_role_blockers": sorted(SUPPORT_ONLY_ROLE_BLOCKERS),
        "support_only_failure": decision["support_only_failure"],
        "continuation_allowed": decision["continuation_allowed"],
        "continuation_required": decision["continuation_required"],
        "continuation_attempt_count": decision["continuation_attempt_count"],
        "continuation_reason_codes": decision["continuation_reason_codes"],
        "blocking_reason_codes": decision["blocking_reason_codes"],
        "frozen_maximum_enforced": True,
        "maximum_continuation_attempt_count": MAXIMUM_CONTINUATION_ATTEMPT_COUNT,
        "duplicate_excluded_input_count": len(duplicate_excluded_inputs),
        "duplicate_excluded_inputs": duplicate_excluded_inputs,
        "unique_batch_progress_count": len(unique_batch_pins),
        "batch_ids": batch_preflight["batch_ids"],
        "capture_run_id_count": batch_preflight["capture_run_id_count"],
        "forbidden_batch_field_paths": batch_preflight[
            "forbidden_batch_field_paths"
        ],
        "collector_batch_error_count": batch_preflight[
            "collector_batch_error_count"
        ],
        "labels_or_outcomes_opened_for_support_or_continuation": False,
        "settlement_pnl_opened_for_support_or_continuation": False,
        "oracle_actions_opened_for_support_or_continuation": False,
        "uses_oof_validation_confirmatory_or_pnl_for_continuation": False,
        "source_scores_mutated": False,
        "execution_thresholds_mutated": False,
        "confirmatory_labels_opened": False,
        **_hybrid_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "hybrid_pairwise_fresh_collection_support_gate_report.json"
    markdown_path = run_dir / "hybrid_pairwise_fresh_collection_support_gate_report.md"
    _write_json(report_path, report)
    _write_text(markdown_path, _support_markdown(report))

    continuation_manifest: dict[str, Any] | None = None
    continuation_manifest_path: Path | None = None
    if report["continuation_allowed"]:
        continuation_manifest = {
            "schema_version": f"{SCHEMA_PREFIX}-continuation-manifest-v1",
            "run_id": config.run_id,
            "collection_launch_plan": _descriptor(launch_plan_path),
            "hybrid_precollection_freeze": _descriptor(freeze_path),
            "attempted_capture_count": attempted_capture_count,
            "continuation_attempt_count": report["continuation_attempt_count"],
            "maximum_total_capture_attempt_count": MAXIMUM_CAPTURE_ATTEMPT_COUNT,
            "continuation_is_support_only": report["support_only_failure"],
            "continuation_reason_codes": report["continuation_reason_codes"],
            "labels_or_outcomes_opened_for_continuation": False,
            "oof_validation_confirmatory_or_pnl_used_for_continuation": False,
            **_hybrid_safety_fields(),
        }
        continuation_manifest["continuation_id"] = canonical_json_sha256(
            continuation_manifest
        )
        continuation_manifest_path = (
            run_dir / "hybrid_pairwise_fresh_collection_continuation_manifest.json"
        )
        _write_json(continuation_manifest_path, continuation_manifest)

    manifest = {
        "schema_version": SUPPORT_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "collection_launch_plan": _descriptor(launch_plan_path),
        "hybrid_precollection_freeze": _descriptor(freeze_path),
        "batch_progress_inputs": [
            _descriptor(path.resolve()) for path, _ in unique_batch_pins
        ],
        "support_gate_report": _descriptor(report_path),
        "support_gate_markdown": _descriptor(markdown_path),
        "role_assignment_manifest": (
            None
            if role_result is None
            else _descriptor(role_result["manifest_path"])
        ),
        "role_assignment_report": (
            None
            if role_result is None
            else _descriptor(role_result["report_path"])
        ),
        "continuation_manifest": (
            None
            if continuation_manifest_path is None
            else _descriptor(continuation_manifest_path)
        ),
        "attempted_capture_count": attempted_capture_count,
        "selected_market_count": report["selected_market_count"],
        "continuation_allowed": report["continuation_allowed"],
        "continuation_attempt_count": report["continuation_attempt_count"],
        "blocking_reason_codes": report["blocking_reason_codes"],
        "labels_or_outcomes_opened_for_support_or_continuation": False,
        "confirmatory_labels_opened": False,
        **_hybrid_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "hybrid_pairwise_fresh_collection_support_gate_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "markdown_path": markdown_path,
        "markdown_sha256": _sha256_file(markdown_path),
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "continuation_manifest_path": continuation_manifest_path,
        "continuation_manifest": continuation_manifest,
        "role_assignment_result": role_result,
        "report": report,
        "manifest": manifest,
    }


def _assign_hybrid_fresh_roles(
    *,
    run_id: str,
    run_dir: Path,
    freeze: dict[str, Any],
    freeze_path: Path,
    launch_plan: dict[str, Any],
    launch_plan_path: Path,
    batch_progress_pins: tuple[tuple[Path, str], ...],
    collector_contract: dict[str, Any],
    training_root: Path,
) -> dict[str, Any]:
    quarantine_descriptor = _verified_descriptor(
        freeze.get("final_prior_lineage_quarantine"),
        name="final prior quarantine",
    )
    quarantine = _load_json(Path(quarantine_descriptor["path"]))
    prior_market_ids = {
        str(value) for value in quarantine.get("prior_market_ids") or []
    }
    if (
        not prior_market_ids
        or "" in prior_market_ids
        or canonical_json_sha256(list(quarantine["prior_market_ids"]))
        != quarantine.get("prior_market_ids_sha256")
    ):
        raise ValueError("final prior quarantine market identity is incomplete")
    minimum_collection_decision_ts = int(
        freeze.get("minimum_collection_decision_ts") or 0
    )
    candidates: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    role_blockers: list[str] = []
    batch_descriptors: list[dict[str, str]] = []
    for batch_ordinal, (path, expected_sha256) in enumerate(batch_progress_pins):
        batch_path = path.resolve()
        batch = _load_json(batch_path)
        captures = [dict(row) for row in batch.get("captures") or []]
        finalizations = [dict(row) for row in batch.get("finalizations") or []]
        finalization_by_run_id = {
            str(row.get("run_id") or ""): row for row in finalizations
        }
        for capture in captures:
            capture_row = {
                **capture,
                "source_batch_ordinal": batch_ordinal,
                "source_batch_id": str(batch.get("batch_id") or ""),
                "source_batch_progress_sha256": expected_sha256,
            }
            finalization = finalization_by_run_id.get(
                str(capture.get("run_id") or "")
            )
            audit = _capture_quality_audit(
                capture_row,
                collector_contract=collector_contract,
                finalization=finalization,
            )
            if audit["reason_codes"]:
                excluded.append(audit)
                continue
            finalization_reasons = _finalization_quality_reasons(finalization)
            if finalization_reasons:
                audit["reason_codes"] = finalization_reasons
                excluded.append(audit)
                role_blockers.append(
                    "quality_valid_capture_finalization_incomplete"
                )
                continue
            assert finalization is not None
            corpus_dir = Path(
                str(finalization["exported_training_corpus_dir"])
            ).expanduser().resolve()
            corpus_reasons: list[str] = []
            if not corpus_dir.is_relative_to(training_root):
                corpus_reasons.append("exported_corpus_outside_training_root")
            elif not corpus_dir.is_dir():
                corpus_reasons.append("exported_corpus_directory_missing")
            corpus_audit: dict[str, Any] | None = None
            execution_audit: dict[str, Any] | None = None
            if not corpus_reasons:
                corpus_audit = _outcome_blind_corpus_role_audit(
                    corpus_dir=corpus_dir,
                    prior_market_ids=prior_market_ids,
                    minimum_decision_ts=minimum_collection_decision_ts,
                )
                corpus_reasons.extend(corpus_audit["reason_codes"])
            if not corpus_reasons:
                execution_audit = _execution_compatibility_audit(
                    corpus_dir=corpus_dir,
                    collector_contract=collector_contract,
                )
                corpus_reasons.extend(
                    execution_audit["blocking_reason_codes"]
                )
            if corpus_reasons:
                audit["reason_codes"] = sorted(set(corpus_reasons))
                excluded.append(audit)
                continue
            assert corpus_audit is not None
            assert execution_audit is not None
            feature_rows = _load_jsonl(
                Path(str(corpus_audit["feature_rows"]["path"]))
            )
            slugs = {str(row.get("slug") or "") for row in feature_rows}
            if len(slugs) != 1 or "" in slugs:
                audit["reason_codes"] = ["feature_market_slug_incomplete"]
                excluded.append(audit)
                continue
            candidates.append(
                {
                    **audit,
                    "market_id": str(corpus_audit["market_id"]),
                    "market_slug": next(iter(slugs)),
                    "minimum_decision_ts": int(
                        corpus_audit["minimum_decision_ts"]
                    ),
                    "maximum_decision_ts": int(
                        corpus_audit["maximum_decision_ts"]
                    ),
                    "decision_row_count": int(
                        corpus_audit["decision_row_count"]
                    ),
                    "source_corpus_dir": str(corpus_dir),
                    "corpus_manifest": corpus_audit["corpus_manifest"],
                    "feature_rows": corpus_audit["feature_rows"],
                    "execution_compatibility_audit": execution_audit,
                    "reason_codes": [],
                }
            )
        batch_descriptors.append(_descriptor(batch_path))

    candidates.sort(
        key=lambda row: (
            int(row["minimum_decision_ts"]),
            int(row["maximum_decision_ts"]),
            str(row["market_id"]),
            str(row["capture_run_id"]),
        )
    )
    unique_candidates: list[dict[str, Any]] = []
    seen_market_ids: set[str] = set()
    seen_corpus_dirs: set[str] = set()
    for candidate in candidates:
        market_id = str(candidate["market_id"])
        corpus_dir = str(candidate["source_corpus_dir"])
        reasons: list[str] = []
        if market_id in prior_market_ids:
            reasons.append("prior_market_overlap_detected")
        if market_id in seen_market_ids:
            reasons.append("duplicate_market_identity")
        if corpus_dir in seen_corpus_dirs:
            reasons.append("duplicate_exported_corpus_path")
        if reasons:
            excluded.append({**candidate, "reason_codes": reasons})
            continue
        seen_market_ids.add(market_id)
        seen_corpus_dirs.add(corpus_dir)
        unique_candidates.append(candidate)

    selected = []
    freeze_descriptor = _descriptor(freeze_path)
    for index, candidate in enumerate(
        unique_candidates[:TOTAL_FRESH_MARKET_COUNT]
    ):
        selection_rank = index + 1
        selected.append(
            {
                **candidate,
                "selected": True,
                "selection_rank": selection_rank,
                "role": (
                    CALIBRATION_ROLE
                    if selection_rank <= CALIBRATION_MARKET_COUNT
                    else CONFIRMATORY_ROLE
                ),
                "source_precollection_freeze_sha256": freeze_descriptor[
                    "sha256"
                ],
                "source_final_quarantine_sha256": quarantine_descriptor[
                    "sha256"
                ],
                "source_collection_launch_plan_sha256": _sha256_file(
                    launch_plan_path
                ),
                "execution_compatibility_validated_before_label_access": True,
                "labels_or_outcomes_opened_for_role_assignment": False,
                "reason_codes": [],
            }
        )
    for candidate in unique_candidates[TOTAL_FRESH_MARKET_COUNT:]:
        excluded.append(
            {**candidate, "reason_codes": ["selection_target_already_met"]}
        )

    selected_market_ids = {str(row["market_id"]) for row in selected}
    role_counts = Counter(str(row["role"]) for row in selected)
    role_sets = {
        role: {
            str(row["market_id"]) for row in selected if row["role"] == role
        }
        for role in (CALIBRATION_ROLE, CONFIRMATORY_ROLE)
    }
    role_overlap = role_sets[CALIBRATION_ROLE] & role_sets[CONFIRMATORY_ROLE]
    prior_overlap = selected_market_ids & prior_market_ids
    chronology_passed = _role_chronology_passed(selected)
    if len(selected) != TOTAL_FRESH_MARKET_COUNT:
        role_blockers.append("insufficient_quality_valid_unique_market_support")
    if role_counts != Counter(
        {
            CALIBRATION_ROLE: CALIBRATION_MARKET_COUNT,
            CONFIRMATORY_ROLE: CONFIRMATORY_MARKET_COUNT,
        }
    ):
        role_blockers.append("role_market_count_mismatch")
    if role_overlap:
        role_blockers.append("role_market_overlap_detected")
    if prior_overlap:
        role_blockers.append("prior_market_overlap_detected")
    if len(selected_market_ids) != len(selected):
        role_blockers.append("selected_market_identity_not_unique")
    if not chronology_passed:
        role_blockers.append("calibration_confirmatory_chronology_failed")
    role_blockers = sorted(set(role_blockers))
    role_assignment_ready = not role_blockers

    selected_path = run_dir / "hybrid_pairwise_fresh_role_assignment_rows.jsonl"
    excluded_path = (
        run_dir / "hybrid_pairwise_fresh_role_assignment_excluded_rows.jsonl"
    )
    _write_jsonl(selected_path, selected)
    _write_jsonl(excluded_path, excluded)
    execution_report = {
        "schema_version": EXECUTION_REPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "selected_market_count": len(selected),
        "selected_decision_row_count": sum(
            int(row["execution_compatibility_audit"]["decision_row_count"])
            for row in selected
        ),
        "execution_compatible_decision_row_count": sum(
            int(
                row["execution_compatibility_audit"][
                    "execution_compatible_row_count"
                ]
            )
            for row in selected
        ),
        "selected_market_failure_count": sum(
            int(
                bool(
                    row["execution_compatibility_audit"][
                        "blocking_reason_codes"
                    ]
                )
            )
            for row in selected
        ),
        "market_audits": [
            {
                "market_id": row["market_id"],
                "market_slug": row["market_slug"],
                "selection_rank": row["selection_rank"],
                "role": row["role"],
                **row["execution_compatibility_audit"],
            }
            for row in selected
        ],
        "labels_or_outcomes_opened": False,
        **_hybrid_safety_fields(),
    }
    execution_report_path = (
        run_dir / "hybrid_pairwise_fresh_execution_compatibility_report.json"
    )
    execution_markdown_path = (
        run_dir / "hybrid_pairwise_fresh_execution_compatibility_report.md"
    )
    _write_json(execution_report_path, execution_report)
    _write_text(
        execution_markdown_path,
        _execution_markdown(execution_report),
    )
    report = {
        "schema_version": ROLE_REPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": (
            "outcome_blind_role_assignment_ready"
            if role_assignment_ready
            else "blocked_fail_closed"
        ),
        "role_assignment_ready": role_assignment_ready,
        "blocking_reason_codes": role_blockers,
        "attempted_capture_count": sum(
            int(_load_json(path.resolve()).get("capture_count") or 0)
            for path, _ in batch_progress_pins
        ),
        "quality_valid_unique_market_count": len(unique_candidates),
        "selected_market_count": len(selected),
        "excluded_capture_count": len(excluded),
        "role_market_counts": dict(sorted(role_counts.items())),
        "role_market_overlap_count": len(role_overlap),
        "prior_market_overlap_count": len(prior_overlap),
        "chronology_validation_passed": chronology_passed,
        "excluded_reason_distribution": dict(
            sorted(
                Counter(
                    reason
                    for row in excluded
                    for reason in row.get("reason_codes") or []
                ).items()
            )
        ),
        "role_assignment_method": (
            "earliest_quality_valid_unique_markets_chronological_outcome_blind"
        ),
        "role_assignment_uses_outcomes": False,
        "role_assignment_uses_settlement_pnl": False,
        "role_assignment_uses_oracle_actions": False,
        "labels_or_outcomes_opened_for_role_assignment": False,
        "execution_compatibility_validated_before_label_access": True,
        "model_fit_started": False,
        "confirmatory_validation_started": False,
        "confirmatory_labels_opened": False,
        **_hybrid_safety_fields(),
    }
    report_path = run_dir / "hybrid_pairwise_fresh_role_assignment_report.json"
    markdown_path = run_dir / "hybrid_pairwise_fresh_role_assignment_report.md"
    _write_json(report_path, report)
    _write_text(markdown_path, _role_markdown(report))
    manifest = {
        "schema_version": ROLE_ASSIGNMENT_SCHEMA_VERSION,
        "run_id": run_id,
        "role_assignment_ready": role_assignment_ready,
        "blocking_reason_codes": role_blockers,
        "selected_market_count": len(selected),
        "hybrid_precollection_freeze": freeze_descriptor,
        "final_prior_lineage_quarantine": quarantine_descriptor,
        "collection_launch_plan": _descriptor(launch_plan_path),
        "collection_launch_plan_id": launch_plan["launch_plan_id"],
        "batch_progress_inputs": batch_descriptors,
        "selected_rows": _descriptor(selected_path),
        "excluded_rows": _descriptor(excluded_path),
        "report": _descriptor(report_path),
        "execution_compatibility_report": _descriptor(
            execution_report_path
        ),
        "role_market_counts": dict(sorted(role_counts.items())),
        "selected_market_ids_sha256": canonical_json_sha256(
            [str(row["market_id"]) for row in selected]
        ),
        "prior_market_overlap_count": len(prior_overlap),
        "role_market_overlap_count": len(role_overlap),
        "chronology_validation_passed": chronology_passed,
        "labels_or_outcomes_opened_for_role_assignment": False,
        "model_fit_started": False,
        "confirmatory_validation_started": False,
        "confirmatory_labels_opened": False,
        **_hybrid_safety_fields(),
    }
    manifest["role_assignment_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "hybrid_pairwise_fresh_role_assignment_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "selected_rows_path": selected_path,
        "excluded_rows_path": excluded_path,
        "report_path": report_path,
        "markdown_path": markdown_path,
        "manifest_path": manifest_path,
        "execution_report_path": execution_report_path,
        "report": report,
        "manifest": manifest,
    }


def _freeze_validation_blockers(
    *,
    freeze: dict[str, Any],
    freeze_descriptor: dict[str, str],
    readiness: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if _find_fields(freeze, FORBIDDEN_REGISTRY_FIELDS):
        blockers.append("precollection_freeze_forbidden_outcome_fields_present")
    if freeze.get("schema_version") != PRECOLLECTION_FREEZE_SCHEMA_VERSION:
        blockers.append("precollection_freeze_schema_mismatch")
    if readiness.get("precollection_freeze_manifest") != freeze_descriptor:
        blockers.append("readiness_precollection_freeze_identity_mismatch")
    collection = dict(freeze.get("collection_plan") or {})
    if (
        collection.get("target_valid_unique_market_count")
        != TOTAL_FRESH_MARKET_COUNT
        or collection.get("initial_capture_attempt_count")
        != INITIAL_CAPTURE_ATTEMPT_COUNT
        or collection.get("maximum_total_capture_attempt_count")
        != MAXIMUM_CAPTURE_ATTEMPT_COUNT
        or collection.get("bounded_continuation_only_for_support_shortfall")
        is not True
        or collection.get("replacement_only_for_pre_label_capture_quality_failure")
        is not True
    ):
        blockers.append("precollection_freeze_collection_plan_mismatch")
    if freeze.get("ranker_retraining_allowed") is not False:
        blockers.append("precollection_freeze_ranker_retraining_not_disabled")
    if freeze.get("ranker_score_mutation_allowed") is not False:
        blockers.append("precollection_freeze_score_mutation_not_disabled")
    if freeze.get("labels_or_outcomes_opened_for_role_assignment") is not False:
        blockers.append("precollection_freeze_role_assignment_leakage_boundary_failed")
    if not _blocked_safety_valid(freeze):
        blockers.append("precollection_freeze_safety_contract_failed")
    return blockers


def _quarantine_validation_blockers(
    *,
    quarantine: dict[str, Any],
    descriptor: dict[str, str],
    freeze: dict[str, Any] | None,
) -> list[str]:
    blockers: list[str] = []
    if _find_fields(quarantine, FORBIDDEN_REGISTRY_FIELDS):
        blockers.append("final_quarantine_forbidden_outcome_fields_present")
    if quarantine.get("schema_version") != FINAL_QUARANTINE_SCHEMA_VERSION:
        blockers.append("final_quarantine_schema_mismatch")
    if (
        quarantine.get("status") != "prior_lineage_complete"
        or quarantine.get("final") is not True
        or quarantine.get("active_prior_lineage_complete") is not True
        or quarantine.get("includes_issue175_through_issue179") is not True
    ):
        blockers.append("final_quarantine_not_terminal")
    market_ids = [str(value) for value in quarantine.get("prior_market_ids") or []]
    if (
        not market_ids
        or "" in market_ids
        or len(market_ids) != len(set(market_ids))
        or canonical_json_sha256(market_ids)
        != quarantine.get("prior_market_ids_sha256")
    ):
        blockers.append("final_quarantine_market_identity_invalid")
    if not _blocked_safety_valid(quarantine):
        blockers.append("final_quarantine_safety_contract_failed")
    if freeze is not None:
        if freeze.get("final_prior_lineage_quarantine") != descriptor:
            blockers.append("freeze_final_quarantine_identity_mismatch")
        if int(freeze.get("minimum_collection_decision_ts") or 0) <= int(
            quarantine.get("maximum_prior_decision_ts") or 0
        ):
            blockers.append("fresh_collection_boundary_not_after_quarantine")
    return blockers


def _authorization_validation_blockers(
    *,
    authorization: dict[str, Any],
    freeze: dict[str, Any] | None,
    freeze_descriptor: dict[str, str] | None,
    quarantine_descriptor: dict[str, str] | None,
    collector_script_sha256: str,
    collector_git_commit: str,
) -> list[str]:
    blockers: list[str] = []
    if _find_fields(authorization, FORBIDDEN_REGISTRY_FIELDS):
        blockers.append("collection_authorization_forbidden_outcome_fields_present")
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA_VERSION:
        blockers.append("collection_authorization_schema_mismatch")
    if authorization.get("issue_number") != 185:
        blockers.append("collection_authorization_issue_mismatch")
    if authorization.get("authorized") is not True:
        blockers.append("collection_authorization_not_granted")
    if authorization.get("read_only_public_provider") is not True:
        blockers.append("collection_authorization_not_read_only")
    if authorization.get("collector_execution_authorized") is not True:
        blockers.append("collector_execution_not_explicitly_authorized")
    if authorization.get("hybrid_precollection_freeze") != freeze_descriptor:
        blockers.append("authorization_precollection_freeze_identity_mismatch")
    if authorization.get("final_prior_lineage_quarantine") != quarantine_descriptor:
        blockers.append("authorization_final_quarantine_identity_mismatch")
    if authorization.get("collector_script_sha256") != collector_script_sha256:
        blockers.append("authorization_collector_script_hash_mismatch")
    if authorization.get("collector_git_commit") != collector_git_commit:
        blockers.append("authorization_collector_git_commit_mismatch")
    collection = dict(authorization.get("collection_plan") or {})
    if (
        collection.get("market_family") != "btc_updown_5m"
        or collection.get("initial_capture_attempt_count")
        != INITIAL_CAPTURE_ATTEMPT_COUNT
        or collection.get("maximum_total_capture_attempt_count")
        != MAXIMUM_CAPTURE_ATTEMPT_COUNT
    ):
        blockers.append("authorization_collection_plan_mismatch")
    if freeze is not None and int(
        authorization.get("authorization_created_at_ts") or 0
    ) <= int(freeze.get("freeze_created_at_ts") or 0):
        blockers.append("authorization_predates_precollection_freeze")
    if (
        not str(authorization.get("collection_output_dir") or "")
        or not str(authorization.get("market_identity_cache_path") or "")
    ):
        blockers.append("authorization_collection_paths_missing")
    if freeze is not None:
        source_descriptor = freeze.get("source_pairwise_protocol") or {}
        if not isinstance(source_descriptor, dict):
            blockers.append("authorization_source_protocol_missing")
    if not _hybrid_safety_valid(authorization):
        blockers.append("collection_authorization_safety_contract_failed")
    return blockers


def _collection_launch_plan(
    *,
    run_id: str,
    freeze: dict[str, Any],
    freeze_descriptor: dict[str, str],
    quarantine_descriptor: dict[str, str],
    authorization: dict[str, Any],
    authorization_descriptor: dict[str, str],
    collector_contract: dict[str, Any],
    collector_script_path: Path,
    collector_script_sha256: str,
    collector_git_commit: str,
) -> dict[str, Any]:
    collection_output_dir = str(
        Path(str(authorization["collection_output_dir"]))
        .expanduser()
        .resolve()
    )
    training_root = str(
        Path(str(collector_contract["training_corpus_root"]))
        .expanduser()
        .resolve()
    )
    market_identity_cache_path = str(
        Path(str(authorization["market_identity_cache_path"]))
        .expanduser()
        .resolve()
    )
    batch_id_prefix = str(
        authorization.get("batch_id_prefix")
        or f"issue185-hybrid-pairwise-fresh-{run_id}"
    )
    command_argv = [
        "python",
        str(collector_script_path),
        "--batch-id",
        f"{batch_id_prefix}-initial-120",
        "--output-dir",
        collection_output_dir,
        "--round-count",
        str(INITIAL_CAPTURE_ATTEMPT_COUNT),
        "--market-family",
        "btc_updown_5m",
        "--public-provider-timeout-seconds",
        str(collector_contract["public_provider_timeout_seconds"]),
        "--public-provider-http-timeout-seconds",
        str(collector_contract["public_provider_http_timeout_seconds"]),
        "--orderbook-snapshot-interval-seconds",
        str(collector_contract["orderbook_snapshot_interval_seconds"]),
        "--orderbook-ws-initial-complete-book-timeout-seconds",
        str(
            collector_contract[
                "orderbook_ws_initial_complete_book_timeout_seconds"
            ]
        ),
        "--rest-orderbook-fallback-collection-seconds",
        str(collector_contract["rest_orderbook_fallback_collection_seconds"]),
        "--settlement-poll-interval-seconds",
        str(collector_contract["settlement_poll_interval_seconds"]),
        "--settlement-grace-seconds",
        str(collector_contract["settlement_grace_seconds"]),
        "--training-corpus-root",
        training_root,
        "--chainlink-rtds-warmup-seconds",
        str(collector_contract["chainlink_rtds_warmup_seconds"]),
        "--chainlink-rtds-stale-reconnect-seconds",
        str(collector_contract["chainlink_rtds_stale_reconnect_seconds"]),
        "--market-identity-cache-path",
        market_identity_cache_path,
        "--gamma-market-identity-prefetch-round-count",
        str(collector_contract["gamma_market_identity_prefetch_round_count"]),
        "--market-identity-cache-max-age-seconds",
        str(collector_contract["market_identity_cache_max_age_seconds"]),
        "--clob-identity-revalidation-max-attempts",
        str(
            collector_contract[
                "market_identity_cache_clob_revalidation_max_attempts"
            ]
        ),
        "--clob-identity-revalidation-retry-seconds",
        str(
            collector_contract[
                "market_identity_cache_clob_revalidation_retry_seconds"
            ]
        ),
        "--feature-enrichment-max-attempts",
        str(collector_contract["feature_enrichment_max_attempts"]),
    ]
    plan = {
        "schema_version": LAUNCH_PLAN_SCHEMA_VERSION,
        "run_id": run_id,
        "candidate_lineage": freeze["candidate_lineage"],
        "hybrid_precollection_freeze": freeze_descriptor,
        "final_prior_lineage_quarantine": quarantine_descriptor,
        "collection_authorization": authorization_descriptor,
        "collector_script": {
            "path": str(collector_script_path),
            "sha256": collector_script_sha256,
        },
        "collector_git_commit": collector_git_commit,
        "minimum_collection_decision_ts": freeze[
            "minimum_collection_decision_ts"
        ],
        "market_family": "btc_updown_5m",
        "initial_capture_attempt_count": INITIAL_CAPTURE_ATTEMPT_COUNT,
        "maximum_total_capture_attempt_count": MAXIMUM_CAPTURE_ATTEMPT_COUNT,
        "maximum_continuation_attempt_count": (
            MAXIMUM_CONTINUATION_ATTEMPT_COUNT
        ),
        "batch_id_prefix": batch_id_prefix,
        "collection_output_dir": collection_output_dir,
        "training_corpus_root": training_root,
        "market_identity_cache_path": market_identity_cache_path,
        "collector_contract": collector_contract,
        "collector_contract_sha256": canonical_json_sha256(
            collector_contract
        ),
        "initial_collection_command_argv": command_argv,
        "collector_execution_attempted": False,
        "collection_start_allowed": True,
        "read_only_public_provider": True,
        "orderbook_websocket_primary": True,
        "causal_rest_orderbook_fallback_only": True,
        "per_round_raw_evidence_required": True,
        "asynchronous_settlement_required": True,
        "labels_or_outcomes_opened_for_launch_plan": False,
        **_hybrid_safety_fields(),
    }
    plan["launch_plan_id"] = canonical_json_sha256(plan)
    return plan


def _launch_plan_validation_blockers(
    *,
    launch_plan: dict[str, Any],
    freeze: dict[str, Any],
    freeze_descriptor: dict[str, str],
) -> list[str]:
    blockers: list[str] = []
    if _find_fields(launch_plan, FORBIDDEN_REGISTRY_FIELDS):
        blockers.append("launch_plan_forbidden_outcome_fields_present")
    if launch_plan.get("schema_version") != LAUNCH_PLAN_SCHEMA_VERSION:
        blockers.append("launch_plan_schema_mismatch")
    expected_id = launch_plan.get("launch_plan_id")
    payload = dict(launch_plan)
    payload.pop("launch_plan_id", None)
    if expected_id != canonical_json_sha256(payload):
        blockers.append("launch_plan_identity_mismatch")
    if launch_plan.get("hybrid_precollection_freeze") != freeze_descriptor:
        blockers.append("launch_plan_precollection_freeze_identity_mismatch")
    if (
        launch_plan.get("collection_start_allowed") is not True
        or launch_plan.get("collector_execution_attempted") is not False
        or launch_plan.get("read_only_public_provider") is not True
    ):
        blockers.append("launch_plan_collection_authorization_invalid")
    if (
        int(launch_plan.get("initial_capture_attempt_count") or 0)
        != INITIAL_CAPTURE_ATTEMPT_COUNT
        or int(launch_plan.get("maximum_total_capture_attempt_count") or 0)
        != MAXIMUM_CAPTURE_ATTEMPT_COUNT
        or int(launch_plan.get("maximum_continuation_attempt_count") or 0)
        != MAXIMUM_CONTINUATION_ATTEMPT_COUNT
    ):
        blockers.append("launch_plan_attempt_contract_mismatch")
    if launch_plan.get("candidate_lineage") != freeze.get("candidate_lineage"):
        blockers.append("launch_plan_candidate_lineage_mismatch")
    if not _hybrid_safety_valid(launch_plan):
        blockers.append("launch_plan_safety_contract_failed")
    return blockers


def _batch_launch_plan_blockers(
    *,
    pins: tuple[tuple[Path, str], ...],
    launch_plan: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    prefix = str(launch_plan.get("batch_id_prefix") or "")
    collector_contract = dict(launch_plan.get("collector_contract") or {})
    for path, _ in pins:
        batch = _load_json(path.resolve())
        if not str(batch.get("batch_id") or "").startswith(prefix):
            blockers.append("batch_id_not_from_authorized_launch_plan")
        captures = [dict(row) for row in batch.get("captures") or []]
        for capture in captures:
            if capture.get("market_family") != "btc_updown_5m":
                blockers.append("batch_capture_market_family_drift")
            if float(
                capture.get("orderbook_snapshot_interval_seconds") or 0.0
            ) != float(
                collector_contract.get("orderbook_snapshot_interval_seconds")
                or 0.0
            ):
                blockers.append("batch_capture_collector_contract_drift")
    return sorted(set(blockers))


def _support_decision(
    *,
    attempted_capture_count: int,
    preflight_blockers: list[str],
    role_report: dict[str, Any] | None,
) -> dict[str, Any]:
    if preflight_blockers:
        return _blocked_support_decision(preflight_blockers)
    if attempted_capture_count < INITIAL_CAPTURE_ATTEMPT_COUNT:
        return {
            "status": "initial_collection_incomplete",
            "support_only_failure": False,
            "continuation_allowed": False,
            "continuation_required": False,
            "continuation_attempt_count": 0,
            "continuation_reason_codes": [],
            "blocking_reason_codes": [
                "frozen_initial_capture_attempts_incomplete"
            ],
        }
    if role_report is None:
        return _blocked_support_decision(
            ["outcome_blind_role_assignment_missing"]
        )
    role_blockers = {
        str(reason)
        for reason in role_report.get("blocking_reason_codes") or []
    }
    if role_report.get("role_assignment_ready") is True:
        if (
            int(role_report.get("selected_market_count") or 0)
            != TOTAL_FRESH_MARKET_COUNT
            or role_blockers
        ):
            return _blocked_support_decision(
                ["role_assignment_ready_state_inconsistent"]
            )
        return {
            "status": "outcome_blind_support_target_ready",
            "support_only_failure": False,
            "continuation_allowed": False,
            "continuation_required": False,
            "continuation_attempt_count": 0,
            "continuation_reason_codes": [
                "target_valid_market_support_reached"
            ],
            "blocking_reason_codes": [],
        }
    support_only = bool(role_blockers) and role_blockers <= (
        SUPPORT_ONLY_ROLE_BLOCKERS
    )
    if not support_only:
        return _blocked_support_decision(
            sorted(
                role_blockers
                or {"role_assignment_failed_without_reason_codes"}
            )
        )
    remaining = MAXIMUM_CAPTURE_ATTEMPT_COUNT - attempted_capture_count
    if remaining <= 0:
        return {
            "status": "blocked_insufficient_support_at_frozen_maximum",
            "support_only_failure": True,
            "continuation_allowed": False,
            "continuation_required": False,
            "continuation_attempt_count": 0,
            "continuation_reason_codes": [],
            "blocking_reason_codes": [
                "insufficient_support_at_frozen_maximum"
            ],
        }
    continuation_count = min(remaining, MAXIMUM_CONTINUATION_ATTEMPT_COUNT)
    return {
        "status": "bounded_support_continuation_allowed",
        "support_only_failure": True,
        "continuation_allowed": True,
        "continuation_required": True,
        "continuation_attempt_count": continuation_count,
        "continuation_reason_codes": [
            "support_only_role_assignment_failure",
            "continue_within_frozen_maximum",
        ],
        "blocking_reason_codes": [],
    }


def _blocked_support_decision(reasons: list[str]) -> dict[str, Any]:
    return {
        "status": "blocked_fail_closed",
        "support_only_failure": False,
        "continuation_allowed": False,
        "continuation_required": False,
        "continuation_attempt_count": 0,
        "continuation_reason_codes": [],
        "blocking_reason_codes": sorted(set(reasons)),
    }


def _role_chronology_passed(selected: list[dict[str, Any]]) -> bool:
    calibration = [
        row for row in selected if row.get("role") == CALIBRATION_ROLE
    ]
    confirmatory = [
        row for row in selected if row.get("role") == CONFIRMATORY_ROLE
    ]
    if (
        len(calibration) != CALIBRATION_MARKET_COUNT
        or len(confirmatory) != CONFIRMATORY_MARKET_COUNT
    ):
        return False
    calibration_max = max(
        int(row.get("maximum_decision_ts") or 0) for row in calibration
    )
    confirmatory_min = min(
        int(row.get("minimum_decision_ts") or 0) for row in confirmatory
    )
    return calibration_max > 0 and confirmatory_min > calibration_max


def _verified_descriptor(value: Any, *, name: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} descriptor is missing")
    path = Path(str(value.get("path") or "")).expanduser().resolve()
    digest = str(value.get("sha256") or "")
    _require_sha256(digest, name=f"{name} SHA-256")
    _verify_pin(path, digest, name=name)
    return {"path": str(path), "sha256": digest}


def _blocked_safety_valid(payload: dict[str, Any]) -> bool:
    return all(
        payload.get(name) is expected
        for name, expected in _blocked_safety_fields().items()
    )


def _hybrid_safety_valid(payload: dict[str, Any]) -> bool:
    return all(
        payload.get(name) is expected
        for name, expected in _hybrid_safety_fields().items()
    )


def _hybrid_safety_fields() -> dict[str, Any]:
    return {
        **_blocked_safety_fields(),
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
    }


def _prepare_run_dir(
    output_dir: Path,
    run_id: str,
    *,
    overwrite: bool,
) -> Path:
    run_dir = (output_dir / run_id).expanduser().resolve()
    if run_dir.exists():
        if not overwrite:
            raise ValueError(f"run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _start_gate_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Hybrid Fresh Collection Start Gate",
            "",
            f"- status: `{report['status']}`",
            (
                "- collection start allowed: "
                f"`{str(report['collection_start_allowed']).lower()}`"
            ),
            (
                "- launch command generated: "
                f"`{str(report['collection_start_command_generated']).lower()}`"
            ),
            "- collector execution attempted: `false`",
            f"- blocking reasons: `{json.dumps(report['blocking_reason_codes'])}`",
            "- labels/outcomes/PnL used: `false`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )


def _support_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Hybrid Fresh Collection Support Gate",
            "",
            f"- status: `{report['status']}`",
            f"- attempted captures: `{report['attempted_capture_count']}`",
            f"- selected markets: `{report['selected_market_count']}`",
            f"- continuation allowed: `{str(report['continuation_allowed']).lower()}`",
            f"- continuation attempts: `{report['continuation_attempt_count']}`",
            f"- blocking reasons: `{json.dumps(report['blocking_reason_codes'])}`",
            "- support/continuation uses labels or outcomes: `false`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )


def _role_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Hybrid Fresh Outcome-Blind 45/60 Role Assignment",
            "",
            f"- status: `{report['status']}`",
            f"- selected markets: `{report['selected_market_count']}`",
            f"- role counts: `{json.dumps(report['role_market_counts'], sort_keys=True)}`",
            (
                "- chronology passed: "
                f"`{str(report['chronology_validation_passed']).lower()}`"
            ),
            f"- blocking reasons: `{json.dumps(report['blocking_reason_codes'])}`",
            "- labels/outcomes opened for role assignment: `false`",
            "- confirmatory labels opened: `false`",
            "",
        ]
    )


def _execution_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Hybrid Fresh Execution Compatibility",
            "",
            f"- selected markets: `{report['selected_market_count']}`",
            f"- decision rows: `{report['selected_decision_row_count']}`",
            (
                "- execution-compatible decision rows: "
                f"`{report['execution_compatible_decision_row_count']}`"
            ),
            f"- selected market failures: `{report['selected_market_failure_count']}`",
            "- validated before label access: `true`",
            "- labels/outcomes opened: `false`",
            "",
        ]
    )
