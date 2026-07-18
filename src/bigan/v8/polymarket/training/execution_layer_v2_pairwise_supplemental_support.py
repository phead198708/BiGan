"""Pre-registered supplemental support recovery after a frozen-max failure."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb import (
    FORBIDDEN_REGISTRY_FIELDS,
    _blocked_safety_fields,
    _descriptor,
    _find_fields,
    _load_json,
    _load_jsonl,
    _require_sha256,
    _sha256_file,
    _verify_pin,
    _write_json,
    _write_text,
)
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_precollection_support import (
    PairwisePrecollectionSupportGateConfig,
    _batch_preflight,
    run_pairwise_precollection_support_gate,
)

SUPPLEMENTAL_ATTEMPT_COUNT = 20
PARENT_ATTEMPT_COUNT = 260
PARENT_SELECTED_MARKET_COUNT = 191
TARGET_MARKET_COUNT = 195
REQUIRED_SUPPLEMENTAL_MARKET_COUNT = 4
EXPECTED_ROLE_COUNTS = {
    "confirmatory_validation": 60,
    "development_calibration": 45,
    "development_train": 90,
}
PARENT_SUPPORT_STATUS = "BLOCKED_INSUFFICIENT_SUPPORT_AT_FROZEN_MAXIMUM"
SUCCESSOR_FREEZE_SCHEMA_VERSION = (
    "bigan-v8-pairwise-supplemental-support-role-freeze-v1"
)
SUCCESSOR_FREEZE_REPORT_SCHEMA_VERSION = (
    "bigan-v8-pairwise-supplemental-support-freeze-report-v1"
)
SUCCESSOR_FREEZE_DESCRIPTOR_SCHEMA_VERSION = (
    "bigan-v8-pairwise-supplemental-support-freeze-descriptor-v1"
)
SUCCESSOR_GATE_REPORT_SCHEMA_VERSION = (
    "bigan-v8-pairwise-supplemental-support-gate-report-v1"
)
SUCCESSOR_GATE_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-pairwise-supplemental-support-gate-manifest-v1"
)


@dataclass(frozen=True, slots=True)
class PairwiseSupplementalSupportFreezeConfig:
    """Hash-pinned parent evidence used before supplemental collection."""

    run_id: str
    output_dir: Path | str
    freeze_created_ts: int
    parent_precollection_freeze_path: Path | str
    parent_precollection_freeze_sha256: str
    parent_terminal_reconciliation_report_path: Path | str
    parent_terminal_reconciliation_report_sha256: str
    parent_terminal_reconciliation_manifest_path: Path | str
    parent_terminal_reconciliation_manifest_sha256: str
    parent_support_report_path: Path | str
    parent_support_report_sha256: str
    parent_support_manifest_path: Path | str
    parent_support_manifest_sha256: str
    successor_freeze_builder_git_commit: str

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if self.freeze_created_ts <= 0:
            raise ValueError("freeze_created_ts must be positive")
        if not _is_git_commit(self.successor_freeze_builder_git_commit):
            raise ValueError(
                "successor_freeze_builder_git_commit must be a 40-character hex digest"
            )
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        for field in (
            "parent_precollection_freeze_path",
            "parent_terminal_reconciliation_report_path",
            "parent_terminal_reconciliation_manifest_path",
            "parent_support_report_path",
            "parent_support_manifest_path",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)))
        for field in (
            "parent_precollection_freeze_sha256",
            "parent_terminal_reconciliation_report_sha256",
            "parent_terminal_reconciliation_manifest_sha256",
            "parent_support_report_sha256",
            "parent_support_manifest_sha256",
        ):
            _require_sha256(getattr(self, field), name=field)
            object.__setattr__(self, field, getattr(self, field).lower())


@dataclass(frozen=True, slots=True)
class PairwiseSupplementalSupportGateConfig:
    """Final gate inputs after the fixed supplemental collection."""

    run_id: str
    output_dir: Path | str
    successor_freeze_path: Path | str
    successor_freeze_sha256: str
    supplemental_batch_progress_pins: tuple[tuple[Path | str, str], ...]
    training_corpus_root: Path | str = Path("/Volumes/PHILIPS/v8")

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(
            self.successor_freeze_sha256,
            name="successor freeze SHA-256",
        )
        pins: list[tuple[Path, str]] = []
        for path, digest in self.supplemental_batch_progress_pins:
            _require_sha256(digest, name="supplemental batch SHA-256")
            pins.append((Path(path), digest.lower()))
        if not pins:
            raise ValueError("at least one supplemental batch pin is required")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "successor_freeze_path", Path(self.successor_freeze_path))
        object.__setattr__(self, "supplemental_batch_progress_pins", tuple(pins))
        object.__setattr__(self, "training_corpus_root", Path(self.training_corpus_root))


def create_pairwise_supplemental_support_freeze(
    config: PairwiseSupplementalSupportFreezeConfig,
) -> dict[str, Any]:
    """Create the successor freeze before any supplemental capture begins."""

    parent_freeze_path = config.parent_precollection_freeze_path.resolve()
    terminal_report_path = config.parent_terminal_reconciliation_report_path.resolve()
    terminal_manifest_path = config.parent_terminal_reconciliation_manifest_path.resolve()
    support_report_path = config.parent_support_report_path.resolve()
    support_manifest_path = config.parent_support_manifest_path.resolve()
    for path, digest, name in (
        (parent_freeze_path, config.parent_precollection_freeze_sha256, "parent freeze"),
        (
            terminal_report_path,
            config.parent_terminal_reconciliation_report_sha256,
            "parent terminal reconciliation report",
        ),
        (
            terminal_manifest_path,
            config.parent_terminal_reconciliation_manifest_sha256,
            "parent terminal reconciliation manifest",
        ),
        (support_report_path, config.parent_support_report_sha256, "parent support report"),
        (
            support_manifest_path,
            config.parent_support_manifest_sha256,
            "parent support manifest",
        ),
    ):
        _verify_pin(path, digest, name=name)

    parent_freeze = _load_json(parent_freeze_path)
    terminal_report = _load_json(terminal_report_path)
    terminal_manifest = _load_json(terminal_manifest_path)
    support_report = _load_json(support_report_path)
    support_manifest = _load_json(support_manifest_path)
    role_report_descriptor = _verified_descriptor(
        support_manifest.get("role_assignment_report"),
        name="parent role report",
    )
    role_manifest_descriptor = _verified_descriptor(
        support_manifest.get("role_assignment_manifest"),
        name="parent role manifest",
    )
    role_report = _load_json(Path(role_report_descriptor["path"]))
    role_manifest = _load_json(Path(role_manifest_descriptor["path"]))
    selected_rows_descriptor = _verified_descriptor(
        role_manifest.get("selected_rows"),
        name="parent selected role rows",
    )
    selected_rows = _load_jsonl(Path(selected_rows_descriptor["path"]))
    parent_batch_pins = _descriptor_pins(
        support_manifest.get("batch_progress_inputs"),
        name="parent batch progress inputs",
    )
    _, duplicate_inputs, parent_preflight = _batch_preflight(parent_batch_pins)
    blockers = _parent_evidence_blockers(
        parent_freeze=parent_freeze,
        terminal_report=terminal_report,
        terminal_manifest=terminal_manifest,
        terminal_report_path=terminal_report_path,
        support_report=support_report,
        support_manifest=support_manifest,
        support_report_path=support_report_path,
        role_report=role_report,
        role_manifest=role_manifest,
        role_report_descriptor=role_report_descriptor,
        selected_rows=selected_rows,
        selected_rows_descriptor=selected_rows_descriptor,
        parent_preflight=parent_preflight,
        duplicate_inputs=duplicate_inputs,
    )
    if blockers:
        raise ValueError(
            "parent evidence is not eligible for a supplemental freeze: "
            + ", ".join(blockers)
        )

    parent_capture_max_ts = _maximum_scheduled_capture_ts(parent_batch_pins)
    supplemental_minimum_ts = max(
        config.freeze_created_ts + 1,
        parent_capture_max_ts + 1,
    )
    run_dir = config.output_dir / config.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    collection_dir = (run_dir / "collection").resolve()
    successor = copy.deepcopy(parent_freeze)
    successor.pop("precollection_freeze_id", None)
    successor.update(
        {
            "schema_version": SUCCESSOR_FREEZE_SCHEMA_VERSION,
            "run_id": config.run_id,
            "freeze_created_ts": config.freeze_created_ts,
            "successor_freeze_builder_git_commit": (
                config.successor_freeze_builder_git_commit.lower()
            ),
            "parent_frozen_experiment_status": PARENT_SUPPORT_STATUS,
            "parent_frozen_experiment_remains_failed": True,
            "parent_precollection_freeze": _descriptor(parent_freeze_path),
            "parent_terminal_reconciliation_report": _descriptor(
                terminal_report_path
            ),
            "parent_terminal_reconciliation_manifest": _descriptor(
                terminal_manifest_path
            ),
            "parent_support_report": _descriptor(support_report_path),
            "parent_support_manifest": _descriptor(support_manifest_path),
            "parent_role_report": role_report_descriptor,
            "parent_role_manifest": role_manifest_descriptor,
            "parent_selected_rows": selected_rows_descriptor,
            "parent_selected_market_ids_sha256": role_manifest[
                "selected_market_ids_sha256"
            ],
            "parent_batch_progress_inputs": [
                _descriptor(path) for path, _ in parent_batch_pins
            ],
            "parent_capture_attempt_count": PARENT_ATTEMPT_COUNT,
            "parent_selected_market_count": PARENT_SELECTED_MARKET_COUNT,
            "required_supplemental_valid_market_count": (
                REQUIRED_SUPPLEMENTAL_MARKET_COUNT
            ),
            "supplemental_capture_attempt_count": SUPPLEMENTAL_ATTEMPT_COUNT,
            "initial_capture_attempt_count": (
                PARENT_ATTEMPT_COUNT + SUPPLEMENTAL_ATTEMPT_COUNT
            ),
            "maximum_total_capture_attempt_count": (
                PARENT_ATTEMPT_COUNT + SUPPLEMENTAL_ATTEMPT_COUNT
            ),
            "target_valid_market_count": TARGET_MARKET_COUNT,
            "supplemental_minimum_collection_decision_ts": (
                supplemental_minimum_ts
            ),
            "supplemental_collection_must_be_strictly_later": True,
            "supplemental_collection_stop_early_allowed": False,
            "supplemental_dynamic_extension_allowed": False,
            "supplemental_replacement_attempts_allowed": False,
            "supplemental_attempt_count_source": (
                "pre_registered_fixed_20_before_supplemental_collection"
            ),
            "collection_output_dir": str(collection_dir),
            "collection_batch_id_prefix": f"issue188-{config.run_id}",
            "collection_started": False,
            "supplemental_collection_started": False,
            "supplemental_collection_start_allowed": True,
            "supplemental_support_freeze_ready": True,
            "parent_source_files_mutated": False,
            "labels_or_outcomes_opened_for_support_planning": False,
            "settlement_pnl_opened_for_support_planning": False,
            "oof_validation_pnl_used_for_support_planning": False,
            "execution_thresholds_mutated": False,
            "collector_contract_mutated": False,
            "model_fit_started": False,
            "confirmatory_validation_started": False,
            **_blocked_safety_fields(),
        }
    )
    successor["precollection_freeze_id"] = canonical_json_sha256(successor)
    freeze_path = run_dir / "precollection_role_freeze_manifest.json"
    _write_json(freeze_path, successor)

    report = {
        "schema_version": SUCCESSOR_FREEZE_REPORT_SCHEMA_VERSION,
        "run_id": config.run_id,
        "status": "SUPPLEMENTAL_SUPPORT_FREEZE_READY",
        "supplemental_support_freeze_ready": True,
        "supplemental_collection_start_allowed": True,
        "parent_capture_attempt_count": PARENT_ATTEMPT_COUNT,
        "parent_selected_market_count": PARENT_SELECTED_MARKET_COUNT,
        "target_valid_market_count": TARGET_MARKET_COUNT,
        "required_supplemental_valid_market_count": (
            REQUIRED_SUPPLEMENTAL_MARKET_COUNT
        ),
        "supplemental_capture_attempt_count": SUPPLEMENTAL_ATTEMPT_COUNT,
        "maximum_total_capture_attempt_count": (
            PARENT_ATTEMPT_COUNT + SUPPLEMENTAL_ATTEMPT_COUNT
        ),
        "parent_maximum_scheduled_capture_ts": parent_capture_max_ts,
        "supplemental_minimum_collection_decision_ts": supplemental_minimum_ts,
        "parent_selected_market_ids_sha256": role_manifest[
            "selected_market_ids_sha256"
        ],
        "parent_batch_progress_count": len(parent_batch_pins),
        "parent_source_files_mutated": False,
        "supplemental_collection_stop_early_allowed": False,
        "supplemental_dynamic_extension_allowed": False,
        "labels_or_outcomes_opened_for_support_planning": False,
        "settlement_pnl_opened_for_support_planning": False,
        "oof_validation_pnl_used_for_support_planning": False,
        "execution_thresholds_mutated": False,
        "collector_contract_mutated": False,
        "blocking_reason_codes": [],
        **_blocked_safety_fields(),
    }
    report["supplemental_support_freeze_report_id"] = canonical_json_sha256(
        report
    )
    report_path = run_dir / "pairwise_supplemental_support_freeze_report.json"
    markdown_path = run_dir / "pairwise_supplemental_support_freeze_report.md"
    _write_json(report_path, report)
    _write_text(markdown_path, _freeze_markdown(report))
    descriptor = {
        "schema_version": SUCCESSOR_FREEZE_DESCRIPTOR_SCHEMA_VERSION,
        "run_id": config.run_id,
        "successor_freeze": _descriptor(freeze_path),
        "freeze_report": _descriptor(report_path),
        "freeze_markdown": _descriptor(markdown_path),
        "supplemental_support_freeze_ready": True,
        "supplemental_collection_start_allowed": True,
        "supplemental_capture_attempt_count": SUPPLEMENTAL_ATTEMPT_COUNT,
        "labels_or_outcomes_opened_for_support_planning": False,
        **_blocked_safety_fields(),
    }
    descriptor["supplemental_support_freeze_descriptor_id"] = (
        canonical_json_sha256(descriptor)
    )
    descriptor_path = run_dir / "pairwise_supplemental_support_freeze_descriptor.json"
    _write_json(descriptor_path, descriptor)
    return {
        "run_dir": run_dir,
        "freeze_path": freeze_path,
        "freeze_sha256": _sha256_file(freeze_path),
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "markdown_path": markdown_path,
        "markdown_sha256": _sha256_file(markdown_path),
        "descriptor_path": descriptor_path,
        "descriptor_sha256": _sha256_file(descriptor_path),
        "freeze": successor,
        "report": report,
    }


def run_pairwise_supplemental_support_gate(
    config: PairwiseSupplementalSupportGateConfig,
) -> dict[str, Any]:
    """Run the unchanged role gate after exactly 20 supplemental attempts."""

    freeze_path = config.successor_freeze_path.resolve()
    _verify_pin(
        freeze_path,
        config.successor_freeze_sha256,
        name="successor supplemental support freeze",
    )
    freeze = _load_json(freeze_path)
    parent_pins = _descriptor_pins(
        freeze.get("parent_batch_progress_inputs"),
        name="frozen parent batch progress inputs",
    )
    parent_preflight = _batch_preflight(parent_pins)[2]
    supplemental_preflight = _batch_preflight(
        config.supplemental_batch_progress_pins
    )[2]
    combined_pins = (*parent_pins, *config.supplemental_batch_progress_pins)
    combined_preflight = _batch_preflight(combined_pins)[2]
    blockers = _successor_gate_preflight_blockers(
        freeze=freeze,
        parent_preflight=parent_preflight,
        supplemental_preflight=supplemental_preflight,
        combined_preflight=combined_preflight,
        supplemental_pins=config.supplemental_batch_progress_pins,
    )

    core_result: dict[str, Any] | None = None
    core_report: dict[str, Any] = {}
    role_counts: dict[str, int] = {}
    if not blockers:
        core_result = run_pairwise_precollection_support_gate(
            PairwisePrecollectionSupportGateConfig(
                run_id=f"{config.run_id}-core",
                output_dir=config.output_dir,
                precollection_freeze_manifest_path=freeze_path,
                expected_precollection_freeze_manifest_sha256=(
                    config.successor_freeze_sha256
                ),
                batch_progress_pins=combined_pins,
                training_corpus_root=config.training_corpus_root,
            )
        )
        core_report = dict(core_result["report"])
        role_result = core_result.get("role_assignment_result") or {}
        role_report = dict(role_result.get("report") or {})
        role_counts = {
            str(key): int(value)
            for key, value in dict(
                role_report.get("role_market_counts") or {}
            ).items()
        }
        if core_report.get("status") != "OUTCOME_BLIND_SUPPORT_TARGET_READY":
            blockers.extend(core_report.get("blocking_reason_codes") or [])
        if int(core_report.get("selected_market_count") or 0) != TARGET_MARKET_COUNT:
            blockers.append("successor_target_market_count_not_reached")
        if role_counts != EXPECTED_ROLE_COUNTS:
            blockers.append("successor_role_market_counts_mismatch")
        if core_report.get("continuation_allowed") is not False:
            blockers.append("successor_dynamic_continuation_not_disabled")
    blockers = sorted({str(reason) for reason in blockers})
    ready = not blockers
    run_dir = config.output_dir / config.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": SUCCESSOR_GATE_REPORT_SCHEMA_VERSION,
        "run_id": config.run_id,
        "status": (
            "OUTCOME_BLIND_SUPPLEMENTAL_SUPPORT_TARGET_READY"
            if ready
            else "BLOCKED_FAIL_CLOSED"
        ),
        "supplemental_support_target_ready": ready,
        "parent_capture_attempt_count": int(
            parent_preflight.get("attempted_capture_count") or 0
        ),
        "supplemental_capture_attempt_count": int(
            supplemental_preflight.get("attempted_capture_count") or 0
        ),
        "combined_capture_attempt_count": int(
            combined_preflight.get("attempted_capture_count") or 0
        ),
        "parent_selected_market_count": PARENT_SELECTED_MARKET_COUNT,
        "selected_market_count": int(
            core_report.get("selected_market_count") or 0
        ),
        "new_selected_market_count": max(
            0,
            int(core_report.get("selected_market_count") or 0)
            - PARENT_SELECTED_MARKET_COUNT,
        ),
        "target_valid_market_count": TARGET_MARKET_COUNT,
        "role_market_counts": role_counts,
        "expected_role_market_counts": EXPECTED_ROLE_COUNTS,
        "core_support_gate_status": core_report.get("status"),
        "core_support_gate_blocking_reason_codes": list(
            core_report.get("blocking_reason_codes") or []
        ),
        "supplemental_dynamic_extension_allowed": False,
        "continuation_allowed": False,
        "blocking_reason_codes": blockers,
        "labels_or_outcomes_opened_for_support_gate": False,
        "settlement_pnl_opened_for_support_gate": False,
        "oof_validation_pnl_used_for_support_gate": False,
        "execution_thresholds_mutated": False,
        "source_scores_mutated": False,
        **_blocked_safety_fields(),
    }
    report["supplemental_support_gate_report_id"] = canonical_json_sha256(
        report
    )
    report_path = run_dir / "pairwise_supplemental_support_gate_report.json"
    markdown_path = run_dir / "pairwise_supplemental_support_gate_report.md"
    _write_json(report_path, report)
    _write_text(markdown_path, _gate_markdown(report))
    manifest = {
        "schema_version": SUCCESSOR_GATE_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "successor_freeze": _descriptor(freeze_path),
        "parent_batch_progress_inputs": [
            _descriptor(path) for path, _ in parent_pins
        ],
        "supplemental_batch_progress_inputs": [
            _descriptor(path.resolve())
            for path, _ in config.supplemental_batch_progress_pins
        ],
        "core_support_gate_report": (
            None
            if core_result is None
            else _descriptor(Path(core_result["report_path"]))
        ),
        "core_support_gate_manifest": (
            None
            if core_result is None
            else _descriptor(Path(core_result["manifest_path"]))
        ),
        "report": _descriptor(report_path),
        "markdown": _descriptor(markdown_path),
        "supplemental_support_target_ready": ready,
        "selected_market_count": report["selected_market_count"],
        "role_market_counts": role_counts,
        "blocking_reason_codes": blockers,
        "supplemental_dynamic_extension_allowed": False,
        "continuation_allowed": False,
        "labels_or_outcomes_opened_for_support_gate": False,
        **_blocked_safety_fields(),
    }
    manifest["supplemental_support_gate_manifest_id"] = canonical_json_sha256(
        manifest
    )
    manifest_path = run_dir / "pairwise_supplemental_support_gate_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "markdown_path": markdown_path,
        "markdown_sha256": _sha256_file(markdown_path),
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "report": report,
        "manifest": manifest,
        "core_result": core_result,
    }


def _parent_evidence_blockers(
    *,
    parent_freeze: dict[str, Any],
    terminal_report: dict[str, Any],
    terminal_manifest: dict[str, Any],
    terminal_report_path: Path,
    support_report: dict[str, Any],
    support_manifest: dict[str, Any],
    support_report_path: Path,
    role_report: dict[str, Any],
    role_manifest: dict[str, Any],
    role_report_descriptor: dict[str, str],
    selected_rows: list[dict[str, Any]],
    selected_rows_descriptor: dict[str, str],
    parent_preflight: dict[str, Any],
    duplicate_inputs: list[dict[str, str]],
) -> list[str]:
    blockers: list[str] = []
    payloads = (
        parent_freeze,
        terminal_report,
        terminal_manifest,
        support_report,
        support_manifest,
        role_report,
        role_manifest,
    )
    if any(_find_fields(payload, FORBIDDEN_REGISTRY_FIELDS) for payload in payloads):
        blockers.append("parent_evidence_contains_forbidden_outcome_fields")
    if any(not _safe_payload(payload) for payload in payloads):
        blockers.append("parent_evidence_safety_contract_failed")
    if terminal_report.get("status") != "TERMINAL_RECONCILIATION_READY":
        blockers.append("parent_terminal_reconciliation_not_ready")
    if int(terminal_report.get("source_capture_count") or 0) != PARENT_ATTEMPT_COUNT:
        blockers.append("parent_terminal_capture_count_mismatch")
    if terminal_report.get("labels_or_outcomes_opened_for_reconciliation") is not False:
        blockers.append("parent_terminal_reconciliation_opened_targets")
    terminal_report_descriptor = terminal_manifest.get("report") or {}
    if not _descriptor_matches(terminal_report_descriptor, terminal_report_path):
        blockers.append("parent_terminal_report_manifest_mismatch")
    if support_report.get("status") != PARENT_SUPPORT_STATUS:
        blockers.append("parent_support_status_invalid")
    if int(support_report.get("attempted_capture_count") or 0) != PARENT_ATTEMPT_COUNT:
        blockers.append("parent_support_attempt_count_mismatch")
    if int(support_report.get("selected_market_count") or 0) != PARENT_SELECTED_MARKET_COUNT:
        blockers.append("parent_support_selected_market_count_mismatch")
    if support_report.get("continuation_allowed") is not False:
        blockers.append("parent_support_continuation_still_allowed")
    if support_report.get("labels_or_outcomes_opened_for_continuation") is not False:
        blockers.append("parent_support_opened_targets")
    if support_report.get("settlement_pnl_opened_for_continuation") is not False:
        blockers.append("parent_support_opened_settlement_pnl")
    if not _descriptor_matches(
        support_manifest.get("support_gate_report") or {},
        support_report_path,
    ):
        blockers.append("parent_support_report_manifest_mismatch")
    if role_report.get("role_assignment_ready") is not False:
        blockers.append("parent_role_assignment_unexpectedly_ready")
    if int(role_report.get("selected_market_count") or 0) != PARENT_SELECTED_MARKET_COUNT:
        blockers.append("parent_role_selected_market_count_mismatch")
    if role_report.get("labels_or_outcomes_opened_for_role_assignment") is not False:
        blockers.append("parent_role_assignment_opened_targets")
    if role_manifest.get("labels_or_outcomes_opened_for_role_assignment") is not False:
        blockers.append("parent_role_manifest_opened_targets")
    if not _descriptor_matches(
        role_manifest.get("report") or {},
        Path(role_report_descriptor["path"]),
    ):
        blockers.append("parent_role_report_descriptor_mismatch")
    if not _descriptor_matches(
        role_manifest.get("selected_rows") or {},
        Path(selected_rows_descriptor["path"]),
    ):
        blockers.append("parent_selected_rows_descriptor_mismatch")
    market_ids = [str(row.get("market_id") or "") for row in selected_rows]
    if (
        len(selected_rows) != PARENT_SELECTED_MARKET_COUNT
        or any(not value for value in market_ids)
        or len(set(market_ids)) != PARENT_SELECTED_MARKET_COUNT
    ):
        blockers.append("parent_selected_market_identity_invalid")
    if canonical_json_sha256(sorted(market_ids)) != role_manifest.get(
        "selected_market_ids_sha256"
    ):
        blockers.append("parent_selected_market_identity_hash_mismatch")
    if duplicate_inputs:
        blockers.append("parent_batch_progress_inputs_duplicated")
    if int(parent_preflight.get("attempted_capture_count") or 0) != PARENT_ATTEMPT_COUNT:
        blockers.append("parent_batch_capture_count_mismatch")
    blockers.extend(parent_preflight.get("blocking_reason_codes") or [])
    if int(parent_freeze.get("maximum_total_capture_attempt_count") or 0) != (
        PARENT_ATTEMPT_COUNT
    ):
        blockers.append("parent_frozen_maximum_mismatch")
    if int(parent_freeze.get("target_valid_market_count") or 0) != TARGET_MARKET_COUNT:
        blockers.append("parent_target_market_count_mismatch")
    return sorted({str(reason) for reason in blockers})


def _successor_gate_preflight_blockers(
    *,
    freeze: dict[str, Any],
    parent_preflight: dict[str, Any],
    supplemental_preflight: dict[str, Any],
    combined_preflight: dict[str, Any],
    supplemental_pins: tuple[tuple[Path, str], ...],
) -> list[str]:
    blockers: list[str] = []
    if freeze.get("schema_version") != SUCCESSOR_FREEZE_SCHEMA_VERSION:
        blockers.append("successor_freeze_schema_invalid")
    if freeze.get("supplemental_support_freeze_ready") is not True:
        blockers.append("successor_freeze_not_ready")
    if freeze.get("supplemental_dynamic_extension_allowed") is not False:
        blockers.append("successor_dynamic_extension_not_disabled")
    if int(freeze.get("supplemental_capture_attempt_count") or 0) != (
        SUPPLEMENTAL_ATTEMPT_COUNT
    ):
        blockers.append("successor_supplemental_attempt_contract_mismatch")
    if int(freeze.get("maximum_total_capture_attempt_count") or 0) != (
        PARENT_ATTEMPT_COUNT + SUPPLEMENTAL_ATTEMPT_COUNT
    ):
        blockers.append("successor_frozen_maximum_mismatch")
    if _find_fields(freeze, FORBIDDEN_REGISTRY_FIELDS):
        blockers.append("successor_freeze_contains_forbidden_outcome_fields")
    if not _safe_payload(freeze):
        blockers.append("successor_freeze_safety_contract_failed")
    if int(parent_preflight.get("attempted_capture_count") or 0) != PARENT_ATTEMPT_COUNT:
        blockers.append("successor_parent_capture_count_mismatch")
    if int(supplemental_preflight.get("attempted_capture_count") or 0) != (
        SUPPLEMENTAL_ATTEMPT_COUNT
    ):
        blockers.append("supplemental_capture_attempt_count_mismatch")
    if int(combined_preflight.get("attempted_capture_count") or 0) != (
        PARENT_ATTEMPT_COUNT + SUPPLEMENTAL_ATTEMPT_COUNT
    ):
        blockers.append("successor_combined_capture_count_mismatch")
    blockers.extend(parent_preflight.get("blocking_reason_codes") or [])
    blockers.extend(supplemental_preflight.get("blocking_reason_codes") or [])
    blockers.extend(combined_preflight.get("blocking_reason_codes") or [])
    minimum_ts = int(
        freeze.get("supplemental_minimum_collection_decision_ts") or 0
    )
    for path, digest in supplemental_pins:
        _verify_pin(path.resolve(), digest, name="supplemental batch progress")
        payload = _load_json(path.resolve())
        for capture in payload.get("captures") or []:
            if int(capture.get("scheduled_round_start_ts") or 0) < minimum_ts:
                blockers.append("supplemental_capture_not_strictly_later")
    return sorted({str(reason) for reason in blockers})


def _descriptor_pins(
    values: Any,
    *,
    name: str,
) -> tuple[tuple[Path, str], ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty descriptor list")
    pins: list[tuple[Path, str]] = []
    for index, value in enumerate(values):
        descriptor = _verified_descriptor(value, name=f"{name}[{index}]")
        pins.append((Path(descriptor["path"]), descriptor["sha256"]))
    return tuple(pins)


def _verified_descriptor(value: Any, *, name: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} descriptor is missing")
    path = Path(str(value.get("path") or "")).expanduser().resolve()
    digest = str(value.get("sha256") or "").lower()
    _require_sha256(digest, name=f"{name} SHA-256")
    _verify_pin(path, digest, name=name)
    return {"path": str(path), "sha256": digest}


def _descriptor_matches(value: dict[str, Any], path: Path) -> bool:
    return (
        Path(str(value.get("path") or "")).expanduser().resolve()
        == path.resolve()
        and str(value.get("sha256") or "").lower() == _sha256_file(path)
    )


def _maximum_scheduled_capture_ts(
    pins: tuple[tuple[Path, str], ...],
) -> int:
    return max(
        (
            int(capture.get("scheduled_round_start_ts") or 0)
            for path, _ in pins
            for capture in (_load_json(path).get("captures") or [])
        ),
        default=0,
    )


def _safe_payload(payload: dict[str, Any]) -> bool:
    return (
        payload.get("paper_only") is True
        and payload.get("capital_at_risk") is False
        and payload.get("polymarket_write_enabled", False) is False
        and payload.get("wallet_signing_enabled", False) is False
        and payload.get("source_model_candidate_eligible", False) is False
        and payload.get("promotion_evidence_eligible", False) is False
        and payload.get("v8_execution_handoff_allowed", False) is False
    )


def _is_git_commit(value: str) -> bool:
    normalized = value.lower()
    return len(normalized) == 40 and all(
        character in "0123456789abcdef" for character in normalized
    )


def _freeze_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Pairwise Supplemental Support Freeze",
            "",
            f"- status: `{report['status']}`",
            f"- parent support: `{report['parent_selected_market_count']}`",
            f"- target support: `{report['target_valid_market_count']}`",
            f"- fixed supplemental attempts: `{report['supplemental_capture_attempt_count']}`",
            "- dynamic extension allowed: `false`",
            "- labels/outcomes/PnL opened: `false`",
            "- execution/collector thresholds mutated: `false`",
            "- paper/live/handoff unlock: `false`",
        ]
    ) + "\n"


def _gate_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Pairwise Supplemental Support Gate",
            "",
            f"- status: `{report['status']}`",
            f"- combined captures: `{report['combined_capture_attempt_count']}`",
            f"- selected markets: `{report['selected_market_count']}`",
            f"- target markets: `{report['target_valid_market_count']}`",
            f"- role counts: `{json.dumps(report['role_market_counts'], sort_keys=True)}`",
            f"- blockers: `{json.dumps(report['blocking_reason_codes'])}`",
            "- dynamic extension allowed: `false`",
            "- labels/outcomes/PnL opened: `false`",
            "- paper/live/handoff unlock: `false`",
        ]
    ) + "\n"
