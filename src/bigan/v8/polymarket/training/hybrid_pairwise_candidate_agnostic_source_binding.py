"""Bind a bounded candidate-agnostic raw window to the frozen hybrid protocol."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb import (
    _descriptor,
    _find_fields,
    _load_json,
    _require_sha256,
    _sha256_file,
    _verify_pin,
    _write_json,
    _write_jsonl,
    _write_text,
    validate_pairwise_action_advantage_lcb_protocol,
)
from bigan.v8.polymarket.training.hybrid_pairwise_fresh_collection_roles import (
    INITIAL_CAPTURE_ATTEMPT_COUNT,
    MAXIMUM_CAPTURE_ATTEMPT_COUNT,
    _assign_hybrid_fresh_roles,
    _hybrid_safety_fields,
    _prepare_run_dir,
    _verified_descriptor,
)

SCHEMA_PREFIX = "bigan-v8-hybrid-pairwise-candidate-agnostic-source"
BINDING_REPORT_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-binding-report-v1"
BINDING_MANIFEST_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-binding-manifest-v1"
SNAPSHOT_REPORT_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-snapshot-report-v1"
SNAPSHOT_MANIFEST_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-snapshot-manifest-v1"
BOUND_SOURCE_PLAN_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-role-source-plan-v1"
PRECOLLECTION_FREEZE_SCHEMA_VERSION = (
    "bigan-v8-hybrid-pairwise-precollection-freeze-manifest-v1"
)
READINESS_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-hybrid-pairwise-precollection-readiness-manifest-v1"
)
FUTURE_PREREGISTRATION_SCHEMA_VERSION = (
    "bigan-v8-pairwise-future-unseen-holdout-pre-registration-manifest-v1"
)
FUTURE_COLLECTION_FREEZE_SCHEMA_VERSION = (
    "bigan-v8-pairwise-future-unseen-collection-freeze-manifest-v1"
)
TERMINAL_COLLECTION_STOP_REASONS = frozenset(
    {
        "outcome_blind_quality_target_reached",
        "frozen_maximum_capture_attempt_count_reached_without_target",
    }
)
FORBIDDEN_SOURCE_FIELDS = frozenset(
    {
        "resolved_outcome",
        "outcomePrices",
        "payout_up",
        "payout_down",
        "settlement_pnl",
        "realized_pnl",
        "oracle_action",
        "future_return",
        "target_net_pnl_per_contract",
    }
)


@dataclass(frozen=True, slots=True)
class HybridCandidateAgnosticSourceBindingConfig:
    """Pinned active-source inputs used before any outcome access."""

    run_id: str
    output_dir: Path | str
    readiness_manifest_path: Path | str
    expected_readiness_manifest_sha256: str
    precollection_freeze_manifest_path: Path | str
    expected_precollection_freeze_manifest_sha256: str
    source_pre_registration_manifest_path: Path | str
    expected_source_pre_registration_manifest_sha256: str
    source_collection_freeze_manifest_path: Path | str
    expected_source_collection_freeze_manifest_sha256: str
    source_batch_progress_path: Path | str
    expected_source_batch_progress_sha256: str
    source_batch_id: str
    source_raw_root: Path | str
    builder_git_commit: str
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip() or not self.source_batch_id.strip():
            raise ValueError("run_id and source_batch_id are required")
        for name in (
            "expected_readiness_manifest_sha256",
            "expected_precollection_freeze_manifest_sha256",
            "expected_source_pre_registration_manifest_sha256",
            "expected_source_collection_freeze_manifest_sha256",
            "expected_source_batch_progress_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        if len(self.builder_git_commit) != 40 or any(
            character not in "0123456789abcdef"
            for character in self.builder_git_commit.lower()
        ):
            raise ValueError("builder_git_commit must be a 40-character hex digest")
        for name in (
            "output_dir",
            "readiness_manifest_path",
            "precollection_freeze_manifest_path",
            "source_pre_registration_manifest_path",
            "source_collection_freeze_manifest_path",
            "source_batch_progress_path",
            "source_raw_root",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class HybridCandidateAgnosticSourceSnapshotConfig:
    """Terminal source used to freeze the first 150 capture attempts."""

    run_id: str
    output_dir: Path | str
    binding_manifest_path: Path | str
    expected_binding_manifest_sha256: str
    terminal_batch_progress_path: Path | str
    expected_terminal_batch_progress_sha256: str
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(
            self.expected_binding_manifest_sha256,
            name="binding manifest SHA-256",
        )
        _require_sha256(
            self.expected_terminal_batch_progress_sha256,
            name="terminal batch progress SHA-256",
        )
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(
            self,
            "binding_manifest_path",
            Path(self.binding_manifest_path),
        )
        object.__setattr__(
            self,
            "terminal_batch_progress_path",
            Path(self.terminal_batch_progress_path),
        )


@dataclass(frozen=True, slots=True)
class HybridBoundFreshRoleAssignmentConfig:
    """Finalized first-150 evidence adapted into the existing #185 role schema."""

    run_id: str
    output_dir: Path | str
    snapshot_manifest_path: Path | str
    expected_snapshot_manifest_sha256: str
    finalized_batch_progress_path: Path | str
    expected_finalized_batch_progress_sha256: str
    training_corpus_root: Path | str = Path("/Volumes/PHILIPS/v8")
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(
            self.expected_snapshot_manifest_sha256,
            name="snapshot manifest SHA-256",
        )
        _require_sha256(
            self.expected_finalized_batch_progress_sha256,
            name="finalized batch progress SHA-256",
        )
        for name in (
            "output_dir",
            "snapshot_manifest_path",
            "finalized_batch_progress_path",
            "training_corpus_root",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class HybridBoundedFinalizationViewConfig:
    """Build a filesystem view that exposes only snapshot-allowlisted rounds."""

    run_id: str
    output_dir: Path | str
    snapshot_manifest_path: Path | str
    expected_snapshot_manifest_sha256: str
    finalizer_script_path: Path | str
    expected_finalizer_script_sha256: str
    finalizer_git_commit: str
    python_executable: Path | str
    training_corpus_root: Path | str = Path("/Volumes/PHILIPS/v8")
    settlement_poll_interval_seconds: float = 15.0
    settlement_grace_seconds: float = 1_200.0
    public_provider_http_timeout_seconds: float = 5.0
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(
            self.expected_snapshot_manifest_sha256,
            name="snapshot manifest SHA-256",
        )
        _require_sha256(
            self.expected_finalizer_script_sha256,
            name="finalizer script SHA-256",
        )
        if len(self.finalizer_git_commit) != 40 or any(
            character not in "0123456789abcdef"
            for character in self.finalizer_git_commit.lower()
        ):
            raise ValueError("finalizer_git_commit must be a 40-character hex digest")
        if self.settlement_poll_interval_seconds <= 0:
            raise ValueError("settlement_poll_interval_seconds must be positive")
        if self.settlement_grace_seconds < 0:
            raise ValueError("settlement_grace_seconds must be non-negative")
        if self.public_provider_http_timeout_seconds <= 0:
            raise ValueError("public_provider_http_timeout_seconds must be positive")
        for name in (
            "output_dir",
            "snapshot_manifest_path",
            "finalizer_script_path",
            "python_executable",
            "training_corpus_root",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


def create_hybrid_candidate_agnostic_source_binding(
    config: HybridCandidateAgnosticSourceBindingConfig,
) -> dict[str, Any]:
    """Freeze source identity and scan boundaries while collection stays sealed."""

    paths_and_hashes = (
        (
            config.readiness_manifest_path.resolve(),
            config.expected_readiness_manifest_sha256,
            "#183 readiness manifest",
        ),
        (
            config.precollection_freeze_manifest_path.resolve(),
            config.expected_precollection_freeze_manifest_sha256,
            "#183 precollection freeze manifest",
        ),
        (
            config.source_pre_registration_manifest_path.resolve(),
            config.expected_source_pre_registration_manifest_sha256,
            "#190 pre-registration manifest",
        ),
        (
            config.source_collection_freeze_manifest_path.resolve(),
            config.expected_source_collection_freeze_manifest_sha256,
            "#190 collection freeze manifest",
        ),
        (
            config.source_batch_progress_path.resolve(),
            config.expected_source_batch_progress_sha256,
            "#190 batch progress snapshot",
        ),
    )
    for path, digest, name in paths_and_hashes:
        _verify_pin(path, digest, name=name)
    readiness = _load_json(paths_and_hashes[0][0])
    freeze = _load_json(paths_and_hashes[1][0])
    pre_registration = _load_json(paths_and_hashes[2][0])
    collection_freeze = _load_json(paths_and_hashes[3][0])
    batch = _load_json(paths_and_hashes[4][0])
    source_identity_audit = _source_market_identity_audit(
        freeze=freeze,
        batch=batch,
    )
    blockers = _binding_blockers(
        readiness=readiness,
        readiness_path=paths_and_hashes[0][0],
        freeze=freeze,
        freeze_path=paths_and_hashes[1][0],
        pre_registration=pre_registration,
        pre_registration_path=paths_and_hashes[2][0],
        collection_freeze=collection_freeze,
        collection_freeze_path=paths_and_hashes[3][0],
        batch=batch,
        source_batch_id=config.source_batch_id,
        source_raw_root=config.source_raw_root.resolve(),
        source_identity_audit=source_identity_audit,
    )
    run_dir = _prepare_run_dir(
        config.output_dir,
        config.run_id,
        overwrite=config.overwrite_existing,
    )
    current_capture_count = int(batch.get("capture_count") or 0)
    terminal = str(batch.get("collection_stop_reason") or "") in (
        TERMINAL_COLLECTION_STOP_REASONS
    )
    identity_snapshot_path = (
        run_dir / "hybrid_pairwise_candidate_agnostic_source_identity_snapshot.json"
    )
    initial_batch_snapshot_path = (
        run_dir / "hybrid_pairwise_candidate_agnostic_initial_batch_snapshot.json"
    )
    _write_json(identity_snapshot_path, source_identity_audit)
    _write_json(initial_batch_snapshot_path, batch)
    report = {
        "schema_version": BINDING_REPORT_SCHEMA_VERSION,
        "run_id": config.run_id,
        "status": "binding_ready" if not blockers else "blocked_fail_closed",
        "source_binding_ready": not blockers,
        "source_collection_terminal": terminal,
        "source_snapshot_allowed": not blockers
        and terminal
        and current_capture_count >= MAXIMUM_CAPTURE_ATTEMPT_COUNT,
        "source_batch_id": config.source_batch_id,
        "source_raw_root": str(config.source_raw_root.resolve()),
        "source_current_capture_count": current_capture_count,
        "source_current_quality_valid_capture_count": int(
            batch.get("outcome_blind_quality_valid_capture_count") or 0
        ),
        "source_observed_market_count": source_identity_audit[
            "observed_market_count"
        ],
        "source_prior_market_overlap_count": source_identity_audit[
            "prior_market_overlap_count"
        ],
        "source_observed_market_ids_sha256": source_identity_audit[
            "observed_market_ids_sha256"
        ],
        "initial_capture_attempt_count": INITIAL_CAPTURE_ATTEMPT_COUNT,
        "maximum_source_capture_attempt_ordinal": MAXIMUM_CAPTURE_ATTEMPT_COUNT,
        "target_fresh_unique_market_count": 105,
        "fresh_development_calibration_market_count": 45,
        "fresh_confirmatory_validation_market_count": 60,
        "source_attempts_after_150_eligible": False,
        "duplicate_collector_started": False,
        "terminal_state_required_before_finalization": True,
        "finalization_attempted": False,
        "training_corpus_export_attempted": False,
        "labels_or_outcomes_opened": False,
        "confirmatory_labels_opened": False,
        "uses_issue189_oof_development_calibration_or_pnl": False,
        "blocking_reason_codes": sorted(set(blockers)),
        **_hybrid_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "hybrid_pairwise_candidate_agnostic_source_binding_report.json"
    markdown_path = run_dir / "hybrid_pairwise_candidate_agnostic_source_binding_report.md"
    _write_json(report_path, report)
    _write_text(markdown_path, _binding_markdown(report))
    manifest = {
        "schema_version": BINDING_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "builder_git_commit": config.builder_git_commit.lower(),
        "source_binding_ready": report["source_binding_ready"],
        "readiness_manifest": _descriptor(paths_and_hashes[0][0]),
        "hybrid_precollection_freeze": _descriptor(paths_and_hashes[1][0]),
        "source_pre_registration_manifest": _descriptor(paths_and_hashes[2][0]),
        "source_collection_freeze_manifest": _descriptor(paths_and_hashes[3][0]),
        "source_initial_batch_progress_snapshot": _descriptor(
            initial_batch_snapshot_path
        ),
        "source_market_identity_snapshot": _descriptor(identity_snapshot_path),
        "source_batch_id": config.source_batch_id,
        "source_batch_progress_path": str(paths_and_hashes[4][0]),
        "source_raw_root": str(config.source_raw_root.resolve()),
        "initial_capture_attempt_count": INITIAL_CAPTURE_ATTEMPT_COUNT,
        "maximum_source_capture_attempt_ordinal": MAXIMUM_CAPTURE_ATTEMPT_COUNT,
        "selection_method": (
            "earliest_105_quality_valid_unique_markets_from_first_150_attempts"
        ),
        "roles": {
            "fresh_development_calibration": 45,
            "fresh_confirmatory_validation": 60,
        },
        "terminal_state_required_before_finalization": True,
        "labels_or_outcomes_opened": False,
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(markdown_path),
        "blocking_reason_codes": report["blocking_reason_codes"],
        **_hybrid_safety_fields(),
    }
    manifest["binding_manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "hybrid_pairwise_candidate_agnostic_source_binding_manifest.json"
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


def freeze_hybrid_candidate_agnostic_source_snapshot(
    config: HybridCandidateAgnosticSourceSnapshotConfig,
) -> dict[str, Any]:
    """Freeze exactly attempts 1-150 after the source collector is terminal."""

    binding_path = config.binding_manifest_path.resolve()
    batch_path = config.terminal_batch_progress_path.resolve()
    _verify_pin(
        binding_path,
        config.expected_binding_manifest_sha256,
        name="source binding manifest",
    )
    _verify_pin(
        batch_path,
        config.expected_terminal_batch_progress_sha256,
        name="terminal source batch progress",
    )
    binding = _load_json(binding_path)
    batch = _load_json(batch_path)
    blockers = _snapshot_blockers(binding=binding, batch=batch, batch_path=batch_path)
    run_dir = _prepare_run_dir(
        config.output_dir,
        config.run_id,
        overwrite=config.overwrite_existing,
    )
    ordered_captures = _ordered_unique_captures(batch)
    selected_captures = ordered_captures[:MAXIMUM_CAPTURE_ATTEMPT_COUNT]
    rows_path = run_dir / "hybrid_pairwise_candidate_agnostic_source_snapshot_rows.jsonl"
    allowlist_path = run_dir / "hybrid_pairwise_candidate_agnostic_finalization_allowlist.json"
    if not blockers:
        _write_jsonl(rows_path, selected_captures)
        allowlist = {
            "schema_version": f"{SCHEMA_PREFIX}-finalization-allowlist-v1",
            "run_id": config.run_id,
            "source_binding_manifest": _descriptor(binding_path),
            "source_terminal_batch_progress": _descriptor(batch_path),
            "allowed_capture_attempt_count": len(selected_captures),
            "maximum_source_capture_attempt_ordinal": MAXIMUM_CAPTURE_ATTEMPT_COUNT,
            "run_ids": [str(row["run_id"]) for row in selected_captures],
            "run_dirs": [str(row["run_dir"]) for row in selected_captures],
            "capture_row_sha256s": [
                canonical_json_sha256(row) for row in selected_captures
            ],
            "attempts_after_150_allowed": False,
            "read_only_finalization_only": True,
            "labels_or_outcomes_opened_for_allowlist_creation": False,
            **_hybrid_safety_fields(),
        }
        allowlist["allowlist_id"] = canonical_json_sha256(allowlist)
        _write_json(allowlist_path, allowlist)
    report = {
        "schema_version": SNAPSHOT_REPORT_SCHEMA_VERSION,
        "run_id": config.run_id,
        "status": "snapshot_ready" if not blockers else "blocked_fail_closed",
        "source_snapshot_ready": not blockers,
        "source_collection_terminal": str(batch.get("collection_stop_reason") or "")
        in TERMINAL_COLLECTION_STOP_REASONS,
        "source_total_capture_count": int(batch.get("capture_count") or 0),
        "bounded_capture_attempt_count": (
            len(selected_captures) if not blockers else 0
        ),
        "initial_capture_attempt_count": INITIAL_CAPTURE_ATTEMPT_COUNT,
        "maximum_source_capture_attempt_ordinal": MAXIMUM_CAPTURE_ATTEMPT_COUNT,
        "attempts_after_150_included": False,
        "duplicate_capture_run_id_count": (
            len(batch.get("captures") or []) - len(ordered_captures)
        ),
        "finalization_attempted": False,
        "labels_or_outcomes_opened": False,
        "blocking_reason_codes": sorted(set(blockers)),
        **_hybrid_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "hybrid_pairwise_candidate_agnostic_source_snapshot_report.json"
    _write_json(report_path, report)
    manifest = {
        "schema_version": SNAPSHOT_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "source_snapshot_ready": report["source_snapshot_ready"],
        "source_binding_manifest": _descriptor(binding_path),
        "source_terminal_batch_progress": _descriptor(batch_path),
        "bounded_capture_rows": _descriptor(rows_path) if not blockers else None,
        "finalization_allowlist": _descriptor(allowlist_path) if not blockers else None,
        "bounded_capture_attempt_count": report["bounded_capture_attempt_count"],
        "maximum_source_capture_attempt_ordinal": MAXIMUM_CAPTURE_ATTEMPT_COUNT,
        "attempts_after_150_included": False,
        "labels_or_outcomes_opened": False,
        "report": _descriptor(report_path),
        "blocking_reason_codes": report["blocking_reason_codes"],
        **_hybrid_safety_fields(),
    }
    manifest["snapshot_manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "hybrid_pairwise_candidate_agnostic_source_snapshot_manifest.json"
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


def prepare_hybrid_bounded_finalization_view(
    config: HybridBoundedFinalizationViewConfig,
) -> dict[str, Any]:
    """Expose only attempts 1-150 to the existing read-only finalizer."""

    snapshot_path = config.snapshot_manifest_path.resolve()
    finalizer_script_path = config.finalizer_script_path.resolve()
    _verify_pin(
        snapshot_path,
        config.expected_snapshot_manifest_sha256,
        name="source snapshot manifest",
    )
    _verify_pin(
        finalizer_script_path,
        config.expected_finalizer_script_sha256,
        name="frozen finalizer script",
    )
    snapshot = _load_json(snapshot_path)
    if snapshot.get("schema_version") != SNAPSHOT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("source snapshot manifest schema mismatch")
    if snapshot.get("source_snapshot_ready") is not True:
        raise ValueError("source snapshot is not ready")
    rows_descriptor = _verified_descriptor(
        snapshot.get("bounded_capture_rows"),
        name="bounded capture rows",
    )
    allowlist_descriptor = _verified_descriptor(
        snapshot.get("finalization_allowlist"),
        name="bounded finalization allowlist",
    )
    terminal_batch_descriptor = _verified_descriptor(
        snapshot.get("source_terminal_batch_progress"),
        name="source terminal batch progress",
    )
    rows = _load_jsonl(Path(rows_descriptor["path"]))
    allowlist = _load_json(Path(allowlist_descriptor["path"]))
    source_batch_path = Path(terminal_batch_descriptor["path"])
    source_batch_hash_before = _sha256_file(source_batch_path)
    source_batch = _load_json(source_batch_path)
    blockers = _finalization_view_blockers(
        snapshot=snapshot,
        rows=rows,
        allowlist=allowlist,
        source_batch=source_batch,
    )
    if blockers:
        raise ValueError(
            "bounded finalization view validation failed: "
            + ", ".join(sorted(set(blockers)))
        )
    run_dir = _prepare_run_dir(
        config.output_dir,
        config.run_id,
        overwrite=config.overwrite_existing,
    )
    view_root = run_dir / "bounded_finalization_view"
    view_root.mkdir()
    source_batch_id = str(source_batch["batch_id"])
    view_batch_dir = view_root / source_batch_id
    view_batch_dir.mkdir()
    selected_run_ids = {str(row["run_id"]) for row in rows}
    bounded_finalizations = [
        dict(row)
        for row in source_batch.get("finalizations") or []
        if str(row.get("run_id") or "") in selected_run_ids
    ]
    bounded_batch = {
        **{
            key: value
            for key, value in source_batch.items()
            if key not in {"captures", "finalizations"}
        },
        "capture_count": len(rows),
        "captures": rows,
        "finalizations": bounded_finalizations,
        "finalization_attempt_count": len(bounded_finalizations),
        "hybrid_source_capture_attempt_limit": MAXIMUM_CAPTURE_ATTEMPT_COUNT,
        "source_attempts_after_150_included": False,
    }
    view_batch_progress_path = view_batch_dir / "batch_progress.json"
    _write_json(view_batch_progress_path, bounded_batch)
    symlink_rows: list[dict[str, Any]] = []
    for row in rows:
        source_run_dir = Path(str(row["run_dir"])).resolve()
        link_path = view_root / str(row["run_id"])
        link_path.symlink_to(source_run_dir, target_is_directory=True)
        symlink_rows.append(
            {
                "run_id": row["run_id"],
                "source_run_dir": str(source_run_dir),
                "bounded_view_path": str(link_path),
                "capture_row_sha256": canonical_json_sha256(row),
            }
        )
    symlink_rows_path = run_dir / "hybrid_pairwise_bounded_finalization_view_rows.jsonl"
    _write_jsonl(symlink_rows_path, symlink_rows)
    command_argv = [
        str(config.python_executable.resolve()),
        str(finalizer_script_path),
        "--batch-id",
        source_batch_id,
        "--output-dir",
        str(view_root),
        "--finalize-only",
        "--training-corpus-root",
        str(config.training_corpus_root.expanduser().resolve()),
        "--settlement-poll-interval-seconds",
        str(config.settlement_poll_interval_seconds),
        "--settlement-grace-seconds",
        str(config.settlement_grace_seconds),
        "--public-provider-http-timeout-seconds",
        str(config.public_provider_http_timeout_seconds),
    ]
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-bounded-finalization-view-report-v1",
        "run_id": config.run_id,
        "status": "bounded_finalization_view_ready",
        "bounded_finalization_view_ready": True,
        "bounded_capture_attempt_count": len(rows),
        "maximum_source_capture_attempt_ordinal": MAXIMUM_CAPTURE_ATTEMPT_COUNT,
        "source_attempts_after_150_included": False,
        "source_batch_progress_sha256_before_view_creation": source_batch_hash_before,
        "source_batch_progress_sha256_after_view_creation": _sha256_file(
            source_batch_path
        ),
        "source_batch_progress_mutated": (
            source_batch_hash_before != _sha256_file(source_batch_path)
        ),
        "finalizer_script": _descriptor(finalizer_script_path),
        "finalizer_git_commit": config.finalizer_git_commit.lower(),
        "finalizer_command_generated": True,
        "finalizer_executed": False,
        "source_collection_was_terminal_before_view_creation": True,
        "labels_or_outcomes_opened_for_view_creation": False,
        **_hybrid_safety_fields(),
    }
    if report["source_batch_progress_mutated"]:
        raise ValueError("source batch progress mutated during view creation")
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "hybrid_pairwise_bounded_finalization_view_report.json"
    _write_json(report_path, report)
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-bounded-finalization-view-manifest-v1",
        "run_id": config.run_id,
        "source_snapshot_manifest": _descriptor(snapshot_path),
        "source_terminal_batch_progress": terminal_batch_descriptor,
        "bounded_view_batch_progress": _descriptor(view_batch_progress_path),
        "bounded_view_rows": _descriptor(symlink_rows_path),
        "bounded_view_root": str(view_root),
        "bounded_batch_id": source_batch_id,
        "bounded_capture_attempt_count": len(rows),
        "source_attempts_after_150_included": False,
        "finalizer_script": _descriptor(finalizer_script_path),
        "finalizer_git_commit": config.finalizer_git_commit.lower(),
        "finalizer_command_argv": command_argv,
        "finalizer_executed": False,
        "labels_or_outcomes_opened": False,
        "report": _descriptor(report_path),
        **_hybrid_safety_fields(),
    }
    manifest["bounded_finalization_view_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "hybrid_pairwise_bounded_finalization_view_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "view_root": view_root,
        "view_batch_progress_path": view_batch_progress_path,
        "view_batch_progress_sha256": _sha256_file(view_batch_progress_path),
        "report": report,
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def assign_bound_hybrid_fresh_roles(
    config: HybridBoundFreshRoleAssignmentConfig,
) -> dict[str, Any]:
    """Use only finalized captures whose identities were frozen in the snapshot."""

    snapshot_path = config.snapshot_manifest_path.resolve()
    finalized_path = config.finalized_batch_progress_path.resolve()
    _verify_pin(
        snapshot_path,
        config.expected_snapshot_manifest_sha256,
        name="source snapshot manifest",
    )
    _verify_pin(
        finalized_path,
        config.expected_finalized_batch_progress_sha256,
        name="finalized source batch progress",
    )
    snapshot = _load_json(snapshot_path)
    if snapshot.get("schema_version") != SNAPSHOT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("source snapshot manifest schema mismatch")
    if snapshot.get("source_snapshot_ready") is not True:
        raise ValueError("source snapshot is not ready")
    rows_descriptor = _verified_descriptor(
        snapshot.get("bounded_capture_rows"),
        name="bounded capture rows",
    )
    snapshot_rows = _load_jsonl(Path(rows_descriptor["path"]))
    finalized = _load_json(finalized_path)
    bounded_batch, blockers = _bounded_finalized_batch(
        snapshot_rows=snapshot_rows,
        finalized_batch=finalized,
    )
    if blockers:
        raise ValueError(
            "bounded finalized source validation failed: "
            + ", ".join(sorted(set(blockers)))
        )
    binding_descriptor = _verified_descriptor(
        snapshot.get("source_binding_manifest"),
        name="source binding manifest",
    )
    binding = _load_json(Path(binding_descriptor["path"]))
    freeze_descriptor = _verified_descriptor(
        binding.get("hybrid_precollection_freeze"),
        name="hybrid precollection freeze",
    )
    freeze_path = Path(freeze_descriptor["path"])
    freeze = _load_json(freeze_path)
    source_protocol_descriptor = _verified_descriptor(
        freeze.get("source_pairwise_protocol"),
        name="source pairwise protocol",
    )
    source_protocol = _load_json(Path(source_protocol_descriptor["path"]))
    validate_pairwise_action_advantage_lcb_protocol(source_protocol)
    collector_contract = dict(source_protocol["collector_contract"])
    training_root = config.training_corpus_root.expanduser().resolve()
    if training_root != Path(
        str(collector_contract["training_corpus_root"])
    ).expanduser().resolve():
        raise ValueError("training corpus root drift")
    run_dir = _prepare_run_dir(
        config.output_dir,
        config.run_id,
        overwrite=config.overwrite_existing,
    )
    bounded_path = run_dir / "hybrid_pairwise_bounded_finalized_batch_progress.json"
    _write_json(bounded_path, bounded_batch)
    source_plan = {
        "schema_version": BOUND_SOURCE_PLAN_SCHEMA_VERSION,
        "run_id": config.run_id,
        "source_binding_manifest": binding_descriptor,
        "source_snapshot_manifest": _descriptor(snapshot_path),
        "source_finalized_batch_progress": _descriptor(finalized_path),
        "initial_capture_attempt_count": INITIAL_CAPTURE_ATTEMPT_COUNT,
        "maximum_total_capture_attempt_count": MAXIMUM_CAPTURE_ATTEMPT_COUNT,
        "selection_method": (
            "earliest_105_quality_valid_unique_markets_from_first_150_attempts"
        ),
        "duplicate_collector_started": False,
        "labels_or_outcomes_opened_for_role_assignment": False,
        **_hybrid_safety_fields(),
    }
    source_plan["launch_plan_id"] = canonical_json_sha256(source_plan)
    source_plan_path = run_dir / "hybrid_pairwise_bound_source_plan.json"
    _write_json(source_plan_path, source_plan)
    role_result = _assign_hybrid_fresh_roles(
        run_id=config.run_id,
        run_dir=run_dir,
        freeze=freeze,
        freeze_path=freeze_path,
        launch_plan=source_plan,
        launch_plan_path=source_plan_path,
        batch_progress_pins=((bounded_path, _sha256_file(bounded_path)),),
        collector_contract=collector_contract,
        training_root=training_root,
    )
    return {
        "run_dir": run_dir,
        "bounded_batch_progress_path": bounded_path,
        "bounded_batch_progress_sha256": _sha256_file(bounded_path),
        "source_plan_path": source_plan_path,
        "source_plan_sha256": _sha256_file(source_plan_path),
        "role_assignment_result": role_result,
    }


def _binding_blockers(
    *,
    readiness: dict[str, Any],
    readiness_path: Path,
    freeze: dict[str, Any],
    freeze_path: Path,
    pre_registration: dict[str, Any],
    pre_registration_path: Path,
    collection_freeze: dict[str, Any],
    collection_freeze_path: Path,
    batch: dict[str, Any],
    source_batch_id: str,
    source_raw_root: Path,
    source_identity_audit: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if readiness.get("schema_version") != READINESS_MANIFEST_SCHEMA_VERSION:
        blockers.append("issue183_readiness_schema_mismatch")
    if readiness.get("precollection_readiness_passed") is not True:
        blockers.append("issue183_precollection_readiness_not_passed")
    if readiness.get("precollection_freeze_created") is not True:
        blockers.append("issue183_precollection_freeze_not_created")
    if freeze.get("schema_version") != PRECOLLECTION_FREEZE_SCHEMA_VERSION:
        blockers.append("issue183_precollection_freeze_schema_mismatch")
    if readiness.get("precollection_freeze_manifest") != _descriptor(freeze_path):
        blockers.append("issue183_readiness_freeze_lineage_mismatch")
    if freeze.get("collection_started") is not False:
        blockers.append("issue183_freeze_collection_state_invalid")
    if freeze.get("labels_or_outcomes_opened_for_role_assignment") is not False:
        blockers.append("issue183_freeze_outcome_blindness_failed")
    if pre_registration.get("schema_version") != FUTURE_PREREGISTRATION_SCHEMA_VERSION:
        blockers.append("issue190_pre_registration_schema_mismatch")
    for field, expected in (
        ("pre_registration_ready", True),
        ("candidate_agnostic_raw_collection", True),
        ("outcome_blind_collection_only_required", True),
        ("collection_stop_rule_is_outcome_blind", True),
        ("holdout_labels_or_outcomes_opened_before_pre_registration", False),
        ("settlement_finalizer_started_during_collection", False),
        ("training_corpus_export_during_collection_allowed", False),
        ("collection_control_uses_model_scores_bets_or_pnl", False),
    ):
        if pre_registration.get(field) is not expected:
            blockers.append(f"issue190_pre_registration_{field}_invalid")
    if collection_freeze.get("schema_version") != FUTURE_COLLECTION_FREEZE_SCHEMA_VERSION:
        blockers.append("issue190_collection_freeze_schema_mismatch")
    if collection_freeze.get("pre_registration_manifest") != _descriptor(
        pre_registration_path
    ):
        blockers.append("issue190_collection_freeze_pre_registration_mismatch")
    for field, expected in (
        ("collection_control_is_outcome_blind", True),
        ("outcome_blind_collection_only_required", True),
        ("labels_or_outcomes_opened_for_collection_freeze", False),
        ("settlement_finalizer_started_during_collection", False),
        ("training_corpus_export_during_collection_allowed", False),
    ):
        if collection_freeze.get(field) is not expected:
            blockers.append(f"issue190_collection_freeze_{field}_invalid")
    if int(collection_freeze.get("minimum_collection_decision_ts") or 0) <= int(
        freeze.get("minimum_collection_decision_ts") or 0
    ):
        blockers.append("issue190_source_not_strictly_later_than_issue183_freeze")
    if batch.get("batch_id") != source_batch_id:
        blockers.append("issue190_source_batch_id_mismatch")
    if batch.get("future_holdout_collection_freeze_manifest") != _descriptor(
        collection_freeze_path
    ):
        blockers.append("issue190_batch_collection_freeze_mismatch")
    for field, expected in (
        ("outcome_blind_collection_only", True),
        ("labels_or_outcomes_opened_during_collection", False),
        ("labels_or_outcomes_opened_for_collection_control", False),
        ("resolution_provider_called", False),
        ("settlement_finalizer_started", False),
        ("training_corpus_export_attempted", False),
        ("uses_accepted_bet_count_for_collection_control", False),
        ("uses_model_scores_for_collection_control", False),
    ):
        if batch.get(field) is not expected:
            blockers.append(f"issue190_batch_{field}_invalid")
    if _find_fields(batch, FORBIDDEN_SOURCE_FIELDS):
        blockers.append("issue190_batch_forbidden_outcome_field_present")
    blockers.extend(source_identity_audit["blocking_reason_codes"])
    captures = [dict(row) for row in batch.get("captures") or []]
    if len(captures) != int(batch.get("capture_count") or 0):
        blockers.append("issue190_batch_capture_count_mismatch")
    source_protocol_descriptor = _verified_descriptor(
        freeze.get("source_pairwise_protocol"),
        name="source pairwise protocol",
    )
    source_protocol = _load_json(Path(source_protocol_descriptor["path"]))
    validate_pairwise_action_advantage_lcb_protocol(source_protocol)
    collector_contract = dict(source_protocol["collector_contract"])
    for capture in captures:
        if Path(str(capture.get("run_dir") or "")).resolve().parent != source_raw_root:
            blockers.append("issue190_capture_outside_bound_raw_root")
        if int(capture.get("scheduled_round_start_ts") or 0) < int(
            collection_freeze.get("minimum_collection_decision_ts") or 0
        ):
            blockers.append("issue190_capture_before_collection_boundary")
        if capture.get("market_family") != collector_contract.get("market_family"):
            blockers.append("issue190_capture_market_family_drift")
        for field in (
            "public_provider_timeout_seconds",
            "public_provider_http_timeout_seconds",
            "orderbook_snapshot_interval_seconds",
            "orderbook_ws_initial_complete_book_timeout_seconds",
            "rest_orderbook_fallback_collection_seconds",
            "feature_enrichment_max_attempts",
        ):
            if float(capture.get(field) or 0.0) != float(
                collector_contract.get(field) or 0.0
            ):
                blockers.append(f"issue190_capture_{field}_drift")
    if not _safety_blocked(readiness, freeze, pre_registration, collection_freeze, batch):
        blockers.append("source_binding_safety_contract_failed")
    return blockers


def _source_market_identity_audit(
    *,
    freeze: dict[str, Any],
    batch: dict[str, Any],
) -> dict[str, Any]:
    quarantine_descriptor = _verified_descriptor(
        freeze.get("final_prior_lineage_quarantine"),
        name="final prior lineage quarantine",
    )
    quarantine = _load_json(Path(quarantine_descriptor["path"]))
    prior_market_ids = {
        str(value) for value in quarantine.get("prior_market_ids") or []
    }
    observed_market_ids: set[str] = set()
    raw_market_descriptors: list[dict[str, str]] = []
    blockers: list[str] = []
    if not prior_market_ids or "" in prior_market_ids:
        blockers.append("issue183_prior_quarantine_market_identity_incomplete")
    if canonical_json_sha256(sorted(prior_market_ids)) != str(
        quarantine.get("prior_market_ids_sha256") or ""
    ):
        blockers.append("issue183_prior_quarantine_market_identity_hash_mismatch")
    minimum_collection_decision_ts = int(
        freeze.get("minimum_collection_decision_ts") or 0
    )
    for capture in batch.get("captures") or []:
        run_dir = Path(str(capture.get("run_dir") or "")).resolve()
        raw_market_path = run_dir / "raw/raw_polymarket_markets.jsonl"
        if not raw_market_path.is_file():
            blockers.append("source_raw_market_identity_artifact_missing")
            continue
        raw_rows = _load_jsonl(raw_market_path)
        market_ids = {
            str(row.get("market_id") or row.get("condition_id") or "")
            for row in raw_rows
        }
        if len(market_ids) != 1 or "" in market_ids:
            blockers.append("source_raw_market_identity_not_unique")
            continue
        market_id = next(iter(market_ids))
        observed_market_ids.add(market_id)
        if any(
            int(row.get("market_start_ts") or 0) < minimum_collection_decision_ts
            for row in raw_rows
        ):
            blockers.append("source_raw_market_before_issue183_boundary")
        if _find_fields({"rows": raw_rows}, FORBIDDEN_SOURCE_FIELDS):
            blockers.append("source_raw_market_forbidden_outcome_field_present")
        raw_market_descriptors.append(_descriptor(raw_market_path))
    prior_overlap = observed_market_ids & prior_market_ids
    if prior_overlap:
        blockers.append("source_market_overlaps_issue183_prior_quarantine")
    if len(observed_market_ids) != len(batch.get("captures") or []):
        blockers.append("source_capture_market_identity_coverage_incomplete")
    return {
        "schema_version": f"{SCHEMA_PREFIX}-market-identity-audit-v1",
        "observed_market_count": len(observed_market_ids),
        "observed_market_ids_sha256": canonical_json_sha256(
            sorted(observed_market_ids)
        ),
        "prior_market_count": len(prior_market_ids),
        "prior_market_ids_sha256": canonical_json_sha256(sorted(prior_market_ids)),
        "prior_market_overlap_count": len(prior_overlap),
        "prior_market_overlap_ids_sha256": canonical_json_sha256(
            sorted(prior_overlap)
        ),
        "raw_market_artifacts": raw_market_descriptors,
        "blocking_reason_codes": sorted(set(blockers)),
        "labels_or_outcomes_opened": False,
    }


def _snapshot_blockers(
    *,
    binding: dict[str, Any],
    batch: dict[str, Any],
    batch_path: Path,
) -> list[str]:
    blockers: list[str] = []
    if binding.get("schema_version") != BINDING_MANIFEST_SCHEMA_VERSION:
        blockers.append("source_binding_manifest_schema_mismatch")
    if binding.get("source_binding_ready") is not True:
        blockers.append("source_binding_not_ready")
    try:
        readiness_descriptor = _verified_descriptor(
            binding.get("readiness_manifest"),
            name="binding readiness manifest",
        )
        freeze_descriptor = _verified_descriptor(
            binding.get("hybrid_precollection_freeze"),
            name="binding precollection freeze",
        )
        pre_registration_descriptor = _verified_descriptor(
            binding.get("source_pre_registration_manifest"),
            name="binding source pre-registration",
        )
        collection_freeze_descriptor = _verified_descriptor(
            binding.get("source_collection_freeze_manifest"),
            name="binding source collection freeze",
        )
        readiness = _load_json(Path(readiness_descriptor["path"]))
        freeze = _load_json(Path(freeze_descriptor["path"]))
        pre_registration = _load_json(Path(pre_registration_descriptor["path"]))
        collection_freeze = _load_json(Path(collection_freeze_descriptor["path"]))
        terminal_identity_audit = _source_market_identity_audit(
            freeze=freeze,
            batch=batch,
        )
        blockers.extend(
            _binding_blockers(
                readiness=readiness,
                readiness_path=Path(readiness_descriptor["path"]),
                freeze=freeze,
                freeze_path=Path(freeze_descriptor["path"]),
                pre_registration=pre_registration,
                pre_registration_path=Path(pre_registration_descriptor["path"]),
                collection_freeze=collection_freeze,
                collection_freeze_path=Path(collection_freeze_descriptor["path"]),
                batch=batch,
                source_batch_id=str(binding.get("source_batch_id") or ""),
                source_raw_root=Path(str(binding.get("source_raw_root") or "")),
                source_identity_audit=terminal_identity_audit,
            )
        )
    except (TypeError, ValueError) as exc:
        blockers.append(f"source_terminal_lineage_validation_failed:{exc}")
    if binding.get("source_batch_progress_path") != str(batch_path):
        blockers.append("terminal_batch_path_not_bound")
    if batch.get("batch_id") != binding.get("source_batch_id"):
        blockers.append("terminal_batch_id_not_bound")
    if batch.get("future_holdout_collection_freeze_manifest") != binding.get(
        "source_collection_freeze_manifest"
    ):
        blockers.append("terminal_batch_collection_freeze_mismatch")
    if str(batch.get("collection_stop_reason") or "") not in (
        TERMINAL_COLLECTION_STOP_REASONS
    ):
        blockers.append("source_collection_not_terminal")
    captures = [dict(row) for row in batch.get("captures") or []]
    if len(captures) < MAXIMUM_CAPTURE_ATTEMPT_COUNT:
        blockers.append("source_first_150_capture_attempts_incomplete")
    if len(captures) != int(batch.get("capture_count") or 0):
        blockers.append("source_terminal_capture_count_mismatch")
    ordered = _ordered_unique_captures(batch)
    if len(ordered) != len(captures):
        blockers.append("source_duplicate_capture_identity_detected")
    if [int(row.get("round_index") or 0) for row in ordered[:150]] != list(
        range(1, 151)
    ):
        blockers.append("source_first_150_round_index_sequence_invalid")
    for field, expected in (
        ("outcome_blind_collection_only", True),
        ("labels_or_outcomes_opened_during_collection", False),
        ("resolution_provider_called", False),
        ("settlement_finalizer_started", False),
        ("training_corpus_export_attempted", False),
    ):
        if batch.get(field) is not expected:
            blockers.append(f"source_terminal_{field}_invalid")
    if _find_fields(batch, FORBIDDEN_SOURCE_FIELDS):
        blockers.append("source_terminal_forbidden_outcome_field_present")
    if not _safety_blocked(binding, batch):
        blockers.append("source_snapshot_safety_contract_failed")
    return blockers


def _ordered_unique_captures(batch: dict[str, Any]) -> list[dict[str, Any]]:
    captures = sorted(
        [dict(row) for row in batch.get("captures") or []],
        key=lambda row: (
            int(row.get("round_index") or 0),
            int(row.get("scheduled_round_start_ts") or 0),
            str(row.get("run_id") or ""),
        ),
    )
    unique: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()
    for capture in captures:
        run_id = str(capture.get("run_id") or "")
        if not run_id or run_id in seen_run_ids:
            continue
        seen_run_ids.add(run_id)
        unique.append(capture)
    return unique


def _finalization_view_blockers(
    *,
    snapshot: dict[str, Any],
    rows: list[dict[str, Any]],
    allowlist: dict[str, Any],
    source_batch: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    snapshot_identity_payload = dict(snapshot)
    snapshot_identity = str(snapshot_identity_payload.pop("snapshot_manifest_id", ""))
    if canonical_json_sha256(snapshot_identity_payload) != snapshot_identity:
        blockers.append("bounded_view_snapshot_manifest_id_mismatch")
    allowlist_identity_payload = dict(allowlist)
    allowlist_identity = str(allowlist_identity_payload.pop("allowlist_id", ""))
    if canonical_json_sha256(allowlist_identity_payload) != allowlist_identity:
        blockers.append("bounded_view_allowlist_id_mismatch")
    if len(rows) != MAXIMUM_CAPTURE_ATTEMPT_COUNT:
        blockers.append("bounded_view_capture_attempt_count_mismatch")
    if int(snapshot.get("bounded_capture_attempt_count") or 0) != len(rows):
        blockers.append("bounded_view_snapshot_capture_count_mismatch")
    if int(snapshot.get("maximum_source_capture_attempt_ordinal") or 0) != (
        MAXIMUM_CAPTURE_ATTEMPT_COUNT
    ):
        blockers.append("bounded_view_snapshot_attempt_limit_mismatch")
    if snapshot.get("attempts_after_150_included") is not False:
        blockers.append("bounded_view_snapshot_includes_late_attempts")
    expected_allowlist_schema = f"{SCHEMA_PREFIX}-finalization-allowlist-v1"
    if allowlist.get("schema_version") != expected_allowlist_schema:
        blockers.append("bounded_view_allowlist_schema_mismatch")
    if int(allowlist.get("allowed_capture_attempt_count") or 0) != len(rows):
        blockers.append("bounded_view_allowlist_capture_count_mismatch")
    if int(allowlist.get("maximum_source_capture_attempt_ordinal") or 0) != (
        MAXIMUM_CAPTURE_ATTEMPT_COUNT
    ):
        blockers.append("bounded_view_allowlist_attempt_limit_mismatch")
    if allowlist.get("attempts_after_150_allowed") is not False:
        blockers.append("bounded_view_allowlist_permits_late_attempts")
    if allowlist.get("read_only_finalization_only") is not True:
        blockers.append("bounded_view_allowlist_not_read_only_finalization")
    if allowlist.get("labels_or_outcomes_opened_for_allowlist_creation") is not False:
        blockers.append("bounded_view_allowlist_creation_not_outcome_blind")
    row_run_ids = [str(row.get("run_id") or "") for row in rows]
    row_run_dirs = [str(row.get("run_dir") or "") for row in rows]
    row_hashes = [canonical_json_sha256(row) for row in rows]
    if "" in row_run_ids or len(set(row_run_ids)) != len(row_run_ids):
        blockers.append("bounded_view_capture_identity_not_unique")
    if [int(row.get("round_index") or 0) for row in rows] != list(
        range(1, MAXIMUM_CAPTURE_ATTEMPT_COUNT + 1)
    ):
        blockers.append("bounded_view_round_index_sequence_invalid")
    if allowlist.get("run_ids") != row_run_ids:
        blockers.append("bounded_view_allowlist_run_ids_mismatch")
    if allowlist.get("run_dirs") != row_run_dirs:
        blockers.append("bounded_view_allowlist_run_dirs_mismatch")
    if allowlist.get("capture_row_sha256s") != row_hashes:
        blockers.append("bounded_view_allowlist_capture_hashes_mismatch")
    if str(source_batch.get("collection_stop_reason") or "") not in (
        TERMINAL_COLLECTION_STOP_REASONS
    ):
        blockers.append("bounded_view_source_collection_not_terminal")
    source_captures = [dict(row) for row in source_batch.get("captures") or []]
    if len(source_captures) != int(source_batch.get("capture_count") or 0):
        blockers.append("bounded_view_source_capture_count_mismatch")
    ordered_source_rows = _ordered_unique_captures(source_batch)
    if len(ordered_source_rows) != len(source_captures):
        blockers.append("bounded_view_source_capture_identity_not_unique")
    if [canonical_json_sha256(row) for row in ordered_source_rows[:150]] != row_hashes:
        blockers.append("bounded_view_snapshot_rows_do_not_match_source_first_150")
    late_source_run_ids = {
        str(row.get("run_id") or "") for row in ordered_source_rows[150:]
    }
    if late_source_run_ids.intersection(row_run_ids):
        blockers.append("bounded_view_late_source_attempt_in_allowlist")
    for row in rows:
        run_dir = Path(str(row.get("run_dir") or ""))
        if not run_dir.is_dir():
            blockers.append("bounded_view_source_run_dir_missing")
            continue
        if not (run_dir / "pending_round_capture_manifest.json").is_file():
            blockers.append("bounded_view_pending_capture_manifest_missing")
    if _find_fields({"rows": rows}, FORBIDDEN_SOURCE_FIELDS):
        blockers.append("bounded_view_forbidden_outcome_field_present")
    if not _safety_blocked(snapshot, allowlist, source_batch):
        blockers.append("bounded_view_safety_contract_failed")
    return sorted(set(blockers))


def _bounded_finalized_batch(
    *,
    snapshot_rows: list[dict[str, Any]],
    finalized_batch: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    if len(snapshot_rows) != MAXIMUM_CAPTURE_ATTEMPT_COUNT:
        blockers.append("snapshot_capture_attempt_count_mismatch")
    snapshot_by_run_id = {
        str(row.get("run_id") or ""): row for row in snapshot_rows
    }
    if len(snapshot_by_run_id) != len(snapshot_rows) or "" in snapshot_by_run_id:
        blockers.append("snapshot_capture_identity_not_unique")
    finalized_captures = {
        str(row.get("run_id") or ""): dict(row)
        for row in finalized_batch.get("captures") or []
    }
    for run_id, row in snapshot_by_run_id.items():
        if run_id not in finalized_captures:
            blockers.append("snapshot_capture_missing_from_finalized_batch")
        elif canonical_json_sha256(row) != canonical_json_sha256(
            finalized_captures[run_id]
        ):
            blockers.append("snapshot_capture_mutated_during_finalization")
    bounded_finalizations = [
        dict(row)
        for row in finalized_batch.get("finalizations") or []
        if str(row.get("run_id") or "") in snapshot_by_run_id
    ]
    bounded = {
        **{
            key: value
            for key, value in finalized_batch.items()
            if key not in {"captures", "finalizations"}
        },
        "batch_id": str(finalized_batch.get("batch_id") or "")
        + "-hybrid-first150",
        "capture_count": len(snapshot_rows),
        "captures": snapshot_rows,
        "finalizations": bounded_finalizations,
        "finalization_attempt_count": len(bounded_finalizations),
        "hybrid_source_capture_attempt_limit": MAXIMUM_CAPTURE_ATTEMPT_COUNT,
        "source_attempts_after_150_included": False,
        "labels_or_outcomes_opened_for_role_assignment": False,
    }
    return bounded, sorted(set(blockers))


def _safety_blocked(*payloads: dict[str, Any]) -> bool:
    expected = {
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
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
    }
    for payload in payloads:
        if payload.get("paper_only") is not True:
            return False
        if payload.get("capital_at_risk") is not False:
            return False
        if any(
            payload.get(key) != value
            for key, value in expected.items()
            if key in payload
        ):
            return False
    return True


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _binding_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Hybrid Candidate-Agnostic Source Binding",
        "",
        f"- status: `{report['status']}`",
        f"- source binding ready: `{str(report['source_binding_ready']).lower()}`",
        f"- source collection terminal: `{str(report['source_collection_terminal']).lower()}`",
        f"- current capture count: `{report['source_current_capture_count']}`",
        f"- frozen attempt boundary: `1-{report['maximum_source_capture_attempt_ordinal']}`",
        "- duplicate collector started: `false`",
        "- labels/outcomes opened: `false`",
        "- #189 OOF/development/PnL used: `false`",
        "",
        "## Blocking Reasons",
    ]
    reasons = report["blocking_reason_codes"]
    lines.extend(f"- `{reason}`" for reason in reasons or ["none"])
    return "\n".join(lines) + "\n"
