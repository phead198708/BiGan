"""Byte-verified, outcome-blind service-root migration for promotion v1."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.residual_promotion_collection import (
    assert_outcome_blind,
    verify_attempt_chain,
)
from bigan.v8.polymarket.residual_promotion_v1 import (
    CANDIDATE_ID,
    LINEAGE_ID,
    MAXIMUM_ATTEMPTS,
    TARGET_MARKETS,
)

SCHEMA_VERSION = "bigan-btc-15m-residual-promotion-service-root-migration-v1"
IMPLEMENTATION_REPOSITORY_PATH = (
    "src/bigan/v8/polymarket/residual_promotion_service_root_migration.py"
)
ROOT_STATE_FILES = {
    "collection_progress.json",
    "collection_resume_record_v3.json",
    "collection_start_record.json",
    "outcome_blind_attempts.jsonl",
}
IGNORED_SOURCE_METADATA = {".DS_Store"}
FORBIDDEN_CAPTURE_FILE_TOKENS = (
    "outcome",
    "settlement",
    "realized_pnl",
    "unit_pnl",
)
EMPTY_RESOLUTION_STREAM_NAME = "raw_polymarket_resolutions.jsonl"


def migrate_service_root(
    *,
    source_root: Path | str,
    destination_root: Path | str,
    report_path: Path | str,
    created_at: str,
) -> dict[str, Any]:
    """Copy one closed outcome-blind service-root snapshot and verify every byte."""

    source = Path(source_root).resolve()
    destination = Path(destination_root).resolve()
    report_file = Path(report_path).resolve()
    _validate_disjoint_roots(source=source, destination=destination)
    source_snapshot = snapshot_service_root(source)
    destination.mkdir(parents=True, exist_ok=True)
    _remove_generated_appledouble(destination)
    _copy_snapshot(
        source=source,
        destination=destination,
        snapshot=source_snapshot,
    )
    _remove_generated_appledouble(destination)
    destination_snapshot = snapshot_service_root(destination)
    if destination_snapshot != source_snapshot:
        raise ValueError("migrated service-root snapshot does not match source bytes")
    _verify_runtime_write_metadata_safety(destination)
    progress = _load_json(destination / "collection_progress.json")
    report = {
        "schema_version": SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at": created_at,
        "source_root": str(source),
        "destination_root": str(destination),
        "attempt_count": int(progress["attempts_consumed"]),
        "quality_valid_market_count": int(progress["quality_valid_market_count"]),
        "maximum_attempts": MAXIMUM_ATTEMPTS,
        "target_quality_valid_market_count": TARGET_MARKETS,
        "candidate_bundle_sha256": progress["candidate_bundle_sha256"],
        "authorization_sha256": progress["authorization_sha256"],
        "collector_protocol_sha256": progress["collector_protocol_sha256"],
        "source_snapshot": source_snapshot,
        "destination_snapshot": destination_snapshot,
        "implementation": {
            "path": IMPLEMENTATION_REPOSITORY_PATH,
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "migration_complete": True,
        "source_bytes_verified": True,
        "destination_bytes_verified": True,
        "runtime_write_metadata_safe": True,
        "source_capture_mutated": False,
        "source_capture_deleted": False,
        "collection_population_changed": False,
        "collector_decisions_changed": False,
        "outcomes_accessed": False,
        "settlement_accessed": False,
        "pnl_accessed": False,
        "fresh_outcomes_opened": False,
        "interim_pnl_evaluated": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_candidate_allowed": False,
        "v8_execution_handoff_allowed": False,
        "live_trading_allowed": False,
        "wallet_signing_allowed": False,
        "polymarket_write_allowed": False,
        "capital_at_risk": False,
        "safety": dict(SAFETY),
    }
    assert_outcome_blind(report)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    _write_new_json(report_file, report)
    _write_new_text(
        report_file.with_suffix(report_file.suffix + ".sha256"),
        sha256_file(report_file) + "\n",
    )
    return verify_service_root_migration(report_path=report_file)


def verify_service_root_migration(
    *, report_path: Path | str, require_source_match: bool = True
) -> dict[str, Any]:
    """Verify a frozen migration report and exact destination/source snapshots."""

    report_file = Path(report_path).resolve()
    sidecar = report_file.with_suffix(report_file.suffix + ".sha256")
    if (
        not report_file.is_file()
        or not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").strip() != sha256_file(report_file)
    ):
        raise ValueError("service-root migration report hash mismatch")
    report = _load_json(report_file)
    assert_outcome_blind(report)
    false_fields = (
        "source_capture_mutated",
        "source_capture_deleted",
        "collection_population_changed",
        "collector_decisions_changed",
        "outcomes_accessed",
        "settlement_accessed",
        "pnl_accessed",
        "fresh_outcomes_opened",
        "interim_pnl_evaluated",
        "source_model_candidate_eligible",
        "freeze_ready",
        "promotion_evidence_eligible",
        "paper_candidate_allowed",
        "v8_execution_handoff_allowed",
        "live_trading_allowed",
        "wallet_signing_allowed",
        "polymarket_write_allowed",
        "capital_at_risk",
    )
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("lineage_id") != LINEAGE_ID
        or report.get("candidate_id") != CANDIDATE_ID
        or report.get("migration_complete") is not True
        or report.get("source_bytes_verified") is not True
        or report.get("destination_bytes_verified") is not True
        or report.get("runtime_write_metadata_safe") is not True
        or any(report.get(field) is not False for field in false_fields)
        or dict(report.get("safety") or {}) != SAFETY
    ):
        raise ValueError("service-root migration governance mismatch")
    implementation = dict(report.get("implementation") or {})
    if (
        implementation.get("path") != IMPLEMENTATION_REPOSITORY_PATH
        or implementation.get("sha256") != sha256_file(Path(__file__).resolve())
    ):
        raise ValueError("service-root migration implementation drift")
    destination = Path(str(report["destination_root"])).resolve()
    destination_snapshot = snapshot_service_root(destination)
    if destination_snapshot != report.get("destination_snapshot"):
        raise ValueError("service-root migration destination drift")
    if report.get("source_snapshot") != report.get("destination_snapshot"):
        raise ValueError("service-root migration snapshot identity mismatch")
    if require_source_match:
        source = Path(str(report["source_root"])).resolve()
        if snapshot_service_root(source) != report.get("source_snapshot"):
            raise ValueError("service-root migration source drift")
    progress = _load_json(destination / "collection_progress.json")
    if not (
        int(progress.get("attempts_consumed", -1)) == int(report["attempt_count"])
        and int(progress.get("quality_valid_market_count", -1))
        == int(report["quality_valid_market_count"])
        and progress.get("candidate_bundle_sha256")
        == report.get("candidate_bundle_sha256")
        and progress.get("authorization_sha256")
        == report.get("authorization_sha256")
        and progress.get("collector_protocol_sha256")
        == report.get("collector_protocol_sha256")
    ):
        raise ValueError("service-root migration frozen binding mismatch")
    return {
        **report,
        "report_sha256": sha256_file(report_file),
        "verification_passed": True,
    }


def snapshot_service_root(root: Path | str) -> dict[str, Any]:
    """Return a deterministic byte-and-directory identity after safety validation."""

    service_root = Path(root).resolve()
    if not service_root.is_dir():
        raise ValueError("collection service root is unavailable")
    root_files = {
        path.name
        for path in service_root.iterdir()
        if path.is_file() and path.name not in IGNORED_SOURCE_METADATA
    }
    root_directories = {path.name for path in service_root.iterdir() if path.is_dir()}
    if root_files != ROOT_STATE_FILES or root_directories != {"captures"}:
        raise ValueError("collection service-root top-level set mismatch")
    attempts = _load_jsonl(service_root / "outcome_blind_attempts.jsonl")
    verify_attempt_chain(attempts)
    for attempt in attempts:
        assert_outcome_blind(attempt)
    progress = _load_json(service_root / "collection_progress.json")
    assert_outcome_blind(progress)
    valid_market_ids = {
        str(attempt["market_id"])
        for attempt in attempts
        if bool(dict(attempt.get("quality") or {}).get("quality_valid"))
        and isinstance(attempt.get("market_id"), str)
    }
    if not (
        progress.get("lineage_id") == LINEAGE_ID
        and progress.get("candidate_id") == CANDIDATE_ID
        and int(progress.get("attempts_consumed", -1)) == len(attempts)
        and int(progress.get("quality_valid_market_count", -1))
        == len(valid_market_ids)
        and int(progress.get("maximum_attempts", -1)) == MAXIMUM_ATTEMPTS
        and int(progress.get("target_quality_valid_market_count", -1))
        == TARGET_MARKETS
        and progress.get("hash_chain_status") == "valid"
        and progress.get("fresh_outcomes_opened") is False
        and progress.get("interim_pnl_evaluated") is False
        and dict(progress.get("safety") or {}) == SAFETY
    ):
        raise ValueError("collection service-root progress mismatch")
    closed_ids = {str(attempt["attempt_id"]) for attempt in attempts}
    for directory in (service_root / "captures").iterdir():
        if directory.name in IGNORED_SOURCE_METADATA:
            continue
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError("capture root contains a non-directory entry")
        if directory.name not in closed_ids:
            unledgered_entries = list(directory.rglob("*"))
            if any(path.is_file() or path.is_symlink() for path in unledgered_entries):
                raise ValueError("unledgered capture directory contains bytes")
    files: list[dict[str, Any]] = []
    directories: list[str] = []
    for path in sorted(service_root.rglob("*")):
        relative = path.relative_to(service_root)
        if path.name in IGNORED_SOURCE_METADATA:
            continue
        if path.name.startswith("._"):
            raise ValueError("service root contains AppleDouble metadata")
        if path.is_symlink():
            raise ValueError("service-root migration refuses symbolic links")
        if path.is_dir():
            directories.append(relative.as_posix())
            continue
        if not path.is_file():
            raise ValueError("service-root migration encountered a special file")
        if relative.parts[:1] == ("captures",):
            lowered = path.name.lower()
            if any(token in lowered for token in FORBIDDEN_CAPTURE_FILE_TOKENS):
                raise ValueError("capture contains an outcome-bearing filename")
            if path.name == EMPTY_RESOLUTION_STREAM_NAME and path.stat().st_size != 0:
                raise ValueError("capture contains a nonempty resolution stream")
        files.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    identity = {"directories": directories, "files": files}
    return {**identity, "tree_sha256": canonical_json_sha256(identity)}


def _copy_snapshot(
    *, source: Path, destination: Path, snapshot: Mapping[str, Any]
) -> None:
    for relative in snapshot["directories"]:
        (destination / str(relative)).mkdir(parents=True, exist_ok=True)
    for descriptor in snapshot["files"]:
        relative = str(descriptor["path"])
        source_file = source / relative
        destination_file = destination / relative
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination_file.with_name(destination_file.name + ".migration-tmp")
        if destination_file.exists():
            if (
                not destination_file.is_file()
                or destination_file.is_symlink()
                or destination_file.stat().st_size != int(descriptor["size_bytes"])
                or sha256_file(destination_file) != descriptor["sha256"]
            ):
                raise ValueError("migration destination contains byte drift")
            continue
        if temporary.exists():
            raise ValueError("migration destination contains a stale temporary file")
        with source_file.open("rb") as input_stream, temporary.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        if (
            temporary.stat().st_size != int(descriptor["size_bytes"])
            or sha256_file(temporary) != descriptor["sha256"]
        ):
            raise ValueError("migration temporary file verification failed")
        temporary.replace(destination_file)


def _remove_generated_appledouble(root: Path) -> None:
    paths = sorted(root.rglob("._*"), reverse=True)
    for path in paths:
        if not path.is_file() and not path.is_symlink():
            raise ValueError("migration generated non-file AppleDouble metadata")
        path.unlink()
    if any(root.rglob("._*")):
        raise ValueError("migration AppleDouble cleanup failed")


def _verify_runtime_write_metadata_safety(root: Path) -> None:
    probe = root / ".bigan-runtime-write-probe"
    sidecar = root / "._.bigan-runtime-write-probe"
    if probe.exists() or sidecar.exists():
        raise ValueError("migration runtime write probe path already exists")
    probe.mkdir()
    (probe / "probe.json").write_text("{}\n", encoding="utf-8")
    appledouble = sorted(path for path in root.rglob("._*") if path.name.startswith("._"))
    shutil.rmtree(probe)
    for path in sorted(appledouble, reverse=True):
        if path.exists() or path.is_symlink():
            path.unlink()
    if appledouble:
        raise ValueError(
            "destination filesystem generates AppleDouble during runtime writes"
        )


def _validate_disjoint_roots(*, source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise ValueError("collection service root is unavailable")
    if (
        source == destination
        or source.is_relative_to(destination)
        or destination.is_relative_to(source)
    ):
        raise ValueError("source and destination service roots must be disjoint")


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_new_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_new_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not all(isinstance(value, dict) for value in values):
        raise ValueError("attempt ledger contains a non-object row")
    return values


__all__ = [
    "migrate_service_root",
    "snapshot_service_root",
    "verify_service_root_migration",
]
