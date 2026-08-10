"""Byte-exact, outcome-blind archive snapshots for promotion capture storage."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
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

SCHEMA_VERSION = "bigan-btc-15m-residual-promotion-capture-archive-v1"
IMPLEMENTATION_REPOSITORY_PATH = (
    "src/bigan/v8/polymarket/residual_promotion_capture_archive.py"
)
FORBIDDEN_CAPTURE_FILE_TOKENS = ("outcome", "settlement", "realized_pnl", "unit_pnl")
EMPTY_RESOLUTION_STREAM_NAME = "raw_polymarket_resolutions.jsonl"
ROOT_STATE_FILES = (
    "collection_progress.json",
    "collection_resume_record_v3.json",
    "collection_start_record.json",
    "outcome_blind_attempts.jsonl",
)


def mirror_completed_capture_snapshot(
    *,
    service_root: Path | str,
    archive_root: Path | str,
    created_at: str,
) -> dict[str, Any]:
    """Copy only ledger-closed attempts and freeze a byte-level archive manifest."""

    source = Path(service_root).resolve()
    archive = Path(archive_root).resolve()
    _validate_roots(source=source, archive=archive)
    if archive.exists() and _appledouble_paths(archive):
        raise ValueError("capture archive contains pre-existing AppleDouble metadata")
    ledger_path = source / "outcome_blind_attempts.jsonl"
    progress_path = source / "collection_progress.json"
    attempts = _load_jsonl(ledger_path)
    verify_attempt_chain(attempts)
    for attempt in attempts:
        assert_outcome_blind(attempt)
    progress = _load_json(progress_path)
    assert_outcome_blind(progress)
    _validate_progress(progress, attempts=attempts)
    ledger_sha = sha256_file(ledger_path)
    snapshot = archive / "state_snapshots" / ledger_sha
    manifest_path = snapshot / "capture_archive_manifest.json"
    sidecar_path = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if manifest_path.exists() or sidecar_path.exists():
        return verify_capture_archive_snapshot(
            manifest_path=manifest_path,
            expected_source_root=source,
        )

    archive.mkdir(parents=True, exist_ok=True)
    capture_descriptors: list[dict[str, Any]] = []
    for attempt in attempts:
        attempt_id = _attempt_id(attempt)
        capture_source = _strict_child(source / "captures" / attempt_id, source)
        if not capture_source.is_dir():
            if attempt.get("capture_manifest_sha256") is not None:
                raise ValueError("ledger-bound capture directory is missing")
            continue
        files = _regular_file_inventory(capture_source, base=source)
        _validate_capture_binding(attempt, files=files)
        for descriptor in files:
            relative = str(descriptor["path"])
            destination = _strict_child(archive / relative, archive)
            source_path = _strict_child(source / relative, source)
            _copy_new_or_verify(source_path, destination, descriptor=descriptor)
            capture_descriptors.append(descriptor)

    capture_descriptors.sort(key=lambda row: str(row["path"]))
    root_descriptors = []
    with tempfile.TemporaryDirectory(prefix="snapshot-", dir=archive) as temporary:
        staging = Path(temporary)
        for name in ROOT_STATE_FILES:
            source_path = source / name
            if not source_path.is_file():
                raise ValueError(f"required collection state file is missing: {name}")
            if source_path.suffix == ".json":
                assert_outcome_blind(_load_json(source_path))
            destination = staging / name
            shutil.copyfile(source_path, destination)
            descriptor = _file_descriptor(destination, base=staging)
            descriptor["path"] = f"state_snapshots/{ledger_sha}/{name}"
            root_descriptors.append(descriptor)
        identity = {
            "attempt_ledger_sha256": ledger_sha,
            "root_state_files": root_descriptors,
            "capture_files": capture_descriptors,
        }
        capture_attempt_ids = sorted(
            {
                Path(str(row["path"])).parts[1]
                for row in capture_descriptors
                if Path(str(row["path"])).parts[:1] == ("captures",)
            }
        )
        report = {
            "schema_version": SCHEMA_VERSION,
            "lineage_id": LINEAGE_ID,
            "candidate_id": CANDIDATE_ID,
            "created_at": created_at,
            "source_service_root_name": source.name,
            "attempt_count": len(attempts),
            "maximum_attempts": MAXIMUM_ATTEMPTS,
            "quality_valid_market_count": int(
                progress["quality_valid_market_count"]
            ),
            "target_quality_valid_market_count": TARGET_MARKETS,
            "capture_attempt_count": len(capture_attempt_ids),
            "capture_attempt_ids": capture_attempt_ids,
            "archive_tree_sha256": canonical_json_sha256(identity),
            **identity,
            "implementation": {
                "path": IMPLEMENTATION_REPOSITORY_PATH,
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "archive_complete": True,
            "source_capture_mutated": False,
            "source_capture_deleted": False,
            "collector_paused": False,
            "collector_restarted": False,
            "collection_population_changed": False,
            "storage_archive_only": True,
            "archive_influences_collection": False,
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
        staged_manifest = staging / manifest_path.name
        staged_manifest.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staged_sidecar = staging / sidecar_path.name
        staged_sidecar.write_text(sha256_file(staged_manifest) + "\n", encoding="utf-8")
        snapshot.mkdir(parents=True, exist_ok=False)
        staged_names = [
            *ROOT_STATE_FILES,
            staged_manifest.name,
            staged_sidecar.name,
        ]
        for name in staged_names:
            (staging / name).replace(snapshot / name)
    _remove_generated_appledouble(archive)
    return verify_capture_archive_snapshot(
        manifest_path=manifest_path,
        expected_source_root=source,
    )


def verify_capture_archive_snapshot(
    *,
    manifest_path: Path | str,
    expected_source_root: Path | str | None = None,
) -> dict[str, Any]:
    """Verify a snapshot, its exact capture file sets, and optional source bytes."""

    manifest_file = Path(manifest_path).resolve()
    snapshot = manifest_file.parent
    archive = snapshot.parent.parent
    if _appledouble_paths(archive):
        raise ValueError("capture archive contains AppleDouble metadata")
    if not manifest_file.is_file() or manifest_file.name != "capture_archive_manifest.json":
        raise ValueError("capture archive manifest is unavailable")
    sidecar = manifest_file.with_suffix(manifest_file.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != sha256_file(
        manifest_file
    ):
        raise ValueError("capture archive manifest sidecar mismatch")
    manifest = _load_json(manifest_file)
    assert_outcome_blind(manifest)
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("lineage_id") != LINEAGE_ID
        or manifest.get("candidate_id") != CANDIDATE_ID
        or int(manifest.get("maximum_attempts", -1)) != MAXIMUM_ATTEMPTS
        or int(manifest.get("target_quality_valid_market_count", -1))
        != TARGET_MARKETS
        or manifest.get("archive_complete") is not True
        or manifest.get("storage_archive_only") is not True
        or manifest.get("archive_influences_collection") is not False
        or any(
            manifest.get(key) is not False
            for key in (
                "source_capture_mutated",
                "source_capture_deleted",
                "collector_paused",
                "collector_restarted",
                "collection_population_changed",
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
        )
        or dict(manifest.get("safety") or {}) != SAFETY
    ):
        raise ValueError("capture archive governance metadata mismatch")
    implementation = dict(manifest.get("implementation") or {})
    if (
        implementation.get("path") != IMPLEMENTATION_REPOSITORY_PATH
        or implementation.get("sha256") != sha256_file(Path(__file__).resolve())
    ):
        raise ValueError("capture archive implementation binding mismatch")
    root_descriptors = list(manifest.get("root_state_files") or [])
    capture_descriptors = list(manifest.get("capture_files") or [])
    ledger_sha = str(manifest.get("attempt_ledger_sha256") or "")
    if (
        len(ledger_sha) != 64
        or any(character not in "0123456789abcdef" for character in ledger_sha)
        or snapshot.parent.name != "state_snapshots"
        or snapshot.name != ledger_sha
    ):
        raise ValueError("capture archive snapshot path does not match ledger identity")
    root_paths = [str(row.get("path") or "") for row in root_descriptors]
    expected_root_paths = [
        f"state_snapshots/{ledger_sha}/{name}" for name in ROOT_STATE_FILES
    ]
    if sorted(root_paths) != sorted(expected_root_paths) or len(set(root_paths)) != len(
        root_paths
    ):
        raise ValueError("capture archive root-state descriptor set mismatch")
    capture_paths = [str(row.get("path") or "") for row in capture_descriptors]
    if len(set(capture_paths)) != len(capture_paths):
        raise ValueError("capture archive contains duplicate capture descriptors")
    identity = {
        "attempt_ledger_sha256": manifest.get("attempt_ledger_sha256"),
        "root_state_files": root_descriptors,
        "capture_files": capture_descriptors,
    }
    if canonical_json_sha256(identity) != manifest.get("archive_tree_sha256"):
        raise ValueError("capture archive identity mismatch")
    expected_snapshot_files = {
        Path(str(row["path"])).name for row in root_descriptors
    } | {manifest_file.name, sidecar.name}
    actual_snapshot_entries = {path.name for path in snapshot.iterdir()}
    if actual_snapshot_entries != expected_snapshot_files:
        raise ValueError("capture archive snapshot has missing or extra files")
    for descriptor in root_descriptors:
        path = _strict_child(archive / str(descriptor["path"]), archive)
        _verify_file(path, descriptor=descriptor)
        if path.suffix == ".json":
            assert_outcome_blind(_load_json(path))
    by_attempt: dict[str, list[dict[str, Any]]] = {}
    for descriptor in capture_descriptors:
        relative = Path(str(descriptor["path"]))
        if len(relative.parts) < 3 or relative.parts[0] != "captures":
            raise ValueError("capture archive descriptor escaped capture namespace")
        _validate_safe_capture_descriptor(descriptor)
        by_attempt.setdefault(relative.parts[1], []).append(descriptor)
        _verify_file(_strict_child(archive / relative, archive), descriptor=descriptor)
    if sorted(by_attempt) != list(manifest.get("capture_attempt_ids") or []):
        raise ValueError("capture archive attempt identity mismatch")
    for attempt_id, descriptors in by_attempt.items():
        directory = _strict_child(archive / "captures" / attempt_id, archive)
        expected = {str(row["path"]) for row in descriptors}
        actual = {
            path.relative_to(archive).as_posix()
            for path in directory.rglob("*")
            if path.is_file()
        }
        if actual != expected:
            raise ValueError("capture archive attempt has missing or extra files")
        expected_directories = {
            parent.as_posix()
            for descriptor in descriptors
            for parent in _relative_parent_directories(
                Path(str(descriptor["path"])), base=directory.relative_to(archive)
            )
        }
        actual_directories = {
            path.relative_to(archive).as_posix()
            for path in directory.rglob("*")
            if path.is_dir()
        }
        if actual_directories != expected_directories:
            raise ValueError("capture archive attempt has missing or extra directories")
    ledger_descriptor = next(
        (
            row
            for row in root_descriptors
            if Path(str(row["path"])).name == "outcome_blind_attempts.jsonl"
        ),
        None,
    )
    if ledger_descriptor is None:
        raise ValueError("capture archive ledger descriptor is missing")
    ledger = _strict_child(archive / str(ledger_descriptor["path"]), archive)
    attempts = _load_jsonl(ledger)
    verify_attempt_chain(attempts)
    for attempt in attempts:
        assert_outcome_blind(attempt)
    if len(attempts) != int(manifest["attempt_count"]):
        raise ValueError("capture archive attempt count mismatch")
    progress_descriptor = next(
        (
            row
            for row in root_descriptors
            if Path(str(row["path"])).name == "collection_progress.json"
        ),
        None,
    )
    if progress_descriptor is None:
        raise ValueError("capture archive progress descriptor is missing")
    progress = _load_json(
        _strict_child(archive / str(progress_descriptor["path"]), archive)
    )
    assert_outcome_blind(progress)
    _validate_progress(progress, attempts=attempts)
    if int(manifest["quality_valid_market_count"]) != int(
        progress["quality_valid_market_count"]
    ):
        raise ValueError("capture archive quality-valid count mismatch")
    attempts_by_id = {_attempt_id(attempt): attempt for attempt in attempts}
    if len(attempts_by_id) != len(attempts):
        raise ValueError("capture archive ledger contains duplicate attempt ids")
    for attempt_id, descriptors in by_attempt.items():
        if attempt_id not in attempts_by_id:
            raise ValueError("capture archive attempt is absent from the ledger")
        _validate_capture_binding(attempts_by_id[attempt_id], files=descriptors)
    required_capture_ids = sorted(
        _attempt_id(attempt)
        for attempt in attempts
        if attempt.get("capture_manifest_sha256") is not None
        or attempt.get("capture_report_sha256") is not None
    )
    if sorted(by_attempt) != required_capture_ids:
        raise ValueError("capture archive does not reconcile to ledger-bound captures")
    if int(manifest.get("capture_attempt_count", -1)) != len(by_attempt):
        raise ValueError("capture archive capture-attempt count mismatch")
    source_verified = False
    if expected_source_root is not None:
        source = Path(expected_source_root).resolve()
        if source.name != manifest.get("source_service_root_name"):
            raise ValueError("capture archive source root identity mismatch")
        for descriptor in [*root_descriptors, *capture_descriptors]:
            relative = Path(str(descriptor["path"]))
            if relative.parts[:2] == ("state_snapshots", str(manifest["attempt_ledger_sha256"])):
                relative = Path(*relative.parts[2:])
            source_path = _strict_child(source / relative, source)
            _verify_file(source_path, descriptor=descriptor)
        source_verified = True
    return {
        **manifest,
        "manifest_sha256": sha256_file(manifest_file),
        "verification_passed": True,
        "source_bytes_verified": source_verified,
        "verified_file_count": len(root_descriptors) + len(capture_descriptors),
    }


def _validate_roots(*, source: Path, archive: Path) -> None:
    if not source.is_dir():
        raise ValueError("collection service root is unavailable")
    if source == archive or source.is_relative_to(archive) or archive.is_relative_to(source):
        raise ValueError("archive and collection roots must be disjoint")


def _validate_progress(
    progress: Mapping[str, Any], *, attempts: Sequence[Mapping[str, Any]]
) -> None:
    unique_valid_markets = {
        str(attempt["market_id"])
        for attempt in attempts
        if bool(dict(attempt.get("quality") or {}).get("quality_valid"))
        and isinstance(attempt.get("market_id"), str)
    }
    if (
        progress.get("lineage_id") != LINEAGE_ID
        or progress.get("candidate_id") != CANDIDATE_ID
        or int(progress.get("attempts_consumed", -1)) != len(attempts)
        or int(progress.get("maximum_attempts", -1)) != MAXIMUM_ATTEMPTS
        or int(progress.get("target_quality_valid_market_count", -1))
        != TARGET_MARKETS
        or int(progress.get("quality_valid_market_count", -1))
        != len(unique_valid_markets)
        or progress.get("hash_chain_status") != "valid"
        or progress.get("fresh_outcomes_opened") is not False
        or progress.get("interim_pnl_evaluated") is not False
        or dict(progress.get("safety") or {}) != SAFETY
    ):
        raise ValueError("collection progress does not reconcile to the attempt ledger")


def _validate_capture_binding(
    attempt: Mapping[str, Any], *, files: Sequence[Mapping[str, Any]]
) -> None:
    if (
        attempt.get("capture_manifest_sha256") is None
        and attempt.get("capture_report_sha256") is None
        and files
    ):
        raise ValueError("unbound failed attempt unexpectedly contains capture files")
    hashes = {Path(str(row["path"])).name: row["sha256"] for row in files}
    if attempt.get("capture_manifest_sha256") != hashes.get(
        "pending_round_capture_manifest.json"
    ):
        raise ValueError("capture manifest SHA does not match attempt ledger")
    if attempt.get("capture_report_sha256") != hashes.get(
        "pending_round_capture_report.json"
    ):
        raise ValueError("capture report SHA does not match attempt ledger")


def _attempt_id(attempt: Mapping[str, Any]) -> str:
    value = str(attempt.get("attempt_id") or "")
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError("attempt id is not a safe directory name")
    return value


def _regular_file_inventory(directory: Path, *, base: Path) -> list[dict[str, Any]]:
    output = []
    for path in sorted(directory.rglob("*")):
        if path.name.startswith("._"):
            raise ValueError("source capture contains AppleDouble metadata")
        if path.is_symlink():
            raise ValueError("capture archive refuses symbolic links")
        if path.is_file():
            lowered = path.name.lower()
            if any(token in lowered for token in FORBIDDEN_CAPTURE_FILE_TOKENS):
                raise ValueError("source capture contains an outcome-bearing filename")
            if path.name == EMPTY_RESOLUTION_STREAM_NAME and path.stat().st_size != 0:
                raise ValueError("source capture has a non-empty resolution stream")
            output.append(_file_descriptor(path, base=base))
    return output


def _validate_safe_capture_descriptor(descriptor: Mapping[str, Any]) -> None:
    path = Path(str(descriptor.get("path") or ""))
    lowered = path.name.lower()
    if any(token in lowered for token in FORBIDDEN_CAPTURE_FILE_TOKENS):
        raise ValueError("capture archive contains an outcome-bearing filename")
    if path.name == EMPTY_RESOLUTION_STREAM_NAME and int(
        descriptor.get("size_bytes", -1)
    ) != 0:
        raise ValueError("capture archive contains a non-empty resolution stream")


def _relative_parent_directories(path: Path, *, base: Path) -> list[Path]:
    output = []
    parent = path.parent
    while parent != base:
        if not parent.is_relative_to(base):
            raise ValueError("capture archive descriptor parent escaped attempt root")
        output.append(parent)
        parent = parent.parent
    return output


def _file_descriptor(path: Path, *, base: Path) -> dict[str, Any]:
    resolved = _strict_child(path, base)
    return {
        "path": resolved.relative_to(base).as_posix(),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _copy_new_or_verify(
    source: Path, destination: Path, *, descriptor: Mapping[str, Any]
) -> None:
    if destination.exists():
        _verify_file(destination, descriptor=descriptor)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".archive-tmp")
    if temporary.exists():
        raise FileExistsError("stale capture archive temporary file exists")
    with source.open("rb") as input_stream, temporary.open("xb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    _verify_file(temporary, descriptor=descriptor)
    temporary.replace(destination)


def _verify_file(path: Path, *, descriptor: Mapping[str, Any]) -> None:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(descriptor["size_bytes"])
        or sha256_file(path) != descriptor["sha256"]
    ):
        raise ValueError(f"capture archive file mismatch: {path}")


def _strict_child(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("capture archive path escaped its root")
    return resolved


def _appledouble_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("._*") if path.name.startswith("._"))


def _remove_generated_appledouble(root: Path) -> None:
    for path in sorted(_appledouble_paths(root), reverse=True):
        if not path.is_file() and not path.is_symlink():
            raise ValueError("capture archive generated non-file AppleDouble metadata")
        path.unlink()
    if _appledouble_paths(root):
        raise ValueError("capture archive AppleDouble cleanup failed")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


__all__ = [
    "mirror_completed_capture_snapshot",
    "verify_capture_archive_snapshot",
]
