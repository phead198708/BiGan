"""Outcome-blind capture archive tests for residual promotion v1."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.residual_promotion_capture_archive import (
    mirror_completed_capture_snapshot,
    verify_capture_archive_snapshot,
)
from bigan.v8.polymarket.residual_promotion_collection import canonical_attempt_hash
from bigan.v8.polymarket.residual_promotion_v1 import (
    CANDIDATE_ID,
    LINEAGE_ID,
    MAXIMUM_ATTEMPTS,
    TARGET_MARKETS,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _refreeze_archive_manifest(path: Path, payload: dict) -> None:
    identity = {
        "attempt_ledger_sha256": payload["attempt_ledger_sha256"],
        "root_state_files": payload["root_state_files"],
        "capture_files": payload["capture_files"],
    }
    payload["archive_tree_sha256"] = canonical_json_sha256(identity)
    _write_json(path, payload)
    path.with_suffix(path.suffix + ".sha256").write_text(sha256_file(path) + "\n")


def _fixture(root: Path, *, outcome_opened: bool = False) -> list[dict]:
    attempts = []
    previous = "0" * 64
    for index in range(1, 3):
        attempt_id = f"attempt-{index:04d}"
        run = root / "captures" / attempt_id
        run.mkdir(parents=True)
        manifest = run / "pending_round_capture_manifest.json"
        report = run / "pending_round_capture_report.json"
        raw = run / "raw/raw_polymarket_orderbooks.jsonl"
        raw.parent.mkdir()
        manifest.write_bytes(f'{{"run_id":"{attempt_id}"}}\n'.encode())
        report.write_bytes(b'{"resolution_provider_called":false}\n')
        raw.write_bytes(f'{{"attempt":{index}}}\n'.encode())
        attempt = {
            "attempt_index": index,
            "attempt_id": attempt_id,
            "market_id": f"market-{index}",
            "capture_manifest_sha256": sha256_file(manifest),
            "capture_report_sha256": sha256_file(report),
            "quality": {"quality_valid": True},
            "provider_health": {"provider_failed": False, "retry_used": False},
            "previous_attempt_hash": previous,
            "outcomes_accessed": outcome_opened,
            "settlement_accessed": False,
            "pnl_accessed": False,
            "fresh_outcomes_opened": outcome_opened,
            "interim_pnl_evaluated": False,
            "safety": dict(SAFETY),
        }
        attempt["attempt_hash"] = canonical_attempt_hash(attempt)
        previous = attempt["attempt_hash"]
        attempts.append(attempt)
    (root / "outcome_blind_attempts.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in attempts)
    )
    _write_json(
        root / "collection_progress.json",
        {
            "lineage_id": LINEAGE_ID,
            "candidate_id": CANDIDATE_ID,
            "attempts_consumed": len(attempts),
            "maximum_attempts": MAXIMUM_ATTEMPTS,
            "quality_valid_market_count": len(attempts),
            "target_quality_valid_market_count": TARGET_MARKETS,
            "hash_chain_status": "valid",
            "fresh_outcomes_opened": outcome_opened,
            "interim_pnl_evaluated": False,
            "outcomes_accessed": outcome_opened,
            "settlement_accessed": False,
            "pnl_accessed": False,
            "safety": dict(SAFETY),
        },
    )
    _write_json(root / "collection_resume_record_v3.json", {"safety": SAFETY})
    _write_json(root / "collection_start_record.json", {"safety": SAFETY})
    return attempts


def test_archive_mirrors_only_closed_attempts_and_verifies_exact_bytes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "service"
    archive = tmp_path / "archive"
    _fixture(source)
    active = source / "captures/attempt-0003"
    active.mkdir()
    (active / "active.tmp").write_bytes(b"not closed")
    before = {
        path.relative_to(source).as_posix(): sha256_file(path)
        for path in source.rglob("*")
        if path.is_file()
    }
    report = mirror_completed_capture_snapshot(
        service_root=source,
        archive_root=archive,
        created_at="2030-01-01T00:00:00+00:00",
    )
    after = {
        path.relative_to(source).as_posix(): sha256_file(path)
        for path in source.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert report["verification_passed"] is True
    assert report["source_bytes_verified"] is True
    assert report["attempt_count"] == 2
    assert report["capture_attempt_count"] == 2
    assert report["capture_attempt_ids"] == ["attempt-0001", "attempt-0002"]
    assert not (archive / "captures/attempt-0003").exists()
    assert report["source_capture_mutated"] is False
    assert report["source_capture_deleted"] is False
    assert report["storage_archive_only"] is True
    assert report["archive_influences_collection"] is False
    assert report["outcomes_accessed"] is False
    assert report["settlement_accessed"] is False
    assert report["pnl_accessed"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["freeze_ready"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["paper_candidate_allowed"] is False
    assert report["v8_execution_handoff_allowed"] is False
    assert report["safety"] == SAFETY


def test_archive_is_idempotent_for_same_ledger_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "service"
    archive = tmp_path / "archive"
    _fixture(source)
    first = mirror_completed_capture_snapshot(
        service_root=source,
        archive_root=archive,
        created_at="2030-01-01T00:00:00+00:00",
    )
    second = mirror_completed_capture_snapshot(
        service_root=source,
        archive_root=archive,
        created_at="2030-01-01T00:01:00+00:00",
    )
    assert second["manifest_sha256"] == first["manifest_sha256"]


def test_archive_fails_closed_on_outcome_opening(tmp_path: Path) -> None:
    source = tmp_path / "service"
    _fixture(source, outcome_opened=True)
    with pytest.raises(ValueError, match="outcome-blind safety field"):
        mirror_completed_capture_snapshot(
            service_root=source,
            archive_root=tmp_path / "archive",
            created_at="2030-01-01T00:00:00+00:00",
        )


def test_archive_fails_closed_on_ledger_capture_hash_drift(tmp_path: Path) -> None:
    source = tmp_path / "service"
    _fixture(source)
    (source / "captures/attempt-0002/pending_round_capture_report.json").write_bytes(
        b"drift\n"
    )
    with pytest.raises(ValueError, match="capture report SHA"):
        mirror_completed_capture_snapshot(
            service_root=source,
            archive_root=tmp_path / "archive",
            created_at="2030-01-01T00:00:00+00:00",
        )


def test_archive_fails_closed_on_extra_appledouble_file(tmp_path: Path) -> None:
    source = tmp_path / "service"
    archive = tmp_path / "archive"
    _fixture(source)
    report = mirror_completed_capture_snapshot(
        service_root=source,
        archive_root=archive,
        created_at="2030-01-01T00:00:00+00:00",
    )
    manifest = next(archive.glob("state_snapshots/*/capture_archive_manifest.json"))
    extra = archive / "captures/attempt-0001/._pending_round_capture_manifest.json"
    extra.write_bytes(b"appledouble")
    with pytest.raises(ValueError, match="AppleDouble"):
        verify_capture_archive_snapshot(
            manifest_path=manifest,
            expected_source_root=source,
        )
    assert report["source_capture_deleted"] is False


def test_archive_never_overwrites_existing_byte_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "service"
    archive = tmp_path / "archive"
    _fixture(source)
    destination = archive / "captures/attempt-0001/pending_round_capture_manifest.json"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"existing drift\n")
    with pytest.raises(ValueError, match="archive file mismatch"):
        mirror_completed_capture_snapshot(
            service_root=source,
            archive_root=archive,
            created_at="2030-01-01T00:00:00+00:00",
        )


def test_archive_fails_closed_on_progress_count_drift(tmp_path: Path) -> None:
    source = tmp_path / "service"
    _fixture(source)
    progress_path = source / "collection_progress.json"
    progress = json.loads(progress_path.read_text())
    progress["quality_valid_market_count"] = 1
    _write_json(progress_path, progress)
    with pytest.raises(ValueError, match="does not reconcile"):
        mirror_completed_capture_snapshot(
            service_root=source,
            archive_root=tmp_path / "archive",
            created_at="2030-01-01T00:00:00+00:00",
        )


def test_archive_verifier_fails_closed_on_archived_byte_drift(tmp_path: Path) -> None:
    source = tmp_path / "service"
    archive = tmp_path / "archive"
    _fixture(source)
    mirror_completed_capture_snapshot(
        service_root=source,
        archive_root=archive,
        created_at="2030-01-01T00:00:00+00:00",
    )
    manifest = next(archive.glob("state_snapshots/*/capture_archive_manifest.json"))
    archived = archive / "captures/attempt-0002/raw/raw_polymarket_orderbooks.jsonl"
    archived.write_bytes(b"archive drift\n")
    with pytest.raises(ValueError, match="archive file mismatch"):
        verify_capture_archive_snapshot(
            manifest_path=manifest,
            expected_source_root=source,
        )


def test_archive_rejects_nested_source_and_destination(tmp_path: Path) -> None:
    source = tmp_path / "service"
    _fixture(source)
    with pytest.raises(ValueError, match="must be disjoint"):
        mirror_completed_capture_snapshot(
            service_root=source,
            archive_root=source / "archive",
            created_at="2030-01-01T00:00:00+00:00",
        )


def test_archive_rejects_nonempty_resolution_without_archiving_it(tmp_path: Path) -> None:
    source = tmp_path / "service"
    archive = tmp_path / "archive"
    _fixture(source)
    resolution = source / "captures/attempt-0001/raw/raw_polymarket_resolutions.jsonl"
    resolution.write_bytes(b'{"winner":"UP"}\n')
    with pytest.raises(ValueError, match="non-empty resolution"):
        mirror_completed_capture_snapshot(
            service_root=source,
            archive_root=archive,
            created_at="2030-01-01T00:00:00+00:00",
        )
    assert not any(path.is_file() for path in archive.rglob("*"))


def test_archive_rejects_outcome_bearing_file_names(tmp_path: Path) -> None:
    source = tmp_path / "service"
    _fixture(source)
    forbidden = source / "captures/attempt-0002/settlement.json"
    forbidden.write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="outcome-bearing filename"):
        mirror_completed_capture_snapshot(
            service_root=source,
            archive_root=tmp_path / "archive",
            created_at="2030-01-01T00:00:00+00:00",
        )


def test_verifier_rejects_rehashed_manifest_with_missing_progress(
    tmp_path: Path,
) -> None:
    source = tmp_path / "service"
    archive = tmp_path / "archive"
    _fixture(source)
    mirror_completed_capture_snapshot(
        service_root=source,
        archive_root=archive,
        created_at="2030-01-01T00:00:00+00:00",
    )
    manifest = next(archive.glob("state_snapshots/*/capture_archive_manifest.json"))
    payload = json.loads(manifest.read_text())
    progress = next(
        row
        for row in payload["root_state_files"]
        if Path(row["path"]).name == "collection_progress.json"
    )
    payload["root_state_files"].remove(progress)
    (archive / progress["path"]).unlink()
    _refreeze_archive_manifest(manifest, payload)
    with pytest.raises(ValueError, match="root-state descriptor set"):
        verify_capture_archive_snapshot(manifest_path=manifest)


def test_verifier_rejects_rehashed_manifest_missing_ledger_bound_report(
    tmp_path: Path,
) -> None:
    source = tmp_path / "service"
    archive = tmp_path / "archive"
    _fixture(source)
    mirror_completed_capture_snapshot(
        service_root=source,
        archive_root=archive,
        created_at="2030-01-01T00:00:00+00:00",
    )
    manifest = next(archive.glob("state_snapshots/*/capture_archive_manifest.json"))
    payload = json.loads(manifest.read_text())
    report = next(
        row
        for row in payload["capture_files"]
        if row["path"].endswith("attempt-0001/pending_round_capture_report.json")
    )
    payload["capture_files"].remove(report)
    (archive / report["path"]).unlink()
    _refreeze_archive_manifest(manifest, payload)
    with pytest.raises(ValueError, match="capture report SHA"):
        verify_capture_archive_snapshot(manifest_path=manifest)


def test_verifier_rejects_rehashed_safety_unlock(tmp_path: Path) -> None:
    source = tmp_path / "service"
    archive = tmp_path / "archive"
    _fixture(source)
    mirror_completed_capture_snapshot(
        service_root=source,
        archive_root=archive,
        created_at="2030-01-01T00:00:00+00:00",
    )
    manifest = next(archive.glob("state_snapshots/*/capture_archive_manifest.json"))
    payload = json.loads(manifest.read_text())
    payload["live_trading_allowed"] = True
    _refreeze_archive_manifest(manifest, payload)
    with pytest.raises(ValueError, match="governance metadata"):
        verify_capture_archive_snapshot(manifest_path=manifest)
