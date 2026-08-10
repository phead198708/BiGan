"""Outcome-blind service-root migration tests for residual promotion v1."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.residual_promotion_collection import canonical_attempt_hash
from bigan.v8.polymarket.residual_promotion_migrated_restart import (
    validate_migrated_restart_authorization,
    verify_existing_coverage_resume_record,
)
from bigan.v8.polymarket.residual_promotion_service_root_migration import (
    migrate_service_root,
    snapshot_service_root,
    verify_service_root_migration,
)
from bigan.v8.polymarket.residual_promotion_v1 import (
    CANDIDATE_ID,
    LINEAGE_ID,
    MAXIMUM_ATTEMPTS,
    TARGET_MARKETS,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _fixture(root: Path) -> None:
    run = root / "captures/attempt-0001"
    raw = run / "raw/raw_polymarket_orderbooks.jsonl"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b'{"book":"decision-time-only"}\n')
    (run / "raw/raw_polymarket_resolutions.jsonl").write_bytes(b"")
    manifest = run / "pending_round_capture_manifest.json"
    report = run / "pending_round_capture_report.json"
    manifest.write_bytes(b'{"resolution_provider_called":false}\n')
    report.write_bytes(b'{"resolution_provider_called":false}\n')
    attempt = {
        "schema_version": "bigan-btc-15m-residual-promotion-attempt-v1",
        "lineage_id": LINEAGE_ID,
        "attempt_index": 1,
        "attempt_id": run.name,
        "market_id": "market-1",
        "quality": {"quality_valid": True},
        "provider_health": {"provider_failed": False, "retry_used": False},
        "decision_rows": [],
        "previous_attempt_hash": "0" * 64,
        "outcomes_accessed": False,
        "settlement_accessed": False,
        "pnl_accessed": False,
        "fresh_outcomes_opened": False,
        "interim_pnl_evaluated": False,
        "wallet_signing_allowed": False,
        "polymarket_write_allowed": False,
        "capital_at_risk": False,
        "safety": dict(SAFETY),
    }
    attempt["attempt_hash"] = canonical_attempt_hash(attempt)
    (root / "outcome_blind_attempts.jsonl").write_text(
        json.dumps(attempt, sort_keys=True) + "\n"
    )
    progress = {
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "attempts_consumed": 1,
        "quality_valid_market_count": 1,
        "maximum_attempts": MAXIMUM_ATTEMPTS,
        "target_quality_valid_market_count": TARGET_MARKETS,
        "hash_chain_status": "valid",
        "candidate_bundle_sha256": "a" * 64,
        "authorization_sha256": "b" * 64,
        "collector_protocol_sha256": "c" * 64,
        "fresh_outcomes_opened": False,
        "interim_pnl_evaluated": False,
        "safety": dict(SAFETY),
    }
    _write_json(root / "collection_progress.json", progress)
    _write_json(root / "collection_start_record.json", {"safety": SAFETY})
    _write_json(root / "collection_resume_record_v3.json", {"safety": SAFETY})
    (root / ".DS_Store").write_bytes(b"ignored metadata")
    (root / "captures/.DS_Store").write_bytes(b"ignored metadata")
    (root / "captures/attempt-0002/provider_raw").mkdir(parents=True)
    (root / "captures/attempt-0002/raw").mkdir()


def test_migration_copies_and_verifies_exact_service_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    report = tmp_path / "audit/migration.json"
    _fixture(source)

    result = migrate_service_root(
        source_root=source,
        destination_root=destination,
        report_path=report,
        created_at="2030-01-01T00:00:00Z",
    )

    assert result["verification_passed"] is True
    assert result["source_snapshot"] == result["destination_snapshot"]
    assert result["runtime_write_metadata_safe"] is True
    assert result["attempt_count"] == 1
    assert result["quality_valid_market_count"] == 1
    assert not (destination / ".DS_Store").exists()
    assert (destination / "captures/attempt-0002").is_dir()
    assert list((destination / "captures/attempt-0002/provider_raw").iterdir()) == []
    assert list((destination / "captures/attempt-0002/raw").iterdir()) == []
    assert result["outcomes_accessed"] is False
    assert result["settlement_accessed"] is False
    assert result["pnl_accessed"] is False
    assert result["source_capture_mutated"] is False
    assert result["source_capture_deleted"] is False
    assert result["collection_population_changed"] is False
    assert result["collector_decisions_changed"] is False
    assert result["safety"] == SAFETY


def test_migration_fails_on_nonempty_unledgered_attempt(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _fixture(source)
    (source / "captures/attempt-0002/partial.bin").write_bytes(b"partial")

    with pytest.raises(ValueError, match="unledgered capture directory contains bytes"):
        snapshot_service_root(source)


def test_migration_fails_on_nonempty_resolution_stream(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _fixture(source)
    (source / "captures/attempt-0001/raw/raw_polymarket_resolutions.jsonl").write_bytes(
        b'{"forbidden":"opened"}\n'
    )

    with pytest.raises(ValueError, match="nonempty resolution stream"):
        snapshot_service_root(source)


def test_migration_fails_on_appledouble_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _fixture(source)
    (source / "captures/attempt-0001/._book").write_bytes(b"metadata")

    with pytest.raises(ValueError, match="AppleDouble"):
        snapshot_service_root(source)


def test_migration_verifier_fails_on_destination_drift(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    report = tmp_path / "audit/migration.json"
    _fixture(source)
    migrate_service_root(
        source_root=source,
        destination_root=destination,
        report_path=report,
        created_at="2030-01-01T00:00:00Z",
    )
    (destination / "captures/attempt-0001/raw/raw_polymarket_orderbooks.jsonl").write_bytes(
        b"drift\n"
    )

    with pytest.raises(ValueError, match="destination drift"):
        verify_service_root_migration(report_path=report)


def test_migration_verifier_fails_on_report_hash_drift(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    report = tmp_path / "audit/migration.json"
    _fixture(source)
    migrate_service_root(
        source_root=source,
        destination_root=destination,
        report_path=report,
        created_at="2030-01-01T00:00:00Z",
    )
    report.write_text(report.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="report hash mismatch"):
        verify_service_root_migration(report_path=report)


def test_migration_resumes_only_when_existing_destination_bytes_match(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    report = tmp_path / "audit/migration.json"
    _fixture(source)
    destination.mkdir()
    (destination / "collection_progress.json").write_bytes(
        (source / "collection_progress.json").read_bytes()
    )
    (destination / "._collection_progress.json").write_bytes(b"metadata")

    result = migrate_service_root(
        source_root=source,
        destination_root=destination,
        report_path=report,
        created_at="2030-01-01T00:00:00Z",
    )

    assert result["verification_passed"] is True
    assert not list(destination.rglob("._*"))


def test_migration_fails_closed_on_existing_destination_byte_drift(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _fixture(source)
    destination.mkdir()
    (destination / "collection_progress.json").write_bytes(b"drift\n")

    with pytest.raises(ValueError, match="destination contains byte drift"):
        migrate_service_root(
            source_root=source,
            destination_root=destination,
            report_path=tmp_path / "audit/migration.json",
            created_at="2030-01-01T00:00:00Z",
        )


def _repo_descriptor(root: Path, relative: str) -> dict[str, str]:
    return {"path": relative, "sha256": sha256_file(root / relative)}


def _test_repository(tmp_path: Path) -> Path:
    source = Path(__file__).resolve().parents[2]
    destination = tmp_path / "repository"
    config = (
        "examples/v8/polymarket_configs/"
        "BTC-15M-cost-aware-market-residual-promotion-v1"
    )
    paths = (
        "src/bigan/v8/polymarket/residual_promotion_migrated_restart.py",
        "examples/v8/run_residual_promotion_v1_collector.py",
        "examples/v8/run_residual_promotion_v1_migrated_collector.py",
        f"{config}/manual_collection_authorization_v3.json",
        f"{config}/prospective_collector_protocol_v3.json",
        f"{config}/candidate_bundle/bundle_manifest.json",
    )
    for relative in paths:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    return destination


def _restart_authorization(
    *, root: Path, destination: Path, report: Path
) -> dict:
    config = (
        "examples/v8/polymarket_configs/"
        "BTC-15M-cost-aware-market-residual-promotion-v1"
    )
    return {
        "schema_version": (
            "bigan-btc-15m-residual-promotion-"
            "external-service-root-restart-authorization-v1"
        ),
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "authorization_scope": (
            "one_byte_verified_external_service_root_restart_only"
        ),
        "destination_root": str(destination.resolve()),
        "migration_report": {
            "path": str(report.resolve()),
            "sha256": sha256_file(report),
        },
        "service_root_tree_sha256": json.loads(report.read_text())[
            "source_snapshot"
        ]["tree_sha256"],
        "ledger_closed_attempt_count": 1,
        "quality_valid_market_count": 1,
        "maximum_attempts": MAXIMUM_ATTEMPTS,
        "target_quality_valid_market_count": TARGET_MARKETS,
        "implementation": _repo_descriptor(
            root,
            "src/bigan/v8/polymarket/residual_promotion_migrated_restart.py",
        ),
        "original_manual_collection_authorization": _repo_descriptor(
            root, f"{config}/manual_collection_authorization_v3.json"
        ),
        "original_collector_protocol": _repo_descriptor(
            root, f"{config}/prospective_collector_protocol_v3.json"
        ),
        "candidate_bundle_manifest": _repo_descriptor(
            root, f"{config}/candidate_bundle/bundle_manifest.json"
        ),
        "original_collector_cli": _repo_descriptor(
            root, "examples/v8/run_residual_promotion_v1_collector.py"
        ),
        "restart_cli": _repo_descriptor(
            root, "examples/v8/run_residual_promotion_v1_migrated_collector.py"
        ),
        "collector_restart_authorized": True,
        "collector_restart_completed": False,
        "source_capture_deletion_authorized": False,
        "source_service_root_mutation_authorized": False,
        "model_bytes_changed": False,
        "router_features_cost_action_baseline_or_population_changed": False,
        "collector_decisions_changed": False,
        "fresh_outcomes_opened": False,
        "outcomes_accessed": False,
        "settlement_accessed": False,
        "pnl_accessed": False,
        "paper_candidate_allowed": False,
        "v8_execution_handoff_allowed": False,
        "live_trading_allowed": False,
        "wallet_signing_allowed": False,
        "polymarket_write_allowed": False,
        "capital_at_risk": False,
        "safety": dict(SAFETY),
    }


def test_migrated_restart_authorization_reconciles_exact_tree_and_bindings(
    tmp_path: Path,
) -> None:
    root = _test_repository(tmp_path)
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    report = tmp_path / "audit/migration.json"
    _fixture(source)
    config = (
        root
        / "examples/v8/polymarket_configs/"
        "BTC-15M-cost-aware-market-residual-promotion-v1"
    )
    progress_path = source / "collection_progress.json"
    progress = json.loads(progress_path.read_text())
    progress.update(
        {
            "authorization_sha256": sha256_file(
                config / "manual_collection_authorization_v3.json"
            ),
            "collector_protocol_sha256": sha256_file(
                config / "prospective_collector_protocol_v3.json"
            ),
            "candidate_bundle_sha256": sha256_file(
                config / "candidate_bundle/bundle_manifest.json"
            ),
        }
    )
    _write_json(progress_path, progress)
    migrate_service_root(
        source_root=source,
        destination_root=destination,
        report_path=report,
        created_at="2030-01-01T00:00:00Z",
    )
    authorization = root / "restart_authorization.json"
    _write_json(
        authorization,
        _restart_authorization(root=root, destination=destination, report=report),
    )
    authorization.with_suffix(".json.sha256").write_text(
        sha256_file(authorization) + "\n"
    )

    result = validate_migrated_restart_authorization(
        authorization_path=authorization,
        repository_root=root,
        service_root=destination,
    )

    assert result["validation_passed"] is True
    assert result["attempts_consumed"] == 1
    assert result["fresh_outcomes_opened"] is False
    assert result["safety"] == SAFETY


def test_migrated_restart_authorization_fails_on_restart_cli_drift(
    tmp_path: Path,
) -> None:
    root = _test_repository(tmp_path)
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    report = tmp_path / "audit/migration.json"
    _fixture(source)
    config = (
        root
        / "examples/v8/polymarket_configs/"
        "BTC-15M-cost-aware-market-residual-promotion-v1"
    )
    progress = json.loads((source / "collection_progress.json").read_text())
    progress.update(
        {
            "authorization_sha256": sha256_file(
                config / "manual_collection_authorization_v3.json"
            ),
            "collector_protocol_sha256": sha256_file(
                config / "prospective_collector_protocol_v3.json"
            ),
            "candidate_bundle_sha256": sha256_file(
                config / "candidate_bundle/bundle_manifest.json"
            ),
        }
    )
    _write_json(source / "collection_progress.json", progress)
    migrate_service_root(
        source_root=source,
        destination_root=destination,
        report_path=report,
        created_at="2030-01-01T00:00:00Z",
    )
    payload = _restart_authorization(
        root=root, destination=destination, report=report
    )
    payload["restart_cli"]["sha256"] = "0" * 64
    authorization = root / "restart_authorization.json"
    _write_json(authorization, payload)
    authorization.with_suffix(".json.sha256").write_text(
        sha256_file(authorization) + "\n"
    )

    with pytest.raises(ValueError, match="restart_cli"):
        validate_migrated_restart_authorization(
            authorization_path=authorization,
            repository_root=root,
            service_root=destination,
        )


def test_migrated_restart_preserves_original_resume_record_bytes(
    tmp_path: Path,
) -> None:
    validation = {
        "authorization_sha256": "a" * 64,
        "collector_protocol_sha256": "b" * 64,
        "bundle": {"sha256": "c" * 64},
    }
    _write_json(
        tmp_path / "collection_start_record.json",
        {"fresh_outcomes_opened": False, "safety": SAFETY},
    )
    resume = {
        "schema_version": "bigan-btc-15m-residual-promotion-resume-v3",
        "resumed_attempt_count": 1,
        "prior_attempts_preserved": True,
        "coverage_instrumentation_corrected": True,
        "rest_fallback_collection_seconds": 5.0,
        "authorization_sha256": "a" * 64,
        "collector_protocol_sha256": "b" * 64,
        "candidate_bundle_sha256": "c" * 64,
        "fresh_outcomes_opened": False,
        "zero_capital_read_only": True,
        "wallet_signing_allowed": False,
        "polymarket_write_allowed": False,
        "capital_at_risk": False,
        "safety": SAFETY,
    }
    path = tmp_path / "collection_resume_record_v3.json"
    _write_json(path, resume)
    frozen_bytes = path.read_bytes()

    verify_existing_coverage_resume_record(
        tmp_path,
        original_validation=validation,
        resumed_attempt_count=9,
        rest_fallback_collection_seconds=5.0,
    )

    assert path.read_bytes() == frozen_bytes
