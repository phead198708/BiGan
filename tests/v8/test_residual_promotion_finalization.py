from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.residual_promotion_collection import canonical_attempt_hash
from bigan.v8.polymarket.residual_promotion_finalization import (
    freeze_exact_outcome_blind_population,
    select_exact_population,
    validate_frozen_population,
)
from bigan.v8.polymarket.residual_promotion_v1 import LINEAGE_ID

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "examples/v8/polymarket_configs" / LINEAGE_ID
AUTHORIZATION = CONFIG / "manual_collection_authorization_v2.json"
COLLECTOR_PROTOCOL = CONFIG / "prospective_collector_protocol_v2.json"
BUNDLE_SHA = (
    "7a5b872b5a2a010a0868bf7d22fb4bdc39a941dd04464bb53890d34aa1846b3e"
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _capture(service_root: Path, *, attempt_id: str) -> tuple[str, str]:
    run_dir = service_root / "captures" / attempt_id
    manifest = run_dir / "pending_round_capture_manifest.json"
    report = run_dir / "pending_round_capture_report.json"
    _write_json(
        manifest,
        {
            "schema_version": "fixture",
            "resolution_provider_called": False,
            "outcomes_accessed": False,
        },
    )
    _write_json(
        report,
        {
            "schema_version": "fixture",
            "resolution_provider_called": False,
            "outcomes_accessed": False,
        },
    )
    resolution = run_dir / "raw" / "raw_polymarket_resolutions.jsonl"
    resolution.parent.mkdir(parents=True, exist_ok=True)
    resolution.write_text("", encoding="utf-8")
    return sha256_file(manifest), sha256_file(report)


def _attempt(
    service_root: Path,
    *,
    index: int,
    valid: bool,
    market_id: str | None = None,
) -> dict:
    attempt_id = f"attempt-{index}"
    manifest_sha, report_sha = _capture(service_root, attempt_id=attempt_id)
    resolved_market = market_id or (f"market-{index}" if valid else None)
    decisions = (
        [
            {
                "market_id": resolved_market,
                "decision_ts": 1_900_000_000_000 + index,
                "candidate_action_values": {
                    "NO_TRADE": 0.0,
                    "BUY_UP_HOLD": 0.01,
                    "BUY_DOWN_HOLD": -0.02,
                },
                "candidate_selected_action": "BUY_UP_HOLD",
                "candidate_accepted_at_this_decision": True,
                "baseline_action_values": {
                    "NO_TRADE": 0.0,
                    "BUY_UP_HOLD": -0.01,
                    "BUY_DOWN_HOLD": -0.02,
                },
                "baseline_selected_action": "NO_TRADE",
                "baseline_accepted_at_this_decision": False,
                "baseline_fail_closed": False,
                "baseline_fail_closed_reasons": [],
                "candidate_bundle_sha256": BUNDLE_SHA,
                "decision_influenced_collection": False,
                "outcomes_accessed": False,
                "settlement_accessed": False,
                "pnl_accessed": False,
                "safety": dict(SAFETY),
            }
        ]
        if valid
        else []
    )
    return {
        "attempt_index": index,
        "attempt_id": attempt_id,
        "market_id": resolved_market,
        "scheduled_round_start_ts": 1_900_000_000_000 + index,
        "capture_manifest_sha256": manifest_sha,
        "capture_report_sha256": report_sha,
        "quality": {
            "quality_valid": valid,
            "invalid_reason_codes": [] if valid else ["provider_incomplete"],
        },
        "provider_health": {
            "provider_failed": not valid,
            "retry_used": False,
        },
        "decision_rows": decisions,
        "outcomes_accessed": False,
        "settlement_accessed": False,
        "pnl_accessed": False,
        "safety": dict(SAFETY),
    }


def _write_chain(service_root: Path, attempts: list[dict]) -> list[dict]:
    previous = "0" * 64
    chained = []
    for source in attempts:
        row = dict(source)
        row["previous_attempt_hash"] = previous
        row["attempt_hash"] = canonical_attempt_hash(row)
        previous = row["attempt_hash"]
        chained.append(row)
    ledger = service_root / "outcome_blind_attempts.jsonl"
    ledger.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in chained),
        encoding="utf-8",
    )
    return chained


def test_selects_chronological_first_valid_unique_population() -> None:
    attempts = [
        {"attempt_index": 1, "quality": {"quality_valid": False}, "market_id": None},
        {"attempt_index": 2, "quality": {"quality_valid": True}, "market_id": "a"},
        {"attempt_index": 3, "quality": {"quality_valid": True}, "market_id": "a"},
        {"attempt_index": 4, "quality": {"quality_valid": True}, "market_id": "b"},
    ]
    result = select_exact_population(attempts, target_market_count=2)
    assert result["population_complete"] is True
    assert [row["market_id"] for row in result["selected_attempts"]] == ["a", "b"]
    assert result["stop_attempt_index"] == 4
    assert result["post_boundary_attempt_count"] == 0


def test_incomplete_population_does_not_expose_partial_selection() -> None:
    result = select_exact_population(
        [{"attempt_index": 1, "quality": {"quality_valid": True}, "market_id": "a"}],
        target_market_count=2,
    )
    assert result["population_complete"] is False
    assert result["selected_attempts"] == []


def test_freezes_and_revalidates_outcome_blind_fixture(tmp_path: Path) -> None:
    attempts = [
        _attempt(tmp_path, index=1, valid=False),
        _attempt(tmp_path, index=2, valid=True),
        _attempt(tmp_path, index=3, valid=True),
    ]
    _write_chain(tmp_path, attempts)
    result = freeze_exact_outcome_blind_population(
        service_root=tmp_path,
        repository_root=REPO_ROOT,
        authorization_path=AUTHORIZATION,
        collector_protocol_path=COLLECTOR_PROTOCOL,
        created_at="2026-08-10T15:00:00+00:00",
        target_market_count=2,
        validation_fixture_only=True,
    )
    assert result["outcome_access_authorized"] is False
    assert result["outcomes_accessed"] is False
    validation = result["population_validation"]
    assert validation["validation_passed"] is True
    assert validation["exact_market_count"] == 2
    assert validation["candidate_decision_row_count"] == 2
    assert validation["baseline_decision_row_count"] == 2
    manifest = json.loads(
        (tmp_path / "exact_population_freeze/exact_population_manifest.json").read_text()
    )
    assert manifest["ordered_market_ids_sha256"]
    assert manifest["source_capture_mutated"] is False
    assert manifest["outcome_access_authorized"] is False
    assert manifest["safety"] == SAFETY


def test_post_boundary_attempt_fails_closed(tmp_path: Path) -> None:
    attempts = [
        _attempt(tmp_path, index=1, valid=True),
        _attempt(tmp_path, index=2, valid=True),
        _attempt(tmp_path, index=3, valid=False),
    ]
    _write_chain(tmp_path, attempts)
    with pytest.raises(ValueError, match="after the exact population boundary"):
        freeze_exact_outcome_blind_population(
            service_root=tmp_path,
            repository_root=REPO_ROOT,
            authorization_path=AUTHORIZATION,
            collector_protocol_path=COLLECTOR_PROTOCOL,
            target_market_count=2,
            validation_fixture_only=True,
        )


def test_outcome_bearing_attempt_fails_closed(tmp_path: Path) -> None:
    attempts = [
        _attempt(tmp_path, index=1, valid=True),
        _attempt(tmp_path, index=2, valid=True),
    ]
    attempts[0]["resolved_outcome"] = "UP"
    _write_chain(tmp_path, attempts)
    with pytest.raises(ValueError, match="forbidden outcome-bearing field"):
        freeze_exact_outcome_blind_population(
            service_root=tmp_path,
            repository_root=REPO_ROOT,
            authorization_path=AUTHORIZATION,
            collector_protocol_path=COLLECTOR_PROTOCOL,
            target_market_count=2,
            validation_fixture_only=True,
        )


def test_nonprospective_population_fails_closed(tmp_path: Path) -> None:
    attempts = [
        _attempt(tmp_path, index=1, valid=True),
        _attempt(tmp_path, index=2, valid=True),
    ]
    attempts[0]["scheduled_round_start_ts"] = 1_700_000_000_000
    _write_chain(tmp_path, attempts)
    with pytest.raises(ValueError, match="not strictly prospective"):
        freeze_exact_outcome_blind_population(
            service_root=tmp_path,
            repository_root=REPO_ROOT,
            authorization_path=AUTHORIZATION,
            collector_protocol_path=COLLECTOR_PROTOCOL,
            target_market_count=2,
            validation_fixture_only=True,
        )


def test_frozen_artifact_byte_drift_fails_closed(tmp_path: Path) -> None:
    attempts = [
        _attempt(tmp_path, index=1, valid=True),
        _attempt(tmp_path, index=2, valid=True),
    ]
    _write_chain(tmp_path, attempts)
    result = freeze_exact_outcome_blind_population(
        service_root=tmp_path,
        repository_root=REPO_ROOT,
        authorization_path=AUTHORIZATION,
        collector_protocol_path=COLLECTOR_PROTOCOL,
        target_market_count=2,
        validation_fixture_only=True,
    )
    candidate = tmp_path / "exact_population_freeze/candidate_decision_rows.jsonl"
    candidate.write_bytes(candidate.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="artifact drift"):
        validate_frozen_population(
            freeze_dir=tmp_path / "exact_population_freeze",
            service_root=tmp_path,
            repository_root=REPO_ROOT,
            expected_manifest_sha256=result["manifest"]["sha256"],
            target_market_count=2,
            validation_fixture_only=True,
        )


def test_raw_capture_byte_drift_fails_closed(tmp_path: Path) -> None:
    attempts = [
        _attempt(tmp_path, index=1, valid=True),
        _attempt(tmp_path, index=2, valid=True),
    ]
    _write_chain(tmp_path, attempts)
    result = freeze_exact_outcome_blind_population(
        service_root=tmp_path,
        repository_root=REPO_ROOT,
        authorization_path=AUTHORIZATION,
        collector_protocol_path=COLLECTOR_PROTOCOL,
        target_market_count=2,
        validation_fixture_only=True,
    )
    raw = tmp_path / "captures/attempt-1/raw/raw_polymarket_resolutions.jsonl"
    raw.write_text("byte drift without an outcome value\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source artifact drift"):
        validate_frozen_population(
            freeze_dir=tmp_path / "exact_population_freeze",
            service_root=tmp_path,
            repository_root=REPO_ROOT,
            expected_manifest_sha256=result["manifest"]["sha256"],
            target_market_count=2,
            validation_fixture_only=True,
        )


def test_extra_frozen_file_fails_closed(tmp_path: Path) -> None:
    attempts = [
        _attempt(tmp_path, index=1, valid=True),
        _attempt(tmp_path, index=2, valid=True),
    ]
    _write_chain(tmp_path, attempts)
    result = freeze_exact_outcome_blind_population(
        service_root=tmp_path,
        repository_root=REPO_ROOT,
        authorization_path=AUTHORIZATION,
        collector_protocol_path=COLLECTOR_PROTOCOL,
        target_market_count=2,
        validation_fixture_only=True,
    )
    extra = tmp_path / "exact_population_freeze/unregistered.json"
    _write_json(extra, {"unexpected": True})
    with pytest.raises(ValueError, match="directory file set mismatch"):
        validate_frozen_population(
            freeze_dir=tmp_path / "exact_population_freeze",
            service_root=tmp_path,
            repository_root=REPO_ROOT,
            expected_manifest_sha256=result["manifest"]["sha256"],
            target_market_count=2,
            validation_fixture_only=True,
        )


def test_repository_implementation_drift_fails_closed(tmp_path: Path) -> None:
    attempts = [
        _attempt(tmp_path, index=1, valid=True),
        _attempt(tmp_path, index=2, valid=True),
    ]
    _write_chain(tmp_path, attempts)
    freeze_exact_outcome_blind_population(
        service_root=tmp_path,
        repository_root=REPO_ROOT,
        authorization_path=AUTHORIZATION,
        collector_protocol_path=COLLECTOR_PROTOCOL,
        target_market_count=2,
        validation_fixture_only=True,
    )
    manifest_path = tmp_path / "exact_population_freeze/exact_population_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["finalization_implementation"]["sha256"] = "0" * 64
    _write_json(manifest_path, manifest)
    manifest_path.with_suffix(".json.sha256").write_text(
        sha256_file(manifest_path) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="repository artifact SHA-256 mismatch"):
        validate_frozen_population(
            freeze_dir=tmp_path / "exact_population_freeze",
            service_root=tmp_path,
            repository_root=REPO_ROOT,
            expected_manifest_sha256=sha256_file(manifest_path),
            target_market_count=2,
            validation_fixture_only=True,
        )
