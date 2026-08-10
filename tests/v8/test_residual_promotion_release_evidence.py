"""Outcome-blind release-evidence tests for residual promotion v1."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.residual_promotion_collection import canonical_attempt_hash
from bigan.v8.polymarket.residual_promotion_release_evidence import (
    _validate_complete_progress,
    build_outcome_blind_shadow_stability_report,
    run_outcome_blind_operational_rollback_drill,
)
from bigan.v8.polymarket.residual_promotion_release_readiness import (
    FUNCTIONAL_ROLLBACK_REPOSITORY_PATH,
    PARITY_REPOSITORY_PATH,
)
from bigan.v8.polymarket.residual_promotion_v1 import (
    CANDIDATE_ID,
    LINEAGE_ID,
    MAXIMUM_ATTEMPTS,
    TARGET_MARKETS,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE = (
    REPO_ROOT
    / "examples/v8/polymarket_configs"
    / LINEAGE_ID
    / "candidate_bundle/bundle_manifest.json"
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    path.with_suffix(path.suffix + ".sha256").write_text(sha256_file(path) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> dict[str, str]:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    )
    path.with_suffix(path.suffix + ".sha256").write_text(sha256_file(path) + "\n")
    return {"path": path.name, "sha256": sha256_file(path)}


def _repo_descriptor(relative_path: str) -> dict[str, str]:
    path = REPO_ROOT / relative_path
    return {"path": relative_path, "sha256": sha256_file(path)}


def _fixture(service_root: Path) -> tuple[Path, str]:
    freeze = service_root / "exact_population_freeze"
    freeze.mkdir(parents=True)
    market_ids = [f"market-{index:04d}" for index in range(TARGET_MARKETS)]
    population = [
        {"population_position": index, "market_id": market_id}
        for index, market_id in enumerate(market_ids, start=1)
    ]
    candidate = [dict(row) for row in population]
    baseline = [dict(row) for row in population]
    artifacts = {
        "population_rows": _write_jsonl(freeze / "exact_population_rows.jsonl", population),
        "candidate_decision_rows": _write_jsonl(
            freeze / "candidate_decision_rows.jsonl", candidate
        ),
        "baseline_decision_rows": _write_jsonl(freeze / "baseline_decision_rows.jsonl", baseline),
    }
    manifest = {
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "candidate_bundle": {
            "path": BUNDLE.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(BUNDLE),
        },
        "runtime_parity_report": _repo_descriptor(PARITY_REPOSITORY_PATH),
        "ordered_market_ids_sha256": canonical_json_sha256(market_ids),
        "artifacts": artifacts,
    }
    manifest_path = freeze / "exact_population_manifest.json"
    _write_json(manifest_path, manifest)

    attempts: list[dict] = []
    previous = "0" * 64
    for index in range(1, TARGET_MARKETS + 1):
        attempt = {
            "attempt_index": index,
            "previous_attempt_hash": previous,
            "outcomes_accessed": False,
            "settlement_accessed": False,
            "pnl_accessed": False,
            "fresh_outcomes_opened": False,
            "interim_pnl_evaluated": False,
            "safety": dict(SAFETY),
        }
        attempt["attempt_hash"] = canonical_attempt_hash(attempt)
        previous = attempt["attempt_hash"]
        attempts.append(attempt)
    (service_root / "outcome_blind_attempts.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in attempts)
    )
    progress = {
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "attempts_consumed": TARGET_MARKETS,
        "maximum_attempts": MAXIMUM_ATTEMPTS,
        "quality_valid_market_count": TARGET_MARKETS,
        "target_quality_valid_market_count": TARGET_MARKETS,
        "remaining_quality_valid_markets": 0,
        "collection_complete": True,
        "attempt_cap_exhausted": False,
        "hash_chain_status": "valid",
        "fresh_outcomes_opened": False,
        "interim_pnl_evaluated": False,
        "collection_influenced_by_model_decisions": False,
        "wallet_signing_allowed": False,
        "polymarket_write_allowed": False,
        "capital_at_risk": False,
        "safety": dict(SAFETY),
    }
    (service_root / "collection_progress.json").write_text(
        json.dumps(progress, indent=2, sort_keys=True) + "\n"
    )
    return freeze, sha256_file(manifest_path)


def test_exact_population_shadow_report_is_outcome_blind_and_single_use(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    freeze, manifest_sha = _fixture(tmp_path)
    monkeypatch.setattr(
        "bigan.v8.polymarket.residual_promotion_release_evidence.validate_frozen_population",
        lambda **_kwargs: {"validation_passed": True},
    )
    output = tmp_path / "release/shadow_stability.json"
    report = build_outcome_blind_shadow_stability_report(
        repository_root=REPO_ROOT,
        service_root=tmp_path,
        freeze_dir=freeze,
        expected_population_manifest_sha256=manifest_sha,
        output_path=output,
        created_at="2030-01-01T00:00:00+00:00",
    )
    assert report["shadow_stability_passed"] is True
    assert report["candidate_row_count"] == TARGET_MARKETS
    assert report["baseline_row_count"] == TARGET_MARKETS
    assert report["paired_row_count"] == TARGET_MARKETS
    assert report["runtime_decision_parity_passed"] is True
    assert report["kill_switch_wired"] is True
    assert report["outcomes_accessed"] is False
    assert report["settlement_accessed"] is False
    assert report["pnl_accessed"] is False
    assert report["safety"] == SAFETY
    assert output.with_suffix(".json.sha256").read_text().strip() == sha256_file(output)
    with pytest.raises(FileExistsError, match="already exists"):
        build_outcome_blind_shadow_stability_report(
            repository_root=REPO_ROOT,
            service_root=tmp_path,
            freeze_dir=freeze,
            expected_population_manifest_sha256=manifest_sha,
            output_path=output,
            created_at="2030-01-01T00:00:00+00:00",
        )


def test_shadow_report_rejects_population_reordering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    freeze, manifest_sha = _fixture(tmp_path)
    monkeypatch.setattr(
        "bigan.v8.polymarket.residual_promotion_release_evidence.validate_frozen_population",
        lambda **_kwargs: {"validation_passed": True},
    )
    baseline_path = freeze / "baseline_decision_rows.jsonl"
    rows = [json.loads(line) for line in baseline_path.read_text().splitlines()]
    rows[0], rows[1] = rows[1], rows[0]
    manifest_path = freeze / "exact_population_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["baseline_decision_rows"] = _write_jsonl(baseline_path, rows)
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="identity/order mismatch"):
        build_outcome_blind_shadow_stability_report(
            repository_root=REPO_ROOT,
            service_root=tmp_path,
            freeze_dir=freeze,
            expected_population_manifest_sha256=sha256_file(manifest_path),
            output_path=tmp_path / "shadow.json",
            created_at="2030-01-01T00:00:00+00:00",
        )


def test_shadow_report_rejects_any_outcome_opening(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    freeze, manifest_sha = _fixture(tmp_path)
    monkeypatch.setattr(
        "bigan.v8.polymarket.residual_promotion_release_evidence.validate_frozen_population",
        lambda **_kwargs: {"validation_passed": True},
    )
    progress_path = tmp_path / "collection_progress.json"
    progress = json.loads(progress_path.read_text())
    progress["fresh_outcomes_opened"] = True
    progress_path.write_text(json.dumps(progress) + "\n")
    with pytest.raises(ValueError, match="outcome-blind safety field"):
        build_outcome_blind_shadow_stability_report(
            repository_root=REPO_ROOT,
            service_root=tmp_path,
            freeze_dir=freeze,
            expected_population_manifest_sha256=manifest_sha,
            output_path=tmp_path / "shadow.json",
            created_at="2030-01-01T00:00:00+00:00",
        )


def test_functional_rollback_descriptor_is_repository_pinned() -> None:
    descriptor = _repo_descriptor(FUNCTIONAL_ROLLBACK_REPOSITORY_PATH)
    assert len(descriptor["sha256"]) == 64


def test_operational_rollback_is_zero_capital_and_deterministically_timed(
    tmp_path: Path,
) -> None:
    clock_values = iter([0, 1_000_000, 2_000_000, 4_000_000, 5_000_000, 8_000_000])
    output = tmp_path / "operational_rollback.json"
    report = run_outcome_blind_operational_rollback_drill(
        repository_root=REPO_ROOT,
        output_path=output,
        created_at="2030-01-01T00:00:00+00:00",
        iterations=3,
        clock_ns=lambda: next(clock_values),
    )
    assert report["latency_measurements_ms"] == [1.0, 2.0, 3.0]
    assert report["maximum_observed_latency_ms"] == 3.0
    assert report["rollback_drill_passed"] is True
    assert report["rollback_target"] == "NO_TRADE"
    assert report["phase6_zero_capital_authorized"] is False
    assert report["micro_live_authorized"] is False
    assert report["development_outcomes_accessed"] is True
    assert report["development_outcome_scope"] == "development_only_forever"
    assert report["fresh_outcomes_accessed"] is False
    assert report["outcomes_accessed"] is True
    assert report["wallet_signing_allowed"] is False
    assert report["polymarket_write_allowed"] is False
    assert report["capital_at_risk"] is False
    assert output.with_suffix(".json.sha256").read_text().strip() == sha256_file(output)


def test_shadow_progress_allows_target_reached_on_final_authorized_attempt() -> None:
    progress = {
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "attempts_consumed": MAXIMUM_ATTEMPTS,
        "maximum_attempts": MAXIMUM_ATTEMPTS,
        "quality_valid_market_count": TARGET_MARKETS,
        "target_quality_valid_market_count": TARGET_MARKETS,
        "remaining_quality_valid_markets": 0,
        "collection_complete": True,
        "attempt_cap_exhausted": True,
        "hash_chain_status": "valid",
        "fresh_outcomes_opened": False,
        "interim_pnl_evaluated": False,
        "collection_influenced_by_model_decisions": False,
        "wallet_signing_allowed": False,
        "polymarket_write_allowed": False,
        "capital_at_risk": False,
        "safety": dict(SAFETY),
    }
    _validate_complete_progress(
        progress,
        attempt_count=MAXIMUM_ATTEMPTS,
        target_market_count=TARGET_MARKETS,
        validation_fixture_only=False,
    )


def test_operational_rollback_latency_breach_writes_failure_then_fails_closed(
    tmp_path: Path,
) -> None:
    clock_values = iter([0, 251_000_000])
    output = tmp_path / "operational_rollback_failed.json"
    with pytest.raises(ValueError, match="failed closed"):
        run_outcome_blind_operational_rollback_drill(
            repository_root=REPO_ROOT,
            output_path=output,
            created_at="2030-01-01T00:00:00+00:00",
            iterations=1,
            clock_ns=lambda: next(clock_values),
        )
    report = json.loads(output.read_text())
    assert report["maximum_observed_latency_ms"] == 251.0
    assert report["rollback_drill_passed"] is False
    assert report["micro_live_authorized"] is False
    assert report["capital_at_risk"] is False
