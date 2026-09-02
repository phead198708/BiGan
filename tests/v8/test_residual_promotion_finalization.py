from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.residual_promotion_collection import canonical_attempt_hash
from bigan.v8.polymarket.residual_promotion_evaluation import build_market_results
from bigan.v8.polymarket.residual_promotion_finalization import (
    _execution_features,
    _market_level_decisions,
    _validate_decision_rows,
    _validate_quality_contract,
    freeze_exact_outcome_blind_population,
    select_exact_population,
    validate_frozen_population,
)
from bigan.v8.polymarket.residual_promotion_v1 import LINEAGE_ID

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "examples/v8/polymarket_configs" / LINEAGE_ID
AUTHORIZATION = CONFIG / "manual_collection_authorization_v3.json"
COLLECTOR_PROTOCOL = CONFIG / "prospective_collector_protocol_v3.json"
BUNDLE_SHA = (
    "7a5b872b5a2a010a0868bf7d22fb4bdc39a941dd04464bb53890d34aa1846b3e"
)


def _production_quality(*, missing_count: int = 12) -> dict:
    missing_counts = (
        {
            "selected_recent_trade_volume": 4,
            "opposite_recent_trade_volume": 4,
            "selected_minus_opposite_recent_trade_volume": 4,
        }
        if missing_count == 12
        else {}
    )
    return {
        "quality_valid": True,
        "quality_observations": {
            "book_capture_complete": True,
            "chainlink_capture_complete": True,
            "market_identity_complete": True,
            "paired_executable_asks_complete": True,
            "provider_capture_complete": True,
        },
        "invalid_reason_codes": [],
        "observed_decision_count": 2,
        "paired_executable_ask_decision_count": 2,
        "btc_feature_complete_decision_count": 2,
        "causality_violation_count": 0,
        "missing_feature_count": missing_count,
        "missing_feature_counts": missing_counts,
        "missing_values_encoded_as_zero": False,
    }


def test_production_quality_contract_preserves_native_missingness() -> None:
    _validate_quality_contract(
        {"quality": _production_quality()}, validation_fixture_only=False
    )


def test_production_execution_features_use_frozen_feature_envelope() -> None:
    execution = {
        "up_ask": 0.47,
        "up_bid": 0.46,
        "up_liquidity_depth": 15_022.94,
        "down_ask": 0.54,
        "down_bid": 0.53,
        "down_liquidity_depth": 15_022.94,
    }
    row = {
        "market_id": "market-1",
        "decision_ts": 1_900_000_000_001,
        "features": {**execution, "btc_return_1m": -0.001},
    }
    assert _execution_features(
        feature_rows=[row],
        market_id="market-1",
        decision_ts=1_900_000_000_001,
        validation_fixture_only=False,
    ) == execution


def test_production_execution_features_missing_envelope_fails_closed() -> None:
    row = {
        "market_id": "market-1",
        "decision_ts": 1_900_000_000_001,
        "up_ask": 0.47,
        "up_bid": 0.46,
        "up_liquidity_depth": 15_022.94,
        "down_ask": 0.54,
        "down_bid": 0.53,
        "down_liquidity_depth": 15_022.94,
    }
    with pytest.raises(ValueError, match="feature envelope is incomplete"):
        _execution_features(
            feature_rows=[row],
            market_id="market-1",
            decision_ts=1_900_000_000_001,
            validation_fixture_only=False,
        )


def test_finalizer_market_rows_feed_evaluator_cost_pipeline() -> None:
    candidate_rows = []
    baseline_rows = []
    settlements = []
    for index in range(10):
        market_id = f"synthetic-integration-{index:02d}"
        decision_ts = 1_900_000_000_000 + index
        decisions = [
            {
                "market_id": market_id,
                "decision_ts": decision_ts,
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
        feature_rows = [
            {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "features": {
                    "up_ask": 0.47,
                    "up_bid": 0.46,
                    "up_liquidity_depth": 15_022.94,
                    "down_ask": 0.54,
                    "down_bid": 0.53,
                    "down_liquidity_depth": 15_022.94,
                },
            }
        ]
        _validate_decision_rows(
            decisions,
            market_id=market_id,
            expected_candidate_bundle_sha256=BUNDLE_SHA,
        )
        candidate, baseline = _market_level_decisions(
            decisions=decisions,
            population_position=index + 1,
            attempt_index=index + 1,
            market_id=market_id,
            candidate_bundle_sha256=BUNDLE_SHA,
            baseline_artifact_sha256="b" * 64,
            feature_rows=feature_rows,
            validation_fixture_only=False,
        )
        candidate_rows.append(candidate)
        baseline_rows.append(baseline)
        settlements.append(
            {
                "market_id": market_id,
                "settlement_source": "synthetic_dry_run_only",
                "official_final": True,
                "inferred": False,
                "unresolved": False,
                "payout_up": int(index % 2 == 0),
                "payout_down": int(index % 2 != 0),
            }
        )

    results, reconciliation = build_market_results(
        candidate_rows=candidate_rows,
        baseline_rows=baseline_rows,
        settlements=settlements,
        target_market_count=10,
        synthetic_dry_run=True,
    )
    assert reconciliation["passed"] is True
    assert reconciliation["paired_market_count"] == 10
    assert {row["chronological_block"] for row in results} == {1, 2, 3, 4, 5}
    assert {row["chronological_half"] for row in results} == {"first", "second"}
    assert all(row["candidate_accepted"] is True for row in results)
    assert all(row["baseline_accepted"] is False for row in results)
    assert all(
        {
            "gross_price_edge",
            "entry_spread_cost",
            "fees",
            "slippage",
            "liquidity_impact",
            "total_cost",
            "unit_net_pnl",
        }
        <= set(row["candidate_cost_decomposition"])
        for row in results
    )
    assert all(
        row["baseline_cost_decomposition"]
        == {
            "gross_price_edge": 0.0,
            "entry_spread_cost": 0.0,
            "fees": 0.0,
            "slippage": 0.0,
            "liquidity_impact": 0.0,
            "total_cost": 0.0,
            "unit_net_pnl": 0.0,
        }
        for row in results
    )


@pytest.mark.parametrize(
    "mutation",
    (
        {"missing_feature_count": 11},
        {"missing_feature_counts": {"selected_recent_trade_volume": 11}},
        {"missing_feature_counts": {"selected_recent_trade_volume": -1}},
        {"missing_values_encoded_as_zero": True},
    ),
)
def test_production_quality_missingness_mismatch_fails_closed(
    mutation: dict,
) -> None:
    quality = _production_quality()
    quality.update(mutation)
    with pytest.raises(ValueError, match="internally inconsistent"):
        _validate_quality_contract(
            {"quality": quality}, validation_fixture_only=False
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
    candidate_rows = [
        json.loads(line)
        for line in (
            tmp_path / "exact_population_freeze/candidate_decision_rows.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert len(candidate_rows) == 2
    assert candidate_rows[0]["selected_side"] == "UP"
    assert candidate_rows[0]["execution_features"]["up_ask"] == 0.55
    assert candidate_rows[0]["execution_features_sha256"]
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
