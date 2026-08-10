"""Outcome-blind slug/two-signal feed tests for residual promotion v1."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.residual_promotion_collection import canonical_attempt_hash
from bigan.v8.polymarket.residual_promotion_signal_feed import (
    SignalFeedError,
    export_outcome_blind_signal_feed,
)
from bigan.v8.polymarket.residual_promotion_v1 import CANDIDATE_ID, LINEAGE_ID


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _fixture(root: Path) -> list[dict]:
    attempt_id = "promotion-attempt-0001"
    run = root / "captures" / attempt_id
    raw_market = run / "raw/raw_polymarket_markets.jsonl"
    raw_resolution = run / "raw/raw_polymarket_resolutions.jsonl"
    start = 1_800_000_000_000
    market_id = "market-001"
    _write_jsonl(
        raw_market,
        [
            {
                "market_id": market_id,
                "condition_id": "condition-001",
                "slug": "btc-updown-15m-1800000000",
                "market_family": "btc_updown_15m",
                "market_start_ts": start,
                "market_end_ts": start + 900_000,
            }
        ],
    )
    raw_resolution.write_bytes(b"")
    manifest = run / "pending_round_capture_manifest.json"
    report = run / "pending_round_capture_report.json"
    _write_json(
        manifest,
        {
            "run_id": attempt_id,
            "resolution_provider_called": False,
            "wallet_signing_enabled": False,
            "polymarket_write_enabled": False,
            "capital_at_risk": False,
            "raw_artifact_hashes": {
                "raw_polymarket_markets.jsonl": sha256_file(raw_market),
                "raw_polymarket_resolutions.jsonl": sha256_file(raw_resolution),
            },
        },
    )
    _write_json(report, {"resolution_provider_called": False})
    values_a = {"NO_TRADE": 0.0, "BUY_UP_HOLD": -0.01, "BUY_DOWN_HOLD": -0.02}
    values_b = {"NO_TRADE": 0.0, "BUY_UP_HOLD": 0.02, "BUY_DOWN_HOLD": -0.01}
    first = {
        "schema_version": "bigan-btc-15m-residual-promotion-attempt-v1",
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "attempt_index": 1,
        "attempt_id": attempt_id,
        "market_id": market_id,
        "capture_manifest_sha256": sha256_file(manifest),
        "capture_report_sha256": sha256_file(report),
        "quality": {"quality_valid": True, "invalid_reason_codes": []},
        "provider_health": {"provider_failed": False, "retry_used": False},
        "decision_rows": [
            {
                "market_id": market_id,
                "decision_ts": start + 300_000,
                "candidate_action_values": values_a,
                "candidate_selected_action": "NO_TRADE",
                "candidate_accepted_at_this_decision": False,
                "decision_influenced_collection": False,
                "outcomes_accessed": False,
                "settlement_accessed": False,
                "pnl_accessed": False,
                "safety": dict(SAFETY),
            },
            {
                "market_id": market_id,
                "decision_ts": start + 600_000,
                "candidate_action_values": values_b,
                "candidate_selected_action": "BUY_UP_HOLD",
                "candidate_accepted_at_this_decision": True,
                "decision_influenced_collection": False,
                "outcomes_accessed": False,
                "settlement_accessed": False,
                "pnl_accessed": False,
                "safety": dict(SAFETY),
            },
        ],
        "outcomes_accessed": False,
        "settlement_accessed": False,
        "pnl_accessed": False,
        "wallet_signing_allowed": False,
        "polymarket_write_allowed": False,
        "capital_at_risk": False,
        "safety": dict(SAFETY),
        "previous_attempt_hash": "0" * 64,
    }
    first["attempt_hash"] = canonical_attempt_hash(first)
    second = {
        "schema_version": "bigan-btc-15m-residual-promotion-attempt-v1",
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "attempt_index": 2,
        "attempt_id": "promotion-attempt-0002",
        "market_id": None,
        "capture_manifest_sha256": None,
        "capture_report_sha256": None,
        "quality": {
            "quality_valid": False,
            "invalid_reason_codes": ["capture_exception"],
        },
        "provider_health": {"provider_failed": True, "retry_used": False},
        "decision_rows": [],
        "outcomes_accessed": False,
        "settlement_accessed": False,
        "pnl_accessed": False,
        "wallet_signing_allowed": False,
        "polymarket_write_allowed": False,
        "capital_at_risk": False,
        "safety": dict(SAFETY),
        "previous_attempt_hash": first["attempt_hash"],
    }
    second["attempt_hash"] = canonical_attempt_hash(second)
    attempts = [first, second]
    _write_jsonl(root / "outcome_blind_attempts.jsonl", attempts)
    _write_json(
        root / "collection_progress.json",
        {
            "lineage_id": LINEAGE_ID,
            "candidate_id": CANDIDATE_ID,
            "updated_at": "2030-01-01T00:00:00+00:00",
            "authorization_sha256": "a" * 64,
            "collector_protocol_sha256": "b" * 64,
            "candidate_bundle_sha256": "c" * 64,
            "attempts_consumed": 2,
            "quality_valid_market_count": 1,
            "target_quality_valid_market_count": 2500,
            "remaining_quality_valid_markets": 2499,
            "hash_chain_status": "valid",
            "fresh_outcomes_opened": False,
            "interim_pnl_evaluated": False,
            "collection_influenced_by_model_decisions": False,
            "wallet_signing_allowed": False,
            "polymarket_write_allowed": False,
            "capital_at_risk": False,
            "safety": dict(SAFETY),
        },
    )
    return attempts


def _inventory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in root.rglob("*")
        if path.is_file()
    }


def _rehash_ledger(root: Path, attempts: list[dict]) -> None:
    previous = "0" * 64
    for index, attempt in enumerate(attempts, start=1):
        attempt["attempt_index"] = index
        attempt["previous_attempt_hash"] = previous
        attempt["attempt_hash"] = canonical_attempt_hash(attempt)
        previous = attempt["attempt_hash"]
    _write_jsonl(root / "outcome_blind_attempts.jsonl", attempts)


def test_feed_exports_slug_and_exact_two_signals_without_source_mutation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "service"
    output = tmp_path / "monitoring/feed.json"
    _fixture(source)
    before = _inventory(source)
    feed = export_outcome_blind_signal_feed(service_root=source, output_path=output)
    assert _inventory(source) == before
    assert json.loads(output.read_text()) == feed
    assert feed["quality_valid_match_count"] == 1
    assert feed["excluded_attempts"][0]["reason_codes"] == ["capture_exception"]
    match = feed["matches"][0]
    assert match["population_position"] == 1
    assert match["slug"] == "btc-updown-15m-1800000000"
    assert [row["market_age_seconds"] for row in match["signals"]] == [300, 600]
    assert [row["selected_action"] for row in match["signals"]] == [
        "NO_TRADE",
        "BUY_UP_HOLD",
    ]
    assert match["accepted_action"] == "BUY_UP_HOLD"
    assert match["accepted_decision_number"] == 2
    assert feed["raw_signal_distribution"] == {"BUY_UP_HOLD": 1, "NO_TRADE": 1}
    assert feed["accepted_match_signal_distribution"] == {"BUY_UP_HOLD": 1}
    assert feed["population_order"] == (
        "chronological_first_quality_valid_unique_without_reordering"
    )
    assert feed["signal_contract"]["refresh_boundary"] == "after_ledger_close_only"
    assert feed["signal_contract"]["in_progress_attempts_read"] is False
    assert feed["monitoring_influences_collection"] is False
    assert feed["fresh_outcomes_accessed"] is False
    assert feed["outcomes_accessed"] is False
    assert feed["settlement_accessed"] is False
    assert feed["pnl_accessed"] is False
    assert feed["paper_run_started"] is False
    assert feed["safety"] == SAFETY


def test_feed_is_deterministic_and_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "service"
    output = tmp_path / "monitoring/feed.json"
    _fixture(source)
    first = export_outcome_blind_signal_feed(service_root=source, output_path=output)
    first_bytes = output.read_bytes()
    second = export_outcome_blind_signal_feed(service_root=source, output_path=output)
    assert second == first
    assert output.read_bytes() == first_bytes
    identity = dict(second)
    content_sha256 = identity.pop("content_sha256")
    from bigan.v8.polymarket.contracts import canonical_json_sha256

    assert content_sha256 == canonical_json_sha256(identity)


def test_output_inside_service_root_is_forbidden(tmp_path: Path) -> None:
    source = tmp_path / "service"
    _fixture(source)
    with pytest.raises(SignalFeedError, match="disjoint"):
        export_outcome_blind_signal_feed(
            service_root=source,
            output_path=source / "signal-feed.json",
        )


def test_nonempty_resolution_fails_closed_without_reading_it(tmp_path: Path) -> None:
    source = tmp_path / "service"
    _fixture(source)
    resolution = (
        source
        / "captures/promotion-attempt-0001/raw/raw_polymarket_resolutions.jsonl"
    )
    resolution.write_bytes(b"opaque-nonempty-byte")
    with pytest.raises(SignalFeedError, match="non-empty resolution"):
        export_outcome_blind_signal_feed(
            service_root=source,
            output_path=tmp_path / "feed.json",
        )
    assert not (tmp_path / "feed.json").exists()


def test_outcome_bearing_ledger_field_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "service"
    attempts = _fixture(source)
    attempts[0]["outcome_winner"] = "UP"
    _rehash_ledger(source, attempts)
    with pytest.raises(SignalFeedError, match="stable reconciled"):
        export_outcome_blind_signal_feed(
            service_root=source,
            output_path=tmp_path / "feed.json",
            stable_read_attempts=1,
        )


def test_valid_market_requires_exact_frozen_two_signal_schedule(tmp_path: Path) -> None:
    source = tmp_path / "service"
    attempts = _fixture(source)
    attempts[0]["decision_rows"] = attempts[0]["decision_rows"][:1]
    _rehash_ledger(source, attempts)
    with pytest.raises(SignalFeedError, match="exactly two"):
        export_outcome_blind_signal_feed(
            service_root=source,
            output_path=tmp_path / "feed.json",
        )


def test_first_non_no_trade_acceptance_semantics_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "service"
    attempts = _fixture(source)
    attempts[0]["decision_rows"][1]["candidate_accepted_at_this_decision"] = False
    _rehash_ledger(source, attempts)
    with pytest.raises(SignalFeedError, match="acceptance semantics"):
        export_outcome_blind_signal_feed(
            service_root=source,
            output_path=tmp_path / "feed.json",
        )


def test_capture_hash_drift_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "service"
    _fixture(source)
    report = source / "captures/promotion-attempt-0001/pending_round_capture_report.json"
    report.write_text("{}\n")
    with pytest.raises(SignalFeedError, match="capture hash mismatch"):
        export_outcome_blind_signal_feed(
            service_root=source,
            output_path=tmp_path / "feed.json",
        )


def test_feed_does_not_mutate_passed_decision_objects(tmp_path: Path) -> None:
    source = tmp_path / "service"
    attempts = _fixture(source)
    before = copy.deepcopy(attempts)
    export_outcome_blind_signal_feed(
        service_root=source,
        output_path=tmp_path / "feed.json",
    )
    assert attempts == before
