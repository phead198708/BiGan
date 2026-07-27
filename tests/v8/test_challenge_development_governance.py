from __future__ import annotations

import json
from pathlib import Path

from bigan.v8.polymarket.challenge_development_governance import (
    audit_capture,
    build_lane_health_summary,
    build_training_readiness,
    run_transfer_diagnostic_if_ready,
    validate_training_protocol,
    validate_transfer_protocol,
)
from bigan.v8.polymarket.challenge_development_lane import SAFETY, sha256_file

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "examples" / "v8" / "polymarket_configs"
TRANSFER = CONFIG / "challenge_model_15m_transfer_diagnostic_protocol.json"
TRAINING = CONFIG / "challenge_model_15m_training_protocol_preregistration.json"


def test_frozen_protocols_preserve_development_only_gates() -> None:
    transfer = json.loads(TRANSFER.read_text(encoding="utf-8"))
    training = json.loads(TRAINING.read_text(encoding="utf-8"))
    validate_transfer_protocol(transfer, verify_artifact_bytes=False)
    validate_training_protocol(training)
    assert transfer["promotion_evidence_eligible"] is False
    assert training["readiness_gate"]["minimum_quality_valid_outcome_finalized_market_count"] == 120
    assert training["representation"]["missing_encoded_as_numeric_zero_allowed"] is False
    assert training["safety"] == SAFETY


def test_capture_audit_counts_true_paired_asks_and_causal_streams(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    corpus = run_dir / "phase2_corpus"
    corpus.mkdir(parents=True)
    rows = [_feature_row(1000), _feature_row(2000)]
    _write_jsonl(corpus / "polymarket_feature_rows.jsonl", rows)
    (run_dir / "pending_round_capture_report.json").write_text(
        json.dumps(
            {
                "resolution_provider_called": False,
                "paper_only": True,
                "capital_at_risk": False,
                "polymarket_write_enabled": False,
                "wallet_signing_enabled": False,
                "live_exchange_write_enabled": False,
                "broker_exchange_write_enabled": False,
            }
        ),
        encoding="utf-8",
    )
    audit = audit_capture(_capture(run_dir))
    assert audit["quality_valid"] is True
    assert audit["paired_executable_ask_decision_count"] == 2
    assert audit["book_causal_complete_decision_count"] == 2
    assert audit["chainlink_causal_complete_decision_count"] == 2
    assert audit["trade_tape_causal_complete_decision_count"] == 2

    rows[0]["features"]["down_ask"] = None
    _write_jsonl(corpus / "polymarket_feature_rows.jsonl", rows)
    failed = audit_capture(_capture(run_dir))
    assert failed["quality_valid"] is False
    assert "paired_executable_ask_coverage_failed" in failed["exclusion_reason_codes"]


def test_health_and_training_gate_remain_closed_below_threshold(
    tmp_path: Path,
) -> None:
    lane = tmp_path / "lane"
    run_dir = lane / "capture"
    corpus = run_dir / "phase2_corpus"
    corpus.mkdir(parents=True)
    _write_jsonl(
        corpus / "polymarket_feature_rows.jsonl",
        [_feature_row(1000), _feature_row(2000)],
    )
    (run_dir / "pending_round_capture_report.json").write_text(
        json.dumps(
            {
                "resolution_provider_called": False,
                "paper_only": True,
                "capital_at_risk": False,
                "polymarket_write_enabled": False,
                "wallet_signing_enabled": False,
                "live_exchange_write_enabled": False,
                "broker_exchange_write_enabled": False,
            }
        ),
        encoding="utf-8",
    )
    batch_summary = lane / "batch_summary.json"
    batch_summary.write_text(
        json.dumps({"captures": [_capture(run_dir)]}),
        encoding="utf-8",
    )
    _write_jsonl(
        lane / "outcome_blind_capture_batch_index.jsonl",
        [
            {
                "batch_summary_path": str(batch_summary),
                "collected_at": "2026-07-27T00:00:00+00:00",
            }
        ],
    )
    manifest = lane / "development_corpus" / "polymarket_corpus_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    _write_jsonl(
        lane / "finalized_development_corpus_index.jsonl",
        [
            {
                "run_id": "run-1",
                "exported_corpus_manifest_path": str(manifest),
            }
        ],
    )
    health = build_lane_health_summary(
        lane_root=lane,
        date_utc="2026-07-27",
        write=False,
    )
    assert health["cumulative"]["attempted_market_count"] == 1
    assert health["cumulative"]["quality_valid_outcome_finalized_market_count"] == 1
    assert health["cumulative"]["paired_up_down_executable_ask"]["coverage"] == 1.0

    readiness = build_training_readiness(
        lane_root=lane,
        training_protocol_path=TRAINING,
        expected_training_protocol_sha256=sha256_file(TRAINING),
        transfer_protocol_path=TRANSFER,
        expected_transfer_protocol_sha256=sha256_file(TRANSFER),
        write=False,
    )
    assert readiness["training_start_allowed"] is False
    assert readiness["attempt_120_authorized"] is False
    assert (
        "quality_valid_outcome_finalized_market_count_at_least_120"
        in readiness["blocking_reason_codes"]
    )
    waiting = run_transfer_diagnostic_if_ready(
        lane_root=lane,
        protocol_path=TRANSFER,
        expected_protocol_sha256=sha256_file(TRANSFER),
    )
    assert waiting["transfer_diagnostic_started"] is False
    assert waiting["required_market_count"] == 40


def _capture(run_dir: Path) -> dict:
    return {
        "run_id": "run-1",
        "run_dir": str(run_dir),
        "round_index": 1,
        "scheduled_round_start_ts": 1,
        "capture_start_boundary_validation_passed": True,
        "raw_polymarket_market_count": 1,
        "provider_raw_orderbook_snapshot_count": 100,
        "orderbook_full_window_coverage_passed": True,
        "raw_btc_candle_row_count": 10,
        "raw_chainlink_price_row_count": 10,
        "chainlink_rtds_price_stream_fresh": True,
        "reject_reason_counts": {},
    }


def _feature_row(decision_ts: int) -> dict:
    provenance = {
        name: {
            "available_at_ts": decision_ts,
            "input_end_ts": decision_ts,
        }
        for name in ("up_ask", "down_ask")
    }
    provenance.update(
        {
            name: {
                "available_at_ts": decision_ts,
                "max_input_ts": decision_ts,
                "provenance_valid": True,
            }
            for name in (
                "chainlink_price_at_decision",
                "chainlink_reference_price_at_market_start",
            )
        }
    )
    return {
        "market_id": "market-1",
        "decision_ts": decision_ts,
        "available_at_ts": decision_ts,
        "max_input_ts": decision_ts,
        "features": {
            "up_ask": 0.51,
            "down_ask": 0.50,
            "recent_trade_volume_coverage_complete": 1,
            "trade_tape_available_at_ts": decision_ts,
            "trade_tape_max_causal_input_ts": decision_ts,
            "trade_tape_provider_timeout": 0,
            "trade_tape_truncated": 0,
            "trade_tape_censored": 0,
            "trade_tape_historical_backfill": 0,
        },
        "feature_provenance": provenance,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
