"""Operator tests for the Polymarket real corpus recorder."""

from __future__ import annotations

import json
from pathlib import Path

from bigan.v8.polymarket.contracts import looks_like_sha256
from bigan.v8.polymarket.corpus import RAW_CORPUS_FILENAMES
from bigan.v8.polymarket.recorder import (
    PolymarketRealCorpusRecorderConfig,
    record_polymarket_real_corpus,
)


def test_recorder_writes_raw_files_manifest_and_phase2_corpus(tmp_path: Path) -> None:
    result = record_polymarket_real_corpus(
        PolymarketRealCorpusRecorderConfig(
            run_id="happy",
            output_dir=tmp_path,
        )
    )

    assert result.report["training_eligible"] is False
    assert result.report["phase2_corpus_build_eligible"] is True
    assert result.report["real_historical_training_eligible"] is False
    assert result.report["manual_live_evidence_eligible"] is False
    assert result.report["phase2_corpus_built"] is True
    assert result.report["raw_polymarket_market_count"] == 3
    assert result.report["raw_orderbook_row_count"] == 24
    assert result.report["raw_trade_row_count"] == 6
    assert result.report["raw_resolution_count"] == 3
    assert result.report["raw_btc_candle_row_count"] > 0
    assert result.report["rejected_row_count"] == 0
    assert result.report["deterministic_replay"] is True
    assert result.report["mock_public_data_used"] is True
    assert result.report["synthetic_public_data_used"] is True
    assert result.report["synthetic_corpus_used"] is True
    assert result.report["real_historical_corpus_used"] is False
    assert result.report["fixture_corpus_used"] is False
    assert result.report["requested_live_public_collection"] is False
    assert result.report["public_collection_status"] == "mocked"
    assert result.report["public_collection_reason_codes"] == []
    assert result.report["paper_only"] is True
    assert result.report["capital_at_risk"] is False
    assert result.phase2_result is not None

    for filename in RAW_CORPUS_FILENAMES:
        assert (result.raw_dir / filename).exists(), filename
    for digest in result.manifest["raw_artifact_hashes"].values():
        assert looks_like_sha256(digest)
    assert looks_like_sha256(result.report["phase2_corpus_manifest_sha256"])
    assert (
        result.manifest["phase2_corpus_manifest_sha256"]
        == result.report["phase2_corpus_manifest_sha256"]
    )

    markets = _read_jsonl(result.raw_dir / "raw_polymarket_markets.jsonl")
    assert {row["market_family"] for row in markets} == {
        "btc_updown_5m",
        "btc_updown_15m",
        "btc_updown_1h",
    }
    assert {row["reference_price_source"] for row in markets} == {
        "polymarket_official_btc_usd_reference"
    }

    normalized_markets = _read_jsonl(
        result.phase2_result.output_dir / "polymarket_market_metadata.jsonl"
    )
    assert len(normalized_markets) == 3


def test_recorder_fail_closes_on_missing_complete_up_down_book(tmp_path: Path) -> None:
    result = record_polymarket_real_corpus(
        PolymarketRealCorpusRecorderConfig(
            run_id="missing-book",
            output_dir=tmp_path,
            market_families=("btc_updown_5m",),
            inject_missing_down_book=True,
        )
    )

    assert result.report["training_eligible"] is False
    assert result.report["phase2_corpus_build_eligible"] is False
    assert result.report["real_historical_training_eligible"] is False
    assert result.report["manual_live_evidence_eligible"] is False
    assert result.report["phase2_corpus_built"] is False
    assert result.report["raw_polymarket_market_count"] == 0
    assert result.report["raw_orderbook_row_count"] == 0
    assert result.report["reject_reason_counts"]["missing_complete_up_down_orderbook"] == 1
    rejected = _read_jsonl(result.artifact_paths["real_corpus_rejected_rows"])
    assert rejected[0]["reject_reasons"] == ["missing_complete_up_down_orderbook"]


def test_recorder_fail_closes_on_missing_official_settlement_source(tmp_path: Path) -> None:
    result = record_polymarket_real_corpus(
        PolymarketRealCorpusRecorderConfig(
            run_id="missing-source",
            output_dir=tmp_path,
            market_families=("btc_updown_5m",),
            inject_missing_reference_source=True,
        )
    )

    assert result.report["training_eligible"] is False
    assert result.report["phase2_corpus_build_eligible"] is False
    assert result.report["real_historical_training_eligible"] is False
    assert result.report["manual_live_evidence_eligible"] is False
    assert result.report["phase2_corpus_built"] is False
    assert result.report["raw_polymarket_market_count"] == 0
    assert result.report["reject_reason_counts"]["missing_verified_resolution_source"] == 1
    rejected = _read_jsonl(result.artifact_paths["real_corpus_rejected_rows"])
    assert "missing_verified_resolution_source" in rejected[0]["reject_reasons"]


def test_recorder_fail_closes_on_unknown_token_book(tmp_path: Path) -> None:
    result = record_polymarket_real_corpus(
        PolymarketRealCorpusRecorderConfig(
            run_id="unknown-token",
            output_dir=tmp_path,
            market_families=("btc_updown_5m",),
            inject_unknown_token_book=True,
        )
    )

    assert result.report["training_eligible"] is False
    assert result.report["phase2_corpus_build_eligible"] is False
    assert result.report["real_historical_training_eligible"] is False
    assert result.report["manual_live_evidence_eligible"] is False
    assert result.report["raw_polymarket_market_count"] == 0
    assert result.report["reject_reason_counts"]["unknown_token_id"] == 1
    assert result.report["reject_reason_counts"]["missing_complete_up_down_orderbook"] == 1


def test_non_mock_public_collection_fails_closed_with_provider_reasons(
    tmp_path: Path,
) -> None:
    result = record_polymarket_real_corpus(
        PolymarketRealCorpusRecorderConfig(
            run_id="real-not-wired",
            output_dir=tmp_path,
            mock_public_data=False,
        )
    )

    assert result.report["training_eligible"] is False
    assert result.report["phase2_corpus_build_eligible"] is False
    assert result.report["real_historical_training_eligible"] is False
    assert result.report["manual_live_evidence_eligible"] is False
    assert result.report["phase2_corpus_built"] is False
    assert result.report["requested_live_public_collection"] is True
    assert result.report["public_collection_status"] == "blocked_fail_closed"
    assert result.report["public_collection_reason_codes"] == [
        "real_public_collection_not_configured"
    ]
    assert result.report["live_polymarket_data"] is False
    assert result.report["live_btc_reference_data"] is False
    assert result.report["mock_public_data_used"] is False
    assert result.report["synthetic_public_data_used"] is False
    assert result.report["synthetic_corpus_used"] is False
    assert result.report["real_historical_corpus_used"] is False
    assert result.report["raw_polymarket_market_count"] == 0
    assert result.report["raw_orderbook_row_count"] == 0
    assert result.report["raw_trade_row_count"] == 0
    assert result.report["raw_btc_candle_row_count"] == 0
    assert result.report["raw_resolution_count"] == 0
    assert result.report["rejected_row_count"] == 4
    assert result.report["reject_reason_counts"]["real_public_collection_not_configured"] == 4

    rejected = _read_jsonl(result.artifact_paths["real_corpus_rejected_rows"])
    assert {row["provider"] for row in rejected} == {
        "polymarket_gamma",
        "polymarket_clob",
        "btc_reference",
        "polymarket_resolution",
    }
    for filename in RAW_CORPUS_FILENAMES:
        assert (result.raw_dir / filename).exists()
        assert _read_jsonl(result.raw_dir / filename) == []


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
