"""Operator tests for the Polymarket real corpus recorder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256, looks_like_sha256
from bigan.v8.polymarket.corpus import RAW_CORPUS_FILENAMES
from bigan.v8.polymarket.recorder import (
    PolymarketRealCorpusRecorderConfig,
    record_polymarket_real_corpus,
)
from bigan.v8.polymarket.recorder.btc_reference import mock_btc_feature_candle_rows
from bigan.v8.polymarket.recorder.market_discovery import discover_mock_market_rows
from bigan.v8.polymarket.recorder.orderbook_state import (
    mock_orderbook_rows,
    mock_trade_rows,
    sample_times_for_market,
    validate_market_books,
)
from bigan.v8.polymarket.recorder.resolution import mock_resolution_rows


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
    assert result.report["live_polymarket_data_read"] is False
    assert result.report["live_btc_reference_data_read"] is False
    assert result.report["live_polymarket_data"] is False
    assert result.report["live_btc_reference_data"] is False
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
    assert result.report["live_polymarket_data_read"] is False
    assert result.report["live_btc_reference_data_read"] is False
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


def test_non_mock_public_collection_can_complete_with_configured_provider(
    tmp_path: Path,
) -> None:
    result = record_polymarket_real_corpus(
        PolymarketRealCorpusRecorderConfig(
            run_id="real-provider-happy",
            output_dir=tmp_path,
            mock_public_data=False,
        ),
        public_provider=FakeRealPublicProvider(),
    )

    assert result.report["training_eligible"] is True
    assert result.report["phase2_corpus_build_eligible"] is True
    assert result.report["real_historical_training_eligible"] is True
    assert result.report["manual_live_evidence_eligible"] is True
    assert result.report["phase2_corpus_built"] is True
    assert result.report["public_collection_status"] == "completed"
    assert result.report["public_collection_reason_codes"] == []
    assert result.report["live_polymarket_data_read"] is True
    assert result.report["live_btc_reference_data_read"] is True
    assert result.report["live_polymarket_data"] is True
    assert result.report["live_btc_reference_data"] is True
    assert result.report["mock_public_data_used"] is False
    assert result.report["synthetic_public_data_used"] is False
    assert result.report["synthetic_corpus_used"] is False
    assert result.report["real_historical_corpus_used"] is True
    assert result.report["deterministic_replay"] is False
    assert result.report["raw_polymarket_market_count"] == 3
    assert result.report["raw_orderbook_row_count"] == 24
    assert result.report["raw_trade_row_count"] == 6
    assert result.report["raw_resolution_count"] == 3
    assert result.report["raw_btc_candle_row_count"] > 0
    assert result.report["rejected_row_count"] == 0
    assert result.phase2_result is not None
    assert looks_like_sha256(result.report["phase2_corpus_manifest_sha256"])

    recorder_manifest = _read_json(result.artifact_paths["real_corpus_recorder_manifest"])
    assert recorder_manifest["real_historical_corpus_used"] is True
    assert recorder_manifest["mock_public_data_used"] is False
    assert recorder_manifest["synthetic_corpus_used"] is False


def test_non_mock_public_collection_accepts_causal_off_grid_orderbook_snapshots(
    tmp_path: Path,
) -> None:
    result = record_polymarket_real_corpus(
        PolymarketRealCorpusRecorderConfig(
            run_id="real-provider-asof-books",
            output_dir=tmp_path,
            market_families=("btc_updown_5m",),
            mock_public_data=False,
        ),
        public_provider=OffGridOrderbookProvider(),
    )

    assert result.report["training_eligible"] is True
    assert result.report["phase2_corpus_built"] is True
    assert result.report["raw_polymarket_market_count"] == 1
    assert result.report["raw_orderbook_row_count"] == 10
    assert result.report["reject_reason_counts"] == {}
    features = _read_jsonl(result.phase2_result.output_dir / "polymarket_feature_rows.jsonl")
    assert len(features) == 5
    assert all(row["max_input_ts"] <= row["decision_ts"] for row in features)
    assert all(row["available_at_ts"] <= row["decision_ts"] for row in features)


def test_orderbook_validation_rejects_future_only_snapshots() -> None:
    config = PolymarketRealCorpusRecorderConfig(
        run_id="future-books",
        output_dir="/tmp/future-books",
        market_families=("btc_updown_5m",),
    )
    market = discover_mock_market_rows(config)[0]
    first_sample = sample_times_for_market(market, config)[0]
    rows = []
    for outcome, token_id, mid in (
        ("UP", market["up_token_id"], 0.56),
        ("DOWN", market["down_token_id"], 0.44),
    ):
        rows.append(
            {
                "market_id": market["market_id"],
                "token_id": token_id,
                "outcome": outcome,
                "ts": first_sample + 1_000,
                "available_at_ts": first_sample + 1_000,
                "bid_price": mid - 0.01,
                "ask_price": mid + 0.01,
                "mid_price": mid,
                "bid_size": 100.0,
                "ask_size": 100.0,
                "liquidity_depth": 200.0,
            }
        )

    valid, reasons = validate_market_books(market=market, book_rows=rows, config=config)

    assert valid == []
    assert reasons == ["missing_complete_up_down_orderbook"]


def test_non_mock_public_collection_provider_error_fails_closed(
    tmp_path: Path,
) -> None:
    result = record_polymarket_real_corpus(
        PolymarketRealCorpusRecorderConfig(
            run_id="real-provider-clob-failure",
            output_dir=tmp_path,
            mock_public_data=False,
        ),
        public_provider=FailingOrderbookProvider(),
    )

    assert result.report["training_eligible"] is False
    assert result.report["phase2_corpus_build_eligible"] is False
    assert result.report["real_historical_training_eligible"] is False
    assert result.report["manual_live_evidence_eligible"] is False
    assert result.report["phase2_corpus_built"] is False
    assert result.report["public_collection_status"] == "blocked_fail_closed"
    assert result.report["public_collection_reason_codes"] == [
        "real_public_collection_provider_error"
    ]
    assert result.report["live_polymarket_data_read"] is False
    assert result.report["live_btc_reference_data_read"] is False
    assert result.report["real_historical_corpus_used"] is False
    assert result.report["raw_polymarket_market_count"] == 0
    assert result.report["reject_reason_counts"]["real_public_collection_provider_error"] == 1
    assert result.report["reject_reason_counts"]["missing_complete_up_down_orderbook"] == 3

    rejected = _read_jsonl(result.artifact_paths["real_corpus_rejected_rows"])
    assert any(
        row.get("provider") == "polymarket_clob"
        and row.get("provider_stage") == "orderbook_collection"
        and row.get("reject_reasons") == ["real_public_collection_provider_error"]
        for row in rejected
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class FakeRealPublicProvider:
    read_only = True
    write_capable = False
    paper_only = True
    capital_at_risk = False
    broker_exchange_write_enabled = False
    live_exchange_write_enabled = False
    polymarket_write_enabled = False
    wallet_signing_enabled = False

    def market_rows(
        self,
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        return [_as_real_public_market_row(row) for row in discover_mock_market_rows(config)]

    def orderbook_rows(
        self,
        markets: list[dict[str, Any]],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        return mock_orderbook_rows(markets, config)

    def trade_rows(
        self,
        markets: list[dict[str, Any]],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        del config
        return mock_trade_rows(markets)

    def btc_feature_candle_rows(
        self,
        markets: list[dict[str, Any]],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        return mock_btc_feature_candle_rows(markets, config)

    def resolution_rows(
        self,
        markets: list[dict[str, Any]],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        return mock_resolution_rows(markets, config)


class FailingOrderbookProvider(FakeRealPublicProvider):
    def orderbook_rows(
        self,
        markets: list[dict[str, Any]],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        del markets, config
        raise RuntimeError("CLOB orderbook endpoint unavailable")


class OffGridOrderbookProvider(FakeRealPublicProvider):
    def orderbook_rows(
        self,
        markets: list[dict[str, Any]],
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for market in markets[:1]:
            for sample_index, decision_ts in enumerate(sample_times_for_market(market, config)):
                book_ts = decision_ts if sample_index == 0 else decision_ts - 1_000
                for outcome, token_id, mid in (
                    ("UP", market["up_token_id"], 0.56 + sample_index * 0.01),
                    ("DOWN", market["down_token_id"], 0.44 - sample_index * 0.01),
                ):
                    rows.append(
                        {
                            "market_id": market["market_id"],
                            "token_id": token_id,
                            "outcome": outcome,
                            "ts": book_ts,
                            "available_at_ts": book_ts,
                            "bid_price": mid - 0.01,
                            "ask_price": mid + 0.01,
                            "mid_price": mid,
                            "bid_size": 100.0,
                            "ask_size": 100.0,
                            "liquidity_depth": 200.0,
                        }
                    )
        return rows


def _as_real_public_market_row(row: dict[str, Any]) -> dict[str, Any]:
    market = dict(row)
    market["raw_market_sha256"] = canonical_json_sha256(
        {
            "market_id": market["market_id"],
            "family": market["market_family"],
            "source": "fake_real_public_provider",
        }
    )
    market["raw_public_payload"] = {
        "mock_public_data": False,
        "provider": "fake_real_public_provider",
    }
    return market
