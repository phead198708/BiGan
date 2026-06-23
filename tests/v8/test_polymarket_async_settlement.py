"""Async settlement tests for round-scoped Polymarket corpus collection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.recorder import (
    PolymarketRealCorpusRecorderConfig,
    capture_polymarket_pending_round,
    finalize_polymarket_pending_round,
)
from bigan.v8.polymarket.recorder.btc_reference import mock_btc_feature_candle_rows
from bigan.v8.polymarket.recorder.market_discovery import discover_mock_market_rows
from bigan.v8.polymarket.recorder.orderbook_state import mock_orderbook_rows, mock_trade_rows


def test_pending_capture_does_not_call_resolution_or_export(tmp_path: Path) -> None:
    provider = AsyncSettlementFakeProvider(resolved=False)
    config = PolymarketRealCorpusRecorderConfig(
        run_id="pending-round",
        output_dir=tmp_path,
        market_families=("btc_updown_5m",),
        mock_public_data=False,
    )

    result = capture_polymarket_pending_round(config, public_provider=provider)

    assert provider.resolution_calls == 0
    assert result.report["capture_status"] == "pending_resolution"
    assert result.report["pending_resolution"] is True
    assert result.report["resolution_provider_called"] is False
    assert result.report["raw_polymarket_market_count"] == 1
    assert result.report["raw_orderbook_row_count"] > 0
    assert result.report["raw_resolution_count"] == 0
    assert result.report["training_eligible"] is False
    assert result.report["exported_training_corpus_dir"] is None
    assert (result.raw_dir / "raw_polymarket_resolutions.jsonl").read_text() == ""


def test_pending_finalization_waits_for_resolution_before_export(tmp_path: Path) -> None:
    provider = AsyncSettlementFakeProvider(resolved=False)
    config = PolymarketRealCorpusRecorderConfig(
        run_id="pending-round",
        output_dir=tmp_path,
        market_families=("btc_updown_5m",),
        mock_public_data=False,
    )
    capture = capture_polymarket_pending_round(config, public_provider=provider)

    unresolved = finalize_polymarket_pending_round(
        capture.run_dir,
        public_provider=provider,
        destination_root=tmp_path / "training_root",
    )

    assert provider.resolution_calls == 1
    assert unresolved.report["finalization_status"] == "pending_resolution"
    assert unresolved.report["pending_resolution"] is True
    assert unresolved.report["training_eligible"] is False
    assert unresolved.exported_training_corpus_dir is None
    assert unresolved.report["reject_reason_counts"] == {"missing_resolution": 1}

    provider.resolved = True
    finalized = finalize_polymarket_pending_round(
        capture.run_dir,
        public_provider=provider,
        destination_root=tmp_path / "training_root",
        overwrite_existing=True,
    )

    assert provider.resolution_calls == 2
    assert finalized.report["finalization_status"] == "exported"
    assert finalized.report["pending_resolution"] is False
    assert finalized.report["training_eligible"] is True
    assert finalized.report["raw_resolution_count"] == 1
    assert finalized.exported_training_corpus_dir is not None
    assert finalized.exported_training_corpus_dir.name.startswith("btc-updown-5m-")
    assert (finalized.exported_training_corpus_dir / "polymarket_label_rows.jsonl").exists()
    provenance = finalized.exported_training_corpus_dir / "training_corpus_provenance.json"
    assert '"round_scoped_export": true' in provenance.read_text(encoding="utf-8")


class AsyncSettlementFakeProvider:
    read_only = True
    write_capable = False
    paper_only = True
    capital_at_risk = False
    broker_exchange_write_enabled = False
    live_exchange_write_enabled = False
    polymarket_write_enabled = False
    wallet_signing_enabled = False

    def __init__(self, *, resolved: bool) -> None:
        self.resolved = resolved
        self.resolution_calls = 0

    def market_rows(
        self,
        config: PolymarketRealCorpusRecorderConfig,
    ) -> list[dict[str, Any]]:
        return [_as_real_public_market_row(row) for row in discover_mock_market_rows(config)[:1]]

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
        del config
        self.resolution_calls += 1
        if not self.resolved:
            return []
        return [
            {
                "market_id": market["market_id"],
                "reference_price_start": 65000.0,
                "reference_price_end": 65025.0,
                "reference_price_source": market["reference_price_source"],
                "resolution_status": "normal",
                "raw_resolution_text": "Resolved from official test reference.",
                "paper_only": True,
                "capital_at_risk": False,
                "broker_exchange_write_enabled": False,
                "live_exchange_write_enabled": False,
                "polymarket_write_enabled": False,
                "wallet_signing_enabled": False,
            }
            for market in markets
        ]


def _as_real_public_market_row(row: dict[str, Any]) -> dict[str, Any]:
    market = dict(row)
    market["raw_market_sha256"] = canonical_json_sha256(
        {
            "market_id": market["market_id"],
            "family": market["market_family"],
            "source": "async_settlement_fake_provider",
        }
    )
    market["raw_public_payload"] = {"mock_public_data": False}
    return market
