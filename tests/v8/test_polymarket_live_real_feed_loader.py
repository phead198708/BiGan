"""Real live feed loader contract tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import bigan.v8.polymarket.live.real_feed_loader as real_feed_loader
import bigan.v8.polymarket.recorder.public_provider as public_provider
from bigan.v8.polymarket.live.contracts import BinanceBTCCandle, PolymarketLivePaperConfig


def test_partial_resolution_rows_do_not_emit_zero_price_candles(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        public_provider,
        "PolymarketPublicHTTPRealCorpusProvider",
        _PartialResolutionProvider,
    )
    monkeypatch.setattr(
        real_feed_loader,
        "_fetch_json",
        lambda _url: {"bid": "65000.0", "ask": "65001.0"},
    )
    monotonic_values = iter((0.0, 0.1, 0.2, 2.0, 2.1))
    monkeypatch.setattr(
        real_feed_loader.time,
        "monotonic",
        lambda: next(monotonic_values, 2.1),
    )
    monkeypatch.setattr(real_feed_loader.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(real_feed_loader.time, "time", lambda: 1_782_369_000.0)

    _market_rows, _orderbook_rows, _trade_rows, tick_rows, candle_rows = (
        real_feed_loader.load_real_live_feed_rows(
            PolymarketLivePaperConfig(
                run_id="partial-resolution",
                output_dir=tmp_path,
                mock_live=False,
                market_families=("btc_updown_5m",),
                duration_seconds=1,
                poll_interval_seconds=1,
                settlement_mode="delayed",
            )
        )
    )

    assert tick_rows
    assert len(candle_rows) == 1
    candle = BinanceBTCCandle(**candle_rows[0])
    assert candle.open_price > 0.0
    assert candle.close_price > 0.0
    assert candle.high_price > 0.0
    assert candle.low_price > 0.0
    assert candle.source == "binance_btcusdt"


class _PartialResolutionProvider:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def market_rows(self, _config: Any) -> list[dict[str, Any]]:
        return [_market_row()]

    def orderbook_rows(
        self,
        _markets: list[dict[str, Any]],
        _config: Any,
    ) -> list[dict[str, Any]]:
        market = _market_row()
        return [
            {
                "market_id": market["market_id"],
                "token_id": market["up_token_id"],
                "outcome": "UP",
                "ts": market["market_start_ts"],
                "available_at_ts": market["market_start_ts"] + 1_000,
                "bid_price": 0.49,
                "ask_price": 0.51,
                "mid_price": 0.50,
                "bid_size": 100.0,
                "ask_size": 100.0,
                "liquidity_depth": 200.0,
            },
            {
                "market_id": market["market_id"],
                "token_id": market["down_token_id"],
                "outcome": "DOWN",
                "ts": market["market_start_ts"],
                "available_at_ts": market["market_start_ts"] + 1_000,
                "bid_price": 0.49,
                "ask_price": 0.51,
                "mid_price": 0.50,
                "bid_size": 100.0,
                "ask_size": 100.0,
                "liquidity_depth": 200.0,
            },
        ]

    def trade_rows(
        self,
        _markets: list[dict[str, Any]],
        _config: Any,
    ) -> list[dict[str, Any]]:
        return []

    def resolution_rows(
        self,
        _markets: list[dict[str, Any]],
        _config: Any,
    ) -> list[dict[str, Any]]:
        return [
            {
                "market_id": _market_row()["market_id"],
                "reference_price_source": "https://data.chain.link/streams/btc-usd",
            }
        ]


def _market_row() -> dict[str, Any]:
    start_ts = 1_782_369_000_000
    return {
        "market_id": "0xpartial",
        "condition_id": "0xcondition",
        "slug": "btc-up-or-down-partial",
        "market_family": "btc_updown_5m",
        "horizon_ms": 300_000,
        "market_start_ts": start_ts,
        "market_end_ts": start_ts + 300_000,
        "settlement_ts": start_ts + 360_000,
        "up_token_id": "up-token",
        "down_token_id": "down-token",
        "reference_price_source": "https://data.chain.link/streams/btc-usd",
        "settlement_rule": "UP wins if BTC closes above the price to beat.",
        "reference_price_at_start": 65_000.0,
        "raw_market_sha256": "a" * 64,
    }
