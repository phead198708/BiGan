"""Binance BTCUSDT read-only reference feeds for Polymarket live paper runs."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Any

from bigan.v8.polymarket.live.contracts import (
    BinanceBTCCandle,
    BinanceBTCReferenceTick,
    PolymarketLiveMarket,
    PolymarketLivePaperConfig,
    PolymarketLivePaperError,
)


class MockBinanceBTCReferenceFeed:
    """Deterministic BTC reference feed for mocked-live tests."""

    read_only = True
    write_capable = False
    paper_only = True
    capital_at_risk = False
    broker_exchange_write_enabled = False
    live_exchange_write_enabled = False
    polymarket_write_enabled = False
    wallet_signing_enabled = False

    def __init__(self, config: PolymarketLivePaperConfig) -> None:
        self.config = config

    def tick_rows(self, markets: tuple[PolymarketLiveMarket, ...]) -> list[dict[str, Any]]:
        rows = []
        for market_index, market in enumerate(markets):
            step_ms = max(60_000, market.horizon_ms // 3)
            for sample_index in range(3):
                ts = min(
                    market.market_end_ts - 1_000,
                    market.market_start_ts + sample_index * step_ms,
                )
                received_ts = ts + 1_000
                if self.config.inject_stale_reference and market_index == 0 and sample_index == 0:
                    received_ts = ts + (self.config.max_stale_reference_seconds + 5) * 1000
                mid = market.reference_price_at_start + sample_index * 12.5 + market_index * 4.0
                rows.append(
                    {
                        "ts": ts,
                        "received_ts": received_ts,
                        "bid_price": mid - 0.5,
                        "ask_price": mid + 0.5,
                        "mid_price": mid,
                        "last_price": mid,
                        **_readonly_flags(),
                    }
                )
        return rows

    def ticks(
        self,
        markets: tuple[PolymarketLiveMarket, ...],
    ) -> tuple[BinanceBTCReferenceTick, ...]:
        return tuple(BinanceBTCReferenceTick(**row) for row in self.tick_rows(markets))

    def candle_rows(self, markets: tuple[PolymarketLiveMarket, ...]) -> list[dict[str, Any]]:
        rows = []
        for market_index, market in enumerate(markets):
            close_delta = 24.0 + market_index * 3.0
            rows.append(
                {
                    "market_id": market.market_id,
                    "market_family": market.market_family,
                    "open_ts": market.market_start_ts,
                    "close_ts": market.market_end_ts,
                    "open_price": market.reference_price_at_start,
                    "close_price": market.reference_price_at_start + close_delta,
                    "high_price": market.reference_price_at_start + close_delta + 4.0,
                    "low_price": market.reference_price_at_start - 3.0,
                    **_readonly_flags(),
                }
            )
        return rows

    def candles(
        self,
        markets: tuple[PolymarketLiveMarket, ...],
    ) -> tuple[BinanceBTCCandle, ...]:
        return tuple(BinanceBTCCandle(**row) for row in self.candle_rows(markets))


class BinanceBTCReferenceHTTPFeed:
    """Public Binance read-only reference feed shell."""

    read_only = True
    write_capable = False
    paper_only = True
    capital_at_risk = False
    broker_exchange_write_enabled = False
    live_exchange_write_enabled = False
    polymarket_write_enabled = False
    wallet_signing_enabled = False

    def __init__(
        self,
        *,
        ticker_endpoint: str = "https://api.binance.com/api/v3/ticker/bookTicker",
        klines_endpoint: str = "https://api.binance.com/api/v3/klines",
        timeout_seconds: float = 10.0,
    ) -> None:
        self.ticker_endpoint = ticker_endpoint
        self.klines_endpoint = klines_endpoint
        self.timeout_seconds = timeout_seconds

    def fetch_json(self, url: str) -> Mapping[str, Any] | list[Any]:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "bigan-v8-binance-reference-readonly/1.0"},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def fetch_tick_payload(self, *, symbol: str = "BTCUSDT") -> dict[str, Any]:
        params = urllib.parse.urlencode({"symbol": symbol})
        payload = self.fetch_json(f"{self.ticker_endpoint}?{params}")
        if not isinstance(payload, dict):
            raise PolymarketLivePaperError(
                "invalid Binance tick payload",
                reason_codes=("invalid_binance_tick_payload",),
            )
        return dict(payload)

    def fetch_klines_payload(
        self,
        *,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
        limit: int = 2,
    ) -> list[Any]:
        params = urllib.parse.urlencode(
            {"symbol": symbol, "interval": interval, "limit": limit}
        )
        payload = self.fetch_json(f"{self.klines_endpoint}?{params}")
        if not isinstance(payload, list):
            raise PolymarketLivePaperError(
                "invalid Binance klines payload",
                reason_codes=("invalid_binance_klines_payload",),
            )
        return payload


def _readonly_flags() -> dict[str, bool]:
    return {
        "read_only": True,
        "write_capable": False,
        "paper_only": True,
        "capital_at_risk": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }
