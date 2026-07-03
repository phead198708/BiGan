"""Polymarket live read-only feeds for BTC UP/DOWN paper runs."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.corpus import BTC_UPDOWN_MARKET_HORIZONS_MS
from bigan.v8.polymarket.live.contracts import (
    PolymarketLiveMarket,
    PolymarketLiveOrderBook,
    PolymarketLivePaperConfig,
    PolymarketLivePaperError,
    PolymarketLiveTrade,
)

MOCK_LIVE_BASE_TS = 1_780_300_000_000


class MockPolymarketLiveFeed:
    """Deterministic public Polymarket feed for CI and local smoke tests."""

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

    def market_rows(self) -> list[dict[str, Any]]:
        rows = []
        for index, family in enumerate(self.config.market_families):
            horizon_ms = BTC_UPDOWN_MARKET_HORIZONS_MS[family]
            start_ts = MOCK_LIVE_BASE_TS + index * 7_200_000
            end_ts = start_ts + horizon_ms
            settlement_rule = (
                ""
                if self.config.inject_missing_market_rule and index == 0
                else (
                    "UP wins if the BTC reference price at market end is greater "
                    "than the BTC reference price at market start; otherwise DOWN wins."
                )
            )
            row = {
                "market_id": f"pm-live-{family}-{index}",
                "condition_id": f"0xlivecondition{index:04d}",
                "slug": f"bitcoin-up-or-down-live-{family}-{index}",
                "market_family": family,
                "horizon_ms": horizon_ms,
                "market_start_ts": start_ts,
                "market_end_ts": end_ts,
                "settlement_ts": end_ts + 60_000,
                "up_token_id": f"live-up-token-{index}",
                "down_token_id": f"live-down-token-{index}",
                "reference_price_source": "binance_btcusdt",
                "settlement_rule": settlement_rule,
                "reference_price_at_start": 65_000.0 + index * 125.0,
                "status": (
                    "settlement_pending"
                    if self.config.settlement_mode == "delayed"
                    else "resolved"
                ),
                "resolution_available": self.config.settlement_mode == "resolved",
                "raw_market_sha256": canonical_json_sha256(
                    {"family": family, "index": index, "source": "mock_live"}
                ),
                **_readonly_flags(),
            }
            rows.append(row)
        return rows

    def markets(self) -> tuple[PolymarketLiveMarket, ...]:
        return tuple(PolymarketLiveMarket(**row) for row in self.market_rows())

    def orderbook_rows(
        self,
        markets: tuple[PolymarketLiveMarket, ...],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for market_index, market in enumerate(markets):
            step_ms = max(60_000, market.horizon_ms // 3)
            for sample_index in range(3):
                ts = min(
                    market.market_end_ts - 1_000,
                    market.market_start_ts + sample_index * step_ms,
                )
                received_ts = ts + 1_000
                if self.config.inject_stale_orderbook and market_index == 0 and sample_index == 0:
                    received_ts = ts + (self.config.max_stale_orderbook_seconds + 5) * 1000
                up_mid = 0.46 + market_index * 0.01 + sample_index * 0.015
                down_mid = 1.0 - up_mid
                for outcome, mid in (("UP", up_mid), ("DOWN", down_mid)):
                    if (
                        self.config.inject_missing_token_book
                        and market_index == 0
                        and sample_index == 0
                        and outcome == "DOWN"
                    ):
                        continue
                    rows.append(
                        {
                            "market_id": market.market_id,
                            "token_id": market.token_id_for_outcome(outcome),
                            "outcome": outcome,
                            "ts": ts,
                            "received_ts": received_ts,
                            "bid_price": round(mid - 0.01, 6),
                            "ask_price": round(mid + 0.01, 6),
                            "mid_price": round(mid, 6),
                            "bid_size": 500.0 + 20.0 * sample_index,
                            "ask_size": 480.0 + 20.0 * sample_index,
                            "liquidity_depth": 980.0 + 40.0 * sample_index,
                            **_readonly_flags(),
                        }
                    )
        return rows

    def orderbooks(
        self,
        markets: tuple[PolymarketLiveMarket, ...],
    ) -> tuple[PolymarketLiveOrderBook, ...]:
        return tuple(PolymarketLiveOrderBook(**row) for row in self.orderbook_rows(markets))

    def trade_rows(
        self,
        markets: tuple[PolymarketLiveMarket, ...],
    ) -> list[dict[str, Any]]:
        rows = []
        for index, market in enumerate(markets):
            ts = market.market_start_ts + 30_000
            rows.append(
                {
                    "market_id": market.market_id,
                    "token_id": market.up_token_id,
                    "outcome": "UP",
                    "ts": ts,
                    "price": 0.47 + index * 0.01,
                    "size": 25.0 + index,
                    "side": "BUY",
                    **_readonly_flags(),
                }
            )
        return rows

    def trades(
        self,
        markets: tuple[PolymarketLiveMarket, ...],
    ) -> tuple[PolymarketLiveTrade, ...]:
        return tuple(PolymarketLiveTrade(**row) for row in self.trade_rows(markets))


class PolymarketHTTPReadOnlyFeed:
    """Public HTTP feed shell for real read-only Polymarket data."""

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
        markets_endpoint: str = "https://gamma-api.polymarket.com/markets",
        clob_book_endpoint: str = "https://clob.polymarket.com/book",
        timeout_seconds: float = 10.0,
    ) -> None:
        self.markets_endpoint = markets_endpoint
        self.clob_book_endpoint = clob_book_endpoint
        self.timeout_seconds = timeout_seconds

    def fetch_json(self, url: str) -> Mapping[str, Any] | list[Any]:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "bigan-v8-polymarket-live-readonly/1.0"},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def fetch_market_payloads(self, *, query: str = "bitcoin up or down") -> list[dict[str, Any]]:
        params = urllib.parse.urlencode({"q": query, "active": "true"})
        payload = self.fetch_json(f"{self.markets_endpoint}?{params}")
        if isinstance(payload, list):
            return [dict(row) for row in payload]
        if isinstance(payload, dict):
            data = payload.get("data", payload.get("markets", []))
            if isinstance(data, list):
                return [dict(row) for row in data]
        raise PolymarketLivePaperError(
            "invalid Polymarket markets payload",
            reason_codes=("invalid_polymarket_market_payload",),
        )

    def fetch_orderbook_payload(self, token_id: str) -> dict[str, Any]:
        if not token_id.strip():
            raise ValueError("token_id is required")
        params = urllib.parse.urlencode({"token_id": token_id})
        payload = self.fetch_json(f"{self.clob_book_endpoint}?{params}")
        if not isinstance(payload, dict):
            raise PolymarketLivePaperError(
                "invalid Polymarket orderbook payload",
                reason_codes=("invalid_polymarket_orderbook_payload",),
            )
        return dict(payload)


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
