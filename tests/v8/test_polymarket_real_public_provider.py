"""Read-only public provider tests for real Polymarket corpus collection."""

from __future__ import annotations

import json
import urllib.parse

from bigan.v8.polymarket import (
    PolymarketPublicHTTPRealCorpusProvider,
    PolymarketRealCorpusRecorderConfig,
)


def test_public_http_provider_normalizes_public_market_rows_without_fake_resolution() -> None:
    provider = PolymarketPublicHTTPRealCorpusProvider(
        current_time_ms=1_700_001_000_000,
        fetch_json=FakePublicFetch(include_reference_prices=False),
    )
    config = PolymarketRealCorpusRecorderConfig(
        run_id="provider",
        output_dir="/tmp/provider",
        market_families=("btc_updown_5m",),
        mock_public_data=False,
    )

    markets = provider.market_rows(config)
    books = provider.orderbook_rows(markets, config)
    trades = provider.trade_rows(markets, config)
    candles = provider.btc_feature_candle_rows(markets, config)
    resolutions = provider.resolution_rows(markets, config)

    assert len(markets) == 1
    assert markets[0]["slug"] == "btc-updown-5m-1700000000"
    assert markets[0]["market_start_ts"] == 1_700_000_000_000
    assert markets[0]["market_end_ts"] == 1_700_000_300_000
    assert markets[0]["reference_price_source"] == "https://data.chain.link/streams/btc-usd"
    assert markets[0]["up_token_id"] == "up-token"
    assert markets[0]["down_token_id"] == "down-token"
    assert len(books) == 2
    assert {row["outcome"] for row in books} == {"UP", "DOWN"}
    assert len(trades) == 2
    assert len(candles) == 2
    assert resolutions == []


def test_public_http_provider_uses_official_reference_prices_when_present() -> None:
    provider = PolymarketPublicHTTPRealCorpusProvider(
        current_time_ms=1_700_001_000_000,
        fetch_json=FakePublicFetch(include_reference_prices=True),
    )
    config = PolymarketRealCorpusRecorderConfig(
        run_id="provider",
        output_dir="/tmp/provider",
        market_families=("btc_updown_5m",),
        mock_public_data=False,
    )

    markets = provider.market_rows(config)
    resolutions = provider.resolution_rows(markets, config)

    assert len(resolutions) == 1
    assert resolutions[0]["reference_price_source"] == "https://data.chain.link/streams/btc-usd"
    assert resolutions[0]["reference_price_start"] == 65000.0
    assert resolutions[0]["reference_price_end"] == 65025.0


class FakePublicFetch:
    def __init__(self, *, include_reference_prices: bool) -> None:
        self.include_reference_prices = include_reference_prices

    def __call__(self, url: str):
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        if "data-api.polymarket.com" in parsed.netloc:
            if "market" in query:
                return [
                    {
                        "asset": "up-token",
                        "outcome": "Up",
                        "side": "BUY",
                        "price": 0.57,
                        "size": 12.5,
                        "timestamp": 1_700_000_060,
                    },
                    {
                        "asset": "down-token",
                        "outcome": "Down",
                        "side": "SELL",
                        "price": 0.43,
                        "size": 11.0,
                        "timestamp": 1_700_000_061,
                    },
                ]
            return [{"slug": "btc-updown-5m-1700000000"}]
        if "gamma-api.polymarket.com" in parsed.netloc:
            payload = {
                "conditionId": "0xcondition",
                "slug": "btc-updown-5m-1700000000",
                "question": "Bitcoin Up or Down - test",
                "description": "The resolution source is Chainlink BTC/USD.",
                "resolutionSource": "https://data.chain.link/streams/btc-usd",
                "outcomes": json.dumps(["Up", "Down"]),
                "clobTokenIds": json.dumps(["up-token", "down-token"]),
                "endDate": "2023-11-14T22:18:20Z",
            }
            if self.include_reference_prices:
                payload["referencePriceStart"] = "65000"
                payload["referencePriceEnd"] = "65025"
            return [payload]
        if "clob.polymarket.com" in parsed.netloc:
            token_id = query["token_id"][0]
            if token_id == "up-token":
                return _book_payload(token_id=token_id, bid=0.56, ask=0.58)
            return _book_payload(token_id=token_id, bid=0.42, ask=0.44)
        if "api.binance.com" in parsed.netloc:
            return [
                [1_699_999_100_000, "65000", "65010", "64990", "65005", "10"],
                [1_699_999_160_000, "65005", "65020", "65000", "65012", "11"],
            ]
        raise AssertionError(f"unexpected url: {url}")


def _book_payload(*, token_id: str, bid: float, ask: float) -> dict:
    return {
        "market": "0xcondition",
        "asset_id": token_id,
        "timestamp": "1700000000000",
        "bids": [{"price": str(bid), "size": "100"}],
        "asks": [{"price": str(ask), "size": "120"}],
    }
