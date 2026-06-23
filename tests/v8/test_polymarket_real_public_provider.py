"""Read-only public provider tests for real Polymarket corpus collection."""

from __future__ import annotations

import json
import urllib.parse

from bigan.v8.polymarket import (
    PolymarketCLOBWebSocketOrderBookSource,
    PolymarketPublicHTTPRealCorpusProvider,
    PolymarketRealCorpusRecorderConfig,
)


def test_public_http_provider_normalizes_public_market_rows_without_fake_resolution() -> None:
    orderbook_source = FakeOrderBookSource()
    provider = PolymarketPublicHTTPRealCorpusProvider(
        current_time_ms=1_700_001_000_000,
        fetch_json=FakePublicFetch(
            include_reference_prices=False,
            fail_clob_books=True,
            fail_binance=True,
        ),
        orderbook_source=orderbook_source,
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
    assert markets[0]["slug"] == "btc-updown-5m-1700001000"
    assert markets[0]["market_start_ts"] == 1_700_001_000_000
    assert markets[0]["market_end_ts"] == 1_700_001_300_000
    assert markets[0]["reference_price_source"] == "https://data.chain.link/streams/btc-usd"
    assert markets[0]["up_token_id"] == "up-token"
    assert markets[0]["down_token_id"] == "down-token"
    assert len(books) == 2
    assert {row["outcome"] for row in books} == {"UP", "DOWN"}
    assert orderbook_source.requested_token_ids == ("up-token", "down-token")
    assert len(trades) == 2
    assert len(candles) == 2
    assert {row["source"] for row in candles} == {"coinbase_btc_usd"}
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


def test_public_http_provider_uses_settled_gamma_outcome_prices_after_round() -> None:
    provider = PolymarketPublicHTTPRealCorpusProvider(
        current_time_ms=1_700_001_400_000,
        market_slugs=("btc-updown-5m-1700001000",),
        fetch_json=FakePublicFetch(
            include_reference_prices=False,
            outcome_prices=("0", "1"),
        ),
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
    assert "reference_price_start" not in resolutions[0]
    assert resolutions[0]["resolution_source_type"] == "gamma_outcome_prices"
    assert resolutions[0]["resolved_outcome"] == "DOWN"
    assert resolutions[0]["payout_up"] == 0.0
    assert resolutions[0]["payout_down"] == 1.0


def test_public_http_provider_refetches_gamma_resolution_after_round() -> None:
    fetch_json = DeferredResolutionFetch()
    provider = PolymarketPublicHTTPRealCorpusProvider(
        current_time_ms=1_700_001_400_000,
        market_slugs=("btc-updown-5m-1700001000",),
        fetch_json=fetch_json,
    )
    config = PolymarketRealCorpusRecorderConfig(
        run_id="provider",
        output_dir="/tmp/provider",
        market_families=("btc_updown_5m",),
        mock_public_data=False,
    )

    markets = provider.market_rows(config)
    resolutions = provider.resolution_rows(markets, config)

    assert fetch_json.gamma_fetch_count == 2
    assert len(resolutions) == 1
    assert resolutions[0]["resolution_source_type"] == "gamma_outcome_prices"
    assert resolutions[0]["resolved_outcome"] == "UP"
    assert resolutions[0]["payout_up"] == 1.0
    assert resolutions[0]["payout_down"] == 0.0


def test_public_http_provider_does_not_treat_live_outcome_prices_as_resolution() -> None:
    provider = PolymarketPublicHTTPRealCorpusProvider(
        current_time_ms=1_700_001_000_000,
        fetch_json=FakePublicFetch(
            include_reference_prices=False,
            outcome_prices=("0.57", "0.43"),
        ),
    )
    config = PolymarketRealCorpusRecorderConfig(
        run_id="provider",
        output_dir="/tmp/provider",
        market_families=("btc_updown_5m",),
        mock_public_data=False,
    )

    markets = provider.market_rows(config)
    resolutions = provider.resolution_rows(markets, config)

    assert resolutions == []


def test_public_http_provider_can_use_rest_orderbook_fallback_when_explicit() -> None:
    provider = PolymarketPublicHTTPRealCorpusProvider(
        current_time_ms=1_700_001_000_000,
        fetch_json=FakePublicFetch(include_reference_prices=False),
        use_rest_orderbooks=True,
    )
    config = PolymarketRealCorpusRecorderConfig(
        run_id="provider",
        output_dir="/tmp/provider",
        market_families=("btc_updown_5m",),
        mock_public_data=False,
    )

    markets = provider.market_rows(config)
    books = provider.orderbook_rows(markets, config)

    assert len(books) == 2
    assert {row["outcome"] for row in books} == {"UP", "DOWN"}


def test_public_http_provider_configures_default_websocket_snapshot_interval() -> None:
    provider = PolymarketPublicHTTPRealCorpusProvider(
        current_time_ms=1_700_001_000_000,
        fetch_json=FakePublicFetch(include_reference_prices=False),
        orderbook_snapshot_interval_seconds=2.5,
    )

    assert isinstance(provider.orderbook_source, PolymarketCLOBWebSocketOrderBookSource)
    assert provider.orderbook_source.snapshot_interval_seconds == 2.5


def test_public_http_provider_falls_back_to_kraken_feature_candles() -> None:
    provider = PolymarketPublicHTTPRealCorpusProvider(
        current_time_ms=1_700_001_000_000,
        fetch_json=FakePublicFetch(
            include_reference_prices=False,
            fail_coinbase=True,
            fail_binance=True,
        ),
    )
    config = PolymarketRealCorpusRecorderConfig(
        run_id="provider",
        output_dir="/tmp/provider",
        market_families=("btc_updown_5m",),
        mock_public_data=False,
    )

    markets = provider.market_rows(config)
    candles = provider.btc_feature_candle_rows(markets, config)

    assert len(candles) == 2
    assert {row["source"] for row in candles} == {"kraken_xbt_usd"}


def test_public_http_provider_uses_binance_feature_candles_as_last_resort() -> None:
    provider = PolymarketPublicHTTPRealCorpusProvider(
        current_time_ms=1_700_001_000_000,
        fetch_json=FakePublicFetch(
            include_reference_prices=False,
            fail_coinbase=True,
            fail_kraken=True,
        ),
    )
    config = PolymarketRealCorpusRecorderConfig(
        run_id="provider",
        output_dir="/tmp/provider",
        market_families=("btc_updown_5m",),
        mock_public_data=False,
    )

    markets = provider.market_rows(config)
    candles = provider.btc_feature_candle_rows(markets, config)

    assert len(candles) == 2
    assert {row["source"] for row in candles} == {"binance_btcusdt"}


def test_websocket_orderbook_source_projects_market_channel_events() -> None:
    source = PolymarketCLOBWebSocketOrderBookSource(timeout_seconds=1.0)
    book_payloads: dict[str, dict] = {}
    fallback_payloads: dict[str, dict] = {}
    resolution_payloads: dict[str, dict] = {}

    source._update_payload_maps(
        payload={
            "asset_id": "up-token",
            "market": "0xcondition",
            "timestamp": "1700000000000",
            "hash": "book-hash",
            "bids": [{"price": "0.56", "size": "100"}],
            "asks": [{"price": "0.58", "size": "120"}],
        },
        receive_time_ms=1_700_000_000_123,
        target_tokens={"up-token", "down-token"},
        book_payloads=book_payloads,
        fallback_payloads=fallback_payloads,
        resolution_payloads=resolution_payloads,
    )
    source._update_payload_maps(
        payload={
            "market": "0xcondition",
            "price_changes": [
                {
                    "asset_id": "down-token",
                    "price": "0.42",
                    "size": "25",
                    "side": "BUY",
                    "hash": "delta-hash",
                    "best_bid": "0.42",
                    "best_ask": "0.44",
                }
            ],
        },
        receive_time_ms=1_700_000_000_124,
        target_tokens={"up-token", "down-token"},
        book_payloads=book_payloads,
        fallback_payloads=fallback_payloads,
        resolution_payloads=resolution_payloads,
    )
    source._update_payload_maps(
        payload={
            "event_type": "market_resolved",
            "market": "0xcondition",
            "timestamp": "1700000300000",
            "assets_ids": ["up-token", "down-token"],
            "outcomes": ["Up", "Down"],
            "winning_asset_id": "up-token",
            "winning_outcome": "Up",
        },
        receive_time_ms=1_700_000_300_001,
        target_tokens={"up-token", "down-token"},
        book_payloads=book_payloads,
        fallback_payloads=fallback_payloads,
        resolution_payloads=resolution_payloads,
    )

    assert book_payloads["up-token"]["bids"] == [{"price": "0.56", "size": "100"}]
    assert book_payloads["up-token"]["source_event_type"] == "book"
    assert fallback_payloads["down-token"]["bids"] == [{"price": "0.42", "size": "0"}]
    assert fallback_payloads["down-token"]["timestamp"] == 1_700_000_000_124
    assert fallback_payloads["down-token"]["source_event_type"] == "price_change"
    assert resolution_payloads["0xcondition"]["winning_asset_id"] == "up-token"
    assert resolution_payloads["0xcondition"]["winning_outcome"] == "Up"


def test_public_http_provider_uses_websocket_market_resolved_event() -> None:
    provider = PolymarketPublicHTTPRealCorpusProvider(
        current_time_ms=1_700_001_000_000,
        fetch_json=FakePublicFetch(
            include_reference_prices=False,
            outcome_prices=("0.57", "0.43"),
        ),
        orderbook_source=FakeResolvedOrderBookSource(),
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
    assert resolutions[0]["resolution_source_type"] == "polymarket_clob_ws_market_resolved"
    assert resolutions[0]["resolved_outcome"] == "UP"
    assert resolutions[0]["payout_up"] == 1.0
    assert resolutions[0]["payout_down"] == 0.0


def test_public_http_provider_prefers_stream_orderbook_snapshots_when_available() -> None:
    orderbook_source = FakeStreamOrderBookSource()
    provider = PolymarketPublicHTTPRealCorpusProvider(
        current_time_ms=1_700_001_000_000,
        fetch_json=FakePublicFetch(include_reference_prices=True),
        orderbook_source=orderbook_source,
    )
    config = PolymarketRealCorpusRecorderConfig(
        run_id="provider",
        output_dir="/tmp/provider",
        market_families=("btc_updown_5m",),
        mock_public_data=False,
    )

    markets = provider.market_rows(config)
    books = provider.orderbook_rows(markets, config)

    assert orderbook_source.requested_token_ids == ("up-token", "down-token")
    assert len(books) == 4
    assert {row["available_at_ts"] for row in books} == {
        1_700_000_000_100,
        1_700_000_059_100,
    }


def test_public_http_provider_seeds_rest_book_before_websocket_snapshots() -> None:
    orderbook_source = FakeWebSocketStreamOrderBookSource()
    provider = PolymarketPublicHTTPRealCorpusProvider(
        current_time_ms=1_700_001_000_000,
        fetch_json=FakePublicFetch(include_reference_prices=True),
        orderbook_source=orderbook_source,
    )
    config = PolymarketRealCorpusRecorderConfig(
        run_id="provider",
        output_dir="/tmp/provider",
        market_families=("btc_updown_5m",),
        mock_public_data=False,
    )

    markets = provider.market_rows(config)
    books = provider.orderbook_rows(markets, config)

    assert orderbook_source.requested_token_ids == ("up-token", "down-token")
    assert len(books) == 4
    assert {row["available_at_ts"] for row in books} == {
        1_700_000_000_000,
        1_700_000_059_100,
    }
    assert {row["collection_end_ts"] for row in books} == {1_700_000_059_100}


class FakePublicFetch:
    def __init__(
        self,
        *,
        include_reference_prices: bool,
        outcome_prices: tuple[str, str] | None = None,
        fail_clob_books: bool = False,
        fail_coinbase: bool = False,
        fail_kraken: bool = False,
        fail_binance: bool = False,
    ) -> None:
        self.include_reference_prices = include_reference_prices
        self.outcome_prices = outcome_prices
        self.fail_clob_books = fail_clob_books
        self.fail_coinbase = fail_coinbase
        self.fail_kraken = fail_kraken
        self.fail_binance = fail_binance

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
            raise AssertionError("market discovery should use the current round slug, not trades")
        if "gamma-api.polymarket.com" in parsed.netloc:
            slug = query.get("slug", ["btc-updown-5m-1700001000"])[0]
            payload = {
                "conditionId": "0xcondition",
                "slug": slug,
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
            if self.outcome_prices is not None:
                payload["outcomePrices"] = json.dumps(list(self.outcome_prices))
            return [payload]
        if "clob.polymarket.com" in parsed.netloc:
            if self.fail_clob_books:
                raise AssertionError("REST CLOB /book should not be called")
            token_id = query["token_id"][0]
            if token_id == "up-token":
                return _book_payload(token_id=token_id, bid=0.56, ask=0.58)
            return _book_payload(token_id=token_id, bid=0.42, ask=0.44)
        if "api.exchange.coinbase.com" in parsed.netloc:
            if self.fail_coinbase:
                raise RuntimeError("Coinbase unavailable")
            return [
                [1_699_999_100, "64990", "65010", "65000", "65005", "10"],
                [1_699_999_160, "65000", "65020", "65005", "65012", "11"],
            ]
        if "api.kraken.com" in parsed.netloc:
            if self.fail_kraken:
                raise RuntimeError("Kraken unavailable")
            return {
                "error": [],
                "result": {
                    "XXBTZUSD": [
                        [
                            1_699_999_100,
                            "65000",
                            "65010",
                            "64990",
                            "65005",
                            "65002",
                            "10",
                            2,
                        ],
                        [
                            1_699_999_160,
                            "65005",
                            "65020",
                            "65000",
                            "65012",
                            "65008",
                            "11",
                            3,
                        ],
                    ],
                    "last": "1699999160",
                },
            }
        if "api.binance.com" in parsed.netloc:
            if self.fail_binance:
                raise AssertionError("Binance should only be used as last-resort backfill")
            return [
                [1_699_999_100_000, "65000", "65010", "64990", "65005", "10"],
                [1_699_999_160_000, "65005", "65020", "65000", "65012", "11"],
            ]
        raise AssertionError(f"unexpected url: {url}")


class DeferredResolutionFetch(FakePublicFetch):
    def __init__(self) -> None:
        super().__init__(include_reference_prices=False)
        self.gamma_fetch_count = 0

    def __call__(self, url: str):
        parsed = urllib.parse.urlparse(url)
        if "gamma-api.polymarket.com" not in parsed.netloc:
            return super().__call__(url)
        self.gamma_fetch_count += 1
        payload = super().__call__(url)[0]
        if self.gamma_fetch_count >= 2:
            payload["outcomePrices"] = json.dumps(["1", "0"])
        return [payload]


class FakeOrderBookSource:
    def __init__(self) -> None:
        self.requested_token_ids: tuple[str, ...] = ()

    def book_payloads(self, token_ids: tuple[str, ...]) -> dict[str, dict]:
        self.requested_token_ids = token_ids
        return {
            "up-token": _book_payload(token_id="up-token", bid=0.56, ask=0.58),
            "down-token": _book_payload(token_id="down-token", bid=0.42, ask=0.44),
        }


class FakeStreamOrderBookSource(FakeOrderBookSource):
    def book_payload_snapshots(self, token_ids: tuple[str, ...]) -> list[dict[str, dict]]:
        self.requested_token_ids = token_ids
        return [
            {
                "up-token": _book_payload(
                    token_id="up-token",
                    bid=0.56,
                    ask=0.58,
                    receive_time=1_700_000_000_100,
                ),
                "down-token": _book_payload(
                    token_id="down-token",
                    bid=0.42,
                    ask=0.44,
                    receive_time=1_700_000_000_100,
                ),
            },
            {
                "up-token": _book_payload(
                    token_id="up-token",
                    bid=0.57,
                    ask=0.59,
                    timestamp="1700000059000",
                    receive_time=1_700_000_059_100,
                ),
                "down-token": _book_payload(
                    token_id="down-token",
                    bid=0.41,
                    ask=0.43,
                    timestamp="1700000059000",
                    receive_time=1_700_000_059_100,
                ),
            },
        ]


class FakeWebSocketStreamOrderBookSource(PolymarketCLOBWebSocketOrderBookSource):
    def __init__(self) -> None:
        super().__init__(
            ws_url="wss://example.invalid/ws/market",
            timeout_seconds=1.0,
        )
        self.requested_token_ids: tuple[str, ...] = ()

    def book_payload_snapshots(self, token_ids: tuple[str, ...]) -> list[dict[str, dict]]:
        self.requested_token_ids = token_ids
        return [
            {
                "up-token": _book_payload(
                    token_id="up-token",
                    bid=0.57,
                    ask=0.59,
                    timestamp="1700000059000",
                    receive_time=1_700_000_059_100,
                ),
                "down-token": _book_payload(
                    token_id="down-token",
                    bid=0.41,
                    ask=0.43,
                    timestamp="1700000059000",
                    receive_time=1_700_000_059_100,
                ),
            }
        ]


class FakeResolvedOrderBookSource(FakeOrderBookSource):
    def market_resolution_payloads(self, token_ids: tuple[str, ...]) -> dict[str, dict]:
        self.requested_token_ids = token_ids
        return {
            "0xcondition": {
                "event_type": "market_resolved",
                "market": "0xcondition",
                "timestamp": 1_700_000_300_000,
                "receive_time": 1_700_000_300_001,
                "assets_ids": ["up-token", "down-token"],
                "outcomes": ["Up", "Down"],
                "winning_asset_id": "up-token",
                "winning_outcome": "Up",
            }
        }


def _book_payload(
    *,
    token_id: str,
    bid: float,
    ask: float,
    timestamp: str = "1700000000000",
    receive_time: int | None = None,
) -> dict:
    payload = {
        "market": "0xcondition",
        "asset_id": token_id,
        "timestamp": timestamp,
        "bids": [{"price": str(bid), "size": "100"}],
        "asks": [{"price": str(ask), "size": "120"}],
    }
    if receive_time is not None:
        payload["receive_time"] = receive_time
    return payload
