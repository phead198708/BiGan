"""DEV-04 Polymarket CLOB websocket feed handler."""

from __future__ import annotations

import logging

import orjson
import pytest

from bigan.data.polymarket_clob import MarketSnapshot, PolymarketFeedHandler


def _book_payload() -> dict[str, object]:
    return {
        "event_type": "book",
        "market": "0xcondition",
        "window_id": "btc-updown-15m-1",
        "timestamp": "1700000000000",
        "sequence": 10,
        "yes": {
            "bids": [{"price": "0.56", "size": "100"}],
            "asks": [{"price": "0.58", "size": "120"}],
        },
        "no": {
            "bids": [["0.42", "25"], ["0.41", "10"]],
            "asks": [["0.44", "30"]],
        },
        "last_trade_price": "0.57",
    }


def test_parse_orderbook_payload() -> None:
    handler = PolymarketFeedHandler(
        window_id="btc-updown-15m-1",
        yes_token_id="yes-token",
        no_token_id="no-token",
    )
    snapshot = handler.parse_orderbook_delta(_book_payload())
    assert snapshot is not None
    assert snapshot.timestamp_ms == 1_700_000_000_000
    assert snapshot.window_id == "btc-updown-15m-1"
    assert snapshot.yes_bid == pytest.approx(0.56)
    assert snapshot.yes_ask == pytest.approx(0.58)
    assert snapshot.no_bid == pytest.approx(0.42)
    assert snapshot.no_ask == pytest.approx(0.44)
    assert snapshot.last_traded_price == pytest.approx(0.57)
    assert snapshot.yes_bid_size == pytest.approx(100.0)
    assert snapshot.yes_ask_size == pytest.approx(120.0)
    assert snapshot.no_bid_size == pytest.approx(25.0)
    assert snapshot.no_ask_size == pytest.approx(30.0)

    yes_only = handler.parse_orderbook_delta(
        {
            "event_type": "book",
            "asset_id": "yes-token",
            "timestamp": "1700000000100",
            "sequence": 11,
            "bids": [{"price": "0.60", "size": "8"}],
            "asks": [{"price": "0.62", "size": "9"}],
        }
    )
    assert yes_only is not None
    assert yes_only.yes_bid == pytest.approx(0.60)
    assert yes_only.yes_ask == pytest.approx(0.62)
    assert yes_only.yes_bid_size == pytest.approx(8.0)
    assert yes_only.yes_ask_size == pytest.approx(9.0)
    assert yes_only.no_bid == pytest.approx(0.42)

    stale = handler.parse_orderbook_delta(
        {
            "event_type": "book",
            "timestamp": "1699999999999",
            "sequence": 9,
            "yes_bid": 0.99,
            "yes_ask": 0.99,
            "no_bid": 0.01,
            "no_ask": 0.01,
        }
    )
    assert stale is None
    assert handler.dropped_stale >= 1


@pytest.mark.asyncio
async def test_invalid_json_graceful_handling(caplog: pytest.LogCaptureFixture) -> None:
    handler = PolymarketFeedHandler(window_id="btc-updown-5m-1")
    await handler.connect()
    with caplog.at_level(logging.WARNING):
        result = await handler.ingest_raw("{not-json")
    assert result is None
    assert handler.parse_errors == 1
    assert handler.connected is True
    assert any("invalid_json" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_snapshot_callback_trigger() -> None:
    handler = PolymarketFeedHandler(
        window_id="btc-updown-15m-1",
        yes_token_id="yes-token",
        no_token_id="no-token",
    )
    received: list[MarketSnapshot] = []

    async def on_book(snapshot: MarketSnapshot) -> None:
        received.append(snapshot)

    handler.on_snapshot(on_book)
    await handler.connect()
    snapshot = await handler.ingest_raw(orjson.dumps(_book_payload()))
    assert snapshot is not None
    assert len(received) == 1
    assert received[0] == snapshot
    assert received[0].yes_ask == pytest.approx(0.58)

    await handler.ingest_payload(
        {
            "event_type": "price_change",
            "timestamp": "1700000000200",
            "sequence": 12,
            "price_changes": [
                {
                    "asset_id": "no-token",
                    "best_bid": "0.40",
                    "best_ask": "0.43",
                }
            ],
        }
    )
    assert len(received) == 2
    assert received[1].no_bid == pytest.approx(0.40)
    assert received[1].no_ask == pytest.approx(0.43)
    assert received[1].yes_bid == pytest.approx(0.56)

    trade = await handler.ingest_payload(
        {
            "event_type": "last_trade_price",
            "asset_id": "yes-token",
            "timestamp": "1700000000300",
            "sequence": 13,
            "price": "0.55",
        }
    )
    assert trade is not None
    assert trade.timestamp_ms == 1_700_000_000_300
    assert trade.last_traded_price == pytest.approx(0.55)
    assert len(received) == 3
    assert received[-1] == trade


@pytest.mark.asyncio
async def test_application_heartbeat_and_pong_handling() -> None:
    handler = PolymarketFeedHandler(
        window_id="btc-updown-15m-1",
        heartbeat_interval_seconds=0.001,
    )

    class Socket:
        def __init__(self) -> None:
            self.messages: list[str] = []

        async def send(self, message: str) -> None:
            self.messages.append(message)
            handler._closed = True

    socket = Socket()
    await handler._heartbeat_loop(socket)
    assert socket.messages == ["PING"]
    assert await handler.ingest_raw("PONG") is None
    assert handler.parse_errors == 0


def test_empty_full_book_side_clears_stale_quote() -> None:
    handler = PolymarketFeedHandler(
        window_id="btc-updown-15m-1",
        yes_token_id="yes-token",
        no_token_id="no-token",
    )
    assert handler.parse_orderbook_delta(_book_payload()) is not None
    cleared = handler.parse_orderbook_delta(
        {
            "event_type": "book",
            "asset_id": "yes-token",
            "timestamp": "1700000000100",
            "sequence": 11,
            "bids": [{"price": "0.56", "size": "100"}],
            "asks": [],
        }
    )
    assert cleared is None
    assert handler._yes_ask is None
    assert handler._yes_ask_size == 0.0


def test_snapshot_requires_fresh_quotes_from_both_tokens() -> None:
    handler = PolymarketFeedHandler(
        window_id="btc-updown-15m-1",
        yes_token_id="yes-token",
        no_token_id="no-token",
        max_quote_age_ms=5_000,
    )
    assert handler.parse_orderbook_delta(_book_payload()) is not None
    stale_pair = handler.parse_orderbook_delta(
        {
            "event_type": "book",
            "asset_id": "yes-token",
            "timestamp": "1700000006001",
            "sequence": 11,
            "bids": [{"price": "0.57", "size": "100"}],
            "asks": [{"price": "0.59", "size": "100"}],
        }
    )
    assert stale_pair is None
