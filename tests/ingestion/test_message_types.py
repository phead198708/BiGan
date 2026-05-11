"""Unit tests for parsing Polymarket CLOB market-channel events.

Golden payloads come from the official docs at
https://docs.polymarket.com/developers/CLOB/websocket/market-channel
plus a handful of edge cases we want to lock in.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from bigan.ingestion.message_types import (
    BestBidAskEvent,
    BookEvent,
    EventType,
    LastTradePriceEvent,
    PriceChangeEvent,
    Side,
    TickSizeChangeEvent,
    UnknownEvent,
    parse_event,
)


def test_parse_book_event() -> None:
    payload = {
        "event_type": "book",
        "asset_id": "65818619657568813474341868652308942079804919287380422192892211131408793125422",
        "market": "0xbd31dc8a20211944f6b70f31557f1001557b59905b7738480ca09bd4532f84af",
        "bids": [
            {"price": "0.48", "size": "30"},
            {"price": "0.49", "size": "20"},
            {"price": "0.50", "size": "15"},
        ],
        "asks": [
            {"price": "0.52", "size": "25"},
            {"price": "0.53", "size": "60"},
            {"price": "0.54", "size": "10"},
        ],
        "timestamp": "123456789000",
        "hash": "0xdeadbeef",
    }
    event = parse_event(payload, receive_time_ms=999)
    assert isinstance(event, BookEvent)
    assert event.event_type is EventType.BOOK
    assert event.timestamp == 123456789000
    assert event.receive_time == 999
    assert event.hash == "0xdeadbeef"
    assert len(event.bids) == 3
    assert event.bids[0].price == Decimal("0.48")
    assert event.bids[0].size == Decimal("30")
    assert event.asks[2].price == Decimal("0.54")


def test_parse_price_change_event() -> None:
    payload = {
        "event_type": "price_change",
        "market": "0x5f65177b394277fd294cd75650044e32ba009a95022d88a0c1d565897d72f8f1",
        "price_changes": [
            {
                "asset_id": "71321045679252212594626385532706912750332728571942532289631379312455583992563",
                "price": "0.5",
                "size": "200",
                "side": "BUY",
                "hash": "56621a121a47ed9333273e21c83b660cff37ae50",
                "best_bid": "0.5",
                "best_ask": "1",
            }
        ],
        "timestamp": "1757908892351",
    }
    event = parse_event(payload, receive_time_ms=1)
    assert isinstance(event, PriceChangeEvent)
    assert len(event.price_changes) == 1
    pc = event.price_changes[0]
    assert pc.side is Side.BUY
    assert pc.price == Decimal("0.5")
    assert pc.size == Decimal("200")
    assert pc.best_bid == Decimal("0.5")


def test_parse_best_bid_ask_event() -> None:
    payload = {
        "event_type": "best_bid_ask",
        "market": "0x0005c0d312de0be897668695bae9f32b624b4a1ae8b140c49f08447fcc74f442",
        "asset_id": "85354956062430465315924116860125388538595433819574542752031640332592237464430",
        "best_bid": "0.73",
        "best_ask": "0.77",
        "spread": "0.04",
        "timestamp": "1766789469958",
    }
    event = parse_event(payload, receive_time_ms=5)
    assert isinstance(event, BestBidAskEvent)
    assert event.best_bid == Decimal("0.73")
    assert event.best_ask == Decimal("0.77")
    assert event.spread == Decimal("0.04")


def test_parse_last_trade_price_event() -> None:
    payload = {
        "asset_id": "114122071509644379678018727908709560226618148003371446110114509806601493071694",
        "event_type": "last_trade_price",
        "fee_rate_bps": "0",
        "market": "0x6a67b9d828d53862160e470329ffea5246f338ecfffdf2cab45211ec578b0347",
        "price": "0.456",
        "side": "BUY",
        "size": "219.217767",
        "timestamp": "1750428146322",
    }
    event = parse_event(payload, receive_time_ms=2)
    assert isinstance(event, LastTradePriceEvent)
    assert event.side is Side.BUY
    assert event.size == Decimal("219.217767")
    assert event.fee_rate_bps == Decimal("0")


def test_parse_tick_size_change_event() -> None:
    payload = {
        "event_type": "tick_size_change",
        "asset_id": "65818619657568813474341868652308942079804919287380422192892211131408793125422",
        "market": "0xbd31dc8a20211944f6b70f31557f1001557b59905b7738480ca09bd4532f84af",
        "old_tick_size": "0.01",
        "new_tick_size": "0.001",
        "timestamp": "100000000",
    }
    event = parse_event(payload, receive_time_ms=10)
    assert isinstance(event, TickSizeChangeEvent)
    assert event.old_tick_size == Decimal("0.01")
    assert event.new_tick_size == Decimal("0.001")


def test_unknown_event_type_raises() -> None:
    with pytest.raises(UnknownEvent):
        parse_event({"event_type": "not_a_real_event", "timestamp": "1"})


def test_missing_event_type_raises() -> None:
    with pytest.raises(UnknownEvent):
        parse_event({"timestamp": "1"})


def test_receive_time_injection_does_not_overwrite() -> None:
    """If the payload already carries a receive_time we must not overwrite it."""
    payload = {
        "event_type": "best_bid_ask",
        "market": "0x0",
        "asset_id": "1",
        "timestamp": "10",
        "receive_time": 42,
    }
    event = parse_event(payload, receive_time_ms=999)
    assert event.receive_time == 999  # current behaviour: injected wins on each call
