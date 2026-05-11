"""Unit tests for the Polymarket CLOB REST client (issue #5).

We never touch the live API: a fake :class:`aiohttp.ClientSession` is
injected so we can assert URL/params and feed canned responses.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import pytest

from bigan.ingestion.clob_rest import (
    PolymarketRestClient,
    RestOrderbook,
    RestTrade,
    _as_int_ms,
    _parse_orderbook,
    _parse_trade,
)

# ---------------------------------------------------------------------------
# Fake aiohttp session
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, *, status: int, payload: Any) -> None:
        self.status = status
        self._payload = payload

    async def json(self) -> Any:
        return self._payload

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None


class _FakeSession:
    """Records ``get`` calls and replays canned responses in order."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, dict]] = []
        self.closed = False

    def get(self, url: str, params: dict | None = None):  # type: ignore[no-untyped-def]
        self.calls.append((url, dict(params or {})))
        if not self._responses:
            return _FakeResponse(status=500, payload=None)
        nxt = self._responses.pop(0)
        if isinstance(nxt, _FakeResponse):
            return nxt
        return _FakeResponse(status=200, payload=nxt)

    async def close(self) -> None:
        self.closed = True


@asynccontextmanager
async def _client_with(session: _FakeSession):
    client = PolymarketRestClient(
        "https://clob.polymarket.com", session=session  # type: ignore[arg-type]
    )
    async with client:
        yield client


# ---------------------------------------------------------------------------
# Parser unit tests
# ---------------------------------------------------------------------------


def test_as_int_ms_handles_seconds_and_milliseconds() -> None:
    # 1700000000 (s) becomes 1700000000000 (ms)
    assert _as_int_ms("1700000000") == 1_700_000_000_000
    # already milliseconds — preserved.
    assert _as_int_ms("1700000000123") == 1_700_000_000_123
    assert _as_int_ms(None) is None
    assert _as_int_ms("not-a-number") is None
    assert _as_int_ms("-5") is None


def test_parse_trade_minimum_fields() -> None:
    trade = _parse_trade(
        {
            "asset_id": "tok-1",
            "market": "0xmkt",
            "price": "0.51",
            "size": "10",
            "side": "BUY",
            "match_time": "1700000000",
        }
    )
    assert isinstance(trade, RestTrade)
    assert trade.price == 0.51
    assert trade.size == 10.0
    assert trade.side == "BUY"
    assert trade.match_time_ms == 1_700_000_000_000


def test_parse_trade_rejects_invalid_side() -> None:
    assert (
        _parse_trade(
            {
                "asset_id": "tok-1",
                "market": "0xmkt",
                "price": "0.5",
                "size": "1",
                "side": "WAT",
                "match_time": "1700000000",
            }
        )
        is None
    )


def test_parse_orderbook_handles_ask_alias() -> None:
    book = _parse_orderbook(
        {
            "asset_id": "tok-1",
            "market": "0xmkt",
            "timestamp": "1700000000",
            "hash": "h0",
            "bids": [{"price": "0.50", "size": "100"}],
            "ask": [{"price": "0.52", "size": "50"}],  # <-- legacy alias
        }
    )
    assert isinstance(book, RestOrderbook)
    assert book.bids == [(0.50, 100.0)]
    assert book.asks == [(0.52, 50.0)]
    assert book.timestamp_ms == 1_700_000_000_000


# ---------------------------------------------------------------------------
# Client behaviour
# ---------------------------------------------------------------------------


def test_fetch_orderbook_round_trip() -> None:
    payload = {
        "asset_id": "tok-1",
        "market": "0xmkt",
        "timestamp": "1700000000",
        "hash": "h0",
        "bids": [{"price": "0.49", "size": "100"}],
        "asks": [{"price": "0.51", "size": "50"}],
    }
    session = _FakeSession([payload])

    async def go() -> RestOrderbook | None:
        async with _client_with(session) as client:
            return await client.fetch_orderbook("tok-1")

    result = asyncio.run(go())
    assert isinstance(result, RestOrderbook)
    assert result.asset_id == "tok-1"
    assert session.calls[0][0].endswith("/book")
    assert session.calls[0][1] == {"token_id": "tok-1"}


def test_fetch_trades_paginates_until_cursor_exhausted() -> None:
    page1 = {
        "data": [
            {
                "asset_id": "tok-1",
                "market": "0xmkt",
                "price": "0.51",
                "size": "10",
                "side": "BUY",
                "match_time": "1700000000",
            }
        ],
        "next_cursor": "abc",
    }
    page2 = {
        "data": [
            {
                "asset_id": "tok-1",
                "market": "0xmkt",
                "price": "0.50",
                "size": "5",
                "side": "SELL",
                "match_time": "1700000005",
            }
        ],
        "next_cursor": None,
    }
    session = _FakeSession([page1, page2])

    async def go() -> list[RestTrade]:
        async with _client_with(session) as client:
            return await client.fetch_trades("0xmkt")

    trades = asyncio.run(go())
    assert len(trades) == 2
    assert {t.side for t in trades} == {"BUY", "SELL"}
    assert len(session.calls) == 2
    # Second call carried the cursor.
    assert session.calls[1][1].get("next_cursor") == "abc"


def test_fetch_trades_filters_by_window_and_stops_early_on_old_data() -> None:
    page1 = {
        "data": [
            {
                "asset_id": "tok-1",
                "market": "0xmkt",
                "price": "0.51",
                "size": "10",
                "side": "BUY",
                "match_time": "1700000010",  # in window
            },
            {
                "asset_id": "tok-1",
                "market": "0xmkt",
                "price": "0.50",
                "size": "5",
                "side": "SELL",
                "match_time": "1600000000",  # before window — triggers stop
            },
        ],
        "next_cursor": "abc",  # would normally fetch more
    }
    session = _FakeSession([page1])

    async def go() -> list[RestTrade]:
        async with _client_with(session) as client:
            return await client.fetch_trades(
                "0xmkt",
                since_ms=1_700_000_000_000,
                until_ms=1_700_000_020_000,
            )

    trades = asyncio.run(go())
    # Only the in-window trade is yielded; old trade triggers early stop.
    assert len(trades) == 1
    assert trades[0].side == "BUY"
    assert len(session.calls) == 1


def test_non_200_returns_none() -> None:
    session = _FakeSession([_FakeResponse(status=500, payload=None)])

    async def go() -> RestOrderbook | None:
        async with _client_with(session) as client:
            return await client.fetch_orderbook("tok-1")

    assert asyncio.run(go()) is None


def test_client_requires_active_context() -> None:
    client = PolymarketRestClient("https://clob.polymarket.com")
    with pytest.raises(RuntimeError):
        asyncio.run(client.fetch_orderbook("tok-1"))
