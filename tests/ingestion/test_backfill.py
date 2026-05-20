"""Unit tests for the BackfillService (issue #5)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from bigan.canonical.schemas import PROVENANCE_BACKFILL, PROVENANCE_REST_SEED
from bigan.ingestion.backfill import BackfillService, GapWindow, synth_orderbook_record
from bigan.ingestion.clob_rest import RestOrderbook, RestTrade

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeRest:
    """Minimal in-memory stand-in for PolymarketRestClient."""

    def __init__(
        self,
        *,
        trades: list[RestTrade] | None = None,
        orderbook: RestOrderbook | None = None,
        raise_on_trades: Exception | None = None,
        raise_on_book: Exception | None = None,
    ) -> None:
        self._trades = trades or []
        self._orderbook = orderbook
        self._raise_on_trades = raise_on_trades
        self._raise_on_book = raise_on_book
        self.trades_calls: list[tuple[str, int | None, int | None]] = []
        self.book_calls: list[str] = []

    async def fetch_trades(
        self,
        market_condition_id: str,
        *,
        since_ms: int | None = None,
        until_ms: int | None = None,
        max_pages: int = 50,
    ) -> list[RestTrade]:
        self.trades_calls.append((market_condition_id, since_ms, until_ms))
        if self._raise_on_trades is not None:
            raise self._raise_on_trades
        return list(self._trades)

    async def iter_trades(
        self,
        market_condition_id: str,
        *,
        since_ms: int | None = None,
        until_ms: int | None = None,
        max_pages: int = 50,
    ) -> AsyncIterator[RestTrade]:
        for t in await self.fetch_trades(
            market_condition_id, since_ms=since_ms, until_ms=until_ms
        ):
            yield t

    async def fetch_orderbook(self, asset_id: str) -> RestOrderbook | None:
        self.book_calls.append(asset_id)
        if self._raise_on_book is not None:
            raise self._raise_on_book
        return self._orderbook


class _RecordingSink:
    """Captures records so we can assert provenance + content."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    async def write(self, record: dict[str, Any]) -> None:
        self.records.append(record)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _trade(asset_id: str, side: str, ts_ms: int, price: float = 0.51) -> RestTrade:
    return RestTrade(
        asset_id=asset_id,
        market="0xmkt",
        price=price,
        size=10.0,
        side=side,
        match_time_ms=ts_ms,
        raw={},
    )


def _book(asset_id: str, ts_ms: int) -> RestOrderbook:
    return RestOrderbook(
        asset_id=asset_id,
        market="0xmkt",
        timestamp_ms=ts_ms,
        hash="h0",
        bids=[(0.49, 100.0)],
        asks=[(0.51, 50.0)],
        raw={},
    )


async def _resolver(asset_id: str) -> str | None:
    if asset_id == "tok-1":
        return "0xmkt"
    return None


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_replays_trades_and_orderbook_with_backfill_provenance() -> None:
    rest = _FakeRest(
        trades=[
            _trade("tok-1", "BUY", 1_700_000_010_000),
            _trade("tok-1", "SELL", 1_700_000_015_000),
            # different asset_id from same market — must be filtered out
            _trade("tok-OTHER", "BUY", 1_700_000_020_000),
        ],
        orderbook=_book("tok-1", 1_700_000_030_000),
    )
    sink = _RecordingSink()
    service = BackfillService(rest, sink, _resolver, clock_ms=lambda: 999)

    report = asyncio.run(
        service.handle_gap(
            GapWindow(
                asset_id="tok-1",
                gap_start_ms=1_700_000_000_000,
                gap_end_ms=1_700_000_030_000,
            )
        )
    )

    assert report.trades_replayed == 2
    assert report.orderbook_replayed is True
    assert report.errors == []
    assert report.market == "0xmkt"
    assert report.total_records == 3

    # Sink saw 2 trades + 1 book = 3 records, all tagged with backfill provenance.
    assert len(sink.records) == 3
    for rec in sink.records:
        assert rec["raw"]["provenance"] == PROVENANCE_BACKFILL
        assert rec["receive_time"] == 999
        assert rec["capture_timestamp_ms"] == 999
        assert rec["source_channel"] == "clob-rest"

    event_types = [r["raw"]["event_type"] for r in sink.records]
    assert event_types.count("last_trade_price") == 2
    assert event_types.count("book") == 1
    book_record = next(r for r in sink.records if r["raw"]["event_type"] == "book")
    assert book_record["source_timestamp_ms"] == 1_700_000_030_000
    assert book_record["raw"]["source_timestamp_ms"] == 1_700_000_030_000
    assert book_record["raw"]["capture_timestamp_ms"] == 999

    # Trade rest call was scoped to the gap window.
    assert rest.trades_calls == [("0xmkt", 1_700_000_000_000, 1_700_000_030_000)]


def test_synth_orderbook_record_carries_seed_provenance_and_timestamps() -> None:
    record = synth_orderbook_record(
        _book("tok-1", 1_700_000_030_000),
        receive_time_ms=1_700_000_030_111,
        provenance=PROVENANCE_REST_SEED,
    )

    assert record["receive_time"] == 1_700_000_030_111
    assert record["source_timestamp_ms"] == 1_700_000_030_000
    assert record["capture_timestamp_ms"] == 1_700_000_030_111
    assert record["source_channel"] == "clob-rest"
    assert record["provenance"] == PROVENANCE_REST_SEED
    assert record["raw"]["event_type"] == "book"
    assert record["raw"]["provenance"] == PROVENANCE_REST_SEED
    assert record["raw"]["source_timestamp_ms"] == 1_700_000_030_000
    assert record["raw"]["capture_timestamp_ms"] == 1_700_000_030_111


def test_backfill_service_calls_rest_gate_before_each_fetch() -> None:
    rest = _FakeRest(
        trades=[_trade("tok-1", "BUY", 1_700_000_010_000)],
        orderbook=_book("tok-1", 1_700_000_030_000),
    )
    sink = _RecordingSink()
    calls = 0

    async def before_rest_call() -> None:
        nonlocal calls
        calls += 1

    service = BackfillService(
        rest,
        sink,
        _resolver,
        before_rest_call=before_rest_call,
    )

    report = asyncio.run(
        service.handle_gap(
            GapWindow(
                asset_id="tok-1",
                gap_start_ms=1_700_000_000_000,
                gap_end_ms=1_700_000_030_000,
            )
        )
    )

    assert report.total_records == 2
    assert calls == 2


def test_falls_back_when_market_resolver_returns_none() -> None:
    rest = _FakeRest(orderbook=_book("tok-2", 1_700_000_030_000))
    sink = _RecordingSink()
    service = BackfillService(rest, sink, _resolver)

    report = asyncio.run(
        service.handle_gap(
            GapWindow(
                asset_id="tok-2",  # not in resolver
                gap_start_ms=1,
                gap_end_ms=2,
            )
        )
    )

    assert report.market is None
    assert report.trades_replayed == 0
    # Orderbook still attempted (doesn't need market).
    assert report.orderbook_replayed is True
    assert any("market resolver" in e for e in report.errors)


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_records_error_when_trades_fetch_raises() -> None:
    rest = _FakeRest(
        raise_on_trades=RuntimeError("boom"),
        orderbook=_book("tok-1", 1_700_000_030_000),
    )
    sink = _RecordingSink()
    service = BackfillService(rest, sink, _resolver)

    report = asyncio.run(
        service.handle_gap(
            GapWindow(asset_id="tok-1", gap_start_ms=1, gap_end_ms=2)
        )
    )

    assert report.trades_replayed == 0
    assert any("trades_fetch_failed" in e for e in report.errors)
    # Orderbook leg must still run despite trades failure.
    assert report.orderbook_replayed is True


def test_records_error_when_book_fetch_returns_none() -> None:
    rest = _FakeRest(orderbook=None)
    sink = _RecordingSink()
    service = BackfillService(rest, sink, _resolver)

    report = asyncio.run(
        service.handle_gap(
            GapWindow(asset_id="tok-1", gap_start_ms=1, gap_end_ms=2)
        )
    )

    assert report.orderbook_replayed is False
    assert "book_fetch_returned_none" in report.errors
