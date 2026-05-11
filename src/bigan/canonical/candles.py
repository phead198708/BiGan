"""1-minute candle aggregation.

Reads canonical ``raw_top_of_book`` and ``raw_trades`` rows and emits
``raw_candles_1m`` rows. The aggregation grain is
``(source, source_symbol, bucket_ts)`` where ``bucket_ts`` is the start of
a UTC minute (``floor(ts / 60_000) * 60_000`` ms).

Within each bucket:

- ``bid_*`` / ``ask_*`` / ``mid_*`` OHLC come from the ordered sequence of
  ``raw_top_of_book`` rows (open = first, close = last, high/low =
  extrema). ``mid = (bid + ask) / 2`` is computed per-row when both sides
  are present.
- ``trade_*`` OHLC come from the ordered sequence of ``raw_trades`` rows.
- ``trade_volume`` = sum of ``size``.
- ``trade_count`` = number of trade rows.
- ``top_of_book_count`` = number of ToB rows.
- ``vwap`` = ``sum(price * size) / sum(size)``; NULL if no trades.
- ``ts_open`` / ``ts_close`` = first / last *any* event timestamp inside
  the bucket so consumers can detect quiet minutes.

Buckets with zero events are not emitted; downstream gap-detection (#8) is
responsible for surfacing missing minutes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

BUCKET_MS = 60_000


def _bucket_ts(ts_ms: int) -> int:
    return (ts_ms // BUCKET_MS) * BUCKET_MS


def _grouping_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (str(row["source"]), str(row["source_symbol"]))


def _bucket_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    source, source_symbol = _grouping_key(row)
    return (source, source_symbol, _bucket_ts(int(row["ts"])))


@dataclass(slots=True)
class _BucketState:
    """Mutable accumulator for a single (source, symbol, bucket_ts)."""

    source: str
    source_symbol: str
    source_market: str | None = None
    canonical_symbol: str | None = None
    bucket_ts: int = 0

    ts_open: int | None = None
    ts_close: int | None = None

    bid_open: float | None = None
    bid_high: float | None = None
    bid_low: float | None = None
    bid_close: float | None = None

    ask_open: float | None = None
    ask_high: float | None = None
    ask_low: float | None = None
    ask_close: float | None = None

    mid_open: float | None = None
    mid_high: float | None = None
    mid_low: float | None = None
    mid_close: float | None = None

    trade_open: float | None = None
    trade_high: float | None = None
    trade_low: float | None = None
    trade_close: float | None = None
    trade_volume: float = 0.0
    trade_count: int = 0
    _vwap_num: float = 0.0
    _vwap_denom: float = 0.0

    top_of_book_count: int = 0

    # Track the latest ts seen on each side so we can decide which OHLC
    # to update; we receive events in ts-ascending order (caller sorts).

    def update_market(self, source_market: str | None, canonical: str | None) -> None:
        if source_market is not None:
            self.source_market = source_market
        if canonical is not None:
            self.canonical_symbol = canonical

    def observe_ts(self, ts: int) -> None:
        if self.ts_open is None or ts < self.ts_open:
            self.ts_open = ts
        if self.ts_close is None or ts > self.ts_close:
            self.ts_close = ts

    def observe_top_of_book(
        self, *, ts: int, bid: float | None, ask: float | None
    ) -> None:
        self.top_of_book_count += 1
        self.observe_ts(ts)
        if bid is not None:
            if self.bid_open is None:
                self.bid_open = bid
            if self.bid_high is None or bid > self.bid_high:
                self.bid_high = bid
            if self.bid_low is None or bid < self.bid_low:
                self.bid_low = bid
            self.bid_close = bid
        if ask is not None:
            if self.ask_open is None:
                self.ask_open = ask
            if self.ask_high is None or ask > self.ask_high:
                self.ask_high = ask
            if self.ask_low is None or ask < self.ask_low:
                self.ask_low = ask
            self.ask_close = ask
        if bid is not None and ask is not None:
            mid = (bid + ask) / 2.0
            if self.mid_open is None:
                self.mid_open = mid
            if self.mid_high is None or mid > self.mid_high:
                self.mid_high = mid
            if self.mid_low is None or mid < self.mid_low:
                self.mid_low = mid
            self.mid_close = mid

    def observe_trade(self, *, ts: int, price: float, size: float) -> None:
        self.trade_count += 1
        self.trade_volume += size
        self._vwap_num += price * size
        self._vwap_denom += size
        self.observe_ts(ts)
        if self.trade_open is None:
            self.trade_open = price
        if self.trade_high is None or price > self.trade_high:
            self.trade_high = price
        if self.trade_low is None or price < self.trade_low:
            self.trade_low = price
        self.trade_close = price

    def to_row(self, *, ingest_ts: int) -> dict[str, Any]:
        vwap = self._vwap_num / self._vwap_denom if self._vwap_denom > 0 else None
        return {
            "ts": self.bucket_ts,
            "message_ts": self.bucket_ts,
            "ingest_ts": ingest_ts,
            "source": self.source,
            "source_symbol": self.source_symbol,
            "source_market": self.source_market,
            "canonical_symbol": self.canonical_symbol,
            "bucket_ts": self.bucket_ts,
            "ts_open": self.ts_open,
            "ts_close": self.ts_close,
            "bid_open": self.bid_open,
            "bid_high": self.bid_high,
            "bid_low": self.bid_low,
            "bid_close": self.bid_close,
            "ask_open": self.ask_open,
            "ask_high": self.ask_high,
            "ask_low": self.ask_low,
            "ask_close": self.ask_close,
            "mid_open": self.mid_open,
            "mid_high": self.mid_high,
            "mid_low": self.mid_low,
            "mid_close": self.mid_close,
            "trade_open": self.trade_open,
            "trade_high": self.trade_high,
            "trade_low": self.trade_low,
            "trade_close": self.trade_close,
            "trade_volume": self.trade_volume if self.trade_count > 0 else None,
            "trade_count": self.trade_count,
            "top_of_book_count": self.top_of_book_count,
            "vwap": vwap,
        }


@dataclass(slots=True)
class CandleAggregator:
    """Streaming aggregator for 1-minute candles.

    Rows can be added in any order; the final output is sorted by
    ``(source, source_symbol, bucket_ts)``. Within a bucket, OHLC values
    rely on rows being fed in **ts-ascending order**; callers should sort
    upstream (the ETL runner does so).
    """

    _buckets: dict[tuple[str, str, int], _BucketState] = field(default_factory=dict)

    def add_top_of_book(self, row: Mapping[str, Any]) -> None:
        bucket = self._get_bucket(row)
        bucket.update_market(row.get("source_market"), row.get("canonical_symbol"))
        bucket.observe_top_of_book(
            ts=int(row["ts"]),
            bid=_as_float(row.get("bid_price")),
            ask=_as_float(row.get("ask_price")),
        )

    def add_trade(self, row: Mapping[str, Any]) -> None:
        price = _as_float(row.get("price"))
        size = _as_float(row.get("size"))
        if price is None or size is None:
            return
        bucket = self._get_bucket(row)
        bucket.update_market(row.get("source_market"), row.get("canonical_symbol"))
        bucket.observe_trade(ts=int(row["ts"]), price=price, size=size)

    def emit(self, *, ingest_ts: int) -> list[dict[str, Any]]:
        rows = [b.to_row(ingest_ts=ingest_ts) for b in self._buckets.values()]
        rows.sort(key=lambda r: (r["source"], r["source_symbol"], r["bucket_ts"]))
        return rows

    def _get_bucket(self, row: Mapping[str, Any]) -> _BucketState:
        key = _bucket_key(row)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _BucketState(
                source=key[0],
                source_symbol=key[1],
                bucket_ts=key[2],
            )
            self._buckets[key] = bucket
        return bucket


def aggregate_1m_candles(
    top_of_book_rows: Iterable[Mapping[str, Any]],
    trades_rows: Iterable[Mapping[str, Any]],
    *,
    ingest_ts: int,
) -> list[dict[str, Any]]:
    """One-shot helper: aggregate two row streams into 1-minute candles.

    Both streams must already be sorted by ``ts`` ascending within each
    ``(source, source_symbol)`` group. Cross-group ordering is irrelevant.
    """
    agg = CandleAggregator()
    for row in top_of_book_rows:
        agg.add_top_of_book(row)
    for row in trades_rows:
        agg.add_trade(row)
    return agg.emit(ingest_ts=ingest_ts)


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
