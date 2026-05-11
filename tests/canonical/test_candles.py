"""Unit tests for :mod:`bigan.canonical.candles`."""

from __future__ import annotations

from datetime import UTC, datetime

from bigan.canonical.candles import BUCKET_MS, aggregate_1m_candles


def _ts_at(year: int, month: int, day: int, hour: int, minute: int, second: int = 0) -> int:
    return int(
        datetime(year, month, day, hour, minute, second, tzinfo=UTC).timestamp() * 1000
    )


def _tob(ts: int, bid: float, ask: float, *, source_symbol: str = "1") -> dict:
    return {
        "ts": ts,
        "source": "polymarket",
        "source_symbol": source_symbol,
        "source_market": "0xmkt",
        "canonical_symbol": None,
        "bid_price": bid,
        "ask_price": ask,
    }


def _trade(ts: int, price: float, size: float, *, source_symbol: str = "1") -> dict:
    return {
        "ts": ts,
        "source": "polymarket",
        "source_symbol": source_symbol,
        "source_market": "0xmkt",
        "canonical_symbol": None,
        "price": price,
        "size": size,
    }


def test_single_bucket_full_ohlc() -> None:
    base = _ts_at(2026, 5, 10, 12, 0)  # bucket start
    tob_rows = [
        _tob(base + 1_000, 0.50, 0.52),  # open
        _tob(base + 5_000, 0.52, 0.54),  # high
        _tob(base + 10_000, 0.49, 0.51),  # low (mid 0.50)
        _tob(base + 30_000, 0.51, 0.53),  # close
    ]
    trades = [
        _trade(base + 2_000, 0.51, 100),
        _trade(base + 20_000, 0.53, 50),
    ]
    candles = aggregate_1m_candles(tob_rows, trades, ingest_ts=999)
    assert len(candles) == 1
    c = candles[0]
    assert c["bucket_ts"] == base
    assert c["ts_open"] == base + 1_000
    assert c["ts_close"] == base + 30_000
    assert c["bid_open"] == 0.50
    assert c["bid_close"] == 0.51
    assert c["bid_high"] == 0.52
    assert c["bid_low"] == 0.49
    assert c["ask_open"] == 0.52
    assert c["ask_close"] == 0.53
    # mid_open = (0.50+0.52)/2 = 0.51
    assert abs(c["mid_open"] - 0.51) < 1e-9
    assert c["trade_count"] == 2
    assert c["trade_volume"] == 150
    assert c["trade_open"] == 0.51
    assert c["trade_close"] == 0.53
    assert c["trade_high"] == 0.53
    assert c["trade_low"] == 0.51
    # vwap = (0.51*100 + 0.53*50) / 150 = (51 + 26.5)/150 = 0.5166666...
    assert abs(c["vwap"] - (0.51 * 100 + 0.53 * 50) / 150) < 1e-9
    assert c["top_of_book_count"] == 4


def test_multiple_buckets_separated_correctly() -> None:
    a = _ts_at(2026, 5, 10, 12, 0)
    b = a + BUCKET_MS  # next minute
    rows = [
        _tob(a + 5_000, 0.50, 0.52),
        _tob(a + 30_000, 0.51, 0.53),
        _tob(b + 5_000, 0.55, 0.57),
    ]
    candles = aggregate_1m_candles(rows, [], ingest_ts=0)
    bucket_to_candle = {c["bucket_ts"]: c for c in candles}
    assert set(bucket_to_candle) == {a, b}
    assert bucket_to_candle[a]["bid_close"] == 0.51
    assert bucket_to_candle[b]["bid_open"] == 0.55


def test_quiet_minute_with_only_trades() -> None:
    """Trade-only minutes still emit a candle with NULL quote columns."""
    base = _ts_at(2026, 5, 10, 12, 0)
    candles = aggregate_1m_candles(
        [], [_trade(base + 30_000, 0.6, 10.0)], ingest_ts=0
    )
    assert len(candles) == 1
    c = candles[0]
    assert c["bid_open"] is None
    assert c["mid_open"] is None
    assert c["trade_volume"] == 10.0
    assert c["trade_count"] == 1


def test_multiple_symbols_kept_separate() -> None:
    base = _ts_at(2026, 5, 10, 12, 0)
    rows = [
        _tob(base + 1_000, 0.50, 0.52, source_symbol="A"),
        _tob(base + 1_000, 0.40, 0.42, source_symbol="B"),
    ]
    candles = aggregate_1m_candles(rows, [], ingest_ts=0)
    syms = sorted(c["source_symbol"] for c in candles)
    assert syms == ["A", "B"]


def test_empty_input_yields_empty_output() -> None:
    assert aggregate_1m_candles([], [], ingest_ts=0) == []
