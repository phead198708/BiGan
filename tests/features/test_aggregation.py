"""Tests for minute-grain features_15m_v1 aggregation."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from bigan.canonical.writer import WarehouseWriter, warehouse_files
from bigan.features.aggregation import aggregate_features_15m_v1, run_feature_batch
from bigan.features.registry import FEATURE_VERSION


def _ts_at(year: int, month: int, day: int, hour: int, minute: int, second: int = 0) -> int:
    return int(
        datetime(year, month, day, hour, minute, second, tzinfo=UTC).timestamp() * 1000
    )


def _identity(ts: int, *, source_symbol: str = "tok-1") -> dict:
    return {
        "ts": ts,
        "message_ts": ts,
        "ingest_ts": ts + 100,
        "source": "polymarket",
        "source_symbol": source_symbol,
        "source_market": "0xmkt",
        "canonical_symbol": "BTC-UP-15M",
        "provenance": "ws",
    }


def _tob(ts: int, bid: float, ask: float) -> dict:
    return {
        **_identity(ts),
        "bid_price": bid,
        "ask_price": ask,
        "spread": ask - bid,
    }


def _depth(ts: int, side: str, level: int, size: float) -> dict:
    price = 0.50 - level * 0.01 if side == "BID" else 0.52 + level * 0.01
    return {
        **_identity(ts),
        "side": side,
        "level": level,
        "price": price,
        "size": size,
        "snapshot_hash": "h0",
    }


def _trade(ts: int, price: float, size: float, side: str) -> dict:
    return {
        **_identity(ts),
        "price": price,
        "size": size,
        "side": side,
        "fee_rate_bps": 0.0,
        "trade_id": f"trade-{ts}-{side}-{size}",
    }


def test_aggregates_microprice_obi_trade_flow_returns_and_rv() -> None:
    t0 = _ts_at(2026, 5, 13, 12, 0)
    t1 = t0 + 60_000
    top_of_book = [
        _tob(t0, 0.49, 0.51),
        _tob(t1, 0.51, 0.53),
        _tob(t1 + 1_000, 0.90, 0.92),  # future relative to the t1 feature row
    ]
    orderbook = []
    for level in range(10):
        orderbook.append(_depth(t1, "BID", level, 100 if level == 0 else 10))
        orderbook.append(_depth(t1, "ASK", level, 50 if level == 0 else 5))
    trades = [
        _trade(t0, 0.50, 100, "BUY"),  # excluded from (t1 - 60s, t1]
        _trade(t1 - 50_000, 0.51, 10, "BUY"),
        _trade(t1 - 10_000, 0.52, 4, "SELL"),
    ]

    rows = aggregate_features_15m_v1(
        top_of_book_rows=top_of_book,
        orderbook_rows=orderbook,
        trade_rows=trades,
        ingest_ts=999,
    )
    row = next(row for row in rows if row["feature_ts"] == t1)

    assert row["feature_version"] == FEATURE_VERSION
    assert row["symbol"] == "BTC-UP-15M"
    assert row["spread"] == pytest.approx(0.02)
    assert row["mid_price"] == pytest.approx(0.52)
    assert row["microprice"] == pytest.approx((0.53 * 100 + 0.51 * 50) / 150)
    assert row["obi_l1"] == pytest.approx((100 - 50) / 150)
    assert row["obi_l5"] == pytest.approx((140 - 70) / 210)
    assert row["obi_l10"] == pytest.approx((190 - 95) / 285)
    assert row["signed_volume_1m"] == pytest.approx(6)
    assert row["trade_volume_1m"] == pytest.approx(14)
    assert row["trade_count_1m"] == 2
    assert row["trade_imbalance_1m"] == pytest.approx(6 / 14)
    assert row["ret_1m"] == pytest.approx(math.log(0.52 / 0.50))
    assert row["rv_1m"] == pytest.approx(abs(math.log(0.52 / 0.50)))


def test_aggregation_emits_minute_close_rows_only_from_backward_inputs() -> None:
    t0 = _ts_at(2026, 5, 13, 12, 0)
    rows = aggregate_features_15m_v1(
        top_of_book_rows=[
            _tob(t0 + 5_000, 0.49, 0.51),
            _tob(t0 + 65_000, 0.80, 0.82),
        ],
        orderbook_rows=[],
        trade_rows=[],
        ingest_ts=999,
    )

    by_ts = {row["feature_ts"]: row for row in rows}

    assert sorted(by_ts) == [t0 + 60_000, t0 + 120_000]
    assert by_ts[t0 + 60_000]["mid_price"] == pytest.approx(0.50)
    assert by_ts[t0 + 120_000]["mid_price"] == pytest.approx(0.81)


def test_run_feature_batch_writes_features_table(tmp_path: Path) -> None:
    warehouse = tmp_path / "warehouse"
    t0 = _ts_at(2026, 5, 13, 12, 0)
    with WarehouseWriter(warehouse, max_rows_per_partition=10) as writer:
        writer.append_rows("raw_top_of_book", [_tob(t0, 0.49, 0.51)])
        writer.append_rows(
            "raw_orderbook_snapshot",
            [_depth(t0, "BID", 0, 10), _depth(t0, "ASK", 0, 5)],
        )
        writer.append_rows("raw_trades", [_trade(t0, 0.50, 3, "BUY")])

    report = run_feature_batch(warehouse, ingest_ts=123)

    assert report.rows_generated == 1
    assert report.rows_written == 1
    files = warehouse_files(warehouse, "features_15m_v1")
    assert len(files) == 1
    row = pq.ParquetFile(files[0]).read().to_pylist()[0]
    assert row["feature_ts"] == t0
    assert row["ts"] == t0
    assert row["message_ts"] == t0
    assert row["symbol"] == "BTC-UP-15M"
    assert row["feature_version"] == FEATURE_VERSION
    assert row["trade_count_1m"] == 1
