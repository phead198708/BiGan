"""End-to-end test for the canonical ETL runner.

Builds a synthetic NDJSON.gz archive that contains one of every relevant
event type, runs :func:`run_etl_batch`, and asserts that all four canonical
tables receive rows.
"""

from __future__ import annotations

import gzip
from datetime import UTC, datetime
from pathlib import Path

import orjson
import pyarrow.parquet as pq

from bigan.canonical.etl import run_etl_batch
from bigan.canonical.writer import warehouse_files


def _ts(year: int, month: int, day: int, hour: int, minute: int, second: int = 0) -> int:
    return int(
        datetime(year, month, day, hour, minute, second, tzinfo=UTC).timestamp() * 1000
    )


def _write_ndjson_gz(path: Path, records: list[dict]) -> None:
    with gzip.open(path, "wb") as fp:
        for rec in records:
            fp.write(orjson.dumps(rec) + b"\n")


def test_etl_round_trip_populates_all_four_tables(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    base = _ts(2026, 5, 10, 12, 0)
    src = raw_dir / "2026-05-10.ndjson.gz"
    _write_ndjson_gz(
        src,
        [
            {
                "receive_time": base + 100,
                "raw": {
                    "event_type": "book",
                    "asset_id": "tok-1",
                    "market": "0xmkt",
                    "timestamp": str(base),
                    "hash": "h0",
                    "bids": [
                        {"price": "0.49", "size": "100"},
                        {"price": "0.50", "size": "50"},
                    ],
                    "asks": [
                        {"price": "0.52", "size": "30"},
                        {"price": "0.53", "size": "60"},
                    ],
                },
            },
            {
                "receive_time": base + 200,
                "raw": {
                    "event_type": "best_bid_ask",
                    "asset_id": "tok-1",
                    "market": "0xmkt",
                    "best_bid": "0.50",
                    "best_ask": "0.52",
                    "spread": "0.02",
                    "timestamp": str(base + 100),
                },
            },
            {
                "receive_time": base + 30_000,
                "raw": {
                    "event_type": "best_bid_ask",
                    "asset_id": "tok-1",
                    "market": "0xmkt",
                    "best_bid": "0.51",
                    "best_ask": "0.53",
                    "spread": "0.02",
                    "timestamp": str(base + 29_000),
                },
            },
            {
                "receive_time": base + 35_000,
                "raw": {
                    "event_type": "last_trade_price",
                    "asset_id": "tok-1",
                    "market": "0xmkt",
                    "price": "0.51",
                    "size": "10",
                    "side": "BUY",
                    "fee_rate_bps": "0",
                    "timestamp": str(base + 34_000),
                },
            },
            # Event types we don't materialise should be silently ignored.
            {
                "receive_time": base + 40_000,
                "raw": {
                    "event_type": "tick_size_change",
                    "asset_id": "tok-1",
                    "market": "0xmkt",
                    "old_tick_size": "0.01",
                    "new_tick_size": "0.001",
                    "timestamp": str(base + 39_000),
                },
            },
        ],
    )

    # Backdate file mtime so the lag-seconds gate doesn't skip it.
    import os

    os.utime(src, (1, 1))

    warehouse = tmp_path / "warehouse"
    report = run_etl_batch(
        raw_dir=raw_dir, warehouse_dir=warehouse, lag_seconds=0.0
    )
    assert report.files_processed == 1
    assert report.records_read == 5
    # 2 explicit best_bid_ask events + 1 derived from the book snapshot.
    assert report.rows_per_table["raw_top_of_book"] == 3
    assert report.rows_per_table["raw_orderbook_snapshot"] == 4  # 2 bids + 2 asks
    assert report.rows_per_table["raw_trades"] == 1
    # 1 trade in same minute as 1 tob, plus another tob in next minute later? No —
    # second tob at base+29s is still in bucket 0 (minute 0). All in bucket 0.
    # Plus trade at base+34s is also bucket 0. So 1 candle.
    assert report.rows_per_table["raw_candles_1m"] == 1

    # Spot-check raw_trades parquet
    trade_files = warehouse_files(warehouse, "raw_trades")
    assert len(trade_files) == 1
    tbl = pq.ParquetFile(trade_files[0]).read()
    assert tbl.num_rows == 1
    assert tbl.column("price").to_pylist() == [0.51]

    # Spot-check raw_candles_1m parquet has both quote and trade fields populated
    cand_files = warehouse_files(warehouse, "raw_candles_1m")
    assert len(cand_files) == 1
    cand = pq.ParquetFile(cand_files[0]).read().to_pylist()[0]
    assert cand["trade_count"] == 1
    assert cand["top_of_book_count"] == 3
    assert cand["bid_open"] == 0.50
    assert cand["bid_close"] == 0.51
    assert cand["trade_close"] == 0.51
    assert cand["vwap"] == 0.51


def test_etl_skips_in_flight_files(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    src = raw_dir / "in-flight.ndjson.gz"
    with gzip.open(src, "wb") as fp:
        fp.write(b"")  # empty
    # Don't backdate mtime: with a positive lag, the file should be skipped.
    warehouse = tmp_path / "warehouse"
    report = run_etl_batch(
        raw_dir=raw_dir, warehouse_dir=warehouse, lag_seconds=3600.0
    )
    assert report.files_processed == 0
    assert report.records_read == 0
