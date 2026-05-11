"""Unit tests for :mod:`bigan.canonical.writer`."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from bigan.canonical.writer import WarehouseWriter, warehouse_files


def _ts_at(year: int, month: int, day: int, hour: int = 12) -> int:
    return int(datetime(year, month, day, hour, 0, 0, tzinfo=UTC).timestamp() * 1000)


def test_writer_creates_partitioned_parquet(tmp_path: Path) -> None:
    rows = [
        {
            "ts": _ts_at(2026, 5, 10),
            "message_ts": _ts_at(2026, 5, 10),
            "ingest_ts": 0,
            "source": "polymarket",
            "source_symbol": "1",
            "source_market": "m1",
            "canonical_symbol": None,
            "bid_price": 0.5,
            "ask_price": 0.55,
            "spread": 0.05,
        }
    ]
    with WarehouseWriter(tmp_path) as w:
        w.append_rows("raw_top_of_book", rows)

    parts = warehouse_files(tmp_path, "raw_top_of_book")
    assert len(parts) == 1
    expected_dir = tmp_path / "raw_top_of_book" / "source=polymarket" / "dt=2026-05-10"
    assert parts[0].parent == expected_dir

    tbl = pq.ParquetFile(parts[0]).read()
    assert tbl.num_rows == 1
    assert tbl.column("bid_price").to_pylist() == [0.5]


def test_writer_partitions_by_utc_date(tmp_path: Path) -> None:
    rows = [
        {
            "ts": _ts_at(2026, 5, 10, hour=23),
            "message_ts": _ts_at(2026, 5, 10, hour=23),
            "ingest_ts": 0,
            "source": "polymarket",
            "source_symbol": "1",
            "bid_price": 0.5,
            "ask_price": 0.55,
            "spread": 0.05,
        },
        {
            "ts": _ts_at(2026, 5, 11, hour=0),
            "message_ts": _ts_at(2026, 5, 11, hour=0),
            "ingest_ts": 0,
            "source": "polymarket",
            "source_symbol": "1",
            "bid_price": 0.6,
            "ask_price": 0.65,
            "spread": 0.05,
        },
    ]
    with WarehouseWriter(tmp_path) as w:
        w.append_rows("raw_top_of_book", rows)

    files = warehouse_files(tmp_path, "raw_top_of_book")
    dirs = {f.parent.name for f in files}
    assert dirs == {"dt=2026-05-10", "dt=2026-05-11"}


def test_writer_flushes_when_partition_buffer_full(tmp_path: Path) -> None:
    rows = []
    for i in range(5):
        rows.append(
            {
                "ts": _ts_at(2026, 5, 10),
                "message_ts": _ts_at(2026, 5, 10),
                "ingest_ts": i,
                "source": "polymarket",
                "source_symbol": "1",
                "bid_price": 0.5,
                "ask_price": 0.55,
                "spread": 0.05,
            }
        )
    with WarehouseWriter(tmp_path, max_rows_per_partition=2) as w:
        w.append_rows("raw_top_of_book", rows)

    files = warehouse_files(tmp_path, "raw_top_of_book")
    # 5 rows / 2 max = ceil(5/2) = 3 files (auto-flush at 2,4 and final on close)
    assert len(files) == 3
    total_rows = sum(pq.ParquetFile(f).read().num_rows for f in files)
    assert total_rows == 5


def test_writer_unknown_table_raises(tmp_path: Path) -> None:
    with (
        WarehouseWriter(tmp_path) as w,
        pytest.raises(ValueError, match="unknown canonical table"),
    ):
        w.append_rows("not_a_real_table", [{"ts": 1}])


def test_writer_skips_rows_missing_required_fields(tmp_path: Path, caplog) -> None:
    bad_rows = [
        {"ts": None, "source": "polymarket", "source_symbol": "1"},
        {"ts": 1, "source": None, "source_symbol": "1"},
    ]
    with WarehouseWriter(tmp_path) as w:
        w.append_rows("raw_top_of_book", bad_rows)
    assert warehouse_files(tmp_path, "raw_top_of_book") == []


def test_writer_drops_extra_columns(tmp_path: Path) -> None:
    rows = [
        {
            "ts": _ts_at(2026, 5, 10),
            "message_ts": _ts_at(2026, 5, 10),
            "ingest_ts": 0,
            "source": "polymarket",
            "source_symbol": "1",
            "bid_price": 0.5,
            "ask_price": 0.55,
            "spread": 0.05,
            "this_field_does_not_exist": "should_be_dropped",
        }
    ]
    with WarehouseWriter(tmp_path) as w:
        w.append_rows("raw_top_of_book", rows)
    parts = warehouse_files(tmp_path, "raw_top_of_book")
    cols = pq.ParquetFile(parts[0]).schema_arrow.names
    assert "this_field_does_not_exist" not in cols
