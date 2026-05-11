"""Unit tests for :mod:`bigan.ingestion.rollup`."""

from __future__ import annotations

import gzip
from datetime import UTC, datetime
from pathlib import Path

import orjson
import pyarrow.parquet as pq

from bigan.ingestion.rollup import rollup_file


def _write_ndjson_gz(path: Path, records: list[dict]) -> None:
    with gzip.open(path, "wb") as fp:
        for rec in records:
            fp.write(orjson.dumps(rec) + b"\n")


def test_rollup_file_produces_partitioned_parquet(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    rollup_dir = tmp_path / "rollup"
    raw_dir.mkdir()

    src = raw_dir / "2026-05-10.ndjson.gz"
    rt_ms = int(datetime(2026, 5, 10, 12, 0, 0, tzinfo=UTC).timestamp() * 1000)
    _write_ndjson_gz(
        src,
        [
            {
                "receive_time": rt_ms,
                "raw": {
                    "event_type": "book",
                    "asset_id": "1",
                    "market": "m1",
                    "timestamp": str(rt_ms),
                },
            },
            {
                "receive_time": rt_ms + 100,
                "raw": {
                    "event_type": "price_change",
                    "market": "m1",
                    "timestamp": str(rt_ms + 100),
                },
            },
        ],
    )

    n = rollup_file(src, rollup_dir, done_dir=raw_dir / "_done")
    assert n == 2

    book_part = rollup_dir / "date=2026-05-10" / "event_type=book"
    pc_part = rollup_dir / "date=2026-05-10" / "event_type=price_change"
    book_files = list(book_part.glob("*.parquet"))
    pc_files = list(pc_part.glob("*.parquet"))
    assert len(book_files) == 1
    assert len(pc_files) == 1

    book_tbl = pq.ParquetFile(book_files[0]).read()
    assert book_tbl.num_rows == 1
    cols = book_tbl.column_names
    assert "receive_time" in cols
    assert "raw_payload" in cols
    # event_type is in the partition path, not the columns.
    assert "event_type" not in cols

    # Source file must have been moved to _done.
    assert not src.exists()
    assert (raw_dir / "_done" / src.name).exists()


def test_rollup_empty_file_still_archived(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    rollup_dir = tmp_path / "rollup"
    raw_dir.mkdir()
    src = raw_dir / "empty.ndjson.gz"
    with gzip.open(src, "wb"):
        pass  # zero bytes after gzip framing

    n = rollup_file(src, rollup_dir, done_dir=raw_dir / "_done")
    assert n == 0
    assert (raw_dir / "_done" / src.name).exists()
