"""Unit tests for :mod:`bigan.ingestion.sink`."""

from __future__ import annotations

import gzip
from datetime import UTC, datetime
from pathlib import Path

import orjson
import pytest

from bigan.ingestion.sink import NdjsonGzipSink, _utc_date_str


@pytest.fixture
def sink_root(tmp_path: Path) -> Path:
    return tmp_path / "ws_market"


async def test_write_then_flush_creates_partition_file(sink_root: Path) -> None:
    sink = NdjsonGzipSink(sink_root, flush_interval_seconds=60, max_buffer_records=1000)
    rt = int(datetime(2026, 5, 10, 12, 0, 0, tzinfo=UTC).timestamp() * 1000)
    try:
        await sink.write({"receive_time": rt, "raw": {"event_type": "book"}})
        await sink.flush()
    finally:
        await sink.close()

    expected_path = sink_root / "2026-05-10.ndjson.gz"
    assert expected_path.exists()
    with gzip.open(expected_path, "rb") as fp:
        lines = [orjson.loads(line) for line in fp if line.strip()]
    assert len(lines) == 1
    assert lines[0]["receive_time"] == rt
    assert lines[0]["raw"] == {"event_type": "book"}


async def test_write_partitions_by_utc_date(sink_root: Path) -> None:
    sink = NdjsonGzipSink(sink_root, flush_interval_seconds=60, max_buffer_records=1000)
    a = int(datetime(2026, 5, 10, 23, 59, 59, tzinfo=UTC).timestamp() * 1000)
    b = int(datetime(2026, 5, 11, 0, 0, 1, tzinfo=UTC).timestamp() * 1000)
    try:
        await sink.write({"receive_time": a, "raw": {"event_type": "book"}})
        await sink.write({"receive_time": b, "raw": {"event_type": "price_change"}})
        await sink.flush()
    finally:
        await sink.close()

    assert (sink_root / "2026-05-10.ndjson.gz").exists()
    assert (sink_root / "2026-05-11.ndjson.gz").exists()


async def test_auto_flush_on_buffer_full(sink_root: Path) -> None:
    sink = NdjsonGzipSink(sink_root, flush_interval_seconds=60, max_buffer_records=3)
    rt = int(datetime(2026, 5, 10, 12, 0, 0, tzinfo=UTC).timestamp() * 1000)
    try:
        for i in range(3):
            await sink.write({"receive_time": rt, "raw": {"event_type": "book", "i": i}})
        # buffer should have flushed automatically at the 3rd write
    finally:
        await sink.close()

    with gzip.open(sink_root / "2026-05-10.ndjson.gz", "rb") as fp:
        lines = [orjson.loads(line) for line in fp if line.strip()]
    assert len(lines) == 3


async def test_write_after_close_raises(sink_root: Path) -> None:
    sink = NdjsonGzipSink(sink_root, flush_interval_seconds=60, max_buffer_records=1000)
    await sink.close()
    with pytest.raises(RuntimeError):
        await sink.write({"receive_time": 1, "raw": {}})


def test_utc_date_str_correct_for_known_epoch() -> None:
    assert _utc_date_str(0) == "1970-01-01"
    ms = int(datetime(2026, 5, 10, 23, 59, 0, tzinfo=UTC).timestamp() * 1000)
    assert _utc_date_str(ms) == "2026-05-10"
