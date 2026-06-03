"""Unit tests for :mod:`bigan.ingestion.sink`."""

from __future__ import annotations

import asyncio
import gzip
from datetime import UTC, datetime
from pathlib import Path

import orjson
import pytest

from bigan.ingestion.sink import NdjsonGzipSink, _utc_date_str


@pytest.fixture
def sink_root(tmp_path: Path) -> Path:
    return tmp_path / "ws_market"


def test_write_then_flush_creates_partition_file(sink_root: Path) -> None:
    rt = int(datetime(2026, 5, 10, 12, 0, 0, tzinfo=UTC).timestamp() * 1000)

    async def go() -> None:
        sink = NdjsonGzipSink(sink_root, flush_interval_seconds=60, max_buffer_records=1000)
        try:
            await sink.write({"receive_time": rt, "raw": {"event_type": "book"}})
            await sink.flush()
        finally:
            await sink.close()

    asyncio.run(go())

    expected_path = sink_root / "2026-05-10.ndjson.gz"
    assert expected_path.exists()
    with gzip.open(expected_path, "rb") as fp:
        lines = [orjson.loads(line) for line in fp if line.strip()]
    assert len(lines) == 1
    assert lines[0]["receive_time"] == rt
    assert lines[0]["raw"] == {"event_type": "book"}


def test_write_partitions_by_utc_date(sink_root: Path) -> None:
    a = int(datetime(2026, 5, 10, 23, 59, 59, tzinfo=UTC).timestamp() * 1000)
    b = int(datetime(2026, 5, 11, 0, 0, 1, tzinfo=UTC).timestamp() * 1000)

    async def go() -> None:
        sink = NdjsonGzipSink(sink_root, flush_interval_seconds=60, max_buffer_records=1000)
        try:
            await sink.write({"receive_time": a, "raw": {"event_type": "book"}})
            await sink.write({"receive_time": b, "raw": {"event_type": "price_change"}})
            await sink.flush()
        finally:
            await sink.close()

    asyncio.run(go())

    assert (sink_root / "2026-05-10.ndjson.gz").exists()
    assert (sink_root / "2026-05-11.ndjson.gz").exists()


def test_segmented_sink_writes_time_bucketed_files(sink_root: Path) -> None:
    a = int(datetime(2026, 5, 10, 12, 0, 5, tzinfo=UTC).timestamp() * 1000)
    b = int(datetime(2026, 5, 10, 12, 1, 5, tzinfo=UTC).timestamp() * 1000)

    async def go() -> None:
        sink = NdjsonGzipSink(
            sink_root,
            flush_interval_seconds=60,
            max_buffer_records=1000,
            segment_duration_seconds=60,
        )
        try:
            await sink.write({"receive_time": a, "raw": {"event_type": "book"}})
            await sink.write({"receive_time": b, "raw": {"event_type": "price_change"}})
            await sink.flush()
        finally:
            await sink.close()

    asyncio.run(go())

    assert (sink_root / "2026-05-10T120000Z.ndjson.gz").exists()
    assert (sink_root / "2026-05-10T120100Z.ndjson.gz").exists()
    assert not (sink_root / "2026-05-10.ndjson.gz").exists()


def test_segmented_sink_publishes_only_rotated_segments_before_close(
    sink_root: Path,
) -> None:
    a = int(datetime(2026, 5, 10, 12, 0, 5, tzinfo=UTC).timestamp() * 1000)
    b = int(datetime(2026, 5, 10, 12, 1, 5, tzinfo=UTC).timestamp() * 1000)

    async def go() -> None:
        sink = NdjsonGzipSink(
            sink_root,
            flush_interval_seconds=60,
            max_buffer_records=1000,
            segment_duration_seconds=60,
        )
        try:
            await sink.write({"receive_time": a, "raw": {"i": 1}})
            await sink.flush()
            assert not (sink_root / "2026-05-10T120000Z.ndjson.gz").exists()
            assert _tmp_files(sink_root)

            await sink.write({"receive_time": b, "raw": {"i": 2}})
            await sink.flush()
            assert (sink_root / "2026-05-10T120000Z.ndjson.gz").exists()
            assert not (sink_root / "2026-05-10T120100Z.ndjson.gz").exists()
        finally:
            await sink.close()

    asyncio.run(go())

    with gzip.open(sink_root / "2026-05-10T120000Z.ndjson.gz", "rb") as fp:
        first_segment = [orjson.loads(line) for line in fp if line.strip()]
    with gzip.open(sink_root / "2026-05-10T120100Z.ndjson.gz", "rb") as fp:
        second_segment = [orjson.loads(line) for line in fp if line.strip()]

    assert [row["raw"]["i"] for row in first_segment] == [1]
    assert [row["raw"]["i"] for row in second_segment] == [2]
    assert not _tmp_files(sink_root)


def test_segmented_sink_merges_late_records_into_published_segment(
    sink_root: Path,
) -> None:
    a = int(datetime(2026, 5, 10, 12, 0, 5, tzinfo=UTC).timestamp() * 1000)

    async def write_once(value: int) -> None:
        sink = NdjsonGzipSink(
            sink_root,
            flush_interval_seconds=60,
            max_buffer_records=1000,
            segment_duration_seconds=60,
        )
        try:
            await sink.write({"receive_time": a, "raw": {"i": value}})
        finally:
            await sink.close()

    asyncio.run(write_once(1))
    asyncio.run(write_once(2))

    with gzip.open(sink_root / "2026-05-10T120000Z.ndjson.gz", "rb") as fp:
        lines = [orjson.loads(line) for line in fp if line.strip()]

    assert [row["raw"]["i"] for row in lines] == [1, 2]


def test_auto_flush_on_buffer_full(sink_root: Path) -> None:
    rt = int(datetime(2026, 5, 10, 12, 0, 0, tzinfo=UTC).timestamp() * 1000)

    async def go() -> None:
        sink = NdjsonGzipSink(sink_root, flush_interval_seconds=60, max_buffer_records=3)
        try:
            for i in range(3):
                await sink.write({"receive_time": rt, "raw": {"event_type": "book", "i": i}})
            # buffer should have flushed automatically at the 3rd write
        finally:
            await sink.close()

    asyncio.run(go())

    with gzip.open(sink_root / "2026-05-10.ndjson.gz", "rb") as fp:
        lines = [orjson.loads(line) for line in fp if line.strip()]
    assert len(lines) == 3


def test_flush_file_is_readable_before_sink_close(sink_root: Path) -> None:
    rt = int(datetime(2026, 5, 10, 12, 0, 0, tzinfo=UTC).timestamp() * 1000)

    async def go() -> tuple[list[dict], list[dict]]:
        sink = NdjsonGzipSink(sink_root, flush_interval_seconds=60, max_buffer_records=1000)
        try:
            await sink.write({"receive_time": rt, "raw": {"event_type": "book", "i": 1}})
            await sink.flush()

            with gzip.open(sink_root / "2026-05-10.ndjson.gz", "rb") as fp:
                first_lines = [orjson.loads(line) for line in fp if line.strip()]

            await sink.write({"receive_time": rt, "raw": {"event_type": "book", "i": 2}})
            await sink.flush()

            with gzip.open(sink_root / "2026-05-10.ndjson.gz", "rb") as fp:
                all_lines = [orjson.loads(line) for line in fp if line.strip()]
        finally:
            await sink.close()
        return first_lines, all_lines

    first_lines, all_lines = asyncio.run(go())

    assert [row["raw"]["i"] for row in first_lines] == [1]
    assert [row["raw"]["i"] for row in all_lines] == [1, 2]


def test_write_after_close_raises(sink_root: Path) -> None:
    async def go() -> None:
        sink = NdjsonGzipSink(sink_root, flush_interval_seconds=60, max_buffer_records=1000)
        await sink.close()
        with pytest.raises(RuntimeError):
            await sink.write({"receive_time": 1, "raw": {}})

    asyncio.run(go())


def test_utc_date_str_correct_for_known_epoch() -> None:
    assert _utc_date_str(0) == "1970-01-01"
    ms = int(datetime(2026, 5, 10, 23, 59, 0, tzinfo=UTC).timestamp() * 1000)
    assert _utc_date_str(ms) == "2026-05-10"


def _tmp_files(root: Path) -> list[Path]:
    return list(root.glob("*.tmp"))
