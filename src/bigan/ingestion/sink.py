"""Raw-message persistence sinks.

The default :class:`NdjsonGzipSink` writes one gzip-compressed NDJSON file per
UTC date under ``<data_dir>/<raw_subdir>/YYYY-MM-DD.ndjson.gz``. Records are
buffered in memory and flushed on a timer (or when the buffer hits
``max_buffer_records``). Non-segmented files append and close a complete gzip
member on each flush. Segmented files are first written under a temporary name
and only atomically published to ``*.ndjson.gz`` after the segment rotates or
the sink closes, so readers never see half-written gzip footers.

The :class:`Sink` Protocol exists so issue #3 (canonical DB schema) can plug
in a Postgres / TimescaleDB sink without changing the WS client.

Each record is a JSON object containing:

- ``receive_time``: ms epoch of arrival on our side
- ``raw``: the verbatim payload returned by the CLOB server

The verbatim payload is intentionally preserved as a black-box archive so any
schema reinterpretation later (issue #6 / #4) can re-parse historical data.
"""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import logging
import os
import shutil
import threading
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import orjson

from .metrics import SINK_FLUSH_SECONDS, SINK_RECORDS_WRITTEN_TOTAL

logger = logging.getLogger(__name__)


class Sink(Protocol):
    """Async sink interface for raw market-channel payloads."""

    async def write(self, record: Mapping[str, Any]) -> None: ...

    async def flush(self) -> None: ...

    async def close(self) -> None: ...


def _utc_date_str(epoch_ms: int) -> str:
    return datetime.fromtimestamp(epoch_ms / 1000, tz=UTC).strftime("%Y-%m-%d")


def _utc_segment_str(epoch_ms: int, segment_duration_seconds: int) -> str:
    segment_ms = segment_duration_seconds * 1_000
    bucket_ms = (epoch_ms // segment_ms) * segment_ms
    return datetime.fromtimestamp(bucket_ms / 1000, tz=UTC).strftime("%Y-%m-%dT%H%M%SZ")


class NdjsonGzipSink:
    """Buffered date-partitioned gzip NDJSON sink.

    Thread-safety: an ``asyncio.Lock`` serialises buffer mutations and flushes
    so the sink can be shared by concurrent producers (Gamma poller errors,
    metric exporters, ...).
    """

    def __init__(
        self,
        root: Path,
        *,
        flush_interval_seconds: float = 2.0,
        max_buffer_records: int = 1000,
        segment_duration_seconds: int = 0,
    ) -> None:
        if segment_duration_seconds < 0:
            raise ValueError("segment_duration_seconds must be non-negative")
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._flush_interval = flush_interval_seconds
        self._max_buffer = max_buffer_records
        self._segment_duration_seconds = segment_duration_seconds
        self._buffer: list[bytes] = []
        self._buffer_bytes = 0
        self._lock = asyncio.Lock()
        self._flusher_task: asyncio.Task[None] | None = None
        self._closed = False
        self._last_flush = time.monotonic()
        self._flush_thread_lock = threading.Lock()
        self._segment_tmp_paths: dict[Path, Path] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def write(self, record: Mapping[str, Any]) -> None:
        """Buffer one record; flush opportunistically."""
        if self._closed:
            raise RuntimeError("sink is closed")
        line = orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE)
        async with self._lock:
            self._buffer.append(line)
            self._buffer_bytes += len(line)
            should_flush = len(self._buffer) >= self._max_buffer
        if should_flush:
            await self.flush()

    async def flush(self) -> None:
        await self._flush(publish_all=False)

    async def _flush(self, *, publish_all: bool) -> None:
        async with self._lock:
            if not self._buffer:
                if publish_all and self._segment_tmp_paths:
                    await asyncio.to_thread(self._publish_segment_tmp_paths, True)
                return
            buffer, self._buffer = self._buffer, []
            self._buffer_bytes = 0

        start = time.monotonic()
        await asyncio.to_thread(self._flush_blocking, buffer, publish_all)
        SINK_RECORDS_WRITTEN_TOTAL.inc(len(buffer))
        SINK_FLUSH_SECONDS.observe(time.monotonic() - start)
        self._last_flush = time.monotonic()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._flusher_task is not None:
            self._flusher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._flusher_task
            self._flusher_task = None
        await self._flush(publish_all=True)

    async def start_background_flusher(self) -> None:
        """Spawn a background task that flushes on a fixed interval.

        Idempotent: a second call is a no-op.
        """
        if self._flusher_task is not None:
            return
        self._flusher_task = asyncio.create_task(self._flusher_loop(), name="sink-flusher")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _flusher_loop(self) -> None:
        try:
            while not self._closed:
                await asyncio.sleep(self._flush_interval)
                try:
                    await self.flush()
                except Exception:  # noqa: BLE001
                    logger.exception("sink.flush_failed")
        except asyncio.CancelledError:
            raise

    def _flush_blocking(self, buffer: list[bytes], publish_all: bool) -> None:
        # Group records by UTC date partition via the first 4 bytes pattern.
        # Cheaper: just look for ``receive_time`` field in JSON; but simplest
        # is to re-parse the small subset of bytes needed.
        with self._flush_thread_lock:
            groups: dict[Path, list[bytes]] = {}
            for line in buffer:
                try:
                    rec = orjson.loads(line)
                    rt = int(rec.get("receive_time") or rec.get("ts") or 0)
                except (orjson.JSONDecodeError, TypeError, ValueError):
                    rt = int(time.time() * 1000)
                path = self._path_for_receive_time(rt)
                groups.setdefault(path, []).append(line)

            for path, lines in groups.items():
                try:
                    if self._segment_duration_seconds > 0:
                        self._write_segment_tmp(path, lines)
                    else:
                        self._append_gzip_member(path, lines)
                except Exception:  # noqa: BLE001
                    logger.exception("sink.flush_file_failed", extra={"path": str(path)})
                    raise

            self._publish_segment_tmp_paths(publish_all)

    def _append_gzip_member(self, path: Path, lines: list[bytes]) -> None:
        # ``ab`` appends a new gzip member. Closing every flush makes the
        # file readable by live ETL without waiting for process shutdown.
        with gzip.open(path, mode="ab") as fp:
            for line in lines:
                fp.write(line)

    def _write_segment_tmp(self, final_path: Path, lines: list[bytes]) -> None:
        tmp_path = self._segment_tmp_paths.get(final_path)
        if tmp_path is None:
            tmp_path = final_path.with_name(
                f".{final_path.name}.{os.getpid()}.tmp"
            )
            self._segment_tmp_paths[final_path] = tmp_path
        with gzip.open(tmp_path, mode="ab") as fp:
            for line in lines:
                fp.write(line)

    def _publish_segment_tmp_paths(self, publish_all: bool) -> None:
        if self._segment_duration_seconds <= 0 or not self._segment_tmp_paths:
            return

        latest_path = max(self._segment_tmp_paths)
        for final_path, tmp_path in list(self._segment_tmp_paths.items()):
            if not publish_all and final_path == latest_path:
                continue
            self._publish_segment_tmp(final_path, tmp_path)
            self._segment_tmp_paths.pop(final_path, None)

    def _publish_segment_tmp(self, final_path: Path, tmp_path: Path) -> None:
        if not tmp_path.exists():
            return
        merge_path = final_path.with_name(
            f".{final_path.name}.{os.getpid()}.merge"
        )
        try:
            with merge_path.open("wb") as out:
                if final_path.exists():
                    with final_path.open("rb") as existing:
                        shutil.copyfileobj(existing, out)
                with tmp_path.open("rb") as pending:
                    shutil.copyfileobj(pending, out)
            os.replace(merge_path, final_path)
            tmp_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            with contextlib.suppress(FileNotFoundError):
                merge_path.unlink()
            logger.exception(
                "sink.publish_segment_failed",
                extra={"path": str(final_path), "tmp_path": str(tmp_path)},
            )
            raise

    def _path_for_receive_time(self, receive_time_ms: int) -> Path:
        if self._segment_duration_seconds > 0:
            return self._root / f"{_utc_segment_str(receive_time_ms, self._segment_duration_seconds)}.ndjson.gz"
        return self._root / f"{_utc_date_str(receive_time_ms)}.ndjson.gz"
