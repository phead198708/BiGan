"""Raw-message persistence sinks.

The default :class:`NdjsonGzipSink` writes one gzip-compressed NDJSON file per
UTC date under ``<data_dir>/<raw_subdir>/YYYY-MM-DD.ndjson.gz``. Records are
buffered in memory and flushed on a timer (or when the buffer hits
``max_buffer_records``) to amortise gzip framing overhead.

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
import time
from collections.abc import Mapping
from dataclasses import dataclass
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


@dataclass
class _OpenFile:
    """Lazy handle wrapping a gzip writer for one UTC date partition."""

    path: Path
    fp: gzip.GzipFile


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
    ) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._flush_interval = flush_interval_seconds
        self._max_buffer = max_buffer_records
        self._buffer: list[bytes] = []
        self._buffer_bytes = 0
        self._lock = asyncio.Lock()
        self._open_files: dict[str, _OpenFile] = {}
        self._flusher_task: asyncio.Task[None] | None = None
        self._closed = False
        self._last_flush = time.monotonic()

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
        async with self._lock:
            if not self._buffer:
                return
            buffer, self._buffer = self._buffer, []
            self._buffer_bytes = 0

        start = time.monotonic()
        await asyncio.to_thread(self._flush_blocking, buffer)
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
        await self.flush()
        await asyncio.to_thread(self._close_files)

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

    def _flush_blocking(self, buffer: list[bytes]) -> None:
        # Group records by UTC date partition via the first 4 bytes pattern.
        # Cheaper: just look for ``receive_time`` field in JSON; but simplest
        # is to re-parse the small subset of bytes needed.
        groups: dict[str, list[bytes]] = {}
        for line in buffer:
            try:
                rec = orjson.loads(line)
                rt = int(rec.get("receive_time") or rec.get("ts") or 0)
            except (orjson.JSONDecodeError, TypeError, ValueError):
                rt = int(time.time() * 1000)
            partition = _utc_date_str(rt)
            groups.setdefault(partition, []).append(line)

        for partition, lines in groups.items():
            handle = self._open_files.get(partition)
            if handle is None:
                path = self._root / f"{partition}.ndjson.gz"
                # ``ab`` ensures we append to existing partition file across restarts.
                # Long-lived: closed in ``_close_files`` to amortise gzip framing overhead.
                fp = gzip.open(path, mode="ab")  # type: ignore[assignment]  # noqa: SIM115
                handle = _OpenFile(path=path, fp=fp)
                self._open_files[partition] = handle
            for line in lines:
                handle.fp.write(line)
            handle.fp.flush()

    def _close_files(self) -> None:
        for handle in self._open_files.values():
            try:
                handle.fp.close()
            except Exception:  # noqa: BLE001
                logger.exception("sink.close_file_failed", extra={"path": str(handle.path)})
        self._open_files.clear()
