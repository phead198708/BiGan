"""NDJSON-to-Parquet rollup worker.

Once per ``rollup_interval_seconds``, scan ``<data_dir>/<raw_subdir>`` for
NDJSON.gz files older than ``rollup_lag_seconds`` (i.e. files the sink is no
longer actively writing). For each such file, decode every record and write a
Parquet file under ``<data_dir>/<rollup_subdir>/`` partitioned by UTC date and
event_type, e.g.::

    rollup/ws_market/date=2026-05-10/event_type=book.parquet

The rollup is **idempotent**: if the target Parquet already exists, the source
NDJSON.gz is moved into ``rollup/ws_market/_done/`` rather than re-processed.
Failed rollups leave NDJSON in place; the next cycle will retry.

Parquet schema is intentionally permissive (one column per top-level JSON
field, plus the verbatim payload as a JSON string in ``raw_payload``). Issue
#3 / #6 will produce a stricter, HF-aligned schema later — this file is just
the v0 archival pipeline.
"""

from __future__ import annotations

import asyncio
import gzip
import logging
import shutil
import time
from collections import defaultdict
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

from .metrics import ROLLUP_FILES_TOTAL

logger = logging.getLogger(__name__)


def _iter_ndjson_gz(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, mode="rb") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                yield orjson.loads(line)
            except orjson.JSONDecodeError:
                logger.warning("rollup.bad_line", extra={"path": str(path)})


def _flatten(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Project the verbatim payload to ``(event_type, row)``.

    ``event_type`` is returned separately because it becomes the Hive partition
    key and must NOT appear as a column in the row (otherwise pyarrow's
    Dataset reader sees it twice with conflicting dtypes).

    Keeps the original payload as ``raw_payload`` (JSON string) so the rollup
    is lossless even though we surface only a few top-level columns.
    """
    raw = record.get("raw") if "raw" in record else record
    event_type = (raw or {}).get("event_type") or "unknown"
    timestamp = (raw or {}).get("timestamp")
    asset_id = (raw or {}).get("asset_id")
    market = (raw or {}).get("market")
    receive_time = record.get("receive_time")
    row = {
        "receive_time": int(receive_time) if receive_time is not None else None,
        "exchange_time": int(timestamp) if timestamp is not None else None,
        "asset_id": str(asset_id) if asset_id is not None else None,
        "market": str(market) if market is not None else None,
        "raw_payload": orjson.dumps(raw).decode("utf-8") if raw is not None else None,
    }
    return event_type, row


def _date_partition(ms_epoch: int | None) -> str:
    if ms_epoch is None:
        ms_epoch = int(time.time() * 1000)
    return datetime.fromtimestamp(ms_epoch / 1000, tz=UTC).strftime("%Y-%m-%d")


def rollup_file(src: Path, out_root: Path, *, done_dir: Path) -> int:
    """Convert one NDJSON.gz into per-(date, event_type) Parquet shards.

    Returns the number of records processed.
    """
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for raw in _iter_ndjson_gz(src):
        event_type, row = _flatten(raw)
        key = (_date_partition(row["receive_time"]), event_type)
        grouped[key].append(row)

    if not grouped:
        # Empty file: still archive to avoid reprocessing.
        done_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(done_dir / src.name))
        return 0

    total = 0
    for (date, etype), rows in grouped.items():
        partition_dir = out_root / f"date={date}" / f"event_type={etype}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        # Suffix with source-file stem to make outputs unique across rollup runs.
        out_path = partition_dir / f"{src.stem.removesuffix('.ndjson')}.parquet"
        if out_path.exists():
            # Already rolled up; pretend we did the work and archive the source.
            logger.info("rollup.already_done", extra={"out": str(out_path)})
            continue

        table = pa.Table.from_pylist(rows)
        pq.write_table(table, out_path, compression="snappy")
        total += len(rows)

    done_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(done_dir / src.name))
    return total


def _eligible_files(raw_dir: Path, lag_seconds: float) -> list[Path]:
    """Return NDJSON.gz files older than ``lag_seconds`` (mtime-based)."""
    if not raw_dir.exists():
        return []
    cutoff = time.time() - lag_seconds
    out = []
    for p in sorted(raw_dir.glob("*.ndjson.gz")):
        try:
            if p.stat().st_mtime <= cutoff:
                out.append(p)
        except FileNotFoundError:
            continue
    return out


async def run_rollup_worker(
    raw_dir: Path,
    rollup_dir: Path,
    *,
    interval_seconds: float,
    lag_seconds: float,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Long-running async worker that rolls up NDJSON.gz files periodically.

    The actual work (gzip read + Parquet write) happens in a thread to keep
    the event loop responsive.
    """
    stop = stop_event or asyncio.Event()
    raw_dir = Path(raw_dir)
    rollup_dir = Path(rollup_dir)
    done_dir = raw_dir / "_done"

    while not stop.is_set():
        try:
            files = _eligible_files(raw_dir, lag_seconds)
            for src in files:
                try:
                    n = await asyncio.to_thread(
                        rollup_file, src, rollup_dir, done_dir=done_dir
                    )
                    ROLLUP_FILES_TOTAL.labels(outcome="ok").inc()
                    logger.info("rollup.ok", extra={"src": src.name, "records": n})
                except Exception:  # noqa: BLE001
                    ROLLUP_FILES_TOTAL.labels(outcome="error").inc()
                    logger.exception("rollup.failed", extra={"src": str(src)})
        except Exception:  # noqa: BLE001
            logger.exception("rollup.cycle_failed")

        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
        except TimeoutError:
            continue
