"""Batch ETL: read NDJSON.gz raw archive -> canonical Parquet warehouse.

Scans the raw NDJSON tree (both the active partition root and the rollup
``_done`` directory), decodes events, projects them through ``transform``
into canonical row dicts, buffers via :class:`WarehouseWriter`, and finally
runs 1-minute candle aggregation.

Idempotency: this runner is **forward-only**; running it twice on the same
input produces duplicate rows. Issue #3 acceptance only requires the tables
to be append-only and queryable, not deduplicated. A future processed-file
sentinel can be added in a follow-up if re-runs become a routine workflow.

Safety: a file is considered "active" (in-flight) and is skipped if its
mtime is within ``lag_seconds`` of now. This mirrors the rollup worker's
behaviour and avoids reading partially-written gzip frames.
"""

from __future__ import annotations

import gzip
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

from .candles import CandleAggregator
from .schemas import TABLE_NAMES
from .transform import transform_event
from .validation import RowValidator
from .writer import WarehouseWriter

logger = logging.getLogger(__name__)

#: Tables the validator inspects. ``raw_candles_1m`` is derived from already-
#: validated rows so it does not go through per-row validation again.
_VALIDATED_TABLES: frozenset[str] = frozenset(
    {"raw_top_of_book", "raw_orderbook_snapshot", "raw_trades"}
)


@dataclass(slots=True)
class EtlReport:
    """Summary of one ETL run; useful for tests and CLI output."""

    files_processed: int = 0
    records_read: int = 0
    rows_per_table: dict[str, int] = None  # type: ignore[assignment]
    quarantined_by_rule: dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.rows_per_table is None:
            self.rows_per_table = dict.fromkeys(TABLE_NAMES, 0)
        if self.quarantined_by_rule is None:
            self.quarantined_by_rule = {}

    @property
    def quarantined_total(self) -> int:
        return sum(self.quarantined_by_rule.values())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_etl_batch(
    *,
    raw_dir: Path | str,
    warehouse_dir: Path | str,
    lag_seconds: float = 60.0,
    max_rows_per_partition: int = 50_000,
) -> EtlReport:
    """Process every eligible NDJSON.gz under ``raw_dir`` into the warehouse.

    Returns an :class:`EtlReport` with per-table row counts.
    """
    raw_dir = Path(raw_dir)
    warehouse_dir = Path(warehouse_dir)
    report = EtlReport()
    aggregator = CandleAggregator()
    validator = RowValidator()
    ingest_ts_now_ms = int(time.time() * 1000)

    files = _eligible_files(raw_dir, lag_seconds)
    logger.info("etl.start", extra={"files": len(files), "raw_dir": str(raw_dir)})

    with WarehouseWriter(
        warehouse_dir, max_rows_per_partition=max_rows_per_partition
    ) as writer:
        for src in files:
            try:
                _process_file(
                    src,
                    writer=writer,
                    aggregator=aggregator,
                    validator=validator,
                    report=report,
                )
            except Exception:  # noqa: BLE001
                logger.exception("etl.file_failed", extra={"src": str(src)})
                raise
            report.files_processed += 1

        # Candle aggregation runs after all raw rows are in flight to ensure
        # OHLC reflects the full set of events seen during this batch.
        candle_rows = aggregator.emit(ingest_ts=ingest_ts_now_ms)
        if candle_rows:
            writer.append_rows("raw_candles_1m", candle_rows)

    # Pull final per-table totals from the writer's stats.
    for table in TABLE_NAMES:
        report.rows_per_table[table] = writer.stats.rows_written.get(table, 0)
    report.quarantined_by_rule = dict(validator.stats.rows_quarantined_by_rule)

    logger.info(
        "etl.done",
        extra={
            "files": report.files_processed,
            "records": report.records_read,
            "rows": report.rows_per_table,
            "quarantined_by_rule": report.quarantined_by_rule,
            "quarantined_total": report.quarantined_total,
        },
    )
    return report


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _eligible_files(raw_dir: Path, lag_seconds: float) -> list[Path]:
    """All NDJSON.gz under ``raw_dir`` whose mtime is older than ``lag_seconds``."""
    if not raw_dir.exists():
        return []
    cutoff = time.time() - lag_seconds
    out: list[Path] = []
    for p in sorted(raw_dir.rglob("*.ndjson.gz")):
        try:
            if p.stat().st_mtime <= cutoff:
                out.append(p)
        except FileNotFoundError:
            continue
    return out


def _process_file(
    src: Path,
    *,
    writer: WarehouseWriter,
    aggregator: CandleAggregator,
    validator: RowValidator,
    report: EtlReport,
) -> None:
    logger.info("etl.file_start", extra={"src": str(src)})
    for record in _iter_ndjson_gz(src):
        report.records_read += 1
        tables = transform_event(record)
        for table_name, rows in tables.items():
            if not rows:
                continue
            _route_rows(
                table_name,
                rows,
                writer=writer,
                aggregator=aggregator,
                validator=validator,
            )
    logger.info("etl.file_done", extra={"src": str(src)})


def _route_rows(
    table_name: str,
    rows: list[dict[str, Any]],
    *,
    writer: WarehouseWriter,
    aggregator: CandleAggregator,
    validator: RowValidator,
) -> None:
    """Split ``rows`` into clean and quarantined groups and emit them."""
    if table_name not in _VALIDATED_TABLES:
        writer.append_rows(table_name, rows)
        return

    clean: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    for row in rows:
        errors = validator.validate(table_name, row)
        if errors:
            quarantine.extend(validator.to_quarantine_rows(table_name, row, errors))
        else:
            clean.append(row)

    if clean:
        writer.append_rows(table_name, clean)
        if table_name == "raw_top_of_book":
            for row in clean:
                aggregator.add_top_of_book(row)
        elif table_name == "raw_trades":
            for row in clean:
                aggregator.add_trade(row)

    if quarantine:
        writer.append_rows("quarantine", quarantine)


def _iter_ndjson_gz(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, mode="rb") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                yield orjson.loads(line)
            except orjson.JSONDecodeError:
                logger.warning("etl.bad_line", extra={"path": str(path)})
