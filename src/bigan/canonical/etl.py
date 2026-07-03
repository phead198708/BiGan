"""Batch ETL: read NDJSON.gz raw archive -> canonical Parquet warehouse.

Scans the raw NDJSON tree (both the active partition root and the rollup
``_done`` directory), decodes events, projects them through ``transform``
into canonical row dicts, buffers via :class:`WarehouseWriter`, and finally
runs 1-minute candle aggregation.

Idempotency: this runner is **append-only**. Re-running quote/orderbook inputs
will append fresh rows, but ``raw_trades`` is guarded by a partition-local
``trade_id`` read-check so replayed backfills do not double-count volume.

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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson
import pyarrow.parquet as pq

from .candles import CandleAggregator
from .schemas import TABLE_NAMES
from .symbols import SymbolMapper
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
    cross_batch_duplicates_skipped: int = 0
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
    max_files_per_batch: int | None = None,
    symbol_mapper: SymbolMapper | None = None,
    symbol_mapping_path: Path | str | None = None,
    processed_manifest_path: Path | str | None = None,
    timestamp_future_grace_seconds: float = 5.0,
    timestamp_stale_threshold_seconds: float = 600.0,
) -> EtlReport:
    """Process every eligible NDJSON.gz under ``raw_dir`` into the warehouse.

    Returns an :class:`EtlReport` with per-table row counts.
    """
    if symbol_mapper is not None and symbol_mapping_path is not None:
        raise ValueError("pass either symbol_mapper or symbol_mapping_path, not both")

    raw_dir = Path(raw_dir)
    warehouse_dir = Path(warehouse_dir)
    if symbol_mapping_path is not None:
        symbol_mapper = SymbolMapper.from_path(symbol_mapping_path)

    report = EtlReport()
    aggregator = CandleAggregator()
    validator = RowValidator(
        future_grace_ms=int(timestamp_future_grace_seconds * 1000),
        stale_threshold_ms=int(timestamp_stale_threshold_seconds * 1000),
    )
    trade_deduper = CrossBatchTradeDeduper(warehouse_dir)
    ingest_ts_now_ms = int(time.time() * 1000)

    processed_manifest = None if processed_manifest_path is None else Path(processed_manifest_path)
    processed_keys = _load_processed_manifest(processed_manifest)
    files = [
        path
        for path in _eligible_files(raw_dir, lag_seconds)
        if _processed_manifest_key(path) not in processed_keys
    ]
    if max_files_per_batch is not None:
        if max_files_per_batch <= 0:
            raise ValueError("max_files_per_batch must be positive")
        files = files[:max_files_per_batch]
    processed_this_run: list[Path] = []
    logger.info(
        "etl.start",
        extra={
            "files": len(files),
            "raw_dir": str(raw_dir),
            "max_files_per_batch": max_files_per_batch,
        },
    )

    with WarehouseWriter(
        warehouse_dir, max_rows_per_partition=max_rows_per_partition
    ) as writer:
        if symbol_mapper is not None:
            mapping_rows = symbol_mapper.to_rows()
            if mapping_rows:
                writer.append_rows("symbol_mapping", mapping_rows)

        for src in files:
            try:
                _process_file(
                    src,
                    writer=writer,
                    aggregator=aggregator,
                    validator=validator,
                    trade_deduper=trade_deduper,
                    symbol_mapper=symbol_mapper,
                    report=report,
                )
            except Exception:  # noqa: BLE001
                logger.exception("etl.file_failed", extra={"src": str(src)})
                raise
            report.files_processed += 1
            processed_this_run.append(src)

        # Candle aggregation runs after all raw rows are in flight to ensure
        # OHLC reflects the full set of events seen during this batch.
        candle_rows = aggregator.emit(ingest_ts=ingest_ts_now_ms)
        if candle_rows:
            writer.append_rows("raw_candles_1m", candle_rows)

    # Pull final per-table totals from the writer's stats.
    for table in TABLE_NAMES:
        report.rows_per_table[table] = writer.stats.rows_written.get(table, 0)
    report.quarantined_by_rule = dict(validator.stats.rows_quarantined_by_rule)
    _append_processed_manifest(processed_manifest, processed_this_run)

    logger.info(
        "etl.done",
        extra={
            "files": report.files_processed,
            "records": report.records_read,
            "rows": report.rows_per_table,
            "quarantined_by_rule": report.quarantined_by_rule,
            "quarantined_total": report.quarantined_total,
            "cross_batch_duplicates_skipped": report.cross_batch_duplicates_skipped,
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


def _processed_manifest_key(path: Path) -> str:
    return path.name


def _load_processed_manifest(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    return {
        Path(line.strip()).name
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _append_processed_manifest(path: Path | None, processed: list[Path]) -> None:
    if path is None or not processed:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        for src in processed:
            fp.write(_processed_manifest_key(src) + "\n")


def _process_file(
    src: Path,
    *,
    writer: WarehouseWriter,
    aggregator: CandleAggregator,
    validator: RowValidator,
    trade_deduper: CrossBatchTradeDeduper,
    symbol_mapper: SymbolMapper | None,
    report: EtlReport,
) -> None:
    logger.info("etl.file_start", extra={"src": str(src)})
    for record in _iter_ndjson_gz(src):
        report.records_read += 1
        tables = transform_event(record)
        if symbol_mapper is not None:
            tables = symbol_mapper.enrich_tables(tables)
        for table_name, rows in tables.items():
            if not rows:
                continue
            _route_rows(
                table_name,
                rows,
                writer=writer,
                aggregator=aggregator,
                validator=validator,
                trade_deduper=trade_deduper,
                report=report,
            )
    logger.info("etl.file_done", extra={"src": str(src)})


def _route_rows(
    table_name: str,
    rows: list[dict[str, Any]],
    *,
    writer: WarehouseWriter,
    aggregator: CandleAggregator,
    validator: RowValidator,
    trade_deduper: CrossBatchTradeDeduper,
    report: EtlReport,
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
        if table_name == "raw_trades":
            filtered: list[dict[str, Any]] = []
            for row in clean:
                if trade_deduper.is_duplicate(row):
                    report.cross_batch_duplicates_skipped += 1
                    continue
                filtered.append(row)
            clean = filtered
            if not clean and not quarantine:
                return

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


class CrossBatchTradeDeduper:
    """Partition-aware read-before-write de-dup for ``raw_trades`` (#27)."""

    def __init__(self, warehouse_dir: Path | str) -> None:
        self._warehouse_dir = Path(warehouse_dir)
        self._seen_by_partition: dict[tuple[str, str], set[str]] = {}

    def is_duplicate(self, row: dict[str, Any]) -> bool:
        trade_id = row.get("trade_id")
        source = row.get("source")
        ts = row.get("ts")
        if not trade_id or not source or ts is None:
            return False
        key = (str(source), _utc_date_str(int(ts)))
        seen = self._seen_by_partition.get(key)
        if seen is None:
            seen = self._load_existing_trade_ids(*key)
            self._seen_by_partition[key] = seen

        tid = str(trade_id)
        if tid in seen:
            return True
        seen.add(tid)
        return False

    def _load_existing_trade_ids(self, source: str, dt: str) -> set[str]:
        partition = self._warehouse_dir / "raw_trades" / f"source={source}" / f"dt={dt}"
        if not partition.exists():
            return set()
        out: set[str] = set()
        for path in sorted(partition.glob("part-*.parquet")):
            try:
                table = pq.ParquetFile(path).read(columns=["trade_id"])
            except (FileNotFoundError, OSError, KeyError, ValueError):
                continue
            for trade_id in table.column("trade_id").to_pylist():
                if trade_id:
                    out.add(str(trade_id))
        return out


def _utc_date_str(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).strftime("%Y-%m-%d")
