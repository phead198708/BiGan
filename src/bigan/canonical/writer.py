"""Buffered Parquet writer for the canonical warehouse.

Layout (Hive-partitioned, append-only)::

    <root>/<table>/source=<source>/dt=YYYY-MM-DD/part-<ts_ns>-<rand>.parquet

Behaviour:

- Rows are buffered in memory keyed by ``(table, source, dt)`` and flushed
  either explicitly or when the buffer for any partition exceeds
  ``max_rows_per_partition``.
- Schema is enforced at write time: rows missing a column are filled with
  the column's default (``None`` for nullable fields). Extra keys are
  silently dropped — keeping the writer permissive across minor producer
  versions while never letting unknown columns leak into the parquet.
- Partitioning by ``dt`` uses the row's ``ts`` (canonical event time, UTC).
- Filenames are ``part-{ts_ns}-{rand_hex}.parquet`` so each writer call is
  collision-free across processes without a coordination service.

The writer is **not** thread-safe by itself; the ETL runner is single-
threaded and asynchronous in the sense that it ``asyncio.to_thread``s the
Parquet write step.
"""

from __future__ import annotations

import logging
import secrets
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .schemas import SCHEMAS, TABLE_NAMES

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WriterStats:
    """Accumulated statistics for a writer's lifetime."""

    rows_written: dict[str, int] = field(default_factory=dict)
    files_written: dict[str, int] = field(default_factory=dict)

    def record(self, table: str, rows: int) -> None:
        self.rows_written[table] = self.rows_written.get(table, 0) + rows
        self.files_written[table] = self.files_written.get(table, 0) + 1


class WarehouseWriter:
    """Buffered Parquet appender with Hive partitioning.

    Use as a context manager (``with WarehouseWriter(root) as w: ...``) so
    pending buffers are flushed even on exceptions.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        max_rows_per_partition: int = 50_000,
        compression: str = "snappy",
    ) -> None:
        self._root = Path(root)
        self._max_rows = max_rows_per_partition
        self._compression = compression
        # Buffer keyed by (table, source, dt) -> list[row_dict].
        self._buffers: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        self.stats = WriterStats()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> WarehouseWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.flush()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append_rows(self, table: str, rows: Iterable[Mapping[str, Any]]) -> None:
        """Buffer ``rows`` for ``table``, auto-flushing oversized partitions."""
        if table not in SCHEMAS:
            raise ValueError(f"unknown canonical table: {table!r}")
        schema = SCHEMAS[table]

        for row in rows:
            ts_field = "bucket_ts" if "bucket_ts" in {f.name for f in schema} else "ts"
            ts = row.get(ts_field) or row.get("ts")
            if ts is None:
                logger.warning("warehouse.row_missing_ts", extra={"table": table})
                continue
            source = row.get("source")
            if not source:
                logger.warning("warehouse.row_missing_source", extra={"table": table})
                continue
            dt = _utc_date_str(int(ts))
            key = (table, str(source), dt)
            self._buffers[key].append(dict(row))
            if len(self._buffers[key]) >= self._max_rows:
                self._flush_partition(*key)

    def flush(self, table: str | None = None) -> None:
        """Flush buffered rows. Pass ``table=None`` to flush everything."""
        keys = [k for k in self._buffers if table is None or k[0] == table]
        for key in keys:
            self._flush_partition(*key)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _flush_partition(self, table: str, source: str, dt: str) -> None:
        rows = self._buffers.pop((table, source, dt), None)
        if not rows:
            return
        schema = SCHEMAS[table]
        target_dir = self._root / table / f"source={source}" / f"dt={dt}"
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / _make_part_filename()

        # Project rows onto the schema to get deterministic column order
        # and drop any extras silently.
        projected = [_project_row(row, schema) for row in rows]
        try:
            table_obj = pa.Table.from_pylist(projected, schema=schema)
        except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
            logger.exception(
                "warehouse.schema_violation",
                extra={"table": table, "rows": len(rows), "err": str(exc)},
            )
            raise

        pq.write_table(table_obj, path, compression=self._compression)
        self.stats.record(table, len(rows))
        logger.info(
            "warehouse.flushed",
            extra={
                "table": table,
                "source": source,
                "dt": dt,
                "rows": len(rows),
                "path": str(path),
            },
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _utc_date_str(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).strftime("%Y-%m-%d")


def _make_part_filename() -> str:
    """``part-{nanos}-{rand}.parquet`` — sortable + collision-free."""
    return f"part-{time.time_ns()}-{secrets.token_hex(4)}.parquet"


def _project_row(row: Mapping[str, Any], schema: pa.Schema) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field_ in schema:
        out[field_.name] = row.get(field_.name)
    return out


# ---------------------------------------------------------------------------
# Convenience: enumerate existing partitions for queries / sanity checks.
# ---------------------------------------------------------------------------


def warehouse_files(root: Path | str, table: str) -> list[Path]:
    """Return all parquet files under ``<root>/<table>/...`` (sorted)."""
    if table not in TABLE_NAMES:
        raise ValueError(f"unknown canonical table: {table!r}")
    root = Path(root)
    base = root / table
    if not base.exists():
        return []
    return sorted(base.rglob("part-*.parquet"))
