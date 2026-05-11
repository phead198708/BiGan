"""DuckDB query helpers for the canonical warehouse.

The warehouse is just Hive-partitioned Parquet on disk. DuckDB consumes
this natively via ``read_parquet(..., hive_partitioning=true)`` so we
don't actually need an authoritative catalog — but for ergonomic SQL we
register one VIEW per canonical table.

Usage::

    from bigan.canonical.query import open_warehouse

    with open_warehouse("data/warehouse") as conn:
        df = conn.execute(
            "SELECT count(*) FROM raw_top_of_book WHERE source = 'polymarket'"
        ).fetchdf()
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb

from .schemas import TABLE_NAMES


@contextmanager
def open_warehouse(root: Path | str, *, read_only: bool = True) -> Iterator[duckdb.DuckDBPyConnection]:
    """Context manager: open an in-memory DuckDB conn with all canonical
    tables registered as VIEWs over the on-disk Parquet partitions.

    Tables that don't yet have any partitions are silently skipped — the
    SELECT will fail with a clear DuckDB error if the user queries one.
    """
    root = Path(root)
    conn = duckdb.connect(":memory:", read_only=False)
    # ``read_only`` currently has no effect on in-memory dbs but kept in
    # the signature to allow swapping to a persistent .duckdb file later.
    try:
        for table in TABLE_NAMES:
            base = root / table
            if not base.exists():
                continue
            glob = str(base / "**/*.parquet")
            conn.execute(
                f"CREATE OR REPLACE VIEW {table} AS "
                f"SELECT * FROM read_parquet(?, hive_partitioning=true)",
                [glob],
            )
        yield conn
    finally:
        conn.close()


def warehouse_summary(root: Path | str) -> dict[str, int]:
    """Return ``{table_name: row_count}`` across all partitions."""
    summary: dict[str, int] = {}
    with open_warehouse(root) as conn:
        for table in TABLE_NAMES:
            try:
                row_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except duckdb.CatalogException:
                # Table has no partitions yet; treat as zero rows.
                row_count = 0
            except duckdb.IOException:
                # Empty parquet glob.
                row_count = 0
            summary[table] = int(row_count)
    return summary
