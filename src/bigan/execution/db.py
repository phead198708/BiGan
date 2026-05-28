"""Lightweight DuckDB helpers for execution tables."""

from __future__ import annotations

from pathlib import Path

import duckdb

DEFAULT_MLOPS_DB_PATH = Path("data/mlops/champion_catalog.duckdb")


def connect_mlops_db(path: Path | str = DEFAULT_MLOPS_DB_PATH) -> duckdb.DuckDBPyConnection:
    """Open the execution/MLOps DuckDB catalog."""

    return duckdb.connect(str(path))
