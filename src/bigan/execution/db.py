"""Lightweight DuckDB helpers for execution tables."""

from __future__ import annotations

import os
import time
from pathlib import Path

import duckdb

DEFAULT_MLOPS_DB_PATH = Path("data/mlops/champion_catalog.duckdb")
DEFAULT_EXECUTION_DB_CONNECT_RETRIES = 30
DEFAULT_EXECUTION_DB_CONNECT_RETRY_DELAY_SECONDS = 0.25


def connect_mlops_db(
    path: Path | str = DEFAULT_MLOPS_DB_PATH,
    *,
    retry_attempts: int | None = None,
    retry_delay_seconds: float | None = None,
    read_only: bool = False,
) -> duckdb.DuckDBPyConnection:
    """Open the execution/MLOps DuckDB catalog with transient lock retry."""

    attempts = _connect_retry_attempts(retry_attempts)
    delay = _connect_retry_delay(retry_delay_seconds)
    last_exc: duckdb.Error | None = None
    for attempt in range(attempts):
        try:
            return _connect_duckdb(path, read_only=read_only)
        except duckdb.Error as exc:
            if not _is_duckdb_lock_error(exc):
                raise
            last_exc = exc
            if attempt == attempts - 1:
                break
            if delay > 0:
                time.sleep(delay)
    if last_exc is not None:
        raise last_exc
    return _connect_duckdb(path, read_only=read_only)


def _connect_duckdb(path: Path | str, *, read_only: bool) -> duckdb.DuckDBPyConnection:
    if read_only:
        return duckdb.connect(str(path), read_only=True)
    return duckdb.connect(str(path))


def _connect_retry_attempts(value: int | None) -> int:
    if value is None:
        value = _int_env(
            "BIGAN_EXECUTION_DB_CONNECT_RETRIES",
            _int_env("BIGAN_MLOPS_CONNECT_RETRIES", DEFAULT_EXECUTION_DB_CONNECT_RETRIES),
        )
    return max(1, int(value))


def _connect_retry_delay(value: float | None) -> float:
    if value is None:
        value = _float_env(
            "BIGAN_EXECUTION_DB_CONNECT_RETRY_DELAY_SECONDS",
            _float_env(
                "BIGAN_MLOPS_CONNECT_RETRY_DELAY_SECONDS",
                DEFAULT_EXECUTION_DB_CONNECT_RETRY_DELAY_SECONDS,
            ),
        )
    return max(0.0, float(value))


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _is_duckdb_lock_error(exc: duckdb.Error) -> bool:
    text = str(exc)
    return "Could not set lock" in text or "Conflicting lock" in text
