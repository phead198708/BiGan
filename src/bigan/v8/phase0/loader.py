"""Market data loading and normalization for v8 Phase 0."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq

from bigan.v8.phase0.contracts import MarketData


class MarketDataLoader:
    """Load source rows into deterministic ``MarketData`` records."""

    def load_rows(self, rows: Iterable[Mapping[str, Any]]) -> list[MarketData]:
        records = [MarketData(**self._normalize_row(row)) for row in rows]
        return self._dedupe_and_sort(records)

    def load_jsonl(self, path: Path | str) -> list[MarketData]:
        rows: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    rows.append(json.loads(text))
        return self.load_rows(rows)

    def load_csv(self, path: Path | str) -> list[MarketData]:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            return self.load_rows(csv.DictReader(handle))

    def load_parquet(self, path: Path | str) -> list[MarketData]:
        table = pq.read_table(path)
        return self.load_rows(table.to_pylist())

    def load_duckdb(
        self,
        database: duckdb.DuckDBPyConnection | Path | str,
        query: str,
    ) -> list[MarketData]:
        if isinstance(database, duckdb.DuckDBPyConnection):
            rows = database.execute(query).fetchall()
            columns = [column[0] for column in database.description or []]
            return self.load_rows(dict(zip(columns, row, strict=True)) for row in rows)

        conn = duckdb.connect(str(database), read_only=True)
        try:
            rows = conn.execute(query).fetchall()
            columns = [column[0] for column in conn.description or []]
            return self.load_rows(dict(zip(columns, row, strict=True)) for row in rows)
        finally:
            conn.close()

    def _normalize_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        normalized = dict(row)
        if "instrument_id" not in normalized:
            normalized["instrument_id"] = (
                normalized.get("source_symbol")
                or normalized.get("symbol")
                or normalized.get("canonical_symbol")
            )
        if "mid_price" not in normalized:
            normalized["mid_price"] = normalized.get("mid")
        if normalized.get("mid_price") is None and normalized.get("price") is not None:
            normalized["last_price"] = normalized.get("price")
        if "available_at_ts" not in normalized:
            normalized["available_at_ts"] = (
                normalized.get("available_ts")
                or normalized.get("feature_available_at_ts")
                or normalized.get("ts")
            )
        return {
            key: normalized.get(key)
            for key in (
                "ts",
                "available_at_ts",
                "source",
                "instrument_id",
                "bid_price",
                "ask_price",
                "mid_price",
                "last_price",
                "volume",
                "trade_count",
                "bid_size",
                "ask_size",
                "liquidity_depth",
                "timeframe_ms",
                "sequence",
            )
        }

    def _dedupe_and_sort(self, records: list[MarketData]) -> list[MarketData]:
        by_key: dict[tuple[str, str, int, int | None], MarketData] = {}
        for record in records:
            key = (record.source, record.instrument_id, record.ts, record.sequence)
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = record
                continue
            if existing.to_row() != record.to_row():
                raise ValueError(
                    "conflicting duplicate market rows for "
                    f"{record.source}/{record.instrument_id} at {record.ts}"
                )
        return sorted(
            by_key.values(),
            key=lambda row: (
                row.source,
                row.instrument_id,
                row.ts,
                row.available_at_ts or row.ts,
                row.sequence if row.sequence is not None else -1,
            ),
        )

