"""Symbol mapping helpers for canonical source identity (issue #22).

The raw tables keep both the source-native symbol identity and an optional
``canonical_symbol``. This module owns the small lookup layer that turns
``(source, source_symbol[, source_market], ts)`` into that canonical value.

Mapping rows are intentionally stored as normal dictionaries so they can be
written directly to the ``symbol_mapping`` Parquet table.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

DEFAULT_EFFECTIVE_FROM_TS = 0


@dataclass(frozen=True, slots=True)
class SymbolMappingEntry:
    """One source-symbol to canonical-symbol mapping.

    ``effective_to_ts`` is end-exclusive when present. A ``source_market`` of
    ``None`` is a generic mapping for the source symbol; market-specific rows
    win over generic rows during lookup.
    """

    source: str
    source_symbol: str
    canonical_symbol: str
    source_market: str | None = None
    effective_from_ts: int = DEFAULT_EFFECTIVE_FROM_TS
    effective_to_ts: int | None = None
    symbol_kind: str | None = None
    metadata_json: str | None = None
    ts: int = DEFAULT_EFFECTIVE_FROM_TS
    message_ts: int = DEFAULT_EFFECTIVE_FROM_TS
    ingest_ts: int = DEFAULT_EFFECTIVE_FROM_TS

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> SymbolMappingEntry:
        effective_from_ts = _optional_int(
            row.get("effective_from_ts", row.get("ts"))
        )
        if effective_from_ts is None:
            effective_from_ts = DEFAULT_EFFECTIVE_FROM_TS

        ts = _optional_int(row.get("ts"))
        message_ts = _optional_int(row.get("message_ts"))
        ingest_ts = _optional_int(row.get("ingest_ts"))

        return cls(
            source=_required_str(row.get("source"), "source"),
            source_symbol=_required_str(row.get("source_symbol"), "source_symbol"),
            source_market=_optional_str(row.get("source_market")),
            canonical_symbol=_required_str(
                row.get("canonical_symbol"), "canonical_symbol"
            ),
            effective_from_ts=effective_from_ts,
            effective_to_ts=_optional_int(row.get("effective_to_ts")),
            symbol_kind=_optional_str(row.get("symbol_kind")),
            metadata_json=_metadata_json(row),
            ts=effective_from_ts if ts is None else ts,
            message_ts=effective_from_ts if message_ts is None else message_ts,
            ingest_ts=effective_from_ts if ingest_ts is None else ingest_ts,
        )

    def is_active_at(self, ts: int) -> bool:
        if ts < self.effective_from_ts:
            return False
        return self.effective_to_ts is None or ts < self.effective_to_ts

    def to_row(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "message_ts": self.message_ts,
            "ingest_ts": self.ingest_ts,
            "source": self.source,
            "source_symbol": self.source_symbol,
            "source_market": self.source_market,
            "canonical_symbol": self.canonical_symbol,
            "effective_from_ts": self.effective_from_ts,
            "effective_to_ts": self.effective_to_ts,
            "symbol_kind": self.symbol_kind,
            "metadata_json": self.metadata_json,
        }


def symbol_mapping_row(
    *,
    source: str,
    source_symbol: str,
    canonical_symbol: str,
    source_market: str | None = None,
    effective_from_ts: int = DEFAULT_EFFECTIVE_FROM_TS,
    effective_to_ts: int | None = None,
    ingest_ts: int | None = None,
    message_ts: int | None = None,
    symbol_kind: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    metadata_json: str | None = None,
) -> dict[str, Any]:
    """Build a schema-compatible ``symbol_mapping`` row."""

    if metadata is not None and metadata_json is not None:
        raise ValueError("pass either metadata or metadata_json, not both")

    entry = SymbolMappingEntry(
        source=_required_str(source, "source"),
        source_symbol=_required_str(source_symbol, "source_symbol"),
        source_market=_optional_str(source_market),
        canonical_symbol=_required_str(canonical_symbol, "canonical_symbol"),
        effective_from_ts=int(effective_from_ts),
        effective_to_ts=(
            int(effective_to_ts) if effective_to_ts is not None else None
        ),
        symbol_kind=_optional_str(symbol_kind),
        metadata_json=(
            orjson.dumps(dict(metadata)).decode("utf-8")
            if metadata is not None
            else _optional_str(metadata_json)
        ),
        ts=int(effective_from_ts),
        message_ts=int(effective_from_ts)
        if message_ts is None
        else int(message_ts),
        ingest_ts=int(effective_from_ts) if ingest_ts is None else int(ingest_ts),
    )
    return entry.to_row()


class SymbolMapper:
    """Resolve canonical symbols from a set of mapping rows."""

    def __init__(
        self, rows: Iterable[Mapping[str, Any] | SymbolMappingEntry] = ()
    ) -> None:
        entries: list[SymbolMappingEntry] = []
        by_key: dict[tuple[str, str], list[SymbolMappingEntry]] = defaultdict(list)
        for row in rows:
            entry = (
                row
                if isinstance(row, SymbolMappingEntry)
                else SymbolMappingEntry.from_row(row)
            )
            entries.append(entry)
            by_key[(entry.source, entry.source_symbol)].append(entry)

        for bucket in by_key.values():
            bucket.sort(
                key=lambda e: (e.source_market is not None, e.effective_from_ts),
                reverse=True,
            )

        self._entries = tuple(entries)
        self._by_key = {key: tuple(bucket) for key, bucket in by_key.items()}

    @classmethod
    def from_path(cls, path: Path | str) -> SymbolMapper:
        return cls(load_symbol_mapping_rows(path))

    @property
    def entries(self) -> tuple[SymbolMappingEntry, ...]:
        return self._entries

    def to_rows(self) -> list[dict[str, Any]]:
        return [entry.to_row() for entry in self._entries]

    def resolve(
        self,
        *,
        source: str,
        source_symbol: str,
        ts: int,
        source_market: str | None = None,
    ) -> str | None:
        key = (_required_str(source, "source"), _required_str(source_symbol, "source_symbol"))
        market = _optional_str(source_market)
        for entry in self._by_key.get(key, ()):
            if entry.source_market is not None and entry.source_market != market:
                continue
            if entry.is_active_at(int(ts)):
                return entry.canonical_symbol
        return None

    def enrich_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Return ``row`` with ``canonical_symbol`` filled when resolvable."""

        out = dict(row)
        if _optional_str(out.get("canonical_symbol")):
            return out
        source = _optional_str(out.get("source"))
        source_symbol = _optional_str(out.get("source_symbol"))
        ts = _optional_int(out.get("ts"))
        if source is None or source_symbol is None or ts is None:
            return out
        canonical_symbol = self.resolve(
            source=source,
            source_symbol=source_symbol,
            source_market=_optional_str(out.get("source_market")),
            ts=ts,
        )
        if canonical_symbol is not None:
            out["canonical_symbol"] = canonical_symbol
        return out

    def enrich_tables(
        self, tables: Mapping[str, Iterable[Mapping[str, Any]]]
    ) -> dict[str, list[dict[str, Any]]]:
        return {
            table_name: [self.enrich_row(row) for row in rows]
            for table_name, rows in tables.items()
        }


def load_symbol_mapping_rows(path: Path | str) -> list[dict[str, Any]]:
    """Load mapping rows from CSV, JSON, JSONL, or a directory of those files."""

    path = Path(path)
    if path.is_dir():
        rows: list[dict[str, Any]] = []
        for child in sorted(path.iterdir()):
            if child.suffix.lower() in {".csv", ".json", ".jsonl"}:
                rows.extend(load_symbol_mapping_rows(child))
        return rows

    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fp:
            return [symbol_mapping_row(**_drop_empty_values(row)) for row in csv.DictReader(fp)]

    if suffix == ".jsonl":
        rows = []
        with path.open("rb") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    rows.append(symbol_mapping_row(**orjson.loads(line)))
        return rows

    if suffix == ".json":
        data = orjson.loads(path.read_bytes())
        if isinstance(data, dict):
            data = data.get("mappings")
        if not isinstance(data, list):
            raise ValueError("JSON symbol mapping file must contain a list or {'mappings': [...]}")
        return [symbol_mapping_row(**row) for row in data]

    raise ValueError(f"unsupported symbol mapping file type: {path}")


def _drop_empty_values(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if value not in ("", None)}


def _required_str(value: Any, field_name: str) -> str:
    text = _optional_str(value)
    if text is None:
        raise ValueError(f"{field_name} is required")
    return text


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _metadata_json(row: Mapping[str, Any]) -> str | None:
    metadata_json = _optional_str(row.get("metadata_json"))
    if metadata_json is not None:
        return metadata_json
    metadata = row.get("metadata")
    if metadata is None:
        return None
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    return orjson.dumps(dict(metadata)).decode("utf-8")
