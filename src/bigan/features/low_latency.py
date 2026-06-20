"""Low-latency BTC-15M feature path primitives for Phase 4.

The classes here are intentionally small: they provide a direct raw-row queue
and an incremental feature engine that shares the locked ``features_15m_v1``
formula with batch recompute while avoiding the raw gzip segment/warehouse hop.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.canonical.writer import WarehouseWriter
from bigan.monitoring.market_quality import (
    is_degenerate_quote,
    round_end_ts_from_canonical_symbol,
    round_start_ts_from_canonical_symbol,
)

from .aggregation import aggregate_features_15m_v1
from .quality import DEFAULT_QUALITY_CONFIG, FeatureQualityConfig

RAW_QUEUE_TABLES = frozenset(
    {
        "raw_top_of_book",
        "raw_orderbook_snapshot",
        "raw_trades",
    }
)


@dataclass(frozen=True, slots=True)
class RawQueueItem:
    """One canonical raw row published to the low-latency queue."""

    table: str
    row: dict[str, Any]
    published_at_ms: int


@dataclass(frozen=True, slots=True)
class LowLatencyFeatureQueueReport:
    """Summary for one low-latency raw-queue feature batch."""

    rows_read: int = 0
    rows_generated: int = 0
    rows_written: int = 0
    start_cursor: int = 0
    next_cursor: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_read": self.rows_read,
            "rows_generated": self.rows_generated,
            "rows_written": self.rows_written,
            "start_cursor": self.start_cursor,
            "next_cursor": self.next_cursor,
        }


class JsonlRawQueue:
    """Append-only JSONL queue for canonical raw rows.

    Cursors are zero-based line numbers. Batch consumers also persist the
    matching file offset in their state so repeated live scans can resume with
    ``seek`` instead of walking large consumed prefixes.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        table: str,
        row: Mapping[str, Any],
        *,
        published_at_ms: int | None = None,
    ) -> RawQueueItem:
        """Append one canonical raw row and return the queued item."""

        table = _validate_table(table)
        item = RawQueueItem(
            table=table,
            row=dict(row),
            published_at_ms=(
                int(time.time() * 1000)
                if published_at_ms is None
                else int(published_at_ms)
            ),
        )
        payload = {
            "table": item.table,
            "row": item.row,
            "published_at_ms": item.published_at_ms,
        }
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            fp.write("\n")
            fp.flush()
        return item

    def read_from(
        self,
        cursor: int = 0,
        *,
        max_records: int | None = None,
    ) -> tuple[list[RawQueueItem], int]:
        """Read queued rows starting at ``cursor`` and return ``(items, next)``."""

        items, next_cursor, _ = self.read_from_line_cursor(
            cursor,
            max_records=max_records,
        )
        return items, next_cursor

    def read_from_line_cursor(
        self,
        cursor: int = 0,
        *,
        max_records: int | None = None,
    ) -> tuple[list[RawQueueItem], int, int]:
        """Read from a line cursor and return ``(items, next_line, next_offset)``."""

        if cursor < 0:
            raise ValueError("cursor must be non-negative")
        if max_records is not None and max_records < 0:
            raise ValueError("max_records must be non-negative")
        if not self.path.exists():
            return [], cursor, 0

        items: list[RawQueueItem] = []
        next_cursor = cursor
        next_offset = 0
        line_number = 0
        with self.path.open("r", encoding="utf-8") as fp:
            while True:
                if max_records is not None and len(items) >= max_records:
                    break
                line = fp.readline()
                if not line:
                    break
                line_offset = fp.tell()
                if line_number < cursor:
                    next_offset = line_offset
                    line_number += 1
                    continue
                next_cursor = line_number + 1
                next_offset = line_offset
                line_number += 1
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                items.append(
                    RawQueueItem(
                        table=_validate_table(str(payload["table"])),
                        row=dict(payload["row"]),
                        published_at_ms=int(payload["published_at_ms"]),
                    )
                )
        return items, next_cursor, next_offset

    def read_from_offset(
        self,
        offset: int = 0,
        *,
        max_records: int | None = None,
    ) -> tuple[list[RawQueueItem], int, int]:
        """Read from a file offset and return ``(items, next_offset, lines_read)``."""

        if offset < 0:
            raise ValueError("offset must be non-negative")
        if max_records is not None and max_records < 0:
            raise ValueError("max_records must be non-negative")
        if not self.path.exists():
            return [], offset, 0

        items: list[RawQueueItem] = []
        lines_read = 0
        next_offset = offset
        with self.path.open("r", encoding="utf-8") as fp:
            fp.seek(offset)
            while True:
                if max_records is not None and len(items) >= max_records:
                    break
                line = fp.readline()
                if not line:
                    break
                lines_read += 1
                next_offset = fp.tell()
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                items.append(
                    RawQueueItem(
                        table=_validate_table(str(payload["table"])),
                        row=dict(payload["row"]),
                        published_at_ms=int(payload["published_at_ms"]),
                    )
                )
        return items, next_offset, lines_read


class IncrementalBtc15mFeaturePath:
    """Maintain BTC-15M raw state and emit changed feature rows incrementally."""

    def __init__(
        self,
        *,
        ingest_ts: int | None = None,
        canonical_symbol_prefix: str = "BTC-15M:",
        quality_config: FeatureQualityConfig = DEFAULT_QUALITY_CONFIG,
        bucket_ms: int | None = None,
    ) -> None:
        self._fixed_ingest_ts = None if ingest_ts is None else int(ingest_ts)
        self._canonical_symbol_prefix = canonical_symbol_prefix.upper()
        self._quality_config = quality_config
        self._bucket_ms = bucket_ms
        self._top_of_book: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self._orderbook: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self._trades: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self._latest_features: dict[tuple[int, str, str], dict[str, Any]] = {}

    def apply_queue_item(self, item: RawQueueItem) -> list[dict[str, Any]]:
        """Apply one queue item and return new or changed feature rows."""

        return self.apply_table_row(item.table, item.row)

    def apply_queue_items(self, items: Sequence[RawQueueItem]) -> list[dict[str, Any]]:
        """Apply many queue items and return all changed feature rows in order."""

        dirty_keys: set[tuple[str, str]] = set()
        for item in items:
            key = self._append_table_row(item.table, item.row)
            if key is not None:
                dirty_keys.add(key)
        changed: list[dict[str, Any]] = []
        for key in sorted(dirty_keys):
            changed.extend(self._recompute_key(key))
        return _sort_feature_rows(changed)

    def apply_table_row(
        self,
        table: str,
        row: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Apply one canonical raw table row and return changed feature rows."""

        key = self._append_table_row(table, row)
        if key is None:
            return []
        return self._recompute_key(key)

    def _append_table_row(
        self,
        table: str,
        row: Mapping[str, Any],
    ) -> tuple[str, str] | None:
        """Append one target raw row to state and return its aggregation key."""

        table = _validate_table(table)
        row_dict = dict(row)
        if not self._is_target_row(row_dict):
            return None
        key = _row_key(row_dict)
        if key is None:
            return None
        if table == "raw_top_of_book":
            self._top_of_book[key].append(row_dict)
        elif table == "raw_orderbook_snapshot":
            self._orderbook[key].append(row_dict)
        else:
            self._trades[key].append(row_dict)
        return key

    def latest_feature_rows(self) -> list[dict[str, Any]]:
        """Return the current latest feature view sorted like batch output."""

        return _sort_feature_rows(list(self._latest_features.values()))

    def prune_expired(self, *, emit_until_ms: int) -> None:
        """Drop raw/feature state for rounds that can no longer trade."""

        expired_keys = {
            key
            for key in set(self._top_of_book) | set(self._orderbook) | set(self._trades)
            if _group_expired(
                self._top_of_book.get(key, []),
                self._orderbook.get(key, []),
                self._trades.get(key, []),
                emit_until_ms=emit_until_ms,
            )
        }
        for key in expired_keys:
            self._top_of_book.pop(key, None)
            self._orderbook.pop(key, None)
            self._trades.pop(key, None)

        for feature_key, row in list(self._latest_features.items()):
            _, source, source_symbol = feature_key
            if (source, source_symbol) in expired_keys or _feature_row_expired(
                row,
                emit_until_ms=emit_until_ms,
            ):
                self._latest_features.pop(feature_key, None)

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
        *,
        ingest_ts: int | None = None,
        canonical_symbol_prefix: str = "BTC-15M:",
        quality_config: FeatureQualityConfig = DEFAULT_QUALITY_CONFIG,
        bucket_ms: int | None = None,
    ) -> IncrementalBtc15mFeaturePath:
        """Restore an incremental path from ``to_state`` output."""

        path = cls(
            ingest_ts=ingest_ts,
            canonical_symbol_prefix=canonical_symbol_prefix,
            quality_config=quality_config,
            bucket_ms=bucket_ms,
        )
        path._top_of_book = _group_state_rows(state.get("top_of_book_rows"))
        path._orderbook = _group_state_rows(state.get("orderbook_rows"))
        path._trades = _group_state_rows(state.get("trade_rows"))
        latest = state.get("latest_feature_rows")
        if isinstance(latest, Sequence) and not isinstance(latest, (str, bytes)):
            for row in latest:
                if not isinstance(row, Mapping):
                    continue
                row_dict = dict(row)
                key = _feature_key(row_dict)
                if key is not None:
                    path._latest_features[key] = row_dict
        return path

    def to_state(self) -> dict[str, Any]:
        """Return JSON-serializable state for the next queue batch."""

        return {
            "top_of_book_rows": _flatten_grouped_rows(self._top_of_book),
            "orderbook_rows": _flatten_grouped_rows(self._orderbook),
            "trade_rows": _flatten_grouped_rows(self._trades),
            "latest_feature_rows": self.latest_feature_rows(),
        }

    def _recompute_key(self, key: tuple[str, str]) -> list[dict[str, Any]]:
        aggregate_kwargs: dict[str, Any] = {}
        if self._bucket_ms is not None:
            aggregate_kwargs["bucket_ms"] = self._bucket_ms
        rows = aggregate_features_15m_v1(
            top_of_book_rows=self._top_of_book.get(key, []),
            orderbook_rows=self._orderbook.get(key, []),
            trade_rows=self._trades.get(key, []),
            ingest_ts=self._ingest_ts(),
            quality_config=self._quality_config,
            **aggregate_kwargs,
        )
        changed: list[dict[str, Any]] = []
        for row in rows:
            feature_key = _feature_key(row)
            if feature_key is None:
                continue
            previous = self._latest_features.get(feature_key)
            if previous is None or _feature_content(previous) != _feature_content(row):
                self._latest_features[feature_key] = row
                changed.append(row)
        return _sort_feature_rows(changed)

    def _is_target_row(self, row: Mapping[str, Any]) -> bool:
        canonical = str(row.get("canonical_symbol") or "").upper()
        return canonical.startswith(self._canonical_symbol_prefix)

    def _ingest_ts(self) -> int:
        if self._fixed_ingest_ts is not None:
            return self._fixed_ingest_ts
        return int(time.time() * 1000)


def run_low_latency_feature_queue_batch(
    warehouse_dir: Path | str,
    queue_path: Path | str,
    *,
    cursor_path: Path | str | None = None,
    state_path: Path | str | None = None,
    max_records: int | None = None,
    max_rows_per_partition: int = 50_000,
    ingest_ts: int | None = None,
    canonical_symbol_prefix: str = "BTC-15M:",
    bucket_seconds: float | None = None,
) -> LowLatencyFeatureQueueReport:
    """Consume queued BTC-15M raw rows and append changed feature rows."""

    bucket_ms = None
    if bucket_seconds is not None:
        if bucket_seconds <= 0:
            raise ValueError("bucket_seconds must be positive")
        bucket_ms = int(round(bucket_seconds * 1000))
        if bucket_ms <= 0:
            raise ValueError("bucket_seconds must be positive")
    cursor_file = None if cursor_path is None else Path(cursor_path)
    state_file = None if state_path is None else Path(state_path)
    emit_until_ms = int(time.time() * 1000) if ingest_ts is None else int(ingest_ts)
    start_cursor = _read_cursor(cursor_file)
    queue = JsonlRawQueue(queue_path)
    state = _read_state(state_file)
    start_offset = _read_queue_progress_offset(state, queue.path, start_cursor)
    if start_offset is None:
        items, next_cursor, next_offset = queue.read_from_line_cursor(
            start_cursor,
            max_records=max_records,
        )
    else:
        items, next_offset, lines_read = queue.read_from_offset(
            start_offset,
            max_records=max_records,
        )
        next_cursor = start_cursor + lines_read
    emitted_signatures = _read_emitted_signatures(state)
    path = IncrementalBtc15mFeaturePath.from_state(
        state,
        ingest_ts=emit_until_ms,
        canonical_symbol_prefix=canonical_symbol_prefix,
        bucket_ms=bucket_ms,
    )
    path.apply_queue_items(items)
    rows_to_write, emitted_signatures = _mature_feature_rows_to_write(
        path.latest_feature_rows(),
        emitted_signatures=emitted_signatures,
        emit_until_ms=emit_until_ms,
    )
    with WarehouseWriter(
        warehouse_dir,
        max_rows_per_partition=max_rows_per_partition,
    ) as writer:
        writer.append_rows("features_15m_v1", rows_to_write)
        writer.flush("features_15m_v1")
        rows_written = writer.stats.rows_written.get("features_15m_v1", 0)
    path.prune_expired(emit_until_ms=emit_until_ms)
    next_state = path.to_state()
    next_state["emitted_feature_signatures"] = emitted_signatures
    next_state["raw_queue_progress"] = _raw_queue_progress(
        queue.path,
        line_cursor=next_cursor,
        byte_offset=next_offset,
    )
    _write_state(state_file, next_state)
    _write_cursor(cursor_file, next_cursor)
    return LowLatencyFeatureQueueReport(
        rows_read=len(items),
        rows_generated=len(rows_to_write),
        rows_written=rows_written,
        start_cursor=start_cursor,
        next_cursor=next_cursor,
    )


def _validate_table(table: str) -> str:
    if table not in RAW_QUEUE_TABLES:
        raise ValueError(f"unsupported low-latency raw table: {table}")
    return table


def _row_key(row: Mapping[str, Any]) -> tuple[str, str] | None:
    source = row.get("source")
    source_symbol = row.get("source_symbol")
    if not source or not source_symbol:
        return None
    return str(source), str(source_symbol)


def _feature_key(row: Mapping[str, Any]) -> tuple[int, str, str] | None:
    feature_ts = row.get("feature_ts")
    source = row.get("source")
    source_symbol = row.get("source_symbol")
    if feature_ts is None or not source or not source_symbol:
        return None
    return int(feature_ts), str(source), str(source_symbol)


def _feature_content(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "ingest_ts"}


def _feature_key_text(row: Mapping[str, Any]) -> str | None:
    key = _feature_key(row)
    if key is None:
        return None
    feature_ts, source, source_symbol = key
    return f"{feature_ts}|{source}|{source_symbol}"


def _feature_signature(row: Mapping[str, Any]) -> str:
    return json.dumps(_feature_content(row), sort_keys=True, separators=(",", ":"))


def _read_emitted_signatures(state: Mapping[str, Any]) -> dict[str, str]:
    value = state.get("emitted_feature_signatures")
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(signature) for key, signature in value.items()}


def _raw_queue_progress(
    path: Path,
    *,
    line_cursor: int,
    byte_offset: int,
) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "line_cursor": int(line_cursor),
        "byte_offset": int(byte_offset),
    }


def _read_queue_progress_offset(
    state: Mapping[str, Any],
    path: Path,
    line_cursor: int,
) -> int | None:
    progress = state.get("raw_queue_progress")
    if not isinstance(progress, Mapping):
        return None
    if str(progress.get("path") or "") != str(path.resolve()):
        return None
    try:
        stored_line_cursor = int(progress.get("line_cursor"))
        byte_offset = int(progress.get("byte_offset"))
    except (TypeError, ValueError):
        return None
    if stored_line_cursor != int(line_cursor) or byte_offset < 0:
        return None
    if path.exists() and byte_offset > path.stat().st_size:
        return None
    return byte_offset


def _mature_feature_rows_to_write(
    rows: Sequence[dict[str, Any]],
    *,
    emitted_signatures: Mapping[str, str],
    emit_until_ms: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    updated_signatures = dict(emitted_signatures)
    rows_to_write: list[dict[str, Any]] = []
    for row in _sort_feature_rows(list(rows)):
        if not _is_tradable_feature_row(row):
            continue
        if int(row["feature_ts"]) > int(emit_until_ms):
            continue
        key = _feature_key_text(row)
        if key is None:
            continue
        signature = _feature_signature(row)
        if updated_signatures.get(key) == signature:
            continue
        row_to_write = dict(row)
        row_to_write["ingest_ts"] = int(emit_until_ms)
        rows_to_write.append(row_to_write)
        updated_signatures[key] = signature
    return rows_to_write, updated_signatures


def _is_tradable_feature_row(row: Mapping[str, Any]) -> bool:
    feature_ts = _optional_int(row.get("feature_ts"))
    if feature_ts is None:
        return False
    canonical_symbol = str(row.get("canonical_symbol") or row.get("symbol") or "")
    round_start_ts = round_start_ts_from_canonical_symbol(canonical_symbol)
    round_end_ts = round_end_ts_from_canonical_symbol(canonical_symbol)
    if (
        round_start_ts is not None
        and round_end_ts is not None
        and (feature_ts < round_start_ts or feature_ts >= round_end_ts)
    ):
        return False

    market = _optional_float(row.get("market_implied_prob"))
    if market is None or market <= 0.0 or market >= 1.0:
        return False
    return not is_degenerate_quote(row, market_implied_prob=market)


def _group_expired(
    *groups: Sequence[Mapping[str, Any]],
    emit_until_ms: int,
) -> bool:
    for group in groups:
        for row in group:
            if not _feature_row_expired(row, emit_until_ms=emit_until_ms):
                return False
    return any(groups)


def _feature_row_expired(row: Mapping[str, Any], *, emit_until_ms: int) -> bool:
    canonical_symbol = str(row.get("canonical_symbol") or row.get("symbol") or "")
    round_end_ts = round_end_ts_from_canonical_symbol(canonical_symbol)
    return round_end_ts is not None and int(emit_until_ms) >= round_end_ts


def _sort_feature_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (row["feature_ts"], row["source"], row["source_symbol"]))


def _group_state_rows(value: Any) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return grouped
    for row in value:
        if not isinstance(row, Mapping):
            continue
        row_dict = dict(row)
        key = _row_key(row_dict)
        if key is not None:
            grouped[key].append(row_dict)
    return grouped


def _flatten_grouped_rows(
    grouped: Mapping[tuple[str, str], Sequence[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = [dict(row) for group in grouped.values() for row in group]
    return sorted(rows, key=lambda row: (row.get("source"), row.get("source_symbol"), row.get("ts")))


def _read_cursor(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    text = path.read_text(encoding="utf-8").strip()
    return 0 if not text else int(text)


def _write_cursor(path: Path | None, cursor: int) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(f"{int(cursor)}\n", encoding="utf-8")
    tmp.replace(path)


def _read_state(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_state(path: Path | None, state: Mapping[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    tmp.replace(path)


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
