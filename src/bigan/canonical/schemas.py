"""PyArrow schemas for the four canonical raw tables (issue #3).

All four tables follow the same column-naming contract:

- **Timestamps** (per #23 ``Timestamp Contract``):
  - ``ts``         — server / event time (UTC ms epoch)
  - ``message_ts`` — protocol-level timestamp inside the original payload
                     (UTC ms epoch). For Polymarket this equals ``ts``;
                     it exists as a separate column so multi-source pipelines
                     can keep wire-level provenance distinct from canonical
                     event time.
  - ``ingest_ts``  — local receive time when our WS client decoded the frame
                     (UTC ms epoch).

- **Symbol identity** (per #22 ``Symbol Mapping Table``):
  - ``source``           — exchange / data-source key (e.g. ``polymarket``)
  - ``source_symbol``    — source-specific raw identifier (Polymarket
                           ``asset_id`` / CLOB token id)
  - ``source_market``    — source-specific market grouping (Polymarket
                           ``condition_id``-style market hash)
  - ``canonical_symbol`` — cross-source canonical name. Nullable on raw rows
                           when a mapping is not available yet; non-null in
                           the ``symbol_mapping`` table itself.

- **Append-only**: every ETL run writes a new ``part-*.parquet`` file under
  the appropriate Hive partition directory. Existing files are never edited
  or deleted. Schema migrations bump ``SCHEMA_VERSION`` in file metadata.

Numeric encoding uses ``float64`` for v1. Polymarket prices are tick-aligned
(default 0.01) and live in [0, 1]; float64 has well over 14 decimal digits
of precision and aligns trivially with DuckDB / pandas. A future migration
to ``decimal128(18, 9)`` is documented as out-of-scope for #3.
"""

from __future__ import annotations

import pyarrow as pa

#: Bumped on incompatible schema changes. Written into Parquet file
#: KeyValueMetadata so consumers can hard-fail on unknown versions.
SCHEMA_VERSION = "1"


# Common identity columns (timestamp + symbol contract).
#
# ``provenance`` (#5) tags the upstream channel that produced each row:
#   - NULL or "ws"                  → realtime WebSocket stream
#   - "polymarket-rest-backfill"    → injected by the gap-recovery backfill
#   - "manual"                      → operator-triggered replay via CLI
# Downstream consumers can filter or weight rows by provenance for
# robustness analysis. Nullable to keep historical Parquet readable.
_COMMON_IDENTITY_FIELDS: list[pa.Field] = [
    pa.field("ts", pa.int64(), nullable=False),
    pa.field("message_ts", pa.int64(), nullable=False),
    pa.field("ingest_ts", pa.int64(), nullable=False),
    pa.field("source", pa.string(), nullable=False),
    pa.field("source_symbol", pa.string(), nullable=False),
    pa.field("source_market", pa.string(), nullable=True),
    pa.field("canonical_symbol", pa.string(), nullable=True),
    pa.field("provenance", pa.string(), nullable=True),
]


#: Provenance tag values used across the codebase.
PROVENANCE_WS = "ws"
PROVENANCE_BACKFILL = "polymarket-rest-backfill"
PROVENANCE_MANUAL = "manual"


def _table_metadata(table_name: str) -> dict[bytes, bytes]:
    return {
        b"bigan.schema_version": SCHEMA_VERSION.encode("ascii"),
        b"bigan.table_name": table_name.encode("ascii"),
    }


# ---------------------------------------------------------------------------
# raw_top_of_book — one row per ``best_bid_ask`` event
# ---------------------------------------------------------------------------

SCHEMA_RAW_TOP_OF_BOOK: pa.Schema = pa.schema(
    [
        *_COMMON_IDENTITY_FIELDS,
        pa.field("bid_price", pa.float64(), nullable=True),
        pa.field("ask_price", pa.float64(), nullable=True),
        pa.field("spread", pa.float64(), nullable=True),
    ],
    metadata=_table_metadata("raw_top_of_book"),
)


# ---------------------------------------------------------------------------
# raw_orderbook_snapshot — one row per (snapshot, side, level)
# ---------------------------------------------------------------------------
#
# Long format: a single ``book`` event with N bids + M asks expands into
# N + M rows that share the same (ts, source_symbol, snapshot_hash). The
# ``level`` field is 0-indexed depth where 0 = best price on that side, 1 =
# next best, etc. This format is more SQL-friendly than struct-of-arrays
# and trivially supports top-K queries via ``WHERE level < K``.
#
# Bid levels are sorted descending by price; ask levels ascending.

SCHEMA_RAW_ORDERBOOK_SNAPSHOT: pa.Schema = pa.schema(
    [
        *_COMMON_IDENTITY_FIELDS,
        pa.field("side", pa.string(), nullable=False),  # "BID" or "ASK"
        pa.field("level", pa.int32(), nullable=False),
        pa.field("price", pa.float64(), nullable=False),
        pa.field("size", pa.float64(), nullable=False),
        pa.field("snapshot_hash", pa.string(), nullable=True),
    ],
    metadata=_table_metadata("raw_orderbook_snapshot"),
)


# ---------------------------------------------------------------------------
# raw_trades — one row per ``last_trade_price`` event
# ---------------------------------------------------------------------------
#
# Polymarket's ``last_trade_price`` doesn't carry an exchange-assigned
# trade_id; we synthesize one as
#   ``polymarket-{source_symbol}-{ts}-{price}-{size}-{side}``
# so downstream dedup can run idempotently across re-ETL cycles without
# requiring a uniqueness contract on the exchange side.

SCHEMA_RAW_TRADES: pa.Schema = pa.schema(
    [
        *_COMMON_IDENTITY_FIELDS,
        pa.field("price", pa.float64(), nullable=False),
        pa.field("size", pa.float64(), nullable=False),
        pa.field("side", pa.string(), nullable=False),  # "BUY" or "SELL"
        pa.field("fee_rate_bps", pa.float64(), nullable=True),
        pa.field("trade_id", pa.string(), nullable=False),
    ],
    metadata=_table_metadata("raw_trades"),
)


# ---------------------------------------------------------------------------
# raw_candles_1m — derived 1-minute aggregations
# ---------------------------------------------------------------------------
#
# Aggregation grain: ``(source, source_symbol, bucket_ts)`` where
# ``bucket_ts`` is the start of a UTC minute (i.e. ``floor(ts, 60s)``).
#
# OHLC encodes both top-of-book quote evolution (``bid_*``, ``ask_*``,
# ``mid_*``) and traded-price evolution (``trade_*``). When no trades
# happened in the minute, trade_* columns are NULL but the bid/ask side
# is still populated provided ``best_bid_ask`` events were seen.
#
# ``ts`` and ``message_ts`` are the bucket start (``bucket_ts``); we keep
# them for partition-pruning consistency with raw_* tables. ``ts_close`` /
# ``ts_open`` retain the actual first / last update timestamp inside the
# bucket so consumers can detect quiet minutes.

SCHEMA_RAW_CANDLES_1M: pa.Schema = pa.schema(
    [
        pa.field("ts", pa.int64(), nullable=False),  # == bucket_ts
        pa.field("message_ts", pa.int64(), nullable=False),  # == bucket_ts
        pa.field("ingest_ts", pa.int64(), nullable=False),  # == ETL run time
        pa.field("source", pa.string(), nullable=False),
        pa.field("source_symbol", pa.string(), nullable=False),
        pa.field("source_market", pa.string(), nullable=True),
        pa.field("canonical_symbol", pa.string(), nullable=True),
        pa.field("bucket_ts", pa.int64(), nullable=False),
        pa.field("ts_open", pa.int64(), nullable=True),
        pa.field("ts_close", pa.int64(), nullable=True),
        pa.field("mid_open", pa.float64(), nullable=True),
        pa.field("mid_high", pa.float64(), nullable=True),
        pa.field("mid_low", pa.float64(), nullable=True),
        pa.field("mid_close", pa.float64(), nullable=True),
        pa.field("bid_open", pa.float64(), nullable=True),
        pa.field("bid_high", pa.float64(), nullable=True),
        pa.field("bid_low", pa.float64(), nullable=True),
        pa.field("bid_close", pa.float64(), nullable=True),
        pa.field("ask_open", pa.float64(), nullable=True),
        pa.field("ask_high", pa.float64(), nullable=True),
        pa.field("ask_low", pa.float64(), nullable=True),
        pa.field("ask_close", pa.float64(), nullable=True),
        pa.field("trade_open", pa.float64(), nullable=True),
        pa.field("trade_high", pa.float64(), nullable=True),
        pa.field("trade_low", pa.float64(), nullable=True),
        pa.field("trade_close", pa.float64(), nullable=True),
        pa.field("trade_volume", pa.float64(), nullable=True),
        pa.field("trade_count", pa.int32(), nullable=False),
        pa.field("top_of_book_count", pa.int32(), nullable=False),
        pa.field("vwap", pa.float64(), nullable=True),
    ],
    metadata=_table_metadata("raw_candles_1m"),
)


# ---------------------------------------------------------------------------
# symbol_mapping — source-native symbol -> canonical symbol lookup
# ---------------------------------------------------------------------------
#
# ``effective_from_ts`` / ``effective_to_ts`` make the table temporal so a
# source can rename or recycle identifiers without rewriting old raw rows.
# ``ts`` mirrors ``effective_from_ts`` for warehouse partitioning.

SCHEMA_SYMBOL_MAPPING: pa.Schema = pa.schema(
    [
        pa.field("ts", pa.int64(), nullable=False),
        pa.field("message_ts", pa.int64(), nullable=False),
        pa.field("ingest_ts", pa.int64(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("source_symbol", pa.string(), nullable=False),
        pa.field("source_market", pa.string(), nullable=True),
        pa.field("canonical_symbol", pa.string(), nullable=False),
        pa.field("effective_from_ts", pa.int64(), nullable=False),
        pa.field("effective_to_ts", pa.int64(), nullable=True),
        pa.field("symbol_kind", pa.string(), nullable=True),
        pa.field("metadata_json", pa.string(), nullable=True),
    ],
    metadata=_table_metadata("symbol_mapping"),
)


# ---------------------------------------------------------------------------
# quarantine — abnormal rows isolated by the validation layer (issue #4)
# ---------------------------------------------------------------------------
#
# Any canonical row that fails a validation rule is rerouted from its target
# raw table into ``quarantine`` along with a machine-readable ``rule`` tag,
# a free-form ``detail`` message, and the original payload as JSON. This
# preserves auditability without polluting the main raw_* tables.
#
# Identity columns mirror the raw_* tables so the quarantine table can be
# joined back into them by (source, source_symbol, ts).
#
# ``ts`` retains the offending row's event time when available, falling back
# to ``ingest_ts`` (validator detection time) when ``ts`` is missing/zero.
# Partitioning therefore still works for both well-formed and malformed
# inputs.

SCHEMA_QUARANTINE: pa.Schema = pa.schema(
    [
        *_COMMON_IDENTITY_FIELDS,
        pa.field("target_table", pa.string(), nullable=False),
        pa.field("rule", pa.string(), nullable=False),
        pa.field("detail", pa.string(), nullable=True),
        pa.field("payload_json", pa.string(), nullable=False),
    ],
    metadata=_table_metadata("quarantine"),
)


#: Stable mapping ``table_name -> pyarrow.Schema``. Kept as a tuple of pairs
#: so the iteration order is deterministic.
TABLE_NAMES: tuple[str, ...] = (
    "raw_top_of_book",
    "raw_orderbook_snapshot",
    "raw_trades",
    "raw_candles_1m",
    "symbol_mapping",
    "quarantine",
)


SCHEMAS: dict[str, pa.Schema] = {
    "raw_top_of_book": SCHEMA_RAW_TOP_OF_BOOK,
    "raw_orderbook_snapshot": SCHEMA_RAW_ORDERBOOK_SNAPSHOT,
    "raw_trades": SCHEMA_RAW_TRADES,
    "raw_candles_1m": SCHEMA_RAW_CANDLES_1M,
    "symbol_mapping": SCHEMA_SYMBOL_MAPPING,
    "quarantine": SCHEMA_QUARANTINE,
}
