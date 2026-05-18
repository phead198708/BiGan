"""Schema-level invariants for the canonical tables (issue #3)."""

from __future__ import annotations

import pyarrow as pa

from bigan.canonical.schemas import (
    SCHEMA_RAW_CANDLES_1M,
    SCHEMA_RAW_ORACLE_PRICE,
    SCHEMA_RAW_ORDERBOOK_SNAPSHOT,
    SCHEMA_RAW_SPOT_PRICE,
    SCHEMA_RAW_TOP_OF_BOOK,
    SCHEMA_RAW_TRADES,
    SCHEMA_SYMBOL_MAPPING,
    SCHEMA_VERSION,
    SCHEMAS,
    TABLE_NAMES,
)


def test_table_names_match_schemas_dict() -> None:
    assert tuple(SCHEMAS) == TABLE_NAMES


def test_all_tables_carry_three_timestamp_columns() -> None:
    """Per #23: every raw table must expose ts / message_ts / ingest_ts."""
    required = {"ts", "message_ts", "ingest_ts"}
    for name, schema in SCHEMAS.items():
        cols = {f.name for f in schema}
        missing = required - cols
        assert not missing, f"{name} missing timestamp columns: {missing}"


def test_timestamp_columns_are_int64() -> None:
    """Timestamp contract stores UTC ms epochs, never timezone-bearing strings."""
    timestamp_cols = {
        "ts",
        "message_ts",
        "ingest_ts",
        "capture_timestamp_ms",
        "bucket_ts",
        "ts_open",
        "ts_close",
        "effective_from_ts",
        "effective_to_ts",
    }
    for name, schema in SCHEMAS.items():
        for field in schema:
            if field.name in timestamp_cols:
                assert field.type == pa.int64(), f"{name}.{field.name} must be int64"


def test_all_tables_carry_symbol_identity_columns() -> None:
    """Per #22: every raw table must expose source / source_symbol / canonical_symbol."""
    required = {"source", "source_symbol", "canonical_symbol"}
    for name, schema in SCHEMAS.items():
        cols = {f.name for f in schema}
        missing = required - cols
        assert not missing, f"{name} missing symbol columns: {missing}"


def test_raw_tables_allow_unmapped_canonical_symbol() -> None:
    """Raw rows can still be ingested before a source symbol is mapped."""
    for name, schema in SCHEMAS.items():
        if name == "symbol_mapping":
            continue
        idx = schema.get_field_index("canonical_symbol")
        assert idx >= 0
        assert schema.field(idx).nullable, "canonical_symbol must be nullable"


def test_symbol_mapping_requires_canonical_symbol() -> None:
    idx = SCHEMA_SYMBOL_MAPPING.get_field_index("canonical_symbol")
    assert idx >= 0
    assert not SCHEMA_SYMBOL_MAPPING.field(idx).nullable


def test_symbol_mapping_has_temporal_lookup_columns() -> None:
    cols = {f.name for f in SCHEMA_SYMBOL_MAPPING}
    assert {
        "effective_from_ts",
        "effective_to_ts",
        "symbol_kind",
        "metadata_json",
    } <= cols


def test_source_and_source_symbol_are_not_nullable() -> None:
    """We always know which source produced a row."""
    for name, schema in SCHEMAS.items():
        for col in ("source", "source_symbol"):
            idx = schema.get_field_index(col)
            assert idx >= 0, f"{name} missing {col}"
            assert not schema.field(idx).nullable, f"{name}.{col} must not be nullable"


def test_raw_event_tables_carry_nullable_source_channel() -> None:
    """Issue #29 keeps transport channel separate from provenance."""
    for schema in (
        SCHEMA_RAW_TOP_OF_BOOK,
        SCHEMA_RAW_ORDERBOOK_SNAPSHOT,
        SCHEMA_RAW_TRADES,
        SCHEMA_RAW_SPOT_PRICE,
        SCHEMA_RAW_ORACLE_PRICE,
        SCHEMAS["quarantine"],
    ):
        idx = schema.get_field_index("source_channel")
        assert idx >= 0
        assert schema.field(idx).type == pa.string()
        assert schema.field(idx).nullable


def test_polymarket_raw_tables_carry_nullable_capture_timestamp_ms() -> None:
    """Issue #30 persists the raw sink capture timestamp for audits."""
    for schema in (
        SCHEMA_RAW_TOP_OF_BOOK,
        SCHEMA_RAW_ORDERBOOK_SNAPSHOT,
        SCHEMA_RAW_TRADES,
    ):
        idx = schema.get_field_index("capture_timestamp_ms")
        assert idx >= 0
        assert schema.field(idx).type == pa.int64()
        assert schema.field(idx).nullable
        assert schema.get_field_index("source_timestamp_ms") == -1


def test_top_of_book_has_quote_columns() -> None:
    cols = {f.name for f in SCHEMA_RAW_TOP_OF_BOOK}
    assert {"bid_price", "ask_price", "spread"} <= cols


def test_orderbook_snapshot_has_level_columns() -> None:
    cols = {f.name for f in SCHEMA_RAW_ORDERBOOK_SNAPSHOT}
    assert {"side", "level", "price", "size", "snapshot_hash"} <= cols
    assert SCHEMA_RAW_ORDERBOOK_SNAPSHOT.field("level").type == pa.int32()


def test_trades_has_trade_columns() -> None:
    cols = {f.name for f in SCHEMA_RAW_TRADES}
    assert {"price", "size", "side", "fee_rate_bps", "trade_id"} <= cols


def test_candles_has_full_ohlc_columns() -> None:
    cols = {f.name for f in SCHEMA_RAW_CANDLES_1M}
    for prefix in ("bid", "ask", "mid", "trade"):
        for stat in ("open", "high", "low", "close"):
            assert f"{prefix}_{stat}" in cols, f"{prefix}_{stat} missing"
    assert {"trade_volume", "trade_count", "top_of_book_count", "vwap", "bucket_ts"} <= cols


def test_spot_price_has_reference_price_columns() -> None:
    cols = {f.name for f in SCHEMA_RAW_SPOT_PRICE}
    assert {"price", "bid_price", "ask_price"} <= cols


def test_oracle_price_has_chainlink_columns() -> None:
    cols = {f.name for f in SCHEMA_RAW_ORACLE_PRICE}
    assert {"price", "answer", "decimals", "round_id", "answered_in_round"} <= cols
    assert SCHEMA_RAW_ORACLE_PRICE.field("decimals").type == pa.int32()


def test_schema_version_metadata_is_present() -> None:
    for name, schema in SCHEMAS.items():
        meta = schema.metadata or {}
        assert b"bigan.schema_version" in meta, f"{name} missing schema_version metadata"
        assert meta[b"bigan.schema_version"].decode() == SCHEMA_VERSION
        assert meta[b"bigan.table_name"].decode() == name
