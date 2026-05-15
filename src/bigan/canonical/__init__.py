"""Canonical raw-data layer for BiGan.

Issue #3 deliverable: append-only Parquet tables that downstream feature /
model / backtest jobs read instead of source-specific WebSocket payloads.

Design choices live in module docstrings; the high-level summary is in
``docs/adr/0001-market-data-source.md`` and the README ``Canonical layer``
section.
"""

from .schemas import (
    SCHEMA_FEATURES_15M_V1,
    SCHEMA_QUARANTINE,
    SCHEMA_RAW_CANDLES_1M,
    SCHEMA_RAW_ORACLE_PRICE,
    SCHEMA_RAW_ORDERBOOK_SNAPSHOT,
    SCHEMA_RAW_SPOT_PRICE,
    SCHEMA_RAW_TOP_OF_BOOK,
    SCHEMA_RAW_TRADES,
    SCHEMA_SYMBOL_MAPPING,
    SCHEMA_VERSION,
    TABLE_NAMES,
)
from .symbols import (
    SymbolMapper,
    SymbolMappingEntry,
    load_symbol_mapping_rows,
    symbol_mapping_row,
)
from .transform import (
    derive_top_of_book_from_book,
    derive_top_of_book_from_price_change,
    transform_book_event,
    transform_event,
    transform_last_trade_price_event,
    transform_top_of_book_event,
)
from .validation import (
    RowValidator,
    ValidationError,
    ValidationRule,
    ValidationStats,
)

__all__ = [
    "SCHEMA_QUARANTINE",
    "SCHEMA_RAW_CANDLES_1M",
    "SCHEMA_RAW_ORACLE_PRICE",
    "SCHEMA_RAW_ORDERBOOK_SNAPSHOT",
    "SCHEMA_RAW_SPOT_PRICE",
    "SCHEMA_RAW_TOP_OF_BOOK",
    "SCHEMA_RAW_TRADES",
    "SCHEMA_FEATURES_15M_V1",
    "SCHEMA_VERSION",
    "SCHEMA_SYMBOL_MAPPING",
    "TABLE_NAMES",
    "RowValidator",
    "SymbolMapper",
    "SymbolMappingEntry",
    "ValidationError",
    "ValidationRule",
    "ValidationStats",
    "derive_top_of_book_from_book",
    "derive_top_of_book_from_price_change",
    "load_symbol_mapping_rows",
    "symbol_mapping_row",
    "transform_book_event",
    "transform_event",
    "transform_last_trade_price_event",
    "transform_top_of_book_event",
]
