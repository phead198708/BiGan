"""Canonical raw-data layer for BiGan.

Issue #3 deliverable: append-only Parquet tables (``raw_top_of_book``,
``raw_orderbook_snapshot``, ``raw_trades``, ``raw_candles_1m``) that
downstream feature / model / backtest jobs read instead of source-specific
WebSocket payloads.

Design choices live in module docstrings; the high-level summary is in
``docs/adr/0001-market-data-source.md`` and the README ``Canonical layer``
section.
"""

from .schemas import (
    SCHEMA_RAW_CANDLES_1M,
    SCHEMA_RAW_ORDERBOOK_SNAPSHOT,
    SCHEMA_RAW_TOP_OF_BOOK,
    SCHEMA_RAW_TRADES,
    SCHEMA_VERSION,
    TABLE_NAMES,
)
from .transform import (
    derive_top_of_book_from_book,
    derive_top_of_book_from_price_change,
    transform_book_event,
    transform_event,
    transform_last_trade_price_event,
    transform_top_of_book_event,
)

__all__ = [
    "SCHEMA_RAW_CANDLES_1M",
    "SCHEMA_RAW_ORDERBOOK_SNAPSHOT",
    "SCHEMA_RAW_TOP_OF_BOOK",
    "SCHEMA_RAW_TRADES",
    "SCHEMA_VERSION",
    "TABLE_NAMES",
    "derive_top_of_book_from_book",
    "derive_top_of_book_from_price_change",
    "transform_book_event",
    "transform_event",
    "transform_last_trade_price_event",
    "transform_top_of_book_event",
]
