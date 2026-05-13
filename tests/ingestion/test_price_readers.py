"""Tests for issue #24 reference-price readers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq

from bigan.canonical.symbols import SymbolMapper, symbol_mapping_row
from bigan.canonical.writer import warehouse_files
from bigan.ingestion.price_readers import (
    WarehousePriceSink,
    build_chainlink_oracle_row,
    decode_decimals,
    decode_latest_round_data,
    parse_coinbase_ticker_message,
    parse_kraken_ticker_message,
)


def _word(value: int) -> str:
    return f"{value:064x}"


def _ms(year: int, month: int, day: int, hour: int, minute: int, second: int, ms: int = 0) -> int:
    return int(
        datetime(year, month, day, hour, minute, second, ms * 1000, tzinfo=UTC).timestamp()
        * 1000
    )


def test_parse_coinbase_ticker_message() -> None:
    payload = {
        "channel": "ticker",
        "timestamp": "2026-05-10T12:00:00Z",
        "events": [
            {
                "type": "update",
                "tickers": [
                    {
                        "product_id": "BTC-USD",
                        "price": "65000.10",
                        "best_bid": "65000.00",
                        "best_ask": "65000.20",
                        "time": "2026-05-10T12:00:00.123456789Z",
                    }
                ],
            }
        ],
    }

    rows = parse_coinbase_ticker_message(payload, ingest_ts=1_778_412_000_500)

    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "coinbase"
    assert row["source_symbol"] == "BTC-USD"
    assert row["ts"] == _ms(2026, 5, 10, 12, 0, 0, 123)
    assert row["message_ts"] == _ms(2026, 5, 10, 12, 0, 0)
    assert row["price"] == 65000.10
    assert row["bid_price"] == 65000.00
    assert row["ask_price"] == 65000.20


def test_parse_kraken_ticker_message() -> None:
    payload = {
        "channel": "ticker",
        "type": "update",
        "data": [
            {
                "symbol": "BTC/USD",
                "bid": 65000.0,
                "ask": 65001.0,
                "last": 65000.5,
                "timestamp": "2026-05-10T12:00:01.123456Z",
            }
        ],
    }

    rows = parse_kraken_ticker_message(payload, ingest_ts=1_778_412_002_000)

    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "kraken"
    assert row["source_symbol"] == "BTC/USD"
    assert row["ts"] == _ms(2026, 5, 10, 12, 0, 1, 123)
    assert row["price"] == 65000.5
    assert row["bid_price"] == 65000.0
    assert row["ask_price"] == 65001.0


def test_decode_chainlink_latest_round_data() -> None:
    answer = 6_500_012_345_678
    updated_at = 1_700_000_000
    result = "0x" + "".join(
        [
            _word(42),
            _word(answer),
            _word(updated_at - 5),
            _word(updated_at),
            _word(42),
        ]
    )

    assert decode_decimals("0x" + _word(8)) == 8
    round_data = decode_latest_round_data(result)
    row = build_chainlink_oracle_row(
        source_symbol="BTC/USD",
        feed_address="0xfeed",
        decimals=8,
        round_data=round_data,
        ingest_ts=1_700_000_000_500,
    )

    assert row["source"] == "chainlink"
    assert row["source_symbol"] == "BTC/USD"
    assert row["source_market"] == "0xfeed"
    assert row["ts"] == 1_700_000_000_000
    assert row["answer"] == answer
    assert row["decimals"] == 8
    assert row["price"] == 65000.12345678
    assert row["round_id"] == "42"


async def test_warehouse_price_sink_applies_symbol_mapping(tmp_path: Path) -> None:
    mapper = SymbolMapper(
        [
            symbol_mapping_row(
                source="coinbase",
                source_symbol="BTC-USD",
                canonical_symbol="BTC/USD",
                effective_from_ts=0,
            )
        ]
    )
    sink = WarehousePriceSink(
        tmp_path,
        symbol_mapper=mapper,
        max_rows_per_partition=1,
    )

    await sink.write_price_row(
        "raw_spot_price",
        {
            "ts": 1_700_000_000_000,
            "message_ts": 1_700_000_000_000,
            "ingest_ts": 1_700_000_000_100,
            "source": "coinbase",
            "source_symbol": "BTC-USD",
            "source_market": None,
            "canonical_symbol": None,
            "provenance": "ws",
            "price": 65000.0,
            "bid_price": 64999.0,
            "ask_price": 65001.0,
        },
    )
    await sink.close()

    files = warehouse_files(tmp_path, "raw_spot_price")
    assert len(files) == 1
    row = pq.ParquetFile(files[0]).read().to_pylist()[0]
    assert row["canonical_symbol"] == "BTC/USD"
