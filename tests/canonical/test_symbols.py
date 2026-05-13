"""Unit tests for issue #22 symbol mapping helpers."""

from __future__ import annotations

from pathlib import Path

import orjson

from bigan.canonical.symbols import (
    SymbolMapper,
    load_symbol_mapping_rows,
    symbol_mapping_row,
)


def test_symbol_mapper_resolves_market_specific_before_generic() -> None:
    mapper = SymbolMapper(
        [
            symbol_mapping_row(
                source="polymarket",
                source_symbol="tok-1",
                canonical_symbol="generic-election-yes",
                effective_from_ts=100,
            ),
            symbol_mapping_row(
                source="polymarket",
                source_symbol="tok-1",
                source_market="0xmkt",
                canonical_symbol="election-2026-yes",
                effective_from_ts=100,
            ),
        ]
    )

    assert (
        mapper.resolve(
            source="polymarket",
            source_symbol="tok-1",
            source_market="0xmkt",
            ts=101,
        )
        == "election-2026-yes"
    )
    assert (
        mapper.resolve(source="polymarket", source_symbol="tok-1", ts=101)
        == "generic-election-yes"
    )


def test_symbol_mapper_honors_effective_window() -> None:
    mapper = SymbolMapper(
        [
            symbol_mapping_row(
                source="polymarket",
                source_symbol="tok-1",
                canonical_symbol="old-name",
                effective_from_ts=100,
                effective_to_ts=200,
            ),
            symbol_mapping_row(
                source="polymarket",
                source_symbol="tok-1",
                canonical_symbol="new-name",
                effective_from_ts=200,
            ),
        ]
    )

    assert (
        mapper.resolve(source="polymarket", source_symbol="tok-1", ts=199)
        == "old-name"
    )
    assert (
        mapper.resolve(source="polymarket", source_symbol="tok-1", ts=200)
        == "new-name"
    )
    assert mapper.resolve(source="polymarket", source_symbol="tok-1", ts=99) is None


def test_enrich_row_fills_missing_canonical_symbol_only() -> None:
    mapper = SymbolMapper(
        [
            symbol_mapping_row(
                source="polymarket",
                source_symbol="tok-1",
                canonical_symbol="mapped-symbol",
                effective_from_ts=0,
            )
        ]
    )

    row = {
        "ts": 10,
        "source": "polymarket",
        "source_symbol": "tok-1",
        "canonical_symbol": None,
    }
    assert mapper.enrich_row(row)["canonical_symbol"] == "mapped-symbol"

    prefilled = {**row, "canonical_symbol": "already-set"}
    assert mapper.enrich_row(prefilled)["canonical_symbol"] == "already-set"


def test_load_symbol_mapping_rows_from_json_and_csv(tmp_path: Path) -> None:
    json_path = tmp_path / "mappings.json"
    json_path.write_bytes(
        orjson.dumps(
            {
                "mappings": [
                    {
                        "source": "polymarket",
                        "source_symbol": "tok-1",
                        "canonical_symbol": "json-symbol",
                        "effective_from_ts": 10,
                    }
                ]
            }
        )
    )

    csv_path = tmp_path / "mappings.csv"
    csv_path.write_text(
        "source,source_symbol,canonical_symbol,effective_from_ts\n"
        "polymarket,tok-2,csv-symbol,20\n",
        encoding="utf-8",
    )

    rows = load_symbol_mapping_rows(tmp_path)
    by_symbol = {row["source_symbol"]: row["canonical_symbol"] for row in rows}
    assert by_symbol == {"tok-1": "json-symbol", "tok-2": "csv-symbol"}
