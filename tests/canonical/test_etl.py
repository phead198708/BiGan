"""End-to-end test for the canonical ETL runner.

Builds a synthetic NDJSON.gz archive that contains one of every relevant
event type, runs :func:`run_etl_batch`, and asserts that all four canonical
tables receive rows.
"""

from __future__ import annotations

import gzip
from datetime import UTC, datetime
from pathlib import Path

import orjson
import pyarrow.parquet as pq

from bigan.canonical.etl import run_etl_batch
from bigan.canonical.symbols import SymbolMapper, symbol_mapping_row
from bigan.canonical.writer import warehouse_files


def _ts(year: int, month: int, day: int, hour: int, minute: int, second: int = 0) -> int:
    return int(
        datetime(year, month, day, hour, minute, second, tzinfo=UTC).timestamp() * 1000
    )


def _write_ndjson_gz(path: Path, records: list[dict]) -> None:
    with gzip.open(path, "wb") as fp:
        for rec in records:
            fp.write(orjson.dumps(rec) + b"\n")


def _trade_record(
    *,
    asset_id: str,
    market: str,
    ts: int,
    receive_time: int | None = None,
    price: str = "0.51",
    size: str = "10",
    side: str = "BUY",
    provenance: str | None = None,
) -> dict:
    raw = {
        "event_type": "last_trade_price",
        "asset_id": asset_id,
        "market": market,
        "price": price,
        "size": size,
        "side": side,
        "fee_rate_bps": "0",
        "timestamp": str(ts),
    }
    if provenance is not None:
        raw["provenance"] = provenance
    return {"receive_time": receive_time or ts + 100, "raw": raw}


def test_etl_round_trip_populates_all_four_tables(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    base = _ts(2026, 5, 10, 12, 0)
    src = raw_dir / "2026-05-10.ndjson.gz"
    _write_ndjson_gz(
        src,
        [
            {
                "receive_time": base + 100,
                "raw": {
                    "event_type": "book",
                    "asset_id": "tok-1",
                    "market": "0xmkt",
                    "timestamp": str(base),
                    "hash": "h0",
                    "bids": [
                        {"price": "0.49", "size": "100"},
                        {"price": "0.50", "size": "50"},
                    ],
                    "asks": [
                        {"price": "0.52", "size": "30"},
                        {"price": "0.53", "size": "60"},
                    ],
                },
            },
            {
                "receive_time": base + 200,
                "raw": {
                    "event_type": "best_bid_ask",
                    "asset_id": "tok-1",
                    "market": "0xmkt",
                    "best_bid": "0.50",
                    "best_ask": "0.52",
                    "spread": "0.02",
                    "timestamp": str(base + 100),
                },
            },
            {
                "receive_time": base + 30_000,
                "raw": {
                    "event_type": "best_bid_ask",
                    "asset_id": "tok-1",
                    "market": "0xmkt",
                    "best_bid": "0.51",
                    "best_ask": "0.53",
                    "spread": "0.02",
                    "timestamp": str(base + 29_000),
                },
            },
            {
                "receive_time": base + 35_000,
                "raw": {
                    "event_type": "last_trade_price",
                    "asset_id": "tok-1",
                    "market": "0xmkt",
                    "price": "0.51",
                    "size": "10",
                    "side": "BUY",
                    "fee_rate_bps": "0",
                    "timestamp": str(base + 34_000),
                },
            },
            # Event types we don't materialise should be silently ignored.
            {
                "receive_time": base + 40_000,
                "raw": {
                    "event_type": "tick_size_change",
                    "asset_id": "tok-1",
                    "market": "0xmkt",
                    "old_tick_size": "0.01",
                    "new_tick_size": "0.001",
                    "timestamp": str(base + 39_000),
                },
            },
        ],
    )

    # Backdate file mtime so the lag-seconds gate doesn't skip it.
    import os

    os.utime(src, (1, 1))

    warehouse = tmp_path / "warehouse"
    report = run_etl_batch(
        raw_dir=raw_dir, warehouse_dir=warehouse, lag_seconds=0.0
    )
    assert report.files_processed == 1
    assert report.records_read == 5
    # 2 explicit best_bid_ask events + 1 derived from the book snapshot.
    assert report.rows_per_table["raw_top_of_book"] == 3
    assert report.rows_per_table["raw_orderbook_snapshot"] == 4  # 2 bids + 2 asks
    assert report.rows_per_table["raw_trades"] == 1
    # 1 trade in same minute as 1 tob, plus another tob in next minute later? No —
    # second tob at base+29s is still in bucket 0 (minute 0). All in bucket 0.
    # Plus trade at base+34s is also bucket 0. So 1 candle.
    assert report.rows_per_table["raw_candles_1m"] == 1

    # Spot-check raw_trades parquet
    trade_files = warehouse_files(warehouse, "raw_trades")
    assert len(trade_files) == 1
    tbl = pq.ParquetFile(trade_files[0]).read()
    assert tbl.num_rows == 1
    assert tbl.column("price").to_pylist() == [0.51]

    # Spot-check raw_candles_1m parquet has both quote and trade fields populated
    cand_files = warehouse_files(warehouse, "raw_candles_1m")
    assert len(cand_files) == 1
    cand = pq.ParquetFile(cand_files[0]).read().to_pylist()[0]
    assert cand["trade_count"] == 1
    assert cand["top_of_book_count"] == 3
    assert cand["bid_open"] == 0.50
    assert cand["bid_close"] == 0.51
    assert cand["trade_close"] == 0.51
    assert cand["vwap"] == 0.51


def test_etl_skips_in_flight_files(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    src = raw_dir / "in-flight.ndjson.gz"
    with gzip.open(src, "wb") as fp:
        fp.write(b"")  # empty
    # Don't backdate mtime: with a positive lag, the file should be skipped.
    warehouse = tmp_path / "warehouse"
    report = run_etl_batch(
        raw_dir=raw_dir, warehouse_dir=warehouse, lag_seconds=3600.0
    )
    assert report.files_processed == 0
    assert report.records_read == 0


def test_etl_quarantines_crossed_book_without_affecting_clean_rows(
    tmp_path: Path,
) -> None:
    """A crossed best_bid_ask must go to quarantine; the surrounding clean
    events must still land in their main tables and feed the candle agg."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    base = _ts(2026, 5, 10, 12, 0)
    src = raw_dir / "2026-05-10.ndjson.gz"
    _write_ndjson_gz(
        src,
        [
            # Clean best_bid_ask.
            {
                "receive_time": base + 100,
                "raw": {
                    "event_type": "best_bid_ask",
                    "asset_id": "tok-1",
                    "market": "0xmkt",
                    "best_bid": "0.50",
                    "best_ask": "0.52",
                    "spread": "0.02",
                    "timestamp": str(base),
                },
            },
            # CROSSED best_bid_ask (bid > ask) — should be quarantined.
            {
                "receive_time": base + 200,
                "raw": {
                    "event_type": "best_bid_ask",
                    "asset_id": "tok-1",
                    "market": "0xmkt",
                    "best_bid": "0.60",
                    "best_ask": "0.55",
                    "spread": "-0.05",
                    "timestamp": str(base + 100),
                },
            },
            # Clean trade.
            {
                "receive_time": base + 300,
                "raw": {
                    "event_type": "last_trade_price",
                    "asset_id": "tok-1",
                    "market": "0xmkt",
                    "price": "0.51",
                    "size": "10",
                    "side": "BUY",
                    "fee_rate_bps": "0",
                    "timestamp": str(base + 200),
                },
            },
            # Duplicate trade — same trade_id => quarantined as duplicate_trade_id.
            {
                "receive_time": base + 400,
                "raw": {
                    "event_type": "last_trade_price",
                    "asset_id": "tok-1",
                    "market": "0xmkt",
                    "price": "0.51",
                    "size": "10",
                    "side": "BUY",
                    "fee_rate_bps": "0",
                    "timestamp": str(base + 200),
                },
            },
            # Negative-size trade — quarantined as negative_size.
            {
                "receive_time": base + 500,
                "raw": {
                    "event_type": "last_trade_price",
                    "asset_id": "tok-1",
                    "market": "0xmkt",
                    "price": "0.51",
                    "size": "-5",
                    "side": "SELL",
                    "fee_rate_bps": "0",
                    "timestamp": str(base + 300),
                },
            },
        ],
    )

    import os

    os.utime(src, (1, 1))

    warehouse = tmp_path / "warehouse"
    report = run_etl_batch(
        raw_dir=raw_dir, warehouse_dir=warehouse, lag_seconds=0.0
    )

    # 1 clean best_bid_ask lands; crossed is quarantined.
    assert report.rows_per_table["raw_top_of_book"] == 1
    # 1 clean trade; 2 quarantined trades (duplicate + negative_size).
    assert report.rows_per_table["raw_trades"] == 1
    # Quarantine table sees 1 (crossed_book) + 1 (duplicate) + 1 (negative_size) = 3 rows.
    assert report.rows_per_table["quarantine"] == 3

    assert report.quarantined_by_rule.get("crossed_book") == 1
    assert report.quarantined_by_rule.get("duplicate_trade_id") == 1
    assert report.quarantined_by_rule.get("negative_size") == 1
    assert report.quarantined_total == 3

    # Spot-check the quarantine parquet itself.
    q_files = warehouse_files(warehouse, "quarantine")
    assert q_files, "quarantine partition should exist"
    q_tbl = pq.ParquetFile(q_files[0]).read().to_pylist()
    rules = {r["rule"] for r in q_tbl}
    assert {"crossed_book", "duplicate_trade_id", "negative_size"} <= rules
    for r in q_tbl:
        assert r["target_table"] in {"raw_top_of_book", "raw_trades"}
        assert r["payload_json"]  # never empty
        assert r["source_symbol"] == "tok-1"


def test_etl_preserves_provenance_in_parquet(tmp_path: Path) -> None:
    """Records carrying ``provenance="polymarket-rest-backfill"`` must
    land in the canonical Parquet with that tag intact (issue #5)."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    base = _ts(2026, 5, 10, 12, 0)
    src = raw_dir / "2026-05-10.ndjson.gz"
    _write_ndjson_gz(
        src,
        [
            # WS-shaped trade (no provenance -> "ws").
            {
                "receive_time": base + 100,
                "raw": {
                    "event_type": "last_trade_price",
                    "asset_id": "tok-1",
                    "market": "0xmkt",
                    "price": "0.51",
                    "size": "10",
                    "side": "BUY",
                    "fee_rate_bps": "0",
                    "timestamp": str(base),
                },
            },
            # Backfill-shaped trade — provenance carried inside ``raw``.
            {
                "receive_time": base + 200,
                "raw": {
                    "event_type": "last_trade_price",
                    "asset_id": "tok-1",
                    "market": "0xmkt",
                    "price": "0.52",
                    "size": "5",
                    "side": "SELL",
                    "fee_rate_bps": "0",
                    "timestamp": str(base + 100),
                    "provenance": "polymarket-rest-backfill",
                },
            },
        ],
    )
    import os

    os.utime(src, (1, 1))

    warehouse = tmp_path / "warehouse"
    report = run_etl_batch(
        raw_dir=raw_dir, warehouse_dir=warehouse, lag_seconds=0.0
    )
    assert report.rows_per_table["raw_trades"] == 2

    trade_files = warehouse_files(warehouse, "raw_trades")
    assert len(trade_files) == 1
    rows = pq.ParquetFile(trade_files[0]).read().to_pylist()
    provenances = sorted(r["provenance"] for r in rows)
    assert provenances == ["polymarket-rest-backfill", "ws"]


def test_etl_prefers_outer_seed_timestamps_and_provenance(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    base = _ts(2026, 5, 10, 12, 0)
    source_ts = base + 1_234
    capture_ts = source_ts + 321
    src = raw_dir / "2026-05-10.ndjson.gz"
    _write_ndjson_gz(
        src,
        [
            {
                # Deliberately wrong legacy field: ETL should prefer
                # capture_timestamp_ms.
                "receive_time": base + 999_000,
                "source_timestamp_ms": source_ts,
                "capture_timestamp_ms": capture_ts,
                "source_channel": "clob-rest",
                "provenance": "polymarket-rest-seed",
                "raw": {
                    "event_type": "book",
                    "asset_id": "tok-seed",
                    "market": "0xmkt",
                    # Deliberately wrong inner timestamp: ETL should prefer
                    # source_timestamp_ms.
                    "timestamp": str(base - 99_000),
                    "hash": "seed-hash",
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "50"}],
                    "provenance": "ws",
                },
            }
        ],
    )

    import os

    os.utime(src, (1, 1))

    warehouse = tmp_path / "warehouse"
    report = run_etl_batch(
        raw_dir=raw_dir, warehouse_dir=warehouse, lag_seconds=0.0
    )

    assert report.rows_per_table["raw_orderbook_snapshot"] == 2
    assert report.rows_per_table["raw_top_of_book"] == 1

    snapshot_files = warehouse_files(warehouse, "raw_orderbook_snapshot")
    snapshot_rows = pq.ParquetFile(snapshot_files[0]).read().to_pylist()
    assert len(snapshot_rows) == 2
    for row in snapshot_rows:
        assert row["ts"] == source_ts
        assert row["message_ts"] == source_ts
        assert row["ingest_ts"] == capture_ts
        assert row["capture_timestamp_ms"] == capture_ts
        assert row["source_channel"] == "clob-rest"
        assert row["provenance"] == "polymarket-rest-seed"

    tob_files = warehouse_files(warehouse, "raw_top_of_book")
    tob = pq.ParquetFile(tob_files[0]).read().to_pylist()[0]
    assert tob["ts"] == source_ts
    assert tob["message_ts"] == source_ts
    assert tob["ingest_ts"] == capture_ts
    assert tob["capture_timestamp_ms"] == capture_ts
    assert tob["source_channel"] == "clob-rest"
    assert tob["provenance"] == "polymarket-rest-seed"


def test_etl_enriches_canonical_symbol_from_mapping(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    base = _ts(2026, 5, 10, 12, 0)
    src = raw_dir / "2026-05-10.ndjson.gz"
    _write_ndjson_gz(
        src,
        [
            {
                "receive_time": base + 100,
                "raw": {
                    "event_type": "best_bid_ask",
                    "asset_id": "tok-1",
                    "market": "0xmkt",
                    "best_bid": "0.50",
                    "best_ask": "0.52",
                    "spread": "0.02",
                    "timestamp": str(base),
                },
            },
            {
                "receive_time": base + 200,
                "raw": {
                    "event_type": "last_trade_price",
                    "asset_id": "tok-1",
                    "market": "0xmkt",
                    "price": "0.51",
                    "size": "10",
                    "side": "BUY",
                    "fee_rate_bps": "0",
                    "timestamp": str(base + 100),
                },
            },
        ],
    )

    import os

    os.utime(src, (1, 1))

    mapper = SymbolMapper(
        [
            symbol_mapping_row(
                source="polymarket",
                source_symbol="tok-1",
                source_market="0xmkt",
                canonical_symbol="polymarket-election-2026-yes",
                effective_from_ts=base - 1,
                symbol_kind="binary_outcome",
                metadata={"outcome": "yes"},
            )
        ]
    )
    warehouse = tmp_path / "warehouse"
    report = run_etl_batch(
        raw_dir=raw_dir,
        warehouse_dir=warehouse,
        lag_seconds=0.0,
        symbol_mapper=mapper,
    )

    assert report.rows_per_table["symbol_mapping"] == 1
    assert report.rows_per_table["raw_top_of_book"] == 1
    assert report.rows_per_table["raw_trades"] == 1
    assert report.rows_per_table["raw_candles_1m"] == 1

    tob_files = warehouse_files(warehouse, "raw_top_of_book")
    tob = pq.ParquetFile(tob_files[0]).read().to_pylist()[0]
    assert tob["canonical_symbol"] == "polymarket-election-2026-yes"

    candle_files = warehouse_files(warehouse, "raw_candles_1m")
    candle = pq.ParquetFile(candle_files[0]).read().to_pylist()[0]
    assert candle["canonical_symbol"] == "polymarket-election-2026-yes"

    mapping_files = warehouse_files(warehouse, "symbol_mapping")
    mapping = pq.ParquetFile(mapping_files[0]).read().to_pylist()[0]
    assert mapping["canonical_symbol"] == "polymarket-election-2026-yes"
    assert mapping["metadata_json"] == '{"outcome":"yes"}'


def test_etl_persists_gamma_symbol_mapping_and_annotated_quote(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    base = _ts(2026, 5, 10, 12, 0)
    canonical_symbol = "BTC-15M:btc-updown-15m-1778423700:UP"
    src = raw_dir / "2026-05-10.ndjson.gz"
    _write_ndjson_gz(
        src,
        [
            {
                "receive_time": base,
                "source_timestamp_ms": base,
                "raw": {
                    "event_type": "symbol_mapping",
                    "mappings": [
                        {
                            "source": "polymarket",
                            "source_symbol": "tok-up",
                            "source_market": "0xmkt",
                            "canonical_symbol": canonical_symbol,
                            "effective_from_ts": base,
                            "symbol_kind": "btc_15m_outcome",
                            "metadata_json": '{"outcome_side":"UP"}',
                        }
                    ],
                },
            },
            {
                "receive_time": base + 100,
                "raw": {
                    "event_type": "best_bid_ask",
                    "asset_id": "tok-up",
                    "market": "0xmkt",
                    "canonical_symbol": canonical_symbol,
                    "outcome_side": "UP",
                    "best_bid": "0.50",
                    "best_ask": "0.52",
                    "spread": "0.02",
                    "timestamp": str(base + 100),
                },
            },
        ],
    )

    import os

    os.utime(src, (1, 1))

    warehouse = tmp_path / "warehouse"
    report = run_etl_batch(
        raw_dir=raw_dir,
        warehouse_dir=warehouse,
        lag_seconds=0.0,
    )

    assert report.rows_per_table["symbol_mapping"] == 1
    assert report.rows_per_table["raw_top_of_book"] == 1
    tob = pq.ParquetFile(warehouse_files(warehouse, "raw_top_of_book")[0]).read().to_pylist()[0]
    mapping = pq.ParquetFile(warehouse_files(warehouse, "symbol_mapping")[0]).read().to_pylist()[0]
    assert tob["canonical_symbol"] == canonical_symbol
    assert mapping["canonical_symbol"] == canonical_symbol
    assert mapping["metadata_json"] == '{"outcome_side":"UP"}'


def test_etl_quarantines_stale_timestamp_with_configurable_threshold(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    base = _ts(2026, 5, 10, 12, 0)
    src = raw_dir / "2026-05-10.ndjson.gz"
    _write_ndjson_gz(
        src,
        [
            {
                "receive_time": base + 2_000,
                "raw": {
                    "event_type": "best_bid_ask",
                    "asset_id": "tok-1",
                    "market": "0xmkt",
                    "best_bid": "0.50",
                    "best_ask": "0.52",
                    "spread": "0.02",
                    "timestamp": str(base),
                },
            }
        ],
    )

    import os

    os.utime(src, (1, 1))

    warehouse = tmp_path / "warehouse"
    report = run_etl_batch(
        raw_dir=raw_dir,
        warehouse_dir=warehouse,
        lag_seconds=0.0,
        timestamp_stale_threshold_seconds=1.0,
    )

    assert report.rows_per_table["raw_top_of_book"] == 0
    assert report.rows_per_table["quarantine"] == 1
    assert report.quarantined_by_rule == {"ts_too_stale": 1}


def test_etl_skips_cross_batch_duplicate_trade_id(tmp_path: Path) -> None:
    base = _ts(2026, 5, 10, 12, 0)
    warehouse = tmp_path / "warehouse"

    raw_first = tmp_path / "raw-first"
    raw_first.mkdir()
    first_src = raw_first / "first.ndjson.gz"
    _write_ndjson_gz(
        first_src,
        [_trade_record(asset_id="tok-1", market="0xmkt", ts=base)],
    )

    import os

    os.utime(first_src, (1, 1))
    first = run_etl_batch(
        raw_dir=raw_first, warehouse_dir=warehouse, lag_seconds=0.0
    )
    assert first.rows_per_table["raw_trades"] == 1
    assert first.cross_batch_duplicates_skipped == 0

    raw_second = tmp_path / "raw-second"
    raw_second.mkdir()
    second_src = raw_second / "second.ndjson.gz"
    _write_ndjson_gz(
        second_src,
        [
            _trade_record(
                asset_id="tok-1",
                market="0xmkt",
                ts=base,
                provenance="polymarket-rest-backfill",
            ),
            _trade_record(
                asset_id="tok-1",
                market="0xmkt",
                ts=base + 1_000,
                price="0.52",
                provenance="polymarket-rest-backfill",
            ),
        ],
    )
    os.utime(second_src, (1, 1))

    second = run_etl_batch(
        raw_dir=raw_second, warehouse_dir=warehouse, lag_seconds=0.0
    )
    assert second.cross_batch_duplicates_skipped == 1
    assert second.rows_per_table["raw_trades"] == 1

    rows = []
    for path in warehouse_files(warehouse, "raw_trades"):
        rows.extend(pq.ParquetFile(path).read().to_pylist())
    trade_ids = [row["trade_id"] for row in rows]
    assert len(trade_ids) == 2
    assert len(set(trade_ids)) == 2


def test_etl_skips_replayed_backfill_window(tmp_path: Path) -> None:
    base = _ts(2026, 5, 10, 12, 0)
    warehouse = tmp_path / "warehouse"
    import os

    for name in ("first", "second"):
        raw_dir = tmp_path / f"raw-{name}"
        raw_dir.mkdir()
        src = raw_dir / f"{name}.ndjson.gz"
        _write_ndjson_gz(
            src,
            [
                _trade_record(
                    asset_id="tok-1",
                    market="0xmkt",
                    ts=base,
                    provenance="polymarket-rest-backfill",
                )
            ],
        )
        os.utime(src, (1, 1))
        report = run_etl_batch(
            raw_dir=raw_dir, warehouse_dir=warehouse, lag_seconds=0.0
        )
        if name == "first":
            assert report.rows_per_table["raw_trades"] == 1
            assert report.cross_batch_duplicates_skipped == 0
        else:
            assert report.rows_per_table["raw_trades"] == 0
            assert report.cross_batch_duplicates_skipped == 1


def test_etl_large_unique_trade_set_has_no_cross_batch_skips(tmp_path: Path) -> None:
    base = _ts(2026, 5, 10, 12, 0)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    src = raw_dir / "trades.ndjson.gz"
    _write_ndjson_gz(
        src,
        [
            _trade_record(
                asset_id="tok-1",
                market="0xmkt",
                ts=base + i,
                price=f"0.{50 + (i % 10)}",
                size=str(i + 1),
            )
            for i in range(500)
        ],
    )
    import os

    os.utime(src, (1, 1))
    report = run_etl_batch(
        raw_dir=raw_dir,
        warehouse_dir=tmp_path / "warehouse",
        lag_seconds=0.0,
    )
    assert report.rows_per_table["raw_trades"] == 500
    assert report.cross_batch_duplicates_skipped == 0
