"""Unit tests for :mod:`bigan.canonical.transform`."""

from __future__ import annotations

from bigan.canonical.transform import (
    SOURCE_POLYMARKET,
    derive_top_of_book_from_book,
    derive_top_of_book_from_price_change,
    transform_book_event,
    transform_event,
    transform_last_trade_price_event,
    transform_top_of_book_event,
)

# ---------------------------------------------------------------------------
# best_bid_ask
# ---------------------------------------------------------------------------


def test_top_of_book_basic() -> None:
    raw = {
        "event_type": "best_bid_ask",
        "asset_id": "1",
        "market": "0xmkt",
        "canonical_symbol": "BTC-15M:btc-updown-15m-1:UP",
        "best_bid": "0.73",
        "best_ask": "0.77",
        "spread": "0.04",
        "timestamp": "1766789469958",
    }
    row = transform_top_of_book_event(raw, ingest_ts=999)
    assert row is not None
    assert row["ts"] == 1766789469958
    assert row["message_ts"] == 1766789469958
    assert row["ingest_ts"] == 999
    assert row["capture_timestamp_ms"] == 999
    assert row["source"] == SOURCE_POLYMARKET
    assert row["source_symbol"] == "1"
    assert row["source_market"] == "0xmkt"
    assert row["canonical_symbol"] == "BTC-15M:btc-updown-15m-1:UP"
    assert row["bid_price"] == 0.73
    assert row["ask_price"] == 0.77
    assert row["spread"] == 0.04


def test_top_of_book_recomputes_missing_spread() -> None:
    raw = {
        "event_type": "best_bid_ask",
        "asset_id": "1",
        "market": "0x",
        "best_bid": "0.40",
        "best_ask": "0.45",
        "timestamp": "10",
    }
    row = transform_top_of_book_event(raw, ingest_ts=1)
    assert row is not None
    assert row["spread"] is not None
    assert abs(row["spread"] - 0.05) < 1e-9


def test_top_of_book_returns_none_for_wrong_type() -> None:
    assert transform_top_of_book_event(
        {"event_type": "book", "asset_id": "1", "timestamp": "1"}, ingest_ts=1
    ) is None


def test_top_of_book_returns_none_without_asset_id() -> None:
    assert transform_top_of_book_event(
        {"event_type": "best_bid_ask", "timestamp": "1"}, ingest_ts=1
    ) is None


# ---------------------------------------------------------------------------
# book
# ---------------------------------------------------------------------------


def test_book_event_long_format() -> None:
    raw = {
        "event_type": "book",
        "asset_id": "abc",
        "market": "0xmkt",
        "timestamp": "100",
        "hash": "snap-h",
        # Intentionally unsorted to verify we sort by canonical ordering.
        "bids": [
            {"price": "0.48", "size": "30"},
            {"price": "0.50", "size": "15"},
            {"price": "0.49", "size": "20"},
        ],
        "asks": [
            {"price": "0.54", "size": "10"},
            {"price": "0.52", "size": "25"},
            {"price": "0.53", "size": "60"},
        ],
    }
    rows = transform_book_event(raw, ingest_ts=42)
    assert len(rows) == 6
    bids = [r for r in rows if r["side"] == "BID"]
    asks = [r for r in rows if r["side"] == "ASK"]

    # Bids: descending price, level 0 = best bid (highest)
    assert [(r["level"], r["price"]) for r in bids] == [(0, 0.50), (1, 0.49), (2, 0.48)]
    # Asks: ascending price, level 0 = best ask (lowest)
    assert [(r["level"], r["price"]) for r in asks] == [(0, 0.52), (1, 0.53), (2, 0.54)]

    assert all(r["snapshot_hash"] == "snap-h" for r in rows)
    assert all(r["ts"] == 100 for r in rows)
    assert all(r["source"] == "polymarket" for r in rows)


def test_book_event_empty_levels_yields_no_rows() -> None:
    raw = {
        "event_type": "book",
        "asset_id": "abc",
        "market": "0xmkt",
        "timestamp": "100",
        "bids": [],
        "asks": [],
    }
    assert transform_book_event(raw, ingest_ts=1) == []


def test_book_event_with_garbage_levels_filters_them_out() -> None:
    raw = {
        "event_type": "book",
        "asset_id": "abc",
        "market": "0xmkt",
        "timestamp": "100",
        "bids": [{"price": "0.5", "size": "10"}, {"price": None, "size": "x"}, "junk"],
        "asks": [],
    }
    rows = transform_book_event(raw, ingest_ts=1)
    assert len(rows) == 1
    assert rows[0]["price"] == 0.5


# ---------------------------------------------------------------------------
# last_trade_price
# ---------------------------------------------------------------------------


def test_trade_event_basic() -> None:
    raw = {
        "event_type": "last_trade_price",
        "asset_id": "x",
        "market": "0xmkt",
        "price": "0.456",
        "side": "BUY",
        "size": "219.217767",
        "fee_rate_bps": "0",
        "timestamp": "1750428146322",
    }
    row = transform_last_trade_price_event(raw, ingest_ts=2)
    assert row is not None
    assert row["price"] == 0.456
    assert row["size"] == 219.217767
    assert row["side"] == "BUY"
    assert row["fee_rate_bps"] == 0.0
    assert row["trade_id"].startswith("polymarket-x-1750428146322-")


def test_trade_event_rejects_invalid_side() -> None:
    raw = {
        "event_type": "last_trade_price",
        "asset_id": "x",
        "market": "m",
        "price": "0.5",
        "size": "1",
        "side": "MAYBE",
        "timestamp": "1",
    }
    assert transform_last_trade_price_event(raw, ingest_ts=1) is None


# ---------------------------------------------------------------------------
# Generic dispatcher
# ---------------------------------------------------------------------------


def test_dispatch_dispatches_correctly() -> None:
    rec_book = {
        "receive_time": 1,
        "raw": {
            "event_type": "book",
            "asset_id": "a",
            "market": "m",
            "timestamp": "1",
            "bids": [{"price": "0.5", "size": "1"}],
            "asks": [{"price": "0.6", "size": "1"}],
        },
    }
    out = transform_event(rec_book)
    assert "raw_orderbook_snapshot" in out
    assert len(out["raw_orderbook_snapshot"]) == 2
    # book events also yield a derived top_of_book row from best levels.
    assert "raw_top_of_book" in out
    assert len(out["raw_top_of_book"]) == 1
    assert out["raw_top_of_book"][0]["bid_price"] == 0.5
    assert out["raw_top_of_book"][0]["ask_price"] == 0.6

    rec_bba = {
        "receive_time": 2,
        "raw": {
            "event_type": "best_bid_ask",
            "asset_id": "a",
            "market": "m",
            "best_bid": "0.5",
            "best_ask": "0.6",
            "timestamp": "1",
        },
    }
    assert "raw_top_of_book" in transform_event(rec_bba)

    rec_unknown = {
        "receive_time": 3,
        "raw": {"event_type": "tick_size_change", "asset_id": "a", "timestamp": "1"},
    }
    assert transform_event(rec_unknown) == {}

    # No raw payload at all
    assert transform_event({"receive_time": 1}) == {}
    # No receive_time
    assert transform_event({"raw": {"event_type": "book", "asset_id": "a"}}) == {}


def test_dispatch_symbol_mapping_event_persists_mapping_rows() -> None:
    rec = {
        "receive_time": 1_700_000_000_000,
        "source_timestamp_ms": 1_700_000_000_000,
        "raw": {
            "event_type": "symbol_mapping",
            "mappings": [
                {
                    "source": "polymarket",
                    "source_symbol": "tok-up",
                    "source_market": "0xmkt",
                    "canonical_symbol": "BTC-15M:btc-updown-15m-1:UP",
                    "effective_from_ts": 1_700_000_000_000,
                    "symbol_kind": "btc_15m_outcome",
                    "metadata_json": '{"outcome_side":"UP"}',
                }
            ],
        },
    }

    out = transform_event(rec)

    assert set(out) == {"symbol_mapping"}
    row = out["symbol_mapping"][0]
    assert row["source_symbol"] == "tok-up"
    assert row["canonical_symbol"] == "BTC-15M:btc-updown-15m-1:UP"
    assert row["metadata_json"] == '{"outcome_side":"UP"}'


def test_dispatch_persists_outer_capture_and_channel_metadata() -> None:
    record = {
        "receive_time": 999_999,
        "source_timestamp_ms": 123_456,
        "capture_timestamp_ms": 123_999,
        "source_channel": "clob-rest",
        "provenance": "polymarket-rest-seed",
        "raw": {
            "event_type": "book",
            "asset_id": "a",
            "market": "m",
            "timestamp": "1",
            "provenance": "ws",
            "bids": [{"price": "0.5", "size": "1"}],
            "asks": [{"price": "0.6", "size": "1"}],
        },
    }

    out = transform_event(record)

    rows = out["raw_orderbook_snapshot"] + out["raw_top_of_book"]
    assert rows
    for row in rows:
        assert row["ts"] == 123_456
        assert row["message_ts"] == 123_456
        assert row["ingest_ts"] == 123_999
        assert row["capture_timestamp_ms"] == 123_999
        assert row["source_channel"] == "clob-rest"
        assert row["provenance"] == "polymarket-rest-seed"


# ---------------------------------------------------------------------------
# Derivations: book / price_change -> raw_top_of_book
# ---------------------------------------------------------------------------


def test_derive_top_of_book_from_book_picks_best_levels() -> None:
    raw = {
        "event_type": "book",
        "asset_id": "x",
        "market": "0xmkt",
        "timestamp": "100",
        "bids": [
            {"price": "0.48", "size": "30"},
            {"price": "0.50", "size": "15"},
            {"price": "0.49", "size": "20"},
        ],
        "asks": [
            {"price": "0.54", "size": "10"},
            {"price": "0.52", "size": "25"},
        ],
    }
    row = derive_top_of_book_from_book(raw, ingest_ts=1)
    assert row is not None
    assert row["bid_price"] == 0.50
    assert row["ask_price"] == 0.52
    assert abs(row["spread"] - 0.02) < 1e-9


def test_derive_top_of_book_from_book_handles_empty_book() -> None:
    raw = {
        "event_type": "book",
        "asset_id": "x",
        "market": "0xmkt",
        "timestamp": "100",
        "bids": [],
        "asks": [],
    }
    assert derive_top_of_book_from_book(raw, ingest_ts=1) is None


def test_derive_top_of_book_from_price_change_emits_one_row_per_asset() -> None:
    raw = {
        "event_type": "price_change",
        "market": "0xmkt",
        "timestamp": "1757908892351",
        "price_changes": [
            {
                "asset_id": "A",
                "canonical_symbol": "BTC-15M:btc-updown-15m-1:UP",
                "price": "0.5",
                "size": "200",
                "side": "BUY",
                "hash": "h1",
                "best_bid": "0.49",
                "best_ask": "0.51",
            },
            {
                "asset_id": "B",
                "price": "0.5",
                "size": "200",
                "side": "SELL",
                "hash": "h2",
                "best_bid": "0.55",
                "best_ask": "0.57",
            },
        ],
    }
    rows = derive_top_of_book_from_price_change(raw, ingest_ts=1)
    by_asset = {r["source_symbol"]: r for r in rows}
    assert set(by_asset) == {"A", "B"}
    assert by_asset["A"]["bid_price"] == 0.49
    assert by_asset["A"]["canonical_symbol"] == "BTC-15M:btc-updown-15m-1:UP"
    assert by_asset["B"]["ask_price"] == 0.57


def test_derive_top_of_book_from_price_change_skips_entries_missing_quotes() -> None:
    raw = {
        "event_type": "price_change",
        "market": "0xmkt",
        "timestamp": "1",
        "price_changes": [
            {"asset_id": "A", "price": "0.5", "size": "1", "side": "BUY", "hash": "h"}
        ],
    }
    assert derive_top_of_book_from_price_change(raw, ingest_ts=1) == []


def test_dispatch_price_change_yields_top_of_book_rows() -> None:
    rec = {
        "receive_time": 5,
        "raw": {
            "event_type": "price_change",
            "market": "0xmkt",
            "timestamp": "10",
            "price_changes": [
                {
                    "asset_id": "A",
                    "price": "0.5",
                    "size": "1",
                    "side": "BUY",
                    "hash": "h",
                    "best_bid": "0.49",
                    "best_ask": "0.51",
                }
            ],
        },
    }
    out = transform_event(rec)
    assert "raw_top_of_book" in out
    assert len(out["raw_top_of_book"]) == 1
    assert out["raw_top_of_book"][0]["bid_price"] == 0.49


# ---------------------------------------------------------------------------
# provenance propagation (issue #5)
# ---------------------------------------------------------------------------


def test_top_of_book_defaults_to_ws_provenance() -> None:
    raw = {
        "event_type": "best_bid_ask",
        "asset_id": "1",
        "market": "0xmkt",
        "best_bid": "0.50",
        "best_ask": "0.52",
        "spread": "0.02",
        "timestamp": "1700000000000",
    }
    row = transform_top_of_book_event(raw, ingest_ts=42)
    assert row is not None
    assert row["provenance"] == "ws"


def test_top_of_book_propagates_backfill_provenance() -> None:
    raw = {
        "event_type": "best_bid_ask",
        "asset_id": "1",
        "market": "0xmkt",
        "best_bid": "0.50",
        "best_ask": "0.52",
        "spread": "0.02",
        "timestamp": "1700000000000",
        "provenance": "polymarket-rest-backfill",
    }
    row = transform_top_of_book_event(raw, ingest_ts=42)
    assert row is not None
    assert row["provenance"] == "polymarket-rest-backfill"


def test_book_event_propagates_provenance_to_all_rows() -> None:
    raw = {
        "event_type": "book",
        "asset_id": "abc",
        "market": "0xmkt",
        "timestamp": "1",
        "hash": "h0",
        "bids": [{"price": "0.49", "size": "100"}],
        "asks": [{"price": "0.51", "size": "50"}],
        "provenance": "polymarket-rest-backfill",
    }
    rows = transform_book_event(raw, ingest_ts=99)
    assert rows
    assert all(r["provenance"] == "polymarket-rest-backfill" for r in rows)


def test_trade_event_propagates_backfill_provenance() -> None:
    raw = {
        "event_type": "last_trade_price",
        "asset_id": "x",
        "market": "0xmkt",
        "price": "0.51",
        "size": "10",
        "side": "BUY",
        "fee_rate_bps": "0",
        "timestamp": "1700000000000",
        "provenance": "polymarket-rest-backfill",
    }
    row = transform_last_trade_price_event(raw, ingest_ts=99)
    assert row is not None
    assert row["provenance"] == "polymarket-rest-backfill"


def test_price_change_derivation_propagates_provenance() -> None:
    raw = {
        "event_type": "price_change",
        "market": "0xmkt",
        "timestamp": "1700000000000",
        "provenance": "polymarket-rest-backfill",
        "price_changes": [
            {
                "asset_id": "A",
                "best_bid": "0.50",
                "best_ask": "0.52",
                "price": "0.50",
                "size": "1",
                "side": "BUY",
                "hash": "h",
            }
        ],
    }
    rows = derive_top_of_book_from_price_change(raw, ingest_ts=1)
    assert rows
    assert rows[0]["provenance"] == "polymarket-rest-backfill"


# Source identifier sanity check (kept as the last assertion).
def test_source_polymarket_constant() -> None:
    assert SOURCE_POLYMARKET == "polymarket"
