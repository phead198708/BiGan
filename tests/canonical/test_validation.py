"""Unit tests for the validation / quarantine routing layer (issue #4)."""

from __future__ import annotations

import orjson
import pytest

from bigan.canonical.validation import (
    UNKNOWN_SYMBOL,
    RowValidator,
    ValidationRule,
)


def _base_row(**overrides):
    """Minimal raw_top_of_book-shaped row with sane identity fields."""
    row = {
        "ts": 1_700_000_000_000,
        "message_ts": 1_700_000_000_000,
        "ingest_ts": 1_700_000_000_500,
        "source": "polymarket",
        "source_symbol": "tok-1",
        "source_market": "0xmkt",
        "canonical_symbol": None,
        "bid_price": 0.49,
        "ask_price": 0.51,
        "spread": 0.02,
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# Identity rules
# ---------------------------------------------------------------------------


def test_clean_row_returns_no_errors() -> None:
    v = RowValidator()
    assert v.validate("raw_top_of_book", _base_row()) == []


def test_empty_symbol_fires() -> None:
    v = RowValidator()
    errs = v.validate("raw_top_of_book", _base_row(source_symbol=""))
    rules = {e.rule for e in errs}
    assert ValidationRule.EMPTY_SYMBOL in rules


def test_null_symbol_fires() -> None:
    v = RowValidator()
    errs = v.validate("raw_top_of_book", _base_row(source_symbol=None))
    rules = {e.rule for e in errs}
    assert ValidationRule.EMPTY_SYMBOL in rules


def test_whitespace_only_symbol_fires() -> None:
    v = RowValidator()
    errs = v.validate("raw_top_of_book", _base_row(source_symbol="   "))
    rules = {e.rule for e in errs}
    assert ValidationRule.EMPTY_SYMBOL in rules


def test_zero_ts_fires_empty_time() -> None:
    v = RowValidator()
    errs = v.validate("raw_top_of_book", _base_row(ts=0))
    rules = {e.rule for e in errs}
    assert ValidationRule.EMPTY_TIME in rules


def test_negative_ts_fires_empty_time() -> None:
    v = RowValidator()
    errs = v.validate("raw_top_of_book", _base_row(ts=-1))
    rules = {e.rule for e in errs}
    assert ValidationRule.EMPTY_TIME in rules


def test_null_ts_fires_empty_time() -> None:
    v = RowValidator()
    errs = v.validate("raw_top_of_book", _base_row(ts=None))
    rules = {e.rule for e in errs}
    assert ValidationRule.EMPTY_TIME in rules


def test_future_ts_fires() -> None:
    v = RowValidator(future_grace_ms=100)
    errs = v.validate(
        "raw_top_of_book",
        _base_row(ts=1_700_000_000_200, ingest_ts=1_700_000_000_000),
    )
    rules = {e.rule for e in errs}
    assert ValidationRule.TS_IN_FUTURE in rules


def test_stale_ts_fires() -> None:
    v = RowValidator(stale_threshold_ms=100)
    errs = v.validate(
        "raw_top_of_book",
        _base_row(ts=1_700_000_000_000, ingest_ts=1_700_000_000_200),
    )
    rules = {e.rule for e in errs}
    assert ValidationRule.TS_TOO_STALE in rules


def test_timestamp_thresholds_are_tunable() -> None:
    v = RowValidator(future_grace_ms=1_000, stale_threshold_ms=1_000)
    assert (
        v.validate(
            "raw_top_of_book",
            _base_row(ts=1_700_000_000_000, ingest_ts=1_700_000_000_900),
        )
        == []
    )


def test_invalid_timestamp_thresholds_raise() -> None:
    with pytest.raises(ValueError, match="future_grace_ms"):
        RowValidator(future_grace_ms=-1)
    with pytest.raises(ValueError, match="stale_threshold_ms"):
        RowValidator(stale_threshold_ms=-1)


# ---------------------------------------------------------------------------
# Top-of-book rules
# ---------------------------------------------------------------------------


def test_crossed_book_fires() -> None:
    v = RowValidator()
    # bid > ask is a crossed book.
    errs = v.validate("raw_top_of_book", _base_row(bid_price=0.55, ask_price=0.50))
    rules = {e.rule for e in errs}
    assert ValidationRule.CROSSED_BOOK in rules


def test_locked_book_is_not_crossed() -> None:
    """bid == ask is technically locked, not crossed; we tolerate it."""
    v = RowValidator()
    errs = v.validate("raw_top_of_book", _base_row(bid_price=0.50, ask_price=0.50))
    rules = {e.rule for e in errs}
    assert ValidationRule.CROSSED_BOOK not in rules


def test_negative_bid_price_fires() -> None:
    v = RowValidator()
    errs = v.validate("raw_top_of_book", _base_row(bid_price=-0.01, ask_price=0.50))
    rules = {e.rule for e in errs}
    assert ValidationRule.NEGATIVE_PRICE in rules


def test_one_sided_book_is_clean() -> None:
    v = RowValidator()
    # Missing ask shouldn't fire crossed_book.
    errs = v.validate("raw_top_of_book", _base_row(ask_price=None))
    rules = {e.rule for e in errs}
    assert ValidationRule.CROSSED_BOOK not in rules


# ---------------------------------------------------------------------------
# Orderbook snapshot rules
# ---------------------------------------------------------------------------


def _snapshot_row(**overrides):
    row = {
        "ts": 1_700_000_000_000,
        "message_ts": 1_700_000_000_000,
        "ingest_ts": 1_700_000_000_500,
        "source": "polymarket",
        "source_symbol": "tok-1",
        "source_market": "0xmkt",
        "canonical_symbol": None,
        "side": "BID",
        "level": 0,
        "price": 0.50,
        "size": 100.0,
        "snapshot_hash": "h0",
    }
    row.update(overrides)
    return row


def test_negative_size_fires_on_snapshot() -> None:
    v = RowValidator()
    errs = v.validate("raw_orderbook_snapshot", _snapshot_row(size=-1.0))
    rules = {e.rule for e in errs}
    assert ValidationRule.NEGATIVE_SIZE in rules


def test_negative_price_fires_on_snapshot() -> None:
    v = RowValidator()
    errs = v.validate("raw_orderbook_snapshot", _snapshot_row(price=-0.01))
    rules = {e.rule for e in errs}
    assert ValidationRule.NEGATIVE_PRICE in rules


def test_zero_size_is_clean_on_snapshot() -> None:
    """Zero size = level was canceled mid-snapshot, which is legal."""
    v = RowValidator()
    errs = v.validate("raw_orderbook_snapshot", _snapshot_row(size=0.0))
    rules = {e.rule for e in errs}
    assert ValidationRule.NEGATIVE_SIZE not in rules


# ---------------------------------------------------------------------------
# Trade rules
# ---------------------------------------------------------------------------


def _trade_row(**overrides):
    row = {
        "ts": 1_700_000_000_000,
        "message_ts": 1_700_000_000_000,
        "ingest_ts": 1_700_000_000_500,
        "source": "polymarket",
        "source_symbol": "tok-1",
        "source_market": "0xmkt",
        "canonical_symbol": None,
        "price": 0.51,
        "size": 10.0,
        "side": "BUY",
        "fee_rate_bps": 0.0,
        "trade_id": "polymarket-tok-1-1700000000000-0.51-10.0-BUY",
    }
    row.update(overrides)
    return row


def test_negative_size_fires_on_trade() -> None:
    v = RowValidator()
    errs = v.validate("raw_trades", _trade_row(size=-1.0))
    rules = {e.rule for e in errs}
    assert ValidationRule.NEGATIVE_SIZE in rules


def test_duplicate_trade_id_fires_on_second_occurrence() -> None:
    v = RowValidator()
    first = v.validate("raw_trades", _trade_row())
    assert first == []

    second = v.validate("raw_trades", _trade_row())
    rules = {e.rule for e in second}
    assert ValidationRule.DUPLICATE_TRADE_ID in rules


def test_distinct_trade_ids_are_clean() -> None:
    v = RowValidator()
    assert v.validate("raw_trades", _trade_row(trade_id="tid-a")) == []
    assert v.validate("raw_trades", _trade_row(trade_id="tid-b")) == []


# ---------------------------------------------------------------------------
# Quarantine row construction
# ---------------------------------------------------------------------------


def test_to_quarantine_rows_emits_one_row_per_rule() -> None:
    v = RowValidator()
    bad = _base_row(source_symbol="", bid_price=0.6, ask_price=0.5)
    errs = v.validate("raw_top_of_book", bad)
    # Should at least fire empty_symbol + crossed_book.
    assert len(errs) >= 2

    q_rows = v.to_quarantine_rows("raw_top_of_book", bad, errs)
    assert len(q_rows) == len(errs)

    for q in q_rows:
        assert q["target_table"] == "raw_top_of_book"
        assert q["source"] == "polymarket"
        # source_symbol was empty in the offending row -> substituted.
        assert q["source_symbol"] == UNKNOWN_SYMBOL
        assert isinstance(q["rule"], str)
        assert isinstance(q["detail"], str) and q["detail"]
        # payload_json must round-trip through json.
        payload = orjson.loads(q["payload_json"])
        assert payload["bid_price"] == 0.6
        assert payload["ask_price"] == 0.5


def test_quarantine_ts_falls_back_to_ingest_when_missing() -> None:
    v = RowValidator()
    bad = _base_row(ts=None)
    errs = v.validate("raw_top_of_book", bad)
    q_rows = v.to_quarantine_rows("raw_top_of_book", bad, errs)
    assert q_rows
    for q in q_rows:
        # ts must be positive so the partitioner can compute a date.
        assert q["ts"] > 0
        assert q["ts"] == bad["ingest_ts"]


def test_validator_stats_tracks_counts() -> None:
    v = RowValidator()
    v.validate("raw_top_of_book", _base_row())  # clean
    v.validate("raw_top_of_book", _base_row(source_symbol=""))  # 1 quarantine
    v.validate("raw_trades", _trade_row())  # clean
    v.validate("raw_trades", _trade_row())  # duplicate

    assert v.stats.rows_checked["raw_top_of_book"] == 2
    assert v.stats.rows_checked["raw_trades"] == 2
    assert v.stats.rows_quarantined_by_rule[ValidationRule.EMPTY_SYMBOL.value] == 1
    assert (
        v.stats.rows_quarantined_by_rule[ValidationRule.DUPLICATE_TRADE_ID.value] == 1
    )
    assert v.stats.total_quarantined == 2
