"""Acceptance tests for issue #8 feature quality fields."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from bigan.canonical.schemas import SCHEMA_FEATURES_15M_V1
from bigan.features.aggregation import aggregate_features_15m_v1
from bigan.features.quality import (
    feature_row_passes_quality,
    filter_trainable_feature_rows,
)


def _ts_at(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    second: int = 0,
) -> int:
    return int(
        datetime(year, month, day, hour, minute, second, tzinfo=UTC).timestamp() * 1000
    )


def _identity(ts: int, *, source_symbol: str = "tok-1") -> dict:
    return {
        "ts": ts,
        "message_ts": ts,
        "ingest_ts": ts + 100,
        "source": "polymarket",
        "source_symbol": source_symbol,
        "source_market": "0xmkt",
        "canonical_symbol": "BTC-UP-15M",
        "provenance": "ws",
    }


def _tob(ts: int, bid: float = 0.49, ask: float = 0.51) -> dict:
    return {
        **_identity(ts),
        "bid_price": bid,
        "ask_price": ask,
        "spread": ask - bid,
    }


def _depth(ts: int, side: str, level: int, size: float) -> dict:
    price = 0.50 - level * 0.01 if side == "BID" else 0.52 + level * 0.01
    return {
        **_identity(ts),
        "side": side,
        "level": level,
        "price": price,
        "size": size,
        "snapshot_hash": "h0",
    }


def _trade(ts: int, price: float = 0.50, size: float = 3, side: str = "BUY") -> dict:
    return {
        **_identity(ts),
        "price": price,
        "size": size,
        "side": side,
        "fee_rate_bps": 0.0,
        "trade_id": f"trade-{ts}-{side}-{size}",
    }


def test_features_schema_declares_quality_columns() -> None:
    names = set(SCHEMA_FEATURES_15M_V1.names)

    assert {
        "completeness_score",
        "data_gap_flag",
        "quality_filter_pass",
        "quote_age_ms",
        "depth_age_ms",
        "trade_age_ms",
    }.issubset(names)


def test_complete_feature_row_gets_full_score_and_passes_quality_filter() -> None:
    feature_ts = _ts_at(2026, 5, 13, 12, 1)

    rows = aggregate_features_15m_v1(
        top_of_book_rows=[_tob(feature_ts)],
        orderbook_rows=[
            _depth(feature_ts, "BID", 0, 10),
            _depth(feature_ts, "ASK", 0, 5),
        ],
        trade_rows=[_trade(feature_ts - 10_000)],
        ingest_ts=feature_ts + 500,
    )
    row = next(row for row in rows if row["feature_ts"] == feature_ts)

    assert row["quote_age_ms"] == 0
    assert row["depth_age_ms"] == 0
    assert row["trade_age_ms"] == 10_000
    assert row["completeness_score"] == pytest.approx(1.0)
    assert row["data_gap_flag"] is False
    assert row["quality_filter_pass"] is True


def test_stale_asof_inputs_lower_score_and_set_gap_flag() -> None:
    t0 = _ts_at(2026, 5, 13, 12, 0)
    feature_ts = t0 + 180_000

    rows = aggregate_features_15m_v1(
        top_of_book_rows=[_tob(t0)],
        orderbook_rows=[_depth(t0, "BID", 0, 10), _depth(t0, "ASK", 0, 5)],
        trade_rows=[_trade(feature_ts)],
        ingest_ts=feature_ts + 500,
    )
    row = next(row for row in rows if row["feature_ts"] == feature_ts)

    assert row["quote_age_ms"] == 180_000
    assert row["depth_age_ms"] == 180_000
    assert row["trade_age_ms"] == 0
    assert 0 <= row["completeness_score"] < 1.0
    assert row["data_gap_flag"] is True
    assert row["quality_filter_pass"] is False


def test_quality_filter_keeps_only_trainable_rows() -> None:
    good = {
        "symbol": "BTC-UP-15M",
        "completeness_score": 0.95,
        "data_gap_flag": False,
        "quality_filter_pass": True,
    }
    low_score = {
        "symbol": "BTC-DOWN-15M",
        "completeness_score": 0.5,
        "data_gap_flag": False,
        "quality_filter_pass": False,
    }
    gappy = {
        "symbol": "BTC-FLAT-15M",
        "completeness_score": 0.95,
        "data_gap_flag": True,
        "quality_filter_pass": False,
    }

    assert feature_row_passes_quality(good)
    assert not feature_row_passes_quality(low_score)
    assert not feature_row_passes_quality(gappy)
    assert filter_trainable_feature_rows([good, low_score, gappy]) == [good]
