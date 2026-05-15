"""SQL quality checks for generated feature tables."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from bigan.canonical.writer import WarehouseWriter
from bigan.features.aggregation import run_feature_batch
from bigan.features.quality_sql import run_feature_quality_sql_checks


def _ts_at(year: int, month: int, day: int, hour: int, minute: int) -> int:
    return int(datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp() * 1000)


def _identity(ts: int) -> dict:
    return {
        "ts": ts,
        "message_ts": ts,
        "ingest_ts": ts + 100,
        "source": "polymarket",
        "source_symbol": "tok-1",
        "source_market": "0xmkt",
        "canonical_symbol": "BTC-UP-15M",
        "provenance": "ws",
    }


def _tob(ts: int) -> dict:
    return {
        **_identity(ts),
        "bid_price": 0.49,
        "ask_price": 0.51,
        "spread": 0.02,
    }


def _depth(ts: int, side: str, size: float) -> dict:
    return {
        **_identity(ts),
        "side": side,
        "level": 0,
        "price": 0.49 if side == "BID" else 0.51,
        "size": size,
        "snapshot_hash": "h0",
    }


def _trade(ts: int) -> dict:
    return {
        **_identity(ts),
        "price": 0.50,
        "size": 1.0,
        "side": "BUY",
        "fee_rate_bps": 0.0,
        "trade_id": f"trade-{ts}",
    }


def test_feature_quality_sql_report_passes_on_clean_features(tmp_path: Path) -> None:
    warehouse = tmp_path / "warehouse"
    ts = _ts_at(2026, 5, 13, 12, 0)
    with WarehouseWriter(warehouse, max_rows_per_partition=10) as writer:
        writer.append_rows("raw_top_of_book", [_tob(ts)])
        writer.append_rows(
            "raw_orderbook_snapshot",
            [_depth(ts, "BID", 10), _depth(ts, "ASK", 5)],
        )
        writer.append_rows("raw_trades", [_trade(ts)])

    run_feature_batch(warehouse, ingest_ts=ts + 500)

    report = run_feature_quality_sql_checks(warehouse)

    assert report.passed
    assert {check.name: check.failures for check in report.checks} == {
        "row_count": 0,
        "duplicate_symbol_feature_ts": 0,
        "minute_alignment": 0,
        "required_identity_not_null": 0,
        "quality_score_bounds": 0,
        "gap_flag_consistency": 0,
        "training_filter_has_rows": 0,
    }


def test_feature_quality_sql_report_fails_when_table_is_missing(tmp_path: Path) -> None:
    report = run_feature_quality_sql_checks(tmp_path / "warehouse")

    assert not report.passed
    assert any(check.failures for check in report.checks)
