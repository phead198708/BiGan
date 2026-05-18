"""SQL quality checks for generated feature tables."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

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


def test_feature_quality_sql_report_checks_real_parquet_partition(
    tmp_path: Path,
) -> None:
    warehouse = tmp_path / "warehouse"
    base = _ts_at(2026, 5, 13, 12, 0)
    rows = [_feature_row(base + minute * 60_000) for minute in range(10)]
    # A duplicate (source, source_symbol, feature_ts) group should fail only
    # the duplicate-key check.
    rows.append({**rows[0], "ingest_ts": rows[0]["ingest_ts"] + 1})
    # A gappy but correctly filtered row covers the gap/filter path without
    # failing the consistency check.
    rows.append(
        _feature_row(
            base + 10 * 60_000,
            completeness_score=0.5,
            data_gap_flag=True,
            quality_filter_pass=False,
            quote_age_ms=180_000,
            depth_age_ms=180_000,
        )
    )
    # The SQL check should catch nullable identity values even if the Parquet
    # file itself is permissive enough to store them.
    rows.append({**_feature_row(base + 11 * 60_000), "symbol": None})
    _write_feature_partition(warehouse, rows)

    report = run_feature_quality_sql_checks(warehouse)

    checks = {check.name: check.failures for check in report.checks}
    assert not report.passed
    assert checks["row_count"] == 0
    assert checks["duplicate_symbol_feature_ts"] == 1
    assert checks["minute_alignment"] == 0
    assert checks["required_identity_not_null"] == 1
    assert checks["quality_score_bounds"] == 0
    assert checks["gap_flag_consistency"] == 0
    assert checks["training_filter_has_rows"] == 0


def _feature_row(
    feature_ts: int,
    *,
    completeness_score: float = 1.0,
    data_gap_flag: bool = False,
    quality_filter_pass: bool = True,
    quote_age_ms: int = 0,
    depth_age_ms: int = 0,
) -> dict:
    return {
        "ts": feature_ts,
        "message_ts": feature_ts,
        "feature_ts": feature_ts,
        "ingest_ts": feature_ts + 100,
        "source": "polymarket",
        "source_symbol": "tok-1",
        "source_market": "0xmkt",
        "canonical_symbol": "BTC-UP-15M",
        "symbol": "BTC-UP-15M",
        "feature_version": "15m-v1",
        "completeness_score": completeness_score,
        "data_gap_flag": data_gap_flag,
        "quality_filter_pass": quality_filter_pass,
        "quote_age_ms": quote_age_ms,
        "depth_age_ms": depth_age_ms,
        "trade_age_ms": 0,
        "spread": 0.02,
        "mid_price": 0.50,
        "microprice": 0.50,
        "obi_l1": 0.0,
        "obi_l5": 0.0,
        "obi_l10": 0.0,
        "signed_volume_1m": 1.0,
        "trade_imbalance_1m": 1.0,
        "trade_count_1m": 1,
        "trade_volume_1m": 1.0,
        "ret_1m": 0.0,
        "ret_5m": 0.0,
        "ret_15m": 0.0,
        "rv_1m": 0.0,
        "rv_5m": 0.0,
        "rv_15m": 0.0,
    }


def _write_feature_partition(warehouse: Path, rows: list[dict]) -> None:
    partition = (
        warehouse
        / "features_15m_v1"
        / "source=polymarket"
        / "dt=2026-05-13"
    )
    partition.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), partition / "part-test.parquet")
