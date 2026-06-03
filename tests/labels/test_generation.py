"""TDD contract for issue #9 labels_15m_v1 generation."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from urllib import error

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bigan.canonical.writer import WarehouseWriter, warehouse_files

HORIZON_MS = 15 * 60_000


def _ts_at(year: int, month: int, day: int, hour: int, minute: int) -> int:
    return int(datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp() * 1000)


def _feature_row(
    ts: int,
    *,
    source_symbol: str = "tok-up",
    source_market: str = "0xmkt",
    canonical_symbol: str = "BTC-15M:btc-updown-15m-test:UP",
    market_implied_prob: float = 0.40,
) -> dict:
    return {
        "ts": ts,
        "message_ts": ts,
        "feature_ts": ts,
        "ingest_ts": ts + 100,
        "source": "polymarket",
        "source_symbol": source_symbol,
        "source_market": source_market,
        "canonical_symbol": canonical_symbol,
        "symbol": canonical_symbol,
        "feature_version": "bigan-mvp-v1.0.0",
        "completeness_score": 1.0,
        "data_gap_flag": False,
        "quality_filter_pass": True,
        "quote_age_ms": 0,
        "depth_age_ms": 0,
        "trade_age_ms": 0,
        "market_implied_prob": market_implied_prob,
    }


def _round_row(
    start_ts: int,
    start_price: float,
    target_price: float,
    *,
    source_market: str = "0xmkt",
    ingest_offset_ms: int = 1_000,
) -> dict:
    return {
        "ts": start_ts,
        "message_ts": start_ts,
        "ingest_ts": start_ts + HORIZON_MS + ingest_offset_ms,
        "source": "polymarket",
        "source_market": source_market,
        "round_slug": f"btc-updown-15m-{start_ts // 1000}",
        "round_start_ts": start_ts,
        "round_end_ts": start_ts + HORIZON_MS,
        "start_price": start_price,
        "target_price": target_price,
        "label_source": "polymarket_gamma_event_metadata",
    }


def test_labels_table_schema_is_registered_with_required_columns() -> None:
    from bigan.canonical.schemas import SCHEMAS, TABLE_NAMES

    assert "labels_15m_v1" in TABLE_NAMES
    schema = SCHEMAS["labels_15m_v1"]
    cols = {field.name for field in schema}

    assert {
        "ts",
        "message_ts",
        "feature_ts",
        "target_ts",
        "ingest_ts",
        "source",
        "source_symbol",
        "source_market",
        "canonical_symbol",
        "symbol",
        "label_version",
        "label_kind",
        "round_slug",
        "round_start_ts",
        "round_end_ts",
        "start_price",
        "target_price",
        "direction_up_15m",
        "entry_ask_price",
        "settlement_price",
        "entry_fee",
        "entry_cost",
        "realized_return",
        "fee_bps",
        "settlement_margin",
        "settlement_abs_margin",
        "settlement_neutral_margin",
        "label_settlement_3way",
        "max_exit_gain_up",
        "max_exit_gain_down",
        "max_exit_return_per_usdc_up",
        "max_exit_return_per_usdc_down",
        "time_to_best_exit_up",
        "time_to_best_exit_down",
        "best_exit_price_up",
        "best_exit_price_down",
        "label_volatility_up",
        "label_volatility_down",
        "volatility_path_validity_up",
        "volatility_path_validity_down",
        "label_profit_up_15m",
        "label_profit_down_15m",
        "label_up_15m",
        "label_down_15m",
        "label_source",
    } <= cols
    assert schema.field("feature_ts").type == pa.int64()
    assert schema.field("target_ts").type == pa.int64()
    assert schema.field("round_start_ts").type == pa.int64()
    assert schema.field("round_end_ts").type == pa.int64()
    assert schema.field("start_price").type == pa.float64()
    assert schema.field("target_price").type == pa.float64()
    assert schema.field("direction_up_15m").type == pa.bool_()
    assert schema.field("entry_ask_price").type == pa.float64()
    assert schema.field("settlement_price").type == pa.float64()
    assert schema.field("entry_fee").type == pa.float64()
    assert schema.field("entry_cost").type == pa.float64()
    assert schema.field("realized_return").type == pa.float64()
    assert schema.field("fee_bps").type == pa.float64()
    assert schema.field("settlement_margin").type == pa.float64()
    assert schema.field("settlement_abs_margin").type == pa.float64()
    assert schema.field("settlement_neutral_margin").type == pa.float64()
    assert schema.field("label_settlement_3way").type == pa.string()
    assert schema.field("max_exit_gain_up").type == pa.float64()
    assert schema.field("max_exit_gain_down").type == pa.float64()
    assert schema.field("label_volatility_up").type == pa.bool_()
    assert schema.field("label_volatility_down").type == pa.bool_()
    assert schema.field("volatility_path_validity_up").type == pa.string()
    assert schema.field("volatility_path_validity_down").type == pa.string()
    assert schema.field("label_profit_up_15m").type == pa.bool_()
    assert schema.field("label_profit_down_15m").type == pa.bool_()
    assert schema.field("label_up_15m").type == pa.bool_()
    assert schema.field("label_down_15m").type == pa.bool_()


def test_generate_labels_uses_polymarket_round_prices_and_aligns_to_feature_ts() -> None:
    from bigan.labels.generation import LABEL_KIND, LABEL_VERSION, generate_labels_15m_v1

    t0 = _ts_at(2026, 5, 13, 12, 0)
    rows = generate_labels_15m_v1(
        feature_rows=[_feature_row(t0 + 60_000), _feature_row(t0 + 10 * 60_000)],
        round_rows=[_round_row(t0, 100.0, 101.0)],
        ingest_ts=t0 + 2 * HORIZON_MS,
    )

    by_ts = {row["feature_ts"]: row for row in rows}
    first = by_ts[t0 + 60_000]
    second = by_ts[t0 + 10 * 60_000]

    assert first["ts"] == t0 + 60_000
    assert first["message_ts"] == t0 + 60_000
    assert first["target_ts"] == t0 + HORIZON_MS
    assert first["start_price"] == 100.0
    assert first["target_price"] == 101.0
    assert first["direction_up_15m"] is True
    assert first["entry_ask_price"] == pytest.approx(0.40)
    assert first["settlement_price"] == pytest.approx(1.0)
    assert first["entry_fee"] == pytest.approx(0.0)
    assert first["entry_cost"] == pytest.approx(0.40)
    assert first["realized_return"] == pytest.approx(0.60)
    assert first["fee_bps"] == pytest.approx(0.0)
    assert first["settlement_margin"] == pytest.approx(1.0)
    assert first["settlement_abs_margin"] == pytest.approx(1.0)
    assert first["settlement_neutral_margin"] == pytest.approx(0.0)
    assert first["label_settlement_3way"] == "UP"
    assert first["label_volatility_up"] is None
    assert first["label_volatility_down"] is None
    assert first["volatility_path_validity_up"] == "missing_price_path"
    assert first["volatility_path_validity_down"] == "missing_price_path"
    assert first["label_profit_up_15m"] is True
    assert first["label_up_15m"] is True
    assert first["label_version"] == LABEL_VERSION
    assert first["label_kind"] == LABEL_KIND
    assert first["round_slug"] == f"btc-updown-15m-{t0 // 1000}"
    assert first["round_start_ts"] == t0
    assert first["round_end_ts"] == t0 + HORIZON_MS

    assert second["target_ts"] == t0 + HORIZON_MS
    assert second["start_price"] == 100.0
    assert second["target_price"] == 101.0
    assert second["realized_return"] == pytest.approx(0.60)
    assert second["label_up_15m"] is True


def test_generate_labels_uses_profitability_not_direction_only() -> None:
    from bigan.labels.generation import generate_labels_15m_v1

    t0 = _ts_at(2026, 5, 13, 12, 0)
    rows = generate_labels_15m_v1(
        feature_rows=[_feature_row(t0 + 60_000, market_implied_prob=1.0)],
        round_rows=[_round_row(t0, 100.0, 100.0)],
        ingest_ts=t0 + 2 * HORIZON_MS,
    )

    assert rows[0]["settlement_price"] == pytest.approx(1.0)
    assert rows[0]["realized_return"] == pytest.approx(0.0)
    assert rows[0]["label_profit_up_15m"] is False
    assert rows[0]["label_up_15m"] is False


def test_generate_labels_populates_directional_3way_settlement_abstention() -> None:
    from bigan.labels.generation import generate_labels_15m_v1

    t0 = _ts_at(2026, 5, 13, 12, 0)
    rows = generate_labels_15m_v1(
        feature_rows=[
            _feature_row(t0 + 60_000, source_symbol="tok-up-small", source_market="0xsmall"),
            _feature_row(t0 + 120_000, source_symbol="tok-up-big", source_market="0xbig"),
            _feature_row(t0 + 180_000, source_symbol="tok-up-down", source_market="0xdown"),
        ],
        round_rows=[
            _round_row(t0, 100.0, 100.03, source_market="0xsmall"),
            _round_row(t0, 100.0, 100.20, source_market="0xbig"),
            _round_row(t0, 100.0, 99.80, source_market="0xdown"),
        ],
        ingest_ts=t0 + 2 * HORIZON_MS,
        settlement_neutral_margin=0.05,
    )

    by_market = {row["source_market"]: row for row in rows}
    assert by_market["0xsmall"]["settlement_margin"] == pytest.approx(0.03)
    assert by_market["0xsmall"]["label_settlement_3way"] == "NEUTRAL"
    assert by_market["0xbig"]["label_settlement_3way"] == "UP"
    assert by_market["0xdown"]["settlement_margin"] == pytest.approx(-0.20)
    assert by_market["0xdown"]["label_settlement_3way"] == "DOWN"


def test_generate_labels_allows_buy_before_round_start_for_same_market() -> None:
    from bigan.labels.generation import generate_labels_15m_v1

    t0 = _ts_at(2026, 5, 13, 12, 0)
    rows = generate_labels_15m_v1(
        feature_rows=[_feature_row(t0 - 5 * 60_000, market_implied_prob=0.45)],
        round_rows=[_round_row(t0, 100.0, 101.0)],
        ingest_ts=t0 + 2 * HORIZON_MS,
    )

    assert len(rows) == 1
    assert rows[0]["feature_ts"] == t0 - 5 * 60_000
    assert rows[0]["target_ts"] == t0 + HORIZON_MS
    assert rows[0]["realized_return"] == pytest.approx(0.55)
    assert rows[0]["label_profit_up_15m"] is True


def test_generate_labels_skips_features_after_market_settlement() -> None:
    from bigan.labels.generation import generate_labels_15m_v1

    t0 = _ts_at(2026, 5, 13, 12, 0)
    rows = generate_labels_15m_v1(
        feature_rows=[_feature_row(t0 + HORIZON_MS)],
        round_rows=[_round_row(t0, 100.0, 101.0)],
        ingest_ts=t0 + 2 * HORIZON_MS,
    )

    assert rows == []


def test_generate_labels_requires_source_market_match_when_present() -> None:
    from bigan.labels.generation import generate_labels_15m_v1

    t0 = _ts_at(2026, 5, 13, 12, 0)
    rows = generate_labels_15m_v1(
        feature_rows=[_feature_row(t0 + 60_000, source_market="0xfeature")],
        round_rows=[_round_row(t0, 100.0, 101.0, source_market="0xother")],
        ingest_ts=t0 + 2 * HORIZON_MS,
    )

    assert rows == []


def test_generate_labels_applies_fee_to_realized_return() -> None:
    from bigan.labels.generation import generate_labels_15m_v1

    t0 = _ts_at(2026, 5, 13, 12, 0)
    rows = generate_labels_15m_v1(
        feature_rows=[_feature_row(t0 + 60_000, market_implied_prob=0.9999)],
        round_rows=[_round_row(t0, 100.0, 101.0)],
        ingest_ts=t0 + 2 * HORIZON_MS,
        fee_bps=2.0,
    )

    assert rows[0]["entry_fee"] == pytest.approx(0.9999 * 0.0002)
    assert rows[0]["entry_cost"] == pytest.approx(0.9999 + rows[0]["entry_fee"])
    assert rows[0]["realized_return"] < 0.0
    assert rows[0]["label_profit_up_15m"] is False
    assert rows[0]["label_up_15m"] is False


def test_generate_labels_skips_missing_market_implied_probability() -> None:
    from bigan.labels.generation import generate_labels_15m_v1

    t0 = _ts_at(2026, 5, 13, 12, 0)
    feature = _feature_row(t0 + 60_000)
    feature.pop("market_implied_prob")

    rows = generate_labels_15m_v1(
        feature_rows=[feature],
        round_rows=[_round_row(t0, 100.0, 101.0)],
        ingest_ts=t0 + 2 * HORIZON_MS,
    )

    assert rows == []


def test_generate_labels_populates_down_token_profitability() -> None:
    from bigan.labels.generation import DOWN_LABEL_KIND, generate_labels_15m_v1

    t0 = _ts_at(2026, 5, 13, 12, 0)
    up_won_rows = generate_labels_15m_v1(
        feature_rows=[
            _feature_row(
                t0 + 60_000,
                source_symbol="tok-down",
                canonical_symbol="BTC-15M:btc-updown-15m-test:DOWN",
                market_implied_prob=0.40,
            )
        ],
        round_rows=[_round_row(t0, 100.0, 101.0)],
        ingest_ts=t0 + 2 * HORIZON_MS,
    )

    assert len(up_won_rows) == 1
    up_won = up_won_rows[0]
    assert up_won["label_kind"] == DOWN_LABEL_KIND
    assert up_won["settlement_price"] == pytest.approx(0.0)
    assert up_won["entry_cost"] == pytest.approx(0.40)
    assert up_won["realized_return"] == pytest.approx(-0.40)
    assert up_won["label_profit_up_15m"] is None
    assert up_won["label_profit_down_15m"] is False
    assert up_won["label_up_15m"] is False
    assert up_won["label_down_15m"] is False

    down_won_rows = generate_labels_15m_v1(
        feature_rows=[
            _feature_row(
                t0 + 60_000,
                source_symbol="tok-down",
                canonical_symbol="BTC-15M:btc-updown-15m-test:DOWN",
                market_implied_prob=0.40,
            )
        ],
        round_rows=[_round_row(t0, 100.0, 99.0)],
        ingest_ts=t0 + 2 * HORIZON_MS,
    )
    down_won = down_won_rows[0]
    assert down_won["direction_up_15m"] is False
    assert down_won["settlement_price"] == pytest.approx(1.0)
    assert down_won["realized_return"] == pytest.approx(0.60)
    assert down_won["label_profit_down_15m"] is True
    assert down_won["label_down_15m"] is True


def test_generate_labels_skips_unmapped_token_features() -> None:
    from bigan.labels.generation import generate_labels_15m_v1

    t0 = _ts_at(2026, 5, 13, 12, 0)
    feature = _feature_row(t0 + 60_000)
    feature["canonical_symbol"] = None
    feature["symbol"] = "tok-up"

    rows = generate_labels_15m_v1(
        feature_rows=[feature],
        round_rows=[_round_row(t0, 100.0, 101.0)],
        ingest_ts=t0 + 2 * HORIZON_MS,
    )

    assert rows == []


def test_generate_labels_ignores_late_corrected_round_rows() -> None:
    from bigan.labels.generation import generate_labels_15m_v1

    t0 = _ts_at(2026, 5, 13, 12, 0)
    rows = generate_labels_15m_v1(
        feature_rows=[_feature_row(t0 + 60_000)],
        round_rows=[
            _round_row(t0, 100.0, 101.0, ingest_offset_ms=100),
            # Same round, but discovered much later. The label must be
            # reproducible from the round metadata known to this label run, not
            # from a later correction/backfill.
            _round_row(t0, 100.0, 90.0, ingest_offset_ms=HORIZON_MS),
        ],
        ingest_ts=t0 + HORIZON_MS + 1_000,
    )

    assert len(rows) == 1
    assert rows[0]["target_price"] == 101.0
    assert rows[0]["label_up_15m"] is True


def test_run_label_batch_writes_independent_labels_table(tmp_path: Path) -> None:
    from bigan.labels.generation import LABEL_VERSION, run_label_batch
    from bigan.monitoring import record_prediction_rows_as_events

    warehouse = tmp_path / "warehouse"
    mlops_db = tmp_path / "mlops.duckdb"
    t0 = _ts_at(2026, 5, 13, 12, 0)
    with WarehouseWriter(warehouse, max_rows_per_partition=10) as writer:
        writer.append_rows("features_15m_v1", [_feature_row(t0 + 60_000)])
    conn = duckdb.connect(str(mlops_db))
    record_prediction_rows_as_events(
        conn,
        [
            {
                "prediction_ts": t0 + 60_000,
                "source": "polymarket",
                "source_symbol": "tok-up",
                "source_market": "mkt-1",
                "canonical_symbol": "BTC-UP-15M",
                "symbol": "BTC-UP-15M",
                "feature_version": "bigan-mvp-v1.0.0",
                "model_version": "xgboost-v3",
                "prob_up_15m": 0.80,
                "market_implied_prob": 0.40,
                "confidence_bucket": "high_up",
                "top_features_json": "[]",
                "feature_values_json": "{}",
            }
        ],
    )
    conn.close()

    report = run_label_batch(
        warehouse,
        ingest_ts=t0 + 2 * HORIZON_MS,
        round_rows=[_round_row(t0, 100.0, 101.0)],
        monitoring_db_path=mlops_db,
        monitoring_model_version="xgboost-v3",
    )

    assert report.rows_generated == 1
    assert report.rows_written == 1
    assert report.label_version == LABEL_VERSION
    assert report.monitoring_outcomes_written == 1
    files = warehouse_files(warehouse, "labels_15m_v1")
    assert len(files) == 1
    row = pq.ParquetFile(files[0]).read().to_pylist()[0]
    assert row["feature_ts"] == t0 + 60_000
    assert row["target_ts"] == t0 + HORIZON_MS
    assert row["start_price"] == 100.0
    assert row["target_price"] == 101.0
    assert row["direction_up_15m"] is True
    assert row["entry_ask_price"] == pytest.approx(0.40)
    assert row["settlement_price"] == pytest.approx(1.0)
    assert row["entry_cost"] == pytest.approx(0.40)
    assert row["realized_return"] == pytest.approx(0.60)
    assert row["label_profit_up_15m"] is True
    assert row["label_profit_down_15m"] is None
    assert row["label_up_15m"] is True
    assert row["label_down_15m"] is None


def test_run_label_batch_can_skip_existing_label_rows(tmp_path: Path) -> None:
    from bigan.canonical.query import open_warehouse
    from bigan.labels.generation import run_label_batch

    warehouse = tmp_path / "warehouse"
    t0 = _ts_at(2026, 5, 13, 12, 0)
    with WarehouseWriter(warehouse, max_rows_per_partition=10) as writer:
        writer.append_rows("features_15m_v1", [_feature_row(t0 + 60_000)])

    first = run_label_batch(
        warehouse,
        ingest_ts=t0 + 2 * HORIZON_MS,
        round_rows=[_round_row(t0, 100.0, 101.0)],
        skip_existing_labels=True,
    )
    second = run_label_batch(
        warehouse,
        ingest_ts=t0 + 2 * HORIZON_MS + 1,
        round_rows=[_round_row(t0, 100.0, 101.0)],
        skip_existing_labels=True,
    )

    assert first.rows_generated == 1
    assert first.rows_written == 1
    assert second.rows_generated == 1
    assert second.rows_written == 0
    with open_warehouse(warehouse) as conn:
        assert conn.execute("SELECT COUNT(*) FROM labels_15m_v1").fetchone()[0] == 1


def test_run_label_batch_can_limit_feature_window(tmp_path: Path) -> None:
    from bigan.canonical.query import open_warehouse
    from bigan.labels.generation import run_label_batch

    warehouse = tmp_path / "warehouse"
    t0 = _ts_at(2026, 5, 13, 12, 0)
    with WarehouseWriter(warehouse, max_rows_per_partition=10) as writer:
        writer.append_rows(
            "features_15m_v1",
            [
                _feature_row(t0 + 60_000, source_symbol="early"),
                _feature_row(t0 + 120_000, source_symbol="late"),
            ],
        )

    report = run_label_batch(
        warehouse,
        ingest_ts=t0 + 2 * HORIZON_MS,
        round_rows=[_round_row(t0, 100.0, 101.0)],
        since_ms=t0 + 120_000,
        until_ms=t0 + 180_000,
    )

    assert report.rows_generated == 1
    assert report.rows_written == 1
    with open_warehouse(warehouse) as conn:
        rows = conn.execute(
            "SELECT source_symbol FROM labels_15m_v1 ORDER BY source_symbol"
        ).fetchall()
    assert rows == [("late",)]


def test_polymarket_gamma_event_metadata_maps_to_round_row() -> None:
    from bigan.labels.generation import polymarket_round_row_from_event

    row = polymarket_round_row_from_event(
        {
            "slug": "btc-updown-15m-1778940000",
            "startTime": "2026-05-16T14:00:00Z",
            "endDate": "2026-05-16T14:15:00Z",
            "eventMetadata": {
                "priceToBeat": 77981.58552825246,
                "finalPrice": 77926.177315,
            },
            "markets": [{"conditionId": "0xc672"}],
        },
        ingest_ts=1_778_942_720_000,
    )

    assert row is not None
    assert row["source_market"] == "0xc672"
    assert row["round_start_ts"] == _ts_at(2026, 5, 16, 14, 0)
    assert row["round_end_ts"] == _ts_at(2026, 5, 16, 14, 15)
    assert row["start_price"] == 77981.58552825246
    assert row["target_price"] == 77926.177315


def test_polymarket_gamma_market_outcome_prices_map_to_round_row() -> None:
    from bigan.labels.generation import polymarket_round_row_from_market

    row = polymarket_round_row_from_market(
        {
            "slug": "btc-updown-15m-1779460200",
            "conditionId": "0xclosed",
            "closed": True,
            "eventStartTime": "2026-05-22T14:30:00Z",
            "endDate": "2026-05-22T14:45:00Z",
            "outcomes": '["Up", "Down"]',
            "outcomePrices": '["0", "1"]',
        },
        ingest_ts=1_779_461_200_000,
    )

    assert row is not None
    assert row["source_market"] == "0xclosed"
    assert row["round_slug"] == "btc-updown-15m-1779460200"
    assert row["round_start_ts"] == _ts_at(2026, 5, 22, 14, 30)
    assert row["round_end_ts"] == _ts_at(2026, 5, 22, 14, 45)
    assert row["start_price"] == 0.5
    assert row["target_price"] == 0.0


def test_round_metadata_fetch_includes_canonical_market_slug(monkeypatch: pytest.MonkeyPatch) -> None:
    from bigan.labels import generation

    seen_slugs: list[str] = []

    def fake_fetch(_base: str, slug: str, *, request_timeout_seconds: float):
        seen_slugs.append(slug)
        return {
            "slug": slug,
            "startTime": "2026-05-13T20:00:00Z",
            "endDate": "2026-05-13T20:15:00Z",
            "eventMetadata": {"priceToBeat": 100.0, "finalPrice": 101.0},
            "markets": [{"conditionId": "0xmkt"}],
        }

    monkeypatch.setattr(generation, "_fetch_gamma_event_by_slug", fake_fetch)

    t0 = _ts_at(2026, 5, 13, 12, 0)
    rows = generation.fetch_polymarket_round_rows_for_features(
        [
            _feature_row(
                t0,
                canonical_symbol="BTC-15M:btc-updown-15m-1778712000:UP",
            )
        ],
        ingest_ts=t0,
    )

    assert "btc-updown-15m-1778712000" in seen_slugs
    assert rows


def test_round_metadata_fetch_infers_eth_5m_slug_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bigan.labels import generation

    seen_slugs: list[str] = []

    def fake_fetch(_base: str, slug: str, *, request_timeout_seconds: float):
        seen_slugs.append(slug)
        return {
            "slug": slug,
            "startTime": "2026-05-13T20:00:00Z",
            "endDate": "2026-05-13T20:05:00Z",
            "eventMetadata": {"priceToBeat": 100.0, "finalPrice": 101.0},
            "markets": [{"conditionId": "0xeth"}],
        }

    monkeypatch.setattr(generation, "_fetch_gamma_event_by_slug", fake_fetch)

    t0 = _ts_at(2026, 5, 13, 12, 0)
    rows = generation.fetch_polymarket_round_rows_for_features(
        [
            _feature_row(
                t0,
                canonical_symbol="ETH-5M:eth-updown-5m-1778712000:UP",
            )
        ],
        ingest_ts=t0,
    )

    assert "eth-updown-5m-1778712000" in seen_slugs
    assert all(slug.startswith("eth-updown-5m-") for slug in seen_slugs)
    assert rows


def test_round_metadata_fetch_falls_back_to_market_slug(monkeypatch: pytest.MonkeyPatch) -> None:
    from bigan.labels import generation

    seen_event_slugs: list[str] = []
    seen_market_slugs: list[str] = []

    def fake_event_fetch(_base: str, slug: str, *, request_timeout_seconds: float):
        seen_event_slugs.append(slug)
        return None

    def fake_market_fetch(_base: str, slug: str, *, request_timeout_seconds: float):
        seen_market_slugs.append(slug)
        return {
            "slug": slug,
            "conditionId": "0xmkt",
            "closed": True,
            "eventStartTime": "2026-05-13T20:00:00Z",
            "endDate": "2026-05-13T20:15:00Z",
            "outcomes": ["Up", "Down"],
            "outcomePrices": ["1", "0"],
        }

    monkeypatch.setattr(generation, "_fetch_gamma_event_by_slug", fake_event_fetch)
    monkeypatch.setattr(generation, "_fetch_gamma_market_by_slug", fake_market_fetch)

    t0 = _ts_at(2026, 5, 13, 12, 0)
    rows = generation.fetch_polymarket_round_rows_for_features(
        [
            _feature_row(
                t0,
                canonical_symbol="BTC-15M:btc-updown-15m-1778712000:UP",
            )
        ],
        ingest_ts=t0,
    )

    assert "btc-updown-15m-1778712000" in seen_event_slugs
    assert "btc-updown-15m-1778712000" in seen_market_slugs
    assert rows


def test_gamma_fetch_warning_messages_include_slug_and_error_context(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from bigan.labels import generation

    def fail_urlopen(*_args: object, **_kwargs: object) -> object:
        raise error.URLError("timed out")

    monkeypatch.setattr(generation.request, "urlopen", fail_urlopen)

    with caplog.at_level(logging.WARNING, logger="bigan.labels.generation"):
        assert (
            generation._fetch_gamma_event_by_slug(
                "https://gamma.example",
                "btc-updown-15m-1778712000",
                request_timeout_seconds=1.0,
            )
            is None
        )
        assert (
            generation._fetch_gamma_market_by_slug(
                "https://gamma.example",
                "btc-updown-15m-1778712000",
                request_timeout_seconds=1.0,
            )
            is None
        )

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "labels.gamma_event_fetch_failed slug=btc-updown-15m-1778712000" in message
        and "timed out" in message
        for message in messages
    )
    market_messages = [
        message for message in messages if "labels.gamma_market_fetch_failed" in message
    ]
    assert len(market_messages) == 1
    assert "slug=btc-updown-15m-1778712000" in market_messages[0]
    assert "failed_attempts=3" in market_messages[0]
    assert "timed out" in market_messages[0]
