"""TDD contract for issue #9 labels_15m_v1 generation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

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
        "label_profit_up_15m",
        "label_up_15m",
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
    assert schema.field("label_profit_up_15m").type == pa.bool_()
    assert schema.field("label_up_15m").type == pa.bool_()


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


def test_generate_labels_skips_down_token_features() -> None:
    from bigan.labels.generation import generate_labels_15m_v1

    t0 = _ts_at(2026, 5, 13, 12, 0)
    rows = generate_labels_15m_v1(
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

    assert rows == []


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

    warehouse = tmp_path / "warehouse"
    t0 = _ts_at(2026, 5, 13, 12, 0)
    with WarehouseWriter(warehouse, max_rows_per_partition=10) as writer:
        writer.append_rows("features_15m_v1", [_feature_row(t0 + 60_000)])

    report = run_label_batch(
        warehouse,
        ingest_ts=t0 + 2 * HORIZON_MS,
        round_rows=[_round_row(t0, 100.0, 101.0)],
    )

    assert report.rows_generated == 1
    assert report.rows_written == 1
    assert report.label_version == LABEL_VERSION
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
    assert row["label_up_15m"] is True


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
