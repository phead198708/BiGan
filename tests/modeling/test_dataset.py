"""TDD contract for issue #15 training dataset assembly."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from bigan.canonical.writer import WarehouseWriter

HORIZON_MS = 15 * 60_000


def _ts_at(year: int, month: int, day: int, hour: int, minute: int) -> int:
    return int(datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp() * 1000)


def _feature_row(
    ts: int,
    *,
    source_symbol: str = "tok-up",
    completeness_score: float = 1.0,
    data_gap_flag: bool = False,
    quality_filter_pass: bool = True,
    mid_price: float = 0.50,
) -> dict:
    return {
        "ts": ts,
        "message_ts": ts,
        "feature_ts": ts,
        "ingest_ts": ts + 100,
        "source": "polymarket",
        "source_symbol": source_symbol,
        "source_market": "0xmkt",
        "canonical_symbol": "BTC-UP-15M",
        "symbol": "BTC-UP-15M",
        "feature_version": "bigan-mvp-v1.0.0",
        "completeness_score": completeness_score,
        "data_gap_flag": data_gap_flag,
        "quality_filter_pass": quality_filter_pass,
        "quote_age_ms": 0,
        "depth_age_ms": 0,
        "trade_age_ms": 0,
        "spread": 0.02,
        "mid_price": mid_price,
        "microprice": mid_price + 0.001,
        "obi_l1": 0.1,
        "obi_l5": 0.2,
        "obi_l10": 0.3,
        "signed_volume_1m": 5.0,
        "trade_imbalance_1m": 0.25,
        "trade_count_1m": 3,
        "trade_volume_1m": 12.0,
        "ret_1m": 0.01,
        "ret_5m": 0.02,
        "ret_15m": 0.03,
        "rv_1m": 0.001,
        "rv_5m": 0.002,
        "rv_15m": 0.003,
    }


def _label_row(
    feature_ts: int,
    *,
    source_symbol: str = "tok-up",
    label_up_15m: bool = True,
    target_ts: int | None = None,
) -> dict:
    start_ts = feature_ts - 60_000
    resolved_target_ts = feature_ts + HORIZON_MS if target_ts is None else target_ts
    return {
        "ts": feature_ts,
        "message_ts": feature_ts,
        "feature_ts": feature_ts,
        "target_ts": resolved_target_ts,
        "ingest_ts": resolved_target_ts + 1_000,
        "source": "polymarket",
        "source_symbol": source_symbol,
        "source_market": "0xmkt",
        "canonical_symbol": "BTC-UP-15M",
        "symbol": "BTC-UP-15M",
        "label_version": "bigan-labels-15m-v1.0.0",
        "round_slug": f"btc-updown-15m-{start_ts // 1000}",
        "round_start_ts": start_ts,
        "round_end_ts": resolved_target_ts,
        "start_price": 100.0,
        "target_price": 101.0 if label_up_15m else 99.0,
        "label_up_15m": label_up_15m,
        "label_source": "polymarket_gamma_event_metadata",
    }


def _write_training_fixture(
    warehouse: Path,
    *,
    feature_rows: list[dict],
    label_rows: list[dict],
) -> None:
    with WarehouseWriter(warehouse, max_rows_per_partition=100) as writer:
        writer.append_rows("features_15m_v1", feature_rows)
        writer.append_rows("labels_15m_v1", label_rows)


def test_assemble_training_dataset_joins_filters_and_time_splits(tmp_path: Path) -> None:
    from bigan.modeling.dataset import SplitConfig, assemble_training_dataset

    warehouse = tmp_path / "warehouse"
    output_dir = tmp_path / "dataset"
    t0 = _ts_at(2026, 5, 13, 12, 0)
    good_ts = [t0 + minute * 60_000 for minute in range(6)]
    gap_ts = t0 + 6 * 60_000
    low_quality_ts = t0 + 7 * 60_000
    missing_label_ts = t0 + 8 * 60_000
    feature_rows = [
        _feature_row(ts, mid_price=0.50 + idx * 0.01)
        for idx, ts in enumerate(good_ts)
    ]
    feature_rows.extend(
        [
            _feature_row(gap_ts, data_gap_flag=True, quality_filter_pass=False),
            _feature_row(low_quality_ts, completeness_score=0.40, quality_filter_pass=False),
            _feature_row(missing_label_ts),
        ]
    )
    label_rows = [
        _label_row(ts, label_up_15m=(idx % 2 == 0))
        for idx, ts in enumerate(good_ts)
    ]
    label_rows.extend([_label_row(gap_ts), _label_row(low_quality_ts)])
    _write_training_fixture(warehouse, feature_rows=feature_rows, label_rows=label_rows)

    report = assemble_training_dataset(
        warehouse,
        output_dir,
        split_config=SplitConfig(train_fraction=0.60, val_fraction=0.20),
        min_completeness_score=0.80,
    )

    assert report.rows_joined == 8
    assert report.rows_written == 6
    assert report.rows_filtered_quality == 2
    assert report.rows_missing_label == 1
    assert report.splits["train"].row_count == 3
    assert report.splits["val"].row_count == 1
    assert report.splits["test"].row_count == 2
    assert report.splits["train"].positive_count == 2
    assert report.splits["train"].negative_count == 1
    assert report.feature_versions == ("bigan-mvp-v1.0.0",)
    assert report.label_versions == ("bigan-labels-15m-v1.0.0",)
    assert {"spread", "mid_price", "ret_15m"} <= set(report.feature_columns)

    train = pq.read_table(output_dir / "train.parquet")
    val = pq.read_table(output_dir / "val.parquet")
    test = pq.read_table(output_dir / "test.parquet")
    assert train.column("feature_ts").to_pylist() == good_ts[:3]
    assert val.column("feature_ts").to_pylist() == good_ts[3:4]
    assert test.column("feature_ts").to_pylist() == good_ts[4:]
    assert all(
        target_ts > feature_ts
        for feature_ts, target_ts in zip(
            test.column("feature_ts").to_pylist(),
            test.column("target_ts").to_pylist(),
            strict=True,
        )
    )
    assert "label_up_15m" in train.schema.names
    assert "mid_price" in train.schema.names

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rows_written"] == 6
    assert manifest["splits"]["test"]["positive_count"] == 1
    assert manifest["split_config"] == {"train_fraction": 0.6, "val_fraction": 0.2}


def test_assemble_training_dataset_is_reproducible(tmp_path: Path) -> None:
    from bigan.modeling.dataset import assemble_training_dataset

    warehouse = tmp_path / "warehouse"
    output_dir = tmp_path / "dataset"
    t0 = _ts_at(2026, 5, 13, 12, 0)
    feature_rows = [_feature_row(t0 + minute * 60_000) for minute in range(5)]
    label_rows = [
        _label_row(t0 + minute * 60_000, label_up_15m=(minute < 3))
        for minute in range(5)
    ]
    _write_training_fixture(warehouse, feature_rows=feature_rows, label_rows=label_rows)

    first = assemble_training_dataset(warehouse, output_dir).to_dict()
    first_rows = pq.read_table(output_dir / "train.parquet").to_pylist()
    second = assemble_training_dataset(warehouse, output_dir).to_dict()
    second_rows = pq.read_table(output_dir / "train.parquet").to_pylist()

    assert second == first
    assert second_rows == first_rows


def test_assemble_training_dataset_rejects_label_leakage(tmp_path: Path) -> None:
    from bigan.modeling.dataset import assemble_training_dataset

    warehouse = tmp_path / "warehouse"
    t0 = _ts_at(2026, 5, 13, 12, 0)
    _write_training_fixture(
        warehouse,
        feature_rows=[_feature_row(t0)],
        label_rows=[_label_row(t0, target_ts=t0)],
    )

    with pytest.raises(ValueError, match="future information leakage"):
        assemble_training_dataset(warehouse, tmp_path / "dataset")
