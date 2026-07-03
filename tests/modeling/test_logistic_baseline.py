"""TDD contract for issue #16 logistic regression baseline."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _sample(feature_ts: int, mid_price: float, label: bool) -> dict:
    return {
        "source": "polymarket",
        "source_symbol": "tok-up",
        "source_market": "0xmkt",
        "canonical_symbol": "BTC-UP-15M",
        "symbol": "BTC-UP-15M",
        "feature_ts": feature_ts,
        "feature_version": "bigan-mvp-v1.0.0",
        "label_version": "bigan-labels-15m-v1.0.0",
        "target_ts": feature_ts + 900_000,
        "round_start_ts": feature_ts - 60_000,
        "round_end_ts": feature_ts + 900_000,
        "start_price": 100.0,
        "target_price": 101.0 if label else 99.0,
        "label_up_15m": label,
        "completeness_score": 1.0,
        "data_gap_flag": False,
        "quality_filter_pass": True,
        "spread": 0.02,
        "mid_price": mid_price,
        "ret_15m": mid_price - 0.50,
    }


def _write_split(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def _write_dataset(dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True)
    train_mid_prices = [0.40, 0.42, 0.44, 0.46, 0.54, 0.56, 0.58, 0.60]
    val_mid_prices = [0.43, 0.47, 0.55, 0.59]
    test_mid_prices = [0.41, 0.45, 0.57, 0.61]

    _write_split(
        dataset_dir / "train.parquet",
        [
            _sample(idx * 60_000, mid, label=mid >= 0.50)
            for idx, mid in enumerate(train_mid_prices)
        ],
    )
    _write_split(
        dataset_dir / "val.parquet",
        [
            _sample(600_000 + idx * 60_000, mid, label=mid >= 0.50)
            for idx, mid in enumerate(val_mid_prices)
        ],
    )
    _write_split(
        dataset_dir / "test.parquet",
        [
            _sample(900_000 + idx * 60_000, mid, label=mid >= 0.50)
            for idx, mid in enumerate(test_mid_prices)
        ],
    )
    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": "bigan-training-15m-v1.0.0",
                "feature_columns": ["spread", "mid_price", "ret_15m"],
                "feature_versions": ["bigan-mvp-v1.0.0"],
                "label_versions": ["bigan-labels-15m-v1.0.0"],
                "rows_written": 16,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_train_logistic_baseline_saves_reproducible_artifacts(tmp_path: Path) -> None:
    from bigan.modeling import LogisticBaselineConfig, train_logistic_baseline

    dataset_dir = tmp_path / "dataset"
    first_output = tmp_path / "run-a"
    second_output = tmp_path / "run-b"
    _write_dataset(dataset_dir)

    config = LogisticBaselineConfig(epochs=400, learning_rate=0.50, l2_penalty=0.0)
    first = train_logistic_baseline(dataset_dir, first_output, config=config)
    second = train_logistic_baseline(dataset_dir, second_output, config=config)

    assert first.model_version == "logreg-baseline-v1"
    assert first.metrics["train"]["sample_count"] == 8
    assert first.metrics["val"]["sample_count"] == 4
    assert first.metrics["test"]["sample_count"] == 4
    assert first.metrics["test"]["accuracy"] >= 0.75
    assert first.metrics["test"]["roc_auc"] == pytest.approx(1.0)
    assert first.metrics["test"]["brier_score"] < 0.25
    assert first.metrics["test"]["ece"] is not None
    assert 0.0 <= first.metrics["test"]["ece"] <= 1.0
    assert (first_output / "model.json").exists()
    assert (first_output / "baseline_config.json").exists()
    assert (first_output / "metrics.json").exists()
    assert (first_output / "family_metrics.json").exists()
    assert (first_output / "manifest.json").exists()
    assert json.loads((first_output / "baseline_config.json").read_text(encoding="utf-8"))[
        "learning_rate"
    ] == 0.50
    assert json.loads((first_output / "metrics.json").read_text(encoding="utf-8")) == first.metrics
    family_metrics = json.loads((first_output / "family_metrics.json").read_text(encoding="utf-8"))
    assert family_metrics["test"]["BTC-15M"]["sample_count"] == 4
    assert first.family_metrics["test"]["BTC-15M"]["sample_count"] == 4
    assert json.loads((second_output / "model.json").read_text(encoding="utf-8")) == json.loads(
        (first_output / "model.json").read_text(encoding="utf-8")
    )
    assert second.metrics == first.metrics


def test_saved_logistic_baseline_predicts_with_training_schema(tmp_path: Path) -> None:
    from bigan.modeling import (
        LogisticBaselineConfig,
        load_logistic_baseline,
        train_logistic_baseline,
    )

    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "baseline"
    _write_dataset(dataset_dir)

    train_logistic_baseline(
        dataset_dir,
        output_dir,
        config=LogisticBaselineConfig(epochs=400, learning_rate=0.50, l2_penalty=0.0),
    )
    model = load_logistic_baseline(output_dir / "model.json")
    low_prob, high_prob = model.predict_proba_many(
        [
            {"spread": 0.02, "mid_price": 0.42, "ret_15m": -0.08},
            {"spread": 0.02, "mid_price": 0.60, "ret_15m": 0.10},
        ]
    )

    assert model.feature_columns == ("spread", "mid_price", "ret_15m")
    assert 0.0 <= low_prob <= 1.0
    assert 0.0 <= high_prob <= 1.0
    assert high_prob > low_prob


def test_train_logistic_baseline_rejects_missing_feature_columns(tmp_path: Path) -> None:
    from bigan.modeling import train_logistic_baseline

    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["feature_columns"] = ["spread", "not_present"]
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="missing feature columns"):
        train_logistic_baseline(dataset_dir, tmp_path / "baseline")
