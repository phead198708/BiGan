"""Same-dataset model evaluation for promotion gate checks."""

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
        "market_implied_prob": mid_price,
        "ret_15m": mid_price - 0.50,
    }


def _write_split(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def _write_dataset(dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True)
    train_mid_prices = [0.38, 0.41, 0.44, 0.47, 0.53, 0.56, 0.59, 0.62]
    val_mid_prices = [0.40, 0.46, 0.55, 0.61]
    test_mid_prices = [0.39, 0.45, 0.57, 0.63]
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


def test_evaluate_probability_model_on_same_dataset_writes_bootstrap_compatible_metrics(
    tmp_path: Path,
) -> None:
    from bigan.modeling import (
        XGBoostV1Config,
        evaluate_probability_model_on_dataset,
        train_xgboost_v1,
    )

    dataset_dir = tmp_path / "dataset"
    model_dir = tmp_path / "model"
    eval_dir = tmp_path / "eval"
    _write_dataset(dataset_dir)
    trained = train_xgboost_v1(
        dataset_dir,
        model_dir,
        config=XGBoostV1Config(
            rounds_grid=(5,),
            learning_rate_grid=(0.60,),
            max_depth_grid=(2,),
        ),
    )

    report = evaluate_probability_model_on_dataset(
        model_dir / "model.json",
        dataset_dir,
        eval_dir,
    )

    assert report.model_version == "xgboost-v1"
    assert report.metrics == trained.metrics
    assert report.family_metrics["test"]["BTC-15M"]["sample_count"] == 4
    assert report.probability_distributions["val"]["split"] == "val"
    assert report.probability_distributions["val"]["probability_distribution"]["count"] == 4
    assert json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8")) == report.metrics
    assert (eval_dir / "family_metrics.json").exists()
    assert (eval_dir / "probability_distributions.json").exists()
    offline_reference = json.loads((eval_dir / "offline_reference.json").read_text(encoding="utf-8"))
    assert offline_reference["split"] == "val"
    assert offline_reference["model_version"] == "xgboost-v1"
    assert offline_reference["dataset_version"] == "bigan-training-15m-v1.0.0"
    assert offline_reference["probability_distribution"]["count"] == 4
    assert offline_reference["edge_distribution"]["count"] == 4
    assert 0.0 <= offline_reference["edge_trigger_rate_at_0_30"] <= 1.0
    manifest = json.loads((eval_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_dir"] == str(dataset_dir)
    assert manifest["model_path"] == str(model_dir / "model.json")
    assert manifest["probability_distributions"]["val"]["split"] == "val"


def test_evaluate_probability_model_can_apply_saved_calibration(tmp_path: Path) -> None:
    from bigan.modeling import (
        XGBoostV1Config,
        evaluate_probability_model_on_dataset,
        fit_probability_calibration,
        train_xgboost_v1,
    )

    dataset_dir = tmp_path / "dataset"
    model_dir = tmp_path / "model"
    calibration_dir = tmp_path / "calibration"
    eval_dir = tmp_path / "eval"
    _write_dataset(dataset_dir)
    train_xgboost_v1(
        dataset_dir,
        model_dir,
        config=XGBoostV1Config(
            rounds_grid=(5,),
            learning_rate_grid=(0.60,),
            max_depth_grid=(2,),
        ),
    )
    calibration = fit_probability_calibration(
        model_dir / "model.json",
        dataset_dir,
        calibration_dir,
    )

    report = evaluate_probability_model_on_dataset(
        model_dir / "model.json",
        dataset_dir,
        eval_dir,
        calibration_path=calibration_dir / "calibration.json",
    )

    assert report.calibration_method == calibration.method
    assert report.metrics["val"]["sample_count"] == 4
    assert report.metrics["val"]["ece"] is not None
    assert report.metrics["test"]["brier_score"] == pytest.approx(
        json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))["test"]["brier_score"]
    )
