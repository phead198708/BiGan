"""TDD contract for issue #18 probability calibration."""

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
    _write_split(
        dataset_dir / "train.parquet",
        [
            _sample(idx * 60_000, mid, label=mid >= 0.50)
            for idx, mid in enumerate([0.38, 0.41, 0.44, 0.47, 0.53, 0.56, 0.59, 0.62])
        ],
    )
    _write_split(
        dataset_dir / "val.parquet",
        [
            _sample(600_000 + idx * 60_000, mid, label=mid >= 0.50)
            for idx, mid in enumerate([0.40, 0.46, 0.55, 0.61])
        ],
    )
    _write_split(
        dataset_dir / "test.parquet",
        [
            _sample(900_000 + idx * 60_000, mid, label=mid >= 0.50)
            for idx, mid in enumerate([0.39, 0.45, 0.57, 0.63])
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


def test_fit_probability_calibration_compares_methods_and_saves_artifact(tmp_path: Path) -> None:
    from bigan.modeling import (
        CalibrationConfig,
        XGBoostV1Config,
        fit_probability_calibration,
        train_xgboost_v1,
    )

    dataset_dir = tmp_path / "dataset"
    model_dir = tmp_path / "xgb"
    output_dir = tmp_path / "calibration"
    _write_dataset(dataset_dir)
    train_xgboost_v1(
        dataset_dir,
        model_dir,
        config=XGBoostV1Config(rounds_grid=(3,), learning_rate_grid=(0.20,)),
    )

    report = fit_probability_calibration(
        model_dir / "model.json",
        dataset_dir,
        output_dir,
        config=CalibrationConfig(methods=("platt", "isotonic"), ece_bins=2),
    )

    assert report.model_version == "xgboost-v1"
    assert report.method in {"platt", "isotonic"}
    assert report.raw_metrics["brier_score"] is not None
    assert report.calibrated_metrics["brier_score"] <= report.raw_metrics["brier_score"]
    assert report.calibrated_metrics["ece"] <= report.raw_metrics["ece"]
    assert report.improved is True
    assert (output_dir / "calibration.json").exists()
    assert (output_dir / "calibration_report.json").exists()
    assert json.loads((output_dir / "calibration_report.json").read_text(encoding="utf-8"))[
        "method"
    ] == report.method


def test_loaded_probability_calibrator_transforms_online_probabilities(tmp_path: Path) -> None:
    from bigan.modeling import (
        CalibrationConfig,
        fit_calibration_from_predictions,
        load_probability_calibrator,
    )

    output_dir = tmp_path / "calibration"
    fit_calibration_from_predictions(
        y_true=[0, 0, 1, 1],
        y_prob=[0.35, 0.40, 0.60, 0.65],
        output_dir=output_dir,
        model_version="xgboost-v1",
        config=CalibrationConfig(methods=("isotonic",), ece_bins=2),
    )

    calibrator = load_probability_calibrator(output_dir / "calibration.json")
    low, high = calibrator.transform_many([0.38, 0.62])

    assert calibrator.method == "isotonic"
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert high >= low
    assert high > 0.80


def test_calibration_rejects_single_class_labels(tmp_path: Path) -> None:
    from bigan.modeling import fit_calibration_from_predictions

    with pytest.raises(ValueError, match="both positive and negative"):
        fit_calibration_from_predictions(
            y_true=[1, 1, 1],
            y_prob=[0.60, 0.70, 0.80],
            output_dir=tmp_path / "calibration",
            model_version="xgboost-v1",
        )
