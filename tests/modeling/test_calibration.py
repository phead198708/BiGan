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


def test_family_aware_calibration_trains_and_loads_per_family_artifact(tmp_path: Path) -> None:
    from bigan.modeling import (
        CalibrationConfig,
        FamilyAwareProbabilityCalibrator,
        family_key_from_feature,
        fit_family_aware_calibration_from_predictions,
        load_probability_calibrator,
        transform_probability,
    )

    output_dir = tmp_path / "family-calibration"
    report = fit_family_aware_calibration_from_predictions(
        y_true=[0, 0, 1, 1, 0, 1, 0, 1],
        y_prob=[0.05, 0.20, 0.80, 0.95, 0.40, 0.62, 0.35, 0.72],
        family_keys=[
            "BTC-15M",
            "BTC-15M",
            "BTC-15M",
            "BTC-15M",
            "ETH-5M",
            "ETH-5M",
            "ETH-5M",
            "ETH-5M",
        ],
        output_dir=output_dir,
        model_version="xgboost-v5",
        config=CalibrationConfig(
            methods=("platt", "isotonic", "temperature", "beta"),
            ece_bins=2,
            platt_epochs=100,
            beta_epochs=100,
            clip_bounds=(0.03, 0.97),
        ),
    )

    calibrator = load_probability_calibrator(output_dir / "calibration.json")

    assert report.selection_metric == "ece"
    assert set(report.family_metrics or {}) == {"BTC-15M", "ETH-5M"}
    assert isinstance(calibrator, FamilyAwareProbabilityCalibrator)
    assert set(calibrator.family_calibrators) == {"BTC-15M", "ETH-5M"}
    assert calibrator.transform(0.999, family_key="BTC-15M") <= 0.97
    assert family_key_from_feature({"canonical_symbol": "ETH-UP-5M"}) == "ETH-5M"
    assert family_key_from_feature({"underlying_id": 1.0, "horizon_minutes": 15.0}) == "BTC-15M"
    assert 0.0 <= transform_probability(
        calibrator,
        0.62,
        feature={"underlying_id": 2.0, "horizon_minutes": 5.0},
    ) <= 1.0


def test_family_calibration_reports_execution_subset_and_clip_grid(tmp_path: Path) -> None:
    from bigan.modeling import (
        CalibrationConfig,
        fit_family_aware_calibration_from_predictions,
        load_probability_calibrator,
    )

    output_dir = tmp_path / "execution-weighted-calibration"
    report = fit_family_aware_calibration_from_predictions(
        y_true=[0, 0, 1, 1, 0, 1, 0, 1],
        y_prob=[0.10, 0.30, 0.58, 0.92, 0.20, 0.64, 0.36, 0.86],
        family_keys=[
            "BTC-15M",
            "BTC-15M",
            "BTC-15M",
            "BTC-15M",
            "ETH-5M",
            "ETH-5M",
            "ETH-5M",
            "ETH-5M",
        ],
        sample_weights=[1.0, 1.0, 3.0, 3.0, 1.0, 3.0, 1.0, 3.0],
        execution_mask=[False, True, True, False, False, True, True, False],
        output_dir=output_dir,
        model_version="xgboost-v5",
        config=CalibrationConfig(
            methods=("platt", "temperature", "beta"),
            ece_bins=2,
            platt_epochs=80,
            beta_epochs=80,
            clip_bounds_grid=((0.05, 0.90), (0.08, 0.95)),
        ),
    )

    calibrator = load_probability_calibrator(output_dir / "calibration.json")
    execution_metrics = report.execution_subset_metrics
    btc_metrics = (report.family_metrics or {})["BTC-15M"]["execution_subset_metrics"]
    artifact = json.loads((output_dir / "calibration.json").read_text(encoding="utf-8"))

    assert execution_metrics is not None
    assert execution_metrics["raw_metrics"]["sample_count"] == 4
    assert execution_metrics["calibrated_metrics"]["ece"] is not None
    assert btc_metrics["raw_metrics"]["sample_count"] == 2
    assert any("@clip=" in name for name in report.candidates)
    assert calibrator.transform(0.99, family_key="BTC-15M") <= 0.95
    assert artifact["family_calibrators"]["BTC-15M"]["params"]["clip_bounds"] in (
        [0.05, 0.9],
        [0.08, 0.95],
    )


def test_calibration_rejects_single_class_labels(tmp_path: Path) -> None:
    from bigan.modeling import fit_calibration_from_predictions

    with pytest.raises(ValueError, match="both positive and negative"):
        fit_calibration_from_predictions(
            y_true=[1, 1, 1],
            y_prob=[0.60, 0.70, 0.80],
            output_dir=tmp_path / "calibration",
            model_version="xgboost-v1",
        )
