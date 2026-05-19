"""Prediction-quality metric tests for issue #11."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.backtest import evaluate_predictions, save_evaluation_report


def test_evaluate_predictions_computes_classification_metrics() -> None:
    report = evaluate_predictions(
        y_true=[0, 0, 1, 1],
        y_prob=[0.1, 0.4, 0.35, 0.8],
        thresholds=[0.3, 0.5],
        calibration_bin_count=2,
    )

    assert report.sample_count == 4
    assert report.positive_count == 2
    assert report.negative_count == 2
    assert report.brier_score == pytest.approx(0.158125)
    assert report.roc_auc == pytest.approx(0.75)
    assert report.pr_auc == pytest.approx((1.0 + 2 / 3) / 2)

    by_threshold = {row.threshold: row for row in report.thresholds}
    assert by_threshold[0.3].true_positive == 2
    assert by_threshold[0.3].false_positive == 1
    assert by_threshold[0.3].true_negative == 1
    assert by_threshold[0.3].false_negative == 0
    assert by_threshold[0.3].precision == pytest.approx(2 / 3)
    assert by_threshold[0.3].recall == pytest.approx(1.0)
    assert by_threshold[0.3].f1 == pytest.approx(0.8)

    assert by_threshold[0.5].true_positive == 1
    assert by_threshold[0.5].false_positive == 0
    assert by_threshold[0.5].true_negative == 2
    assert by_threshold[0.5].false_negative == 1
    assert by_threshold[0.5].accuracy == pytest.approx(0.75)
    assert by_threshold[0.5].precision == pytest.approx(1.0)
    assert by_threshold[0.5].recall == pytest.approx(0.5)
    assert by_threshold[0.5].f1 == pytest.approx(2 / 3)


def test_evaluate_predictions_outputs_calibration_bins() -> None:
    report = evaluate_predictions(
        y_true=[0, 0, 1, 1],
        y_prob=[0.1, 0.4, 0.35, 0.8],
        calibration_bin_count=2,
    )

    low, high = report.calibration_bins
    assert low.bin_start == 0.0
    assert low.bin_end == 0.5
    assert low.count == 3
    assert low.mean_predicted_probability == pytest.approx((0.1 + 0.4 + 0.35) / 3)
    assert low.observed_positive_rate == pytest.approx(1 / 3)
    assert high.bin_start == 0.5
    assert high.bin_end == 1.0
    assert high.count == 1
    assert high.mean_predicted_probability == pytest.approx(0.8)
    assert high.observed_positive_rate == pytest.approx(1.0)


def test_evaluation_report_can_be_saved(tmp_path: Path) -> None:
    report = evaluate_predictions(
        y_true=[False, True],
        y_prob=[0.2, 0.9],
        thresholds=[0.5],
    )
    path = tmp_path / "nested" / "prediction-report.json"

    save_evaluation_report(report, path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["sample_count"] == 2
    assert data["thresholds"][0]["threshold"] == 0.5
    assert data["thresholds"][0]["true_positive"] == 1
    assert data["calibration_bins"]


def test_evaluate_predictions_returns_none_for_undefined_auc() -> None:
    report = evaluate_predictions(y_true=[1, 1], y_prob=[0.6, 0.8])

    assert report.roc_auc is None
    assert report.pr_auc == pytest.approx(1.0)


def test_evaluate_predictions_validates_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        evaluate_predictions(y_true=[0], y_prob=[0.1, 0.2])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        evaluate_predictions(y_true=[0], y_prob=[1.2])
    with pytest.raises(ValueError, match="threshold"):
        evaluate_predictions(y_true=[0], y_prob=[0.2], thresholds=[])
