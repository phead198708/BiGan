"""TDD contract for issue #19 model promotion rules."""

from __future__ import annotations

import json
from pathlib import Path


def _write_model_run(
    run_dir: Path,
    *,
    model_version: str,
    dataset_version: str,
    test_auc: float,
    test_brier: float,
) -> None:
    run_dir.mkdir(parents=True)
    metrics = {
        "train": {"sample_count": 10, "roc_auc": test_auc, "brier_score": test_brier},
        "val": {"sample_count": 4, "roc_auc": test_auc, "brier_score": test_brier},
        "test": {
            "sample_count": 4,
            "roc_auc": test_auc,
            "brier_score": test_brier,
            "accuracy": 1.0,
        },
    }
    manifest = {
        "model_version": model_version,
        "dataset_version": dataset_version,
        "feature_columns": ["spread", "mid_price", "ret_15m"],
        "metrics": metrics,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _write_calibration(run_dir: Path, *, improved: bool = True) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "calibration_report.json").write_text(
        json.dumps(
            {
                "model_version": "xgboost-v1",
                "method": "isotonic",
                "improved": improved,
                "raw_metrics": {"brier_score": 0.20, "ece": 0.30},
                "calibrated_metrics": {"brier_score": 0.10, "ece": 0.05},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_evaluate_model_promotion_writes_report_and_checklist(tmp_path: Path) -> None:
    from bigan.modeling import PromotionRules, evaluate_model_promotion

    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    calibration_dir = tmp_path / "calibration"
    backtest_path = tmp_path / "backtest_summary.json"
    output_dir = tmp_path / "promotion"
    _write_model_run(
        baseline_dir,
        model_version="logreg-baseline-v1",
        dataset_version="bigan-training-15m-v1.0.0",
        test_auc=0.70,
        test_brier=0.22,
    )
    _write_model_run(
        candidate_dir,
        model_version="xgboost-v1",
        dataset_version="bigan-training-15m-v1.0.0",
        test_auc=0.82,
        test_brier=0.16,
    )
    _write_calibration(calibration_dir, improved=True)
    backtest_path.write_text(
        json.dumps([{"threshold": 0.60, "trade_count": 12, "net_pnl": 1.25}]),
        encoding="utf-8",
    )

    report = evaluate_model_promotion(
        baseline_dir,
        candidate_dir,
        calibration_dir,
        backtest_path,
        output_dir,
        rules=PromotionRules(min_roc_auc_delta=0.02, max_brier_delta=0.0),
    )

    assert report.passed is True
    assert report.decision == "promote"
    assert report.baseline_model_version == "logreg-baseline-v1"
    assert report.candidate_model_version == "xgboost-v1"
    assert report.dataset_version == "bigan-training-15m-v1.0.0"
    assert all(check.passed for check in report.checks)
    assert (output_dir / "promotion_report.json").exists()
    checklist = (output_dir / "promotion_checklist.md").read_text(encoding="utf-8")
    assert "ROC AUC" in checklist
    assert "Brier" in checklist
    assert "xgboost-v1" in checklist


def test_evaluate_model_promotion_rejects_candidate_with_weak_test_metrics(tmp_path: Path) -> None:
    from bigan.modeling import evaluate_model_promotion

    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    calibration_dir = tmp_path / "calibration"
    backtest_path = tmp_path / "backtest_summary.json"
    output_dir = tmp_path / "promotion"
    _write_model_run(
        baseline_dir,
        model_version="logreg-baseline-v1",
        dataset_version="bigan-training-15m-v1.0.0",
        test_auc=0.80,
        test_brier=0.15,
    )
    _write_model_run(
        candidate_dir,
        model_version="xgboost-v1",
        dataset_version="bigan-training-15m-v1.0.0",
        test_auc=0.70,
        test_brier=0.24,
    )
    _write_calibration(calibration_dir, improved=True)
    backtest_path.write_text(
        json.dumps([{"threshold": 0.60, "trade_count": 12, "net_pnl": 1.25}]),
        encoding="utf-8",
    )

    report = evaluate_model_promotion(
        baseline_dir,
        candidate_dir,
        calibration_dir,
        backtest_path,
        output_dir,
    )

    assert report.passed is False
    assert report.decision == "reject"
    failed = {check.name for check in report.checks if not check.passed}
    assert {"test_roc_auc_vs_baseline", "test_brier_vs_baseline"} <= failed
