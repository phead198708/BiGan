"""First-champion bootstrap decision contracts."""

from __future__ import annotations

import json
from pathlib import Path


def _write_model_run(
    run_dir: Path,
    *,
    model_version: str,
    test_auc: float,
    test_brier: float,
    test_pr_auc: float = 0.55,
) -> None:
    run_dir.mkdir(parents=True)
    metrics = {
        "test": {
            "sample_count": 100,
            "roc_auc": test_auc,
            "pr_auc": test_pr_auc,
            "brier_score": test_brier,
            "accuracy": 0.57,
        }
    }
    manifest = {
        "model_version": model_version,
        "dataset_version": "bigan-training-15m-v1.0.0",
        "feature_columns": ["spread", "mid_price", "ret_15m"],
        "best_params": {"max_depth": 2, "rounds": 20},
        "metrics": metrics,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "model.json").write_text(json.dumps({"model_version": model_version}), encoding="utf-8")


def _write_calibration(run_dir: Path, *, improved: bool = True) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "calibration_report.json").write_text(
        json.dumps(
            {
                "model_version": "xgboost-v1",
                "method": "platt",
                "improved": improved,
                "raw_metrics": {"brier_score": 0.24, "ece": 0.08},
                "calibrated_metrics": {"brier_score": 0.21, "ece": 0.03},
            }
        ),
        encoding="utf-8",
    )


def _write_backtest(path: Path, *, net_pnl: float, fee_bps: float = 2.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [
                {
                    "threshold": 0.60,
                    "trade_count": 25,
                    "net_pnl": net_pnl,
                    "max_drawdown": 0.05,
                    "sharpe": 1.2,
                    "turnover": 0.4,
                    "settings": {"fee_bps": fee_bps, "slippage_bps": 1.0},
                }
            ]
        ),
        encoding="utf-8",
    )


def _write_serving(path: Path, *, status: str = "ok") -> None:
    path.write_text(
        json.dumps(
            {
                "status": status,
                "p95_latency_ms": 4.5,
                "latency_sla_ms": 50.0,
                "error_rate": 0.0,
                "max_error_rate": 0.01,
            }
        ),
        encoding="utf-8",
    )


def _write_schema(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "feature_columns": ["spread", "mid_price", "ret_15m"],
                "feature_types": {
                    "spread": "float64",
                    "mid_price": "float64",
                    "ret_15m": "float64",
                },
                "schema_hash": "hash-ok",
                "model_version": "xgboost-v1",
            }
        ),
        encoding="utf-8",
    )


def test_bootstrap_promotes_first_champion_when_all_hard_gates_pass(tmp_path: Path) -> None:
    from bigan.modeling import BootstrapCandidateInput, evaluate_bootstrap_champion

    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    calibration_dir = tmp_path / "calibration"
    serving_path = tmp_path / "serving.json"
    baseline_backtest = tmp_path / "baseline-backtest.json"
    candidate_backtest = tmp_path / "candidate-backtest.json"
    runbook = tmp_path / "rollback.md"
    _write_model_run(baseline_dir, model_version="logreg-baseline-v1", test_auc=0.55, test_brier=0.24)
    _write_model_run(candidate_dir, model_version="xgboost-v1", test_auc=0.59, test_brier=0.22)
    _write_calibration(calibration_dir)
    _write_backtest(baseline_backtest, net_pnl=0.10)
    _write_backtest(candidate_backtest, net_pnl=0.80)
    _write_serving(serving_path)
    _write_schema(candidate_dir / "feature_schema.json")
    runbook.write_text("# Rollback\n", encoding="utf-8")

    report = evaluate_bootstrap_champion(
        baseline_dir=baseline_dir,
        baseline_backtest_summary_path=baseline_backtest,
        candidates=(
            BootstrapCandidateInput(
                candidate_dir=candidate_dir,
                calibration_dir=calibration_dir,
                candidate_backtest_summary_path=candidate_backtest,
                serving_readiness_path=serving_path,
            ),
        ),
        rollback_runbook_path=runbook,
        output_dir=tmp_path / "decision",
    )

    assert report.recommended_action == "PROMOTE_FIRST_CHAMPION:xgboost-v1"
    assert report.confidence_level == "HIGH"
    assert report.promotion_checklist.backtest_acceptable is True
    assert all(gate.passed for gate in report.hard_gate_results)
    markdown = (tmp_path / "decision" / "bootstrap_decision.md").read_text(encoding="utf-8")
    assert markdown.startswith("# Bootstrap Champion Decision")
    assert "PROMOTE_FIRST_CHAMPION:xgboost-v1" in markdown


def test_bootstrap_keeps_baseline_when_candidate_backtest_is_unacceptable(
    tmp_path: Path,
) -> None:
    from bigan.modeling import BootstrapCandidateInput, evaluate_bootstrap_champion

    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    calibration_dir = tmp_path / "calibration"
    candidate_backtest = tmp_path / "candidate-backtest.json"
    runbook = tmp_path / "rollback.md"
    _write_model_run(baseline_dir, model_version="logreg-baseline-v1", test_auc=0.50, test_brier=0.246)
    _write_model_run(candidate_dir, model_version="xgboost-v1", test_auc=0.54, test_brier=0.245)
    _write_calibration(calibration_dir)
    _write_backtest(candidate_backtest, net_pnl=-0.55, fee_bps=0.0)
    runbook.write_text("# Rollback\n", encoding="utf-8")

    report = evaluate_bootstrap_champion(
        baseline_dir=baseline_dir,
        candidates=(
            BootstrapCandidateInput(
                candidate_dir=candidate_dir,
                calibration_dir=calibration_dir,
                candidate_backtest_summary_path=candidate_backtest,
            ),
        ),
        rollback_runbook_path=runbook,
        output_dir=tmp_path / "decision",
    )

    assert report.recommended_action == "KEEP_BASELINE_TEMPORARILY"
    assert report.promotion_checklist.backtest_acceptable is False
    assert report.promotion_checklist.serving_readiness_acceptable is False
    assert report.promotion_checklist.schema_stable is False
    assert not report.hard_gate_results[0].passed
    assert "Serving latency/error readiness report missing" in report.missing_or_weak_evidence


def test_bootstrap_continues_experimentation_when_promising_candidate_is_incomplete(
    tmp_path: Path,
) -> None:
    from bigan.modeling import BootstrapCandidateInput, evaluate_bootstrap_champion

    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    calibration_dir = tmp_path / "calibration"
    runbook = tmp_path / "rollback.md"
    _write_model_run(baseline_dir, model_version="logreg-baseline-v1", test_auc=0.50, test_brier=0.246)
    _write_model_run(candidate_dir, model_version="xgboost-v1", test_auc=0.54, test_brier=0.245)
    _write_calibration(calibration_dir)
    runbook.write_text("# Rollback\n", encoding="utf-8")

    report = evaluate_bootstrap_champion(
        baseline_dir=baseline_dir,
        candidates=(
            BootstrapCandidateInput(
                candidate_dir=candidate_dir,
                calibration_dir=calibration_dir,
            ),
        ),
        rollback_runbook_path=runbook,
        output_dir=tmp_path / "decision",
    )

    assert report.recommended_action == "CONTINUE_BOOTSTRAP_EXPERIMENTATION"
    assert report.confidence_level == "MEDIUM"
    assert "Candidate cost-adjusted backtest summary missing" in report.missing_or_weak_evidence
