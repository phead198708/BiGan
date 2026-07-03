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
    best_params: dict | None = None,
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
        "best_params": {"max_depth": 2, "rounds": 20} if best_params is None else best_params,
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


def _write_backtest(
    path: Path,
    *,
    net_pnl: float,
    fee_bps: float = 2.0,
    sharpe: float = 1.2,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [
                {
                    "threshold": 0.60,
                    "trade_count": 25,
                    "net_pnl": net_pnl,
                    "max_drawdown": 0.05,
                    "sharpe": sharpe,
                    "sharpe_ratio": sharpe,
                    "sortino_ratio": 1.5,
                    "turnover": 0.4,
                    "concentration": {
                        "top1_abs_net_pnl_share": 0.35,
                        "top5_abs_net_pnl_share": 0.70,
                    },
                    "top1_market_abs_net_pnl_share": 0.35,
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
    assert report.artifact_paths["baseline_eval_dir"] == str(baseline_dir)
    assert report.artifact_paths["candidate_eval_dir"] == str(candidate_dir)
    assert report.artifact_paths["candidate_backtest_summary_path"] == str(candidate_backtest)
    assert report.artifact_paths["serving_readiness_path"] == str(serving_path)
    assert report.artifact_paths["rollback_runbook_path"] == str(runbook)
    saved = json.loads((tmp_path / "decision" / "bootstrap_decision.json").read_text(encoding="utf-8"))
    assert saved["artifact_paths"]["candidate_dir"] == str(candidate_dir)
    assert saved["artifact_paths"]["candidate_eval_dir"] == str(candidate_dir)
    markdown = (tmp_path / "decision" / "bootstrap_decision.md").read_text(encoding="utf-8")
    assert markdown.startswith("# Bootstrap Champion Decision")
    assert "PROMOTE_FIRST_CHAMPION:xgboost-v1" in markdown
    assert "delta_vs_baseline 0.7000" in markdown


def test_bootstrap_can_emit_replacement_champion_action(tmp_path: Path) -> None:
    from bigan.modeling import BootstrapCandidateInput, evaluate_bootstrap_champion

    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    calibration_dir = tmp_path / "calibration"
    serving_path = tmp_path / "serving.json"
    baseline_backtest = tmp_path / "baseline-backtest.json"
    candidate_backtest = tmp_path / "candidate-backtest.json"
    runbook = tmp_path / "rollback.md"
    _write_model_run(baseline_dir, model_version="xgboost-v3", test_auc=0.55, test_brier=0.24)
    _write_model_run(candidate_dir, model_version="xgboost-v4", test_auc=0.59, test_brier=0.22)
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
        promotion_action="replace_champion",
    )

    assert report.recommended_action == "PROMOTE_CHAMPION"
    assert report.hard_gate_results[0].model_version == "xgboost-v4"
    saved = json.loads((tmp_path / "decision" / "bootstrap_decision.json").read_text(encoding="utf-8"))
    assert saved["recommended_action"] == "PROMOTE_CHAMPION"
    assert saved["hard_gate_results"][0]["model_version"] == "xgboost-v4"


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


def test_bootstrap_uses_model_card_as_complexity_evidence(tmp_path: Path) -> None:
    from bigan.modeling import BootstrapCandidateInput, evaluate_bootstrap_champion

    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    calibration_dir = tmp_path / "calibration"
    serving_path = tmp_path / "serving.json"
    baseline_backtest = tmp_path / "baseline-backtest.json"
    candidate_backtest = tmp_path / "candidate-backtest.json"
    model_card = tmp_path / "xgboost-v3.md"
    runbook = tmp_path / "rollback.md"
    _write_model_run(baseline_dir, model_version="logreg-baseline-v1", test_auc=0.55, test_brier=0.24)
    _write_model_run(
        candidate_dir,
        model_version="xgboost-v3",
        test_auc=0.75,
        test_brier=0.18,
        best_params={"max_depth": 4, "rounds": 200},
    )
    _write_calibration(calibration_dir)
    _write_backtest(baseline_backtest, net_pnl=0.10)
    _write_backtest(candidate_backtest, net_pnl=0.80)
    _write_serving(serving_path)
    _write_schema(candidate_dir / "feature_schema.json")
    model_card.write_text(
        "\n".join(
            [
                "# XGBoost-v3 Model Card",
                "Dependencies: xgboost and pyarrow.",
                "Training cost: local CPU retraining time is documented.",
                "Retraining: rerun the fixed dataset pipeline.",
                "Interpretability: feature importance and contribution examples are reviewed.",
                "Feature stability: feature_schema.json is the online contract.",
                "Monitoring: watch probability drift, label shift, PnL, and latency.",
            ]
        ),
        encoding="utf-8",
    )
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
                model_complexity_notes_path=model_card,
            ),
        ),
        rollback_runbook_path=runbook,
        output_dir=tmp_path / "decision",
    )

    assert report.promotion_checklist.simple_enough is True
    assert "Model complexity notes missing" not in report.missing_or_weak_evidence
    assert "Model card complexity notes present" in report.comparison_rows[1].simplicity


def test_bootstrap_allows_lower_sharpe_when_brier_gap_is_material(
    tmp_path: Path,
) -> None:
    from bigan.modeling import BootstrapCandidateInput, evaluate_bootstrap_champion

    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    calibration_dir = tmp_path / "calibration"
    serving_path = tmp_path / "serving.json"
    baseline_backtest = tmp_path / "baseline-backtest.json"
    candidate_backtest = tmp_path / "candidate-backtest.json"
    runbook = tmp_path / "rollback.md"
    _write_model_run(baseline_dir, model_version="logreg-baseline-v1", test_auc=0.55, test_brier=0.24)
    _write_model_run(candidate_dir, model_version="xgboost-v3", test_auc=0.75, test_brier=0.18)
    _write_calibration(calibration_dir)
    _write_backtest(baseline_backtest, net_pnl=0.10, sharpe=1.20)
    _write_backtest(candidate_backtest, net_pnl=0.80, sharpe=0.30)
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

    assert report.promotion_checklist.backtest_acceptable is True
    assert report.recommended_action == "PROMOTE_FIRST_CHAMPION:xgboost-v3"
    assert "lower_sharpe_allowed_brier_gap 0.0600" in report.comparison_rows[1].backtest


def test_bootstrap_uses_shadow_evaluation_as_promotion_precondition(
    tmp_path: Path,
) -> None:
    from bigan.modeling import BootstrapCandidateInput, evaluate_bootstrap_champion

    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    calibration_dir = tmp_path / "calibration"
    serving_path = tmp_path / "serving.json"
    baseline_backtest = tmp_path / "baseline-backtest.json"
    candidate_backtest = tmp_path / "candidate-backtest.json"
    shadow_evaluation = tmp_path / "shadow-evaluation.json"
    runbook = tmp_path / "rollback.md"
    _write_model_run(baseline_dir, model_version="logreg-baseline-v1", test_auc=0.55, test_brier=0.24)
    _write_model_run(candidate_dir, model_version="xgboost-v3", test_auc=0.75, test_brier=0.18)
    _write_calibration(calibration_dir)
    _write_backtest(baseline_backtest, net_pnl=0.10, sharpe=1.20)
    _write_backtest(candidate_backtest, net_pnl=0.80, sharpe=1.30)
    _write_serving(serving_path)
    _write_schema(candidate_dir / "feature_schema.json")
    shadow_evaluation.write_text(
        json.dumps(
            {
                "overall_passed": False,
                "challenger_model_version": "xgboost-v3",
                "challenger_edge_trigger_rate": 0.0,
                "schema_error_rate": 0.0,
                "latency_ms": {"xgboost-v3": {"p95": 0.4}},
                "checks": {
                    "edge_trigger_rate": {
                        "passed": False,
                        "detail": "edge trigger rate is zero",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
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
                shadow_evaluation_path=shadow_evaluation,
            ),
        ),
        rollback_runbook_path=runbook,
        output_dir=tmp_path / "decision",
    )

    assert report.recommended_action == "KEEP_BASELINE_TEMPORARILY"
    assert report.promotion_checklist.serving_readiness_acceptable is False
    assert "shadow FAIL" in report.comparison_rows[1].production_readiness
    assert any("Shadow evaluation failed" in risk for risk in report.risks)


def test_bootstrap_rejects_failed_bucket_level_calibration_gate(
    tmp_path: Path,
) -> None:
    from bigan.modeling import (
        BootstrapCandidateInput,
        BootstrapRules,
        evaluate_bootstrap_champion,
    )

    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    calibration_dir = tmp_path / "calibration"
    serving_path = tmp_path / "serving.json"
    baseline_backtest = tmp_path / "baseline-backtest.json"
    candidate_backtest = tmp_path / "candidate-backtest.json"
    runbook = tmp_path / "rollback.md"
    _write_model_run(baseline_dir, model_version="xgboost-v4", test_auc=0.55, test_brier=0.24)
    _write_model_run(candidate_dir, model_version="xgboost-v5", test_auc=0.75, test_brier=0.18)
    calibration_dir.mkdir(parents=True)
    (calibration_dir / "calibration_report.json").write_text(
        json.dumps(
            {
                "model_version": "xgboost-v5",
                "method": "family_aware",
                "improved": True,
                "raw_metrics": {"brier_score": 0.24, "ece": 0.20},
                "calibrated_metrics": {"brier_score": 0.18, "ece": 0.40},
                "bucket_metrics": {
                    "high_up": {"realized_up_rate": 0.58},
                    "high_down": {"realized_up_rate": 0.53},
                },
                "family_metrics": {
                    "BTC-15M": {"avg_realized_return": 0.01},
                    "ETH-5M": {"avg_realized_return": 0.02},
                },
            }
        ),
        encoding="utf-8",
    )
    _write_backtest(baseline_backtest, net_pnl=0.10, sharpe=1.20)
    _write_backtest(candidate_backtest, net_pnl=0.80, sharpe=1.30)
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
        promotion_action="replace_champion",
        rules=BootstrapRules(
            max_global_ece=0.4784,
            min_high_up_realized_up_rate=0.55,
            min_high_down_realized_down_rate=0.55,
            require_positive_avg_return_by_family=True,
        ),
    )

    assert report.recommended_action == "KEEP_BASELINE_TEMPORARILY"
    assert report.promotion_checklist.calibration_acceptable is False
    assert any("high_down realized down rate" in risk for risk in report.risks)


def test_bootstrap_rejects_failed_execution_subset_calibration_gate(
    tmp_path: Path,
) -> None:
    from bigan.modeling import (
        BootstrapCandidateInput,
        BootstrapRules,
        evaluate_bootstrap_champion,
    )

    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    calibration_dir = tmp_path / "calibration"
    serving_path = tmp_path / "serving.json"
    baseline_backtest = tmp_path / "baseline-backtest.json"
    candidate_backtest = tmp_path / "candidate-backtest.json"
    runbook = tmp_path / "rollback.md"
    _write_model_run(baseline_dir, model_version="xgboost-v4", test_auc=0.55, test_brier=0.24)
    _write_model_run(candidate_dir, model_version="xgboost-v5", test_auc=0.75, test_brier=0.18)
    calibration_dir.mkdir(parents=True)
    (calibration_dir / "calibration_report.json").write_text(
        json.dumps(
            {
                "model_version": "xgboost-v5",
                "method": "family_aware",
                "improved": True,
                "raw_metrics": {"brier_score": 0.24, "ece": 0.20},
                "calibrated_metrics": {"brier_score": 0.18, "ece": 0.04},
                "execution_subset_metrics": {
                    "raw_metrics": {"brier_score": 0.30, "ece": 0.35},
                    "calibrated_metrics": {"brier_score": 0.22, "ece": 0.12},
                },
            }
        ),
        encoding="utf-8",
    )
    _write_backtest(baseline_backtest, net_pnl=0.10, sharpe=1.20)
    _write_backtest(candidate_backtest, net_pnl=0.80, sharpe=1.30)
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
        promotion_action="replace_champion",
        rules=BootstrapRules(max_execution_subset_ece=0.08),
    )

    assert report.recommended_action == "KEEP_BASELINE_TEMPORARILY"
    assert report.promotion_checklist.calibration_acceptable is False
    assert any("Execution subset ECE" in risk for risk in report.risks)


def test_bootstrap_rejects_lower_sharpe_when_brier_gap_is_small(
    tmp_path: Path,
) -> None:
    from bigan.modeling import BootstrapCandidateInput, evaluate_bootstrap_champion

    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    calibration_dir = tmp_path / "calibration"
    serving_path = tmp_path / "serving.json"
    baseline_backtest = tmp_path / "baseline-backtest.json"
    candidate_backtest = tmp_path / "candidate-backtest.json"
    runbook = tmp_path / "rollback.md"
    _write_model_run(baseline_dir, model_version="logreg-baseline-v1", test_auc=0.55, test_brier=0.24)
    _write_model_run(candidate_dir, model_version="xgboost-v3", test_auc=0.75, test_brier=0.22)
    _write_calibration(calibration_dir)
    _write_backtest(baseline_backtest, net_pnl=0.10, sharpe=1.20)
    _write_backtest(candidate_backtest, net_pnl=0.80, sharpe=0.30)
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

    assert report.promotion_checklist.backtest_acceptable is False
    assert report.recommended_action == "KEEP_BASELINE_TEMPORARILY"
    assert any("Sharpe underperforms" in risk for risk in report.risks)
