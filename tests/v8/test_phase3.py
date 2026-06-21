"""Phase 3 differentiable PnL optimization tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.v8.phase0.costs import CostModelConfig
from bigan.v8.phase1 import PolicyPrediction, PolicyTrainShadowSplit
from bigan.v8.phase1.model import XGBoostPolicyModel
from bigan.v8.phase2 import Phase2EvaluationConfig, run_phase2_evaluation
from bigan.v8.phase3 import (
    PHASE3_DIFFERENTIABLE_PNL_PHASE,
    DifferentiableExecutionConfig,
    DifferentiablePnlOptimizationConfig,
    Phase3OptimizationError,
    run_phase3_optimization,
)
from tests.v8.test_phase2 import _accepted_phase15_candidate


def _fake_cost_aware_predictions(
    self: XGBoostPolicyModel,
    examples,
) -> tuple[PolicyPrediction, ...]:
    predictions: list[PolicyPrediction] = []
    for example in examples:
        signal = float(example.features["signal"])
        score = 0.95 if signal > 0.0 else 0.05
        predictions.append(
            PolicyPrediction(
                decision_ts=example.decision_ts,
                source=example.source,
                instrument_id=example.instrument_id,
                action=0.50,
                confidence=0.90,
                regime_embedding=(0.0, 0.0, 0.0, 0.0),
                score=score,
            )
        )
    return tuple(predictions)


def _phase3_config(output_dir: Path | None = None) -> DifferentiablePnlOptimizationConfig:
    return DifferentiablePnlOptimizationConfig(
        execution_config=DifferentiableExecutionConfig(
            cost_model_config=CostModelConfig(
                fee_bps=0.1,
                base_slippage_bps=0.1,
                volatility_slippage_factor=0.0,
                liquidity_impact_factor=0.0,
            ),
            risk_penalty_factor=0.0,
            turnover_penalty_factor=0.0,
        ),
        initial_parameters=(0.0, 1.0, 0.0, 0.0),
        learning_rate=1.0,
        max_steps=80,
        min_sharpe_improvement_ratio_over_phase2=0.0,
        min_oos_sharpe=0.0,
        max_cost_stress_sharpe_drop_ratio=1.0,
        output_dir=output_dir,
        created_at="2026-06-21T00:20:00Z",
    )


def _write_frozen_phase2_report(tmp_path: Path, phase15) -> Path:
    phase3_config = _phase3_config()
    result = run_phase2_evaluation(
        phase15.artifact_dir,
        phase15.split,
        Phase2EvaluationConfig(
            execution_config=phase3_config.execution_config.to_phase2_execution_config(),
            min_sharpe_improvement_ratio=-10.0,
            min_turnover_reduction_ratio=-10.0,
            max_cost_to_abs_gross_return_ratio=100.0,
            output_dir=tmp_path / "phase2",
            created_at="2026-06-21T00:19:00Z",
        ),
    )
    assert result.report.passed
    assert result.report_path is not None
    return result.report_path


def test_phase3_optimizes_differentiable_pnl_and_writes_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase15 = _accepted_phase15_candidate(tmp_path)
    assert phase15.artifact_dir is not None
    monkeypatch.setattr(XGBoostPolicyModel, "predict_examples", _fake_cost_aware_predictions)
    phase2_report_path = _write_frozen_phase2_report(tmp_path, phase15)

    result = run_phase3_optimization(
        phase15.artifact_dir,
        phase15.split,
        _phase3_config(output_dir=tmp_path / "phase3"),
        phase2_report_path=phase2_report_path,
    )

    assert result.passed
    assert result.report.phase == PHASE3_DIFFERENTIABLE_PNL_PHASE
    assert result.report.candidate_run_id == phase15.run_manifest["run_id"]
    assert result.report.phase1_5_hashes["model_sha256"] == (
        phase15.run_manifest["artifacts"]["model_sha256"]
    )
    assert result.report.phase1_5_hashes["dataset_profile_sha256"] == (
        phase15.run_manifest["artifacts"]["dataset_profile_sha256"]
    )
    assert result.report.phase2_baseline_source == "frozen_phase2_report"
    assert result.report.phase2_report_path == str(phase2_report_path)
    assert result.report.phase2_report_sha256 is not None
    assert result.report.phase2_execution_config_sha256 is not None
    assert result.report.phase2_execution_config_verified is True
    assert result.report.acceptance_criteria["frozen_phase2_report_verified"] is True
    assert result.report.acceptance_criteria["phase2_execution_config_verified"] is True
    assert result.report.acceptance_criteria["direct_pnl_optimization"] is True
    assert result.report.acceptance_criteria["gradient_flow_verified"] is True
    assert result.report.acceptance_criteria["optimization_loss_decreased"] is True
    assert result.report.acceptance_criteria["sharpe_improvement_over_phase2"] is True
    assert result.report.acceptance_criteria["cost_perturbation_robust"] is True
    assert result.report.comparison_metrics["phase3_oos_differentiable_sharpe"] > (
        result.report.comparison_metrics["phase2_execution_sharpe"]
    )
    assert result.report.comparison_metrics["mean_net_return_delta_over_phase2"] > 0.0
    assert result.report.optimization_trace[0]["loss"] > (
        result.report.optimization_trace[-1]["loss"]
    )
    assert max(row["gradient_norm"] for row in result.report.optimization_trace) > 0.0
    assert set(result.report.optimized_parameters) == {
        "bias",
        "score_weight",
        "confidence_weight",
        "baseline_action_weight",
    }
    assert set(result.report.cost_stress_metrics) == {"1.2", "1.5", "2"}
    assert result.report_path is not None
    saved = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert saved["phase"] == PHASE3_DIFFERENTIABLE_PNL_PHASE
    assert saved["passed"] is True
    assert saved["phase2_report_sha256"] == result.report.phase2_report_sha256
    assert saved["phase2_execution_config_sha256"] == (
        result.report.phase2_execution_config_sha256
    )


def test_phase3_marks_recomputed_phase2_baseline_as_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase15 = _accepted_phase15_candidate(tmp_path)
    assert phase15.artifact_dir is not None
    monkeypatch.setattr(XGBoostPolicyModel, "predict_examples", _fake_cost_aware_predictions)

    result = run_phase3_optimization(
        phase15.artifact_dir,
        phase15.split,
        _phase3_config(),
    )

    assert result.report.phase2_baseline_source == "diagnostic_recomputed_phase2_baseline"
    assert result.report.phase2_report_path is None
    assert result.report.phase2_report_sha256 is None
    assert result.report.phase2_execution_config_sha256 is not None
    assert result.report.phase2_execution_config_verified is False
    assert result.report.acceptance_criteria["frozen_phase2_report_verified"] is False
    assert result.report.acceptance_criteria["phase2_execution_config_verified"] is False
    assert not result.passed


def test_phase3_rejects_phase2_report_candidate_mismatch_before_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase15 = _accepted_phase15_candidate(tmp_path)
    assert phase15.artifact_dir is not None
    monkeypatch.setattr(XGBoostPolicyModel, "predict_examples", _fake_cost_aware_predictions)
    phase2_report_path = _write_frozen_phase2_report(tmp_path, phase15)
    report = json.loads(phase2_report_path.read_text(encoding="utf-8"))
    report["candidate_run_id"] = "wrong-candidate"
    phase2_report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    def fail_predict(*args, **kwargs):
        raise AssertionError("prediction should not run after Phase 2 report mismatch")

    monkeypatch.setattr(XGBoostPolicyModel, "predict_examples", fail_predict)

    with pytest.raises(Phase3OptimizationError, match="candidate_run_id"):
        run_phase3_optimization(
            phase15.artifact_dir,
            phase15.split,
            _phase3_config(),
            phase2_report_path=phase2_report_path,
        )


def test_phase3_rejects_phase2_execution_config_mismatch_before_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase15 = _accepted_phase15_candidate(tmp_path)
    assert phase15.artifact_dir is not None
    monkeypatch.setattr(XGBoostPolicyModel, "predict_examples", _fake_cost_aware_predictions)
    phase2_report_path = _write_frozen_phase2_report(tmp_path, phase15)
    report = json.loads(phase2_report_path.read_text(encoding="utf-8"))
    report["config"]["execution_config"]["slippage_multiplier"] = 2.0
    phase2_report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    def fail_predict(*args, **kwargs):
        raise AssertionError("prediction should not run after Phase 2 config mismatch")

    monkeypatch.setattr(XGBoostPolicyModel, "predict_examples", fail_predict)

    with pytest.raises(Phase3OptimizationError, match="execution_config"):
        run_phase3_optimization(
            phase15.artifact_dir,
            phase15.split,
            _phase3_config(),
            phase2_report_path=phase2_report_path,
        )


def test_phase3_split_mismatch_fails_before_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase15 = _accepted_phase15_candidate(tmp_path)
    assert phase15.artifact_dir is not None
    bad_split = PolicyTrainShadowSplit(
        train_examples=phase15.split.train_examples,
        shadow_examples=phase15.split.shadow_examples,
        split_ts=phase15.split.split_ts,
        split_hash="wrong-split-hash",
        train_dataset_hash=phase15.split.train_dataset_hash,
        shadow_dataset_hash=phase15.split.shadow_dataset_hash,
    )

    def fail_predict(*args, **kwargs):
        raise AssertionError("prediction should not run after split provenance failure")

    monkeypatch.setattr(XGBoostPolicyModel, "predict_examples", fail_predict)

    with pytest.raises(Phase3OptimizationError, match="split_hash"):
        run_phase3_optimization(
            phase15.artifact_dir,
            bad_split,
            _phase3_config(),
        )


def test_phase3_oos_stability_gate_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase15 = _accepted_phase15_candidate(tmp_path)
    assert phase15.artifact_dir is not None
    monkeypatch.setattr(XGBoostPolicyModel, "predict_examples", _fake_cost_aware_predictions)
    phase2_report_path = _write_frozen_phase2_report(tmp_path, phase15)
    config = DifferentiablePnlOptimizationConfig(
        execution_config=_phase3_config().execution_config,
        initial_parameters=(0.0, 1.0, 0.0, 0.0),
        learning_rate=1.0,
        max_steps=20,
        min_sharpe_improvement_ratio_over_phase2=-10.0,
        min_oos_sharpe=1_000_000.0,
        max_cost_stress_sharpe_drop_ratio=1.0,
        created_at="2026-06-21T00:21:00Z",
    )

    result = run_phase3_optimization(
        phase15.artifact_dir,
        phase15.split,
        config,
        phase2_report_path=phase2_report_path,
    )

    assert result.report.acceptance_criteria["stable_oos_performance"] is False
    assert not result.passed
