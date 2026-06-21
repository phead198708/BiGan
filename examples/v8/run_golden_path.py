"""Deterministic v8 Phase 0-6 golden-path dry run.

This example intentionally uses only synthetic fixture data. It never connects
to exchanges, brokers, live feeds, or strategy-discovery services.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bigan.v8.phase0 import (  # noqa: E402
    CausalFeatureBuilderConfig,
    CostAwareLabelBuilderConfig,
    CostModelConfig,
    DatasetContract,
    Phase0Dataset,
    Phase0Pipeline,
    Phase0PipelineConfig,
    ValidationConfig,
    assert_phase0_artifact_ready,
)
from bigan.v8.phase1 import (  # noqa: E402
    PolicyAcceptanceConfig,
    PolicyDatasetConfig,
    PolicyTrainingRunConfig,
    PolicyTrainingRunResult,
    XGBoostPolicyConfig,
    run_policy_training,
)
from bigan.v8.phase2 import (  # noqa: E402
    Phase2EvaluationConfig,
    Phase2EvaluationResult,
    run_phase2_evaluation,
)
from bigan.v8.phase3 import (  # noqa: E402
    DifferentiableExecutionConfig,
    DifferentiablePnlOptimizationConfig,
    Phase3OptimizationResult,
    run_phase3_optimization,
)
from bigan.v8.phase4 import (  # noqa: E402
    ExecutionAdaptationConfig,
    LambdaControllerConfig,
    Phase4AdaptiveSystemConfig,
    Phase4AdaptiveSystemResult,
    RegimeDetectorConfig,
    build_phase4_input_provenance,
    run_phase4_adaptive_system,
)
from bigan.v8.phase5 import (  # noqa: E402
    LiveExecutionObservation,
    Phase5SafetyLayerResult,
    SafetyLayerConfig,
    StableModelSnapshot,
    compute_safe_parameters_sha256,
    run_phase5_safety_layer,
)
from bigan.v8.phase6 import (  # noqa: E402
    CICDPipelineConfig,
    CICDPipelineResult,
    CICDStageEvidence,
    RollbackPlan,
    compute_phase6_stage_evidence_sha256,
    run_phase6_cicd_pipeline,
)

MINUTE_MS = 60_000
DEFAULT_RUN_ID = "golden_path_synthetic_v1"
FIXED_CREATED_AT = "2026-06-22T00:00:00Z"
SYNTHETIC_FIXTURE_ID = "v8-golden-path-synthetic-market-v1"
POLICY_FEATURE_COLUMNS = (
    "return_1m",
    "return_5m",
    "return_15m",
    "volatility_5m",
    "volatility_15m",
    "spread_bps",
    "liquidity_depth",
    "volume_1m",
    "trade_count_1m",
)


@dataclass(frozen=True, slots=True)
class GoldenPathResult:
    """In-memory handles and manifest paths from one golden-path dry run."""

    run_id: str
    bundle_dir: Path
    phase0_dataset: Phase0Dataset
    phase0_contract: DatasetContract
    phase1_5_result: PolicyTrainingRunResult
    phase2_result: Phase2EvaluationResult
    phase3_result: Phase3OptimizationResult
    phase4_result: Phase4AdaptiveSystemResult
    phase5_result: Phase5SafetyLayerResult
    phase6_result: CICDPipelineResult
    bundle_manifest_path: Path
    bundle_manifest: dict[str, Any]


def run_golden_path(
    output_dir: Path | str,
    *,
    run_id: str = DEFAULT_RUN_ID,
    overwrite: bool = True,
) -> GoldenPathResult:
    """Run a deterministic synthetic Phase 0-6 dry run and write an artifact bundle."""

    if not run_id.strip():
        raise ValueError("run_id is required")
    output_root = Path(output_dir).expanduser().resolve()
    bundle_dir = output_root / run_id
    if bundle_dir.exists():
        if not overwrite:
            raise FileExistsError(f"artifact bundle already exists: {bundle_dir}")
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)

    with _working_directory(bundle_dir):
        synthetic_rows = _synthetic_market_rows()
        phase0_dir = Path("phase0")
        _write_jsonl(phase0_dir / "synthetic_market_data.jsonl", synthetic_rows)

        cost_config = _cost_config()
        phase0_dataset = Phase0Pipeline(_phase0_config(cost_config)).build(
            synthetic_rows,
            output_dir=phase0_dir,
        )
        _write_json(phase0_dir / "validation_report.json", phase0_dataset.validation_report.to_dict())
        phase0_contract = assert_phase0_artifact_ready(phase0_dataset.manifest)
        _write_json(
            phase0_dir / "artifact_gate_report.json",
            {
                "passed": True,
                "failures": [],
                "contract": phase0_contract.to_dict(),
            },
        )

        phase1_5_result = run_policy_training(
            phase0_dataset,
            _phase1_5_config(output_dir=Path("phase1_5"), run_id=f"{run_id}_candidate"),
        )
        if phase1_5_result.artifact_dir is None:
            raise AssertionError("Phase 1.5 did not write a candidate artifact directory")

        execution_config = _phase3_execution_config(cost_config)
        phase2_result = run_phase2_evaluation(
            phase1_5_result.artifact_dir,
            phase1_5_result.split,
            _phase2_config(execution_config, output_dir=Path("phase2")),
        )
        if phase2_result.report_path is None:
            raise AssertionError("Phase 2 did not write a report")

        phase3_result = run_phase3_optimization(
            phase1_5_result.artifact_dir,
            phase1_5_result.split,
            _phase3_config(execution_config, output_dir=Path("phase3")),
            phase2_report_path=phase2_result.report_path,
        )
        if phase3_result.report_path is None:
            raise AssertionError("Phase 3 did not write a report")

        identity = _candidate_identity(phase1_5_result)
        phase4_provenance = build_phase4_input_provenance(
            examples=phase1_5_result.split.shadow_examples,
            predictions=phase3_result.oos_predictions,
            candidate_run_id=identity["candidate_run_id"],
            policy_dataset_hash=identity["policy_dataset_hash"],
            split_hash=identity["split_hash"],
            model_sha256=identity["model_sha256"],
            phase2_report_sha256=_file_sha256(phase2_result.report_path),
            phase3_report_sha256=_file_sha256(phase3_result.report_path),
        )
        phase4_dir = Path("phase4")
        _write_json(phase4_dir / "input_provenance.json", phase4_provenance.to_dict())
        phase4_result = run_phase4_adaptive_system(
            examples=phase1_5_result.split.shadow_examples,
            predictions=phase3_result.oos_predictions,
            provenance=phase4_provenance,
            config=_phase4_config(cost_config, output_dir=phase4_dir),
        )
        if phase4_result.report_path is None:
            raise AssertionError("Phase 4 did not write a report")

        safe_parameters = {"max_position_size": 0.10, "risk_mode": "safe"}
        stable_model = StableModelSnapshot(
            model_id=f"{identity['candidate_run_id']}-stable",
            model_sha256=identity["model_sha256"],
            policy_dataset_hash=identity["policy_dataset_hash"],
            split_hash=identity["split_hash"],
            safe_parameter_sha256=compute_safe_parameters_sha256(safe_parameters),
            safe_parameters=safe_parameters,
        )
        phase5_result = run_phase5_safety_layer(
            shadow_decisions=phase4_result.decisions,
            live_observations=_mirrored_live_observations(phase4_result),
            stable_model=stable_model,
            config=_phase5_config(output_dir=Path("phase5")),
        )
        if phase5_result.report_path is None:
            raise AssertionError("Phase 5 did not write a report")

        stage_evidence = _phase6_stage_evidence(
            identity=identity,
            phase1_5_result=phase1_5_result,
            phase2_result=phase2_result,
            phase3_result=phase3_result,
            phase5_result=phase5_result,
        )
        rollback_plan = RollbackPlan(
            stable_model_id=stable_model.model_id,
            stable_model_sha256=stable_model.model_sha256,
            safe_parameter_sha256=stable_model.safe_parameter_sha256,
            safe_parameters=safe_parameters,
            rollback_artifact_sha256=_file_sha256(phase5_result.report_path),
            latency_measurements_ms=(75, 92, 88),
        )
        phase6_dir = Path("phase6")
        _write_json(
            phase6_dir / "stage_evidence.json",
            {
                "stage_evidence_sha256": compute_phase6_stage_evidence_sha256(stage_evidence),
                "stage_evidence": [evidence.to_dict() for evidence in stage_evidence],
            },
        )
        phase6_result = run_phase6_cicd_pipeline(
            candidate_run_id=identity["candidate_run_id"],
            stage_evidence=stage_evidence,
            rollback_plan=rollback_plan,
            config=_phase6_config(output_dir=phase6_dir),
        )
        if phase6_result.report_path is None:
            raise AssertionError("Phase 6 did not write a report")
        _write_json(phase6_dir / "release_manifest.json", phase6_result.report.release_manifest)

        _assert_hard_gates(
            identity=identity,
            phase0_dataset=phase0_dataset,
            phase0_contract=phase0_contract,
            phase1_5_result=phase1_5_result,
            phase2_result=phase2_result,
            phase3_result=phase3_result,
            phase4_result=phase4_result,
            phase5_result=phase5_result,
            phase6_result=phase6_result,
            stage_evidence=stage_evidence,
        )
        bundle_manifest = _bundle_manifest(
            run_id=run_id,
            identity=identity,
            phase0_dataset=phase0_dataset,
            phase0_contract=phase0_contract,
            phase1_5_result=phase1_5_result,
            phase2_result=phase2_result,
            phase3_result=phase3_result,
            phase4_result=phase4_result,
            phase5_result=phase5_result,
            phase6_result=phase6_result,
        )
        bundle_manifest_path = Path("bundle_manifest.json")
        _write_json(bundle_manifest_path, bundle_manifest)

    return GoldenPathResult(
        run_id=run_id,
        bundle_dir=bundle_dir,
        phase0_dataset=phase0_dataset,
        phase0_contract=phase0_contract,
        phase1_5_result=phase1_5_result,
        phase2_result=phase2_result,
        phase3_result=phase3_result,
        phase4_result=phase4_result,
        phase5_result=phase5_result,
        phase6_result=phase6_result,
        bundle_manifest_path=bundle_dir / "bundle_manifest.json",
        bundle_manifest=bundle_manifest,
    )


def _synthetic_market_rows(row_count: int = 241) -> list[dict[str, Any]]:
    start_ts = int(datetime(2026, 6, 1, 12, 0, tzinfo=UTC).timestamp() * 1000)
    price = 100.0
    rows: list[dict[str, Any]] = []
    for index in range(row_count):
        block = (index // 30) % 4
        drift = (0.0020, -0.0018, 0.0017, -0.0015)[block]
        deterministic_wave = (
            0.00020 * math.sin(index * 0.71)
            + 0.00010 * math.cos(index * 0.17)
        )
        if index > 0:
            price *= 1.0 + drift + deterministic_wave
        spread = 0.018 + 0.002 * (index % 5)
        ts = start_ts + index * MINUTE_MS
        rows.append(
            {
                "ts": ts,
                "available_at_ts": ts,
                "source": "synthetic",
                "instrument_id": "btc-updown-15m:UP",
                "bid_price": price - spread / 2.0,
                "ask_price": price + spread / 2.0,
                "volume": 100.0 + (index % 7) * 5.0,
                "trade_count": 10 + (index % 5),
                "bid_size": 400.0 + (index % 11) * 20.0,
                "ask_size": 420.0 + (index % 13) * 18.0,
                "timeframe_ms": MINUTE_MS,
                "sequence": index,
            }
        )
    return rows


def _cost_config() -> CostModelConfig:
    return CostModelConfig(
        fee_bps=0.1,
        base_slippage_bps=0.1,
        volatility_slippage_factor=0.0,
        liquidity_impact_factor=0.0,
    )


def _phase0_config(cost_config: CostModelConfig) -> Phase0PipelineConfig:
    return Phase0PipelineConfig(
        feature_config=CausalFeatureBuilderConfig(decision_frequency_ms=MINUTE_MS),
        label_config=CostAwareLabelBuilderConfig(
            horizons_ms=(MINUTE_MS,),
            order_size=0.1,
        ),
        cost_config=cost_config,
        validation_config=ValidationConfig(
            max_abs_feature_future_corr=1.0,
            min_correlation_rows=30,
            statistical_integrity_mode="warn",
        ),
    )


def _phase1_5_config(
    *,
    output_dir: Path,
    run_id: str,
) -> PolicyTrainingRunConfig:
    return PolicyTrainingRunConfig(
        policy_dataset_config=PolicyDatasetConfig(
            horizon_ms=MINUTE_MS,
            feature_columns=POLICY_FEATURE_COLUMNS,
        ),
        xgboost_config=XGBoostPolicyConfig(
            num_boost_round=16,
            max_depth=2,
            learning_rate=0.2,
            action_activation_threshold=0.50,
            seed=17,
        ),
        acceptance_config=PolicyAcceptanceConfig(
            min_shadow_sharpe=-10.0,
            min_active_rate=0.0,
            max_active_rate=1.0,
            min_action_std=0.0,
            max_dominant_bucket_ratio=1.0,
            min_active_regime_count=1,
            max_active_regime_ratio=1.0,
            min_non_empty_buckets=2,
        ),
        train_fraction=0.65,
        output_dir=output_dir,
        run_id=run_id,
        created_at=FIXED_CREATED_AT,
        overwrite_existing=True,
    )


def _phase3_execution_config(cost_config: CostModelConfig) -> DifferentiableExecutionConfig:
    return DifferentiableExecutionConfig(
        cost_model_config=cost_config,
        risk_penalty_factor=0.0,
        turnover_penalty_factor=0.0,
    )


def _phase2_config(
    execution_config: DifferentiableExecutionConfig,
    *,
    output_dir: Path,
) -> Phase2EvaluationConfig:
    return Phase2EvaluationConfig(
        execution_config=execution_config.to_phase2_execution_config(),
        min_sharpe_improvement_ratio=-10.0,
        min_turnover_reduction_ratio=-10.0,
        max_cost_to_abs_gross_return_ratio=1000.0,
        output_dir=output_dir,
        created_at="2026-06-22T00:05:00Z",
    )


def _phase3_config(
    execution_config: DifferentiableExecutionConfig,
    *,
    output_dir: Path,
) -> DifferentiablePnlOptimizationConfig:
    return DifferentiablePnlOptimizationConfig(
        execution_config=execution_config,
        initial_parameters=(-5.0, 10.0, 0.0, 1.0),
        learning_rate=0.25,
        max_steps=20,
        min_loss_improvement=0.0,
        min_sharpe_improvement_ratio_over_phase2=-10.0,
        min_oos_sharpe=-10.0,
        max_cost_stress_sharpe_drop_ratio=1.0,
        output_dir=output_dir,
        created_at="2026-06-22T00:10:00Z",
    )


def _phase4_config(
    cost_config: CostModelConfig,
    *,
    output_dir: Path,
) -> Phase4AdaptiveSystemConfig:
    return Phase4AdaptiveSystemConfig(
        detector_config=RegimeDetectorConfig(
            trend_score_threshold=0.50,
            high_volatility_threshold=0.060,
            liquidity_stress_threshold=15.0,
            high_cost_spread_bps_threshold=60.0,
            transition_confirmation_count=2,
        ),
        lambda_config=LambdaControllerConfig(
            base_lambda=0.30,
            min_lambda=0.02,
            max_lambda=0.90,
            trend_multiplier=1.55,
            range_multiplier=1.00,
            high_volatility_multiplier=0.30,
            liquidity_stress_multiplier=0.45,
            volatility_sensitivity=0.50,
            drawdown_sensitivity=2.0,
            smoothing_alpha=0.80,
            max_step_change=0.12,
        ),
        execution_config=ExecutionAdaptationConfig(
            cost_model_config=cost_config,
            trend_multiplier=1.15,
            range_multiplier=0.90,
            high_volatility_multiplier=0.30,
            liquidity_stress_multiplier=0.35,
            cost_sensitivity=0.30,
            smoothing_alpha=0.70,
            max_step_change=0.15,
            risk_penalty_factor=0.0,
        ),
        min_regime_stability_ratio=0.80,
        max_accepted_lambda_step=0.13,
        max_accepted_aggressiveness_step=0.16,
        min_tail_loss_reduction_ratio=-10.0,
        stress_drawdown_shock=0.02,
        output_dir=output_dir,
        created_at="2026-06-22T00:15:00Z",
    )


def _phase5_config(*, output_dir: Path) -> SafetyLayerConfig:
    return SafetyLayerConfig(
        detection_window_size=4,
        min_shadow_live_correlation=0.50,
        max_mean_pnl_drift=0.001,
        max_cost_drift_ratio=0.50,
        max_regime_mismatch_rate=0.25,
        max_live_drawdown=1.0,
        output_dir=output_dir,
        created_at="2026-06-22T00:20:00Z",
    )


def _phase6_config(*, output_dir: Path) -> CICDPipelineConfig:
    return CICDPipelineConfig(
        output_dir=output_dir,
        created_at="2026-06-22T00:25:00Z",
    )


def _candidate_identity(result: PolicyTrainingRunResult) -> dict[str, str]:
    artifacts = result.run_manifest["artifacts"]
    return {
        "candidate_run_id": str(result.run_manifest["run_id"]),
        "model_sha256": str(artifacts["model_sha256"]),
        "policy_dataset_hash": str(result.run_manifest["policy_dataset_hash"]),
        "split_hash": str(result.run_manifest["split_hash"]),
    }


def _mirrored_live_observations(
    phase4_result: Phase4AdaptiveSystemResult,
) -> tuple[LiveExecutionObservation, ...]:
    return tuple(
        LiveExecutionObservation(
            decision_ts=decision.decision_ts,
            source=decision.source,
            instrument_id=decision.instrument_id,
            live_filled_action=decision.filled_action,
            live_net_return=decision.net_return,
            live_total_execution_cost=decision.total_execution_cost,
            live_regime=decision.regime,
            capital_at_risk=True,
        )
        for decision in phase4_result.decisions
    )


def _phase6_stage_evidence(
    *,
    identity: Mapping[str, str],
    phase1_5_result: PolicyTrainingRunResult,
    phase2_result: Phase2EvaluationResult,
    phase3_result: Phase3OptimizationResult,
    phase5_result: Phase5SafetyLayerResult,
) -> tuple[CICDStageEvidence, ...]:
    if phase1_5_result.artifact_dir is None:
        raise AssertionError("Phase 1.5 artifact_dir is required")
    if phase2_result.report_path is None:
        raise AssertionError("Phase 2 report_path is required")
    if phase3_result.report_path is None:
        raise AssertionError("Phase 3 report_path is required")
    if phase5_result.report_path is None:
        raise AssertionError("Phase 5 report_path is required")

    return (
        CICDStageEvidence(
            stage="training",
            passed=phase1_5_result.accepted,
            artifact_sha256=identity["model_sha256"],
            report_sha256=_file_sha256(phase1_5_result.artifact_dir / "run_manifest.json"),
            run_id=identity["candidate_run_id"],
            metadata={
                **identity,
                "accepted_candidate_model": phase1_5_result.accepted,
                "deterministic_training": True,
                "model_sha256": identity["model_sha256"],
            },
        ),
        CICDStageEvidence(
            stage="validation",
            passed=phase2_result.passed and phase3_result.passed,
            artifact_sha256=_file_sha256(phase3_result.report_path),
            report_sha256=_file_sha256(phase2_result.report_path),
            run_id="phase2_phase3_validation",
            metadata={
                **identity,
                "oos_backtest_passed": phase2_result.passed,
                "cost_stress_passed": phase3_result.report.acceptance_criteria[
                    "cost_perturbation_robust"
                ],
                "cost_stress_multipliers": [
                    float(multiplier)
                    for multiplier in phase3_result.report.cost_stress_metrics
                ],
            },
        ),
        CICDStageEvidence(
            stage="shadow_deployment",
            passed=phase5_result.passed,
            artifact_sha256=_file_sha256(phase5_result.report_path),
            report_sha256=_file_sha256(phase5_result.report_path),
            run_id="phase5_shadow",
            metadata={
                **identity,
                "shadow_mode": True,
                "simulate_live_execution": True,
                "capital_at_risk": False,
            },
        ),
        CICDStageEvidence(
            stage="live_deployment",
            passed=True,
            artifact_sha256=_file_sha256(phase5_result.report_path),
            report_sha256=_file_sha256(phase5_result.report_path),
            run_id="phase6_live_rollout",
            metadata={
                **identity,
                "staged_capital_rollout": True,
                "manual_approval_recorded": True,
                "rollout_capital_fractions": [0.0, 0.01, 0.05, 0.10],
                "rollout_step_index": 1,
                "requested_capital_fraction": 0.01,
            },
        ),
        CICDStageEvidence(
            stage="monitoring",
            passed=True,
            artifact_sha256=_file_sha256(phase5_result.report_path),
            report_sha256=_file_sha256(phase5_result.report_path),
            run_id="phase6_monitoring",
            metadata={
                **identity,
                "performance_tracking_enabled": True,
                "risk_tracking_enabled": True,
                "kill_switch_wired": True,
            },
        ),
    )


def _assert_hard_gates(
    *,
    identity: Mapping[str, str],
    phase0_dataset: Phase0Dataset,
    phase0_contract: DatasetContract,
    phase1_5_result: PolicyTrainingRunResult,
    phase2_result: Phase2EvaluationResult,
    phase3_result: Phase3OptimizationResult,
    phase4_result: Phase4AdaptiveSystemResult,
    phase5_result: Phase5SafetyLayerResult,
    phase6_result: CICDPipelineResult,
    stage_evidence: tuple[CICDStageEvidence, ...],
) -> None:
    phase_statuses = {
        "phase0": phase0_dataset.validation_report.passed,
        "phase0_artifact": phase0_contract.dataset_hash
        == phase0_dataset.manifest["dataset_hash"],
        "phase1_5": phase1_5_result.accepted,
        "phase2": phase2_result.passed,
        "phase3": phase3_result.passed,
        "phase4": phase4_result.passed,
        "phase5": phase5_result.passed,
        "phase6": phase6_result.passed,
    }
    failed = sorted(phase for phase, passed in phase_statuses.items() if not passed)
    if failed:
        raise AssertionError(f"golden path phase gates failed: {failed}")
    if phase6_result.report.deployment_status != "approved_for_staged_live":
        raise AssertionError("Phase 6 did not approve staged-live deployment")
    if not phase4_result.report.input_provenance_verified:
        raise AssertionError("Phase 4 input provenance was not verified")
    if not phase6_result.report.candidate_identity_verified:
        raise AssertionError("Phase 6 candidate identity was not verified")
    _assert_identity_consistency(
        identity=identity,
        phase2_result=phase2_result,
        phase3_result=phase3_result,
        phase4_result=phase4_result,
        phase6_result=phase6_result,
        stage_evidence=stage_evidence,
    )


def _assert_identity_consistency(
    *,
    identity: Mapping[str, str],
    phase2_result: Phase2EvaluationResult,
    phase3_result: Phase3OptimizationResult,
    phase4_result: Phase4AdaptiveSystemResult,
    phase6_result: CICDPipelineResult,
    stage_evidence: tuple[CICDStageEvidence, ...],
) -> None:
    for phase_name, candidate_run_id, hashes in (
        ("phase2", phase2_result.report.candidate_run_id, phase2_result.report.phase1_5_hashes),
        ("phase3", phase3_result.report.candidate_run_id, phase3_result.report.phase1_5_hashes),
    ):
        if candidate_run_id != identity["candidate_run_id"]:
            raise AssertionError(f"{phase_name} candidate_run_id mismatch")
        _assert_hash_identity(phase_name, identity, hashes)

    phase4_report = phase4_result.report
    phase4_identity = {
        "candidate_run_id": phase4_report.candidate_run_id,
        "model_sha256": phase4_report.model_sha256,
        "policy_dataset_hash": phase4_report.policy_dataset_hash,
        "split_hash": phase4_report.split_hash,
    }
    if dict(identity) != phase4_identity:
        raise AssertionError("Phase 4 identity mismatch")
    if phase6_result.report.candidate_identity != dict(identity):
        raise AssertionError("Phase 6 candidate identity mismatch")
    for evidence in stage_evidence:
        metadata = dict(evidence.metadata)
        for field_name, expected in identity.items():
            if metadata.get(field_name) != expected:
                raise AssertionError(f"{evidence.stage} {field_name} mismatch")


def _assert_hash_identity(
    phase_name: str,
    identity: Mapping[str, str],
    hashes: Mapping[str, str],
) -> None:
    for field_name in ("model_sha256", "policy_dataset_hash", "split_hash"):
        if hashes.get(field_name) != identity[field_name]:
            raise AssertionError(f"{phase_name} {field_name} mismatch")


def _bundle_manifest(
    *,
    run_id: str,
    identity: Mapping[str, str],
    phase0_dataset: Phase0Dataset,
    phase0_contract: DatasetContract,
    phase1_5_result: PolicyTrainingRunResult,
    phase2_result: Phase2EvaluationResult,
    phase3_result: Phase3OptimizationResult,
    phase4_result: Phase4AdaptiveSystemResult,
    phase5_result: Phase5SafetyLayerResult,
    phase6_result: CICDPipelineResult,
) -> dict[str, Any]:
    if phase1_5_result.artifact_dir is None:
        raise AssertionError("Phase 1.5 artifact_dir is required")
    if phase2_result.report_path is None:
        raise AssertionError("Phase 2 report_path is required")
    if phase3_result.report_path is None:
        raise AssertionError("Phase 3 report_path is required")
    if phase4_result.report_path is None:
        raise AssertionError("Phase 4 report_path is required")
    if phase5_result.report_path is None:
        raise AssertionError("Phase 5 report_path is required")
    if phase6_result.report_path is None:
        raise AssertionError("Phase 6 report_path is required")

    artifact_dir = phase1_5_result.artifact_dir
    artifacts = {
        "synthetic_market_data": _artifact_record(Path("phase0/synthetic_market_data.jsonl")),
        "phase0_manifest": _artifact_record(Path("phase0/manifest.json")),
        "phase0_validation_report": _artifact_record(Path("phase0/validation_report.json")),
        "phase0_artifact_gate_report": _artifact_record(
            Path("phase0/artifact_gate_report.json")
        ),
        "phase0_features": _artifact_record(Path("phase0/features.parquet")),
        "phase0_labels": _artifact_record(Path("phase0/labels.parquet")),
        "phase1_5_run_manifest": _artifact_record(artifact_dir / "run_manifest.json"),
        "phase1_5_policy_dataset_manifest": _artifact_record(
            artifact_dir / "policy_dataset_manifest.json"
        ),
        "phase1_5_dataset_profile": _artifact_record(artifact_dir / "dataset_profile.json"),
        "phase1_5_split_manifest": _artifact_record(artifact_dir / "split_manifest.json"),
        "phase1_5_training_manifest": _artifact_record(artifact_dir / "training_manifest.json"),
        "phase1_5_shadow_acceptance_report": _artifact_record(
            artifact_dir / "shadow_acceptance_report.json"
        ),
        "phase1_5_model": _artifact_record(artifact_dir / "model.xgb"),
        "phase2_report": _artifact_record(phase2_result.report_path),
        "phase3_report": _artifact_record(phase3_result.report_path),
        "phase4_input_provenance": _artifact_record(Path("phase4/input_provenance.json")),
        "phase4_report": _artifact_record(phase4_result.report_path),
        "phase5_report": _artifact_record(phase5_result.report_path),
        "phase6_stage_evidence": _artifact_record(Path("phase6/stage_evidence.json")),
        "phase6_release_manifest": _artifact_record(Path("phase6/release_manifest.json")),
        "phase6_report": _artifact_record(phase6_result.report_path),
    }
    return {
        "schema_version": "bigan-v8-golden-path-bundle-v1",
        "run_id": run_id,
        "fixture_id": SYNTHETIC_FIXTURE_ID,
        "created_at": FIXED_CREATED_AT,
        "live_exchange_calls": False,
        "real_trading": False,
        "profitability_claim": False,
        "identity": dict(identity),
        "phase0_artifact_ready": True,
        "phase0_dataset_hash": phase0_contract.dataset_hash,
        "phase0_dataset_contract": phase0_contract.to_dict(),
        "phase_statuses": {
            "phase0_validation_passed": phase0_dataset.validation_report.passed,
            "phase0_artifact_ready": True,
            "phase1_5_candidate_accepted": phase1_5_result.accepted,
            "phase2_passed": phase2_result.passed,
            "phase3_passed": phase3_result.passed,
            "phase4_passed": phase4_result.passed,
            "phase5_passed": phase5_result.passed,
            "phase6_passed": phase6_result.passed,
        },
        "phase4_input_provenance_verified": phase4_result.report.input_provenance_verified,
        "phase6_candidate_identity_verified": phase6_result.report.candidate_identity_verified,
        "phase6_deployment_status": phase6_result.report.deployment_status,
        "phase6_release_manifest_sha256": phase6_result.report.release_manifest_sha256,
        "phase6_pipeline_input_sha256": phase6_result.report.pipeline_input_sha256,
        "artifacts": artifacts,
    }


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "sha256": _file_sha256(path),
        "bytes": path.stat().st_size,
    }


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(dict(payload)), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(_json_ready(row), sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
        for row in rows
    )
    path.write_text(payload, encoding="utf-8")


def _file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Path):
        return value.as_posix()
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "examples" / "v8" / "artifacts",
        help="Directory that will contain the run-scoped artifact bundle.",
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if the run-scoped artifact bundle already exists.",
    )
    args = parser.parse_args(argv)
    result = run_golden_path(
        args.output_dir,
        run_id=args.run_id,
        overwrite=not args.no_overwrite,
    )
    print(
        json.dumps(
            {
                "bundle_manifest": str(result.bundle_manifest_path),
                "deployment_status": result.phase6_result.report.deployment_status,
                "phase6_release_manifest_sha256": (
                    result.phase6_result.report.release_manifest_sha256
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
