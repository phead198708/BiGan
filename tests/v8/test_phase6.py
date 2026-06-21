"""Phase 6 CI/CD deployment-pipeline tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.v8.phase5 import compute_safe_parameters_sha256
from bigan.v8.phase6 import (
    PHASE6_CICD_PHASE,
    CICDPipelineConfig,
    CICDStageEvidence,
    Phase6CICDError,
    RollbackPlan,
    compute_phase6_stage_evidence_sha256,
    run_phase6_cicd_pipeline,
)


def _stage(
    stage: str,
    *,
    passed: bool = True,
    artifact_sha256: str | None = None,
    report_sha256: str | None = None,
    run_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> CICDStageEvidence:
    return CICDStageEvidence(
        stage=stage,  # type: ignore[arg-type]
        passed=passed,
        artifact_sha256=artifact_sha256 or "a" * 64,
        report_sha256=report_sha256 or "b" * 64,
        run_id=run_id,
        metadata=metadata or {},
    )


def _stage_evidence() -> tuple[CICDStageEvidence, ...]:
    model_sha256 = "1" * 64
    return (
        _stage(
            "training",
            artifact_sha256=model_sha256,
            report_sha256="2" * 64,
            run_id="phase1_5_candidate_001",
            metadata={
                "accepted_candidate_model": True,
                "deterministic_training": True,
                "model_sha256": model_sha256,
            },
        ),
        _stage(
            "validation",
            artifact_sha256="3" * 64,
            report_sha256="4" * 64,
            run_id="phase2_validation_001",
            metadata={
                "oos_backtest_passed": True,
                "cost_stress_passed": True,
                "cost_stress_multipliers": [1.2, 1.5, 2.0],
            },
        ),
        _stage(
            "shadow_deployment",
            artifact_sha256="5" * 64,
            report_sha256="6" * 64,
            run_id="phase5_shadow_001",
            metadata={
                "shadow_mode": True,
                "simulate_live_execution": True,
                "capital_at_risk": False,
            },
        ),
        _stage(
            "live_deployment",
            artifact_sha256="7" * 64,
            report_sha256="8" * 64,
            run_id="phase6_live_rollout_001",
            metadata={
                "staged_capital_rollout": True,
                "manual_approval_recorded": True,
                "requested_capital_fraction": 0.05,
            },
        ),
        _stage(
            "monitoring",
            artifact_sha256="9" * 64,
            report_sha256="0" * 64,
            run_id="phase6_monitoring_001",
            metadata={
                "performance_tracking_enabled": True,
                "risk_tracking_enabled": True,
                "kill_switch_wired": True,
            },
        ),
    )


def _rollback_plan(
    *,
    latency_measurements_ms: tuple[int, ...] = (75, 92, 88),
) -> RollbackPlan:
    safe_parameters = {
        "max_position_size": 0.10,
        "risk_mode": "safe",
    }
    return RollbackPlan(
        stable_model_id="stable-phase5-model",
        stable_model_sha256="c" * 64,
        safe_parameter_sha256=compute_safe_parameters_sha256(safe_parameters),
        safe_parameters=safe_parameters,
        rollback_artifact_sha256="d" * 64,
        latency_measurements_ms=latency_measurements_ms,
    )


def _config(output_dir: Path | None = None) -> CICDPipelineConfig:
    return CICDPipelineConfig(
        output_dir=output_dir,
        created_at="2026-06-21T06:00:00Z",
    )


def test_phase6_approves_deterministic_pipeline_and_writes_report(
    tmp_path: Path,
) -> None:
    stage_evidence = _stage_evidence()
    rollback_plan = _rollback_plan()
    config = _config(output_dir=tmp_path)

    result = run_phase6_cicd_pipeline(
        candidate_run_id="phase6-candidate-001",
        stage_evidence=stage_evidence,
        rollback_plan=rollback_plan,
        config=config,
    )
    repeated = run_phase6_cicd_pipeline(
        candidate_run_id="phase6-candidate-001",
        stage_evidence=stage_evidence,
        rollback_plan=rollback_plan,
        config=config,
    )

    assert result.passed
    assert result.report.phase == PHASE6_CICD_PHASE
    assert result.report.deployment_status == "approved_for_staged_live"
    assert all(gate["allowed"] for gate in result.report.stage_gates)
    assert result.report.rollback_gate["available"] is True
    assert result.report.rollback_gate["max_observed_latency_ms"] == 92
    assert result.report.acceptance_criteria["full_pipeline_deterministic"] is True
    assert result.report.acceptance_criteria["no_unvalidated_strategy_goes_live"] is True
    assert result.report.pipeline_input_sha256 == repeated.report.pipeline_input_sha256
    assert result.report.release_manifest_sha256 == (
        repeated.report.release_manifest_sha256
    )
    assert len(result.report.release_manifest_sha256) == 64
    assert result.report_path is not None
    saved = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert saved["phase"] == PHASE6_CICD_PHASE
    assert saved["passed"] is True
    assert saved["release_manifest_sha256"] == result.report.release_manifest_sha256


def test_phase6_blocks_live_deployment_when_validation_failed() -> None:
    evidence = list(_stage_evidence())
    validation = evidence[1]
    evidence[1] = CICDStageEvidence(
        stage=validation.stage,
        passed=False,
        artifact_sha256=validation.artifact_sha256,
        report_sha256=validation.report_sha256,
        run_id=validation.run_id,
        metadata={
            "oos_backtest_passed": False,
            "cost_stress_passed": False,
            "cost_stress_multipliers": [1.2, 1.5, 2.0],
        },
    )

    result = run_phase6_cicd_pipeline(
        candidate_run_id="phase6-candidate-002",
        stage_evidence=tuple(evidence),
        rollback_plan=_rollback_plan(),
        config=_config(),
    )

    assert not result.passed
    gate_by_stage = {gate["stage"]: gate for gate in result.report.stage_gates}
    assert gate_by_stage["validation"]["allowed"] is False
    assert "validation_failed" in gate_by_stage["validation"]["reason_codes"]
    assert gate_by_stage["live_deployment"]["allowed"] is False
    assert "upstream_gate_failed" in gate_by_stage["live_deployment"]["reason_codes"]
    assert result.report.deployment_status == "blocked_fail_closed"
    assert result.report.acceptance_criteria["validation_passed"] is False
    assert result.report.acceptance_criteria["no_unvalidated_strategy_goes_live"] is True


def test_phase6_blocks_live_deployment_when_rollback_latency_exceeds_threshold() -> None:
    result = run_phase6_cicd_pipeline(
        candidate_run_id="phase6-candidate-003",
        stage_evidence=_stage_evidence(),
        rollback_plan=_rollback_plan(latency_measurements_ms=(80, 260, 90)),
        config=_config(),
    )

    assert not result.passed
    gate_by_stage = {gate["stage"]: gate for gate in result.report.stage_gates}
    assert result.report.rollback_gate["available"] is False
    assert result.report.rollback_gate["latency_within_threshold"] is False
    assert gate_by_stage["live_deployment"]["allowed"] is False
    assert "rollback_latency_exceeds_threshold" in (
        gate_by_stage["live_deployment"]["reason_codes"]
    )
    assert result.report.acceptance_criteria["rollback_latency_within_threshold"] is False


def test_phase6_rejects_missing_required_stage_sequence() -> None:
    with pytest.raises(Phase6CICDError, match="required CI/CD stages"):
        run_phase6_cicd_pipeline(
            candidate_run_id="phase6-candidate-004",
            stage_evidence=_stage_evidence()[1:],
            rollback_plan=_rollback_plan(),
            config=_config(),
        )


def test_phase6_malformed_stage_metadata_fails_closed_without_exception() -> None:
    evidence = list(_stage_evidence())
    validation = evidence[1]
    evidence[1] = CICDStageEvidence(
        stage=validation.stage,
        passed=True,
        artifact_sha256=validation.artifact_sha256,
        report_sha256=validation.report_sha256,
        run_id=validation.run_id,
        metadata={
            "oos_backtest_passed": True,
            "cost_stress_passed": True,
            "cost_stress_multipliers": ["not-a-number"],
        },
    )

    result = run_phase6_cicd_pipeline(
        candidate_run_id="phase6-candidate-005",
        stage_evidence=tuple(evidence),
        rollback_plan=_rollback_plan(),
        config=_config(),
    )

    validation_gate = {
        gate["stage"]: gate for gate in result.report.stage_gates
    }["validation"]
    assert result.report.deployment_status == "blocked_fail_closed"
    assert validation_gate["allowed"] is False
    assert "cost_stress_multipliers_incomplete" in validation_gate["reason_codes"]


def test_phase6_rejects_mismatched_safe_parameter_hash() -> None:
    with pytest.raises(ValueError, match="safe_parameter_sha256 mismatch"):
        RollbackPlan(
            stable_model_id="stable-phase5-model",
            stable_model_sha256="c" * 64,
            safe_parameter_sha256="e" * 64,
            safe_parameters={
                "max_position_size": 0.10,
                "risk_mode": "safe",
            },
            rollback_artifact_sha256="d" * 64,
            latency_measurements_ms=(75, 92, 88),
        )


def test_phase6_stage_hash_changes_when_evidence_changes() -> None:
    evidence = _stage_evidence()
    changed = list(evidence)
    training = changed[0]
    changed[0] = CICDStageEvidence(
        stage=training.stage,
        passed=training.passed,
        artifact_sha256="f" * 64,
        report_sha256=training.report_sha256,
        run_id=training.run_id,
        metadata={
            "accepted_candidate_model": True,
            "deterministic_training": True,
            "model_sha256": "f" * 64,
        },
    )

    assert compute_phase6_stage_evidence_sha256(tuple(changed)) != (
        compute_phase6_stage_evidence_sha256(evidence)
    )
