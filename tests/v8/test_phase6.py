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


def _identity_metadata(candidate_run_id: str) -> dict[str, str]:
    return {
        "candidate_run_id": candidate_run_id,
        "model_sha256": "1" * 64,
        "policy_dataset_hash": "e" * 64,
        "split_hash": "f" * 64,
    }


def _stage_evidence(
    candidate_run_id: str = "phase6-candidate-001",
) -> tuple[CICDStageEvidence, ...]:
    model_sha256 = "1" * 64
    identity = _identity_metadata(candidate_run_id)
    return (
        _stage(
            "training",
            artifact_sha256=model_sha256,
            report_sha256="2" * 64,
            run_id="phase1_5_candidate_001",
            metadata={
                **identity,
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
                **identity,
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
                **identity,
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
                **identity,
                "staged_capital_rollout": True,
                "manual_approval_recorded": True,
                "rollout_capital_fractions": [0.0, 0.01, 0.05, 0.10],
                "rollout_step_index": 1,
                "requested_capital_fraction": 0.01,
            },
        ),
        _stage(
            "monitoring",
            artifact_sha256="9" * 64,
            report_sha256="0" * 64,
            run_id="phase6_monitoring_001",
            metadata={
                **identity,
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
    assert result.report.acceptance_criteria["candidate_identity_consistent"] is True
    assert result.report.acceptance_criteria["no_unvalidated_strategy_goes_live"] is True
    assert result.report.candidate_identity_verified is True
    assert result.report.candidate_identity["candidate_run_id"] == (
        "phase6-candidate-001"
    )
    assert len(result.report.candidate_identity_sha256) == 64
    assert result.report.pipeline_input_sha256 == repeated.report.pipeline_input_sha256
    assert result.report.release_manifest_sha256 == (
        repeated.report.release_manifest_sha256
    )
    assert len(result.report.release_manifest_sha256) == 64
    assert result.report_path is not None
    assert result.report_path.name == (
        "phase6_cicd_pipeline_report_"
        f"{result.report.release_manifest['release_id']}.json"
    )
    saved = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert saved["phase"] == PHASE6_CICD_PHASE
    assert saved["passed"] is True
    assert saved["release_manifest_sha256"] == result.report.release_manifest_sha256


def test_phase6_blocks_live_deployment_when_validation_failed() -> None:
    candidate_run_id = "phase6-candidate-002"
    evidence = list(_stage_evidence(candidate_run_id))
    validation = evidence[1]
    evidence[1] = CICDStageEvidence(
        stage=validation.stage,
        passed=False,
        artifact_sha256=validation.artifact_sha256,
        report_sha256=validation.report_sha256,
        run_id=validation.run_id,
        metadata={
            **_identity_metadata(candidate_run_id),
            "oos_backtest_passed": False,
            "cost_stress_passed": False,
            "cost_stress_multipliers": [1.2, 1.5, 2.0],
        },
    )

    result = run_phase6_cicd_pipeline(
        candidate_run_id=candidate_run_id,
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
    candidate_run_id = "phase6-candidate-003"
    result = run_phase6_cicd_pipeline(
        candidate_run_id=candidate_run_id,
        stage_evidence=_stage_evidence(candidate_run_id),
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
    candidate_run_id = "phase6-candidate-004"
    with pytest.raises(Phase6CICDError, match="required CI/CD stages"):
        run_phase6_cicd_pipeline(
            candidate_run_id=candidate_run_id,
            stage_evidence=_stage_evidence(candidate_run_id)[1:],
            rollback_plan=_rollback_plan(),
            config=_config(),
        )


def test_phase6_malformed_stage_metadata_fails_closed_without_exception() -> None:
    candidate_run_id = "phase6-candidate-005"
    evidence = list(_stage_evidence(candidate_run_id))
    validation = evidence[1]
    evidence[1] = CICDStageEvidence(
        stage=validation.stage,
        passed=True,
        artifact_sha256=validation.artifact_sha256,
        report_sha256=validation.report_sha256,
        run_id=validation.run_id,
        metadata={
            **_identity_metadata(candidate_run_id),
            "oos_backtest_passed": True,
            "cost_stress_passed": True,
            "cost_stress_multipliers": ["not-a-number"],
        },
    )

    result = run_phase6_cicd_pipeline(
        candidate_run_id=candidate_run_id,
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


def test_phase6_blocks_cross_stage_model_identity_mismatch() -> None:
    candidate_run_id = "phase6-candidate-006"
    evidence = list(_stage_evidence(candidate_run_id))
    validation = evidence[1]
    mismatched_identity = _identity_metadata(candidate_run_id)
    mismatched_identity["model_sha256"] = "a" * 64
    evidence[1] = CICDStageEvidence(
        stage=validation.stage,
        passed=True,
        artifact_sha256=validation.artifact_sha256,
        report_sha256=validation.report_sha256,
        run_id=validation.run_id,
        metadata={
            **mismatched_identity,
            "oos_backtest_passed": True,
            "cost_stress_passed": True,
            "cost_stress_multipliers": [1.2, 1.5, 2.0],
        },
    )

    result = run_phase6_cicd_pipeline(
        candidate_run_id=candidate_run_id,
        stage_evidence=tuple(evidence),
        rollback_plan=_rollback_plan(),
        config=_config(),
    )

    gate_by_stage = {gate["stage"]: gate for gate in result.report.stage_gates}
    assert not result.passed
    assert result.report.candidate_identity_verified is False
    assert result.report.acceptance_criteria["candidate_identity_consistent"] is False
    assert gate_by_stage["validation"]["allowed"] is False
    assert "model_identity_mismatch" in gate_by_stage["validation"]["reason_codes"]
    assert gate_by_stage["live_deployment"]["allowed"] is False


@pytest.mark.parametrize(
    ("field_name", "reason_code"),
    (
        ("model_sha256", "model_sha256_invalid"),
        ("policy_dataset_hash", "policy_dataset_hash_invalid"),
        ("split_hash", "split_hash_invalid"),
    ),
)
def test_phase6_blocks_invalid_stage_identity_hash(
    field_name: str,
    reason_code: str,
) -> None:
    candidate_run_id = f"phase6-invalid-{field_name}"
    evidence = list(_stage_evidence(candidate_run_id))
    validation = evidence[1]
    invalid_identity = dict(validation.metadata)
    invalid_identity[field_name] = "not-a-sha256"
    evidence[1] = CICDStageEvidence(
        stage=validation.stage,
        passed=True,
        artifact_sha256=validation.artifact_sha256,
        report_sha256=validation.report_sha256,
        run_id=validation.run_id,
        metadata=invalid_identity,
    )

    result = run_phase6_cicd_pipeline(
        candidate_run_id=candidate_run_id,
        stage_evidence=tuple(evidence),
        rollback_plan=_rollback_plan(),
        config=_config(),
    )

    validation_gate = {
        gate["stage"]: gate for gate in result.report.stage_gates
    }["validation"]
    assert not result.passed
    assert result.report.candidate_identity_verified is False
    assert validation_gate["allowed"] is False
    assert reason_code in validation_gate["reason_codes"]


def test_phase6_blocks_live_candidate_run_id_mismatch() -> None:
    candidate_run_id = "phase6-candidate-007"
    evidence = list(_stage_evidence(candidate_run_id))
    live = evidence[3]
    evidence[3] = CICDStageEvidence(
        stage=live.stage,
        passed=True,
        artifact_sha256=live.artifact_sha256,
        report_sha256=live.report_sha256,
        run_id=live.run_id,
        metadata={
            **dict(live.metadata),
            "candidate_run_id": "wrong-candidate",
        },
    )

    result = run_phase6_cicd_pipeline(
        candidate_run_id=candidate_run_id,
        stage_evidence=tuple(evidence),
        rollback_plan=_rollback_plan(),
        config=_config(),
    )

    live_gate = {gate["stage"]: gate for gate in result.report.stage_gates}[
        "live_deployment"
    ]
    assert not result.passed
    assert live_gate["allowed"] is False
    assert "candidate_run_id_mismatch" in live_gate["reason_codes"]


def test_phase6_blocks_initial_live_capital_fraction_above_limit() -> None:
    candidate_run_id = "phase6-candidate-008"
    evidence = list(_stage_evidence(candidate_run_id))
    live = evidence[3]
    evidence[3] = CICDStageEvidence(
        stage=live.stage,
        passed=True,
        artifact_sha256=live.artifact_sha256,
        report_sha256=live.report_sha256,
        run_id=live.run_id,
        metadata={
            **dict(live.metadata),
            "rollout_step_index": 0,
            "requested_capital_fraction": 0.05,
        },
    )

    result = run_phase6_cicd_pipeline(
        candidate_run_id=candidate_run_id,
        stage_evidence=tuple(evidence),
        rollback_plan=_rollback_plan(),
        config=_config(),
    )

    live_gate = {gate["stage"]: gate for gate in result.report.stage_gates}[
        "live_deployment"
    ]
    assert not result.passed
    assert live_gate["allowed"] is False
    assert "initial_capital_fraction_exceeds_limit" in live_gate["reason_codes"]


def test_phase6_blocks_missing_live_rollout_plan() -> None:
    candidate_run_id = "phase6-candidate-009"
    evidence = list(_stage_evidence(candidate_run_id))
    live = evidence[3]
    live_metadata = dict(live.metadata)
    live_metadata.pop("rollout_capital_fractions")
    evidence[3] = CICDStageEvidence(
        stage=live.stage,
        passed=True,
        artifact_sha256=live.artifact_sha256,
        report_sha256=live.report_sha256,
        run_id=live.run_id,
        metadata=live_metadata,
    )

    result = run_phase6_cicd_pipeline(
        candidate_run_id=candidate_run_id,
        stage_evidence=tuple(evidence),
        rollback_plan=_rollback_plan(),
        config=_config(),
    )

    live_gate = {gate["stage"]: gate for gate in result.report.stage_gates}[
        "live_deployment"
    ]
    assert not result.passed
    assert live_gate["allowed"] is False
    assert "rollout_plan_missing" in live_gate["reason_codes"]


def test_phase6_blocks_live_rollout_plan_mismatch() -> None:
    candidate_run_id = "phase6-candidate-010"
    evidence = list(_stage_evidence(candidate_run_id))
    live = evidence[3]
    evidence[3] = CICDStageEvidence(
        stage=live.stage,
        passed=True,
        artifact_sha256=live.artifact_sha256,
        report_sha256=live.report_sha256,
        run_id=live.run_id,
        metadata={
            **dict(live.metadata),
            "rollout_capital_fractions": [0.0, 0.02, 0.05, 0.10],
        },
    )

    result = run_phase6_cicd_pipeline(
        candidate_run_id=candidate_run_id,
        stage_evidence=tuple(evidence),
        rollback_plan=_rollback_plan(),
        config=_config(),
    )

    live_gate = {gate["stage"]: gate for gate in result.report.stage_gates}[
        "live_deployment"
    ]
    assert not result.passed
    assert live_gate["allowed"] is False
    assert "rollout_plan_mismatch" in live_gate["reason_codes"]


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
    changed_metadata = dict(training.metadata)
    changed_metadata["model_sha256"] = "a" * 64
    changed[0] = CICDStageEvidence(
        stage=training.stage,
        passed=training.passed,
        artifact_sha256="a" * 64,
        report_sha256=training.report_sha256,
        run_id=training.run_id,
        metadata=changed_metadata,
    )

    assert compute_phase6_stage_evidence_sha256(tuple(changed)) != (
        compute_phase6_stage_evidence_sha256(evidence)
    )
