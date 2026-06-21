"""Phase 6 CI/CD deployment-pipeline runner."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.phase6.contracts import (
    PHASE6_CICD_PHASE,
    REQUIRED_STAGE_ORDER,
    CICDPipelineConfig,
    CICDPipelineReport,
    CICDStageEvidence,
    Phase6CICDError,
    RollbackPlan,
    _canonical_payload_sha256,
    _json_ready,
    compute_phase6_release_manifest_sha256,
    compute_phase6_stage_evidence_sha256,
)

_IDENTITY_FIELDS = (
    "candidate_run_id",
    "model_sha256",
    "policy_dataset_hash",
    "split_hash",
)

_IDENTITY_REASON_CODES = {
    "candidate_run_id": (
        "candidate_run_id_missing",
        "candidate_run_id_mismatch",
    ),
    "model_sha256": (
        "model_sha256_missing",
        "model_identity_mismatch",
    ),
    "policy_dataset_hash": (
        "policy_dataset_hash_missing",
        "policy_dataset_identity_mismatch",
    ),
    "split_hash": (
        "split_hash_missing",
        "split_identity_mismatch",
    ),
}


@dataclass(frozen=True, slots=True)
class CICDPipelineResult:
    """Result of a Phase 6 CI/CD lifecycle gate run."""

    report: CICDPipelineReport
    report_path: Path | None = None

    @property
    def passed(self) -> bool:
        return self.report.passed


def run_phase6_cicd_pipeline(
    *,
    candidate_run_id: str,
    stage_evidence: tuple[CICDStageEvidence, ...],
    rollback_plan: RollbackPlan,
    config: CICDPipelineConfig | None = None,
) -> CICDPipelineResult:
    """Validate the full strategy lifecycle before staged live deployment."""

    if not candidate_run_id.strip():
        raise Phase6CICDError("candidate_run_id is required")
    resolved_config = config or CICDPipelineConfig()
    _assert_stage_sequence(stage_evidence)

    pipeline_input_sha256 = compute_phase6_pipeline_input_sha256(
        candidate_run_id=candidate_run_id,
        stage_evidence=stage_evidence,
        rollback_plan=rollback_plan,
        config=resolved_config,
    )
    identity_audit = _candidate_identity_audit(
        candidate_run_id=candidate_run_id,
        stage_evidence=stage_evidence,
    )
    rollback_gate = _rollback_gate(rollback_plan, resolved_config)
    stage_gates = _stage_gates(
        stage_evidence=stage_evidence,
        identity_audit=identity_audit,
        rollback_gate=rollback_gate,
        config=resolved_config,
    )
    deployment_status = (
        "approved_for_staged_live"
        if all(gate["allowed"] for gate in stage_gates)
        else "blocked_fail_closed"
    )
    release_manifest = _release_manifest(
        candidate_run_id=candidate_run_id,
        pipeline_input_sha256=pipeline_input_sha256,
        deployment_status=deployment_status,
        identity_audit=identity_audit,
        stage_evidence=stage_evidence,
        stage_gates=stage_gates,
        rollback_gate=rollback_gate,
        rollback_plan=rollback_plan,
        config=resolved_config,
    )
    release_manifest_sha256 = compute_phase6_release_manifest_sha256(
        release_manifest
    )
    acceptance_criteria = _acceptance_criteria(
        release_manifest=release_manifest,
        release_manifest_sha256=release_manifest_sha256,
        stage_gates=stage_gates,
        rollback_gate=rollback_gate,
    )
    report = CICDPipelineReport(
        phase=PHASE6_CICD_PHASE,
        candidate_run_id=candidate_run_id,
        pipeline_input_sha256=pipeline_input_sha256,
        candidate_identity=identity_audit["candidate_identity"],
        candidate_identity_verified=identity_audit["candidate_identity_verified"],
        candidate_identity_sha256=identity_audit["candidate_identity_sha256"],
        release_manifest_sha256=release_manifest_sha256,
        deployment_status=deployment_status,
        stage_gates=stage_gates,
        rollback_gate=rollback_gate,
        release_manifest=release_manifest,
        acceptance_criteria=acceptance_criteria,
        config=resolved_config.to_dict(),
        created_at=resolved_config.created_at,
    )
    report_path = None
    if resolved_config.output_dir is not None:
        output_dir = Path(resolved_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        release_id = str(release_manifest["release_id"])
        report_path = output_dir / f"phase6_cicd_pipeline_report_{release_id}.json"
        report_path.write_text(
            json.dumps(
                _json_ready(report.to_dict()),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
    return CICDPipelineResult(report=report, report_path=report_path)


def compute_phase6_pipeline_input_sha256(
    *,
    candidate_run_id: str,
    stage_evidence: tuple[CICDStageEvidence, ...],
    rollback_plan: RollbackPlan,
    config: CICDPipelineConfig,
) -> str:
    """Hash every deterministic input used by the Phase 6 lifecycle gate."""

    payload = {
        "candidate_run_id": candidate_run_id,
        "stage_evidence": [evidence.to_dict() for evidence in stage_evidence],
        "stage_evidence_sha256": compute_phase6_stage_evidence_sha256(
            stage_evidence
        ),
        "rollback_plan": rollback_plan.to_dict(),
        "config": config.deterministic_payload(),
    }
    return _canonical_payload_sha256(payload)


def _assert_stage_sequence(stage_evidence: tuple[CICDStageEvidence, ...]) -> None:
    stages = tuple(evidence.stage for evidence in stage_evidence)
    if stages != REQUIRED_STAGE_ORDER:
        raise Phase6CICDError(
            "stage_evidence must contain exactly the required CI/CD stages in order: "
            + ", ".join(REQUIRED_STAGE_ORDER)
        )


def _rollback_gate(
    rollback_plan: RollbackPlan,
    config: CICDPipelineConfig,
) -> dict[str, Any]:
    latency_within_threshold = (
        rollback_plan.max_observed_latency_ms <= config.max_rollback_latency_ms
    )
    safe_parameter_hash_verified = (
        rollback_plan.safe_parameter_sha256
        == _canonical_payload_sha256(dict(rollback_plan.safe_parameters))
    )
    reason_codes: list[str] = []
    if not safe_parameter_hash_verified:
        reason_codes.append("rollback_safe_parameter_hash_mismatch")
    if not latency_within_threshold:
        reason_codes.append("rollback_latency_exceeds_threshold")
    return {
        "available": safe_parameter_hash_verified and latency_within_threshold,
        "safe_parameter_hash_verified": safe_parameter_hash_verified,
        "rollback_artifact_present": True,
        "max_observed_latency_ms": rollback_plan.max_observed_latency_ms,
        "latency_threshold_ms": config.max_rollback_latency_ms,
        "latency_within_threshold": latency_within_threshold,
        "reason_codes": reason_codes,
    }


def _candidate_identity_audit(
    *,
    candidate_run_id: str,
    stage_evidence: tuple[CICDStageEvidence, ...],
) -> dict[str, Any]:
    training_metadata = dict(stage_evidence[0].metadata)
    candidate_identity = {
        "candidate_run_id": candidate_run_id,
        "model_sha256": training_metadata.get("model_sha256"),
        "policy_dataset_hash": training_metadata.get("policy_dataset_hash"),
        "split_hash": training_metadata.get("split_hash"),
    }
    stage_reason_codes: dict[str, list[str]] = {}
    for evidence in stage_evidence:
        metadata = dict(evidence.metadata)
        reason_codes: list[str] = []
        for field_name in _IDENTITY_FIELDS:
            missing_code, mismatch_code = _IDENTITY_REASON_CODES[field_name]
            observed = metadata.get(field_name)
            expected = candidate_identity[field_name]
            if observed is None:
                reason_codes.append(missing_code)
                continue
            if observed != expected:
                reason_codes.append(mismatch_code)
        stage_reason_codes[evidence.stage] = _dedupe(reason_codes)

    reason_codes = _dedupe(
        [
            code
            for stage_codes in stage_reason_codes.values()
            for code in stage_codes
        ]
    )
    return {
        "candidate_identity": candidate_identity,
        "candidate_identity_verified": not reason_codes,
        "candidate_identity_sha256": _canonical_payload_sha256(candidate_identity),
        "stage_reason_codes": stage_reason_codes,
        "reason_codes": reason_codes,
    }


def _stage_gates(
    *,
    stage_evidence: tuple[CICDStageEvidence, ...],
    identity_audit: dict[str, Any],
    rollback_gate: dict[str, Any],
    config: CICDPipelineConfig,
) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    upstream_allowed = True
    for evidence in stage_evidence:
        reason_codes: list[str] = []
        if not upstream_allowed:
            reason_codes.append("upstream_gate_failed")
        if not evidence.passed:
            reason_codes.append(f"{evidence.stage}_failed")
        reason_codes.extend(identity_audit["stage_reason_codes"][evidence.stage])
        reason_codes.extend(_stage_specific_reason_codes(evidence, config))
        if evidence.stage == "live_deployment":
            reason_codes.extend(_live_deployment_reason_codes(rollback_gate))
        reason_codes = _dedupe(reason_codes)
        allowed = upstream_allowed and evidence.passed and not reason_codes
        gates.append(
            {
                "stage": evidence.stage,
                "allowed": allowed,
                "artifact_sha256": evidence.artifact_sha256,
                "report_sha256": evidence.report_sha256,
                "run_id": evidence.run_id,
                "reason_codes": reason_codes,
            }
        )
        upstream_allowed = allowed
    return gates


def _stage_specific_reason_codes(
    evidence: CICDStageEvidence,
    config: CICDPipelineConfig,
) -> list[str]:
    metadata = dict(evidence.metadata)
    if evidence.stage == "training":
        return _training_reason_codes(evidence, metadata)
    if evidence.stage == "validation":
        return _validation_reason_codes(metadata, config)
    if evidence.stage == "shadow_deployment":
        return _shadow_reason_codes(metadata)
    if evidence.stage == "live_deployment":
        return _live_reason_codes(metadata, config)
    if evidence.stage == "monitoring":
        return _monitoring_reason_codes(metadata)
    return ["unknown_stage"]


def _training_reason_codes(
    evidence: CICDStageEvidence,
    metadata: dict[str, Any],
) -> list[str]:
    reason_codes: list[str] = []
    if evidence.run_id is None:
        reason_codes.append("training_run_id_missing")
    if metadata.get("accepted_candidate_model") is not True:
        reason_codes.append("candidate_model_not_accepted")
    if metadata.get("deterministic_training") is not True:
        reason_codes.append("training_not_reproducible")
    if metadata.get("model_sha256") != evidence.artifact_sha256:
        reason_codes.append("training_model_hash_mismatch")
    return reason_codes


def _validation_reason_codes(
    metadata: dict[str, Any],
    config: CICDPipelineConfig,
) -> list[str]:
    reason_codes: list[str] = []
    if metadata.get("oos_backtest_passed") is not True:
        reason_codes.append("oos_backtest_not_passed")
    if metadata.get("cost_stress_passed") is not True:
        reason_codes.append("cost_stress_not_passed")
    if not _covers_required_multipliers(
        metadata.get("cost_stress_multipliers", ()),
        config.required_cost_stress_multipliers,
    ):
        reason_codes.append("cost_stress_multipliers_incomplete")
    return reason_codes


def _shadow_reason_codes(metadata: dict[str, Any]) -> list[str]:
    reason_codes: list[str] = []
    if metadata.get("shadow_mode") is not True:
        reason_codes.append("shadow_mode_not_enabled")
    if metadata.get("simulate_live_execution") is not True:
        reason_codes.append("shadow_live_simulation_missing")
    if metadata.get("capital_at_risk") is not False:
        reason_codes.append("shadow_capital_at_risk")
    return reason_codes


def _live_reason_codes(
    metadata: dict[str, Any],
    config: CICDPipelineConfig,
) -> list[str]:
    reason_codes: list[str] = []
    if metadata.get("staged_capital_rollout") is not True:
        reason_codes.append("staged_capital_rollout_missing")
    if (
        config.require_manual_approval_for_live
        and metadata.get("manual_approval_recorded") is not True
    ):
        reason_codes.append("manual_approval_missing")

    rollout_payload = metadata.get("rollout_capital_fractions")
    rollout_plan = _coerce_rollout_plan(rollout_payload)
    if rollout_payload is None:
        reason_codes.append("rollout_plan_missing")
    elif rollout_plan is None or not _rollout_plans_match(
        rollout_plan,
        config.rollout_capital_fractions,
    ):
        reason_codes.append("rollout_plan_mismatch")

    rollout_step_index = metadata.get("rollout_step_index")
    step_index_valid = (
        not isinstance(rollout_step_index, bool)
        and isinstance(rollout_step_index, int)
        and rollout_step_index >= 0
        and rollout_step_index < len(config.rollout_capital_fractions)
    )
    if not step_index_valid:
        reason_codes.append("rollout_step_missing")

    requested_fraction = metadata.get("requested_capital_fraction")
    requested_fraction_valid = (
        not isinstance(requested_fraction, bool)
        and isinstance(requested_fraction, int | float)
        and math.isfinite(float(requested_fraction))
    )
    requested_value = float(requested_fraction) if requested_fraction_valid else None
    if (
        not requested_fraction_valid
    ):
        reason_codes.append("requested_capital_fraction_missing")
    elif requested_value > config.max_live_capital_fraction:
        reason_codes.append("requested_capital_fraction_exceeds_limit")

    if requested_value is not None and rollout_plan is not None:
        requested_fraction_not_in_rollout_plan = not any(
            _float_equal(requested_value, fraction) for fraction in rollout_plan
        ) or (
            step_index_valid
            and not _float_equal(
                requested_value,
                rollout_plan[int(rollout_step_index)],
            )
        )
        if requested_fraction_not_in_rollout_plan:
            reason_codes.append("requested_capital_fraction_not_in_rollout_plan")
        first_live_step_index = _first_non_zero_rollout_index(rollout_plan)
        if (
            step_index_valid
            and int(rollout_step_index) in {0, first_live_step_index}
            and requested_value > config.max_initial_live_capital_fraction
        ):
            reason_codes.append("initial_capital_fraction_exceeds_limit")
    return reason_codes


def _monitoring_reason_codes(metadata: dict[str, Any]) -> list[str]:
    reason_codes: list[str] = []
    if metadata.get("performance_tracking_enabled") is not True:
        reason_codes.append("performance_tracking_missing")
    if metadata.get("risk_tracking_enabled") is not True:
        reason_codes.append("risk_tracking_missing")
    if metadata.get("kill_switch_wired") is not True:
        reason_codes.append("kill_switch_not_wired")
    return reason_codes


def _live_deployment_reason_codes(rollback_gate: dict[str, Any]) -> list[str]:
    if rollback_gate["available"]:
        return []
    return list(rollback_gate["reason_codes"]) or ["rollback_unavailable"]


def _release_manifest(
    *,
    candidate_run_id: str,
    pipeline_input_sha256: str,
    deployment_status: str,
    identity_audit: dict[str, Any],
    stage_evidence: tuple[CICDStageEvidence, ...],
    stage_gates: list[dict[str, Any]],
    rollback_gate: dict[str, Any],
    rollback_plan: RollbackPlan,
    config: CICDPipelineConfig,
) -> dict[str, Any]:
    release_id = "phase6_release_" + pipeline_input_sha256[:16]
    return {
        "release_id": release_id,
        "candidate_run_id": candidate_run_id,
        "phase": PHASE6_CICD_PHASE,
        "deployment_status": deployment_status,
        "pipeline_input_sha256": pipeline_input_sha256,
        "candidate_identity": identity_audit["candidate_identity"],
        "candidate_identity_verified": identity_audit["candidate_identity_verified"],
        "candidate_identity_sha256": identity_audit["candidate_identity_sha256"],
        "stage_order": list(REQUIRED_STAGE_ORDER),
        "stage_evidence": [evidence.to_dict() for evidence in stage_evidence],
        "stage_gates": stage_gates,
        "rollback_gate": rollback_gate,
        "rollback_plan": rollback_plan.to_dict(),
        "rollout_capital_fractions": list(config.rollout_capital_fractions),
        "manual_approval_required": config.require_manual_approval_for_live,
        "created_at": config.created_at,
    }


def _acceptance_criteria(
    *,
    release_manifest: dict[str, Any],
    release_manifest_sha256: str,
    stage_gates: list[dict[str, Any]],
    rollback_gate: dict[str, Any],
) -> dict[str, bool]:
    gates = {gate["stage"]: bool(gate["allowed"]) for gate in stage_gates}
    prereqs_validated = all(
        gates[stage]
        for stage in (
            "training",
            "validation",
            "shadow_deployment",
        )
    )
    live_allowed = gates["live_deployment"]
    no_unvalidated_strategy_goes_live = (not live_allowed) or prereqs_validated
    return {
        "full_pipeline_deterministic": (
            release_manifest_sha256
            == compute_phase6_release_manifest_sha256(release_manifest)
        ),
        "candidate_identity_consistent": bool(
            release_manifest["candidate_identity_verified"]
        ),
        "reproducible_training_pipeline": gates["training"],
        "validation_passed": gates["validation"],
        "shadow_deployment_passed": gates["shadow_deployment"],
        "staged_live_deployment_passed": gates["live_deployment"],
        "monitoring_enabled": gates["monitoring"],
        "rollback_available": bool(rollback_gate["available"]),
        "rollback_latency_within_threshold": bool(
            rollback_gate["latency_within_threshold"]
        ),
        "no_unvalidated_strategy_goes_live": no_unvalidated_strategy_goes_live,
    }


def _covers_required_multipliers(
    observed: Any,
    required: tuple[float, ...],
) -> bool:
    if not isinstance(observed, (list, tuple)):
        return False
    observed_values: list[float] = []
    for value in observed:
        try:
            observed_value = float(value)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(observed_value):
            return False
        observed_values.append(observed_value)
    return all(
        any(
            abs(observed_value - required_value) <= 1e-12
            for observed_value in observed_values
        )
        for required_value in required
    )


def _coerce_rollout_plan(value: Any) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)):
        return None
    fractions: list[float] = []
    for fraction in value:
        if isinstance(fraction, bool):
            return None
        try:
            fraction_value = float(fraction)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(fraction_value):
            return None
        fractions.append(fraction_value)
    return tuple(fractions)


def _rollout_plans_match(
    observed: tuple[float, ...],
    expected: tuple[float, ...],
) -> bool:
    return len(observed) == len(expected) and all(
        _float_equal(observed_fraction, expected_fraction)
        for observed_fraction, expected_fraction in zip(observed, expected, strict=True)
    )


def _first_non_zero_rollout_index(rollout_plan: tuple[float, ...]) -> int:
    for index, fraction in enumerate(rollout_plan):
        if fraction > 0.0:
            return index
    return 0


def _float_equal(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-12


def _dedupe(reason_codes: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for code in reason_codes:
        if code not in seen:
            deduped.append(code)
            seen.add(code)
    return deduped
