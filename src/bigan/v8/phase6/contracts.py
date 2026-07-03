"""Phase 6 CI/CD deployment-pipeline contracts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from bigan.v8.phase5 import compute_safe_parameters_sha256

PHASE6_CICD_PHASE = "phase6_cicd_strategy_deployment_pipeline"
DEFAULT_PHASE6_CREATED_AT = "1970-01-01T00:00:00Z"

CICDStageName = Literal[
    "training",
    "validation",
    "shadow_deployment",
    "live_deployment",
    "monitoring",
]

REQUIRED_STAGE_ORDER: tuple[CICDStageName, ...] = (
    "training",
    "validation",
    "shadow_deployment",
    "live_deployment",
    "monitoring",
)


class Phase6CICDError(RuntimeError):
    """Raised when Phase 6 receives invalid lifecycle evidence."""


@dataclass(frozen=True, slots=True)
class CICDStageEvidence:
    """Immutable evidence for one CI/CD lifecycle stage."""

    stage: CICDStageName
    passed: bool
    artifact_sha256: str
    report_sha256: str | None = None
    run_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stage not in REQUIRED_STAGE_ORDER:
            raise ValueError("stage must be a known CI/CD stage")
        if not isinstance(self.passed, bool):
            raise ValueError("passed must be a boolean")
        if not _looks_like_sha256(self.artifact_sha256):
            raise ValueError("artifact_sha256 must be a SHA-256 hex digest")
        if self.report_sha256 is not None and not _looks_like_sha256(
            self.report_sha256
        ):
            raise ValueError("report_sha256 must be a SHA-256 hex digest")
        if self.run_id is not None and not self.run_id.strip():
            raise ValueError("run_id must be non-empty when provided")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "passed": self.passed,
            "artifact_sha256": self.artifact_sha256,
            "report_sha256": self.report_sha256,
            "run_id": self.run_id,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RollbackPlan:
    """Auditable rollback target and latency evidence."""

    stable_model_id: str
    stable_model_sha256: str
    safe_parameter_sha256: str
    safe_parameters: Mapping[str, Any]
    rollback_artifact_sha256: str
    latency_measurements_ms: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.stable_model_id:
            raise ValueError("stable_model_id is required")
        for field_name in (
            "stable_model_sha256",
            "safe_parameter_sha256",
            "rollback_artifact_sha256",
        ):
            if not _looks_like_sha256(str(getattr(self, field_name))):
                raise ValueError(f"{field_name} must be a SHA-256 hex digest")
        if not self.safe_parameters:
            raise ValueError("safe_parameters must not be empty")
        if self.safe_parameter_sha256 != compute_safe_parameters_sha256(
            self.safe_parameters
        ):
            raise ValueError("safe_parameter_sha256 mismatch")
        if not self.latency_measurements_ms:
            raise ValueError("latency_measurements_ms must not be empty")
        for latency_ms in self.latency_measurements_ms:
            if latency_ms < 0:
                raise ValueError("latency_measurements_ms must be non-negative")

    @property
    def max_observed_latency_ms(self) -> int:
        return max(self.latency_measurements_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stable_model_id": self.stable_model_id,
            "stable_model_sha256": self.stable_model_sha256,
            "safe_parameter_sha256": self.safe_parameter_sha256,
            "safe_parameters": dict(self.safe_parameters),
            "rollback_artifact_sha256": self.rollback_artifact_sha256,
            "latency_measurements_ms": list(self.latency_measurements_ms),
            "max_observed_latency_ms": self.max_observed_latency_ms,
        }


@dataclass(frozen=True, slots=True)
class CICDPipelineConfig:
    """Deterministic Phase 6 lifecycle gate settings."""

    required_cost_stress_multipliers: tuple[float, ...] = (1.2, 1.5, 2.0)
    rollout_capital_fractions: tuple[float, ...] = (0.0, 0.01, 0.05, 0.10)
    max_initial_live_capital_fraction: float = 0.01
    max_live_capital_fraction: float = 0.10
    max_rollback_latency_ms: int = 250
    require_manual_approval_for_live: bool = True
    output_dir: Path | str | None = None
    created_at: str = DEFAULT_PHASE6_CREATED_AT

    def __post_init__(self) -> None:
        if self.output_dir is not None and not isinstance(self.output_dir, Path):
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not self.required_cost_stress_multipliers:
            raise ValueError("required_cost_stress_multipliers must not be empty")
        for multiplier in self.required_cost_stress_multipliers:
            if multiplier <= 0.0 or not math.isfinite(multiplier):
                raise ValueError(
                    "required_cost_stress_multipliers must be positive finite values"
                )
        if not self.rollout_capital_fractions:
            raise ValueError("rollout_capital_fractions must not be empty")
        previous = -1.0
        for fraction in self.rollout_capital_fractions:
            if not math.isfinite(fraction) or fraction < 0.0:
                raise ValueError("rollout_capital_fractions must be non-negative")
            if fraction < previous:
                raise ValueError("rollout_capital_fractions must be non-decreasing")
            previous = fraction
        if self.rollout_capital_fractions[0] > self.max_initial_live_capital_fraction:
            raise ValueError("first rollout fraction exceeds initial capital limit")
        if self.rollout_capital_fractions[-1] > self.max_live_capital_fraction:
            raise ValueError("rollout fractions exceed live capital limit")
        if self.max_initial_live_capital_fraction < 0.0:
            raise ValueError("max_initial_live_capital_fraction must be non-negative")
        if self.max_live_capital_fraction <= 0.0:
            raise ValueError("max_live_capital_fraction must be positive")
        if self.max_initial_live_capital_fraction > self.max_live_capital_fraction:
            raise ValueError("initial capital limit cannot exceed live capital limit")
        if self.max_rollback_latency_ms <= 0:
            raise ValueError("max_rollback_latency_ms must be positive")
        if not self.created_at:
            raise ValueError("created_at is required")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["required_cost_stress_multipliers"] = list(
            self.required_cost_stress_multipliers
        )
        payload["rollout_capital_fractions"] = list(self.rollout_capital_fractions)
        payload["output_dir"] = None if self.output_dir is None else str(self.output_dir)
        return payload

    def deterministic_payload(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("output_dir", None)
        return payload


@dataclass(frozen=True, slots=True)
class CICDPipelineReport:
    """Auditable Phase 6 CI/CD deployment report."""

    phase: str
    candidate_run_id: str
    pipeline_input_sha256: str
    candidate_identity: dict[str, Any]
    candidate_identity_verified: bool
    candidate_identity_sha256: str
    release_manifest_sha256: str
    deployment_status: str
    stage_gates: list[dict[str, Any]]
    rollback_gate: dict[str, Any]
    release_manifest: dict[str, Any]
    acceptance_criteria: dict[str, bool]
    config: dict[str, Any]
    created_at: str

    @property
    def passed(self) -> bool:
        return all(self.acceptance_criteria.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "passed": self.passed,
            "candidate_run_id": self.candidate_run_id,
            "pipeline_input_sha256": self.pipeline_input_sha256,
            "candidate_identity": self.candidate_identity,
            "candidate_identity_verified": self.candidate_identity_verified,
            "candidate_identity_sha256": self.candidate_identity_sha256,
            "release_manifest_sha256": self.release_manifest_sha256,
            "deployment_status": self.deployment_status,
            "stage_gates": self.stage_gates,
            "rollback_gate": self.rollback_gate,
            "release_manifest": self.release_manifest,
            "acceptance_criteria": self.acceptance_criteria,
            "config": self.config,
            "created_at": self.created_at,
        }


def compute_phase6_stage_evidence_sha256(
    stage_evidence: tuple[CICDStageEvidence, ...],
) -> str:
    """Hash the exact ordered CI/CD stage-evidence stream."""

    return _canonical_payload_sha256([evidence.to_dict() for evidence in stage_evidence])


def compute_phase6_release_manifest_sha256(
    release_manifest: Mapping[str, Any],
) -> str:
    """Hash a Phase 6 release manifest."""

    return _canonical_payload_sha256(dict(release_manifest))


def _canonical_payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        _json_ready(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _looks_like_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())
