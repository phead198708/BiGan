"""Phase 6 CI/CD strategy deployment pipeline."""

from bigan.v8.phase6.contracts import (
    PHASE6_CICD_PHASE,
    REQUIRED_STAGE_ORDER,
    CICDPipelineConfig,
    CICDPipelineReport,
    CICDStageEvidence,
    Phase6CICDError,
    RollbackPlan,
    compute_phase6_release_manifest_sha256,
    compute_phase6_stage_evidence_sha256,
)
from bigan.v8.phase6.pipeline import (
    CICDPipelineResult,
    compute_phase6_pipeline_input_sha256,
    run_phase6_cicd_pipeline,
)

__all__ = [
    "PHASE6_CICD_PHASE",
    "REQUIRED_STAGE_ORDER",
    "CICDPipelineConfig",
    "CICDPipelineReport",
    "CICDPipelineResult",
    "CICDStageEvidence",
    "Phase6CICDError",
    "RollbackPlan",
    "compute_phase6_pipeline_input_sha256",
    "compute_phase6_release_manifest_sha256",
    "compute_phase6_stage_evidence_sha256",
    "run_phase6_cicd_pipeline",
]
