"""Phase 3 differentiable PnL optimization for v8."""

from bigan.v8.phase3.contracts import (
    DEFAULT_PHASE3_CREATED_AT,
    PHASE3_DIFFERENTIABLE_PNL_PHASE,
    PHASE3_PARAMETER_NAMES,
    DifferentiableExecutionConfig,
    DifferentiablePnlOptimizationConfig,
    Phase3OptimizationError,
    Phase3OptimizationReport,
)
from bigan.v8.phase3.optimizer import (
    Phase3OptimizationResult,
    run_phase3_optimization,
)

__all__ = [
    "DEFAULT_PHASE3_CREATED_AT",
    "PHASE3_DIFFERENTIABLE_PNL_PHASE",
    "PHASE3_PARAMETER_NAMES",
    "DifferentiableExecutionConfig",
    "DifferentiablePnlOptimizationConfig",
    "Phase3OptimizationError",
    "Phase3OptimizationReport",
    "Phase3OptimizationResult",
    "run_phase3_optimization",
]
