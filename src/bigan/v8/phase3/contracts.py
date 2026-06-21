"""Phase 3 differentiable PnL optimization contracts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bigan.v8.phase0.costs import CostModelConfig
from bigan.v8.phase2.contracts import ExecutionSimulationConfig

PHASE3_DIFFERENTIABLE_PNL_PHASE = "phase3_differentiable_pnl_optimization"
DEFAULT_PHASE3_CREATED_AT = "1970-01-01T00:00:00Z"
PHASE3_PARAMETER_NAMES: tuple[str, ...] = (
    "bias",
    "score_weight",
    "confidence_weight",
    "baseline_action_weight",
)


class Phase3OptimizationError(RuntimeError):
    """Raised when Phase 3 receives unsafe inputs or cannot optimize safely."""


@dataclass(frozen=True, slots=True)
class DifferentiableExecutionConfig:
    """Smooth execution assumptions used by the Phase 3 optimization loss."""

    cost_model_config: CostModelConfig = CostModelConfig()
    slippage_multiplier: float = 1.0
    order_size: float = 1.0
    min_order_size: float = 1e-12
    min_fill_probability: float = 0.0
    risk_penalty_factor: float = 0.05
    turnover_penalty_factor: float = 0.0
    max_position_size: float = 1.0
    active_action_epsilon: float = 1e-12
    smooth_abs_epsilon: float = 1e-8
    action_temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.slippage_multiplier <= 0.0:
            raise ValueError("slippage_multiplier must be positive")
        if self.order_size <= 0.0:
            raise ValueError("order_size must be positive")
        if self.min_order_size <= 0.0:
            raise ValueError("min_order_size must be positive")
        if not 0.0 <= self.min_fill_probability <= 1.0:
            raise ValueError("min_fill_probability must be in [0, 1]")
        if self.risk_penalty_factor < 0.0:
            raise ValueError("risk_penalty_factor must be non-negative")
        if self.turnover_penalty_factor < 0.0:
            raise ValueError("turnover_penalty_factor must be non-negative")
        if not 0.0 < self.max_position_size <= 1.0:
            raise ValueError("max_position_size must be in (0, 1]")
        if self.active_action_epsilon < 0.0:
            raise ValueError("active_action_epsilon must be non-negative")
        if self.smooth_abs_epsilon <= 0.0:
            raise ValueError("smooth_abs_epsilon must be positive")
        if self.action_temperature <= 0.0:
            raise ValueError("action_temperature must be positive")

    def to_phase2_execution_config(self) -> ExecutionSimulationConfig:
        """Use the same execution assumptions for the Phase 2 baseline."""

        return ExecutionSimulationConfig(
            cost_model_config=self.cost_model_config,
            slippage_multiplier=self.slippage_multiplier,
            order_size=self.order_size,
            min_order_size=self.min_order_size,
            min_fill_probability=self.min_fill_probability,
            risk_penalty_factor=self.risk_penalty_factor,
            turnover_penalty_factor=self.turnover_penalty_factor,
            apply_cost_aware_filter=False,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cost_model_config"] = self.cost_model_config.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class DifferentiablePnlOptimizationConfig:
    """Training and acceptance thresholds for Phase 3."""

    execution_config: DifferentiableExecutionConfig = DifferentiableExecutionConfig()
    phase2_baseline_execution_config: ExecutionSimulationConfig | None = None
    initial_parameters: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)
    learning_rate: float = 0.25
    max_steps: int = 80
    finite_difference_epsilon: float = 1e-4
    return_variance_penalty: float = 0.0
    min_loss_improvement: float = 1e-8
    min_gradient_norm: float = 1e-8
    max_gradient_norm: float = 1_000.0
    max_abs_parameter: float = 12.0
    min_sharpe_improvement_ratio_over_phase2: float = 0.0
    min_oos_sharpe: float = 0.0
    max_cost_stress_sharpe_drop_ratio: float = 1.0
    cost_stress_multipliers: tuple[float, ...] = (1.2, 1.5, 2.0)
    output_dir: Path | str | None = None
    created_at: str = DEFAULT_PHASE3_CREATED_AT

    def __post_init__(self) -> None:
        if self.output_dir is not None and not isinstance(self.output_dir, Path):
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if len(self.initial_parameters) != len(PHASE3_PARAMETER_NAMES):
            raise ValueError("initial_parameters must contain four values")
        for value in self.initial_parameters:
            if not math.isfinite(value):
                raise ValueError("initial_parameters must be finite")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.finite_difference_epsilon <= 0.0:
            raise ValueError("finite_difference_epsilon must be positive")
        if self.return_variance_penalty < 0.0:
            raise ValueError("return_variance_penalty must be non-negative")
        if self.min_loss_improvement < 0.0:
            raise ValueError("min_loss_improvement must be non-negative")
        if self.min_gradient_norm < 0.0:
            raise ValueError("min_gradient_norm must be non-negative")
        if self.max_gradient_norm <= 0.0:
            raise ValueError("max_gradient_norm must be positive")
        if self.max_abs_parameter <= 0.0:
            raise ValueError("max_abs_parameter must be positive")
        if not math.isfinite(self.min_sharpe_improvement_ratio_over_phase2):
            raise ValueError("min_sharpe_improvement_ratio_over_phase2 must be finite")
        if not math.isfinite(self.min_oos_sharpe):
            raise ValueError("min_oos_sharpe must be finite")
        if (
            not math.isfinite(self.max_cost_stress_sharpe_drop_ratio)
            or self.max_cost_stress_sharpe_drop_ratio < 0.0
        ):
            raise ValueError(
                "max_cost_stress_sharpe_drop_ratio must be finite and non-negative"
            )
        if not self.cost_stress_multipliers:
            raise ValueError("cost_stress_multipliers must not be empty")
        for multiplier in self.cost_stress_multipliers:
            if multiplier <= 0.0 or not math.isfinite(multiplier):
                raise ValueError("cost_stress_multipliers must be positive finite values")
        if not self.created_at:
            raise ValueError("created_at is required")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["execution_config"] = self.execution_config.to_dict()
        payload["phase2_baseline_execution_config"] = (
            None
            if self.phase2_baseline_execution_config is None
            else self.phase2_baseline_execution_config.to_dict()
        )
        payload["output_dir"] = None if self.output_dir is None else str(self.output_dir)
        return payload


@dataclass(frozen=True, slots=True)
class Phase3OptimizationReport:
    """Auditable Phase 3 optimization and OOS acceptance report."""

    phase: str
    candidate_run_id: str
    candidate_artifact_dir: str
    phase1_5_hashes: dict[str, str]
    phase2_report_path: str | None
    phase2_report_sha256: str | None
    phase2_baseline_source: str
    phase2_baseline_metrics: dict[str, Any]
    train_metrics: dict[str, Any]
    oos_metrics: dict[str, Any]
    comparison_metrics: dict[str, Any]
    cost_stress_metrics: dict[str, dict[str, Any]]
    optimization_trace: list[dict[str, float | int]]
    optimized_parameters: dict[str, float]
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
            "candidate_artifact_dir": self.candidate_artifact_dir,
            "phase1_5_hashes": self.phase1_5_hashes,
            "phase2_report_path": self.phase2_report_path,
            "phase2_report_sha256": self.phase2_report_sha256,
            "phase2_baseline_source": self.phase2_baseline_source,
            "phase2_baseline_metrics": self.phase2_baseline_metrics,
            "train_metrics": self.train_metrics,
            "oos_metrics": self.oos_metrics,
            "comparison_metrics": self.comparison_metrics,
            "cost_stress_metrics": self.cost_stress_metrics,
            "optimization_trace": self.optimization_trace,
            "optimized_parameters": self.optimized_parameters,
            "acceptance_criteria": self.acceptance_criteria,
            "config": self.config,
            "created_at": self.created_at,
        }
