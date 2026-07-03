"""Phase 2 execution-consistent PnL evaluation contracts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bigan.v8.phase0.costs import CostModelConfig

PHASE2_EVALUATION_PHASE = "phase2_hybrid_pnl_evaluation"
DEFAULT_PHASE2_CREATED_AT = "1970-01-01T00:00:00Z"


class Phase2ArtifactError(RuntimeError):
    """Raised when Phase 2 receives unsafe Phase 1.5 artifacts."""


@dataclass(frozen=True, slots=True)
class ExecutionSimulationConfig:
    """Execution and risk assumptions for Phase 2 offline evaluation."""

    cost_model_config: CostModelConfig = CostModelConfig()
    slippage_multiplier: float = 1.0
    order_size: float = 1.0
    min_order_size: float = 1e-12
    min_fill_probability: float = 0.0
    risk_penalty_factor: float = 0.05
    turnover_penalty_factor: float = 0.0
    pnl_lambda: float = 0.25
    policy_edge_scale: float = 0.02
    min_expected_net_edge: float = 0.0
    apply_cost_aware_filter: bool = True
    active_action_epsilon: float = 1e-12

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
        if self.pnl_lambda < 0.0:
            raise ValueError("pnl_lambda must be non-negative")
        if self.policy_edge_scale < 0.0:
            raise ValueError("policy_edge_scale must be non-negative")
        if self.active_action_epsilon < 0.0:
            raise ValueError("active_action_epsilon must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cost_model_config"] = self.cost_model_config.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class Phase2EvaluationConfig:
    """Acceptance thresholds and output settings for Phase 2."""

    execution_config: ExecutionSimulationConfig = ExecutionSimulationConfig()
    min_sharpe_improvement_ratio: float = 0.10
    min_turnover_reduction_ratio: float = 0.0
    max_cost_to_abs_gross_return_ratio: float | None = None
    require_cost_aware_filter_or_turnover_reduction: bool = False
    output_dir: Path | str | None = None
    created_at: str = DEFAULT_PHASE2_CREATED_AT

    def __post_init__(self) -> None:
        if self.output_dir is not None and not isinstance(self.output_dir, Path):
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not math.isfinite(self.min_sharpe_improvement_ratio):
            raise ValueError("min_sharpe_improvement_ratio must be finite")
        if not math.isfinite(self.min_turnover_reduction_ratio):
            raise ValueError("min_turnover_reduction_ratio must be finite")
        if self.max_cost_to_abs_gross_return_ratio is not None and (
            not math.isfinite(self.max_cost_to_abs_gross_return_ratio)
            or self.max_cost_to_abs_gross_return_ratio < 0.0
        ):
            raise ValueError("max_cost_to_abs_gross_return_ratio must be finite and non-negative")
        if not self.created_at:
            raise ValueError("created_at is required")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["execution_config"] = self.execution_config.to_dict()
        payload["output_dir"] = None if self.output_dir is None else str(self.output_dir)
        return payload


@dataclass(frozen=True, slots=True)
class ExecutionFill:
    """One execution-adjusted shadow decision."""

    decision_ts: int
    source: str
    instrument_id: str
    raw_action: float
    adjusted_action: float
    fill_probability: float
    filled_action: float
    confidence: float
    score: float
    shadow_net_return: float
    gross_policy_return: float
    spread_cost: float
    fee_cost: float
    slippage_cost: float
    liquidity_impact_cost: float
    total_execution_cost: float
    risk_penalty: float
    turnover_penalty: float
    net_execution_return: float
    turnover: float
    estimated_policy_edge: float
    estimated_friction: float
    expected_net_edge: float
    low_ev_filtered: bool

    def __post_init__(self) -> None:
        for field_name in (
            "raw_action",
            "adjusted_action",
            "fill_probability",
            "filled_action",
            "confidence",
        ):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")
        for field_name in (
            "score",
            "shadow_net_return",
            "gross_policy_return",
            "spread_cost",
            "fee_cost",
            "slippage_cost",
            "liquidity_impact_cost",
            "total_execution_cost",
            "risk_penalty",
            "turnover_penalty",
            "net_execution_return",
            "turnover",
            "estimated_policy_edge",
            "estimated_friction",
            "expected_net_edge",
        ):
            if not math.isfinite(float(getattr(self, field_name))):
                raise ValueError(f"{field_name} must be finite")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Phase2EvaluationReport:
    """Auditable Phase 2 acceptance report."""

    phase: str
    candidate_run_id: str
    candidate_artifact_dir: str
    phase1_5_hashes: dict[str, str]
    phase1_5_shadow_acceptance_metrics: dict[str, Any]
    execution_metrics: dict[str, Any]
    comparison_metrics: dict[str, Any]
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
            "phase1_5_shadow_acceptance_metrics": self.phase1_5_shadow_acceptance_metrics,
            "execution_metrics": self.execution_metrics,
            "comparison_metrics": self.comparison_metrics,
            "acceptance_criteria": self.acceptance_criteria,
            "config": self.config,
            "created_at": self.created_at,
        }
