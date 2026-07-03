"""Phase 4 online adaptive-system contracts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from bigan.v8.phase0.costs import CostModelConfig

PHASE4_ADAPTIVE_SYSTEM_PHASE = "phase4_online_adaptive_system"
DEFAULT_PHASE4_CREATED_AT = "1970-01-01T00:00:00Z"

RegimeName = Literal["trend", "range", "high_volatility", "liquidity_stress"]


class Phase4AdaptiveError(RuntimeError):
    """Raised when Phase 4 receives unsafe inputs or cannot adapt safely."""


@dataclass(frozen=True, slots=True)
class Phase4InputProvenance:
    """Upstream artifact and stream hashes replayed by Phase 4."""

    candidate_run_id: str
    policy_dataset_hash: str
    split_hash: str
    model_sha256: str
    example_stream_sha256: str
    prediction_stream_sha256: str
    phase2_report_sha256: str | None = None
    phase3_report_sha256: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "candidate_run_id",
            "policy_dataset_hash",
            "split_hash",
            "model_sha256",
            "example_stream_sha256",
            "prediction_stream_sha256",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} is required")
        for field_name in (
            "policy_dataset_hash",
            "split_hash",
            "model_sha256",
            "example_stream_sha256",
            "prediction_stream_sha256",
            "phase2_report_sha256",
            "phase3_report_sha256",
        ):
            value = getattr(self, field_name)
            if value is not None and not _looks_like_sha256(value):
                raise ValueError(f"{field_name} must be a SHA-256 hex digest")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RegimeDetectorConfig:
    """Causal market-state thresholds for online regime classification."""

    trend_score_threshold: float = 0.55
    high_volatility_threshold: float = 0.05
    liquidity_stress_threshold: float = 10.0
    high_cost_spread_bps_threshold: float = 75.0
    transition_confirmation_count: int = 2
    volatility_smoothing_alpha: float = 0.15

    def __post_init__(self) -> None:
        if not 0.0 <= self.trend_score_threshold <= 1.0:
            raise ValueError("trend_score_threshold must be in [0, 1]")
        if self.high_volatility_threshold <= 0.0:
            raise ValueError("high_volatility_threshold must be positive")
        if self.liquidity_stress_threshold < 0.0:
            raise ValueError("liquidity_stress_threshold must be non-negative")
        if self.high_cost_spread_bps_threshold <= 0.0:
            raise ValueError("high_cost_spread_bps_threshold must be positive")
        if self.transition_confirmation_count <= 0:
            raise ValueError("transition_confirmation_count must be positive")
        if not 0.0 < self.volatility_smoothing_alpha <= 1.0:
            raise ValueError("volatility_smoothing_alpha must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LambdaControllerConfig:
    """Risk-appetite controller used by the adaptive overlay."""

    base_lambda: float = 0.25
    min_lambda: float = 0.02
    max_lambda: float = 0.80
    trend_multiplier: float = 1.35
    range_multiplier: float = 1.00
    high_volatility_multiplier: float = 0.45
    liquidity_stress_multiplier: float = 0.60
    volatility_sensitivity: float = 1.50
    drawdown_sensitivity: float = 3.00
    min_drawdown_multiplier: float = 0.25
    smoothing_alpha: float = 0.40
    max_step_change: float = 0.08

    def __post_init__(self) -> None:
        if self.base_lambda <= 0.0:
            raise ValueError("base_lambda must be positive")
        if self.min_lambda < 0.0:
            raise ValueError("min_lambda must be non-negative")
        if self.max_lambda <= self.min_lambda:
            raise ValueError("max_lambda must exceed min_lambda")
        if not self.min_lambda <= self.base_lambda <= self.max_lambda:
            raise ValueError("base_lambda must be within [min_lambda, max_lambda]")
        for field_name in (
            "trend_multiplier",
            "range_multiplier",
            "high_volatility_multiplier",
            "liquidity_stress_multiplier",
        ):
            if getattr(self, field_name) <= 0.0:
                raise ValueError(f"{field_name} must be positive")
        if self.volatility_sensitivity < 0.0:
            raise ValueError("volatility_sensitivity must be non-negative")
        if self.drawdown_sensitivity < 0.0:
            raise ValueError("drawdown_sensitivity must be non-negative")
        if not 0.0 <= self.min_drawdown_multiplier <= 1.0:
            raise ValueError("min_drawdown_multiplier must be in [0, 1]")
        if not 0.0 < self.smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        if self.max_step_change <= 0.0:
            raise ValueError("max_step_change must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExecutionAdaptationConfig:
    """Cost/liquidity-aware execution aggressiveness controller."""

    cost_model_config: CostModelConfig = CostModelConfig()
    slippage_multiplier: float = 1.0
    order_size: float = 1.0
    min_order_size: float = 1e-12
    min_fill_probability: float = 0.0
    base_aggressiveness: float = 1.0
    min_aggressiveness: float = 0.05
    max_aggressiveness: float = 1.0
    trend_multiplier: float = 1.10
    range_multiplier: float = 0.90
    high_volatility_multiplier: float = 0.45
    liquidity_stress_multiplier: float = 0.45
    cost_sensitivity: float = 0.35
    liquidity_sensitivity: float = 1.0
    smoothing_alpha: float = 0.45
    max_step_change: float = 0.12
    risk_penalty_factor: float = 0.05
    turnover_penalty_factor: float = 0.0
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
        if self.base_aggressiveness <= 0.0:
            raise ValueError("base_aggressiveness must be positive")
        if self.min_aggressiveness < 0.0:
            raise ValueError("min_aggressiveness must be non-negative")
        if self.max_aggressiveness <= self.min_aggressiveness:
            raise ValueError("max_aggressiveness must exceed min_aggressiveness")
        if not self.min_aggressiveness <= self.base_aggressiveness <= self.max_aggressiveness:
            raise ValueError(
                "base_aggressiveness must be within "
                "[min_aggressiveness, max_aggressiveness]"
            )
        for field_name in (
            "trend_multiplier",
            "range_multiplier",
            "high_volatility_multiplier",
            "liquidity_stress_multiplier",
        ):
            if getattr(self, field_name) <= 0.0:
                raise ValueError(f"{field_name} must be positive")
        if self.cost_sensitivity < 0.0:
            raise ValueError("cost_sensitivity must be non-negative")
        if self.liquidity_sensitivity < 0.0:
            raise ValueError("liquidity_sensitivity must be non-negative")
        if not 0.0 < self.smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        if self.max_step_change <= 0.0:
            raise ValueError("max_step_change must be positive")
        if self.risk_penalty_factor < 0.0:
            raise ValueError("risk_penalty_factor must be non-negative")
        if self.turnover_penalty_factor < 0.0:
            raise ValueError("turnover_penalty_factor must be non-negative")
        if self.active_action_epsilon < 0.0:
            raise ValueError("active_action_epsilon must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cost_model_config"] = self.cost_model_config.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class Phase4AdaptiveSystemConfig:
    """Phase 4 replay settings and acceptance thresholds."""

    detector_config: RegimeDetectorConfig = RegimeDetectorConfig()
    lambda_config: LambdaControllerConfig = LambdaControllerConfig()
    execution_config: ExecutionAdaptationConfig = ExecutionAdaptationConfig()
    tail_quantile: float = 0.05
    min_regime_stability_ratio: float = 0.85
    max_accepted_lambda_step: float = 0.10
    max_accepted_lambda_oscillation_rate: float = 0.15
    max_accepted_aggressiveness_step: float = 0.15
    min_tail_loss_reduction_ratio: float = 0.0
    max_raw_regime_transition_rate: float = 0.35
    max_pending_regime_rate: float = 0.50
    stress_volatility_multipliers: tuple[float, ...] = (1.2, 1.5, 2.0)
    stress_drawdown_shock: float = 0.0
    output_dir: Path | str | None = None
    created_at: str = DEFAULT_PHASE4_CREATED_AT

    def __post_init__(self) -> None:
        if self.output_dir is not None and not isinstance(self.output_dir, Path):
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not 0.0 < self.tail_quantile < 0.5:
            raise ValueError("tail_quantile must be in (0, 0.5)")
        if not 0.0 <= self.min_regime_stability_ratio <= 1.0:
            raise ValueError("min_regime_stability_ratio must be in [0, 1]")
        if self.max_accepted_lambda_step <= 0.0:
            raise ValueError("max_accepted_lambda_step must be positive")
        if not 0.0 <= self.max_accepted_lambda_oscillation_rate <= 1.0:
            raise ValueError("max_accepted_lambda_oscillation_rate must be in [0, 1]")
        if self.max_accepted_aggressiveness_step <= 0.0:
            raise ValueError("max_accepted_aggressiveness_step must be positive")
        if not math.isfinite(self.min_tail_loss_reduction_ratio):
            raise ValueError("min_tail_loss_reduction_ratio must be finite")
        if not 0.0 <= self.max_raw_regime_transition_rate <= 1.0:
            raise ValueError("max_raw_regime_transition_rate must be in [0, 1]")
        if not 0.0 <= self.max_pending_regime_rate <= 1.0:
            raise ValueError("max_pending_regime_rate must be in [0, 1]")
        if not self.stress_volatility_multipliers:
            raise ValueError("stress_volatility_multipliers must not be empty")
        for multiplier in self.stress_volatility_multipliers:
            if multiplier <= 0.0 or not math.isfinite(multiplier):
                raise ValueError(
                    "stress_volatility_multipliers must be positive finite values"
                )
        if self.stress_drawdown_shock < 0.0:
            raise ValueError("stress_drawdown_shock must be non-negative")
        if not self.created_at:
            raise ValueError("created_at is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector_config": self.detector_config.to_dict(),
            "lambda_config": self.lambda_config.to_dict(),
            "execution_config": self.execution_config.to_dict(),
            "tail_quantile": self.tail_quantile,
            "min_regime_stability_ratio": self.min_regime_stability_ratio,
            "max_accepted_lambda_step": self.max_accepted_lambda_step,
            "max_accepted_lambda_oscillation_rate": (
                self.max_accepted_lambda_oscillation_rate
            ),
            "max_accepted_aggressiveness_step": (
                self.max_accepted_aggressiveness_step
            ),
            "min_tail_loss_reduction_ratio": self.min_tail_loss_reduction_ratio,
            "max_raw_regime_transition_rate": self.max_raw_regime_transition_rate,
            "max_pending_regime_rate": self.max_pending_regime_rate,
            "stress_volatility_multipliers": list(self.stress_volatility_multipliers),
            "stress_drawdown_shock": self.stress_drawdown_shock,
            "output_dir": None if self.output_dir is None else str(self.output_dir),
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class RegimeClassification:
    """Causal regime decision emitted for one policy row."""

    raw_regime: RegimeName
    regime: RegimeName
    trend_score: float
    volatility: float
    liquidity_depth: float
    spread_bps: float
    confirmation_count: int
    pending_regime_active: bool
    transitioned: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AdaptiveDecision:
    """One online-adaptive execution decision."""

    decision_ts: int
    source: str
    instrument_id: str
    raw_action: float
    adapted_action: float
    filled_action: float
    confidence: float
    score: float
    regime: RegimeName
    raw_regime: RegimeName
    pending_regime_active: bool
    transitioned: bool
    lambda_value: float
    execution_aggressiveness: float
    fill_probability: float
    turnover: float
    shadow_net_return: float
    gross_return: float
    spread_cost: float
    fee_cost: float
    slippage_cost: float
    liquidity_impact_cost: float
    total_execution_cost: float
    risk_penalty: float
    turnover_penalty: float
    net_return: float
    baseline_net_return: float
    drawdown: float

    def __post_init__(self) -> None:
        for field_name in (
            "raw_action",
            "adapted_action",
            "filled_action",
            "confidence",
            "fill_probability",
        ):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")
        for field_name in (
            "score",
            "lambda_value",
            "execution_aggressiveness",
            "turnover",
            "shadow_net_return",
            "gross_return",
            "spread_cost",
            "fee_cost",
            "slippage_cost",
            "liquidity_impact_cost",
            "total_execution_cost",
            "risk_penalty",
            "turnover_penalty",
            "net_return",
            "baseline_net_return",
            "drawdown",
        ):
            if not math.isfinite(float(getattr(self, field_name))):
                raise ValueError(f"{field_name} must be finite")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Phase4AdaptiveSystemReport:
    """Auditable Phase 4 adaptive-system replay report."""

    phase: str
    row_count: int
    candidate_run_id: str | None
    policy_dataset_hash: str | None
    split_hash: str | None
    model_sha256: str | None
    phase2_report_sha256: str | None
    phase3_report_sha256: str | None
    example_stream_sha256: str
    prediction_stream_sha256: str
    input_provenance_verified: bool
    baseline_type: str
    baseline_execution_config_sha256: str
    decision_trace_sha256: str
    decision_count: int
    first_decision_ts: int | None
    last_decision_ts: int | None
    adaptive_metrics: dict[str, Any]
    baseline_metrics: dict[str, Any]
    comparison_metrics: dict[str, Any]
    stress_metrics: dict[str, dict[str, Any]]
    regime_counts: dict[str, int]
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
            "row_count": self.row_count,
            "candidate_run_id": self.candidate_run_id,
            "policy_dataset_hash": self.policy_dataset_hash,
            "split_hash": self.split_hash,
            "model_sha256": self.model_sha256,
            "phase2_report_sha256": self.phase2_report_sha256,
            "phase3_report_sha256": self.phase3_report_sha256,
            "example_stream_sha256": self.example_stream_sha256,
            "prediction_stream_sha256": self.prediction_stream_sha256,
            "input_provenance_verified": self.input_provenance_verified,
            "baseline_type": self.baseline_type,
            "baseline_execution_config_sha256": (
                self.baseline_execution_config_sha256
            ),
            "decision_trace_sha256": self.decision_trace_sha256,
            "decision_count": self.decision_count,
            "first_decision_ts": self.first_decision_ts,
            "last_decision_ts": self.last_decision_ts,
            "adaptive_metrics": self.adaptive_metrics,
            "baseline_metrics": self.baseline_metrics,
            "comparison_metrics": self.comparison_metrics,
            "stress_metrics": self.stress_metrics,
            "regime_counts": self.regime_counts,
            "acceptance_criteria": self.acceptance_criteria,
            "config": self.config,
            "created_at": self.created_at,
        }


def _looks_like_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())
