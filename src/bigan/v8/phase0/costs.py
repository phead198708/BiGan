"""Execution-realistic cost model for Phase 0 labels."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from bigan.v8.phase0.contracts import MarketData


@dataclass(frozen=True, slots=True)
class CostModelConfig:
    """Cost assumptions expressed in return units."""

    fee_bps: float = 8.0
    base_slippage_bps: float = 2.0
    volatility_slippage_factor: float = 0.25
    liquidity_impact_factor: float = 0.015
    minimum_liquidity: float = 1.0

    def __post_init__(self) -> None:
        if self.fee_bps < 0:
            raise ValueError("fee_bps must be non-negative")
        if self.base_slippage_bps < 0:
            raise ValueError("base_slippage_bps must be non-negative")
        if self.volatility_slippage_factor < 0:
            raise ValueError("volatility_slippage_factor must be non-negative")
        if self.liquidity_impact_factor < 0:
            raise ValueError("liquidity_impact_factor must be non-negative")
        if self.minimum_liquidity <= 0:
            raise ValueError("minimum_liquidity must be positive")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Cost components in return units."""

    spread_cost: float
    fee_cost: float
    slippage_cost: float
    liquidity_impact_cost: float

    @property
    def total_cost(self) -> float:
        return (
            self.spread_cost
            + self.fee_cost
            + self.slippage_cost
            + self.liquidity_impact_cost
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "spread_cost": self.spread_cost,
            "fee_cost": self.fee_cost,
            "slippage_cost": self.slippage_cost,
            "liquidity_impact_cost": self.liquidity_impact_cost,
            "total_cost": self.total_cost,
        }


@dataclass(frozen=True, slots=True)
class ExecutionCostSample:
    """Observed execution-cost sample used to validate the cost model."""

    entry: MarketData
    exit: MarketData | None
    observed_total_cost: float
    order_size: float = 1.0
    volatility: float | None = None

    def __post_init__(self) -> None:
        if self.observed_total_cost < 0:
            raise ValueError("observed_total_cost must be non-negative")
        if self.order_size <= 0:
            raise ValueError("order_size must be positive")
        if self.volatility is not None and self.volatility < 0:
            raise ValueError("volatility must be non-negative")


@dataclass(frozen=True, slots=True)
class CostCalibrationConfig:
    """Bounded-error thresholds for real-execution cost validation."""

    min_samples: int = 5
    max_mean_absolute_error: float = 0.0025
    max_mean_absolute_percentage_error: float = 0.35
    max_abs_bias: float = 0.0015

    def __post_init__(self) -> None:
        if self.min_samples <= 0:
            raise ValueError("min_samples must be positive")
        if self.max_mean_absolute_error < 0:
            raise ValueError("max_mean_absolute_error must be non-negative")
        if self.max_mean_absolute_percentage_error < 0:
            raise ValueError("max_mean_absolute_percentage_error must be non-negative")
        if self.max_abs_bias < 0:
            raise ValueError("max_abs_bias must be non-negative")


@dataclass(frozen=True, slots=True)
class CostCalibrationReport:
    """Cost-model error report against observed executions."""

    sample_count: int
    passed: bool
    estimated_mean_cost: float | None
    observed_mean_cost: float | None
    mean_absolute_error: float | None
    mean_absolute_percentage_error: float | None
    bias: float | None
    max_absolute_error: float | None

    def to_dict(self) -> dict[str, float | int | bool | None]:
        return {
            "sample_count": self.sample_count,
            "passed": self.passed,
            "estimated_mean_cost": self.estimated_mean_cost,
            "observed_mean_cost": self.observed_mean_cost,
            "mean_absolute_error": self.mean_absolute_error,
            "mean_absolute_percentage_error": self.mean_absolute_percentage_error,
            "bias": self.bias,
            "max_absolute_error": self.max_absolute_error,
        }


class TradingCostModel:
    """Spread, fee, volatility-slippage, and liquidity-impact costs."""

    def __init__(self, config: CostModelConfig | None = None) -> None:
        self.config = config or CostModelConfig()

    def estimate(
        self,
        *,
        entry: MarketData,
        exit: MarketData | None = None,
        order_size: float = 1.0,
        volatility: float | None = None,
        slippage_multiplier: float = 1.0,
    ) -> CostBreakdown:
        if order_size <= 0:
            raise ValueError("order_size must be positive")
        if slippage_multiplier <= 0:
            raise ValueError("slippage_multiplier must be positive")
        entry_mid = entry.effective_mid_price
        spread_cost = self._spread_cost(entry, exit)
        fee_cost = self.config.fee_bps / 10_000.0
        volatility_component = max(0.0, float(volatility or 0.0))
        slippage_cost = (
            self.config.base_slippage_bps / 10_000.0
            + volatility_component * self.config.volatility_slippage_factor
        ) * slippage_multiplier
        liquidity = entry.liquidity_depth
        if liquidity is None and entry.bid_size is not None and entry.ask_size is not None:
            liquidity = entry.bid_size + entry.ask_size
        effective_liquidity = max(self.config.minimum_liquidity, float(liquidity or 0.0))
        liquidity_impact_cost = self.config.liquidity_impact_factor * math.sqrt(
            order_size / effective_liquidity
        )
        if entry_mid <= 0:
            raise ValueError("entry mid price must be positive")
        return CostBreakdown(
            spread_cost=spread_cost,
            fee_cost=fee_cost,
            slippage_cost=slippage_cost,
            liquidity_impact_cost=liquidity_impact_cost,
        )

    def stress(
        self,
        *,
        entry: MarketData,
        exit: MarketData | None = None,
        order_size: float = 1.0,
        volatility: float | None = None,
        multipliers: tuple[float, ...] = (1.2, 1.5, 2.0),
    ) -> dict[float, CostBreakdown]:
        return {
            multiplier: self.estimate(
                entry=entry,
                exit=exit,
                order_size=order_size,
                volatility=volatility,
                slippage_multiplier=multiplier,
            )
            for multiplier in multipliers
        }

    def validate_calibration(
        self,
        samples: list[ExecutionCostSample],
        *,
        config: CostCalibrationConfig | None = None,
    ) -> CostCalibrationReport:
        """Validate cost estimates against observed execution costs."""

        calibration_config = config or CostCalibrationConfig()
        if len(samples) < calibration_config.min_samples:
            return CostCalibrationReport(
                sample_count=len(samples),
                passed=False,
                estimated_mean_cost=None,
                observed_mean_cost=None,
                mean_absolute_error=None,
                mean_absolute_percentage_error=None,
                bias=None,
                max_absolute_error=None,
            )

        estimates = [
            self.estimate(
                entry=sample.entry,
                exit=sample.exit,
                order_size=sample.order_size,
                volatility=sample.volatility,
            ).total_cost
            for sample in samples
        ]
        observed = [sample.observed_total_cost for sample in samples]
        errors = [
            estimate - actual
            for estimate, actual in zip(estimates, observed, strict=True)
        ]
        abs_errors = [abs(error) for error in errors]
        pct_errors = [
            abs(error) / max(actual, 1e-12)
            for error, actual in zip(errors, observed, strict=True)
        ]
        mean_absolute_error = sum(abs_errors) / len(abs_errors)
        mean_absolute_percentage_error = sum(pct_errors) / len(pct_errors)
        bias = sum(errors) / len(errors)
        passed = (
            mean_absolute_error <= calibration_config.max_mean_absolute_error
            and mean_absolute_percentage_error
            <= calibration_config.max_mean_absolute_percentage_error
            and abs(bias) <= calibration_config.max_abs_bias
        )
        return CostCalibrationReport(
            sample_count=len(samples),
            passed=passed,
            estimated_mean_cost=sum(estimates) / len(estimates),
            observed_mean_cost=sum(observed) / len(observed),
            mean_absolute_error=mean_absolute_error,
            mean_absolute_percentage_error=mean_absolute_percentage_error,
            bias=bias,
            max_absolute_error=max(abs_errors),
        )

    def _spread_cost(self, entry: MarketData, exit: MarketData | None) -> float:
        entry_cost = _spread_fraction(entry)
        exit_cost = _spread_fraction(exit) if exit is not None else entry_cost
        # Half spread to enter and half spread to exit.
        return 0.5 * entry_cost + 0.5 * exit_cost


def _spread_fraction(row: MarketData | None) -> float:
    if row is None or row.bid_price is None or row.ask_price is None:
        return 0.0
    mid = row.effective_mid_price
    if mid <= 0.0:
        return 0.0
    return max(0.0, row.ask_price - row.bid_price) / mid
