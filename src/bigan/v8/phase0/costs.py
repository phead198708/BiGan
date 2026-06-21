"""Execution-realistic cost model for Phase 0 labels."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
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
    percentage_error_floor: float = 1e-6
    max_weighted_mean_absolute_percentage_error: float | None = None
    max_median_absolute_percentage_error: float | None = None

    def __post_init__(self) -> None:
        if self.min_samples <= 0:
            raise ValueError("min_samples must be positive")
        if self.max_mean_absolute_error < 0:
            raise ValueError("max_mean_absolute_error must be non-negative")
        if self.max_mean_absolute_percentage_error < 0:
            raise ValueError("max_mean_absolute_percentage_error must be non-negative")
        if self.max_abs_bias < 0:
            raise ValueError("max_abs_bias must be non-negative")
        if self.percentage_error_floor <= 0:
            raise ValueError("percentage_error_floor must be positive")
        if (
            self.max_weighted_mean_absolute_percentage_error is not None
            and self.max_weighted_mean_absolute_percentage_error < 0
        ):
            raise ValueError("max_weighted_mean_absolute_percentage_error must be non-negative")
        if (
            self.max_median_absolute_percentage_error is not None
            and self.max_median_absolute_percentage_error < 0
        ):
            raise ValueError("max_median_absolute_percentage_error must be non-negative")


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
    weighted_mean_absolute_percentage_error: float | None = None
    median_absolute_error: float | None = None
    median_absolute_percentage_error: float | None = None
    symmetric_mean_absolute_percentage_error: float | None = None

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
            "weighted_mean_absolute_percentage_error": (
                self.weighted_mean_absolute_percentage_error
            ),
            "median_absolute_error": self.median_absolute_error,
            "median_absolute_percentage_error": self.median_absolute_percentage_error,
            "symmetric_mean_absolute_percentage_error": (
                self.symmetric_mean_absolute_percentage_error
            ),
        }


@dataclass(frozen=True, slots=True)
class CostCalibrationBucketConfig:
    """Bucket definitions for regime-aware cost calibration."""

    min_bucket_samples: int = 5
    bucket_by_source: bool = True
    bucket_by_instrument: bool = True
    volatility_edges: tuple[float, ...] = (0.001, 0.005, 0.01, 0.02)
    spread_edges: tuple[float, ...] = (0.0005, 0.001, 0.0025, 0.005)
    liquidity_edges: tuple[float, ...] = (25.0, 100.0, 500.0, 1_000.0)
    order_size_edges: tuple[float, ...] = (1.0, 5.0, 20.0, 100.0)

    def __post_init__(self) -> None:
        if self.min_bucket_samples <= 0:
            raise ValueError("min_bucket_samples must be positive")
        for name, edges in (
            ("volatility_edges", self.volatility_edges),
            ("spread_edges", self.spread_edges),
            ("liquidity_edges", self.liquidity_edges),
            ("order_size_edges", self.order_size_edges),
        ):
            if tuple(sorted(edges)) != edges:
                raise ValueError(f"{name} must be sorted ascending")
            if any(edge < 0 for edge in edges):
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True, slots=True)
class CostCalibrationBucketReport:
    """Aggregate and per-regime cost-calibration result."""

    aggregate: CostCalibrationReport
    buckets: dict[str, CostCalibrationReport]
    skipped_buckets: tuple[str, ...]
    failed_buckets: tuple[str, ...]
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "aggregate": self.aggregate.to_dict(),
            "buckets": {
                key: report.to_dict()
                for key, report in sorted(self.buckets.items())
            },
            "skipped_buckets": list(self.skipped_buckets),
            "failed_buckets": list(self.failed_buckets),
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
            abs(error) / max(abs(actual), calibration_config.percentage_error_floor)
            for error, actual in zip(errors, observed, strict=True)
        ]
        mean_absolute_error = sum(abs_errors) / len(abs_errors)
        mean_absolute_percentage_error = sum(pct_errors) / len(pct_errors)
        weighted_mean_absolute_percentage_error = sum(abs_errors) / max(
            sum(abs(actual) for actual in observed),
            calibration_config.percentage_error_floor,
        )
        median_absolute_error = statistics.median(abs_errors)
        median_absolute_percentage_error = statistics.median(pct_errors)
        symmetric_pct_errors = [
            abs(error)
            / max(
                (abs(estimate) + abs(actual)) / 2.0,
                calibration_config.percentage_error_floor,
            )
            for error, estimate, actual in zip(errors, estimates, observed, strict=True)
        ]
        symmetric_mean_absolute_percentage_error = (
            sum(symmetric_pct_errors) / len(symmetric_pct_errors)
        )
        bias = sum(errors) / len(errors)
        passed = (
            mean_absolute_error <= calibration_config.max_mean_absolute_error
            and mean_absolute_percentage_error
            <= calibration_config.max_mean_absolute_percentage_error
            and abs(bias) <= calibration_config.max_abs_bias
        )
        if calibration_config.max_weighted_mean_absolute_percentage_error is not None:
            passed = (
                passed
                and weighted_mean_absolute_percentage_error
                <= calibration_config.max_weighted_mean_absolute_percentage_error
            )
        if calibration_config.max_median_absolute_percentage_error is not None:
            passed = (
                passed
                and median_absolute_percentage_error
                <= calibration_config.max_median_absolute_percentage_error
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
            weighted_mean_absolute_percentage_error=weighted_mean_absolute_percentage_error,
            median_absolute_error=median_absolute_error,
            median_absolute_percentage_error=median_absolute_percentage_error,
            symmetric_mean_absolute_percentage_error=symmetric_mean_absolute_percentage_error,
        )

    def validate_calibration_by_bucket(
        self,
        samples: list[ExecutionCostSample],
        *,
        bucket_config: CostCalibrationBucketConfig | None = None,
        config: CostCalibrationConfig | None = None,
    ) -> CostCalibrationBucketReport:
        """Validate aggregate and sufficiently sampled cost regimes."""

        calibration_config = config or CostCalibrationConfig()
        buckets_config = bucket_config or CostCalibrationBucketConfig()
        aggregate = self.validate_calibration(samples, config=calibration_config)
        grouped: dict[str, list[ExecutionCostSample]] = defaultdict(list)
        for sample in samples:
            grouped[_bucket_key(sample, buckets_config)].append(sample)

        bucket_reports: dict[str, CostCalibrationReport] = {}
        skipped_buckets: list[str] = []
        failed_buckets: list[str] = []
        for bucket, bucket_samples in sorted(grouped.items()):
            if len(bucket_samples) < buckets_config.min_bucket_samples:
                skipped_buckets.append(bucket)
                continue
            report = self.validate_calibration(bucket_samples, config=calibration_config)
            bucket_reports[bucket] = report
            if not report.passed:
                failed_buckets.append(bucket)

        return CostCalibrationBucketReport(
            aggregate=aggregate,
            buckets=bucket_reports,
            skipped_buckets=tuple(skipped_buckets),
            failed_buckets=tuple(failed_buckets),
            passed=aggregate.passed and not failed_buckets,
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


def _bucket_key(
    sample: ExecutionCostSample,
    config: CostCalibrationBucketConfig,
) -> str:
    parts: list[str] = []
    if config.bucket_by_source:
        parts.append(f"source={sample.entry.source}")
    if config.bucket_by_instrument:
        parts.append(f"instrument={sample.entry.instrument_id}")
    parts.append(
        "volatility="
        + _bucket_label(sample.volatility or 0.0, config.volatility_edges)
    )
    parts.append(
        "spread="
        + _bucket_label(_spread_fraction(sample.entry), config.spread_edges)
    )
    liquidity = sample.entry.liquidity_depth
    if liquidity is None and sample.entry.bid_size is not None and sample.entry.ask_size is not None:
        liquidity = sample.entry.bid_size + sample.entry.ask_size
    parts.append(
        "liquidity="
        + _bucket_label(float(liquidity or 0.0), config.liquidity_edges)
    )
    parts.append(
        "order_size="
        + _bucket_label(sample.order_size, config.order_size_edges)
    )
    return "|".join(parts)


def _bucket_label(value: float, edges: tuple[float, ...]) -> str:
    if not edges:
        return "all"
    previous = 0.0
    for edge in edges:
        if value < edge:
            return f"[{previous:g},{edge:g})"
        previous = edge
    return f">={edges[-1]:g}"
