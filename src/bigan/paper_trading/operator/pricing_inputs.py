"""Bounded dynamic pricing inputs from independent spot and oracle sources."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from bigan.pipeline.strategy_runner import PricingInputs


@dataclass(frozen=True, slots=True)
class ReferencePriceSample:
    timestamp_ms: int
    received_at_ms: int
    price: float
    source: str

    def validate(self) -> None:
        if self.timestamp_ms < 0:
            raise ValueError("reference timestamp must be non-negative")
        if self.received_at_ms < self.timestamp_ms:
            raise ValueError("received timestamp cannot precede source timestamp")
        if not math.isfinite(float(self.price)) or self.price <= 0.0:
            raise ValueError("reference price must be positive and finite")
        if not self.source.strip():
            raise ValueError("reference source must be non-empty")


@dataclass(frozen=True, slots=True)
class PricingProviderHealth:
    ready: bool
    fresh: bool
    timestamp_ms: int | None
    age_ms: int | None
    spot_sample_count: int
    oracle_sample_count: int
    return_sample_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "fresh": self.fresh,
            "timestamp_ms": self.timestamp_ms,
            "age_ms": self.age_ms,
            "spot_sample_count": self.spot_sample_count,
            "oracle_sample_count": self.oracle_sample_count,
            "return_sample_count": self.return_sample_count,
        }


class RollingPricingInputsProvider:
    """Point-in-time spot, oracle TWAP, and annualized realized volatility.

    Spot and oracle samples are independent. Oracle TWAP is the arithmetic
    mean of accepted event-time samples in ``twap_window_ms``. Volatility uses
    non-overlapping-at-least-interval log returns and population variance,
    annualized by ``sqrt(annualization_seconds / return_interval_seconds)``.
    Buffers are bounded and reconnect explicitly clears warm state.
    """

    def __init__(
        self,
        *,
        window_start_ts_ms: int,
        window_end_ts_ms: int,
        spot_source: str,
        oracle_source: str,
        max_age_ms: int,
        max_samples: int,
        twap_window_ms: int,
        return_interval_ms: int,
        volatility_window_ms: int,
        volatility_min_samples: int,
        volatility_max_abs_log_return: float,
        annualization_seconds: int,
    ) -> None:
        if window_end_ts_ms <= window_start_ts_ms:
            raise ValueError("pricing window end must be after start")
        if not spot_source.strip() or not oracle_source.strip():
            raise ValueError("pricing sources must be non-empty")
        integer_bounds = (
            max_age_ms,
            max_samples,
            twap_window_ms,
            return_interval_ms,
            volatility_window_ms,
            volatility_min_samples,
            annualization_seconds,
        )
        if any(value <= 0 for value in integer_bounds):
            raise ValueError("pricing buffer, interval, and freshness bounds must be positive")
        if not 0.0 < volatility_max_abs_log_return <= 1.0:
            raise ValueError("volatility outlier bound must be in (0, 1]")
        self.window_start_ts_ms = int(window_start_ts_ms)
        self.window_end_ts_ms = int(window_end_ts_ms)
        self.spot_source = spot_source
        self.oracle_source = oracle_source
        self.max_age_ms = int(max_age_ms)
        self.max_samples = int(max_samples)
        self.twap_window_ms = int(twap_window_ms)
        self.return_interval_ms = int(return_interval_ms)
        self.volatility_window_ms = int(volatility_window_ms)
        self.volatility_min_samples = int(volatility_min_samples)
        self.volatility_max_abs_log_return = float(volatility_max_abs_log_return)
        self.annualization_seconds = int(annualization_seconds)
        self._spot_samples: deque[ReferencePriceSample] = deque(maxlen=max_samples)
        self._oracle_samples: deque[ReferencePriceSample] = deque(maxlen=max_samples)
        self._returns: deque[tuple[int, float]] = deque(maxlen=max_samples)
        self._last_volatility_sample: ReferencePriceSample | None = None
        self.source_mismatch_count = 0
        self.out_of_order_count = 0
        self.outlier_count = 0
        self.future_input_count = 0
        self.stale_input_count = 0
        self.missing_input_count = 0

    @property
    def spot_sample_count(self) -> int:
        return len(self._spot_samples)

    @property
    def oracle_sample_count(self) -> int:
        return len(self._oracle_samples)

    @property
    def return_sample_count(self) -> int:
        return len(self._returns)

    @property
    def last_spot_timestamp_ms(self) -> int | None:
        return None if not self._spot_samples else self._spot_samples[-1].timestamp_ms

    @property
    def last_oracle_timestamp_ms(self) -> int | None:
        return None if not self._oracle_samples else self._oracle_samples[-1].timestamp_ms

    def config_identity(self) -> dict[str, object]:
        return {
            "window_start_ts_ms": self.window_start_ts_ms,
            "window_end_ts_ms": self.window_end_ts_ms,
            "spot_source": self.spot_source,
            "oracle_source": self.oracle_source,
            "max_age_ms": self.max_age_ms,
            "max_samples": self.max_samples,
            "twap_window_ms": self.twap_window_ms,
            "return_interval_ms": self.return_interval_ms,
            "volatility_window_ms": self.volatility_window_ms,
            "volatility_min_samples": self.volatility_min_samples,
            "volatility_max_abs_log_return": self.volatility_max_abs_log_return,
            "annualization_seconds": self.annualization_seconds,
            "twap_sampling": "event_time_arithmetic_mean",
            "volatility_returns": "log_non_overlapping_min_interval",
            "volatility_variance": "population",
        }

    def ingest_spot(self, sample: ReferencePriceSample) -> bool:
        sample.validate()
        if sample.source != self.spot_source:
            self.source_mismatch_count += 1
            return False
        if self._spot_samples and sample.timestamp_ms <= self._spot_samples[-1].timestamp_ms:
            self.out_of_order_count += 1
            return False
        volatility_base = self._last_volatility_sample
        if (
            volatility_base is not None
            and sample.timestamp_ms - volatility_base.timestamp_ms >= self.return_interval_ms
        ):
            log_return = math.log(sample.price / volatility_base.price)
            if abs(log_return) > self.volatility_max_abs_log_return:
                self.outlier_count += 1
                return False
            self._returns.append((sample.timestamp_ms, log_return))
            self._last_volatility_sample = sample
            self._evict_returns(sample.timestamp_ms)
        elif volatility_base is None:
            self._last_volatility_sample = sample
        self._spot_samples.append(sample)
        return True

    def ingest_oracle(self, sample: ReferencePriceSample) -> bool:
        sample.validate()
        if sample.source != self.oracle_source:
            self.source_mismatch_count += 1
            return False
        if self._oracle_samples and sample.timestamp_ms <= self._oracle_samples[-1].timestamp_ms:
            self.out_of_order_count += 1
            return False
        self._oracle_samples.append(sample)
        return True

    def reset_for_reconnect(self) -> None:
        """Discard unprovable rolling state; callers must warm up again."""

        self._spot_samples.clear()
        self._oracle_samples.clear()
        self._returns.clear()
        self._last_volatility_sample = None

    def __call__(self, decision_ts_ms: int) -> PricingInputs | None:
        if not self._spot_samples or not self._oracle_samples:
            self.missing_input_count += 1
            return None
        spot = self._spot_samples[-1]
        oracle = self._oracle_samples[-1]
        source_ts = min(spot.timestamp_ms, oracle.timestamp_ms)
        if spot.timestamp_ms > decision_ts_ms or oracle.timestamp_ms > decision_ts_ms:
            self.future_input_count += 1
            return None
        age_ms = max(
            decision_ts_ms - spot.timestamp_ms,
            decision_ts_ms - oracle.timestamp_ms,
        )
        if age_ms > self.max_age_ms:
            self.stale_input_count += 1
            return None
        self._evict_returns(decision_ts_ms)
        if len(self._returns) < self.volatility_min_samples:
            self.missing_input_count += 1
            return None
        oracle_cutoff = decision_ts_ms - self.twap_window_ms
        oracle_prices = [
            sample.price
            for sample in self._oracle_samples
            if oracle_cutoff <= sample.timestamp_ms <= decision_ts_ms
        ]
        if not oracle_prices:
            self.missing_input_count += 1
            return None
        twap = math.fsum(oracle_prices) / len(oracle_prices)
        volatility = self._annualized_volatility()
        progress = (decision_ts_ms - self.window_start_ts_ms) / (
            self.window_end_ts_ms - self.window_start_ts_ms
        )
        return PricingInputs(
            timestamp_ms=source_ts,
            spot_price=spot.price,
            oracle_twap_so_far=twap,
            twap_weight=min(1.0, max(0.0, progress)),
            volatility_annualized=volatility,
        )

    def health(self, *, now_ms: int) -> PricingProviderHealth:
        latest = (
            None
            if not self._spot_samples or not self._oracle_samples
            else min(
                self._spot_samples[-1].timestamp_ms,
                self._oracle_samples[-1].timestamp_ms,
            )
        )
        age = None if latest is None else now_ms - latest
        ready = bool(
            latest is not None and len(self._returns) >= self.volatility_min_samples
        )
        fresh = bool(ready and age is not None and 0 <= age <= self.max_age_ms)
        return PricingProviderHealth(
            ready=ready,
            fresh=fresh,
            timestamp_ms=latest,
            age_ms=age,
            spot_sample_count=len(self._spot_samples),
            oracle_sample_count=len(self._oracle_samples),
            return_sample_count=len(self._returns),
        )

    def _evict_returns(self, now_ms: int) -> None:
        cutoff = now_ms - self.volatility_window_ms
        while self._returns and self._returns[0][0] < cutoff:
            self._returns.popleft()

    def _annualized_volatility(self) -> float:
        values = [value for _, value in self._returns]
        mean = math.fsum(values) / len(values)
        variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
        periods_per_year = self.annualization_seconds * 1_000 / self.return_interval_ms
        result = math.sqrt(max(0.0, variance)) * math.sqrt(periods_per_year)
        if not math.isfinite(result):
            raise ValueError("annualized volatility is non-finite")
        return result
