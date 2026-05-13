"""Backfill backpressure and circuit breaking controls (issue #28)."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import TypeVar

from .metrics import (
    BACKFILL_CIRCUIT_STATE,
    BACKFILL_IN_FLIGHT,
    BACKFILL_THROTTLED_TOTAL,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(IntEnum):
    CLOSED = 0
    OPEN = 1
    HALF_OPEN = 2


class BackfillCircuitOpen(RuntimeError):
    """Raised when a backfill request is skipped by the open circuit."""


@dataclass(frozen=True, slots=True)
class BackfillControlConfig:
    max_concurrency: int = 4
    rate_limit_per_second: float = 10.0
    circuit_failure_threshold: int = 5
    circuit_cool_down_seconds: float = 30.0


class CircuitBreaker:
    """Small state machine: closed -> open -> half-open -> closed/open."""

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        cool_down_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if cool_down_seconds < 0:
            raise ValueError("cool_down_seconds must be non-negative")
        self._failure_threshold = failure_threshold
        self._cool_down_seconds = cool_down_seconds
        self._clock = clock
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at: float | None = None
        self._half_open_probe_in_flight = False

    @property
    def state(self) -> CircuitState:
        self._refresh_state()
        return self._state

    def before_request(self) -> bool:
        self._refresh_state()
        if self._state is CircuitState.OPEN:
            return False
        if self._state is CircuitState.HALF_OPEN:
            if self._half_open_probe_in_flight:
                return False
            self._half_open_probe_in_flight = True
        return True

    def record_success(self) -> None:
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at = None
        self._half_open_probe_in_flight = False

    def record_failure(self) -> None:
        self._half_open_probe_in_flight = False
        if self._state is CircuitState.HALF_OPEN:
            self._open()
            return
        self._failures += 1
        if self._failures >= self._failure_threshold:
            self._open()

    def _open(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = self._clock()
        self._failures = self._failure_threshold

    def _refresh_state(self) -> None:
        if self._state is not CircuitState.OPEN or self._opened_at is None:
            return
        if self._clock() - self._opened_at >= self._cool_down_seconds:
            self._state = CircuitState.HALF_OPEN
            self._half_open_probe_in_flight = False


class TokenBucketRateLimiter:
    """Async token bucket used before REST calls."""

    def __init__(
        self,
        *,
        rate_per_second: float = 10.0,
        burst: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be positive")
        self._rate = rate_per_second
        self._capacity = float(burst if burst is not None else max(1, int(rate_per_second)))
        self._tokens = self._capacity
        self._updated_at = clock()
        self._clock = clock
        self._sleep = sleep
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Acquire one token. Returns True when the caller had to wait."""

        waited = False
        async with self._lock:
            self._refill()
            if self._tokens < 1.0:
                delay = (1.0 - self._tokens) / self._rate
                waited = True
                await self._sleep(delay)
                self._refill()
            self._tokens = max(0.0, self._tokens - 1.0)
        return waited

    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._updated_at)
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._updated_at = now


class BackfillCoordinator:
    """Coordinates concurrency, rate limiting, and circuit state."""

    def __init__(
        self,
        config: BackfillControlConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if config.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        self._rate_limiter = TokenBucketRateLimiter(
            rate_per_second=config.rate_limit_per_second,
            clock=clock,
            sleep=sleep,
        )
        self._circuit = CircuitBreaker(
            failure_threshold=config.circuit_failure_threshold,
            cool_down_seconds=config.circuit_cool_down_seconds,
            clock=clock,
        )
        self._update_circuit_metric()

    @property
    def circuit_state(self) -> CircuitState:
        state = self._circuit.state
        self._update_circuit_metric()
        return state

    async def acquire_rest_slot(self) -> None:
        waited = await self._rate_limiter.acquire()
        if waited:
            BACKFILL_THROTTLED_TOTAL.labels(reason="rate_limiter").inc()

    async def run(
        self,
        *,
        asset_id: str,
        operation: Callable[[Callable[[], Awaitable[None]]], Awaitable[T]],
        is_failure: Callable[[T], bool] = lambda _: False,
    ) -> T:
        if not self._circuit.before_request():
            self._update_circuit_metric()
            BACKFILL_THROTTLED_TOTAL.labels(reason="circuit_open").inc()
            logger.warning("backfill.circuit_open", extra={"asset_id": asset_id})
            raise BackfillCircuitOpen("backfill circuit is open")

        self._update_circuit_metric()
        if self._semaphore.locked():
            BACKFILL_THROTTLED_TOTAL.labels(reason="semaphore").inc()

        async with self._semaphore:
            BACKFILL_IN_FLIGHT.inc()
            try:
                result = await operation(self.acquire_rest_slot)
            except Exception:
                self._circuit.record_failure()
                self._update_circuit_metric()
                raise
            finally:
                BACKFILL_IN_FLIGHT.dec()

        if is_failure(result):
            self._circuit.record_failure()
        else:
            self._circuit.record_success()
        self._update_circuit_metric()
        return result

    def _update_circuit_metric(self) -> None:
        BACKFILL_CIRCUIT_STATE.set(float(self._circuit.state))
