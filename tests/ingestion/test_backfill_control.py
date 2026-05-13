"""Tests for backfill backpressure controls (issue #28)."""

from __future__ import annotations

import asyncio

import pytest

from bigan.ingestion.backfill_control import (
    BackfillCircuitOpen,
    BackfillControlConfig,
    BackfillCoordinator,
    CircuitBreaker,
    CircuitState,
    TokenBucketRateLimiter,
)


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay


def test_token_bucket_waits_after_burst_is_exhausted() -> None:
    clock = _FakeClock()
    limiter = TokenBucketRateLimiter(
        rate_per_second=2.0,
        burst=2,
        clock=clock,
        sleep=clock.sleep,
    )

    async def go() -> list[bool]:
        return [
            await limiter.acquire(),
            await limiter.acquire(),
            await limiter.acquire(),
        ]

    waited = asyncio.run(go())
    assert waited == [False, False, True]
    assert clock.sleeps == [0.5]


def test_circuit_breaker_opens_and_half_open_success_closes() -> None:
    clock = _FakeClock()
    breaker = CircuitBreaker(
        failure_threshold=2,
        cool_down_seconds=30.0,
        clock=clock,
    )

    assert breaker.before_request() is True
    breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED

    assert breaker.before_request() is True
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    assert breaker.before_request() is False

    clock.now = 30.0
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.before_request() is True
    breaker.record_success()
    assert breaker.state is CircuitState.CLOSED


def test_backfill_coordinator_caps_concurrency() -> None:
    current = 0
    max_seen = 0
    started = asyncio.Event()

    async def go() -> None:
        coordinator = BackfillCoordinator(
            BackfillControlConfig(
                max_concurrency=2,
                rate_limit_per_second=1000.0,
                circuit_failure_threshold=100,
            )
        )

        async def operation(_before_rest_call):  # type: ignore[no-untyped-def]
            nonlocal current, max_seen
            current += 1
            max_seen = max(max_seen, current)
            started.set()
            await asyncio.sleep(0.01)
            current -= 1
            return "ok"

        await asyncio.gather(
            *[
                coordinator.run(asset_id=f"tok-{i}", operation=operation)
                for i in range(10)
            ]
        )

    asyncio.run(go())
    assert started.is_set()
    assert max_seen == 2


def test_backfill_coordinator_opens_circuit_after_failures() -> None:
    calls = 0

    async def go() -> None:
        nonlocal calls
        coordinator = BackfillCoordinator(
            BackfillControlConfig(
                max_concurrency=2,
                rate_limit_per_second=1000.0,
                circuit_failure_threshold=2,
                circuit_cool_down_seconds=30.0,
            )
        )

        async def failing(_before_rest_call):  # type: ignore[no-untyped-def]
            nonlocal calls
            calls += 1
            raise RuntimeError("REST down")

        with pytest.raises(RuntimeError):
            await coordinator.run(asset_id="tok-1", operation=failing)
        with pytest.raises(RuntimeError):
            await coordinator.run(asset_id="tok-2", operation=failing)
        with pytest.raises(BackfillCircuitOpen):
            await coordinator.run(asset_id="tok-3", operation=failing)

    asyncio.run(go())
    assert calls == 2


def test_backfill_coordinator_half_open_probe_success_recovers() -> None:
    clock = _FakeClock()
    calls = 0

    async def go() -> None:
        nonlocal calls
        coordinator = BackfillCoordinator(
            BackfillControlConfig(
                max_concurrency=1,
                rate_limit_per_second=1000.0,
                circuit_failure_threshold=1,
                circuit_cool_down_seconds=30.0,
            ),
            clock=clock,
            sleep=clock.sleep,
        )

        async def failing(_before_rest_call):  # type: ignore[no-untyped-def]
            nonlocal calls
            calls += 1
            raise RuntimeError("REST down")

        async def success(_before_rest_call):  # type: ignore[no-untyped-def]
            nonlocal calls
            calls += 1
            return "ok"

        with pytest.raises(RuntimeError):
            await coordinator.run(asset_id="tok-1", operation=failing)
        with pytest.raises(BackfillCircuitOpen):
            await coordinator.run(asset_id="tok-2", operation=success)
        clock.now = 30.0
        assert await coordinator.run(asset_id="tok-3", operation=success) == "ok"
        assert coordinator.circuit_state is CircuitState.CLOSED

    asyncio.run(go())
    assert calls == 2
