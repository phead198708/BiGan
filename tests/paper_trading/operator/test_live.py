from __future__ import annotations

import asyncio

import pytest

from bigan.paper_trading.operator.live import LiveFeedSupervisor
from bigan.paper_trading.operator.read_model import OperatorState


async def test_supervisor_always_shuts_down_after_unexpected_failure() -> None:
    class FailingOperator:
        def __init__(self) -> None:
            self.shutdown_called = False

        async def start(self) -> None:
            raise RuntimeError("unexpected supervisor failure")

        async def shutdown(self) -> None:
            self.shutdown_called = True

    operator = FailingOperator()
    supervisor = LiveFeedSupervisor(operator=operator)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="unexpected supervisor failure"):
        await supervisor.run(asyncio.Event())

    assert operator.shutdown_called is True


async def test_exhausted_operator_stops_supervisor_without_starting_feeds() -> None:
    class ExhaustedOperator:
        state = OperatorState.EXHAUSTED
        shutdown_called = False

        async def start(self):
            return None

        async def shutdown(self):
            self.shutdown_called = True

    operator = ExhaustedOperator()
    stop = asyncio.Event()
    await LiveFeedSupervisor(operator=operator).run(stop)  # type: ignore[arg-type]
    assert operator.shutdown_called
    assert stop.is_set()


async def test_supervisor_shuts_down_when_poll_raises() -> None:
    class FailingPollOperator:
        def __init__(self) -> None:
            self.state = OperatorState.SYNCING
            self.session = object()
            self.active_market = object()
            self.generation = 1
            self.shutdown_called = False

        async def start(self) -> None:
            return None

        async def poll(self) -> None:
            raise RuntimeError("unexpected poll failure")

        async def shutdown(self) -> None:
            self.shutdown_called = True

    class ImmediateSupervisor(LiveFeedSupervisor):
        def _start_window_feeds(
            self, *_args: object
        ) -> tuple[asyncio.Task[None], ...]:
            return ()

        async def _wait_interval(self, _stop_event: asyncio.Event) -> None:
            return None

    operator = FailingPollOperator()
    supervisor = ImmediateSupervisor(operator=operator)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="unexpected poll failure"):
        await supervisor.run(asyncio.Event())

    assert operator.shutdown_called is True
