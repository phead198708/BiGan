from __future__ import annotations

from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer

from bigan.paper_trading.dashboard.reader import DashboardReader
from bigan.paper_trading.dashboard.server import create_app
from bigan.paper_trading.operator.runtime import PaperTradingOperator
from tests.paper_trading.operator.test_runtime import (
    FakeClock,
    FakeDiscovery,
    FakeResolution,
    _config,
    _final,
    _market,
    _ready_operator,
    _selection,
)


@pytest.fixture
async def bundle(tmp_path):
    clock, markets = FakeClock(10_000), [_market(i + 1, start=i * 900_000) for i in range(3)]
    config = _config(tmp_path)
    operator = PaperTradingOperator(
        config=config, discovery=FakeDiscovery([_selection(m) for m in markets]),
        resolution=FakeResolution([_final(m) for m in markets]), clock_ms=clock,
    )
    await operator.start()
    await _ready_operator(operator, clock)
    try:
        yield SimpleNamespace(config=config, operator=operator, clock=clock, markets=markets,
                              reader=DashboardReader(config, clock_ms=clock))
    finally:
        await operator.shutdown()


@pytest.fixture
async def client(bundle):
    async with TestClient(TestServer(create_app(bundle.reader))) as client:
        yield client
