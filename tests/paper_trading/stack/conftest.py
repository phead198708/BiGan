from __future__ import annotations

import json
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from bigan.paper_trading.dashboard.reader import DashboardReader
from bigan.paper_trading.operator.runtime import PaperTradingOperator
from bigan.paper_trading.stack.preflight import Preflight
from bigan.paper_trading.stack.report import SoakReport
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

COMMIT = "83efdef055ef3211d395296a3a72fde6ede05019"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def write_config(root: Path, **overrides) -> Path:
    values = {"operator_id": "stack-test", "strategy_id": "strategy-test", "paper_account_id": "account-test",
              "source_commit": COMMIT, "output_dir": str(root / "paper-output"), "mock": False, "dry_run": False}
    values.update(overrides)
    path = root / "config.toml"
    path.write_text("\n".join(f"{k} = {json.dumps(v)}" for k, v in values.items()), encoding="utf-8")
    return path


@pytest.fixture
async def observation(tmp_path):
    clock = FakeClock(10000)
    markets = [_market(i + 1, start=i * 900000) for i in range(3)]
    config = _config(tmp_path / "paper")
    operator = PaperTradingOperator(config=config, discovery=FakeDiscovery([_selection(m) for m in markets]),
                                    resolution=FakeResolution([_final(m) for m in markets]), clock_ms=clock)
    await operator.start()
    await _ready_operator(operator, clock)
    check = Preflight(config, tmp_path / "config.toml", tmp_path, "127.0.0.1", free_port(), tmp_path / "report", True)
    reader = DashboardReader(config, clock_ms=clock)
    try:
        yield SimpleNamespace(clock=clock, markets=markets, operator=operator, reader=reader,
                              check=check, report=SoakReport(check), payload=reader.read())
    finally:
        await operator.shutdown()
