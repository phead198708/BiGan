"""Smoke-check an isolated wheel installation; not collected as a pytest test."""

from __future__ import annotations

import asyncio
import sys
import tempfile
from importlib.metadata import distributions
from importlib.resources import files
from pathlib import Path


def main() -> None:
    target = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(target))
    from aiohttp.test_utils import TestClient, TestServer

    import bigan.paper_trading.dashboard as dashboard
    from bigan.paper_trading.dashboard.reader import DashboardReader
    from bigan.paper_trading.dashboard.server import create_app
    from bigan.paper_trading.operator.config import OperatorConfig

    assert Path(dashboard.__file__).is_relative_to(target)
    assert any(
        point.name == "bigan-paper-dashboard" and point.value == "bigan.paper_trading.dashboard.__main__:main"
        for distribution in distributions(path=[str(target)]) for point in distribution.entry_points
    )
    for name in ("index.html", "app.js", "styles.css"):
        assert files(dashboard).joinpath("static", name).read_bytes()

    async def check() -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = OperatorConfig(
                operator_id="wheel-check", strategy_id="test", paper_account_id="test",
                source_commit="test", output_dir=Path(temporary) / "never-created",
            )
            async with TestClient(TestServer(create_app(DashboardReader(config)))) as client:
                for path in ("/", "/static/app.js", "/static/styles.css", "/healthz"):
                    assert (await client.get(path)).status == 200
                assert (await client.get("/readyz")).status == 503
            assert not await asyncio.to_thread(Path(config.output_dir).exists)
    asyncio.run(check())
    print("Installed wheel serves all packaged assets independently of source cwd; no paper output created.")


if __name__ == "__main__":
    main()
