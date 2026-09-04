"""Smoke-check an isolated wheel installation; not collected as a pytest test."""

from __future__ import annotations

import asyncio
import json
import socket
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
    import bigan.paper_trading.stack as stack
    from bigan.build_provenance import (
        BuildProvenanceError,
        require_source_commit,
        runtime_provenance,
    )
    from bigan.paper_trading.dashboard.reader import DashboardReader
    from bigan.paper_trading.dashboard.server import create_app
    from bigan.paper_trading.operator.config import OperatorConfig
    from bigan.paper_trading.stack.preflight import PreflightError, preflight

    assert Path(dashboard.__file__).is_relative_to(target)
    assert Path(stack.__file__).is_relative_to(target)
    assert any(
        point.name == "bigan-paper-stack" and point.value == "bigan.paper_trading.stack.__main__:main"
        for distribution in distributions(path=[str(target)]) for point in distribution.entry_points
    )
    assert any(
        point.name == "bigan-paper-dashboard" and point.value == "bigan.paper_trading.dashboard.__main__:main"
        for distribution in distributions(path=[str(target)]) for point in distribution.entry_points
    )
    for name in ("index.html", "app.js", "styles.css"):
        assert files(dashboard).joinpath("static", name).read_bytes()
    provenance = runtime_provenance()
    assert provenance["source_commit"] == sys.argv[2], "wheel must seal the Git revision supplied by the build job"
    try:
        require_source_commit("0" * 40)
    except BuildProvenanceError:
        pass
    else:
        raise AssertionError("mismatched source was accepted")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        config_path = root / "live.toml"
        values = {"operator_id": "wheel-test", "strategy_id": "test", "paper_account_id": "test",
                  "source_commit": provenance["source_commit"], "output_dir": str(root / "never-created"),
                  "mock": False, "dry_run": False}
        config_path.write_text("\n".join(f"{key} = {json.dumps(value)}" for key, value in values.items()))
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        assert preflight(config_path=config_path, port=port).build_provenance == provenance
        values["source_commit"] = "0" * 40
        config_path.write_text("\n".join(f"{key} = {json.dumps(value)}" for key, value in values.items()))
        try:
            preflight(config_path=config_path, port=port)
        except PreflightError:
            pass
        else:
            raise AssertionError("incorrect declared wheel source was accepted")
        assert not (root / "never-created").exists()

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
    print("Installed wheel verifies its sealed source, rejects mismatches, and serves assets; no paper output created.")


if __name__ == "__main__":
    main()
