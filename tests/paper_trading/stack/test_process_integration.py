from __future__ import annotations

import asyncio
import json
import os
import signal
import sys

import aiohttp
import pytest

from tests.paper_trading.stack.conftest import free_port, write_config


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
async def test_os_signal_stops_stack_children_and_releases_port(tmp_path, signum):
    config = write_config(tmp_path, mock=True, pricing_tail_cutoff_ms=1, status_interval_ms=100,
                          volatility_min_samples=2, ofi_min_samples=2, volatility_return_interval_ms=1)
    port = free_port()
    child = await asyncio.create_subprocess_exec(
        sys.executable, "-u", "-m", "bigan.paper_trading.stack", "--config", str(config), "--mock-demo",
        "--dashboard-port", str(port), "--report-dir", str(tmp_path / "report"), "--poll-interval", "1s",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    try:
        async with asyncio.timeout(15):
            while True:
                line = await child.stdout.readline()
                assert line, "stack exited before readiness"
                if b"[soak] ready http" in line:
                    break
        await asyncio.sleep(0.3)
        os.kill(child.pid, signum)
        stdout, _ = await asyncio.wait_for(child.communicate(), 10)
        assert child.returncode == 0, stdout
        report = json.loads((tmp_path / "report" / "soak_report.json").read_text())
        assert report["final_state"] == "STOPPED"
        async with aiohttp.ClientSession() as http:
            with pytest.raises(aiohttp.ClientConnectionError):
                await http.get(f"http://127.0.0.1:{port}/healthz")
        assert b"[operator] stopped" in stdout and b"[dashboard] stopped" in stdout
    finally:
        if child.returncode is None:
            child.terminate()
            await asyncio.wait_for(child.wait(), 20)
