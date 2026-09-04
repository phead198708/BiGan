from __future__ import annotations

import json
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

import pytest


@pytest.mark.parametrize("shutdown_signal", [signal.SIGINT, signal.SIGTERM])
def test_cli_is_independent_read_only_and_exits_cleanly(tmp_path, shutdown_signal):
    config = tmp_path / "operator.toml"
    config.write_text(
        'operator_id = "dashboard-test"\nstrategy_id = "test"\npaper_account_id = "test"\n'
        f'source_commit = "test"\noutput_dir = "{tmp_path / "absent-output"}"\n'
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    process = subprocess.Popen(
        [sys.executable, "-m", "bigan.paper_trading.dashboard", "--config", str(config), "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                with urlopen(f"http://127.0.0.1:{port}/healthz", timeout=.25) as response:
                    assert json.load(response)["read_only"] is True
                break
            except URLError:
                if process.poll() is not None or time.monotonic() >= deadline:
                    pytest.fail("dashboard did not start its loopback listener")
                time.sleep(.02)
        process.send_signal(shutdown_signal)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0
        assert "Traceback" not in stdout + stderr
        assert not (tmp_path / "absent-output").exists()
        assert list(tmp_path.iterdir()) == [config]
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=10)
