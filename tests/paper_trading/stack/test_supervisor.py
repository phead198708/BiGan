from __future__ import annotations

import asyncio
import json
import signal
import socket
import sys

import pytest

from bigan.paper_trading.operator.ownership import AccountProcessLock
from bigan.paper_trading.operator.runtime import PaperTradingOperator
from bigan.paper_trading.stack.preflight import preflight
from bigan.paper_trading.stack.supervisor import PaperStackSupervisor
from tests.paper_trading.stack.conftest import free_port, write_config


def supervisor(tmp_path, **kwargs):
    config = write_config(tmp_path, mock=True, status_interval_ms=100, pricing_tail_cutoff_ms=1,
                          volatility_min_samples=2, ofi_min_samples=2, volatility_return_interval_ms=1)
    check = preflight(config_path=config, mock=True, port=free_port(), report_dir=tmp_path / "report")
    return PaperStackSupervisor(check, duration=3, poll_interval=0.27, shutdown_grace=1, **kwargs)


async def wait_ready(stack):
    async with asyncio.timeout(10):
        while not stack.report.data["polls"]["successful"]:
            if stack.report.failed:
                pytest.fail(str(stack.report.data["hard_failures"]))
            await asyncio.sleep(0.02)


def assert_reaped(stack):
    assert all(p.returncode is not None for p in stack.children.values())
    assert all(t.done() for t in stack._drainers)
    assert stack.observer.http is None or stack.observer.http.closed


@pytest.mark.parametrize("hold", [False, True])
async def test_real_children_e2e_rollover_and_stop(tmp_path, hold):
    logs = []
    stack = supervisor(tmp_path, log=logs.append)
    commands = stack.commands()
    if hold:
        commands["operator"].append("--hold-quotes")
    stack.commands = lambda: commands
    assert await stack.run() == 0
    assert_reaped(stack)
    assert logs.index("[soak] dashboard healthy") < logs.index("[operator] started")
    assert "[soak] operator ready" in logs
    report = json.loads((tmp_path / "report" / "soak_report.json").read_text())
    # A concurrent SQLite rollback journal can legitimately make one bounded
    # read unavailable. Never hide that warning just to manufacture a PASS.
    assert report["result"] in {"PASS", "WARN"}, report["warnings"]
    assert all(w["severity"] == "info" or w["code"] in {
        "DASHBOARD_SECTION_WARNING", "ACCOUNT_UNAVAILABLE",
    } for w in report["warnings"])
    assert report["rollovers"] == 1
    assert report["final_state"] == "STOPPED"
    assert report["activity"]["decisions"] > 0
    assert report["activity"]["settlements"] == 1
    assert (report["activity"]["fills"] == 0) is hold
    status = json.loads((tmp_path / "paper-output" / "stack-test" / "operator_status.json").read_text())
    assert status["state"] == "STOPPED"
    lock = AccountProcessLock(output_dir=stack.check.config.output_dir, operator_id="stack-test", account_id="account-test")
    lock.acquire()
    lock.release()
    assert not any(str(tmp_path) in line for line in logs)


@pytest.mark.parametrize("which", ["operator", "dashboard"])
async def test_unexpected_child_exit_no_restart(tmp_path, which):
    stack = supervisor(tmp_path)
    task = asyncio.create_task(stack.run())
    await wait_ready(stack)
    process = stack.children[which]
    process.kill()
    assert await task == 1
    assert stack.children[which] is process
    assert len(stack.children) == 2
    assert_reaped(stack)
    assert any(x["code"] == which.upper() + "_UNEXPECTED_EXIT" for x in stack.report.data["hard_failures"])


async def test_shutdown_escalation_is_failure(tmp_path):
    stack = supervisor(tmp_path)
    stack.duration, stack.shutdown_grace = 0.3, 0.1
    task = asyncio.create_task(stack.run())
    await wait_ready(stack)
    stack.children["operator"].send_signal(signal.SIGSTOP)
    assert await task == 1
    assert_reaped(stack)
    assert any(x["code"] == "OPERATOR_FORCED_TERMINATION" for x in stack.report.data["hard_failures"])


async def test_startup_timeout_and_sanitized_child_logs(tmp_path):
    logs = []
    stack = supervisor(tmp_path, startup_timeout=0.25, log=logs.append)
    commands = stack.commands()
    commands["dashboard"] = [sys.executable, "-u", "-c", "import time; print('Traceback /private/key SECRET'); time.sleep(20)"]
    stack.commands = lambda: commands
    assert await stack.run() == 1
    assert_reaped(stack)
    assert "operator" not in stack.children
    assert any(x["code"] == "STARTUP_TIMEOUT" for x in stack.report.data["hard_failures"])
    assert any(line.startswith("[dashboard] output") for line in logs)
    assert all("SECRET" not in line and "/private" not in line and "Traceback" not in line for line in logs)


async def test_writer_lock_owned_by_another_process_is_not_bypassed(tmp_path):
    stack = supervisor(tmp_path)
    lock = AccountProcessLock(output_dir=stack.check.config.output_dir, operator_id="stack-test", account_id="account-test")
    lock.acquire()
    try:
        assert await stack.run() == 1
        assert lock.held
        assert not list(stack.check.config.output_dir.glob("paper-*"))
    finally:
        lock.release()
    assert_reaped(stack)


async def test_port_race_after_preflight_stops_before_writer(tmp_path):
    stack = supervisor(tmp_path)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", stack.check.port))
        sock.listen()
        assert await stack.run() == 1
    assert "operator" not in stack.children
    assert_reaped(stack)


async def test_config_change_between_preflight_and_children_rejected(tmp_path):
    stack = supervisor(tmp_path)
    path = stack.check.config_path
    path.write_text(path.read_text().replace('strategy-test', 'changed-strategy'))
    assert await stack.run() == 1
    assert not stack.check.config.output_dir.exists()
    assert_reaped(stack)


async def test_parent_never_constructs_writer_or_takes_account_lock(tmp_path, monkeypatch):
    stack = supervisor(tmp_path)
    stack.duration = 0.2

    def forbidden(*args, **kwargs):
        pytest.fail("supervisor/observer attempted writer ownership")

    monkeypatch.setattr(AccountProcessLock, "acquire", forbidden)
    monkeypatch.setattr(PaperTradingOperator, "__init__", forbidden)
    assert await stack.run() == 0  # Children import their own unpatched modules.
    assert_reaped(stack)
