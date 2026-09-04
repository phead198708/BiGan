"""Own child processes only. Operator retains exclusive paper writer ownership."""

from __future__ import annotations

import asyncio
import signal
import sys
import time
import uuid
from collections.abc import Callable
from contextlib import suppress
from typing import Any

import aiohttp

from .observer import (
    HTTPUnavailable,
    ObservationError,
    ObservationPolicy,
    PaperSoakObserver,
    live_inputs_ready,
)
from .preflight import Preflight
from .report import ReportWriter, SoakReport, now_ms


class StackFailure(RuntimeError):
    pass


class StackStopped(Exception):
    pass


class PaperStackSupervisor:
    def __init__(self, check: Preflight, *, duration: float | None = None,
                 startup_timeout: float = 60, poll_interval: float = 2,
                 shutdown_grace: float = 15, policy: ObservationPolicy | None = None,
                 log: Callable[[str], Any] = print) -> None:
        for value in (startup_timeout, poll_interval, shutdown_grace, 1 if duration is None else duration):
            if not 0 < value <= 604800:
                raise ValueError("invalid supervisor deadline")
        self.check, self.duration = check, duration
        self.startup_timeout, self.poll_interval, self.shutdown_grace = startup_timeout, poll_interval, shutdown_grace
        self.log = log
        self.stop = asyncio.Event()
        self.children: dict[str, asyncio.subprocess.Process] = {}
        self._drainers: list[asyncio.Task[None]] = []
        self.instance_id = uuid.uuid4().hex
        self.report = SoakReport(check)
        self.report.data["requested_duration_ms"] = None if duration is None else round(duration * 1000)
        self.observer = PaperSoakObserver(report=self.report, policy=policy, instance_id=self.instance_id,
                                          history_limit=min(check.config.recent_query_max, 50))
        self.operator_started_ms: int | None = None

    def commands(self) -> dict[str, list[str]]:
        shared = ["--config", str(self.check.config_path), "--expected-config-sha256", self.check.config.config_sha256]
        if not self.check.mock:
            shared += ["--expected-source-commit", self.check.config.source_commit]
        operator_module = "bigan.paper_trading.stack.mock_operator" if self.check.mock else "bigan.paper_trading.operator"
        return {
            "dashboard": [sys.executable, "-m", "bigan.paper_trading.dashboard", *shared,
                          "--host", self.check.host, "--port", str(self.check.port), "--instance-id", self.instance_id],
            "operator": [sys.executable, "-m", operator_module, *shared],
        }

    async def run(self) -> int:
        writer: ReportWriter | None = None
        started: float | None = None
        loop = asyncio.get_running_loop()
        installed: list[signal.Signals] = []
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(signum, self.stop.set)
                installed.append(signum)
            except NotImplementedError:
                pass
        try:
            if self.check.report_dir is not None:
                writer = ReportWriter(self.check.report_dir, output=self.check.cwd / self.check.config.output_dir)
            async with self.observer:
                try:
                    commands = self.commands()
                    startup_deadline = time.monotonic() + self.startup_timeout
                    await self._spawn("dashboard", commands["dashboard"])
                    await self._await_ready(operator=False, deadline=startup_deadline)
                    self.operator_started_ms = now_ms()
                    await self._spawn("operator", commands["operator"])
                    await self._await_ready(operator=True, deadline=startup_deadline)
                    self.log("[soak] ready " + self.check.url + " — PAPER / SIMULATED — NO REAL FUNDS")
                    started = time.monotonic()
                    while not self.stop.is_set():
                        self._check_children()
                        if self.duration is not None and time.monotonic() - started >= self.duration:
                            break
                        await self._poll_or_stop()
                        if self.report.failed:
                            break
                        remaining = self.poll_interval
                        if self.duration is not None:
                            remaining = min(remaining, max(0, self.duration - (time.monotonic() - started)))
                        await self._pause(remaining)
                except StackStopped:
                    pass
                except (StackFailure, ObservationError) as exc:
                    self.observer.freeze_readiness_failure()
                    self.report.issue(str(exc), hard=True)
                except asyncio.CancelledError:
                    self.report.issue("SUPERVISOR_CANCELLED", hard=True)
                except Exception:
                    self.report.issue("STACK_STARTUP_OR_RUNTIME_FAILURE", hard=True)
                finally:
                    if started is not None:
                        self.report.data["measurement_duration_ms"] = int(max(0, time.monotonic() - started) * 1000)
                    # No new polls. Keep the reader alive for final STOPPED observation.
                    if "operator" in self.children:
                        await self._terminate("operator")
                        dashboard = self.children.get("dashboard")
                        if dashboard is not None and dashboard.returncode is None:
                            final_read = await self.observer.poll(final=True)
                            if final_read and not self.report.failed:
                                # One dashboard refresh interval to expose STOPPED to
                                # an already-open browser before disconnecting it.
                                try:
                                    await asyncio.wait_for(dashboard.wait(), 2.5)
                                except TimeoutError:
                                    pass
                                else:
                                    self.report.issue("DASHBOARD_UNEXPECTED_EXIT", hard=True)
                    await self._terminate("dashboard")
        except Exception:
            self.report.issue("STACK_SETUP_OR_CLEANUP_FAILURE", hard=True)
        finally:
            # Also protects partial-spawn/cancellation failures outside the reader context.
            for name in ("operator", "dashboard"):
                await self._terminate(name)
            for task in self._drainers:
                task.cancel()
            await asyncio.gather(*self._drainers, return_exceptions=True)
            for signum in installed:
                loop.remove_signal_handler(signum)
            self.report.finish()
            if writer is not None:
                try:
                    writer.write(self.report)
                except Exception:
                    self.report.issue("REPORT_WRITE_FAILED", hard=True)
                finally:
                    try:
                        writer.close()
                    except OSError:
                        # write() has already decided/published the outcome.
                        # Closing a read-only directory FD cannot revoke that
                        # decision or introduce a different exit/disk result.
                        self.log("[soak] REPORT_DESCRIPTOR_CLOSE_FAILED (result unchanged)")
        self.log("[soak] " + self.report.data["result"])
        return 1 if self.report.failed else 0

    async def _spawn(self, name: str, command: list[str]) -> None:
        self.children[name] = await asyncio.create_subprocess_exec(
            *command, cwd=self.check.cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        self.log(f"[{name}] started")
        self._drainers.append(asyncio.create_task(self._drain(name, self.children[name])))

    async def _drain(self, name: str, child: asyncio.subprocess.Process) -> None:
        assert child.stdout is not None
        announced = False
        while await child.stdout.read(16384):
            # Child tracebacks/config dumps are untrusted. Do not attempt fragile
            # secret regex redaction; consume bounded chunks and expose no raw text.
            if not announced:
                self.log(f"[{name}] output received (details suppressed for privacy)")
                announced = True

    def _check_children(self) -> None:
        for name, child in self.children.items():
            if child.returncode is not None:
                raise StackFailure(name.upper() + "_UNEXPECTED_EXIT")

    async def _wait_ready(self, *, operator: bool) -> None:
        deadline = time.monotonic() + self.startup_timeout
        waiting_announced = False
        while time.monotonic() < deadline:
            self._check_children()
            if self.stop.is_set():
                raise StackStopped()
            try:
                self.observer.validate_health(await self.observer.get("/healthz"))
                if operator:
                    ready = await self.observer.get("/readyz")
                    if ready.get("ready") is not True:
                        raise ObservationError("INVALID_READY_SCHEMA")
                    payload = await self.observer.get("/api/v1/dashboard")
                    if payload.get("operator_identity") != self.check.identity:
                        raise ObservationError("CONFIG_IDENTITY_CHANGED")
                    status = payload.get("status", {})
                    # A readable status from a previous process is not this child's readiness.
                    if status.get("process_started_at_ms", 0) < (self.operator_started_ms or 0):
                        raise HTTPUnavailable("OLD_OPERATOR_STATUS")
                    if status.get("state") == "FAILED":
                        raise StackFailure("OPERATOR_FAILED")
                    if not self.check.mock:
                        self.observer.record_readiness(payload, phase="startup")
                    if not self.check.mock and not live_inputs_ready(payload):
                        if not waiting_announced:
                            self.log("[soak] waiting for live RUNNING state and fresh trading inputs")
                            waiting_announced = True
                        raise HTTPUnavailable("LIVE_INPUTS_NOT_READY")
                self._check_children()
                self.log("[soak] " + ("operator ready" if operator else "dashboard healthy"))
                return
            except (aiohttp.ClientError, TimeoutError, HTTPUnavailable):
                await self._pause(min(0.1, max(0, deadline - time.monotonic())))
        raise StackFailure("STARTUP_TIMEOUT")

    async def _await_ready(self, *, operator: bool, deadline: float) -> None:
        try:
            await asyncio.wait_for(self._wait_ready(operator=operator), max(0, deadline - time.monotonic()))
        except TimeoutError:
            raise StackFailure("STARTUP_TIMEOUT") from None

    async def _poll_or_stop(self) -> None:
        poll = asyncio.create_task(self.observer.poll())
        stop = asyncio.create_task(self.stop.wait())
        exits = [asyncio.create_task(child.wait()) for child in self.children.values()]
        try:
            await asyncio.wait([poll, stop, *exits], return_when=asyncio.FIRST_COMPLETED)
            self._check_children()
            if poll.done():
                await poll
        finally:
            for task in (poll, stop, *exits):
                task.cancel()
            await asyncio.gather(poll, stop, *exits, return_exceptions=True)

    async def _pause(self, seconds: float) -> None:
        # Check owned processes during waits without queuing polls.
        deadline = time.monotonic() + seconds
        while not self.stop.is_set() and time.monotonic() < deadline:
            self._check_children()
            with suppress(TimeoutError):
                await asyncio.wait_for(self.stop.wait(), timeout=min(0.1, max(0, deadline - time.monotonic())))

    async def _terminate(self, name: str) -> None:
        child = self.children.get(name)
        if child is None or child.returncode is not None:
            return
        with suppress(ProcessLookupError):
            child.terminate()
        try:
            await asyncio.wait_for(child.wait(), self.shutdown_grace)
        except TimeoutError:
            self.report.issue(name.upper() + "_FORCED_TERMINATION", hard=True)
            with suppress(ProcessLookupError):
                child.kill()
            await child.wait()
        if child.returncode != 0:
            self.report.issue(name.upper() + "_UNCLEAN_SHUTDOWN", hard=True)
        self.log(f"[{name}] stopped")
