"""HTTP-only observer. No paper file handles, sessions, writers or recovery."""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import aiohttp

from bigan.paper_trading.contracts import PaperDecisionEvent, PaperSettlementEvent
from bigan.paper_trading.operator.read_model import OperatorStatus

from .diagnostics import TRACE_LIMIT, readiness_snapshot
from .preflight import SAFETY
from .report import SoakReport, require_finite


class ObservationError(ValueError):
    pass


class HTTPUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class ObservationPolicy:
    request_timeout: float = 3.0
    unreadable_seconds: float = 30.0
    stale_seconds: float = 30.0
    rollover_seconds: float = 900.0
    max_response_bytes: int = 2_000_000
    max_runs: int = 1024

    def __post_init__(self) -> None:
        for value in (self.request_timeout, self.unreadable_seconds, self.stale_seconds, self.rollover_seconds):
            if not math.isfinite(value) or not 0 < value <= 604800:
                raise ValueError("invalid observation deadline")
        if not 1024 <= self.max_response_bytes <= 8_000_000 or not 1 <= self.max_runs <= 4096:
            raise ValueError("invalid observation memory bounds")


def _object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ObservationError("INVALID_API_SCHEMA")
    return value


def _integer(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ObservationError("INVALID_API_SCHEMA")
    return value


def _number(value: Any, *, nonnegative: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ObservationError("INVALID_API_SCHEMA")
    if nonnegative and value < 0:
        raise ObservationError("NEGATIVE_ACCOUNT_VALUE")
    return float(value)


def _run_id(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"paper-[0-9a-f]{24}", value):
        raise ObservationError("INVALID_API_SCHEMA")
    return value


def _text(value: Any) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 256:
        raise ObservationError("INVALID_API_SCHEMA")
    return value


def live_inputs_ready(payload: dict[str, Any]) -> bool:
    """Trading readiness is stricter than the dashboard's read-model /readyz."""
    status = _object(payload.get("status"))
    market = status.get("active_market")
    if (payload.get("stale") is not False or status.get("state") != "RUNNING"
            or not status.get("run_id") or not isinstance(market, dict)):
        return False
    timestamp = _integer(payload.get("generated_at_ms"))
    if not (_integer(market.get("window_start_ts_ms")) <= timestamp
            < _integer(market.get("window_end_ts_ms"))):
        return False
    feeds = _object(status.get("feeds"))
    for name in ("binance", "polymarket", "chainlink"):
        health = _object(feeds.get(name))
        if any(health.get(key) is not True for key in ("connected", "synchronized", "fresh")):
            return False
    pricing = _object(status.get("pricing_inputs"))
    session = _object(status.get("session"))
    return bool(
        pricing.get("ready") is True and pricing.get("fresh") is True
        and _object(status.get("alpha")).get("fresh") is True
        and session.get("healthy") is True and session.get("failure_reason") is None
    )


class PaperSoakObserver:
    def __init__(self, *, report: SoakReport, policy: ObservationPolicy | None = None,
                 instance_id: str | None = None, history_limit: int = 50) -> None:
        self.report = report
        self.policy = policy or ObservationPolicy()
        self.url = report.data["dashboard_url"]
        self.identity = report.data["operator_identity"]
        self.instance_id = instance_id
        if not 1 <= history_limit <= 500:
            raise ValueError("invalid history bound")
        self.history_limit = history_limit
        self.http: aiohttp.ClientSession | None = None
        self._poll_lock = asyncio.Lock()
        self._failure_since: float | None = None
        self._stale_since: float | None = None
        self._live_unready_since: float | None = None
        self._rollover_since: float | None = None
        self._last_state: str | None = None
        self._last_sample_time: float | None = None
        self._runs: dict[str, dict[str, Any]] = {}
        self._current_run: str | None = None
        self._current_index = -1
        self._last_counters: dict[str, int] = {}
        self._counter_run: str | None = None
        self.last_process_started_ms: int | None = None
        self.last_updated_ms: int | None = None
        self.last_state: str | None = None

    def record_readiness(self, payload: dict[str, Any], *, phase: str) -> None:
        trace = self.report.data["readiness_diagnostics"]
        sample = readiness_snapshot(payload, phase=phase)
        samples = trace["samples"]
        # Startup probes run at 10 Hz; retain at most one per second. Runtime
        # probes retain every observation, including the one that trips a gate.
        if phase == "startup" and samples:
            previous = samples[-1]
            if (previous["phase"] == phase and sample["reasons"] == previous["reasons"]
                    and sample["observed_at_ms"] is not None and previous["observed_at_ms"] is not None
                    and 0 <= sample["observed_at_ms"] - previous["observed_at_ms"] < 1000):
                return
        samples.append(sample)
        if len(samples) > TRACE_LIMIT:
            del samples[0]
            trace["evicted_samples"] += 1
        for code in sample["reasons"]:
            counts = trace["reason_counts"]
            counts[code] = counts.get(code, 0) + 1

    def freeze_readiness_failure(self) -> None:
        trace = self.report.data["readiness_diagnostics"]
        if trace["failure_snapshot"] is None and trace["samples"]:
            trace["failure_snapshot"] = deepcopy(trace["samples"][-1])

    async def __aenter__(self) -> PaperSoakObserver:
        self.http = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.policy.request_timeout),
            trust_env=False, cookie_jar=aiohttp.DummyCookieJar(),
        )
        return self

    async def __aexit__(self, *_args: Any) -> None:
        if self.http is not None:
            await self.http.close()

    async def get(self, path: str) -> dict[str, Any]:
        if path not in ("/healthz", "/readyz", "/api/v1/dashboard"):
            raise ValueError("observer endpoint not allowed")
        assert self.http is not None
        params = {"limit": str(self.history_limit)} if path == "/api/v1/dashboard" else None
        async with self.http.get(self.url + path, params=params, allow_redirects=False) as response:
            if response.status != 200:
                raise HTTPUnavailable("DASHBOARD_HTTP_UNAVAILABLE")
            body = bytearray()
            async for chunk in response.content.iter_chunked(16384):
                body.extend(chunk)
                if len(body) > self.policy.max_response_bytes:
                    raise ObservationError("API_RESPONSE_TOO_LARGE")
            try:
                payload = json.loads(body)
                require_finite(payload)
            except (ValueError, RecursionError, OverflowError):
                raise ObservationError("INVALID_OR_NONFINITE_API_JSON") from None
            return _object(payload)

    def validate_health(self, health: dict[str, Any]) -> None:
        if any(health.get(key) is not True for key in ("alive", "paper_only", "read_only")):
            raise ObservationError("INVALID_HEALTH_OR_SAFETY")
        if health.get("operator_identity") != self.identity:
            raise ObservationError("CONFIG_IDENTITY_CHANGED")
        if self.instance_id is not None and health.get("instance_id") != self.instance_id:
            raise ObservationError("DASHBOARD_INSTANCE_CHANGED")

    async def poll(self, *, final: bool = False) -> bool:
        async with self._poll_lock:
            polls = self.report.data["polls"]
            polls["attempted"] += 1
            timestamp = time.monotonic()
            try:
                self.validate_health(await self.get("/healthz"))
                ready = await self.get("/readyz")
                if ready.get("ready") is not True or type(ready.get("stale")) is not bool:
                    raise ObservationError("INVALID_READY_SCHEMA")
                payload = await self.get("/api/v1/dashboard")
                self.observe(payload, at=timestamp, final=final)
                if final and self.last_state != "STOPPED":
                    raise ObservationError("FINAL_STATUS_NOT_STOPPED")
            except (aiohttp.ClientError, TimeoutError, HTTPUnavailable):
                self._record_failure(timestamp)
                if final:
                    self.report.issue("FINAL_STATUS_UNREADABLE", hard=True)
                return False
            except ObservationError as exc:
                polls["failed"] += 1
                self.report.issue(str(exc), hard=True)
                return False
            except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
                polls["failed"] += 1
                self.report.issue("INVALID_API_SCHEMA", hard=True)
                return False
            polls["successful"] += 1
            self._end_failure(timestamp)
            return True

    def _end_failure(self, at: float) -> None:
        if self._failure_since is not None:
            self._failure_length(at)
        self._failure_since = None

    def _failure_length(self, at: float) -> float:
        length = 0.0 if self._failure_since is None else max(0.0, at - self._failure_since)
        polls = self.report.data["polls"]
        polls["longest_failure_streak_ms"] = max(polls["longest_failure_streak_ms"], round(length * 1000))
        return length

    def _record_failure(self, at: float) -> None:
        self.report.data["polls"]["failed"] += 1
        if self._failure_since is None:
            self._failure_since = at
        self.report.issue("POLL_UNAVAILABLE")
        if self._failure_length(at) >= self.policy.unreadable_seconds:
            self.report.issue("STATUS_UNREADABLE_DEADLINE", hard=True)

    def observe(self, payload: dict[str, Any], *, at: float, final: bool = False) -> None:
        """Validate a bounded HTTP view, then aggregate safe numeric observations."""
        require_finite(payload)
        if payload.get("schema_version") != 1 or type(payload.get("schema_version")) is not int:
            raise ObservationError("INVALID_API_SCHEMA")
        if payload.get("operator_identity") != self.identity:
            raise ObservationError("CONFIG_IDENTITY_CHANGED")
        status = _object(payload["status"])
        safety = _object(status["safety"])
        if (status.get("paper_only") is not True or set(safety) != set(SAFETY) - {"paper_only"}
                or any(value is not False for value in safety.values())):
            raise ObservationError("PAPER_SAFETY_VIOLATION")
        # Existing pure schema contract; never instantiate an operator/session.
        OperatorStatus.from_dict(status)
        for key in ("cash", "equity"):
            _number(status["account"][key], nonnegative=True)
        for key in ("operator_id", "strategy_id", "source_commit"):
            if status[key] != self.identity[key]:
                raise ObservationError("CONFIG_IDENTITY_CHANGED")
        process_started = _integer(status["process_started_at_ms"])
        updated = _integer(status["updated_at_ms"])
        if self.last_process_started_ms is not None and process_started != self.last_process_started_ms:
            raise ObservationError("OPERATOR_PROCESS_IDENTITY_CHANGED")
        if self.last_updated_ms is not None and updated < self.last_updated_ms:
            raise ObservationError("STATUS_TIMESTAMP_REGRESSED")
        self.last_process_started_ms, self.last_updated_ms = process_started, updated
        self.last_state = state = status["state"]
        self.report.data["final_state"] = state
        if state == "FAILED":
            raise ObservationError("OPERATOR_FAILED")
        if state == "STOPPED" and not final:
            raise ObservationError("OPERATOR_STOPPED_UNEXPECTEDLY")
        generated = _integer(payload["generated_at_ms"])
        age = payload["status_age_ms"]
        stale_after = _integer(payload["stale_after_ms"])
        if type(age) is not int or age != generated - updated or type(payload["stale"]) is not bool:
            raise ObservationError("INVALID_API_SCHEMA")
        if payload["stale"] != (age < 0 or age > stale_after):
            raise ObservationError("INVALID_API_SCHEMA")
        if payload["stale"] and not final:
            self._stale_since = at if self._stale_since is None else self._stale_since
            self.report.issue("STATUS_STALE")
            if at - self._stale_since >= self.policy.stale_seconds:
                raise ObservationError("STATUS_STALE_DEADLINE")
        else:
            self._stale_since = None
        market = status["active_market"]
        if market is not None:
            market = _object(market)
            if any(_object(payload["active_market"]).get(key) != market.get(key) for key in (
                "market_id", "window_id", "window_start_ts_ms", "window_end_ts_ms",
            )):
                raise ObservationError("MIXED_MARKET_IDENTITY")
            end = _integer(market["window_end_ts_ms"])
            start = _integer(market["window_start_ts_ms"])
            if end <= start:
                raise ObservationError("INVALID_API_SCHEMA")
        elif payload["active_market"] is not None:
            raise ObservationError("MIXED_MARKET_IDENTITY")
        pending = state in {"ROLLING_OVER", "SETTLEMENT_PENDING"} or (
            market is not None and updated > market["window_end_ts_ms"] and state not in {"STOPPED", "EXHAUSTED"}
        )
        if pending:
            self._rollover_since = at if self._rollover_since is None else self._rollover_since
            if at - self._rollover_since >= self.policy.rollover_seconds:
                raise ObservationError("ROLLOVER_DEADLINE")
        else:
            self._rollover_since = None
        self._history(payload, status, market)
        account = payload["account"]
        if account is not None:
            account = _object(account)
            if account.get("run_id") != status["run_id"] or status["run_id"] is None:
                raise ObservationError("MIXED_ACCOUNT_RUN")
            self._account(account)
        else:
            self.report.issue("ACCOUNT_UNAVAILABLE", informational=status["run_id"] is None or state == "ROLLING_OVER")
        if payload["positions"] is not None and not isinstance(payload["positions"], list):
            raise ObservationError("INVALID_API_SCHEMA")
        for position in payload["positions"] or []:
            if market is None or _object(position).get("window_id") != market["window_id"]:
                raise ObservationError("MIXED_POSITION_WINDOW")
        if status["last_decision"] is not None and (
            market is None or status["last_decision"].get("window_id") != market["window_id"]
        ):
            raise ObservationError("MIXED_DECISION_WINDOW")
        if status["last_fill"] is not None and status["last_fill"].get("run_id") != status["run_id"]:
            raise ObservationError("MIXED_FILL_RUN")
        self._aggregate(status, at, final=final)
        if self.report.data["mode"] == "live_public_feeds_paper_execution" and not final:
            self.record_readiness(payload, phase="runtime")
            coverage = self.report.data["live_readiness"]
            if live_inputs_ready(payload):
                coverage["ready_samples"] += 1
                self._live_unready_since = None
            else:
                coverage["unready_samples"] += 1
                self._live_unready_since = at if self._live_unready_since is None else self._live_unready_since
                self.report.issue("LIVE_INPUTS_NOT_READY")
                # Settlement has its own, longer bounded handoff deadline.
                deadline = self.policy.rollover_seconds if pending else self.policy.stale_seconds
                if at - self._live_unready_since >= deadline:
                    self.freeze_readiness_failure()
                    failure = self.report.data["readiness_diagnostics"]["failure_snapshot"]
                    if failure is not None:
                        failure["unready_duration_ms"] = int((at - self._live_unready_since) * 1000)
                        failure["deadline_ms"] = int(deadline * 1000)
                    raise ObservationError("LIVE_INPUTS_UNAVAILABLE_DEADLINE")
        if not isinstance(payload["warnings"], list) or len(payload["warnings"]) > 32:
            raise ObservationError("INVALID_API_SCHEMA")
        for warning in payload["warnings"]:
            code = _object(warning).get("code")
            if code not in {"NO_ACTIVE_RUN", "FRONTIER_UNAVAILABLE", "ACCOUNT_UNAVAILABLE", "HISTORY_UNAVAILABLE", "STATUS_STALE"}:
                raise ObservationError("INVALID_API_SCHEMA")
            self.report.issue("DASHBOARD_SECTION_WARNING", informational=(
                code == "NO_ACTIVE_RUN" or code in {"FRONTIER_UNAVAILABLE", "HISTORY_UNAVAILABLE"} and state == "ROLLING_OVER"
            ))

    def _history(self, payload: dict[str, Any], status: dict[str, Any], market: dict[str, Any] | None) -> None:
        recent = _object(payload["recent"])
        for name in ("runs", "decisions", "fills", "settlements"):
            value = recent[name]
            if value is not None and (not isinstance(value, list) or len(value) > self.history_limit):
                raise ObservationError("INVALID_API_SCHEMA")
        rows = recent["runs"]
        for newer, older in zip(rows or [], (rows or [])[1:], strict=False):
            if (_object(newer).get("predecessor_run_id") != _object(older).get("run_id")
                    or _integer(newer["run_index"]) != _integer(older["run_index"]) + 1):
                raise ObservationError("MIXED_RUN_CHAIN")
        for row in rows or []:
            row = _object(row)
            rid = _run_id(row["run_id"])
            index = _integer(row["run_index"])
            if not isinstance(row["window_ids"], list) or len(row["window_ids"]) != 1:
                raise ObservationError("INVALID_API_SCHEMA")
            window = _text(row["window_ids"][0])
            market_id = _text(row["market_id"])
            predecessor = row["predecessor_run_id"]
            if predecessor is not None:
                _run_id(predecessor)
            opening = _number(row["opening_cash"], nonnegative=True)
            cash = _number(row["cash"], nonnegative=True)
            if type(row["settled"]) is not bool:
                raise ObservationError("INVALID_API_SCHEMA")
            identity = (index, window, market_id, predecessor, opening)
            known = self._runs.get(rid)
            if known is None:
                if len(self._runs) >= self.policy.max_runs:
                    raise ObservationError("OBSERVATION_RUN_CAPACITY_EXCEEDED")
                known = {"identity": identity, "settlement_cash": None}
                self._runs[rid] = known
            elif known["identity"] != identity:
                raise ObservationError("RUN_IDENTITY_CHANGED")
            if row["settled"]:
                if known["settlement_cash"] is not None and not math.isclose(known["settlement_cash"], cash, abs_tol=1e-8):
                    raise ObservationError("SETTLED_CASH_CHANGED")
                known["settlement_cash"] = cash
        current = status["run_id"]
        if current is None:
            if market is not None or self._current_index >= 0:
                raise ObservationError("MIXED_ACTIVE_RUN")
        else:
            _run_id(current)
            if market is None:
                raise ObservationError("MIXED_ACTIVE_RUN")
            if rows is not None and (not rows or rows[0]["run_id"] != current):
                raise ObservationError("MIXED_RUN_HISTORY")
            known = self._runs.get(current)
            if known is not None:
                index, window, market_id, predecessor, opening = known["identity"]
                if window != market["window_id"] or market_id != market["market_id"]:
                    raise ObservationError("MIXED_RUN_MARKET")
                if index < self._current_index or (index == self._current_index and current != self._current_run):
                    raise ObservationError("RUN_INDEX_REGRESSED")
                if current != self._current_run:
                    if self._current_index >= 0:
                        self.report.data["rollovers"] += index - self._current_index
                    self.report.data["runs_observed"].append({"run_id": current, "run_index": index})
                    self._current_run, self._current_index = current, index
        for name in ("decisions", "fills", "settlements"):
            for row in recent[name] or []:
                if name == "settlements":
                    event = PaperSettlementEvent.from_dict(_object(row))
                    window = event.settlement.window_id
                    cash = _number(event.cash_after, nonnegative=True)
                    _number(event.equity, nonnegative=True)
                else:
                    decision = PaperDecisionEvent.from_dict(_object(row))
                    window = decision.decision.window_id
                known = self._runs.get(row["run_id"])
                if known is None:
                    _run_id(row["run_id"])
                    if rows is not None and len(rows) < self.history_limit:
                        raise ObservationError("MIXED_HISTORY_RUN")
                    # Bounded independent history pages can reach older than the run page.
                    self.report.issue("HISTORY_IDENTITY_OUTSIDE_OBSERVED_PAGE")
                elif window != known["identity"][1]:
                    raise ObservationError("MIXED_HISTORY_WINDOW")
                elif name == "settlements":
                    known["settlement_cash"] = cash
        for known in self._runs.values():
            index, window, market_id, predecessor, opening = known["identity"]
            previous = self._runs.get(predecessor)
            if previous is not None:
                if previous["identity"][0] != index - 1:
                    raise ObservationError("INVALID_RUN_CHAIN")
                cash = previous["settlement_cash"]
                if cash is None:
                    self.report.issue("SETTLEMENT_HANDOFF_NOT_OBSERVED")
                elif not math.isclose(cash, opening, rel_tol=1e-10, abs_tol=1e-8):
                    raise ObservationError("SETTLEMENT_OPENING_CASH_MISMATCH")

    def _account(self, account: dict[str, Any]) -> None:
        target = self.report.data["account"]
        equity = _number(account["equity"], nonnegative=True)
        cash = _number(account["cash"], nonnegative=True)
        drawdown = _number(account["drawdown"], nonnegative=True)
        if target["initial_equity"] is None:
            target["initial_equity"] = equity
        target.update(final_equity=equity, final_cash=cash,
                      realized_pnl=_number(account["realized_pnl"]),
                      unrealized_pnl=_number(account["unrealized_pnl"]),
                      fees=_number(account["total_fees"], nonnegative=True),
                      max_observed_drawdown=max(target["max_observed_drawdown"] or 0.0, drawdown))

    def _aggregate(self, status: dict[str, Any], at: float, *, final: bool) -> None:
        state = status["state"]
        states = self.report.data["states"]
        states.setdefault(state, {"samples": 0, "observed_ms": 0})["samples"] += 1
        if self._last_sample_time is not None and self._last_state is not None:
            states[self._last_state]["observed_ms"] += max(0, round((at - self._last_sample_time) * 1000))
        self._last_sample_time, self._last_state = at, state
        if state in {"DISCOVERING", "SYNCING", "DEGRADED", "SETTLEMENT_PENDING", "ROLLING_OVER"} and not final:
            self.report.issue("TRANSIENT_NON_RUNNING_STATE", informational=state in {"DISCOVERING", "SYNCING", "ROLLING_OVER"})
        feeds = _object(status["feeds"])
        if set(feeds) != {"binance", "polymarket", "chainlink"}:
            raise ObservationError("INVALID_API_SCHEMA")
        for name, health in feeds.items():
            health = _object(health)
            if type(health.get("fresh")) is not bool:
                raise ObservationError("INVALID_API_SCHEMA")
            record = self.report.data["feeds"].setdefault(name, {"samples": 0, "fresh_samples": 0, "fresh_ratio": 0.0})
            if not final:
                record["samples"] += 1
                record["fresh_samples"] += int(health["fresh"])
                record["fresh_ratio"] = record["fresh_samples"] / record["samples"]
        counters = status["counters"]
        if status["run_id"] != self._counter_run:
            # Status may publish the new run before its history section is readable.
            # Reset per-run observation baselines on status identity, not history availability.
            self._last_counters = {k: v for k, v in self._last_counters.items() if k == "settlement_completed"}
            self._counter_run = status["run_id"]
        for name in self.report.data["activity"]:
            key = "settlement_completed" if name == "settlements" else name
            count = _integer(counters[key])
            previous = self._last_counters.get(key, count)  # first poll is a baseline, not historic activity
            if count < previous:
                raise ObservationError("COUNTERS_REGRESSED_WITHIN_RUN")
            self.report.data["activity"][name] += count - previous
            self._last_counters[key] = count
