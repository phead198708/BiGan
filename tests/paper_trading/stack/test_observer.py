from __future__ import annotations

import asyncio
import copy
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from bigan.paper_trading.stack.observer import (
    ObservationError,
    ObservationPolicy,
    PaperSoakObserver,
)
from tests.paper_trading.operator.test_runtime import _ready_operator


def make_observer(observation, **kwargs):
    return PaperSoakObserver(report=observation.report, **kwargs)


@pytest.mark.parametrize("mutation,code", [
    (lambda p: p.update(schema_version=99), "INVALID_API_SCHEMA"),
    (lambda p: p["status"].update(paper_only=False), "PAPER_SAFETY_VIOLATION"),
    (lambda p: p["status"]["safety"].update(wallet_signing_enabled=True), "PAPER_SAFETY_VIOLATION"),
    (lambda p: p["status"]["safety"].update(wallet_signing_enabled=0), "PAPER_SAFETY_VIOLATION"),
    (lambda p: p["account"].update(equity=-1), "NEGATIVE_ACCOUNT_VALUE"),
    (lambda p: p["account"].update(cash=-1), "NEGATIVE_ACCOUNT_VALUE"),
    (lambda p: p["account"].update(equity=float("nan")), "NONFINITE_API_NUMBER"),
    (lambda p: p["status"].update(state="FAILED"), "OPERATOR_FAILED"),
    (lambda p: p["operator_identity"].update(config_sha256="0" * 64), "CONFIG_IDENTITY_CHANGED"),
    (lambda p: p["status"].update(source_commit="changed"), "CONFIG_IDENTITY_CHANGED"),
    (lambda p: p["account"].update(run_id="paper-" + "f" * 24), "MIXED_ACCOUNT_RUN"),
    (lambda p: p["active_market"].update(market_id="different"), "MIXED_MARKET_IDENTITY"),
    (lambda p: p["recent"]["decisions"][0].update(run_id="paper-" + "f" * 24), "MIXED_HISTORY_RUN"),
    (lambda p: p["recent"]["runs"].clear(), "MIXED_RUN_HISTORY"),
    (lambda p: p.update(positions={}), "INVALID_API_SCHEMA"),
])
async def test_invalid_payloads_fail(observation, mutation, code):
    payload = copy.deepcopy(observation.payload)
    mutation(payload)
    with pytest.raises(ValueError, match=code):
        make_observer(observation).observe(payload, at=0)


async def test_rollover_and_settlement_handoff(observation):
    observer = make_observer(observation)
    observer.observe(observation.payload, at=0)
    observation.clock.now_ms = 900001
    await observation.operator.poll()
    await _ready_operator(observation.operator, observation.clock)
    payload = observation.reader.read()
    observer.observe(payload, at=1)
    assert observation.report.data["rollovers"] == 1
    assert len(observer._runs) == 2
    assert observation.report.data["activity"]["settlements"] == 1
    assert payload["recent"]["runs"][0]["opening_cash"] == payload["recent"]["runs"][1]["cash"]

    earlier = copy.deepcopy(observation.payload)
    earlier["status"]["updated_at_ms"] = earlier["generated_at_ms"] = observation.clock.now_ms
    with pytest.raises(ObservationError, match="RUN_INDEX_REGRESSED"):
        observer.observe(earlier, at=2)


async def test_settlement_wrong_next_opening(observation):
    observation.clock.now_ms = 900001
    await observation.operator.poll()
    payload = observation.reader.read()
    payload["recent"]["runs"][0]["opening_cash"] += 1
    with pytest.raises(ObservationError, match="SETTLEMENT_OPENING_CASH_MISMATCH"):
        make_observer(observation).observe(payload, at=0)


async def test_same_run_cannot_change_market(observation):
    observer = make_observer(observation)
    observer.observe(observation.payload, at=0)
    payload = copy.deepcopy(observation.payload)
    payload["recent"]["runs"][0]["market_id"] = "another-market"
    with pytest.raises(ObservationError, match="RUN_IDENTITY_CHANGED"):
        observer.observe(payload, at=1)


async def test_run_memory_cap_fails_instead_of_eviction(observation):
    observer = make_observer(observation, policy=ObservationPolicy(max_runs=1))
    observer.observe(observation.payload, at=0)
    observation.clock.now_ms = 900001
    await observation.operator.poll()
    with pytest.raises(ObservationError, match="OBSERVATION_RUN_CAPACITY_EXCEEDED"):
        observer.observe(observation.reader.read(), at=1)
    assert len(observer._runs) == 1


async def test_new_run_unavailable_history_does_not_reuse_old_counters(observation):
    observer = make_observer(observation)
    observer.observe(observation.payload, at=0)
    observation.clock.now_ms = 900001
    await observation.operator.poll()
    payload = observation.reader.read()
    payload["recent"] = dict.fromkeys(payload["recent"])
    observer.observe(payload, at=1)
    assert not observation.report.failed


async def test_stale_and_pending_deadlines(observation):
    observer = make_observer(observation, policy=ObservationPolicy(stale_seconds=1, rollover_seconds=1))
    payload = copy.deepcopy(observation.payload)
    payload.update(stale=True, status_age_ms=6000, generated_at_ms=payload["generated_at_ms"] + 6000)
    observer.observe(payload, at=0)
    with pytest.raises(ObservationError, match="STATUS_STALE_DEADLINE"):
        observer.observe(payload, at=2)
    payload = copy.deepcopy(observation.payload)
    payload["status"]["state"] = "SETTLEMENT_PENDING"
    observer = make_observer(observation, policy=ObservationPolicy(rollover_seconds=1))
    observer.observe(payload, at=0)
    with pytest.raises(ObservationError, match="ROLLOVER_DEADLINE"):
        observer.observe(payload, at=2)


async def test_optional_sections_and_no_market_are_not_hard_failures(observation):
    observer = make_observer(observation)
    payload = copy.deepcopy(observation.payload)
    payload.update(account=None, positions=None)
    payload["recent"]["fills"] = None
    payload["warnings"] = [{"code": "HISTORY_UNAVAILABLE", "section": "fills", "message": "unavailable"}]
    observer.observe(payload, at=0)
    assert not observation.report.failed
    assert observation.report.data["result"] == "WARN"
    payload["status"].update(run_id=None, active_market=None, last_decision=None, last_fill=None, state="DISCOVERING")
    payload.update(active_market=None, recent=dict.fromkeys(payload["recent"]))
    make_observer(observation).observe(payload, at=1)
    assert not observation.report.failed


async def test_memory_does_not_grow_with_polls(observation):
    observer = make_observer(observation)
    for i in range(2000):
        observer.observe(observation.payload, at=i)
    assert len(observer._runs) == 1
    assert len(observation.report.data["runs_observed"]) == 1
    assert len(json.dumps(observation.report.data)) < 6000
    assert observation.payload["recent"]["decisions"][0]["source_snapshot_id"] not in json.dumps(observation.report.data)


async def test_http_serial_polling_recovery_and_close(observation):
    observer = make_observer(observation)
    active, maximum, calls = 0, 0, []
    fail = True

    async def handler(request):
        nonlocal active, maximum, fail
        active += 1
        maximum = max(maximum, active)
        calls.append((request.method, request.path))
        try:
            await asyncio.sleep(0.002)
            if fail:
                fail = False
                return web.Response(status=503)
            if request.path == "/healthz":
                return web.json_response({"alive": True, "paper_only": True, "read_only": True,
                                          "operator_identity": observation.check.identity})
            if request.path == "/readyz":
                return web.json_response({"ready": True, "stale": False})
            return web.json_response(observation.payload)
        finally:
            active -= 1

    app = web.Application()
    app.router.add_get("/{path:.*}", handler)
    async with TestServer(app) as server:
        observer.url = str(server.make_url("")).rstrip("/")
        async with observer:
            assert await observer.poll() is False
            assert all(await asyncio.gather(observer.poll(), observer.poll(), observer.poll()))
            with pytest.raises(ValueError, match="endpoint"):
                await observer.get("/private")
        assert observer.http.closed
    assert maximum == 1
    assert all(method == "GET" and path in {"/healthz", "/readyz", "/api/v1/dashboard"} for method, path in calls)
    assert observation.report.data["polls"]["failed"] == 1
    assert observation.report.data["polls"]["successful"] == 3


async def test_timeout_and_continuous_unreadable(observation):
    observer = make_observer(observation, policy=ObservationPolicy(request_timeout=0.02, unreadable_seconds=0.02))

    async def handler(request):
        await asyncio.sleep(0.1)
        return web.json_response({})

    app = web.Application()
    app.router.add_get("/healthz", handler)
    async with TestServer(app) as server:
        observer.url = str(server.make_url("")).rstrip("/")
        async with observer:
            assert not await observer.poll()
            await asyncio.sleep(0.03)
            assert not await observer.poll()
        assert observer.http.closed
    assert observation.report.failed
    assert observation.report.data["polls"]["longest_failure_streak_ms"] >= 20
