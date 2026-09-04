from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

from bigan.paper_trading.operator.diagnostics import DiagnosticCode
from bigan.paper_trading.stack.diagnostics import TRACE_LIMIT, readiness_reasons, readiness_snapshot
from bigan.paper_trading.stack.observer import (
    ObservationError,
    ObservationPolicy,
    PaperSoakObserver,
    live_inputs_ready,
)
from bigan.paper_trading.stack.report import ReportWriter, SoakReport, load_completed_report


@pytest.mark.parametrize("component,field,code", [
    ("binance", "fresh", "BINANCE_NOT_FRESH"),
    ("polymarket", "synchronized", "POLYMARKET_UNSYNCHRONIZED"),
    ("chainlink", "connected", "CHAINLINK_DISCONNECTED"),
    ("alpha", "fresh", "ALPHA_NOT_FRESH"),
    ("pricing_inputs", "ready", "PRICING_NOT_READY"),
    ("pricing_inputs", "fresh", "PRICING_NOT_FRESH"),
    ("session", "healthy", "SESSION_UNHEALTHY"),
])
async def test_reasons_match_existing_gate_without_changing_it(observation, component, field, code):
    payload = copy.deepcopy(observation.payload)
    assert live_inputs_ready(payload) and not readiness_reasons(payload)
    status = payload["status"]
    target = status["feeds"][component] if component in status["feeds"] else status[component]
    target[field] = False
    assert not live_inputs_ready(payload)
    assert readiness_reasons(payload) == [code]


async def test_failure_snapshot_survives_shutdown_and_report_publication(observation, tmp_path):
    report = SoakReport(replace(observation.check, mock=False))
    observer = PaperSoakObserver(report=report, policy=ObservationPolicy(stale_seconds=1))
    observer.observe(observation.payload, at=0)
    payload = copy.deepcopy(observation.payload)
    payload["status"]["feeds"]["polymarket"].update(fresh=False, synchronized=False)
    payload["status"]["alpha"].update(fresh=False, age_ms=2501)
    observer.observe(payload, at=1)
    with pytest.raises(ObservationError, match="LIVE_INPUTS_UNAVAILABLE_DEADLINE"):
        observer.observe(payload, at=2.1)
    report.issue("LIVE_INPUTS_UNAVAILABLE_DEADLINE", hard=True)
    failure = copy.deepcopy(report.data["readiness_diagnostics"]["failure_snapshot"])
    assert failure["reasons"] == ["POLYMARKET_UNSYNCHRONIZED", "POLYMARKET_NOT_FRESH", "ALPHA_NOT_FRESH"]
    assert failure["components"]["alpha"]["age_ms"] == 2501
    await observation.operator.shutdown()
    observer.observe(observation.reader.read(), at=3, final=True)
    observer.freeze_readiness_failure()
    assert report.data["readiness_diagnostics"]["failure_snapshot"] == failure
    writer = ReportWriter(tmp_path / "diagnostic-report", output=tmp_path / "other-output")
    try:
        report.finish()
        writer.write(report)
    finally:
        writer.close()
    persisted = load_completed_report(tmp_path / "diagnostic-report")
    assert persisted["readiness_diagnostics"]["failure_snapshot"] == failure
    assert "ALPHA_NOT_FRESH" in report.markdown()


async def test_trace_bounds_startup_and_runtime_and_scrubs_unknown_data(observation):
    observer = PaperSoakObserver(report=observation.report)
    payload = copy.deepcopy(observation.payload)
    feed = payload["status"]["feeds"]["polymarket"]
    feed["diagnostics"] = {"counts": {"SECRET_COUNTER": 1, "DEPTH_CROSSED": 5}, "recent": [
        {"code": "SECRET_ERROR", "timestamp_ms": 1},
        {"code": {"SECRET": True}},
        {"code": "DEPTH_CROSSED", "timestamp_ms": 2, "raw_payload": "SECRET_PAYLOAD"},
    ]}
    payload["status"]["state_reason"] = "SECRET_REASON"
    payload["status"]["session"]["failure_reason"] = "SECRET_EXCEPTION"
    for _ in range(1000):
        payload["generated_at_ms"] += 1
        observer.record_readiness(payload, phase="startup")
    trace = observation.report.data["readiness_diagnostics"]
    assert len(trace["samples"]) == 1
    observer.freeze_readiness_failure()
    saved = copy.deepcopy(trace["failure_snapshot"])
    for _ in range(500):
        observer.record_readiness(payload, phase="runtime")
    assert len(trace["samples"]) == TRACE_LIMIT
    assert trace["evicted_samples"] == 501 - TRACE_LIMIT
    assert trace["failure_snapshot"] == saved
    assert "SECRET" not in json.dumps(trace)
    assert trace["samples"][-1]["diagnostics"]["polymarket"] == {
        "counts": {"DEPTH_CROSSED": 5}, "recent": [{"code": "DEPTH_CROSSED", "timestamp_ms": 2}],
    }


async def test_live_transport_error_reaches_projection_and_stale_generation_is_fenced(observation):
    operator = observation.operator
    await operator.record_transport_diagnostic(
        "polymarket", DiagnosticCode.WS_HEARTBEAT_TIMEOUT,
        window_generation=operator.generation, connection_generation=99, timestamp_ms=12345,
    )
    payload = dict(observation.payload, status=operator.status().to_dict())
    sample = readiness_snapshot(payload, phase="runtime")
    assert sample["diagnostics"]["polymarket"]["counts"]["WS_HEARTBEAT_TIMEOUT"] == 1
    await operator.record_transport_diagnostic(
        "polymarket", DiagnosticCode.WS_HEARTBEAT_TIMEOUT,
        window_generation=operator.generation - 1, connection_generation=99, timestamp_ms=12345,
    )
    assert operator.market_sync.diagnostics.counts["WS_HEARTBEAT_TIMEOUT"] == 1
    await operator.shutdown()
    assert operator.status().feeds["polymarket"]["diagnostics"]["counts"]["WS_HEARTBEAT_TIMEOUT"] == 1
