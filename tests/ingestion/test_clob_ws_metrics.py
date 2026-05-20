"""Ingestion-lag metric tests for the CLOB WebSocket client."""

from __future__ import annotations

import asyncio
import logging

import orjson
import pytest

from bigan.ingestion.clob_ws import (
    ClobWsClient,
    WsClientConfig,
    _connection_error_context,
    _consume_future_exception,
    _expected_connection_exception,
    _log_connection_failed,
)
from bigan.ingestion.config import IngestionSettings
from bigan.ingestion.metrics import REGISTRY


def _sample(name: str, labels: dict[str, str]) -> float:
    return float(REGISTRY.get_sample_value(name, labels) or 0.0)


def test_dispatch_observes_ingest_lag_metric_and_warns(monkeypatch, caplog) -> None:
    events = []

    async def handler(event, raw) -> None:  # type: ignore[no-untyped-def]
        events.append((event, raw))

    client = ClobWsClient(
        WsClientConfig(
            url="ws://example.invalid",
            ingest_lag_warn_seconds=0.1,
        ),
        handler,
    )
    labels = {"source": "polymarket", "event_type": "book"}
    before_count = _sample("bigan_ingest_lag_seconds_count", labels)
    before_sum = _sample("bigan_ingest_lag_seconds_sum", labels)
    monkeypatch.setattr("bigan.ingestion.clob_ws.time.time", lambda: 1_700_000_000.500)

    payload = {
        "event_type": "book",
        "asset_id": "tok-1",
        "market": "0xmkt",
        "bids": [],
        "asks": [],
        "timestamp": "1700000000000",
        "hash": "h0",
    }

    with caplog.at_level(logging.WARNING):
        asyncio.run(client._dispatch(orjson.dumps(payload)))

    assert len(events) == 1
    assert _sample("bigan_ingest_lag_seconds_count", labels) == before_count + 1
    assert _sample("bigan_ingest_lag_seconds_sum", labels) == pytest.approx(
        before_sum + 0.5
    )
    assert any(record.message == "ingest_lag.high" for record in caplog.records)


def test_keepalive_frame_refreshes_liveness_metric(monkeypatch) -> None:
    async def handler(event, raw) -> None:  # type: ignore[no-untyped-def]
        raise AssertionError("keepalive should not dispatch to handler")

    client = ClobWsClient(WsClientConfig(url="ws://example.invalid"), handler)
    monkeypatch.setattr("bigan.ingestion.clob_ws.time.time", lambda: 1_700_000_123.456)

    asyncio.run(client._dispatch(b"PONG"))

    assert REGISTRY.get_sample_value("bigan_last_event_receive_time_seconds") == (
        1_700_000_123.456
    )


def test_non_json_frame_refreshes_liveness_metric_and_counts_parse_error(
    monkeypatch,
) -> None:
    async def handler(event, raw) -> None:  # type: ignore[no-untyped-def]
        raise AssertionError("non-json frame should not dispatch to handler")

    client = ClobWsClient(WsClientConfig(url="ws://example.invalid"), handler)
    before = _sample("bigan_ws_parse_errors_total", {"kind": "json"})
    monkeypatch.setattr("bigan.ingestion.clob_ws.time.time", lambda: 1_700_000_456.789)

    asyncio.run(client._dispatch(b"subscribed"))

    assert REGISTRY.get_sample_value("bigan_last_event_receive_time_seconds") == (
        1_700_000_456.789
    )
    assert _sample("bigan_ws_parse_errors_total", {"kind": "json"}) == before + 1


def test_receive_loop_probes_idle_connection_before_reconnect() -> None:
    pings = 0

    async def handler(event, raw) -> None:  # type: ignore[no-untyped-def]
        raise AssertionError("idle ping should not dispatch to handler")

    client = ClobWsClient(
        WsClientConfig(
            url="ws://example.invalid",
            message_timeout_seconds=0.01,
            idle_probe_timeout_seconds=0.1,
        ),
        handler,
    )

    class QuietWs:
        async def recv(self):  # type: ignore[no-untyped-def]
            await asyncio.sleep(60)

        async def ping(self):  # type: ignore[no-untyped-def]
            nonlocal pings
            pings += 1
            client.cancel()
            pong = asyncio.get_running_loop().create_future()
            pong.set_result(None)
            return pong

    asyncio.run(client._receive_loop(QuietWs()))  # type: ignore[arg-type]

    assert pings == 1


def test_default_message_timeout_probes_before_soak_liveness_gate(monkeypatch) -> None:
    monkeypatch.delenv("BIGAN_WS_MESSAGE_TIMEOUT_SECONDS", raising=False)

    assert WsClientConfig(url="ws://example.invalid").message_timeout_seconds == 45.0
    assert IngestionSettings(_env_file=None).ws_message_timeout_seconds == 45.0
    assert WsClientConfig(url="ws://example.invalid").message_timeout_seconds < 60.0


def test_default_library_ping_timeout_disabled_in_favor_of_idle_probe(monkeypatch) -> None:
    monkeypatch.delenv("BIGAN_WS_PING_INTERVAL_SECONDS", raising=False)
    monkeypatch.delenv("BIGAN_WS_PING_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("BIGAN_WS_IDLE_PROBE_TIMEOUT_SECONDS", raising=False)

    cfg = WsClientConfig(url="ws://example.invalid")
    settings = IngestionSettings(_env_file=None)

    assert cfg.ping_interval_seconds == 20.0
    assert cfg.ping_timeout_seconds is None
    assert cfg.idle_probe_timeout_seconds == 10.0
    assert settings.ws_ping_interval_seconds == 20.0
    assert settings.ws_ping_timeout_seconds is None
    assert settings.ws_idle_probe_timeout_seconds == 10.0


def test_protocol_ping_loop_consumes_late_pong_exception() -> None:
    pings = 0

    async def handler(event, raw) -> None:  # type: ignore[no-untyped-def]
        raise AssertionError("protocol pings should not dispatch to handler")

    client = ClobWsClient(
        WsClientConfig(
            url="ws://example.invalid",
            ping_interval_seconds=0.01,
            ping_timeout_seconds=None,
        ),
        handler,
    )

    class PingWs:
        async def ping(self):  # type: ignore[no-untyped-def]
            nonlocal pings
            pings += 1
            future = asyncio.get_running_loop().create_future()
            future.set_exception(RuntimeError("connection already closed"))
            client.cancel()
            return future

    async def go() -> None:
        await client._protocol_ping_loop(PingWs())  # type: ignore[arg-type]
        await asyncio.sleep(0)

    asyncio.run(go())

    assert pings == 1


def test_consume_future_exception_marks_exception_retrieved() -> None:
    async def go() -> asyncio.Future[object]:
        future: asyncio.Future[object] = asyncio.get_running_loop().create_future()
        _consume_future_exception(future)
        future.set_exception(RuntimeError("closed"))
        await asyncio.sleep(0)
        return future

    future = asyncio.run(go())

    assert isinstance(future.exception(), RuntimeError)


def test_backoff_resets_after_stable_connection(monkeypatch) -> None:
    async def handler(event, raw) -> None:  # type: ignore[no-untyped-def]
        raise AssertionError("backoff test should not dispatch to handler")

    client = ClobWsClient(
        WsClientConfig(
            url="ws://example.invalid",
            reconnect_min_seconds=1.0,
            reconnect_reset_after_seconds=60.0,
        ),
        handler,
    )
    monkeypatch.setattr("bigan.ingestion.clob_ws.time.monotonic", lambda: 200.0)

    assert client._backoff_for_failed_connection(30.0, attempt_started_at=120.0) == 1.0
    assert client._backoff_for_failed_connection(30.0, attempt_started_at=180.0) == 30.0


def test_expected_connection_exception_unwraps_taskgroup_exception_group() -> None:
    expected = OSError("connection reset")
    wrapped = ExceptionGroup("unhandled errors in a TaskGroup", [expected])

    assert _expected_connection_exception(wrapped) is expected
    assert _expected_connection_exception(ValueError("bad payload")) is None


def test_connection_error_context_includes_plain_log_diagnostics() -> None:
    cause = TimeoutError("pong timed out")
    exc = OSError("network unreachable")
    exc.__cause__ = cause

    context = _connection_error_context(exc, backoff_s=2.5)

    assert context == {
        "err_type": "OSError",
        "err": "network unreachable",
        "close_code": None,
        "close_reason": None,
        "cause_type": "TimeoutError",
        "cause": "pong timed out",
        "backoff_s": 2.5,
    }


def test_connection_failed_log_message_includes_plain_diagnostics(caplog) -> None:
    cause = TimeoutError("pong timed out")
    exc = OSError("network unreachable")
    exc.__cause__ = cause

    with caplog.at_level(logging.WARNING):
        _log_connection_failed(exc, backoff_s=2.5)

    message = caplog.records[-1].getMessage()
    assert "ws.connection_failed" in message
    assert "err_type=OSError" in message
    assert "err='network unreachable'" in message
    assert "cause_type=TimeoutError" in message
    assert "cause='pong timed out'" in message
    assert "backoff_s=2.5" in message
