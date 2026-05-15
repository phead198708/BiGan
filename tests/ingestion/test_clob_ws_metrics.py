"""Ingestion-lag metric tests for the CLOB WebSocket client."""

from __future__ import annotations

import asyncio
import logging

import orjson
import pytest

from bigan.ingestion.clob_ws import (
    ClobWsClient,
    WsClientConfig,
    _expected_connection_exception,
)
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
            ping_timeout_seconds=0.1,
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


def test_expected_connection_exception_unwraps_taskgroup_exception_group() -> None:
    expected = OSError("connection reset")
    wrapped = ExceptionGroup("unhandled errors in a TaskGroup", [expected])

    assert _expected_connection_exception(wrapped) is expected
    assert _expected_connection_exception(ValueError("bad payload")) is None
