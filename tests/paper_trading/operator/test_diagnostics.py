from __future__ import annotations

import asyncio
import json

import pytest

from bigan.paper_trading.operator.diagnostics import (
    DiagnosticBuffer,
    FeedResyncRequired,
    HeartbeatTimeout,
)
from bigan.paper_trading.operator.diagnostics import (
    DiagnosticCode as Code,
)
from bigan.paper_trading.operator.transports import PublicWebSocketTransport
from tests.paper_trading.operator.test_feeds import _binance, _book, _delta, _polymarket, _snapshot
from tests.paper_trading.operator.test_pricing_inputs import _provider, _sample
from tests.paper_trading.operator.test_transports import FakeConnector, FakeSocket


def test_diagnostics_have_bounded_events_and_only_allowlisted_values():
    buffer = DiagnosticBuffer()
    for i in range(10_000):
        buffer.record(Code.DEPTH_INVALID, timestamp_ms=i)
    assert buffer.counts == {"DEPTH_INVALID": 10_000}
    assert len(buffer.recent) == 32
    assert buffer.recent[0]["timestamp_ms"] == 9968
    copy = buffer.to_dict()
    copy["recent"][0]["timestamp_ms"] = -1
    assert buffer.recent[0]["timestamp_ms"] == 9968
    with pytest.raises(ValueError):
        buffer.record("SECRET_TOKEN")
    with pytest.raises(ValueError):
        buffer.record(Code.DEPTH_INVALID, raw_payload="SECRET_TOKEN")


@pytest.mark.parametrize("mutation,code", [
    (lambda p: p.update(bids=[]), Code.DEPTH_EMPTY_SIDE),
    (lambda p: p["bids"][0].update(price="0.9"), Code.DEPTH_CROSSED),
    (lambda p: p["asks"][0].update(price="1.2"), Code.DEPTH_INVALID_PRICE),
    (lambda p: p["asks"][0].update(size="-1"), Code.DEPTH_INVALID_SIZE),
    (lambda p: p.update(timestamp=1002), Code.EVENT_FROM_FUTURE),
])
def test_market_rejection_records_reason_without_loosening_validation(mutation, code):
    feed = _polymarket()
    feed.begin_generation(1)
    payload = _book("yes-token", 1, 1000, bid="0.4", ask="0.5")
    mutation(payload)
    assert feed.ingest(payload, generation=1, received_at_ms=1001) is None
    assert feed.diagnostics.counts[code] == 1
    before = feed.diagnostics.to_dict()
    feed.disconnect()
    feed.begin_generation(2)
    assert feed.diagnostics.to_dict() == before
    assert not feed.health(now_ms=1002).fresh


def test_binance_reports_clock_lead_and_update_gap():
    feed = _binance()
    feed.begin_generation(1)
    assert not feed.ingest_delta(_delta(11, 11, ts_ms=1200), generation=1, received_at_ms=1000)
    assert feed.diagnostics.recent[-1]["code"] == Code.EVENT_FROM_FUTURE
    assert feed.diagnostics.recent[-1]["event_timestamp_ms"] == 1200
    assert feed.ingest_snapshot(_snapshot(), generation=1, received_at_ms=1300)
    assert not feed.ingest_delta(_delta(12, 12, ts_ms=1400), generation=1, received_at_ms=1401)
    assert feed.diagnostics.recent[-1]["code"] == Code.DEPTH_SEQUENCE_GAP
    assert feed.diagnostics.recent[-1]["expected_update_id"] == 11


def test_pricing_distinguishes_future_spot_oracle_stale_and_warmup():
    provider = _provider(lookback=60)
    assert provider(1000) is None
    assert provider.diagnostics.recent[-1]["code"] == Code.PRICING_MISSING_SAMPLES
    provider.ingest_spot(_sample(2000, 100, provider.spot_source))
    provider.ingest_oracle(_sample(1000, 100, provider.oracle_source))
    assert provider(1500) is None
    assert provider.diagnostics.recent[-1]["code"] == Code.PRICING_FUTURE_SPOT
    assert provider(2000) is None
    assert provider.diagnostics.recent[-1]["code"] == Code.PRICING_VOLATILITY_WARMUP
    provider.ingest_oracle(_sample(2100, 101, provider.oracle_source))
    assert provider(2000) is None
    assert provider.diagnostics.recent[-1]["code"] == Code.PRICING_FUTURE_ORACLE
    assert provider(4000) is None
    assert provider.diagnostics.counts[Code.PRICING_STALE_SPOT] == 1
    assert provider.diagnostics.counts[Code.PRICING_STALE_ORACLE] == 1
    assert provider.diagnostics.recent[-1]["oracle_timestamp_ms"] == 2100


@pytest.mark.parametrize("error,code", [
    (OSError("SECRET_ENDPOINT"), Code.WS_IO_FAILURE),
    (ValueError("SECRET_PAYLOAD"), Code.WS_INVALID_PAYLOAD),
    (FeedResyncRequired("SECRET_BODY"), Code.WS_REBOOTSTRAP_REQUIRED),
    (HeartbeatTimeout("SECRET_PONG"), Code.WS_HEARTBEAT_TIMEOUT),
])
async def test_transport_exports_only_typed_reason_before_disconnect(error, code):
    stop = asyncio.Event()
    events = []
    def record(reason, generation, timestamp):
        events.append((reason, generation, timestamp))
        stop.set()
    transport = PublicWebSocketTransport(
        endpoint="wss://stream.binance.us/ws", subscription={"method": "SUBSCRIBE"}, queue_size=2,
        reconnect_min_seconds=1, reconnect_max_seconds=2, heartbeat_interval_seconds=5,
        clock_ms=lambda: 1000, on_generation=lambda _: None, on_payload=lambda *_: None,
        on_disconnect=lambda: events.append("disconnected"), on_diagnostic=record,
        connect_factory=FakeConnector([FakeSocket([error])]),
    )
    await asyncio.wait_for(transport.run(stop), 1)
    assert events == [(code, 1, 1000), "disconnected"]
    assert "SECRET" not in json.dumps(events)
