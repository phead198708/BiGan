"""Live read-only feed adapter tests for v8."""

from __future__ import annotations

from dataclasses import replace

import pytest

from bigan.v8.paper import (
    LiveProviderPayloadError,
    LiveReadOnlyFeedConfig,
    LiveReadOnlyFeedError,
    PublicTickerLiveReadOnlyFeed,
    build_feed_health_acceptance_report,
)


def test_public_live_adapter_emits_readonly_events() -> None:
    clock = _MutableClock(1_000.0)
    adapter = PublicTickerLiveReadOnlyFeed(
        config=_live_config(max_event_count=2),
        request_json=_ticker_request(clock),
        clock=clock,
        sleep=clock.advance,
    )

    events = list(adapter.iter_events())
    health = adapter.health_snapshot()

    assert len(events) == 2
    assert all(event.read_only is True for event in events)
    assert all(event.write_capable is False for event in events)
    assert all(event.paper_only is True for event in events)
    assert all(event.capital_at_risk is False for event in events)
    assert events[0].source == "mock_public_ticker"
    assert events[0].instrument_id == "BTCUSDT"
    assert health.feed_event_count == 2
    assert health.last_successful_receive_ts == events[-1].received_ts
    metadata = adapter.metadata_snapshot(ended_at="unused")
    assert metadata.started_at_wall_clock == "1970-01-01T00:16:40Z"
    assert metadata.ended_at_wall_clock == "1970-01-01T00:18:40Z"
    assert metadata.wall_clock_duration_seconds == 120
    assert metadata.configured_duration_seconds == 300


def test_public_live_adapter_records_disconnect_and_reconnect() -> None:
    clock = _MutableClock(1_000.0)
    calls = {"count": 0}

    def flaky_request(_url: str, _timeout: float) -> dict[str, object]:
        calls["count"] += 1
        if calls["count"] == 1:
            raise TimeoutError("temporary provider timeout")
        return _ticker_payload(clock)

    adapter = PublicTickerLiveReadOnlyFeed(
        config=_live_config(max_event_count=1, max_reconnect_attempts=1),
        request_json=flaky_request,
        clock=clock,
        sleep=clock.advance,
    )

    events = list(adapter.iter_events())
    health = adapter.health_snapshot()

    assert len(events) == 1
    assert health.provider_disconnect_count == 1
    assert health.provider_reconnect_count == 1
    assert health.provider_error_count == 1


def test_public_live_adapter_normalizes_coinbase_ticker_payload() -> None:
    clock = _MutableClock(1_000.0)

    def coinbase_request(_url: str, _timeout: float) -> dict[str, object]:
        return {
            "bid": "99.95",
            "ask": "100.05",
            "price": "100.00",
            "volume": "42.5",
            "time": "1970-01-01T00:16:39.123456789Z",
        }

    adapter = PublicTickerLiveReadOnlyFeed(
        config=_live_config(
            max_event_count=1,
            provider_endpoint="https://api.exchange.coinbase.com/products/BTC-USD/ticker",
            instrument_id="BTC-USD",
        ),
        request_json=coinbase_request,
        clock=clock,
        sleep=clock.advance,
    )

    events = list(adapter.iter_events())

    assert events[0].instrument_id == "BTC-USD"
    assert events[0].bid_price == 99.95
    assert events[0].ask_price == 100.05
    assert events[0].mid_price == 100.0
    assert events[0].event_ts == 999_123


def test_public_live_adapter_transport_timeout_uses_provider_error_reason() -> None:
    clock = _MutableClock(1_000.0)

    def timeout_request(_url: str, _timeout: float) -> dict[str, object]:
        raise TimeoutError("provider timeout")

    adapter = PublicTickerLiveReadOnlyFeed(
        config=_live_config(max_event_count=1, max_reconnect_attempts=0),
        request_json=timeout_request,
        clock=clock,
        sleep=clock.advance,
    )

    with pytest.raises(LiveReadOnlyFeedError, match="reconnect budget") as exc_info:
        list(adapter.iter_events())
    assert exc_info.value.reason_codes == (
        "provider_error_breach",
        "provider_reconnect_budget_exhausted",
    )


def test_public_live_adapter_missing_price_fails_closed() -> None:
    clock = _MutableClock(1_000.0)

    def missing_ask(_url: str, _timeout: float) -> dict[str, object]:
        return {"bidPrice": "100.0", "closeTime": int(clock() * 1000)}

    adapter = PublicTickerLiveReadOnlyFeed(
        config=_live_config(max_event_count=1, max_reconnect_attempts=0),
        request_json=missing_ask,
        clock=clock,
        sleep=clock.advance,
    )

    with pytest.raises(LiveProviderPayloadError, match="missing_ask_price") as exc_info:
        list(adapter.iter_events())
    assert exc_info.value.reason_codes == (
        "invalid_provider_payload",
        "missing_ask_price",
    )


def test_public_live_adapter_non_positive_price_fails_closed() -> None:
    clock = _MutableClock(1_000.0)

    def non_positive(_url: str, _timeout: float) -> dict[str, object]:
        payload = _ticker_payload(clock)
        payload["bidPrice"] = "0"
        return payload

    adapter = PublicTickerLiveReadOnlyFeed(
        config=_live_config(max_event_count=1, max_reconnect_attempts=0),
        request_json=non_positive,
        clock=clock,
        sleep=clock.advance,
    )

    with pytest.raises(
        LiveProviderPayloadError,
        match="non_positive_bid_price",
    ) as exc_info:
        list(adapter.iter_events())
    assert exc_info.value.reason_codes == (
        "invalid_provider_payload",
        "non_positive_bid_price",
    )


def test_public_live_adapter_stale_data_produces_reason_code() -> None:
    clock = _MutableClock(1_000.0)

    def stale_request(_url: str, _timeout: float) -> dict[str, object]:
        payload = _ticker_payload(clock)
        payload["closeTime"] = int((clock() - 300.0) * 1000)
        return payload

    adapter = PublicTickerLiveReadOnlyFeed(
        config=_live_config(max_event_count=1, max_stale_seconds=120.0),
        request_json=stale_request,
        clock=clock,
        sleep=clock.advance,
    )
    events = list(adapter.iter_events())
    report = build_feed_health_acceptance_report(
        adapter.health_snapshot(),
        heartbeat_count=len(events),
        max_allowed_gap_seconds=120.0,
        max_event_lag_seconds=10.0,
    )

    assert len(events) == 1
    assert adapter.health_snapshot().stale_event_count == 1
    assert report.passed is False
    assert report.stale_event_breach is True
    assert "stale_event_breach" in report.reason_codes


def test_feed_health_snapshot_rejects_write_capable_health() -> None:
    health = PublicTickerLiveReadOnlyFeed(
        config=_live_config(max_event_count=1),
        request_json=_ticker_request(_MutableClock(1_000.0)),
        clock=_MutableClock(1_000.0),
        sleep=lambda _seconds: None,
    ).health_snapshot()

    with pytest.raises(Exception, match="write-capable"):
        replace(health, write_capable=True)


def _live_config(
    *,
    max_event_count: int,
    max_reconnect_attempts: int = 3,
    max_stale_seconds: float = 120.0,
    provider_endpoint: str = "https://example.test/ticker",
    instrument_id: str = "BTCUSDT",
) -> LiveReadOnlyFeedConfig:
    return LiveReadOnlyFeedConfig(
        provider_name="mock_public_ticker",
        provider_endpoint=provider_endpoint,
        instrument_id=instrument_id,
        poll_interval_seconds=60.0,
        request_timeout_seconds=1.0,
        max_reconnect_attempts=max_reconnect_attempts,
        max_stale_seconds=max_stale_seconds,
        expected_wall_clock_duration_seconds=300,
        max_event_count=max_event_count,
    )


def _ticker_request(clock: _MutableClock):
    def request(_url: str, _timeout: float) -> dict[str, object]:
        return _ticker_payload(clock)

    return request


def _ticker_payload(clock: _MutableClock) -> dict[str, object]:
    return {
        "bidPrice": "99.95",
        "askPrice": "100.05",
        "weightedAvgPrice": "100.0",
        "volume": "1234.5",
        "count": 42,
        "closeTime": int(clock() * 1000),
    }


class _MutableClock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds
