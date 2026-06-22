"""Concrete public/read-only live market-feed adapters."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from typing import Any

from bigan.v8.paper.feed import (
    FeedHealthSnapshot,
    ReadOnlyFeedEvent,
    compute_feed_health,
)
from bigan.v8.paper.live_feed import (
    LIVE_READONLY_FEED_MODE,
    LiveFeedMetadata,
    LiveReadOnlyFeedConfig,
    LiveReadOnlyFeedError,
    build_live_feed_metadata,
)

JSONRequest = Callable[[str, float], Mapping[str, Any]]
ClockFn = Callable[[], float]
SleepFn = Callable[[float], None]


class PublicTickerLiveReadOnlyFeed:
    """Public REST ticker adapter that cannot place or cancel orders."""

    read_only = True
    write_capable = False
    paper_only = True
    capital_at_risk = False
    broker_exchange_write_enabled = False
    live_exchange_write_enabled = False
    feed_mode = LIVE_READONLY_FEED_MODE

    def __init__(
        self,
        *,
        config: LiveReadOnlyFeedConfig,
        request_json: JSONRequest | None = None,
        clock: ClockFn | None = None,
        sleep: SleepFn | None = None,
    ) -> None:
        self.config = config
        self._request_json = request_json or _request_json
        self._clock = clock or time.time
        self._sleep = sleep or time.sleep
        self._events: list[ReadOnlyFeedEvent] = []
        self._closed = False
        self._provider_disconnect_count = 0
        self._provider_reconnect_count = 0
        self._provider_error_count = 0
        self._stale_event_count = 0
        self._empty_response_count = 0
        self._rate_limit_count = 0
        self._last_successful_receive_ts: int | None = None
        self._started_monotonic: float | None = None
        self._ended_monotonic: float | None = None
        self._was_disconnected = False

    def iter_events(self) -> Iterator[ReadOnlyFeedEvent]:
        if self._closed:
            return
        self._started_monotonic = self._clock()
        sequence = 0
        while not self._closed:
            if (
                self.config.max_event_count is not None
                and sequence >= self.config.max_event_count
            ):
                break
            if self._elapsed_wall_clock_seconds() >= (
                self.config.expected_wall_clock_duration_seconds
            ):
                break
            try:
                payload = dict(
                    self._request_json(
                        _provider_url(self.config),
                        self.config.request_timeout_seconds,
                    )
                )
                if not payload:
                    self._empty_response_count += 1
                    self._fail_if_reconnect_budget_exhausted("empty provider response")
                    self._sleep(self.config.poll_interval_seconds)
                    continue
                event = self._event_from_payload(payload, sequence)
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    self._rate_limit_count += 1
                self._record_provider_error(exc)
                continue
            except Exception as exc:
                self._record_provider_error(exc)
                continue

            if self._was_disconnected:
                self._provider_reconnect_count += 1
                self._was_disconnected = False
            self._events.append(event)
            self._last_successful_receive_ts = event.received_ts
            sequence += 1
            yield event
            self._sleep(self.config.poll_interval_seconds)
        self._ended_monotonic = self._clock()

    def health_snapshot(self) -> FeedHealthSnapshot:
        base = compute_feed_health(
            tuple(self._events),
            max_allowed_gap_ms=int(self.config.max_allowed_gap_seconds * 1000),
            max_event_lag_ms=int(self.config.max_event_lag_seconds * 1000),
        )
        return replace(
            base,
            provider_disconnect_count=self._provider_disconnect_count,
            provider_reconnect_count=self._provider_reconnect_count,
            provider_error_count=self._provider_error_count,
            stale_event_count=self._stale_event_count,
            empty_response_count=self._empty_response_count,
            rate_limit_count=self._rate_limit_count,
            last_successful_receive_ts=self._last_successful_receive_ts,
        )

    def metadata_snapshot(
        self,
        *,
        ended_at: str,
        wall_clock_duration_seconds: int | None = None,
    ) -> LiveFeedMetadata:
        observed_seconds = (
            self._elapsed_wall_clock_seconds()
            if wall_clock_duration_seconds is None
            else wall_clock_duration_seconds
        )
        return build_live_feed_metadata(
            config=self.config,
            ended_at=ended_at,
            wall_clock_duration_seconds=int(observed_seconds),
        )

    def close(self) -> None:
        self._closed = True
        if self._ended_monotonic is None:
            self._ended_monotonic = self._clock()

    def _event_from_payload(
        self,
        payload: Mapping[str, Any],
        sequence: int,
    ) -> ReadOnlyFeedEvent:
        now_ms = int(self._clock() * 1000)
        bid = _required_positive_float(payload, "bidPrice", "bid_price")
        ask = _required_positive_float(payload, "askPrice", "ask_price")
        mid_payload = payload.get("midPrice", payload.get("mid_price"))
        mid = (bid + ask) / 2.0 if mid_payload is None else float(mid_payload)
        if mid <= 0.0:
            raise LiveReadOnlyFeedError("mid price must be positive")
        volume = float(payload.get("volume", payload.get("quoteVolume", 0.0)) or 0.0)
        if volume < 0.0:
            raise LiveReadOnlyFeedError("volume must be non-negative")
        trade_count = int(payload.get("count", payload.get("trade_count", 0)) or 0)
        close_time = int(payload.get("closeTime", payload.get("event_ts", now_ms)))
        if now_ms - close_time > int(self.config.max_stale_seconds * 1000):
            self._stale_event_count += 1
        spread_bps = ((ask - bid) / mid) * 10_000.0
        return ReadOnlyFeedEvent(
            event_ts=close_time,
            received_ts=now_ms,
            source=self.config.provider_name,
            instrument_id=self.config.instrument_id,
            bid_price=bid,
            ask_price=ask,
            mid_price=mid,
            volume=volume,
            trade_count=trade_count,
            spread_bps=spread_bps,
            feed_sequence=sequence,
            read_only=True,
            write_capable=False,
            paper_only=True,
            capital_at_risk=False,
        )

    def _record_provider_error(self, exc: Exception) -> None:
        self._provider_error_count += 1
        self._provider_disconnect_count += 1
        self._was_disconnected = True
        self._fail_if_reconnect_budget_exhausted(str(exc))
        self._sleep(self.config.poll_interval_seconds)

    def _fail_if_reconnect_budget_exhausted(self, reason: str) -> None:
        failures = max(self._provider_disconnect_count, self._empty_response_count)
        if failures > self.config.max_reconnect_attempts:
            raise LiveReadOnlyFeedError(
                "live provider reconnect budget exhausted: " + reason
            )

    def _elapsed_wall_clock_seconds(self) -> float:
        if self._started_monotonic is None:
            return 0.0
        end = self._ended_monotonic if self._ended_monotonic is not None else self._clock()
        return max(0.0, end - self._started_monotonic)


def create_public_live_readonly_feed(
    config: LiveReadOnlyFeedConfig,
) -> PublicTickerLiveReadOnlyFeed:
    """Create the default public/read-only market data adapter."""

    return PublicTickerLiveReadOnlyFeed(config=config)


def _provider_url(config: LiveReadOnlyFeedConfig) -> str:
    parsed = urllib.parse.urlparse(config.provider_endpoint)
    query = dict(urllib.parse.parse_qsl(parsed.query))
    query.setdefault("symbol", config.instrument_id)
    return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query)))


def _request_json(url: str, timeout: float) -> Mapping[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "bigan-v8-live-readonly-feed/1.0"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if getattr(response, "status", 200) == 429:
            raise urllib.error.HTTPError(
                url=url,
                code=429,
                msg="rate limited",
                hdrs=response.headers,
                fp=None,
            )
        return json.loads(response.read().decode("utf-8"))


def _required_positive_float(
    payload: Mapping[str, Any],
    *names: str,
) -> float:
    for name in names:
        if name in payload and payload[name] is not None:
            value = float(payload[name])
            if value <= 0.0:
                raise LiveReadOnlyFeedError(f"{name} must be positive")
            return value
    raise LiveReadOnlyFeedError(
        "provider payload missing required price field: " + "/".join(names)
    )
