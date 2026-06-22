"""Concrete public/read-only live market-feed adapters."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from bigan.v8.paper.feed import (
    FeedHealthSnapshot,
    ReadOnlyFeedEvent,
    compute_feed_health,
)
from bigan.v8.paper.live_feed import (
    LIVE_READONLY_FEED_MODE,
    LiveFeedMetadata,
    LiveProviderPayloadError,
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
        self._started_epoch_seconds: float | None = None
        self._ended_epoch_seconds: float | None = None
        self._started_at_wall_clock: str | None = None
        self._ended_at_wall_clock: str | None = None
        self._was_disconnected = False

    def iter_events(self) -> Iterator[ReadOnlyFeedEvent]:
        if self._closed:
            return
        self._started_epoch_seconds = self._clock()
        self._started_at_wall_clock = _utc_iso(self._started_epoch_seconds)
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
            except LiveReadOnlyFeedError:
                self._mark_ended()
                raise
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
        self._mark_ended()

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
            started_at_wall_clock=self._started_at_wall_clock,
            ended_at_wall_clock=self._ended_at_wall_clock or ended_at,
            wall_clock_duration_seconds=int(observed_seconds),
        )

    def close(self) -> None:
        self._closed = True
        self._mark_ended()

    def _event_from_payload(
        self,
        payload: Mapping[str, Any],
        sequence: int,
    ) -> ReadOnlyFeedEvent:
        now_ms = int(self._clock() * 1000)
        bid = _required_positive_float(
            payload,
            ("bidPrice", "bid_price", "bid"),
            missing_reason="missing_bid_price",
            non_positive_reason="non_positive_bid_price",
        )
        ask = _required_positive_float(
            payload,
            ("askPrice", "ask_price", "ask"),
            missing_reason="missing_ask_price",
            non_positive_reason="non_positive_ask_price",
        )
        mid_payload = payload.get(
            "midPrice",
            payload.get("mid_price", payload.get("price")),
        )
        try:
            mid = (bid + ask) / 2.0 if mid_payload is None else float(mid_payload)
        except (TypeError, ValueError) as exc:
            raise LiveProviderPayloadError(
                "invalid_mid_price",
                "mid price must be numeric",
            ) from exc
        if mid <= 0.0:
            raise LiveProviderPayloadError(
                "non_positive_mid_price",
                "mid price must be positive",
            )
        if ask < bid:
            raise LiveProviderPayloadError(
                "ask_below_bid_price",
                "ask price must be greater than or equal to bid price",
            )
        try:
            volume = float(payload.get("volume", payload.get("quoteVolume", 0.0)) or 0.0)
        except (TypeError, ValueError) as exc:
            raise LiveProviderPayloadError(
                "invalid_volume",
                "volume must be numeric",
            ) from exc
        if volume < 0.0:
            raise LiveProviderPayloadError(
                "negative_volume",
                "volume must be non-negative",
            )
        try:
            trade_count = int(payload.get("count", payload.get("trade_count", 0)) or 0)
        except (TypeError, ValueError) as exc:
            raise LiveProviderPayloadError(
                "invalid_trade_count",
                "trade_count must be an integer",
            ) from exc
        try:
            close_time = _event_timestamp_ms(payload, default_ts=now_ms)
        except (TypeError, ValueError) as exc:
            raise LiveProviderPayloadError(
                "invalid_event_timestamp",
                "event timestamp must be an integer",
            ) from exc
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
                "live provider reconnect budget exhausted: " + reason,
                reason_codes=(
                    "provider_error_breach",
                    "provider_reconnect_budget_exhausted",
                ),
            )

    def _elapsed_wall_clock_seconds(self) -> float:
        if self._started_epoch_seconds is None:
            return 0.0
        end = (
            self._ended_epoch_seconds
            if self._ended_epoch_seconds is not None
            else self._clock()
        )
        return max(0.0, end - self._started_epoch_seconds)

    def _mark_ended(self) -> None:
        if self._ended_epoch_seconds is None:
            self._ended_epoch_seconds = self._clock()
            self._ended_at_wall_clock = _utc_iso(self._ended_epoch_seconds)


def create_public_live_readonly_feed(
    config: LiveReadOnlyFeedConfig,
) -> PublicTickerLiveReadOnlyFeed:
    """Create the default public/read-only market data adapter."""

    return PublicTickerLiveReadOnlyFeed(config=config)


def _provider_url(config: LiveReadOnlyFeedConfig) -> str:
    endpoint = (
        config.provider_endpoint.replace("{instrument_id}", config.instrument_id)
        .replace("{instrument}", config.instrument_id)
        .replace("{symbol}", config.instrument_id)
    )
    parsed = urllib.parse.urlparse(endpoint)
    if "binance.com" not in parsed.netloc:
        return endpoint
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
    names: tuple[str, ...],
    *,
    missing_reason: str,
    non_positive_reason: str,
) -> float:
    for name in names:
        if name in payload and payload[name] is not None:
            try:
                value = float(payload[name])
            except (TypeError, ValueError) as exc:
                raise LiveProviderPayloadError(
                    non_positive_reason.replace("non_positive", "invalid"),
                    f"{name} must be numeric",
                ) from exc
            if value <= 0.0:
                raise LiveProviderPayloadError(
                    non_positive_reason,
                    f"{name} must be positive",
                )
            return value
    raise LiveProviderPayloadError(
        missing_reason,
        "provider payload missing required price field: " + "/".join(names),
    )


def _event_timestamp_ms(payload: Mapping[str, Any], *, default_ts: int) -> int:
    value = payload.get("closeTime", payload.get("event_ts", payload.get("time")))
    if value is None:
        return default_ts
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value)
    if text.isdigit():
        return int(text)
    return int(_parse_provider_iso8601(text).timestamp() * 1000)


def _parse_provider_iso8601(value: str) -> datetime:
    text = value
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if "." in text:
        before, after = text.split(".", 1)
        if "+" in after:
            fraction, timezone = after.split("+", 1)
            text = f"{before}.{fraction[:6]}+{timezone}"
        elif "-" in after:
            fraction, timezone = after.split("-", 1)
            text = f"{before}.{fraction[:6]}-{timezone}"
        else:
            text = f"{before}.{after[:6]}"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _utc_iso(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
