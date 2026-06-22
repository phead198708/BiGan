"""Read-only feed contracts for v8 paper shadow soak runs."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Protocol


class ReadOnlyFeedError(RuntimeError):
    """Raised when a feed can mutate external state or is otherwise unsafe."""


@dataclass(frozen=True, slots=True)
class ReadOnlyFeedEvent:
    """One read-only market-feed event used by a paper shadow run."""

    event_ts: int
    received_ts: int
    source: str
    instrument_id: str
    bid_price: float
    ask_price: float
    mid_price: float
    volume: float
    trade_count: int
    spread_bps: float
    feed_sequence: int
    read_only: bool = True
    write_capable: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False

    def __post_init__(self) -> None:
        if self.event_ts < 0 or self.received_ts < 0:
            raise ValueError("timestamps must be non-negative")
        if self.received_ts < self.event_ts:
            raise ValueError("received_ts must be greater than or equal to event_ts")
        if not self.source:
            raise ValueError("source is required")
        if not self.instrument_id:
            raise ValueError("instrument_id is required")
        for field_name in (
            "bid_price",
            "ask_price",
            "mid_price",
            "volume",
            "spread_bps",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if self.bid_price <= 0.0 or self.ask_price <= 0.0 or self.mid_price <= 0.0:
            raise ValueError("prices must be positive")
        if self.ask_price < self.bid_price:
            raise ValueError("ask_price must be greater than or equal to bid_price")
        if self.volume < 0.0:
            raise ValueError("volume must be non-negative")
        if self.trade_count < 0:
            raise ValueError("trade_count must be non-negative")
        if self.spread_bps < 0.0:
            raise ValueError("spread_bps must be non-negative")
        if self.feed_sequence < 0:
            raise ValueError("feed_sequence must be non-negative")
        if self.read_only is not True:
            raise ReadOnlyFeedError("feed events must be read-only")
        if self.write_capable is not False:
            raise ReadOnlyFeedError("write-capable feed events are forbidden")
        if self.paper_only is not True:
            raise ReadOnlyFeedError("feed events must be paper-only")
        if self.capital_at_risk is not False:
            raise ReadOnlyFeedError("feed events cannot put capital at risk")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeedHealthSnapshot:
    """Feed-health metrics for one bounded read-only paper shadow run."""

    feed_event_count: int
    first_event_ts: int | None
    last_event_ts: int | None
    feed_gap_count: int
    max_feed_gap_seconds: float
    feed_late_event_count: int
    feed_out_of_order_count: int
    read_only: bool = True
    write_capable: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False

    def __post_init__(self) -> None:
        if self.feed_event_count < 0:
            raise ValueError("feed_event_count must be non-negative")
        if self.feed_gap_count < 0:
            raise ValueError("feed_gap_count must be non-negative")
        if self.feed_late_event_count < 0:
            raise ValueError("feed_late_event_count must be non-negative")
        if self.feed_out_of_order_count < 0:
            raise ValueError("feed_out_of_order_count must be non-negative")
        if self.max_feed_gap_seconds < 0.0 or not math.isfinite(
            self.max_feed_gap_seconds
        ):
            raise ValueError("max_feed_gap_seconds must be finite and non-negative")
        if self.read_only is not True:
            raise ReadOnlyFeedError("feed health must be read-only")
        if self.write_capable is not False:
            raise ReadOnlyFeedError("feed health cannot be write-capable")
        if self.paper_only is not True:
            raise ReadOnlyFeedError("feed health must be paper-only")
        if self.capital_at_risk is not False:
            raise ReadOnlyFeedError("feed health cannot put capital at risk")
        if self.broker_exchange_write_enabled:
            raise ReadOnlyFeedError("broker/exchange writes are forbidden")
        if self.live_exchange_write_enabled:
            raise ReadOnlyFeedError("live exchange writes are forbidden")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeedHealthAcceptanceReport:
    """Hard-gate acceptance result for one read-only feed-health snapshot."""

    passed: bool
    reason_codes: tuple[str, ...]
    feed_gap_breach: bool
    feed_late_event_breach: bool
    feed_out_of_order_breach: bool
    heartbeat_missing: bool
    feed_event_count: int
    heartbeat_count: int
    max_allowed_gap_seconds: float
    max_event_lag_seconds: float
    read_only: bool = True
    write_capable: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise ValueError("passed must be a boolean")
        if any(not code for code in self.reason_codes):
            raise ValueError("reason_codes must not contain empty values")
        if self.feed_event_count < 0:
            raise ValueError("feed_event_count must be non-negative")
        if self.heartbeat_count < 0:
            raise ValueError("heartbeat_count must be non-negative")
        if self.max_allowed_gap_seconds <= 0.0:
            raise ValueError("max_allowed_gap_seconds must be positive")
        if self.max_event_lag_seconds < 0.0:
            raise ValueError("max_event_lag_seconds must be non-negative")
        if self.passed != (not self.reason_codes):
            raise ValueError("passed must match reason_codes emptiness")
        if self.read_only is not True:
            raise ReadOnlyFeedError("feed acceptance must be read-only")
        if self.write_capable is not False:
            raise ReadOnlyFeedError("feed acceptance cannot be write-capable")
        if self.paper_only is not True:
            raise ReadOnlyFeedError("feed acceptance must be paper-only")
        if self.capital_at_risk is not False:
            raise ReadOnlyFeedError("feed acceptance cannot put capital at risk")
        if self.broker_exchange_write_enabled:
            raise ReadOnlyFeedError("broker/exchange writes are forbidden")
        if self.live_exchange_write_enabled:
            raise ReadOnlyFeedError("live exchange writes are forbidden")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["reason_codes"] = list(self.reason_codes)
        return payload


class ReadOnlyMarketFeed(Protocol):
    """Minimal read-only market feed interface for paper shadow runs."""

    read_only: bool
    write_capable: bool

    def iter_events(self) -> Iterator[ReadOnlyFeedEvent]:
        """Yield read-only feed events in replay or feed order."""

    def health_snapshot(self) -> FeedHealthSnapshot:
        """Return the latest feed-health snapshot."""

    def close(self) -> None:
        """Release read-only feed resources."""


class DeterministicReplayFeed:
    """Deterministic replay feed used for CI-safe paper shadow soak tests."""

    def __init__(
        self,
        *,
        events: tuple[ReadOnlyFeedEvent, ...],
        read_only: bool = True,
        write_capable: bool = False,
        max_allowed_gap_seconds: float = 120.0,
        max_event_lag_seconds: float = 10.0,
    ) -> None:
        if read_only is not True:
            raise ReadOnlyFeedError("replay feed must be read-only")
        if write_capable is not False:
            raise ReadOnlyFeedError("write-capable replay feed is forbidden")
        if max_allowed_gap_seconds <= 0.0:
            raise ValueError("max_allowed_gap_seconds must be positive")
        if max_event_lag_seconds < 0.0:
            raise ValueError("max_event_lag_seconds must be non-negative")
        self.read_only = read_only
        self.write_capable = write_capable
        self._events = events
        self._max_allowed_gap_ms = int(max_allowed_gap_seconds * 1000)
        self._max_event_lag_ms = int(max_event_lag_seconds * 1000)
        self._closed = False
        self._health = compute_feed_health(
            events,
            max_allowed_gap_ms=self._max_allowed_gap_ms,
            max_event_lag_ms=self._max_event_lag_ms,
        )

    def iter_events(self) -> Iterator[ReadOnlyFeedEvent]:
        if self._closed:
            return
        yield from self._events

    def health_snapshot(self) -> FeedHealthSnapshot:
        return self._health

    def close(self) -> None:
        self._closed = True


def synthetic_readonly_feed_events(
    *,
    row_count: int,
    start_ts: int = 2_600_000,
    interval_ms: int = 60_000,
    source: str = "readonly_fixture",
    instrument_id: str = "btc-up",
) -> tuple[ReadOnlyFeedEvent, ...]:
    """Build deterministic read-only feed events for short or 24h replay runs."""

    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if interval_ms <= 0:
        raise ValueError("interval_ms must be positive")
    rows: list[ReadOnlyFeedEvent] = []
    for index in range(row_count):
        mid_price = 100.0 + 0.02 * (index % 17) + 0.001 * index
        spread_bps = 5.0 + (index % 4)
        half_spread = mid_price * spread_bps / 20_000.0
        event_ts = start_ts + index * interval_ms
        rows.append(
            ReadOnlyFeedEvent(
                event_ts=event_ts,
                received_ts=event_ts + 250 + 10 * (index % 3),
                source=source,
                instrument_id=instrument_id,
                bid_price=mid_price - half_spread,
                ask_price=mid_price + half_spread,
                mid_price=mid_price,
                volume=1_000.0 + 7.0 * (index % 11),
                trade_count=10 + (index % 5),
                spread_bps=spread_bps,
                feed_sequence=index,
            )
        )
    return tuple(rows)


def compute_feed_health(
    events: tuple[ReadOnlyFeedEvent, ...],
    *,
    max_allowed_gap_ms: int,
    max_event_lag_ms: int,
) -> FeedHealthSnapshot:
    """Compute deterministic feed-health metrics for read-only feed events."""

    gap_count = 0
    max_gap_ms = 0
    late_count = 0
    out_of_order_count = 0
    previous_ts: int | None = None
    for event in events:
        if event.received_ts - event.event_ts > max_event_lag_ms:
            late_count += 1
        if previous_ts is not None:
            gap_ms = event.event_ts - previous_ts
            if gap_ms < 0:
                out_of_order_count += 1
            else:
                max_gap_ms = max(max_gap_ms, gap_ms)
                if gap_ms > max_allowed_gap_ms:
                    gap_count += 1
        previous_ts = event.event_ts
    return FeedHealthSnapshot(
        feed_event_count=len(events),
        first_event_ts=None if not events else events[0].event_ts,
        last_event_ts=None if not events else events[-1].event_ts,
        feed_gap_count=gap_count,
        max_feed_gap_seconds=max_gap_ms / 1000.0,
        feed_late_event_count=late_count,
        feed_out_of_order_count=out_of_order_count,
    )


def build_feed_health_acceptance_report(
    feed_health: FeedHealthSnapshot,
    *,
    heartbeat_count: int,
    max_allowed_gap_seconds: float,
    max_event_lag_seconds: float,
) -> FeedHealthAcceptanceReport:
    """Build the fail-closed acceptance report used by Phase 6 monitoring."""

    if heartbeat_count < 0:
        raise ValueError("heartbeat_count must be non-negative")
    feed_gap_breach = feed_health.feed_gap_count > 0
    feed_late_event_breach = feed_health.feed_late_event_count > 0
    feed_out_of_order_breach = feed_health.feed_out_of_order_count > 0
    heartbeat_missing = heartbeat_count <= 0
    reason_codes: list[str] = []
    if feed_gap_breach:
        reason_codes.append("feed_gap_breach")
    if feed_late_event_breach:
        reason_codes.append("feed_late_event_breach")
    if feed_out_of_order_breach:
        reason_codes.append("feed_out_of_order_breach")
    if heartbeat_missing:
        reason_codes.append("heartbeat_missing")
    return FeedHealthAcceptanceReport(
        passed=not reason_codes,
        reason_codes=tuple(reason_codes),
        feed_gap_breach=feed_gap_breach,
        feed_late_event_breach=feed_late_event_breach,
        feed_out_of_order_breach=feed_out_of_order_breach,
        heartbeat_missing=heartbeat_missing,
        feed_event_count=feed_health.feed_event_count,
        heartbeat_count=heartbeat_count,
        max_allowed_gap_seconds=max_allowed_gap_seconds,
        max_event_lag_seconds=max_event_lag_seconds,
    )


def assert_readonly_feed_safe(feed: ReadOnlyMarketFeed) -> None:
    """Fail closed if a feed can write to an exchange or is not read-only."""

    if getattr(feed, "read_only", False) is not True:
        raise ReadOnlyFeedError("feed must be read-only")
    if getattr(feed, "write_capable", True) is not False:
        raise ReadOnlyFeedError("write-capable feed is forbidden")
