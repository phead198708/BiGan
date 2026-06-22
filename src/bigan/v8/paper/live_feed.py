"""Live read-only market-feed contracts for v8 paper runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from bigan.v8.paper.feed import ReadOnlyFeedError

LIVE_READONLY_FEED_SCHEMA_VERSION = "bigan-v8-live-readonly-feed-v1"
DETERMINISTIC_REPLAY_FEED_MODE = "deterministic-replay"
LIVE_READONLY_FEED_MODE = "live-readonly"
DEFAULT_LIVE_FEED_STARTED_AT = "2026-06-22T08:00:00Z"
DEFAULT_PUBLIC_PROVIDER_NAME = "binance_public_24hr_ticker"
DEFAULT_PUBLIC_PROVIDER_ENDPOINT = "https://api.binance.com/api/v3/ticker/24hr"
DEFAULT_PUBLIC_INSTRUMENT_ID = "BTCUSDT"

FeedMode = Literal["deterministic-replay", "live-readonly"]


class LiveReadOnlyFeedError(ReadOnlyFeedError):
    """Raised when a live read-only feed cannot be consumed safely."""


@dataclass(frozen=True, slots=True)
class LiveReadOnlyFeedConfig:
    """Configuration for one public/read-only live market data adapter."""

    provider_name: str = DEFAULT_PUBLIC_PROVIDER_NAME
    provider_endpoint: str = DEFAULT_PUBLIC_PROVIDER_ENDPOINT
    instrument_id: str = DEFAULT_PUBLIC_INSTRUMENT_ID
    poll_interval_seconds: float = 60.0
    request_timeout_seconds: float = 10.0
    max_reconnect_attempts: int = 3
    max_allowed_gap_seconds: float = 120.0
    max_event_lag_seconds: float = 10.0
    max_stale_seconds: float = 120.0
    expected_wall_clock_duration_seconds: int = 24 * 60 * 60
    started_at: str = DEFAULT_LIVE_FEED_STARTED_AT
    max_event_count: int | None = None
    read_only: bool = True
    write_capable: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False

    def __post_init__(self) -> None:
        if not self.provider_name.strip():
            raise ValueError("provider_name is required")
        if not self.provider_endpoint.strip():
            raise ValueError("provider_endpoint is required")
        if not self.instrument_id.strip():
            raise ValueError("instrument_id is required")
        if self.poll_interval_seconds <= 0.0:
            raise ValueError("poll_interval_seconds must be positive")
        if self.request_timeout_seconds <= 0.0:
            raise ValueError("request_timeout_seconds must be positive")
        if self.max_reconnect_attempts < 0:
            raise ValueError("max_reconnect_attempts must be non-negative")
        if self.max_allowed_gap_seconds <= 0.0:
            raise ValueError("max_allowed_gap_seconds must be positive")
        if self.max_event_lag_seconds < 0.0:
            raise ValueError("max_event_lag_seconds must be non-negative")
        if self.max_stale_seconds <= 0.0:
            raise ValueError("max_stale_seconds must be positive")
        if self.expected_wall_clock_duration_seconds <= 0:
            raise ValueError("expected_wall_clock_duration_seconds must be positive")
        if self.max_event_count is not None and self.max_event_count <= 0:
            raise ValueError("max_event_count must be positive when provided")
        if not self.started_at.strip():
            raise ValueError("started_at is required")
        if self.read_only is not True:
            raise LiveReadOnlyFeedError("live feed config must be read-only")
        if self.write_capable is not False:
            raise LiveReadOnlyFeedError("write-capable live feeds are forbidden")
        if self.paper_only is not True:
            raise LiveReadOnlyFeedError("live feed config must be paper-only")
        if self.capital_at_risk is not False:
            raise LiveReadOnlyFeedError("live feed config cannot put capital at risk")
        if self.broker_exchange_write_enabled:
            raise LiveReadOnlyFeedError("broker/exchange writes are forbidden")
        if self.live_exchange_write_enabled:
            raise LiveReadOnlyFeedError("live exchange writes are forbidden")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LiveFeedMetadata:
    """Provider and wall-clock metadata for a live read-only feed run."""

    schema_version: str
    feed_mode: FeedMode
    provider_name: str
    provider_endpoint_or_endpoint_type: str
    instrument_id: str
    poll_interval_seconds: float
    request_timeout_seconds: float
    started_at: str
    ended_at: str
    wall_clock_duration_seconds: int
    read_only: bool
    write_capable: bool
    paper_only: bool
    capital_at_risk: bool
    broker_exchange_write_enabled: bool
    live_exchange_write_enabled: bool

    def __post_init__(self) -> None:
        if self.feed_mode not in ("deterministic-replay", "live-readonly"):
            raise ValueError("feed_mode must be deterministic-replay or live-readonly")
        if self.wall_clock_duration_seconds < 0:
            raise ValueError("wall_clock_duration_seconds must be non-negative")
        if self.read_only is not True:
            raise LiveReadOnlyFeedError("live feed metadata must be read-only")
        if self.write_capable is not False:
            raise LiveReadOnlyFeedError("live feed metadata cannot be write-capable")
        if self.paper_only is not True:
            raise LiveReadOnlyFeedError("live feed metadata must be paper-only")
        if self.capital_at_risk is not False:
            raise LiveReadOnlyFeedError("live feed metadata cannot put capital at risk")
        if self.broker_exchange_write_enabled:
            raise LiveReadOnlyFeedError("broker/exchange writes are forbidden")
        if self.live_exchange_write_enabled:
            raise LiveReadOnlyFeedError("live exchange writes are forbidden")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_live_feed_metadata(
    *,
    config: LiveReadOnlyFeedConfig,
    ended_at: str,
    wall_clock_duration_seconds: int,
) -> LiveFeedMetadata:
    """Build deterministic metadata for one live read-only feed run."""

    return LiveFeedMetadata(
        schema_version=LIVE_READONLY_FEED_SCHEMA_VERSION,
        feed_mode=LIVE_READONLY_FEED_MODE,
        provider_name=config.provider_name,
        provider_endpoint_or_endpoint_type=config.provider_endpoint,
        instrument_id=config.instrument_id,
        poll_interval_seconds=config.poll_interval_seconds,
        request_timeout_seconds=config.request_timeout_seconds,
        started_at=config.started_at,
        ended_at=ended_at,
        wall_clock_duration_seconds=wall_clock_duration_seconds,
        read_only=True,
        write_capable=False,
        paper_only=True,
        capital_at_risk=False,
        broker_exchange_write_enabled=False,
        live_exchange_write_enabled=False,
    )
