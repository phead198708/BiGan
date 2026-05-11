"""Pydantic models for Polymarket CLOB ``market`` channel events.

The schemas mirror the official docs:
https://docs.polymarket.com/developers/CLOB/websocket/market-channel

Every message carries a server-side ``timestamp`` (ms epoch as string). We
augment each parsed event with ``receive_time`` (our local arrival epoch ms,
populated by the WS client).

Performance note: ``orjson`` is used for the initial bytes -> dict step; the
pydantic ``model_validate`` step then does field coercion. Both steps are
candidate hot paths for a future Rust replacement (see package docstring).
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

# ---------------------------------------------------------------------------
# Common types
# ---------------------------------------------------------------------------


def _as_int_ms(v: Any) -> int:
    """Coerce a stringified ms-epoch timestamp into an int."""
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        return int(v)
    if isinstance(v, float):
        return int(v)
    raise ValueError(f"cannot coerce {v!r} to int ms epoch")


MsEpoch = Annotated[int, BeforeValidator(_as_int_ms)]


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class PriceLevel(BaseModel):
    """A single price level in a book/depth payload."""

    model_config = ConfigDict(frozen=True)

    price: Decimal
    size: Decimal


class EventType(StrEnum):
    BOOK = "book"
    PRICE_CHANGE = "price_change"
    BEST_BID_ASK = "best_bid_ask"
    LAST_TRADE_PRICE = "last_trade_price"
    TICK_SIZE_CHANGE = "tick_size_change"
    NEW_MARKET = "new_market"
    MARKET_RESOLVED = "market_resolved"


# ---------------------------------------------------------------------------
# Event payloads
# ---------------------------------------------------------------------------


class BaseEvent(BaseModel):
    """Common fields present across most market-channel events.

    ``receive_time`` is filled in by our WS client at message arrival; the
    server does not emit it. It is the canonical clock for downstream gap
    detection and ingestion lag metrics.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    event_type: EventType
    timestamp: MsEpoch = Field(
        ...,
        description="Server-side exchange_time (ms epoch).",
    )
    receive_time: MsEpoch | None = Field(
        default=None,
        description="Our local arrival timestamp (ms epoch); populated by the WS client.",
    )


class BookEvent(BaseEvent):
    """Full order-book snapshot for one asset_id."""

    event_type: Literal[EventType.BOOK] = EventType.BOOK
    asset_id: str
    market: str
    bids: list[PriceLevel]
    asks: list[PriceLevel]
    hash: str


class PriceChange(BaseModel):
    """A single delta inside a ``price_change`` event."""

    model_config = ConfigDict(extra="allow")

    asset_id: str
    price: Decimal
    size: Decimal
    side: Side
    hash: str
    best_bid: Decimal | None = None
    best_ask: Decimal | None = None


class PriceChangeEvent(BaseEvent):
    """Incremental order-book delta. Size ``0`` means the level was removed."""

    event_type: Literal[EventType.PRICE_CHANGE] = EventType.PRICE_CHANGE
    market: str
    price_changes: list[PriceChange]


class BestBidAskEvent(BaseEvent):
    """Top-of-book ticker, only when ``custom_feature_enabled=True``."""

    event_type: Literal[EventType.BEST_BID_ASK] = EventType.BEST_BID_ASK
    market: str
    asset_id: str
    best_bid: Decimal | None = None
    best_ask: Decimal | None = None
    spread: Decimal | None = None


class LastTradePriceEvent(BaseEvent):
    """A single executed trade."""

    event_type: Literal[EventType.LAST_TRADE_PRICE] = EventType.LAST_TRADE_PRICE
    asset_id: str
    market: str
    price: Decimal
    side: Side
    size: Decimal
    fee_rate_bps: Decimal | None = None


class TickSizeChangeEvent(BaseEvent):
    event_type: Literal[EventType.TICK_SIZE_CHANGE] = EventType.TICK_SIZE_CHANGE
    asset_id: str
    market: str
    old_tick_size: Decimal
    new_tick_size: Decimal


class NewMarketEvent(BaseEvent):
    """Lifecycle: a new market just opened."""

    event_type: Literal[EventType.NEW_MARKET] = EventType.NEW_MARKET
    # Schema not fully specified in docs; allow extra and surface raw payload.


class MarketResolvedEvent(BaseEvent):
    """Lifecycle: a market just resolved."""

    event_type: Literal[EventType.MARKET_RESOLVED] = EventType.MARKET_RESOLVED


MarketEvent = (
    BookEvent
    | PriceChangeEvent
    | BestBidAskEvent
    | LastTradePriceEvent
    | TickSizeChangeEvent
    | NewMarketEvent
    | MarketResolvedEvent
)


_EVENT_REGISTRY: dict[str, type[BaseEvent]] = {
    EventType.BOOK.value: BookEvent,
    EventType.PRICE_CHANGE.value: PriceChangeEvent,
    EventType.BEST_BID_ASK.value: BestBidAskEvent,
    EventType.LAST_TRADE_PRICE.value: LastTradePriceEvent,
    EventType.TICK_SIZE_CHANGE.value: TickSizeChangeEvent,
    EventType.NEW_MARKET.value: NewMarketEvent,
    EventType.MARKET_RESOLVED.value: MarketResolvedEvent,
}


class UnknownEvent(Exception):
    """Raised when an incoming payload has no recognised ``event_type``."""


def parse_event(payload: dict[str, Any], *, receive_time_ms: int | None = None) -> MarketEvent:
    """Dispatch a raw decoded JSON dict into the correct event model.

    Args:
        payload: dict produced by ``orjson.loads`` (or equivalent).
        receive_time_ms: local arrival timestamp; injected before validation so
            it survives into the stored model.

    Raises:
        UnknownEvent: if ``event_type`` is missing or not recognised.
    """
    et = payload.get("event_type")
    if et is None:
        raise UnknownEvent(f"payload missing event_type: {payload!r}")
    model_cls = _EVENT_REGISTRY.get(et)
    if model_cls is None:
        raise UnknownEvent(f"unknown event_type={et!r}")
    if receive_time_ms is not None:
        # Inject our local clock if caller hasn't set it.
        payload = {**payload, "receive_time": receive_time_ms}
    return model_cls.model_validate(payload)
