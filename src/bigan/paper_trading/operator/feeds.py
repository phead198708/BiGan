"""Bounded, generation-fenced state machines for public read-only feeds."""

from __future__ import annotations

import asyncio
import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar

from bigan.data.polymarket_clob import MarketSnapshot, PolymarketFeedHandler
from bigan.features.binance_ofi import BinanceOFICalculator

from .market_depth import MarketDepth

T = TypeVar("T")


class _DepthBookOverflow(ValueError):
    pass


class FeedConnectionState(StrEnum):
    DISCONNECTED = "DISCONNECTED"
    SYNCING = "SYNCING"
    READY = "READY"
    STALE = "STALE"


@dataclass(frozen=True, slots=True)
class FeedHealth:
    state: FeedConnectionState
    connected: bool
    synchronized: bool
    fresh: bool
    last_event_ts_ms: int | None
    age_ms: int | None
    last_message_received_ms: int | None
    gap_count: int
    reconnect_count: int
    error_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "connected": self.connected,
            "synchronized": self.synchronized,
            "fresh": self.fresh,
            "last_event_ts_ms": self.last_event_ts_ms,
            "age_ms": self.age_ms,
            "last_message_received_ms": self.last_message_received_ms,
            "gap_count": self.gap_count,
            "reconnect_count": self.reconnect_count,
            "error_count": self.error_count,
        }


class BoundedEventQueue(Generic[T]):
    """Async queue with explicit drop-oldest backpressure."""

    def __init__(self, *, maxsize: int) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=maxsize)
        self.dropped_count = 0

    @property
    def size(self) -> int:
        return self._queue.qsize()

    def put_nowait(self, item: T) -> None:
        if self._queue.full():
            self._queue.get_nowait()
            self.dropped_count += 1
        self._queue.put_nowait(item)

    async def get(self) -> T:
        return await self._queue.get()


class FakeReadOnlyTransport:
    """Deterministic fake transport that exercises reconnect/resubscribe logic."""

    read_only = True
    write_capable = False
    paper_only = True
    capital_at_risk = False
    broker_exchange_write_enabled = False
    live_exchange_write_enabled = False
    polymarket_write_enabled = False
    wallet_signing_enabled = False

    def __init__(self, *, subscription: Mapping[str, object], queue_size: int) -> None:
        self.subscription = dict(subscription)
        self.queue: BoundedEventQueue[Mapping[str, object]] = BoundedEventQueue(
            maxsize=queue_size
        )
        self.connected = False
        self.generation = 0
        self.subscription_count = 0
        self.reconnect_count = 0

    async def connect(self) -> int:
        if self.generation > 0:
            self.reconnect_count += 1
        self.generation += 1
        self.connected = True
        self.subscription_count += 1
        return self.generation

    async def disconnect(self) -> None:
        self.connected = False

    def inject(self, payload: Mapping[str, object]) -> None:
        self.queue.put_nowait(dict(payload))

    async def receive(self) -> Mapping[str, object]:
        return await self.queue.get()


class BinanceDepthSynchronizer:
    """Synchronize REST depth snapshots with ordered Binance WS deltas."""

    def __init__(
        self,
        *,
        calculator: BinanceOFICalculator,
        symbol: str,
        max_age_ms: int,
        delta_buffer_size: int,
        book_level_limit: int = 5_000,
    ) -> None:
        if not symbol or symbol != symbol.upper():
            raise ValueError("symbol must be uppercase and non-empty")
        if max_age_ms < 0 or delta_buffer_size <= 0 or book_level_limit <= 0:
            raise ValueError("freshness and buffer bounds are invalid")
        if calculator.symbol != symbol:
            raise ValueError("calculator and synchronizer symbols differ")
        self.calculator = calculator
        self.symbol = symbol
        self.max_age_ms = int(max_age_ms)
        self.delta_buffer_size = int(delta_buffer_size)
        self.book_level_limit = int(book_level_limit)
        self.state = FeedConnectionState.DISCONNECTED
        self.connected = False
        self.generation = 0
        self.needs_bootstrap = True
        self.last_update_id: int | None = None
        self.last_event_ts_ms: int | None = None
        self.last_message_received_ms: int | None = None
        self.gap_count = 0
        self.reconnect_count = 0
        self.error_count = 0
        self.symbol_mismatch_count = 0
        self.out_of_order_count = 0
        self.dropped_generation_count = 0
        self.buffer_overflow_count = 0
        self.book_overflow_count = 0
        self.last_bid_price: float | None = None
        self.last_ask_price: float | None = None
        self.last_top_changed = False
        self._has_applied_delta = False
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}
        self._buffer: deque[tuple[dict[str, object], int]] = deque()
        self._buffer_compromised = False

    def begin_generation(self, generation: int) -> None:
        if generation <= self.generation:
            raise ValueError("feed generation must increase")
        if self.generation > 0:
            self.reconnect_count += 1
        self.generation = generation
        self.connected = True
        self._invalidate(clear_buffer=True)

    def disconnect(self) -> None:
        self.connected = False
        self.state = FeedConnectionState.DISCONNECTED
        self.needs_bootstrap = True
        self.calculator.reset()

    def ingest_delta(
        self,
        payload: Mapping[str, object],
        *,
        generation: int,
        received_at_ms: int | None = None,
    ) -> bool:
        if generation != self.generation:
            self.dropped_generation_count += 1
            return False
        if _is_binance_subscription_ack(payload):
            return False
        try:
            delta = _unwrap_binance(payload)
            if _binance_symbol(payload, delta) != self.symbol:
                self.symbol_mismatch_count += 1
                return False
            event_ts = _positive_int(delta.get("E"), "Binance event timestamp")
            received = event_ts if received_at_ms is None else int(received_at_ms)
            self.last_message_received_ms = received
            if event_ts > received:
                self.out_of_order_count += 1
                if not self.needs_bootstrap:
                    self._invalidate(clear_buffer=True)
                return False
            if (
                self.last_event_ts_ms is not None
                and event_ts < self.last_event_ts_ms
                and (self.needs_bootstrap or self._has_applied_delta)
            ):
                self.out_of_order_count += 1
                if not self.needs_bootstrap:
                    self._invalidate(clear_buffer=True)
                return False
            if self.needs_bootstrap:
                if len(self._buffer) >= self.delta_buffer_size:
                    self._buffer.popleft()
                    self.buffer_overflow_count += 1
                    self._buffer_compromised = True
                self._buffer.append((dict(delta), received))
                return False
            return self._apply_delta(delta, received_at_ms=received)
        except (TypeError, ValueError) as exc:
            self.error_count += 1
            if isinstance(exc, _DepthBookOverflow):
                self.book_overflow_count += 1
            if not self.needs_bootstrap:
                self._invalidate(clear_buffer=True)
            return False

    def ingest_snapshot(
        self,
        payload: Mapping[str, object],
        *,
        generation: int,
        received_at_ms: int,
    ) -> bool:
        if generation != self.generation:
            self.dropped_generation_count += 1
            return False
        if self._buffer_compromised:
            self._invalidate(clear_buffer=True)
            return False
        try:
            update_id = _positive_int(payload.get("lastUpdateId"), "lastUpdateId")
            bids = _depth_side(
                payload.get("bids"),
                "bids",
                allow_empty=False,
                max_levels=self.book_level_limit,
            )
            asks = _depth_side(
                payload.get("asks"),
                "asks",
                allow_empty=False,
                max_levels=self.book_level_limit,
            )
            bid_price, bid_qty, ask_price, ask_qty = _book_top(bids, asks)
            buffered = tuple(self._buffer)
            self.calculator.reset()
            self._bids = bids
            self._asks = asks
            self.last_bid_price = bid_price
            self.last_ask_price = ask_price
            self.last_top_changed = False
            self._has_applied_delta = False
            self.last_update_id = update_id
            # REST has no exchange event time. Its receipt time must never
            # seed OFI/spot ahead of the deltas buffered during HTTP latency.
            self.last_event_ts_ms = None
            self.last_message_received_ms = int(received_at_ms)
            self.needs_bootstrap = False
            self._buffer.clear()
            for delta, received in buffered:
                final_update = _positive_int(delta.get("u"), "final update id")
                if final_update <= update_id:
                    continue
                if not self._apply_delta(delta, received_at_ms=received):
                    return False
            self.state = FeedConnectionState.READY if self._has_applied_delta else FeedConnectionState.SYNCING
            return True
        except (TypeError, ValueError) as exc:
            self.error_count += 1
            if isinstance(exc, _DepthBookOverflow):
                self.book_overflow_count += 1
            self._invalidate(clear_buffer=True)
            return False

    def health(self, *, now_ms: int) -> FeedHealth:
        age = (
            None
            if self.last_event_ts_ms is None
            else int(now_ms) - self.last_event_ts_ms
        )
        fresh = bool(
            self.connected
            and not self.needs_bootstrap
            and self._has_applied_delta
            and age is not None
            and 0 <= age <= self.max_age_ms
        )
        state = self.state
        if self.connected and self._has_applied_delta and not fresh and not self.needs_bootstrap:
            state = FeedConnectionState.STALE
        return FeedHealth(
            state=state,
            connected=self.connected,
            synchronized=not self.needs_bootstrap,
            fresh=fresh,
            last_event_ts_ms=self.last_event_ts_ms,
            age_ms=age,
            last_message_received_ms=self.last_message_received_ms,
            gap_count=self.gap_count,
            reconnect_count=self.reconnect_count,
            error_count=self.error_count,
        )

    @property
    def mid_price(self) -> float | None:
        if self.last_bid_price is None or self.last_ask_price is None:
            return None
        return (self.last_bid_price + self.last_ask_price) / 2.0

    @property
    def bid_level_count(self) -> int:
        return len(self._bids)

    @property
    def ask_level_count(self) -> int:
        return len(self._asks)

    def _apply_delta(self, delta: Mapping[str, object], *, received_at_ms: int) -> bool:
        if self.last_update_id is None:
            self._invalidate(clear_buffer=True)
            return False
        first_update = _positive_int(delta.get("U"), "first update id")
        final_update = _positive_int(delta.get("u"), "final update id")
        if final_update <= self.last_update_id:
            return False
        expected = self.last_update_id + 1
        if not first_update <= expected <= final_update:
            self.gap_count += 1
            self._invalidate(clear_buffer=True)
            return False
        event_ts = _positive_int(delta.get("E"), "Binance event timestamp")
        if (
            self.last_event_ts_ms is not None
            and event_ts < self.last_event_ts_ms
        ):
            self.out_of_order_count += 1
            self._invalidate(clear_buffer=True)
            return False
        bids = dict(self._bids)
        asks = dict(self._asks)
        _apply_depth_updates(
            bids,
            delta.get("b"),
            "bids",
            max_levels=self.book_level_limit,
        )
        _apply_depth_updates(
            asks,
            delta.get("a"),
            "asks",
            max_levels=self.book_level_limit,
        )
        bid_price, bid_qty, ask_price, ask_qty = _book_top(bids, asks)
        previous_top = self._current_top()
        current_top = (bid_price, bid_qty, ask_price, ask_qty)
        top_changed = current_top != previous_top
        first_delta = not self._has_applied_delta
        if first_delta and previous_top is not None:
            self.calculator.reset()
            self.calculator.update_and_get_z(
                bid_price=previous_top[0],
                bid_qty=previous_top[1],
                ask_price=previous_top[2],
                ask_qty=previous_top[3],
                ts_ms=event_ts,
            )
        if top_changed:
            self.calculator.update_and_get_z(
                bid_price=bid_price,
                bid_qty=bid_qty,
                ask_price=ask_price,
                ask_qty=ask_qty,
                ts_ms=event_ts,
            )
        self._bids = bids
        self._asks = asks
        self.last_bid_price = bid_price
        self.last_ask_price = ask_price
        self.last_top_changed = top_changed or first_delta
        self._has_applied_delta = True
        self.last_update_id = final_update
        self.last_event_ts_ms = event_ts
        self.last_message_received_ms = received_at_ms
        self.state = FeedConnectionState.READY
        return True

    def _current_top(self) -> tuple[float, float, float, float] | None:
        if self.last_bid_price is None or self.last_ask_price is None:
            return None
        bid_qty = self._bids.get(self.last_bid_price)
        ask_qty = self._asks.get(self.last_ask_price)
        if bid_qty is None or ask_qty is None:
            return None
        return self.last_bid_price, bid_qty, self.last_ask_price, ask_qty

    def _invalidate(self, *, clear_buffer: bool) -> None:
        self.state = FeedConnectionState.SYNCING
        self.needs_bootstrap = True
        self.last_update_id = None
        self.last_event_ts_ms = None
        self.last_bid_price = None
        self.last_ask_price = None
        self.last_top_changed = False
        self._has_applied_delta = False
        self._bids.clear()
        self._asks.clear()
        self.calculator.reset()
        if clear_buffer:
            self._buffer.clear()
            self._buffer_compromised = False


class PolymarketBookSynchronizer:
    """Generation-fenced YES/NO top-of-book synchronizer."""

    def __init__(
        self,
        *,
        window_id: str,
        yes_token_id: str,
        no_token_id: str,
        max_age_ms: int,
        condition_id: str | None = None,
    ) -> None:
        if not window_id or not yes_token_id or not no_token_id:
            raise ValueError("window and token identities must be non-empty")
        if yes_token_id == no_token_id:
            raise ValueError("YES/NO token identities must differ")
        if max_age_ms < 0:
            raise ValueError("max_age_ms must be non-negative")
        self.window_id = window_id
        self.yes_token_id = yes_token_id
        self.no_token_id = no_token_id
        self.max_age_ms = int(max_age_ms)
        self.condition_id = condition_id
        self.state = FeedConnectionState.DISCONNECTED
        self.connected = False
        self.generation = 0
        self.gap_count = 0
        self.reconnect_count = 0
        self.error_count = 0
        self.parse_error_count = 0
        self.unknown_message_count = 0
        self.dropped_generation_count = 0
        self.token_mismatch_count = 0
        self.out_of_order_count = 0
        self.last_message_received_ms: int | None = None
        self._sequence_by_token: dict[str, int] = {}
        self._timestamp_by_token: dict[str, int] = {}
        self._full_books: set[str] = set()
        self._depth: dict[str, MarketDepth] = {}
        self._pending_tops: dict[str, tuple[int, Mapping[str, object]]] = {}
        self.needs_bootstrap = False
        self._handler = self._new_handler()

    def subscription_message(self) -> dict[str, object]:
        return {
            "assets_ids": [self.yes_token_id, self.no_token_id],
            "type": "market",
            "custom_feature_enabled": True,
        }

    def begin_generation(self, generation: int) -> None:
        if generation <= self.generation:
            raise ValueError("feed generation must increase")
        if self.generation > 0:
            self.reconnect_count += 1
        self.generation = generation
        self.connected = True
        self._invalidate()
        self.needs_bootstrap = False

    def disconnect(self) -> None:
        self.connected = False
        self._invalidate()
        self.state = FeedConnectionState.DISCONNECTED

    def ingest(
        self,
        payload: Mapping[str, object],
        *,
        generation: int,
        received_at_ms: int | None = None,
    ) -> MarketSnapshot | None:
        if generation != self.generation:
            self.dropped_generation_count += 1
            return None
        try:
            event_type = str(payload.get("event_type") or payload.get("type") or "")
            if event_type and event_type not in {"book", "price_change", "best_bid_ask"}:
                # custom_feature_enabled broadcasts lifecycle events, including
                # other markets without asset_id. They are not broken depth and
                # cannot refresh/invalidate this token pair or trigger recovery.
                self.unknown_message_count += 1
                return None
            changes = payload.get("price_changes")
            if (
                not (payload.get("asset_id") or payload.get("asset"))
                and isinstance(changes, Sequence)
                and not isinstance(changes, (str, bytes))
            ):
                last: MarketSnapshot | None = None
                for change in changes:
                    if not isinstance(change, Mapping):
                        self.parse_error_count += 1
                        self._invalidate()
                        return None
                    normalized = {
                        key: value
                        for key, value in payload.items()
                        if key != "price_changes"
                    }
                    normalized.update(change)
                    normalized["event_type"] = "price_change"
                    snapshot = self.ingest(
                        normalized,
                        generation=generation,
                        received_at_ms=received_at_ms,
                    )
                    if self.needs_bootstrap:
                        return None
                    if snapshot is not None:
                        last = snapshot
                return None if self._pending_tops else last
            token_id = _text(payload.get("asset_id") or payload.get("asset"), "asset_id")
            if token_id not in {self.yes_token_id, self.no_token_id}:
                self.token_mismatch_count += 1
                return None
            payload_window = payload.get("window_id")
            if payload_window is not None and payload_window != self.window_id:
                self.token_mismatch_count += 1
                return None
            payload_condition = payload.get("market") or payload.get("condition_id")
            if (
                self.condition_id is not None
                and payload_condition is not None
                and payload_condition != self.condition_id
            ):
                self.token_mismatch_count += 1
                return None
            timestamp = _positive_int(
                payload.get("timestamp") or payload.get("timestamp_ms"),
                "Polymarket timestamp",
            )
            raw_sequence = payload.get("sequence", payload.get("seq"))
            sequence = (
                None
                if raw_sequence is None
                else _positive_int(raw_sequence, "Polymarket sequence")
            )
            received = timestamp if received_at_ms is None else int(received_at_ms)
            self.last_message_received_ms = received
            if timestamp > received:
                self.out_of_order_count += 1
                return None
            if any(received - pending[0] > self.max_age_ms for pending in self._pending_tops.values()):
                self.gap_count += 1
                self._invalidate()
                return None
            previous_ts = self._timestamp_by_token.get(token_id)
            if previous_ts is not None and timestamp < previous_ts:
                self.out_of_order_count += 1
                return None
            event_type = str(payload.get("event_type") or payload.get("type") or "")
            if event_type == "best_bid_ask":
                # Advisory top prices carry no size. Never replace depth or
                # refresh executable timestamps with these notifications.
                book = self._depth.get(token_id)
                if book is None:
                    return None
                pending = self._pending_tops.get(token_id)
                if pending is not None and timestamp < pending[0]:
                    self.out_of_order_count += 1
                    return None
                if not book.matches(payload):
                    self._pending_tops[token_id] = (timestamp, {
                        "best_bid": payload.get("best_bid"), "best_ask": payload.get("best_ask"),
                    })
                    self.state = FeedConnectionState.SYNCING
                return None
            if event_type == "book":
                depth = MarketDepth.from_payload(payload)
            elif event_type == "price_change":
                previous_sequence = self._sequence_by_token.get(token_id)
                if token_id not in self._full_books:
                    self._invalidate()
                    return None
                if (
                    sequence is not None
                    and previous_sequence is not None
                    and sequence != previous_sequence + 1
                ):
                    self.gap_count += 1
                    self._invalidate()
                    return None
                if sequence is None and not _has_complete_top(payload):
                    self.gap_count += 1
                    self._invalidate()
                    return None
                depth = self._depth[token_id].updated(payload)
                if "best_bid" in payload and "best_ask" in payload and not depth.matches(payload):
                    # Trade-driven full books and price-change notifications can
                    # interleave. Wait for authoritative depth, not a new socket
                    # on every such boundary; pending tokens cannot execute.
                    self._pending_tops[token_id] = (timestamp, {
                        "best_bid": payload.get("best_bid"), "best_ask": payload.get("best_ask"),
                    })
            else:
                self.unknown_message_count += 1
                return None
            # The parser consumes a merged as-of quote, while token freshness
            # below retains each exchange timestamp. Different token books may
            # arrive interleaved; do not compare YES event time to NO event time.
            decision_ts = max(timestamp, max(self._timestamp_by_token.values(), default=timestamp))
            normalized = {
                "event_type": "book", "asset_id": token_id,
                "window_id": self.window_id, "timestamp": decision_ts,
                **depth.top_payload(),
            }
            previous_errors = self._handler.parse_errors + self._handler.dropped_stale
            snapshot = self._handler.parse_orderbook_delta(normalized)
            if self._handler.parse_errors + self._handler.dropped_stale > previous_errors:
                self.parse_error_count += 1
                self._invalidate()
                return None
            if event_type == "book":
                self._full_books.add(token_id)
            self._depth[token_id] = depth
            pending = self._pending_tops.get(token_id)
            if pending is not None and (
                timestamp > pending[0] or timestamp == pending[0] and depth.matches(pending[1])
            ):
                self._pending_tops.pop(token_id, None)
            if sequence is not None:
                self._sequence_by_token[token_id] = sequence
            self._timestamp_by_token[token_id] = timestamp
            if snapshot is None:
                return None
            if min(
                snapshot.yes_bid_size,
                snapshot.yes_ask_size,
                snapshot.no_bid_size,
                snapshot.no_ask_size,
            ) <= 0.0:
                self.parse_error_count += 1
                self._invalidate()
                return None
            if self._full_books != {self.yes_token_id, self.no_token_id}:
                return None
            self.needs_bootstrap = False
            if self._pending_tops:
                self.state = FeedConnectionState.SYNCING
                return None
            if not self._tokens_fresh(now_ms=decision_ts):
                self.state = FeedConnectionState.STALE
                return None
            self.state = FeedConnectionState.READY
            return snapshot
        except (TypeError, ValueError):
            self.error_count += 1
            self._invalidate()
            return None

    def health(self, *, now_ms: int) -> FeedHealth:
        latest = max(self._timestamp_by_token.values(), default=None)
        age = None if latest is None else int(now_ms) - latest
        synchronized = self._full_books == {self.yes_token_id, self.no_token_id} and not self._pending_tops
        fresh = bool(self.connected and synchronized and self._tokens_fresh(now_ms=now_ms))
        state = self.state
        if self.connected and synchronized and not fresh:
            state = FeedConnectionState.STALE
        return FeedHealth(
            state=state,
            connected=self.connected,
            synchronized=synchronized,
            fresh=fresh,
            last_event_ts_ms=latest,
            age_ms=age,
            last_message_received_ms=self.last_message_received_ms,
            gap_count=self.gap_count,
            reconnect_count=self.reconnect_count,
            error_count=self.error_count + self.parse_error_count,
        )

    def token_health(self, *, now_ms: int) -> dict[str, dict[str, object]]:
        """Expose distinct YES/NO causal timestamps for the dashboard gate."""

        output: dict[str, dict[str, object]] = {}
        for label, token_id in (
            ("yes", self.yes_token_id),
            ("no", self.no_token_id),
        ):
            timestamp = self._timestamp_by_token.get(token_id)
            age = None if timestamp is None else int(now_ms) - timestamp
            output[label] = {
                "token_id": token_id,
                "timestamp_ms": timestamp,
                "age_ms": age,
                "fresh": bool(token_id not in self._pending_tops and age is not None and 0 <= age <= self.max_age_ms),
            }
        return output

    def quote_observations(self, *, now_ms: int) -> dict[str, dict[str, object]]:
        """Bounded per-token quotes; never expose unreconciled candidate depth."""
        output = self.token_health(now_ms=now_ms)
        for observation in output.values():
            token_id = str(observation["token_id"])
            depth = self._depth.get(token_id)
            confirmed = bool(
                depth is not None
                and token_id in self._full_books
                and token_id not in self._pending_tops
            )
            bid, ask = depth.top() if confirmed and depth is not None else (None, None)
            observation.update({
                "source": "polymarket_clob",
                "bid": bid,
                "ask": ask,
                "bid_size": None if bid is None or depth is None else depth.bids[bid],
                "ask_size": None if ask is None or depth is None else depth.asks[ask],
                "confirmed": confirmed,
                "connected": self.connected,
                "fresh": bool(self.connected and confirmed and observation["fresh"]),
                "max_age_ms": self.max_age_ms,
            })
        return output

    def _tokens_fresh(self, *, now_ms: int) -> bool:
        expected = (self.yes_token_id, self.no_token_id)
        return all(
            token in self._timestamp_by_token
            and token not in self._pending_tops
            and 0 <= now_ms - self._timestamp_by_token[token] <= self.max_age_ms
            for token in expected
        )

    def _invalidate(self) -> None:
        self.state = FeedConnectionState.SYNCING
        self.needs_bootstrap = True
        self._sequence_by_token.clear()
        self._timestamp_by_token.clear()
        self._full_books.clear()
        self._depth.clear()
        self._pending_tops.clear()
        self._handler = self._new_handler()

    def _new_handler(self) -> PolymarketFeedHandler:
        return PolymarketFeedHandler(
            window_id=self.window_id,
            yes_token_id=self.yes_token_id,
            no_token_id=self.no_token_id,
            max_quote_age_ms=self.max_age_ms,
            mock=True,
        )


def _unwrap_binance(payload: Mapping[str, object]) -> Mapping[str, object]:
    data = payload.get("data")
    return data if isinstance(data, Mapping) else payload


def _is_binance_subscription_ack(payload: Mapping[str, object]) -> bool:
    return set(payload) <= {"id", "result"} and "id" in payload and "result" in payload


def _binance_symbol(
    outer: Mapping[str, object],
    payload: Mapping[str, object],
) -> str:
    symbol = payload.get("s") or payload.get("symbol")
    if isinstance(symbol, str) and symbol:
        return symbol.upper()
    stream = outer.get("stream")
    if isinstance(stream, str) and "@" in stream:
        return stream.split("@", 1)[0].upper()
    raise ValueError("Binance delta is missing symbol identity")


def _depth_side(
    value: object,
    name: str,
    *,
    allow_empty: bool,
    max_levels: int,
) -> dict[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"depth {name} must be an array")
    if not value and not allow_empty:
        raise ValueError(f"depth {name} must not be empty")
    levels: dict[float, float] = {}
    for raw_level in value:
        price, quantity = _level(raw_level)
        if quantity > 0:
            levels[price] = quantity
            if len(levels) > max_levels:
                raise _DepthBookOverflow(f"depth {name} exceeds configured level limit")
    if not levels and not allow_empty:
        raise ValueError(f"depth {name} has no positive-quantity levels")
    return levels


def _apply_depth_updates(
    book: dict[float, float],
    value: object,
    name: str,
    *,
    max_levels: int,
) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"depth {name} must be an array")
    for raw_level in value:
        price, quantity = _level(raw_level)
        if quantity == 0:
            book.pop(price, None)
        else:
            book[price] = quantity
            if len(book) > max_levels:
                raise _DepthBookOverflow(f"depth {name} exceeds configured level limit")


def _book_top(
    bids: Mapping[float, float],
    asks: Mapping[float, float],
) -> tuple[float, float, float, float]:
    if not bids or not asks:
        raise ValueError("local depth book requires at least one bid and ask")
    bid_price = max(bids)
    ask_price = min(asks)
    bid_qty = bids[bid_price]
    ask_qty = asks[ask_price]
    if bid_price >= ask_price:
        raise ValueError("depth top is crossed")
    return bid_price, bid_qty, ask_price, ask_qty


def _has_complete_top(payload: Mapping[str, object]) -> bool:
    return bool(
        (payload.get("best_bid") is not None and payload.get("best_ask") is not None)
        or (payload.get("bids") is not None and payload.get("asks") is not None)
    )


def _level(value: object) -> tuple[float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) < 2
    ):
        raise ValueError("depth level is malformed")
    price = float(value[0])
    quantity = float(value[1])
    if not math.isfinite(price) or not math.isfinite(quantity) or price <= 0 or quantity < 0:
        raise ValueError("depth level is invalid")
    return price, quantity


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value.strip()
