"""Polymarket CLOB market-channel feed for the DEV-01/02/03 stack.

Parses book, best-bid/ask, and last-trade payloads into a YES/NO
``MarketSnapshot`` that downstream pricing and OMS callbacks can consume.
Live connections use exponential backoff; tests inject payloads through
the mock feeder without opening a socket.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass

import orjson

logger = logging.getLogger(__name__)

DEFAULT_CLOB_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
DEFAULT_RECONNECT_MIN_SECONDS = 1.0
DEFAULT_RECONNECT_MAX_SECONDS = 30.0
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 10.0
DEFAULT_MAX_QUOTE_AGE_MS = 5_000

SnapshotCallback = Callable[["MarketSnapshot"], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """Top-of-book YES/NO quotes at one exchange timestamp."""

    timestamp_ms: int
    window_id: str
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    last_traded_price: float
    yes_bid_size: float = 0.0
    yes_ask_size: float = 0.0
    no_bid_size: float = 0.0
    no_ask_size: float = 0.0


class PolymarketFeedHandler:
    """Async CLOB market feed with mock injection and reconnect backoff."""

    __slots__ = (
        "window_id",
        "yes_token_id",
        "no_token_id",
        "ws_url",
        "mock",
        "reconnect_min_seconds",
        "reconnect_max_seconds",
        "heartbeat_interval_seconds",
        "max_quote_age_ms",
        "connected",
        "parse_errors",
        "dropped_stale",
        "_callbacks",
        "_yes_bid",
        "_yes_ask",
        "_yes_bid_size",
        "_yes_ask_size",
        "_no_bid",
        "_no_ask",
        "_no_bid_size",
        "_no_ask_size",
        "_yes_timestamp_ms",
        "_no_timestamp_ms",
        "_last_traded_price",
        "_last_timestamp_ms",
        "_last_seq",
        "_closed",
        "_run_task",
    )

    def __init__(
        self,
        *,
        window_id: str,
        yes_token_id: str = "yes",
        no_token_id: str = "no",
        ws_url: str = DEFAULT_CLOB_WS_URL,
        mock: bool = True,
        reconnect_min_seconds: float = DEFAULT_RECONNECT_MIN_SECONDS,
        reconnect_max_seconds: float = DEFAULT_RECONNECT_MAX_SECONDS,
        heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        max_quote_age_ms: int = DEFAULT_MAX_QUOTE_AGE_MS,
    ) -> None:
        if not str(window_id).strip():
            raise ValueError("window_id must be non-empty")
        if not str(yes_token_id).strip() or not str(no_token_id).strip():
            raise ValueError("token ids must be non-empty")
        if reconnect_min_seconds <= 0.0 or reconnect_max_seconds < reconnect_min_seconds:
            raise ValueError("reconnect backoff bounds are invalid")
        if heartbeat_interval_seconds <= 0.0:
            raise ValueError("heartbeat_interval_seconds must be positive")
        if int(max_quote_age_ms) < 0:
            raise ValueError("max_quote_age_ms must be non-negative")
        self.window_id = str(window_id)
        self.yes_token_id = str(yes_token_id)
        self.no_token_id = str(no_token_id)
        self.ws_url = str(ws_url)
        self.mock = bool(mock)
        self.reconnect_min_seconds = float(reconnect_min_seconds)
        self.reconnect_max_seconds = float(reconnect_max_seconds)
        self.heartbeat_interval_seconds = float(heartbeat_interval_seconds)
        self.max_quote_age_ms = int(max_quote_age_ms)
        self.connected = False
        self.parse_errors = 0
        self.dropped_stale = 0
        self._callbacks: list[SnapshotCallback] = []
        self._yes_bid: float | None = None
        self._yes_ask: float | None = None
        self._yes_bid_size = 0.0
        self._yes_ask_size = 0.0
        self._no_bid: float | None = None
        self._no_ask: float | None = None
        self._no_bid_size = 0.0
        self._no_ask_size = 0.0
        self._yes_timestamp_ms: int | None = None
        self._no_timestamp_ms: int | None = None
        self._last_traded_price = 0.0
        self._last_timestamp_ms: int | None = None
        self._last_seq: int | None = None
        self._closed = False
        self._run_task: asyncio.Task[None] | None = None

    def on_snapshot(self, callback: SnapshotCallback) -> None:
        """Register an async listener invoked on each accepted snapshot."""

        self._callbacks.append(callback)

    def subscription_message(self) -> dict[str, object]:
        return {
            "assets_ids": [self.yes_token_id, self.no_token_id],
            "type": "market",
            "custom_feature_enabled": True,
        }

    async def connect(self) -> None:
        """Open the mock feeder, or start the live reconnect loop."""

        self._closed = False
        if self.mock:
            self.connected = True
            return
        if self._run_task is None or self._run_task.done():
            self._run_task = asyncio.create_task(self._run_forever(), name="clob-ws")
        self.connected = True

    async def close(self) -> None:
        self._closed = True
        self.connected = False
        task = self._run_task
        self._run_task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    def parse_orderbook_delta(self, payload: Mapping[str, object]) -> MarketSnapshot | None:
        """Parse one CLOB market-channel dict into a YES/NO snapshot.

        Stale timestamps and out-of-order sequence numbers are dropped.
        """

        if not isinstance(payload, Mapping):
            self.parse_errors += 1
            return None
        payload_window = payload.get("window_id")
        if (
            isinstance(payload_window, str)
            and payload_window.strip()
            and payload_window != self.window_id
        ):
            self.dropped_stale += 1
            return None
        timestamp_ms = _timestamp_ms(payload)
        sequence = _sequence(payload)
        if self._is_stale(timestamp_ms, sequence):
            self.dropped_stale += 1
            return None
        updated_sides = self._apply_delta(payload)
        if not updated_sides:
            return None
        if timestamp_ms is not None:
            self._last_timestamp_ms = timestamp_ms
        else:
            self._last_timestamp_ms = int(time.time() * 1000)
        if sequence is not None:
            self._last_seq = sequence
        if "YES" in updated_sides:
            self._yes_timestamp_ms = self._last_timestamp_ms
        if "NO" in updated_sides:
            self._no_timestamp_ms = self._last_timestamp_ms
        return self._snapshot(self._last_timestamp_ms)

    async def ingest_raw(self, raw: str | bytes) -> MarketSnapshot | None:
        """Decode a WS frame; malformed JSON is logged and skipped."""

        stripped = raw.strip() if isinstance(raw, bytes) else raw.strip().encode()
        if stripped.upper() in {b"PING", b"PONG"}:
            return None
        try:
            decoded = orjson.loads(raw)
        except (orjson.JSONDecodeError, TypeError, ValueError) as exc:
            self.parse_errors += 1
            logger.warning("clob.payload.invalid_json error=%s", exc)
            return None
        last: MarketSnapshot | None = None
        for payload in _as_payloads(decoded):
            snapshot = await self.ingest_payload(payload)
            if snapshot is not None:
                last = snapshot
        return last

    async def ingest_payload(self, payload: Mapping[str, object]) -> MarketSnapshot | None:
        snapshot = self.parse_orderbook_delta(payload)
        if snapshot is None:
            return None
        await self._dispatch(snapshot)
        return snapshot

    def _is_stale(self, timestamp_ms: int | None, sequence: int | None) -> bool:
        if sequence is not None and self._last_seq is not None and sequence <= self._last_seq:
            return True
        return bool(
            timestamp_ms is not None
            and self._last_timestamp_ms is not None
            and timestamp_ms < self._last_timestamp_ms
        )

    def _apply_delta(self, payload: Mapping[str, object]) -> set[str]:
        updated_sides: set[str] = set()
        event_type = str(payload.get("event_type") or payload.get("type") or "book")

        yes_book = payload.get("yes")
        no_book = payload.get("no")
        if isinstance(yes_book, Mapping) and self._update_side("YES", yes_book):
            updated_sides.add("YES")
        if isinstance(no_book, Mapping) and self._update_side("NO", no_book):
            updated_sides.add("NO")

        updated_sides.update(self._update_named_quotes(payload))

        asset_id = str(payload.get("asset_id") or payload.get("asset") or "")
        side = self._side_for_asset(asset_id)
        if event_type in {"book", "price_change", "best_bid_ask"} or "bids" in payload or "asks" in payload:
            if side is not None:
                if self._update_side(side, payload):
                    updated_sides.add(side)
            elif asset_id == "":
                # Combined snapshot without nested yes/no books.
                pass

        changes = payload.get("price_changes")
        if isinstance(changes, Sequence) and not isinstance(changes, (str, bytes)):
            for change in changes:
                if not isinstance(change, Mapping):
                    continue
                change_side = self._side_for_asset(str(change.get("asset_id") or ""))
                if change_side is None:
                    continue
                if self._update_side(change_side, change):
                    updated_sides.add(change_side)

        last_price = payload.get("last_trade_price")
        if last_price is None and event_type == "last_trade_price":
            last_price = payload.get("price")
        coerced = _optional_float(last_price)
        if coerced is not None:
            self._last_traded_price = coerced
            updated_sides.add("TRADE")
        return updated_sides

    def _update_named_quotes(self, payload: Mapping[str, object]) -> set[str]:
        updated_sides: set[str] = set()
        mapping = (
            ("yes_bid", "yes_bid_size", "YES", True),
            ("yes_ask", "yes_ask_size", "YES", False),
            ("no_bid", "no_bid_size", "NO", True),
            ("no_ask", "no_ask_size", "NO", False),
        )
        for key, size_key, side, is_bid in mapping:
            value = _optional_market_price(payload.get(key))
            if value is None:
                continue
            size = _optional_size(payload.get(size_key))
            self._set_quote(side, is_bid=is_bid, price=value, size=size)
            updated_sides.add(side)
        return updated_sides

    def _update_side(self, side: str, payload: Mapping[str, object]) -> bool:
        updated = False
        bids = payload.get("bids")
        asks = payload.get("asks")
        best_bid = _optional_market_price(_first_present(payload, "best_bid", "bid"))
        best_ask = _optional_market_price(_first_present(payload, "best_ask", "ask"))
        best_bid_size: float | None = None
        best_ask_size: float | None = None
        if isinstance(bids, Sequence) and not isinstance(bids, (str, bytes)):
            level = _best_level(bids, is_bid=True)
            if level is None:
                self._set_quote(side, is_bid=True, price=None, size=0.0)
                updated = True
            elif best_bid is None:
                best_bid, best_bid_size = level
        if isinstance(asks, Sequence) and not isinstance(asks, (str, bytes)):
            level = _best_level(asks, is_bid=False)
            if level is None:
                self._set_quote(side, is_bid=False, price=None, size=0.0)
                updated = True
            elif best_ask is None:
                best_ask, best_ask_size = level
        change_price = _optional_market_price(payload.get("price"))
        change_size = _optional_size(payload.get("size"))
        change_side = str(payload.get("side") or "").upper()
        if best_bid is not None and change_side == "BUY" and change_price == best_bid:
            best_bid_size = change_size
        if best_ask is not None and change_side == "SELL" and change_price == best_ask:
            best_ask_size = change_size
        if best_bid is not None:
            self._set_quote(side, is_bid=True, price=best_bid, size=best_bid_size)
            updated = True
        if best_ask is not None:
            self._set_quote(side, is_bid=False, price=best_ask, size=best_ask_size)
            updated = True
        return updated

    def _set_quote(
        self,
        side: str,
        *,
        is_bid: bool,
        price: float | None,
        size: float | None,
    ) -> None:
        if side == "YES":
            if is_bid:
                unchanged = price is not None and price == self._yes_bid
                self._yes_bid = price
                self._yes_bid_size = self._yes_bid_size if unchanged and size is None else (size or 0.0)
            else:
                unchanged = price is not None and price == self._yes_ask
                self._yes_ask = price
                self._yes_ask_size = self._yes_ask_size if unchanged and size is None else (size or 0.0)
            return
        if is_bid:
            unchanged = price is not None and price == self._no_bid
            self._no_bid = price
            self._no_bid_size = self._no_bid_size if unchanged and size is None else (size or 0.0)
        else:
            unchanged = price is not None and price == self._no_ask
            self._no_ask = price
            self._no_ask_size = self._no_ask_size if unchanged and size is None else (size or 0.0)

    def _side_for_asset(self, asset_id: str) -> str | None:
        if not asset_id:
            return None
        if asset_id == self.yes_token_id or asset_id.lower() == "yes":
            return "YES"
        if asset_id == self.no_token_id or asset_id.lower() == "no":
            return "NO"
        return None

    def _snapshot(self, timestamp_ms: int) -> MarketSnapshot | None:
        quotes = (self._yes_bid, self._yes_ask, self._no_bid, self._no_ask)
        if any(value is None for value in quotes):
            return None
        yes_timestamp_ms = self._yes_timestamp_ms
        no_timestamp_ms = self._no_timestamp_ms
        if yes_timestamp_ms is None or no_timestamp_ms is None:
            return None
        if any(
            timestamp_ms - side_timestamp > self.max_quote_age_ms
            for side_timestamp in (yes_timestamp_ms, no_timestamp_ms)
        ):
            return None
        yes_bid = self._yes_bid
        yes_ask = self._yes_ask
        no_bid = self._no_bid
        no_ask = self._no_ask
        assert yes_bid is not None
        assert yes_ask is not None
        assert no_bid is not None
        assert no_ask is not None
        if yes_bid > yes_ask or no_bid > no_ask:
            self.parse_errors += 1
            return None
        return MarketSnapshot(
            timestamp_ms=int(timestamp_ms),
            window_id=self.window_id,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            last_traded_price=self._last_traded_price,
            yes_bid_size=self._yes_bid_size,
            yes_ask_size=self._yes_ask_size,
            no_bid_size=self._no_bid_size,
            no_ask_size=self._no_ask_size,
        )

    async def _dispatch(self, snapshot: MarketSnapshot) -> None:
        for callback in self._callbacks:
            try:
                await callback(snapshot)
            except Exception:
                logger.exception("clob.snapshot.callback_failed")

    async def _run_forever(self) -> None:
        backoff = self.reconnect_min_seconds
        while not self._closed:
            try:
                await self._connect_and_listen()
                backoff = self.reconnect_min_seconds
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("clob.ws.disconnected error=%s backoff_s=%s", exc, backoff)
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    raise
                backoff = min(backoff * 2.0, self.reconnect_max_seconds)

    async def _connect_and_listen(self) -> None:
        import websockets

        try:
            async with websockets.connect(
                self.ws_url,
                ping_interval=None,
                ping_timeout=None,
                close_timeout=5,
                max_size=2**24,
            ) as ws:
                self._reset_quotes()
                self.connected = True
                await ws.send(orjson.dumps(self.subscription_message()))
                async with asyncio.TaskGroup() as tasks:
                    tasks.create_task(self._receive_loop(ws), name="clob-ws-recv")
                    tasks.create_task(self._heartbeat_loop(ws), name="clob-ws-heartbeat")
        finally:
            self.connected = False

    async def _receive_loop(self, ws: object) -> None:
        while not self._closed:
            raw = await ws.recv()  # type: ignore[attr-defined]
            await self.ingest_raw(raw)

    async def _heartbeat_loop(self, ws: object) -> None:
        while not self._closed:
            await ws.send("PING")  # type: ignore[attr-defined]
            await asyncio.sleep(self.heartbeat_interval_seconds)

    def _reset_quotes(self) -> None:
        self._yes_bid = None
        self._yes_ask = None
        self._yes_bid_size = 0.0
        self._yes_ask_size = 0.0
        self._no_bid = None
        self._no_ask = None
        self._no_bid_size = 0.0
        self._no_ask_size = 0.0
        self._yes_timestamp_ms = None
        self._no_timestamp_ms = None
        self._last_timestamp_ms = None
        self._last_seq = None


def _as_payloads(decoded: object) -> list[Mapping[str, object]]:
    if isinstance(decoded, Mapping):
        return [decoded]
    if isinstance(decoded, Sequence) and not isinstance(decoded, (str, bytes)):
        return [item for item in decoded if isinstance(item, Mapping)]
    return []


def _timestamp_ms(payload: Mapping[str, object]) -> int | None:
    for key in ("timestamp_ms", "timestamp", "T", "t"):
        value = _optional_int(payload.get(key))
        if value is not None:
            return value
    return None


def _sequence(payload: Mapping[str, object]) -> int | None:
    for key in ("sequence", "seq", "s"):
        value = _optional_int(payload.get(key))
        if value is not None:
            return value
    return None


def _best_level(levels: Sequence[object], *, is_bid: bool) -> tuple[float, float] | None:
    parsed: list[tuple[float, float]] = []
    for level in levels:
        item = _level(level)
        if item is not None:
            parsed.append(item)
    if not parsed:
        return None
    return max(parsed, key=_level_price) if is_bid else min(parsed, key=_level_price)


def _level_price(level: tuple[float, float]) -> float:
    return level[0]


def _level(level: object) -> tuple[float, float] | None:
    if isinstance(level, Mapping):
        price = _optional_market_price(level.get("price"))
        size = _optional_size(level.get("size"))
        return (price, size) if price is not None and size is not None else None
    if isinstance(level, Sequence) and not isinstance(level, (str, bytes)) and level:
        price = _optional_market_price(level[0])
        size = _optional_size(level[1]) if len(level) > 1 else None
        return (price, size) if price is not None and size is not None else None
    return None


def _first_present(payload: Mapping[str, object], *keys: str) -> object | None:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _optional_market_price(value: object) -> float | None:
    out = _optional_float(value)
    if out is None or not 0.0 < out <= 1.0:
        return None
    return out


def _optional_size(value: object) -> float | None:
    out = _optional_float(value)
    if out is None or out < 0.0:
        return None
    return out


def _optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return int(float(value))
        except ValueError:
            return None
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    return None
