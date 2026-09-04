"""Public read-only HTTP/WebSocket transports for the paper operator."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, replace
from typing import Protocol
from urllib.parse import urlsplit

import aiohttp
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from bigan.data.polymarket_clob import MarketSnapshot
from bigan.v8.polymarket.recorder.chainlink_rtds import (
    ChainlinkRTDSMessageError,
    parse_chainlink_rtds_message,
)

from .chainlink_twap import TWAP_TOPICS, parse_twap_sample
from .diagnostics import DiagnosticCode, FeedResyncRequired, HeartbeatTimeout
from .discovery import (
    DiscoveryFilters,
    DiscoverySelection,
    parse_gamma_markets,
    select_market_windows,
)
from .feeds import BinanceDepthSynchronizer, BoundedEventQueue, PolymarketBookSynchronizer
from .opening_reference import OPENING_REFERENCE_ENDPOINT, fetch_opening_reference
from .pricing_inputs import ReferencePriceSample

logger = logging.getLogger(__name__)

Clock = Callable[[], int]
Sleep = Callable[[float], Awaitable[None]]
PayloadHandler = Callable[[Mapping[str, object], int, int], object]
GenerationHandler = Callable[[int], object]
DisconnectHandler = Callable[[], object]
DiagnosticHandler = Callable[[DiagnosticCode, int, int], object]


class PublicSocket(Protocol):
    async def send(self, message: str | bytes) -> None: ...
    async def recv(self) -> str | bytes: ...
    async def ping(self) -> object: ...


SocketContextFactory = Callable[[str], AbstractAsyncContextManager[PublicSocket]]


class PublicJSONClient(Protocol):
    async def get_json(
        self,
        endpoint: str,
        *,
        params: Mapping[str, str | int] | None = None,
    ) -> object: ...


class AiohttpPublicJSONClient:
    """GET-only public JSON client; authentication material is not accepted."""

    read_only = True
    write_capable = False
    paper_only = True

    def __init__(self, *, timeout_seconds: float = 10.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("HTTP timeout must be positive")
        self.timeout_seconds = float(timeout_seconds)

    async def get_json(
        self,
        endpoint: str,
        *,
        params: Mapping[str, str | int] | None = None,
    ) -> object:
        _require_public_read_url(endpoint)
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        async with (
            aiohttp.ClientSession(timeout=timeout, max_field_size=32768) as session,
            session.get(endpoint, params=dict(params or {}), allow_redirects=False) as response,
        ):
            response.raise_for_status()
            if response.status != 200:
                raise ValueError("unexpected public HTTP status")
            body = bytearray()
            async for chunk in response.content.iter_chunked(16384):
                body.extend(chunk)
                if len(body) > 2_000_000:
                    raise ValueError("public JSON response exceeds memory bound")
            return json.loads(body)


@dataclass(frozen=True, slots=True)
class TransportHealth:
    connected: bool
    generation: int
    reconnect_count: int
    message_count: int
    parse_error_count: int
    connection_error_count: int
    queue_dropped_count: int
    last_message_received_ms: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "connected": self.connected,
            "generation": self.generation,
            "reconnect_count": self.reconnect_count,
            "message_count": self.message_count,
            "parse_error_count": self.parse_error_count,
            "connection_error_count": self.connection_error_count,
            "queue_dropped_count": self.queue_dropped_count,
            "last_message_received_ms": self.last_message_received_ms,
        }


class PublicWebSocketTransport:
    """Bounded reconnecting pump that can only send one public subscription."""

    read_only = True
    write_capable = False
    paper_only = True
    capital_at_risk = False
    broker_exchange_write_enabled = False
    live_exchange_write_enabled = False
    polymarket_write_enabled = False
    wallet_signing_enabled = False

    def __init__(
        self,
        *,
        endpoint: str,
        subscription: Mapping[str, object],
        queue_size: int,
        reconnect_min_seconds: float,
        reconnect_max_seconds: float,
        heartbeat_interval_seconds: float,
        clock_ms: Clock,
        on_payload: PayloadHandler,
        on_generation: GenerationHandler,
        on_disconnect: DisconnectHandler,
        on_diagnostic: DiagnosticHandler | None = None,
        application_heartbeat: str | None = None,
        connect_factory: SocketContextFactory | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        _require_public_websocket_url(endpoint)
        if queue_size <= 0:
            raise ValueError("transport queue size must be positive")
        if reconnect_min_seconds <= 0 or reconnect_max_seconds < reconnect_min_seconds:
            raise ValueError("transport reconnect bounds are invalid")
        if heartbeat_interval_seconds <= 0:
            raise ValueError("transport heartbeat must be positive")
        self.endpoint = endpoint
        self.subscription = _safe_subscription(subscription)
        self.queue: BoundedEventQueue[tuple[Mapping[str, object], int, int]] = (
            BoundedEventQueue(maxsize=queue_size)
        )
        self.reconnect_min_seconds = float(reconnect_min_seconds)
        self.reconnect_max_seconds = float(reconnect_max_seconds)
        self.heartbeat_interval_seconds = float(heartbeat_interval_seconds)
        self.clock_ms = clock_ms
        self.on_payload = on_payload
        self.on_generation = on_generation
        self.on_disconnect = on_disconnect
        self.on_diagnostic = on_diagnostic
        self.application_heartbeat = application_heartbeat
        self.connect_factory = connect_factory or _websocket_context
        self.sleep = sleep
        self.connected = False
        self.generation = 0
        self.reconnect_count = 0
        self.message_count = 0
        self.parse_error_count = 0
        self.connection_error_count = 0
        self.last_message_received_ms: int | None = None

    async def run(self, stop_event: asyncio.Event) -> None:
        backoff = self.reconnect_min_seconds
        while not stop_event.is_set():
            self.generation += 1
            if self.generation > 1:
                self.reconnect_count += 1
            generation = self.generation
            try:
                async with self.connect_factory(self.endpoint) as socket:
                    self.connected = True
                    while self.queue.size:
                        await self.queue.get()
                    await socket.send(
                        json.dumps(
                            self.subscription,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                    )
                    receiver = asyncio.create_task(
                        self._receive_to_queue(socket, generation, stop_event),
                        name=f"public-feed-receiver-{generation}",
                    )
                    logger.info(
                        "paper_operator.feed.connected",
                        extra={"endpoint_host": urlsplit(self.endpoint).hostname, "generation": generation},
                    )
                    try:
                        # The receive pump starts before bootstrap so diff-depth events
                        # are boundedly buffered while the REST snapshot is in flight.
                        await _maybe_await(self.on_generation(generation))
                        while not stop_event.is_set():
                            # A failed heartbeat/reader fences queued data too;
                            # a busy queue must not hide a dead connection.
                            if receiver.done():
                                await receiver
                                if stop_event.is_set():
                                    break
                                raise ConnectionError("public websocket receive pump stopped")
                            if self.queue.size:
                                item = await self.queue.get()
                            else:
                                item = await self._next_queue_item(receiver, stop_event)
                                if item is None:
                                    break
                            payload, queued_generation, queued_received = item
                            await _maybe_await(
                                self.on_payload(payload, queued_generation, queued_received)
                            )
                            self.message_count += 1
                            backoff = self.reconnect_min_seconds
                    finally:
                        receiver.cancel()
                        await asyncio.gather(receiver, return_exceptions=True)
            except asyncio.CancelledError:
                raise
            except (
                OSError,
                ConnectionError,
                aiohttp.ClientError,
                ConnectionClosed,
                WebSocketException,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                self.connection_error_count += 1
                if isinstance(exc, (ValueError, json.JSONDecodeError)):
                    self.parse_error_count += 1
                if self.on_diagnostic is not None:
                    await _maybe_await(self.on_diagnostic(_transport_reason(exc), generation, self.clock_ms()))
                logger.warning(
                    "paper_operator.feed.disconnected",
                    extra={"error_type": type(exc).__name__, "generation": generation},
                )
            finally:
                self.connected = False
                await _maybe_await(self.on_disconnect())
            if not stop_event.is_set():
                await self.sleep(backoff)
                backoff = min(backoff * 2.0, self.reconnect_max_seconds)

    async def _receive_to_queue(
        self,
        socket: PublicSocket,
        generation: int,
        stop_event: asyncio.Event,
    ) -> None:
        # Never stop draining recv() while waiting for PONG. A bounded WS
        # library queue can fill with market messages and prevent its protocol
        # reader from reaching the PONG, causing a self-inflicted timeout.
        reader = asyncio.create_task(
            self._receive_messages(socket, generation, stop_event),
            name=f"public-feed-messages-{generation}",
        )
        heartbeat = asyncio.create_task(
            self._heartbeat(socket, stop_event),
            name=f"public-feed-heartbeat-{generation}",
        )
        stopped = asyncio.create_task(stop_event.wait())
        tasks = (reader, heartbeat, stopped)
        try:
            done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            if stopped in done and stop_event.is_set():
                return
            for task in (reader, heartbeat):
                if task in done:
                    await task
            raise ConnectionError("public websocket connection task stopped")
        finally:
            for pending_task in tasks:
                pending_task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _receive_messages(
        self,
        socket: PublicSocket,
        generation: int,
        stop_event: asyncio.Event,
    ) -> None:
        while not stop_event.is_set():
            raw = await socket.recv()
            received = self.clock_ms()
            self.last_message_received_ms = received
            for payload in _decode_payloads(raw):
                self.queue.put_nowait((payload, generation, received))

    async def _heartbeat(self, socket: PublicSocket, stop_event: asyncio.Event) -> None:
        heartbeat_due = time.monotonic() + self.heartbeat_interval_seconds
        while not stop_event.is_set():
            await asyncio.sleep(max(0, heartbeat_due - time.monotonic()))
            if stop_event.is_set():
                return
            heartbeat_due = time.monotonic() + self.heartbeat_interval_seconds
            # One outstanding PING, with a deadline covering send and PONG.
            # Busy streams still send application heartbeats on schedule.
            try:
                await asyncio.wait_for(
                    self._send_heartbeat(socket), timeout=self.heartbeat_interval_seconds,
                )
            except TimeoutError:
                raise HeartbeatTimeout("public feed heartbeat deadline exceeded") from None

    async def _send_heartbeat(self, socket: PublicSocket) -> None:
        if self.application_heartbeat is not None:
            await socket.send(self.application_heartbeat)
        pong = await socket.ping()
        if inspect.isawaitable(pong):
            await pong

    async def _next_queue_item(
        self,
        receiver: asyncio.Task[None],
        stop_event: asyncio.Event,
    ) -> tuple[Mapping[str, object], int, int] | None:
        queued = asyncio.create_task(self.queue.get())
        stopped = asyncio.create_task(stop_event.wait())
        try:
            done, _pending = await asyncio.wait(
                {queued, stopped, receiver},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stopped in done and stop_event.is_set():
                return None
            if receiver in done:
                await receiver
                raise ConnectionError("public websocket receive pump stopped")
            if queued in done:
                return queued.result()
            return None
        finally:
            for task in (queued, stopped):
                if not task.done():
                    task.cancel()
            await asyncio.gather(queued, stopped, return_exceptions=True)

    def health(self) -> TransportHealth:
        return TransportHealth(
            connected=self.connected,
            generation=self.generation,
            reconnect_count=self.reconnect_count,
            message_count=self.message_count,
            parse_error_count=self.parse_error_count,
            connection_error_count=self.connection_error_count,
            queue_dropped_count=self.queue.dropped_count,
            last_message_received_ms=self.last_message_received_ms,
        )


class GammaDiscoveryClient:
    def __init__(self, *, endpoint: str, http: PublicJSONClient,
                 opening_reference_endpoint: str = OPENING_REFERENCE_ENDPOINT) -> None:
        self.endpoint = endpoint
        self.http = http
        self.opening_reference_endpoint = opening_reference_endpoint

    async def discover(
        self,
        *,
        filters: DiscoveryFilters,
        now_ms: int,
    ) -> DiscoverySelection:
        duration_name = {
            300_000: "5m",
            900_000: "15m",
        }.get(filters.window_duration_ms)
        if filters.market_type != "binary_up_down" or duration_name is None:
            raise ValueError("public Gamma discovery supports exact 5m/15m up/down families")
        current_start = now_ms - (now_ms % filters.window_duration_ms)
        lookahead = max(1, (filters.max_preopen_ms + filters.window_duration_ms - 1) // filters.window_duration_ms)
        rows: list[object] = []
        for offset in range(0, lookahead + 1):
            start_ms = current_start + offset * filters.window_duration_ms
            slug = (
                f"{filters.underlying.lower()}-updown-{duration_name}-{start_ms // 1_000}"
            )
            payload = await self.http.get_json(self.endpoint, params={"slug": slug})
            rows.extend(_gamma_rows(payload))
        candidates = parse_gamma_markets(
            rows,
            source_endpoint=self.endpoint,
            discovered_at_ms=now_ms,
        )
        selection = select_market_windows(candidates, filters=filters, now_ms=now_ms)
        current = selection.current
        if current is not None and current.oracle_twap_lookback_seconds is not None:
            current = await fetch_opening_reference(
                current, http=self.http, endpoint=self.opening_reference_endpoint,
            )
            selection = replace(selection, current=current)
        return selection


class BinanceReadonlyFeed:
    """REST bootstrap + public incremental depth transport."""

    def __init__(
        self,
        *,
        symbol: str,
        depth_endpoint: str,
        synchronizer: BinanceDepthSynchronizer,
        http: PublicJSONClient,
        clock_ms: Clock,
    ) -> None:
        self.symbol = symbol
        self.depth_endpoint = depth_endpoint
        self.synchronizer = synchronizer
        self.http = http
        self.clock_ms = clock_ms

    async def on_generation(self, generation: int) -> None:
        self.synchronizer.begin_generation(generation)
        snapshot = await self.http.get_json(
            self.depth_endpoint,
            params={"symbol": self.symbol, "limit": 1000},
        )
        if not isinstance(snapshot, Mapping):
            raise ValueError("Binance depth snapshot must be an object")
        if not self.synchronizer.ingest_snapshot(
            snapshot,
            generation=generation,
            received_at_ms=self.clock_ms(),
        ):
            raise ValueError("Binance depth bootstrap failed")

    def on_payload(self, payload: Mapping[str, object], generation: int, received: int) -> None:
        accepted = self.synchronizer.ingest_delta(
            payload,
            generation=generation,
            received_at_ms=received,
        )
        if not accepted and self.synchronizer.needs_bootstrap:
            raise ConnectionError("Binance depth gap requires immediate re-bootstrap")

    def on_disconnect(self) -> None:
        self.synchronizer.disconnect()


class PolymarketReadonlyFeed:
    """Public dual-token CLOB transport with generation fencing."""

    def __init__(
        self,
        *,
        synchronizer: PolymarketBookSynchronizer,
        on_snapshot: Callable[[MarketSnapshot, int], object],
    ) -> None:
        self.synchronizer = synchronizer
        self.on_snapshot = on_snapshot

    def on_generation(self, generation: int) -> None:
        self.synchronizer.begin_generation(generation)

    async def on_payload(
        self,
        payload: Mapping[str, object],
        generation: int,
        received: int,
    ) -> None:
        snapshot = self.synchronizer.ingest(
            payload,
            generation=generation,
            received_at_ms=received,
        )
        if self.synchronizer.needs_bootstrap:
            raise ConnectionError("Polymarket depth requires a fresh full-book subscription")
        if snapshot is not None:
            await _maybe_await(self.on_snapshot(snapshot, generation))

    def on_disconnect(self) -> None:
        self.synchronizer.disconnect()


class ChainlinkReadonlyFeed:
    """Parse Polymarket RTDS Chainlink messages into independent oracle samples."""

    def __init__(
        self,
        *,
        expected_symbol: str,
        source: str,
        on_sample: Callable[[ReferencePriceSample, int], object],
        lookback_seconds: int | None = None,
    ) -> None:
        self.expected_symbol = expected_symbol.lower()
        self.source = source
        self.on_sample = on_sample
        self.lookback_seconds = lookback_seconds
        self.parse_error_count = 0
        self.symbol_mismatch_count = 0

    async def on_raw(self, raw: str | bytes, *, generation: int, received_at_ms: int) -> None:
        if self.lookback_seconds is not None:
            try:
                sample = parse_twap_sample(json.loads(raw), symbol=self.expected_symbol,
                                           lookback_seconds=self.lookback_seconds, received_at_ms=received_at_ms)
            except (ValueError, TypeError):
                self.parse_error_count += 1
                return
            if sample is not None:
                await _maybe_await(self.on_sample(sample, generation))
            return
        try:
            rows = parse_chainlink_rtds_message(raw, received_at_ts=received_at_ms)
        except ChainlinkRTDSMessageError:
            self.parse_error_count += 1
            return
        for row in rows:
            if str(row["symbol"]).lower() != self.expected_symbol:
                self.symbol_mismatch_count += 1
                continue
            sample = ReferencePriceSample(
                timestamp_ms=int(row["source_ts"]),
                received_at_ms=int(row["available_at_ts"]),
                price=float(row["price"]),
                source=self.source,
            )
            await _maybe_await(self.on_sample(sample, generation))


def binance_subscription(symbol: str) -> dict[str, object]:
    return {
        "method": "SUBSCRIBE",
        "params": [f"{symbol.lower()}@depth@100ms"],
        "id": 1,
    }


def chainlink_subscription(symbol: str, lookback_seconds: int | None = None) -> dict[str, object]:
    return {
        "action": "subscribe",
        "subscriptions": [
            {"topic": "crypto_prices_chainlink" if lookback_seconds is None else TWAP_TOPICS[lookback_seconds],
             "type": "update", "filters": json.dumps({"symbol": symbol.lower()}, separators=(",", ":"))}
        ],
    }


def _decode_payloads(raw: str | bytes) -> tuple[Mapping[str, object], ...]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    # RTDS may send an empty frame immediately after subscription. It is a
    # transport control message, never an observation or freshness heartbeat.
    if raw.strip().upper() in {"", "PING", "PONG"}:
        return ()
    value = json.loads(raw)
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, list) and all(isinstance(row, Mapping) for row in value):
        return tuple(value)
    raise ValueError("public websocket message must contain object payloads")


def _gamma_rows(payload: object) -> list[object]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        rows = payload.get("data", payload.get("markets"))
        if isinstance(rows, list):
            return rows
    raise ValueError("Gamma slug response must contain a market list")


def _safe_subscription(payload: Mapping[str, object]) -> dict[str, object]:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    lowered = encoded.lower()
    if any(word in lowered for word in ("private_key", "signature", "authorization", "cookie")):
        raise ValueError("subscription cannot contain authentication or signing material")
    if any(word in lowered for word in ("create_order", "cancel_order", "place_order")):
        raise ValueError("subscription cannot contain a write operation")
    return dict(payload)


def _require_public_read_url(endpoint: str) -> None:
    parsed = urlsplit(endpoint)
    path = parsed.path.lower()
    if parsed.scheme != "https" or parsed.username or parsed.password:
        raise ValueError("HTTP transport requires a public HTTPS URL")
    if any(word in path for word in ("/order", "cancel", "trade", "wallet", "signature")):
        raise ValueError("HTTP transport rejects write/trading endpoints")


def _require_public_websocket_url(endpoint: str) -> None:
    parsed = urlsplit(endpoint)
    if parsed.scheme != "wss" or parsed.username or parsed.password or parsed.query:
        raise ValueError("websocket transport requires a public WSS URL")
    if any(word in parsed.path.lower() for word in ("order", "trade", "private", "wallet")):
        raise ValueError("websocket transport rejects write/trading endpoints")


def _transport_reason(exc: Exception) -> DiagnosticCode:
    if isinstance(exc, HeartbeatTimeout):
        return DiagnosticCode.WS_HEARTBEAT_TIMEOUT
    if isinstance(exc, TimeoutError):
        return DiagnosticCode.WS_TIMEOUT
    if isinstance(exc, FeedResyncRequired):
        return DiagnosticCode.WS_REBOOTSTRAP_REQUIRED
    if isinstance(exc, ConnectionClosed):
        return DiagnosticCode.WS_CLOSED
    if isinstance(exc, aiohttp.ClientError):
        return DiagnosticCode.WS_HTTP_FAILURE
    if isinstance(exc, ValueError):
        return DiagnosticCode.WS_INVALID_PAYLOAD
    if isinstance(exc, WebSocketException):
        return DiagnosticCode.WS_PROTOCOL_ERROR
    return DiagnosticCode.WS_IO_FAILURE


async def _maybe_await(value: object) -> None:
    if inspect.isawaitable(value):
        await value


def _websocket_context(endpoint: str) -> AbstractAsyncContextManager[PublicSocket]:
    return websockets.connect(
        endpoint,
        ping_interval=None,
        ping_timeout=None,
        close_timeout=5,
        max_size=2**24,
    )
