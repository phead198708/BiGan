"""Polymarket CLOB ``market`` channel WebSocket client.

Responsibilities:
- Maintain a single WebSocket connection with exponential-backoff reconnect.
- Send / refresh subscription payloads as the active market set changes.
- Receive messages with a ``ws_message_timeout_seconds`` watchdog; quiet
  periods trigger a ping probe before the client reconnects.
- Hand parsed events plus the verbatim payload to a caller-supplied callback.

The class does **not** persist anything itself; the runner wires it up to a
sink + book registry + metrics.

Design notes:
- The client owns periodic protocol pings rather than relying on
  ``websockets`` keepalive. This lets us keep ping-timeout disconnects disabled
  while still consuming pong-future exceptions after remote closes. Quiet
  receive periods are checked separately by the explicit idle watchdog below.
- The CLOB market channel does not require an initial "subscribe" message at
  connect time — the subscription payload is the only message we send before
  receiving snapshots. To change subscriptions mid-flight we send a new
  subscribe payload listing the *full desired set*; the server treats this as
  a replacement set. This matches the documented behaviour and avoids needing
  separate unsubscribe semantics.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import orjson
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from .message_types import MarketEvent, UnknownEvent, parse_event
from .metrics import (
    INGEST_LAG_SECONDS,
    LAST_EVENT_RECEIVE_TIME,
    WS_MESSAGES_TOTAL,
    WS_PARSE_ERRORS_TOTAL,
    WS_RECONNECTS_TOTAL,
    WS_SUBSCRIBED_MARKETS,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WsClientConfig:
    url: str
    custom_feature_enabled: bool = True
    reconnect_min_seconds: float = 1.0
    reconnect_max_seconds: float = 30.0
    reconnect_reset_after_seconds: float = 60.0
    ping_interval_seconds: float | None = 20.0
    ping_timeout_seconds: float | None = None
    idle_probe_timeout_seconds: float = 10.0
    message_timeout_seconds: float = 45.0
    ingest_lag_warn_seconds: float = 0.5


# Callback signature: (parsed_event, raw_payload_dict) -> awaitable
EventHandler = Callable[[MarketEvent, dict], Awaitable[None]]


class ClobWsClient:
    """Resilient CLOB market-channel WS client.

    Lifecycle:
        client = ClobWsClient(cfg, handler)
        await client.set_subscription(asset_ids)
        await client.run()       # blocks until cancel()
        client.cancel()
    """

    def __init__(self, config: WsClientConfig, handler: EventHandler) -> None:
        self._cfg = config
        self._handler = handler
        self._desired_assets: set[str] = set()
        self._desired_updated = asyncio.Event()
        self._cancelled = asyncio.Event()
        self._connection: websockets.WebSocketClientProtocol | None = None

    # ------------------------------------------------------------------
    # Subscription management (called from any task)
    # ------------------------------------------------------------------

    async def set_subscription(self, asset_ids: set[str] | list[str] | tuple[str, ...]) -> None:
        """Set the desired subscription set.

        If the set differs from the current desired set, the next loop
        iteration will send a fresh subscribe payload.
        """
        new_set = set(asset_ids)
        if new_set == self._desired_assets:
            return
        self._desired_assets = new_set
        WS_SUBSCRIBED_MARKETS.set(len(new_set))
        self._desired_updated.set()

    def cancel(self) -> None:
        """Stop the run loop on next iteration."""
        self._cancelled.set()
        self._desired_updated.set()

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main loop: connect, subscribe, dispatch messages, reconnect on failure."""
        backoff = self._cfg.reconnect_min_seconds
        while not self._cancelled.is_set():
            attempt_started_at = time.monotonic()
            try:
                await self._connect_and_run()
                backoff = self._cfg.reconnect_min_seconds  # reset on clean exit
            except asyncio.CancelledError:
                raise
            except (ConnectionClosed, WebSocketException, OSError) as exc:
                backoff = self._backoff_for_failed_connection(
                    backoff,
                    attempt_started_at=attempt_started_at,
                )
                WS_RECONNECTS_TOTAL.inc()
                _log_connection_failed(exc, backoff_s=backoff)
                await self._sleep_with_jitter(backoff)
                backoff = min(backoff * 2, self._cfg.reconnect_max_seconds)
            except Exception as exc:  # noqa: BLE001
                backoff = self._backoff_for_failed_connection(
                    backoff,
                    attempt_started_at=attempt_started_at,
                )
                expected = _expected_connection_exception(exc)
                if expected is not None:
                    WS_RECONNECTS_TOTAL.inc()
                    _log_connection_failed(expected, backoff_s=backoff)
                else:
                    WS_RECONNECTS_TOTAL.inc()
                    logger.exception(
                        "ws.unexpected_error backoff_s=%s",
                        backoff,
                        extra={"backoff_s": backoff},
                    )
                await self._sleep_with_jitter(backoff)
                backoff = min(backoff * 2, self._cfg.reconnect_max_seconds)

    def _backoff_for_failed_connection(
        self,
        current_backoff: float,
        *,
        attempt_started_at: float,
    ) -> float:
        if time.monotonic() - attempt_started_at >= self._cfg.reconnect_reset_after_seconds:
            return self._cfg.reconnect_min_seconds
        return current_backoff

    async def _sleep_with_jitter(self, base: float) -> None:
        delay = base * (0.5 + random.random())
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._cancelled.wait(), timeout=delay)

    async def _connect_and_run(self) -> None:
        logger.info("ws.connecting", extra={"url": self._cfg.url})
        async with websockets.connect(
            self._cfg.url,
            ping_interval=None,
            ping_timeout=None,
            close_timeout=5,
            max_size=2**24,  # 16 MB, handles large book snapshots
        ) as ws:
            self._connection = ws
            try:
                # Send the initial subscribe only when we actually have a
                # non-empty desired set. Polymarket's market channel rejects
                # empty subscriptions with a plain-text error response, which
                # also leaves the connection unable to receive further data.
                if self._desired_assets:
                    await self._send_subscription()
                    self._desired_updated.clear()

                # Two tasks: receive loop + subscription refresh loop.
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self._receive_loop(ws), name="ws-recv")
                    tg.create_task(self._subscription_refresher(ws), name="ws-sub-refresh")
                    tg.create_task(self._protocol_ping_loop(ws), name="ws-protocol-ping")
            finally:
                self._connection = None

    async def _send_subscription(self) -> None:
        """Push the current desired set as a subscribe payload."""
        ws = self._connection
        if ws is None:
            return
        if not self._desired_assets:
            # Avoid sending an empty subscription (see _connect_and_run note).
            return
        payload = {
            "assets_ids": sorted(self._desired_assets),
            "type": "market",
        }
        if self._cfg.custom_feature_enabled:
            payload["custom_feature_enabled"] = True
        await ws.send(orjson.dumps(payload))
        logger.info(
            "ws.subscribed",
            extra={"n": len(self._desired_assets)},
        )

    async def _subscription_refresher(self, ws: websockets.WebSocketClientProtocol) -> None:
        """Watch for ``_desired_updated`` and resend subscribe payload."""
        while not self._cancelled.is_set():
            await self._desired_updated.wait()
            if self._cancelled.is_set():
                return
            self._desired_updated.clear()
            await self._send_subscription()

    async def _protocol_ping_loop(self, ws: websockets.WebSocketClientProtocol) -> None:
        """Send protocol pings without closing healthy streams on late pongs."""
        interval = self._cfg.ping_interval_seconds
        if interval is None:
            return
        while not self._cancelled.is_set():
            try:
                await asyncio.wait_for(self._cancelled.wait(), timeout=interval)
                return
            except TimeoutError:
                pass

            try:
                pong_waiter = await ws.ping()
            except (ConnectionClosed, WebSocketException, OSError):
                return
            _consume_future_exception(pong_waiter)

            timeout = self._cfg.ping_timeout_seconds
            if timeout is None:
                continue
            try:
                await asyncio.wait_for(asyncio.shield(pong_waiter), timeout=timeout)
            except TimeoutError as exc:
                raise ConnectionClosed(None, None) from exc
            except (ConnectionClosed, WebSocketException, OSError):
                return

    async def _receive_loop(self, ws: websockets.WebSocketClientProtocol) -> None:
        """Pump messages from the WS into the handler with a watchdog."""
        timeout = self._cfg.message_timeout_seconds
        while not self._cancelled.is_set():
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            except TimeoutError:
                await self._confirm_idle_connection(ws)
                continue

            raw_bytes = raw if isinstance(raw, bytes) else raw.encode("utf-8")

            await self._dispatch(raw_bytes)

    async def _confirm_idle_connection(
        self,
        ws: websockets.WebSocketClientProtocol,
    ) -> None:
        """Probe a quiet connection before counting it as a reconnect."""
        try:
            pong_waiter = await ws.ping()
            await asyncio.wait_for(
                pong_waiter,
                timeout=self._cfg.idle_probe_timeout_seconds,
            )
        except (TimeoutError, ConnectionClosed, WebSocketException, OSError) as exc:
            raise ConnectionClosed(None, None) from exc

        LAST_EVENT_RECEIVE_TIME.set(time.time())
        logger.debug(
            "ws.idle_ping_ok",
            extra={
                "idle_timeout_s": self._cfg.message_timeout_seconds,
                "probe_timeout_s": self._cfg.idle_probe_timeout_seconds,
            },
        )

    # Polymarket sometimes sends plain-text application-level keepalive
    # frames (e.g. ``PONG``) outside the JSON envelope. We accept any plain
    # ASCII frame of <= 8 bytes as keepalive and ignore it silently.
    _KEEPALIVE_TOKENS = (b"PONG", b"PING", b"pong", b"ping")

    async def _dispatch(self, raw_bytes: bytes) -> None:
        receive_time_ms = int(time.time() * 1000)
        LAST_EVENT_RECEIVE_TIME.set(receive_time_ms / 1000.0)
        stripped = raw_bytes.strip()
        if stripped in self._KEEPALIVE_TOKENS:
            return
        try:
            payload = orjson.loads(raw_bytes)
        except orjson.JSONDecodeError:
            WS_PARSE_ERRORS_TOTAL.labels(kind="json").inc()
            logger.warning(
                "ws.json_decode_failed",
                extra={"len": len(raw_bytes), "preview": raw_bytes[:80].decode("utf-8", "replace")},
            )
            return

        # The CLOB market channel sometimes batches several events in a single
        # JSON array message; handle both flavours.
        if isinstance(payload, list):
            for item in payload:
                await self._dispatch_single(item, receive_time_ms)
            return
        await self._dispatch_single(payload, receive_time_ms)

    async def _dispatch_single(self, payload: dict, receive_time_ms: int) -> None:
        try:
            event = parse_event(payload, receive_time_ms=receive_time_ms)
        except UnknownEvent as exc:
            WS_PARSE_ERRORS_TOTAL.labels(kind="unknown_event").inc()
            logger.debug("ws.unknown_event", extra={"err": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001 — pydantic validation etc.
            WS_PARSE_ERRORS_TOTAL.labels(kind="validation").inc()
            logger.warning("ws.validation_failed", extra={"err": str(exc)})
            return

        WS_MESSAGES_TOTAL.labels(event_type=event.event_type.value).inc()
        LAST_EVENT_RECEIVE_TIME.set(receive_time_ms / 1000.0)
        _observe_ingest_lag(
            event=event,
            payload=payload,
            receive_time_ms=receive_time_ms,
            warn_threshold_seconds=self._cfg.ingest_lag_warn_seconds,
        )
        await self._handler(event, payload)


def _observe_ingest_lag(
    *,
    event: MarketEvent,
    payload: dict,
    receive_time_ms: int,
    warn_threshold_seconds: float,
) -> None:
    message_ts_ms = int(event.timestamp)
    lag_seconds = (receive_time_ms - message_ts_ms) / 1000.0
    event_type = event.event_type.value
    INGEST_LAG_SECONDS.labels(source="polymarket", event_type=event_type).observe(
        lag_seconds
    )
    if lag_seconds > warn_threshold_seconds:
        logger.warning(
            "ingest_lag.high",
            extra={
                "asset_id": _asset_id_for_log(event, payload),
                "event_type": event_type,
                "lag_ms": int(receive_time_ms - message_ts_ms),
                "threshold_ms": int(warn_threshold_seconds * 1000),
            },
        )


def _asset_id_for_log(event: MarketEvent, payload: dict) -> str | None:
    asset_id = getattr(event, "asset_id", None)
    if asset_id is not None:
        return str(asset_id)
    price_changes = payload.get("price_changes")
    if isinstance(price_changes, list) and price_changes:
        first = price_changes[0]
        if isinstance(first, dict) and first.get("asset_id") is not None:
            return str(first["asset_id"])
    return None


def _consume_future_exception(future: asyncio.Future[object]) -> None:
    """Mark future exceptions as retrieved when pings are intentionally non-fatal."""

    def _done(done: asyncio.Future[object]) -> None:
        try:
            done.exception()
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    future.add_done_callback(_done)


def _expected_connection_exception(exc: BaseException) -> BaseException | None:
    """Return a reconnect-worthy exception, unwrapping TaskGroup noise.

    Python 3.11+ wraps child-task exceptions in ``ExceptionGroup``. The receive
    loop deliberately raises ``ConnectionClosed`` on message-watchdog timeout,
    so a grouped ``ConnectionClosed`` is still a normal reconnect path.
    """

    if isinstance(exc, (ConnectionClosed, WebSocketException, OSError)):
        return exc
    if isinstance(exc, BaseExceptionGroup):
        for child in exc.exceptions:
            found = _expected_connection_exception(child)
            if found is not None:
                return found
    return None


def _log_connection_failed(exc: BaseException, *, backoff_s: float) -> None:
    context = _connection_error_context(exc, backoff_s=backoff_s)
    logger.warning(
        (
            "ws.connection_failed err_type=%s err=%r close_code=%s "
            "close_reason=%r cause_type=%s cause=%r backoff_s=%s"
        ),
        context["err_type"],
        context["err"],
        context["close_code"],
        context["close_reason"],
        context["cause_type"],
        context["cause"],
        context["backoff_s"],
        extra=context,
    )


def _connection_error_context(
    exc: BaseException,
    *,
    backoff_s: float,
) -> dict[str, object]:
    """Return reconnect diagnostics that survive plain logging formatters."""

    close_code = getattr(exc, "code", None)
    close_reason = getattr(exc, "reason", None)
    rcvd = getattr(exc, "rcvd", None)
    if rcvd is not None:
        close_code = close_code if close_code is not None else getattr(rcvd, "code", None)
        close_reason = (
            close_reason
            if close_reason is not None
            else getattr(rcvd, "reason", None)
        )
    cause = exc.__cause__
    return {
        "err_type": type(exc).__name__,
        "err": str(exc),
        "close_code": close_code,
        "close_reason": close_reason,
        "cause_type": type(cause).__name__ if cause is not None else None,
        "cause": str(cause) if cause is not None else None,
        "backoff_s": backoff_s,
    }
