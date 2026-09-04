"""Lifecycle wiring for the operator's three public read-only streams."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from contextlib import suppress

from .chainlink_twap import oracle_source
from .pricing_inputs import ReferencePriceSample
from .read_model import OperatorState
from .runtime import PaperTradingOperator
from .transports import (
    AiohttpPublicJSONClient,
    ChainlinkReadonlyFeed,
    PublicJSONClient,
    PublicWebSocketTransport,
    binance_subscription,
    chainlink_subscription,
)


class LiveFeedSupervisor:
    """Restart window-scoped feed pumps after every successful rollover."""

    def __init__(
        self,
        *,
        operator: PaperTradingOperator,
        http: PublicJSONClient | None = None,
    ) -> None:
        self.operator = operator
        self.http = http or AiohttpPublicJSONClient()

    async def run(self, stop_event: asyncio.Event) -> None:
        try:
            await self.operator.start()
            while not stop_event.is_set():
                if self.operator.state in {OperatorState.FAILED, OperatorState.EXHAUSTED}:
                    stop_event.set()
                    break
                if self.operator.session is None or self.operator.active_market is None:
                    await self._wait_interval(stop_event)
                    if not stop_event.is_set():
                        await self.operator.start()
                    continue
                window_generation = self.operator.generation
                tasks = self._start_window_feeds(window_generation, stop_event)
                try:
                    while (
                        not stop_event.is_set()
                        and window_generation == self.operator.generation
                        and self.operator.state not in {OperatorState.FAILED, OperatorState.EXHAUSTED}
                    ):
                        await self._wait_interval(stop_event)
                        if not stop_event.is_set():
                            await self.operator.poll()
                    if self.operator.state in {OperatorState.FAILED, OperatorState.EXHAUSTED}:
                        stop_event.set()
                finally:
                    for task in tasks:
                        task.cancel()
                    for task in tasks:
                        with suppress(asyncio.CancelledError):
                            await task
        finally:
            await self.operator.shutdown()

    def _start_window_feeds(
        self,
        window_generation: int,
        stop_event: asyncio.Event,
    ) -> tuple[asyncio.Task[None], ...]:
        config = self.operator.config
        market = self.operator.active_market
        if market is None:
            return ()

        async def binance_generation(connection_generation: int) -> None:
            snapshot = await self.http.get_json(
                config.binance_depth_endpoint,
                params={"symbol": config.binance_symbol, "limit": 1000},
            )
            if not isinstance(snapshot, dict):
                raise ValueError("Binance depth snapshot must be an object")
            accepted = await self.operator.begin_binance_connection(
                window_generation=window_generation,
                connection_generation=connection_generation,
                snapshot=snapshot,
                received_at_ms=self.operator.clock_ms(),
            )
            if not accepted:
                raise ConnectionError("Binance depth bootstrap did not synchronize")

        async def binance_payload(
            payload: Mapping[str, object],
            connection_generation: int,
            received_at_ms: int,
        ) -> None:
            # Some exchange clocks lead local receipt by a few milliseconds.
            # Wait boundedly, keep both original timestamps, and let the
            # synchronizer re-check local time. Large leads still fail closed.
            event = payload.get("data", payload)
            event_ts = event.get("E") if isinstance(event, Mapping) else None
            if type(event_ts) is int:
                delay_ms = event_ts - self.operator.clock_ms()
                tolerance = config.binance_clock_ahead_tolerance_ms
                if (0 < delay_ms <= tolerance
                        and event_ts <= received_at_ms + tolerance):
                    await asyncio.sleep(delay_ms / 1_000)
            accepted = await self.operator.ingest_binance_connection_delta(
                dict(payload),
                window_generation=window_generation,
                connection_generation=connection_generation,
                received_at_ms=received_at_ms,
            )
            synchronizer = self.operator.binance_sync
            if not accepted and synchronizer is not None and synchronizer.needs_bootstrap:
                raise ConnectionError("Binance depth gap requires immediate re-bootstrap")

        async def market_payload(
            payload: Mapping[str, object],
            connection_generation: int,
            received_at_ms: int,
        ) -> None:
            await self.operator.ingest_market_connection_payload(
                dict(payload),
                window_generation=window_generation,
                connection_generation=connection_generation,
                received_at_ms=received_at_ms,
            )
            synchronizer = self.operator.market_sync
            if synchronizer is not None and synchronizer.needs_bootstrap:
                raise ConnectionError("Polymarket depth requires a fresh full-book subscription")

        async def oracle_sample(
            sample: ReferencePriceSample,
            _connection_generation: int,
        ) -> None:
            await self.operator.ingest_oracle(
                sample,
                generation=window_generation,
            )

        chainlink = ChainlinkReadonlyFeed(
            expected_symbol=config.chainlink_symbol,
            source=oracle_source(config.chainlink_symbol, market.oracle_twap_lookback_seconds),
            on_sample=oracle_sample,
            lookback_seconds=market.oracle_twap_lookback_seconds,
        )

        async def chainlink_payload(
            payload: Mapping[str, object],
            connection_generation: int,
            received_at_ms: int,
        ) -> None:
            await chainlink.on_raw(
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
                generation=connection_generation,
                received_at_ms=received_at_ms,
            )

        binance = PublicWebSocketTransport(
            endpoint=config.binance_ws_url,
            subscription=binance_subscription(config.binance_symbol),
            queue_size=config.binance_queue_size,
            on_payload=binance_payload,
            on_generation=binance_generation,
            on_disconnect=lambda: self.operator.disconnect_feed(
                "binance", window_generation=window_generation
            ),
            reconnect_min_seconds=config.reconnect_min_seconds,
            reconnect_max_seconds=config.reconnect_max_seconds,
            heartbeat_interval_seconds=config.heartbeat_interval_seconds,
            clock_ms=self.operator.clock_ms,
        )
        polymarket = PublicWebSocketTransport(
            endpoint=config.polymarket_ws_url,
            subscription={
                "assets_ids": [market.yes_token_id, market.no_token_id],
                "type": "market",
                "custom_feature_enabled": True,
            },
            queue_size=config.market_queue_size,
            on_payload=market_payload,
            on_generation=lambda connection_generation: self.operator.begin_market_connection(
                window_generation=window_generation,
                connection_generation=connection_generation,
            ),
            on_disconnect=lambda: self.operator.disconnect_feed(
                "polymarket", window_generation=window_generation
            ),
            reconnect_min_seconds=config.reconnect_min_seconds,
            reconnect_max_seconds=config.reconnect_max_seconds,
            heartbeat_interval_seconds=config.heartbeat_interval_seconds,
            clock_ms=self.operator.clock_ms,
            application_heartbeat="PING",
        )
        chainlink_transport = PublicWebSocketTransport(
            endpoint=config.chainlink_ws_url,
            subscription=chainlink_subscription(config.chainlink_symbol, market.oracle_twap_lookback_seconds),
            queue_size=config.binance_queue_size,
            on_payload=chainlink_payload,
            on_generation=lambda connection_generation: self.operator.begin_oracle_connection(
                window_generation=window_generation,
                connection_generation=connection_generation,
            ),
            on_disconnect=lambda: self.operator.disconnect_feed(
                "chainlink", window_generation=window_generation
            ),
            reconnect_min_seconds=config.reconnect_min_seconds,
            reconnect_max_seconds=config.reconnect_max_seconds,
            heartbeat_interval_seconds=min(5.0, config.heartbeat_interval_seconds),
            clock_ms=self.operator.clock_ms,
            application_heartbeat="PING",
        )
        return (
            asyncio.create_task(binance.run(stop_event), name="paper-binance-depth"),
            asyncio.create_task(polymarket.run(stop_event), name="paper-polymarket-clob"),
            asyncio.create_task(chainlink_transport.run(stop_event), name="paper-chainlink-rtds"),
        )

    async def _wait_interval(self, stop_event: asyncio.Event) -> None:
        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=self.operator.config.status_interval_ms / 1_000.0,
            )
        except TimeoutError:
            return
