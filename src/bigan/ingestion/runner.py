"""Orchestration layer: wires Gamma poll + WS client + sink + rollup together.

Designed for ``asyncio.run(IngestionRunner(...).serve())``. Handles graceful
shutdown on SIGINT/SIGTERM.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from contextlib import suppress

from prometheus_client import start_http_server

from .book_state import BookRegistry
from .clob_ws import ClobWsClient, EventHandler, WsClientConfig
from .config import IngestionSettings
from .gamma_client import ActiveMarket, GammaClient
from .message_types import BookEvent, MarketEvent, PriceChangeEvent
from .metrics import REGISTRY, WS_HASH_MISMATCH_TOTAL
from .rollup import run_rollup_worker
from .sink import NdjsonGzipSink

logger = logging.getLogger(__name__)


class IngestionRunner:
    """Owns the ingestion lifecycle for one runtime.

    Pure composition: each collaborator is constructed externally then
    plugged in here. Tests can inject fakes by subclassing or by calling
    :meth:`make_handler` directly.
    """

    def __init__(self, settings: IngestionSettings) -> None:
        self._settings = settings
        self._stop = asyncio.Event()
        self._books = BookRegistry()
        self._sink = NdjsonGzipSink(
            settings.raw_dir,
            flush_interval_seconds=settings.sink_flush_interval_seconds,
            max_buffer_records=settings.sink_max_buffer_records,
        )
        ws_cfg = WsClientConfig(
            url=settings.clob_ws_url,
            custom_feature_enabled=settings.ws_custom_feature_enabled,
            reconnect_min_seconds=settings.ws_reconnect_min_seconds,
            reconnect_max_seconds=settings.ws_reconnect_max_seconds,
            ping_interval_seconds=settings.ws_ping_interval_seconds,
            ping_timeout_seconds=settings.ws_ping_timeout_seconds,
            message_timeout_seconds=settings.ws_message_timeout_seconds,
        )
        self._ws = ClobWsClient(ws_cfg, self.make_handler())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def serve(self) -> None:
        """Run forever (until SIGINT/SIGTERM)."""
        self._install_signal_handlers()
        if self._settings.metrics_enabled:
            start_http_server(self._settings.metrics_port, registry=REGISTRY)
            logger.info("metrics.serving", extra={"port": self._settings.metrics_port})

        await self._sink.start_background_flusher()

        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._ws.run(), name="ws-client")
            tg.create_task(self._gamma_poller(), name="gamma-poller")
            if self._settings.rollup_enabled:
                tg.create_task(
                    run_rollup_worker(
                        self._settings.raw_dir,
                        self._settings.rollup_dir,
                        interval_seconds=self._settings.rollup_interval_seconds,
                        lag_seconds=self._settings.rollup_lag_seconds,
                        stop_event=self._stop,
                    ),
                    name="rollup-worker",
                )
            tg.create_task(self._shutdown_watcher(), name="shutdown-watcher")

        await self._sink.close()

    def stop(self) -> None:
        self._stop.set()
        self._ws.cancel()

    # ------------------------------------------------------------------
    # Event handler factory (also used by tests)
    # ------------------------------------------------------------------

    def make_handler(self) -> EventHandler:
        async def handler(event: MarketEvent, raw: dict) -> None:
            # Persist verbatim payload first; this is the contract with #4 and
            # downstream replay tooling.
            await self._sink.write({"receive_time": event.receive_time, "raw": raw})

            if isinstance(event, BookEvent):
                self._books.upsert_snapshot(event)
            elif isinstance(event, PriceChangeEvent):
                for change in event.price_changes:
                    book = self._books.apply_price_change(change.asset_id, change)
                    if book is None:
                        # We received a delta before the initial snapshot —
                        # rare under normal operation; will be reconciled when
                        # the snapshot arrives.
                        logger.debug(
                            "ws.delta_without_snapshot",
                            extra={"asset_id": change.asset_id},
                        )
                        continue
                    if book.last_hash and book.last_hash != change.hash:
                        WS_HASH_MISMATCH_TOTAL.labels(asset_id=change.asset_id).inc()
            # best_bid_ask / last_trade_price / tick_size_change / lifecycle:
            # raw archive is enough for v0; downstream features built in #6/#7.

        return handler

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _gamma_poller(self) -> None:
        async with GammaClient(
            self._settings.gamma_api_base,
            self._settings.market_slug_prefix,
        ) as gamma:
            while not self._stop.is_set():
                try:
                    markets = await gamma.list_active_markets()
                    asset_ids = self._asset_ids_from_markets(markets)
                    await self._ws.set_subscription(asset_ids)
                    logger.info(
                        "gamma.refreshed",
                        extra={"markets": len(markets), "assets": len(asset_ids)},
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("gamma.poll_failed")

                try:
                    await asyncio.wait_for(
                        self._stop.wait(),
                        timeout=self._settings.gamma_poll_interval_seconds,
                    )
                except TimeoutError:
                    continue

    @staticmethod
    def _asset_ids_from_markets(markets: list[ActiveMarket]) -> set[str]:
        out: set[str] = set()
        for m in markets:
            out.add(m.asset_id_up)
            out.add(m.asset_id_down)
        return out

    async def _shutdown_watcher(self) -> None:
        await self._stop.wait()
        self._ws.cancel()
        # Allow the task group to drain naturally.

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with suppress(NotImplementedError):  # Windows doesn't support add_signal_handler
                loop.add_signal_handler(sig, self.stop)
