"""Orchestration layer: wires Gamma poll + WS client + sink + rollup together.

Designed for ``asyncio.run(IngestionRunner(...).serve())``. Handles graceful
shutdown on SIGINT/SIGTERM.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from contextlib import suppress

from prometheus_client import start_http_server

from .backfill import BackfillReport, BackfillService, GapWindow
from .backfill_control import (
    BackfillCircuitOpen,
    BackfillControlConfig,
    BackfillCoordinator,
)
from .book_state import BookRegistry
from .clob_rest import PolymarketRestClient
from .clob_ws import ClobWsClient, EventHandler, WsClientConfig
from .config import IngestionSettings
from .gamma_client import ActiveMarket, GammaClient
from .gap_detector import GapDetector, GapEvent
from .message_types import BookEvent, MarketEvent, PriceChangeEvent
from .metrics import (
    BACKFILL_INVOCATIONS_TOTAL,
    BACKFILL_RECORDS_TOTAL,
    GAP_DETECTED_TOTAL,
    GAP_RESOLVED_TOTAL,
    GAP_SILENCE_DURATION_SECONDS,
    REGISTRY,
    WS_HASH_MISMATCH_TOTAL,
)
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
            ingest_lag_warn_seconds=settings.ingest_lag_warn_seconds,
        )
        self._ws = ClobWsClient(ws_cfg, self.make_handler())

        # --- Gap detection / backfill (issue #5) -------------------------
        self._gap_detector: GapDetector | None = None
        self._asset_market_map: dict[str, str] = {}
        self._backfill_coordinator = BackfillCoordinator(
            BackfillControlConfig(
                max_concurrency=settings.backfill_max_concurrency,
                rate_limit_per_second=settings.backfill_rate_limit_per_second,
                circuit_failure_threshold=settings.backfill_circuit_failure_threshold,
                circuit_cool_down_seconds=settings.backfill_circuit_cool_down_seconds,
            )
        )
        if settings.gap_detection_enabled:
            self._gap_detector = GapDetector(
                silence_threshold_ms=int(
                    settings.gap_silence_threshold_seconds * 1000
                ),
                min_gap_resume_ms=int(settings.gap_min_resume_seconds * 1000),
                on_gap_started=self._on_gap_started,
            )

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
            if self._gap_detector is not None:
                tg.create_task(self._gap_watchdog(), name="gap-watchdog")
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

            # Feed gap detector — one note() per asset_id seen in this event.
            if self._gap_detector is not None:
                for asset_id in _asset_ids_in_event(event, raw):
                    resolved = self._gap_detector.note(asset_id, event.receive_time)
                    if resolved is not None:
                        # Spawn backfill asynchronously so the live handler
                        # never blocks on a REST round-trip.
                        asyncio.create_task(  # noqa: RUF006 — fire-and-forget by design
                            self._run_backfill(resolved),
                            name=f"backfill-{resolved.asset_id}",
                        )

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
                    self._refresh_asset_market_map(markets)
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

    def _refresh_asset_market_map(self, markets: list[ActiveMarket]) -> None:
        for m in markets:
            self._asset_market_map[m.asset_id_up] = m.condition_id
            self._asset_market_map[m.asset_id_down] = m.condition_id

    async def _resolve_market(self, asset_id: str) -> str | None:
        return self._asset_market_map.get(asset_id)

    # ------------------------------------------------------------------
    # Gap watchdog + backfill (issue #5)
    # ------------------------------------------------------------------

    async def _gap_watchdog(self) -> None:
        """Periodically prods the gap detector so silence-into-gap
        transitions are logged even before activity resumes.

        The actual gap-resolved -> backfill trigger lives in the WS
        handler (see :meth:`make_handler`), so this task only needs to
        emit detection alerts.
        """
        if self._gap_detector is None:
            return
        interval = self._settings.gap_check_interval_seconds
        while not self._stop.is_set():
            try:
                self._gap_detector.tick(_now_ms())
            except Exception:  # noqa: BLE001
                logger.exception("gap.tick_failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except TimeoutError:
                continue

    def _on_gap_started(self, asset_id: str, last_seen_ms: int) -> None:
        GAP_DETECTED_TOTAL.labels(asset_id=asset_id).inc()

    async def _run_backfill(self, gap: GapEvent) -> None:
        GAP_RESOLVED_TOTAL.labels(asset_id=gap.asset_id).inc()
        GAP_SILENCE_DURATION_SECONDS.observe(gap.silence_duration_ms / 1000.0)

        try:
            report = await self._backfill_coordinator.run(
                asset_id=gap.asset_id,
                operation=lambda before_rest_call: self._run_backfill_once(
                    gap,
                    before_rest_call=before_rest_call,
                ),
                is_failure=_backfill_report_has_rest_failure,
            )
        except BackfillCircuitOpen:
            BACKFILL_INVOCATIONS_TOTAL.labels(outcome="skipped").inc()
            return
        except Exception:  # noqa: BLE001
            BACKFILL_INVOCATIONS_TOTAL.labels(outcome="error").inc()
            logger.exception(
                "backfill.invocation_failed",
                extra={"asset_id": gap.asset_id},
            )
            return

        outcome = "ok" if not report.errors else "partial"
        BACKFILL_INVOCATIONS_TOTAL.labels(outcome=outcome).inc()
        if report.trades_replayed:
            BACKFILL_RECORDS_TOTAL.labels(kind="trade").inc(report.trades_replayed)
        if report.orderbook_replayed:
            BACKFILL_RECORDS_TOTAL.labels(kind="orderbook").inc()

    async def _run_backfill_once(
        self,
        gap: GapEvent,
        *,
        before_rest_call,
    ) -> BackfillReport:
        async with PolymarketRestClient(
            self._settings.clob_rest_url,
            data_api_base_url=self._settings.polymarket_data_api_url,
            timeout_seconds=self._settings.backfill_rest_timeout_seconds,
        ) as rest:
            service = BackfillService(
                rest,
                self._sink,
                self._resolve_market,
                before_rest_call=before_rest_call,
            )
            return await service.handle_gap(
                GapWindow(
                    asset_id=gap.asset_id,
                    gap_start_ms=gap.gap_start_ms,
                    gap_end_ms=gap.gap_end_ms,
                )
            )

    async def _shutdown_watcher(self) -> None:
        await self._stop.wait()
        self._ws.cancel()
        # Allow the task group to drain naturally.

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with suppress(NotImplementedError):  # Windows doesn't support add_signal_handler
                loop.add_signal_handler(sig, self.stop)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _now_ms() -> int:
    return int(time.time() * 1000)


def _asset_ids_in_event(event: MarketEvent, raw: dict) -> set[str]:
    """Best-effort extraction of every asset_id touched by ``event``.

    Most CLOB events carry ``asset_id`` at the top level. ``price_change``
    is the exception: its asset_ids live inside the per-entry payload.
    """
    out: set[str] = set()
    asset_id = raw.get("asset_id")
    if asset_id:
        out.add(str(asset_id))
    for entry in raw.get("price_changes") or []:
        if isinstance(entry, dict) and entry.get("asset_id"):
            out.add(str(entry["asset_id"]))
    return out


def _backfill_report_has_rest_failure(report: BackfillReport) -> bool:
    return any("fetch_failed" in error for error in report.errors)
