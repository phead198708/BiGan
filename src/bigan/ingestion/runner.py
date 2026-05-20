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

from bigan.canonical.schemas import PROVENANCE_REST_SEED, PROVENANCE_WS

from .backfill import (
    BackfillReport,
    BackfillService,
    GapWindow,
    synth_orderbook_record,
)
from .backfill_control import (
    BackfillCircuitOpen,
    BackfillControlConfig,
    BackfillCoordinator,
)
from .book_state import BookRegistry
from .clob_rest import PolymarketRestClient
from .clob_ws import ClobWsClient, EventHandler, WsClientConfig
from .config import IngestionSettings
from .gamma_client import ActiveMarket, GammaClient, active_market_symbol_mapping_rows
from .gap_detector import GapDetector, GapEvent
from .message_types import BookEvent, MarketEvent, PriceChangeEvent
from .metrics import (
    BACKFILL_INVOCATIONS_TOTAL,
    BACKFILL_RECORDS_TOTAL,
    GAP_DETECTED_TOTAL,
    GAP_RESOLVED_TOTAL,
    GAP_SILENCE_DURATION_SECONDS,
    INITIAL_SNAPSHOT_RECORDS_TOTAL,
    INITIAL_SNAPSHOT_REQUESTS_TOTAL,
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
            reconnect_reset_after_seconds=settings.ws_reconnect_reset_after_seconds,
            ping_interval_seconds=settings.ws_ping_interval_seconds,
            ping_timeout_seconds=settings.ws_ping_timeout_seconds,
            idle_probe_timeout_seconds=settings.ws_idle_probe_timeout_seconds,
            message_timeout_seconds=settings.ws_message_timeout_seconds,
            ingest_lag_warn_seconds=settings.ingest_lag_warn_seconds,
        )
        self._ws = ClobWsClient(ws_cfg, self.make_handler())

        # --- Gap detection / backfill (issue #5) -------------------------
        self._gap_detector: GapDetector | None = None
        self._asset_market_map: dict[str, str] = {}
        self._asset_metadata_map: dict[str, dict[str, str]] = {}
        self._persisted_symbol_mapping_keys: set[tuple[str, str, str, str]] = set()
        self._active_asset_ids: set[str] = set()
        self._snapshot_seeded_assets: set[str] = set()
        self._snapshot_seed_inflight: set[str] = set()
        self._snapshot_seed_tasks: set[asyncio.Task[None]] = set()
        self._backfill_tasks: set[asyncio.Task[None]] = set()
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

        await self._drain_background_tasks()
        await self._sink.close()

    def stop(self) -> None:
        self._stop.set()
        self._ws.cancel()
        self._cancel_background_tasks()

    # ------------------------------------------------------------------
    # Event handler factory (also used by tests)
    # ------------------------------------------------------------------

    def make_handler(self) -> EventHandler:
        async def handler(event: MarketEvent, raw: dict) -> None:
            # Preserve the original payload under ``raw`` and attach ordering
            # metadata beside it so WS and REST snapshots can be compared.
            await self._sink.write(
                self._annotate_raw_record(_raw_record_from_ws_event(event, raw))
            )

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
                        self._schedule_backfill(resolved)

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
                    mapping_rows = self._refresh_market_metadata(markets)
                    await self._persist_symbol_mapping_rows(mapping_rows)
                    snapshot_assets = self._initial_snapshot_candidates(asset_ids)
                    await self._ws.set_subscription(asset_ids)
                    if self._settings.initial_snapshot_enabled:
                        self._schedule_initial_orderbook_snapshots(snapshot_assets)
                    logger.info(
                        "gamma.refreshed",
                        extra={
                            "markets": len(markets),
                            "assets": len(asset_ids),
                            "initial_snapshot_assets": len(snapshot_assets)
                            if self._settings.initial_snapshot_enabled
                            else 0,
                        },
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

    def _refresh_market_metadata(self, markets: list[ActiveMarket]) -> list[dict]:
        mapping_rows = active_market_symbol_mapping_rows(markets, ingest_ts=_now_ms())
        for m in markets:
            self._asset_market_map[m.asset_id_up] = m.condition_id
            self._asset_market_map[m.asset_id_down] = m.condition_id
        for row in mapping_rows:
            source_symbol = str(row["source_symbol"])
            canonical_symbol = str(row["canonical_symbol"])
            self._asset_metadata_map[source_symbol] = {
                "canonical_symbol": canonical_symbol,
                "outcome_side": canonical_symbol.rsplit(":", 1)[-1],
            }
        return mapping_rows

    async def _persist_symbol_mapping_rows(self, rows: list[dict]) -> None:
        new_rows = []
        for row in rows:
            key = (
                str(row.get("source") or ""),
                str(row.get("source_symbol") or ""),
                str(row.get("source_market") or ""),
                str(row.get("canonical_symbol") or ""),
            )
            if key in self._persisted_symbol_mapping_keys:
                continue
            self._persisted_symbol_mapping_keys.add(key)
            new_rows.append(row)
        if not new_rows:
            return
        now_ms = _now_ms()
        await self._sink.write(
            {
                "receive_time": now_ms,
                "source_timestamp_ms": now_ms,
                "capture_timestamp_ms": now_ms,
                "source_channel": "gamma-rest",
                "provenance": "gamma-rest",
                "raw": {
                    "event_type": "symbol_mapping",
                    "timestamp": str(now_ms),
                    "mappings": new_rows,
                },
            }
        )

    def _annotate_raw_record(self, record: dict) -> dict:
        raw = record.get("raw")
        if not isinstance(raw, dict):
            return record

        enriched_raw = dict(raw)
        asset_id = enriched_raw.get("asset_id")
        if asset_id is not None:
            metadata = self._asset_metadata_map.get(str(asset_id))
            if metadata is not None:
                enriched_raw.setdefault("canonical_symbol", metadata["canonical_symbol"])
                enriched_raw.setdefault("outcome_side", metadata["outcome_side"])

        price_changes = enriched_raw.get("price_changes")
        if isinstance(price_changes, list):
            enriched_entries = []
            changed = False
            for entry in price_changes:
                if not isinstance(entry, dict):
                    enriched_entries.append(entry)
                    continue
                asset_id = entry.get("asset_id")
                metadata = (
                    self._asset_metadata_map.get(str(asset_id))
                    if asset_id is not None
                    else None
                )
                if metadata is None:
                    enriched_entries.append(entry)
                    continue
                enriched_entry = dict(entry)
                enriched_entry.setdefault("canonical_symbol", metadata["canonical_symbol"])
                enriched_entry.setdefault("outcome_side", metadata["outcome_side"])
                enriched_entries.append(enriched_entry)
                changed = True
            if changed:
                enriched_raw["price_changes"] = enriched_entries

        return {**record, "raw": enriched_raw}

    def _initial_snapshot_candidates(self, asset_ids: set[str]) -> set[str]:
        self._active_asset_ids = set(asset_ids)
        return {
            asset_id
            for asset_id in asset_ids
            if asset_id not in self._snapshot_seeded_assets
            and asset_id not in self._snapshot_seed_inflight
        }

    def _schedule_initial_orderbook_snapshots(self, asset_ids: set[str]) -> None:
        candidates = sorted(
            asset_id
            for asset_id in asset_ids
            if asset_id not in self._snapshot_seeded_assets
            and asset_id not in self._snapshot_seed_inflight
        )
        if not candidates:
            return
        logger.info("snapshot_seed.scheduled", extra={"assets": len(candidates)})
        for asset_id in candidates:
            self._snapshot_seed_inflight.add(asset_id)
            task = asyncio.create_task(
                self._seed_initial_orderbook_snapshot(asset_id),
                name=f"snapshot-seed-{asset_id[:12]}",
            )
            self._snapshot_seed_tasks.add(task)
            task.add_done_callback(
                lambda done, aid=asset_id: self._on_snapshot_seed_done(aid, done)
            )

    def _on_snapshot_seed_done(
        self,
        asset_id: str,
        task: asyncio.Task[None],
    ) -> None:
        self._snapshot_seed_inflight.discard(asset_id)
        self._snapshot_seed_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                "snapshot_seed.task_failed",
                extra={"asset_id": asset_id},
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    async def _seed_initial_orderbook_snapshot(self, asset_id: str) -> None:
        try:
            outcome = await self._backfill_coordinator.run(
                asset_id=asset_id,
                operation=lambda before_rest_call: (
                    self._seed_initial_orderbook_snapshot_once(
                        asset_id,
                        before_rest_call=before_rest_call,
                    )
                ),
                is_failure=lambda result: result != "ok",
            )
        except BackfillCircuitOpen:
            INITIAL_SNAPSHOT_REQUESTS_TOTAL.labels(outcome="skipped_circuit").inc()
            logger.warning("snapshot_seed.circuit_open", extra={"asset_id": asset_id})
            return
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            INITIAL_SNAPSHOT_REQUESTS_TOTAL.labels(outcome="error").inc()
            logger.exception("snapshot_seed.failed", extra={"asset_id": asset_id})
            return

        INITIAL_SNAPSHOT_REQUESTS_TOTAL.labels(outcome=outcome).inc()
        if outcome == "ok":
            self._snapshot_seeded_assets.add(asset_id)

    async def _seed_initial_orderbook_snapshot_once(
        self,
        asset_id: str,
        *,
        before_rest_call,
    ) -> str:
        await before_rest_call()
        async with PolymarketRestClient(
            self._settings.clob_rest_url,
            data_api_base_url=self._settings.polymarket_data_api_url,
            timeout_seconds=self._settings.backfill_rest_timeout_seconds,
        ) as rest:
            book = await rest.fetch_orderbook(asset_id)
        if book is None:
            logger.warning("snapshot_seed.book_missing", extra={"asset_id": asset_id})
            return "missing"

        receive_time_ms = _now_ms()
        await self._sink.write(
            self._annotate_raw_record(
                synth_orderbook_record(
                    book,
                    receive_time_ms=receive_time_ms,
                    provenance=PROVENANCE_REST_SEED,
                )
            )
        )
        INITIAL_SNAPSHOT_RECORDS_TOTAL.inc()
        logger.info(
            "snapshot_seed.done",
            extra={
                "asset_id": asset_id,
                "market": book.market,
                "source_timestamp_ms": book.timestamp_ms,
                "capture_timestamp_ms": receive_time_ms,
                "hash": book.hash,
            },
        )
        return "ok"

    def _cancel_snapshot_seed_tasks(self) -> None:
        for task in list(self._snapshot_seed_tasks):
            task.cancel()

    def _schedule_backfill(self, gap: GapEvent) -> None:
        if self._stop.is_set():
            return
        task = asyncio.create_task(
            self._run_backfill(gap),
            name=f"backfill-{gap.asset_id[:12]}",
        )
        self._backfill_tasks.add(task)
        task.add_done_callback(
            lambda done, aid=gap.asset_id: self._on_backfill_done(aid, done)
        )

    def _on_backfill_done(self, asset_id: str, task: asyncio.Task[None]) -> None:
        self._backfill_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                "backfill.task_failed",
                extra={"asset_id": asset_id},
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    def _cancel_backfill_tasks(self) -> None:
        for task in list(self._backfill_tasks):
            task.cancel()

    def _cancel_background_tasks(self) -> None:
        self._cancel_snapshot_seed_tasks()
        self._cancel_backfill_tasks()

    async def _drain_background_tasks(self) -> None:
        self._cancel_background_tasks()
        tasks = [*self._snapshot_seed_tasks, *self._backfill_tasks]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

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
                _MetadataAnnotatingSink(self._sink, self._annotate_raw_record),
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


def _raw_record_from_ws_event(event: MarketEvent, raw: dict) -> dict:
    capture_timestamp_ms = int(event.receive_time or _now_ms())
    source_timestamp_ms = int(event.timestamp)
    return {
        "receive_time": capture_timestamp_ms,
        "source_timestamp_ms": source_timestamp_ms,
        "capture_timestamp_ms": capture_timestamp_ms,
        "source_channel": "clob-ws",
        "provenance": PROVENANCE_WS,
        "raw": raw,
    }


class _MetadataAnnotatingSink:
    def __init__(self, sink, annotate) -> None:
        self._sink = sink
        self._annotate = annotate

    async def write(self, record: dict) -> None:
        await self._sink.write(self._annotate(record))


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
