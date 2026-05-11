"""REST-based backfill service for stream-outage recovery (issue #5).

When :class:`bigan.ingestion.gap_detector.GapDetector` resolves a gap,
this service is invoked to recover the missing market data via the CLOB
REST API and re-inject it into the same NDJSON sink the WebSocket pipe
uses. Backfilled records are tagged with ``provenance =
"polymarket-rest-backfill"`` so downstream consumers (canonical ETL,
features, models) can distinguish them from realtime WS records.

Inputs:
- one :class:`GapEvent` carrying the gap window and the affected asset_id
- a resolver that maps asset_id → market condition_id (provided by the
  runner via the Gamma poller's active-market cache)

Outputs:
- ``BackfillReport`` with counts of trades replayed, the orderbook
  refresh outcome, and any errors

The service does **not** mutate the local order-book registry directly;
the canonical pipeline reconstructs that from the synthesised events.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from bigan.canonical.schemas import PROVENANCE_BACKFILL

from .clob_rest import PolymarketRestClient, RestOrderbook, RestTrade
from .sink import Sink

logger = logging.getLogger(__name__)


#: Resolver signature: given an asset_id, return its market condition_id
#: (or ``None`` if not currently known). Async-friendly so the runner can
#: lazily query Gamma if the local cache misses.
MarketResolver = Callable[[str], Awaitable[str | None]]


@dataclass(slots=True)
class GapWindow:
    """Convenience wrapper paralleling :class:`GapEvent` from gap_detector.

    Kept as a separate dataclass so the backfill module stays decoupled
    from gap_detector for testing.
    """

    asset_id: str
    gap_start_ms: int
    gap_end_ms: int


@dataclass(slots=True)
class BackfillReport:
    """Result of a single backfill invocation."""

    asset_id: str
    market: str | None
    gap_start_ms: int
    gap_end_ms: int
    trades_replayed: int = 0
    orderbook_replayed: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def total_records(self) -> int:
        return self.trades_replayed + (1 if self.orderbook_replayed else 0)


class BackfillService:
    """Coordinates REST calls and sink replay for one gap event."""

    def __init__(
        self,
        rest_client: PolymarketRestClient,
        sink: Sink,
        market_resolver: MarketResolver,
        *,
        provenance: str = PROVENANCE_BACKFILL,
        clock_ms: Callable[[], int] = lambda: int(time.time() * 1000),
    ) -> None:
        self._rest = rest_client
        self._sink = sink
        self._resolver = market_resolver
        self._provenance = provenance
        self._clock_ms = clock_ms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def handle_gap(self, gap: GapWindow) -> BackfillReport:
        """Fetch missed data for ``gap`` and append it to the sink."""
        report = BackfillReport(
            asset_id=gap.asset_id,
            market=None,
            gap_start_ms=gap.gap_start_ms,
            gap_end_ms=gap.gap_end_ms,
        )
        market = await self._resolver(gap.asset_id)
        report.market = market

        logger.info(
            "backfill.start",
            extra={
                "asset_id": gap.asset_id,
                "market": market,
                "gap_start_ms": gap.gap_start_ms,
                "gap_end_ms": gap.gap_end_ms,
                "silence_duration_ms": gap.gap_end_ms - gap.gap_start_ms,
            },
        )

        if market is None:
            err = "market resolver returned None; cannot fetch trades"
            report.errors.append(err)
            logger.warning("backfill.no_market", extra={"asset_id": gap.asset_id})
        else:
            await self._replay_trades(gap, market, report)

        await self._replay_orderbook(gap, report)

        logger.info(
            "backfill.done",
            extra={
                "asset_id": gap.asset_id,
                "market": market,
                "trades_replayed": report.trades_replayed,
                "orderbook_replayed": report.orderbook_replayed,
                "errors": report.errors,
                "total_records": report.total_records,
            },
        )
        return report

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _replay_trades(
        self,
        gap: GapWindow,
        market: str,
        report: BackfillReport,
    ) -> None:
        try:
            trades = await self._rest.fetch_trades(
                market,
                since_ms=gap.gap_start_ms,
                until_ms=gap.gap_end_ms,
            )
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"trades_fetch_failed: {exc}")
            logger.exception(
                "backfill.trades_fetch_failed",
                extra={"asset_id": gap.asset_id, "market": market},
            )
            return

        for trade in trades:
            if trade.asset_id != gap.asset_id:
                # Same market, opposite outcome: not in the gap we're recovering.
                continue
            record = self._synth_trade_record(trade)
            try:
                await self._sink.write(record)
                report.trades_replayed += 1
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"trade_sink_failed: {exc}")
                logger.exception(
                    "backfill.trade_sink_failed",
                    extra={"asset_id": gap.asset_id},
                )
                break

    async def _replay_orderbook(
        self,
        gap: GapWindow,
        report: BackfillReport,
    ) -> None:
        try:
            book = await self._rest.fetch_orderbook(gap.asset_id)
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"book_fetch_failed: {exc}")
            logger.exception(
                "backfill.book_fetch_failed",
                extra={"asset_id": gap.asset_id},
            )
            return
        if book is None:
            report.errors.append("book_fetch_returned_none")
            return
        record = self._synth_book_record(book)
        try:
            await self._sink.write(record)
            report.orderbook_replayed = True
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"book_sink_failed: {exc}")
            logger.exception(
                "backfill.book_sink_failed",
                extra={"asset_id": gap.asset_id},
            )

    # ------------------------------------------------------------------
    # Synthesisers — produce NDJSON-shaped records identical to WS frames
    # ------------------------------------------------------------------

    def _synth_trade_record(self, trade: RestTrade) -> dict[str, Any]:
        return {
            "receive_time": self._clock_ms(),
            "raw": {
                "event_type": "last_trade_price",
                "asset_id": trade.asset_id,
                "market": trade.market,
                "price": str(trade.price),
                "size": str(trade.size),
                "side": trade.side,
                "fee_rate_bps": "0",
                "timestamp": str(trade.match_time_ms),
                "provenance": self._provenance,
            },
        }

    def _synth_book_record(self, book: RestOrderbook) -> dict[str, Any]:
        return {
            "receive_time": self._clock_ms(),
            "raw": {
                "event_type": "book",
                "asset_id": book.asset_id,
                "market": book.market,
                "timestamp": str(book.timestamp_ms),
                "hash": book.hash,
                "bids": [
                    {"price": str(p), "size": str(s)} for p, s in book.bids
                ],
                "asks": [
                    {"price": str(p), "size": str(s)} for p, s in book.asks
                ],
                "provenance": self._provenance,
            },
        }
