"""Single-session, generation-fenced paper trading operator state machine."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from bigan.data.polymarket_clob import MarketSnapshot, PolymarketFeedHandler
from bigan.execution.polymarket_oms import PolymarketOMS
from bigan.features.binance_ofi import BinanceOFICalculator
from bigan.paper_trading.contracts import PaperSettlementInput
from bigan.paper_trading.session import PaperSessionFailedError, PaperTradingSession
from bigan.paper_trading.storage import MANIFEST_FILE, PaperRunStore
from bigan.pipeline.events import DecisionDisposition
from bigan.pipeline.strategy_runner import StrategyRunner
from bigan.strategies.polymarket_pricing import (
    MarketWindow,
    PolymarketPricingEngine,
)

from .chainlink_twap import oracle_source
from .checkpoint import (
    AccountCheckpoint,
    AccountCheckpointStore,
    fsync_directory,
    load_run_link,
    run_link_store,
)
from .config import OperatorConfig
from .diagnostics import DiagnosticBuffer, DiagnosticCode
from .discovery import DiscoveredMarket, DiscoveryFilters, DiscoverySelection
from .feeds import BinanceDepthSynchronizer, FeedHealth, PolymarketBookSynchronizer
from .market_data import reference_observation
from .ownership import AccountProcessLock
from .pricing_inputs import ReferencePriceSample, RollingPricingInputsProvider
from .read_model import (
    OPERATOR_STATUS_SCHEMA_VERSION,
    OperatorReadRepository,
    OperatorState,
    OperatorStatus,
    OperatorStatusWriter,
    account_totals,
)
from .resolution import FinalResolution

logger = logging.getLogger(__name__)

Clock = Callable[[], int]


class DiscoveryProvider(Protocol):
    async def discover(
        self,
        *,
        filters: DiscoveryFilters,
        now_ms: int,
    ) -> DiscoverySelection: ...


class ResolutionProvider(Protocol):
    async def resolve(
        self,
        market: DiscoveredMarket,
        *,
        now_ms: int,
    ) -> FinalResolution | None: ...


class AuthoritativeReferenceUnavailable(RuntimeError):
    """Market cannot start without an independently proven start reference."""


class PaperTradingOperator:
    """Own exactly one fixed-window paper session and roll it safely."""

    def __init__(
        self,
        *,
        config: OperatorConfig,
        discovery: DiscoveryProvider,
        resolution: ResolutionProvider,
        clock_ms: Clock,
        status_writer: OperatorStatusWriter | None = None,
    ) -> None:
        self.config = config
        self.discovery = discovery
        self.resolution = resolution
        self.clock_ms = clock_ms
        self.process_started_at_ms = int(clock_ms())
        self.state = OperatorState.STARTING
        self.state_reason = "configuration_validated"
        self.active_market: DiscoveredMarket | None = None
        self.next_market: DiscoveredMarket | None = None
        self.session: PaperTradingSession | None = None
        self._session_owner_token: object | None = None
        self.pricing_provider: RollingPricingInputsProvider | None = None
        self.binance_sync: BinanceDepthSynchronizer | None = None
        self.market_sync: PolymarketBookSynchronizer | None = None
        self.generation = 0
        self._lock = asyncio.Lock()
        self._accepting_snapshots = False
        self._permanently_failed = False
        self._projection_error: str | None = None
        self._last_status_write_ms: int | None = None
        self._last_settlement: FinalResolution | None = None
        self._checkpoint: AccountCheckpoint | None = None
        self._account_lock = AccountProcessLock(
            output_dir=Path(config.output_dir), operator_id=config.operator_id,
            account_id=config.paper_account_id,
        )
        self.checkpoint_store = AccountCheckpointStore(
            Path(config.output_dir) / config.operator_id / "account_checkpoint.json"
        )
        self._oracle_connected = False
        self._oracle_connection_generation = 0
        self._oracle_reconnect_count = 0
        self._oracle_diagnostics = DiagnosticBuffer()
        self.counters: dict[str, int] = {
            "decisions": 0,
            "fills": 0,
            "rejects": 0,
            "holds": 0,
            "drops": 0,
            "snapshot_deduplicated": 0,
            "snapshot_generation_dropped": 0,
            "snapshot_freshness_dropped": 0,
            "snapshot_window_dropped": 0,
            "settlement_pending": 0,
            "settlement_completed": 0,
            "rollovers": 0,
            "projection_errors": 0,
            "operator_errors": 0,
        }
        self.status_writer = status_writer or OperatorStatusWriter(
            Path(config.output_dir) / config.operator_id / config.status_filename,
            fsync=config.fsync,
        )

    @property
    def run_id(self) -> str | None:
        return None if self.session is None else self.session.store.manifest.run_id

    @property
    def read_repository(self) -> OperatorReadRepository:
        return OperatorReadRepository(
            status_path=self.status_writer.path,
            run_store=None if self.session is None else self.session.store,
            checkpoint=self._checkpoint,
            default_limit=self.config.recent_query_default,
            max_limit=self.config.recent_query_max,
        )

    async def start(self) -> None:
        """Discover and create/resume a stable window run."""

        async with self._lock:
            self._require_not_failed()
            if self.session is not None:
                return
            # No checkpoint, ledger, index or shared status access before ownership.
            self._account_lock.acquire()
            try:
                self._checkpoint = self._load_account_checkpoint()
                if self._checkpoint is not None:
                    # Recover the durable frontier, not today's discovered market.
                    # poll() will settle/roll this run forward exactly once.
                    self._activate_market(
                        self._checkpoint.market, bankroll=self._checkpoint.opening_cash
                    )
                    return
            except Exception as exc:
                self._fail_permanently("account_checkpoint_recovery_failure", exc)
                return
            self._transition(OperatorState.DISCOVERING, "startup_discovery")
            try:
                selection = await self.discovery.discover(
                    filters=self._filters(),
                    now_ms=self.clock_ms(),
                )
            except Exception as exc:
                self._degrade("discovery_failed", exc)
                return
            market = selection.current or selection.next
            if market is None:
                self._degrade(
                    "discovery_failed",
                    RuntimeError("discovery returned no current or next market"),
                )
                return
            self.next_market = selection.next if selection.current is not None else None
            try:
                self._activate_market(market, bankroll=None)
            except AuthoritativeReferenceUnavailable as exc:
                self._degrade("authoritative_start_reference_unavailable", exc)
            except Exception as exc:
                self._fail_permanently("session_create_or_resume_failure", exc)

    async def ingest_binance_snapshot(
        self,
        payload: dict[str, object],
        *,
        generation: int,
        received_at_ms: int | None = None,
    ) -> bool:
        async with self._lock:
            if not self._can_accept_generation(generation) or self.binance_sync is None:
                return False
            received = self.clock_ms() if received_at_ms is None else int(received_at_ms)
            accepted = self.binance_sync.ingest_snapshot(
                payload,
                generation=generation,
                received_at_ms=received,
            )
            if accepted and self.binance_sync.last_top_changed:
                event_ts = self.binance_sync.last_event_ts_ms
                if event_ts is not None:
                    accepted = self._ingest_spot_from_binance(event_ts)
            self._update_gate_state()
            self._publish_status()
            return accepted

    async def ingest_binance_delta(
        self,
        payload: dict[str, object],
        *,
        generation: int,
        received_at_ms: int | None = None,
    ) -> bool:
        async with self._lock:
            if not self._can_accept_generation(generation) or self.binance_sync is None:
                return False
            received = self.clock_ms() if received_at_ms is None else int(received_at_ms)
            accepted = self.binance_sync.ingest_delta(
                payload,
                generation=generation,
                received_at_ms=received,
                now_ms=self.clock_ms(),
            )
            if accepted and self.binance_sync.last_top_changed:
                event_ts = self.binance_sync.last_event_ts_ms
                accepted = self._ingest_spot_from_binance(received if event_ts is None else event_ts)
            self._update_gate_state()
            self._publish_status()
            return accepted

    async def ingest_oracle(
        self,
        sample: ReferencePriceSample,
        *,
        generation: int,
    ) -> bool:
        async with self._lock:
            if not self._can_accept_generation(generation) or self.pricing_provider is None:
                return False
            accepted = self.pricing_provider.ingest_oracle(sample)
            if accepted:
                self._oracle_connected = True
                self._oracle_connection_generation = max(
                    1, self._oracle_connection_generation
                )
            self._update_gate_state()
            self._publish_status()
            return accepted

    async def ingest_market_message(
        self,
        payload: dict[str, object],
        *,
        generation: int,
        received_at_ms: int | None = None,
    ) -> object | None:
        """Parse one CLOB event and process only a complete, all-fresh snapshot."""

        async with self._lock:
            if not self._can_accept_generation(generation) or self.market_sync is None:
                return None
            snapshot = self.market_sync.ingest(
                payload,
                generation=generation,
                received_at_ms=(self.clock_ms() if received_at_ms is None else received_at_ms),
            )
            if snapshot is None:
                self._update_gate_state()
                self._publish_status()
                return None
            return await self._process_snapshot_locked(snapshot, generation=generation)

    async def begin_binance_connection(
        self,
        *,
        window_generation: int,
        connection_generation: int,
        snapshot: dict[str, object],
        received_at_ms: int,
    ) -> bool:
        """Re-bootstrap a live Binance connection inside the operator fence."""

        async with self._lock:
            if not self._can_accept_generation(window_generation) or self.binance_sync is None:
                return False
            generation = _connection_generation(window_generation, connection_generation)
            self.binance_sync.begin_generation(generation)
            if self.pricing_provider is not None:
                self.pricing_provider.reset_for_reconnect()
            accepted = self.binance_sync.ingest_snapshot(
                snapshot,
                generation=generation,
                received_at_ms=received_at_ms,
            )
            if accepted and self.binance_sync.last_top_changed:
                event_ts = self.binance_sync.last_event_ts_ms
                if event_ts is not None:
                    accepted = self._ingest_spot_from_binance(event_ts)
            self._update_gate_state()
            self._publish_status()
            return accepted

    async def ingest_binance_connection_delta(
        self,
        payload: dict[str, object],
        *,
        window_generation: int,
        connection_generation: int,
        received_at_ms: int,
    ) -> bool:
        async with self._lock:
            if not self._can_accept_generation(window_generation) or self.binance_sync is None:
                return False
            generation = _connection_generation(window_generation, connection_generation)
            accepted = self.binance_sync.ingest_delta(
                payload,
                generation=generation,
                received_at_ms=received_at_ms,
                now_ms=self.clock_ms(),
            )
            if accepted and self.binance_sync.last_top_changed:
                accepted = self._ingest_spot_from_binance(
                    self.binance_sync.last_event_ts_ms or received_at_ms
                )
            self._update_gate_state()
            self._publish_status()
            return accepted

    async def begin_market_connection(
        self,
        *,
        window_generation: int,
        connection_generation: int,
    ) -> None:
        async with self._lock:
            if not self._can_accept_generation(window_generation) or self.market_sync is None:
                return
            self.market_sync.begin_generation(
                _connection_generation(window_generation, connection_generation)
            )
            self._update_gate_state()
            self._publish_status()

    async def ingest_market_connection_payload(
        self,
        payload: dict[str, object],
        *,
        window_generation: int,
        connection_generation: int,
        received_at_ms: int,
    ) -> object | None:
        async with self._lock:
            if not self._can_accept_generation(window_generation) or self.market_sync is None:
                return None
            snapshot = self.market_sync.ingest(
                payload,
                generation=_connection_generation(window_generation, connection_generation),
                received_at_ms=received_at_ms,
            )
            if snapshot is None:
                self._update_gate_state()
                self._publish_status()
                return None
            return await self._process_snapshot_locked(snapshot, generation=window_generation)

    async def begin_oracle_connection(
        self,
        *,
        window_generation: int,
        connection_generation: int = 1,
    ) -> None:
        """Reconnect invalidates TWAP/volatility warm state until re-observed."""

        async with self._lock:
            if not self._can_accept_generation(window_generation):
                return
            if connection_generation <= self._oracle_connection_generation:
                return
            if self._oracle_connection_generation > 0:
                self._oracle_reconnect_count += 1
            self._oracle_connection_generation = connection_generation
            self._oracle_connected = True
            if self.pricing_provider is not None:
                self.pricing_provider.reset_for_reconnect()
            self._update_gate_state()
            self._publish_status()

    async def record_transport_diagnostic(
        self, source: str, code: DiagnosticCode, *, window_generation: int,
        connection_generation: int, timestamp_ms: int,
    ) -> None:
        async with self._lock:
            if not self._can_accept_generation(window_generation):
                return
            if source == "chainlink":
                if connection_generation < self._oracle_connection_generation:
                    return
                target = self._oracle_diagnostics
            elif source in {"binance", "polymarket"}:
                sync = self.binance_sync if source == "binance" else self.market_sync
                if sync is None or _connection_generation(window_generation, connection_generation) < sync.generation:
                    return
                target = sync.diagnostics
            else:
                raise ValueError("unknown diagnostic source")
            target.record(code, timestamp_ms=timestamp_ms, generation=connection_generation)
            self._publish_status()

    async def disconnect_feed(self, source: str, *, window_generation: int) -> None:
        async with self._lock:
            if window_generation != self.generation:
                return
            if source == "binance" and self.binance_sync is not None:
                self.binance_sync.disconnect()
            elif source == "polymarket" and self.market_sync is not None:
                self.market_sync.disconnect()
            elif source == "chainlink":
                self._oracle_connected = False
            self._update_gate_state()
            self._publish_status()

    async def process_snapshot(
        self,
        snapshot: MarketSnapshot,
        *,
        generation: int,
    ) -> object | None:
        """Token-fenced entry used by the read-only Polymarket transport."""

        async with self._lock:
            return await self._process_snapshot_locked(snapshot, generation=generation)

    async def poll(self) -> None:
        """Advance expiry, final settlement, and deterministic rollover."""

        async with self._lock:
            if self._permanently_failed or self.state in {
                OperatorState.STOPPING,
                OperatorState.STOPPED,
                OperatorState.EXHAUSTED,
            }:
                return
            market = self.active_market
            session = self.session
            if market is None or session is None:
                self._transition(OperatorState.DISCOVERING, "no_active_market")
                return
            now_ms = self.clock_ms()
            if now_ms < market.end_ts_ms:
                self._update_gate_state()
                self._publish_status()
                return
            self._accepting_snapshots = False
            self._transition(OperatorState.SETTLEMENT_PENDING, "window_expired_waiting_final_resolution")
            if market.window_id in session.current_snapshot.settled_window_ids:
                try:
                    await self._rollover_locked(session.current_snapshot.cash)
                except AuthoritativeReferenceUnavailable as exc:
                    self._degrade("next_window_reference_unavailable", exc)
                except Exception as exc:
                    self._degrade("rollover_discovery_unavailable", exc)
                return
            self.counters["settlement_pending"] += 1
            try:
                final = await self.resolution.resolve(market, now_ms=now_ms)
                if final is None:
                    self._publish_status()
                    return
                self._validate_final_resolution(final, market)
                session._settle_operator(
                    PaperSettlementInput(
                        window_id=final.window_id,
                        yes_payout=final.yes_payout,
                        settlement_ts_ms=final.settlement_ts_ms,
                        source=final.source,
                        source_ts_ms=final.source_ts_ms,
                        received_ts_ms=final.received_ts_ms,
                        source_reference=final.source_reference,
                    ),
                    owner_token=self._require_session_token(),
                )
                self._last_settlement = final
                self.counters["settlement_completed"] += 1
                self._transition(OperatorState.ROLLING_OVER, "settlement_persisted")
                await self._rollover_locked(session.current_snapshot.cash)
            except Exception as exc:
                if session.failed or isinstance(exc, PaperSessionFailedError):
                    self._fail_permanently("settlement_persistence_failure", exc)
                else:
                    self._degrade("resolution_unavailable_or_invalid", exc)

    async def shutdown(self) -> None:
        """Fence new callbacks, then wait for the in-flight lock holder."""

        self._accepting_snapshots = False
        async with self._lock:
            if not self._account_lock.held:
                self._revoke_session()
                return
            try:
                exhausted = self.state is OperatorState.EXHAUSTED
                if not exhausted:
                    self._transition(OperatorState.STOPPING, "shutdown_requested")
                if self.binance_sync is not None:
                    self.binance_sync.disconnect()
                if self.market_sync is not None:
                    self.market_sync.disconnect()
                if not exhausted:
                    self._transition(OperatorState.STOPPED, "shutdown_complete")
                else:
                    self._publish_status(force=True)
            finally:
                try:
                    self._revoke_session()
                finally:
                    self._account_lock.release()

    def status(self) -> OperatorStatus:
        now_ms = self.clock_ms()
        market = self.active_market
        session = self.session
        snapshot = None if session is None else session.current_snapshot
        binance_health = None if self.binance_sync is None else self.binance_sync.health(now_ms=now_ms)
        market_health = None if self.market_sync is None else self.market_sync.health(now_ms=now_ms)
        binance_health_payload = _health_dict(binance_health)
        binance_health_payload.update(self.config.binance_source_identity())
        if self.binance_sync is not None:
            binance_health_payload.update(
                {
                    "bid_level_count": self.binance_sync.bid_level_count,
                    "ask_level_count": self.binance_sync.ask_level_count,
                    "book_level_limit": self.binance_sync.book_level_limit,
                    "book_overflow_count": self.binance_sync.book_overflow_count,
                    "diagnostics": self.binance_sync.diagnostics.to_dict(),
                }
            )
        market_health_payload = _health_dict(market_health)
        if self.market_sync is not None:
            market_health_payload["tokens"] = self.market_sync.token_health(now_ms=now_ms)
            market_health_payload["diagnostics"] = self.market_sync.diagnostics.to_dict()
        pricing_health = (
            None if self.pricing_provider is None else self.pricing_provider.health(now_ms=now_ms)
        )
        alpha_ts = None if session is None else session.runner.ofi_engine.last_timestamp_ms
        alpha_age = None if alpha_ts is None else now_ms - alpha_ts
        alpha_fresh = bool(
            alpha_age is not None and 0 <= alpha_age <= self.config.max_alpha_age_ms
        )
        oracle_ts = (
            None
            if self.pricing_provider is None
            else self.pricing_provider.last_oracle_timestamp_ms
        )
        oracle_age = None if oracle_ts is None else now_ms - oracle_ts
        oracle_fresh = bool(
            self._oracle_connected
            and oracle_age is not None
            and 0 <= oracle_age <= self.config.max_pricing_age_ms
        )
        last_decision = None
        last_fill = None
        if session is not None:
            if session.runner.last_decision is not None:
                last_decision = session.runner.last_decision.to_dict()
            fills = session.store.recent_fills(limit=1, max_limit=self.config.recent_query_max)
            if fills:
                last_fill = fills[-1].to_dict()
        positions = [] if snapshot is None else [position.to_dict() for position in snapshot.positions]
        settlement_status = "NONE"
        settlement_reference = None
        if market is not None:
            settlement_status = (
                "SETTLED"
                if snapshot is not None and market.window_id in snapshot.settled_window_ids
                else "PENDING"
                if self.state is OperatorState.SETTLEMENT_PENDING
                else "OPEN"
            )
        if (
            market is not None
            and self._last_settlement is not None
            and self._last_settlement.window_id == market.window_id
        ):
            settlement_reference = self._last_settlement.source_reference
        return OperatorStatus(
            schema_version=OPERATOR_STATUS_SCHEMA_VERSION,
            operator_id=self.config.operator_id,
            strategy_id=self.config.strategy_id,
            run_id=self.run_id,
            state=self.state,
            state_reason=self.state_reason,
            process_started_at_ms=self.process_started_at_ms,
            updated_at_ms=max(now_ms, self.process_started_at_ms),
            source_commit=self.config.source_commit,
            paper_only=True,
            safety={
                "capital_at_risk": False,
                "broker_exchange_write_enabled": False,
                "live_exchange_write_enabled": False,
                "polymarket_write_enabled": False,
                "wallet_signing_enabled": False,
            },
            active_market=(
                None
                if market is None
                else {
                    **market.provenance(),
                    "seconds_to_end": max(0, market.end_ts_ms - now_ms) // 1_000,
                }
            ),
            feeds={
                "binance": binance_health_payload,
                "polymarket": market_health_payload,
                "chainlink": {
                    "state": "READY" if oracle_fresh else "STALE" if self._oracle_connected else "DISCONNECTED",
                    "connected": self._oracle_connected,
                    "synchronized": oracle_ts is not None,
                    "fresh": oracle_fresh,
                    "last_event_ts_ms": oracle_ts,
                    "age_ms": oracle_age,
                    "last_message_received_ms": oracle_ts,
                    "gap_count": 0,
                    "reconnect_count": self._oracle_reconnect_count,
                    "error_count": 0,
                    "diagnostics": self._oracle_diagnostics.to_dict(),
                },
            },
            pricing_inputs=(
                _unavailable_pricing_health()
                if pricing_health is None or self.pricing_provider is None
                else {**pricing_health.to_dict(), "diagnostics": self.pricing_provider.diagnostics.to_dict()}
            ),
            alpha={
                "venue": self.config.binance_venue,
                "source": self._spot_source,
                "symbol": self.config.binance_symbol,
                "timestamp_ms": alpha_ts,
                "age_ms": alpha_age,
                "fresh": alpha_fresh,
                "z_score": 0.0 if session is None else session.runner.ofi_engine.get_normalized_ofi(),
            },
            session={
                "healthy": session is not None and not session.failed,
                "failure_reason": (
                    self._projection_error
                    if session is None
                    else session.failure_reason or self._projection_error
                ),
            },
            account={
                **account_totals(snapshot, self._checkpoint, self.config.initial_bankroll),
                "open_positions": positions,
            },
            counters=dict(self.counters),
            last_decision=last_decision,
            last_fill=last_fill,
            settlement={
                "status": settlement_status,
                "source_reference": settlement_reference,
            },
            market_data=self._market_data(now_ms=now_ms),
        )

    def _market_data(self, *, now_ms: int) -> dict[str, object]:
        """Project accepted feed data even when another feed blocks decisions."""
        market, provider = self.active_market, self.pricing_provider
        if market is None:
            return {}
        active = self.state not in {
            OperatorState.STOPPING, OperatorState.STOPPED, OperatorState.FAILED,
            OperatorState.EXHAUSTED,
        }
        quotes = {} if self.market_sync is None else self.market_sync.quote_observations(now_ms=now_ms)
        if not active:
            for quote in quotes.values():
                quote.update(connected=False, fresh=False)
        lookback = market.oracle_twap_lookback_seconds
        spot = reference_observation(
            None if provider is None else provider.last_spot_sample,
            source=self._spot_source, symbol=self.config.binance_symbol,
            kind="midpoint", currency="USDT", now_ms=now_ms,
            max_age_ms=self.config.max_pricing_age_ms,
            connected=bool(active and self.binance_sync is not None and self.binance_sync.connected),
        )
        spot["fresh"] = bool(
            spot["fresh"] and self.binance_sync is not None
            and self.binance_sync.health(now_ms=now_ms).fresh
        )
        spot["venue"] = self.config.binance_venue
        return {
            "window_id": market.window_id,
            "market_id": market.market_id,
            "underlying": self.config.underlying,
            "spot": spot,
            "oracle": {
                **reference_observation(
                    None if provider is None else provider.last_oracle_sample,
                    source=self._oracle_source, symbol=self.config.chainlink_symbol,
                    kind="published_twap" if lookback is not None else "oracle_price",
                    currency="USD", now_ms=now_ms,
                    max_age_ms=self.config.max_pricing_age_ms,
                    connected=active and self._oracle_connected,
                ),
                "lookback_seconds": lookback,
            },
            "up": quotes.get("yes", {}),
            "down": quotes.get("no", {}),
        }

    async def _process_snapshot_locked(
        self,
        snapshot: MarketSnapshot,
        *,
        generation: int,
    ) -> object | None:
        if not self._can_accept_generation(generation):
            return None
        market = self.active_market
        session = self.session
        if market is None or session is None:
            return None
        now_ms = self.clock_ms()
        if (
            snapshot.window_id != market.window_id
            or snapshot.timestamp_ms < market.start_ts_ms
            or now_ms < market.start_ts_ms
        ):
            self.counters["snapshot_window_dropped"] += 1
            self._transition(OperatorState.SYNCING, "snapshot_outside_active_window")
            return None
        if now_ms >= market.end_ts_ms or snapshot.timestamp_ms >= market.end_ts_ms:
            self._accepting_snapshots = False
            self.counters["snapshot_window_dropped"] += 1
            self._transition(OperatorState.SETTLEMENT_PENDING, "window_expired")
            return None
        if not self._all_inputs_fresh(now_ms):
            self.counters["snapshot_freshness_dropped"] += 1
            self._transition(OperatorState.SYNCING, "freshness_gate_closed")
            return None
        # Publishing the transition can itself fail and leave us DEGRADED.
        # Feed ingestion remains enabled for warmup/recovery, execution does not.
        self._update_gate_state()
        if self.state is not OperatorState.RUNNING or self._projection_error is not None:
            return None
        before = session.runner.decision_count
        try:
            result = await session._process_operator_snapshot(
                snapshot, owner_token=self._require_session_token()
            )
            if session.runner.decision_count == before:
                self.counters["snapshot_deduplicated"] += 1
            else:
                self._record_last_disposition(session)
            self._transition(OperatorState.RUNNING, "all_sources_fresh")
            return result
        except Exception as exc:
            self._fail_permanently("paper_session_failure", exc)
            raise

    async def _rollover_locked(self, bankroll: float) -> None:
        if bankroll == 0:
            self._accepting_snapshots = False
            self._transition(OperatorState.EXHAUSTED, "settled_account_has_no_capital")
            return
        self._transition(OperatorState.ROLLING_OVER, "discovering_next_window")
        try:
            selection = await self.discovery.discover(
                filters=self._filters(),
                now_ms=self.clock_ms(),
            )
        except Exception as exc:
            self._degrade("rollover_discovery_unavailable", exc)
            return
        market = selection.current or selection.next
        if market is None or (
            self.active_market is not None and market.window_id == self.active_market.window_id
        ):
            self._transition(OperatorState.SETTLEMENT_PENDING, "next_window_not_yet_discoverable")
            return
        self.next_market = selection.next if selection.current is not None else None
        try:
            self._activate_market(market, bankroll=bankroll)
        except AuthoritativeReferenceUnavailable as exc:
            self._degrade("next_window_reference_unavailable", exc)
            return
        except Exception as exc:
            self._fail_permanently("rollover_session_create_or_resume_failure", exc)
            return
        self.counters["rollovers"] += 1
        self._publish_status(force=True)

    def _activate_market(self, market: DiscoveredMarket, *, bankroll: float | None) -> None:
        if market.reference_price_at_start is None or (
            market.oracle_twap_lookback_seconds is not None and market.opening_reference is None
        ):
            raise AuthoritativeReferenceUnavailable(
                "market discovery lacks authoritative reference price at start"
            )
        if (
            market.underlying != self.config.underlying
            or market.market_type != self.config.market_type
            or market.window_duration_ms != self.config.window_duration_ms
            or market.end_ts_ms - market.start_ts_ms != self.config.window_duration_ms
        ):
            raise ValueError("active market does not match configured asset/window identity")
        self.generation += 1
        generation = self.generation
        run_id = stable_run_id(
            strategy_id=self.config.strategy_id,
            market_id=market.market_id,
            window_id=market.window_id,
            paper_account_id=self.config.paper_account_id,
        )
        run_path = Path(self.config.output_dir) / run_id
        starting_bankroll = self.config.initial_bankroll if bankroll is None else bankroll
        checkpoint = self._checkpoint
        if checkpoint is not None and checkpoint.market.window_id == market.window_id:
            if checkpoint.run_id != run_id or checkpoint.opening_cash != starting_bankroll:
                raise ValueError("account checkpoint run identity or opening cash mismatch")
        else:
            previous = self.session
            if checkpoint is not None and previous is None:
                raise ValueError("cannot advance account without recovering its prior run")
            if previous is not None and (
                self.active_market is None
                or self.active_market.window_id not in previous.current_snapshot.settled_window_ids
                or previous.current_snapshot.cash != starting_bankroll
                or market.start_ts_ms < self.active_market.end_ts_ms
            ):
                raise ValueError("successor requires a settled predecessor with matching cash")
            checkpoint = AccountCheckpoint(
                config_sha256=self.config.config_sha256, run_id=run_id,
                market=market, opening_cash=starting_bankroll,
                initial_bankroll=self.config.initial_bankroll if checkpoint is None else checkpoint.initial_bankroll,
                run_index=0 if checkpoint is None else checkpoint.run_index + 1,
                prior_realized_pnl=(
                    0.0 if previous is None or checkpoint is None
                    else checkpoint.prior_realized_pnl + previous.current_snapshot.realized_pnl
                ),
                prior_fees=(
                    0.0 if previous is None or checkpoint is None
                    else checkpoint.prior_fees + previous.current_snapshot.commission_paid
                ),
                predecessor_run_id=None if previous is None else self.run_id,
                predecessor_window_id=None if previous is None else previous.runner.window.window_id,
                predecessor_settled_cash=None if previous is None else previous.current_snapshot.cash,
            )
            # Durable activation intent precedes every create/resume side effect.
            self.checkpoint_store.write(checkpoint)
        if checkpoint.activation_state == "ACTIVE":
            if not run_path.is_dir():
                raise ValueError("ACTIVE account run directory is missing")
            if load_run_link(Path(self.config.output_dir), run_id, checkpoint.config_sha256) != checkpoint:
                raise ValueError("active account checkpoint disagrees with its run link")
        if run_path.is_dir():
            manifest_bankroll = PaperRunStore.load_manifest(
                output_dir=self.config.output_dir,
                run_id=run_id,
            ).initial_bankroll
            if manifest_bankroll != starting_bankroll:
                raise ValueError("run opening cash disagrees with account checkpoint")
        runner, pricing, binance, market_sync = self._build_runner(
            market,
            bankroll=starting_bankroll,
        )
        session_config = self._session_config(market)
        if run_path.is_dir():
            session = PaperTradingSession.resume_existing(
                runner=runner,
                output_dir=self.config.output_dir,
                run_id=run_id,
                source_commit=self.config.source_commit,
                config=session_config,
                fsync=True,
                snapshot_dedupe_cache_size=self.config.snapshot_lru_size,
            )
            lifecycle_reason = "session_resumed"
        else:
            session = PaperTradingSession.create_new(
                runner=runner,
                output_dir=self.config.output_dir,
                run_id=run_id,
                source_commit=self.config.source_commit,
                config=session_config,
                fsync=True,
                snapshot_dedupe_cache_size=self.config.snapshot_lru_size,
            )
            lifecycle_reason = "session_created"
        owner_token = object()
        session._bind_operator(
            owner_token=owner_token,
            ownership_checker=lambda: self._owns_session_writes(owner_token),
        )
        if checkpoint.activation_state == "ACTIVATING":
            if session.current_snapshot.last_event_sequence != 0:
                raise ValueError("ACTIVATING run cannot contain trading or settlement events")
            active = replace(checkpoint, activation_state="ACTIVE")
            links = run_link_store(Path(self.config.output_dir), run_id)
            existing = links.load(config_sha256=checkpoint.config_sha256)
            if existing is not None and existing != active:
                raise ValueError("conflicting account run link")
            if existing is None:
                links.write(active)
            # Persist the new directory entry before publishing ACTIVE.
            fsync_directory(run_path)
            fsync_directory(Path(self.config.output_dir))
            self.checkpoint_store.write(active)
            checkpoint = active
        self._checkpoint = checkpoint
        self._revoke_session()
        self._session_owner_token = owner_token
        self.session = session
        self.active_market = market
        self.pricing_provider = pricing
        self._oracle_diagnostics = DiagnosticBuffer()
        self.binance_sync = binance
        self.market_sync = market_sync
        self._oracle_connected = False
        self._oracle_connection_generation = 0
        self._oracle_reconnect_count = 0
        self.binance_sync.begin_generation(generation)
        self.market_sync.begin_generation(generation)
        self._accepting_snapshots = market.window_id not in session.current_snapshot.settled_window_ids
        self._initialize_run_counters(session)
        if not self._accepting_snapshots and session.current_snapshot.cash == 0:
            self._transition(OperatorState.EXHAUSTED, "settled_account_has_no_capital")
        elif self.clock_ms() >= market.end_ts_ms:
            self._accepting_snapshots = False
            self._transition(
                OperatorState.SETTLEMENT_PENDING,
                f"{lifecycle_reason}_expired",
            )
        else:
            self._transition(OperatorState.SYNCING, lifecycle_reason)

    def _build_runner(
        self,
        market: DiscoveredMarket,
        *,
        bankroll: float,
    ) -> tuple[
        StrategyRunner,
        RollingPricingInputsProvider,
        BinanceDepthSynchronizer,
        PolymarketBookSynchronizer,
    ]:
        reference_price = market.reference_price_at_start
        if reference_price is None:
            raise AuthoritativeReferenceUnavailable(
                "market discovery lacks authoritative reference price at start"
            )
        ofi = BinanceOFICalculator(
            ema_alpha=self.config.ofi_ema_alpha,
            window_ms=self.config.ofi_window_ms,
            zscore_min_samples=self.config.ofi_min_samples,
            zscore_clip=self.config.ofi_clip,
            max_events_cap=self.config.ofi_max_events,
            symbol=self.config.binance_symbol,
            venue=self.config.binance_venue,
        )
        pricing_provider = RollingPricingInputsProvider(
            window_start_ts_ms=market.start_ts_ms,
            window_end_ts_ms=market.end_ts_ms,
            spot_source=self._spot_source,
            oracle_source=oracle_source(self.config.chainlink_symbol, market.oracle_twap_lookback_seconds),
            max_age_ms=self.config.max_pricing_age_ms,
            max_samples=self.config.pricing_sample_buffer_size,
            twap_window_ms=self.config.twap_window_ms,
            return_interval_ms=self.config.volatility_return_interval_ms,
            volatility_window_ms=self.config.volatility_window_ms,
            volatility_min_samples=self.config.volatility_min_samples,
            volatility_max_abs_log_return=self.config.volatility_max_abs_log_return,
            annualization_seconds=self.config.annualization_seconds,
            oracle_twap_lookback_seconds=market.oracle_twap_lookback_seconds,
        )
        provider_identity = hashlib.sha256(
            json.dumps(
                pricing_provider.config_identity(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        window_type = {
            300_000: "5m",
            900_000: "15m",
        }.get(market.window_duration_ms)
        if window_type is None:
            raise ValueError("paper operator supports only 5m and 15m pricing windows")
        window = MarketWindow(
            window_id=market.window_id,
            symbol=self.config.underlying,
            strike_price=reference_price,
            start_ts_ms=market.start_ts_ms,
            end_ts_ms=market.end_ts_ms,
            window_type=window_type,
        )
        feed_handler = PolymarketFeedHandler(
            window_id=market.window_id,
            yes_token_id=market.yes_token_id,
            no_token_id=market.no_token_id,
            ws_url=self.config.polymarket_ws_url,
            mock=True,
            reconnect_min_seconds=self.config.reconnect_min_seconds,
            reconnect_max_seconds=self.config.reconnect_max_seconds,
            heartbeat_interval_seconds=self.config.heartbeat_interval_seconds,
            max_quote_age_ms=self.config.max_market_age_ms,
        )
        oms = PolymarketOMS(
            max_single_trade_pct=self.config.max_single_trade_pct,
            max_position_pct=self.config.max_position_pct,
            max_window_exposure_pct=self.config.max_window_exposure_pct,
            min_order_usd=self.config.min_order_usd,
            max_spread_allowed=self.config.max_spread_allowed,
            slippage_tolerance=self.config.slippage_tolerance,
            symbol=self.config.underlying,
            signal_cache_size=self.config.oms_signal_cache_size,
        )
        runner = StrategyRunner(
            ofi_engine=ofi,
            pricing_engine=PolymarketPricingEngine(
                ofi_gamma=self.config.pricing_ofi_gamma,
                min_edge_5m=self.config.pricing_min_edge_5m,
                min_edge_15m=self.config.pricing_min_edge_15m,
                kelly_fraction=self.config.pricing_kelly_fraction,
                tail_cutoff_ms=self.config.pricing_tail_cutoff_ms,
                reference_model="window_average" if market.oracle_twap_lookback_seconds is None else "published_twap",
            ),
            oms=oms,
            feed_handler=feed_handler,
            window=window,
            initial_bankroll=bankroll,
            spot_price=reference_price,
            oracle_twap_so_far=reference_price,
            pricing_inputs_provider=pricing_provider,
            pricing_inputs_provider_identity=provider_identity,
            reference_max_age_ms=self.config.max_pricing_age_ms,
            ofi_max_age_ms=self.config.max_alpha_age_ms,
            fee_bps=self.config.fee_bps,
            execution_history_limit=self.config.execution_history_limit,
        )
        binance = BinanceDepthSynchronizer(
            calculator=ofi,
            symbol=self.config.binance_symbol,
            max_age_ms=self.config.max_alpha_age_ms,
            delta_buffer_size=self.config.binance_delta_buffer_size,
            book_level_limit=self.config.binance_book_level_limit,
            clock_ahead_tolerance_ms=self.config.binance_clock_ahead_tolerance_ms,
        )
        market_sync = PolymarketBookSynchronizer(
            window_id=market.window_id,
            yes_token_id=market.yes_token_id,
            no_token_id=market.no_token_id,
            max_age_ms=self.config.max_market_age_ms,
            condition_id=market.condition_id,
        )
        return runner, pricing_provider, binance, market_sync

    @property
    def _spot_source(self) -> str:
        return self.config.binance_spot_source

    @property
    def _oracle_source(self) -> str:
        return oracle_source(self.config.chainlink_symbol, None if self.active_market is None
                             else self.active_market.oracle_twap_lookback_seconds)

    def _load_account_checkpoint(self) -> AccountCheckpoint | None:
        checkpoint = self.checkpoint_store.load(config_sha256=self.config.config_sha256)
        if checkpoint is None:
            if self._has_existing_run_data():
                raise ValueError(
                    "existing paper runs without account checkpoint; explicit migration required"
                )
            return None
        if checkpoint.predecessor_run_id is not None:
            previous = load_run_link(
                Path(self.config.output_dir), checkpoint.predecessor_run_id, checkpoint.config_sha256
            )
            manifest = PaperRunStore.load_manifest(
                output_dir=self.config.output_dir, run_id=checkpoint.predecessor_run_id
            )
            store = PaperRunStore.resume_existing(
                output_dir=self.config.output_dir, expected_manifest=manifest, fsync=True
            )
            settled = store.recover_ledger().snapshot()
            if (
                checkpoint.predecessor_window_id not in settled.settled_window_ids
                or settled.cash != checkpoint.opening_cash
                or checkpoint.run_index != previous.run_index + 1
                or checkpoint.initial_bankroll != previous.initial_bankroll
                or checkpoint.prior_realized_pnl != previous.prior_realized_pnl + settled.realized_pnl
                or checkpoint.prior_fees != previous.prior_fees + settled.commission_paid
            ):
                raise ValueError("checkpoint predecessor ledger is not settled at carried cash")
        return checkpoint

    def _has_existing_run_data(self) -> bool:
        root = Path(self.config.output_dir)
        for path in root.iterdir():
            if path.is_dir() and (path / MANIFEST_FILE).exists():
                manifest = PaperRunStore.load_manifest(output_dir=root, run_id=path.name)
                if manifest.run_id != path.name:
                    raise ValueError("existing run directory and manifest identity disagree")
                return True
            # A canonical run without a manifest is incomplete, not an empty account.
            if re.fullmatch(r"paper-[0-9a-f]{24}", path.name):
                raise ValueError("incomplete paper run without an account checkpoint")
        return False

    def _owns_session_writes(self, owner_token: object) -> bool:
        return bool(
            self._account_lock.held
            and self._session_owner_token is owner_token
            and not self._permanently_failed
        )

    def _require_session_token(self) -> object:
        token = self._session_owner_token
        if token is None or not self._owns_session_writes(token):
            raise RuntimeError("operator has no current session write ownership")
        return token

    def _revoke_session(self) -> None:
        token = self._session_owner_token
        self._session_owner_token = None
        if self.session is not None and token is not None:
            self.session._close_operator(owner_token=token)

    def _ingest_spot_from_binance(self, timestamp_ms: int) -> bool:
        if self.binance_sync is None or self.pricing_provider is None:
            return False
        mid = self.binance_sync.mid_price
        if mid is None:
            return False
        accepted = self.pricing_provider.ingest_spot(
            ReferencePriceSample(
                timestamp_ms=timestamp_ms,
                received_at_ms=max(timestamp_ms, self.clock_ms()),
                price=mid,
                source=self._spot_source,
            )
        )
        if not accepted:
            # Never combine a newly accepted alpha with a rejected reference.
            # False also asks the live transport to reconnect/bootstrap.
            self.binance_sync.disconnect()
            self.pricing_provider.reset_for_reconnect()
        return accepted

    def _all_inputs_fresh(self, now_ms: int) -> bool:
        if (
            self.session is None
            or self.binance_sync is None
            or self.market_sync is None
            or self.pricing_provider is None
        ):
            return False
        alpha_ts_ms = self.session.runner.ofi_engine.last_timestamp_ms
        alpha_fresh = bool(
            alpha_ts_ms is not None
            and 0 <= now_ms - alpha_ts_ms <= self.config.max_alpha_age_ms
        )
        return bool(
            self._oracle_connected
            and alpha_fresh
            and self.binance_sync.health(now_ms=now_ms).fresh
            and self.market_sync.health(now_ms=now_ms).fresh
            and self.pricing_provider.health(now_ms=now_ms).fresh
        )

    def _update_gate_state(self) -> None:
        if self._permanently_failed or self.state in {
            OperatorState.SETTLEMENT_PENDING,
            OperatorState.STOPPING,
            OperatorState.STOPPED,
            OperatorState.EXHAUSTED,
        }:
            return
        if self.active_market is None:
            self._transition(OperatorState.DISCOVERING, "no_active_market")
        elif self.clock_ms() < self.active_market.start_ts_ms:
            self._transition(OperatorState.SYNCING, "window_preopen")
        elif self._all_inputs_fresh(self.clock_ms()):
            self._transition(OperatorState.RUNNING, "all_sources_fresh")
        else:
            self._transition(OperatorState.SYNCING, "freshness_gate_closed")

    def _can_accept_generation(self, generation: int) -> bool:
        accepted = bool(
            self._accepting_snapshots
            and self._account_lock.held
            and not self._permanently_failed
            and self.state not in {OperatorState.STOPPING, OperatorState.STOPPED}
            and generation == self.generation
        )
        if generation != self.generation:
            self.counters["snapshot_generation_dropped"] += 1
        return accepted

    def _filters(self) -> DiscoveryFilters:
        return DiscoveryFilters(
            underlying=self.config.underlying,
            market_type=self.config.market_type,
            window_duration_ms=self.config.window_duration_ms,
            slug_pattern=self.config.slug_pattern,
            title_pattern=self.config.title_pattern,
            max_preopen_ms=self.config.max_preopen_ms,
        )

    def _session_config(self, market: DiscoveredMarket) -> dict[str, object]:
        return {
            "operator_config": self.config.config_identity(),
            "market_identity": {
                "market_id": market.market_id,
                "condition_id": market.condition_id,
                "window_id": market.window_id,
                "yes_token_id": market.yes_token_id,
                "no_token_id": market.no_token_id,
                "slug": market.slug,
                "resolution_source": market.resolution_source,
                "resolution_identity": market.resolution_identity,
                "reference_price_at_start": market.reference_price_at_start,
                "oracle_twap_lookback_seconds": market.oracle_twap_lookback_seconds,
                "opening_reference": market.provenance()["opening_reference"],
            },
        }

    def _initialize_run_counters(self, session: PaperTradingSession) -> None:
        for key in ("decisions", "fills", "rejects", "holds", "drops"):
            self.counters[key] = 0
        for event in session.store.load_decision_events():
            self._increment_disposition(event.decision.disposition)

    def _record_last_disposition(self, session: PaperTradingSession) -> None:
        decision = session.runner.last_decision
        if decision is not None:
            self._increment_disposition(decision.disposition)

    def _increment_disposition(self, disposition: DecisionDisposition) -> None:
        self.counters["decisions"] += 1
        key = {
            DecisionDisposition.FILLED: "fills",
            DecisionDisposition.REJECTED: "rejects",
            DecisionDisposition.HOLD: "holds",
            DecisionDisposition.DROPPED: "drops",
            DecisionDisposition.NO_ORDER: "drops",
        }[disposition]
        self.counters[key] += 1

    def _validate_final_resolution(
        self,
        final: FinalResolution,
        market: DiscoveredMarket,
    ) -> None:
        if (
            final.market_id != market.market_id
            or final.condition_id != market.condition_id
            or final.window_id != market.window_id
            or final.resolution_identity != market.resolution_identity
            or final.yes_payout not in {0.0, 1.0}
        ):
            raise ValueError("final resolution identity or payout mismatch")

    def _transition(self, state: OperatorState, reason: str) -> None:
        if self._permanently_failed and state is not OperatorState.FAILED:
            return
        changed = state is not self.state or reason != self.state_reason
        self.state = state
        self.state_reason = reason
        if changed:
            logger.info(
                "paper_operator.state_transition",
                extra={"state": state.value, "reason": reason, "run_id": self.run_id},
            )
        self._publish_status(force=changed)

    def _degrade(self, reason: str, exc: Exception) -> None:
        self.counters["operator_errors"] += 1
        self.state = OperatorState.DEGRADED
        self.state_reason = f"{reason}:{type(exc).__name__}"
        logger.warning(
            "paper_operator.degraded",
            extra={"reason": reason, "error_type": type(exc).__name__},
        )
        self._publish_status(force=True)

    def _fail_permanently(self, reason: str, exc: Exception) -> None:
        self.counters["operator_errors"] += 1
        self._permanently_failed = True
        self._accepting_snapshots = False
        self._revoke_session()
        self.state = OperatorState.FAILED
        self.state_reason = f"{reason}:{type(exc).__name__}"
        logger.exception("paper_operator.failed", extra={"reason": reason})
        self._publish_status(force=True)

    def _publish_status(self, *, force: bool = False) -> None:
        if not self._account_lock.held:
            return
        now_ms = self.clock_ms()
        if (
            not force
            and self._last_status_write_ms is not None
            and now_ms - self._last_status_write_ms < self.config.status_interval_ms
        ):
            return
        self._last_status_write_ms = now_ms
        try:
            self.status_writer.write(self.status())
            self._projection_error = None
        except Exception as exc:
            self.counters["projection_errors"] += 1
            self._projection_error = f"{type(exc).__name__}: {exc}"
            if not self._permanently_failed and self.state is not OperatorState.EXHAUSTED:
                self.state = OperatorState.DEGRADED
                self.state_reason = "status_projection_write_failed"
            logger.error(
                "paper_operator.status_projection_failed",
                extra={"error_type": type(exc).__name__},
            )

    def _require_not_failed(self) -> None:
        if self._permanently_failed:
            raise RuntimeError("paper operator is permanently failed")


def stable_run_id(
    *,
    strategy_id: str,
    market_id: str,
    window_id: str,
    paper_account_id: str,
) -> str:
    identity = {
        "strategy_id": strategy_id,
        "market_id": market_id,
        "window_id": window_id,
        "paper_account_id": paper_account_id,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"paper-{digest[:24]}"


def _health_dict(health: FeedHealth | None) -> dict[str, object]:
    if health is None:
        return _unavailable_health()
    return health.to_dict()


def _unavailable_health() -> dict[str, object]:
    return {
        "state": "DISCONNECTED",
        "connected": False,
        "synchronized": False,
        "fresh": False,
        "last_event_ts_ms": None,
        "age_ms": None,
        "last_message_received_ms": None,
        "gap_count": 0,
        "reconnect_count": 0,
        "error_count": 0,
    }


def _unavailable_pricing_health() -> dict[str, object]:
    return {
        "ready": False,
        "fresh": False,
        "timestamp_ms": None,
        "age_ms": None,
        "spot_sample_count": 0,
        "oracle_sample_count": 0,
        "return_sample_count": 0,
    }


def _connection_generation(window_generation: int, connection_generation: int) -> int:
    if window_generation < 1 or connection_generation < 1:
        raise ValueError("window and connection generations must be positive")
    return window_generation * 1_000_000 + connection_generation
