"""Fixed-window orchestration for the auditable paper trading data path."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path

from bigan.data.polymarket_clob import MarketSnapshot
from bigan.execution.polymarket_oms import OrderResult, Position, SignalIdentity
from bigan.pipeline.events import (
    DecisionDisposition,
    StrategyDecisionEvent,
    market_snapshot_identity,
)
from bigan.pipeline.strategy_runner import StrategyRunner

from .contracts import (
    PAPER_SCHEMA_VERSION,
    PaperAccountSnapshot,
    PaperDecisionEvent,
    PaperRunManifest,
    PaperSettlementEvent,
    PaperSettlementInput,
    PaperWindowRegistration,
)
from .ledger import PaperAccountLedger
from .storage import PaperRunStore


class PaperSessionFailedError(RuntimeError):
    """Raised after a ledger or durability failure permanently closes a session."""


class PaperTradingSession:
    """Connect one fixed-window StrategyRunner to ledger and durable storage."""

    def __init__(
        self,
        *,
        runner: StrategyRunner,
        ledger: PaperAccountLedger,
        store: PaperRunStore,
        processed_snapshot_ids: Iterable[str] = (),
    ) -> None:
        self.runner = runner
        self.ledger = ledger
        self.store = store
        self.failed = False
        self.failure_reason: str | None = None
        self._feed_callback_registered = False
        self._processed_snapshot_ids = set(processed_snapshot_ids)
        self.runner.on_decision(self._on_decision)
        self._assert_cash_consistency()

    @classmethod
    def create_new(
        cls,
        *,
        runner: StrategyRunner,
        output_dir: str | Path,
        run_id: str,
        source_commit: str,
        config: Mapping[str, object] | None = None,
        created_at: str | None = None,
        fsync: bool = False,
    ) -> PaperTradingSession:
        """Create a new fixed-window session and its immutable manifest."""

        _require_fresh_runner(runner)
        manifest = _manifest_for(
            runner=runner,
            run_id=run_id,
            source_commit=source_commit,
            config=config,
            created_at=created_at or datetime.now(UTC).isoformat(),
        )
        ledger = PaperAccountLedger(
            run_id=run_id,
            initial_bankroll=manifest.initial_bankroll,
            windows=manifest.windows,
        )
        store = PaperRunStore.create_new(
            output_dir=output_dir,
            manifest=manifest,
            fsync=fsync,
        )
        return cls(runner=runner, ledger=ledger, store=store)

    @classmethod
    def resume_existing(
        cls,
        *,
        runner: StrategyRunner,
        output_dir: str | Path,
        run_id: str,
        source_commit: str,
        config: Mapping[str, object] | None = None,
        fsync: bool = False,
    ) -> PaperTradingSession:
        """Replay and verify an existing run before accepting another decision."""

        _require_fresh_runner(runner)
        actual = PaperRunStore.load_manifest(output_dir=output_dir, run_id=run_id)
        expected = _manifest_for(
            runner=runner,
            run_id=run_id,
            source_commit=source_commit,
            config=config,
            created_at=actual.created_at,
        )
        store = PaperRunStore.resume_existing(
            output_dir=output_dir,
            expected_manifest=expected,
            fsync=fsync,
        )
        ledger = store.recover_ledger()
        decisions = store.load_decision_events()
        snapshot = ledger.snapshot()
        oms_positions = tuple(
            Position(
                window_id=position.window_id,
                symbol=position.market_symbol,
                side=position.side,
                shares=position.shares,
                avg_entry_price=position.average_entry_price,
                total_cost_usdc=position.cost_usdc,
            )
            for position in snapshot.positions
        )
        runner.oms.restore_paper_state(
            current_bankroll=snapshot.cash,
            positions=oms_positions,
            order_sequence_floor=snapshot.last_event_sequence,
            processed_signal_identities=tuple(
                identity
                for event in decisions
                if (identity := _filled_signal_identity(event)) is not None
            ),
        )
        runner.current_bankroll = snapshot.cash
        return cls(
            runner=runner,
            ledger=ledger,
            store=store,
            processed_snapshot_ids=(event.source_snapshot_id for event in decisions),
        )

    @property
    def current_snapshot(self) -> PaperAccountSnapshot:
        """Return the current immutable account state."""

        return self.ledger.snapshot()

    async def start(self) -> None:
        """Connect the configured feed through the session's fail-closed path."""

        self._require_healthy()
        if not self._feed_callback_registered:
            self.runner.feed_handler.on_snapshot(self._on_snapshot)
            self._feed_callback_registered = True
        await self.runner.feed_handler.connect()

    async def stop(self) -> None:
        """Close the configured feed."""

        await self.runner.feed_handler.close()

    def process_snapshot_sync(self, snapshot: MarketSnapshot) -> OrderResult | None:
        """Process one snapshot and fail closed if persistence did not complete."""

        self._require_healthy()
        if _snapshot_id(snapshot) in self._processed_snapshot_ids:
            return None
        try:
            result = self.runner.process_snapshot_sync(snapshot)
        except Exception as exc:
            self._fail_after_runner_exception(exc)
            raise
        self._require_healthy()
        self._assert_cash_consistency()
        return result

    async def process_snapshot(self, snapshot: MarketSnapshot) -> OrderResult | None:
        """Async wrapper retaining StrategyRunner's return contract."""

        self._require_healthy()
        if _snapshot_id(snapshot) in self._processed_snapshot_ids:
            return None
        try:
            result = await self.runner.process_snapshot(snapshot)
        except Exception as exc:
            self._fail_after_runner_exception(exc)
            raise
        self._require_healthy()
        self._assert_cash_consistency()
        return result

    async def _on_snapshot(self, snapshot: MarketSnapshot) -> None:
        await self.process_snapshot(snapshot)

    def settle(self, settlement: PaperSettlementInput) -> PaperSettlementEvent:
        """Settle the fixed window and durably persist the resulting account."""

        self._require_healthy()
        sequence = self.ledger.last_event_sequence + 1
        event_id = _event_id(self.store.manifest.run_id, "settlement", sequence)
        try:
            event = self.ledger.settle(
                settlement,
                event_id=event_id,
                event_sequence=sequence,
            )
            if event.event_sequence != sequence:
                return event
            self.runner.oms.close_window(
                settlement.window_id,
                current_bankroll=event.cash_after,
            )
            self.runner.current_bankroll = event.cash_after
            snapshot = self.ledger.snapshot()
            self.store.append_settlement(
                settlement=event,
                ledger_event=self.ledger.settlement_ledger_event(event),
                snapshot=snapshot,
            )
            self._assert_cash_consistency()
            return event
        except Exception as exc:
            self._fail(exc)
            raise

    def _on_decision(self, decision: StrategyDecisionEvent) -> None:
        if self.failed:
            raise PaperSessionFailedError(self.failure_reason or "paper session failed")
        sequence = self.ledger.last_event_sequence + 1
        source_snapshot_id = decision.source_snapshot_id
        if source_snapshot_id in self._processed_snapshot_ids:
            raise ValueError("source snapshot reached decision callback twice")
        event_id = f"{self.store.manifest.run_id}:decision:{source_snapshot_id}"
        paper_event = PaperDecisionEvent(
            schema_version=PAPER_SCHEMA_VERSION,
            run_id=self.store.manifest.run_id,
            event_id=event_id,
            event_sequence=sequence,
            source_snapshot_id=source_snapshot_id,
            decision=decision,
        )
        try:
            ledger_event = self.ledger.apply_decision(paper_event)
            if ledger_event is None:
                raise ValueError("new session decision unexpectedly duplicated")
            self._assert_decision_cash(decision)
            self.store.append_decision(
                decision=paper_event,
                ledger_event=ledger_event,
                snapshot=self.ledger.snapshot(),
            )
            self._processed_snapshot_ids.add(source_snapshot_id)
        except Exception as exc:
            self._fail(exc)
            raise

    def _assert_decision_cash(self, decision: StrategyDecisionEvent) -> None:
        if not math.isclose(self.ledger.cash, decision.cash_after, abs_tol=1e-8):
            raise ValueError("ledger cash differs from decision cash_after")
        self._assert_cash_consistency()

    def _assert_cash_consistency(self) -> None:
        values = (self.ledger.cash, self.runner.current_bankroll, self.runner.oms.bankroll)
        if not math.isclose(values[0], values[1], abs_tol=1e-8) or not math.isclose(
            values[1], values[2], abs_tol=1e-8
        ):
            raise ValueError("ledger, StrategyRunner, and OMS cash are inconsistent")

    def _fail(self, exc: Exception) -> None:
        self.failed = True
        self.failure_reason = f"{type(exc).__name__}: {exc}"

    def _fail_after_runner_exception(self, exc: Exception) -> None:
        try:
            self._assert_cash_consistency()
        except Exception as consistency_error:
            self._fail(consistency_error)
        else:
            self._fail(exc)

    def _require_healthy(self) -> None:
        if self.failed:
            raise PaperSessionFailedError(self.failure_reason or "paper session failed")


def _manifest_for(
    *,
    runner: StrategyRunner,
    run_id: str,
    source_commit: str,
    config: Mapping[str, object] | None,
    created_at: str,
) -> PaperRunManifest:
    if (
        runner.pricing_inputs_provider is not None
        and runner.pricing_inputs_provider_identity is None
    ):
        raise ValueError(
            "paper sessions require pricing_inputs_provider_identity for dynamic inputs"
        )
    window = runner.window
    registration = PaperWindowRegistration(
        window_id=window.window_id,
        market_symbol=window.symbol,
        start_ts_ms=window.start_ts_ms,
        end_ts_ms=window.end_ts_ms,
    )
    identity = {
        "strategy": runner.config_identity(),
        "session_config": dict(config or {}),
    }
    encoded = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return PaperRunManifest(
        schema_version=PAPER_SCHEMA_VERSION,
        run_id=run_id,
        created_at=created_at,
        source_commit=source_commit,
        initial_bankroll=runner.current_bankroll,
        fee_bps=runner.fee_bps,
        market_symbols=(window.symbol,),
        window_ids=(window.window_id,),
        windows=(registration,),
        config_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def _event_id(run_id: str, kind: str, sequence: int) -> str:
    return f"{run_id}:{kind}:{sequence:020d}"


def _snapshot_id(snapshot: MarketSnapshot) -> str:
    return market_snapshot_identity(
        timestamp_ms=snapshot.timestamp_ms,
        window_id=snapshot.window_id,
        yes_bid=snapshot.yes_bid,
        yes_ask=snapshot.yes_ask,
        yes_bid_size=snapshot.yes_bid_size,
        yes_ask_size=snapshot.yes_ask_size,
        no_bid=snapshot.no_bid,
        no_ask=snapshot.no_ask,
        no_bid_size=snapshot.no_bid_size,
        no_ask_size=snapshot.no_ask_size,
        last_traded_price=snapshot.last_traded_price,
    )


def _filled_signal_identity(event: PaperDecisionEvent) -> SignalIdentity | None:
    decision = event.decision
    if decision.disposition is not DecisionDisposition.FILLED:
        return None
    if (
        decision.direction not in {"BUY_YES", "BUY_NO"}
        or decision.market_price is None
        or decision.recommended_size_pct is None
    ):
        raise ValueError("persisted fill is missing its OMS signal identity")
    return (
        decision.window_id,
        decision.timestamp_ms,
        decision.direction,
        decision.market_price,
        decision.recommended_size_pct,
    )


def _require_fresh_runner(runner: StrategyRunner) -> None:
    if (
        runner.decision_count != 0
        or runner.execution_count != 0
        or runner.oms.positions()
        or runner.oms.open_limit_orders()
    ):
        raise ValueError("paper session requires a fresh StrategyRunner/OMS")
