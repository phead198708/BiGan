"""Deterministic in-memory accounting and replay for paper trading."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping

from bigan.pipeline.events import DecisionDisposition, DecisionReason

from .contracts import (
    PAPER_SCHEMA_VERSION,
    LedgerEventKind,
    PaperAccountSnapshot,
    PaperDecisionEvent,
    PaperLedgerEvent,
    PaperLot,
    PaperPosition,
    PaperSettlementEvent,
    PaperSettlementInput,
    PaperWindowRegistration,
)


class PaperAccountLedger:
    """Pure deterministic BUY-only ledger with explicit settlement truth."""

    def __init__(
        self,
        *,
        run_id: str,
        initial_bankroll: float,
        windows: Iterable[PaperWindowRegistration],
    ) -> None:
        bankroll = float(initial_bankroll)
        if not run_id.strip():
            raise ValueError("run_id must be non-empty")
        if not math.isfinite(bankroll) or bankroll <= 0.0:
            raise ValueError("initial_bankroll must be positive and finite")
        registrations = tuple(windows)
        if not registrations:
            raise ValueError("at least one window must be registered")
        self.run_id = run_id
        self.initial_bankroll = bankroll
        self._windows = {row.window_id: row for row in registrations}
        if len(self._windows) != len(registrations):
            raise ValueError("window ids must be unique")
        self.cash = bankroll
        self.realized_pnl = 0.0
        self.commission_paid = 0.0
        self.equity = bankroll
        self.peak_equity = bankroll
        self.drawdown = 0.0
        self.last_event_sequence = 0
        self.timestamp_ms = 0
        self._lots: list[PaperLot] = []
        self._marks: dict[tuple[str, str], float] = {}
        self._event_fingerprints: dict[str, str] = {}
        self._settlements: dict[str, PaperSettlementEvent] = {}

    def apply_decision(self, event: PaperDecisionEvent) -> PaperLedgerEvent | None:
        """Apply one decision exactly once and return its account observation."""

        fingerprint = _fingerprint(event.to_dict())
        duplicate = self._duplicate_result(event.event_id, fingerprint)
        if duplicate:
            return None
        self._validate_new_identity(event.run_id, event.event_sequence)
        decision = event.decision
        window = self._windows.get(decision.window_id)
        if window is None:
            if not (
                decision.disposition is DecisionDisposition.DROPPED
                and decision.reason_code is DecisionReason.WINDOW_MISMATCH
            ):
                raise ValueError(f"unregistered window: {decision.window_id}")
        else:
            if decision.window_id in self._settlements:
                raise ValueError("cannot apply a decision after window settlement")
            if (
                decision.market_symbol != window.market_symbol
                or decision.window_start_ts_ms != window.start_ts_ms
                or decision.window_end_ts_ms != window.end_ts_ms
            ):
                raise ValueError("decision window metadata does not match registration")
        _assert_close("decision cash_before", decision.cash_before, self.cash)

        if window is not None:
            self._marks[(decision.window_id, "YES")] = _bid("yes_bid", decision.yes_bid)
            self._marks[(decision.window_id, "NO")] = _bid("no_bid", decision.no_bid)
        if decision.disposition is DecisionDisposition.FILLED:
            self._apply_fill(event)
        else:
            _assert_close("non-fill cash_after", decision.cash_after, self.cash)

        self.last_event_sequence = event.event_sequence
        self.timestamp_ms = decision.timestamp_ms
        self._event_fingerprints[event.event_id] = fingerprint
        self._revalue()
        return self._ledger_event(
            event_id=f"{event.event_id}:ledger",
            source_event_id=event.event_id,
            kind=LedgerEventKind.DECISION,
            sequence=event.event_sequence,
            timestamp_ms=decision.timestamp_ms,
            window_id=decision.window_id,
        )

    def mark_to_market(
        self,
        *,
        window_id: str,
        yes_bid: float,
        no_bid: float,
        timestamp_ms: int,
    ) -> PaperAccountSnapshot:
        """Update executable bid marks without creating a synthetic trade event."""

        if window_id not in self._windows:
            raise ValueError(f"unregistered window: {window_id}")
        if timestamp_ms < self.timestamp_ms:
            raise ValueError("mark timestamp cannot move backwards")
        valid_yes_bid = _bid("yes_bid", yes_bid)
        valid_no_bid = _bid("no_bid", no_bid)
        self._marks[(window_id, "YES")] = valid_yes_bid
        self._marks[(window_id, "NO")] = valid_no_bid
        self.timestamp_ms = timestamp_ms
        self._revalue()
        return self.snapshot()

    def settle(
        self,
        settlement: PaperSettlementInput,
        *,
        event_id: str,
        event_sequence: int,
    ) -> PaperSettlementEvent:
        """Settle one registered window once using explicit external truth."""

        window = self._windows.get(settlement.window_id)
        if window is None:
            raise ValueError(f"unregistered window: {settlement.window_id}")
        if settlement.settlement_ts_ms < window.end_ts_ms:
            raise ValueError("settlement cannot occur before window expiry")
        existing = self._settlements.get(settlement.window_id)
        if existing is not None:
            if existing.settlement == settlement:
                return existing
            raise ValueError("conflicting settlement for window")
        identity_payload = {
            "run_id": self.run_id,
            "event_id": event_id,
            "event_sequence": event_sequence,
            "settlement": settlement.to_dict(),
        }
        fingerprint = _fingerprint(identity_payload)
        if self._duplicate_result(event_id, fingerprint):
            raise ValueError("settlement event id exists without settled window")
        self._validate_new_identity(self.run_id, event_sequence)

        window_lots = [lot for lot in self._lots if lot.window_id == settlement.window_id]
        no_payout = 1.0 - settlement.yes_payout
        proceeds = sum(
            lot.shares * (settlement.yes_payout if lot.side == "YES" else no_payout)
            for lot in window_lots
        )
        realized_delta = sum(
            lot.shares
            * ((settlement.yes_payout if lot.side == "YES" else no_payout) - lot.entry_price)
            - lot.entry_fee_usdc
            for lot in window_lots
        )
        self.cash += proceeds
        self.realized_pnl += realized_delta
        self._lots = [lot for lot in self._lots if lot.window_id != settlement.window_id]
        self._marks.pop((settlement.window_id, "YES"), None)
        self._marks.pop((settlement.window_id, "NO"), None)
        self.last_event_sequence = event_sequence
        self.timestamp_ms = settlement.settlement_ts_ms
        self._revalue()
        result = PaperSettlementEvent(
            schema_version=PAPER_SCHEMA_VERSION,
            run_id=self.run_id,
            event_id=event_id,
            event_sequence=event_sequence,
            settlement=settlement,
            proceeds_usdc=proceeds,
            realized_pnl_delta=realized_delta,
            cash_after=self.cash,
            realized_pnl=self.realized_pnl,
            commission_paid=self.commission_paid,
            equity=self.equity,
        )
        self._settlements[settlement.window_id] = result
        self._event_fingerprints[event_id] = fingerprint
        return result

    def settlement_ledger_event(self, event: PaperSettlementEvent) -> PaperLedgerEvent:
        """Create the ledger observation corresponding to a settlement event."""

        return self._ledger_event(
            event_id=f"{event.event_id}:ledger",
            source_event_id=event.event_id,
            kind=LedgerEventKind.SETTLEMENT,
            sequence=event.event_sequence,
            timestamp_ms=event.settlement.settlement_ts_ms,
            window_id=event.settlement.window_id,
        )

    def snapshot(self) -> PaperAccountSnapshot:
        """Return an immutable account snapshot."""

        return PaperAccountSnapshot(
            schema_version=PAPER_SCHEMA_VERSION,
            run_id=self.run_id,
            last_event_sequence=self.last_event_sequence,
            timestamp_ms=self.timestamp_ms,
            cash=self.cash,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=self._unrealized_pnl(),
            equity=self.equity,
            peak_equity=self.peak_equity,
            drawdown=self.drawdown,
            commission_paid=self.commission_paid,
            positions=self.positions(),
            open_lots=tuple(self._lots),
            settled_window_ids=tuple(sorted(self._settlements)),
        )

    def positions(self) -> tuple[PaperPosition, ...]:
        """Aggregate immutable lots by window and YES/NO side."""

        grouped: dict[tuple[str, str], list[PaperLot]] = {}
        for lot in self._lots:
            grouped.setdefault((lot.window_id, lot.side), []).append(lot)
        positions: list[PaperPosition] = []
        for key in sorted(grouped):
            lots = grouped[key]
            shares = sum(lot.shares for lot in lots)
            cost = sum(lot.shares * lot.entry_price for lot in lots)
            fees = sum(lot.entry_fee_usdc for lot in lots)
            mark = self._marks.get(key, 0.0)
            market_value = shares * mark
            positions.append(
                PaperPosition(
                    window_id=key[0],
                    market_symbol=lots[0].market_symbol,
                    side=key[1],
                    shares=shares,
                    average_entry_price=cost / shares,
                    cost_usdc=cost,
                    entry_fees_usdc=fees,
                    mark_bid=mark,
                    market_value_usdc=market_value,
                    unrealized_pnl=market_value - cost - fees,
                )
            )
        return tuple(positions)

    def replay(
        self,
        events: Iterable[PaperDecisionEvent | PaperSettlementEvent],
    ) -> PaperAccountSnapshot:
        """Replay authoritative events and verify persisted settlement results."""

        for event in events:
            if isinstance(event, PaperDecisionEvent):
                self.apply_decision(event)
                continue
            expected = event
            actual = self.settle(
                expected.settlement,
                event_id=expected.event_id,
                event_sequence=expected.event_sequence,
            )
            if actual != expected:
                raise ValueError("persisted settlement result does not match replay")
        return self.snapshot()

    def _apply_fill(self, event: PaperDecisionEvent) -> None:
        decision = event.decision
        if decision.order_status != "FILLED" or decision.order_side not in {"YES", "NO"}:
            raise ValueError("FILLED decision requires a filled YES/NO order")
        if (
            decision.shares is None
            or decision.fill_price is None
            or decision.fee_usdc is None
        ):
            raise ValueError("FILLED decision requires shares, price, and fee")
        shares = float(decision.shares)
        price = float(decision.fill_price)
        fee = float(decision.fee_usdc)
        if (
            not math.isfinite(shares)
            or not math.isfinite(price)
            or not math.isfinite(fee)
            or shares <= 0.0
            or not 0.0 < price <= 1.0
            or fee < 0.0
        ):
            raise ValueError("filled order values are invalid")
        expected_cash = self.cash - shares * price - fee
        if expected_cash < -1e-8:
            raise ValueError("paper fill exceeds available cash")
        _assert_close("filled decision cash_after", decision.cash_after, max(0.0, expected_cash))
        self.cash = max(0.0, expected_cash)
        self.commission_paid += fee
        self._lots.append(
            PaperLot(
                lot_id=f"{event.event_id}:lot",
                decision_event_id=event.event_id,
                window_id=decision.window_id,
                market_symbol=decision.market_symbol,
                side=decision.order_side,
                shares=shares,
                entry_price=price,
                entry_fee_usdc=fee,
                entry_ts_ms=decision.timestamp_ms,
            )
        )

    def _ledger_event(
        self,
        *,
        event_id: str,
        source_event_id: str,
        kind: LedgerEventKind,
        sequence: int,
        timestamp_ms: int,
        window_id: str,
    ) -> PaperLedgerEvent:
        snapshot = self.snapshot()
        return PaperLedgerEvent(
            schema_version=PAPER_SCHEMA_VERSION,
            run_id=self.run_id,
            event_id=event_id,
            event_sequence=sequence,
            kind=kind,
            source_event_id=source_event_id,
            timestamp_ms=timestamp_ms,
            window_id=window_id,
            cash=snapshot.cash,
            realized_pnl=snapshot.realized_pnl,
            unrealized_pnl=snapshot.unrealized_pnl,
            equity=snapshot.equity,
            peak_equity=snapshot.peak_equity,
            drawdown=snapshot.drawdown,
            commission_paid=snapshot.commission_paid,
            positions=snapshot.positions,
            open_lots=snapshot.open_lots,
        )

    def _unrealized_pnl(self) -> float:
        return sum(position.unrealized_pnl for position in self.positions())

    def _revalue(self) -> None:
        market_value = sum(position.market_value_usdc for position in self.positions())
        self.equity = self.cash + market_value
        self.peak_equity = max(self.peak_equity, self.equity)
        self.drawdown = (
            (self.peak_equity - self.equity) / self.peak_equity
            if self.peak_equity > 0.0
            else 0.0
        )

    def _duplicate_result(self, event_id: str, fingerprint: str) -> bool:
        existing = self._event_fingerprints.get(event_id)
        if existing is None:
            return False
        if existing != fingerprint:
            raise ValueError(f"conflicting duplicate event_id: {event_id}")
        return True

    def _validate_new_identity(self, run_id: str, sequence: int) -> None:
        if run_id != self.run_id:
            raise ValueError("event run_id does not match ledger")
        expected = self.last_event_sequence + 1
        if sequence != expected:
            raise ValueError(f"event_sequence must be {expected}, got {sequence}")


def replay_paper_events(
    *,
    run_id: str,
    initial_bankroll: float,
    windows: Iterable[PaperWindowRegistration],
    events: Iterable[PaperDecisionEvent | PaperSettlementEvent],
) -> PaperAccountLedger:
    """Construct and fully replay a ledger from authoritative events."""

    ledger = PaperAccountLedger(
        run_id=run_id,
        initial_bankroll=initial_bankroll,
        windows=windows,
    )
    ordered = sorted(events, key=lambda event: event.event_sequence)
    ledger.replay(ordered)
    return ledger


def _bid(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def _fingerprint(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_close(name: str, actual: float, expected: float) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError(f"{name} mismatch: expected {expected}, got {actual}")
