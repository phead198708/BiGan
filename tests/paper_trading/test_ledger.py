"""Deterministic paper ledger accounting, idempotency, and settlement tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from bigan.paper_trading.contracts import PaperSettlementInput
from bigan.paper_trading.ledger import PaperAccountLedger, replay_paper_events
from bigan.pipeline.events import DecisionDisposition
from tests.paper_trading.helpers import RUN_ID, WINDOW, paper_decision


def _ledger() -> PaperAccountLedger:
    return PaperAccountLedger(
        run_id=RUN_ID,
        initial_bankroll=1_000.0,
        windows=(WINDOW,),
    )


def _settlement(payout: float = 1.0) -> PaperSettlementInput:
    return PaperSettlementInput(
        window_id=WINDOW.window_id,
        yes_payout=payout,
        settlement_ts_ms=WINDOW.end_ts_ms,
        source="test-oracle",
        source_ts_ms=WINDOW.end_ts_ms,
        received_ts_ms=WINDOW.end_ts_ms + 1,
        source_reference="fixture-1",
    )


def test_initial_snapshot_and_single_yes_fill_mtm() -> None:
    ledger = _ledger()
    initial = ledger.snapshot()
    assert initial.cash == initial.equity == initial.peak_equity == 1_000.0
    assert initial.positions == initial.open_lots == ()

    event = paper_decision(1, yes_bid=0.39)
    mutation = ledger.apply_decision(event)
    snapshot = ledger.snapshot()
    assert mutation is not None
    assert snapshot.cash == pytest.approx(995.96)
    assert snapshot.commission_paid == pytest.approx(0.04)
    assert snapshot.positions[0].side == "YES"
    assert snapshot.positions[0].shares == 10.0
    assert snapshot.unrealized_pnl == pytest.approx(-0.14)
    assert snapshot.equity == pytest.approx(999.86)
    assert snapshot.drawdown == pytest.approx(0.00014)


def test_yes_no_and_multiple_lots_are_aggregated() -> None:
    ledger = _ledger()
    first = paper_decision(1)
    ledger.apply_decision(first)
    second = paper_decision(2, shares=5.0, price=0.50, fee=0.025, cash_before=995.96)
    ledger.apply_decision(second)
    third = paper_decision(
        3,
        side="NO",
        shares=4.0,
        price=0.60,
        fee=0.024,
        cash_before=993.435,
    )
    ledger.apply_decision(third)

    positions = ledger.snapshot().positions
    assert len(ledger.snapshot().open_lots) == 3
    assert [(row.side, row.shares) for row in positions] == [("NO", 4.0), ("YES", 15.0)]


@pytest.mark.parametrize(
    "disposition",
    [
        DecisionDisposition.HOLD,
        DecisionDisposition.REJECTED,
        DecisionDisposition.NO_ORDER,
        DecisionDisposition.DROPPED,
    ],
)
def test_non_fills_do_not_change_balance(disposition: DecisionDisposition) -> None:
    ledger = _ledger()
    assert ledger.apply_decision(paper_decision(1, disposition=disposition)) is not None
    assert ledger.snapshot().cash == 1_000.0
    assert ledger.snapshot().positions == ()


def test_mark_to_market_uses_executable_bid() -> None:
    ledger = _ledger()
    ledger.apply_decision(paper_decision(1))
    snapshot = ledger.mark_to_market(
        window_id=WINDOW.window_id,
        yes_bid=0.80,
        no_bid=0.19,
        timestamp_ms=5_000,
    )
    assert snapshot.equity == pytest.approx(1_003.96)
    assert snapshot.peak_equity == pytest.approx(1_003.96)
    assert snapshot.drawdown == 0.0


@pytest.mark.parametrize("payout", [0.0, 0.25, 1.0])
def test_settlement_realizes_lots_and_fee_once(payout: float) -> None:
    ledger = _ledger()
    decision = paper_decision(1)
    ledger.apply_decision(decision)
    event = ledger.settle(_settlement(payout), event_id="settle-2", event_sequence=2)
    expected_proceeds = 10.0 * payout
    assert event.proceeds_usdc == pytest.approx(expected_proceeds)
    assert event.realized_pnl_delta == pytest.approx(expected_proceeds - 4.0 - 0.04)
    snapshot = ledger.snapshot()
    assert snapshot.cash == pytest.approx(995.96 + expected_proceeds)
    assert snapshot.equity == snapshot.cash
    assert snapshot.positions == snapshot.open_lots == ()
    assert snapshot.commission_paid == pytest.approx(0.04)


def test_losing_no_position_settlement() -> None:
    ledger = _ledger()
    ledger.apply_decision(paper_decision(1, side="NO", price=0.60, fee=0.06))
    event = ledger.settle(_settlement(1.0), event_id="settle-2", event_sequence=2)
    assert event.proceeds_usdc == 0.0
    assert event.realized_pnl_delta == pytest.approx(-6.06)


@pytest.mark.parametrize("payout", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_settlement_payout_is_rejected(payout: float) -> None:
    with pytest.raises(ValueError):
        _settlement(payout)


def test_settlement_requires_expiry_and_registered_window() -> None:
    ledger = _ledger()
    with pytest.raises(ValueError, match="before"):
        ledger.settle(
            replace(_settlement(), settlement_ts_ms=WINDOW.end_ts_ms - 1),
            event_id="settle-1",
            event_sequence=1,
        )
    with pytest.raises(ValueError, match="unregistered"):
        ledger.settle(
            replace(_settlement(), window_id="other"),
            event_id="settle-1",
            event_sequence=1,
        )


def test_duplicate_and_conflicting_settlement() -> None:
    ledger = _ledger()
    original = ledger.settle(_settlement(), event_id="settle-1", event_sequence=1)
    assert ledger.settle(_settlement(), event_id="another-id", event_sequence=2) == original
    with pytest.raises(ValueError, match="conflicting"):
        ledger.settle(
            replace(_settlement(), yes_payout=0.0),
            event_id="settle-2",
            event_sequence=2,
        )


def test_duplicate_decision_is_idempotent_and_conflict_fails() -> None:
    ledger = _ledger()
    event = paper_decision(1)
    ledger.apply_decision(event)
    before = ledger.snapshot()
    assert ledger.apply_decision(event) is None
    assert ledger.snapshot() == before
    with pytest.raises(ValueError, match="conflicting duplicate"):
        ledger.apply_decision(replace(event, decision=replace(event.decision, yes_bid=0.20)))


def test_sequence_is_strict_and_replay_matches_online_state() -> None:
    first = paper_decision(1)
    second = paper_decision(2, disposition=DecisionDisposition.HOLD, cash_before=995.96)
    ledger = _ledger()
    ledger.apply_decision(first)
    ledger.apply_decision(second)
    settlement = ledger.settle(_settlement(), event_id="settle-3", event_sequence=3)

    replayed = replay_paper_events(
        run_id=RUN_ID,
        initial_bankroll=1_000.0,
        windows=(WINDOW,),
        events=(first, second, settlement),
    )
    assert replayed.snapshot() == ledger.snapshot()

    with pytest.raises(ValueError, match="event_sequence"):
        _ledger().apply_decision(paper_decision(2))
