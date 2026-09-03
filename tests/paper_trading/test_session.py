"""Fixed-window StrategyRunner → ledger → storage integration tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.data.polymarket_clob import MarketSnapshot, PolymarketFeedHandler
from bigan.execution.polymarket_oms import PolymarketOMS
from bigan.features.binance_ofi import BinanceOFICalculator, TopOfBook
from bigan.paper_trading.contracts import PaperSettlementInput
from bigan.paper_trading.session import PaperSessionFailedError, PaperTradingSession
from bigan.paper_trading.storage import (
    LEDGER_EVENTS_FILE,
    SETTLEMENT_EVENTS_FILE,
    SIGNAL_EVENTS_FILE,
)
from bigan.pipeline.events import DecisionDisposition
from bigan.pipeline.strategy_runner import PricingInputs, StrategyRunner
from bigan.strategies.polymarket_pricing import MarketWindow, PolymarketPricingEngine

WINDOW = MarketWindow(
    window_id="btc-paper-window",
    symbol="BTC",
    strike_price=100_000.0,
    start_ts_ms=0,
    end_ts_ms=900_000,
    window_type="15m",
)


def _runner(*, fee_bps: float = 100.0) -> StrategyRunner:
    return StrategyRunner(
        ofi_engine=BinanceOFICalculator(zscore_min_samples=2, ema_alpha=1.0),
        pricing_engine=PolymarketPricingEngine(),
        oms=PolymarketOMS(max_spread_allowed=0.08),
        feed_handler=PolymarketFeedHandler(window_id=WINDOW.window_id, mock=True),
        window=WINDOW,
        initial_bankroll=1_000.0,
        spot_price=WINDOW.strike_price,
        pricing_inputs_provider=lambda timestamp_ms: PricingInputs(
            timestamp_ms=timestamp_ms,
            spot_price=WINDOW.strike_price,
            oracle_twap_so_far=WINDOW.strike_price,
            twap_weight=0.0,
            volatility_annualized=0.60,
        ),
        fee_bps=fee_bps,
    )


def _snapshot(
    timestamp_ms: int,
    *,
    yes_bid: float,
    yes_ask: float,
    no_bid: float,
    no_ask: float,
) -> MarketSnapshot:
    return MarketSnapshot(
        timestamp_ms=timestamp_ms,
        window_id=WINDOW.window_id,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        last_traded_price=yes_ask,
        yes_bid_size=100.0,
        yes_ask_size=100.0,
        no_bid_size=100.0,
        no_ask_size=100.0,
    )


def _alpha(timestamp_ms: int, price: float) -> TopOfBook:
    return TopOfBook(
        ts_ms=timestamp_ms,
        bid_price=price,
        bid_qty=10.0,
        ask_price=price + 1.0,
        ask_qty=10.0,
    )


def _settlement() -> PaperSettlementInput:
    return PaperSettlementInput(
        window_id=WINDOW.window_id,
        yes_payout=1.0,
        settlement_ts_ms=WINDOW.end_ts_ms,
        source="deterministic-test-oracle",
        source_ts_ms=WINDOW.end_ts_ms,
        received_ts_ms=WINDOW.end_ts_ms + 1,
        source_reference="fixture://btc-paper-window",
    )


def test_fixed_window_e2e_resume_and_settlement(tmp_path: Path) -> None:
    runner = _runner()
    session = PaperTradingSession.create_new(
        runner=runner,
        output_dir=tmp_path,
        run_id="paper-e2e",
        source_commit="fbd64d9",
        config={"feed": "offline-fixture"},
        created_at="2026-09-03T00:00:00+00:00",
    )

    runner.push_alpha_tick(_alpha(100_000, 100_000.0))
    session.process_snapshot_sync(
        _snapshot(100_000, yes_bid=0.98, yes_ask=0.99, no_bid=0.0, no_ask=0.99)
    )
    runner.push_alpha_tick(_alpha(100_100, 100_010.0))
    fill = session.process_snapshot_sync(
        _snapshot(100_100, yes_bid=0.39, yes_ask=0.40, no_bid=0.09, no_ask=0.90)
    )
    runner.push_alpha_tick(_alpha(100_200, 100_020.0))
    rejection = session.process_snapshot_sync(
        _snapshot(100_200, yes_bid=0.20, yes_ask=0.40, no_bid=0.09, no_ask=0.90)
    )

    assert fill is not None and fill.status == "FILLED"
    assert rejection is not None and rejection.status == "REJECTED"
    before_resume = session.current_snapshot
    assert before_resume.cash == pytest.approx(runner.current_bankroll)
    assert before_resume.cash == pytest.approx(runner.oms.bankroll)
    assert len(before_resume.open_lots) == 1

    rows = [
        json.loads(line)
        for line in (session.store.run_dir / SIGNAL_EVENTS_FILE).read_text().splitlines()
    ]
    assert [row["event_sequence"] for row in rows] == [1, 2, 3]
    assert [row["decision"]["disposition"] for row in rows] == [
        DecisionDisposition.HOLD,
        DecisionDisposition.FILLED,
        DecisionDisposition.REJECTED,
    ]
    assert all(row["paper_only"] is True for row in rows)
    assert all(row["polymarket_write_enabled"] is False for row in rows)

    resumed = PaperTradingSession.resume_existing(
        runner=_runner(),
        output_dir=tmp_path,
        run_id="paper-e2e",
        source_commit="fbd64d9",
        config={"feed": "offline-fixture"},
    )
    assert resumed.current_snapshot == before_resume
    assert resumed.runner.oms.positions()
    settlement = resumed.settle(_settlement())
    final = resumed.current_snapshot
    assert settlement.event_sequence == 4
    assert final.last_event_sequence == 4
    assert final.positions == final.open_lots == ()
    assert final.equity == final.cash
    assert resumed.runner.oms.positions() == ()
    assert final.cash == pytest.approx(resumed.runner.current_bankroll)
    assert final.cash == pytest.approx(resumed.runner.oms.bankroll)
    assert len((resumed.store.run_dir / SETTLEMENT_EVENTS_FILE).read_text().splitlines()) == 1
    assert len((resumed.store.run_dir / LEDGER_EVENTS_FILE).read_text().splitlines()) == 4


def test_resume_rejects_changed_configuration(tmp_path: Path) -> None:
    PaperTradingSession.create_new(
        runner=_runner(fee_bps=0.0),
        output_dir=tmp_path,
        run_id="config-test",
        source_commit="commit-a",
        created_at="2026-09-03T00:00:00+00:00",
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        PaperTradingSession.resume_existing(
            runner=_runner(fee_bps=50.0),
            output_dir=tmp_path,
            run_id="config-test",
            source_commit="commit-a",
        )


def test_window_mismatch_is_a_durable_non_mutating_decision(tmp_path: Path) -> None:
    session = PaperTradingSession.create_new(
        runner=_runner(),
        output_dir=tmp_path,
        run_id="mismatch-test",
        source_commit="commit-a",
    )
    foreign = _snapshot(
        100_000,
        yes_bid=0.39,
        yes_ask=0.40,
        no_bid=0.09,
        no_ask=0.90,
    )
    foreign = MarketSnapshot(
        timestamp_ms=foreign.timestamp_ms,
        window_id="foreign-window",
        yes_bid=foreign.yes_bid,
        yes_ask=foreign.yes_ask,
        no_bid=foreign.no_bid,
        no_ask=foreign.no_ask,
        last_traded_price=foreign.last_traded_price,
        yes_bid_size=foreign.yes_bid_size,
        yes_ask_size=foreign.yes_ask_size,
        no_bid_size=foreign.no_bid_size,
        no_ask_size=foreign.no_ask_size,
    )
    assert session.process_snapshot_sync(foreign) is None
    assert session.failed is False
    assert session.current_snapshot.cash == 1_000.0
    assert session.current_snapshot.last_event_sequence == 1


def test_storage_failure_permanently_fails_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = PaperTradingSession.create_new(
        runner=_runner(),
        output_dir=tmp_path,
        run_id="failure-test",
        source_commit="commit-a",
    )

    def fail_append(**_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(session.store, "append_decision", fail_append)
    with pytest.raises(PaperSessionFailedError, match="disk full"):
        session.process_snapshot_sync(
            _snapshot(100_000, yes_bid=0.98, yes_ask=0.99, no_bid=0.0, no_ask=0.99)
        )
    assert session.failed is True
    with pytest.raises(PaperSessionFailedError, match="disk full"):
        session.process_snapshot_sync(
            _snapshot(100_001, yes_bid=0.39, yes_ask=0.40, no_bid=0.09, no_ask=0.90)
        )
