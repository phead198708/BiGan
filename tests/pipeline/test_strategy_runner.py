"""E2E StrategyRunner pipeline: OFI → pricing → OMS via CLOB snapshots."""

from __future__ import annotations

import pytest

from bigan.data.polymarket_clob import MarketSnapshot, PolymarketFeedHandler
from bigan.execution.polymarket_oms import REJECT_SPREAD_TOO_WIDE, PolymarketOMS
from bigan.features.binance_ofi import BinanceOFICalculator, TopOfBook
from bigan.pipeline.events import DecisionDisposition, DecisionReason
from bigan.pipeline.strategy_runner import PricingInputs, StrategyRunner
from bigan.strategies.polymarket_pricing import MarketWindow, PolymarketPricingEngine


def _window(*, end_ts_ms: int = 900_000) -> MarketWindow:
    return MarketWindow(
        window_id="btc-updown-15m-e2e",
        symbol="BTC",
        strike_price=100_000.0,
        start_ts_ms=0,
        end_ts_ms=end_ts_ms,
        window_type="15m",
    )


def _runner(
    *,
    bankroll: float = 1_000.0,
    window: MarketWindow | None = None,
    max_spread_allowed: float = 0.08,
    ofi_bid_qty: float = 1.0,
    ofi_ask_qty: float = 1.0,
    fee_bps: float = 0.0,
) -> StrategyRunner:
    market = window or _window()
    feed = PolymarketFeedHandler(
        window_id=market.window_id,
        yes_token_id="yes-token",
        no_token_id="no-token",
        mock=True,
    )
    return StrategyRunner(
        ofi_engine=BinanceOFICalculator(zscore_min_samples=5, ema_alpha=1.0),
        pricing_engine=PolymarketPricingEngine(),
        oms=PolymarketOMS(max_spread_allowed=max_spread_allowed),
        feed_handler=feed,
        window=market,
        initial_bankroll=bankroll,
        spot_price=market.strike_price,
        ofi_bid_qty=ofi_bid_qty,
        ofi_ask_qty=ofi_ask_qty,
        fee_bps=fee_bps,
        pricing_inputs_provider=lambda timestamp_ms: PricingInputs(
            timestamp_ms=timestamp_ms,
            spot_price=market.strike_price,
            oracle_twap_so_far=market.strike_price,
            twap_weight=0.0,
            volatility_annualized=0.60,
        ),
    )


def _payload(
    *,
    timestamp_ms: int,
    sequence: int,
    yes_bid: float,
    yes_ask: float,
    no_bid: float = 0.48,
    no_ask: float = 0.90,
) -> dict[str, object]:
    return {
        "event_type": "book",
        "window_id": "btc-updown-15m-e2e",
        "timestamp": timestamp_ms,
        "sequence": sequence,
        "yes": {
            "bids": [{"price": str(yes_bid), "size": "10"}],
            "asks": [{"price": str(yes_ask), "size": "10000"}],
        },
        "no": {
            "bids": [{"price": str(no_bid), "size": "1"}],
            "asks": [{"price": str(no_ask), "size": "10000"}],
        },
    }


def _buy_pressure_books(*, count: int = 28) -> list[float]:
    """Lift YES bid on a subset of ticks so raw OFI is not constant.

    Constant lifts produce identical ``raw_ofi`` and a zero Z-score because of
    the calculator variance floor. Mixing holds with a terminal lift burst
    yields positive ``z_ofi`` under buy pressure.
    """

    bids: list[float] = []
    bid = 0.30
    for i in range(count):
        in_burst = i >= count - 8
        should_lift = in_burst or i % 3 == 0
        if should_lift and i > 0:
            bid += 0.002
        bids.append(bid)
    return bids


def _alpha_book(*, timestamp_ms: int, bid_indicator: float) -> TopOfBook:
    bid = 100_000.0 + bid_indicator * 100.0
    return TopOfBook(
        ts_ms=timestamp_ms,
        bid_price=bid,
        bid_qty=10.0,
        ask_price=bid + 1.0,
        ask_qty=10.0,
    )


def test_push_tick_accepts_legacy_market_snapshot() -> None:
    runner = _runner(ofi_bid_qty=2.0, ofi_ask_qty=3.0)
    first = MarketSnapshot(
        timestamp_ms=100_000,
        window_id=runner.window.window_id,
        yes_bid=0.39,
        yes_ask=0.40,
        no_bid=0.09,
        no_ask=0.90,
        last_traded_price=0.40,
    )
    second = MarketSnapshot(
        timestamp_ms=100_100,
        window_id=runner.window.window_id,
        yes_bid=0.40,
        yes_ask=0.40,
        no_bid=0.09,
        no_ask=0.90,
        last_traded_price=0.40,
    )

    assert runner.push_tick(first) == 0.0
    assert runner.push_tick(second) == 0.0
    assert runner.ofi_engine.last_timestamp_ms == second.timestamp_ms
    assert runner.ofi_engine.last_raw_ofi == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_e2e_bullish_ofi_triggers_buy_yes() -> None:
    runner = _runner()
    starting = runner.current_bankroll
    await runner.start()
    last_z = 0.0
    for i, bid in enumerate(_buy_pressure_books()):
        timestamp_ms = 100_000 + i * 200
        runner.push_alpha_tick(
            _alpha_book(timestamp_ms=timestamp_ms, bid_indicator=bid)
        )
        await runner.feed_handler.ingest_payload(
            _payload(
                timestamp_ms=timestamp_ms,
                sequence=i + 1,
                yes_bid=bid,
                yes_ask=0.40,
            )
        )
        last_z = runner.ofi_engine.get_normalized_ofi()
    await runner.stop()

    assert runner.ofi_engine.last_raw_ofi > 0.0
    assert last_z > 0.0
    assert runner.oms_calls >= 1
    filled = [row for row in runner.execution_history if row.status == "FILLED"]
    assert filled
    assert filled[-1].side == "YES"
    assert filled[-1].shares > 0.0
    assert runner.current_bankroll < starting
    assert runner.current_bankroll == pytest.approx(runner.oms.bankroll)
    assert runner.callback_errors == 0


@pytest.mark.asyncio
async def test_e2e_tail_cutoff_blocks_trade() -> None:
    window = _window(end_ts_ms=900_000)
    runner = _runner(window=window)
    starting = runner.current_bankroll
    await runner.start()
    for i, bid in enumerate(_buy_pressure_books()):
        timestamp_ms = window.end_ts_ms - 10_000 + i
        runner.push_alpha_tick(
            _alpha_book(timestamp_ms=timestamp_ms, bid_indicator=bid)
        )
        await runner.feed_handler.ingest_payload(
            _payload(
                timestamp_ms=timestamp_ms,
                sequence=i + 1,
                yes_bid=bid,
                yes_ask=0.40,
            )
        )
    await runner.stop()

    assert runner.ofi_engine.get_normalized_ofi() > 0.0
    assert runner.oms_calls == 0
    assert runner.execution_history == []
    assert runner.current_bankroll == pytest.approx(starting)
    assert runner.oms.positions() == ()
    assert runner.callback_errors == 0


@pytest.mark.asyncio
async def test_polymarket_book_never_mutates_binance_ofi_state() -> None:
    runner = _runner()
    await runner.start()
    for i, bid in enumerate(_buy_pressure_books(count=8)):
        await runner.feed_handler.ingest_payload(
            _payload(
                timestamp_ms=100_000 + i * 200,
                sequence=i + 1,
                yes_bid=bid,
                yes_ask=0.40,
            )
        )
    await runner.stop()

    assert runner.ofi_engine.last_timestamp_ms is None
    assert runner.ofi_engine.last_raw_ofi == 0.0
    assert runner.ofi_engine.get_normalized_ofi() == 0.0


@pytest.mark.asyncio
async def test_e2e_wide_spread_rejection() -> None:
    runner = _runner()
    starting = runner.current_bankroll
    await runner.start()
    await runner.feed_handler.ingest_payload(
        _payload(
            timestamp_ms=100_000,
            sequence=1,
            yes_bid=0.30,
            yes_ask=0.42,
        )
    )
    await runner.feed_handler.ingest_payload(
        _payload(
            timestamp_ms=100_200,
            sequence=2,
            yes_bid=0.30,
            yes_ask=0.42,
        )
    )
    await runner.stop()

    assert runner.oms_calls >= 1
    assert runner.execution_history
    last = runner.execution_history[-1]
    assert last.status == "REJECTED"
    assert last.reject_reason == REJECT_SPREAD_TOO_WIDE
    assert last.shares == 0.0
    assert runner.current_bankroll == pytest.approx(starting)
    assert runner.oms.positions() == ()
    assert runner.callback_errors == 0


def _snapshot(
    runner: StrategyRunner,
    *,
    timestamp_ms: int = 100_000,
    window_id: str | None = None,
    yes_bid: float = 0.39,
    yes_ask: float = 0.40,
    no_bid: float = 0.39,
    no_ask: float = 0.90,
) -> MarketSnapshot:
    return MarketSnapshot(
        timestamp_ms=timestamp_ms,
        window_id=window_id or runner.window.window_id,
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


def test_every_fail_closed_and_hold_path_emits_a_decision() -> None:
    runner = _runner()
    events = []
    runner.on_decision(events.append)

    runner.process_snapshot_sync(
        _snapshot(runner, window_id="different-window")
    )
    runner.pricing_inputs_provider = lambda _timestamp: None
    runner.process_snapshot_sync(_snapshot(runner, timestamp_ms=100_001))
    runner.pricing_inputs_provider = lambda timestamp: PricingInputs(
        timestamp_ms=timestamp - runner.reference_max_age_ms - 1,
        spot_price=100_000.0,
        oracle_twap_so_far=100_000.0,
        twap_weight=0.0,
        volatility_annualized=0.60,
    )
    runner.process_snapshot_sync(_snapshot(runner, timestamp_ms=100_002))
    runner.pricing_inputs_provider = lambda timestamp: PricingInputs(
        timestamp_ms=timestamp,
        spot_price=100_000.0,
        oracle_twap_so_far=100_000.0,
        twap_weight=0.0,
        volatility_annualized=0.60,
    )
    runner.process_snapshot_sync(
        _snapshot(runner, timestamp_ms=runner.window.end_ts_ms - 1)
    )

    assert [event.disposition for event in events] == [
        DecisionDisposition.DROPPED,
        DecisionDisposition.DROPPED,
        DecisionDisposition.DROPPED,
        DecisionDisposition.HOLD,
    ]
    assert [event.reason_code for event in events] == [
        DecisionReason.WINDOW_MISMATCH,
        DecisionReason.PRICING_INPUTS_MISSING,
        DecisionReason.PRICING_INPUTS_STALE,
        DecisionReason.SIGNAL_HOLD,
    ]
    assert runner.oms_calls == 0
    assert runner.decision_count == 4


def test_buy_yes_fee_and_cash_are_recorded_consistently() -> None:
    runner = _runner(fee_bps=100.0)

    result = runner.process_snapshot_sync(_snapshot(runner))

    assert result is not None and result.status == "FILLED"
    assert result.side == "YES"
    expected_fee = result.shares * result.price * 0.01
    assert result.fee_usdc == pytest.approx(expected_fee)
    assert runner.current_bankroll == pytest.approx(
        1_000.0 - result.shares * result.price - expected_fee
    )
    assert runner.current_bankroll == pytest.approx(runner.oms.bankroll)
    assert runner.execution_history[-1] == result
    assert runner.last_decision is not None
    assert runner.last_decision.disposition is DecisionDisposition.FILLED
    assert runner.last_decision.fee_usdc == pytest.approx(expected_fee)
    assert runner.last_decision.to_dict()["disposition"] == "FILLED"


def test_buy_no_rejection_and_no_order_events() -> None:
    buy_no = _runner()
    buy_no.pricing_inputs_provider = lambda timestamp: PricingInputs(
        timestamp_ms=timestamp,
        spot_price=90_000.0,
        oracle_twap_so_far=100_000.0,
        twap_weight=0.0,
        volatility_annualized=0.10,
    )
    no_result = buy_no.process_snapshot_sync(
        _snapshot(buy_no, yes_bid=0.09, yes_ask=0.90, no_bid=0.39, no_ask=0.40)
    )
    assert no_result is not None and no_result.side == "NO"
    assert buy_no.last_decision is not None
    assert buy_no.last_decision.disposition is DecisionDisposition.FILLED

    rejected = _runner(max_spread_allowed=0.01)
    reject_result = rejected.process_snapshot_sync(
        _snapshot(rejected, yes_bid=0.30, yes_ask=0.40)
    )
    assert reject_result is not None and reject_result.status == "REJECTED"
    assert rejected.last_decision is not None
    assert rejected.last_decision.disposition is DecisionDisposition.REJECTED
    assert rejected.last_decision.cash_before == rejected.last_decision.cash_after

    duplicate = _runner()
    snapshot = _snapshot(duplicate)
    assert duplicate.process_snapshot_sync(snapshot) is not None
    assert duplicate.process_snapshot_sync(snapshot) is None
    assert duplicate.last_decision is not None
    assert duplicate.last_decision.disposition is DecisionDisposition.NO_ORDER


def test_alpha_missing_stale_and_fresh_are_distinguishable() -> None:
    runner = _runner()
    runner.process_snapshot_sync(_snapshot(runner, timestamp_ms=100_000))
    missing = runner.last_decision
    assert missing is not None
    assert missing.alpha_reason_code is DecisionReason.ALPHA_MISSING
    assert missing.z_ofi == 0.0

    runner.push_alpha_tick(_alpha_book(timestamp_ms=90_000, bid_indicator=0.3))
    runner.process_snapshot_sync(_snapshot(runner, timestamp_ms=100_001))
    stale = runner.last_decision
    assert stale is not None
    assert stale.alpha_reason_code is DecisionReason.ALPHA_STALE
    assert stale.alpha_timestamp_ms == 90_000
    assert stale.z_ofi == 0.0


def test_decision_callback_failure_is_isolated_from_fill_and_other_callbacks() -> None:
    runner = _runner()
    observed = []

    def broken(_event: object) -> None:
        raise RuntimeError("subscriber failed")

    runner.on_decision(broken)
    runner.on_decision(observed.append)
    result = runner.process_snapshot_sync(_snapshot(runner))

    assert result is not None and result.status == "FILLED"
    assert len(observed) == 1
    assert runner.decision_callback_errors == 1
    assert runner.current_bankroll < 1_000.0


@pytest.mark.parametrize("fee_bps", [-1.0, float("nan"), float("inf")])
def test_invalid_paper_fee_is_rejected(fee_bps: float) -> None:
    with pytest.raises(ValueError, match="fee_bps"):
        _runner(fee_bps=fee_bps)
