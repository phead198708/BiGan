"""E2E StrategyRunner pipeline: OFI → pricing → OMS via CLOB snapshots."""

from __future__ import annotations

import pytest

from bigan.data.polymarket_clob import PolymarketFeedHandler
from bigan.execution.polymarket_oms import REJECT_SPREAD_TOO_WIDE, PolymarketOMS
from bigan.features.binance_ofi import BinanceOFICalculator
from bigan.pipeline.strategy_runner import StrategyRunner
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
            "asks": [{"price": str(yes_ask), "size": "1"}],
        },
        "no": {
            "bids": [{"price": str(no_bid), "size": "1"}],
            "asks": [{"price": str(no_ask), "size": "1"}],
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


@pytest.mark.asyncio
async def test_e2e_bullish_ofi_triggers_buy_yes() -> None:
    runner = _runner()
    starting = runner.current_bankroll
    await runner.start()
    last_z = 0.0
    for i, bid in enumerate(_buy_pressure_books()):
        await runner.feed_handler.ingest_payload(
            _payload(
                timestamp_ms=100_000 + i * 200,
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
        await runner.feed_handler.ingest_payload(
            _payload(
                timestamp_ms=window.end_ts_ms - 10_000 + i,
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
