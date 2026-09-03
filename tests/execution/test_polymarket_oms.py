"""DEV-03 Polymarket OMS pre-trade gates and simulated fills."""

from __future__ import annotations

import pytest

from bigan.execution.polymarket_oms import (
    REJECT_LIQUIDITY_UNAVAILABLE,
    REJECT_SIZE_BELOW_MINIMUM,
    REJECT_SPREAD_TOO_WIDE,
    LimitOrder,
    PolymarketOMS,
)
from bigan.strategies.polymarket_pricing import PricingSignal, SignalDirection


def _signal(
    *,
    direction: SignalDirection,
    recommended_size_pct: float = 0.10,
    market_price: float = 0.50,
    window_id: str = "btc-updown-test",
    edge: float = 0.10,
    ts_ms: int = 1_000,
) -> PricingSignal:
    return PricingSignal(
        ts_ms=ts_ms,
        window_id=window_id,
        spot_price=100_000.0,
        effective_strike=100_000.0,
        model_prob=0.70,
        market_price=market_price,
        edge=edge,
        ev=0.20,
        direction=direction,
        recommended_size_pct=recommended_size_pct,
    )


def test_hold_signal_returns_none() -> None:
    oms = PolymarketOMS()
    filled = oms.process_signal(
        _signal(direction=SignalDirection.BUY_YES, recommended_size_pct=0.05),
        current_bankroll=1_000.0,
        current_bid=0.49,
        current_ask_size=10_000.0,
    )
    assert filled is not None
    assert filled.status == "FILLED"
    before = oms.positions()

    result = oms.process_signal(
        _signal(direction=SignalDirection.HOLD, recommended_size_pct=0.50),
        current_bankroll=10_000.0,
        current_bid=0.10,
    )
    assert result is None
    after = oms.positions()
    assert after == before
    assert after[0].shares == pytest.approx(before[0].shares)
    assert after[0].total_cost_usdc == pytest.approx(before[0].total_cost_usdc)


def test_max_trade_cap_enforcement() -> None:
    oms = PolymarketOMS(max_single_trade_pct=0.05)
    bankroll = 1_000.0
    market_price = 0.50
    result = oms.process_signal(
        _signal(
            direction=SignalDirection.BUY_YES,
            recommended_size_pct=0.15,
            market_price=market_price,
        ),
        current_bankroll=bankroll,
        current_bid=0.49,
        current_ask_size=10_000.0,
    )
    assert result is not None
    assert result.status == "FILLED"
    fill_price = min(1.0, market_price * (1.0 + oms.slippage_tolerance))
    expected_usd = bankroll * 0.05
    assert result.price == pytest.approx(fill_price)
    assert result.shares == pytest.approx(expected_usd / fill_price)
    position = oms.get_position("btc-updown-test", "YES")
    assert position is not None
    assert position.total_cost_usdc == pytest.approx(expected_usd)
    assert position.shares == pytest.approx(expected_usd / fill_price)
    assert position.total_cost_usdc < bankroll * 0.15


def test_min_order_threshold_rejection() -> None:
    oms = PolymarketOMS(min_order_usd=1.0)
    result = oms.process_signal(
        _signal(direction=SignalDirection.BUY_YES, recommended_size_pct=0.05),
        current_bankroll=10.0,
        current_bid=0.49,
        current_ask_size=10_000.0,
    )
    assert result is not None
    assert result.status == "REJECTED"
    assert result.reject_reason == REJECT_SIZE_BELOW_MINIMUM
    assert result.shares == 0.0
    assert oms.positions() == ()


def test_spread_guard_rejection() -> None:
    oms = PolymarketOMS(max_spread_allowed=0.08)
    result = oms.process_signal(
        _signal(direction=SignalDirection.BUY_YES, market_price=0.60),
        current_bankroll=1_000.0,
        current_bid=0.50,
        current_ask_size=10_000.0,
    )
    assert result is not None
    assert result.status == "REJECTED"
    assert result.reject_reason == REJECT_SPREAD_TOO_WIDE
    assert oms.get_position("btc-updown-test", "YES") is None

    tight = oms.process_signal(
        _signal(direction=SignalDirection.BUY_YES, market_price=0.58),
        current_bankroll=1_000.0,
        current_bid=0.50,
        current_ask_size=10_000.0,
    )
    assert tight is not None
    assert tight.status == "FILLED"


def test_position_state_update() -> None:
    oms = PolymarketOMS()
    bankroll = 1_000.0
    first_ask = 0.40
    first = oms.process_signal(
        _signal(
            direction=SignalDirection.BUY_YES,
            recommended_size_pct=0.05,
            market_price=first_ask,
            window_id="window-a",
        ),
        current_bankroll=bankroll,
        current_bid=0.39,
        current_ask_size=10_000.0,
    )
    assert first is not None and first.status == "FILLED"
    first_fill = min(1.0, first_ask * (1.0 + oms.slippage_tolerance))
    first_cost = bankroll * 0.05
    yes = oms.get_position("window-a", "YES")
    assert yes is not None
    assert yes.side == "YES"
    assert yes.shares == pytest.approx(first.shares)
    assert yes.avg_entry_price == pytest.approx(first_fill)
    assert yes.total_cost_usdc == pytest.approx(first_cost)

    second_ask = 0.60
    second = oms.process_signal(
        _signal(
            direction=SignalDirection.BUY_YES,
            recommended_size_pct=0.10,
            market_price=second_ask,
            window_id="window-a",
            ts_ms=2_000,
        ),
        current_bankroll=oms.bankroll,
        current_bid=0.59,
        current_ask_size=10_000.0,
    )
    assert second is not None and second.status == "FILLED"
    second_fill = min(1.0, second_ask * (1.0 + oms.slippage_tolerance))
    second_cost = bankroll * 0.05
    yes = oms.get_position("window-a", "YES")
    assert yes is not None
    expected_shares = first.shares + second.shares
    expected_cost = first_cost + second_cost
    assert yes.shares == pytest.approx(expected_shares)
    assert yes.total_cost_usdc == pytest.approx(expected_cost)
    assert yes.avg_entry_price == pytest.approx(expected_cost / expected_shares)
    assert yes.avg_entry_price == pytest.approx(
        (first.shares * first_fill + second.shares * second_fill) / expected_shares
    )

    no_ask = 0.30
    buy_no = oms.process_signal(
        _signal(
            direction=SignalDirection.BUY_NO,
            recommended_size_pct=0.05,
            market_price=no_ask,
            window_id="window-a",
            ts_ms=3_000,
        ),
        current_bankroll=oms.bankroll,
        current_bid=0.29,
        current_ask_size=10_000.0,
    )
    assert buy_no is not None and buy_no.status == "FILLED"
    assert buy_no.side == "NO"
    no_pos = oms.get_position("window-a", "NO")
    assert no_pos is not None
    assert no_pos.side == "NO"
    assert no_pos.shares == pytest.approx(buy_no.shares)
    assert no_pos.avg_entry_price == pytest.approx(buy_no.price)
    assert oms.get_position("window-a", "YES") is not None
    assert len(oms.positions()) == 2


def test_repeated_signal_targets_total_position_instead_of_rebuying_target() -> None:
    oms = PolymarketOMS(max_single_trade_pct=0.05)
    cash = 1_000.0
    for ts_ms in (1_000, 2_000):
        result = oms.process_signal(
            _signal(
                direction=SignalDirection.BUY_YES,
                recommended_size_pct=0.05,
                ts_ms=ts_ms,
            ),
            current_bankroll=cash,
            current_bid=0.49,
            current_ask_size=10_000.0,
        )
        if ts_ms == 1_000:
            assert result is not None and result.status == "FILLED"
            cash = oms.bankroll
        else:
            assert result is None
    position = oms.get_position("btc-updown-test", "YES")
    assert position is not None
    assert position.total_cost_usdc == pytest.approx(50.0)


def test_available_ask_liquidity_caps_fill_and_missing_depth_rejects() -> None:
    oms = PolymarketOMS(min_order_usd=1.0, slippage_tolerance=0.0)
    missing = oms.process_signal(
        _signal(direction=SignalDirection.BUY_YES, ts_ms=1_000),
        current_bankroll=1_000.0,
        current_bid=0.49,
    )
    assert missing is not None
    assert missing.status == "REJECTED"
    assert missing.reject_reason == REJECT_LIQUIDITY_UNAVAILABLE

    partial = oms.process_signal(
        _signal(direction=SignalDirection.BUY_YES, ts_ms=2_000),
        current_bankroll=1_000.0,
        current_bid=0.49,
        current_ask_size=10.0,
    )
    assert partial is not None and partial.status == "FILLED"
    assert partial.shares == pytest.approx(10.0)
    assert partial.shares * partial.price == pytest.approx(5.0)


def test_duplicate_signal_is_idempotent() -> None:
    oms = PolymarketOMS()
    signal = _signal(direction=SignalDirection.BUY_YES)
    first = oms.process_signal(signal, 1_000.0, 0.49, 10_000.0)
    second = oms.process_signal(signal, oms.bankroll, 0.49, 10_000.0)
    assert first is not None and first.status == "FILLED"
    assert second is None
    assert len(oms.positions()) == 1


def test_local_rejection_does_not_consume_signal_idempotency_key() -> None:
    oms = PolymarketOMS()
    signal = _signal(direction=SignalDirection.BUY_YES)
    rejected = oms.process_signal(
        signal,
        current_bankroll=1_000.0,
        current_bid=0.49,
        current_ask_size=0.0,
    )
    assert rejected is not None
    assert rejected.status == "REJECTED"
    assert rejected.reject_reason == REJECT_LIQUIDITY_UNAVAILABLE

    retried = oms.process_signal(
        signal,
        current_bankroll=1_000.0,
        current_bid=0.49,
        current_ask_size=100.0,
    )
    assert retried is not None
    assert retried.status == "FILLED"
    assert oms.get_position("btc-updown-test", "YES") is not None


@pytest.mark.parametrize("crossed_ask", [0.37, 0.20])
def test_limit_order_fixes_shares_before_price_improvement(crossed_ask: float) -> None:
    oms = PolymarketOMS(slippage_tolerance=0.01)
    placed = oms.prepare_limit_order(
        _signal(
            direction=SignalDirection.BUY_YES,
            market_price=0.42,
            recommended_size_pct=0.10,
        ),
        current_bankroll=1_000.0,
        current_bid=0.38,
    )
    assert isinstance(placed, LimitOrder)
    assert placed.shares == pytest.approx(50.0 / 0.38)

    filled, remaining = oms.fill_limit_order(
        placed,
        current_bankroll=1_000.0,
        current_ask=crossed_ask,
        current_ask_size=10_000.0,
    )

    assert filled is not None and filled.status == "FILLED"
    assert filled.shares == pytest.approx(placed.shares)
    assert filled.price <= placed.limit_price
    assert remaining is None


def test_limit_order_partial_fills_only_its_remaining_shares() -> None:
    oms = PolymarketOMS(slippage_tolerance=0.0)
    placed = oms.prepare_limit_order(
        _signal(
            direction=SignalDirection.BUY_YES,
            market_price=0.42,
            recommended_size_pct=0.10,
        ),
        current_bankroll=1_000.0,
        current_bid=0.38,
    )
    assert isinstance(placed, LimitOrder)

    first, remaining = oms.fill_limit_order(
        placed,
        current_bankroll=1_000.0,
        current_ask=0.37,
        current_ask_size=10.0,
    )
    assert first is not None and first.status == "FILLED"
    assert first.shares == pytest.approx(10.0)
    assert remaining is not None
    assert remaining.remaining_shares == pytest.approx(placed.shares - 10.0)

    second, completed = oms.fill_limit_order(
        remaining,
        current_bankroll=oms.bankroll,
        current_ask=0.20,
        current_ask_size=10_000.0,
    )
    assert second is not None and second.status == "FILLED"
    assert second.shares == pytest.approx(placed.shares - 10.0)
    assert completed is None
    position = oms.get_position(placed.window_id, placed.side)
    assert position is not None
    assert position.shares == pytest.approx(placed.shares)
