"""Polymarket outcome-token ledger tests."""

from __future__ import annotations

import pytest

from bigan.v8.polymarket import (
    PolymarketPositionLedger,
    normalize_btc15m_binary_market,
    synthetic_btc15m_market_payload,
)


def test_buy_up_uses_ask_price_not_mid_price() -> None:
    market = normalize_btc15m_binary_market(synthetic_btc15m_market_payload())
    ledger = _ledger()

    event = ledger.buy(
        ts=market.market_start_ts,
        outcome="UP",
        qty=10.0,
        ask_price=0.62,
    )

    assert event.action == "BUY"
    assert event.outcome == "UP"
    assert event.fill_price == 0.62
    assert event.cash_delta == pytest.approx(-6.2)
    assert event.avg_entry_up == 0.62
    assert event.paper_only is True
    assert event.capital_at_risk is False
    assert event.polymarket_write_enabled is False
    assert event.wallet_signing_enabled is False


def test_sell_up_uses_bid_price_and_reduces_open_position() -> None:
    ledger = _ledger()
    ledger.buy(ts=1, outcome="UP", qty=10.0, ask_price=0.60)

    event = ledger.sell(ts=2, outcome="UP", qty=4.0, bid_price=0.72)

    assert event.action == "SELL"
    assert event.fill_price == 0.72
    assert event.position_up == pytest.approx(6.0)
    assert event.realized_trade_pnl == pytest.approx((0.72 - 0.60) * 4.0)
    assert event.unrealized_mark_pnl == pytest.approx(6.0 * (0.72 - 0.60))


def test_buy_down_and_sell_down_use_down_ask_bid_prices() -> None:
    ledger = _ledger()
    buy = ledger.buy(ts=1, outcome="DOWN", qty=5.0, ask_price=0.44)
    sell = ledger.sell(ts=2, outcome="DOWN", qty=5.0, bid_price=0.48)

    assert buy.fill_price == 0.44
    assert buy.position_down == pytest.approx(5.0)
    assert sell.fill_price == 0.48
    assert sell.position_down == 0.0
    assert sell.realized_trade_pnl == pytest.approx((0.48 - 0.44) * 5.0)


def test_early_sell_rejects_more_than_open_position() -> None:
    ledger = _ledger()
    ledger.buy(ts=1, outcome="UP", qty=2.0, ask_price=0.55)

    with pytest.raises(ValueError, match="cannot sell more"):
        ledger.sell(ts=2, outcome="UP", qty=3.0, bid_price=0.60)


def test_settlement_event_clears_open_positions() -> None:
    ledger = _ledger()
    ledger.buy(ts=1, outcome="UP", qty=4.0, ask_price=0.50)
    event = ledger.settle(ts=2, payout_up=1.0, payout_down=0.0)

    assert event.action == "SETTLE"
    assert event.position_up == 0.0
    assert event.position_down == 0.0
    assert event.settlement_pnl == pytest.approx(2.0)
    assert event.total_pnl == pytest.approx(2.0)


def test_hold_and_no_trade_lifecycle_events_are_paper_only() -> None:
    ledger = _ledger()
    hold = ledger.hold(ts=1, mark_up=0.55, mark_down=0.45)
    no_trade = ledger.no_trade(ts=2)

    assert hold.action == "HOLD"
    assert no_trade.action == "NO_TRADE"
    assert hold.paper_only is True
    assert no_trade.paper_only is True
    assert hold.capital_at_risk is False
    assert no_trade.capital_at_risk is False


def test_ledger_events_never_emit_real_order_or_wallet_fields() -> None:
    ledger = _ledger()
    ledger.buy(ts=1, outcome="UP", qty=1.0, ask_price=0.5)
    ledger.no_trade(ts=2)
    forbidden = {"order_id", "wallet_signature", "private_key"}

    for event in ledger.events:
        assert forbidden.isdisjoint(event.to_dict())


def _ledger() -> PolymarketPositionLedger:
    market = normalize_btc15m_binary_market(synthetic_btc15m_market_payload())
    return PolymarketPositionLedger(
        market_id=market.market_id,
        condition_id=market.condition_id,
        slug=market.slug,
        up_token_id=market.up_token_id,
        down_token_id=market.down_token_id,
    )
