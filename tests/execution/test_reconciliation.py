"""Stale open execution reconciliation tests for issue #85."""

from __future__ import annotations

import pytest

from bigan.execution import (
    PositionManager,
    PolymarketCashFlow,
    reconcile_stale_open_positions,
)


def _flow(
    *,
    action: str,
    usdc: float,
    tokens: float,
    side: str,
    slug: str,
    ts: int,
) -> PolymarketCashFlow:
    return PolymarketCashFlow(
        market_name="Bitcoin Up or Down - May 26, 1:45AM-2:00AM ET",
        action=action,  # type: ignore[arg-type]
        usdc_amount=usdc,
        token_amount=tokens,
        token_name=side,
        timestamp=ts,
        tx_hash=f"0x{action}{ts}",
        round_slug=slug,
        side=side,  # type: ignore[arg-type]
    )


def test_reconcile_stale_open_position_from_sell_history(tmp_path) -> None:
    manager = PositionManager(tmp_path / "positions.duckdb")
    event_id = "phase4-btc-updown-15m-1779786000-UP-10f86d62"
    manager.open_position(event_id, "BTC-15M:btc-updown-15m-1779786000:UP", "UP", 0.31, 3.22, "buy")
    flows = [
        _flow(action="BUY", usdc=1.048289, tokens=3.225805, side="UP", slug="btc-updown-15m-1779786000", ts=1),
        _flow(action="SELL", usdc=0.33214, tokens=3.22, side="UP", slug="btc-updown-15m-1779786000", ts=2),
    ]

    results = reconcile_stale_open_positions(manager, flows)

    assert len(results) == 1
    assert results[0].action == "closed_from_sell"
    assert manager.get_position(event_id).status == "closed"
    assert manager.get_position(event_id).realized_pnl == pytest.approx(-0.66606, rel=1e-3)


def test_reconcile_stale_open_position_from_redeem_history(tmp_path) -> None:
    manager = PositionManager(tmp_path / "positions.duckdb")
    event_id = "phase4-btc-updown-15m-1779774300-UP-a7fc2f63"
    manager.open_position(event_id, "BTC-15M:btc-updown-15m-1779774300:UP", "UP", 0.47, 2.127658, "buy")
    flows = [
        _flow(action="BUY", usdc=1.037089, tokens=2.127658, side="UP", slug="btc-updown-15m-1779774300", ts=1),
        PolymarketCashFlow(
            market_name="Bitcoin Up or Down - May 26, 1:45AM-2:00AM ET",
            action="REDEEM",
            usdc_amount=2.127658,
            token_amount=2.127658,
            token_name="",
            timestamp=2,
            tx_hash="0xredeem",
            round_slug="btc-updown-15m-1779774300",
            side=None,
        ),
    ]

    results = reconcile_stale_open_positions(manager, flows)

    assert results[0].action == "settled_from_redeem"
    assert manager.get_position(event_id).status == "expired"
    assert manager.get_position(event_id).settlement_result == "UP"
