"""Execution cash-leg persistence tests for issue #85."""

from __future__ import annotations

import duckdb
import pytest

from bigan.execution.cash_legs import (
    account_cash_pnl_from_legs,
    leg_from_clob_fill,
    read_execution_cash_legs,
    record_execution_cash_legs,
    signed_cash_delta,
)


def test_signed_cash_delta_matches_polymarket_semantics() -> None:
    assert signed_cash_delta("BUY", 1.05) == pytest.approx(-1.05)
    assert signed_cash_delta("SELL", 0.33) == pytest.approx(0.33)


def test_leg_from_clob_fill_tracks_theory_delta() -> None:
    leg = leg_from_clob_fill(
        event_id="phase4-btc-updown-15m-1-UP-abc",
        round_slug="btc-updown-15m-1",
        action="BUY",
        fill={"price": "0.31", "size": "3.2", "usdcAmount": "1.05", "timestamp": 100},
        order_id="order-1",
    )

    assert leg.cash_delta == pytest.approx(-1.05)
    assert leg.theoretical_usdc == pytest.approx(-0.992)
    assert leg.usdc_delta_vs_theory == pytest.approx(-0.058)


def test_execution_cash_legs_are_persisted(tmp_path) -> None:
    leg = leg_from_clob_fill(
        event_id="phase4-btc-updown-15m-1-UP-abc",
        round_slug="btc-updown-15m-1",
        action="SELL",
        fill={"price": "0.11", "size": "3.19", "usdcAmount": "0.33", "timestamp": 200},
        order_id="order-2",
        dust_token_amount=0.01,
    )
    conn = duckdb.connect()
    record_execution_cash_legs(conn, [leg])

    rows = read_execution_cash_legs(conn, event_id=leg.event_id)
    assert len(rows) == 1
    assert rows[0]["dust_token_amount"] == pytest.approx(0.01)
    assert account_cash_pnl_from_legs([leg]) == pytest.approx(0.33)
