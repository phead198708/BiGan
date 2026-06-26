"""Polymarket v8 contract tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from bigan.v8.polymarket import (
    POLYMARKET_BTC15M_HORIZON_MS,
    PolymarketAdapterError,
    PolymarketPolicySignal,
    normalize_btc15m_binary_market,
    normalize_token_snapshots,
    synthetic_btc15m_market_payload,
    synthetic_token_snapshot_rows,
)


def test_valid_btc15m_market_parses_into_contract() -> None:
    market = normalize_btc15m_binary_market(synthetic_btc15m_market_payload())

    assert market.market_family == "btc_15m_up_down"
    assert market.horizon_ms == POLYMARKET_BTC15M_HORIZON_MS
    assert market.outcome_up == "UP"
    assert market.outcome_down == "DOWN"
    assert market.up_token_id
    assert market.down_token_id
    assert market.settlement_rule == "btc_reference_price_end_gt_start_up_else_down"
    assert market.paper_only is True
    assert market.capital_at_risk is False
    assert market.polymarket_write_enabled is False
    assert market.wallet_signing_enabled is False


def test_market_safety_boundary_rejects_write_and_wallet_flags() -> None:
    payload = synthetic_btc15m_market_payload()
    payload["polymarket_write_enabled"] = True

    with pytest.raises(ValueError, match="polymarket_write_enabled"):
        normalize_btc15m_binary_market(payload)

    payload = synthetic_btc15m_market_payload()
    payload["wallet_signing_enabled"] = True

    with pytest.raises(ValueError, match="wallet_signing_enabled"):
        normalize_btc15m_binary_market(payload)


def test_token_snapshots_normalize_mid_spread_and_safety() -> None:
    market = normalize_btc15m_binary_market(synthetic_btc15m_market_payload())
    snapshots = normalize_token_snapshots(
        market=market,
        rows=synthetic_token_snapshot_rows(market),
    )

    assert {snapshot.outcome for snapshot in snapshots} == {"UP", "DOWN"}
    first = snapshots[0]
    assert first.mid_price == pytest.approx(
        (first.bid_price + first.ask_price) / 2.0
    )
    assert first.spread_bps > 0.0
    assert first.read_only is True
    assert first.write_capable is False
    assert first.paper_only is True
    assert first.capital_at_risk is False
    assert first.to_market_data().available_at_ts == first.ts


def test_token_snapshot_rejects_write_capable_rows() -> None:
    market = normalize_btc15m_binary_market(synthetic_btc15m_market_payload())
    rows = synthetic_token_snapshot_rows(market)
    rows[0]["write_capable"] = True

    with pytest.raises(ValueError, match="read-only"):
        normalize_token_snapshots(market=market, rows=rows)


def test_policy_signal_contract_bounds() -> None:
    signal = PolymarketPolicySignal(
        decision_ts=1,
        action=0.7,
        confidence=0.8,
        score=0.5,
        estimated_up_probability=0.6,
    )

    assert signal.estimated_up_probability == 0.6

    with pytest.raises(ValueError, match="estimated_up_probability"):
        replace(signal, estimated_up_probability=1.1)


def test_unknown_outcome_token_is_rejected() -> None:
    market = normalize_btc15m_binary_market(synthetic_btc15m_market_payload())
    rows = synthetic_token_snapshot_rows(market)
    rows[0]["token_id"] = "unknown-token"
    rows[0].pop("outcome")

    with pytest.raises(PolymarketAdapterError, match="unknown_token_outcome"):
        normalize_token_snapshots(market=market, rows=rows)
