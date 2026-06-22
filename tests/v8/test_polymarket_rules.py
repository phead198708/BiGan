"""Polymarket resolution-rule tests."""

from __future__ import annotations

import pytest

from bigan.v8.polymarket import (
    PolymarketAdapterError,
    build_btc15m_resolution_rule,
    normalize_btc15m_binary_market,
    resolve_polymarket_rule,
    synthetic_btc15m_market_payload,
)


def test_close_gt_open_resolves_up_only_when_close_is_greater() -> None:
    market = normalize_btc15m_binary_market(synthetic_btc15m_market_payload())
    rule = build_btc15m_resolution_rule(market, comparator="close_gt_open")

    up = resolve_polymarket_rule(
        rule,
        reference_price_start=100.0,
        reference_price_end=100.01,
    )
    tie = resolve_polymarket_rule(
        rule,
        reference_price_start=100.0,
        reference_price_end=100.0,
    )

    assert up.resolved_outcome == "UP"
    assert up.payout_up == 1.0
    assert up.payout_down == 0.0
    assert tie.resolved_outcome == "DOWN"
    assert tie.payout_up == 0.0
    assert tie.payout_down == 1.0


def test_close_gte_open_resolves_up_on_tie_and_down_below_open() -> None:
    market = normalize_btc15m_binary_market(synthetic_btc15m_market_payload())
    rule = build_btc15m_resolution_rule(market, comparator="close_gte_open")

    tie = resolve_polymarket_rule(
        rule,
        reference_price_start=100.0,
        reference_price_end=100.0,
    )
    down = resolve_polymarket_rule(
        rule,
        reference_price_start=100.0,
        reference_price_end=99.99,
    )

    assert tie.resolved_outcome == "UP"
    assert down.resolved_outcome == "DOWN"


def test_close_gt_open_rule_with_50_50_fallback_still_resolves_tie_down() -> None:
    market = normalize_btc15m_binary_market(synthetic_btc15m_market_payload())
    rule = build_btc15m_resolution_rule(
        market,
        comparator="close_gt_open",
        tie_breaker="unknown",
        unknown_50_50_enabled=True,
        raw_rule_text="If the candle cannot be resolved, market resolves 50-50.",
    )

    resolution = resolve_polymarket_rule(
        rule,
        reference_price_start=100.0,
        reference_price_end=100.0,
    )

    assert resolution.resolution_status == "normal"
    assert resolution.resolved_outcome == "DOWN"
    assert resolution.payout_up == 0.0
    assert resolution.payout_down == 1.0


def test_explicit_unknown_50_50_status_produces_half_payouts() -> None:
    market = normalize_btc15m_binary_market(synthetic_btc15m_market_payload())
    rule = build_btc15m_resolution_rule(
        market,
        comparator="close_gt_open",
        tie_breaker="unknown",
        unknown_50_50_enabled=True,
        raw_rule_text="If the candle cannot be resolved, market resolves 50-50.",
    )

    resolution = resolve_polymarket_rule(
        rule,
        reference_price_start=100.0,
        reference_price_end=100.0,
        resolution_status="unknown_50_50",
    )

    assert resolution.resolution_status == "unknown_50_50"
    assert resolution.resolved_outcome == "UNKNOWN_50_50"
    assert resolution.payout_up == 0.5
    assert resolution.payout_down == 0.5


def test_unknown_comparator_fails_closed() -> None:
    market = normalize_btc15m_binary_market(synthetic_btc15m_market_payload())

    with pytest.raises(PolymarketAdapterError, match="unknown_resolution_comparator"):
        build_btc15m_resolution_rule(
            market,
            raw_rule_text="Oracle discretion decides the market.",
        )
