"""DEV-02 Polymarket pricing and signal engine."""

from __future__ import annotations

import math

import pytest

from bigan.strategies.polymarket_pricing import (
    MarketWindow,
    PolymarketPricingEngine,
    SignalDirection,
)


def _window(
    window_type: str,
    *,
    start_ts_ms: int = 0,
    end_ts_ms: int | None = None,
    strike_price: float = 100_000.0,
    window_id: str = "btc-updown-test",
) -> MarketWindow:
    if end_ts_ms is None:
        end_ts_ms = 300_000 if window_type == "5m" else 900_000
    return MarketWindow(
        window_id=window_id,
        symbol="BTC",
        strike_price=strike_price,
        start_ts_ms=start_ts_ms,
        end_ts_ms=end_ts_ms,
        window_type=window_type,
    )


def _signal(
    engine: PolymarketPricingEngine,
    window: MarketWindow,
    *,
    current_ts_ms: int,
    spot_price: float,
    oracle_twap_so_far: float | None = None,
    twap_weight: float = 0.0,
    z_ofi: float = 0.0,
    volatility_annualized: float = 0.60,
    yes_ask_price: float = 0.50,
    no_ask_price: float = 0.50,
):
    if oracle_twap_so_far is None:
        oracle_twap_so_far = window.strike_price
    return engine.evaluate_signal(
        window=window,
        current_ts_ms=current_ts_ms,
        spot_price=spot_price,
        oracle_twap_so_far=oracle_twap_so_far,
        twap_weight=twap_weight,
        z_ofi=z_ofi,
        volatility_annualized=volatility_annualized,
        yes_ask_price=yes_ask_price,
        no_ask_price=no_ask_price,
    )


def test_probability_bounds_and_monotonicity() -> None:
    engine = PolymarketPricingEngine()
    strike = 100_000.0
    deep_itm = engine.calculate_probability(
        spot_price=strike * 1.10,
        effective_strike=strike,
        time_to_expiry_sec=120.0,
        volatility_annualized=0.60,
        z_ofi=0.0,
    )
    deep_otm = engine.calculate_probability(
        spot_price=strike * 0.90,
        effective_strike=strike,
        time_to_expiry_sec=120.0,
        volatility_annualized=0.60,
        z_ofi=0.0,
    )
    assert 0.0 <= deep_otm < 0.01
    assert 0.99 < deep_itm <= 1.0

    probs = [
        engine.calculate_probability(
            spot_price=strike,
            effective_strike=strike,
            time_to_expiry_sec=120.0,
            volatility_annualized=0.60,
            z_ofi=z_ofi,
        )
        for z_ofi in (-3.0, -1.5, 0.0, 1.5, 3.0)
    ]
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert probs == sorted(probs)
    assert len(set(probs)) == len(probs)


def test_twap_effective_strike_magnification() -> None:
    engine = PolymarketPricingEngine()
    strike = 100_000.0
    pulled_twap = 99_000.0
    weight = 0.8
    k_eff = engine.effective_strike(
        strike_price=strike,
        oracle_twap_so_far=pulled_twap,
        twap_weight=weight,
    )
    expected = (strike - weight * pulled_twap) / (1.0 - weight)
    assert k_eff == pytest.approx(expected)
    assert k_eff == pytest.approx(104_000.0)
    assert k_eff > strike

    window = _window("5m", strike_price=strike)
    # Minute 4 of a 5m window: w = 0.8, remaining path must chase the TWAP.
    signal = _signal(
        engine,
        window,
        current_ts_ms=240_000,
        spot_price=strike,
        oracle_twap_so_far=pulled_twap,
        twap_weight=weight,
    )
    assert signal.effective_strike == pytest.approx(104_000.0)
    assert signal.effective_strike > window.strike_price


def test_tail_cutoff_safety() -> None:
    engine = PolymarketPricingEngine()
    window = _window("5m", strike_price=100_000.0)
    signal = _signal(
        engine,
        window,
        current_ts_ms=window.end_ts_ms - 10_000,
        spot_price=120_000.0,
        yes_ask_price=0.80,
        no_ask_price=0.20,
    )
    assert signal.edge == pytest.approx(0.20, abs=1e-6)
    assert signal.direction is SignalDirection.HOLD
    assert signal.recommended_size_pct == 0.0


def test_window_edge_thresholds() -> None:
    engine = PolymarketPricingEngine()
    strike = 100_000.0
    remaining_ms = 120_000
    model_prob = engine.calculate_probability(
        spot_price=strike,
        effective_strike=strike,
        time_to_expiry_sec=remaining_ms / 1000.0,
        volatility_annualized=0.60,
        z_ofi=0.0,
    )
    yes_ask = model_prob - 0.06
    assert 0.0 < yes_ask < 1.0

    hold_5m = _signal(
        engine,
        _window("5m", end_ts_ms=300_000, strike_price=strike),
        current_ts_ms=300_000 - remaining_ms,
        spot_price=strike,
        yes_ask_price=yes_ask,
        no_ask_price=1.0 - yes_ask,
    )
    buy_15m = _signal(
        engine,
        _window("15m", end_ts_ms=900_000, strike_price=strike),
        current_ts_ms=900_000 - remaining_ms,
        spot_price=strike,
        yes_ask_price=yes_ask,
        no_ask_price=1.0 - yes_ask,
    )
    assert hold_5m.edge == pytest.approx(0.06, abs=1e-12)
    assert buy_15m.edge == pytest.approx(0.06, abs=1e-12)
    assert hold_5m.direction is SignalDirection.HOLD
    assert hold_5m.recommended_size_pct == 0.0
    assert buy_15m.direction is SignalDirection.BUY_YES
    assert buy_15m.recommended_size_pct > 0.0


def test_kelly_zero_position_on_negative_ev() -> None:
    engine = PolymarketPricingEngine()
    window = _window("15m", strike_price=100_000.0)
    signal = _signal(
        engine,
        window,
        current_ts_ms=300_000,
        spot_price=window.strike_price * 0.98,
        yes_ask_price=0.90,
        no_ask_price=0.10,
    )
    assert signal.model_prob < signal.market_price
    assert signal.edge < 0.0
    assert signal.ev < 0.0
    assert signal.direction is SignalDirection.HOLD
    assert signal.recommended_size_pct == 0.0


def test_expired_window_is_deterministic() -> None:
    engine = PolymarketPricingEngine()
    assert (
        engine.calculate_probability(
            spot_price=101.0,
            effective_strike=100.0,
            time_to_expiry_sec=0.0,
            volatility_annualized=0.60,
            z_ofi=3.0,
        )
        == 1.0
    )
    assert (
        engine.calculate_probability(
            spot_price=99.0,
            effective_strike=100.0,
            time_to_expiry_sec=-5.0,
            volatility_annualized=0.60,
            z_ofi=-3.0,
        )
        == 0.0
    )


def test_fully_sampled_twap_returns_original_strike() -> None:
    engine = PolymarketPricingEngine()
    assert engine.effective_strike(
        strike_price=100_000.0,
        oracle_twap_so_far=90_000.0,
        twap_weight=1.0,
    ) == pytest.approx(100_000.0)


def test_zero_volatility_does_not_raise() -> None:
    engine = PolymarketPricingEngine()
    itm = engine.calculate_probability(
        spot_price=110.0,
        effective_strike=100.0,
        time_to_expiry_sec=60.0,
        volatility_annualized=0.0,
        z_ofi=0.0,
    )
    otm = engine.calculate_probability(
        spot_price=90.0,
        effective_strike=100.0,
        time_to_expiry_sec=60.0,
        volatility_annualized=0.0,
        z_ofi=0.0,
    )
    assert itm == 1.0
    assert otm == 0.0
    assert math.isfinite(itm) and math.isfinite(otm)
