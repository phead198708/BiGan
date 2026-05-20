"""Taker execution simulator tests for issue #12."""

from __future__ import annotations

import pytest

from bigan.backtest import (
    BacktestConfig,
    NoQuoteAvailableError,
    Quote,
    TakerExecutionSettings,
    simulate_taker_long_trade,
)


def test_taker_long_trade_uses_ask_entry_bid_exit_latency_costs() -> None:
    settings = TakerExecutionSettings(fee_bps=100, slippage_bps=100, latency_ms=500)
    quotes = [
        Quote(ts=1_000, bid_price=0.49, ask_price=0.51),
        Quote(ts=1_600, bid_price=0.50, ask_price=0.52),
        Quote(ts=20_000, bid_price=0.57, ask_price=0.59),
    ]

    trade = simulate_taker_long_trade(
        quotes=quotes,
        decision_ts=1_000,
        exit_decision_ts=16_000,
        settings=settings,
    )

    assert trade.entry_target_ts == 1_500
    assert trade.entry_ts == 1_600
    assert trade.exit_target_ts == 16_500
    assert trade.exit_ts == 20_000
    assert trade.gross_entry_price == pytest.approx(0.52)
    assert trade.gross_exit_price == pytest.approx(0.57)
    assert trade.entry_slippage_price == pytest.approx(0.5252)
    assert trade.exit_slippage_price == pytest.approx(0.5643)
    assert trade.entry_fee == pytest.approx(0.005252)
    assert trade.exit_fee == pytest.approx(0.005643)
    assert trade.net_entry_price == pytest.approx(0.530452)
    assert trade.net_exit_price == pytest.approx(0.558657)
    assert trade.gross_pnl == pytest.approx(0.05)
    assert trade.net_pnl == pytest.approx(0.028205)
    assert trade.gross_return == pytest.approx(0.05 / 0.52)
    assert trade.net_return == pytest.approx(0.028205 / 0.530452)


def test_taker_execution_settings_from_backtest_config() -> None:
    config = BacktestConfig.model_validate(
        {
            "schema_version": "backtest_config_v1",
            "strategy": {"long_threshold": 0.6},
            "costs": {"fee_bps": 2.0, "slippage_bps": 1.0},
            "execution": {"latency_ms": 750},
            "dataset": {"dataset_version": "features-labels-v1"},
            "model": {"model_version": "baseline-v0"},
        }
    )

    settings = TakerExecutionSettings.from_backtest_config(config)

    assert settings.fee_bps == 2.0
    assert settings.slippage_bps == 1.0
    assert settings.latency_ms == 750


def test_taker_long_trade_rejects_missing_exit_quote() -> None:
    with pytest.raises(NoQuoteAvailableError, match="exit quote"):
        simulate_taker_long_trade(
            quotes=[Quote(ts=1_000, bid_price=0.49, ask_price=0.51)],
            decision_ts=1_000,
            exit_decision_ts=2_000,
            settings=TakerExecutionSettings(fee_bps=0, slippage_bps=0, latency_ms=0),
        )


def test_taker_long_trade_rejects_unrealistic_crossed_quote() -> None:
    with pytest.raises(ValueError, match="bid_price"):
        simulate_taker_long_trade(
            quotes=[Quote(ts=1_000, bid_price=0.52, ask_price=0.51)],
            decision_ts=1_000,
            exit_decision_ts=2_000,
            settings=TakerExecutionSettings(fee_bps=0, slippage_bps=0, latency_ms=0),
        )
