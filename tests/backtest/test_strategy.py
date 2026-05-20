"""Threshold strategy tests for issue #13."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.backtest import (
    DEFAULT_HOLD_MS,
    PredictionSignal,
    Quote,
    TakerExecutionSettings,
    run_threshold_strategy,
    run_threshold_sweep,
    save_threshold_strategy_outputs,
)


def _quotes() -> list[Quote]:
    return [
        Quote(ts=0, bid_price=0.49, ask_price=0.51),
        Quote(ts=DEFAULT_HOLD_MS, bid_price=0.56, ask_price=0.58),
        Quote(ts=DEFAULT_HOLD_MS + 1_000, bid_price=0.55, ask_price=0.57),
        Quote(ts=2 * DEFAULT_HOLD_MS + 10_000, bid_price=0.52, ask_price=0.54),
    ]


def test_threshold_strategy_runs_long_flat_non_overlapping_trades() -> None:
    result = run_threshold_strategy(
        signals=[
            PredictionSignal(ts=0, prob_up_15m=0.70, source_symbol="tok-up"),
            PredictionSignal(ts=60_000, prob_up_15m=0.80, source_symbol="tok-up"),
            PredictionSignal(ts=DEFAULT_HOLD_MS + 1_000, prob_up_15m=0.66, source_symbol="tok-up"),
            PredictionSignal(ts=DEFAULT_HOLD_MS + 2_000, prob_up_15m=0.40, source_symbol="tok-up"),
        ],
        quotes=_quotes(),
        settings=TakerExecutionSettings(fee_bps=0, slippage_bps=0, latency_ms=0),
        threshold=0.05,
    )

    assert len(result.trades) == 2
    assert result.summary.signals_considered == 4
    assert result.summary.threshold_signals == 3
    assert result.summary.overlap_skipped == 1
    assert result.summary.unfilled_signals == 0
    assert result.summary.trade_count == 2
    assert result.trades[0].execution.gross_entry_price == pytest.approx(0.51)
    assert result.trades[0].execution.gross_exit_price == pytest.approx(0.56)
    assert result.trades[1].execution.gross_entry_price == pytest.approx(0.57)
    assert result.trades[1].execution.gross_exit_price == pytest.approx(0.52)
    assert result.summary.gross_pnl == pytest.approx(0.0)
    assert result.summary.win_rate == pytest.approx(0.5)


def test_threshold_strategy_summary_uses_net_returns_after_costs() -> None:
    result = run_threshold_strategy(
        signals=[PredictionSignal(ts=0, prob_up_15m=0.70, source_symbol="tok-up")],
        quotes=_quotes(),
        settings=TakerExecutionSettings(fee_bps=100, slippage_bps=100, latency_ms=0),
        threshold=0.05,
    )

    trade = result.trades[0]
    assert trade.execution.gross_pnl == pytest.approx(0.05)
    assert trade.execution.net_pnl < trade.execution.gross_pnl
    assert result.summary.net_pnl == pytest.approx(trade.execution.net_pnl)
    assert result.summary.average_net_return == pytest.approx(trade.execution.net_return)


def test_threshold_strategy_counts_missing_quote_as_unfilled() -> None:
    result = run_threshold_strategy(
        signals=[PredictionSignal(ts=0, prob_up_15m=0.70, source_symbol="tok-up")],
        quotes=[Quote(ts=0, bid_price=0.49, ask_price=0.51)],
        settings=TakerExecutionSettings(fee_bps=0, slippage_bps=0, latency_ms=0),
        threshold=0.05,
    )

    assert result.summary.threshold_signals == 1
    assert result.summary.unfilled_signals == 1
    assert result.summary.trade_count == 0


def test_threshold_strategy_can_exit_at_signal_target_ts() -> None:
    result = run_threshold_strategy(
        signals=[
            PredictionSignal(
                ts=0,
                prob_up_15m=0.70,
                source_symbol="tok-up",
                target_ts=DEFAULT_HOLD_MS + 1_000,
            )
        ],
        quotes=_quotes(),
        settings=TakerExecutionSettings(fee_bps=0, slippage_bps=0, latency_ms=0),
        threshold=0.05,
        hold_ms=DEFAULT_HOLD_MS,
    )

    assert result.summary.trade_count == 1
    assert result.trades[0].execution.exit_decision_ts == DEFAULT_HOLD_MS + 1_000
    assert result.trades[0].execution.gross_exit_price == pytest.approx(0.55)


def test_threshold_strategy_can_hold_to_binary_settlement_without_exit_quote() -> None:
    result = run_threshold_strategy(
        signals=[
            PredictionSignal(
                ts=0,
                prob_up_15m=0.70,
                source_symbol="tok-up",
                target_ts=DEFAULT_HOLD_MS,
                settlement_price=1.0,
            )
        ],
        quotes=[Quote(ts=0, bid_price=0.49, ask_price=0.51)],
        settings=TakerExecutionSettings(fee_bps=0, slippage_bps=0, latency_ms=0),
        threshold=0.05,
    )

    assert result.summary.trade_count == 1
    trade = result.trades[0]
    assert trade.execution.exit_ts == DEFAULT_HOLD_MS
    assert trade.execution.gross_exit_price == pytest.approx(1.0)
    assert trade.execution.gross_pnl == pytest.approx(0.49)


def test_threshold_strategy_rejects_non_future_target_ts() -> None:
    with pytest.raises(ValueError, match="target_ts"):
        run_threshold_strategy(
            signals=[PredictionSignal(ts=1_000, prob_up_15m=0.70, target_ts=1_000)],
            quotes=_quotes(),
            settings=TakerExecutionSettings(fee_bps=0, slippage_bps=0, latency_ms=0),
            threshold=0.05,
        )


def test_threshold_strategy_surfaces_bad_quote_validation_errors() -> None:
    with pytest.raises(ValueError, match="bid_price"):
        run_threshold_strategy(
            signals=[PredictionSignal(ts=0, prob_up_15m=0.70, source_symbol="tok-up")],
            quotes=[Quote(ts=0, bid_price=0.52, ask_price=0.51)],
            settings=TakerExecutionSettings(fee_bps=0, slippage_bps=0, latency_ms=0),
            threshold=0.05,
        )


def test_threshold_sweep_defaults_to_requested_thresholds() -> None:
    results = run_threshold_sweep(
        signals=[PredictionSignal(ts=0, prob_up_15m=0.60)],
        quotes=_quotes(),
        settings=TakerExecutionSettings(fee_bps=0, slippage_bps=0, latency_ms=0),
    )

    assert [result.threshold for result in results] == [0.00, 0.03, 0.05]
    assert [result.summary.trade_count for result in results] == [1, 1, 1]


def test_threshold_strategy_outputs_trade_logs_and_summary(tmp_path: Path) -> None:
    results = run_threshold_sweep(
        signals=[PredictionSignal(ts=0, prob_up_15m=0.70, source_symbol="tok-up")],
        quotes=_quotes(),
        settings=TakerExecutionSettings(fee_bps=0, slippage_bps=0, latency_ms=0),
        thresholds=[0.05],
    )

    save_threshold_strategy_outputs(results, tmp_path)

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    trade_log = (tmp_path / "trade_log_threshold_0_05.jsonl").read_text(encoding="utf-8")
    assert summary[0]["threshold"] == 0.05
    assert summary[0]["trade_count"] == 1
    trade = json.loads(trade_log)
    assert trade["prob_up_15m"] == 0.70
    assert trade["market_implied_prob"] == pytest.approx(0.51)
    assert trade["edge"] == pytest.approx(0.19)


def test_threshold_strategy_uses_explicit_market_implied_probability() -> None:
    result = run_threshold_strategy(
        signals=[
            PredictionSignal(
                ts=0,
                prob_up_15m=0.70,
                market_implied_prob=0.68,
                source_symbol="tok-up",
            )
        ],
        quotes=_quotes(),
        settings=TakerExecutionSettings(fee_bps=0, slippage_bps=0, latency_ms=0),
        threshold=0.05,
    )

    assert result.summary.trade_count == 0
    assert result.summary.threshold_signals == 0
