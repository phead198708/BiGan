"""Long/flat threshold strategy backtest (issue #13)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .execution import (
    NoQuoteAvailableError,
    Quote,
    SimulatedTakerTrade,
    TakerExecutionSettings,
    simulate_taker_long_settlement_trade,
    simulate_taker_long_trade,
)

DEFAULT_EDGE_THRESHOLDS: tuple[float, ...] = (0.00, 0.03, 0.05)
DEFAULT_THRESHOLDS: tuple[float, ...] = DEFAULT_EDGE_THRESHOLDS
DEFAULT_HOLD_MS = 15 * 60 * 1000


@dataclass(frozen=True, slots=True)
class PredictionSignal:
    """One model prediction available at ``ts``."""

    ts: int
    prob_up_15m: float
    source: str = "polymarket"
    source_symbol: str = ""
    target_ts: int | None = None
    market_implied_prob: float | None = None
    settlement_price: float | None = None


@dataclass(frozen=True, slots=True)
class ThresholdTrade:
    """Trade log row for one edge-threshold strategy fill."""

    threshold: float
    source: str
    source_symbol: str
    prob_up_15m: float
    market_implied_prob: float
    edge: float
    execution: SimulatedTakerTrade

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "threshold": self.threshold,
            "edge_threshold": self.threshold,
            "source": self.source,
            "source_symbol": self.source_symbol,
            "prob_up_15m": self.prob_up_15m,
            "market_implied_prob": self.market_implied_prob,
            "edge": self.edge,
            **self.execution.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ThresholdStrategySummary:
    """Aggregate performance for one edge threshold."""

    threshold: float
    edge_threshold: float
    signals_considered: int
    threshold_signals: int
    overlap_skipped: int
    unfilled_signals: int
    trade_count: int
    gross_pnl: float
    net_pnl: float
    gross_return_sum: float
    net_return_sum: float
    average_gross_return: float | None
    average_net_return: float | None
    win_rate: float | None


@dataclass(frozen=True, slots=True)
class ThresholdStrategyResult:
    """Trade log plus summary for one edge threshold."""

    threshold: float
    trades: tuple[ThresholdTrade, ...]
    summary: ThresholdStrategySummary


def run_threshold_strategy(
    *,
    signals: Sequence[PredictionSignal],
    quotes: Sequence[Quote],
    settings: TakerExecutionSettings,
    threshold: float,
    hold_ms: int = DEFAULT_HOLD_MS,
) -> ThresholdStrategyResult:
    """Run one long/flat edge-threshold strategy with non-overlapping positions."""

    if threshold < -1.0 or threshold > 1.0:
        raise ValueError("edge threshold must be in [-1, 1]")
    if hold_ms <= 0:
        raise ValueError("hold_ms must be positive")

    ordered_signals = sorted((_validate_signal(signal) for signal in signals), key=lambda row: row.ts)
    ordered_quotes = sorted(quotes, key=lambda quote: quote.ts)
    trades: list[ThresholdTrade] = []
    next_available_ts = -1
    threshold_signals = 0
    overlap_skipped = 0
    unfilled_signals = 0

    for signal in ordered_signals:
        try:
            market_implied_prob = _market_implied_probability(
                signal=signal,
                quotes=ordered_quotes,
                settings=settings,
            )
        except NoQuoteAvailableError:
            unfilled_signals += 1
            continue
        edge = signal.prob_up_15m - market_implied_prob
        if edge < threshold:
            continue
        threshold_signals += 1
        if signal.ts < next_available_ts:
            overlap_skipped += 1
            continue
        try:
            if not ordered_quotes:
                raise NoQuoteAvailableError("no quote available for execution")
            if signal.settlement_price is not None and signal.target_ts is not None:
                execution = simulate_taker_long_settlement_trade(
                    quotes=ordered_quotes,
                    decision_ts=signal.ts,
                    settlement_ts=signal.target_ts,
                    settlement_price=signal.settlement_price,
                    settings=settings,
                )
            else:
                exit_decision_ts = signal.target_ts if signal.target_ts is not None else signal.ts + hold_ms
                execution = simulate_taker_long_trade(
                    quotes=ordered_quotes,
                    decision_ts=signal.ts,
                    exit_decision_ts=exit_decision_ts,
                    settings=settings,
                )
        except NoQuoteAvailableError:
            unfilled_signals += 1
            continue
        trades.append(
            ThresholdTrade(
                threshold=threshold,
                source=signal.source,
                source_symbol=signal.source_symbol,
                prob_up_15m=signal.prob_up_15m,
                market_implied_prob=market_implied_prob,
                edge=edge,
                execution=execution,
            )
        )
        next_available_ts = execution.exit_ts

    trade_tuple = tuple(trades)
    return ThresholdStrategyResult(
        threshold=threshold,
        trades=trade_tuple,
        summary=_summarize_strategy(
            threshold=threshold,
            signals_considered=len(ordered_signals),
            threshold_signals=threshold_signals,
            overlap_skipped=overlap_skipped,
            unfilled_signals=unfilled_signals,
            trades=trade_tuple,
        ),
    )


def run_threshold_sweep(
    *,
    signals: Sequence[PredictionSignal],
    quotes: Sequence[Quote],
    settings: TakerExecutionSettings,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    hold_ms: int = DEFAULT_HOLD_MS,
) -> tuple[ThresholdStrategyResult, ...]:
    """Run the standard edge-threshold sweep."""

    return tuple(
        run_threshold_strategy(
            signals=signals,
            quotes=quotes,
            settings=settings,
            threshold=threshold,
            hold_ms=hold_ms,
        )
        for threshold in thresholds
    )


def save_threshold_strategy_outputs(
    results: Sequence[ThresholdStrategyResult],
    output_dir: Path | str,
) -> None:
    """Write ``summary.json`` plus one JSONL trade log per edge threshold."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    summary_rows = [asdict(result.summary) for result in results]
    (target / "summary.json").write_text(
        json.dumps(summary_rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    for result in results:
        suffix = str(result.threshold).replace(".", "_")
        lines = [json.dumps(trade.to_dict(), sort_keys=True) for trade in result.trades]
        (target / f"trade_log_threshold_{suffix}.jsonl").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )


def _summarize_strategy(
    *,
    threshold: float,
    signals_considered: int,
    threshold_signals: int,
    overlap_skipped: int,
    unfilled_signals: int,
    trades: Sequence[ThresholdTrade],
) -> ThresholdStrategySummary:
    trade_count = len(trades)
    gross_pnl = sum(trade.execution.gross_pnl for trade in trades)
    net_pnl = sum(trade.execution.net_pnl for trade in trades)
    gross_return_sum = sum(trade.execution.gross_return for trade in trades)
    net_return_sum = sum(trade.execution.net_return for trade in trades)
    wins = sum(1 for trade in trades if trade.execution.net_pnl > 0)
    return ThresholdStrategySummary(
        threshold=threshold,
        edge_threshold=threshold,
        signals_considered=signals_considered,
        threshold_signals=threshold_signals,
        overlap_skipped=overlap_skipped,
        unfilled_signals=unfilled_signals,
        trade_count=trade_count,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        gross_return_sum=gross_return_sum,
        net_return_sum=net_return_sum,
        average_gross_return=None if trade_count == 0 else gross_return_sum / trade_count,
        average_net_return=None if trade_count == 0 else net_return_sum / trade_count,
        win_rate=None if trade_count == 0 else wins / trade_count,
    )


def _validate_signal(signal: PredictionSignal) -> PredictionSignal:
    if signal.ts < 0:
        raise ValueError("signal ts must be non-negative")
    if signal.prob_up_15m < 0.0 or signal.prob_up_15m > 1.0:
        raise ValueError("prob_up_15m must be in [0, 1]")
    if signal.market_implied_prob is not None and (
        signal.market_implied_prob < 0.0 or signal.market_implied_prob > 1.0
    ):
        raise ValueError("market_implied_prob must be in [0, 1]")
    if signal.settlement_price is not None and (
        signal.settlement_price < 0.0 or signal.settlement_price > 1.0
    ):
        raise ValueError("settlement_price must be in [0, 1]")
    if signal.settlement_price is not None and signal.target_ts is None:
        raise ValueError("signal target_ts is required with settlement_price")
    if signal.target_ts is not None and signal.target_ts <= signal.ts:
        raise ValueError("signal target_ts must be greater than ts")
    return signal


def _market_implied_probability(
    *,
    signal: PredictionSignal,
    quotes: Sequence[Quote],
    settings: TakerExecutionSettings,
) -> float:
    if signal.market_implied_prob is not None:
        return signal.market_implied_prob
    target_ts = signal.ts + settings.latency_ms
    for quote in quotes:
        if quote.ts >= target_ts:
            return quote.ask_price
    raise NoQuoteAvailableError(f"no quote available for market implied probability at {target_ts}")
