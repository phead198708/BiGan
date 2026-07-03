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
    outcome_side: str | None = None
    family_key: str | None = None


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
    outcome_side: str | None = None
    realized_label: bool | None = None

    def to_dict(self) -> dict[str, float | int | str | bool]:
        row = {
            "threshold": self.threshold,
            "edge_threshold": self.threshold,
            "source": self.source,
            "source_symbol": self.source_symbol,
            "prob_up_15m": self.prob_up_15m,
            "market_implied_prob": self.market_implied_prob,
            "edge": self.edge,
            **self.execution.to_dict(),
        }
        if self.outcome_side is not None:
            row["outcome_side"] = self.outcome_side
        if self.realized_label is not None:
            row["realized_label"] = self.realized_label
        return row


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
    brier_score: float | None
    brier_sample_count: int


@dataclass(frozen=True, slots=True)
class ThresholdStrategyResult:
    """Trade log plus summary for one edge threshold."""

    threshold: float
    trades: tuple[ThresholdTrade, ...]
    summary: ThresholdStrategySummary


@dataclass(frozen=True, slots=True)
class PerFamilyThresholdSelection:
    """Best edge threshold and diagnostics for one market family."""

    family_key: str
    selected_threshold: float | None
    selected_net_pnl: float | None
    selected_trade_count: int
    selected_expected_value: float | None
    selected_trades_per_day: float | None
    eligible_thresholds: tuple[float, ...]
    summaries: tuple[ThresholdStrategySummary, ...]


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
                outcome_side=signal.outcome_side,
                realized_label=(
                    None if signal.settlement_price is None else signal.settlement_price >= 0.5
                ),
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


def run_per_family_threshold_search(
    *,
    signals: Sequence[PredictionSignal],
    quotes: Sequence[Quote],
    settings: TakerExecutionSettings,
    thresholds: Sequence[float] = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
    hold_ms: int = DEFAULT_HOLD_MS,
    min_expected_value: float = 0.0,
    max_trades_per_family_per_day: float | None = None,
) -> tuple[PerFamilyThresholdSelection, ...]:
    """Search edge thresholds independently by market family."""

    groups: dict[str, list[PredictionSignal]] = {}
    for signal in signals:
        key = signal.family_key or _family_key_from_signal(signal)
        groups.setdefault(key, []).append(signal)

    selections: list[PerFamilyThresholdSelection] = []
    for family_key, family_signals in sorted(groups.items()):
        results = run_threshold_sweep(
            signals=family_signals,
            quotes=quotes,
            settings=settings,
            thresholds=thresholds,
            hold_ms=hold_ms,
        )
        eligible = [
            result
            for result in results
            if _threshold_summary_eligible(
                result.summary,
                signals=family_signals,
                min_expected_value=min_expected_value,
                max_trades_per_day=max_trades_per_family_per_day,
            )
        ]
        selected = max(eligible, key=lambda result: result.summary.net_pnl) if eligible else None
        selected_summary = None if selected is None else selected.summary
        selections.append(
            PerFamilyThresholdSelection(
                family_key=family_key,
                selected_threshold=None if selected_summary is None else selected_summary.threshold,
                selected_net_pnl=None if selected_summary is None else selected_summary.net_pnl,
                selected_trade_count=0 if selected_summary is None else selected_summary.trade_count,
                selected_expected_value=(
                    None
                    if selected_summary is None or selected_summary.trade_count == 0
                    else selected_summary.net_pnl / selected_summary.trade_count
                ),
                selected_trades_per_day=(
                    None
                    if selected_summary is None
                    else _trades_per_day(selected_summary.trade_count, family_signals)
                ),
                eligible_thresholds=tuple(result.threshold for result in eligible),
                summaries=tuple(result.summary for result in results),
            )
        )
    return tuple(selections)


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


def _threshold_summary_eligible(
    summary: ThresholdStrategySummary,
    *,
    signals: Sequence[PredictionSignal],
    min_expected_value: float,
    max_trades_per_day: float | None,
) -> bool:
    if summary.trade_count <= 0:
        return False
    expected_value = summary.net_pnl / summary.trade_count
    if expected_value < min_expected_value:
        return False
    trades_per_day = _trades_per_day(summary.trade_count, signals)
    return max_trades_per_day is None or trades_per_day <= max_trades_per_day


def _trades_per_day(trade_count: int, signals: Sequence[PredictionSignal]) -> float:
    if trade_count <= 0 or not signals:
        return 0.0
    min_ts = min(signal.ts for signal in signals)
    max_ts = max(signal.ts for signal in signals)
    span_days = max(1.0 / 24.0, (max_ts - min_ts) / 86_400_000)
    return trade_count / span_days


def _family_key_from_signal(signal: PredictionSignal) -> str:
    if signal.outcome_side:
        symbol = signal.outcome_side.upper()
    else:
        symbol = signal.source_symbol or signal.source or "unknown"
    return symbol


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
    brier_components = [
        (trade.prob_up_15m - (1.0 if trade.realized_label else 0.0)) ** 2
        for trade in trades
        if trade.realized_label is not None
    ]
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
        brier_score=None if not brier_components else sum(brier_components) / len(brier_components),
        brier_sample_count=len(brier_components),
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
