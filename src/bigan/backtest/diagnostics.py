"""Backtest sanity checks for model-promotion evidence."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq

from .execution import Quote, TakerExecutionSettings
from .strategy import (
    DEFAULT_HOLD_MS,
    DEFAULT_THRESHOLDS,
    PredictionSignal,
    ThresholdTrade,
    run_threshold_strategy,
)


@dataclass(frozen=True, slots=True)
class GroupedThresholdBacktestReport:
    """Grouped long/flat edge-threshold backtest summary."""

    model_version: str
    summary: tuple[dict[str, Any], ...]
    output_dir: str
    issues: tuple[str, ...] = ()
    required_outcome_side: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "summary": list(self.summary),
            "output_dir": self.output_dir,
            "issues": list(self.issues),
            "required_outcome_side": self.required_outcome_side,
        }


def run_grouped_threshold_backtest(
    *,
    signals: Sequence[PredictionSignal],
    quotes: Sequence[Mapping[str, Any]],
    output_dir: Path | str,
    model_version: str,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    settings: TakerExecutionSettings | None = None,
    hold_ms: int = DEFAULT_HOLD_MS,
    trade_log_sample_size: int = 100,
    issues: Sequence[str] = (),
    required_outcome_side: str | None = None,
) -> GroupedThresholdBacktestReport:
    """Run edge-threshold strategy independently per ``source_symbol``.

    The basic strategy function is single-instrument. Promotion evidence for
    Polymarket outcome tokens must group by token first, otherwise one token's
    open position suppresses another token's signal.
    """

    active_settings = settings or TakerExecutionSettings(
        fee_bps=0.0,
        slippage_bps=0.0,
        latency_ms=0,
    )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    signals_by_symbol = _signals_by_symbol(signals)
    quotes_by_symbol = _quotes_by_symbol(quotes)
    summary_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        trades: list[ThresholdTrade] = []
        signals_considered = threshold_signals = overlap_skipped = unfilled_signals = 0
        symbols_with_quotes = 0
        for source_symbol, symbol_signals in sorted(signals_by_symbol.items()):
            symbol_quotes = quotes_by_symbol.get(source_symbol, ())
            signals_considered += len(symbol_signals)
            if symbol_quotes:
                symbols_with_quotes += 1
            result = run_threshold_strategy(
                signals=symbol_signals,
                quotes=symbol_quotes,
                settings=active_settings,
                threshold=threshold,
                hold_ms=hold_ms,
            )
            threshold_signals += result.summary.threshold_signals
            overlap_skipped += result.summary.overlap_skipped
            unfilled_signals += result.summary.unfilled_signals
            trades.extend(result.trades)
        summary_rows.append(
            _summarize_grouped_threshold(
                threshold=threshold,
                signals_considered=signals_considered,
                threshold_signals=threshold_signals,
                overlap_skipped=overlap_skipped,
                unfilled_signals=unfilled_signals,
                trades=trades,
                symbols_considered=len(signals_by_symbol),
                symbols_with_quotes=symbols_with_quotes,
                settings=active_settings,
                hold_ms=hold_ms,
            )
        )
        _write_trade_sample(target, threshold, trades, trade_log_sample_size)

    report = GroupedThresholdBacktestReport(
        model_version=model_version,
        summary=tuple(summary_rows),
        output_dir=str(target),
        issues=tuple(issues),
        required_outcome_side=required_outcome_side,
    )
    _write_grouped_report(report, target)
    return report


def run_oracle_label_sanity_backtest(
    *,
    dataset_dir: Path | str,
    warehouse_dir: Path | str,
    output_dir: Path | str,
    thresholds: Sequence[float] = (0.00, 0.03, 0.05),
    use_label_target_ts: bool = True,
    required_outcome_side: str | None = "UP",
) -> GroupedThresholdBacktestReport:
    """Backtest perfect profitability labels against token quotes.

    If this oracle cannot win before costs, model promotion evidence is invalid:
    the trading instrument, direction, or label/price alignment is wrong.
    """

    outcome_side = _normalise_required_outcome_side(required_outcome_side)
    signals = _oracle_label_signals(
        dataset_dir,
        use_label_target_ts=use_label_target_ts,
        required_outcome_side=outcome_side,
    )
    quotes = _warehouse_quotes(warehouse_dir, required_outcome_side=outcome_side)
    initial_issues = []
    if outcome_side is not None and not signals:
        initial_issues.append("oracle_label_required_outcome_missing")
    if outcome_side is not None and not quotes:
        initial_issues.append("oracle_quote_required_outcome_missing")
    provisional = run_grouped_threshold_backtest(
        signals=signals,
        quotes=quotes,
        output_dir=output_dir,
        model_version="oracle-label-up",
        thresholds=thresholds,
        issues=initial_issues,
        required_outcome_side=outcome_side,
    )
    issues = _merge_issues(initial_issues, _oracle_issues(provisional.summary))
    if not issues:
        return provisional
    report = GroupedThresholdBacktestReport(
        model_version=provisional.model_version,
        summary=provisional.summary,
        output_dir=provisional.output_dir,
        issues=issues,
        required_outcome_side=outcome_side,
    )
    _write_grouped_report(report, Path(output_dir))
    return report


def run_prediction_threshold_backtest(
    *,
    warehouse_dir: Path | str,
    output_dir: Path | str,
    model_version: str | None = None,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    settings: TakerExecutionSettings | None = None,
    required_outcome_side: str | None = "UP",
) -> GroupedThresholdBacktestReport:
    """Run a grouped threshold backtest from the warehouse predictions table."""

    outcome_side = _normalise_required_outcome_side(required_outcome_side)
    signals = _prediction_signals(
        warehouse_dir,
        model_version=model_version,
        required_outcome_side=outcome_side,
    )
    quotes = _warehouse_quotes(warehouse_dir, required_outcome_side=outcome_side)
    initial_issues = []
    if outcome_side is not None and not signals:
        initial_issues.append("prediction_required_outcome_missing")
    if outcome_side is not None and not quotes:
        initial_issues.append("prediction_quote_required_outcome_missing")
    return run_grouped_threshold_backtest(
        signals=signals,
        quotes=quotes,
        output_dir=output_dir,
        model_version=model_version or "predictions",
        thresholds=thresholds,
        settings=settings,
        issues=initial_issues,
        required_outcome_side=outcome_side,
    )


def _summarize_grouped_threshold(
    *,
    threshold: float,
    signals_considered: int,
    threshold_signals: int,
    overlap_skipped: int,
    unfilled_signals: int,
    trades: Sequence[ThresholdTrade],
    symbols_considered: int,
    symbols_with_quotes: int,
    settings: TakerExecutionSettings,
    hold_ms: int,
) -> dict[str, Any]:
    trade_count = len(trades)
    gross_pnl = sum(trade.execution.gross_pnl for trade in trades)
    net_pnl = sum(trade.execution.net_pnl for trade in trades)
    gross_return_sum = sum(trade.execution.gross_return for trade in trades)
    net_return_sum = sum(trade.execution.net_return for trade in trades)
    wins = sum(1 for trade in trades if trade.execution.net_pnl > 0)
    net_returns = [trade.execution.net_return for trade in trades]
    risk = _risk_metrics(trades, signals_considered=signals_considered)
    return {
        "threshold": threshold,
        "edge_threshold": threshold,
        "signals_considered": signals_considered,
        "threshold_signals": threshold_signals,
        "overlap_skipped": overlap_skipped,
        "unfilled_signals": unfilled_signals,
        "trade_count": trade_count,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "gross_return_sum": gross_return_sum,
        "net_return_sum": net_return_sum,
        "average_gross_return": None if trade_count == 0 else gross_return_sum / trade_count,
        "average_net_return": None if trade_count == 0 else net_return_sum / trade_count,
        "net_return_stddev": _sample_stddev(net_returns),
        "win_rate": None if trade_count == 0 else wins / trade_count,
        **risk,
        "symbols_considered": symbols_considered,
        "symbols_with_quotes": symbols_with_quotes,
        "hold_ms": hold_ms,
        "settings": asdict(settings),
    }


def _risk_metrics(
    trades: Sequence[ThresholdTrade],
    *,
    signals_considered: int,
) -> dict[str, Any]:
    net_returns = [trade.execution.net_return for trade in trades]
    max_drawdown = _max_drawdown(trades)
    sharpe_ratio = _sharpe_ratio(net_returns)
    sortino_ratio = _sortino_ratio(net_returns)
    turnover = None if signals_considered == 0 else len(trades) / signals_considered
    trades_per_1000_signals = None if turnover is None else turnover * 1_000.0
    trades_per_day = _trades_per_day(trades)
    concentration = _market_concentration(trades)
    return {
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": _max_drawdown_pct(trades, max_drawdown),
        "sharpe": sharpe_ratio,
        "sharpe_ratio": sharpe_ratio,
        "sortino": sortino_ratio,
        "sortino_ratio": sortino_ratio,
        "turnover": turnover,
        "turnover_trades_per_signal": turnover,
        "trades_per_1000_signals": trades_per_1000_signals,
        "turnover_trades_per_1000_signals": trades_per_1000_signals,
        "trades_per_day": trades_per_day,
        "concentration": concentration,
        "top1_market_abs_net_pnl_share": concentration["top1_abs_net_pnl_share"],
        "top5_market_abs_net_pnl_share": concentration["top5_abs_net_pnl_share"],
        "top1_market_source_symbol": concentration["top1_source_symbol"],
        "top5_market_source_symbols": concentration["top5_source_symbols"],
    }


def _max_drawdown(trades: Sequence[ThresholdTrade]) -> float:
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for trade in _trades_in_equity_order(trades):
        equity += trade.execution.net_pnl
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    return max_drawdown


def _max_drawdown_pct(trades: Sequence[ThresholdTrade], max_drawdown: float) -> float | None:
    equity = 0.0
    peak = 0.0
    for trade in _trades_in_equity_order(trades):
        equity += trade.execution.net_pnl
        peak = max(peak, equity)
    if peak <= 0.0:
        return None
    return max_drawdown / peak


def _trades_in_equity_order(trades: Sequence[ThresholdTrade]) -> tuple[ThresholdTrade, ...]:
    return tuple(
        sorted(
            trades,
            key=lambda trade: (
                trade.execution.exit_ts,
                trade.execution.entry_ts,
                trade.source_symbol,
            ),
        )
    )


def _sharpe_ratio(returns: Sequence[float]) -> float | None:
    stddev = _sample_stddev(returns)
    if stddev is None or stddev == 0.0:
        return None
    return (sum(returns) / len(returns)) / stddev


def _sortino_ratio(returns: Sequence[float]) -> float | None:
    if not returns:
        return None
    downside_squares = [min(0.0, value) ** 2 for value in returns]
    downside_deviation = math.sqrt(sum(downside_squares) / len(downside_squares))
    if downside_deviation == 0.0:
        return None
    return (sum(returns) / len(returns)) / downside_deviation


def _sample_stddev(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _trades_per_day(trades: Sequence[ThresholdTrade]) -> float | None:
    if len(trades) < 2:
        return None
    ordered = _trades_in_equity_order(trades)
    start_ts = min(trade.execution.decision_ts for trade in ordered)
    end_ts = max(trade.execution.exit_ts for trade in ordered)
    span_ms = end_ts - start_ts
    if span_ms <= 0:
        return None
    return len(trades) / (span_ms / 86_400_000.0)


def _market_concentration(trades: Sequence[ThresholdTrade]) -> dict[str, Any]:
    pnl_by_symbol: dict[str, float] = defaultdict(float)
    for trade in trades:
        pnl_by_symbol[trade.source_symbol] += trade.execution.net_pnl
    ordered = sorted(
        pnl_by_symbol.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    total_abs_net_pnl = sum(abs(net_pnl) for net_pnl in pnl_by_symbol.values())
    top1 = ordered[:1]
    top5 = ordered[:5]
    return {
        "total_abs_net_pnl": total_abs_net_pnl,
        "top1_abs_net_pnl_share": _abs_pnl_share(top1, total_abs_net_pnl),
        "top5_abs_net_pnl_share": _abs_pnl_share(top5, total_abs_net_pnl),
        "top1_source_symbol": top1[0][0] if top1 else None,
        "top5_source_symbols": [source_symbol for source_symbol, _ in top5],
    }


def _abs_pnl_share(rows: Sequence[tuple[str, float]], total_abs_net_pnl: float) -> float | None:
    if total_abs_net_pnl == 0.0:
        return None
    return sum(abs(net_pnl) for _, net_pnl in rows) / total_abs_net_pnl


def _write_trade_sample(
    output_dir: Path,
    threshold: float,
    trades: Sequence[ThresholdTrade],
    sample_size: int,
) -> None:
    suffix = str(threshold).replace(".", "_")
    lines = [
        json.dumps(trade.to_dict(), sort_keys=True)
        for trade in trades[: max(0, sample_size)]
    ]
    (output_dir / f"trade_log_sample_threshold_{suffix}.jsonl").write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )


def _write_grouped_report(report: GroupedThresholdBacktestReport, output_dir: Path) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(list(report.summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "diagnostics.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _signals_by_symbol(
    signals: Sequence[PredictionSignal],
) -> dict[str, tuple[PredictionSignal, ...]]:
    out: dict[str, list[PredictionSignal]] = defaultdict(list)
    for signal in signals:
        out[signal.source_symbol].append(signal)
    return {key: tuple(sorted(value, key=lambda signal: signal.ts)) for key, value in out.items()}


def _quotes_by_symbol(
    quotes: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[Quote, ...]]:
    out: dict[str, list[Quote]] = defaultdict(list)
    for row in quotes:
        source_symbol = row.get("source_symbol")
        if source_symbol is None:
            raise ValueError("grouped backtest quote rows require source_symbol")
        out[str(source_symbol)].append(
            Quote(
                ts=int(row["ts"]),
                bid_price=float(row["bid_price"]),
                ask_price=float(row["ask_price"]),
            )
        )
    return {key: tuple(sorted(value, key=lambda quote: quote.ts)) for key, value in out.items()}


def _oracle_label_signals(
    dataset_dir: Path | str,
    *,
    use_label_target_ts: bool,
    required_outcome_side: str | None,
) -> tuple[PredictionSignal, ...]:
    paths = _dataset_split_paths(dataset_dir)
    market_implied_sql = (
        "market_implied_prob"
        if _all_parquet_files_have_column(paths, "market_implied_prob")
        else "NULL::DOUBLE as market_implied_prob"
    )
    settlement_sql = (
        "settlement_price"
        if _all_parquet_files_have_column(paths, "settlement_price")
        else "NULL::DOUBLE as settlement_price"
    )
    label_sql = (
        "label_profit_up_15m as backtest_label"
        if _all_parquet_files_have_column(paths, "label_profit_up_15m")
        else "label_up_15m as backtest_label"
    )
    rows = _query_rows(
        f"""
        select
            feature_ts,
            target_ts,
            source,
            source_symbol,
            canonical_symbol,
            {market_implied_sql},
            {settlement_sql},
            {label_sql}
        from read_parquet({_duckdb_path_list(paths)})
        where target_ts > feature_ts
        order by feature_ts, source, source_symbol
        """
    )
    return tuple(
        PredictionSignal(
            ts=int(row["feature_ts"]),
            target_ts=int(row["target_ts"]) if use_label_target_ts else None,
            prob_up_15m=1.0 if bool(row["backtest_label"]) else 0.0,
            source=str(row["source"]),
            source_symbol=str(row["source_symbol"]),
            market_implied_prob=_optional_float(row.get("market_implied_prob")),
            settlement_price=_optional_float(row.get("settlement_price")),
        )
        for row in rows
        if _matches_outcome_side(row.get("canonical_symbol"), required_outcome_side)
    )


def _warehouse_quotes(
    warehouse_dir: Path | str,
    *,
    required_outcome_side: str | None,
) -> tuple[dict[str, Any], ...]:
    root = Path(warehouse_dir)
    path = str(root / "raw_top_of_book" / "**/*.parquet")
    rows = _query_rows(
        f"""
        select ts, source_symbol, canonical_symbol, bid_price, ask_price
        from read_parquet({_duckdb_string(path)})
        where bid_price is not null
          and ask_price is not null
          and bid_price <= ask_price
        order by source_symbol, ts
        """
    )
    return tuple(
        row
        for row in rows
        if _matches_outcome_side(row.get("canonical_symbol"), required_outcome_side)
    )


def _prediction_signals(
    warehouse_dir: Path | str,
    *,
    model_version: str | None,
    required_outcome_side: str | None,
) -> tuple[PredictionSignal, ...]:
    root = Path(warehouse_dir)
    prediction_paths = _table_parquet_paths(root, "predictions")
    if not prediction_paths:
        return ()
    label_paths = _table_parquet_paths(root, "labels_15m_v1")
    has_market_implied_prob = _all_parquet_files_have_column(
        prediction_paths,
        "market_implied_prob",
    )
    market_implied_sql = (
        "p.market_implied_prob" if has_market_implied_prob else "NULL::DOUBLE"
    )
    unaliased_market_implied_sql = (
        "market_implied_prob" if has_market_implied_prob else "NULL::DOUBLE"
    )
    params: list[Any] = []
    if label_paths:
        label_has_settlement_price = _all_parquet_files_have_column(
            label_paths,
            "settlement_price",
        )
        settlement_sql = (
            "l.settlement_price" if label_has_settlement_price else "NULL::DOUBLE"
        )
        query = f"""
            select
                p.prediction_ts,
                p.source,
                p.source_symbol,
                p.canonical_symbol,
                p.model_version,
                p.prob_up_15m,
                {market_implied_sql} as market_implied_prob,
                l.target_ts,
                {settlement_sql} as settlement_price
            from read_parquet({_duckdb_path_list(prediction_paths)}) p
            inner join read_parquet({_duckdb_path_list(label_paths)}) l
              on p.source = l.source
             and p.source_symbol = l.source_symbol
             and p.prediction_ts = l.feature_ts
            {_model_version_where_clause(model_version, params, alias="p")}
            order by p.prediction_ts, p.source, p.source_symbol
        """
    else:
        query = f"""
            select
                prediction_ts,
                source,
                source_symbol,
                canonical_symbol,
                model_version,
                prob_up_15m,
                {unaliased_market_implied_sql} as market_implied_prob,
                NULL::BIGINT as target_ts,
                NULL::DOUBLE as settlement_price
            from read_parquet({_duckdb_path_list(prediction_paths)})
            {_model_version_where_clause(model_version, params)}
            order by prediction_ts, source, source_symbol
        """
    rows = _query_rows(query, params=params)
    return tuple(
        PredictionSignal(
            ts=int(row["prediction_ts"]),
            target_ts=int(row["target_ts"]) if row.get("target_ts") is not None else None,
            prob_up_15m=float(row["prob_up_15m"]),
            source=str(row["source"]),
            source_symbol=str(row["source_symbol"]),
            market_implied_prob=_optional_float(row.get("market_implied_prob")),
            settlement_price=_optional_float(row.get("settlement_price")),
        )
        for row in rows
        if _matches_outcome_side(row.get("canonical_symbol"), required_outcome_side)
    )


def _table_parquet_paths(root: Path, table_name: str) -> list[str]:
    base = root / table_name
    if not base.exists():
        return []
    return [str(path) for path in sorted(base.rglob("*.parquet"))]


def _all_parquet_files_have_column(paths: Sequence[str], column: str) -> bool:
    if not paths:
        return False
    for path in paths:
        try:
            names = pq.ParquetFile(path).schema_arrow.names
        except (FileNotFoundError, OSError):
            return False
        if column not in names:
            return False
    return True


def _model_version_where_clause(
    model_version: str | None,
    params: list[Any],
    *,
    alias: str | None = None,
) -> str:
    if model_version is None:
        return ""
    params.append(model_version)
    prefix = "" if alias is None else f"{alias}."
    return f"where {prefix}model_version = ?"


def _dataset_split_paths(dataset_dir: Path | str) -> list[str]:
    root = Path(dataset_dir)
    paths = [str(root / f"{split}.parquet") for split in ("train", "val", "test")]
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        raise ValueError(f"dataset split not found: {missing[0]}")
    return paths


def _query_rows(sql: str, *, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
    conn = duckdb.connect()
    try:
        rows = conn.execute(sql, list(params)).fetchall()
        columns = [column[0] for column in conn.description]
        return [dict(zip(columns, row, strict=True)) for row in rows]
    finally:
        conn.close()


def _duckdb_path_list(paths: Sequence[str]) -> str:
    return "[" + ", ".join(_duckdb_string(path) for path in paths) + "]"


def _duckdb_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalise_required_outcome_side(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    if text not in {"UP", "DOWN"}:
        raise ValueError("required_outcome_side must be UP, DOWN, or None")
    return text


def _matches_outcome_side(canonical_symbol: Any, required_outcome_side: str | None) -> bool:
    if required_outcome_side is None:
        return True
    if canonical_symbol is None:
        return False
    return str(canonical_symbol).strip().upper().endswith(f":{required_outcome_side}")


def _merge_issues(
    first: Sequence[str],
    second: Sequence[str],
) -> tuple[str, ...]:
    out: list[str] = []
    for issue in (*first, *second):
        if issue not in out:
            out.append(issue)
    return tuple(out)


def _oracle_issues(summary_rows: Sequence[dict[str, Any]]) -> tuple[str, ...]:
    traded = [row for row in summary_rows if int(row.get("trade_count") or 0) > 0]
    if not traded:
        return ("oracle_label_no_trades",)
    issues: list[str] = []
    if all(float(row.get("win_rate") or 0.0) == 0.0 for row in traded):
        issues.append("oracle_label_long_up_never_wins")
    if all(float(row.get("net_pnl") or 0.0) < 0.0 for row in traded):
        issues.append("oracle_label_negative_net_pnl")
    return tuple(issues)
