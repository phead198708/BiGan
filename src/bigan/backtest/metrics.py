"""Trading-performance metrics for CLOB strategy backtests."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

MS_PER_YEAR = 365.25 * 24.0 * 3600.0 * 1000.0
_EPS = 1e-18


@dataclass(frozen=True, slots=True)
class ClosedTrade:
    """One round-trip from fill to window settlement."""

    window_id: str
    side: str
    entry_ts_ms: int
    exit_ts_ms: int
    shares: float
    entry_price: float
    exit_price: float
    pnl: float
    fee_usdc: float


@dataclass(frozen=True, slots=True)
class StrategyBacktestMetrics:
    """Structured report for one CLOB strategy replay."""

    total_return: float
    cagr: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    total_trades: int
    avg_trade_duration_ms: float
    avg_pnl_per_trade: float
    commission_paid: float


def compute_backtest_metrics(
    *,
    equity_ts_ms: Sequence[int],
    equity: Sequence[float],
    trades: Sequence[ClosedTrade],
    initial_bankroll: float,
    commission_paid: float,
) -> StrategyBacktestMetrics:
    """Score an equity curve and the closed round-trips that produced it."""

    start_cash = float(initial_bankroll)
    if start_cash <= 0.0 or not math.isfinite(start_cash):
        raise ValueError("initial_bankroll must be positive and finite")
    ts = np.asarray(equity_ts_ms, dtype=np.int64)
    eq = np.asarray(equity, dtype=np.float64)
    if ts.size != eq.size:
        raise ValueError("equity_ts_ms and equity must have the same length")
    if eq.size == 0:
        return _empty_metrics(commission_paid=float(commission_paid))

    final = float(eq[-1])
    total_return = (final / start_cash) - 1.0
    elapsed_ms = int(ts[-1] - ts[0]) if ts.size >= 2 else 0
    years = elapsed_ms / MS_PER_YEAR if elapsed_ms > 0 else 0.0
    cagr = _cagr(start_cash, final, years) if years > 0.0 and final > 0.0 else 0.0

    max_drawdown = _max_drawdown(eq)
    sharpe, sortino = _risk_adjusted(ts, eq)
    calmar = cagr / max_drawdown if max_drawdown > _EPS else 0.0

    pnls = tuple(float(trade.pnl) for trade in trades)
    n_trades = len(pnls)
    wins = sum(1 for pnl in pnls if pnl > 0.0)
    gross_profit = sum(pnl for pnl in pnls if pnl > 0.0)
    gross_loss = sum(pnl for pnl in pnls if pnl < 0.0)
    if n_trades == 0:
        win_rate = 0.0
        profit_factor = 0.0
        avg_pnl = 0.0
        avg_duration = 0.0
    else:
        win_rate = wins / n_trades
        if gross_loss < 0.0:
            profit_factor = gross_profit / abs(gross_loss)
        else:
            profit_factor = math.inf if gross_profit > 0.0 else 0.0
        avg_pnl = sum(pnls) / n_trades
        avg_duration = sum(float(t.exit_ts_ms - t.entry_ts_ms) for t in trades) / n_trades

    return StrategyBacktestMetrics(
        total_return=float(total_return),
        cagr=float(cagr),
        win_rate=float(win_rate),
        profit_factor=float(profit_factor),
        max_drawdown=float(max_drawdown),
        sharpe_ratio=float(sharpe),
        sortino_ratio=float(sortino),
        calmar_ratio=float(calmar),
        total_trades=n_trades,
        avg_trade_duration_ms=float(avg_duration),
        avg_pnl_per_trade=float(avg_pnl),
        commission_paid=float(commission_paid),
    )


def _cagr(start_cash: float, final: float, years: float) -> float:
    ratio = final / start_cash
    if ratio <= 0.0 or years <= 0.0:
        return 0.0
    scaled = math.log(ratio) / years
    if scaled > 700.0:
        return math.inf
    if scaled < -700.0:
        return -1.0
    out = math.exp(scaled) - 1.0
    return out if math.isfinite(out) else 0.0


def _empty_metrics(*, commission_paid: float) -> StrategyBacktestMetrics:
    return StrategyBacktestMetrics(
        total_return=0.0,
        cagr=0.0,
        win_rate=0.0,
        profit_factor=0.0,
        max_drawdown=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        calmar_ratio=0.0,
        total_trades=0,
        avg_trade_duration_ms=0.0,
        avg_pnl_per_trade=0.0,
        commission_paid=commission_paid,
    )


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    denom = np.maximum(peak, _EPS)
    drawdown = (peak - equity) / denom
    return float(np.max(drawdown))


def _risk_adjusted(ts_ms: np.ndarray, equity: np.ndarray) -> tuple[float, float]:
    if ts_ms.size < 2:
        return 0.0, 0.0
    prev = np.maximum(equity[:-1], _EPS)
    returns = (equity[1:] - equity[:-1]) / prev
    dt = np.diff(ts_ms.astype(np.float64))
    valid = np.isfinite(returns) & (dt > 0.0)
    if not np.any(valid):
        return 0.0, 0.0
    rets = returns[valid]
    steps = dt[valid]
    mean_dt = float(np.mean(steps))
    if mean_dt <= 0.0:
        return 0.0, 0.0
    periods = MS_PER_YEAR / mean_dt
    mean_r = float(np.mean(rets))
    std_r = float(np.std(rets, ddof=1)) if rets.size >= 2 else 0.0
    sharpe = 0.0 if std_r <= _EPS else mean_r / std_r * math.sqrt(periods)
    downside = rets[rets < 0.0]
    if downside.size < 2:
        sortino = 0.0
    else:
        down_std = float(np.std(downside, ddof=1))
        sortino = 0.0 if down_std <= _EPS else mean_r / down_std * math.sqrt(periods)
    if not math.isfinite(sharpe):
        sharpe = 0.0
    if not math.isfinite(sortino):
        sortino = 0.0
    return sharpe, sortino
