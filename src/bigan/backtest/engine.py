"""Event-driven CLOB strategy backtest over historical ``MarketSnapshot`` tapes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from bigan.data.polymarket_clob import MarketSnapshot
from bigan.execution.polymarket_oms import LimitOrder, OrderResult, PolymarketOMS
from bigan.features.binance_ofi import BinanceOFICalculator, TopOfBook
from bigan.strategies.polymarket_pricing import (
    MarketWindow,
    PolymarketPricingEngine,
    PricingSignal,
    SignalDirection,
)

from .metrics import ClosedTrade, StrategyBacktestMetrics, compute_backtest_metrics

_BPS = 10_000.0


@dataclass(frozen=True, slots=True)
class StrategyBacktestParams:
    """One replay configuration for OFI, pricing, OMS, and costs."""

    ofi_window_ms: int = 60_000
    ofi_zscore_min_samples: int = 20
    ofi_ema_alpha: float = 0.2
    min_abs_z_ofi: float = 0.0
    volatility_annualized: float = 0.60
    max_spread_allowed: float = 0.08
    ofi_gamma: float = 0.0015
    min_edge_15m: float = 0.05
    min_edge_5m: float = 0.08
    kelly_fraction: float = 0.25
    tail_cutoff_ms: int = 30_000
    slippage_tolerance: float = 0.01
    max_single_trade_pct: float = 0.05
    min_order_usd: float = 1.0
    fee_bps: float = 0.0
    initial_bankroll: float = 1_000.0
    spot_price: float | None = None
    ofi_max_age_ms: int = 2_000
    execution_mode: str = "market"


@dataclass(frozen=True, slots=True)
class BacktestFill:
    """One OMS result annotated with book prices, slippage, and fees."""

    timestamp_ms: int
    window_id: str
    order: OrderResult
    ask_price: float
    bid_price: float
    slippage: float
    fee_usdc: float
    cash_after: float


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Replay output: equity curve, fills, settlements, and scored metrics."""

    params: StrategyBacktestParams
    initial_bankroll: float
    final_cash: float
    final_equity: float
    equity_ts_ms: tuple[int, ...]
    equity: tuple[float, ...]
    fills: tuple[BacktestFill, ...]
    trades: tuple[ClosedTrade, ...]
    rejected: int
    oms_calls: int
    metrics: StrategyBacktestMetrics


@dataclass(frozen=True, slots=True)
class _OpenLot:
    timestamp_ms: int
    window_id: str
    side: str
    shares: float
    entry_price: float
    fee_usdc: float


class BacktestEngine:
    """Tick-by-tick replay of OFI → pricing → OMS against a historical book.

    Market orders fill through ``PolymarketOMS`` at the slipped ask. Limit
    orders rest at the signal-time bid and become marketable when the ask
    trades through that limit. Window settlement pays $1 / $0 per share.
    """

    __slots__ = ("window", "windows", "params")

    def __init__(
        self,
        *,
        window: MarketWindow,
        windows: Mapping[str, MarketWindow] | None = None,
        params: StrategyBacktestParams | None = None,
    ) -> None:
        self.window = window
        registered = dict(windows) if windows is not None else {}
        existing = registered.get(window.window_id)
        if existing is not None and existing != window:
            raise ValueError("primary window conflicts with windows mapping")
        registered[window.window_id] = window
        for window_id, metadata in registered.items():
            if window_id != metadata.window_id:
                raise ValueError("windows mapping keys must match MarketWindow.window_id")
            if metadata.symbol != window.symbol:
                raise ValueError("all backtest windows must use the same symbol")
            if metadata.end_ts_ms <= metadata.start_ts_ms:
                raise ValueError("MarketWindow end_ts_ms must be greater than start_ts_ms")
        self.windows = registered
        self.params = params if params is not None else StrategyBacktestParams()
        mode = str(self.params.execution_mode).strip().lower()
        if mode not in {"market", "limit"}:
            raise ValueError("execution_mode must be 'market' or 'limit'")
        if float(self.params.initial_bankroll) <= 0.0:
            raise ValueError("initial_bankroll must be positive")
        if float(self.params.fee_bps) < 0.0:
            raise ValueError("fee_bps must be non-negative")
        if int(self.params.ofi_max_age_ms) < 0:
            raise ValueError("ofi_max_age_ms must be non-negative")

    def run(
        self,
        snapshots: Sequence[MarketSnapshot],
        *,
        spot_prices: Sequence[float] | None = None,
        alpha_books: Sequence[TopOfBook] | None = None,
        alpha_symbol: str | None = None,
        settlement: Mapping[str, float] | None = None,
    ) -> BacktestResult:
        """Drive event-time tapes and return scored equity.

        ``alpha_books`` is an independent, ordered Binance event stream; every
        event available by a Polymarket snapshot is ingested before its pricing
        decision, and non-empty tapes require a matching ``alpha_symbol``.
        ``settlement`` must contain a payout for every replay window.
        """

        tape = tuple(snapshots)
        n = len(tape)
        params = self.params
        if spot_prices is not None and len(spot_prices) != n:
            raise ValueError("spot_prices must match snapshots length")
        alpha_tape = tuple(alpha_books) if alpha_books is not None else ()
        if alpha_tape:
            expected_alpha_symbol = _binance_symbol(self.window.symbol)
            if not isinstance(alpha_symbol, str) or not alpha_symbol.strip():
                raise ValueError("alpha_symbol is required when alpha_books is non-empty")
            if alpha_symbol.strip().upper() != expected_alpha_symbol:
                raise ValueError(
                    f"alpha_symbol must match backtest symbol {expected_alpha_symbol}"
                )
        if any(
            current.ts_ms < previous.ts_ms
            for previous, current in zip(alpha_tape, alpha_tape[1:], strict=False)
        ):
            raise ValueError("alpha_books must be non-decreasing by event time")
        missing_windows = sorted({row.window_id for row in tape} - set(self.windows))
        if missing_windows:
            raise ValueError(f"missing MarketWindow metadata for: {missing_windows}")
        for snapshot in tape:
            metadata = self.windows[snapshot.window_id]
            if not metadata.start_ts_ms <= snapshot.timestamp_ms <= metadata.end_ts_ms:
                raise ValueError(
                    f"snapshot timestamp is outside MarketWindow {snapshot.window_id!r}"
                )
        payouts = _validated_settlement_payouts(settlement)
        missing_payouts = sorted({row.window_id for row in tape} - set(payouts))
        if missing_payouts:
            raise ValueError(f"missing settlement payout for: {missing_payouts}")
        ofi, pricing, oms = _build_stack(params, symbol=self.window.symbol)
        cash = float(params.initial_bankroll)
        yes_shares = 0.0
        no_shares = 0.0
        lots: list[_OpenLot] = []
        fills: list[BacktestFill] = []
        trades: list[ClosedTrade] = []
        settled_windows: set[str] = set()
        resting: LimitOrder | None = None
        oms_calls = 0
        rejected = 0
        commission = 0.0
        equity_ts: list[int] = []
        equity: list[float] = []
        last_window = tape[0].window_id if tape else self.window.window_id
        active_window = self.windows[last_window]
        mode = str(params.execution_mode).strip().lower()
        twap = float(active_window.strike_price)
        alpha_index = 0

        for i, snap in enumerate(tape):
            ts_ms = snap.timestamp_ms
            if snap.window_id != last_window:
                previous_end_ts = active_window.end_ts_ms
                if ts_ms < previous_end_ts:
                    raise ValueError("cannot switch windows before the active window expires")
                if last_window not in settled_windows:
                    cash, yes_shares, no_shares, commission = _settle_window(
                        window_id=last_window,
                        timestamp_ms=previous_end_ts,
                        cash=cash,
                        lots=lots,
                        trades=trades,
                        payouts=payouts,
                        commission=commission,
                    )
                    oms.close_window(last_window, current_bankroll=cash)
                    settled_windows.add(last_window)
                    _append_equity_point(equity_ts, equity, previous_end_ts, cash)
                resting = None
                last_window = snap.window_id
                active_window = self.windows[last_window]
                twap = float(active_window.strike_price)
            if snap.window_id in settled_windows:
                continue
            if ts_ms >= active_window.end_ts_ms:
                cash, yes_shares, no_shares, commission = _settle_window(
                    window_id=snap.window_id,
                    timestamp_ms=active_window.end_ts_ms,
                    cash=cash,
                    lots=lots,
                    trades=trades,
                    payouts=payouts,
                    commission=commission,
                )
                oms.close_window(snap.window_id, current_bankroll=cash)
                settled_windows.add(snap.window_id)
                resting = None
                _append_equity_point(
                    equity_ts,
                    equity,
                    active_window.end_ts_ms,
                    cash,
                )
                continue

            while alpha_index < len(alpha_tape) and alpha_tape[alpha_index].ts_ms <= ts_ms:
                alpha = alpha_tape[alpha_index]
                ofi.update_and_get_z(
                    bid_price=alpha.bid_price,
                    bid_qty=alpha.bid_qty,
                    ask_price=alpha.ask_price,
                    ask_qty=alpha.ask_qty,
                    ts_ms=alpha.ts_ms,
                )
                alpha_index += 1
            alpha_ts_ms = ofi.last_timestamp_ms
            alpha_age_ms = None if alpha_ts_ms is None else ts_ms - alpha_ts_ms
            z_ofi = (
                ofi.get_normalized_ofi()
                if alpha_age_ms is not None
                and 0 <= alpha_age_ms <= params.ofi_max_age_ms
                else 0.0
            )
            if mode == "limit" and resting is not None:
                (
                    oms_calls,
                    cash,
                    yes_shares,
                    no_shares,
                    commission,
                    resting,
                    rejected,
                ) = _maybe_fill_limit_order(
                    order=resting,
                    snapshot=snap,
                    oms=oms,
                    cash=cash,
                    yes_shares=yes_shares,
                    no_shares=no_shares,
                    lots=lots,
                    fills=fills,
                    fee_bps=params.fee_bps,
                    oms_calls=oms_calls,
                    rejected=rejected,
                    commission=commission,
                )
            spot = (
                float(spot_prices[i])
                if spot_prices is not None
                else (
                    float(params.spot_price)
                    if params.spot_price is not None
                    else float(active_window.strike_price)
                )
            )
            signal = pricing.evaluate_signal(
                window=active_window,
                current_ts_ms=ts_ms,
                spot_price=spot,
                oracle_twap_so_far=twap,
                twap_weight=0.0,
                z_ofi=z_ofi,
                volatility_annualized=params.volatility_annualized,
                yes_ask_price=snap.yes_ask,
                no_ask_price=snap.no_ask,
            )
            if (
                abs(z_ofi) >= params.min_abs_z_ofi
                and signal.direction is not SignalDirection.HOLD
            ):
                if mode == "limit":
                    side = _side_for_signal(signal)
                    if resting is not None and resting.side != side:
                        if not oms.cancel_limit_order(resting):
                            raise RuntimeError("OMS rejected canonical limit cancellation")
                        resting = None
                    if resting is None:
                        bid, ask, _ = _book_for_signal(signal, snap)
                        oms_calls += 1
                        placement = oms.prepare_limit_order(signal, cash, bid)
                        if isinstance(placement, LimitOrder):
                            resting = placement
                            if ask <= resting.limit_price:
                                (
                                    oms_calls,
                                    cash,
                                    yes_shares,
                                    no_shares,
                                    commission,
                                    resting,
                                    rejected,
                                ) = _maybe_fill_limit_order(
                                    order=resting,
                                    snapshot=snap,
                                    oms=oms,
                                    cash=cash,
                                    yes_shares=yes_shares,
                                    no_shares=no_shares,
                                    lots=lots,
                                    fills=fills,
                                    fee_bps=params.fee_bps,
                                    oms_calls=oms_calls,
                                    rejected=rejected,
                                    commission=commission,
                                )
                        elif placement is not None:
                            rejected += 1
                            _append_rejected_order(
                                result=placement,
                                snapshot=snap,
                                ask=ask,
                                bid=bid,
                                cash=cash,
                                fills=fills,
                            )
                else:
                    (
                        oms_calls,
                        cash,
                        yes_shares,
                        no_shares,
                        commission,
                        rejected,
                    ) = _execute_market_signal(
                        signal=signal,
                        snapshot=snap,
                        oms=oms,
                        cash=cash,
                        yes_shares=yes_shares,
                        no_shares=no_shares,
                        lots=lots,
                        fills=fills,
                        fee_bps=params.fee_bps,
                        oms_calls=oms_calls,
                        rejected=rejected,
                        commission=commission,
                    )
            elif signal.direction is SignalDirection.HOLD and resting is not None:
                if not oms.cancel_limit_order(resting):
                    raise RuntimeError("OMS rejected canonical limit cancellation")
                resting = None

            mtm = cash + yes_shares * snap.yes_bid + no_shares * snap.no_bid
            _append_equity_point(equity_ts, equity, ts_ms, mtm)

        if resting is not None:
            if not oms.cancel_limit_order(resting):
                raise RuntimeError("OMS rejected canonical limit cancellation")
            resting = None
        if tape and last_window not in settled_windows:
            settlement_ts_ms = active_window.end_ts_ms
            cash, yes_shares, no_shares, commission = _settle_window(
                window_id=last_window,
                timestamp_ms=settlement_ts_ms,
                cash=cash,
                lots=lots,
                trades=trades,
                payouts=payouts,
                commission=commission,
            )
            oms.close_window(last_window, current_bankroll=cash)
            _append_equity_point(equity_ts, equity, settlement_ts_ms, cash)

        metrics = compute_backtest_metrics(
            equity_ts_ms=equity_ts,
            equity=equity,
            trades=trades,
            initial_bankroll=params.initial_bankroll,
            commission_paid=commission,
        )
        return BacktestResult(
            params=params,
            initial_bankroll=float(params.initial_bankroll),
            final_cash=float(cash),
            final_equity=float(equity[-1]) if equity else float(cash),
            equity_ts_ms=tuple(equity_ts),
            equity=tuple(equity),
            fills=tuple(fills),
            trades=tuple(trades),
            rejected=rejected,
            oms_calls=oms_calls,
            metrics=metrics,
        )


def _append_equity_point(
    timestamps: list[int],
    values: list[float],
    timestamp_ms: int,
    equity: float,
) -> None:
    ts_ms = int(timestamp_ms)
    value = float(equity)
    if timestamps and ts_ms < timestamps[-1]:
        raise ValueError("equity timestamps must be non-decreasing")
    if timestamps and ts_ms == timestamps[-1]:
        values[-1] = value
        return
    timestamps.append(ts_ms)
    values.append(value)


def _build_stack(
    params: StrategyBacktestParams,
    *,
    symbol: str,
) -> tuple[BinanceOFICalculator, PolymarketPricingEngine, PolymarketOMS]:
    ofi = BinanceOFICalculator(
        ema_alpha=params.ofi_ema_alpha,
        window_ms=params.ofi_window_ms,
        zscore_min_samples=params.ofi_zscore_min_samples,
        symbol=_binance_symbol(symbol),
    )
    pricing = PolymarketPricingEngine(
        ofi_gamma=params.ofi_gamma,
        min_edge_5m=params.min_edge_5m,
        min_edge_15m=params.min_edge_15m,
        kelly_fraction=params.kelly_fraction,
        tail_cutoff_ms=params.tail_cutoff_ms,
    )
    oms = PolymarketOMS(
        max_single_trade_pct=params.max_single_trade_pct,
        min_order_usd=params.min_order_usd,
        max_spread_allowed=params.max_spread_allowed,
        slippage_tolerance=params.slippage_tolerance,
        symbol=symbol,
    )
    return ofi, pricing, oms


def _binance_symbol(market_symbol: str) -> str:
    symbol = str(market_symbol).strip().upper()
    return symbol if symbol.endswith("USDT") else f"{symbol}USDT"


def _execute_market_signal(
    *,
    signal: PricingSignal,
    snapshot: MarketSnapshot,
    oms: PolymarketOMS,
    cash: float,
    yes_shares: float,
    no_shares: float,
    lots: list[_OpenLot],
    fills: list[BacktestFill],
    fee_bps: float,
    oms_calls: int,
    rejected: int,
    commission: float,
) -> tuple[int, float, float, float, float, int]:
    bid, ask, ask_size = _book_for_signal(signal, snapshot)
    oms_calls += 1
    result = oms.process_signal(
        signal,
        cash,
        bid,
        ask_size,
    )
    if result is None:
        return oms_calls, cash, yes_shares, no_shares, commission, rejected
    if result.status != "FILLED":
        rejected += 1
        _append_rejected_order(
            result=result,
            snapshot=snapshot,
            ask=ask,
            bid=bid,
            cash=cash,
            fills=fills,
        )
        return (
            oms_calls,
            cash,
            yes_shares,
            no_shares,
            commission,
            rejected,
        )
    cash, yes_shares, no_shares, commission = _record_filled_order(
        result=result,
        snapshot=snapshot,
        oms=oms,
        cash=cash,
        yes_shares=yes_shares,
        no_shares=no_shares,
        lots=lots,
        fills=fills,
        ask=ask,
        bid=bid,
        fee_bps=fee_bps,
        commission=commission,
    )
    return oms_calls, cash, yes_shares, no_shares, commission, rejected


def _maybe_fill_limit_order(
    *,
    order: LimitOrder,
    snapshot: MarketSnapshot,
    oms: PolymarketOMS,
    cash: float,
    yes_shares: float,
    no_shares: float,
    lots: list[_OpenLot],
    fills: list[BacktestFill],
    fee_bps: float,
    oms_calls: int,
    rejected: int,
    commission: float,
) -> tuple[int, float, float, float, float, LimitOrder | None, int]:
    bid, ask, ask_size = _book_for_limit_order(order, snapshot)
    if ask > order.limit_price:
        return oms_calls, cash, yes_shares, no_shares, commission, order, rejected
    oms_calls += 1
    result, next_order = oms.fill_limit_order(order, cash, ask, ask_size)
    if result is None:
        return oms_calls, cash, yes_shares, no_shares, commission, next_order, rejected
    if result.status != "FILLED":
        rejected += 1
        _append_rejected_order(
            result=result,
            snapshot=snapshot,
            ask=ask,
            bid=bid,
            cash=cash,
            fills=fills,
        )
        return (
            oms_calls,
            cash,
            yes_shares,
            no_shares,
            commission,
            next_order,
            rejected,
        )
    cash, yes_shares, no_shares, commission = _record_filled_order(
        result=result,
        snapshot=snapshot,
        oms=oms,
        cash=cash,
        yes_shares=yes_shares,
        no_shares=no_shares,
        lots=lots,
        fills=fills,
        ask=ask,
        bid=bid,
        fee_bps=fee_bps,
        commission=commission,
    )
    return (
        oms_calls,
        cash,
        yes_shares,
        no_shares,
        commission,
        next_order,
        rejected,
    )


def _record_filled_order(
    *,
    result: OrderResult,
    snapshot: MarketSnapshot,
    oms: PolymarketOMS,
    cash: float,
    yes_shares: float,
    no_shares: float,
    lots: list[_OpenLot],
    fills: list[BacktestFill],
    ask: float,
    bid: float,
    fee_bps: float,
    commission: float,
) -> tuple[float, float, float, float]:
    notional = result.shares * result.price
    fee = notional * float(fee_bps) / _BPS
    cash = oms.bankroll - fee
    oms.bankroll = cash
    commission += fee
    if result.side == "YES":
        yes_shares += result.shares
    else:
        no_shares += result.shares
    lots.append(
        _OpenLot(
            timestamp_ms=snapshot.timestamp_ms,
            window_id=snapshot.window_id,
            side=result.side,
            shares=result.shares,
            entry_price=result.price,
            fee_usdc=fee,
        )
    )
    fills.append(
        BacktestFill(
            timestamp_ms=snapshot.timestamp_ms,
            window_id=snapshot.window_id,
            order=result,
            ask_price=ask,
            bid_price=bid,
            slippage=result.price - ask,
            fee_usdc=fee,
            cash_after=cash,
        )
    )
    return cash, yes_shares, no_shares, commission


def _append_rejected_order(
    *,
    result: OrderResult,
    snapshot: MarketSnapshot,
    ask: float,
    bid: float,
    cash: float,
    fills: list[BacktestFill],
) -> None:
    fills.append(
        BacktestFill(
            timestamp_ms=snapshot.timestamp_ms,
            window_id=snapshot.window_id,
            order=result,
            ask_price=ask,
            bid_price=bid,
            slippage=0.0,
            fee_usdc=0.0,
            cash_after=cash,
        )
    )


def _book_for_signal(
    signal: PricingSignal,
    snapshot: MarketSnapshot,
) -> tuple[float, float, float]:
    if signal.direction is SignalDirection.BUY_NO:
        return snapshot.no_bid, snapshot.no_ask, snapshot.no_ask_size
    return snapshot.yes_bid, snapshot.yes_ask, snapshot.yes_ask_size


def _book_for_limit_order(
    order: LimitOrder,
    snapshot: MarketSnapshot,
) -> tuple[float, float, float]:
    if order.side == "NO":
        return snapshot.no_bid, snapshot.no_ask, snapshot.no_ask_size
    return snapshot.yes_bid, snapshot.yes_ask, snapshot.yes_ask_size


def _side_for_signal(signal: PricingSignal) -> str:
    if signal.direction is SignalDirection.BUY_NO:
        return "NO"
    if signal.direction is SignalDirection.BUY_YES:
        return "YES"
    raise ValueError("HOLD has no executable side")


def _settle_window(
    *,
    window_id: str,
    timestamp_ms: int,
    cash: float,
    lots: list[_OpenLot],
    trades: list[ClosedTrade],
    payouts: Mapping[str, float],
    commission: float,
) -> tuple[float, float, float, float]:
    yes_payout = payouts[window_id]
    no_payout = 1.0 - yes_payout
    for lot in lots:
        exit_px = yes_payout if lot.side == "YES" else no_payout
        proceeds = lot.shares * exit_px
        cash += proceeds
        trades.append(
            ClosedTrade(
                window_id=lot.window_id,
                side=lot.side,
                entry_ts_ms=lot.timestamp_ms,
                exit_ts_ms=timestamp_ms,
                shares=lot.shares,
                entry_price=lot.entry_price,
                exit_price=exit_px,
                pnl=proceeds - lot.shares * lot.entry_price - lot.fee_usdc,
                fee_usdc=lot.fee_usdc,
            )
        )
    lots.clear()
    return cash, 0.0, 0.0, commission


def _validated_settlement_payouts(
    settlement: Mapping[str, float] | None,
) -> dict[str, float]:
    if settlement is None:
        return {}
    payouts: dict[str, float] = {}
    for window_id, value in settlement.items():
        try:
            payout = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"settlement payout for {window_id!r} must be finite and in [0, 1]"
            ) from exc
        if not math.isfinite(payout) or not 0.0 <= payout <= 1.0:
            raise ValueError(
                f"settlement payout for {window_id!r} must be finite and in [0, 1]"
            )
        payouts[window_id] = payout
    return payouts
