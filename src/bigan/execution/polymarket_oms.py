"""Polymarket order and position manager for DEV-02 pricing signals.

Applies pre-trade spread and size gates, caps each fill at a fraction of
bankroll, then simulates a slipped fill and updates in-memory positions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from bigan.strategies.polymarket_pricing import PricingSignal, SignalDirection

DEFAULT_MAX_SINGLE_TRADE_PCT = 0.05
DEFAULT_MIN_ORDER_USD = 1.0
DEFAULT_MAX_SPREAD_ALLOWED = 0.08
DEFAULT_SLIPPAGE_TOLERANCE = 0.01
REJECT_SPREAD_TOO_WIDE = "Spread too wide"
REJECT_SIZE_BELOW_MINIMUM = "Order size below minimum threshold"


@dataclass(frozen=True, slots=True)
class Position:
    """One OMS lot on a window YES or NO token."""

    window_id: str
    symbol: str
    side: str
    shares: float
    avg_entry_price: float
    total_cost_usdc: float


@dataclass(frozen=True, slots=True)
class OrderResult:
    """Simulated order outcome for one ``process_signal`` call."""

    order_id: str
    status: str
    side: str
    shares: float
    price: float
    fee_usdc: float
    reject_reason: str | None


class PolymarketOMS:
    """Pre-trade risk, bankroll-capped sizing, and simulated fills."""

    __slots__ = (
        "max_single_trade_pct",
        "min_order_usd",
        "max_spread_allowed",
        "slippage_tolerance",
        "symbol",
        "bankroll",
        "_positions",
        "_order_seq",
    )

    def __init__(
        self,
        *,
        max_single_trade_pct: float = DEFAULT_MAX_SINGLE_TRADE_PCT,
        min_order_usd: float = DEFAULT_MIN_ORDER_USD,
        max_spread_allowed: float = DEFAULT_MAX_SPREAD_ALLOWED,
        slippage_tolerance: float = DEFAULT_SLIPPAGE_TOLERANCE,
        symbol: str = "BTC",
    ) -> None:
        cap = _finite_float("max_single_trade_pct", max_single_trade_pct)
        minimum = _finite_float("min_order_usd", min_order_usd)
        spread = _finite_float("max_spread_allowed", max_spread_allowed)
        slippage = _finite_float("slippage_tolerance", slippage_tolerance)
        if not 0.0 < cap <= 1.0:
            raise ValueError("max_single_trade_pct must be in (0, 1]")
        if minimum <= 0.0:
            raise ValueError("min_order_usd must be positive")
        if spread < 0.0:
            raise ValueError("max_spread_allowed must be non-negative")
        if slippage < 0.0:
            raise ValueError("slippage_tolerance must be non-negative")
        if not str(symbol).strip():
            raise ValueError("symbol must be non-empty")
        self.max_single_trade_pct = cap
        self.min_order_usd = minimum
        self.max_spread_allowed = spread
        self.slippage_tolerance = slippage
        self.symbol = str(symbol)
        self.bankroll = 0.0
        self._positions: dict[tuple[str, str], Position] = {}
        self._order_seq = 0

    def get_position(self, window_id: str, side: str) -> Position | None:
        return self._positions.get((window_id, side))

    def positions(self) -> tuple[Position, ...]:
        return tuple(self._positions.values())

    def process_signal(
        self,
        signal: PricingSignal,
        current_bankroll: float,
        current_bid: float,
    ) -> OrderResult | None:
        """Gate, size, and optionally fill one pricing signal.

        ``HOLD`` returns ``None`` and leaves positions unchanged. Spread and
        minimum-notional failures return a ``REJECTED`` result. A fill deducts
        cash from ``current_bankroll`` and upserts the matching ``Position``.
        """

        if signal.direction is SignalDirection.HOLD:
            return None

        bankroll = _finite_float("current_bankroll", current_bankroll)
        bid = _finite_float("current_bid", current_bid)
        ask = _finite_float("market_price", signal.market_price)
        size_pct = _finite_float("recommended_size_pct", signal.recommended_size_pct)
        if bankroll < 0.0:
            raise ValueError("current_bankroll must be non-negative")
        side = _side_from_direction(signal.direction)
        if side is None:
            return self._rejected(
                side="YES",
                reason=f"Unsupported signal direction {signal.direction!r}",
            )

        if (ask - bid) > self.max_spread_allowed:
            return self._rejected(side=side, reason=REJECT_SPREAD_TOO_WIDE)

        target_usd = bankroll * max(0.0, size_pct)
        execution_usd = min(target_usd, bankroll * self.max_single_trade_pct)
        if execution_usd < self.min_order_usd:
            return self._rejected(side=side, reason=REJECT_SIZE_BELOW_MINIMUM)

        fill_price = min(1.0, ask * (1.0 + self.slippage_tolerance))
        if fill_price <= 0.0 or not math.isfinite(fill_price):
            return self._rejected(side=side, reason="Invalid fill price")
        shares = execution_usd / fill_price
        cost = shares * fill_price
        self._apply_fill(
            window_id=signal.window_id,
            side=side,
            shares=shares,
            fill_price=fill_price,
            cost_usdc=cost,
        )
        self.bankroll = bankroll - cost
        return OrderResult(
            order_id=self._next_order_id(),
            status="FILLED",
            side=side,
            shares=shares,
            price=fill_price,
            fee_usdc=0.0,
            reject_reason=None,
        )

    def _apply_fill(
        self,
        *,
        window_id: str,
        side: str,
        shares: float,
        fill_price: float,
        cost_usdc: float,
    ) -> None:
        key = (window_id, side)
        existing = self._positions.get(key)
        if existing is None:
            self._positions[key] = Position(
                window_id=window_id,
                symbol=self.symbol,
                side=side,
                shares=shares,
                avg_entry_price=fill_price,
                total_cost_usdc=cost_usdc,
            )
            return
        total_shares = existing.shares + shares
        total_cost = existing.total_cost_usdc + cost_usdc
        avg_price = total_cost / total_shares if total_shares > 0.0 else fill_price
        self._positions[key] = Position(
            window_id=window_id,
            symbol=existing.symbol,
            side=side,
            shares=total_shares,
            avg_entry_price=avg_price,
            total_cost_usdc=total_cost,
        )

    def _rejected(self, *, side: str, reason: str) -> OrderResult:
        return OrderResult(
            order_id=self._next_order_id(),
            status="REJECTED",
            side=side,
            shares=0.0,
            price=0.0,
            fee_usdc=0.0,
            reject_reason=reason,
        )

    def _next_order_id(self) -> str:
        self._order_seq += 1
        return f"oms-{self._order_seq}"


def _side_from_direction(direction: SignalDirection) -> str | None:
    if direction is SignalDirection.BUY_YES:
        return "YES"
    if direction is SignalDirection.BUY_NO:
        return "NO"
    return None


def _finite_float(name: str, value: float) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out
