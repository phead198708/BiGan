"""Polymarket order and position manager for DEV-02 pricing signals.

Applies pre-trade spread and size gates, caps each fill at a fraction of
bankroll, then simulates a slipped fill and updates in-memory positions.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace

from bigan.strategies.polymarket_pricing import PricingSignal, SignalDirection

DEFAULT_MAX_SINGLE_TRADE_PCT = 0.05
DEFAULT_MAX_POSITION_PCT = 0.25
DEFAULT_MAX_WINDOW_EXPOSURE_PCT = 0.25
DEFAULT_SIGNAL_CACHE_SIZE = 100_000
DEFAULT_MIN_ORDER_USD = 1.0
DEFAULT_MAX_SPREAD_ALLOWED = 0.08
DEFAULT_SLIPPAGE_TOLERANCE = 0.01
REJECT_SPREAD_TOO_WIDE = "Spread too wide"
REJECT_SIZE_BELOW_MINIMUM = "Order size below minimum threshold"
REJECT_LIQUIDITY_UNAVAILABLE = "Available ask liquidity unavailable"
REJECT_UNKNOWN_LIMIT_ORDER = "Unknown or closed limit order"
REJECT_STALE_LIMIT_ORDER = "Stale limit order version"
SignalIdentity = tuple[str, int, str, float, float]


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


@dataclass(frozen=True, slots=True)
class LimitOrder:
    """Versioned handle to an OMS-owned passive order."""

    order_id: str
    version: int
    window_id: str
    placed_ts_ms: int
    side: str
    shares: float
    remaining_shares: float
    limit_price: float


class PolymarketOMS:
    """Pre-trade risk, canonical order reservations, and simulated fills."""

    __slots__ = (
        "max_single_trade_pct",
        "max_position_pct",
        "max_window_exposure_pct",
        "min_order_usd",
        "max_spread_allowed",
        "slippage_tolerance",
        "symbol",
        "bankroll",
        "_positions",
        "_open_limit_orders",
        "_order_seq",
        "_processed_signals",
        "_processed_signal_order",
        "_signal_cache_size",
    )

    def __init__(
        self,
        *,
        max_single_trade_pct: float = DEFAULT_MAX_SINGLE_TRADE_PCT,
        max_position_pct: float = DEFAULT_MAX_POSITION_PCT,
        max_window_exposure_pct: float = DEFAULT_MAX_WINDOW_EXPOSURE_PCT,
        min_order_usd: float = DEFAULT_MIN_ORDER_USD,
        max_spread_allowed: float = DEFAULT_MAX_SPREAD_ALLOWED,
        slippage_tolerance: float = DEFAULT_SLIPPAGE_TOLERANCE,
        symbol: str = "BTC",
        signal_cache_size: int = DEFAULT_SIGNAL_CACHE_SIZE,
    ) -> None:
        cap = _finite_float("max_single_trade_pct", max_single_trade_pct)
        position_cap = _finite_float("max_position_pct", max_position_pct)
        window_cap = _finite_float("max_window_exposure_pct", max_window_exposure_pct)
        minimum = _finite_float("min_order_usd", min_order_usd)
        spread = _finite_float("max_spread_allowed", max_spread_allowed)
        slippage = _finite_float("slippage_tolerance", slippage_tolerance)
        if not 0.0 < cap <= 1.0:
            raise ValueError("max_single_trade_pct must be in (0, 1]")
        if not 0.0 < position_cap <= 1.0:
            raise ValueError("max_position_pct must be in (0, 1]")
        if not 0.0 < window_cap <= 1.0:
            raise ValueError("max_window_exposure_pct must be in (0, 1]")
        if position_cap > window_cap:
            raise ValueError("max_position_pct cannot exceed max_window_exposure_pct")
        if minimum <= 0.0:
            raise ValueError("min_order_usd must be positive")
        if spread < 0.0:
            raise ValueError("max_spread_allowed must be non-negative")
        if slippage < 0.0:
            raise ValueError("slippage_tolerance must be non-negative")
        if not str(symbol).strip():
            raise ValueError("symbol must be non-empty")
        if int(signal_cache_size) < 1:
            raise ValueError("signal_cache_size must be positive")
        self.max_single_trade_pct = cap
        self.max_position_pct = position_cap
        self.max_window_exposure_pct = window_cap
        self.min_order_usd = minimum
        self.max_spread_allowed = spread
        self.slippage_tolerance = slippage
        self.symbol = str(symbol)
        self.bankroll = 0.0
        self._positions: dict[tuple[str, str], Position] = {}
        self._open_limit_orders: dict[str, LimitOrder] = {}
        self._order_seq = 0
        self._processed_signals: set[tuple[object, ...]] = set()
        self._processed_signal_order: deque[tuple[object, ...]] = deque()
        self._signal_cache_size = int(signal_cache_size)

    def get_position(self, window_id: str, side: str) -> Position | None:
        return self._positions.get((window_id, side))

    def positions(self) -> tuple[Position, ...]:
        return tuple(self._positions.values())

    def open_limit_orders(self) -> tuple[LimitOrder, ...]:
        """Return the current canonical versions of all live passive orders."""

        return tuple(self._open_limit_orders.values())

    def config_identity(self) -> dict[str, object]:
        """Return every OMS setting that can affect a simulated order."""

        return {
            "max_single_trade_pct": self.max_single_trade_pct,
            "max_position_pct": self.max_position_pct,
            "max_window_exposure_pct": self.max_window_exposure_pct,
            "min_order_usd": self.min_order_usd,
            "max_spread_allowed": self.max_spread_allowed,
            "slippage_tolerance": self.slippage_tolerance,
            "symbol": self.symbol,
            "signal_cache_size": self._signal_cache_size,
        }

    def restore_paper_state(
        self,
        *,
        current_bankroll: float,
        positions: tuple[Position, ...],
        order_sequence_floor: int = 0,
        processed_signal_identities: tuple[SignalIdentity, ...] = (),
    ) -> None:
        """Hydrate a fresh OMS from a verified paper-ledger snapshot.

        This deliberately refuses to overwrite live in-memory state. Resting
        orders are not recoverable from the BUY-only paper ledger and must be
        absent when this recovery boundary is used. Restored signal identities
        seed only the bounded OMS hot cache; durable paper replay protection is
        owned by ``PaperRunStore`` and checked by ``StrategyRunner``.
        """

        bankroll = _finite_float("current_bankroll", current_bankroll)
        if bankroll < 0.0:
            raise ValueError("current_bankroll must be non-negative")
        if self._positions or self._open_limit_orders or self._processed_signals:
            raise ValueError("OMS must be empty before paper-state recovery")
        restored: dict[tuple[str, str], Position] = {}
        for position in positions:
            if position.side not in {"YES", "NO"}:
                raise ValueError("restored position side must be YES or NO")
            values = (
                position.shares,
                position.avg_entry_price,
                position.total_cost_usdc,
            )
            if any(not math.isfinite(value) or value <= 0.0 for value in values):
                raise ValueError("restored position values must be positive and finite")
            key = (position.window_id, position.side)
            if key in restored:
                raise ValueError("duplicate restored position")
            restored[key] = position
        sequence_floor = int(order_sequence_floor)
        if sequence_floor < 0:
            raise ValueError("order_sequence_floor must be non-negative")
        validated_identities = tuple(
            _validated_signal_identity(identity)
            for identity in processed_signal_identities
        )
        self._positions = restored
        self._order_seq = max(self._order_seq, sequence_floor)
        for identity in validated_identities:
            self._remember_signal(identity)
        self.bankroll = bankroll

    def close_window(self, window_id: str, *, current_bankroll: float) -> None:
        """Release settled positions and cancel the window's open orders."""

        bankroll = _finite_float("current_bankroll", current_bankroll)
        if bankroll < 0.0:
            raise ValueError("current_bankroll must be non-negative")
        keys = [key for key in self._positions if key[0] == window_id]
        for key in keys:
            del self._positions[key]
        order_ids = [
            order_id
            for order_id, order in self._open_limit_orders.items()
            if order.window_id == window_id
        ]
        for order_id in order_ids:
            del self._open_limit_orders[order_id]
        self.bankroll = bankroll

    def process_signal(
        self,
        signal: PricingSignal,
        current_bankroll: float,
        current_bid: float,
        current_ask_size: float | None = None,
        *,
        max_fill_price: float | None = None,
        fee_bps: float = 0.0,
    ) -> OrderResult | None:
        """Gate, size, and optionally fill one pricing signal.

        ``HOLD`` returns ``None`` and leaves positions unchanged. Spread and
        minimum-notional failures return a ``REJECTED`` result. A fill deducts
        cash from ``current_bankroll`` and upserts the matching ``Position``.
        """

        if signal.direction is SignalDirection.HOLD:
            return None

        signal_key = _signal_key(signal)
        if signal_key in self._processed_signals:
            return None

        bankroll = _finite_float("current_bankroll", current_bankroll)
        bid = _finite_float("current_bid", current_bid)
        ask = _finite_float("market_price", signal.market_price)
        size_pct = _finite_float("recommended_size_pct", signal.recommended_size_pct)
        fee_rate_bps = _finite_float("fee_bps", fee_bps)
        if not 0.0 <= fee_rate_bps <= 10_000.0:
            raise ValueError("fee_bps must be in [0, 10_000]")
        fee_rate = fee_rate_bps / 10_000.0
        if bankroll < 0.0:
            raise ValueError("current_bankroll must be non-negative")
        side = _side_from_direction(signal.direction)
        if side is None:
            return self._rejected(
                side="YES",
                reason=f"Unsupported signal direction {signal.direction!r}",
            )

        if not 0.0 <= bid <= ask <= 1.0:
            return self._rejected(side=side, reason="Invalid bid/ask market")
        if (ask - bid) > self.max_spread_allowed:
            return self._rejected(side=side, reason=REJECT_SPREAD_TOO_WIDE)

        if current_ask_size is None:
            return self._rejected(side=side, reason=REJECT_LIQUIDITY_UNAVAILABLE)
        ask_size = _finite_float("current_ask_size", current_ask_size)
        if ask_size <= 0.0:
            return self._rejected(side=side, reason=REJECT_LIQUIDITY_UNAVAILABLE)

        risk_budget_usd = self._risk_budget_usd(
            window_id=signal.window_id,
            side=side,
            size_pct=size_pct,
            bankroll=bankroll,
        )
        if risk_budget_usd <= 0.0:
            return None

        fill_price = min(1.0, ask * (1.0 + self.slippage_tolerance))
        if max_fill_price is not None:
            fill_cap = _finite_float("max_fill_price", max_fill_price)
            if not ask <= fill_cap <= 1.0:
                return self._rejected(side=side, reason="Invalid maximum fill price")
            fill_price = min(fill_price, fill_cap)
        if fill_price <= 0.0 or not math.isfinite(fill_price):
            return self._rejected(side=side, reason="Invalid fill price")
        liquidity_usd = ask_size * fill_price
        execution_usd = min(
            risk_budget_usd,
            liquidity_usd,
            bankroll / (1.0 + fee_rate),
        )
        if execution_usd < self.min_order_usd:
            return self._rejected(side=side, reason=REJECT_SIZE_BELOW_MINIMUM)

        shares = execution_usd / fill_price
        cost = shares * fill_price
        fee = cost * fee_rate
        self._apply_fill(
            window_id=signal.window_id,
            side=side,
            shares=shares,
            fill_price=fill_price,
            cost_usdc=cost,
        )
        self._remember_signal(signal_key)
        self.bankroll = max(0.0, bankroll - cost - fee)
        return OrderResult(
            order_id=self._next_order_id(),
            status="FILLED",
            side=side,
            shares=shares,
            price=fill_price,
            fee_usdc=fee,
            reject_reason=None,
        )

    def prepare_limit_order(
        self,
        signal: PricingSignal,
        current_bankroll: float,
        current_bid: float,
    ) -> LimitOrder | OrderResult | None:
        """Validate and size a passive order without filling it.

        The returned order fixes its side, total shares, and limit price at
        submission time. Ask liquidity is intentionally not consumed while
        the order is resting.
        """

        if signal.direction is SignalDirection.HOLD:
            return None
        signal_key = _signal_key(signal)
        if signal_key in self._processed_signals:
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
        if not 0.0 < bid <= ask <= 1.0:
            return self._rejected(side=side, reason="Invalid bid/ask market")
        if (ask - bid) > self.max_spread_allowed:
            return self._rejected(side=side, reason=REJECT_SPREAD_TOO_WIDE)

        execution_usd = self._risk_budget_usd(
            window_id=signal.window_id,
            side=side,
            size_pct=size_pct,
            bankroll=bankroll,
        )
        if execution_usd <= 0.0:
            return None
        if execution_usd < self.min_order_usd:
            return self._rejected(side=side, reason=REJECT_SIZE_BELOW_MINIMUM)

        shares = execution_usd / bid
        order = LimitOrder(
            order_id=self._next_order_id(),
            version=0,
            window_id=signal.window_id,
            placed_ts_ms=signal.ts_ms,
            side=side,
            shares=shares,
            remaining_shares=shares,
            limit_price=bid,
        )
        self._open_limit_orders[order.order_id] = order
        self._remember_signal(signal_key)
        return order

    def cancel_limit_order(self, order: LimitOrder) -> bool:
        """Cancel an order only when the supplied handle is its current version."""

        canonical = self._open_limit_orders.get(order.order_id)
        if canonical is None or canonical != order:
            return False
        del self._open_limit_orders[order.order_id]
        return True

    def fill_limit_order(
        self,
        order: LimitOrder,
        current_bankroll: float,
        current_ask: float,
        current_ask_size: float | None,
    ) -> tuple[OrderResult | None, LimitOrder | None]:
        """Atomically fill the current version of an OMS-owned passive order."""

        bankroll = _finite_float("current_bankroll", current_bankroll)
        ask = _finite_float("current_ask", current_ask)
        if bankroll < 0.0:
            raise ValueError("current_bankroll must be non-negative")
        canonical = self._open_limit_orders.get(order.order_id)
        if canonical is None:
            return self._limit_rejected(order, REJECT_UNKNOWN_LIMIT_ORDER), None
        if canonical != order:
            return self._limit_rejected(order, REJECT_STALE_LIMIT_ORDER), canonical

        limit_price = canonical.limit_price
        remaining = canonical.remaining_shares
        if not 0.0 < ask <= 1.0 or not 0.0 < limit_price <= 1.0:
            raise ValueError("limit and ask prices must be in (0, 1]")
        if remaining <= 0.0 or remaining > canonical.shares:
            raise ValueError("remaining_shares must be in (0, shares]")
        if ask > limit_price:
            return None, canonical
        if current_ask_size is None:
            return (
                self._limit_rejected(canonical, REJECT_LIQUIDITY_UNAVAILABLE),
                canonical,
            )
        ask_size = _finite_float("current_ask_size", current_ask_size)
        if ask_size <= 0.0:
            return (
                self._limit_rejected(canonical, REJECT_LIQUIDITY_UNAVAILABLE),
                canonical,
            )

        fill_price = min(limit_price, ask * (1.0 + self.slippage_tolerance))
        fill_shares = min(remaining, ask_size, bankroll / fill_price)
        if fill_shares <= 0.0:
            return (
                self._limit_rejected(canonical, REJECT_SIZE_BELOW_MINIMUM),
                canonical,
            )
        cost = fill_shares * fill_price
        self._apply_fill(
            window_id=canonical.window_id,
            side=canonical.side,
            shares=fill_shares,
            fill_price=fill_price,
            cost_usdc=cost,
        )
        self.bankroll = bankroll - cost
        remaining_after = max(0.0, remaining - fill_shares)
        if remaining_after > 1e-12:
            next_order = replace(
                canonical,
                version=canonical.version + 1,
                remaining_shares=remaining_after,
            )
            self._open_limit_orders[canonical.order_id] = next_order
        else:
            next_order = None
            del self._open_limit_orders[canonical.order_id]
        return (
            OrderResult(
                order_id=canonical.order_id,
                status="FILLED",
                side=canonical.side,
                shares=fill_shares,
                price=fill_price,
                fee_usdc=0.0,
                reject_reason=None,
            ),
            next_order,
        )

    def _risk_budget_usd(
        self,
        *,
        window_id: str,
        side: str,
        size_pct: float,
        bankroll: float,
    ) -> float:
        capital = bankroll + sum(
            position.total_cost_usdc for position in self._positions.values()
        )
        reserved_cash = sum(
            order.remaining_shares * order.limit_price
            for order in self._open_limit_orders.values()
        )
        reserved_side_usd = sum(
            order.remaining_shares * order.limit_price
            for order in self._open_limit_orders.values()
            if order.window_id == window_id and order.side == side
        )
        reserved_window_usd = sum(
            order.remaining_shares * order.limit_price
            for order in self._open_limit_orders.values()
            if order.window_id == window_id
        )
        existing = self.get_position(window_id, side)
        existing_usd = 0.0 if existing is None else existing.total_cost_usdc
        window_usd = sum(
            position.total_cost_usdc
            for position in self._positions.values()
            if position.window_id == window_id
        )
        target_pct = min(max(0.0, size_pct), self.max_position_pct)
        remaining_target_usd = max(
            0.0,
            capital * target_pct - existing_usd - reserved_side_usd,
        )
        remaining_window_usd = max(
            0.0,
            capital * self.max_window_exposure_pct - window_usd - reserved_window_usd,
        )
        available_cash = max(0.0, bankroll - reserved_cash)
        return min(
            remaining_target_usd,
            remaining_window_usd,
            capital * self.max_single_trade_pct,
            available_cash,
        )

    def _remember_signal(self, key: tuple[object, ...]) -> None:
        self._processed_signals.add(key)
        self._processed_signal_order.append(key)
        while len(self._processed_signal_order) > self._signal_cache_size:
            expired = self._processed_signal_order.popleft()
            self._processed_signals.discard(expired)

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

    @staticmethod
    def _limit_rejected(order: LimitOrder, reason: str) -> OrderResult:
        return OrderResult(
            order_id=order.order_id,
            status="REJECTED",
            side=order.side,
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


def signal_identity(signal: PricingSignal) -> SignalIdentity:
    """Return the stable OMS idempotency identity for a pricing signal."""

    return (
        signal.window_id,
        signal.ts_ms,
        signal.direction.value,
        signal.market_price,
        signal.recommended_size_pct,
    )


def _signal_key(signal: PricingSignal) -> SignalIdentity:
    return signal_identity(signal)


def _validated_signal_identity(identity: SignalIdentity) -> SignalIdentity:
    if len(identity) != 5:
        raise ValueError("processed signal identity must contain five fields")
    window_id, ts_ms, direction, market_price, size_pct = identity
    if not str(window_id).strip() or direction not in {"BUY_YES", "BUY_NO"}:
        raise ValueError("processed signal identity is invalid")
    price = _finite_float("processed signal market_price", market_price)
    size = _finite_float("processed signal recommended_size_pct", size_pct)
    if not 0.0 < price <= 1.0 or not 0.0 <= size <= 1.0:
        raise ValueError("processed signal identity values are invalid")
    return str(window_id), int(ts_ms), direction, price, size


def _finite_float(name: str, value: float) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out
