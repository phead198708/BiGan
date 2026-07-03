"""Paper-only position ledger for Polymarket outcome tokens."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

from bigan.v8.polymarket.contracts import PolymarketOutcome

PolymarketLedgerAction = Literal[
    "BUY",
    "SELL",
    "HOLD",
    "SETTLE",
    "NO_TRADE",
    "MERGE_COMPLETE_SET",
]
PolymarketLedgerOutcome = Literal["UP", "DOWN", "BOTH", "NONE"]


@dataclass(frozen=True, slots=True)
class PolymarketLedgerEvent:
    """Paper-only ledger event for one Polymarket outcome-token action."""

    ts: int
    market_id: str
    action: PolymarketLedgerAction
    outcome: PolymarketLedgerOutcome
    token_id: str | None
    qty: float
    fill_price: float
    cash_delta: float
    position_up: float
    position_down: float
    avg_entry_up: float
    avg_entry_down: float
    realized_trade_pnl: float
    unrealized_mark_pnl: float
    settlement_pnl: float
    total_pnl: float
    reason_codes: tuple[str, ...]
    fees: float = 0.0
    slippage: float = 0.0
    complete_set_pnl: float = 0.0
    condition_id: str = ""
    slug: str = ""
    paper_only: bool = True
    capital_at_risk: bool = False
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        _require_non_empty(market_id=self.market_id, action=self.action)
        if self.ts < 0:
            raise ValueError("ts must be non-negative")
        if self.action not in (
            "BUY",
            "SELL",
            "HOLD",
            "SETTLE",
            "NO_TRADE",
            "MERGE_COMPLETE_SET",
        ):
            raise ValueError("unsupported ledger action")
        if self.outcome not in ("UP", "DOWN", "BOTH", "NONE"):
            raise ValueError("unsupported ledger outcome")
        if self.action in {"BUY", "SELL"} and self.outcome not in {"UP", "DOWN"}:
            raise ValueError("BUY/SELL events require UP or DOWN outcome")
        if self.qty < 0.0 or not math.isfinite(self.qty):
            raise ValueError("qty must be non-negative and finite")
        for field_name in (
            "fill_price",
            "cash_delta",
            "position_up",
            "position_down",
            "avg_entry_up",
            "avg_entry_down",
            "realized_trade_pnl",
            "unrealized_mark_pnl",
            "settlement_pnl",
            "total_pnl",
            "fees",
            "slippage",
            "complete_set_pnl",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if self.fill_price < 0.0:
            raise ValueError("fill_price must be non-negative")
        if self.position_up < -1e-12 or self.position_down < -1e-12:
            raise ValueError("positions cannot be negative")
        if self.avg_entry_up < 0.0 or self.avg_entry_down < 0.0:
            raise ValueError("average entries cannot be negative")
        if self.fees < 0.0 or self.slippage < 0.0:
            raise ValueError("fees and slippage cannot be negative")
        if not self.reason_codes:
            raise ValueError("reason_codes are required")
        _validate_safety_boundary(self)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reason_codes"] = list(self.reason_codes)
        return payload


@dataclass(slots=True)
class _OutcomePosition:
    qty: float = 0.0
    avg_entry: float = 0.0
    mark_price: float = 0.0


class PolymarketPositionLedger:
    """Stateful paper ledger for UP/DOWN token positions."""

    def __init__(
        self,
        *,
        market_id: str,
        condition_id: str,
        slug: str,
        up_token_id: str,
        down_token_id: str,
    ) -> None:
        _require_non_empty(
            market_id=market_id,
            condition_id=condition_id,
            slug=slug,
            up_token_id=up_token_id,
            down_token_id=down_token_id,
        )
        if up_token_id == down_token_id:
            raise ValueError("UP and DOWN token ids must differ")
        self.market_id = market_id
        self.condition_id = condition_id
        self.slug = slug
        self.up_token_id = up_token_id
        self.down_token_id = down_token_id
        self._positions = {
            "UP": _OutcomePosition(),
            "DOWN": _OutcomePosition(),
        }
        self._realized_trade_pnl = 0.0
        self._settlement_pnl = 0.0
        self._complete_set_pnl = 0.0
        self._fees = 0.0
        self._slippage = 0.0
        self._events: list[PolymarketLedgerEvent] = []

    @property
    def events(self) -> tuple[PolymarketLedgerEvent, ...]:
        return tuple(self._events)

    @property
    def realized_trade_pnl(self) -> float:
        return self._realized_trade_pnl

    @property
    def settlement_pnl(self) -> float:
        return self._settlement_pnl

    @property
    def complete_set_pnl(self) -> float:
        return self._complete_set_pnl

    @property
    def fees(self) -> float:
        return self._fees

    @property
    def slippage(self) -> float:
        return self._slippage

    @property
    def unrealized_mark_pnl(self) -> float:
        return sum(
            position.qty * (position.mark_price - position.avg_entry)
            for position in self._positions.values()
        )

    @property
    def total_pnl(self) -> float:
        return (
            self._realized_trade_pnl
            + self._settlement_pnl
            + self._complete_set_pnl
            + self.unrealized_mark_pnl
            - self._fees
            - self._slippage
        )

    def buy(
        self,
        *,
        ts: int,
        outcome: PolymarketOutcome,
        qty: float,
        ask_price: float,
        fees: float = 0.0,
        slippage: float = 0.0,
        reason_codes: tuple[str, ...] = ("paper_buy",),
    ) -> PolymarketLedgerEvent:
        """Apply a paper BUY using the executable ask price."""

        _validate_qty_price(qty=qty, price=ask_price, price_name="ask_price")
        self._validate_costs(fees=fees, slippage=slippage)
        position = self._positions[outcome]
        total_qty = position.qty + qty
        if total_qty > 0.0:
            position.avg_entry = (
                position.qty * position.avg_entry + qty * ask_price
            ) / total_qty
        position.qty = total_qty
        position.mark_price = ask_price
        self._fees += fees
        self._slippage += slippage
        return self._append_event(
            ts=ts,
            action="BUY",
            outcome=outcome,
            token_id=self._token_id(outcome),
            qty=qty,
            fill_price=ask_price,
            cash_delta=-(qty * ask_price),
            fees=fees,
            slippage=slippage,
            reason_codes=reason_codes,
        )

    def sell(
        self,
        *,
        ts: int,
        outcome: PolymarketOutcome,
        qty: float,
        bid_price: float,
        fees: float = 0.0,
        slippage: float = 0.0,
        reason_codes: tuple[str, ...] = ("paper_sell",),
    ) -> PolymarketLedgerEvent:
        """Apply a paper SELL using the executable bid price."""

        _validate_qty_price(qty=qty, price=bid_price, price_name="bid_price")
        self._validate_costs(fees=fees, slippage=slippage)
        position = self._positions[outcome]
        if qty > position.qty + 1e-12:
            raise ValueError("cannot sell more than the open paper position")
        self._realized_trade_pnl += (bid_price - position.avg_entry) * qty
        position.qty = max(0.0, position.qty - qty)
        position.mark_price = bid_price
        if position.qty <= 1e-12:
            position.qty = 0.0
            position.avg_entry = 0.0
        self._fees += fees
        self._slippage += slippage
        return self._append_event(
            ts=ts,
            action="SELL",
            outcome=outcome,
            token_id=self._token_id(outcome),
            qty=qty,
            fill_price=bid_price,
            cash_delta=qty * bid_price,
            fees=fees,
            slippage=slippage,
            reason_codes=reason_codes,
        )

    def hold(
        self,
        *,
        ts: int,
        mark_up: float | None = None,
        mark_down: float | None = None,
        reason_codes: tuple[str, ...] = ("paper_hold",),
    ) -> PolymarketLedgerEvent:
        if mark_up is not None:
            _validate_price(mark_up, "mark_up")
            self._positions["UP"].mark_price = mark_up
        if mark_down is not None:
            _validate_price(mark_down, "mark_down")
            self._positions["DOWN"].mark_price = mark_down
        return self._append_event(
            ts=ts,
            action="HOLD",
            outcome="NONE",
            token_id=None,
            qty=0.0,
            fill_price=0.0,
            cash_delta=0.0,
            fees=0.0,
            slippage=0.0,
            reason_codes=reason_codes,
        )

    def no_trade(
        self,
        *,
        ts: int,
        reason_codes: tuple[str, ...] = ("paper_no_trade",),
    ) -> PolymarketLedgerEvent:
        return self._append_event(
            ts=ts,
            action="NO_TRADE",
            outcome="NONE",
            token_id=None,
            qty=0.0,
            fill_price=0.0,
            cash_delta=0.0,
            fees=0.0,
            slippage=0.0,
            reason_codes=reason_codes,
        )

    def merge_complete_sets(
        self,
        *,
        ts: int,
        reason_codes: tuple[str, ...] = ("paper_merge_complete_set",),
    ) -> PolymarketLedgerEvent:
        qty = min(self._positions["UP"].qty, self._positions["DOWN"].qty)
        if qty <= 1e-12:
            return self.hold(ts=ts, reason_codes=("no_complete_set_to_merge",))
        basis = (
            qty * self._positions["UP"].avg_entry
            + qty * self._positions["DOWN"].avg_entry
        )
        cash_delta = qty
        self._complete_set_pnl += cash_delta - basis
        for outcome in ("UP", "DOWN"):
            position = self._positions[outcome]
            position.qty = max(0.0, position.qty - qty)
            if position.qty <= 1e-12:
                position.qty = 0.0
                position.avg_entry = 0.0
        return self._append_event(
            ts=ts,
            action="MERGE_COMPLETE_SET",
            outcome="BOTH",
            token_id=None,
            qty=qty,
            fill_price=1.0,
            cash_delta=cash_delta,
            fees=0.0,
            slippage=0.0,
            reason_codes=reason_codes,
        )

    def settle(
        self,
        *,
        ts: int,
        payout_up: float,
        payout_down: float,
        reason_codes: tuple[str, ...] = ("paper_settlement",),
    ) -> PolymarketLedgerEvent:
        for field_name, payout in (
            ("payout_up", payout_up),
            ("payout_down", payout_down),
        ):
            if not 0.0 <= payout <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")
        qty_up = self._positions["UP"].qty
        qty_down = self._positions["DOWN"].qty
        basis = (
            qty_up * self._positions["UP"].avg_entry
            + qty_down * self._positions["DOWN"].avg_entry
        )
        cash_delta = qty_up * payout_up + qty_down * payout_down
        self._settlement_pnl += cash_delta - basis
        self._positions["UP"] = _OutcomePosition(mark_price=payout_up)
        self._positions["DOWN"] = _OutcomePosition(mark_price=payout_down)
        return self._append_event(
            ts=ts,
            action="SETTLE",
            outcome="BOTH",
            token_id=None,
            qty=qty_up + qty_down,
            fill_price=0.0,
            cash_delta=cash_delta,
            fees=0.0,
            slippage=0.0,
            reason_codes=reason_codes,
        )

    def position_snapshot(self) -> dict[str, Any]:
        return {
            "market_id": self.market_id,
            "condition_id": self.condition_id,
            "slug": self.slug,
            "position_up": self._positions["UP"].qty,
            "position_down": self._positions["DOWN"].qty,
            "avg_entry_up": self._positions["UP"].avg_entry,
            "avg_entry_down": self._positions["DOWN"].avg_entry,
            "realized_trade_pnl": self._realized_trade_pnl,
            "unrealized_mark_pnl": self.unrealized_mark_pnl,
            "settlement_pnl": self._settlement_pnl,
            "complete_set_pnl": self._complete_set_pnl,
            "fees": self._fees,
            "slippage": self._slippage,
            "total_pnl": self.total_pnl,
            "paper_only": True,
            "capital_at_risk": False,
            "broker_exchange_write_enabled": False,
            "live_exchange_write_enabled": False,
            "polymarket_write_enabled": False,
            "wallet_signing_enabled": False,
        }

    def _append_event(
        self,
        *,
        ts: int,
        action: PolymarketLedgerAction,
        outcome: PolymarketLedgerOutcome,
        token_id: str | None,
        qty: float,
        fill_price: float,
        cash_delta: float,
        fees: float,
        slippage: float,
        reason_codes: tuple[str, ...],
    ) -> PolymarketLedgerEvent:
        event = PolymarketLedgerEvent(
            ts=ts,
            market_id=self.market_id,
            condition_id=self.condition_id,
            slug=self.slug,
            action=action,
            outcome=outcome,
            token_id=token_id,
            qty=qty,
            fill_price=fill_price,
            cash_delta=cash_delta,
            position_up=self._positions["UP"].qty,
            position_down=self._positions["DOWN"].qty,
            avg_entry_up=self._positions["UP"].avg_entry,
            avg_entry_down=self._positions["DOWN"].avg_entry,
            realized_trade_pnl=self._realized_trade_pnl,
            unrealized_mark_pnl=self.unrealized_mark_pnl,
            settlement_pnl=self._settlement_pnl,
            complete_set_pnl=self._complete_set_pnl,
            total_pnl=self.total_pnl,
            fees=self._fees,
            slippage=self._slippage,
            reason_codes=reason_codes,
        )
        self._events.append(event)
        return event

    def _token_id(self, outcome: PolymarketOutcome) -> str:
        return self.up_token_id if outcome == "UP" else self.down_token_id

    def _validate_costs(self, *, fees: float, slippage: float) -> None:
        for field_name, value in (("fees", fees), ("slippage", slippage)):
            if value < 0.0 or not math.isfinite(value):
                raise ValueError(f"{field_name} must be non-negative and finite")


def _validate_qty_price(*, qty: float, price: float, price_name: str) -> None:
    if qty <= 0.0 or not math.isfinite(qty):
        raise ValueError("qty must be positive and finite")
    _validate_price(price, price_name)


def _validate_price(price: float, price_name: str) -> None:
    if price < 0.0 or price > 1.0 or not math.isfinite(price):
        raise ValueError(f"{price_name} must be in [0, 1]")


def _require_non_empty(**values: str) -> None:
    for field_name, value in values.items():
        if not str(value).strip():
            raise ValueError(f"{field_name} is required")


def _validate_safety_boundary(payload: Any) -> None:
    checks = {
        "paper_only": True,
        "capital_at_risk": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }
    for field_name, expected in checks.items():
        if getattr(payload, field_name) is not expected:
            raise ValueError(f"{field_name} must be {str(expected).lower()}")
