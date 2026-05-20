"""Conservative taker execution simulator (issue #12)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .config import BacktestConfig


class NoQuoteAvailableError(LookupError):
    """Raised when no quote exists for a requested execution timestamp."""


@dataclass(frozen=True, slots=True)
class TakerExecutionSettings:
    """Cost and latency assumptions for taker fills."""

    fee_bps: float
    slippage_bps: float
    latency_ms: int

    @classmethod
    def from_backtest_config(cls, config: BacktestConfig) -> TakerExecutionSettings:
        return cls(
            fee_bps=config.costs.fee_bps,
            slippage_bps=config.costs.slippage_bps,
            latency_ms=config.execution.latency_ms,
        )


@dataclass(frozen=True, slots=True)
class Quote:
    """Top-of-book quote used by the simulator."""

    ts: int
    bid_price: float
    ask_price: float


@dataclass(frozen=True, slots=True)
class SimulatedTakerTrade:
    """One long/flat round trip with conservative taker economics."""

    decision_ts: int
    entry_target_ts: int
    entry_ts: int
    exit_decision_ts: int
    exit_target_ts: int
    exit_ts: int
    gross_entry_price: float
    gross_exit_price: float
    entry_slippage_price: float
    exit_slippage_price: float
    entry_fee: float
    exit_fee: float
    net_entry_price: float
    net_exit_price: float
    gross_pnl: float
    net_pnl: float
    gross_return: float
    net_return: float
    fee_bps: float
    slippage_bps: float
    latency_ms: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "decision_ts": self.decision_ts,
            "entry_target_ts": self.entry_target_ts,
            "entry_ts": self.entry_ts,
            "exit_decision_ts": self.exit_decision_ts,
            "exit_target_ts": self.exit_target_ts,
            "exit_ts": self.exit_ts,
            "gross_entry_price": self.gross_entry_price,
            "gross_exit_price": self.gross_exit_price,
            "entry_slippage_price": self.entry_slippage_price,
            "exit_slippage_price": self.exit_slippage_price,
            "entry_fee": self.entry_fee,
            "exit_fee": self.exit_fee,
            "net_entry_price": self.net_entry_price,
            "net_exit_price": self.net_exit_price,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "gross_return": self.gross_return,
            "net_return": self.net_return,
            "fee_bps": self.fee_bps,
            "slippage_bps": self.slippage_bps,
            "latency_ms": self.latency_ms,
        }


def simulate_taker_long_trade(
    *,
    quotes: Sequence[Quote],
    decision_ts: int,
    exit_decision_ts: int,
    settings: TakerExecutionSettings,
) -> SimulatedTakerTrade:
    """Simulate a long-UP taker entry and taker exit.

    Entry uses the first quote at or after ``decision_ts + latency_ms`` and
    pays the ask. Exit uses the first quote at or after
    ``exit_decision_ts + latency_ms`` and receives the bid.
    """

    _validate_settings(settings)
    if exit_decision_ts <= decision_ts:
        raise ValueError("exit_decision_ts must be greater than decision_ts")
    checked_quotes = sorted((_validate_quote(quote) for quote in quotes), key=lambda quote: quote.ts)
    if not checked_quotes:
        raise ValueError("at least one quote is required")

    entry_target_ts = decision_ts + settings.latency_ms
    exit_target_ts = exit_decision_ts + settings.latency_ms
    entry_quote = _first_quote_at_or_after(checked_quotes, entry_target_ts, "entry")
    exit_quote = _first_quote_at_or_after(checked_quotes, exit_target_ts, "exit")

    fee_rate = settings.fee_bps / 10_000.0
    slippage_rate = settings.slippage_bps / 10_000.0

    gross_entry_price = entry_quote.ask_price
    gross_exit_price = exit_quote.bid_price
    entry_slippage_price = gross_entry_price * (1.0 + slippage_rate)
    exit_slippage_price = gross_exit_price * (1.0 - slippage_rate)
    entry_fee = entry_slippage_price * fee_rate
    exit_fee = exit_slippage_price * fee_rate
    net_entry_price = entry_slippage_price + entry_fee
    net_exit_price = exit_slippage_price - exit_fee

    gross_pnl = gross_exit_price - gross_entry_price
    net_pnl = net_exit_price - net_entry_price
    return SimulatedTakerTrade(
        decision_ts=decision_ts,
        entry_target_ts=entry_target_ts,
        entry_ts=entry_quote.ts,
        exit_decision_ts=exit_decision_ts,
        exit_target_ts=exit_target_ts,
        exit_ts=exit_quote.ts,
        gross_entry_price=gross_entry_price,
        gross_exit_price=gross_exit_price,
        entry_slippage_price=entry_slippage_price,
        exit_slippage_price=exit_slippage_price,
        entry_fee=entry_fee,
        exit_fee=exit_fee,
        net_entry_price=net_entry_price,
        net_exit_price=net_exit_price,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        gross_return=gross_pnl / gross_entry_price,
        net_return=net_pnl / net_entry_price,
        fee_bps=settings.fee_bps,
        slippage_bps=settings.slippage_bps,
        latency_ms=settings.latency_ms,
    )


def simulate_taker_long_settlement_trade(
    *,
    quotes: Sequence[Quote],
    decision_ts: int,
    settlement_ts: int,
    settlement_price: float,
    settings: TakerExecutionSettings,
) -> SimulatedTakerTrade:
    """Simulate a long-UP taker entry held to binary settlement.

    Entry uses the first quote at or after ``decision_ts + latency_ms`` and
    pays the ask. Exit is the resolved token payoff, so no exit quote, exit
    slippage, or exit taker fee is applied.
    """

    _validate_settings(settings)
    if settlement_ts <= decision_ts:
        raise ValueError("settlement_ts must be greater than decision_ts")
    if settlement_price < 0.0:
        raise ValueError("settlement_price must be non-negative")
    checked_quotes = sorted((_validate_quote(quote) for quote in quotes), key=lambda quote: quote.ts)
    if not checked_quotes:
        raise ValueError("at least one quote is required")

    entry_target_ts = decision_ts + settings.latency_ms
    entry_quote = _first_quote_at_or_after(checked_quotes, entry_target_ts, "entry")
    fee_rate = settings.fee_bps / 10_000.0
    slippage_rate = settings.slippage_bps / 10_000.0

    gross_entry_price = entry_quote.ask_price
    gross_exit_price = settlement_price
    entry_slippage_price = gross_entry_price * (1.0 + slippage_rate)
    exit_slippage_price = gross_exit_price
    entry_fee = entry_slippage_price * fee_rate
    exit_fee = 0.0
    net_entry_price = entry_slippage_price + entry_fee
    net_exit_price = exit_slippage_price

    gross_pnl = gross_exit_price - gross_entry_price
    net_pnl = net_exit_price - net_entry_price
    return SimulatedTakerTrade(
        decision_ts=decision_ts,
        entry_target_ts=entry_target_ts,
        entry_ts=entry_quote.ts,
        exit_decision_ts=settlement_ts,
        exit_target_ts=settlement_ts,
        exit_ts=settlement_ts,
        gross_entry_price=gross_entry_price,
        gross_exit_price=gross_exit_price,
        entry_slippage_price=entry_slippage_price,
        exit_slippage_price=exit_slippage_price,
        entry_fee=entry_fee,
        exit_fee=exit_fee,
        net_entry_price=net_entry_price,
        net_exit_price=net_exit_price,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        gross_return=gross_pnl / gross_entry_price,
        net_return=net_pnl / net_entry_price,
        fee_bps=settings.fee_bps,
        slippage_bps=settings.slippage_bps,
        latency_ms=settings.latency_ms,
    )


def _first_quote_at_or_after(
    quotes: Sequence[Quote],
    target_ts: int,
    leg: str,
) -> Quote:
    for quote in quotes:
        if quote.ts >= target_ts:
            return quote
    raise NoQuoteAvailableError(f"no {leg} quote available at or after {target_ts}")


def _validate_settings(settings: TakerExecutionSettings) -> None:
    if settings.fee_bps < 0:
        raise ValueError("fee_bps must be non-negative")
    if settings.slippage_bps < 0:
        raise ValueError("slippage_bps must be non-negative")
    if settings.latency_ms < 0:
        raise ValueError("latency_ms must be non-negative")


def _validate_quote(quote: Quote) -> Quote:
    if quote.ts < 0:
        raise ValueError("quote ts must be non-negative")
    if quote.bid_price < 0 or quote.ask_price < 0:
        raise ValueError("quote prices must be non-negative")
    if quote.bid_price > quote.ask_price:
        raise ValueError("quote bid_price must be <= ask_price")
    return quote
