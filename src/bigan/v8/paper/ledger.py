"""Deterministic paper ledger and position accounting."""

from __future__ import annotations

from dataclasses import dataclass

from bigan.v8.paper.contracts import (
    PaperFill,
    PaperLedgerEntry,
    PaperOrder,
    PaperPositionSnapshot,
)


@dataclass(slots=True)
class _MutablePosition:
    position_size: float = 0.0
    average_entry_price: float = 0.0
    mark_price: float = 100.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    last_update_ts: int = 0


class PaperLedger:
    """Stateful, paper-only ledger with deterministic position snapshots."""

    def __init__(self, *, initial_cash: float) -> None:
        if initial_cash <= 0.0:
            raise ValueError("initial_cash must be positive")
        self._cash = float(initial_cash)
        self._cumulative_net_return = 0.0
        self._positions: dict[tuple[str, str], _MutablePosition] = {}
        self._entries: list[PaperLedgerEntry] = []

    @property
    def entries(self) -> tuple[PaperLedgerEntry, ...]:
        return tuple(self._entries)

    def position_size(self, *, source: str, instrument_id: str) -> float:
        return self._positions.get((source, instrument_id), _MutablePosition()).position_size

    def apply_fill(self, order: PaperOrder, fill: PaperFill) -> PaperLedgerEntry:
        key = (fill.source, fill.instrument_id)
        position = self._positions.setdefault(key, _MutablePosition())
        position_before = position.position_size
        cash_before = self._cash
        signed_fill_size = _signed_fill_size(fill)
        position_after = position_before + signed_fill_size
        trade_cash = signed_fill_size * fill.fill_price
        self._cash = cash_before - trade_cash - fill.total_execution_cost
        self._cumulative_net_return += fill.net_return

        if abs(position_after) <= 1e-12:
            average_entry_price = 0.0
        elif abs(position_before) <= 1e-12 or _same_sign(position_before, signed_fill_size):
            total_abs_position = abs(position_before) + abs(signed_fill_size)
            average_entry_price = (
                abs(position_before) * position.average_entry_price
                + abs(signed_fill_size) * fill.fill_price
            ) / max(total_abs_position, 1e-12)
        else:
            average_entry_price = position.average_entry_price

        position.position_size = position_after
        position.average_entry_price = average_entry_price
        position.mark_price = fill.mark_price
        position.realized_pnl += fill.net_return
        position.unrealized_pnl = position_after * (fill.mark_price - average_entry_price)
        position.last_update_ts = fill.decision_ts

        entry = PaperLedgerEntry(
            entry_id=f"ledger-{len(self._entries):06d}",
            order_id=order.order_id,
            fill_id=fill.fill_id,
            decision_ts=fill.decision_ts,
            source=fill.source,
            instrument_id=fill.instrument_id,
            position_before=position_before,
            position_after=position.position_size,
            cash_before=cash_before,
            cash_after=self._cash,
            realized_pnl=position.realized_pnl,
            unrealized_pnl=position.unrealized_pnl,
            net_return=fill.net_return,
            cumulative_net_return=self._cumulative_net_return,
            total_execution_cost=fill.total_execution_cost,
        )
        self._entries.append(entry)
        return entry

    def snapshots(self) -> tuple[PaperPositionSnapshot, ...]:
        snapshots: list[PaperPositionSnapshot] = []
        for source, instrument_id in sorted(self._positions):
            position = self._positions[(source, instrument_id)]
            snapshots.append(
                PaperPositionSnapshot(
                    source=source,
                    instrument_id=instrument_id,
                    position_size=position.position_size,
                    average_entry_price=position.average_entry_price,
                    mark_price=position.mark_price,
                    unrealized_pnl=position.unrealized_pnl,
                    realized_pnl=position.realized_pnl,
                    last_update_ts=position.last_update_ts,
                )
            )
        return tuple(snapshots)


def _signed_fill_size(fill: PaperFill) -> float:
    if fill.side == "buy":
        return fill.filled_size
    if fill.side == "sell":
        return -fill.filled_size
    return 0.0


def _same_sign(left: float, right: float) -> bool:
    return (left >= 0.0 and right >= 0.0) or (left <= 0.0 and right <= 0.0)
