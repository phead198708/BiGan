"""DuckDB-backed live execution position tracking."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import duckdb

from .db import DEFAULT_MLOPS_DB_PATH, connect_mlops_db

PositionSide = Literal["UP", "DOWN"]
PositionStatus = Literal["open", "closed", "expired"]
PositionSleeve = Literal["settlement", "volatility"]

POSITIONS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS execution_positions (
    event_id VARCHAR PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    side VARCHAR NOT NULL CHECK (side IN ('UP', 'DOWN')),
    sleeve VARCHAR NOT NULL DEFAULT 'settlement' CHECK (sleeve IN ('settlement', 'volatility')),
    status VARCHAR NOT NULL CHECK (status IN ('open', 'closed', 'expired')),
    entry_time BIGINT NOT NULL,
    entry_price DOUBLE NOT NULL,
    fill_price DOUBLE,
    size DOUBLE NOT NULL,
    order_id VARCHAR NOT NULL,
    current_price DOUBLE,
    unrealized_pnl DOUBLE,
    exit_price DOUBLE,
    exit_time BIGINT,
    realized_pnl DOUBLE,
    settlement_result VARCHAR,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL
)
"""


@dataclass(frozen=True, slots=True)
class Position:
    """One live or historical Polymarket token position."""

    event_id: str
    symbol: str
    side: PositionSide
    sleeve: PositionSleeve
    status: PositionStatus
    entry_time: int
    entry_price: float
    fill_price: float | None
    size: float
    order_id: str
    current_price: float | None
    unrealized_pnl: float | None
    exit_price: float | None
    exit_time: int | None
    realized_pnl: float | None
    settlement_result: str | None
    created_at: int | None = None
    updated_at: int | None = None

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        now = _now_ms()
        row["created_at"] = self.created_at or now
        row["updated_at"] = self.updated_at or now
        return row


class PositionManager:
    """Persist and query intra-round execution state."""

    def __init__(
        self,
        db_path: Path | str = DEFAULT_MLOPS_DB_PATH,
        *,
        conn: duckdb.DuckDBPyConnection | None = None,
    ) -> None:
        self.db_path = db_path
        self._conn = conn
        with self._connection() as active:
            initialize_position_tables(active)

    def open_position(
        self,
        event_id: str,
        symbol: str,
        side: PositionSide | str,
        entry_price: float,
        size: float,
        order_id: str,
        *,
        sleeve: PositionSleeve | str = "settlement",
        entry_time: int | None = None,
        fill_price: float | None = None,
    ) -> Position:
        """Persist a newly opened position, rejecting duplicate open rounds."""

        normal_side = _normalise_side(side)
        normal_sleeve = _normalise_sleeve(sleeve)
        _require_text("event_id", event_id)
        _require_text("symbol", symbol)
        _require_text("order_id", order_id)
        if entry_price <= 0:
            raise ValueError("entry_price must be positive")
        if size <= 0:
            raise ValueError("size must be positive")
        if self.has_open_position(event_id):
            raise ValueError(f"open position already exists for event_id={event_id}")
        ts = entry_time or _now_ms()
        position = Position(
            event_id=event_id,
            symbol=symbol,
            side=normal_side,
            sleeve=normal_sleeve,
            status="open",
            entry_time=ts,
            entry_price=float(entry_price),
            fill_price=float(fill_price) if fill_price is not None else None,
            size=float(size),
            order_id=order_id,
            current_price=float(fill_price if fill_price is not None else entry_price),
            unrealized_pnl=0.0,
            exit_price=None,
            exit_time=None,
            realized_pnl=None,
            settlement_result=None,
            created_at=ts,
            updated_at=ts,
        )
        self._upsert(position)
        return position

    def update_price(self, event_id: str, current_price: float) -> Position:
        """Update an open position mark price and unrealized PnL."""

        position = self._require_position(event_id)
        if position.status != "open":
            raise ValueError(f"cannot update non-open position {event_id}")
        if current_price < 0:
            raise ValueError("current_price must be non-negative")
        cost_basis = _cost_basis(position)
        marked = _replace_position(
            position,
            current_price=float(current_price),
            unrealized_pnl=(float(current_price) - cost_basis) * position.size,
            updated_at=_now_ms(),
        )
        self._upsert(marked)
        return marked

    def adjust_open_position(
        self,
        event_id: str,
        *,
        fill_price: float,
        size: float,
        current_price: float | None = None,
    ) -> Position:
        """Update an open position's average fill price and remaining size."""

        position = self._require_position(event_id)
        if position.status != "open":
            raise ValueError(f"cannot adjust non-open position {event_id}")
        if fill_price <= 0:
            raise ValueError("fill_price must be positive")
        if size <= 0:
            raise ValueError("size must be positive")
        mark_price = fill_price if current_price is None else current_price
        if mark_price < 0:
            raise ValueError("current_price must be non-negative")
        adjusted = _replace_position(
            position,
            fill_price=float(fill_price),
            size=float(size),
            current_price=float(mark_price),
            unrealized_pnl=(float(mark_price) - float(fill_price)) * float(size),
            updated_at=_now_ms(),
        )
        self._upsert(adjusted)
        return adjusted

    def close_position(
        self,
        event_id: str,
        exit_price: float,
        *,
        exit_time: int | None = None,
    ) -> Position:
        """Close an open position from an explicit SELL/exit signal."""

        position = self._require_position(event_id)
        if position.status != "open":
            raise ValueError(f"cannot close non-open position {event_id}")
        if exit_price < 0:
            raise ValueError("exit_price must be non-negative")
        ts = exit_time or _now_ms()
        cost_basis = _cost_basis(position)
        realized = (float(exit_price) - cost_basis) * position.size
        closed = _replace_position(
            position,
            status="closed",
            current_price=float(exit_price),
            unrealized_pnl=realized,
            exit_price=float(exit_price),
            exit_time=ts,
            realized_pnl=realized,
            updated_at=ts,
        )
        self._upsert(closed)
        return closed

    def settle_position(
        self,
        event_id: str,
        result: PositionSide | str | bool,
        *,
        settlement_time: int | None = None,
    ) -> Position:
        """Settle a position at payout 1.0 or 0.0 after the round resolves."""

        position = self._require_position(event_id)
        winning_side = _normalise_result(result)
        exit_price = 1.0 if winning_side == position.side else 0.0
        ts = settlement_time or _now_ms()
        cost_basis = _cost_basis(position)
        realized = (exit_price - cost_basis) * position.size
        settled = _replace_position(
            position,
            status="expired",
            current_price=exit_price,
            unrealized_pnl=realized,
            exit_price=exit_price,
            exit_time=ts,
            realized_pnl=realized,
            settlement_result=winning_side,
            updated_at=ts,
        )
        self._upsert(settled)
        return settled

    def get_position(self, event_id: str) -> Position | None:
        """Return one position by event id."""

        _require_text("event_id", event_id)
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM execution_positions
                WHERE event_id = ?
                """,
                [event_id],
            ).fetchall()
            columns = [column[0] for column in conn.description]
        return None if not rows else _position_from_row(columns, rows[0])

    def has_open_position(self, event_id: str) -> bool:
        """Return whether a position is currently open for this event/round."""

        _require_text("event_id", event_id)
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM execution_positions
                WHERE event_id = ?
                  AND status = 'open'
                LIMIT 1
                """,
                [event_id],
            ).fetchone()
        return row is not None

    def get_all_open(self) -> list[Position]:
        """Return all currently open positions."""

        return self.list_positions(status="open")

    def list_positions(self, *, status: str | None = None) -> list[Position]:
        """Return persisted position history for dashboards and settlement jobs."""

        params: list[str] = []
        predicate = ""
        if status is not None:
            if status not in {"open", "closed", "expired"}:
                raise ValueError("status must be open, closed, expired, or None")
            predicate = "WHERE status = ?"
            params.append(status)
        with self._connection() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM execution_positions
                {predicate}
                ORDER BY entry_time ASC, event_id ASC
                """,
                params,
            ).fetchall()
            columns = [column[0] for column in conn.description]
        return [_position_from_row(columns, row) for row in rows]

    def _require_position(self, event_id: str) -> Position:
        position = self.get_position(event_id)
        if position is None:
            raise ValueError(f"unknown position event_id={event_id}")
        return position

    def _upsert(self, position: Position) -> None:
        row = position.to_row()
        columns = tuple(row)
        placeholders = ", ".join("?" for _ in columns)
        with self._connection() as conn:
            initialize_position_tables(conn)
            conn.execute(
                f"INSERT OR REPLACE INTO execution_positions "
                f"({', '.join(columns)}) VALUES ({placeholders})",
                [row[column] for column in columns],
            )

    @contextmanager
    def _connection(self) -> Iterator[duckdb.DuckDBPyConnection]:
        if self._conn is not None:
            yield self._conn
            return
        conn = connect_mlops_db(self.db_path)
        try:
            yield conn
        finally:
            conn.close()


def initialize_position_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create execution position tables."""

    conn.execute(POSITIONS_TABLE_DDL)
    _ensure_position_schema(conn)


def _position_from_row(columns: list[str], row: tuple[Any, ...]) -> Position:
    payload = dict(zip(columns, row, strict=True))
    return Position(
        event_id=str(payload["event_id"]),
        symbol=str(payload["symbol"]),
        side=_normalise_side(payload["side"]),
        sleeve=_normalise_sleeve(payload.get("sleeve") or "settlement"),
        status=str(payload["status"]),  # type: ignore[arg-type]
        entry_time=int(payload["entry_time"]),
        entry_price=float(payload["entry_price"]),
        fill_price=_optional_float(payload["fill_price"]),
        size=float(payload["size"]),
        order_id=str(payload["order_id"]),
        current_price=_optional_float(payload["current_price"]),
        unrealized_pnl=_optional_float(payload["unrealized_pnl"]),
        exit_price=_optional_float(payload["exit_price"]),
        exit_time=None if payload["exit_time"] is None else int(payload["exit_time"]),
        realized_pnl=_optional_float(payload["realized_pnl"]),
        settlement_result=None
        if payload["settlement_result"] is None
        else str(payload["settlement_result"]),
        created_at=int(payload["created_at"]),
        updated_at=int(payload["updated_at"]),
    )


def _replace_position(position: Position, **updates: Any) -> Position:
    payload = asdict(position)
    payload.update(updates)
    return Position(**payload)


def _normalise_side(value: Any) -> PositionSide:
    text = str(value).strip().upper()
    if text not in {"UP", "DOWN"}:
        raise ValueError("side must be UP or DOWN")
    return text  # type: ignore[return-value]


def _normalise_result(value: PositionSide | str | bool) -> PositionSide:
    if isinstance(value, bool):
        return "UP" if value else "DOWN"
    return _normalise_side(value)


def _normalise_sleeve(value: Any) -> PositionSleeve:
    text = str(value or "settlement").strip().lower()
    if text not in {"settlement", "volatility"}:
        raise ValueError("sleeve must be settlement or volatility")
    return text  # type: ignore[return-value]


def _ensure_position_schema(conn: duckdb.DuckDBPyConnection) -> None:
    columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info('execution_positions')").fetchall()
    }
    if "sleeve" not in columns:
        conn.execute("ALTER TABLE execution_positions ADD COLUMN sleeve VARCHAR")
        conn.execute("UPDATE execution_positions SET sleeve = 'settlement' WHERE sleeve IS NULL")


def _cost_basis(position: Position) -> float:
    return position.fill_price if position.fill_price is not None else position.entry_price


def _require_text(field_name: str, value: str) -> None:
    if not str(value).strip():
        raise ValueError(f"{field_name} is required")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _now_ms() -> int:
    return int(time.time() * 1000)
