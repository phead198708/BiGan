"""Persist per-leg execution cash amounts alongside fill-price theory."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

import duckdb

CashLegAction = Literal["BUY", "SELL", "REDEEM"]
CashLegSource = Literal["clob_fill", "history_csv", "manual"]

EXECUTION_CASH_LEGS_DDL = """
CREATE TABLE IF NOT EXISTS execution_cash_legs (
    leg_id VARCHAR PRIMARY KEY,
    event_id VARCHAR NOT NULL,
    round_slug VARCHAR NOT NULL,
    action VARCHAR NOT NULL CHECK (action IN ('BUY', 'SELL', 'REDEEM')),
    usdc_amount DOUBLE NOT NULL,
    cash_delta DOUBLE NOT NULL,
    token_amount DOUBLE NOT NULL,
    fill_price DOUBLE,
    theoretical_usdc DOUBLE,
    usdc_delta_vs_theory DOUBLE,
    dust_token_amount DOUBLE NOT NULL DEFAULT 0,
    order_id VARCHAR,
    tx_hash VARCHAR,
    source VARCHAR NOT NULL CHECK (source IN ('clob_fill', 'history_csv', 'manual')),
    leg_ts BIGINT NOT NULL,
    details_json VARCHAR NOT NULL,
    created_at BIGINT NOT NULL
)
"""


@dataclass(frozen=True, slots=True)
class ExecutionCashLeg:
    """One signed account cash movement for an execution position."""

    leg_id: str
    event_id: str
    round_slug: str
    action: CashLegAction
    usdc_amount: float
    cash_delta: float
    token_amount: float
    fill_price: float | None
    theoretical_usdc: float | None
    usdc_delta_vs_theory: float | None
    dust_token_amount: float
    order_id: str | None
    tx_hash: str | None
    source: CashLegSource
    leg_ts: int
    details_json: str
    created_at: int | None = None

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["created_at"] = self.created_at or _now_ms()
        return row


def initialize_cash_leg_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create execution cash-leg tables."""

    conn.execute(EXECUTION_CASH_LEGS_DDL)


def signed_cash_delta(action: CashLegAction, usdc_amount: float) -> float:
    """Return signed USDC cash delta for a trade leg."""

    if action == "BUY":
        return -abs(usdc_amount)
    return abs(usdc_amount)


def theoretical_usdc(fill_price: float | None, token_amount: float, action: CashLegAction) -> float | None:
    """Return fill-price theory for one leg."""

    if fill_price is None or token_amount <= 0:
        return None
    value = float(fill_price) * float(token_amount)
    return value if action == "SELL" else -value


def leg_from_clob_fill(
    *,
    event_id: str,
    round_slug: str,
    action: CashLegAction,
    fill: dict[str, Any],
    order_id: str | None = None,
    dust_token_amount: float = 0.0,
    leg_id_suffix: str | None = None,
) -> ExecutionCashLeg:
    """Build a cash leg from a confirmed CLOB trade payload."""

    fill_price = _optional_float(fill.get("price"))
    token_amount = _optional_float(fill.get("size")) or 0.0
    usdc_amount = _optional_float(fill.get("usdcAmount"))
    if usdc_amount is None and fill_price is not None and token_amount > 0:
        usdc_amount = fill_price * token_amount
    if usdc_amount is None:
        raise ValueError("fill must include usdcAmount or price/size")
    theory = theoretical_usdc(fill_price, token_amount, action)
    cash_delta = signed_cash_delta(action, usdc_amount)
    leg_ts = int(fill.get("timestamp") or fill.get("match_time") or _now_ms())
    suffix = leg_id_suffix or order_id or str(leg_ts)
    return ExecutionCashLeg(
        leg_id=f"{event_id}:{action}:{suffix}",
        event_id=event_id,
        round_slug=round_slug,
        action=action,
        usdc_amount=abs(usdc_amount),
        cash_delta=cash_delta,
        token_amount=token_amount,
        fill_price=fill_price,
        theoretical_usdc=theory,
        usdc_delta_vs_theory=None if theory is None else cash_delta - theory,
        dust_token_amount=max(0.0, dust_token_amount),
        order_id=order_id,
        tx_hash=_optional_text(fill.get("transaction_hash") or fill.get("tx_hash")),
        source="clob_fill",
        leg_ts=leg_ts,
        details_json=json.dumps(fill, sort_keys=True, default=str),
    )


def leg_from_polymarket_history(
    *,
    event_id: str,
    round_slug: str,
    action: CashLegAction,
    usdc_amount: float,
    token_amount: float,
    fill_price: float | None,
    leg_ts: int,
    tx_hash: str | None = None,
) -> ExecutionCashLeg:
    """Build a cash leg from a Polymarket account-history row."""

    cash_delta = signed_cash_delta(action, usdc_amount)
    theory = theoretical_usdc(fill_price, token_amount, action)
    suffix = tx_hash or str(leg_ts)
    return ExecutionCashLeg(
        leg_id=f"{event_id}:{action}:{suffix}",
        event_id=event_id,
        round_slug=round_slug,
        action=action,
        usdc_amount=abs(usdc_amount),
        cash_delta=cash_delta,
        token_amount=token_amount,
        fill_price=fill_price,
        theoretical_usdc=theory,
        usdc_delta_vs_theory=None if theory is None else cash_delta - theory,
        dust_token_amount=0.0,
        order_id=None,
        tx_hash=tx_hash,
        source="history_csv",
        leg_ts=leg_ts,
        details_json=json.dumps(
            {
                "usdc_amount": usdc_amount,
                "token_amount": token_amount,
                "fill_price": fill_price,
            },
            sort_keys=True,
        ),
    )


def record_execution_cash_legs(
    conn: duckdb.DuckDBPyConnection,
    legs: list[ExecutionCashLeg],
    *,
    replace: bool = True,
) -> None:
    """Persist execution cash-leg rows."""

    initialize_cash_leg_tables(conn)
    for leg in legs:
        _validate_leg(leg)
        row = leg.to_row()
        columns = tuple(row)
        placeholders = ", ".join("?" for _ in columns)
        conn.execute(
            f"INSERT {'OR REPLACE ' if replace else ''}INTO execution_cash_legs "
            f"({', '.join(columns)}) VALUES ({placeholders})",
            [row[column] for column in columns],
        )


def read_execution_cash_legs(
    conn: duckdb.DuckDBPyConnection,
    *,
    event_id: str | None = None,
) -> list[dict[str, Any]]:
    """Read persisted execution cash-leg rows."""

    initialize_cash_leg_tables(conn)
    predicate = ""
    params: list[str] = []
    if event_id is not None:
        predicate = "WHERE event_id = ?"
        params.append(event_id)
    rows = conn.execute(
        f"""
        SELECT *
        FROM execution_cash_legs
        {predicate}
        ORDER BY leg_ts ASC, leg_id ASC
        """,
        params,
    ).fetchall()
    columns = [column[0] for column in conn.description]
    return [dict(zip(columns, row, strict=True)) for row in rows]


def account_cash_pnl_from_legs(legs: list[ExecutionCashLeg]) -> float:
    """Return signed account cash PnL from persisted legs."""

    return sum(leg.cash_delta for leg in legs)


def _validate_leg(leg: ExecutionCashLeg) -> None:
    if not leg.leg_id or not leg.event_id or not leg.round_slug:
        raise ValueError("leg_id, event_id, and round_slug are required")
    if leg.action not in {"BUY", "SELL", "REDEEM"}:
        raise ValueError("invalid action")
    if leg.usdc_amount < 0:
        raise ValueError("usdc_amount must be non-negative")
    if leg.token_amount < 0:
        raise ValueError("token_amount must be non-negative")
    if leg.source not in {"clob_fill", "history_csv", "manual"}:
        raise ValueError("invalid source")
    json.loads(leg.details_json)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _now_ms() -> int:
    return int(time.time() * 1000)
