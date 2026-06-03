"""Account cash-flow reconciliation for Polymarket execution history."""

from __future__ import annotations

import csv
import json
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import duckdb

from .position_manager import Position, PositionSide

CashFlowAction = Literal["BUY", "SELL", "REDEEM"]
CashFlowMatchStatus = Literal["matched", "missing_cash_flow", "ambiguous_redeem"]

EASTERN_TZ = ZoneInfo("America/New_York")

EXECUTION_CASHFLOW_RECONCILIATIONS_DDL = """
CREATE TABLE IF NOT EXISTS execution_cashflow_reconciliations (
    event_id VARCHAR PRIMARY KEY,
    round_slug VARCHAR NOT NULL,
    side VARCHAR NOT NULL CHECK (side IN ('UP', 'DOWN')),
    sleeve VARCHAR NOT NULL DEFAULT 'settlement' CHECK (sleeve IN ('settlement', 'volatility')),
    position_status VARCHAR NOT NULL,
    theoretical_pnl DOUBLE,
    account_cash_pnl DOUBLE,
    cash_pnl_delta DOUBLE,
    bought_token_amount DOUBLE NOT NULL,
    sold_token_amount DOUBLE NOT NULL,
    redeemed_token_amount DOUBLE NOT NULL,
    dust_token_amount DOUBLE NOT NULL,
    cash_flow_count INTEGER NOT NULL,
    first_cash_flow_ts BIGINT,
    last_cash_flow_ts BIGINT,
    match_status VARCHAR NOT NULL CHECK (
        match_status IN ('matched', 'missing_cash_flow', 'ambiguous_redeem')
    ),
    cash_flows_json VARCHAR NOT NULL,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL
)
"""


@dataclass(frozen=True, slots=True)
class PolymarketCashFlow:
    """One account-history cash movement from a Polymarket export."""

    market_name: str
    action: CashFlowAction
    usdc_amount: float
    token_amount: float
    token_name: str
    timestamp: int
    tx_hash: str
    round_slug: str
    side: PositionSide | None

    @property
    def cash_delta(self) -> float:
        """Signed account cash delta in USDC."""

        if self.action == "BUY":
            return -self.usdc_amount
        return self.usdc_amount

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cash_delta"] = self.cash_delta
        return payload


@dataclass(frozen=True, slots=True)
class CashFlowReconciliation:
    """Account-level cash reconciliation for one execution position."""

    event_id: str
    round_slug: str
    side: PositionSide
    sleeve: str
    position_status: str
    theoretical_pnl: float | None
    account_cash_pnl: float | None
    cash_pnl_delta: float | None
    bought_token_amount: float
    sold_token_amount: float
    redeemed_token_amount: float
    dust_token_amount: float
    cash_flow_count: int
    first_cash_flow_ts: int | None
    last_cash_flow_ts: int | None
    match_status: CashFlowMatchStatus
    cash_flows_json: str
    created_at: int | None = None
    updated_at: int | None = None

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        now = _now_ms()
        row["created_at"] = self.created_at or now
        row["updated_at"] = self.updated_at or now
        return row


def initialize_cashflow_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create account cash-flow reconciliation tables."""

    conn.execute(EXECUTION_CASHFLOW_RECONCILIATIONS_DDL)
    _ensure_cashflow_schema(conn)


def read_polymarket_history_csv(path: Path | str) -> list[PolymarketCashFlow]:
    """Read Polymarket account-history CSV rows as signed cash-flow records."""

    rows: list[PolymarketCashFlow] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            flow = _cash_flow_from_csv_row(raw)
            if flow is not None:
                rows.append(flow)
    return rows


def reconcile_cash_flows(
    positions: list[Position],
    cash_flows: list[PolymarketCashFlow],
) -> list[CashFlowReconciliation]:
    """Reconcile execution positions against account-history cash flows."""

    by_key: dict[tuple[str, PositionSide], list[PolymarketCashFlow]] = defaultdict(list)
    redeem_by_round: dict[str, list[PolymarketCashFlow]] = defaultdict(list)
    positions_by_round: dict[str, list[Position]] = defaultdict(list)
    for position in positions:
        positions_by_round[_round_slug_from_position(position)].append(position)
    for flow in cash_flows:
        if flow.side is None:
            redeem_by_round[flow.round_slug].append(flow)
        else:
            by_key[(flow.round_slug, flow.side)].append(flow)

    reconciled: list[CashFlowReconciliation] = []
    for position in positions:
        round_slug = _round_slug_from_position(position)
        flows = list(by_key.get((round_slug, position.side), ()))
        neutral_flows = redeem_by_round.get(round_slug, [])
        match_status: CashFlowMatchStatus = "matched"
        if neutral_flows:
            round_positions = positions_by_round.get(round_slug, [])
            if len(round_positions) == 1:
                flows.extend(neutral_flows)
            else:
                match_status = "ambiguous_redeem"
        if not flows and match_status != "ambiguous_redeem":
            match_status = "missing_cash_flow"
        reconciled.append(_reconcile_position(position, round_slug, flows, match_status))
    return reconciled


def record_cashflow_reconciliations(
    conn: duckdb.DuckDBPyConnection,
    records: list[CashFlowReconciliation],
    *,
    replace: bool = True,
) -> None:
    """Persist account cash-flow reconciliation rows."""

    initialize_cashflow_tables(conn)
    for record in records:
        _validate_record(record)
        row = record.to_row()
        columns = tuple(row)
        placeholders = ", ".join("?" for _ in columns)
        conn.execute(
            f"INSERT {'OR REPLACE ' if replace else ''}INTO execution_cashflow_reconciliations "
            f"({', '.join(columns)}) VALUES ({placeholders})",
            [row[column] for column in columns],
        )


def read_cashflow_reconciliations(
    conn: duckdb.DuckDBPyConnection,
    *,
    event_id: str | None = None,
) -> list[dict[str, Any]]:
    """Read persisted account cash-flow reconciliation rows."""

    initialize_cashflow_tables(conn)
    predicate = ""
    params: list[str] = []
    if event_id is not None:
        predicate = "WHERE event_id = ?"
        params.append(event_id)
    rows = conn.execute(
        f"""
        SELECT *
        FROM execution_cashflow_reconciliations
        {predicate}
        ORDER BY first_cash_flow_ts NULLS LAST, event_id ASC
        """,
        params,
    ).fetchall()
    columns = [column[0] for column in conn.description]
    return [dict(zip(columns, row, strict=True)) for row in rows]


def account_cash_pnl(cash_flows: list[PolymarketCashFlow]) -> float:
    """Return signed account cash PnL for a collection of history rows."""

    return sum(flow.cash_delta for flow in cash_flows)


def _cash_flow_from_csv_row(row: dict[str, str]) -> PolymarketCashFlow | None:
    market_name = (row.get("marketName") or "").strip()
    if not market_name.startswith("Bitcoin Up or Down - "):
        return None
    action = _normalise_action(row.get("action"))
    timestamp = int(row.get("timestamp") or 0)
    round_slug = _round_slug_from_market_name(market_name, timestamp)
    token_name = (row.get("tokenName") or "").strip()
    return PolymarketCashFlow(
        market_name=market_name,
        action=action,
        usdc_amount=float(row.get("usdcAmount") or 0.0),
        token_amount=float(row.get("tokenAmount") or 0.0),
        token_name=token_name,
        timestamp=timestamp,
        tx_hash=(row.get("hash") or "").strip(),
        round_slug=round_slug,
        side=_normalise_side_or_none(token_name),
    )


def _reconcile_position(
    position: Position,
    round_slug: str,
    flows: list[PolymarketCashFlow],
    match_status: CashFlowMatchStatus,
) -> CashFlowReconciliation:
    ordered = sorted(flows, key=lambda item: (item.timestamp, item.action, item.tx_hash))
    bought = sum(flow.token_amount for flow in ordered if flow.action == "BUY")
    sold = sum(flow.token_amount for flow in ordered if flow.action == "SELL")
    redeemed = sum(flow.token_amount for flow in ordered if flow.action == "REDEEM")
    account_pnl = account_cash_pnl(ordered) if ordered else None
    theoretical_pnl = position.realized_pnl
    cash_pnl_delta = (
        None
        if account_pnl is None or theoretical_pnl is None
        else account_pnl - theoretical_pnl
    )
    return CashFlowReconciliation(
        event_id=position.event_id,
        round_slug=round_slug,
        side=position.side,
        sleeve=position.sleeve,
        position_status=position.status,
        theoretical_pnl=theoretical_pnl,
        account_cash_pnl=account_pnl,
        cash_pnl_delta=cash_pnl_delta,
        bought_token_amount=bought,
        sold_token_amount=sold,
        redeemed_token_amount=redeemed,
        dust_token_amount=_dust_token_amount(bought, sold, redeemed),
        cash_flow_count=len(ordered),
        first_cash_flow_ts=None if not ordered else ordered[0].timestamp,
        last_cash_flow_ts=None if not ordered else ordered[-1].timestamp,
        match_status=match_status,
        cash_flows_json=json.dumps(
            [flow.to_json_dict() for flow in ordered],
            sort_keys=True,
        ),
    )


def _dust_token_amount(bought: float, sold: float, redeemed: float) -> float:
    if sold <= 0 and redeemed <= 0:
        return 0.0
    return max(0.0, bought - sold - redeemed)


def _round_slug_from_position(position: Position) -> str:
    match = re.search(r"(btc-updown-15m-\d+)", position.event_id)
    if match:
        return match.group(1)
    parts = position.symbol.split(":")
    if len(parts) >= 2:
        return parts[-2]
    raise ValueError(f"cannot derive round slug from position {position.event_id}")


def _round_slug_from_market_name(market_name: str, timestamp: int) -> str:
    match = re.search(
        r"Bitcoin Up or Down - ([A-Za-z]+) (\d{1,2}), "
        r"(\d{1,2}):(\d{2})(AM|PM)-",
        market_name,
    )
    if match is None:
        raise ValueError(f"unsupported Polymarket marketName: {market_name}")
    month_name, day, hour, minute, meridiem = match.groups()
    year = datetime.fromtimestamp(timestamp, tz=EASTERN_TZ).year if timestamp else _now_et().year
    hour_int = int(hour)
    if meridiem == "PM" and hour_int != 12:
        hour_int += 12
    if meridiem == "AM" and hour_int == 12:
        hour_int = 0
    start = datetime(
        year=year,
        month=datetime.strptime(month_name, "%B").month,
        day=int(day),
        hour=hour_int,
        minute=int(minute),
        tzinfo=EASTERN_TZ,
    )
    return f"btc-updown-15m-{int(start.timestamp())}"


def _normalise_action(value: str | None) -> CashFlowAction:
    text = str(value or "").strip().upper()
    if text not in {"BUY", "SELL", "REDEEM"}:
        raise ValueError(f"unsupported Polymarket action: {value}")
    return text  # type: ignore[return-value]


def _normalise_side_or_none(value: str | None) -> PositionSide | None:
    text = str(value or "").strip().upper()
    if text == "":
        return None
    if text not in {"UP", "DOWN"}:
        raise ValueError(f"unsupported Polymarket tokenName: {value}")
    return text  # type: ignore[return-value]


def _validate_record(record: CashFlowReconciliation) -> None:
    if not record.event_id:
        raise ValueError("event_id is required")
    if record.side not in {"UP", "DOWN"}:
        raise ValueError("side must be UP or DOWN")
    if record.sleeve not in {"settlement", "volatility"}:
        raise ValueError("sleeve must be settlement or volatility")
    if record.cash_flow_count < 0:
        raise ValueError("cash_flow_count must be non-negative")
    if record.match_status not in {"matched", "missing_cash_flow", "ambiguous_redeem"}:
        raise ValueError("invalid match_status")
    json.loads(record.cash_flows_json)


def _ensure_cashflow_schema(conn: duckdb.DuckDBPyConnection) -> None:
    columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info('execution_cashflow_reconciliations')").fetchall()
    }
    if "sleeve" not in columns:
        conn.execute("ALTER TABLE execution_cashflow_reconciliations ADD COLUMN sleeve VARCHAR")
        conn.execute(
            "UPDATE execution_cashflow_reconciliations "
            "SET sleeve = 'settlement' WHERE sleeve IS NULL"
        )


def _now_et() -> datetime:
    return datetime.now(tz=EASTERN_TZ)


def _now_ms() -> int:
    return int(time.time() * 1000)
