#!/usr/bin/env python3
"""Reconcile Polymarket account-history cash flow against execution positions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bigan.execution import (
    PositionManager,
    read_polymarket_history_csv,
    reconcile_cash_flows,
    record_cashflow_reconciliations,
)
from bigan.mlops.registry import connect_mlops_db


def main() -> int:
    args = _parse_args()
    positions = [
        position
        for position in PositionManager(args.db_path).list_positions()
        if position.event_id.startswith(args.event_prefix)
    ]
    cash_flows = read_polymarket_history_csv(args.history_csv)
    records = reconcile_cash_flows(positions, cash_flows)
    summary = _summary(records)

    if args.write_db:
        with connect_mlops_db(args.db_path) as conn:
            record_cashflow_reconciliations(conn, records, replace=True)

    if args.report_path:
        Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_path).write_text(_markdown_report(records, summary), encoding="utf-8")
    if args.summary_json_path:
        Path(args.summary_json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json_path).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-csv", required=True, help="Polymarket account history CSV")
    parser.add_argument("--db-path", default="data/mlops/champion_catalog.duckdb")
    parser.add_argument("--event-prefix", default="phase4-")
    parser.add_argument("--write-db", action="store_true")
    parser.add_argument("--report-path", default="")
    parser.add_argument("--summary-json-path", default="")
    return parser.parse_args()


def _summary(records: list[Any]) -> dict[str, Any]:
    matched = [record for record in records if record.match_status == "matched"]
    theoretical = [
        record.theoretical_pnl
        for record in records
        if record.theoretical_pnl is not None and record.match_status == "matched"
    ]
    deltas = [
        record.cash_pnl_delta
        for record in records
        if record.cash_pnl_delta is not None and record.match_status == "matched"
    ]
    return {
        "positions": len(records),
        "matched": len(matched),
        "missing_cash_flow": sum(1 for record in records if record.match_status == "missing_cash_flow"),
        "ambiguous_redeem": sum(1 for record in records if record.match_status == "ambiguous_redeem"),
        "account_cash_pnl": round(
            sum(record.account_cash_pnl or 0.0 for record in matched),
            8,
        ),
        "theoretical_pnl": round(sum(theoretical), 8),
        "cash_minus_theoretical": round(sum(deltas), 8),
        "dust_token_amount": round(sum(record.dust_token_amount for record in matched), 8),
    }


def _markdown_report(records: list[Any], summary: dict[str, Any]) -> str:
    lines = [
        "# Polymarket Cash-Flow Reconciliation",
        "",
        "## Summary",
        "",
        f"- Positions: {summary['positions']}",
        f"- Matched cash-flow rows: {summary['matched']}",
        f"- Missing cash-flow rows: {summary['missing_cash_flow']}",
        f"- Ambiguous redeem rows: {summary['ambiguous_redeem']}",
        f"- Account cash-flow PnL: {summary['account_cash_pnl']:.6f}",
        f"- Theoretical fill-price PnL: {summary['theoretical_pnl']:.6f}",
        f"- Cash minus theoretical: {summary['cash_minus_theoretical']:.6f}",
        f"- Dust token amount: {summary['dust_token_amount']:.6f}",
        "",
        "## Positions",
        "",
        "| Event | Status | Side | Match | Account PnL | Theoretical PnL | Delta | Dust |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for record in sorted(records, key=lambda item: (item.round_slug, item.event_id)):
        lines.append(
            "| "
            f"{record.event_id} | "
            f"{record.position_status} | "
            f"{record.side} | "
            f"{record.match_status} | "
            f"{_fmt(record.account_cash_pnl)} | "
            f"{_fmt(record.theoretical_pnl)} | "
            f"{_fmt(record.cash_pnl_delta)} | "
            f"{record.dust_token_amount:.6f} |"
        )
    lines.extend(
        [
            "",
            "Account PnL uses Polymarket account-history cash flow: "
            "`BUY=-usdcAmount`, `SELL=+usdcAmount`, `REDEEM=+usdcAmount`.",
            "Theoretical PnL is kept separate because it is derived from executor fill price and size.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
