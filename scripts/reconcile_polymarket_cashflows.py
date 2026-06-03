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
from bigan.execution.db import connect_mlops_db


def main() -> int:
    args = _parse_args()
    positions = _select_positions(
        PositionManager(args.db_path).list_positions(),
        event_prefix=args.event_prefix,
        event_ids=set(args.event_id),
    )
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
    parser.add_argument(
        "--event-id",
        action="append",
        default=[],
        help="Restrict reconciliation to a specific execution event_id; may be repeated.",
    )
    parser.add_argument("--write-db", action="store_true")
    parser.add_argument("--report-path", default="")
    parser.add_argument("--summary-json-path", default="")
    return parser.parse_args()


def _select_positions(
    positions: list[Any],
    *,
    event_prefix: str,
    event_ids: set[str],
) -> list[Any]:
    if event_ids:
        return [position for position in positions if position.event_id in event_ids]
    return [position for position in positions if position.event_id.startswith(event_prefix)]


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
    by_sleeve: dict[str, dict[str, Any]] = {}
    for sleeve in sorted({str(getattr(record, "sleeve", "settlement")) for record in records}):
        sleeve_records = [record for record in records if getattr(record, "sleeve", "settlement") == sleeve]
        sleeve_matched = [record for record in sleeve_records if record.match_status == "matched"]
        sleeve_theoretical = [
            record.theoretical_pnl
            for record in sleeve_matched
            if record.theoretical_pnl is not None
        ]
        sleeve_deltas = [
            record.cash_pnl_delta
            for record in sleeve_matched
            if record.cash_pnl_delta is not None
        ]
        by_sleeve[sleeve] = {
            "positions": len(sleeve_records),
            "matched": len(sleeve_matched),
            "missing_cash_flow": sum(
                1 for record in sleeve_records if record.match_status == "missing_cash_flow"
            ),
            "account_cash_pnl": round(
                sum(record.account_cash_pnl or 0.0 for record in sleeve_matched),
                8,
            ),
            "theoretical_pnl": round(sum(sleeve_theoretical), 8),
            "cash_minus_theoretical": round(sum(sleeve_deltas), 8),
        }
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
        "by_sleeve": by_sleeve,
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
        "## By Sleeve",
        "",
        "| Sleeve | Positions | Matched | Missing Cash Flow | Account PnL | Theoretical PnL | Delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for sleeve, item in summary["by_sleeve"].items():
        lines.append(
            "| "
            f"{sleeve} | "
            f"{item['positions']} | "
            f"{item['matched']} | "
            f"{item['missing_cash_flow']} | "
            f"{item['account_cash_pnl']:.6f} | "
            f"{item['theoretical_pnl']:.6f} | "
            f"{item['cash_minus_theoretical']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Positions",
            "",
            "| Event | Sleeve | Status | Side | Match | Account PnL | Theoretical PnL | Delta | Dust |",
            "|---|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for record in sorted(records, key=lambda item: (item.round_slug, item.event_id)):
        lines.append(
            "| "
            f"{record.event_id} | "
            f"{getattr(record, 'sleeve', 'settlement')} | "
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
