#!/usr/bin/env python3
"""Reconcile stale open execution rows from Polymarket account history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bigan.execution import (
    PositionManager,
    read_polymarket_history_csv,
    reconcile_stale_open_positions,
)
from bigan.execution.db import connect_mlops_db


def main() -> int:
    args = _parse_args()
    manager = PositionManager(args.db_path)
    flows = read_polymarket_history_csv(args.history_csv)
    results = reconcile_stale_open_positions(
        manager,
        flows,
        settlement_results=_parse_settlement_results(args.settlement_result),
    )
    summary = _summary(results)

    if args.report_path:
        Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_path).write_text(_markdown_report(results, summary), encoding="utf-8")
    if args.summary_json_path:
        Path(args.summary_json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json_path).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.write_cashflow_db:
        from bigan.execution import reconcile_cash_flows, record_cashflow_reconciliations

        positions = [
            position
            for position in manager.list_positions()
            if position.event_id.startswith(args.event_prefix)
        ]
        records = reconcile_cash_flows(positions, flows)
        with connect_mlops_db(args.db_path) as conn:
            record_cashflow_reconciliations(conn, records, replace=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-csv", required=True)
    parser.add_argument("--db-path", default="data/mlops/champion_catalog.duckdb")
    parser.add_argument("--event-prefix", default="phase4-")
    parser.add_argument(
        "--settlement-result",
        action="append",
        default=[],
        metavar="EVENT_ID=UP|DOWN",
        help=(
            "Authoritative resolved side for a stale event without SELL/REDEEM cash flow; "
            "may be repeated."
        ),
    )
    parser.add_argument("--write-cashflow-db", action="store_true")
    parser.add_argument("--report-path", default="")
    parser.add_argument("--summary-json-path", default="")
    return parser.parse_args()


def _parse_settlement_results(values: list[str]) -> dict[str, str]:
    results: dict[str, str] = {}
    for value in values:
        event_id, separator, side = value.partition("=")
        event_id = event_id.strip()
        side = side.strip().upper()
        if not separator or not event_id or side not in {"UP", "DOWN"}:
            raise SystemExit(
                "--settlement-result must be formatted as EVENT_ID=UP or EVENT_ID=DOWN"
            )
        results[event_id] = side
    return results


def _summary(results: list[Any]) -> dict[str, Any]:
    changed = [result for result in results if result.action != "unchanged"]
    return {
        "open_positions_seen": len(results),
        "reconciled": len(changed),
        "closed_from_sell": sum(1 for result in changed if result.action == "closed_from_sell"),
        "settled_from_redeem": sum(1 for result in changed if result.action == "settled_from_redeem"),
        "settled_from_provider": sum(
            1 for result in changed if result.action == "settled_from_provider"
        ),
        "unchanged": sum(1 for result in results if result.action == "unchanged"),
    }


def _markdown_report(results: list[Any], summary: dict[str, Any]) -> str:
    lines = [
        "# Stale Execution Position Reconciliation",
        "",
        "## Summary",
        "",
        f"- Open positions seen: {summary['open_positions_seen']}",
        f"- Reconciled: {summary['reconciled']}",
        f"- Closed from sell history: {summary['closed_from_sell']}",
        f"- Settled from redeem history: {summary['settled_from_redeem']}",
        f"- Unchanged: {summary['unchanged']}",
        "",
        "## Rows",
        "",
        "| Event | Action | Prior | New | Account PnL | Theoretical PnL |",
        "|---|---|---|---|---:|---:|",
    ]
    for result in results:
        lines.append(
            "| "
            f"{result.event_id} | "
            f"{result.action} | "
            f"{result.prior_status} | "
            f"{result.new_status} | "
            f"{_fmt(result.account_cash_pnl)} | "
            f"{_fmt(result.theoretical_pnl)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
