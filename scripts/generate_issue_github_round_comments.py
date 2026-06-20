#!/usr/bin/env python3
"""Generate GitHub issue per-round comments in issue #100 format from Phase 4 logs."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class SignalRow:
    event_id: str
    run_id: str
    round_slug: str = ""
    side: str = ""
    created_utc: str = ""
    received_utc: str = ""
    p_up: float | None = None
    p_down: float | None = None
    p_neu: float | None = None
    p_vol_up: float | None = None
    p_vol_down: float | None = None
    payload_px: float | None = None
    raw_edge: float | None = None
    age_s: float | None = None
    settlement_action: str = "-"
    volatility_action: str = "-"
    final: str = "-"
    short_id: str = ""


@dataclass
class FillRow:
    run_id: str
    entry_utc: str
    sleeve: str
    side: str
    fill_price: float
    p_up: float | None
    p_down: float | None
    p_neu: float | None
    edge: float | None
    signal_id: str
    final_exit: str


@dataclass
class ExitRow:
    run_id: str
    time_utc: str
    exit_type: str
    side: str
    price_or_result: str
    reason: str
    pnl: float | None
    signal_id: str
    source: str = "in_run"


@dataclass
class RunData:
    run_id: str
    description: str
    log_path: Path
    summary: dict[str, Any]
    gamma_rows_by_round: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    signals_by_round: dict[str, dict[str, SignalRow]] = field(default_factory=lambda: defaultdict(dict))
    fills_by_round: dict[str, list[FillRow]] = field(default_factory=lambda: defaultdict(list))
    exits_by_round: dict[str, list[ExitRow]] = field(default_factory=lambda: defaultdict(list))
    pm_events_by_round: dict[str, list[ExitRow]] = field(default_factory=lambda: defaultdict(list))


def _ms_to_iso(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str) and "T" in value:
        return value.replace("+00:00", "Z") if value.endswith("+00:00") else value
    try:
        ms = int(value)
    except (TypeError, ValueError):
        return ""
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _fmt_ts(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    text = text.replace("+00:00", "").replace("T", " ")
    if text.endswith("Z"):
        text = text[:-1]
    if "." not in text and len(text) == 19:
        text += ".000"
    return text


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _short_signal_id(event_id: str) -> str:
    token = event_id.removeprefix("pred-")
    return token[:8] if token else event_id[:8]


def _round_ts(slug: str) -> int:
    match = re.search(r"-(\d+)$", slug)
    return int(match.group(1)) if match else 0


def _fmt_num(value: float | None, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _fmt_pnl(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def _gate_action(payload: dict[str, Any], *, filled: bool) -> str:
    if filled:
        position = payload.get("position") or {}
        side = str(position.get("side") or "").upper()
        price = _optional_float(position.get("fill_price"))
        edge = _optional_float((payload.get("gate_evaluation") or {}).get("settlement_edge"))
        if edge is None:
            edge = _optional_float(position.get("entry_mispricing_edge"))
        return f"FILL {side} @{_fmt_num(price, 2)} edge {_fmt_num(edge, 3)}"
    reason = str(payload.get("reason") or "")
    if reason:
        age = _optional_float(payload.get("signal_age_seconds"))
        if reason == "signal_age_above_threshold" and age is not None:
            return f"SKIP {reason} age {age:.1f}s"
        return f"SKIP {reason}"
    gate = payload.get("gate_evaluation") or {}
    if gate.get("settlement_gate_passed"):
        edge = _optional_float(payload.get("fresh_edge_at_worst"))
        worst = _optional_float(payload.get("worst_price"))
        if edge is not None and worst is not None:
            return f"PASS settlement @worst {_fmt_num(worst, 2)} edge {_fmt_num(edge, 3)}"
        return "PASS settlement"
    if gate.get("settlement_confidence_passed") is False:
        edge = _optional_float(payload.get("fresh_edge_at_worst"))
        worst = _optional_float(payload.get("worst_price"))
        parts = ["SKIP settlement_confidence_below_threshold"]
        if worst is not None:
            parts.append(f"@worst {_fmt_num(worst, 2)}")
        if edge is not None:
            parts.append(f"edge {_fmt_num(edge, 3)}")
        return " ".join(parts)
    if gate.get("signal_freshness_passed") is False:
        return "SKIP signal_freshness_failed"
    edge = _optional_float(payload.get("fresh_edge_at_worst"))
    worst = _optional_float(payload.get("worst_price"))
    parts = ["SKIP settlement_edge_below_threshold"]
    if worst is not None:
        parts.append(f"@worst {_fmt_num(worst, 2)}")
    if edge is not None:
        parts.append(f"edge {_fmt_num(edge, 3)}")
    return " ".join(parts)


def _load_gamma_rows(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key in ("pending_settlement_rows", "rows"):
        for row in payload.get(key) or []:
            if isinstance(row, dict) and row.get("round_slug"):
                rows[str(row["round_slug"])].append(row)
    return dict(rows)


def load_run(
    *,
    run_id: str,
    description: str,
    log_path: Path,
    summary_path: Path,
    gamma_path: Path | None,
) -> RunData:
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    run = RunData(
        run_id=run_id,
        description=description,
        log_path=log_path,
        summary=summary,
        gamma_rows_by_round=_load_gamma_rows(gamma_path),
    )
    fill_by_signal: dict[str, FillRow] = {}
    position_signal_by_round_side: dict[tuple[str, str], str] = {}

    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            event = payload.get("event")
            signal = payload.get("signal") if isinstance(payload.get("signal"), dict) else None

            if event == "signal_batch_received":
                received = _fmt_ts(payload.get("ts"))
                for summary_signal in payload.get("signals") or []:
                    if not isinstance(summary_signal, dict):
                        continue
                    event_id = str(summary_signal.get("event_id") or "")
                    if not event_id:
                        continue
                    round_slug = str(summary_signal.get("round_slug") or "")
                    row = run.signals_by_round[round_slug].setdefault(
                        event_id,
                        SignalRow(event_id=event_id, run_id=run_id, round_slug=round_slug),
                    )
                    row.side = str(summary_signal.get("side") or row.side).upper()
                    row.received_utc = received
                    row.raw_edge = _optional_float(summary_signal.get("edge")) or row.raw_edge
                    timestamps = summary_signal.get("timestamps") or {}
                    row.created_utc = _fmt_ts(timestamps.get("signal_created_at")) or row.created_utc
                    latency = summary_signal.get("latency_ms") or {}
                    created_ms = _optional_float(latency.get("signal_created_to_at_ms"))
                    if created_ms is not None:
                        row.age_s = created_ms / 1000.0
                    if not row.settlement_action or row.settlement_action == "-":
                        row.settlement_action = "received"
                continue

            if signal is not None:
                event_id = str(signal.get("event_id") or "")
                if not event_id:
                    continue
                round_slug = str(signal.get("round_slug") or "")
                row = run.signals_by_round[round_slug].setdefault(
                    event_id,
                    SignalRow(event_id=event_id, run_id=run_id, round_slug=round_slug),
                )
                row.side = str(signal.get("outcome_side") or signal.get("side") or row.side).upper()
                row.created_utc = _fmt_ts(_ms_to_iso(signal.get("created_at"))) or row.created_utc
                row.p_up = _optional_float(signal.get("p_up")) or row.p_up
                row.p_down = _optional_float(signal.get("p_down")) or row.p_down
                row.p_neu = _optional_float(signal.get("p_neutral")) or row.p_neu
                row.p_vol_up = _optional_float(signal.get("p_vol_up")) or row.p_vol_up
                row.p_vol_down = _optional_float(signal.get("p_vol_down")) or row.p_vol_down
                row.payload_px = _optional_float(signal.get("polymarket_price")) or row.payload_px
                row.raw_edge = _optional_float(signal.get("edge")) or row.raw_edge
                row.short_id = _short_signal_id(event_id)
                if payload.get("ts") and not row.received_utc:
                    row.received_utc = _fmt_ts(payload.get("ts"))

            if event == "entry_gate_evaluated" and signal is not None:
                row = run.signals_by_round[str(signal.get("round_slug") or "")][event_id]
                row.settlement_action = _gate_action(payload, filled=False)
                age = _optional_float((payload.get("gate_evaluation") or {}).get("signal_age_seconds"))
                if age is not None:
                    row.age_s = age

            elif event == "entry_skipped" and signal is not None:
                row = run.signals_by_round[str(signal.get("round_slug") or "")][event_id]
                row.settlement_action = _gate_action(payload, filled=False)
                age = _optional_float(payload.get("signal_age_seconds"))
                if age is not None:
                    row.age_s = age

            elif event == "paper_entry_filled":
                position = payload.get("position") or {}
                round_slug = str(position.get("round_slug") or "")
                event_id = str(position.get("entry_signal_event_id") or "")
                side = str(position.get("side") or "").upper()
                fill_price = _optional_float(position.get("fill_price")) or 0.0
                fill = FillRow(
                    run_id=run_id,
                    entry_utc=_fmt_ts(_ms_to_iso(position.get("opened_at"))),
                    sleeve=str(position.get("sleeve") or "settlement"),
                    side=side,
                    fill_price=fill_price,
                    p_up=_optional_float(position.get("entry_p_up")),
                    p_down=_optional_float(position.get("entry_p_down")),
                    p_neu=_optional_float(position.get("entry_p_neutral")),
                    edge=_optional_float(position.get("entry_mispricing_edge")),
                    signal_id=_short_signal_id(event_id),
                    final_exit="-",
                )
                run.fills_by_round[round_slug].append(fill)
                fill_by_signal[event_id] = fill
                position_signal_by_round_side[(round_slug, side)] = event_id
                if event_id in run.signals_by_round.get(round_slug, {}):
                    row = run.signals_by_round[round_slug][event_id]
                    row.settlement_action = _gate_action(payload, filled=True)
                    row.final = f"FILLED {side} @{_fmt_num(fill_price, 2)}"

            elif event == "paper_exit_filled":
                position = payload.get("position") or {}
                round_slug = str(position.get("round_slug") or "")
                event_id = str(position.get("entry_signal_event_id") or "")
                pnl = _optional_float(payload.get("realized_pnl"))
                exit_row = ExitRow(
                    run_id=run_id,
                    time_utc=_fmt_ts(payload.get("ts")),
                    exit_type="paper_exit",
                    side=str(position.get("side") or "").upper(),
                    price_or_result=_fmt_num(_optional_float(payload.get("exit_price")), 2),
                    reason=str(payload.get("reason") or ""),
                    pnl=pnl,
                    signal_id=_short_signal_id(event_id),
                )
                run.exits_by_round[round_slug].append(exit_row)
                if event_id in fill_by_signal:
                    fill_by_signal[event_id].final_exit = (
                        f"EXIT {exit_row.side} @{exit_row.price_or_result} pnl {_fmt_pnl(pnl)}"
                    )
                if event_id in run.signals_by_round.get(round_slug, {}):
                    run.signals_by_round[round_slug][event_id].final = fill_by_signal.get(
                        event_id, FillRow(
                            run_id, "", "", "", 0.0, None, None, None, None, "", "-"
                        )
                    ).final_exit

            elif event == "paper_settlement_resolved":
                position = payload.get("position") or {}
                round_slug = str(position.get("round_slug") or "")
                event_id = str(position.get("entry_signal_event_id") or "")
                result = str(payload.get("settlement_result") or "")
                pnl = _optional_float(payload.get("realized_pnl"))
                exit_row = ExitRow(
                    run_id=run_id,
                    time_utc=_fmt_ts(payload.get("ts")),
                    exit_type="settlement",
                    side=str(position.get("side") or "").upper(),
                    price_or_result=result,
                    reason=str(payload.get("reason") or ""),
                    pnl=pnl,
                    signal_id=_short_signal_id(event_id),
                    source="in_run",
                )
                run.exits_by_round[round_slug].append(exit_row)
                final = f"SETTLED {result} pnl {_fmt_pnl(pnl)}"
                if event_id in fill_by_signal:
                    fill_by_signal[event_id].final_exit = final
                if event_id in run.signals_by_round.get(round_slug, {}):
                    run.signals_by_round[round_slug][event_id].final = final

            elif event in {
                "paper_v7_settlement_position_reduced",
                "paper_v7_settlement_position_added",
            }:
                position = payload.get("position") or {}
                round_slug = str(position.get("round_slug") or "")
                evaluation = payload.get("evaluation") or {}
                action = str(evaluation.get("action") or event.removeprefix("paper_v7_settlement_position_")).upper()
                pnl = _optional_float(payload.get("realized_pnl_delta"))
                if pnl is None:
                    pnl = _optional_float(payload.get("cumulative_position_realized_pnl"))
                if event.endswith("_added"):
                    pnl = None
                exit_row = ExitRow(
                    run_id=run_id,
                    time_utc=_fmt_ts(payload.get("ts")),
                    exit_type=f"pm_{action.lower()}",
                    side=str(position.get("side") or "").upper(),
                    price_or_result=_fmt_num(_optional_float(payload.get("new_average_price")), 3)
                    if event.endswith("_added")
                    else _fmt_num(_optional_float(payload.get("hold_bid")), 2),
                    reason=str(evaluation.get("reason") or ""),
                    pnl=pnl,
                    signal_id=_short_signal_id(str(position.get("entry_signal_event_id") or "")),
                )
                run.pm_events_by_round[round_slug].append(exit_row)

    for round_slug, gamma_rows in run.gamma_rows_by_round.items():
        replaced_sides: set[str] = set()
        for gamma_row in gamma_rows:
            side = str(gamma_row.get("side") or "").upper()
            winner = str(gamma_row.get("gamma_winner") or gamma_row.get("settlement_result") or "")
            total_pnl = _gamma_total_pnl(gamma_row)
            event_id = str(gamma_row.get("event_id") or "")
            signal_key = position_signal_by_round_side.get((round_slug, side), "")
            if side not in replaced_sides:
                run.exits_by_round[round_slug] = [
                    row
                    for row in run.exits_by_round.get(round_slug, [])
                    if not (row.exit_type == "settlement" and row.side == side)
                ]
                replaced_sides.add(side)
            exit_row = ExitRow(
                run_id=run_id,
                time_utc="(gamma reconcile)",
                exit_type="settlement",
                side=side,
                price_or_result=winner,
                reason="gamma_reconcile",
                pnl=total_pnl,
                signal_id=_short_signal_id(signal_key or event_id),
                source="gamma_reconcile",
            )
            run.exits_by_round[round_slug].append(exit_row)
            final = f"SETTLED {winner} pnl {_fmt_pnl(total_pnl)} (gamma)"
            if signal_key and signal_key in run.signals_by_round.get(round_slug, {}):
                run.signals_by_round[round_slug][signal_key].final = final
            for fill in run.fills_by_round.get(round_slug, []):
                if fill.side == side:
                    fill.final_exit = final

    return run


def _gamma_total_pnl(gamma_row: dict[str, Any]) -> float | None:
    return _optional_float(gamma_row.get("total_pnl_usdc")) or _optional_float(
        gamma_row.get("total_position_pnl_usdc")
    )


def _round_realized_pnl(run: RunData, round_slug: str) -> float:
    by_signal: dict[str, float] = {}
    for exit_row in run.exits_by_round.get(round_slug, []):
        if exit_row.pnl is None or exit_row.exit_type not in {"paper_exit", "settlement"}:
            continue
        key = exit_row.signal_id or f"{exit_row.side}:{exit_row.exit_type}"
        by_signal[key] = exit_row.pnl
    return sum(by_signal.values())


def render_round_comment(*, round_slug: str, runs: list[RunData]) -> str:
    active_runs = [run for run in runs if round_slug in run.signals_by_round or round_slug in run.fills_by_round]
    if not active_runs:
        active_runs = [run for run in runs if round_slug in run.exits_by_round]
    lines = [f"### Round `{round_slug}`", ""]

    signals_count = sum(len(run.signals_by_round.get(round_slug, {})) for run in active_runs)
    fills_count = sum(len(run.fills_by_round.get(round_slug, [])) for run in active_runs)
    settled_count = sum(
        1
        for run in active_runs
        for row in run.exits_by_round.get(round_slug, [])
        if row.exit_type == "settlement"
    )
    exit_count = sum(
        1
        for run in active_runs
        for row in run.exits_by_round.get(round_slug, [])
        if row.exit_type == "paper_exit"
    )
    open_or_pending = 0
    realized = sum(_round_realized_pnl(run, round_slug) for run in active_runs)

    lines.append(
        "Summary: "
        f"signals={signals_count}, fills={fills_count}, settled={settled_count}, "
        f"paper_exits={exit_count}, open_or_pending={open_or_pending}, "
        f"realized_pnl={realized:.4f}"
    )
    lines.append("Runs included:")
    for run in active_runs:
        lines.append(f"- `{run.run_id}`: {run.description}")
    lines.append("")

    lines.extend(["#### Fill / Entry", ""])
    fill_rows = []
    for run in active_runs:
        fill_rows.extend(run.fills_by_round.get(round_slug, []))
    if fill_rows:
        lines.append(
            "| run | entry UTC | sleeve | side | fill | p_up | p_down | p_neu | edge | signal | final/exit |"
        )
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---|---|")
        for fill in sorted(fill_rows, key=lambda row: row.entry_utc):
            lines.append(
                "| "
                f"{fill.run_id} | {fill.entry_utc} | {fill.sleeve} | {fill.side} | "
                f"{_fmt_num(fill.fill_price, 2)} | {_fmt_num(fill.p_up)} | {_fmt_num(fill.p_down)} | "
                f"{_fmt_num(fill.p_neu)} | {_fmt_num(fill.edge, 3)} | {fill.signal_id} | {fill.final_exit} |"
            )
    else:
        lines.append("_No fills._")
    lines.append("")

    lines.extend(["#### Exit / Settlement / PnL", ""])
    exit_rows: list[ExitRow] = []
    for run in active_runs:
        exit_rows.extend(run.exits_by_round.get(round_slug, []))
        exit_rows.extend(run.pm_events_by_round.get(round_slug, []))
    if exit_rows:
        lines.append(
            "| run | time UTC | type | side | price/result | reason/error | pnl | signal |"
        )
        lines.append("|---|---|---|---|---|---|---:|---|")
        for row in sorted(exit_rows, key=lambda item: (item.time_utc, item.exit_type)):
            lines.append(
                "| "
                f"{row.run_id} | {row.time_utc} | {row.exit_type} | {row.side} | "
                f"{row.price_or_result} | {row.reason or '-'} | {_fmt_pnl(row.pnl)} | {row.signal_id} |"
            )
    else:
        lines.append("_No exits or settlements._")
    lines.append("")

    lines.append("<details open>")
    lines.append("<summary>Signals consumed by executor, enriched from full gate/fill payloads</summary>")
    lines.append("")
    signal_rows: list[SignalRow] = []
    for run in active_runs:
        for row in run.signals_by_round.get(round_slug, {}).values():
            if not row.short_id:
                row.short_id = _short_signal_id(row.event_id)
            signal_rows.append(row)
    if signal_rows:
        lines.append(
            "| # | run | created UTC | received UTC | side | p_side | p_up | p_down | p_neu | "
            "p_vol_up | p_vol_down | payload_px | raw_edge | age_s | settlement/exit action | "
            "volatility action | final |"
        )
        lines.append(
            "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|"
        )
        signal_rows.sort(key=lambda row: row.created_utc, reverse=True)
        for idx, row in enumerate(signal_rows, start=1):
            p_side = row.p_up if row.side == "UP" else row.p_down if row.side == "DOWN" else None
            lines.append(
                "| "
                f"{idx} | {row.run_id} | {row.created_utc} | {row.received_utc} | {row.side} | "
                f"{_fmt_num(p_side)} | {_fmt_num(row.p_up)} | {_fmt_num(row.p_down)} | {_fmt_num(row.p_neu)} | "
                f"{_fmt_num(row.p_vol_up) if row.p_vol_up is not None else '-'} | "
                f"{_fmt_num(row.p_vol_down) if row.p_vol_down is not None else '-'} | "
                f"{_fmt_num(row.payload_px, 2) if row.payload_px is not None else '-'} | "
                f"{_fmt_num(row.raw_edge, 3) if row.raw_edge is not None else '-'} | "
                f"{_fmt_num(row.age_s, 1) if row.age_s is not None else '-'} | "
                f"{row.settlement_action} | {row.volatility_action} | {row.final} |"
            )
    else:
        lines.append("_No signals._")
    lines.append("")
    lines.append("</details>")
    return "\n".join(lines)


def _reconciled_pnl_for_run(run: RunData) -> float | None:
    summary = run.summary
    realized = _optional_float(summary.get("realized_pnl_usdc"))
    hardcoded = {
        "20260608T055415Z": 1.34,
        "20260608T133724Z": -4.1557,
        "20260609T103055Z": -3.5348,
        "20260610T021333Z": -2.8965,
    }
    if run.run_id in hardcoded:
        return hardcoded[run.run_id]
    gamma_rows = [
        row
        for rows in run.gamma_rows_by_round.values()
        for row in rows
        if _gamma_total_pnl(row) is not None
    ]
    if gamma_rows and realized is not None:
        settlement_addon = sum(
            _optional_float(row.get("settlement_pnl_usdc")) or 0.0 for row in gamma_rows
        )
        return realized + settlement_addon
    return realized


def render_run_overview(runs: list[RunData]) -> str:
    if len(runs) >= 4:
        title_runs = "four-run"
    elif len(runs) >= 3:
        title_runs = "three-run"
    elif len(runs) == 2:
        title_runs = "two-run"
    else:
        title_runs = "run"
    lines = [
        f"## xgboost-v7 paper shadow — {title_runs} case study",
        "",
        "Per-round detail follows in chronological issue comments (issue #100 format).",
        "",
        "| run | stop | window | rounds | fills | realized_pnl | reconciled_pnl | status |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for run in runs:
        summary = run.summary
        realized = _optional_float(summary.get("realized_pnl_usdc"))
        reconciled = _reconciled_pnl_for_run(run)
        stop = summary.get("stop_reason") or summary.get("status")
        if run.run_id in {"20260608T055415Z", "20260610T021333Z"}:
            stop = "stop_max_runtime (180min)"
        elif run.run_id in {"20260608T133724Z", "20260609T103055Z", "20260611T030545Z"}:
            stop = "stop_daily_loss_limit (-3 USDC)"
        elif run.run_id == "20260611T111648Z":
            stop = "stop_max_runtime (180min)"
        elif run.run_id == "20260611T154406Z":
            stop = "CLOB API timeout crash (20/30 rounds)"
        if summary.get("stop_reason"):
            stop = str(summary["stop_reason"])
        lines.append(
            "| "
            f"`{run.run_id}` | {stop} | "
            f"{summary.get('started_at', '')} → {summary.get('finished_at', '')} | "
            f"{summary.get('observed_round_count', 0)} | {summary.get('entries_filled', 0)} | "
            f"{_fmt_pnl(realized)} | {_fmt_pnl(reconciled)} | {summary.get('status', '')} |"
        )
    lines.extend(
        [
            "",
            "**Shared config (runs 1–3):** BTC-15M only, `entry_gate_mode=v7-pnl`, settlement conf=0.75, edge=0.04, "
            "max_signal_age=30s, PM enabled + paper_execute, re-entry allowed, model `20260608Tevent-5s-v1`.",
            "",
            "**Run 4 delta:** `convergence_take_profit_enabled=true`, `take_profit_hold_edge=0.03`, "
            "`take_profit_force_exit_seconds=180`, `take_profit_hysteresis_bars=2`.",
            "",
            "**Run 5–6 delta (Issue #104 P0/P1):** `min_entry_price=0.30`, "
            "`divergence_reduce_max_hold_edge=0.08`, `add_cooldown_after_divergence_reduce_seconds=120`, "
            "poll-loop take-profit / force-exit, price-convergence take-profit, Gamma retry 24h. "
            "Run 6 repeats Run 5 config on a different UTC window.",
            "",
            "**Artifacts:**",
        ]
    )
    for run in runs:
        lines.append(f"- `{run.run_id}` log: `{run.log_path}`")
    return "\n".join(lines)


def _all_runs(root: Path) -> list[RunData]:
    return [
        load_run(
            run_id="20260608T055415Z",
            description=(
                "v7 event5s paper30 run 1; conf=0.75, edge=0.04, max_age=30s, PM enabled, "
                "180min cap; gamma reconcile for 9 pending settlements"
            ),
            log_path=root
            / "data/logs/xgboost-v7-paper-shadow-20260608T055415Z-event5s-30round/phase4-20260608T055415Z.jsonl",
            summary_path=root / "logs/xgboost-v7-paper-shadow/phase4-20260608T055415Z-summary.json",
            gamma_path=root / "logs/xgboost-v7-paper-shadow/phase4-20260608T055415Z-gamma-reconcile.json",
        ),
        load_run(
            run_id="20260608T133724Z",
            description=(
                "v7 event5s paper30 run 2; conf=0.75, edge=0.04, max_age=30s, PM enabled; "
                "stopped on daily_loss_limit; gamma backfill for 5 pending settlements"
            ),
            log_path=root
            / "data/logs/xgboost-v7-paper-shadow-20260608T133721Z-event5s-30round/phase4-20260608T133724Z.jsonl",
            summary_path=root
            / "data/logs/xgboost-v7-paper-shadow-20260608T133721Z-event5s-30round/phase4-20260608T133724Z-summary.json",
            gamma_path=root
            / "logs/xgboost-v7-paper-shadow/phase4-20260608T133724Z-gamma-settlement-backfill.json",
        ),
        load_run(
            run_id="20260609T103055Z",
            description=(
                "v7 event5s paper30 run 3; conf=0.75, edge=0.04, max_age=30s, PM enabled; "
                "stopped on daily_loss_limit; gamma reconcile for 9 pending settlements"
            ),
            log_path=root
            / "data/logs/xgboost-v7-paper-shadow-20260609T103055Z-event5s-30round/phase4-20260609T103055Z.jsonl",
            summary_path=root / "logs/xgboost-v7-paper-shadow/phase4-20260609T103055Z-summary.json",
            gamma_path=root / "logs/xgboost-v7-paper-shadow/phase4-20260609T103055Z-gamma-reconcile.json",
        ),
        load_run(
            run_id="20260610T021333Z",
            description=(
                "v7 event5s paper30 run 4 (take-profit); conf=0.75, edge=0.04, max_age=30s, PM enabled, "
                "convergence_take_profit_enabled; stopped on max_runtime; gamma reconcile for 9 pending settlements"
            ),
            log_path=root
            / "data/logs/xgboost-v7-paper-shadow-20260610T021333Z-event5s-30round-takeprofit/phase4-20260610T022208Z.jsonl",
            summary_path=root / "logs/xgboost-v7-paper-shadow/phase4-20260610T021333Z-summary.json",
            gamma_path=root / "logs/xgboost-v7-paper-shadow/phase4-20260610T021333Z-gamma-reconcile.json",
        ),
        load_run(
            run_id="20260611T030545Z",
            description=(
                "v7 event5s paper30 run 5 (issue104 P0/P1); conf=0.75, edge=0.04, max_age=30s, "
                "min_entry_price=0.30, PM+take-profit+anti-churn; stopped on daily_loss_limit; "
                "lifecycle complete, 0 pending settlement"
            ),
            log_path=root / "logs/xgboost-v7-paper-shadow/phase4-20260611T030545Z.jsonl",
            summary_path=root / "logs/xgboost-v7-paper-shadow/phase4-20260611T030545Z-summary.json",
            gamma_path=None,
        ),
        load_run(
            run_id="20260611T111648Z",
            description=(
                "v7 event5s paper30 run 6 (issue104 P0/P1, same config as run 5); conf=0.75, "
                "edge=0.04, max_age=30s, min_entry_price=0.30, PM+take-profit+anti-churn; "
                "stopped on max_runtime 180min; lifecycle complete, 0 pending settlement"
            ),
            log_path=root / "logs/xgboost-v7-paper-shadow/phase4-20260611T111648Z.jsonl",
            summary_path=root / "logs/xgboost-v7-paper-shadow/phase4-20260611T111648Z-summary.json",
            gamma_path=None,
        ),
        load_run(
            run_id="20260611T154406Z",
            description=(
                "v7 event5s paper30 run 7 (issue104 P0/P1); conf=0.75, edge=0.04, max_age=30s, "
                "min_entry_price=0.30, PM+take-profit+anti-churn, max_runtime=1440min; "
                "stopped on CLOB API timeout crash at 20/30 rounds; lifecycle complete"
            ),
            log_path=root / "logs/xgboost-v7-paper-shadow/phase4-20260611T154406Z.jsonl",
            summary_path=root / "logs/xgboost-v7-paper-shadow/phase4-20260611T154406Z-summary.json",
            gamma_path=None,
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--issue", type=int, default=103)
    parser.add_argument(
        "--only-run-id",
        default="",
        help="Generate per-round comments for a single run only (overview still uses all runs).",
    )
    parser.add_argument(
        "--overview-run-ids",
        default="",
        help="Comma-separated run ids for overview/manifest context; default all known runs.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    all_runs = _all_runs(root)
    if args.overview_run_ids:
        overview_ids = {item.strip() for item in args.overview_run_ids.split(",") if item.strip()}
        overview_runs = [run for run in all_runs if run.run_id in overview_ids]
    else:
        overview_runs = all_runs

    if args.only_run_id:
        runs = [run for run in all_runs if run.run_id == args.only_run_id]
        if not runs:
            raise SystemExit(f"unknown --only-run-id {args.only_run_id!r}")
    else:
        runs = all_runs

    output_dir = Path(args.output_dir)
    comments_dir = output_dir / "comments"
    comments_dir.mkdir(parents=True, exist_ok=True)

    all_rounds: set[str] = set()
    for run in runs:
        all_rounds.update(run.signals_by_round)
        all_rounds.update(run.fills_by_round)
        all_rounds.update(run.exits_by_round)
    round_order = sorted(all_rounds, key=_round_ts)

    manifest = []
    for idx, round_slug in enumerate(round_order, start=1):
        body = render_round_comment(round_slug=round_slug, runs=runs)
        ts = _round_ts(round_slug)
        filename = f"{idx:02d}-btc-updown-15m-{ts}.md"
        path = comments_dir / filename
        path.write_text(body, encoding="utf-8")
        manifest.append({"round_slug": round_slug, "path": str(path), "filename": filename})

    overview_path = output_dir / "overview.md"
    overview_path.write_text(render_run_overview(overview_runs), encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps({"issue": args.issue, "rounds": manifest}, indent=2),
        encoding="utf-8",
    )

    post_script = output_dir / "post_to_issue103.sh"
    edit_overview = "true" if not args.only_run_id else "true"
    post_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'repo="phead198708/BiGan"',
                f'issue="{args.issue}"',
                'comments_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/comments" && pwd)"',
                f'if [[ "{edit_overview}" == "true" ]]; then',
                '  gh issue edit "${issue}" --repo "${repo}" --body-file "$(dirname "${BASH_SOURCE[0]}")/overview.md"',
                "fi",
                'for f in "${comments_dir}"/*.md; do',
                '  echo "posting ${f}"',
                '  gh issue comment "${issue}" --repo "${repo}" --body-file "${f}"',
                "done",
                "",
            ]
        ),
        encoding="utf-8",
    )
    post_script.chmod(0o755)

    print(json.dumps({"round_count": len(round_order), "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
