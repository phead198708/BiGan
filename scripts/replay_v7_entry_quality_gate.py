#!/usr/bin/env python3
"""Replay v7 entry-quality gates against actual paper fills.

This is a lightweight, log-derived counterfactual: it filters actual
paper_entry_filled bets by entry metadata and recomputes PnL from the matching
per-bet records. It does not synthesize replacement fills that might have
appeared after freeing a concurrent-position slot.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def main() -> int:
    args = _parse_args()
    entries = _load_entry_metadata([Path(item) for item in args.executor_jsonl])
    bets = _load_per_bet_records([Path(item) for item in args.per_bet_json])
    report = _build_report(entries=entries, bets=bets, args=args)
    if args.output_json_path:
        path = Path(args.output_json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.report_path:
        path = Path(args.report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_markdown_report(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executor-jsonl", action="append", required=True)
    parser.add_argument("--per-bet-json", action="append", required=True)
    parser.add_argument("--entry-max-signal-age-seconds", type=float, default=None)
    parser.add_argument("--entry-max-price-drift-from-signal", type=float, default=None)
    parser.add_argument("--entry-raw-side-min-probability", type=float, default=None)
    parser.add_argument("--entry-raw-side-min-margin", type=float, default=None)
    parser.add_argument("--entry-raw-side-max-opposite-lead", type=float, default=None)
    parser.add_argument("--output-json-path", default="")
    parser.add_argument("--report-path", default="")
    args = parser.parse_args()
    if (
        args.entry_max_signal_age_seconds is not None
        and args.entry_max_signal_age_seconds < 0
    ):
        raise ValueError("--entry-max-signal-age-seconds must be non-negative")
    if (
        args.entry_max_price_drift_from_signal is not None
        and args.entry_max_price_drift_from_signal < 0
    ):
        raise ValueError("--entry-max-price-drift-from-signal must be non-negative")
    if (
        args.entry_raw_side_min_probability is not None
        and not 0.0 <= args.entry_raw_side_min_probability <= 1.0
    ):
        raise ValueError("--entry-raw-side-min-probability must be between 0 and 1")
    if (
        args.entry_raw_side_max_opposite_lead is not None
        and args.entry_raw_side_max_opposite_lead < 0
    ):
        raise ValueError("--entry-raw-side-max-opposite-lead must be non-negative")
    if args.entry_raw_side_min_margin is not None and args.entry_raw_side_min_margin < 0:
        raise ValueError("--entry-raw-side-min-margin must be non-negative")
    return args


def _load_entry_metadata(paths: list[Path]) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload.get("event") != "paper_entry_filled":
                    continue
                position = payload.get("position") or {}
                signal = payload.get("signal") or {}
                event_id = str(position.get("event_id") or "")
                if not event_id:
                    continue
                gate = payload.get("gate_evaluation") or {}
                signal_age = _float(gate.get("signal_age_seconds"))
                if signal_age is None:
                    opened_at = _millis(position.get("opened_at") or payload.get("ts"))
                    created_at = _millis(signal.get("created_at") or position.get("entry_signal_created_at"))
                    if opened_at > 0 and created_at > 0:
                        signal_age = max(0.0, (opened_at - created_at) / 1000.0)
                entry_price = _float(position.get("entry_price") or position.get("fill_price"))
                signal_price = (
                    _float(position.get("entry_polymarket_price"))
                    or _float(signal.get("polymarket_price"))
                    or _float(signal.get("market_implied_prob"))
                )
                price_drift = None
                if entry_price is not None and signal_price is not None:
                    price_drift = entry_price - signal_price
                entries[event_id] = {
                    "event_id": event_id,
                    "source_path": str(path),
                    "source_line": line_number,
                    "round_slug": position.get("round_slug") or signal.get("round_slug"),
                    "side": position.get("side") or signal.get("outcome_side"),
                    "entry_p_up": _float(signal.get("p_up") or position.get("entry_p_up")),
                    "entry_p_down": _float(signal.get("p_down") or position.get("entry_p_down")),
                    "entry_price": entry_price,
                    "entry_signal_price": signal_price,
                    "entry_price_drift_from_signal": price_drift,
                    "entry_signal_age_seconds": signal_age,
                    "entry_model_probability": (
                        _float(position.get("entry_model_probability"))
                        or _float(signal.get("model_probability"))
                        or _float(signal.get("token_expected_win_probability"))
                    ),
                    "entry_edge": (
                        _float(gate.get("settlement_edge"))
                        or _float(position.get("entry_mispricing_edge"))
                        or _float(signal.get("mispricing_edge"))
                        or _float(signal.get("edge"))
                    ),
                    "signal_id": signal.get("event_id") or position.get("entry_signal_event_id"),
                    "entry_ts": payload.get("ts"),
                }
    return entries


def _load_per_bet_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        run_id = str(payload.get("run_id") or path.stem)
        for bet in payload.get("bets") or []:
            event_id = str(bet.get("event_id") or "")
            if not event_id:
                continue
            pnl = _float(bet.get("total_pnl"))
            if pnl is None:
                pnl = _float(bet.get("realized_pnl"))
            record = dict(bet)
            record["run_id"] = run_id
            record["event_id"] = event_id
            record["pnl"] = pnl
            record["source_path"] = str(path)
            records.append(record)
    return records


def _build_report(
    *,
    entries: dict[str, dict[str, Any]],
    bets: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    unmatched_bets = 0
    for bet in bets:
        entry = entries.get(str(bet.get("event_id") or ""))
        if entry is None:
            unmatched_bets += 1
            continue
        reason = _entry_quality_skip_reason(entry, args)
        pnl = _float(bet.get("pnl")) or 0.0
        row = {
            "run_id": bet.get("run_id"),
            "event_id": bet.get("event_id"),
            "round_slug": bet.get("round_slug") or entry.get("round_slug"),
            "side": bet.get("side") or entry.get("side"),
            "entry_price": bet.get("entry_price") or entry.get("entry_price"),
            "entry_signal_price": entry.get("entry_signal_price"),
            "entry_price_drift_from_signal": entry.get("entry_price_drift_from_signal"),
            "entry_signal_age_seconds": entry.get("entry_signal_age_seconds"),
            "entry_edge": entry.get("entry_edge"),
            "entry_p_up": entry.get("entry_p_up"),
            "entry_p_down": entry.get("entry_p_down"),
            "exit_reason": bet.get("exit_reason"),
            "pnl": pnl,
            "skip_reason": reason,
        }
        rows.append(row)
        if reason is not None:
            skipped_rows.append(row)
    kept_rows = [row for row in rows if row["skip_reason"] is None]
    baseline_pnl = sum(float(row["pnl"]) for row in rows)
    replay_pnl = sum(float(row["pnl"]) for row in kept_rows)
    skipped_pnl = sum(float(row["pnl"]) for row in skipped_rows)
    return {
        "config": {
            "entry_max_signal_age_seconds": args.entry_max_signal_age_seconds,
            "entry_max_price_drift_from_signal": args.entry_max_price_drift_from_signal,
            "entry_raw_side_min_probability": args.entry_raw_side_min_probability,
            "entry_raw_side_min_margin": args.entry_raw_side_min_margin,
            "entry_raw_side_max_opposite_lead": args.entry_raw_side_max_opposite_lead,
        },
        "summary": {
            "matched_bets": len(rows),
            "unmatched_bets": unmatched_bets,
            "kept_bets": len(kept_rows),
            "skipped_bets": len(skipped_rows),
            "baseline_pnl": baseline_pnl,
            "replay_pnl": replay_pnl,
            "skipped_pnl": skipped_pnl,
            "pnl_delta": replay_pnl - baseline_pnl,
            "skip_reason_counts": dict(sorted(Counter(row["skip_reason"] for row in skipped_rows).items())),
            "by_run": _group_rows_by_run(rows),
        },
        "skipped_entries": skipped_rows,
        "rows": rows,
    }


def _entry_quality_skip_reason(entry: dict[str, Any], args: argparse.Namespace) -> str | None:
    max_age = args.entry_max_signal_age_seconds
    age = _float(entry.get("entry_signal_age_seconds"))
    if max_age is not None and age is not None and age > max_age:
        return "entry_signal_age_above_replay_threshold"
    max_drift = args.entry_max_price_drift_from_signal
    drift = _float(entry.get("entry_price_drift_from_signal"))
    if max_drift is not None and drift is not None and drift > max_drift:
        return "entry_price_drift_above_replay_threshold"
    raw_side_reason = _raw_side_skip_reason(entry, args)
    if raw_side_reason is not None:
        return raw_side_reason
    return None


def _raw_side_skip_reason(entry: dict[str, Any], args: argparse.Namespace) -> str | None:
    min_probability = args.entry_raw_side_min_probability
    min_margin = args.entry_raw_side_min_margin
    max_opposite_lead = args.entry_raw_side_max_opposite_lead
    if min_probability is None and min_margin is None and max_opposite_lead is None:
        return None
    side = str(entry.get("side") or "").upper()
    p_up = _float(entry.get("entry_p_up"))
    p_down = _float(entry.get("entry_p_down"))
    if side == "UP":
        p_side = p_up
        p_opposite = p_down
    elif side == "DOWN":
        p_side = p_down
        p_opposite = p_up
    else:
        return "entry_raw_side_missing"
    if p_side is None or p_opposite is None:
        return "entry_raw_side_missing"
    if min_probability is not None and p_side < min_probability:
        return "entry_raw_side_probability_below_replay_threshold"
    if min_margin is not None and p_side - p_opposite < min_margin:
        return "entry_raw_side_margin_below_replay_threshold"
    if max_opposite_lead is not None and p_opposite - p_side > max_opposite_lead:
        return "entry_raw_side_opposite_lead_above_replay_threshold"
    return None


def _group_rows_by_run(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("run_id") or "")].append(row)
    output: list[dict[str, Any]] = []
    for run_id in sorted(groups):
        items = groups[run_id]
        kept = [row for row in items if row["skip_reason"] is None]
        skipped = [row for row in items if row["skip_reason"] is not None]
        baseline_pnl = sum(float(row["pnl"]) for row in items)
        replay_pnl = sum(float(row["pnl"]) for row in kept)
        output.append(
            {
                "run_id": run_id,
                "matched_bets": len(items),
                "kept_bets": len(kept),
                "skipped_bets": len(skipped),
                "baseline_pnl": baseline_pnl,
                "replay_pnl": replay_pnl,
                "pnl_delta": replay_pnl - baseline_pnl,
                "skip_reason_counts": dict(
                    sorted(Counter(row["skip_reason"] for row in skipped).items())
                ),
            }
        )
    return output


def _markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# xgboost-v7 Entry Quality Gate Replay",
        "",
        "## Summary",
        "",
        f"- Matched bets: {summary['matched_bets']}",
        f"- Kept / skipped: {summary['kept_bets']} / {summary['skipped_bets']}",
        f"- Baseline PnL: {summary['baseline_pnl']:.6f}",
        f"- Replay PnL: {summary['replay_pnl']:.6f}",
        f"- PnL delta: {summary['pnl_delta']:.6f}",
        f"- Skip reasons: `{json.dumps(summary['skip_reason_counts'], sort_keys=True)}`",
        "",
        "## By Run",
        "",
        "|Run|Matched|Kept|Skipped|Baseline PnL|Replay PnL|Delta|Reasons|",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["by_run"]:
        lines.append(
            "|{run_id}|{matched_bets}|{kept_bets}|{skipped_bets}|{baseline_pnl:.6f}|{replay_pnl:.6f}|{pnl_delta:.6f}|`{reasons}`|".format(
                reasons=json.dumps(row["skip_reason_counts"], sort_keys=True),
                **row,
            )
        )
    lines.extend(
        [
            "",
            "## Skipped Entries",
            "",
            "|Run|Round|Side|Entry|Signal Px|p_up|p_down|Drift|Age s|PnL|Reason|",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["skipped_entries"]:
        lines.append(
            "|{run}|{round}|{side}|{entry}|{signal_px}|{p_up}|{p_down}|{drift}|{age}|{pnl:.6f}|{reason}|".format(
                run=row.get("run_id") or "",
                round=row.get("round_slug") or "",
                side=row.get("side") or "",
                entry=_fmt(row.get("entry_price")),
                signal_px=_fmt(row.get("entry_signal_price")),
                p_up=_fmt(row.get("entry_p_up")),
                p_down=_fmt(row.get("entry_p_down")),
                drift=_fmt(row.get("entry_price_drift_from_signal")),
                age=_fmt(row.get("entry_signal_age_seconds")),
                pnl=float(row.get("pnl") or 0.0),
                reason=row.get("skip_reason") or "",
            )
        )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    number = _float(value)
    if number is None:
        return ""
    return f"{number:.4f}"


def _millis(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        raw = float(value)
        return int(raw if raw > 10_000_000_000 else raw * 1000)
    text = str(value).strip()
    if not text:
        return 0
    try:
        raw = float(text)
    except ValueError:
        return 0
    return int(raw if raw > 10_000_000_000 else raw * 1000)


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
