#!/usr/bin/env python3
"""Classify v7 divergence reduce/add churn and whether triggering moves look like noise.

Reads a phase4 executor JSONL (Run #4 take-profit paper) and, per position:
  - lists divergence-driven REDUCE / subsequent ADD cycles
  - labels each divergence reduce as likely_noise vs likely_signal using post-reduce recovery
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _parse_ts(value: Any) -> dt.datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        ms = int(value)
        if ms > 1_000_000_000_000:
            return dt.datetime.fromtimestamp(ms / 1000.0, tz=dt.timezone.utc)
        return dt.datetime.fromtimestamp(ms, tz=dt.timezone.utc)
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(slots=True)
class EvalPoint:
    ts: dt.datetime
    action: str
    reason: str
    hold_edge: float | None
    hold_bid: float | None
    current_price: float | None
    price_move_toward_model: float | None
    price_diverged: bool | None
    model_degraded: bool | None
    diverged: bool | None
    entry_price: float | None
    executed: bool = False


@dataclass(slots=True)
class PmAction:
    ts: dt.datetime
    kind: str  # REDUCE | ADD
    reason: str
    hold_bid: float | None
    hold_edge: float | None
    realized_pnl_delta: float | None
    cumulative_position_realized_pnl: float | None
    add_usdc: float | None = None
    convergence: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReduceRecovery:
    window_seconds: int
    price_reverted: bool
    diverged_cleared: bool
    min_hold_edge: float | None
    max_price_move_toward_model: float | None
    eval_count: int


@dataclass(slots=True)
class DivergenceReduceAnalysis:
    ts: str
    hold_bid: float | None
    hold_edge: float | None
    price_move_toward_model: float | None
    trigger_type: str
    edge_intact: bool
    recovery: list[ReduceRecovery]
    verdict: str
    add_within_180s: bool
    add_within_180s_ts: str | None


@dataclass(slots=True)
class PositionChurnReport:
    event_id: str
    round_slug: str
    side: str
    entry_price: float | None
    entry_ts: str | None
    reduce_count: int
    add_count: int
    divergence_reduce_count: int
    churn_cycles: int
    v7_intra_pnl_delta_sum: float
    pm_actions: list[dict[str, Any]]
    divergence_reduces: list[DivergenceReduceAnalysis]
    churn_score: str


def _trigger_type(conv: dict[str, Any]) -> str:
    price_div = bool(conv.get("price_diverged"))
    model_deg = bool(conv.get("model_degraded"))
    if price_div and model_deg:
        return "price_and_model"
    if price_div:
        return "price_only"
    if model_deg:
        return "model_decay"
    return "unknown"


def _recovery_in_window(
    evals: list[EvalPoint],
    *,
    start: dt.datetime,
    window_seconds: int,
    price_tolerance: float,
    weak_hold_edge: float,
) -> ReduceRecovery:
    end = start.timestamp() + window_seconds
    window = [
        ev
        for ev in evals
        if start.timestamp() < ev.ts.timestamp() <= end
    ]
    max_pmtm = None
    min_hold = None
    price_reverted = False
    diverged_cleared = False
    for ev in window:
        if ev.price_move_toward_model is not None:
            max_pmtm = (
                ev.price_move_toward_model
                if max_pmtm is None
                else max(max_pmtm, ev.price_move_toward_model)
            )
            if ev.price_move_toward_model >= -price_tolerance:
                price_reverted = True
        if ev.hold_edge is not None:
            min_hold = ev.hold_edge if min_hold is None else min(min_hold, ev.hold_edge)
        if ev.diverged is False:
            diverged_cleared = True
    return ReduceRecovery(
        window_seconds=window_seconds,
        price_reverted=price_reverted,
        diverged_cleared=diverged_cleared,
        min_hold_edge=min_hold,
        max_price_move_toward_model=max_pmtm,
        eval_count=len(window),
    )


def _verdict(
    *,
    trigger_type: str,
    edge_intact: bool,
    recovery_60: ReduceRecovery,
    recovery_120: ReduceRecovery,
    weak_hold_edge: float,
) -> str:
    if trigger_type == "model_decay":
        return "likely_signal"
    if not edge_intact:
        return "likely_signal"
    if recovery_60.price_reverted or recovery_60.diverged_cleared:
        if recovery_60.min_hold_edge is not None and recovery_60.min_hold_edge >= weak_hold_edge:
            return "likely_noise"
    if recovery_120.price_reverted or recovery_120.diverged_cleared:
        if recovery_120.min_hold_edge is not None and recovery_120.min_hold_edge >= weak_hold_edge:
            return "likely_noise"
    if recovery_120.max_price_move_toward_model is not None and recovery_120.max_price_move_toward_model < -0.05:
        return "likely_signal"
    return "ambiguous"


def _load_positions(path: Path) -> dict[str, dict[str, Any]]:
    positions: dict[str, dict[str, Any]] = {}
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        event = row.get("event")
        position = row.get("position") or {}
        event_id = str(position.get("event_id") or "")
        if not event_id:
            continue
        bucket = positions.setdefault(
            event_id,
            {
                "event_id": event_id,
                "round_slug": str(position.get("round_slug") or ""),
                "side": str(position.get("side") or "").upper(),
                "entry_price": _float(position.get("entry_price") or position.get("fill_price")),
                "entry_ts": None,
                "evals": [],
                "pm_actions": [],
            },
        )
        ts = _parse_ts(row.get("ts"))
        if event == "paper_entry_filled" and bucket["entry_ts"] is None:
            bucket["entry_ts"] = row.get("ts")
        evaluation = row.get("evaluation") or {}
        if event == "v7_settlement_position_management_evaluated" and ts is not None:
            conv = evaluation.get("convergence") or {}
            bucket["evals"].append(
                EvalPoint(
                    ts=ts,
                    action=str(evaluation.get("action") or ""),
                    reason=str(evaluation.get("reason") or ""),
                    hold_edge=_float(evaluation.get("hold_edge")),
                    hold_bid=_float(evaluation.get("hold_bid")),
                    current_price=_float(conv.get("current_price")),
                    price_move_toward_model=_float(conv.get("price_move_toward_model")),
                    price_diverged=conv.get("price_diverged"),
                    model_degraded=conv.get("model_degraded"),
                    diverged=conv.get("diverged"),
                    entry_price=_float(conv.get("entry_price")),
                    executed=False,
                )
            )
        if event == "paper_v7_settlement_position_reduced" and ts is not None:
            bucket["pm_actions"].append(
                PmAction(
                    ts=ts,
                    kind="REDUCE",
                    reason=str(evaluation.get("reason") or ""),
                    hold_bid=_float(row.get("hold_bid")),
                    hold_edge=_float(evaluation.get("hold_edge")),
                    realized_pnl_delta=_float(row.get("realized_pnl_delta")),
                    cumulative_position_realized_pnl=_float(row.get("cumulative_position_realized_pnl")),
                    convergence=dict(evaluation.get("convergence") or {}),
                )
            )
            if bucket["evals"]:
                bucket["evals"][-1].executed = True
        if event == "paper_v7_settlement_position_added" and ts is not None:
            bucket["pm_actions"].append(
                PmAction(
                    ts=ts,
                    kind="ADD",
                    reason=str(evaluation.get("reason") or ""),
                    hold_bid=_float(evaluation.get("hold_bid")),
                    hold_edge=_float(evaluation.get("hold_edge")),
                    realized_pnl_delta=None,
                    cumulative_position_realized_pnl=None,
                    add_usdc=_float(row.get("add_usdc")),
                    convergence=dict(evaluation.get("convergence") or {}),
                )
            )
    for bucket in positions.values():
        bucket["evals"].sort(key=lambda item: item.ts)
        bucket["pm_actions"].sort(key=lambda item: item.ts)
    return positions


def _analyze_position(
    bucket: dict[str, Any],
    *,
    price_tolerance: float,
    weak_hold_edge: float,
    edge_intact_threshold: float,
    churn_window_seconds: int,
) -> PositionChurnReport | None:
    pm_actions: list[PmAction] = bucket["pm_actions"]
    evals: list[EvalPoint] = bucket["evals"]
    if not pm_actions and not evals:
        return None
    reduce_count = sum(1 for action in pm_actions if action.kind == "REDUCE")
    add_count = sum(1 for action in pm_actions if action.kind == "ADD")
    div_reduces = [action for action in pm_actions if action.kind == "REDUCE" and action.reason == "residual_divergence_reduce"]
    churn_cycles = 0
    for idx, action in enumerate(pm_actions):
        if action.kind != "REDUCE" or action.reason != "residual_divergence_reduce":
            continue
        deadline = action.ts.timestamp() + churn_window_seconds
        for later in pm_actions[idx + 1 :]:
            if later.ts.timestamp() > deadline:
                break
            if later.kind == "ADD":
                churn_cycles += 1
                break
    v7_delta = sum(action.realized_pnl_delta or 0.0 for action in pm_actions if action.kind == "REDUCE")
    divergence_analyses: list[DivergenceReduceAnalysis] = []
    for action in div_reduces:
        conv = action.convergence
        trigger = _trigger_type(conv)
        edge_intact = (action.hold_edge or 0.0) >= edge_intact_threshold
        rec30 = _recovery_in_window(
            evals,
            start=action.ts,
            window_seconds=30,
            price_tolerance=price_tolerance,
            weak_hold_edge=weak_hold_edge,
        )
        rec60 = _recovery_in_window(
            evals,
            start=action.ts,
            window_seconds=60,
            price_tolerance=price_tolerance,
            weak_hold_edge=weak_hold_edge,
        )
        rec120 = _recovery_in_window(
            evals,
            start=action.ts,
            window_seconds=120,
            price_tolerance=price_tolerance,
            weak_hold_edge=weak_hold_edge,
        )
        verdict = _verdict(
            trigger_type=trigger,
            edge_intact=edge_intact,
            recovery_60=rec60,
            recovery_120=rec120,
            weak_hold_edge=weak_hold_edge,
        )
        add_follow = None
        add_ts = None
        deadline = action.ts.timestamp() + churn_window_seconds
        for later in pm_actions:
            if later.ts <= action.ts:
                continue
            if later.ts.timestamp() > deadline:
                break
            if later.kind == "ADD":
                add_follow = True
                add_ts = later.ts.isoformat()
                break
        divergence_analyses.append(
            DivergenceReduceAnalysis(
                ts=action.ts.isoformat(),
                hold_bid=action.hold_bid,
                hold_edge=action.hold_edge,
                price_move_toward_model=_float(conv.get("price_move_toward_model")),
                trigger_type=trigger,
                edge_intact=edge_intact,
                recovery=[rec30, rec60, rec120],
                verdict=verdict,
                add_within_180s=bool(add_follow),
                add_within_180s_ts=add_ts,
            )
        )
    if churn_cycles >= 2 or (div_reduces and add_count >= 2):
        churn_score = "high"
    elif churn_cycles >= 1 or div_reduces:
        churn_score = "medium"
    else:
        churn_score = "low"
    return PositionChurnReport(
        event_id=bucket["event_id"],
        round_slug=bucket["round_slug"],
        side=bucket["side"],
        entry_price=bucket["entry_price"],
        entry_ts=bucket["entry_ts"],
        reduce_count=reduce_count,
        add_count=add_count,
        divergence_reduce_count=len(div_reduces),
        churn_cycles=churn_cycles,
        v7_intra_pnl_delta_sum=round(v7_delta, 6),
        pm_actions=[
            {
                "ts": action.ts.isoformat(),
                "kind": action.kind,
                "reason": action.reason,
                "hold_bid": action.hold_bid,
                "hold_edge": action.hold_edge,
                "realized_pnl_delta": action.realized_pnl_delta,
                "add_usdc": action.add_usdc,
            }
            for action in pm_actions
        ],
        divergence_reduces=divergence_analyses,
        churn_score=churn_score,
    )


def _serialize_report(report: dict[str, Any]) -> dict[str, Any]:
    out_positions = []
    for pos in report["positions"]:
        item = asdict(pos)
        out_positions.append(item)
    report = dict(report)
    report["positions"] = out_positions
    return report


def _markdown_summary(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# v7 Divergence Churn / Noise Analysis",
        "",
        f"- Log: `{report['source_log']}`",
        f"- Positions with PM activity: **{summary['positions_with_pm']}** / {summary['total_entries']}",
        f"- Divergence reduces: **{summary['divergence_reduce_total']}** "
        f"(noise **{summary['verdict_counts'].get('likely_noise', 0)}**, "
        f"signal **{summary['verdict_counts'].get('likely_signal', 0)}**, "
        f"ambiguous **{summary['verdict_counts'].get('ambiguous', 0)}**)",
        f"- Reduce→add churn cycles (180s): **{summary['churn_cycles_total']}**",
        f"- Sum v7 reduce deltas (all positions): **{summary['v7_reduce_delta_sum_usdc']:.4f}** USDC",
        "",
        "## Per-position",
        "",
        "| Round | Side | Entry | Reduces | Adds | Div↓ | Churn | v7 Δ | Noise/Signal/Amb |",
        "|-------|------|-------|---------|------|------|-------|------|------------------|",
    ]
    for pos in sorted(report["positions"], key=lambda item: item["round_slug"]):
        slug = pos["round_slug"].split("-")[-1]
        verdicts = [dr["verdict"] for dr in pos["divergence_reduces"]]
        vc = {
            "likely_noise": verdicts.count("likely_noise"),
            "likely_signal": verdicts.count("likely_signal"),
            "ambiguous": verdicts.count("ambiguous"),
        }
        lines.append(
            f"| {slug} | {pos['side']} | {pos['entry_price']} | {pos['reduce_count']} | "
            f"{pos['add_count']} | {pos['divergence_reduce_count']} | {pos['churn_cycles']} | "
            f"{pos['v7_intra_pnl_delta_sum']:.3f} | {vc['likely_noise']}/{vc['likely_signal']}/{vc['ambiguous']} |"
        )
    lines.extend(["", "## High-churn detail", ""])
    for pos in report["positions"]:
        if pos["churn_score"] != "high":
            continue
        slug = pos["round_slug"]
        lines.append(f"### {slug} {pos['side']} @ {pos['entry_price']}")
        for dr in pos["divergence_reduces"]:
            rec60 = next(r for r in dr["recovery"] if r["window_seconds"] == 60)
            lines.append(
                f"- {dr['ts']}: bid={dr['hold_bid']}, hold_edge={dr['hold_edge']:.3f}, "
                f"pmtm={dr['price_move_toward_model']}, **{dr['verdict']}** "
                f"(60s revert={rec60['price_reverted']}, diverged_cleared={rec60['diverged_cleared']}, "
                f"add_follow={dr['add_within_180s']})"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def analyze_log(
    path: Path,
    *,
    price_tolerance: float = 0.02,
    weak_hold_edge: float = 0.02,
    edge_intact_threshold: float = 0.08,
    churn_window_seconds: int = 180,
) -> dict[str, Any]:
    positions = _load_positions(path)
    entries = sum(1 for _ in positions.values())
    reports: list[PositionChurnReport] = []
    for bucket in positions.values():
        report = _analyze_position(
            bucket,
            price_tolerance=price_tolerance,
            weak_hold_edge=weak_hold_edge,
            edge_intact_threshold=edge_intact_threshold,
            churn_window_seconds=churn_window_seconds,
        )
        if report is not None and (report.pm_actions or report.divergence_reduces):
            reports.append(report)
    verdict_counts: dict[str, int] = {}
    div_total = 0
    churn_total = 0
    v7_delta_sum = 0.0
    for pos in reports:
        churn_total += pos.churn_cycles
        v7_delta_sum += pos.v7_intra_pnl_delta_sum
        for dr in pos.divergence_reduces:
            div_total += 1
            verdict_counts[dr.verdict] = verdict_counts.get(dr.verdict, 0) + 1
    return {
        "source_log": str(path),
        "parameters": {
            "price_tolerance": price_tolerance,
            "weak_hold_edge": weak_hold_edge,
            "edge_intact_threshold": edge_intact_threshold,
            "churn_window_seconds": churn_window_seconds,
        },
        "summary": {
            "total_entries": entries,
            "positions_with_pm": len(reports),
            "divergence_reduce_total": div_total,
            "churn_cycles_total": churn_total,
            "verdict_counts": verdict_counts,
            "v7_reduce_delta_sum_usdc": round(v7_delta_sum, 6),
            "noise_rate": round(
                verdict_counts.get("likely_noise", 0) / div_total, 3
            )
            if div_total
            else None,
        },
        "positions": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executor-jsonl", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--report-md", default="")
    parser.add_argument("--price-tolerance", type=float, default=0.02)
    parser.add_argument("--weak-hold-edge", type=float, default=0.02)
    parser.add_argument("--edge-intact-threshold", type=float, default=0.08)
    parser.add_argument("--churn-window-seconds", type=int, default=180)
    args = parser.parse_args()
    report = analyze_log(
        Path(args.executor_jsonl),
        price_tolerance=args.price_tolerance,
        weak_hold_edge=args.weak_hold_edge,
        edge_intact_threshold=args.edge_intact_threshold,
        churn_window_seconds=args.churn_window_seconds,
    )
    serialized = _serialize_report(report)
    print(json.dumps(serialized["summary"], indent=2, sort_keys=True))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(serialized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.report_md:
        out = Path(args.report_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_markdown_summary(serialized), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
