#!/usr/bin/env python3
"""Replay value-calibrated v7 entry gates against actual paper fills.

This is a log-derived counterfactual. It filters actual `paper_entry_filled`
bets and removes the matching realized PnL when a hypothetical entry gate would
have blocked the fill. It does not synthesize replacement fills after a slot is
freed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class EntryRow:
    run_id: str
    source_path: str
    source_line: int
    event_id: str
    round_slug: str
    side: str
    entry_price: float
    entry_market_price: float | None
    entry_model_value: float | None
    entry_side_probability: float | None
    entry_signal_age_seconds: float | None
    entry_signal_ts_ms: int
    opened_at_ms: int
    exit_price: float | None
    exit_reason: str
    pnl: float

    @property
    def entry_fill_edge(self) -> float | None:
        if self.entry_model_value is None:
            return None
        return self.entry_model_value - self.entry_price


@dataclass(frozen=True, slots=True)
class GridConfig:
    model_value_haircut: float
    min_calibrated_edge: float
    min_model_value: float
    min_side_probability: float


@dataclass(frozen=True, slots=True)
class ReplayRow:
    entry: EntryRow
    calibrated_model_value: float | None
    calibrated_edge: float | None
    keep: bool
    skip_reasons: tuple[str, ...]


def main() -> int:
    args = _parse_args()
    entries = _load_entries([Path(path) for path in args.executor_jsonl])
    configs = list(_grid_configs(args))
    grid_results = [_evaluate_config(entries, config) for config in configs]
    recommended = _select_recommended_config(
        grid_results,
        min_kept_ratio=args.min_kept_ratio_for_recommendation,
    )
    report = {
        "inputs": {
            "executor_jsonl": args.executor_jsonl,
        },
        "config_grid": {
            "model_value_haircut": args.model_value_haircut_grid,
            "min_calibrated_edge": args.min_calibrated_edge_grid,
            "min_model_value": args.min_model_value_grid,
            "min_side_probability": args.min_side_probability_grid,
            "min_kept_ratio_for_recommendation": args.min_kept_ratio_for_recommendation,
        },
        "baseline": _baseline_summary(entries),
        "recommended": recommended,
        "top_configs": _top_configs(grid_results, limit=args.top_limit),
        "grid_results": grid_results,
        "recommended_rows": (
            _rows_for_config(entries, GridConfig(**recommended["config"]))
            if recommended is not None
            else []
        ),
        "entries": [asdict(entry) for entry in entries],
    }
    if args.output_json_path:
        path = Path(args.output_json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.report_path:
        path = Path(args.report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_markdown_report(report), encoding="utf-8")
    print(json.dumps({"baseline": report["baseline"], "recommended": recommended}, indent=2))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executor-jsonl", action="append", required=True)
    parser.add_argument(
        "--model-value-haircut-grid",
        default="0,0.05,0.10,0.15,0.20,0.25",
        help="Comma-separated absolute haircut values subtracted from model value.",
    )
    parser.add_argument(
        "--min-calibrated-edge-grid",
        default="0.04,0.08,0.12,0.16,0.20,0.25,0.30,0.35",
        help="Comma-separated minimum calibrated edge values.",
    )
    parser.add_argument(
        "--min-model-value-grid",
        default="0,0.70,0.75,0.80,0.85,0.90",
        help="Comma-separated minimum calibrated model value floors; 0 disables the floor.",
    )
    parser.add_argument(
        "--min-side-probability-grid",
        default="0,0.50,0.55,0.60,0.65,0.70,0.75,0.80",
        help="Comma-separated minimum base-head probability for the selected side; 0 disables it.",
    )
    parser.add_argument("--min-kept-ratio-for-recommendation", type=float, default=0.20)
    parser.add_argument("--top-limit", type=int, default=20)
    parser.add_argument("--output-json-path", default="")
    parser.add_argument("--report-path", default="")
    args = parser.parse_args()
    args.model_value_haircut_grid = _parse_float_grid(args.model_value_haircut_grid)
    args.min_calibrated_edge_grid = _parse_float_grid(args.min_calibrated_edge_grid)
    args.min_model_value_grid = _parse_float_grid(args.min_model_value_grid)
    args.min_side_probability_grid = _parse_float_grid(args.min_side_probability_grid)
    if not 0.0 <= args.min_kept_ratio_for_recommendation <= 1.0:
        raise ValueError("--min-kept-ratio-for-recommendation must be between 0 and 1")
    if args.top_limit <= 0:
        raise ValueError("--top-limit must be positive")
    return args


def _parse_float_grid(text: str) -> list[float]:
    values: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if not math.isfinite(value):
            raise ValueError(f"non-finite grid value: {item}")
        values.append(value)
    if not values:
        raise ValueError("grid must contain at least one numeric value")
    return sorted(set(values))


def _grid_configs(args: argparse.Namespace) -> list[GridConfig]:
    return [
        GridConfig(*items)
        for items in itertools.product(
            args.model_value_haircut_grid,
            args.min_calibrated_edge_grid,
            args.min_model_value_grid,
            args.min_side_probability_grid,
        )
    ]


def _load_entries(paths: list[Path]) -> list[EntryRow]:
    entries: dict[str, EntryRow] = {}
    exits: dict[str, dict[str, Any]] = {}
    for path in paths:
        run_id = _run_id(path)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                event = payload.get("event")
                position = payload.get("position") or {}
                event_id = str(position.get("event_id") or "")
                if not event_id:
                    continue
                if event == "paper_entry_filled":
                    entry = _entry_from_payload(
                        payload,
                        path=path,
                        run_id=run_id,
                        line_number=line_number,
                    )
                    if entry is not None:
                        entries[event_id] = entry
                elif event in {
                    "paper_exit_filled",
                    "paper_settlement_resolved",
                    "v7_settlement_position_exit_filled",
                }:
                    exits[event_id] = payload

    rows: list[EntryRow] = []
    for event_id, entry in entries.items():
        exit_payload = exits.get(event_id)
        if exit_payload is None:
            rows.append(entry)
            continue
        rows.append(_entry_with_exit(entry, exit_payload))
    return sorted(rows, key=lambda item: (item.opened_at_ms, item.event_id))


def _entry_from_payload(
    payload: dict[str, Any],
    *,
    path: Path,
    run_id: str,
    line_number: int,
) -> EntryRow | None:
    position = payload.get("position") or {}
    signal = payload.get("signal") or {}
    gate = payload.get("gate_evaluation") or {}
    event_id = str(position.get("event_id") or "")
    entry_price = _first_float(position, "entry_price", "fill_price")
    if not event_id or entry_price is None:
        return None
    side = str(position.get("side") or signal.get("outcome_side") or "").upper()
    side_probability = _side_probability(position, signal, side)
    return EntryRow(
        run_id=run_id,
        source_path=str(path),
        source_line=line_number,
        event_id=event_id,
        round_slug=str(position.get("round_slug") or signal.get("round_slug") or ""),
        side=side,
        entry_price=entry_price,
        entry_market_price=(
            _first_float(position, "entry_polymarket_price")
            or _first_float(signal, "polymarket_price", "market_implied_prob")
        ),
        entry_model_value=(
            _first_float(position, "entry_model_probability")
            or _first_float(signal, "model_probability", "token_expected_win_probability")
        ),
        entry_side_probability=side_probability,
        entry_signal_age_seconds=_first_float(gate, "signal_age_seconds"),
        entry_signal_ts_ms=_millis(position.get("entry_signal_ts") or signal.get("ts")),
        opened_at_ms=_millis(position.get("opened_at") or payload.get("ts")),
        exit_price=None,
        exit_reason="",
        pnl=0.0,
    )


def _entry_with_exit(entry: EntryRow, payload: dict[str, Any]) -> EntryRow:
    return EntryRow(
        run_id=entry.run_id,
        source_path=entry.source_path,
        source_line=entry.source_line,
        event_id=entry.event_id,
        round_slug=entry.round_slug,
        side=entry.side,
        entry_price=entry.entry_price,
        entry_market_price=entry.entry_market_price,
        entry_model_value=entry.entry_model_value,
        entry_side_probability=entry.entry_side_probability,
        entry_signal_age_seconds=entry.entry_signal_age_seconds,
        entry_signal_ts_ms=entry.entry_signal_ts_ms,
        opened_at_ms=entry.opened_at_ms,
        exit_price=_first_float(payload, "exit_price", "bid"),
        exit_reason=str(payload.get("reason") or (payload.get("position") or {}).get("last_lifecycle_reason") or ""),
        pnl=_first_float(payload, "realized_pnl", "realized_account_pnl") or 0.0,
    )


def _side_probability(
    position: dict[str, Any],
    signal: dict[str, Any],
    side: str,
) -> float | None:
    if side == "UP":
        return _first_float(position, "entry_p_up") or _first_float(signal, "p_up", "prob_up_15m")
    if side == "DOWN":
        return _first_float(position, "entry_p_down") or _first_float(signal, "p_down")
    return None


def _run_id(path: Path) -> str:
    match = re.search(r"phase4-(.+?)\.jsonl$", path.name)
    if match:
        return match.group(1)
    return path.stem


def _evaluate_config(entries: list[EntryRow], config: GridConfig) -> dict[str, Any]:
    replay_rows = [_evaluate_entry(entry, config) for entry in entries]
    kept = [row for row in replay_rows if row.keep]
    skipped = [row for row in replay_rows if not row.keep]
    baseline_pnl = sum(row.entry.pnl for row in replay_rows)
    replay_pnl = sum(row.entry.pnl for row in kept)
    return {
        "config": asdict(config),
        "summary": {
            "matched_bets": len(replay_rows),
            "kept_bets": len(kept),
            "skipped_bets": len(skipped),
            "kept_ratio": len(kept) / len(replay_rows) if replay_rows else 0.0,
            "baseline_pnl": baseline_pnl,
            "replay_pnl": replay_pnl,
            "pnl_delta": replay_pnl - baseline_pnl,
            "kept_wins": sum(1 for row in kept if row.entry.pnl > 0),
            "kept_losses": sum(1 for row in kept if row.entry.pnl < 0),
            "skipped_pnl": sum(row.entry.pnl for row in skipped),
            "skip_reason_counts": dict(_reason_counts(skipped)),
            "by_run": _by_run_summary(replay_rows),
        },
    }


def _evaluate_entry(entry: EntryRow, config: GridConfig) -> ReplayRow:
    reasons: list[str] = []
    calibrated_model_value = None
    calibrated_edge = None
    if entry.entry_model_value is None:
        reasons.append("missing_model_value")
    else:
        calibrated_model_value = entry.entry_model_value - config.model_value_haircut
        calibrated_edge = calibrated_model_value - entry.entry_price
        if config.min_model_value > 0.0 and calibrated_model_value < config.min_model_value:
            reasons.append("calibrated_model_value_below_floor")
        if calibrated_edge < config.min_calibrated_edge:
            reasons.append("calibrated_edge_below_floor")
    if (
        config.min_side_probability > 0.0
        and (
            entry.entry_side_probability is None
            or entry.entry_side_probability < config.min_side_probability
        )
    ):
        reasons.append("side_probability_below_floor")
    return ReplayRow(
        entry=entry,
        calibrated_model_value=calibrated_model_value,
        calibrated_edge=calibrated_edge,
        keep=not reasons,
        skip_reasons=tuple(reasons),
    )


def _baseline_summary(entries: list[EntryRow]) -> dict[str, Any]:
    by_run: dict[str, list[EntryRow]] = defaultdict(list)
    for entry in entries:
        by_run[entry.run_id].append(entry)
    return {
        "matched_bets": len(entries),
        "baseline_pnl": sum(entry.pnl for entry in entries),
        "wins": sum(1 for entry in entries if entry.pnl > 0),
        "losses": sum(1 for entry in entries if entry.pnl < 0),
        "side_counts": dict(sorted(Counter(entry.side for entry in entries).items())),
        "exit_reason_counts": dict(sorted(Counter(entry.exit_reason for entry in entries).items())),
        "by_run": [
            {
                "run_id": run_id,
                "matched_bets": len(items),
                "baseline_pnl": sum(item.pnl for item in items),
                "wins": sum(1 for item in items if item.pnl > 0),
                "losses": sum(1 for item in items if item.pnl < 0),
                "side_counts": dict(sorted(Counter(item.side for item in items).items())),
            }
            for run_id, items in sorted(by_run.items())
        ],
    }


def _select_recommended_config(
    grid_results: list[dict[str, Any]],
    *,
    min_kept_ratio: float,
) -> dict[str, Any] | None:
    if not grid_results:
        return None
    candidates = [
        result
        for result in grid_results
        if result["summary"]["kept_ratio"] >= min_kept_ratio and result["summary"]["kept_bets"] > 0
    ]
    if not candidates:
        candidates = [result for result in grid_results if result["summary"]["kept_bets"] > 0]
    if not candidates:
        return None
    best = max(
        candidates,
        key=lambda result: (
            result["summary"]["replay_pnl"],
            result["summary"]["pnl_delta"],
            result["summary"]["kept_bets"],
            -result["config"]["model_value_haircut"],
            -result["config"]["min_calibrated_edge"],
            -result["config"]["min_model_value"],
            -result["config"]["min_side_probability"],
        ),
    )
    return {
        "config": best["config"],
        "summary": best["summary"],
    }


def _top_configs(grid_results: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    non_empty = [result for result in grid_results if result["summary"]["kept_bets"] > 0]
    ordered = sorted(
        non_empty,
        key=lambda result: (
            result["summary"]["replay_pnl"],
            result["summary"]["pnl_delta"],
            result["summary"]["kept_bets"],
        ),
        reverse=True,
    )
    return ordered[:limit]


def _rows_for_config(entries: list[EntryRow], config: GridConfig) -> list[dict[str, Any]]:
    return [
        {
            "entry": asdict(row.entry),
            "calibrated_model_value": row.calibrated_model_value,
            "calibrated_edge": row.calibrated_edge,
            "keep": row.keep,
            "skip_reasons": list(row.skip_reasons),
        }
        for row in (_evaluate_entry(entry, config) for entry in entries)
    ]


def _by_run_summary(rows: list[ReplayRow]) -> list[dict[str, Any]]:
    grouped: dict[str, list[ReplayRow]] = defaultdict(list)
    for row in rows:
        grouped[row.entry.run_id].append(row)
    output: list[dict[str, Any]] = []
    for run_id, items in sorted(grouped.items()):
        kept = [row for row in items if row.keep]
        skipped = [row for row in items if not row.keep]
        baseline_pnl = sum(row.entry.pnl for row in items)
        replay_pnl = sum(row.entry.pnl for row in kept)
        output.append(
            {
                "run_id": run_id,
                "matched_bets": len(items),
                "kept_bets": len(kept),
                "skipped_bets": len(skipped),
                "baseline_pnl": baseline_pnl,
                "replay_pnl": replay_pnl,
                "pnl_delta": replay_pnl - baseline_pnl,
                "kept_wins": sum(1 for row in kept if row.entry.pnl > 0),
                "kept_losses": sum(1 for row in kept if row.entry.pnl < 0),
                "skipped_pnl": sum(row.entry.pnl for row in skipped),
                "skip_reason_counts": dict(_reason_counts(skipped)),
            }
        )
    return output


def _reason_counts(rows: list[ReplayRow]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        if row.skip_reasons:
            counts["+".join(row.skip_reasons)] += 1
    return counts


def _markdown_report(report: dict[str, Any]) -> str:
    baseline = report["baseline"]
    recommended = report["recommended"]
    lines = [
        "# xgboost-v7 Value-Calibrated Entry Replay",
        "",
        "This replay filters actual paper fills. Blocked entries contribute zero PnL and no replacement fills are synthesized.",
        "",
        "## Baseline",
        "",
        f"- Matched bets: {baseline['matched_bets']}",
        f"- Baseline PnL: {baseline['baseline_pnl']:.6f}",
        f"- Wins / losses: {baseline['wins']} / {baseline['losses']}",
        f"- Side counts: `{json.dumps(baseline['side_counts'], sort_keys=True)}`",
        f"- Exit reasons: `{json.dumps(baseline['exit_reason_counts'], sort_keys=True)}`",
        "",
        "### By Run",
        "",
        "|Run|Bets|Baseline PnL|Wins|Losses|Sides|",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in baseline["by_run"]:
        lines.append(
            "|{run_id}|{matched_bets}|{baseline_pnl:.6f}|{wins}|{losses}|`{sides}`|".format(
                sides=json.dumps(row["side_counts"], sort_keys=True),
                **row,
            )
        )
    lines.extend(["", "## Recommended Config", ""])
    if recommended is None:
        lines.append("No non-empty recommended config was found.")
    else:
        config = recommended["config"]
        summary = recommended["summary"]
        lines.extend(
            [
                f"- Model value haircut: {config['model_value_haircut']:.4f}",
                f"- Min calibrated edge: {config['min_calibrated_edge']:.4f}",
                f"- Min calibrated model value: {config['min_model_value']:.4f}",
                f"- Min selected-side base probability: {config['min_side_probability']:.4f}",
                f"- Kept / skipped: {summary['kept_bets']} / {summary['skipped_bets']}",
                f"- Replay PnL: {summary['replay_pnl']:.6f}",
                f"- Delta vs baseline: {summary['pnl_delta']:.6f}",
                f"- Kept wins / losses: {summary['kept_wins']} / {summary['kept_losses']}",
                f"- Skip reasons: `{json.dumps(summary['skip_reason_counts'], sort_keys=True)}`",
                "",
                "### Recommended By Run",
                "",
                "|Run|Kept|Skipped|Baseline PnL|Replay PnL|Delta|Kept W/L|Skipped PnL|Reasons|",
                "|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in summary["by_run"]:
            lines.append(
                "|{run_id}|{kept_bets}|{skipped_bets}|{baseline_pnl:.6f}|{replay_pnl:.6f}|{pnl_delta:.6f}|{kept_wins}/{kept_losses}|{skipped_pnl:.6f}|`{reasons}`|".format(
                    reasons=json.dumps(row["skip_reason_counts"], sort_keys=True),
                    **row,
                )
            )
    lines.extend(
        [
            "",
            "## Top Configs",
            "",
            "|Rank|Haircut|Min Edge|Min Model Value|Min Side P|Kept|Skipped|Replay PnL|Delta|Kept W/L|",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for index, result in enumerate(report["top_configs"], start=1):
        config = result["config"]
        summary = result["summary"]
        lines.append(
            "|{rank}|{haircut:.4f}|{edge:.4f}|{model:.4f}|{side:.4f}|{kept}|{skipped}|{pnl:.6f}|{delta:.6f}|{wins}/{losses}|".format(
                rank=index,
                haircut=config["model_value_haircut"],
                edge=config["min_calibrated_edge"],
                model=config["min_model_value"],
                side=config["min_side_probability"],
                kept=summary["kept_bets"],
                skipped=summary["skipped_bets"],
                pnl=summary["replay_pnl"],
                delta=summary["pnl_delta"],
                wins=summary["kept_wins"],
                losses=summary["kept_losses"],
            )
        )
    if recommended is not None:
        lines.extend(
            [
                "",
                "## Recommended Entry Decisions",
                "",
                "|Run|Round|Side|Entry|Model Value|Cal Edge|Side P|PnL|Exit|Decision|Reasons|",
                "|---|---|---:|---:|---:|---:|---:|---:|---|---|---|",
            ]
        )
        for row in report["recommended_rows"]:
            entry = row["entry"]
            lines.append(
                "|{run}|{round_slug}|{side}|{entry_price}|{model}|{edge}|{side_p}|{pnl:.6f}|{exit}|{decision}|`{reasons}`|".format(
                    run=entry.get("run_id") or "",
                    round_slug=entry.get("round_slug") or "",
                    side=entry.get("side") or "",
                    entry_price=_fmt(entry.get("entry_price")),
                    model=_fmt(entry.get("entry_model_value")),
                    edge=_fmt(row.get("calibrated_edge")),
                    side_p=_fmt(entry.get("entry_side_probability")),
                    pnl=float(entry.get("pnl") or 0.0),
                    exit=entry.get("exit_reason") or "",
                    decision="KEEP" if row["keep"] else "SKIP",
                    reasons=json.dumps(row["skip_reasons"], sort_keys=True),
                )
            )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    number = _float(value)
    if number is None:
        return ""
    return f"{number:.4f}"


def _first_float(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _float(payload.get(key))
        if value is not None:
            return value
    return None


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


if __name__ == "__main__":
    raise SystemExit(main())
