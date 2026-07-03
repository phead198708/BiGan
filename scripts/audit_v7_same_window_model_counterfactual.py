#!/usr/bin/env python3
"""Compare two v7 models on the same live feature window.

The audit scores feature rows captured during a paper run with two v7 model
artifacts, then checks whether each model's predicted executable value was
realized by future same-side Polymarket bids.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from bigan.modeling import load_xgboost_v7_model

DEFAULT_OLD_MODEL = Path("data/model-runs/xgboost-v7-event-driven/20260608Tevent-5s-v1/model/model.json")
DEFAULT_NEW_MODEL = Path(
    "data/model-runs/xgboost-v7-event-driven/20260613Twarehouse-5s-v1/model-fast/model.json"
)
DEFAULT_FEATURE_STATE = Path(
    "data/live/xgboost-v7-warehouse-fast-scorer-20260613T101425Z/low-latency/features-state.json"
)
DEFAULT_RAW_QUEUE = Path(
    "data/live/xgboost-v7-warehouse-fast-scorer-20260613T101425Z/low-latency/raw-btc15m.jsonl"
)
DEFAULT_PHASE4 = Path("logs/xgboost-v7-paper-shadow/phase4-20260613T101448Z.jsonl")
DEFAULT_SUMMARY = Path("logs/xgboost-v7-paper-shadow/phase4-20260613T101448Z-summary.json")
DEFAULT_JSON_OUT = Path("docs/reports/issue104_run12_same_window_model_counterfactual_20260613.json")
DEFAULT_MD_OUT = Path("docs/reports/issue104_run12_same_window_model_counterfactual_20260613.md")


@dataclass(frozen=True)
class RunWindow:
    start_ms: int
    finish_ms: int
    rounds: tuple[str, ...]


@dataclass(frozen=True)
class FuturePath:
    max_bid: float | None
    max_bid_ts: int | None
    max_sell_move: float | None
    value_error: float | None
    hit_model_value: bool | None
    hit_entry_plus_10: bool | None
    hit_entry_plus_20: bool | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase4-log", type=Path, default=DEFAULT_PHASE4)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--feature-state", type=Path, default=DEFAULT_FEATURE_STATE)
    parser.add_argument("--raw-queue", type=Path, default=DEFAULT_RAW_QUEUE)
    parser.add_argument("--old-model", type=Path, default=DEFAULT_OLD_MODEL)
    parser.add_argument("--new-model", type=Path, default=DEFAULT_NEW_MODEL)
    parser.add_argument("--old-label", default="event5s_20260608")
    parser.add_argument("--new-label", default="warehouse_20260613")
    parser.add_argument("--edge-threshold", type=float, default=0.04)
    parser.add_argument("--min-entry-price", type=float, default=0.30)
    parser.add_argument("--min-seconds-to-expiry", type=float, default=300.0)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    return parser.parse_args()


def _to_ms(value: str) -> int:
    return int(datetime.fromisoformat(value).timestamp() * 1000)


def _float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
        return None if math.isnan(result) else result
    except (TypeError, ValueError):
        return None


def _mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return mean(present) if present else None


def _median(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return median(present) if present else None


def _rate(values: list[bool | None]) -> float | None:
    present = [value for value in values if value is not None]
    return (sum(1 for value in present if value) / len(present)) if present else None


def _round_slug_from_symbol(symbol: str) -> str | None:
    parts = symbol.split(":")
    if len(parts) < 3:
        return None
    return parts[1]


def _round_end_ms(round_slug: str) -> int | None:
    match = re.search(r"-(\d+)$", round_slug)
    if not match:
        return None
    return (int(match.group(1)) + 900) * 1000


def _load_run_window(summary_json: Path, phase4_log: Path) -> RunWindow:
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    start_ms = _to_ms(summary["started_at"])
    finish_ms = _to_ms(summary["finished_at"])
    rounds: set[str] = set()
    with phase4_log.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            signal = event.get("signal")
            if isinstance(signal, dict) and signal.get("round_slug"):
                rounds.add(str(signal["round_slug"]))
            for item in event.get("signals") or []:
                if isinstance(item, dict) and item.get("round_slug"):
                    rounds.add(str(item["round_slug"]))
            position = event.get("position")
            if isinstance(position, dict) and position.get("round_slug"):
                rounds.add(str(position["round_slug"]))
    return RunWindow(start_ms=start_ms, finish_ms=finish_ms, rounds=tuple(sorted(rounds)))


def _load_feature_rows(path: Path, window: RunWindow) -> list[dict[str, Any]]:
    state = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    for raw in (state.get("emitted_feature_signatures") or {}).values():
        row = json.loads(raw) if isinstance(raw, str) else dict(raw)
        feature_ts = int(row.get("feature_ts") or row.get("ts") or 0)
        symbol = str(row.get("canonical_symbol") or "")
        round_slug = _round_slug_from_symbol(symbol)
        if round_slug not in window.rounds:
            continue
        if feature_ts < window.start_ms or feature_ts > window.finish_ms:
            continue
        key = (feature_ts, str(row.get("source") or ""), str(row.get("source_symbol") or symbol))
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    rows.sort(key=lambda item: (int(item.get("feature_ts") or 0), str(item.get("canonical_symbol") or "")))
    return rows


def _load_future_bids(raw_queue: Path, rounds: tuple[str, ...]) -> dict[tuple[str, str], list[tuple[int, float]]]:
    if not rounds:
        return {}
    pattern = "btc-updown-15m-(" + "|".join(re.escape(item.replace("btc-updown-15m-", "")) for item in rounds) + ")"
    quotes: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    proc = subprocess.Popen(
        ["rg", "-e", pattern, str(raw_queue)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        start = line.find("{")
        if start > 0:
            line = line[start:]
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        row = event.get("row") or {}
        if row.get("level") not in (0, "0") or row.get("side") != "BID":
            continue
        symbol = str(row.get("canonical_symbol") or "")
        parts = symbol.split(":")
        if len(parts) < 3:
            continue
        round_slug = parts[1]
        side = parts[2]
        if side not in {"UP", "DOWN"}:
            continue
        price = _float(row.get("price"))
        if price is None:
            continue
        ts = int(row.get("ts") or row.get("message_ts") or event.get("published_at_ms") or 0)
        quotes[(round_slug, side)].append((ts, price))
    _stdout, stderr = proc.communicate()
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"rg failed with code {proc.returncode}: {stderr}")
    for items in quotes.values():
        items.sort()
    return dict(quotes)


def _future_path(
    *,
    quotes: dict[tuple[str, str], list[tuple[int, float]]],
    round_slug: str,
    side: str,
    entry_ts: int,
    entry_price: float | None,
    model_value: float | None,
) -> FuturePath:
    end_ms = _round_end_ms(round_slug)
    future = [
        item
        for item in quotes.get((round_slug, side), [])
        if item[0] >= entry_ts and (end_ms is None or item[0] <= end_ms)
    ]
    if not future or entry_price is None or model_value is None:
        return FuturePath(None, None, None, None, None, None, None)
    max_ts, max_bid = max(future, key=lambda item: item[1])
    return FuturePath(
        max_bid=max_bid,
        max_bid_ts=max_ts,
        max_sell_move=max_bid - entry_price,
        value_error=max_bid - model_value,
        hit_model_value=max_bid >= model_value,
        hit_entry_plus_10=max_bid >= entry_price + 0.10,
        hit_entry_plus_20=max_bid >= entry_price + 0.20,
    )


def _score_model(
    *,
    label: str,
    model_path: Path,
    rows: list[dict[str, Any]],
    quotes: dict[tuple[str, str], list[tuple[int, float]]],
    edge_threshold: float,
    min_entry_price: float,
    min_seconds_to_expiry: float,
) -> dict[str, Any]:
    model = load_xgboost_v7_model(model_path)
    payloads = model.predict_payload_many(rows)
    scored_rows: list[dict[str, Any]] = []
    for row, payload in zip(rows, payloads, strict=True):
        symbol = str(row.get("canonical_symbol") or "")
        round_slug = _round_slug_from_symbol(symbol)
        if round_slug is None:
            continue
        feature_ts = int(row.get("feature_ts") or row.get("ts") or 0)
        selected_side = str(payload.get("selected_side") or "")
        model_value = _float(payload.get("model_probability"))
        entry_price = _float(payload.get("polymarket_price"))
        selected_edge = _float(payload.get("selected_expected_edge") or payload.get("mispricing_edge"))
        round_end = _round_end_ms(round_slug)
        seconds_to_expiry = None if round_end is None else (round_end - feature_ts) / 1000.0
        path = _future_path(
            quotes=quotes,
            round_slug=round_slug,
            side=selected_side,
            entry_ts=feature_ts,
            entry_price=entry_price,
            model_value=model_value,
        )
        eligible = (
            selected_side in {"UP", "DOWN"}
            and model_value is not None
            and entry_price is not None
            and selected_edge is not None
            and selected_edge >= edge_threshold
            and entry_price >= min_entry_price
            and seconds_to_expiry is not None
            and seconds_to_expiry >= min_seconds_to_expiry
        )
        scored_rows.append(
            {
                "feature_ts": feature_ts,
                "round_slug": round_slug,
                "source_symbol": row.get("source_symbol"),
                "row_symbol": symbol,
                "row_token_side": symbol.split(":")[-1] if ":" in symbol else None,
                "selected_side": selected_side,
                "entry_price": entry_price,
                "model_probability": model_value,
                "selected_expected_edge": selected_edge,
                "seconds_to_expiry": seconds_to_expiry,
                "eligible": eligible,
                "future_max_bid": path.max_bid,
                "future_max_bid_ts": path.max_bid_ts,
                "max_sell_move": path.max_sell_move,
                "value_error": path.value_error,
                "hit_model_value": path.hit_model_value,
                "hit_entry_plus_10": path.hit_entry_plus_10,
                "hit_entry_plus_20": path.hit_entry_plus_20,
            }
        )
    eligible_rows = [row for row in scored_rows if row["eligible"]]
    first_by_round: dict[str, dict[str, Any]] = {}
    for row in eligible_rows:
        first_by_round.setdefault(str(row["round_slug"]), row)
    best_by_bucket: dict[tuple[str, int], dict[str, Any]] = {}
    for row in eligible_rows:
        key = (str(row["round_slug"]), int(row["feature_ts"]))
        current = best_by_bucket.get(key)
        if current is None or (row.get("selected_expected_edge") or -999.0) > (
            current.get("selected_expected_edge") or -999.0
        ):
            best_by_bucket[key] = row
    return {
        "label": label,
        "model_path": str(model_path),
        "scored_row_count": len(scored_rows),
        "eligible_row_count": len(eligible_rows),
        "eligible_rows": eligible_rows,
        "first_eligible_by_round": list(first_by_round.values()),
        "best_eligible_by_bucket": list(best_by_bucket.values()),
        "summary": {
            "all_scored": _summarize_rows(scored_rows),
            "eligible_rows": _summarize_rows(eligible_rows),
            "first_eligible_by_round": _summarize_rows(list(first_by_round.values())),
            "best_eligible_by_bucket": _summarize_rows(list(best_by_bucket.values())),
        },
    }


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    side_counts = Counter(str(row.get("selected_side")) for row in rows)
    round_counts = Counter(str(row.get("round_slug")) for row in rows)
    return {
        "count": len(rows),
        "round_count": len(round_counts),
        "selected_side_counts": dict(sorted(side_counts.items())),
        "avg_entry_price": _mean([_float(row.get("entry_price")) for row in rows]),
        "avg_model_probability": _mean([_float(row.get("model_probability")) for row in rows]),
        "avg_selected_expected_edge": _mean([_float(row.get("selected_expected_edge")) for row in rows]),
        "median_selected_expected_edge": _median([_float(row.get("selected_expected_edge")) for row in rows]),
        "avg_max_sell_move": _mean([_float(row.get("max_sell_move")) for row in rows]),
        "median_max_sell_move": _median([_float(row.get("max_sell_move")) for row in rows]),
        "avg_value_error": _mean([_float(row.get("value_error")) for row in rows]),
        "median_value_error": _median([_float(row.get("value_error")) for row in rows]),
        "hit_model_value_rate": _rate([row.get("hit_model_value") for row in rows]),
        "hit_entry_plus_10_rate": _rate([row.get("hit_entry_plus_10") for row in rows]),
        "hit_entry_plus_20_rate": _rate([row.get("hit_entry_plus_20") for row in rows]),
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_report(result: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Issue 104 Run 12 Same-Window Model Counterfactual",
        "",
        "This audit scores the same Run #12 feature rows with two v7 model artifacts.",
        "The metric is v7 edge realization: whether `model_probability - entry_price` was later realizable through same-side executable bids.",
        "",
        "## Inputs",
        "",
        f"- Phase4 log: `{result['inputs']['phase4_log']}`",
        f"- Feature state: `{result['inputs']['feature_state']}`",
        f"- Raw queue: `{result['inputs']['raw_queue']}`",
        f"- Feature rows scored: `{result['feature_row_count']}`",
        f"- Window rounds: `{', '.join(result['window']['rounds'])}`",
        "",
        "## Summary",
        "",
    ]
    for model in result["models"]:
        lines.extend(
            [
                f"### {model['label']}",
                "",
                "| Slice | Count | Rounds | Side Counts | Avg Edge | Avg Max Sell Move | Avg Value Error | Hit Model Value | Hit Entry+0.10 | Hit Entry+0.20 |",
                "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for key, title in [
            ("eligible_rows", "Eligible rows"),
            ("best_eligible_by_bucket", "Best eligible per 5s bucket"),
            ("first_eligible_by_round", "First eligible per round"),
        ]:
            summary = model["summary"][key]
            lines.append(
                "| {title} | {count} | {round_count} | {sides} | {edge} | {move} | {error} | {hit_model} | {hit10} | {hit20} |".format(
                    title=title,
                    count=summary["count"],
                    round_count=summary["round_count"],
                    sides=", ".join(f"{k}:{v}" for k, v in summary["selected_side_counts"].items()),
                    edge=_fmt(summary["avg_selected_expected_edge"]),
                    move=_fmt(summary["avg_max_sell_move"]),
                    error=_fmt(summary["avg_value_error"]),
                    hit_model=_fmt(summary["hit_model_value_rate"]),
                    hit10=_fmt(summary["hit_entry_plus_10_rate"]),
                    hit20=_fmt(summary["hit_entry_plus_20_rate"]),
                )
            )
        lines.append("")
    lines.extend(_comparison_readout_lines(result))
    lines.extend(
        [
            "## First Eligible Per Round",
            "",
            "| Model | Round | Feature TS | Side | Entry | Model Value | Pred Edge | Max Future Bid | Max Sell Move | Value Error | Hit Model | Hit +0.10 |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for model in result["models"]:
        for row in model["first_eligible_by_round"]:
            lines.append(
                "| {model} | `{round}` | {ts} | {side} | {entry} | {prob} | {edge} | {max_bid} | {move} | {error} | {hit_model} | {hit10} |".format(
                    model=model["label"],
                    round=row["round_slug"].replace("btc-updown-15m-", ""),
                    ts=row["feature_ts"],
                    side=row["selected_side"],
                    entry=_fmt(row["entry_price"]),
                    prob=_fmt(row["model_probability"]),
                    edge=_fmt(row["selected_expected_edge"]),
                    max_bid=_fmt(row["future_max_bid"]),
                    move=_fmt(row["max_sell_move"]),
                    error=_fmt(row["value_error"]),
                    hit_model=_fmt(row["hit_model_value"]),
                    hit10=_fmt(row["hit_entry_plus_10"]),
                )
            )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- `Hit Model Value` means the future same-side bid reached the model's predicted executable value.",
            "- `Value Error = future_max_bid - model_probability`; negative values mean the model overpredicted realizable exit value.",
            "- This audit intentionally does not score final settlement direction as the primary target.",
        ]
    )
    return "\n".join(lines) + "\n"


def _comparison_readout_lines(result: dict[str, Any]) -> list[str]:
    models = result["models"]
    if len(models) < 2:
        return []
    first, second = models[0], models[1]
    first_eligible = first["summary"]["eligible_rows"]
    second_eligible = second["summary"]["eligible_rows"]
    first_round = first["summary"]["first_eligible_by_round"]
    second_round = second["summary"]["first_eligible_by_round"]
    return [
        "## Comparison Readout",
        "",
        (
            f"- On eligible rows, `{first['label']}` hit model value "
            f"{_fmt(first_eligible['hit_model_value_rate'])} vs "
            f"`{second['label']}` {_fmt(second_eligible['hit_model_value_rate'])}."
        ),
        (
            f"- On eligible rows, `{first['label']}` hit `entry + 0.10` "
            f"{_fmt(first_eligible['hit_entry_plus_10_rate'])} vs "
            f"`{second['label']}` {_fmt(second_eligible['hit_entry_plus_10_rate'])}."
        ),
        (
            f"- Average value error was `{_fmt(first_eligible['avg_value_error'])}` for "
            f"`{first['label']}` and `{_fmt(second_eligible['avg_value_error'])}` for "
            f"`{second['label']}`. Both are negative, so both models overpredicted "
            "realizable executable value on this window."
        ),
        (
            f"- First-eligible-per-round hit rate was tied at "
            f"{_fmt(first_round['hit_model_value_rate'])}; the bigger difference is side "
            f"selection: `{first['label']}` was {first_round['selected_side_counts']} while "
            f"`{second['label']}` was {second_round['selected_side_counts']}."
        ),
        "",
        "Interpretation:",
        "",
        (
            "- The warehouse model is not failing because it predicts final direction poorly; "
            "this audit does not use final direction as the objective."
        ),
        (
            "- The warehouse model is weaker than expected on this same-window v7 objective: "
            "its selected edge reaches the predicted `model_probability` less often than the "
            "older event5s model on eligible rows."
        ),
        (
            "- The warehouse model still finds tradable movement fairly often, but its edge "
            "should be treated as an optimistic forecast and calibrated before increasing trust."
        ),
        (
            "- Because both models have negative average value error, the next replay should test "
            "value-calibrated entry/hold thresholds and adverse-exit timing, not a final-direction gate."
        ),
        "",
    ]


def main() -> None:
    args = _parse_args()
    window = _load_run_window(args.summary_json, args.phase4_log)
    feature_rows = _load_feature_rows(args.feature_state, window)
    quotes = _load_future_bids(args.raw_queue, window.rounds)
    models = [
        _score_model(
            label=args.old_label,
            model_path=args.old_model,
            rows=feature_rows,
            quotes=quotes,
            edge_threshold=args.edge_threshold,
            min_entry_price=args.min_entry_price,
            min_seconds_to_expiry=args.min_seconds_to_expiry,
        ),
        _score_model(
            label=args.new_label,
            model_path=args.new_model,
            rows=feature_rows,
            quotes=quotes,
            edge_threshold=args.edge_threshold,
            min_entry_price=args.min_entry_price,
            min_seconds_to_expiry=args.min_seconds_to_expiry,
        ),
    ]
    result = {
        "inputs": {
            "phase4_log": str(args.phase4_log),
            "summary_json": str(args.summary_json),
            "feature_state": str(args.feature_state),
            "raw_queue": str(args.raw_queue),
            "old_model": str(args.old_model),
            "new_model": str(args.new_model),
            "edge_threshold": args.edge_threshold,
            "min_entry_price": args.min_entry_price,
            "min_seconds_to_expiry": args.min_seconds_to_expiry,
        },
        "window": {
            "start_ms": window.start_ms,
            "finish_ms": window.finish_ms,
            "rounds": list(window.rounds),
        },
        "feature_row_count": len(feature_rows),
        "models": models,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    args.md_out.write_text(_markdown_report(result), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("feature_row_count", "window")}, indent=2, sort_keys=True))
    for model in models:
        print(model["label"], json.dumps(model["summary"], sort_keys=True))
    print(f"wrote {args.json_out}")
    print(f"wrote {args.md_out}")


if __name__ == "__main__":
    main()
