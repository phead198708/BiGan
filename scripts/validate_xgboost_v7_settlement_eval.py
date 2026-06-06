#!/usr/bin/env python3
"""Small-scope validation for the xgboost-v7 settlement evaluation function.

This is intentionally not a v7 trainer. It reuses the current v6 settlement
probabilities as candidate model scores, then evaluates issue #99's proposed
metric of record: executable one-way settlement PnL over eligible rows.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REPLAY_SCRIPT = REPO_ROOT / "scripts" / "replay_v6_btc15m_execution_restricted.py"


@dataclass(frozen=True, slots=True)
class SettlementCandidate:
    split: str
    feature_ts: int
    round_slug: str
    side: str
    true_label: str
    p_side: float
    p_up: float
    p_down: float
    p_neutral: float
    entry_ask_price: float
    entry_worst_price: float
    expected_edge: float
    realized_pnl: float
    realized_edge: float
    seconds_to_expiry: float


@dataclass(frozen=True, slots=True)
class PolicySpec:
    name: str
    min_confidence: float
    min_expected_edge: float


def main() -> int:
    args = _parse_args()
    replay = _load_replay_module()
    model = replay.load_xgboost_v6_model(Path(args.model_json_path))
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    policies = _policy_grid(
        confidence_grid=_float_grid(args.confidence_grid),
        edge_grid=_float_grid(args.edge_grid),
    )
    v6_policy = PolicySpec(
        name="v6_current_gate",
        min_confidence=args.v6_min_confidence,
        min_expected_edge=args.v6_min_expected_edge,
    )
    candidates_by_split: dict[str, list[SettlementCandidate]] = {}
    candidate_counts: dict[str, dict[str, int]] = {}
    for split in splits:
        rows = replay._load_btc15_rows(Path(args.dataset_dir) / f"{split}.parquet")
        payloads = model.predict_payload_many(rows)
        skip_counts: dict[str, int] = {}
        candidates = _collect_candidates(
            replay,
            split=split,
            rows=rows,
            payloads=payloads,
            settlement_threshold=args.settlement_threshold,
            buy_slippage=args.buy_slippage,
            fee_bps=args.fee_bps,
            min_seconds_to_expiry=args.min_seconds_to_expiry,
            max_seconds_to_expiry=args.max_seconds_to_expiry,
            no_new_entry_before_expiry_seconds=args.no_new_entry_before_expiry_seconds,
            skip_counts=skip_counts,
        )
        candidates_by_split[split] = candidates
        candidate_counts[split] = {
            "candidate_count": len(candidates),
            "candidate_round_count": len({candidate.round_slug for candidate in candidates}),
            "skip_counts": dict(sorted(skip_counts.items())),
        }

    validation_split = args.validation_split
    if validation_split not in candidates_by_split:
        raise SystemExit(f"validation split {validation_split!r} was not loaded")
    test_split = args.test_split
    if test_split not in candidates_by_split:
        raise SystemExit(f"test split {test_split!r} was not loaded")

    validation_results = [
        _evaluate_policy(candidates_by_split[validation_split], policy)
        for policy in policies
    ]
    best_policy_result = _select_best_policy(
        validation_results,
        min_trades=args.min_validation_trades,
    )
    if best_policy_result is None:
        best_policy = PolicySpec("v7_selected_on_validation", 1.0, 1.0)
    else:
        best_policy = PolicySpec(
            name="v7_selected_on_validation",
            min_confidence=float(best_policy_result["policy"]["min_confidence"]),
            min_expected_edge=float(best_policy_result["policy"]["min_expected_edge"]),
        )

    split_results: dict[str, dict[str, Any]] = {}
    for split, candidates in candidates_by_split.items():
        split_results[split] = {
            "v6_current_gate": _evaluate_policy(candidates, v6_policy),
            "v7_selected_on_validation": _evaluate_policy(candidates, best_policy),
            "first_model_side_no_edge_floor": _evaluate_policy(
                candidates,
                PolicySpec(
                    name="first_model_side_no_edge_floor",
                    min_confidence=args.settlement_threshold,
                    min_expected_edge=-1.0,
                ),
            ),
        }

    v6_test = split_results[test_split]["v6_current_gate"]["summary"]
    v7_test = split_results[test_split]["v7_selected_on_validation"]["summary"]
    v6_val = split_results[validation_split]["v6_current_gate"]["summary"]
    v7_val = split_results[validation_split]["v7_selected_on_validation"]["summary"]
    status = _status(v6_val=v6_val, v7_val=v7_val, v6_test=v6_test, v7_test=v7_test)
    report = {
        "status": status,
        "purpose": "small_scope_v7_eval_function_reliability_check",
        "model_json_path": args.model_json_path,
        "dataset_dir": args.dataset_dir,
        "splits": splits,
        "validation_split": validation_split,
        "test_split": test_split,
        "metric_of_record": "executable_one_way_settlement_pnl",
        "candidate_counts": candidate_counts,
        "candidate_generation": {
            "settlement_threshold": args.settlement_threshold,
            "buy_slippage": args.buy_slippage,
            "fee_bps": args.fee_bps,
            "min_seconds_to_expiry": args.min_seconds_to_expiry,
            "max_seconds_to_expiry": args.max_seconds_to_expiry,
            "no_new_entry_before_expiry_seconds": args.no_new_entry_before_expiry_seconds,
            "round_first_per_policy": True,
        },
        "v6_policy": asdict(v6_policy),
        "best_v7_policy_selected_on_validation": asdict(best_policy),
        "top_validation_grid_results": validation_results[: args.report_limit],
        "split_results": split_results,
        "interpretation": _interpretation(status=status, v6_test=v6_test, v7_test=v7_test),
    }

    if args.output_json_path:
        path = Path(args.output_json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.report_path:
        path = Path(args.report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_markdown_report(report), encoding="utf-8")
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    default_model = (
        "data/model-runs/xgboost-v6-issue93-94-15m-only-volatility-20260602T135044Z/"
        "model-single-grid/model.json"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-json-path", default=default_model)
    parser.add_argument(
        "--dataset-dir",
        default=(
            "data/model-runs/xgboost-v6-issue93-94-15m-only-volatility-20260602T135044Z/dataset"
        ),
    )
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--validation-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--settlement-threshold", type=float, default=0.50)
    parser.add_argument("--confidence-grid", default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85")
    parser.add_argument("--edge-grid", default="0.00,0.02,0.04,0.06,0.08,0.082,0.10,0.12,0.15")
    parser.add_argument("--v6-min-confidence", type=float, default=0.80)
    parser.add_argument("--v6-min-expected-edge", type=float, default=0.082)
    parser.add_argument("--min-validation-trades", type=int, default=5)
    parser.add_argument("--buy-slippage", type=float, default=0.02)
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--min-seconds-to-expiry", type=float, default=300.0)
    parser.add_argument("--max-seconds-to-expiry", type=float, default=1200.0)
    parser.add_argument("--no-new-entry-before-expiry-seconds", type=float, default=300.0)
    parser.add_argument("--output-json-path", default="")
    parser.add_argument("--report-path", default="")
    parser.add_argument("--report-limit", type=int, default=12)
    return parser.parse_args()


def _load_replay_module():
    spec = importlib.util.spec_from_file_location("replay_v6_btc15m_execution_restricted", REPLAY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import replay script: {REPLAY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _float_grid(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _policy_grid(*, confidence_grid: list[float], edge_grid: list[float]) -> list[PolicySpec]:
    policies = [
        PolicySpec(
            name=f"conf_{confidence:.3f}_edge_{edge:.3f}",
            min_confidence=confidence,
            min_expected_edge=edge,
        )
        for confidence in confidence_grid
        for edge in edge_grid
    ]
    policies.sort(key=lambda item: (item.min_confidence, item.min_expected_edge))
    return policies


def _collect_candidates(
    replay: Any,
    *,
    split: str,
    rows: list[dict[str, Any]],
    payloads: list[dict[str, float | str]],
    settlement_threshold: float,
    buy_slippage: float,
    fee_bps: float,
    min_seconds_to_expiry: float,
    max_seconds_to_expiry: float,
    no_new_entry_before_expiry_seconds: float,
    skip_counts: dict[str, int],
) -> list[SettlementCandidate]:
    candidates: list[SettlementCandidate] = []
    for row, payload in zip(rows, payloads, strict=True):
        token_side = replay._outcome_side(row)
        market_payload = replay.market_v6_payload_from_token_payload(payload, token_side=token_side)
        p_up = float(market_payload["p_up"])
        p_down = float(market_payload["p_down"])
        p_neutral = float(market_payload["p_neutral"])
        side = _settlement_side(p_up=p_up, p_down=p_down, settlement_threshold=settlement_threshold)
        if side is None:
            _bump(skip_counts, "settlement_threshold_miss")
            continue
        seconds_to_expiry = replay._seconds_to_expiry(row)
        seconds_since_start = replay._seconds_since_round_start(row)
        if seconds_to_expiry is None or seconds_since_start is None:
            _bump(skip_counts, "missing_round_timing")
            continue
        if seconds_since_start < 0.0:
            _bump(skip_counts, "before_round_start")
            continue
        if seconds_to_expiry < no_new_entry_before_expiry_seconds:
            _bump(skip_counts, "no_new_entry_window")
            continue
        if seconds_to_expiry < min_seconds_to_expiry:
            _bump(skip_counts, "below_min_seconds_to_expiry")
            continue
        if seconds_to_expiry > max_seconds_to_expiry:
            _bump(skip_counts, "above_max_seconds_to_expiry")
            continue
        ask = replay._entry_ask(row, side)
        if ask is None:
            _bump(skip_counts, "missing_entry_quote")
            continue
        p_side = p_up if side == "UP" else p_down
        entry_worst = _entry_worst_price(float(ask), buy_slippage=buy_slippage, fee_bps=fee_bps)
        true_label = replay._settlement_label(row)
        win = 1.0 if true_label == side else 0.0
        realized_pnl = _one_way_settlement_pnl(true_label=true_label, side=side, entry_worst_price=entry_worst)
        candidates.append(
            SettlementCandidate(
                split=split,
                feature_ts=int(row.get("feature_ts") or 0),
                round_slug=replay._round_slug(row),
                side=side,
                true_label=true_label,
                p_side=p_side,
                p_up=p_up,
                p_down=p_down,
                p_neutral=p_neutral,
                entry_ask_price=float(ask),
                entry_worst_price=entry_worst,
                expected_edge=p_side - entry_worst,
                realized_pnl=realized_pnl,
                realized_edge=win - entry_worst,
                seconds_to_expiry=float(seconds_to_expiry),
            )
        )
    candidates.sort(key=lambda item: (item.feature_ts, item.round_slug, item.side))
    return candidates


def _settlement_side(*, p_up: float, p_down: float, settlement_threshold: float) -> str | None:
    if p_up >= p_down and p_up >= settlement_threshold:
        return "UP"
    if p_down > p_up and p_down >= settlement_threshold:
        return "DOWN"
    return None


def _entry_worst_price(ask: float, *, buy_slippage: float, fee_bps: float) -> float:
    fee = ask * fee_bps / 10_000.0
    return max(0.0, min(0.99, ask + buy_slippage + fee))


def _one_way_settlement_pnl(*, true_label: str, side: str, entry_worst_price: float) -> float:
    if true_label == side:
        return 1.0 - entry_worst_price
    return -entry_worst_price


def _evaluate_policy(candidates: list[SettlementCandidate], policy: PolicySpec) -> dict[str, Any]:
    selected: list[SettlementCandidate] = []
    seen_rounds: set[str] = set()
    reject_counts: dict[str, int] = {}
    for candidate in candidates:
        if candidate.round_slug in seen_rounds:
            _bump(reject_counts, "round_first_blocked")
            continue
        if candidate.p_side < policy.min_confidence:
            _bump(reject_counts, "confidence_below_threshold")
            continue
        if candidate.expected_edge < policy.min_expected_edge:
            _bump(reject_counts, "expected_edge_below_threshold")
            continue
        selected.append(candidate)
        seen_rounds.add(candidate.round_slug)
    return {
        "policy": asdict(policy),
        "summary": _summarize(selected, candidate_round_count=len({c.round_slug for c in candidates})),
        "reject_counts": dict(sorted(reject_counts.items())),
        "buckets": {
            "entry_worst_price": _bucket_summary(
                selected,
                [0.35, 0.50, 0.65, 0.80],
                value_fn=lambda item: item.entry_worst_price,
            ),
            "seconds_to_expiry": _bucket_summary(
                selected,
                [420.0, 600.0, 900.0],
                value_fn=lambda item: item.seconds_to_expiry,
            ),
            "expected_edge": _bucket_summary(
                selected,
                [0.0, 0.04, 0.082, 0.12],
                value_fn=lambda item: item.expected_edge,
            ),
        },
        "sample_trades": [asdict(candidate) for candidate in selected[:10]],
    }


def _summarize(
    selected: list[SettlementCandidate],
    *,
    candidate_round_count: int,
) -> dict[str, Any]:
    pnl = sum(candidate.realized_pnl for candidate in selected)
    wins = sum(1 for candidate in selected if candidate.realized_pnl > 0.0)
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for candidate in selected:
        cumulative += candidate.realized_pnl
        peak = max(peak, cumulative)
        max_drawdown = min(max_drawdown, cumulative - peak)
    return {
        "metric_of_record": "executable_one_way_settlement_pnl",
        "trade_count": len(selected),
        "candidate_round_count": candidate_round_count,
        "coverage": (len(selected) / candidate_round_count) if candidate_round_count else None,
        "pnl": pnl,
        "avg_pnl": (pnl / len(selected)) if selected else None,
        "hit_rate": (wins / len(selected)) if selected else None,
        "max_drawdown": max_drawdown,
        "mean_expected_edge": _mean([candidate.expected_edge for candidate in selected]),
        "mean_realized_edge": _mean([candidate.realized_edge for candidate in selected]),
        "mean_p_side": _mean([candidate.p_side for candidate in selected]),
        "mean_entry_worst_price": _mean([candidate.entry_worst_price for candidate in selected]),
        "mean_seconds_to_expiry": _mean([candidate.seconds_to_expiry for candidate in selected]),
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _bucket_summary(
    selected: list[SettlementCandidate],
    bounds: list[float],
    *,
    value_fn: Any,
) -> list[dict[str, Any]]:
    buckets: list[tuple[str, list[SettlementCandidate]]] = []
    lower: float | None = None
    for upper in bounds:
        if lower is None:
            label = f"<{upper:g}"
        else:
            label = f"{lower:g}-{upper:g}"
        buckets.append((label, []))
        lower = upper
    buckets.append((f">={bounds[-1]:g}" if bounds else "all", []))
    for candidate in selected:
        value = float(value_fn(candidate))
        placed = False
        previous: float | None = None
        for idx, upper in enumerate(bounds):
            if (previous is None and value < upper) or (
                previous is not None and previous <= value < upper
            ):
                buckets[idx][1].append(candidate)
                placed = True
                break
            previous = upper
        if not placed:
            buckets[-1][1].append(candidate)
    return [
        {
            "bucket": label,
            "trade_count": len(items),
            "pnl": sum(item.realized_pnl for item in items),
            "hit_rate": (
                sum(1 for item in items if item.realized_pnl > 0.0) / len(items)
                if items
                else None
            ),
        }
        for label, items in buckets
    ]


def _select_best_policy(
    validation_results: list[dict[str, Any]],
    *,
    min_trades: int,
) -> dict[str, Any] | None:
    eligible = [
        result
        for result in validation_results
        if int(result["summary"]["trade_count"]) >= min_trades
    ]
    if not eligible:
        eligible = [result for result in validation_results if int(result["summary"]["trade_count"]) > 0]
    if not eligible:
        return None
    eligible.sort(
        key=lambda result: (
            float(result["summary"]["pnl"]),
            float(result["summary"]["avg_pnl"] or -999.0),
            int(result["summary"]["trade_count"]),
        ),
        reverse=True,
    )
    validation_results.sort(
        key=lambda result: (
            float(result["summary"]["pnl"]),
            float(result["summary"]["avg_pnl"] or -999.0),
            int(result["summary"]["trade_count"]),
        ),
        reverse=True,
    )
    return eligible[0]


def _status(
    *,
    v6_val: dict[str, Any],
    v7_val: dict[str, Any],
    v6_test: dict[str, Any],
    v7_test: dict[str, Any],
) -> str:
    if int(v7_val["trade_count"]) == 0:
        return "NO_VALIDATION_TRADES"
    val_beats = float(v7_val["pnl"]) > float(v6_val["pnl"])
    test_beats = float(v7_test["pnl"]) > float(v6_test["pnl"])
    test_positive = float(v7_test["pnl"]) > 0.0 and int(v7_test["trade_count"]) > 0
    if val_beats and test_beats and test_positive:
        return "V7_EVAL_PROMISING_SMALL_SCOPE"
    if val_beats and not test_beats:
        return "V7_EVAL_OVERFITS_VALIDATION"
    return "V7_EVAL_NOT_BETTER_THAN_V6_BASELINE"


def _interpretation(
    *,
    status: str,
    v6_test: dict[str, Any],
    v7_test: dict[str, Any],
) -> str:
    if status == "V7_EVAL_PROMISING_SMALL_SCOPE":
        return (
            "The validation-selected v7 metric beats the fixed v6 gate on the held-out "
            "split and produces positive one-way settlement PnL. This is only a small "
            "scope check; the next step is to train a true v7 residual/probability head."
        )
    if status == "V7_EVAL_OVERFITS_VALIDATION":
        return (
            "The metric can find a better validation rule, but it does not generalize "
            "against the v6 fixed gate on the held-out split. Treat this as useful for "
            "diagnosis, not as evidence that v7 will outperform yet."
        )
    if int(v7_test["trade_count"]) == 0:
        return (
            "The selected v7 rule produces no held-out trades. The metric is too sparse "
            "with the current probabilities and price filters."
        )
    return (
        "The selected v7 rule is not yet better than the v6 fixed gate on held-out "
        f"one-way PnL (v7={float(v7_test['pnl']):.4f}, v6={float(v6_test['pnl']):.4f})."
    )


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    validation_split = report["validation_split"]
    test_split = report["test_split"]
    return {
        "status": report["status"],
        "best_v7_policy_selected_on_validation": report["best_v7_policy_selected_on_validation"],
        "validation": {
            "v6": report["split_results"][validation_split]["v6_current_gate"]["summary"],
            "v7": report["split_results"][validation_split]["v7_selected_on_validation"]["summary"],
        },
        "test": {
            "v6": report["split_results"][test_split]["v6_current_gate"]["summary"],
            "v7": report["split_results"][test_split]["v7_selected_on_validation"]["summary"],
        },
    }


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# xgboost-v7 Settlement Evaluation Function Smoke Test",
        "",
        f"Status: **{report['status']}**",
        "",
        f"Model used for candidate probabilities: `{report['model_json_path']}`",
        f"Dataset: `{report['dataset_dir']}`",
        "",
        "This test does not train v7. It uses v6 probabilities to check whether the "
        "proposed v7 metric of record can select a settlement policy that survives "
        "a held-out split.",
        "",
        "## Metric",
        "",
        "- Candidate edge: `p_side - entry_worst_price`.",
        "- Entry worst price: `ask + buy_slippage + fee`, capped at `0.99`.",
        "- PnL: win pays `1 - entry_worst_price`; loss pays `-entry_worst_price`.",
        "- Settlement buy-and-hold does not subtract the old volatility round-trip cost.",
        "- Each policy admits at most one settlement trade per round.",
        "",
        "## Selected Policy",
        "",
        f"- validation split: `{report['validation_split']}`",
        f"- test split: `{report['test_split']}`",
        f"- min confidence: `{report['best_v7_policy_selected_on_validation']['min_confidence']}`",
        f"- min expected edge: `{report['best_v7_policy_selected_on_validation']['min_expected_edge']}`",
        "",
        "## Split Results",
        "",
        "| Split | Policy | Trades | Coverage | PnL | Avg PnL | Hit rate | Mean edge | Mean price | Max DD |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, results in report["split_results"].items():
        for policy_name in ("v6_current_gate", "v7_selected_on_validation", "first_model_side_no_edge_floor"):
            summary = results[policy_name]["summary"]
            lines.append(
                f"| {split} | {policy_name} | {summary['trade_count']} | "
                f"{_fmt(summary['coverage'])} | {_fmt(summary['pnl'])} | "
                f"{_fmt(summary['avg_pnl'])} | {_fmt(summary['hit_rate'])} | "
                f"{_fmt(summary['mean_expected_edge'])} | "
                f"{_fmt(summary['mean_entry_worst_price'])} | "
                f"{_fmt(summary['max_drawdown'])} |"
            )
    lines.extend(
        [
            "",
            "## Candidate Counts",
            "",
            "| Split | Candidate rows | Candidate rounds | Skips |",
            "|---|---:|---:|---|",
        ]
    )
    for split, counts in report["candidate_counts"].items():
        lines.append(
            f"| {split} | {counts['candidate_count']} | {counts['candidate_round_count']} | "
            f"`{json.dumps(counts['skip_counts'], sort_keys=True)}` |"
        )
    lines.extend(
        [
            "",
            "## Top Validation Grid Results",
            "",
            "| Rank | Min confidence | Min edge | Trades | PnL | Avg PnL | Hit rate | Mean price |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for idx, result in enumerate(report["top_validation_grid_results"], start=1):
        summary = result["summary"]
        policy = result["policy"]
        lines.append(
            f"| {idx} | {policy['min_confidence']:.3f} | "
            f"{policy['min_expected_edge']:.3f} | {summary['trade_count']} | "
            f"{_fmt(summary['pnl'])} | {_fmt(summary['avg_pnl'])} | "
            f"{_fmt(summary['hit_rate'])} | {_fmt(summary['mean_entry_worst_price'])} |"
        )
    lines.extend(["", "## Interpretation", "", report["interpretation"], ""])
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def _bump(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


if __name__ == "__main__":
    raise SystemExit(main())
