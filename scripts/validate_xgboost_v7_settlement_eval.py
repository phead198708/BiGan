#!/usr/bin/env python3
"""Execution-restricted PnL comparison for v6/v7 settlement policies.

The primary metric is executable one-way settlement PnL after the same entry
window, slippage, one-trade-per-round, and price-edge constraints the executor
can enforce. Settlement accuracy and hit rate are diagnostics only.
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
    residual_p_side: float | None
    residual_expected_edge: float | None
    market_implied_prob_side: float
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
    signal_source: str = "probability"
    min_residual_expected_edge: float | None = None


def main() -> int:
    args = _parse_args()
    replay = _load_replay_module()
    model, model_artifact_kind = _load_model(Path(args.model_json_path), replay)
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    selection_splits = [split.strip() for split in args.selection_splits.split(",") if split.strip()]
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
    selection_splits = [split for split in selection_splits if split in candidates_by_split]
    if not selection_splits:
        selection_splits = [validation_split]

    validation_results = [
        _evaluate_policy(candidates_by_split[validation_split], policy)
        for policy in policies
    ]
    validation_results.sort(
        key=lambda result: (
            float(result["summary"]["pnl"]),
            float(result["summary"]["avg_pnl"] or -999.0),
            int(result["summary"]["trade_count"]),
        ),
        reverse=True,
    )
    best_policy_result = _select_best_policy_by_pnl_stability(
        policies,
        candidates_by_split,
        selection_splits=selection_splits,
        min_trades=args.min_validation_trades,
        min_avg_pnl=args.min_selection_avg_pnl,
    )
    if best_policy_result is None:
        best_policy = PolicySpec("v7_selected_by_pnl_stability", 1.0, 1.0)
    else:
        best_policy = PolicySpec(
            name="v7_selected_by_pnl_stability",
            min_confidence=float(best_policy_result["policy"]["min_confidence"]),
            min_expected_edge=float(best_policy_result["policy"]["min_expected_edge"]),
            signal_source=str(best_policy_result["policy"].get("signal_source") or "probability"),
        )

    split_results: dict[str, dict[str, Any]] = {}
    for split, candidates in candidates_by_split.items():
        split_results[split] = {
            "v6_current_gate": _evaluate_policy(candidates, v6_policy),
            "v7_selected_by_pnl_stability": _evaluate_policy(candidates, best_policy),
            "v7_residual_edge_gate": _evaluate_policy(
                candidates,
                PolicySpec(
                    name="v7_residual_edge_gate",
                    min_confidence=args.residual_min_confidence,
                    min_expected_edge=args.residual_min_expected_edge,
                    signal_source="residual",
                ),
            ),
            "v7_hybrid_edge_gate": _evaluate_policy(
                candidates,
                PolicySpec(
                    name="v7_hybrid_edge_gate",
                    min_confidence=best_policy.min_confidence,
                    min_expected_edge=best_policy.min_expected_edge,
                    signal_source="hybrid",
                    min_residual_expected_edge=args.hybrid_min_residual_expected_edge,
                ),
            ),
            "market_favorite_baseline": _evaluate_policy(
                candidates,
                PolicySpec(
                    name="market_favorite_baseline",
                    min_confidence=0.0,
                    min_expected_edge=-1.0,
                    signal_source="market",
                ),
            ),
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
    v7_test = split_results[test_split]["v7_selected_by_pnl_stability"]["summary"]
    v6_val = split_results[validation_split]["v6_current_gate"]["summary"]
    v7_val = split_results[validation_split]["v7_selected_by_pnl_stability"]["summary"]
    status = _status(v6_val=v6_val, v7_val=v7_val, v6_test=v6_test, v7_test=v7_test)
    report = {
        "status": status,
        "purpose": "execution_restricted_v7_pnl_policy_comparison",
        "model_json_path": args.model_json_path,
        "model_artifact_kind": model_artifact_kind,
        "dataset_dir": args.dataset_dir,
        "splits": splits,
        "selection_splits": selection_splits,
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
        "best_v7_policy_selected_by_pnl_stability": {
            **asdict(best_policy),
            "selection": None if best_policy_result is None else best_policy_result["selection"],
        },
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
    parser.add_argument("--selection-splits", default="train,val")
    parser.add_argument("--validation-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--settlement-threshold", type=float, default=0.50)
    parser.add_argument("--confidence-grid", default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85")
    parser.add_argument("--edge-grid", default="0.00,0.02,0.04,0.06,0.08,0.082,0.10,0.12,0.15")
    parser.add_argument("--v6-min-confidence", type=float, default=0.80)
    parser.add_argument("--v6-min-expected-edge", type=float, default=0.082)
    parser.add_argument("--residual-min-confidence", type=float, default=0.75)
    parser.add_argument("--residual-min-expected-edge", type=float, default=0.04)
    parser.add_argument("--hybrid-min-residual-expected-edge", type=float, default=0.0)
    parser.add_argument("--min-validation-trades", type=int, default=5)
    parser.add_argument("--min-selection-avg-pnl", type=float, default=0.08)
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


def _load_model(model_path: Path, replay: Any) -> tuple[Any, str]:
    artifact = json.loads(model_path.read_text(encoding="utf-8"))
    schema = str(artifact.get("schema_version") or "")
    if schema.startswith("xgboost_v7"):
        from bigan.modeling import load_xgboost_v7_model

        return load_xgboost_v7_model(model_path), "xgboost-v7"
    if schema.startswith("xgboost_v6"):
        return replay.load_xgboost_v6_model(model_path), "xgboost-v6"
    raise ValueError(f"unsupported artifact schema for PnL eval: {schema!r}")


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
    payloads: list[dict[str, Any]],
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
        market_payload = _market_payload(payload, replay=replay, token_side=token_side)
        p_up = float(market_payload["p_up"])
        p_down = float(market_payload["p_down"])
        p_neutral = float(market_payload["p_neutral"])
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
        for side in ("UP", "DOWN"):
            ask = replay._entry_ask(row, side)
            if ask is None:
                _bump(skip_counts, f"missing_entry_quote_{side.lower()}")
                continue
            p_side = p_up if side == "UP" else p_down
            entry_worst = _entry_worst_price(float(ask), buy_slippage=buy_slippage, fee_bps=fee_bps)
            true_label = replay._settlement_label(row)
            win = 1.0 if true_label == side else 0.0
            realized_pnl = _one_way_settlement_pnl(true_label=true_label, side=side, entry_worst_price=entry_worst)
            residual_p_side = _optional_float(
                market_payload.get("p_up_residual_adjusted")
                if side == "UP"
                else market_payload.get("p_down_residual_adjusted")
            )
            residual_expected_edge = _optional_float(
                market_payload.get("residual_expected_edge_up")
                if side == "UP"
                else market_payload.get("residual_expected_edge_down")
            )
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
                    residual_p_side=residual_p_side,
                    residual_expected_edge=residual_expected_edge,
                    market_implied_prob_side=float(ask),
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


def _market_payload(payload: dict[str, Any], *, replay: Any, token_side: str) -> dict[str, Any]:
    if "expected_edge_up" in payload or "settlement_residual" in payload:
        return payload
    return replay.market_v6_payload_from_token_payload(payload, token_side=token_side)


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
    for _, group in _candidate_signal_groups(candidates):
        round_slug = group[0].round_slug
        if round_slug in seen_rounds:
            _bump(reject_counts, "round_first_blocked")
            continue
        passing: list[tuple[tuple[float, ...], SettlementCandidate]] = []
        for candidate in group:
            passed, reason = _candidate_passes_policy(candidate, policy)
            if not passed:
                _bump(reject_counts, reason)
                continue
            passing.append((_policy_candidate_score(candidate, policy), candidate))
        if not passing:
            continue
        passing.sort(key=lambda item: item[0], reverse=True)
        selected.append(passing[0][1])
        seen_rounds.add(round_slug)
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


def _candidate_signal_groups(
    candidates: list[SettlementCandidate],
) -> list[tuple[tuple[int, str], list[SettlementCandidate]]]:
    groups: dict[tuple[int, str], list[SettlementCandidate]] = {}
    for candidate in candidates:
        key = (candidate.feature_ts, candidate.round_slug)
        groups.setdefault(key, []).append(candidate)
    return sorted(groups.items(), key=lambda item: item[0])


def _candidate_passes_policy(candidate: SettlementCandidate, policy: PolicySpec) -> tuple[bool, str]:
    confidence, edge = _policy_confidence_and_edge(candidate, policy)
    if confidence is None or edge is None:
        return False, f"{policy.signal_source}_signal_missing"
    if confidence < policy.min_confidence:
        return False, "confidence_below_threshold"
    if edge < policy.min_expected_edge:
        return False, "expected_edge_below_threshold"
    if policy.signal_source == "hybrid":
        residual_floor = (
            policy.min_expected_edge
            if policy.min_residual_expected_edge is None
            else policy.min_residual_expected_edge
        )
        if candidate.residual_expected_edge is None:
            return False, "residual_signal_missing"
        if candidate.residual_expected_edge < residual_floor:
            return False, "residual_edge_below_threshold"
    return True, ""


def _policy_confidence_and_edge(
    candidate: SettlementCandidate,
    policy: PolicySpec,
) -> tuple[float | None, float | None]:
    if policy.signal_source in {"probability", "hybrid"}:
        return candidate.p_side, candidate.expected_edge
    if policy.signal_source == "residual":
        return candidate.residual_p_side, candidate.residual_expected_edge
    if policy.signal_source == "market":
        return candidate.market_implied_prob_side, candidate.market_implied_prob_side - candidate.entry_worst_price
    raise ValueError(f"unknown policy signal_source: {policy.signal_source!r}")


def _policy_candidate_score(candidate: SettlementCandidate, policy: PolicySpec) -> tuple[float, ...]:
    confidence, edge = _policy_confidence_and_edge(candidate, policy)
    residual_edge = -999.0 if candidate.residual_expected_edge is None else candidate.residual_expected_edge
    return (
        -999.0 if edge is None else edge,
        -999.0 if confidence is None else confidence,
        residual_edge,
        -candidate.entry_worst_price,
    )


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


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _select_best_policy_by_pnl_stability(
    policies: list[PolicySpec],
    candidates_by_split: dict[str, list[SettlementCandidate]],
    *,
    selection_splits: list[str],
    min_trades: int,
    min_avg_pnl: float,
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for policy in policies:
        split_results = {
            split: _evaluate_policy(candidates_by_split[split], policy)
            for split in selection_splits
        }
        summaries = {split: result["summary"] for split, result in split_results.items()}
        trade_counts = {split: int(summary["trade_count"]) for split, summary in summaries.items()}
        pnls = {split: float(summary["pnl"]) for split, summary in summaries.items()}
        avg_pnls = {split: summary["avg_pnl"] for split, summary in summaries.items()}
        min_trade_count = min(trade_counts.values()) if trade_counts else 0
        min_pnl = min(pnls.values()) if pnls else 0.0
        min_split_avg_pnl = (
            min(float(value) for value in avg_pnls.values() if value is not None)
            if avg_pnls and all(value is not None for value in avg_pnls.values())
            else None
        )
        preferred = (
            min_trade_count >= min_trades
            and min_pnl > 0.0
            and min_split_avg_pnl is not None
            and min_split_avg_pnl >= min_avg_pnl
        )
        score = (
            float(preferred),
            min_pnl,
            _score_float(min_split_avg_pnl),
            float(summaries[selection_splits[-1]]["pnl"]),
            _score_float(summaries[selection_splits[-1]]["avg_pnl"]),
            float(min_trade_count),
        )
        candidates.append(
            {
                "policy": asdict(policy),
                "selection": {
                    "selection_splits": selection_splits,
                    "min_trades_per_split": min_trades,
                    "min_avg_pnl": min_avg_pnl,
                    "preferred": preferred,
                    "score": score,
                    "trade_counts": trade_counts,
                    "pnls": pnls,
                    "avg_pnls": avg_pnls,
                    "min_trade_count": min_trade_count,
                    "min_pnl": min_pnl,
                    "min_split_avg_pnl": min_split_avg_pnl,
                },
                "split_results": split_results,
            }
        )
    candidates = [
        candidate
        for candidate in candidates
        if any(int(result["summary"]["trade_count"]) > 0 for result in candidate["split_results"].values())
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda candidate: tuple(candidate["selection"]["score"]), reverse=True)
    return candidates[0]


def _score_float(value: Any) -> float:
    if value is None:
        return -1_000_000_000.0
    return float(value)


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
            "The PnL-stability-selected v7 policy beats the fixed v6 gate on the held-out "
            "split and produces positive one-way settlement PnL. This is a small-scope "
            "offline check; the next step is executor integration or paper-only shadow "
            "with the same policy thresholds."
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
    policies = (
        "v6_current_gate",
        "v7_selected_by_pnl_stability",
        "v7_residual_edge_gate",
        "v7_hybrid_edge_gate",
        "market_favorite_baseline",
    )
    return {
        "status": report["status"],
        "best_v7_policy_selected_by_pnl_stability": report["best_v7_policy_selected_by_pnl_stability"],
        "validation": {
            policy: report["split_results"][validation_split][policy]["summary"]
            for policy in policies
        },
        "test": {
            policy: report["split_results"][test_split][policy]["summary"]
            for policy in policies
        },
    }


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# xgboost-v7 Execution-Restricted PnL Evaluation",
        "",
        f"Status: **{report['status']}**",
        "",
        f"Model artifact: `{report['model_json_path']}`",
        f"Model artifact kind: `{report['model_artifact_kind']}`",
        f"Dataset: `{report['dataset_dir']}`",
        "",
        "This report ranks policies by executable one-way settlement PnL under "
        "entry-window, slippage, and one-trade-per-round constraints. Hit rate and "
        "settlement accuracy are diagnostics, not the metric of record.",
        "",
        "## Metric",
        "",
        "- Probability edge: `p_side - entry_worst_price`.",
        "- Residual edge: `residual_expected_edge_side`, when the artifact emits it.",
        "- Hybrid edge: probability gate plus residual edge floor.",
        "- Market baseline: buy the first market-favorite side without model edge.",
        "- Entry worst price: `ask + buy_slippage + fee`, capped at `0.99`.",
        "- PnL: win pays `1 - entry_worst_price`; loss pays `-entry_worst_price`.",
        "- Settlement buy-and-hold does not subtract the old volatility round-trip cost.",
        "- Each policy admits at most one settlement trade per round.",
        "",
        "## Selected Policy",
        "",
        f"- selection splits: `{','.join(report['selection_splits'])}`",
        f"- validation split: `{report['validation_split']}`",
        f"- test split: `{report['test_split']}`",
        f"- signal source: `{report['best_v7_policy_selected_by_pnl_stability']['signal_source']}`",
        f"- min confidence: `{report['best_v7_policy_selected_by_pnl_stability']['min_confidence']}`",
        f"- min expected edge: `{report['best_v7_policy_selected_by_pnl_stability']['min_expected_edge']}`",
        "",
        "## Split Results",
        "",
        "| Split | Policy | Trades | Coverage | PnL | Avg PnL | Hit rate | Mean edge | Mean price | Max DD |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    policy_order = (
        "v6_current_gate",
        "v7_selected_by_pnl_stability",
        "v7_residual_edge_gate",
        "v7_hybrid_edge_gate",
        "market_favorite_baseline",
        "first_model_side_no_edge_floor",
    )
    for split, results in report["split_results"].items():
        for policy_name in policy_order:
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
