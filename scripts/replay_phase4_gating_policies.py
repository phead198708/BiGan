#!/usr/bin/env python3
"""Replay Phase 4 gating policy variants against a signal opportunity report."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PolicySpec:
    name: str
    edge_threshold: float
    min_entry_price: float
    fresh_edge_threshold: float | None = None
    near_min_price_band: float = 0.05
    near_min_fresh_edge_threshold: float = 0.50
    near_min_seconds_to_expiry: float = 420.0
    min_seconds_to_expiry: float = 300.0
    max_seconds_to_expiry: float = 1200.0
    no_new_entry_before_expiry_seconds: float = 300.0

    @property
    def effective_fresh_edge_threshold(self) -> float:
        return self.edge_threshold if self.fresh_edge_threshold is None else self.fresh_edge_threshold


def main() -> int:
    args = _parse_args()
    payload = json.loads(Path(args.input_json_path).read_text(encoding="utf-8"))
    rows = payload["rows"]
    policies = _default_policy_grid(
        edge_thresholds=_parse_float_list(args.edge_thresholds),
        min_entry_prices=_parse_float_list(args.min_entry_prices),
    )
    if args.include_cheap_token_gates:
        policies.extend(_cheap_token_policy_grid())
    report_payload = {
        "input_json_path": args.input_json_path,
        "base_summary": payload.get("summary", {}),
        "policies": [asdict(policy) for policy in policies],
        "metrics": [score_policy(rows, policy) for policy in policies],
    }
    report_payload["rankings"] = {
        "best_signal_f1": _rank(report_payload["metrics"], "signal_f1", limit=args.rank_limit),
        "best_round_first_f1": _rank(
            report_payload["metrics"], "round_first_f1", limit=args.rank_limit
        ),
        "best_signal_recall_with_precision_floor": _rank_with_floor(
            report_payload["metrics"],
            sort_key="signal_recall",
            floor_key="signal_precision",
            floor=args.precision_floor,
            limit=args.rank_limit,
        ),
    }
    if args.output_json_path:
        output_json_path = Path(args.output_json_path)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n")
    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(markdown_report(report_payload), encoding="utf-8")
    print(json.dumps(report_payload["rankings"], indent=2, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json-path", required=True)
    parser.add_argument("--output-json-path", default="")
    parser.add_argument("--report-path", default="")
    parser.add_argument("--edge-thresholds", default="0.45,0.30,0.20,0.10")
    parser.add_argument("--min-entry-prices", default="0.35,0.30,0.25,0.20,0.15")
    parser.add_argument("--precision-floor", type=float, default=0.45)
    parser.add_argument("--rank-limit", type=int, default=8)
    parser.add_argument(
        "--include-cheap-token-gates",
        action="store_true",
        help="Also score ask/fresh-edge/seconds-to-expiry cheap-token gate variants.",
    )
    return parser.parse_args()


def _default_policy_grid(
    *, edge_thresholds: list[float], min_entry_prices: list[float]
) -> list[PolicySpec]:
    policies: list[PolicySpec] = []
    for edge_threshold in edge_thresholds:
        for min_entry_price in min_entry_prices:
            policies.append(
                PolicySpec(
                    name=f"edge_{edge_threshold:.2f}_min_entry_{min_entry_price:.2f}",
                    edge_threshold=edge_threshold,
                    min_entry_price=min_entry_price,
                )
            )
    return policies


def _cheap_token_policy_grid() -> list[PolicySpec]:
    specs = [
        (0.20, 0.40, 420.0),
        (0.20, 0.50, 420.0),
        (0.25, 0.40, 420.0),
        (0.25, 0.50, 420.0),
    ]
    return [
        PolicySpec(
            name=f"cheap_ask_{ask_floor:.2f}_fresh_{fresh_edge:.2f}_seconds_{seconds:.0f}",
            edge_threshold=-999.0,
            min_entry_price=ask_floor,
            fresh_edge_threshold=fresh_edge,
            near_min_fresh_edge_threshold=fresh_edge,
            min_seconds_to_expiry=seconds,
            near_min_seconds_to_expiry=seconds,
        )
        for ask_floor, fresh_edge, seconds in specs
    ]


def score_policy(rows: list[dict[str, Any]], policy: PolicySpec) -> dict[str, Any]:
    opportunity_rows = [row for row in rows if _has_opportunity(row)]
    candidates = [row for row in rows if is_candidate(row, policy)]
    candidate_opportunities = [row for row in candidates if _has_opportunity(row)]
    first_candidates = _first_candidate_by_round(candidates)
    first_candidate_opportunities = [row for row in first_candidates if _has_opportunity(row)]
    opportunity_rounds = {row["round_slug"] for row in opportunity_rows}
    covered_opportunity_rounds = {
        row["round_slug"] for row in first_candidate_opportunities if row["round_slug"] in opportunity_rounds
    }
    candidate_rounds = {row["round_slug"] for row in candidates}
    signal_precision = _safe_ratio(len(candidate_opportunities), len(candidates))
    signal_recall = _safe_ratio(len(candidate_opportunities), len(opportunity_rows))
    round_first_precision = _safe_ratio(
        len(first_candidate_opportunities), len(first_candidates)
    )
    round_first_recall = _safe_ratio(len(covered_opportunity_rounds), len(opportunity_rounds))
    volatility_metrics = _opportunity_type_metrics(
        rows,
        candidates,
        first_candidates,
        key="volatility_exit_opportunity",
        prefix="volatility",
    )
    settlement_metrics = _opportunity_type_metrics(
        rows,
        candidates,
        first_candidates,
        key="settlement_hold_opportunity",
        prefix="settlement",
    )
    return {
        "policy": asdict(policy),
        "candidate_count": len(candidates),
        "candidate_round_count": len(candidate_rounds),
        "opportunities_allowed": len(candidate_opportunities),
        "opportunities_blocked": len(opportunity_rows) - len(candidate_opportunities),
        "false_positive_candidates": len(candidates) - len(candidate_opportunities),
        "signal_precision": signal_precision,
        "signal_recall": signal_recall,
        "signal_f1": _f1(signal_precision, signal_recall),
        "round_first_candidate_count": len(first_candidates),
        "round_first_opportunities": len(first_candidate_opportunities),
        "round_opportunity_count": len(opportunity_rounds),
        "round_first_precision": round_first_precision,
        "round_first_recall": round_first_recall,
        "round_first_f1": _f1(round_first_precision, round_first_recall),
        "avg_allowed_opportunity_gain": _avg_gain(candidate_opportunities),
        "avg_round_first_opportunity_gain": _avg_gain(first_candidate_opportunities),
        "candidate_side_counts": _side_counts(candidates),
        "candidate_opportunity_side_counts": _side_counts(candidate_opportunities),
        "round_first_side_counts": _side_counts(first_candidates),
        "round_first_opportunity_side_counts": _side_counts(first_candidate_opportunities),
        **volatility_metrics,
        **settlement_metrics,
    }


def is_candidate(row: dict[str, Any], policy: PolicySpec) -> bool:
    seconds_to_expiry = float(row["seconds_to_expiry_at_decision"])
    if seconds_to_expiry < policy.no_new_entry_before_expiry_seconds:
        return False
    if seconds_to_expiry < policy.min_seconds_to_expiry:
        return False
    if seconds_to_expiry > policy.max_seconds_to_expiry:
        return False
    if float(row["edge"]) < policy.edge_threshold:
        return False
    entry_ask = row.get("entry_ask")
    entry_worst_price = row.get("entry_worst_price")
    if entry_ask is None or entry_worst_price is None:
        return False
    entry_ask = float(entry_ask)
    entry_worst_price = float(entry_worst_price)
    if entry_ask < policy.min_entry_price or entry_worst_price < policy.min_entry_price:
        return False
    fresh_edge = float(row["token_probability"]) - entry_worst_price
    near_min_ceiling = policy.min_entry_price + policy.near_min_price_band
    if entry_ask < near_min_ceiling or entry_worst_price < near_min_ceiling:
        if fresh_edge < policy.near_min_fresh_edge_threshold:
            return False
        if seconds_to_expiry < policy.near_min_seconds_to_expiry:
            return False
    return fresh_edge >= policy.effective_fresh_edge_threshold


def markdown_report(payload: dict[str, Any]) -> str:
    base_summary = payload["base_summary"]
    metrics = payload["metrics"]
    lines = [
        "# Phase 4 Gating Policy Replay",
        "",
        "## Scope",
        "",
        f"- Input: `{payload['input_json_path']}`",
        f"- Signals: {base_summary.get('signals')}",
        f"- Volatility-exit opportunities: {base_summary.get('volatility_exit_opportunities')}",
        f"- Settlement-hold opportunities: {base_summary.get('settlement_hold_opportunities')}",
        f"- Combined opportunities: {base_summary.get('opportunities')}",
        f"- Current-policy opportunity recall: {_fmt(base_summary.get('gating_confusion', {}).get('opportunity_recall'))}",
        "",
        "Signal-level metrics count every eligible signal. Round-first metrics approximate live execution by taking the first eligible signal per round.",
        "This is a signal-side policy replay; it does not simulate complement-token orderbook checks, order fill failures, or realized account cashflow.",
        "",
        "## Policy Grid",
        "",
        "| Policy | Cand. | Opps Allowed | Precision | Recall | F1 | Vol Allowed | Vol P | Vol R | Settle Allowed | Settle P | Settle R | Round First Cand. | Round First Opps | Round Precision | Round Recall | Round F1 | Cand. Sides | Opp Sides |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in metrics:
        policy_name = item["policy"]["name"]
        lines.append(
            "| "
            f"`{policy_name}` | "
            f"{item['candidate_count']} | "
            f"{item['opportunities_allowed']} | "
            f"{_fmt(item['signal_precision'])} | "
            f"{_fmt(item['signal_recall'])} | "
            f"{_fmt(item['signal_f1'])} | "
            f"{item['volatility_opportunities_allowed']} | "
            f"{_fmt(item['volatility_precision'])} | "
            f"{_fmt(item['volatility_recall'])} | "
            f"{item['settlement_opportunities_allowed']} | "
            f"{_fmt(item['settlement_precision'])} | "
            f"{_fmt(item['settlement_recall'])} | "
            f"{item['round_first_candidate_count']} | "
            f"{item['round_first_opportunities']} | "
            f"{_fmt(item['round_first_precision'])} | "
            f"{_fmt(item['round_first_recall'])} | "
            f"{_fmt(item['round_first_f1'])} | "
            f"{_counts(item['candidate_side_counts'])} | "
            f"{_counts(item['candidate_opportunity_side_counts'])} |"
        )
    lines.extend(["", "## Rankings", ""])
    for title, rows in payload["rankings"].items():
        lines.extend([f"### {title}", ""])
        if not rows:
            lines.append("- No policy met the ranking filter.")
            lines.append("")
            continue
        lines.append("| Policy | Cand. | Opps Allowed | Precision | Recall | F1 | Round First Cand. | Round First Opps | Round Precision | Round Recall | Round F1 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for item in rows:
            lines.append(
                "| "
                f"`{item['policy']['name']}` | "
                f"{item['candidate_count']} | "
                f"{item['opportunities_allowed']} | "
                f"{_fmt(item['signal_precision'])} | "
                f"{_fmt(item['signal_recall'])} | "
                f"{_fmt(item['signal_f1'])} | "
                f"{item['round_first_candidate_count']} | "
                f"{item['round_first_opportunities']} | "
                f"{_fmt(item['round_first_precision'])} | "
                f"{_fmt(item['round_first_recall'])} | "
                f"{_fmt(item['round_first_f1'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Readout",
            "",
            "- Lowering edge alone improves recall but does not produce a clean selector.",
            "- Lowering `min_entry_price` materially increases cheap-token volatility capture; treat those variants as paper-only until slippage/fill quality is verified.",
            "- Round-first metrics are the safer proxy for live behavior because Phase 4 caps one filled position per round.",
            "- A production change should still pass a live paper/shadow run because this replay does not model complement-token validation or fill quality.",
            "",
        ]
    )
    return "\n".join(lines)


def _first_candidate_by_round(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    first: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: (item["decision_ts"], item["event_id"])):
        first.setdefault(row["round_slug"], row)
    return list(first.values())


def _has_opportunity(row: dict[str, Any]) -> bool:
    return bool(row.get("volatility_exit_opportunity") or row.get("settlement_hold_opportunity"))


def _opportunity_type_metrics(
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    first_candidates: list[dict[str, Any]],
    *,
    key: str,
    prefix: str,
) -> dict[str, Any]:
    opportunity_rows = [row for row in rows if bool(row.get(key))]
    candidate_opportunities = [row for row in candidates if bool(row.get(key))]
    first_candidate_opportunities = [row for row in first_candidates if bool(row.get(key))]
    opportunity_rounds = {row["round_slug"] for row in opportunity_rows}
    covered_opportunity_rounds = {
        row["round_slug"]
        for row in first_candidate_opportunities
        if row["round_slug"] in opportunity_rounds
    }
    precision = _safe_ratio(len(candidate_opportunities), len(candidates))
    recall = _safe_ratio(len(candidate_opportunities), len(opportunity_rows))
    round_first_precision = _safe_ratio(
        len(first_candidate_opportunities), len(first_candidates)
    )
    round_first_recall = _safe_ratio(len(covered_opportunity_rounds), len(opportunity_rounds))
    return {
        f"{prefix}_opportunity_count": len(opportunity_rows),
        f"{prefix}_opportunities_allowed": len(candidate_opportunities),
        f"{prefix}_opportunities_blocked": len(opportunity_rows) - len(candidate_opportunities),
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": _f1(precision, recall),
        f"{prefix}_round_first_opportunities": len(first_candidate_opportunities),
        f"{prefix}_round_opportunity_count": len(opportunity_rounds),
        f"{prefix}_round_first_precision": round_first_precision,
        f"{prefix}_round_first_recall": round_first_recall,
        f"{prefix}_round_first_f1": _f1(round_first_precision, round_first_recall),
    }


def _side_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        side = str(row.get("outcome_side") or "UNKNOWN")
        counts[side] = counts.get(side, 0) + 1
    return dict(sorted(counts.items()))


def _avg_gain(rows: list[dict[str, Any]]) -> float | None:
    gains = [float(row["max_exit_gain"]) for row in rows if row.get("max_exit_gain") is not None]
    if not gains:
        return None
    return sum(gains) / len(gains)


def _rank(rows: list[dict[str, Any]], key: str, *, limit: int) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (row.get(key) is not None, row.get(key) or -1.0),
        reverse=True,
    )[:limit]


def _rank_with_floor(
    rows: list[dict[str, Any]],
    *,
    sort_key: str,
    floor_key: str,
    floor: float,
    limit: int,
) -> list[dict[str, Any]]:
    filtered = [row for row in rows if (row.get(floor_key) or 0.0) >= floor]
    return sorted(filtered, key=lambda row: row.get(sort_key) or -1.0, reverse=True)[:limit]


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _counts(counts: dict[str, int]) -> str:
    if not counts:
        return "-"
    return ", ".join(f"{key}:{value}" for key, value in counts.items())


if __name__ == "__main__":
    raise SystemExit(main())
