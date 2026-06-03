#!/usr/bin/env python3
"""Replay v6 BTC-15M settlement gate with Phase-4-style execution restrictions."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from bigan.execution.phase4_policy import Phase4EntryPolicy, entry_price_skip_reason
from bigan.execution.v6_gate import V6JointGateConfig, evaluate_v6_settlement_side
from bigan.modeling.families import market_family_from_symbol
from bigan.modeling.xgboost_v6 import (
    XGBOOST_V6_MODEL_VERSION,
    load_xgboost_v6_model,
)

BTC_15M_FAMILY = "BTC-15M"
DEFAULT_JOINT_RULE = {
    "settlement_threshold": 0.50,
    "neutral_cap": 0.25,
    "volatility_threshold": 0.60,
}


@dataclass(frozen=True, slots=True)
class ReplayTrade:
    feature_ts: int
    round_slug: str
    outcome_side: str
    side: str
    entry_cost: float
    entry_worst_price: float
    pnl: float
    true_label: str
    joint_gate: str
    execution_gate: str


def main() -> int:
    args = _parse_args()
    model = load_xgboost_v6_model(Path(args.model_json_path))
    if model.model_version != XGBOOST_V6_MODEL_VERSION:
        raise SystemExit(f"expected model_version={XGBOOST_V6_MODEL_VERSION!r}")

    artifact = json.loads(Path(args.model_json_path).read_text(encoding="utf-8"))
    gain_priors = {
        str(key): float(value)
        for key, value in artifact.get("volatility_gain_priors", {}).items()
    }
    joint_rule = {
        **DEFAULT_JOINT_RULE,
        "settlement_threshold": args.settlement_threshold,
        "neutral_cap": args.neutral_cap,
        "volatility_threshold": args.volatility_threshold,
    }
    policy = Phase4EntryPolicy(
        min_entry_price=args.min_entry_price,
        near_min_price_band=args.near_min_price_band,
        near_min_fresh_edge_threshold=args.near_min_fresh_edge_threshold,
        near_min_seconds_to_expiry=args.near_min_seconds_to_expiry,
        edge_threshold=-999.0,
        settlement_edge_threshold=0.0,
    )
    config = json.loads(Path(args.model_config_path).read_text(encoding="utf-8"))
    round_trip_cost = float(config.get("round_trip_cost", 0.072))
    ev_margin = float(config.get("ev_margin", 0.01))

    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in args.splits.split(","):
        split = split.strip()
        if not split:
            continue
        table_path = Path(args.dataset_dir) / f"{split}.parquet"
        rows_by_split[split] = _load_btc15_rows(table_path)

    report = {
        "status": "PROMISING_BUT_NOT_PROMOTION_EVIDENCE",
        "model_run": str(Path(args.model_json_path).parent),
        "dataset_dir": args.dataset_dir,
        "family_filter": BTC_15M_FAMILY,
        "joint_rule": joint_rule,
        "round_trip_cost": round_trip_cost,
        "ev_margin": ev_margin,
        "gain_priors": gain_priors,
        "execution_policy": {
            "min_entry_price": policy.min_entry_price,
            "near_min_price_band": policy.near_min_price_band,
            "near_min_fresh_edge_threshold": policy.near_min_fresh_edge_threshold,
            "near_min_seconds_to_expiry": policy.near_min_seconds_to_expiry,
            "min_seconds_to_expiry": args.min_seconds_to_expiry,
            "max_seconds_to_expiry": args.max_seconds_to_expiry,
            "no_new_entry_before_expiry_seconds": args.no_new_entry_before_expiry_seconds,
            "buy_slippage": args.buy_slippage,
            "round_first_per_round_slug": True,
            "require_observed_exit_path": False,
        },
        "splits": {},
        "paper_shadow_gate": {
            "required_next_step": "paper/orderbook-only shadow with account-cashflow reconciliation",
            "direct_promotion_allowed": False,
        },
    }

    all_pass = True
    for split, rows in rows_by_split.items():
        payloads = model.predict_payload_many(rows)
        offline = _offline_summary(
            rows,
            payloads,
            joint_rule=joint_rule,
            round_trip_cost=round_trip_cost,
            ev_margin=ev_margin,
            gain_priors=gain_priors,
        )
        restricted = _execution_restricted_summary(
            rows,
            payloads,
            joint_rule=joint_rule,
            round_trip_cost=round_trip_cost,
            ev_margin=ev_margin,
            gain_priors=gain_priors,
            policy=policy,
            buy_slippage=args.buy_slippage,
            min_seconds_to_expiry=args.min_seconds_to_expiry,
            max_seconds_to_expiry=args.max_seconds_to_expiry,
            no_new_entry_before_expiry_seconds=args.no_new_entry_before_expiry_seconds,
        )
        split_pass = bool(restricted["passes_paper_shadow_prerequisite"])
        all_pass = all_pass and split_pass
        report["splits"][split] = {
            "row_count": len(rows),
            "offline_settlement_gate": offline,
            "execution_restricted": restricted,
            "passes_paper_shadow_prerequisite": split_pass,
        }

    report["paper_shadow_gate"]["execution_restricted_replay_pass"] = all_pass
    if args.output_json_path:
        output_json_path = Path(args.output_json_path)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(_markdown_report(report), encoding="utf-8")
    print(json.dumps(report["paper_shadow_gate"], indent=2, sort_keys=True))
    return 0 if all_pass else 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_model = (
        "data/model-runs/xgboost-v6-issue93-94-15m-only-volatility-20260602T135044Z/"
        "model-single-grid/model.json"
    )
    parser.add_argument("--model-json-path", default=default_model)
    parser.add_argument(
        "--model-config-path",
        default=str(Path(default_model).parent / "xgboost_v6_config.json"),
    )
    parser.add_argument(
        "--dataset-dir",
        default=(
            "data/model-runs/xgboost-v6-issue93-94-15m-only-volatility-20260602T135044Z/dataset"
        ),
    )
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--settlement-threshold", type=float, default=0.50)
    parser.add_argument("--neutral-cap", type=float, default=0.25)
    parser.add_argument("--volatility-threshold", type=float, default=0.60)
    parser.add_argument("--min-entry-price", type=float, default=0.35)
    parser.add_argument("--near-min-price-band", type=float, default=0.05)
    parser.add_argument("--near-min-fresh-edge-threshold", type=float, default=0.50)
    parser.add_argument("--near-min-seconds-to-expiry", type=float, default=420.0)
    parser.add_argument("--min-seconds-to-expiry", type=float, default=300.0)
    parser.add_argument("--max-seconds-to-expiry", type=float, default=1200.0)
    parser.add_argument("--no-new-entry-before-expiry-seconds", type=float, default=300.0)
    parser.add_argument("--buy-slippage", type=float, default=0.02)
    parser.add_argument("--output-json-path", default="")
    parser.add_argument("--report-path", default="")
    return parser.parse_args()


def _load_btc15_rows(table_path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(table_path)
    rows = [
        row
        for row in table.to_pylist()
        if market_family_from_symbol(row.get("canonical_symbol") or row.get("symbol"))
        == BTC_15M_FAMILY
    ]
    rows.sort(key=lambda row: (int(row.get("feature_ts") or 0), str(row.get("canonical_symbol") or "")))
    return rows


def _offline_summary(
    rows: list[dict[str, Any]],
    payloads: list[dict[str, float | str]],
    *,
    joint_rule: dict[str, Any],
    round_trip_cost: float,
    ev_margin: float,
    gain_priors: dict[str, float],
) -> dict[str, Any]:
    return _summarize_trades(
        _collect_offline_trades(
            rows,
            payloads,
            joint_rule=joint_rule,
            round_trip_cost=round_trip_cost,
            ev_margin=ev_margin,
            gain_priors=gain_priors,
        )
    )


def _execution_restricted_summary(
    rows: list[dict[str, Any]],
    payloads: list[dict[str, float | str]],
    *,
    joint_rule: dict[str, Any],
    round_trip_cost: float,
    ev_margin: float,
    gain_priors: dict[str, float],
    policy: Phase4EntryPolicy,
    buy_slippage: float,
    min_seconds_to_expiry: float,
    max_seconds_to_expiry: float,
    no_new_entry_before_expiry_seconds: float,
) -> dict[str, Any]:
    gate_counts: dict[str, int] = {}
    trades = _collect_execution_restricted_trades(
        rows,
        payloads,
        joint_rule=joint_rule,
        round_trip_cost=round_trip_cost,
        ev_margin=ev_margin,
        gain_priors=gain_priors,
        policy=policy,
        buy_slippage=buy_slippage,
        min_seconds_to_expiry=min_seconds_to_expiry,
        max_seconds_to_expiry=max_seconds_to_expiry,
        no_new_entry_before_expiry_seconds=no_new_entry_before_expiry_seconds,
        gate_counts=gate_counts,
    )
    summary = _summarize_trades(trades)
    summary["gate_counts"] = dict(sorted(gate_counts.items()))
    summary["passes_paper_shadow_prerequisite"] = (
        int(summary["trade_count"]) >= 5
        and float(summary["pnl"]) > 0.0
        and float(summary.get("hit_rate") or 0.0) >= 0.50
    )
    return summary


def _collect_offline_trades(
    rows: list[dict[str, Any]],
    payloads: list[dict[str, float | str]],
    *,
    joint_rule: dict[str, Any],
    round_trip_cost: float,
    ev_margin: float,
    gain_priors: dict[str, float],
) -> list[ReplayTrade]:
    trades: list[ReplayTrade] = []
    for row, payload in zip(rows, payloads, strict=True):
        side = _settlement_decision_from_payload(payload, joint_rule=joint_rule)
        if side is None:
            continue
        entry_cost = _entry_cost_for_side(row, side)
        true_label = _settlement_label(row)
        pnl = _trade_pnl(true_label, side, entry_cost, round_trip_cost)
        trades.append(
            ReplayTrade(
                feature_ts=int(row.get("feature_ts") or 0),
                round_slug=_round_slug(row),
                outcome_side=_outcome_side(row),
                side=side,
                entry_cost=entry_cost,
                entry_worst_price=entry_cost,
                pnl=pnl,
                true_label=true_label,
                joint_gate="settlement_pass",
                execution_gate="offline_proxy",
            )
        )
    return trades


def _collect_execution_restricted_trades(
    rows: list[dict[str, Any]],
    payloads: list[dict[str, float | str]],
    *,
    joint_rule: dict[str, Any],
    round_trip_cost: float,
    ev_margin: float,
    gain_priors: dict[str, float],
    policy: Phase4EntryPolicy,
    buy_slippage: float,
    min_seconds_to_expiry: float,
    max_seconds_to_expiry: float,
    no_new_entry_before_expiry_seconds: float,
    gate_counts: dict[str, int],
) -> list[ReplayTrade]:
    trades: list[ReplayTrade] = []
    seen_rounds: set[str] = set()

    for row, payload in zip(rows, payloads, strict=True):
        side = _settlement_decision_from_payload(payload, joint_rule=joint_rule)
        if side is None:
            _bump(gate_counts, "settlement_gate_miss")
            continue

        round_slug = _round_slug(row)
        if round_slug in seen_rounds:
            _bump(gate_counts, "round_first_blocked")
            continue

        seconds_to_expiry = _seconds_to_expiry(row)
        seconds_since_start = _seconds_since_round_start(row)
        if seconds_to_expiry is None or seconds_since_start is None:
            _bump(gate_counts, "missing_round_timing")
            continue
        if seconds_since_start < 0.0:
            _bump(gate_counts, "before_round_start")
            continue
        if seconds_to_expiry < no_new_entry_before_expiry_seconds:
            _bump(gate_counts, "no_new_entry_window")
            continue
        if seconds_to_expiry < min_seconds_to_expiry:
            _bump(gate_counts, "below_min_seconds_to_expiry")
            continue
        if seconds_to_expiry > max_seconds_to_expiry:
            _bump(gate_counts, "above_max_seconds_to_expiry")
            continue

        ask = _entry_ask(row, side)
        if ask is None:
            _bump(gate_counts, "missing_entry_quote")
            continue
        worst_price = min(1.0, ask + buy_slippage)
        token_probability = float(payload["p_up"] if side == "UP" else payload["p_down"])
        skip_reason = entry_price_skip_reason(
            ask=ask,
            worst_price=worst_price,
            fresh_edge_at_worst=token_probability - worst_price,
            seconds_to_expiry=seconds_to_expiry,
            policy=policy,
        )
        if skip_reason is not None:
            _bump(gate_counts, skip_reason)
            continue

        entry_cost = worst_price
        true_label = _settlement_label(row)
        pnl = _trade_pnl(true_label, side, entry_cost, round_trip_cost)
        seen_rounds.add(round_slug)
        trades.append(
            ReplayTrade(
                feature_ts=int(row.get("feature_ts") or 0),
                round_slug=round_slug,
                outcome_side=_outcome_side(row),
                side=side,
                entry_cost=entry_cost,
                entry_worst_price=worst_price,
                pnl=pnl,
                true_label=true_label,
                joint_gate="settlement_pass",
                execution_gate="executor_candidate",
            )
        )
    return trades


def _settlement_decision_from_payload(
    payload: dict[str, float | str],
    *,
    joint_rule: dict[str, Any],
) -> str | None:
    return evaluate_v6_settlement_side(
        payload,
        V6JointGateConfig(
            settlement_threshold=float(joint_rule["settlement_threshold"]),
            neutral_cap=float(joint_rule.get("neutral_cap", 0.25)),
            volatility_threshold=float(joint_rule.get("volatility_threshold", 0.60)),
        ),
    )


def _summarize_trades(trades: list[ReplayTrade]) -> dict[str, Any]:
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    wins = 0
    for trade in trades:
        cumulative += trade.pnl
        peak = max(peak, cumulative)
        max_drawdown = min(max_drawdown, cumulative - peak)
        wins += int(trade.pnl > 0.0)
    return {
        "metric_of_record": "cost_adjusted_account_cashflow_proxy_pnl",
        "trade_count": len(trades),
        "pnl": cumulative,
        "avg_pnl": cumulative / len(trades) if trades else None,
        "hit_rate": wins / len(trades) if trades else None,
        "max_drawdown": max_drawdown,
        "sample_trades": [asdict(trade) for trade in trades[:12]],
    }


def _trade_pnl(true_label: str, side: str, entry_cost: float, round_trip_cost: float) -> float:
    if true_label == side:
        return 1.0 - entry_cost - round_trip_cost
    return -entry_cost - round_trip_cost


def _settlement_label(row: dict[str, Any]) -> str:
    label = row.get("label_settlement_3way")
    if label is None:
        return "NEUTRAL"
    return str(label)


def _entry_cost_for_side(row: dict[str, Any], side: str) -> float:
    ask = _entry_ask(row, side)
    if ask is not None:
        return ask
    implied = row.get("market_implied_prob")
    if implied is not None:
        value = float(implied)
        return value if side == "UP" else 1.0 - value
    return 0.5


def _entry_ask(row: dict[str, Any], side: str) -> float | None:
    ask = row.get("entry_ask_price")
    if ask is None:
        return None
    value = max(0.0, min(1.0, float(ask)))
    token_side = _outcome_side(row)
    if token_side == side:
        return value
    return max(0.0, min(1.0, 1.0 - value))


def _observed_max_exit_gain(row: dict[str, Any], side: str) -> float | None:
    column = "max_exit_gain_up" if side == "UP" else "max_exit_gain_down"
    value = row.get(column)
    return None if value is None else float(value)


def _round_bounds_ms(row: dict[str, Any]) -> tuple[int | None, int | None]:
    slug = _round_slug(row)
    slug_parts = slug.rsplit("-", 1)
    if len(slug_parts) == 2 and slug_parts[-1].isdigit():
        start_ms = int(slug_parts[-1]) * 1000
        end_ms = start_ms + 15 * 60 * 1000
        return start_ms, end_ms
    round_end_ts = row.get("round_end_ts")
    feature_ts = row.get("feature_ts")
    if round_end_ts is None:
        return None, None
    end_ms = int(round_end_ts)
    start_ms = end_ms - 15 * 60 * 1000 if feature_ts is not None else None
    return start_ms, end_ms


def _seconds_to_expiry(row: dict[str, Any]) -> float | None:
    feature_ts = row.get("feature_ts")
    start_ms, end_ms = _round_bounds_ms(row)
    if feature_ts is None or end_ms is None:
        return None
    return max(0.0, (end_ms - int(feature_ts)) / 1000.0)


def _seconds_since_round_start(row: dict[str, Any]) -> float | None:
    feature_ts = row.get("feature_ts")
    start_ms, _end_ms = _round_bounds_ms(row)
    if feature_ts is None or start_ms is None:
        return None
    return (int(feature_ts) - start_ms) / 1000.0


def _round_slug(row: dict[str, Any]) -> str:
    symbol = str(row.get("canonical_symbol") or row.get("symbol") or "")
    if ":" in symbol:
        return symbol.split(":", 2)[1]
    return str(row.get("source_market") or symbol)


def _outcome_side(row: dict[str, Any]) -> str:
    symbol = str(row.get("canonical_symbol") or "")
    if ":" in symbol:
        return symbol.rsplit(":", 1)[-1]
    return "UP"


def _bump(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Issue 93/94 v6 BTC-15M Execution-Restricted Replay",
        "",
        f"Status: **{report['status']}**",
        "",
        f"Model run: `{report['model_run']}`",
        "",
        "## Settlement Rule (15M mixed model)",
        "",
        f"- settlement_threshold={report['joint_rule']['settlement_threshold']}",
        f"- neutral_cap={report['joint_rule']['neutral_cap']}",
        f"- volatility_threshold={report['joint_rule']['volatility_threshold']}",
        "",
        "## Execution Restrictions",
        "",
        "- Phase 4 min-entry and near-min fresh-edge checks on observed ask + buy slippage",
        "- Seconds-to-expiry window and no-new-entry window",
        "- Round-first: one admitted trade per round slug",
        "- Settlement gate ignores volatility heads; volatility sleeve is evaluated separately in live paper execution",
        "",
        "This replay is closer to live execution than the offline proxy, but it still "
        "does not model complement-token checks, fill failures, or realized account cashflow.",
        "",
        "## Split Results",
        "",
        "| Split | Offline trades | Offline PnL | Restricted trades | Restricted PnL | Hit rate | Pass paper prereq |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for split, payload in report["splits"].items():
        offline = payload["offline_settlement_gate"]
        restricted = payload["execution_restricted"]
        lines.append(
            f"| {split} | {offline['trade_count']} | {offline['pnl']:.4f} | "
            f"{restricted['trade_count']} | {restricted['pnl']:.4f} | "
            f"{(restricted.get('hit_rate') or 0.0):.4f} | "
            f"{restricted['passes_paper_shadow_prerequisite']} |"
        )
    lines.extend(
        [
            "",
            "## Paper Shadow Gate",
            "",
            f"- execution_restricted_replay_pass: `{report['paper_shadow_gate']['execution_restricted_replay_pass']}`",
            f"- next step: {report['paper_shadow_gate']['required_next_step']}",
            f"- direct promotion allowed: `{report['paper_shadow_gate']['direct_promotion_allowed']}`",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
