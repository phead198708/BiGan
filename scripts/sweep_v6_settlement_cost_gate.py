#!/usr/bin/env python3
"""Sweep v6 BTC-15M settlement thresholds with cost-aware execution gates."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REPLAY_SCRIPT = REPO_ROOT / "scripts" / "replay_v6_btc15m_execution_restricted.py"


def main() -> int:
    args = _parse_args()
    replay = _load_replay_module()
    model = replay.load_xgboost_v6_model(Path(args.model_json_path))
    config = json.loads(Path(args.model_config_path).read_text(encoding="utf-8"))
    round_trip_cost = float(config.get("round_trip_cost", 0.072))
    ev_margin = float(config.get("ev_margin", 0.01))
    artifact = json.loads(Path(args.model_json_path).read_text(encoding="utf-8"))
    gain_priors = {
        str(key): float(value)
        for key, value in artifact.get("volatility_gain_priors", {}).items()
    }
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    settlement_thresholds = _float_grid(args.settlement_thresholds)
    edge_thresholds = _float_grid(args.settlement_min_edges)

    rows_by_split = {
        split: replay._load_btc15_rows(Path(args.dataset_dir) / f"{split}.parquet")
        for split in splits
    }
    payloads_by_split = {
        split: model.predict_payload_many(rows)
        for split, rows in rows_by_split.items()
    }
    results: list[dict[str, Any]] = []
    for settlement_threshold in settlement_thresholds:
        for min_edge in edge_thresholds:
            split_summaries: dict[str, Any] = {}
            combined_trades = []
            combined_gate_counts: dict[str, int] = {}
            for split in splits:
                gate_counts: dict[str, int] = {}
                trades = replay._collect_execution_restricted_trades(
                    rows_by_split[split],
                    payloads_by_split[split],
                    joint_rule={
                        "settlement_threshold": settlement_threshold,
                        "neutral_cap": args.neutral_cap,
                        "volatility_threshold": args.volatility_threshold,
                    },
                    round_trip_cost=round_trip_cost,
                    ev_margin=ev_margin,
                    gain_priors=gain_priors,
                    policy=replay.Phase4EntryPolicy(
                        min_entry_price=args.min_entry_price,
                        near_min_price_band=args.near_min_price_band,
                        near_min_fresh_edge_threshold=args.near_min_fresh_edge_threshold,
                        near_min_seconds_to_expiry=args.near_min_seconds_to_expiry,
                        edge_threshold=-999.0,
                        settlement_edge_threshold=min_edge,
                    ),
                    buy_slippage=args.buy_slippage,
                    min_seconds_to_expiry=args.min_seconds_to_expiry,
                    max_seconds_to_expiry=args.max_seconds_to_expiry,
                    no_new_entry_before_expiry_seconds=args.no_new_entry_before_expiry_seconds,
                    gate_counts=gate_counts,
                )
                combined_trades.extend(trades)
                _merge_counts(combined_gate_counts, gate_counts)
                split_summaries[split] = _slim_summary(replay._summarize_trades(trades))
                split_summaries[split]["gate_counts"] = dict(sorted(gate_counts.items()))
            combined = _slim_summary(replay._summarize_trades(combined_trades))
            combined["gate_counts"] = dict(sorted(combined_gate_counts.items()))
            results.append(
                {
                    "settlement_threshold": settlement_threshold,
                    "settlement_min_edge_after_cost": min_edge,
                    "splits": split_summaries,
                    "combined": combined,
                }
            )

    results.sort(
        key=lambda row: (
            float(row["combined"]["pnl"]),
            int(row["combined"]["trade_count"]),
        ),
        reverse=True,
    )
    report = {
        "model_json_path": args.model_json_path,
        "dataset_dir": args.dataset_dir,
        "splits": splits,
        "round_trip_cost": round_trip_cost,
        "ev_margin": ev_margin,
        "results": results,
        "best_by_combined_pnl": results[0] if results else None,
        "best_executed_by_combined_pnl": next(
            (row for row in results if int(row["combined"]["trade_count"]) > 0),
            None,
        ),
    }
    if args.output_json_path:
        path = Path(args.output_json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.report_path:
        path = Path(args.report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_markdown_report(report, limit=args.report_limit), encoding="utf-8")
    print(json.dumps(report["best_executed_by_combined_pnl"], indent=2, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    default_model = (
        "data/model-runs/xgboost-v6-issue93-94-15m-only-volatility-20260602T135044Z/"
        "model-single-grid/model.json"
    )
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--settlement-thresholds", default="0.50,0.55,0.60,0.65,0.70,0.75")
    parser.add_argument("--settlement-min-edges", default="0.04,0.06,0.08,0.082,0.10,0.12,0.15")
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
    parser.add_argument("--report-limit", type=int, default=20)
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


def _merge_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def _slim_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "trade_count": int(summary["trade_count"]),
        "pnl": float(summary["pnl"]),
        "avg_pnl": summary["avg_pnl"],
        "hit_rate": summary["hit_rate"],
        "max_drawdown": float(summary["max_drawdown"]),
    }


def _markdown_report(report: dict[str, Any], *, limit: int) -> str:
    lines = [
        "# v6 BTC-15M Settlement Cost-Gate Sweep",
        "",
        f"Model: `{report['model_json_path']}`",
        f"Dataset: `{report['dataset_dir']}`",
        f"Round-trip cost: `{report['round_trip_cost']}`",
        f"EV margin: `{report['ev_margin']}`",
        "",
        "| Rank | settlement_threshold | min_edge_after_cost | Trades | PnL | Avg PnL | Hit rate | Max DD |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    executable_rows = [
        row for row in report["results"] if int(row["combined"]["trade_count"]) > 0
    ]
    display_rows = executable_rows[:limit]
    if not display_rows:
        display_rows = report["results"][:limit]
    for idx, row in enumerate(display_rows, start=1):
        combined = row["combined"]
        hit_rate = combined["hit_rate"]
        lines.append(
            f"| {idx} | {row['settlement_threshold']:.3f} | "
            f"{row['settlement_min_edge_after_cost']:.3f} | "
            f"{combined['trade_count']} | {combined['pnl']:.4f} | "
            f"{(combined['avg_pnl'] or 0.0):.4f} | "
            f"{(hit_rate or 0.0):.4f} | {combined['max_drawdown']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
