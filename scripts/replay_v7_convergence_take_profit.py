#!/usr/bin/env python3
"""Replay v7 paper logs with convergence take-profit policy counterfactuals."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TakeProfitPolicy:
    take_profit_hold_edge: float = 0.03
    take_profit_residual_ratio: float = 0.40
    take_profit_price_convergence_move: float = 0.10
    take_profit_price_convergence_hold_edge_ratio: float = 0.50
    take_profit_force_exit_seconds: float = 180.0
    take_profit_hysteresis_bars: int = 2
    sell_slippage: float = 0.02
    up_hold_edge_tighten: float = 0.01


V7_PROFIT_LOCK_BEFORE_EXPIRY_REASON = "convergence_profit_lock_before_expiry"
V7_LOSS_SALVAGE_BEFORE_EXPIRY_REASON = "convergence_loss_salvage_before_expiry"
V7_SLOT_RELEASE_BEFORE_EXPIRY_REASON = "convergence_slot_release_before_expiry"


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _round_end_ts(round_slug: str) -> int | None:
    try:
        return int(round_slug.rsplit("-", 1)[-1]) * 1000 + 15 * 60 * 1000
    except ValueError:
        return None


def take_profit_candidate(
    *,
    policy: TakeProfitPolicy,
    side: str,
    hold_edge: float | None,
    convergence: dict[str, Any],
    seconds_to_expiry: float | None,
    hold_bid: float | None = None,
    avg_price: float | None = None,
) -> tuple[bool, str]:
    tau = policy.take_profit_hold_edge
    if side.upper() == "UP":
        tau = max(0.0, tau - policy.up_hold_edge_tighten)
    if seconds_to_expiry is not None and seconds_to_expiry <= policy.take_profit_force_exit_seconds:
        return True, _late_force_exit_reason(hold_bid=hold_bid, avg_price=avg_price)
    if hold_edge is not None and hold_edge <= tau:
        return True, "convergence_edge_captured_take_profit"
    if not convergence.get("available"):
        return False, ""
    if convergence.get("price_converged") and convergence.get("model_degraded"):
        return True, "convergence_fake_convergence_model_decay"
    residual_ratio = float(convergence.get("residual_abs_ratio") or 1.0)
    if convergence.get("price_converged") and residual_ratio <= policy.take_profit_residual_ratio:
        return True, "convergence_gap_filled_take_profit"
    entry_residual = abs(float(convergence.get("entry_residual") or 0.0))
    price_move_toward_model = float(convergence.get("price_move_toward_model") or 0.0)
    if (
        hold_edge is not None
        and entry_residual > 1e-12
        and price_move_toward_model >= policy.take_profit_price_convergence_move
        and hold_edge <= entry_residual * policy.take_profit_price_convergence_hold_edge_ratio
    ):
        return True, "convergence_price_move_take_profit"
    return False, ""


def _late_force_exit_reason(
    *,
    hold_bid: float | None,
    avg_price: float | None,
) -> str:
    if hold_bid is None or avg_price is None or avg_price <= 0:
        return V7_SLOT_RELEASE_BEFORE_EXPIRY_REASON
    if hold_bid >= avg_price:
        return V7_PROFIT_LOCK_BEFORE_EXPIRY_REASON
    return V7_LOSS_SALVAGE_BEFORE_EXPIRY_REASON


def _load_gamma(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") or payload.get("pending_settlement_rows") or []
    return {(row["round_slug"], str(row.get("side", "")).upper()): row for row in rows}


def _actual_pnl(
    pos: dict[str, Any],
    gamma_rows: dict[tuple[str, str], dict[str, Any]],
) -> tuple[float, str]:
    if pos["exit_pnl"] is not None:
        return float(pos["exit_pnl"]), "paper_exit"
    gamma = gamma_rows.get((pos["round_slug"], pos["side"]))
    if gamma:
        return float(
            gamma.get("total_position_pnl_usdc") or gamma.get("total_pnl_usdc") or 0.0
        ), "gamma_settlement"
    return float(pos.get("pm_reduce_pnl") or 0.0), "unknown"


def _simulate_position(
    *,
    pos: dict[str, Any],
    ticks: list[dict[str, Any]],
    policy: TakeProfitPolicy,
    baseline_exit_pnl: float | None,
) -> dict[str, Any]:
    tp_count = 0
    last_reason = ""
    realized_reduce = 0.0
    shares = float(pos["entry_shares"])
    avg_price = float(pos["entry_price"])

    for tick in ticks:
        if baseline_exit_pnl is not None and tick["ts"] and pos["exit_ts"] and tick["ts"] >= pos["exit_ts"]:
            break
        realized_reduce = float(tick.get("pm_reduce_pnl") or realized_reduce)
        shares = float(tick.get("shares") or shares)
        avg_price = float(tick.get("avg_price") or avg_price)
        seconds_to_expiry = None
        round_end = _round_end_ts(pos["round_slug"])
        if round_end is not None and tick["ts"] is not None:
            seconds_to_expiry = (round_end - int(tick["ts"].timestamp() * 1000)) / 1000.0
        candidate, reason = take_profit_candidate(
            policy=policy,
            side=pos["side"],
            hold_edge=tick.get("hold_edge"),
            convergence=tick.get("convergence") or {},
            seconds_to_expiry=seconds_to_expiry,
            hold_bid=tick.get("hold_bid"),
            avg_price=avg_price,
        )
        if candidate:
            tp_count += 1
            last_reason = reason
        else:
            tp_count = 0
            last_reason = ""
        if tp_count >= policy.take_profit_hysteresis_bars and tick.get("hold_bid") is not None:
            exit_bid = float(tick["hold_bid"]) * (1.0 - policy.sell_slippage)
            exit_leg = shares * (exit_bid - avg_price) if shares > 0 else 0.0
            return {
                "pnl": realized_reduce + exit_leg,
                "exit_reason": last_reason,
                "exit_ts": tick["ts"].isoformat() if tick["ts"] else None,
                "mode": "take_profit",
            }

    if baseline_exit_pnl is not None:
        return {
            "pnl": baseline_exit_pnl,
            "exit_reason": pos.get("exit_reason") or "paper_exit",
            "exit_ts": pos["exit_ts"].isoformat() if pos.get("exit_ts") else None,
            "mode": "baseline_paper_exit",
        }
    return {
        "pnl": None,
        "exit_reason": "no_take_profit_trigger",
        "exit_ts": None,
        "mode": "unsettled",
    }


def load_positions(log_path: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    positions: dict[str, dict[str, Any]] = {}
    ticks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in log_path.open(encoding="utf-8"):
        payload = json.loads(line)
        event = payload.get("event")
        if event == "paper_entry_filled":
            position = payload.get("position") or {}
            event_id = str(position.get("event_id") or "")
            if not event_id:
                continue
            positions[event_id] = {
                "event_id": event_id,
                "round_slug": str(position.get("round_slug") or ""),
                "side": str(position.get("side") or "").upper(),
                "entry_ts": _parse_ts(payload.get("ts")),
                "entry_price": float(position.get("fill_price") or position.get("entry_price") or 0.0),
                "entry_shares": float(position.get("size") or 0.0),
                "exit_pnl": None,
                "exit_ts": None,
                "exit_reason": None,
                "pm_reduce_pnl": 0.0,
            }
        elif event == "paper_exit_filled":
            position = payload.get("position") or {}
            event_id = str(position.get("event_id") or "")
            if event_id in positions:
                positions[event_id]["exit_pnl"] = float(payload.get("realized_pnl") or 0.0)
                positions[event_id]["exit_ts"] = _parse_ts(payload.get("ts"))
                positions[event_id]["exit_reason"] = str(position.get("last_lifecycle_reason") or "")
        elif event == "paper_v7_settlement_position_reduced":
            position = payload.get("position") or {}
            event_id = str(position.get("event_id") or "")
            if event_id in positions:
                positions[event_id]["pm_reduce_pnl"] = float(
                    payload.get("cumulative_position_realized_pnl") or 0.0
                )
        evaluation = payload.get("evaluation")
        if not isinstance(evaluation, dict):
            continue
        convergence = evaluation.get("convergence") or {}
        if not convergence.get("available"):
            continue
        position = payload.get("position") or {}
        event_id = str(position.get("event_id") or "")
        if event_id not in positions or positions[event_id]["exit_pnl"] is not None:
            continue
        if evaluation.get("hold_bid") is None:
            continue
        cumulative = payload.get("cumulative_position_realized_pnl")
        ticks[event_id].append(
            {
                "ts": _parse_ts(payload.get("ts")),
                "hold_bid": float(evaluation["hold_bid"]),
                "hold_edge": evaluation.get("hold_edge"),
                "shares": float(position.get("size") or 0.0),
                "avg_price": float(position.get("fill_price") or positions[event_id]["entry_price"]),
                "pm_reduce_pnl": float(cumulative)
                if cumulative is not None
                else float(positions[event_id]["pm_reduce_pnl"]),
                "convergence": convergence,
            }
        )
    ordered = []
    for event_id, pos in positions.items():
        ordered.append(pos)
        ticks[event_id].sort(key=lambda item: item["ts"] or datetime.min.replace(tzinfo=timezone.utc))
    return ordered, ticks


def replay_run(
    *,
    run_id: str,
    log_path: Path,
    gamma_path: Path,
    policy: TakeProfitPolicy,
) -> dict[str, Any]:
    gamma_rows = _load_gamma(gamma_path)
    positions, ticks = load_positions(log_path)
    rows = []
    for pos in positions:
        actual_pnl, actual_src = _actual_pnl(pos, gamma_rows)
        simulated = _simulate_position(
            pos=pos,
            ticks=ticks.get(pos["event_id"], []),
            policy=policy,
            baseline_exit_pnl=pos["exit_pnl"],
        )
        if simulated["pnl"] is None:
            simulated["pnl"] = actual_pnl
            simulated["mode"] = "fallback_actual"
        rows.append(
            {
                "run_id": run_id,
                "round_slug": pos["round_slug"],
                "side": pos["side"],
                "actual_pnl": actual_pnl,
                "actual_src": actual_src,
                "simulated_pnl": float(simulated["pnl"]),
                "delta": float(simulated["pnl"]) - actual_pnl,
                "exit_reason": simulated["exit_reason"],
                "mode": simulated["mode"],
            }
        )
    actual_sum = sum(row["actual_pnl"] for row in rows)
    simulated_sum = sum(row["simulated_pnl"] for row in rows)
    return {
        "run_id": run_id,
        "fills": len(rows),
        "actual_sum": actual_sum,
        "simulated_sum": simulated_sum,
        "delta": simulated_sum - actual_sum,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--take-profit-hold-edge", type=float, default=0.03)
    parser.add_argument("--take-profit-residual-ratio", type=float, default=0.40)
    parser.add_argument("--take-profit-price-convergence-move", type=float, default=0.10)
    parser.add_argument("--take-profit-price-convergence-hold-edge-ratio", type=float, default=0.50)
    parser.add_argument("--take-profit-force-exit-seconds", type=float, default=180.0)
    parser.add_argument("--take-profit-hysteresis-bars", type=int, default=2)
    parser.add_argument("--sell-slippage", type=float, default=0.02)
    parser.add_argument("--up-hold-edge-tighten", type=float, default=0.01)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    policy = TakeProfitPolicy(
        take_profit_hold_edge=args.take_profit_hold_edge,
        take_profit_residual_ratio=args.take_profit_residual_ratio,
        take_profit_price_convergence_move=args.take_profit_price_convergence_move,
        take_profit_price_convergence_hold_edge_ratio=(
            args.take_profit_price_convergence_hold_edge_ratio
        ),
        take_profit_force_exit_seconds=args.take_profit_force_exit_seconds,
        take_profit_hysteresis_bars=args.take_profit_hysteresis_bars,
        sell_slippage=args.sell_slippage,
        up_hold_edge_tighten=args.up_hold_edge_tighten,
    )
    runs = [
        (
            "20260608T055415Z",
            root
            / "data/logs/xgboost-v7-paper-shadow-20260608T055415Z-event5s-30round/phase4-20260608T055415Z.jsonl",
            root / "logs/xgboost-v7-paper-shadow/phase4-20260608T055415Z-gamma-reconcile.json",
            1.34,
        ),
        (
            "20260608T133724Z",
            root
            / "data/logs/xgboost-v7-paper-shadow-20260608T133721Z-event5s-30round/phase4-20260608T133724Z.jsonl",
            root / "logs/xgboost-v7-paper-shadow/phase4-20260608T133724Z-gamma-settlement-backfill.json",
            -4.1557,
        ),
        (
            "20260609T103055Z",
            root
            / "data/logs/xgboost-v7-paper-shadow-20260609T103055Z-event5s-30round/phase4-20260609T103055Z.jsonl",
            root / "logs/xgboost-v7-paper-shadow/phase4-20260609T103055Z-gamma-reconcile.json",
            -3.5348,
        ),
    ]
    summaries = []
    all_rows = []
    for run_id, log_path, gamma_path, reconciled in runs:
        result = replay_run(run_id=run_id, log_path=log_path, gamma_path=gamma_path, policy=policy)
        summaries.append(
            {
                "run_id": run_id,
                "fills": result["fills"],
                "actual_sum": round(result["actual_sum"], 4),
                "simulated_sum": round(result["simulated_sum"], 4),
                "delta": round(result["delta"], 4),
                "reconciled_pnl": reconciled,
            }
        )
        all_rows.extend(result["rows"])
    payload = {
        "policy": asdict(policy),
        "summaries": summaries,
        "all": {
            "fills": len(all_rows),
            "actual_sum": round(sum(row["actual_pnl"] for row in all_rows), 4),
            "simulated_sum": round(sum(row["simulated_pnl"] for row in all_rows), 4),
            "delta": round(
                sum(row["simulated_pnl"] for row in all_rows)
                - sum(row["actual_pnl"] for row in all_rows),
                4,
            ),
            "reconciled_sum": round(sum(item["reconciled_pnl"] for item in summaries), 4),
        },
        "rows": all_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["all"], indent=2))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
