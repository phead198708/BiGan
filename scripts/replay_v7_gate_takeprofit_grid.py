#!/usr/bin/env python3
"""Replay v7 paper logs with entry-price gate and take-profit policy grids.

Counterfactual analysis only — does not modify the live executor.  Combines:
  - optional ``min_entry_price`` hard block (or UP-only variant)
  - optional convergence take-profit simulation from PM evaluation ticks

Default grid: min_entry_price in {0.25, 0.30, 0.35} x take_profit on/off x
up_only on/off, plus baseline actual and take-profit-only rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from replay_v7_convergence_take_profit import (
    TakeProfitPolicy,
    load_positions,
    replay_run,
)


@dataclass(frozen=True, slots=True)
class RunSpec:
    run_id: str
    log_path: Path
    gamma_path: Path | None = None


@dataclass(frozen=True, slots=True)
class EntryGateSpec:
    min_entry_price: float
    up_only: bool = False

    @property
    def scenario_key(self) -> str:
        prefix = "up_low" if self.up_only else "min"
        return f"{prefix}_{self.min_entry_price:.2f}"


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    key: str
    use_take_profit: bool
    gate: EntryGateSpec | None = None

    @property
    def label(self) -> str:
        if self.gate is None:
            return "take_profit" if self.use_take_profit else "actual"
        base = self.gate.scenario_key
        return f"{base}_take_profit" if self.use_take_profit else base


def entry_blocked(*, side: str, entry_price: float, gate: EntryGateSpec) -> bool:
    if entry_price < gate.min_entry_price:
        if gate.up_only:
            return side.upper() == "UP"
        return True
    return False


def scenario_pnl(
    *,
    actual_pnl: float,
    take_profit_pnl: float,
    use_take_profit: bool,
    gate: EntryGateSpec | None,
    side: str,
    entry_price: float,
) -> float:
    if gate is not None and entry_blocked(side=side, entry_price=entry_price, gate=gate):
        return 0.0
    return take_profit_pnl if use_take_profit else actual_pnl


def build_scenario_grid(
    min_prices: list[float],
    *,
    include_baselines: bool = True,
) -> list[ScenarioSpec]:
    scenarios: list[ScenarioSpec] = []
    if include_baselines:
        scenarios.append(ScenarioSpec(key="actual", use_take_profit=False, gate=None))
        scenarios.append(ScenarioSpec(key="take_profit", use_take_profit=True, gate=None))
    for min_px in min_prices:
        for up_only in (False, True):
            gate = EntryGateSpec(min_entry_price=min_px, up_only=up_only)
            scenarios.append(ScenarioSpec(key=gate.scenario_key, use_take_profit=False, gate=gate))
            scenarios.append(
                ScenarioSpec(
                    key=f"{gate.scenario_key}_take_profit",
                    use_take_profit=True,
                    gate=gate,
                )
            )
    return scenarios


def _synthesize_run4_gamma(root: Path) -> Path | None:
    manual = root / "logs/xgboost-v7-paper-shadow/v7_run4_20260610T021333Z_pending_gamma_reconcile.json"
    if not manual.exists():
        return None
    payload = json.loads(manual.read_text(encoding="utf-8"))
    rows = []
    for rnd in payload.get("rounds", []):
        pending = rnd.get("position_at_pending") or {}
        rows.append(
            {
                "round_slug": rnd["round_slug"],
                "side": pending.get("side"),
                "total_pnl_usdc": rnd.get("pnl", {}).get("round_total_manual_usdc"),
                "total_position_pnl_usdc": rnd.get("pnl", {}).get("round_total_manual_usdc"),
            }
        )
    out = root / "logs/xgboost-v7-paper-shadow/phase4-20260610T021333Z-gamma-reconcile-synth.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    return out


def default_runs(root: Path) -> list[RunSpec]:
    run4_gamma = _synthesize_run4_gamma(root)
    return [
        RunSpec(
            run_id="20260608T055415Z",
            log_path=root
            / "data/logs/xgboost-v7-paper-shadow-20260608T055415Z-event5s-30round/phase4-20260608T055415Z.jsonl",
            gamma_path=root / "logs/xgboost-v7-paper-shadow/phase4-20260608T055415Z-gamma-reconcile.json",
        ),
        RunSpec(
            run_id="20260608T133724Z",
            log_path=root
            / "data/logs/xgboost-v7-paper-shadow-20260608T133721Z-event5s-30round/phase4-20260608T133724Z.jsonl",
            gamma_path=root
            / "logs/xgboost-v7-paper-shadow/phase4-20260608T133724Z-gamma-settlement-backfill.json",
        ),
        RunSpec(
            run_id="20260609T103055Z",
            log_path=root
            / "data/logs/xgboost-v7-paper-shadow-20260609T103055Z-event5s-30round/phase4-20260609T103055Z.jsonl",
            gamma_path=root / "logs/xgboost-v7-paper-shadow/phase4-20260609T103055Z-gamma-reconcile.json",
        ),
        RunSpec(
            run_id="20260610T021333Z",
            log_path=root
            / "data/logs/xgboost-v7-paper-shadow-20260610T021333Z-event5s-30round-takeprofit/phase4-20260610T022208Z.jsonl",
            gamma_path=run4_gamma,
        ),
    ]


def _empty_gamma_path(root: Path) -> Path:
    path = root / "logs/xgboost-v7-paper-shadow/empty-gamma.json"
    if not path.exists():
        path.write_text('{"rows": []}', encoding="utf-8")
    return path


def load_fill_rows(
    *,
    runs: list[RunSpec],
    policy: TakeProfitPolicy,
    root: Path,
) -> list[dict[str, Any]]:
    empty_gamma = _empty_gamma_path(root)
    fills: list[dict[str, Any]] = []
    for spec in runs:
        if not spec.log_path.exists():
            continue
        gamma_path = spec.gamma_path if spec.gamma_path and spec.gamma_path.exists() else empty_gamma
        replay = replay_run(
            run_id=spec.run_id,
            log_path=spec.log_path,
            gamma_path=gamma_path,
            policy=policy,
        )
        positions, _ = load_positions(spec.log_path)
        pos_by_key = {(p["round_slug"], p["side"]): p for p in positions}
        for row in replay["rows"]:
            pos = pos_by_key.get((row["round_slug"], row["side"]), {})
            entry_ts = pos.get("entry_ts")
            fills.append(
                {
                    **row,
                    "entry_price": float(pos.get("entry_price") or 0.0),
                    "entry_ts": entry_ts.isoformat() if entry_ts is not None else None,
                }
            )
    return fills


def summarize_scenario(
    fills: list[dict[str, Any]],
    scenario: ScenarioSpec,
    *,
    exclude_run_ids: set[str] | None = None,
) -> dict[str, Any]:
    subset = fills
    if exclude_run_ids:
        subset = [fill for fill in fills if fill["run_id"] not in exclude_run_ids]
    pnls = [
        scenario_pnl(
            actual_pnl=float(fill["actual_pnl"]),
            take_profit_pnl=float(fill["simulated_pnl"]),
            use_take_profit=scenario.use_take_profit,
            gate=scenario.gate,
            side=str(fill["side"]),
            entry_price=float(fill["entry_price"]),
        )
        for fill in subset
    ]
    blocked = 0
    if scenario.gate is not None:
        blocked = sum(
            1
            for fill in subset
            if entry_blocked(
                side=str(fill["side"]),
                entry_price=float(fill["entry_price"]),
                gate=scenario.gate,
            )
        )
    actual_baseline = sum(float(fill["actual_pnl"]) for fill in subset)
    return {
        "scenario": scenario.label,
        "use_take_profit": scenario.use_take_profit,
        "gate": None if scenario.gate is None else asdict(scenario.gate),
        "fills": len(subset),
        "fills_blocked": blocked,
        "fills_kept": len(subset) - blocked,
        "wins": sum(1 for pnl in pnls if pnl > 0),
        "losses": sum(1 for pnl in pnls if pnl < 0),
        "total_pnl_usdc": round(sum(pnls), 6),
        "delta_vs_actual_usdc": round(sum(pnls) - actual_baseline, 6),
    }


def summarize_per_run(
    fills: list[dict[str, Any]],
    scenarios: list[ScenarioSpec],
) -> dict[str, dict[str, float]]:
    per_run: dict[str, dict[str, float]] = {}
    for run_id in sorted({fill["run_id"] for fill in fills}):
        subset = [fill for fill in fills if fill["run_id"] == run_id]
        per_run[run_id] = {
            scenario.label: summarize_scenario(subset, scenario)["total_pnl_usdc"]
            for scenario in scenarios
        }
    return per_run


def run_grid(
    *,
    runs: list[RunSpec],
    policy: TakeProfitPolicy,
    min_prices: list[float],
    root: Path,
    exclude_run_ids: set[str] | None = None,
    include_fill_rows: bool = False,
) -> dict[str, Any]:
    fills = load_fill_rows(runs=runs, policy=policy, root=root)
    scenarios = build_scenario_grid(min_prices)
    grid = [summarize_scenario(fills, scenario, exclude_run_ids=exclude_run_ids) for scenario in scenarios]
    payload: dict[str, Any] = {
        "artifact_type": "v7_gate_takeprofit_grid_replay",
        "policy": asdict(policy),
        "min_entry_prices": min_prices,
        "runs": [{"run_id": spec.run_id, "log_path": str(spec.log_path)} for spec in runs],
        "exclude_run_ids": sorted(exclude_run_ids or []),
        "actual_baseline": summarize_scenario(fills, ScenarioSpec("actual", False, None), exclude_run_ids=exclude_run_ids),
        "grid": grid,
        "per_run": summarize_per_run(fills, scenarios),
        "notes": [
            "Entry gate is counterfactual: blocked fills contribute 0 PnL.",
            "Take-profit replay uses PM evaluation ticks (signal-driven); sparse ticks understate force_exit.",
            "Run 4 gamma may be synthesized from manual reconcile when official gamma JSON is absent.",
        ],
    }
    if include_fill_rows:
        payload["fills"] = fills
    return payload


def _parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="JSON output path")
    parser.add_argument(
        "--min-entry-prices",
        default="0.25,0.30,0.35",
        help="Comma-separated min_entry_price grid (default: 0.25,0.30,0.35)",
    )
    parser.add_argument("--exclude-run-ids", default="", help="Comma-separated run_ids to exclude from grid summary")
    parser.add_argument("--include-fill-rows", action="store_true", help="Embed per-fill replay rows in output JSON")
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
    exclude = {part.strip() for part in args.exclude_run_ids.split(",") if part.strip()}
    payload = run_grid(
        runs=default_runs(root),
        policy=policy,
        min_prices=_parse_float_list(args.min_entry_prices),
        root=root,
        exclude_run_ids=exclude or None,
        include_fill_rows=args.include_fill_rows,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"actual_baseline": payload["actual_baseline"], "top_grid": payload["grid"][:6]}, indent=2))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
