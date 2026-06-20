from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "replay_v7_gate_takeprofit_grid.py"

spec = importlib.util.spec_from_file_location("replay_v7_gate_takeprofit_grid", SCRIPT)
assert spec is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


def test_entry_blocked_all_sides_when_not_up_only() -> None:
    gate = module.EntryGateSpec(min_entry_price=0.30, up_only=False)
    assert module.entry_blocked(side="UP", entry_price=0.12, gate=gate)
    assert module.entry_blocked(side="DOWN", entry_price=0.12, gate=gate)
    assert not module.entry_blocked(side="DOWN", entry_price=0.31, gate=gate)


def test_entry_blocked_up_only_leaves_low_down() -> None:
    gate = module.EntryGateSpec(min_entry_price=0.30, up_only=True)
    assert module.entry_blocked(side="UP", entry_price=0.12, gate=gate)
    assert not module.entry_blocked(side="DOWN", entry_price=0.12, gate=gate)


def test_scenario_pnl_blocked_returns_zero() -> None:
    gate = module.EntryGateSpec(min_entry_price=0.30, up_only=False)
    assert (
        module.scenario_pnl(
            actual_pnl=-1.0,
            take_profit_pnl=0.5,
            use_take_profit=True,
            gate=gate,
            side="UP",
            entry_price=0.10,
        )
        == 0.0
    )
    assert (
        module.scenario_pnl(
            actual_pnl=-1.0,
            take_profit_pnl=0.5,
            use_take_profit=False,
            gate=gate,
            side="UP",
            entry_price=0.40,
        )
        == -1.0
    )


def test_build_scenario_grid_includes_baselines_and_combinations() -> None:
    scenarios = module.build_scenario_grid([0.30, 0.35], include_baselines=True)
    labels = {scenario.label for scenario in scenarios}
    assert "actual" in labels
    assert "take_profit" in labels
    assert "min_0.30" in labels
    assert "min_0.30_take_profit" in labels
    assert "up_low_0.30" in labels
    assert "up_low_0.30_take_profit" in labels
    assert len(scenarios) == 2 + 2 * 2 * 2
