from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "replay_v7_convergence_take_profit.py"

spec = importlib.util.spec_from_file_location("replay_v7_convergence_take_profit", SCRIPT)
assert spec is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


def test_take_profit_candidate_edge_captured() -> None:
    policy = module.TakeProfitPolicy()
    candidate, reason = module.take_profit_candidate(
        policy=policy,
        side="DOWN",
        hold_edge=0.02,
        convergence={"available": True, "price_converged": False, "residual_abs_ratio": 0.8},
        seconds_to_expiry=600.0,
    )
    assert candidate is True
    assert reason == "convergence_edge_captured_take_profit"


def test_take_profit_candidate_force_exit_before_expiry() -> None:
    policy = module.TakeProfitPolicy()
    candidate, reason = module.take_profit_candidate(
        policy=policy,
        side="DOWN",
        hold_edge=0.20,
        convergence={"available": True, "price_converged": False, "residual_abs_ratio": 0.9},
        seconds_to_expiry=120.0,
    )
    assert candidate is True
    assert reason == "convergence_slot_release_before_expiry"


def test_take_profit_candidate_force_exit_profit_lock_before_expiry() -> None:
    policy = module.TakeProfitPolicy()
    candidate, reason = module.take_profit_candidate(
        policy=policy,
        side="DOWN",
        hold_edge=0.20,
        convergence={"available": True, "price_converged": False, "residual_abs_ratio": 0.9},
        seconds_to_expiry=120.0,
        hold_bid=0.72,
        avg_price=0.50,
    )
    assert candidate is True
    assert reason == "convergence_profit_lock_before_expiry"


def test_take_profit_candidate_up_is_tighter() -> None:
    policy = module.TakeProfitPolicy(take_profit_hold_edge=0.03, up_hold_edge_tighten=0.01)
    down_candidate, _ = module.take_profit_candidate(
        policy=policy,
        side="DOWN",
        hold_edge=0.025,
        convergence={"available": True},
        seconds_to_expiry=600.0,
    )
    up_candidate, _ = module.take_profit_candidate(
        policy=policy,
        side="UP",
        hold_edge=0.025,
        convergence={"available": True},
        seconds_to_expiry=600.0,
    )
    assert down_candidate is True
    assert up_candidate is False


def test_take_profit_candidate_price_convergence_move() -> None:
    policy = module.TakeProfitPolicy(
        take_profit_residual_ratio=0.40,
        take_profit_price_convergence_move=0.10,
        take_profit_price_convergence_hold_edge_ratio=0.50,
    )
    candidate, reason = module.take_profit_candidate(
        policy=policy,
        side="DOWN",
        hold_edge=0.14,
        convergence={
            "available": True,
            "entry_residual": 0.30,
            "price_converged": True,
            "price_move_toward_model": 0.12,
            "residual_abs_ratio": 0.60,
            "model_degraded": False,
        },
        seconds_to_expiry=600.0,
    )

    assert candidate is True
    assert reason == "convergence_price_move_take_profit"
