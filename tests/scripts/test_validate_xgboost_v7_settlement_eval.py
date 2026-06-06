from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "validate_xgboost_v7_settlement_eval.py"

spec = importlib.util.spec_from_file_location("validate_xgboost_v7_settlement_eval", SCRIPT)
assert spec is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


def test_one_way_settlement_pnl_does_not_subtract_round_trip_cost() -> None:
    assert module._one_way_settlement_pnl(
        true_label="UP",
        side="UP",
        entry_worst_price=0.42,
    ) == pytest.approx(0.58)
    assert module._one_way_settlement_pnl(
        true_label="DOWN",
        side="UP",
        entry_worst_price=0.42,
    ) == pytest.approx(-0.42)


def test_entry_worst_price_includes_slippage_and_fee_and_caps() -> None:
    assert module._entry_worst_price(0.50, buy_slippage=0.02, fee_bps=100.0) == 0.525
    assert module._entry_worst_price(0.98, buy_slippage=0.02, fee_bps=0.0) == 0.99


def test_policy_selects_first_passing_signal_per_round() -> None:
    candidates = [
        module.SettlementCandidate(
            split="val",
            feature_ts=1,
            round_slug="round-a",
            side="UP",
            true_label="UP",
            p_side=0.70,
            p_up=0.70,
            p_down=0.20,
            p_neutral=0.10,
            entry_ask_price=0.50,
            entry_worst_price=0.52,
            expected_edge=0.18,
            realized_pnl=0.48,
            realized_edge=0.48,
            seconds_to_expiry=700.0,
        ),
        module.SettlementCandidate(
            split="val",
            feature_ts=2,
            round_slug="round-a",
            side="UP",
            true_label="UP",
            p_side=0.90,
            p_up=0.90,
            p_down=0.05,
            p_neutral=0.05,
            entry_ask_price=0.40,
            entry_worst_price=0.42,
            expected_edge=0.48,
            realized_pnl=0.58,
            realized_edge=0.58,
            seconds_to_expiry=640.0,
        ),
    ]

    result = module._evaluate_policy(
        candidates,
        module.PolicySpec(name="test", min_confidence=0.60, min_expected_edge=0.10),
    )

    assert result["summary"]["trade_count"] == 1
    assert result["summary"]["pnl"] == 0.48
    assert result["reject_counts"]["round_first_blocked"] == 1


def test_best_policy_is_selected_from_validation_pnl() -> None:
    validation_results = [
        {
            "policy": {"name": "a", "min_confidence": 0.5, "min_expected_edge": 0.0},
            "summary": {"trade_count": 5, "pnl": -1.0, "avg_pnl": -0.2},
        },
        {
            "policy": {"name": "b", "min_confidence": 0.8, "min_expected_edge": 0.1},
            "summary": {"trade_count": 5, "pnl": 1.0, "avg_pnl": 0.2},
        },
    ]

    best = module._select_best_policy(validation_results, min_trades=5)

    assert best is not None
    assert best["policy"]["name"] == "b"
