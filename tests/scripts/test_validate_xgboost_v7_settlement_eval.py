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
            residual_p_side=0.72,
            residual_expected_edge=0.20,
            market_implied_prob_side=0.50,
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
            residual_p_side=0.90,
            residual_expected_edge=0.48,
            market_implied_prob_side=0.40,
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


def test_residual_policy_uses_residual_edge_signal() -> None:
    candidate = module.SettlementCandidate(
        split="test",
        feature_ts=1,
        round_slug="round-residual",
        side="DOWN",
        true_label="DOWN",
        p_side=0.40,
        p_up=0.55,
        p_down=0.40,
        p_neutral=0.05,
        residual_p_side=0.82,
        residual_expected_edge=0.21,
        market_implied_prob_side=0.35,
        entry_ask_price=0.35,
        entry_worst_price=0.37,
        expected_edge=0.03,
        realized_pnl=0.63,
        realized_edge=0.63,
        seconds_to_expiry=600.0,
    )

    probability_result = module._evaluate_policy(
        [candidate],
        module.PolicySpec(name="prob", min_confidence=0.75, min_expected_edge=0.04),
    )
    residual_result = module._evaluate_policy(
        [candidate],
        module.PolicySpec(
            name="resid",
            min_confidence=0.75,
            min_expected_edge=0.04,
            signal_source="residual",
        ),
    )

    assert probability_result["summary"]["trade_count"] == 0
    assert residual_result["summary"]["trade_count"] == 1
    assert residual_result["summary"]["pnl"] == pytest.approx(0.63)


def test_pnl_stability_selector_uses_train_and_validation() -> None:
    def candidate(round_slug: str, pnl: float, *, p_side: float = 0.80) -> module.SettlementCandidate:
        win = pnl > 0
        return module.SettlementCandidate(
            split="",
            feature_ts=1,
            round_slug=round_slug,
            side="UP",
            true_label="UP" if win else "DOWN",
            p_side=p_side,
            p_up=p_side,
            p_down=1.0 - p_side,
            p_neutral=0.0,
            residual_p_side=None,
            residual_expected_edge=None,
            market_implied_prob_side=0.50,
            entry_ask_price=0.50,
            entry_worst_price=0.50 if win else -pnl,
            expected_edge=p_side - (0.50 if win else -pnl),
            realized_pnl=pnl,
            realized_edge=pnl,
            seconds_to_expiry=600.0,
        )

    conservative = module.PolicySpec("conservative", min_confidence=0.75, min_expected_edge=0.04)
    loose = module.PolicySpec("loose", min_confidence=0.70, min_expected_edge=0.04)
    candidates_by_split = {
        "train": [
            candidate("train-a", 0.50, p_side=0.80),
            candidate("train-b", 0.50, p_side=0.80),
            candidate("train-c", -0.50, p_side=0.72),
        ],
        "val": [
            candidate("val-a", 0.50, p_side=0.80),
            candidate("val-b", 0.50, p_side=0.80),
            candidate("val-c", 0.50, p_side=0.72),
        ],
    }

    selected = module._select_best_policy_by_pnl_stability(
        [loose, conservative],
        candidates_by_split,
        selection_splits=["train", "val"],
        min_trades=2,
        min_avg_pnl=0.2,
    )

    assert selected is not None
    assert selected["policy"]["name"] == "conservative"
    assert selected["selection"]["preferred"] is True
