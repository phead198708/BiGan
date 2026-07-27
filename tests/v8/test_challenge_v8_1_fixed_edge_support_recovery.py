from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.challenge_historical_development import SAFE_FALSES
from bigan.v8.polymarket.challenge_v8_1_fixed_edge_support_recovery import (
    CANDIDATE_ID,
    ChallengeFixedEdgeSupportRecoveryError,
    build_fixed_edge_support_recovery_comparison,
    materialize_fixed_edge_support_recovery_decisions,
    validate_fixed_edge_support_recovery_profile,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = (
    ROOT
    / "examples/v8/polymarket_configs"
    / "challenge_v8_1_fixed_edge_support_recovery_0_025_profile.json"
)


def _profile() -> dict:
    payload = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _guard(
    market_id: str,
    *,
    score: float,
    baseline_action: str = "BUY_DOWN_SELL_BEFORE_CLOSE",
    baseline_side: str = "DOWN",
) -> dict:
    return {
        "candidate_name": "adaptive_support_controller_v8_1",
        "market_id": market_id,
        "decision_ts": 1_000,
        "market_close_ts": 241_000,
        "max_input_ts": 999,
        "baseline_action": baseline_action,
        "v6_7_baseline_action": baseline_action,
        "baseline_side": baseline_side,
        "baseline_decision_ts": 1_000,
        "baseline_max_input_ts": 999,
        "selected_action": "NO_TRADE",
        "selected_side": "NONE",
        "execution_guard_order_allowed": False,
        "point_selected_predicted_return": score,
        "fixed_edge_buffer": 0.025,
        "target_used_as_decision_time_input": False,
        "outcome_or_pnl_field_used_at_inference": False,
        "labels_outcomes_or_pnl_opened": False,
        "source_score_mutated": False,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
        "promotion_evidence_eligible": False,
        "live_trading_enabled": False,
        "guard_replay_row_id": f"guard-{market_id}",
        "controller_state_before_id": f"before-{market_id}",
        "controller_state_after_id": f"after-{market_id}",
    }


def test_profile_is_hash_pinned_and_semantically_exact() -> None:
    expected = PROFILE_PATH.with_suffix(".sha256").read_text().strip()
    assert hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest() == expected
    validate_fixed_edge_support_recovery_profile(_profile())

    tampered = copy.deepcopy(_profile())
    tampered["policy"]["fixed_edge_threshold_inclusive"] = 0.02
    with pytest.raises(
        ChallengeFixedEdgeSupportRecoveryError,
        match="policy",
    ):
        validate_fixed_edge_support_recovery_profile(tampered)


def test_fixed_edge_boundary_selects_only_at_or_above_threshold() -> None:
    market_ids = ["below", "boundary", "above"]
    decisions = materialize_fixed_edge_support_recovery_decisions(
        base_guard_rows=[
            _guard("below", score=0.024999),
            _guard("boundary", score=0.025),
            _guard(
                "above",
                score=0.08,
                baseline_action="BUY_UP_SELL_BEFORE_CLOSE",
                baseline_side="UP",
            ),
        ],
        frozen_market_ids=market_ids,
        profile=_profile(),
    )

    assert [row["selected_action"] for row in decisions] == [
        "NO_TRADE",
        "BUY_DOWN_SELL_BEFORE_CLOSE",
        "BUY_UP_SELL_BEFORE_CLOSE",
    ]
    assert [row["fixed_edge_threshold_passed"] for row in decisions] == [
        False,
        True,
        True,
    ]
    assert all(row["safety"] == SAFE_FALSES for row in decisions)
    assert all(
        row["outcome_or_pnl_field_used_at_inference"] is False
        for row in decisions
    )


def test_comparison_declares_sizes_and_reconciles_unit_pnl() -> None:
    market_ids = ["veto", "trade"]
    decisions = materialize_fixed_edge_support_recovery_decisions(
        base_guard_rows=[
            _guard("veto", score=-0.1),
            _guard("trade", score=0.1),
        ],
        frozen_market_ids=market_ids,
        profile=_profile(),
    )
    comparison = build_fixed_edge_support_recovery_comparison(
        candidate_decisions=decisions,
        base_comparison_rows=[
            {
                "market_id": "veto",
                "v6_7_action": "BUY_DOWN_SELL_BEFORE_CLOSE",
                "v6_7_after_cost_pnl": -0.04,
            },
            {
                "market_id": "trade",
                "v6_7_action": "BUY_DOWN_SELL_BEFORE_CLOSE",
                "v6_7_after_cost_pnl": 0.02,
            },
        ],
        frozen_market_ids=market_ids,
    )

    assert [row["candidate_after_cost_pnl"] for row in comparison] == [
        0.0,
        0.02,
    ]
    assert [
        row["candidate_unit_after_cost_pnl"] for row in comparison
    ] == pytest.approx([
        0.0,
        0.1,
    ])
    assert [
        row["baseline_unit_after_cost_pnl"] for row in comparison
    ] == pytest.approx([
        -0.2,
        0.1,
    ])
    assert all(
        row["candidate_declared_position_size"] == 0.2
        and row["baseline_declared_position_size"] == 0.2
        for row in comparison
    )
    assert all(row["candidate_id"] == CANDIDATE_ID for row in comparison)


def test_nonfinite_score_vetoes_and_frozen_order_fails_closed() -> None:
    decisions = materialize_fixed_edge_support_recovery_decisions(
        base_guard_rows=[_guard("market", score=float("nan"))],
        frozen_market_ids=["market"],
        profile=_profile(),
    )
    assert decisions[0]["selected_action"] == "NO_TRADE"
    assert decisions[0]["point_selected_predicted_return"] is None
    assert decisions[0]["selection_reason"] == (
        "point_selected_predicted_return_missing_or_nonfinite"
    )

    with pytest.raises(
        ChallengeFixedEdgeSupportRecoveryError,
        match="frozen chronological",
    ):
        materialize_fixed_edge_support_recovery_decisions(
            base_guard_rows=[
                _guard("first", score=0.1),
                _guard("second", score=0.1),
            ],
            frozen_market_ids=["second", "first"],
            profile=_profile(),
        )
