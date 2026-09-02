from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.challenge_historical_development import SAFE_FALSES
from bigan.v8.polymarket.challenge_v8_1_fixed_edge_price_floor import (
    CANDIDATE_ID,
    ChallengeFixedEdgePriceFloorError,
    build_fixed_edge_price_floor_comparison,
    materialize_fixed_edge_price_floor_decisions,
    validate_fixed_edge_price_floor_profile,
)
from bigan.v8.polymarket.contracts import canonical_json_sha256

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "examples/v8/polymarket_configs"
PROFILE_PATH = (
    CONFIG_DIR / "challenge_v8_1_fixed_edge_0_025_price_floor_0_30_profile.json"
)
FIXED_EDGE_PROFILE_PATH = (
    CONFIG_DIR
    / "challenge_v8_1_fixed_edge_support_recovery_0_025_profile.json"
)


def _json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _guard(market_id: str, *, score: float) -> dict:
    return {
        "candidate_name": "adaptive_support_controller_v8_1",
        "market_id": market_id,
        "decision_ts": 1_000,
        "market_close_ts": 241_000,
        "max_input_ts": 999,
        "baseline_action": "BUY_DOWN_SELL_BEFORE_CLOSE",
        "v6_7_baseline_action": "BUY_DOWN_SELL_BEFORE_CLOSE",
        "baseline_side": "DOWN",
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


def _action_row(market_id: str, *, price: float) -> dict:
    row = {
        "market_id": market_id,
        "decision_ts": 1_000,
        "max_input_ts": 999,
        "action": "BUY_DOWN_SELL_BEFORE_CLOSE",
        "decision_time_features": {"execution_price": price},
        "microstructure_snapshot": {"entry_ask": price},
        "target_used_as_decision_input": False,
        "outcome_fields_used_as_decision_input": False,
        "paper_only": True,
        "capital_at_risk": False,
    }
    row["action_row_sha256"] = canonical_json_sha256(row)
    return row


def test_profile_is_hash_pinned_and_semantically_exact() -> None:
    expected = PROFILE_PATH.with_suffix(".sha256").read_text().strip()
    assert hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest() == expected
    validate_fixed_edge_price_floor_profile(_json(PROFILE_PATH))

    tampered = copy.deepcopy(_json(PROFILE_PATH))
    tampered["policy"]["entry_price_floor_inclusive"] = 0.29
    with pytest.raises(ChallengeFixedEdgePriceFloorError, match="policy"):
        validate_fixed_edge_price_floor_profile(tampered)


def test_both_preregistered_thresholds_are_required() -> None:
    market_ids = ["score-veto", "price-veto", "pass"]
    decisions = materialize_fixed_edge_price_floor_decisions(
        base_guard_rows=[
            _guard("score-veto", score=0.02),
            _guard("price-veto", score=0.03),
            _guard("pass", score=0.03),
        ],
        five_action_rows=[
            _action_row("score-veto", price=0.50),
            _action_row("price-veto", price=0.2999),
            _action_row("pass", price=0.30),
        ],
        frozen_market_ids=market_ids,
        fixed_edge_profile=_json(FIXED_EDGE_PROFILE_PATH),
        profile=_json(PROFILE_PATH),
    )

    assert [row["selected_action"] for row in decisions] == [
        "NO_TRADE",
        "NO_TRADE",
        "BUY_DOWN_SELL_BEFORE_CLOSE",
    ]
    assert decisions[0]["entry_price"] is None
    assert decisions[1]["entry_price_filter_passed"] is False
    assert decisions[2]["entry_price_filter_passed"] is True
    assert all(row["safety"] == SAFE_FALSES for row in decisions)


def test_comparison_is_unit_sizing_ready() -> None:
    decisions = materialize_fixed_edge_price_floor_decisions(
        base_guard_rows=[_guard("trade", score=0.04)],
        five_action_rows=[_action_row("trade", price=0.40)],
        frozen_market_ids=["trade"],
        fixed_edge_profile=_json(FIXED_EDGE_PROFILE_PATH),
        profile=_json(PROFILE_PATH),
    )
    rows = build_fixed_edge_price_floor_comparison(
        candidate_decisions=decisions,
        base_comparison_rows=[
            {
                "market_id": "trade",
                "v6_7_action": "BUY_DOWN_SELL_BEFORE_CLOSE",
                "v6_7_after_cost_pnl": 0.02,
            }
        ],
        frozen_market_ids=["trade"],
    )

    assert rows[0]["candidate_id"] == CANDIDATE_ID
    assert rows[0]["candidate_declared_position_size"] == 0.2
    assert rows[0]["baseline_declared_position_size"] == 0.2
    assert rows[0]["candidate_unit_after_cost_pnl"] == pytest.approx(0.1)
    assert rows[0]["baseline_unit_after_cost_pnl"] == pytest.approx(0.1)


def test_action_row_hash_drift_fails_closed() -> None:
    action = _action_row("market", price=0.40)
    action["decision_time_features"]["execution_price"] = 0.41
    with pytest.raises(
        ValueError,
        match="matched decision-time action row",
    ):
        materialize_fixed_edge_price_floor_decisions(
            base_guard_rows=[_guard("market", score=0.04)],
            five_action_rows=[action],
            frozen_market_ids=["market"],
            fixed_edge_profile=_json(FIXED_EDGE_PROFILE_PATH),
            profile=_json(PROFILE_PATH),
        )
