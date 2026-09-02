from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.challenge_historical_development import SAFE_FALSES
from bigan.v8.polymarket.challenge_v8_1_entry_price_floor_sizing import (
    CANDIDATE_ID,
    ChallengeEntryPriceFloorSizingError,
    build_entry_price_floor_sizing_comparison,
    materialize_entry_price_floor_sizing_decisions,
    validate_entry_price_floor_sizing_profile,
)
from bigan.v8.polymarket.contracts import canonical_json_sha256

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "examples/v8/polymarket_configs"
PROFILE_PATH = (
    CONFIG_DIR
    / "challenge_v8_1_entry_price_floor_0_30_sized_1_0_profile.json"
)
ENTRY_PRICE_FLOOR_PROFILE_PATH = (
    CONFIG_DIR / "challenge_v8_1_entry_price_floor_0_30_profile.json"
)


def _json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _guard(
    market_id: str,
    *,
    action: str,
    side: str,
    decision_ts: int,
) -> dict:
    return {
        "candidate_name": "adaptive_support_controller_v8_1",
        "market_id": market_id,
        "decision_ts": decision_ts,
        "market_close_ts": decision_ts + 240_000,
        "max_input_ts": decision_ts - 1,
        "selected_action": action,
        "selected_side": side,
        "execution_guard_order_allowed": action != "NO_TRADE",
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


def _action_row(
    market_id: str,
    *,
    action: str,
    decision_ts: int,
    entry_price: float,
) -> dict:
    row = {
        "market_id": market_id,
        "decision_ts": decision_ts,
        "max_input_ts": decision_ts - 1,
        "action": action,
        "decision_time_features": {"execution_price": entry_price},
        "microstructure_snapshot": {"entry_ask": entry_price},
        "target_used_as_decision_input": False,
        "outcome_fields_used_as_decision_input": False,
        "paper_only": True,
        "capital_at_risk": False,
    }
    row["action_row_sha256"] = canonical_json_sha256(row)
    return row


def _target(
    market_id: str,
    *,
    action: str,
    side: str,
    per_contract_pnl: float,
) -> dict:
    return {
        "schema_version": (
            "bigan-v8-runtime-aligned-sbc-net-return-v6-4-target-row-v1"
        ),
        "market_id": market_id,
        "action": action,
        "side": side,
        "runtime_policy_after_cost_net_pnl_at_frozen_size": (
            per_contract_pnl * 0.2
        ),
        "runtime_policy_after_cost_net_pnl_per_contract": per_contract_pnl,
        "target_row_id": f"target-{market_id}",
        "target_used_as_decision_time_input": False,
        "target_available_only_post_exit_or_official_resolution": True,
        "paper_position_size": 0.2,
        "cost_fields_subtracted_exactly_once": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
        "promotion_evidence_eligible": False,
    }


def test_sizing_profile_and_sidecar_are_exactly_pinned() -> None:
    expected = PROFILE_PATH.with_suffix(".sha256").read_text().strip()
    assert hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest() == expected
    validate_entry_price_floor_sizing_profile(_json(PROFILE_PATH))

    tampered = copy.deepcopy(_json(PROFILE_PATH))
    tampered["policy"]["fixed_candidate_position_size"] = 0.9
    with pytest.raises(
        ChallengeEntryPriceFloorSizingError,
        match="policy",
    ):
        validate_entry_price_floor_sizing_profile(tampered)


def test_target_free_selection_is_unchanged_and_size_is_frozen() -> None:
    market_ids = ["low", "high"]
    decisions = materialize_entry_price_floor_sizing_decisions(
        base_guard_rows=[
            _guard(
                "low",
                action="BUY_DOWN_SELL_BEFORE_CLOSE",
                side="DOWN",
                decision_ts=1_000,
            ),
            _guard(
                "high",
                action="BUY_UP_SELL_BEFORE_CLOSE",
                side="UP",
                decision_ts=2_000,
            ),
        ],
        five_action_rows=[
            _action_row(
                "low",
                action="BUY_DOWN_SELL_BEFORE_CLOSE",
                decision_ts=1_000,
                entry_price=0.20,
            ),
            _action_row(
                "high",
                action="BUY_UP_SELL_BEFORE_CLOSE",
                decision_ts=2_000,
                entry_price=0.40,
            ),
        ],
        frozen_market_ids=market_ids,
        profile=_json(PROFILE_PATH),
        entry_price_floor_profile=_json(ENTRY_PRICE_FLOOR_PROFILE_PATH),
    )

    assert [row["selected_action"] for row in decisions] == [
        "NO_TRADE",
        "BUY_UP_SELL_BEFORE_CLOSE",
    ]
    assert [row["candidate_position_size"] for row in decisions] == [0.0, 1.0]
    assert all(row["selected_trade_set_changed"] is False for row in decisions)
    assert all(row["position_sizing_changed"] is True for row in decisions)
    assert all(row["target_used_as_decision_time_input"] is False for row in decisions)
    assert all(row["safety"] == SAFE_FALSES for row in decisions)


def test_comparison_scales_selected_trade_only_and_keeps_baseline_frozen() -> None:
    market_ids = ["low", "high"]
    decisions = materialize_entry_price_floor_sizing_decisions(
        base_guard_rows=[
            _guard(
                "low",
                action="BUY_DOWN_SELL_BEFORE_CLOSE",
                side="DOWN",
                decision_ts=1_000,
            ),
            _guard(
                "high",
                action="BUY_UP_SELL_BEFORE_CLOSE",
                side="UP",
                decision_ts=2_000,
            ),
        ],
        five_action_rows=[
            _action_row(
                "low",
                action="BUY_DOWN_SELL_BEFORE_CLOSE",
                decision_ts=1_000,
                entry_price=0.20,
            ),
            _action_row(
                "high",
                action="BUY_UP_SELL_BEFORE_CLOSE",
                decision_ts=2_000,
                entry_price=0.40,
            ),
        ],
        frozen_market_ids=market_ids,
        profile=_json(PROFILE_PATH),
        entry_price_floor_profile=_json(ENTRY_PRICE_FLOOR_PROFILE_PATH),
    )
    targets = [
        _target(
            "low",
            action="BUY_DOWN_SELL_BEFORE_CLOSE",
            side="DOWN",
            per_contract_pnl=-0.25,
        ),
        _target(
            "high",
            action="BUY_UP_SELL_BEFORE_CLOSE",
            side="UP",
            per_contract_pnl=0.40,
        ),
    ]
    rows = build_entry_price_floor_sizing_comparison(
        candidate_decisions=decisions,
        base_comparison_rows=[
            {
                "market_id": "low",
                "challenge_after_cost_pnl": -0.05,
                "v6_7_action": "BUY_DOWN_SELL_BEFORE_CLOSE",
                "v6_7_after_cost_pnl": -0.05,
            },
            {
                "market_id": "high",
                "challenge_after_cost_pnl": 0.08,
                "v6_7_action": "BUY_UP_SELL_BEFORE_CLOSE",
                "v6_7_after_cost_pnl": 0.08,
            },
        ],
        base_runtime_targets=targets,
        frozen_market_ids=market_ids,
    )

    assert [row["candidate_after_cost_pnl"] for row in rows] == [0.0, 0.40]
    assert [row["baseline_after_cost_pnl"] for row in rows] == [-0.05, 0.08]
    assert [row["candidate_minus_baseline_pnl"] for row in rows] == [
        0.05,
        0.32,
    ]
    assert all(row["candidate_id"] == CANDIDATE_ID for row in rows)
    assert all(row["candidate_fixed_position_size"] == 1.0 for row in rows)


def test_non_linear_runtime_target_fails_closed() -> None:
    decision = materialize_entry_price_floor_sizing_decisions(
        base_guard_rows=[
            _guard(
                "high",
                action="BUY_UP_SELL_BEFORE_CLOSE",
                side="UP",
                decision_ts=2_000,
            )
        ],
        five_action_rows=[
            _action_row(
                "high",
                action="BUY_UP_SELL_BEFORE_CLOSE",
                decision_ts=2_000,
                entry_price=0.40,
            )
        ],
        frozen_market_ids=["high"],
        profile=_json(PROFILE_PATH),
        entry_price_floor_profile=_json(ENTRY_PRICE_FLOOR_PROFILE_PATH),
    )
    target = _target(
        "high",
        action="BUY_UP_SELL_BEFORE_CLOSE",
        side="UP",
        per_contract_pnl=0.40,
    )
    target["runtime_policy_after_cost_net_pnl_at_frozen_size"] = 0.09
    with pytest.raises(
        ChallengeEntryPriceFloorSizingError,
        match="linear",
    ):
        build_entry_price_floor_sizing_comparison(
            candidate_decisions=decision,
            base_comparison_rows=[
                {
                    "market_id": "high",
                    "challenge_after_cost_pnl": 0.09,
                    "v6_7_action": "BUY_UP_SELL_BEFORE_CLOSE",
                    "v6_7_after_cost_pnl": 0.08,
                }
            ],
            base_runtime_targets=[target],
            frozen_market_ids=["high"],
        )
