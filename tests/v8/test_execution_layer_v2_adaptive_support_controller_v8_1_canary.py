from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from bigan.v8.polymarket.training import (
    execution_layer_v2_adaptive_support_controller_v8_1_canary as canary,
)

ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_adaptive_support_controller_v8_1_canary_plan.json"
)


def _plan() -> dict[str, Any]:
    return json.loads(PLAN_PATH.read_text())


def test_canary_plan_freezes_strictly_later_outcome_blind_gate() -> None:
    plan = _plan()
    canary.validate_v8_1_canary_plan(
        plan,
        historical_manifest_sha256=plan["historical_gate"][
            "historical_manifest_sha256"
        ],
        model_sha256=plan["historical_gate"]["model_sha256"],
        profile_sha256=plan["historical_gate"]["profile_sha256"],
    )
    assert plan["collection"]["target_quality_valid_market_count"] == 12
    assert plan["collection"]["maximum_attempted_market_count"] == 18
    assert plan["target_free_actionability_gate"][
        "minimum_guard_accepted_market_count"
    ] == 4
    assert plan["target_free_actionability_gate"]["no_side_quota"] is True
    assert plan["future_unseen_holdout"]["exact_quality_valid_market_count"] == 120

    changed = copy.deepcopy(plan)
    changed["target_free_actionability_gate"][
        "minimum_guard_accepted_market_count"
    ] = 3
    with pytest.raises(ValueError, match="gate"):
        canary.validate_v8_1_canary_plan(
            changed,
            historical_manifest_sha256=plan["historical_gate"][
                "historical_manifest_sha256"
            ],
            model_sha256=plan["historical_gate"]["model_sha256"],
            profile_sha256=plan["historical_gate"]["profile_sha256"],
        )


def test_canary_window_advances_controller_only_after_full_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_histories: list[list[bool]] = []

    def fake_score(
        market: dict[str, Any],
        *,
        model_artifact: dict[str, Any],
        prior_rank_state: dict[str, Any],
    ) -> dict[str, Any]:
        del model_artifact
        observed_histories.append(
            list(prior_rank_state["controller_guard_acceptance_history"])
        )
        decision = {
            "market_id": market["market_id"],
            "selected_policy_decision": "KEEP_V6_7",
            "selected_action": "BUY_UP_SELL_BEFORE_CLOSE",
            "selected_side": "UP",
            "rank_controller_decision": {
                "controller_band": "low_support_q25",
            },
            "next_rank_state": copy.deepcopy(prior_rank_state),
            "current_guard_result_used_for_own_controller_decision": False,
            "source_score_mutated": False,
            "outcome_or_pnl_field_used_at_inference": False,
            "paper_only": True,
            "capital_at_risk": False,
        }
        decision["decision_id"] = hashlib.sha256(
            market["market_id"].encode()
        ).hexdigest()
        return decision

    monkeypatch.setattr(
        canary.v81, "score_adaptive_support_controller_v8_1_market", fake_score
    )
    monkeypatch.setattr(
        canary, "_microstructure_blocking_reasons", lambda source, guard: []
    )
    selected = ["m1", "m2"]
    canonical: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    baselines: list[dict[str, Any]] = []
    for index, market_id in enumerate(selected):
        decision_ts = 1000 + index
        for action in (
            "BUY_UP_SELL_BEFORE_CLOSE",
            "BUY_DOWN_SELL_BEFORE_CLOSE",
        ):
            row = {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "market_close_ts": 2000 + index,
                "action": action,
            }
            canonical.append(dict(row))
            actions.append(dict(row))
        baselines.append(
            {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "action": "BUY_UP_SELL_BEFORE_CLOSE",
            }
        )
    initial_state = {
        "rank_state_id": "a" * 64,
        "controller_guard_acceptance_history": [False] * 20,
        "paper_only": True,
        "capital_at_risk": False,
    }
    decisions, guard_rows, final_state = canary._score_window(
        selected,
        canonical_rows=canonical,
        baseline_rows=baselines,
        action_rows=actions,
        model={"final_rank_state": initial_state},
        v6_7_profile={"hard_execution_safety": {}},
    )
    assert len(decisions) == 2
    assert all(row["execution_guard_order_allowed"] for row in guard_rows)
    assert observed_histories[0] == [False] * 20
    assert observed_histories[1][-1] is True
    assert final_state["controller_guard_acceptance_history"][-2:] == [True, True]
    assert all(
        row["current_guard_result_used_for_own_controller_decision"] is False
        and row["current_guard_result_added_after_decision_freeze"] is True
        for row in guard_rows
    )
    assert final_state["capital_at_risk"] is False


def test_canary_plan_is_hashable_and_outcome_free() -> None:
    payload = PLAN_PATH.read_bytes()
    assert len(hashlib.sha256(payload).hexdigest()) == 64
    text = payload.decode()
    assert '"outcomes_resolution_labels_or_pnl_opened": false' in text
    assert '"paper_candidate_allowed": false' in text
    assert '"live_trading_enabled": false' in text
