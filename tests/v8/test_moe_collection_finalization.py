from __future__ import annotations

from bigan.v8.polymarket.moe_collection_boundary_r2 import AuthorizationExpectation
from bigan.v8.polymarket.moe_collection_finalization import (
    _baseline_decision_row,
    _candidate_decision_row,
)


def _observation() -> dict:
    common = {
        "requested_route": "bearish",
        "actual_model_used": "moe_expert_bearish",
        "expert_id": "moe_expert_bearish",
        "expert_training_support": 57,
        "expert_available": True,
        "fallback_used": False,
    }
    return {
        "market_id": "market-1",
        "attempt_index": 9,
        "market_start_ts": 1_000,
        "decision_rows": [
            {
                **common,
                "decision_ts": 1_300,
                "selected_side": None,
                "accepted": False,
                "baseline_selected_side": "UP",
                "baseline_accepted": True,
            },
            {
                **common,
                "decision_ts": 1_600,
                "selected_side": "DOWN",
                "accepted": True,
                "baseline_selected_side": None,
                "baseline_accepted": False,
            },
        ],
    }


def test_candidate_and_baseline_freeze_independent_actions() -> None:
    authorization = AuthorizationExpectation(
        artifact_path=None,  # type: ignore[arg-type]
        expected_artifact_sha256="0" * 64,
        authorization_request_text_sha256="1" * 64,
        authorization_decision_text_sha256="2" * 64,
        approver_identity="test",
        authorization_source_id="test",
        authorization_source_url="https://example.invalid",
        authorization_timestamp="2030-01-01T00:00:00Z",
        strictly_later_than_timestamp="2030-01-01T00:00:00Z",
        expected_frozen_artifact_sha256={
            "matched_baseline_artifact": "3" * 64
        },
    )
    candidate = _candidate_decision_row(_observation())
    baseline = _baseline_decision_row(_observation(), authorization)

    assert candidate["decision"] == "TRADE"
    assert candidate["selected_side"] == "DOWN"
    assert candidate["decision_ts"] == 1_600
    assert baseline["decision"] == "TRADE"
    assert baseline["selected_side"] == "UP"
    assert baseline["decision_ts"] == 1_300
    assert candidate["market_id"] == baseline["market_id"]
    assert candidate["outcomes_accessed"] is False
    assert baseline["settlement_accessed"] is False
