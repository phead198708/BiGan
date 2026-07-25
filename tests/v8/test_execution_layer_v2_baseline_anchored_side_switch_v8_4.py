from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from bigan.v8.polymarket.training import (
    execution_layer_v2_baseline_anchored_side_switch_v8_4 as v84,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = (
    ROOT
    / "examples"
    / "v8"
    / "polymarket_configs"
    / "execution_layer_v2_baseline_anchored_side_switch_v8_4_profile.json"
)


def _profile() -> dict:
    return json.loads(PROFILE_PATH.read_text())


def _target(
    market_id: str,
    *,
    side: str,
    pnl: float,
    decision_ts: int,
) -> dict:
    return {
        "market_id": market_id,
        "decision_ts": decision_ts,
        "action": f"BUY_{side}_SELL_BEFORE_CLOSE",
        "side": side,
        "after_cost_net_pnl_at_frozen_size": pnl,
        "target_used_as_decision_time_input": False,
    }


def _guard(
    market_id: str,
    *,
    side: str,
    allowed: bool = True,
) -> dict:
    return {
        "market_id": market_id,
        "decision_ts": 1_000,
        "selected_action": f"BUY_{side}_SELL_BEFORE_CLOSE",
        "selected_side": side,
        "execution_guard_order_allowed": allowed,
    }


def _artifact(*, aggregate: bool, direction: bool) -> dict:
    profile = _profile()
    candidate_pnls = [0.2] * 8 if aggregate and direction else [0.2, -0.3]
    candidate = [
        _target(
            f"m-{index}",
            side="DOWN",
            pnl=pnl,
            decision_ts=index,
        )
        for index, pnl in enumerate(candidate_pnls)
    ]
    baseline = [
        _target(f"m-{index}", side="UP", pnl=-0.2, decision_ts=index)
        for index in range(len(candidate_pnls))
    ]
    artifact = v84.build_side_switch_evidence_artifact(
        candidate_rows=candidate,
        baseline_rows=baseline,
        profile=profile,
        evidence_created_ts=2_000,
        source_manifest_sha256="a" * 64,
        excluded_future_registry_hash="b" * 64,
    )
    assert artifact["aggregate_switch_eligible"] is aggregate
    assert artifact["direction_class_eligibility"]["UP_TO_DOWN"] is direction
    return artifact


def test_v8_4_profile_is_frozen_and_fail_closed() -> None:
    profile = _profile()
    v84.validate_baseline_anchored_side_switch_v8_4_profile(profile)
    assert profile["selection_contract"]["side_quota_enabled"] is False
    assert profile["safety"]["paper_candidate_allowed"] is False
    mutated = deepcopy(profile)
    mutated["selection_contract"]["ineligible_disagreement_behavior"] = (
        "use_v8_1"
    )
    with pytest.raises(ValueError, match="selection"):
        v84.validate_baseline_anchored_side_switch_v8_4_profile(mutated)


def test_v8_4_uncertain_historical_switch_evidence_is_ineligible() -> None:
    candidate = [
        _target("m1", side="DOWN", pnl=0.2, decision_ts=1),
        _target("m2", side="DOWN", pnl=-0.3, decision_ts=2),
        _target("m3", side="UP", pnl=0.1, decision_ts=3),
    ]
    baseline = [
        _target("m1", side="UP", pnl=0.0, decision_ts=1),
        _target("m2", side="UP", pnl=0.0, decision_ts=2),
        _target("m3", side="DOWN", pnl=0.0, decision_ts=3),
    ]
    artifact = v84.build_side_switch_evidence_artifact(
        candidate_rows=candidate,
        baseline_rows=baseline,
        profile=_profile(),
        evidence_created_ts=2_000,
        source_manifest_sha256="a" * 64,
        excluded_future_registry_hash="b" * 64,
    )
    assert artifact["aggregate_switch_eligible"] is False
    assert artifact["eligible_switch_classes"] == []
    assert artifact["issue250_outcomes_or_pnl_used_for_fit"] is False


def test_v8_4_synthetic_stable_positive_switch_can_be_eligible() -> None:
    artifact = _artifact(aggregate=True, direction=True)
    v84.validate_side_switch_evidence_artifact(artifact, profile=_profile())
    assert artifact["switch_class_metrics"]["AGGREGATE"]["eligible"] is True
    assert artifact["switch_class_metrics"]["UP_TO_DOWN"]["eligible"] is True
    mutated = deepcopy(artifact)
    mutated["aggregate_switch_eligible"] = False
    mutated["artifact_id"] = v84.canonical_json_sha256(
        {key: value for key, value in mutated.items() if key != "artifact_id"}
    )
    with pytest.raises(ValueError, match="eligibility_consistency"):
        v84.validate_side_switch_evidence_artifact(
            mutated,
            profile=_profile(),
        )


def test_v8_4_ineligible_switch_preserves_v6_7() -> None:
    row = v84.select_baseline_anchored_side_switch_v8_4_decision(
        candidate_row=_guard("m1", side="DOWN"),
        baseline_row=_guard("m1", side="UP"),
        evidence_artifact=_artifact(aggregate=False, direction=False),
        profile=_profile(),
    )
    assert row["selected_side"] == "UP"
    assert row["selection_source"] == "v6_7_baseline_preserved"
    assert row["execution_guard_order_allowed"] is True


def test_v8_4_missing_evidence_preserves_guarded_v6_7() -> None:
    row = v84.select_baseline_anchored_side_switch_v8_4_decision(
        candidate_row=_guard("m1", side="DOWN"),
        baseline_row=_guard("m1", side="UP"),
        evidence_artifact=None,
        profile=_profile(),
    )
    assert row["selected_side"] == "UP"
    assert row["switch_evidence_artifact_valid"] is False
    assert "switch_evidence_invalid" in row["selection_reason_codes"]


def test_v8_4_eligible_switch_requires_candidate_guard() -> None:
    artifact = _artifact(aggregate=True, direction=True)
    passed = v84.select_baseline_anchored_side_switch_v8_4_decision(
        candidate_row=_guard("m1", side="DOWN"),
        baseline_row=_guard("m1", side="UP"),
        evidence_artifact=artifact,
        profile=_profile(),
    )
    blocked = v84.select_baseline_anchored_side_switch_v8_4_decision(
        candidate_row=_guard("m1", side="DOWN", allowed=False),
        baseline_row=_guard("m1", side="UP"),
        evidence_artifact=artifact,
        profile=_profile(),
    )
    assert passed["selected_side"] == "DOWN"
    assert passed["selection_source"] == "eligible_v8_1_side_switch"
    assert blocked["selected_side"] == "UP"
    assert blocked["selection_source"] == "v6_7_baseline_preserved"


def test_v8_4_rejects_outcome_fields_at_inference() -> None:
    candidate = _guard("m1", side="DOWN")
    candidate["resolved_outcome"] = "DOWN"
    with pytest.raises(ValueError, match="target fields"):
        v84.select_baseline_anchored_side_switch_v8_4_decision(
            candidate_row=candidate,
            baseline_row=_guard("m1", side="UP"),
            evidence_artifact=_artifact(aggregate=False, direction=False),
            profile=_profile(),
        )


def test_v8_4_historical_replay_preserves_baseline_when_switch_ineligible() -> None:
    profile = _profile()
    candidate_targets = [
        _target("m1", side="DOWN", pnl=-0.2, decision_ts=1),
        _target("m2", side="DOWN", pnl=0.1, decision_ts=2),
    ]
    baseline_targets = [
        _target("m1", side="UP", pnl=0.2, decision_ts=1),
        _target("m2", side="DOWN", pnl=0.1, decision_ts=2),
    ]
    evidence = v84.build_side_switch_evidence_artifact(
        candidate_rows=candidate_targets,
        baseline_rows=baseline_targets,
        profile=profile,
        evidence_created_ts=2_000,
        source_manifest_sha256="a" * 64,
        excluded_future_registry_hash="b" * 64,
    )
    decisions = [
        {
            "market_id": "m1",
            "decision_ts": 1,
            "original_v8_1_action": "BUY_DOWN_SELL_BEFORE_CLOSE",
            "original_v8_1_side": "DOWN",
            "original_v8_1_guard_allowed": True,
            "original_v6_7_action": "BUY_UP_SELL_BEFORE_CLOSE",
            "original_v6_7_side": "UP",
            "original_v6_7_guard_allowed": True,
        },
        {
            "market_id": "m2",
            "decision_ts": 2,
            "original_v8_1_action": "BUY_DOWN_SELL_BEFORE_CLOSE",
            "original_v8_1_side": "DOWN",
            "original_v8_1_guard_allowed": True,
            "original_v6_7_action": "BUY_DOWN_SELL_BEFORE_CLOSE",
            "original_v6_7_side": "DOWN",
            "original_v6_7_guard_allowed": True,
        },
    ]
    result = v84.build_v8_4_historical_replay(
        decision_rows=decisions,
        candidate_target_rows=candidate_targets,
        baseline_target_rows=list(reversed(baseline_targets)),
        evidence_artifact=evidence,
        profile=profile,
        evaluation_started_ts=3_000,
    )
    report = result["report"]
    assert report["historical_noninferiority_gate_passed"] is True
    assert report["candidate_minus_v6_7_total_after_cost_pnl"] == 0.0
    assert report["final_policy_difference_market_count"] == 0
    assert report["model_improvement_demonstrated"] is False
    assert report["new_future_challenger_collection_justified"] is False
    assert report["paper_candidate_allowed"] is False
