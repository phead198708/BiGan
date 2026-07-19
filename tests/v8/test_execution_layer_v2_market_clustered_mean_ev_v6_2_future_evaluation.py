from __future__ import annotations

import json
from pathlib import Path

import pytest

import bigan.v8.polymarket.training.execution_layer_v2_market_clustered_mean_ev_v6_2_future_evaluation as subject
from bigan.v8.polymarket.training.execution_layer_v2_market_clustered_mean_ev_v6_2_future_evaluation import (
    _bound_single_use_claim_path,
    _claim_single_use,
    _select_exact_future_index_rows,
    _target_free_support,
    _validate_collection_profile,
    _validate_exact_feature_action_grid,
    _validate_freeze_manifest_for_target_access,
    build_market_clustered_mean_ev_v6_2_side_only_gate,
    validate_market_clustered_mean_ev_v6_2_future_profile,
)

PROFILE_PATH = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_market_clustered_mean_ev_v6_2_future_evaluation_v1.json"
)
COLLECTION_PROFILE_PATH = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_market_clustered_mean_ev_v6_2_future_holdout_v1.json"
)


def _profile() -> dict:
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def test_profile_freezes_exact_window_support_and_side_only_gate() -> None:
    profile = _profile()
    validate_market_clustered_mean_ev_v6_2_future_profile(profile)
    assert profile["window"]["quality_valid_market_count"] == 200
    assert profile["window"]["maximum_index_scan_count"] == 240
    assert profile["support_and_pnl_gates"][
        "minimum_guard_accepted_unique_market_count"
    ] == 120
    assert profile["support_and_pnl_gates"]["minimum_supported_side_market_count"] == 17
    assert profile["support_and_pnl_gates"]["pnl_hard_gate_aggregation"] == (
        "selected_side_buy_up_buy_down_only"
    )
    assert profile["access_sequence"]["future_result_driven_rerun_allowed"] is False
    assert profile["safety"]["promotion_evidence_eligible"] is False


def test_collection_profile_is_bound_to_candidate_and_exact_window() -> None:
    collection = json.loads(COLLECTION_PROFILE_PATH.read_text(encoding="utf-8"))
    candidate_sha256 = _profile()["candidate_manifest_sha256"]
    _validate_collection_profile(collection, candidate_sha256=candidate_sha256)
    collection["candidate"]["candidate_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="candidate_hash"):
        _validate_collection_profile(collection, candidate_sha256=candidate_sha256)


def test_exact_window_uses_earliest_200_quality_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subject, "_prior_market_reference", lambda candidate: (set(), "a" * 64))
    rows = [_index_row(sequence) for sequence in range(313, 553)]
    selected, attempted = _select_exact_future_index_rows(
        rows,
        profile=_profile(),
        candidate={},
    )
    assert len(selected) == 200
    assert selected[0]["sequence"] == 313
    assert selected[-1]["sequence"] == 512
    assert len(attempted) == 200


def test_exact_window_scans_invalid_rows_but_never_selects_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "_prior_market_reference", lambda candidate: (set(), "a" * 64))
    rows = [_index_row(sequence) for sequence in range(313, 553)]
    rows[0]["capture_quality_valid"] = False
    selected, attempted = _select_exact_future_index_rows(
        rows,
        profile=_profile(),
        candidate={},
    )
    assert selected[0]["sequence"] == 314
    assert selected[-1]["sequence"] == 513
    assert len(attempted) == 201


def test_exact_window_rejects_resolution_or_target_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "_prior_market_reference", lambda candidate: (set(), "a" * 64))
    rows = [_index_row(sequence) for sequence in range(313, 553)]
    rows[5]["raw_resolution_row_count"] = 1
    with pytest.raises(ValueError, match="resolution"):
        _select_exact_future_index_rows(rows, profile=_profile(), candidate={})
    rows[5]["raw_resolution_row_count"] = 0
    rows[5]["labels_outcomes_or_pnl_opened"] = True
    with pytest.raises(ValueError, match="target"):
        _select_exact_future_index_rows(rows, profile=_profile(), candidate={})


def test_target_free_support_is_unique_market_and_side_scoped() -> None:
    replay = [
        {
            "market_id": f"market-{index}",
            "selected_side": "UP" if index < 60 else "DOWN",
            "execution_guard_order_allowed": True,
        }
        for index in range(120)
    ]
    support = _target_free_support(replay, profile=_profile())
    assert support["target_free_support_gate_passed"] is True
    assert support["guard_accepted_unique_market_count"] == 120
    assert support["guard_accepted_unique_market_count_by_side"] == {
        "UP": 60,
        "DOWN": 60,
    }
    replay.extend([dict(replay[0]) for _ in range(100)])
    assert _target_free_support(replay, profile=_profile())[
        "guard_accepted_unique_market_count"
    ] == 120


def test_exact_materialized_grid_requires_five_actions_and_causal_features() -> None:
    selected = [
        {
            "market_id": "market-313",
            "market_start_ts": 1_784_471_300_000,
        }
    ]
    feature_rows = [
        {
            "market_id": "market-313",
            "decision_ts": 1_784_471_400_000,
            "max_input_ts": 1_784_471_399_999,
        }
    ]
    actions = [
        {
            **feature_rows[0],
            "action": action,
        }
        for action in sorted(subject.EXPECTED_ACTIONS)
    ]
    candidate = {"future_collection_minimum_created_ts_exclusive": 1_784_470_529_364}
    _validate_exact_feature_action_grid(
        feature_rows,
        actions,
        selected_rows=selected,
        candidate=candidate,
    )
    actions[0]["max_input_ts"] = actions[0]["decision_ts"] + 1
    with pytest.raises(ValueError, match="causality"):
        _validate_exact_feature_action_grid(
            feature_rows,
            actions,
            selected_rows=selected,
            candidate=candidate,
        )


def test_support_failure_keeps_target_access_fail_closed() -> None:
    manifest = {
        "schema_version": f"{subject.SCHEMA_PREFIX}-prediction-freeze-manifest-v1",
        "decision_freeze_written_before_target_access": True,
        "future_target_access_allowed": False,
        "labels_outcomes_or_pnl_opened": False,
        "resolution_artifact_opened": False,
        "settlement_provider_called": False,
        **subject._blocked_safety_fields(),
    }
    with pytest.raises(ValueError, match="support"):
        _validate_freeze_manifest_for_target_access(manifest)


def test_side_only_gate_passes_positive_two_sided_market_grouped_evidence() -> None:
    markets = [f"market-{index:03d}" for index in range(200)]
    candidate = [
        _evaluation_row(
            market_id=market,
            side="UP" if index < 60 else "DOWN",
            pnl=0.02,
        )
        for index, market in enumerate(markets[:120])
    ]
    gate = build_market_clustered_mean_ev_v6_2_side_only_gate(
        candidate,
        matched_v5_rows=[],
        evaluation_market_ids=markets,
        profile=_profile(),
        decision_freeze_sha256="a" * 64,
    )
    assert gate["future_gate_passed"] is True
    assert gate["candidate_post_cost_net_pnl"] == pytest.approx(2.4)
    assert gate["matched_v5_post_cost_net_pnl"] == 0.0
    assert gate["accepted_side_metrics"]["UP"]["accepted_bet_net_pnl_sum"] > 0.0
    assert gate["accepted_side_metrics"]["DOWN"]["accepted_bet_net_pnl_sum"] > 0.0
    assert gate["promotion_evidence_eligible"] is False
    assert gate["#134_resume_allowed"] is False
    assert gate["#146_start_allowed"] is False


def test_side_only_gate_fails_when_one_side_loses_even_if_total_is_positive() -> None:
    markets = [f"market-{index:03d}" for index in range(200)]
    candidate = [
        _evaluation_row(
            market_id=market,
            side="UP" if index < 20 else "DOWN",
            pnl=-0.01 if index < 20 else 0.03,
        )
        for index, market in enumerate(markets[:120])
    ]
    gate = build_market_clustered_mean_ev_v6_2_side_only_gate(
        candidate,
        matched_v5_rows=[],
        evaluation_market_ids=markets,
        profile=_profile(),
        decision_freeze_sha256="b" * 64,
    )
    assert gate["candidate_post_cost_net_pnl"] > 0.0
    assert gate["future_gate_passed"] is False
    assert "supported_side_post_cost_pnl_gate_failed" in gate[
        "future_gate_blocking_reason_codes"
    ]


def test_action_and_family_metrics_are_diagnostic_only() -> None:
    markets = [f"market-{index:03d}" for index in range(200)]
    candidate = [
        _evaluation_row(
            market_id=market,
            side="UP" if index < 60 else "DOWN",
            pnl=0.02,
            action=(
                "BUY_UP_HOLD_TO_SETTLEMENT"
                if index == 0
                else (
                    "BUY_UP_SELL_BEFORE_CLOSE"
                    if index < 60
                    else "BUY_DOWN_SELL_BEFORE_CLOSE"
                )
            ),
        )
        for index, market in enumerate(markets[:120])
    ]
    gate = build_market_clustered_mean_ev_v6_2_side_only_gate(
        candidate,
        matched_v5_rows=[],
        evaluation_market_ids=markets,
        profile=_profile(),
        decision_freeze_sha256="c" * 64,
    )
    assert gate["future_gate_passed"] is True
    assert gate["accepted_action_metrics"]["BUY_UP_HOLD_TO_SETTLEMENT"][
        "diagnostic_only"
    ] is True
    assert gate["accepted_action_family_metrics"]["HOLD_TO_SETTLEMENT"][
        "diagnostic_only"
    ] is True


def test_single_use_claim_is_atomic_and_cannot_be_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "single-use.json"
    _claim_single_use(path, {"claim_id": "first"})
    assert json.loads(path.read_text(encoding="utf-8"))["claim_id"] == "first"
    with pytest.raises(ValueError, match="already consumed"):
        _claim_single_use(path, {"claim_id": "second"})
    assert json.loads(path.read_text(encoding="utf-8"))["claim_id"] == "first"


def test_single_use_claim_path_is_deterministically_bound_to_freeze(tmp_path: Path) -> None:
    freeze = tmp_path / "freeze" / "v6_2_future_prediction_freeze_manifest.json"
    assert _bound_single_use_claim_path(freeze) == (
        freeze.parent.resolve() / subject.SINGLE_USE_CLAIM_FILENAME
    )


def _index_row(sequence: int) -> dict:
    return {
        "sequence": sequence,
        "market_id": f"market-{sequence}",
        "market_start_ts": 1_784_471_000_000 + sequence * 300_000,
        "capture_quality_valid": True,
        "raw_resolution_row_count": 0,
        "labels_outcomes_or_pnl_opened": False,
    }


def _evaluation_row(
    *,
    market_id: str,
    side: str,
    pnl: float,
    action: str | None = None,
) -> dict:
    selected_action = action or f"BUY_{side}_SELL_BEFORE_CLOSE"
    return {
        "market_id": market_id,
        "execution_guard_order_allowed": True,
        "accepted_bet_net_pnl": pnl,
        "selected_side": side,
        "executed_action": selected_action,
        "settlement_resolved": True,
        "target_joined_after_decision_freeze": True,
        "target_used_as_decision_input": False,
        "forbidden_outcome_field_used_for_decision": False,
        "feature_causality_violation": False,
        "provenance_violation": False,
        "runtime_state_violation": False,
    }
