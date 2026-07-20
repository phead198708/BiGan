from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_sbc_exit_reliability_v6_3_fit import (
    apply_exit_reliability_model,
    build_exit_reliability_calibration,
    build_v6_3_side_only_oof_gate,
    fit_regularized_logistic_exit_model,
    select_replay_rows_for_role,
    validate_sbc_exit_reliability_v6_3_fit_profile,
)


def test_fit_profile_freezes_model_calibration_oof_and_safety() -> None:
    profile = _profile()
    validate_sbc_exit_reliability_v6_3_fit_profile(profile)
    assert profile["model"]["hyperparameter_search_enabled"] is False
    assert profile["calibration"]["threshold_search_uses_oof_labels"] is False
    assert profile["historical_oof_side_only_gate"]["aggregation"] == (
        "buy_up_buy_down_side_only"
    )
    assert profile["safety"]["promotion_evidence_eligible"] is False


def test_regularized_model_uses_decision_time_features_and_is_deterministic() -> None:
    profile = _small_profile()
    rows = _training_rows(40)
    first = fit_regularized_logistic_exit_model(rows, profile=profile)
    second = fit_regularized_logistic_exit_model(rows, profile=profile)
    assert first == second
    assert first["coefficients_finite"] is True
    assert first["coefficients_within_bound"] is True
    scored = apply_exit_reliability_model(rows, model=first, profile=profile)
    assert all("target" not in row for row in scored)
    assert all(row["target_used_for_inference"] is False for row in scored)


def test_calibration_uses_only_calibration_targets_and_freezes_threshold() -> None:
    profile = _small_profile()
    rows = _training_rows(40)
    model = fit_regularized_logistic_exit_model(rows, profile=profile)
    scored = apply_exit_reliability_model(rows, model=model, profile=profile)
    calibration = build_exit_reliability_calibration(
        rows,
        scored,
        model=model,
        profile=profile,
        stability={"coefficient_stability_gate_passed": True},
    )
    assert calibration["threshold_search_uses_oof_labels"] is False
    assert calibration["threshold_search_uses_pnl"] is False
    assert calibration["selected_threshold"] in profile["calibration"][
        "threshold_candidates"
    ]


def test_side_only_oof_gate_ignores_action_family_as_blocker() -> None:
    profile = _small_profile()
    rows = []
    for index in range(12):
        side = "UP" if index % 2 == 0 else "DOWN"
        rows.append(
            {
                "market_id": f"market-{index}",
                "selected_side": side,
                "selected_action_family": (
                    "SELL_BEFORE_CLOSE" if index % 3 else "HOLD_TO_SETTLEMENT"
                ),
                "v6_2_guard_order_allowed": True,
                "v6_3_guard_order_allowed": True,
                "v6_2_accepted_bet_net_pnl": -0.01,
                "v6_3_accepted_bet_net_pnl": 0.02,
            }
        )
    report = build_v6_3_side_only_oof_gate(rows, profile=profile)
    assert report["historical_side_only_oof_gate_passed"] is True
    assert report["action_and_family_metrics_diagnostic_only"] is True
    assert report["source_model_candidate_eligible"] is False
    assert report["#134_resume_allowed"] is False
    assert report["#146_start_allowed"] is False


def test_side_only_oof_gate_fails_closed_on_negative_side() -> None:
    profile = _small_profile()
    rows = []
    for index in range(12):
        side = "UP" if index % 2 == 0 else "DOWN"
        pnl = -0.02 if side == "UP" else 0.03
        rows.append(
            {
                "market_id": f"market-{index}",
                "selected_side": side,
                "v6_2_guard_order_allowed": True,
                "v6_3_guard_order_allowed": True,
                "v6_2_accepted_bet_net_pnl": -0.01,
                "v6_3_accepted_bet_net_pnl": pnl,
            }
        )
    report = build_v6_3_side_only_oof_gate(rows, profile=profile)
    assert report["historical_side_only_oof_gate_passed"] is False
    assert "candidate_each_side_pnl_gate_failed" in report[
        "historical_side_only_oof_gate_reason_codes"
    ]


def test_target_free_replay_role_is_derived_from_frozen_market_lineage() -> None:
    replay_rows = [
        {"market_id": "train", "decision_ts": 1, "decision_index": 1},
        {"market_id": "oof", "decision_ts": 2, "decision_index": 1},
    ]
    role_rows = [
        {"market_id": "train", "role": "development_train"},
        {"market_id": "oof", "role": "confirmatory_validation"},
    ]
    selected = select_replay_rows_for_role(
        replay_rows,
        role_rows=role_rows,
        role="confirmatory_validation",
    )
    assert [row["market_id"] for row in selected] == ["oof"]
    assert "historical_source_role" not in selected[0]


def test_target_free_replay_role_selection_fails_on_missing_lineage_market() -> None:
    with pytest.raises(ValueError, match="missing frozen lineage markets"):
        select_replay_rows_for_role(
            [{"market_id": "train", "decision_ts": 1}],
            role_rows=[
                {"market_id": "oof", "role": "confirmatory_validation"}
            ],
            role="confirmatory_validation",
        )


def _profile() -> dict:
    path = Path(
        "examples/v8/polymarket_configs/"
        "execution_layer_v2_sbc_exit_reliability_v6_3_fit_profile.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _small_profile() -> dict:
    profile = copy.deepcopy(_profile())
    profile["model"]["gradient_descent_iterations"] = 200
    profile["coefficient_stability"]["bootstrap_resample_count"] = 2
    profile["coefficient_stability"]["bootstrap_fit_iterations"] = 20
    profile["calibration"].update(
        {
            "threshold_candidates": [0.5],
            "minimum_selected_row_count": 1,
            "minimum_selected_unique_market_count": 1,
            "minimum_selected_unique_market_count_per_side": 1,
            "minimum_precision": 0.5,
            "minimum_market_bootstrap_precision_lcb": 0.0,
            "minimum_recall_per_side": 0.0,
            "minimum_roc_auc": 0.5,
            "bootstrap_resample_count": 20,
        }
    )
    profile["historical_oof_side_only_gate"].update(
        {
            "minimum_guard_accepted_unique_market_count": 10,
            "minimum_guard_accepted_unique_market_count_per_side": 5,
            "bootstrap_resample_count": 100,
        }
    )
    return profile


def _training_rows(count: int) -> list[dict]:
    columns = _profile()["model"]["feature_columns"]
    rows = []
    for index in range(count):
        target = int(index % 4 != 0)
        side = "UP" if index % 2 == 0 else "DOWN"
        values = {name: float((index + offset) % 7) for offset, name in enumerate(columns)}
        values["time_to_close_seconds"] = 240.0 if target else 20.0
        rows.append(
            {
                "market_id": f"market-{index // 2}",
                "decision_ts": 1000 + index,
                "side": side,
                "action": f"BUY_{side}_SELL_BEFORE_CLOSE",
                "features": values,
                "target": target,
            }
        )
    return rows
