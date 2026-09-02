from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_direct_decision_group_advantage_v2_fit import (
    _adaptive_boundaries,
    _attach_direct_estimands,
    _market_bootstrap_shrunken_estimator,
    _strip_training_targets,
    _validate_fit_materialization,
    _validate_target_stripped_rows,
    _viability_report,
    validate_direct_decision_group_advantage_v2_fit_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_action_value import (
    REQUIRED_ACTIONS,
)

PROFILE_PATH = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_direct_decision_group_advantage_v2_fit_profile.json"
)


def test_issue198_fit_profile_freezes_model_calibration_and_evidence_scope() -> None:
    profile = _load_json(PROFILE_PATH)

    validate_direct_decision_group_advantage_v2_fit_profile(profile)

    assert profile["fit_role"] == "development_train"
    assert profile["required_fit_market_count"] == 90
    assert profile["cross_fit"]["hyperparameter_search_enabled"] is False
    assert profile["calibration"]["estimands"] == [
        "absolute_post_cost_net_return",
        "advantage_vs_no_trade",
        "advantage_vs_best_alternative",
    ]
    assert profile["calibration"]["bootstrap_complete_shrunken_estimator_required"] is True
    assert (
        profile["calibration"]["convex_combination_of_separately_estimated_lcbs_allowed"] is False
    )
    assert profile["output_contract"]["future_evaluation_in_this_fit_issue_allowed"] is False


@pytest.mark.parametrize(
    ("section", "key", "value", "reason"),
    [
        ("cross_fit", "hyperparameter_search_enabled", True, "fixed_ranker"),
        (
            "calibration",
            "current_issue189_oof_files_may_be_opened",
            True,
            "old_and_future_evidence_sealed",
        ),
        (
            "calibration",
            "convex_combination_of_separately_estimated_lcbs_allowed",
            True,
            "full_estimator_bootstrap",
        ),
        (
            "output_contract",
            "future_evaluation_in_this_fit_issue_allowed",
            True,
            "research_only",
        ),
    ],
)
def test_issue198_fit_profile_rejects_drift(
    section: str,
    key: str,
    value: object,
    reason: str,
) -> None:
    profile = _load_json(PROFILE_PATH)
    profile[section][key] = value

    with pytest.raises(ValueError, match=reason):
        validate_direct_decision_group_advantage_v2_fit_profile(profile)


def test_issue198_adaptive_boundaries_merge_duplicate_quantiles() -> None:
    boundaries, merged = _adaptive_boundaries(
        [0.0] * 12 + [1.0] * 3,
        quantiles=[1.0 / 3.0, 2.0 / 3.0],
    )

    assert boundaries == [0.0]
    assert merged == 1
    assert all(
        current > previous for previous, current in zip(boundaries, boundaries[1:], strict=False)
    )

    terminal_boundaries, terminal_merged = _adaptive_boundaries(
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        quantiles=[1.0 / 3.0, 2.0 / 3.0],
    )
    assert terminal_boundaries == []
    assert terminal_merged == 2


def test_issue198_direct_estimands_are_decision_group_relative_and_cost_aware() -> None:
    returns = {
        "BUY_UP_HOLD_TO_SETTLEMENT": 0.20,
        "BUY_DOWN_HOLD_TO_SETTLEMENT": -0.20,
        "BUY_UP_SELL_BEFORE_CLOSE": 0.05,
        "BUY_DOWN_SELL_BEFORE_CLOSE": -0.05,
        "NO_TRADE": 0.0,
    }
    rows = [
        {
            "market_id": "market-1",
            "decision_ts": 1_000,
            "action": action,
            "target_net_pnl_per_contract": returns[action],
        }
        for action in REQUIRED_ACTIONS
    ]

    output = _attach_direct_estimands(rows)
    up = next(row for row in output if row["action"] == "BUY_UP_HOLD_TO_SETTLEMENT")
    down = next(row for row in output if row["action"] == "BUY_DOWN_HOLD_TO_SETTLEMENT")

    assert up["training_target_absolute_post_cost_net_return"] == pytest.approx(0.20)
    assert up["training_target_advantage_vs_no_trade"] == pytest.approx(0.20)
    assert up["training_target_advantage_vs_best_alternative"] == pytest.approx(0.15)
    assert down["training_target_advantage_vs_best_alternative"] == pytest.approx(-0.40)
    assert all(row["training_targets_include_costs"] is True for row in output)
    assert all(row["training_targets_used_as_decision_inputs"] is False for row in output)


def test_issue198_market_bootstrap_resamples_complete_shrunken_estimator() -> None:
    action_rows = [
        {"market_id": f"market-{index}", "target": value}
        for index, value in enumerate((0.10, 0.20, -0.10, 0.30), start=1)
    ]
    group_rows = action_rows[:3]

    first = _market_bootstrap_shrunken_estimator(
        action_rows,
        group_rows,
        target_field="target",
        prior_market_count=2,
        minimum_group_markets=2,
        bootstrap_resample_count=2_000,
        confidence_level=0.95,
        seed=123,
    )
    second = _market_bootstrap_shrunken_estimator(
        action_rows,
        group_rows,
        target_field="target",
        prior_market_count=2,
        minimum_group_markets=2,
        bootstrap_resample_count=2_000,
        confidence_level=0.95,
        seed=123,
    )

    assert first == second
    assert first["estimate_source"] == "complete_shrunken_estimator_market_bootstrap"
    assert first["group_support_passed"] is True
    assert first["convex_combination_of_separate_lcbs_used"] is False
    assert first["bootstrap_unit"] == "market_id"


def test_issue198_target_stripping_removes_training_and_outcome_fields() -> None:
    row = {
        "market_id": "market-1",
        "decision_ts": 1_000,
        "action": "NO_TRADE",
        "side": "NONE",
        "action_family": "NO_TRADE",
        "target_net_pnl_per_contract": 0.0,
        "training_target_absolute_post_cost_net_return": 0.0,
        "target_resolved_outcome": "UP",
        "direct_advantage_all_lcb_checks_passed": False,
        "action_advantage_lcb_net_return": 0.0,
        "paper_only": True,
        "capital_at_risk": False,
    }

    stripped = _strip_training_targets(row)

    assert "target_net_pnl_per_contract" not in stripped
    assert "training_target_absolute_post_cost_net_return" not in stripped
    assert "target_resolved_outcome" not in stripped
    assert stripped["training_target_fields_stripped"] is True
    _validate_target_stripped_rows([stripped])


def test_issue198_materialization_rejects_quarantined_roles() -> None:
    rows = [{"market_id": f"market-{index}", "role": "development_train"} for index in range(90)]
    audits = [
        {
            "blocking_reason_codes": [],
            "feature_causality_violation_count": 0,
            "cost_component_violation_count": 0,
        }
        for _ in range(90)
    ]

    _validate_fit_materialization(rows, audits)

    rows[-1]["role"] = "development_calibration"
    with pytest.raises(ValueError, match="quarantined role"):
        _validate_fit_materialization(rows, audits)


def test_issue198_zero_viability_blocks_future_evaluation() -> None:
    report = _viability_report(
        run_id="test-run",
        viability_rows=[
            {
                "source_selected_action": "NO_TRADE",
                "first_terminal_stage": "selected_no_trade",
                "execution_guard_evaluated": False,
                "execution_guard_order_allowed": False,
            }
        ],
        scored_predictions=[
            {
                "action": "NO_TRADE",
                "direct_advantage_all_lcb_checks_passed": False,
            }
        ],
    )

    assert report["outcome_blind_viability_passed"] is False
    assert report["outcome_blind_viability_blocking_reason_codes"] == [
        "zero_direct_lcb_passed_action_support",
        "zero_execution_guard_evaluable_decision_support",
    ]
    assert report["source_model_candidate_eligible"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["v8_execution_handoff_allowed"] is False


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
