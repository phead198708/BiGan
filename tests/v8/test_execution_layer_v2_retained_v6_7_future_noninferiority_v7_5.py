from __future__ import annotations

import json
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_retained_v6_7_future_noninferiority_v7_5 import (
    FIVE_ACTIONS,
    FROZEN_PROFILE_SHA256,
    _complete_five_action_grid,
    _safety_fields,
    _target_free_checks,
    build_retained_v6_7_future_noninferiority_gate,
    validate_retained_v6_7_future_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
    _sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / (
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_retained_v6_7_future_noninferiority_v7_5_profile.json"
)


def _profile() -> dict:
    return json.loads(PROFILE_PATH.read_text())


def _target_row(index: int, pnl: float, *, side: str = "UP") -> dict:
    action = f"BUY_{side}_SELL_BEFORE_CLOSE"
    return {
        "market_id": f"market-{index:03d}",
        "decision_ts": 1_000_000 + index,
        "max_input_ts": 999_000 + index,
        "side": side,
        "action": action,
        "runtime_policy_after_cost_net_pnl_at_frozen_size": pnl,
        "target_available_only_post_exit_or_official_resolution": True,
        "target_used_as_decision_time_input": False,
    }


def _decision_row(index: int, *, side: str = "UP") -> dict:
    decision_ts = 1_000_000 + index
    return {
        "market_id": f"market-{index:03d}",
        "decision_ts": decision_ts,
        "max_input_ts": decision_ts - 1,
        "market_close_ts": decision_ts + 100,
        "side": side,
        "action": f"BUY_{side}_SELL_BEFORE_CLOSE",
        "v6_7_base_score": 0.1,
        "microstructure_safety_passed": True,
        "source_score_mutated": False,
        "labels_outcomes_resolution_or_pnl_opened": False,
    }


def test_frozen_profile_hash_and_inclusive_comparison_contract() -> None:
    profile = _profile()

    validate_retained_v6_7_future_profile(profile)

    assert _sha256_file(PROFILE_PATH) == FROZEN_PROFILE_SHA256
    gate = profile["future_pnl_gate"]
    assert gate["comparison_operator"] == "greater_than_or_equal"
    assert gate["equality_passes_noninferiority"] is True
    assert gate["candidate_minus_v6_7_after_cost_pnl_minimum_inclusive"] == 0.0
    assert gate["candidate_minus_v6_7_largest_winner_removed_minimum_inclusive"] == 0.0


def test_equal_positive_candidate_passes_noninferiority_without_claiming_improvement() -> None:
    market_ids = [f"market-{index:03d}" for index in range(120)]
    candidate = [_target_row(index, 0.01) for index in range(120)]
    baseline = [_target_row(index, 0.01) for index in range(120)]

    report = build_retained_v6_7_future_noninferiority_gate(
        candidate,
        baseline_rows=baseline,
        evaluation_market_ids=market_ids,
        profile=_profile(),
        target_free_freeze_sha256="a" * 64,
    )

    assert report["candidate_after_cost_pnl"] == report["v6_7_after_cost_pnl"]
    assert report["candidate_minus_v6_7_after_cost_pnl"] == 0.0
    assert report["candidate_minus_v6_7_market_bootstrap"]["lower_confidence_bound"] == 0.0
    assert report["historical_noninferiority_gate_passed"] is True
    assert report["future_noninferiority_gate_passed"] is True
    assert report["model_improvement_demonstrated"] is False
    assert report["future_pnl_gate_passed"] is True
    assert report["bounded_paper_candidate_review_allowed"] is True
    assert report["paper_candidate_allowed"] is False
    assert report["v8_execution_handoff_allowed"] is False


def test_equal_but_negative_absolute_pnl_still_fails_closed() -> None:
    market_ids = [f"market-{index:03d}" for index in range(120)]
    candidate = [_target_row(index, -0.01) for index in range(120)]
    baseline = [_target_row(index, -0.01) for index in range(120)]

    report = build_retained_v6_7_future_noninferiority_gate(
        candidate,
        baseline_rows=baseline,
        evaluation_market_ids=market_ids,
        profile=_profile(),
        target_free_freeze_sha256="b" * 64,
    )

    assert report["future_noninferiority_gate_passed"] is True
    assert report["future_pnl_gate_passed"] is False
    assert "accepted_total_after_cost_pnl_not_positive" in report[
        "future_pnl_gate_blocking_reason_codes"
    ]
    assert "largest_winner_removed_after_cost_pnl_not_positive" in report[
        "future_pnl_gate_blocking_reason_codes"
    ]
    assert report["bounded_paper_candidate_review_allowed"] is False


def test_inferior_candidate_fails_inclusive_delta_gate() -> None:
    market_ids = [f"market-{index:03d}" for index in range(120)]
    candidate = [_target_row(index, 0.009) for index in range(120)]
    baseline = [_target_row(index, 0.01) for index in range(120)]

    report = build_retained_v6_7_future_noninferiority_gate(
        candidate,
        baseline_rows=baseline,
        evaluation_market_ids=market_ids,
        profile=_profile(),
        target_free_freeze_sha256="c" * 64,
    )

    assert report["future_noninferiority_gate_passed"] is False
    assert "candidate_total_pnl_inferior_to_v6_7" in report[
        "future_pnl_gate_blocking_reason_codes"
    ]
    assert "candidate_minus_v6_7_bootstrap_lcb_negative" in report[
        "future_pnl_gate_blocking_reason_codes"
    ]
    assert "retained_candidate_runtime_policy_differs_from_v6_7" in report[
        "future_pnl_gate_blocking_reason_codes"
    ]


def test_one_sided_regime_has_no_side_quota_or_side_pnl_gate() -> None:
    market_ids = [f"market-{index:03d}" for index in range(120)]
    rows = [_target_row(index, 0.01, side="DOWN") for index in range(120)]

    report = build_retained_v6_7_future_noninferiority_gate(
        rows,
        baseline_rows=[dict(row) for row in rows],
        evaluation_market_ids=market_ids,
        profile=_profile(),
        target_free_freeze_sha256="d" * 64,
    )

    assert report["accepted_side_distribution_diagnostic"] == {"DOWN": 120}
    assert report["side_quota_enabled"] is False
    assert report["side_pnl_hard_gate_enabled"] is False
    assert report["future_pnl_gate_passed"] is True


def test_target_free_checks_require_exact_closed_causal_outcome_free_rows() -> None:
    decisions = [
        _decision_row(index, side="UP" if index < 31 else "DOWN") for index in range(120)
    ]
    selected = [
        {
            "market_id": f"market-{index:03d}",
            "market_end_ts": 1_100_000 + index,
        }
        for index in range(120)
    ]
    source_decision = {
        "all_selected_markets_closed_before_freeze": True,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "settlement_provider_called": False,
        "source_score_mutated": False,
    }

    checks, side_distribution = _target_free_checks(
        selected,
        decisions,
        source_decision=source_decision,
        stage_started_ts=2_000_000,
        minimum_support=40,
    )

    assert all(checks.values())
    assert side_distribution == {"DOWN": 89, "UP": 31}

    decisions[0]["settlement_pnl"] = 1.0
    checks, _ = _target_free_checks(
        selected,
        decisions,
        source_decision=source_decision,
        stage_started_ts=2_000_000,
        minimum_support=40,
    )
    assert checks["target_fields_absent"] is False


def test_complete_five_action_grid_is_causal_and_outcome_free() -> None:
    rows = [
        {
            "market_id": f"market-{market_index:03d}",
            "decision_ts": 1_000_000 + market_index,
            "max_input_ts": 999_000 + market_index,
            "action": action,
        }
        for market_index in range(120)
        for action in sorted(FIVE_ACTIONS)
    ]

    assert _complete_five_action_grid(rows) is True

    rows.pop()
    assert _complete_five_action_grid(rows) is False


def test_all_safety_flags_remain_blocked() -> None:
    safety = _safety_fields()
    assert safety == {
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        "paper_candidate_allowed": False,
    }
