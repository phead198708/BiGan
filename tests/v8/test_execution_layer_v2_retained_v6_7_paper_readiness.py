from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from bigan.v8.polymarket.training import (
    execution_layer_v2_retained_v6_7_paper_readiness as readiness,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = (
    ROOT
    / "examples"
    / "v8"
    / "polymarket_configs"
    / "execution_layer_v2_retained_v6_7_paper_readiness_v1.json"
)


def _profile() -> dict:
    return json.loads(PROFILE_PATH.read_text())


def _future(name: str, *, start: int, pnl: list[float]) -> dict:
    return {
        "name": name,
        "role": "independent_future_variance_planning_only",
        "market_ids": [f"{name}-{index}" for index in range(len(pnl))],
        "quality_valid_market_count": len(pnl),
        "guard_accepted_market_count": len(pnl) - 1,
        "guard_blocked_zero_market_count": 1,
        "minimum_market_start_ts": start,
        "maximum_market_start_ts": start + len(pnl),
        "market_pnl_values": pnl,
        "source_manifest_sha256": "a" * 64,
        "target_free_manifest_sha256": "b" * 64,
        "settlement_sha256": "c" * 64,
        "lineage_valid": True,
    }


def _historical() -> dict:
    return {
        "name": "issue251_historical_screening",
        "role": "historical_screening_lineage_only",
        "market_ids": ["h1", "h2"],
        "quality_valid_market_count": 2,
        "guard_accepted_market_count": 2,
        "guard_blocked_zero_market_count": 0,
        "minimum_market_start_ts": 1,
        "maximum_market_start_ts": 2,
        "market_pnl_values": [],
        "source_manifest_sha256": "d" * 64,
        "target_free_manifest_sha256": "",
        "settlement_sha256": "",
        "lineage_valid": True,
    }


def test_profile_is_frozen_and_fail_closed() -> None:
    profile = _profile()
    readiness.validate_retained_v6_7_paper_readiness_profile(profile)
    assert profile["champion_contract"][
        "model_score_cost_sizing_threshold_and_guard_mutation_allowed"
    ] is False
    assert profile["safety"]["paper_candidate_allowed"] is False
    mutated = deepcopy(profile)
    mutated["power_design"]["result_selected_extension_allowed"] = True
    with pytest.raises(ValueError, match="power"):
        readiness.validate_retained_v6_7_paper_readiness_profile(mutated)


def test_required_market_count_is_monotonic() -> None:
    smaller_effect = readiness.required_market_count(
        standard_deviation=0.05,
        mean_effect=0.005,
        one_sided_alpha=0.05,
        power=0.8,
        robustness_inflation_factor=1.25,
    )
    larger_effect = readiness.required_market_count(
        standard_deviation=0.05,
        mean_effect=0.01,
        one_sided_alpha=0.05,
        power=0.8,
        robustness_inflation_factor=1.25,
    )
    assert smaller_effect > larger_effect
    assert larger_effect == 194


def test_reports_use_future_windows_for_power_and_historical_for_lineage() -> None:
    result = readiness.build_paper_readiness_reports(
        issue238_window=_future(
            "issue238",
            start=100,
            pnl=[0.08, -0.04, 0.0, 0.03],
        ),
        issue250_window=_future(
            "issue250",
            start=1_000,
            pnl=[0.05, -0.03, 0.0, 0.02],
        ),
        historical_inventory=_historical(),
        profile=_profile(),
        report_created_ts=2_000,
    )
    inventory = result["inventory"]
    power = result["power_report"]
    gate = result["gate_plan"]
    assert inventory["future_quality_valid_market_count"] == 8
    assert inventory["future_guard_blocked_zero_market_count"] == 2
    assert inventory["future_market_overlap_count"] == 0
    assert inventory["evidence_inventory_passed"] is True
    assert power["planning_market_count"] == 8
    assert power["completed_future_outcomes_used_for_variance_planning_only"] is True
    assert power[
        "completed_future_outcomes_used_for_model_or_threshold_tuning"
    ] is False
    assert gate["side_quota_enabled"] is False
    assert gate["paper_candidate_allowed"] is False
    assert gate["paper_candidate_auto_unlock_allowed"] is False


def test_overlapping_future_windows_fail_closed() -> None:
    first = _future("issue238", start=100, pnl=[0.01, 0.02])
    second = _future("issue250", start=1_000, pnl=[0.01, 0.02])
    second["market_ids"][0] = first["market_ids"][0]
    with pytest.raises(ValueError, match="market_overlap"):
        readiness.build_paper_readiness_reports(
            issue238_window=first,
            issue250_window=second,
            historical_inventory=_historical(),
            profile=_profile(),
            report_created_ts=2_000,
        )


def test_non_chronological_future_windows_fail_closed() -> None:
    with pytest.raises(ValueError, match="not_strictly_chronological"):
        readiness.build_paper_readiness_reports(
            issue238_window=_future(
                "issue238",
                start=1_000,
                pnl=[0.01, 0.02],
            ),
            issue250_window=_future(
                "issue250",
                start=100,
                pnl=[0.01, 0.02],
            ),
            historical_inventory=_historical(),
            profile=_profile(),
            report_created_ts=2_000,
        )


def test_forward_gate_is_diagnostic_and_single_use() -> None:
    result = readiness.build_paper_readiness_reports(
        issue238_window=_future(
            "issue238",
            start=100,
            pnl=[0.01, 0.02, -0.01],
        ),
        issue250_window=_future(
            "issue250",
            start=1_000,
            pnl=[0.03, -0.02, 0.0],
        ),
        historical_inventory=_historical(),
        profile=_profile(),
        report_created_ts=2_000,
    )
    gate = result["gate_plan"]
    assert gate["one_evaluation_only"] is True
    assert gate["result_selected_rerun_allowed"] is False
    assert gate["result_selected_extension_allowed"] is False
    assert gate["separate_manual_paper_authorization_issue_required"] is True
    for key, value in readiness.SAFETY.items():
        assert gate[key] == value
