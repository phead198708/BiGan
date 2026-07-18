from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_side_only_accepted_bet_power import (
    load_and_validate_side_only_accepted_bet_power_manifest,
    run_side_only_accepted_bet_power_analysis,
    validate_side_only_accepted_bet_power_design,
)

DESIGN_PATH = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_side_only_accepted_bet_power_v1.json"
)


def test_design_is_prospective_side_only_and_does_not_change_204() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))

    validate_side_only_accepted_bet_power_design(design)

    assert design["uses_204_outcomes_for_planning"] is False
    assert design["changes_204_gate"] is False
    assert design["hard_gate_components"] == [
        "total_candidate_post_cost_pnl_positive",
        "buy_up_post_cost_pnl_positive",
        "buy_down_post_cost_pnl_positive",
        "candidate_minus_matched_baseline_positive",
        "market_grouped_bootstrap_delta_lcb_positive",
        "largest_winning_market_removed_pnl_positive",
    ]
    assert design["paper_only"] is True
    assert design["capital_at_risk"] is False


def test_design_fails_closed_on_204_result_usage() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    design["uses_204_outcomes_for_planning"] = True

    with pytest.raises(ValueError, match="uses_204_outcomes_for_planning"):
        validate_side_only_accepted_bet_power_design(design)


def test_power_analysis_is_deterministic_market_grouped_and_side_specific(
    tmp_path: Path,
) -> None:
    first = run_side_only_accepted_bet_power_analysis(
        run_id="side-only-power",
        output_dir=tmp_path / "first",
        design_path=DESIGN_PATH,
        expected_design_sha256=_sha256(DESIGN_PATH),
    )
    second = run_side_only_accepted_bet_power_analysis(
        run_id="side-only-power",
        output_dir=tmp_path / "second",
        design_path=DESIGN_PATH,
        expected_design_sha256=_sha256(DESIGN_PATH),
    )
    report = first["report"]

    assert first["report_sha256"] == second["report_sha256"]
    assert report["statistical_unit"] == "unique_execution_layer_accepted_market"
    assert report["uses_204_outcomes_for_planning"] is False
    assert report["changes_204_gate"] is False
    assert report["recommended_minimum_buy_up_accepted_markets"] > 10
    assert report["recommended_minimum_buy_down_accepted_markets"] > 10
    assert (
        report["recommended_minimum_buy_up_accepted_markets"]
        + report["recommended_minimum_buy_down_accepted_markets"]
        <= report["recommended_minimum_accepted_unique_markets"]
    )
    assert (
        report["diagnostic_one_sided_confidence_minimum_market_count_per_side"]
        > report["recommended_minimum_buy_up_accepted_markets"]
    )
    assert report["recommendation_uses_monte_carlo_confidence_lower_bound"] is True
    assert all(row["market_grouped_bootstrap_used"] for row in report["power_by_scenario"])
    assert all(row["one_row_per_unique_market"] for row in report["power_by_scenario"])
    current = report["current_204_planning_effect_power_range"]
    assert 0.0 <= current["minimum_combined_power"] <= 1.0
    assert current["minimum_combined_power_wilson_lower_bound"] < 0.9
    assert current["target_power_met_conservatively"] is False
    assert current["diagnostic_only_not_used_to_change_204_gate"] is True
    assert report["source_model_candidate_eligible"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["v8_execution_handoff_allowed"] is False

    _, audit = load_and_validate_side_only_accepted_bet_power_manifest(
        first["manifest_path"], first["manifest_sha256"]
    )
    assert audit["recommended_minimum_accepted_unique_markets"] == report[
        "recommended_minimum_accepted_unique_markets"
    ]
    assert audit["uses_204_outcomes_for_planning"] is False


def test_manifest_validation_fails_closed_on_tampered_report(tmp_path: Path) -> None:
    result = run_side_only_accepted_bet_power_analysis(
        run_id="side-only-power-tamper",
        output_dir=tmp_path,
        design_path=DESIGN_PATH,
        expected_design_sha256=_sha256(DESIGN_PATH),
    )
    report_path = Path(result["report_path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["uses_204_outcomes_for_planning"] = True
    report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="descriptor_invalid"):
        load_and_validate_side_only_accepted_bet_power_manifest(
            result["manifest_path"], result["manifest_sha256"]
        )


def test_heavier_tail_or_side_imbalance_does_not_inflate_reported_support() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    assert design["market_pnl_distribution_scenarios"] == [
        "normal",
        "student_t_df5",
    ]
    assert design["up_market_fraction_scenarios"] == [0.35, 0.5, 0.65]
    assert design["accepted_unique_market_support_grid"][1] == 88


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
