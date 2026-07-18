"""Prospective side-only accepted-bet power analysis for future v8 designs."""

from __future__ import annotations

import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_accepted_bet_power import (
    required_independent_market_count,
)
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb import (
    _blocked_safety_fields,
    _descriptor,
    _load_json,
    _require_sha256,
    _sha256_file,
    _verify_pin,
    _write_json,
    _write_text,
)

DESIGN_SCHEMA_VERSION = "bigan-v8-side-only-accepted-bet-power-design-v1"
REPORT_SCHEMA_VERSION = "bigan-v8-side-only-accepted-bet-power-analysis-report-v1"
MANIFEST_SCHEMA_VERSION = "bigan-v8-side-only-accepted-bet-power-analysis-manifest-v1"


def run_side_only_accepted_bet_power_analysis(
    *,
    run_id: str,
    output_dir: Path | str,
    design_path: Path | str,
    expected_design_sha256: str,
) -> dict[str, Any]:
    """Simulate the frozen side-only hard gate without reading result artifacts."""

    if not run_id.strip():
        raise ValueError("run_id is required")
    _require_sha256(expected_design_sha256, name="side-only power design SHA-256")
    design_path = Path(design_path).resolve()
    _verify_pin(design_path, expected_design_sha256, name="side-only power design")
    design = _load_json(design_path)
    validate_side_only_accepted_bet_power_design(design)

    rows: list[dict[str, Any]] = []
    for support in design["accepted_unique_market_support_grid"]:
        for effect in design["standardized_effect_size_scenarios"]:
            for distribution in design["market_pnl_distribution_scenarios"]:
                for up_fraction in design["up_market_fraction_scenarios"]:
                    rows.append(
                        _simulate_scenario(
                            design=design,
                            accepted_market_count=int(support),
                            standardized_effect_size=float(effect),
                            distribution=str(distribution),
                            up_fraction=float(up_fraction),
                        )
                    )

    planning_effect = float(design["planning_scenario"]["standardized_effect_size"])
    planning_rows = [
        row
        for row in rows
        if row["standardized_effect_size"] == planning_effect
    ]
    support_grid = [int(value) for value in design["accepted_unique_market_support_grid"]]
    target_power = float(design["target_power"])
    recommended_accepted = next(
        (
            support
            for support in support_grid
            if min(
                row["combined_side_only_hard_gate_power_wilson_lower_bound"]
                for row in planning_rows
                if row["accepted_unique_market_count"] == support
            )
            >= target_power
        ),
        None,
    )
    recommendation_within_grid = recommended_accepted is not None

    alpha = float(design["one_sided_alpha"])
    directional_side_required = math.ceil(
        (NormalDist().inv_cdf(target_power) / planning_effect) ** 2
        * float(design["side_robustness_inflation_factor"])
    )
    confidence_side_required = math.ceil(
        required_independent_market_count(
            alpha=alpha,
            power=target_power,
            standardized_effect_size=planning_effect,
        )
        * float(design["side_robustness_inflation_factor"])
    )
    if recommended_accepted is None:
        recommended_accepted = support_grid[-1]

    sizing_scenarios = []
    for accepted_rate in design["execution_acceptance_rate_scenarios"]:
        quality_valid = math.ceil(recommended_accepted / float(accepted_rate))
        for quality_rate in design["capture_quality_rate_scenarios"]:
            sizing_scenarios.append(
                {
                    "execution_acceptance_rate": float(accepted_rate),
                    "capture_quality_rate": float(quality_rate),
                    "required_quality_valid_market_count": quality_valid,
                    "required_capture_attempt_count": math.ceil(
                        quality_valid / float(quality_rate)
                    ),
                }
            )

    current_reference = dict(design["current_204_reference"])
    current_rows = [
        row
        for row in planning_rows
        if row["accepted_unique_market_count"]
        == int(current_reference["minimum_accepted_unique_market_count"])
    ]
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "design": _descriptor(design_path),
        "assumption_config_sha256": canonical_json_sha256(design),
        "analysis_method": (
            "deterministic_market_level_monte_carlo_with_multinomial_group_bootstrap"
        ),
        "statistical_unit": "unique_execution_layer_accepted_market",
        "prospective_effect_size_scenarios": list(
            design["standardized_effect_size_scenarios"]
        ),
        "market_level_variance_tail_scenarios": list(
            design["market_pnl_distribution_scenarios"]
        ),
        "accepted_market_support_grid": support_grid,
        "side_support_grid": sorted(
            {
                row["up_market_count"] for row in rows
            }
            | {row["down_market_count"] for row in rows}
        ),
        "execution_acceptance_rate_grid": list(
            design["execution_acceptance_rate_scenarios"]
        ),
        "hard_gate_components": list(design["hard_gate_components"]),
        "power_by_scenario": rows,
        "current_204_reference": current_reference,
        "current_204_planning_effect_power_range": {
            "minimum_combined_power": min(
                row["combined_side_only_hard_gate_power"] for row in current_rows
            ),
            "maximum_combined_power": max(
                row["combined_side_only_hard_gate_power"] for row in current_rows
            ),
            "minimum_combined_power_wilson_lower_bound": min(
                row["combined_side_only_hard_gate_power_wilson_lower_bound"]
                for row in current_rows
            ),
            "maximum_combined_power_wilson_lower_bound": max(
                row["combined_side_only_hard_gate_power_wilson_lower_bound"]
                for row in current_rows
            ),
            "target_power": target_power,
            "target_power_met_conservatively": min(
                row["combined_side_only_hard_gate_power_wilson_lower_bound"]
                for row in current_rows
            )
            >= target_power,
            "diagnostic_only_not_used_to_change_204_gate": True,
        },
        "recommended_minimum_accepted_unique_markets": recommended_accepted,
        "recommended_minimum_buy_up_accepted_markets": directional_side_required,
        "recommended_minimum_buy_down_accepted_markets": directional_side_required,
        "recommended_directional_side_support_method": (
            "ceil((z_target_power / standardized_effect_size)^2 * "
            "side_robustness_inflation_factor)"
        ),
        "diagnostic_one_sided_confidence_minimum_market_count_per_side": (
            confidence_side_required
        ),
        "recommendation_reaches_target_power_within_reported_grid": (
            recommendation_within_grid
        ),
        "recommendation_uses_monte_carlo_confidence_lower_bound": True,
        "attempted_market_sizing_scenarios": sizing_scenarios,
        "uses_204_outcomes_for_planning": False,
        "uses_current_oof_validation_or_confirmatory_pnl": False,
        "uses_realized_candidate_pnl_for_design": False,
        "changes_204_gate": False,
        "result_dependent_extension_allowed": False,
        "action_and_action_family_pnl_diagnostic_only": True,
        "power_analysis_ready": True,
        "blocking_reason_codes": [],
        **_blocked_safety_fields(),
    }

    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    report_path = run_dir / "execution_layer_v2_accepted_bet_power_analysis_report.json"
    markdown_path = run_dir / "execution_layer_v2_accepted_bet_power_analysis_report.md"
    _write_json(report_path, report)
    _write_text(markdown_path, _markdown(report))
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "design": _descriptor(design_path),
        "report": _descriptor(report_path),
        "markdown": _descriptor(markdown_path),
        "assumption_config_sha256": report["assumption_config_sha256"],
        "recommended_minimum_accepted_unique_markets": recommended_accepted,
        "recommended_minimum_buy_up_accepted_markets": directional_side_required,
        "recommended_minimum_buy_down_accepted_markets": directional_side_required,
        "uses_204_outcomes_for_planning": False,
        "changes_204_gate": False,
        "power_analysis_ready": True,
        "blocking_reason_codes": [],
        **_blocked_safety_fields(),
    }
    manifest["power_analysis_manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "execution_layer_v2_accepted_bet_power_analysis_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "report": report,
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "markdown_path": markdown_path,
        "markdown_sha256": _sha256_file(markdown_path),
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def validate_side_only_accepted_bet_power_design(design: dict[str, Any]) -> None:
    """Fail closed on assumption drift or any outcome-dependent design input."""

    expected = {
        "schema_version": DESIGN_SCHEMA_VERSION,
        "frozen": True,
        "one_sided_alpha": 0.05,
        "target_power": 0.9,
        "simulation_repetitions": 256,
        "bootstrap_repetitions": 256,
        "monte_carlo_confidence_level": 0.95,
        "random_seed": 205,
        "accepted_unique_market_support_grid": [60, 88, 120, 160, 220, 320],
        "standardized_effect_size_scenarios": [0.2, 0.35, 0.5],
        "market_pnl_distribution_scenarios": ["normal", "student_t_df5"],
        "up_market_fraction_scenarios": [0.35, 0.5, 0.65],
        "execution_acceptance_rate_scenarios": [0.4, 0.5, 0.6],
        "capture_quality_rate_scenarios": [0.65, 0.7, 0.73],
        "minimum_side_support": 10,
        "side_robustness_inflation_factor": 1.25,
        "candidate_pnl_and_delta_correlation": 0.5,
        "uses_204_outcomes_for_planning": False,
        "uses_current_oof_validation_or_confirmatory_pnl": False,
        "uses_realized_candidate_pnl_for_design": False,
        "changes_204_gate": False,
        "result_dependent_extension_allowed": False,
    }
    blockers = [key for key, value in expected.items() if design.get(key) != value]
    expected_components = [
        "total_candidate_post_cost_pnl_positive",
        "buy_up_post_cost_pnl_positive",
        "buy_down_post_cost_pnl_positive",
        "candidate_minus_matched_baseline_positive",
        "market_grouped_bootstrap_delta_lcb_positive",
        "largest_winning_market_removed_pnl_positive",
    ]
    if design.get("hard_gate_components") != expected_components:
        blockers.append("hard_gate_components")
    if design.get("planning_scenario") != {
        "standardized_effect_size": 0.35,
        "description": "prospective_minimum_relevant_effect_not_fitted_from_results",
    }:
        blockers.append("planning_scenario")
    if design.get("current_204_reference") != {
        "quality_valid_market_count": 220,
        "minimum_accepted_unique_market_count": 88,
        "minimum_side_market_count": 10,
        "outcomes_used_for_this_design": False,
        "changes_allowed": False,
    }:
        blockers.append("current_204_reference")
    if any(design.get(key) != value for key, value in _blocked_safety_fields().items()):
        blockers.append("safety")
    if blockers:
        raise ValueError("side-only power design validation failed: " + ", ".join(blockers))


def load_and_validate_side_only_accepted_bet_power_manifest(
    path: Path | str,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the frozen power artifact before it can size a future window."""

    path = Path(path).resolve()
    _verify_pin(path, expected_sha256, name="side-only accepted-bet power manifest")
    manifest = _load_json(path)
    blockers: list[str] = []
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        blockers.append("side_only_power_manifest_schema_invalid")
    expected_id = canonical_json_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key != "power_analysis_manifest_id"
        }
    )
    if manifest.get("power_analysis_manifest_id") != expected_id:
        blockers.append("side_only_power_manifest_id_mismatch")

    expected_manifest = {
        "uses_204_outcomes_for_planning": False,
        "changes_204_gate": False,
        "power_analysis_ready": True,
        "blocking_reason_codes": [],
        **_blocked_safety_fields(),
    }
    blockers.extend(
        f"side_only_power_manifest_{key}_invalid"
        for key, value in expected_manifest.items()
        if manifest.get(key) != value
    )

    descriptors: dict[str, dict[str, str]] = {}
    for field in ("design", "report", "markdown"):
        descriptor = manifest.get(field)
        try:
            if not isinstance(descriptor, dict):
                raise ValueError("descriptor must be an object")
            descriptor_path = Path(str(descriptor.get("path") or "")).resolve()
            descriptor_sha256 = str(descriptor.get("sha256") or "")
            _verify_pin(
                descriptor_path,
                descriptor_sha256,
                name=f"side-only accepted-bet power {field}",
            )
            descriptors[field] = {
                "path": str(descriptor_path),
                "sha256": descriptor_sha256.lower(),
            }
        except (OSError, TypeError, ValueError) as exc:
            blockers.append(f"side_only_power_{field}_descriptor_invalid:{exc}")

    if "design" in descriptors:
        try:
            design = _load_json(Path(descriptors["design"]["path"]))
            validate_side_only_accepted_bet_power_design(design)
            if manifest.get("assumption_config_sha256") != canonical_json_sha256(design):
                blockers.append("side_only_power_assumption_hash_mismatch")
        except ValueError as exc:
            blockers.append(f"side_only_power_design_invalid:{exc}")

    report: dict[str, Any] = {}
    if "report" in descriptors:
        report = _load_json(Path(descriptors["report"]["path"]))
        expected_report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "uses_204_outcomes_for_planning": False,
            "changes_204_gate": False,
            "power_analysis_ready": True,
            "blocking_reason_codes": [],
            **_blocked_safety_fields(),
        }
        blockers.extend(
            f"side_only_power_report_{key}_invalid"
            for key, value in expected_report.items()
            if report.get(key) != value
        )
        if report.get("design") != descriptors.get("design"):
            blockers.append("side_only_power_report_design_lineage_mismatch")
        for field in (
            "recommended_minimum_accepted_unique_markets",
            "recommended_minimum_buy_up_accepted_markets",
            "recommended_minimum_buy_down_accepted_markets",
        ):
            if manifest.get(field) != report.get(field):
                blockers.append(f"side_only_power_{field}_mismatch")

    if blockers:
        raise ValueError(
            "side-only accepted-bet power manifest validation failed: "
            + ", ".join(sorted(set(blockers)))
        )
    return manifest, {
        "power_analysis_manifest": _descriptor(path),
        "power_analysis_report": descriptors["report"],
        "power_analysis_ready": True,
        "recommended_minimum_accepted_unique_markets": report[
            "recommended_minimum_accepted_unique_markets"
        ],
        "recommended_minimum_buy_up_accepted_markets": report[
            "recommended_minimum_buy_up_accepted_markets"
        ],
        "recommended_minimum_buy_down_accepted_markets": report[
            "recommended_minimum_buy_down_accepted_markets"
        ],
        "uses_204_outcomes_for_planning": False,
        "changes_204_gate": False,
    }


def _simulate_scenario(
    *,
    design: dict[str, Any],
    accepted_market_count: int,
    standardized_effect_size: float,
    distribution: str,
    up_fraction: float,
) -> dict[str, Any]:
    simulations = int(design["simulation_repetitions"])
    bootstraps = int(design["bootstrap_repetitions"])
    alpha = float(design["one_sided_alpha"])
    n_up = round(accepted_market_count * up_fraction)
    n_down = accepted_market_count - n_up
    scenario_seed = int(
        canonical_json_sha256(
            {
                "base_seed": design["random_seed"],
                "accepted_market_count": accepted_market_count,
                "standardized_effect_size": standardized_effect_size,
                "distribution": distribution,
                "up_fraction": up_fraction,
            }
        )[:16],
        16,
    )
    rng = np.random.default_rng(scenario_seed)
    candidate_noise = _unit_variance_noise(
        rng, distribution=distribution, size=(simulations, accepted_market_count)
    )
    independent_noise = _unit_variance_noise(
        rng, distribution=distribution, size=(simulations, accepted_market_count)
    )
    correlation = float(design["candidate_pnl_and_delta_correlation"])
    delta_noise = (
        correlation * candidate_noise
        + math.sqrt(1.0 - correlation**2) * independent_noise
    )
    candidate_pnl = standardized_effect_size + candidate_noise
    delta_pnl = standardized_effect_size + delta_noise

    weights = rng.multinomial(
        accepted_market_count,
        np.full(accepted_market_count, 1.0 / accepted_market_count),
        size=bootstraps,
    )
    bootstrap_delta_means = delta_pnl @ weights.T / accepted_market_count
    bootstrap_lcb = np.quantile(
        bootstrap_delta_means, alpha, axis=1, method="lower"
    )

    total_positive = candidate_pnl.sum(axis=1) > 0.0
    up_positive = candidate_pnl[:, :n_up].sum(axis=1) > 0.0
    down_positive = candidate_pnl[:, n_up:].sum(axis=1) > 0.0
    delta_positive = delta_pnl.sum(axis=1) > 0.0
    bootstrap_positive = bootstrap_lcb > 0.0
    largest_removed_positive = (
        candidate_pnl.sum(axis=1) - candidate_pnl.max(axis=1)
    ) > 0.0
    support_passed = (
        n_up >= int(design["minimum_side_support"])
        and n_down >= int(design["minimum_side_support"])
    )
    combined = (
        total_positive
        & up_positive
        & down_positive
        & delta_positive
        & bootstrap_positive
        & largest_removed_positive
        & support_passed
    )
    combined_success_count = int(combined.sum())
    combined_power = combined_success_count / simulations
    combined_power_lcb = _wilson_lower_bound(
        success_count=combined_success_count,
        trial_count=simulations,
        confidence_level=float(design["monte_carlo_confidence_level"]),
    )
    return {
        "accepted_unique_market_count": accepted_market_count,
        "up_market_fraction": up_fraction,
        "up_market_count": n_up,
        "down_market_count": n_down,
        "standardized_effect_size": standardized_effect_size,
        "market_pnl_distribution": distribution,
        "simulation_repetitions": simulations,
        "bootstrap_repetitions": bootstraps,
        "side_support_passed": support_passed,
        "total_candidate_post_cost_pnl_positive_power": float(total_positive.mean()),
        "buy_up_post_cost_pnl_positive_power": float(up_positive.mean()),
        "buy_down_post_cost_pnl_positive_power": float(down_positive.mean()),
        "candidate_minus_matched_baseline_positive_power": float(delta_positive.mean()),
        "market_grouped_bootstrap_delta_lcb_positive_power": float(
            bootstrap_positive.mean()
        ),
        "largest_winning_market_removed_pnl_positive_power": float(
            largest_removed_positive.mean()
        ),
        "combined_side_only_hard_gate_power": combined_power,
        "combined_side_only_hard_gate_power_wilson_lower_bound": (
            combined_power_lcb
        ),
        "market_grouped_bootstrap_used": True,
        "one_row_per_unique_market": True,
    }


def _unit_variance_noise(
    rng: np.random.Generator, *, distribution: str, size: tuple[int, int]
) -> np.ndarray:
    if distribution == "normal":
        return rng.standard_normal(size)
    if distribution == "student_t_df5":
        return rng.standard_t(5, size=size) / math.sqrt(5.0 / 3.0)
    raise ValueError(f"unsupported market PnL distribution: {distribution}")


def _wilson_lower_bound(
    *, success_count: int, trial_count: int, confidence_level: float
) -> float:
    if not 0 <= success_count <= trial_count or trial_count <= 0:
        raise ValueError("invalid Monte Carlo success/trial counts")
    z = NormalDist().inv_cdf(confidence_level)
    p_hat = success_count / trial_count
    denominator = 1.0 + z**2 / trial_count
    center = p_hat + z**2 / (2.0 * trial_count)
    margin = z * math.sqrt(
        p_hat * (1.0 - p_hat) / trial_count
        + z**2 / (4.0 * trial_count**2)
    )
    return (center - margin) / denominator


def _markdown(report: dict[str, Any]) -> str:
    reference = report["current_204_planning_effect_power_range"]
    return "\n".join(
        [
            "# #205 Side-Only Accepted-Bet Power Analysis",
            "",
            f"- method: `{report['analysis_method']}`",
            f"- prospective effects: `{report['prospective_effect_size_scenarios']}`",
            f"- current #204 combined-power range: `{reference}`",
            f"- recommended accepted unique markets: `{report['recommended_minimum_accepted_unique_markets']}`",
            f"- recommended BUY_UP markets: `{report['recommended_minimum_buy_up_accepted_markets']}`",
            f"- recommended BUY_DOWN markets: `{report['recommended_minimum_buy_down_accepted_markets']}`",
            f"- diagnostic per-side confidence support: `{report['diagnostic_one_sided_confidence_minimum_market_count_per_side']}`",
            f"- reaches target in grid: `{report['recommendation_reaches_target_power_within_reported_grid']}`",
            "- #204 outcomes used for planning: `false`",
            "- #204 gate changed: `false`",
            "- action/family PnL blocker: `false`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )
