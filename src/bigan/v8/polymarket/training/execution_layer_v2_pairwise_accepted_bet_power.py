"""Prospective independent-market power design for #190 future holdout."""

from __future__ import annotations

import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
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

DESIGN_SCHEMA_VERSION = "bigan-v8-pairwise-accepted-bet-power-design-v1"
REPORT_SCHEMA_VERSION = "bigan-v8-pairwise-accepted-bet-power-analysis-report-v1"
MANIFEST_SCHEMA_VERSION = "bigan-v8-pairwise-accepted-bet-power-analysis-manifest-v1"


def run_pairwise_accepted_bet_power_analysis(
    *,
    run_id: str,
    output_dir: Path | str,
    design_path: Path | str,
    expected_design_sha256: str,
) -> dict[str, Any]:
    """Calculate a prospective design without reading any model-result artifact."""

    if not run_id.strip():
        raise ValueError("run_id is required")
    _require_sha256(expected_design_sha256, name="power design SHA-256")
    design_path = Path(design_path).resolve()
    _verify_pin(design_path, expected_design_sha256, name="power design")
    design = _load_json(design_path)
    validate_pairwise_accepted_bet_power_design(design)

    alpha = float(design["one_sided_alpha"])
    inflation = float(design["robustness_inflation_factor"])
    rows: list[dict[str, Any]] = []
    for power in design["reported_power_levels"]:
        for effect_size in design["reported_standardized_effect_sizes"]:
            base = required_independent_market_count(
                alpha=alpha,
                power=float(power),
                standardized_effect_size=float(effect_size),
            )
            rows.append(
                {
                    "target_power": float(power),
                    "standardized_effect_size": float(effect_size),
                    "base_required_accepted_unique_market_count": base,
                    "robustness_inflated_required_accepted_unique_market_count": (
                        math.ceil(base * inflation)
                    ),
                }
            )

    recommended = dict(design["recommended_design"])
    selected_row = next(
        row
        for row in rows
        if row["target_power"] == float(recommended["target_power"])
        and row["standardized_effect_size"]
        == float(recommended["minimum_relevant_standardized_effect_size"])
    )
    required_accepted = int(
        selected_row["robustness_inflated_required_accepted_unique_market_count"]
    )
    sizing_scenarios = []
    for accepted_rate in design["accepted_market_rate_scenarios"]:
        valid_markets = math.ceil(required_accepted / float(accepted_rate))
        for quality_rate in design["capture_quality_rate_scenarios"]:
            sizing_scenarios.append(
                {
                    "accepted_market_rate": float(accepted_rate),
                    "capture_quality_rate": float(quality_rate),
                    "required_quality_valid_market_count": valid_markets,
                    "required_capture_attempt_count": math.ceil(
                        valid_markets / float(quality_rate)
                    ),
                }
            )
    planning_accepted_rate = float(recommended["planning_accepted_market_rate"])
    planning_quality_rate = float(recommended["planning_capture_quality_rate"])
    recommended_valid_count = math.ceil(required_accepted / planning_accepted_rate)
    unrounded_attempt_cap = math.ceil(recommended_valid_count / planning_quality_rate)
    attempt_cap = _round_up(
        unrounded_attempt_cap,
        int(recommended["maximum_capture_attempt_rounding_multiple"]),
    )
    current = dict(design["current_design_reference"])
    current_accepted = int(current["planning_accepted_unique_market_count"])
    current_detectable = {
        str(power): detectable_standardized_effect_size(
            alpha=alpha,
            power=float(power),
            independent_market_count=current_accepted,
            robustness_inflation_factor=inflation,
        )
        for power in design["reported_power_levels"]
    }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "design": _descriptor(design_path),
        "design_config_hash": canonical_json_sha256(design),
        "formula": "ceil(((z_1_minus_alpha + z_power) / standardized_effect_size) ** 2)",
        "statistical_unit": design["statistical_unit"],
        "primary_estimand": design["primary_estimand"],
        "sample_size_table": rows,
        "sizing_scenarios": sizing_scenarios,
        "recommended_target_power": float(recommended["target_power"]),
        "recommended_minimum_relevant_standardized_effect_size": float(
            recommended["minimum_relevant_standardized_effect_size"]
        ),
        "recommended_required_accepted_unique_market_count": required_accepted,
        "recommended_quality_valid_market_count": recommended_valid_count,
        "recommended_maximum_capture_attempt_count": attempt_cap,
        "recommended_stop_rule": (
            f"earliest_{recommended_valid_count}_quality_valid_markets_or_max_"
            f"{attempt_cap}_attempts"
        ),
        "current_design_reference": current,
        "current_design_detectable_standardized_effect_size_by_power": (
            current_detectable
        ),
        "current_60_market_design_has_limited_power": True,
        "uses_current_oof_validation_or_confirmatory_pnl": False,
        "uses_realized_candidate_pnl_for_design": False,
        "result_dependent_extension_allowed": False,
        "power_analysis_ready": True,
        "blocking_reason_codes": [],
        **_blocked_safety_fields(),
    }
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    report_path = run_dir / "pairwise_accepted_bet_power_analysis_report.json"
    markdown_path = run_dir / "pairwise_accepted_bet_power_analysis_report.md"
    _write_json(report_path, report)
    _write_text(markdown_path, _markdown(report))
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "design": _descriptor(design_path),
        "report": _descriptor(report_path),
        "markdown": _descriptor(markdown_path),
        "power_analysis_ready": True,
        "recommended_quality_valid_market_count": recommended_valid_count,
        "recommended_maximum_capture_attempt_count": attempt_cap,
        "uses_current_oof_validation_or_confirmatory_pnl": False,
        "uses_realized_candidate_pnl_for_design": False,
        "result_dependent_extension_allowed": False,
        "blocking_reason_codes": [],
        **_blocked_safety_fields(),
    }
    manifest["power_analysis_manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "pairwise_accepted_bet_power_analysis_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "report": report,
        "manifest": manifest,
    }


def validate_pairwise_accepted_bet_power_design(design: dict[str, Any]) -> None:
    expected = {
        "schema_version": DESIGN_SCHEMA_VERSION,
        "frozen": True,
        "statistical_unit": "unique_accepted_bet_market",
        "one_sided_alpha": 0.05,
        "reported_power_levels": [0.8, 0.9],
        "reported_standardized_effect_sizes": [0.2, 0.25, 0.3, 0.35, 0.4],
        "accepted_market_rate_scenarios": [0.4, 0.5, 0.6],
        "capture_quality_rate_scenarios": [0.65, 0.7, 0.73],
        "robustness_inflation_factor": 1.25,
        "uses_current_oof_validation_or_confirmatory_pnl": False,
        "uses_realized_candidate_pnl_for_design": False,
        "result_dependent_extension_allowed": False,
    }
    blockers = [key for key, value in expected.items() if design.get(key) != value]
    recommendation = dict(design.get("recommended_design") or {})
    if recommendation != {
        "target_power": 0.9,
        "minimum_relevant_standardized_effect_size": 0.35,
        "planning_accepted_market_rate": 0.4,
        "planning_capture_quality_rate": 0.65,
        "maximum_capture_attempt_rounding_multiple": 10,
    }:
        blockers.append("recommended_design")
    if any(design.get(key) != value for key, value in _blocked_safety_fields().items()):
        blockers.append("safety")
    if blockers:
        raise ValueError("power design validation failed: " + ", ".join(blockers))


def required_independent_market_count(
    *, alpha: float, power: float, standardized_effect_size: float
) -> int:
    if not 0 < alpha < 0.5 or not 0.5 < power < 1.0:
        raise ValueError("alpha/power are outside the supported range")
    if standardized_effect_size <= 0:
        raise ValueError("standardized_effect_size must be positive")
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1.0 - alpha)
    z_power = normal.inv_cdf(power)
    return math.ceil(((z_alpha + z_power) / standardized_effect_size) ** 2)


def detectable_standardized_effect_size(
    *,
    alpha: float,
    power: float,
    independent_market_count: int,
    robustness_inflation_factor: float,
) -> float:
    effective_n = independent_market_count / robustness_inflation_factor
    normal = NormalDist()
    return (normal.inv_cdf(1.0 - alpha) + normal.inv_cdf(power)) / math.sqrt(
        effective_n
    )


def _round_up(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("rounding multiple must be positive")
    return math.ceil(value / multiple) * multiple


def _markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# #191 Prospective Accepted-Bet Power Analysis",
            "",
            f"- statistical unit: `{report['statistical_unit']}`",
            f"- target power: `{report['recommended_target_power']}`",
            f"- minimum relevant standardized effect: `{report['recommended_minimum_relevant_standardized_effect_size']}`",
            f"- required accepted unique markets: `{report['recommended_required_accepted_unique_market_count']}`",
            f"- recommended quality-valid markets: `{report['recommended_quality_valid_market_count']}`",
            f"- hard capture-attempt cap: `{report['recommended_maximum_capture_attempt_count']}`",
            f"- current design detectable effects: `{report['current_design_detectable_standardized_effect_size_by_power']}`",
            "- current OOF/validation/confirmatory PnL used: `false`",
            "- result-dependent extension: `false`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )
