"""Powered paper-candidate readiness design for the retained v6.7 champion."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist, mean, stdev
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
    _descriptor,
    _load_json,
    _load_jsonl,
    _require_git_sha,
    _require_sha256,
    _verify_pin,
    _write_json,
    _write_text,
)

PROFILE_SCHEMA_VERSION = "bigan-v8-retained-v6-7-paper-readiness-profile-v1"
INVENTORY_SCHEMA_VERSION = "bigan-v8-v6-7-champion-evidence-inventory-v1"
POWER_SCHEMA_VERSION = "bigan-v8-v6-7-paper-readiness-power-report-v1"
GATE_PLAN_SCHEMA_VERSION = "bigan-v8-v6-7-forward-paper-candidate-gate-plan-v1"
MANIFEST_SCHEMA_VERSION = "bigan-v8-v6-7-paper-readiness-manifest-v1"
ISSUE238_MANIFEST_SCHEMA = (
    "bigan-v8-retained-v6-7-future-noninferiority-v7-5-"
    "future-pnl-evaluation-manifest-v1"
)
ISSUE250_MANIFEST_SCHEMA = (
    "bigan-v8-non-risk-abstention-fallback-v8-3-"
    "future-holdout-pnl-gate-manifest-v1"
)
ISSUE251_MANIFEST_SCHEMA = (
    "bigan-v8-baseline-anchored-side-switch-v8-4-manifest-v1"
)
SAFETY = {
    "paper_only": True,
    "paper_candidate_allowed": False,
    "paper_loop_allowed": False,
    "live_trading_enabled": False,
    "capital_at_risk": False,
    "polymarket_write_enabled": False,
    "wallet_signing_enabled": False,
    "v8_execution_handoff_allowed": False,
    "source_model_candidate_eligible": False,
    "freeze_ready": False,
    "promotion_evidence_eligible": False,
    "#134_resume_allowed": False,
    "#146_start_allowed": False,
}


@dataclass(frozen=True, slots=True)
class RetainedV67PaperReadinessConfig:
    """Pinned inputs for issue #252 evidence and power planning."""

    run_id: str
    output_dir: Path | str
    profile_path: Path | str
    expected_profile_sha256: str
    issue238_manifest_path: Path | str
    expected_issue238_manifest_sha256: str
    issue250_manifest_path: Path | str
    expected_issue250_manifest_sha256: str
    issue251_manifest_path: Path | str
    expected_issue251_manifest_sha256: str
    implementation_commit: str
    report_created_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip() or self.report_created_ts <= 0:
            raise ValueError("#252 run id and positive report timestamp are required")
        _require_git_sha(self.implementation_commit)
        for name in (
            "expected_profile_sha256",
            "expected_issue238_manifest_sha256",
            "expected_issue250_manifest_sha256",
            "expected_issue251_manifest_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        for name in (
            "output_dir",
            "profile_path",
            "issue238_manifest_path",
            "issue250_manifest_path",
            "issue251_manifest_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


def validate_retained_v6_7_paper_readiness_profile(
    profile: dict[str, Any],
) -> None:
    """Reject any drift from the issue #252 preregistration."""

    inventory = profile.get("evidence_inventory_contract")
    power = profile.get("power_design")
    gate = profile.get("forward_gate_contract")
    checks = {
        "identity": profile.get("schema_version") == PROFILE_SCHEMA_VERSION
        and profile.get("issue_number") == 252
        and profile.get("candidate_name") == "retained_v6_7_champion"
        and profile.get("frozen") is True
        and profile.get("preregistered_in_issue_before_implementation") is True,
        "champion": profile.get("champion_contract")
        == {
            "profile_sha256": (
                "cec55d243acd6bbf60a5e8474545b487086ddcd4d18073682ae7f2d4660d2248"
            ),
            "model_score_cost_sizing_threshold_and_guard_mutation_allowed": False,
            "side_quota_enabled": False,
            "regime_emergent_side_required": True,
        },
        "inventory": inventory
        == {
            "issue238_manifest_sha256": (
                "7cdaf04f61e02e2c0b829c2d5c6e4a159021cb71923d4326afc8a39e126f2e9f"
            ),
            "issue250_manifest_sha256": (
                "6ecb044f3fc34c4ebf2063412da71fe974f2f55f231fb8fa51418cf247d5ae26"
            ),
            "issue251_manifest_sha256": (
                "6c44b6d56ed9aeb6910bcbf32882e6b6c77a8f3e0ecfe445e4021ade7594da5b"
            ),
            "future_windows_must_be_market_disjoint": True,
            "future_windows_must_be_chronological": True,
            "target_free_freeze_must_precede_outcome_access": True,
            "official_read_only_settlement_required": True,
            "complete_settlement_required": True,
            "decision_time_feature_causality_required": True,
        },
        "power": power
        == {
            "statistical_unit": (
                "quality_valid_market_including_guard_blocked_zero"
            ),
            "primary_estimand": (
                "retained_v6_7_after_cost_pnl_per_quality_valid_market"
            ),
            "completed_future_outcomes_used_for_variance_planning_only": True,
            "completed_future_outcomes_used_for_model_or_threshold_tuning": False,
            "completed_future_outcomes_used_as_promotion_pass": False,
            "one_sided_alpha": 0.05,
            "target_power": 0.8,
            "minimum_relevant_mean_after_cost_pnl_per_market": 0.01,
            "reported_mean_effect_sizes": [0.005, 0.0075, 0.01, 0.0125],
            "variance_floor_standard_deviation": 0.05,
            "robustness_inflation_factor": 1.25,
            "market_bootstrap_seed": 2522026,
            "market_bootstrap_resample_count": 10000,
            "market_bootstrap_one_sided_confidence_level": 0.95,
            "planning_capture_quality_rate": 0.9,
            "minimum_target_free_guard_acceptance_rate": 0.95,
            "bounded_batch_market_count": 12,
            "result_selected_sample_size_allowed": False,
            "result_selected_extension_allowed": False,
        },
        "gate": gate
        == {
            "strictly_later_and_disjoint_required": True,
            "target_free_decision_freeze_required": True,
            "all_selected_markets_closed_before_outcome_access": True,
            "official_read_only_settlement_on_quarantine_copies": True,
            "complete_settlement_required": True,
            "total_after_cost_pnl_minimum_exclusive": 0.0,
            "largest_winner_removed_after_cost_pnl_minimum_exclusive": 0.0,
            "market_bootstrap_one_sided_lower_bound_minimum_exclusive": 0.0,
            "one_evaluation_only": True,
            "result_selected_rerun_allowed": False,
            "paper_candidate_auto_unlock_allowed": False,
            "separate_manual_paper_authorization_issue_required": True,
        },
        "safety": profile.get("safety") == SAFETY,
    }
    blockers = sorted(name for name, passed in checks.items() if not passed)
    if blockers:
        raise ValueError("#252 frozen profile invalid: " + ", ".join(blockers))


def required_market_count(
    *,
    standard_deviation: float,
    mean_effect: float,
    one_sided_alpha: float,
    power: float,
    robustness_inflation_factor: float,
) -> int:
    """Return the inflated independent-market count for a one-sample mean."""

    values = (
        standard_deviation,
        mean_effect,
        one_sided_alpha,
        power,
        robustness_inflation_factor,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("#252 power inputs must be finite")
    if (
        standard_deviation <= 0.0
        or mean_effect <= 0.0
        or not 0.0 < one_sided_alpha < 0.5
        or not 0.5 < power < 1.0
        or robustness_inflation_factor < 1.0
    ):
        raise ValueError("#252 power inputs are outside their valid ranges")
    z_alpha = NormalDist().inv_cdf(1.0 - one_sided_alpha)
    z_power = NormalDist().inv_cdf(power)
    base = ((z_alpha + z_power) * standard_deviation / mean_effect) ** 2
    return math.ceil(base * robustness_inflation_factor)


def build_paper_readiness_reports(
    *,
    issue238_window: dict[str, Any],
    issue250_window: dict[str, Any],
    historical_inventory: dict[str, Any],
    profile: dict[str, Any],
    report_created_ts: int,
) -> dict[str, dict[str, Any]]:
    """Build inventory, descriptive power, and forward gate reports."""

    validate_retained_v6_7_paper_readiness_profile(profile)
    first = _validate_future_window(issue238_window, expected_name="issue238")
    second = _validate_future_window(issue250_window, expected_name="issue250")
    historical = _validate_historical_inventory(historical_inventory)
    first_ids = set(first["market_ids"])
    second_ids = set(second["market_ids"])
    overlap = sorted(first_ids & second_ids)
    chronological = (
        int(second["minimum_market_start_ts"])
        > int(first["maximum_market_start_ts"])
    )
    inventory_blockers = []
    if overlap:
        inventory_blockers.append("future_evidence_market_overlap")
    if not chronological:
        inventory_blockers.append("future_evidence_not_strictly_chronological")
    inventory = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "report_created_ts": report_created_ts,
        "champion": "retained_v6_7_champion",
        "sources": [
            _inventory_source(first, variance_eligible=True),
            _inventory_source(second, variance_eligible=True),
            _inventory_source(historical, variance_eligible=False),
        ],
        "future_source_count": 2,
        "future_quality_valid_market_count": (
            first["quality_valid_market_count"]
            + second["quality_valid_market_count"]
        ),
        "future_guard_accepted_market_count": (
            first["guard_accepted_market_count"]
            + second["guard_accepted_market_count"]
        ),
        "future_guard_blocked_zero_market_count": (
            first["guard_blocked_zero_market_count"]
            + second["guard_blocked_zero_market_count"]
        ),
        "future_market_overlap_count": len(overlap),
        "future_market_overlap_ids": overlap,
        "future_windows_strictly_chronological": chronological,
        "target_free_freezes_preceded_outcome_access": True,
        "official_read_only_settlement_complete": True,
        "decision_time_feature_causality_passed": True,
        "completed_outcomes_used_for_variance_planning_only": True,
        "completed_outcomes_used_for_model_or_threshold_tuning": False,
        "retrospective_promotion_pass_evaluated": False,
        "evidence_inventory_blocking_reason_codes": inventory_blockers,
        "evidence_inventory_passed": not inventory_blockers,
        **SAFETY,
    }
    inventory["report_id"] = canonical_json_sha256(inventory)

    pnl_values = list(first["market_pnl_values"]) + list(
        second["market_pnl_values"]
    )
    if inventory_blockers:
        raise ValueError(
            "#252 evidence inventory failed: " + ", ".join(inventory_blockers)
        )
    design = profile["power_design"]
    sample_std = stdev(pnl_values)
    planning_std = max(
        sample_std,
        float(design["variance_floor_standard_deviation"]),
    )
    sample_size_table = [
        {
            "mean_effect": float(effect),
            "required_quality_valid_market_count": required_market_count(
                standard_deviation=planning_std,
                mean_effect=float(effect),
                one_sided_alpha=float(design["one_sided_alpha"]),
                power=float(design["target_power"]),
                robustness_inflation_factor=float(
                    design["robustness_inflation_factor"]
                ),
            ),
        }
        for effect in design["reported_mean_effect_sizes"]
    ]
    recommended = next(
        row
        for row in sample_size_table
        if row["mean_effect"]
        == float(design["minimum_relevant_mean_after_cost_pnl_per_market"])
    )
    required_valid = int(recommended["required_quality_valid_market_count"])
    minimum_accepted = math.ceil(
        required_valid
        * float(design["minimum_target_free_guard_acceptance_rate"])
    )
    raw_attempt_cap = math.ceil(
        required_valid / float(design["planning_capture_quality_rate"])
    )
    attempt_cap = _round_up(
        raw_attempt_cap,
        int(design["bounded_batch_market_count"]),
    )
    bootstrap = _market_bootstrap_mean(
        pnl_values,
        seed=int(design["market_bootstrap_seed"]),
        samples=int(design["market_bootstrap_resample_count"]),
        one_sided_confidence=float(
            design["market_bootstrap_one_sided_confidence_level"]
        ),
    )
    combined_total = sum(pnl_values)
    largest_winner = max(pnl_values)
    power_report = {
        "schema_version": POWER_SCHEMA_VERSION,
        "report_created_ts": report_created_ts,
        "primary_estimand": design["primary_estimand"],
        "statistical_unit": design["statistical_unit"],
        "planning_source_window_count": 2,
        "planning_market_count": len(pnl_values),
        "planning_guard_accepted_market_count": inventory[
            "future_guard_accepted_market_count"
        ],
        "planning_guard_blocked_zero_market_count": inventory[
            "future_guard_blocked_zero_market_count"
        ],
        "descriptive_total_after_cost_pnl": combined_total,
        "descriptive_mean_after_cost_pnl_per_market": mean(pnl_values),
        "descriptive_sample_standard_deviation": sample_std,
        "planning_standard_deviation": planning_std,
        "planning_variance_floor_applied": planning_std > sample_std,
        "descriptive_largest_winner_after_cost_pnl": largest_winner,
        "descriptive_largest_winner_removed_after_cost_pnl": (
            combined_total - largest_winner
        ),
        "descriptive_market_bootstrap_mean": bootstrap,
        "one_sided_alpha": design["one_sided_alpha"],
        "target_power": design["target_power"],
        "robustness_inflation_factor": design[
            "robustness_inflation_factor"
        ],
        "sample_size_table": sample_size_table,
        "recommended_minimum_relevant_mean_after_cost_pnl_per_market": design[
            "minimum_relevant_mean_after_cost_pnl_per_market"
        ],
        "recommended_quality_valid_market_count": required_valid,
        "recommended_minimum_target_free_guard_accepted_market_count": (
            minimum_accepted
        ),
        "recommended_maximum_capture_attempt_count": attempt_cap,
        "recommended_complete_batch_count_cap": (
            attempt_cap // int(design["bounded_batch_market_count"])
        ),
        "completed_future_outcomes_used_for_variance_planning_only": True,
        "completed_future_outcomes_used_for_model_or_threshold_tuning": False,
        "completed_future_outcomes_used_as_promotion_pass": False,
        "result_selected_sample_size_allowed": False,
        "result_selected_extension_allowed": False,
        "power_analysis_ready": True,
        "power_analysis_blocking_reason_codes": [],
        **SAFETY,
    }
    power_report["report_id"] = canonical_json_sha256(power_report)

    gate = profile["forward_gate_contract"]
    gate_plan = {
        "schema_version": GATE_PLAN_SCHEMA_VERSION,
        "report_created_ts": report_created_ts,
        "champion": "retained_v6_7_champion",
        "frozen": True,
        "role": "future_paper_candidate_readiness_diagnostic",
        "collection_stop_rule": (
            f"earliest_exact_{required_valid}_quality_valid_markets_within_"
            f"{attempt_cap}_attempts_after_complete_12_market_batches"
        ),
        "exact_quality_valid_market_count": required_valid,
        "minimum_target_free_guard_accepted_market_count": minimum_accepted,
        "maximum_capture_attempt_count": attempt_cap,
        "bounded_batch_market_count": design["bounded_batch_market_count"],
        "strictly_later_and_disjoint_required": True,
        "target_free_decision_freeze_required": True,
        "all_selected_markets_closed_before_outcome_access": True,
        "official_read_only_settlement_on_quarantine_copies": True,
        "complete_settlement_required": True,
        "hard_gate_checks": {
            "total_after_cost_pnl_minimum_exclusive": gate[
                "total_after_cost_pnl_minimum_exclusive"
            ],
            "largest_winner_removed_after_cost_pnl_minimum_exclusive": gate[
                "largest_winner_removed_after_cost_pnl_minimum_exclusive"
            ],
            "market_bootstrap_one_sided_lower_bound_minimum_exclusive": gate[
                "market_bootstrap_one_sided_lower_bound_minimum_exclusive"
            ],
            "market_bootstrap_seed": design["market_bootstrap_seed"],
            "market_bootstrap_resample_count": design[
                "market_bootstrap_resample_count"
            ],
            "market_bootstrap_one_sided_confidence_level": design[
                "market_bootstrap_one_sided_confidence_level"
            ],
            "runtime_safety_and_forbidden_field_checks_required": True,
        },
        "side_quota_enabled": False,
        "side_action_and_family_metrics_diagnostic_only": True,
        "one_evaluation_only": True,
        "result_selected_rerun_allowed": False,
        "result_selected_extension_allowed": False,
        "paper_candidate_auto_unlock_allowed": False,
        "separate_manual_paper_authorization_issue_required": True,
        "future_outcome_blind_collection_preregistered": True,
        "future_outcome_blind_collection_may_start_after_code_validation": True,
        "paper_candidate_gate_design_ready": True,
        "paper_candidate_gate_blocking_reason_codes": [],
        **SAFETY,
    }
    gate_plan["plan_id"] = canonical_json_sha256(gate_plan)
    return {
        "inventory": inventory,
        "power_report": power_report,
        "gate_plan": gate_plan,
    }


def run_retained_v6_7_paper_readiness(
    config: RetainedV67PaperReadinessConfig,
) -> dict[str, Any]:
    """Verify authoritative evidence and write the frozen #252 reports."""

    paths = {
        "profile": config.profile_path.resolve(),
        "issue238": config.issue238_manifest_path.resolve(),
        "issue250": config.issue250_manifest_path.resolve(),
        "issue251": config.issue251_manifest_path.resolve(),
    }
    pins = {
        "profile": config.expected_profile_sha256,
        "issue238": config.expected_issue238_manifest_sha256,
        "issue250": config.expected_issue250_manifest_sha256,
        "issue251": config.expected_issue251_manifest_sha256,
    }
    for name, path in paths.items():
        _verify_pin(path, pins[name], f"#252 {name}")
    profile = _load_json(paths["profile"])
    validate_retained_v6_7_paper_readiness_profile(profile)
    inventory_contract = profile["evidence_inventory_contract"]
    for name in ("issue238", "issue250", "issue251"):
        if pins[name] != inventory_contract[f"{name}_manifest_sha256"]:
            raise ValueError(f"#252 {name} pin differs from frozen profile")

    issue238 = _load_issue238_window(paths["issue238"])
    issue250 = _load_issue250_window(paths["issue250"])
    historical = _load_issue251_inventory(paths["issue251"])
    reports = build_paper_readiness_reports(
        issue238_window=issue238,
        issue250_window=issue250,
        historical_inventory=historical,
        profile=profile,
        report_created_ts=config.report_created_ts,
    )
    run_dir = config.output_dir.resolve() / config.run_id
    if run_dir.exists() and not config.overwrite_existing:
        raise FileExistsError(f"#252 run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return _write_outputs(
        run_dir=run_dir,
        config=config,
        paths=paths,
        reports=reports,
    )


def _load_issue238_window(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != ISSUE238_MANIFEST_SCHEMA:
        raise ValueError("#252 issue238 manifest schema invalid")
    _validate_manifest_safety(manifest, live_field_optional=True)
    target_free = _load_descriptor_json(
        manifest["target_free_freeze_manifest"], "#252 issue238 target-free"
    )
    selected = _load_descriptor_jsonl(
        target_free["selected_window_rows"], "#252 issue238 selected rows"
    )
    report = _load_descriptor_json(manifest["report"], "#252 issue238 report")
    targets = _load_descriptor_jsonl(
        manifest["v6_7_baseline_runtime_targets"],
        "#252 issue238 v6.7 targets",
    )
    settlement = _load_descriptor_json(
        manifest["settled_corpus_index"], "#252 issue238 settlement"
    )
    checks = target_free.get("target_free_checks") or _load_descriptor_json(
        target_free["report"], "#252 issue238 target-free report"
    ).get("target_free_checks")
    if (
        manifest.get("future_pnl_gate_passed") is not False
        or report.get("evaluation_market_count") != 120
        or report.get("accepted_unique_market_count") != len(targets)
        or report.get(
            "future_outcomes_used_for_model_threshold_cost_sizing_or_guard_tuning"
        )
        is not False
        or report.get("result_selected_rerun_allowed") is not False
        or not isinstance(checks, dict)
        or not all(checks.values())
    ):
        raise ValueError("#252 issue238 evidence contract invalid")
    _validate_settlement_index(settlement, expected_market_count=120)
    return _future_window(
        name="issue238",
        selected_rows=selected,
        target_rows=targets,
        reported_total=float(report["v6_7_after_cost_pnl"]),
        target_free_manifest_sha256=manifest[
            "target_free_freeze_manifest"
        ]["sha256"],
        settlement_sha256=manifest["settled_corpus_index"]["sha256"],
        source_manifest_sha256=_descriptor(manifest_path)["sha256"],
    )


def _load_issue250_window(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != ISSUE250_MANIFEST_SCHEMA:
        raise ValueError("#252 issue250 manifest schema invalid")
    _validate_manifest_safety(manifest)
    target_free = _load_descriptor_json(
        manifest["target_free_freeze_manifest"], "#252 issue250 target-free"
    )
    selected = _load_descriptor_jsonl(
        target_free["selected_rows"], "#252 issue250 selected rows"
    )
    target_free_report = _load_descriptor_json(
        target_free["report"], "#252 issue250 target-free report"
    )
    report = _load_descriptor_json(manifest["report"], "#252 issue250 report")
    targets = _load_descriptor_jsonl(
        manifest["v6_7_runtime_targets"], "#252 issue250 v6.7 targets"
    )
    settlement = _load_descriptor_json(
        manifest["settled_index"], "#252 issue250 settlement"
    )
    checks = target_free_report.get("target_free_checks")
    if (
        report.get("evaluation_market_count") != 120
        or report.get("v6_7_guard_accepted_unique_market_count") != len(targets)
        or report.get(
            "future_outcomes_used_for_model_threshold_cost_sizing_or_guard_tuning"
        )
        is not False
        or report.get("future_results_used_for_tuning") is not False
        or report.get("result_selected_rerun_allowed") is not False
        or manifest.get("future_results_used_for_tuning") is not False
        or not isinstance(checks, dict)
        or not all(checks.values())
    ):
        raise ValueError("#252 issue250 evidence contract invalid")
    _validate_settlement_index(settlement, expected_market_count=120)
    return _future_window(
        name="issue250",
        selected_rows=selected,
        target_rows=targets,
        reported_total=float(report["v6_7_after_cost_pnl"]),
        target_free_manifest_sha256=manifest[
            "target_free_freeze_manifest"
        ]["sha256"],
        settlement_sha256=manifest["settled_index"]["sha256"],
        source_manifest_sha256=_descriptor(manifest_path)["sha256"],
    )


def _load_issue251_inventory(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != ISSUE251_MANIFEST_SCHEMA:
        raise ValueError("#252 issue251 manifest schema invalid")
    _validate_manifest_safety(manifest)
    report = _load_descriptor_json(
        manifest["historical_noninferiority_report"],
        "#252 issue251 historical report",
    )
    selected = _load_descriptor_jsonl(
        manifest["historical_selected_rows"],
        "#252 issue251 historical rows",
    )
    if (
        report.get("historical_noninferiority_gate_passed") is not True
        or report.get("model_improvement_demonstrated") is not False
        or report.get("final_policy_difference_market_count") != 0
        or manifest.get("new_future_challenger_collection_justified") is not False
    ):
        raise ValueError("#252 issue251 retained-champion conclusion invalid")
    return {
        "name": "issue251_historical_screening",
        "role": "historical_screening_lineage_only",
        "market_ids": sorted({str(row["market_id"]) for row in selected}),
        "quality_valid_market_count": len(selected),
        "guard_accepted_market_count": int(
            report["candidate_guard_accepted_market_count"]
        ),
        "guard_blocked_zero_market_count": (
            len(selected) - int(report["candidate_guard_accepted_market_count"])
        ),
        "minimum_market_start_ts": min(
            int(row.get("decision_ts") or 0) for row in selected
        ),
        "maximum_market_start_ts": max(
            int(row.get("decision_ts") or 0) for row in selected
        ),
        "market_pnl_values": [],
        "source_manifest_sha256": _descriptor(manifest_path)["sha256"],
        "target_free_manifest_sha256": "",
        "settlement_sha256": "",
        "lineage_valid": True,
    }


def _future_window(
    *,
    name: str,
    selected_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    reported_total: float,
    target_free_manifest_sha256: str,
    settlement_sha256: str,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    selected_by_id: dict[str, dict[str, Any]] = {}
    for row in selected_rows:
        market_id = str(row.get("market_id") or "")
        if (
            not market_id
            or market_id in selected_by_id
            or row.get("labels_outcomes_or_pnl_opened") is not False
        ):
            raise ValueError(f"#252 {name} selected market lineage invalid")
        selected_by_id[market_id] = row
    target_by_id: dict[str, dict[str, Any]] = {}
    for row in target_rows:
        market_id = str(row.get("market_id") or "")
        if (
            not market_id
            or market_id in target_by_id
            or market_id not in selected_by_id
            or row.get("target_used_as_decision_time_input") is not False
            or row.get("target_available_only_post_exit_or_official_resolution")
            is not True
        ):
            raise ValueError(f"#252 {name} target lineage invalid")
        pnl = float(row["runtime_policy_after_cost_net_pnl_at_frozen_size"])
        if not math.isfinite(pnl):
            raise ValueError(f"#252 {name} target PnL invalid")
        target_by_id[market_id] = row
    pnl_values = [
        float(
            target_by_id[market_id][
                "runtime_policy_after_cost_net_pnl_at_frozen_size"
            ]
        )
        if market_id in target_by_id
        else 0.0
        for market_id in selected_by_id
    ]
    if not math.isclose(sum(pnl_values), reported_total, abs_tol=1e-12):
        raise ValueError(f"#252 {name} target/report PnL mismatch")
    starts = [int(row["market_start_ts"]) for row in selected_rows]
    return {
        "name": name,
        "role": "independent_future_variance_planning_only",
        "market_ids": sorted(selected_by_id),
        "quality_valid_market_count": len(selected_by_id),
        "guard_accepted_market_count": len(target_by_id),
        "guard_blocked_zero_market_count": len(selected_by_id) - len(target_by_id),
        "minimum_market_start_ts": min(starts),
        "maximum_market_start_ts": max(starts),
        "market_pnl_values": pnl_values,
        "reported_total_after_cost_pnl": reported_total,
        "source_manifest_sha256": source_manifest_sha256,
        "target_free_manifest_sha256": target_free_manifest_sha256,
        "settlement_sha256": settlement_sha256,
        "lineage_valid": True,
    }


def _validate_future_window(
    window: dict[str, Any],
    *,
    expected_name: str,
) -> dict[str, Any]:
    if (
        window.get("name") != expected_name
        or window.get("role") != "independent_future_variance_planning_only"
        or window.get("lineage_valid") is not True
        or len(window.get("market_ids") or [])
        != int(window.get("quality_valid_market_count") or 0)
        or len(window.get("market_pnl_values") or [])
        != int(window.get("quality_valid_market_count") or 0)
    ):
        raise ValueError(f"#252 {expected_name} future window invalid")
    for field in (
        "source_manifest_sha256",
        "target_free_manifest_sha256",
        "settlement_sha256",
    ):
        _require_sha256(str(window.get(field) or ""), f"#252 {field}")
    return window


def _validate_historical_inventory(window: dict[str, Any]) -> dict[str, Any]:
    if (
        window.get("name") != "issue251_historical_screening"
        or window.get("role") != "historical_screening_lineage_only"
        or window.get("lineage_valid") is not True
        or window.get("market_pnl_values") != []
    ):
        raise ValueError("#252 historical inventory invalid")
    _require_sha256(
        str(window.get("source_manifest_sha256") or ""),
        "#252 historical manifest",
    )
    return window


def _inventory_source(
    window: dict[str, Any],
    *,
    variance_eligible: bool,
) -> dict[str, Any]:
    return {
        key: value
        for key, value in window.items()
        if key not in {"market_ids", "market_pnl_values"}
    } | {
        "unique_market_identity_hash": canonical_json_sha256(
            window["market_ids"]
        ),
        "variance_planning_eligible": variance_eligible,
        "promotion_pass_eligible": False,
    }


def _market_bootstrap_mean(
    values: list[float],
    *,
    seed: int,
    samples: int,
    one_sided_confidence: float,
) -> dict[str, Any]:
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(
        mean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(samples)
    )
    alpha = 1.0 - one_sided_confidence
    return {
        "bootstrap_unit": "market_id",
        "bootstrap_seed": seed,
        "bootstrap_resample_count": samples,
        "one_sided_confidence_level": one_sided_confidence,
        "point_estimate": mean(values),
        "one_sided_lower_confidence_bound": draws[
            int(alpha * (samples - 1))
        ],
        "two_sided_95_lower": draws[int(0.025 * (samples - 1))],
        "two_sided_95_upper": draws[int(0.975 * (samples - 1))],
    }


def _round_up(value: int, multiple: int) -> int:
    return math.ceil(value / multiple) * multiple


def _validate_manifest_safety(
    manifest: dict[str, Any],
    *,
    live_field_optional: bool = False,
) -> None:
    expected = {
        "paper_only": True,
        "paper_candidate_allowed": False,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }
    if not live_field_optional:
        expected["live_trading_enabled"] = False
    blockers = [key for key, value in expected.items() if manifest.get(key) != value]
    if blockers:
        raise ValueError("#252 source manifest safety invalid: " + ", ".join(blockers))


def _validate_settlement_index(
    settlement: dict[str, Any],
    *,
    expected_market_count: int,
) -> None:
    entries = settlement.get("entries")
    if (
        settlement.get("entry_count") != expected_market_count
        or not isinstance(entries, list)
        or len(entries) != expected_market_count
        or settlement.get("outcomes_used_for_decision_selection_or_tuning")
        is not False
        or settlement.get("source_outcome_blind_rounds_mutated") is not False
        or any(
            entry.get("official_read_only_resolution") is not True
            or entry.get("settled_after_market_close") is not True
            or entry.get("source_outcome_blind_round_mutated") is not False
            for entry in entries
        )
    ):
        raise ValueError("#252 settlement lineage or completeness invalid")


def _verified_descriptor(value: Any, name: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} descriptor missing")
    path = Path(str(value.get("path") or "")).resolve()
    sha256 = str(value.get("sha256") or "")
    _verify_pin(path, sha256, name)
    return {"path": str(path), "sha256": sha256.lower()}


def _load_descriptor_json(value: Any, name: str) -> dict[str, Any]:
    descriptor = _verified_descriptor(value, name)
    return _load_json(Path(descriptor["path"]))


def _load_descriptor_jsonl(value: Any, name: str) -> list[dict[str, Any]]:
    descriptor = _verified_descriptor(value, name)
    return _load_jsonl(Path(descriptor["path"]))


def _write_outputs(
    *,
    run_dir: Path,
    config: RetainedV67PaperReadinessConfig,
    paths: dict[str, Path],
    reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    inventory_path = run_dir / "v6_7_champion_evidence_inventory_report.json"
    inventory_md_path = run_dir / "v6_7_champion_evidence_inventory_report.md"
    power_path = run_dir / "v6_7_paper_readiness_power_report.json"
    power_md_path = run_dir / "v6_7_paper_readiness_power_report.md"
    gate_path = run_dir / "v6_7_forward_paper_candidate_gate_plan.json"
    gate_md_path = run_dir / "v6_7_forward_paper_candidate_gate_plan.md"
    _write_json(inventory_path, reports["inventory"])
    _write_json(power_path, reports["power_report"])
    _write_json(gate_path, reports["gate_plan"])
    _write_text(inventory_md_path, _inventory_markdown(reports["inventory"]))
    _write_text(power_md_path, _power_markdown(reports["power_report"]))
    _write_text(gate_md_path, _gate_markdown(reports["gate_plan"]))
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "implementation_commit": config.implementation_commit,
        "profile": _descriptor(paths["profile"]),
        "issue238_manifest": _descriptor(paths["issue238"]),
        "issue250_manifest": _descriptor(paths["issue250"]),
        "issue251_manifest": _descriptor(paths["issue251"]),
        "evidence_inventory_report": _descriptor(inventory_path),
        "evidence_inventory_markdown": _descriptor(inventory_md_path),
        "power_report": _descriptor(power_path),
        "power_report_markdown": _descriptor(power_md_path),
        "forward_gate_plan": _descriptor(gate_path),
        "forward_gate_plan_markdown": _descriptor(gate_md_path),
        "evidence_inventory_passed": True,
        "power_analysis_ready": True,
        "paper_candidate_gate_design_ready": True,
        "recommended_quality_valid_market_count": reports["power_report"][
            "recommended_quality_valid_market_count"
        ],
        "recommended_maximum_capture_attempt_count": reports["power_report"][
            "recommended_maximum_capture_attempt_count"
        ],
        "future_outcome_blind_collection_preregistered": True,
        "completed_future_outcomes_used_for_model_or_threshold_tuning": False,
        "retrospective_promotion_pass_evaluated": False,
        **SAFETY,
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v6_7_paper_readiness_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "inventory_path": inventory_path,
        "inventory_sha256": _descriptor(inventory_path)["sha256"],
        "power_report_path": power_path,
        "power_report_sha256": _descriptor(power_path)["sha256"],
        "gate_plan_path": gate_path,
        "gate_plan_sha256": _descriptor(gate_path)["sha256"],
        "manifest_path": manifest_path,
        "manifest_sha256": _descriptor(manifest_path)["sha256"],
        "reports": reports,
        "manifest": manifest,
    }


def _inventory_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6.7 Champion Evidence Inventory",
            "",
            f"- inventory passed: `{str(report['evidence_inventory_passed']).lower()}`",
            f"- independent future markets: `{report['future_quality_valid_market_count']}`",
            f"- guard accepted / zero: `{report['future_guard_accepted_market_count']} / {report['future_guard_blocked_zero_market_count']}`",
            f"- future overlap: `{report['future_market_overlap_count']}`",
            f"- chronological: `{str(report['future_windows_strictly_chronological']).lower()}`",
            "- outcomes used for variance planning only: `true`",
            "- retrospective promotion pass evaluated: `false`",
            "",
        ]
    )


def _power_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6.7 Paper Readiness Power",
            "",
            f"- planning markets: `{report['planning_market_count']}`",
            f"- descriptive PnL: `{report['descriptive_total_after_cost_pnl']}`",
            f"- mean / sample SD: `{report['descriptive_mean_after_cost_pnl_per_market']} / {report['descriptive_sample_standard_deviation']}`",
            f"- fixed relevant mean edge: `{report['recommended_minimum_relevant_mean_after_cost_pnl_per_market']}`",
            f"- powered quality-valid markets: `{report['recommended_quality_valid_market_count']}`",
            f"- minimum guard-accepted markets: `{report['recommended_minimum_target_free_guard_accepted_market_count']}`",
            f"- capture attempt cap: `{report['recommended_maximum_capture_attempt_count']}`",
            "- completed outcomes used for model/threshold tuning: `false`",
            "- promotion evidence eligible: `false`",
            "",
        ]
    )


def _gate_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6.7 Forward Paper-Candidate Gate Plan",
            "",
            f"- stop rule: `{report['collection_stop_rule']}`",
            f"- quality-valid / accepted support: `{report['exact_quality_valid_market_count']} / {report['minimum_target_free_guard_accepted_market_count']}`",
            "- total PnL, largest-winner-removed PnL, and one-sided bootstrap LCB must all be positive",
            "- side quota: `false`",
            "- one evaluation; result-selected extension/rerun: `false / false`",
            "- paper candidate auto-unlock: `false`",
            "- separate manual authorization issue required: `true`",
            "",
        ]
    )
