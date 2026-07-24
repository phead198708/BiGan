"""Non-risk abstention fallback for issue #248."""

from __future__ import annotations

import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training import (
    execution_layer_v2_support_preserving_overlay_v8_2 as v82,
)
from bigan.v8.polymarket.training.execution_layer_v2_abstention_aware_expected_net_pnl_v7_0 import (
    _v7_0_blocked_safety_fields,
)
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
    _descriptor,
    _load_json,
    _load_jsonl,
    _sha256_file,
    _verify_pin,
    _write_json,
    _write_jsonl,
    _write_text,
)

CANDIDATE_NAME = "non_risk_abstention_fallback_v8_3"
PROFILE_SCHEMA_VERSION = (
    "bigan-v8-non-risk-abstention-fallback-v8-3-profile-v1"
)
DECISION_SCHEMA_VERSION = (
    "bigan-v8-non-risk-abstention-fallback-v8-3-decision-v1"
)
HISTORICAL_REPORT_SCHEMA_VERSION = (
    "bigan-v8-non-risk-abstention-fallback-v8-3-historical-report-v1"
)
CANARY_REPORT_SCHEMA_VERSION = (
    "bigan-v8-non-risk-abstention-fallback-v8-3-canary-report-v1"
)
MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-non-risk-abstention-fallback-v8-3-manifest-v1"
)
POLICY_LEVEL_ABSTENTION_REASON_CODES = {
    "policy_selected_no_trade",
    "v6_7_no_positive_guard_compatible_action",
    "v8_1_veto_to_no_trade",
}


@dataclass(frozen=True, slots=True)
class NonRiskAbstentionFallbackV83HistoricalConfig:
    run_id: str
    output_dir: str
    profile_path: str
    expected_profile_sha256: str
    historical_manifest_path: str
    expected_historical_manifest_sha256: str
    implementation_commit: str
    evaluation_started_ts: int
    overwrite_existing: bool = False


@dataclass(frozen=True, slots=True)
class NonRiskAbstentionFallbackV83CanaryConfig:
    run_id: str
    output_dir: str
    profile_path: str
    expected_profile_sha256: str
    historical_gate_manifest_path: str
    expected_historical_gate_manifest_sha256: str
    issue246_target_free_manifest_path: str
    expected_issue246_target_free_manifest_sha256: str
    implementation_commit: str
    canary_started_ts: int
    overwrite_existing: bool = False


def validate_non_risk_abstention_fallback_v8_3_profile(
    profile: dict[str, Any],
) -> None:
    checks = {
        "identity": profile.get("schema_version") == PROFILE_SCHEMA_VERSION
        and profile.get("issue_number") == 248
        and profile.get("candidate_name") == CANDIDATE_NAME
        and profile.get(
            "preregistered_before_implementation_and_historical_target_access"
        )
        is True,
        "policy": profile.get("policy_contract")
        == {
            "candidate_primary": "adaptive_support_controller_v8_1",
            "explicit_execution_risk_blocker_bypass_allowed": False,
            "fallback_baseline": "p_up_semantic_compatibility_v6_7",
            "fallback_requires_independent_full_guard_pass": True,
            "fallback_trigger": "v8_1_policy_level_non_risk_abstention",
            "full_execution_guard_unchanged": True,
            "model_threshold_quantile_cost_sizing_or_guard_changed": False,
            "policy_level_abstention_reason_codes": sorted(
                POLICY_LEVEL_ABSTENTION_REASON_CODES
            ),
            "source_or_o_score_mutation_allowed": False,
        },
        "historical_gate": profile.get("historical_gate")
        == {
            "candidate_minus_v6_7_largest_winner_removed_after_cost_pnl_minimum_inclusive": 0.0,
            "candidate_minus_v6_7_total_after_cost_pnl_minimum_inclusive": 0.0,
            "candidate_support_not_below_v6_7": True,
            "equality_passes_noninferiority": True,
            "side_quota_enabled": False,
        },
        "canary_gate": profile.get("target_free_canary_gate")
        == {
            "exact_market_count": 120,
            "minimum_guard_accepted_market_count": 40,
            "outcomes_labels_resolution_or_pnl_opened": False,
            "side_quota_enabled": False,
        },
        "lineage": profile.get("lineage", {}).get(
            "issue246_outcomes_allowed_for_v8_3"
        )
        is False,
        "safety": profile.get("safety") == _expected_safety(),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("#248 v8.3 profile invalid: " + ", ".join(blockers))


def select_non_risk_abstention_fallback_v8_3_decision(
    *,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    v82._validate_decision_input(candidate, label="v8.1")
    v82._validate_decision_input(baseline, label="v6.7")
    if candidate["market_id"] != baseline["market_id"]:
        raise ValueError("#248 candidate/baseline market mismatch")
    candidate_action = str(candidate["selected_action"])
    baseline_action = str(baseline["selected_action"])
    candidate_allowed = candidate["execution_guard_order_allowed"] is True
    baseline_allowed = baseline["execution_guard_order_allowed"] is True
    candidate_blockers = set(candidate["execution_blocking_reason_codes"])
    policy_abstention_only = (
        candidate_action == "NO_TRADE"
        and candidate_allowed is False
        and bool(candidate_blockers)
        and candidate_blockers <= POLICY_LEVEL_ABSTENTION_REASON_CODES
    )

    if candidate_allowed and candidate_action in v82.TRADE_ACTIONS:
        action = candidate_action
        side = str(candidate["selected_side"])
        source = "v8_1_primary"
        allowed = True
        reasons = ["v8_1_primary_full_guard_passed"]
    elif (
        policy_abstention_only
        and baseline_allowed
        and baseline_action in v82.TRADE_ACTIONS
    ):
        action = baseline_action
        side = str(baseline["selected_side"])
        source = "v6_7_non_risk_abstention_fallback"
        allowed = True
        reasons = [
            "v8_1_policy_level_non_risk_abstention",
            "v6_7_independent_full_guard_passed",
        ]
    else:
        action = "NO_TRADE"
        side = "NONE"
        source = "fail_closed_no_trade"
        allowed = False
        reasons = _no_trade_reasons(
            candidate=candidate,
            baseline=baseline,
            policy_abstention_only=policy_abstention_only,
        )
    decision = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "market_id": candidate["market_id"],
        "decision_ts": candidate.get("decision_ts"),
        "selected_action": action,
        "selected_side": side,
        "execution_guard_order_allowed": allowed,
        "selection_source": source,
        "selection_reason_codes": reasons,
        "original_v8_1_action": candidate_action,
        "original_v8_1_side": candidate["selected_side"],
        "original_v8_1_guard_allowed": candidate_allowed,
        "original_v8_1_blocking_reason_codes": sorted(candidate_blockers),
        "original_v8_1_rank_abstention_passed": candidate.get(
            "rank_abstention_passed"
        ),
        "original_v8_1_point_selected_action": candidate.get(
            "point_selected_action"
        ),
        "original_v6_7_action": baseline_action,
        "original_v6_7_side": baseline["selected_side"],
        "original_v6_7_guard_allowed": baseline_allowed,
        "original_v6_7_blocking_reason_codes": sorted(
            baseline["execution_blocking_reason_codes"]
        ),
        "fallback_applied": source == "v6_7_non_risk_abstention_fallback",
        "fallback_requires_independent_full_guard_pass": True,
        "explicit_execution_risk_blocker_bypass_used": False,
        "full_execution_guard_unchanged": True,
        "target_or_outcome_used_for_selection": False,
        "source_score_mutated": False,
        **_v7_0_blocked_safety_fields(),
    }
    v82._assert_target_free_decision(decision)
    decision["overlay_decision_id"] = canonical_json_sha256(decision)
    return decision


def build_non_risk_abstention_fallback_v8_3_historical(
    rows: list[dict[str, Any]],
    *,
    profile: dict[str, Any],
) -> dict[str, Any]:
    validate_non_risk_abstention_fallback_v8_3_profile(profile)
    rows_by_market: dict[str, dict[str, Any]] = {}
    decisions: list[dict[str, Any]] = []
    for row in rows:
        market_id = str(row["market_id"])
        if market_id in rows_by_market:
            raise ValueError(f"#248 duplicate historical market: {market_id}")
        rows_by_market[market_id] = row
        decisions.append(
            select_non_risk_abstention_fallback_v8_3_decision(
                candidate=v82._historical_candidate_projection(row),
                baseline=v82._historical_baseline_projection(row),
            )
        )
    evaluation = _evaluate_historical_decisions(
        decisions=decisions,
        rows_by_market=rows_by_market,
    )
    return {"decisions": decisions, **evaluation}


def build_non_risk_abstention_fallback_v8_3_canary(
    *,
    candidate_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    profile: dict[str, Any],
) -> dict[str, Any]:
    validate_non_risk_abstention_fallback_v8_3_profile(profile)
    baseline_by_market = {str(row["market_id"]): row for row in baseline_rows}
    if len(baseline_by_market) != len(baseline_rows):
        raise ValueError("#248 duplicate v6.7 canary market")
    decisions: list[dict[str, Any]] = []
    for row in candidate_rows:
        market_id = str(row["market_id"])
        baseline = baseline_by_market.get(market_id)
        if baseline is None:
            raise ValueError(f"#248 missing v6.7 canary row: {market_id}")
        decisions.append(
            select_non_risk_abstention_fallback_v8_3_decision(
                candidate=_future_candidate_projection(row),
                baseline=_future_baseline_projection(baseline),
            )
        )
    support = sum(
        row["execution_guard_order_allowed"] is True for row in decisions
    )
    checks = {
        "exact_120_markets": len(decisions) == 120,
        "minimum_guard_accepted_support": support >= 40,
        "targets_outcomes_resolution_or_pnl_sealed": True,
        "source_scores_unchanged": all(
            row["source_score_mutated"] is False for row in decisions
        ),
        "risk_blocker_bypass_absent": all(
            row["explicit_execution_risk_blocker_bypass_used"] is False
            for row in decisions
        ),
    }
    reason_map = {
        "exact_120_markets": "target_free_exact_market_count_not_met",
        "minimum_guard_accepted_support": (
            "target_free_guard_accepted_support_insufficient"
        ),
        "targets_outcomes_resolution_or_pnl_sealed": (
            "target_free_outcome_access_detected"
        ),
        "source_scores_unchanged": "source_score_mutation_detected",
        "risk_blocker_bypass_absent": "execution_risk_blocker_bypass_detected",
    }
    blockers = [
        reason_map[name] for name, passed in checks.items() if not passed
    ]
    report = {
        "schema_version": CANARY_REPORT_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "market_count": len(decisions),
        "guard_accepted_market_count": support,
        "selection_source_distribution": _distribution(
            decisions, "selection_source"
        ),
        "selected_action_distribution": _distribution(
            decisions, "selected_action"
        ),
        "selected_side_distribution_diagnostic": _distribution(
            [
                row
                for row in decisions
                if row["selected_side"] != "NONE"
            ],
            "selected_side",
        ),
        "side_quota_enabled": False,
        "issue246_outcomes_opened": False,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "threshold_model_quantile_cost_sizing_or_guard_changed": False,
        "checks": checks,
        "target_free_canary_passed": not blockers,
        "target_free_canary_blocking_reason_codes": blockers,
        "new_future_holdout_collection_allowed": not blockers,
        **_v7_0_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return {"decisions": decisions, "report": report}


def run_non_risk_abstention_fallback_v8_3_historical_gate(
    config: NonRiskAbstentionFallbackV83HistoricalConfig,
) -> dict[str, Any]:
    profile_path = Path(config.profile_path).resolve()
    input_manifest_path = Path(config.historical_manifest_path).resolve()
    _verify_pin(profile_path, config.expected_profile_sha256, "#248 profile")
    _verify_pin(
        input_manifest_path,
        config.expected_historical_manifest_sha256,
        "#248 v8.1 historical manifest",
    )
    profile = _load_json(profile_path)
    validate_non_risk_abstention_fallback_v8_3_profile(profile)
    if (
        profile["lineage"]["v8_1_historical_manifest_sha256"]
        != config.expected_historical_manifest_sha256
    ):
        raise ValueError("#248 historical manifest lineage mismatch")
    input_manifest = _load_json(input_manifest_path)
    rows_path = _verified_descriptor_path(
        input_manifest["prequential_rows"],
        label="#248 historical prequential rows",
    )
    result = build_non_risk_abstention_fallback_v8_3_historical(
        _load_jsonl(rows_path),
        profile=profile,
    )
    report = result["report"]
    report.update(
        {
            "run_id": config.run_id,
            "implementation_commit": config.implementation_commit,
            "evaluation_started_ts": config.evaluation_started_ts,
        }
    )
    report["report_id"] = canonical_json_sha256(report)
    return _write_historical_outputs(
        config=config,
        profile_path=profile_path,
        input_manifest_path=input_manifest_path,
        result={**result, "report": report},
    )


def run_non_risk_abstention_fallback_v8_3_canary(
    config: NonRiskAbstentionFallbackV83CanaryConfig,
) -> dict[str, Any]:
    profile_path = Path(config.profile_path).resolve()
    historical_manifest_path = Path(
        config.historical_gate_manifest_path
    ).resolve()
    issue246_manifest_path = Path(
        config.issue246_target_free_manifest_path
    ).resolve()
    _verify_pin(profile_path, config.expected_profile_sha256, "#248 profile")
    _verify_pin(
        historical_manifest_path,
        config.expected_historical_gate_manifest_sha256,
        "#248 historical gate manifest",
    )
    _verify_pin(
        issue246_manifest_path,
        config.expected_issue246_target_free_manifest_sha256,
        "#248 issue246 target-free manifest",
    )
    profile = _load_json(profile_path)
    validate_non_risk_abstention_fallback_v8_3_profile(profile)
    historical_manifest = _load_json(historical_manifest_path)
    if historical_manifest.get("historical_noninferiority_gate_passed") is not True:
        raise ValueError("#248 historical gate did not pass")
    issue246 = _load_json(issue246_manifest_path)
    if (
        issue246.get("labels_outcomes_resolution_or_pnl_opened") is not False
        or issue246.get("settlement_provider_called") is not False
        or issue246.get("source_scores_mutated") is not False
    ):
        raise ValueError("#248 issue246 input is not target-free")
    candidate_path = _verified_descriptor_path(
        issue246["candidate_guard"], label="#248 issue246 candidate guard"
    )
    baseline_path = _verified_descriptor_path(
        issue246["baseline_guard"], label="#248 issue246 v6.7 guard"
    )
    result = build_non_risk_abstention_fallback_v8_3_canary(
        candidate_rows=_load_jsonl(candidate_path),
        baseline_rows=_load_jsonl(baseline_path),
        profile=profile,
    )
    report = result["report"]
    report.update(
        {
            "run_id": config.run_id,
            "implementation_commit": config.implementation_commit,
            "canary_started_ts": config.canary_started_ts,
            "historical_gate_manifest_sha256": (
                config.expected_historical_gate_manifest_sha256
            ),
            "issue246_target_free_manifest_sha256": (
                config.expected_issue246_target_free_manifest_sha256
            ),
        }
    )
    report["report_id"] = canonical_json_sha256(report)
    return _write_canary_outputs(
        config=config,
        profile_path=profile_path,
        historical_manifest_path=historical_manifest_path,
        issue246_manifest_path=issue246_manifest_path,
        result={**result, "report": report},
    )


def _evaluate_historical_decisions(
    *,
    decisions: list[dict[str, Any]],
    rows_by_market: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidate_selected: list[dict[str, Any]] = []
    baseline_selected: list[dict[str, Any]] = []
    candidate_by_market: dict[str, float] = {}
    baseline_by_market: dict[str, float] = {}
    for decision in decisions:
        row = rows_by_market[str(decision["market_id"])]
        source = str(decision["selection_source"])
        if source == "v8_1_primary":
            candidate_target = float(
                row["selected_target_after_cost_net_pnl_per_contract"]
            )
        elif source == "v6_7_non_risk_abstention_fallback":
            candidate_target = float(
                row["baseline_target_after_cost_net_pnl_per_contract"]
            )
        else:
            candidate_target = 0.0
        baseline_target = (
            float(row["baseline_target_after_cost_net_pnl_per_contract"])
            if row["baseline_execution_guard_order_allowed"] is True
            else 0.0
        )
        candidate_pnl = candidate_target * 0.2
        baseline_pnl = baseline_target * 0.2
        market_id = str(decision["market_id"])
        candidate_by_market[market_id] = candidate_pnl
        baseline_by_market[market_id] = baseline_pnl
        if decision["execution_guard_order_allowed"] is True:
            candidate_selected.append(
                v82._evaluation_row(
                    decision=decision,
                    target=candidate_target,
                    pnl=candidate_pnl,
                    source=source,
                )
            )
        if row["baseline_execution_guard_order_allowed"] is True:
            baseline_selected.append(
                {
                    "market_id": market_id,
                    "action": row["baseline_action"],
                    "side": row["baseline_side"],
                    "target_after_cost_net_pnl_per_contract": baseline_target,
                    "fixed_position_size": 0.2,
                    "after_cost_net_pnl_at_frozen_size": baseline_pnl,
                    "target_used_as_decision_time_input": False,
                    "target_opened_only_after_overlay_decision_freeze": True,
                }
            )
    candidate_total = sum(candidate_by_market.values())
    baseline_total = sum(baseline_by_market.values())
    candidate_largest = max(candidate_by_market.values(), default=0.0)
    baseline_largest = max(baseline_by_market.values(), default=0.0)
    candidate_lwr = candidate_total - max(candidate_largest, 0.0)
    baseline_lwr = baseline_total - max(baseline_largest, 0.0)
    total_delta = candidate_total - baseline_total
    lwr_delta = candidate_lwr - baseline_lwr
    support_delta = len(candidate_selected) - len(baseline_selected)
    checks = {
        "candidate_total_pnl_noninferior_to_v6_7": total_delta >= 0.0,
        "candidate_largest_winner_removed_pnl_noninferior_to_v6_7": (
            lwr_delta >= 0.0
        ),
        "candidate_guard_accepted_support_not_below_v6_7": support_delta >= 0,
        "decisions_frozen_before_historical_target_access": all(
            row["target_or_outcome_used_for_selection"] is False
            for row in decisions
        ),
        "risk_blocker_bypass_absent": all(
            row["explicit_execution_risk_blocker_bypass_used"] is False
            for row in decisions
        ),
        "source_scores_unchanged": all(
            row["source_score_mutated"] is False for row in decisions
        ),
    }
    reason_map = {
        "candidate_total_pnl_noninferior_to_v6_7": (
            "historical_total_after_cost_pnl_inferior_to_v6_7"
        ),
        "candidate_largest_winner_removed_pnl_noninferior_to_v6_7": (
            "historical_largest_winner_removed_pnl_inferior_to_v6_7"
        ),
        "candidate_guard_accepted_support_not_below_v6_7": (
            "historical_guard_accepted_support_below_v6_7"
        ),
        "decisions_frozen_before_historical_target_access": (
            "historical_target_accessed_before_decision_freeze"
        ),
        "risk_blocker_bypass_absent": "execution_risk_blocker_bypass_detected",
        "source_scores_unchanged": "source_score_mutation_detected",
    }
    blockers = [
        reason_map[name] for name, passed in checks.items() if not passed
    ]
    report = {
        "schema_version": HISTORICAL_REPORT_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "historical_market_count": len(decisions),
        "candidate_guard_accepted_market_count": len(candidate_selected),
        "v6_7_guard_accepted_market_count": len(baseline_selected),
        "candidate_minus_v6_7_guard_accepted_market_count": support_delta,
        "candidate_total_after_cost_net_pnl_at_frozen_size": candidate_total,
        "v6_7_total_after_cost_net_pnl_at_frozen_size": baseline_total,
        "candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size": (
            total_delta
        ),
        "candidate_largest_winner_after_cost_net_pnl_at_frozen_size": (
            candidate_largest
        ),
        "v6_7_largest_winner_after_cost_net_pnl_at_frozen_size": baseline_largest,
        "candidate_largest_winner_removed_after_cost_net_pnl_at_frozen_size": (
            candidate_lwr
        ),
        "v6_7_largest_winner_removed_after_cost_net_pnl_at_frozen_size": (
            baseline_lwr
        ),
        "candidate_minus_v6_7_largest_winner_removed_after_cost_net_pnl_at_frozen_size": (
            lwr_delta
        ),
        "selection_source_distribution": _distribution(
            decisions, "selection_source"
        ),
        "selected_action_distribution": _distribution(
            decisions, "selected_action"
        ),
        "selected_side_distribution_diagnostic": _distribution(
            [
                row
                for row in decisions
                if row["selected_side"] != "NONE"
            ],
            "selected_side",
        ),
        "side_quota_enabled": False,
        "historical_targets_opened_only_after_overlay_decision_freeze": True,
        "issue246_outcomes_opened": False,
        "checks": checks,
        "historical_noninferiority_gate_passed": not blockers,
        "historical_gate_blocking_reason_codes": blockers,
        "target_free_canary_allowed": not blockers,
        "future_holdout_collection_allowed": False,
        **_v7_0_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return {
        "candidate_selected_rows": candidate_selected,
        "baseline_selected_rows": baseline_selected,
        "report": report,
    }


def _future_candidate_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "market_id": row["market_id"],
        "decision_ts": row.get("decision_ts"),
        "selected_action": row["selected_action"],
        "selected_side": row["selected_side"],
        "execution_guard_order_allowed": row["execution_guard_order_allowed"],
        "execution_blocking_reason_codes": row[
            "execution_blocking_reason_codes"
        ],
        "rank_abstention_passed": row.get("rank_abstention_passed"),
        "point_selected_action": row.get("point_selected_action"),
    }


def _future_baseline_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "market_id": row["market_id"],
        "decision_ts": row.get("decision_ts"),
        "selected_action": row["selected_action"],
        "selected_side": row["selected_side"],
        "execution_guard_order_allowed": row["execution_guard_order_allowed"],
        "execution_blocking_reason_codes": row[
            "execution_blocking_reason_codes"
        ],
        "rank_abstention_passed": None,
        "point_selected_action": row["selected_action"],
    }


def _no_trade_reasons(
    *,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    policy_abstention_only: bool,
) -> list[str]:
    reasons = list(candidate["execution_blocking_reason_codes"])
    if not policy_abstention_only:
        reasons.append("v8_1_not_policy_level_non_risk_abstention")
    if baseline["execution_guard_order_allowed"] is False:
        reasons.extend(
            f"v6_7_{code}"
            for code in baseline["execution_blocking_reason_codes"]
        )
        reasons.append("v6_7_independent_full_guard_failed")
    if baseline["selected_action"] not in v82.TRADE_ACTIONS:
        reasons.append("v6_7_trade_action_unavailable")
    return sorted(set(reasons or {"non_risk_abstention_overlay_no_trade"}))


def _verified_descriptor_path(
    descriptor: dict[str, Any],
    *,
    label: str,
) -> Path:
    path = Path(descriptor["path"])
    _verify_pin(path, descriptor["sha256"], label)
    return path


def _distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[key]) for row in rows).items()))


def _expected_safety() -> dict[str, bool]:
    safety = _v7_0_blocked_safety_fields()
    safety["paper_only"] = True
    return safety


def _write_historical_outputs(
    *,
    config: NonRiskAbstentionFallbackV83HistoricalConfig,
    profile_path: Path,
    input_manifest_path: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    run_dir = _fresh_run_dir(
        output_dir=config.output_dir,
        run_id=config.run_id,
        overwrite_existing=config.overwrite_existing,
    )
    outputs = {
        "report": run_dir / "v8_3_historical_noninferiority_report.json",
        "report_markdown": (
            run_dir / "v8_3_historical_noninferiority_report.md"
        ),
        "decision_rows": run_dir / "v8_3_historical_frozen_decisions.jsonl",
        "candidate_selected_rows": (
            run_dir / "v8_3_historical_candidate_selected_rows.jsonl"
        ),
        "v6_7_baseline_selected_rows": (
            run_dir / "v8_3_historical_v6_7_selected_rows.jsonl"
        ),
    }
    _write_json(outputs["report"], result["report"])
    _write_text(outputs["report_markdown"], _markdown(result["report"]))
    _write_jsonl(outputs["decision_rows"], result["decisions"])
    _write_jsonl(
        outputs["candidate_selected_rows"],
        result["candidate_selected_rows"],
    )
    _write_jsonl(
        outputs["v6_7_baseline_selected_rows"],
        result["baseline_selected_rows"],
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "stage": "historical_noninferiority",
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        "profile": _descriptor(profile_path),
        "v8_1_historical_manifest": _descriptor(input_manifest_path),
        **{name: _descriptor(path) for name, path in outputs.items()},
        "historical_noninferiority_gate_passed": result["report"][
            "historical_noninferiority_gate_passed"
        ],
        "historical_gate_blocking_reason_codes": result["report"][
            "historical_gate_blocking_reason_codes"
        ],
        "target_free_canary_allowed": result["report"][
            "target_free_canary_allowed"
        ],
        "future_holdout_collection_allowed": False,
        "issue246_outcomes_opened": False,
        **_v7_0_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v8_3_historical_noninferiority_manifest.json"
    _write_json(manifest_path, manifest)
    return _run_result(
        run_dir=run_dir,
        report=result["report"],
        manifest=manifest,
        manifest_path=manifest_path,
        outputs=outputs,
    )


def _write_canary_outputs(
    *,
    config: NonRiskAbstentionFallbackV83CanaryConfig,
    profile_path: Path,
    historical_manifest_path: Path,
    issue246_manifest_path: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    run_dir = _fresh_run_dir(
        output_dir=config.output_dir,
        run_id=config.run_id,
        overwrite_existing=config.overwrite_existing,
    )
    outputs = {
        "report": run_dir / "v8_3_target_free_canary_report.json",
        "report_markdown": run_dir / "v8_3_target_free_canary_report.md",
        "decision_rows": run_dir / "v8_3_target_free_canary_decisions.jsonl",
    }
    _write_json(outputs["report"], result["report"])
    _write_text(outputs["report_markdown"], _markdown(result["report"]))
    _write_jsonl(outputs["decision_rows"], result["decisions"])
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "stage": "target_free_canary",
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        "profile": _descriptor(profile_path),
        "historical_gate_manifest": _descriptor(historical_manifest_path),
        "issue246_target_free_manifest": _descriptor(issue246_manifest_path),
        **{name: _descriptor(path) for name, path in outputs.items()},
        "target_free_canary_passed": result["report"][
            "target_free_canary_passed"
        ],
        "target_free_canary_blocking_reason_codes": result["report"][
            "target_free_canary_blocking_reason_codes"
        ],
        "new_future_holdout_collection_allowed": result["report"][
            "new_future_holdout_collection_allowed"
        ],
        "issue246_outcomes_opened": False,
        **_v7_0_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v8_3_target_free_canary_manifest.json"
    _write_json(manifest_path, manifest)
    return _run_result(
        run_dir=run_dir,
        report=result["report"],
        manifest=manifest,
        manifest_path=manifest_path,
        outputs=outputs,
    )


def _fresh_run_dir(
    *,
    output_dir: str,
    run_id: str,
    overwrite_existing: bool,
) -> Path:
    run_dir = Path(output_dir).resolve() / run_id
    if run_dir.exists():
        if not overwrite_existing:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    return run_dir


def _run_result(
    *,
    run_dir: Path,
    report: dict[str, Any],
    manifest: dict[str, Any],
    manifest_path: Path,
    outputs: dict[str, Path],
) -> dict[str, Any]:
    return {
        "run_dir": run_dir,
        "report": report,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "outputs": outputs,
    }


def _markdown(report: dict[str, Any]) -> str:
    if "historical_noninferiority_gate_passed" in report:
        lines = [
            "# Execution Layer v2 v8.3 Historical Non-Inferiority",
            "",
            f"- candidate support: `{report['candidate_guard_accepted_market_count']}`",
            f"- v6.7 support: `{report['v6_7_guard_accepted_market_count']}`",
            f"- candidate PnL: `{report['candidate_total_after_cost_net_pnl_at_frozen_size']}`",
            f"- v6.7 PnL: `{report['v6_7_total_after_cost_net_pnl_at_frozen_size']}`",
            f"- total delta: `{report['candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size']}`",
            f"- LWR delta: `{report['candidate_minus_v6_7_largest_winner_removed_after_cost_net_pnl_at_frozen_size']}`",
            f"- gate passed: `{str(report['historical_noninferiority_gate_passed']).lower()}`",
        ]
    else:
        lines = [
            "# Execution Layer v2 v8.3 Target-Free Canary",
            "",
            f"- markets: `{report['market_count']}`",
            f"- guard accepted: `{report['guard_accepted_market_count']}`",
            f"- selection sources: `{report['selection_source_distribution']}`",
            f"- canary passed: `{str(report['target_free_canary_passed']).lower()}`",
        ]
    return "\n".join(
        [
            *lines,
            f"- blockers: `{report.get('historical_gate_blocking_reason_codes', report.get('target_free_canary_blocking_reason_codes'))}`",
            "- issue #246 outcomes opened: `false`",
            "- paper/live/write/wallet/capital remain blocked.",
            "",
        ]
    )


__all__ = [
    "NonRiskAbstentionFallbackV83CanaryConfig",
    "NonRiskAbstentionFallbackV83HistoricalConfig",
    "build_non_risk_abstention_fallback_v8_3_canary",
    "build_non_risk_abstention_fallback_v8_3_historical",
    "run_non_risk_abstention_fallback_v8_3_canary",
    "run_non_risk_abstention_fallback_v8_3_historical_gate",
    "select_non_risk_abstention_fallback_v8_3_decision",
    "validate_non_risk_abstention_fallback_v8_3_profile",
]
