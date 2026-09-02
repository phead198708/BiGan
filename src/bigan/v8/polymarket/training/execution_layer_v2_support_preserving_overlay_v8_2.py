"""Support-preserving v8.1 overlay for issue #247."""

from __future__ import annotations

import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
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

CANDIDATE_NAME = "support_preserving_overlay_v8_2"
PROFILE_SCHEMA_VERSION = "bigan-v8-support-preserving-overlay-v8-2-profile-v1"
REPORT_SCHEMA_VERSION = "bigan-v8-support-preserving-overlay-v8-2-report-v1"
MANIFEST_SCHEMA_VERSION = "bigan-v8-support-preserving-overlay-v8-2-manifest-v1"
DECISION_SCHEMA_VERSION = "bigan-v8-support-preserving-overlay-v8-2-decision-v1"

TRADE_ACTIONS = {
    "BUY_UP_SELL_BEFORE_CLOSE",
    "BUY_DOWN_SELL_BEFORE_CLOSE",
}
ALLOWED_RANK_VETO_REASON_CODES = {
    "policy_selected_no_trade",
    "v8_1_veto_to_no_trade",
}
FORBIDDEN_DECISION_FIELD_FRAGMENTS = {
    "outcome",
    "pnl",
    "settlement",
    "target_after",
    "resolved",
    "oracle",
    "future_return",
}


@dataclass(frozen=True, slots=True)
class SupportPreservingOverlayV82Config:
    run_id: str
    output_dir: str
    profile_path: str
    expected_profile_sha256: str
    historical_manifest_path: str
    expected_historical_manifest_sha256: str
    implementation_commit: str
    evaluation_started_ts: int
    overwrite_existing: bool = False


def validate_support_preserving_overlay_v8_2_profile(
    profile: dict[str, Any],
) -> None:
    checks = {
        "identity": profile.get("schema_version") == PROFILE_SCHEMA_VERSION
        and profile.get("issue_number") == 247
        and profile.get("candidate_name") == CANDIDATE_NAME
        and profile.get(
            "preregistered_before_implementation_and_historical_target_access"
        )
        is True,
        "policy": profile.get("policy_contract")
        == {
            "allowed_rank_veto_reason_codes": sorted(
                ALLOWED_RANK_VETO_REASON_CODES
            ),
            "candidate_primary": "adaptive_support_controller_v8_1",
            "fallback_baseline": "p_up_semantic_compatibility_v6_7",
            "fallback_requires_independent_full_guard_pass": True,
            "fallback_trigger": "v8_1_rank_threshold_abstention_only",
            "full_execution_guard_unchanged": True,
            "model_threshold_quantile_cost_sizing_or_guard_changed": False,
            "risk_blocker_bypass_allowed": False,
            "source_or_o_score_mutation_allowed": False,
        },
        "gate": profile.get("historical_gate")
        == {
            "candidate_minus_v6_7_largest_winner_removed_after_cost_pnl_minimum_inclusive": 0.0,
            "candidate_minus_v6_7_total_after_cost_pnl_minimum_inclusive": 0.0,
            "candidate_support_not_below_v6_7": True,
            "equality_passes_noninferiority": True,
            "side_quota_enabled": False,
        },
        "lineage": profile.get("lineage", {}).get(
            "issue246_outcomes_allowed_for_v8_2"
        )
        is False,
        "safety": profile.get("safety") == _expected_safety(),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("#247 v8.2 profile invalid: " + ", ".join(blockers))


def select_support_preserving_overlay_decision(
    *,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Select v8.1 first, then v6.7 only for a rank-threshold abstention."""

    _validate_decision_input(candidate, label="v8.1")
    _validate_decision_input(baseline, label="v6.7")
    if candidate["market_id"] != baseline["market_id"]:
        raise ValueError("#247 candidate/baseline market mismatch")

    candidate_action = str(candidate["selected_action"])
    baseline_action = str(baseline["selected_action"])
    candidate_allowed = candidate["execution_guard_order_allowed"] is True
    baseline_allowed = baseline["execution_guard_order_allowed"] is True
    candidate_blockers = set(candidate["execution_blocking_reason_codes"])
    rank_veto_only = (
        candidate_action == "NO_TRADE"
        and candidate_allowed is False
        and candidate.get("rank_abstention_passed") is False
        and candidate.get("point_selected_action") in TRADE_ACTIONS
        and bool(candidate_blockers)
        and candidate_blockers <= ALLOWED_RANK_VETO_REASON_CODES
    )

    if candidate_allowed and candidate_action in TRADE_ACTIONS:
        action = candidate_action
        side = str(candidate["selected_side"])
        source = "v8_1_primary"
        allowed = True
        reason_codes = ["v8_1_primary_full_guard_passed"]
    elif rank_veto_only and baseline_allowed and baseline_action in TRADE_ACTIONS:
        action = baseline_action
        side = str(baseline["selected_side"])
        source = "v6_7_rank_abstention_fallback"
        allowed = True
        reason_codes = [
            "v8_1_rank_threshold_abstention",
            "v6_7_independent_full_guard_passed",
        ]
    else:
        action = "NO_TRADE"
        side = "NONE"
        source = "fail_closed_no_trade"
        allowed = False
        reason_codes = _no_trade_reason_codes(
            candidate=candidate,
            baseline=baseline,
            rank_veto_only=rank_veto_only,
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
        "selection_reason_codes": reason_codes,
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
        "fallback_applied": source == "v6_7_rank_abstention_fallback",
        "fallback_requires_independent_full_guard_pass": True,
        "risk_blocker_bypass_used": False,
        "full_execution_guard_unchanged": True,
        "target_or_outcome_used_for_selection": False,
        "source_score_mutated": False,
        **_v7_0_blocked_safety_fields(),
    }
    _assert_target_free_decision(decision)
    decision["overlay_decision_id"] = canonical_json_sha256(decision)
    return decision


def build_historical_support_preserving_overlay_v8_2(
    rows: list[dict[str, Any]],
    *,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Freeze overlay decisions, then evaluate the preregistered historical gate."""

    validate_support_preserving_overlay_v8_2_profile(profile)
    decisions: list[dict[str, Any]] = []
    rows_by_market: dict[str, dict[str, Any]] = {}
    for row in rows:
        market_id = str(row["market_id"])
        if market_id in rows_by_market:
            raise ValueError(f"#247 duplicate historical market: {market_id}")
        rows_by_market[market_id] = row
        decisions.append(
            select_support_preserving_overlay_decision(
                candidate=_historical_candidate_projection(row),
                baseline=_historical_baseline_projection(row),
            )
        )

    candidate_selected_rows: list[dict[str, Any]] = []
    baseline_selected_rows: list[dict[str, Any]] = []
    candidate_by_market: dict[str, float] = {}
    baseline_by_market: dict[str, float] = {}
    frozen_size = 0.2
    for decision in decisions:
        row = rows_by_market[str(decision["market_id"])]
        market_id = str(decision["market_id"])
        candidate_target = _candidate_target_for_frozen_decision(
            decision, historical_row=row
        )
        baseline_target = (
            float(row["baseline_target_after_cost_net_pnl_per_contract"])
            if row["baseline_execution_guard_order_allowed"] is True
            else 0.0
        )
        candidate_pnl = candidate_target * frozen_size
        baseline_pnl = baseline_target * frozen_size
        candidate_by_market[market_id] = candidate_pnl
        baseline_by_market[market_id] = baseline_pnl
        if decision["execution_guard_order_allowed"] is True:
            candidate_selected_rows.append(
                _evaluation_row(
                    decision=decision,
                    target=candidate_target,
                    pnl=candidate_pnl,
                    source=str(decision["selection_source"]),
                )
            )
        if row["baseline_execution_guard_order_allowed"] is True:
            baseline_selected_rows.append(
                {
                    "market_id": market_id,
                    "decision_ts": row.get("baseline_decision_ts"),
                    "action": row["baseline_action"],
                    "side": row["baseline_side"],
                    "selection_source": "v6_7_baseline",
                    "target_after_cost_net_pnl_per_contract": baseline_target,
                    "fixed_position_size": frozen_size,
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
    support_delta = len(candidate_selected_rows) - len(baseline_selected_rows)
    checks = {
        "candidate_total_pnl_noninferior_to_v6_7": total_delta >= 0.0,
        "candidate_largest_winner_removed_pnl_noninferior_to_v6_7": (
            lwr_delta >= 0.0
        ),
        "candidate_guard_accepted_support_not_below_v6_7": support_delta >= 0,
        "overlay_decisions_frozen_before_historical_target_access": all(
            decision["target_or_outcome_used_for_selection"] is False
            for decision in decisions
        ),
        "source_scores_unchanged": all(
            decision["source_score_mutated"] is False
            for decision in decisions
        ),
        "risk_blocker_bypass_absent": all(
            decision["risk_blocker_bypass_used"] is False
            for decision in decisions
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
        "overlay_decisions_frozen_before_historical_target_access": (
            "historical_target_accessed_before_decision_freeze"
        ),
        "source_scores_unchanged": "source_score_mutation_detected",
        "risk_blocker_bypass_absent": "execution_risk_blocker_bypass_detected",
    }
    blockers = [
        reason_map[name] for name, passed in checks.items() if not passed
    ]
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "historical_market_count": len(rows),
        "candidate_guard_accepted_market_count": len(candidate_selected_rows),
        "v6_7_guard_accepted_market_count": len(baseline_selected_rows),
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
        "selection_source_distribution": dict(
            sorted(Counter(row["selection_source"] for row in decisions).items())
        ),
        "selected_action_distribution": dict(
            sorted(Counter(row["selected_action"] for row in decisions).items())
        ),
        "selected_side_distribution_diagnostic": dict(
            sorted(
                Counter(
                    row["selected_side"]
                    for row in decisions
                    if row["selected_side"] != "NONE"
                ).items()
            )
        ),
        "side_quota_enabled": False,
        "historical_targets_opened_only_after_overlay_decision_freeze": True,
        "issue246_outcomes_opened": False,
        "threshold_model_quantile_cost_sizing_or_guard_changed": False,
        "checks": checks,
        "historical_noninferiority_gate_passed": not blockers,
        "historical_gate_blocking_reason_codes": blockers,
        "future_target_free_canary_allowed": not blockers,
        "future_holdout_collection_allowed": False,
        **_v7_0_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return {
        "decisions": decisions,
        "candidate_selected_rows": candidate_selected_rows,
        "baseline_selected_rows": baseline_selected_rows,
        "report": report,
    }


def run_support_preserving_overlay_v8_2_historical_gate(
    config: SupportPreservingOverlayV82Config,
) -> dict[str, Any]:
    profile_path = Path(config.profile_path).resolve()
    historical_manifest_path = Path(config.historical_manifest_path).resolve()
    _verify_pin(profile_path, config.expected_profile_sha256, "#247 profile")
    _verify_pin(
        historical_manifest_path,
        config.expected_historical_manifest_sha256,
        "#247 v8.1 historical manifest",
    )
    profile = _load_json(profile_path)
    validate_support_preserving_overlay_v8_2_profile(profile)
    if (
        profile["lineage"]["v8_1_historical_manifest_sha256"]
        != config.expected_historical_manifest_sha256
    ):
        raise ValueError("#247 historical manifest lineage mismatch")
    historical_manifest = _load_json(historical_manifest_path)
    rows_descriptor = historical_manifest["prequential_rows"]
    rows_path = Path(rows_descriptor["path"])
    _verify_pin(
        rows_path,
        rows_descriptor["sha256"],
        "#247 historical prequential rows",
    )
    result = build_historical_support_preserving_overlay_v8_2(
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

    run_dir = Path(config.output_dir).resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    outputs = {
        "report": run_dir / "v8_2_historical_noninferiority_report.json",
        "report_markdown": (
            run_dir / "v8_2_historical_noninferiority_report.md"
        ),
        "decision_rows": run_dir / "v8_2_historical_frozen_decisions.jsonl",
        "candidate_selected_rows": (
            run_dir / "v8_2_historical_candidate_selected_rows.jsonl"
        ),
        "v6_7_baseline_selected_rows": (
            run_dir / "v8_2_historical_v6_7_selected_rows.jsonl"
        ),
    }
    _write_json(outputs["report"], report)
    _write_text(outputs["report_markdown"], _report_markdown(report))
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
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        "profile": _descriptor(profile_path),
        "v8_1_historical_manifest": _descriptor(historical_manifest_path),
        **{name: _descriptor(path) for name, path in outputs.items()},
        "historical_noninferiority_gate_passed": report[
            "historical_noninferiority_gate_passed"
        ],
        "historical_gate_blocking_reason_codes": report[
            "historical_gate_blocking_reason_codes"
        ],
        "future_target_free_canary_allowed": report[
            "future_target_free_canary_allowed"
        ],
        "future_holdout_collection_allowed": False,
        "issue246_outcomes_opened": False,
        **_v7_0_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v8_2_historical_noninferiority_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "report": report,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "outputs": outputs,
    }


def _historical_candidate_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "market_id": row["market_id"],
        "decision_ts": row.get("market_close_ts"),
        "selected_action": row["selected_action"],
        "selected_side": row["selected_side"],
        "execution_guard_order_allowed": row[
            "candidate_execution_guard_order_allowed"
        ],
        "execution_blocking_reason_codes": row[
            "candidate_execution_blocking_reason_codes"
        ],
        "rank_abstention_passed": row["rank_abstention_passed"],
        "point_selected_action": row["point_selected_action"],
    }


def _historical_baseline_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "market_id": row["market_id"],
        "decision_ts": row.get("baseline_decision_ts"),
        "selected_action": row["baseline_action"],
        "selected_side": row["baseline_side"],
        "execution_guard_order_allowed": row[
            "baseline_execution_guard_order_allowed"
        ],
        "execution_blocking_reason_codes": row[
            "baseline_execution_blocking_reason_codes"
        ],
        "rank_abstention_passed": None,
        "point_selected_action": row["baseline_action"],
    }


def _candidate_target_for_frozen_decision(
    decision: dict[str, Any],
    *,
    historical_row: dict[str, Any],
) -> float:
    source = decision["selection_source"]
    if source == "v8_1_primary":
        return float(
            historical_row[
                "selected_target_after_cost_net_pnl_per_contract"
            ]
        )
    if source == "v6_7_rank_abstention_fallback":
        return float(
            historical_row[
                "baseline_target_after_cost_net_pnl_per_contract"
            ]
        )
    return 0.0


def _evaluation_row(
    *,
    decision: dict[str, Any],
    target: float,
    pnl: float,
    source: str,
) -> dict[str, Any]:
    return {
        "market_id": decision["market_id"],
        "decision_ts": decision["decision_ts"],
        "action": decision["selected_action"],
        "side": decision["selected_side"],
        "selection_source": source,
        "target_after_cost_net_pnl_per_contract": target,
        "fixed_position_size": 0.2,
        "after_cost_net_pnl_at_frozen_size": pnl,
        "target_used_as_decision_time_input": False,
        "target_opened_only_after_overlay_decision_freeze": True,
    }


def _validate_decision_input(row: dict[str, Any], *, label: str) -> None:
    required = {
        "market_id",
        "selected_action",
        "selected_side",
        "execution_guard_order_allowed",
        "execution_blocking_reason_codes",
    }
    missing = sorted(required - row.keys())
    if missing:
        raise ValueError(f"#247 {label} decision missing fields: {missing}")
    if row["selected_action"] not in {*TRADE_ACTIONS, "NO_TRADE"}:
        raise ValueError(f"#247 {label} action invalid")
    if row["selected_side"] not in {"UP", "DOWN", "NONE"}:
        raise ValueError(f"#247 {label} side invalid")
    if not isinstance(row["execution_guard_order_allowed"], bool):
        raise ValueError(f"#247 {label} guard result invalid")
    if not isinstance(row["execution_blocking_reason_codes"], list):
        raise ValueError(f"#247 {label} blocker list invalid")


def _no_trade_reason_codes(
    *,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    rank_veto_only: bool,
) -> list[str]:
    reasons: list[str] = []
    if candidate["execution_guard_order_allowed"] is False:
        reasons.extend(candidate["execution_blocking_reason_codes"])
    if not rank_veto_only:
        reasons.append("v8_1_not_rank_threshold_abstention_only")
    if baseline["execution_guard_order_allowed"] is False:
        reasons.extend(
            f"v6_7_{code}"
            for code in baseline["execution_blocking_reason_codes"]
        )
        reasons.append("v6_7_independent_full_guard_failed")
    if baseline["selected_action"] not in TRADE_ACTIONS:
        reasons.append("v6_7_trade_action_unavailable")
    return sorted(set(reasons or {"support_preserving_overlay_no_trade"}))


def _assert_target_free_decision(decision: dict[str, Any]) -> None:
    forbidden = [
        key
        for key in decision
        if any(fragment in key.lower() for fragment in FORBIDDEN_DECISION_FIELD_FRAGMENTS)
        and key != "target_or_outcome_used_for_selection"
    ]
    if forbidden or decision["target_or_outcome_used_for_selection"] is not False:
        raise ValueError(f"#247 target field entered decision: {forbidden}")


def _expected_safety() -> dict[str, bool]:
    safety = _v7_0_blocked_safety_fields()
    safety["paper_only"] = True
    return safety


def _report_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Execution Layer v2 v8.2 Historical Non-Inferiority",
            "",
            f"- candidate: `{report['candidate_name']}`",
            f"- markets: `{report['historical_market_count']}`",
            (
                "- candidate / v6.7 support: "
                f"`{report['candidate_guard_accepted_market_count']}` / "
                f"`{report['v6_7_guard_accepted_market_count']}`"
            ),
            (
                "- candidate / v6.7 PnL: "
                f"`{report['candidate_total_after_cost_net_pnl_at_frozen_size']}` / "
                f"`{report['v6_7_total_after_cost_net_pnl_at_frozen_size']}`"
            ),
            (
                "- total PnL delta: "
                f"`{report['candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size']}`"
            ),
            (
                "- largest-winner-removed delta: "
                f"`{report['candidate_minus_v6_7_largest_winner_removed_after_cost_net_pnl_at_frozen_size']}`"
            ),
            (
                "- selection sources: "
                f"`{report['selection_source_distribution']}`"
            ),
            (
                "- historical non-inferiority passed: "
                f"`{str(report['historical_noninferiority_gate_passed']).lower()}`"
            ),
            (
                "- blockers: "
                f"`{report['historical_gate_blocking_reason_codes']}`"
            ),
            "- issue #246 outcomes opened: `false`",
            "- future holdout collection allowed: `false`",
            "- paper/live/write/wallet/capital remain blocked.",
            "",
        ]
    )


__all__ = [
    "SupportPreservingOverlayV82Config",
    "build_historical_support_preserving_overlay_v8_2",
    "run_support_preserving_overlay_v8_2_historical_gate",
    "select_support_preserving_overlay_decision",
    "validate_support_preserving_overlay_v8_2_profile",
]
