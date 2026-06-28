"""Promotion support gates for SELL_BEFORE_CLOSE source candidates."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.contracts import compact_safety_fields
from bigan.v8.polymarket.training.sell_before_close_source_candidates import (
    SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_CANDIDATE_NAME,
    SELL_BEFORE_CLOSE_SUPPORT_AWARE_P_UP_ALIGNED_CANDIDATE_NAME,
)

SELL_BEFORE_CLOSE_PROMOTION_SUPPORT_GATE_SCHEMA_VERSION = (
    "bigan-v8-polymarket-sell-before-close-promotion-support-gate-v1"
)
SELL_BEFORE_CLOSE_PROMOTION_SUPPORT_THRESHOLDS = {
    "min_promotion_entry_decision_count": 20,
    "min_promotion_market_count": 10,
    "min_promotion_side_count": 2,
    "min_promotion_sell_decision_count": 20,
    "min_promotion_total_pnl": 0.10,
    "min_promotion_mean_pnl_per_entry": 0.0025,
    "max_promotion_drawdown_to_pnl_ratio": None,
    "min_promotion_replay_duration_minutes": None,
}


def build_sell_before_close_promotion_support_gate_report(
    *,
    action_family_counterfactual_replays: tuple[dict[str, Any], ...],
    sell_before_close_exit_reliability: dict[str, Any],
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build fail-closed support/coverage evidence for source promotion."""

    gate_thresholds = _thresholds(thresholds)
    replay_by_variant = {
        replay["variant"]: replay for replay in action_family_counterfactual_replays
    }
    rows = []
    for candidate_report in sell_before_close_exit_reliability.get(
        "candidate_reports",
        [],
    ):
        candidate_name = candidate_report["candidate_name"]
        replay = replay_by_variant.get(candidate_name)
        if replay is None:
            continue
        rows.append(
            evaluate_sell_before_close_promotion_support(
                candidate_name=candidate_name,
                decisions=replay["decisions"],
                replay_report=replay["replay_report"],
                exit_reliability_summary=candidate_report["summary"],
                thresholds=gate_thresholds,
            )
        )
    selected = _selected_candidate_row(rows)
    report = {
        "schema_version": SELL_BEFORE_CLOSE_PROMOTION_SUPPORT_GATE_SCHEMA_VERSION,
        "candidate_name": selected.get(
            "candidate_name",
            SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_CANDIDATE_NAME,
        ),
        "thresholds": gate_thresholds,
        "candidate_count": len(rows),
        "candidate_rows": rows,
        "i_vs_j_vs_k_promotion_support_comparison": rows,
        "i_vs_j_vs_k_vs_l_promotion_support_comparison": rows,
        "entry_decision_count": selected.get("entry_decision_count", 0),
        "sell_decision_count": selected.get("sell_decision_count", 0),
        "unique_market_count": selected.get("unique_market_count", 0),
        "side_count": selected.get("side_count", 0),
        "side_distribution": selected.get("side_distribution", {}),
        "total_pnl": selected.get("total_pnl", 0.0),
        "mean_pnl_per_entry": selected.get("mean_pnl_per_entry", 0.0),
        "max_drawdown": selected.get("max_drawdown", 0.0),
        "replay_residual_settlement_drag": selected.get(
            "replay_residual_settlement_drag",
            0.0,
        ),
        "positions_opened_but_not_closed_before_settlement": selected.get(
            "positions_opened_but_not_closed_before_settlement",
            0,
        ),
        "candidate_scoped_p_up_action_disagreement_rate": selected.get(
            "candidate_scoped_p_up_action_disagreement_rate",
            0.0,
        ),
        "support_gate_passed": selected.get("support_gate_passed", False),
        "support_gate_reason_codes": selected.get("support_gate_reason_codes", []),
        "promotion_support_eligible": selected.get(
            "promotion_support_eligible",
            False,
        ),
        "promotion_evidence_eligible": selected.get(
            "promotion_support_eligible",
            False,
        ),
        "paper_run_resume_allowed": False,
        **compact_safety_fields(),
    }
    report["sell_before_close_promotion_support_gate_report_id"] = (
        canonical_json_sha256(report)
    )
    return report


def evaluate_sell_before_close_promotion_support(
    *,
    candidate_name: str,
    decisions: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    replay_report: dict[str, Any],
    exit_reliability_summary: dict[str, Any] | None = None,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate replay support without mutating promotion state."""

    gate_thresholds = _thresholds(thresholds)
    entry_decisions = [
        decision
        for decision in decisions
        if str(decision.get("action")) in {"BUY_UP", "BUY_DOWN"}
        and str(decision.get("intended_exit_policy")) == "sell_before_close"
    ]
    sell_decisions = [
        decision
        for decision in decisions
        if str(decision.get("action")) in {"SELL_UP", "SELL_DOWN"}
    ]
    side_distribution = Counter(_entry_side(decision) for decision in entry_decisions)
    side_distribution = Counter(
        {side: count for side, count in side_distribution.items() if side}
    )
    entry_count = len(entry_decisions)
    sell_count = len(sell_decisions)
    market_count = len({str(decision.get("market_id")) for decision in entry_decisions})
    side_count = len(side_distribution)
    total_pnl = float(replay_report.get("total_polymarket_pnl", 0.0))
    mean_pnl_per_entry = 0.0 if entry_count <= 0 else total_pnl / entry_count
    max_drawdown = float(replay_report.get("max_drawdown", 0.0))
    residual_drag = min(0.0, float(replay_report.get("settlement_pnl", 0.0)))
    residual_count = int(replay_report.get("settled_position_count", 0))
    p_up_rate = 0.0
    if exit_reliability_summary:
        residual_drag = float(
            exit_reliability_summary.get(
                "replay_residual_settlement_drag",
                residual_drag,
            )
        )
        residual_count = int(
            exit_reliability_summary.get(
                "positions_opened_but_not_closed_before_settlement",
                residual_count,
            )
        )
        p_up_rate = float(
            exit_reliability_summary.get(
                "candidate_scoped_p_up_action_disagreement_rate",
                0.0,
            )
        )
    reason_codes = []
    if entry_count < int(gate_thresholds["min_promotion_entry_decision_count"]):
        reason_codes.append("promotion_replay_entry_support_insufficient")
    if market_count < int(gate_thresholds["min_promotion_market_count"]):
        reason_codes.append("promotion_replay_market_support_insufficient")
    if side_count < int(gate_thresholds["min_promotion_side_count"]):
        reason_codes.append("promotion_replay_side_coverage_insufficient")
    if sell_count < int(gate_thresholds["min_promotion_sell_decision_count"]):
        reason_codes.append("promotion_replay_sell_support_insufficient")
    if total_pnl < float(gate_thresholds["min_promotion_total_pnl"]):
        reason_codes.append("promotion_replay_total_pnl_insufficient")
    if mean_pnl_per_entry < float(
        gate_thresholds["min_promotion_mean_pnl_per_entry"]
    ):
        reason_codes.append("promotion_replay_mean_pnl_per_entry_insufficient")
    max_drawdown_ratio = None
    ratio_limit = gate_thresholds.get("max_promotion_drawdown_to_pnl_ratio")
    if ratio_limit is not None and total_pnl > 0.0:
        max_drawdown_ratio = max_drawdown / total_pnl
        if max_drawdown_ratio > float(ratio_limit):
            reason_codes.append("promotion_replay_drawdown_too_large")
    support_passed = not reason_codes
    return {
        "candidate_name": candidate_name,
        "entry_decision_count": entry_count,
        "entry_count": entry_count,
        "sell_decision_count": sell_count,
        "sell_count": sell_count,
        "unique_market_count": market_count,
        "market_count": market_count,
        "side_count": side_count,
        "side_distribution": dict(sorted(side_distribution.items())),
        "total_pnl": total_pnl,
        "mean_pnl_per_entry": mean_pnl_per_entry,
        "max_drawdown": max_drawdown,
        "max_drawdown_to_pnl_ratio": max_drawdown_ratio,
        "replay_residual_settlement_drag": residual_drag,
        "residual_count": residual_count,
        "positions_opened_but_not_closed_before_settlement": residual_count,
        "candidate_scoped_p_up_action_disagreement_rate": p_up_rate,
        "support_gate_passed": support_passed,
        "support_gate_reason_codes": sorted(set(reason_codes)),
        "promotion_support_eligible": support_passed,
        "promotion_evidence_eligible": support_passed,
        "paper_run_resume_allowed": False,
        **compact_safety_fields(),
    }


def sell_before_close_promotion_support_gate_summary(
    report: dict[str, Any],
) -> dict[str, Any]:
    """Return compact support-gate evidence for embedding."""

    fields = (
        "schema_version",
        "candidate_name",
        "thresholds",
        "entry_decision_count",
        "sell_decision_count",
        "unique_market_count",
        "side_count",
        "side_distribution",
        "total_pnl",
        "mean_pnl_per_entry",
        "max_drawdown",
        "replay_residual_settlement_drag",
        "positions_opened_but_not_closed_before_settlement",
        "candidate_scoped_p_up_action_disagreement_rate",
        "support_gate_passed",
        "support_gate_reason_codes",
        "threshold_selection_passed",
        "threshold_selection_failed",
        "threshold_selection_failure_reason_codes",
        "support_aware_threshold_selection_failed",
        "threshold_selection_failure_interpretation",
        "recommended_next_action",
        "failure_attribution_report_path",
        "failure_attribution_report_sha256",
        "promotion_support_eligible",
        "promotion_evidence_eligible",
        "paper_run_resume_allowed",
    )
    return {field: report.get(field) for field in fields}


def sell_before_close_promotion_support_gate_markdown(
    report: dict[str, Any],
) -> str:
    """Render support-gate report markdown."""

    lines = [
        "# SELL_BEFORE_CLOSE Promotion Support Gate",
        "",
        f"- candidate_name: `{report['candidate_name']}`",
        f"- support_gate_passed: `{str(report['support_gate_passed']).lower()}`",
        "- support_gate_reason_codes: "
        f"`{json.dumps(report['support_gate_reason_codes'])}`",
        "- threshold_selection_passed: "
        f"`{str(report.get('threshold_selection_passed')).lower()}`",
        "- threshold_selection_failed: "
        f"`{str(report.get('threshold_selection_failed')).lower()}`",
        "- threshold_selection_failure_reason_codes: "
        f"`{json.dumps(report.get('threshold_selection_failure_reason_codes', []))}`",
        "- threshold_selection_failure_interpretation: "
        f"`{report.get('threshold_selection_failure_interpretation')}`",
        f"- recommended_next_action: `{report.get('recommended_next_action')}`",
        "- failure_attribution_report_path: "
        f"`{report.get('failure_attribution_report_path')}`",
        "- failure_attribution_report_sha256: "
        f"`{report.get('failure_attribution_report_sha256')}`",
        "- promotion_evidence_eligible: "
        f"`{str(report['promotion_evidence_eligible']).lower()}`",
        f"- paper_run_resume_allowed: `{str(report['paper_run_resume_allowed']).lower()}`",
        f"- thresholds: `{json.dumps(report['thresholds'], sort_keys=True)}`",
        "",
        "## I/J/K Support Comparison",
        "",
        "| candidate | entries | markets | sides | sells | total_pnl | mean_pnl_per_entry | max_drawdown | residual | p_up_disagreement | support | reasons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in report.get("i_vs_j_vs_k_promotion_support_comparison", []):
        lines.append(
            "| {candidate} | {entries} | {markets} | {sides} | {sells} | "
            "{total:.6f} | {mean:.6f} | {drawdown:.6f} | {residual} | "
            "{p_up:.6f} | {support} | {reasons} |".format(
                candidate=row["candidate_name"],
                entries=row["entry_decision_count"],
                markets=row["unique_market_count"],
                sides=row["side_count"],
                sells=row["sell_decision_count"],
                total=row["total_pnl"],
                mean=row["mean_pnl_per_entry"],
                drawdown=row["max_drawdown"],
                residual=row["positions_opened_but_not_closed_before_settlement"],
                p_up=row["candidate_scoped_p_up_action_disagreement_rate"],
                support=str(row["support_gate_passed"]).lower(),
                reasons=", ".join(row["support_gate_reason_codes"]) or "none",
            )
        )
    lines.extend(
        [
            "",
            "- paper_only: true",
            "- capital_at_risk: false",
            "- polymarket_write_enabled: false",
            "- wallet_signing_enabled: false",
            "",
        ]
    )
    return "\n".join(lines)


def _thresholds(thresholds: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(SELL_BEFORE_CLOSE_PROMOTION_SUPPORT_THRESHOLDS)
    if thresholds:
        merged.update(thresholds)
    return merged


def _selected_candidate_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if (
            row["candidate_name"]
            == SELL_BEFORE_CLOSE_SUPPORT_AWARE_P_UP_ALIGNED_CANDIDATE_NAME
        ):
            return row
    for row in rows:
        if row["candidate_name"] == SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_CANDIDATE_NAME:
            return row
    return rows[-1] if rows else {}


def _entry_side(decision: dict[str, Any]) -> str:
    selected = str(decision.get("selected_outcome") or "")
    if selected in {"UP", "DOWN"}:
        return selected
    action = str(decision.get("action") or "")
    if action.endswith("_UP"):
        return "UP"
    if action.endswith("_DOWN"):
        return "DOWN"
    return ""
