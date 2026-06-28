"""SELL_BEFORE_CLOSE replay exit reliability diagnostics."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from bigan.v8.polymarket.action_value_guards import action_value_bucket_payload
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.execution_ev import PolymarketEVDecision
from bigan.v8.polymarket.training.contracts import (
    PolymarketPolicyPrediction,
    PolymarketPolicyTrainingConfig,
    compact_safety_fields,
)
from bigan.v8.polymarket.training.sell_before_close_source_candidates import (
    SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_CANDIDATE_NAME,
    SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_EXIT_POLICY,
    SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_THRESHOLDS,
    SELL_BEFORE_CLOSE_ONLY_SOURCE_CANDIDATE_NAME,
    SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_CANDIDATE_NAME,
    SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_THRESHOLDS,
)

SELL_BEFORE_CLOSE_EXIT_RELIABILITY_SCHEMA_VERSION = (
    "bigan-v8-polymarket-sell-before-close-exit-reliability-v1"
)
P_UP_ACTION_DISAGREEMENT_FAIL_THRESHOLD = 0.50
P_UP_MATERIAL_DISAGREEMENT_THRESHOLD = 0.55
EXIT_VARIANTS = (
    "first_executable_exit_after_entry",
    "scheduled_exit_before_close",
    "forced_preclose_exit_if_executable",
    "take_profit_stop_loss_exit",
)
GROUP_FIELDS = (
    "exit_lifecycle_class",
    "entry_action",
    "entry_side",
    "time_to_close_bucket",
    "price_bucket",
    "queue_fill_probability_bucket",
    "executable_liquidity_bucket",
    "missed_exit_reason",
)


def build_sell_before_close_exit_reliability_guard_decisions(
    *,
    predictions: tuple[PolymarketPolicyPrediction, ...],
    config: PolymarketPolicyTrainingConfig,
    thresholds: dict[str, float] | None = None,
    exit_policy: str = SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_EXIT_POLICY,
    candidate_name: str = SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_CANDIDATE_NAME,
    p_up_side_alignment_filter_enabled: bool = False,
) -> tuple[tuple[PolymarketEVDecision, ...], dict[str, Any]]:
    """Build guarded SELL_BEFORE_CLOSE decisions without live writes or future labels."""

    guard_thresholds = dict(
        thresholds or SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_THRESHOLDS
    )
    if candidate_name == SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_CANDIDATE_NAME:
        merged = dict(SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_THRESHOLDS)
        merged.update(guard_thresholds)
        guard_thresholds = merged
        p_up_side_alignment_filter_enabled = True
    if exit_policy != SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_EXIT_POLICY:
        raise ValueError(f"unsupported exit reliability guard policy: {exit_policy}")
    decisions = []
    positions: dict[str, dict[str, Any]] = {}
    entries_per_market: dict[str, int] = defaultdict(int)
    last_exit_ts_by_market: dict[str, int] = {}
    before_guard = 0
    after_exit_guard = 0
    after_p_up_alignment = 0
    after_guard = 0
    blocked_count = 0
    quality_blocked_count = 0
    p_up_alignment_blocked_count = 0
    reentry_blocked_count = 0
    p_up_disagreement_count = 0
    p_up_disagreement_denominator = 0
    reason_counts: dict[str, int] = defaultdict(int)
    for prediction in sorted(
        predictions,
        key=lambda row: (int(row.decision_ts), str(row.market_id)),
    ):
        position = positions.setdefault(prediction.market_id, _empty_guard_position())
        open_side = _open_guard_side(position)
        if open_side is not None:
            decision, closed = _guard_exit_decision(
                prediction=prediction,
                config=config,
                position=position,
                side=open_side,
                exit_policy=exit_policy,
            )
            decisions.append(decision)
            for reason in decision.reason_codes:
                reason_counts[reason] += 1
            if closed:
                last_exit_ts_by_market[prediction.market_id] = int(
                    prediction.decision_ts
                )
                positions[prediction.market_id] = _empty_guard_position()
            continue

        selected_action = str(prediction.calibrated_best_policy_action)
        if selected_action not in {
            "BUY_UP_SELL_BEFORE_CLOSE",
            "BUY_DOWN_SELL_BEFORE_CLOSE",
        }:
            decision = _guard_no_trade_decision(
                prediction=prediction,
                config=config,
                reason_codes=(
                    "action_value_no_sell_before_close_entry_selected",
                    "exit_reliability_guard_candidate",
                ),
            )
            decisions.append(decision)
            for reason in decision.reason_codes:
                reason_counts[reason] += 1
            continue
        before_guard += 1
        guard = _entry_guard_assessment(
            prediction=prediction,
            action=selected_action,
            config=config,
            thresholds=guard_thresholds,
        )
        if not guard["passed"]:
            blocked_count += 1
            quality_blocked_count += 1
            decision = _guard_no_trade_decision(
                prediction=prediction,
                config=config,
                reason_codes=tuple(guard["reason_codes"]),
                entry_policy_action=selected_action,
            )
            decisions.append(decision)
            for reason in decision.reason_codes:
                reason_counts[reason] += 1
            continue
        after_exit_guard += 1
        p_up_alignment = _p_up_alignment_assessment(
            prediction=prediction,
            action=selected_action,
            thresholds=guard_thresholds,
            enabled=p_up_side_alignment_filter_enabled,
        )
        if not p_up_alignment["passed"]:
            blocked_count += 1
            p_up_alignment_blocked_count += 1
            decision = _guard_no_trade_decision(
                prediction=prediction,
                config=config,
                reason_codes=tuple(p_up_alignment["reason_codes"]),
                entry_policy_action=selected_action,
            )
            decisions.append(decision)
            for reason in decision.reason_codes:
                reason_counts[reason] += 1
            continue
        after_p_up_alignment += 1
        turnover = _turnover_guard_assessment(
            market_id=prediction.market_id,
            decision_ts=int(prediction.decision_ts),
            thresholds=guard_thresholds,
            entries_per_market=entries_per_market,
            last_exit_ts_by_market=last_exit_ts_by_market,
        )
        if not turnover["passed"]:
            blocked_count += 1
            if "entry_blocked_reentry_cooldown" in turnover["reason_codes"]:
                reentry_blocked_count += 1
            decision = _guard_no_trade_decision(
                prediction=prediction,
                config=config,
                reason_codes=tuple(turnover["reason_codes"]),
                entry_policy_action=selected_action,
            )
            decisions.append(decision)
            for reason in decision.reason_codes:
                reason_counts[reason] += 1
            continue
        after_guard += 1
        decision = _guard_entry_decision(
            prediction=prediction,
            config=config,
            action=selected_action,
            guard_reason_codes=tuple(
                dict.fromkeys(
                    (
                        *guard["reason_codes"],
                        *p_up_alignment["reason_codes"],
                        *turnover["reason_codes"],
                    )
                )
            ),
        )
        decisions.append(decision)
        for reason in decision.reason_codes:
            reason_counts[reason] += 1
        entries_per_market[prediction.market_id] += 1
        p_up_disagreement_denominator += 1
        if _action_p_up_disagrees(
            action=selected_action,
            p_up=_prediction_p_up(prediction),
        ):
            p_up_disagreement_count += 1
        _open_guard_position(position=position, decision=decision, side=guard["side"])

    p_up_rate = (
        0.0
        if p_up_disagreement_denominator == 0
        else p_up_disagreement_count / p_up_disagreement_denominator
    )
    summary = {
        "candidate_name": candidate_name,
        "exit_reliability_guard_enabled": True,
        "p_up_side_alignment_filter_enabled": p_up_side_alignment_filter_enabled,
        "exit_policy": exit_policy,
        "entry_filter_thresholds": guard_thresholds,
        "entry_decision_count_before_guard": before_guard,
        "entry_decision_count_after_exit_guard": after_exit_guard,
        "entry_decision_count_after_p_up_alignment": after_p_up_alignment,
        "entry_decision_count_after_guard": after_guard,
        "entry_filter_blocked_count": blocked_count,
        "entry_filter_blocked_by_p_up_alignment_count": (
            p_up_alignment_blocked_count
        ),
        "entry_filter_blocked_by_quality_count": quality_blocked_count,
        "reentry_cooldown_seconds": float(
            guard_thresholds.get("min_reentry_cooldown_seconds", 0.0)
        ),
        "reentry_blocked_count": reentry_blocked_count,
        "entries_per_market_distribution": dict(
            sorted(Counter(entries_per_market.values()).items())
        ),
        "candidate_scoped_p_up_action_disagreement_count": (
            p_up_disagreement_count
        ),
        "candidate_scoped_p_up_action_disagreement_denominator": (
            p_up_disagreement_denominator
        ),
        "candidate_scoped_p_up_action_disagreement_rate": p_up_rate,
        "candidate_scoped_p_up_action_disagreement_within_limit": (
            p_up_rate <= P_UP_ACTION_DISAGREEMENT_FAIL_THRESHOLD
        ),
        "reason_counts": dict(sorted(reason_counts.items())),
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        **compact_safety_fields(),
    }
    return tuple(decisions), summary


def build_sell_before_close_exit_reliability_report(
    *,
    dataset: Any,
    action_family_counterfactual_replays: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Build diagnostic-only SELL_BEFORE_CLOSE exit reliability evidence."""

    i_payload = _candidate_exit_reliability_payload(
        dataset=dataset,
        replay=_replay_by_variant(
            action_family_counterfactual_replays,
            SELL_BEFORE_CLOSE_ONLY_SOURCE_CANDIDATE_NAME,
        ),
        candidate_name=SELL_BEFORE_CLOSE_ONLY_SOURCE_CANDIDATE_NAME,
    )
    candidate_reports = [i_payload]
    j_replay = _optional_replay_by_variant(
        action_family_counterfactual_replays,
        SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_CANDIDATE_NAME,
    )
    if j_replay is not None:
        candidate_reports.append(
            _candidate_exit_reliability_payload(
                dataset=dataset,
                replay=j_replay,
                candidate_name=SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_CANDIDATE_NAME,
            )
        )
    k_replay = _optional_replay_by_variant(
        action_family_counterfactual_replays,
        SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_CANDIDATE_NAME,
    )
    if k_replay is not None:
        candidate_reports.append(
            _candidate_exit_reliability_payload(
                dataset=dataset,
                replay=k_replay,
                candidate_name=SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_CANDIDATE_NAME,
            )
        )
    summary = i_payload["summary"]
    comparison = _candidate_replay_comparison(candidate_reports)
    report = {
        "schema_version": SELL_BEFORE_CLOSE_EXIT_RELIABILITY_SCHEMA_VERSION,
        "candidate_name": SELL_BEFORE_CLOSE_ONLY_SOURCE_CANDIDATE_NAME,
        "counterfactual_replay_variant": SELL_BEFORE_CLOSE_ONLY_SOURCE_CANDIDATE_NAME,
        "diagnostic_only": True,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "diagnostic_counterfactual": True,
        "causal_replay_decision_stream_used": True,
        "uses_live_orders": False,
        "wallet_transaction_signing_used": False,
        "summary": summary,
        "position_lifecycle_rows": i_payload["position_lifecycle_rows"],
        "grouped_summaries": i_payload["grouped_summaries"],
        "multi_dimensional_group_summaries": i_payload[
            "multi_dimensional_group_summaries"
        ],
        "diagnostic_exit_variants": i_payload["diagnostic_exit_variants"],
        "candidate_report_count": len(candidate_reports),
        "candidate_reports": candidate_reports,
        "i_vs_j_replay_comparison": [
            row
            for row in comparison
            if row["candidate_name"]
            in {
                SELL_BEFORE_CLOSE_ONLY_SOURCE_CANDIDATE_NAME,
                SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_CANDIDATE_NAME,
            }
        ],
        "i_vs_j_vs_k_replay_comparison": comparison,
        "exit_reliability_guard_candidate_summary": _guard_candidate_summary(
            candidate_reports,
            candidate_name=SELL_BEFORE_CLOSE_EXIT_RELIABILITY_GUARD_CANDIDATE_NAME,
        ),
        "exit_reliability_p_up_aligned_candidate_summary": _guard_candidate_summary(
            candidate_reports,
            candidate_name=SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_CANDIDATE_NAME,
        ),
        "sell_before_close_exit_failure_interpretation": summary[
            "sell_before_close_exit_failure_interpretation"
        ],
        **compact_safety_fields(),
    }
    report["sell_before_close_exit_reliability_report_id"] = canonical_json_sha256(
        report
    )
    return report


def _candidate_exit_reliability_payload(
    *,
    dataset: Any,
    replay: dict[str, Any],
    candidate_name: str,
) -> dict[str, Any]:
    decisions = sorted(
        (dict(row) for row in replay["decisions"]),
        key=lambda row: (int(row["decision_ts"]), str(row["market_id"])),
    )
    predictions = [dict(row) for row in replay["predictions"]]
    prediction_by_key = {
        (row["market_id"], int(row["decision_ts"])): row for row in predictions
    }
    predictions_by_market: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        predictions_by_market[prediction["market_id"]].append(prediction)
    for market_predictions in predictions_by_market.values():
        market_predictions.sort(key=lambda row: int(row["decision_ts"]))

    positions = _position_lifecycle_rows(
        decisions=decisions,
        prediction_by_key=prediction_by_key,
        predictions_by_market=predictions_by_market,
        market_metadata=dataset.market_metadata,
        resolution_events=dataset.resolution_events,
    )
    summary = _replay_summary(rows=positions, replay=replay)
    p_up_disagreement = _decision_stream_p_up_disagreement(
        decisions=decisions,
        prediction_by_key=prediction_by_key,
    )
    summary.update(p_up_disagreement)
    diagnostic_variants = _diagnostic_exit_variants(
        positions=positions,
        predictions_by_market=predictions_by_market,
        resolution_events=dataset.resolution_events,
    )
    best_variant = _best_variant(diagnostic_variants)
    summary["sell_before_close_exit_failure_interpretation"] = _interpretation(
        summary=summary,
        diagnostic_variants=diagnostic_variants,
        replay=replay,
    )
    summary["sell_before_close_best_diagnostic_exit_variant"] = best_variant["variant"]
    summary["sell_before_close_best_diagnostic_exit_variant_total_pnl"] = best_variant[
        "total_pnl"
    ]
    guard_summary = dict(replay.get("exit_reliability_guard_summary") or {})
    if guard_summary:
        summary["exit_reliability_guard_enabled"] = True
        for field in (
            "exit_policy",
            "entry_filter_thresholds",
            "entry_decision_count_before_guard",
            "entry_decision_count_after_guard",
            "entry_filter_blocked_count",
        ):
            summary[field] = guard_summary.get(field)
    return {
        "candidate_name": candidate_name,
        "counterfactual_replay_variant": replay["variant"],
        "diagnostic_only": True,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "exit_reliability_guard_summary": guard_summary,
        "summary": summary,
        "candidate_scoped_p_up_action_disagreement": p_up_disagreement,
        "replay_report": {
            "max_drawdown": replay.get("replay_report", {}).get("max_drawdown"),
            "total_polymarket_pnl": replay.get("replay_report", {}).get(
                "total_polymarket_pnl"
            ),
            "settlement_pnl": replay.get("replay_report", {}).get("settlement_pnl"),
            "realized_trade_pnl": replay.get("replay_report", {}).get(
                "realized_trade_pnl"
            ),
        },
        "position_lifecycle_rows": positions,
        "grouped_summaries": {
            field: _group_summaries(rows=positions, group_fields=(field,))
            for field in GROUP_FIELDS
        },
        "multi_dimensional_group_summaries": _group_summaries(
            rows=positions,
            group_fields=GROUP_FIELDS,
        ),
        "diagnostic_exit_variants": diagnostic_variants,
        **compact_safety_fields(),
    }


def _candidate_replay_comparison(
    candidate_reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for report in candidate_reports:
        summary = report["summary"]
        rows.append(
            {
                "candidate_name": report["candidate_name"],
                "entry_count": summary["positions_opened_count"],
                "sell_count": summary["sell_decision_count"],
                "residual_count": summary[
                    "positions_opened_but_not_closed_before_settlement"
                ],
                "realized_trade_pnl": summary["realized_trade_pnl"],
                "settlement_pnl": summary["settlement_pnl"],
                "total_pnl": summary["total_polymarket_pnl"],
                "max_drawdown": report.get("replay_report", {}).get("max_drawdown"),
                "replay_residual_settlement_drag": summary[
                    "replay_residual_settlement_drag"
                ],
                "p_up_disagreement_rate": summary[
                    "candidate_scoped_p_up_action_disagreement_rate"
                ],
                "source_model_candidate_eligible": False,
                "ineligible_reason_codes": [],
            }
        )
    if rows:
        baseline = rows[0]
        for row in rows[1:]:
            row["total_pnl_improved_vs_i_candidate"] = (
                float(row["total_pnl"]) > float(baseline["total_pnl"])
            )
            row["residual_count_reduced_vs_i_candidate"] = (
                int(row["residual_count"]) < int(baseline["residual_count"])
            )
            row["residual_drag_reduced_vs_i_candidate"] = (
                float(row["replay_residual_settlement_drag"])
                > float(baseline["replay_residual_settlement_drag"])
            )
    return rows


def _guard_candidate_summary(
    candidate_reports: list[dict[str, Any]],
    *,
    candidate_name: str,
) -> dict[str, Any] | None:
    for report in candidate_reports:
        if report["candidate_name"] == candidate_name:
            summary = report["summary"]
            guard = report["exit_reliability_guard_summary"]
            return {
                "candidate_name": report["candidate_name"],
                "exit_reliability_guard_enabled": True,
                "p_up_side_alignment_filter_enabled": bool(
                    guard.get("p_up_side_alignment_filter_enabled", False)
                ),
                "exit_policy": guard.get("exit_policy"),
                "entry_filter_thresholds": guard.get("entry_filter_thresholds", {}),
                "entry_decision_count_before_guard": guard.get(
                    "entry_decision_count_before_guard",
                ),
                "entry_decision_count_after_exit_guard": guard.get(
                    "entry_decision_count_after_exit_guard",
                ),
                "entry_decision_count_after_p_up_alignment": guard.get(
                    "entry_decision_count_after_p_up_alignment",
                ),
                "entry_decision_count_after_guard": guard.get(
                    "entry_decision_count_after_guard",
                ),
                "entry_filter_blocked_count": guard.get("entry_filter_blocked_count"),
                "entry_filter_blocked_by_p_up_alignment_count": guard.get(
                    "entry_filter_blocked_by_p_up_alignment_count",
                    0,
                ),
                "entry_filter_blocked_by_quality_count": guard.get(
                    "entry_filter_blocked_by_quality_count",
                    0,
                ),
                "reentry_cooldown_seconds": guard.get("reentry_cooldown_seconds"),
                "reentry_blocked_count": guard.get("reentry_blocked_count", 0),
                "entries_per_market_distribution": guard.get(
                    "entries_per_market_distribution",
                    {},
                ),
                "positions_opened_count": summary["positions_opened_count"],
                "positions_closed_before_settlement_count": summary[
                    "positions_closed_before_settlement_count"
                ],
                "positions_opened_but_not_closed_before_settlement": summary[
                    "positions_opened_but_not_closed_before_settlement"
                ],
                "replay_realized_trade_pnl": summary["realized_trade_pnl"],
                "replay_settlement_pnl": summary["settlement_pnl"],
                "replay_total_polymarket_pnl": summary["total_polymarket_pnl"],
                "replay_residual_settlement_drag": summary[
                    "replay_residual_settlement_drag"
                ],
                "candidate_scoped_p_up_action_disagreement_count": summary[
                    "candidate_scoped_p_up_action_disagreement_count"
                ],
                "candidate_scoped_p_up_action_disagreement_denominator": summary[
                    "candidate_scoped_p_up_action_disagreement_denominator"
                ],
                "candidate_scoped_p_up_action_disagreement_rate": summary[
                    "candidate_scoped_p_up_action_disagreement_rate"
                ],
                "candidate_scoped_p_up_action_disagreement_within_limit": summary[
                    "candidate_scoped_p_up_action_disagreement_within_limit"
                ],
                "promotion_evidence_eligible": False,
                "paper_run_resume_allowed": False,
            }
    return None


def _decision_stream_p_up_disagreement(
    *,
    decisions: list[dict[str, Any]],
    prediction_by_key: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    entry_decisions = [
        decision
        for decision in decisions
        if decision["action"] in {"BUY_UP", "BUY_DOWN"}
        and str(decision.get("entry_policy_action", "")).endswith(
            "_SELL_BEFORE_CLOSE"
        )
    ]
    disagreement_count = 0
    for decision in entry_decisions:
        prediction = prediction_by_key[
            (decision["market_id"], int(decision["decision_ts"]))
        ]
        p_up = _p_up(prediction)
        if _decision_p_up_action_disagrees(action=decision["action"], p_up=p_up):
            disagreement_count += 1
    denominator = len(entry_decisions)
    rate = 0.0 if denominator == 0 else disagreement_count / denominator
    return {
        "candidate_scoped_p_up_action_disagreement_count": disagreement_count,
        "candidate_scoped_p_up_action_disagreement_denominator": denominator,
        "candidate_scoped_p_up_action_disagreement_rate": rate,
        "candidate_scoped_p_up_action_disagreement_within_limit": (
            rate <= P_UP_ACTION_DISAGREEMENT_FAIL_THRESHOLD
        ),
    }


def _p_up(prediction: dict[str, Any]) -> float:
    value = prediction.get("p_up_auxiliary")
    if value is None:
        value = prediction.get("estimated_up_probability", 0.5)
    return float(value)


def _decision_p_up_action_disagrees(*, action: str, p_up: float) -> bool:
    if action == "BUY_DOWN":
        return p_up >= P_UP_MATERIAL_DISAGREEMENT_THRESHOLD
    if action == "BUY_UP":
        return p_up <= 1.0 - P_UP_MATERIAL_DISAGREEMENT_THRESHOLD
    return False


def sell_before_close_exit_reliability_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Return compact exit reliability summary for embedding into other artifacts."""

    summary = report["summary"]
    return {
        "schema_version": report["schema_version"],
        "candidate_name": report["candidate_name"],
        "counterfactual_replay_variant": report["counterfactual_replay_variant"],
        "diagnostic_only": True,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "diagnostic_counterfactual": True,
        "sell_before_close_exit_failure_interpretation": summary[
            "sell_before_close_exit_failure_interpretation"
        ],
        "entry_decision_count": summary["entry_decision_count"],
        "sell_decision_count": summary["sell_decision_count"],
        "positions_opened_count": summary["positions_opened_count"],
        "positions_closed_before_settlement_count": summary[
            "positions_closed_before_settlement_count"
        ],
        "positions_opened_but_not_closed_before_settlement": summary[
            "positions_opened_but_not_closed_before_settlement"
        ],
        "realized_trade_pnl": summary["realized_trade_pnl"],
        "settlement_pnl": summary["settlement_pnl"],
        "total_polymarket_pnl": summary["total_polymarket_pnl"],
        "replay_residual_settlement_drag": summary[
            "replay_residual_settlement_drag"
        ],
        "sell_before_close_best_diagnostic_exit_variant": summary[
            "sell_before_close_best_diagnostic_exit_variant"
        ],
        "sell_before_close_best_diagnostic_exit_variant_total_pnl": summary[
            "sell_before_close_best_diagnostic_exit_variant_total_pnl"
        ],
        "candidate_scoped_p_up_action_disagreement_rate": summary[
            "candidate_scoped_p_up_action_disagreement_rate"
        ],
        "candidate_scoped_p_up_action_disagreement_within_limit": summary[
            "candidate_scoped_p_up_action_disagreement_within_limit"
        ],
        "exit_reliability_guard_candidate_summary": report.get(
            "exit_reliability_guard_candidate_summary"
        ),
        "exit_reliability_p_up_aligned_candidate_summary": report.get(
            "exit_reliability_p_up_aligned_candidate_summary"
        ),
        "i_vs_j_replay_comparison": report.get("i_vs_j_replay_comparison", []),
        "i_vs_j_vs_k_replay_comparison": report.get(
            "i_vs_j_vs_k_replay_comparison",
            [],
        ),
    }


def sell_before_close_exit_reliability_markdown(report: dict[str, Any]) -> str:
    """Render exit reliability report markdown."""

    summary = report["summary"]
    lines = [
        "# SELL_BEFORE_CLOSE Exit Reliability Report",
        "",
        f"- schema_version: `{report['schema_version']}`",
        f"- candidate_name: `{report['candidate_name']}`",
        f"- diagnostic_only: `{str(report['diagnostic_only']).lower()}`",
        f"- promotion_evidence_eligible: `{str(report['promotion_evidence_eligible']).lower()}`",
        f"- paper_run_resume_allowed: `{str(report['paper_run_resume_allowed']).lower()}`",
        f"- sell_before_close_exit_failure_interpretation: `{summary['sell_before_close_exit_failure_interpretation']}`",
        "",
        "## Replay Position Lifecycle",
        "",
        f"- entry_decision_count: `{summary['entry_decision_count']}`",
        f"- sell_decision_count: `{summary['sell_decision_count']}`",
        f"- positions_opened_count: `{summary['positions_opened_count']}`",
        f"- positions_closed_before_settlement_count: `{summary['positions_closed_before_settlement_count']}`",
        f"- positions_opened_but_not_closed_before_settlement: `{summary['positions_opened_but_not_closed_before_settlement']}`",
        f"- replay_residual_settlement_drag: `{summary['replay_residual_settlement_drag']}`",
        "",
        "## Diagnostic Exit Variants",
        "",
        "| variant | exit_count | residual_count | trade_pnl | settlement_pnl | total_pnl | max_drawdown |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in report["diagnostic_exit_variants"]:
        lines.append(
            "| {variant} | {exit_count} | {residual_count} | {trade:.6f} | "
            "{settlement:.6f} | {total:.6f} | {drawdown:.6f} |".format(
                variant=variant["variant"],
                exit_count=variant["exit_count"],
                residual_count=variant["residual_count"],
                trade=variant["trade_pnl"],
                settlement=variant["settlement_pnl"],
                total=variant["total_pnl"],
                drawdown=variant["max_drawdown"],
            )
        )
    lines.extend(
        [
            "",
            "## Lifecycle Counts",
            "",
            "| exit_lifecycle_class | position_count | total_pnl | settlement_pnl |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in report["grouped_summaries"]["exit_lifecycle_class"]:
        lines.append(
            "| {klass} | {count} | {total:.6f} | {settlement:.6f} |".format(
                klass=row["exit_lifecycle_class"],
                count=row["position_count"],
                total=row["sum_total_pnl"],
                settlement=row["sum_settlement_pnl"],
            )
        )
    lines.extend(
        [
            "",
            "## I vs J vs K Replay Comparison",
            "",
            "| candidate | entries | sells | residual | trade_pnl | settlement_pnl | total_pnl | residual_drag | p_up_disagreement |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report.get("i_vs_j_vs_k_replay_comparison", []):
        lines.append(
            "| {candidate} | {entries} | {sells} | {residual} | {trade:.6f} | "
            "{settlement:.6f} | {total:.6f} | {drag:.6f} | {p_up:.6f} |".format(
                candidate=row["candidate_name"],
                entries=row["entry_count"],
                sells=row["sell_count"],
                residual=row["residual_count"],
                trade=row["realized_trade_pnl"],
                settlement=row["settlement_pnl"],
                total=row["total_pnl"],
                drag=row["replay_residual_settlement_drag"],
                p_up=row["p_up_disagreement_rate"],
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


def _empty_guard_position() -> dict[str, Any]:
    return {
        "UP_qty": 0.0,
        "DOWN_qty": 0.0,
        "UP_notional": 0.0,
        "DOWN_notional": 0.0,
        "UP_entry_policy_action": None,
        "DOWN_entry_policy_action": None,
        "UP_planned_exit_before_ts": None,
        "DOWN_planned_exit_before_ts": None,
    }


def _open_guard_side(position: dict[str, Any]) -> str | None:
    if float(position["UP_qty"]) > 0.0:
        return "UP"
    if float(position["DOWN_qty"]) > 0.0:
        return "DOWN"
    return None


def _open_guard_position(
    *,
    position: dict[str, Any],
    decision: PolymarketEVDecision,
    side: str,
) -> None:
    prefix = side
    position[f"{prefix}_qty"] = decision.paper_notional / decision.execution_price
    position[f"{prefix}_notional"] = decision.paper_notional
    position[f"{prefix}_entry_policy_action"] = decision.entry_policy_action
    position[f"{prefix}_planned_exit_before_ts"] = decision.planned_exit_before_ts


def _guard_exit_decision(
    *,
    prediction: PolymarketPolicyPrediction,
    config: PolymarketPolicyTrainingConfig,
    position: dict[str, Any],
    side: str,
    exit_policy: str,
) -> tuple[PolymarketEVDecision, bool]:
    bid = _side_feature(prediction.features, side, "bid") or 0.0
    liquidity = _side_feature(
        prediction.features,
        side,
        "executable_bid_notional",
    )
    qty = float(position[f"{side}_qty"])
    entry_notional = float(position[f"{side}_notional"])
    planned_exit_before_ts = position[f"{side}_planned_exit_before_ts"]
    entry_policy_action = position[f"{side}_entry_policy_action"]
    executable = (
        bid > 0.0
        and liquidity is not None
        and float(liquidity) + 1e-12 >= entry_notional
    )
    due = (
        planned_exit_before_ts is not None
        and int(prediction.decision_ts) >= int(planned_exit_before_ts)
    )
    if executable:
        action = "SELL_UP" if side == "UP" else "SELL_DOWN"
        decision = _guard_decision(
            prediction=prediction,
            config=config,
            action=action,
            selected_outcome=side,
            execution_price=bid,
            used_price_side="bid",
            paper_notional=qty * bid,
            reason_codes=(
                "exit_executed_policy_constrained",
                exit_policy,
                "bid_price_execution",
            ),
            entry_policy_action=entry_policy_action,
            intended_exit_policy="sell_before_close",
            planned_exit_before_ts=planned_exit_before_ts,
            policy_exit_reason=exit_policy,
            action_value_head_used=prediction.action_value_head_enabled,
            probability_ev_fallback_used=False,
        )
        return decision, True
    reason = _exit_blocked_reason(
        bid=bid,
        liquidity=liquidity,
        entry_notional=entry_notional,
        due=due,
    )
    decision = _guard_decision(
        prediction=prediction,
        config=config,
        action="HOLD",
        selected_outcome="NO_TRADE",
        execution_price=0.0,
        used_price_side="none",
        paper_notional=0.0,
        reason_codes=(
            reason,
            "forced_settlement_after_exit_failure" if due else "exit_waiting_for_policy_constrained_liquidity",
            exit_policy,
        ),
        entry_policy_action=entry_policy_action,
        intended_exit_policy="sell_before_close",
        planned_exit_before_ts=planned_exit_before_ts,
        policy_exit_reason=reason,
        action_value_head_used=prediction.action_value_head_enabled,
        probability_ev_fallback_used=False,
    )
    return decision, False


def _guard_entry_decision(
    *,
    prediction: PolymarketPolicyPrediction,
    config: PolymarketPolicyTrainingConfig,
    action: str,
    guard_reason_codes: tuple[str, ...],
) -> PolymarketEVDecision:
    side = "UP" if action.startswith("BUY_UP") else "DOWN"
    ask = _side_feature(prediction.features, side, "ask") or 0.0
    best_return = float(prediction.calibrated_expected_pnl_per_notional or 0.0)
    decision_action = "BUY_UP" if side == "UP" else "BUY_DOWN"
    return _guard_decision(
        prediction=prediction,
        config=config,
        action=decision_action,
        selected_outcome=side,
        execution_price=ask,
        used_price_side="ask",
        paper_notional=_paper_notional(best_return, config),
        reason_codes=(
            "exit_reliability_guard_passed",
            "positive_action_value_buy_" + side.lower(),
            "calibrated_action_value_used",
            "policy_" + action.lower(),
            "ask_price_execution",
            *guard_reason_codes,
        ),
        entry_policy_action=action,
        intended_exit_policy="sell_before_close",
        planned_exit_before_ts=_planned_exit_before_ts(
            prediction=prediction,
            config=config,
        ),
        policy_exit_reason="sell_before_close",
        action_value_head_used=True,
        probability_ev_fallback_used=False,
    )


def _guard_no_trade_decision(
    *,
    prediction: PolymarketPolicyPrediction,
    config: PolymarketPolicyTrainingConfig,
    reason_codes: tuple[str, ...],
    entry_policy_action: str | None = None,
) -> PolymarketEVDecision:
    return _guard_decision(
        prediction=prediction,
        config=config,
        action="NO_TRADE",
        selected_outcome="NO_TRADE",
        execution_price=0.0,
        used_price_side="none",
        paper_notional=0.0,
        reason_codes=reason_codes,
        entry_policy_action=entry_policy_action,
        intended_exit_policy="none",
        planned_exit_before_ts=None,
        policy_exit_reason="exit_reliability_guard",
        action_value_head_used=prediction.action_value_head_enabled,
        probability_ev_fallback_used=False,
    )


def _guard_decision(
    *,
    prediction: PolymarketPolicyPrediction,
    config: PolymarketPolicyTrainingConfig,
    action: str,
    selected_outcome: str,
    execution_price: float,
    used_price_side: str,
    paper_notional: float,
    reason_codes: tuple[str, ...],
    entry_policy_action: str | None,
    intended_exit_policy: str,
    planned_exit_before_ts: int | None,
    policy_exit_reason: str,
    action_value_head_used: bool,
    probability_ev_fallback_used: bool,
) -> PolymarketEVDecision:
    features = prediction.features
    up_ask = float(features["up_ask"])
    down_ask = float(features["down_ask"])
    cost = _execution_cost(features, config)
    ev_buy_up = float(prediction.estimated_up_probability) - up_ask - cost
    ev_buy_down = (1.0 - float(prediction.estimated_up_probability)) - down_ask - cost
    return PolymarketEVDecision(
        market_id=prediction.market_id,
        condition_id=prediction.condition_id,
        slug=prediction.slug,
        market_family=prediction.market_family,
        horizon_ms=prediction.horizon_ms,
        decision_ts=prediction.decision_ts,
        action=action,
        selected_outcome=selected_outcome,
        estimated_up_probability=prediction.estimated_up_probability,
        confidence=prediction.confidence,
        ev_buy_up=ev_buy_up,
        ev_buy_down=ev_buy_down,
        execution_price=execution_price,
        used_price_side=used_price_side,
        paper_notional=paper_notional,
        reason_codes=tuple(
            dict.fromkeys((*reason_codes, "trained_model_used", "paper_only_guard"))
        ),
        p_up_auxiliary=prediction.p_up_auxiliary or prediction.estimated_up_probability,
        expected_return_by_action=dict(prediction.expected_return_by_action),
        best_policy_action=prediction.best_policy_action,
        best_action_expected_return=prediction.best_action_expected_return,
        second_best_action_expected_return=prediction.second_best_action_expected_return,
        best_action_margin=prediction.best_action_margin,
        calibrated_expected_pnl_per_notional_by_action=dict(
            prediction.calibrated_expected_pnl_per_notional_by_action
        ),
        calibrated_best_policy_action=prediction.calibrated_best_policy_action,
        calibrated_expected_pnl_per_notional=(
            prediction.calibrated_expected_pnl_per_notional
        ),
        calibrated_second_best_expected_pnl_per_notional=(
            prediction.calibrated_second_best_expected_pnl_per_notional
        ),
        calibrated_action_margin=prediction.calibrated_action_margin,
        action_value_calibration_used=(
            prediction.action_value_calibration_applied and action_value_head_used
        ),
        action_value_calibration_id=prediction.action_value_calibration_id,
        calibration_support_count=prediction.calibration_support_count,
        calibration_bucket_count=prediction.calibration_bucket_count,
        policy_confidence=prediction.policy_confidence,
        entry_policy_action=entry_policy_action,
        intended_exit_policy=intended_exit_policy,
        planned_exit_before_ts=planned_exit_before_ts,
        policy_exit_reason=policy_exit_reason,
        action_value_head_used=action_value_head_used,
        action_value_model_family=prediction.action_value_model_family,
        feature_conditioned_action_value_model_used=(
            prediction.feature_conditioned_action_value_model_enabled
            and action_value_head_used
        ),
        probability_ev_fallback_used=probability_ev_fallback_used,
    )


def _entry_guard_assessment(
    *,
    prediction: PolymarketPolicyPrediction,
    action: str,
    config: PolymarketPolicyTrainingConfig,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    side = "UP" if action.startswith("BUY_UP") else "DOWN"
    features = prediction.features
    time_to_close = float(features.get("time_to_close_seconds", 0.0))
    executable_bid_notional = _side_feature(features, side, "executable_bid_notional")
    queue_fill = _side_feature(features, side, "queue_fill_probability_proxy")
    spread = _side_feature(features, side, "spread_bps")
    staleness = _side_feature(features, side, "book_staleness_ms")
    if staleness is None:
        staleness = _side_feature(features, side, "book_update_lag_ms")
    recent_updates = _side_feature(features, side, "recent_book_update_count_1m")
    best_margin = float(prediction.calibrated_action_margin or 0.0)
    best_score = float(prediction.calibrated_expected_pnl_per_notional or 0.0)
    reasons = []
    if time_to_close < float(thresholds["min_seconds_to_close"]):
        reasons.append("entry_blocked_too_close_to_close")
    if (
        executable_bid_notional is None
        or executable_bid_notional < float(thresholds["min_executable_bid_notional"])
    ):
        reasons.append("entry_blocked_insufficient_executable_bid_notional")
    if (
        queue_fill is None
        or queue_fill < float(thresholds["min_queue_fill_probability_proxy"])
    ):
        reasons.append("entry_blocked_low_queue_fill_probability")
    if spread is None or spread > float(thresholds["max_spread"]):
        reasons.append("entry_blocked_spread_too_wide")
    if staleness is None or staleness > float(thresholds["max_book_staleness_ms"]):
        reasons.append("entry_blocked_stale_book")
    if (
        recent_updates is None
        or recent_updates < float(thresholds["min_recent_book_update_count_1m"])
    ):
        reasons.append("entry_blocked_stale_book")
    if best_margin < float(thresholds["min_best_action_margin"]):
        reasons.append("entry_blocked_exit_reliability_guard")
    min_score = max(
        float(thresholds["min_calibrated_action_score"]),
        float(config.ev_threshold),
    )
    if best_score < min_score:
        reasons.append("entry_blocked_exit_reliability_guard")
    passed = not reasons
    return {
        "passed": passed,
        "side": side,
        "reason_codes": (
            ("exit_reliability_guard_thresholds_passed",)
            if passed
            else tuple(
                dict.fromkeys(("entry_blocked_exit_reliability_guard", *reasons))
            )
        ),
    }


def _p_up_alignment_assessment(
    *,
    prediction: PolymarketPolicyPrediction,
    action: str,
    thresholds: dict[str, float],
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {
            "passed": True,
            "reason_codes": ("p_up_side_alignment_filter_disabled",),
        }
    p_up = _prediction_p_up(prediction)
    alignment_min = float(thresholds.get("p_up_alignment_min", 0.55))
    if action.startswith("BUY_UP"):
        passed = p_up >= alignment_min
    elif action.startswith("BUY_DOWN"):
        passed = p_up <= 1.0 - alignment_min
    else:
        passed = True
    return {
        "passed": passed,
        "reason_codes": (
            ("p_up_side_alignment_passed",)
            if passed
            else (
                "entry_blocked_p_up_action_disagreement",
                "entry_blocked_p_up_side_alignment_failed",
            )
        ),
    }


def _turnover_guard_assessment(
    *,
    market_id: str,
    decision_ts: int,
    thresholds: dict[str, float],
    entries_per_market: dict[str, int],
    last_exit_ts_by_market: dict[str, int],
) -> dict[str, Any]:
    reasons = []
    max_entries = int(float(thresholds.get("max_entries_per_market", 10**9)))
    cooldown_seconds = float(thresholds.get("min_reentry_cooldown_seconds", 0.0))
    if entries_per_market.get(market_id, 0) >= max_entries:
        reasons.append("entry_blocked_max_entries_per_market")
    last_exit_ts = last_exit_ts_by_market.get(market_id)
    if last_exit_ts is not None and cooldown_seconds > 0.0:
        elapsed_seconds = max(0.0, (int(decision_ts) - int(last_exit_ts)) / 1000.0)
        if elapsed_seconds < cooldown_seconds:
            reasons.append("entry_blocked_reentry_cooldown")
    passed = not reasons
    return {
        "passed": passed,
        "reason_codes": (
            ("turnover_guard_passed",)
            if passed
            else tuple(dict.fromkeys(("entry_blocked_turnover_guard", *reasons)))
        ),
    }


def _prediction_p_up(prediction: PolymarketPolicyPrediction) -> float:
    value = prediction.p_up_auxiliary
    if value is None:
        value = prediction.estimated_up_probability
    return float(value)


def _action_p_up_disagrees(*, action: str, p_up: float) -> bool:
    if action.startswith("BUY_DOWN"):
        return p_up >= P_UP_MATERIAL_DISAGREEMENT_THRESHOLD
    if action.startswith("BUY_UP"):
        return p_up <= 1.0 - P_UP_MATERIAL_DISAGREEMENT_THRESHOLD
    return False


def _exit_blocked_reason(
    *,
    bid: float,
    liquidity: float | None,
    entry_notional: float,
    due: bool,
) -> str:
    if due and bid <= 0.0:
        return "exit_blocked_no_executable_bid"
    if liquidity is None or float(liquidity) + 1e-12 < entry_notional:
        return "exit_blocked_liquidity_insufficient"
    if due:
        return "exit_blocked_timing_too_late"
    return "exit_blocked_no_executable_bid"


def _planned_exit_before_ts(
    *,
    prediction: PolymarketPolicyPrediction,
    config: PolymarketPolicyTrainingConfig,
) -> int:
    time_to_close_ms = max(
        0,
        int(float(prediction.features.get("time_to_close_seconds", 0.0)) * 1000),
    )
    market_end_ts = int(prediction.decision_ts) + time_to_close_ms
    exit_buffer_ms = int(config.sell_before_close_exit_buffer_seconds * 1000)
    return max(int(prediction.decision_ts), market_end_ts - exit_buffer_ms)


def _paper_notional(ev: float, config: PolymarketPolicyTrainingConfig) -> float:
    return min(config.max_paper_notional, max(0.01, ev * config.max_paper_notional * 5.0))


def _execution_cost(
    features: dict[str, float],
    config: PolymarketPolicyTrainingConfig,
) -> float:
    liquidity = max(
        1.0,
        float(features.get("up_liquidity_depth", 0.0))
        + float(features.get("down_liquidity_depth", 0.0)),
    )
    return (
        config.fee_rate
        + config.slippage_rate
        + config.liquidity_impact_rate * (config.max_paper_notional / liquidity)
    )


def _position_lifecycle_rows(
    *,
    decisions: list[dict[str, Any]],
    prediction_by_key: dict[tuple[str, int], dict[str, Any]],
    predictions_by_market: dict[str, list[dict[str, Any]]],
    market_metadata: dict[str, dict[str, Any]],
    resolution_events: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    open_positions: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    rows = []
    position_index = 0
    for decision in decisions:
        action = str(decision["action"])
        side = _side_from_decision(decision)
        if action in {"BUY_UP", "BUY_DOWN"} and decision.get(
            "intended_exit_policy"
        ) == "sell_before_close":
            prediction = prediction_by_key[(decision["market_id"], int(decision["decision_ts"]))]
            position = _open_position_row(
                decision=decision,
                prediction=prediction,
                market_metadata=market_metadata[decision["market_id"]],
                position_index=position_index,
            )
            position_index += 1
            open_positions[(position["market_id"], side)].append(position)
            continue
        if action in {"SELL_UP", "SELL_DOWN"}:
            key = (decision["market_id"], side)
            closing = list(open_positions.get(key, []))
            if not closing:
                continue
            open_positions[key] = []
            for position in closing:
                rows.append(
                    _closed_position_row(
                        position=position,
                        decision=decision,
                    )
                )
    for remaining in open_positions.values():
        for position in remaining:
            rows.append(
                _residual_position_row(
                    position=position,
                    predictions_by_market=predictions_by_market,
                    resolution=resolution_events[position["market_id"]],
                )
            )
    return sorted(rows, key=lambda row: (row["entry_decision_ts"], row["position_id"]))


def _open_position_row(
    *,
    decision: dict[str, Any],
    prediction: dict[str, Any],
    market_metadata: dict[str, Any],
    position_index: int,
) -> dict[str, Any]:
    features = prediction["features"]
    entry_side = _side_from_decision(decision)
    entry_price = float(decision["execution_price"])
    entry_notional = float(decision["paper_notional"])
    qty = 0.0 if entry_price <= 0.0 else entry_notional / entry_price
    raw_score = float(
        (decision.get("expected_return_by_action") or {}).get(
            decision.get("entry_policy_action"),
            0.0,
        )
    )
    bucket = action_value_bucket_payload(
        action=str(decision["entry_policy_action"]),
        features=features,
        raw_score=raw_score,
    )
    return {
        "market_id": decision["market_id"],
        "slug": decision["slug"],
        "position_id": canonical_json_sha256(
            {
                "market_id": decision["market_id"],
                "entry_decision_ts": decision["decision_ts"],
                "entry_action": decision["action"],
                "position_index": position_index,
            }
        )[:24],
        "entry_decision_ts": int(decision["decision_ts"]),
        "entry_action": decision["action"],
        "entry_side": entry_side,
        "entry_price": entry_price,
        "entry_notional": entry_notional,
        "entry_qty": qty,
        "intended_exit_policy": decision.get("intended_exit_policy"),
        "planned_exit_before_ts": decision.get("planned_exit_before_ts"),
        "market_end_ts": int(market_metadata["market_end_ts"]),
        "seconds_from_entry_to_close": (
            int(market_metadata["market_end_ts"]) - int(decision["decision_ts"])
        )
        / 1000.0,
        "entry_features": features,
        "time_to_close_bucket": bucket["time_to_close_bucket"],
        "price_bucket": bucket["price_bucket"],
        "queue_fill_probability_bucket": _queue_fill_probability_bucket(
            _side_feature(features, entry_side, "queue_fill_probability_proxy")
        ),
        "executable_liquidity_bucket": _executable_liquidity_bucket(
            _side_feature(features, entry_side, "executable_bid_notional")
        ),
    }


def _closed_position_row(
    *,
    position: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    qty = float(position["entry_qty"])
    exit_price = float(decision["execution_price"])
    trade_pnl = (exit_price - float(position["entry_price"])) * qty
    return {
        **_row_base(position),
        "exit_lifecycle_class": "closed_before_settlement",
        "exit_decision_ts": int(decision["decision_ts"]),
        "exit_action": decision["action"],
        "exit_price": exit_price,
        "exit_notional": qty * exit_price,
        "exit_reason_codes": list(decision.get("reason_codes", [])),
        "best_available_exit_bid_before_close": exit_price,
        "best_available_exit_ts_before_close": int(decision["decision_ts"]),
        "best_available_exit_queue_fill_probability": None,
        "best_available_exit_executable_liquidity_notional": None,
        "missed_exit_opportunity": False,
        "missed_exit_reason": "none",
        "trade_pnl": trade_pnl,
        "settlement_pnl": 0.0,
        "total_pnl": trade_pnl,
        "would_have_exited_under_policy_constrained_exit": True,
        "policy_constrained_exit_ts": int(decision["decision_ts"]),
        "policy_constrained_exit_price": exit_price,
        "policy_constrained_exit_pnl": trade_pnl,
        "reason_codes": ["closed_before_settlement"],
    }


def _residual_position_row(
    *,
    position: dict[str, Any],
    predictions_by_market: dict[str, list[dict[str, Any]]],
    resolution: dict[str, Any],
) -> dict[str, Any]:
    future = _future_predictions(position, predictions_by_market)
    best_exit = _best_exit_candidate(position=position, candidates=future)
    policy_exit = _policy_constrained_exit(position=position, candidates=future)
    payout = float(
        resolution["payout_up"] if position["entry_side"] == "UP" else resolution["payout_down"]
    )
    settlement_pnl = (payout - float(position["entry_price"])) * float(
        position["entry_qty"]
    )
    missed_reason = _missed_exit_reason(position=position, candidates=future)
    lifecycle_class = _residual_lifecycle_class(missed_reason)
    return {
        **_row_base(position),
        "exit_lifecycle_class": lifecycle_class,
        "exit_decision_ts": None,
        "exit_action": None,
        "exit_price": None,
        "exit_notional": 0.0,
        "exit_reason_codes": [],
        "best_available_exit_bid_before_close": best_exit["bid"],
        "best_available_exit_ts_before_close": best_exit["ts"],
        "best_available_exit_queue_fill_probability": best_exit["queue_fill"],
        "best_available_exit_executable_liquidity_notional": best_exit["liquidity"],
        "missed_exit_opportunity": best_exit["bid"] is not None,
        "missed_exit_reason": missed_reason,
        "trade_pnl": 0.0,
        "settlement_pnl": settlement_pnl,
        "total_pnl": settlement_pnl,
        "would_have_exited_under_policy_constrained_exit": policy_exit["ts"] is not None,
        "policy_constrained_exit_ts": policy_exit["ts"],
        "policy_constrained_exit_price": policy_exit["bid"],
        "policy_constrained_exit_pnl": policy_exit["pnl"],
        "reason_codes": ["held_to_settlement_residual", "forced_settlement", missed_reason],
    }


def _row_base(position: dict[str, Any]) -> dict[str, Any]:
    return {
        key: position[key]
        for key in (
            "market_id",
            "slug",
            "position_id",
            "entry_decision_ts",
            "entry_action",
            "entry_side",
            "entry_price",
            "entry_notional",
            "entry_qty",
            "intended_exit_policy",
            "planned_exit_before_ts",
            "market_end_ts",
            "seconds_from_entry_to_close",
            "time_to_close_bucket",
            "price_bucket",
            "queue_fill_probability_bucket",
            "executable_liquidity_bucket",
        )
    }


def _future_predictions(
    position: dict[str, Any],
    predictions_by_market: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    return [
        prediction
        for prediction in predictions_by_market.get(position["market_id"], [])
        if int(prediction["decision_ts"]) > int(position["entry_decision_ts"])
        and int(prediction["decision_ts"]) <= int(position["market_end_ts"])
    ]


def _best_exit_candidate(
    *,
    position: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    executable = [
        _exit_candidate(position=position, prediction=prediction)
        for prediction in candidates
    ]
    executable = [candidate for candidate in executable if candidate["executable"]]
    if not executable:
        return {"bid": None, "ts": None, "queue_fill": None, "liquidity": None}
    return sorted(executable, key=lambda row: (-float(row["bid"]), int(row["ts"])))[0]


def _policy_constrained_exit(
    *,
    position: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    due_ts = position.get("planned_exit_before_ts")
    due_candidates = [
        prediction
        for prediction in candidates
        if due_ts is not None and int(prediction["decision_ts"]) >= int(due_ts)
    ]
    for prediction in due_candidates:
        candidate = _exit_candidate(position=position, prediction=prediction)
        if candidate["executable"]:
            pnl = (float(candidate["bid"]) - float(position["entry_price"])) * float(
                position["entry_qty"]
            )
            return {"ts": candidate["ts"], "bid": candidate["bid"], "pnl": pnl}
    return {"ts": None, "bid": None, "pnl": 0.0}


def _exit_candidate(
    *,
    position: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    side = position["entry_side"]
    features = prediction["features"]
    bid = _side_feature(features, side, "bid")
    liquidity = _side_feature(features, side, "executable_bid_notional")
    queue_fill = _side_feature(features, side, "queue_fill_probability_proxy")
    executable = (
        bid is not None
        and float(bid) > 0.0
        and liquidity is not None
        and float(liquidity) + 1e-12 >= float(position["entry_notional"])
    )
    return {
        "ts": int(prediction["decision_ts"]),
        "bid": None if bid is None else float(bid),
        "liquidity": None if liquidity is None else float(liquidity),
        "queue_fill": None if queue_fill is None else float(queue_fill),
        "executable": executable,
    }


def _missed_exit_reason(
    *,
    position: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> str:
    if not candidates:
        return "missing_replay_event_trace"
    due_ts = position.get("planned_exit_before_ts")
    due_candidates = [
        prediction
        for prediction in candidates
        if due_ts is not None and int(prediction["decision_ts"]) >= int(due_ts)
    ]
    if not due_candidates:
        return "no_exit_signal_generated"
    if any(_exit_candidate(position=position, prediction=row)["executable"] for row in due_candidates):
        return "exit_signal_generated_but_blocked_by_gate"
    if any((_exit_candidate(position=position, prediction=row)["bid"] or 0.0) > 0.0 for row in due_candidates):
        return "exit_signal_generated_but_blocked_by_liquidity"
    return "exit_signal_generated_but_not_executable"


def _residual_lifecycle_class(missed_reason: str) -> str:
    if missed_reason in {
        "missing_replay_event_trace",
        "no_exit_signal_generated",
        "exit_signal_generated_but_blocked_by_gate",
        "exit_signal_generated_but_blocked_by_cooldown",
        "exit_signal_generated_but_blocked_by_liquidity",
        "exit_signal_generated_but_too_close_to_market_end",
        "exit_signal_generated_but_not_executable",
    }:
        return missed_reason
    return "held_to_settlement_residual"


def _replay_summary(*, rows: list[dict[str, Any]], replay: dict[str, Any]) -> dict[str, Any]:
    variant_summary = replay["summary"]
    action_counts = variant_summary["action_counts"]
    entry_count = int(action_counts.get("BUY_UP", 0)) + int(action_counts.get("BUY_DOWN", 0))
    sell_count = int(action_counts.get("SELL_UP", 0)) + int(action_counts.get("SELL_DOWN", 0))
    closed_count = sum(row["exit_lifecycle_class"] == "closed_before_settlement" for row in rows)
    residual_count = len(rows) - closed_count
    reason_counts = variant_summary.get("reason_counts", {})
    missed_counts = _missed_exit_reason_counts(reason_counts)
    return {
        "entry_decision_count": int(variant_summary.get("entry_decision_count", entry_count)),
        "sell_decision_count": sell_count,
        "positions_opened_count": len(rows),
        "positions_closed_before_settlement_count": closed_count,
        "positions_opened_but_not_closed_before_settlement": residual_count,
        "positions_reached_sell_before_close_intent_but_settled": residual_count,
        "forced_settlement_event_count": residual_count,
        "exit_signal_missing_count": missed_counts["exit_signal_missing"],
        "exit_opportunities_missed_due_to_gate_count": missed_counts["gate"],
        "exit_opportunities_missed_due_to_cooldown_count": missed_counts["cooldown"],
        "exit_opportunities_missed_due_to_liquidity_count": missed_counts["liquidity"],
        "exit_opportunities_missed_due_to_timing_count": missed_counts["timing"],
        "realized_trade_pnl": float(variant_summary["realized_trade_pnl"]),
        "settlement_pnl": float(variant_summary["settlement_pnl"]),
        "total_polymarket_pnl": float(variant_summary["total_polymarket_pnl"]),
        "replay_residual_settlement_drag": min(0.0, float(variant_summary["settlement_pnl"])),
    }


def _diagnostic_exit_variants(
    *,
    positions: list[dict[str, Any]],
    predictions_by_market: dict[str, list[dict[str, Any]]],
    resolution_events: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        _diagnostic_variant(
            variant=variant,
            positions=positions,
            predictions_by_market=predictions_by_market,
            resolution_events=resolution_events,
        )
        for variant in EXIT_VARIANTS
    ]


def _diagnostic_variant(
    *,
    variant: str,
    positions: list[dict[str, Any]],
    predictions_by_market: dict[str, list[dict[str, Any]]],
    resolution_events: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    events = []
    trade_pnl = 0.0
    settlement_pnl = 0.0
    exit_count = 0
    residual_count = 0
    for position in positions:
        candidates = _future_predictions(position, predictions_by_market)
        exit_candidate = _variant_exit_candidate(
            variant=variant,
            position=position,
            candidates=candidates,
        )
        if exit_candidate["ts"] is None:
            residual_count += 1
            resolution = resolution_events[position["market_id"]]
            payout = float(
                resolution["payout_up"]
                if position["entry_side"] == "UP"
                else resolution["payout_down"]
            )
            pnl = (payout - float(position["entry_price"])) * float(position["entry_qty"])
            settlement_pnl += pnl
            events.append((int(position["market_end_ts"]), pnl))
            continue
        exit_count += 1
        pnl = (float(exit_candidate["bid"]) - float(position["entry_price"])) * float(
            position["entry_qty"]
        )
        trade_pnl += pnl
        events.append((int(exit_candidate["ts"]), pnl))
    total_pnl = trade_pnl + settlement_pnl
    return {
        "variant": variant,
        "diagnostic_counterfactual": True,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "entry_count": len(positions),
        "exit_count": exit_count,
        "residual_count": residual_count,
        "trade_pnl": trade_pnl,
        "settlement_pnl": settlement_pnl,
        "total_pnl": total_pnl,
        "max_drawdown": _max_drawdown(events),
        **compact_safety_fields(),
    }


def _variant_exit_candidate(
    *,
    variant: str,
    position: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    executable = [
        _exit_candidate(position=position, prediction=prediction)
        for prediction in candidates
    ]
    executable = [candidate for candidate in executable if candidate["executable"]]
    if not executable:
        return {"ts": None, "bid": None}
    if variant == "first_executable_exit_after_entry":
        return sorted(executable, key=lambda row: int(row["ts"]))[0]
    if variant == "scheduled_exit_before_close":
        due_ts = position.get("planned_exit_before_ts")
        due = [
            candidate
            for candidate in executable
            if due_ts is not None and int(candidate["ts"]) >= int(due_ts)
        ]
        return (
            sorted(due, key=lambda row: int(row["ts"]))[0]
            if due
            else {"ts": None, "bid": None}
        )
    if variant == "forced_preclose_exit_if_executable":
        return sorted(executable, key=lambda row: int(row["ts"]))[-1]
    if variant == "take_profit_stop_loss_exit":
        for candidate in sorted(executable, key=lambda row: int(row["ts"])):
            edge = float(candidate["bid"]) - float(position["entry_price"])
            if edge >= 0.05 or edge <= -0.05:
                return candidate
        return {"ts": None, "bid": None}
    raise ValueError(f"unsupported diagnostic exit variant: {variant}")


def _group_summaries(
    *,
    rows: list[dict[str, Any]],
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(field)) for field in group_fields)].append(row)
    summaries = []
    for key, group_rows in groups.items():
        payload = {field: key[index] for index, field in enumerate(group_fields)}
        payload.update(_metrics(group_rows))
        summaries.append(payload)
    return sorted(summaries, key=lambda row: (-int(row["position_count"]), tuple(str(row.get(field)) for field in group_fields)))


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    position_count = len(rows)
    return {
        "position_count": position_count,
        "closed_count": sum(row["exit_lifecycle_class"] == "closed_before_settlement" for row in rows),
        "residual_count": sum(row["exit_lifecycle_class"] != "closed_before_settlement" for row in rows),
        "forced_settlement_count": sum(row["settlement_pnl"] != 0.0 for row in rows),
        "mean_trade_pnl": _mean(rows, "trade_pnl"),
        "sum_trade_pnl": _sum(rows, "trade_pnl"),
        "mean_settlement_pnl": _mean(rows, "settlement_pnl"),
        "sum_settlement_pnl": _sum(rows, "settlement_pnl"),
        "mean_total_pnl": _mean(rows, "total_pnl"),
        "sum_total_pnl": _sum(rows, "total_pnl"),
        "positive_total_pnl_rate": 0.0
        if not rows
        else sum(row["total_pnl"] > 0.0 for row in rows) / len(rows),
        "mean_exit_delay_seconds": _mean_exit_delay(rows),
        "mean_best_available_exit_bid_before_close": _mean_optional(
            rows,
            "best_available_exit_bid_before_close",
        ),
        "mean_policy_constrained_exit_pnl": _mean(rows, "policy_constrained_exit_pnl"),
    }


def _interpretation(
    *,
    summary: dict[str, Any],
    diagnostic_variants: list[dict[str, Any]],
    replay: dict[str, Any],
) -> str:
    residual = int(summary["positions_opened_but_not_closed_before_settlement"])
    if residual <= 0:
        return "insufficient_evidence"
    planned_exit_count = int(replay["replay_report"].get("planned_sell_before_close_exit_count", 0))
    forced_variant = next(
        variant
        for variant in diagnostic_variants
        if variant["variant"] == "forced_preclose_exit_if_executable"
    )
    if planned_exit_count == 0 and int(forced_variant["exit_count"]) > int(
        summary["positions_closed_before_settlement_count"]
    ):
        return "policy_does_not_enforce_planned_exit"
    active_reasons = sum(
        1
        for key in (
            "exit_signal_missing_count",
            "exit_opportunities_missed_due_to_gate_count",
            "exit_opportunities_missed_due_to_liquidity_count",
            "exit_opportunities_missed_due_to_timing_count",
        )
        if int(summary[key]) > 0
    )
    if active_reasons >= 2:
        return "mixed_exit_reliability_failure"
    if int(summary["exit_signal_missing_count"]) > 0:
        return "exit_signal_missing"
    if int(summary["exit_opportunities_missed_due_to_gate_count"]) > 0:
        return "exit_gate_too_strict"
    if int(summary["exit_opportunities_missed_due_to_liquidity_count"]) > 0:
        return "exit_liquidity_insufficient"
    if int(summary["exit_opportunities_missed_due_to_timing_count"]) > 0:
        return "exit_timing_too_late"
    return "mixed_exit_reliability_failure"


def _missed_exit_reason_counts(reason_counts: dict[str, Any]) -> dict[str, int]:
    buckets = {"gate": 0, "cooldown": 0, "liquidity": 0, "timing": 0, "exit_signal_missing": 0}
    for reason, value in reason_counts.items():
        count = int(value)
        normalized = str(reason).lower()
        if any(token in normalized for token in ("threshold", "gate", "low_confidence")):
            buckets["gate"] += count
        if "cooldown" in normalized:
            buckets["cooldown"] += count
        if "liquidity" in normalized:
            buckets["liquidity"] += count
        if any(token in normalized for token in ("timing", "time", "close")):
            buckets["timing"] += count
        if any(token in normalized for token in ("hold", "no_exit", "missing")):
            buckets["exit_signal_missing"] += count
    return buckets


def _best_variant(variants: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(variants, key=lambda row: (-float(row["total_pnl"]), row["variant"]))[0]


def _side_from_decision(decision: dict[str, Any]) -> str:
    if str(decision["action"]).endswith("_UP") or decision.get("selected_outcome") == "UP":
        return "UP"
    return "DOWN"


def _side_feature(features: dict[str, Any], side: str, field: str) -> float | None:
    prefix = "up" if side == "UP" else "down"
    value = features.get(f"{prefix}_{field}")
    return None if value is None else float(value)


def _queue_fill_probability_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 0.25:
        return "<0.25"
    if value < 0.50:
        return "0.25-0.50"
    if value < 0.75:
        return "0.50-0.75"
    return ">=0.75"


def _executable_liquidity_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value <= 0.0:
        return "0"
    if value < 0.20:
        return "<0.20"
    if value < 1.00:
        return "0.20-1.00"
    return ">=1.00"


def _max_drawdown(events: list[tuple[int, float]]) -> float:
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for _, pnl in sorted(events):
        cumulative += float(pnl)
        peak = max(peak, cumulative)
        max_drawdown = min(max_drawdown, cumulative - peak)
    return abs(max_drawdown)


def _mean_exit_delay(rows: list[dict[str, Any]]) -> float:
    delays = [
        (int(row["exit_decision_ts"]) - int(row["entry_decision_ts"])) / 1000.0
        for row in rows
        if row["exit_decision_ts"] is not None
    ]
    return 0.0 if not delays else sum(delays) / len(delays)


def _sum(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows)


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return 0.0 if not rows else _sum(rows, field) / len(rows)


def _mean_optional(rows: list[dict[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows if row[field] is not None]
    return 0.0 if not values else sum(values) / len(values)


def _replay_by_variant(
    replays: tuple[dict[str, Any], ...],
    variant: str,
) -> dict[str, Any]:
    for replay in replays:
        if replay["variant"] == variant:
            return replay
    raise ValueError(f"missing counterfactual replay variant: {variant}")


def _optional_replay_by_variant(
    replays: tuple[dict[str, Any], ...],
    variant: str,
) -> dict[str, Any] | None:
    for replay in replays:
        if replay["variant"] == variant:
            return replay
    return None
