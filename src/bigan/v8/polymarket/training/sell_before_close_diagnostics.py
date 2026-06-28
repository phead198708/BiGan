"""SELL_BEFORE_CLOSE-only p_up/action disagreement diagnostics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from bigan.v8.polymarket.action_value_guards import action_value_bucket_payload
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.contracts import (
    ACTION_VALUE_LABEL_ACTIONS,
    POLYMARKET_POLICY_SCHEMA_VERSION,
    POLYMARKET_POLICY_TRAINING_PHASE,
    PolymarketPolicyExample,
    compact_safety_fields,
)
from bigan.v8.polymarket.training.model_ranking_diagnostics import (
    P_UP_MATERIAL_DISAGREEMENT_THRESHOLD,
    SELL_BEFORE_CLOSE_ONLY_SOURCE_CANDIDATE_NAME,
)

SELL_BEFORE_CLOSE_P_UP_DISAGREEMENT_DIAGNOSTIC_SCHEMA_VERSION = (
    "bigan-v8-polymarket-sell-before-close-p-up-disagreement-diagnostic-v1"
)
SELL_BEFORE_CLOSE_ACTIONS = (
    "BUY_UP_SELL_BEFORE_CLOSE",
    "BUY_DOWN_SELL_BEFORE_CLOSE",
)
P_UP_DISAGREEMENT_INTERPRETATIONS = (
    "likely_model_direction_error",
    "likely_auxiliary_p_up_action_value_semantic_mismatch",
    "mixed_evidence",
    "insufficient_evidence",
)
GROUP_FIELDS = (
    "selected_action",
    "selected_side",
    "p_up_action_disagrees",
    "time_to_close_bucket",
    "price_bucket",
    "raw_score_bucket",
    "fine_action_family",
    "queue_fill_probability_bucket",
    "executable_liquidity_bucket",
)


def build_sell_before_close_p_up_disagreement_diagnostic_report(
    *,
    shadow_examples: tuple[PolymarketPolicyExample, ...],
    model_ranking_candidate_comparison: dict[str, Any],
    action_family_counterfactual_replays: tuple[dict[str, Any], ...],
    pnl_notional: float,
) -> dict[str, Any]:
    """Build a diagnostic-only report for the SELL_BEFORE_CLOSE source candidate."""

    candidate = _candidate_by_name(
        model_ranking_candidate_comparison,
        SELL_BEFORE_CLOSE_ONLY_SOURCE_CANDIDATE_NAME,
    )
    candidate_predictions = tuple(candidate.get("candidate_predictions", []))
    if candidate_predictions:
        _validate_aligned(shadow_examples, candidate_predictions)
    rows = [
        _diagnostic_row(
            example=example,
            prediction=prediction,
            pnl_notional=pnl_notional,
        )
        for example, prediction in zip(
            shadow_examples,
            candidate_predictions,
            strict=False,
        )
        if _selected_action(prediction) in SELL_BEFORE_CLOSE_ACTIONS
    ]
    replay = _replay_by_variant(
        action_family_counterfactual_replays,
        SELL_BEFORE_CLOSE_ONLY_SOURCE_CANDIDATE_NAME,
    )
    comparisons = _comparison_tables(rows)
    replay_attribution = _replay_attribution(rows=rows, replay=replay)
    summary = _diagnostic_summary(
        rows=rows,
        comparisons=comparisons,
        replay_attribution=replay_attribution,
    )
    report = {
        "schema_version": SELL_BEFORE_CLOSE_P_UP_DISAGREEMENT_DIAGNOSTIC_SCHEMA_VERSION,
        "policy_schema_version": POLYMARKET_POLICY_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": SELL_BEFORE_CLOSE_ONLY_SOURCE_CANDIDATE_NAME,
        "diagnostic_only": True,
        "promotion_evidence_eligible": False,
        "source_model_candidate_eligible": False,
        "paper_run_resume_allowed": False,
        "paper_run_resume_blocked_reason": "diagnostic_only_not_promotion_evidence",
        "p_up_material_disagreement_threshold": P_UP_MATERIAL_DISAGREEMENT_THRESHOLD,
        "enabled_actions_considered": list(SELL_BEFORE_CLOSE_ACTIONS),
        "disabled_actions_not_counted_as_blockers": [
            "BUY_UP_HOLD_TO_SETTLEMENT",
            "BUY_DOWN_HOLD_TO_SETTLEMENT",
        ],
        "row_count": len(rows),
        "pnl_notional": float(pnl_notional),
        "pnl_contribution_unit": "paper_notional_label_return",
        "candidate_scoped_p_up_action_disagreement_rate": candidate[
            "candidate_scoped_p_up_action_disagreement_rate"
        ],
        "candidate_scoped_p_up_action_disagreement_within_limit": candidate[
            "candidate_scoped_p_up_action_disagreement_within_limit"
        ],
        "source_candidate_ineligible_reason_codes": candidate[
            "ineligible_reason_codes"
        ],
        "summary": summary,
        "row_level_diagnostics": rows,
        "grouped_summaries": {
            field: _group_summaries(rows=rows, group_fields=(field,))
            for field in GROUP_FIELDS
        },
        "multi_dimensional_group_summaries": _group_summaries(
            rows=rows,
            group_fields=GROUP_FIELDS,
        ),
        "comparison_tables": comparisons,
        "counterfactual_replay_attribution": replay_attribution,
        "p_up_disagreement_interpretation": summary[
            "p_up_disagreement_interpretation"
        ],
        **compact_safety_fields(),
    }
    report["sell_before_close_p_up_disagreement_diagnostic_report_id"] = (
        canonical_json_sha256(report)
    )
    return report


def sell_before_close_p_up_disagreement_diagnostic_markdown(
    report: dict[str, Any],
) -> str:
    """Render compact SELL_BEFORE_CLOSE p_up disagreement diagnostics."""

    summary = report["summary"]
    attribution = report["counterfactual_replay_attribution"]
    lines = [
        "# SELL_BEFORE_CLOSE p_up Disagreement Diagnostic",
        "",
        f"- schema_version: `{report['schema_version']}`",
        f"- candidate_name: `{report['candidate_name']}`",
        f"- diagnostic_only: `{str(report['diagnostic_only']).lower()}`",
        f"- promotion_evidence_eligible: `{str(report['promotion_evidence_eligible']).lower()}`",
        f"- paper_run_resume_allowed: `{str(report['paper_run_resume_allowed']).lower()}`",
        f"- row_count: `{report['row_count']}`",
        f"- p_up_disagreement_interpretation: `{report['p_up_disagreement_interpretation']}`",
        "",
        "## Agreed vs Disagreed",
        "",
        "| group | support | disagreement_rate | trade_pnl | settlement_pnl | total_pnl | positive_total_pnl_rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["comparison_tables"]["disagreed_vs_agreed_sell_before_close"]:
        lines.append(_comparison_markdown_row(row, "p_up_action_disagrees"))
    lines.extend(
        [
            "",
            "## Replay Attribution",
            "",
            f"- total_polymarket_pnl: `{attribution['total_polymarket_pnl']}`",
            f"- realized_trade_pnl: `{attribution['realized_trade_pnl']}`",
            f"- settlement_pnl: `{attribution['settlement_pnl']}`",
            f"- positions_opened_but_not_closed_before_settlement: `{attribution['positions_opened_but_not_closed_before_settlement']}`",
            f"- positions_reached_sell_before_close_intent_but_settled: `{attribution['positions_reached_sell_before_close_intent_but_settled']}`",
            "",
            "## Summary Fields",
            "",
            f"- sell_before_close_disagreed_total_pnl_sum: `{summary['sell_before_close_disagreed_total_pnl_sum']}`",
            f"- sell_before_close_agreed_total_pnl_sum: `{summary['sell_before_close_agreed_total_pnl_sum']}`",
            f"- sell_before_close_disagreed_trade_pnl_sum: `{summary['sell_before_close_disagreed_trade_pnl_sum']}`",
            f"- sell_before_close_disagreed_settlement_pnl_sum: `{summary['sell_before_close_disagreed_settlement_pnl_sum']}`",
            f"- sell_before_close_residual_settlement_drag: `{summary['sell_before_close_residual_settlement_drag']}`",
            "",
            "- paper_only: true",
            "- capital_at_risk: false",
            "- polymarket_write_enabled: false",
            "- wallet_signing_enabled: false",
            "",
        ]
    )
    return "\n".join(lines)


def sell_before_close_p_up_disagreement_summary(
    report: dict[str, Any],
) -> dict[str, Any]:
    """Return the compact summary embedded into other artifacts."""

    return {
        "diagnostic_report_schema_version": report["schema_version"],
        "diagnostic_only": True,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "candidate_name": report["candidate_name"],
        "row_count": report["row_count"],
        "p_up_disagreement_interpretation": report[
            "p_up_disagreement_interpretation"
        ],
        **{
            key: report["summary"][key]
            for key in (
                "sell_before_close_disagreed_total_pnl_sum",
                "sell_before_close_agreed_total_pnl_sum",
                "sell_before_close_disagreed_trade_pnl_sum",
                "sell_before_close_disagreed_settlement_pnl_sum",
                "sell_before_close_residual_settlement_drag",
                "candidate_scoped_p_up_action_disagreement_rate",
                "disagreed_support_count",
                "agreed_support_count",
            )
        },
    }


def _diagnostic_row(
    *,
    example: PolymarketPolicyExample,
    prediction: dict[str, Any],
    pnl_notional: float,
) -> dict[str, Any]:
    action = _selected_action(prediction)
    features = dict(prediction["features"])
    scores = _score_map(prediction)
    raw_scores = {
        label_action: float(prediction["expected_return_by_action"][label_action])
        for label_action in ACTION_VALUE_LABEL_ACTIONS
    }
    ranked_scores = sorted(
        ((label_action, float(scores[label_action])) for label_action in ACTION_VALUE_LABEL_ACTIONS),
        key=lambda item: (-item[1], item[0]),
    )
    second_best_action = next(
        candidate_action
        for candidate_action, _ in ranked_scores
        if candidate_action != action
    )
    p_up = _p_up(prediction)
    p_down = 1.0 - p_up
    bucket = action_value_bucket_payload(
        action=action,
        features=features,
        raw_score=raw_scores[action],
    )
    side = "UP" if action.startswith("BUY_UP_") else "DOWN"
    entry_ask = _side_feature(features, side, "ask")
    entry_bid = _side_feature(features, side, "bid")
    exit_path = example.sell_before_close_exit_path_targets.get(action, {})
    exit_bid = float(example.sell_before_close_exit_bid_targets.get(action, 0.0))
    executable_liquidity = float(
        example.sell_before_close_executable_liquidity_notional_targets.get(
            action,
            _side_feature(features, side, "executable_bid_notional") or 0.0,
        )
    )
    realized_trade_return = float(
        example.realized_trade_return_targets.get(action, 0.0)
    )
    settlement_return = float(example.settlement_return_targets.get(action, 0.0))
    realized_total_return = float(example.action_return_targets.get(action, 0.0))
    p_up_action_disagrees = _p_up_action_disagrees(action=action, p_up=p_up)
    trade_pnl = realized_trade_return * float(pnl_notional)
    settlement_pnl = settlement_return * float(pnl_notional)
    total_pnl = realized_total_return * float(pnl_notional)
    reason_codes = _row_reason_codes(
        p_up_action_disagrees=p_up_action_disagrees,
        realized_trade_return=realized_trade_return,
        settlement_return=settlement_return,
        exit_path=exit_path,
        executable_liquidity=executable_liquidity,
    )
    seconds_to_close = float(features.get("time_to_close_seconds", 0.0))
    return {
        "market_id": example.market_id,
        "slug": example.slug,
        "decision_ts": int(example.decision_ts),
        "market_end_ts": int(example.decision_ts + round(seconds_to_close * 1000.0)),
        "seconds_to_close": seconds_to_close,
        "selected_action": action,
        "selected_side": side,
        "p_up": p_up,
        "p_down": p_down,
        "p_up_direction": _p_up_direction(p_up),
        "action_side": side,
        "p_up_action_disagrees": p_up_action_disagrees,
        "calibrated_action_score": float(scores[action]),
        "second_best_action": second_best_action,
        "best_action_margin": float(scores[action]) - float(scores[second_best_action]),
        "entry_ask": entry_ask,
        "entry_bid": entry_bid,
        "exit_bid": exit_bid,
        "exit_ts": _exit_ts(exit_path),
        "sell_before_close_execution_class": (
            example.sell_before_close_execution_class_targets.get(
                action,
                "not_applicable",
            )
        ),
        "queue_fill_probability_estimate": float(
            example.sell_before_close_queue_fill_probability_targets.get(action, 0.0)
        ),
        "executable_liquidity_notional": executable_liquidity,
        "realized_trade_return": realized_trade_return,
        "settlement_return": settlement_return,
        "realized_total_return": realized_total_return,
        "trade_pnl_contribution": trade_pnl,
        "settlement_pnl_contribution": settlement_pnl,
        "total_pnl_contribution": total_pnl,
        "counterfactual_replay_variant": SELL_BEFORE_CLOSE_ONLY_SOURCE_CANDIDATE_NAME,
        "reason_codes": reason_codes,
        "time_to_close_bucket": bucket["time_to_close_bucket"],
        "price_bucket": bucket["price_bucket"],
        "raw_score_bucket": bucket["raw_score_bucket"],
        "fine_action_family": bucket["fine_action_family"],
        "queue_fill_probability_bucket": _queue_fill_probability_bucket(
            float(
                example.sell_before_close_queue_fill_probability_targets.get(
                    action,
                    0.0,
                )
            )
        ),
        "executable_liquidity_bucket": _executable_liquidity_bucket(
            executable_liquidity
        ),
    }


def _diagnostic_summary(
    *,
    rows: list[dict[str, Any]],
    comparisons: dict[str, Any],
    replay_attribution: dict[str, Any],
) -> dict[str, Any]:
    disagreed = [row for row in rows if row["p_up_action_disagrees"]]
    agreed = [row for row in rows if not row["p_up_action_disagrees"]]
    disagreed_total = _sum(disagreed, "total_pnl_contribution")
    agreed_total = _sum(agreed, "total_pnl_contribution")
    disagreed_trade = _sum(disagreed, "trade_pnl_contribution")
    disagreed_settlement = _sum(disagreed, "settlement_pnl_contribution")
    residual_drag = min(0.0, _sum(rows, "settlement_pnl_contribution"))
    return {
        "row_count": len(rows),
        "disagreed_support_count": len(disagreed),
        "agreed_support_count": len(agreed),
        "candidate_scoped_p_up_action_disagreement_rate": (
            0.0 if not rows else len(disagreed) / len(rows)
        ),
        "sell_before_close_disagreed_total_pnl_sum": disagreed_total,
        "sell_before_close_agreed_total_pnl_sum": agreed_total,
        "sell_before_close_disagreed_trade_pnl_sum": disagreed_trade,
        "sell_before_close_disagreed_settlement_pnl_sum": disagreed_settlement,
        "sell_before_close_residual_settlement_drag": residual_drag,
        "positive_trade_negative_settlement_disagreed_count": len(
            comparisons[
                "high_p_up_disagreement_rows_with_positive_trade_pnl_negative_settlement_pnl"
            ]
        ),
        "replay_total_polymarket_pnl": replay_attribution["total_polymarket_pnl"],
        "replay_realized_trade_pnl": replay_attribution["realized_trade_pnl"],
        "replay_settlement_pnl": replay_attribution["settlement_pnl"],
        "p_up_disagreement_interpretation": _interpretation(
            rows=rows,
            disagreed_total=disagreed_total,
            disagreed_trade=disagreed_trade,
            disagreed_settlement=disagreed_settlement,
            replay_attribution=replay_attribution,
        ),
    }


def _comparison_tables(rows: list[dict[str, Any]]) -> dict[str, Any]:
    disagreed_rows = [row for row in rows if row["p_up_action_disagrees"]]
    positive_trade_negative_settlement = [
        row
        for row in disagreed_rows
        if row["trade_pnl_contribution"] > 0.0
        and row["settlement_pnl_contribution"] < 0.0
    ]
    return {
        "disagreed_vs_agreed_sell_before_close": _group_summaries(
            rows=rows,
            group_fields=("p_up_action_disagrees",),
        ),
        "buy_up_sell_before_close_disagreed_vs_agreed": _group_summaries(
            rows=[
                row
                for row in rows
                if row["selected_action"] == "BUY_UP_SELL_BEFORE_CLOSE"
            ],
            group_fields=("p_up_action_disagrees",),
        ),
        "buy_down_sell_before_close_disagreed_vs_agreed": _group_summaries(
            rows=[
                row
                for row in rows
                if row["selected_action"] == "BUY_DOWN_SELL_BEFORE_CLOSE"
            ],
            group_fields=("p_up_action_disagrees",),
        ),
        "profitable_disagreed_rows_vs_unprofitable_disagreed_rows": (
            _group_summaries(
                rows=[
                    {
                        **row,
                        "total_pnl_positive": row["total_pnl_contribution"] > 0.0,
                    }
                    for row in disagreed_rows
                ],
                group_fields=("total_pnl_positive",),
            )
        ),
        "high_p_up_disagreement_rows_with_positive_trade_pnl_negative_settlement_pnl": (
            _top_rows(
                positive_trade_negative_settlement,
                sort_fields=(
                    "settlement_pnl_contribution",
                    "-trade_pnl_contribution",
                    "decision_ts",
                    "market_id",
                ),
                limit=20,
            )
        ),
        "high_p_up_disagreement_positive_trade_negative_settlement_summary": (
            _metrics(positive_trade_negative_settlement)
        ),
    }


def _group_summaries(
    *,
    rows: list[dict[str, Any]],
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[field]) for field in group_fields)].append(row)
    summaries = []
    for key, group_rows in groups.items():
        payload = {field: key[index] for index, field in enumerate(group_fields)}
        payload.update(_metrics(group_rows))
        summaries.append(payload)
    return sorted(
        summaries,
        key=lambda row: (
            -int(row["support_count"]),
            tuple(str(row[field]) for field in group_fields),
        ),
    )


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    support_count = len(rows)
    return {
        "support_count": support_count,
        "entry_count": support_count,
        "trade_count": sum(
            1
            for row in rows
            if row["sell_before_close_execution_class"]
            == "realizable_sell_before_close"
        ),
        "mean_calibrated_action_score": _mean(rows, "calibrated_action_score"),
        "mean_best_action_margin": _mean(rows, "best_action_margin"),
        "mean_p_up": _mean(rows, "p_up"),
        "p_up_disagreement_rate": (
            0.0
            if support_count == 0
            else sum(1 for row in rows if row["p_up_action_disagrees"])
            / support_count
        ),
        "mean_realized_trade_return": _mean(rows, "realized_trade_return"),
        "sum_realized_trade_return": _sum(rows, "realized_trade_return"),
        "mean_settlement_return": _mean(rows, "settlement_return"),
        "sum_settlement_return": _sum(rows, "settlement_return"),
        "mean_total_return": _mean(rows, "realized_total_return"),
        "sum_total_return": _sum(rows, "realized_total_return"),
        "mean_trade_pnl": _mean(rows, "trade_pnl_contribution"),
        "sum_trade_pnl": _sum(rows, "trade_pnl_contribution"),
        "mean_settlement_pnl": _mean(rows, "settlement_pnl_contribution"),
        "sum_settlement_pnl": _sum(rows, "settlement_pnl_contribution"),
        "mean_total_pnl": _mean(rows, "total_pnl_contribution"),
        "sum_total_pnl": _sum(rows, "total_pnl_contribution"),
        "positive_total_pnl_rate": (
            0.0
            if support_count == 0
            else sum(1 for row in rows if row["total_pnl_contribution"] > 0.0)
            / support_count
        ),
    }


def _replay_attribution(
    *,
    rows: list[dict[str, Any]],
    replay: dict[str, Any] | None,
) -> dict[str, Any]:
    summary = dict(replay.get("summary", {})) if replay else {}
    action_counts = dict(summary.get("action_counts", {}))
    reason_counts = dict(summary.get("reason_counts", {}))
    entry_count = int(summary.get("entry_decision_count", 0))
    sell_count = int(action_counts.get("SELL_UP", 0)) + int(
        action_counts.get("SELL_DOWN", 0)
    )
    residual_positions = max(0, entry_count - sell_count)
    realized_trade_pnl = float(summary.get("realized_trade_pnl", 0.0))
    settlement_pnl = float(summary.get("settlement_pnl", 0.0))
    total_pnl = float(summary.get("total_polymarket_pnl", 0.0))
    missed_exit_counts = _missed_exit_reason_counts(reason_counts)
    return {
        "variant": SELL_BEFORE_CLOSE_ONLY_SOURCE_CANDIDATE_NAME,
        "attribution_method": (
            "counterfactual_replay_summary_plus_label_row_diagnostics"
        ),
        "total_polymarket_pnl": total_pnl,
        "realized_trade_pnl": realized_trade_pnl,
        "settlement_pnl": settlement_pnl,
        "entry_decision_count": entry_count,
        "sell_decision_count": sell_count,
        "positions_opened_but_not_closed_before_settlement": residual_positions,
        "positions_reached_sell_before_close_intent_but_settled": residual_positions,
        "exit_opportunities_missed_due_to_gate_count": missed_exit_counts["gate"],
        "exit_opportunities_missed_due_to_cooldown_count": missed_exit_counts[
            "cooldown"
        ],
        "exit_opportunities_missed_due_to_liquidity_count": missed_exit_counts[
            "liquidity"
        ],
        "exit_opportunities_missed_due_to_timing_count": missed_exit_counts["timing"],
        "exit_signal_missing_count": missed_exit_counts["exit_signal_missing"],
        "pnl_by_attribution_category": [
            {
                "category": "closed_before_settlement_trades",
                "event_count": sell_count,
                "pnl": realized_trade_pnl,
            },
            {
                "category": "held_to_settlement_residuals",
                "event_count": residual_positions,
                "pnl": settlement_pnl,
            },
            {
                "category": "forced_settlement_events",
                "event_count": int(summary.get("settlement_event_count", 0)),
                "pnl": settlement_pnl,
            },
            {
                "category": "no_exit_available_events",
                "event_count": sum(
                    1
                    for row in rows
                    if row["sell_before_close_execution_class"]
                    != "realizable_sell_before_close"
                ),
                "pnl": _sum(
                    [
                        row
                        for row in rows
                        if row["sell_before_close_execution_class"]
                        != "realizable_sell_before_close"
                    ],
                    "total_pnl_contribution",
                ),
            },
            {
                "category": "exit_signal_missing_events",
                "event_count": missed_exit_counts["exit_signal_missing"],
                "pnl": 0.0,
            },
        ],
        "realized_trade_positive_but_settlement_negative": (
            realized_trade_pnl > 0.0 and settlement_pnl < 0.0
        ),
        "realized_trade_vs_settlement_explanation": (
            "The replay closed some positions profitably before settlement, but "
            "residual positions that were not closed before resolution created a "
            "negative settlement drag."
            if realized_trade_pnl > 0.0 and settlement_pnl < 0.0
            else "Replay PnL does not show positive realized trade PnL with negative settlement drag."
        ),
        "action_counts": action_counts,
        "reason_counts": reason_counts,
    }


def _missed_exit_reason_counts(reason_counts: dict[str, Any]) -> dict[str, int]:
    buckets = {
        "gate": 0,
        "cooldown": 0,
        "liquidity": 0,
        "timing": 0,
        "exit_signal_missing": 0,
    }
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


def _interpretation(
    *,
    rows: list[dict[str, Any]],
    disagreed_total: float,
    disagreed_trade: float,
    disagreed_settlement: float,
    replay_attribution: dict[str, Any],
) -> str:
    disagreed_count = sum(1 for row in rows if row["p_up_action_disagrees"])
    if disagreed_count < 10:
        return "insufficient_evidence"
    replay_trade = float(replay_attribution["realized_trade_pnl"])
    replay_settlement = float(replay_attribution["settlement_pnl"])
    if (
        disagreed_total < 0.0
        and (disagreed_trade > 0.0 or replay_trade > 0.0)
        and (disagreed_settlement < 0.0 or replay_settlement < 0.0)
    ):
        return "likely_auxiliary_p_up_action_value_semantic_mismatch"
    if disagreed_total < 0.0 and disagreed_trade <= 0.0:
        return "likely_model_direction_error"
    return "mixed_evidence"


def _row_reason_codes(
    *,
    p_up_action_disagrees: bool,
    realized_trade_return: float,
    settlement_return: float,
    exit_path: dict[str, Any],
    executable_liquidity: float,
) -> list[str]:
    reasons = []
    if p_up_action_disagrees:
        reasons.append("p_up_action_disagreement")
    if realized_trade_return > 0.0 and settlement_return < 0.0:
        reasons.append("positive_trade_return_negative_settlement_return")
    if settlement_return < 0.0:
        reasons.append("held_to_settlement_residual_risk")
    if not exit_path:
        reasons.append("exit_path_detail_unavailable")
    if executable_liquidity <= 0.0:
        reasons.append("zero_executable_liquidity_notional")
    return reasons


def _candidate_by_name(report: dict[str, Any], candidate_name: str) -> dict[str, Any]:
    for candidate in report["candidates"]:
        if candidate["candidate_name"] == candidate_name:
            return candidate
    raise ValueError(f"missing candidate: {candidate_name}")


def _replay_by_variant(
    replays: tuple[dict[str, Any], ...],
    variant: str,
) -> dict[str, Any] | None:
    for replay in replays:
        if replay["variant"] == variant:
            return replay
    return None


def _validate_aligned(
    examples: tuple[PolymarketPolicyExample, ...],
    predictions: tuple[dict[str, Any], ...],
) -> None:
    if len(examples) != len(predictions):
        raise ValueError("SELL_BEFORE_CLOSE diagnostic examples/predictions length mismatch")
    for example, prediction in zip(examples, predictions, strict=True):
        if (example.market_id, example.decision_ts) != (
            prediction["market_id"],
            prediction["decision_ts"],
        ):
            raise ValueError("SELL_BEFORE_CLOSE diagnostic examples/predictions misaligned")


def _score_map(prediction: dict[str, Any]) -> dict[str, float]:
    scores = prediction.get("calibrated_expected_pnl_per_notional_by_action") or (
        prediction["expected_return_by_action"]
    )
    return {action: float(scores[action]) for action in ACTION_VALUE_LABEL_ACTIONS}


def _selected_action(prediction: dict[str, Any]) -> str:
    return str(
        prediction.get("calibrated_best_policy_action")
        or prediction.get("best_policy_action")
    )


def _p_up(prediction: dict[str, Any]) -> float:
    value = prediction.get("p_up_auxiliary")
    if value is None:
        value = prediction["estimated_up_probability"]
    return float(value)


def _p_up_action_disagrees(*, action: str, p_up: float) -> bool:
    if action.startswith("BUY_DOWN_"):
        return p_up >= P_UP_MATERIAL_DISAGREEMENT_THRESHOLD
    if action.startswith("BUY_UP_"):
        return p_up <= 1.0 - P_UP_MATERIAL_DISAGREEMENT_THRESHOLD
    return False


def _p_up_direction(p_up: float) -> str:
    if p_up >= P_UP_MATERIAL_DISAGREEMENT_THRESHOLD:
        return "UP"
    if p_up <= 1.0 - P_UP_MATERIAL_DISAGREEMENT_THRESHOLD:
        return "DOWN"
    return "NEUTRAL"


def _side_feature(features: dict[str, Any], side: str, field: str) -> float | None:
    prefix = "up" if side == "UP" else "down"
    value = features.get(f"{prefix}_{field}")
    return None if value is None else float(value)


def _exit_ts(exit_path: dict[str, Any]) -> int | None:
    for field in ("best_executable_exit_ts", "exit_ts", "candidate_exit_ts"):
        value = exit_path.get(field)
        if value is not None:
            return int(value)
    return None


def _queue_fill_probability_bucket(value: float) -> str:
    if value < 0.25:
        return "<0.25"
    if value < 0.50:
        return "0.25-0.50"
    if value < 0.75:
        return "0.50-0.75"
    return ">=0.75"


def _executable_liquidity_bucket(value: float) -> str:
    if value <= 0.0:
        return "0"
    if value < 0.20:
        return "<0.20"
    if value < 1.00:
        return "0.20-1.00"
    return ">=1.00"


def _sum(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows)


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return _sum(rows, field) / len(rows)


def _top_rows(
    rows: list[dict[str, Any]],
    *,
    sort_fields: tuple[str, ...],
    limit: int,
) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        values = []
        for field in sort_fields:
            reverse_numeric = field.startswith("-")
            clean_field = field[1:] if reverse_numeric else field
            value = row[clean_field]
            if reverse_numeric and isinstance(value, (int, float)):
                value = -float(value)
            values.append(value)
        return tuple(values)

    return sorted(rows, key=key)[:limit]


def _comparison_markdown_row(row: dict[str, Any], group_field: str) -> str:
    return (
        "| {group} | {support} | {rate:.6f} | {trade:.6f} | "
        "{settlement:.6f} | {total:.6f} | {positive:.6f} |"
    ).format(
        group=row[group_field],
        support=row["support_count"],
        rate=row["p_up_disagreement_rate"],
        trade=row["sum_trade_pnl"],
        settlement=row["sum_settlement_pnl"],
        total=row["sum_total_pnl"],
        positive=row["positive_total_pnl_rate"],
    )
