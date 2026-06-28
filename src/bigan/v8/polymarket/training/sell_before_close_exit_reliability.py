"""SELL_BEFORE_CLOSE replay exit reliability diagnostics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from bigan.v8.polymarket.action_value_guards import action_value_bucket_payload
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.contracts import compact_safety_fields
from bigan.v8.polymarket.training.model_ranking_diagnostics import (
    SELL_BEFORE_CLOSE_ONLY_SOURCE_CANDIDATE_NAME,
)

SELL_BEFORE_CLOSE_EXIT_RELIABILITY_SCHEMA_VERSION = (
    "bigan-v8-polymarket-sell-before-close-exit-reliability-v1"
)
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


def build_sell_before_close_exit_reliability_report(
    *,
    dataset: Any,
    action_family_counterfactual_replays: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Build diagnostic-only SELL_BEFORE_CLOSE exit reliability evidence."""

    replay = _replay_by_variant(
        action_family_counterfactual_replays,
        SELL_BEFORE_CLOSE_ONLY_SOURCE_CANDIDATE_NAME,
    )
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
    summary = _replay_summary(
        rows=positions,
        replay=replay,
    )
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
    summary["sell_before_close_best_diagnostic_exit_variant"] = best_variant[
        "variant"
    ]
    summary["sell_before_close_best_diagnostic_exit_variant_total_pnl"] = (
        best_variant["total_pnl"]
    )
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
        "sell_before_close_exit_failure_interpretation": summary[
            "sell_before_close_exit_failure_interpretation"
        ],
        **compact_safety_fields(),
    }
    report["sell_before_close_exit_reliability_report_id"] = canonical_json_sha256(
        report
    )
    return report


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
            "- paper_only: true",
            "- capital_at_risk: false",
            "- polymarket_write_enabled: false",
            "- wallet_signing_enabled: false",
            "",
        ]
    )
    return "\n".join(lines)


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
