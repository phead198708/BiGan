"""Action-family eligibility diagnostics for calibrated Polymarket policies."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Any

from bigan.v8.polymarket.action_value_guards import (
    ACTION_FAMILY_HIGH_SCORE_UNPROFITABLE,
    ACTION_FAMILY_HOLD_TO_SETTLEMENT,
    ACTION_FAMILY_INELIGIBLE,
    ACTION_FAMILY_NO_TRADE,
    ACTION_FAMILY_SELL_BEFORE_CLOSE,
    BUY_DOWN_HOLD_TO_SETTLEMENT_UNPROFITABLE,
    BUY_UP_HOLD_TO_SETTLEMENT_UNPROFITABLE,
    HOLD_TO_SETTLEMENT_HIGH_SCORE_UNPROFITABLE,
    HOLD_TO_SETTLEMENT_LONGSHOT_GUARD,
    LONGSHOT_GUARD_PRICE_BUCKETS,
    LONGSHOT_GUARD_RAW_SCORE_BUCKETS,
    LONGSHOT_GUARD_TIME_TO_CLOSE_BUCKETS,
    action_value_action_family,
    action_value_bucket_payload,
)
from bigan.v8.polymarket.training.action_value_calibration import (
    ACTION_VALUE_HIGH_SCORE_MIN_SUPPORT,
    ACTION_VALUE_HIGH_SCORE_THRESHOLD,
)
from bigan.v8.polymarket.training.contracts import (
    ACTION_VALUE_LABEL_ACTIONS,
    POLYMARKET_POLICY_SCHEMA_VERSION,
    POLYMARKET_POLICY_TRAINING_PHASE,
    PolymarketPolicyExample,
    PolymarketPolicyPrediction,
    compact_safety_fields,
)

ACTION_FAMILY_ELIGIBILITY_SCHEMA_VERSION = (
    "bigan-v8-polymarket-action-family-eligibility-v1"
)
HOLD_TO_SETTLEMENT_LONGSHOT_GUARD_SCHEMA_VERSION = (
    "bigan-v8-polymarket-hold-to-settlement-longshot-guard-v1"
)
ACTION_FAMILY_REPLAY_VARIANTS_SCHEMA_VERSION = (
    "bigan-v8-polymarket-action-family-replay-variants-v1"
)
ACTION_FAMILY_COUNTERFACTUAL_REPLAY_SCHEMA_VERSION = (
    "bigan-v8-polymarket-action-family-counterfactual-replay-v1"
)
ACTION_FAMILY_MIN_HIGH_SCORE_SUPPORT = ACTION_VALUE_HIGH_SCORE_MIN_SUPPORT
P_UP_MATERIAL_DISAGREEMENT_THRESHOLD = 0.55


def build_action_family_eligibility_report(
    *,
    examples: tuple[PolymarketPolicyExample, ...],
    predictions: tuple[PolymarketPolicyPrediction, ...],
    execution_buffer: float,
    min_support: int = ACTION_FAMILY_MIN_HIGH_SCORE_SUPPORT,
    high_score_threshold: float = ACTION_VALUE_HIGH_SCORE_THRESHOLD,
) -> dict[str, Any]:
    """Build out-of-sample high-score action-family gate diagnostics."""

    rows = _high_score_rows(
        examples=examples,
        predictions=predictions,
        high_score_threshold=high_score_threshold,
    )
    family_gate_results = _gate_results(
        rows=rows,
        group_field="action_family",
        execution_buffer=execution_buffer,
        min_support=min_support,
    )
    action_gate_results = _gate_results(
        rows=rows,
        group_field="action",
        execution_buffer=execution_buffer,
        min_support=min_support,
    )
    enabled_families = sorted(
        family
        for family, gate in family_gate_results.items()
        if int(gate["support_count"]) > 0
    )
    eligible_families = sorted(
        family
        for family, gate in family_gate_results.items()
        if int(gate["support_count"]) > 0 and bool(gate["gate_passed"])
    )
    ineligible_families = sorted(set(enabled_families) - set(eligible_families))
    reason_codes = _eligibility_reason_codes(
        enabled_families=enabled_families,
        ineligible_families=ineligible_families,
        action_gate_results=action_gate_results,
    )
    paper_decision_eligible = bool(enabled_families) and not reason_codes
    return {
        "schema_version": ACTION_FAMILY_ELIGIBILITY_SCHEMA_VERSION,
        "policy_schema_version": POLYMARKET_POLICY_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "replay_split": "shadow",
        "out_of_sample_replay": True,
        "high_score_threshold": high_score_threshold,
        "min_family_high_score_support": min_support,
        "family_high_score_execution_buffer": execution_buffer,
        "high_score_support_count": len(rows),
        "high_score_realized_return_mean": _mean(
            [row["realized_return"] for row in rows]
        ),
        "high_score_realized_return_sum": _sum(
            [row["realized_return"] for row in rows]
        ),
        "high_score_calibrated_score_mean": _mean(
            [row["calibrated_score"] for row in rows]
        ),
        "enabled_action_families": enabled_families,
        "eligible_action_families": eligible_families,
        "ineligible_action_families": ineligible_families,
        "action_family_gate_results": family_gate_results,
        "action_gate_results": action_gate_results,
        "action_family_paper_decision_eligible": paper_decision_eligible,
        "action_family_paper_decision_ineligible_reasons": reason_codes,
        "reason_codes": reason_codes,
        "high_score_by_action": _group_summaries(rows, ("action",)),
        "high_score_by_action_family": _group_summaries(rows, ("action_family",)),
        "high_score_by_side": _group_summaries(rows, ("side",)),
        "high_score_by_price_bucket": _group_summaries(rows, ("price_bucket",)),
        "high_score_by_time_to_close_bucket": _group_summaries(
            rows,
            ("time_to_close_bucket",),
        ),
        "high_score_by_raw_score_bucket": _group_summaries(
            rows,
            ("raw_score_bucket",),
        ),
        "high_score_by_action_family_side_price_time_raw_bucket": _group_summaries(
            rows,
            (
                "action_family",
                "side",
                "price_bucket",
                "time_to_close_bucket",
                "raw_score_bucket",
            ),
        ),
        "negative_high_score_examples": _negative_examples(rows),
        **compact_safety_fields(),
    }


def build_hold_to_settlement_longshot_guard_report(
    *,
    examples: tuple[PolymarketPolicyExample, ...],
    predictions: tuple[PolymarketPolicyPrediction, ...],
    execution_buffer: float,
    high_score_threshold: float = ACTION_VALUE_HIGH_SCORE_THRESHOLD,
) -> dict[str, Any]:
    """Build diagnostics for the initial HOLD_TO_SETTLEMENT long-shot guard."""

    rows = _high_score_rows(
        examples=examples,
        predictions=predictions,
        high_score_threshold=high_score_threshold,
    )
    guarded_rows = [
        row for row in rows if row["hold_to_settlement_longshot_guard_applies"]
    ]
    return {
        "schema_version": HOLD_TO_SETTLEMENT_LONGSHOT_GUARD_SCHEMA_VERSION,
        "policy_schema_version": POLYMARKET_POLICY_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "replay_split": "shadow",
        "out_of_sample_replay": True,
        "guard_enabled": True,
        "guard_mode": "block_to_no_trade",
        "guard_reason_codes": [
            HOLD_TO_SETTLEMENT_LONGSHOT_GUARD,
            ACTION_FAMILY_INELIGIBLE,
        ],
        "price_buckets": list(LONGSHOT_GUARD_PRICE_BUCKETS),
        "time_to_close_buckets": list(LONGSHOT_GUARD_TIME_TO_CLOSE_BUCKETS),
        "raw_score_buckets": list(LONGSHOT_GUARD_RAW_SCORE_BUCKETS),
        "high_score_threshold": high_score_threshold,
        "execution_buffer": execution_buffer,
        "high_score_support_count": len(rows),
        "guarded_high_score_count": len(guarded_rows),
        "guarded_high_score_realized_return_mean": _mean(
            [row["realized_return"] for row in guarded_rows]
        ),
        "guarded_high_score_realized_return_sum": _sum(
            [row["realized_return"] for row in guarded_rows]
        ),
        "guarded_by_action": _group_summaries(guarded_rows, ("action",)),
        "guarded_by_side": _group_summaries(guarded_rows, ("side",)),
        "guarded_by_price_time_raw_bucket": _group_summaries(
            guarded_rows,
            ("price_bucket", "time_to_close_bucket", "raw_score_bucket"),
        ),
        "negative_guarded_examples": _negative_examples(guarded_rows),
        **compact_safety_fields(),
    }


def build_action_family_replay_variants_report(
    *,
    examples: tuple[PolymarketPolicyExample, ...],
    predictions: tuple[PolymarketPolicyPrediction, ...],
    execution_buffer: float,
    thresholds: tuple[float, ...] = (0.0, 0.03, 0.05),
    min_support: int = ACTION_FAMILY_MIN_HIGH_SCORE_SUPPORT,
) -> dict[str, Any]:
    """Estimate before/after policy variants on the same shadow evidence set."""

    baseline_rows = _high_score_rows(
        examples=examples,
        predictions=predictions,
        high_score_threshold=ACTION_VALUE_HIGH_SCORE_THRESHOLD,
    )
    baseline_gates = _gate_results(
        rows=baseline_rows,
        group_field="action_family",
        execution_buffer=execution_buffer,
        min_support=min_support,
    )
    passed_baseline_families = {
        family for family, gate in baseline_gates.items() if gate["gate_passed"]
    }
    passed_bucket_keys = _passed_bucket_keys(
        rows=baseline_rows,
        execution_buffer=execution_buffer,
        min_support=min_support,
    )
    variants = [
        _variant_report(
            variant="A_baseline_current_calibrated_policy_blocked",
            rows=baseline_rows,
            candidate_rows=baseline_rows,
            threshold=ACTION_VALUE_HIGH_SCORE_THRESHOLD,
            gate_mode="no_action_family_filter",
            execution_buffer=execution_buffer,
            min_support=min_support,
            blocked=True,
            reason_codes=[ACTION_FAMILY_HIGH_SCORE_UNPROFITABLE],
        ),
        _variant_report(
            variant="B_hold_to_settlement_disabled",
            rows=[
                row
                for row in baseline_rows
                if row["action_family"] != ACTION_FAMILY_HOLD_TO_SETTLEMENT
            ],
            candidate_rows=baseline_rows,
            threshold=ACTION_VALUE_HIGH_SCORE_THRESHOLD,
            gate_mode="hold_to_settlement_disabled",
            execution_buffer=execution_buffer,
            min_support=min_support,
        ),
        _variant_report(
            variant="C_sell_before_close_only",
            rows=[
                row
                for row in baseline_rows
                if row["action_family"] == ACTION_FAMILY_SELL_BEFORE_CLOSE
            ],
            candidate_rows=baseline_rows,
            threshold=ACTION_VALUE_HIGH_SCORE_THRESHOLD,
            gate_mode="sell_before_close_only",
            execution_buffer=execution_buffer,
            min_support=min_support,
        ),
        _variant_report(
            variant="D_hold_to_settlement_allowed_only_for_passed_buckets",
            rows=[
                row
                for row in baseline_rows
                if (
                    row["action_family"] != ACTION_FAMILY_HOLD_TO_SETTLEMENT
                    and row["action_family"] in passed_baseline_families
                )
                or (
                    row["action_family"] == ACTION_FAMILY_HOLD_TO_SETTLEMENT
                    and row["action_family"] in passed_baseline_families
                    and _bucket_key(row) in passed_bucket_keys
                )
            ],
            candidate_rows=baseline_rows,
            threshold=ACTION_VALUE_HIGH_SCORE_THRESHOLD,
            gate_mode="passed_family_and_bucket_only",
            execution_buffer=execution_buffer,
            min_support=min_support,
        ),
    ]
    threshold_sweep = []
    for threshold in thresholds:
        candidate_rows = _high_score_rows(
            examples=examples,
            predictions=predictions,
            high_score_threshold=threshold,
        )
        family_gates = _gate_results(
            rows=candidate_rows,
            group_field="action_family",
            execution_buffer=execution_buffer,
            min_support=min_support,
        )
        passed_families = {
            family for family, gate in family_gates.items() if gate["gate_passed"]
        }
        selected_rows = [
            row for row in candidate_rows if row["action_family"] in passed_families
        ]
        threshold_sweep.append(
            _variant_report(
                variant=f"E_threshold_{threshold:.2f}_action_family_gates_enabled",
                rows=selected_rows,
                candidate_rows=candidate_rows,
                threshold=threshold,
                gate_mode="action_family_gates_enabled",
                execution_buffer=execution_buffer,
                min_support=min_support,
                family_gate_results=family_gates,
                eligible_action_families=sorted(passed_families),
                reason_codes=(
                    []
                    if passed_families
                    else [ACTION_FAMILY_HIGH_SCORE_UNPROFITABLE]
                ),
            )
        )
    return {
        "schema_version": ACTION_FAMILY_REPLAY_VARIANTS_SCHEMA_VERSION,
        "policy_schema_version": POLYMARKET_POLICY_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "report_mode": "filtered_high_score_estimate",
        "promotion_evidence_eligible": False,
        "counterfactual_replay_required_for_promotion": True,
        "replay_split": "shadow",
        "out_of_sample_replay": True,
        "execution_buffer": execution_buffer,
        "min_family_high_score_support": min_support,
        "variants": variants,
        "threshold_sweep_with_action_family_gates": threshold_sweep,
        **compact_safety_fields(),
    }


def build_action_family_counterfactual_prediction_sets(
    *,
    examples: tuple[PolymarketPolicyExample, ...],
    predictions: tuple[PolymarketPolicyPrediction, ...],
    execution_buffer: float,
    thresholds: tuple[float, ...] = (0.0, 0.03, 0.05),
    min_support: int = ACTION_FAMILY_MIN_HIGH_SCORE_SUPPORT,
) -> tuple[dict[str, Any], ...]:
    """Build re-ranked counterfactual prediction sets for replay."""

    _validate_aligned(examples, predictions)
    baseline_rows = _high_score_rows(
        examples=examples,
        predictions=predictions,
        high_score_threshold=ACTION_VALUE_HIGH_SCORE_THRESHOLD,
    )
    baseline_gates = _gate_results(
        rows=baseline_rows,
        group_field="action_family",
        execution_buffer=execution_buffer,
        min_support=min_support,
    )
    passed_baseline_families = {
        family for family, gate in baseline_gates.items() if gate["gate_passed"]
    }
    passed_bucket_keys = _passed_bucket_keys(
        rows=baseline_rows,
        execution_buffer=execution_buffer,
        min_support=min_support,
    )
    variants = [
        _counterfactual_variant(
            variant="A_baseline_current_policy_with_runtime_guards",
            predictions=predictions,
            ev_threshold=execution_buffer,
            allowed_mode="baseline",
            family_gate_results=baseline_gates,
            eligible_action_families=sorted(passed_baseline_families),
            description="baseline calibrated policy replay using runtime guards",
        ),
        _counterfactual_variant(
            variant="B_hold_to_settlement_disabled_reranked",
            predictions=predictions,
            ev_threshold=execution_buffer,
            allowed_mode="hold_to_settlement_disabled",
            description="disable HOLD_TO_SETTLEMENT, then re-rank remaining calibrated actions",
        ),
        _counterfactual_variant(
            variant="C_sell_before_close_only_reranked",
            predictions=predictions,
            ev_threshold=execution_buffer,
            allowed_mode="sell_before_close_only",
            description="allow SELL_BEFORE_CLOSE actions only, then re-rank",
        ),
        _counterfactual_variant(
            variant="D_hold_to_settlement_allowed_only_for_passed_buckets_reranked",
            predictions=predictions,
            ev_threshold=execution_buffer,
            allowed_mode="passed_family_and_bucket_only",
            family_gate_results=baseline_gates,
            eligible_action_families=sorted(passed_baseline_families),
            passed_bucket_keys=passed_bucket_keys,
            description="allow actions only when family and HOLD bucket gates pass",
        ),
    ]
    for threshold in thresholds:
        candidate_rows = _high_score_rows(
            examples=examples,
            predictions=predictions,
            high_score_threshold=threshold,
        )
        family_gates = _gate_results(
            rows=candidate_rows,
            group_field="action_family",
            execution_buffer=execution_buffer,
            min_support=min_support,
        )
        passed_families = {
            family for family, gate in family_gates.items() if gate["gate_passed"]
        }
        variants.append(
            _counterfactual_variant(
                variant=f"E_threshold_{threshold:.2f}_action_family_gates_reranked",
                predictions=predictions,
                ev_threshold=threshold,
                allowed_mode="action_family_gates_enabled",
                family_gate_results=family_gates,
                eligible_action_families=sorted(passed_families),
                description=(
                    "re-rank using only action families that pass gates at the "
                    f"{threshold:.2f} high-score threshold"
                ),
            )
        )
    return tuple(variants)


def action_family_eligibility_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown summary for issue evidence."""

    lines = [
        "# Polymarket Action-Family Eligibility Report",
        "",
        f"- high_score_support_count: {report['high_score_support_count']}",
        "- high_score_realized_return_mean: "
        f"{report['high_score_realized_return_mean']}",
        f"- high_score_realized_return_sum: {report['high_score_realized_return_sum']}",
        f"- execution_buffer: {report['family_high_score_execution_buffer']}",
        f"- min_family_high_score_support: {report['min_family_high_score_support']}",
        "- action_family_paper_decision_eligible: "
        f"{str(report['action_family_paper_decision_eligible']).lower()}",
        "- action_family_paper_decision_ineligible_reasons: "
        f"{json.dumps(report['action_family_paper_decision_ineligible_reasons'])}",
        "- enabled_action_families: "
        f"{json.dumps(report['enabled_action_families'])}",
        "- eligible_action_families: "
        f"{json.dumps(report['eligible_action_families'])}",
        "",
        "## Family Gates",
        "",
    ]
    for family, gate in sorted(report["action_family_gate_results"].items()):
        lines.append(
            "- "
            f"{family}: support={gate['support_count']} "
            f"mean={gate['realized_return_mean']} "
            f"sum={gate['realized_return_sum']} "
            f"passed={str(gate['gate_passed']).lower()}"
        )
    lines.extend(["", "## Action Gates", ""])
    for action, gate in sorted(report["action_gate_results"].items()):
        lines.append(
            "- "
            f"{action}: support={gate['support_count']} "
            f"mean={gate['realized_return_mean']} "
            f"sum={gate['realized_return_sum']} "
            f"passed={str(gate['gate_passed']).lower()}"
        )
    lines.extend(
        [
            "",
            "## Negative Examples",
            "",
        ]
    )
    for row in report["negative_high_score_examples"][:10]:
        lines.append(
            "- "
            f"{row['action']} market_id={row['market_id']} "
            f"ts={row['decision_ts']} "
            f"realized={row['realized_return']} "
            f"score={row['calibrated_score']} "
            f"price={row['price_bucket']} "
            f"time={row['time_to_close_bucket']} "
            f"raw={row['raw_score_bucket']}"
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


def hold_to_settlement_longshot_guard_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown summary for the long-shot guard."""

    lines = [
        "# HOLD_TO_SETTLEMENT Long-Shot Guard Report",
        "",
        f"- guard_enabled: {str(report['guard_enabled']).lower()}",
        f"- guard_mode: {report['guard_mode']}",
        f"- guard_reason_codes: {json.dumps(report['guard_reason_codes'])}",
        f"- high_score_support_count: {report['high_score_support_count']}",
        f"- guarded_high_score_count: {report['guarded_high_score_count']}",
        "- guarded_high_score_realized_return_mean: "
        f"{report['guarded_high_score_realized_return_mean']}",
        "- guarded_high_score_realized_return_sum: "
        f"{report['guarded_high_score_realized_return_sum']}",
        f"- price_buckets: {json.dumps(report['price_buckets'])}",
        f"- time_to_close_buckets: {json.dumps(report['time_to_close_buckets'])}",
        f"- raw_score_buckets: {json.dumps(report['raw_score_buckets'])}",
        "",
        "## Guarded By Action",
        "",
    ]
    for row in report["guarded_by_action"]:
        lines.append(
            "- "
            f"{row['action']}: support={row['support_count']} "
            f"mean={row['realized_return_mean']} "
            f"sum={row['realized_return_sum']}"
        )
    lines.extend(["", "## Negative Guarded Examples", ""])
    for row in report["negative_guarded_examples"][:10]:
        lines.append(
            "- "
            f"{row['action']} market_id={row['market_id']} "
            f"ts={row['decision_ts']} "
            f"realized={row['realized_return']} "
            f"score={row['calibrated_score']} "
            f"price={row['price_bucket']} "
            f"time={row['time_to_close_bucket']} "
            f"raw={row['raw_score_bucket']}"
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


def action_family_replay_variants_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown summary for the replay variants."""

    lines = [
        "# Action-Family Replay Variants",
        "",
        f"- execution_buffer: {report['execution_buffer']}",
        f"- min_family_high_score_support: {report['min_family_high_score_support']}",
        "",
        "## Variants",
        "",
    ]
    for variant in report["variants"]:
        lines.append(_variant_markdown_line(variant))
    lines.extend(["", "## Threshold Sweep With Family Gates", ""])
    for variant in report["threshold_sweep_with_action_family_gates"]:
        lines.append(_variant_markdown_line(variant))
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


def _high_score_rows(
    *,
    examples: tuple[PolymarketPolicyExample, ...],
    predictions: tuple[PolymarketPolicyPrediction, ...],
    high_score_threshold: float,
) -> list[dict[str, Any]]:
    _validate_aligned(examples, predictions)
    rows = []
    for example, prediction in zip(examples, predictions, strict=True):
        action = _execution_policy_action(prediction)
        if action == "NO_TRADE":
            continue
        score = _execution_score(prediction, action)
        if score < high_score_threshold:
            continue
        raw_score = float(prediction.expected_return_by_action[action])
        bucket = action_value_bucket_payload(
            action=action,
            features=prediction.features,
            raw_score=raw_score,
        )
        p_up = _p_up(prediction)
        rows.append(
            {
                "market_id": example.market_id,
                "condition_id": example.condition_id,
                "slug": example.slug,
                "market_family": example.market_family,
                "decision_ts": int(example.decision_ts),
                "action": action,
                "calibrated_score": score,
                "raw_score": raw_score,
                "realized_return": float(example.action_return_targets[action]),
                "p_up_auxiliary": p_up,
                "estimated_up_probability": float(prediction.estimated_up_probability),
                "p_up_action_disagreement": _p_up_action_disagrees(
                    action=action,
                    p_up=p_up,
                ),
                **bucket,
            }
        )
    return rows


def _gate_results(
    *,
    rows: list[dict[str, Any]],
    group_field: str,
    execution_buffer: float,
    min_support: int,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_field])].append(row)
    return {
        key: _gate_payload(
            key=key,
            rows=group_rows,
            execution_buffer=execution_buffer,
            min_support=min_support,
        )
        for key, group_rows in sorted(grouped.items())
    }


def _gate_payload(
    *,
    key: str,
    rows: list[dict[str, Any]],
    execution_buffer: float,
    min_support: int,
) -> dict[str, Any]:
    realized_returns = [row["realized_return"] for row in rows]
    support_count = len(rows)
    realized_return_mean = _mean(realized_returns)
    realized_return_sum = _sum(realized_returns)
    support_passed = support_count >= min_support
    mean_exceeds_buffer = support_passed and realized_return_mean > execution_buffer
    sum_positive = realized_return_sum > 0.0
    gate_passed = support_passed and mean_exceeds_buffer and sum_positive
    return {
        "name": key,
        "support_count": support_count,
        "min_support": min_support,
        "support_passed": support_passed,
        "realized_return_mean": realized_return_mean,
        "realized_return_sum": realized_return_sum,
        "realized_return_mean_exceeds_execution_buffer": mean_exceeds_buffer,
        "realized_return_sum_positive": sum_positive,
        "execution_buffer": execution_buffer,
        "gate_passed": gate_passed,
    }


def _eligibility_reason_codes(
    *,
    enabled_families: list[str],
    ineligible_families: list[str],
    action_gate_results: dict[str, dict[str, Any]],
) -> list[str]:
    reasons = set()
    if not enabled_families or ineligible_families:
        reasons.add(ACTION_FAMILY_HIGH_SCORE_UNPROFITABLE)
    if ACTION_FAMILY_HOLD_TO_SETTLEMENT in ineligible_families:
        reasons.add(HOLD_TO_SETTLEMENT_HIGH_SCORE_UNPROFITABLE)
    for action, reason_code in (
        ("BUY_UP_HOLD_TO_SETTLEMENT", BUY_UP_HOLD_TO_SETTLEMENT_UNPROFITABLE),
        ("BUY_DOWN_HOLD_TO_SETTLEMENT", BUY_DOWN_HOLD_TO_SETTLEMENT_UNPROFITABLE),
    ):
        gate = action_gate_results.get(action)
        if gate is not None and int(gate["support_count"]) > 0 and not gate["gate_passed"]:
            reasons.add(reason_code)
    return sorted(reasons)


def _group_summaries(
    rows: list[dict[str, Any]],
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[field]) for field in group_fields)].append(row)
    summaries = []
    for key, group_rows in grouped.items():
        payload = {field: key[index] for index, field in enumerate(group_fields)}
        payload.update(_row_metrics(group_rows))
        summaries.append(payload)
    return sorted(
        summaries,
        key=lambda row: (-int(row["support_count"]), tuple(str(row[field]) for field in group_fields)),
    )


def _row_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    realized_returns = [row["realized_return"] for row in rows]
    calibrated_scores = [row["calibrated_score"] for row in rows]
    action_counts = Counter(str(row["action"]) for row in rows)
    family_counts = Counter(str(row["action_family"]) for row in rows)
    side_counts = Counter(str(row["side"]) for row in rows)
    unique_market_count = len({str(row["market_id"]) for row in rows})
    support_count = len(rows)
    max_side_count = max(side_counts.values(), default=0)
    return {
        "support_count": support_count,
        "realized_return_mean": _mean(realized_returns),
        "realized_return_sum": _sum(realized_returns),
        "calibrated_score_mean": _mean(calibrated_scores),
        "action_distribution": dict(sorted(action_counts.items())),
        "action_family_distribution": dict(sorted(family_counts.items())),
        "side_distribution": dict(sorted(side_counts.items())),
        "paper_decision_count_estimate": support_count,
        "unique_market_count": unique_market_count,
        "churn_repeated_decision_estimate": max(0, support_count - unique_market_count),
        "side_concentration": 0.0 if support_count == 0 else max_side_count / support_count,
        "p_up_action_disagreement_count": sum(
            bool(row["p_up_action_disagreement"]) for row in rows
        ),
        "p_up_action_disagreement_rate": (
            0.0
            if support_count == 0
            else sum(bool(row["p_up_action_disagreement"]) for row in rows)
            / support_count
        ),
    }


def _variant_report(
    *,
    variant: str,
    rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    threshold: float,
    gate_mode: str,
    execution_buffer: float,
    min_support: int,
    blocked: bool = False,
    reason_codes: list[str] | None = None,
    family_gate_results: dict[str, dict[str, Any]] | None = None,
    eligible_action_families: list[str] | None = None,
) -> dict[str, Any]:
    metrics = _row_metrics(rows)
    candidate_metrics = _row_metrics(candidate_rows)
    selected_reason_codes = [] if reason_codes is None else sorted(set(reason_codes))
    support_passed = metrics["support_count"] >= min_support
    mean_exceeds_buffer = (
        support_passed and metrics["realized_return_mean"] > execution_buffer
    )
    sum_positive = metrics["realized_return_sum"] > 0.0
    execution_buffer_gate_passed = support_passed and mean_exceeds_buffer and sum_positive
    blocked = blocked or not execution_buffer_gate_passed
    if not selected_reason_codes and blocked:
        selected_reason_codes = [ACTION_FAMILY_HIGH_SCORE_UNPROFITABLE]
    return {
        "variant": variant,
        "threshold": threshold,
        "gate_mode": gate_mode,
        "blocked": blocked or not rows,
        "reason_codes": selected_reason_codes,
        "min_support": min_support,
        "support_passed": support_passed,
        "execution_buffer": execution_buffer,
        "realized_return_mean_exceeds_execution_buffer": mean_exceeds_buffer,
        "realized_return_sum_positive": sum_positive,
        "execution_buffer_gate_passed": execution_buffer_gate_passed,
        "candidate_high_score_support_count": candidate_metrics["support_count"],
        "candidate_high_score_realized_return_mean": candidate_metrics[
            "realized_return_mean"
        ],
        "candidate_high_score_realized_return_sum": candidate_metrics[
            "realized_return_sum"
        ],
        "high_score_support_count": metrics["support_count"],
        "high_score_realized_return_mean": metrics["realized_return_mean"],
        "high_score_realized_return_sum": metrics["realized_return_sum"],
        "high_score_calibrated_score_mean": metrics["calibrated_score_mean"],
        "action_distribution": metrics["action_distribution"],
        "action_family_distribution": metrics["action_family_distribution"],
        "side_distribution": metrics["side_distribution"],
        "paper_decision_count_estimate": metrics["paper_decision_count_estimate"],
        "churn_repeated_decision_estimate": metrics["churn_repeated_decision_estimate"],
        "side_concentration": metrics["side_concentration"],
        "p_up_action_disagreement_count": metrics["p_up_action_disagreement_count"],
        "p_up_action_disagreement_rate": metrics["p_up_action_disagreement_rate"],
        "family_gate_results": family_gate_results or {},
        "eligible_action_families": eligible_action_families or [],
    }


def _passed_bucket_keys(
    *,
    rows: list[dict[str, Any]],
    execution_buffer: float,
    min_support: int,
) -> set[tuple[str, str, str, str]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_bucket_key(row)].append(row)
    passed = set()
    for key, group_rows in grouped.items():
        gate = _gate_payload(
            key="|".join(key),
            rows=group_rows,
            execution_buffer=execution_buffer,
            min_support=min_support,
        )
        if gate["gate_passed"]:
            passed.add(key)
    return passed


def _counterfactual_variant(
    *,
    variant: str,
    predictions: tuple[PolymarketPolicyPrediction, ...],
    ev_threshold: float,
    allowed_mode: str,
    description: str,
    family_gate_results: dict[str, dict[str, Any]] | None = None,
    eligible_action_families: list[str] | None = None,
    passed_bucket_keys: set[tuple[str, str, str, str]] | None = None,
) -> dict[str, Any]:
    eligible_families = tuple(eligible_action_families or ())
    bucket_keys = passed_bucket_keys or set()
    if allowed_mode == "baseline":
        replay_predictions = predictions
    else:
        replay_predictions = tuple(
            _rerank_counterfactual_prediction(
                prediction=prediction,
                allowed_mode=allowed_mode,
                eligible_action_families=eligible_families,
                passed_bucket_keys=bucket_keys,
            )
            for prediction in predictions
        )
    return {
        "variant": variant,
        "description": description,
        "counterfactual_replay_mode": "re_ranked_counterfactual_policy_replay",
        "allowed_mode": allowed_mode,
        "ev_threshold": ev_threshold,
        "eligible_action_families": list(eligible_families),
        "family_gate_results": family_gate_results or {},
        "prediction_count": len(replay_predictions),
        "predictions": replay_predictions,
        **compact_safety_fields(),
    }


def _rerank_counterfactual_prediction(
    *,
    prediction: PolymarketPolicyPrediction,
    allowed_mode: str,
    eligible_action_families: tuple[str, ...],
    passed_bucket_keys: set[tuple[str, str, str, str]],
) -> PolymarketPolicyPrediction:
    allowed_actions = [
        action
        for action in ACTION_VALUE_LABEL_ACTIONS
        if _counterfactual_action_allowed(
            action=action,
            prediction=prediction,
            allowed_mode=allowed_mode,
            eligible_action_families=eligible_action_families,
            passed_bucket_keys=passed_bucket_keys,
        )
    ]
    if "NO_TRADE" not in allowed_actions:
        allowed_actions.append("NO_TRADE")
    calibrated_returns = prediction.calibrated_expected_pnl_per_notional_by_action
    if not calibrated_returns:
        calibrated_returns = prediction.expected_return_by_action
    calibrated_best, calibrated_best_return, calibrated_second, calibrated_margin = (
        _rank_allowed_actions(
            returns={action: float(calibrated_returns[action]) for action in allowed_actions},
        )
    )
    raw_best, raw_best_return, raw_second, raw_margin = _rank_allowed_actions(
        returns={
            action: float(prediction.expected_return_by_action[action])
            for action in allowed_actions
        },
    )
    return replace(
        prediction,
        best_policy_action=raw_best,
        best_action_expected_return=raw_best_return,
        second_best_action_expected_return=raw_second,
        best_action_margin=raw_margin,
        calibrated_best_policy_action=calibrated_best,
        calibrated_expected_pnl_per_notional=calibrated_best_return,
        calibrated_second_best_expected_pnl_per_notional=calibrated_second,
        calibrated_action_margin=calibrated_margin,
    )


def _counterfactual_action_allowed(
    *,
    action: str,
    prediction: PolymarketPolicyPrediction,
    allowed_mode: str,
    eligible_action_families: tuple[str, ...],
    passed_bucket_keys: set[tuple[str, str, str, str]],
) -> bool:
    family = action_value_action_family(action)
    if family == ACTION_FAMILY_NO_TRADE:
        return True
    if allowed_mode == "hold_to_settlement_disabled":
        return family != ACTION_FAMILY_HOLD_TO_SETTLEMENT
    if allowed_mode == "sell_before_close_only":
        return family == ACTION_FAMILY_SELL_BEFORE_CLOSE
    if allowed_mode == "passed_family_and_bucket_only":
        if family not in eligible_action_families:
            return False
        if family != ACTION_FAMILY_HOLD_TO_SETTLEMENT:
            return True
        return _prediction_bucket_key(action=action, prediction=prediction) in passed_bucket_keys
    if allowed_mode == "action_family_gates_enabled":
        return family in eligible_action_families
    raise ValueError(f"unsupported counterfactual allowed_mode: {allowed_mode}")


def _prediction_bucket_key(
    *,
    action: str,
    prediction: PolymarketPolicyPrediction,
) -> tuple[str, str, str, str]:
    raw_score = float(prediction.expected_return_by_action[action])
    bucket = action_value_bucket_payload(
        action=action,
        features=prediction.features,
        raw_score=raw_score,
    )
    return (
        action,
        str(bucket["price_bucket"]),
        str(bucket["time_to_close_bucket"]),
        str(bucket["raw_score_bucket"]),
    )


def _rank_allowed_actions(returns: dict[str, float]) -> tuple[str, float, float, float]:
    ranked = sorted(returns.items(), key=lambda item: (-item[1], item[0]))
    best_action, best_return = ranked[0]
    second_return = ranked[1][1] if len(ranked) > 1 else best_return
    return best_action, best_return, second_return, best_return - second_return


def _bucket_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["action"]),
        str(row["price_bucket"]),
        str(row["time_to_close_bucket"]),
        str(row["raw_score_bucket"]),
    )


def _negative_examples(rows: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    keys = (
        "market_id",
        "decision_ts",
        "action",
        "action_family",
        "side",
        "realized_return",
        "calibrated_score",
        "raw_score",
        "price_bucket",
        "time_to_close_bucket",
        "raw_score_bucket",
        "p_up_auxiliary",
        "p_up_action_disagreement",
        "hold_to_settlement_longshot_guard_applies",
    )
    return [
        {key: row[key] for key in keys}
        for row in sorted(rows, key=lambda item: (item["realized_return"], item["decision_ts"]))[
            :limit
        ]
    ]


def _execution_policy_action(prediction: PolymarketPolicyPrediction) -> str:
    calibrated_action = prediction.calibrated_best_policy_action
    if calibrated_action is not None:
        return str(calibrated_action)
    return str(prediction.best_policy_action)


def _execution_score(prediction: PolymarketPolicyPrediction, action: str) -> float:
    if prediction.calibrated_expected_pnl_per_notional is not None:
        return float(prediction.calibrated_expected_pnl_per_notional)
    return float(prediction.expected_return_by_action[action])


def _p_up(prediction: PolymarketPolicyPrediction) -> float:
    if prediction.p_up_auxiliary is not None:
        return float(prediction.p_up_auxiliary)
    return float(prediction.estimated_up_probability)


def _p_up_action_disagrees(*, action: str, p_up: float) -> bool:
    if action.startswith("BUY_DOWN_"):
        return p_up >= P_UP_MATERIAL_DISAGREEMENT_THRESHOLD
    if action.startswith("BUY_UP_"):
        return p_up <= 1.0 - P_UP_MATERIAL_DISAGREEMENT_THRESHOLD
    return False


def _validate_aligned(
    examples: tuple[PolymarketPolicyExample, ...],
    predictions: tuple[PolymarketPolicyPrediction, ...],
) -> None:
    if len(examples) != len(predictions):
        raise ValueError("action-family examples/predictions length mismatch")
    for example, prediction in zip(examples, predictions, strict=True):
        if (example.market_id, example.decision_ts) != (
            prediction.market_id,
            prediction.decision_ts,
        ):
            raise ValueError("action-family examples/predictions misaligned")
        missing_targets = set(ACTION_VALUE_LABEL_ACTIONS) - set(
            example.action_return_targets
        )
        if missing_targets:
            raise ValueError(
                "action-family example missing targets: "
                + ", ".join(sorted(missing_targets))
            )
        missing_predictions = set(ACTION_VALUE_LABEL_ACTIONS) - set(
            prediction.expected_return_by_action
        )
        if missing_predictions:
            raise ValueError(
                "action-family prediction missing actions: "
                + ", ".join(sorted(missing_predictions))
            )


def _variant_markdown_line(variant: dict[str, Any]) -> str:
    return (
        "- "
        f"{variant['variant']}: "
        f"support={variant['high_score_support_count']} "
        f"mean={variant['high_score_realized_return_mean']} "
        f"sum={variant['high_score_realized_return_sum']} "
        f"paper_decisions={variant['paper_decision_count_estimate']} "
        f"churn={variant['churn_repeated_decision_estimate']} "
        f"side_concentration={variant['side_concentration']} "
        f"p_up_disagreement={variant['p_up_action_disagreement_rate']} "
        f"actions={json.dumps(variant['action_distribution'], sort_keys=True)} "
        f"blocked={str(variant['blocked']).lower()} "
        f"reasons={json.dumps(variant['reason_codes'])}"
    )


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return _sum(values) / len(values)


def _sum(values: list[float]) -> float:
    return float(sum(values))
