"""Diagnostic coverage reports for guard-compatible SELL_BEFORE_CLOSE candidates."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.action_family_eligibility import (
    build_sell_before_close_side_balanced_prediction_set,
)
from bigan.v8.polymarket.training.contracts import (
    POLYMARKET_POLICY_TRAINING_PHASE,
    PolymarketPolicyDataset,
    PolymarketPolicyExample,
    PolymarketPolicyPrediction,
    compact_safety_fields,
)
from bigan.v8.polymarket.training.sell_before_close_source_candidates import (
    SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_THRESHOLDS,
    SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
)

GUARD_COMPATIBLE_CANDIDATE_COVERAGE_SCHEMA_VERSION = (
    "bigan-v8-polymarket-guard-compatible-candidate-coverage-v1"
)
SELL_BEFORE_CLOSE_ACTIONS = (
    "BUY_UP_SELL_BEFORE_CLOSE",
    "BUY_DOWN_SELL_BEFORE_CLOSE",
)
SIDES = ("UP", "DOWN")
COVERAGE_TARGETS = {
    "min_guard_compatible_candidate_count": 100,
    "min_guard_compatible_up_entry_count": 30,
    "min_guard_compatible_down_entry_count": 30,
    "min_guard_compatible_up_market_count": 20,
    "min_guard_compatible_down_market_count": 20,
    "two_sided_guard_compatible_entry_set_exists": True,
    "validation_guard_compatible_up_entry_count": 10,
    "validation_guard_compatible_down_entry_count": 10,
    "shadow_guard_compatible_up_entry_count": 10,
    "shadow_guard_compatible_down_entry_count": 10,
}
GUARD_ABLATION_VARIANTS = (
    {
        "variant_name": "baseline_current_guard",
        "threshold_overrides": {},
        "description": "Current p_up aligned exit-quality guard.",
    },
    {
        "variant_name": "without_p_up_alignment",
        "threshold_overrides": {"p_up_alignment_min": 0.0},
        "description": "Exit-quality guard with p_up side-alignment disabled.",
    },
    {
        "variant_name": "p_up_alignment_min_0_50",
        "threshold_overrides": {"p_up_alignment_min": 0.50},
        "description": "Current guard with p_up side-alignment minimum set to 0.50.",
    },
    {
        "variant_name": "p_up_alignment_min_0_52",
        "threshold_overrides": {"p_up_alignment_min": 0.52},
        "description": "Current guard with p_up side-alignment minimum set to 0.52.",
    },
    {
        "variant_name": "p_up_alignment_min_0_55",
        "threshold_overrides": {"p_up_alignment_min": 0.55},
        "description": "Current guard with p_up side-alignment minimum set to 0.55.",
    },
    {
        "variant_name": "p_up_alignment_min_0_58",
        "threshold_overrides": {"p_up_alignment_min": 0.58},
        "description": "Current guard with p_up side-alignment minimum set to 0.58.",
    },
    {
        "variant_name": "relaxed_spread_1200",
        "threshold_overrides": {"max_spread": 1200.0},
        "description": "Current guard with max spread relaxed to 1200 bps.",
    },
    {
        "variant_name": "relaxed_queue_fill_0_50",
        "threshold_overrides": {"min_queue_fill_probability_proxy": 0.50},
        "description": "Current guard with queue-fill proxy minimum relaxed to 0.50.",
    },
    {
        "variant_name": "relaxed_time_to_close_60",
        "threshold_overrides": {"min_seconds_to_close": 60.0},
        "description": "Current guard with minimum time-to-close relaxed to 60 seconds.",
    },
)


def build_guard_compatible_coverage_reports(
    *,
    dataset: PolymarketPolicyDataset,
    train_predictions: tuple[PolymarketPolicyPrediction, ...],
    validation_predictions: tuple[PolymarketPolicyPrediction, ...],
    shadow_predictions: tuple[PolymarketPolicyPrediction, ...],
    execution_buffer: float,
) -> dict[str, dict[str, Any]]:
    """Build #149 diagnostic-only coverage reports."""

    split_inputs = {
        "train": (dataset.train_examples, train_predictions),
        "validation": (dataset.validation_examples, validation_predictions),
        "shadow": (dataset.shadow_examples, shadow_predictions),
    }
    all_predictions = (
        *train_predictions,
        *validation_predictions,
        *shadow_predictions,
    )
    prediction_alignment = _prediction_alignment_diagnostics(
        examples=dataset.examples,
        predictions=all_predictions,
        split_name="overall",
    )
    if not prediction_alignment["prediction_alignment_passed"]:
        raise ValueError(
            "overall prediction alignment failed: "
            f"{json.dumps(prediction_alignment, sort_keys=True)}"
        )
    prediction_by_key = {
        (prediction.market_id, int(prediction.decision_ts)): prediction
        for prediction in all_predictions
    }
    overall_predictions = tuple(
        prediction_by_key[(example.market_id, int(example.decision_ts))]
        for example in dataset.examples
    )
    by_split = {
        split_name: _split_coverage(
            split_name=split_name,
            examples=examples,
            predictions=predictions,
            execution_buffer=execution_buffer,
        )
        for split_name, (examples, predictions) in split_inputs.items()
    }
    overall = _split_coverage(
        split_name="overall",
        examples=dataset.examples,
        predictions=overall_predictions,
        execution_buffer=execution_buffer,
    )
    target_results = _coverage_target_results(overall=overall, by_split=by_split)
    coverage_targets_passed = all(row["passed"] for row in target_results)
    base_report = {
        "schema_version": GUARD_COMPATIBLE_CANDIDATE_COVERAGE_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
        "diagnostic_only": True,
        "report_type": "guard_compatible_candidate_coverage",
        "selection_pool": "guard_compatible_rows",
        "execution_buffer": float(execution_buffer),
        "coverage_targets": dict(COVERAGE_TARGETS),
        "coverage_target_results": target_results,
        "coverage_targets_passed": coverage_targets_passed,
        "coverage_target_failed_reason_codes": [
            row["reason_code"] for row in target_results if not row["passed"]
        ],
        "#145_ready_for_rerun": coverage_targets_passed,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        "corpus_market_count": int(dataset.corpus_manifest.get("market_count", 0)),
        "dataset_hash": dataset.dataset_hash,
        "training_corpus_hash": dataset.training_corpus_hash,
        "prediction_alignment": prediction_alignment,
        "overall": overall,
        "by_split": by_split,
        **compact_safety_fields(),
    }
    _attach_report_id(base_report, "guard_compatible_candidate_coverage_report_id")
    reports = {
        "guard_compatible_candidate_coverage_report": base_report,
        "side_coverage_by_split_report": _coverage_view_report(
            base_report,
            report_type="side_coverage_by_split",
            metric_fields=(
                "pre_guard_candidate_count",
                "guard_compatible_candidate_count",
                "guard_compatible_up_entry_count",
                "guard_compatible_down_entry_count",
                "guard_compatible_up_market_count",
                "guard_compatible_down_market_count",
                "guard_compatible_side_count",
                "guard_compatible_side_entry_ratio",
                "guard_compatible_two_sided_entry_set_exists",
                "two_sided_guard_compatible_market_count",
            ),
        ),
        "entry_guard_pass_rate_by_side_report": _coverage_view_report(
            base_report,
            report_type="entry_guard_pass_rate_by_side",
            metric_fields=(
                "candidate_count_by_side",
                "guard_compatible_candidate_count_by_side",
                "guard_compatible_pass_rate_by_side",
                "exit_reliability_guard_pass_count_by_side",
                "p_up_side_alignment_pass_count_by_side",
            ),
        ),
        "exit_reliability_pass_rate_by_side_report": _coverage_view_report(
            base_report,
            report_type="exit_reliability_pass_rate_by_side",
            metric_fields=(
                "candidate_count_by_side",
                "exit_reliability_guard_pass_count_by_side",
                "exit_reliability_guard_pass_rate_by_side",
            ),
        ),
        "p_up_alignment_pass_rate_by_side_report": _coverage_view_report(
            base_report,
            report_type="p_up_alignment_pass_rate_by_side",
            metric_fields=(
                "candidate_count_by_side",
                "p_up_side_alignment_pass_count_by_side",
                "p_up_side_alignment_pass_rate_by_side",
            ),
        ),
        "liquidity_spread_staleness_regime_report": _coverage_view_report(
            base_report,
            report_type="liquidity_spread_staleness_regime",
            metric_fields=(
                "candidate_count_by_side",
                "liquidity_guard_pass_count_by_side",
                "liquidity_guard_pass_rate_by_side",
                "spread_guard_pass_count_by_side",
                "spread_guard_pass_rate_by_side",
                "staleness_guard_pass_count_by_side",
                "staleness_guard_pass_rate_by_side",
                "queue_fill_guard_pass_count_by_side",
                "queue_fill_guard_pass_rate_by_side",
                "regime_rows",
            ),
        ),
        "round_guard_coverage_report": _round_guard_coverage_report(base_report),
        "guard_ablation_coverage_report": _guard_ablation_coverage_report(
            dataset=dataset,
            split_inputs=split_inputs,
            overall_predictions=overall_predictions,
            execution_buffer=execution_buffer,
            prediction_alignment=prediction_alignment,
        ),
    }
    for report_name, report in reports.items():
        if report is base_report:
            continue
        _attach_report_id(report, f"{report_name}_id")
    return reports


def guard_compatible_coverage_markdown(report: dict[str, Any]) -> str:
    """Render a compact markdown view for any #149 coverage report."""

    if report["report_type"] == "guard_ablation_coverage":
        return _guard_ablation_markdown(report)

    overall = report["overall"]
    lines = [
        f"# {str(report['report_type']).replace('_', ' ').title()}",
        "",
        f"- candidate_name: `{report['candidate_name']}`",
        f"- diagnostic_only: `{str(report['diagnostic_only']).lower()}`",
        f"- coverage_targets_passed: `{str(report['coverage_targets_passed']).lower()}`",
        f"- #145_ready_for_rerun: `{str(report['#145_ready_for_rerun']).lower()}`",
        "- coverage_target_failed_reason_codes: "
        f"`{json.dumps(report['coverage_target_failed_reason_codes'])}`",
        f"- prediction_alignment_passed: "
        f"`{str(report.get('prediction_alignment', {}).get('prediction_alignment_passed', True)).lower()}`",
        "",
    ]
    if report["report_type"] == "round_guard_coverage":
        lines.extend(
            [
                "| split | zero_pre_guard | pre_guard_zero_compatible | one_sided | two_sided |",
                "|---|---:|---:|---:|---:|",
                _round_markdown_row("overall", overall),
            ]
        )
        for split_name in ("train", "validation", "shadow"):
            lines.append(_round_markdown_row(split_name, report["by_split"][split_name]))
    else:
        lines.extend(
            [
                "| split | pre_guard | guard_compatible | up | down | up_markets | down_markets | two_sided | side_ratio |",
                "|---|---:|---:|---:|---:|---:|---:|---|---:|",
                _coverage_markdown_row("overall", overall),
            ]
        )
        for split_name in ("train", "validation", "shadow"):
            lines.append(
                _coverage_markdown_row(split_name, report["by_split"][split_name])
            )
    if report["report_type"] == "liquidity_spread_staleness_regime":
        lines.extend(
            [
                "",
                "## Top Regimes",
                "",
                "| split | side | liquidity | queue | spread | staleness | candidates | guard_compatible | pass_rate |",
                "|---|---|---|---|---|---|---:|---:|---:|",
            ]
        )
        for row in overall.get("regime_rows", [])[:25]:
            lines.append(
                "| {split} | {side} | {liquidity} | {queue} | {spread} | {staleness} | "
                "{candidates} | {compatible} | {rate:.6f} |".format(
                    split=row["split_name"],
                    side=row["side"],
                    liquidity=row["liquidity_bucket"],
                    queue=row["queue_fill_bucket"],
                    spread=row["spread_bucket"],
                    staleness=row["staleness_bucket"],
                    candidates=row["candidate_count"],
                    compatible=row["guard_compatible_candidate_count"],
                    rate=row["guard_compatible_pass_rate"],
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


def guard_compatible_coverage_summary(report: dict[str, Any]) -> dict[str, Any]:
    overall = report["overall"]
    validation = report["by_split"]["validation"]
    shadow = report["by_split"]["shadow"]
    return {
        "schema_version": report["schema_version"],
        "report_type": report["report_type"],
        "coverage_targets_passed": report["coverage_targets_passed"],
        "coverage_target_failed_reason_codes": report[
            "coverage_target_failed_reason_codes"
        ],
        "#145_ready_for_rerun": report["#145_ready_for_rerun"],
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        "pre_guard_candidate_count": overall["pre_guard_candidate_count"],
        "guard_compatible_candidate_count": overall[
            "guard_compatible_candidate_count"
        ],
        "guard_compatible_up_entry_count": overall[
            "guard_compatible_up_entry_count"
        ],
        "guard_compatible_down_entry_count": overall[
            "guard_compatible_down_entry_count"
        ],
        "guard_compatible_up_market_count": overall[
            "guard_compatible_up_market_count"
        ],
        "guard_compatible_down_market_count": overall[
            "guard_compatible_down_market_count"
        ],
        "guard_compatible_two_sided_entry_set_exists": overall[
            "guard_compatible_two_sided_entry_set_exists"
        ],
        "validation_guard_compatible_up_entry_count": validation[
            "guard_compatible_up_entry_count"
        ],
        "validation_guard_compatible_down_entry_count": validation[
            "guard_compatible_down_entry_count"
        ],
        "shadow_guard_compatible_up_entry_count": shadow[
            "guard_compatible_up_entry_count"
        ],
        "shadow_guard_compatible_down_entry_count": shadow[
            "guard_compatible_down_entry_count"
        ],
    }


def _split_coverage(
    *,
    split_name: str,
    examples: tuple[PolymarketPolicyExample, ...],
    predictions: tuple[PolymarketPolicyPrediction, ...],
    execution_buffer: float,
    guard_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    prediction_alignment = _prediction_alignment_diagnostics(
        examples=examples,
        predictions=predictions,
        split_name=split_name,
    )
    if not prediction_alignment["prediction_alignment_passed"]:
        raise ValueError(
            f"{split_name} prediction alignment failed: "
            f"{json.dumps(prediction_alignment, sort_keys=True)}"
        )
    _validate_aligned(examples=examples, predictions=predictions, split_name=split_name)
    prediction_set = build_sell_before_close_side_balanced_prediction_set(
        predictions=predictions,
        execution_buffer=execution_buffer,
        guard_thresholds=guard_thresholds,
    )
    rows = [dict(row) for row in prediction_set["side_balance_candidate_entries"]]
    example_by_key = {
        (example.market_id, int(example.decision_ts)): example for example in examples
    }
    prediction_by_key = {
        (prediction.market_id, int(prediction.decision_ts)): prediction
        for prediction in predictions
    }
    candidates_by_side = Counter(row["selected_side"] for row in rows)
    guard_rows = [
        row for row in rows if bool(row.get("side_balance_guard_compatible_entry", False))
    ]
    guard_by_side = Counter(row["selected_side"] for row in guard_rows)
    guard_market_by_side: dict[str, set[str]] = defaultdict(set)
    market_sides: dict[str, set[str]] = defaultdict(set)
    for row in guard_rows:
        side = str(row["selected_side"])
        market_id = str(row["market_id"])
        guard_market_by_side[side].add(market_id)
        market_sides[market_id].add(side)
    positive_by_side = Counter()
    negative_by_side = Counter()
    positive_guard_by_side = Counter()
    negative_guard_by_side = Counter()
    guard_return_by_side: defaultdict[str, float] = defaultdict(float)
    exit_quality_return_by_side: defaultdict[str, float] = defaultdict(float)
    p_up_alignment_return_by_side: defaultdict[str, float] = defaultdict(float)
    exit_quality_positive_by_side = Counter()
    exit_quality_negative_by_side = Counter()
    p_up_alignment_positive_by_side = Counter()
    p_up_alignment_negative_by_side = Counter()
    pass_counts = {
        "exit_reliability_guard_pass_count_by_side": Counter(),
        "p_up_side_alignment_pass_count_by_side": Counter(),
        "liquidity_guard_pass_count_by_side": Counter(),
        "spread_guard_pass_count_by_side": Counter(),
        "staleness_guard_pass_count_by_side": Counter(),
        "queue_fill_guard_pass_count_by_side": Counter(),
    }
    regime_counts: dict[tuple[str, str, str, str], dict[str, int]] = defaultdict(
        lambda: {"candidate_count": 0, "guard_compatible_candidate_count": 0}
    )
    for row in rows:
        side = str(row["selected_side"])
        action = str(row["action"])
        key = (str(row["market_id"]), int(row["decision_ts"]))
        example = example_by_key[key]
        prediction = prediction_by_key[key]
        realized = float(example.action_return_targets.get(action, 0.0))
        if realized > 0.0:
            positive_by_side[side] += 1
        else:
            negative_by_side[side] += 1
        if bool(row.get("side_balance_guard_compatible_entry", False)):
            guard_return_by_side[side] += realized
            if realized > 0.0:
                positive_guard_by_side[side] += 1
            else:
                negative_guard_by_side[side] += 1
        if bool(row.get("exit_reliability_guard_passed", False)):
            pass_counts["exit_reliability_guard_pass_count_by_side"][side] += 1
            exit_quality_return_by_side[side] += realized
            if realized > 0.0:
                exit_quality_positive_by_side[side] += 1
            else:
                exit_quality_negative_by_side[side] += 1
        if bool(row.get("p_up_side_alignment_passed", False)):
            pass_counts["p_up_side_alignment_pass_count_by_side"][side] += 1
            p_up_alignment_return_by_side[side] += realized
            if realized > 0.0:
                p_up_alignment_positive_by_side[side] += 1
            else:
                p_up_alignment_negative_by_side[side] += 1
        reason_codes = set(row.get("side_balance_guard_reason_codes", ()))
        if "entry_blocked_insufficient_executable_bid_notional" not in reason_codes:
            pass_counts["liquidity_guard_pass_count_by_side"][side] += 1
        if "entry_blocked_spread_too_wide" not in reason_codes:
            pass_counts["spread_guard_pass_count_by_side"][side] += 1
        if "entry_blocked_stale_book" not in reason_codes:
            pass_counts["staleness_guard_pass_count_by_side"][side] += 1
        if "entry_blocked_low_queue_fill_probability" not in reason_codes:
            pass_counts["queue_fill_guard_pass_count_by_side"][side] += 1
        regime_key = _regime_key(prediction=prediction, side=side)
        regime_counts[regime_key]["candidate_count"] += 1
        if bool(row.get("side_balance_guard_compatible_entry", False)):
            regime_counts[regime_key]["guard_compatible_candidate_count"] += 1
    side_count = len(guard_by_side)
    total_guard = sum(guard_by_side.values())
    max_guard_side = max(guard_by_side.values(), default=0)
    side_ratio = 0.0 if total_guard == 0 else max_guard_side / total_guard
    candidate_count_by_side = _side_counter_payload(candidates_by_side)
    p_up_alignment_pass_by_side = pass_counts[
        "p_up_side_alignment_pass_count_by_side"
    ]
    p_up_disagreement_by_side = Counter(
        {
            side: int(candidates_by_side.get(side, 0))
            - int(p_up_alignment_pass_by_side.get(side, 0))
            for side in SIDES
        }
    )
    total_candidates = sum(candidates_by_side.values())
    p_up_disagreement_count = sum(p_up_disagreement_by_side.values())
    report = {
        "split_name": split_name,
        "row_count": len(examples),
        "pre_guard_candidate_count": len(rows),
        "candidate_count_by_side": candidate_count_by_side,
        "guard_compatible_candidate_count": total_guard,
        "guard_compatible_candidate_count_by_side": _side_counter_payload(
            guard_by_side
        ),
        "guard_compatible_up_entry_count": int(guard_by_side.get("UP", 0)),
        "guard_compatible_down_entry_count": int(guard_by_side.get("DOWN", 0)),
        "guard_compatible_up_market_count": len(guard_market_by_side.get("UP", set())),
        "guard_compatible_down_market_count": len(
            guard_market_by_side.get("DOWN", set())
        ),
        "guard_compatible_side_count": side_count,
        "guard_compatible_side_entry_ratio": side_ratio,
        "guard_compatible_two_sided_entry_set_exists": side_count >= 2,
        "two_sided_guard_compatible_market_count": sum(
            1 for sides in market_sides.values() if {"UP", "DOWN"} <= sides
        ),
        "estimated_total_label_return": float(sum(guard_return_by_side.values())),
        "estimated_total_label_return_by_side": _side_float_payload(
            guard_return_by_side
        ),
        "p_up_disagreement_count": int(p_up_disagreement_count),
        "p_up_disagreement_count_by_side": _side_counter_payload(
            p_up_disagreement_by_side
        ),
        "p_up_disagreement_rate": (
            0.0 if total_candidates == 0 else p_up_disagreement_count / total_candidates
        ),
        "p_up_disagreement_rate_by_side": _side_rate_payload(
            p_up_disagreement_by_side,
            candidates_by_side,
        ),
        "positive_negative_counts_source": "action_return_targets",
        "positive_label_candidate_count_by_side": _side_counter_payload(
            positive_by_side
        ),
        "negative_label_candidate_count_by_side": _side_counter_payload(
            negative_by_side
        ),
        "positive_guard_compatible_label_candidate_count_by_side": (
            _side_counter_payload(positive_guard_by_side)
        ),
        "negative_guard_compatible_label_candidate_count_by_side": (
            _side_counter_payload(negative_guard_by_side)
        ),
        "exit_quality_only": _subguard_summary(
            pass_counter=pass_counts["exit_reliability_guard_pass_count_by_side"],
            positive_counter=exit_quality_positive_by_side,
            negative_counter=exit_quality_negative_by_side,
            return_by_side=exit_quality_return_by_side,
        ),
        "p_up_alignment_only": _subguard_summary(
            pass_counter=pass_counts["p_up_side_alignment_pass_count_by_side"],
            positive_counter=p_up_alignment_positive_by_side,
            negative_counter=p_up_alignment_negative_by_side,
            return_by_side=p_up_alignment_return_by_side,
        ),
        "regime_rows": _regime_rows(split_name=split_name, regime_counts=regime_counts),
        "round_guard_coverage_rows": _round_guard_coverage_rows(
            split_name=split_name,
            examples=examples,
            candidate_rows=rows,
        ),
        "prediction_alignment": prediction_alignment,
    }
    for field_name, counter in pass_counts.items():
        report[field_name] = _side_counter_payload(counter)
        rate_field = field_name.replace("_count_by_side", "_rate_by_side")
        report[rate_field] = _side_rate_payload(counter, candidates_by_side)
    report["guard_compatible_pass_rate_by_side"] = _side_rate_payload(
        guard_by_side,
        candidates_by_side,
    )
    report["reason_counts"] = dict(
        sorted(
            Counter(
                reason
                for row in rows
                for reason in row.get("side_balance_guard_reason_codes", ())
            ).items()
        )
    )
    return report


def _prediction_alignment_diagnostics(
    *,
    examples: tuple[PolymarketPolicyExample, ...],
    predictions: tuple[PolymarketPolicyPrediction, ...],
    split_name: str,
) -> dict[str, Any]:
    expected_keys = [_example_key(example) for example in examples]
    expected_key_set = set(expected_keys)
    prediction_keys = [_prediction_key(prediction) for prediction in predictions]
    prediction_key_counts = Counter(prediction_keys)
    prediction_key_set = set(prediction_keys)
    missing = sorted(expected_key_set - prediction_key_set)
    unexpected = sorted(prediction_key_set - expected_key_set)
    duplicates = sorted(
        key for key, count in prediction_key_counts.items() if count > 1
    )
    return {
        "split_name": split_name,
        "expected_prediction_count": len(expected_keys),
        "actual_prediction_count": len(prediction_keys),
        "missing_prediction_key_count": len(missing),
        "duplicate_prediction_key_count": len(duplicates),
        "unexpected_prediction_key_count": len(unexpected),
        "missing_prediction_keys": [_key_payload(key) for key in missing[:50]],
        "duplicate_prediction_keys": [_key_payload(key) for key in duplicates[:50]],
        "unexpected_prediction_keys": [_key_payload(key) for key in unexpected[:50]],
        "prediction_alignment_passed": not missing and not duplicates and not unexpected,
    }


def _round_guard_coverage_rows(
    *,
    split_name: str,
    examples: tuple[PolymarketPolicyExample, ...],
    candidate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    slug_by_market_id: dict[str, str] = {}
    rows_by_market_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        slug_by_market_id.setdefault(example.market_id, example.slug)
    for row in candidate_rows:
        rows_by_market_id[str(row["market_id"])].append(row)
    round_rows = []
    for market_id in sorted(slug_by_market_id):
        rows = rows_by_market_id.get(market_id, [])
        guard_rows = [
            row
            for row in rows
            if bool(row.get("side_balance_guard_compatible_entry", False))
        ]
        sides = sorted({str(row["selected_side"]) for row in guard_rows})
        reason_counts = Counter(
            reason
            for row in rows
            for reason in row.get("side_balance_guard_reason_codes", ())
        )
        pre_guard_count = len(rows)
        guard_count = len(guard_rows)
        if pre_guard_count == 0:
            coverage_class = "zero_pre_guard_candidates"
        elif guard_count == 0:
            coverage_class = "pre_guard_but_zero_guard_compatible"
        elif len(sides) == 1:
            coverage_class = "one_sided_guard_compatible"
        else:
            coverage_class = "two_sided_guard_compatible"
        round_rows.append(
            {
                "split_name": split_name,
                "market_id": market_id,
                "slug": slug_by_market_id[market_id],
                "pre_guard_candidate_count": pre_guard_count,
                "guard_compatible_candidate_count": guard_count,
                "guard_compatible_sides": sides,
                "guard_compatible_up_entry_count": sum(
                    1 for row in guard_rows if row["selected_side"] == "UP"
                ),
                "guard_compatible_down_entry_count": sum(
                    1 for row in guard_rows if row["selected_side"] == "DOWN"
                ),
                "coverage_class": coverage_class,
                "top_failure_reason_counts": dict(reason_counts.most_common(10)),
            }
        )
    return round_rows


def _subguard_summary(
    *,
    pass_counter: Counter[str],
    positive_counter: Counter[str],
    negative_counter: Counter[str],
    return_by_side: defaultdict[str, float],
) -> dict[str, Any]:
    count_by_side = _side_counter_payload(pass_counter)
    return {
        "candidate_count": int(sum(count_by_side.values())),
        "candidate_count_by_side": count_by_side,
        "up_entry_count": count_by_side["UP"],
        "down_entry_count": count_by_side["DOWN"],
        "positive_label_candidate_count_by_side": _side_counter_payload(
            positive_counter
        ),
        "negative_label_candidate_count_by_side": _side_counter_payload(
            negative_counter
        ),
        "estimated_total_label_return": float(sum(return_by_side.values())),
        "estimated_total_label_return_by_side": _side_float_payload(return_by_side),
    }


def _coverage_target_results(
    *,
    overall: dict[str, Any],
    by_split: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    checks = [
        (
            "min_guard_compatible_candidate_count",
            overall["guard_compatible_candidate_count"],
            ">=",
        ),
        (
            "min_guard_compatible_up_entry_count",
            overall["guard_compatible_up_entry_count"],
            ">=",
        ),
        (
            "min_guard_compatible_down_entry_count",
            overall["guard_compatible_down_entry_count"],
            ">=",
        ),
        (
            "min_guard_compatible_up_market_count",
            overall["guard_compatible_up_market_count"],
            ">=",
        ),
        (
            "min_guard_compatible_down_market_count",
            overall["guard_compatible_down_market_count"],
            ">=",
        ),
        (
            "two_sided_guard_compatible_entry_set_exists",
            overall["guard_compatible_two_sided_entry_set_exists"],
            "is",
        ),
        (
            "validation_guard_compatible_up_entry_count",
            by_split["validation"]["guard_compatible_up_entry_count"],
            ">=",
        ),
        (
            "validation_guard_compatible_down_entry_count",
            by_split["validation"]["guard_compatible_down_entry_count"],
            ">=",
        ),
        (
            "shadow_guard_compatible_up_entry_count",
            by_split["shadow"]["guard_compatible_up_entry_count"],
            ">=",
        ),
        (
            "shadow_guard_compatible_down_entry_count",
            by_split["shadow"]["guard_compatible_down_entry_count"],
            ">=",
        ),
    ]
    results = []
    for target_name, actual, operator in checks:
        required = COVERAGE_TARGETS[target_name]
        passed = bool(actual == required) if operator == "is" else float(actual) >= float(required)
        results.append(
            {
                "target_name": target_name,
                "actual": actual,
                "required": required,
                "operator": operator,
                "passed": passed,
                "reason_code": f"coverage_target_{target_name}_failed",
            }
        )
    return results


def _coverage_view_report(
    base_report: dict[str, Any],
    *,
    report_type: str,
    metric_fields: tuple[str, ...],
) -> dict[str, Any]:
    base_fields = (
        "pre_guard_candidate_count",
        "guard_compatible_candidate_count",
        "guard_compatible_up_entry_count",
        "guard_compatible_down_entry_count",
        "guard_compatible_up_market_count",
        "guard_compatible_down_market_count",
        "guard_compatible_side_count",
        "guard_compatible_side_entry_ratio",
        "guard_compatible_two_sided_entry_set_exists",
    )
    fields = tuple(dict.fromkeys((*base_fields, *metric_fields)))
    return {
        "schema_version": GUARD_COMPATIBLE_CANDIDATE_COVERAGE_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": base_report["candidate_name"],
        "diagnostic_only": True,
        "report_type": report_type,
        "selection_pool": base_report["selection_pool"],
        "execution_buffer": base_report["execution_buffer"],
        "coverage_targets_passed": base_report["coverage_targets_passed"],
        "coverage_target_failed_reason_codes": base_report[
            "coverage_target_failed_reason_codes"
        ],
        "#145_ready_for_rerun": base_report["#145_ready_for_rerun"],
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        "overall": _select_fields(base_report["overall"], fields),
        "by_split": {
            split_name: _select_fields(split_report, fields)
            for split_name, split_report in base_report["by_split"].items()
        },
        **compact_safety_fields(),
    }


def _round_guard_coverage_report(base_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": GUARD_COMPATIBLE_CANDIDATE_COVERAGE_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": base_report["candidate_name"],
        "diagnostic_only": True,
        "report_type": "round_guard_coverage",
        "selection_pool": base_report["selection_pool"],
        "execution_buffer": base_report["execution_buffer"],
        "coverage_targets_passed": base_report["coverage_targets_passed"],
        "coverage_target_failed_reason_codes": base_report[
            "coverage_target_failed_reason_codes"
        ],
        "#145_ready_for_rerun": base_report["#145_ready_for_rerun"],
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        "prediction_alignment": base_report["prediction_alignment"],
        "overall": _round_summary(base_report["overall"]),
        "by_split": {
            split_name: _round_summary(split_report)
            for split_name, split_report in base_report["by_split"].items()
        },
        **compact_safety_fields(),
    }


def _guard_ablation_coverage_report(
    *,
    dataset: PolymarketPolicyDataset,
    split_inputs: dict[
        str,
        tuple[
            tuple[PolymarketPolicyExample, ...],
            tuple[PolymarketPolicyPrediction, ...],
        ],
    ],
    overall_predictions: tuple[PolymarketPolicyPrediction, ...],
    execution_buffer: float,
    prediction_alignment: dict[str, Any],
) -> dict[str, Any]:
    variants = []
    for spec in GUARD_ABLATION_VARIANTS:
        thresholds = dict(SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_THRESHOLDS)
        thresholds.update(spec["threshold_overrides"])
        by_split = {
            split_name: _split_coverage(
                split_name=split_name,
                examples=examples,
                predictions=predictions,
                execution_buffer=execution_buffer,
                guard_thresholds=thresholds,
            )
            for split_name, (examples, predictions) in split_inputs.items()
        }
        overall = _split_coverage(
            split_name="overall",
            examples=dataset.examples,
            predictions=overall_predictions,
            execution_buffer=execution_buffer,
            guard_thresholds=thresholds,
        )
        target_results = _coverage_target_results(overall=overall, by_split=by_split)
        variants.append(
            _guard_ablation_variant_summary(
                spec=spec,
                thresholds=thresholds,
                overall=overall,
                by_split=by_split,
                target_results=target_results,
            )
        )
    report = {
        "schema_version": GUARD_COMPATIBLE_CANDIDATE_COVERAGE_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
        "diagnostic_only": True,
        "report_type": "guard_ablation_coverage",
        "selection_pool": "guard_compatible_rows",
        "execution_buffer": float(execution_buffer),
        "coverage_targets": dict(COVERAGE_TARGETS),
        "baseline_thresholds": dict(SELL_BEFORE_CLOSE_P_UP_ALIGNED_GUARD_THRESHOLDS),
        "prediction_alignment": prediction_alignment,
        "variant_count": len(variants),
        "variants": variants,
        "#145_ready_for_rerun": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        "interpretation_rules": {
            "p_up_relaxation_adds_profitable_up_candidates": (
                "p_up guard may be too strict or p_up calibration may need review"
            ),
            "p_up_relaxation_adds_unprofitable_up_candidates": (
                "p_up guard is likely protecting the policy from weak ranking"
            ),
            "microstructure_relaxation_adds_candidates": (
                "spread, queue-fill, time-to-close, or liquidity coverage needs review"
            ),
            "no_variant_adds_two_sided_support": "primary blocker is data coverage",
        },
        **compact_safety_fields(),
    }
    _attach_report_id(report, "guard_ablation_coverage_report_id")
    return report


def _guard_ablation_variant_summary(
    *,
    spec: dict[str, Any],
    thresholds: dict[str, float],
    overall: dict[str, Any],
    by_split: dict[str, dict[str, Any]],
    target_results: list[dict[str, Any]],
) -> dict[str, Any]:
    coverage_targets_passed = all(row["passed"] for row in target_results)
    validation = by_split["validation"]
    shadow = by_split["shadow"]
    return {
        "variant_name": spec["variant_name"],
        "description": spec["description"],
        "threshold_overrides": dict(spec["threshold_overrides"]),
        "effective_thresholds": dict(thresholds),
        "guard_compatible_candidate_count": overall[
            "guard_compatible_candidate_count"
        ],
        "guard_compatible_up_entry_count": overall[
            "guard_compatible_up_entry_count"
        ],
        "guard_compatible_down_entry_count": overall[
            "guard_compatible_down_entry_count"
        ],
        "validation_guard_compatible_up_entry_count": validation[
            "guard_compatible_up_entry_count"
        ],
        "validation_guard_compatible_down_entry_count": validation[
            "guard_compatible_down_entry_count"
        ],
        "shadow_guard_compatible_up_entry_count": shadow[
            "guard_compatible_up_entry_count"
        ],
        "shadow_guard_compatible_down_entry_count": shadow[
            "guard_compatible_down_entry_count"
        ],
        "positive_label_candidate_count_by_side": overall[
            "positive_label_candidate_count_by_side"
        ],
        "negative_label_candidate_count_by_side": overall[
            "negative_label_candidate_count_by_side"
        ],
        "positive_guard_compatible_label_candidate_count_by_side": overall[
            "positive_guard_compatible_label_candidate_count_by_side"
        ],
        "negative_guard_compatible_label_candidate_count_by_side": overall[
            "negative_guard_compatible_label_candidate_count_by_side"
        ],
        "estimated_total_label_return": overall["estimated_total_label_return"],
        "estimated_total_label_return_by_side": overall[
            "estimated_total_label_return_by_side"
        ],
        "p_up_disagreement_count": overall["p_up_disagreement_count"],
        "p_up_disagreement_count_by_side": overall[
            "p_up_disagreement_count_by_side"
        ],
        "p_up_disagreement_rate": overall["p_up_disagreement_rate"],
        "p_up_disagreement_rate_by_side": overall[
            "p_up_disagreement_rate_by_side"
        ],
        "coverage_targets_passed": coverage_targets_passed,
        "coverage_target_failed_reason_codes": [
            row["reason_code"] for row in target_results if not row["passed"]
        ],
        "coverage_target_results": target_results,
        "exit_quality_only": overall["exit_quality_only"],
        "p_up_alignment_only": overall["p_up_alignment_only"],
        "by_split": {
            split_name: _guard_ablation_split_summary(split_report)
            for split_name, split_report in by_split.items()
        },
    }


def _guard_ablation_split_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "split_name": report["split_name"],
        "guard_compatible_candidate_count": report[
            "guard_compatible_candidate_count"
        ],
        "guard_compatible_up_entry_count": report[
            "guard_compatible_up_entry_count"
        ],
        "guard_compatible_down_entry_count": report[
            "guard_compatible_down_entry_count"
        ],
        "positive_label_candidate_count_by_side": report[
            "positive_label_candidate_count_by_side"
        ],
        "negative_label_candidate_count_by_side": report[
            "negative_label_candidate_count_by_side"
        ],
        "estimated_total_label_return": report["estimated_total_label_return"],
        "estimated_total_label_return_by_side": report[
            "estimated_total_label_return_by_side"
        ],
        "p_up_disagreement_rate": report["p_up_disagreement_rate"],
        "exit_quality_only": report["exit_quality_only"],
        "p_up_alignment_only": report["p_up_alignment_only"],
    }


def _guard_ablation_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Guard Ablation Coverage",
        "",
        f"- diagnostic_only: `{str(report['diagnostic_only']).lower()}`",
        f"- variant_count: `{report['variant_count']}`",
        f"- #145_ready_for_rerun: `{str(report['#145_ready_for_rerun']).lower()}`",
        f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
        f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
        "",
        "| variant | guard_compatible | up | down | val_up | val_down | shadow_up | shadow_down | label_return | p_up_disagreement | targets_passed | exit_quality | p_up_only |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for variant in report["variants"]:
        lines.append(
            "| {name} | {compatible} | {up} | {down} | {val_up} | {val_down} | "
            "{shadow_up} | {shadow_down} | {label_return:.6f} | {p_up_rate:.6f} | "
            "{targets} | {exit_quality} | {p_up_only} |".format(
                name=variant["variant_name"],
                compatible=variant["guard_compatible_candidate_count"],
                up=variant["guard_compatible_up_entry_count"],
                down=variant["guard_compatible_down_entry_count"],
                val_up=variant["validation_guard_compatible_up_entry_count"],
                val_down=variant["validation_guard_compatible_down_entry_count"],
                shadow_up=variant["shadow_guard_compatible_up_entry_count"],
                shadow_down=variant["shadow_guard_compatible_down_entry_count"],
                label_return=variant["estimated_total_label_return"],
                p_up_rate=variant["p_up_disagreement_rate"],
                targets=str(variant["coverage_targets_passed"]).lower(),
                exit_quality=variant["exit_quality_only"]["candidate_count"],
                p_up_only=variant["p_up_alignment_only"]["candidate_count"],
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


def _round_summary(report: dict[str, Any]) -> dict[str, Any]:
    rows = report["round_guard_coverage_rows"]
    class_counts = Counter(row["coverage_class"] for row in rows)
    reason_counts = Counter()
    for row in rows:
        reason_counts.update(row["top_failure_reason_counts"])
    return {
        "split_name": report["split_name"],
        "round_count": len(rows),
        "rounds_with_zero_pre_guard_candidates": int(
            class_counts.get("zero_pre_guard_candidates", 0)
        ),
        "rounds_with_pre_guard_but_zero_guard_compatible": int(
            class_counts.get("pre_guard_but_zero_guard_compatible", 0)
        ),
        "rounds_with_one_sided_guard_compatible": int(
            class_counts.get("one_sided_guard_compatible", 0)
        ),
        "rounds_with_two_sided_guard_compatible": int(
            class_counts.get("two_sided_guard_compatible", 0)
        ),
        "top_round_failure_reason_counts": dict(reason_counts.most_common(20)),
        "round_guard_coverage_rows": rows,
    }


def _select_fields(report: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {
        "split_name": report["split_name"],
        **{field: report[field] for field in fields if field in report},
    }


def _coverage_markdown_row(split_name: str, report: dict[str, Any]) -> str:
    return (
        "| {split} | {pre} | {compatible} | {up} | {down} | {up_markets} | "
        "{down_markets} | {two_sided} | {ratio:.6f} |"
    ).format(
        split=split_name,
        pre=report["pre_guard_candidate_count"],
        compatible=report["guard_compatible_candidate_count"],
        up=report["guard_compatible_up_entry_count"],
        down=report["guard_compatible_down_entry_count"],
        up_markets=report["guard_compatible_up_market_count"],
        down_markets=report["guard_compatible_down_market_count"],
        two_sided=str(report["guard_compatible_two_sided_entry_set_exists"]).lower(),
        ratio=report["guard_compatible_side_entry_ratio"],
    )


def _round_markdown_row(split_name: str, report: dict[str, Any]) -> str:
    return (
        "| {split} | {zero} | {zero_guard} | {one_sided} | {two_sided} |"
    ).format(
        split=split_name,
        zero=report["rounds_with_zero_pre_guard_candidates"],
        zero_guard=report["rounds_with_pre_guard_but_zero_guard_compatible"],
        one_sided=report["rounds_with_one_sided_guard_compatible"],
        two_sided=report["rounds_with_two_sided_guard_compatible"],
    )


def _regime_key(
    *,
    prediction: PolymarketPolicyPrediction,
    side: str,
) -> tuple[str, str, str, str, str]:
    return (
        side,
        _bucket_notional(_side_feature(prediction.features, side, "executable_bid_notional")),
        _bucket_probability(_side_feature(prediction.features, side, "queue_fill_probability_proxy")),
        _bucket_spread(_side_feature(prediction.features, side, "spread_bps")),
        _bucket_staleness(
            _side_feature(prediction.features, side, "book_staleness_ms")
            or _side_feature(prediction.features, side, "book_update_lag_ms")
        ),
    )


def _regime_rows(
    *,
    split_name: str,
    regime_counts: dict[tuple[str, str, str, str, str], dict[str, int]],
) -> list[dict[str, Any]]:
    rows = []
    for (side, liquidity, queue, spread, staleness), counts in regime_counts.items():
        candidates = int(counts["candidate_count"])
        compatible = int(counts["guard_compatible_candidate_count"])
        rows.append(
            {
                "split_name": split_name,
                "side": side,
                "liquidity_bucket": liquidity,
                "queue_fill_bucket": queue,
                "spread_bucket": spread,
                "staleness_bucket": staleness,
                "candidate_count": candidates,
                "guard_compatible_candidate_count": compatible,
                "guard_compatible_pass_rate": 0.0
                if candidates == 0
                else compatible / candidates,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -int(row["candidate_count"]),
            str(row["liquidity_bucket"]),
            str(row["queue_fill_bucket"]),
            str(row["spread_bucket"]),
            str(row["staleness_bucket"]),
        ),
    )


def _side_feature(features: dict[str, Any], side: str, field: str) -> float | None:
    prefix = "up" if side == "UP" else "down"
    value = features.get(f"{prefix}_{field}")
    return None if value is None else float(value)


def _bucket_notional(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 0.20:
        return "<0.20"
    if value < 1.00:
        return "0.20-1.00"
    return ">=1.00"


def _bucket_probability(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 0.50:
        return "<0.50"
    if value < 0.65:
        return "0.50-0.65"
    if value < 0.80:
        return "0.65-0.80"
    return ">=0.80"


def _bucket_spread(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value <= 300.0:
        return "<=300"
    if value <= 900.0:
        return "300-900"
    return ">900"


def _bucket_staleness(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value <= 1_000.0:
        return "<=1s"
    if value <= 10_000.0:
        return "1s-10s"
    return ">10s"


def _side_counter_payload(counter: Counter[str]) -> dict[str, int]:
    return {side: int(counter.get(side, 0)) for side in SIDES}


def _side_float_payload(counter: defaultdict[str, float]) -> dict[str, float]:
    return {side: float(counter.get(side, 0.0)) for side in SIDES}


def _side_rate_payload(
    numerator: Counter[str],
    denominator: Counter[str],
) -> dict[str, float]:
    return {
        side: 0.0
        if int(denominator.get(side, 0)) == 0
        else int(numerator.get(side, 0)) / int(denominator.get(side, 0))
        for side in SIDES
    }


def _validate_aligned(
    *,
    examples: tuple[PolymarketPolicyExample, ...],
    predictions: tuple[PolymarketPolicyPrediction, ...],
    split_name: str,
) -> None:
    if len(examples) != len(predictions):
        raise ValueError(f"{split_name} coverage examples/predictions length mismatch")
    for example, prediction in zip(examples, predictions, strict=True):
        if (example.market_id, int(example.decision_ts)) != (
            prediction.market_id,
            int(prediction.decision_ts),
        ):
            raise ValueError(f"{split_name} coverage examples/predictions misaligned")


def _attach_report_id(report: dict[str, Any], field_name: str) -> None:
    payload = dict(report)
    payload.pop(field_name, None)
    report[field_name] = canonical_json_sha256(payload)


def _example_key(example: PolymarketPolicyExample) -> tuple[str, int]:
    return (example.market_id, int(example.decision_ts))


def _prediction_key(prediction: PolymarketPolicyPrediction) -> tuple[str, int]:
    return (prediction.market_id, int(prediction.decision_ts))


def _key_payload(key: tuple[str, int]) -> dict[str, Any]:
    return {"market_id": key[0], "decision_ts": key[1]}
