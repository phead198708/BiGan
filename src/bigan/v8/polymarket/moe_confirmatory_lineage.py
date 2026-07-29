"""Fail-closed provenance, reconciliation, and attribution for BTC 15m MoE."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from bigan.v8.polymarket.challenge_development_lane import (
    atomic_write_json,
    load_jsonl,
    sha256_file,
)
from bigan.v8.polymarket.challenge_model_15m_training import (
    SIDES,
    _load_side_symmetric_rows,
    _verify_finalized_index,
)
from bigan.v8.polymarket.regime_adaptive_lineage import REPO_ROOT

LINEAGE_ID = "BTC-15M-MoE-confirmatory-v1"
PARENT_LINEAGE_ID = "BTC-15M-regime-adaptive-v1"
RECONCILIATION_SCHEMA_VERSION = (
    "bigan-btc-15m-moe-development-metric-reconciliation-v1"
)
ATTRIBUTION_SCHEMA_VERSION = "bigan-btc-15m-moe-route-attribution-report-v1"
FLOAT_TOLERANCE = 1e-12
SAFETY = {
    "source_model_candidate_eligible": False,
    "freeze_ready": False,
    "promotion_evidence_eligible": False,
    "paper_candidate_allowed": False,
    "v8_execution_handoff_allowed": False,
    "#134_resume_allowed": False,
    "#146_start_allowed": False,
    "live_trading_allowed": False,
    "wallet_signing_allowed": False,
    "polymarket_write_allowed": False,
    "capital_at_risk": False,
}
FORBIDDEN_ROUTER_FIELDS = {
    "settlement_outcome",
    "resolved_outcome",
    "oracle_outcome",
    "target",
    "realized_pnl",
    "future_price",
    "future_return",
    "post_close_resolution",
}


def deterministic_moe_route(
    router_inputs: Mapping[str, Any],
) -> str:
    """Apply the frozen parent router using causal decision-time fields only."""

    prohibited = FORBIDDEN_ROUTER_FIELDS & set(router_inputs)
    if prohibited:
        raise ValueError(
            "outcome or future fields are forbidden from MoE routing: "
            + ", ".join(sorted(prohibited))
        )
    required = {
        "decision_ts",
        "available_at_ts",
        "max_input_ts",
        "volatility_bucket",
        "btc_return_regime",
    }
    missing = required - set(router_inputs)
    if missing:
        raise ValueError(
            "MoE router inputs missing: " + ", ".join(sorted(missing))
        )
    decision_ts = int(router_inputs["decision_ts"])
    if (
        int(router_inputs["available_at_ts"]) > decision_ts
        or int(router_inputs["max_input_ts"]) > decision_ts
    ):
        raise ValueError("MoE router input causality violation")
    return _route(
        str(router_inputs["volatility_bucket"]),
        str(router_inputs["btc_return_regime"]),
    )


def frozen_expert_or_fallback(
    *,
    route: str,
    expert_training_market_count: int,
) -> str:
    """Apply the immutable support-20 expert/fallback boundary."""

    if route not in {"high_vol", "bullish", "bearish", "low_vol"}:
        raise ValueError(f"unknown frozen MoE route: {route}")
    if expert_training_market_count < 0:
        raise ValueError("expert training market count cannot be negative")
    return (
        f"moe_expert_{route}"
        if expert_training_market_count >= 20
        else "global_baseline_fallback"
    )


def reconcile_parent_development_metrics(
    *,
    provenance_attestation_path: Path | str,
    parent_config_dir: Path | str,
    parent_artifact_dir: Path | str,
    development_index_path: Path | str,
    output_json_path: Path | str,
    output_markdown_path: Path | str,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Independently recompute every frozen parent candidate metric."""

    attestation = _load_pinned_json(provenance_attestation_path)
    if attestation.get("lineage_id") != LINEAGE_ID:
        raise ValueError("MoE provenance attestation lineage mismatch")
    parent_config = Path(parent_config_dir).resolve()
    parent_artifacts = Path(parent_artifact_dir).resolve()
    family_path = parent_config / "candidate_family_protocol.json"
    evaluation_path = parent_config / "rolling_origin_evaluation_protocol.json"
    result_path = parent_config / "development_evaluation_result.json"
    report_path = parent_artifacts / "development_evaluation_report.json"
    predictions_path = parent_artifacts / "development_oof_predictions.jsonl"
    folds_path = parent_artifacts / "development_fold_audits.jsonl"
    for path in (
        family_path,
        evaluation_path,
        result_path,
        report_path,
        predictions_path,
        folds_path,
    ):
        _verify_parent_artifact_hash(path, attestation)

    family = _load_json(family_path)
    evaluation = _load_json(evaluation_path)
    committed_result = _load_json(result_path)
    committed_report = _load_json(report_path)
    predictions = load_jsonl(predictions_path)
    folds = load_jsonl(folds_path)
    joined = _join_cost_and_outcome_rows(
        predictions=predictions,
        development_index_path=Path(development_index_path).resolve(),
    )
    _validate_fold_audits(folds, family)
    ordered_oof_markets = _ordered_oof_markets(joined)
    committed_by_candidate = {
        str(candidate["candidate_id"]): candidate
        for candidate in committed_report["candidate_results"]
    }
    result_by_candidate = {
        str(candidate["candidate_id"]): candidate
        for candidate in committed_result["candidate_results"]
    }
    recomputed: list[dict[str, Any]] = []
    all_passed = True
    for candidate in family["candidates"]:
        candidate_id = str(candidate["candidate_id"])
        candidate_rows = [
            row for row in joined if row["candidate_id"] == candidate_id
        ]
        metrics = _independent_candidate_metrics(
            candidate_id=candidate_id,
            candidate_ordinal=int(candidate["ordinal"]),
            rows=candidate_rows,
            ordered_markets=ordered_oof_markets,
            evaluation=evaluation,
        )
        comparison = _compare_candidate_metrics(
            recomputed=metrics,
            committed=committed_by_candidate[candidate_id],
            result_summary=result_by_candidate[candidate_id],
        )
        all_passed = all_passed and bool(comparison["passed"])
        recomputed.append(
            {
                "candidate_id": candidate_id,
                "recomputed": metrics,
                "committed": committed_by_candidate[candidate_id],
                "comparison": comparison,
            }
        )
    parent_selection_unchanged = (
        committed_report["selection"]["selected_candidate_id"] is None
        and committed_result["selection"]["selected_candidate_id"] is None
        and committed_report["selection"]["fresh_collection_allowed"] is False
        and committed_result["selection"]["fresh_collection_allowed"] is False
    )
    all_passed = all_passed and parent_selection_unchanged
    report = {
        "schema_version": RECONCILIATION_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "parent_lineage_id": PARENT_LINEAGE_ID,
        "role": "independent_parent_metric_reconciliation_diagnostic",
        "created_at": created_at or datetime.now(tz=UTC).isoformat(),
        "reconciliation_passed": all_passed,
        "floating_point_tolerance": {
            "absolute": FLOAT_TOLERANCE,
            "relative": 0.0,
            "bootstrap_requires_exact_seed_resamples_and_population": True,
        },
        "inputs": {
            "provenance_attestation": _repo_descriptor(
                Path(provenance_attestation_path)
            ),
            "candidate_family_protocol": _repo_descriptor(family_path),
            "rolling_origin_evaluation_protocol": _repo_descriptor(
                evaluation_path
            ),
            "parent_result": _repo_descriptor(result_path),
            "parent_report": _repo_descriptor(report_path),
            "parent_predictions": _repo_descriptor(predictions_path),
            "parent_fold_audits": _repo_descriptor(folds_path),
            "development_corpus_index": _repo_descriptor(
                Path(development_index_path),
                allow_untracked=True,
            ),
        },
        "population": {
            "candidate_count": len(recomputed),
            "prediction_row_count": len(joined),
            "fold_audit_count": len(folds),
            "oof_market_count": len(ordered_oof_markets),
            "target_or_future_label_leakage_count": 0,
        },
        "candidate_reconciliation": recomputed,
        "parent_selection_reconciliation": {
            "passed": parent_selection_unchanged,
            "selected_candidate_id": None,
            "fresh_collection_allowed": False,
            "candidate_budget_consumed": 5,
            "candidate_budget_maximum": 5,
        },
        "cost_reconciliation": {
            "source": "hash_pinned_normalized_development_label_rows",
            "target_identity": (
                "settlement_payout_minus_entry_mid_minus_entry_spread_cost_"
                "minus_fees_minus_slippage_minus_liquidity_impact"
            ),
            "row_decomposition_mismatch_count": sum(
                not math.isclose(
                    float(row["gross_price_edge"])
                    - float(row["entry_spread_cost"])
                    - float(row["fees"])
                    - float(row["slippage"])
                    - float(row["liquidity_impact"]),
                    float(row["target"]),
                    rel_tol=0.0,
                    abs_tol=FLOAT_TOLERANCE,
                )
                for row in joined
            ),
        },
        "provenance_gate": {
            "attestation_passed": bool(
                attestation["attestation_status"]["passed"]
            ),
            "original_source_commit_reachable": bool(
                attestation["original_source_commit"]["exact_commit_reachable"]
            ),
            "new_candidate_freeze_allowed": False,
        },
        "interpretation": (
            "metrics_reconcile_but_provenance_failure_still_blocks_candidate_freeze"
            if all_passed
            else "metric_reconciliation_failed_and_candidate_freeze_is_blocked"
        ),
        "parent_invariants": dict(attestation["parent_invariants"]),
        "safety": dict(SAFETY),
    }
    atomic_write_json(output_json_path, report)
    _atomic_write_text(
        Path(output_markdown_path),
        render_metric_reconciliation_markdown(report),
    )
    return report


def build_moe_route_attribution(
    *,
    provenance_attestation_path: Path | str,
    reconciliation_report_path: Path | str,
    parent_config_dir: Path | str,
    parent_artifact_dir: Path | str,
    development_index_path: Path | str,
    output_jsonl_path: Path | str,
    output_report_path: Path | str,
    output_markdown_path: Path | str,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Attribute every parent MoE OOF market to its expert or fallback."""

    attestation = _load_pinned_json(provenance_attestation_path)
    reconciliation = _load_pinned_json(reconciliation_report_path)
    if reconciliation.get("reconciliation_passed") is not True:
        raise ValueError("MoE attribution requires passed metric reconciliation")
    parent_config = Path(parent_config_dir).resolve()
    parent_artifacts = Path(parent_artifact_dir).resolve()
    family = _load_json(parent_config / "candidate_family_protocol.json")
    feature_contract = _load_json(parent_config / "regime_feature_contract.json")
    predictions_path = parent_artifacts / "development_oof_predictions.jsonl"
    folds_path = parent_artifacts / "development_fold_audits.jsonl"
    report_path = parent_artifacts / "development_evaluation_report.json"
    predictions = load_jsonl(predictions_path)
    joined = _join_cost_and_outcome_rows(
        predictions=predictions,
        development_index_path=Path(development_index_path).resolve(),
    )
    folds = load_jsonl(folds_path)
    moe_rows = [
        row for row in joined if row["candidate_id"] == "mixture_of_experts"
    ]
    moe_folds = {
        str(row["target_market_id"]): row
        for row in folds
        if row["candidate_id"] == "mixture_of_experts"
    }
    support_by_target = _expert_support_by_target(
        development_index_path=Path(development_index_path).resolve(),
        feature_contract=feature_contract,
    )
    ordered_markets = _ordered_oof_markets(moe_rows)
    midpoint = len(ordered_markets) // 2
    half_by_market = {
        market_id: "first" if index < midpoint else "second"
        for index, (_, market_id) in enumerate(ordered_markets)
    }
    selected_by_market = {
        str(row["market_id"]): row for row in _selected_rows(moe_rows)
    }
    by_market: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in moe_rows:
        by_market[str(row["market_id"])].append(row)
    attribution: list[dict[str, Any]] = []
    for _, market_id in ordered_markets:
        rows = by_market[market_id]
        selected = selected_by_market.get(market_id)
        representative = selected or _no_trade_representative(rows)
        route = str(representative["expert_route"])
        fold = moe_folds[market_id]
        fallback_used = bool(fold[f"expert_{route}_fallback"])
        training_count = support_by_target[(market_id, route)]
        if fallback_used != (training_count < 20):
            raise ValueError(
                f"MoE expert support and frozen fallback disagree: {market_id}"
            )
        attribution.append(
            {
                "lineage_id": LINEAGE_ID,
                "parent_lineage_id": PARENT_LINEAGE_ID,
                "market_id": market_id,
                "decision_ts": int(representative["decision_ts"]),
                "router_input_values": {
                    "btc_return_15m": representative["unsigned_btc_return_15m"],
                    "btc_return_regime": representative["btc_return_regime"],
                    "btc_volatility_15m": representative["btc_volatility_15m"],
                    "volatility_bucket": representative["volatility_bucket"],
                },
                "assigned_route": route,
                "requested_expert": route,
                "expert_training_market_count": training_count,
                "expert_available": not fallback_used,
                "fallback_used": fallback_used,
                "actual_model_used": (
                    "global_baseline_fallback"
                    if fallback_used
                    else f"moe_expert_{route}"
                ),
                "selected_side": (
                    str(selected["side"]) if selected is not None else None
                ),
                "accepted": selected is not None,
                "unit_net_pnl": (
                    float(selected["target"]) if selected is not None else 0.0
                ),
                "costs": {
                    "gross_price_edge": (
                        float(selected["gross_price_edge"])
                        if selected is not None
                        else 0.0
                    ),
                    "entry_spread_cost": (
                        float(selected["entry_spread_cost"])
                        if selected is not None
                        else 0.0
                    ),
                    "fees": (
                        float(selected["fees"]) if selected is not None else 0.0
                    ),
                    "slippage": (
                        float(selected["slippage"])
                        if selected is not None
                        else 0.0
                    ),
                    "liquidity_impact": (
                        float(selected["liquidity_impact"])
                        if selected is not None
                        else 0.0
                    ),
                },
                "chronological_half": half_by_market[market_id],
                "provider_missingness": {
                    "provider_health_score": representative[
                        "provider_health_score"
                    ],
                    "trade_volume_missing": representative[
                        "trade_volume_missing"
                    ],
                    "depth_missing": representative["depth_missing"],
                    "spread_missing": representative["spread_missing"],
                    "chainlink_reference_missing": representative[
                        "chainlink_reference_missing"
                    ],
                    "feature_complete": representative["feature_complete"],
                },
                "development_only_forever": True,
                "promotion_evidence_eligible": False,
                "safety": dict(SAFETY),
            }
        )
    _atomic_write_jsonl(Path(output_jsonl_path), attribution)
    parent_report = _load_json(report_path)
    parent_moe = next(
        candidate
        for candidate in parent_report["candidate_results"]
        if candidate["candidate_id"] == "mixture_of_experts"
    )
    report = _attribution_report(
        rows=attribution,
        parent_moe=parent_moe,
        family=family,
        feature_contract=feature_contract,
        created_at=created_at or datetime.now(tz=UTC).isoformat(),
        inputs={
            "provenance_attestation": _repo_descriptor(
                Path(provenance_attestation_path)
            ),
            "metric_reconciliation": _repo_descriptor(
                Path(reconciliation_report_path)
            ),
            "parent_predictions": _repo_descriptor(predictions_path),
            "parent_fold_audits": _repo_descriptor(folds_path),
            "development_corpus_index": _repo_descriptor(
                Path(development_index_path),
                allow_untracked=True,
            ),
            "attribution_rows": _repo_descriptor(Path(output_jsonl_path)),
        },
        provenance_passed=bool(attestation["attestation_status"]["passed"]),
    )
    atomic_write_json(output_report_path, report)
    _atomic_write_text(
        Path(output_markdown_path),
        render_moe_attribution_markdown(report),
    )
    return report


def render_metric_reconciliation_markdown(
    report: Mapping[str, Any],
) -> str:
    """Render the independent metric reconciliation."""

    lines = [
        "# BTC 15m MoE parent metric reconciliation",
        "",
        f"- Reconciliation passed: "
        f"`{str(report['reconciliation_passed']).lower()}`",
        f"- Prediction rows: {report['population']['prediction_row_count']}",
        f"- Fold audits: {report['population']['fold_audit_count']}",
        f"- OOF markets: {report['population']['oof_market_count']}",
        f"- Floating tolerance: {report['floating_point_tolerance']['absolute']}",
        "",
        "| Candidate | Accepted | PnL | 95% LCB | Metric match | Gate match |",
        "|---|---:|---:|---:|:---:|:---:|",
    ]
    for item in report["candidate_reconciliation"]:
        metrics = item["recomputed"]["trading_metrics"]
        lines.append(
            f"| {item['candidate_id']} | "
            f"{metrics['accepted_market_count']} | "
            f"{metrics['total_unit_net_pnl']:.6f} | "
            f"{metrics['mean_unit_net_pnl_bootstrap_interval']['lower']:.6f} | "
            f"{'yes' if item['comparison']['metric_payload_match'] else 'no'} | "
            f"{'yes' if item['comparison']['gate_payload_match'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Metric reconciliation does not repair the unresolvable recorded source "
            "commit. Candidate freeze remains blocked by provenance.",
            "",
        ]
    )
    return "\n".join(lines)


def render_moe_attribution_markdown(report: Mapping[str, Any]) -> str:
    """Render the expert and fallback attribution summary."""

    lines = [
        "# BTC 15m MoE route and fallback attribution",
        "",
        f"- Attribution reconciled: "
        f"`{str(report['attribution_reconciliation_passed']).lower()}`",
        f"- Markets: {report['market_count']}",
        f"- Accepted: {report['accepted_market_count']}",
        f"- Total PnL: {report['total_unit_net_pnl']:.6f}",
        f"- Fallback share: {report['fallback']['overall_share']:.6f}",
        f"- Native expert PnL: {report['pnl_attribution']['native_expert_pnl']:.6f}",
        f"- Global fallback PnL: "
        f"{report['pnl_attribution']['global_fallback_pnl']:.6f}",
        "",
        "| Route | Markets | Fallback | PnL |",
        "|---|---:|---:|---:|",
    ]
    for route, count in report["route_counts"].items():
        lines.append(
            f"| {route} | {count} | "
            f"{report['fallback']['count_by_requested_expert'].get(route, 0)} | "
            f"{report['pnl_attribution']['pnl_by_route'].get(route, 0.0):.6f} |"
        )
    lines.extend(
        [
            "",
            "This is diagnostic development attribution only. No router, expert, "
            "filter, support threshold, or fallback behavior was changed.",
            "",
        ]
    )
    return "\n".join(lines)


def assert_metric_payload_matches(
    recomputed: Mapping[str, Any],
    committed: Mapping[str, Any],
    *,
    path: str = "root",
) -> None:
    """Fail closed when a committed metric payload is tampered or drifts."""

    mismatches: list[str] = []
    _compare_values(recomputed, committed, path=path, mismatches=mismatches)
    if mismatches:
        raise ValueError(
            "development metric reconciliation mismatch: " + ", ".join(mismatches)
        )


def _independent_candidate_metrics(
    *,
    candidate_id: str,
    candidate_ordinal: int,
    rows: Sequence[Mapping[str, Any]],
    ordered_markets: Sequence[tuple[int, str]],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    selected = _selected_rows(rows)
    selected_by_market = {str(row["market_id"]): row for row in selected}
    market_pnl = [
        float(selected_by_market[market_id]["target"])
        if market_id in selected_by_market
        else 0.0
        for _, market_id in ordered_markets
    ]
    bootstrap = evaluation["trading_metrics"]["bootstrap"]
    interval = _bootstrap_interval(
        market_pnl,
        resamples=int(bootstrap["resamples"]),
        seed=int(bootstrap["seed"]),
    )
    midpoint = len(ordered_markets) // 2
    first_ids = {market_id for _, market_id in ordered_markets[:midpoint]}
    second_ids = {market_id for _, market_id in ordered_markets[midpoint:]}
    gross = sum(float(row["gross_price_edge"]) for row in selected)
    spread = sum(float(row["entry_spread_cost"]) for row in selected)
    fees = sum(float(row["fees"]) for row in selected)
    slippage = sum(float(row["slippage"]) for row in selected)
    impact = sum(float(row["liquidity_impact"]) for row in selected)
    total_cost = spread + fees + slippage + impact
    largest_removed = (
        sum(market_pnl) - max(market_pnl) if market_pnl else 0.0
    )
    trading = {
        "market_count": len(ordered_markets),
        "accepted_market_count": len(selected),
        "acceptance_rate": len(selected) / len(ordered_markets),
        "accepted_up_count": sum(row["side"] == "UP" for row in selected),
        "accepted_down_count": sum(row["side"] == "DOWN" for row in selected),
        "total_unit_net_pnl": sum(market_pnl),
        "mean_unit_net_pnl": float(np.mean(market_pnl)),
        "mean_unit_net_pnl_bootstrap_interval": interval,
        "largest_winner_removed_total_unit_net_pnl": largest_removed,
        "first_chronological_half_total_unit_net_pnl": sum(
            float(row["target"])
            for row in selected
            if row["market_id"] in first_ids
        ),
        "second_chronological_half_total_unit_net_pnl": sum(
            float(row["target"])
            for row in selected
            if row["market_id"] in second_ids
        ),
        "gross_price_edge": gross,
        "entry_spread_cost": spread,
        "fees": fees,
        "slippage": slippage,
        "liquidity_impact": impact,
        "total_cost": total_cost,
        "cost_signal_ratio": total_cost / gross if gross > 0.0 else None,
    }
    probability = _probability_metrics(rows)
    stratified = {
        field: _stratified_metrics(selected, field)
        for field in (
            "btc_return_regime",
            "side",
            "volatility_bucket",
            "spread_bucket",
            "volume_bucket",
            "depth_bucket",
        )
    }
    thresholds = evaluation["development_candidate_selection_rule"][
        "eligibility_requirements"
    ]
    gates = {
        "target_or_future_label_leakage": True,
        "minimum_accepted_market_count": (
            trading["accepted_market_count"]
            >= int(thresholds["minimum_accepted_market_count"])
        ),
        "minimum_acceptance_rate": (
            trading["acceptance_rate"]
            >= float(thresholds["minimum_acceptance_rate"])
        ),
        "bootstrap_lcb": (
            interval["lower"]
            > float(
                thresholds[
                    "mean_unit_net_pnl_bootstrap_95pct_lower_must_be_gt"
                ]
            )
        ),
        "largest_winner_removed": (
            largest_removed
            > float(
                thresholds[
                    "largest_winner_removed_total_unit_net_pnl_must_be_gt"
                ]
            )
        ),
        "first_chronological_half": (
            trading["first_chronological_half_total_unit_net_pnl"]
            >= float(
                thresholds[
                    "first_chronological_half_total_unit_net_pnl_must_be_gte"
                ]
            )
        ),
        "second_chronological_half": (
            trading["second_chronological_half_total_unit_net_pnl"]
            >= float(
                thresholds[
                    "second_chronological_half_total_unit_net_pnl_must_be_gte"
                ]
            )
        ),
    }
    return {
        "candidate_id": candidate_id,
        "candidate_ordinal": candidate_ordinal,
        "target_or_future_label_leakage_count": 0,
        "probability_metrics": probability,
        "trading_metrics": trading,
        "stratified_diagnostics": stratified,
        "development_selection_gate_results": gates,
        "development_selection_eligible": all(gates.values()),
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
    }


def _compare_candidate_metrics(
    *,
    recomputed: Mapping[str, Any],
    committed: Mapping[str, Any],
    result_summary: Mapping[str, Any],
) -> dict[str, Any]:
    metric_mismatches: list[str] = []
    gate_mismatches: list[str] = []
    for field in ("probability_metrics", "trading_metrics", "stratified_diagnostics"):
        _compare_values(
            recomputed[field],
            committed[field],
            path=field,
            mismatches=metric_mismatches,
        )
    for field in (
        "development_selection_gate_results",
        "development_selection_eligible",
        "target_or_future_label_leakage_count",
    ):
        _compare_values(
            recomputed[field],
            committed[field],
            path=field,
            mismatches=gate_mismatches,
        )
    summary_mismatches: list[str] = []
    summary_mapping = {
        "accepted_market_count": recomputed["trading_metrics"][
            "accepted_market_count"
        ],
        "total_unit_net_pnl": recomputed["trading_metrics"][
            "total_unit_net_pnl"
        ],
        "mean_unit_net_pnl_bootstrap_95pct_lower": recomputed[
            "trading_metrics"
        ]["mean_unit_net_pnl_bootstrap_interval"]["lower"],
        "largest_winner_removed_total_unit_net_pnl": recomputed[
            "trading_metrics"
        ]["largest_winner_removed_total_unit_net_pnl"],
        "first_chronological_half_total_unit_net_pnl": recomputed[
            "trading_metrics"
        ]["first_chronological_half_total_unit_net_pnl"],
        "second_chronological_half_total_unit_net_pnl": recomputed[
            "trading_metrics"
        ]["second_chronological_half_total_unit_net_pnl"],
        "development_selection_eligible": recomputed[
            "development_selection_eligible"
        ],
    }
    _compare_values(
        summary_mapping,
        {
            field: result_summary[field]
            for field in summary_mapping
        },
        path="result_summary",
        mismatches=summary_mismatches,
    )
    passed = not metric_mismatches and not gate_mismatches and not summary_mismatches
    return {
        "passed": passed,
        "metric_payload_match": not metric_mismatches,
        "gate_payload_match": not gate_mismatches,
        "result_summary_match": not summary_mismatches,
        "metric_mismatches": metric_mismatches,
        "gate_mismatches": gate_mismatches,
        "result_summary_mismatches": summary_mismatches,
    }


def _compare_values(
    left: Any,
    right: Any,
    *,
    path: str,
    mismatches: list[str],
) -> None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            mismatches.append(f"{path}.keys")
            return
        for key in left:
            _compare_values(
                left[key],
                right[key],
                path=f"{path}.{key}",
                mismatches=mismatches,
            )
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            mismatches.append(f"{path}.length")
            return
        for index, (left_item, right_item) in enumerate(
            zip(left, right, strict=True)
        ):
            _compare_values(
                left_item,
                right_item,
                path=f"{path}[{index}]",
                mismatches=mismatches,
            )
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if isinstance(left, bool) or isinstance(right, bool):
            if left is not right:
                mismatches.append(path)
            return
        if not math.isclose(
            float(left),
            float(right),
            rel_tol=0.0,
            abs_tol=FLOAT_TOLERANCE,
        ):
            mismatches.append(path)
        return
    if left != right:
        mismatches.append(path)


def _join_cost_and_outcome_rows(
    *,
    predictions: Sequence[Mapping[str, Any]],
    development_index_path: Path,
) -> list[dict[str, Any]]:
    index_rows = _verify_finalized_index(
        index_path=development_index_path,
        repo_root=REPO_ROOT,
    )
    development_rows, _ = _load_side_symmetric_rows(
        index_rows,
        repo_root=REPO_ROOT,
    )
    by_key = {
        (str(row["market_id"]), int(row["decision_ts"]), str(row["side"])): row
        for row in development_rows
    }
    joined: list[dict[str, Any]] = []
    for prediction in predictions:
        key = (
            str(prediction["market_id"]),
            int(prediction["decision_ts"]),
            str(prediction["side"]),
        )
        source = by_key.get(key)
        if source is None:
            raise ValueError(f"development cost row missing: {key}")
        if not math.isclose(
            float(prediction["target"]),
            float(source["target"]),
            rel_tol=0.0,
            abs_tol=FLOAT_TOLERANCE,
        ):
            raise ValueError(f"development target mismatch: {key}")
        features = dict(source["features"])
        unsigned_return = float(features["signed_btc_return_15m"])
        if source["side"] == "DOWN":
            unsigned_return = -unsigned_return
        joined.append(
            {
                **dict(prediction),
                "settlement_payout": float(source["settlement_payout"]),
                "gross_price_edge": float(source["gross_price_edge"]),
                "entry_spread_cost": float(source["entry_spread_cost"]),
                "fees": float(source["fees"]),
                "slippage": float(source["slippage"]),
                "liquidity_impact": float(source["liquidity_impact"]),
                "unsigned_btc_return_15m": unsigned_return,
                "btc_volatility_15m": float(features["btc_volatility_15m"]),
                "provider_health_score": float(
                    features["provider_health_score"]
                ),
                "trade_volume_missing": not (
                    math.isfinite(float(features["selected_recent_trade_volume"]))
                    and math.isfinite(
                        float(features["opposite_recent_trade_volume"])
                    )
                ),
                "depth_missing": not (
                    math.isfinite(float(features["selected_liquidity_depth"]))
                    and math.isfinite(
                        float(features["opposite_liquidity_depth"])
                    )
                ),
                "spread_missing": not math.isfinite(
                    float(features["combined_spread_bps"])
                ),
                "chainlink_reference_missing": not math.isfinite(
                    float(features["signed_chainlink_reference_distance"])
                ),
                "feature_complete": all(
                    math.isfinite(float(features[name]))
                    for name in (
                        "selected_recent_trade_volume",
                        "opposite_recent_trade_volume",
                        "selected_liquidity_depth",
                        "opposite_liquidity_depth",
                        "combined_spread_bps",
                        "signed_chainlink_reference_distance",
                    )
                ),
                "expert_route": _route(
                    str(prediction["volatility_bucket"]),
                    str(prediction["btc_return_regime"]),
                ),
            }
        )
    return joined


def _route(volatility_bucket: str, btc_return_regime: str) -> str:
    if volatility_bucket == "high":
        return "high_vol"
    if btc_return_regime == "bullish":
        return "bullish"
    if btc_return_regime == "bearish":
        return "bearish"
    return "low_vol"


def _expert_support_by_target(
    *,
    development_index_path: Path,
    feature_contract: Mapping[str, Any],
) -> dict[tuple[str, str], int]:
    index_rows = _verify_finalized_index(
        index_path=development_index_path,
        repo_root=REPO_ROOT,
    )
    development_rows, _ = _load_side_symmetric_rows(
        index_rows,
        repo_root=REPO_ROOT,
    )
    market_order = sorted(
        {
            (int(row["market_start_ts"]), str(row["market_id"]))
            for row in development_rows
        }
    )
    routes_by_market: dict[str, set[str]] = defaultdict(set)
    for row in development_rows:
        routes_by_market[str(row["market_id"])].add(
            _raw_row_route(row, feature_contract)
        )
    support: dict[tuple[str, str], int] = {}
    routes = ("high_vol", "bullish", "bearish", "low_vol")
    for target_index, (_, target_market_id) in enumerate(market_order):
        prior_ids = [market_id for _, market_id in market_order[:target_index]]
        for route in routes:
            support[(target_market_id, route)] = sum(
                route in routes_by_market[market_id] for market_id in prior_ids
            )
    return support


def _raw_row_route(
    row: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
) -> str:
    features = dict(row["features"])
    signed_return = float(features["signed_btc_return_15m"])
    unsigned_return = signed_return if row["side"] == "UP" else -signed_return
    regimes = feature_contract["derived_regime_features"]
    return_contract = regimes["btc_return_regime"]
    volatility_contract = regimes["volatility_bucket"]
    volatility = float(features["btc_volatility_15m"])
    volatility_bucket = (
        "missing"
        if not math.isfinite(volatility)
        else "low"
        if volatility <= float(volatility_contract["low_if_lte"])
        else "medium"
        if volatility <= float(volatility_contract["medium_if_lte"])
        else "high"
    )
    return_regime = (
        "missing"
        if not math.isfinite(unsigned_return)
        else "bearish"
        if unsigned_return < float(return_contract["bearish_if_lt"])
        else "bullish"
        if unsigned_return > float(return_contract["bullish_if_gt"])
        else "sideways"
    )
    return _route(volatility_bucket, return_regime)


def _selected_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    by_market: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_market[str(row["market_id"])].append(row)
    selected: list[Mapping[str, Any]] = []
    for market_rows in by_market.values():
        by_decision: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in market_rows:
            by_decision[int(row["decision_ts"])].append(row)
        for decision_ts in sorted(by_decision):
            candidate = max(
                by_decision[decision_ts],
                key=lambda row: (
                    float(row["selection_score"]),
                    -SIDES.index(str(row["side"])),
                ),
            )
            if float(candidate["selection_score"]) > 0.0:
                selected.append(candidate)
                break
    return selected


def _no_trade_representative(
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    earliest = min(int(row["decision_ts"]) for row in rows)
    return max(
        (row for row in rows if int(row["decision_ts"]) == earliest),
        key=lambda row: (
            float(row["selection_score"]),
            -SIDES.index(str(row["side"])),
        ),
    )


def _ordered_oof_markets(
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[int, str]]:
    return sorted(
        {
            (int(row["market_start_ts"]), str(row["market_id"]))
            for row in rows
        }
    )


def _validate_fold_audits(
    folds: Sequence[Mapping[str, Any]],
    family: Mapping[str, Any],
) -> None:
    candidate_ids = {
        str(candidate["candidate_id"]) for candidate in family["candidates"]
    }
    if len(folds) != 5 * 73:
        raise ValueError("parent fold audit population changed")
    if {str(fold["candidate_id"]) for fold in folds} != candidate_ids:
        raise ValueError("parent fold audit candidate set changed")
    for fold in folds:
        if not (
            fold.get("target_market_used_for_fit") is False
            and fold.get("future_market_used_for_fit") is False
            and int(fold.get("target_or_future_label_leakage_count") or 0) == 0
            and fold.get("promotion_evidence_eligible") is False
        ):
            raise ValueError("parent fold audit leakage or governance mismatch")


def _bootstrap_interval(
    values: Sequence[float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    population = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        means[index] = float(
            np.mean(
                generator.choice(
                    population,
                    size=len(population),
                    replace=True,
                )
            )
        )
    return {
        "method": "market_bootstrap_percentile_with_NO_TRADE_as_zero",
        "confidence": 0.95,
        "lower": float(np.quantile(means, 0.025)),
        "upper": float(np.quantile(means, 0.975)),
        "resamples": resamples,
        "seed": seed,
    }


def _probability_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    probabilities = np.clip(
        np.asarray(
            [float(row["win_probability"]) for row in rows],
            dtype=np.float64,
        ),
        1e-12,
        1.0 - 1e-12,
    )
    outcomes = np.asarray(
        [float(row["settlement_payout"]) for row in rows],
        dtype=np.float64,
    )
    intercept, slope = _calibration_fit(probabilities, outcomes)
    return {
        "side_row_count": len(rows),
        "brier_score": float(np.mean(np.square(probabilities - outcomes))),
        "log_loss": float(
            -np.mean(
                outcomes * np.log(probabilities)
                + (1.0 - outcomes) * np.log(1.0 - probabilities)
            )
        ),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def _calibration_fit(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
) -> tuple[float, float]:
    logits = np.log(probabilities / (1.0 - probabilities))
    design = np.column_stack((np.ones(len(logits)), logits))
    coefficients = np.zeros(2, dtype=np.float64)
    for _ in range(100):
        fitted = 1.0 / (
            1.0 + np.exp(-np.clip(design @ coefficients, -30.0, 30.0))
        )
        weights = np.clip(fitted * (1.0 - fitted), 1e-9, None)
        hessian = design.T @ (weights[:, None] * design)
        gradient = design.T @ (outcomes - fitted)
        step = np.linalg.solve(hessian, gradient)
        coefficients += step
        if float(np.max(np.abs(step))) < 1e-12:
            break
    return float(coefficients[0]), float(coefficients[1])


def _stratified_metrics(
    selected: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        groups[str(row[field])].append(row)
    return {
        group: {
            "accepted_market_count": len(group_rows),
            "total_unit_net_pnl": sum(
                float(row["target"]) for row in group_rows
            ),
            "mean_unit_net_pnl": float(
                np.mean([float(row["target"]) for row in group_rows])
            ),
        }
        for group, group_rows in sorted(groups.items())
    }


def _attribution_report(
    *,
    rows: Sequence[Mapping[str, Any]],
    parent_moe: Mapping[str, Any],
    family: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    created_at: str,
    inputs: Mapping[str, Any],
    provenance_passed: bool,
) -> dict[str, Any]:
    accepted = [row for row in rows if row["accepted"]]
    native = [row for row in accepted if not row["fallback_used"]]
    fallback = [row for row in accepted if row["fallback_used"]]
    route_counts = Counter(str(row["assigned_route"]) for row in rows)
    fallback_counts = Counter(
        str(row["requested_expert"]) for row in rows if row["fallback_used"]
    )
    pnl_by_route = {
        route: sum(
            float(row["unit_net_pnl"])
            for row in rows
            if row["assigned_route"] == route
        )
        for route in sorted(route_counts)
    }
    half_pnl = {
        half: sum(
            float(row["unit_net_pnl"])
            for row in rows
            if row["chronological_half"] == half
        )
        for half in ("first", "second")
    }
    side_counts_by_route = {
        route: dict(
            Counter(
                str(row["selected_side"])
                for row in accepted
                if row["assigned_route"] == route
            )
        )
        for route in sorted(route_counts)
    }
    regime_counts_by_route = {
        route: dict(
            Counter(
                str(row["router_input_values"]["btc_return_regime"])
                for row in rows
                if row["assigned_route"] == route
            )
        )
        for route in sorted(route_counts)
    }
    support_evolution = {
        route: {
            "minimum": min(
                int(row["expert_training_market_count"])
                for row in rows
                if row["assigned_route"] == route
            ),
            "maximum": max(
                int(row["expert_training_market_count"])
                for row in rows
                if row["assigned_route"] == route
            ),
            "first": next(
                int(row["expert_training_market_count"])
                for row in rows
                if row["assigned_route"] == route
            ),
            "last": next(
                int(row["expert_training_market_count"])
                for row in reversed(rows)
                if row["assigned_route"] == route
            ),
        }
        for route in sorted(route_counts)
    }
    largest = max(accepted, key=lambda row: float(row["unit_net_pnl"]))
    complete = [
        row
        for row in accepted
        if row["provider_missingness"]["feature_complete"]
    ]
    incomplete = [
        row
        for row in accepted
        if not row["provider_missingness"]["feature_complete"]
    ]
    total_pnl = sum(float(row["unit_net_pnl"]) for row in rows)
    parent_metrics = parent_moe["trading_metrics"]
    reconciliations = {
        "market_count": len(rows) == int(parent_metrics["market_count"]),
        "accepted_market_count": len(accepted)
        == int(parent_metrics["accepted_market_count"]),
        "total_unit_net_pnl": math.isclose(
            total_pnl,
            float(parent_metrics["total_unit_net_pnl"]),
            rel_tol=0.0,
            abs_tol=FLOAT_TOLERANCE,
        ),
        "native_plus_fallback_pnl": math.isclose(
            sum(float(row["unit_net_pnl"]) for row in native)
            + sum(float(row["unit_net_pnl"]) for row in fallback),
            total_pnl,
            rel_tol=0.0,
            abs_tol=FLOAT_TOLERANCE,
        ),
        "up_plus_down_count": sum(
            row["selected_side"] == "UP" for row in accepted
        )
        + sum(row["selected_side"] == "DOWN" for row in accepted)
        == len(accepted),
        "chronological_halves_pnl": math.isclose(
            half_pnl["first"] + half_pnl["second"],
            total_pnl,
            rel_tol=0.0,
            abs_tol=FLOAT_TOLERANCE,
        ),
        "missingness_partition": len(complete) + len(incomplete) == len(accepted),
    }
    return {
        "schema_version": ATTRIBUTION_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "parent_lineage_id": PARENT_LINEAGE_ID,
        "role": "development_only_moe_route_and_fallback_diagnostic",
        "created_at": created_at,
        "inputs": dict(inputs),
        "router": {
            "precedence": family["candidates"][2]["router_precedence"],
            "expert_ids": family["candidates"][2]["expert_ids"],
            "minimum_expert_training_markets": family["candidates"][2][
                "minimum_expert_training_markets"
            ],
            "fallback": family["candidates"][2][
                "fallback_when_expert_markets_insufficient"
            ],
            "feature_contract_sha256": sha256_file(
                REPO_ROOT
                / "examples/v8/polymarket_configs/"
                "BTC-15M-regime-adaptive-v1/regime_feature_contract.json"
            ),
            "router_or_threshold_change_performed": False,
        },
        "market_count": len(rows),
        "accepted_market_count": len(accepted),
        "total_unit_net_pnl": total_pnl,
        "route_counts": dict(sorted(route_counts.items())),
        "fallback": {
            "count": sum(row["fallback_used"] for row in rows),
            "overall_share": sum(row["fallback_used"] for row in rows) / len(rows),
            "count_by_requested_expert": dict(sorted(fallback_counts.items())),
            "share_by_chronological_half": {
                half: (
                    sum(
                        row["fallback_used"]
                        for row in rows
                        if row["chronological_half"] == half
                    )
                    / sum(
                        row["chronological_half"] == half for row in rows
                    )
                )
                for half in ("first", "second")
            },
        },
        "pnl_attribution": {
            "pnl_by_route": pnl_by_route,
            "native_expert_pnl": sum(
                float(row["unit_net_pnl"]) for row in native
            ),
            "global_fallback_pnl": sum(
                float(row["unit_net_pnl"]) for row in fallback
            ),
            "chronological_half_pnl": half_pnl,
        },
        "support_evolution_by_route": support_evolution,
        "side_distribution_by_route": side_counts_by_route,
        "regime_distribution_by_route": regime_counts_by_route,
        "largest_winner_attribution": {
            "market_id": largest["market_id"],
            "unit_net_pnl": largest["unit_net_pnl"],
            "assigned_route": largest["assigned_route"],
            "fallback_used": largest["fallback_used"],
            "actual_model_used": largest["actual_model_used"],
            "selected_side": largest["selected_side"],
        },
        "provider_and_missingness": {
            "provider_health_score": _numeric_summary(
                [
                    float(row["provider_missingness"]["provider_health_score"])
                    for row in rows
                ]
            ),
            "trade_volume_missing_count": sum(
                row["provider_missingness"]["trade_volume_missing"] for row in rows
            ),
            "depth_missing_count": sum(
                row["provider_missingness"]["depth_missing"] for row in rows
            ),
            "spread_missing_count": sum(
                row["provider_missingness"]["spread_missing"] for row in rows
            ),
            "chainlink_reference_missing_count": sum(
                row["provider_missingness"]["chainlink_reference_missing"]
                for row in rows
            ),
            "accepted_complete_feature_market_count": len(complete),
            "accepted_missing_feature_market_count": len(incomplete),
            "complete_feature_pnl": sum(
                float(row["unit_net_pnl"]) for row in complete
            ),
            "missing_feature_pnl": sum(
                float(row["unit_net_pnl"]) for row in incomplete
            ),
            "by_model_use": {
                model_use: {
                    "market_count": len(group),
                    "trade_volume_missing_count": sum(
                        row["provider_missingness"]["trade_volume_missing"]
                        for row in group
                    ),
                    "feature_complete_count": sum(
                        row["provider_missingness"]["feature_complete"]
                        for row in group
                    ),
                }
                for model_use, group in _group_rows(
                    rows,
                    field="actual_model_used",
                ).items()
            },
        },
        "reconciliation_checks": reconciliations,
        "attribution_reconciliation_passed": all(reconciliations.values()),
        "provenance_attestation_passed": provenance_passed,
        "new_candidate_freeze_allowed": False,
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "safety": dict(SAFETY),
    }


def _group_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return dict(sorted(groups.items()))


def _numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    finite = [value for value in values if math.isfinite(value)]
    return {
        "observed_count": len(finite),
        "missing_count": len(values) - len(finite),
        "minimum": min(finite) if finite else None,
        "maximum": max(finite) if finite else None,
        "mean": float(np.mean(finite)) if finite else None,
    }


def _verify_parent_artifact_hash(
    path: Path,
    attestation: Mapping[str, Any],
) -> None:
    expected_hashes = set(attestation["result_artifact_hashes"].values())
    expected_hashes.update(attestation["frozen_protocol_hashes"].values())
    observed = sha256_file(path)
    if observed not in expected_hashes:
        raise ValueError(f"parent artifact hash is not attested: {path}")


def _load_pinned_json(path: Path | str) -> dict[str, Any]:
    artifact = Path(path).resolve()
    sidecar = artifact.with_suffix(".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != sha256_file(
        artifact
    ):
        raise ValueError(f"pinned JSON SHA-256 mismatch: {artifact}")
    return _load_json(artifact)


def _repo_descriptor(
    path: Path,
    *,
    allow_untracked: bool = False,
) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_relative_to(REPO_ROOT):
        raise ValueError(f"artifact escaped repository root: {resolved}")
    return {
        "path": resolved.relative_to(REPO_ROOT).as_posix(),
        "sha256": sha256_file(resolved),
        "repository_tracked": not allow_untracked,
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _atomic_write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    _atomic_write_text(
        path,
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
