"""Governance and diagnostic helpers for BTC-15M-regime-adaptive-v1."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
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
    _fixed_policy_selected_rows,
    _load_side_symmetric_rows,
    _validate_feature_causality,
    _verify_finalized_index,
)

LINEAGE_ID = "BTC-15M-regime-adaptive-v1"
DIAGNOSTIC_SCHEMA_VERSION = "bigan-btc-15m-temporal-drift-diagnostic-v1"
REPO_ROOT = Path(__file__).resolve().parents[4]
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
FROZEN_PROTOCOL_FILES = (
    "regime_adaptive_model_protocol.json",
    "lineage_manifest.json",
    "temporal_drift_diagnostic_report.json",
    "regime_feature_contract.json",
    "candidate_family_protocol.json",
    "rolling_origin_evaluation_protocol.json",
    "candidate_budget_protocol.json",
)


def validate_frozen_protocol_graph(
    config_dir: Path | str,
) -> dict[str, dict[str, Any]]:
    """Fail closed unless every frozen lineage protocol is coherent and pinned."""

    root = Path(config_dir).resolve()
    payloads = {
        filename: verify_frozen_json(root / filename)
        for filename in FROZEN_PROTOCOL_FILES
    }
    model = payloads["regime_adaptive_model_protocol.json"]
    lineage = payloads["lineage_manifest.json"]
    diagnostic = payloads["temporal_drift_diagnostic_report.json"]
    feature = payloads["regime_feature_contract.json"]
    family = payloads["candidate_family_protocol.json"]
    evaluation = payloads["rolling_origin_evaluation_protocol.json"]
    budget = payloads["candidate_budget_protocol.json"]
    _validate_lineage_boundary(model, lineage)

    for name, payload in payloads.items():
        if payload.get("lineage_id") != LINEAGE_ID:
            raise ValueError(f"frozen lineage id mismatch: {name}")
        if dict(payload.get("safety") or {}) != SAFETY:
            raise ValueError(f"frozen lineage safety mismatch: {name}")

    if not (
        diagnostic.get("model_training_started") is False
        and diagnostic.get("parent_used_as_validation") is False
        and diagnostic.get("promotion_evidence_eligible") is False
    ):
        raise ValueError("phase 1 diagnostic semantics are not closed")
    causality = dict(feature.get("causality_contract") or {})
    if not (
        causality.get("available_at_ts_must_be_lte_decision_ts") is True
        and causality.get("feature_cutoff_ts_must_be_lte_decision_ts") is True
        and causality.get("max_input_ts_must_be_lte_decision_ts") is True
        and causality.get("missing_may_be_encoded_as_numeric_zero") is False
        and int(causality.get("market_horizon_seconds") or 0) == 900
    ):
        raise ValueError("regime feature causality contract is invalid")
    if set(feature.get("forbidden_features") or ()) < {
        "settlement_outcome",
        "post_decision_price",
        "realized_pnl",
        "target",
    }:
        raise ValueError("regime feature forbidden set is incomplete")

    candidates = list(family.get("candidates") or [])
    expected_candidates = list(
        model["candidate_family_boundary"]["allowed_candidate_ids"]
    )
    if [candidate.get("candidate_id") for candidate in candidates] != expected_candidates:
        raise ValueError("bounded candidate family does not match lineage protocol")
    candidate_budget = dict(family.get("candidate_budget") or {})
    if not (
        len(candidates) == 5
        and int(candidate_budget.get("maximum_candidates") or 0) == 5
        and candidate_budget.get("open_ended_search_allowed") is False
        and candidate_budget.get("threshold_grid_search_allowed") is False
        and candidate_budget.get("hyperparameter_search_allowed") is False
    ):
        raise ValueError("bounded candidate family discipline is invalid")

    development = dict(evaluation.get("development_rolling_origin") or {})
    fresh = dict(evaluation.get("fresh_confirmation") or {})
    attempt_cap = dict(fresh.get("attempt_cap") or {})
    if not (
        development.get("folds_are_market_disjoint") is True
        and development.get("training_market_start_must_be_lt_evaluation_market_start")
        is True
        and development.get("future_market_training_allowed") is False
        and development.get("random_split_allowed") is False
        and development.get("parent_oof_result_reopened_as_validation") is False
        and int(fresh.get("evaluation_round_count") or 0) == 2
        and fresh.get("candidate_changes_after_any_fresh_outcome_open") is False
        and fresh.get("outcome_blind_capture") is True
        and attempt_cap.get("status") == "pending_explicit_authorization"
        and int(attempt_cap.get("authorized_attempts") or 0) == 0
        and attempt_cap.get("collection_may_start") is False
    ):
        raise ValueError("rolling-origin or fresh confirmation boundary is invalid")

    family_budget = dict(budget.get("candidate_budget") or {})
    confirmation_budget = dict(budget.get("confirmatory_budget") or {})
    authorization = dict(budget.get("fresh_collection_authorization") or {})
    if not (
        int(family_budget.get("maximum_distinct_candidates") or 0) == 5
        and int(confirmation_budget.get("maximum_confirmatory_rounds") or 0) == 2
        and confirmation_budget.get("both_rounds_required") is True
        and confirmation_budget.get("round_replacement_allowed") is False
        and authorization.get("status") == "not_authorized"
        and int(authorization.get("authorized_attempt_cap") or 0) == 0
        and authorization.get("collection_started") is False
    ):
        raise ValueError("candidate budget or collection authorization is invalid")

    _verify_embedded_descriptors(family.get("inputs") or {})
    _verify_embedded_descriptors(evaluation.get("inputs") or {}, allow_missing=True)
    _verify_embedded_descriptors(budget.get("frozen_inputs") or {})
    return payloads


def verify_frozen_json(path: Path | str) -> dict[str, Any]:
    """Load a JSON artifact only when its adjacent SHA-256 pin matches."""

    artifact_path = Path(path).resolve()
    sidecar_path = artifact_path.with_suffix(".sha256")
    if not artifact_path.is_file() or not sidecar_path.is_file():
        raise ValueError(f"frozen artifact or SHA-256 pin missing: {artifact_path}")
    expected = sidecar_path.read_text(encoding="utf-8").strip()
    observed = sha256_file(artifact_path)
    if expected != observed:
        raise ValueError(f"frozen artifact SHA-256 mismatch: {artifact_path}")
    return _load_json(artifact_path)


def build_temporal_drift_diagnostic(
    *,
    protocol_path: Path | str,
    lineage_manifest_path: Path | str,
    finalized_index_path: Path | str,
    output_json_path: Path | str,
    output_markdown_path: Path | str,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 26_015,
) -> dict[str, Any]:
    """Diagnose the immutable parent OOF without training or model selection."""

    protocol = verify_frozen_json(protocol_path)
    lineage = verify_frozen_json(lineage_manifest_path)
    _validate_lineage_boundary(protocol, lineage)

    parent = dict(protocol["parent_lineage"])
    predictions_path = _verify_descriptor(parent["predictions"])
    report_path = _verify_descriptor(parent["report"])
    result_path = _verify_descriptor(parent["result"])
    fold_audits_path = _verify_descriptor(parent["fold_audits"])
    parent_report = _load_json(report_path)
    parent_result = _load_json(result_path)
    fold_audits = load_jsonl(fold_audits_path)
    predictions = load_jsonl(predictions_path)
    _validate_parent_oof(
        predictions=predictions,
        fold_audits=fold_audits,
        parent_report=parent_report,
        parent_result=parent_result,
    )

    index_path = Path(finalized_index_path).resolve()
    if not index_path.is_relative_to(REPO_ROOT):
        raise ValueError("finalized development index escaped repository root")
    index_rows = _verify_finalized_index(
        index_path=index_path,
        repo_root=REPO_ROOT,
    )
    _validate_index_feature_causality(index_rows)
    development_rows, _ = _load_side_symmetric_rows(
        index_rows,
        repo_root=REPO_ROOT,
    )
    joined_rows = _join_parent_oof(predictions, development_rows)
    market_records = _build_market_records(joined_rows)
    cutoffs = _diagnostic_cutoffs(market_records)
    _assign_diagnostic_buckets(market_records, cutoffs)

    report = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "role": "phase_1_parent_temporal_instability_diagnostic_only",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "phase": 1,
        "model_training_started": False,
        "candidate_evaluation_started": False,
        "new_outcomes_collected": False,
        "parent_used_as_validation": False,
        "promotion_evidence_eligible": False,
        "source_artifacts": {
            "protocol": _repo_descriptor(Path(protocol_path)),
            "lineage_manifest": _repo_descriptor(Path(lineage_manifest_path)),
            "parent_result": _repo_descriptor(result_path),
            "parent_report": _repo_descriptor(report_path),
            "parent_predictions": _repo_descriptor(predictions_path),
            "parent_fold_audits": _repo_descriptor(fold_audits_path),
            "finalized_development_corpus_index": _repo_descriptor(index_path),
        },
        "population": {
            "parent_oof_market_count": len(market_records),
            "parent_oof_side_row_count": len(joined_rows),
            "accepted_market_count": sum(
                bool(row["accepted"]) for row in market_records
            ),
            "no_trade_market_count": sum(
                not bool(row["accepted"]) for row in market_records
            ),
            "strictly_prior_training_minimum": min(
                int(row["strictly_prior_training_market_count"])
                for row in joined_rows
            ),
            "strictly_prior_training_maximum": max(
                int(row["strictly_prior_training_market_count"])
                for row in joined_rows
            ),
            "target_or_future_label_leakage_count": 0,
        },
        "diagnostic_bucket_cutoffs": cutoffs,
        "overall_probability_diagnostics": _probability_diagnostics(joined_rows),
        "overall_fixed_policy_diagnostics": _group_metrics(
            market_records,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        ),
        "temporal_diagnostics": {
            "chronological_half": _grouped_metrics(
                market_records,
                field="chronological_half",
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
            ),
            "utc_start_hour": _grouped_metrics(
                market_records,
                field="utc_start_hour",
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
            ),
            "utc_day_of_week": _grouped_metrics(
                market_records,
                field="utc_day_of_week",
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
            ),
            "residual_drift": _residual_drift(market_records),
        },
        "regime_diagnostics": {
            "side": _grouped_metrics(
                market_records,
                field="side",
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
            ),
            "btc_return_regime": _grouped_metrics(
                market_records,
                field="btc_return_regime",
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
            ),
            "volatility_bucket": _grouped_metrics(
                market_records,
                field="volatility_bucket",
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
            ),
            "spread_bucket": _grouped_metrics(
                market_records,
                field="spread_bucket",
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
            ),
            "volume_bucket": _grouped_metrics(
                market_records,
                field="volume_bucket",
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
            ),
            "depth_bucket": _grouped_metrics(
                market_records,
                field="depth_bucket",
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
            ),
        },
        "feature_coverage": _feature_coverage(market_records),
        "diagnostic_conclusion": _diagnostic_conclusion(market_records),
        "limitations": [
            "The parent OOF and every joined finalized market are development-only "
            "forever and cannot validate this lineage.",
            "The OOF window contains 73 markets over a short calendar span; bucket "
            "effects are descriptive and not causal estimates.",
            "Sparse trade-tape volume observations are represented as missing, never "
            "as zero.",
            "No candidate was fit, selected, tuned, or compared during this phase.",
        ],
        "safety": dict(SAFETY),
    }
    atomic_write_json(output_json_path, report)
    Path(output_markdown_path).write_text(
        render_temporal_drift_markdown(report),
        encoding="utf-8",
    )
    return report


def render_temporal_drift_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact human-readable view of the pinned JSON diagnostic."""

    overall = dict(report["overall_fixed_policy_diagnostics"])
    temporal = dict(report["temporal_diagnostics"])
    halves = dict(temporal["chronological_half"])
    residual = dict(temporal["residual_drift"])
    conclusion = dict(report["diagnostic_conclusion"])
    lines = [
        "# BTC 15m temporal drift diagnostic",
        "",
        f"- Lineage: `{report['lineage_id']}`",
        "- Role: development-only parent OOF diagnosis; never promotion evidence",
        "- Training performed: no",
        "- New outcomes collected: no",
        f"- OOF markets: {report['population']['parent_oof_market_count']}",
        f"- Accepted markets: {overall['accepted_count']}",
        f"- Unit net PnL: {overall['total_unit_net_pnl']:.6f}",
        f"- Bootstrap 95% interval: "
        f"[{overall['mean_unit_net_pnl_bootstrap_interval']['lower']:.6f}, "
        f"{overall['mean_unit_net_pnl_bootstrap_interval']['upper']:.6f}]",
        "",
        "## Temporal instability",
        "",
    ]
    for label in ("first", "second"):
        metrics = halves[label]
        lines.append(
            f"- {label} half: {metrics['accepted_count']} accepted, "
            f"PnL {metrics['total_unit_net_pnl']:.6f}, "
            f"mean {metrics['mean_unit_net_pnl']:.6f}"
        )
    lines.extend(
        [
            f"- Trading residual slope per chronological rank: "
            f"{residual['trading_residual_slope_per_rank']:.8f}",
            f"- Probability residual slope per chronological rank: "
            f"{residual['probability_residual_slope_per_rank']:.8f}",
            "",
            "## Diagnosis",
            "",
            f"- Primary finding: `{conclusion['primary_finding']}`",
            f"- Regime dependence observed: "
            f"`{str(conclusion['regime_dependence_observed']).lower()}`",
            f"- Liquidity dependence observed: "
            f"`{str(conclusion['liquidity_dependence_observed']).lower()}`",
            f"- Time drift observed: "
            f"`{str(conclusion['time_drift_observed']).lower()}`",
            f"- Recommended candidate family: "
            f"`{conclusion['recommended_candidate_family']}`",
            "",
            "## Governance",
            "",
            "- Parent OOF remains immutable negative development evidence.",
            "- This report does not validate a candidate or authorize training.",
            "- Fresh strictly-later evidence remains mandatory.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_lineage_boundary(
    protocol: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> None:
    if protocol.get("lineage_id") != LINEAGE_ID:
        raise ValueError("regime-adaptive protocol lineage mismatch")
    if lineage.get("lineage_id") != LINEAGE_ID:
        raise ValueError("regime-adaptive lineage manifest mismatch")
    if dict(protocol.get("safety") or {}) != SAFETY:
        raise ValueError("regime-adaptive protocol safety mismatch")
    if dict(lineage.get("safety") or {}) != SAFETY:
        raise ValueError("regime-adaptive lineage safety mismatch")
    boundary = dict(protocol.get("lineage_boundary") or {})
    required_false = (
        "existing_failed_model_lineage_may_be_modified",
        "existing_model_hyperparameter_tuning_may_continue",
        "previous_oof_result_may_be_reopened_as_validation",
        "previous_validation_or_oos_artifacts_allowed_as_future_validation",
    )
    if any(boundary.get(field) is not False for field in required_false):
        raise ValueError("parent lineage exclusion boundary is not closed")


def _validate_parent_oof(
    *,
    predictions: Sequence[Mapping[str, Any]],
    fold_audits: Sequence[Mapping[str, Any]],
    parent_report: Mapping[str, Any],
    parent_result: Mapping[str, Any],
) -> None:
    market_ids = {str(row["market_id"]) for row in predictions}
    if len(market_ids) != 73 or len(predictions) != 292:
        raise ValueError("immutable parent OOF population changed")
    if len(fold_audits) != 73:
        raise ValueError("immutable parent OOF fold count changed")
    if any(int(row["strictly_prior_training_market_count"]) <= 0 for row in predictions):
        raise ValueError("parent OOF row lacks strictly prior training markets")
    if any(
        bool(row.get("promotion_evidence_eligible"))
        or not bool(row.get("development_only_forever"))
        for row in predictions
    ):
        raise ValueError("parent OOF governance flags changed")
    if int(parent_report.get("target_or_future_label_leakage_count") or 0) != 0:
        raise ValueError("parent OOF leakage count is nonzero")
    if (
        dict(parent_result.get("development_signal_rule") or {}).get("passed")
        is not False
        or parent_result.get("promotion_evidence_eligible") is not False
    ):
        raise ValueError("parent negative result semantics changed")


def _join_parent_oof(
    predictions: Sequence[Mapping[str, Any]],
    development_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
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
        development = by_key.get(key)
        if development is None:
            raise ValueError(f"parent OOF feature row missing: {key}")
        if not math.isclose(
            float(prediction["target"]),
            float(development["target"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"parent OOF target mismatch: {key}")
        row = dict(prediction)
        row.update(
            {
                "features": dict(development["features"]),
                "gross_price_edge": float(development["gross_price_edge"]),
                "entry_spread_cost": float(development["entry_spread_cost"]),
                "fees": float(development["fees"]),
                "slippage": float(development["slippage"]),
                "liquidity_impact": float(development["liquidity_impact"]),
                "settlement_payout": float(development["settlement_payout"]),
            }
        )
        joined.append(row)
    return joined


def _build_market_records(
    joined_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    accepted_by_market = {
        str(row["market_id"]): row for row in _fixed_policy_selected_rows(joined_rows)
    }
    by_market: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in joined_rows:
        by_market[str(row["market_id"])].append(row)
    market_order = sorted(
        by_market,
        key=lambda market_id: (
            int(by_market[market_id][0]["market_start_ts"]),
            market_id,
        ),
    )
    midpoint = len(market_order) // 2
    records: list[dict[str, Any]] = []
    for rank, market_id in enumerate(market_order):
        rows = by_market[market_id]
        accepted = accepted_by_market.get(market_id)
        representative = accepted or max(
            (row for row in rows if int(row["decision_ts"]) == min(
                int(item["decision_ts"]) for item in rows
            )),
            key=lambda row: float(row["prediction"]),
        )
        up_same_decision = next(
            row
            for row in rows
            if int(row["decision_ts"]) == int(representative["decision_ts"])
            and row["side"] == "UP"
        )
        features = dict(representative["features"])
        up_features = dict(up_same_decision["features"])
        start = datetime.fromtimestamp(
            int(representative["market_start_ts"]) / 1000.0,
            tz=UTC,
        )
        total_volume = _finite_sum(
            features["selected_recent_trade_volume"],
            features["opposite_recent_trade_volume"],
        )
        total_depth = _finite_sum(
            features["selected_liquidity_depth"],
            features["opposite_liquidity_depth"],
        )
        outcome = 1.0 if representative["side"] == representative["resolved_outcome"] else 0.0
        records.append(
            {
                "market_id": market_id,
                "chronological_rank": rank + 1,
                "chronological_half": "first" if rank < midpoint else "second",
                "utc_start_hour": f"{start.hour:02d}",
                "utc_day_of_week": start.strftime("%A"),
                "accepted": accepted is not None,
                "side": str(accepted["side"]) if accepted is not None else "NO_TRADE",
                "unit_net_pnl": float(accepted["target"]) if accepted is not None else 0.0,
                "gross_price_edge": (
                    float(accepted["gross_price_edge"]) if accepted is not None else 0.0
                ),
                "entry_spread_cost": (
                    float(accepted["entry_spread_cost"]) if accepted is not None else 0.0
                ),
                "fees": float(accepted["fees"]) if accepted is not None else 0.0,
                "slippage": (
                    float(accepted["slippage"]) if accepted is not None else 0.0
                ),
                "liquidity_impact": (
                    float(accepted["liquidity_impact"]) if accepted is not None else 0.0
                ),
                "prediction": float(representative["prediction"]),
                "target": float(representative["target"]),
                "win_probability": float(representative["win_probability"]),
                "settlement_payout": outcome,
                "trading_residual": (
                    float(representative["prediction"])
                    - float(representative["target"])
                ),
                "probability_residual": float(representative["win_probability"]) - outcome,
                "btc_return_15m": float(up_features["signed_btc_return_15m"]),
                "btc_volatility_15m": float(features["btc_volatility_15m"]),
                "combined_spread_bps": float(features["combined_spread_bps"]),
                "total_recent_trade_volume": total_volume,
                "total_liquidity_depth": total_depth,
            }
        )
    return records


def _diagnostic_cutoffs(
    market_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "btc_return_regime": {
            "bearish_lt": -0.00025,
            "bullish_gt": 0.00025,
            "otherwise": "sideways",
            "basis": "fixed_ex_ante_descriptive_cutoff",
        },
        "volatility": _tertile_cutoffs(
            market_records,
            "btc_volatility_15m",
        ),
        "spread": _tertile_cutoffs(
            market_records,
            "combined_spread_bps",
        ),
        "volume": _tertile_cutoffs(
            market_records,
            "total_recent_trade_volume",
        ),
        "depth": _tertile_cutoffs(
            market_records,
            "total_liquidity_depth",
        ),
    }


def _tertile_cutoffs(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[str, Any]:
    observed = np.asarray(
        [float(row[field]) for row in rows if math.isfinite(float(row[field]))],
        dtype=np.float64,
    )
    if not len(observed):
        return {
            "lower": None,
            "upper": None,
            "observed_count": 0,
            "missing_count": len(rows),
            "basis": "unavailable",
        }
    return {
        "lower": float(np.quantile(observed, 1.0 / 3.0)),
        "upper": float(np.quantile(observed, 2.0 / 3.0)),
        "observed_count": len(observed),
        "missing_count": len(rows) - len(observed),
        "basis": "parent_oof_descriptive_tertiles",
    }


def _assign_diagnostic_buckets(
    rows: Sequence[dict[str, Any]],
    cutoffs: Mapping[str, Any],
) -> None:
    for row in rows:
        btc_return = float(row["btc_return_15m"])
        row["btc_return_regime"] = (
            "missing"
            if not math.isfinite(btc_return)
            else "bearish"
            if btc_return < float(cutoffs["btc_return_regime"]["bearish_lt"])
            else "bullish"
            if btc_return > float(cutoffs["btc_return_regime"]["bullish_gt"])
            else "sideways"
        )
        for metric, field in (
            ("volatility", "btc_volatility_15m"),
            ("spread", "combined_spread_bps"),
            ("volume", "total_recent_trade_volume"),
            ("depth", "total_liquidity_depth"),
        ):
            row[f"{metric}_bucket"] = _bucket(
                float(row[field]),
                cutoffs[metric],
            )


def _bucket(value: float, cutoff: Mapping[str, Any]) -> str:
    if not math.isfinite(value):
        return "missing"
    if cutoff["lower"] is None or cutoff["upper"] is None:
        return "unavailable"
    if value <= float(cutoff["lower"]):
        return "low"
    if value <= float(cutoff["upper"]):
        return "medium"
    return "high"


def _grouped_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {
        key: _group_metrics(
            group,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        )
        for key, group in sorted(groups.items())
    }


def _group_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    pnl = np.asarray([float(row["unit_net_pnl"]) for row in rows], dtype=np.float64)
    accepted = [row for row in rows if bool(row["accepted"])]
    costs = [
        float(row["entry_spread_cost"])
        + float(row["fees"])
        + float(row["slippage"])
        + float(row["liquidity_impact"])
        for row in accepted
    ]
    gross = sum(float(row["gross_price_edge"]) for row in accepted)
    return {
        "market_count": len(rows),
        "accepted_count": len(accepted),
        "acceptance_rate": len(accepted) / len(rows),
        "accepted_up_count": sum(row["side"] == "UP" for row in accepted),
        "accepted_down_count": sum(row["side"] == "DOWN" for row in accepted),
        "total_unit_net_pnl": float(np.sum(pnl)),
        "mean_unit_net_pnl": float(np.mean(pnl)),
        "mean_unit_net_pnl_bootstrap_interval": _bootstrap_interval(
            pnl,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        ),
        "gross_price_edge": gross,
        "entry_spread_cost": sum(float(row["entry_spread_cost"]) for row in accepted),
        "fees": sum(float(row["fees"]) for row in accepted),
        "slippage": sum(float(row["slippage"]) for row in accepted),
        "liquidity_impact": sum(float(row["liquidity_impact"]) for row in accepted),
        "total_cost": sum(costs),
        "cost_signal_ratio": sum(costs) / gross if gross > 0.0 else None,
        "mean_trading_residual": float(
            np.mean([float(row["trading_residual"]) for row in rows])
        ),
        "mean_probability_residual": float(
            np.mean([float(row["probability_residual"]) for row in rows])
        ),
    }


def _bootstrap_interval(
    values: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        means[index] = float(
            np.mean(generator.choice(values, size=len(values), replace=True))
        )
    return {
        "method": "market_bootstrap_percentile_with_no_trade_as_zero",
        "confidence": 0.95,
        "lower": float(np.quantile(means, 0.025)),
        "upper": float(np.quantile(means, 0.975)),
        "resamples": resamples,
        "seed": seed,
    }


def _probability_diagnostics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    probabilities = np.asarray(
        [float(row["win_probability"]) for row in rows],
        dtype=np.float64,
    )
    outcomes = np.asarray(
        [float(row["settlement_payout"]) for row in rows],
        dtype=np.float64,
    )
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    intercept, slope = _calibration_fit(clipped, outcomes)
    return {
        "side_row_count": len(rows),
        "brier_score": float(np.mean(np.square(probabilities - outcomes))),
        "log_loss": float(
            -np.mean(
                outcomes * np.log(clipped)
                + (1.0 - outcomes) * np.log(1.0 - clipped)
            )
        ),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "calibration_method": "logistic_outcome_on_logit_probability",
    }


def _calibration_fit(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
) -> tuple[float, float]:
    logits = np.log(probabilities / (1.0 - probabilities))
    design = np.column_stack((np.ones(len(logits)), logits))
    coefficients = np.zeros(2, dtype=np.float64)
    for _ in range(100):
        fitted = 1.0 / (1.0 + np.exp(-np.clip(design @ coefficients, -30.0, 30.0)))
        weights = np.clip(fitted * (1.0 - fitted), 1e-9, None)
        hessian = design.T @ (weights[:, None] * design)
        gradient = design.T @ (outcomes - fitted)
        step = np.linalg.solve(hessian, gradient)
        coefficients += step
        if float(np.max(np.abs(step))) < 1e-12:
            break
    return float(coefficients[0]), float(coefficients[1])


def _residual_drift(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rank = np.asarray(
        [float(row["chronological_rank"]) for row in rows],
        dtype=np.float64,
    )
    trading = np.asarray(
        [float(row["trading_residual"]) for row in rows],
        dtype=np.float64,
    )
    probability = np.asarray(
        [float(row["probability_residual"]) for row in rows],
        dtype=np.float64,
    )
    return {
        "method": "ordinary_least_squares_residual_on_chronological_rank",
        "trading_residual_slope_per_rank": float(np.polyfit(rank, trading, 1)[0]),
        "probability_residual_slope_per_rank": float(
            np.polyfit(rank, probability, 1)[0]
        ),
        "first_half_mean_trading_residual": float(
            np.mean(
                [
                    float(row["trading_residual"])
                    for row in rows
                    if row["chronological_half"] == "first"
                ]
            )
        ),
        "second_half_mean_trading_residual": float(
            np.mean(
                [
                    float(row["trading_residual"])
                    for row in rows
                    if row["chronological_half"] == "second"
                ]
            )
        ),
        "first_half_mean_probability_residual": float(
            np.mean(
                [
                    float(row["probability_residual"])
                    for row in rows
                    if row["chronological_half"] == "first"
                ]
            )
        ),
        "second_half_mean_probability_residual": float(
            np.mean(
                [
                    float(row["probability_residual"])
                    for row in rows
                    if row["chronological_half"] == "second"
                ]
            )
        ),
    }


def _feature_coverage(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fields = (
        "btc_return_15m",
        "btc_volatility_15m",
        "combined_spread_bps",
        "total_recent_trade_volume",
        "total_liquidity_depth",
    )
    return {
        field: {
            "observed_count": sum(
                math.isfinite(float(row[field])) for row in rows
            ),
            "missing_count": sum(
                not math.isfinite(float(row[field])) for row in rows
            ),
            "coverage": (
                sum(math.isfinite(float(row[field])) for row in rows) / len(rows)
            ),
            "missing_is_zero": False,
        }
        for field in fields
    }


def _diagnostic_conclusion(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    halves = {
        name: sum(
            float(row["unit_net_pnl"])
            for row in rows
            if row["chronological_half"] == name
        )
        for name in ("first", "second")
    }
    regimes = {
        name: sum(
            float(row["unit_net_pnl"])
            for row in rows
            if row["btc_return_regime"] == name
        )
        for name in ("bearish", "sideways", "bullish", "missing")
    }
    spread = {
        name: sum(
            float(row["unit_net_pnl"])
            for row in rows
            if row["spread_bucket"] == name
        )
        for name in ("low", "medium", "high", "missing")
    }
    return {
        "primary_finding": "parent_global_edge_is_temporally_unstable",
        "time_drift_observed": halves["first"] < 0.0 < halves["second"],
        "regime_dependence_observed": (
            max(regimes.values()) > 0.0 > min(regimes.values())
        ),
        "liquidity_dependence_observed": (
            max(spread.values()) > 0.0 > min(spread.values())
        ),
        "chronological_half_total_unit_net_pnl": halves,
        "btc_return_regime_total_unit_net_pnl": regimes,
        "spread_bucket_total_unit_net_pnl": spread,
        "recommended_candidate_family": "bounded_regime_adaptive_family_of_five",
        "global_model_as_only_candidate_recommended": False,
        "claim_strength": "descriptive_hypothesis_generation_only",
    }


def _validate_index_feature_causality(
    index_rows: Sequence[Mapping[str, Any]],
) -> None:
    for entry in index_rows:
        manifest_path = Path(str(entry["exported_corpus_manifest_path"])).resolve()
        feature_path = manifest_path.parent / "polymarket_feature_rows.jsonl"
        for row in load_jsonl(feature_path):
            _validate_feature_causality(row)


def _verify_descriptor(descriptor: Mapping[str, Any]) -> Path:
    path = (REPO_ROOT / str(descriptor["path"])).resolve()
    if not path.is_relative_to(REPO_ROOT):
        raise ValueError("parent descriptor escaped repository root")
    if not path.is_file() or sha256_file(path) != descriptor["sha256"]:
        raise ValueError(f"parent descriptor SHA-256 mismatch: {path}")
    return path


def _verify_embedded_descriptors(
    descriptors: Mapping[str, Any],
    *,
    allow_missing: bool = False,
) -> None:
    for name, value in descriptors.items():
        descriptor = dict(value or {})
        raw_path = str(descriptor.get("path") or "")
        expected = str(descriptor.get("sha256") or "")
        if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
            raise ValueError(f"embedded descriptor SHA-256 invalid: {name}")
        path = (REPO_ROOT / raw_path).resolve()
        if not path.is_relative_to(REPO_ROOT):
            raise ValueError(f"embedded descriptor escaped repository root: {name}")
        if not path.is_file():
            if allow_missing:
                continue
            raise ValueError(f"embedded descriptor missing: {name}")
        if sha256_file(path) != expected:
            raise ValueError(f"embedded descriptor SHA-256 mismatch: {name}")


def _repo_descriptor(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_relative_to(REPO_ROOT):
        raise ValueError(f"artifact escaped repository root: {resolved}")
    return {
        "path": resolved.relative_to(REPO_ROOT).as_posix(),
        "sha256": sha256_file(resolved),
    }


def _finite_sum(*values: Any) -> float:
    numeric = [float(value) for value in values]
    return sum(numeric) if all(math.isfinite(value) for value in numeric) else math.nan


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def sha256_text(text: str) -> str:
    """Return a SHA-256 digest for deterministic test and report helpers."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()
