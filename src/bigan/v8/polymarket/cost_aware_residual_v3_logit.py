"""Final logit-offset challenger for BTC 15m residual lineage v3."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.cost_aware_residual import _public_dataset_row
from bigan.v8.polymarket.cost_aware_residual_v3 import (
    DEFAULT_CONFIG_DIR,
    LINEAGE_ID,
    _build_report,
    _descriptor,
    _load_frozen_development_rows,
    _load_json,
    _looks_like_git_sha,
    _market_results_from_predictions,
    _verify_descriptor,
    _write_new_frozen_json,
    _write_new_frozen_text,
    _write_new_jsonl,
    render_v3_markdown,
    validate_residual_v3_protocol,
)
from bigan.v8.polymarket.cost_aware_residual_v3 import (
    DEFAULT_PROTOCOL as PRIMARY_PROTOCOL,
)
from bigan.v8.polymarket.cost_aware_residual_v3 import (
    FOLD_SCHEMA_VERSION as PRIMARY_FOLD_SCHEMA_VERSION,
)
from bigan.v8.polymarket.moe_collection_observability import FEATURE_NAMES
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.regime_adaptive_lineage import REPO_ROOT

CHALLENGER_PROTOCOL_SCHEMA_VERSION = (
    "bigan-btc-15m-logit-offset-residual-oof-protocol-v3"
)
PREDICTION_SCHEMA_VERSION = "bigan-btc-15m-logit-offset-residual-oof-prediction-v3"
FOLD_SCHEMA_VERSION = "bigan-btc-15m-logit-offset-residual-oof-fold-v3"
REPORT_SCHEMA_VERSION = "bigan-btc-15m-logit-offset-residual-oof-report-v3"
MANIFEST_SCHEMA_VERSION = "bigan-btc-15m-logit-offset-residual-oof-manifest-v3"
DATASET_SCHEMA_VERSION = "bigan-btc-15m-logit-offset-residual-dataset-row-v3"

DEFAULT_CHALLENGER_PROTOCOL = (
    DEFAULT_CONFIG_DIR / "residual_v3_challenger_slot_002_protocol.json"
)
DEFAULT_CHALLENGER_OUTPUT_DIR = (
    DEFAULT_CONFIG_DIR / "residual_v3_challenger_slot_002_oof"
)

STRUCTURAL_CHANGE = {
    "changed_component": "probability_link_and_training_likelihood",
    "from": (
        "probability_scale_squared_error_residual_with_fixed_causal_interactions_"
        "and_exponential_recency_weights"
    ),
    "to": (
        "binomial_log_likelihood_with_decision_time_market_logit_offset_"
        "unchanged_source_features_and_equal_training_weights"
    ),
    "reason": (
        "slot_1_failed_two_chronological_block_gates_and_required_3576_markets;_"
        "the_bounded_logit_link_targets_probability_calibration_and_temporal_"
        "stability_without_post_hoc_filtering"
    ),
    "expected_mechanism": (
        "using logit(selected_mid) as a per-row offset constrains the learner to a "
        "multiplicative odds correction, while binomial likelihood keeps predictions "
        "bounded and the frozen cost subtraction makes marginal edges NO_TRADE at the "
        "unchanged zero threshold"
    ),
    "threshold_changed": False,
    "cost_baseline_population_or_gate_changed": False,
    "route_side_missingness_or_outlier_filter_added": False,
    "parameter_or_threshold_search_performed": False,
}


def validate_logit_challenger_protocol(
    payload: Mapping[str, Any],
    *,
    repository_root: Path | str = REPO_ROOT,
    verify_artifacts: bool = True,
) -> None:
    """Validate the second and final v3 candidate with shared contracts frozen."""

    blockers: list[str] = []
    if payload.get("schema_version") != CHALLENGER_PROTOCOL_SCHEMA_VERSION:
        blockers.append("schema_version")
    if payload.get("lineage_id") != LINEAGE_ID:
        blockers.append("lineage_id")
    if payload.get("slot_id") != "residual-v3-challenger-slot-002":
        blockers.append("slot_id")
    if payload.get("candidate_role") != "challenger":
        blockers.append("candidate_role")
    if dict(payload.get("candidate_budget") or {}) != {
        "maximum_total_slots": 2,
        "this_slot_ordinal": 2,
        "slots_consumed_before_run": 1,
        "slots_remaining_after_run": 0,
        "slot_budget_may_be_increased": False,
    }:
        blockers.append("candidate_budget")
    if dict(payload.get("target") or {}) != _logit_target_contract():
        blockers.append("target")
    if dict(payload.get("feature_contract") or {}) != _logit_feature_contract():
        blockers.append("feature_contract")
    if dict(payload.get("temporal_adaptation") or {}) != _equal_weight_contract():
        blockers.append("temporal_adaptation")
    if dict(payload.get("model") or {}) != _logit_model_contract():
        blockers.append("model")
    if dict(payload.get("structural_change") or {}) != STRUCTURAL_CHANGE:
        blockers.append("structural_change")
    prior = dict(payload.get("prior_slot_result") or {})
    if not (
        set(prior) == {"manifest", "report", "failed_gates"}
        and prior.get("failed_gates")
        == [
            "every_chronological_block_candidate_total_gte_zero",
            "every_chronological_block_paired_delta_total_gte_zero",
            "prospective_power_required_market_count_lte_2000",
        ]
    ):
        blockers.append("prior_slot_result")
    root = Path(repository_root).resolve()
    if verify_artifacts and not blockers:
        try:
            for field in ("manifest", "report"):
                _verify_descriptor(dict(prior[field]), repository_root=root)
            _verify_descriptor(
                dict(payload["inputs"]["candidate_implementation"]),
                repository_root=root,
            )
        except (KeyError, OSError, TypeError, ValueError):
            blockers.append("challenger_artifact_binding")

    normalized = deepcopy(dict(payload))
    primary = _load_json(Path(PRIMARY_PROTOCOL))
    for field in (
        "schema_version",
        "slot_id",
        "candidate_role",
        "candidate_budget",
        "target",
        "feature_contract",
        "temporal_adaptation",
        "model",
    ):
        normalized[field] = deepcopy(primary[field])
    normalized["inputs"]["candidate_implementation"] = primary["inputs"][
        "candidate_implementation"
    ]
    normalized.pop("prior_slot_result", None)
    normalized.pop("structural_change", None)
    try:
        validate_residual_v3_protocol(
            normalized,
            repository_root=root,
            verify_artifacts=verify_artifacts,
        )
    except ValueError as error:
        blockers.append(f"shared_protocol:{error}")
    if blockers:
        raise ValueError(
            "residual v3 logit challenger protocol invalid: " + ", ".join(blockers)
        )


def run_logit_challenger_oof(
    *,
    protocol_path: Path | str = DEFAULT_CHALLENGER_PROTOCOL,
    expected_protocol_sha256: str,
    output_dir: Path | str = DEFAULT_CHALLENGER_OUTPUT_DIR,
    source_commit: str,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Execute the second and final preregistered v3 candidate exactly once."""

    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    output = Path(output_dir).resolve()
    if not protocol_file.is_relative_to(root) or not output.is_relative_to(root):
        raise ValueError("logit challenger paths must remain repository-local")
    if sha256_file(protocol_file) != expected_protocol_sha256.lower():
        raise ValueError("logit challenger protocol SHA-256 mismatch")
    sidecar = protocol_file.with_suffix(".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != (
        expected_protocol_sha256
    ):
        raise ValueError("logit challenger protocol is not SHA-frozen")
    if not _looks_like_git_sha(source_commit):
        raise ValueError("source_commit must be a full lowercase Git SHA")
    protocol = _load_json(protocol_file)
    validate_logit_challenger_protocol(protocol, repository_root=root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"logit challenger output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    dataset_rows, baseline_by_market, population_order = _load_frozen_development_rows(
        protocol=protocol,
        repository_root=root,
    )
    predictions, folds = _rolling_origin_logit_predict(
        rows=dataset_rows,
        population_order=population_order,
        protocol=protocol,
    )
    markets = _market_results_from_predictions(
        predictions=predictions,
        baseline_by_market=baseline_by_market,
        population_order=population_order,
        protocol=protocol,
    )
    report = _build_logit_report(
        protocol=protocol,
        protocol_sha256=expected_protocol_sha256,
        source_commit=source_commit,
        market_results=markets,
        fold_audits=folds,
    )

    dataset_path = output / "residual_v3_logit_development_dataset_rows.jsonl"
    prediction_path = output / "residual_v3_logit_oof_predictions.jsonl"
    fold_path = output / "residual_v3_logit_oof_fold_audits.jsonl"
    market_path = output / "residual_v3_logit_oof_market_results.jsonl"
    report_path = output / "residual_v3_logit_oof_report.json"
    markdown_path = output / "residual_v3_logit_oof_report.md"
    _write_new_jsonl(dataset_path, [_public_logit_dataset_row(row) for row in dataset_rows])
    _write_new_jsonl(prediction_path, predictions)
    _write_new_jsonl(fold_path, folds)
    _write_new_jsonl(market_path, markets)
    report_artifact = _write_new_frozen_json(report_path, report)
    markdown_artifact = _write_new_frozen_text(
        markdown_path, render_logit_challenger_markdown(report)
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "slot_id": protocol["slot_id"],
        "candidate_role": "challenger",
        "created_at": protocol["created_at"],
        "source_commit": source_commit,
        "protocol": _descriptor(protocol_file, root),
        "prior_slot_result": dict(protocol["prior_slot_result"]),
        "candidate_implementation": dict(protocol["inputs"]["candidate_implementation"]),
        "immutable_gate_implementation": dict(protocol["inputs"]["gate_implementation"]),
        "artifacts": {
            "dataset_rows": _descriptor(dataset_path, root),
            "predictions": _descriptor(prediction_path, root),
            "fold_audits": _descriptor(fold_path, root),
            "market_results": _descriptor(market_path, root),
            "report": _descriptor(Path(report_artifact["path"]), root),
            "report_markdown": _descriptor(Path(markdown_artifact["path"]), root),
        },
        "evaluation_executed_exactly_once": True,
        "candidate_budget_exhausted": True,
        "additional_candidate_allowed": False,
        "candidate_freeze_allowed": report["all_gates_passed"],
        "next_stage_authorization_required_even_if_all_gates_pass": True,
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "safety": dict(SAFETY),
    }
    manifest_artifact = _write_new_frozen_json(
        output / "residual_v3_logit_oof_manifest.json", manifest
    )
    return {
        "manifest": _descriptor(Path(manifest_artifact["path"]), root),
        "report": _descriptor(Path(report_artifact["path"]), root),
        "all_gates_passed": report["all_gates_passed"],
        "failed_gates": report["failed_gates"],
        "candidate_budget_exhausted": True,
        "oof_market_count": len(markets),
        "next_stage_authorization_required": True,
        "safety": dict(SAFETY),
    }


def logit_offset_action_values(
    rows: Sequence[Mapping[str, Any]],
    probabilities: Sequence[float],
) -> list[dict[str, float]]:
    """Normalize bounded binomial probabilities by pair, then subtract frozen cost."""

    if len(rows) != len(probabilities):
        raise ValueError("logit challenger prediction row count mismatch")
    grouped: dict[tuple[str, int], list[tuple[int, Mapping[str, Any], float]]] = (
        defaultdict(list)
    )
    for index, (row, probability) in enumerate(
        zip(rows, probabilities, strict=True)
    ):
        value = float(probability)
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError("logit challenger probability is invalid")
        grouped[(str(row["market_id"]), int(row["decision_ts"]))].append(
            (index, row, value)
        )
    output: list[dict[str, float] | None] = [None] * len(rows)
    for key, members in grouped.items():
        if sorted(str(item[1]["side"]) for item in members) != ["DOWN", "UP"]:
            raise ValueError(f"logit challenger UP/DOWN pair is incomplete: {key}")
        denominator = sum(item[2] for item in members)
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("logit challenger pair denominator is invalid")
        for index, row, raw_probability in members:
            anchor = _anchor_probability(row)
            probability = raw_probability / denominator
            cost = dict(row["cost_decomposition"])
            entry_ask = float(cost["entry_ask"])
            non_entry_cost = float(cost["total_cost_excluding_entry_ask"])
            action_value = probability - entry_ask - non_entry_cost
            if not all(
                math.isfinite(value)
                for value in (probability, entry_ask, non_entry_cost, action_value)
            ):
                raise ValueError("logit challenger action value is invalid")
            output[index] = {
                "market_anchor_probability": anchor,
                "market_anchor_logit": _logit(anchor),
                "predicted_probability_before_pair_normalization": raw_probability,
                "predicted_probability": probability,
                "entry_ask": entry_ask,
                "non_entry_cost": non_entry_cost,
                "action_value": action_value,
            }
    if any(item is None for item in output):
        raise ValueError("logit challenger action output is incomplete")
    return [dict(item) for item in output if item is not None]


def _rolling_origin_logit_predict(
    *,
    rows: Sequence[Mapping[str, Any]],
    population_order: Sequence[str],
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows_by_market: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_market[str(row["market_id"])].append(row)
    rolling = dict(protocol["rolling_origin"])
    initial = int(rolling["initial_training_market_count"])
    block_size = int(rolling["target_block_size"])
    block_count = int(rolling["target_block_count"])
    parameters = dict(protocol["model"]["parameters"])
    boost_rounds = int(protocol["model"]["fixed_num_boost_round"])
    predictions: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for block_index in range(block_count):
        target_start = initial + block_index * block_size
        target_end = target_start + block_size
        training_ids = list(population_order[:target_start])
        target_ids = list(population_order[target_start:target_end])
        if len(target_ids) != block_size:
            raise ValueError("logit challenger target block population mismatch")
        train_rows = [row for market_id in training_ids for row in rows_by_market[market_id]]
        target_rows = [row for market_id in target_ids for row in rows_by_market[market_id]]
        labels = [_binary_payout_label(row) for row in train_rows]
        train_matrix = _dmatrix(train_rows, labels=labels)
        target_matrix = _dmatrix(target_rows, labels=None)
        booster = xgb.train(
            params=parameters,
            dtrain=train_matrix,
            num_boost_round=boost_rounds,
            verbose_eval=False,
        )
        raw_probabilities = [float(value) for value in booster.predict(target_matrix)]
        actions = logit_offset_action_values(target_rows, raw_probabilities)
        for row, action in zip(target_rows, actions, strict=True):
            predictions.append(
                {
                    "schema_version": PREDICTION_SCHEMA_VERSION,
                    "lineage_id": LINEAGE_ID,
                    "slot_id": protocol["slot_id"],
                    "market_id": row["market_id"],
                    "market_start_ts": row["market_start_ts"],
                    "decision_ts": row["decision_ts"],
                    "side": row["side"],
                    "prediction": action["action_value"],
                    "market_anchor_probability": action["market_anchor_probability"],
                    "market_anchor_logit": action["market_anchor_logit"],
                    "predicted_probability_before_pair_normalization": action[
                        "predicted_probability_before_pair_normalization"
                    ],
                    "predicted_probability": action["predicted_probability"],
                    "realized_unit_net_pnl_if_action": row["target"],
                    "resolved_outcome": row["resolved_outcome"],
                    "cost_decomposition": row["cost_decomposition"],
                    "feature_row_sha256": row["feature_row_sha256"],
                    "chronological_block": block_index + 1,
                    "strictly_prior_training_market_count": len(training_ids),
                    "target_or_future_label_used_for_fit": False,
                    "development_only_forever": True,
                    "promotion_evidence_eligible": False,
                    "safety": dict(SAFETY),
                }
            )
        audits.append(
            {
                "schema_version": FOLD_SCHEMA_VERSION,
                "lineage_id": LINEAGE_ID,
                "slot_id": protocol["slot_id"],
                "chronological_block": block_index + 1,
                "strictly_prior_training_market_count": len(training_ids),
                "target_market_count": len(target_ids),
                "training_market_ids_sha256": canonical_json_sha256(training_ids),
                "target_market_ids_sha256": canonical_json_sha256(target_ids),
                "training_binary_labels_sha256": canonical_json_sha256(labels),
                "training_base_margins_sha256": canonical_json_sha256(
                    [_logit(_anchor_probability(row)) for row in train_rows]
                ),
                "last_training_market_position": target_start,
                "first_target_market_position": target_start + 1,
                "target_or_future_label_leakage_count": 0,
                "fixed_num_boost_round": boost_rounds,
                "model_parameters_sha256": canonical_json_sha256(parameters),
                "base_margin": "logit(decision_time_selected_mid)",
                "pair_coherence_applied_before_cost_subtraction": True,
                "development_only_forever": True,
                "promotion_evidence_eligible": False,
                "safety": dict(SAFETY),
            }
        )
    return predictions, audits


def _dmatrix(
    rows: Sequence[Mapping[str, Any]], *, labels: Sequence[float] | None
) -> xgb.DMatrix:
    values = np.vstack([np.asarray(row["features"], dtype=float) for row in rows])
    label_values = np.asarray(labels, dtype=float) if labels is not None else None
    base_margin = np.asarray(
        [_logit(_anchor_probability(row)) for row in rows], dtype=float
    )
    return xgb.DMatrix(
        values,
        label=label_values,
        base_margin=base_margin,
        feature_names=list(FEATURE_NAMES),
        missing=np.nan,
    )


def _anchor_probability(row: Mapping[str, Any]) -> float:
    anchor = float(
        np.asarray(row["features"], dtype=float)[FEATURE_NAMES.index("selected_mid")]
    )
    if not math.isfinite(anchor) or not 0.0 < anchor < 1.0:
        raise ValueError("logit challenger selected_mid anchor is invalid")
    return anchor


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def _binary_payout_label(row: Mapping[str, Any]) -> float:
    side = str(row["side"])
    outcome = str(row["resolved_outcome"])
    if side not in {"UP", "DOWN"} or outcome not in {"UP", "DOWN"}:
        raise ValueError("logit challenger side or outcome is invalid")
    payout = 1.0 if side == outcome else 0.0
    cost = dict(row["cost_decomposition"])
    expected_target = (
        payout
        - float(cost["entry_ask"])
        - float(cost["total_cost_excluding_entry_ask"])
    )
    if not math.isclose(
        expected_target, float(row["target"]), rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("logit challenger target/cost reconciliation failed")
    return payout


def _build_logit_report(
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    source_commit: str,
    market_results: Sequence[Mapping[str, Any]],
    fold_audits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    report = _build_report(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        source_commit=source_commit,
        market_results=market_results,
        fold_audits=_primary_schema_folds(fold_audits),
    )
    report["schema_version"] = REPORT_SCHEMA_VERSION
    report["candidate_role"] = "challenger"
    report["architecture_type"] = (
        "pooled_side_symmetric_binomial_model_with_decision_time_market_logit_"
        "offset_and_deterministic_pair_coherence"
    )
    report["candidate_budget_exhausted"] = True
    report["remaining_candidate_slots"] = 0
    report["additional_candidate_allowed"] = False
    report["structural_change"] = dict(protocol["structural_change"])
    return report


def render_logit_challenger_markdown(report: Mapping[str, Any]) -> str:
    base = render_v3_markdown(report).replace(
        "# BTC 15m causal time-adaptive residual v3 primary slot 001",
        "# BTC 15m logit-offset residual v3 challenger slot 002",
        1,
    ).rstrip()
    return (
        base
        + "\n\n## Logit-offset challenger and candidate budget\n\n"
        + "- Training likelihood: fixed binary log loss.\n"
        + "- Per-row base margin: `logit(decision_time_selected_mid)`.\n"
        + "- Source feature bytes and native NaN semantics reused: `True`\n"
        + "- Acceptance threshold changed from zero: `False`\n"
        + "- Second and final candidate slot consumed: `True`\n"
        + "- Additional candidate allowed in this lineage: `False`\n"
    )


def _public_logit_dataset_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = _public_dataset_row(row)
    output["schema_version"] = DATASET_SCHEMA_VERSION
    output["lineage_id"] = LINEAGE_ID
    output["market_anchor_probability"] = _anchor_probability(row)
    output["market_anchor_logit"] = _logit(_anchor_probability(row))
    output["binary_payout_target"] = _binary_payout_label(row)
    return output


def _primary_schema_folds(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        item = deepcopy(dict(row))
        item["schema_version"] = PRIMARY_FOLD_SCHEMA_VERSION
        output.append(item)
    return output


def _logit_target_contract() -> dict[str, Any]:
    return {
        "action_value_formula": (
            "pair_normalized_binomial_probability-entry_ask-frozen_fees-"
            "slippage-liquidity_impact"
        ),
        "execution_policy": "HOLD_TO_SETTLEMENT",
        "NO_TRADE_value": 0.0,
        "post_close_training_label_only": True,
        "binary_label": "settlement_payout",
        "base_margin": "logit(decision_time_selected_mid)",
    }


def _logit_feature_contract() -> dict[str, Any]:
    return {
        "ordered_feature_count": 108,
        "base_feature_count": 54,
        "ordered_feature_names_sha256": canonical_json_sha256(list(FEATURE_NAMES)),
        "shared_side_symmetric_model": True,
        "side_identity_feature_allowed": False,
        "decision_time_causal_inputs_only": True,
        "native_missing_value": "nan",
        "missing_values_encoded_as_zero": False,
        "feature_search_allowed": False,
        "market_horizon_seconds": 900,
        "source_contract_reused_without_feature_addition_or_removal": True,
    }


def _equal_weight_contract() -> dict[str, Any]:
    return {
        "method": "equal_training_row_weight",
        "fixed_weight": 1.0,
        "market_order": "frozen_capture_order",
        "future_market_used_to_compute_weight": False,
        "weight_or_window_search_allowed": False,
    }


def _logit_model_parameters() -> dict[str, Any]:
    return {
        "colsample_bytree": 1.0,
        "eta": 0.03,
        "eval_metric": "logloss",
        "gamma": 0.0,
        "max_bin": 64,
        "max_depth": 2,
        "min_child_weight": 16.0,
        "nthread": 1,
        "objective": "binary:logistic",
        "reg_alpha": 0.0,
        "reg_lambda": 20.0,
        "seed": 26421,
        "subsample": 1.0,
        "tree_method": "hist",
    }


def _logit_model_contract() -> dict[str, Any]:
    return {
        "family": (
            "pooled_side_symmetric_binomial_xgboost_with_decision_time_market_"
            "logit_offset"
        ),
        "base_margin": "logit(decision_time_selected_mid)",
        "route_or_expert_allowed": False,
        "fixed_num_boost_round": 128,
        "model_selection_or_early_stopping_performed": False,
        "parameters": _logit_model_parameters(),
    }


__all__ = [
    "logit_offset_action_values",
    "run_logit_challenger_oof",
    "validate_logit_challenger_protocol",
]
