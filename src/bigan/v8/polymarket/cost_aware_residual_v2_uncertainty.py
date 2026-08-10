"""Sole uncertainty-aware challenger for BTC 15m residual lineage v2."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import xgboost as xgb

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.cost_aware_residual import _validate_frozen_population
from bigan.v8.polymarket.cost_aware_residual_v2 import (
    DEFAULT_CONFIG_DIR,
    LINEAGE_ID,
    MARKET_RESULT_SCHEMA_VERSION,
    _as_v1_folds,
    _as_v1_market_results,
    _as_v1_predictions,
    _build_v2_report,
    _dmatrix,
    _load_frozen_development_rows,
    _load_json,
    _load_jsonl,
    _looks_like_git_sha,
    _model_parameters,
    _public_v2_dataset_row,
    _v2_market_results_from_predictions,
    _verified_json,
    _verify_descriptor,
    _write_new_frozen_json,
    _write_new_frozen_text,
    _write_new_jsonl,
    pair_anchored_action_values,
    render_residual_v2_markdown,
    validate_residual_v2_protocol,
)
from bigan.v8.polymarket.cost_aware_residual_v2 import (
    DEFAULT_PROTOCOL as PRIMARY_PROTOCOL,
)
from bigan.v8.polymarket.cost_aware_residual_v2 import (
    FOLD_SCHEMA_VERSION as PRIMARY_FOLD_SCHEMA_VERSION,
)
from bigan.v8.polymarket.cost_aware_residual_v2 import _descriptor as _descriptor
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.moe_terminal_diagnostic import _assert_semantically_equal
from bigan.v8.polymarket.regime_adaptive_lineage import REPO_ROOT

CHALLENGER_PROTOCOL_SCHEMA_VERSION = (
    "bigan-btc-15m-market-anchored-residual-uncertainty-oof-protocol-v2"
)
PREDICTION_SCHEMA_VERSION = (
    "bigan-btc-15m-market-anchored-residual-uncertainty-oof-prediction-v2"
)
FOLD_SCHEMA_VERSION = (
    "bigan-btc-15m-market-anchored-residual-uncertainty-oof-fold-v2"
)
REPORT_SCHEMA_VERSION = (
    "bigan-btc-15m-market-anchored-residual-uncertainty-oof-report-v2"
)
MANIFEST_SCHEMA_VERSION = (
    "bigan-btc-15m-market-anchored-residual-uncertainty-oof-manifest-v2"
)

DEFAULT_CHALLENGER_PROTOCOL = (
    DEFAULT_CONFIG_DIR / "residual_v2_challenger_slot_002_protocol.json"
)
DEFAULT_CHALLENGER_OUTPUT_DIR = (
    DEFAULT_CONFIG_DIR / "residual_v2_challenger_slot_002_oof"
)

STRUCTURAL_CHANGE = {
    "changed_component": "fixed_three_head_training_and_uncertainty_adjusted_action_value",
    "from": "single_conditional_mean_probability_residual_head",
    "to": (
        "fixed_mean_plus_q25_q75_probability_residual_heads_with_half_IQR_"
        "uncertainty_deduction"
    ),
    "reason": (
        "slot_1_passed_10_of_11_gates_but_required_2764_markets_because_"
        "accepted_578_of_600_markets"
    ),
    "expected_mechanism": (
        "a fixed conditional residual half-IQR deduction abstains when the mean "
        "after-cost edge is small relative to model uncertainty, reducing variance "
        "without changing the zero threshold"
    ),
    "threshold_changed": False,
    "feature_set_changed": False,
    "rolling_population_changed": False,
    "gate_or_power_cap_changed": False,
    "route_side_missingness_or_outlier_filter_added": False,
}


def validate_uncertainty_challenger_protocol(
    payload: Mapping[str, Any],
    *,
    repository_root: Path | str = REPO_ROOT,
    verify_artifacts: bool = True,
) -> None:
    """Validate the second and final v2 slot without relaxing shared contracts."""

    blockers: list[str] = []
    if payload.get("schema_version") != CHALLENGER_PROTOCOL_SCHEMA_VERSION:
        blockers.append("schema_version")
    if payload.get("slot_id") != "residual-v2-challenger-slot-002":
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
    if dict(payload.get("target") or {}) != _uncertainty_target_contract():
        blockers.append("target")
    model = dict(payload.get("model") or {})
    if model != _uncertainty_model_contract():
        blockers.append("model")
    if dict(payload.get("structural_change") or {}) != STRUCTURAL_CHANGE:
        blockers.append("structural_change")
    prior = dict(payload.get("prior_slot_result") or {})
    if not (
        set(prior) == {"manifest", "report", "failed_gates"}
        and prior.get("failed_gates")
        == ["prospective_power_required_market_count_lte_2000"]
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
    normalized["schema_version"] = (
        "bigan-btc-15m-market-anchored-residual-oof-protocol-v2"
    )
    normalized["slot_id"] = "residual-v2-primary-slot-001"
    normalized["candidate_role"] = "primary"
    normalized["candidate_budget"] = {
        "maximum_total_slots": 2,
        "this_slot_ordinal": 1,
        "slots_consumed_before_run": 0,
        "slots_remaining_after_run": 1,
        "slot_budget_may_be_increased": False,
    }
    normalized["target"] = {
        "action_value_formula": (
            "pair_normalized_clipped_selected_mid_plus_predicted_probability_"
            "residual-entry_ask-frozen_fees-slippage-liquidity_impact"
        ),
        "execution_policy": "HOLD_TO_SETTLEMENT",
        "NO_TRADE_value": 0.0,
        "post_close_training_label_only": True,
        "regression_label": "settlement_payout-selected_mid",
    }
    normalized["model"] = {
        "family": (
            "pooled_side_symmetric_market_anchored_probability_residual_xgboost"
        ),
        "route_or_expert_allowed": False,
        "fixed_num_boost_round": 128,
        "model_selection_or_early_stopping_performed": False,
        "parameters": _model_parameters(),
    }
    primary = _load_json(Path(PRIMARY_PROTOCOL))
    normalized["inputs"]["candidate_implementation"] = primary["inputs"][
        "candidate_implementation"
    ]
    normalized.pop("prior_slot_result", None)
    normalized.pop("structural_change", None)
    try:
        validate_residual_v2_protocol(
            normalized,
            repository_root=root,
            verify_artifacts=verify_artifacts,
        )
    except ValueError as error:
        blockers.append(f"shared_protocol:{error}")
    if blockers:
        raise ValueError(
            "residual v2 uncertainty challenger protocol invalid: "
            + ", ".join(blockers)
        )


def run_uncertainty_challenger_oof(
    *,
    protocol_path: Path | str = DEFAULT_CHALLENGER_PROTOCOL,
    expected_protocol_sha256: str,
    output_dir: Path | str = DEFAULT_CHALLENGER_OUTPUT_DIR,
    source_commit: str,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Execute the second and final preregistered v2 candidate exactly once."""

    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    output = Path(output_dir).resolve()
    if not protocol_file.is_relative_to(root) or not output.is_relative_to(root):
        raise ValueError("uncertainty challenger paths must remain repository-local")
    if sha256_file(protocol_file) != expected_protocol_sha256.lower():
        raise ValueError("uncertainty challenger protocol SHA-256 mismatch")
    sidecar = protocol_file.with_suffix(".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != (
        expected_protocol_sha256
    ):
        raise ValueError("uncertainty challenger protocol is not SHA-frozen")
    if not _looks_like_git_sha(source_commit):
        raise ValueError("source_commit must be a full lowercase Git SHA")
    protocol = _load_json(protocol_file)
    validate_uncertainty_challenger_protocol(protocol, repository_root=root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"uncertainty challenger output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    dataset_rows, baseline_by_market, population_order = _load_frozen_development_rows(
        protocol=protocol,
        repository_root=root,
    )
    predictions, folds = _rolling_origin_uncertainty_predict(
        rows=dataset_rows,
        population_order=population_order,
        protocol=protocol,
    )
    markets = _v2_market_results_from_predictions(
        predictions=predictions,
        baseline_by_market=baseline_by_market,
        population_order=population_order,
        protocol=protocol,
    )
    report = _build_uncertainty_report(
        protocol=protocol,
        protocol_sha256=expected_protocol_sha256,
        source_commit=source_commit,
        market_results=markets,
        fold_audits=folds,
    )

    dataset_path = output / "residual_v2_development_dataset_rows.jsonl"
    prediction_path = output / "residual_v2_uncertainty_oof_predictions.jsonl"
    fold_path = output / "residual_v2_uncertainty_oof_fold_audits.jsonl"
    market_path = output / "residual_v2_uncertainty_oof_market_results.jsonl"
    report_path = output / "residual_v2_uncertainty_oof_report.json"
    markdown_path = output / "residual_v2_uncertainty_oof_report.md"
    _write_new_jsonl(dataset_path, [_public_v2_dataset_row(row) for row in dataset_rows])
    _write_new_jsonl(prediction_path, predictions)
    _write_new_jsonl(fold_path, folds)
    _write_new_jsonl(market_path, markets)
    report_artifact = _write_new_frozen_json(report_path, report)
    markdown_artifact = _write_new_frozen_text(
        markdown_path, render_uncertainty_challenger_markdown(report)
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
        "immutable_v1_gate_implementation": dict(
            protocol["inputs"]["gate_implementation"]
        ),
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
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "safety": dict(SAFETY),
    }
    manifest_artifact = _write_new_frozen_json(
        output / "residual_v2_uncertainty_oof_manifest.json", manifest
    )
    return {
        "manifest": _descriptor(Path(manifest_artifact["path"]), root),
        "report": _descriptor(Path(report_artifact["path"]), root),
        "all_gates_passed": report["all_gates_passed"],
        "failed_gates": report["failed_gates"],
        "candidate_budget_exhausted": True,
        "oof_market_count": len(markets),
        "safety": dict(SAFETY),
    }


def verify_frozen_uncertainty_challenger_oof(
    *,
    protocol_path: Path | str = DEFAULT_CHALLENGER_PROTOCOL,
    output_dir: Path | str = DEFAULT_CHALLENGER_OUTPUT_DIR,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Verify all frozen challenger bytes and independently rebuild its report."""

    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    output = Path(output_dir).resolve()
    protocol = _verified_json(protocol_file)
    validate_uncertainty_challenger_protocol(protocol, repository_root=root)
    manifest = _verified_json(output / "residual_v2_uncertainty_oof_manifest.json")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("uncertainty challenger manifest schema mismatch")
    if manifest.get("protocol") != _descriptor(protocol_file, root):
        raise ValueError("uncertainty challenger protocol binding mismatch")
    artifacts = {
        name: _verify_descriptor(dict(descriptor), repository_root=root)
        for name, descriptor in dict(manifest.get("artifacts") or {}).items()
    }
    if set(artifacts) != {
        "dataset_rows",
        "predictions",
        "fold_audits",
        "market_results",
        "report",
        "report_markdown",
    }:
        raise ValueError("uncertainty challenger artifact set mismatch")
    predictions = _load_jsonl(artifacts["predictions"])
    folds = _load_jsonl(artifacts["fold_audits"])
    markets = _load_jsonl(artifacts["market_results"])
    _validate_uncertainty_population(
        predictions=predictions,
        fold_audits=folds,
        market_results=markets,
        protocol=protocol,
    )
    rebuilt = _build_uncertainty_report(
        protocol=protocol,
        protocol_sha256=sha256_file(protocol_file),
        source_commit=str(manifest["source_commit"]),
        market_results=markets,
        fold_audits=folds,
    )
    frozen = _load_json(artifacts["report"])
    _assert_semantically_equal(
        rebuilt, frozen, path="residual_v2_uncertainty_oof_report"
    )
    if render_uncertainty_challenger_markdown(rebuilt) != artifacts[
        "report_markdown"
    ].read_text(encoding="utf-8"):
        raise ValueError("uncertainty challenger Markdown does not reproduce")
    return {
        "verification_passed": True,
        "all_gates_passed": bool(frozen["all_gates_passed"]),
        "failed_gates": list(frozen["failed_gates"]),
        "candidate_budget_exhausted": True,
        "oof_market_count": len(markets),
        "manifest_sha256": sha256_file(
            output / "residual_v2_uncertainty_oof_manifest.json"
        ),
        "parent_v1_immutable": True,
        "safety": dict(SAFETY),
    }


def uncertainty_adjusted_action_values(
    mean_actions: Sequence[Mapping[str, float]],
    lower_residual_predictions: Sequence[float],
    upper_residual_predictions: Sequence[float],
) -> list[dict[str, float]]:
    """Deduct one fixed conditional half-IQR from each coherent mean action value."""

    if not (
        len(mean_actions)
        == len(lower_residual_predictions)
        == len(upper_residual_predictions)
    ):
        raise ValueError("uncertainty head row count mismatch")
    output = []
    for mean, lower, upper in zip(
        mean_actions,
        lower_residual_predictions,
        upper_residual_predictions,
        strict=True,
    ):
        lower_value = float(lower)
        upper_value = float(upper)
        if not all(math.isfinite(value) for value in (lower_value, upper_value)):
            raise ValueError("uncertainty quantile prediction is not finite")
        penalty = max(0.0, upper_value - lower_value) / 2.0
        item = dict(mean)
        item.update(
            {
                "lower_probability_residual_prediction": lower_value,
                "upper_probability_residual_prediction": upper_value,
                "conditional_half_IQR_uncertainty_penalty": penalty,
                "action_value_before_uncertainty_penalty": float(
                    mean["action_value"]
                ),
                "action_value": float(mean["action_value"]) - penalty,
            }
        )
        output.append(item)
    return output


def render_uncertainty_challenger_markdown(report: Mapping[str, Any]) -> str:
    """Render shared gate evidence plus terminal two-slot governance."""

    base = render_residual_v2_markdown(report)
    base = base.replace(
        "# BTC 15m market-anchored residual v2 primary slot 001",
        "# BTC 15m market-anchored residual v2 uncertainty challenger slot 002",
        1,
    ).rstrip()
    return (
        base
        + "\n\n## Uncertainty challenger and candidate budget\n\n"
        + "- Fixed heads: conditional mean, q25, q75.\n"
        + "- Action-value uncertainty deduction: `max(0, q75-q25)/2`.\n"
        + "- Acceptance threshold changed from zero: `False`\n"
        + "- Second and final candidate slot consumed: `True`\n"
        + "- Additional candidate allowed in this lineage: `False`\n"
    )


def _rolling_origin_uncertainty_predict(
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
    model = dict(protocol["model"])
    parameters = dict(model["head_parameters"])
    rounds = dict(model["fixed_num_boost_round_by_head"])
    predictions: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for block_index in range(block_count):
        target_start = initial + block_index * block_size
        target_end = target_start + block_size
        training_ids = list(population_order[:target_start])
        target_ids = list(population_order[target_start:target_end])
        if len(target_ids) != block_size:
            raise ValueError("uncertainty challenger target population mismatch")
        train_rows = [row for market_id in training_ids for row in rows_by_market[market_id]]
        target_rows = [row for market_id in target_ids for row in rows_by_market[market_id]]
        labels = [_probability_residual_label(row) for row in train_rows]
        train_matrix = _dmatrix(train_rows, labels=labels)
        target_matrix = _dmatrix(target_rows, labels=None)
        head_values = {}
        for head in ("mean", "lower_q25", "upper_q75"):
            booster = xgb.train(
                params=dict(parameters[head]),
                dtrain=train_matrix,
                num_boost_round=int(rounds[head]),
                verbose_eval=False,
            )
            head_values[head] = [
                float(value) for value in booster.predict(target_matrix)
            ]
        mean_actions = pair_anchored_action_values(
            target_rows, head_values["mean"]
        )
        actions = uncertainty_adjusted_action_values(
            mean_actions,
            head_values["lower_q25"],
            head_values["upper_q75"],
        )
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
                    "action_value_before_uncertainty_penalty": action[
                        "action_value_before_uncertainty_penalty"
                    ],
                    "conditional_half_IQR_uncertainty_penalty": action[
                        "conditional_half_IQR_uncertainty_penalty"
                    ],
                    "market_anchor_probability": action[
                        "market_anchor_probability"
                    ],
                    "predicted_probability_residual": action[
                        "predicted_probability_residual"
                    ],
                    "lower_probability_residual_prediction": action[
                        "lower_probability_residual_prediction"
                    ],
                    "upper_probability_residual_prediction": action[
                        "upper_probability_residual_prediction"
                    ],
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
                "training_residual_labels_sha256": canonical_json_sha256(labels),
                "last_training_market_position": target_start,
                "first_target_market_position": target_start + 1,
                "target_or_future_label_leakage_count": 0,
                "fixed_num_boost_round_by_head": rounds,
                "head_parameters_sha256": canonical_json_sha256(parameters),
                "uncertainty_penalty": "max(0,q75-q25)/2",
                "development_only_forever": True,
                "promotion_evidence_eligible": False,
                "safety": dict(SAFETY),
            }
        )
    return predictions, audits


def _build_uncertainty_report(
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    source_commit: str,
    market_results: Sequence[Mapping[str, Any]],
    fold_audits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    report = _build_v2_report(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        source_commit=source_commit,
        market_results=market_results,
        fold_audits=_primary_schema_folds(fold_audits),
    )
    report["schema_version"] = REPORT_SCHEMA_VERSION
    report["candidate_role"] = "challenger"
    report["architecture_type"] = (
        "pooled_side_symmetric_market_anchored_probability_residual_three_head_"
        "uncertainty_adjusted_with_deterministic_pair_coherence"
    )
    report["candidate_budget_exhausted"] = True
    report["remaining_candidate_slots"] = 0
    report["additional_candidate_allowed"] = False
    report["structural_change"] = dict(protocol["structural_change"])
    return report


def _validate_uncertainty_population(
    *,
    predictions: Sequence[Mapping[str, Any]],
    fold_audits: Sequence[Mapping[str, Any]],
    market_results: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> None:
    if any(
        row.get("schema_version") != PREDICTION_SCHEMA_VERSION
        or row.get("lineage_id") != LINEAGE_ID
        or row.get("target_or_future_label_used_for_fit") is not False
        or dict(row.get("safety") or {}) != SAFETY
        for row in predictions
    ):
        raise ValueError("uncertainty challenger prediction governance mismatch")
    if any(
        row.get("schema_version") != FOLD_SCHEMA_VERSION
        or row.get("lineage_id") != LINEAGE_ID
        or dict(row.get("safety") or {}) != SAFETY
        for row in fold_audits
    ):
        raise ValueError("uncertainty challenger fold governance mismatch")
    if any(
        row.get("schema_version") != MARKET_RESULT_SCHEMA_VERSION
        or row.get("lineage_id") != LINEAGE_ID
        or dict(row.get("safety") or {}) != SAFETY
        for row in market_results
    ):
        raise ValueError("uncertainty challenger market governance mismatch")
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[(str(row["market_id"]), int(row["decision_ts"]))].append(row)
        cost = dict(row["cost_decomposition"])
        expected_before = (
            float(row["predicted_probability"])
            - float(cost["entry_ask"])
            - float(cost["total_cost_excluding_entry_ask"])
        )
        expected = expected_before - float(
            row["conditional_half_IQR_uncertainty_penalty"]
        )
        if not (
            math.isclose(
                expected_before,
                float(row["action_value_before_uncertainty_penalty"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            and math.isclose(
                expected,
                float(row["prediction"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("uncertainty challenger action-value mismatch")
    for rows in grouped.values():
        if not math.isclose(
            sum(float(row["predicted_probability"]) for row in rows),
            1.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("uncertainty challenger pair coherence mismatch")
    _validate_frozen_population(
        predictions=_as_v1_predictions(predictions),
        fold_audits=_as_v1_folds(_primary_schema_folds(fold_audits)),
        market_results=_as_v1_market_results(market_results),
        protocol=protocol,
    )


def _primary_schema_folds(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        item = deepcopy(dict(row))
        item["schema_version"] = PRIMARY_FOLD_SCHEMA_VERSION
        output.append(item)
    return output


def _probability_residual_label(row: Mapping[str, Any]) -> float:
    from bigan.v8.polymarket.cost_aware_residual_v2 import (
        _probability_residual_label as primary_label,
    )

    return primary_label(row)


def _uncertainty_target_contract() -> dict[str, Any]:
    return {
        "action_value_formula": (
            "pair_normalized_mean_probability-entry_ask-frozen_fees-slippage-"
            "liquidity_impact-max(0,q75_probability_residual-q25_probability_"
            "residual)/2"
        ),
        "execution_policy": "HOLD_TO_SETTLEMENT",
        "NO_TRADE_value": 0.0,
        "post_close_training_label_only": True,
        "regression_label": "settlement_payout-selected_mid",
        "uncertainty_penalty": "conditional_residual_half_IQR",
    }


def _uncertainty_model_contract() -> dict[str, Any]:
    mean = _model_parameters()
    lower = dict(mean)
    lower.update(
        {
            "objective": "reg:quantileerror",
            "eval_metric": "quantile",
            "quantile_alpha": 0.25,
            "seed": 26422,
        }
    )
    upper = dict(mean)
    upper.update(
        {
            "objective": "reg:quantileerror",
            "eval_metric": "quantile",
            "quantile_alpha": 0.75,
            "seed": 26423,
        }
    )
    return {
        "family": (
            "pooled_side_symmetric_market_anchored_probability_residual_"
            "three_head_uncertainty_xgboost"
        ),
        "route_or_expert_allowed": False,
        "model_selection_or_early_stopping_performed": False,
        "fixed_num_boost_round_by_head": {
            "mean": 128,
            "lower_q25": 128,
            "upper_q75": 128,
        },
        "head_parameters": {
            "mean": mean,
            "lower_q25": lower,
            "upper_q75": upper,
        },
    }


__all__ = [
    "render_uncertainty_challenger_markdown",
    "run_uncertainty_challenger_oof",
    "uncertainty_adjusted_action_values",
    "validate_uncertainty_challenger_protocol",
    "verify_frozen_uncertainty_challenger_oof",
]
