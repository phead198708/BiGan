"""Nested rolling-origin soft-stacking challenger for residual lineage v4."""

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
from bigan.v8.polymarket.cost_aware_residual import (
    FOLD_SCHEMA_VERSION as V1_FOLD_SCHEMA_VERSION,
)
from bigan.v8.polymarket.cost_aware_residual import (
    LINEAGE_ID as V1_LINEAGE_ID,
)
from bigan.v8.polymarket.cost_aware_residual import (
    MARKET_RESULT_SCHEMA_VERSION as V1_MARKET_RESULT_SCHEMA_VERSION,
)
from bigan.v8.polymarket.cost_aware_residual import (
    PREDICTION_SCHEMA_VERSION as V1_PREDICTION_SCHEMA_VERSION,
)
from bigan.v8.polymarket.cost_aware_residual import (
    _descriptor,
    _load_frozen_development_rows,
    _load_json,
    _load_jsonl,
    _looks_like_git_sha,
    _validate_frozen_population,
    _verified_json,
    _verify_descriptor,
    build_residual_oof_report,
    market_results_from_predictions,
    render_residual_oof_markdown,
)
from bigan.v8.polymarket.cost_aware_residual_v2 import (
    GATE_NAMES,
    _action_policy,
    _bootstrap_contract,
    _cost_stress_contract,
    _power_contract,
    _probability_residual_label,
    _rolling_contract,
    pair_anchored_action_values,
)
from bigan.v8.polymarket.cost_aware_residual_v2 import (
    _dmatrix as _residual_dmatrix,
)
from bigan.v8.polymarket.cost_aware_residual_v3_logit import (
    _binary_payout_label,
    logit_offset_action_values,
)
from bigan.v8.polymarket.cost_aware_residual_v3_logit import (
    _dmatrix as _logit_dmatrix,
)
from bigan.v8.polymarket.cost_aware_residual_v4 import (
    DATASET_SCHEMA_VERSION as PRIMARY_DATASET_SCHEMA_VERSION,
)
from bigan.v8.polymarket.cost_aware_residual_v4 import (
    IMMUTABLE_GATE_IMPLEMENTATION,
    IMMUTABLE_LOGIT_BASE_IMPLEMENTATION,
    IMMUTABLE_RESIDUAL_BASE_IMPLEMENTATION,
    LINEAGE_ID,
    _baseline_contract,
    _dataset_contract,
    _feature_contract,
    _public_dataset_v4_row,
    validate_v4_lineage_authorization,
)
from bigan.v8.polymarket.moe_collection_boundary_r2 import (
    _write_new_frozen_json,
    _write_new_jsonl,
)
from bigan.v8.polymarket.moe_confirmatory_evaluation import _write_new_frozen_text
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.moe_terminal_diagnostic import _assert_semantically_equal
from bigan.v8.polymarket.regime_adaptive_lineage import REPO_ROOT

PROTOCOL_SCHEMA_VERSION = "bigan-btc-15m-nested-soft-stacking-oof-protocol-v4"
PREDICTION_SCHEMA_VERSION = "bigan-btc-15m-nested-soft-stacking-oof-prediction-v4"
FOLD_SCHEMA_VERSION = "bigan-btc-15m-nested-soft-stacking-oof-fold-v4"
MARKET_RESULT_SCHEMA_VERSION = "bigan-btc-15m-nested-soft-stacking-market-result-v4"
REPORT_SCHEMA_VERSION = "bigan-btc-15m-nested-soft-stacking-oof-report-v4"
MANIFEST_SCHEMA_VERSION = "bigan-btc-15m-nested-soft-stacking-oof-manifest-v4"
DATASET_SCHEMA_VERSION = "bigan-btc-15m-nested-soft-stacking-dataset-row-v4"

DEFAULT_CONFIG_DIR = (
    REPO_ROOT / "examples/v8/polymarket_configs/BTC-15M-cost-aware-market-residual-v4"
)
DEFAULT_PROTOCOL = DEFAULT_CONFIG_DIR / "residual_v4_challenger_slot_002_protocol.json"
DEFAULT_OUTPUT_DIR = DEFAULT_CONFIG_DIR / "residual_v4_challenger_slot_002_oof"
PRIMARY_OUTPUT_DIR = DEFAULT_CONFIG_DIR / "residual_v4_primary_slot_001_oof"
PRIMARY_REPORT_PATH = PRIMARY_OUTPUT_DIR / "residual_v4_oof_report.json"
PRIMARY_MANIFEST_PATH = PRIMARY_OUTPUT_DIR / "residual_v4_oof_manifest.json"

INNER_INITIAL_TRAINING_MARKETS = 100
INNER_TARGET_BLOCK_MARKETS = 100
META_REGULARIZATION = 20.0
META_MAX_ITERATIONS = 64
META_TOLERANCE = 1e-10
PROBABILITY_CLIP_EPSILON = 1e-6

STRUCTURAL_CHANGE = {
    "changed_component": "ensemble_combiner_and_calibration_training_design",
    "from": ("horizon_adaptive_hedge_weights_from_prior_oof_aggregate_binary_log_loss"),
    "to": ("nested_rolling_origin_l2_logistic_soft_stacker_on_base_probability_logits"),
    "reason": (
        "slot_1_required_2999_markets_and_failed_the_first_paired_chronological_"
        "block;_aggregate_hedge_loss_did_not_learn_market_level_calibration"
    ),
    "expected_mechanism": (
        "strictly_prior inner-OOF base predictions let a fixed regularized pooled "
        "stacker learn smooth market-level calibration without hard routing, while "
        "pair normalization and frozen cost subtraction preserve action semantics"
    ),
    "threshold_changed": False,
    "cost_baseline_population_or_gate_changed": False,
    "route_side_missingness_or_outlier_filter_added": False,
    "parameter_weight_or_threshold_search_performed": False,
}


def require_stacking_candidate_implementation_binding(
    payload: Mapping[str, Any],
    *,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, str]:
    """Bind the challenger declaration to this exact executing module."""

    root = Path(repository_root).resolve()
    expected = _descriptor(Path(__file__), root)
    try:
        declared = dict(payload["inputs"]["candidate_implementation"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("stacking implementation descriptor unavailable") from exc
    if declared != expected:
        raise ValueError("stacking candidate implementation does not identify the executing module")
    if _verify_descriptor(declared, repository_root=root) != Path(__file__).resolve():
        raise ValueError("stacking implementation resolved to another module")
    return expected


def validate_stacking_challenger_protocol(
    payload: Mapping[str, Any],
    *,
    repository_root: Path | str = REPO_ROOT,
    verify_artifacts: bool = True,
) -> None:
    """Validate the second and final v4 slot with all shared bytes unchanged."""

    blockers: list[str] = []
    if payload.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        blockers.append("schema_version")
    if payload.get("lineage_id") != LINEAGE_ID:
        blockers.append("lineage_id")
    if payload.get("slot_id") != "residual-v4-challenger-slot-002":
        blockers.append("slot_id")
    if payload.get("candidate_role") != "challenger":
        blockers.append("candidate_role")
    if payload.get("development_only_forever") is not True:
        blockers.append("development_only_forever")
    if payload.get("promotion_evidence_eligible") is not False:
        blockers.append("promotion_evidence_eligible")
    if dict(payload.get("candidate_budget") or {}) != _candidate_budget_contract():
        blockers.append("candidate_budget")
    if dict(payload.get("target") or {}) != _target_contract():
        blockers.append("target")
    if dict(payload.get("pair_coherence") or {}) != _pair_contract():
        blockers.append("pair_coherence")
    if dict(payload.get("feature_contract") or {}) != _feature_contract():
        blockers.append("feature_contract")
    if dict(payload.get("model") or {}) != _model_contract():
        blockers.append("model")
    if dict(payload.get("nested_stacking") or {}) != _nested_contract():
        blockers.append("nested_stacking")
    if dict(payload.get("structural_change") or {}) != STRUCTURAL_CHANGE:
        blockers.append("structural_change")
    if dict(payload.get("action_policy") or {}) != _action_policy():
        blockers.append("action_policy")
    if dict(payload.get("bootstrap") or {}) != _bootstrap_contract():
        blockers.append("bootstrap")
    if dict(payload.get("cost_stress") or {}) != _cost_stress_contract():
        blockers.append("cost_stress")
    gates = dict(payload.get("gates") or {})
    if set(gates) != GATE_NAMES or any(value is not True for value in gates.values()):
        blockers.append("gates")
    if dict(payload.get("prospective_power") or {}) != _power_contract():
        blockers.append("prospective_power")
    if dict(payload.get("rolling_origin") or {}) != _rolling_contract():
        blockers.append("rolling_origin")
    if dict(payload.get("dataset") or {}) != _dataset_contract():
        blockers.append("dataset")
    if dict(payload.get("baseline") or {}) != _baseline_contract():
        blockers.append("baseline")
    if dict(payload.get("development_discipline") or {}) != _discipline_contract():
        blockers.append("development_discipline")
    if dict(payload.get("state") or {}) != _state_contract():
        blockers.append("state")
    if dict(payload.get("safety") or {}) != SAFETY:
        blockers.append("safety")
    prior = dict(payload.get("prior_slot_result") or {})
    if not (
        set(prior) == {"manifest", "report", "failed_gates"}
        and prior.get("failed_gates")
        == [
            "every_chronological_block_paired_delta_total_gte_zero",
            "prospective_power_required_market_count_lte_2000",
        ]
    ):
        blockers.append("prior_slot_result")
    root = Path(repository_root).resolve()
    try:
        require_stacking_candidate_implementation_binding(payload, repository_root=root)
    except ValueError:
        blockers.append("candidate_implementation_exact_binding")
    inputs = dict(payload.get("inputs") or {})
    required_inputs = {
        "lineage_authorization",
        "development_data_registry",
        "parent_v3_terminal_review",
        "parent_v3_binding_audit",
        "primary_manifest",
        "primary_report",
        "terminal_diagnostic_scored_rows",
        "confirmatory_capture_manifest",
        "confirmatory_market_evaluation_rows",
        "baseline_decision_rows",
        "matched_global_baseline_contract",
        "parent_feature_contract",
        "parent_cost_and_action_contract",
        "raw_capture_recovery_bundle_manifest",
        "candidate_implementation",
        "residual_base_implementation",
        "logit_base_implementation",
        "gate_implementation",
    }
    if set(inputs) != required_inputs:
        blockers.append("inputs")
    if dict(inputs.get("gate_implementation") or {}) != IMMUTABLE_GATE_IMPLEMENTATION:
        blockers.append("gate_implementation")
    if dict(inputs.get("residual_base_implementation") or {}) != (
        IMMUTABLE_RESIDUAL_BASE_IMPLEMENTATION
    ):
        blockers.append("residual_base_implementation")
    if dict(inputs.get("logit_base_implementation") or {}) != (IMMUTABLE_LOGIT_BASE_IMPLEMENTATION):
        blockers.append("logit_base_implementation")
    if verify_artifacts and not blockers:
        resolved: dict[str, Path] = {}
        for name, descriptor in inputs.items():
            try:
                resolved[name] = _verify_descriptor(dict(descriptor), repository_root=root)
            except (KeyError, OSError, TypeError, ValueError):
                blockers.append(f"inputs.{name}")
        for field in ("manifest", "report"):
            try:
                prior_path = _verify_descriptor(dict(prior[field]), repository_root=root)
                expected = (
                    resolved["primary_manifest"]
                    if field == "manifest"
                    else resolved["primary_report"]
                )
                if prior_path != expected:
                    blockers.append(f"prior_slot_result.{field}.path")
            except (KeyError, OSError, TypeError, ValueError):
                blockers.append(f"prior_slot_result.{field}")
        if not blockers:
            try:
                authorization = validate_v4_lineage_authorization(
                    authorization_path=resolved["lineage_authorization"],
                    registry_path=resolved["development_data_registry"],
                    repository_root=root,
                )
                if authorization["maximum_total_slots"] != 2:
                    blockers.append("lineage_authorization.slot_budget")
            except ValueError:
                blockers.append("lineage_authorization")
            primary = _load_json(resolved["primary_report"])
            if not (
                primary.get("all_gates_passed") is False
                and primary.get("failed_gates") == prior["failed_gates"]
                and primary.get("remaining_candidate_slots") == 1
                and dict(primary.get("safety") or {}) == SAFETY
            ):
                blockers.append("primary_report_semantics")
            terminal = _load_json(resolved["parent_v3_terminal_review"])
            audit = _load_json(resolved["parent_v3_binding_audit"])
            if not (
                terminal.get("phase_1_terminal_failed") is True
                and terminal.get("candidate_budget_exhausted") is True
                and audit.get("audit_passed") is True
                and dict(terminal.get("safety") or {}) == SAFETY
                and dict(audit.get("safety") or {}) == SAFETY
            ):
                blockers.append("parent_v3_boundary")
    if blockers:
        raise ValueError("residual v4 stacking challenger protocol invalid: " + ", ".join(blockers))


def run_stacking_challenger_oof(
    *,
    protocol_path: Path | str = DEFAULT_PROTOCOL,
    expected_protocol_sha256: str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    source_commit: str,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Execute the second and final preregistered v4 candidate exactly once."""

    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    output = Path(output_dir).resolve()
    if not protocol_file.is_relative_to(root) or not output.is_relative_to(root):
        raise ValueError("stacking challenger paths must remain repository-local")
    if sha256_file(protocol_file) != expected_protocol_sha256.lower():
        raise ValueError("stacking challenger protocol SHA-256 mismatch")
    sidecar = protocol_file.with_suffix(".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != (
        expected_protocol_sha256
    ):
        raise ValueError("stacking challenger protocol is not SHA-frozen")
    if not _looks_like_git_sha(source_commit):
        raise ValueError("source_commit must be a full lowercase Git SHA")
    protocol = _load_json(protocol_file)
    validate_stacking_challenger_protocol(protocol, repository_root=root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"stacking challenger output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    dataset_rows, baseline_by_market, population_order = _load_frozen_development_rows(
        protocol=protocol,
        repository_root=root,
    )
    predictions, folds = nested_rolling_origin_soft_stacking_predict(
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
    report = _build_report(
        protocol=protocol,
        protocol_sha256=expected_protocol_sha256,
        source_commit=source_commit,
        market_results=markets,
        fold_audits=folds,
    )

    dataset_path = output / "residual_v4_stacking_development_dataset_rows.jsonl"
    prediction_path = output / "residual_v4_stacking_oof_predictions.jsonl"
    fold_path = output / "residual_v4_stacking_oof_fold_audits.jsonl"
    market_path = output / "residual_v4_stacking_oof_market_results.jsonl"
    report_path = output / "residual_v4_stacking_oof_report.json"
    markdown_path = output / "residual_v4_stacking_oof_report.md"
    _write_new_jsonl(dataset_path, [_public_dataset_row(row) for row in dataset_rows])
    _write_new_jsonl(prediction_path, predictions)
    _write_new_jsonl(fold_path, folds)
    _write_new_jsonl(market_path, markets)
    report_artifact = _write_new_frozen_json(report_path, report)
    markdown_artifact = _write_new_frozen_text(markdown_path, render_stacking_markdown(report))
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
        "residual_base_implementation": dict(protocol["inputs"]["residual_base_implementation"]),
        "logit_base_implementation": dict(protocol["inputs"]["logit_base_implementation"]),
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
        output / "residual_v4_stacking_oof_manifest.json", manifest
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


def verify_frozen_stacking_challenger_oof(
    *,
    protocol_path: Path | str = DEFAULT_PROTOCOL,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Verify all frozen hashes, population semantics, and report reconstruction."""

    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    output = Path(output_dir).resolve()
    protocol = _verified_json(protocol_file)
    validate_stacking_challenger_protocol(protocol, repository_root=root)
    manifest = _verified_json(output / "residual_v4_stacking_oof_manifest.json")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("stacking challenger manifest schema mismatch")
    if manifest.get("protocol") != _descriptor(protocol_file, root):
        raise ValueError("stacking challenger manifest protocol binding mismatch")
    if manifest.get("candidate_implementation") != (
        require_stacking_candidate_implementation_binding(protocol, repository_root=root)
    ):
        raise ValueError("stacking manifest implementation binding mismatch")
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
        raise ValueError("stacking manifest artifact set mismatch")
    dataset = _load_jsonl(artifacts["dataset_rows"])
    predictions = _load_jsonl(artifacts["predictions"])
    folds = _load_jsonl(artifacts["fold_audits"])
    markets = _load_jsonl(artifacts["market_results"])
    if len(dataset) != 3200 or any(
        row.get("schema_version") != DATASET_SCHEMA_VERSION
        or row.get("lineage_id") != LINEAGE_ID
        or row.get("development_only_forever") is not True
        or row.get("promotion_evidence_eligible") is not False
        or dict(row.get("safety") or {}) != SAFETY
        for row in dataset
    ):
        raise ValueError("stacking dataset governance mismatch")
    _validate_population(
        predictions=predictions,
        fold_audits=folds,
        market_results=markets,
        protocol=protocol,
    )
    rebuilt = _build_report(
        protocol=protocol,
        protocol_sha256=sha256_file(protocol_file),
        source_commit=str(manifest["source_commit"]),
        market_results=markets,
        fold_audits=folds,
    )
    frozen = _load_json(artifacts["report"])
    _assert_semantically_equal(rebuilt, frozen, path="residual_v4_stacking_report")
    if render_stacking_markdown(rebuilt) != artifacts["report_markdown"].read_text(
        encoding="utf-8"
    ):
        raise ValueError("stacking challenger Markdown does not reproduce")
    return {
        "verification_passed": True,
        "all_gates_passed": bool(frozen["all_gates_passed"]),
        "failed_gates": list(frozen["failed_gates"]),
        "candidate_budget_exhausted": True,
        "oof_market_count": len(markets),
        "manifest_sha256": sha256_file(output / "residual_v4_stacking_oof_manifest.json"),
        "actual_executing_module_binding_verified": True,
        "parent_v1_v2_v3_and_primary_immutable": True,
        "safety": dict(SAFETY),
    }


def fit_fixed_l2_logistic_stacker(
    feature_rows: Sequence[Sequence[float]],
    labels: Sequence[float],
    *,
    regularization: float = META_REGULARIZATION,
    max_iterations: int = META_MAX_ITERATIONS,
    tolerance: float = META_TOLERANCE,
) -> np.ndarray:
    """Fit a deterministic three-coefficient L2 logistic soft stacker."""

    matrix = np.asarray(feature_rows, dtype=float)
    target = np.asarray(labels, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 3 or matrix.shape[0] != target.size:
        raise ValueError("stacking meta design shape is invalid")
    if matrix.shape[0] == 0 or not np.all(np.isfinite(matrix)):
        raise ValueError("stacking meta design is empty or non-finite")
    if not np.all(np.isin(target, [0.0, 1.0])):
        raise ValueError("stacking meta labels must be binary")
    if regularization <= 0.0 or max_iterations <= 0 or tolerance <= 0.0:
        raise ValueError("stacking solver contract is invalid")
    beta = np.asarray([0.0, 0.5, 0.5], dtype=float)
    penalty = np.diag([0.0, regularization, regularization])
    for _ in range(max_iterations):
        probabilities = _sigmoid_vector(matrix @ beta)
        weights = np.maximum(probabilities * (1.0 - probabilities), 1e-12)
        gradient = matrix.T @ (probabilities - target) + penalty @ beta
        hessian = matrix.T @ (weights[:, None] * matrix) + penalty
        step = np.linalg.solve(hessian, gradient)
        updated = beta - step
        if not np.all(np.isfinite(updated)):
            raise ValueError("stacking solver produced non-finite coefficients")
        beta = updated
        if float(np.max(np.abs(step))) <= tolerance:
            break
    else:
        raise ValueError("stacking solver did not converge")
    return beta


def soft_stacking_action_values(
    rows: Sequence[Mapping[str, Any]],
    residual_actions: Sequence[Mapping[str, float]],
    logit_actions: Sequence[Mapping[str, float]],
    *,
    coefficients: Sequence[float],
) -> list[dict[str, float]]:
    """Apply the smooth stacker, pair-normalize probability, then subtract cost."""

    if not (len(rows) == len(residual_actions) == len(logit_actions)):
        raise ValueError("soft stacking row count mismatch")
    beta = np.asarray(coefficients, dtype=float)
    if beta.shape != (3,) or not np.all(np.isfinite(beta)):
        raise ValueError("soft stacking coefficients are invalid")
    grouped: dict[tuple[str, int], list[tuple[int, Mapping[str, Any], float]]] = defaultdict(list)
    raw_rows: list[dict[str, float]] = []
    for index, (row, residual, logit) in enumerate(
        zip(rows, residual_actions, logit_actions, strict=True)
    ):
        residual_probability = float(residual["predicted_probability"])
        logit_probability = float(logit["predicted_probability"])
        features = _meta_features(residual_probability, logit_probability)
        raw_probability = _sigmoid(float(np.asarray(features) @ beta))
        raw_rows.append(
            {
                "probability_residual_probability": residual_probability,
                "logit_offset_probability": logit_probability,
                "stacking_probability_before_pair_normalization": raw_probability,
            }
        )
        grouped[(str(row["market_id"]), int(row["decision_ts"]))].append(
            (index, row, raw_probability)
        )
    output: list[dict[str, float] | None] = [None] * len(rows)
    for key, members in grouped.items():
        if sorted(str(item[1]["side"]) for item in members) != ["DOWN", "UP"]:
            raise ValueError(f"soft stacking UP/DOWN pair is incomplete: {key}")
        denominator = math.fsum(item[2] for item in members)
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("soft stacking pair denominator is invalid")
        for index, row, raw_probability in members:
            probability = raw_probability / denominator
            cost = dict(row["cost_decomposition"])
            entry_ask = float(cost["entry_ask"])
            non_entry_cost = float(cost["total_cost_excluding_entry_ask"])
            action_value = probability - entry_ask - non_entry_cost
            item = dict(raw_rows[index])
            item.update(
                {
                    "predicted_probability": probability,
                    "entry_ask": entry_ask,
                    "non_entry_cost": non_entry_cost,
                    "action_value": action_value,
                }
            )
            if not all(math.isfinite(float(value)) for value in item.values()):
                raise ValueError("soft stacking action value is non-finite")
            output[index] = item
    if any(item is None for item in output):
        raise ValueError("soft stacking output is incomplete")
    return [dict(item) for item in output if item is not None]


def nested_rolling_origin_soft_stacking_predict(
    *,
    rows: Sequence[Mapping[str, Any]],
    population_order: Sequence[str],
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run nested base OOF calibration inside each outer rolling-origin fold."""

    rows_by_market: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_market[str(row["market_id"])].append(row)
    rolling = dict(protocol["rolling_origin"])
    initial = int(rolling["initial_training_market_count"])
    block_size = int(rolling["target_block_size"])
    block_count = int(rolling["target_block_count"])
    predictions: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for block_index in range(block_count):
        target_start = initial + block_index * block_size
        target_end = target_start + block_size
        training_ids = list(population_order[:target_start])
        target_ids = list(population_order[target_start:target_end])
        if len(target_ids) != block_size:
            raise ValueError("stacking target block population mismatch")
        meta_features, meta_labels, meta_audit = _nested_meta_training_data(
            training_ids=training_ids,
            rows_by_market=rows_by_market,
            protocol=protocol,
        )
        coefficients = fit_fixed_l2_logistic_stacker(meta_features, meta_labels)
        train_rows = [row for market_id in training_ids for row in rows_by_market[market_id]]
        target_rows = [row for market_id in target_ids for row in rows_by_market[market_id]]
        residual_actions, logit_actions = _fit_base_actions(
            train_rows=train_rows,
            target_rows=target_rows,
            protocol=protocol,
        )
        actions = soft_stacking_action_values(
            target_rows,
            residual_actions,
            logit_actions,
            coefficients=coefficients,
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
                    "probability_residual_probability": action["probability_residual_probability"],
                    "logit_offset_probability": action["logit_offset_probability"],
                    "stacking_probability_before_pair_normalization": action[
                        "stacking_probability_before_pair_normalization"
                    ],
                    "predicted_probability": action["predicted_probability"],
                    "stacking_coefficients": coefficients.tolist(),
                    "realized_unit_net_pnl_if_action": row["target"],
                    "resolved_outcome": row["resolved_outcome"],
                    "cost_decomposition": row["cost_decomposition"],
                    "feature_row_sha256": row["feature_row_sha256"],
                    "chronological_block": block_index + 1,
                    "strictly_prior_training_market_count": len(training_ids),
                    "nested_oof_training_market_count": meta_audit[
                        "nested_oof_training_market_count"
                    ],
                    "target_or_future_label_used_for_fit": False,
                    "target_or_future_label_used_for_stacker": False,
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
                "last_training_market_position": target_start,
                "first_target_market_position": target_start + 1,
                "target_or_future_label_leakage_count": 0,
                "nested_oof_training_market_count": meta_audit["nested_oof_training_market_count"],
                "nested_oof_side_decision_row_count": len(meta_labels),
                "nested_oof_market_ids_sha256": meta_audit["nested_oof_market_ids_sha256"],
                "meta_features_sha256": canonical_json_sha256(meta_features),
                "meta_labels_sha256": canonical_json_sha256(meta_labels),
                "stacking_coefficients": coefficients.tolist(),
                "stacking_coefficients_sha256": canonical_json_sha256(coefficients.tolist()),
                "meta_regularization": META_REGULARIZATION,
                "meta_max_iterations": META_MAX_ITERATIONS,
                "meta_tolerance": META_TOLERANCE,
                "current_or_future_outer_label_used_for_stacker_count": 0,
                "pair_coherence_applied_after_soft_stacking": True,
                "cost_subtraction_applied_after_pair_coherence": True,
                "development_only_forever": True,
                "promotion_evidence_eligible": False,
                "safety": dict(SAFETY),
            }
        )
    return predictions, audits


def render_stacking_markdown(report: Mapping[str, Any]) -> str:
    base = (
        render_residual_oof_markdown(report)
        .replace(
            "# BTC 15m cost-aware residual primary slot 001",
            "# BTC 15m nested soft-stacking residual v4 challenger slot 002",
            1,
        )
        .rstrip()
    )
    return (
        base
        + "\n\n## Architecture and terminal candidate budget\n\n"
        + "- Architecture: nested rolling-origin L2 logistic soft stacker over two pooled base learners.\n"
        + "- Meta inputs: base probability logits from strictly prior inner-OOF rows only.\n"
        + f"- Fixed L2 regularization: `{META_REGULARIZATION:g}`; parameter or threshold search: `False`\n"
        + "- Hard routing, side filters, missingness filters and outlier deletion: `False`\n"
        + "- Existing gates, zero threshold, N_max, costs, baseline and population changed: `False`\n"
        + "- Second and final v4 candidate slot consumed: `True`\n"
        + "- Collection, shadow, paper/live, wallet, write, promotion or capital authorized: `False`\n"
    )


def _nested_meta_training_data(
    *,
    training_ids: Sequence[str],
    rows_by_market: Mapping[str, Sequence[Mapping[str, Any]]],
    protocol: Mapping[str, Any],
) -> tuple[list[list[float]], list[float], dict[str, Any]]:
    if len(training_ids) < INNER_INITIAL_TRAINING_MARKETS + INNER_TARGET_BLOCK_MARKETS:
        raise ValueError("insufficient markets for nested stacking calibration")
    features: list[list[float]] = []
    labels: list[float] = []
    oof_ids: list[str] = []
    for target_start in range(
        INNER_INITIAL_TRAINING_MARKETS,
        len(training_ids),
        INNER_TARGET_BLOCK_MARKETS,
    ):
        target_end = target_start + INNER_TARGET_BLOCK_MARKETS
        inner_training_ids = list(training_ids[:target_start])
        inner_target_ids = list(training_ids[target_start:target_end])
        if len(inner_target_ids) != INNER_TARGET_BLOCK_MARKETS:
            raise ValueError("nested stacking inner block population mismatch")
        train_rows = [row for market_id in inner_training_ids for row in rows_by_market[market_id]]
        target_rows = [row for market_id in inner_target_ids for row in rows_by_market[market_id]]
        residual_actions, logit_actions = _fit_base_actions(
            train_rows=train_rows,
            target_rows=target_rows,
            protocol=protocol,
        )
        for row, residual, logit in zip(target_rows, residual_actions, logit_actions, strict=True):
            features.append(
                _meta_features(
                    float(residual["predicted_probability"]),
                    float(logit["predicted_probability"]),
                )
            )
            labels.append(_binary_payout_label(row))
        oof_ids.extend(inner_target_ids)
    if len(set(oof_ids)) != len(oof_ids):
        raise ValueError("nested stacking OOF market overlap")
    return (
        features,
        labels,
        {
            "nested_oof_training_market_count": len(oof_ids),
            "nested_oof_market_ids_sha256": canonical_json_sha256(oof_ids),
        },
    )


def _fit_base_actions(
    *,
    train_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    model = dict(protocol["model"])
    residual_spec = dict(model["base_learners"]["probability_residual"])
    logit_spec = dict(model["base_learners"]["logit_offset_binomial"])
    residual_labels = [_probability_residual_label(row) for row in train_rows]
    residual_booster = xgb.train(
        params=dict(residual_spec["parameters"]),
        dtrain=_residual_dmatrix(train_rows, labels=residual_labels),
        num_boost_round=int(residual_spec["fixed_num_boost_round"]),
        verbose_eval=False,
    )
    residual_predictions = [
        float(value)
        for value in residual_booster.predict(_residual_dmatrix(target_rows, labels=None))
    ]
    residual_actions = pair_anchored_action_values(target_rows, residual_predictions)
    binary_labels = [_binary_payout_label(row) for row in train_rows]
    logit_booster = xgb.train(
        params=dict(logit_spec["parameters"]),
        dtrain=_logit_dmatrix(train_rows, labels=binary_labels),
        num_boost_round=int(logit_spec["fixed_num_boost_round"]),
        verbose_eval=False,
    )
    logit_predictions = [
        float(value) for value in logit_booster.predict(_logit_dmatrix(target_rows, labels=None))
    ]
    return residual_actions, logit_offset_action_values(target_rows, logit_predictions)


def _meta_features(residual_probability: float, logit_probability: float) -> list[float]:
    return [
        1.0,
        _clipped_logit(residual_probability),
        _clipped_logit(logit_probability),
    ]


def _clipped_logit(probability: float) -> float:
    value = min(
        1.0 - PROBABILITY_CLIP_EPSILON,
        max(PROBABILITY_CLIP_EPSILON, float(probability)),
    )
    if not math.isfinite(value):
        raise ValueError("stacking base probability is invalid")
    return math.log(value / (1.0 - value))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _sigmoid_vector(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=float)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _market_results_from_predictions(
    *,
    predictions: Sequence[Mapping[str, Any]],
    baseline_by_market: Mapping[str, Mapping[str, Any]],
    population_order: Sequence[str],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    v1_rows = market_results_from_predictions(
        predictions=_as_v1_predictions(predictions),
        baseline_by_market=baseline_by_market,
        population_order=population_order,
        initial_training_market_count=int(
            protocol["rolling_origin"]["initial_training_market_count"]
        ),
        target_block_size=int(protocol["rolling_origin"]["target_block_size"]),
    )
    return [_replace_governance(row, MARKET_RESULT_SCHEMA_VERSION) for row in v1_rows]


def _build_report(
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    source_commit: str,
    market_results: Sequence[Mapping[str, Any]],
    fold_audits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    report = build_residual_oof_report(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        source_commit=source_commit,
        market_results=_as_v1_market_results(market_results),
        fold_audits=_as_v1_folds(fold_audits),
    )
    report["schema_version"] = REPORT_SCHEMA_VERSION
    report["lineage_id"] = LINEAGE_ID
    report["candidate_role"] = "challenger"
    report["architecture_type"] = (
        "nested_rolling_origin_l2_logistic_soft_stacker_over_two_pooled_"
        "side_symmetric_probability_learners"
    )
    report["immutable_gate_implementation_sha256"] = IMMUTABLE_GATE_IMPLEMENTATION["sha256"]
    report["actual_executing_module_exactly_bound"] = True
    report["existing_gate_threshold_cost_baseline_population_changed"] = False
    report["parent_v1_v2_v3_or_primary_failed_artifacts_changed"] = False
    report["candidate_budget_exhausted"] = True
    report["remaining_candidate_slots"] = 0
    report["additional_candidate_allowed"] = False
    report["next_stage_authorization_required_even_if_all_gates_pass"] = True
    report["structural_change"] = dict(protocol["structural_change"])
    report["nested_stacking"] = {
        "inner_oof_training_market_counts": [
            int(row["nested_oof_training_market_count"]) for row in fold_audits
        ],
        "stacking_coefficients": [list(row["stacking_coefficients"]) for row in fold_audits],
        "current_or_future_outer_label_used_for_stacker": False,
    }
    return report


def _validate_population(
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
        or row.get("target_or_future_label_used_for_stacker") is not False
        or dict(row.get("safety") or {}) != SAFETY
        for row in predictions
    ):
        raise ValueError("stacking prediction governance mismatch")
    if any(
        row.get("schema_version") != FOLD_SCHEMA_VERSION
        or row.get("lineage_id") != LINEAGE_ID
        or row.get("current_or_future_outer_label_used_for_stacker_count") != 0
        or dict(row.get("safety") or {}) != SAFETY
        for row in fold_audits
    ):
        raise ValueError("stacking fold governance mismatch")
    if any(
        row.get("schema_version") != MARKET_RESULT_SCHEMA_VERSION
        or row.get("lineage_id") != LINEAGE_ID
        or dict(row.get("safety") or {}) != SAFETY
        for row in market_results
    ):
        raise ValueError("stacking market governance mismatch")
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[(str(row["market_id"]), int(row["decision_ts"]))].append(row)
        cost = dict(row["cost_decomposition"])
        expected = (
            float(row["predicted_probability"])
            - float(cost["entry_ask"])
            - float(cost["total_cost_excluding_entry_ask"])
        )
        if not math.isclose(expected, float(row["prediction"]), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("stacking action-value reconciliation failed")
    for rows in grouped.values():
        if sorted(str(row["side"]) for row in rows) != ["DOWN", "UP"]:
            raise ValueError("stacking pair side mismatch")
        if not math.isclose(
            math.fsum(float(row["predicted_probability"]) for row in rows),
            1.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("stacking pair probability does not sum to one")
    _validate_frozen_population(
        predictions=_as_v1_predictions(predictions),
        fold_audits=_as_v1_folds(fold_audits),
        market_results=_as_v1_market_results(market_results),
        protocol=protocol,
    )


def _public_dataset_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = _public_dataset_v4_row(row)
    if output.get("schema_version") != PRIMARY_DATASET_SCHEMA_VERSION:
        raise ValueError("primary v4 dataset schema drift")
    output["schema_version"] = DATASET_SCHEMA_VERSION
    return output


def _as_v1_predictions(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_replace_governance(row, V1_PREDICTION_SCHEMA_VERSION, V1_LINEAGE_ID) for row in rows]


def _as_v1_folds(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_replace_governance(row, V1_FOLD_SCHEMA_VERSION, V1_LINEAGE_ID) for row in rows]


def _as_v1_market_results(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        _replace_governance(row, V1_MARKET_RESULT_SCHEMA_VERSION, V1_LINEAGE_ID) for row in rows
    ]


def _replace_governance(
    row: Mapping[str, Any], schema_version: str, lineage_id: str = LINEAGE_ID
) -> dict[str, Any]:
    output = deepcopy(dict(row))
    output["schema_version"] = schema_version
    output["lineage_id"] = lineage_id
    return output


def _candidate_budget_contract() -> dict[str, Any]:
    return {
        "maximum_total_slots": 2,
        "this_slot_ordinal": 2,
        "slots_consumed_before_run": 1,
        "slots_remaining_after_run": 0,
        "slot_budget_may_be_increased": False,
    }


def _target_contract() -> dict[str, Any]:
    return {
        "action_value_formula": (
            "pair_normalized_nested_soft_stacking_probability-entry_ask-frozen_"
            "fees-slippage-liquidity_impact"
        ),
        "execution_policy": "HOLD_TO_SETTLEMENT",
        "NO_TRADE_value": 0.0,
        "post_close_training_label_only": True,
        "base_labels": {
            "probability_residual": "settlement_payout-selected_mid",
            "logit_offset_binomial": "settlement_payout",
        },
        "meta_label": "settlement_payout",
    }


def _pair_contract() -> dict[str, Any]:
    return {
        "base_probabilities_pair_normalized_before_meta_fit": True,
        "stacking_probability_pair_normalized_before_cost": True,
        "cost_subtraction_happens_after_pair_normalization": True,
        "probability_clip_epsilon": PROBABILITY_CLIP_EPSILON,
        "missing_anchor_behavior": "fail_closed_NO_TRADE_in_runtime",
        "missing_values_encoded_as_zero": False,
    }


def _model_contract() -> dict[str, Any]:
    from bigan.v8.polymarket.cost_aware_residual_v4 import _model_contract as primary

    base = primary()
    return {
        "family": "nested_rolling_origin_l2_logistic_soft_stacking",
        "route_or_expert_filtering_allowed": False,
        "model_selection_or_early_stopping_performed": False,
        "base_learners": deepcopy(base["base_learners"]),
        "meta_learner": {
            "family": "l2_penalized_binomial_logistic_regression",
            "features": [
                "intercept",
                "logit_probability_residual_probability",
                "logit_logit_offset_probability",
            ],
            "regularization": META_REGULARIZATION,
            "intercept_penalized": False,
            "solver": "deterministic_newton_raphson",
            "max_iterations": META_MAX_ITERATIONS,
            "tolerance": META_TOLERANCE,
        },
    }


def _nested_contract() -> dict[str, Any]:
    return {
        "outer_method": "frozen_six_block_rolling_origin",
        "inner_method": "expanding_window_market_grouped_rolling_origin",
        "inner_initial_training_market_count": INNER_INITIAL_TRAINING_MARKETS,
        "inner_target_block_market_count": INNER_TARGET_BLOCK_MARKETS,
        "inner_oof_predictions_only_for_meta_fit": True,
        "outer_target_or_future_label_used_for_meta_fit": False,
        "base_and_meta_parameter_search_allowed": False,
        "hard_routing_or_expert_selection_allowed": False,
    }


def _discipline_contract() -> dict[str, Any]:
    return {
        "one_candidate_this_slot": True,
        "hyperparameter_search_allowed": False,
        "meta_regularization_search_allowed": False,
        "threshold_search_allowed": False,
        "feature_search_allowed": False,
        "route_side_missingness_or_outlier_filtering_allowed": False,
        "post_result_mutation_allowed": False,
        "additional_candidate_allowed": False,
    }


def _state_contract() -> dict[str, Any]:
    return {
        "training_started": False,
        "candidate_frozen": False,
        "live_shadow_started": False,
        "fresh_collection_started": False,
        "fresh_outcomes_opened": False,
        "promotion_started": False,
    }


__all__ = [
    "fit_fixed_l2_logistic_stacker",
    "nested_rolling_origin_soft_stacking_predict",
    "require_stacking_candidate_implementation_binding",
    "run_stacking_challenger_oof",
    "soft_stacking_action_values",
    "validate_stacking_challenger_protocol",
    "verify_frozen_stacking_challenger_oof",
]
