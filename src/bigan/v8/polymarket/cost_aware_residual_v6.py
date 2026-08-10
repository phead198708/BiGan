"""Nested fitted-Q dynamic stopping candidate for residual lineage v6."""

from __future__ import annotations

import hashlib
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
    _rolling_contract,
)
from bigan.v8.polymarket.cost_aware_residual_v4 import (
    IMMUTABLE_GATE_IMPLEMENTATION,
    _baseline_contract,
    _dataset_contract,
)
from bigan.v8.polymarket.cost_aware_residual_v4 import (
    _feature_contract as _parent_feature_contract,
)
from bigan.v8.polymarket.moe_collection_boundary_r2 import (
    _write_new_frozen_json,
    _write_new_jsonl,
)
from bigan.v8.polymarket.moe_collection_observability import FEATURE_NAMES
from bigan.v8.polymarket.moe_confirmatory_evaluation import _write_new_frozen_text
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.moe_terminal_diagnostic import _assert_semantically_equal
from bigan.v8.polymarket.regime_adaptive_lineage import REPO_ROOT

LINEAGE_ID = "BTC-15M-cost-aware-market-residual-v6"
PARENT_LINEAGE_ID = "BTC-15M-cost-aware-market-residual-v5"
PROTOCOL_SCHEMA_VERSION = "bigan-btc-15m-dynamic-stopping-oof-protocol-v6"
PREDICTION_SCHEMA_VERSION = "bigan-btc-15m-dynamic-stopping-oof-prediction-v6"
FOLD_SCHEMA_VERSION = "bigan-btc-15m-dynamic-stopping-oof-fold-v6"
MARKET_RESULT_SCHEMA_VERSION = "bigan-btc-15m-dynamic-stopping-market-result-v6"
REPORT_SCHEMA_VERSION = "bigan-btc-15m-dynamic-stopping-oof-report-v6"
MANIFEST_SCHEMA_VERSION = "bigan-btc-15m-dynamic-stopping-oof-manifest-v6"
DATASET_SCHEMA_VERSION = "bigan-btc-15m-dynamic-stopping-dataset-row-v6"

DEFAULT_CONFIG_DIR = (
    REPO_ROOT / "examples/v8/polymarket_configs/BTC-15M-cost-aware-market-residual-v6"
)
DEFAULT_PROTOCOL = DEFAULT_CONFIG_DIR / "residual_v6_primary_slot_001_protocol.json"
DEFAULT_OUTPUT_DIR = DEFAULT_CONFIG_DIR / "residual_v6_primary_slot_001_oof"
DEFAULT_AUTHORIZATION = DEFAULT_CONFIG_DIR / "lineage_authorization.json"
DEFAULT_REGISTRY = DEFAULT_CONFIG_DIR / "development_data_registry.json"

AUTHORIZATION_INSTRUCTION = "给予授权"
AUTHORIZATION_INSTRUCTION_SHA256 = (
    "3bb787244b497dd0f55976140c3d562380fabc863704b74c81ed79deaf4180da"
)
AUTHORIZED_REQUEST_SCOPE = [
    "authorize_BTC-15M-cost-aware-market-residual-v6_with_at_most_two_preregistered_candidate_slots",
    "authorize_a_narrow_dependency-closed_integration_chain_from_current_main",
]
PARENT_V5_TERMINAL_SHA256 = "22b4cd829c3f042088ab9e37a855653869bc8d7069da8d2bfd60f3b90719e472"
FIXED_NUM_BOOST_ROUND = 128
FIXED_PARAMETERS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.03,
    "max_depth": 2,
    "min_child_weight": 16.0,
    "gamma": 0.0,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 20.0,
    "tree_method": "hist",
    "max_bin": 64,
    "seed": 26461,
    "nthread": 1,
}
INNER_INITIAL_TRAINING_MARKETS = 100
INNER_TARGET_BLOCK_MARKETS = 100

STRUCTURAL_CHANGE = {
    "changed_component": "sequential_decision_target_and_training_graph",
    "from": "score_each_decision_as_an_independent_immediate_action",
    "to": (
        "nested_fitted_Q_optimal_stopping_with_late_direct_value_and_early_"
        "incremental_value_over_strictly_prior_inner_OOF_continuation_policy"
    ),
    "reason": (
        "v1_through_v5_treated_early_and_late_actions_independently;_a_positive_early_"
        "score_could_trade_without_pricing_the_option_to_wait_for_the_late_decision"
    ),
    "expected_mechanism": (
        "the unchanged zero threshold now means act early only when predicted early value "
        "exceeds a cross-fitted late continuation policy; otherwise preserve the option to "
        "wait and use a direct after-cost late action value"
    ),
    "threshold_changed": False,
    "cost_baseline_population_or_gate_changed": False,
    "route_side_missingness_or_outlier_filter_added": False,
    "parameter_feature_weight_or_threshold_search_performed": False,
}


def validate_v6_lineage_authorization(
    *,
    authorization_path: Path | str = DEFAULT_AUTHORIZATION,
    registry_path: Path | str = DEFAULT_REGISTRY,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Verify the scoped reply authorization and immutable v5 boundary."""

    root = Path(repository_root).resolve()
    authorization_file = Path(authorization_path).resolve()
    registry_file = Path(registry_path).resolve()
    authorization = _verified_json(authorization_file)
    registry = _verified_json(registry_file)
    blockers: list[str] = []
    source = dict(authorization.get("authorization_source") or {})
    if authorization.get("schema_version") != (
        "bigan-btc-15m-cost-aware-residual-lineage-authorization-v6"
    ):
        blockers.append("authorization.schema_version")
    if source.get("type") != "explicit_user_reply_to_immediately_preceding_scope":
        blockers.append("authorization_source.type")
    if source.get("instruction") != AUTHORIZATION_INSTRUCTION:
        blockers.append("authorization_source.instruction")
    if _raw_text_sha256(str(source.get("instruction") or "")) != (
        AUTHORIZATION_INSTRUCTION_SHA256
    ):
        blockers.append("authorization_source.instruction_sha256")
    if source.get("instruction_sha256") != AUTHORIZATION_INSTRUCTION_SHA256:
        blockers.append("authorization_source.recorded_sha256")
    if source.get("immediately_preceding_requested_scope") != AUTHORIZED_REQUEST_SCOPE:
        blockers.append("authorization_source.requested_scope")
    if authorization.get("lineage_id") != LINEAGE_ID:
        blockers.append("authorization.lineage_id")
    if dict(authorization.get("authorization_scope") or {}) != _authorization_scope():
        blockers.append("authorization_scope")
    if dict(authorization.get("state") or {}) != _authorization_state_contract():
        blockers.append("authorization_state")
    if dict(authorization.get("safety") or {}) != SAFETY:
        blockers.append("authorization.safety")
    parent = dict(authorization.get("parent_lineage") or {})
    if not (
        parent.get("lineage_id") == PARENT_LINEAGE_ID
        and parent.get("status") == "phase_1_terminal_failed"
        and parent.get("candidate_budget_consumed") == 2
        and parent.get("candidate_budget_maximum") == 2
        and parent.get("failed_artifacts_mutable") is False
        and parent.get("gate_or_threshold_change_allowed") is False
    ):
        blockers.append("parent_lineage")
    try:
        terminal_path = _verify_descriptor(dict(parent["terminal_review"]), repository_root=root)
        terminal = _load_json(terminal_path)
        if not (
            sha256_file(terminal_path) == PARENT_V5_TERMINAL_SHA256
            and terminal.get("lineage_id") == PARENT_LINEAGE_ID
            and terminal.get("phase_1_terminal_failed") is True
            and terminal.get("candidate_budget_exhausted") is True
            and terminal.get("candidate_selected") is None
            and terminal.get("candidate_freeze_allowed") is False
            and dict(terminal.get("safety") or {}) == SAFETY
        ):
            blockers.append("parent_terminal_semantics")
    except (KeyError, OSError, TypeError, ValueError):
        blockers.append("parent_terminal_review")
    try:
        registered = _verify_descriptor(
            dict(authorization["registered_development_data"]), repository_root=root
        )
        if registered != registry_file:
            blockers.append("registered_development_data.path")
    except (KeyError, OSError, TypeError, ValueError):
        blockers.append("registered_development_data")
    if not (
        registry.get("lineage_id") == LINEAGE_ID
        and registry.get("development_only_forever") is True
        and registry.get("promotion_evidence_eligible") is False
        and dict(registry.get("safety") or {}) == SAFETY
    ):
        blockers.append("development_data_registry")
    for name, descriptor in dict(registry.get("registered_sources") or {}).items():
        try:
            _verify_descriptor(dict(descriptor), repository_root=root)
        except (KeyError, OSError, TypeError, ValueError):
            blockers.append(f"development_data_registry.{name}")
    if blockers:
        raise ValueError("residual v6 authorization invalid: " + ", ".join(blockers))
    return {
        "authorization_valid": True,
        "lineage_id": LINEAGE_ID,
        "maximum_total_slots": 2,
        "parent_v5_immutable": True,
        "narrow_main_integration_preparation_authorized": True,
        "fresh_collection_or_execution_authorized": False,
        "authorization_sha256": sha256_file(authorization_file),
        "registry_sha256": sha256_file(registry_file),
        "safety": dict(SAFETY),
    }


def require_v6_candidate_implementation_binding(
    payload: Mapping[str, Any], *, repository_root: Path | str = REPO_ROOT
) -> dict[str, str]:
    """Bind the slot protocol to the exact executing module."""

    root = Path(repository_root).resolve()
    expected = _descriptor(Path(__file__), root)
    try:
        declared = dict(payload["inputs"]["candidate_implementation"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("v6 candidate implementation descriptor unavailable") from exc
    if declared != expected:
        raise ValueError("v6 candidate implementation does not identify the executing module")
    if _verify_descriptor(declared, repository_root=root) != Path(__file__).resolve():
        raise ValueError("v6 candidate implementation resolved to another module")
    return expected


def validate_residual_v6_protocol(
    payload: Mapping[str, Any],
    *,
    repository_root: Path | str = REPO_ROOT,
    verify_artifacts: bool = True,
) -> None:
    """Fail closed unless the first v6 candidate is fully preregistered."""

    blockers: list[str] = []
    scalars = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "slot_id": "residual-v6-primary-slot-001",
        "candidate_role": "primary",
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
    }
    for field, expected in scalars.items():
        if payload.get(field) != expected:
            blockers.append(field)
    contracts = {
        "candidate_budget": _candidate_budget_contract(),
        "target": _target_contract(),
        "feature_contract": _feature_contract(),
        "model": _model_contract(),
        "sequential_training": _sequential_contract(),
        "structural_change": STRUCTURAL_CHANGE,
        "action_policy": _action_policy(),
        "bootstrap": _bootstrap_contract(),
        "cost_stress": _cost_stress_contract(),
        "prospective_power": _power_contract(),
        "rolling_origin": _rolling_contract(),
        "dataset": _dataset_contract(),
        "baseline": _baseline_contract(),
        "development_discipline": _discipline_contract(),
        "state": _state_contract(),
        "safety": SAFETY,
    }
    for field, expected in contracts.items():
        if dict(payload.get(field) or {}) != expected:
            blockers.append(field)
    gates = dict(payload.get("gates") or {})
    if set(gates) != GATE_NAMES or any(value is not True for value in gates.values()):
        blockers.append("gates")
    root = Path(repository_root).resolve()
    try:
        require_v6_candidate_implementation_binding(payload, repository_root=root)
    except ValueError:
        blockers.append("candidate_implementation_exact_binding")
    inputs = dict(payload.get("inputs") or {})
    required_inputs = {
        "lineage_authorization",
        "development_data_registry",
        "parent_v5_terminal_review",
        "parent_v4_development_dataset_rows",
        "parent_v4_market_results",
        "matched_global_baseline_contract",
        "parent_feature_contract",
        "parent_cost_and_action_contract",
        "candidate_implementation",
        "gate_implementation",
    }
    if set(inputs) != required_inputs:
        blockers.append("inputs")
    if dict(inputs.get("gate_implementation") or {}) != IMMUTABLE_GATE_IMPLEMENTATION:
        blockers.append("gate_implementation")
    if verify_artifacts and not blockers:
        resolved: dict[str, Path] = {}
        for name, descriptor in inputs.items():
            try:
                resolved[name] = _verify_descriptor(dict(descriptor), repository_root=root)
            except (KeyError, OSError, TypeError, ValueError):
                blockers.append(f"inputs.{name}")
        if not blockers:
            try:
                validate_v6_lineage_authorization(
                    authorization_path=resolved["lineage_authorization"],
                    registry_path=resolved["development_data_registry"],
                    repository_root=root,
                )
            except ValueError:
                blockers.append("lineage_authorization")
            terminal = _load_json(resolved["parent_v5_terminal_review"])
            if not (
                terminal.get("phase_1_terminal_failed") is True
                and terminal.get("candidate_budget_exhausted") is True
                and terminal.get("candidate_selected") is None
                and dict(terminal.get("safety") or {}) == SAFETY
            ):
                blockers.append("parent_v5_terminal_boundary")
    if blockers:
        raise ValueError("residual v6 protocol invalid: " + ", ".join(blockers))


def run_residual_v6_rolling_origin_oof(
    *,
    protocol_path: Path | str = DEFAULT_PROTOCOL,
    expected_protocol_sha256: str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    source_commit: str,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Execute the first preregistered v6 candidate exactly once."""

    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    output = Path(output_dir).resolve()
    if not protocol_file.is_relative_to(root) or not output.is_relative_to(root):
        raise ValueError("residual v6 OOF paths must remain repository-local")
    if sha256_file(protocol_file) != expected_protocol_sha256.lower():
        raise ValueError("residual v6 protocol SHA-256 mismatch")
    sidecar = protocol_file.with_suffix(".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != (
        expected_protocol_sha256
    ):
        raise ValueError("residual v6 protocol is not SHA-frozen")
    if not _looks_like_git_sha(source_commit):
        raise ValueError("source_commit must be a full lowercase Git SHA")
    protocol = _load_json(protocol_file)
    validate_residual_v6_protocol(protocol, repository_root=root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"residual v6 OOF output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    rows, parent_markets, population_order = _load_registered_parent_rows(
        protocol=protocol, repository_root=root
    )
    rows = [_canonicalize_row(row) for row in rows]
    predictions, folds = nested_dynamic_stopping_predict(
        rows=rows, population_order=population_order, protocol=protocol
    )
    markets = _market_results_from_predictions(
        predictions=predictions,
        baseline_by_market=_baseline_rows(parent_markets),
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

    dataset_path = output / "residual_v6_development_dataset_rows.jsonl"
    prediction_path = output / "residual_v6_oof_predictions.jsonl"
    fold_path = output / "residual_v6_oof_fold_audits.jsonl"
    market_path = output / "residual_v6_oof_market_results.jsonl"
    report_path = output / "residual_v6_oof_report.json"
    markdown_path = output / "residual_v6_oof_report.md"
    _write_new_jsonl(dataset_path, [_public_dataset_row(row) for row in rows])
    _write_new_jsonl(prediction_path, predictions)
    _write_new_jsonl(fold_path, folds)
    _write_new_jsonl(market_path, markets)
    report_artifact = _write_new_frozen_json(report_path, report)
    markdown_artifact = _write_new_frozen_text(markdown_path, render_v6_markdown(report))
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "slot_id": protocol["slot_id"],
        "candidate_role": "primary",
        "created_at": protocol["created_at"],
        "source_commit": source_commit,
        "protocol": _descriptor(protocol_file, root),
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
        "remaining_candidate_slots": 1,
        "candidate_freeze_allowed": report["all_gates_passed"],
        "next_stage_authorization_required_even_if_all_gates_pass": True,
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "safety": dict(SAFETY),
    }
    manifest_artifact = _write_new_frozen_json(output / "residual_v6_oof_manifest.json", manifest)
    return {
        "manifest": _descriptor(Path(manifest_artifact["path"]), root),
        "report": _descriptor(Path(report_artifact["path"]), root),
        "all_gates_passed": report["all_gates_passed"],
        "failed_gates": report["failed_gates"],
        "remaining_candidate_slots": 1,
        "oof_market_count": len(markets),
        "next_stage_authorization_required": True,
        "safety": dict(SAFETY),
    }


def verify_frozen_residual_v6_oof(
    *,
    protocol_path: Path | str = DEFAULT_PROTOCOL,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Rebuild the frozen v6 report and verify all artifact bindings."""

    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    output = Path(output_dir).resolve()
    protocol = _verified_json(protocol_file)
    validate_residual_v6_protocol(protocol, repository_root=root)
    manifest = _verified_json(output / "residual_v6_oof_manifest.json")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("residual v6 manifest schema mismatch")
    if manifest.get("protocol") != _descriptor(protocol_file, root):
        raise ValueError("residual v6 manifest protocol binding mismatch")
    if manifest.get("candidate_implementation") != require_v6_candidate_implementation_binding(
        protocol, repository_root=root
    ):
        raise ValueError("residual v6 manifest implementation binding mismatch")
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
        raise ValueError("residual v6 manifest artifact set mismatch")
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
        raise ValueError("residual v6 dataset governance mismatch")
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
    _assert_semantically_equal(rebuilt, frozen, path="residual_v6_oof_report")
    if render_v6_markdown(rebuilt) != artifacts["report_markdown"].read_text(encoding="utf-8"):
        raise ValueError("residual v6 Markdown does not reproduce")
    return {
        "verification_passed": True,
        "all_gates_passed": bool(frozen["all_gates_passed"]),
        "failed_gates": list(frozen["failed_gates"]),
        "remaining_candidate_slots": 1,
        "oof_market_count": len(markets),
        "manifest_sha256": sha256_file(output / "residual_v6_oof_manifest.json"),
        "actual_executing_module_binding_verified": True,
        "parent_v1_through_v5_immutable": True,
        "safety": dict(SAFETY),
    }


def nested_dynamic_stopping_predict(
    *,
    rows: Sequence[Mapping[str, Any]],
    population_order: Sequence[str],
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run nested fitted-Q training with no outer target labels used for fit."""

    rows_by_market: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_market[str(row["market_id"])].append(row)
    rolling = dict(protocol["rolling_origin"])
    sequential = dict(protocol["sequential_training"])
    initial = int(rolling["initial_training_market_count"])
    block_size = int(rolling["target_block_size"])
    block_count = int(rolling["target_block_count"])
    inner_initial = int(sequential["inner_initial_training_market_count"])
    inner_block = int(sequential["inner_target_block_market_count"])
    predictions: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for block_index in range(block_count):
        target_start = initial + block_index * block_size
        target_end = target_start + block_size
        training_ids = list(population_order[:target_start])
        target_ids = list(population_order[target_start:target_end])
        if len(target_ids) != block_size:
            raise ValueError("v6 outer target block population mismatch")
        early_training_rows, early_labels, continuation_audit = _inner_continuation_training(
            training_ids=training_ids,
            rows_by_market=rows_by_market,
            protocol=protocol,
            inner_initial=inner_initial,
            inner_block=inner_block,
        )
        train_rows = _ordered_rows(training_ids, rows_by_market)
        target_rows = _ordered_rows(target_ids, rows_by_market)
        train_early, train_late = _split_stages(train_rows)
        target_early, target_late = _split_stages(target_rows)
        late_model = _fit_model(train_late, [float(row["target"]) for row in train_late], protocol)
        early_model = _fit_model(early_training_rows, early_labels, protocol)
        early_scores = _predict_model(early_model, target_early)
        late_scores = _predict_model(late_model, target_late)
        score_by_key = {
            _row_key(row): (score, "early_incremental_over_continuation")
            for row, score in zip(target_early, early_scores, strict=True)
        }
        score_by_key.update(
            {
                _row_key(row): (score, "late_direct_after_cost")
                for row, score in zip(target_late, late_scores, strict=True)
            }
        )
        for row in target_rows:
            score, semantics = score_by_key[_row_key(row)]
            predictions.append(
                {
                    "schema_version": PREDICTION_SCHEMA_VERSION,
                    "lineage_id": LINEAGE_ID,
                    "slot_id": protocol["slot_id"],
                    "market_id": row["market_id"],
                    "market_start_ts": row["market_start_ts"],
                    "decision_ts": row["decision_ts"],
                    "side": row["side"],
                    "prediction": score,
                    "score_semantics": semantics,
                    "realized_unit_net_pnl_if_action": row["target"],
                    "resolved_outcome": row["resolved_outcome"],
                    "cost_decomposition": row["cost_decomposition"],
                    "feature_row_sha256": row["feature_row_sha256"],
                    "chronological_block": block_index + 1,
                    "strictly_prior_training_market_count": len(training_ids),
                    "nested_continuation_training_market_count": continuation_audit[
                        "market_count"
                    ],
                    "target_or_future_label_used_for_fit": False,
                    "outer_target_late_feature_used_for_early_score": False,
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
                "nested_continuation_training_market_count": continuation_audit[
                    "market_count"
                ],
                "nested_continuation_market_ids_sha256": continuation_audit[
                    "market_ids_sha256"
                ],
                "nested_continuation_labels_sha256": continuation_audit["labels_sha256"],
                "early_training_side_row_count": len(early_training_rows),
                "late_training_side_row_count": len(train_late),
                "outer_target_late_feature_used_for_early_score_count": 0,
                "fixed_num_boost_round": int(protocol["model"]["fixed_num_boost_round"]),
                "model_parameters_sha256": canonical_json_sha256(
                    dict(protocol["model"]["parameters"])
                ),
                "development_only_forever": True,
                "promotion_evidence_eligible": False,
                "safety": dict(SAFETY),
            }
        )
    return predictions, audits


def _inner_continuation_training(
    *,
    training_ids: Sequence[str],
    rows_by_market: Mapping[str, Sequence[Mapping[str, Any]]],
    protocol: Mapping[str, Any],
    inner_initial: int,
    inner_block: int,
) -> tuple[list[Mapping[str, Any]], list[float], dict[str, Any]]:
    if len(training_ids) < inner_initial + inner_block:
        raise ValueError("insufficient markets for nested continuation training")
    early_rows: list[Mapping[str, Any]] = []
    labels: list[float] = []
    oof_ids: list[str] = []
    continuation_values: list[float] = []
    for target_start in range(inner_initial, len(training_ids), inner_block):
        inner_target_ids = list(training_ids[target_start : target_start + inner_block])
        if len(inner_target_ids) != inner_block:
            raise ValueError("nested continuation inner block population mismatch")
        inner_train_rows = _ordered_rows(training_ids[:target_start], rows_by_market)
        inner_target_rows = _ordered_rows(inner_target_ids, rows_by_market)
        _, inner_train_late = _split_stages(inner_train_rows)
        inner_target_early, inner_target_late = _split_stages(inner_target_rows)
        late_model = _fit_model(
            inner_train_late,
            [float(row["target"]) for row in inner_train_late],
            protocol,
        )
        late_predictions = _predict_model(late_model, inner_target_late)
        continuation_by_market = _realized_continuation_values(
            inner_target_late, late_predictions, inner_target_ids
        )
        for row in inner_target_early:
            continuation = continuation_by_market[str(row["market_id"])]
            early_rows.append(row)
            labels.append(float(row["target"]) - continuation)
        continuation_values.extend(continuation_by_market[market_id] for market_id in inner_target_ids)
        oof_ids.extend(inner_target_ids)
    if len(set(oof_ids)) != len(oof_ids):
        raise ValueError("nested continuation OOF market overlap")
    return (
        early_rows,
        labels,
        {
            "market_count": len(oof_ids),
            "market_ids_sha256": canonical_json_sha256(oof_ids),
            "labels_sha256": canonical_json_sha256(labels),
            "continuation_values_sha256": canonical_json_sha256(continuation_values),
        },
    )


def _realized_continuation_values(
    late_rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[float],
    market_order: Sequence[str],
) -> dict[str, float]:
    grouped: dict[str, list[tuple[Mapping[str, Any], float]]] = defaultdict(list)
    for row, prediction in zip(late_rows, predictions, strict=True):
        grouped[str(row["market_id"])].append((row, float(prediction)))
    output: dict[str, float] = {}
    for market_id in market_order:
        actions = grouped.get(market_id, [])
        if [str(row["side"]) for row, _ in actions] != ["UP", "DOWN"]:
            raise ValueError("nested continuation late UP/DOWN pair mismatch")
        best_row, best_score = max(
            actions,
            key=lambda item: (item[1], 0 if item[0]["side"] == "UP" else -1),
        )
        output[market_id] = float(best_row["target"]) if best_score > 0.0 else 0.0
    return output


def _fit_model(
    rows: Sequence[Mapping[str, Any]], labels: Sequence[float], protocol: Mapping[str, Any]
) -> xgb.Booster:
    return xgb.train(
        params=dict(protocol["model"]["parameters"]),
        dtrain=_dmatrix(rows, labels=labels),
        num_boost_round=int(protocol["model"]["fixed_num_boost_round"]),
        verbose_eval=False,
    )


def _predict_model(model: xgb.Booster, rows: Sequence[Mapping[str, Any]]) -> list[float]:
    values = [float(value) for value in model.predict(_dmatrix(rows, labels=None))]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("v6 model produced a non-finite prediction")
    return values


def _dmatrix(rows: Sequence[Mapping[str, Any]], *, labels: Sequence[float] | None) -> xgb.DMatrix:
    values = np.vstack([np.asarray(row["features"], dtype=float) for row in rows])
    target = None if labels is None else np.asarray(labels, dtype=float)
    if target is not None and target.shape != (len(rows),):
        raise ValueError("v6 labels do not align with feature rows")
    return xgb.DMatrix(
        values,
        label=target,
        feature_names=list(FEATURE_NAMES),
        missing=np.nan,
    )


def _split_stages(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    by_market: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_market[str(row["market_id"])].append(row)
    early: list[Mapping[str, Any]] = []
    late: list[Mapping[str, Any]] = []
    for market_rows in by_market.values():
        timestamps = sorted({int(row["decision_ts"]) for row in market_rows})
        if len(timestamps) != 2:
            raise ValueError("v6 requires exactly two decisions per market")
        early.extend(row for row in market_rows if int(row["decision_ts"]) == timestamps[0])
        late.extend(row for row in market_rows if int(row["decision_ts"]) == timestamps[1])
    return early, late


def _ordered_rows(
    market_ids: Sequence[str], rows_by_market: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[Mapping[str, Any]]:
    output: list[Mapping[str, Any]] = []
    for market_id in market_ids:
        rows = sorted(
            rows_by_market.get(market_id, []),
            key=lambda row: (int(row["decision_ts"]), 0 if row["side"] == "UP" else 1),
        )
        if len(rows) != 4 or [str(row["side"]) for row in rows] != ["UP", "DOWN", "UP", "DOWN"]:
            raise ValueError("v6 market action grid changed")
        output.extend(rows)
    return output


def _canonicalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = deepcopy(dict(row))
    features = dict(row["features"])
    if set(features) != set(FEATURE_NAMES) or len(features) != len(FEATURE_NAMES):
        raise ValueError("v6 inherited feature names changed")
    output["features"] = np.asarray(
        [np.nan if features[name] is None else float(features[name]) for name in FEATURE_NAMES],
        dtype=float,
    )
    return output


def _row_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return str(row["market_id"]), int(row["decision_ts"]), str(row["side"])


def _load_registered_parent_rows(
    *, protocol: Mapping[str, Any], repository_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Load only the two frozen parent artifacts used by the v6 executor."""

    inputs = dict(protocol["inputs"])
    dataset = _load_jsonl(
        _verify_descriptor(
            inputs["parent_v4_development_dataset_rows"], repository_root=repository_root
        )
    )
    markets = _load_jsonl(
        _verify_descriptor(inputs["parent_v4_market_results"], repository_root=repository_root)
    )
    positions: dict[str, int] = {}
    for row in dataset:
        market_id = str(row["market_id"])
        position = int(row["market_position"])
        if market_id in positions and positions[market_id] != position:
            raise ValueError("v6 parent dataset market position mismatch")
        positions[market_id] = position
    population_order = [item[0] for item in sorted(positions.items(), key=lambda item: item[1])]
    if len(dataset) != 3200 or len(population_order) != 800:
        raise ValueError("v6 parent dataset population changed")
    expected_oof_ids = set(population_order[int(protocol["rolling_origin"]["initial_training_market_count"]) :])
    if len(markets) != 600 or {str(row["market_id"]) for row in markets} != expected_oof_ids:
        raise ValueError("v6 parent market-result population changed")
    return dataset, markets, population_order


def _baseline_rows(parent_markets: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row["market_id"]): {
            "baseline_accepted": bool(row["baseline_accepted"]),
            "baseline_selected_side": row["baseline_selected_side"],
            "decision_ts": int(row["baseline_decision_ts"]),
            "baseline_unit_net_pnl": float(row["baseline_unit_net_pnl"]),
            "cost_decomposition": {
                "baseline": {"total_cost": float(row["baseline_total_cost_relative_to_mid"])}
            },
        }
        for row in parent_markets
    }


def _market_results_from_predictions(
    *,
    predictions: Sequence[Mapping[str, Any]],
    baseline_by_market: Mapping[str, Mapping[str, Any]],
    population_order: Sequence[str],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = market_results_from_predictions(
        predictions=_as_v1_predictions(predictions),
        baseline_by_market=baseline_by_market,
        population_order=population_order,
        initial_training_market_count=int(
            protocol["rolling_origin"]["initial_training_market_count"]
        ),
        target_block_size=int(protocol["rolling_origin"]["target_block_size"]),
    )
    return [_replace_governance(row, MARKET_RESULT_SCHEMA_VERSION) for row in rows]


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
    report["architecture_type"] = (
        "nested_fitted_Q_dynamic_optimal_stopping_with_cross_fitted_late_continuation"
    )
    report["immutable_gate_implementation_sha256"] = IMMUTABLE_GATE_IMPLEMENTATION["sha256"]
    report["actual_executing_module_exactly_bound"] = True
    report["existing_gate_threshold_cost_baseline_population_changed"] = False
    report["parent_v1_through_v5_failed_artifacts_changed"] = False
    report["remaining_candidate_slots"] = 1
    report["additional_candidate_requires_preregistration"] = True
    report["next_stage_authorization_required_even_if_all_gates_pass"] = True
    report["structural_change"] = dict(protocol["structural_change"])
    report["sequential_training"] = {
        "nested_continuation_training_market_counts": [
            int(row["nested_continuation_training_market_count"]) for row in fold_audits
        ],
        "outer_target_late_feature_used_for_early_score": False,
        "current_or_future_outer_label_used_for_fit": False,
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
        or row.get("outer_target_late_feature_used_for_early_score") is not False
        or dict(row.get("safety") or {}) != SAFETY
        for row in predictions
    ):
        raise ValueError("v6 prediction governance mismatch")
    if any(
        row.get("schema_version") != FOLD_SCHEMA_VERSION
        or row.get("lineage_id") != LINEAGE_ID
        or row.get("target_or_future_label_leakage_count") != 0
        or row.get("outer_target_late_feature_used_for_early_score_count") != 0
        or dict(row.get("safety") or {}) != SAFETY
        for row in fold_audits
    ):
        raise ValueError("v6 fold governance mismatch")
    if any(
        row.get("schema_version") != MARKET_RESULT_SCHEMA_VERSION
        or row.get("lineage_id") != LINEAGE_ID
        or dict(row.get("safety") or {}) != SAFETY
        for row in market_results
    ):
        raise ValueError("v6 market governance mismatch")
    _validate_frozen_population(
        predictions=_as_v1_predictions(predictions),
        fold_audits=_as_v1_folds(fold_audits),
        market_results=_as_v1_market_results(market_results),
        protocol=protocol,
    )


def _public_dataset_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = deepcopy(dict(row))
    values = np.asarray(row["features"], dtype=float)
    output["features"] = {
        name: (float(value) if math.isfinite(float(value)) else None)
        for name, value in zip(FEATURE_NAMES, values, strict=True)
    }
    output["schema_version"] = DATASET_SCHEMA_VERSION
    output["lineage_id"] = LINEAGE_ID
    return output


def render_v6_markdown(report: Mapping[str, Any]) -> str:
    base = (
        render_residual_oof_markdown(report)
        .replace(
            "# BTC 15m cost-aware residual primary slot 001",
            "# BTC 15m dynamic optimal-stopping residual v6 primary slot 001",
            1,
        )
        .rstrip()
    )
    return (
        base
        + "\n\n## Sequential architecture\n\n"
        + "- Late decision: direct after-cost action-value model.\n"
        + "- Early decision: incremental value over a strictly prior, inner-OOF late continuation policy.\n"
        + "- Outer target late features used for early scoring: `False`\n"
        + "- Grid, feature, weight or threshold search: `False`\n"
        + "- Existing gates, zero threshold, N_max, costs, baseline and population changed: `False`\n"
        + "- Collection, shadow, paper/live, wallet, write, promotion or capital authorized: `False`\n"
    )


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


def _raw_text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _authorization_scope() -> dict[str, Any]:
    return {
        "candidate_slot_budget": {
            "maximum_total_slots": 2,
            "slot_budget_may_be_increased": False,
        },
        "actual_executing_module_path_and_sha_binding_required": True,
        "fresh_collection_authorized": False,
        "fresh_outcome_opening_authorized": False,
        "live_shadow_authorized": False,
        "narrow_main_integration_preparation_authorized": True,
        "outcome_aware_development_authorized": True,
        "paper_or_live_execution_authorized": False,
        "promotion_authorized": False,
        "training_may_start_only_after_slot_protocol_is_sha_frozen": True,
        "wallet_or_write_authorized": False,
    }


def _candidate_budget_contract() -> dict[str, Any]:
    return {
        "maximum_total_slots": 2,
        "this_slot_ordinal": 1,
        "slots_consumed_before_run": 0,
        "slots_remaining_after_run": 1,
        "slot_budget_may_be_increased": False,
    }


def _target_contract() -> dict[str, Any]:
    return {
        "execution_policy": "HOLD_TO_SETTLEMENT",
        "NO_TRADE_value": 0.0,
        "post_close_training_labels_only": True,
        "late_target": "direct_after_cost_action_value",
        "late_formula": (
            "settlement_payout-executable_ask-frozen_fees-slippage-liquidity_impact"
        ),
        "continuation_target": "realized_PnL_of_strictly_inner_OOF_late_policy_or_zero",
        "early_target": "early_direct_after_cost_action_value-continuation_target",
        "early_score_semantics": "incremental_value_of_acting_now_instead_of_waiting",
    }


def _feature_contract() -> dict[str, Any]:
    parent = _parent_feature_contract()
    return {
        **parent,
        "early_model_uses_early_decision_features_only": True,
        "late_model_uses_late_decision_features_only": True,
        "same_108_feature_contract_at_both_stages": True,
        "new_raw_feature_added": False,
    }


def _model_contract() -> dict[str, Any]:
    stage = {
        "family": "pooled_side_symmetric_xgboost_regressor",
        "fixed_num_boost_round": FIXED_NUM_BOOST_ROUND,
        "parameters": dict(FIXED_PARAMETERS),
    }
    return {
        "family": "nested_fitted_Q_dynamic_optimal_stopping",
        "late_direct_model": deepcopy(stage),
        "early_incremental_model": deepcopy(stage),
        "route_or_expert_filtering_allowed": False,
        "model_selection_or_early_stopping_performed": False,
        "parameters": dict(FIXED_PARAMETERS),
        "fixed_num_boost_round": FIXED_NUM_BOOST_ROUND,
    }


def _sequential_contract() -> dict[str, Any]:
    return {
        "outer_method": "frozen_six_block_market_grouped_rolling_origin",
        "inner_method": "expanding_market_grouped_rolling_origin_for_late_continuation",
        "inner_initial_training_market_count": INNER_INITIAL_TRAINING_MARKETS,
        "inner_target_block_market_count": INNER_TARGET_BLOCK_MARKETS,
        "late_continuation_action_policy": "unchanged_zero_threshold_UP_first_tie_break",
        "inner_OOF_continuation_only_for_early_training": True,
        "outer_target_late_features_used_for_early_score": False,
        "current_or_future_outer_label_used_for_fit": False,
        "parameter_feature_weight_or_threshold_search_allowed": False,
    }


def _discipline_contract() -> dict[str, Any]:
    return {
        "one_candidate_this_slot": True,
        "hyperparameter_search_allowed": False,
        "feature_search_allowed": False,
        "weight_search_allowed": False,
        "threshold_search_allowed": False,
        "route_side_missingness_or_outlier_filtering_allowed": False,
        "post_result_mutation_allowed": False,
        "challenger_requires_separate_preregistration": True,
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


def _authorization_state_contract() -> dict[str, Any]:
    return {
        "candidate_frozen": False,
        "fresh_collection_started": False,
        "fresh_outcomes_opened": False,
        "lineage_authorized_for_governed_development": True,
        "live_shadow_started": False,
        "promotion_started": False,
        "training_started": False,
    }


__all__ = [
    "nested_dynamic_stopping_predict",
    "require_v6_candidate_implementation_binding",
    "run_residual_v6_rolling_origin_oof",
    "validate_residual_v6_protocol",
    "validate_v6_lineage_authorization",
    "verify_frozen_residual_v6_oof",
]
