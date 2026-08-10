"""Canonical-order execution adapter for the second and final v5 slot."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from bigan.v8.polymarket import cost_aware_residual_v5 as primary
from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.cost_aware_residual import (
    _descriptor,
    _load_json,
    _load_jsonl,
    _looks_like_git_sha,
    _verified_json,
    _verify_descriptor,
    build_residual_oof_report,
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
from bigan.v8.polymarket.moe_collection_boundary_r2 import (
    _write_new_frozen_json,
    _write_new_jsonl,
)
from bigan.v8.polymarket.moe_collection_observability import FEATURE_NAMES
from bigan.v8.polymarket.moe_confirmatory_evaluation import _write_new_frozen_text
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.moe_terminal_diagnostic import _assert_semantically_equal
from bigan.v8.polymarket.regime_adaptive_lineage import REPO_ROOT

LINEAGE_ID = primary.LINEAGE_ID
PROTOCOL_SCHEMA_VERSION = "bigan-btc-15m-prequential-market-residual-oof-protocol-v5-slot2"
PREDICTION_SCHEMA_VERSION = "bigan-btc-15m-prequential-market-residual-oof-prediction-v5-slot2"
FOLD_SCHEMA_VERSION = "bigan-btc-15m-prequential-market-residual-oof-fold-v5-slot2"
MARKET_RESULT_SCHEMA_VERSION = "bigan-btc-15m-prequential-market-residual-result-v5-slot2"
REPORT_SCHEMA_VERSION = "bigan-btc-15m-prequential-market-residual-oof-report-v5-slot2"
MANIFEST_SCHEMA_VERSION = "bigan-btc-15m-prequential-market-residual-oof-manifest-v5-slot2"
DATASET_SCHEMA_VERSION = "bigan-btc-15m-prequential-market-residual-dataset-row-v5-slot2"

DEFAULT_CONFIG_DIR = primary.DEFAULT_CONFIG_DIR
DEFAULT_PROTOCOL = DEFAULT_CONFIG_DIR / "residual_v5_challenger_slot_002_protocol.json"
DEFAULT_OUTPUT_DIR = DEFAULT_CONFIG_DIR / "residual_v5_challenger_slot_002_oof"

PRIMARY_IMPLEMENTATION = {
    "path": "src/bigan/v8/polymarket/cost_aware_residual_v5.py",
    "sha256": "daf9da7efeba997a077e21bca5b600aed4fe1c52b2bd37c966a7592d680080cb",
}
PRIMARY_FAILURE = {
    "path": (
        "examples/v8/polymarket_configs/BTC-15M-cost-aware-market-residual-v5/"
        "residual_v5_primary_slot_001_execution_failure.json"
    ),
    "sha256": "7fdf02f4b42c84ff2667f630876e32c6c7db9f94b12b160bb35345079e3a51a3",
}

STRUCTURAL_CHANGE = {
    "changed_component": "serialized_feature_object_to_semantic_feature_vector_adapter",
    "from": "require_json_object_insertion_order_to_equal_FEATURE_NAMES",
    "to": "resolve_each_feature_explicitly_in_frozen_FEATURE_NAMES_order",
    "reason": (
        "slot_1_failed_closed_before_model_fit_because_canonical_JSON_serialization_"
        "orders_object_keys_lexically"
    ),
    "candidate_algorithm_changed": False,
    "model_parameters_changed": False,
    "threshold_changed": False,
    "cost_baseline_population_or_gate_changed": False,
    "route_side_missingness_or_outlier_filter_added": False,
    "parameter_weight_feature_or_threshold_search_performed": False,
}


def require_challenger_implementation_binding(
    payload: Mapping[str, Any], *, repository_root: Path | str = REPO_ROOT
) -> dict[str, str]:
    """Bind the second-slot protocol to this exact adapter module."""

    root = Path(repository_root).resolve()
    expected = _descriptor(Path(__file__), root)
    try:
        declared = dict(payload["inputs"]["candidate_implementation"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("v5 challenger implementation descriptor unavailable") from exc
    if declared != expected:
        raise ValueError("v5 challenger implementation does not identify the executing module")
    if _verify_descriptor(declared, repository_root=root) != Path(__file__).resolve():
        raise ValueError("v5 challenger implementation resolved to another module")
    return expected


def validate_challenger_protocol(
    payload: Mapping[str, Any],
    *,
    repository_root: Path | str = REPO_ROOT,
    verify_artifacts: bool = True,
) -> None:
    """Verify the final v5 slot preserves the slot-1 candidate semantics exactly."""

    blockers: list[str] = []
    expected_scalars = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "slot_id": "residual-v5-challenger-slot-002",
        "candidate_role": "challenger",
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
    }
    for field, expected in expected_scalars.items():
        if payload.get(field) != expected:
            blockers.append(field)
    expected_contracts = {
        "candidate_budget": _candidate_budget_contract(),
        "target": primary._target_contract(),
        "pair_coherence": primary._pair_contract(),
        "feature_contract": primary._feature_contract(),
        "model": primary._model_contract(),
        "prequential_correction": primary._prequential_contract(),
        "structural_change": STRUCTURAL_CHANGE,
        "action_policy": _action_policy(),
        "bootstrap": _bootstrap_contract(),
        "cost_stress": _cost_stress_contract(),
        "prospective_power": _power_contract(),
        "rolling_origin": _rolling_contract(),
        "dataset": _dataset_contract(),
        "baseline": _baseline_contract(),
        "development_discipline": _discipline_contract(),
        "state": primary._state_contract(),
        "safety": SAFETY,
        "prior_slot_failure": _prior_slot_failure_contract(),
    }
    for field, expected in expected_contracts.items():
        if dict(payload.get(field) or {}) != expected:
            blockers.append(field)
    gates = dict(payload.get("gates") or {})
    if set(gates) != GATE_NAMES or any(value is not True for value in gates.values()):
        blockers.append("gates")
    root = Path(repository_root).resolve()
    try:
        require_challenger_implementation_binding(payload, repository_root=root)
    except ValueError:
        blockers.append("candidate_implementation_exact_binding")
    inputs = dict(payload.get("inputs") or {})
    required_inputs = {
        "lineage_authorization",
        "development_data_registry",
        "parent_v4_terminal_review",
        "parent_v4_challenger_protocol",
        "parent_v4_challenger_manifest",
        "parent_v4_challenger_report",
        "parent_v4_challenger_dataset_rows",
        "parent_v4_challenger_predictions",
        "parent_v4_challenger_market_results",
        "matched_global_baseline_contract",
        "parent_feature_contract",
        "parent_cost_and_action_contract",
        "candidate_implementation",
        "primary_candidate_implementation",
        "primary_slot_execution_failure",
        "parent_v4_stacking_implementation",
        "gate_implementation",
    }
    if set(inputs) != required_inputs:
        blockers.append("inputs")
    if dict(inputs.get("gate_implementation") or {}) != IMMUTABLE_GATE_IMPLEMENTATION:
        blockers.append("gate_implementation")
    if dict(inputs.get("parent_v4_stacking_implementation") or {}) != (
        primary.PARENT_V4_STACKING_IMPLEMENTATION
    ):
        blockers.append("parent_v4_stacking_implementation")
    if dict(inputs.get("primary_candidate_implementation") or {}) != PRIMARY_IMPLEMENTATION:
        blockers.append("primary_candidate_implementation")
    if dict(inputs.get("primary_slot_execution_failure") or {}) != PRIMARY_FAILURE:
        blockers.append("primary_slot_execution_failure")
    if verify_artifacts and not blockers:
        resolved: dict[str, Path] = {}
        for name, descriptor in inputs.items():
            try:
                resolved[name] = _verify_descriptor(dict(descriptor), repository_root=root)
            except (KeyError, OSError, TypeError, ValueError):
                blockers.append(f"inputs.{name}")
        if not blockers:
            try:
                primary.validate_v5_lineage_authorization(
                    authorization_path=resolved["lineage_authorization"],
                    registry_path=resolved["development_data_registry"],
                    repository_root=root,
                )
            except ValueError:
                blockers.append("lineage_authorization")
            failure = _load_json(resolved["primary_slot_execution_failure"])
            if not (
                failure.get("slot_id") == "residual-v5-primary-slot-001"
                and failure.get("failure", {}).get("fail_closed") is True
                and failure.get("evaluation", {}).get("candidate_metrics_computed") is False
                and failure.get("evaluation", {}).get("gate_evaluation_performed") is False
                and failure.get("candidate_budget", {}).get("slot_1_consumed") is True
                and failure.get("candidate_budget", {}).get("remaining_candidate_slots") == 1
                and dict(failure.get("safety") or {}) == SAFETY
            ):
                blockers.append("primary_slot_failure_semantics")
    if blockers:
        raise ValueError("residual v5 challenger protocol invalid: " + ", ".join(blockers))


def run_challenger_oof(
    *,
    protocol_path: Path | str = DEFAULT_PROTOCOL,
    expected_protocol_sha256: str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    source_commit: str,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Execute the second and final v5 candidate exactly once."""

    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    output = Path(output_dir).resolve()
    if not protocol_file.is_relative_to(root) or not output.is_relative_to(root):
        raise ValueError("v5 challenger paths must remain repository-local")
    if sha256_file(protocol_file) != expected_protocol_sha256.lower():
        raise ValueError("v5 challenger protocol SHA-256 mismatch")
    sidecar = protocol_file.with_suffix(".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != (
        expected_protocol_sha256
    ):
        raise ValueError("v5 challenger protocol is not SHA-frozen")
    if not _looks_like_git_sha(source_commit):
        raise ValueError("source_commit must be a full lowercase Git SHA")
    protocol = _load_json(protocol_file)
    validate_challenger_protocol(protocol, repository_root=root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"v5 challenger output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    rows, base_predictions, parent_markets, population_order = primary._load_frozen_v4_rows(
        protocol=protocol, repository_root=root
    )
    canonical_rows = [_canonicalize_feature_order(row) for row in rows]
    predictions, folds = primary.prequential_market_residual_corrector_predict(
        rows=canonical_rows,
        frozen_base_predictions=base_predictions,
        population_order=population_order,
        protocol=protocol,
    )
    predictions = [_replace_schema(row, PREDICTION_SCHEMA_VERSION) for row in predictions]
    folds = [_replace_schema(row, FOLD_SCHEMA_VERSION) for row in folds]
    markets = primary._market_results_from_predictions(
        predictions=predictions,
        baseline_by_market=primary._baseline_rows(parent_markets),
        population_order=population_order,
        protocol=protocol,
    )
    markets = [_replace_schema(row, MARKET_RESULT_SCHEMA_VERSION) for row in markets]
    report = _build_report(
        protocol=protocol,
        protocol_sha256=expected_protocol_sha256,
        source_commit=source_commit,
        market_results=markets,
        fold_audits=folds,
    )

    dataset_path = output / "residual_v5_challenger_development_dataset_rows.jsonl"
    prediction_path = output / "residual_v5_challenger_oof_predictions.jsonl"
    fold_path = output / "residual_v5_challenger_oof_fold_audits.jsonl"
    market_path = output / "residual_v5_challenger_oof_market_results.jsonl"
    report_path = output / "residual_v5_challenger_oof_report.json"
    markdown_path = output / "residual_v5_challenger_oof_report.md"
    _write_new_jsonl(dataset_path, [_public_dataset_row(row) for row in canonical_rows])
    _write_new_jsonl(prediction_path, predictions)
    _write_new_jsonl(fold_path, folds)
    _write_new_jsonl(market_path, markets)
    report_artifact = _write_new_frozen_json(report_path, report)
    markdown_artifact = _write_new_frozen_text(markdown_path, render_challenger_markdown(report))
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "slot_id": protocol["slot_id"],
        "candidate_role": "challenger",
        "created_at": protocol["created_at"],
        "source_commit": source_commit,
        "protocol": _descriptor(protocol_file, root),
        "prior_slot_failure": dict(protocol["prior_slot_failure"]),
        "candidate_implementation": dict(protocol["inputs"]["candidate_implementation"]),
        "primary_candidate_implementation": dict(
            protocol["inputs"]["primary_candidate_implementation"]
        ),
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
        output / "residual_v5_challenger_oof_manifest.json", manifest
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


def verify_frozen_challenger_oof(
    *,
    protocol_path: Path | str = DEFAULT_PROTOCOL,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Reconcile every challenger artifact and recompute its immutable report."""

    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    output = Path(output_dir).resolve()
    protocol = _verified_json(protocol_file)
    validate_challenger_protocol(protocol, repository_root=root)
    manifest = _verified_json(output / "residual_v5_challenger_oof_manifest.json")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("v5 challenger manifest schema mismatch")
    if manifest.get("protocol") != _descriptor(protocol_file, root):
        raise ValueError("v5 challenger manifest protocol binding mismatch")
    if manifest.get("candidate_implementation") != require_challenger_implementation_binding(
        protocol, repository_root=root
    ):
        raise ValueError("v5 challenger manifest implementation binding mismatch")
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
        raise ValueError("v5 challenger manifest artifact set mismatch")
    dataset = _load_jsonl(artifacts["dataset_rows"])
    predictions = _load_jsonl(artifacts["predictions"])
    folds = _load_jsonl(artifacts["fold_audits"])
    markets = _load_jsonl(artifacts["market_results"])
    if len(dataset) != 3200 or any(
        row.get("schema_version") != DATASET_SCHEMA_VERSION
        or row.get("lineage_id") != LINEAGE_ID
        or dict(row.get("safety") or {}) != SAFETY
        for row in dataset
    ):
        raise ValueError("v5 challenger dataset governance mismatch")
    primary._validate_population(
        predictions=[_replace_schema(row, primary.PREDICTION_SCHEMA_VERSION) for row in predictions],
        fold_audits=[_replace_schema(row, primary.FOLD_SCHEMA_VERSION) for row in folds],
        market_results=[
            _replace_schema(row, primary.MARKET_RESULT_SCHEMA_VERSION) for row in markets
        ],
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
    _assert_semantically_equal(rebuilt, frozen, path="residual_v5_challenger_report")
    if render_challenger_markdown(rebuilt) != artifacts["report_markdown"].read_text(
        encoding="utf-8"
    ):
        raise ValueError("v5 challenger Markdown does not reproduce")
    return {
        "verification_passed": True,
        "all_gates_passed": bool(frozen["all_gates_passed"]),
        "failed_gates": list(frozen["failed_gates"]),
        "candidate_budget_exhausted": True,
        "oof_market_count": len(markets),
        "manifest_sha256": sha256_file(output / "residual_v5_challenger_oof_manifest.json"),
        "actual_executing_module_binding_verified": True,
        "slot_1_failure_and_parent_v1_through_v4_immutable": True,
        "safety": dict(SAFETY),
    }


def _canonicalize_feature_order(row: Mapping[str, Any]) -> dict[str, Any]:
    output = deepcopy(dict(row))
    raw = dict(row["features"])
    if set(raw) != set(FEATURE_NAMES) or len(raw) != len(FEATURE_NAMES):
        raise ValueError("v5 challenger inherited feature names changed")
    output["features"] = {name: raw[name] for name in FEATURE_NAMES}
    return output


def _replace_schema(row: Mapping[str, Any], schema_version: str) -> dict[str, Any]:
    output = deepcopy(dict(row))
    output["schema_version"] = schema_version
    output["lineage_id"] = LINEAGE_ID
    return output


def _public_dataset_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = deepcopy(dict(row))
    output["schema_version"] = DATASET_SCHEMA_VERSION
    output["lineage_id"] = LINEAGE_ID
    return output


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
        market_results=primary._as_v1_market_results(market_results),
        fold_audits=primary._as_v1_folds(fold_audits),
    )
    report["schema_version"] = REPORT_SCHEMA_VERSION
    report["lineage_id"] = LINEAGE_ID
    report["candidate_role"] = "challenger"
    report["architecture_type"] = (
        "frozen_v4_nested_soft_stacking_with_expanding_prequential_market_level_"
        "probability_residual_corrector"
    )
    report["immutable_gate_implementation_sha256"] = IMMUTABLE_GATE_IMPLEMENTATION["sha256"]
    report["actual_executing_module_exactly_bound"] = True
    report["canonical_feature_order_adapter_applied"] = True
    report["candidate_algorithm_or_parameters_changed_from_slot_1"] = False
    report["existing_gate_threshold_cost_baseline_population_changed"] = False
    report["parent_v1_v2_v3_v4_or_slot_1_failure_artifacts_changed"] = False
    report["candidate_budget_exhausted"] = True
    report["remaining_candidate_slots"] = 0
    report["additional_candidate_allowed"] = False
    report["next_stage_authorization_required_even_if_all_gates_pass"] = True
    report["structural_change"] = dict(protocol["structural_change"])
    report["prequential_correction"] = {
        "strictly_prior_corrector_training_market_counts": [
            int(row["strictly_prior_corrector_training_market_count"])
            for row in fold_audits
        ],
        "first_block_frozen_v4_identity_fallback": True,
        "current_or_future_label_used_for_corrector": False,
    }
    return report


def render_challenger_markdown(report: Mapping[str, Any]) -> str:
    base = (
        render_residual_oof_markdown(report)
        .replace(
            "# BTC 15m cost-aware residual primary slot 001",
            "# BTC 15m prequential market-residual v5 challenger slot 002",
            1,
        )
        .rstrip()
    )
    return (
        base
        + "\n\n## Final slot and execution adapter\n\n"
        + "- Candidate algorithm and fixed parameters changed from slot 1: `False`\n"
        + "- Adapter: feature values are resolved explicitly in frozen FEATURE_NAMES order.\n"
        + "- Grid, feature, weight or threshold search: `False`\n"
        + "- Existing gates, zero threshold, N_max, costs, baseline and population changed: `False`\n"
        + "- Second and final v5 candidate slot consumed: `True`\n"
        + "- Collection, outcome opening, shadow, paper/live, wallet, write, promotion or capital authorized: `False`\n"
    )


def _candidate_budget_contract() -> dict[str, Any]:
    return {
        "maximum_total_slots": 2,
        "this_slot_ordinal": 2,
        "slots_consumed_before_run": 1,
        "slots_remaining_after_run": 0,
        "slot_budget_may_be_increased": False,
    }


def _prior_slot_failure_contract() -> dict[str, Any]:
    return {
        "artifact": dict(PRIMARY_FAILURE),
        "candidate_metrics_computed": False,
        "gate_evaluation_performed": False,
        "failure_stage": "corrector_feature_schema_validation_before_first_model_fit",
        "slot_consumed": True,
    }


def _discipline_contract() -> dict[str, Any]:
    return {
        "one_candidate_this_slot": True,
        "hyperparameter_search_allowed": False,
        "weight_search_allowed": False,
        "threshold_search_allowed": False,
        "feature_search_allowed": False,
        "route_side_missingness_or_outlier_filtering_allowed": False,
        "post_result_mutation_allowed": False,
        "additional_candidate_allowed": False,
    }


__all__ = [
    "require_challenger_implementation_binding",
    "run_challenger_oof",
    "validate_challenger_protocol",
    "verify_frozen_challenger_oof",
]
