"""Governed prequential market-level residual corrector for lineage v5."""

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

LINEAGE_ID = "BTC-15M-cost-aware-market-residual-v5"
PARENT_LINEAGE_ID = "BTC-15M-cost-aware-market-residual-v4"
PROTOCOL_SCHEMA_VERSION = "bigan-btc-15m-prequential-market-residual-oof-protocol-v5"
PREDICTION_SCHEMA_VERSION = "bigan-btc-15m-prequential-market-residual-oof-prediction-v5"
FOLD_SCHEMA_VERSION = "bigan-btc-15m-prequential-market-residual-oof-fold-v5"
MARKET_RESULT_SCHEMA_VERSION = "bigan-btc-15m-prequential-market-residual-result-v5"
REPORT_SCHEMA_VERSION = "bigan-btc-15m-prequential-market-residual-oof-report-v5"
MANIFEST_SCHEMA_VERSION = "bigan-btc-15m-prequential-market-residual-oof-manifest-v5"
DATASET_SCHEMA_VERSION = "bigan-btc-15m-prequential-market-residual-dataset-row-v5"

DEFAULT_CONFIG_DIR = (
    REPO_ROOT / "examples/v8/polymarket_configs/BTC-15M-cost-aware-market-residual-v5"
)
DEFAULT_PROTOCOL = DEFAULT_CONFIG_DIR / "residual_v5_primary_slot_001_protocol.json"
DEFAULT_OUTPUT_DIR = DEFAULT_CONFIG_DIR / "residual_v5_primary_slot_001_oof"
DEFAULT_AUTHORIZATION = DEFAULT_CONFIG_DIR / "lineage_authorization.json"
DEFAULT_REGISTRY = DEFAULT_CONFIG_DIR / "development_data_registry.json"

AUTHORIZATION_INSTRUCTION = (
    "授权建立 BTC-15M-cost-aware-market-residual-v5，最多 2 个预注册候选 slot，并保持"
    "既有 gates、零阈值、N_max=2000、成本、基线、人口、失败 artifacts 和全部安全状态不变。"
)
AUTHORIZATION_INSTRUCTION_SHA256 = (
    "ea656aa81e9a1c0f29ff397a047427517168159c3800461fe82feb74f16d5869"
)
PARENT_V4_TERMINAL_SHA256 = "85102e916f7304090ab203fb4d0128f88aa5157fd71f2ec4dec80e4ad8abbd74"
PARENT_V4_STACKING_IMPLEMENTATION = {
    "path": "src/bigan/v8/polymarket/cost_aware_residual_v4_stacking.py",
    "sha256": "904175ff2a2272e41a903bf53798ed72df32d3dffa076e9350d7dcfb09c9ae60",
}

BASE_FEATURE_NAMES = tuple(FEATURE_NAMES)
CORRECTOR_CONTEXT_FEATURE_NAMES = (
    "frozen_v4_stacked_probability",
    "frozen_v4_action_value",
    "entry_ask",
    "total_cost_excluding_entry_ask",
)
CORRECTOR_FEATURE_NAMES = BASE_FEATURE_NAMES + CORRECTOR_CONTEXT_FEATURE_NAMES
PAIR_CLIP_EPSILON = 1e-6
FIXED_NUM_BOOST_ROUND = 96
FIXED_PARAMETERS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.025,
    "max_depth": 2,
    "min_child_weight": 32.0,
    "gamma": 0.01,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 40.0,
    "tree_method": "hist",
    "max_bin": 64,
    "seed": 26451,
    "nthread": 1,
}

STRUCTURAL_CHANGE = {
    "changed_component": "market_level_after_cost_probability_residual_correction",
    "from": "frozen_v4_nested_soft_stacking_probability_used_directly",
    "to": (
        "expanding_prequential_xgboost_residual_correction_trained_only_on_strictly_"
        "prior_frozen_v4_oof_rows"
    ),
    "reason": (
        "v4_slot_2_passed_every_non_power_gate_but_required_2488_markets;_its_score_"
        "deciles_were_non_monotonic_and_market_level_variance_was_too_high"
    ),
    "expected_mechanism": (
        "a_fixed_strongly_regularized_corrector uses the unchanged 108 causal features, "
        "the frozen v4 probability and frozen costs to learn strictly prior probability "
        "residuals; pair normalization preserves coherence before unchanged cost subtraction"
    ),
    "threshold_changed": False,
    "cost_baseline_population_or_gate_changed": False,
    "route_side_missingness_or_outlier_filter_added": False,
    "parameter_weight_or_threshold_search_performed": False,
}


def validate_v5_lineage_authorization(
    *,
    authorization_path: Path | str = DEFAULT_AUTHORIZATION,
    registry_path: Path | str = DEFAULT_REGISTRY,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Verify the exact v5 authorization and immutable v4 terminal boundary."""

    root = Path(repository_root).resolve()
    authorization_file = Path(authorization_path).resolve()
    registry_file = Path(registry_path).resolve()
    authorization = _verified_json(authorization_file)
    registry = _verified_json(registry_file)
    blockers: list[str] = []
    source = dict(authorization.get("authorization_source") or {})
    if authorization.get("schema_version") != (
        "bigan-btc-15m-cost-aware-residual-lineage-authorization-v5"
    ):
        blockers.append("authorization.schema_version")
    if source.get("type") != "explicit_user_instruction":
        blockers.append("authorization_source.type")
    if source.get("instruction") != AUTHORIZATION_INSTRUCTION:
        blockers.append("authorization_source.instruction")
    if _raw_text_sha256(str(source.get("instruction") or "")) != (
        AUTHORIZATION_INSTRUCTION_SHA256
    ):
        blockers.append("authorization_source.instruction_sha256")
    if source.get("instruction_sha256") != AUTHORIZATION_INSTRUCTION_SHA256:
        blockers.append("authorization_source.recorded_sha256")
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
            sha256_file(terminal_path) == PARENT_V4_TERMINAL_SHA256
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
        and dict(registry.get("rules") or {}).get(
            "candidate_evaluation_requires_actual_executing_module_path_and_sha_binding"
        )
        is True
    ):
        blockers.append("development_data_registry")
    for name, descriptor in dict(registry.get("registered_sources") or {}).items():
        try:
            _verify_descriptor(dict(descriptor), repository_root=root)
        except (KeyError, OSError, TypeError, ValueError):
            blockers.append(f"development_data_registry.{name}")
    if blockers:
        raise ValueError("residual v5 authorization invalid: " + ", ".join(blockers))
    return {
        "authorization_valid": True,
        "lineage_id": LINEAGE_ID,
        "maximum_total_slots": 2,
        "parent_v4_immutable": True,
        "actual_executing_module_binding_required": True,
        "authorization_sha256": sha256_file(authorization_file),
        "registry_sha256": sha256_file(registry_file),
        "safety": dict(SAFETY),
    }


def require_v5_candidate_implementation_binding(
    payload: Mapping[str, Any], *, repository_root: Path | str = REPO_ROOT
) -> dict[str, str]:
    """Bind a slot protocol to these exact executing bytes."""

    root = Path(repository_root).resolve()
    expected = _descriptor(Path(__file__), root)
    try:
        declared = dict(payload["inputs"]["candidate_implementation"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("v5 candidate implementation descriptor unavailable") from exc
    if declared != expected:
        raise ValueError("v5 candidate implementation does not identify the executing module")
    if _verify_descriptor(declared, repository_root=root) != Path(__file__).resolve():
        raise ValueError("v5 candidate implementation resolved to another module")
    return expected


def validate_residual_v5_protocol(
    payload: Mapping[str, Any],
    *,
    repository_root: Path | str = REPO_ROOT,
    verify_artifacts: bool = True,
) -> None:
    """Fail closed unless slot 1 and every inherited byte are frozen."""

    blockers: list[str] = []
    expected_scalars = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "slot_id": "residual-v5-primary-slot-001",
        "candidate_role": "primary",
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
    }
    for field, expected in expected_scalars.items():
        if payload.get(field) != expected:
            blockers.append(field)
    expected_contracts = {
        "candidate_budget": _candidate_budget_contract(),
        "target": _target_contract(),
        "pair_coherence": _pair_contract(),
        "feature_contract": _feature_contract(),
        "model": _model_contract(),
        "prequential_correction": _prequential_contract(),
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
    for field, expected in expected_contracts.items():
        if dict(payload.get(field) or {}) != expected:
            blockers.append(field)
    gates = dict(payload.get("gates") or {})
    if set(gates) != GATE_NAMES or any(value is not True for value in gates.values()):
        blockers.append("gates")
    root = Path(repository_root).resolve()
    try:
        require_v5_candidate_implementation_binding(payload, repository_root=root)
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
        "parent_v4_stacking_implementation",
        "gate_implementation",
    }
    if set(inputs) != required_inputs:
        blockers.append("inputs")
    if dict(inputs.get("gate_implementation") or {}) != IMMUTABLE_GATE_IMPLEMENTATION:
        blockers.append("gate_implementation")
    if dict(inputs.get("parent_v4_stacking_implementation") or {}) != (
        PARENT_V4_STACKING_IMPLEMENTATION
    ):
        blockers.append("parent_v4_stacking_implementation")
    if verify_artifacts and not blockers:
        resolved: dict[str, Path] = {}
        for name, descriptor in inputs.items():
            try:
                resolved[name] = _verify_descriptor(dict(descriptor), repository_root=root)
            except (KeyError, OSError, TypeError, ValueError):
                blockers.append(f"inputs.{name}")
        if not blockers:
            try:
                validate_v5_lineage_authorization(
                    authorization_path=resolved["lineage_authorization"],
                    registry_path=resolved["development_data_registry"],
                    repository_root=root,
                )
            except ValueError:
                blockers.append("lineage_authorization")
            terminal = _load_json(resolved["parent_v4_terminal_review"])
            report = _load_json(resolved["parent_v4_challenger_report"])
            if not (
                terminal.get("phase_1_terminal_failed") is True
                and terminal.get("candidate_budget_exhausted") is True
                and terminal.get("candidate_selected") is None
                and report.get("all_gates_passed") is False
                and report.get("failed_gates")
                == ["prospective_power_required_market_count_lte_2000"]
                and int(report["prospective_power"]["required_market_count"]) == 2488
                and dict(terminal.get("safety") or {}) == SAFETY
                and dict(report.get("safety") or {}) == SAFETY
            ):
                blockers.append("parent_v4_terminal_boundary")
    if blockers:
        raise ValueError("residual v5 protocol invalid: " + ", ".join(blockers))


def run_residual_v5_rolling_origin_oof(
    *,
    protocol_path: Path | str = DEFAULT_PROTOCOL,
    expected_protocol_sha256: str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    source_commit: str,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Execute the first preregistered v5 candidate exactly once."""

    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    output = Path(output_dir).resolve()
    if not protocol_file.is_relative_to(root) or not output.is_relative_to(root):
        raise ValueError("residual v5 OOF paths must remain repository-local")
    if sha256_file(protocol_file) != expected_protocol_sha256.lower():
        raise ValueError("residual v5 protocol SHA-256 mismatch")
    sidecar = protocol_file.with_suffix(".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != (
        expected_protocol_sha256
    ):
        raise ValueError("residual v5 protocol is not SHA-frozen")
    if not _looks_like_git_sha(source_commit):
        raise ValueError("source_commit must be a full lowercase Git SHA")
    protocol = _load_json(protocol_file)
    validate_residual_v5_protocol(protocol, repository_root=root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"residual v5 OOF output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    dataset_rows, base_predictions, parent_markets, population_order = _load_frozen_v4_rows(
        protocol=protocol, repository_root=root
    )
    predictions, folds = prequential_market_residual_corrector_predict(
        rows=dataset_rows,
        frozen_base_predictions=base_predictions,
        population_order=population_order,
        protocol=protocol,
    )
    baseline_by_market = _baseline_rows(parent_markets)
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

    dataset_path = output / "residual_v5_development_dataset_rows.jsonl"
    prediction_path = output / "residual_v5_oof_predictions.jsonl"
    fold_path = output / "residual_v5_oof_fold_audits.jsonl"
    market_path = output / "residual_v5_oof_market_results.jsonl"
    report_path = output / "residual_v5_oof_report.json"
    markdown_path = output / "residual_v5_oof_report.md"
    _write_new_jsonl(dataset_path, [_public_dataset_row(row) for row in dataset_rows])
    _write_new_jsonl(prediction_path, predictions)
    _write_new_jsonl(fold_path, folds)
    _write_new_jsonl(market_path, markets)
    report_artifact = _write_new_frozen_json(report_path, report)
    markdown_artifact = _write_new_frozen_text(markdown_path, render_v5_markdown(report))
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "slot_id": protocol["slot_id"],
        "candidate_role": "primary",
        "created_at": protocol["created_at"],
        "source_commit": source_commit,
        "protocol": _descriptor(protocol_file, root),
        "candidate_implementation": dict(protocol["inputs"]["candidate_implementation"]),
        "parent_v4_stacking_implementation": dict(
            protocol["inputs"]["parent_v4_stacking_implementation"]
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
        "remaining_candidate_slots": 1,
        "candidate_freeze_allowed": report["all_gates_passed"],
        "next_stage_authorization_required_even_if_all_gates_pass": True,
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "safety": dict(SAFETY),
    }
    manifest_artifact = _write_new_frozen_json(
        output / "residual_v5_oof_manifest.json", manifest
    )
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


def verify_frozen_residual_v5_oof(
    *,
    protocol_path: Path | str = DEFAULT_PROTOCOL,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Verify frozen hashes, population, implementation binding and report rebuild."""

    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    output = Path(output_dir).resolve()
    protocol = _verified_json(protocol_file)
    validate_residual_v5_protocol(protocol, repository_root=root)
    manifest = _verified_json(output / "residual_v5_oof_manifest.json")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("residual v5 manifest schema mismatch")
    if manifest.get("protocol") != _descriptor(protocol_file, root):
        raise ValueError("residual v5 manifest protocol binding mismatch")
    if manifest.get("candidate_implementation") != require_v5_candidate_implementation_binding(
        protocol, repository_root=root
    ):
        raise ValueError("residual v5 manifest implementation binding mismatch")
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
        raise ValueError("residual v5 manifest artifact set mismatch")
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
        raise ValueError("residual v5 dataset governance mismatch")
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
    _assert_semantically_equal(rebuilt, frozen, path="residual_v5_oof_report")
    if render_v5_markdown(rebuilt) != artifacts["report_markdown"].read_text(encoding="utf-8"):
        raise ValueError("residual v5 Markdown does not reproduce")
    return {
        "verification_passed": True,
        "all_gates_passed": bool(frozen["all_gates_passed"]),
        "failed_gates": list(frozen["failed_gates"]),
        "remaining_candidate_slots": 1,
        "oof_market_count": len(markets),
        "manifest_sha256": sha256_file(output / "residual_v5_oof_manifest.json"),
        "actual_executing_module_binding_verified": True,
        "parent_v1_through_v4_immutable": True,
        "safety": dict(SAFETY),
    }


def prequential_market_residual_corrector_predict(
    *,
    rows: Sequence[Mapping[str, Any]],
    frozen_base_predictions: Sequence[Mapping[str, Any]],
    population_order: Sequence[str],
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Correct frozen v4 OOF probabilities using only earlier OOF labels."""

    rows_by_market: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_market[str(row["market_id"])].append(row)
    base_by_key = {
        (str(row["market_id"]), int(row["decision_ts"]), str(row["side"])): row
        for row in frozen_base_predictions
    }
    if len(base_by_key) != len(frozen_base_predictions):
        raise ValueError("frozen v4 base prediction keys are not unique")
    rolling = dict(protocol["rolling_origin"])
    initial = int(rolling["initial_training_market_count"])
    block_size = int(rolling["target_block_size"])
    block_count = int(rolling["target_block_count"])
    predictions: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for block_index in range(block_count):
        target_start = initial + block_index * block_size
        target_end = target_start + block_size
        prior_oof_ids = list(population_order[initial:target_start])
        target_ids = list(population_order[target_start:target_end])
        if len(target_ids) != block_size:
            raise ValueError("v5 target block population mismatch")
        prior_rows = _ordered_rows(prior_oof_ids, rows_by_market)
        target_rows = _ordered_rows(target_ids, rows_by_market)
        prior_pairs = _bind_base_predictions(prior_rows, base_by_key)
        target_pairs = _bind_base_predictions(target_rows, base_by_key)
        booster: xgb.Booster | None = None
        if prior_pairs:
            labels = [
                float(row["binary_payout_target"]) - float(base["predicted_probability"])
                for row, base in prior_pairs
            ]
            booster = xgb.train(
                params=dict(protocol["model"]["parameters"]),
                dtrain=_corrector_dmatrix(prior_pairs, labels=labels),
                num_boost_round=int(protocol["model"]["fixed_num_boost_round"]),
                verbose_eval=False,
            )
            corrections = [
                float(value)
                for value in booster.predict(_corrector_dmatrix(target_pairs, labels=None))
            ]
        else:
            corrections = [0.0] * len(target_pairs)
        actions = _corrected_actions(target_pairs, corrections)
        for (row, base), action in zip(target_pairs, actions, strict=True):
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
                    "frozen_v4_predicted_probability": float(base["predicted_probability"]),
                    "frozen_v4_action_value": float(base["prediction"]),
                    "predicted_probability_residual": action["residual"],
                    "corrected_probability_before_pair_normalization": action[
                        "probability_before_pair_normalization"
                    ],
                    "predicted_probability": action["predicted_probability"],
                    "realized_unit_net_pnl_if_action": row["target"],
                    "resolved_outcome": row["resolved_outcome"],
                    "cost_decomposition": row["cost_decomposition"],
                    "feature_row_sha256": row["feature_row_sha256"],
                    "chronological_block": block_index + 1,
                    "strictly_prior_corrector_training_market_count": len(prior_oof_ids),
                    "strictly_prior_corrector_training_row_count": len(prior_pairs),
                    "corrector_applied": booster is not None,
                    "current_or_future_label_used_for_corrector": False,
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
                "strictly_prior_training_market_count": target_start,
                "strictly_prior_corrector_training_market_count": len(prior_oof_ids),
                "strictly_prior_corrector_training_row_count": len(prior_pairs),
                "target_market_count": len(target_ids),
                "training_market_ids_sha256": canonical_json_sha256(
                    list(population_order[:target_start])
                ),
                "corrector_training_market_ids_sha256": canonical_json_sha256(prior_oof_ids),
                "target_market_ids_sha256": canonical_json_sha256(target_ids),
                "last_training_market_position": target_start,
                "first_target_market_position": target_start + 1,
                "target_or_future_label_leakage_count": 0,
                "frozen_v4_oof_base_only": True,
                "corrector_applied": booster is not None,
                "fixed_num_boost_round": int(protocol["model"]["fixed_num_boost_round"]),
                "model_parameters_sha256": canonical_json_sha256(
                    dict(protocol["model"]["parameters"])
                ),
                "corrector_feature_names_sha256": canonical_json_sha256(
                    list(CORRECTOR_FEATURE_NAMES)
                ),
                "development_only_forever": True,
                "promotion_evidence_eligible": False,
                "safety": dict(SAFETY),
            }
        )
    return predictions, audits


def _corrector_dmatrix(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    labels: Sequence[float] | None,
) -> xgb.DMatrix:
    values = np.vstack([_corrector_features(row, base) for row, base in pairs])
    target = None if labels is None else np.asarray(labels, dtype=float)
    if target is not None and target.shape != (len(pairs),):
        raise ValueError("corrector labels do not align")
    return xgb.DMatrix(
        values,
        label=target,
        feature_names=list(CORRECTOR_FEATURE_NAMES),
        missing=np.nan,
    )


def _corrector_features(row: Mapping[str, Any], base: Mapping[str, Any]) -> np.ndarray:
    raw = dict(row["features"])
    if tuple(raw) != BASE_FEATURE_NAMES:
        raise ValueError("v5 inherited 108-feature order changed")
    cost = dict(row["cost_decomposition"])
    values = [np.nan if raw[name] is None else float(raw[name]) for name in BASE_FEATURE_NAMES]
    values.extend(
        [
            float(base["predicted_probability"]),
            float(base["prediction"]),
            float(cost["entry_ask"]),
            float(cost["total_cost_excluding_entry_ask"]),
        ]
    )
    output = np.asarray(values, dtype=float)
    if output.shape != (len(CORRECTOR_FEATURE_NAMES),):
        raise ValueError("v5 corrector feature vector size changed")
    return output


def _corrected_actions(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    corrections: Sequence[float],
) -> list[dict[str, float]]:
    if len(pairs) != len(corrections):
        raise ValueError("v5 correction row count mismatch")
    grouped: dict[tuple[str, int], list[tuple[int, Mapping[str, Any], float, float]]] = (
        defaultdict(list)
    )
    output: list[dict[str, float] | None] = [None] * len(pairs)
    for index, ((row, base), correction) in enumerate(zip(pairs, corrections, strict=True)):
        residual = float(correction)
        raw = min(
            1.0 - PAIR_CLIP_EPSILON,
            max(PAIR_CLIP_EPSILON, float(base["predicted_probability"]) + residual),
        )
        if not math.isfinite(residual) or not math.isfinite(raw):
            raise ValueError("v5 correction is non-finite")
        grouped[(str(row["market_id"]), int(row["decision_ts"]))].append(
            (index, row, residual, raw)
        )
    for key, members in grouped.items():
        if [str(item[1]["side"]) for item in members] != ["UP", "DOWN"]:
            raise ValueError(f"v5 UP/DOWN pair is incomplete or reordered: {key}")
        denominator = math.fsum(item[3] for item in members)
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("v5 pair denominator is invalid")
        for index, row, residual, raw in members:
            probability = raw / denominator
            cost = dict(row["cost_decomposition"])
            action_value = (
                probability
                - float(cost["entry_ask"])
                - float(cost["total_cost_excluding_entry_ask"])
            )
            output[index] = {
                "residual": residual,
                "probability_before_pair_normalization": raw,
                "predicted_probability": probability,
                "action_value": action_value,
            }
    if any(item is None for item in output):
        raise ValueError("v5 corrected action output is incomplete")
    return [dict(item) for item in output if item is not None]


def render_v5_markdown(report: Mapping[str, Any]) -> str:
    base = (
        render_residual_oof_markdown(report)
        .replace(
            "# BTC 15m cost-aware residual primary slot 001",
            "# BTC 15m prequential market-residual v5 primary slot 001",
            1,
        )
        .rstrip()
    )
    return (
        base
        + "\n\n## Architecture and authorization boundary\n\n"
        + "- Architecture: frozen v4 soft-stacking base plus expanding prequential residual corrector.\n"
        + "- Corrector labels: strictly prior v4 OOF binary-payout probability residuals only.\n"
        + f"- Fixed rounds: `{FIXED_NUM_BOOST_ROUND}`; grid or threshold search: `False`\n"
        + "- First OOF block uses the frozen v4 probability unchanged because no prior v4 OOF label exists.\n"
        + "- Existing gates, zero threshold, N_max, costs, baseline and population changed: `False`\n"
        + "- Collection, outcome opening, shadow, paper/live, wallet, write, promotion or capital authorized: `False`\n"
    )


def _load_frozen_v4_rows(
    *, protocol: Mapping[str, Any], repository_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    inputs = dict(protocol["inputs"])
    dataset = _load_jsonl(
        _verify_descriptor(inputs["parent_v4_challenger_dataset_rows"], repository_root=repository_root)
    )
    predictions = _load_jsonl(
        _verify_descriptor(inputs["parent_v4_challenger_predictions"], repository_root=repository_root)
    )
    markets = _load_jsonl(
        _verify_descriptor(inputs["parent_v4_challenger_market_results"], repository_root=repository_root)
    )
    positions: dict[str, int] = {}
    for row in dataset:
        market_id = str(row["market_id"])
        position = int(row["market_position"])
        if market_id in positions and positions[market_id] != position:
            raise ValueError("v5 parent dataset market position mismatch")
        positions[market_id] = position
    population_order = [item[0] for item in sorted(positions.items(), key=lambda item: item[1])]
    if len(dataset) != 3200 or len(population_order) != 800:
        raise ValueError("v5 parent dataset population changed")
    if len(predictions) != 2400 or len(markets) != 600:
        raise ValueError("v5 frozen v4 OOF population changed")
    return dataset, predictions, markets, population_order


def _ordered_rows(
    market_ids: Sequence[str], rows_by_market: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[Mapping[str, Any]]:
    output: list[Mapping[str, Any]] = []
    for market_id in market_ids:
        rows = sorted(
            rows_by_market.get(market_id, []),
            key=lambda row: (int(row["decision_ts"]), 0 if row["side"] == "UP" else 1),
        )
        if len(rows) != 4:
            raise ValueError("v5 market must retain two decisions and paired UP/DOWN rows")
        output.extend(rows)
    return output


def _bind_base_predictions(
    rows: Sequence[Mapping[str, Any]],
    base_by_key: Mapping[tuple[str, int, str], Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    output = []
    for row in rows:
        key = (str(row["market_id"]), int(row["decision_ts"]), str(row["side"]))
        base = base_by_key.get(key)
        if base is None:
            raise ValueError("v5 frozen v4 base prediction missing")
        if not (
            base.get("feature_row_sha256") == row.get("feature_row_sha256")
            and math.isclose(
                float(base["realized_unit_net_pnl_if_action"]),
                float(row["target"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("v5 frozen v4 base row binding mismatch")
        output.append((row, base))
    return output


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
        "frozen_v4_nested_soft_stacking_with_expanding_prequential_market_level_"
        "probability_residual_corrector"
    )
    report["immutable_gate_implementation_sha256"] = IMMUTABLE_GATE_IMPLEMENTATION["sha256"]
    report["actual_executing_module_exactly_bound"] = True
    report["existing_gate_threshold_cost_baseline_population_changed"] = False
    report["parent_v1_v2_v3_v4_failed_artifacts_changed"] = False
    report["remaining_candidate_slots"] = 1
    report["additional_candidate_requires_preregistration"] = True
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
        or row.get("current_or_future_label_used_for_corrector") is not False
        or dict(row.get("safety") or {}) != SAFETY
        for row in predictions
    ):
        raise ValueError("v5 prediction governance mismatch")
    if any(
        row.get("schema_version") != FOLD_SCHEMA_VERSION
        or row.get("lineage_id") != LINEAGE_ID
        or row.get("target_or_future_label_leakage_count") != 0
        or dict(row.get("safety") or {}) != SAFETY
        for row in fold_audits
    ):
        raise ValueError("v5 fold governance mismatch")
    if any(
        row.get("schema_version") != MARKET_RESULT_SCHEMA_VERSION
        or row.get("lineage_id") != LINEAGE_ID
        or dict(row.get("safety") or {}) != SAFETY
        for row in market_results
    ):
        raise ValueError("v5 market governance mismatch")
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
            raise ValueError("v5 action-value reconciliation failed")
    for rows in grouped.values():
        if [str(row["side"]) for row in rows] != ["UP", "DOWN"]:
            raise ValueError("v5 prediction pair side mismatch")
        if not math.isclose(
            math.fsum(float(row["predicted_probability"]) for row in rows),
            1.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("v5 pair probability does not sum to one")
    _validate_frozen_population(
        predictions=_as_v1_predictions(predictions),
        fold_audits=_as_v1_folds(fold_audits),
        market_results=_as_v1_market_results(market_results),
        protocol=protocol,
    )


def _public_dataset_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = deepcopy(dict(row))
    output["schema_version"] = DATASET_SCHEMA_VERSION
    output["lineage_id"] = LINEAGE_ID
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
        "corrector_label": "settlement_payout-frozen_v4_stacked_probability",
        "action_value_formula": (
            "pair_normalized_clipped_frozen_v4_probability_plus_prequential_residual-"
            "entry_ask-frozen_fees-slippage-liquidity_impact"
        ),
        "execution_policy": "HOLD_TO_SETTLEMENT",
        "NO_TRADE_value": 0.0,
        "post_close_training_label_only": True,
    }


def _pair_contract() -> dict[str, Any]:
    return {
        "frozen_v4_probability_is_pair_coherent": True,
        "corrected_probability_clipped_then_pair_normalized": True,
        "cost_subtraction_happens_after_pair_normalization": True,
        "probability_clip_epsilon": PAIR_CLIP_EPSILON,
        "missing_anchor_behavior": "fail_closed_NO_TRADE_in_runtime",
        "missing_values_encoded_as_zero": False,
    }


def _feature_contract() -> dict[str, Any]:
    parent = _parent_feature_contract()
    return {
        **parent,
        "corrector_ordered_feature_count": len(CORRECTOR_FEATURE_NAMES),
        "corrector_ordered_feature_names_sha256": canonical_json_sha256(
            list(CORRECTOR_FEATURE_NAMES)
        ),
        "corrector_context_features": list(CORRECTOR_CONTEXT_FEATURE_NAMES),
        "new_raw_feature_added": False,
    }


def _model_contract() -> dict[str, Any]:
    return {
        "family": "expanding_prequential_market_level_probability_residual_xgboost",
        "frozen_base": "residual_v4_nested_rolling_origin_soft_stacking_slot_002",
        "fixed_num_boost_round": FIXED_NUM_BOOST_ROUND,
        "parameters": dict(FIXED_PARAMETERS),
        "route_or_expert_filtering_allowed": False,
        "model_selection_or_early_stopping_performed": False,
    }


def _prequential_contract() -> dict[str, Any]:
    return {
        "base_predictions": "immutable_v4_outer_oof_predictions",
        "corrector_training_population": "strictly_prior_v4_oof_side_decision_rows_only",
        "first_oof_block_behavior": "identity_frozen_v4_probability_no_corrector_fit",
        "later_block_behavior": "expanding_prior_oof_residual_fit",
        "current_or_future_block_label_used": False,
        "base_or_corrector_parameter_search_allowed": False,
        "hard_routing_or_filtering_allowed": False,
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
    "prequential_market_residual_corrector_predict",
    "require_v5_candidate_implementation_binding",
    "run_residual_v5_rolling_origin_oof",
    "validate_residual_v5_protocol",
    "validate_v5_lineage_authorization",
    "verify_frozen_residual_v5_oof",
]
