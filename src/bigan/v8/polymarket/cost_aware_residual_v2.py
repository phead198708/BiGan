"""Governed market-anchored residual development for BTC 15m lineage v2."""

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
from bigan.v8.polymarket.challenge_model_15m_training import SIDES
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
    _public_dataset_row,
    _validate_frozen_population,
    _verified_json,
    _verify_descriptor,
    build_residual_oof_report,
    market_results_from_predictions,
    render_residual_oof_markdown,
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

LINEAGE_ID = "BTC-15M-cost-aware-market-residual-v2"
PROTOCOL_SCHEMA_VERSION = "bigan-btc-15m-market-anchored-residual-oof-protocol-v2"
PREDICTION_SCHEMA_VERSION = "bigan-btc-15m-market-anchored-residual-oof-prediction-v2"
FOLD_SCHEMA_VERSION = "bigan-btc-15m-market-anchored-residual-oof-fold-v2"
MARKET_RESULT_SCHEMA_VERSION = "bigan-btc-15m-market-anchored-residual-market-result-v2"
REPORT_SCHEMA_VERSION = "bigan-btc-15m-market-anchored-residual-oof-report-v2"
MANIFEST_SCHEMA_VERSION = "bigan-btc-15m-market-anchored-residual-oof-manifest-v2"
DATASET_SCHEMA_VERSION = "bigan-btc-15m-market-anchored-residual-dataset-row-v2"

DEFAULT_CONFIG_DIR = (
    REPO_ROOT
    / "examples/v8/polymarket_configs/BTC-15M-cost-aware-market-residual-v2"
)
DEFAULT_PROTOCOL = DEFAULT_CONFIG_DIR / "residual_v2_primary_slot_001_protocol.json"
DEFAULT_OUTPUT_DIR = DEFAULT_CONFIG_DIR / "residual_v2_primary_slot_001_oof"
DEFAULT_AUTHORIZATION = DEFAULT_CONFIG_DIR / "lineage_authorization.json"
DEFAULT_REGISTRY = DEFAULT_CONFIG_DIR / "development_data_registry.json"

AUTHORIZATION_INSTRUCTION = (
    "建立新的 residual lineage，并授予新的候选 slot；旧 gate、阈值、失败报告及安全状态保持不可修改。"
)
AUTHORIZATION_INSTRUCTION_SHA256 = (
    "9ec526fd63a8e50b6c3000ca50a6cbedc440d70e563fab1ca3617ec7075c272b"
)
PAIR_CLIP_EPSILON = 1e-6
SELECTED_MID_INDEX = FEATURE_NAMES.index("selected_mid")

GATE_NAMES = {
    "absolute_market_bootstrap_97_5pct_lcb_gt_zero",
    "paired_delta_market_bootstrap_97_5pct_lcb_gt_zero",
    "every_chronological_block_candidate_total_gte_zero",
    "every_chronological_block_paired_delta_total_gte_zero",
    "largest_winner_removed_candidate_total_gte_zero",
    "largest_positive_delta_removed_total_gte_zero",
    "stable_score_to_realized_pnl_ordering",
    "all_cost_stress_candidate_totals_gte_zero",
    "all_cost_stress_paired_delta_totals_gte_zero",
    "prospective_power_required_market_count_lte_2000",
    "population_and_leakage_reconciliation",
}


def validate_v2_lineage_authorization(
    *,
    authorization_path: Path | str = DEFAULT_AUTHORIZATION,
    registry_path: Path | str = DEFAULT_REGISTRY,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Verify the explicit user authorization and immutable v1 boundary."""

    root = Path(repository_root).resolve()
    authorization_file = Path(authorization_path).resolve()
    registry_file = Path(registry_path).resolve()
    authorization = _verified_json(authorization_file)
    registry = _verified_json(registry_file)
    blockers: list[str] = []
    source = dict(authorization.get("authorization_source") or {})
    if authorization.get("schema_version") != (
        "bigan-btc-15m-cost-aware-residual-lineage-authorization-v2"
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
    scope = dict(authorization.get("authorization_scope") or {})
    if scope != {
        "candidate_slot_budget": {
            "challenger_slots": 1,
            "maximum_total_slots": 2,
            "primary_slots": 1,
            "slot_budget_may_be_increased": False,
        },
        "fresh_confirmatory_collection_authorized": False,
        "live_shadow_start_authorized": False,
        "micro_live_authorized": False,
        "outcome_aware_development_training_authorized": True,
        "paper_or_live_execution_authorized": False,
        "promotion_authorized": False,
        "training_may_start_only_after_slot_protocol_is_sha_frozen": True,
    }:
        blockers.append("authorization_scope")
    if dict(authorization.get("safety") or {}) != SAFETY:
        blockers.append("authorization.safety")
    state = dict(authorization.get("state") or {})
    if state != {
        "candidate_frozen": False,
        "fresh_confirmatory_collection_started": False,
        "fresh_outcomes_opened": False,
        "lineage_authorized_for_governed_development": True,
        "live_shadow_started": False,
        "micro_live_started": False,
        "training_started": False,
    }:
        blockers.append("authorization.state")
    parent = dict(authorization.get("parent_lineage") or {})
    if not (
        parent.get("lineage_id") == V1_LINEAGE_ID
        and parent.get("status") == "phase_1_terminal_failed"
        and parent.get("candidate_budget_consumed") == 2
        and parent.get("candidate_budget_maximum") == 2
        and parent.get("failed_artifacts_mutable") is False
        and parent.get("gate_or_threshold_change_allowed") is False
    ):
        blockers.append("parent_lineage")
    try:
        terminal_path = _verify_descriptor(
            dict(parent["terminal_review"]), repository_root=root
        )
        terminal = _load_json(terminal_path)
        if not (
            terminal.get("lineage_id") == V1_LINEAGE_ID
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
            dict(authorization["registered_development_data"]),
            repository_root=root,
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
        except (OSError, TypeError, ValueError):
            blockers.append(f"development_data_registry.{name}")
    if blockers:
        raise ValueError("residual v2 authorization invalid: " + ", ".join(blockers))
    return {
        "authorization_valid": True,
        "lineage_id": LINEAGE_ID,
        "maximum_total_slots": 2,
        "parent_v1_immutable": True,
        "authorization_sha256": sha256_file(authorization_file),
        "registry_sha256": sha256_file(registry_file),
        "safety": dict(SAFETY),
    }


def validate_residual_v2_protocol(
    payload: Mapping[str, Any],
    *,
    repository_root: Path | str = REPO_ROOT,
    verify_artifacts: bool = True,
) -> None:
    """Fail closed unless slot 1 is frozen with all v1 gates unchanged."""

    blockers: list[str] = []
    if payload.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        blockers.append("schema_version")
    if payload.get("lineage_id") != LINEAGE_ID:
        blockers.append("lineage_id")
    if payload.get("slot_id") != "residual-v2-primary-slot-001":
        blockers.append("slot_id")
    if payload.get("candidate_role") != "primary":
        blockers.append("candidate_role")
    if payload.get("development_only_forever") is not True:
        blockers.append("development_only_forever")
    if payload.get("promotion_evidence_eligible") is not False:
        blockers.append("promotion_evidence_eligible")
    if dict(payload.get("candidate_budget") or {}) != {
        "maximum_total_slots": 2,
        "this_slot_ordinal": 1,
        "slots_consumed_before_run": 0,
        "slots_remaining_after_run": 1,
        "slot_budget_may_be_increased": False,
    }:
        blockers.append("candidate_budget")
    if dict(payload.get("target") or {}) != {
        "action_value_formula": (
            "pair_normalized_clipped_selected_mid_plus_predicted_probability_"
            "residual-entry_ask-frozen_fees-slippage-liquidity_impact"
        ),
        "execution_policy": "HOLD_TO_SETTLEMENT",
        "NO_TRADE_value": 0.0,
        "post_close_training_label_only": True,
        "regression_label": "settlement_payout-selected_mid",
    }:
        blockers.append("target")
    if dict(payload.get("pair_coherence") or {}) != {
        "anchor": "decision_time_selected_mid",
        "clip_epsilon": PAIR_CLIP_EPSILON,
        "normalization": "UP_DOWN_probabilities_sum_to_one_per_decision",
        "normalization_happens_before_cost_subtraction": True,
        "missing_anchor_behavior": "fail_closed_NO_TRADE_in_runtime",
        "missing_values_encoded_as_zero": False,
    }:
        blockers.append("pair_coherence")
    feature = dict(payload.get("feature_contract") or {})
    if not (
        feature.get("ordered_feature_count") == 108
        and feature.get("base_feature_count") == 54
        and feature.get("shared_side_symmetric_model") is True
        and feature.get("side_identity_feature_allowed") is False
        and feature.get("native_missing_value") == "nan"
        and feature.get("missing_values_encoded_as_zero") is False
        and feature.get("feature_search_allowed") is False
        and feature.get("market_horizon_seconds") == 900
        and feature.get("source_contract_reused_without_feature_addition_or_removal")
        is True
    ):
        blockers.append("feature_contract")
    model = dict(payload.get("model") or {})
    if not (
        model.get("family")
        == "pooled_side_symmetric_market_anchored_probability_residual_xgboost"
        and model.get("route_or_expert_allowed") is False
        and model.get("fixed_num_boost_round") == 128
        and model.get("model_selection_or_early_stopping_performed") is False
        and dict(model.get("parameters") or {}) == _model_parameters()
    ):
        blockers.append("model")
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
    dataset = dict(payload.get("dataset") or {})
    if not (
        dataset.get("market_count") == 800
        and dataset.get("side_decision_row_count") == 3200
        and dataset.get("decision_rows_per_market") == 2
        and dataset.get("sides_per_decision") == 2
        and dataset.get("development_only_forever") is True
        and dataset.get("population_order")
        == "frozen_confirmatory_capture_manifest_order"
    ):
        blockers.append("dataset")
    discipline = dict(payload.get("development_discipline") or {})
    if discipline != {
        "one_candidate_this_slot": True,
        "hyperparameter_search_allowed": False,
        "threshold_search_allowed": False,
        "route_side_missingness_or_outlier_filtering_allowed": False,
        "post_result_mutation_allowed": False,
        "challenger_requires_separate_preregistration": True,
    }:
        blockers.append("development_discipline")
    if dict(payload.get("state") or {}) != {
        "training_started": False,
        "candidate_frozen": False,
        "live_shadow_started": False,
        "fresh_confirmatory_collection_started": False,
        "fresh_outcomes_opened": False,
    }:
        blockers.append("state")
    if dict(payload.get("safety") or {}) != SAFETY:
        blockers.append("safety")
    inputs = dict(payload.get("inputs") or {})
    required_inputs = {
        "lineage_authorization",
        "development_data_registry",
        "parent_v1_terminal_review",
        "terminal_diagnostic_scored_rows",
        "confirmatory_capture_manifest",
        "confirmatory_market_evaluation_rows",
        "baseline_decision_rows",
        "matched_global_baseline_contract",
        "parent_feature_contract",
        "parent_cost_and_action_contract",
        "raw_capture_recovery_bundle_manifest",
        "candidate_implementation",
        "gate_implementation",
    }
    if set(inputs) != required_inputs:
        blockers.append("inputs")
    root = Path(repository_root).resolve()
    if verify_artifacts and not blockers:
        resolved: dict[str, Path] = {}
        for name, descriptor in inputs.items():
            try:
                resolved[name] = _verify_descriptor(
                    dict(descriptor), repository_root=root
                )
            except (KeyError, OSError, TypeError, ValueError):
                blockers.append(f"inputs.{name}")
        if not blockers:
            try:
                authorization_result = validate_v2_lineage_authorization(
                    authorization_path=resolved["lineage_authorization"],
                    registry_path=resolved["development_data_registry"],
                    repository_root=root,
                )
                if authorization_result["maximum_total_slots"] != 2:
                    blockers.append("lineage_authorization.slot_budget")
            except ValueError:
                blockers.append("lineage_authorization")
            terminal = _load_json(resolved["parent_v1_terminal_review"])
            if not (
                terminal.get("phase_1_terminal_failed") is True
                and terminal.get("candidate_budget_exhausted") is True
                and dict(terminal.get("safety") or {}) == SAFETY
            ):
                blockers.append("parent_v1_terminal_review")
            gate_descriptor = dict(inputs["gate_implementation"])
            if gate_descriptor != {
                "path": "src/bigan/v8/polymarket/cost_aware_residual.py",
                "sha256": (
                    "491a329f708a16d5aecdd952552cbff3fa13d8f7446bfe3e0c78fade3b36f78c"
                ),
            }:
                blockers.append("gate_implementation")
    if blockers:
        raise ValueError("residual v2 protocol invalid: " + ", ".join(blockers))


def run_residual_v2_rolling_origin_oof(
    *,
    protocol_path: Path | str = DEFAULT_PROTOCOL,
    expected_protocol_sha256: str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    source_commit: str,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Execute the first v2 candidate exactly once after SHA preregistration."""

    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    output = Path(output_dir).resolve()
    if not protocol_file.is_relative_to(root) or not output.is_relative_to(root):
        raise ValueError("residual v2 OOF paths must remain repository-local")
    if sha256_file(protocol_file) != expected_protocol_sha256.lower():
        raise ValueError("residual v2 protocol SHA-256 mismatch")
    sidecar = protocol_file.with_suffix(".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != (
        expected_protocol_sha256
    ):
        raise ValueError("residual v2 protocol is not SHA-frozen")
    if not _looks_like_git_sha(source_commit):
        raise ValueError("source_commit must be a full lowercase Git SHA")
    protocol = _load_json(protocol_file)
    validate_residual_v2_protocol(protocol, repository_root=root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"residual v2 OOF output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    dataset_rows, baseline_by_market, population_order = _load_frozen_development_rows(
        protocol=protocol,
        repository_root=root,
    )
    predictions, folds = _rolling_origin_pair_anchored_predict(
        rows=dataset_rows,
        population_order=population_order,
        protocol=protocol,
    )
    market_results = _v2_market_results_from_predictions(
        predictions=predictions,
        baseline_by_market=baseline_by_market,
        population_order=population_order,
        protocol=protocol,
    )
    report = _build_v2_report(
        protocol=protocol,
        protocol_sha256=expected_protocol_sha256,
        source_commit=source_commit,
        market_results=market_results,
        fold_audits=folds,
    )

    dataset_path = output / "residual_v2_development_dataset_rows.jsonl"
    prediction_path = output / "residual_v2_oof_predictions.jsonl"
    fold_path = output / "residual_v2_oof_fold_audits.jsonl"
    market_path = output / "residual_v2_oof_market_results.jsonl"
    report_path = output / "residual_v2_oof_report.json"
    markdown_path = output / "residual_v2_oof_report.md"
    _write_new_jsonl(dataset_path, [_public_v2_dataset_row(row) for row in dataset_rows])
    _write_new_jsonl(prediction_path, predictions)
    _write_new_jsonl(fold_path, folds)
    _write_new_jsonl(market_path, market_results)
    report_artifact = _write_new_frozen_json(report_path, report)
    markdown_artifact = _write_new_frozen_text(
        markdown_path, render_residual_v2_markdown(report)
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "slot_id": protocol["slot_id"],
        "candidate_role": "primary",
        "created_at": protocol["created_at"],
        "source_commit": source_commit,
        "protocol": _descriptor(protocol_file, root),
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
        "remaining_candidate_slots": 1,
        "candidate_freeze_allowed": report["all_gates_passed"],
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "safety": dict(SAFETY),
    }
    manifest_artifact = _write_new_frozen_json(
        output / "residual_v2_oof_manifest.json", manifest
    )
    return {
        "manifest": _descriptor(Path(manifest_artifact["path"]), root),
        "report": _descriptor(Path(report_artifact["path"]), root),
        "all_gates_passed": report["all_gates_passed"],
        "failed_gates": report["failed_gates"],
        "remaining_candidate_slots": 1,
        "oof_market_count": len(market_results),
        "safety": dict(SAFETY),
    }


def verify_frozen_residual_v2_oof(
    *,
    protocol_path: Path | str = DEFAULT_PROTOCOL,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Verify hashes and semantically rebuild the frozen v2 OOF report."""

    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    output = Path(output_dir).resolve()
    protocol = _verified_json(protocol_file)
    validate_residual_v2_protocol(protocol, repository_root=root)
    manifest = _verified_json(output / "residual_v2_oof_manifest.json")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("residual v2 manifest schema mismatch")
    if manifest.get("protocol") != _descriptor(protocol_file, root):
        raise ValueError("residual v2 manifest protocol binding mismatch")
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
        raise ValueError("residual v2 manifest artifact set mismatch")
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
        raise ValueError("residual v2 dataset governance mismatch")
    _validate_v2_population(
        predictions=predictions,
        fold_audits=folds,
        market_results=markets,
        protocol=protocol,
    )
    rebuilt = _build_v2_report(
        protocol=protocol,
        protocol_sha256=sha256_file(protocol_file),
        source_commit=str(manifest["source_commit"]),
        market_results=markets,
        fold_audits=folds,
    )
    frozen = _load_json(artifacts["report"])
    _assert_semantically_equal(rebuilt, frozen, path="residual_v2_oof_report")
    if render_residual_v2_markdown(rebuilt) != artifacts["report_markdown"].read_text(
        encoding="utf-8"
    ):
        raise ValueError("residual v2 Markdown does not reproduce")
    return {
        "verification_passed": True,
        "all_gates_passed": bool(frozen["all_gates_passed"]),
        "failed_gates": list(frozen["failed_gates"]),
        "remaining_candidate_slots": 1,
        "oof_market_count": len(markets),
        "manifest_sha256": sha256_file(output / "residual_v2_oof_manifest.json"),
        "parent_v1_immutable": True,
        "safety": dict(SAFETY),
    }


def pair_anchored_action_values(
    rows: Sequence[Mapping[str, Any]],
    residual_predictions: Sequence[float],
    *,
    clip_epsilon: float = PAIR_CLIP_EPSILON,
) -> list[dict[str, float]]:
    """Convert residual predictions to coherent pair probabilities and action values."""

    if len(rows) != len(residual_predictions):
        raise ValueError("residual prediction row count mismatch")
    if not 0.0 < clip_epsilon < 0.5:
        raise ValueError("pair probability clip epsilon is invalid")
    grouped: dict[tuple[str, int], list[tuple[int, Mapping[str, Any], float]]] = (
        defaultdict(list)
    )
    for index, (row, prediction) in enumerate(
        zip(rows, residual_predictions, strict=True)
    ):
        if not math.isfinite(float(prediction)):
            raise ValueError("residual prediction is not finite")
        grouped[(str(row["market_id"]), int(row["decision_ts"]))].append(
            (index, row, float(prediction))
        )
    output: list[dict[str, float] | None] = [None] * len(rows)
    for key, members in grouped.items():
        ordered = sorted(members, key=lambda item: SIDES.index(str(item[1]["side"])))
        if [str(item[1]["side"]) for item in ordered] != list(SIDES):
            raise ValueError(f"UP/DOWN pair is incomplete: {key}")
        provisional = []
        for _, row, residual in ordered:
            anchor = float(np.asarray(row["features"], dtype=float)[SELECTED_MID_INDEX])
            if not math.isfinite(anchor):
                raise ValueError("selected_mid anchor is missing or invalid")
            raw_probability = min(
                1.0 - clip_epsilon,
                max(clip_epsilon, anchor + residual),
            )
            provisional.append((anchor, raw_probability))
        denominator = sum(raw for _, raw in provisional)
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("pair probability denominator is invalid")
        for (index, row, residual), (anchor, raw_probability) in zip(
            ordered, provisional, strict=True
        ):
            probability = raw_probability / denominator
            cost = dict(row["cost_decomposition"])
            entry_ask = float(cost["entry_ask"])
            non_entry_cost = float(cost["total_cost_excluding_entry_ask"])
            action_value = probability - entry_ask - non_entry_cost
            if not all(
                math.isfinite(value)
                for value in (entry_ask, non_entry_cost, probability, action_value)
            ):
                raise ValueError("pair-anchored action value is invalid")
            output[index] = {
                "market_anchor_probability": anchor,
                "predicted_probability_residual": residual,
                "predicted_probability_before_pair_normalization": raw_probability,
                "predicted_probability": probability,
                "entry_ask": entry_ask,
                "non_entry_cost": non_entry_cost,
                "action_value": action_value,
            }
    if any(item is None for item in output):
        raise ValueError("pair-anchored action output is incomplete")
    return [dict(item) for item in output if item is not None]


def render_residual_v2_markdown(report: Mapping[str, Any]) -> str:
    """Render the unchanged gates with explicit v2 architecture semantics."""

    base = render_residual_oof_markdown(report)
    base = base.replace(
        "# BTC 15m cost-aware residual primary slot 001",
        "# BTC 15m market-anchored residual v2 primary slot 001",
        1,
    ).rstrip()
    return (
        base
        + "\n\n## Architecture and governance\n\n"
        + "- Architecture: pooled side-symmetric market-anchored probability residual.\n"
        + "- Pair coherence: UP/DOWN probabilities normalized to sum to one before costs.\n"
        + "- Parent v1 gates and thresholds changed: `False`\n"
        + "- Candidate slots remaining after this evaluation: `1`\n"
        + "- Live shadow or fresh collection authorized by this report: `False`\n"
    )


def _rolling_origin_pair_anchored_predict(
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
            raise ValueError("residual v2 target block population mismatch")
        train_rows = [row for market_id in training_ids for row in rows_by_market[market_id]]
        target_rows = [row for market_id in target_ids for row in rows_by_market[market_id]]
        residual_labels = [_probability_residual_label(row) for row in train_rows]
        train_matrix = _dmatrix(train_rows, labels=residual_labels)
        target_matrix = _dmatrix(target_rows, labels=None)
        booster = xgb.train(
            params=parameters,
            dtrain=train_matrix,
            num_boost_round=boost_rounds,
            verbose_eval=False,
        )
        residual_values = [float(value) for value in booster.predict(target_matrix)]
        action_rows = pair_anchored_action_values(target_rows, residual_values)
        for row, action in zip(target_rows, action_rows, strict=True):
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
                    "predicted_probability_residual": action[
                        "predicted_probability_residual"
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
                "training_residual_labels_sha256": canonical_json_sha256(
                    residual_labels
                ),
                "last_training_market_position": target_start,
                "first_target_market_position": target_start + 1,
                "target_or_future_label_leakage_count": 0,
                "fixed_num_boost_round": boost_rounds,
                "model_parameters_sha256": canonical_json_sha256(parameters),
                "pair_coherence_applied_before_cost_subtraction": True,
                "development_only_forever": True,
                "promotion_evidence_eligible": False,
                "safety": dict(SAFETY),
            }
        )
    return predictions, audits


def _probability_residual_label(row: Mapping[str, Any]) -> float:
    side = str(row["side"])
    outcome = str(row["resolved_outcome"])
    if side not in SIDES or outcome not in SIDES:
        raise ValueError("residual v2 side or outcome is invalid")
    anchor = float(np.asarray(row["features"], dtype=float)[SELECTED_MID_INDEX])
    cost = dict(row["cost_decomposition"])
    payout = 1.0 if side == outcome else 0.0
    expected_target = (
        payout
        - float(cost["entry_ask"])
        - float(cost["total_cost_excluding_entry_ask"])
    )
    if not math.isclose(
        expected_target,
        float(row["target"]),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("residual v2 target/cost reconciliation failed")
    if not math.isfinite(anchor):
        raise ValueError("residual v2 selected_mid anchor is invalid")
    return payout - anchor


def _dmatrix(
    rows: Sequence[Mapping[str, Any]], *, labels: Sequence[float] | None
) -> xgb.DMatrix:
    values = np.vstack([np.asarray(row["features"], dtype=float) for row in rows])
    label_values = np.asarray(labels, dtype=float) if labels is not None else None
    return xgb.DMatrix(
        values,
        label=label_values,
        feature_names=list(FEATURE_NAMES),
        missing=np.nan,
    )


def _v2_market_results_from_predictions(
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


def _build_v2_report(
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
        "pooled_side_symmetric_market_anchored_probability_residual_"
        "with_deterministic_pair_coherence"
    )
    report["parent_v1_gate_and_threshold_bytes_changed"] = False
    report["parent_v1_failed_artifacts_changed"] = False
    report["remaining_candidate_slots"] = 1
    report["additional_candidate_may_be_preregistered_only_if_primary_fails"] = (
        not report["all_gates_passed"]
    )
    return report


def _validate_v2_population(
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
        raise ValueError("residual v2 prediction governance mismatch")
    if any(
        row.get("schema_version") != FOLD_SCHEMA_VERSION
        or row.get("lineage_id") != LINEAGE_ID
        or dict(row.get("safety") or {}) != SAFETY
        for row in fold_audits
    ):
        raise ValueError("residual v2 fold governance mismatch")
    if any(
        row.get("schema_version") != MARKET_RESULT_SCHEMA_VERSION
        or row.get("lineage_id") != LINEAGE_ID
        or dict(row.get("safety") or {}) != SAFETY
        for row in market_results
    ):
        raise ValueError("residual v2 market governance mismatch")
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[(str(row["market_id"]), int(row["decision_ts"]))].append(row)
        cost = dict(row["cost_decomposition"])
        expected = (
            float(row["predicted_probability"])
            - float(cost["entry_ask"])
            - float(cost["total_cost_excluding_entry_ask"])
        )
        if not math.isclose(
            expected,
            float(row["prediction"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("residual v2 action-value reconciliation failed")
    for rows in grouped.values():
        if sorted(str(row["side"]) for row in rows) != sorted(SIDES):
            raise ValueError("residual v2 pair side mismatch")
        if not math.isclose(
            sum(float(row["predicted_probability"]) for row in rows),
            1.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("residual v2 pair probability does not sum to one")
    _validate_frozen_population(
        predictions=_as_v1_predictions(predictions),
        fold_audits=_as_v1_folds(fold_audits),
        market_results=_as_v1_market_results(market_results),
        protocol=protocol,
    )


def _public_v2_dataset_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = _public_dataset_row(row)
    output["schema_version"] = DATASET_SCHEMA_VERSION
    output["lineage_id"] = LINEAGE_ID
    output["market_anchor_probability"] = float(
        np.asarray(row["features"], dtype=float)[SELECTED_MID_INDEX]
    )
    output["probability_residual_target"] = _probability_residual_label(row)
    return output


def _as_v1_predictions(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        _replace_governance(row, V1_PREDICTION_SCHEMA_VERSION, V1_LINEAGE_ID)
        for row in rows
    ]


def _as_v1_folds(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        _replace_governance(row, V1_FOLD_SCHEMA_VERSION, V1_LINEAGE_ID)
        for row in rows
    ]


def _as_v1_market_results(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        _replace_governance(row, V1_MARKET_RESULT_SCHEMA_VERSION, V1_LINEAGE_ID)
        for row in rows
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


def _model_parameters() -> dict[str, Any]:
    return {
        "colsample_bytree": 1.0,
        "eta": 0.03,
        "eval_metric": "rmse",
        "gamma": 0.0,
        "max_bin": 64,
        "max_depth": 2,
        "min_child_weight": 16.0,
        "nthread": 1,
        "objective": "reg:squarederror",
        "reg_alpha": 0.0,
        "reg_lambda": 20.0,
        "seed": 26421,
        "subsample": 1.0,
        "tree_method": "hist",
    }


def _action_policy() -> dict[str, Any]:
    return {
        "decision_order": "chronological",
        "accept_if": "highest_side_prediction>0",
        "fixed_acceptance_threshold": 0.0,
        "side_tie_break_order": ["UP", "DOWN"],
        "one_trade_maximum_per_market": True,
        "NO_TRADE_if_no_positive_prediction": True,
        "NO_TRADE_unit_pnl": 0.0,
        "threshold_search_allowed": False,
    }


def _bootstrap_contract() -> dict[str, Any]:
    return {
        "NO_TRADE_participates_as_zero": True,
        "candidate_and_baseline_share_indices": True,
        "confidence": 0.975,
        "lower_quantile": 0.025,
        "method": "market_level_paired_percentile_bootstrap",
        "resamples": 10000,
        "seed": 26401,
        "unit": "unique_market",
    }


def _cost_stress_contract() -> dict[str, Any]:
    return {
        "action_selection_reused_from_base_cost": True,
        "formula": "gross_price_edge-multiplier*total_cost_relative_to_mid",
        "multipliers": [1.2, 1.5, 2.0],
    }


def _power_contract() -> dict[str, Any]:
    return {
        "confidence": 0.975,
        "target_power": 0.8,
        "effect_haircut": 0.5,
        "maximum_market_count": 2000,
        "required_n_rule": "max_absolute_and_paired_plugin_normal_approximation",
    }


def _rolling_contract() -> dict[str, Any]:
    return {
        "future_market_used_for_fit": False,
        "initial_training_market_count": 200,
        "market_grouped": True,
        "market_order": "frozen_capture_order",
        "oof_market_count": 600,
        "strictly_prior_market_labels_only": True,
        "target_block_count": 6,
        "target_block_size": 100,
    }


__all__ = [
    "pair_anchored_action_values",
    "render_residual_v2_markdown",
    "run_residual_v2_rolling_origin_oof",
    "validate_residual_v2_protocol",
    "validate_v2_lineage_authorization",
    "verify_frozen_residual_v2_oof",
]
