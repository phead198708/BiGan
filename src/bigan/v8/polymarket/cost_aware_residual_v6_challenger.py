"""Prequential lower-quantile proposal gate for residual lineage v6 slot 2."""

from __future__ import annotations

import math
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
from bigan.v8.polymarket.cost_aware_residual_v4 import IMMUTABLE_GATE_IMPLEMENTATION
from bigan.v8.polymarket.cost_aware_residual_v5 import CORRECTOR_FEATURE_NAMES
from bigan.v8.polymarket.cost_aware_residual_v6 import (
    LINEAGE_ID,
    _action_policy,
    _baseline_contract,
    _baseline_rows,
    _bootstrap_contract,
    _cost_stress_contract,
    _dataset_contract,
    _load_registered_parent_rows,
    _power_contract,
    _rolling_contract,
    validate_v6_lineage_authorization,
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

PROTOCOL_SCHEMA_VERSION = "bigan-btc-15m-prequential-lower-quantile-protocol-v6-slot2"
PREDICTION_SCHEMA_VERSION = "bigan-btc-15m-prequential-lower-quantile-prediction-v6-slot2"
FOLD_SCHEMA_VERSION = "bigan-btc-15m-prequential-lower-quantile-fold-v6-slot2"
MARKET_RESULT_SCHEMA_VERSION = "bigan-btc-15m-prequential-lower-quantile-result-v6-slot2"
REPORT_SCHEMA_VERSION = "bigan-btc-15m-prequential-lower-quantile-report-v6-slot2"
MANIFEST_SCHEMA_VERSION = "bigan-btc-15m-prequential-lower-quantile-manifest-v6-slot2"
DATASET_SCHEMA_VERSION = "bigan-btc-15m-prequential-lower-quantile-dataset-row-v6-slot2"

DEFAULT_CONFIG_DIR = (
    REPO_ROOT / "examples/v8/polymarket_configs/BTC-15M-cost-aware-market-residual-v6"
)
DEFAULT_PROTOCOL = DEFAULT_CONFIG_DIR / "residual_v6_challenger_slot_002_protocol.json"
DEFAULT_OUTPUT_DIR = DEFAULT_CONFIG_DIR / "residual_v6_challenger_slot_002_oof"

FIXED_QUANTILE_ALPHA = 0.4
FIXED_NUM_BOOST_ROUND = 96
FIXED_PARAMETERS: dict[str, Any] = {
    "objective": "reg:quantileerror",
    "quantile_alpha": FIXED_QUANTILE_ALPHA,
    "eval_metric": "quantile",
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
    "seed": 26462,
    "nthread": 1,
}

STRUCTURAL_CHANGE = {
    "changed_component": "market_level_proposal_quality_target_and_loss",
    "from": "accept_frozen_v5_proposal_when_its_mean_action_value_is_positive",
    "to": (
        "accept_frozen_v5_proposal_only_when_a_strictly_prior_prequential_"
        "conditional_40th_percentile_unit_PnL_estimate_is_positive"
    ),
    "reason": (
        "v5_passed_all_effect_and_robustness_gates_but_required_N_2598_exceeded_"
        "N_max_2000;_slot_1_dynamic_stopping_degraded_both_LCBs"
    ),
    "expected_mechanism": (
        "retain_v5_side_and_decision_proposals_but_use_a_fixed_lower_conditional_"
        "quantile_to_abstain_when_downside-adjusted_expected_unit_PnL_is_not_positive"
    ),
    "fixed_quantile_alpha": FIXED_QUANTILE_ALPHA,
    "zero_acceptance_threshold_changed": False,
    "cost_baseline_population_or_gate_changed": False,
    "route_side_missingness_or_outlier_filter_added": False,
    "parameter_feature_weight_or_threshold_search_performed": False,
}


def require_challenger_implementation_binding(
    payload: Mapping[str, Any], *, repository_root: Path | str = REPO_ROOT
) -> dict[str, str]:
    """Bind the final slot protocol to this exact executing module."""

    root = Path(repository_root).resolve()
    expected = _descriptor(Path(__file__), root)
    try:
        declared = dict(payload["inputs"]["candidate_implementation"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("v6 challenger implementation descriptor unavailable") from exc
    if declared != expected:
        raise ValueError("v6 challenger implementation does not identify executing module")
    if _verify_descriptor(declared, repository_root=root) != Path(__file__).resolve():
        raise ValueError("v6 challenger implementation resolved to another module")
    return expected


def validate_challenger_protocol(
    payload: Mapping[str, Any],
    *,
    repository_root: Path | str = REPO_ROOT,
    verify_artifacts: bool = True,
) -> None:
    """Fail closed unless the final v6 slot is exactly preregistered."""

    blockers: list[str] = []
    scalars = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "slot_id": "residual-v6-challenger-slot-002",
        "candidate_role": "challenger",
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
        "proposal_quality": _proposal_quality_contract(),
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
    from bigan.v8.polymarket.cost_aware_residual_v2 import GATE_NAMES

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
        "primary_slot_protocol",
        "primary_slot_report",
        "primary_slot_manifest",
        "parent_v4_development_dataset_rows",
        "parent_v4_market_results",
        "parent_v5_predictions",
        "parent_v5_market_results",
        "parent_v5_manifest",
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
            primary = _load_json(resolved["primary_slot_report"])
            if not (
                primary.get("all_gates_passed") is False
                and primary.get("candidate_freeze_allowed") is False
                and primary.get("remaining_candidate_slots") == 1
                and dict(primary.get("safety") or {}) == SAFETY
            ):
                blockers.append("primary_slot_failed_boundary")
    if blockers:
        raise ValueError("residual v6 challenger protocol invalid: " + ", ".join(blockers))


def prequential_lower_quantile_proposal_predict(
    *,
    rows: Sequence[Mapping[str, Any]],
    frozen_v5_predictions: Sequence[Mapping[str, Any]],
    frozen_v5_market_results: Sequence[Mapping[str, Any]],
    population_order: Sequence[str],
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score frozen v5 proposals with strictly prior lower-quantile fits."""

    rows_by_key = {_row_key(row): row for row in rows}
    base_by_key = {_row_key(row): row for row in frozen_v5_predictions}
    result_by_market = {str(row["market_id"]): row for row in frozen_v5_market_results}
    rolling = dict(protocol["rolling_origin"])
    initial = int(rolling["initial_training_market_count"])
    block_size = int(rolling["target_block_size"])
    block_count = int(rolling["target_block_count"])
    oof_order = list(population_order[initial:])
    if (
        len(rows_by_key) != len(rows)
        or len(base_by_key) != len(oof_order) * 4
        or set(result_by_market) != set(oof_order)
    ):
        raise ValueError("v6 challenger frozen proposal population mismatch")
    predictions: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for block_index in range(block_count):
        target_start = block_index * block_size
        prior_ids = oof_order[:target_start]
        target_ids = oof_order[target_start : target_start + block_size]
        if len(target_ids) != block_size:
            raise ValueError("v6 challenger target block population mismatch")
        prior_pairs = _accepted_proposals(
            prior_ids, rows_by_key, base_by_key, result_by_market
        )
        target_pairs = _all_target_rows(
            target_ids, rows_by_key, base_by_key, result_by_market
        )
        model: xgb.Booster | None = None
        if prior_ids:
            if not prior_pairs:
                raise ValueError("v6 challenger has no strictly prior accepted proposals")
            model = _fit_quality_model(prior_pairs, protocol)
        scored_selected: dict[tuple[str, int, str], float] = {}
        selected_pairs = [item for item in target_pairs if item[2] is True]
        if model is None:
            scored_selected = {
                _row_key(base): float(base["prediction"])
                for _, base, selected in selected_pairs
                if selected
            }
        else:
            scores = _predict_quality_model(model, [(row, base) for row, base, _ in selected_pairs])
            scored_selected = {
                _row_key(base): score
                for (_, base, _), score in zip(selected_pairs, scores, strict=True)
            }
        for row, base, selected in target_pairs:
            key = _row_key(base)
            score = float(scored_selected[key]) if selected else 0.0
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
                    "score_semantics": (
                        "frozen_v5_identity_no_prior_meta_history"
                        if model is None
                        else "strictly_prior_conditional_40th_percentile_unit_PnL"
                    ),
                    "frozen_v5_proposal_selected": selected,
                    "frozen_v5_prediction": float(base["prediction"]),
                    "frozen_v5_predicted_probability": float(base["predicted_probability"]),
                    "realized_unit_net_pnl_if_action": float(
                        base["realized_unit_net_pnl_if_action"]
                    ),
                    "resolved_outcome": base["resolved_outcome"],
                    "cost_decomposition": base["cost_decomposition"],
                    "feature_row_sha256": base["feature_row_sha256"],
                    "chronological_block": block_index + 1,
                    "strictly_prior_training_market_count": initial + target_start,
                    "strictly_prior_meta_training_market_count": len(prior_ids),
                    "strictly_prior_meta_training_proposal_count": len(prior_pairs),
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
                "strictly_prior_training_market_count": initial + target_start,
                "target_market_count": len(target_ids),
                "training_market_ids_sha256": canonical_json_sha256(
                    list(population_order[: initial + target_start])
                ),
                "target_market_ids_sha256": canonical_json_sha256(target_ids),
                "last_training_market_position": initial + target_start,
                "first_target_market_position": initial + target_start + 1,
                "target_or_future_label_leakage_count": 0,
                "strictly_prior_meta_training_market_count": len(prior_ids),
                "strictly_prior_meta_training_proposal_count": len(prior_pairs),
                "meta_model_applied": model is not None,
                "fixed_quantile_alpha": FIXED_QUANTILE_ALPHA,
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


def run_challenger_oof(
    *,
    protocol_path: Path | str = DEFAULT_PROTOCOL,
    expected_protocol_sha256: str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    source_commit: str,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Execute the final preregistered v6 candidate exactly once."""

    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    output = Path(output_dir).resolve()
    if not protocol_file.is_relative_to(root) or not output.is_relative_to(root):
        raise ValueError("residual v6 challenger paths must remain repository-local")
    if sha256_file(protocol_file) != expected_protocol_sha256.lower():
        raise ValueError("residual v6 challenger protocol SHA-256 mismatch")
    sidecar = protocol_file.with_suffix(".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != (
        expected_protocol_sha256
    ):
        raise ValueError("residual v6 challenger protocol is not SHA-frozen")
    if not _looks_like_git_sha(source_commit):
        raise ValueError("source_commit must be a full lowercase Git SHA")
    protocol = _load_json(protocol_file)
    validate_challenger_protocol(protocol, repository_root=root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"residual v6 challenger output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    rows, parent_markets, population_order = _load_registered_parent_rows(
        protocol=protocol, repository_root=root
    )
    rows = [_canonicalize_row(row) for row in rows]
    inputs = dict(protocol["inputs"])
    base_predictions = _load_jsonl(
        _verify_descriptor(inputs["parent_v5_predictions"], repository_root=root)
    )
    base_markets = _load_jsonl(
        _verify_descriptor(inputs["parent_v5_market_results"], repository_root=root)
    )
    predictions, folds = prequential_lower_quantile_proposal_predict(
        rows=rows,
        frozen_v5_predictions=base_predictions,
        frozen_v5_market_results=base_markets,
        population_order=population_order,
        protocol=protocol,
    )
    markets = _market_results(
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
    dataset_path = output / "residual_v6_challenger_development_dataset_rows.jsonl"
    prediction_path = output / "residual_v6_challenger_oof_predictions.jsonl"
    fold_path = output / "residual_v6_challenger_oof_fold_audits.jsonl"
    market_path = output / "residual_v6_challenger_oof_market_results.jsonl"
    report_path = output / "residual_v6_challenger_oof_report.json"
    markdown_path = output / "residual_v6_challenger_oof_report.md"
    _write_new_jsonl(dataset_path, [_public_dataset_row(row) for row in rows])
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
        "candidate_freeze_allowed": report["all_gates_passed"],
        "next_stage_authorization_required_even_if_all_gates_pass": True,
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "safety": dict(SAFETY),
    }
    manifest_artifact = _write_new_frozen_json(
        output / "residual_v6_challenger_oof_manifest.json", manifest
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
    """Rebuild the final v6 report and verify every frozen binding."""

    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    output = Path(output_dir).resolve()
    protocol = _verified_json(protocol_file)
    validate_challenger_protocol(protocol, repository_root=root)
    manifest = _verified_json(output / "residual_v6_challenger_oof_manifest.json")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("residual v6 challenger manifest schema mismatch")
    if manifest.get("protocol") != _descriptor(protocol_file, root):
        raise ValueError("residual v6 challenger manifest protocol binding mismatch")
    if manifest.get("candidate_implementation") != require_challenger_implementation_binding(
        protocol, repository_root=root
    ):
        raise ValueError("residual v6 challenger implementation binding mismatch")
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
        raise ValueError("residual v6 challenger artifact set mismatch")
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
        raise ValueError("residual v6 challenger dataset governance mismatch")
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
    _assert_semantically_equal(rebuilt, frozen, path="residual_v6_challenger_oof_report")
    if render_challenger_markdown(rebuilt) != artifacts["report_markdown"].read_text(
        encoding="utf-8"
    ):
        raise ValueError("residual v6 challenger Markdown does not reproduce")
    return {
        "verification_passed": True,
        "all_gates_passed": bool(frozen["all_gates_passed"]),
        "failed_gates": list(frozen["failed_gates"]),
        "candidate_budget_exhausted": True,
        "oof_market_count": len(markets),
        "manifest_sha256": sha256_file(output / "residual_v6_challenger_oof_manifest.json"),
        "actual_executing_module_binding_verified": True,
        "parent_v1_through_v5_and_primary_slot_immutable": True,
        "safety": dict(SAFETY),
    }


def _accepted_proposals(
    market_ids: Sequence[str],
    rows_by_key: Mapping[tuple[str, int, str], Mapping[str, Any]],
    base_by_key: Mapping[tuple[str, int, str], Mapping[str, Any]],
    result_by_market: Mapping[str, Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    output = []
    for market_id in market_ids:
        result = result_by_market[market_id]
        if result.get("candidate_accepted") is not True:
            continue
        key = (
            market_id,
            int(result["candidate_decision_ts"]),
            str(result["candidate_selected_side"]),
        )
        row = rows_by_key.get(key)
        base = base_by_key.get(key)
        if row is None or base is None:
            raise ValueError("v6 challenger accepted proposal row missing")
        if not math.isclose(
            float(result["candidate_unit_net_pnl"]),
            float(base["realized_unit_net_pnl_if_action"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("v6 challenger proposal outcome binding mismatch")
        output.append((row, base))
    return output


def _all_target_rows(
    market_ids: Sequence[str],
    rows_by_key: Mapping[tuple[str, int, str], Mapping[str, Any]],
    base_by_key: Mapping[tuple[str, int, str], Mapping[str, Any]],
    result_by_market: Mapping[str, Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any], bool]]:
    output = []
    for market_id in market_ids:
        result = result_by_market[market_id]
        selected_key = None
        if result.get("candidate_accepted") is True:
            selected_key = (
                market_id,
                int(result["candidate_decision_ts"]),
                str(result["candidate_selected_side"]),
            )
        keys = sorted(
            (key for key in base_by_key if key[0] == market_id),
            key=lambda key: (key[1], 0 if key[2] == "UP" else 1),
        )
        if len(keys) != 4 or [key[2] for key in keys] != ["UP", "DOWN", "UP", "DOWN"]:
            raise ValueError("v6 challenger frozen proposal action grid changed")
        for key in keys:
            row = rows_by_key.get(key)
            if row is None:
                raise ValueError("v6 challenger feature row missing")
            output.append((row, base_by_key[key], key == selected_key))
    return output


def _fit_quality_model(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    protocol: Mapping[str, Any],
) -> xgb.Booster:
    labels = [float(base["realized_unit_net_pnl_if_action"]) for _, base in pairs]
    return xgb.train(
        params=dict(protocol["model"]["parameters"]),
        dtrain=_quality_dmatrix(pairs, labels=labels),
        num_boost_round=int(protocol["model"]["fixed_num_boost_round"]),
        verbose_eval=False,
    )


def _predict_quality_model(
    model: xgb.Booster,
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> list[float]:
    values = [float(value) for value in model.predict(_quality_dmatrix(pairs, labels=None))]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("v6 challenger produced a non-finite prediction")
    return values


def _quality_dmatrix(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    labels: Sequence[float] | None,
) -> xgb.DMatrix:
    values = np.vstack([_quality_features(row, base) for row, base in pairs])
    target = None if labels is None else np.asarray(labels, dtype=float)
    if target is not None and target.shape != (len(pairs),):
        raise ValueError("v6 challenger labels do not align")
    return xgb.DMatrix(
        values,
        label=target,
        feature_names=list(CORRECTOR_FEATURE_NAMES),
        missing=np.nan,
    )


def _quality_features(row: Mapping[str, Any], base: Mapping[str, Any]) -> np.ndarray:
    raw = dict(row["features"])
    if tuple(raw) != tuple(FEATURE_NAMES):
        raise ValueError("v6 challenger inherited 108-feature order changed")
    cost = dict(row["cost_decomposition"])
    values = [np.nan if raw[name] is None else float(raw[name]) for name in FEATURE_NAMES]
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
        raise ValueError("v6 challenger proposal feature vector size changed")
    return output


def _canonicalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = deepcopy(dict(row))
    raw = dict(row["features"])
    if set(raw) != set(FEATURE_NAMES) or len(raw) != len(FEATURE_NAMES):
        raise ValueError("v6 challenger inherited feature names changed")
    output["features"] = {name: raw[name] for name in FEATURE_NAMES}
    return output


def _row_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return str(row["market_id"]), int(row["decision_ts"]), str(row["side"])


def _market_results(
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
        "frozen_v5_proposal_with_prequential_conditional_lower_quantile_quality_model"
    )
    report["candidate_role"] = "challenger"
    report["immutable_gate_implementation_sha256"] = IMMUTABLE_GATE_IMPLEMENTATION["sha256"]
    report["actual_executing_module_exactly_bound"] = True
    report["existing_gate_threshold_cost_baseline_population_changed"] = False
    report["parent_v1_through_v5_and_primary_slot_artifacts_changed"] = False
    report["remaining_candidate_slots"] = 0
    report["candidate_budget_exhausted"] = True
    report["additional_candidate_allowed"] = False
    report["next_stage_authorization_required_even_if_all_gates_pass"] = True
    report["structural_change"] = dict(protocol["structural_change"])
    report["proposal_quality"] = {
        "fixed_quantile_alpha": FIXED_QUANTILE_ALPHA,
        "meta_model_applied_by_block": [bool(row["meta_model_applied"]) for row in fold_audits],
        "strictly_prior_meta_training_market_counts": [
            int(row["strictly_prior_meta_training_market_count"]) for row in fold_audits
        ],
        "target_or_future_label_used_for_fit": False,
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
        or dict(row.get("safety") or {}) != SAFETY
        for row in predictions
    ):
        raise ValueError("v6 challenger prediction governance mismatch")
    if any(
        row.get("schema_version") != FOLD_SCHEMA_VERSION
        or row.get("lineage_id") != LINEAGE_ID
        or row.get("target_or_future_label_leakage_count") != 0
        or dict(row.get("safety") or {}) != SAFETY
        for row in fold_audits
    ):
        raise ValueError("v6 challenger fold governance mismatch")
    if any(
        row.get("schema_version") != MARKET_RESULT_SCHEMA_VERSION
        or row.get("lineage_id") != LINEAGE_ID
        or dict(row.get("safety") or {}) != SAFETY
        for row in market_results
    ):
        raise ValueError("v6 challenger market governance mismatch")
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


def render_challenger_markdown(report: Mapping[str, Any]) -> str:
    base = (
        render_residual_oof_markdown(report)
        .replace(
            "# BTC 15m cost-aware residual primary slot 001",
            "# BTC 15m lower-quantile residual v6 challenger slot 002",
            1,
        )
        .rstrip()
    )
    return (
        base
        + "\n\n## Proposal-quality architecture\n\n"
        + "- Proposal source: frozen v5 side and decision action.\n"
        + "- Risk target: strictly prior conditional 40th percentile of unit PnL.\n"
        + "- First block: frozen v5 identity because no prior v5 OOF proposal labels exist.\n"
        + "- Grid, feature, weight or threshold search: `False`\n"
        + "- Existing gates, zero threshold, N_max, costs, baseline and population changed: `False`\n"
        + "- Candidate budget exhausted after this evaluation: `True`\n"
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
        "execution_policy": "HOLD_TO_SETTLEMENT",
        "NO_TRADE_value": 0.0,
        "post_close_training_labels_only": True,
        "proposal_source": "immutable_v5_selected_side_and_decision",
        "proposal_quality_target": "realized_unit_net_PnL_of_frozen_v5_proposal",
        "score_semantics": "conditional_40th_percentile_unit_PnL",
    }


def _feature_contract() -> dict[str, Any]:
    return {
        "base_feature_count": 54,
        "ordered_feature_count": 108,
        "ordered_feature_names_sha256": (
            "a61ef8953a2801d3e404d376261f306846cae5a4bf3ef3c97918572c48720e89"
        ),
        "proposal_context_features": [
            "frozen_v5_predicted_probability",
            "frozen_v5_action_value",
            "entry_ask",
            "total_cost_excluding_entry_ask",
        ],
        "proposal_quality_ordered_feature_count": 112,
        "proposal_quality_ordered_feature_names_sha256": canonical_json_sha256(
            list(CORRECTOR_FEATURE_NAMES)
        ),
        "decision_time_causal_inputs_only": True,
        "market_horizon_seconds": 900,
        "native_missing_value": "nan",
        "missing_values_encoded_as_zero": False,
        "shared_side_symmetric_model": True,
        "side_identity_feature_allowed": False,
        "feature_search_allowed": False,
        "new_raw_feature_added": False,
    }


def _model_contract() -> dict[str, Any]:
    return {
        "family": "prequential_market_proposal_lower_quantile_xgboost",
        "fixed_num_boost_round": FIXED_NUM_BOOST_ROUND,
        "fixed_quantile_alpha": FIXED_QUANTILE_ALPHA,
        "parameters": dict(FIXED_PARAMETERS),
        "frozen_proposal_source": "residual_v5_challenger_slot_002",
        "model_selection_or_early_stopping_performed": False,
        "route_or_expert_filtering_allowed": False,
    }


def _proposal_quality_contract() -> dict[str, Any]:
    return {
        "method": "expanding_prequential_market_grouped_lower_quantile_regression",
        "training_population": "strictly_prior_accepted_frozen_v5_OOF_proposals_only",
        "target_population": "all_frozen_v5_OOF_markets_without_filtering_or_reordering",
        "first_block_behavior": "identity_frozen_v5_proposal_no_prior_meta_labels",
        "later_block_behavior": "accept_only_if_fixed_conditional_quantile_score_gt_zero",
        "frozen_v5_side_or_decision_may_change": False,
        "current_or_future_target_label_used": False,
        "fixed_quantile_alpha": FIXED_QUANTILE_ALPHA,
        "threshold_search_allowed": False,
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
    "prequential_lower_quantile_proposal_predict",
    "require_challenger_implementation_binding",
    "run_challenger_oof",
    "validate_challenger_protocol",
    "verify_frozen_challenger_oof",
]
