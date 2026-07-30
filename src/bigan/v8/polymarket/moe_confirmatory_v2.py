"""Build and verify the terminalized BTC 15m MoE v2 precollection lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import xgboost as xgb

from bigan.v8.polymarket.challenge_development_lane import (
    atomic_write_json,
    load_jsonl,
    sha256_file,
)
from bigan.v8.polymarket.challenge_model_15m_training import (
    _load_side_symmetric_rows,
    _matrix,
    _verify_finalized_index,
)
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.moe_confirmatory_lineage import (
    _raw_row_route,
    deterministic_moe_route,
    frozen_expert_or_fallback,
)
from bigan.v8.polymarket.regime_adaptive_candidate_evaluation import (
    FEATURE_NAMES,
    _selected_rows,
)
from bigan.v8.polymarket.regime_adaptive_lineage import REPO_ROOT

V1_LINEAGE_ID = "BTC-15M-MoE-confirmatory-v1"
V2_LINEAGE_ID = "BTC-15M-MoE-confirmatory-v2"
PARENT_LINEAGE_ID = "BTC-15M-regime-adaptive-v1"
CANDIDATE_ID = "mixture_of_experts"
ARCHITECTURE_TYPE = (
    "deterministic_regime_router_with_conditional_experts_and_global_fallback"
)
BASE_COMMIT = "a37320d4f8aa8ccb8176132def6429c1127758ba"
BASE_TREE = "f79e97fe58f7858bfa03e2851770de9eee506142"
PARENT_RESULT_COMMIT = "fd848defad7874db5decc96b9ef2de07e8007b1e"
ORIGINAL_RECORDED_SOURCE_COMMIT = "364fd65b08849cb36227a3c4bb1b55a62cc68825"
REACHABLE_EVALUATOR_COMMIT = "364fd65afa4908170dbc2ae5ff4f71c8a2475573"
V1_BUNDLE_HASH = "30d180b028c83146fafd81c8b81269f51fa567b30bc5ab4d3577dd99c256dcf8"
EXPERT_IDS = ("high_vol", "bullish", "bearish", "low_vol")
CANDIDATE_SAMPLE_SIZES = (160, 240, 320, 480, 640, 800, 960, 1200)
CREATED_AT = "2026-07-30T09:00:00+00:00"
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


def build_v2_precollection_lineage(
    *,
    repository_root: Path | str | None = None,
    created_at: str = CREATED_AT,
) -> dict[str, Any]:
    """Build the v1 terminal record and the complete blocked v2 artifact graph."""

    repo_root = Path(repository_root or REPO_ROOT).resolve()
    v1_dir = repo_root / "examples/v8/polymarket_configs" / V1_LINEAGE_ID
    v2_dir = repo_root / "examples/v8/polymarket_configs" / V2_LINEAGE_ID
    artifact_root = repo_root / "examples/v8/polymarket_artifacts/sha256"
    if v2_dir.exists():
        raise FileExistsError(f"v2 config directory already exists: {v2_dir}")
    v2_dir.mkdir(parents=True)

    terminal = _terminalize_v1(v1_dir=v1_dir, created_at=created_at)
    genesis = _write_genesis(v2_dir=v2_dir, created_at=created_at)
    contracts = _write_pretraining_contracts(
        repo_root=repo_root,
        v2_dir=v2_dir,
        created_at=created_at,
    )
    bundle = _train_and_freeze_bundle(
        repo_root=repo_root,
        v2_dir=v2_dir,
        artifact_root=artifact_root,
        contracts=contracts,
        created_at=created_at,
    )
    baseline = _write_matched_baseline_contract(
        repo_root=repo_root,
        v2_dir=v2_dir,
        bundle=bundle,
        created_at=created_at,
    )
    runtime = validate_v2_artifact_in_fresh_environment(
        graph_path=v2_dir / "moe_artifact_graph.json",
        expected_graph_sha256=sha256_file(v2_dir / "moe_artifact_graph.json"),
        repository_root=repo_root,
    )
    runtime_report = _write_runtime_report(
        v2_dir=v2_dir,
        bundle=bundle,
        runtime=runtime,
        created_at=created_at,
    )
    power = _write_power_protocol_and_analysis(
        repo_root=repo_root,
        v2_dir=v2_dir,
        bundle=bundle,
        baseline=baseline,
        created_at=created_at,
    )
    quality = _write_collection_quality_analysis(
        repo_root=repo_root,
        v2_dir=v2_dir,
        target=int(power["analysis"]["selected_confirmatory_market_count"]),
        created_at=created_at,
    )
    reporting = _write_reporting_contract(v2_dir=v2_dir, created_at=created_at)
    collector = _write_collector_protocol(
        v2_dir=v2_dir,
        power=power,
        quality=quality,
        created_at=created_at,
    )
    protocol = _write_confirmatory_protocol(
        v2_dir=v2_dir,
        bundle=bundle,
        baseline=baseline,
        runtime_report=runtime_report,
        power=power,
        collector=collector,
        reporting=reporting,
        created_at=created_at,
    )
    authorization = _write_authorization_template(
        v2_dir=v2_dir,
        bundle=bundle,
        baseline=baseline,
        runtime_report=runtime_report,
        power=power,
        quality=quality,
        collector=collector,
        protocol=protocol,
        reporting=reporting,
        created_at=created_at,
    )
    lineage = _write_lineage_manifest(
        v2_dir=v2_dir,
        genesis=genesis,
        terminal=terminal,
        bundle=bundle,
        runtime_report=runtime_report,
        power=power,
        quality=quality,
        protocol=protocol,
        collector=collector,
        reporting=reporting,
        authorization=authorization,
        created_at=created_at,
    )
    return {
        "v1_terminal_record_sha256": terminal["sha256"],
        "v2_lineage_manifest_sha256": lineage["sha256"],
        "v2_genesis_decision_sha256": genesis["sha256"],
        "selected_num_boost_round": bundle["selected_num_boost_round"],
        "bundle_hash": bundle["bundle_hash"],
        "baseline_sha256": bundle["fallback_sha256"],
        "expert_hashes": bundle["expert_hashes"],
        "runtime_validation_passed": runtime["mandatory_gate_passed"],
        "selected_confirmatory_market_count": power["analysis"][
            "selected_confirmatory_market_count"
        ],
        "attempt_cap": quality["analysis"]["attempt_cap"],
        "fresh_collection_authorized": False,
        "fresh_outcomes_opened": False,
        "safety": dict(SAFETY),
    }


def _terminalize_v1(*, v1_dir: Path, created_at: str) -> dict[str, Any]:
    inventory = []
    for path in sorted(v1_dir.iterdir()):
        if not path.is_file() or path.name.startswith(
            f"{V1_LINEAGE_ID}-terminal-record."
        ):
            continue
        inventory.append(
            {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    if len(inventory) != 44:
        raise ValueError("v1 terminal inventory must contain exactly 44 base artifacts")
    output = v1_dir / f"{V1_LINEAGE_ID}-terminal-record.json"
    payload = {
        "schema_version": "bigan-btc-15m-moe-confirmatory-terminal-record-v1",
        "lineage_id": V1_LINEAGE_ID,
        "created_at": created_at,
        "terminalized_at_base_commit": BASE_COMMIT,
        "terminalized_at_base_tree": BASE_TREE,
        "status": "terminal_permanently_collection_forbidden",
        "terminal_reason_codes": [
            "matched_baseline_information_budget_mismatch",
            "confirmatory_protocol_underpowered_at_observed_development_effect",
            "phase_0_provenance_failure_not_resolved",
            "internal_resume_record_not_independently_auditable",
            "runtime_fixture_coverage_incomplete",
            "full_suite_not_clean_or_baseline_compared",
        ],
        "v1_fresh_collection_permanently_forbidden": True,
        "v1_fresh_outcome_access_permanently_forbidden": True,
        "v1_candidate_artifacts_preserved_for_audit": True,
        "v1_artifacts_may_not_be_mutated_into_v2": True,
        "artifact_inventory_scope": (
            "all_44_tracked_v1_artifacts_at_a37320d_before_this_terminal_record"
        ),
        "artifact_inventory_count": len(inventory),
        "artifact_inventory": inventory,
        "v1_bundle_hash": V1_BUNDLE_HASH,
        "collection_state": {
            "fresh_collection_authorized": False,
            "fresh_collection_started": False,
            "fresh_outcomes_opened": False,
        },
        "safety": dict(SAFETY),
    }
    _write_frozen_json(output, payload)
    return _descriptor(output)


def _write_genesis(*, v2_dir: Path, created_at: str) -> dict[str, Any]:
    output = v2_dir / "lineage_genesis_decision.json"
    payload = {
        "schema_version": "bigan-btc-15m-moe-confirmatory-lineage-genesis-v2",
        "lineage_id": V2_LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at": created_at,
        "parent_lineage_id": PARENT_LINEAGE_ID,
        "parent_result_commit": PARENT_RESULT_COMMIT,
        "v1_engineering_reference_lineage": V1_LINEAGE_ID,
        "v1_engineering_reference_commit": BASE_COMMIT,
        "original_recorded_source_commit": ORIGINAL_RECORDED_SOURCE_COMMIT,
        "original_recorded_source_commit_reachable": False,
        "reachable_evaluator_commit": REACHABLE_EVALUATOR_COMMIT,
        "exact_identity_proven": False,
        "new_lineage_created_instead_of_overriding_failed_v1": True,
        "parent_evidence_role": "hypothesis_generation_only",
        "parent_evidence_is_not_fresh_validation": True,
        "no_parent_gate_was_relaxed": True,
        "allowed_hypothesis_generation_inputs": [
            "immutable_parent_prediction_rows",
            "immutable_parent_fold_audits",
            "passed_independent_metric_reconciliation",
            "passed_moe_route_and_fallback_attribution",
            "v1_artifact_design_as_engineering_reference",
        ],
        "approver": None,
        "request_url": None,
        "issue_id": None,
        "approval_timestamp": None,
        "fresh_collection_authorized": False,
        "fresh_collection_started": False,
        "fresh_outcomes_opened": False,
        "safety": dict(SAFETY),
    }
    _write_frozen_json(output, payload)
    return _descriptor(output)


def _write_pretraining_contracts(
    *,
    repo_root: Path,
    v2_dir: Path,
    created_at: str,
) -> dict[str, dict[str, Any]]:
    v1_dir = repo_root / "examples/v8/polymarket_configs" / V1_LINEAGE_ID
    parent_dir = repo_root / "examples/v8/polymarket_configs" / PARENT_LINEAGE_ID
    index_path = (
        repo_root
        / "examples/v8/polymarket_runs/challenge-model-development-btc-updown-15m-v1"
        / "finalized_development_corpus_index.jsonl"
    )
    router = _load_json(v1_dir / "moe_router_contract.json")
    router.update(
        {
            "schema_version": "bigan-btc-15m-moe-router-contract-v2",
            "lineage_id": V2_LINEAGE_ID,
            "created_at": created_at,
            "source_v1_contract": _descriptor(
                v1_dir / "moe_router_contract.json"
            ),
        }
    )
    router["support_and_fallback"][
        "fallback_training_population"
    ] = "all_113_development_markets_equal_to_matched_global_baseline"
    router_path = v2_dir / "moe_router_contract.json"
    _write_frozen_json(router_path, router)

    feature = _load_json(v1_dir / "moe_feature_contract.json")
    feature.update(
        {
            "schema_version": "bigan-btc-15m-moe-static-feature-contract-v2",
            "lineage_id": V2_LINEAGE_ID,
            "created_at": created_at,
            "source_v1_contract": _descriptor(
                v1_dir / "moe_feature_contract.json"
            ),
        }
    )
    feature_path = v2_dir / "moe_feature_contract.json"
    _write_frozen_json(feature_path, feature)

    cost = _load_json(v1_dir / "moe_cost_and_action_contract.json")
    cost.update(
        {
            "schema_version": "bigan-btc-15m-moe-cost-and-action-contract-v2",
            "lineage_id": V2_LINEAGE_ID,
            "created_at": created_at,
            "source_v1_contract": _descriptor(
                v1_dir / "moe_cost_and_action_contract.json"
            ),
        }
    )
    cost_path = v2_dir / "moe_cost_and_action_contract.json"
    _write_frozen_json(cost_path, cost)

    route_support = {"high_vol": 46, "bullish": 40, "bearish": 57, "low_vol": 18}
    candidate = {
        "schema_version": "bigan-btc-15m-moe-confirmatory-candidate-contract-v2",
        "lineage_id": V2_LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "architecture_type": ARCHITECTURE_TYPE,
        "created_at": created_at,
        "role": "single_static_fair_information_budget_candidate_freeze",
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "parent_hypothesis": {
            "parent_lineage_id": PARENT_LINEAGE_ID,
            "parent_result_commit": PARENT_RESULT_COMMIT,
            "parent_evidence_role": "hypothesis_generation_only",
            "parent_evidence_is_fresh_validation": False,
            "parent_selected_candidate_id": None,
            "candidate_budget_consumed": 5,
            "candidate_budget_maximum": 5,
        },
        "provenance_boundary": {
            "original_recorded_source_commit": ORIGINAL_RECORDED_SOURCE_COMMIT,
            "original_recorded_source_commit_reachable": False,
            "reachable_evaluator_commit": REACHABLE_EVALUATOR_COMMIT,
            "exact_identity_proven": False,
        },
        "hypothesis_inputs": {
            "metric_reconciliation": _descriptor(
                v1_dir / "development_metric_reconciliation_report.json"
            ),
            "route_attribution": _descriptor(
                v1_dir / "moe_route_attribution_report.json"
            ),
            "parent_predictions": _descriptor(
                repo_root
                / "examples/v8/polymarket_training_artifacts/"
                "BTC-15M-regime-adaptive-v1-development-evaluation/"
                "development_oof_predictions.jsonl"
            ),
            "parent_fold_audits": _descriptor(
                repo_root
                / "examples/v8/polymarket_training_artifacts/"
                "BTC-15M-regime-adaptive-v1-development-evaluation/"
                "development_fold_audits.jsonl"
            ),
        },
        "contracts": {
            "router": _descriptor(router_path),
            "features": _descriptor(feature_path),
            "cost_and_action": _descriptor(cost_path),
        },
        "development_population": {
            **_descriptor(index_path),
            "market_count": 113,
            "role": "development_training_only",
            "promotion_evidence_eligible": False,
        },
        "model_family": {
            "name": (
                "xgboost_shared_side_symmetric_pair_normalized_win_probability"
            ),
            "parameters": {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "nthread": 1,
                "max_depth": 2,
                "eta": 0.05,
                "min_child_weight": 8.0,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_lambda": 10.0,
                "reg_alpha": 0.0,
                "gamma": 0.0,
                "max_bin": 64,
                "seed": 26015,
            },
            "maximum_num_boost_round": 400,
            "early_stopping_rounds": 30,
            "hyperparameter_search_allowed": False,
        },
        "static_training_protocol": {
            "round_selection_train_market_count": 93,
            "round_selection_validation_market_count": 20,
            "round_selection_split_use": "selected_num_boost_round_only",
            "discard_round_selection_fitted_model": True,
            "final_global_baseline_training_market_count": 113,
            "final_global_baseline_uses_validation_labels": True,
            "expert_training_population": (
                "all_113_development_markets_with_rows_assigned_to_frozen_route"
            ),
            "same_selected_num_boost_round_for_baseline_and_available_experts": True,
            "candidate_and_baseline_have_equal_development_information_budget": True,
            "minimum_expert_support": 20,
            "route_support": route_support,
            "expert_availability": {
                route: count >= 20 for route, count in route_support.items()
            },
            "low_vol_behavior": "full_113_matched_global_baseline_fallback",
        },
        "frozen_behavior": {
            "candidate_id": CANDIDATE_ID,
            "router_unchanged_from_v1": True,
            "expert_ids": list(EXPERT_IDS),
            "pair_normalization_required": True,
            "fixed_action_threshold": 0.0,
            "true_paired_executable_asks_required": True,
            "complement_proxy_allowed": False,
            "unit_sizing": True,
            "execution_policy": "HOLD_TO_SETTLEMENT",
            "side_filters_allowed": False,
            "missingness_filters_allowed": False,
            "route_filters_allowed": False,
            "uncertainty_overlays_allowed": False,
            "dynamic_sizing_allowed": False,
            "post_hoc_abstention_allowed": False,
        },
        "state": {
            "candidate_contract_frozen": True,
            "fresh_collection_authorized": False,
            "fresh_collection_started": False,
            "fresh_outcomes_opened": False,
        },
        "safety": dict(SAFETY),
    }
    if sha256_file(parent_dir / "candidate_family_protocol.json") != (
        "b1008633ddd5fad3228bbf29031bd0cb5f994663cabae10931c1913f639a8c26"
    ):
        raise ValueError("parent candidate family protocol changed")
    candidate_path = v2_dir / "moe_candidate_contract.json"
    _write_frozen_json(candidate_path, candidate)
    return {
        "candidate": {"path": candidate_path, "payload": candidate},
        "router": {"path": router_path, "payload": router},
        "features": {"path": feature_path, "payload": feature},
        "cost": {"path": cost_path, "payload": cost},
    }


def _train_and_freeze_bundle(
    *,
    repo_root: Path,
    v2_dir: Path,
    artifact_root: Path,
    contracts: Mapping[str, Mapping[str, Any]],
    created_at: str,
) -> dict[str, Any]:
    candidate = contracts["candidate"]["payload"]
    index_path = repo_root / candidate["development_population"]["path"]
    index_rows = _verify_finalized_index(index_path=index_path, repo_root=repo_root)
    rows, input_corpora = _load_side_symmetric_rows(index_rows, repo_root=repo_root)
    ordered_markets = sorted(
        {
            (int(row["market_start_ts"]), str(row["market_id"]))
            for row in rows
        }
    )
    if len(ordered_markets) != 113:
        raise ValueError("v2 training requires exactly 113 development markets")
    router = contracts["router"]["payload"]
    route_feature_contract = {
        "derived_regime_features": {
            "btc_return_regime": router["router_inputs"]["btc_return_regime"],
            "volatility_bucket": router["router_inputs"]["volatility_bucket"],
        }
    }
    rows_by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_route[_raw_row_route(row, route_feature_contract)].append(row)
    market_order = {
        market_id: index for index, (_, market_id) in enumerate(ordered_markets)
    }
    route_market_ids = {
        route: sorted(
            {str(row["market_id"]) for row in rows_by_route[route]},
            key=market_order.__getitem__,
        )
        for route in EXPERT_IDS
    }
    expected_support = candidate["static_training_protocol"]["route_support"]
    if {
        route: len(route_market_ids[route]) for route in EXPERT_IDS
    } != expected_support:
        raise ValueError("v2 route support changed")

    train_ids = [market_id for _, market_id in ordered_markets[:93]]
    validation_ids = [market_id for _, market_id in ordered_markets[93:]]
    train_id_set = set(train_ids)
    validation_id_set = set(validation_ids)
    train_rows = [
        row for row in rows if str(row["market_id"]) in train_id_set
    ]
    validation_rows = [
        row for row in rows if str(row["market_id"]) in validation_id_set
    ]
    parameters = dict(candidate["model_family"]["parameters"])
    selection_model = xgb.train(
        params=parameters,
        dtrain=_matrix(train_rows, FEATURE_NAMES, label_field="settlement_payout"),
        num_boost_round=int(candidate["model_family"]["maximum_num_boost_round"]),
        evals=[
            (
                _matrix(train_rows, FEATURE_NAMES, label_field="settlement_payout"),
                "train",
            ),
            (
                _matrix(
                    validation_rows,
                    FEATURE_NAMES,
                    label_field="settlement_payout",
                ),
                "validation",
            ),
        ],
        early_stopping_rounds=int(
            candidate["model_family"]["early_stopping_rounds"]
        ),
        verbose_eval=False,
    )
    selected_num_boost_round = int(selection_model.best_iteration) + 1
    selection_best_score = float(selection_model.best_score)
    del selection_model

    staging = Path(
        tempfile.mkdtemp(prefix=".BTC-15M-MoE-confirmatory-v2.", dir=artifact_root)
    )
    staging_moved = False
    try:
        for key, filename in (
            ("candidate", "moe_candidate_contract.json"),
            ("router", "moe_router_contract.json"),
            ("features", "moe_feature_contract.json"),
            ("cost", "moe_cost_and_action_contract.json"),
        ):
            shutil.copyfile(contracts[key]["path"], staging / filename)
        ordered_feature_path = staging / "ordered_feature_names.json"
        atomic_write_json(
            ordered_feature_path,
            {
                "schema_version": "bigan-btc-15m-moe-ordered-features-v2",
                "feature_count": len(FEATURE_NAMES),
                "feature_names": list(FEATURE_NAMES),
                "feature_names_sha256": canonical_json_sha256(list(FEATURE_NAMES)),
            },
        )
        fallback = xgb.train(
            params=parameters,
            dtrain=_matrix(rows, FEATURE_NAMES, label_field="settlement_payout"),
            num_boost_round=selected_num_boost_round,
            verbose_eval=False,
        )
        fallback_path = staging / "moe_global_fallback.json"
        fallback.save_model(fallback_path)

        expert_boosters: dict[str, xgb.Booster] = {}
        expert_specs: dict[str, dict[str, Any]] = {}
        for route in EXPERT_IDS:
            support = len(route_market_ids[route])
            available = support >= 20
            expert_path = staging / f"moe_expert_{route}.json"
            if available:
                booster = xgb.train(
                    params=parameters,
                    dtrain=_matrix(
                        rows_by_route[route],
                        FEATURE_NAMES,
                        label_field="settlement_payout",
                    ),
                    num_boost_round=selected_num_boost_round,
                    verbose_eval=False,
                )
                booster.save_model(expert_path)
                expert_boosters[route] = booster
                model_format = "xgboost_json"
                rounds = selected_num_boost_round
            else:
                atomic_write_json(
                    expert_path,
                    {
                        "schema_version": (
                            "bigan-btc-15m-moe-unavailable-expert-v2"
                        ),
                        "lineage_id": V2_LINEAGE_ID,
                        "expert_id": route,
                        "available": False,
                        "training_market_count": support,
                        "minimum_training_market_count": 20,
                        "frozen_behavior": "full_113_matched_global_baseline",
                    },
                )
                model_format = "support_below_minimum_stub_json"
                rounds = 0
            expert_specs[route] = {
                "route": route,
                "available": available,
                "model_format": model_format,
                "training_market_count": support,
                "training_market_ids": route_market_ids[route],
                "training_market_ids_sha256": canonical_json_sha256(
                    route_market_ids[route]
                ),
                "training_side_row_count": len(rows_by_route[route]),
                "num_boost_round": rounds,
            }

        population_path = staging / "training_population_manifest.json"
        atomic_write_json(
            population_path,
            {
                "schema_version": "bigan-btc-15m-moe-training-population-v2",
                "lineage_id": V2_LINEAGE_ID,
                "development_index": _descriptor(index_path),
                "development_market_count": 113,
                "ordered_market_ids": [
                    market_id for _, market_id in ordered_markets
                ],
                "ordered_market_ids_sha256": canonical_json_sha256(
                    [market_id for _, market_id in ordered_markets]
                ),
                "round_selection_train_market_count": 93,
                "round_selection_train_market_ids": train_ids,
                "round_selection_validation_market_count": 20,
                "round_selection_validation_market_ids": validation_ids,
                "round_selection_split_used_only_for_boosting_round_selection": True,
                "round_selection_fitted_model_discarded": True,
                "final_global_baseline_training_market_count": 113,
                "final_global_baseline_uses_validation_labels": True,
                "candidate_and_baseline_have_equal_development_information_budget": True,
                "selected_num_boost_round": selected_num_boost_round,
                "experts": expert_specs,
                "input_corpus_manifest_count": len(input_corpora),
                "input_corpus_manifest_set_sha256": canonical_json_sha256(
                    input_corpora
                ),
                "development_only_forever": True,
                "promotion_evidence_eligible": False,
                "safety": dict(SAFETY),
            },
        )
        fixtures_path = staging / "synthetic_prediction_fixture.json"
        atomic_write_json(
            fixtures_path,
            _build_v2_runtime_fixtures(
                fallback=fallback,
                experts=expert_boosters,
                route_support=expected_support,
            ),
        )
        primary_filenames = (
            "moe_candidate_contract.json",
            "moe_router_contract.json",
            "moe_feature_contract.json",
            "moe_cost_and_action_contract.json",
            "ordered_feature_names.json",
            "training_population_manifest.json",
            "moe_global_fallback.json",
            "moe_expert_high_vol.json",
            "moe_expert_bullish.json",
            "moe_expert_bearish.json",
            "moe_expert_low_vol.json",
            "synthetic_prediction_fixture.json",
        )
        primary_hashes = {
            filename: sha256_file(staging / filename)
            for filename in primary_filenames
        }
        bundle_hash = canonical_json_sha256(primary_hashes)
        if bundle_hash == V1_BUNDLE_HASH:
            raise ValueError("v2 bundle hash must differ from v1")
        bundle_dir = artifact_root / bundle_hash
        if bundle_dir.exists():
            raise FileExistsError(f"v2 bundle already exists: {bundle_dir}")
        os.replace(staging, bundle_dir)
        staging_moved = True

        manifest = {
            "schema_version": "bigan-btc-15m-moe-model-manifest-v2",
            "lineage_id": V2_LINEAGE_ID,
            "candidate_id": CANDIDATE_ID,
            "architecture_type": ARCHITECTURE_TYPE,
            "created_at": created_at,
            "implementation_base_commit": BASE_COMMIT,
            "bundle_hash_method": (
                "canonical_json_sha256_of_primary_filename_to_full_content_sha256"
            ),
            "bundle_hash": bundle_hash,
            "bundle_repo_path": bundle_dir.relative_to(repo_root).as_posix(),
            "primary_artifact_hashes": primary_hashes,
            "development_market_count": 113,
            "round_selection_train_market_count": 93,
            "round_selection_validation_market_count": 20,
            "round_selection_model_discarded": True,
            "final_global_baseline_training_market_count": 113,
            "final_global_baseline_uses_validation_labels": True,
            "selected_num_boost_round": selected_num_boost_round,
            "round_selection_best_score": selection_best_score,
            "candidate_and_baseline_information_budget_matched": True,
            "global_fallback": {
                **_descriptor(bundle_dir / "moe_global_fallback.json"),
                "model_format": "xgboost_json",
                "training_market_count": 113,
                "num_boost_round": selected_num_boost_round,
            },
            "matched_global_baseline": {
                "definition": "byte_identical_global_fallback_model",
                **_descriptor(bundle_dir / "moe_global_fallback.json"),
                "training_market_count": 113,
                "num_boost_round": selected_num_boost_round,
            },
            "experts": {
                route: {
                    **expert_specs[route],
                    **_descriptor(bundle_dir / f"moe_expert_{route}.json"),
                }
                for route in EXPERT_IDS
            },
            "contracts": {
                name: _descriptor(bundle_dir / filename)
                for name, filename in (
                    ("candidate", "moe_candidate_contract.json"),
                    ("router", "moe_router_contract.json"),
                    ("features", "moe_feature_contract.json"),
                    ("cost_and_action", "moe_cost_and_action_contract.json"),
                )
            },
            "ordered_features": _descriptor(
                bundle_dir / "ordered_feature_names.json"
            ),
            "training_population": _descriptor(
                bundle_dir / "training_population_manifest.json"
            ),
            "synthetic_prediction_fixture": _descriptor(
                bundle_dir / "synthetic_prediction_fixture.json"
            ),
            "model_training_completed": True,
            "static_model_frozen": True,
            "fresh_collection_authorized": False,
            "fresh_collection_started": False,
            "fresh_outcomes_opened": False,
            "promotion_evidence_eligible": False,
            "safety": dict(SAFETY),
        }
        manifest_path = bundle_dir / "moe_model_manifest.json"
        atomic_write_json(manifest_path, manifest)
        graph_artifacts = {
            filename: _descriptor(bundle_dir / filename)
            for filename in primary_filenames
        }
        graph_artifacts["moe_model_manifest.json"] = _descriptor(manifest_path)
        graph = {
            "schema_version": "bigan-btc-15m-moe-artifact-graph-v2",
            "lineage_id": V2_LINEAGE_ID,
            "candidate_id": CANDIDATE_ID,
            "architecture_type": ARCHITECTURE_TYPE,
            "bundle_hash": bundle_hash,
            "bundle_repo_path": bundle_dir.relative_to(repo_root).as_posix(),
            "artifact_count": len(graph_artifacts),
            "artifacts": graph_artifacts,
            "graph_content_sha256": canonical_json_sha256(graph_artifacts),
            "bundle_hash_excludes_manifest_and_self_referential_graph": True,
            "all_paths_repository_relative": True,
            "machine_local_absolute_paths_allowed": False,
            "fresh_collection_authorized": False,
            "fresh_collection_started": False,
            "fresh_outcomes_opened": False,
            "promotion_evidence_eligible": False,
            "safety": dict(SAFETY),
        }
        graph_path = bundle_dir / "moe_artifact_graph.json"
        atomic_write_json(graph_path, graph)
        for source, output_name in (
            (manifest_path, "moe_model_manifest.json"),
            (graph_path, "moe_artifact_graph.json"),
        ):
            destination = v2_dir / output_name
            shutil.copyfile(source, destination)
            _write_sha_sidecar(destination)
        return {
            "bundle_hash": bundle_hash,
            "bundle_dir": bundle_dir,
            "manifest": manifest,
            "manifest_path": v2_dir / "moe_model_manifest.json",
            "graph": graph,
            "graph_path": v2_dir / "moe_artifact_graph.json",
            "selected_num_boost_round": selected_num_boost_round,
            "fallback_sha256": graph_artifacts["moe_global_fallback.json"]["sha256"],
            "expert_hashes": {
                route: graph_artifacts[f"moe_expert_{route}.json"]["sha256"]
                for route in EXPERT_IDS
            },
        }
    finally:
        if not staging_moved and staging.exists():
            shutil.rmtree(staging)


def _write_matched_baseline_contract(
    *,
    repo_root: Path,
    v2_dir: Path,
    bundle: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    manifest = bundle["manifest"]
    output = v2_dir / "moe_matched_global_baseline_contract.json"
    payload = {
        "schema_version": "bigan-btc-15m-moe-matched-global-baseline-v2",
        "lineage_id": V2_LINEAGE_ID,
        "contract_id": "matched_global_baseline",
        "created_at": created_at,
        "artifact": {
            **manifest["matched_global_baseline"],
            "byte_identical_to_candidate_global_fallback": True,
            "candidate_global_fallback_sha256": manifest["global_fallback"][
                "sha256"
            ],
        },
        "information_budget": {
            "development_market_count": 113,
            "round_selection_train_market_count": 93,
            "round_selection_validation_market_count": 20,
            "round_selection_split_used_only_for_num_boost_round": True,
            "round_selection_fitted_model_discarded": True,
            "final_model_retrained_after_round_selection": True,
            "final_training_market_count": 113,
            "final_model_uses_validation_labels": True,
            "selected_num_boost_round": bundle["selected_num_boost_round"],
            "candidate_and_baseline_have_equal_development_information_budget": True,
        },
        "features": {
            "feature_contract": _descriptor(
                v2_dir / "moe_feature_contract.json"
            ),
            "ordered_feature_names": manifest["ordered_features"],
            "same_feature_order_as_candidate": True,
            "same_side_symmetric_transformation_as_candidate": True,
            "same_pair_normalization_as_candidate": True,
            "missing_value": "nan",
            "missing_as_numeric_zero_allowed": False,
        },
        "cost_and_behavior": {
            "contract": _descriptor(v2_dir / "moe_cost_and_action_contract.json"),
            "same_cost_model_as_candidate": True,
            "same_fixed_action_threshold_as_candidate": True,
            "sizing": "unit",
            "execution_policy": "HOLD_TO_SETTLEMENT",
            "true_paired_executable_asks_required": True,
            "complement_quote_proxy_allowed": False,
            "NO_TRADE_unit_net_pnl": 0.0,
            "NO_TRADE_retained_in_bootstrap": True,
        },
        "bootstrap_participation": {
            "included_for_every_frozen_confirmatory_market": True,
            "paired_by_market_id_with_candidate": True,
            "same_resample_indices_as_candidate": True,
            "missing_market_drop_allowed": False,
        },
        "repository_root_used_for_descriptor_resolution": repo_root.name,
        "state": {
            "contract_frozen": True,
            "fresh_collection_authorized": False,
            "fresh_collection_started": False,
            "fresh_outcomes_opened": False,
        },
        "safety": dict(SAFETY),
    }
    _write_frozen_json(output, payload)
    return {"path": output, "sha256": sha256_file(output), "payload": payload}


def _build_v2_runtime_fixtures(
    *,
    fallback: xgb.Booster,
    experts: Mapping[str, xgb.Booster],
    route_support: Mapping[str, int],
) -> dict[str, Any]:
    base_router = {
        "decision_ts": 10_000,
        "available_at_ts": 10_000,
        "max_input_ts": 9_999,
    }
    definitions: list[dict[str, Any]] = [
        {
            "fixture_id": "high_vol_native_expert",
            "router_inputs": {
                **base_router,
                "volatility_bucket": "high",
                "btc_return_regime": "bullish",
            },
            "side_feature_overrides": [
                {"btc_volatility_15m": 0.001},
                {"btc_volatility_15m": 0.001},
            ],
            "execution_costs": [0.45, 0.45],
            "required_path": "high_vol_native_expert",
        },
        {
            "fixture_id": "bullish_native_expert",
            "router_inputs": {
                **base_router,
                "volatility_bucket": "medium",
                "btc_return_regime": "bullish",
            },
            "side_feature_overrides": [
                {"signed_btc_return_15m": 0.005},
                {"signed_btc_return_15m": -0.005},
            ],
            "execution_costs": [0.45, 0.45],
            "required_path": "bullish_native_expert",
        },
        {
            "fixture_id": "bearish_native_expert",
            "router_inputs": {
                **base_router,
                "volatility_bucket": "medium",
                "btc_return_regime": "bearish",
            },
            "side_feature_overrides": [
                {"signed_btc_return_15m": -0.005},
                {"signed_btc_return_15m": 0.005},
            ],
            "execution_costs": [0.45, 0.45],
            "required_path": "bearish_native_expert",
        },
        {
            "fixture_id": "low_vol_global_fallback",
            "router_inputs": {
                **base_router,
                "volatility_bucket": "low",
                "btc_return_regime": "sideways",
            },
            "side_feature_overrides": [{}, {}],
            "execution_costs": [0.45, 0.45],
            "required_path": "low_vol_global_fallback",
        },
        {
            "fixture_id": "selected_side_up",
            "router_inputs": {
                **base_router,
                "volatility_bucket": "high",
                "btc_return_regime": "sideways",
            },
            "side_feature_overrides": [{}, {}],
            "execution_costs": [0.0, 1.0],
            "required_path": "UP_selection",
        },
        {
            "fixture_id": "selected_side_down",
            "router_inputs": {
                **base_router,
                "volatility_bucket": "high",
                "btc_return_regime": "sideways",
            },
            "side_feature_overrides": [{}, {}],
            "execution_costs": [1.0, 0.0],
            "required_path": "DOWN_selection",
        },
        {
            "fixture_id": "no_trade_both_scores_nonpositive",
            "router_inputs": {
                **base_router,
                "volatility_bucket": "medium",
                "btc_return_regime": "bullish",
            },
            "side_feature_overrides": [{}, {}],
            "execution_costs": [1.0, 1.0],
            "required_path": "NO_TRADE",
        },
        {
            "fixture_id": "asymmetric_paired_feature_rows",
            "router_inputs": {
                **base_router,
                "volatility_bucket": "medium",
                "btc_return_regime": "bearish",
            },
            "side_feature_overrides": [
                {
                    "signed_btc_return_15m": 0.02,
                    "selected_minus_opposite_mid": 0.25,
                },
                {
                    "signed_btc_return_15m": -0.02,
                    "selected_minus_opposite_mid": -0.25,
                },
            ],
            "execution_costs": [0.45, 0.45],
            "required_path": "asymmetric_pair",
        },
    ]
    fixtures = [
        _freeze_prediction_fixture(
            definition=definition,
            fallback=fallback,
            experts=experts,
            route_support=route_support,
        )
        for definition in definitions
    ]
    threshold_seed = _freeze_prediction_fixture(
        definition={
            "fixture_id": "threshold_seed",
            "router_inputs": {
                **base_router,
                "volatility_bucket": "high",
                "btc_return_regime": "sideways",
            },
            "side_feature_overrides": [{}, {}],
            "execution_costs": [0.0, 0.0],
            "required_path": "threshold_seed",
        },
        fallback=fallback,
        experts=experts,
        route_support=route_support,
    )
    threshold_costs = list(threshold_seed["expected_normalized_probabilities"])
    fixtures.append(
        _freeze_prediction_fixture(
            definition={
                "fixture_id": "exact_threshold_boundary_score_zero",
                "router_inputs": threshold_seed["router_inputs"],
                "side_feature_overrides": [{}, {}],
                "execution_costs": threshold_costs,
                "required_path": "threshold_boundary",
            },
            fallback=fallback,
            experts=experts,
            route_support=route_support,
        )
    )
    fixtures.append(
        _freeze_prediction_fixture(
            definition={
                "fixture_id": "up_down_tie_frozen_up_tiebreak",
                "router_inputs": threshold_seed["router_inputs"],
                "side_feature_overrides": [{}, {}],
                "execution_costs": [
                    threshold_costs[0] - 0.1,
                    threshold_costs[1] - 0.1,
                ],
                "required_path": "UP_tie_break",
            },
            fallback=fallback,
            experts=experts,
            route_support=route_support,
        )
    )
    router_fixtures = [
        {
            "fixture_id": "high_vol_precedence_over_bullish",
            "inputs": {
                **base_router,
                "volatility_bucket": "high",
                "btc_return_regime": "bullish",
            },
            "expected_route": "high_vol",
        },
        {
            "fixture_id": "bullish_router_path",
            "inputs": {
                **base_router,
                "volatility_bucket": "medium",
                "btc_return_regime": "bullish",
            },
            "expected_route": "bullish",
        },
        {
            "fixture_id": "bearish_router_path",
            "inputs": {
                **base_router,
                "volatility_bucket": "medium",
                "btc_return_regime": "bearish",
            },
            "expected_route": "bearish",
        },
        {
            "fixture_id": "sideways_low_vol_router_path",
            "inputs": {
                **base_router,
                "volatility_bucket": "low",
                "btc_return_regime": "sideways",
            },
            "expected_route": "low_vol",
        },
    ]
    rejection_fixtures = [
        {
            "fixture_id": "missing_required_router_field",
            "operation": "route",
            "inputs": {
                key: value
                for key, value in router_fixtures[1]["inputs"].items()
                if key != "max_input_ts"
            },
            "expected_error": "MoE router inputs missing",
        },
        {
            "fixture_id": "forbidden_future_outcome_router_field",
            "operation": "route",
            "inputs": {
                **router_fixtures[1]["inputs"],
                "future_return": 1.0,
                "target": 1.0,
            },
            "expected_error": "outcome or future fields are forbidden",
        },
        {
            "fixture_id": "available_after_decision",
            "operation": "route",
            "inputs": {
                **router_fixtures[1]["inputs"],
                "available_at_ts": 10_001,
            },
            "expected_error": "causality violation",
        },
        {
            "fixture_id": "max_input_after_decision",
            "operation": "route",
            "inputs": {
                **router_fixtures[1]["inputs"],
                "max_input_ts": 10_001,
            },
            "expected_error": "causality violation",
        },
        {
            "fixture_id": "unknown_route",
            "operation": "expert_resolution",
            "route": "unknown",
            "expert_training_market_count": 20,
            "expected_error": "unknown frozen MoE route",
        },
    ]
    return {
        "schema_version": "bigan-btc-15m-moe-runtime-fixtures-v2",
        "lineage_id": V2_LINEAGE_ID,
        "feature_names_sha256": canonical_json_sha256(list(FEATURE_NAMES)),
        "prediction_fixtures": fixtures,
        "router_fixtures": router_fixtures,
        "rejection_fixtures": rejection_fixtures,
    }


def _freeze_prediction_fixture(
    *,
    definition: Mapping[str, Any],
    fallback: xgb.Booster,
    experts: Mapping[str, xgb.Booster],
    route_support: Mapping[str, int],
) -> dict[str, Any]:
    result = _execute_prediction_fixture(
        fixture=definition,
        fallback=fallback,
        experts=experts,
        route_support=route_support,
        feature_names=FEATURE_NAMES,
    )
    return {
        **definition,
        "default_feature_value": 0.0,
        "expected_route": result["route"],
        "expected_actual_model_used": result["actual_model_used"],
        "expected_raw_probabilities": result["raw_probabilities"],
        "expected_normalized_probabilities": result["normalized_probabilities"],
        "expected_scores": result["scores"],
        "expected_selected_side": result["selected_side"],
        "expected_accepted": result["accepted"],
    }


def _execute_prediction_fixture(
    *,
    fixture: Mapping[str, Any],
    fallback: xgb.Booster,
    experts: Mapping[str, xgb.Booster],
    route_support: Mapping[str, int],
    feature_names: Sequence[str],
) -> dict[str, Any]:
    route = deterministic_moe_route(fixture["router_inputs"])
    actual_model = frozen_expert_or_fallback(
        route=route,
        expert_training_market_count=int(route_support[route]),
    )
    booster = (
        fallback if actual_model == "global_baseline_fallback" else experts[route]
    )
    matrix_rows = [
        _fixture_values(overrides, feature_names=feature_names)
        for overrides in fixture["side_feature_overrides"]
    ]
    raw = booster.predict(
        xgb.DMatrix(
            np.asarray(matrix_rows, dtype=np.float64),
            feature_names=list(feature_names),
            missing=np.nan,
        )
    )
    raw_sum = float(np.sum(raw))
    if not math.isfinite(raw_sum) or raw_sum <= 0.0:
        raise ValueError("invalid synthetic raw probability sum")
    normalized = [float(value) / raw_sum for value in raw]
    scores = [
        normalized[index] - float(fixture["execution_costs"][index])
        for index in range(2)
    ]
    selected_index = max(range(2), key=lambda index: (scores[index], -index))
    accepted = scores[selected_index] > 0.0
    return {
        "fixture_id": fixture["fixture_id"],
        "required_path": fixture["required_path"],
        "route": route,
        "actual_model_used": actual_model,
        "raw_probabilities": [float(value) for value in raw],
        "normalized_probabilities": normalized,
        "scores": scores,
        "selected_side": ("UP", "DOWN")[selected_index] if accepted else None,
        "tie_break_candidate_side": ("UP", "DOWN")[selected_index],
        "accepted": accepted,
    }


def _fixture_values(
    overrides: Mapping[str, float],
    *,
    feature_names: Sequence[str],
) -> list[float]:
    values = dict.fromkeys(feature_names, 0.0)
    for name, value in overrides.items():
        if name not in values:
            raise ValueError(f"unknown runtime fixture feature: {name}")
        values[name] = float(value)
    return [values[name] for name in feature_names]


def load_and_verify_v2_artifact(
    *,
    graph_path: Path | str,
    expected_graph_sha256: str,
    repository_root: Path | str | None = None,
) -> dict[str, Any]:
    """Fail closed while loading and executing every frozen v2 runtime path."""

    repo_root = Path(repository_root or REPO_ROOT).resolve()
    graph_file = Path(graph_path).resolve()
    if sha256_file(graph_file) != expected_graph_sha256:
        raise ValueError("v2 artifact graph SHA-256 mismatch")
    graph = _load_json(graph_file)
    if not (
        graph.get("schema_version") == "bigan-btc-15m-moe-artifact-graph-v2"
        and graph.get("lineage_id") == V2_LINEAGE_ID
        and graph.get("candidate_id") == CANDIDATE_ID
        and graph.get("machine_local_absolute_paths_allowed") is False
        and graph.get("fresh_collection_authorized") is False
        and graph.get("fresh_outcomes_opened") is False
        and dict(graph.get("safety") or {}) == SAFETY
    ):
        raise ValueError("v2 artifact graph governance mismatch")
    bundle_path = Path(str(graph["bundle_repo_path"]))
    if bundle_path.is_absolute() or bundle_path.name != graph["bundle_hash"]:
        raise ValueError("v2 bundle path is not repository content addressed")
    resolved: dict[str, Path] = {}
    for filename, descriptor in graph["artifacts"].items():
        if Path(str(descriptor["path"])) != bundle_path / filename:
            raise ValueError(f"v2 artifact is not bundle local: {filename}")
        path = (repo_root / descriptor["path"]).resolve()
        if not path.is_relative_to(repo_root):
            raise ValueError(f"v2 artifact escaped repository: {filename}")
        if not path.is_file() or sha256_file(path) != descriptor["sha256"]:
            raise ValueError(f"v2 artifact SHA-256 mismatch: {filename}")
        resolved[filename] = path
    if canonical_json_sha256(graph["artifacts"]) != graph["graph_content_sha256"]:
        raise ValueError("v2 artifact graph content hash mismatch")
    primary_hashes = {
        filename: descriptor["sha256"]
        for filename, descriptor in graph["artifacts"].items()
        if filename != "moe_model_manifest.json"
    }
    if canonical_json_sha256(primary_hashes) != graph["bundle_hash"]:
        raise ValueError("v2 bundle hash mismatch")

    manifest = _load_json(resolved["moe_model_manifest.json"])
    if not (
        manifest["lineage_id"] == V2_LINEAGE_ID
        and manifest["bundle_hash"] == graph["bundle_hash"]
        and manifest["candidate_and_baseline_information_budget_matched"] is True
        and manifest["final_global_baseline_training_market_count"] == 113
        and manifest["final_global_baseline_uses_validation_labels"] is True
        and manifest["fresh_collection_authorized"] is False
        and manifest["fresh_outcomes_opened"] is False
        and dict(manifest["safety"]) == SAFETY
    ):
        raise ValueError("v2 model manifest governance mismatch")
    if (
        manifest["global_fallback"]["sha256"]
        != manifest["matched_global_baseline"]["sha256"]
    ):
        raise ValueError("v2 matched baseline is not candidate fallback identical")
    features = _load_json(resolved["ordered_feature_names.json"])
    fallback = xgb.Booster()
    fallback.load_model(resolved["moe_global_fallback.json"])
    selected_rounds = int(manifest["selected_num_boost_round"])
    if int(fallback.num_boosted_rounds()) != selected_rounds:
        raise ValueError("v2 fallback boosting-round mismatch")
    experts: dict[str, xgb.Booster] = {}
    loaded_experts: dict[str, str] = {}
    for route in EXPERT_IDS:
        spec = manifest["experts"][route]
        path = resolved[f"moe_expert_{route}.json"]
        if spec["available"]:
            booster = xgb.Booster()
            booster.load_model(path)
            if int(booster.num_boosted_rounds()) != selected_rounds:
                raise ValueError(f"v2 expert boosting-round mismatch: {route}")
            experts[route] = booster
            loaded_experts[route] = "xgboost_json"
        else:
            stub = _load_json(path)
            if not (
                stub["available"] is False
                and int(stub["training_market_count"]) < 20
                and stub["frozen_behavior"]
                == "full_113_matched_global_baseline"
            ):
                raise ValueError(f"v2 unavailable expert stub invalid: {route}")
            loaded_experts[route] = "support_below_minimum_stub_json"

    fixtures = _load_json(resolved["synthetic_prediction_fixture.json"])
    route_support = {
        route: int(manifest["experts"][route]["training_market_count"])
        for route in EXPERT_IDS
    }
    prediction_results = []
    for fixture in fixtures["prediction_fixtures"]:
        result = _execute_prediction_fixture(
            fixture=fixture,
            fallback=fallback,
            experts=experts,
            route_support=route_support,
            feature_names=features["feature_names"],
        )
        expected = {
            "route": fixture["expected_route"],
            "actual_model_used": fixture["expected_actual_model_used"],
            "raw_probabilities": fixture["expected_raw_probabilities"],
            "normalized_probabilities": fixture[
                "expected_normalized_probabilities"
            ],
            "scores": fixture["expected_scores"],
            "selected_side": fixture["expected_selected_side"],
            "accepted": fixture["expected_accepted"],
        }
        for field, value in expected.items():
            if result[field] != value:
                raise ValueError(
                    f"v2 deterministic prediction fixture drifted: "
                    f"{fixture['fixture_id']}:{field}"
                )
        prediction_results.append(result)
    router_results = []
    for fixture in fixtures["router_fixtures"]:
        observed_route = deterministic_moe_route(fixture["inputs"])
        if observed_route != fixture["expected_route"]:
            raise ValueError(f"v2 router fixture drifted: {fixture['fixture_id']}")
        router_results.append(
            {
                "fixture_id": fixture["fixture_id"],
                "observed_route": observed_route,
                "passed": True,
            }
        )
    rejection_results = []
    for fixture in fixtures["rejection_fixtures"]:
        try:
            if fixture["operation"] == "route":
                deterministic_moe_route(fixture["inputs"])
            else:
                frozen_expert_or_fallback(
                    route=fixture["route"],
                    expert_training_market_count=int(
                        fixture["expert_training_market_count"]
                    ),
                )
        except ValueError as error:
            if fixture["expected_error"] not in str(error):
                raise ValueError(
                    f"v2 rejection fixture raised wrong error: "
                    f"{fixture['fixture_id']}"
                ) from error
            rejection_results.append(
                {
                    "fixture_id": fixture["fixture_id"],
                    "rejected": True,
                    "error": str(error),
                }
            )
        else:
            raise ValueError(
                f"v2 rejection fixture did not fail closed: {fixture['fixture_id']}"
            )
    return {
        "lineage_id": V2_LINEAGE_ID,
        "bundle_hash": graph["bundle_hash"],
        "artifact_count": len(resolved),
        "verified_child_sha256_count": len(resolved),
        "selected_num_boost_round": selected_rounds,
        "loaded_experts": loaded_experts,
        "fallback_loaded": True,
        "prediction_results": prediction_results,
        "router_results": router_results,
        "rejection_results": rejection_results,
        "fresh_collection_authorized": False,
        "fresh_outcomes_opened": False,
        "safety": dict(SAFETY),
    }


def validate_v2_artifact_in_fresh_environment(
    *,
    graph_path: Path | str,
    expected_graph_sha256: str,
    repository_root: Path | str | None = None,
) -> dict[str, Any]:
    """Copy only the frozen graph and children, then exercise all runtime paths."""

    repo_root = Path(repository_root or REPO_ROOT).resolve()
    source_graph = Path(graph_path).resolve()
    if not source_graph.is_relative_to(repo_root):
        raise ValueError("v2 source graph escaped repository")
    if sha256_file(source_graph) != expected_graph_sha256:
        raise ValueError("v2 artifact graph SHA-256 mismatch")
    graph = _load_json(source_graph)
    with tempfile.TemporaryDirectory(prefix="bigan-moe-v2-runtime-") as temporary:
        fresh_root = Path(temporary) / "fresh-repository"
        fresh_graph = fresh_root / source_graph.relative_to(repo_root)
        fresh_graph.parent.mkdir(parents=True)
        shutil.copyfile(source_graph, fresh_graph)
        for filename, descriptor in graph["artifacts"].items():
            source = (repo_root / descriptor["path"]).resolve()
            if (
                not source.is_relative_to(repo_root)
                or not source.is_file()
                or sha256_file(source) != descriptor["sha256"]
            ):
                raise ValueError(f"v2 source artifact mismatch: {filename}")
            destination = fresh_root / descriptor["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        first = load_and_verify_v2_artifact(
            graph_path=fresh_graph,
            expected_graph_sha256=expected_graph_sha256,
            repository_root=fresh_root,
        )
        second = load_and_verify_v2_artifact(
            graph_path=fresh_graph,
            expected_graph_sha256=expected_graph_sha256,
            repository_root=fresh_root,
        )
        if first != second:
            raise ValueError("v2 runtime prediction output is nondeterministic")
        tampered_name = "moe_global_fallback.json"
        tampered_path = fresh_root / graph["artifacts"][tampered_name]["path"]
        with tampered_path.open("ab") as handle:
            handle.write(b"\n")
        try:
            load_and_verify_v2_artifact(
                graph_path=fresh_graph,
                expected_graph_sha256=expected_graph_sha256,
                repository_root=fresh_root,
            )
        except ValueError as error:
            if "artifact SHA-256 mismatch" not in str(error):
                raise ValueError(
                    "v2 artifact mismatch rejection raised wrong error"
                ) from error
            artifact_hash_mismatch_rejected = True
        else:
            raise ValueError("v2 artifact hash mismatch did not fail closed")

    prediction_by_path = {
        result["required_path"]: result for result in first["prediction_results"]
    }
    executed_models = {
        result["actual_model_used"] for result in first["prediction_results"]
    }
    available_models = {
        f"moe_expert_{route}"
        for route, model_format in first["loaded_experts"].items()
        if model_format == "xgboost_json"
    }
    all_available_experts_executed = available_models <= executed_models
    fallback_executed = "global_baseline_fallback" in executed_models
    up_executed = prediction_by_path["UP_selection"]["selected_side"] == "UP"
    down_executed = prediction_by_path["DOWN_selection"]["selected_side"] == "DOWN"
    no_trade_executed = (
        prediction_by_path["NO_TRADE"]["accepted"] is False
        and prediction_by_path["NO_TRADE"]["selected_side"] is None
    )
    threshold_boundary_executed = (
        prediction_by_path["threshold_boundary"]["accepted"] is False
        and max(prediction_by_path["threshold_boundary"]["scores"]) == 0.0
    )
    tie_break_executed = (
        prediction_by_path["UP_tie_break"]["selected_side"] == "UP"
        and len(set(prediction_by_path["UP_tie_break"]["scores"])) == 1
    )
    router_boundary_passed = all(
        result["passed"] for result in first["router_results"]
    )
    causality_ids = {"available_after_decision", "max_input_after_decision"}
    causality_passed = causality_ids <= {
        result["fixture_id"]
        for result in first["rejection_results"]
        if result["rejected"]
    }
    mandatory = all(
        (
            all_available_experts_executed,
            fallback_executed,
            up_executed,
            down_executed,
            no_trade_executed,
            threshold_boundary_executed,
            tie_break_executed,
            router_boundary_passed,
            causality_passed,
            artifact_hash_mismatch_rejected,
        )
    )
    if not mandatory:
        raise ValueError("v2 mandatory runtime validation gate failed")
    return {
        "fresh_environment_resolution": True,
        "bundle_hash_verified": True,
        "child_sha_verification_passed": True,
        "verified_child_sha256_count": first["verified_child_sha256_count"],
        "all_available_experts_executed": all_available_experts_executed,
        "fallback_executed": fallback_executed,
        "UP_selection_executed": up_executed,
        "DOWN_selection_executed": down_executed,
        "NO_TRADE_executed": no_trade_executed,
        "threshold_boundary_executed": threshold_boundary_executed,
        "UP_tie_break_executed": tie_break_executed,
        "router_boundary_tests_passed": router_boundary_passed,
        "causality_failure_tests_passed": causality_passed,
        "artifact_hash_mismatch_rejection_passed": (
            artifact_hash_mismatch_rejected
        ),
        "prediction_results": first["prediction_results"],
        "router_results": first["router_results"],
        "rejection_results": first["rejection_results"],
        "mandatory_gate_passed": mandatory,
        "fresh_collection_authorized": False,
        "fresh_outcomes_opened": False,
        "safety": dict(SAFETY),
    }


def _write_runtime_report(
    *,
    v2_dir: Path,
    bundle: Mapping[str, Any],
    runtime: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    output = v2_dir / "moe_artifact_runtime_validation_report.json"
    payload = {
        "schema_version": "bigan-btc-15m-moe-runtime-validation-report-v2",
        "lineage_id": V2_LINEAGE_ID,
        "created_at": created_at,
        "role": "mandatory_precollection_runtime_gate",
        "inputs": {
            "bundle_hash": bundle["bundle_hash"],
            "artifact_graph": _descriptor(bundle["graph_path"]),
            "model_manifest": _descriptor(bundle["manifest_path"]),
        },
        "checks": dict(runtime),
        "mandatory_gate_passed": runtime["mandatory_gate_passed"],
        "failure_semantics": "any_mismatch_blocks_collection_authorization",
        "state": {
            "fresh_collection_authorized": False,
            "fresh_collection_started": False,
            "fresh_outcomes_opened": False,
        },
        "safety": dict(SAFETY),
    }
    _write_frozen_json(output, payload)
    return {"path": output, "sha256": sha256_file(output), "payload": payload}


def _write_power_protocol_and_analysis(
    *,
    repo_root: Path,
    v2_dir: Path,
    bundle: Mapping[str, Any],
    baseline: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    prediction_path = (
        repo_root
        / "examples/v8/polymarket_training_artifacts/"
        "BTC-15M-regime-adaptive-v1-development-evaluation/"
        "development_oof_predictions.jsonl"
    )
    attribution_path = (
        repo_root
        / "examples/v8/polymarket_configs/"
        f"{V1_LINEAGE_ID}/moe_route_attribution.jsonl"
    )
    protocol_path = v2_dir / "moe_confirmatory_power_protocol.json"
    power_protocol = {
        "schema_version": "bigan-btc-15m-moe-confirmatory-power-protocol-v2",
        "lineage_id": V2_LINEAGE_ID,
        "created_at": created_at,
        "role": "pre_sample_size_selection_power_protocol",
        "candidate_bundle_hash": bundle["bundle_hash"],
        "matched_baseline_contract": _descriptor(baseline["path"]),
        "planning_inputs": {
            "development_oof_predictions": _descriptor(prediction_path),
            "moe_route_attribution": _descriptor(attribution_path),
            "evidence_role": "hypothesis_generation_planning_only",
            "promotion_evidence_eligible": False,
        },
        "estimands": {
            "primary": (
                "mean_market_level_unit_net_pnl_delta_moe_minus_"
                "matched_global_baseline"
            ),
            "secondary_absolute": "mean_market_level_unit_net_pnl_of_moe",
        },
        "method": {
            "name": (
                "normal_approximation_to_one_sided_market_bootstrap_"
                "LCB_crossing_probability"
            ),
            "unit": "unique_market",
            "one_sided_confidence": 0.975,
            "z_value": NormalDist().inv_cdf(0.975),
            "NO_TRADE": 0.0,
            "paired_candidate_and_baseline_rows_required": True,
            "candidate_and_baseline_share_market_population": True,
        },
        "candidate_fixed_window_sizes": list(CANDIDATE_SAMPLE_SIZES),
        "effect_size_scenarios": [
            "observed_development_effect",
            "75pct_of_observed_effect",
            "50pct_of_observed_effect",
            "zero_effect",
        ],
        "variance_multipliers": [1.0, 1.25, 1.5],
        "provider_missingness_scenarios": [
            "development_missingness_mix",
            "reduced_missingness",
            "complete_feature_heavy_future_population",
        ],
        "regime_scenarios": [
            "development_regime_mix",
            "increased_bearish_share",
            "reduced_high_vol_share",
        ],
        "design_selection_rule": {
            "selection": "smallest_tested_fixed_window_satisfying_both",
            "effect_size_scenario": "observed_development_effect",
            "variance_multiplier": 1.25,
            "minimum_primary_delta_LCB_crossing_probability": 0.8,
            "minimum_absolute_moe_LCB_crossing_probability": 0.8,
        },
        "report_only_sensitivities": [
            "75pct_of_observed_effect",
            "50pct_of_observed_effect",
            "variance_multiplier_1_5",
            "provider_missingness_distribution_shifts",
            "regime_shifts",
        ],
        "one_fixed_window_required": True,
        "optional_stopping_allowed": False,
        "interim_outcome_looks_allowed": False,
        "failed_round_replacement_allowed": False,
        "state": {
            "confirmatory_design_ready": False,
            "fresh_collection_authorized": False,
            "fresh_outcomes_opened": False,
        },
        "safety": dict(SAFETY),
    }
    _write_frozen_json(protocol_path, power_protocol)

    planning_rows = _paired_planning_rows(
        prediction_path=prediction_path,
        attribution_path=attribution_path,
    )
    planning_path = v2_dir / "development_paired_planning_rows.jsonl"
    _write_jsonl(planning_path, planning_rows)
    _write_sha_sidecar(planning_path)
    observed = _distribution_metrics(planning_rows, lambda _: 1.0)
    effect_factors = {
        "observed_development_effect": 1.0,
        "75pct_of_observed_effect": 0.75,
        "50pct_of_observed_effect": 0.5,
        "zero_effect": 0.0,
    }
    core_sensitivity = []
    for effect_name, effect_factor in effect_factors.items():
        for variance_multiplier in (1.0, 1.25, 1.5):
            for sample_size in CANDIDATE_SAMPLE_SIZES:
                core_sensitivity.append(
                    _power_row(
                        scenario_family="effect_and_variance",
                        scenario_name=effect_name,
                        sample_size=sample_size,
                        primary_mean=(
                            observed["primary_delta"]["mean"] * effect_factor
                        ),
                        primary_variance=(
                            observed["primary_delta"]["sample_variance"]
                            * variance_multiplier
                        ),
                        absolute_mean=(
                            observed["absolute_moe"]["mean"] * effect_factor
                        ),
                        absolute_variance=(
                            observed["absolute_moe"]["sample_variance"]
                            * variance_multiplier
                        ),
                        variance_multiplier=variance_multiplier,
                    )
                )

    provider_scenarios: dict[str, Callable[[Mapping[str, Any]], float]] = {
        "development_missingness_mix": lambda _: 1.0,
        "reduced_missingness": _target_share_weight(
            planning_rows,
            field="feature_complete",
            target_value=True,
            target_share=0.5,
        ),
        "complete_feature_heavy_future_population": _target_share_weight(
            planning_rows,
            field="feature_complete",
            target_value=True,
            target_share=0.8,
        ),
    }
    regime_scenarios: dict[str, Callable[[Mapping[str, Any]], float]] = {
        "development_regime_mix": lambda _: 1.0,
        "increased_bearish_share": _target_share_weight(
            planning_rows,
            field="btc_return_regime",
            target_value="bearish",
            target_share=0.5,
        ),
        "reduced_high_vol_share": _target_share_weight(
            planning_rows,
            field="requested_route",
            target_value="high_vol",
            target_share=0.15,
        ),
    }
    distribution_sensitivity = []
    for family, scenarios in (
        ("provider_missingness", provider_scenarios),
        ("regime", regime_scenarios),
    ):
        for scenario_name, weight_fn in scenarios.items():
            metrics = _distribution_metrics(planning_rows, weight_fn)
            for sample_size in CANDIDATE_SAMPLE_SIZES:
                distribution_sensitivity.append(
                    {
                        **_power_row(
                            scenario_family=family,
                            scenario_name=scenario_name,
                            sample_size=sample_size,
                            primary_mean=metrics["primary_delta"]["mean"],
                            primary_variance=metrics["primary_delta"][
                                "sample_variance"
                            ],
                            absolute_mean=metrics["absolute_moe"]["mean"],
                            absolute_variance=metrics["absolute_moe"][
                                "sample_variance"
                            ],
                            variance_multiplier=1.0,
                        ),
                        "weighted_distribution": metrics,
                    }
                )
    design_rows = [
        row
        for row in core_sensitivity
        if row["scenario_name"] == "observed_development_effect"
        and row["variance_multiplier"] == 1.25
    ]
    selected = next(
        (
            row["sample_size"]
            for row in design_rows
            if row["estimated_probability_primary_delta_LCB_gt_0"] >= 0.8
            and row["estimated_probability_absolute_moe_LCB_gt_0"] >= 0.8
        ),
        None,
    )
    ready = selected is not None
    analysis_path = v2_dir / "moe_confirmatory_power_analysis.json"
    analysis = {
        "schema_version": "bigan-btc-15m-moe-confirmatory-power-analysis-v2",
        "lineage_id": V2_LINEAGE_ID,
        "created_at": created_at,
        "role": "planning_only_precollection_sample_size_selection",
        "power_protocol": _descriptor(protocol_path),
        "paired_planning_rows": _descriptor(planning_path),
        "planning_market_count": len(planning_rows),
        "observed_development_distribution": observed,
        "candidate_fixed_window_sizes": list(CANDIDATE_SAMPLE_SIZES),
        "core_effect_and_variance_sensitivity": core_sensitivity,
        "provider_and_regime_sensitivity": distribution_sensitivity,
        "design_selection_rule": power_protocol["design_selection_rule"],
        "design_selection_rows": design_rows,
        "selected_confirmatory_market_count": selected,
        "confirmatory_design_ready": ready,
        "selected_design_is_one_fixed_window": ready,
        "optional_stopping_allowed": False,
        "interim_outcome_looks_allowed": False,
        "failed_round_replacement_allowed": False,
        "development_result_is_validation_or_promotion_evidence": False,
        "state": {
            "fresh_collection_authorized": False,
            "fresh_collection_started": False,
            "fresh_outcomes_opened": False,
        },
        "safety": dict(SAFETY),
    }
    _write_frozen_json(analysis_path, analysis)
    return {
        "protocol": power_protocol,
        "protocol_path": protocol_path,
        "protocol_sha256": sha256_file(protocol_path),
        "analysis": analysis,
        "analysis_path": analysis_path,
        "analysis_sha256": sha256_file(analysis_path),
        "planning_rows_path": planning_path,
    }


def _paired_planning_rows(
    *,
    prediction_path: Path,
    attribution_path: Path,
) -> list[dict[str, Any]]:
    predictions = load_jsonl(prediction_path)
    attribution = {
        str(row["market_id"]): row for row in load_jsonl(attribution_path)
    }
    candidate_rows = [
        row for row in predictions if row["candidate_id"] == CANDIDATE_ID
    ]
    baseline_rows = [
        row for row in predictions if row["candidate_id"] == "global_baseline"
    ]
    ordered_markets = sorted(
        {
            (int(row["market_start_ts"]), str(row["market_id"]))
            for row in candidate_rows
        }
    )
    candidate_selected = {
        str(row["market_id"]): float(row["target"])
        for row in _selected_rows(candidate_rows)
    }
    baseline_selected = {
        str(row["market_id"]): float(row["target"])
        for row in _selected_rows(baseline_rows)
    }
    result = []
    for market_start_ts, market_id in ordered_markets:
        route = attribution[market_id]
        candidate_pnl = candidate_selected.get(market_id, 0.0)
        baseline_pnl = baseline_selected.get(market_id, 0.0)
        result.append(
            {
                "market_id": market_id,
                "market_start_ts": market_start_ts,
                "moe_unit_net_pnl": candidate_pnl,
                "matched_global_baseline_proxy_unit_net_pnl": baseline_pnl,
                "paired_delta_unit_net_pnl": candidate_pnl - baseline_pnl,
                "requested_route": route["requested_route"],
                "btc_return_regime": route["regime_bucket"][
                    "btc_return_regime"
                ],
                "volatility_bucket": route["regime_bucket"][
                    "volatility_bucket"
                ],
                "feature_complete": route["provider_missingness"][
                    "feature_complete"
                ],
                "provider_health_score": route["provider_missingness"][
                    "provider_health_score"
                ],
                "development_only_forever": True,
                "promotion_evidence_eligible": False,
            }
        )
    if len(result) != 73:
        raise ValueError("power planning requires exactly 73 paired OOF markets")
    return result


def _distribution_metrics(
    rows: Sequence[Mapping[str, Any]],
    weight_fn: Callable[[Mapping[str, Any]], float],
) -> dict[str, Any]:
    weights = [float(weight_fn(row)) for row in rows]
    if any(weight < 0.0 or not math.isfinite(weight) for weight in weights):
        raise ValueError("invalid sensitivity weight")
    if sum(weights) <= 0.0:
        raise ValueError("sensitivity weights sum to zero")
    return {
        "weight_sum": sum(weights),
        "effective_sample_size": sum(weights) ** 2
        / sum(weight * weight for weight in weights),
        "primary_delta": _weighted_moments(
            [float(row["paired_delta_unit_net_pnl"]) for row in rows],
            weights,
        ),
        "absolute_moe": _weighted_moments(
            [float(row["moe_unit_net_pnl"]) for row in rows],
            weights,
        ),
    }


def _weighted_moments(
    values: Sequence[float],
    weights: Sequence[float],
) -> dict[str, float]:
    weight_sum = sum(weights)
    mean = sum(weight * value for weight, value in zip(weights, values, strict=True))
    mean /= weight_sum
    variance_numerator = sum(
        weight * (value - mean) ** 2
        for weight, value in zip(weights, values, strict=True)
    )
    denominator = weight_sum - (
        sum(weight * weight for weight in weights) / weight_sum
    )
    variance = variance_numerator / denominator
    return {
        "mean": mean,
        "sample_variance": variance,
        "sample_standard_deviation": math.sqrt(variance),
    }


def _target_share_weight(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    target_value: Any,
    target_share: float,
) -> Callable[[Mapping[str, Any]], float]:
    current_count = sum(row[field] == target_value for row in rows)
    other_count = len(rows) - current_count
    if current_count == 0 or other_count == 0:
        raise ValueError(f"cannot reweight degenerate field: {field}")
    target_weight = target_share / current_count
    other_weight = (1.0 - target_share) / other_count
    return lambda row: target_weight if row[field] == target_value else other_weight


def _power_row(
    *,
    scenario_family: str,
    scenario_name: str,
    sample_size: int,
    primary_mean: float,
    primary_variance: float,
    absolute_mean: float,
    absolute_variance: float,
    variance_multiplier: float,
) -> dict[str, Any]:
    z_value = NormalDist().inv_cdf(0.975)
    primary_probability = NormalDist().cdf(
        primary_mean * math.sqrt(sample_size) / math.sqrt(primary_variance)
        - z_value
    )
    absolute_probability = NormalDist().cdf(
        absolute_mean * math.sqrt(sample_size) / math.sqrt(absolute_variance)
        - z_value
    )
    return {
        "scenario_family": scenario_family,
        "scenario_name": scenario_name,
        "sample_size": sample_size,
        "variance_multiplier": variance_multiplier,
        "assumed_primary_delta_mean": primary_mean,
        "assumed_primary_delta_variance": primary_variance,
        "assumed_absolute_moe_mean": absolute_mean,
        "assumed_absolute_moe_variance": absolute_variance,
        "estimated_probability_primary_delta_LCB_gt_0": primary_probability,
        "estimated_probability_absolute_moe_LCB_gt_0": absolute_probability,
    }


def _write_collection_quality_analysis(
    *,
    repo_root: Path,
    v2_dir: Path,
    target: int,
    created_at: str,
) -> dict[str, Any]:
    health_path = (
        repo_root
        / "examples/v8/polymarket_runs/"
        "challenge-model-development-btc-updown-15m-v1/"
        "development_lane_health_latest.json"
    )
    health = _load_json(health_path)
    if health["outcomes_labels_or_pnl_read_for_health"] is not False:
        raise ValueError("collection quality ledger is not outcome blind")
    attempted = int(health["cumulative"]["attempted_market_count"])
    quality_valid = int(health["cumulative"]["quality_valid_market_count"])
    if (attempted, quality_valid) != (120, 113):
        raise ValueError("unexpected outcome-blind collection quality population")
    confidence = 0.975
    z_value = NormalDist().inv_cdf(confidence)
    observed = quality_valid / attempted
    denominator = 1.0 + (z_value * z_value / attempted)
    center = (
        observed + z_value * z_value / (2.0 * attempted)
    ) / denominator
    margin = z_value * math.sqrt(
        observed * (1.0 - observed) / attempted
        + z_value * z_value / (4.0 * attempted * attempted)
    ) / denominator
    lower = center - margin
    attempt_cap = math.ceil(target / lower)
    analysis_path = v2_dir / "collection_quality_rate_analysis.json"
    analysis = {
        "schema_version": "bigan-btc-15m-collection-quality-rate-analysis-v2",
        "lineage_id": V2_LINEAGE_ID,
        "created_at": created_at,
        "role": "outcome_blind_attempt_cap_derivation",
        "source_attempt_health_ledger": _descriptor(health_path),
        "source_fields_used": [
            "cumulative.attempted_market_count",
            "cumulative.quality_valid_market_count",
            "outcomes_labels_or_pnl_read_for_health",
        ],
        "outcomes_labels_or_pnl_read_for_cap_selection": False,
        "model_outputs_used_for_cap_selection": False,
        "attempted_market_count": attempted,
        "quality_valid_market_count": quality_valid,
        "observed_quality_valid_rate": observed,
        "conservative_rate_method": "one_sided_97_5pct_wilson_lower_bound",
        "confidence": confidence,
        "z_value": z_value,
        "conservative_quality_rate_lower_bound": lower,
        "target_quality_valid_market_count": target,
        "attempt_cap_formula": (
            "ceil(target_quality_valid_market_count/"
            "conservative_quality_rate_lower_bound)"
        ),
        "attempt_cap": attempt_cap,
        "maximum_quality_valid_markets_per_attempt": 1,
        "target_is_mathematically_reachable_under_cap": attempt_cap >= target,
        "attempt_cap_ready": True,
        "state": {
            "fresh_collection_authorized": False,
            "fresh_collection_started": False,
            "fresh_outcomes_opened": False,
        },
        "safety": dict(SAFETY),
    }
    _write_frozen_json(analysis_path, analysis)
    return {
        "analysis": analysis,
        "analysis_path": analysis_path,
        "analysis_sha256": sha256_file(analysis_path),
    }


def _write_reporting_contract(
    *,
    v2_dir: Path,
    created_at: str,
) -> dict[str, Any]:
    output = v2_dir / "moe_future_evaluation_reporting_contract.json"
    payload = {
        "schema_version": "bigan-btc-15m-moe-future-reporting-contract-v2",
        "lineage_id": V2_LINEAGE_ID,
        "created_at": created_at,
        "population": {
            "one_row_per_frozen_market_required": True,
            "NO_TRADE_rows_required": True,
            "all_frozen_markets_must_reconcile_to_target": True,
            "dropped_market_count_must_equal": 0,
        },
        "required_market_fields": [
            "market_id",
            "decision_ts",
            "requested_route",
            "expert_id",
            "expert_training_market_count",
            "expert_available",
            "fallback_used",
            "actual_model_used",
            "candidate_selected_side",
            "baseline_selected_side",
            "candidate_accepted",
            "baseline_accepted",
            "candidate_unit_net_pnl",
            "baseline_unit_net_pnl",
            "paired_delta_unit_net_pnl",
            "provider_health",
            "feature_missingness",
            "cost_decomposition",
            "chronological_half",
        ],
        "required_panels": [
            "overall",
            "requested_route",
            "actual_model",
            "expert_vs_fallback",
            "UP_vs_DOWN",
            "regime",
            "provider_health",
            "complete_feature_vs_missing_feature",
            "chronological_half",
            "largest_winner_attribution",
        ],
        "attribution_completeness_required": True,
        "missingness_semantics": {
            "native_missing_value": "nan",
            "explicit_missing_indicator_required": True,
            "missing_value_encoded_as_numeric_zero_allowed": False,
        },
        "forbidden_reporting_behavior": {
            "route_filtering": True,
            "missingness_filtering": True,
            "post_hoc_exclusions": True,
            "outlier_deletion": True,
            "winner_language_before_all_gates_pass": True,
            "pooled_panels_rescuing_failed_gates": True,
        },
        "state": {
            "contract_frozen": True,
            "fresh_collection_authorized": False,
            "fresh_collection_started": False,
            "fresh_outcomes_opened": False,
        },
        "safety": dict(SAFETY),
    }
    _write_frozen_json(output, payload)
    return {"path": output, "sha256": sha256_file(output), "payload": payload}


def _write_collector_protocol(
    *,
    v2_dir: Path,
    power: Mapping[str, Any],
    quality: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    output = v2_dir / "moe_confirmatory_collector_protocol.json"
    target = int(power["analysis"]["selected_confirmatory_market_count"])
    attempt_cap = int(quality["analysis"]["attempt_cap"])
    payload = {
        "schema_version": "bigan-btc-15m-moe-confirmatory-collector-v2",
        "lineage_id": V2_LINEAGE_ID,
        "created_at": created_at,
        "collection_design": {
            "one_fixed_window": True,
            "target_quality_valid_market_count": target,
            "attempt_cap": attempt_cap,
            "maximum_quality_valid_markets_per_attempt": 1,
            "target_is_mathematically_reachable_under_cap": attempt_cap >= target,
            "chronological_earliest_quality_valid_unique_markets_required": True,
        },
        "attempt_cap_derivation": _descriptor(quality["analysis_path"]),
        "power_analysis": _descriptor(power["analysis_path"]),
        "outcome_blind_capture_required": True,
        "capture_control_may_read_outcomes_labels_or_pnl": False,
        "no_optional_stopping": True,
        "no_interim_outcome_looks": True,
        "no_failed_window_replacement": True,
        "state": {
            "collector_protocol_frozen": True,
            "fresh_collection_authorized": False,
            "fresh_collection_started": False,
            "fresh_outcomes_opened": False,
        },
        "safety": dict(SAFETY),
    }
    _write_frozen_json(output, payload)
    return {"path": output, "sha256": sha256_file(output), "payload": payload}


def _write_confirmatory_protocol(
    *,
    v2_dir: Path,
    bundle: Mapping[str, Any],
    baseline: Mapping[str, Any],
    runtime_report: Mapping[str, Any],
    power: Mapping[str, Any],
    collector: Mapping[str, Any],
    reporting: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    if power["analysis"]["confirmatory_design_ready"] is not True:
        raise ValueError("cannot freeze v2 confirmatory protocol without power")
    target = int(power["analysis"]["selected_confirmatory_market_count"])
    output = v2_dir / "moe_confirmatory_protocol.json"
    payload = {
        "schema_version": "bigan-btc-15m-moe-confirmatory-protocol-v2",
        "lineage_id": V2_LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at": created_at,
        "design": {
            "one_fixed_confirmatory_window": True,
            "target_quality_valid_market_count": target,
            "optional_stopping_allowed": False,
            "interim_outcome_looks_allowed": False,
            "failed_window_replacement_allowed": False,
            "combined_result_may_rescue_failed_gate": False,
        },
        "frozen_inputs": {
            "candidate_bundle_hash": bundle["bundle_hash"],
            "artifact_graph": _descriptor(bundle["graph_path"]),
            "matched_baseline_contract": _descriptor(baseline["path"]),
            "runtime_validation_report": _descriptor(runtime_report["path"]),
            "power_protocol": _descriptor(power["protocol_path"]),
            "power_analysis": _descriptor(power["analysis_path"]),
            "collector_protocol": _descriptor(collector["path"]),
            "reporting_contract": _descriptor(reporting["path"]),
        },
        "bootstrap": {
            "unit": "unique_market",
            "method": "market_level_paired_percentile_bootstrap",
            "seed": 26015,
            "resamples": 10000,
            "confidence": 0.975,
            "quantile": 0.025,
            "candidate_and_baseline_use_identical_resample_indices": True,
            "NO_TRADE": 0.0,
            "route_level_resampling_forbidden": True,
            "trade_level_resampling_forbidden": True,
        },
        "gates": {
            "quality_valid_market_count": {"operator": "gte", "value": target},
            "paired_executable_ask_coverage": {
                "operator": "gte",
                "value": 0.95,
            },
            "moe_total_after_cost_pnl": {"operator": "gt", "value": 0.0},
            "moe_mean_pnl_bootstrap_lcb": {"operator": "gt", "value": 0.0},
            "paired_delta_mean_pnl_bootstrap_lcb": {
                "operator": "gt",
                "value": 0.0,
            },
            "moe_largest_winner_removed_total_pnl": {
                "operator": "gt",
                "value": 0.0,
            },
            "paired_delta_largest_positive_removed_total": {
                "operator": "gt",
                "value": 0.0,
            },
            "first_chronological_half_moe_pnl": {
                "operator": "gte",
                "value": 0.0,
            },
            "second_chronological_half_moe_pnl": {
                "operator": "gte",
                "value": 0.0,
            },
            "first_chronological_half_paired_delta": {
                "operator": "gte",
                "value": 0.0,
            },
            "second_chronological_half_paired_delta": {
                "operator": "gte",
                "value": 0.0,
            },
            "target_or_future_leakage_count": {
                "operator": "eq",
                "value": 0,
            },
            "runtime_artifact_validation_passed": {
                "operator": "eq",
                "value": True,
            },
            "reporting_contract_complete": {
                "operator": "eq",
                "value": True,
            },
            "expert_fallback_attribution_complete": {
                "operator": "eq",
                "value": True,
            },
            "all_gate_booleans_must_be_true": True,
        },
        "state": {
            "protocol_frozen": True,
            "fresh_collection_authorized": False,
            "fresh_collection_started": False,
            "fresh_outcomes_opened": False,
            "confirmatory_evaluation_started": False,
        },
        "safety": dict(SAFETY),
    }
    _write_frozen_json(output, payload)
    return {"path": output, "sha256": sha256_file(output), "payload": payload}


def _write_authorization_template(
    *,
    v2_dir: Path,
    bundle: Mapping[str, Any],
    baseline: Mapping[str, Any],
    runtime_report: Mapping[str, Any],
    power: Mapping[str, Any],
    quality: Mapping[str, Any],
    collector: Mapping[str, Any],
    protocol: Mapping[str, Any],
    reporting: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    output = v2_dir / "moe_fresh_collection_authorization_template.json"
    payload = {
        "schema_version": "bigan-btc-15m-moe-authorization-template-v2",
        "lineage_id": V2_LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at": created_at,
        "artifact_role": "inactive_template_only_not_collection_authority",
        "template_usable_as_collection_authorization": False,
        "frozen_inputs": {
            "candidate_bundle_hash": bundle["bundle_hash"],
            "artifact_graph": _descriptor(bundle["graph_path"]),
            "matched_baseline_contract": _descriptor(baseline["path"]),
            "runtime_validation_report": _descriptor(runtime_report["path"]),
            "power_protocol": _descriptor(power["protocol_path"]),
            "power_analysis": _descriptor(power["analysis_path"]),
            "collection_quality_rate_analysis": _descriptor(
                quality["analysis_path"]
            ),
            "collector_protocol": _descriptor(collector["path"]),
            "statistical_protocol": _descriptor(protocol["path"]),
            "reporting_contract": _descriptor(reporting["path"]),
        },
        "activation_placeholders": {
            "authorization_artifact_id": None,
            "authorized_by": None,
            "authorized_at": None,
            "authorization_source_url": None,
            "authorization_source_id": None,
            "authorization_request_sha256": None,
            "authorization_decision_sha256": None,
            "strictly_later_than_timestamp": None,
            "maximum_attempts": quality["analysis"]["attempt_cap"],
            "maximum_markets": power["analysis"][
                "selected_confirmatory_market_count"
            ],
            "explicit_request_received": False,
        },
        "later_authorization_requirements": [
            "real_approver_identity",
            "stable_governance_source_id",
            "exact_approval_request_text_sha256",
            "exact_approval_decision_text_sha256",
            "approval_timestamp",
            "every_frozen_v2_protocol_and_artifact_hash",
        ],
        "state": {
            "fresh_collection_authorized": False,
            "fresh_collection_started": False,
            "fresh_outcomes_opened": False,
        },
        "safety": dict(SAFETY),
    }
    _write_frozen_json(output, payload)
    return {"path": output, "sha256": sha256_file(output), "payload": payload}


def _write_lineage_manifest(
    *,
    v2_dir: Path,
    genesis: Mapping[str, Any],
    terminal: Mapping[str, Any],
    bundle: Mapping[str, Any],
    runtime_report: Mapping[str, Any],
    power: Mapping[str, Any],
    quality: Mapping[str, Any],
    protocol: Mapping[str, Any],
    collector: Mapping[str, Any],
    reporting: Mapping[str, Any],
    authorization: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    output = v2_dir / "lineage_manifest.json"
    payload = {
        "schema_version": "bigan-btc-15m-moe-confirmatory-lineage-manifest-v2",
        "lineage_id": V2_LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "architecture_type": ARCHITECTURE_TYPE,
        "created_at": created_at,
        "implementation_base_commit": BASE_COMMIT,
        "parent_lineage_id": PARENT_LINEAGE_ID,
        "parent_result_commit": PARENT_RESULT_COMMIT,
        "v1_terminal_record": dict(terminal),
        "genesis_decision": dict(genesis),
        "candidate_bundle": {
            "bundle_hash": bundle["bundle_hash"],
            "artifact_graph": _descriptor(bundle["graph_path"]),
            "model_manifest": _descriptor(bundle["manifest_path"]),
        },
        "protocols": {
            "runtime_validation": _descriptor(runtime_report["path"]),
            "power_protocol": _descriptor(power["protocol_path"]),
            "power_analysis": _descriptor(power["analysis_path"]),
            "collection_quality_rate_analysis": _descriptor(
                quality["analysis_path"]
            ),
            "collector": _descriptor(collector["path"]),
            "confirmatory": _descriptor(protocol["path"]),
            "reporting": _descriptor(reporting["path"]),
            "authorization_template": _descriptor(authorization["path"]),
        },
        "confirmatory_design_ready": power["analysis"][
            "confirmatory_design_ready"
        ],
        "attempt_cap_ready": quality["analysis"]["attempt_cap_ready"],
        "separate_auditable_manual_authorization_decision_required": True,
        "state": {
            "lineage_frozen": True,
            "fresh_collection_authorized": False,
            "fresh_collection_started": False,
            "fresh_outcomes_opened": False,
        },
        "safety": dict(SAFETY),
    }
    _write_frozen_json(output, payload)
    return {"path": output, "sha256": sha256_file(output), "payload": payload}


def build_regression_failure_ledger(
    *,
    base_pytest_junit_path: Path | str,
    head_pytest_junit_path: Path | str,
    base_ruff_json_path: Path | str,
    head_ruff_json_path: Path | str,
    output_path: Path | str,
    base_commit: str,
    head_commit: str,
    created_at: str = CREATED_AT,
) -> dict[str, Any]:
    """Freeze exact normalized base/head pytest and Ruff failure reconciliation."""

    base_pytest_path = Path(base_pytest_junit_path).resolve()
    head_pytest_path = Path(head_pytest_junit_path).resolve()
    base_ruff_path = Path(base_ruff_json_path).resolve()
    head_ruff_path = Path(head_ruff_json_path).resolve()
    base_pytest = _parse_pytest_junit(base_pytest_path)
    head_pytest = _parse_pytest_junit(head_pytest_path)
    base_ruff = _parse_ruff_json(base_ruff_path)
    head_ruff = _parse_ruff_json(head_ruff_path)

    base_by_node = {row["node_id"]: row for row in base_pytest["failures"]}
    head_by_node = {row["node_id"]: row for row in head_pytest["failures"]}
    base_nodes = set(base_by_node)
    head_nodes = set(head_by_node)
    added_nodes = sorted(head_nodes - base_nodes)
    removed_nodes = sorted(base_nodes - head_nodes)
    unchanged_nodes = sorted(
        node
        for node in base_nodes & head_nodes
        if base_by_node[node]["normalized_message_sha256"]
        == head_by_node[node]["normalized_message_sha256"]
    )
    changed_nodes = sorted(
        node
        for node in base_nodes & head_nodes
        if base_by_node[node]["normalized_message_sha256"]
        != head_by_node[node]["normalized_message_sha256"]
    )
    base_ruff_by_key = {row["identity"]: row for row in base_ruff}
    head_ruff_by_key = {row["identity"]: row for row in head_ruff}
    added_ruff = sorted(set(head_ruff_by_key) - set(base_ruff_by_key))
    removed_ruff = sorted(set(base_ruff_by_key) - set(head_ruff_by_key))
    unchanged_ruff = sorted(set(base_ruff_by_key) & set(head_ruff_by_key))
    head_subset_base = not added_nodes and not changed_nodes
    payload = {
        "schema_version": "bigan-btc-15m-moe-regression-failure-ledger-v2",
        "lineage_id": V2_LINEAGE_ID,
        "created_at": created_at,
        "base_commit": base_commit,
        "head_commit": head_commit,
        "commands": {
            "pytest": "PYTHONPATH=src:. python -m pytest tests/v8 -q",
            "ruff": (
                "PYTHONPATH=src python -m ruff check "
                "src/bigan/v8/polymarket examples/v8 tests/v8"
            ),
            "junit_capture_addition": "--junitxml=<temporary-path>",
            "ruff_machine_capture_addition": "--output-format=json",
        },
        "capture_artifact_hashes": {
            "base_pytest_junit_sha256": sha256_file(base_pytest_path),
            "head_pytest_junit_sha256": sha256_file(head_pytest_path),
            "base_ruff_json_sha256": sha256_file(base_ruff_path),
            "head_ruff_json_sha256": sha256_file(head_ruff_path),
        },
        "message_hash_method": (
            "sha256_of_junit_failure_message_after_repository_tmp_and_"
            "object_address_normalization"
        ),
        "base_pytest": base_pytest,
        "head_pytest": head_pytest,
        "pytest_reconciliation": {
            "base_failure_node_ids": sorted(base_nodes),
            "head_failure_node_ids": sorted(head_nodes),
            "added_failure_node_ids": added_nodes,
            "removed_failure_node_ids": removed_nodes,
            "unchanged_failure_node_ids": unchanged_nodes,
            "changed_message_failure_node_ids": changed_nodes,
            "new_test_failure_count": len(added_nodes) + len(changed_nodes),
            "head_failures_subset_of_base_failures": head_subset_base,
        },
        "base_ruff_errors": base_ruff,
        "head_ruff_errors": head_ruff,
        "ruff_reconciliation": {
            "added_error_identities": added_ruff,
            "removed_error_identities": removed_ruff,
            "unchanged_error_identities": unchanged_ruff,
            "new_ruff_error_count": len(added_ruff),
        },
        "required_condition_passed": (
            head_subset_base and not added_ruff
        ),
        "state": {
            "fresh_collection_authorized": False,
            "fresh_collection_started": False,
            "fresh_outcomes_opened": False,
        },
        "safety": dict(SAFETY),
    }
    output = Path(output_path).resolve()
    _write_frozen_json(output, payload)
    return payload


def _parse_pytest_junit(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suite = next(root.iter("testsuite"))
    failures = []
    for testcase in root.iter("testcase"):
        failure = testcase.find("failure")
        error = testcase.find("error")
        failure_element = failure if failure is not None else error
        if failure_element is None:
            continue
        classname = str(testcase.attrib["classname"])
        name = str(testcase.attrib["name"])
        node_id = classname.replace(".", "/") + ".py::" + name
        raw_message = str(failure_element.attrib.get("message") or "")
        normalized = _normalize_failure_message(raw_message)
        failures.append(
            {
                "node_id": node_id,
                "failure_type": failure_element.tag,
                "message": raw_message,
                "message_sha256": hashlib.sha256(
                    raw_message.encode("utf-8")
                ).hexdigest(),
                "normalized_message": normalized,
                "normalized_message_sha256": hashlib.sha256(
                    normalized.encode("utf-8")
                ).hexdigest(),
            }
        )
    failures.sort(key=lambda row: row["node_id"])
    return {
        "tests_collected": int(suite.attrib["tests"]),
        "failure_count": int(suite.attrib["failures"]),
        "error_count": int(suite.attrib["errors"]),
        "skipped_count": int(suite.attrib["skipped"]),
        "passed": (
            int(suite.attrib["tests"])
            - int(suite.attrib["failures"])
            - int(suite.attrib["errors"])
            - int(suite.attrib["skipped"])
        ),
        "failure_records": len(failures),
        "failures_by_node": failures,
        "failures": failures,
    }


def _normalize_failure_message(message: str) -> str:
    normalized = message.replace(str(REPO_ROOT.resolve()), "<REPO_ROOT>")
    normalized = re.sub(
        r"(?:/private)?/tmp/bigan-moe-v2-regression\.[^/]+/base",
        "<REPO_ROOT>",
        normalized,
    )
    normalized = re.sub(
        r"/private/var/folders/[^']+/pytest-of-[^/]+/pytest-\d+",
        "<PYTEST_TMP>",
        normalized,
    )
    return re.sub(r"object at 0x[0-9a-fA-F]+", "object at <ADDRESS>", normalized)


def _parse_ruff_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Ruff JSON output must be a list")
    results = []
    for row in payload:
        filename = str(row["filename"])
        if filename.startswith(str(REPO_ROOT.resolve())):
            filename = Path(filename).relative_to(REPO_ROOT).as_posix()
        message = str(row["message"])
        identity = (
            f"{filename}:{row['location']['row']}:{row['location']['column']}:"
            f"{row['code']}"
        )
        results.append(
            {
                "identity": identity,
                "filename": filename,
                "code": row["code"],
                "location": row["location"],
                "end_location": row["end_location"],
                "message": message,
                "message_sha256": hashlib.sha256(
                    message.encode("utf-8")
                ).hexdigest(),
            }
        )
    return sorted(results, key=lambda row: row["identity"])


def _descriptor(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    repo_root = REPO_ROOT.resolve()
    if not resolved.is_relative_to(repo_root):
        raise ValueError(f"artifact path escaped repository: {resolved}")
    return {
        "path": resolved.relative_to(repo_root).as_posix(),
        "sha256": sha256_file(resolved),
    }


def _write_frozen_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_json(path, dict(payload))
    _write_sha_sidecar(path)


def _write_sha_sidecar(path: Path) -> None:
    path.with_suffix(".sha256").write_text(
        sha256_file(path) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """Build the v2 precollection lineage without granting collection authority."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=REPO_ROOT,
    )
    args = parser.parse_args(argv)
    result = build_v2_precollection_lineage(
        repository_root=args.repository_root,
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
