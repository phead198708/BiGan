"""Train, freeze, resolve, and verify the static BTC 15m MoE artifact."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from bigan.v8.polymarket.challenge_development_lane import (
    atomic_write_json,
    sha256_file,
)
from bigan.v8.polymarket.challenge_model_15m_training import (
    _load_side_symmetric_rows,
    _matrix,
    _verify_finalized_index,
)
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.moe_confirmatory_lineage import (
    LINEAGE_ID,
    REPO_ROOT,
    SAFETY,
    _raw_row_route,
    deterministic_moe_route,
    frozen_expert_or_fallback,
)
from bigan.v8.polymarket.regime_adaptive_candidate_evaluation import FEATURE_NAMES

ARTIFACT_GRAPH_SCHEMA_VERSION = "bigan-btc-15m-moe-artifact-graph-v1"
MODEL_MANIFEST_SCHEMA_VERSION = "bigan-btc-15m-moe-model-manifest-v1"
EXPERT_IDS = ("high_vol", "bullish", "bearish", "low_vol")


def train_static_moe_artifact(
    *,
    candidate_contract_path: Path | str,
    artifact_root: Path | str,
    manifest_output_path: Path | str,
    graph_output_path: Path | str,
    source_commit: str,
    repository_root: Path | str | None = None,
) -> dict[str, Any]:
    """Train exactly one static MoE and freeze it in a content-addressed bundle."""

    repo_root = Path(repository_root or REPO_ROOT).resolve()
    if source_commit != _full_git_sha(source_commit):
        raise ValueError("source_commit must be a full lowercase Git SHA")
    candidate_path = Path(candidate_contract_path).resolve()
    candidate = _load_pinned_json(candidate_path)
    _validate_candidate_contract(candidate)
    router_path = _verify_contract_descriptor(candidate["contracts"]["router"], repo_root)
    feature_path = _verify_contract_descriptor(
        candidate["contracts"]["features"],
        repo_root,
    )
    cost_path = _verify_contract_descriptor(
        candidate["contracts"]["cost_and_action"],
        repo_root,
    )
    reconciliation_path = _verify_contract_descriptor(
        candidate["hardening_inputs"]["metric_reconciliation"],
        repo_root,
    )
    reconciliation = _load_json(reconciliation_path)
    if reconciliation.get("reconciliation_passed") is not True:
        raise ValueError("static MoE training requires passed reconciliation")
    feature_contract = _load_json(feature_path)
    index_path = (
        repo_root / str(candidate["development_population"]["finalized_index_path"])
    ).resolve()
    if (
        not index_path.is_relative_to(repo_root)
        or sha256_file(index_path)
        != candidate["development_population"]["finalized_index_sha256"]
    ):
        raise ValueError("static MoE development population changed")
    index_rows = _verify_finalized_index(
        index_path=index_path,
        repo_root=repo_root,
    )
    rows, input_corpora = _load_side_symmetric_rows(
        index_rows,
        repo_root=repo_root,
    )
    ordered_markets = sorted(
        {
            (int(row["market_start_ts"]), str(row["market_id"]))
            for row in rows
        }
    )
    if len(ordered_markets) != 113:
        raise ValueError("static MoE requires exactly 113 development markets")
    rows_by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_route[_raw_row_route(row, feature_contract)].append(row)
    route_market_ids = {
        route: sorted(
            {str(row["market_id"]) for row in rows_by_route[route]},
            key=lambda market_id: next(
                index
                for index, (_, ordered_id) in enumerate(ordered_markets)
                if ordered_id == market_id
            ),
        )
        for route in EXPERT_IDS
    }
    expected_support = candidate["static_training_protocol"]["route_support"]
    if {
        route: len(route_market_ids[route]) for route in EXPERT_IDS
    } != expected_support:
        raise ValueError("static MoE expert support changed after candidate freeze")

    artifact_root_path = Path(artifact_root).resolve()
    if not artifact_root_path.is_relative_to(repo_root):
        raise ValueError("static MoE artifact root escaped repository")
    artifact_root_path.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".BTC-15M-MoE-confirmatory-v1.",
            dir=artifact_root_path,
        )
    )
    staging_moved = False
    try:
        for source, filename in (
            (candidate_path, "moe_candidate_contract.json"),
            (router_path, "moe_router_contract.json"),
            (feature_path, "moe_feature_contract.json"),
            (cost_path, "moe_cost_and_action_contract.json"),
        ):
            shutil.copyfile(source, staging / filename)
        ordered_feature_path = staging / "ordered_feature_names.json"
        atomic_write_json(
            ordered_feature_path,
            {
                "schema_version": "bigan-btc-15m-moe-ordered-features-v1",
                "feature_count": len(FEATURE_NAMES),
                "feature_names": list(FEATURE_NAMES),
                "feature_names_sha256": canonical_json_sha256(list(FEATURE_NAMES)),
            },
        )
        train_ids = [market_id for _, market_id in ordered_markets[:93]]
        validation_ids = [market_id for _, market_id in ordered_markets[93:]]
        global_train_rows = [
            row for row in rows if str(row["market_id"]) in set(train_ids)
        ]
        global_validation_rows = [
            row for row in rows if str(row["market_id"]) in set(validation_ids)
        ]
        parameters = dict(candidate["model_family"]["parameters"])
        train_matrix = _matrix(
            global_train_rows,
            FEATURE_NAMES,
            label_field="settlement_payout",
        )
        validation_matrix = _matrix(
            global_validation_rows,
            FEATURE_NAMES,
            label_field="settlement_payout",
        )
        fallback = xgb.train(
            params=parameters,
            dtrain=train_matrix,
            num_boost_round=int(
                candidate["model_family"]["maximum_num_boost_round"]
            ),
            evals=[
                (train_matrix, "train"),
                (validation_matrix, "validation"),
            ],
            early_stopping_rounds=int(
                candidate["model_family"]["early_stopping_rounds"]
            ),
            verbose_eval=False,
        )
        fallback_path = staging / "moe_global_fallback.json"
        fallback.save_model(fallback_path)
        expert_boosters: dict[str, xgb.Booster] = {}
        expert_specs: dict[str, dict[str, Any]] = {}
        num_boost_round = int(fallback.best_iteration) + 1
        for route in EXPERT_IDS:
            support = len(route_market_ids[route])
            available = support >= 20
            expert_path = staging / f"moe_expert_{route}.json"
            if available:
                matrix = _matrix(
                    rows_by_route[route],
                    FEATURE_NAMES,
                    label_field="settlement_payout",
                )
                booster = xgb.train(
                    params=parameters,
                    dtrain=matrix,
                    num_boost_round=num_boost_round,
                    verbose_eval=False,
                )
                booster.save_model(expert_path)
                expert_boosters[route] = booster
                model_format = "xgboost_json"
            else:
                atomic_write_json(
                    expert_path,
                    {
                        "schema_version": "bigan-btc-15m-moe-unavailable-expert-v1",
                        "expert_id": route,
                        "available": False,
                        "training_market_count": support,
                        "minimum_training_market_count": 20,
                        "frozen_behavior": "global_fallback",
                    },
                )
                model_format = "support_below_minimum_stub_json"
            expert_specs[route] = {
                "filename": expert_path.name,
                "available": available,
                "model_format": model_format,
                "training_market_count": support,
                "training_market_ids": route_market_ids[route],
                "training_market_ids_sha256": canonical_json_sha256(
                    route_market_ids[route]
                ),
                "training_side_row_count": len(rows_by_route[route]),
                "num_boost_round": num_boost_round if available else 0,
            }
        training_population_path = staging / "training_population_manifest.json"
        atomic_write_json(
            training_population_path,
            {
                "schema_version": "bigan-btc-15m-moe-training-population-v1",
                "development_index": _repo_descriptor(index_path, repo_root),
                "market_count": len(ordered_markets),
                "ordered_market_ids": [
                    market_id for _, market_id in ordered_markets
                ],
                "ordered_market_ids_sha256": canonical_json_sha256(
                    [market_id for _, market_id in ordered_markets]
                ),
                "global_core_training_market_ids": train_ids,
                "global_core_training_market_ids_sha256": canonical_json_sha256(
                    train_ids
                ),
                "global_validation_market_ids": validation_ids,
                "global_validation_market_ids_sha256": canonical_json_sha256(
                    validation_ids
                ),
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
            _build_prediction_fixtures(
                fallback=fallback,
                expert_boosters=expert_boosters,
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
            "synthetic_prediction_fixture.json",
            "moe_expert_high_vol.json",
            "moe_expert_bullish.json",
            "moe_expert_bearish.json",
            "moe_expert_low_vol.json",
            "moe_global_fallback.json",
        )
        primary_hashes = {
            filename: sha256_file(staging / filename)
            for filename in primary_filenames
        }
        bundle_hash = canonical_json_sha256(primary_hashes)
        bundle_dir = artifact_root_path / bundle_hash
        if bundle_dir.exists():
            raise FileExistsError(f"static MoE bundle already exists: {bundle_dir}")
        os.replace(staging, bundle_dir)
        staging_moved = True

        manifest = {
            "schema_version": MODEL_MANIFEST_SCHEMA_VERSION,
            "lineage_id": LINEAGE_ID,
            "candidate_id": "mixture_of_experts",
            "architecture_type": (
                "deterministic_regime_router_with_conditional_experts_"
                "and_global_fallback"
            ),
            "source_commit": source_commit,
            "bundle_hash_method": (
                "canonical_json_sha256_of_primary_filename_to_full_content_sha256"
            ),
            "bundle_hash": bundle_hash,
            "bundle_repo_path": bundle_dir.relative_to(repo_root).as_posix(),
            "primary_artifact_hashes": primary_hashes,
            "router_contract": _bundle_descriptor(
                bundle_dir / "moe_router_contract.json",
                repo_root,
            ),
            "feature_contract": _bundle_descriptor(
                bundle_dir / "moe_feature_contract.json",
                repo_root,
            ),
            "cost_and_action_contract": _bundle_descriptor(
                bundle_dir / "moe_cost_and_action_contract.json",
                repo_root,
            ),
            "ordered_features": _bundle_descriptor(
                bundle_dir / "ordered_feature_names.json",
                repo_root,
            ),
            "training_population": _bundle_descriptor(
                bundle_dir / "training_population_manifest.json",
                repo_root,
            ),
            "global_fallback": {
                **_bundle_descriptor(fallback_path := bundle_dir / "moe_global_fallback.json", repo_root),
                "model_format": "xgboost_json",
                "best_iteration": int(fallback.best_iteration),
                "best_score": float(fallback.best_score),
                "num_boost_round_used": num_boost_round,
            },
            "matched_global_baseline": {
                "definition": "byte_identical_global_fallback_model",
                **_bundle_descriptor(fallback_path, repo_root),
            },
            "experts": {
                route: {
                    **expert_specs[route],
                    **_bundle_descriptor(
                        bundle_dir / f"moe_expert_{route}.json",
                        repo_root,
                    ),
                }
                for route in EXPERT_IDS
            },
            "synthetic_prediction_fixture": _bundle_descriptor(
                bundle_dir / "synthetic_prediction_fixture.json",
                repo_root,
            ),
            "model_training_completed": True,
            "static_model_frozen": True,
            "fresh_collection_authorized": False,
            "fresh_collection_started": False,
            "fresh_outcomes_opened": False,
            "development_only_forever": True,
            "promotion_evidence_eligible": False,
            "safety": dict(SAFETY),
        }
        manifest_path = bundle_dir / "moe_model_manifest.json"
        atomic_write_json(manifest_path, manifest)
        graph_artifacts = {
            filename: _bundle_descriptor(bundle_dir / filename, repo_root)
            for filename in primary_filenames
        }
        graph_artifacts["moe_model_manifest.json"] = _bundle_descriptor(
            manifest_path,
            repo_root,
        )
        graph = {
            "schema_version": ARTIFACT_GRAPH_SCHEMA_VERSION,
            "lineage_id": LINEAGE_ID,
            "candidate_id": "mixture_of_experts",
            "architecture_type": (
                "deterministic_regime_router_with_conditional_experts_"
                "and_global_fallback"
            ),
            "bundle_hash": bundle_hash,
            "bundle_repo_path": bundle_dir.relative_to(repo_root).as_posix(),
            "artifact_count": len(graph_artifacts),
            "artifacts": graph_artifacts,
            "graph_content_sha256": canonical_json_sha256(graph_artifacts),
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
        manifest_output = Path(manifest_output_path).resolve()
        graph_output = Path(graph_output_path).resolve()
        shutil.copyfile(manifest_path, manifest_output)
        shutil.copyfile(graph_path, graph_output)
        _write_sha_sidecar(manifest_output)
        _write_sha_sidecar(graph_output)
        return {
            "bundle_hash": bundle_hash,
            "bundle_dir": str(bundle_dir),
            "manifest_path": str(manifest_output),
            "manifest_sha256": sha256_file(manifest_output),
            "graph_path": str(graph_output),
            "graph_sha256": sha256_file(graph_output),
            "expert_hashes": {
                route: graph_artifacts[f"moe_expert_{route}.json"]["sha256"]
                for route in EXPERT_IDS
            },
            "fallback_sha256": graph_artifacts["moe_global_fallback.json"][
                "sha256"
            ],
            "fresh_collection_authorized": False,
            "safety": dict(SAFETY),
        }
    finally:
        if not staging_moved and staging.exists():
            shutil.rmtree(staging)


def load_and_verify_static_moe_artifact(
    *,
    graph_path: Path | str,
    expected_graph_sha256: str,
    repository_root: Path | str | None = None,
) -> dict[str, Any]:
    """Resolve a fresh-clone graph, verify every byte, and reproduce fixtures."""

    repo_root = Path(repository_root or REPO_ROOT).resolve()
    graph_file = Path(graph_path).resolve()
    if sha256_file(graph_file) != expected_graph_sha256:
        raise ValueError("static MoE graph SHA-256 mismatch")
    graph = _load_json(graph_file)
    if not (
        graph.get("schema_version") == ARTIFACT_GRAPH_SCHEMA_VERSION
        and graph.get("lineage_id") == LINEAGE_ID
        and graph.get("candidate_id") == "mixture_of_experts"
        and graph.get("machine_local_absolute_paths_allowed") is False
        and graph.get("fresh_collection_authorized") is False
        and graph.get("promotion_evidence_eligible") is False
        and dict(graph.get("safety") or {}) == SAFETY
    ):
        raise ValueError("static MoE graph governance mismatch")
    resolved: dict[str, Path] = {}
    for filename, descriptor in graph["artifacts"].items():
        path = (repo_root / str(descriptor["path"])).resolve()
        if not path.is_relative_to(repo_root):
            raise ValueError(f"static MoE artifact escaped repository: {filename}")
        if not path.is_file() or sha256_file(path) != descriptor["sha256"]:
            raise ValueError(f"static MoE artifact SHA-256 mismatch: {filename}")
        resolved[str(filename)] = path
    if canonical_json_sha256(graph["artifacts"]) != graph["graph_content_sha256"]:
        raise ValueError("static MoE graph content hash mismatch")
    manifest = _load_json(resolved["moe_model_manifest.json"])
    router = _load_json(resolved["moe_router_contract.json"])
    features = _load_json(resolved["ordered_feature_names.json"])
    cost = _load_json(resolved["moe_cost_and_action_contract.json"])
    fallback = xgb.Booster()
    fallback.load_model(resolved["moe_global_fallback.json"])
    boosters: dict[str, xgb.Booster] = {}
    loaded_experts: dict[str, str] = {}
    for route in EXPERT_IDS:
        spec = manifest["experts"][route]
        path = resolved[f"moe_expert_{route}.json"]
        if spec["available"]:
            booster = xgb.Booster()
            booster.load_model(path)
            boosters[route] = booster
            loaded_experts[route] = "xgboost_json"
        else:
            stub = _load_json(path)
            if not (
                stub.get("available") is False
                and int(stub.get("training_market_count") or 0) < 20
            ):
                raise ValueError(f"static MoE unavailable expert stub invalid: {route}")
            loaded_experts[route] = "support_below_minimum_stub_json"
    fixtures = _load_json(resolved["synthetic_prediction_fixture.json"])
    fixture_results = [
        _reproduce_fixture(
            fixture=fixture,
            fallback=fallback,
            experts=boosters,
            route_support={
                route: int(manifest["experts"][route]["training_market_count"])
                for route in EXPERT_IDS
            },
            feature_names=features["feature_names"],
        )
        for fixture in fixtures["fixtures"]
    ]
    if router["architecture_type"] != manifest["architecture_type"]:
        raise ValueError("static MoE architecture semantics mismatch")
    if cost["cost_semantics"]["complement_quote_proxy_allowed"] is not False:
        raise ValueError("static MoE complement proxy unexpectedly enabled")
    return {
        "lineage_id": graph["lineage_id"],
        "candidate_id": graph["candidate_id"],
        "architecture_type": graph["architecture_type"],
        "bundle_hash": graph["bundle_hash"],
        "artifact_count": len(resolved),
        "loaded_experts": loaded_experts,
        "fallback_loaded": True,
        "fixture_results": fixture_results,
        "all_fixtures_reproduced": all(
            result["reproduced"] for result in fixture_results
        ),
        "fresh_collection_authorized": False,
        "safety": dict(SAFETY),
    }


def _build_prediction_fixtures(
    *,
    fallback: xgb.Booster,
    expert_boosters: Mapping[str, xgb.Booster],
    route_support: Mapping[str, int],
) -> dict[str, Any]:
    definitions = (
        {
            "fixture_id": "high_vol_native_expert",
            "router_inputs": {
                "decision_ts": 1_000,
                "available_at_ts": 1_000,
                "max_input_ts": 999,
                "volatility_bucket": "high",
                "btc_return_regime": "bearish",
            },
            "feature_overrides": {"btc_volatility_15m": 0.0005},
            "execution_costs": [0.45, 0.45],
        },
        {
            "fixture_id": "low_vol_global_fallback",
            "router_inputs": {
                "decision_ts": 2_000,
                "available_at_ts": 2_000,
                "max_input_ts": 1_999,
                "volatility_bucket": "medium",
                "btc_return_regime": "sideways",
            },
            "feature_overrides": {"btc_volatility_15m": 0.00035},
            "execution_costs": [0.45, 0.45],
        },
    )
    fixtures = []
    for definition in definitions:
        route = deterministic_moe_route(definition["router_inputs"])
        model_used = frozen_expert_or_fallback(
            route=route,
            expert_training_market_count=int(route_support[route]),
        )
        booster = (
            fallback
            if model_used == "global_baseline_fallback"
            else expert_boosters[route]
        )
        values = _fixture_values(definition["feature_overrides"])
        raw = booster.predict(
            xgb.DMatrix(
                np.asarray([values, values], dtype=np.float64),
                feature_names=list(FEATURE_NAMES),
                missing=np.nan,
            )
        )
        normalized = raw / float(np.sum(raw))
        scores = [
            float(normalized[index]) - float(definition["execution_costs"][index])
            for index in range(2)
        ]
        selected_index = max(range(2), key=lambda index: (scores[index], -index))
        accepted = scores[selected_index] > 0.0
        fixtures.append(
            {
                **definition,
                "default_feature_value": 0.0,
                "route": route,
                "expert_training_market_count": int(route_support[route]),
                "actual_model_used": model_used,
                "expected_raw_probabilities": [float(value) for value in raw],
                "expected_normalized_probabilities": [
                    float(value) for value in normalized
                ],
                "expected_scores": scores,
                "expected_selected_side": (
                    ("UP", "DOWN")[selected_index] if accepted else None
                ),
                "expected_accepted": accepted,
            }
        )
    return {
        "schema_version": "bigan-btc-15m-moe-synthetic-fixtures-v1",
        "feature_names_sha256": canonical_json_sha256(list(FEATURE_NAMES)),
        "fixtures": fixtures,
    }


def _reproduce_fixture(
    *,
    fixture: Mapping[str, Any],
    fallback: xgb.Booster,
    experts: Mapping[str, xgb.Booster],
    route_support: Mapping[str, int],
    feature_names: Sequence[str],
) -> dict[str, Any]:
    route = deterministic_moe_route(fixture["router_inputs"])
    model_used = frozen_expert_or_fallback(
        route=route,
        expert_training_market_count=int(route_support[route]),
    )
    booster = (
        fallback if model_used == "global_baseline_fallback" else experts[route]
    )
    values = _fixture_values(
        fixture["feature_overrides"],
        feature_names=feature_names,
    )
    raw = booster.predict(
        xgb.DMatrix(
            np.asarray([values, values], dtype=np.float64),
            feature_names=list(feature_names),
            missing=np.nan,
        )
    )
    normalized = raw / float(np.sum(raw))
    scores = [
        float(normalized[index]) - float(fixture["execution_costs"][index])
        for index in range(2)
    ]
    selected_index = max(range(2), key=lambda index: (scores[index], -index))
    accepted = scores[selected_index] > 0.0
    selected_side = ("UP", "DOWN")[selected_index] if accepted else None
    reproduced = (
        route == fixture["route"]
        and model_used == fixture["actual_model_used"]
        and np.allclose(
            raw,
            fixture["expected_raw_probabilities"],
            rtol=0.0,
            atol=1e-12,
        )
        and np.allclose(
            normalized,
            fixture["expected_normalized_probabilities"],
            rtol=0.0,
            atol=1e-12,
        )
        and np.allclose(
            scores,
            fixture["expected_scores"],
            rtol=0.0,
            atol=1e-12,
        )
        and accepted is fixture["expected_accepted"]
        and selected_side == fixture["expected_selected_side"]
    )
    if not reproduced:
        raise ValueError(f"static MoE fixture drifted: {fixture['fixture_id']}")
    return {
        "fixture_id": fixture["fixture_id"],
        "route": route,
        "actual_model_used": model_used,
        "selected_side": selected_side,
        "accepted": accepted,
        "reproduced": reproduced,
    }


def _fixture_values(
    overrides: Mapping[str, float],
    *,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> list[float]:
    values = dict.fromkeys(feature_names, 0.0)
    for name, value in overrides.items():
        if name not in values:
            raise ValueError(f"synthetic fixture feature unknown: {name}")
        values[str(name)] = float(value)
    return [values[name] for name in feature_names]


def _validate_candidate_contract(candidate: Mapping[str, Any]) -> None:
    if not (
        candidate.get("lineage_id") == LINEAGE_ID
        and candidate.get("candidate_id") == "mixture_of_experts"
        and candidate.get("architecture_type")
        == "deterministic_regime_router_with_conditional_experts_and_global_fallback"
        and candidate.get("development_only_forever") is True
        and candidate.get("promotion_evidence_eligible") is False
        and dict(candidate.get("safety") or {}) == SAFETY
        and candidate["hardening_inputs"]["metric_reconciliation"][
            "reconciliation_passed"
        ]
        is True
        and candidate["frozen_behavior"]["complement_proxy_allowed"] is False
        and candidate["state"]["fresh_collection_authorized"] is False
    ):
        raise ValueError("static MoE candidate contract invalid")


def _verify_contract_descriptor(
    descriptor: Mapping[str, Any],
    repo_root: Path,
) -> Path:
    path = (repo_root / str(descriptor["path"])).resolve()
    if (
        not path.is_relative_to(repo_root)
        or not path.is_file()
        or sha256_file(path) != descriptor["sha256"]
    ):
        raise ValueError(f"static MoE contract descriptor mismatch: {path}")
    return path


def _bundle_descriptor(path: Path, repo_root: Path) -> dict[str, str]:
    return _repo_descriptor(path, repo_root)


def _repo_descriptor(path: Path, repo_root: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_relative_to(repo_root):
        raise ValueError(f"artifact escaped repository root: {resolved}")
    return {
        "path": resolved.relative_to(repo_root).as_posix(),
        "sha256": sha256_file(resolved),
    }


def _load_pinned_json(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != sha256_file(
        path
    ):
        raise ValueError(f"static MoE pinned JSON mismatch: {path}")
    return _load_json(path)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _write_sha_sidecar(path: Path) -> None:
    sidecar = path.with_suffix(".sha256")
    sidecar.write_text(sha256_file(path) + "\n", encoding="utf-8")


def _full_git_sha(value: str) -> str:
    if len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        return ""
    return value
