"""Frozen final-fit and read-only runtime for residual promotion v1.

This module carries the exact BTC-15M cost-aware residual v4 challenger into a
single-slot prospective program.  It deliberately exposes no settlement,
wallet, order, or exchange-write surface.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.challenge_model_15m_training import (
    BASE_FEATURE_NAMES,
    GLOBAL_RAW_DEPENDENCIES,
    SIDE_RAW_SUFFIXES,
    side_symmetric_features,
)
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.cost_aware_residual_v4_stacking import (
    META_MAX_ITERATIONS,
    META_REGULARIZATION,
    META_TOLERANCE,
    PROBABILITY_CLIP_EPSILON,
    _nested_meta_training_data,
    fit_fixed_l2_logistic_stacker,
)
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.regime_adaptive_candidate_evaluation import FEATURE_NAMES

LINEAGE_ID = "BTC-15M-cost-aware-market-residual-promotion-v1"
CANDIDATE_ID = "residual-v4-challenger-carry-forward-final-fit-001"
SCHEMA_VERSION = "bigan-btc-15m-residual-promotion-v1"
BUNDLE_SCHEMA_VERSION = "bigan-btc-15m-residual-promotion-stacker-bundle-v1"
RUNTIME_RESULT_SCHEMA_VERSION = "bigan-btc-15m-residual-promotion-runtime-result-v1"
MARKET_FAMILY = "btc_updown_15m"
MARKET_HORIZON_MS = 900_000
TARGET_MARKETS = 2_500
MAXIMUM_ATTEMPTS = 3_000
SIDES = ("UP", "DOWN")
REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_DIR = (
    REPO_ROOT
    / "examples/v8/polymarket_configs"
    / LINEAGE_ID
)
SOURCE_V4_DIR = (
    REPO_ROOT
    / "examples/v8/polymarket_configs"
    / "BTC-15M-cost-aware-market-residual-v4"
)
SOURCE_PROTOCOL = SOURCE_V4_DIR / "residual_v4_challenger_slot_002_protocol.json"
SOURCE_OOF_DIR = SOURCE_V4_DIR / "residual_v4_challenger_slot_002_oof"
SOURCE_DATASET = SOURCE_OOF_DIR / "residual_v4_stacking_development_dataset_rows.jsonl"
SOURCE_OOF_MANIFEST = SOURCE_OOF_DIR / "residual_v4_stacking_oof_manifest.json"
SOURCE_OOF_REPORT = SOURCE_OOF_DIR / "residual_v4_stacking_oof_report.json"
SOURCE_TERMINAL = SOURCE_V4_DIR / "residual_v4_development_terminal_review.json"
BASELINE_CONTRACT = (
    REPO_ROOT
    / "examples/v8/polymarket_configs/BTC-15M-MoE-confirmatory-v2/"
    "moe_matched_global_baseline_contract.json"
)
FEATURE_CONTRACT = (
    REPO_ROOT
    / "examples/v8/polymarket_configs/BTC-15M-MoE-confirmatory-v2/"
    "moe_feature_contract.json"
)
COST_CONTRACT = (
    REPO_ROOT
    / "examples/v8/polymarket_configs/BTC-15M-MoE-confirmatory-v2/"
    "moe_cost_and_action_contract.json"
)
IMPLEMENTATION_PATH = "src/bigan/v8/polymarket/residual_promotion_v1.py"
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
HEX_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class ResidualPromotionError(ValueError):
    """Raised whenever a frozen promotion invariant fails closed."""


@dataclass(frozen=True, slots=True)
class ResidualPromotionRuntime:
    """Hash-verified, read-only two-base-model soft-stacking runtime."""

    candidate_id: str
    lineage_id: str
    manifest_sha256: str
    residual_model_sha256: str
    logit_model_sha256: str
    adapter_sha256: str
    maximum_decision_lag_ms: int
    maximum_source_age_ms: int
    coefficients: tuple[float, float, float]
    residual_booster: xgb.Booster
    logit_booster: xgb.Booster

    def score_feature_row(
        self,
        feature_row: Mapping[str, Any],
        *,
        observed_at_ts: int,
    ) -> dict[str, Any]:
        """Score one decision-time pair; invalid inputs deterministically NO_TRADE."""

        try:
            matrix, decision_ts = _runtime_matrix(
                feature_row,
                observed_at_ts=observed_at_ts,
                maximum_decision_lag_ms=self.maximum_decision_lag_ms,
                maximum_source_age_ms=self.maximum_source_age_ms,
            )
            return self._score_matrix(
                matrix,
                feature_row=feature_row,
                decision_ts=decision_ts,
                observed_at_ts=int(observed_at_ts),
            )
        except (KeyError, TypeError, ValueError, xgb.core.XGBoostError) as exc:
            return _runtime_result(
                runtime=self,
                feature_row=feature_row,
                decision_ts=_safe_int(feature_row.get("decision_ts")),
                observed_at_ts=int(observed_at_ts),
                action_values=None,
                probabilities=None,
                selected_action="NO_TRADE",
                fail_closed_reasons=[str(exc) or exc.__class__.__name__],
            )

    def score_offline_side_rows(
        self,
        side_rows: Mapping[str, Mapping[str, Any]],
        *,
        market_id: str,
        decision_ts: int,
        observed_at_ts: int,
    ) -> dict[str, Any]:
        """Score already transformed side rows for offline/live parity tests."""

        if set(side_rows) != set(SIDES):
            raise ResidualPromotionError("offline side population must be UP/DOWN")
        values = []
        for side in SIDES:
            transformed = dict(side_rows[side])
            if tuple(transformed) != FEATURE_NAMES:
                raise ResidualPromotionError("offline feature order mismatch")
            values.append([float(transformed[name]) for name in FEATURE_NAMES])
        feature_row = {
            "market_id": market_id,
            "decision_ts": int(decision_ts),
        }
        return self._score_matrix(
            np.asarray(values, dtype=np.float64),
            feature_row=feature_row,
            decision_ts=int(decision_ts),
            observed_at_ts=int(observed_at_ts),
        )

    def _score_matrix(
        self,
        matrix: np.ndarray,
        *,
        feature_row: Mapping[str, Any],
        decision_ts: int,
        observed_at_ts: int,
    ) -> dict[str, Any]:
        residual_raw = self.residual_booster.predict(
            xgb.DMatrix(matrix, feature_names=list(FEATURE_NAMES), missing=np.nan)
        )
        anchors = matrix[:, FEATURE_NAMES.index("selected_mid")]
        if residual_raw.shape != (2,) or not np.all(np.isfinite(residual_raw)):
            raise ValueError("residual model did not emit a finite UP/DOWN pair")
        residual_probabilities = _pair_anchored_probabilities(anchors, residual_raw)
        base_margin = np.asarray([_logit(float(value)) for value in anchors])
        logit_raw = self.logit_booster.predict(
            xgb.DMatrix(
                matrix,
                base_margin=base_margin,
                feature_names=list(FEATURE_NAMES),
                missing=np.nan,
            )
        )
        logit_probabilities = _pair_normalize(logit_raw)
        beta = np.asarray(self.coefficients, dtype=np.float64)
        stacked_raw = np.asarray(
            [
                _sigmoid(
                    float(
                        np.asarray(
                            [
                                1.0,
                                _clipped_logit(float(residual_probabilities[index])),
                                _clipped_logit(float(logit_probabilities[index])),
                            ]
                        )
                        @ beta
                    )
                )
                for index in range(2)
            ]
        )
        probabilities = _pair_normalize(stacked_raw)
        costs = _execution_costs_from_matrix(matrix)
        scores = probabilities - costs
        best_index = max(range(2), key=lambda index: (float(scores[index]), -index))
        selected_action = (
            f"BUY_{SIDES[best_index]}_HOLD"
            if float(scores[best_index]) > 0.0
            else "NO_TRADE"
        )
        return _runtime_result(
            runtime=self,
            feature_row=feature_row,
            decision_ts=decision_ts,
            observed_at_ts=observed_at_ts,
            action_values={
                "NO_TRADE": 0.0,
                "BUY_UP_HOLD": float(scores[0]),
                "BUY_DOWN_HOLD": float(scores[1]),
            },
            probabilities={
                "probability_residual": {
                    "UP": float(residual_probabilities[0]),
                    "DOWN": float(residual_probabilities[1]),
                },
                "logit_offset": {
                    "UP": float(logit_probabilities[0]),
                    "DOWN": float(logit_probabilities[1]),
                },
                "soft_stacker": {
                    "UP": float(probabilities[0]),
                    "DOWN": float(probabilities[1]),
                },
            },
            selected_action=selected_action,
            fail_closed_reasons=[],
        )


def prepare_pretraining_freeze(
    *,
    repository_root: Path | str = REPO_ROOT,
    source_commit: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Freeze the one-slot authorization and exact v4 carry-forward protocol."""

    root = Path(repository_root).resolve()
    if not HEX_GIT_SHA.fullmatch(source_commit):
        raise ResidualPromotionError("source_commit must be a full Git SHA")
    stamp = created_at or datetime.now(UTC).isoformat()
    config = root / "examples/v8/polymarket_configs" / LINEAGE_ID
    authorization_path = config / "lineage_authorization.json"
    protocol_path = config / "final_fit_protocol.json"
    source_freeze_path = config / "candidate_source_freeze_manifest.json"
    v4_protocol = _verified_json(SOURCE_PROTOCOL)
    source_descriptors = {
        "v4_challenger_protocol": _descriptor(SOURCE_PROTOCOL, root),
        "v4_challenger_oof_manifest": _descriptor(SOURCE_OOF_MANIFEST, root),
        "v4_challenger_oof_report": _descriptor(SOURCE_OOF_REPORT, root),
        "v4_terminal_review": _descriptor(SOURCE_TERMINAL, root),
        "development_final_fit_rows": _descriptor(SOURCE_DATASET, root),
        "matched_global_baseline_contract": _descriptor(BASELINE_CONTRACT, root),
        "feature_contract": _descriptor(FEATURE_CONTRACT, root),
        "cost_and_action_contract": _descriptor(COST_CONTRACT, root),
        "promotion_implementation": _descriptor(root / IMPLEMENTATION_PATH, root),
    }
    authorization = {
        "schema_version": f"{SCHEMA_VERSION}-authorization",
        "lineage_id": LINEAGE_ID,
        "created_at": stamp,
        "authorization_source": {
            "issue": "https://github.com/phead198708/BiGan/issues/264",
            "scope": "explicit_user_authorization_in_codex_task",
        },
        "candidate_slot": {
            "maximum_slots": 1,
            "candidate_id": CANDIDATE_ID,
            "prediction_changing_modification_requires_reauthorization": True,
            "byte_equivalent_engineering_fix_consumes_slot": False,
            "architecture_feature_threshold_or_hyperparameter_search_allowed": False,
        },
        "carry_forward": {
            "source_lineage": "BTC-15M-cost-aware-market-residual-v4",
            "source_slot": "residual-v4-challenger-slot-002",
            "architecture_features_threshold_and_training_configuration_fixed": True,
            "historical_v4_terminal_result_rewritten": False,
            "historical_v4_n_max_2000_gate_changed": False,
            "historical_v4_failed_gate": "prospective_power_required_market_count_lte_2000",
            "new_sample_plan_is_separate_explicit_authorization": True,
        },
        "final_fit": {
            "opened_development_only_forever_data_allowed": True,
            "fit_count": 1,
            "outcome_aware_development_fit": True,
            "fresh_outcomes_allowed": False,
        },
        "prospective_program": {
            "strictly_later_than_all_existing_data": True,
            "target_quality_valid_market_count": TARGET_MARKETS,
            "maximum_attempts": MAXIMUM_ATTEMPTS,
            "zero_capital": True,
            "read_only": True,
            "outcome_blind_until_population_freeze": True,
            "interim_pnl_evaluation_allowed": False,
            "optional_stopping_allowed": False,
            "candidate_switch_allowed": False,
            "fresh_population_reuse_after_failed_gate_allowed": False,
        },
        "micro_live": {
            "automatic_launch_allowed": False,
            "maximum_capital_fraction_after_separate_approval": 0.01,
            "fresh_confirmation_runtime_parity_phase6_and_rollback_required": True,
        },
        "safety": dict(SAFETY),
    }
    source_freeze = {
        "schema_version": f"{SCHEMA_VERSION}-candidate-source-freeze",
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at": stamp,
        "source_commit": source_commit,
        "sources": source_descriptors,
        "v4_model": v4_protocol["model"],
        "v4_feature_contract": v4_protocol["feature_contract"],
        "v4_action_policy": v4_protocol["action_policy"],
        "v4_pair_coherence": v4_protocol["pair_coherence"],
        "v4_target": v4_protocol["target"],
        "candidate_prediction_bytes_frozen_before_final_fit": True,
        "historical_evidence_modified": False,
        "safety": dict(SAFETY),
    }
    _write_frozen_json(authorization_path, authorization)
    _write_frozen_json(source_freeze_path, source_freeze)
    protocol = {
        "schema_version": f"{SCHEMA_VERSION}-final-fit-protocol",
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at": stamp,
        "source_commit": source_commit,
        "authorization": _descriptor(authorization_path, root),
        "candidate_source_freeze": _descriptor(source_freeze_path, root),
        "development_dataset": source_descriptors["development_final_fit_rows"],
        "model": v4_protocol["model"],
        "feature_contract": v4_protocol["feature_contract"],
        "action_policy": v4_protocol["action_policy"],
        "pair_coherence": v4_protocol["pair_coherence"],
        "target": v4_protocol["target"],
        "final_fit": {
            "market_count": 800,
            "side_decision_row_count": 3_200,
            "base_models_fit_on_all_opened_development_rows": True,
            "meta_fit_method": "expanding_market_grouped_inner_oof_100_then_7x100",
            "meta_fit_oof_market_count": 700,
            "meta_fit_side_decision_row_count": 2_800,
            "fit_exactly_once": True,
            "model_selection_or_early_stopping": False,
            "parameter_or_threshold_search": False,
        },
        "outputs": {
            "residual_model_format": "xgboost_ubj",
            "logit_model_format": "xgboost_ubj",
            "stacker_coefficients_format": "canonical_json",
            "repository_local_hash_bound_bundle": True,
        },
        "fresh_outcomes_accessed": False,
        "promotion_evidence_eligible": False,
        "safety": dict(SAFETY),
    }
    _write_frozen_json(protocol_path, protocol)
    return {
        "authorization": _descriptor(authorization_path, root),
        "candidate_source_freeze": _descriptor(source_freeze_path, root),
        "final_fit_protocol": _descriptor(protocol_path, root),
        "source_commit": source_commit,
        "safety": dict(SAFETY),
    }


def run_final_fit(
    *,
    protocol_path: Path | str,
    expected_protocol_sha256: str,
    repository_root: Path | str = REPO_ROOT,
    source_commit: str,
) -> dict[str, Any]:
    """Consume the sole slot and write one immutable deployable stacker bundle."""

    root = Path(repository_root).resolve()
    protocol_file = _repo_file(protocol_path, root, "final-fit protocol")
    if sha256_file(protocol_file) != _require_sha256(expected_protocol_sha256):
        raise ResidualPromotionError("final-fit protocol SHA-256 mismatch")
    protocol = _verified_json(protocol_file)
    validate_final_fit_protocol(protocol, repository_root=root)
    if source_commit != protocol["source_commit"]:
        raise ResidualPromotionError("final-fit source commit mismatch")
    bundle_dir = protocol_file.parent / "candidate_bundle"
    if bundle_dir.exists() and any(bundle_dir.iterdir()):
        raise FileExistsError("the single final-fit slot is already consumed")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    public_rows = _load_jsonl(_repo_file(SOURCE_DATASET, root, "development dataset"))
    rows, population_order = _internal_training_rows(public_rows)
    rows_by_market: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_market[str(row["market_id"])].append(row)
    meta_features, meta_labels, meta_audit = _nested_meta_training_data(
        training_ids=population_order,
        rows_by_market=rows_by_market,
        protocol=_verified_json(SOURCE_PROTOCOL),
    )
    coefficients = fit_fixed_l2_logistic_stacker(meta_features, meta_labels)
    residual_spec = dict(protocol["model"]["base_learners"]["probability_residual"])
    logit_spec = dict(protocol["model"]["base_learners"]["logit_offset_binomial"])
    matrix = np.vstack([np.asarray(row["features"], dtype=np.float64) for row in rows])
    residual_labels = np.asarray(
        [float(row["probability_residual_target"]) for row in rows]
    )
    binary_labels = np.asarray([float(row["binary_payout_target"]) for row in rows])
    residual_booster = xgb.train(
        params=dict(residual_spec["parameters"]),
        dtrain=xgb.DMatrix(
            matrix,
            label=residual_labels,
            feature_names=list(FEATURE_NAMES),
            missing=np.nan,
        ),
        num_boost_round=int(residual_spec["fixed_num_boost_round"]),
        verbose_eval=False,
    )
    anchors = matrix[:, FEATURE_NAMES.index("selected_mid")]
    logit_booster = xgb.train(
        params=dict(logit_spec["parameters"]),
        dtrain=xgb.DMatrix(
            matrix,
            label=binary_labels,
            base_margin=np.asarray([_logit(float(value)) for value in anchors]),
            feature_names=list(FEATURE_NAMES),
            missing=np.nan,
        ),
        num_boost_round=int(logit_spec["fixed_num_boost_round"]),
        verbose_eval=False,
    )
    residual_path = bundle_dir / "probability_residual_model.ubj"
    logit_path = bundle_dir / "logit_offset_model.ubj"
    residual_booster.save_model(residual_path)
    logit_booster.save_model(logit_path)
    stacker_path = bundle_dir / "soft_stacker_adapter.json"
    feature_path = bundle_dir / "feature_schema.json"
    action_path = bundle_dir / "calibration_action_adapter.json"
    graph_path = bundle_dir / "model_graph.json"
    freeze_path = bundle_dir / "candidate_freeze.json"
    stacker = {
        "schema_version": f"{SCHEMA_VERSION}-soft-stacker-adapter",
        "candidate_id": CANDIDATE_ID,
        "coefficients": [float(value) for value in coefficients],
        "coefficient_order": [
            "intercept",
            "logit_probability_residual_probability",
            "logit_logit_offset_probability",
        ],
        "regularization": META_REGULARIZATION,
        "max_iterations": META_MAX_ITERATIONS,
        "tolerance": META_TOLERANCE,
        "probability_clip_epsilon": PROBABILITY_CLIP_EPSILON,
        "pair_normalize_after_stacker": True,
        "meta_fit_audit": {
            **meta_audit,
            "meta_feature_rows": len(meta_features),
            "meta_features_sha256": canonical_json_sha256(meta_features),
            "meta_labels_sha256": canonical_json_sha256(meta_labels),
        },
        "threshold_or_parameter_search_performed": False,
        "safety": dict(SAFETY),
    }
    feature = {
        "schema_version": f"{SCHEMA_VERSION}-feature-schema",
        "candidate_id": CANDIDATE_ID,
        "base_feature_names": list(BASE_FEATURE_NAMES),
        "ordered_feature_names": list(FEATURE_NAMES),
        "ordered_feature_names_sha256": canonical_json_sha256(list(FEATURE_NAMES)),
        "ordered_feature_count": len(FEATURE_NAMES),
        "side_symmetric": True,
        "side_identity_feature_allowed": False,
        "native_missing_value": "nan",
        "missing_numeric_zero_allowed": False,
        "market_horizon_seconds": 900,
        "causal_decision_time_only": True,
        "source_feature_contract": _descriptor(FEATURE_CONTRACT, root),
        "safety": dict(SAFETY),
    }
    action = {
        "schema_version": f"{SCHEMA_VERSION}-calibration-action-adapter",
        "candidate_id": CANDIDATE_ID,
        "architecture": (
            "fixed_v4_two_base_probability_models_plus_l2_logistic_soft_stacker"
        ),
        "base_pair_normalization": True,
        "stacker_pair_normalization": True,
        "cost_subtraction_after_pair_normalization": True,
        "fees": 0.0002,
        "slippage": "max(0.0001,(executable_ask-executable_bid)/2)",
        "liquidity_impact": "0.00005_if_depth_positive_else_0.001",
        "actions": ["NO_TRADE", "BUY_UP_HOLD", "BUY_DOWN_HOLD"],
        "fixed_acceptance_threshold": 0.0,
        "accept_if": "highest_side_action_value>0",
        "side_tie_break_order": ["UP", "DOWN"],
        "decision_order": "chronological_first_positive_decision",
        "one_trade_maximum_per_market": True,
        "execution_policy": "HOLD_TO_SETTLEMENT",
        "unit_sizing": True,
        "complement_quote_proxy_allowed": False,
        "source_cost_contract": _descriptor(COST_CONTRACT, root),
        "safety": dict(SAFETY),
    }
    _write_frozen_json(stacker_path, stacker)
    _write_frozen_json(feature_path, feature)
    _write_frozen_json(action_path, action)
    model_descriptors = {
        "probability_residual_model": _model_descriptor(
            residual_path, root, residual_spec
        ),
        "logit_offset_model": _model_descriptor(logit_path, root, logit_spec),
        "soft_stacker_adapter": _descriptor(stacker_path, root),
        "feature_schema": _descriptor(feature_path, root),
        "calibration_action_adapter": _descriptor(action_path, root),
    }
    graph = {
        "schema_version": f"{SCHEMA_VERSION}-model-graph",
        "candidate_id": CANDIDATE_ID,
        "lineage_id": LINEAGE_ID,
        "nodes": model_descriptors,
        "edges": [
            ["feature_schema", "probability_residual_model"],
            ["feature_schema", "logit_offset_model"],
            ["probability_residual_model", "soft_stacker_adapter"],
            ["logit_offset_model", "soft_stacker_adapter"],
            ["soft_stacker_adapter", "calibration_action_adapter"],
        ],
        "prediction_semantics": (
            "base_pair_normalize_then_soft_stack_logits_then_pair_normalize_then_cost"
        ),
        "graph_content_sha256": canonical_json_sha256(model_descriptors),
        "safety": dict(SAFETY),
    }
    _write_frozen_json(graph_path, graph)
    freeze = {
        "schema_version": f"{SCHEMA_VERSION}-candidate-freeze",
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "source_commit": source_commit,
        "final_fit_protocol": _descriptor(protocol_file, root),
        "candidate_source_freeze": dict(protocol["candidate_source_freeze"]),
        "single_candidate_slot_consumed": True,
        "fit_executed_exactly_once": True,
        "development_market_count": len(population_order),
        "development_population_sha256": canonical_json_sha256(population_order),
        "model_graph": _descriptor(graph_path, root),
        "children": model_descriptors,
        "fresh_outcomes_accessed": False,
        "candidate_bytes_frozen": True,
        "promotion_evidence_eligible": False,
        "paper_or_live_execution_allowed": False,
        "safety": dict(SAFETY),
    }
    _write_frozen_json(freeze_path, freeze)
    manifest_path = bundle_dir / "bundle_manifest.json"
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "source_commit": source_commit,
        "final_fit_protocol": _descriptor(protocol_file, root),
        "candidate_source_freeze": dict(protocol["candidate_source_freeze"]),
        "candidate_freeze": _descriptor(freeze_path, root),
        "model_graph": _descriptor(graph_path, root),
        "artifacts": model_descriptors,
        "market_contract": {
            "family": MARKET_FAMILY,
            "horizon_seconds": 900,
            "sides": list(SIDES),
        },
        "freshness_contract": {
            "maximum_decision_lag_ms": 5_000,
            "maximum_source_age_ms": 5_000,
            "stale_input_action": "NO_TRADE",
        },
        "runtime_authorization": {
            "zero_capital_read_only_shadow": True,
            "paper_or_live_execution_authorized": False,
            "wallet_signing_authorized": False,
            "polymarket_write_authorized": False,
            "capital_at_risk": False,
        },
        "safety": dict(SAFETY),
    }
    _write_frozen_json(manifest_path, manifest)
    runtime = load_residual_promotion_runtime(
        manifest_path=manifest_path,
        expected_manifest_sha256=sha256_file(manifest_path),
        repository_root=root,
    )
    fixture = _runtime_fixture_from_public_rows(public_rows[:2])
    live = runtime.score_feature_row(
        fixture["live_feature_row"], observed_at_ts=fixture["observed_at_ts"]
    )
    offline = runtime.score_offline_side_rows(
        fixture["side_rows"],
        market_id=fixture["live_feature_row"]["market_id"],
        decision_ts=fixture["live_feature_row"]["decision_ts"],
        observed_at_ts=fixture["observed_at_ts"],
    )
    parity = _parity_projection(live) == _parity_projection(offline)
    if not parity:
        raise ResidualPromotionError("offline/live deterministic parity failed")
    parity_path = bundle_dir / "offline_live_parity_report.json"
    _write_frozen_json(
        parity_path,
        {
            "schema_version": f"{SCHEMA_VERSION}-offline-live-parity",
            "candidate_id": CANDIDATE_ID,
            "fixture_sha256": canonical_json_sha256(fixture),
            "offline_projection": _parity_projection(offline),
            "live_projection": _parity_projection(live),
            "prediction_and_decision_parity": True,
            "fresh_outcomes_accessed": False,
            "safety": dict(SAFETY),
        },
    )
    return {
        "bundle_manifest": _descriptor(manifest_path, root),
        "model_graph": _descriptor(graph_path, root),
        "candidate_freeze": _descriptor(freeze_path, root),
        "parity_report": _descriptor(parity_path, root),
        "single_candidate_slot_consumed": True,
        "fresh_outcomes_accessed": False,
        "safety": dict(SAFETY),
    }


def freeze_prospective_program(
    *,
    repository_root: Path | str = REPO_ROOT,
    bundle_manifest_path: Path | str,
    expected_bundle_manifest_sha256: str,
    collector_implementation_path: Path | str,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Freeze all confirmatory semantics before any fresh outcome is accessible."""

    root = Path(repository_root).resolve()
    bundle_path = _repo_file(bundle_manifest_path, root, "bundle manifest")
    bundle_sha = _require_sha256(expected_bundle_manifest_sha256)
    load_residual_promotion_runtime(
        manifest_path=bundle_path,
        expected_manifest_sha256=bundle_sha,
        repository_root=root,
    )
    collector_path = _repo_file(
        collector_implementation_path, root, "collector implementation"
    )
    config = root / "examples/v8/polymarket_configs" / LINEAGE_ID
    stamp = created_at or datetime.now(UTC).isoformat()
    reporting = {
        "schema_version": f"{SCHEMA_VERSION}-prospective-reporting-contract",
        "population": "exact_chronological_first_2500_quality_valid_unique_markets",
        "required_panels": [
            "overall_candidate_baseline_and_paired_delta",
            "chronological_five_blocks",
            "chronological_halves",
            "largest_winner_removed_candidate_and_delta",
            "cost_stress_1.2x_1.5x_2.0x",
            "identity_missingness_and_runtime_parity",
        ],
        "no_post_hoc_exclusions": True,
        "no_route_side_missingness_or_outlier_filtering": True,
        "safety": dict(SAFETY),
    }
    reporting_path = config / "prospective_reporting_contract.json"
    _write_frozen_json(reporting_path, reporting)
    protocol = {
        "schema_version": f"{SCHEMA_VERSION}-prospective-statistical-protocol",
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at": stamp,
        "candidate_bundle": {"path": bundle_path.relative_to(root).as_posix(), "sha256": bundle_sha},
        "baseline_contract": _descriptor(BASELINE_CONTRACT, root),
        "baseline_artifact": _baseline_model_descriptor(root),
        "feature_contract": _descriptor(FEATURE_CONTRACT, root),
        "cost_contract": _descriptor(COST_CONTRACT, root),
        "reporting_contract": _descriptor(reporting_path, root),
        "population": {
            "target_quality_valid_market_count": TARGET_MARKETS,
            "maximum_attempts": MAXIMUM_ATTEMPTS,
            "strictly_later_than_all_registered_development_and_confirmatory_data": True,
            "chronological_earliest_quality_valid_unique_markets": True,
            "no_replacement_after_outcome_access": True,
            "failed_population_reuse_for_another_candidate": False,
        },
        "bootstrap": {
            "method": "market_level_paired_percentile_bootstrap",
            "seed": 2642500,
            "resamples": 10_000,
            "confidence": 0.975,
            "lower_quantile": 0.025,
            "candidate_and_baseline_share_indices": True,
            "NO_TRADE_participates_as_zero": True,
        },
        "gates": {
            "absolute_candidate_bootstrap_97_5pct_lcb_gt_zero": True,
            "paired_delta_bootstrap_97_5pct_lcb_gt_zero": True,
            "every_one_of_five_chronological_500_market_blocks_candidate_total_gte_zero": True,
            "every_one_of_five_chronological_500_market_blocks_delta_total_gte_zero": True,
            "both_chronological_halves_candidate_total_gte_zero": True,
            "both_chronological_halves_delta_total_gte_zero": True,
            "largest_winner_removed_candidate_total_gte_zero": True,
            "largest_positive_delta_removed_total_gte_zero": True,
            "cost_stress_1_2_1_5_2x_candidate_and_delta_totals_gte_zero": True,
            "market_identity_and_population_reconciliation": True,
            "missingness_and_causality_reconciliation": True,
            "offline_live_prediction_and_decision_parity": True,
        },
        "cost_stress_multipliers": [1.2, 1.5, 2.0],
        "failure_semantics": {
            "any_gate_failure_terminalizes_lineage": True,
            "gate_waiver_allowed": False,
            "rerun_allowed": False,
            "optional_stopping_allowed": False,
            "interim_pnl_evaluation_allowed": False,
        },
        "fresh_outcomes_accessed": False,
        "promotion_evidence_eligible_before_final_evaluation": False,
        "safety": dict(SAFETY),
    }
    protocol_path = config / "prospective_statistical_protocol.json"
    _write_frozen_json(protocol_path, protocol)
    collector = {
        "schema_version": f"{SCHEMA_VERSION}-collector-protocol",
        "lineage_id": LINEAGE_ID,
        "created_at": stamp,
        "candidate_bundle": {"path": bundle_path.relative_to(root).as_posix(), "sha256": bundle_sha},
        "statistical_protocol": _descriptor(protocol_path, root),
        "collector_implementation": _descriptor(collector_path, root),
        "market_family": MARKET_FAMILY,
        "target_quality_valid_market_count": TARGET_MARKETS,
        "maximum_attempts": MAXIMUM_ATTEMPTS,
        "population_order": "market_start_ts_then_market_id_then_attempt_index",
        "capture_only": True,
        "resolution_provider_enabled": False,
        "settlement_finalizer_enabled": False,
        "training_export_enabled": False,
        "outcome_fields_forbidden": True,
        "collection_may_not_use_candidate_or_baseline_decisions": True,
        "stop_only_at_target_or_attempt_cap": True,
        "safety": dict(SAFETY),
    }
    collector_protocol_path = config / "prospective_collector_protocol.json"
    _write_frozen_json(collector_protocol_path, collector)
    authorization = {
        "schema_version": f"{SCHEMA_VERSION}-manual-collection-authorization",
        "lineage_id": LINEAGE_ID,
        "created_at": stamp,
        "authorization_source": "explicit_user_authorization_for_issue_264_promotion_program",
        "candidate_bundle": {"path": bundle_path.relative_to(root).as_posix(), "sha256": bundle_sha},
        "statistical_protocol": _descriptor(protocol_path, root),
        "collector_protocol": _descriptor(collector_protocol_path, root),
        "reporting_contract": _descriptor(reporting_path, root),
        "target_quality_valid_market_count": TARGET_MARKETS,
        "maximum_attempts": MAXIMUM_ATTEMPTS,
        "fresh_collection_authorized": True,
        "fresh_collection_started": False,
        "fresh_outcomes_opened": False,
        "authorization_scope": "zero_capital_read_only_outcome_blind_capture_only",
        "wallet_order_write_or_capital_authorized": False,
        "safety": dict(SAFETY),
    }
    authorization_path = config / "manual_collection_authorization.json"
    _write_frozen_json(authorization_path, authorization)
    return {
        "statistical_protocol": _descriptor(protocol_path, root),
        "collector_protocol": _descriptor(collector_protocol_path, root),
        "reporting_contract": _descriptor(reporting_path, root),
        "manual_collection_authorization": _descriptor(authorization_path, root),
        "fresh_collection_authorized": True,
        "fresh_collection_started": False,
        "fresh_outcomes_opened": False,
        "safety": dict(SAFETY),
    }


def validate_final_fit_protocol(
    payload: Mapping[str, Any], *, repository_root: Path | str = REPO_ROOT
) -> None:
    """Prove that the final fit is the fixed v4 challenger, not a new search."""

    root = Path(repository_root).resolve()
    blockers: list[str] = []
    source = _verified_json(SOURCE_PROTOCOL)
    if payload.get("schema_version") != f"{SCHEMA_VERSION}-final-fit-protocol":
        blockers.append("schema_version")
    if payload.get("lineage_id") != LINEAGE_ID or payload.get("candidate_id") != CANDIDATE_ID:
        blockers.append("identity")
    for field in ("model", "feature_contract", "action_policy", "pair_coherence", "target"):
        if payload.get(field) != source.get(field):
            blockers.append(f"fixed_v4.{field}")
    fit = dict(payload.get("final_fit") or {})
    if fit != {
        "market_count": 800,
        "side_decision_row_count": 3_200,
        "base_models_fit_on_all_opened_development_rows": True,
        "meta_fit_method": "expanding_market_grouped_inner_oof_100_then_7x100",
        "meta_fit_oof_market_count": 700,
        "meta_fit_side_decision_row_count": 2_800,
        "fit_exactly_once": True,
        "model_selection_or_early_stopping": False,
        "parameter_or_threshold_search": False,
    }:
        blockers.append("final_fit")
    if payload.get("fresh_outcomes_accessed") is not False:
        blockers.append("fresh_outcomes_accessed")
    if dict(payload.get("safety") or {}) != SAFETY:
        blockers.append("safety")
    for name in ("authorization", "candidate_source_freeze", "development_dataset"):
        try:
            _verify_descriptor(payload[name], root, name)
        except (KeyError, OSError, TypeError, ValueError, ResidualPromotionError):
            blockers.append(name)
    if blockers:
        raise ResidualPromotionError(
            "invalid residual promotion final-fit protocol: " + ", ".join(blockers)
        )


def load_residual_promotion_runtime(
    *,
    manifest_path: Path | str,
    expected_manifest_sha256: str,
    repository_root: Path | str = REPO_ROOT,
) -> ResidualPromotionRuntime:
    """Load all bundle children repository-relatively and fail closed on drift."""

    root = Path(repository_root).resolve()
    manifest_file = _repo_file(manifest_path, root, "bundle manifest")
    expected = _require_sha256(expected_manifest_sha256)
    if sha256_file(manifest_file) != expected:
        raise ResidualPromotionError("bundle manifest SHA-256 mismatch")
    manifest = _verified_json(manifest_file)
    if not (
        manifest.get("schema_version") == BUNDLE_SCHEMA_VERSION
        and manifest.get("lineage_id") == LINEAGE_ID
        and manifest.get("candidate_id") == CANDIDATE_ID
        and dict(manifest.get("safety") or {}) == SAFETY
        and dict(manifest.get("market_contract") or {})
        == {"family": MARKET_FAMILY, "horizon_seconds": 900, "sides": list(SIDES)}
    ):
        raise ResidualPromotionError("bundle manifest governance mismatch")
    authorization = dict(manifest.get("runtime_authorization") or {})
    if authorization != {
        "zero_capital_read_only_shadow": True,
        "paper_or_live_execution_authorized": False,
        "wallet_signing_authorized": False,
        "polymarket_write_authorized": False,
        "capital_at_risk": False,
    }:
        raise ResidualPromotionError("bundle runtime authorization mismatch")
    artifacts = dict(manifest.get("artifacts") or {})
    expected_names = {
        "probability_residual_model",
        "logit_offset_model",
        "soft_stacker_adapter",
        "feature_schema",
        "calibration_action_adapter",
    }
    if set(artifacts) != expected_names:
        raise ResidualPromotionError("bundle artifact graph is incomplete or extra")
    resolved = {
        name: _verify_descriptor(descriptor, root, name)
        for name, descriptor in artifacts.items()
    }
    graph_path = _verify_descriptor(manifest["model_graph"], root, "model_graph")
    freeze_path = _verify_descriptor(
        manifest["candidate_freeze"], root, "candidate_freeze"
    )
    graph = _verified_json(graph_path)
    freeze = _verified_json(freeze_path)
    if graph.get("nodes") != artifacts or graph.get("graph_content_sha256") != canonical_json_sha256(artifacts):
        raise ResidualPromotionError("model graph descriptor reconciliation failed")
    if not (
        freeze.get("candidate_id") == CANDIDATE_ID
        and freeze.get("single_candidate_slot_consumed") is True
        and freeze.get("fit_executed_exactly_once") is True
        and freeze.get("candidate_bytes_frozen") is True
        and freeze.get("fresh_outcomes_accessed") is False
        and dict(freeze.get("safety") or {}) == SAFETY
    ):
        raise ResidualPromotionError("candidate freeze is invalid")
    feature = _verified_json(resolved["feature_schema"])
    if not (
        tuple(feature.get("base_feature_names") or ()) == BASE_FEATURE_NAMES
        and tuple(feature.get("ordered_feature_names") or ()) == FEATURE_NAMES
        and feature.get("ordered_feature_names_sha256")
        == canonical_json_sha256(list(FEATURE_NAMES))
        and feature.get("missing_numeric_zero_allowed") is False
        and feature.get("market_horizon_seconds") == 900
    ):
        raise ResidualPromotionError("bundle feature schema mismatch")
    adapter = _verified_json(resolved["calibration_action_adapter"])
    if not (
        adapter.get("fixed_acceptance_threshold") == 0.0
        and adapter.get("execution_policy") == "HOLD_TO_SETTLEMENT"
        and adapter.get("unit_sizing") is True
        and adapter.get("complement_quote_proxy_allowed") is False
        and dict(adapter.get("safety") or {}) == SAFETY
    ):
        raise ResidualPromotionError("bundle action adapter mismatch")
    stacker = _verified_json(resolved["soft_stacker_adapter"])
    coefficients = tuple(float(value) for value in stacker.get("coefficients") or ())
    if len(coefficients) != 3 or not all(math.isfinite(value) for value in coefficients):
        raise ResidualPromotionError("bundle stacker coefficients are invalid")
    residual = xgb.Booster()
    logit_model = xgb.Booster()
    residual.load_model(resolved["probability_residual_model"])
    logit_model.load_model(resolved["logit_offset_model"])
    if tuple(residual.feature_names or ()) != FEATURE_NAMES or tuple(logit_model.feature_names or ()) != FEATURE_NAMES:
        raise ResidualPromotionError("bundle model feature order mismatch")
    residual_objective = json.loads(residual.save_config())["learner"]["objective"]["name"]
    logit_objective = json.loads(logit_model.save_config())["learner"]["objective"]["name"]
    if residual_objective != "reg:squarederror" or logit_objective != "binary:logistic":
        raise ResidualPromotionError("bundle model objective mismatch")
    freshness = dict(manifest.get("freshness_contract") or {})
    if freshness.get("stale_input_action") != "NO_TRADE":
        raise ResidualPromotionError("bundle freshness fail-closed action mismatch")
    return ResidualPromotionRuntime(
        candidate_id=CANDIDATE_ID,
        lineage_id=LINEAGE_ID,
        manifest_sha256=expected,
        residual_model_sha256=sha256_file(resolved["probability_residual_model"]),
        logit_model_sha256=sha256_file(resolved["logit_offset_model"]),
        adapter_sha256=sha256_file(resolved["calibration_action_adapter"]),
        maximum_decision_lag_ms=int(freshness["maximum_decision_lag_ms"]),
        maximum_source_age_ms=int(freshness["maximum_source_age_ms"]),
        coefficients=(coefficients[0], coefficients[1], coefficients[2]),
        residual_booster=residual,
        logit_booster=logit_model,
    )


def load_matched_baseline(
    *, repository_root: Path | str = REPO_ROOT
) -> xgb.Booster:
    """Load the exact frozen matched global baseline and verify its contract."""

    root = Path(repository_root).resolve()
    contract = _verified_json(_repo_file(BASELINE_CONTRACT, root, "baseline contract"))
    artifact = dict(contract.get("artifact") or {})
    path = _verify_descriptor(artifact, root, "baseline model")
    if not (
        artifact.get("byte_identical_to_candidate_global_fallback") is True
        and artifact.get("sha256") == artifact.get("candidate_global_fallback_sha256")
        and contract.get("contract_id") == "matched_global_baseline"
        and dict(contract.get("safety") or {}) == SAFETY
    ):
        raise ResidualPromotionError("matched baseline contract mismatch")
    booster = xgb.Booster()
    booster.load_model(path)
    if tuple(booster.feature_names or ()) != FEATURE_NAMES:
        raise ResidualPromotionError("matched baseline feature order mismatch")
    return booster


def score_matched_baseline(
    booster: xgb.Booster, feature_row: Mapping[str, Any]
) -> dict[str, Any]:
    """Produce an outcome-free decision from the frozen matched baseline."""

    matrix = np.asarray(
        [
            [float(side_symmetric_features(feature_row, side)[name]) for name in FEATURE_NAMES]
            for side in SIDES
        ],
        dtype=np.float64,
    )
    probabilities = _pair_normalize(
        booster.predict(
            xgb.DMatrix(matrix, feature_names=list(FEATURE_NAMES), missing=np.nan)
        )
    )
    scores = probabilities - _execution_costs_from_matrix(matrix)
    index = max(range(2), key=lambda item: (float(scores[item]), -item))
    return {
        "action_values": {
            "NO_TRADE": 0.0,
            "BUY_UP_HOLD": float(scores[0]),
            "BUY_DOWN_HOLD": float(scores[1]),
        },
        "selected_action": f"BUY_{SIDES[index]}_HOLD" if scores[index] > 0.0 else "NO_TRADE",
        "probabilities": {"UP": float(probabilities[0]), "DOWN": float(probabilities[1])},
        "outcomes_accessed": False,
        "safety": dict(SAFETY),
    }


def _runtime_matrix(
    feature_row: Mapping[str, Any],
    *,
    observed_at_ts: int,
    maximum_decision_lag_ms: int,
    maximum_source_age_ms: int,
) -> tuple[np.ndarray, int]:
    if feature_row.get("market_family") != MARKET_FAMILY or int(feature_row["horizon_ms"]) != MARKET_HORIZON_MS:
        raise ValueError("runtime input is not BTC 15m")
    decision_ts = int(feature_row["decision_ts"])
    if int(observed_at_ts) < decision_ts or int(observed_at_ts) - decision_ts > maximum_decision_lag_ms:
        raise ValueError("runtime decision input is stale")
    if decision_ts - int(feature_row["max_input_ts"]) > maximum_source_age_ms:
        raise ValueError("runtime source input is stale")
    if any(int(feature_row[field]) > decision_ts for field in ("available_at_ts", "feature_cutoff_ts", "max_input_ts")):
        raise ValueError("runtime input violates causality")
    rows = []
    for side in SIDES:
        transformed = side_symmetric_features(feature_row, side)
        if tuple(transformed) != FEATURE_NAMES:
            raise ValueError("runtime feature order mismatch")
        values = [float(transformed[name]) for name in FEATURE_NAMES]
        for index, name in enumerate(BASE_FEATURE_NAMES):
            missing = values[index + len(BASE_FEATURE_NAMES)]
            if missing not in (0.0, 1.0) or math.isfinite(values[index]) == bool(missing):
                raise ValueError(f"runtime missingness mismatch: {name}")
        rows.append(values)
    return np.asarray(rows, dtype=np.float64), decision_ts


def _internal_training_rows(
    public_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    if len(public_rows) != 3_200:
        raise ResidualPromotionError("final-fit dataset must contain 3200 rows")
    output = []
    positions: dict[str, int] = {}
    for source in public_rows:
        if source.get("development_only_forever") is not True or source.get("promotion_evidence_eligible") is not False:
            raise ResidualPromotionError("final-fit source is not development-only")
        mapping = dict(source.get("features") or {})
        if tuple(mapping) != FEATURE_NAMES:
            raise ResidualPromotionError("final-fit feature order drifted")
        values = [math.nan if mapping[name] is None else float(mapping[name]) for name in FEATURE_NAMES]
        row = dict(source)
        row["features"] = values
        output.append(row)
        market_id = str(row["market_id"])
        position = int(row["market_position"])
        if market_id in positions and positions[market_id] != position:
            raise ResidualPromotionError("market position is inconsistent")
        positions[market_id] = position
    population = [market_id for market_id, _ in sorted(positions.items(), key=lambda item: item[1])]
    if len(population) != 800 or sorted(positions.values()) != list(range(1, 801)):
        raise ResidualPromotionError("final-fit market population mismatch")
    counts = defaultdict(int)
    for row in output:
        counts[str(row["market_id"])] += 1
    if set(counts.values()) != {4}:
        raise ResidualPromotionError("each final-fit market must have four side-decision rows")
    return output, population


def _runtime_fixture_from_public_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(rows) != 2 or [str(row["side"]) for row in rows] != list(SIDES):
        raise ResidualPromotionError("parity fixture must be one ordered UP/DOWN pair")
    # The committed public rows contain the exact transformed representation.
    # Reverse only the deterministic selected/opposite and directional mapping
    # needed by side_symmetric_features; every resulting side row is asserted.
    up = dict(rows[0]["features"])
    down = dict(rows[1]["features"])
    raw: dict[str, Any] = {"horizon_ms": MARKET_HORIZON_MS}
    for suffix in SIDE_RAW_SUFFIXES:
        raw[f"up_{suffix}"] = up[f"selected_{suffix}"]
        raw[f"down_{suffix}"] = down[f"selected_{suffix}"]
    raw.update(
        {
            "up_down_ask_sum": up["paired_ask_sum"],
            "up_down_bid_sum": up["paired_bid_sum"],
            "up_down_mid_sum": up["paired_mid_sum"],
            "combined_spread_bps": up["combined_spread_bps"],
            "chainlink_reference_distance_at_decision": up["signed_chainlink_reference_distance"],
            "btc_return_10s": up["signed_btc_return_10s"],
            "btc_return_30s": up["signed_btc_return_30s"],
            "btc_return_1m": up["signed_btc_return_1m"],
            "btc_return_5m": up["signed_btc_return_5m"],
            "btc_return_15m": up["signed_btc_return_15m"],
            "btc_volatility_1m": up["btc_volatility_1m"],
            "btc_volatility_5m": up["btc_volatility_5m"],
            "btc_volatility_15m": up["btc_volatility_15m"],
            "market_age_seconds": up["market_progress_fraction"] * 900.0,
            "time_to_close_seconds": up["time_remaining_fraction"] * 900.0,
            "provider_health_score": up["provider_health_score"],
            "book_snapshot_pair_ts_delta_ms": up["book_snapshot_pair_ts_delta_ms"],
        }
    )
    relative = up["signed_btc_mid_to_chainlink_relative_distance"]
    raw["chainlink_price_at_decision"] = 60_000.0
    raw["btc_mid_price"] = 60_000.0 * (1.0 + relative)
    decision_ts = int(rows[0]["decision_ts"])
    live = {
        "market_id": str(rows[0]["market_id"]),
        "market_family": MARKET_FAMILY,
        "horizon_ms": MARKET_HORIZON_MS,
        "decision_ts": decision_ts,
        "available_at_ts": decision_ts,
        "feature_cutoff_ts": decision_ts,
        "max_input_ts": decision_ts,
        "features": raw,
        "feature_provenance": {
            name: {"available_at_ts": decision_ts, "max_input_ts": decision_ts}
            for name in (
                {
                    f"{side}_{suffix}"
                    for side in ("up", "down")
                    for suffix in SIDE_RAW_SUFFIXES
                }
                | {
                    dependency
                    for dependencies in GLOBAL_RAW_DEPENDENCIES.values()
                    for dependency in dependencies
                }
            )
        },
    }
    side_rows = {
        side: side_symmetric_features(live, side)
        for side in SIDES
    }
    for side, source in zip(SIDES, rows, strict=True):
        expected = {
            name: math.nan if source["features"][name] is None else float(source["features"][name])
            for name in FEATURE_NAMES
        }
        actual = side_rows[side]
        for name in FEATURE_NAMES:
            if not (
                (math.isnan(expected[name]) and math.isnan(actual[name]))
                or math.isclose(expected[name], actual[name], rel_tol=0.0, abs_tol=1e-12)
            ):
                raise ResidualPromotionError(f"parity fixture feature mismatch: {side}.{name}")
    return {
        "live_feature_row": live,
        "side_rows": side_rows,
        "observed_at_ts": decision_ts,
    }


def _runtime_result(
    *,
    runtime: ResidualPromotionRuntime,
    feature_row: Mapping[str, Any],
    decision_ts: int | None,
    observed_at_ts: int,
    action_values: Mapping[str, float] | None,
    probabilities: Mapping[str, Any] | None,
    selected_action: str,
    fail_closed_reasons: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
        "lineage_id": runtime.lineage_id,
        "candidate_id": runtime.candidate_id,
        "market_id": str(feature_row.get("market_id") or ""),
        "decision_ts": decision_ts,
        "observed_at_ts": observed_at_ts,
        "action_values": dict(action_values) if action_values is not None else {
            "NO_TRADE": 0.0,
            "BUY_UP_HOLD": None,
            "BUY_DOWN_HOLD": None,
        },
        "probabilities": dict(probabilities) if probabilities is not None else None,
        "selected_action": selected_action,
        "model_scored": action_values is not None,
        "fail_closed": bool(fail_closed_reasons),
        "fail_closed_reasons": list(fail_closed_reasons),
        "manifest_sha256": runtime.manifest_sha256,
        "residual_model_sha256": runtime.residual_model_sha256,
        "logit_model_sha256": runtime.logit_model_sha256,
        "zero_capital_read_only_shadow": True,
        "outcomes_accessed": False,
        "settlement_accessed": False,
        "pnl_accessed": False,
        "wallet_signing_allowed": False,
        "polymarket_write_allowed": False,
        "capital_at_risk": False,
        "safety": dict(SAFETY),
    }


def _execution_costs_from_matrix(matrix: np.ndarray) -> np.ndarray:
    ask_index = FEATURE_NAMES.index("selected_ask")
    bid_index = FEATURE_NAMES.index("selected_bid")
    depth_index = FEATURE_NAMES.index("selected_liquidity_depth")
    costs = []
    for row in matrix:
        ask = float(row[ask_index])
        bid = float(row[bid_index])
        depth = float(row[depth_index])
        if not (0.0 < bid <= ask < 1.0 and math.isfinite(depth)):
            raise ValueError("paired executable cost input is invalid")
        slippage = max(0.0001, (ask - bid) / 2.0)
        impact = 0.00005 if depth > 0.0 else 0.001
        costs.append(ask + 0.0002 + slippage + impact)
    return np.asarray(costs, dtype=np.float64)


def _pair_anchored_probabilities(
    anchors: np.ndarray, residuals: np.ndarray
) -> np.ndarray:
    raw = np.clip(
        anchors + residuals,
        PROBABILITY_CLIP_EPSILON,
        1.0 - PROBABILITY_CLIP_EPSILON,
    )
    return _pair_normalize(raw)


def _pair_normalize(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    total = float(np.sum(array))
    if array.shape != (2,) or not np.all(np.isfinite(array)) or total <= 0.0:
        raise ValueError("probability pair is invalid")
    normalized = array / total
    if np.any(normalized <= 0.0) or np.any(normalized >= 1.0):
        raise ValueError("normalized probability pair is invalid")
    return normalized


def _clipped_logit(value: float) -> float:
    probability = min(1.0 - PROBABILITY_CLIP_EPSILON, max(PROBABILITY_CLIP_EPSILON, value))
    return _logit(probability)


def _logit(value: float) -> float:
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("probability is outside (0,1)")
    return math.log(value / (1.0 - value))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _parity_projection(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "action_values": result["action_values"],
        "probabilities": result["probabilities"],
        "selected_action": result["selected_action"],
        "model_scored": result["model_scored"],
        "fail_closed": result["fail_closed"],
    }


def _baseline_model_descriptor(root: Path) -> dict[str, Any]:
    contract = _verified_json(BASELINE_CONTRACT)
    artifact = dict(contract["artifact"])
    _verify_descriptor(artifact, root, "baseline model")
    return {
        "path": artifact["path"],
        "sha256": artifact["sha256"],
        "execution_policy": "HOLD_TO_SETTLEMENT",
        "unit_sizing": True,
        "NO_TRADE_participates_as_zero": True,
    }


def _model_descriptor(
    path: Path, root: Path, spec: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        **_descriptor(path, root),
        "format": "xgboost_ubj",
        "objective": spec["parameters"]["objective"],
        "fixed_num_boost_round": int(spec["fixed_num_boost_round"]),
        "parameters_sha256": canonical_json_sha256(spec["parameters"]),
        "xgboost_version": xgb.__version__,
        "ordered_feature_names_sha256": canonical_json_sha256(list(FEATURE_NAMES)),
    }


def _descriptor(path: Path, root: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ResidualPromotionError(f"artifact is not repository-local: {path}")
    return {
        "path": resolved.relative_to(root).as_posix(),
        "sha256": sha256_file(resolved),
    }


def _verify_descriptor(value: Any, root: Path, name: str) -> Path:
    descriptor = dict(value or {})
    path_value = descriptor.get("path")
    expected = descriptor.get("sha256")
    if not isinstance(path_value, str) or Path(path_value).is_absolute():
        raise ResidualPromotionError(f"{name} path is not repository-relative")
    path = (root / path_value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ResidualPromotionError(f"{name} artifact is missing")
    if sha256_file(path) != _require_sha256(expected):
        raise ResidualPromotionError(f"{name} SHA-256 mismatch")
    return path


def _repo_file(value: Path | str, root: Path, name: str) -> Path:
    candidate = Path(value)
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ResidualPromotionError(f"{name} must resolve inside the repository")
    return path


def _verified_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ResidualPromotionError(f"missing JSON artifact: {path}")
    sidecars = [
        candidate
        for candidate in (
            path.with_suffix(".sha256"),
            path.with_suffix(path.suffix + ".sha256"),
        )
        if candidate.is_file()
    ]
    if len(sidecars) != 1 or sidecars[0].read_text(encoding="utf-8").strip() != sha256_file(path):
        raise ResidualPromotionError(f"JSON artifact sidecar mismatch: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ResidualPromotionError(f"JSON artifact root must be an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_frozen_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.with_suffix(path.suffix + ".sha256").exists():
        raise FileExistsError(f"frozen artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    path.with_suffix(path.suffix + ".sha256").write_text(
        hashlib.sha256(raw).hexdigest() + "\n", encoding="utf-8"
    )


def _require_sha256(value: Any) -> str:
    normalized = str(value or "").lower()
    if not HEX_SHA256.fullmatch(normalized):
        raise ResidualPromotionError("invalid SHA-256")
    return normalized


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "CANDIDATE_ID",
    "CONFIG_DIR",
    "LINEAGE_ID",
    "MAXIMUM_ATTEMPTS",
    "ResidualPromotionError",
    "ResidualPromotionRuntime",
    "TARGET_MARKETS",
    "freeze_prospective_program",
    "load_matched_baseline",
    "load_residual_promotion_runtime",
    "prepare_pretraining_freeze",
    "run_final_fit",
    "score_matched_baseline",
    "validate_final_fit_protocol",
]
