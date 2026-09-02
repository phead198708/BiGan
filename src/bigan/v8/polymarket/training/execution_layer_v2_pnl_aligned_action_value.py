"""Frozen research-only PnL-aligned action-value model for v8 Polymarket."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.contracts import compact_safety_fields
from bigan.v8.polymarket.training.post_freeze_o_replay_aligned_source_ranking import (
    _v8_apply_simulated_order_to_state,
    _v8_execution_guard_config,
    _v8_execution_guard_decision,
    _v8_initial_runtime_state,
)

SCHEMA_PREFIX = "bigan-v8-execution-layer-v2-pnl-aligned-action-value"
PROTOCOL_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-protocol-v1"
MODEL_FAMILY = "deterministic_regularized_action_conditioned_xgboost_regression"
REQUIRED_ACTIONS = (
    "BUY_UP_HOLD_TO_SETTLEMENT",
    "BUY_DOWN_HOLD_TO_SETTLEMENT",
    "BUY_UP_SELL_BEFORE_CLOSE",
    "BUY_DOWN_SELL_BEFORE_CLOSE",
    "NO_TRADE",
)
FORBIDDEN_DECISION_FIELDS = {
    "resolved_outcome",
    "settlement_pnl",
    "settlement_return",
    "settlement_payout",
    "oracle_action",
    "future_return",
    "total_net_return",
    "total_net_pnl_per_notional",
    "target_net_return_after_cost",
    "selected_side_win_target",
    "evaluation_target_net_pnl_per_contract_by_action",
    "evaluation_target_net_return_after_cost_by_action",
    "evaluation_target_pnl_components_by_action",
}


@dataclass(frozen=True, slots=True)
class PnLAlignedActionValueFitConfig:
    """Inputs for one deterministic historical-fit research artifact."""

    run_id: str
    output_dir: Path | str
    protocol_path: Path | str
    expected_protocol_sha256: str
    historical_corpus_manifest_path: Path | str

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if len(self.expected_protocol_sha256) != 64:
            raise ValueError("expected_protocol_sha256 must be a SHA-256 digest")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "protocol_path", Path(self.protocol_path))
        object.__setattr__(
            self,
            "historical_corpus_manifest_path",
            Path(self.historical_corpus_manifest_path),
        )


@dataclass(frozen=True, slots=True)
class PnLAlignedFutureCollectionFreezeConfig:
    """Inputs for freezing a future, market-disjoint collection window."""

    run_id: str
    output_dir: Path | str
    model_dir: Path | str
    git_commit: str
    expected_round_count: int = 30

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if len(self.git_commit) != 40 or any(
            char not in "0123456789abcdef" for char in self.git_commit.lower()
        ):
            raise ValueError("git_commit must be a 40-character hex digest")
        if self.expected_round_count < 30:
            raise ValueError("expected_round_count must be at least 30")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "model_dir", Path(self.model_dir))


def fit_frozen_pnl_aligned_action_value_model(
    config: PnLAlignedActionValueFitConfig,
) -> dict[str, Any]:
    """Fit a research-only model without using validation or holdout evidence."""

    protocol_path = config.protocol_path.resolve()
    protocol_sha256 = _sha256_file(protocol_path)
    if protocol_sha256 != config.expected_protocol_sha256:
        raise ValueError("protocol SHA-256 mismatch")
    protocol = _load_json(protocol_path)
    validate_pnl_aligned_action_value_protocol(protocol)

    corpus_manifest_path = config.historical_corpus_manifest_path.resolve()
    corpus_manifest = _load_json(corpus_manifest_path)
    rows_descriptor = dict(corpus_manifest.get("development_rows") or {})
    rows_path = Path(str(rows_descriptor.get("path") or "")).resolve()
    if not rows_path.is_file():
        raise ValueError("historical development rows are missing")
    if rows_descriptor.get("sha256") != _sha256_file(rows_path):
        raise ValueError("historical development rows descriptor hash mismatch")
    source_rows = _load_jsonl(rows_path)
    action_rows, audit = build_pnl_aligned_action_conditioned_rows(
        source_rows,
        protocol=protocol,
        require_targets=True,
    )
    if audit["blocking_reason_codes"]:
        raise ValueError(
            "historical action-conditioned row audit failed: "
            + ", ".join(audit["blocking_reason_codes"])
        )

    run_dir = config.output_dir / config.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    action_rows_path = run_dir / "pnl_aligned_action_conditioned_fit_rows.jsonl"
    _write_jsonl(action_rows_path, action_rows)
    audit_path = run_dir / "pnl_aligned_action_value_feature_leakage_audit.json"
    _write_json(audit_path, audit)

    feature_columns = tuple(str(value) for value in protocol["feature_columns"])
    matrix = np.asarray(
        [
            [float(row["decision_time_features"][name]) for name in feature_columns]
            for row in action_rows
        ],
        dtype=np.float64,
    )
    targets = np.asarray(
        [float(row["target_net_pnl_per_contract"]) for row in action_rows],
        dtype=np.float64,
    )
    dtrain = xgb.DMatrix(matrix, label=targets, feature_names=list(feature_columns))
    xgb_config = dict(protocol["xgboost_config"])
    num_boost_round = int(xgb_config.pop("num_boost_round"))
    booster = xgb.train(
        params=xgb_config,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
    )
    model_path = run_dir / "pnl_aligned_action_value_model.xgb.json"
    booster.save_model(model_path)
    predictions = [float(value) for value in booster.predict(dtrain)]
    training_metrics = _regression_metrics(targets.tolist(), predictions)

    dataset_contract = {
        "source_corpus_manifest_sha256": _sha256_file(corpus_manifest_path),
        "source_development_rows_sha256": _sha256_file(rows_path),
        "protocol_sha256": protocol_sha256,
        "feature_columns": list(feature_columns),
        "target_field": protocol["primary_target"],
        "action_row_count": len(action_rows),
        "decision_count": len(
            {(row["market_id"], row["decision_ts"]) for row in action_rows}
        ),
        "market_count": len({row["market_id"] for row in action_rows}),
        "source_run_ids": sorted({row["source_run_id"] for row in action_rows}),
        "historical_fit_only": True,
    }
    dataset_hash = canonical_json_sha256(
        {
            "contract": dataset_contract,
            "row_hashes": [row["action_row_sha256"] for row in action_rows],
        }
    )
    model_contract = {
        "candidate_name": protocol["candidate_name"],
        "model_family": MODEL_FAMILY,
        "model_sha256": _sha256_file(model_path),
        "protocol_sha256": protocol_sha256,
        "historical_fit_dataset_hash": dataset_hash,
        "feature_schema_hash": canonical_json_sha256(list(feature_columns)),
        "action_schema_hash": canonical_json_sha256(list(REQUIRED_ACTIONS)),
        "xgboost_config_hash": canonical_json_sha256(protocol["xgboost_config"]),
        "historical_fit_only": True,
        "uses_validation_labels_for_tuning": False,
        "uses_future_holdout_labels_for_fitting": False,
        "current_oof_pnl_used_for_hyperparameter_selection": False,
        "future_unseen_evaluation_required": True,
    }
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-fit-report-v1",
        "run_id": config.run_id,
        "status": "RESEARCH_MODEL_FIT_AND_FROZEN_FOR_FUTURE_DIAGNOSTIC",
        "protocol": _descriptor(protocol_path),
        "historical_corpus_manifest": _descriptor(corpus_manifest_path),
        "historical_development_rows": _descriptor(rows_path),
        "action_conditioned_fit_rows": _descriptor(action_rows_path),
        "feature_leakage_audit": _descriptor(audit_path),
        "model": _descriptor(model_path),
        "dataset_contract": dataset_contract,
        "historical_fit_dataset_hash": dataset_hash,
        "model_contract": model_contract,
        "training_only_metrics": training_metrics,
        "training_metric_used_for_model_selection": False,
        "validation_evaluation_attempted": False,
        "future_unseen_evaluation_attempted": False,
        "research_artifact_frozen": True,
        "research_artifact_frozen_for_future_evaluation": True,
        "production_model_frozen": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    report_path = run_dir / "pnl_aligned_action_value_fit_report.json"
    _write_json(report_path, report)
    markdown_path = run_dir / "pnl_aligned_action_value_fit_report.md"
    _write_text(markdown_path, _fit_markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-fit-manifest-v1",
        "run_id": config.run_id,
        "protocol": _descriptor(protocol_path),
        "historical_corpus_manifest": _descriptor(corpus_manifest_path),
        "action_conditioned_fit_rows": _descriptor(action_rows_path),
        "feature_leakage_audit": _descriptor(audit_path),
        "model": _descriptor(model_path),
        "fit_report": _descriptor(report_path),
        "fit_markdown": _descriptor(markdown_path),
        "historical_fit_dataset_hash": dataset_hash,
        "model_contract": model_contract,
        "research_artifact_frozen": True,
        "future_unseen_evaluation_required": True,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    manifest_path = run_dir / "pnl_aligned_action_value_fit_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "model_path": model_path,
        "report_path": report_path,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "report": report,
    }


def freeze_pnl_aligned_future_collection(
    config: PnLAlignedFutureCollectionFreezeConfig,
) -> dict[str, Any]:
    """Freeze model and historical lineage before future public-data collection."""

    model_dir = config.model_dir.resolve()
    fit_manifest_path = model_dir / "pnl_aligned_action_value_fit_manifest.json"
    fit_manifest = _load_json(fit_manifest_path)
    model_descriptor = dict(fit_manifest.get("model") or {})
    protocol_descriptor = dict(fit_manifest.get("protocol") or {})
    historical_manifest_descriptor = dict(
        fit_manifest.get("historical_corpus_manifest") or {}
    )
    for name, descriptor in (
        ("model", model_descriptor),
        ("protocol", protocol_descriptor),
        ("historical_corpus_manifest", historical_manifest_descriptor),
    ):
        path = Path(str(descriptor.get("path") or ""))
        if not path.is_file() or descriptor.get("sha256") != _sha256_file(path):
            raise ValueError(f"frozen {name} descriptor hash mismatch")
    protocol = _load_json(Path(protocol_descriptor["path"]))
    validate_pnl_aligned_action_value_protocol(protocol)
    model_contract = dict(fit_manifest.get("model_contract") or {})
    expected_model_contract = {
        "model_sha256": model_descriptor.get("sha256"),
        "protocol_sha256": protocol_descriptor.get("sha256"),
        "historical_fit_dataset_hash": fit_manifest.get(
            "historical_fit_dataset_hash"
        ),
        "historical_fit_only": True,
        "uses_validation_labels_for_tuning": False,
        "uses_future_holdout_labels_for_fitting": False,
        "current_oof_pnl_used_for_hyperparameter_selection": False,
        "future_unseen_evaluation_required": True,
    }
    model_contract_mismatches = sorted(
        key
        for key, expected in expected_model_contract.items()
        if model_contract.get(key) != expected
    )
    if model_contract_mismatches:
        raise ValueError(
            "frozen model contract mismatch: " + ", ".join(model_contract_mismatches)
        )
    if not (
        fit_manifest.get("research_artifact_frozen") is True
        and fit_manifest.get("future_unseen_evaluation_required") is True
        and fit_manifest.get("source_model_candidate_eligible") is False
        and fit_manifest.get("freeze_ready") is False
        and fit_manifest.get("promotion_evidence_eligible") is False
        and fit_manifest.get("v8_execution_handoff_allowed") is False
    ):
        raise ValueError("fit manifest safety or future-evaluation contract mismatch")
    historical_manifest = _load_json(Path(historical_manifest_descriptor["path"]))
    historical_rows_descriptor = dict(
        historical_manifest.get("development_rows") or {}
    )
    historical_rows_path = Path(str(historical_rows_descriptor.get("path") or ""))
    if (
        not historical_rows_path.is_file()
        or historical_rows_descriptor.get("sha256")
        != _sha256_file(historical_rows_path)
    ):
        raise ValueError("historical rows descriptor hash mismatch")
    historical_rows = _load_jsonl(historical_rows_path)
    if not historical_rows:
        raise ValueError("historical rows must not be empty")
    prior_market_ids = sorted({str(row["market_id"]) for row in historical_rows})
    max_prior_decision_ts = max(int(row["decision_ts"]) for row in historical_rows)
    freeze_created_ts = int(time.time() * 1000)
    execution_guard_config = _v8_execution_guard_config()
    output_dir = config.output_dir / config.run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-future-collection-freeze-v1",
        "run_id": config.run_id,
        "freeze_created_ts": freeze_created_ts,
        "git_commit": config.git_commit.lower(),
        "fit_manifest": _descriptor(fit_manifest_path),
        "fit_manifest_sha256": _sha256_file(fit_manifest_path),
        "model": model_descriptor,
        "model_contract": model_contract,
        "protocol": protocol_descriptor,
        "frozen_execution_contract": protocol["frozen_execution_contract"],
        "frozen_execution_contract_sha256": canonical_json_sha256(
            protocol["frozen_execution_contract"]
        ),
        "execution_guard_config": execution_guard_config,
        "execution_guard_config_sha256": canonical_json_sha256(
            execution_guard_config
        ),
        "historical_corpus_manifest": historical_manifest_descriptor,
        "historical_development_rows": _descriptor(historical_rows_path),
        "historical_fit_dataset_hash": fit_manifest[
            "historical_fit_dataset_hash"
        ],
        "prior_market_count": len(prior_market_ids),
        "prior_market_ids_sha256": canonical_json_sha256(prior_market_ids),
        "max_prior_decision_ts": max_prior_decision_ts,
        "minimum_future_window_start_ts": max(
            max_prior_decision_ts + 1,
            freeze_created_ts + 1,
        ),
        "expected_round_count": config.expected_round_count,
        "future_window_must_be_strictly_later": True,
        "future_market_ids_must_be_disjoint": True,
        "future_collection_outcome_blind": True,
        "model_config_or_threshold_mutation_after_freeze_allowed": False,
        "future_evidence_gates": protocol["future_evidence_gates"],
        "collection_started": False,
        "future_evaluation_started": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    manifest["collection_freeze_id"] = canonical_json_sha256(manifest)
    manifest_path = output_dir / "pnl_aligned_future_collection_freeze_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "output_dir": output_dir,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "manifest": manifest,
    }


def validate_pnl_aligned_action_value_protocol(protocol: dict[str, Any]) -> None:
    """Fail closed on semantic drift in the frozen model protocol."""

    checks = {
        "schema_version": protocol.get("schema_version") == PROTOCOL_SCHEMA_VERSION,
        "frozen": protocol.get("frozen") is True,
        "decision_time_safe": protocol.get("decision_time_safe") is True,
        "historical_fit_only": protocol.get("historical_fit_only") is True,
        "artifact_frozen_before_future_evaluation": (
            protocol.get("artifact_frozen_before_future_evaluation") is True
        ),
        "no_current_oof_pnl_tuning": (
            protocol.get("uses_current_oof_pnl_for_hyperparameter_selection")
            is False
        ),
        "no_validation_label_tuning": (
            protocol.get("uses_validation_labels_for_tuning") is False
        ),
        "no_holdout_fit": protocol.get("uses_future_holdout_labels_for_fitting")
        is False,
        "target": protocol.get("primary_target")
        == "total_net_pnl_per_notional",
        "actions": tuple(protocol.get("actions") or ()) == REQUIRED_ACTIONS,
        "market_probability_not_direct_fair_value": (
            protocol.get("market_implied_probability_used_as_direct_fair_value_ev")
            is False
        ),
        "market_probability_conditioning_only": (
            protocol.get("market_implied_probability_used_as_conditioning_feature")
            is True
        ),
        "market_probability_not_direction_vote": (
            protocol.get("market_implied_probability_used_as_regime_direction_vote")
            is False
        ),
    }
    feature_columns = list(protocol.get("feature_columns") or [])
    checks["feature_columns_unique"] = bool(feature_columns) and len(feature_columns) == len(
        set(feature_columns)
    )
    groups = dict(protocol.get("independent_feature_groups") or {})
    grouped_columns = [name for values in groups.values() for name in values]
    checks["feature_groups_partition_columns"] = sorted(grouped_columns) == sorted(
        feature_columns
    ) and len(grouped_columns) == len(set(grouped_columns))
    xgb_config = dict(protocol.get("xgboost_config") or {})
    checks["fixed_xgboost_objective"] = xgb_config.get("objective") == "reg:squarederror"
    checks["fixed_xgboost_seed"] = isinstance(xgb_config.get("seed"), int)
    checks["single_threaded_determinism"] = xgb_config.get("nthread") == 1
    future = dict(protocol.get("future_evidence_gates") or {})
    checks["future_support_predeclared"] = (
        int(future.get("minimum_unique_market_count") or 0) >= 30
        and int(future.get("minimum_accepted_bet_count") or 0) >= 30
        and int(future.get("minimum_accepted_bet_count_per_side") or 0) >= 10
    )
    safety = dict(protocol.get("safety") or {})
    checks["safety_fail_closed"] = (
        safety.get("paper_only") is True
        and safety.get("capital_at_risk") is False
        and safety.get("polymarket_write_enabled") is False
        and safety.get("wallet_signing_enabled") is False
        and safety.get("v8_execution_handoff_allowed") is False
        and safety.get("source_model_candidate_eligible") is False
        and safety.get("freeze_ready") is False
        and safety.get("promotion_evidence_eligible") is False
        and safety.get("#134_resume_allowed") is False
        and safety.get("#146_start_allowed") is False
    )
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("invalid frozen PnL-aligned protocol: " + ", ".join(failed))


def build_pnl_aligned_action_conditioned_rows(
    source_rows: list[dict[str, Any]],
    *,
    protocol: dict[str, Any],
    require_targets: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Expand decision rows into a complete action grid with causal inputs."""

    validate_pnl_aligned_action_value_protocol(protocol)
    action_rows: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    causality_violations: list[dict[str, Any]] = []
    forbidden_violations: list[dict[str, Any]] = []
    incomplete_groups: list[dict[str, Any]] = []
    for source in sorted(
        source_rows,
        key=lambda row: (
            int(row.get("decision_ts") or 0),
            str(row.get("market_id") or ""),
            str(row.get("row_identity") or ""),
        ),
    ):
        decision_ts = int(source.get("decision_ts") or 0)
        market_close_ts = int(source.get("market_close_ts") or 0)
        handoff = dict(source.get("execution_handoff_context") or {})
        max_input_ts = max(
            int(source.get("max_input_ts") or 0),
            int(handoff.get("decision_time_feature_max_input_ts") or 0),
        )
        if max_input_ts > decision_ts:
            reasons["decision_time_feature_causality_violation"] += 1
            causality_violations.append(
                {
                    "market_id": source.get("market_id"),
                    "decision_ts": decision_ts,
                    "max_input_ts": max_input_ts,
                }
            )
            continue
        if market_close_ts <= decision_ts:
            reasons["invalid_decision_time_market_schedule"] += 1
            continue
        decision_inputs: dict[str, Any] = {
            "decision_time_features": dict(source.get("decision_time_features") or {}),
            "execution_handoff_context": handoff,
        }
        forbidden_scope = decision_inputs if require_targets else source
        forbidden = sorted(_find_forbidden_fields(forbidden_scope))
        if forbidden:
            reasons["forbidden_decision_field_present"] += 1
            forbidden_violations.append(
                {
                    "market_id": source.get("market_id"),
                    "decision_ts": decision_ts,
                    "forbidden_fields": forbidden,
                }
            )
            continue
        ranking = list(handoff.get("full_5_action_ranking") or [])
        ranking_by_action = {
            str(row.get("selected_action") or ""): row for row in ranking
        }
        if set(ranking_by_action) != set(REQUIRED_ACTIONS):
            reasons["incomplete_5_action_ranking"] += 1
            incomplete_groups.append(
                {
                    "market_id": source.get("market_id"),
                    "decision_ts": decision_ts,
                    "available_actions": sorted(ranking_by_action),
                }
            )
            continue
        targets = dict(
            source.get("evaluation_target_net_pnl_per_contract_by_action") or {}
        )
        if require_targets and set(targets) != set(REQUIRED_ACTIONS):
            reasons["incomplete_5_action_target_grid"] += 1
            incomplete_groups.append(
                {
                    "market_id": source.get("market_id"),
                    "decision_ts": decision_ts,
                    "available_target_actions": sorted(targets),
                }
            )
            continue
        top_score = max(float(row["corrected_model_score"]) for row in ranking)
        for action in REQUIRED_ACTIONS:
            ranking_row = ranking_by_action[action]
            features = _action_features(
                source=source,
                handoff=handoff,
                ranking_row=ranking_row,
                action=action,
                top_score=top_score,
            )
            missing = [
                name for name in protocol["feature_columns"] if name not in features
            ]
            if missing:
                raise ValueError(
                    "action-conditioned feature construction is incomplete: "
                    + ", ".join(missing)
                )
            if any(not math.isfinite(float(value)) for value in features.values()):
                raise ValueError("action-conditioned features must be finite")
            action_row = {
                "market_id": str(source["market_id"]),
                "decision_ts": decision_ts,
                "market_close_ts": market_close_ts,
                "max_input_ts": max_input_ts,
                "source_run_id": str(source["source_run_id"]),
                "source_row_identity": str(source["row_identity"]),
                "action": action,
                "side": _side(action),
                "action_family": _family(action),
                "decision_time_features": {
                    name: float(features[name]) for name in protocol["feature_columns"]
                },
                "execution_handoff_context": _action_execution_handoff(
                    source=source,
                    handoff=handoff,
                    ranking_row=ranking_row,
                    action=action,
                    max_input_ts=max_input_ts,
                ),
                "target_net_pnl_per_contract": (
                    float(targets[action]) if require_targets else None
                ),
                "target_used_as_decision_input": False,
                "outcome_aware_historical_fit_target": require_targets,
                "paper_only": True,
                "capital_at_risk": False,
            }
            action_row["action_row_sha256"] = canonical_json_sha256(action_row)
            action_rows.append(action_row)
    expected_count = len(source_rows) * len(REQUIRED_ACTIONS)
    if len(action_rows) != expected_count:
        reasons["action_row_count_not_complete"] += 1
    audit = {
        "schema_version": f"{SCHEMA_PREFIX}-feature-leakage-audit-v1",
        "source_decision_count": len(source_rows),
        "action_row_count": len(action_rows),
        "expected_action_row_count": expected_count,
        "complete_5_action_grid": len(action_rows) == expected_count,
        "feature_max_input_ts_violation_count": len(causality_violations),
        "feature_max_input_ts_violations": causality_violations,
        "forbidden_decision_field_violation_count": len(forbidden_violations),
        "forbidden_decision_field_violations": forbidden_violations,
        "incomplete_decision_groups": incomplete_groups,
        "target_used_as_decision_input": False,
        "settlement_outcome_used_as_input": False,
        "pnl_target_used_for_historical_fit_only": require_targets,
        "market_implied_probability_used_as_direct_fair_value_ev": False,
        "market_implied_probability_used_as_conditioning_feature": True,
        "market_implied_probability_used_as_regime_direction_vote": False,
        "blocking_reason_codes": sorted(reasons),
        "reason_distribution": dict(sorted(reasons.items())),
        "passed": not reasons,
        **compact_safety_fields(),
    }
    return action_rows, audit


def predict_frozen_pnl_aligned_action_values(
    *,
    model_dir: Path | str,
    decision_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run outcome-blind five-action inference from a frozen research artifact."""

    model_dir = Path(model_dir).resolve()
    manifest_path = model_dir / "pnl_aligned_action_value_fit_manifest.json"
    manifest = _load_json(manifest_path)
    protocol_descriptor = dict(manifest["protocol"])
    protocol_path = Path(protocol_descriptor["path"])
    if _sha256_file(protocol_path) != protocol_descriptor["sha256"]:
        raise ValueError("frozen protocol descriptor hash mismatch")
    protocol = _load_json(protocol_path)
    validate_pnl_aligned_action_value_protocol(protocol)
    model_descriptor = dict(manifest["model"])
    model_path = Path(model_descriptor["path"])
    if _sha256_file(model_path) != model_descriptor["sha256"]:
        raise ValueError("frozen model descriptor hash mismatch")
    action_rows, audit = build_pnl_aligned_action_conditioned_rows(
        decision_rows,
        protocol=protocol,
        require_targets=False,
    )
    if audit["blocking_reason_codes"]:
        return [], {
            "status": "BLOCKED_FAIL_CLOSED",
            "prediction_attempted": False,
            "blocking_reason_codes": audit["blocking_reason_codes"],
            "feature_leakage_audit": audit,
            "source_model_candidate_eligible": False,
            "promotion_evidence_eligible": False,
            **compact_safety_fields(),
        }
    features = list(protocol["feature_columns"])
    matrix = np.asarray(
        [
            [float(row["decision_time_features"][name]) for name in features]
            for row in action_rows
        ],
        dtype=np.float64,
    )
    booster = xgb.Booster()
    booster.load_model(model_path)
    values = booster.predict(xgb.DMatrix(matrix, feature_names=features))
    predictions = []
    for row, value in zip(action_rows, values, strict=True):
        prediction = {
            key: row[key]
            for key in (
                "market_id",
                "decision_ts",
                "market_close_ts",
                "max_input_ts",
                "source_run_id",
                "source_row_identity",
                "action",
                "side",
                "action_family",
            )
        }
        prediction.update(
            {
                "predicted_net_pnl_per_contract": float(value),
                "ranking_score_source": "model_predicted_net_pnl_per_contract",
                "execution_handoff_context": row["execution_handoff_context"],
                "target_used_as_decision_input": False,
                "source_o_score_mutated": False,
                "source_ranking_mutated": False,
                "paper_only": True,
                "capital_at_risk": False,
            }
        )
        prediction["prediction_sha256"] = canonical_json_sha256(prediction)
        predictions.append(prediction)
    by_decision: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_decision[(row["market_id"], row["decision_ts"])].append(row)
    complete_groups = all(
        {row["action"] for row in rows} == set(REQUIRED_ACTIONS)
        for rows in by_decision.values()
    )
    report = {
        "status": "OUTCOME_BLIND_PREDICTION_COMPLETE",
        "prediction_attempted": True,
        "prediction_count": len(predictions),
        "decision_count": len(by_decision),
        "complete_5_action_prediction_grid": complete_groups,
        "feature_leakage_audit": audit,
        "future_unseen_evaluation_required": True,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    return predictions, report


def run_pnl_aligned_action_value_outcome_blind_shadow(
    *,
    model_dir: Path | str,
    decision_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rank actions, then apply the unchanged execution guard without outcomes."""

    model_dir = Path(model_dir).resolve()
    predictions, prediction_report = predict_frozen_pnl_aligned_action_values(
        model_dir=model_dir,
        decision_rows=decision_rows,
    )
    if prediction_report["prediction_attempted"] is not True:
        return [], {
            "status": "BLOCKED_FAIL_CLOSED_BEFORE_EXECUTION_GUARD",
            "prediction_report": prediction_report,
            "execution_guard_attempted": False,
            "source_model_candidate_eligible": False,
            "promotion_evidence_eligible": False,
            "v8_execution_handoff_allowed": False,
            **compact_safety_fields(),
        }
    manifest = _load_json(model_dir / "pnl_aligned_action_value_fit_manifest.json")
    protocol = _load_json(Path(manifest["protocol"]["path"]))
    threshold = float(
        protocol["frozen_execution_contract"]["entry_edge_threshold"]
    )
    guard_config = _v8_execution_guard_config()
    state = _v8_initial_runtime_state(guard_config)
    market_close_by_open_position: dict[str, int] = {}
    by_decision: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_decision[(str(row["market_id"]), int(row["decision_ts"]))].append(row)
    replay_rows: list[dict[str, Any]] = []
    for index, ((market_id, decision_ts), action_predictions) in enumerate(
        sorted(by_decision.items(), key=lambda item: (item[0][1], item[0][0])),
        start=1,
    ):
        _release_closed_shadow_positions(
            state=state,
            market_close_by_open_position=market_close_by_open_position,
            decision_ts=decision_ts,
        )
        ranked = sorted(
            action_predictions,
            key=lambda row: (-float(row["predicted_net_pnl_per_contract"]), row["action"]),
        )
        selected = ranked[0]
        selected_action = str(selected["action"])
        predicted_value = float(selected["predicted_net_pnl_per_contract"])
        selected_handoff = dict(selected["execution_handoff_context"])
        selected_microstructure = dict(
            selected_handoff.get("microstructure_snapshot") or {}
        )
        signal_passed = selected_action != "NO_TRADE" and predicted_value >= threshold
        blocking_reason_codes: list[str] = []
        guard_row: dict[str, Any] | None = None
        if selected_action == "NO_TRADE":
            blocking_reason_codes.append("pnl_aligned_model_selected_no_trade")
        elif predicted_value < threshold:
            blocking_reason_codes.append(
                "predicted_net_pnl_below_frozen_entry_threshold"
            )
        else:
            guard_row = _v8_execution_guard_decision(
                dict(selected["execution_handoff_context"]),
                guard_config=guard_config,
                runtime_state=state,
                runtime_mode="simulated_runtime_state",
            )
            blocking_reason_codes.extend(
                guard_row["execution_blocking_reason_codes"]
            )
        guard_allowed = bool(guard_row and guard_row["order_allowed"])
        simulated_order_id = None
        if guard_allowed:
            simulated_order_id = f"pnl-aligned-shadow-bet-{index:06d}"
            _v8_apply_simulated_order_to_state(
                state=state,
                decision=guard_row,
                simulated_order_id=simulated_order_id,
            )
            market_close_by_open_position[market_id] = int(
                selected["market_close_ts"]
            )
        replay_row = {
            "decision_index": index,
            "source_row_identity": str(selected["source_row_identity"]),
            "market_id": market_id,
            "decision_ts": decision_ts,
            "market_close_ts": int(selected["market_close_ts"]),
            "selected_action": selected_action,
            "selected_side": selected["side"],
            "selected_action_family": selected["action_family"],
            "predicted_net_pnl_per_contract": predicted_value,
            "selected_execution_price": float(
                selected_microstructure.get("entry_ask") or 0.0
            ),
            "frozen_entry_edge_threshold": threshold,
            "model_signal_passed": signal_passed,
            "execution_guard_evaluated": guard_row is not None,
            "execution_guard_order_allowed": guard_allowed,
            "execution_guarded_action": (
                guard_row.get("execution_guarded_action") if guard_row else None
            ),
            "execution_guarded_side": (
                guard_row.get("execution_guarded_side") if guard_row else None
            ),
            "proposed_order_size": (
                float(guard_row["proposed_order_size"]) if guard_allowed else 0.0
            ),
            "simulated_order_id": simulated_order_id,
            "execution_blocking_reason_codes": sorted(set(blocking_reason_codes)),
            "execution_guard_reason_codes": (
                list(guard_row["execution_guard_reason_codes"])
                if guard_row
                else []
            ),
            "full_5_action_model_ranking": [
                {
                    "rank": rank,
                    "action": row["action"],
                    "side": row["side"],
                    "action_family": row["action_family"],
                    "predicted_net_pnl_per_contract": row[
                        "predicted_net_pnl_per_contract"
                    ],
                }
                for rank, row in enumerate(ranked, start=1)
            ],
            "outcome_fields_used": False,
            "realized_pnl_used": False,
            "source_o_score_mutated": False,
            "source_ranking_mutated": False,
            "paper_only": True,
            "capital_at_risk": False,
            "polymarket_write_enabled": False,
            "wallet_signing_enabled": False,
        }
        replay_row["shadow_replay_row_sha256"] = canonical_json_sha256(replay_row)
        replay_rows.append(replay_row)
    blockers = Counter(
        reason
        for row in replay_rows
        for reason in row["execution_blocking_reason_codes"]
    )
    accepted = [
        row for row in replay_rows if row["execution_guard_order_allowed"] is True
    ]
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-outcome-blind-shadow-report-v1",
        "status": "OUTCOME_BLIND_SHADOW_EXECUTION_COMPLETE",
        "prediction_report": prediction_report,
        "model_dir": str(model_dir),
        "model_sha256": manifest["model"]["sha256"],
        "protocol_sha256": manifest["protocol"]["sha256"],
        "execution_guard_config_sha256": canonical_json_sha256(guard_config),
        "entry_edge_threshold": threshold,
        "decision_count": len(replay_rows),
        "model_trade_candidate_count": sum(
            row["model_signal_passed"] for row in replay_rows
        ),
        "execution_guard_evaluated_count": sum(
            row["execution_guard_evaluated"] for row in replay_rows
        ),
        "executable_shadow_bet_count": len(accepted),
        "blocked_decision_count": len(replay_rows) - len(accepted),
        "blocking_reason_distribution": dict(sorted(blockers.items())),
        "selected_action_distribution": dict(
            sorted(Counter(row["selected_action"] for row in replay_rows).items())
        ),
        "accepted_side_distribution": dict(
            sorted(Counter(row["execution_guarded_side"] for row in accepted).items())
        ),
        "accepted_action_distribution": dict(
            sorted(Counter(row["execution_guarded_action"] for row in accepted).items())
        ),
        "outcome_fields_used": False,
        "settlement_pnl_used": False,
        "source_o_score_mutated": False,
        "source_ranking_mutated": False,
        "future_unseen_outcome_reconciliation_required": True,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    return replay_rows, report


def _release_closed_shadow_positions(
    *,
    state: dict[str, Any],
    market_close_by_open_position: dict[str, int],
    decision_ts: int,
) -> None:
    closed_markets = sorted(
        market_id
        for market_id, close_ts in market_close_by_open_position.items()
        if close_ts <= decision_ts
    )
    for market_id in closed_markets:
        position = state["open_position_by_market_id"].pop(market_id, None)
        market_close_by_open_position.pop(market_id, None)
        if not isinstance(position, dict):
            continue
        side = str(position.get("side") or "NONE")
        size = float(position.get("notional") or 0.0)
        state["open_position_by_market_side"].pop(f"{market_id}|{side}", None)
        state["current_market_exposure_by_market_id"].pop(market_id, None)
        state["current_side_exposure_by_side"][side] = max(
            0.0,
            float(state["current_side_exposure_by_side"].get(side) or 0.0) - size,
        )
        state["current_total_exposure"] = max(
            0.0,
            float(state.get("current_total_exposure") or 0.0) - size,
        )


def _action_features(
    *,
    source: dict[str, Any],
    handoff: dict[str, Any],
    ranking_row: dict[str, Any],
    action: str,
    top_score: float,
) -> dict[str, float]:
    source_features = dict(source["decision_time_features"])
    micro = dict(ranking_row.get("microstructure_snapshot") or {})
    side = _side(action)
    side_sign = 1.0 if side == "UP" else -1.0 if side == "DOWN" else 0.0
    p_up = float(handoff.get("p_up") or 0.5)
    p_down = float(handoff.get("p_down") or 0.5)
    probability = p_up if side == "UP" else p_down if side == "DOWN" else 0.0
    execution_price = float(micro.get("entry_ask") or 0.0)
    reference_distance = float(
        source_features.get("reference_price_to_beat_distance_at_decision") or 0.0
    )
    momentum_60s = float(source_features.get("chainlink_momentum_60s") or 0.0)
    score = float(ranking_row.get("corrected_model_score") or 0.0)
    return {
        "canonical_o_action_score": score,
        "canonical_o_raw_score": float(ranking_row.get("raw_model_score") or 0.0),
        "canonical_action_rank": float(
            ranking_row.get("canonical_rank") or ranking_row.get("rank") or 0.0
        ),
        "canonical_score_gap_from_best": top_score - score,
        "selected_side_probability": probability,
        "execution_price": execution_price,
        "selected_side_probability_minus_execution_price": (
            probability - execution_price
        ),
        "btc_anchor_direction_signal": side_sign
        * ((reference_distance + momentum_60s) / 2.0),
        "chainlink_realized_volatility_120s": float(
            source_features.get("chainlink_realized_volatility_120s") or 0.0
        ),
        "spread_bps": float(micro.get("spread_bps") or 0.0),
        "queue_fill_proxy": float(micro.get("queue_fill_proxy") or 0.0),
        "book_staleness_ms": float(micro.get("book_staleness_ms") or 0.0),
        "time_to_close_seconds": float(micro.get("time_to_close_seconds") or 0.0),
        "cumulative_market_exposure_before_entry": float(
            source_features.get("cumulative_market_exposure_before_entry") or 0.0
        ),
        "same_side_reentry": float(source_features.get("same_side_reentry") or 0.0),
        "side_flip": float(source_features.get("side_flip") or 0.0),
        "action_buy_up": float(side == "UP"),
        "action_buy_down": float(side == "DOWN"),
        "action_hold_to_settlement": float(_family(action) == "HOLD_TO_SETTLEMENT"),
        "action_sell_before_close": float(_family(action) == "SELL_BEFORE_CLOSE"),
        "action_no_trade": float(action == "NO_TRADE"),
    }


def _action_execution_handoff(
    *,
    source: dict[str, Any],
    handoff: dict[str, Any],
    ranking_row: dict[str, Any],
    action: str,
    max_input_ts: int,
) -> dict[str, Any]:
    context = json.loads(json.dumps(handoff, sort_keys=True))
    side = _side(action)
    p_up = float(handoff.get("p_up") or 0.5)
    context.update(
        {
            "market_id": str(source["market_id"]),
            "decision_ts": int(source["decision_ts"]),
            "selected_action": action,
            "selected_side": side,
            "selected_action_family": _family(action),
            "corrected_model_score": float(
                ranking_row.get("corrected_model_score") or 0.0
            ),
            "raw_model_score": float(ranking_row.get("raw_model_score") or 0.0),
            "high_score_flag": bool(ranking_row.get("high_score_flag")),
            "p_up_action_disagreement": bool(
                (side == "UP" and p_up < 0.5)
                or (side == "DOWN" and p_up >= 0.5)
            ),
            "microstructure_snapshot": dict(
                ranking_row.get("microstructure_snapshot") or {}
            ),
            "decision_time_feature_max_input_ts": max_input_ts,
        }
    )
    return context


def _find_forbidden_fields(payload: Any, prefix: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key) in FORBIDDEN_DECISION_FIELDS:
                found.add(path)
            found.update(_find_forbidden_fields(value, path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.update(_find_forbidden_fields(value, f"{prefix}[{index}]"))
    return found


def _regression_metrics(targets: list[float], predictions: list[float]) -> dict[str, Any]:
    errors = [predicted - target for target, predicted in zip(targets, predictions, strict=True)]
    return {
        "scope": "historical_fit_rows_training_diagnostic_only",
        "row_count": len(targets),
        "mae": sum(abs(value) for value in errors) / len(errors),
        "mse": sum(value * value for value in errors) / len(errors),
        "target_mean": sum(targets) / len(targets),
        "prediction_mean": sum(predictions) / len(predictions),
        "used_for_model_selection": False,
        "promotion_evidence": False,
    }


def _side(action: str) -> str:
    if "_UP_" in action:
        return "UP"
    if "_DOWN_" in action:
        return "DOWN"
    return "NONE"


def _family(action: str) -> str:
    if action.endswith("HOLD_TO_SETTLEMENT"):
        return "HOLD_TO_SETTLEMENT"
    if action.endswith("SELL_BEFORE_CLOSE"):
        return "SELL_BEFORE_CLOSE"
    return "NO_TRADE"


def _fit_markdown(report: dict[str, Any]) -> str:
    metrics = report["training_only_metrics"]
    return "\n".join(
        [
            "# PnL-Aligned Action-Value Research Fit",
            "",
            f"- status: `{report['status']}`",
            f"- historical fit rows: `{metrics['row_count']}`",
            f"- training-only MAE: `{metrics['mae']}`",
            f"- training-only MSE: `{metrics['mse']}`",
            "- validation evaluation attempted: `false`",
            "- future unseen evaluation attempted: `false`",
            "- source/freeze/promotion/paper/live unlock: `false`",
            "",
            "This artifact is frozen only as a research candidate for a later unseen "
            "evaluation. Training metrics are not eligibility or promotion evidence.",
            "",
        ]
    )


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")
