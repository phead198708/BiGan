"""Immutable remediation workflow for the v8 pre-promotion boundary."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_regime_conditioned_ev import (
    REGIME_CONDITIONED_EV_V2_FEATURE_GROUPS,
)
from bigan.v8.polymarket.training.execution_layer_v2_regime_conditioned_ev_calibration import (
    V2_REQUIRED_FEATURES,
    _chronological_market_split,
    _feature_value_for_model,
    _fit_feature_transforms,
    _group_scores,
    _market_bootstrap_improvement_intervals,
    _market_level_metrics,
    _predict_matrix,
    _regression_metrics,
    _ridge_fit,
    _validation_coverage_gate,
)

REMEDIATION_CONFIG_SCHEMA_VERSION = (
    "bigan-v8-execution-layer-v2-pre-promotion-remediation-config-v1"
)
REMEDIATION_EXCLUSION_SCHEMA_VERSION = (
    "bigan-v8-execution-layer-v2-pre-promotion-remediation-exclusions-v1"
)
REMEDIATION_STATE_SCHEMA_VERSION = (
    "bigan-v8-execution-layer-v2-pre-promotion-remediation-state-v1"
)


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2PrePromotionRemediationConfig:
    run_id: str
    output_dir: Path | str
    repository_root: Path | str
    created_at: str
    starting_branch: str
    starting_commit: str
    prior_blocked_bundle_dir: Path | str
    prior_corpus_rows_path: Path | str
    prior_split_report_path: Path | str
    prior_calibration_report_path: Path | str
    maximum_wall_clock_seconds: int = 21_600
    collection_window_seconds: int = 3_600
    maximum_collection_windows: int = 4
    collection_poll_interval_seconds: float = 60.0
    settlement_max_wait_seconds: float = 600.0
    settlement_poll_interval_seconds: float = 15.0
    minimum_total_calibration_rows: int = 150
    minimum_total_calibration_markets: int = 30
    minimum_development_fit_rows: int = 100
    minimum_development_fit_markets: int = 20
    minimum_fresh_validation_rows: int = 30
    minimum_fresh_validation_markets: int = 10
    minimum_validation_rows_per_side: int = 5
    minimum_validation_rows_per_action_family: int = 5
    minimum_validation_rows_per_resolved_outcome: int = 5
    minimum_validation_markets_per_category: int = 2
    minimum_relative_mae_improvement: float = 0.05
    minimum_relative_mse_improvement: float = 0.05
    bootstrap_samples: int = 1_000
    bootstrap_confidence_level: float = 0.95
    minimum_bootstrap_improvement_lower_bound: float = 0.0
    maximum_absolute_coefficient: float = 2.0
    maximum_lomo_coefficient_absolute_deviation: float = 0.50
    minimum_lomo_coefficient_sign_agreement: float = 0.75
    maximum_candidate_count: int = 6
    development_grouped_fold_count: int = 3
    candidate_complexity_penalty_per_parameter: float = 0.0001
    statistical_random_seed: int = 17_029
    required_future_shadow_window_count: int = 2
    future_shadow_window_seconds: int = 1_800
    minimum_future_shadow_rows: int = 30
    minimum_future_shadow_markets: int = 10

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if self.minimum_total_calibration_rows < 150:
            raise ValueError("existing corpus smoke requires at least 150 rows")
        if self.maximum_candidate_count > 6:
            raise ValueError("candidate search must remain bounded to at most 6")
        datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        for field in (
            "maximum_wall_clock_seconds",
            "collection_window_seconds",
            "maximum_collection_windows",
            "minimum_total_calibration_rows",
            "minimum_total_calibration_markets",
            "minimum_development_fit_rows",
            "minimum_development_fit_markets",
            "minimum_fresh_validation_rows",
            "minimum_fresh_validation_markets",
            "maximum_candidate_count",
            "required_future_shadow_window_count",
        ):
            if int(getattr(self, field)) <= 0:
                raise ValueError(f"{field} must be positive")
        for field in (
            "output_dir",
            "repository_root",
            "prior_blocked_bundle_dir",
            "prior_corpus_rows_path",
            "prior_split_report_path",
            "prior_calibration_report_path",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)).resolve())

    @property
    def goal_dir(self) -> Path:
        return Path(self.output_dir) / self.run_id / "pre_promotion_readiness"

    def frozen_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        for field in (
            "output_dir",
            "repository_root",
            "prior_blocked_bundle_dir",
            "prior_corpus_rows_path",
            "prior_split_report_path",
            "prior_calibration_report_path",
        ):
            payload[field] = str(payload[field])
        payload.update(
            {
                "schema_version": REMEDIATION_CONFIG_SCHEMA_VERSION,
                "candidate_development_data": "prior_140_row_corpus_only",
                "candidate_search_allowed_model_families": [
                    "five_group_ridge",
                    "reduced_group_ridge",
                    "standardized_feature_ridge",
                    "selected_side_probability_minus_price_baseline",
                ],
                "candidate_search_allowed_regularization_values": [0.1, 1.0],
                "candidate_search_allowed_transforms": [
                    "fit_only_standardization_clip_3",
                    "predeclared_group_weighted_aggregate",
                    "identity_single_feature",
                ],
                "candidate_ranking_rule": (
                    "grouped_market_cv_mse_then_mae_then_parameter_count_then_name"
                ),
                "candidate_selection_uses_fresh_validation": False,
                "fresh_validation_evaluated_exactly_once": True,
                "split_order": [
                    "development_fit",
                    "fresh_unseen_validation",
                    "future_unseen_shadow_reserved",
                ],
                "market_condition_run_and_row_disjoint_required": True,
                "chronological_split_required": True,
                "no_validation_or_shadow_tuning": True,
                "subtract_execution_cost": False,
                "target_semantics": "settled_net_return_after_execution_cost",
                "stop_conditions": [
                    "PRE_PROMOTION_READY",
                    "configured_wall_clock_budget_reached",
                    "configured_data_window_budget_reached",
                    "candidate_or_validation_hard_gate_failed",
                    "public_provider_fail_closed",
                ],
                **safety_fields(),
            }
        )
        return payload


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2PrePromotionRemediationInitializationResult:
    goal_dir: Path
    configuration_path: Path
    configuration_sha256_path: Path
    exclusions_path: Path
    exclusions_sha256_path: Path
    state_path: Path
    state_sha256_path: Path


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2CandidateDevelopmentResult:
    diagnosis_path: Path
    diagnosis_markdown_path: Path
    development_manifest_path: Path
    candidate_protocol_path: Path
    candidate_protocol_sha256_path: Path
    candidate_report_path: Path
    selected_contract_path: Path
    selected_contract_sha256_path: Path
    selected_candidate_name: str


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2FreshSplitResult:
    corpus_dir: Path
    accepted_rows_path: Path
    rejected_rows_path: Path
    corpus_manifest_path: Path
    split_manifest_path: Path
    split_manifest_sha256_path: Path
    leakage_report_path: Path
    fresh_validation_gate_passed: bool


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2FreshValidationResult:
    fit_report_path: Path
    validation_report_path: Path
    artifact_path: Path | None
    artifact_sha256_path: Path | None
    artifact_eligible: bool


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2RemediationFinalizationConfig:
    goal_dir: Path | str
    historical_collection_dirs: tuple[Path | str, ...] = ()
    outcome_reconciliation_dirs: tuple[Path | str, ...] = ()
    fresh_corpus_manifest_path: Path | str | None = None
    stop_reason_codes: tuple[str, ...] = ()
    resumable_next_command: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_dir", Path(self.goal_dir).resolve())
        object.__setattr__(
            self,
            "historical_collection_dirs",
            tuple(Path(path).resolve() for path in self.historical_collection_dirs),
        )
        object.__setattr__(
            self,
            "outcome_reconciliation_dirs",
            tuple(Path(path).resolve() for path in self.outcome_reconciliation_dirs),
        )
        if self.fresh_corpus_manifest_path is not None:
            object.__setattr__(
                self,
                "fresh_corpus_manifest_path",
                Path(self.fresh_corpus_manifest_path).resolve(),
            )


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2RemediationFinalizationResult:
    final_state: str
    report_path: Path
    manifest_path: Path
    manifest_sha256_path: Path


def initialize_pre_promotion_remediation_goal(
    config: ExecutionLayerV2PrePromotionRemediationConfig,
) -> ExecutionLayerV2PrePromotionRemediationInitializationResult:
    goal_dir = config.goal_dir
    if goal_dir.exists():
        raise FileExistsError(f"immutable remediation goal already exists: {goal_dir}")
    for path in (
        Path(config.prior_blocked_bundle_dir),
        Path(config.prior_corpus_rows_path),
        Path(config.prior_split_report_path),
        Path(config.prior_calibration_report_path),
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    goal_dir.mkdir(parents=True)

    configuration_path = goal_dir / "initial_goal_configuration.json"
    _write_json(configuration_path, config.frozen_payload())
    configuration_hash = sha256_file(configuration_path)
    configuration_hash_path = goal_dir / "initial_goal_configuration.sha256"
    configuration_hash_path.write_text(configuration_hash + "\n", encoding="utf-8")

    exclusions = _build_initial_exclusions(config, configuration_hash)
    exclusions_path = goal_dir / "initial_excluded_evidence_manifest.json"
    _write_json(exclusions_path, exclusions)
    exclusions_hash = sha256_file(exclusions_path)
    exclusions_hash_path = goal_dir / "initial_excluded_evidence_manifest.sha256"
    exclusions_hash_path.write_text(exclusions_hash + "\n", encoding="utf-8")

    repository_root = Path(config.repository_root)
    working_tree_rows = _git_status_rows(repository_root)
    state = {
        "schema_version": REMEDIATION_STATE_SCHEMA_VERSION,
        "run_id": config.run_id,
        "starting_branch": config.starting_branch,
        "starting_commit": config.starting_commit,
        "actual_head_at_initialization": _git(repository_root, "rev-parse", "HEAD"),
        "starting_commit_verified": (
            _git(repository_root, "rev-parse", "HEAD") == config.starting_commit
        ),
        "working_tree_clean": not working_tree_rows,
        "working_tree_status_rows": working_tree_rows,
        "tracked_working_tree_clean": not any(
            not row.startswith("?? ") for row in working_tree_rows
        ),
        "relevant_source_tree_sha256": _relevant_source_tree_hash(repository_root),
        "goal_configuration_sha256": configuration_hash,
        "excluded_evidence_manifest_sha256": exclusions_hash,
        "prior_blocked_bundle_manifest_sha256": exclusions[
            "prior_blocked_bundle_manifest_sha256"
        ],
        "created_at": config.created_at,
        "current_phase": "phase_0_immutable_audit_foundation_complete",
        "goal_status": "IN_PROGRESS",
        "resumable": True,
        **safety_fields(),
    }
    state["state_id"] = canonical_json_sha256(state)
    state_path = goal_dir / "initial_goal_state.json"
    _write_json(state_path, state)
    state_hash_path = goal_dir / "initial_goal_state.sha256"
    state_hash_path.write_text(sha256_file(state_path) + "\n", encoding="utf-8")
    return ExecutionLayerV2PrePromotionRemediationInitializationResult(
        goal_dir=goal_dir,
        configuration_path=configuration_path,
        configuration_sha256_path=configuration_hash_path,
        exclusions_path=exclusions_path,
        exclusions_sha256_path=exclusions_hash_path,
        state_path=state_path,
        state_sha256_path=state_hash_path,
    )


def diagnose_and_select_remediation_candidate(
    *,
    goal_dir: Path | str,
) -> ExecutionLayerV2CandidateDevelopmentResult:
    goal_dir = Path(goal_dir).resolve()
    configuration_path = goal_dir / "initial_goal_configuration.json"
    configuration_hash_path = goal_dir / "initial_goal_configuration.sha256"
    exclusions_path = goal_dir / "initial_excluded_evidence_manifest.json"
    _verify_immutable_file(configuration_path, configuration_hash_path)
    configuration = _load_json(configuration_path)
    exclusions = _load_json(exclusions_path)
    rows = _load_jsonl(Path(configuration["prior_corpus_rows_path"]))
    split_report = _load_json(Path(configuration["prior_split_report_path"]))
    calibration_report = _load_json(Path(configuration["prior_calibration_report_path"]))

    fit_rows, validation_rows, split_reasons = _chronological_market_split(
        rows,
        validation_fraction=0.25,
        min_fit_rows=100,
        min_validation_rows=30,
        min_fit_markets=20,
        min_validation_markets=10,
    )
    if split_reasons:
        raise ValueError(f"prior split cannot be reproduced: {split_reasons}")
    diagnosis = _previous_candidate_diagnosis(
        rows=rows,
        fit_rows=fit_rows,
        validation_rows=validation_rows,
        split_report=split_report,
        calibration_report=calibration_report,
        exclusions=exclusions,
    )
    diagnosis_path = goal_dir / "previous_candidate_diagnosis.json"
    _write_json(diagnosis_path, diagnosis)
    diagnosis_markdown_path = goal_dir / "previous_candidate_diagnosis.md"
    diagnosis_markdown_path.write_text(
        _diagnosis_markdown(diagnosis), encoding="utf-8"
    )

    development_manifest = {
        "schema_version": "bigan-v8-pre-promotion-development-evidence-v1",
        "source_corpus_path": configuration["prior_corpus_rows_path"],
        "source_corpus_sha256": sha256_file(Path(configuration["prior_corpus_rows_path"])),
        "row_count": len(rows),
        "market_count": len({row["market_id"] for row in rows}),
        "source_run_ids": sorted({row["source_run_id"] for row in rows}),
        "market_ids": sorted({row["market_id"] for row in rows}),
        "development_evidence_only": True,
        "unseen_validation_eligible": False,
        "future_shadow_eligible": False,
        "promotion_evidence_eligible": False,
        "prior_negative_evidence_preserved": True,
        "diagnosis_report": _descriptor(diagnosis_path),
        **safety_fields(),
    }
    development_manifest["manifest_id"] = canonical_json_sha256(
        development_manifest
    )
    development_manifest_path = goal_dir / "development_evidence_manifest.json"
    _write_json(development_manifest_path, development_manifest)

    candidates = _candidate_specifications()
    if len(candidates) > int(configuration["maximum_candidate_count"]):
        raise ValueError("candidate count exceeds frozen maximum")
    protocol = {
        "schema_version": "bigan-v8-pre-promotion-candidate-search-protocol-v1",
        "goal_configuration_sha256": sha256_file(configuration_path),
        "development_evidence_manifest_sha256": sha256_file(
            development_manifest_path
        ),
        "candidate_count": len(candidates),
        "maximum_candidate_count": configuration["maximum_candidate_count"],
        "candidates": candidates,
        "grouping_unit": "source_run_id",
        "grouped_fold_count": configuration["development_grouped_fold_count"],
        "ranking_rule": configuration["candidate_ranking_rule"],
        "complexity_penalty_per_parameter": configuration[
            "candidate_complexity_penalty_per_parameter"
        ],
        "selection_data": "prior_development_evidence_only",
        "uses_fresh_validation": False,
        "uses_future_shadow": False,
        "open_ended_search": False,
        **safety_fields(),
    }
    protocol["protocol_id"] = canonical_json_sha256(protocol)
    protocol_path = goal_dir / "candidate_search_protocol.json"
    _write_json(protocol_path, protocol)
    protocol_hash_path = goal_dir / "candidate_search_protocol.sha256"
    protocol_hash_path.write_text(sha256_file(protocol_path) + "\n", encoding="utf-8")

    evaluations = [
        _evaluate_development_candidate(
            rows,
            candidate,
            complexity_penalty=float(
                configuration["candidate_complexity_penalty_per_parameter"]
            ),
        )
        for candidate in candidates
    ]
    eligible = [row for row in evaluations if row["development_gate_passed"]]
    ranked = sorted(
        eligible or evaluations,
        key=lambda row: (
            row["selection_score"],
            row["grouped_cv_market_mae"],
            row["parameter_count"],
            row["candidate_name"],
        ),
    )
    selected = ranked[0]
    report = {
        "schema_version": "bigan-v8-pre-promotion-candidate-development-report-v1",
        "candidate_search_protocol_sha256": sha256_file(protocol_path),
        "development_evidence_manifest_sha256": sha256_file(
            development_manifest_path
        ),
        "candidate_evaluations": evaluations,
        "ranked_candidate_names": [row["candidate_name"] for row in ranked],
        "selected_candidate_name": selected["candidate_name"],
        "selected_candidate_development_gate_passed": selected[
            "development_gate_passed"
        ],
        "selection_rule_applied": protocol["ranking_rule"],
        "fresh_validation_rows_read": 0,
        "uses_fresh_validation_for_selection": False,
        "uses_future_shadow_for_selection": False,
        **safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = goal_dir / "candidate_development_report.json"
    _write_json(report_path, report)

    selected_spec = next(
        candidate
        for candidate in candidates
        if candidate["candidate_name"] == selected["candidate_name"]
    )
    contract = {
        "schema_version": "bigan-v8-pre-promotion-selected-candidate-contract-v1",
        "candidate_search_protocol_sha256": sha256_file(protocol_path),
        "candidate_development_report_sha256": sha256_file(report_path),
        "candidate_name": selected["candidate_name"],
        "candidate_specification": selected_spec,
        "development_metrics": selected,
        "feature_contract": {
            "decision_time_only": True,
            "required_features": selected_spec["features"],
            "max_input_ts_must_not_exceed_decision_ts": True,
            "forbidden_fields": [
                "settlement",
                "outcome",
                "pnl",
                "oracle_action",
                "future_return",
                "future_price",
            ],
        },
        "preprocessing_contract": selected_spec["transform"],
        "target_semantics": "settled_net_return_after_execution_cost",
        "subtract_execution_cost": False,
        "immutable_after_hash": True,
        "uses_fresh_validation_for_selection": False,
        **safety_fields(),
    }
    contract["contract_id"] = canonical_json_sha256(contract)
    contract_path = goal_dir / "selected_candidate_contract.json"
    _write_json(contract_path, contract)
    contract_hash_path = goal_dir / "selected_candidate_contract.sha256"
    contract_hash_path.write_text(sha256_file(contract_path) + "\n", encoding="utf-8")
    return ExecutionLayerV2CandidateDevelopmentResult(
        diagnosis_path=diagnosis_path,
        diagnosis_markdown_path=diagnosis_markdown_path,
        development_manifest_path=development_manifest_path,
        candidate_protocol_path=protocol_path,
        candidate_protocol_sha256_path=protocol_hash_path,
        candidate_report_path=report_path,
        selected_contract_path=contract_path,
        selected_contract_sha256_path=contract_hash_path,
        selected_candidate_name=selected["candidate_name"],
    )


def freeze_remediation_fresh_split(
    *,
    goal_dir: Path | str,
    fresh_corpus_rows_path: Path | str,
    fresh_corpus_quality_report_path: Path | str,
) -> ExecutionLayerV2FreshSplitResult:
    goal_dir = Path(goal_dir).resolve()
    configuration_path = goal_dir / "initial_goal_configuration.json"
    configuration_hash_path = goal_dir / "initial_goal_configuration.sha256"
    exclusions_path = goal_dir / "initial_excluded_evidence_manifest.json"
    exclusions_hash_path = goal_dir / "initial_excluded_evidence_manifest.sha256"
    candidate_contract_path = goal_dir / "selected_candidate_contract.json"
    candidate_contract_hash_path = goal_dir / "selected_candidate_contract.sha256"
    for path, hash_path in (
        (configuration_path, configuration_hash_path),
        (exclusions_path, exclusions_hash_path),
        (candidate_contract_path, candidate_contract_hash_path),
    ):
        _verify_immutable_file(path, hash_path)
    config = _load_json(configuration_path)
    exclusions = _load_json(exclusions_path)
    development_rows = _load_jsonl(Path(config["prior_corpus_rows_path"]))
    fresh_rows = _load_jsonl(Path(fresh_corpus_rows_path).resolve())
    fresh_quality = _load_json(Path(fresh_corpus_quality_report_path).resolve())
    development_identities = {row["row_identity"] for row in development_rows}
    development_markets = {row["market_id"] for row in development_rows}
    development_runs = {row["source_run_id"] for row in development_rows}
    fresh_identities = {row["row_identity"] for row in fresh_rows}
    fresh_markets = {row["market_id"] for row in fresh_rows}
    fresh_runs = {row["source_run_id"] for row in fresh_rows}
    overlap = {
        "row_identity_overlap": sorted(development_identities & fresh_identities),
        "market_id_overlap": sorted(development_markets & fresh_markets),
        "source_run_id_overlap": sorted(development_runs & fresh_runs),
    }
    chronology_passed = bool(
        development_rows
        and fresh_rows
        and max(float(row["decision_ts"]) for row in development_rows)
        < min(float(row["decision_ts"]) for row in fresh_rows)
    )
    causality_violations = [
        row["row_identity"]
        for row in [*development_rows, *fresh_rows]
        if float(row["max_input_ts"]) > float(row["decision_ts"])
    ]
    excluded_markets = set(exclusions["prior_evidence_market_ids"])
    exclusion_contract_passed = bool(
        development_markets <= excluded_markets
        and not (fresh_markets & excluded_markets)
    )
    validation_coverage = _validation_coverage_gate(
        fresh_rows,
        min_rows_per_side=int(config["minimum_validation_rows_per_side"]),
        min_rows_per_action_family=int(
            config["minimum_validation_rows_per_action_family"]
        ),
        min_rows_per_resolved_outcome=int(
            config["minimum_validation_rows_per_resolved_outcome"]
        ),
        min_markets_per_category=int(
            config["minimum_validation_markets_per_category"]
        ),
    )
    support_passed = bool(
        len(development_rows) >= int(config["minimum_development_fit_rows"])
        and len(development_markets)
        >= int(config["minimum_development_fit_markets"])
        and len(fresh_rows) >= int(config["minimum_fresh_validation_rows"])
        and len(fresh_markets)
        >= int(config["minimum_fresh_validation_markets"])
        and len(development_rows) + len(fresh_rows)
        >= int(config["minimum_total_calibration_rows"])
        and len(development_markets | fresh_markets)
        >= int(config["minimum_total_calibration_markets"])
    )
    split_gate_passed = bool(
        support_passed
        and validation_coverage["coverage_gate_passed"]
        and chronology_passed
        and exclusion_contract_passed
        and not any(overlap.values())
        and not causality_violations
    )
    blockers = []
    if not support_passed:
        blockers.append("fresh_split_minimum_support_not_met")
    blockers.extend(validation_coverage["blocking_reason_codes"])
    if not chronology_passed:
        blockers.append("fresh_split_chronology_failed")
    if any(overlap.values()):
        blockers.append("fresh_split_identity_overlap_detected")
    if not exclusion_contract_passed:
        blockers.append("excluded_prior_evidence_entered_fresh_validation")
    if causality_violations:
        blockers.append("fresh_split_feature_timestamp_causality_violation")

    corpus_dir = goal_dir / "versioned_calibration_corpus"
    if corpus_dir.exists():
        raise FileExistsError(f"fresh split already frozen: {corpus_dir}")
    corpus_dir.mkdir()
    accepted_path = corpus_dir / "accepted_calibration_rows.jsonl"
    accepted_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in [*development_rows, *fresh_rows]
        ),
        encoding="utf-8",
    )
    rejected_path = corpus_dir / "rejected_calibration_rows.jsonl"
    rejected_path.write_text("", encoding="utf-8")
    lineage_path = corpus_dir / "calibration_row_lineage.jsonl"
    lineage_path.write_text(
        "".join(
            json.dumps(
                {
                    "row_identity": row["row_identity"],
                    "market_id": row["market_id"],
                    "source_run_id": row["source_run_id"],
                    "lineage": lineage,
                },
                sort_keys=True,
            )
            + "\n"
            for lineage, rows in (
                ("development", development_rows),
                ("fresh_validation_candidate", fresh_rows),
            )
            for row in rows
        ),
        encoding="utf-8",
    )
    validation_report = {
        "schema_version": "bigan-v8-remediation-calibration-row-validation-v1",
        "accepted_row_count": len(development_rows) + len(fresh_rows),
        "rejected_row_count": 0,
        "schema_runtime_validation_agreement_passed": fresh_quality.get(
            "schema_runtime_validation_agreement_passed", True
        ),
        "provenance_violation_count": fresh_quality.get(
            "provenance_coverage", {}
        ).get("violation_count", 0),
        "causality_violation_count": len(causality_violations),
        "rejection_reason_distribution": {},
        **safety_fields(),
    }
    validation_report_path = corpus_dir / "calibration_row_validation_report.json"
    _write_json(validation_report_path, validation_report)
    quality_path = corpus_dir / "calibration_corpus_quality_report.json"
    _write_json(
        quality_path,
        {
            "schema_version": "bigan-v8-remediation-corpus-quality-v1",
            "development_row_count": len(development_rows),
            "fresh_validation_row_count": len(fresh_rows),
            "total_row_count": len(development_rows) + len(fresh_rows),
            "development_market_count": len(development_markets),
            "fresh_validation_market_count": len(fresh_markets),
            "total_market_count": len(development_markets | fresh_markets),
            "fresh_validation_coverage": validation_coverage,
            "incremental_full_rebuild_hash_match": fresh_quality.get(
                "incremental_full_rebuild_hash_match"
            ),
            "split_gate_passed": split_gate_passed,
            "blocking_reason_codes": sorted(set(blockers)),
            **safety_fields(),
        },
    )
    corpus_manifest = {
        "schema_version": "bigan-v8-remediation-calibration-corpus-manifest-v1",
        "goal_configuration_sha256": sha256_file(configuration_path),
        "selected_candidate_contract_sha256": sha256_file(
            candidate_contract_path
        ),
        "development_source": _descriptor(Path(config["prior_corpus_rows_path"])),
        "fresh_validation_source": _descriptor(Path(fresh_corpus_rows_path)),
        "fresh_quality_source": _descriptor(Path(fresh_corpus_quality_report_path)),
        "accepted_rows": _descriptor(accepted_path),
        "rejected_rows": _descriptor(rejected_path),
        "row_lineage": _descriptor(lineage_path),
        "row_validation_report": _descriptor(validation_report_path),
        "quality_report": _descriptor(quality_path),
        "total_row_count": len(development_rows) + len(fresh_rows),
        "total_market_count": len(development_markets | fresh_markets),
        **safety_fields(),
    }
    corpus_manifest["manifest_id"] = canonical_json_sha256(corpus_manifest)
    corpus_manifest_path = corpus_dir / "calibration_corpus_manifest.json"
    _write_json(corpus_manifest_path, corpus_manifest)

    split_manifest = {
        "schema_version": "bigan-v8-remediation-fresh-split-manifest-v1",
        "goal_configuration_sha256": sha256_file(configuration_path),
        "selected_candidate_contract_sha256": sha256_file(
            candidate_contract_path
        ),
        "calibration_corpus_manifest_sha256": sha256_file(corpus_manifest_path),
        "development_fit": _split_partition_summary(development_rows),
        "fresh_unseen_validation": _split_partition_summary(fresh_rows),
        "future_unseen_shadow_reserved": {
            "status": "reserved_not_collected_before_artifact_freeze",
            "row_count": 0,
            "market_count": 0,
        },
        "fresh_validation_coverage": validation_coverage,
        "overlap_checks": overlap,
        "chronology_passed": chronology_passed,
        "exclusion_contract_passed": exclusion_contract_passed,
        "feature_timestamp_causality_violation_count": len(causality_violations),
        "fresh_split_gate_passed": split_gate_passed,
        "blocking_reason_codes": sorted(set(blockers)),
        "validation_outcomes_used_for_candidate_selection": False,
        "validation_outcomes_used_for_split_adjustment": False,
        **safety_fields(),
    }
    split_manifest["split_id"] = canonical_json_sha256(split_manifest)
    split_path = goal_dir / "fresh_split_manifest.json"
    _write_json(split_path, split_manifest)
    split_hash_path = goal_dir / "fresh_split_manifest.sha256"
    split_hash_path.write_text(sha256_file(split_path) + "\n", encoding="utf-8")
    leakage_report = {
        "schema_version": "bigan-v8-remediation-split-leakage-report-v1",
        "fresh_split_manifest_sha256": sha256_file(split_path),
        "chronology_passed": chronology_passed,
        "market_disjointness_passed": not overlap["market_id_overlap"],
        "run_disjointness_passed": not overlap["source_run_id_overlap"],
        "economic_row_disjointness_passed": not overlap["row_identity_overlap"],
        "excluded_evidence_check_passed": exclusion_contract_passed,
        "feature_timestamp_causality_passed": not causality_violations,
        "validation_labels_used_for_fitting": False,
        "validation_labels_used_for_threshold_selection": False,
        "future_shadow_contamination": False,
        "leakage_checks_passed": bool(
            chronology_passed
            and exclusion_contract_passed
            and not any(overlap.values())
            and not causality_violations
        ),
        **safety_fields(),
    }
    leakage_report["report_id"] = canonical_json_sha256(leakage_report)
    leakage_path = goal_dir / "split_leakage_report.json"
    _write_json(leakage_path, leakage_report)
    return ExecutionLayerV2FreshSplitResult(
        corpus_dir=corpus_dir,
        accepted_rows_path=accepted_path,
        rejected_rows_path=rejected_path,
        corpus_manifest_path=corpus_manifest_path,
        split_manifest_path=split_path,
        split_manifest_sha256_path=split_hash_path,
        leakage_report_path=leakage_path,
        fresh_validation_gate_passed=split_gate_passed,
    )


def evaluate_remediation_candidate_once(
    *,
    goal_dir: Path | str,
) -> ExecutionLayerV2FreshValidationResult:
    goal_dir = Path(goal_dir).resolve()
    configuration_path = goal_dir / "initial_goal_configuration.json"
    configuration_hash_path = goal_dir / "initial_goal_configuration.sha256"
    exclusions_path = goal_dir / "initial_excluded_evidence_manifest.json"
    exclusions_hash_path = goal_dir / "initial_excluded_evidence_manifest.sha256"
    candidate_path = goal_dir / "selected_candidate_contract.json"
    candidate_hash_path = goal_dir / "selected_candidate_contract.sha256"
    split_path = goal_dir / "fresh_split_manifest.json"
    split_hash_path = goal_dir / "fresh_split_manifest.sha256"
    for path, hash_path in (
        (configuration_path, configuration_hash_path),
        (exclusions_path, exclusions_hash_path),
        (candidate_path, candidate_hash_path),
        (split_path, split_hash_path),
    ):
        _verify_immutable_file(path, hash_path)
    config = _load_json(configuration_path)
    candidate_contract = _load_json(candidate_path)
    split = _load_json(split_path)
    if split.get("fresh_split_gate_passed") is not True:
        raise ValueError("fresh split gate failed; validation evaluation is forbidden")
    evaluation_marker = goal_dir / "fresh_validation_evaluation_started.json"
    if evaluation_marker.exists():
        raise FileExistsError("fresh validation evaluation is exactly-once")
    marker = {
        "schema_version": "bigan-v8-fresh-validation-exactly-once-marker-v1",
        "started_at": utc_now_iso(),
        "selected_candidate_contract_sha256": sha256_file(candidate_path),
        "fresh_split_manifest_sha256": sha256_file(split_path),
        "evaluation_attempt_number": 1,
        **safety_fields(),
    }
    marker["marker_id"] = canonical_json_sha256(marker)
    _write_json(evaluation_marker, marker)

    development_rows = _load_jsonl(Path(config["prior_corpus_rows_path"]))
    fresh_source = Path(
        _load_json(
            goal_dir
            / "versioned_calibration_corpus"
            / "calibration_corpus_manifest.json"
        )["fresh_validation_source"]["path"]
    )
    validation_rows = _load_jsonl(fresh_source)
    blockers = list(split["blocking_reason_codes"])
    candidate_spec = candidate_contract["candidate_specification"]
    model = _fit_candidate(development_rows, candidate_spec)
    candidate_predictions = _predict_candidate(
        validation_rows, candidate_spec, model
    )
    fit_targets = [float(row["target_net_return_after_cost"]) for row in development_rows]
    validation_targets = [
        float(row["target_net_return_after_cost"]) for row in validation_rows
    ]
    constant_predictions = [sum(fit_targets) / len(fit_targets)] * len(
        validation_rows
    )
    legacy_fit_matrix = [
        [float(row["decision_time_features"]["canonical_o_action_score"])]
        for row in development_rows
    ]
    legacy_validation_matrix = [
        [float(row["decision_time_features"]["canonical_o_action_score"])]
        for row in validation_rows
    ]
    legacy_coefficients = _ridge_fit(legacy_fit_matrix, fit_targets, 1.0)
    legacy_predictions = _predict_matrix(
        legacy_validation_matrix, legacy_coefficients
    )
    probability_minus_price_predictions = [
        float(
            row["decision_time_features"][
                "selected_side_probability_minus_execution_price"
            ]
        )
        for row in validation_rows
    ]
    prediction_sets = {
        "candidate": candidate_predictions,
        "constant_baseline": constant_predictions,
        "legacy_o_score_baseline": legacy_predictions,
        "selected_side_probability_minus_execution_price_baseline": (
            probability_minus_price_predictions
        ),
    }
    row_metrics = {
        name: _regression_metrics(validation_targets, predictions)
        for name, predictions in prediction_sets.items()
    }
    market_metrics = {
        name: _market_level_metrics(
            validation_rows, validation_targets, predictions
        )
        for name, predictions in prediction_sets.items()
    }
    relative_improvements = _all_baseline_relative_improvements(
        row_metrics,
        market_metrics,
        minimum_mae=float(config["minimum_relative_mae_improvement"]),
        minimum_mse=float(config["minimum_relative_mse_improvement"]),
    )
    bootstrap = _market_bootstrap_improvement_intervals(
        validation_rows,
        validation_targets,
        candidate_predictions=candidate_predictions,
        baseline_predictions={
            name: predictions
            for name, predictions in prediction_sets.items()
            if name != "candidate"
        },
        samples=int(config["bootstrap_samples"]),
        confidence_level=float(config["bootstrap_confidence_level"]),
        minimum_lower_bound=float(
            config["minimum_bootstrap_improvement_lower_bound"]
        ),
        random_seed=int(config["statistical_random_seed"]),
    )
    stability = _selected_candidate_lomo_stability(
        development_rows,
        candidate_spec,
        model["coefficients"],
        max_deviation=float(
            config["maximum_lomo_coefficient_absolute_deviation"]
        ),
        min_sign_agreement=float(
            config["minimum_lomo_coefficient_sign_agreement"]
        ),
    )
    finite_bounded = all(
        math.isfinite(value)
        and abs(value) <= float(config["maximum_absolute_coefficient"])
        for value in model["coefficients"]
    )
    if not relative_improvements["all_row_and_market_gates_passed"]:
        blockers.append("fresh_validation_relative_improvement_gate_failed")
    if not bootstrap["confidence_gate_passed"]:
        blockers.append("fresh_validation_market_bootstrap_gate_failed")
    if not stability["stability_gate_passed"]:
        blockers.append("fresh_validation_coefficient_stability_gate_failed")
    if not finite_bounded:
        blockers.append("fresh_validation_coefficients_not_finite_and_bounded")
    blockers = sorted(set(blockers))
    artifact_eligible = not blockers

    fit_report = {
        "schema_version": "bigan-v8-remediation-selected-candidate-fit-report-v1",
        "selected_candidate_contract_sha256": sha256_file(candidate_path),
        "fresh_split_manifest_sha256": sha256_file(split_path),
        "development_fit_row_count": len(development_rows),
        "development_fit_market_count": len(
            {row["market_id"] for row in development_rows}
        ),
        "candidate_specification": candidate_spec,
        "coefficients": model["coefficients"],
        "coefficient_hash": canonical_json_sha256(model["coefficients"]),
        "feature_transforms": model["transforms"],
        "coefficient_stability": stability,
        "coefficients_finite_and_bounded": finite_bounded,
        "uses_validation_labels_for_fitting": False,
        "uses_validation_labels_for_threshold_selection": False,
        "subtract_execution_cost": False,
        **safety_fields(),
    }
    fit_report["report_id"] = canonical_json_sha256(fit_report)
    fit_path = goal_dir / "fit_report.json"
    _write_json(fit_path, fit_report)

    residual_rows = [
        {
            **row,
            "prediction": prediction,
            "residual": target - prediction,
            "absolute_error": abs(target - prediction),
            "squared_error": (target - prediction) ** 2,
        }
        for row, target, prediction in zip(
            validation_rows,
            validation_targets,
            candidate_predictions,
            strict=True,
        )
    ]
    validation_report = {
        "schema_version": "bigan-v8-remediation-fresh-validation-report-v1",
        "selected_candidate_contract_sha256": sha256_file(candidate_path),
        "fresh_split_manifest_sha256": sha256_file(split_path),
        "evaluation_attempt_number": 1,
        "fresh_validation_row_count": len(validation_rows),
        "fresh_validation_market_count": len(
            {row["market_id"] for row in validation_rows}
        ),
        "row_level_metrics": row_metrics,
        "market_level_metrics": market_metrics,
        "relative_baseline_improvements": relative_improvements,
        "market_bootstrap_confidence_intervals": bootstrap,
        "coefficient_stability": stability,
        "validation_coverage": split["fresh_validation_coverage"],
        "calibration_slope_intercept": _calibration_slope_intercept(
            candidate_predictions, validation_targets
        ),
        "residual_diagnostics": {
            "by_side": _residual_summary(
                residual_rows, lambda row: row["selected_side"]
            ),
            "by_action_family": _residual_summary(
                residual_rows, lambda row: row["action_family"]
            ),
            "by_market_horizon": _residual_summary(
                residual_rows, lambda row: _market_horizon(row)
            ),
            "by_execution_price_band": _residual_summary(
                residual_rows,
                lambda row: _numeric_bucket(
                    row["decision_time_features"]["execution_price"],
                    (0.3, 0.5, 0.7, 0.9),
                ),
            ),
            "by_time_to_close_band": _residual_summary(
                residual_rows,
                lambda row: _numeric_bucket(
                    row["decision_time_features"]["time_to_close_seconds"],
                    (60.0, 120.0, 240.0, 360.0),
                ),
            ),
        },
        "all_frozen_gates_passed": artifact_eligible,
        "artifact_eligible": artifact_eligible,
        "blocking_reason_codes": blockers,
        "validation_labels_used_for_fitting": False,
        "validation_labels_used_for_threshold_selection": False,
        "candidate_or_split_mutated_after_evaluation": False,
        **safety_fields(),
    }
    validation_report["report_id"] = canonical_json_sha256(validation_report)
    validation_path = goal_dir / "fresh_validation_report.json"
    _write_json(validation_path, validation_report)

    artifact_path = None
    artifact_hash_path = None
    if artifact_eligible:
        artifact = {
            "schema_version": "bigan-v8-frozen-remediation-regime-ev-v1",
            "artifact_name": "execution_layer_v2_frozen_remediation_regime_ev_v1",
            "frozen": True,
            "decision_time_safe": True,
            "goal_configuration_sha256": sha256_file(configuration_path),
            "selected_candidate_contract_sha256": sha256_file(candidate_path),
            "fresh_split_manifest_sha256": sha256_file(split_path),
            "fit_report_sha256": sha256_file(fit_path),
            "fresh_validation_report_sha256": sha256_file(validation_path),
            "candidate_specification": candidate_spec,
            "coefficients": model["coefficients"],
            "feature_transforms": model["transforms"],
            "target_semantics": "settled_net_return_after_execution_cost",
            "subtract_execution_cost": False,
            "statistical_eligibility_passed": True,
            "future_unseen_shadow_required": True,
            **safety_fields(),
        }
        artifact["artifact_id"] = canonical_json_sha256(artifact)
        artifact_path = goal_dir / "frozen_diagnostic_artifact.json"
        _write_json(artifact_path, artifact)
        artifact_hash_path = goal_dir / "frozen_diagnostic_artifact.sha256"
        artifact_hash_path.write_text(
            sha256_file(artifact_path) + "\n", encoding="utf-8"
        )
    return ExecutionLayerV2FreshValidationResult(
        fit_report_path=fit_path,
        validation_report_path=validation_path,
        artifact_path=artifact_path,
        artifact_sha256_path=artifact_hash_path,
        artifact_eligible=artifact_eligible,
    )


def finalize_pre_promotion_remediation_goal(
    config: ExecutionLayerV2RemediationFinalizationConfig,
) -> ExecutionLayerV2RemediationFinalizationResult:
    goal_dir = Path(config.goal_dir)
    initial_files = {
        "configuration": (
            goal_dir / "initial_goal_configuration.json",
            goal_dir / "initial_goal_configuration.sha256",
        ),
        "exclusions": (
            goal_dir / "initial_excluded_evidence_manifest.json",
            goal_dir / "initial_excluded_evidence_manifest.sha256",
        ),
        "state": (
            goal_dir / "initial_goal_state.json",
            goal_dir / "initial_goal_state.sha256",
        ),
        "candidate_protocol": (
            goal_dir / "candidate_search_protocol.json",
            goal_dir / "candidate_search_protocol.sha256",
        ),
        "candidate_contract": (
            goal_dir / "selected_candidate_contract.json",
            goal_dir / "selected_candidate_contract.sha256",
        ),
    }
    immutable_checks = {}
    for name, (path, hash_path) in initial_files.items():
        try:
            _verify_immutable_file(path, hash_path)
            immutable_checks[name] = True
        except (FileNotFoundError, ValueError):
            immutable_checks[name] = False
    configuration = _load_json(initial_files["configuration"][0])
    initial_state = _load_json(initial_files["state"][0])
    exclusions = _load_json(initial_files["exclusions"][0])
    repository_root = Path(configuration["repository_root"])
    starting_commit_verified = bool(
        configuration["starting_commit"] == initial_state["starting_commit"]
        and initial_state["actual_head_at_initialization"]
        == configuration["starting_commit"]
    )
    prior_manifest_path = Path(exclusions["prior_blocked_bundle_manifest_path"])
    prior_bundle_hash_verified = bool(
        prior_manifest_path.is_file()
        and sha256_file(prior_manifest_path)
        == exclusions["prior_blocked_bundle_manifest_sha256"]
    )
    collection_rows = [
        _phase_run_descriptor(path, "one_hour_remap_paper_goal_report.json")
        for path in config.historical_collection_dirs
    ]
    reconciliation_rows = [
        _phase_run_descriptor(path, "clob_settlement_reconciliation_report.json")
        for path in config.outcome_reconciliation_dirs
    ]
    historical_manifest = {
        "schema_version": "bigan-v8-remediation-historical-collection-manifest-v1",
        "collection_window_count": len(collection_rows),
        "collection_windows": collection_rows,
        "complete_round_count": sum(
            int(row["report"].get("complete_round_count", 0))
            for row in collection_rows
        ),
        "paper_fill_count": sum(
            int(row["report"].get("paper_fill_count", 0))
            for row in collection_rows
        ),
        **safety_fields(),
    }
    historical_manifest["manifest_id"] = canonical_json_sha256(historical_manifest)
    historical_path = goal_dir / "historical_collection_manifest.json"
    _write_json(historical_path, historical_manifest)
    unresolved = sum(
        int(row["report"].get("unresolved_fill_count_after", 0))
        for row in reconciliation_rows
    )
    reconciliation_manifest = {
        "schema_version": "bigan-v8-remediation-outcome-reconciliation-manifest-v1",
        "reconciliation_run_count": len(reconciliation_rows),
        "reconciliation_runs": reconciliation_rows,
        "unresolved_fill_count_after": unresolved,
        "original_source_artifacts_immutable": all(
            row["report"].get("original_source_artifacts_mutated") is False
            for row in reconciliation_rows
        ),
        **safety_fields(),
    }
    reconciliation_manifest["manifest_id"] = canonical_json_sha256(
        reconciliation_manifest
    )
    reconciliation_path = goal_dir / "outcome_reconciliation_manifest.json"
    _write_json(reconciliation_path, reconciliation_manifest)

    if config.fresh_corpus_manifest_path is not None:
        corpus_source = Path(config.fresh_corpus_manifest_path)
        corpus_wrapper = {
            "schema_version": "bigan-v8-remediation-final-corpus-manifest-v1",
            "source_manifest": _descriptor(corpus_source),
            "source_manifest_payload": _load_json(corpus_source),
            **safety_fields(),
        }
    else:
        corpus_wrapper = {
            "schema_version": "bigan-v8-remediation-final-corpus-manifest-v1",
            "status": "not_available_fail_closed",
            "blocking_reason_codes": ["fresh_calibration_corpus_not_available"],
            **safety_fields(),
        }
    corpus_wrapper["manifest_id"] = canonical_json_sha256(corpus_wrapper)
    corpus_path = goal_dir / "calibration_corpus_manifest.json"
    _write_json(corpus_path, corpus_wrapper)

    split_path = goal_dir / "fresh_split_manifest.json"
    split_payload = _load_json(split_path) if split_path.is_file() else {}
    validation_path = goal_dir / "fresh_validation_report.json"
    validation_payload = _load_json(validation_path) if validation_path.is_file() else {}
    artifact_path = goal_dir / "frozen_diagnostic_artifact.json"
    artifact_hash_path = goal_dir / "frozen_diagnostic_artifact.sha256"
    artifact_valid = bool(
        artifact_path.is_file()
        and artifact_hash_path.is_file()
        and sha256_file(artifact_path)
        == artifact_hash_path.read_text(encoding="utf-8").strip()
        and validation_payload.get("artifact_eligible") is True
    )
    shadow_manifest_path = goal_dir / "future_shadow_manifest.json"
    shadow_evaluation_path = goal_dir / "future_shadow_evaluation_report.json"
    shadow_payload = (
        _load_json(shadow_evaluation_path)
        if shadow_evaluation_path.is_file()
        else {}
    )
    shadow_complete = bool(
        shadow_manifest_path.is_file()
        and shadow_evaluation_path.is_file()
        and shadow_payload.get("future_shadow_all_gates_passed") is True
    )
    finalization_checks = {
        "goal_configuration_hash_unchanged": immutable_checks["configuration"],
        "initial_exclusion_manifest_hash_unchanged": immutable_checks["exclusions"],
        "initial_state_hash_unchanged": immutable_checks["state"],
        "candidate_search_protocol_hash_unchanged": immutable_checks[
            "candidate_protocol"
        ],
        "selected_candidate_contract_hash_unchanged": immutable_checks[
            "candidate_contract"
        ],
        "starting_commit_verified": starting_commit_verified,
        "prior_blocked_bundle_manifest_hash_verified": prior_bundle_hash_verified,
        "fresh_split_gate_passed": split_payload.get("fresh_split_gate_passed") is True,
        "fresh_validation_all_gates_passed": validation_payload.get(
            "all_frozen_gates_passed"
        )
        is True,
        "frozen_diagnostic_artifact_valid": artifact_valid,
        "required_future_shadow_complete": shadow_complete,
        "unresolved_settlement_count_zero": unresolved == 0,
    }
    blockers = set(config.stop_reason_codes)
    blockers.update(split_payload.get("blocking_reason_codes", []))
    blockers.update(validation_payload.get("blocking_reason_codes", []))
    blockers.update(
        f"finalization_check_failed:{name}"
        for name, passed in finalization_checks.items()
        if not passed
    )
    ready = all(finalization_checks.values()) and not blockers
    final_state = "PRE_PROMOTION_READY" if ready else "PRE_PROMOTION_BLOCKED"
    commits = _git(
        repository_root,
        "rev-list",
        "--reverse",
        f"{configuration['starting_commit']}..HEAD",
    ).splitlines()
    report = {
        "schema_version": "bigan-v8-pre-promotion-remediation-readiness-report-v1",
        "final_state": final_state,
        "pre_promotion_readiness_complete": ready,
        "goal_configuration_sha256": sha256_file(
            initial_files["configuration"][0]
        ),
        "selected_candidate_contract_sha256": sha256_file(
            initial_files["candidate_contract"][0]
        ),
        "historical_collection_summary": {
            "window_count": len(collection_rows),
            "complete_round_count": historical_manifest["complete_round_count"],
            "paper_fill_count": historical_manifest["paper_fill_count"],
        },
        "outcome_reconciliation_summary": {
            "run_count": len(reconciliation_rows),
            "unresolved_fill_count_after": unresolved,
        },
        "fresh_split_summary": split_payload,
        "fresh_validation_summary": validation_payload,
        "frozen_diagnostic_artifact_created": artifact_valid,
        "future_shadow_complete": shadow_complete,
        "finalization_verification": finalization_checks,
        "code_changes_used_for_final_candidate": {
            "starting_commit": configuration["starting_commit"],
            "final_head": _git(repository_root, "rev-parse", "HEAD"),
            "commit_ids": commits,
            "relevant_source_tree_sha256": _relevant_source_tree_hash(
                repository_root
            ),
        },
        "blocking_reason_codes": sorted(blockers),
        "resumable_next_command": config.resumable_next_command,
        **safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = goal_dir / "pre_promotion_readiness_report.json"
    _write_json(report_path, report)
    markdown_path = goal_dir / "pre_promotion_readiness_report.md"
    markdown_path.write_text(_remediation_markdown(report), encoding="utf-8")
    state = {
        "schema_version": "bigan-v8-pre-promotion-remediation-final-state-v1",
        "final_state": final_state,
        "pre_promotion_readiness_complete": ready,
        "resumable": not ready,
        "blocking_reason_codes": sorted(blockers),
        **safety_fields(),
    }
    state["state_id"] = canonical_json_sha256(state)
    state_path = goal_dir / "pre_promotion_goal_state.json"
    _write_json(state_path, state)

    manifest_path = goal_dir / "pre_promotion_readiness_manifest.json"
    manifest_hash_path = goal_dir / "pre_promotion_readiness_manifest.sha256"
    artifact_files = sorted(
        path
        for path in goal_dir.rglob("*")
        if path.is_file()
        and path not in {manifest_path, manifest_hash_path}
    )
    manifest = {
        "schema_version": "bigan-v8-pre-promotion-remediation-manifest-v1",
        "final_state": final_state,
        "pre_promotion_readiness_complete": ready,
        "artifact_count": len(artifact_files),
        "artifacts": [
            {
                "path": str(path.resolve()),
                "relative_path": str(path.relative_to(goal_dir)),
                "sha256": sha256_file(path),
            }
            for path in artifact_files
        ],
        "manifest_self_hash_embedded": False,
        "manifest_hash_descriptor_external": True,
        **safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    _write_json(manifest_path, manifest)
    manifest_hash_path.write_text(sha256_file(manifest_path) + "\n", encoding="utf-8")
    return ExecutionLayerV2RemediationFinalizationResult(
        final_state=final_state,
        report_path=report_path,
        manifest_path=manifest_path,
        manifest_sha256_path=manifest_hash_path,
    )


def _build_initial_exclusions(
    config: ExecutionLayerV2PrePromotionRemediationConfig,
    configuration_hash: str,
) -> dict[str, Any]:
    prior_bundle = Path(config.prior_blocked_bundle_dir)
    prior_manifest = prior_bundle / "pre_promotion_readiness_manifest.json"
    prior_rows = _load_jsonl(Path(config.prior_corpus_rows_path))
    prior_run_ids = sorted({str(row["source_run_id"]) for row in prior_rows})
    prior_market_ids = sorted({str(row["market_id"]) for row in prior_rows})
    artifacts = [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
        for path in sorted(prior_bundle.iterdir())
        if path.is_file()
    ]
    payload = {
        "schema_version": REMEDIATION_EXCLUSION_SCHEMA_VERSION,
        "goal_configuration_sha256": configuration_hash,
        "prior_blocked_bundle_path": str(prior_bundle),
        "prior_blocked_bundle_manifest_path": str(prior_manifest),
        "prior_blocked_bundle_manifest_sha256": sha256_file(prior_manifest),
        "prior_development_corpus_path": str(Path(config.prior_corpus_rows_path)),
        "prior_development_corpus_sha256": sha256_file(
            Path(config.prior_corpus_rows_path)
        ),
        "prior_evidence_row_count": len(prior_rows),
        "prior_evidence_market_count": len(prior_market_ids),
        "prior_evidence_run_ids": prior_run_ids,
        "prior_evidence_market_ids": prior_market_ids,
        "development_evidence_only": True,
        "unseen_validation_eligible": False,
        "future_shadow_eligible": False,
        "promotion_evidence_eligible": False,
        "prior_evidence_usage_contract": {
            "development_evidence_only": True,
            "candidate_diagnosis_allowed": True,
            "candidate_development_allowed": True,
            "unseen_validation_eligible": False,
            "future_shadow_eligible": False,
            "promotion_evidence_eligible": False,
        },
        "prior_bundle_artifacts": artifacts,
        "excluded_run_name_fragments": [
            "future-shadow",
            "future_holdout",
            "post-freeze",
            "schedule-debug",
            "settlement-debug",
        ],
        **safety_fields(),
    }
    payload["manifest_id"] = canonical_json_sha256(payload)
    return payload


def _previous_candidate_diagnosis(
    *,
    rows: list[dict[str, Any]],
    fit_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    split_report: dict[str, Any],
    calibration_report: dict[str, Any],
    exclusions: dict[str, Any],
) -> dict[str, Any]:
    transforms = _fit_feature_transforms(fit_rows)
    fit_x = [_group_scores(row, transforms) for row in fit_rows]
    validation_x = [_group_scores(row, transforms) for row in validation_rows]
    fit_y = [float(row["target_net_return_after_cost"]) for row in fit_rows]
    coefficients = _ridge_fit(fit_x, fit_y, 1.0)
    predictions = _predict_matrix(validation_x, coefficients)
    residual_rows = [
        {
            **row,
            "prediction": prediction,
            "residual": float(row["target_net_return_after_cost"]) - prediction,
            "absolute_error": abs(
                float(row["target_net_return_after_cost"]) - prediction
            ),
            "squared_error": (
                float(row["target_net_return_after_cost"]) - prediction
            )
            ** 2,
        }
        for row, prediction in zip(validation_rows, predictions, strict=True)
    ]
    targets = [float(row["target_net_return_after_cost"]) for row in validation_rows]
    prediction_variance = statistics.pvariance(predictions) if len(predictions) > 1 else 0.0
    target_variance = statistics.pvariance(targets) if len(targets) > 1 else 0.0
    group_contributions = {}
    group_names = list(REGIME_CONDITIONED_EV_V2_FEATURE_GROUPS)
    for index, group_name in enumerate(group_names, start=1):
        values = [abs(coefficients[index] * vector[index - 1]) for vector in validation_x]
        group_contributions[group_name] = {
            "coefficient": coefficients[index],
            "mean_absolute_contribution": sum(values) / len(values),
            "max_absolute_contribution": max(values),
        }
    concentration = {
        "by_side": _residual_summary(residual_rows, lambda row: row["selected_side"]),
        "by_action_family": _residual_summary(
            residual_rows, lambda row: row["action_family"]
        ),
        "by_market_horizon": _residual_summary(
            residual_rows, lambda row: _market_horizon(row)
        ),
        "by_execution_price_band": _residual_summary(
            residual_rows,
            lambda row: _numeric_bucket(
                row["decision_time_features"]["execution_price"],
                (0.3, 0.5, 0.7, 0.9),
            ),
        ),
        "by_time_to_close_band": _residual_summary(
            residual_rows,
            lambda row: _numeric_bucket(
                row["decision_time_features"]["time_to_close_seconds"],
                (60.0, 120.0, 240.0, 360.0),
            ),
        ),
        "by_entry_index": _residual_summary(
            residual_rows,
            lambda row: str(int(row["decision_time_features"]["entry_index_within_market"])),
        ),
        "by_same_side_reentry": _residual_summary(
            residual_rows,
            lambda row: str(int(row["decision_time_features"]["same_side_reentry"])),
        ),
        "by_side_flip": _residual_summary(
            residual_rows,
            lambda row: str(int(row["decision_time_features"]["side_flip"])),
        ),
        "by_market": _residual_summary(residual_rows, lambda row: row["market_id"]),
        "by_run": _residual_summary(residual_rows, lambda row: row["source_run_id"]),
    }
    full_sbc = [row for row in rows if row["action_family"] == "SELL_BEFORE_CLOSE"]
    validation_market_ids = {row["market_id"] for row in validation_rows}
    sbc_by_run = Counter(row["source_run_id"] for row in full_sbc)
    diagnosis = {
        "schema_version": "bigan-v8-previous-candidate-diagnosis-v1",
        "development_evidence_only": True,
        "unseen_validation_eligible": False,
        "future_shadow_eligible": False,
        "promotion_evidence_eligible": False,
        "prior_negative_result_preserved": True,
        "prior_split_reproduced": bool(
            len(fit_rows) == int(split_report["fit_row_count"])
            and len(validation_rows) == int(split_report["validation_row_count"])
        ),
        "fit_row_count": len(fit_rows),
        "validation_row_count": len(validation_rows),
        "validation_metrics": _regression_metrics(targets, predictions),
        "prediction_dispersion": {
            "prediction_variance": prediction_variance,
            "target_variance": target_variance,
            "prediction_to_target_variance_ratio": (
                prediction_variance / target_variance if target_variance else None
            ),
            "classification": (
                "over_shrunk"
                if prediction_variance < target_variance * 0.75
                else "over_dispersed"
                if prediction_variance > target_variance * 1.25
                else "similar_dispersion"
            ),
        },
        "feature_group_contribution_diagnostics": group_contributions,
        "coefficient_stability_from_prior_report": calibration_report.get(
            "coefficient_stability_metrics"
        ),
        "residual_concentration": concentration,
        "top_squared_error_rows": [
            _diagnostic_residual_row(row)
            for row in sorted(
                residual_rows, key=lambda row: row["squared_error"], reverse=True
            )[:10]
        ],
        "largest_market_error_share": _largest_market_error_share(residual_rows),
        "sell_before_close_absence_diagnosis": {
            "full_corpus_sell_before_close_row_count": len(full_sbc),
            "full_corpus_sell_before_close_market_count": len(
                {row["market_id"] for row in full_sbc}
            ),
            "sell_before_close_rows_by_run": dict(sorted(sbc_by_run.items())),
            "validation_sell_before_close_row_count": sum(
                row["action_family"] == "SELL_BEFORE_CLOSE"
                for row in validation_rows
            ),
            "validation_sell_before_close_market_count": len(
                {
                    row["market_id"]
                    for row in full_sbc
                    if row["market_id"] in validation_market_ids
                }
            ),
            "chronological_split_caused_zero_validation_coverage": bool(
                full_sbc
                and not any(
                    row["action_family"] == "SELL_BEFORE_CLOSE"
                    for row in validation_rows
                )
            ),
            "candidate_corpus_contains_realized_fills_only": True,
            "non_selected_counterfactuals_used_as_targets": False,
            "natural_policy_action_distribution_is_sparse": len(full_sbc) < 10,
            "diagnostic_reason_codes": [
                "realized_sell_before_close_fill_support_sparse",
                "sell_before_close_rows_occurred_before_frozen_validation_boundary",
                "chronological_split_correctly_did_not_rebalance_after_outcome_inspection",
            ],
        },
        "baseline_comparison_from_prior_report": {
            "fit_metrics": calibration_report.get("fit_metrics"),
            "market_level_metrics": calibration_report.get("market_level_metrics"),
            "relative_baseline_improvements": calibration_report.get(
                "relative_baseline_improvements"
            ),
        },
        "why_mse_worsened_despite_mae_improvement": (
            "a small number of large residuals outweighed broad small absolute-error "
            "improvements; validation candidate MSE exceeded both baselines"
        ),
        "excluded_evidence_manifest_id": exclusions["manifest_id"],
        **safety_fields(),
    }
    diagnosis["report_id"] = canonical_json_sha256(diagnosis)
    return diagnosis


def _candidate_specifications() -> list[dict[str, Any]]:
    all_features = list(V2_REQUIRED_FEATURES)
    return [
        {
            "candidate_name": "five_group_ridge_alpha_1",
            "model_family": "five_group_ridge",
            "ridge_alpha": 1.0,
            "features": all_features,
            "groups": list(REGIME_CONDITIONED_EV_V2_FEATURE_GROUPS),
            "transform": "fit_only_standardization_clip_3_group_aggregate",
        },
        {
            "candidate_name": "five_group_ridge_alpha_0_1",
            "model_family": "five_group_ridge",
            "ridge_alpha": 0.1,
            "features": all_features,
            "groups": list(REGIME_CONDITIONED_EV_V2_FEATURE_GROUPS),
            "transform": "fit_only_standardization_clip_3_group_aggregate",
        },
        {
            "candidate_name": "reduced_score_value_exposure_ridge_alpha_1",
            "model_family": "reduced_group_ridge",
            "ridge_alpha": 1.0,
            "features": [
                *REGIME_CONDITIONED_EV_V2_FEATURE_GROUPS[
                    "canonical_o_score_and_action_margin"
                ],
                *REGIME_CONDITIONED_EV_V2_FEATURE_GROUPS["market_price_value"],
                *REGIME_CONDITIONED_EV_V2_FEATURE_GROUPS["pre_entry_exposure_state"],
            ],
            "groups": [
                "canonical_o_score_and_action_margin",
                "market_price_value",
                "pre_entry_exposure_state",
            ],
            "transform": "fit_only_standardization_clip_3_group_aggregate",
        },
        {
            "candidate_name": "reduced_score_value_ridge_alpha_0_1",
            "model_family": "reduced_group_ridge",
            "ridge_alpha": 0.1,
            "features": [
                *REGIME_CONDITIONED_EV_V2_FEATURE_GROUPS[
                    "canonical_o_score_and_action_margin"
                ],
                *REGIME_CONDITIONED_EV_V2_FEATURE_GROUPS["market_price_value"],
            ],
            "groups": [
                "canonical_o_score_and_action_margin",
                "market_price_value",
            ],
            "transform": "fit_only_standardization_clip_3_group_aggregate",
        },
        {
            "candidate_name": "standardized_feature_ridge_alpha_1",
            "model_family": "standardized_feature_ridge",
            "ridge_alpha": 1.0,
            "features": all_features,
            "groups": [],
            "transform": "fit_only_standardization_clip_3_each_feature",
        },
        {
            "candidate_name": "selected_side_probability_minus_price_baseline",
            "model_family": "selected_side_probability_minus_price_baseline",
            "ridge_alpha": 1.0,
            "features": ["selected_side_probability_minus_execution_price"],
            "groups": [],
            "transform": "identity_single_feature",
        },
    ]


def _evaluate_development_candidate(
    rows: list[dict[str, Any]],
    candidate: dict[str, Any],
    *,
    complexity_penalty: float,
) -> dict[str, Any]:
    run_ids = sorted({row["source_run_id"] for row in rows})
    folds = []
    all_validation_rows: list[dict[str, Any]] = []
    all_targets: list[float] = []
    all_predictions: list[float] = []
    coefficient_rows: list[list[float]] = []
    for held_out_run_id in run_ids:
        fit_rows = [row for row in rows if row["source_run_id"] != held_out_run_id]
        validation_rows = [row for row in rows if row["source_run_id"] == held_out_run_id]
        if not fit_rows or not validation_rows:
            continue
        model = _fit_candidate(fit_rows, candidate)
        predictions = _predict_candidate(validation_rows, candidate, model)
        targets = [float(row["target_net_return_after_cost"]) for row in validation_rows]
        coefficient_rows.append(model["coefficients"])
        folds.append(
            {
                "held_out_run_id": held_out_run_id,
                "fit_row_count": len(fit_rows),
                "validation_row_count": len(validation_rows),
                "validation_market_count": len(
                    {row["market_id"] for row in validation_rows}
                ),
                "row_metrics": _regression_metrics(targets, predictions),
                "market_metrics": _market_level_metrics(
                    validation_rows, targets, predictions
                ),
            }
        )
        all_validation_rows.extend(validation_rows)
        all_targets.extend(targets)
        all_predictions.extend(predictions)
    row_metrics = _regression_metrics(all_targets, all_predictions)
    market_metrics = _market_level_metrics(
        all_validation_rows, all_targets, all_predictions
    )
    parameter_count = len(coefficient_rows[0]) if coefficient_rows else 0
    max_fold_mse = max(fold["market_metrics"]["mse"] for fold in folds)
    max_coefficient_deviation = _coefficient_deviation(coefficient_rows)
    development_gate_passed = bool(
        len(folds) >= 3
        and math.isfinite(market_metrics["mse"])
        and max_coefficient_deviation <= 1.0
    )
    return {
        "candidate_name": candidate["candidate_name"],
        "model_family": candidate["model_family"],
        "fold_count": len(folds),
        "folds": folds,
        "grouped_cv_row_mae": row_metrics["mae"],
        "grouped_cv_row_mse": row_metrics["mse"],
        "grouped_cv_market_mae": market_metrics["mae"],
        "grouped_cv_market_mse": market_metrics["mse"],
        "worst_fold_market_mse": max_fold_mse,
        "parameter_count": parameter_count,
        "complexity_penalty": parameter_count * complexity_penalty,
        "selection_score": market_metrics["mse"]
        + parameter_count * complexity_penalty,
        "max_cross_fold_coefficient_deviation": max_coefficient_deviation,
        "development_gate_passed": development_gate_passed,
        "side_coverage": dict(Counter(row["selected_side"] for row in rows)),
        "action_family_coverage": dict(
            Counter(row["action_family"] for row in rows)
        ),
        "resolved_outcome_coverage": dict(
            Counter(row["target_provenance"]["resolved_outcome"] for row in rows)
        ),
    }


def _fit_candidate(
    rows: list[dict[str, Any]], candidate: dict[str, Any]
) -> dict[str, Any]:
    model_family = candidate["model_family"]
    targets = [float(row["target_net_return_after_cost"]) for row in rows]
    if model_family in {"five_group_ridge", "reduced_group_ridge"}:
        transforms = _fit_feature_transforms(rows)
        all_groups = list(REGIME_CONDITIONED_EV_V2_FEATURE_GROUPS)
        indices = [all_groups.index(group) for group in candidate["groups"]]
        matrix = [
            [score[index] for index in indices]
            for score in (_group_scores(row, transforms) for row in rows)
        ]
    elif model_family == "standardized_feature_ridge":
        transforms = _fit_feature_transforms(rows)
        matrix = [
            [
                _normalized_feature_value(row, feature, transforms)
                for feature in candidate["features"]
            ]
            for row in rows
        ]
        indices = []
    else:
        transforms = {}
        matrix = [
            [
                float(
                    row["decision_time_features"][
                        "selected_side_probability_minus_execution_price"
                    ]
                )
            ]
            for row in rows
        ]
        indices = []
    return {
        "coefficients": _ridge_fit(matrix, targets, float(candidate["ridge_alpha"])),
        "transforms": transforms,
        "group_indices": indices,
    }


def _predict_candidate(
    rows: list[dict[str, Any]],
    candidate: dict[str, Any],
    model: dict[str, Any],
) -> list[float]:
    model_family = candidate["model_family"]
    if model_family in {"five_group_ridge", "reduced_group_ridge"}:
        matrix = [
            [score[index] for index in model["group_indices"]]
            for score in (
                _group_scores(row, model["transforms"]) for row in rows
            )
        ]
    elif model_family == "standardized_feature_ridge":
        matrix = [
            [
                _normalized_feature_value(row, feature, model["transforms"])
                for feature in candidate["features"]
            ]
            for row in rows
        ]
    else:
        matrix = [
            [
                float(
                    row["decision_time_features"][
                        "selected_side_probability_minus_execution_price"
                    ]
                )
            ]
            for row in rows
        ]
    return _predict_matrix(matrix, model["coefficients"])


def _normalized_feature_value(
    row: dict[str, Any],
    feature: str,
    transforms: dict[str, dict[str, float]],
) -> float:
    transform = transforms[feature]
    normalized = (
        _feature_value_for_model(row, feature) - transform["center"]
    ) / transform["scale"]
    return max(transform["clip_min"], min(transform["clip_max"], normalized))


def _split_partition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "market_count": len({row["market_id"] for row in rows}),
        "market_ids": sorted({row["market_id"] for row in rows}),
        "condition_ids": sorted({row["market_id"] for row in rows}),
        "source_run_ids": sorted({row["source_run_id"] for row in rows}),
        "min_decision_ts": min(float(row["decision_ts"]) for row in rows),
        "max_decision_ts": max(float(row["decision_ts"]) for row in rows),
        "dataset_hash": canonical_json_sha256(
            sorted(row["row_identity"] for row in rows)
        ),
        "side_coverage": dict(sorted(Counter(row["selected_side"] for row in rows).items())),
        "action_family_coverage": dict(
            sorted(Counter(row["action_family"] for row in rows).items())
        ),
        "resolved_outcome_coverage": dict(
            sorted(
                Counter(
                    row["target_provenance"]["resolved_outcome"] for row in rows
                ).items()
            )
        ),
    }


def _all_baseline_relative_improvements(
    row_metrics: dict[str, dict[str, Any]],
    market_metrics: dict[str, dict[str, Any]],
    *,
    minimum_mae: float,
    minimum_mse: float,
) -> dict[str, Any]:
    comparisons = {}
    candidate_row = row_metrics["candidate"]
    candidate_market = market_metrics["candidate"]
    for baseline_name in row_metrics:
        if baseline_name == "candidate":
            continue
        baseline_row = row_metrics[baseline_name]
        baseline_market = market_metrics[baseline_name]
        row_mae = _relative_improvement(baseline_row["mae"], candidate_row["mae"])
        row_mse = _relative_improvement(baseline_row["mse"], candidate_row["mse"])
        market_mae = _relative_improvement(
            baseline_market["mae"], candidate_market["mae"]
        )
        market_mse = _relative_improvement(
            baseline_market["mse"], candidate_market["mse"]
        )
        comparisons[baseline_name] = {
            "row_mae_relative_improvement": row_mae,
            "row_mse_relative_improvement": row_mse,
            "market_mae_relative_improvement": market_mae,
            "market_mse_relative_improvement": market_mse,
            "passed": bool(
                row_mae >= minimum_mae
                and row_mse >= minimum_mse
                and market_mae >= minimum_mae
                and market_mse >= minimum_mse
            ),
        }
    return {
        "minimum_relative_mae_improvement": minimum_mae,
        "minimum_relative_mse_improvement": minimum_mse,
        "comparisons": comparisons,
        "all_row_and_market_gates_passed": all(
            row["passed"] for row in comparisons.values()
        ),
    }


def _relative_improvement(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / baseline if baseline > 0.0 else 0.0


def _selected_candidate_lomo_stability(
    rows: list[dict[str, Any]],
    candidate: dict[str, Any],
    full_coefficients: list[float],
    *,
    max_deviation: float,
    min_sign_agreement: float,
) -> dict[str, Any]:
    coefficients = []
    by_market = {}
    for market_id in sorted({row["market_id"] for row in rows}):
        subset = [row for row in rows if row["market_id"] != market_id]
        fitted = _fit_candidate(subset, candidate)["coefficients"]
        coefficients.append(fitted)
        by_market[market_id] = {
            "max_absolute_deviation": max(
                abs(value - full)
                for value, full in zip(fitted, full_coefficients, strict=True)
            )
        }
    maximum = max(
        value["max_absolute_deviation"] for value in by_market.values()
    )
    sign_checks = []
    for row in coefficients:
        for value, full in zip(row[1:], full_coefficients[1:], strict=True):
            sign_checks.append(
                abs(value) <= max_deviation
                if abs(full) <= 1e-12
                else (value > 0.0) == (full > 0.0)
            )
    agreement = sum(sign_checks) / len(sign_checks) if sign_checks else 1.0
    return {
        "method": "leave_one_market_out",
        "replicate_count": len(coefficients),
        "max_absolute_deviation": maximum,
        "max_absolute_deviation_allowed": max_deviation,
        "coefficient_sign_agreement_rate": agreement,
        "minimum_sign_agreement_required": min_sign_agreement,
        "stability_gate_passed": bool(
            maximum <= max_deviation and agreement >= min_sign_agreement
        ),
        "by_omitted_market": by_market,
    }


def _calibration_slope_intercept(
    predictions: list[float], targets: list[float]
) -> dict[str, float | None]:
    if len(predictions) < 2 or statistics.pvariance(predictions) <= 1e-15:
        return {"slope": None, "intercept": None}
    mean_prediction = statistics.mean(predictions)
    mean_target = statistics.mean(targets)
    covariance = sum(
        (prediction - mean_prediction) * (target - mean_target)
        for prediction, target in zip(predictions, targets, strict=True)
    ) / len(predictions)
    slope = covariance / statistics.pvariance(predictions)
    return {"slope": slope, "intercept": mean_target - slope * mean_prediction}


def _coefficient_deviation(rows: list[list[float]]) -> float:
    if len(rows) < 2:
        return math.inf
    return max(
        max(values) - min(values)
        for values in zip(*rows, strict=True)
    )


def _residual_summary(
    rows: list[dict[str, Any]], key_function: Any
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(key_function(row))].append(row)
    return {
        key: {
            "row_count": len(values),
            "mae": sum(row["absolute_error"] for row in values) / len(values),
            "mse": sum(row["squared_error"] for row in values) / len(values),
            "mean_residual": sum(row["residual"] for row in values) / len(values),
            "target_sum": sum(
                float(row["target_net_return_after_cost"]) for row in values
            ),
        }
        for key, values in sorted(grouped.items())
    }


def _diagnostic_residual_row(row: dict[str, Any]) -> dict[str, Any]:
    features = row["decision_time_features"]
    return {
        "market_id": row["market_id"],
        "source_run_id": row["source_run_id"],
        "decision_ts": row["decision_ts"],
        "selected_side": row["selected_side"],
        "selected_action": row["selected_action"],
        "action_family": row["action_family"],
        "execution_price": features["execution_price"],
        "time_to_close_seconds": features["time_to_close_seconds"],
        "entry_index_within_market": features["entry_index_within_market"],
        "same_side_reentry": features["same_side_reentry"],
        "side_flip": features["side_flip"],
        "target": row["target_net_return_after_cost"],
        "prediction": row["prediction"],
        "residual": row["residual"],
        "squared_error": row["squared_error"],
    }


def _largest_market_error_share(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_market: dict[str, float] = defaultdict(float)
    for row in rows:
        by_market[row["market_id"]] += row["squared_error"]
    total = sum(by_market.values())
    market_id, value = max(by_market.items(), key=lambda item: item[1])
    return {
        "market_id": market_id,
        "squared_error_sum": value,
        "share_of_validation_squared_error": value / total if total else 0.0,
        "single_market_dominance_detected": bool(total and value / total > 0.30),
    }


def _market_horizon(row: dict[str, Any]) -> str:
    source = str(row.get("source_run_id") or "")
    return "5m" if "5m" in source or "pre-promotion" in source else "unknown"


def _numeric_bucket(value: float, boundaries: tuple[float, ...]) -> str:
    numeric = float(value)
    lower = "-inf"
    for boundary in boundaries:
        if numeric < boundary:
            return f"[{lower},{boundary})"
        lower = str(boundary)
    return f"[{lower},inf)"


def _diagnosis_markdown(report: dict[str, Any]) -> str:
    dispersion = report["prediction_dispersion"]
    sbc = report["sell_before_close_absence_diagnosis"]
    market = report["largest_market_error_share"]
    return "\n".join(
        [
            "# Previous Regime-Conditioned EV Candidate Diagnosis",
            "",
            "This report is development evidence only and is not unseen validation or promotion evidence.",
            "",
            "## Findings",
            "",
            f"- Prediction dispersion: `{dispersion['classification']}`",
            f"- Prediction/target variance ratio: `{dispersion['prediction_to_target_variance_ratio']}`",
            f"- Largest market squared-error share: `{market['share_of_validation_squared_error']}`",
            f"- Validation SELL_BEFORE_CLOSE rows: `{sbc['validation_sell_before_close_row_count']}`",
            f"- Full-corpus SELL_BEFORE_CLOSE rows: `{sbc['full_corpus_sell_before_close_row_count']}`",
            "- MSE worsened because a small number of large errors outweighed small broad MAE gains.",
            "- The chronological split was preserved; no outcome-aware rebalancing was performed.",
            "",
        ]
    )


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _phase_run_descriptor(path: Path, report_name: str) -> dict[str, Any]:
    report_path = path / report_name
    if not report_path.is_file():
        matches = sorted(path.rglob(report_name)) if path.is_dir() else []
        report_path = matches[0] if matches else report_path
    report = _load_json(report_path) if report_path.is_file() else {}
    manifests = sorted(path.rglob("*manifest*.json")) if path.is_dir() else []
    return {
        "run_dir": str(path),
        "report": report,
        "report_artifact": _descriptor(report_path) if report_path.is_file() else None,
        "manifest_artifacts": [_descriptor(item) for item in manifests],
    }


def _remediation_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# v8 Pre-Promotion Remediation Readiness",
        "",
        f"- Final state: `{report['final_state']}`",
        f"- Pre-promotion readiness complete: `{str(report['pre_promotion_readiness_complete']).lower()}`",
        "- Promotion evidence stage started: `false`",
        "- Promotion evidence eligible: `false`",
        "- Live evidence stage started: `false`",
        "- Live evidence allowed: `false`",
        "- v8 execution handoff allowed: `false`",
        "",
        "## Blocking Reasons",
        "",
    ]
    blockers = report["blocking_reason_codes"]
    lines.extend(f"- `{reason}`" for reason in blockers)
    if not blockers:
        lines.append("- None")
    lines.extend(
        [
            "",
            "This bundle stops before promotion evidence and does not authorize paper/live writes, wallet signing, capital, freeze, promotion, or handoff.",
            "",
        ]
    )
    return "\n".join(lines)


def _verify_immutable_file(path: Path, hash_path: Path) -> None:
    if not path.is_file() or not hash_path.is_file():
        raise FileNotFoundError(path)
    expected = hash_path.read_text(encoding="utf-8").strip()
    if sha256_file(path) != expected:
        raise ValueError(f"immutable artifact hash mismatch: {path}")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def safety_fields() -> dict[str, Any]:
    return {
        "diagnostic_only": True,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "promotion_evidence_stage_started": False,
        "live_evidence_stage_started": False,
        "live_evidence_allowed": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _git_status_rows(root: Path) -> list[str]:
    output = _git(root, "status", "--porcelain")
    return output.splitlines() if output else []


def _relevant_source_tree_hash(root: Path) -> str:
    tracked = _git(
        root,
        "ls-files",
        "src/bigan/v8/polymarket",
        "examples/v8",
        "tests/v8",
    ).splitlines()
    return canonical_json_sha256(
        [
            {"path": relative, "sha256": sha256_file(root / relative)}
            for relative in sorted(tracked)
            if (root / relative).is_file()
        ]
    )


__all__ = [
    "ExecutionLayerV2CandidateDevelopmentResult",
    "ExecutionLayerV2FreshSplitResult",
    "ExecutionLayerV2FreshValidationResult",
    "ExecutionLayerV2RemediationFinalizationConfig",
    "ExecutionLayerV2RemediationFinalizationResult",
    "ExecutionLayerV2PrePromotionRemediationConfig",
    "ExecutionLayerV2PrePromotionRemediationInitializationResult",
    "diagnose_and_select_remediation_candidate",
    "evaluate_remediation_candidate_once",
    "finalize_pre_promotion_remediation_goal",
    "freeze_remediation_fresh_split",
    "initialize_pre_promotion_remediation_goal",
    "safety_fields",
    "sha256_file",
    "utc_now_iso",
]
