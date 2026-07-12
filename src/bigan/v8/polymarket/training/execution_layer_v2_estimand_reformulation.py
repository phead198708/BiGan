"""Probability-first pre-promotion workflow for Execution Layer v2.

This module deliberately stops at diagnostic pre-promotion readiness.  It never
changes source scores, execution guards, paper/live permissions, or promotion
state.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256

STARTING_COMMIT = "de201a016f6e21a1fd4c9d07a82ab46ba9a6c11d"
STARTING_BRANCH = "codex/v8-pre-promotion-remediation-goal"
WORKING_BRANCH = "codex/v8-pre-promotion-estimand-reformulation-goal"

SUPPORTED_FAMILY = "HOLD_TO_SETTLEMENT"
UNSUPPORTED_FAMILY = "SELL_BEFORE_CLOSE"
MAXIMUM_CANDIDATE_COUNT = 8
MAXIMUM_VALIDATION_ROUNDS = 3
PER_ROUND_CONFIDENCE_LEVEL = 1.0 - (0.05 / MAXIMUM_VALIDATION_ROUNDS)


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


@dataclass(frozen=True, slots=True)
class EstimandReformulationConfig:
    run_id: str
    output_dir: Path | str
    repository_root: Path | str
    prior_blocked_bundle_dir: Path | str
    inspected_rows_path: Path | str
    created_at: str
    additional_excluded_run_ids: tuple[str, ...] = ()
    maximum_total_validation_collection_windows: int = 12
    validation_collection_window_seconds: int = 3600
    maximum_wall_clock_seconds: int = 43_200
    minimum_validation_rows_per_round: int = 80
    minimum_validation_markets_per_round: int = 25
    minimum_validation_up_rows: int = 15
    minimum_validation_down_rows: int = 15
    minimum_resolved_up_markets: int = 8
    minimum_resolved_down_markets: int = 8
    minimum_hts_markets: int = 25
    minimum_sbc_development_rows: int = 30
    minimum_sbc_development_markets: int = 10
    minimum_relative_mae_improvement: float = 0.05
    minimum_relative_mse_improvement: float = 0.05
    minimum_brier_score_improvement: float = 0.03
    minimum_log_loss_improvement: float = 0.03
    bootstrap_samples: int = 1000
    bootstrap_improvement_lower_bound: float = 0.0
    calibration_slope_minimum: float = 0.75
    calibration_slope_maximum: float = 1.25
    maximum_absolute_calibration_intercept: float = 0.10
    maximum_per_market_row_share: float = 0.20
    required_future_shadow_window_count: int = 2
    minimum_total_shadow_rows: int = 80
    minimum_total_shadow_markets: int = 25
    statistical_random_seed: int = 17029

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        for name in (
            "output_dir",
            "repository_root",
            "prior_blocked_bundle_dir",
            "inspected_rows_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())
        if self.maximum_total_validation_collection_windows < 1:
            raise ValueError("validation collection window budget must be positive")
        if self.minimum_validation_rows_per_round < 80:
            raise ValueError("minimum validation rows cannot be lower than 80")
        if self.minimum_validation_markets_per_round < 25:
            raise ValueError("minimum validation markets cannot be lower than 25")

    @property
    def goal_dir(self) -> Path:
        return Path(self.output_dir) / self.run_id / "pre_promotion_readiness"

    def payload(self) -> dict[str, Any]:
        payload = asdict(self)
        for name in (
            "output_dir",
            "repository_root",
            "prior_blocked_bundle_dir",
            "inspected_rows_path",
        ):
            payload[name] = str(payload[name])
        payload.update(
            {
                "schema_version": "bigan-v8-estimand-reformulation-config-v1",
                "starting_branch": STARTING_BRANCH,
                "starting_commit": STARTING_COMMIT,
                "working_branch": WORKING_BRANCH,
                "maximum_candidate_count": MAXIMUM_CANDIDATE_COUNT,
                "maximum_validation_rounds": MAXIMUM_VALIDATION_ROUNDS,
                "per_round_bootstrap_confidence_level": PER_ROUND_CONFIDENCE_LEVEL,
                "prior_negative_evidence_preserved": True,
                "previous_direct_return_regression_route_not_assumed_valid": True,
                "full_content_row_hashes_required": True,
                "immutable_internal_rows_only": True,
                "additional_excluded_run_ids": list(
                    self.additional_excluded_run_ids
                ),
                **safety_fields(),
            }
        )
        return payload


def initialize_estimand_reformulation_goal(
    config: EstimandReformulationConfig,
) -> dict[str, Any]:
    goal_dir = config.goal_dir
    if goal_dir.exists():
        raise FileExistsError(f"goal directory already exists: {goal_dir}")
    goal_dir.mkdir(parents=True)
    prior_manifest = Path(config.prior_blocked_bundle_dir) / "pre_promotion_readiness_manifest.json"
    if not prior_manifest.exists():
        raise FileNotFoundError(f"prior blocked manifest missing: {prior_manifest}")
    rows = _load_jsonl(Path(config.inspected_rows_path))
    normalized, rejected = _normalize_development_rows(rows)
    if rejected:
        raise ValueError(f"inspected evidence contains {len(rejected)} invalid rows")

    sbc = [row for row in normalized if row["action_family"] == UNSUPPORTED_FAMILY]
    sbc_markets = {row["market_id"] for row in sbc}
    full_scope = (
        len(sbc) >= config.minimum_sbc_development_rows
        and len(sbc_markets) >= config.minimum_sbc_development_markets
    )
    scope = "FULL_ACTION_FAMILIES" if full_scope else "HTS_ONLY"
    if full_scope:
        raise ValueError(
            "full-scope SBC estimand is intentionally unsupported until a separate "
            "pre-close exit target contract is implemented"
        )

    config_payload = config.payload()
    config_payload.update(
        {
            "actual_head": _git(config.repository_root, "rev-parse", "HEAD"),
            "working_tree_status": _git(config.repository_root, "status", "--short"),
            "source_tree_hash": _source_tree_hash(config.repository_root),
            "candidate_scope": scope,
            "supported_action_families": [SUPPORTED_FAMILY],
            "unsupported_action_families": [UNSUPPORTED_FAMILY],
            "sbc_scope_support": {
                "row_count": len(sbc),
                "market_count": len(sbc_markets),
                "minimum_row_count": config.minimum_sbc_development_rows,
                "minimum_market_count": config.minimum_sbc_development_markets,
                "full_scope_gate_passed": full_scope,
            },
        }
    )
    _write_hashed_json(goal_dir, "initial_goal_configuration", config_payload)

    excluded = {
        "schema_version": "bigan-v8-estimand-reformulation-exclusions-v1",
        "prior_blocked_bundle": _directory_descriptor(
            Path(config.prior_blocked_bundle_dir)
        ),
        "prior_blocked_manifest": _descriptor(prior_manifest),
        "all_inspected_row_identities": sorted(row["row_identity"] for row in normalized),
        "all_inspected_market_ids": sorted({row["market_id"] for row in normalized}),
        "all_inspected_condition_ids": sorted(
            {row.get("condition_id", row["market_id"]) for row in normalized}
        ),
        "all_inspected_run_ids": sorted({row["source_run_id"] for row in normalized}),
        "additional_excluded_run_ids": list(config.additional_excluded_run_ids),
        "usage": {
            "lineage": "development",
            "development_evidence_only": True,
            "unseen_validation_eligible": False,
            "future_shadow_eligible": False,
            "promotion_evidence_eligible": False,
        },
        "prior_negative_evidence_preserved": True,
        **safety_fields(),
    }
    _write_hashed_json(goal_dir, "initial_excluded_evidence_manifest", excluded)

    estimand = _estimand_protocol(scope)
    _write_hashed_json(goal_dir, "estimand_protocol", estimand)
    weighting = _weighting_contract()
    _write_hashed_json(goal_dir, "market_weighting_contract", weighting)
    sequential = _sequential_protocol(config)
    _write_hashed_json(goal_dir, "sequential_validation_protocol", sequential)

    development_rows = [
        {
            **row,
            "lineage": "development",
            "development_evidence_only": True,
            "unseen_validation_eligible": False,
            "future_shadow_eligible": False,
            "promotion_evidence_eligible": False,
        }
        for row in normalized
    ]
    immutable_path = goal_dir / "immutable_development_rows.jsonl"
    _write_jsonl(immutable_path, development_rows)
    _write_sha_descriptor(immutable_path)
    development_path = goal_dir / "development_corpus_rows.jsonl"
    _write_jsonl(development_path, development_rows)
    _write_sha_descriptor(development_path)
    targets = {
        "schema_version": "bigan-v8-immutable-development-targets-v1",
        "row_count": len(development_rows),
        "row_content_sha256": sha256_file(immutable_path),
        "target_semantics": "selected_side_win_binary_for_hts",
        "target_outcome_available_only_post_resolution": True,
        "source_artifacts": _source_artifact_audit(development_rows),
        "source_artifact_hashes_verified": _source_hashes_verified(development_rows),
        **safety_fields(),
    }
    _write_json(goal_dir / "immutable_development_targets_manifest.json", targets)
    quality = _development_quality(development_rows, scope)
    _write_json(goal_dir / "development_corpus_quality_report.json", quality)
    development_manifest = {
        "schema_version": "bigan-v8-estimand-development-corpus-manifest-v1",
        "development_rows": _descriptor(development_path),
        "immutable_development_rows": _descriptor(immutable_path),
        "quality_report": _descriptor(goal_dir / "development_corpus_quality_report.json"),
        "row_count": len(development_rows),
        "market_count": len({row["market_id"] for row in development_rows}),
        "candidate_scope": scope,
        "full_row_content_frozen": True,
        "prior_negative_evidence_preserved": True,
        **safety_fields(),
    }
    _write_json(goal_dir / "development_corpus_manifest.json", development_manifest)
    state = {
        "schema_version": "bigan-v8-estimand-reformulation-state-v1",
        "goal_status": "IN_PROGRESS",
        "phase": "development_corpus_frozen",
        "final_state": None,
        "pre_promotion_readiness_complete": False,
        "candidate_scope": scope,
        **safety_fields(),
    }
    _write_hashed_json(goal_dir, "initial_goal_state", state)
    return {
        "goal_dir": goal_dir,
        "candidate_scope": scope,
        "development_row_count": len(development_rows),
        "development_market_count": len({row["market_id"] for row in development_rows}),
    }


def develop_probability_candidates(goal_dir: Path | str) -> dict[str, Any]:
    goal_dir = Path(goal_dir).resolve()
    _verify_named_hash(goal_dir, "initial_goal_configuration")
    _verify_named_hash(goal_dir, "estimand_protocol")
    rows = _load_jsonl(goal_dir / "immutable_development_rows.jsonl")
    _verify_sha_descriptor(goal_dir / "immutable_development_rows.jsonl")
    hts_rows = [row for row in rows if row["action_family"] == SUPPORTED_FAMILY]
    if len(hts_rows) < 100:
        raise ValueError("insufficient HTS development rows")
    specs = _candidate_specs()
    protocol = {
        "schema_version": "bigan-v8-probability-candidate-search-protocol-v1",
        "candidate_count": len(specs),
        "maximum_candidate_count": MAXIMUM_CANDIDATE_COUNT,
        "candidate_specifications": specs,
        "selection_data": "immutable_development_rows_only",
        "selection_metrics": [
            "market_weighted_brier_score",
            "market_weighted_log_loss",
            "worst_fold_brier_score",
            "calibration_slope_distance_from_1",
            "parameter_count",
            "candidate_name",
        ],
        "group_by": "source_run_id",
        "cluster_by": "market_id",
        "chronology_preserved_where_feasible": True,
        "fresh_validation_used_for_selection": False,
        **safety_fields(),
    }
    _write_hashed_json(goal_dir, "candidate_search_protocol", protocol)
    reports = []
    for spec in specs:
        reports.append(_cross_validate_candidate(hts_rows, spec))
    reports.sort(key=_candidate_ranking_key)
    order = [
        report["candidate_name"]
        for report in reports
        if report["candidate_specification"]["selectable_for_confirmatory_validation"]
    ]
    report = {
        "schema_version": "bigan-v8-probability-candidate-development-v1",
        "candidate_scope": "HTS_ONLY",
        "development_row_count": len(hts_rows),
        "development_market_count": len({row["market_id"] for row in hts_rows}),
        "market_weighted_primary_ranking": True,
        "candidate_reports": reports,
        "validation_round_candidate_order": order[:MAXIMUM_VALIDATION_ROUNDS],
        "fresh_validation_used_for_selection": False,
        "selected_candidate_name": order[0],
        **safety_fields(),
    }
    _write_json(goal_dir / "candidate_development_report.json", report)
    _write_json(goal_dir / "candidate_stability_report.json", {
        "schema_version": "bigan-v8-probability-candidate-stability-v1",
        "candidates": [
            {"candidate_name": item["candidate_name"], "folds": item["fold_metrics"], "parameter_stability": item["parameter_stability"]}
            for item in reports
        ],
        **safety_fields(),
    })
    selected_spec = next(spec for spec in specs if spec["candidate_name"] == order[0])
    selected_contract = _fit_contract(hts_rows, selected_spec)
    selected_contract.update(
        {
            "candidate_scope": "HTS_ONLY",
            "readiness_scope": "HTS_ONLY",
            "supported_action_families": [SUPPORTED_FAMILY],
            "unsupported_action_families": [UNSUPPORTED_FAMILY],
            "unsupported_action_behavior": "fail_closed_without_calibrated_ev",
            "candidate_round": 1,
            "selected_from_development_only": True,
            **safety_fields(),
        }
    )
    _write_hashed_json(goal_dir, "selected_candidate_contract", selected_contract)
    return {
        "selected_candidate_name": order[0],
        "validation_round_candidate_order": order[:MAXIMUM_VALIDATION_ROUNDS],
        "candidate_development_report": goal_dir / "candidate_development_report.json",
    }


def freeze_and_evaluate_validation_round(
    goal_dir: Path | str,
    *,
    round_number: int,
    fresh_rows_path: Path | str,
    fresh_quality_report_path: Path | str | None = None,
) -> dict[str, Any]:
    goal_dir = Path(goal_dir).resolve()
    if round_number not in range(1, MAXIMUM_VALIDATION_ROUNDS + 1):
        raise ValueError("round_number must be 1..3")
    round_dir = goal_dir / f"round_{round_number}"
    if round_dir.exists():
        raise FileExistsError(f"validation round already frozen: {round_dir}")
    development_report = _load_json(goal_dir / "candidate_development_report.json")
    candidate_name = development_report["validation_round_candidate_order"][round_number - 1]
    specs = {spec["candidate_name"]: spec for spec in _candidate_specs()}
    source_fresh_rows = _load_jsonl(Path(fresh_rows_path).resolve())
    normalized, rejected = _normalize_development_rows(source_fresh_rows)
    if rejected:
        raise ValueError(f"fresh validation has {len(rejected)} invalid rows")
    normalized = [row for row in normalized if row["action_family"] == SUPPORTED_FAMILY]
    development_rows = _development_rows_for_round(goal_dir, round_number)
    excluded = _load_json(goal_dir / "initial_excluded_evidence_manifest.json")
    overlap = _overlap_report(development_rows, normalized, excluded)
    support = _validation_support(normalized)
    config = _load_json(goal_dir / "initial_goal_configuration.json")
    support_passed = (
        support["row_count"] >= config["minimum_validation_rows_per_round"]
        and support["market_count"] >= config["minimum_validation_markets_per_round"]
        and support["side_counts"].get("UP", 0) >= config["minimum_validation_up_rows"]
        and support["side_counts"].get("DOWN", 0) >= config["minimum_validation_down_rows"]
        and support["resolved_outcome_market_counts"].get("UP", 0) >= config["minimum_resolved_up_markets"]
        and support["resolved_outcome_market_counts"].get("DOWN", 0) >= config["minimum_resolved_down_markets"]
        and support["market_count"] >= config["minimum_hts_markets"]
    )
    chronology_passed = (
        bool(development_rows)
        and bool(normalized)
        and min(row["decision_ts"] for row in normalized)
        > max(row["decision_ts"] for row in development_rows)
    )
    causality_violations = [
        row["row_identity"] for row in normalized if row["max_input_ts"] > row["decision_ts"]
    ]
    source_hashes_verified = _source_hashes_verified(normalized)
    split_passed = (
        support_passed
        and chronology_passed
        and not any(overlap.values())
        and not causality_violations
        and source_hashes_verified
    )
    round_dir.mkdir(parents=True)
    dev_path = round_dir / f"round_{round_number}_development_rows.jsonl"
    validation_path = round_dir / f"round_{round_number}_unseen_validation_rows.jsonl"
    _write_jsonl(dev_path, development_rows)
    _write_sha_descriptor(dev_path)
    _write_jsonl(validation_path, normalized)
    _write_sha_descriptor(validation_path)
    split = {
        "schema_version": "bigan-v8-probability-validation-split-v1",
        "round_number": round_number,
        "candidate_name": candidate_name,
        "development_rows": _descriptor(dev_path),
        "unseen_validation_rows": _descriptor(validation_path),
        "external_fresh_source": _descriptor(Path(fresh_rows_path).resolve()),
        "external_quality_report": (
            _descriptor(Path(fresh_quality_report_path).resolve())
            if fresh_quality_report_path else None
        ),
        "strict_chronology_passed": chronology_passed,
        "support": support,
        "support_gate_passed": support_passed,
        "source_artifact_hashes_verified": source_hashes_verified,
        "split_gate_passed": split_passed,
        "complete_row_content_hashes_frozen": True,
        **safety_fields(),
    }
    _write_hashed_json(round_dir, f"round_{round_number}_split_manifest", split)
    leakage = {
        "schema_version": "bigan-v8-probability-validation-leakage-v1",
        "round_number": round_number,
        "overlap": overlap,
        "causality_violation_row_ids": causality_violations,
        "source_artifact_hashes_verified": source_hashes_verified,
        "validation_outcomes_used_during_candidate_selection": False,
        "leakage_report_passed": split_passed,
        **safety_fields(),
    }
    _write_json(round_dir / f"round_{round_number}_leakage_report.json", leakage)
    if not split_passed:
        return {"round_number": round_number, "split_gate_passed": False, "evaluated": False, "support": support}

    marker_path = round_dir / f"round_{round_number}_evaluation_started.json"
    _write_json(marker_path, {
        "round_number": round_number,
        "evaluation_attempt_number": 1,
        "candidate_contract_sha256": sha256_file(goal_dir / "selected_candidate_contract.json") if round_number == 1 else None,
        "split_manifest_sha256": sha256_file(round_dir / f"round_{round_number}_split_manifest.json"),
        "exactly_once": True,
    })
    evaluation = _evaluate_candidate(
        development_rows,
        normalized,
        specs[candidate_name],
        config,
        round_number=round_number,
    )
    _write_json(round_dir / f"round_{round_number}_fit_report.json", evaluation["fit_report"])
    _write_json(round_dir / f"round_{round_number}_fresh_validation_report.json", evaluation["validation_report"])
    _write_json(round_dir / f"round_{round_number}_residual_and_calibration_diagnostics.json", evaluation["diagnostics"])
    if evaluation["all_gates_passed"]:
        artifact = _frozen_artifact(goal_dir, round_dir, evaluation, candidate_name)
        _write_hashed_json(goal_dir, "frozen_diagnostic_artifact", artifact)
    else:
        promoted_to_development = [
            {
                **row,
                "lineage": "development_after_failed_validation",
                "development_evidence_only": True,
                "unseen_validation_eligible": False,
                "future_shadow_eligible": False,
                "promotion_evidence_eligible": False,
            }
            for row in normalized
        ]
        _write_jsonl(round_dir / f"round_{round_number}_failed_wave_development_rows.jsonl", promoted_to_development)
        _write_sha_descriptor(round_dir / f"round_{round_number}_failed_wave_development_rows.jsonl")
    _update_validation_rounds_manifest(goal_dir)
    return {
        "round_number": round_number,
        "candidate_name": candidate_name,
        "split_gate_passed": True,
        "evaluated": True,
        "all_gates_passed": evaluation["all_gates_passed"],
        "blocking_reason_codes": evaluation["blocking_reason_codes"],
        "validation_metrics": evaluation["validation_report"]["probability_metrics"]["candidate"],
    }


def finalize_estimand_reformulation_goal(
    goal_dir: Path | str,
    *,
    stop_reason_codes: list[str] | None = None,
) -> dict[str, Any]:
    goal_dir = Path(goal_dir).resolve()
    rounds = _load_json(goal_dir / "validation_rounds_manifest.json") if (goal_dir / "validation_rounds_manifest.json").exists() else {"rounds": []}
    artifact_path = goal_dir / "frozen_diagnostic_artifact.json"
    shadow_path = goal_dir / "future_shadow_evaluation_report.json"
    artifact_exists = artifact_path.exists()
    shadow = _load_json(shadow_path) if shadow_path.exists() else None
    ready = bool(
        artifact_exists
        and shadow
        and shadow.get("future_shadow_all_gates_passed") is True
    )
    final_state = "PRE_PROMOTION_READY" if ready else "PRE_PROMOTION_BLOCKED"
    blockers = list(stop_reason_codes or [])
    if not artifact_exists:
        blockers.append("no_validation_round_passed_all_frozen_gates")
    if artifact_exists and not shadow:
        blockers.append("future_unseen_shadow_not_completed")
    if shadow and not shadow.get("future_shadow_all_gates_passed"):
        blockers.append("future_unseen_shadow_gate_failed")
    report = {
        "schema_version": "bigan-v8-estimand-pre-promotion-readiness-v1",
        "final_state": final_state,
        "pre_promotion_readiness_complete": ready,
        "readiness_scope": "HTS_ONLY",
        "full_action_family_readiness": False,
        "sell_before_close_calibration_supported": False,
        "blocking_reason_codes": sorted(set(blockers)),
        "prior_negative_evidence_preserved": True,
        "previous_direct_return_regression_route_not_assumed_valid": True,
        "validation_round_history": rounds.get("rounds", []),
        "frozen_diagnostic_artifact": _descriptor(artifact_path) if artifact_exists else None,
        "future_shadow_results": shadow,
        **safety_fields(),
    }
    _write_json(goal_dir / "pre_promotion_readiness_report.json", report)
    _write_text(goal_dir / "pre_promotion_readiness_report.md", _report_markdown(report))
    _write_json(goal_dir / "pre_promotion_goal_state.json", {
        "final_state": final_state,
        "pre_promotion_readiness_complete": ready,
        "readiness_scope": "HTS_ONLY",
        **safety_fields(),
    })
    _ensure_placeholder_final_artifacts(goal_dir)
    artifacts = [
        {"relative_path": str(path.relative_to(goal_dir)), "sha256": sha256_file(path)}
        for path in sorted(goal_dir.rglob("*"))
        if path.is_file() and path.name not in {"pre_promotion_readiness_manifest.json", "pre_promotion_readiness_manifest.sha256"}
    ]
    manifest = {
        "schema_version": "bigan-v8-estimand-pre-promotion-manifest-v1",
        "final_state": final_state,
        "pre_promotion_readiness_complete": ready,
        "readiness_scope": "HTS_ONLY",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "manifest_self_hash_embedded": False,
        "manifest_hash_descriptor_external": True,
        **safety_fields(),
    }
    manifest_path = goal_dir / "pre_promotion_readiness_manifest.json"
    _write_json(manifest_path, manifest)
    _write_sha_descriptor(manifest_path)
    return {
        "final_state": final_state,
        "pre_promotion_readiness_complete": ready,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "blocking_reason_codes": report["blocking_reason_codes"],
    }


def evaluate_future_unseen_shadow(
    goal_dir: Path | str,
    *,
    shadow_rows_paths: tuple[Path | str, ...],
) -> dict[str, Any]:
    goal_dir = Path(goal_dir).resolve()
    if len(shadow_rows_paths) < 2:
        raise ValueError("at least two future shadow windows are required")
    artifact_path = goal_dir / "frozen_diagnostic_artifact.json"
    _verify_sha_descriptor(artifact_path)
    artifact = _load_json(artifact_path)
    artifact_sha256 = sha256_file(artifact_path)
    contract = artifact["probability_model_contract"]
    prior_rows = _load_jsonl(goal_dir / "immutable_development_rows.jsonl")
    for round_dir in sorted(goal_dir.glob("round_*")):
        prior_rows.extend(
            row
            for path in round_dir.glob("round_*_unseen_validation_rows.jsonl")
            for row in _load_jsonl(path)
        )
    prior_markets = {row["market_id"] for row in prior_rows}
    prior_conditions = {row.get("condition_id", row["market_id"]) for row in prior_rows}
    prior_runs = {row["source_run_id"] for row in prior_rows}
    prior_max_ts = max(row["decision_ts"] for row in prior_rows)
    window_reports = []
    combined: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    seen_markets: set[str] = set()
    seen_conditions: set[str] = set()
    seen_runs: set[str] = set()
    for index, source_path in enumerate(shadow_rows_paths, start=1):
        rows, rejected = _normalize_development_rows(
            _load_jsonl(Path(source_path).resolve())
        )
        rows = [row for row in rows if row["action_family"] == SUPPORTED_FAMILY]
        markets = {row["market_id"] for row in rows}
        conditions = {row.get("condition_id", row["market_id"]) for row in rows}
        runs = {row["source_run_id"] for row in rows}
        overlap = {
            "prior_market_ids": sorted(markets & prior_markets),
            "prior_condition_ids": sorted(conditions & prior_conditions),
            "prior_run_ids": sorted(runs & prior_runs),
            "other_shadow_market_ids": sorted(markets & seen_markets),
            "other_shadow_condition_ids": sorted(conditions & seen_conditions),
            "other_shadow_run_ids": sorted(runs & seen_runs),
        }
        chronology_passed = bool(rows) and min(row["decision_ts"] for row in rows) > prior_max_ts
        window_passed = (
            not rejected
            and chronology_passed
            and not any(overlap.values())
            and all(row["max_input_ts"] <= row["decision_ts"] for row in rows)
            and _source_hashes_verified(rows)
        )
        shadow_path = goal_dir / f"future_shadow_window_{index}_rows.jsonl"
        _write_jsonl(shadow_path, rows)
        _write_sha_descriptor(shadow_path)
        predictions = [_predict_probability(row, contract) for row in rows]
        for row, probability in zip(rows, predictions, strict=True):
            decision_rows.append(
                {
                    "row_identity": row["row_identity"],
                    "market_id": row["market_id"],
                    "decision_ts": row["decision_ts"],
                    "selected_action": row["selected_action"],
                    "artifact_applicability": "supported_hts",
                    "frozen_artifact_sha256": artifact_sha256,
                    "calibrated_selected_side_probability": probability,
                    "decision_time_expected_execution_cost": row[
                        "decision_time_expected_execution_cost_per_unit"
                    ],
                    "derived_expected_net_return": probability
                    - float(row["decision_time_features"]["execution_price"])
                    - float(row["decision_time_expected_execution_cost_per_unit"]),
                    **safety_fields(),
                }
            )
        metrics = _probability_metrics(rows, predictions) if rows else None
        window_reports.append(
            {
                "window_number": index,
                "source": _descriptor(Path(source_path).resolve()),
                "immutable_rows": _descriptor(shadow_path),
                "row_count": len(rows),
                "market_count": len(markets),
                "strict_chronology_passed": chronology_passed,
                "overlap": overlap,
                "source_artifact_hashes_verified": _source_hashes_verified(rows),
                "window_provenance_passed": window_passed,
                "probability_metrics": metrics,
            }
        )
        combined.extend(rows)
        seen_markets.update(markets)
        seen_conditions.update(conditions)
        seen_runs.update(runs)
        prior_max_ts = max(prior_max_ts, max((row["decision_ts"] for row in rows), default=prior_max_ts))
    _write_jsonl(goal_dir / "future_shadow_decisions.jsonl", decision_rows)
    combined_predictions = [_predict_probability(row, contract) for row in combined]
    config = _load_json(goal_dir / "initial_goal_configuration.json")
    constant = [_weighted_win_rate(prior_rows)] * len(combined)
    raw = [float(row["decision_time_features"]["selected_side_probability"]) for row in combined]
    legacy = [_legacy_probability(row) for row in combined]
    probability = {
        "candidate": _probability_metrics(combined, combined_predictions),
        "constant_baseline": _probability_metrics(combined, constant),
        "raw_selected_side_market_probability_baseline": _probability_metrics(combined, raw),
        "legacy_o_score_probability_baseline": _probability_metrics(combined, legacy),
    }
    ev = {
        "candidate": _ev_metrics(combined, _ev_predictions(combined, combined_predictions)),
        "constant_baseline": _ev_metrics(combined, _ev_predictions(combined, constant)),
        "raw_probability_minus_price_baseline": _ev_metrics(combined, _ev_predictions(combined, raw)),
        "legacy_o_score_ev_baseline": _ev_metrics(combined, _ev_predictions(combined, legacy)),
    }
    relative = _relative_improvements(probability, ev)
    bootstrap = _bootstrap_improvements(
        combined,
        combined_predictions,
        constant,
        raw,
        legacy,
        int(config["bootstrap_samples"]),
        float(config["per_round_bootstrap_confidence_level"]),
        int(config["statistical_random_seed"]) + 100,
    )
    candidate_metrics = probability["candidate"]
    calibration_passed = (
        candidate_metrics["calibration_slope"] is not None
        and float(config["calibration_slope_minimum"])
        <= candidate_metrics["calibration_slope"]
        <= float(config["calibration_slope_maximum"])
        and candidate_metrics["calibration_intercept"] is not None
        and abs(candidate_metrics["calibration_intercept"])
        <= float(config["maximum_absolute_calibration_intercept"])
    )
    relative_passed = all(
        item["brier_relative_improvement"]
        >= float(config["minimum_brier_score_improvement"])
        and item["log_loss_relative_improvement"]
        >= float(config["minimum_log_loss_improvement"])
        and item["ev_mae_relative_improvement"]
        >= float(config["minimum_relative_mae_improvement"])
        and item["ev_mse_relative_improvement"]
        >= float(config["minimum_relative_mse_improvement"])
        for item in relative.values()
    )
    bootstrap_passed = all(
        comparison[metric]["confidence_interval_lower"]
        >= float(config["bootstrap_improvement_lower_bound"])
        for comparison in bootstrap["comparisons"].values()
        for metric in ("brier", "log_loss", "ev_mae", "ev_mse")
    )
    support_passed = (
        len(combined) >= int(config["minimum_total_shadow_rows"])
        and len({row["market_id"] for row in combined})
        >= int(config["minimum_total_shadow_markets"])
        and len(window_reports) >= int(config["required_future_shadow_window_count"])
    )
    provenance_passed = all(item["window_provenance_passed"] for item in window_reports)
    artifact_identity_passed = bool(decision_rows) and {
        row["frozen_artifact_sha256"] for row in decision_rows
    } == {artifact_sha256}
    all_passed = (
        support_passed
        and provenance_passed
        and artifact_identity_passed
        and calibration_passed
        and relative_passed
        and bootstrap_passed
    )
    blockers = []
    for passed, reason in (
        (support_passed, "future_shadow_support_gate_failed"),
        (provenance_passed, "future_shadow_provenance_gate_failed"),
        (artifact_identity_passed, "future_shadow_artifact_identity_gate_failed"),
        (calibration_passed, "future_shadow_calibration_gate_failed"),
        (relative_passed, "future_shadow_relative_improvement_gate_failed"),
        (bootstrap_passed, "future_shadow_market_bootstrap_gate_failed"),
    ):
        if not passed:
            blockers.append(reason)
    report = {
        "schema_version": "bigan-v8-probability-future-shadow-evaluation-v1",
        "future_shadow_all_gates_passed": all_passed,
        "blocking_reason_codes": blockers,
        "window_reports": window_reports,
        "combined_row_count": len(combined),
        "combined_market_count": len({row["market_id"] for row in combined}),
        "unresolved_official_settlement_count": 0,
        "probability_metrics": probability,
        "derived_ev_metrics": ev,
        "relative_baseline_improvements": relative,
        "market_bootstrap_confidence_intervals": bootstrap,
        "calibration_gate_passed": calibration_passed,
        "support_gate_passed": support_passed,
        "provenance_gate_passed": provenance_passed,
        "artifact_hash_identity_passed": artifact_identity_passed,
        "pnl_diagnostic_only": True,
        **safety_fields(),
    }
    _write_json(goal_dir / "future_shadow_evaluation_report.json", report)
    manifest = {
        "schema_version": "bigan-v8-probability-future-shadow-manifest-v1",
        "frozen_artifact_sha256": artifact_sha256,
        "window_count": len(window_reports),
        "windows": window_reports,
        "decisions": _descriptor(goal_dir / "future_shadow_decisions.jsonl"),
        "evaluation_report": _descriptor(
            goal_dir / "future_shadow_evaluation_report.json"
        ),
        **safety_fields(),
    }
    _write_json(goal_dir / "future_shadow_manifest.json", manifest)
    return {
        "future_shadow_all_gates_passed": all_passed,
        "blocking_reason_codes": blockers,
        "combined_row_count": len(combined),
        "combined_market_count": len({row["market_id"] for row in combined}),
        "frozen_artifact_sha256": artifact_sha256,
    }


def _estimand_protocol(scope: str) -> dict[str, Any]:
    return {
        "schema_version": "bigan-v8-probability-first-estimand-v1",
        "candidate_scope": scope,
        "readiness_scope": scope,
        "model_output_semantics": "selected_side_win_probability",
        "target_semantics": "one_when_resolved_outcome_equals_selected_side_else_zero",
        "target_outcome_available_only_post_resolution": True,
        "decision_time_model_sees_resolved_outcome": False,
        "ev_derivation_semantics": "probability_minus_execution_price_minus_decision_time_cost",
        "expected_net_return_per_unit_formula": "calibrated_selected_side_probability - execution_price - decision_time_expected_execution_cost_per_unit",
        "execution_cost_contract": {
            "contract_id": "decision_time_execution_cost_v1",
            "fixed_fee_per_unit": 0.001,
            "spread_cost": "spread_bps / 20000",
            "queue_impact_cost": "(1 - queue_fill_proxy) * 0.002",
            "staleness_cost": "min(book_staleness_ms / 1000, 1) * 0.001",
            "maximum_total_cost_per_unit": 0.05,
            "execution_cost_subtracted_exactly_once": True,
            "decision_time_fields_only": True,
        },
        "supported_action_families": [SUPPORTED_FAMILY],
        "unsupported_action_families": [UNSUPPORTED_FAMILY],
        "unsupported_action_behavior": "fail_closed_without_calibrated_ev",
        "unsupported_action_reason_code": "unsupported_action_family_sell_before_close",
        **safety_fields(),
    }


def _weighting_contract() -> dict[str, Any]:
    return {
        "schema_version": "bigan-v8-market-balanced-weighting-v1",
        "primary_weighting": "each_market_total_weight_one",
        "row_weight_formula": "1 / rows_in_market",
        "unweighted_metrics_also_reported": True,
        "bootstrap_resampling_unit": "market_id",
        "condition_level_used_when_available": True,
        **safety_fields(),
    }


def _sequential_protocol(config: EstimandReformulationConfig) -> dict[str, Any]:
    return {
        "schema_version": "bigan-v8-sequential-validation-protocol-v1",
        "maximum_validation_rounds": MAXIMUM_VALIDATION_ROUNDS,
        "per_round_bootstrap_confidence_level": PER_ROUND_CONFIDENCE_LEVEL,
        "multiplicity_method": "bonferroni_three_rounds",
        "candidate_ladder_frozen_before_round_one": True,
        "failed_wave_becomes_development_only": True,
        "failed_wave_never_reused_as_unseen": True,
        "exactly_once_evaluation_required": True,
        "thresholds_immutable_across_rounds": True,
        "collection_budget": {
            "maximum_total_windows": config.maximum_total_validation_collection_windows,
            "window_seconds": config.validation_collection_window_seconds,
            "maximum_wall_clock_seconds": config.maximum_wall_clock_seconds,
        },
        "validation_thresholds": {
            "minimum_rows": config.minimum_validation_rows_per_round,
            "minimum_markets": config.minimum_validation_markets_per_round,
            "minimum_up_rows": config.minimum_validation_up_rows,
            "minimum_down_rows": config.minimum_validation_down_rows,
            "minimum_resolved_up_markets": config.minimum_resolved_up_markets,
            "minimum_resolved_down_markets": config.minimum_resolved_down_markets,
            "minimum_hts_markets": config.minimum_hts_markets,
            "minimum_relative_mae_improvement": config.minimum_relative_mae_improvement,
            "minimum_relative_mse_improvement": config.minimum_relative_mse_improvement,
            "minimum_brier_score_improvement": config.minimum_brier_score_improvement,
            "minimum_log_loss_improvement": config.minimum_log_loss_improvement,
            "bootstrap_improvement_lower_bound": config.bootstrap_improvement_lower_bound,
            "calibration_slope_range": [config.calibration_slope_minimum, config.calibration_slope_maximum],
            "maximum_absolute_calibration_intercept": config.maximum_absolute_calibration_intercept,
        },
        "future_shadow_thresholds": {
            "required_window_count": config.required_future_shadow_window_count,
            "minimum_total_rows": config.minimum_total_shadow_rows,
            "minimum_total_markets": config.minimum_total_shadow_markets,
        },
        **safety_fields(),
    }


def _normalize_development_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for source in rows:
        row = json.loads(json.dumps(source))
        reasons: list[str] = []
        side = row.get("selected_side")
        family = row.get("action_family")
        outcome = (row.get("target_provenance") or {}).get("resolved_outcome")
        features = row.get("decision_time_features") or {}
        if side not in {"UP", "DOWN"}:
            reasons.append("invalid_selected_side")
        if family not in {SUPPORTED_FAMILY, UNSUPPORTED_FAMILY}:
            reasons.append("invalid_action_family")
        if outcome not in {"UP", "DOWN"}:
            reasons.append("missing_official_resolved_outcome")
        for field in ("selected_side_probability", "execution_price"):
            value = features.get(field)
            if not _finite(value) or not 0.0 <= float(value) <= 1.0:
                reasons.append(f"invalid_{field}")
        if not _finite(row.get("decision_ts")) or not _finite(row.get("max_input_ts")):
            reasons.append("missing_causal_timestamps")
        elif float(row["max_input_ts"]) > float(row["decision_ts"]):
            reasons.append("decision_time_causality_violation")
        if reasons:
            rejected.append({"row_identity": row.get("row_identity"), "reason_codes": sorted(set(reasons))})
            continue
        row["condition_id"] = str(row.get("condition_id") or row["market_id"])
        row["selected_side_win_target"] = 1 if outcome == side else 0
        row["decision_time_expected_execution_cost_per_unit"] = _decision_time_cost(features)
        row["derived_target_net_return_per_unit"] = (
            float(row["selected_side_win_target"])
            - float(features["execution_price"])
            - row["decision_time_expected_execution_cost_per_unit"]
        )
        row["model_output_semantics"] = "selected_side_win_probability"
        row["ev_derivation_semantics"] = "probability_minus_execution_price_minus_decision_time_cost"
        row["target_outcome_available_only_post_resolution"] = True
        row["row_content_sha256"] = canonical_json_sha256({k: v for k, v in row.items() if k != "row_content_sha256"})
        accepted.append(row)
    accepted.sort(key=lambda row: (float(row["decision_ts"]), row["market_id"], row["row_identity"]))
    return accepted, rejected


def _decision_time_cost(features: dict[str, Any]) -> float:
    spread = max(float(features.get("spread_bps", 0.0)), 0.0) / 20000.0
    queue = min(max(float(features.get("queue_fill_proxy", 1.0)), 0.0), 1.0)
    staleness = max(float(features.get("book_staleness_ms", 0.0)), 0.0)
    return min(0.05, 0.001 + spread + (1.0 - queue) * 0.002 + min(staleness / 1000.0, 1.0) * 0.001)


def _candidate_specs() -> list[dict[str, Any]]:
    common = {
        "applicability_scope": "HTS_ONLY",
        "model_output_semantics": "selected_side_win_probability",
        "probability_bounds": [0.01, 0.99],
        "cost_semantics": "derived_after_probability_prediction_subtracted_exactly_once",
        "unsupported_action_behavior": "fail_closed_without_calibrated_ev",
    }
    return [
        {**common, "candidate_name": "raw_selected_side_market_probability", "model_family": "identity_probability", "features": ["selected_side_probability"], "regularization": None, "parameter_count": 0, "monotonicity": "identity", "selectable_for_confirmatory_validation": False, "diagnostic_baseline_only": True},
        {**common, "candidate_name": "constant_development_win_rate", "model_family": "weighted_constant_probability", "features": [], "regularization": None, "parameter_count": 1, "monotonicity": "constant", "selectable_for_confirmatory_validation": False, "diagnostic_baseline_only": True},
        {**common, "candidate_name": "platt_selected_side_probability", "model_family": "regularized_logistic", "features": ["logit_selected_side_probability"], "regularization": 10.0, "regularization_prior": [0.0, 1.0], "parameter_count": 2, "monotonicity": "positive_probability_slope", "selectable_for_confirmatory_validation": True},
        {**common, "candidate_name": "beta_selected_side_probability", "model_family": "regularized_logistic", "features": ["log_selected_side_probability", "log_one_minus_selected_side_probability"], "regularization": 10.0, "regularization_prior": [0.0, 1.0, -1.0], "parameter_count": 3, "monotonicity": "bounded_beta_calibration", "selectable_for_confirmatory_validation": True},
        {**common, "candidate_name": "logistic_probability_o_score_margin", "model_family": "regularized_logistic", "features": ["logit_selected_side_probability", "canonical_o_action_score", "action_score_margin"], "regularization": 10.0, "regularization_prior": [0.0, 1.0, 0.0, 0.0], "parameter_count": 4, "monotonicity": "positive_probability_slope", "selectable_for_confirmatory_validation": True},
        {**common, "candidate_name": "side_partially_pooled_logistic", "model_family": "regularized_logistic", "features": ["logit_selected_side_probability", "selected_side_down"], "regularization": 10.0, "regularization_prior": [0.0, 1.0, 0.0], "parameter_count": 3, "monotonicity": "positive_probability_slope", "selectable_for_confirmatory_validation": True},
        {**common, "candidate_name": "side_horizon_partially_pooled_logistic", "model_family": "regularized_logistic", "features": ["logit_selected_side_probability", "selected_side_down", "horizon_15m"], "regularization": 10.0, "regularization_prior": [0.0, 1.0, 0.0, 0.0], "parameter_count": 4, "monotonicity": "positive_probability_slope", "selectable_for_confirmatory_validation": True},
        {**common, "candidate_name": "monotonic_quadratic_probability", "model_family": "regularized_logistic", "features": ["selected_side_probability", "selected_side_probability_squared"], "regularization": 10.0, "regularization_prior": [-2.0, 4.0, 0.0], "parameter_count": 3, "monotonicity": "bounded_low_degree_checked_on_grid", "selectable_for_confirmatory_validation": True},
    ]


def _cross_validate_candidate(rows: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    runs = sorted({row["source_run_id"] for row in rows}, key=lambda run: min(r["decision_ts"] for r in rows if r["source_run_id"] == run))
    folds = []
    if len(runs) >= 3:
        for index in range(1, len(runs)):
            train_runs = set(runs[:index])
            valid_run = runs[index]
            train = [row for row in rows if row["source_run_id"] in train_runs]
            valid = [row for row in rows if row["source_run_id"] == valid_run]
            if train and valid:
                folds.append((train, valid))
    if not folds:
        markets = sorted({row["market_id"] for row in rows})
        split = max(1, int(len(markets) * 0.7))
        train_markets = set(markets[:split])
        folds = [([row for row in rows if row["market_id"] in train_markets], [row for row in rows if row["market_id"] not in train_markets])]
    fold_metrics = []
    parameters = []
    for train, valid in folds:
        contract = _fit_contract(train, spec)
        predictions = [_predict_probability(row, contract) for row in valid]
        metrics = _probability_metrics(valid, predictions)
        fold_metrics.append(metrics)
        parameters.append(contract.get("parameters", []))
    return {
        "candidate_name": spec["candidate_name"],
        "candidate_specification": spec,
        "fold_count": len(folds),
        "fold_metrics": fold_metrics,
        "market_weighted_brier_score": statistics.mean(item["market_weighted_brier_score"] for item in fold_metrics),
        "market_weighted_log_loss": statistics.mean(item["market_weighted_log_loss"] for item in fold_metrics),
        "worst_fold_brier_score": max(item["market_weighted_brier_score"] for item in fold_metrics),
        "calibration_slope_distance_from_1": statistics.mean(abs((item["calibration_slope"] or 0.0) - 1.0) for item in fold_metrics),
        "parameter_count": spec["parameter_count"],
        "parameter_stability": _parameter_stability(parameters),
    }


def _candidate_ranking_key(report: dict[str, Any]) -> tuple[Any, ...]:
    return (
        report["market_weighted_brier_score"],
        report["market_weighted_log_loss"],
        report["worst_fold_brier_score"],
        report["calibration_slope_distance_from_1"],
        report["parameter_count"],
        report["candidate_name"],
    )


def _fit_contract(rows: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    if spec["model_family"] == "identity_probability":
        return {"candidate_name": spec["candidate_name"], "model_family": spec["model_family"], "parameters": [], "features": spec["features"], "probability_bounds": spec["probability_bounds"]}
    weights = _market_weights(rows)
    targets = [float(row["selected_side_win_target"]) for row in rows]
    if spec["model_family"] == "weighted_constant_probability":
        probability = sum(weight * target for weight, target in zip(weights, targets, strict=True)) / sum(weights)
        return {"candidate_name": spec["candidate_name"], "model_family": spec["model_family"], "parameters": [probability], "features": [], "probability_bounds": spec["probability_bounds"]}
    matrix = [[1.0, *[_feature(row, name) for name in spec["features"]]] for row in rows]
    parameters = _fit_weighted_logistic(
        matrix,
        targets,
        weights,
        float(spec["regularization"]),
        prior=[float(value) for value in spec["regularization_prior"]],
    )
    if "positive_probability_slope" in spec["monotonicity"] and len(parameters) > 1:
        parameters[1] = max(parameters[1], 0.0)
    parameters = [min(max(value, -8.0), 8.0) for value in parameters]
    return {
        "candidate_name": spec["candidate_name"],
        "model_family": spec["model_family"],
        "parameters": parameters,
        "features": spec["features"],
        "probability_bounds": spec["probability_bounds"],
        "regularization": spec["regularization"],
        "finite_and_bounded": all(math.isfinite(value) and abs(value) <= 8.0 for value in parameters),
    }


def _predict_probability(row: dict[str, Any], contract: dict[str, Any]) -> float:
    low, high = contract.get("probability_bounds", [0.01, 0.99])
    family = contract["model_family"]
    if family == "identity_probability":
        value = float(row["decision_time_features"]["selected_side_probability"])
    elif family == "weighted_constant_probability":
        value = float(contract["parameters"][0])
    else:
        vector = [1.0, *[_feature(row, name) for name in contract["features"]]]
        score = sum(a * b for a, b in zip(contract["parameters"], vector, strict=True))
        value = _sigmoid(score)
    return min(max(value, float(low)), float(high))


def _feature(row: dict[str, Any], name: str) -> float:
    features = row["decision_time_features"]
    probability = min(max(float(features["selected_side_probability"]), 1e-6), 1 - 1e-6)
    if name == "logit_selected_side_probability":
        return math.log(probability / (1.0 - probability))
    if name == "log_selected_side_probability":
        return math.log(probability)
    if name == "log_one_minus_selected_side_probability":
        return math.log(1.0 - probability)
    if name == "selected_side_down":
        return float(row["selected_side"] == "DOWN")
    if name == "horizon_15m":
        return float(float(features.get("time_to_close_seconds", 0.0)) > 600.0 or "15m" in str(row.get("market_slug", "")))
    if name == "selected_side_probability_squared":
        return probability * probability
    return float(features[name])


def _fit_weighted_logistic(
    matrix: list[list[float]],
    targets: list[float],
    weights: list[float],
    regularization: float,
    *,
    prior: list[float] | None = None,
) -> list[float]:
    size = len(matrix[0])
    prior = list(prior or [0.0] * size)
    if len(prior) != size:
        raise ValueError("logistic regularization prior size mismatch")
    parameters = prior[:]
    for _ in range(80):
        gradient = [0.0] * size
        hessian = [[0.0] * size for _ in range(size)]
        for vector, target, weight in zip(matrix, targets, weights, strict=True):
            probability = _sigmoid(sum(a * b for a, b in zip(parameters, vector, strict=True)))
            variance = max(probability * (1.0 - probability), 1e-6)
            for i in range(size):
                gradient[i] += weight * (probability - target) * vector[i]
                for j in range(size):
                    hessian[i][j] += weight * variance * vector[i] * vector[j]
        for i in range(1, size):
            gradient[i] += regularization * (parameters[i] - prior[i])
            hessian[i][i] += regularization
        hessian[0][0] += 1e-8
        step = _solve_linear(hessian, gradient)
        parameters = [value - delta for value, delta in zip(parameters, step, strict=True)]
        if max(abs(delta) for delta in step) < 1e-8:
            break
    return parameters


def _evaluate_candidate(development: list[dict[str, Any]], validation: list[dict[str, Any]], spec: dict[str, Any], config: dict[str, Any], *, round_number: int) -> dict[str, Any]:
    contract = _fit_contract(development, spec)
    candidate = [_predict_probability(row, contract) for row in validation]
    weighted_rate = _weighted_win_rate(development)
    constant = [weighted_rate] * len(validation)
    raw = [float(row["decision_time_features"]["selected_side_probability"]) for row in validation]
    legacy = [_legacy_probability(row) for row in validation]
    probability_metrics = {
        "candidate": _probability_metrics(validation, candidate),
        "constant_baseline": _probability_metrics(validation, constant),
        "raw_selected_side_market_probability_baseline": _probability_metrics(validation, raw),
        "legacy_o_score_probability_baseline": _probability_metrics(validation, legacy),
    }
    ev_predictions = {name: _ev_predictions(validation, values) for name, values in {
        "candidate": candidate,
        "constant_baseline": constant,
        "raw_probability_minus_price_baseline": raw,
        "legacy_o_score_ev_baseline": legacy,
    }.items()}
    ev_metrics = {name: _ev_metrics(validation, values) for name, values in ev_predictions.items()}
    relative = _relative_improvements(probability_metrics, ev_metrics)
    bootstrap = _bootstrap_improvements(validation, candidate, constant, raw, legacy, int(config["bootstrap_samples"]), float(config["per_round_bootstrap_confidence_level"]), int(config["statistical_random_seed"]) + round_number)
    candidate_metrics = probability_metrics["candidate"]
    calibration_passed = (
        candidate_metrics["calibration_slope"] is not None
        and float(config["calibration_slope_minimum"]) <= candidate_metrics["calibration_slope"] <= float(config["calibration_slope_maximum"])
        and candidate_metrics["calibration_intercept"] is not None
        and abs(candidate_metrics["calibration_intercept"]) <= float(config["maximum_absolute_calibration_intercept"])
    )
    relative_passed = all(
        item["brier_relative_improvement"] >= float(config["minimum_brier_score_improvement"])
        and item["log_loss_relative_improvement"] >= float(config["minimum_log_loss_improvement"])
        and item["ev_mae_relative_improvement"] >= float(config["minimum_relative_mae_improvement"])
        and item["ev_mse_relative_improvement"] >= float(config["minimum_relative_mse_improvement"])
        for item in relative.values()
    )
    bootstrap_passed = all(
        comparison[metric]["confidence_interval_lower"] >= float(config["bootstrap_improvement_lower_bound"])
        for comparison in bootstrap["comparisons"].values()
        for metric in ("brier", "log_loss", "ev_mae", "ev_mse")
    )
    stability = _leave_one_market_out_stability(development, spec)
    bounds_passed = all(0.01 <= value <= 0.99 for value in candidate)
    concentration_passed = max(Counter(row["market_id"] for row in validation).values()) / len(validation) <= float(config["maximum_per_market_row_share"])
    all_passed = relative_passed and bootstrap_passed and calibration_passed and stability["passed"] and bounds_passed and concentration_passed
    blockers = []
    for passed, reason in (
        (relative_passed, "confirmatory_relative_improvement_gate_failed"),
        (bootstrap_passed, "confirmatory_market_bootstrap_gate_failed"),
        (calibration_passed, "confirmatory_calibration_slope_intercept_gate_failed"),
        (stability["passed"], "confirmatory_parameter_stability_gate_failed"),
        (bounds_passed, "confirmatory_prediction_bounds_gate_failed"),
        (concentration_passed, "confirmatory_market_concentration_gate_failed"),
    ):
        if not passed:
            blockers.append(reason)
    fit_report = {
        "schema_version": "bigan-v8-probability-fit-report-v1",
        "round_number": round_number,
        "candidate_name": spec["candidate_name"],
        "fit_row_count": len(development),
        "fit_market_count": len({row["market_id"] for row in development}),
        "contract": contract,
        "parameter_stability": stability,
        "market_balanced_weighting": True,
        **safety_fields(),
    }
    validation_report = {
        "schema_version": "bigan-v8-probability-fresh-validation-v1",
        "round_number": round_number,
        "candidate_name": spec["candidate_name"],
        "validation_row_count": len(validation),
        "validation_market_count": len({row["market_id"] for row in validation}),
        "probability_metrics": probability_metrics,
        "derived_ev_metrics": ev_metrics,
        "relative_baseline_improvements": relative,
        "market_bootstrap_confidence_intervals": bootstrap,
        "calibration_gate_passed": calibration_passed,
        "relative_improvement_gate_passed": relative_passed,
        "bootstrap_gate_passed": bootstrap_passed,
        "parameter_stability_gate_passed": stability["passed"],
        "prediction_bounds_gate_passed": bounds_passed,
        "market_concentration_gate_passed": concentration_passed,
        "all_confirmatory_gates_passed": all_passed,
        "blocking_reason_codes": blockers,
        "evaluation_attempt_number": 1,
        "uses_validation_labels_for_tuning": False,
        **safety_fields(),
    }
    diagnostics = {
        "schema_version": "bigan-v8-probability-residual-calibration-diagnostics-v1",
        "round_number": round_number,
        "calibration_bins": _calibration_bins(validation, candidate),
        "side_metrics": _group_probability_metrics(validation, candidate, lambda row: row["selected_side"]),
        "horizon_metrics": _group_probability_metrics(validation, candidate, _horizon),
        "worst_market_error_share": _worst_market_error_share(validation, candidate),
        **safety_fields(),
    }
    return {"fit_report": fit_report, "validation_report": validation_report, "diagnostics": diagnostics, "all_gates_passed": all_passed, "blocking_reason_codes": blockers, "contract": contract}


def _probability_metrics(rows: list[dict[str, Any]], predictions: list[float]) -> dict[str, Any]:
    targets = [float(row["selected_side_win_target"]) for row in rows]
    weights = _market_weights(rows)
    total = sum(weights)
    brier = sum(w * (y - p) ** 2 for w, y, p in zip(weights, targets, predictions, strict=True)) / total
    log_loss = -sum(w * (y * math.log(_clip_probability(p)) + (1 - y) * math.log(1 - _clip_probability(p))) for w, y, p in zip(weights, targets, predictions, strict=True)) / total
    slope, intercept = _calibration_slope_intercept(targets, predictions, weights)
    bins = _calibration_bins(rows, predictions)
    return {
        "market_weighted_brier_score": brier,
        "market_weighted_log_loss": log_loss,
        "unweighted_brier_score": statistics.mean((y - p) ** 2 for y, p in zip(targets, predictions, strict=True)),
        "unweighted_log_loss": -statistics.mean(y * math.log(_clip_probability(p)) + (1 - y) * math.log(1 - _clip_probability(p)) for y, p in zip(targets, predictions, strict=True)),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "expected_calibration_error": sum(bin_["weight"] * abs(bin_["observed_rate"] - bin_["mean_prediction"]) for bin_ in bins),
        "maximum_calibration_error": max((abs(bin_["observed_rate"] - bin_["mean_prediction"]) for bin_ in bins), default=0.0),
    }


def _ev_predictions(rows: list[dict[str, Any]], probabilities: list[float]) -> list[float]:
    return [p - float(row["decision_time_features"]["execution_price"]) - float(row["decision_time_expected_execution_cost_per_unit"]) for row, p in zip(rows, probabilities, strict=True)]


def _ev_metrics(rows: list[dict[str, Any]], predictions: list[float]) -> dict[str, Any]:
    targets = [float(row["derived_target_net_return_per_unit"]) for row in rows]
    weights = _market_weights(rows)
    total = sum(weights)
    return {
        "market_weighted_mae": sum(w * abs(y - p) for w, y, p in zip(weights, targets, predictions, strict=True)) / total,
        "market_weighted_mse": sum(w * (y - p) ** 2 for w, y, p in zip(weights, targets, predictions, strict=True)) / total,
        "unweighted_mae": statistics.mean(abs(y - p) for y, p in zip(targets, predictions, strict=True)),
        "unweighted_mse": statistics.mean((y - p) ** 2 for y, p in zip(targets, predictions, strict=True)),
    }


def _relative_improvements(probability: dict[str, dict[str, Any]], ev: dict[str, dict[str, Any]]) -> dict[str, Any]:
    mapping = {
        "constant_baseline": ("constant_baseline", "constant_baseline"),
        "raw_probability_baseline": ("raw_selected_side_market_probability_baseline", "raw_probability_minus_price_baseline"),
        "legacy_o_score_baseline": ("legacy_o_score_probability_baseline", "legacy_o_score_ev_baseline"),
    }
    result = {}
    for name, (prob_name, ev_name) in mapping.items():
        base_prob = probability[prob_name]
        candidate_prob = probability["candidate"]
        base_ev = ev[ev_name]
        candidate_ev = ev["candidate"]
        result[name] = {
            "brier_relative_improvement": _relative(base_prob["market_weighted_brier_score"], candidate_prob["market_weighted_brier_score"]),
            "log_loss_relative_improvement": _relative(base_prob["market_weighted_log_loss"], candidate_prob["market_weighted_log_loss"]),
            "ev_mae_relative_improvement": _relative(base_ev["market_weighted_mae"], candidate_ev["market_weighted_mae"]),
            "ev_mse_relative_improvement": _relative(base_ev["market_weighted_mse"], candidate_ev["market_weighted_mse"]),
        }
    return result


def _bootstrap_improvements(rows: list[dict[str, Any]], candidate: list[float], constant: list[float], raw: list[float], legacy: list[float], samples: int, confidence: float, seed: int) -> dict[str, Any]:
    markets = sorted({row["market_id"] for row in rows})
    indexes = defaultdict(list)
    for index, row in enumerate(rows):
        indexes[row["market_id"]].append(index)
    rng = random.Random(seed)
    baseline_predictions = {"constant_baseline": constant, "raw_probability_baseline": raw, "legacy_o_score_baseline": legacy}
    distributions = {name: {metric: [] for metric in ("brier", "log_loss", "ev_mae", "ev_mse")} for name in baseline_predictions}
    for _ in range(samples):
        sampled = [rng.choice(markets) for _ in markets]
        sampled_indexes = [index for market in sampled for index in indexes[market]]
        sample_rows = [rows[index] for index in sampled_indexes]
        sample_candidate = [candidate[index] for index in sampled_indexes]
        candidate_prob = _probability_metrics(sample_rows, sample_candidate)
        candidate_ev = _ev_metrics(sample_rows, _ev_predictions(sample_rows, sample_candidate))
        for name, values in baseline_predictions.items():
            sample_base = [values[index] for index in sampled_indexes]
            base_prob = _probability_metrics(sample_rows, sample_base)
            base_ev = _ev_metrics(sample_rows, _ev_predictions(sample_rows, sample_base))
            distributions[name]["brier"].append(base_prob["market_weighted_brier_score"] - candidate_prob["market_weighted_brier_score"])
            distributions[name]["log_loss"].append(base_prob["market_weighted_log_loss"] - candidate_prob["market_weighted_log_loss"])
            distributions[name]["ev_mae"].append(base_ev["market_weighted_mae"] - candidate_ev["market_weighted_mae"])
            distributions[name]["ev_mse"].append(base_ev["market_weighted_mse"] - candidate_ev["market_weighted_mse"])
    alpha = (1.0 - confidence) / 2.0
    comparisons = {}
    for name, metrics in distributions.items():
        comparisons[name] = {}
        for metric, values in metrics.items():
            comparisons[name][metric] = {
                "confidence_interval_lower": _quantile(values, alpha),
                "confidence_interval_upper": _quantile(values, 1.0 - alpha),
            }
    return {"resampling_unit": "market_id", "bootstrap_samples": samples, "confidence_level": confidence, "comparisons": comparisons}


def _frozen_artifact(goal_dir: Path, round_dir: Path, evaluation: dict[str, Any], candidate_name: str) -> dict[str, Any]:
    config = _load_json(goal_dir / "initial_goal_configuration.json")
    return {
        "schema_version": "bigan-v8-frozen-probability-ev-diagnostic-artifact-v1",
        "frozen": True,
        "decision_time_safe": True,
        "diagnostic_artifact_only": True,
        "readiness_scope": "HTS_ONLY",
        "supported_action_families": [SUPPORTED_FAMILY],
        "unsupported_action_families": [UNSUPPORTED_FAMILY],
        "unsupported_action_behavior": "fail_closed_without_calibrated_ev",
        "candidate_name": candidate_name,
        "probability_model_contract": evaluation["contract"],
        "model_output_semantics": "selected_side_win_probability",
        "ev_derivation_semantics": "probability_minus_execution_price_minus_decision_time_cost",
        "execution_cost_subtracted_exactly_once": True,
        "estimand_protocol_sha256": sha256_file(goal_dir / "estimand_protocol.json"),
        "goal_configuration_sha256": sha256_file(goal_dir / "initial_goal_configuration.json"),
        "market_weighting_contract_sha256": sha256_file(goal_dir / "market_weighting_contract.json"),
        "candidate_search_protocol_sha256": sha256_file(goal_dir / "candidate_search_protocol.json"),
        "development_rows_sha256": sha256_file(round_dir / f"round_{evaluation['validation_report']['round_number']}_development_rows.jsonl"),
        "validation_rows_sha256": sha256_file(round_dir / f"round_{evaluation['validation_report']['round_number']}_unseen_validation_rows.jsonl"),
        "split_manifest_sha256": sha256_file(round_dir / f"round_{evaluation['validation_report']['round_number']}_split_manifest.json"),
        "validation_metrics": evaluation["validation_report"],
        "source_code_tree_hash": config["source_tree_hash"],
        "future_unseen_shadow_required": True,
        **safety_fields(),
    }


def _development_rows_for_round(goal_dir: Path, round_number: int) -> list[dict[str, Any]]:
    rows = _load_jsonl(goal_dir / "immutable_development_rows.jsonl")
    for previous in range(1, round_number):
        path = goal_dir / f"round_{previous}" / f"round_{previous}_failed_wave_development_rows.jsonl"
        if not path.exists():
            raise ValueError(f"round {previous} did not produce failed-wave development rows")
        _verify_sha_descriptor(path)
        rows.extend(_load_jsonl(path))
    return [row for row in rows if row["action_family"] == SUPPORTED_FAMILY]


def _update_validation_rounds_manifest(goal_dir: Path) -> None:
    rounds = []
    for round_dir in sorted(goal_dir.glob("round_*")):
        report_paths = list(round_dir.glob("round_*_fresh_validation_report.json"))
        if not report_paths:
            continue
        report = _load_json(report_paths[0])
        rounds.append({
            "round_number": report["round_number"],
            "candidate_name": report["candidate_name"],
            "all_confirmatory_gates_passed": report["all_confirmatory_gates_passed"],
            "blocking_reason_codes": report["blocking_reason_codes"],
            "fresh_validation_report": _descriptor(report_paths[0]),
        })
    _write_json(goal_dir / "validation_rounds_manifest.json", {
        "schema_version": "bigan-v8-validation-rounds-manifest-v1",
        "maximum_validation_rounds": MAXIMUM_VALIDATION_ROUNDS,
        "rounds": rounds,
        **safety_fields(),
    })


def _validation_support(rows: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes_by_market: dict[str, str] = {}
    for row in rows:
        outcomes_by_market[row["market_id"]] = row["target_provenance"]["resolved_outcome"]
    return {
        "row_count": len(rows),
        "market_count": len({row["market_id"] for row in rows}),
        "side_counts": dict(Counter(row["selected_side"] for row in rows)),
        "resolved_outcome_row_counts": dict(Counter(row["target_provenance"]["resolved_outcome"] for row in rows)),
        "resolved_outcome_market_counts": dict(Counter(outcomes_by_market.values())),
        "action_family_counts": dict(Counter(row["action_family"] for row in rows)),
    }


def _overlap_report(development: list[dict[str, Any]], validation: list[dict[str, Any]], excluded: dict[str, Any]) -> dict[str, list[str]]:
    def values(rows: list[dict[str, Any]], key: str) -> set[str]:
        return {str(row.get(key) or (row["market_id"] if key == "condition_id" else "")) for row in rows}
    return {
        "market_ids": sorted(values(development, "market_id") & values(validation, "market_id")),
        "condition_ids": sorted(values(development, "condition_id") & values(validation, "condition_id")),
        "source_run_ids": sorted(values(development, "source_run_id") & values(validation, "source_run_id")),
        "row_identities": sorted(values(development, "row_identity") & values(validation, "row_identity")),
        "excluded_market_ids": sorted(set(excluded["all_inspected_market_ids"]) & values(validation, "market_id")),
        "excluded_row_identities": sorted(set(excluded["all_inspected_row_identities"]) & values(validation, "row_identity")),
    }


def _development_quality(rows: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    markets = Counter(row["market_id"] for row in rows)
    outcomes = {row["market_id"]: row["target_provenance"]["resolved_outcome"] for row in rows}
    return {
        "schema_version": "bigan-v8-estimand-development-quality-v1",
        "candidate_scope": scope,
        "row_count": len(rows),
        "market_count": len(markets),
        "unique_condition_count": len({row["condition_id"] for row in rows}),
        "action_family_counts": dict(Counter(row["action_family"] for row in rows)),
        "action_family_market_counts": {family: len({row["market_id"] for row in rows if row["action_family"] == family}) for family in (SUPPORTED_FAMILY, UNSUPPORTED_FAMILY)},
        "side_counts": dict(Counter(row["selected_side"] for row in rows)),
        "resolved_outcome_counts": dict(Counter(outcomes.values())),
        "horizon_counts": dict(Counter(_horizon(row) for row in rows)),
        "maximum_per_market_row_share": max(markets.values()) / len(rows),
        "repeated_fill_market_count": sum(count > 1 for count in markets.values()),
        "causality_violation_count": sum(row["max_input_ts"] > row["decision_ts"] for row in rows),
        "source_artifact_hashes_verified": _source_hashes_verified(rows),
        **safety_fields(),
    }


def _source_artifact_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    artifacts: dict[str, str] = {}
    for row in rows:
        for key, value in row.get("source_lineage", {}).items():
            if key.endswith("_artifact_path") or key.endswith("_manifest_path"):
                expected = row["source_lineage"].get(key.removesuffix("_path") + "_sha256")
                if expected:
                    artifacts[str(value)] = str(expected)
        provenance = row.get("target_provenance", {})
        if provenance.get("source_artifact_path") and provenance.get("source_artifact_sha256"):
            artifacts[str(provenance["source_artifact_path"])] = str(provenance["source_artifact_sha256"])
    return [
        {"path": path, "expected_sha256": expected, "actual_sha256": sha256_file(Path(path)) if Path(path).exists() else None, "verified": Path(path).exists() and sha256_file(Path(path)) == expected}
        for path, expected in sorted(artifacts.items())
    ]


def _source_hashes_verified(rows: list[dict[str, Any]]) -> bool:
    audit = _source_artifact_audit(rows)
    return bool(audit) and all(item["verified"] for item in audit)


def _market_weights(rows: list[dict[str, Any]]) -> list[float]:
    counts = Counter(row["market_id"] for row in rows)
    return [1.0 / counts[row["market_id"]] for row in rows]


def _weighted_win_rate(rows: list[dict[str, Any]]) -> float:
    weights = _market_weights(rows)
    return sum(weight * row["selected_side_win_target"] for weight, row in zip(weights, rows, strict=True)) / sum(weights)


def _legacy_probability(row: dict[str, Any]) -> float:
    score = float(row["decision_time_features"].get("canonical_o_action_score", 0.0))
    return min(max(_sigmoid(score), 0.01), 0.99)


def _calibration_slope_intercept(targets: list[float], predictions: list[float], weights: list[float]) -> tuple[float | None, float | None]:
    if len(set(targets)) < 2 or len({round(value, 12) for value in predictions}) < 2:
        return None, None
    logits = [math.log(_clip_probability(value) / (1.0 - _clip_probability(value))) for value in predictions]
    matrix = [[1.0, value] for value in logits]
    parameters = _fit_weighted_logistic(matrix, targets, weights, 0.01)
    return parameters[1], parameters[0]


def _calibration_bins(rows: list[dict[str, Any]], predictions: list[float]) -> list[dict[str, Any]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, prediction in enumerate(predictions):
        groups[min(int(prediction * 10), 9)].append(index)
    result = []
    for bucket, indexes in sorted(groups.items()):
        market_weights = _market_weights([rows[index] for index in indexes])
        total = sum(market_weights)
        result.append({
            "bucket": f"{bucket / 10:.1f}-{(bucket + 1) / 10:.1f}",
            "row_count": len(indexes),
            "mean_prediction": sum(weight * predictions[index] for weight, index in zip(market_weights, indexes, strict=True)) / total,
            "observed_rate": sum(weight * rows[index]["selected_side_win_target"] for weight, index in zip(market_weights, indexes, strict=True)) / total,
            "weight": len(indexes) / len(rows),
        })
    return result


def _group_probability_metrics(rows: list[dict[str, Any]], predictions: list[float], grouper: Callable[[dict[str, Any]], str]) -> dict[str, Any]:
    indexes: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        indexes[grouper(row)].append(index)
    return {name: _probability_metrics([rows[index] for index in values], [predictions[index] for index in values]) for name, values in sorted(indexes.items())}


def _leave_one_market_out_stability(rows: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    full = _fit_contract(rows, spec).get("parameters", [])
    if not full or spec["model_family"] in {"identity_probability", "weighted_constant_probability"}:
        return {"method": "not_applicable_or_constant", "replicate_count": 0, "maximum_absolute_parameter_deviation": 0.0, "sign_agreement_rate": 1.0, "passed": True}
    markets = sorted({row["market_id"] for row in rows})
    replicates = [_fit_contract([row for row in rows if row["market_id"] != market], spec).get("parameters", []) for market in markets]
    max_deviation = max(abs(value - reference) for parameters in replicates for value, reference in zip(parameters, full, strict=True))
    sign_pairs = [(value >= 0) == (reference >= 0) for parameters in replicates for value, reference in zip(parameters, full, strict=True) if abs(reference) > 1e-9]
    sign_rate = sum(sign_pairs) / len(sign_pairs) if sign_pairs else 1.0
    return {"method": "leave_one_market_out", "replicate_count": len(replicates), "maximum_absolute_parameter_deviation": max_deviation, "sign_agreement_rate": sign_rate, "passed": max_deviation <= 1.0 and sign_rate >= 0.75}


def _parameter_stability(parameters: list[list[float]]) -> dict[str, Any]:
    if not parameters or not parameters[0]:
        return {"maximum_fold_parameter_range": 0.0, "finite": True}
    width = min(len(values) for values in parameters)
    ranges = [max(values[index] for values in parameters) - min(values[index] for values in parameters) for index in range(width)]
    return {"maximum_fold_parameter_range": max(ranges, default=0.0), "finite": all(math.isfinite(value) for values in parameters for value in values)}


def _worst_market_error_share(rows: list[dict[str, Any]], predictions: list[float]) -> float:
    errors = defaultdict(float)
    for row, prediction in zip(rows, predictions, strict=True):
        errors[row["market_id"]] += (row["selected_side_win_target"] - prediction) ** 2
    total = sum(errors.values())
    return max(errors.values(), default=0.0) / total if total else 0.0


def _horizon(row: dict[str, Any]) -> str:
    text = str(row.get("market_slug", ""))
    if "15m" in text or float(row["decision_time_features"].get("time_to_close_seconds", 0.0)) > 600.0:
        return "15m"
    return "5m"


def _relative(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / baseline if baseline > 0 else 0.0


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector, strict=True)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        if abs(augmented[column][column]) < 1e-12:
            augmented[column][column] = 1e-12
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [left - factor * right for left, right in zip(augmented[row], augmented[column], strict=True)]
    return [augmented[index][-1] for index in range(size)]


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-min(value, 50.0))
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(max(value, -50.0))
    return exponential / (1.0 + exponential)


def _clip_probability(value: float) -> float:
    return min(max(float(value), 1e-9), 1.0 - 1e-9)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _source_tree_hash(repository_root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted((repository_root / "src" / "bigan" / "v8" / "polymarket").rglob("*.py"))
    for path in paths:
        digest.update(str(path.relative_to(repository_root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git(repository_root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repository_root, text=True, check=True, capture_output=True).stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _descriptor(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _directory_descriptor(path: Path) -> dict[str, Any]:
    artifacts = [
        {
            "relative_path": str(file_path.relative_to(path)),
            "sha256": sha256_file(file_path),
        }
        for file_path in sorted(path.rglob("*"))
        if file_path.is_file()
    ]
    return {
        "path": str(path.resolve()),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "tree_sha256": canonical_json_sha256(artifacts),
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_sha_descriptor(path: Path) -> Path:
    descriptor = path.with_suffix(".sha256")
    descriptor.write_text(f"{sha256_file(path)}  {path.name}\n", encoding="utf-8")
    return descriptor


def _write_hashed_json(directory: Path, stem: str, payload: dict[str, Any]) -> tuple[Path, Path]:
    path = directory / f"{stem}.json"
    _write_json(path, payload)
    return path, _write_sha_descriptor(path)


def _verify_sha_descriptor(path: Path) -> None:
    descriptor = path.with_suffix(".sha256")
    expected = descriptor.read_text(encoding="utf-8").split()[0]
    if sha256_file(path) != expected:
        raise ValueError(f"immutable content hash mismatch: {path}")


def _verify_named_hash(directory: Path, stem: str) -> None:
    _verify_sha_descriptor(directory / f"{stem}.json")


def _ensure_placeholder_final_artifacts(goal_dir: Path) -> None:
    placeholders = {
        "final_confirmatory_validation_report.json": {"available": False, "reason_code": "no_validation_round_passed_all_frozen_gates"},
        "frozen_diagnostic_artifact.json": {"available": False, "reason_code": "no_validation_round_passed_all_frozen_gates"},
        "future_shadow_manifest.json": {"available": False, "reason_code": "frozen_diagnostic_artifact_not_available"},
        "future_shadow_evaluation_report.json": {"available": False, "reason_code": "frozen_diagnostic_artifact_not_available"},
    }
    for name, payload in placeholders.items():
        path = goal_dir / name
        if not path.exists():
            _write_json(path, {**payload, **safety_fields()})
        if name == "frozen_diagnostic_artifact.json":
            _write_sha_descriptor(path)


def _report_markdown(report: dict[str, Any]) -> str:
    return "\n".join([
        "# v8 Pre-Promotion Estimand Reformulation",
        "",
        f"- Final state: `{report['final_state']}`",
        f"- Pre-promotion readiness complete: `{str(report['pre_promotion_readiness_complete']).lower()}`",
        "- Readiness scope: `HTS_ONLY`",
        "- Full action-family readiness: `false`",
        "- SELL_BEFORE_CLOSE calibration supported: `false`",
        f"- Blocking reasons: `{report['blocking_reason_codes']}`",
        "- Promotion evidence stage started: `false`",
        "- Live evidence stage started: `false`",
        "- Execution handoff allowed: `false`",
    ])


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "EstimandReformulationConfig",
    "develop_probability_candidates",
    "evaluate_future_unseen_shadow",
    "finalize_estimand_reformulation_goal",
    "freeze_and_evaluate_validation_round",
    "initialize_estimand_reformulation_goal",
    "safety_fields",
    "utc_now_iso",
]
