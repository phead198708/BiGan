"""Pre-register the candidate-agnostic #190 future-unseen holdout protocol."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_accepted_bet_power import (
    RECOMMENDED_MAXIMUM_CAPTURE_ATTEMPT_COUNT,
    RECOMMENDED_QUALITY_VALID_MARKET_COUNT,
    RECOMMENDED_REQUIRED_ACCEPTED_UNIQUE_MARKET_COUNT,
    load_and_validate_pairwise_accepted_bet_power_analysis_manifest,
)
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb import (
    CANDIDATE_NAME,
    FORBIDDEN_REGISTRY_FIELDS,
    _blocked_safety_fields,
    _descriptor,
    _find_fields,
    _load_json,
    _load_jsonl,
    _require_sha256,
    _sha256_file,
    _verify_pin,
    _write_json,
    _write_text,
    validate_pairwise_action_advantage_lcb_feature_contract,
    validate_pairwise_action_advantage_lcb_protocol,
)

PROTOCOL_SCHEMA_VERSION = (
    "bigan-v8-execution-layer-v2-pairwise-future-unseen-holdout-protocol-v1"
)
PRE_REGISTRATION_REPORT_SCHEMA_VERSION = (
    "bigan-v8-pairwise-future-unseen-holdout-pre-registration-report-v1"
)
PRE_REGISTRATION_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-pairwise-future-unseen-holdout-pre-registration-manifest-v1"
)
COLLECTION_FREEZE_REPORT_SCHEMA_VERSION = (
    "bigan-v8-pairwise-future-unseen-collection-freeze-report-v1"
)
COLLECTION_FREEZE_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-pairwise-future-unseen-collection-freeze-manifest-v1"
)
TARGET_VALID_MARKET_COUNT = RECOMMENDED_QUALITY_VALID_MARKET_COUNT
MAXIMUM_CAPTURE_ATTEMPT_COUNT = RECOMMENDED_MAXIMUM_CAPTURE_ATTEMPT_COUNT
MINIMUM_ACCEPTED_BET_COUNT = RECOMMENDED_REQUIRED_ACCEPTED_UNIQUE_MARKET_COUNT
MINIMUM_ACCEPTED_BET_COUNT_PER_SIDE = 10
MINIMUM_ACCEPTED_BET_COUNT_PER_FAMILY = 10
REQUIRED_SIDES = ("UP", "DOWN")
REQUIRED_FAMILIES = ("HOLD_TO_SETTLEMENT", "SELL_BEFORE_CLOSE")


@dataclass(frozen=True, slots=True)
class PairwiseFutureUnseenHoldoutPreRegistrationConfig:
    """Immutable evidence written before confirmatory labels are opened."""

    run_id: str
    output_dir: Path | str
    pre_registration_created_ts: int
    holdout_protocol_path: Path | str
    expected_holdout_protocol_sha256: str
    candidate_protocol_path: Path | str
    expected_candidate_protocol_sha256: str
    feature_contract_path: Path | str
    expected_feature_contract_sha256: str
    power_analysis_manifest_path: Path | str
    expected_power_analysis_manifest_sha256: str
    builder_git_commit: str

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if self.pre_registration_created_ts <= 0:
            raise ValueError("pre_registration_created_ts must be positive")
        if re.fullmatch(r"[0-9a-fA-F]{40}", self.builder_git_commit) is None:
            raise ValueError("builder_git_commit must be a 40-character hex digest")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        for field in (
            "holdout_protocol_path",
            "candidate_protocol_path",
            "feature_contract_path",
            "power_analysis_manifest_path",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)))
        for field in (
            "expected_holdout_protocol_sha256",
            "expected_candidate_protocol_sha256",
            "expected_feature_contract_sha256",
            "expected_power_analysis_manifest_sha256",
        ):
            _require_sha256(getattr(self, field), name=field)
            object.__setattr__(self, field, getattr(self, field).lower())
        object.__setattr__(self, "builder_git_commit", self.builder_git_commit.lower())


@dataclass(frozen=True, slots=True)
class PairwiseFutureUnseenCollectionFreezeConfig:
    """Bind a sealed future collection to the terminal #188 source boundary."""

    run_id: str
    output_dir: Path | str
    collection_freeze_created_ts: int
    pre_registration_manifest_path: Path | str
    expected_pre_registration_manifest_sha256: str
    source_support_gate_manifest_path: Path | str
    expected_source_support_gate_manifest_sha256: str
    builder_git_commit: str

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if self.collection_freeze_created_ts <= 0:
            raise ValueError("collection_freeze_created_ts must be positive")
        if re.fullmatch(r"[0-9a-fA-F]{40}", self.builder_git_commit) is None:
            raise ValueError("builder_git_commit must be a 40-character hex digest")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(
            self,
            "pre_registration_manifest_path",
            Path(self.pre_registration_manifest_path),
        )
        object.__setattr__(
            self,
            "source_support_gate_manifest_path",
            Path(self.source_support_gate_manifest_path),
        )
        for field in (
            "expected_pre_registration_manifest_sha256",
            "expected_source_support_gate_manifest_sha256",
        ):
            _require_sha256(getattr(self, field), name=field)
            object.__setattr__(self, field, getattr(self, field).lower())
        object.__setattr__(self, "builder_git_commit", self.builder_git_commit.lower())


def create_pairwise_future_unseen_holdout_pre_registration(
    config: PairwiseFutureUnseenHoldoutPreRegistrationConfig,
) -> dict[str, Any]:
    """Freeze future collection/evaluation rules without opening any labels."""

    holdout_protocol_path = config.holdout_protocol_path.resolve()
    candidate_protocol_path = config.candidate_protocol_path.resolve()
    feature_contract_path = config.feature_contract_path.resolve()
    power_analysis_manifest_path = config.power_analysis_manifest_path.resolve()
    for path, digest, name in (
        (
            holdout_protocol_path,
            config.expected_holdout_protocol_sha256,
            "future-unseen holdout protocol",
        ),
        (
            candidate_protocol_path,
            config.expected_candidate_protocol_sha256,
            "pairwise candidate protocol",
        ),
        (
            feature_contract_path,
            config.expected_feature_contract_sha256,
            "pairwise feature contract",
        ),
    ):
        _verify_pin(path, digest, name=name)

    holdout_protocol = _load_json(holdout_protocol_path)
    candidate_protocol = _load_json(candidate_protocol_path)
    feature_contract = _load_json(feature_contract_path)
    validate_pairwise_future_unseen_holdout_protocol(holdout_protocol)
    validate_pairwise_action_advantage_lcb_protocol(candidate_protocol)
    validate_pairwise_action_advantage_lcb_feature_contract(
        feature_contract,
        expected_parent_protocol_sha256=config.expected_candidate_protocol_sha256,
    )
    if candidate_protocol.get("candidate_name") != CANDIDATE_NAME:
        raise ValueError("pairwise candidate protocol identity is invalid")
    _, power_audit = load_and_validate_pairwise_accepted_bet_power_analysis_manifest(
        power_analysis_manifest_path,
        config.expected_power_analysis_manifest_sha256,
    )
    if (
        int(power_audit["required_accepted_unique_market_count"])
        != MINIMUM_ACCEPTED_BET_COUNT
        or int(power_audit["recommended_quality_valid_market_count"])
        != TARGET_VALID_MARKET_COUNT
        or int(power_audit["recommended_maximum_capture_attempt_count"])
        != MAXIMUM_CAPTURE_ATTEMPT_COUNT
    ):
        raise ValueError("future holdout sizing does not match prospective power artifact")

    run_dir = config.output_dir / config.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": PRE_REGISTRATION_REPORT_SCHEMA_VERSION,
        "run_id": config.run_id,
        "pre_registration_created_ts": config.pre_registration_created_ts,
        "builder_git_commit": config.builder_git_commit,
        "protocol_name": holdout_protocol["protocol_name"],
        "candidate_name": CANDIDATE_NAME,
        "candidate_agnostic_raw_collection": True,
        "collection_start_condition": (
            "strictly_after_issue188_final_capture_and_source_corpus_max_decision_ts"
        ),
        "collection_may_run_in_parallel_with_issue189_confirmatory": True,
        "collection_stop_rule": (
            f"earliest_{TARGET_VALID_MARKET_COUNT}_outcome_blind_execution_compatible_"
            f"markets_or_max_{MAXIMUM_CAPTURE_ATTEMPT_COUNT}_attempts"
        ),
        "target_valid_market_count": TARGET_VALID_MARKET_COUNT,
        "maximum_capture_attempt_count": MAXIMUM_CAPTURE_ATTEMPT_COUNT,
        "minimum_accepted_unique_market_count": MINIMUM_ACCEPTED_BET_COUNT,
        "collection_sizing_derived_from_prospective_power_analysis": True,
        "power_analysis_uses_current_oof_validation_or_confirmatory_pnl": False,
        "power_analysis_ready": True,
        "dynamic_extension_allowed": False,
        "result_dependent_collection_control_allowed": False,
        "confirmatory_labels_opened_before_pre_registration": False,
        "holdout_labels_or_outcomes_opened_before_pre_registration": False,
        "holdout_collection_started_before_pre_registration": False,
        "holdout_evaluation_started_before_pre_registration": False,
        "pre_registration_ready": True,
        "blocking_reason_codes": [],
        **_blocked_safety_fields(),
    }
    report_path = run_dir / "pairwise_future_unseen_holdout_pre_registration_report.json"
    _write_json(report_path, report)
    _write_text(
        run_dir / "pairwise_future_unseen_holdout_pre_registration_report.md",
        _pre_registration_markdown(report),
    )
    manifest = {
        "schema_version": PRE_REGISTRATION_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "pre_registration_created_ts": config.pre_registration_created_ts,
        "builder_git_commit": config.builder_git_commit,
        "candidate_name": CANDIDATE_NAME,
        "holdout_protocol": _descriptor(holdout_protocol_path),
        "candidate_protocol": _descriptor(candidate_protocol_path),
        "feature_contract": _descriptor(feature_contract_path),
        "power_analysis_manifest": _descriptor(power_analysis_manifest_path),
        "power_analysis_report": power_audit["power_analysis_report"],
        "report": _descriptor(report_path),
        "pre_registration_ready": True,
        "collection_start_allowed_after_issue188_terminal_boundary": True,
        "collection_may_run_before_issue189_confirmatory_result": True,
        "candidate_agnostic_raw_collection": True,
        "target_valid_market_count": TARGET_VALID_MARKET_COUNT,
        "maximum_capture_attempt_count": MAXIMUM_CAPTURE_ATTEMPT_COUNT,
        "minimum_accepted_unique_market_count": MINIMUM_ACCEPTED_BET_COUNT,
        "collection_sizing_derived_from_prospective_power_analysis": True,
        "power_analysis_uses_current_oof_validation_or_confirmatory_pnl": False,
        "collection_stop_rule_is_outcome_blind": True,
        "collection_control_uses_model_scores_bets_or_pnl": False,
        "confirmatory_labels_opened_before_pre_registration": False,
        "holdout_labels_or_outcomes_opened_before_pre_registration": False,
        "holdout_collection_started_before_pre_registration": False,
        "holdout_evaluation_started_before_pre_registration": False,
        "future_holdout_pass_automatically_unlocks_promotion": False,
        "blocking_reason_codes": [],
        **_blocked_safety_fields(),
    }
    manifest["pre_registration_manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = (
        run_dir / "pairwise_future_unseen_holdout_pre_registration_manifest.json"
    )
    _write_json(manifest_path, manifest)
    _write_text(
        run_dir / "pairwise_future_unseen_holdout_pre_registration_manifest.md",
        _pre_registration_manifest_markdown(manifest),
    )
    return {
        "run_dir": run_dir,
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "manifest": manifest,
    }


def create_pairwise_future_unseen_collection_freeze(
    config: PairwiseFutureUnseenCollectionFreezeConfig,
) -> dict[str, Any]:
    """Freeze the source boundary after #188 without opening target values."""

    pre_registration_path = config.pre_registration_manifest_path.resolve()
    pre_registration, _ = (
        load_and_validate_self_contained_future_holdout_pre_registration(
            pre_registration_path,
            config.expected_pre_registration_manifest_sha256,
        )
    )
    support_path = config.source_support_gate_manifest_path.resolve()
    _verify_pin(
        support_path,
        config.expected_source_support_gate_manifest_sha256,
        name="terminal #188 support gate manifest",
    )
    support = _load_json(support_path)
    core_descriptor = _verified_descriptor(
        support.get("core_support_gate_manifest"),
        name="terminal core support gate manifest",
    )
    core = _load_json(Path(core_descriptor["path"]))
    role_descriptor = _verified_descriptor(
        core.get("role_assignment_manifest"),
        name="terminal source role assignment manifest",
    )
    role_manifest = _load_json(Path(role_descriptor["path"]))
    selected_descriptor = _verified_descriptor(
        role_manifest.get("selected_rows"),
        name="terminal source selected role rows",
    )
    selected_rows = _load_jsonl(Path(selected_descriptor["path"]))
    exclusion_descriptor = _verified_descriptor(
        role_manifest.get("prior_evidence_exclusion_registry"),
        name="terminal prior evidence exclusion registry",
    )
    exclusion_registry = _load_json(Path(exclusion_descriptor["path"]))
    blockers = _source_boundary_blockers(
        support=support,
        core=core,
        role_manifest=role_manifest,
        selected_rows=selected_rows,
    )
    if blockers:
        raise ValueError(
            "future holdout source boundary validation failed: "
            + ", ".join(blockers)
        )
    source_market_ids = {str(row["market_id"]) for row in selected_rows}
    prior_market_ids = {
        str(value) for value in exclusion_registry.get("prior_market_ids") or []
    }
    all_prior_market_ids = sorted(source_market_ids | prior_market_ids)
    source_max_decision_ts = max(
        int(row["maximum_decision_ts"]) for row in selected_rows
    )
    minimum_collection_decision_ts = max(
        source_max_decision_ts + 1,
        int(pre_registration["pre_registration_created_ts"]) + 1,
        config.collection_freeze_created_ts + 1,
    )
    run_dir = config.output_dir / config.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": COLLECTION_FREEZE_REPORT_SCHEMA_VERSION,
        "run_id": config.run_id,
        "collection_freeze_created_ts": config.collection_freeze_created_ts,
        "builder_git_commit": config.builder_git_commit,
        "source_selected_market_count": len(source_market_ids),
        "source_max_decision_ts": source_max_decision_ts,
        "minimum_collection_decision_ts": minimum_collection_decision_ts,
        "prior_reference_market_count": len(all_prior_market_ids),
        "prior_reference_market_ids_sha256": canonical_json_sha256(
            all_prior_market_ids
        ),
        "target_valid_market_count": TARGET_VALID_MARKET_COUNT,
        "maximum_capture_attempt_count": MAXIMUM_CAPTURE_ATTEMPT_COUNT,
        "minimum_accepted_unique_market_count": MINIMUM_ACCEPTED_BET_COUNT,
        "collection_sizing_derived_from_prospective_power_analysis": True,
        "collection_started": False,
        "labels_or_outcomes_opened_for_source_boundary": False,
        "settlement_pnl_opened_for_source_boundary": False,
        "source_boundary_validation_passed": True,
        "blocking_reason_codes": [],
        **_blocked_safety_fields(),
    }
    report_path = run_dir / "pairwise_future_unseen_collection_freeze_report.json"
    _write_json(report_path, report)
    _write_text(
        run_dir / "pairwise_future_unseen_collection_freeze_report.md",
        _collection_freeze_markdown(report),
    )
    manifest = {
        "schema_version": COLLECTION_FREEZE_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "collection_freeze_created_ts": config.collection_freeze_created_ts,
        "builder_git_commit": config.builder_git_commit,
        "pre_registration_manifest": _descriptor(pre_registration_path),
        "holdout_protocol": pre_registration["holdout_protocol"],
        "candidate_protocol": pre_registration["candidate_protocol"],
        "feature_contract": pre_registration["feature_contract"],
        "power_analysis_manifest": pre_registration["power_analysis_manifest"],
        "power_analysis_report": pre_registration["power_analysis_report"],
        "source_support_gate_manifest": _descriptor(support_path),
        "source_core_support_gate_manifest": core_descriptor,
        "source_role_assignment_manifest": role_descriptor,
        "source_selected_rows": selected_descriptor,
        "source_prior_evidence_exclusion_registry": exclusion_descriptor,
        "report": _descriptor(report_path),
        "source_selected_market_count": len(source_market_ids),
        "source_max_decision_ts": source_max_decision_ts,
        "minimum_collection_decision_ts": minimum_collection_decision_ts,
        "prior_reference_market_count": len(all_prior_market_ids),
        "prior_reference_market_ids_sha256": canonical_json_sha256(
            all_prior_market_ids
        ),
        "target_valid_market_count": TARGET_VALID_MARKET_COUNT,
        "maximum_capture_attempt_count": MAXIMUM_CAPTURE_ATTEMPT_COUNT,
        "minimum_accepted_unique_market_count": MINIMUM_ACCEPTED_BET_COUNT,
        "collection_sizing_derived_from_prospective_power_analysis": True,
        "stop_when_target_valid_market_count_reached": True,
        "collection_control_is_outcome_blind": True,
        "collection_started": False,
        "labels_or_outcomes_opened_for_collection_freeze": False,
        "settlement_pnl_opened_for_collection_freeze": False,
        "blocking_reason_codes": [],
        **_blocked_safety_fields(),
    }
    manifest["collection_freeze_manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "pairwise_future_unseen_collection_freeze_manifest.json"
    _write_json(manifest_path, manifest)
    _write_text(
        run_dir / "pairwise_future_unseen_collection_freeze_manifest.md",
        _collection_freeze_manifest_markdown(manifest),
    )
    return {
        "run_dir": run_dir,
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "manifest": manifest,
    }


def validate_pairwise_future_unseen_holdout_protocol(
    protocol: dict[str, Any],
) -> None:
    """Reject collection or evaluation rules that permit result-driven drift."""

    collection = dict(protocol.get("collection_protocol") or {})
    power = dict(protocol.get("prospective_power_analysis_contract") or {})
    temporal = dict(protocol.get("temporal_and_identity_contract") or {})
    binding = dict(protocol.get("candidate_binding_contract") or {})
    support = dict(protocol.get("accepted_bet_support_gates") or {})
    pnl = dict(protocol.get("accepted_bet_pnl_gates") or {})
    fail_closed = dict(protocol.get("fail_closed_contract") or {})
    safety = dict(protocol.get("safety") or {})
    blockers: list[str] = []
    required_top_level = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "frozen": True,
        "decision_time_safe": True,
        "candidate_agnostic_raw_collection": True,
        "collection_may_run_in_parallel_with_confirmatory_evaluation": True,
        "uses_confirmatory_results_to_control_collection": False,
        "uses_holdout_outcomes_to_control_collection": False,
    }
    _check_exact(protocol, required_top_level, blockers, prefix="protocol")
    _check_exact(
        collection,
        {
            "market_family": "btc_updown_5m",
            "selection_method": (
                "earliest_execution_compatible_unique_markets_in_chronological_capture_order"
            ),
            "target_valid_market_count": TARGET_VALID_MARKET_COUNT,
            "maximum_capture_attempt_count": MAXIMUM_CAPTURE_ATTEMPT_COUNT,
            "stop_when_target_valid_market_count_reached": True,
            "stop_early_for_model_score_bet_count_or_pnl_allowed": False,
            "dynamic_extension_allowed": False,
            "replacement_attempts_after_maximum_allowed": False,
            "selection_is_outcome_blind": True,
            "labels_or_outcomes_opened_during_collection": False,
            "github_comment_policy": "bounded_batch_summary_only",
        },
        blockers,
        prefix="collection",
    )
    expected_quality_inputs = {
        "raw_evidence_completeness",
        "timestamp_causality",
        "provider_provenance",
        "executable_up_down_book_coverage",
        "required_runtime_feature_coverage",
    }
    if set(collection.get("quality_validity_inputs") or []) != expected_quality_inputs:
        blockers.append("collection_quality_validity_inputs_invalid")
    expected_forbidden_control = {
        "model_score",
        "accepted_bet_count",
        "settlement_outcome",
        "realized_pnl",
        "oracle_action",
        "future_return",
    }
    if set(collection.get("forbidden_collection_control_inputs") or []) != (
        expected_forbidden_control
    ):
        blockers.append("forbidden_collection_control_inputs_invalid")
    _check_exact(
        power,
        {
            "power_analysis_manifest_required_before_pre_registration": True,
            "statistical_unit": "unique_accepted_bet_market",
            "target_power": 0.9,
            "minimum_relevant_standardized_effect_size": 0.35,
            "required_accepted_unique_market_count": MINIMUM_ACCEPTED_BET_COUNT,
            "recommended_quality_valid_market_count": TARGET_VALID_MARKET_COUNT,
            "recommended_maximum_capture_attempt_count": (
                MAXIMUM_CAPTURE_ATTEMPT_COUNT
            ),
            "uses_current_oof_validation_or_confirmatory_pnl": False,
            "uses_realized_candidate_pnl_for_design": False,
            "result_dependent_extension_allowed": False,
            "collection_sizing_must_match_power_analysis": True,
        },
        blockers,
        prefix="prospective_power_analysis",
    )
    _check_all_true(
        temporal,
        (
            "strictly_later_than_source_corpus_required",
            "strictly_later_than_pre_registration_freeze_required",
            "market_id_disjointness_required",
            "slug_disjointness_required",
            "decision_id_disjointness_required",
            "source_row_hash_disjointness_required",
        ),
        blockers,
        prefix="temporal_identity",
    )
    _check_exact(
        binding,
        {
            "binding_occurs_after_candidate_freeze": True,
            "holdout_labels_opened_before_candidate_binding": False,
            "single_evaluated_candidate_per_opened_holdout": True,
            "fit_or_refit_on_holdout_allowed": False,
            "calibration_on_holdout_allowed": False,
            "feature_selection_on_holdout_allowed": False,
            "threshold_selection_on_holdout_allowed": False,
            "guard_tuning_on_holdout_allowed": False,
            "score_mutation_on_holdout_allowed": False,
        },
        blockers,
        prefix="candidate_binding",
    )
    _check_exact(
        support,
        {
            "minimum_accepted_bet_count": MINIMUM_ACCEPTED_BET_COUNT,
            "minimum_accepted_unique_market_count": MINIMUM_ACCEPTED_BET_COUNT,
            "minimum_accepted_bet_count_per_side": MINIMUM_ACCEPTED_BET_COUNT_PER_SIDE,
            "minimum_accepted_bet_count_per_family": (
                MINIMUM_ACCEPTED_BET_COUNT_PER_FAMILY
            ),
            "all_accepted_bets_must_be_settled": True,
            "zero_forbidden_decision_field_violations_required": True,
            "zero_timestamp_provenance_violations_required": True,
            "zero_runtime_state_violations_required": True,
        },
        blockers,
        prefix="support_gate",
    )
    if tuple(support.get("required_sides") or ()) != REQUIRED_SIDES:
        blockers.append("required_side_support_invalid")
    if tuple(support.get("required_families") or ()) != REQUIRED_FAMILIES:
        blockers.append("required_family_support_invalid")
    _check_all_true(
        pnl,
        (
            "candidate_total_post_cost_net_pnl_must_be_positive",
            "candidate_roi_must_be_positive",
            "candidate_post_cost_net_pnl_must_exceed_frozen_baseline",
            "each_required_side_post_cost_net_pnl_must_be_positive",
            "each_required_family_post_cost_net_pnl_must_be_positive",
            "candidate_minus_baseline_market_bootstrap_lower_bound_must_be_positive",
            "largest_winner_removed_candidate_pnl_must_be_positive",
            "leave_one_market_out_candidate_pnl_must_be_positive",
        ),
        blockers,
        prefix="pnl_gate",
    )
    if pnl.get("evaluation_scope") != (
        "frozen_execution_guard_accepted_bets_after_full_costs"
    ):
        blockers.append("accepted_bet_pnl_evaluation_scope_invalid")
    if (
        pnl.get("bootstrap_unit") != "market_id"
        or float(pnl.get("confidence_level") or 0.0) != 0.95
        or int(pnl.get("bootstrap_resample_count") or 0) != 2000
    ):
        blockers.append("market_bootstrap_contract_invalid")
    _check_exact(
        fail_closed,
        {
            "failed_evaluated_holdout_ends_frozen_candidate_cycle": True,
            "tuning_after_holdout_failure_allowed": False,
            "automatic_promotion_allowed": False,
            "explicit_manual_promotion_review_required": True,
            "future_holdout_pass_does_not_unlock_paper_live_or_handoff": True,
        },
        blockers,
        prefix="fail_closed",
    )
    if safety != _blocked_safety_fields():
        blockers.append("holdout_protocol_safety_contract_invalid")
    if blockers:
        raise ValueError(
            "future-unseen holdout protocol validation failed: "
            + ", ".join(sorted(set(blockers)))
        )


def validate_pairwise_future_unseen_holdout_pre_registration_manifest(
    manifest: dict[str, Any],
    *,
    expected_candidate_protocol_path: Path,
    expected_candidate_protocol_sha256: str,
    expected_feature_contract_path: Path,
    expected_feature_contract_sha256: str,
) -> dict[str, Any]:
    """Validate the #190 pin before any #189 JSONL label or prediction access."""

    blockers: list[str] = []
    if manifest.get("schema_version") != PRE_REGISTRATION_MANIFEST_SCHEMA_VERSION:
        blockers.append("future_holdout_pre_registration_schema_invalid")
    if manifest.get("pre_registration_ready") is not True:
        blockers.append("future_holdout_pre_registration_not_ready")
    required_values = {
        "candidate_name": CANDIDATE_NAME,
        "collection_start_allowed_after_issue188_terminal_boundary": True,
        "collection_may_run_before_issue189_confirmatory_result": True,
        "candidate_agnostic_raw_collection": True,
        "target_valid_market_count": TARGET_VALID_MARKET_COUNT,
        "maximum_capture_attempt_count": MAXIMUM_CAPTURE_ATTEMPT_COUNT,
        "minimum_accepted_unique_market_count": MINIMUM_ACCEPTED_BET_COUNT,
        "collection_sizing_derived_from_prospective_power_analysis": True,
        "power_analysis_uses_current_oof_validation_or_confirmatory_pnl": False,
        "collection_stop_rule_is_outcome_blind": True,
        "collection_control_uses_model_scores_bets_or_pnl": False,
        "confirmatory_labels_opened_before_pre_registration": False,
        "holdout_labels_or_outcomes_opened_before_pre_registration": False,
        "holdout_collection_started_before_pre_registration": False,
        "holdout_evaluation_started_before_pre_registration": False,
        "future_holdout_pass_automatically_unlocks_promotion": False,
    }
    _check_exact(
        manifest,
        required_values,
        blockers,
        prefix="pre_registration",
    )
    if manifest.get("blocking_reason_codes") not in ([], None):
        blockers.append("future_holdout_pre_registration_has_blockers")
    if not _top_level_safety_is_blocked(manifest):
        blockers.append("future_holdout_pre_registration_safety_contract_invalid")
    manifest_id = str(manifest.get("pre_registration_manifest_id") or "")
    expected_manifest_id = canonical_json_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key != "pre_registration_manifest_id"
        }
    )
    if manifest_id != expected_manifest_id:
        blockers.append("future_holdout_pre_registration_id_mismatch")

    descriptors: dict[str, dict[str, str]] = {}
    for field in (
        "holdout_protocol",
        "candidate_protocol",
        "feature_contract",
        "power_analysis_manifest",
        "power_analysis_report",
        "report",
    ):
        try:
            descriptors[field] = _verified_descriptor(
                manifest.get(field), name=f"future holdout {field}"
            )
        except (TypeError, ValueError) as exc:
            blockers.append(f"future_holdout_{field}_descriptor_invalid:{exc}")
    candidate_descriptor = descriptors.get("candidate_protocol")
    if candidate_descriptor is not None and (
        Path(candidate_descriptor["path"]).resolve()
        != expected_candidate_protocol_path.resolve()
        or candidate_descriptor["sha256"] != expected_candidate_protocol_sha256.lower()
    ):
        blockers.append("future_holdout_candidate_protocol_lineage_mismatch")
    feature_descriptor = descriptors.get("feature_contract")
    if feature_descriptor is not None and (
        Path(feature_descriptor["path"]).resolve()
        != expected_feature_contract_path.resolve()
        or feature_descriptor["sha256"] != expected_feature_contract_sha256.lower()
    ):
        blockers.append("future_holdout_feature_contract_lineage_mismatch")
    holdout_descriptor = descriptors.get("holdout_protocol")
    if holdout_descriptor is not None:
        try:
            validate_pairwise_future_unseen_holdout_protocol(
                _load_json(Path(holdout_descriptor["path"]))
            )
        except ValueError as exc:
            blockers.append(f"future_holdout_protocol_invalid:{exc}")
    power_manifest_descriptor = descriptors.get("power_analysis_manifest")
    if power_manifest_descriptor is not None:
        try:
            _, power_audit = (
                load_and_validate_pairwise_accepted_bet_power_analysis_manifest(
                    Path(power_manifest_descriptor["path"]),
                    power_manifest_descriptor["sha256"],
                )
            )
            if descriptors.get("power_analysis_report") != power_audit.get(
                "power_analysis_report"
            ):
                blockers.append("future_holdout_power_analysis_report_lineage_mismatch")
        except ValueError as exc:
            blockers.append(f"future_holdout_power_analysis_invalid:{exc}")
    if blockers:
        raise ValueError(
            "future holdout pre-registration validation failed: "
            + ", ".join(sorted(set(blockers)))
        )
    return {
        "future_holdout_protocol": descriptors["holdout_protocol"],
        "future_holdout_pre_registration_report": descriptors["report"],
        "future_holdout_pre_registration_ready": True,
        "future_holdout_power_analysis_manifest": descriptors[
            "power_analysis_manifest"
        ],
        "future_holdout_power_analysis_report": descriptors["power_analysis_report"],
        "future_holdout_minimum_accepted_unique_market_count": (
            MINIMUM_ACCEPTED_BET_COUNT
        ),
        "future_holdout_target_valid_market_count": TARGET_VALID_MARKET_COUNT,
        "future_holdout_maximum_capture_attempt_count": MAXIMUM_CAPTURE_ATTEMPT_COUNT,
        "future_holdout_collection_control_is_outcome_blind": True,
        "future_holdout_labels_or_outcomes_opened": False,
    }


def load_and_validate_pairwise_future_unseen_collection_freeze(
    path: Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the terminal source boundary before scheduling any holdout round."""

    _verify_pin(path, expected_sha256, name="future holdout collection freeze manifest")
    manifest = _load_json(path)
    blockers: list[str] = []
    if manifest.get("schema_version") != COLLECTION_FREEZE_MANIFEST_SCHEMA_VERSION:
        blockers.append("future_holdout_collection_freeze_schema_invalid")
    _check_exact(
        manifest,
        {
            "source_selected_market_count": 195,
            "target_valid_market_count": TARGET_VALID_MARKET_COUNT,
            "maximum_capture_attempt_count": MAXIMUM_CAPTURE_ATTEMPT_COUNT,
            "minimum_accepted_unique_market_count": MINIMUM_ACCEPTED_BET_COUNT,
            "collection_sizing_derived_from_prospective_power_analysis": True,
            "stop_when_target_valid_market_count_reached": True,
            "collection_control_is_outcome_blind": True,
            "collection_started": False,
            "labels_or_outcomes_opened_for_collection_freeze": False,
            "settlement_pnl_opened_for_collection_freeze": False,
        },
        blockers,
        prefix="collection_freeze",
    )
    if manifest.get("blocking_reason_codes") not in ([], None):
        blockers.append("future_holdout_collection_freeze_has_blockers")
    if not _top_level_safety_is_blocked(manifest):
        blockers.append("future_holdout_collection_freeze_safety_contract_invalid")
    manifest_id = str(manifest.get("collection_freeze_manifest_id") or "")
    if manifest_id != canonical_json_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key != "collection_freeze_manifest_id"
        }
    ):
        blockers.append("future_holdout_collection_freeze_id_mismatch")
    descriptors: dict[str, dict[str, str]] = {}
    for field in (
        "pre_registration_manifest",
        "holdout_protocol",
        "candidate_protocol",
        "feature_contract",
        "power_analysis_manifest",
        "power_analysis_report",
        "source_support_gate_manifest",
        "source_core_support_gate_manifest",
        "source_role_assignment_manifest",
        "source_selected_rows",
        "source_prior_evidence_exclusion_registry",
        "report",
    ):
        try:
            descriptors[field] = _verified_descriptor(
                manifest.get(field), name=f"future holdout collection {field}"
            )
        except (TypeError, ValueError) as exc:
            blockers.append(f"future_holdout_collection_{field}_invalid:{exc}")
    pre_registration_descriptor = descriptors.get("pre_registration_manifest")
    pre_registration: dict[str, Any] | None = None
    if pre_registration_descriptor is not None:
        try:
            pre_registration, _ = (
                load_and_validate_self_contained_future_holdout_pre_registration(
                    Path(pre_registration_descriptor["path"]),
                    pre_registration_descriptor["sha256"],
                )
            )
        except ValueError as exc:
            blockers.append(f"future_holdout_pre_registration_invalid:{exc}")
    if pre_registration is not None:
        for field in (
            "holdout_protocol",
            "candidate_protocol",
            "feature_contract",
            "power_analysis_manifest",
            "power_analysis_report",
        ):
            if descriptors.get(field) != pre_registration.get(field):
                blockers.append(f"future_holdout_collection_{field}_lineage_mismatch")
    source_descriptor_names = (
        "source_support_gate_manifest",
        "source_core_support_gate_manifest",
        "source_role_assignment_manifest",
        "source_selected_rows",
        "source_prior_evidence_exclusion_registry",
    )
    if all(name in descriptors for name in source_descriptor_names):
        source_support = _load_json(
            Path(descriptors["source_support_gate_manifest"]["path"])
        )
        source_core = _load_json(
            Path(descriptors["source_core_support_gate_manifest"]["path"])
        )
        source_role = _load_json(
            Path(descriptors["source_role_assignment_manifest"]["path"])
        )
        source_rows = _load_jsonl(Path(descriptors["source_selected_rows"]["path"]))
        source_prior = _load_json(
            Path(descriptors["source_prior_evidence_exclusion_registry"]["path"])
        )
        blockers.extend(
            _source_boundary_blockers(
                support=source_support,
                core=source_core,
                role_manifest=source_role,
                selected_rows=source_rows,
            )
        )
        if not _descriptor_equal(
            source_support.get("core_support_gate_manifest"),
            descriptors["source_core_support_gate_manifest"],
        ):
            blockers.append("future_holdout_source_core_lineage_mismatch")
        if not _descriptor_equal(
            source_core.get("role_assignment_manifest"),
            descriptors["source_role_assignment_manifest"],
        ):
            blockers.append("future_holdout_source_role_lineage_mismatch")
        if not _descriptor_equal(
            source_role.get("selected_rows"),
            descriptors["source_selected_rows"],
        ):
            blockers.append("future_holdout_source_selected_rows_lineage_mismatch")
        if not _descriptor_equal(
            source_role.get("prior_evidence_exclusion_registry"),
            descriptors["source_prior_evidence_exclusion_registry"],
        ):
            blockers.append("future_holdout_source_prior_registry_lineage_mismatch")
        recomputed_source_max = max(
            (int(row.get("maximum_decision_ts") or 0) for row in source_rows),
            default=0,
        )
        if recomputed_source_max != int(manifest.get("source_max_decision_ts") or 0):
            blockers.append("future_holdout_source_max_decision_ts_mismatch")
        all_prior_ids = sorted(
            {str(row.get("market_id") or "") for row in source_rows}
            | {
                str(value)
                for value in source_prior.get("prior_market_ids") or []
            }
        )
        if (
            len(all_prior_ids) != int(manifest.get("prior_reference_market_count") or 0)
            or canonical_json_sha256(all_prior_ids)
            != manifest.get("prior_reference_market_ids_sha256")
        ):
            blockers.append("future_holdout_prior_reference_identity_mismatch")
    source_max = int(manifest.get("source_max_decision_ts") or 0)
    minimum_ts = int(manifest.get("minimum_collection_decision_ts") or 0)
    freeze_ts = int(manifest.get("collection_freeze_created_ts") or 0)
    pre_registration_ts = int(
        (pre_registration or {}).get("pre_registration_created_ts") or 0
    )
    if (
        source_max <= 0
        or freeze_ts <= 0
        or pre_registration_ts <= 0
        or minimum_ts <= max(source_max, freeze_ts, pre_registration_ts)
    ):
        blockers.append("future_holdout_collection_minimum_time_invalid")
    if blockers:
        raise ValueError(
            "future holdout collection freeze validation failed: "
            + ", ".join(sorted(set(blockers)))
        )
    return manifest, {
        "future_holdout_collection_freeze_manifest": _descriptor(path),
        "future_holdout_pre_registration_manifest": descriptors[
            "pre_registration_manifest"
        ],
        "minimum_collection_decision_ts": minimum_ts,
        "source_max_decision_ts": source_max,
        "target_valid_market_count": TARGET_VALID_MARKET_COUNT,
        "maximum_capture_attempt_count": MAXIMUM_CAPTURE_ATTEMPT_COUNT,
        "minimum_accepted_unique_market_count": MINIMUM_ACCEPTED_BET_COUNT,
        "collection_control_is_outcome_blind": True,
        "labels_or_outcomes_opened": False,
    }


def _source_boundary_blockers(
    *,
    support: dict[str, Any],
    core: dict[str, Any],
    role_manifest: dict[str, Any],
    selected_rows: list[dict[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    payloads = (support, core, role_manifest, selected_rows)
    if any(_find_fields(payload, FORBIDDEN_REGISTRY_FIELDS) for payload in payloads):
        blockers.append("source_boundary_contains_forbidden_outcome_fields")
    if support.get("schema_version") != (
        "bigan-v8-pairwise-supplemental-support-gate-manifest-v1"
    ):
        blockers.append("source_support_manifest_schema_invalid")
    if support.get("supplemental_support_target_ready") is not True:
        blockers.append("source_support_target_not_ready")
    if int(support.get("selected_market_count") or 0) != 195:
        blockers.append("source_support_market_count_mismatch")
    if dict(support.get("role_market_counts") or {}) != {
        "development_train": 90,
        "development_calibration": 45,
        "confirmatory_validation": 60,
    }:
        blockers.append("source_support_role_count_mismatch")
    if support.get("blocking_reason_codes") not in ([], None):
        blockers.append("source_support_manifest_has_blockers")
    if core.get("schema_version") != (
        "bigan-v8-pairwise-precollection-continuation-manifest-v1"
    ):
        blockers.append("source_core_support_manifest_schema_invalid")
    if role_manifest.get("role_assignment_ready") is not True:
        blockers.append("source_role_assignment_not_ready")
    if role_manifest.get("labels_or_outcomes_opened_for_role_assignment") is not False:
        blockers.append("source_role_assignment_opened_targets")
    if role_manifest.get("blocking_reason_codes") not in ([], None):
        blockers.append("source_role_assignment_has_blockers")
    if len(selected_rows) != 195:
        blockers.append("source_selected_row_count_mismatch")
    market_ids = [str(row.get("market_id") or "") for row in selected_rows]
    if any(not value for value in market_ids) or len(set(market_ids)) != 195:
        blockers.append("source_selected_market_identity_invalid")
    ranks = [int(row.get("selection_rank") or 0) for row in selected_rows]
    if ranks != list(range(1, 196)):
        blockers.append("source_selected_rank_sequence_invalid")
    timestamps = [int(row.get("maximum_decision_ts") or 0) for row in selected_rows]
    if any(value <= 0 for value in timestamps):
        blockers.append("source_selected_decision_timestamp_invalid")
    if any(
        row.get("labels_or_outcomes_opened_for_role_assignment") is not False
        for row in selected_rows
    ):
        blockers.append("source_selected_rows_opened_targets")
    for payload in (support, core, role_manifest):
        if not _top_level_safety_is_blocked(payload):
            blockers.append("source_boundary_safety_contract_invalid")
            break
    return sorted(set(blockers))


def _verified_descriptor(value: Any, *, name: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} descriptor is missing")
    path = Path(str(value.get("path") or "")).resolve()
    digest = str(value.get("sha256") or "").lower()
    _require_sha256(digest, name=f"{name} SHA-256")
    _verify_pin(path, digest, name=name)
    return {"path": str(path), "sha256": digest}


def _descriptor_equal(value: Any, expected: dict[str, str]) -> bool:
    return (
        isinstance(value, dict)
        and str(Path(str(value.get("path") or "")).resolve()) == expected["path"]
        and str(value.get("sha256") or "").lower() == expected["sha256"]
    )


def _check_exact(
    payload: dict[str, Any],
    expected: dict[str, Any],
    blockers: list[str],
    *,
    prefix: str,
) -> None:
    for key, value in expected.items():
        if payload.get(key) != value:
            blockers.append(f"{prefix}_{key}_invalid")


def _check_all_true(
    payload: dict[str, Any],
    fields: tuple[str, ...],
    blockers: list[str],
    *,
    prefix: str,
) -> None:
    for field in fields:
        if payload.get(field) is not True:
            blockers.append(f"{prefix}_{field}_invalid")


def _top_level_safety_is_blocked(payload: dict[str, Any]) -> bool:
    return all(payload.get(key) == value for key, value in _blocked_safety_fields().items())


def _pre_registration_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# #190 Future-Unseen Holdout Pre-Registration",
            "",
            f"- ready: `{str(report['pre_registration_ready']).lower()}`",
            f"- created ts: `{report['pre_registration_created_ts']}`",
            f"- target quality-valid markets: `{report['target_valid_market_count']}`",
            f"- maximum capture attempts: `{report['maximum_capture_attempt_count']}`",
            f"- collection stop rule: `earliest {TARGET_VALID_MARKET_COUNT} quality-valid or max {MAXIMUM_CAPTURE_ATTEMPT_COUNT} attempts`",
            f"- power-required accepted unique markets: `{MINIMUM_ACCEPTED_BET_COUNT}`",
            "- sizing source: `prospective power analysis; no current OOF/validation/confirmatory PnL`",
            "- confirmatory/PnL controls collection: `false`",
            "- labels/outcomes opened: `false`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )


def _pre_registration_manifest_markdown(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# #190 Future-Unseen Holdout Pre-Registration Manifest",
            "",
            f"- manifest id: `{manifest['pre_registration_manifest_id']}`",
            f"- holdout protocol sha256: `{manifest['holdout_protocol']['sha256']}`",
            f"- candidate protocol sha256: `{manifest['candidate_protocol']['sha256']}`",
            f"- feature contract sha256: `{manifest['feature_contract']['sha256']}`",
            "- collection may run before confirmatory result: `true`",
            "- candidate-agnostic raw collection: `true`",
            "- labels/outcomes opened: `false`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )


def _collection_freeze_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# #190 Future-Unseen Collection Source Boundary",
            "",
            f"- validation passed: `{str(report['source_boundary_validation_passed']).lower()}`",
            f"- source markets: `{report['source_selected_market_count']}`",
            f"- source max decision ts: `{report['source_max_decision_ts']}`",
            f"- minimum future decision ts: `{report['minimum_collection_decision_ts']}`",
            f"- prior reference markets: `{report['prior_reference_market_count']}`",
            "- labels/outcomes/PnL opened: `false`",
            "- collection started: `false`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )


def _collection_freeze_manifest_markdown(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# #190 Future-Unseen Collection Freeze Manifest",
            "",
            f"- manifest id: `{manifest['collection_freeze_manifest_id']}`",
            f"- pre-registration sha256: `{manifest['pre_registration_manifest']['sha256']}`",
            f"- source support sha256: `{manifest['source_support_gate_manifest']['sha256']}`",
            f"- minimum future decision ts: `{manifest['minimum_collection_decision_ts']}`",
            f"- stop rule: `earliest {TARGET_VALID_MARKET_COUNT} quality-valid or max {MAXIMUM_CAPTURE_ATTEMPT_COUNT} attempts`",
            f"- power-required accepted unique markets: `{MINIMUM_ACCEPTED_BET_COUNT}`",
            "- collection control is outcome-blind: `true`",
            "- collection started: `false`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )


def load_and_validate_pairwise_future_unseen_holdout_pre_registration(
    path: Path,
    expected_sha256: str,
    *,
    expected_candidate_protocol_path: Path,
    expected_candidate_protocol_sha256: str,
    expected_feature_contract_path: Path,
    expected_feature_contract_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pin and validate a pre-registration manifest for the #189 fit path."""

    _verify_pin(path, expected_sha256, name="future holdout pre-registration manifest")
    manifest = _load_json(path)
    audit = validate_pairwise_future_unseen_holdout_pre_registration_manifest(
        manifest,
        expected_candidate_protocol_path=expected_candidate_protocol_path,
        expected_candidate_protocol_sha256=expected_candidate_protocol_sha256,
        expected_feature_contract_path=expected_feature_contract_path,
        expected_feature_contract_sha256=expected_feature_contract_sha256,
    )
    audit["future_holdout_pre_registration_manifest"] = _descriptor(path)
    return manifest, audit


def load_and_validate_self_contained_future_holdout_pre_registration(
    path: Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a pinned manifest using the candidate descriptors it froze."""

    _verify_pin(path, expected_sha256, name="future holdout pre-registration manifest")
    manifest = _load_json(path)
    candidate = _verified_descriptor(
        manifest.get("candidate_protocol"), name="future holdout candidate protocol"
    )
    feature = _verified_descriptor(
        manifest.get("feature_contract"), name="future holdout feature contract"
    )
    return load_and_validate_pairwise_future_unseen_holdout_pre_registration(
        path,
        expected_sha256,
        expected_candidate_protocol_path=Path(candidate["path"]),
        expected_candidate_protocol_sha256=candidate["sha256"],
        expected_feature_contract_path=Path(feature["path"]),
        expected_feature_contract_sha256=feature["sha256"],
    )


def protocol_sha256(path: Path) -> str:
    """Return the external file digest used by CLI callers."""

    return _sha256_file(path)


def manifest_json(payload: dict[str, Any]) -> str:
    """Stable JSON helper used by thin example wrappers."""

    return json.dumps(payload, indent=2, sort_keys=True)
