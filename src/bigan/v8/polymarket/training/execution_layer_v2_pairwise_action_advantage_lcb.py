"""Precollection freeze for the #175 pairwise action-advantage LCB."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.contracts import compact_safety_fields

SCHEMA_PREFIX = "bigan-v8-execution-layer-v2-pairwise-action-advantage-lcb"
PROTOCOL_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-protocol-v1"
FEATURE_CONTRACT_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-feature-contract-v1"
CANDIDATE_NAME = "market_grouped_pairwise_action_advantage_lcb_v1"
ROLE_MARKET_COUNTS = {
    "development_train": 90,
    "development_calibration": 45,
    "confirmatory_validation": 60,
}
TARGET_MARKET_COUNT = sum(ROLE_MARKET_COUNTS.values())
FORBIDDEN_REGISTRY_FIELDS = {
    "accepted_bet_net_pnl",
    "evaluation_target_net_pnl_per_contract_by_action",
    "evaluation_target_net_return_after_cost_by_action",
    "future_return",
    "gross_pnl",
    "net_pnl",
    "oracle_action",
    "realized_pnl",
    "resolved_outcome",
    "settlement_pnl",
    "settlement_return",
    "target_net_return_after_cost",
    "total_net_pnl_per_notional",
}


@dataclass(frozen=True, slots=True)
class PairwiseActionAdvantageLCBPrecollectionFreezeConfig:
    """Hash-pinned inputs for freezing collection roles before data arrives."""

    run_id: str
    output_dir: Path | str
    protocol_path: Path | str
    expected_protocol_sha256: str
    feature_contract_path: Path | str
    expected_feature_contract_sha256: str
    git_commit: str
    prior_market_registry_pins: tuple[tuple[Path | str, str], ...]
    prior_evidence_artifact_pins: tuple[tuple[Path | str, str], ...]
    expected_prior_unique_market_count: int

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(self.expected_protocol_sha256, name="protocol SHA-256")
        _require_sha256(
            self.expected_feature_contract_sha256,
            name="feature contract SHA-256",
        )
        if len(self.git_commit) != 40 or any(
            char not in "0123456789abcdef" for char in self.git_commit.lower()
        ):
            raise ValueError("git_commit must be a 40-character hex digest")
        if self.expected_prior_unique_market_count < 1:
            raise ValueError("expected_prior_unique_market_count must be positive")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "protocol_path", Path(self.protocol_path))
        object.__setattr__(
            self,
            "feature_contract_path",
            Path(self.feature_contract_path),
        )
        object.__setattr__(
            self,
            "prior_market_registry_pins",
            _normalize_pins(self.prior_market_registry_pins, name="market registry"),
        )
        object.__setattr__(
            self,
            "prior_evidence_artifact_pins",
            _normalize_pins(self.prior_evidence_artifact_pins, name="prior evidence"),
        )


@dataclass(frozen=True, slots=True)
class PairwiseActionAdvantageLCBRoleAssignmentConfig:
    """Hash-pinned collector inputs for outcome-blind market role assignment."""

    run_id: str
    output_dir: Path | str
    precollection_freeze_manifest_path: Path | str
    expected_precollection_freeze_manifest_sha256: str
    batch_progress_pins: tuple[tuple[Path | str, str], ...]
    training_corpus_root: Path | str = Path("/Volumes/PHILIPS/v8")

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(
            self.expected_precollection_freeze_manifest_sha256,
            name="precollection freeze manifest SHA-256",
        )
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(
            self,
            "precollection_freeze_manifest_path",
            Path(self.precollection_freeze_manifest_path),
        )
        object.__setattr__(
            self,
            "batch_progress_pins",
            _normalize_pins(self.batch_progress_pins, name="batch progress"),
        )
        object.__setattr__(self, "training_corpus_root", Path(self.training_corpus_root))


def validate_pairwise_action_advantage_lcb_protocol(protocol: dict[str, Any]) -> None:
    """Fail closed on any drift in the precollection protocol."""

    roles = dict(protocol.get("role_assignment") or {})
    collector = dict(protocol.get("collector_contract") or {})
    cross_fit = dict(protocol.get("cross_fit_protocol") or {})
    advantage = dict(protocol.get("action_advantage_lcb_protocol") or {})
    development = dict(protocol.get("development_freeze_gates") or {})
    confirmatory = dict(protocol.get("confirmatory_validation_gates") or {})
    safety = dict(protocol.get("safety") or {})
    role_total = sum(
        int(roles.get(name) or 0)
        for name in (
            "development_train_market_count",
            "development_calibration_market_count",
            "confirmatory_validation_market_count",
        )
    )
    checks = {
        "schema_version": protocol.get("schema_version") == PROTOCOL_SCHEMA_VERSION,
        "candidate_name": protocol.get("candidate_name") == CANDIDATE_NAME,
        "frozen": protocol.get("frozen") is True,
        "decision_time_safe": protocol.get("decision_time_safe") is True,
        "no_prior_validation_tuning": protocol.get(
            "uses_prior_validation_or_future_labels_for_tuning"
        )
        is False,
        "issue174_quarantined": protocol.get("uses_issue174_confirmatory_labels_for_tuning")
        is False
        and protocol.get("quarantine_all_issues_through") == 174,
        "role_method": roles.get("method")
        == "earliest_capture_quality_valid_unique_markets_chronological_v1",
        "role_total": role_total
        == int(roles.get("target_valid_market_count") or 0)
        == TARGET_MARKET_COUNT
        and all(
            int(roles.get(f"{role}_market_count") or 0) == count
            for role, count in ROLE_MARKET_COUNTS.items()
        ),
        "outcome_blind_roles": roles.get("outcome_blind_role_assignment") is True,
        "bounded_collection": int(roles.get("initial_capture_attempt_count") or 0)
        >= TARGET_MARKET_COUNT
        and int(roles.get("maximum_total_capture_attempt_count") or 0)
        >= int(roles.get("initial_capture_attempt_count") or 0),
        "ws_first": collector.get("orderbook_source_priority")
        == "clob_websocket_primary_rest_fallback",
        "full_round_ws_collection_window": float(
            collector.get("public_provider_timeout_seconds") or 0.0
        )
        >= 300.0
        and float(collector.get("public_provider_timeout_seconds") or 0.0)
        > float(collector.get("public_provider_http_timeout_seconds") or 0.0),
        "bounded_causal_rest_fallback": float(
            collector.get(
                "orderbook_ws_initial_complete_book_timeout_seconds"
            )
            or 0.0
        )
        == 15.0
        and float(
            collector.get("rest_orderbook_fallback_collection_seconds") or 0.0
        )
        >= 300.0
        and collector.get("rest_orderbook_fallback_stops_at_market_close")
        is True,
        "external_training_root": collector.get("training_corpus_root") == "/Volumes/PHILIPS/v8",
        "raw_evidence": collector.get("per_round_raw_evidence_required") is True,
        "async_settlement": collector.get("asynchronous_settlement_required") is True,
        "execution_compatible_collection": float(
            collector.get("orderbook_snapshot_interval_seconds") or 0.0
        )
        == 1.0
        and float(collector.get("maximum_selected_side_book_staleness_ms") or 0.0) == 2_000.0
        and float(collector.get("maximum_opposite_side_book_staleness_ms") or 0.0) == 2_000.0
        and collector.get("complete_up_down_executable_book_required") is True
        and collector.get("execution_compatibility_validated_before_label_access") is True
        and bool(collector.get("required_runtime_feature_fields")),
        "chainlink_freshness_watchdog": float(
            collector.get("chainlink_rtds_stale_reconnect_seconds") or 0.0
        )
        > 0.0
        and float(collector.get("chainlink_rtds_warmup_seconds") or 0.0)
        >= float(collector.get("chainlink_rtds_stale_reconnect_seconds") or 0.0),
        "cross_fit": int(cross_fit.get("fold_count") or 0) == 5
        and cross_fit.get("group_key") == "market_id"
        and cross_fit.get("fit_split") == "development_train_only"
        and cross_fit.get("fold_assignment") == "chronological_expanding_window_prior_markets_only"
        and int(cross_fit.get("initial_training_market_count") or 0) == 15
        and int(cross_fit.get("validation_market_count_per_fold") or 0) == 15
        and int(cross_fit.get("expected_oof_market_count") or 0) == 75
        and int(cross_fit.get("initial_training_market_count") or 0)
        + int(cross_fit.get("fold_count") or 0)
        * int(cross_fit.get("validation_market_count_per_fold") or 0)
        == ROLE_MARKET_COUNTS["development_train"]
        and cross_fit.get("future_market_labels_excluded_from_each_fold") is True,
        "deterministic_model": cross_fit.get("objective") == "rank:pairwise"
        and cross_fit.get("fixed_model_family")
        == "deterministic_market_grouped_xgboost_pairwise_ranker"
        and cross_fit.get("decision_group_key") == "market_id_decision_ts"
        and cross_fit.get("complete_action_grid_required") is True
        and cross_fit.get("nthread") == 1
        and isinstance(cross_fit.get("seed"), int),
        "calibration_only_action_advantage_lcb": advantage.get("source_split")
        == "development_calibration_only"
        and advantage.get("estimand") == "conditional_cost_aware_action_advantage"
        and advantage.get("grouping") == "action_x_train_oof_group_normalized_rank_score_tertile"
        and advantage.get("score_bucket_boundaries_source")
        == "development_train_oof_group_normalized_rank_scores_only"
        and advantage.get("raw_rank_score_cross_model_comparison_allowed") is False
        and advantage.get("bootstrap_unit") == "market_id"
        and int(advantage.get("bootstrap_resample_count") or 0) >= 1_000
        and isinstance(advantage.get("bootstrap_seed"), int)
        and int(advantage.get("minimum_calibration_unique_markets_per_group") or 0) == 10
        and advantage.get("advantage_against_no_trade_required") is True
        and advantage.get("advantage_against_runner_up_required") is True
        and advantage.get("forced_action_side_or_family_quota_enabled") is False
        and advantage.get("affine_calibration_enabled") is False
        and advantage.get("individual_outcome_quantile_subtraction_enabled") is False,
        "development_gate": int(development.get("required_train_market_count") or 0)
        == ROLE_MARKET_COUNTS["development_train"]
        and int(development.get("required_calibration_market_count") or 0)
        == ROLE_MARKET_COUNTS["development_calibration"]
        and int(development.get("minimum_accepted_bet_count") or 0) >= 30
        and int(development.get("minimum_accepted_bet_count_per_side") or 0) >= 10
        and int(development.get("minimum_accepted_bet_count_per_family") or 0) >= 10
        and development.get(
            "candidate_minus_baseline_market_bootstrap_lower_bound_must_be_positive"
        )
        is True,
        "confirmatory_support": int(confirmatory.get("required_unique_market_count") or 0)
        == ROLE_MARKET_COUNTS["confirmatory_validation"]
        and int(confirmatory.get("minimum_accepted_bet_count") or 0) >= 30
        and int(confirmatory.get("minimum_accepted_bet_count_per_side") or 0) >= 10
        and int(confirmatory.get("minimum_accepted_bet_count_per_family") or 0) >= 10
        and confirmatory.get(
            "candidate_minus_baseline_market_bootstrap_lower_bound_must_be_positive"
        )
        is True,
        "safety": safety.get("paper_only") is True
        and safety.get("capital_at_risk") is False
        and safety.get("polymarket_write_enabled") is False
        and safety.get("wallet_signing_enabled") is False
        and safety.get("source_model_candidate_eligible") is False
        and safety.get("freeze_ready") is False
        and safety.get("promotion_evidence_eligible") is False
        and safety.get("v8_execution_handoff_allowed") is False
        and safety.get("#134_resume_allowed") is False
        and safety.get("#146_start_allowed") is False,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("invalid pairwise action-advantage LCB protocol: " + ", ".join(failed))


def validate_pairwise_action_advantage_lcb_feature_contract(
    contract: dict[str, Any],
    *,
    expected_parent_protocol_sha256: str,
) -> None:
    """Fail closed on feature/target semantics that could leak or drift."""

    feature_columns = list(contract.get("feature_columns") or [])
    checks = {
        "schema_version": contract.get("schema_version") == FEATURE_CONTRACT_SCHEMA_VERSION,
        "candidate_name": contract.get("candidate_name") == CANDIDATE_NAME,
        "parent_protocol": contract.get("parent_protocol_sha256")
        == expected_parent_protocol_sha256,
        "frozen": contract.get("frozen") is True,
        "decision_time_safe": contract.get("decision_time_safe") is True,
        "feature_source": contract.get("feature_source") == "phase2_polymarket_feature_rows_only",
        "chainlink_reference_source": contract.get("reference_price_to_beat_distance_source")
        == "polymarket_rtds_chainlink"
        and contract.get("chainlink_reference_feature_required") is True,
        "cex_candles_momentum_only": contract.get(
            "btc_candle_features_are_independent_momentum_only"
        )
        is True
        and contract.get("btc_candle_features_may_not_supply_price_to_beat") is True,
        "target": contract.get("target_field") == "total_net_pnl_per_notional",
        "cost_aware_target": contract.get("target_includes_fees_slippage_and_liquidity_impact")
        is True,
        "role_before_labels": contract.get("role_assignment_must_complete_before_label_access")
        is True,
        "execution_compatible_before_labels": contract.get(
            "execution_compatibility_must_pass_before_label_access"
        )
        is True,
        "no_individual_outcome_quantile": contract.get(
            "individual_outcome_quantile_subtraction_enabled"
        )
        is False,
        "no_confirmatory_tuning": contract.get("uses_confirmatory_validation_labels_for_tuning")
        is False
        and contract.get("uses_issue174_confirmatory_labels_for_tuning") is False,
        "no_prior_future_tuning": contract.get("uses_prior_or_future_evidence_for_tuning") is False,
        "market_probability_semantics": contract.get(
            "market_implied_probability_used_as_conditioning_feature"
        )
        is True
        and contract.get("market_implied_probability_used_as_direct_fair_value_ev") is False,
        "feature_columns": bool(feature_columns)
        and len(feature_columns) == len(set(feature_columns))
        and "action_no_trade" in feature_columns,
        "decision_group_contract": contract.get("complete_five_action_decision_grid_required")
        is True
        and contract.get("decision_group_key_fields")
        == [
            "market_id",
            "decision_ts",
        ]
        and contract.get("action_advantage_against_no_trade_required") is True
        and contract.get("selected_vs_runner_up_advantage_required") is True
        and contract.get("forced_action_side_or_family_quota_enabled") is False,
        "safety": contract.get("paper_only") is True
        and contract.get("capital_at_risk") is False
        and contract.get("polymarket_write_enabled") is False
        and contract.get("wallet_signing_enabled") is False
        and contract.get("v8_execution_handoff_allowed") is False
        and contract.get("source_model_candidate_eligible") is False
        and contract.get("freeze_ready") is False
        and contract.get("promotion_evidence_eligible") is False,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            "invalid pairwise action-advantage LCB feature contract: " + ", ".join(failed)
        )


def freeze_pairwise_action_advantage_lcb_precollection(
    config: PairwiseActionAdvantageLCBPrecollectionFreezeConfig,
) -> dict[str, Any]:
    """Freeze data roles, exclusions, and collector/model contracts before collection."""

    protocol_path = config.protocol_path.resolve()
    current_git_head = _git_head_for_path(protocol_path)
    if config.git_commit.lower() != current_git_head:
        raise ValueError(
            "git_commit does not match the current HEAD for the frozen protocol repository"
        )
    _verify_pin(protocol_path, config.expected_protocol_sha256, name="protocol")
    protocol = _load_json(protocol_path)
    validate_pairwise_action_advantage_lcb_protocol(protocol)
    feature_contract_path = config.feature_contract_path.resolve()
    _verify_pin(
        feature_contract_path,
        config.expected_feature_contract_sha256,
        name="feature contract",
    )
    feature_contract = _load_json(feature_contract_path)
    validate_pairwise_action_advantage_lcb_feature_contract(
        feature_contract,
        expected_parent_protocol_sha256=config.expected_protocol_sha256,
    )

    registry_descriptors = []
    prior_market_ids: set[str] = set()
    prior_decision_timestamps: list[int] = []
    for path, expected_sha256 in config.prior_market_registry_pins:
        resolved = path.resolve()
        _verify_pin(resolved, expected_sha256, name="prior market registry")
        payload = _load_json_or_jsonl(resolved)
        forbidden = sorted(_find_fields(payload, FORBIDDEN_REGISTRY_FIELDS))
        if forbidden:
            raise ValueError(
                "prior market registry contains forbidden outcome fields: " + ", ".join(forbidden)
            )
        prior_market_ids.update(_extract_market_ids(payload))
        prior_decision_timestamps.extend(_extract_decision_timestamps(payload))
        registry_descriptors.append(_descriptor(resolved))
    if "" in prior_market_ids or len(prior_market_ids) != config.expected_prior_unique_market_count:
        raise ValueError(
            "prior unique market count mismatch: "
            f"expected {config.expected_prior_unique_market_count}, got {len(prior_market_ids)}"
        )
    if not prior_decision_timestamps or any(value <= 0 for value in prior_decision_timestamps):
        raise ValueError("prior decision-time registry is incomplete")

    evidence_descriptors = []
    for path, expected_sha256 in config.prior_evidence_artifact_pins:
        resolved = path.resolve()
        _verify_pin(resolved, expected_sha256, name="prior evidence artifact")
        evidence_descriptors.append(_descriptor(resolved))

    created_ts = int(time.time() * 1000)
    max_prior_decision_ts = max(prior_decision_timestamps)
    roles = dict(protocol["role_assignment"])
    collector = dict(protocol["collector_contract"])
    output_dir = config.output_dir / config.run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    exclusion_registry = {
        "schema_version": f"{SCHEMA_PREFIX}-prior-exclusion-registry-v1",
        "run_id": config.run_id,
        "prior_market_registry_sources": registry_descriptors,
        "prior_evidence_artifacts": evidence_descriptors,
        "prior_unique_market_count": len(prior_market_ids),
        "prior_market_ids": sorted(prior_market_ids),
        "prior_market_ids_sha256": canonical_json_sha256(sorted(prior_market_ids)),
        "maximum_prior_decision_ts": max_prior_decision_ts,
        "prior_outcome_or_pnl_values_loaded": False,
        "prior_validation_or_future_evidence_used_for_tuning": False,
        **_blocked_safety_fields(),
    }
    exclusion_registry["exclusion_registry_id"] = canonical_json_sha256(exclusion_registry)
    exclusion_path = output_dir / "prior_evidence_exclusion_registry.json"
    _write_json(exclusion_path, exclusion_registry)

    role_plan = [
        {
            "role": "development_train",
            "valid_market_rank_start": 1,
            "valid_market_rank_end": int(roles["development_train_market_count"]),
        },
        {
            "role": "development_calibration",
            "valid_market_rank_start": int(roles["development_train_market_count"]) + 1,
            "valid_market_rank_end": int(roles["development_train_market_count"])
            + int(roles["development_calibration_market_count"]),
        },
        {
            "role": "confirmatory_validation",
            "valid_market_rank_start": int(roles["development_train_market_count"])
            + int(roles["development_calibration_market_count"])
            + 1,
            "valid_market_rank_end": int(roles["target_valid_market_count"]),
        },
    ]
    batch_id_prefix = f"issue175-{config.run_id}"
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-precollection-role-freeze-v1",
        "run_id": config.run_id,
        "freeze_created_ts": created_ts,
        "git_commit": config.git_commit.lower(),
        "git_commit_current_head_verified": True,
        "protocol": _descriptor(protocol_path),
        "feature_contract": _descriptor(feature_contract_path),
        "prior_evidence_exclusion_registry": _descriptor(exclusion_path),
        "candidate_name": protocol["candidate_name"],
        "role_assignment_method": roles["method"],
        "role_plan": role_plan,
        "target_valid_market_count": int(roles["target_valid_market_count"]),
        "initial_capture_attempt_count": int(roles["initial_capture_attempt_count"]),
        "maximum_total_capture_attempt_count": int(roles["maximum_total_capture_attempt_count"]),
        "collection_batch_id_prefix": batch_id_prefix,
        "collection_output_dir": str((output_dir / "collection").resolve()),
        "collector_contract": collector,
        "collector_contract_sha256": canonical_json_sha256(collector),
        "cross_fit_protocol": protocol["cross_fit_protocol"],
        "cross_fit_protocol_sha256": canonical_json_sha256(protocol["cross_fit_protocol"]),
        "action_advantage_lcb_protocol": protocol["action_advantage_lcb_protocol"],
        "action_advantage_lcb_protocol_sha256": canonical_json_sha256(
            protocol["action_advantage_lcb_protocol"]
        ),
        "frozen_execution_contract": protocol["frozen_execution_contract"],
        "frozen_execution_contract_sha256": canonical_json_sha256(
            protocol["frozen_execution_contract"]
        ),
        "minimum_collection_decision_ts": max(max_prior_decision_ts + 1, created_ts + 1),
        "collection_must_be_strictly_later": True,
        "new_market_ids_must_be_disjoint": True,
        "role_assignment_outcome_blind": True,
        "settlement_labels_available_only_after_round_close": True,
        "collection_started": False,
        "model_fit_started": False,
        "confirmatory_validation_started": False,
        "future_holdout_started": False,
        **_blocked_safety_fields(),
    }
    manifest["precollection_freeze_id"] = canonical_json_sha256(manifest)
    manifest_path = output_dir / "precollection_role_freeze_manifest.json"
    _write_json(manifest_path, manifest)
    markdown_path = output_dir / "precollection_role_freeze_manifest.md"
    _write_text(markdown_path, _freeze_markdown(manifest))
    descriptor = {
        "schema_version": f"{SCHEMA_PREFIX}-precollection-role-freeze-descriptor-v1",
        "manifest": _descriptor(manifest_path),
        "markdown": _descriptor(markdown_path),
        "precollection_freeze_id": manifest["precollection_freeze_id"],
        "collection_started": False,
        **_blocked_safety_fields(),
    }
    descriptor_path = output_dir / "precollection_role_freeze_descriptor.json"
    _write_json(descriptor_path, descriptor)
    return {
        "output_dir": output_dir,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "descriptor_path": descriptor_path,
        "descriptor_sha256": _sha256_file(descriptor_path),
        "manifest": manifest,
    }


def _git_head_for_path(path: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path.parent), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("unable to resolve current Git HEAD for precollection freeze") from exc
    head = completed.stdout.strip().lower()
    if len(head) != 40 or any(char not in "0123456789abcdef" for char in head):
        raise ValueError("current Git HEAD is not a 40-character hex digest")
    return head


def assign_pairwise_action_advantage_lcb_roles(
    config: PairwiseActionAdvantageLCBRoleAssignmentConfig,
) -> dict[str, Any]:
    """Assign the earliest 90/45/60 execution-compatible markets outcome-blind."""

    freeze_path = config.precollection_freeze_manifest_path.resolve()
    _verify_pin(
        freeze_path,
        config.expected_precollection_freeze_manifest_sha256,
        name="precollection freeze manifest",
    )
    freeze = _load_json(freeze_path)
    protocol_descriptor = _verified_descriptor(freeze.get("protocol"), name="frozen protocol")
    protocol = _load_json(Path(protocol_descriptor["path"]))
    validate_pairwise_action_advantage_lcb_protocol(protocol)
    feature_contract_descriptor = _verified_descriptor(
        freeze.get("feature_contract"), name="frozen feature contract"
    )
    validate_pairwise_action_advantage_lcb_feature_contract(
        _load_json(Path(feature_contract_descriptor["path"])),
        expected_parent_protocol_sha256=protocol_descriptor["sha256"],
    )
    exclusion_descriptor = _verified_descriptor(
        freeze.get("prior_evidence_exclusion_registry"),
        name="prior evidence exclusion registry",
    )
    exclusion_registry = _load_json(Path(exclusion_descriptor["path"]))
    prior_market_ids = {str(value) for value in exclusion_registry.get("prior_market_ids") or []}
    if not prior_market_ids or "" in prior_market_ids:
        raise ValueError("prior market exclusion registry is incomplete")
    if canonical_json_sha256(sorted(prior_market_ids)) != exclusion_registry.get(
        "prior_market_ids_sha256"
    ):
        raise ValueError("prior market exclusion registry identity mismatch")

    target_count = int(freeze.get("target_valid_market_count") or 0)
    maximum_attempts = int(freeze.get("maximum_total_capture_attempt_count") or 0)
    minimum_decision_ts = int(freeze.get("minimum_collection_decision_ts") or 0)
    if (
        target_count != TARGET_MARKET_COUNT
        or maximum_attempts < target_count
        or minimum_decision_ts <= 0
    ):
        raise ValueError("precollection role freeze contract is incomplete")
    collector_contract = dict(freeze.get("collector_contract") or {})

    batch_rows: list[dict[str, Any]] = []
    batch_descriptors: list[dict[str, str]] = []
    blocking_reasons: list[str] = []
    for batch_ordinal, (path, expected_sha256) in enumerate(config.batch_progress_pins):
        resolved = path.resolve()
        _verify_pin(resolved, expected_sha256, name="batch progress")
        batch = _load_json(resolved)
        forbidden = sorted(_find_fields(batch, FORBIDDEN_REGISTRY_FIELDS))
        if forbidden:
            raise ValueError(
                "batch progress contains forbidden outcome fields: " + ", ".join(forbidden)
            )
        captures = [dict(row) for row in batch.get("captures") or []]
        finalizations = [dict(row) for row in batch.get("finalizations") or []]
        errors = [dict(row) for row in batch.get("errors") or []]
        if batch.get("paper_only") is not True or batch.get("capital_at_risk") is not False:
            blocking_reasons.append("collector_batch_safety_contract_failed")
        if int(batch.get("capture_count") or 0) != len(captures):
            blocking_reasons.append("collector_capture_count_mismatch")
        if int(batch.get("error_count") or 0) != len(errors):
            blocking_reasons.append("collector_error_count_mismatch")
        finalization_by_run_id = {str(row.get("run_id") or ""): row for row in finalizations}
        if "" in finalization_by_run_id:
            blocking_reasons.append("collector_finalization_run_id_missing")
        for capture in captures:
            batch_rows.append(
                {
                    **capture,
                    "source_batch_ordinal": batch_ordinal,
                    "source_batch_id": str(batch.get("batch_id") or ""),
                    "source_batch_progress_sha256": expected_sha256,
                    "finalization": finalization_by_run_id.get(str(capture.get("run_id") or "")),
                }
            )
        batch_descriptors.append(_descriptor(resolved))
    if len(batch_rows) > maximum_attempts:
        blocking_reasons.append("maximum_capture_attempt_count_exceeded")
    run_ids = [str(row.get("run_id") or "") for row in batch_rows]
    if any(not value for value in run_ids) or len(run_ids) != len(set(run_ids)):
        blocking_reasons.append("collector_duplicate_or_missing_capture_run_id")

    batch_rows.sort(
        key=lambda row: (
            int(row.get("scheduled_round_start_ts") or 0),
            int(row.get("source_batch_ordinal") or 0),
            int(row.get("round_index") or 0),
            str(row.get("run_id") or ""),
        )
    )
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    selected_market_ids: set[str] = set()
    selected_corpus_dirs: set[Path] = set()
    selection_sequence_blocked = False
    training_root = config.training_corpus_root.expanduser().resolve()
    for capture in batch_rows:
        audit = _capture_quality_audit(
            capture,
            collector_contract=collector_contract,
        )
        if audit["reason_codes"]:
            excluded.append(audit)
            continue
        finalization = capture.get("finalization")
        finalization_reasons = _finalization_quality_reasons(finalization)
        if finalization_reasons:
            audit["reason_codes"] = finalization_reasons
            excluded.append(audit)
            selection_sequence_blocked = True
            blocking_reasons.append("earliest_quality_capture_not_finalized")
            break
        corpus_dir = Path(str(finalization["exported_training_corpus_dir"])).expanduser().resolve()
        corpus_reasons: list[str] = []
        if not corpus_dir.is_relative_to(training_root):
            corpus_reasons.append("exported_corpus_outside_training_root")
        elif not corpus_dir.is_dir():
            corpus_reasons.append("exported_corpus_directory_missing")
        if corpus_dir in selected_corpus_dirs:
            corpus_reasons.append("duplicate_exported_corpus_path")
        corpus_audit: dict[str, Any] | None = None
        if not corpus_reasons:
            corpus_audit = _outcome_blind_corpus_role_audit(
                corpus_dir=corpus_dir,
                prior_market_ids=prior_market_ids,
                minimum_decision_ts=minimum_decision_ts,
            )
            corpus_reasons.extend(corpus_audit["reason_codes"])
        execution_compatibility_audit: dict[str, Any] | None = None
        if not corpus_reasons:
            execution_compatibility_audit = _execution_compatibility_audit(
                corpus_dir=corpus_dir,
                collector_contract=collector_contract,
            )
            corpus_reasons.extend(execution_compatibility_audit["blocking_reason_codes"])
        if corpus_reasons:
            audit["reason_codes"] = sorted(set(corpus_reasons))
            excluded.append(audit)
            continue
        assert corpus_audit is not None
        assert execution_compatibility_audit is not None
        market_id = str(corpus_audit["market_id"])
        if market_id in selected_market_ids:
            audit["reason_codes"] = ["duplicate_market_identity"]
            excluded.append(audit)
            continue
        if len(selected) >= target_count:
            audit["reason_codes"] = ["selection_target_already_met"]
            excluded.append(audit)
            continue
        selection_rank = len(selected) + 1
        role = _role_for_rank(selection_rank)
        selected_market_ids.add(market_id)
        selected_corpus_dirs.add(corpus_dir)
        selected.append(
            {
                **audit,
                "selected": True,
                "selection_rank": selection_rank,
                "role": role,
                "market_id": market_id,
                "minimum_decision_ts": corpus_audit["minimum_decision_ts"],
                "maximum_decision_ts": corpus_audit["maximum_decision_ts"],
                "decision_row_count": corpus_audit["decision_row_count"],
                "source_corpus_dir": str(corpus_dir),
                "corpus_manifest": corpus_audit["corpus_manifest"],
                "feature_rows": corpus_audit["feature_rows"],
                "execution_compatibility_audit": execution_compatibility_audit,
                "execution_compatibility_validated_before_label_access": True,
                "labels_or_outcomes_opened_for_role_assignment": False,
                "reason_codes": [],
            }
        )

    if len(selected) != target_count:
        blocking_reasons.append("insufficient_quality_valid_unique_market_support")
    role_counts = Counter(str(row["role"]) for row in selected)
    expected_role_counts = ROLE_MARKET_COUNTS
    if dict(role_counts) != expected_role_counts:
        blocking_reasons.append("role_market_count_mismatch")
    role_market_sets = {
        role: {str(row["market_id"]) for row in selected if row["role"] == role}
        for role in expected_role_counts
    }
    role_overlap = (
        (role_market_sets["development_train"] & role_market_sets["development_calibration"])
        | (role_market_sets["development_train"] & role_market_sets["confirmatory_validation"])
        | (
            role_market_sets["development_calibration"]
            & role_market_sets["confirmatory_validation"]
        )
    )
    if role_overlap:
        blocking_reasons.append("role_market_overlap_detected")
    prior_overlap = selected_market_ids & prior_market_ids
    if prior_overlap:
        blocking_reasons.append("prior_market_overlap_detected")
    blocking_reasons = sorted(set(blocking_reasons))
    role_assignment_ready = not blocking_reasons

    run_dir = config.output_dir / config.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    selected_path = run_dir / "pairwise_action_advantage_lcb_role_assignment_rows.jsonl"
    excluded_path = run_dir / "pairwise_action_advantage_lcb_role_assignment_excluded_rows.jsonl"
    _write_jsonl(selected_path, selected)
    _write_jsonl(excluded_path, excluded)
    compatibility_report = {
        "schema_version": f"{SCHEMA_PREFIX}-execution-compatible-feature-coverage-v1",
        "run_id": config.run_id,
        "execution_compatibility_validated_before_label_access": True,
        "selected_market_count": len(selected),
        "selected_decision_row_count": sum(
            int(row["execution_compatibility_audit"]["decision_row_count"]) for row in selected
        ),
        "execution_compatible_decision_row_count": sum(
            int(row["execution_compatibility_audit"]["execution_compatible_row_count"])
            for row in selected
        ),
        "selected_market_failure_count": sum(
            int(bool(row["execution_compatibility_audit"]["blocking_reason_codes"]))
            for row in selected
        ),
        "maximum_book_staleness_ms": min(
            float(collector_contract["maximum_selected_side_book_staleness_ms"]),
            float(collector_contract["maximum_opposite_side_book_staleness_ms"]),
        ),
        "market_audits": [
            {
                "market_id": row["market_id"],
                "selection_rank": row["selection_rank"],
                "role": row["role"],
                **row["execution_compatibility_audit"],
            }
            for row in selected
        ],
        "labels_or_outcomes_opened": False,
        **_blocked_safety_fields(),
    }
    compatibility_report_path = run_dir / "execution_compatible_feature_coverage_report.json"
    _write_json(compatibility_report_path, compatibility_report)
    _write_text(
        run_dir / "execution_compatible_feature_coverage_report.md",
        _execution_compatibility_markdown(compatibility_report),
    )
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-role-assignment-report-v1",
        "run_id": config.run_id,
        "status": (
            "OUTCOME_BLIND_ROLE_ASSIGNMENT_READY"
            if role_assignment_ready
            else "BLOCKED_FAIL_CLOSED"
        ),
        "role_assignment_ready": role_assignment_ready,
        "blocking_reason_codes": blocking_reasons,
        "attempted_capture_count": len(batch_rows),
        "selected_market_count": len(selected),
        "excluded_capture_count": len(excluded),
        "role_market_counts": dict(sorted(role_counts.items())),
        "role_market_overlap_count": len(role_overlap),
        "prior_market_overlap_count": len(prior_overlap),
        "selection_sequence_blocked": selection_sequence_blocked,
        "excluded_reason_distribution": dict(
            sorted(
                Counter(
                    reason for row in excluded for reason in row.get("reason_codes") or []
                ).items()
            )
        ),
        "role_assignment_method": freeze["role_assignment_method"],
        "role_assignment_uses_outcomes": False,
        "role_assignment_uses_settlement_pnl": False,
        "role_assignment_uses_oracle_actions": False,
        "labels_or_outcomes_opened_for_role_assignment": False,
        "execution_compatibility_validated_before_label_access": True,
        "execution_compatibility_failure_count": sum(
            int(
                any(
                    str(reason).startswith("execution_compatibility_")
                    for reason in row.get("reason_codes") or []
                )
            )
            for row in excluded
        ),
        "model_fit_started": False,
        "confirmatory_validation_started": False,
        "feature_contract_frozen_before_collection": True,
        **_blocked_safety_fields(),
    }
    report_path = run_dir / "pairwise_action_advantage_lcb_role_assignment_report.json"
    _write_json(report_path, report)
    markdown_path = run_dir / "pairwise_action_advantage_lcb_role_assignment_report.md"
    _write_text(markdown_path, _role_assignment_markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-role-assignment-manifest-v1",
        "run_id": config.run_id,
        "role_assignment_ready": role_assignment_ready,
        "blocking_reason_codes": blocking_reasons,
        "precollection_freeze_manifest": _descriptor(freeze_path),
        "protocol": protocol_descriptor,
        "feature_contract": feature_contract_descriptor,
        "execution_compatible_feature_coverage_report": _descriptor(compatibility_report_path),
        "prior_evidence_exclusion_registry": exclusion_descriptor,
        "batch_progress_inputs": batch_descriptors,
        "selected_rows": _descriptor(selected_path),
        "excluded_rows": _descriptor(excluded_path),
        "report": _descriptor(report_path),
        "role_market_counts": dict(sorted(role_counts.items())),
        "selected_market_ids_sha256": canonical_json_sha256(sorted(selected_market_ids)),
        "labels_or_outcomes_opened_for_role_assignment": False,
        "model_fit_started": False,
        "confirmatory_validation_started": False,
        **_blocked_safety_fields(),
    }
    manifest["role_assignment_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "role_assignment_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "selected_rows_path": selected_path,
        "excluded_rows_path": excluded_path,
        "report_path": report_path,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "report": report,
        "manifest": manifest,
    }


def _capture_quality_audit(
    capture: dict[str, Any],
    *,
    collector_contract: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if capture.get("capture_start_boundary_validation_passed") is not True:
        reasons.append("capture_start_boundary_failed")
    if int(capture.get("scheduled_round_start_ts") or 0) <= 0:
        reasons.append("capture_chronology_missing")
    if int(capture.get("raw_polymarket_market_count") or 0) != 1:
        reasons.append("market_row_coverage_failed")
    if int(capture.get("provider_raw_orderbook_snapshot_count") or 0) <= 0:
        reasons.append("provider_orderbook_snapshot_coverage_failed")
    if int(capture.get("training_sampled_orderbook_row_count") or 0) <= 0:
        reasons.append("sampled_orderbook_coverage_failed")
    if int(capture.get("raw_btc_candle_row_count") or 0) <= 0:
        reasons.append("btc_candle_coverage_failed")
    if int(capture.get("raw_chainlink_price_row_count") or 0) <= 0:
        reasons.append("chainlink_rtds_coverage_failed")
    if float(capture.get("orderbook_snapshot_interval_seconds") or 0.0) != float(
        collector_contract["orderbook_snapshot_interval_seconds"]
    ):
        reasons.append("collector_orderbook_snapshot_interval_contract_failed")
    if float(capture.get("public_provider_timeout_seconds") or 0.0) != float(
        collector_contract["public_provider_timeout_seconds"]
    ):
        reasons.append("collector_public_provider_timeout_contract_failed")
    if float(capture.get("public_provider_http_timeout_seconds") or 0.0) != float(
        collector_contract["public_provider_http_timeout_seconds"]
    ):
        reasons.append("collector_public_provider_http_timeout_contract_failed")
    if float(
        capture.get("orderbook_ws_initial_complete_book_timeout_seconds")
        or 0.0
    ) != float(
        collector_contract[
            "orderbook_ws_initial_complete_book_timeout_seconds"
        ]
    ):
        reasons.append(
            "collector_ws_initial_complete_book_timeout_contract_failed"
        )
    if float(
        capture.get("rest_orderbook_fallback_collection_seconds") or 0.0
    ) != float(
        collector_contract["rest_orderbook_fallback_collection_seconds"]
    ):
        reasons.append(
            "collector_rest_orderbook_fallback_collection_contract_failed"
        )
    if (
        capture.get("rest_orderbook_fallback_stops_at_market_close")
        is not True
    ):
        reasons.append(
            "collector_rest_orderbook_fallback_market_close_contract_failed"
        )
    for reason, count in sorted(dict(capture.get("reject_reason_counts") or {}).items()):
        if int(count or 0) > 0:
            reasons.append(f"capture_reject_{reason}")
    return {
        "source_batch_id": capture.get("source_batch_id"),
        "source_batch_ordinal": capture.get("source_batch_ordinal"),
        "source_batch_progress_sha256": capture.get("source_batch_progress_sha256"),
        "capture_run_id": str(capture.get("run_id") or ""),
        "capture_round_index": int(capture.get("round_index") or 0),
        "scheduled_round_start_ts": int(capture.get("scheduled_round_start_ts") or 0),
        "capture_status": capture.get("capture_status"),
        "provider_raw_orderbook_snapshot_count": int(
            capture.get("provider_raw_orderbook_snapshot_count") or 0
        ),
        "training_sampled_orderbook_row_count": int(
            capture.get("training_sampled_orderbook_row_count") or 0
        ),
        "raw_btc_candle_row_count": int(capture.get("raw_btc_candle_row_count") or 0),
        "raw_chainlink_price_row_count": int(capture.get("raw_chainlink_price_row_count") or 0),
        "orderbook_snapshot_interval_seconds": float(
            capture.get("orderbook_snapshot_interval_seconds") or 0.0
        ),
        "public_provider_timeout_seconds": float(
            capture.get("public_provider_timeout_seconds") or 0.0
        ),
        "public_provider_http_timeout_seconds": float(
            capture.get("public_provider_http_timeout_seconds") or 0.0
        ),
        "orderbook_ws_initial_complete_book_timeout_seconds": float(
            capture.get(
                "orderbook_ws_initial_complete_book_timeout_seconds"
            )
            or 0.0
        ),
        "rest_orderbook_fallback_collection_seconds": float(
            capture.get("rest_orderbook_fallback_collection_seconds") or 0.0
        ),
        "rest_orderbook_fallback_stops_at_market_close": (
            capture.get("rest_orderbook_fallback_stops_at_market_close") is True
        ),
        "reason_codes": sorted(set(reasons)),
    }


def _finalization_quality_reasons(
    finalization: dict[str, Any] | None,
) -> list[str]:
    if finalization is None:
        return ["capture_finalization_missing"]
    reasons: list[str] = []
    if finalization.get("finalization_status") != "exported":
        reasons.append("capture_finalization_not_exported")
    if finalization.get("pending_resolution") is not False:
        reasons.append("capture_resolution_pending")
    if finalization.get("training_eligible") is not True:
        reasons.append("capture_training_ineligible")
    if int(finalization.get("raw_resolution_count") or 0) <= 0:
        reasons.append("capture_resolution_evidence_unavailable")
    if dict(finalization.get("reject_reason_counts") or {}):
        reasons.append("capture_finalization_rejected")
    if not finalization.get("exported_training_corpus_dir"):
        reasons.append("exported_corpus_path_missing")
    return sorted(set(reasons))


def _outcome_blind_corpus_role_audit(
    *,
    corpus_dir: Path,
    prior_market_ids: set[str],
    minimum_decision_ts: int,
) -> dict[str, Any]:
    manifest_path = corpus_dir / "polymarket_corpus_manifest.json"
    feature_path = corpus_dir / "polymarket_feature_rows.jsonl"
    chainlink_path = corpus_dir / "polymarket_chainlink_prices.jsonl"
    chainlink_manifest_path = (
        corpus_dir / "polymarket_chainlink_decision_time_evidence_manifest.json"
    )
    training_provenance_path = corpus_dir / "training_corpus_provenance.json"
    reasons: list[str] = []
    if not manifest_path.is_file():
        reasons.append("corpus_manifest_missing")
    if not feature_path.is_file():
        reasons.append("feature_rows_missing")
    if not chainlink_path.is_file():
        reasons.append("chainlink_feature_evidence_missing")
    if not chainlink_manifest_path.is_file():
        reasons.append("chainlink_feature_evidence_manifest_missing")
    if not training_provenance_path.is_file():
        reasons.append("training_corpus_provenance_missing")
    if reasons:
        return {
            "market_id": "",
            "minimum_decision_ts": 0,
            "maximum_decision_ts": 0,
            "decision_row_count": 0,
            "corpus_manifest": None,
            "feature_rows": None,
            "chainlink_feature_evidence": None,
            "chainlink_feature_evidence_manifest": None,
            "training_corpus_provenance": None,
            "reason_codes": reasons,
        }
    manifest = _load_json(manifest_path)
    manifest_forbidden = sorted(_find_fields(manifest, FORBIDDEN_REGISTRY_FIELDS))
    if manifest_forbidden:
        reasons.append("corpus_manifest_contains_outcome_value")
    expected_feature_sha = str(
        (manifest.get("normalized_artifact_hashes") or {}).get("feature_rows") or ""
    )
    if expected_feature_sha != _sha256_file(feature_path):
        reasons.append("feature_rows_sha256_mismatch")
    features = _load_jsonl(feature_path)
    chainlink_rows = _load_jsonl(chainlink_path)
    chainlink_manifest = _load_json(chainlink_manifest_path)
    training_provenance = _load_json(training_provenance_path)
    normalized_hashes = manifest.get("normalized_artifact_hashes") or {}
    if str(normalized_hashes.get("chainlink_prices") or "") != _sha256_file(chainlink_path):
        reasons.append("chainlink_feature_evidence_sha256_mismatch")
    if str(
        normalized_hashes.get("chainlink_decision_time_evidence_manifest") or ""
    ) != _sha256_file(chainlink_manifest_path):
        reasons.append("chainlink_feature_evidence_manifest_sha256_mismatch")
    manifest_integration = manifest.get("chainlink_decision_time_feature_integration") or {}
    if manifest_integration != chainlink_manifest:
        reasons.append("chainlink_feature_integration_manifest_mismatch")
    if chainlink_manifest.get("source_type") != "polymarket_rtds_chainlink":
        reasons.append("chainlink_feature_source_type_invalid")
    if not chainlink_rows:
        reasons.append("chainlink_feature_evidence_empty")
    if int(chainlink_manifest.get("row_count") or 0) != len(chainlink_rows):
        reasons.append("chainlink_feature_evidence_row_count_mismatch")
    if chainlink_manifest.get("evidence_sha256") != _sha256_file(chainlink_path):
        reasons.append("chainlink_manifest_evidence_sha256_mismatch")
    if any(not _valid_chainlink_role_evidence_row(row) for row in chainlink_rows):
        reasons.append("chainlink_feature_evidence_row_invalid")
    if chainlink_manifest.get("decision_time_only") is not True:
        reasons.append("chainlink_feature_decision_time_contract_failed")
    if chainlink_manifest.get("feature_builder_integration_passed") is not True:
        reasons.append("chainlink_feature_builder_integration_failed")
    if chainlink_manifest.get("feature_builder_integration_required") is not False:
        reasons.append("chainlink_feature_builder_integration_still_required")
    if int(chainlink_manifest.get("timestamp_causality_violation_count") or 0) != 0:
        reasons.append("chainlink_feature_timestamp_causality_violation")
    if int(chainlink_manifest.get("integrated_feature_row_count") or 0) != len(features):
        reasons.append("chainlink_feature_row_coverage_incomplete")
    if int(chainlink_manifest.get("missing_or_invalid_feature_row_count") or 0) != 0:
        reasons.append("chainlink_feature_row_integration_invalid")
    provenance_chainlink = training_provenance.get("chainlink_decision_time_evidence") or {}
    if provenance_chainlink.get("attached") is not True:
        reasons.append("training_provenance_chainlink_not_attached")
    if provenance_chainlink.get("feature_builder_integration_passed") is not True:
        reasons.append("training_provenance_chainlink_integration_failed")
    if provenance_chainlink.get("feature_builder_integration_required") is not False:
        reasons.append("training_provenance_chainlink_integration_still_required")
    if provenance_chainlink.get("evidence_sha256") != _sha256_file(chainlink_path):
        reasons.append("training_provenance_chainlink_evidence_sha256_mismatch")
    if provenance_chainlink.get("manifest_sha256") != _sha256_file(chainlink_manifest_path):
        reasons.append("training_provenance_chainlink_manifest_sha256_mismatch")
    feature_forbidden = sorted(
        {field for row in features for field in _find_fields(row, FORBIDDEN_REGISTRY_FIELDS)}
    )
    if feature_forbidden:
        reasons.append("feature_rows_contain_outcome_value")
    market_ids = {str(row.get("market_id") or "") for row in features}
    if len(market_ids) != 1 or "" in market_ids:
        reasons.append("feature_market_identity_incomplete")
    decision_timestamps = [int(row.get("decision_ts") or 0) for row in features]
    if not decision_timestamps or any(value <= 0 for value in decision_timestamps):
        reasons.append("feature_decision_timestamp_incomplete")
    if any(
        int(row.get("max_input_ts") or 0) > int(row.get("decision_ts") or 0) for row in features
    ):
        reasons.append("feature_timestamp_causality_violation")
    for row in features:
        provenance = (row.get("feature_provenance") or {}).get(
            "reference_price_to_beat_distance_at_decision"
        ) or {}
        source_fields = str(provenance.get("source_fields_used") or "")
        if provenance.get("reference_price_to_beat_source") != (
            "polymarket_rtds_chainlink_market_start"
        ):
            reasons.append("feature_reference_distance_not_chainlink_sourced")
        if (
            "raw_polymarket_chainlink_prices.price_at_or_before_market_start" not in source_fields
            or "raw_polymarket_chainlink_prices.price_at_or_before_decision" not in source_fields
        ):
            reasons.append("feature_chainlink_source_fields_incomplete")
        if provenance.get("provenance_valid") is not True:
            reasons.append("feature_chainlink_provenance_invalid")
        if int(provenance.get("max_input_ts") or 0) > int(row.get("decision_ts") or 0):
            reasons.append("feature_chainlink_max_input_after_decision")
        if int(provenance.get("available_at_ts") or 0) > int(row.get("decision_ts") or 0):
            reasons.append("feature_chainlink_available_after_decision")
        if (row.get("features") or {}).get("reference_price_to_beat_distance_at_decision") is None:
            reasons.append("feature_chainlink_reference_distance_missing")
    if decision_timestamps and min(decision_timestamps) < minimum_decision_ts:
        reasons.append("feature_not_strictly_later_than_precollection_freeze")
    market_id = next(iter(market_ids), "")
    if market_id in prior_market_ids:
        reasons.append("feature_market_overlaps_prior_evidence")
    return {
        "market_id": market_id,
        "minimum_decision_ts": min(decision_timestamps, default=0),
        "maximum_decision_ts": max(decision_timestamps, default=0),
        "decision_row_count": len(features),
        "corpus_manifest": _descriptor(manifest_path),
        "feature_rows": _descriptor(feature_path),
        "chainlink_feature_evidence": _descriptor(chainlink_path),
        "chainlink_feature_evidence_manifest": _descriptor(chainlink_manifest_path),
        "training_corpus_provenance": _descriptor(training_provenance_path),
        "chainlink_feature_integration_passed": not any(
            "chainlink" in reason for reason in reasons
        ),
        "reason_codes": sorted(set(reasons)),
    }


def _execution_compatibility_audit(
    *,
    corpus_dir: Path,
    collector_contract: dict[str, Any],
) -> dict[str, Any]:
    """Validate frozen execution inputs without opening labels or outcomes."""

    feature_path = corpus_dir / "polymarket_feature_rows.jsonl"
    rows = _load_jsonl(feature_path)
    maximum_selected_staleness_ms = float(
        collector_contract["maximum_selected_side_book_staleness_ms"]
    )
    maximum_opposite_staleness_ms = float(
        collector_contract["maximum_opposite_side_book_staleness_ms"]
    )
    required_runtime_fields = tuple(
        str(value) for value in collector_contract.get("required_runtime_feature_fields") or []
    )
    row_audits: list[dict[str, Any]] = []
    blockers: list[str] = []
    for row in rows:
        features = dict(row.get("features") or {})
        decision_ts = int(row.get("decision_ts") or 0)
        max_input_ts = int(row.get("max_input_ts") or 0)
        row_reasons: list[str] = []
        if decision_ts <= 0 or max_input_ts > decision_ts:
            row_reasons.append("execution_compatibility_timestamp_causality_failed")
        missing_runtime = sorted(
            name for name in required_runtime_fields if not _finite_number(features.get(name))
        )
        if missing_runtime:
            row_reasons.append("execution_compatibility_runtime_fields_missing")
        for side in ("up", "down"):
            bid = _finite_float(features.get(f"{side}_bid"))
            ask = _finite_float(features.get(f"{side}_ask"))
            if bid is None or ask is None or not 0.0 < bid <= ask < 1.0:
                row_reasons.append(f"execution_compatibility_{side}_executable_book_invalid")
            staleness = _finite_float(features.get(f"{side}_book_staleness_ms"))
            limit = min(
                maximum_selected_staleness_ms,
                maximum_opposite_staleness_ms,
            )
            if staleness is None or staleness < 0.0 or staleness > limit:
                row_reasons.append(f"execution_compatibility_{side}_book_staleness_exceeded")
            queue = _finite_float(features.get(f"{side}_queue_fill_probability_proxy"))
            if queue is None or not 0.0 <= queue <= 1.0:
                row_reasons.append(f"execution_compatibility_{side}_queue_fill_invalid")
        if _finite_float(features.get("time_to_close_seconds"), minimum=0.0) is None:
            row_reasons.append("execution_compatibility_time_to_close_invalid")
        row_audits.append(
            {
                "market_id": str(row.get("market_id") or ""),
                "decision_ts": decision_ts,
                "max_input_ts": max_input_ts,
                "up_book_staleness_ms": features.get("up_book_staleness_ms"),
                "down_book_staleness_ms": features.get("down_book_staleness_ms"),
                "missing_runtime_fields": missing_runtime,
                "execution_compatible": not row_reasons,
                "reason_codes": sorted(set(row_reasons)),
            }
        )
        blockers.extend(row_reasons)
    if not rows:
        blockers.append("execution_compatibility_feature_rows_empty")
    return {
        "feature_rows": _descriptor(feature_path),
        "decision_row_count": len(rows),
        "execution_compatible_row_count": sum(
            int(row["execution_compatible"]) for row in row_audits
        ),
        "maximum_selected_side_book_staleness_ms": maximum_selected_staleness_ms,
        "maximum_opposite_side_book_staleness_ms": maximum_opposite_staleness_ms,
        "required_runtime_feature_fields": list(required_runtime_fields),
        "row_audits": row_audits,
        "blocking_reason_distribution": dict(sorted(Counter(blockers).items())),
        "blocking_reason_codes": sorted(set(blockers)),
        "labels_or_outcomes_opened": False,
    }


def _finite_number(value: Any) -> bool:
    return _finite_float(value) is not None


def _finite_float(value: Any, *, minimum: float | None = None) -> float | None:
    if not isinstance(value, int | float):
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or (minimum is not None and parsed < minimum):
        return None
    return parsed


def _role_for_rank(selection_rank: int) -> str:
    train_end = ROLE_MARKET_COUNTS["development_train"]
    calibration_end = train_end + ROLE_MARKET_COUNTS["development_calibration"]
    confirmatory_end = calibration_end + ROLE_MARKET_COUNTS["confirmatory_validation"]
    if 1 <= selection_rank <= train_end:
        return "development_train"
    if selection_rank <= calibration_end:
        return "development_calibration"
    if selection_rank <= confirmatory_end:
        return "confirmatory_validation"
    raise ValueError(f"selection rank is outside the frozen {TARGET_MARKET_COUNT}-market role plan")


def _valid_chainlink_role_evidence_row(row: dict[str, Any]) -> bool:
    try:
        price = float(row.get("price") or 0.0)
        source_ts = int(row.get("source_ts") or 0)
        available_at_ts = int(row.get("available_at_ts") or 0)
    except (TypeError, ValueError):
        return False
    return (
        row.get("source_type") == "polymarket_rtds_chainlink"
        and str(row.get("symbol") or "").lower() == "btc/usd"
        and price > 0.0
        and source_ts > 0
        and source_ts <= available_at_ts
    )


def _role_assignment_markdown(report: dict[str, Any]) -> str:
    role_counts = report["role_market_counts"]
    return "\n".join(
        [
            "# #175 Outcome-Blind Role Assignment",
            "",
            f"- status: `{report['status']}`",
            f"- selected markets: `{report['selected_market_count']}`",
            f"- train markets: `{role_counts.get('development_train', 0)}`",
            f"- calibration markets: `{role_counts.get('development_calibration', 0)}`",
            f"- confirmatory markets: `{role_counts.get('confirmatory_validation', 0)}`",
            "- labels or outcomes opened for role assignment: `false`",
            "- model fit started: `false`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )


def _execution_compatibility_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# #175 Execution-Compatible Feature Coverage",
            "",
            f"- selected markets: `{report['selected_market_count']}`",
            f"- selected decision rows: `{report['selected_decision_row_count']}`",
            "- execution-compatible decision rows: "
            f"`{report['execution_compatible_decision_row_count']}`",
            f"- maximum book staleness ms: `{report['maximum_book_staleness_ms']}`",
            "- labels or outcomes opened: `false`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )


def _verified_descriptor(payload: Any, *, name: str) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise ValueError(f"{name} descriptor is missing")
    path = Path(str(payload.get("path") or "")).resolve()
    expected_sha256 = str(payload.get("sha256") or "")
    _verify_pin(path, expected_sha256, name=name)
    return {"path": str(path), "sha256": expected_sha256.lower()}


def _extract_market_ids(payload: Any) -> set[str]:
    market_ids: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "market_id" and isinstance(value, str):
                market_ids.add(value)
            elif key in {
                "market_ids",
                "prior_market_ids",
                "selected_market_ids",
            } and isinstance(value, list):
                market_ids.update(str(item) for item in value)
            else:
                market_ids.update(_extract_market_ids(value))
    elif isinstance(payload, list):
        for value in payload:
            market_ids.update(_extract_market_ids(value))
    return market_ids


def _extract_decision_timestamps(payload: Any) -> list[int]:
    timestamps: list[int] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {
                "decision_ts",
                "minimum_decision_ts",
                "maximum_decision_ts",
                "maximum_prior_decision_ts",
            } and isinstance(value, (int, float)):
                timestamps.append(int(value))
            else:
                timestamps.extend(_extract_decision_timestamps(value))
    elif isinstance(payload, list):
        for value in payload:
            timestamps.extend(_extract_decision_timestamps(value))
    return timestamps


def _find_fields(payload: Any, forbidden: set[str], prefix: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key) in forbidden:
                found.add(path)
            found.update(_find_fields(value, forbidden, path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.update(_find_fields(value, forbidden, f"{prefix}[{index}]"))
    return found


def _normalize_pins(
    pins: tuple[tuple[Path | str, str], ...],
    *,
    name: str,
) -> tuple[tuple[Path, str], ...]:
    if not pins:
        raise ValueError(f"at least one {name} pin is required")
    normalized = []
    for path, digest in pins:
        _require_sha256(digest, name=f"{name} SHA-256")
        normalized.append((Path(path), digest.lower()))
    return tuple(normalized)


def _blocked_safety_fields() -> dict[str, Any]:
    return {
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }


def _freeze_markdown(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# #175 Precollection Role Freeze",
            "",
            f"- candidate: `{manifest['candidate_name']}`",
            f"- target valid markets: `{manifest['target_valid_market_count']}`",
            "- roles: `90 train / 45 calibration / 60 confirmatory`",
            f"- frozen feature contract SHA-256: `{manifest['feature_contract']['sha256']}`",
            f"- prior excluded markets: `{len(_load_json(Path(manifest['prior_evidence_exclusion_registry']['path']))['prior_market_ids'])}`",
            "- role assignment uses outcomes: `false`",
            "- model fit started: `false`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def _verify_pin(path: Path, expected_sha256: str, *, name: str) -> None:
    if not path.is_file():
        raise ValueError(f"{name} is missing: {path}")
    _require_sha256(expected_sha256, name=f"{name} SHA-256")
    if _sha256_file(path) != expected_sha256.lower():
        raise ValueError(f"{name} SHA-256 mismatch")


def _require_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise ValueError(f"{name} must be a SHA-256 digest")


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


def _load_json_or_jsonl(path: Path) -> Any:
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    return _load_json(path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")
