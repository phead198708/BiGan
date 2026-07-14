"""Frozen future accepted-bet evaluation for the v8 PnL-aligned candidate."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.contracts import compact_safety_fields
from bigan.v8.polymarket.training.execution_layer_v2_hts_residual_development_corpus import (
    PHASE2_MARKET_PROBABILITY_MAPPING_RULE_ID,
    _evaluation_pnl_components,
    _forbidden_decision_fields,
    _phase2_feature_to_public_row,
    _side_depth_imbalance,
    _side_feature,
)
from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_action_value import (
    FORBIDDEN_DECISION_FIELDS,
    REQUIRED_ACTIONS,
    _release_closed_shadow_positions,
    build_pnl_aligned_action_conditioned_rows,
    run_pnl_aligned_action_value_outcome_blind_shadow,
    validate_pnl_aligned_action_value_protocol,
)
from bigan.v8.polymarket.training.o_v8_paper_fresh_loop import (
    PINNED_ISSUE_160_MANIFEST_SHA256,
    _fresh_public_ranking_row_from_canonical,
    score_frozen_o_decision_rows,
)
from bigan.v8.polymarket.training.post_freeze_o_replay_aligned_source_ranking import (
    _v8_apply_simulated_order_to_state,
    _v8_execution_guard_config,
    _v8_execution_guard_decision,
    _v8_initial_runtime_state,
)

EVALUATION_SCHEMA_VERSION = "bigan-v8-execution-layer-v2-pnl-aligned-future-evaluation-protocol-v1"
REPORT_SCHEMA_VERSION = "bigan-v8-execution-layer-v2-pnl-aligned-future-accepted-bet-report-v1"
DECISION_INPUT_SCHEMA_VERSION = "bigan-v8-execution-layer-v2-pnl-aligned-future-decision-input-v1"
COLLECTION_HANDOFF_SCHEMA_VERSION = (
    "bigan-v8-execution-layer-v2-pnl-aligned-future-collection-handoff-v1"
)
SETTLEMENT_TARGET_SCHEMA_VERSION = (
    "bigan-v8-execution-layer-v2-pnl-aligned-future-settlement-target-v1"
)
PROHIBITED_FUTURE_OUTCOME_ARTIFACT_NAMES = (
    "polymarket_label_rows.jsonl",
    "polymarket_resolution_events.jsonl",
    "polymarket_settlement_events.jsonl",
    "current_clob_condition_settlement_pnl_rows.csv",
)


@dataclass(frozen=True, slots=True)
class PnLAlignedFutureEvaluationFreezeConfig:
    """Inputs frozen before future settlement targets are reconciled."""

    run_id: str
    output_dir: Path | str
    evaluation_protocol_path: Path | str
    expected_evaluation_protocol_sha256: str
    collection_freeze_manifest_path: Path | str
    expected_collection_freeze_manifest_sha256: str
    model_dir: Path | str
    git_commit: str

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for name, value in (
            ("expected_evaluation_protocol_sha256", self.expected_evaluation_protocol_sha256),
            (
                "expected_collection_freeze_manifest_sha256",
                self.expected_collection_freeze_manifest_sha256,
            ),
        ):
            if not _is_sha256(value):
                raise ValueError(f"{name} must be a SHA-256 digest")
        if len(self.git_commit) != 40 or any(
            char not in "0123456789abcdef" for char in self.git_commit.lower()
        ):
            raise ValueError("git_commit must be a 40-character hex digest")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "evaluation_protocol_path", Path(self.evaluation_protocol_path))
        object.__setattr__(
            self,
            "collection_freeze_manifest_path",
            Path(self.collection_freeze_manifest_path),
        )
        object.__setattr__(self, "model_dir", Path(self.model_dir))


@dataclass(frozen=True, slots=True)
class PnLAlignedFutureDecisionInputConfig:
    """Frozen inputs for outcome-blind Phase 2 decision-row construction."""

    run_id: str
    output_dir: Path | str
    collection_freeze_manifest_path: Path | str
    expected_collection_freeze_manifest_sha256: str
    source_corpus_dirs: tuple[Path | str, ...]
    paper_candidate_unlock_dir: Path | str
    expected_unlock_manifest_sha256: str = PINNED_ISSUE_160_MANIFEST_SHA256
    canonical_o_source_manifest_path: Path | str | None = None
    collection_handoff_manifest_path: Path | str | None = None
    expected_collection_handoff_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for name, value in (
            (
                "expected_collection_freeze_manifest_sha256",
                self.expected_collection_freeze_manifest_sha256,
            ),
            (
                "expected_unlock_manifest_sha256",
                self.expected_unlock_manifest_sha256,
            ),
        ):
            if not _is_sha256(value):
                raise ValueError(f"{name} must be a SHA-256 digest")
        if not self.source_corpus_dirs:
            raise ValueError("source_corpus_dirs must not be empty")
        if (self.collection_handoff_manifest_path is None) != (
            self.expected_collection_handoff_manifest_sha256 is None
        ):
            raise ValueError("collection handoff path and SHA-256 must be provided together")
        if self.expected_collection_handoff_manifest_sha256 is not None and not _is_sha256(
            self.expected_collection_handoff_manifest_sha256
        ):
            raise ValueError("expected_collection_handoff_manifest_sha256 must be a SHA-256 digest")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(
            self,
            "collection_freeze_manifest_path",
            Path(self.collection_freeze_manifest_path),
        )
        object.__setattr__(
            self,
            "source_corpus_dirs",
            tuple(Path(path) for path in self.source_corpus_dirs),
        )
        object.__setattr__(
            self, "paper_candidate_unlock_dir", Path(self.paper_candidate_unlock_dir)
        )
        if self.canonical_o_source_manifest_path is not None:
            object.__setattr__(
                self,
                "canonical_o_source_manifest_path",
                Path(self.canonical_o_source_manifest_path),
            )
        if self.collection_handoff_manifest_path is not None:
            object.__setattr__(
                self,
                "collection_handoff_manifest_path",
                Path(self.collection_handoff_manifest_path),
            )


@dataclass(frozen=True, slots=True)
class PnLAlignedFutureCollectionHandoffConfig:
    """Outcome-blind provenance gate from collector batch to O scoring."""

    run_id: str
    output_dir: Path | str
    batch_progress_path: Path | str
    expected_batch_progress_sha256: str
    collection_freeze_manifest_path: Path | str
    expected_collection_freeze_manifest_sha256: str
    training_corpus_root: Path | str

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for name, value in (
            ("expected_batch_progress_sha256", self.expected_batch_progress_sha256),
            (
                "expected_collection_freeze_manifest_sha256",
                self.expected_collection_freeze_manifest_sha256,
            ),
        ):
            if not _is_sha256(value):
                raise ValueError(f"{name} must be a SHA-256 digest")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "batch_progress_path", Path(self.batch_progress_path))
        object.__setattr__(
            self,
            "collection_freeze_manifest_path",
            Path(self.collection_freeze_manifest_path),
        )
        object.__setattr__(self, "training_corpus_root", Path(self.training_corpus_root))


@dataclass(frozen=True, slots=True)
class PnLAlignedFutureSettlementTargetConfig:
    """Post-shadow inputs for exactly-once future settlement targets."""

    run_id: str
    output_dir: Path | str
    shadow_manifest_path: Path | str
    expected_shadow_manifest_sha256: str

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if not _is_sha256(self.expected_shadow_manifest_sha256):
            raise ValueError("expected_shadow_manifest_sha256 must be a SHA-256 digest")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "shadow_manifest_path", Path(self.shadow_manifest_path))


def validate_pnl_aligned_future_evaluation_protocol(
    protocol: dict[str, Any],
) -> None:
    """Reject metric, baseline, tuning, or safety drift."""

    bootstrap = dict(protocol.get("market_bootstrap") or {})
    gates = dict(protocol.get("future_evidence_gates") or {})
    safety = dict(protocol.get("safety") or {})
    checks = {
        "schema": protocol.get("schema_version") == EVALUATION_SCHEMA_VERSION,
        "frozen": protocol.get("frozen") is True,
        "diagnostic_only": protocol.get("diagnostic_only") is True,
        "candidate": protocol.get("candidate_policy_name")
        == "pnl_aligned_action_conditioned_net_value_v1",
        "baseline": protocol.get("baseline_policy_name")
        == "raw_market_probability_selected_o_action_baseline",
        "baseline_action_source": protocol.get("baseline_action_source")
        == "canonical_o_rank_1_action",
        "baseline_not_calibrated_fair_value": protocol.get(
            "baseline_market_probability_is_calibrated_fair_value"
        )
        is False,
        "cost_rule": protocol.get("decision_time_execution_cost_rule_id")
        == "spread_queue_staleness_cost_proxy_v1",
        "threshold": float(protocol.get("frozen_entry_edge_threshold") or -1.0) == 0.02,
        "bootstrap": (
            int(bootstrap.get("resample_count") or 0) == 2000
            and int(bootstrap.get("seed") or 0) == 20260715
            and float(bootstrap.get("confidence_level") or 0.0) == 0.95
            and bootstrap.get("sampling_unit") == "market_id"
        ),
        "support": (
            int(gates.get("minimum_unique_market_count") or 0) == 30
            and int(gates.get("minimum_accepted_bet_count") or 0) == 30
            and int(gates.get("minimum_accepted_bet_count_per_side") or 0) == 10
            and gates.get("all_accepted_bets_must_be_settled") is True
        ),
        "outcomes_not_selection": protocol.get("outcome_fields_used_for_shadow_selection") is False,
        "outcomes_evaluation_only": protocol.get("outcome_fields_used_for_evaluation_only") is True,
        "no_future_feature_tuning": protocol.get("uses_future_outcomes_for_feature_selection")
        is False,
        "no_future_hyperparameter_tuning": protocol.get(
            "uses_future_outcomes_for_hyperparameter_selection"
        )
        is False,
        "no_future_threshold_tuning": protocol.get("uses_future_outcomes_for_threshold_selection")
        is False,
        "no_future_guard_tuning": protocol.get("uses_future_outcomes_for_guard_or_sizing_selection")
        is False,
        "no_source_score_mutation": protocol.get("source_o_score_mutation_allowed") is False,
        "no_source_ranking_mutation": protocol.get("source_ranking_mutation_allowed") is False,
        "safety": (
            safety.get("paper_only") is True
            and safety.get("capital_at_risk") is False
            and safety.get("polymarket_write_enabled") is False
            and safety.get("wallet_signing_enabled") is False
            and safety.get("source_model_candidate_eligible") is False
            and safety.get("freeze_ready") is False
            and safety.get("promotion_evidence_eligible") is False
            and safety.get("v8_execution_handoff_allowed") is False
            and safety.get("#134_resume_allowed") is False
            and safety.get("#146_start_allowed") is False
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("invalid future evaluation protocol: " + ", ".join(failed))


def freeze_pnl_aligned_future_evaluation(
    config: PnLAlignedFutureEvaluationFreezeConfig,
) -> dict[str, Any]:
    """Freeze all evaluator inputs and metrics before outcome reconciliation."""

    protocol_path = config.evaluation_protocol_path.resolve()
    collection_freeze_path = config.collection_freeze_manifest_path.resolve()
    if _sha256_file(protocol_path) != config.expected_evaluation_protocol_sha256:
        raise ValueError("future evaluation protocol SHA-256 mismatch")
    if _sha256_file(collection_freeze_path) != config.expected_collection_freeze_manifest_sha256:
        raise ValueError("collection freeze manifest SHA-256 mismatch")
    protocol = _load_json(protocol_path)
    validate_pnl_aligned_future_evaluation_protocol(protocol)
    collection_freeze = _load_json(collection_freeze_path)
    model_dir = config.model_dir.resolve()
    fit_manifest_path = model_dir / "pnl_aligned_action_value_fit_manifest.json"
    fit_manifest = _load_json(fit_manifest_path)
    for name, descriptor in (
        ("model", fit_manifest.get("model")),
        ("protocol", fit_manifest.get("protocol")),
    ):
        verified = _verified_descriptor(descriptor, name=name)
        if collection_freeze.get(name) != verified:
            raise ValueError(f"collection freeze {name} lineage mismatch")
    guard_config = _v8_execution_guard_config()
    if collection_freeze.get("execution_guard_config_sha256") != canonical_json_sha256(
        guard_config
    ):
        raise ValueError("collection freeze execution guard hash mismatch")
    if (
        collection_freeze.get("model_config_or_threshold_mutation_after_freeze_allowed")
        is not False
    ):
        raise ValueError("collection freeze mutation policy is not fail closed")

    output_dir = config.output_dir / config.run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": ("bigan-v8-execution-layer-v2-pnl-aligned-future-evaluation-freeze-v1"),
        "run_id": config.run_id,
        "freeze_created_ts": int(time.time() * 1000),
        "git_commit": config.git_commit.lower(),
        "evaluation_protocol": _descriptor(protocol_path),
        "collection_freeze_manifest": _descriptor(collection_freeze_path),
        "collection_freeze_id": collection_freeze["collection_freeze_id"],
        "model": fit_manifest["model"],
        "model_contract": fit_manifest["model_contract"],
        "model_protocol": fit_manifest["protocol"],
        "execution_guard_config": guard_config,
        "execution_guard_config_sha256": canonical_json_sha256(guard_config),
        "minimum_future_window_start_ts": collection_freeze["minimum_future_window_start_ts"],
        "prior_market_ids_sha256": collection_freeze["prior_market_ids_sha256"],
        "future_outcome_targets_loaded": False,
        "shadow_decisions_generated": False,
        "outcome_reconciliation_started": False,
        "exactly_once_evaluation_required": True,
        "future_results_may_mutate_protocol_model_threshold_guard_or_sizing": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    manifest["evaluation_freeze_id"] = canonical_json_sha256(manifest)
    manifest_path = output_dir / "pnl_aligned_future_evaluation_freeze_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "output_dir": output_dir,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "manifest": manifest,
    }


def build_pnl_aligned_future_collection_handoff(
    config: PnLAlignedFutureCollectionHandoffConfig,
) -> dict[str, Any]:
    """Freeze the exact completed collector batch without opening outcomes."""

    batch_path = config.batch_progress_path.resolve()
    freeze_path = config.collection_freeze_manifest_path.resolve()
    if _sha256_file(batch_path) != config.expected_batch_progress_sha256:
        raise ValueError("collector batch progress SHA-256 mismatch")
    if _sha256_file(freeze_path) != config.expected_collection_freeze_manifest_sha256:
        raise ValueError("collection freeze manifest SHA-256 mismatch")
    batch = _load_json(batch_path)
    collection_freeze = _load_json(freeze_path)
    if not (
        collection_freeze.get("future_collection_outcome_blind") is True
        and collection_freeze.get("future_window_must_be_strictly_later") is True
        and collection_freeze.get("future_market_ids_must_be_disjoint") is True
        and collection_freeze.get("model_config_or_threshold_mutation_after_freeze_allowed")
        is False
    ):
        raise ValueError("collection freeze is not outcome-blind and fail-closed")
    historical_descriptor = _verified_descriptor(
        collection_freeze.get("historical_development_rows"),
        name="historical_development_rows",
    )
    historical_rows = _load_jsonl(Path(historical_descriptor["path"]))
    prior_market_ids = {str(row["market_id"]) for row in historical_rows}
    expected_round_count = int(collection_freeze["expected_round_count"])
    captures = [dict(row) for row in batch.get("captures") or []]
    finalizations = [dict(row) for row in batch.get("finalizations") or []]
    errors = [dict(row) for row in batch.get("errors") or []]
    blockers: list[str] = []
    if batch.get("paper_only") is not True or batch.get("capital_at_risk") is not False:
        blockers.append("collector_batch_safety_contract_failed")
    if int(batch.get("capture_count") or 0) != len(captures):
        blockers.append("collector_reported_capture_count_mismatch")
    if len(captures) != expected_round_count:
        blockers.append("collector_capture_count_incomplete")
    if int(batch.get("error_count") or 0) != len(errors) or errors:
        blockers.append("collector_batch_errors_present")
    if int(batch.get("pending_resolution_count") or 0) != 0:
        blockers.append("collector_pending_resolution_present")
    capture_run_ids = [str(row.get("run_id") or "") for row in captures]
    capture_round_indices = [int(row.get("round_index") or 0) for row in captures]
    if len(capture_run_ids) != len(set(capture_run_ids)) or any(
        not value for value in capture_run_ids
    ):
        blockers.append("collector_duplicate_or_missing_capture_run_id")
    if sorted(capture_round_indices) != list(range(1, expected_round_count + 1)):
        blockers.append("collector_round_index_coverage_incomplete")
    for capture in captures:
        if capture.get("capture_start_boundary_validation_passed") is not True:
            blockers.append("collector_capture_start_boundary_failed")
        if int(capture.get("raw_polymarket_market_count") or 0) != 1:
            blockers.append("collector_market_row_coverage_failed")
        if int(capture.get("provider_raw_orderbook_snapshot_count") or 0) <= 0:
            blockers.append("collector_raw_orderbook_coverage_failed")
        if int(capture.get("training_sampled_orderbook_row_count") or 0) <= 0:
            blockers.append("collector_sampled_orderbook_coverage_failed")
        if int(capture.get("raw_chainlink_price_row_count") or 0) <= 0:
            blockers.append("collector_chainlink_coverage_failed")
        if capture.get("reject_reason_counts"):
            blockers.append("collector_capture_rejections_present")

    exported = [row for row in finalizations if row.get("finalization_status") == "exported"]
    if int(batch.get("exported_round_count") or 0) != len(exported):
        blockers.append("collector_reported_export_count_mismatch")
    if len(finalizations) != expected_round_count or len(exported) != expected_round_count:
        blockers.append("collector_export_count_incomplete")
    for finalization in finalizations:
        if not (
            finalization.get("finalization_status") == "exported"
            and finalization.get("pending_resolution") is False
            and finalization.get("training_eligible") is True
            and int(finalization.get("raw_resolution_count") or 0) > 0
            and not finalization.get("reject_reason_counts")
        ):
            blockers.append("collector_finalization_not_exported_and_eligible")
    finalization_run_ids = [str(row.get("run_id") or "") for row in finalizations]
    if len(finalization_run_ids) != len(set(finalization_run_ids)) or any(
        not value for value in finalization_run_ids
    ):
        blockers.append("collector_duplicate_or_missing_finalization_run_id")
    if set(capture_run_ids) != set(finalization_run_ids):
        blockers.append("collector_capture_finalization_identity_mismatch")

    training_root = config.training_corpus_root.expanduser().resolve()
    exported_paths = [
        Path(str(row.get("exported_training_corpus_dir") or "")).expanduser().resolve()
        for row in exported
        if row.get("exported_training_corpus_dir")
    ]
    if len(exported_paths) != len(exported):
        blockers.append("collector_exported_corpus_path_missing")
    if len(exported_paths) != len(set(exported_paths)):
        blockers.append("collector_duplicate_exported_corpus_path")
    safe_paths: list[Path] = []
    for path in sorted(set(exported_paths)):
        if not path.is_relative_to(training_root):
            blockers.append("collector_exported_corpus_outside_training_root")
        elif not path.is_dir():
            blockers.append("collector_exported_corpus_directory_missing")
        else:
            safe_paths.append(path)

    source_corpora: list[dict[str, Any]] = []
    source_market_ids: list[str] = []
    corpus_audits: list[dict[str, Any]] = []
    for corpus_dir in safe_paths:
        audit, public_rows, rejected = _load_outcome_blind_phase2_feature_corpus(
            corpus_dir=corpus_dir,
            prior_market_ids=prior_market_ids,
            minimum_future_window_start_ts=int(collection_freeze["minimum_future_window_start_ts"]),
        )
        corpus_audits.append(audit)
        market_ids = sorted({str(row["market_id"]) for row in public_rows})
        if rejected:
            blockers.append("collector_exported_corpus_feature_rows_rejected")
        if len(market_ids) != 1 or not public_rows:
            blockers.append("collector_exported_corpus_market_coverage_failed")
        source_market_ids.extend(market_ids)
        source_corpora.append(
            {
                "source_corpus_dir": str(corpus_dir),
                "market_ids": market_ids,
                "outcome_blind_decision_source_row_count": len(public_rows),
                "corpus_manifest": _descriptor(corpus_dir / "polymarket_corpus_manifest.json"),
                "feature_rows": _descriptor(corpus_dir / "polymarket_feature_rows.jsonl"),
                "market_metadata": _descriptor(corpus_dir / "polymarket_market_metadata.jsonl"),
                "chainlink_prices": _descriptor(corpus_dir / "polymarket_chainlink_prices.jsonl"),
                "chainlink_manifest": _descriptor(
                    corpus_dir / "polymarket_chainlink_decision_time_evidence_manifest.json"
                ),
                "training_corpus_provenance": _descriptor(
                    corpus_dir / "training_corpus_provenance.json"
                ),
            }
        )
    if len(source_corpora) != expected_round_count:
        blockers.append("collection_handoff_corpus_count_mismatch")
    if len(source_market_ids) != len(set(source_market_ids)):
        blockers.append("collection_handoff_duplicate_market_identity")
    if len(set(source_market_ids)) != expected_round_count:
        blockers.append("collection_handoff_unique_market_count_mismatch")
    blockers = sorted(set(blockers))
    status = "OUTCOME_BLIND_COLLECTION_HANDOFF_READY" if not blockers else "BLOCKED_FAIL_CLOSED"

    output_dir = config.output_dir / config.run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    access_audit_path = output_dir / "pnl_aligned_future_collection_handoff_access_audit.json"
    report_path = output_dir / "pnl_aligned_future_collection_handoff_report.json"
    access_audit = {
        "schema_version": f"{COLLECTION_HANDOFF_SCHEMA_VERSION}-access-audit",
        "permitted_artifact_names_opened": sorted(
            {name for audit in corpus_audits for name in audit["permitted_artifact_names_opened"]}
        ),
        "prohibited_future_outcome_artifact_names": list(PROHIBITED_FUTURE_OUTCOME_ARTIFACT_NAMES),
        "prohibited_future_outcome_artifacts_present_but_not_opened": sorted(
            {
                f"{audit['corpus_dir']}/{name}"
                for audit in corpus_audits
                for name in audit["prohibited_future_outcome_artifacts_present_but_not_opened"]
            }
        ),
        "prohibited_future_outcome_artifact_read_count": 0,
        "outcome_blind_input_access_passed": True,
        "paper_only": True,
        "capital_at_risk": False,
    }
    _write_json(access_audit_path, access_audit)
    report = {
        "schema_version": f"{COLLECTION_HANDOFF_SCHEMA_VERSION}-report",
        "run_id": config.run_id,
        "status": status,
        "blocking_reason_codes": blockers,
        "expected_round_count": expected_round_count,
        "capture_count": len(captures),
        "exported_round_count": len(exported),
        "source_corpus_count": len(source_corpora),
        "source_unique_market_count": len(set(source_market_ids)),
        "collector_error_count": len(errors),
        "collector_pending_resolution_count": int(batch.get("pending_resolution_count") or 0),
        "capture_start_boundary_validation_passed": all(
            row.get("capture_start_boundary_validation_passed") is True for row in captures
        ),
        "outcome_blind_input_access_passed": True,
        "future_outcome_targets_loaded": False,
        "outcome_reconciliation_started": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    _write_json(report_path, report)
    manifest = {
        "schema_version": f"{COLLECTION_HANDOFF_SCHEMA_VERSION}-manifest",
        "run_id": config.run_id,
        "status": status,
        "collection_handoff_ready": not blockers,
        "blocking_reason_codes": blockers,
        "batch_progress": _descriptor(batch_path),
        "collection_freeze_manifest": _descriptor(freeze_path),
        "training_corpus_root": str(training_root),
        "expected_round_count": expected_round_count,
        "source_corpus_dirs": [row["source_corpus_dir"] for row in source_corpora],
        "source_corpora": source_corpora,
        "source_market_ids_sha256": canonical_json_sha256(sorted(source_market_ids)),
        "collection_handoff_access_audit": _descriptor(access_audit_path),
        "collection_handoff_report": _descriptor(report_path),
        "future_outcome_targets_loaded": False,
        "outcome_reconciliation_started": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    manifest_path = output_dir / "pnl_aligned_future_collection_handoff_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "output_dir": output_dir,
        "access_audit_path": access_audit_path,
        "report_path": report_path,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "manifest": manifest,
        "report": report,
    }


def load_pnl_aligned_future_collection_handoff_source_dirs(
    path: Path | str,
    *,
    expected_sha256: str,
) -> tuple[Path, ...]:
    """Verify a ready collection handoff and return its exact corpus set."""

    handoff_path = Path(path).resolve()
    if _sha256_file(handoff_path) != expected_sha256:
        raise ValueError("collection handoff manifest SHA-256 mismatch")
    handoff = _load_json(handoff_path)
    if not (
        handoff.get("status") == "OUTCOME_BLIND_COLLECTION_HANDOFF_READY"
        and handoff.get("collection_handoff_ready") is True
        and handoff.get("blocking_reason_codes") == []
        and handoff.get("future_outcome_targets_loaded") is False
        and handoff.get("outcome_reconciliation_started") is False
    ):
        raise ValueError("collection handoff manifest is not ready")
    access_descriptor = _verified_descriptor(
        handoff.get("collection_handoff_access_audit"),
        name="collection_handoff_access_audit",
    )
    access = _load_json(Path(access_descriptor["path"]))
    if not (
        access.get("outcome_blind_input_access_passed") is True
        and access.get("prohibited_future_outcome_artifact_read_count") == 0
    ):
        raise ValueError("collection handoff outcome-blind access audit failed")
    source_corpora = [dict(row) for row in handoff.get("source_corpora") or []]
    source_dirs = tuple(Path(str(row["source_corpus_dir"])).resolve() for row in source_corpora)
    if [str(path) for path in source_dirs] != handoff.get("source_corpus_dirs"):
        raise ValueError("collection handoff source corpus directory set mismatch")
    if len(source_dirs) != int(handoff.get("expected_round_count") or 0):
        raise ValueError("collection handoff source corpus count mismatch")
    for row, corpus_dir in zip(source_corpora, source_dirs, strict=True):
        if not corpus_dir.is_dir():
            raise ValueError("collection handoff source corpus directory is missing")
        for name in (
            "corpus_manifest",
            "feature_rows",
            "market_metadata",
            "chainlink_prices",
            "chainlink_manifest",
            "training_corpus_provenance",
        ):
            _verified_descriptor(row.get(name), name=f"collection_handoff_{name}")
    return source_dirs


def build_pnl_aligned_future_outcome_blind_decision_inputs(
    config: PnLAlignedFutureDecisionInputConfig,
) -> dict[str, Any]:
    """Build future source rows without opening any outcome-bearing artifact."""

    freeze_path = config.collection_freeze_manifest_path.resolve()
    if _sha256_file(freeze_path) != config.expected_collection_freeze_manifest_sha256:
        raise ValueError("collection freeze manifest SHA-256 mismatch")
    collection_freeze = _load_json(freeze_path)
    if not (
        collection_freeze.get("future_collection_outcome_blind") is True
        and collection_freeze.get("future_window_must_be_strictly_later") is True
        and collection_freeze.get("future_market_ids_must_be_disjoint") is True
        and collection_freeze.get("model_config_or_threshold_mutation_after_freeze_allowed")
        is False
    ):
        raise ValueError("collection freeze is not outcome-blind and fail-closed")
    historical_descriptor = _verified_descriptor(
        collection_freeze.get("historical_development_rows"),
        name="historical_development_rows",
    )
    historical_rows = _load_jsonl(Path(historical_descriptor["path"]))
    prior_market_ids = sorted({str(row["market_id"]) for row in historical_rows})
    if (
        len(prior_market_ids) != int(collection_freeze["prior_market_count"])
        or canonical_json_sha256(prior_market_ids) != collection_freeze["prior_market_ids_sha256"]
        or max(int(row["decision_ts"]) for row in historical_rows)
        != int(collection_freeze["max_prior_decision_ts"])
    ):
        raise ValueError("collection freeze historical lineage mismatch")
    collection_handoff_descriptor: dict[str, str] | None = None
    if config.collection_handoff_manifest_path is not None:
        handoff_path = config.collection_handoff_manifest_path.resolve()
        handoff_source_dirs = load_pnl_aligned_future_collection_handoff_source_dirs(
            handoff_path,
            expected_sha256=str(config.expected_collection_handoff_manifest_sha256),
        )
        if tuple(path.resolve() for path in config.source_corpus_dirs) != handoff_source_dirs:
            raise ValueError("decision input source corpus set differs from collection handoff")
        handoff = _load_json(handoff_path)
        if handoff.get("collection_freeze_manifest") != _descriptor(freeze_path):
            raise ValueError("collection handoff freeze lineage mismatch")
        collection_handoff_descriptor = _descriptor(handoff_path)

    public_rows: list[dict[str, Any]] = []
    corpus_audits: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    for corpus_dir in sorted(path.resolve() for path in config.source_corpus_dirs):
        audit, rows, rejected = _load_outcome_blind_phase2_feature_corpus(
            corpus_dir=corpus_dir,
            prior_market_ids=set(prior_market_ids),
            minimum_future_window_start_ts=int(collection_freeze["minimum_future_window_start_ts"]),
        )
        corpus_audits.append(audit)
        public_rows.extend(rows)
        rejected_rows.extend(rejected)

    duplicate_decisions = sorted(
        key
        for key, count in Counter(
            (str(row["market_id"]), int(row["decision_ts"])) for row in public_rows
        ).items()
        if count > 1
    )
    scoring = (
        score_frozen_o_decision_rows(
            run_id=f"{config.run_id}-frozen-o-scoring",
            decision_rows=public_rows,
            paper_candidate_unlock_dir=config.paper_candidate_unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=(
                config.expected_unlock_manifest_sha256
            ),
            canonical_o_source_manifest_path=config.canonical_o_source_manifest_path,
        )
        if public_rows and not duplicate_decisions
        else {
            "scoring_passed": False,
            "canonical_scorer_report": {
                "canonical_scored_action_rows": [],
                "canonical_selected_decision_rows": [],
            },
        }
    )
    decision_rows: list[dict[str, Any]] = []
    if scoring["scoring_passed"] and not duplicate_decisions:
        scored_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for scored in scoring["canonical_scorer_report"]["canonical_scored_action_rows"]:
            scored_by_group[str(scored["decision_group_id"])].append(scored)
        public_by_decision = {
            (str(row["market_id"]), int(row["decision_ts"])): row for row in public_rows
        }
        for selected in scoring["canonical_scorer_report"]["canonical_selected_decision_rows"]:
            key = (str(selected["market_id"]), int(selected["decision_ts"]))
            decision_rows.append(
                _outcome_blind_future_decision_row(
                    selected=selected,
                    group_scored_rows=scored_by_group[str(selected["decision_group_id"])],
                    public_row=public_by_decision[key],
                    collection_freeze_id=str(collection_freeze["collection_freeze_id"]),
                )
            )
    decision_rows.sort(
        key=lambda row: (
            int(row["decision_ts"]),
            str(row["market_id"]),
            str(row["row_identity"]),
        )
    )

    forbidden_fields = sorted(
        {field for row in decision_rows for field in _find_forbidden_fields(row)}
    )
    causality_violations = [
        str(row["row_identity"])
        for row in decision_rows
        if int(row["max_input_ts"]) > int(row["decision_ts"])
    ]
    source_market_ids = sorted({str(row["market_id"]) for row in public_rows})
    expected_round_count = int(collection_freeze["expected_round_count"])
    blocking_reason_codes: list[str] = []
    if len(config.source_corpus_dirs) != expected_round_count:
        blocking_reason_codes.append("future_source_corpus_count_mismatch")
    if len(source_market_ids) != expected_round_count:
        blocking_reason_codes.append("future_unique_market_count_mismatch")
    if rejected_rows:
        blocking_reason_codes.append("future_source_rows_rejected")
    if duplicate_decisions:
        blocking_reason_codes.append("duplicate_future_decision_identity")
    if not scoring["scoring_passed"]:
        blocking_reason_codes.append("frozen_o_scoring_failed")
    if len(decision_rows) != len(public_rows):
        blocking_reason_codes.append("future_decision_row_count_mismatch")
    if forbidden_fields:
        blocking_reason_codes.append("forbidden_outcome_field_present")
    if causality_violations:
        blocking_reason_codes.append("future_decision_feature_causality_violation")
    blocking_reason_codes = sorted(set(blocking_reason_codes))
    canonical_context = dict(scoring.get("canonical_context") or {})
    canonical_o_lineage = {
        "source_manifest_path": canonical_context.get("source_manifest_path"),
        "source_manifest_sha256": canonical_context.get("source_manifest_sha256"),
        "ranking_objective_report_path": canonical_context.get("ranking_objective_report_path"),
        "ranking_objective_report_sha256": canonical_context.get("ranking_objective_report_sha256"),
        "feature_schema_hash": canonical_context.get("feature_schema_hash"),
        "ranking_correction_config_hash": canonical_context.get("ranking_correction_config_hash"),
        "ranking_correction_config_hash_verified": canonical_context.get(
            "ranking_correction_config_hash_verified"
        ),
        "canonical_inputs_available": canonical_context.get("canonical_inputs_available"),
    }

    output_dir = config.output_dir / config.run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    decision_rows_path = output_dir / "pnl_aligned_future_decision_rows.jsonl"
    rejected_rows_path = output_dir / "pnl_aligned_future_rejected_rows.jsonl"
    access_audit_path = output_dir / "pnl_aligned_future_input_access_audit.json"
    report_path = output_dir / "pnl_aligned_future_decision_input_report.json"
    _write_jsonl(decision_rows_path, decision_rows)
    _write_jsonl(rejected_rows_path, rejected_rows)
    input_access_audit = {
        "schema_version": f"{DECISION_INPUT_SCHEMA_VERSION}-access-audit",
        "permitted_artifact_names_opened": sorted(
            {name for audit in corpus_audits for name in audit["permitted_artifact_names_opened"]}
        ),
        "prohibited_future_outcome_artifact_names": list(PROHIBITED_FUTURE_OUTCOME_ARTIFACT_NAMES),
        "prohibited_future_outcome_artifacts_present_but_not_opened": sorted(
            {
                name
                for audit in corpus_audits
                for name in audit["prohibited_future_outcome_artifacts_present_but_not_opened"]
            }
        ),
        "prohibited_future_outcome_artifact_read_count": 0,
        "label_rows_required": False,
        "resolution_events_required": False,
        "future_outcome_targets_loaded": False,
        "forbidden_decision_field_violation_count": len(forbidden_fields),
        "forbidden_decision_fields": forbidden_fields,
        "feature_max_input_ts_violation_count": len(causality_violations),
        "feature_max_input_ts_violations": causality_violations,
        "outcome_blind_input_access_passed": not forbidden_fields and not causality_violations,
        **compact_safety_fields(),
    }
    _write_json(access_audit_path, input_access_audit)
    report = {
        "schema_version": f"{DECISION_INPUT_SCHEMA_VERSION}-report",
        "run_id": config.run_id,
        "status": (
            "OUTCOME_BLIND_FUTURE_DECISION_INPUT_READY"
            if not blocking_reason_codes
            else "BLOCKED_FAIL_CLOSED"
        ),
        "collection_freeze_id": collection_freeze["collection_freeze_id"],
        "collection_freeze_manifest_sha256": (config.expected_collection_freeze_manifest_sha256),
        "expected_round_count": expected_round_count,
        "source_corpus_count": len(config.source_corpus_dirs),
        "source_unique_market_count": len(source_market_ids),
        "collection_handoff_manifest": collection_handoff_descriptor,
        "collection_handoff_verified": collection_handoff_descriptor is not None,
        "source_decision_count": len(public_rows),
        "outcome_blind_decision_row_count": len(decision_rows),
        "source_corpus_audits": corpus_audits,
        "rejected_row_count": len(rejected_rows),
        "rejected_reason_distribution": dict(
            sorted(Counter(row["reason_code"] for row in rejected_rows).items())
        ),
        "duplicate_decision_identities": [
            {"market_id": market_id, "decision_ts": decision_ts}
            for market_id, decision_ts in duplicate_decisions
        ],
        "frozen_o_scoring_passed": scoring["scoring_passed"],
        "frozen_o_source_lineage": canonical_o_lineage,
        "complete_5_action_ranking_count": sum(
            {
                row["selected_action"]
                for row in decision["execution_handoff_context"]["full_5_action_ranking"]
            }
            == set(REQUIRED_ACTIONS)
            for decision in decision_rows
        ),
        "future_window_time_validation_passed": all(
            int(row["decision_ts"]) >= int(collection_freeze["minimum_future_window_start_ts"])
            for row in decision_rows
        ),
        "future_market_disjointness_passed": not (set(source_market_ids) & set(prior_market_ids)),
        "future_outcome_targets_loaded": False,
        "outcome_reconciliation_started": False,
        "outcome_blind_input_access_passed": input_access_audit[
            "outcome_blind_input_access_passed"
        ],
        "blocking_reason_codes": blocking_reason_codes,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    _write_json(report_path, report)
    manifest = {
        "schema_version": f"{DECISION_INPUT_SCHEMA_VERSION}-manifest",
        "run_id": config.run_id,
        "collection_freeze_manifest": _descriptor(freeze_path),
        "collection_handoff_manifest": collection_handoff_descriptor,
        "collection_handoff_verified": collection_handoff_descriptor is not None,
        "source_corpora": [
            {
                "corpus_id": audit["corpus_id"],
                "corpus_dir": audit["corpus_dir"],
                "corpus_manifest_sha256": audit["corpus_manifest_sha256"],
                "feature_rows_sha256": audit["feature_rows_sha256"],
                "market_metadata_sha256": audit["market_metadata_sha256"],
                "chainlink_evidence_sha256": audit["chainlink_evidence_sha256"],
            }
            for audit in corpus_audits
        ],
        "frozen_o_source_lineage": canonical_o_lineage,
        "decision_rows": _descriptor(decision_rows_path),
        "rejected_rows": _descriptor(rejected_rows_path),
        "input_access_audit": _descriptor(access_audit_path),
        "decision_input_report": _descriptor(report_path),
        "future_outcome_targets_loaded": False,
        "outcome_reconciliation_started": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    manifest_path = output_dir / "pnl_aligned_future_decision_input_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "output_dir": output_dir,
        "decision_rows_path": decision_rows_path,
        "access_audit_path": access_audit_path,
        "report_path": report_path,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "decision_rows": decision_rows,
        "report": report,
    }


def _load_outcome_blind_phase2_feature_corpus(
    *,
    corpus_dir: Path,
    prior_market_ids: set[str],
    minimum_future_window_start_ts: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    permitted_names = (
        "polymarket_corpus_manifest.json",
        "polymarket_feature_rows.jsonl",
        "polymarket_market_metadata.jsonl",
        "polymarket_chainlink_prices.jsonl",
        "polymarket_chainlink_decision_time_evidence_manifest.json",
        "training_corpus_provenance.json",
    )
    missing = sorted(name for name in permitted_names if not (corpus_dir / name).is_file())
    if missing:
        raise ValueError(f"missing outcome-blind Phase 2 artifacts in {corpus_dir}: {missing}")
    manifest_path = corpus_dir / "polymarket_corpus_manifest.json"
    feature_path = corpus_dir / "polymarket_feature_rows.jsonl"
    metadata_path = corpus_dir / "polymarket_market_metadata.jsonl"
    chainlink_path = corpus_dir / "polymarket_chainlink_prices.jsonl"
    chainlink_manifest_path = (
        corpus_dir / "polymarket_chainlink_decision_time_evidence_manifest.json"
    )
    provenance_path = corpus_dir / "training_corpus_provenance.json"
    corpus_manifest = _load_json(manifest_path)
    normalized_hashes = dict(corpus_manifest.get("normalized_artifact_hashes") or {})
    for key, path in (
        ("feature_rows", feature_path),
        ("market_metadata", metadata_path),
    ):
        if normalized_hashes.get(key) != _sha256_file(path):
            raise ValueError(f"Phase 2 normalized artifact hash mismatch: {path.name}")
    chainlink_manifest = _load_json(chainlink_manifest_path)
    if chainlink_manifest.get("evidence_sha256") != _sha256_file(chainlink_path):
        raise ValueError("Chainlink evidence SHA-256 mismatch")
    if chainlink_manifest.get("timestamp_causality_violation_count") != 0:
        raise ValueError("Chainlink evidence reports timestamp causality violations")

    features = _load_jsonl(feature_path)
    metadata_rows = _load_jsonl(metadata_path)
    chainlink_rows = _load_jsonl(chainlink_path)
    provenance = _load_json(provenance_path)
    corpus_id = str(provenance.get("corpus_id") or corpus_dir.name)
    source_run_id = str(provenance.get("run_id") or corpus_id)
    metadata_by_market = {str(row["market_id"]): row for row in metadata_rows}
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    lineage = {
        "source_corpus_dir": str(corpus_dir),
        "source_corpus_manifest_sha256": _sha256_file(manifest_path),
        "source_feature_rows_sha256": _sha256_file(feature_path),
        "source_market_metadata_sha256": _sha256_file(metadata_path),
        "source_chainlink_evidence_sha256": _sha256_file(chainlink_path),
        "market_probability_mapping_rule_id": (PHASE2_MARKET_PROBABILITY_MAPPING_RULE_ID),
        "outcome_artifact_read_count": 0,
    }
    for feature_row in features:
        market_id = str(feature_row.get("market_id") or "")
        decision_ts = int(feature_row.get("decision_ts") or 0)
        reasons: list[str] = []
        if market_id in prior_market_ids:
            reasons.append("future_market_overlaps_historical_fit")
        if decision_ts < minimum_future_window_start_ts:
            reasons.append("decision_before_frozen_future_window")
        if int(feature_row.get("max_input_ts") or 0) > decision_ts:
            reasons.append("phase2_feature_causality_violation")
        metadata = metadata_by_market.get(market_id)
        if metadata is None:
            reasons.append("market_metadata_missing")
        elif not (
            int(metadata.get("market_start_ts") or 0)
            < decision_ts
            < int(metadata.get("market_end_ts") or 0)
        ):
            reasons.append("invalid_decision_time_market_schedule")
        if reasons:
            rejected.extend(
                _future_input_rejection(market_id, decision_ts, reason)
                for reason in sorted(reasons)
            )
            continue
        public_row = _phase2_feature_to_public_row(
            run_id=source_run_id,
            row_index=len(rows),
            feature_row=feature_row,
            market=dict(metadata),
            chainlink_rows=chainlink_rows,
        )
        mapping = dict(public_row.get("market_probability_mapping_provenance") or {})
        if (
            public_row.get("market_probability_mapping_rule_id")
            != PHASE2_MARKET_PROBABILITY_MAPPING_RULE_ID
            or mapping.get("provenance_valid") is not True
        ):
            rejected.append(
                _future_input_rejection(
                    market_id,
                    decision_ts,
                    "market_probability_mapping_contract_violation",
                )
            )
            continue
        required_chainlink_fields = (
            "chainlink_price_at_decision",
            "chainlink_reference_price_at_market_start",
            "chainlink_reference_distance_at_decision",
            "chainlink_momentum_30s",
            "chainlink_momentum_60s",
            "chainlink_momentum_120s",
            "chainlink_realized_volatility_120s",
        )
        missing_chainlink = sorted(
            field
            for field in required_chainlink_fields
            if not isinstance(public_row.get(field), int | float)
            or not math.isfinite(float(public_row[field]))
        )
        chainlink_provenance = dict(public_row.get("chainlink_regime_feature_provenance") or {})
        if missing_chainlink or chainlink_provenance.get("provenance_valid") is not True:
            rejected.append(
                _future_input_rejection(
                    market_id,
                    decision_ts,
                    "complete_causal_chainlink_feature_block_missing",
                    missing_fields=missing_chainlink,
                )
            )
            continue
        if _forbidden_decision_fields(public_row):
            rejected.append(
                _future_input_rejection(
                    market_id,
                    decision_ts,
                    "forbidden_decision_field_present",
                )
            )
            continue
        if int(public_row["decision_time_feature_max_input_ts"]) > decision_ts:
            rejected.append(
                _future_input_rejection(
                    market_id,
                    decision_ts,
                    "joined_feature_causality_violation",
                )
            )
            continue
        public_row["future_source_run_id"] = source_run_id
        public_row["future_source_lineage"] = dict(lineage)
        rows.append(public_row)
    audit = {
        "corpus_id": corpus_id,
        "corpus_dir": str(corpus_dir),
        "corpus_manifest_sha256": lineage["source_corpus_manifest_sha256"],
        "feature_rows_sha256": lineage["source_feature_rows_sha256"],
        "market_metadata_sha256": lineage["source_market_metadata_sha256"],
        "chainlink_evidence_sha256": lineage["source_chainlink_evidence_sha256"],
        "feature_row_count": len(features),
        "outcome_blind_public_row_count": len(rows),
        "rejected_row_count": len(rejected),
        "permitted_artifact_names_opened": list(permitted_names),
        "prohibited_future_outcome_artifact_names": list(PROHIBITED_FUTURE_OUTCOME_ARTIFACT_NAMES),
        "prohibited_future_outcome_artifacts_present_but_not_opened": sorted(
            name
            for name in PROHIBITED_FUTURE_OUTCOME_ARTIFACT_NAMES
            if (corpus_dir / name).is_file()
        ),
        "prohibited_future_outcome_artifact_read_count": 0,
        "label_rows_required": False,
        "resolution_events_required": False,
        "phase2_feature_and_metadata_hashes_verified": True,
        "chainlink_hash_and_causality_verified": True,
        "paper_only": True,
        "capital_at_risk": False,
    }
    return audit, rows, rejected


def _outcome_blind_future_decision_row(
    *,
    selected: dict[str, Any],
    group_scored_rows: list[dict[str, Any]],
    public_row: dict[str, Any],
    collection_freeze_id: str,
) -> dict[str, Any]:
    actions = {str(row.get("action") or "") for row in group_scored_rows}
    if actions != set(REQUIRED_ACTIONS):
        raise ValueError("frozen O scorer did not produce a complete five-action grid")
    action = str(selected["action"])
    side = "UP" if "BUY_UP" in action else "DOWN" if "BUY_DOWN" in action else "NONE"
    family = _family(action)
    ranking = [
        _fresh_public_ranking_row_from_canonical(row)
        for row in sorted(
            group_scored_rows,
            key=lambda row: (
                int(row.get("canonical_rank") or 999),
                str(row.get("action") or ""),
            ),
        )
    ]
    selected_ranking = next(row for row in ranking if row["selected_action"] == action)
    selected_score = float(selected["canonical_corrected_model_score"])
    second_best_score = max(
        float(row["canonical_corrected_model_score"])
        for row in group_scored_rows
        if str(row["action"]) != action
    )
    micro = dict(selected_ranking["microstructure_snapshot"])
    probability = (
        float(public_row["p_up"])
        if side == "UP"
        else float(public_row["p_down"])
        if side == "DOWN"
        else 0.0
    )
    chainlink_provenance = dict(public_row.get("chainlink_regime_feature_provenance") or {})
    decision_ts = int(public_row["decision_ts"])
    max_input_ts = max(
        int(public_row["decision_time_feature_max_input_ts"]),
        int(chainlink_provenance.get("max_input_ts") or 0),
        int(selected.get("canonical_feature_mapping_max_input_ts") or 0),
    )
    source_lineage = dict(public_row["future_source_lineage"])
    features = {
        "canonical_o_action_score": selected_score,
        "action_score_margin": selected_score - second_best_score,
        "btc_momentum": float(public_row["chainlink_momentum_60s"]),
        "reference_price_to_beat_distance_at_decision": float(
            public_row["chainlink_reference_distance_at_decision"]
        ),
        "chainlink_momentum_30s": float(public_row["chainlink_momentum_30s"]),
        "chainlink_momentum_60s": float(public_row["chainlink_momentum_60s"]),
        "chainlink_momentum_120s": float(public_row["chainlink_momentum_120s"]),
        "chainlink_realized_volatility_120s": float(
            public_row["chainlink_realized_volatility_120s"]
        ),
        "selected_side_probability": probability,
        "execution_price": float(micro.get("entry_ask") or 0.0),
        "selected_side_probability_minus_execution_price": probability
        - float(micro.get("entry_ask") or 0.0),
        "spread_bps": float(micro.get("spread_bps") or 0.0),
        "queue_fill_proxy": float(micro.get("queue_fill_proxy") or 0.0),
        "book_staleness_ms": float(micro.get("book_staleness_ms") or 0.0),
        "time_to_close_seconds": float(micro.get("time_to_close_seconds") or 0.0),
        "side_book_depth_imbalance": (
            _side_depth_imbalance(public_row, side) if side in {"UP", "DOWN"} else 0.0
        ),
        "side_book_update_count_1m": (
            _side_feature(public_row, side, "recent_book_update_count_1m")
            if side in {"UP", "DOWN"}
            else 0.0
        ),
        "side_recent_spread_stability_1m": (
            _side_feature(public_row, side, "recent_spread_stability_1m")
            if side in {"UP", "DOWN"}
            else 0.0
        ),
        "cumulative_market_exposure_before_entry": 0.0,
        "same_side_reentry": 0.0,
        "side_flip": 0.0,
    }
    row = {
        "schema_version": DECISION_INPUT_SCHEMA_VERSION,
        "market_id": str(public_row["market_id"]),
        "condition_id": str(public_row["condition_id"]),
        "market_slug": str(public_row["slug"]),
        "decision_ts": decision_ts,
        "market_close_ts": int(public_row["market_end_ts"]),
        "max_input_ts": max_input_ts,
        "selected_action": action,
        "selected_side": side,
        "action_family": family,
        "decision_time_features": features,
        "execution_handoff_context": {
            "decision_group_id": public_row["decision_group_id"],
            "market_id": str(public_row["market_id"]),
            "decision_ts": decision_ts,
            "selected_action": action,
            "selected_side": side,
            "selected_action_family": family,
            "full_5_action_ranking": ranking,
            "corrected_model_score": selected_score,
            "raw_model_score": selected.get("canonical_raw_model_score"),
            "high_score_flag": bool(selected.get("high_score_flag")),
            "p_up": float(public_row["p_up"]),
            "p_down": float(public_row["p_down"]),
            "p_up_action_disagreement": bool(selected.get("p_up_action_disagreement")),
            "microstructure_snapshot": micro,
            "reference_price_feature_provenance": chainlink_provenance,
            "decision_time_feature_max_input_ts": max_input_ts,
        },
        "source_run_id": str(public_row["future_source_run_id"]),
        "source_lineage": source_lineage,
        "collection_freeze_id": collection_freeze_id,
        "future_outcome_targets_loaded": False,
        "target_used_as_decision_input": False,
        "source_o_score_mutated": False,
        "source_ranking_mutated": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    row["row_identity"] = canonical_json_sha256(
        {
            "market_id": row["market_id"],
            "decision_ts": decision_ts,
            "source_run_id": row["source_run_id"],
            "source_feature_rows_sha256": source_lineage["source_feature_rows_sha256"],
            "collection_freeze_id": collection_freeze_id,
        }
    )
    row["row_content_sha256"] = canonical_json_sha256(row)
    return row


def _future_input_rejection(
    market_id: str,
    decision_ts: int,
    reason_code: str,
    **details: Any,
) -> dict[str, Any]:
    return {
        "market_id": market_id,
        "decision_ts": decision_ts,
        "reason_code": reason_code,
        **details,
    }


def build_pnl_aligned_future_settled_evaluation_targets(
    config: PnLAlignedFutureSettlementTargetConfig,
) -> dict[str, Any]:
    """Load post-close targets exactly once after frozen shadows exist."""

    shadow_path = config.shadow_manifest_path.resolve()
    if _sha256_file(shadow_path) != config.expected_shadow_manifest_sha256:
        raise ValueError("shadow manifest SHA-256 mismatch")
    shadow_manifest = _load_json(shadow_path)
    if not (
        shadow_manifest.get("future_outcome_targets_loaded") is False
        and shadow_manifest.get("outcome_reconciliation_started") is False
    ):
        raise ValueError("shadow manifest is not outcome blind")
    decision_input_descriptor = _verified_descriptor(
        shadow_manifest.get("decision_input_manifest"),
        name="decision_input_manifest",
    )
    decision_input_manifest = _load_json(Path(decision_input_descriptor["path"]))
    if not (
        decision_input_manifest.get("future_outcome_targets_loaded") is False
        and decision_input_manifest.get("outcome_reconciliation_started") is False
    ):
        raise ValueError("decision input manifest is not outcome blind")
    decision_rows_descriptor = _verified_descriptor(
        decision_input_manifest.get("decision_rows"), name="decision_rows"
    )
    if decision_rows_descriptor != _verified_descriptor(
        shadow_manifest.get("input_decision_rows"), name="shadow_input_decision_rows"
    ):
        raise ValueError("shadow decision-input lineage mismatch")
    candidate_descriptor = _verified_descriptor(
        shadow_manifest.get("candidate_shadow_rows"), name="candidate_shadow_rows"
    )
    baseline_descriptor = _verified_descriptor(
        shadow_manifest.get("baseline_shadow_rows"), name="baseline_shadow_rows"
    )
    decision_rows = _load_jsonl(Path(decision_rows_descriptor["path"]))
    candidate_rows = _load_jsonl(Path(candidate_descriptor["path"]))
    baseline_rows = _load_jsonl(Path(baseline_descriptor["path"]))
    decision_ids = [str(row["row_identity"]) for row in decision_rows]
    candidate_ids = [str(row["source_row_identity"]) for row in candidate_rows]
    baseline_ids = [str(row["source_row_identity"]) for row in baseline_rows]
    if not (
        len(decision_ids) == len(set(decision_ids))
        and len(candidate_ids) == len(set(candidate_ids))
        and len(baseline_ids) == len(set(baseline_ids))
        and set(decision_ids) == set(candidate_ids) == set(baseline_ids)
    ):
        raise ValueError("shadow and decision-input identities do not match")
    reconciliation_started_ts = int(time.time() * 1000)
    if not decision_rows or any(
        int(row["market_close_ts"]) >= reconciliation_started_ts for row in decision_rows
    ):
        raise ValueError("settlement target loading attempted before all markets closed")
    output_dir = config.output_dir / config.run_id
    if output_dir.exists():
        raise FileExistsError(f"settlement target output directory exists: {output_dir}")
    marker_path = shadow_path.parent / "pnl_aligned_future_outcome_reconciliation_started.json"
    if marker_path.exists():
        raise ValueError("outcome reconciliation already started for this shadow")

    corpus_preflight: dict[str, dict[str, Any]] = {}
    for row in decision_rows:
        lineage = dict(row.get("source_lineage") or {})
        corpus_dir = Path(str(lineage.get("source_corpus_dir") or ""))
        if not corpus_dir.is_dir():
            raise ValueError("decision row source corpus directory is missing")
        corpus_manifest_path = corpus_dir / "polymarket_corpus_manifest.json"
        feature_path = corpus_dir / "polymarket_feature_rows.jsonl"
        label_path = corpus_dir / "polymarket_label_rows.jsonl"
        resolution_path = corpus_dir / "polymarket_resolution_events.jsonl"
        if not label_path.is_file() or not resolution_path.is_file():
            raise ValueError("post-close target artifacts are missing")
        if (
            not corpus_manifest_path.is_file()
            or _sha256_file(corpus_manifest_path) != lineage.get("source_corpus_manifest_sha256")
            or not feature_path.is_file()
            or _sha256_file(feature_path) != lineage.get("source_feature_rows_sha256")
        ):
            raise ValueError("decision row source corpus lineage mismatch")
        corpus_preflight[str(corpus_dir)] = {
            "corpus_dir": corpus_dir,
            "corpus_manifest_path": corpus_manifest_path,
            "feature_path": feature_path,
            "label_path": label_path,
            "resolution_path": resolution_path,
        }

    marker = {
        "schema_version": f"{SETTLEMENT_TARGET_SCHEMA_VERSION}-start-marker",
        "run_id": config.run_id,
        "reconciliation_started_ts": reconciliation_started_ts,
        "shadow_manifest": _descriptor(shadow_path),
        "decision_identity_count": len(decision_ids),
        "future_outcome_targets_loaded_before_marker": False,
        "outcome_reconciliation_started": True,
        "exactly_once": True,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    _write_json_exclusive(marker_path, marker)

    labels_by_decision: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    resolution_by_market: dict[str, dict[str, Any]] = {}
    source_target_artifacts: list[dict[str, Any]] = []
    for entry in corpus_preflight.values():
        corpus_manifest = _load_json(entry["corpus_manifest_path"])
        normalized_hashes = dict(corpus_manifest.get("normalized_artifact_hashes") or {})
        label_path = entry["label_path"]
        resolution_path = entry["resolution_path"]
        if normalized_hashes.get("label_rows") != _sha256_file(label_path):
            raise ValueError("Phase 2 label artifact hash mismatch after marker")
        if normalized_hashes.get("resolution_events") != _sha256_file(resolution_path):
            raise ValueError("Phase 2 resolution artifact hash mismatch after marker")
        labels = _load_jsonl(label_path)
        resolutions = _load_jsonl(resolution_path)
        for label in labels:
            labels_by_decision[(str(label["market_id"]), int(label["decision_ts"]))].append(label)
        for resolution in resolutions:
            market_id = str(resolution["market_id"])
            if market_id in resolution_by_market:
                raise ValueError("duplicate official resolution row")
            resolution_by_market[market_id] = resolution
        source_target_artifacts.append(
            {
                "source_corpus_dir": str(entry["corpus_dir"]),
                "corpus_manifest": _descriptor(entry["corpus_manifest_path"]),
                "feature_rows": _descriptor(entry["feature_path"]),
                "label_rows": _descriptor(label_path),
                "resolution_events": _descriptor(resolution_path),
            }
        )

    targets: list[dict[str, Any]] = []
    for decision in decision_rows:
        market_id = str(decision["market_id"])
        decision_ts = int(decision["decision_ts"])
        labels = labels_by_decision.get((market_id, decision_ts), [])
        labels_by_action = {str(row.get("action") or ""): row for row in labels}
        if len(labels) != len(REQUIRED_ACTIONS) or set(labels_by_action) != set(REQUIRED_ACTIONS):
            raise ValueError("settled evaluation action target grid is incomplete")
        resolution = resolution_by_market.get(market_id)
        if resolution is None or resolution.get("resolved_outcome") not in {
            "UP",
            "DOWN",
        }:
            raise ValueError("official resolved outcome is unavailable")
        resolved_outcome = str(resolution["resolved_outcome"])
        if any(
            label.get("resolved_outcome") != resolved_outcome
            or label.get("raw_resolution_sha256") != resolution.get("raw_resolution_sha256")
            for label in labels
        ):
            raise ValueError("label and official resolution provenance mismatch")
        target_values = {
            action: float(labels_by_action[action]["total_net_pnl_per_notional"])
            for action in REQUIRED_ACTIONS
        }
        components = {
            action: _evaluation_pnl_components(labels_by_action[action])
            for action in REQUIRED_ACTIONS
        }
        if any(
            not math.isfinite(value)
            for value in [
                *target_values.values(),
                *(
                    value
                    for action_components in components.values()
                    for value in action_components.values()
                ),
            ]
        ):
            raise ValueError("settled evaluation target contains non-finite values")
        target = {
            "schema_version": SETTLEMENT_TARGET_SCHEMA_VERSION,
            "row_identity": str(decision["row_identity"]),
            "market_id": market_id,
            "decision_ts": decision_ts,
            "market_close_ts": int(decision["market_close_ts"]),
            "resolved_outcome": resolved_outcome,
            "evaluation_target_net_pnl_per_contract_by_action": dict(sorted(target_values.items())),
            "evaluation_target_pnl_components_by_action": dict(sorted(components.items())),
            "official_resolution_provenance": {
                "resolution_status": resolution.get("resolution_status"),
                "raw_resolution_sha256": resolution.get("raw_resolution_sha256"),
                "resolution_rule_sha256": resolution.get("resolution_rule_sha256"),
                "source_type": "phase2_official_read_only_resolution",
            },
            "target_available_only_after_market_close": True,
            "outcome_used_for_evaluation_only": True,
            "outcome_used_for_shadow_selection": False,
            "future_results_used_for_tuning": False,
            "future_results_used_for_unlock": False,
            "source_model_candidate_eligible": False,
            "freeze_ready": False,
            "promotion_evidence_eligible": False,
            "v8_execution_handoff_allowed": False,
            "#134_resume_allowed": False,
            "#146_start_allowed": False,
            **compact_safety_fields(),
        }
        target["target_row_sha256"] = canonical_json_sha256(target)
        targets.append(target)
    targets.sort(
        key=lambda row: (
            int(row["decision_ts"]),
            str(row["market_id"]),
            str(row["row_identity"]),
        )
    )
    if {str(row["row_identity"]) for row in targets} != set(decision_ids):
        raise ValueError("settlement targets do not reconcile to shadow identities")

    output_dir.mkdir(parents=True, exist_ok=False)
    target_path = output_dir / "pnl_aligned_future_settled_evaluation_targets.jsonl"
    report_path = output_dir / "pnl_aligned_future_settlement_target_report.json"
    _write_jsonl(target_path, targets)
    report = {
        "schema_version": f"{SETTLEMENT_TARGET_SCHEMA_VERSION}-report",
        "run_id": config.run_id,
        "status": "SETTLED_EVALUATION_TARGETS_READY",
        "shadow_manifest_sha256": config.expected_shadow_manifest_sha256,
        "identity_reconciliation_passed": True,
        "settled_target_count": len(targets),
        "settled_market_count": len({str(row["market_id"]) for row in targets}),
        "complete_5_action_target_grid_count": len(targets),
        "all_targets_loaded_after_market_close": True,
        "source_target_artifacts": source_target_artifacts,
        "reconciliation_start_marker": _descriptor(marker_path),
        "future_outcome_targets_loaded": True,
        "outcome_reconciliation_started": True,
        "outcome_used_for_shadow_selection": False,
        "future_results_used_for_tuning": False,
        "future_results_used_for_unlock": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    _write_json(report_path, report)
    manifest = {
        "schema_version": f"{SETTLEMENT_TARGET_SCHEMA_VERSION}-manifest",
        "run_id": config.run_id,
        "shadow_manifest": _descriptor(shadow_path),
        "decision_input_manifest": decision_input_descriptor,
        "settled_evaluation_targets": _descriptor(target_path),
        "settlement_target_report": _descriptor(report_path),
        "reconciliation_start_marker": _descriptor(marker_path),
        "identity_reconciliation_passed": True,
        "future_outcome_targets_loaded": True,
        "outcome_reconciliation_started": True,
        "future_results_used_for_tuning": False,
        "future_results_used_for_unlock": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    manifest_path = output_dir / "pnl_aligned_future_settlement_target_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "output_dir": output_dir,
        "target_path": target_path,
        "report_path": report_path,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "targets": targets,
        "report": report,
    }


def run_pnl_aligned_future_outcome_blind_shadow_comparison(
    *,
    model_dir: Path | str,
    decision_rows: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Emit candidate and baseline shadows without reading future targets."""

    candidate_rows, candidate_report = run_pnl_aligned_action_value_outcome_blind_shadow(
        model_dir=model_dir,
        decision_rows=decision_rows,
    )
    baseline_rows, baseline_report = _run_outcome_blind_raw_probability_baseline(
        model_dir=model_dir,
        decision_rows=decision_rows,
    )
    candidate_ids = {str(row["source_row_identity"]) for row in candidate_rows}
    baseline_ids = {str(row["source_row_identity"]) for row in baseline_rows}
    identity_match = candidate_ids == baseline_ids and len(candidate_ids) == len(decision_rows)
    status = (
        "OUTCOME_BLIND_COMPARISON_SHADOW_COMPLETE"
        if identity_match
        and candidate_report.get("status") == "OUTCOME_BLIND_SHADOW_EXECUTION_COMPLETE"
        and baseline_report.get("status") == "OUTCOME_BLIND_BASELINE_COMPLETE"
        else "BLOCKED_FAIL_CLOSED"
    )
    report = {
        "schema_version": ("bigan-v8-execution-layer-v2-pnl-aligned-future-shadow-comparison-v1"),
        "status": status,
        "decision_count": len(decision_rows),
        "candidate_shadow_report": candidate_report,
        "baseline_shadow_report": baseline_report,
        "candidate_baseline_identity_match": identity_match,
        "future_outcome_targets_loaded": False,
        "outcome_fields_used_for_selection": False,
        "source_o_score_mutated": False,
        "source_ranking_mutated": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    return {"candidate": candidate_rows, "baseline": baseline_rows}, report


def evaluate_pnl_aligned_future_accepted_bets(
    *,
    evaluation_protocol: dict[str, Any],
    collection_freeze_manifest: dict[str, Any],
    candidate_shadow_rows: list[dict[str, Any]],
    baseline_shadow_rows: list[dict[str, Any]],
    settled_evaluation_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reconcile frozen shadows to post-close targets exactly once."""

    validate_pnl_aligned_future_evaluation_protocol(evaluation_protocol)
    prior_rows_path = Path(str(collection_freeze_manifest["historical_development_rows"]["path"]))
    if (
        _sha256_file(prior_rows_path)
        != collection_freeze_manifest["historical_development_rows"]["sha256"]
    ):
        raise ValueError("historical lineage rows descriptor mismatch")
    prior_market_ids = {str(row["market_id"]) for row in _load_jsonl(prior_rows_path)}
    targets_by_identity = _settled_targets_by_identity(settled_evaluation_rows)
    candidate_ids = {str(row["source_row_identity"]) for row in candidate_shadow_rows}
    baseline_ids = {str(row["source_row_identity"]) for row in baseline_shadow_rows}
    target_ids = set(targets_by_identity)
    identity_match = candidate_ids == baseline_ids == target_ids
    all_shadow_rows = [*candidate_shadow_rows, *baseline_shadow_rows]
    future_boundary_violations = [
        str(row["source_row_identity"])
        for row in all_shadow_rows
        if int(row["decision_ts"])
        < int(collection_freeze_manifest["minimum_future_window_start_ts"])
    ]
    overlapping_markets = sorted(
        {str(row["market_id"]) for row in all_shadow_rows} & prior_market_ids
    )
    shadow_forbidden = sorted(
        {field for row in all_shadow_rows for field in _find_forbidden_fields(row)}
    )
    if not identity_match:
        raise ValueError("candidate, baseline, and target identities do not match")
    if future_boundary_violations:
        raise ValueError("future window contains non-future decision timestamps")
    if overlapping_markets:
        raise ValueError("future markets overlap historical fit markets")
    if shadow_forbidden:
        raise ValueError("outcome-blind shadows contain forbidden fields")

    pnl_rows: list[dict[str, Any]] = []
    policy_metrics: dict[str, dict[str, Any]] = {}
    policy_rows = {
        str(evaluation_protocol["candidate_policy_name"]): candidate_shadow_rows,
        str(evaluation_protocol["baseline_policy_name"]): baseline_shadow_rows,
    }
    for policy_name, shadow_rows in policy_rows.items():
        reconciled = [
            _reconcile_shadow_row(
                policy_name=policy_name,
                shadow_row=row,
                target_row=targets_by_identity[str(row["source_row_identity"])],
            )
            for row in shadow_rows
        ]
        pnl_rows.extend(reconciled)
        policy_metrics[policy_name] = _accepted_bet_metrics(reconciled)

    candidate_name = str(evaluation_protocol["candidate_policy_name"])
    baseline_name = str(evaluation_protocol["baseline_policy_name"])
    candidate_metrics = policy_metrics[candidate_name]
    baseline_metrics = policy_metrics[baseline_name]
    market_delta = _market_pnl_delta(
        candidate_rows=[row for row in pnl_rows if row["policy_name"] == candidate_name],
        baseline_rows=[row for row in pnl_rows if row["policy_name"] == baseline_name],
    )
    bootstrap = _market_bootstrap_interval(
        market_delta,
        protocol=evaluation_protocol,
    )
    gates = dict(evaluation_protocol["future_evidence_gates"])
    checks = {
        "minimum_unique_market_count_met": candidate_metrics["accepted_unique_market_count"]
        >= int(gates["minimum_unique_market_count"]),
        "minimum_accepted_bet_count_met": candidate_metrics["accepted_bet_count"]
        >= int(gates["minimum_accepted_bet_count"]),
        "minimum_up_accepted_bet_count_met": candidate_metrics["accepted_bet_count_by_side"].get(
            "UP", 0
        )
        >= int(gates["minimum_accepted_bet_count_per_side"]),
        "minimum_down_accepted_bet_count_met": candidate_metrics["accepted_bet_count_by_side"].get(
            "DOWN", 0
        )
        >= int(gates["minimum_accepted_bet_count_per_side"]),
        "all_accepted_bets_settled": candidate_metrics["unresolved_accepted_bet_count"] == 0,
        "candidate_net_pnl_positive": candidate_metrics["settled_net_pnl_sum"] > 0.0,
        "candidate_roi_positive": candidate_metrics["roi"] > 0.0,
        "candidate_net_pnl_exceeds_baseline": candidate_metrics["settled_net_pnl_sum"]
        > baseline_metrics["settled_net_pnl_sum"],
        "market_bootstrap_interval_reported": bootstrap["reported"],
        "largest_winner_removal_reported": candidate_metrics["largest_winner_removal"]["reported"],
        "zero_forbidden_shadow_fields": not shadow_forbidden,
        "future_window_strictly_later": not future_boundary_violations,
        "future_markets_disjoint": not overlapping_markets,
    }
    reason_map = {
        "minimum_unique_market_count_met": "insufficient_unique_market_support",
        "minimum_accepted_bet_count_met": "insufficient_accepted_bet_support",
        "minimum_up_accepted_bet_count_met": "insufficient_up_accepted_bet_support",
        "minimum_down_accepted_bet_count_met": "insufficient_down_accepted_bet_support",
        "all_accepted_bets_settled": "unresolved_accepted_bets",
        "candidate_net_pnl_positive": "candidate_net_pnl_not_positive",
        "candidate_roi_positive": "candidate_roi_not_positive",
        "candidate_net_pnl_exceeds_baseline": "candidate_not_better_than_baseline",
        "market_bootstrap_interval_reported": "market_bootstrap_not_reported",
        "largest_winner_removal_reported": "largest_winner_removal_not_reported",
        "zero_forbidden_shadow_fields": "forbidden_shadow_fields_present",
        "future_window_strictly_later": "future_window_not_strictly_later",
        "future_markets_disjoint": "future_markets_overlap_historical_fit",
    }
    blockers = [reason_map[name] for name, passed in checks.items() if not passed]
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "FUTURE_ACCEPTED_BET_EVALUATION_COMPLETE",
        "candidate_policy_name": candidate_name,
        "baseline_policy_name": baseline_name,
        "identity_reconciliation_passed": identity_match,
        "future_window_time_validation_passed": not future_boundary_violations,
        "future_market_disjointness_passed": not overlapping_markets,
        "forbidden_shadow_field_violation_count": len(shadow_forbidden),
        "candidate_policy_metrics": candidate_metrics,
        "baseline_policy_metrics": baseline_metrics,
        "candidate_minus_baseline_net_pnl": candidate_metrics["settled_net_pnl_sum"]
        - baseline_metrics["settled_net_pnl_sum"],
        "market_level_candidate_minus_baseline_pnl": market_delta,
        "market_bootstrap_interval": bootstrap,
        "future_evidence_gate_checks": checks,
        "future_evidence_gate_passed": all(checks.values()),
        "future_evidence_gate_blocking_reason_codes": blockers,
        "future_results_used_for_tuning": False,
        "future_results_used_for_unlock": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    return report, sorted(
        pnl_rows,
        key=lambda row: (
            int(row["decision_ts"]),
            str(row["market_id"]),
            str(row["policy_name"]),
        ),
    )


def _run_outcome_blind_raw_probability_baseline(
    *,
    model_dir: Path | str,
    decision_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model_dir = Path(model_dir).resolve()
    fit_manifest = _load_json(model_dir / "pnl_aligned_action_value_fit_manifest.json")
    protocol = _load_json(Path(fit_manifest["protocol"]["path"]))
    validate_pnl_aligned_action_value_protocol(protocol)
    action_rows, audit = build_pnl_aligned_action_conditioned_rows(
        decision_rows,
        protocol=protocol,
        require_targets=False,
    )
    if audit["blocking_reason_codes"]:
        return [], {
            "status": "BLOCKED_FAIL_CLOSED",
            "feature_leakage_audit": audit,
            "source_model_candidate_eligible": False,
            **compact_safety_fields(),
        }
    threshold = float(protocol["frozen_execution_contract"]["entry_edge_threshold"])
    guard_config = _v8_execution_guard_config()
    state = _v8_initial_runtime_state(guard_config)
    market_close_by_open_position: dict[str, int] = {}
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in action_rows:
        grouped[(str(row["market_id"]), int(row["decision_ts"]))].append(row)
    replay_rows: list[dict[str, Any]] = []
    for index, ((market_id, decision_ts), rows) in enumerate(
        sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])), start=1
    ):
        _release_closed_shadow_positions(
            state=state,
            market_close_by_open_position=market_close_by_open_position,
            decision_ts=decision_ts,
        )
        selected = min(
            rows,
            key=lambda row: (
                float(row["decision_time_features"]["canonical_action_rank"]),
                str(row["action"]),
            ),
        )
        handoff = dict(selected["execution_handoff_context"])
        side = str(selected["side"])
        probability = (
            float(handoff["p_up"])
            if side == "UP"
            else float(handoff["p_down"])
            if side == "DOWN"
            else 0.0
        )
        micro = dict(handoff.get("microstructure_snapshot") or {})
        execution_price = float(micro.get("entry_ask") or 0.0)
        cost = _decision_time_execution_cost(micro)
        edge = probability - execution_price - cost
        selected_action = str(selected["action"])
        signal_passed = selected_action != "NO_TRADE" and edge >= threshold
        blockers: list[str] = []
        guard_row: dict[str, Any] | None = None
        if selected_action == "NO_TRADE":
            blockers.append("raw_baseline_selected_no_trade")
        elif edge < threshold:
            blockers.append("raw_baseline_edge_below_frozen_threshold")
        else:
            guard_row = _v8_execution_guard_decision(
                handoff,
                guard_config=guard_config,
                runtime_state=state,
                runtime_mode="simulated_runtime_state",
            )
            blockers.extend(guard_row["execution_blocking_reason_codes"])
        guard_allowed = bool(guard_row and guard_row["order_allowed"])
        order_id = None
        if guard_allowed:
            order_id = f"raw-probability-baseline-bet-{index:06d}"
            _v8_apply_simulated_order_to_state(
                state=state,
                decision=guard_row,
                simulated_order_id=order_id,
            )
            market_close_by_open_position[market_id] = int(selected["market_close_ts"])
        replay_row = {
            "policy_name": "raw_market_probability_selected_o_action_baseline",
            "source_row_identity": str(selected["source_row_identity"]),
            "market_id": market_id,
            "decision_ts": decision_ts,
            "market_close_ts": int(selected["market_close_ts"]),
            "selected_action": selected_action,
            "selected_side": side,
            "selected_action_family": str(selected["action_family"]),
            "selected_side_market_probability": probability,
            "selected_execution_price": execution_price,
            "decision_time_expected_execution_cost_per_unit": cost,
            "model_entry_edge": edge,
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
            "simulated_order_id": order_id,
            "execution_blocking_reason_codes": sorted(set(blockers)),
            "execution_guard_reason_codes": (
                list(guard_row["execution_guard_reason_codes"]) if guard_row else []
            ),
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
    accepted = [row for row in replay_rows if row["execution_guard_order_allowed"]]
    report = {
        "status": "OUTCOME_BLIND_BASELINE_COMPLETE",
        "decision_count": len(replay_rows),
        "model_trade_candidate_count": sum(row["model_signal_passed"] for row in replay_rows),
        "executable_shadow_bet_count": len(accepted),
        "feature_leakage_audit": audit,
        "market_probability_is_diagnostic_baseline_not_calibrated_fair_value": True,
        "outcome_fields_used": False,
        "source_o_score_mutated": False,
        "source_ranking_mutated": False,
        "source_model_candidate_eligible": False,
        "promotion_evidence_eligible": False,
        **compact_safety_fields(),
    }
    return replay_rows, report


def _decision_time_execution_cost(micro: dict[str, Any]) -> float:
    spread = max(float(micro.get("spread_bps") or 0.0), 0.0) / 20000.0
    queue = min(max(float(micro.get("queue_fill_proxy") or 0.0), 0.0), 1.0)
    staleness = max(float(micro.get("book_staleness_ms") or 0.0), 0.0)
    return min(
        0.05,
        0.001 + spread + (1.0 - queue) * 0.002 + min(staleness / 1000.0, 1.0) * 0.001,
    )


def _settled_targets_by_identity(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = str(row.get("row_identity") or "")
        if not identity or identity in result:
            raise ValueError("settled evaluation row identity missing or duplicated")
        targets = dict(row.get("evaluation_target_net_pnl_per_contract_by_action") or {})
        components = dict(row.get("evaluation_target_pnl_components_by_action") or {})
        if set(targets) != set(REQUIRED_ACTIONS) or set(components) != set(REQUIRED_ACTIONS):
            raise ValueError("settled evaluation action target grid is incomplete")
        result[identity] = row
    return result


def _reconcile_shadow_row(
    *,
    policy_name: str,
    shadow_row: dict[str, Any],
    target_row: dict[str, Any],
) -> dict[str, Any]:
    if str(shadow_row["market_id"]) != str(target_row["market_id"]) or int(
        shadow_row["decision_ts"]
    ) != int(target_row["decision_ts"]):
        raise ValueError("shadow and target row provenance mismatch")
    accepted = shadow_row.get("execution_guard_order_allowed") is True
    action = str(shadow_row.get("execution_guarded_action") or "")
    targets = dict(target_row["evaluation_target_net_pnl_per_contract_by_action"])
    components = dict(target_row["evaluation_target_pnl_components_by_action"])
    target = targets.get(action) if accepted else None
    component = dict(components.get(action) or {}) if accepted else {}
    required_components = (
        "gross_pnl_per_contract",
        "execution_cost_per_contract",
        "net_pnl_per_contract",
    )
    settled = (
        accepted
        and _finite(target)
        and all(_finite(component.get(name)) for name in required_components)
    )
    size = float(shadow_row.get("proposed_order_size") or 0.0) if accepted else 0.0
    entry_price = float(shadow_row.get("selected_execution_price") or 0.0)
    gross_pnl = size * float(component["gross_pnl_per_contract"]) if settled else None
    execution_cost = size * float(component["execution_cost_per_contract"]) if settled else None
    net_pnl = size * float(target) if settled else None
    cost_basis = (
        size * (entry_price + float(component["execution_cost_per_contract"])) if settled else 0.0
    )
    return {
        "policy_name": policy_name,
        "source_row_identity": str(shadow_row["source_row_identity"]),
        "market_id": str(shadow_row["market_id"]),
        "decision_ts": int(shadow_row["decision_ts"]),
        "market_close_ts": int(shadow_row["market_close_ts"]),
        "selected_action": str(shadow_row["selected_action"]),
        "execution_guarded_action": action or None,
        "execution_guarded_side": shadow_row.get("execution_guarded_side"),
        "execution_guard_order_allowed": accepted,
        "simulated_order_id": shadow_row.get("simulated_order_id"),
        "paper_bet_contract_size": size,
        "execution_price": entry_price,
        "settlement_target_available": settled,
        "guarded_action_target_net_pnl_per_contract": float(target) if settled else None,
        "gross_pnl": gross_pnl,
        "execution_cost": execution_cost,
        "cost_basis": cost_basis,
        "settled_net_pnl": net_pnl,
        "selection_uses_outcome_fields": False,
        "outcome_aware_evaluation_only": True,
        "source_o_score_mutated": False,
        "source_ranking_mutated": False,
        "paper_only": True,
        "capital_at_risk": False,
    }


def _accepted_bet_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row["execution_guard_order_allowed"]]
    settled = [row for row in accepted if row["settlement_target_available"]]
    pnl_values = [float(row["settled_net_pnl"]) for row in settled]
    cost_basis = sum(float(row["cost_basis"]) for row in settled)
    net_pnl = sum(pnl_values)
    market_pnl: dict[str, float] = defaultdict(float)
    for row in settled:
        market_pnl[str(row["market_id"])] += float(row["settled_net_pnl"])
    ordered = sorted(
        settled,
        key=lambda row: (
            int(row["market_close_ts"]),
            str(row["market_id"]),
            str(row["simulated_order_id"]),
        ),
    )
    equity = peak = max_drawdown = 0.0
    for row in ordered:
        equity += float(row["settled_net_pnl"])
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    largest_market = max(market_pnl, key=market_pnl.get) if market_pnl else None
    largest_value = market_pnl.get(largest_market, 0.0) if largest_market else 0.0
    side_counts = Counter(str(row["execution_guarded_side"]) for row in accepted)
    action_counts = Counter(str(row["execution_guarded_action"]) for row in accepted)
    family_counts = Counter(_family(str(row["execution_guarded_action"])) for row in accepted)
    return {
        "accepted_bet_count": len(accepted),
        "settled_accepted_bet_count": len(settled),
        "unresolved_accepted_bet_count": len(accepted) - len(settled),
        "accepted_unique_market_count": len({row["market_id"] for row in accepted}),
        "accepted_bet_count_by_side": dict(sorted(side_counts.items())),
        "accepted_bet_count_by_action": dict(sorted(action_counts.items())),
        "accepted_bet_count_by_family": dict(sorted(family_counts.items())),
        "contract_size_sum": sum(float(row["paper_bet_contract_size"]) for row in accepted),
        "gross_pnl_sum": sum(float(row["gross_pnl"]) for row in settled),
        "execution_cost_sum": sum(float(row["execution_cost"]) for row in settled),
        "cost_basis_sum": cost_basis,
        "settled_net_pnl_sum": net_pnl,
        "roi": net_pnl / cost_basis if cost_basis > 0.0 else 0.0,
        "win_rate": sum(value > 0.0 for value in pnl_values) / len(pnl_values)
        if pnl_values
        else 0.0,
        "chronological_max_drawdown": max_drawdown,
        "pnl_by_market": dict(sorted(market_pnl.items())),
        "largest_winner_removal": {
            "reported": True,
            "largest_winning_market_id": largest_market,
            "largest_winning_market_pnl": largest_value,
            "net_pnl_after_largest_winner_removed": net_pnl - max(largest_value, 0.0),
        },
    }


def _market_pnl_delta(
    *,
    candidate_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
        values: dict[str, float] = defaultdict(float)
        for row in rows:
            if row["settlement_target_available"]:
                values[str(row["market_id"])] += float(row["settled_net_pnl"])
        return values

    candidate = aggregate(candidate_rows)
    baseline = aggregate(baseline_rows)
    return [
        {
            "market_id": market_id,
            "candidate_net_pnl": candidate.get(market_id, 0.0),
            "baseline_net_pnl": baseline.get(market_id, 0.0),
            "candidate_minus_baseline_net_pnl": candidate.get(market_id, 0.0)
            - baseline.get(market_id, 0.0),
        }
        for market_id in sorted(set(candidate) | set(baseline))
    ]


def _market_bootstrap_interval(
    rows: list[dict[str, Any]],
    *,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    config = dict(protocol["market_bootstrap"])
    values = [float(row["candidate_minus_baseline_net_pnl"]) for row in rows]
    if not values:
        return {
            "reported": True,
            "market_count": 0,
            "point_estimate": 0.0,
            "lower_bound": 0.0,
            "upper_bound": 0.0,
            **config,
        }
    rng = random.Random(int(config["seed"]))
    sample_count = int(config["resample_count"])
    estimates = [sum(rng.choice(values) for _ in values) for _ in range(sample_count)]
    alpha = (1.0 - float(config["confidence_level"])) / 2.0
    return {
        "reported": True,
        "market_count": len(values),
        "point_estimate": sum(values),
        "lower_bound": float(np.quantile(estimates, alpha)),
        "upper_bound": float(np.quantile(estimates, 1.0 - alpha)),
        **config,
    }


def _family(action: str) -> str:
    if action.endswith("HOLD_TO_SETTLEMENT"):
        return "HOLD_TO_SETTLEMENT"
    if action.endswith("SELL_BEFORE_CLOSE"):
        return "SELL_BEFORE_CLOSE"
    return "NO_TRADE"


def _find_forbidden_fields(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).lower()
            if normalized in FORBIDDEN_DECISION_FIELDS or any(
                token in normalized
                for token in (
                    "resolved_outcome",
                    "settlement_pnl",
                    "future_return",
                    "oracle_action",
                    "target_net_pnl",
                )
            ):
                found.add(str(key))
            found.update(_find_forbidden_fields(nested))
    elif isinstance(value, list):
        for nested in value:
            found.update(_find_forbidden_fields(nested))
    return found


def _verified_descriptor(value: Any, *, name: str) -> dict[str, str]:
    descriptor = dict(value or {})
    path = Path(str(descriptor.get("path") or ""))
    if not path.is_file() or descriptor.get("sha256") != _sha256_file(path):
        raise ValueError(f"{name} descriptor hash mismatch")
    return {"path": str(path), "sha256": str(descriptor["sha256"])}


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def _finite(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))
