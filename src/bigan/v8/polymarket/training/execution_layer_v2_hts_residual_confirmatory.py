"""Freeze and evaluate HTS residual confirmatory evidence exactly once."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.contracts import safety_fields
from bigan.v8.polymarket.training.execution_layer_v2_hts_residual_development_corpus import (
    HTS_ACTIONS,
    PHASE2_MARKET_PROBABILITY_MAPPING_RULE_ID,
    _load_verified_phase2_corpus,
    _phase2_feature_to_public_row,
    _residual_row,
)
from bigan.v8.polymarket.training.execution_layer_v2_hts_residual_edge import (
    _bootstrap_market_mean,
    _market_error_deltas,
    _probability_metrics,
    _quantile,
    _relative_improvement,
    _selected_probability,
    predict_residual_offset_probability,
)
from bigan.v8.polymarket.training.o_v8_paper_fresh_loop import (
    PINNED_ISSUE_160_MANIFEST_SHA256,
    score_frozen_o_decision_rows,
)

SCHEMA_PREFIX = "bigan-v8-hts-residual-confirmatory"


def _blocked_safety() -> dict[str, Any]:
    return {
        "diagnostic_only": True,
        **safety_fields(),
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }


@dataclass(frozen=True, slots=True)
class HTSResidualCandidateFreezeConfig:
    """Freeze a development-gated residual candidate before confirmatory data."""

    run_id: str
    output_dir: Path | str
    development_oof_manifest_path: Path | str
    freeze_created_at: str | None = None
    freeze_created_ts: int | None = None
    minimum_confirmatory_market_count: int = 283
    minimum_confirmatory_source_run_count: int = 24
    minimum_input_source_market_count: int = 283
    minimum_input_hts_market_count: int = 283
    both_input_selected_sides_required: bool = True
    minimum_relative_brier_improvement: float = 0.03
    minimum_relative_log_loss_improvement: float = 0.03
    bootstrap_samples: int = 10_000
    bootstrap_confidence_level: float = 0.95
    bootstrap_seed: int = 20260714

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if self.minimum_confirmatory_market_count <= 0:
            raise ValueError("minimum_confirmatory_market_count must be positive")
        if self.minimum_confirmatory_source_run_count <= 0:
            raise ValueError("minimum_confirmatory_source_run_count must be positive")
        if self.minimum_input_source_market_count <= 0:
            raise ValueError("minimum_input_source_market_count must be positive")
        if self.minimum_input_hts_market_count <= 0:
            raise ValueError("minimum_input_hts_market_count must be positive")
        if self.bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples must be positive")
        if not 0.0 < self.bootstrap_confidence_level < 1.0:
            raise ValueError("bootstrap_confidence_level must be in (0, 1)")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(
            self,
            "development_oof_manifest_path",
            Path(self.development_oof_manifest_path),
        )


@dataclass(frozen=True, slots=True)
class HTSResidualConfirmatoryInputConfig:
    """Freeze future source artifacts without reading outcome target values."""

    run_id: str
    output_dir: Path | str
    candidate_freeze_manifest_path: Path | str
    source_corpus_dirs: tuple[Path | str, ...]
    paper_candidate_unlock_dir: Path | str
    expected_unlock_manifest_sha256: str = PINNED_ISSUE_160_MANIFEST_SHA256
    canonical_o_source_manifest_path: Path | str | None = None

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if not self.source_corpus_dirs:
            raise ValueError("source_corpus_dirs must not be empty")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(
            self,
            "paper_candidate_unlock_dir",
            Path(self.paper_candidate_unlock_dir),
        )
        object.__setattr__(
            self,
            "candidate_freeze_manifest_path",
            Path(self.candidate_freeze_manifest_path),
        )
        object.__setattr__(
            self,
            "source_corpus_dirs",
            tuple(Path(path) for path in self.source_corpus_dirs),
        )
        if self.canonical_o_source_manifest_path is not None:
            object.__setattr__(
                self,
                "canonical_o_source_manifest_path",
                Path(self.canonical_o_source_manifest_path),
            )


@dataclass(frozen=True, slots=True)
class HTSResidualConfirmatoryEvaluationConfig:
    """Inputs for one irreversible confirmatory evaluation attempt."""

    confirmatory_input_manifest_path: Path | str
    paper_candidate_unlock_dir: Path | str
    expected_unlock_manifest_sha256: str = PINNED_ISSUE_160_MANIFEST_SHA256
    canonical_o_source_manifest_path: Path | str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "confirmatory_input_manifest_path",
            Path(self.confirmatory_input_manifest_path),
        )
        object.__setattr__(
            self,
            "paper_candidate_unlock_dir",
            Path(self.paper_candidate_unlock_dir),
        )
        if self.canonical_o_source_manifest_path is not None:
            object.__setattr__(
                self,
                "canonical_o_source_manifest_path",
                Path(self.canonical_o_source_manifest_path),
            )


def freeze_hts_residual_candidate(
    config: HTSResidualCandidateFreezeConfig,
) -> dict[str, Any]:
    """Freeze only a candidate that passed the frozen development gate."""

    output_dir = Path(config.output_dir) / config.run_id
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    development_manifest_path = Path(config.development_oof_manifest_path).resolve()
    development_manifest = _load_json(development_manifest_path)
    frozen_o_source_manifest_sha256 = str(
        development_manifest.get("frozen_o_source_manifest_sha256") or ""
    )
    frozen_o_ranking_report_sha256 = str(
        development_manifest.get("frozen_o_ranking_report_sha256") or ""
    )
    if not _is_sha256(frozen_o_source_manifest_sha256) or not _is_sha256(
        frozen_o_ranking_report_sha256
    ):
        raise ValueError("development manifest frozen O scorer lineage is missing")
    report_descriptor = _verified_descriptor(development_manifest["report"])
    rows_descriptor = _verified_descriptor(development_manifest["combined_rows"])
    protocol_descriptor = _verified_descriptor(development_manifest["protocol"])
    development_report = _load_json(Path(report_descriptor["path"]))
    if development_report.get("development_candidate_gate_passed") is not True:
        raise ValueError("development candidate gate did not pass")
    if development_manifest.get("development_candidate_gate_passed") is not True:
        raise ValueError("development manifest candidate gate did not pass")
    contract = dict(development_report.get("selected_candidate_contract") or {})
    if not contract or contract.get("finite_and_bounded") is not True:
        raise ValueError("selected residual contract is missing or invalid")
    selected_name = str(development_report.get("selected_candidate_name") or "")
    if contract.get("candidate_name") != selected_name:
        raise ValueError("selected residual candidate identity mismatch")

    freeze_created_at, freeze_created_ts = _freeze_time(config)
    prior_rows = _load_jsonl(Path(rows_descriptor["path"]))
    prior_market_ids = sorted({str(row["market_id"]) for row in prior_rows})
    prior_max_decision_ts = max(int(row["decision_ts"]) for row in prior_rows)
    if freeze_created_ts <= prior_max_decision_ts:
        raise ValueError("candidate freeze timestamp is not after development data")
    output_dir.mkdir(parents=True)
    frozen_contract = {
        **contract,
        "schema_version": f"{SCHEMA_PREFIX}-frozen-candidate-v1",
        "frozen": True,
        "frozen_at": freeze_created_at,
        "frozen_at_ts": freeze_created_ts,
        "selected_from_development_only": True,
        "development_oof_manifest_sha256": _sha256_file(
            development_manifest_path
        ),
        "development_oof_report_sha256": report_descriptor["sha256"],
        "development_rows_sha256": rows_descriptor["sha256"],
        "development_protocol_sha256": protocol_descriptor["sha256"],
        "frozen_o_source_manifest_sha256": frozen_o_source_manifest_sha256,
        "frozen_o_ranking_report_sha256": frozen_o_ranking_report_sha256,
        "confirmatory_outcomes_used_for_fitting": False,
        **_blocked_safety(),
    }
    frozen_contract["frozen_candidate_id"] = canonical_json_sha256(frozen_contract)
    contract_path = output_dir / "hts_residual_frozen_candidate.json"
    _write_json(contract_path, frozen_contract)

    confirmatory_protocol = {
        "schema_version": f"{SCHEMA_PREFIX}-protocol-v1",
        "frozen": True,
        "frozen_before_confirmatory_collection": True,
        "candidate_name": selected_name,
        "frozen_candidate_sha256": _sha256_file(contract_path),
        "collection_not_before_ts": freeze_created_ts,
        "strictly_later_than_development_required": True,
        "prior_max_decision_ts": prior_max_decision_ts,
        "prior_market_ids_sha256": canonical_json_sha256(prior_market_ids),
        "minimum_input_source_market_count": (
            config.minimum_input_source_market_count
        ),
        "minimum_input_hts_market_count": config.minimum_input_hts_market_count,
        "both_input_selected_sides_required": (
            config.both_input_selected_sides_required
        ),
        "minimum_confirmatory_market_count": (
            config.minimum_confirmatory_market_count
        ),
        "minimum_confirmatory_source_run_count": (
            config.minimum_confirmatory_source_run_count
        ),
        "both_selected_sides_required": True,
        "both_resolved_outcomes_required": True,
        "minimum_relative_brier_improvement": (
            config.minimum_relative_brier_improvement
        ),
        "minimum_relative_log_loss_improvement": (
            config.minimum_relative_log_loss_improvement
        ),
        "positive_market_mean_brier_improvement_required": True,
        "bootstrap_market_mean_improvement_lower_bound_must_exceed_zero": True,
        "bootstrap_samples": config.bootstrap_samples,
        "bootstrap_confidence_level": config.bootstrap_confidence_level,
        "bootstrap_seed": config.bootstrap_seed,
        "exactly_once_evaluation_required": True,
        "market_disjointness_required": True,
        "validation_labels_used_for_tuning": False,
        "no_candidate_or_threshold_selection_after_input_freeze": True,
        **_blocked_safety(),
    }
    confirmatory_protocol["confirmatory_protocol_id"] = canonical_json_sha256(
        confirmatory_protocol
    )
    confirmatory_protocol_path = output_dir / "hts_residual_confirmatory_protocol.json"
    _write_json(confirmatory_protocol_path, confirmatory_protocol)

    report = {
        "schema_version": f"{SCHEMA_PREFIX}-candidate-freeze-report-v1",
        "run_id": config.run_id,
        "candidate_name": selected_name,
        "candidate_frozen": True,
        "development_candidate_gate_verified": True,
        "development_manifest": _descriptor(development_manifest_path),
        "development_report": report_descriptor,
        "development_rows": rows_descriptor,
        "development_protocol": protocol_descriptor,
        "frozen_o_source_manifest_sha256": frozen_o_source_manifest_sha256,
        "frozen_o_ranking_report_sha256": frozen_o_ranking_report_sha256,
        "frozen_candidate": _descriptor(contract_path),
        "confirmatory_protocol": _descriptor(confirmatory_protocol_path),
        "confirmatory_collection_started": False,
        "confirmatory_evaluation_started": False,
        "pre_promotion_ready": False,
        **_blocked_safety(),
    }
    report_path = output_dir / "hts_residual_candidate_freeze_report.json"
    _write_json(report_path, report)
    _write_text(
        output_dir / "hts_residual_candidate_freeze_report.md",
        _freeze_markdown(report),
    )
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-candidate-freeze-manifest-v1",
        "run_id": config.run_id,
        "report": _descriptor(report_path),
        "frozen_candidate": _descriptor(contract_path),
        "confirmatory_protocol": _descriptor(confirmatory_protocol_path),
        "development_manifest": _descriptor(development_manifest_path),
        "development_rows": rows_descriptor,
        "frozen_o_source_manifest_sha256": frozen_o_source_manifest_sha256,
        "frozen_o_ranking_report_sha256": frozen_o_ranking_report_sha256,
        "candidate_frozen": True,
        "confirmatory_evaluation_started": False,
        "pre_promotion_ready": False,
        **_blocked_safety(),
    }
    manifest_path = output_dir / "hts_residual_candidate_freeze_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "output_dir": output_dir,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "report": report,
    }


def freeze_hts_residual_confirmatory_input(
    config: HTSResidualConfirmatoryInputConfig,
) -> dict[str, Any]:
    """Freeze future inputs using metadata/features only, without target values."""

    output_dir = Path(config.output_dir) / config.run_id
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    freeze_manifest_path = Path(config.candidate_freeze_manifest_path).resolve()
    freeze_manifest = _load_json(freeze_manifest_path)
    if freeze_manifest.get("candidate_frozen") is not True:
        raise ValueError("candidate freeze manifest is not frozen")
    candidate_descriptor = _verified_descriptor(freeze_manifest["frozen_candidate"])
    protocol_descriptor = _verified_descriptor(
        freeze_manifest["confirmatory_protocol"]
    )
    development_rows_descriptor = _verified_descriptor(
        freeze_manifest["development_rows"]
    )
    protocol = _load_json(Path(protocol_descriptor["path"]))
    prior_rows = _load_jsonl(Path(development_rows_descriptor["path"]))
    prior_market_ids = {str(row["market_id"]) for row in prior_rows}
    prior_max_decision_ts = max(int(row["decision_ts"]) for row in prior_rows)

    resolved_source_dirs = [
        Path(path).resolve() for path in config.source_corpus_dirs
    ]
    duplicate_source_corpus_dirs = sorted(
        path
        for path, count in Counter(str(path) for path in resolved_source_dirs).items()
        if count > 1
    )
    source_descriptors: list[dict[str, Any]] = []
    source_market_ids: set[str] = set()
    source_market_occurrences: Counter[str] = Counter()
    source_decision_occurrences: Counter[str] = Counter()
    decision_timestamps: list[int] = []
    causality_violations: list[str] = []
    overlap_market_ids: set[str] = set()
    for corpus_dir in sorted(set(resolved_source_dirs)):
        audit = _freeze_source_corpus_without_outcomes(corpus_dir)
        source_descriptors.append(audit)
        source_market_occurrences.update(audit["source_market_ids"])
        for row in _load_jsonl(corpus_dir / "polymarket_feature_rows.jsonl"):
            market_id = str(row["market_id"])
            decision_ts = int(row["decision_ts"])
            source_decision_occurrences[f"{market_id}|{decision_ts}"] += 1
            source_market_ids.add(market_id)
            decision_timestamps.append(decision_ts)
            if market_id in prior_market_ids:
                overlap_market_ids.add(market_id)
            if int(row.get("max_input_ts") or 0) > decision_ts:
                causality_violations.append(
                    f"{market_id}|{decision_ts}|feature_max_input_ts"
                )
    strictly_later = bool(decision_timestamps) and min(decision_timestamps) > max(
        prior_max_decision_ts,
        int(protocol["collection_not_before_ts"]),
    )
    duplicate_source_market_ids = sorted(
        market_id
        for market_id, count in source_market_occurrences.items()
        if count > 1
    )
    duplicate_source_decision_ids = sorted(
        identity
        for identity, count in source_decision_occurrences.items()
        if count > 1
    )
    checks: dict[str, bool] = {
        "minimum_input_source_market_count_met": len(source_market_ids)
        >= int(protocol["minimum_input_source_market_count"]),
        "strict_chronology_passed": strictly_later,
        "market_disjointness_passed": not overlap_market_ids,
        "feature_causality_passed": not causality_violations,
        "source_corpus_uniqueness_passed": not duplicate_source_corpus_dirs,
        "source_market_uniqueness_passed": not duplicate_source_market_ids,
        "source_decision_uniqueness_passed": not duplicate_source_decision_ids,
        "source_hashes_verified": True,
    }
    outcome_blind_public_rows: list[dict[str, Any]] = []
    outcome_blind_construction_rejections: list[dict[str, Any]] = []
    prediction_attempted = False
    scoring_passed = False
    scorer_context_verified = False
    hts_selection_rows: list[dict[str, Any]] = []
    hts_selected_market_ids: set[str] = set()
    hts_selected_source_run_ids: set[str] = set()
    hts_selected_side_counts: Counter[str] = Counter()
    hts_selection_identity_sha256: str | None = None
    if all(checks.values()):
        (
            outcome_blind_public_rows,
            outcome_blind_construction_rejections,
        ) = _build_outcome_blind_confirmatory_public_rows(
            source_descriptors=source_descriptors,
        )
        checks["outcome_blind_feature_construction_passed"] = not (
            outcome_blind_construction_rejections
        )
    else:
        checks["outcome_blind_feature_construction_passed"] = False

    if all(checks.values()):
        prediction_attempted = True
        scoring = score_frozen_o_decision_rows(
            run_id=f"{config.run_id}-outcome-blind-input-freeze",
            decision_rows=outcome_blind_public_rows,
            paper_candidate_unlock_dir=config.paper_candidate_unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=(
                config.expected_unlock_manifest_sha256
            ),
            canonical_o_source_manifest_path=(
                config.canonical_o_source_manifest_path
            ),
        )
        scoring_passed = scoring["scoring_passed"] is True
        context = dict(scoring.get("canonical_context") or {})
        scorer_context_verified = bool(
            scoring_passed
            and context.get("source_manifest_sha256")
            == freeze_manifest.get("frozen_o_source_manifest_sha256")
            and context.get("ranking_objective_report_sha256")
            == freeze_manifest.get("frozen_o_ranking_report_sha256")
        )
        if scoring_passed:
            hts_selection_rows = _outcome_blind_hts_selection_rows(
                scoring, outcome_blind_public_rows
            )
            hts_selected_market_ids = {
                str(row["market_id"]) for row in hts_selection_rows
            }
            hts_selected_source_run_ids = {
                str(row["source_run_id"]) for row in hts_selection_rows
            }
            hts_selected_side_counts.update(
                str(row["selected_side"]) for row in hts_selection_rows
            )
            hts_selection_identity_sha256 = canonical_json_sha256(
                hts_selection_rows
            )
    checks.update(
        {
            "outcome_blind_prediction_passed": scoring_passed,
            "frozen_o_scorer_context_verified": scorer_context_verified,
            "minimum_input_hts_market_count_met": len(hts_selected_market_ids)
            >= int(protocol["minimum_input_hts_market_count"]),
            "minimum_input_hts_source_run_count_met": len(
                hts_selected_source_run_ids
            )
            >= int(protocol["minimum_confirmatory_source_run_count"]),
            "both_input_selected_sides_present": (
                set(hts_selected_side_counts) == {"UP", "DOWN"}
                if protocol.get("both_input_selected_sides_required") is True
                else True
            ),
        }
    )
    reason_by_check = {
        "minimum_input_source_market_count_met": (
            "insufficient_confirmatory_input_market_support"
        ),
        "strict_chronology_passed": "confirmatory_input_not_strictly_later",
        "market_disjointness_passed": "confirmatory_market_overlap_detected",
        "feature_causality_passed": "confirmatory_feature_causality_violation",
        "source_corpus_uniqueness_passed": "duplicate_confirmatory_source_corpus",
        "source_market_uniqueness_passed": "duplicate_confirmatory_source_market",
        "source_decision_uniqueness_passed": (
            "duplicate_confirmatory_source_decision"
        ),
        "source_hashes_verified": "confirmatory_source_hash_verification_failed",
        "outcome_blind_feature_construction_passed": (
            "confirmatory_outcome_blind_feature_construction_failed"
        ),
        "outcome_blind_prediction_passed": (
            "confirmatory_outcome_blind_frozen_o_scoring_failed"
        ),
        "frozen_o_scorer_context_verified": (
            "confirmatory_frozen_o_scorer_context_mismatch"
        ),
        "minimum_input_hts_market_count_met": (
            "insufficient_outcome_blind_hts_market_support"
        ),
        "minimum_input_hts_source_run_count_met": (
            "insufficient_outcome_blind_hts_source_run_support"
        ),
        "both_input_selected_sides_present": (
            "missing_outcome_blind_selected_side_support"
        ),
    }
    input_gate_passed = all(checks.values())
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-input-manifest-v1",
        "run_id": config.run_id,
        "candidate_freeze_manifest": _descriptor(freeze_manifest_path),
        "frozen_candidate": candidate_descriptor,
        "confirmatory_protocol": protocol_descriptor,
        "development_rows": development_rows_descriptor,
        "source_corpora": source_descriptors,
        "source_corpus_count": len(source_descriptors),
        "source_market_count": len(source_market_ids),
        "source_market_ids_sha256": canonical_json_sha256(
            sorted(source_market_ids)
        ),
        "minimum_decision_ts": min(decision_timestamps, default=None),
        "maximum_decision_ts": max(decision_timestamps, default=None),
        "prior_max_decision_ts": prior_max_decision_ts,
        "overlap_market_ids": sorted(overlap_market_ids),
        "duplicate_source_corpus_dirs": duplicate_source_corpus_dirs,
        "duplicate_source_market_ids": duplicate_source_market_ids,
        "duplicate_source_decision_ids": duplicate_source_decision_ids,
        "feature_causality_violation_ids": sorted(causality_violations),
        "outcome_blind_feature_row_count": len(outcome_blind_public_rows),
        "outcome_blind_feature_construction_rejections": (
            outcome_blind_construction_rejections
        ),
        "outcome_blind_prediction_attempted": prediction_attempted,
        "outcome_blind_prediction_passed": scoring_passed,
        "frozen_o_scorer_context_verified": scorer_context_verified,
        "outcome_blind_hts_selected_row_count": len(hts_selection_rows),
        "outcome_blind_hts_selected_market_count": len(
            hts_selected_market_ids
        ),
        "outcome_blind_hts_selected_source_run_count": len(
            hts_selected_source_run_ids
        ),
        "outcome_blind_hts_selected_side_counts": dict(
            sorted(hts_selected_side_counts.items())
        ),
        "outcome_blind_hts_selection_rows": hts_selection_rows,
        "outcome_blind_hts_selection_identity_sha256": (
            hts_selection_identity_sha256
        ),
        "input_gate_checks": checks,
        "input_gate_passed": input_gate_passed,
        "blocking_reason_codes": [
            reason_by_check[name] for name, passed in checks.items() if not passed
        ],
        "outcome_target_files_hashed_but_not_parsed": True,
        "outcome_values_inspected_during_input_freeze": False,
        "confirmatory_evaluation_start_allowed": input_gate_passed,
        "confirmatory_evaluation_started": False,
        "pre_promotion_ready": False,
        **_blocked_safety(),
    }
    manifest["confirmatory_input_id"] = canonical_json_sha256(manifest)
    manifest_path = output_dir / "hts_residual_confirmatory_input_manifest.json"
    _write_json(manifest_path, manifest)
    _write_sha_descriptor(manifest_path)
    _write_text(
        output_dir / "hts_residual_confirmatory_input_manifest.md",
        _input_markdown(manifest),
    )
    return {
        "output_dir": output_dir,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "manifest": manifest,
    }


def evaluate_hts_residual_confirmatory_once(
    config: HTSResidualConfirmatoryEvaluationConfig,
) -> dict[str, Any]:
    """Consume one frozen future input exactly once and evaluate no alternatives."""

    input_manifest_path = Path(config.confirmatory_input_manifest_path).resolve()
    _verify_sha_descriptor(input_manifest_path)
    input_manifest = _load_json(input_manifest_path)
    if input_manifest.get("input_gate_passed") is not True:
        raise ValueError("confirmatory input gate did not pass")
    _verified_descriptor(input_manifest["candidate_freeze_manifest"])
    _verify_confirmatory_source_descriptors(input_manifest)
    output_dir = input_manifest_path.parent
    marker_path = output_dir / "hts_residual_confirmatory_evaluation_started.json"
    marker = {
        "schema_version": f"{SCHEMA_PREFIX}-exactly-once-marker-v1",
        "evaluation_attempt_number": 1,
        "started_at": datetime.now(UTC).isoformat(),
        "confirmatory_input_manifest_sha256": _sha256_file(input_manifest_path),
        "frozen_candidate_sha256": input_manifest["frozen_candidate"]["sha256"],
        "outcome_targets_read_before_marker": False,
        "exactly_once": True,
        **_blocked_safety(),
    }
    marker["marker_id"] = canonical_json_sha256(marker)
    _write_json_exclusive(marker_path, marker)

    freeze_manifest_path = Path(
        input_manifest["candidate_freeze_manifest"]["path"]
    )
    freeze_manifest = _load_json(freeze_manifest_path)
    contract_descriptor = _verified_descriptor(freeze_manifest["frozen_candidate"])
    protocol_descriptor = _verified_descriptor(
        freeze_manifest["confirmatory_protocol"]
    )
    development_rows_descriptor = _verified_descriptor(
        freeze_manifest["development_rows"]
    )
    contract = _load_json(Path(contract_descriptor["path"]))
    protocol = _load_json(Path(protocol_descriptor["path"]))
    prior_rows = _load_jsonl(Path(development_rows_descriptor["path"]))
    prior_market_ids = {str(row["market_id"]) for row in prior_rows}
    prior_max_decision_ts = max(int(row["decision_ts"]) for row in prior_rows)
    loader_protocol = {
        "collection_not_before_ts": int(protocol["collection_not_before_ts"]),
        "excluded_smoke_corpus_ids": [],
    }

    public_rows: list[dict[str, Any]] = []
    targets: dict[tuple[str, int], dict[str, Any]] = {}
    audits: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for source in input_manifest["source_corpora"]:
        corpus_dir = Path(source["corpus_dir"])
        audit, source_public, source_targets, source_rejected = (
            _load_verified_phase2_corpus(
                corpus_dir=corpus_dir,
                protocol=loader_protocol,
                prior_market_ids=prior_market_ids,
                prior_max_decision_ts=prior_max_decision_ts,
            )
        )
        audits.append(audit)
        public_rows.extend(source_public)
        targets.update(source_targets)
        rejected.extend(source_rejected)
    for row in public_rows:
        key = (str(row["market_id"]), int(row["decision_ts"]))
        if key in targets:
            row["source_run_id"] = str(targets[key]["source_run_id"])
    scoring = score_frozen_o_decision_rows(
        run_id=f"{input_manifest['run_id']}-confirmatory-frozen-o",
        decision_rows=public_rows,
        paper_candidate_unlock_dir=config.paper_candidate_unlock_dir,
        expected_paper_candidate_unlock_manifest_sha256=(
            config.expected_unlock_manifest_sha256
        ),
        canonical_o_source_manifest_path=config.canonical_o_source_manifest_path,
    )
    if scoring["scoring_passed"] is not True:
        raise ValueError("frozen O scorer failed during confirmatory evaluation")
    confirmatory_selection_rows = _outcome_blind_hts_selection_rows(
        scoring, public_rows
    )
    confirmatory_selection_hash = canonical_json_sha256(
        confirmatory_selection_rows
    )
    if confirmatory_selection_hash != input_manifest.get(
        "outcome_blind_hts_selection_identity_sha256"
    ):
        raise ValueError("confirmatory frozen O selection identity mismatch")

    scored_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scored in scoring["canonical_scorer_report"]["canonical_scored_action_rows"]:
        scored_by_group[str(scored["decision_group_id"])].append(scored)
    public_by_key = {
        (str(row["market_id"]), int(row["decision_ts"])): row
        for row in public_rows
    }
    confirmatory_rows: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    for selected in scoring["canonical_scorer_report"][
        "canonical_selected_decision_rows"
    ]:
        action = str(selected["action"])
        selected_counts[action] += 1
        if action not in HTS_ACTIONS:
            continue
        key = (str(selected["market_id"]), int(selected["decision_ts"]))
        target = targets.get(key)
        if target is None:
            continue
        row = _residual_row(
            selected=selected,
            group_scored_rows=scored_by_group[str(selected["decision_group_id"])],
            public_row=public_by_key[key],
            target=target,
            protocol_hash=protocol_descriptor["sha256"],
        )
        row["lineage"] = "fresh_confirmatory_exactly_once"
        row["development_evidence_only"] = False
        row["future_confirmatory_validation_eligible"] = True
        row["confirmatory_input_manifest_sha256"] = _sha256_file(
            input_manifest_path
        )
        row["source_lineage"].pop(
            "frozen_development_protocol_sha256", None
        )
        row["source_lineage"]["frozen_confirmatory_protocol_sha256"] = (
            protocol_descriptor["sha256"]
        )
        row["row_content_sha256"] = canonical_json_sha256(
            {key: value for key, value in row.items() if key != "row_content_sha256"}
        )
        confirmatory_rows.append(row)
    confirmatory_rows.sort(
        key=lambda row: (int(row["decision_ts"]), str(row["market_id"]))
    )
    support = _confirmatory_support(confirmatory_rows, protocol)
    predictions = [
        predict_residual_offset_probability(row, contract)
        for row in confirmatory_rows
    ]
    raw_predictions = [_selected_probability(row) for row in confirmatory_rows]
    if confirmatory_rows:
        candidate_metrics = _probability_metrics(confirmatory_rows, predictions)
        raw_metrics = _probability_metrics(confirmatory_rows, raw_predictions)
        market_deltas = _market_error_deltas(
            confirmatory_rows, predictions, raw_predictions
        )
        delta_values = [
            float(row["brier_improvement"])
            for row in market_deltas["by_market"]
        ]
        bootstrap_values = _bootstrap_market_mean(
            delta_values,
            samples=int(protocol["bootstrap_samples"]),
            seed=int(protocol["bootstrap_seed"]),
        )
        alpha = (1.0 - float(protocol["bootstrap_confidence_level"])) / 2.0
        bootstrap_interval = {
            "confidence_level": protocol["bootstrap_confidence_level"],
            "lower": _quantile(bootstrap_values, alpha),
            "upper": _quantile(bootstrap_values, 1.0 - alpha),
        }
        relative_brier = _relative_improvement(
            raw_metrics["market_weighted_brier_score"],
            candidate_metrics["market_weighted_brier_score"],
        )
        relative_log_loss = _relative_improvement(
            raw_metrics["market_weighted_log_loss"],
            candidate_metrics["market_weighted_log_loss"],
        )
    else:
        candidate_metrics = None
        raw_metrics = None
        market_deltas = {
            "market_count": 0,
            "mean_brier_improvement": None,
            "median_brier_improvement": None,
            "positive_market_count": 0,
            "negative_market_count": 0,
            "zero_market_count": 0,
            "by_market": [],
        }
        bootstrap_interval = {
            "confidence_level": protocol["bootstrap_confidence_level"],
            "lower": None,
            "upper": None,
        }
        relative_brier = None
        relative_log_loss = None
    checks = {
        "confirmatory_support_passed": support["passed"],
        "relative_brier_improvement_passed": relative_brier is not None
        and relative_brier
        >= float(protocol["minimum_relative_brier_improvement"]),
        "relative_log_loss_improvement_passed": relative_log_loss is not None
        and relative_log_loss
        >= float(protocol["minimum_relative_log_loss_improvement"]),
        "positive_market_mean_brier_improvement_passed": (
            market_deltas["mean_brier_improvement"] is not None
            and market_deltas["mean_brier_improvement"] > 0.0
        ),
        "market_bootstrap_lower_bound_positive_passed": (
            bootstrap_interval["lower"] is not None
            and bootstrap_interval["lower"] > 0.0
        ),
        "frozen_contract_hash_verified": _sha256_file(
            Path(contract_descriptor["path"])
        )
        == contract_descriptor["sha256"],
        "exactly_once_marker_verified": marker_path.exists(),
    }
    reason_by_check = {
        "confirmatory_support_passed": "confirmatory_support_gate_failed",
        "relative_brier_improvement_passed": (
            "confirmatory_relative_brier_improvement_failed"
        ),
        "relative_log_loss_improvement_passed": (
            "confirmatory_relative_log_loss_improvement_failed"
        ),
        "positive_market_mean_brier_improvement_passed": (
            "confirmatory_positive_market_mean_failed"
        ),
        "market_bootstrap_lower_bound_positive_passed": (
            "confirmatory_market_bootstrap_interval_crosses_zero"
        ),
        "frozen_contract_hash_verified": "frozen_contract_hash_mismatch",
        "exactly_once_marker_verified": "exactly_once_marker_missing",
    }
    passed = all(checks.values())
    prediction_rows = []
    for row, prediction, raw in zip(
        confirmatory_rows, predictions, raw_predictions, strict=True
    ):
        prediction_rows.append(
            {
                "row_identity": row["row_identity"],
                "market_id": row["market_id"],
                "decision_ts": row["decision_ts"],
                "selected_side": row["selected_side"],
                "selected_action": row["selected_action"],
                "target": row["selected_side_win_target"],
                "candidate_probability": prediction,
                "raw_market_probability": raw,
                "outcome_used_for_evaluation_only": True,
            }
        )
    prediction_path = output_dir / "hts_residual_confirmatory_prediction_rows.jsonl"
    _write_jsonl(prediction_path, prediction_rows)
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-evaluation-report-v1",
        "run_id": input_manifest["run_id"],
        "status": "PRE_PROMOTION_READY" if passed else "PRE_PROMOTION_BLOCKED",
        "candidate_name": contract["candidate_name"],
        "frozen_candidate": contract_descriptor,
        "confirmatory_protocol": protocol_descriptor,
        "confirmatory_input_manifest": _descriptor(input_manifest_path),
        "exactly_once_marker": _descriptor(marker_path),
        "source_corpus_audits": audits,
        "source_decision_row_count": len(public_rows),
        "selected_action_distribution": dict(sorted(selected_counts.items())),
        "outcome_blind_input_selection_hash_verified": True,
        "outcome_blind_input_selection_identity_sha256": (
            confirmatory_selection_hash
        ),
        "confirmatory_row_count": len(confirmatory_rows),
        "confirmatory_market_count": len(
            {str(row["market_id"]) for row in confirmatory_rows}
        ),
        "support": support,
        "candidate_metrics": candidate_metrics,
        "raw_market_probability_metrics": raw_metrics,
        "relative_brier_improvement_vs_raw": relative_brier,
        "relative_log_loss_improvement_vs_raw": relative_log_loss,
        "market_level_error_deltas": market_deltas,
        "market_bootstrap_mean_improvement_interval": bootstrap_interval,
        "confirmatory_gate_checks": checks,
        "confirmatory_gate_passed": passed,
        "blocking_reason_codes": sorted(
            {
                *support["blocking_reason_codes"],
                *[
                    reason_by_check[name]
                    for name, check_passed in checks.items()
                    if not check_passed
                ],
            }
        ),
        "candidate_or_threshold_tuning_performed": False,
        "confirmatory_labels_used_for_fitting": False,
        "confirmatory_labels_used_for_evaluation_only": True,
        "pre_promotion_ready": passed,
        **_blocked_safety(),
    }
    report_path = output_dir / "hts_residual_confirmatory_evaluation_report.json"
    _write_json(report_path, report)
    _write_text(
        output_dir / "hts_residual_confirmatory_evaluation_report.md",
        _evaluation_markdown(report),
    )
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-evaluation-manifest-v1",
        "run_id": input_manifest["run_id"],
        "report": _descriptor(report_path),
        "prediction_rows": _descriptor(prediction_path),
        "exactly_once_marker": _descriptor(marker_path),
        "confirmatory_gate_passed": passed,
        "pre_promotion_ready": passed,
        "blocking_reason_codes": report["blocking_reason_codes"],
        **_blocked_safety(),
    }
    manifest_path = output_dir / "hts_residual_confirmatory_evaluation_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "output_dir": output_dir,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "report": report,
    }


def _confirmatory_support(
    rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    markets = {str(row["market_id"]) for row in rows}
    runs = {str(row["source_run_id"]) for row in rows}
    side_counts = Counter(str(row["selected_side"]) for row in rows)
    outcome_counts = Counter(
        str(row["target_provenance"]["resolved_outcome"]) for row in rows
    )
    checks = {
        "minimum_market_count_met": len(markets)
        >= int(protocol["minimum_confirmatory_market_count"]),
        "minimum_source_run_count_met": len(runs)
        >= int(protocol["minimum_confirmatory_source_run_count"]),
        "both_selected_sides_present": set(side_counts) == {"UP", "DOWN"},
        "both_resolved_outcomes_present": set(outcome_counts) == {"UP", "DOWN"},
    }
    reasons = {
        "minimum_market_count_met": "insufficient_confirmatory_market_support",
        "minimum_source_run_count_met": "insufficient_confirmatory_source_runs",
        "both_selected_sides_present": "missing_confirmatory_selected_side_support",
        "both_resolved_outcomes_present": "missing_confirmatory_outcome_support",
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "blocking_reason_codes": [
            reasons[name] for name, passed in checks.items() if not passed
        ],
        "market_count": len(markets),
        "source_run_count": len(runs),
        "selected_side_counts": dict(sorted(side_counts.items())),
        "resolved_outcome_counts": dict(sorted(outcome_counts.items())),
    }


def _build_outcome_blind_confirmatory_public_rows(
    *, source_descriptors: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    public_rows: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    required_chainlink_fields = (
        "chainlink_price_at_decision",
        "chainlink_reference_price_at_market_start",
        "chainlink_reference_distance_at_decision",
        "chainlink_momentum_30s",
        "chainlink_momentum_60s",
        "chainlink_momentum_120s",
        "chainlink_realized_volatility_120s",
    )
    for source in source_descriptors:
        feature_rows = _load_jsonl(Path(source["feature_rows"]["path"]))
        metadata_rows = _load_jsonl(Path(source["market_metadata"]["path"]))
        chainlink_rows = _load_jsonl(Path(source["chainlink_prices"]["path"]))
        provenance = _load_json(Path(source["training_provenance"]["path"]))
        if _forbidden_outcome_fields_present(
            [feature_rows, metadata_rows, chainlink_rows, provenance]
        ):
            rejections.append(
                {
                    "source_corpus_dir": source["corpus_dir"],
                    "reason_code": "forbidden_outcome_field_in_input_freeze_source",
                }
            )
            continue
        metadata_by_market = {
            str(row["market_id"]): row for row in metadata_rows
        }
        for row_index, feature_row in enumerate(feature_rows):
            market_id = str(feature_row["market_id"])
            decision_ts = int(feature_row["decision_ts"])
            market = metadata_by_market.get(market_id)
            if market is None:
                rejections.append(
                    {
                        "market_id": market_id,
                        "decision_ts": decision_ts,
                        "reason_code": "confirmatory_market_metadata_missing",
                    }
                )
                continue
            try:
                public_row = _phase2_feature_to_public_row(
                    run_id=str(
                        provenance.get("run_id")
                        or provenance.get("corpus_id")
                        or Path(source["corpus_dir"]).name
                    ),
                    row_index=row_index,
                    feature_row=feature_row,
                    market=market,
                    chainlink_rows=chainlink_rows,
                )
            except (KeyError, TypeError, ValueError) as exc:
                rejections.append(
                    {
                        "market_id": market_id,
                        "decision_ts": decision_ts,
                        "reason_code": (
                            "confirmatory_decision_time_feature_construction_error"
                        ),
                        "error": str(exc),
                    }
                )
                continue
            chainlink_provenance = dict(
                public_row.get("chainlink_regime_feature_provenance") or {}
            )
            missing_chainlink = [
                field
                for field in required_chainlink_fields
                if not isinstance(public_row.get(field), int | float)
                or not math.isfinite(float(public_row[field]))
            ]
            mapping_provenance = dict(
                public_row.get("market_probability_mapping_provenance") or {}
            )
            reasons = []
            if missing_chainlink or chainlink_provenance.get("provenance_valid") is not True:
                reasons.append("confirmatory_causal_chainlink_features_missing")
            if (
                public_row.get("market_probability_mapping_rule_id")
                != PHASE2_MARKET_PROBABILITY_MAPPING_RULE_ID
                or mapping_provenance.get("provenance_valid") is not True
            ):
                reasons.append("confirmatory_market_probability_mapping_invalid")
            if int(public_row["decision_time_feature_max_input_ts"]) > decision_ts:
                reasons.append("confirmatory_joined_feature_causality_violation")
            if reasons:
                rejections.extend(
                    {
                        "market_id": market_id,
                        "decision_ts": decision_ts,
                        "reason_code": reason,
                        "missing_chainlink_fields": missing_chainlink,
                    }
                    for reason in reasons
                )
                continue
            public_row["source_run_id"] = str(
                provenance.get("run_id")
                or provenance.get("corpus_id")
                or Path(source["corpus_dir"]).name
            )
            public_rows.append(public_row)
    public_rows.sort(
        key=lambda row: (int(row["decision_ts"]), str(row["market_id"]))
    )
    return public_rows, rejections


def _outcome_blind_hts_selection_rows(
    scoring: dict[str, Any], public_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    source_by_key = {
        (str(row["market_id"]), int(row["decision_ts"])): str(row["source_run_id"])
        for row in public_rows
    }
    rows = []
    for selected in scoring["canonical_scorer_report"][
        "canonical_selected_decision_rows"
    ]:
        action = str(selected["action"])
        if action not in HTS_ACTIONS:
            continue
        market_id = str(selected["market_id"])
        decision_ts = int(selected["decision_ts"])
        rows.append(
            {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "selected_action": action,
                "selected_side": "UP" if "BUY_UP" in action else "DOWN",
                "source_run_id": source_by_key[(market_id, decision_ts)],
                "canonical_scored_action_row_hash": selected.get(
                    "canonical_scored_action_row_hash"
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            int(row["decision_ts"]),
            str(row["market_id"]),
            str(row["selected_action"]),
        ),
    )


def _forbidden_outcome_fields_present(payload: Any) -> list[str]:
    forbidden = {
        "resolved_outcome",
        "settlement_pnl",
        "settlement_return",
        "settlement_payout",
        "oracle_action",
        "future_return",
        "selected_side_win_target",
        "target_net_return_after_cost",
        "total_net_return",
    }
    found: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key) in forbidden:
                    found.add(str(key))
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    return sorted(found)


def _freeze_source_corpus_without_outcomes(corpus_dir: Path) -> dict[str, Any]:
    required = (
        "polymarket_corpus_manifest.json",
        "polymarket_corpus_summary.json",
        "polymarket_feature_rows.jsonl",
        "polymarket_label_rows.jsonl",
        "polymarket_market_metadata.jsonl",
        "polymarket_resolution_events.jsonl",
        "polymarket_chainlink_prices.jsonl",
        "polymarket_chainlink_decision_time_evidence_manifest.json",
        "training_corpus_provenance.json",
    )
    missing = [name for name in required if not (corpus_dir / name).exists()]
    if missing:
        raise ValueError(f"confirmatory source artifacts missing: {missing}")
    corpus_manifest = _load_json(corpus_dir / "polymarket_corpus_manifest.json")
    hashes = dict(corpus_manifest.get("normalized_artifact_hashes") or {})
    expected = {
        "corpus_summary": "polymarket_corpus_summary.json",
        "feature_rows": "polymarket_feature_rows.jsonl",
        "label_rows": "polymarket_label_rows.jsonl",
        "market_metadata": "polymarket_market_metadata.jsonl",
        "resolution_events": "polymarket_resolution_events.jsonl",
    }
    for key, name in expected.items():
        if hashes.get(key) != _sha256_file(corpus_dir / name):
            raise ValueError(f"confirmatory source hash mismatch: {name}")
    chainlink_manifest = _load_json(
        corpus_dir / "polymarket_chainlink_decision_time_evidence_manifest.json"
    )
    if chainlink_manifest.get("evidence_sha256") != _sha256_file(
        corpus_dir / "polymarket_chainlink_prices.jsonl"
    ):
        raise ValueError("confirmatory Chainlink hash mismatch")
    feature_rows = _load_jsonl(corpus_dir / "polymarket_feature_rows.jsonl")
    market_ids = sorted({str(row["market_id"]) for row in feature_rows})
    corpus_summary = _load_json(corpus_dir / "polymarket_corpus_summary.json")
    resolution_hashes = dict(corpus_summary.get("resolution_hashes") or {})
    target_structure_checks = {
        "summary_market_count_matches": int(corpus_summary.get("market_count") or 0)
        == len(market_ids),
        "summary_feature_row_count_matches": int(
            corpus_summary.get("feature_row_count") or 0
        )
        == len(feature_rows),
        "complete_5_action_label_count_reported": int(
            corpus_summary.get("label_row_count") or 0
        )
        == len(feature_rows) * 5,
        "resolution_hash_coverage_complete": set(resolution_hashes) == set(market_ids)
        and all(_is_sha256(str(value)) for value in resolution_hashes.values()),
        "sell_before_close_label_gate_passed": (
            corpus_summary.get("sell_before_close_label_gate_passed") is True
        ),
    }
    if not all(target_structure_checks.values()):
        raise ValueError(
            "confirmatory source outcome-blind target structure is incomplete: "
            + ",".join(
                name for name, passed in target_structure_checks.items() if not passed
            )
        )
    return {
        "corpus_dir": str(corpus_dir),
        "corpus_manifest": _descriptor(
            corpus_dir / "polymarket_corpus_manifest.json"
        ),
        "corpus_summary": _descriptor(
            corpus_dir / "polymarket_corpus_summary.json"
        ),
        "feature_rows": _descriptor(corpus_dir / "polymarket_feature_rows.jsonl"),
        "label_rows": _descriptor(corpus_dir / "polymarket_label_rows.jsonl"),
        "market_metadata": _descriptor(
            corpus_dir / "polymarket_market_metadata.jsonl"
        ),
        "resolution_events": _descriptor(
            corpus_dir / "polymarket_resolution_events.jsonl"
        ),
        "chainlink_prices": _descriptor(
            corpus_dir / "polymarket_chainlink_prices.jsonl"
        ),
        "chainlink_manifest": _descriptor(
            corpus_dir / "polymarket_chainlink_decision_time_evidence_manifest.json"
        ),
        "training_provenance": _descriptor(
            corpus_dir / "training_corpus_provenance.json"
        ),
        "source_market_ids": market_ids,
        "source_market_count": len(market_ids),
        "feature_row_count": len(feature_rows),
        "outcome_blind_target_structure_checks": target_structure_checks,
        "outcome_blind_target_structure_verified": True,
        "target_artifact_values_parsed": False,
    }


def _verify_confirmatory_source_descriptors(
    input_manifest: dict[str, Any],
) -> None:
    descriptor_names = (
        "corpus_manifest",
        "corpus_summary",
        "feature_rows",
        "label_rows",
        "market_metadata",
        "resolution_events",
        "chainlink_prices",
        "chainlink_manifest",
        "training_provenance",
    )
    for source in input_manifest.get("source_corpora") or []:
        for name in descriptor_names:
            _verified_descriptor(source[name])


def _freeze_time(config: HTSResidualCandidateFreezeConfig) -> tuple[str, int]:
    if (config.freeze_created_at is None) != (config.freeze_created_ts is None):
        raise ValueError("freeze_created_at and freeze_created_ts must be set together")
    if config.freeze_created_at is not None and config.freeze_created_ts is not None:
        return config.freeze_created_at, config.freeze_created_ts
    now = datetime.now(UTC)
    return now.isoformat(), int(now.timestamp() * 1000)


def _verified_descriptor(descriptor: dict[str, Any]) -> dict[str, str]:
    path = Path(str(descriptor.get("path") or "")).resolve()
    expected = str(descriptor.get("sha256") or "")
    if not path.exists() or _sha256_file(path) != expected:
        raise ValueError(f"artifact descriptor hash mismatch: {path}")
    return {"path": str(path), "sha256": expected}


def _descriptor(path: Path) -> dict[str, str]:
    path = path.resolve()
    return {"path": str(path), "sha256": _sha256_file(path)}


def _write_sha_descriptor(path: Path) -> Path:
    descriptor = path.with_suffix(path.suffix + ".sha256")
    descriptor.write_text(_sha256_file(path) + "\n", encoding="utf-8")
    return descriptor


def _verify_sha_descriptor(path: Path) -> None:
    descriptor = path.with_suffix(path.suffix + ".sha256")
    if not descriptor.exists() or descriptor.read_text(encoding="utf-8").strip() != (
        _sha256_file(path)
    ):
        raise ValueError("confirmatory input manifest SHA-256 mismatch")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
                + "\n"
            )
    except FileExistsError as exc:
        raise FileExistsError("confirmatory evaluation is exactly-once") from exc


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _freeze_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# HTS Residual Candidate Freeze",
            "",
            f"- candidate: `{report['candidate_name']}`",
            "- development gate verified: `true`",
            "- candidate frozen: `true`",
            "- confirmatory collection/evaluation started: `false`",
            "- pre-promotion ready: `false`",
            "- paper/live/promotion unlock: `false`",
            "",
        ]
    )


def _input_markdown(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# HTS Residual Confirmatory Input Freeze",
            "",
            f"- source markets: `{manifest['source_market_count']}`",
            f"- outcome-blind HTS markets: `{manifest['outcome_blind_hts_selected_market_count']}`",
            f"- outcome-blind selected sides: `{manifest['outcome_blind_hts_selected_side_counts']}`",
            f"- frozen O context verified: `{str(manifest['frozen_o_scorer_context_verified']).lower()}`",
            f"- input gate passed: `{str(manifest['input_gate_passed']).lower()}`",
            f"- blocking reasons: `{manifest['blocking_reason_codes']}`",
            "- outcome values inspected during input freeze: `false`",
            "- confirmatory evaluation started: `false`",
            "- pre-promotion ready: `false`",
            "",
        ]
    )


def _evaluation_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# HTS Residual Exactly-Once Confirmatory Evaluation",
            "",
            f"- status: `{report['status']}`",
            f"- candidate: `{report['candidate_name']}`",
            f"- rows / markets: `{report['confirmatory_row_count']}` / `{report['confirmatory_market_count']}`",
            f"- gate passed: `{str(report['confirmatory_gate_passed']).lower()}`",
            f"- blocking reasons: `{report['blocking_reason_codes']}`",
            "- candidate or threshold tuning: `false`",
            "- paper/live/promotion unlock: `false`",
            "",
        ]
    )
