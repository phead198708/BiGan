"""Outcome-blind per-batch canaries for v8 collection and frozen candidates."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xgboost as xgb

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.exact_model_runtime_binding import (
    validate_runtime_binding_summary,
)
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_evaluation import (
    FORBIDDEN_TARGET_FIELDS,
    _blocked_safety_fields,
)
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_prediction_freeze import (
    _materialize_future_action_rows,
    _materialize_selected_window_features,
)
from bigan.v8.polymarket.training.execution_layer_v2_outcome_blind_acceptance_viability import (
    _outcome_blind_acceptance_replay,
)
from bigan.v8.polymarket.training.execution_layer_v2_p_up_aligned_action_value_support import (
    build_execution_compatible_action_universe,
)
from bigan.v8.polymarket.training.execution_layer_v2_persistent_outcome_blind_collector import (
    load_and_validate_persistent_outcome_blind_index,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    REQUIRED_ACTIONS,
    SIDES,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6_fit import (
    _target_free_predictions,
    apply_policy_selected_conformal_scores,
    attach_frozen_execution_compatibility,
)

SCHEMA_PREFIX = "bigan-v8-outcome-blind-batch-canary"
TRADE_ACTIONS = frozenset(action for action in REQUIRED_ACTIONS if action != "NO_TRADE")
DEFAULT_MIN_ACCEPTED_MARKETS = 120
DEFAULT_MIN_SIDE_MARKETS = 17
DEFAULT_MAXIMUM_INDEX_SCAN_COUNT = 462
DEFAULT_CONSECUTIVE_ZERO_BATCH_LIMIT = 3
DEFAULT_CONSECUTIVE_ZERO_QUALITY_MARKET_MINIMUM = 36


@dataclass(frozen=True, slots=True)
class OutcomeBlindDevelopmentBatchCanaryConfig:
    """Pinned inputs for one development-batch structural canary."""

    run_id: str
    output_dir: Path | str
    collector_index_path: Path | str
    expected_collector_index_sha256: str
    batch_id: str
    feature_contract_path: Path | str
    expected_feature_contract_sha256: str
    exact_model_runtime_binding_summary: dict[str, Any] | None = None
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip() or not self.batch_id.strip():
            raise ValueError("run_id and batch_id are required")
        _require_sha256(self.expected_collector_index_sha256, name="collector index sha256")
        _require_sha256(self.expected_feature_contract_sha256, name="feature contract sha256")
        for name in ("output_dir", "collector_index_path", "feature_contract_path"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if self.exact_model_runtime_binding_summary is not None:
            validate_runtime_binding_summary(
                self.exact_model_runtime_binding_summary
            )


@dataclass(frozen=True, slots=True)
class FrozenModelBatchCanaryConfig:
    """Pinned frozen-v6 inputs for one target-free batch score and guard replay."""

    run_id: str
    output_dir: Path | str
    development_batch_canary_manifest_path: Path | str
    expected_development_batch_canary_manifest_sha256: str
    research_candidate_manifest_path: Path | str
    expected_research_candidate_manifest_sha256: str
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(
            self.expected_development_batch_canary_manifest_sha256,
            name="development canary manifest sha256",
        )
        _require_sha256(
            self.expected_research_candidate_manifest_sha256,
            name="research candidate manifest sha256",
        )
        for name in (
            "output_dir",
            "development_batch_canary_manifest_path",
            "research_candidate_manifest_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


def run_outcome_blind_development_batch_canary(
    config: OutcomeBlindDevelopmentBatchCanaryConfig,
) -> dict[str, Any]:
    """Materialize one finalized batch without opening labels, outcomes, or PnL."""

    index_path = config.collector_index_path.resolve()
    feature_contract_path = config.feature_contract_path.resolve()
    _verify_pin(index_path, config.expected_collector_index_sha256, "collector index")
    _verify_pin(
        feature_contract_path,
        config.expected_feature_contract_sha256,
        "feature contract",
    )
    index_rows = load_and_validate_persistent_outcome_blind_index(index_path)
    batch_rows = [row for row in index_rows if str(row.get("batch_id") or "") == config.batch_id]
    if not batch_rows:
        raise ValueError("batch_id_not_present_in_pinned_index")
    batch_rows.sort(key=lambda row: int(row["sequence"]))
    summary_descriptors = {
        (str(row["batch_summary"]["path"]), str(row["batch_summary"]["sha256"]))
        for row in batch_rows
    }
    if len(summary_descriptors) != 1:
        raise ValueError("batch_summary_descriptor_not_unique")
    summary_path_string, summary_sha256 = next(iter(summary_descriptors))
    batch_summary_path = Path(summary_path_string).resolve()
    _verify_pin(batch_summary_path, summary_sha256, "batch summary")
    batch_summary = _load_json(batch_summary_path)
    if str(batch_summary.get("batch_id") or "") != config.batch_id:
        raise ValueError("batch_summary_identity_mismatch")
    expected_batch_safety = {
        "outcome_blind_collection_only": True,
        "settlement_finalizer_started": False,
        "resolution_provider_called": False,
        "training_corpus_export_attempted": False,
        "labels_or_outcomes_opened_during_collection": False,
        "settlement_pnl_opened_during_collection": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }
    batch_safety_mismatches = [
        key for key, expected in expected_batch_safety.items() if batch_summary.get(key) != expected
    ]
    if batch_safety_mismatches:
        raise ValueError(
            "batch_summary_outcome_or_safety_sealing_invalid:"
            + ",".join(batch_safety_mismatches)
        )
    capture_attempts = list(batch_summary.get("captures") or [])
    error_attempts = list(batch_summary.get("errors") or [])
    if not capture_attempts and not error_attempts:
        raise ValueError("batch_summary_contains_no_attempts")
    if len(batch_rows) != len(capture_attempts):
        raise ValueError("batch_index_capture_count_mismatch")
    verified_raw_descriptor_count = 0
    raw_resolution_row_count = 0
    for row in batch_rows:
        _verified_descriptor(row.get("pending_round_capture_manifest"), "pending capture manifest")
        _verified_descriptor(row.get("pending_round_capture_report"), "pending capture report")
        for filename, descriptor_value in dict(row.get("raw_artifacts") or {}).items():
            descriptor = _verified_descriptor(descriptor_value, f"raw artifact {filename}")
            raw_row_count = len(_load_jsonl(Path(descriptor["path"])))
            if raw_row_count != int(descriptor_value.get("row_count") or 0):
                raise ValueError(f"raw_artifact_row_count_mismatch:{filename}")
            verified_raw_descriptor_count += 1
            if filename == "raw_polymarket_resolutions.jsonl":
                raw_resolution_row_count += raw_row_count
    if raw_resolution_row_count:
        raise ValueError("outcome_blind_batch_contains_resolution_rows")
    quality_rows = [row for row in batch_rows if row.get("capture_quality_valid") is True]
    invalid_reasons = Counter(
        reason
        for row in batch_rows
        for reason in list(row.get("capture_quality_reason_codes") or [])
    )
    feature_contract = _load_json(feature_contract_path)
    feature_columns = tuple(str(value) for value in feature_contract.get("feature_columns") or [])
    if not feature_columns:
        raise ValueError("feature_contract_columns_missing")

    feature_rows: list[dict[str, Any]] = []
    opened_raw_artifacts: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    universe_rows: list[dict[str, Any]] = []
    if quality_rows:
        feature_rows, opened_raw_artifacts = _materialize_selected_window_features(quality_rows)
        action_rows = _materialize_future_action_rows(
            feature_rows,
            selected_rows=quality_rows,
            feature_columns=feature_columns,
        )
        universe_rows = build_execution_compatible_action_universe(action_rows)
    forbidden_fields = sorted(
        set(_find_nonempty_fields(feature_rows, FORBIDDEN_TARGET_FIELDS))
        | set(_find_nonempty_fields(action_rows, FORBIDDEN_TARGET_FIELDS))
        | set(_find_nonempty_fields(universe_rows, FORBIDDEN_TARGET_FIELDS))
    )
    if forbidden_fields:
        raise ValueError("outcome_blind_canary_forbidden_fields:" + ",".join(forbidden_fields))

    expected_decision_groups = len(feature_rows)
    action_groups: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in action_rows:
        action_groups[(str(row["market_id"]), int(row["decision_ts"]))].add(str(row["action"]))
    incomplete_groups = sum(actions != set(REQUIRED_ACTIONS) for actions in action_groups.values())
    feature_causality_violations = sum(
        int(row.get("max_input_ts") or 0) > int(row.get("decision_ts") or 0)
        for row in feature_rows
    )
    action_rows_by_group: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in action_rows:
        action_rows_by_group[(str(row["market_id"]), int(row["decision_ts"]))].append(row)
    feature_coverage = {
        column: sum(
            len(group) == len(REQUIRED_ACTIONS)
            and all(_finite(_decision_feature_value(row, column)) for row in group)
            for group in action_rows_by_group.values()
        )
        for column in feature_columns
    }
    feature_contract_missing_or_nonfinite_action_row_count = sum(
        any(not _finite(_decision_feature_value(row, column)) for column in feature_columns)
        for row in action_rows
    )
    trade_universe = [row for row in universe_rows if row.get("action") in TRADE_ACTIONS]
    original_allowed = [row for row in trade_universe if row["full_guard_original_action_allowed"]]
    aligned_quality = [
        row
        for row in trade_universe
        if row["p_up_alignment_passed"] and row["execution_quality_only_passed"]
    ]
    blocking_reasons = []
    if feature_causality_violations:
        blocking_reasons.append("feature_timestamp_causality_violation")
    if incomplete_groups or len(action_groups) != expected_decision_groups:
        blocking_reasons.append("complete_five_action_grid_failed")
    if feature_contract_missing_or_nonfinite_action_row_count:
        blocking_reasons.append("feature_contract_missing_or_nonfinite_value")
    if forbidden_fields:
        blocking_reasons.append("forbidden_target_or_outcome_field_present")
    if not quality_rows:
        blocking_reasons.append("batch_has_zero_quality_valid_markets")

    run_dir = _prepare_run_dir(
        Path(config.output_dir), config.run_id, overwrite=config.overwrite_existing
    )
    features_path = run_dir / "outcome_blind_batch_feature_rows.jsonl"
    actions_path = run_dir / "outcome_blind_batch_five_action_grid.jsonl"
    universe_path = run_dir / "outcome_blind_batch_execution_compatible_universe.jsonl"
    _write_jsonl(features_path, feature_rows)
    _write_jsonl(actions_path, action_rows)
    _write_jsonl(universe_path, universe_rows)
    runtime_binding_fields = (
        {
            "exact_model_runtime_binding_required": True,
            "exact_model_runtime_binding_verified": True,
            "exact_model_runtime_binding_summary": (
                config.exact_model_runtime_binding_summary
            ),
        }
        if config.exact_model_runtime_binding_summary is not None
        else {
            "exact_model_runtime_binding_required": False,
            "exact_model_runtime_binding_verified": False,
            "exact_model_runtime_binding_summary": None,
        }
    )
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-development-report-v1",
        "run_id": config.run_id,
        "batch_id": config.batch_id,
        "canary_mode": "development_structural_outcome_blind",
        "bounded_batch_complete": True,
        "bounded_batch_attempt_count": len(capture_attempts) + len(error_attempts),
        "bounded_batch_capture_count": len(capture_attempts),
        "bounded_batch_error_count": len(error_attempts),
        "indexed_market_count": len(batch_rows),
        "quality_valid_market_count": len(quality_rows),
        "quality_invalid_market_count": len(batch_rows) - len(quality_rows),
        "quality_invalid_reason_distribution": dict(sorted(invalid_reasons.items())),
        "source_sequence_start": int(batch_rows[0]["sequence"]),
        "source_sequence_end": int(batch_rows[-1]["sequence"]),
        "source_market_ids": [str(row["market_id"]) for row in batch_rows],
        "source_market_ids_sha256": canonical_json_sha256(
            [str(row["market_id"]) for row in batch_rows]
        ),
        "raw_artifact_descriptor_count": sum(
            len(dict(row.get("raw_artifacts") or {})) for row in batch_rows
        ),
        "raw_artifact_hash_verified_count": verified_raw_descriptor_count,
        "raw_resolution_row_count": raw_resolution_row_count,
        "opened_raw_feature_artifact_count": sum(
            len(dict(row.get("raw_feature_artifacts") or {})) for row in opened_raw_artifacts
        ),
        "feature_row_count": len(feature_rows),
        "feature_contract_column_count": len(feature_columns),
        "feature_coverage_count_by_field": feature_coverage,
        "feature_contract_missing_or_nonfinite_action_row_count": (
            feature_contract_missing_or_nonfinite_action_row_count
        ),
        "feature_timestamp_causality_violation_count": feature_causality_violations,
        "decision_group_count": len(action_groups),
        "five_action_row_count": len(action_rows),
        "complete_five_action_grid_passed": (
            incomplete_groups == 0
            and len(action_groups) == expected_decision_groups
            and len(action_rows) == expected_decision_groups * len(REQUIRED_ACTIONS)
        ),
        "incomplete_decision_group_count": incomplete_groups,
        "action_distribution": dict(sorted(Counter(row["action"] for row in action_rows).items())),
        "execution_quality_and_p_up_compatible_action_count": len(aligned_quality),
        "static_full_guard_original_action_allowed_count": len(original_allowed),
        "static_full_guard_original_action_allowed_by_side": _side_counts(original_allowed),
        "static_full_guard_original_action_allowed_by_family": dict(
            sorted(Counter(str(row["action_family"]) for row in original_allowed).items())
        ),
        "guard_blocking_reason_distribution": dict(
            sorted(
                Counter(
                    reason
                    for row in trade_universe
                    for reason in row["execution_blocking_reason_codes"]
                ).items()
            )
        ),
        "development_data_canary_passed": not blocking_reasons,
        "development_data_canary_blocking_reason_codes": blocking_reasons,
        "candidate_model_scoring_attempted": False,
        "candidate_model_viability_evaluated": False,
        "candidate_model_viability_not_evaluated_reason": "candidate_model_not_fitted_before_development_freeze",
        "labels_outcomes_or_pnl_opened": False,
        "resolution_artifact_opened": False,
        "settlement_provider_called": False,
        "threshold_or_guard_tuning_performed": False,
        "model_or_source_score_mutated": False,
        "raw_artifacts_mutated": False,
        "direct_training_corpus_exported": False,
        **runtime_binding_fields,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "execution_layer_v2_outcome_blind_batch_canary_report.json"
    report_md_path = run_dir / "execution_layer_v2_outcome_blind_batch_canary_report.md"
    _write_json(report_path, report)
    _write_text(report_md_path, _development_markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-development-manifest-v1",
        "run_id": config.run_id,
        "batch_id": config.batch_id,
        "collector_index": _descriptor(index_path),
        "batch_summary": _descriptor(batch_summary_path),
        "feature_contract": _descriptor(feature_contract_path),
        "feature_rows": _descriptor(features_path),
        "five_action_grid": _descriptor(actions_path),
        "execution_compatible_universe": _descriptor(universe_path),
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_md_path),
        "development_data_canary_passed": report["development_data_canary_passed"],
        "candidate_model_scoring_attempted": False,
        "labels_outcomes_or_pnl_opened": False,
        **runtime_binding_fields,
        **_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "execution_layer_v2_outcome_blind_batch_canary_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(run_dir, report, report_path, manifest, manifest_path)


def run_frozen_model_batch_canary(config: FrozenModelBatchCanaryConfig) -> dict[str, Any]:
    """Score one already-collected batch with an immutable frozen v6 candidate."""

    development_manifest_path = config.development_batch_canary_manifest_path.resolve()
    candidate_manifest_path = config.research_candidate_manifest_path.resolve()
    _verify_pin(
        development_manifest_path,
        config.expected_development_batch_canary_manifest_sha256,
        "development canary manifest",
    )
    _verify_pin(
        candidate_manifest_path,
        config.expected_research_candidate_manifest_sha256,
        "research candidate manifest",
    )
    development_manifest = _load_json(development_manifest_path)
    candidate_manifest = _load_json(candidate_manifest_path)
    if development_manifest.get("labels_outcomes_or_pnl_opened") is not False:
        raise ValueError("development_canary_target_sealing_invalid")
    if candidate_manifest.get("research_candidate_frozen") is not True:
        raise ValueError("research_candidate_not_frozen")
    if candidate_manifest.get("candidate_specific_future_evaluation_allowed") is not True:
        raise ValueError("research_candidate_future_evaluation_not_allowed")
    development_report_descriptor = _verified_descriptor(
        development_manifest.get("report"), "development canary report"
    )
    development_report = _load_json(Path(development_report_descriptor["path"]))
    if development_report.get("development_data_canary_passed") is not True:
        raise ValueError("development_data_canary_not_passed")
    action_descriptor = _verified_descriptor(
        development_manifest.get("five_action_grid"), "five action grid"
    )
    model_descriptor = _verified_descriptor(candidate_manifest.get("model"), "model")
    artifact_descriptor = _verified_descriptor(
        candidate_manifest.get("calibration_artifact"), "calibration artifact"
    )
    profile_descriptor = _verified_descriptor(candidate_manifest.get("profile"), "profile")
    feature_contract_descriptor = _verified_descriptor(
        candidate_manifest.get("feature_contract"), "feature contract"
    )
    action_rows = _load_jsonl(Path(action_descriptor["path"]))
    if _find_nonempty_fields(action_rows, FORBIDDEN_TARGET_FIELDS):
        raise ValueError("frozen_batch_actions_contain_forbidden_targets")
    feature_contract = _load_json(Path(feature_contract_descriptor["path"]))
    feature_columns = tuple(str(value) for value in feature_contract["feature_columns"])
    profile = _load_json(Path(profile_descriptor["path"]))
    calibration_artifact = _load_json(Path(artifact_descriptor["path"]))
    booster = xgb.Booster()
    booster.load_model(model_descriptor["path"])
    raw = _target_free_predictions(booster, action_rows, feature_columns=feature_columns)
    compatible = attach_frozen_execution_compatibility(raw)
    scored = apply_policy_selected_conformal_scores(
        compatible,
        calibration_artifact=calibration_artifact,
        profile=profile,
    )
    predictions = [
        {
            **row,
            "raw_pairwise_rank_score": float(row["raw_direct_predicted_net_return"]),
            "pairwise_group_normalized_rank_score": float(
                row["raw_direct_predicted_net_return"]
            ),
            "action_advantage_lcb_score_bucket": "not_applicable_policy_selected_conformal",
            "action_advantage_lcb_estimate_source": row["ranking_score_source"],
        }
        for row in scored
    ]
    replay = _outcome_blind_acceptance_replay(
        predictions,
        entry_threshold=0.0,
        runner_up_advantage_threshold=0.0,
    )
    accepted = [row for row in replay if row["execution_guard_order_allowed"]]
    positive_trade_lcb = [
        row
        for row in predictions
        if row["action"] in TRADE_ACTIONS
        and row["guard_compatible_before_ranking"]
        and float(row["conformal_net_return_lower_bound"]) > 0.0
    ]
    selected_no_trade = [row for row in replay if row["source_selected_action"] == "NO_TRADE"]
    run_dir = _prepare_run_dir(
        Path(config.output_dir), config.run_id, overwrite=config.overwrite_existing
    )
    predictions_path = run_dir / "frozen_model_batch_target_free_predictions.jsonl"
    replay_path = run_dir / "frozen_model_batch_outcome_blind_guard_replay.jsonl"
    _write_jsonl(predictions_path, predictions)
    _write_jsonl(replay_path, replay)
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-frozen-model-batch-report-v1",
        "run_id": config.run_id,
        "batch_id": development_manifest["batch_id"],
        "canary_mode": "frozen_model_target_free_batch_scoring",
        "bounded_batch_complete": development_report["bounded_batch_complete"],
        "indexed_market_count": int(development_report["indexed_market_count"]),
        "source_sequence_start": int(development_report["source_sequence_start"]),
        "source_sequence_end": int(development_report["source_sequence_end"]),
        "quality_valid_market_count": len({str(row["market_id"]) for row in action_rows}),
        "decision_group_count": len(replay),
        "positive_guard_compatible_trade_lcb_row_count": len(positive_trade_lcb),
        "positive_guard_compatible_trade_lcb_market_count": len(
            {str(row["market_id"]) for row in positive_trade_lcb}
        ),
        "selected_no_trade_count": len(selected_no_trade),
        "selected_no_trade_rate": len(selected_no_trade) / len(replay) if replay else 0.0,
        "guard_accepted_decision_count": len(accepted),
        "guard_accepted_unique_market_count": len(
            {str(row["market_id"]) for row in accepted}
        ),
        "guard_accepted_market_ids": sorted({str(row["market_id"]) for row in accepted}),
        "guard_accepted_market_ids_by_side": {
            side: sorted(
                {
                    str(row["market_id"])
                    for row in accepted
                    if str(row.get("selected_side") or "") == side
                }
            )
            for side in SIDES
        },
        "guard_accepted_by_side": _side_counts(accepted),
        "guard_blocking_reason_distribution": dict(
            sorted(
                Counter(
                    reason
                    for row in replay
                    for reason in row["execution_blocking_reason_codes"]
                ).items()
            )
        ),
        "all_trade_lcbs_nonpositive_decision_count": sum(
            bool(row["all_trade_action_lcbs_nonpositive"]) for row in replay
        ),
        "candidate_model_sha256": model_descriptor["sha256"],
        "calibration_artifact_sha256": artifact_descriptor["sha256"],
        "candidate_model_scoring_attempted": True,
        "target_free_scoring_passed": True,
        "labels_outcomes_or_pnl_opened": False,
        "threshold_or_guard_tuning_performed": False,
        "model_or_source_score_mutated": False,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "execution_layer_v2_frozen_model_batch_canary_report.json"
    _write_json(report_path, report)
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-frozen-model-batch-manifest-v1",
        "run_id": config.run_id,
        "batch_id": report["batch_id"],
        "development_batch_canary_manifest": _descriptor(development_manifest_path),
        "research_candidate_manifest": _descriptor(candidate_manifest_path),
        "target_free_predictions": _descriptor(predictions_path),
        "outcome_blind_guard_replay": _descriptor(replay_path),
        "report": _descriptor(report_path),
        "labels_outcomes_or_pnl_opened": False,
        **_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "execution_layer_v2_frozen_model_batch_canary_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(run_dir, report, report_path, manifest, manifest_path)


def build_frozen_model_cumulative_canary(
    batch_reports: list[dict[str, Any]],
    *,
    run_id: str,
    minimum_accepted_market_count: int = DEFAULT_MIN_ACCEPTED_MARKETS,
    minimum_side_market_count: int = DEFAULT_MIN_SIDE_MARKETS,
    maximum_index_scan_count: int = DEFAULT_MAXIMUM_INDEX_SCAN_COUNT,
    consecutive_zero_batch_limit: int = DEFAULT_CONSECUTIVE_ZERO_BATCH_LIMIT,
    consecutive_zero_quality_market_minimum: int = (
        DEFAULT_CONSECUTIVE_ZERO_QUALITY_MARKET_MINIMUM
    ),
) -> dict[str, Any]:
    """Aggregate target-free frozen-model batches and apply pre-registered early stops."""

    if not batch_reports:
        raise ValueError("at least one frozen-model batch report is required")
    seen_batch_ids: set[str] = set()
    accepted_market_ids: set[str] = set()
    accepted_by_side: dict[str, set[str]] = {side: set() for side in SIDES}
    previous_sequence_end: int | None = None
    for report in batch_reports:
        batch_id = str(report.get("batch_id") or "")
        if not batch_id or batch_id in seen_batch_ids:
            raise ValueError("frozen_model_batch_identity_missing_or_duplicate")
        seen_batch_ids.add(batch_id)
        if report.get("labels_outcomes_or_pnl_opened") is not False:
            raise ValueError("frozen_model_batch_target_sealing_invalid")
        sequence_start = int(report.get("source_sequence_start") or 0)
        sequence_end = int(report.get("source_sequence_end") or 0)
        if sequence_start <= 0 or sequence_end < sequence_start:
            raise ValueError("frozen_model_batch_source_sequence_invalid")
        if previous_sequence_end is not None and sequence_start != previous_sequence_end + 1:
            raise ValueError("frozen_model_batch_source_sequence_not_contiguous")
        previous_sequence_end = sequence_end
        for key, expected in _blocked_safety_fields().items():
            if report.get(key) != expected:
                raise ValueError(f"frozen_model_batch_safety_invalid:{key}")
        batch_markets = {
            str(value) for value in report.get("guard_accepted_market_ids") or []
        }
        accepted_market_ids.update(batch_markets)
        side_map = dict(report.get("guard_accepted_market_ids_by_side") or {})
        if side_map:
            for side in SIDES:
                accepted_by_side[side].update(str(value) for value in side_map.get(side) or [])
        else:
            side_counts = dict(report.get("guard_accepted_by_side") or {})
            if sum(int(value) for value in side_counts.values()) != len(batch_markets):
                raise ValueError("accepted_side_counts_do_not_reconcile")
            # Reports produced by the runner include unique accepted markets. For external
            # summaries without identities, preserve counts using deterministic sentinels.
            cursor = 0
            ordered = sorted(batch_markets)
            for side in SIDES:
                count = int(side_counts.get(side) or 0)
                accepted_by_side[side].update(ordered[cursor : cursor + count])
                cursor += count

    scanned = sum(int(report.get("indexed_market_count") or report["quality_valid_market_count"])
                  for report in batch_reports)
    remaining_capacity = max(0, maximum_index_scan_count - scanned)
    trailing = []
    trailing_quality_count = 0
    for report in reversed(batch_reports):
        if (
            report.get("bounded_batch_complete") is True
            and int(report["positive_guard_compatible_trade_lcb_row_count"]) == 0
            and int(report["guard_accepted_unique_market_count"]) == 0
        ):
            trailing.append(str(report["batch_id"]))
            trailing_quality_count += int(report["quality_valid_market_count"])
        else:
            break
    zero_signal_stop = (
        len(trailing) >= consecutive_zero_batch_limit
        and sum(
            int(report["quality_valid_market_count"])
            for report in batch_reports[-consecutive_zero_batch_limit:]
        )
        >= consecutive_zero_quality_market_minimum
    )
    accepted_capacity_impossible = (
        len(accepted_market_ids) + remaining_capacity < minimum_accepted_market_count
    )
    side_capacity_impossible = [
        side
        for side in SIDES
        if len(accepted_by_side[side]) + remaining_capacity < minimum_side_market_count
    ]
    reason_codes = []
    if zero_signal_stop:
        reason_codes.append("three_consecutive_complete_batches_zero_positive_lcb_and_guard_acceptance")
    if accepted_capacity_impossible:
        reason_codes.append("remaining_scan_capacity_cannot_reach_minimum_accepted_support")
    reason_codes.extend(
        f"remaining_scan_capacity_cannot_reach_{side.lower()}_side_support"
        for side in side_capacity_impossible
    )
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-frozen-model-cumulative-report-v1",
        "run_id": run_id,
        "batch_count": len(batch_reports),
        "batch_ids": [str(report["batch_id"]) for report in batch_reports],
        "indexed_market_count": scanned,
        "quality_valid_market_count": sum(
            int(report["quality_valid_market_count"]) for report in batch_reports
        ),
        "positive_guard_compatible_trade_lcb_row_count": sum(
            int(report["positive_guard_compatible_trade_lcb_row_count"])
            for report in batch_reports
        ),
        "guard_accepted_unique_market_count": len(accepted_market_ids),
        "guard_accepted_unique_market_count_by_side": {
            side: len(accepted_by_side[side]) for side in SIDES
        },
        "minimum_accepted_market_count": minimum_accepted_market_count,
        "minimum_side_market_count": minimum_side_market_count,
        "maximum_index_scan_count": maximum_index_scan_count,
        "remaining_maximum_market_capacity": remaining_capacity,
        "consecutive_zero_signal_batch_count": len(trailing),
        "consecutive_zero_signal_quality_market_count": trailing_quality_count,
        "consecutive_zero_signal_batch_ids": list(reversed(trailing)),
        "target_free_terminal_blocked": bool(reason_codes),
        "target_free_terminal_blocking_reason_codes": reason_codes,
        "single_weak_batch_is_diagnostic_only": True,
        "labels_outcomes_or_pnl_opened": False,
        "threshold_or_guard_tuning_performed": False,
        "model_or_source_score_mutated": False,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def build_v5_retrospective_no_trade_canary_report(
    prediction_rows: list[dict[str, Any]],
    *,
    run_id: str,
    batch_market_count: int = 12,
) -> dict[str, Any]:
    """Show when target-free batch monitoring would have stopped terminal v5."""

    if batch_market_count <= 0:
        raise ValueError("batch_market_count must be positive")
    if _find_nonempty_fields(prediction_rows, FORBIDDEN_TARGET_FIELDS):
        raise ValueError("v5 retrospective rows contain forbidden targets")
    ordered_markets: list[str] = []
    by_market: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(
        prediction_rows,
        key=lambda value: (
            int(value.get("future_window_selection_rank") or 0),
            int(value.get("decision_ts") or 0),
            str(value.get("market_id") or ""),
        ),
    ):
        market_id = str(row.get("market_id") or "")
        if not market_id:
            raise ValueError("v5 retrospective market identity missing")
        if market_id not in by_market:
            ordered_markets.append(market_id)
        by_market[market_id].append(row)
    batch_reports = []
    for offset in range(0, len(ordered_markets), batch_market_count):
        market_ids = ordered_markets[offset : offset + batch_market_count]
        rows = [row for market_id in market_ids for row in by_market[market_id]]
        positive = [
            row
            for row in rows
            if row.get("action") in TRADE_ACTIONS
            and row.get("guard_compatible_before_ranking") is True
            and float(row.get("conformal_net_return_lower_bound") or 0.0) > 0.0
        ]
        accepted = {
            str(row["market_id"])
            for row in rows
            if row.get("execution_guard_order_allowed") is True
        }
        batch_reports.append(
            {
                "batch_id": f"retrospective-batch-{len(batch_reports) + 1:03d}",
                "bounded_batch_complete": len(market_ids) == batch_market_count,
                "quality_valid_market_count": len(market_ids),
                "positive_guard_compatible_trade_lcb_row_count": len(positive),
                "guard_accepted_unique_market_count": len(accepted),
            }
        )
    first_stop_after_batch = None
    for index in range(DEFAULT_CONSECUTIVE_ZERO_BATCH_LIMIT, len(batch_reports) + 1):
        window = batch_reports[index - DEFAULT_CONSECUTIVE_ZERO_BATCH_LIMIT : index]
        if (
            all(report["bounded_batch_complete"] for report in window)
            and sum(report["quality_valid_market_count"] for report in window)
            >= DEFAULT_CONSECUTIVE_ZERO_QUALITY_MARKET_MINIMUM
            and all(
                report["positive_guard_compatible_trade_lcb_row_count"] == 0
                and report["guard_accepted_unique_market_count"] == 0
                for report in window
            )
        ):
            first_stop_after_batch = index
            break
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-v5-retrospective-report-v1",
        "run_id": run_id,
        "diagnostic_only": True,
        "retrospective_changes_historical_v5_result": False,
        "batch_market_count": batch_market_count,
        "market_count": len(ordered_markets),
        "batch_reports": batch_reports,
        "first_target_free_terminal_stop_after_batch": first_stop_after_batch,
        "first_target_free_terminal_stop_after_market_count": (
            first_stop_after_batch * batch_market_count if first_stop_after_batch else None
        ),
        "labels_outcomes_or_pnl_opened": False,
        "v5_would_have_been_blocked_earlier": first_stop_after_batch is not None,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def write_frozen_model_cumulative_canary(
    *,
    report: dict[str, Any],
    batch_report_paths: list[Path],
    output_dir: Path,
    run_id: str,
    overwrite_existing: bool = False,
) -> dict[str, Any]:
    """Persist a cumulative canary with pinned batch evidence."""

    run_dir = _prepare_run_dir(output_dir, run_id, overwrite=overwrite_existing)
    report_path = run_dir / "execution_layer_v2_frozen_model_cumulative_canary_report.json"
    report_md_path = run_dir / "execution_layer_v2_frozen_model_cumulative_canary_report.md"
    _write_json(report_path, report)
    _write_text(report_md_path, _cumulative_markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-frozen-model-cumulative-manifest-v1",
        "run_id": run_id,
        "batch_reports": [_descriptor(path.resolve()) for path in batch_report_paths],
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_md_path),
        "target_free_terminal_blocked": report["target_free_terminal_blocked"],
        "target_free_terminal_blocking_reason_codes": report[
            "target_free_terminal_blocking_reason_codes"
        ],
        "labels_outcomes_or_pnl_opened": False,
        **_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "execution_layer_v2_frozen_model_cumulative_canary_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(run_dir, report, report_path, manifest, manifest_path)


def write_v5_retrospective_no_trade_canary_report(
    *,
    report: dict[str, Any],
    source_prediction_path: Path,
    output_dir: Path,
    run_id: str,
    overwrite_existing: bool = False,
) -> dict[str, Any]:
    """Persist the diagnostic-only v5 historical early-stop analysis."""

    run_dir = _prepare_run_dir(output_dir, run_id, overwrite=overwrite_existing)
    report_path = run_dir / "execution_layer_v2_v5_retrospective_no_trade_canary_report.json"
    report_md_path = run_dir / "execution_layer_v2_v5_retrospective_no_trade_canary_report.md"
    _write_json(report_path, report)
    _write_text(
        report_md_path,
        "\n".join(
            [
                "# v5 retrospective target-free no-trade canary",
                "",
                f"- Markets: `{report['market_count']}`",
                f"- First stop after batch: `{report['first_target_free_terminal_stop_after_batch']}`",
                f"- First stop after markets: `{report['first_target_free_terminal_stop_after_market_count']}`",
                f"- Earlier block detected: `{str(report['v5_would_have_been_blocked_earlier']).lower()}`",
                "- Historical v5 result changed: `false`",
                "- Labels/outcomes/PnL opened: `false`",
                "",
            ]
        ),
    )
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-v5-retrospective-manifest-v1",
        "run_id": run_id,
        "source_target_free_predictions": _descriptor(source_prediction_path.resolve()),
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_md_path),
        "retrospective_changes_historical_v5_result": False,
        "labels_outcomes_or_pnl_opened": False,
        **_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "execution_layer_v2_v5_retrospective_no_trade_canary_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(run_dir, report, report_path, manifest, manifest_path)


def _development_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Outcome-blind batch canary",
            "",
            f"- Batch: `{report['batch_id']}`",
            f"- Quality-valid markets: `{report['quality_valid_market_count']}`",
            f"- Feature rows: `{report['feature_row_count']}`",
            f"- Complete five-action grid: `{str(report['complete_five_action_grid_passed']).lower()}`",
            f"- Static guard-compatible actions: `{report['static_full_guard_original_action_allowed_count']}`",
            f"- Data canary passed: `{str(report['development_data_canary_passed']).lower()}`",
            "- Candidate scoring attempted: `false` (v6 is not fitted before the frozen development window)",
            "- Labels/outcomes/PnL opened: `false`",
            "- Paper/live/promotion unlock: `false`",
            "",
        ]
    )


def _cumulative_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Frozen-model cumulative outcome-blind canary",
            "",
            f"- Batches: `{report['batch_count']}`",
            f"- Quality-valid markets: `{report['quality_valid_market_count']}`",
            f"- Positive trade LCB rows: `{report['positive_guard_compatible_trade_lcb_row_count']}`",
            f"- Guard-accepted markets: `{report['guard_accepted_unique_market_count']}`",
            f"- Terminal blocked: `{str(report['target_free_terminal_blocked']).lower()}`",
            f"- Reasons: `{json.dumps(report['target_free_terminal_blocking_reason_codes'])}`",
            "- Labels/outcomes/PnL opened: `false`",
            "- Paper/live/promotion unlock: `false`",
            "",
        ]
    )


def _side_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("selected_side") or row.get("side") or "") for row in rows)
    return {side: int(counts.get(side, 0)) for side in SIDES}


def _finite(value: Any) -> int:
    return int(isinstance(value, int | float) and math.isfinite(float(value)))


def _decision_feature_value(row: dict[str, Any], column: str) -> Any:
    features = dict(row.get("decision_time_features") or {})
    return features[column] if column in features else row.get(column)


def _find_nonempty_fields(payload: Any, forbidden: set[str] | frozenset[str]) -> list[str]:
    hits: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in forbidden and value not in (None, "", [], {}):
                hits.add(key)
            hits.update(_find_nonempty_fields(value, forbidden))
    elif isinstance(payload, list):
        for value in payload:
            hits.update(_find_nonempty_fields(value, forbidden))
    return sorted(hits)


def _prepare_run_dir(output_dir: Path, run_id: str, *, overwrite: bool) -> Path:
    run_dir = output_dir.resolve() / run_id
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    return run_dir


def _result(
    run_dir: Path,
    report: dict[str, Any],
    report_path: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    return {
        "run_dir": run_dir,
        "report": report,
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def _descriptor(path: Path) -> dict[str, Any]:
    descriptor: dict[str, Any] = {"path": str(path), "sha256": _sha256_file(path)}
    if path.suffix == ".jsonl":
        descriptor["row_count"] = len(_load_jsonl(path))
    return descriptor


def _verified_descriptor(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} descriptor missing")
    path = Path(str(value.get("path") or "")).resolve()
    digest = str(value.get("sha256") or "")
    _verify_pin(path, digest, name)
    return {"path": str(path), "sha256": digest}


def _verify_pin(path: Path, expected: str, name: str) -> None:
    _require_sha256(expected, name=f"{name} sha256")
    if not path.is_file() or _sha256_file(path) != expected:
        raise ValueError(f"{name} hash mismatch")


def _require_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")
