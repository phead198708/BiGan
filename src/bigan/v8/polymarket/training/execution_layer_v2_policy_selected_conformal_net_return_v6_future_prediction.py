"""Freeze #207 v6 future rows, predictions, and accepted-bet decisions target-free."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xgboost as xgb

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_evaluation import (
    FORBIDDEN_TARGET_FIELDS,
)
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_prediction_freeze import (
    _baseline_predictions,
    _blocking_distribution,
    _materialize_future_action_rows,
    _materialize_selected_window_features,
    _side_distribution,
)
from bigan.v8.polymarket.training.execution_layer_v2_guard_compatible_direct_net_return_v4 import (
    validate_guard_compatible_direct_net_return_v4_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_outcome_blind_acceptance_viability import (
    _outcome_blind_acceptance_replay,
)
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb import (
    validate_pairwise_action_advantage_lcb_feature_contract,
)
from bigan.v8.polymarket.training.execution_layer_v2_persistent_outcome_blind_collector import (
    OutcomeBlindWindowFreezeConfig,
    freeze_outcome_blind_window,
    load_and_validate_persistent_outcome_blind_index,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    CANDIDATE_NAME,
    _blocked_safety_fields,
    _descriptor,
    _find_nonempty_fields,
    _load_json,
    _load_jsonl,
    _require_git_sha,
    _require_sha256,
    _sha256_file,
    _verified_descriptor,
    _verify_pin,
    _write_json,
    _write_jsonl,
    _write_text,
    validate_policy_selected_conformal_v6_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6_fit import (
    _target_free_predictions,
    apply_policy_selected_conformal_scores,
    attach_frozen_execution_compatibility,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6_future import (
    PREREG_MANIFEST_SCHEMA_VERSION,
)

SCHEMA_PREFIX = "bigan-v8-policy-selected-conformal-net-return-v6-future-prediction"


@dataclass(frozen=True, slots=True)
class PolicySelectedConformalV6FuturePredictionConfig:
    """Pinned target-free inputs for one immutable 300-market decision freeze."""

    run_id: str
    output_dir: Path | str
    future_preregistration_manifest_path: Path | str
    expected_future_preregistration_manifest_sha256: str
    collector_index_path: Path | str
    expected_collector_index_sha256: str
    feature_contract_path: Path | str
    expected_feature_contract_sha256: str
    builder_git_commit: str
    decision_freeze_created_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for field in (
            "expected_future_preregistration_manifest_sha256",
            "expected_collector_index_sha256",
            "expected_feature_contract_sha256",
        ):
            _require_sha256(str(getattr(self, field)), name=field)
        _require_git_sha(self.builder_git_commit)
        if self.decision_freeze_created_ts <= 0:
            raise ValueError("decision_freeze_created_ts must be positive")
        for field in (
            "output_dir",
            "future_preregistration_manifest_path",
            "collector_index_path",
            "feature_contract_path",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)))


def freeze_policy_selected_conformal_v6_future_predictions(
    config: PolicySelectedConformalV6FuturePredictionConfig,
) -> dict[str, Any]:
    """Freeze the first 300 valid future markets and stop before target access."""

    prereg_path = config.future_preregistration_manifest_path.resolve()
    index_path = config.collector_index_path.resolve()
    feature_contract_path = config.feature_contract_path.resolve()
    _verify_pin(
        prereg_path,
        config.expected_future_preregistration_manifest_sha256,
        "v6 future preregistration",
    )
    _verify_pin(index_path, config.expected_collector_index_sha256, "collector index")
    _verify_pin(
        feature_contract_path,
        config.expected_feature_contract_sha256,
        "feature contract",
    )
    prereg = _load_json(prereg_path)
    _validate_future_preregistration(prereg)
    if config.decision_freeze_created_ts <= int(prereg["preregistration_created_ts"]):
        raise ValueError("future decision freeze must occur after preregistration")
    profile_descriptor = _verified_descriptor(prereg["candidate_profile"], "v6 profile")
    profile = _load_json(Path(profile_descriptor["path"]))
    validate_policy_selected_conformal_v6_profile(profile)
    if (
        config.expected_feature_contract_sha256
        != profile["frozen_upstream"]["feature_contract_sha256"]
    ):
        raise ValueError("future feature contract differs from preregistered v6 contract")
    feature_contract = _load_json(feature_contract_path)
    validate_pairwise_action_advantage_lcb_feature_contract(
        feature_contract,
        expected_parent_protocol_sha256=str(feature_contract["parent_protocol_sha256"]),
    )
    baseline_profile_descriptor = _verified_descriptor(
        prereg["matched_baseline_fit_profile"], "matched v4 fit profile"
    )
    baseline_profile = _load_json(Path(baseline_profile_descriptor["path"]))
    validate_guard_compatible_direct_net_return_v4_profile(baseline_profile)
    current_index_rows = load_and_validate_persistent_outcome_blind_index(index_path)
    _validate_append_only_prefix(prereg, current_index_rows=current_index_rows)

    run_dir = config.output_dir.resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    pre_target_audit = {
        "schema_version": f"{SCHEMA_PREFIX}-pre-target-access-audit-v1",
        "run_id": config.run_id,
        "builder_git_commit": config.builder_git_commit,
        "decision_freeze_created_ts": config.decision_freeze_created_ts,
        "future_preregistration_manifest": _descriptor(prereg_path),
        "collector_index": _descriptor(index_path),
        "candidate_model": prereg["candidate_model"],
        "candidate_calibration_artifact": prereg["candidate_calibration_artifact"],
        "matched_baseline_model": prereg["matched_baseline_model"],
        "feature_contract": _descriptor(feature_contract_path),
        "collector_index_prefix_unchanged": True,
        "raw_feature_artifacts_opened": False,
        "resolution_artifact_opened": False,
        "settlement_provider_called": False,
        "future_labels_outcomes_or_pnl_opened": False,
        "prediction_attempted": False,
        **_blocked_safety_fields(),
    }
    pre_target_audit["audit_id"] = canonical_json_sha256(pre_target_audit)
    audit_path = run_dir / "conformal_v6_future_pre_target_access_audit.json"
    _write_json(audit_path, pre_target_audit)

    protocol_descriptor = _verified_descriptor(prereg["collector_protocol"], "collector protocol")
    boundary_descriptor = _verified_descriptor(
        prereg["source_boundary_manifest"], "future source boundary"
    )
    window_result = freeze_outcome_blind_window(
        OutcomeBlindWindowFreezeConfig(
            run_id="frozen_window",
            output_dir=run_dir,
            protocol_path=protocol_descriptor["path"],
            expected_protocol_sha256=protocol_descriptor["sha256"],
            index_path=index_path,
            expected_index_sha256=config.expected_collector_index_sha256,
            source_boundary_manifest_path=boundary_descriptor["path"],
            expected_source_boundary_manifest_sha256=boundary_descriptor["sha256"],
            target_valid_market_count=int(prereg["target_quality_valid_market_count"]),
            maximum_scan_count=int(prereg["maximum_index_scan_count"]),
            builder_git_commit=config.builder_git_commit,
        )
    )
    if window_result["report"]["window_freeze_ready"] is not True:
        return _write_incomplete_prediction_result(
            config=config,
            run_dir=run_dir,
            prereg_path=prereg_path,
            audit_path=audit_path,
            window_result=window_result,
        )

    window_manifest = window_result["manifest"]
    selected_descriptor = _verified_descriptor(window_manifest["selected_rows"], "future rows")
    selected_rows = _load_jsonl(Path(selected_descriptor["path"]))
    _validate_selected_future_rows(prereg, selected_rows=selected_rows)
    max_selected_market_end_ts = max(int(row["market_end_ts"]) for row in selected_rows)
    if config.decision_freeze_created_ts <= max_selected_market_end_ts:
        raise ValueError("future prediction attempted before every selected market closed")
    feature_rows, raw_descriptors = _materialize_selected_window_features(selected_rows)
    feature_rows_path = run_dir / "conformal_v6_future_target_free_feature_rows.jsonl"
    _write_jsonl(feature_rows_path, feature_rows)
    feature_columns = tuple(str(value) for value in feature_contract["feature_columns"])
    action_rows = _materialize_future_action_rows(
        feature_rows,
        selected_rows=selected_rows,
        feature_columns=feature_columns,
    )
    action_rows_path = run_dir / "conformal_v6_future_target_free_five_action_rows.jsonl"
    _write_jsonl(action_rows_path, action_rows)
    if _find_nonempty_fields(action_rows, FORBIDDEN_TARGET_FIELDS):
        raise ValueError("future action rows contain forbidden target fields")

    candidate_model = _verified_descriptor(prereg["candidate_model"], "v6 model")
    calibration_descriptor = _verified_descriptor(
        prereg["candidate_calibration_artifact"], "v6 calibration artifact"
    )
    candidate_predictions = _candidate_predictions(
        action_rows,
        model_descriptor=candidate_model,
        calibration_artifact=_load_json(Path(calibration_descriptor["path"])),
        profile=profile,
        feature_columns=feature_columns,
    )
    candidate_predictions_path = run_dir / "conformal_v6_future_target_free_predictions.jsonl"
    _write_jsonl(candidate_predictions_path, candidate_predictions)
    candidate_replay = _outcome_blind_acceptance_replay(
        candidate_predictions,
        entry_threshold=0.0,
        runner_up_advantage_threshold=0.0,
    )
    candidate_replay_path = run_dir / "conformal_v6_future_outcome_blind_guard_replay.jsonl"
    _write_jsonl(candidate_replay_path, candidate_replay)

    baseline_model = _verified_descriptor(prereg["matched_baseline_model"], "matched v4 model")
    baseline_predictions = _baseline_predictions(
        action_rows,
        model_descriptor=baseline_model,
        fit_profile=baseline_profile,
        feature_columns=feature_columns,
    )
    baseline_predictions_path = run_dir / "matched_v4_future_target_free_predictions.jsonl"
    _write_jsonl(baseline_predictions_path, baseline_predictions)
    baseline_replay = _outcome_blind_acceptance_replay(
        baseline_predictions,
        entry_threshold=0.0,
        runner_up_advantage_threshold=0.0,
    )
    baseline_replay_path = run_dir / "matched_v4_future_outcome_blind_guard_replay.jsonl"
    _write_jsonl(baseline_replay_path, baseline_replay)

    candidate_allowed = [row for row in candidate_replay if row["execution_guard_order_allowed"]]
    baseline_allowed = [row for row in baseline_replay if row["execution_guard_order_allowed"]]
    candidate_allowed_path = run_dir / "conformal_v6_future_frozen_accepted_bets.jsonl"
    baseline_allowed_path = run_dir / "matched_v4_future_frozen_accepted_bets.jsonl"
    _write_jsonl(candidate_allowed_path, candidate_allowed)
    _write_jsonl(baseline_allowed_path, baseline_allowed)
    support = _future_support_summary(candidate_allowed, prereg=prereg)
    decision_freeze = {
        "schema_version": f"{SCHEMA_PREFIX}-accepted-bet-decision-freeze-v1",
        "run_id": config.run_id,
        "builder_git_commit": config.builder_git_commit,
        "decision_freeze_created_ts": config.decision_freeze_created_ts,
        "future_preregistration_manifest": _descriptor(prereg_path),
        "future_window_manifest": _descriptor(Path(window_result["manifest_path"])),
        "selected_window_rows": selected_descriptor,
        "target_free_feature_rows": _descriptor(feature_rows_path),
        "target_free_five_action_rows": _descriptor(action_rows_path),
        "candidate_target_free_predictions": _descriptor(candidate_predictions_path),
        "candidate_outcome_blind_guard_replay": _descriptor(candidate_replay_path),
        "candidate_frozen_accepted_bets": _descriptor(candidate_allowed_path),
        "matched_baseline_target_free_predictions": _descriptor(baseline_predictions_path),
        "matched_baseline_outcome_blind_guard_replay": _descriptor(baseline_replay_path),
        "matched_baseline_frozen_accepted_bets": _descriptor(baseline_allowed_path),
        "candidate_support": support,
        "candidate_guard_accepted_bet_count": len(candidate_allowed),
        "candidate_guard_accepted_unique_market_count": len(
            {str(row["market_id"]) for row in candidate_allowed}
        ),
        "matched_baseline_guard_accepted_bet_count": len(baseline_allowed),
        "matched_baseline_guard_accepted_unique_market_count": len(
            {str(row["market_id"]) for row in baseline_allowed}
        ),
        "future_target_free_support_gate_passed": support["passed"],
        "future_target_access_allowed_after_decision_freeze": support["passed"],
        "future_target_access_blocking_reason_codes": support["reason_codes"],
        "candidate_and_baseline_same_frozen_window_feature_grid_guard_and_runtime": True,
        "resolution_artifact_opened": False,
        "settlement_provider_called": False,
        "future_labels_outcomes_or_pnl_opened": False,
        "target_or_outcome_used_for_decision": False,
        "decision_freeze_written_before_target_access": True,
        "all_selected_markets_closed_before_decision_freeze": True,
        "result_dependent_extension_allowed": False,
        **_blocked_safety_fields(),
    }
    decision_freeze["decision_freeze_id"] = canonical_json_sha256(decision_freeze)
    freeze_path = run_dir / "conformal_v6_future_accepted_bet_decision_freeze.json"
    _write_json(freeze_path, decision_freeze)
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-report-v1",
        "report_id": None,
        "run_id": config.run_id,
        "selected_market_count": len(selected_rows),
        "target_free_feature_row_count": len(feature_rows),
        "complete_five_action_row_count": len(action_rows),
        "feature_causality_violation_count": sum(
            int(row["max_input_ts"]) > int(row["decision_ts"]) for row in feature_rows
        ),
        "candidate_decision_count": len(candidate_replay),
        "candidate_guard_accepted_bet_count": len(candidate_allowed),
        "candidate_guard_accepted_unique_market_count": len(
            {str(row["market_id"]) for row in candidate_allowed}
        ),
        "candidate_guard_accepted_side_distribution": _side_distribution(candidate_allowed),
        "candidate_blocking_reason_distribution": _blocking_distribution(candidate_replay),
        "matched_baseline_decision_count": len(baseline_replay),
        "matched_baseline_guard_accepted_bet_count": len(baseline_allowed),
        "matched_baseline_guard_accepted_unique_market_count": len(
            {str(row["market_id"]) for row in baseline_allowed}
        ),
        "matched_baseline_guard_accepted_side_distribution": _side_distribution(
            baseline_allowed
        ),
        "matched_baseline_blocking_reason_distribution": _blocking_distribution(
            baseline_replay
        ),
        "future_target_free_support": support,
        "prediction_attempted": True,
        "future_target_free_support_gate_passed": support["passed"],
        "future_target_access_allowed_after_decision_freeze": support["passed"],
        "future_labels_outcomes_or_pnl_opened": False,
        "resolution_artifact_opened": False,
        "settlement_provider_called": False,
        "prediction_and_decision_freeze_passed": True,
        "all_selected_markets_closed_before_decision_freeze": True,
        "blocking_reason_codes": support["reason_codes"],
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "conformal_v6_future_prediction_freeze_report.json"
    _write_json(report_path, report)
    _write_text(report_path.with_suffix(".md"), _report_markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-manifest-v1",
        "run_id": config.run_id,
        "builder_git_commit": config.builder_git_commit,
        "future_preregistration_manifest": _descriptor(prereg_path),
        "pre_target_access_audit": _descriptor(audit_path),
        "future_window_manifest": _descriptor(Path(window_result["manifest_path"])),
        "opened_raw_feature_artifacts": raw_descriptors,
        "target_free_feature_rows": _descriptor(feature_rows_path),
        "target_free_five_action_rows": _descriptor(action_rows_path),
        "candidate_model": candidate_model,
        "candidate_calibration_artifact": calibration_descriptor,
        "candidate_target_free_predictions": _descriptor(candidate_predictions_path),
        "candidate_outcome_blind_guard_replay": _descriptor(candidate_replay_path),
        "candidate_frozen_accepted_bets": _descriptor(candidate_allowed_path),
        "matched_baseline_model": baseline_model,
        "matched_baseline_target_free_predictions": _descriptor(baseline_predictions_path),
        "matched_baseline_outcome_blind_guard_replay": _descriptor(baseline_replay_path),
        "matched_baseline_frozen_accepted_bets": _descriptor(baseline_allowed_path),
        "accepted_bet_decision_freeze": _descriptor(freeze_path),
        "report": _descriptor(report_path),
        "future_target_free_support_gate_passed": support["passed"],
        "future_target_access_allowed_after_decision_freeze": support["passed"],
        "future_labels_outcomes_or_pnl_opened": False,
        "decision_freeze_written_before_target_access": True,
        "all_selected_markets_closed_before_decision_freeze": True,
        "result_dependent_extension_allowed": False,
        **_blocked_safety_fields(),
    }
    manifest["future_prediction_freeze_manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "conformal_v6_future_prediction_freeze_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "window_result": window_result,
        "decision_freeze": decision_freeze,
        "decision_freeze_path": freeze_path,
        "decision_freeze_sha256": _sha256_file(freeze_path),
        "report": report,
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def _candidate_predictions(
    action_rows: list[dict[str, Any]],
    *,
    model_descriptor: dict[str, str],
    calibration_artifact: dict[str, Any],
    profile: dict[str, Any],
    feature_columns: tuple[str, ...],
) -> list[dict[str, Any]]:
    booster = xgb.Booster()
    booster.load_model(model_descriptor["path"])
    raw = _target_free_predictions(booster, action_rows, feature_columns=feature_columns)
    compatible = attach_frozen_execution_compatibility(raw)
    scored = apply_policy_selected_conformal_scores(
        compatible,
        calibration_artifact=calibration_artifact,
        profile=profile,
    )
    output = []
    for row in scored:
        raw_score = float(row["raw_direct_predicted_net_return"])
        output.append(
            {
                **row,
                "raw_pairwise_rank_score": raw_score,
                "pairwise_group_normalized_rank_score": raw_score,
                "action_advantage_lcb_score_bucket": "not_applicable_policy_selected_conformal",
                "action_advantage_lcb_estimate_source": row["ranking_score_source"],
            }
        )
    if _find_nonempty_fields(output, FORBIDDEN_TARGET_FIELDS):
        raise ValueError("v6 future predictions contain forbidden target fields")
    return output


def _validate_future_preregistration(prereg: dict[str, Any]) -> None:
    blockers = []
    if prereg.get("schema_version") != PREREG_MANIFEST_SCHEMA_VERSION:
        blockers.append("future_preregistration_schema_invalid")
    if prereg.get("candidate_name") != CANDIDATE_NAME:
        blockers.append("future_candidate_name_invalid")
    if prereg.get("future_preregistration_ready") is not True:
        blockers.append("future_preregistration_not_ready")
    if prereg.get("future_labels_outcomes_or_pnl_opened") is not False:
        blockers.append("future_preregistration_target_sealing_invalid")
    if prereg.get("prediction_attempted") is not False:
        blockers.append("future_prediction_attempted_before_freeze")
    if prereg.get("blocking_reason_codes") != []:
        blockers.append("future_preregistration_has_blockers")
    for key, expected in _blocked_safety_fields().items():
        if prereg.get(key) != expected:
            blockers.append(f"future_preregistration_safety_invalid:{key}")
    if blockers:
        raise ValueError("future preregistration invalid: " + ", ".join(blockers))
    for field in (
        "candidate_model",
        "candidate_calibration_artifact",
        "candidate_profile",
        "matched_baseline_model",
        "matched_baseline_fit_profile",
        "collector_protocol",
        "collector_index_prefix",
        "source_boundary_manifest",
    ):
        _verified_descriptor(prereg.get(field), field)


def _validate_append_only_prefix(
    prereg: dict[str, Any],
    *,
    current_index_rows: list[dict[str, Any]],
) -> None:
    prefix_descriptor = _verified_descriptor(prereg["collector_index_prefix"], "index prefix")
    prefix_rows = load_and_validate_persistent_outcome_blind_index(prefix_descriptor["path"])
    expected_count = int(prereg["collector_index_prefix_entry_count"])
    if len(prefix_rows) != expected_count:
        raise ValueError("future preregistered index prefix count mismatch")
    if len(current_index_rows) < expected_count:
        raise ValueError("current collector index is shorter than preregistered prefix")
    if [row["entry_sha256"] for row in current_index_rows[:expected_count]] != [
        row["entry_sha256"] for row in prefix_rows
    ]:
        raise ValueError("collector index prefix changed after future preregistration")
    if prereg["collector_index_prefix_last_entry_sha256"] != prefix_rows[-1]["entry_sha256"]:
        raise ValueError("future preregistered last index entry mismatch")
    if int(prereg["minimum_collection_index_sequence"]) != expected_count + 1:
        raise ValueError("future minimum index sequence is inconsistent")


def _validate_selected_future_rows(
    prereg: dict[str, Any],
    *,
    selected_rows: list[dict[str, Any]],
) -> None:
    target = int(prereg["target_quality_valid_market_count"])
    minimum_sequence = int(prereg["minimum_collection_index_sequence"])
    minimum_ts = int(prereg["minimum_collection_decision_ts"])
    if len(selected_rows) != target:
        raise ValueError("future window market count does not match preregistration")
    if len({str(row["market_id"]) for row in selected_rows}) != target:
        raise ValueError("future window market identities are not unique")
    if min(int(row["sequence"]) for row in selected_rows) < minimum_sequence:
        raise ValueError("future window crosses preregistered index boundary")
    if min(int(row["scheduled_round_start_ts"]) for row in selected_rows) < minimum_ts:
        raise ValueError("future window crosses preregistered timestamp boundary")
    if any(row.get("capture_quality_valid") is not True for row in selected_rows):
        raise ValueError("future window contains quality-invalid market")
    if any(row.get("labels_outcomes_or_pnl_opened") is not False for row in selected_rows):
        raise ValueError("future window contains opened target state")


def _future_support_summary(
    allowed_rows: list[dict[str, Any]],
    *,
    prereg: dict[str, Any],
) -> dict[str, Any]:
    market_ids = {str(row["market_id"]) for row in allowed_rows}
    side_markets = {
        side: {str(row["market_id"]) for row in allowed_rows if row.get("selected_side") == side}
        for side in prereg["required_supported_sides"]
    }
    required_total = int(prereg["minimum_guard_accepted_unique_market_count"])
    required_side = int(prereg["minimum_supported_side_market_count"])
    checks = {
        "minimum_guard_accepted_unique_market_count": len(market_ids) >= required_total,
        "minimum_supported_side_market_count": all(
            len(side_markets[side]) >= required_side for side in side_markets
        ),
        "accepted_market_identity_unique": len(market_ids) == len(allowed_rows),
        "outcome_free_support_check": not _find_nonempty_fields(
            allowed_rows, FORBIDDEN_TARGET_FIELDS
        ),
    }
    reason_map = {
        "minimum_guard_accepted_unique_market_count": "insufficient_future_accepted_market_support",
        "minimum_supported_side_market_count": "insufficient_future_side_support",
        "accepted_market_identity_unique": "duplicate_future_accepted_market",
        "outcome_free_support_check": "future_support_rows_contain_forbidden_targets",
    }
    reasons = [reason_map[name] for name, passed in checks.items() if not passed]
    return {
        "accepted_bet_count": len(allowed_rows),
        "accepted_unique_market_count": len(market_ids),
        "accepted_side_market_counts": {
            side: len(side_markets[side]) for side in sorted(side_markets)
        },
        "minimum_required_unique_markets": required_total,
        "minimum_required_markets_per_side": required_side,
        "checks": checks,
        "passed": not reasons,
        "reason_codes": reasons,
        "labels_outcomes_or_pnl_opened": False,
    }


def _write_incomplete_prediction_result(
    *,
    config: PolicySelectedConformalV6FuturePredictionConfig,
    run_dir: Path,
    prereg_path: Path,
    audit_path: Path,
    window_result: dict[str, Any],
) -> dict[str, Any]:
    reasons = ["future_window_not_ready_before_frozen_scan_cap"]
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-report-v1",
        "report_id": None,
        "run_id": config.run_id,
        "selected_market_count": window_result["report"]["selected_market_count"],
        "future_window_freeze_ready": False,
        "prediction_attempted": False,
        "future_target_free_support_gate_passed": False,
        "future_target_access_allowed_after_decision_freeze": False,
        "future_labels_outcomes_or_pnl_opened": False,
        "resolution_artifact_opened": False,
        "settlement_provider_called": False,
        "blocking_reason_codes": reasons,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "conformal_v6_future_prediction_freeze_report.json"
    _write_json(report_path, report)
    _write_text(report_path.with_suffix(".md"), _report_markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-manifest-v1",
        "run_id": config.run_id,
        "future_preregistration_manifest": _descriptor(prereg_path),
        "pre_target_access_audit": _descriptor(audit_path),
        "future_window_manifest": _descriptor(Path(window_result["manifest_path"])),
        "report": _descriptor(report_path),
        "future_window_freeze_ready": False,
        "prediction_attempted": False,
        "future_target_access_allowed_after_decision_freeze": False,
        "future_labels_outcomes_or_pnl_opened": False,
        "blocking_reason_codes": reasons,
        **_blocked_safety_fields(),
    }
    manifest["future_prediction_freeze_manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "conformal_v6_future_prediction_freeze_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "window_result": window_result,
        "report": report,
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def _report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# #207 v6 future target-free prediction freeze",
        "",
        f"- selected markets: `{report['selected_market_count']}`",
        f"- prediction attempted: `{report.get('prediction_attempted', True)}`",
        f"- target-free support gate: `{report['future_target_free_support_gate_passed']}`",
        "- labels/outcomes/PnL opened: `false`",
        "- result-dependent extension: `false`",
        "- paper/live/promotion unlock: `false`",
    ]
    if "candidate_guard_accepted_unique_market_count" in report:
        lines.extend(
            [
                f"- candidate accepted markets: `{report['candidate_guard_accepted_unique_market_count']}`",
                f"- candidate sides: `{report['candidate_guard_accepted_side_distribution']}`",
            ]
        )
    lines.extend(["", f"Blocking reasons: `{report['blocking_reason_codes']}`", ""])
    return "\n".join(lines)
