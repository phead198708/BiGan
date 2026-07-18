"""Fit the one-shot #202 guard-compatible direct net-return v4 candidate."""

from __future__ import annotations

import json
import math
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_direct_advantage_estimand_audit import (
    _market_bootstrap_interval,
)
from bigan.v8.polymarket.training.execution_layer_v2_outcome_blind_acceptance_viability import (
    _outcome_blind_acceptance_replay,
)
from bigan.v8.polymarket.training.execution_layer_v2_p_up_aligned_action_value_support import (
    _claimed_descriptor,
    build_execution_compatible_action_universe,
)
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb import (
    validate_pairwise_action_advantage_lcb_feature_contract,
)
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb_fit import (
    _descriptor,
    _find_fields,
    _load_json,
    _load_jsonl,
    _materialize_role_action_rows,
    _require_sha256,
    _sha256_file,
    _verified_descriptor,
)
from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_action_value import (
    REQUIRED_ACTIONS,
)
from bigan.v8.polymarket.training.post_freeze_o_replay_aligned_source_ranking import (
    _v8_execution_guard_config,
)

PROFILE_SCHEMA_VERSION = "bigan-v8-guard-compatible-direct-net-return-v4-fit-profile-v2"
SCHEMA_PREFIX = "bigan-v8-guard-compatible-direct-net-return-v4"
CANDIDATE_NAME = "guard_compatible_direct_net_return_v4"
FIT_ROLE = "development_train"
MODEL_FILENAME = "guard_compatible_direct_net_return_v4.xgb.json"
TARGET_FIELDS = frozenset(
    {
        "target_net_pnl_per_contract",
        "target_resolved_outcome",
        "target_cost_components",
        "outcome",
        "resolved_outcome",
        "settlement_outcome",
        "net_pnl",
        "realized_pnl",
        "oracle_action",
        "future_return",
    }
)


@dataclass(frozen=True, slots=True)
class GuardCompatibleDirectNetReturnV4Config:
    """Pinned inputs for the one-shot #202 fit."""

    run_id: str
    output_dir: Path | str
    fit_profile_path: Path | str
    expected_fit_profile_sha256: str
    issue198_candidate_manifest_path: Path | str
    expected_issue198_candidate_manifest_sha256: str
    issue201_manifest_path: Path | str
    expected_issue201_manifest_sha256: str
    role_assignment_manifest_path: Path | str
    expected_role_assignment_manifest_sha256: str
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for name in (
            "expected_fit_profile_sha256",
            "expected_issue198_candidate_manifest_sha256",
            "expected_issue201_manifest_sha256",
            "expected_role_assignment_manifest_sha256",
        ):
            _require_sha256(str(getattr(self, name)), name=name)
        for name in (
            "output_dir",
            "fit_profile_path",
            "issue198_candidate_manifest_path",
            "issue201_manifest_path",
            "role_assignment_manifest_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


def validate_guard_compatible_direct_net_return_v4_profile(
    profile: dict[str, Any],
) -> None:
    """Reject any drift from the one-shot #202 research contract."""

    model = dict(profile.get("model") or {})
    oof = dict(profile.get("chronological_oof") or {})
    decision = dict(profile.get("decision_rule") or {})
    gate = dict(profile.get("development_gate") or {})
    access = dict(profile.get("access_sequence") or {})
    output = dict(profile.get("output_contract") or {})
    hashes = (
        "parent_issue_201_manifest_sha256",
        "parent_issue_198_candidate_manifest_sha256",
        "role_assignment_manifest_sha256",
        "role_assignment_rows_sha256",
        "feature_contract_sha256",
        "development_train_target_rows_sha256",
        "execution_guard_config_sha256",
    )
    checks = {
        "schema": profile.get("schema_version") == PROFILE_SCHEMA_VERSION,
        "candidate": profile.get("candidate_name") == CANDIDATE_NAME,
        "frozen": profile.get("frozen") is True,
        "parent_commit": _is_sha1(str(profile.get("parent_issue_201_commit") or "")),
        "hashes": all(_is_sha256(str(profile.get(name) or "")) for name in hashes),
        "scope": profile.get("fit_role") == FIT_ROLE
        and profile.get("required_fit_market_count") == 90
        and profile.get("required_actions") == list(REQUIRED_ACTIONS),
        "fixed_model": model.get("objective") == "reg:squarederror"
        and model.get("eval_metric") == "rmse"
        and model.get("target") == "target_net_pnl_per_contract"
        and model.get("num_boost_round") == 120
        and model.get("max_depth") == 3
        and math.isclose(float(model.get("eta") or 0.0), 0.03)
        and math.isclose(float(model.get("reg_alpha") or 0.0), 1.0)
        and math.isclose(float(model.get("reg_lambda") or 0.0), 10.0)
        and model.get("nthread") == 1
        and model.get("hyperparameter_search_enabled") is False,
        "chronological_oof": oof.get("fold_count") == 5
        and oof.get("initial_training_market_count") == 15
        and oof.get("validation_market_count_per_fold") == 15
        and oof.get("expected_oof_market_count") == 75
        and oof.get("fold_assignment") == "chronological_expanding_window_prior_markets_only"
        and oof.get("future_market_labels_excluded_from_each_fold") is True,
        "decision": decision.get("method")
        == "guard_compatible_mask_before_direct_net_return_argmax"
        and decision.get("no_trade_score") == 0.0
        and decision.get("minimum_selected_predicted_net_return_exclusive") == 0.0
        and decision.get("minimum_runner_up_margin_exclusive") == 0.0
        and decision.get("p_up_side_alignment_required") is True
        and decision.get("frozen_execution_quality_required") is True
        and decision.get("mask_score") == -1_000_000.0
        and all(
            decision.get(name) is False
            for name in (
                "model_score_mutation_allowed",
                "execution_guard_mutation_allowed",
                "cost_model_mutation_allowed",
                "order_sizing_mutation_allowed",
                "exposure_policy_mutation_allowed",
            )
        ),
        "gate": gate.get("minimum_guard_accepted_bet_count") == 10
        and gate.get("minimum_guard_accepted_unique_market_count") == 10
        and gate.get("pnl_hard_gate_aggregation") == "selected_side_buy_up_buy_down_only"
        and gate.get("minimum_supported_side_count") == 2
        and gate.get("minimum_side_support_for_side_pnl_gate") == 5
        and gate.get("action_and_action_family_pnl_diagnostic_only") is True
        and gate.get("accepted_bet_total_pnl_minimum_exclusive") == 0.0
        and gate.get("all_oof_market_policy_pnl_lcb_minimum_exclusive") == 0.0
        and gate.get("bootstrap_unit") == "market_id"
        and gate.get("bootstrap_resample_count") == 2000
        and gate.get("bootstrap_confidence_level") == 0.95,
        "access": access.get("pre_label_audit_required") is True
        and access.get("development_train_labels_may_be_opened_after_audit") is True
        and access.get("target_stripped_oof_decision_freeze_before_oof_evaluation_required") is True
        and all(
            access.get(name) is False
            for name in (
                "development_calibration_files_may_be_opened",
                "confirmatory_files_may_be_opened",
                "issue_190_or_192_future_files_may_be_opened",
                "future_accepted_bet_pnl_may_be_opened",
            )
        ),
        "output": output.get("research_candidate_only") is True
        and output.get("candidate_specific_future_evaluation_requires_development_gate") is True
        and output.get("issue_190_collection_started_before_candidate_freeze_and_is_not_eligible")
        is True
        and output.get("strictly_later_persistent_window_required") is True
        and output.get("promotion_evidence_created") is False,
        "safety": dict(profile.get("safety") or {}) == _blocked_safety_fields(),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("invalid #202 fit profile: " + ", ".join(failed))


def fit_guard_compatible_direct_net_return_v4(
    config: GuardCompatibleDirectNetReturnV4Config,
) -> dict[str, Any]:
    """Run the fixed chronological OOF fit once and remain fail closed."""

    paths = {
        "profile": config.fit_profile_path.resolve(),
        "issue198_candidate": config.issue198_candidate_manifest_path.resolve(),
        "issue201_manifest": config.issue201_manifest_path.resolve(),
        "role_assignment_manifest": config.role_assignment_manifest_path.resolve(),
    }
    expected = {
        "profile": config.expected_fit_profile_sha256,
        "issue198_candidate": config.expected_issue198_candidate_manifest_sha256,
        "issue201_manifest": config.expected_issue201_manifest_sha256,
        "role_assignment_manifest": config.expected_role_assignment_manifest_sha256,
    }
    for name, path in paths.items():
        _verify_file_hash(path, expected[name], name=name)
    profile = {
        **_load_json(paths["profile"]),
        "fit_profile_sha256": expected["profile"],
    }
    validate_guard_compatible_direct_net_return_v4_profile(profile)
    if profile["parent_issue_198_candidate_manifest_sha256"] != expected["issue198_candidate"]:
        raise ValueError("#198 candidate lineage mismatch")
    if profile["parent_issue_201_manifest_sha256"] != expected["issue201_manifest"]:
        raise ValueError("#201 audit lineage mismatch")
    if profile["role_assignment_manifest_sha256"] != expected["role_assignment_manifest"]:
        raise ValueError("role assignment lineage mismatch")

    candidate = _load_json(paths["issue198_candidate"])
    issue201_manifest = _load_json(paths["issue201_manifest"])
    role_manifest = _load_json(paths["role_assignment_manifest"])
    lineage = _validate_lineage(
        candidate=candidate,
        issue201_manifest=issue201_manifest,
        role_manifest=role_manifest,
        profile=profile,
    )
    run_dir = Path(config.output_dir) / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    pre_label = {
        "schema_version": f"{SCHEMA_PREFIX}-pre-label-access-audit-v1",
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "fit_profile": _descriptor(paths["profile"]),
        "issue198_candidate_manifest": _descriptor(paths["issue198_candidate"]),
        "issue201_summary_manifest": _descriptor(paths["issue201_manifest"]),
        "role_assignment_manifest": _descriptor(paths["role_assignment_manifest"]),
        "intended_development_train_target_rows": lineage["target_rows_claim"],
        "parent_research_summary_metadata_opened": True,
        "feature_label_resolution_or_pnl_files_opened_before_audit": False,
        "target_rows_hash_verified_before_audit": False,
        "development_calibration_confirmatory_or_future_files_opened": False,
        "profile_and_model_parameters_frozen_before_target_access": True,
        "pre_label_access_validation_passed": True,
        **_blocked_safety_fields(),
    }
    pre_label["audit_id"] = canonical_json_sha256(pre_label)
    pre_label_path = run_dir / "pre_label_access_lineage_audit.json"
    _write_json_fsync(pre_label_path, pre_label)
    _write_text_fsync(
        run_dir / "pre_label_access_lineage_audit.md",
        _pre_label_markdown(pre_label),
    )

    target_path = Path(lineage["target_rows_claim"]["path"])
    _verify_file_hash(
        target_path,
        profile["development_train_target_rows_sha256"],
        name="development_train target rows after pre-label audit",
    )
    action_rows_by_role, corpus_audits = _materialize_role_action_rows(
        lineage["role_rows"],
        feature_columns=lineage["feature_columns"],
        roles=(FIT_ROLE,),
    )
    train_rows = action_rows_by_role[FIT_ROLE]
    _validate_train_rows(train_rows, corpus_audits=corpus_audits)
    train_rows_path = run_dir / "guard_compatible_v4_development_train_action_rows.jsonl"
    _write_jsonl_fsync(train_rows_path, train_rows)
    if _sha256_file(train_rows_path) != profile["development_train_target_rows_sha256"]:
        raise ValueError("materialized development_train target rows hash drifted")

    decision_rows = sorted(
        (_strip_target_fields(row) for row in train_rows),
        key=_row_sort_key,
    )
    compatibility_rows = build_execution_compatible_action_universe(decision_rows)
    compatibility = {
        _row_key(row): bool(row["p_up_alignment_passed"] and row["execution_quality_only_passed"])
        for row in compatibility_rows
    }
    cross_fit = _chronological_oof_predictions(
        train_rows,
        feature_columns=lineage["feature_columns"],
        profile=profile,
        compatibility=compatibility,
    )
    oof_rows = cross_fit.pop("target_stripped_oof_predictions")
    _validate_target_stripped_prediction_rows(oof_rows, profile=profile)
    oof_path = run_dir / "guard_compatible_v4_target_stripped_oof_predictions.jsonl"
    _write_jsonl_fsync(oof_path, oof_rows)
    oof_coverage = {
        "schema_version": f"{SCHEMA_PREFIX}-chronological-oof-coverage-v1",
        "run_id": config.run_id,
        **cross_fit,
        "target_fields_in_prediction_rows": [],
        "development_calibration_confirmatory_or_future_files_opened": False,
        **_blocked_safety_fields(),
    }
    oof_coverage["report_id"] = canonical_json_sha256(oof_coverage)
    oof_coverage_path = run_dir / "guard_compatible_v4_chronological_oof_coverage_report.json"
    _write_json_fsync(oof_coverage_path, oof_coverage)

    replay_rows = _outcome_blind_acceptance_replay(
        oof_rows,
        entry_threshold=0.0,
        runner_up_advantage_threshold=0.0,
    )
    replay_path = run_dir / "guard_compatible_v4_outcome_blind_guard_replay.jsonl"
    _write_jsonl_fsync(replay_path, replay_rows)
    decision_freeze = _build_decision_freeze(
        run_id=config.run_id,
        profile=profile,
        oof_path=oof_path,
        replay_path=replay_path,
        oof_rows=oof_rows,
        replay_rows=replay_rows,
    )
    decision_freeze_path = run_dir / "guard_compatible_v4_oof_decision_freeze.json"
    _write_json_fsync(decision_freeze_path, decision_freeze)
    decision_freeze_sha256 = _sha256_file(decision_freeze_path)

    evaluation_rows = _build_evaluation_rows(replay_rows, target_rows=train_rows)
    evaluation_path = run_dir / "guard_compatible_v4_oof_evaluation_rows.jsonl"
    _write_jsonl_fsync(evaluation_path, evaluation_rows)
    gate_report = _build_gate_report(
        run_id=config.run_id,
        profile=profile,
        decision_freeze_sha256=decision_freeze_sha256,
        evaluation_rows=evaluation_rows,
        oof_rows=oof_rows,
        train_rows=train_rows,
        corpus_audits=corpus_audits,
    )
    gate_report_path = run_dir / "guard_compatible_v4_development_gate_report.json"
    _write_json_fsync(gate_report_path, gate_report)
    _write_text_fsync(
        run_dir / "guard_compatible_v4_development_gate_report.md",
        _gate_markdown(gate_report),
    )

    booster = _train_regressor(
        train_rows,
        feature_columns=lineage["feature_columns"],
        model_config=dict(profile["model"]),
    )
    model_path = run_dir / MODEL_FILENAME
    booster.save_model(model_path)
    final_predictions = _predict_regressor(
        booster,
        decision_rows,
        feature_columns=lineage["feature_columns"],
    )
    final_target_stripped = _attach_predictions_and_mask(
        decision_rows,
        final_predictions,
        compatibility=compatibility,
        profile=profile,
        fold_index=None,
    )
    final_path = run_dir / "guard_compatible_v4_final_target_stripped_predictions.jsonl"
    _write_jsonl_fsync(final_path, final_target_stripped)

    training_report = {
        "schema_version": f"{SCHEMA_PREFIX}-training-report-v1",
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "fit_market_count": 90,
        "fit_action_row_count": len(train_rows),
        "decision_group_count": len(train_rows) // len(REQUIRED_ACTIONS),
        "chronological_oof_market_count": oof_coverage["oof_market_count"],
        "chronological_oof_prediction_count": oof_coverage["oof_prediction_count"],
        "model_objective": profile["model"]["objective"],
        "model_target": profile["model"]["target"],
        "training_target_includes_costs": True,
        "hyperparameter_search_enabled": False,
        "guard_compatibility_mask_applied_before_action_argmax": True,
        "development_gate_passed": gate_report["development_gate_passed"],
        "candidate_specific_future_evaluation_allowed": gate_report[
            "candidate_specific_future_evaluation_allowed"
        ],
        "model": _descriptor(model_path),
        "development_gate_report": _descriptor(gate_report_path),
        "development_calibration_confirmatory_or_future_files_opened": False,
        **_blocked_safety_fields(),
    }
    training_report["report_id"] = canonical_json_sha256(training_report)
    training_report_path = run_dir / "guard_compatible_v4_training_report.json"
    _write_json_fsync(training_report_path, training_report)

    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-research-candidate-manifest-v1",
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "research_candidate_frozen": True,
        "research_candidate_only": True,
        "fit_profile": _descriptor(paths["profile"]),
        "pre_label_access_audit": _descriptor(pre_label_path),
        "development_train_action_rows": _descriptor(train_rows_path),
        "chronological_oof_coverage_report": _descriptor(oof_coverage_path),
        "target_stripped_oof_predictions": _descriptor(oof_path),
        "outcome_blind_guard_replay": _descriptor(replay_path),
        "oof_decision_freeze": _descriptor(decision_freeze_path),
        "oof_evaluation_rows": _descriptor(evaluation_path),
        "development_gate_report": _descriptor(gate_report_path),
        "model": _descriptor(model_path),
        "final_target_stripped_predictions": _descriptor(final_path),
        "training_report": _descriptor(training_report_path),
        "development_gate_passed": gate_report["development_gate_passed"],
        "candidate_specific_future_evaluation_allowed": gate_report[
            "candidate_specific_future_evaluation_allowed"
        ],
        "candidate_specific_future_evaluation_blocking_reason_codes": gate_report[
            "gate_blocking_reason_codes"
        ],
        "issue_190_collection_eligible_for_this_candidate": False,
        "strictly_later_persistent_window_required": True,
        "development_calibration_files_opened": False,
        "confirmatory_files_opened": False,
        "issue_190_or_192_future_files_opened": False,
        "current_oof_validation_or_future_pnl_used_for_tuning": False,
        "result_driven_rerun_or_parameter_change_allowed": False,
        "guard_cost_threshold_sizing_or_exposure_mutated": False,
        **_blocked_safety_fields(),
    }
    manifest["research_candidate_hash"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "guard_compatible_v4_research_candidate_manifest.json"
    _write_json_fsync(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "gate_report": gate_report,
        "oof_coverage": oof_coverage,
    }


def _chronological_oof_predictions(
    rows: list[dict[str, Any]],
    *,
    feature_columns: tuple[str, ...],
    profile: dict[str, Any],
    compatibility: dict[tuple[str, int, str], bool],
) -> dict[str, Any]:
    first_ts: dict[str, int] = {}
    for row in rows:
        market_id = str(row["market_id"])
        first_ts[market_id] = min(
            first_ts.get(market_id, int(row["decision_ts"])), int(row["decision_ts"])
        )
    markets = sorted(first_ts, key=lambda value: (first_ts[value], value))
    oof_config = dict(profile["chronological_oof"])
    warmup = int(oof_config["initial_training_market_count"])
    fold_size = int(oof_config["validation_market_count_per_fold"])
    fold_count = int(oof_config["fold_count"])
    if len(markets) != 90 or warmup + fold_size * fold_count != 90:
        raise ValueError("chronological OOF market coverage mismatch")
    predictions: list[dict[str, Any]] = []
    fold_reports = []
    for fold_index in range(1, fold_count + 1):
        validation_start = warmup + (fold_index - 1) * fold_size
        fit_markets = markets[:validation_start]
        validation_markets = markets[validation_start : validation_start + fold_size]
        fit_set = set(fit_markets)
        validation_set = set(validation_markets)
        fit_rows = [row for row in rows if str(row["market_id"]) in fit_set]
        validation_rows = [row for row in rows if str(row["market_id"]) in validation_set]
        train_max_ts = max(int(row["decision_ts"]) for row in fit_rows)
        validation_min_ts = min(int(row["decision_ts"]) for row in validation_rows)
        if train_max_ts >= validation_min_ts:
            raise ValueError("OOF training does not strictly precede validation")
        booster = _train_regressor(
            fit_rows,
            feature_columns=feature_columns,
            model_config=dict(profile["model"]),
        )
        decision_rows = sorted(
            (_strip_target_fields(row) for row in validation_rows),
            key=_row_sort_key,
        )
        values = _predict_regressor(booster, decision_rows, feature_columns=feature_columns)
        predictions.extend(
            _attach_predictions_and_mask(
                decision_rows,
                values,
                compatibility=compatibility,
                profile=profile,
                fold_index=fold_index,
            )
        )
        fold_reports.append(
            {
                "fold_index": fold_index,
                "training_market_count": len(fit_markets),
                "validation_market_count": len(validation_markets),
                "training_market_ids_sha256": canonical_json_sha256(fit_markets),
                "validation_market_ids_sha256": canonical_json_sha256(validation_markets),
                "training_max_decision_ts": train_max_ts,
                "validation_min_decision_ts": validation_min_ts,
                "training_strictly_precedes_validation": True,
                "market_overlap_count": len(fit_set & validation_set),
                "future_market_label_access_count": 0,
            }
        )
    predictions.sort(key=_row_sort_key)
    return {
        "method": "five_fold_chronological_expanding_window_direct_regression",
        "fold_count": fold_count,
        "initial_training_market_count": warmup,
        "oof_market_count": len({row["market_id"] for row in predictions}),
        "oof_decision_group_count": len(
            {(row["market_id"], row["decision_ts"]) for row in predictions}
        ),
        "oof_prediction_count": len(predictions),
        "expected_oof_prediction_count": 75 * 4 * len(REQUIRED_ACTIONS),
        "feature_causality_violation_count": sum(
            int(row["max_input_ts"]) > int(row["decision_ts"]) for row in predictions
        ),
        "fold_reports": fold_reports,
        "uses_development_calibration_labels": False,
        "uses_confirmatory_labels": False,
        "uses_issue_190_or_192_future_evidence": False,
        "target_stripped_oof_predictions": predictions,
    }


def _train_regressor(
    rows: list[dict[str, Any]],
    *,
    feature_columns: tuple[str, ...],
    model_config: dict[str, Any],
) -> xgb.Booster:
    ordered = sorted(rows, key=_row_sort_key)
    matrix = xgb.DMatrix(
        np.asarray(
            [
                [float(row["decision_time_features"][name]) for name in feature_columns]
                for row in ordered
            ],
            dtype=np.float32,
        ),
        label=np.asarray(
            [float(row["target_net_pnl_per_contract"]) for row in ordered],
            dtype=np.float32,
        ),
        feature_names=list(feature_columns),
    )
    parameters = {
        key: value
        for key, value in model_config.items()
        if key not in {"target", "num_boost_round", "hyperparameter_search_enabled"}
    }
    return xgb.train(
        parameters,
        matrix,
        num_boost_round=int(model_config["num_boost_round"]),
    )


def _predict_regressor(
    booster: xgb.Booster,
    rows: list[dict[str, Any]],
    *,
    feature_columns: tuple[str, ...],
) -> list[float]:
    ordered = sorted(rows, key=_row_sort_key)
    if ordered != rows:
        raise ValueError("prediction rows must be deterministically sorted")
    matrix = xgb.DMatrix(
        np.asarray(
            [
                [float(row["decision_time_features"][name]) for name in feature_columns]
                for row in rows
            ],
            dtype=np.float32,
        ),
        feature_names=list(feature_columns),
    )
    values = [float(value) for value in booster.predict(matrix)]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("model predictions must be finite")
    return values


def _attach_predictions_and_mask(
    rows: list[dict[str, Any]],
    predictions: list[float],
    *,
    compatibility: dict[tuple[str, int, str], bool],
    profile: dict[str, Any],
    fold_index: int | None,
) -> list[dict[str, Any]]:
    mask_score = float(profile["decision_rule"]["mask_score"])
    output = []
    for row, prediction in zip(rows, predictions, strict=True):
        action = str(row["action"])
        compatible = action == "NO_TRADE" or compatibility[_row_key(row)]
        if action == "NO_TRADE":
            selection_score = 0.0
            source = "frozen_no_trade_zero_anchor"
        elif compatible:
            selection_score = prediction
            source = "direct_predicted_net_return_guard_compatible"
        else:
            selection_score = mask_score
            source = "masked_by_frozen_execution_compatibility"
        updated = {
            **row,
            "fold_index": fold_index,
            "direct_predicted_net_return": prediction,
            "guard_compatible_before_ranking": compatible,
            "guard_compatibility_mask_applied_before_argmax": True,
            "action_selection_score": selection_score,
            "calibrated_action_expected_net_return": prediction,
            "action_advantage_lcb_net_return": selection_score,
            "action_advantage_lcb_score_bucket": "not_applicable_direct_regression",
            "action_advantage_lcb_estimate_source": source,
            "raw_pairwise_rank_score": prediction,
            "pairwise_group_normalized_rank_score": prediction,
            "target_used_as_decision_input": False,
            "outcome_fields_used_as_decision_input": False,
            "target_or_outcome_fields_used": False,
        }
        updated["v4_prediction_row_sha256"] = canonical_json_sha256(updated)
        output.append(updated)
    return output


def _build_decision_freeze(
    *,
    run_id: str,
    profile: dict[str, Any],
    oof_path: Path,
    replay_path: Path,
    oof_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_trade = [row for row in replay_rows if row["source_selected_action"] != "NO_TRADE"]
    return {
        "schema_version": f"{SCHEMA_PREFIX}-oof-decision-freeze-v1",
        "run_id": run_id,
        "candidate_name": CANDIDATE_NAME,
        "fit_profile_sha256": profile["fit_profile_sha256"],
        "target_stripped_oof_predictions": _descriptor(oof_path),
        "outcome_blind_guard_replay": _descriptor(replay_path),
        "oof_prediction_count": len(oof_rows),
        "decision_count": len(replay_rows),
        "selected_trade_decision_count": len(selected_trade),
        "selected_trade_p_up_disagreement_count": sum(
            row["p_up_action_disagreement"] for row in selected_trade
        ),
        "guard_accepted_bet_count": sum(
            row["execution_guard_order_allowed"] for row in replay_rows
        ),
        "target_fields_present": [],
        "target_or_outcome_used_for_decision": False,
        "decision_freeze_written_before_oof_target_evaluation": True,
        **_blocked_safety_fields(),
    }


def _build_evaluation_rows(
    replay_rows: list[dict[str, Any]],
    *,
    target_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    targets = {_row_key(row): float(row["target_net_pnl_per_contract"]) for row in target_rows}
    output = []
    for row in replay_rows:
        action = str(row["executed_action"])
        key = (str(row["market_id"]), int(row["decision_ts"]), action)
        target = targets[key]
        accepted = bool(row["execution_guard_order_allowed"])
        size = float(row["proposed_order_size"]) if accepted else 0.0
        evaluation = {
            **row,
            "evaluation_target_net_pnl_per_contract": target,
            "accepted_bet_net_pnl": size * target,
            "target_joined_after_decision_freeze": True,
            "target_used_as_decision_input": False,
        }
        evaluation["evaluation_row_sha256"] = canonical_json_sha256(evaluation)
        output.append(evaluation)
    return output


def _build_gate_report(
    *,
    run_id: str,
    profile: dict[str, Any],
    decision_freeze_sha256: str,
    evaluation_rows: list[dict[str, Any]],
    oof_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    corpus_audits: list[dict[str, Any]],
) -> dict[str, Any]:
    accepted = [row for row in evaluation_rows if row["execution_guard_order_allowed"]]
    accepted_markets = {str(row["market_id"]) for row in accepted}
    oof_markets = sorted({str(row["market_id"]) for row in evaluation_rows})
    pnl_by_market = dict.fromkeys(oof_markets, 0.0)
    for row in accepted:
        pnl_by_market[str(row["market_id"])] += float(row["accepted_bet_net_pnl"])
    gate = dict(profile["development_gate"])
    interval = _market_bootstrap_interval(
        list(pnl_by_market.values()),
        resample_count=int(gate["bootstrap_resample_count"]),
        confidence_level=float(gate["bootstrap_confidence_level"]),
        seed=int(gate["bootstrap_seed"]),
    )
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        by_action[str(row["executed_action"])].append(row)
    action_metrics = {
        action: {
            "accepted_bet_count": len(rows),
            "accepted_unique_market_count": len({row["market_id"] for row in rows}),
            "accepted_bet_net_pnl_sum": float(
                sum(float(row["accepted_bet_net_pnl"]) for row in rows)
            ),
            "diagnostic_only": True,
        }
        for action, rows in sorted(by_action.items())
    }
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        by_family[_family_for_action(str(row["executed_action"]))].append(row)
    family_metrics = {
        family: {
            "accepted_bet_count": len(rows),
            "accepted_unique_market_count": len({row["market_id"] for row in rows}),
            "accepted_bet_net_pnl_sum": float(
                sum(float(row["accepted_bet_net_pnl"]) for row in rows)
            ),
            "diagnostic_only": True,
        }
        for family, rows in sorted(by_family.items())
    }
    by_side: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        by_side[str(row["selected_side"])].append(row)
    side_metrics = {
        side: {
            "accepted_bet_count": len(rows),
            "accepted_unique_market_count": len({row["market_id"] for row in rows}),
            "accepted_bet_net_pnl_sum": float(
                sum(float(row["accepted_bet_net_pnl"]) for row in rows)
            ),
            "supported_side": len({row["market_id"] for row in rows})
            >= int(gate["minimum_side_support_for_side_pnl_gate"]),
        }
        for side, rows in sorted(by_side.items())
        if side in {"UP", "DOWN"}
    }
    supported_sides = [metrics for metrics in side_metrics.values() if metrics["supported_side"]]
    supported_side_gate = len(supported_sides) >= int(gate["minimum_supported_side_count"])
    supported_side_gate = supported_side_gate and all(
        metrics["accepted_bet_net_pnl_sum"] > 0.0 for metrics in supported_sides
    )
    accepted_pnl_sum = float(sum(float(row["accepted_bet_net_pnl"]) for row in accepted))
    selected_trade = [row for row in evaluation_rows if row["source_selected_action"] != "NO_TRADE"]
    checks = {
        "minimum_guard_accepted_bet_support": len(accepted)
        >= int(gate["minimum_guard_accepted_bet_count"]),
        "minimum_guard_accepted_unique_market_support": len(accepted_markets)
        >= int(gate["minimum_guard_accepted_unique_market_count"]),
        "accepted_bet_total_pnl_positive": accepted_pnl_sum
        > float(gate["accepted_bet_total_pnl_minimum_exclusive"]),
        "all_oof_market_policy_pnl_lcb_positive": interval["lower_confidence_bound"]
        > float(gate["all_oof_market_policy_pnl_lcb_minimum_exclusive"]),
        "supported_side_pnl_gate": supported_side_gate,
        "selected_trade_p_up_alignment": all(
            row["p_up_action_disagreement"] is False for row in selected_trade
        ),
        "accepted_trade_p_up_alignment": all(
            row["p_up_action_disagreement"] is False for row in accepted
        ),
        "feature_causality": all(
            int(row["max_input_ts"]) <= int(row["decision_ts"]) for row in oof_rows
        ),
        "corpus_integrity": all(not audit["blocking_reason_codes"] for audit in corpus_audits),
    }
    reason_map = {
        "minimum_guard_accepted_bet_support": "insufficient_guard_accepted_bet_support",
        "minimum_guard_accepted_unique_market_support": "insufficient_guard_accepted_unique_market_support",
        "accepted_bet_total_pnl_positive": "accepted_bet_total_pnl_not_positive",
        "all_oof_market_policy_pnl_lcb_positive": "all_oof_market_policy_pnl_lcb_not_positive",
        "supported_side_pnl_gate": "supported_side_pnl_gate_failed",
        "selected_trade_p_up_alignment": "selected_trade_p_up_disagreement_present",
        "accepted_trade_p_up_alignment": "accepted_trade_p_up_disagreement_present",
        "feature_causality": "feature_causality_violation",
        "corpus_integrity": "corpus_integrity_failed",
    }
    blockers = [reason_map[name] for name, passed in checks.items() if not passed]
    gate_passed = not blockers
    target_by_key = {_row_key(row): float(row["target_net_pnl_per_contract"]) for row in train_rows}
    prediction_errors = [
        float(row["direct_predicted_net_return"]) - target_by_key[_row_key(row)] for row in oof_rows
    ]
    return {
        "schema_version": f"{SCHEMA_PREFIX}-development-gate-report-v1",
        "run_id": run_id,
        "candidate_name": CANDIDATE_NAME,
        "oof_decision_freeze_sha256": decision_freeze_sha256,
        "oof_market_count": len(oof_markets),
        "oof_decision_count": len(evaluation_rows),
        "selected_trade_decision_count": len(selected_trade),
        "selected_trade_p_up_disagreement_count": sum(
            row["p_up_action_disagreement"] for row in selected_trade
        ),
        "guard_accepted_bet_count": len(accepted),
        "guard_accepted_unique_market_count": len(accepted_markets),
        "accepted_action_distribution": dict(
            sorted(Counter(row["executed_action"] for row in accepted).items())
        ),
        "accepted_side_distribution": dict(
            sorted(Counter(row["selected_side"] for row in accepted).items())
        ),
        "accepted_bet_net_pnl_sum": accepted_pnl_sum,
        "all_oof_market_policy_pnl": interval,
        "pnl_hard_gate_aggregation": gate["pnl_hard_gate_aggregation"],
        "accepted_side_metrics": side_metrics,
        "accepted_action_metrics": action_metrics,
        "accepted_action_family_metrics": family_metrics,
        "action_and_action_family_pnl_diagnostic_only": gate[
            "action_and_action_family_pnl_diagnostic_only"
        ],
        "oof_prediction_mae": float(np.mean(np.abs(prediction_errors))),
        "oof_prediction_rmse": float(np.sqrt(np.mean(np.square(prediction_errors)))),
        "development_gate_checks": checks,
        "development_gate_passed": gate_passed,
        "gate_blocking_reason_codes": blockers,
        "candidate_specific_future_evaluation_allowed": gate_passed,
        "issue_190_collection_eligible_for_this_candidate": False,
        "strictly_later_persistent_window_required": True,
        "development_calibration_confirmatory_or_future_files_opened": False,
        "current_oof_validation_or_future_pnl_used_for_tuning": False,
        **_blocked_safety_fields(),
    }


def _family_for_action(action: str) -> str:
    if action.endswith("HOLD_TO_SETTLEMENT"):
        return "HOLD_TO_SETTLEMENT"
    if action.endswith("SELL_BEFORE_CLOSE"):
        return "SELL_BEFORE_CLOSE"
    if action == "NO_TRADE":
        return "NO_TRADE"
    raise ValueError(f"unknown action: {action}")


def _validate_lineage(
    *,
    candidate: dict[str, Any],
    issue201_manifest: dict[str, Any],
    role_manifest: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    if candidate.get("candidate_name") != "direct_decision_group_action_advantage_v2":
        raise ValueError("unexpected #198 candidate")
    if issue201_manifest.get("support_conclusion") != (
        "positive_point_action_specific_support_but_market_lcb_nonpositive"
    ):
        raise ValueError("unexpected #201 support conclusion")
    if issue201_manifest.get("source_model_candidate_eligible") is not False:
        raise ValueError("#201 safety lineage is not blocked")
    if role_manifest.get("role_assignment_ready") is not True:
        raise ValueError("role assignment is not ready")
    role_rows_descriptor = _verified_descriptor(
        role_manifest.get("selected_rows"), name="role rows"
    )
    feature_contract_descriptor = _verified_descriptor(
        candidate.get("feature_contract"), name="feature contract"
    )
    target_rows_claim = _claimed_descriptor(
        candidate.get("development_train_action_rows"), name="development_train target rows"
    )
    if role_rows_descriptor["sha256"] != profile["role_assignment_rows_sha256"]:
        raise ValueError("role assignment rows hash mismatch")
    if feature_contract_descriptor["sha256"] != profile["feature_contract_sha256"]:
        raise ValueError("feature contract hash mismatch")
    if target_rows_claim["sha256"] != profile["development_train_target_rows_sha256"]:
        raise ValueError("target rows descriptor hash mismatch")
    if (
        canonical_json_sha256(_v8_execution_guard_config())
        != profile["execution_guard_config_sha256"]
    ):
        raise ValueError("execution guard config hash mismatch")
    role_rows = _load_jsonl(Path(role_rows_descriptor["path"]))
    if _find_fields({"rows": role_rows}, set(TARGET_FIELDS)):
        raise ValueError("role assignment rows contain target fields")
    feature_contract = _load_json(Path(feature_contract_descriptor["path"]))
    validate_pairwise_action_advantage_lcb_feature_contract(
        feature_contract,
        expected_parent_protocol_sha256=feature_contract["parent_protocol_sha256"],
    )
    return {
        "role_rows": role_rows,
        "role_rows_descriptor": role_rows_descriptor,
        "feature_columns": tuple(str(value) for value in feature_contract["feature_columns"]),
        "feature_contract_descriptor": feature_contract_descriptor,
        "target_rows_claim": target_rows_claim,
    }


def _validate_train_rows(
    rows: list[dict[str, Any]],
    *,
    corpus_audits: list[dict[str, Any]],
) -> None:
    if len(rows) != 1800 or len({row["market_id"] for row in rows}) != 90:
        raise ValueError("development_train coverage mismatch")
    groups: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows:
        groups[(str(row["market_id"]), int(row["decision_ts"]))].add(str(row["action"]))
        target = float(row["target_net_pnl_per_contract"])
        if not math.isfinite(target):
            raise ValueError("training target is not finite")
        if row.get("target_used_as_decision_input") is not False:
            raise ValueError("target influence flag is not false")
    if len(groups) != 360 or any(actions != set(REQUIRED_ACTIONS) for actions in groups.values()):
        raise ValueError("development_train five-action grid is incomplete")
    if any(audit["blocking_reason_codes"] for audit in corpus_audits):
        raise ValueError("development_train corpus integrity failed")


def _validate_target_stripped_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    profile: dict[str, Any],
) -> None:
    found = _find_fields({"rows": rows}, set(TARGET_FIELDS))
    if found:
        raise ValueError("target-stripped OOF predictions contain target fields")
    if len(rows) != int(profile["chronological_oof"]["expected_oof_market_count"]) * 4 * 5:
        raise ValueError("target-stripped OOF prediction coverage mismatch")
    groups: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows:
        groups[(str(row["market_id"]), int(row["decision_ts"]))].add(str(row["action"]))
    if any(actions != set(REQUIRED_ACTIONS) for actions in groups.values()):
        raise ValueError("target-stripped OOF action grid is incomplete")


def _strip_target_fields(row: dict[str, Any]) -> dict[str, Any]:
    stripped = {key: value for key, value in row.items() if key not in TARGET_FIELDS}
    if _find_fields(stripped, set(TARGET_FIELDS)):
        raise ValueError("target field remains after stripping")
    stripped["target_fields_stripped"] = True
    stripped["target_used_as_decision_input"] = False
    stripped["outcome_fields_used_as_decision_input"] = False
    return stripped


def _row_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return str(row["market_id"]), int(row["decision_ts"]), str(row["action"])


def _row_sort_key(row: dict[str, Any]) -> tuple[int, str, int]:
    return (
        int(row["decision_ts"]),
        str(row["market_id"]),
        REQUIRED_ACTIONS.index(str(row["action"])),
    )


def _verify_file_hash(path: Path, expected: str, *, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    if _sha256_file(path) != expected:
        raise ValueError(f"{name} SHA-256 mismatch")


def _write_json_fsync(path: Path, payload: dict[str, Any]) -> None:
    _write_text_fsync(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl_fsync(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_text_fsync(
        path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )


def _write_text_fsync(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _is_sha1(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _blocked_safety_fields() -> dict[str, Any]:
    return {
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "paper_candidate_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }


def _pre_label_markdown(report: dict[str, Any]) -> str:
    return (
        "# #202 Pre-label Access Audit\n\n"
        f"- Audit ID: `{report['audit_id']}`\n"
        "- Model/profile frozen before target access: `true`\n"
        "- Label/resolution/PnL files opened before audit: `false`\n"
        "- Calibration, confirmatory, and future evidence opened: `false`\n"
    )


def _gate_markdown(report: dict[str, Any]) -> str:
    interval = report["all_oof_market_policy_pnl"]
    side_lines = "".join(
        f"- {side} accepted PnL: `{row['accepted_bet_net_pnl_sum']}` "
        f"across `{row['accepted_unique_market_count']}` markets\n"
        for side, row in report["accepted_side_metrics"].items()
    )
    return (
        "# Guard-compatible Direct Net-return v4 Development Gate\n\n"
        "- Hard-gate aggregation: `selected_side_buy_up_buy_down_only`\n"
        "- Action and action-family PnL: `diagnostic_only`\n"
        f"- Accepted bets: `{report['guard_accepted_bet_count']}`\n"
        f"- Accepted markets: `{report['guard_accepted_unique_market_count']}`\n"
        f"- Accepted-bet PnL: `{report['accepted_bet_net_pnl_sum']}`\n"
        f"{side_lines}"
        f"- All-market policy PnL LCB: `{interval['lower_confidence_bound']}`\n"
        f"- Gate passed: `{str(report['development_gate_passed']).lower()}`\n"
        f"- Blocking reasons: `{report['gate_blocking_reason_codes']}`\n"
        "- Future evaluation is research-only and requires a strictly later window.\n"
        "- Paper/live/promotion remains blocked.\n"
    )
