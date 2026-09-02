"""Fit the #197 direct action-advantage v2 research candidate safely."""

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

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_direct_decision_group_advantage_v2 import (
    CANDIDATE_NAME,
    FIT_ROLE,
    QUARANTINED_ROLES,
    validate_direct_decision_group_advantage_v2_protocol,
)
from bigan.v8.polymarket.training.execution_layer_v2_outcome_blind_acceptance_viability import (
    _outcome_blind_acceptance_replay,
)
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb import (
    validate_pairwise_action_advantage_lcb_feature_contract,
)
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb_fit import (
    FORBIDDEN_DECISION_FIELDS,
    _blocked_safety_fields,
    _cross_fit_training_predictions,
    _descriptor,
    _find_fields,
    _load_json,
    _load_jsonl,
    _materialize_role_action_rows,
    _predict_role_rows,
    _require_sha256,
    _sha256_file,
    _train_pairwise_ranker,
    _validate_role_rows,
    _verified_descriptor,
    _verify_pin,
    _write_json,
    _write_jsonl,
    _write_text,
    _xgb_model_protocol,
)
from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_action_value import (
    REQUIRED_ACTIONS,
)

SCHEMA_PREFIX = "bigan-v8-direct-decision-group-action-advantage-v2-fit"
ESTIMANDS = (
    "absolute_post_cost_net_return",
    "advantage_vs_no_trade",
    "advantage_vs_best_alternative",
)
FIT_PROFILE_SCHEMA_VERSION = "bigan-v8-direct-decision-group-action-advantage-fit-profile-v2"
MODEL_FILENAME = "direct_decision_group_action_advantage_v2_ranker.xgb.json"
FORBIDDEN_ROLE_METADATA_FIELDS = set(FORBIDDEN_DECISION_FIELDS) | {
    "accepted_bet_net_pnl",
    "evaluation_target_net_pnl_per_contract_by_action",
    "oracle_action",
    "realized_pnl",
    "settlement_outcome",
}


@dataclass(frozen=True, slots=True)
class DirectDecisionGroupAdvantageV2FitConfig:
    """Pinned inputs for development-train-only fitting."""

    run_id: str
    output_dir: Path | str
    pre_registration_manifest_path: Path | str
    expected_pre_registration_manifest_sha256: str
    fit_profile_path: Path | str
    expected_fit_profile_sha256: str
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for name in (
            "expected_pre_registration_manifest_sha256",
            "expected_fit_profile_sha256",
        ):
            _require_sha256(str(getattr(self, name)), name=name)
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(
            self,
            "pre_registration_manifest_path",
            Path(self.pre_registration_manifest_path),
        )
        object.__setattr__(self, "fit_profile_path", Path(self.fit_profile_path))


def validate_direct_decision_group_advantage_v2_fit_profile(
    profile: dict[str, Any],
) -> None:
    cross_fit = dict(profile.get("cross_fit") or {})
    calibration = dict(profile.get("calibration") or {})
    decision = dict(profile.get("decision_rule") or {})
    output = dict(profile.get("output_contract") or {})
    checks = {
        "schema_version": profile.get("schema_version") == FIT_PROFILE_SCHEMA_VERSION,
        "candidate_name": profile.get("candidate_name") == CANDIDATE_NAME,
        "frozen": profile.get("frozen") is True,
        "parent_hashes": all(
            _is_sha256(str(profile.get(name) or ""))
            for name in (
                "parent_protocol_sha256",
                "parent_pre_registration_manifest_sha256",
                "feature_contract_sha256",
                "role_assignment_manifest_sha256",
            )
        ),
        "fit_scope": profile.get("fit_role") == FIT_ROLE
        and int(profile.get("required_fit_market_count") or 0) == 90,
        "cross_fit_windows": int(cross_fit.get("fold_count") or 0) == 5
        and int(cross_fit.get("initial_training_market_count") or 0) == 15
        and int(cross_fit.get("validation_market_count_per_fold") or 0) == 15
        and int(cross_fit.get("expected_oof_market_count") or 0) == 75
        and cross_fit.get("fold_assignment") == "chronological_expanding_window_prior_markets_only"
        and cross_fit.get("future_market_labels_excluded_from_each_fold") is True,
        "fixed_ranker": cross_fit.get("objective") == "rank:pairwise"
        and cross_fit.get("eval_metric") == "ndcg@1"
        and int(cross_fit.get("num_boost_round") or 0) == 120
        and int(cross_fit.get("max_depth") or 0) == 3
        and math.isclose(float(cross_fit.get("eta") or 0.0), 0.03)
        and int(cross_fit.get("nthread") or 0) == 1
        and cross_fit.get("hyperparameter_search_enabled") is False,
        "estimands": calibration.get("estimands") == list(ESTIMANDS),
        "adaptive_buckets": int(calibration.get("maximum_bucket_count") or 0) == 3
        and calibration.get("candidate_quantiles") == [1.0 / 3.0, 2.0 / 3.0]
        and calibration.get("strictly_increasing_boundaries_required") is True
        and calibration.get("duplicate_quantile_boundaries_must_merge") is True
        and calibration.get("unreachable_empty_bucket_allowed") is False
        and int(calibration.get("minimum_unique_markets_per_bucket") or 0) >= 10,
        "full_estimator_bootstrap": calibration.get("bootstrap_unit") == "market_id"
        and int(calibration.get("bootstrap_resample_count") or 0) >= 2_000
        and float(calibration.get("confidence_level") or 0.0) >= 0.95
        and calibration.get("bootstrap_complete_shrunken_estimator_required") is True
        and calibration.get("convex_combination_of_separately_estimated_lcbs_allowed") is False,
        "old_and_future_evidence_sealed": calibration.get(
            "current_issue189_oof_files_may_be_opened"
        )
        is False
        and calibration.get("development_calibration_or_confirmatory_labels_may_be_opened") is False
        and calibration.get("validation_or_future_labels_used_for_tuning") is False,
        "joint_decision_rule": decision.get("trade_must_pass_all_lower_confidence_bounds") is True
        and math.isclose(
            float(decision.get("absolute_post_cost_net_return_lcb_minimum") or 0.0),
            0.02,
        )
        and math.isclose(
            float(decision.get("advantage_vs_no_trade_lcb_minimum") or 0.0),
            0.0,
        )
        and math.isclose(
            float(decision.get("advantage_vs_best_alternative_lcb_minimum") or 0.0),
            0.0,
        ),
        "execution_contract_unchanged": decision.get("execution_guard_mutation_allowed") is False
        and decision.get("cost_model_mutation_allowed") is False
        and decision.get("order_sizing_mutation_allowed") is False,
        "research_only": output.get("research_candidate_only") is True
        and output.get("outcome_blind_target_stripped_viability_replay_required") is True
        and output.get("future_evaluation_in_this_fit_issue_allowed") is False
        and output.get("future_labels_must_remain_sealed") is True,
        "safety": _safety_blocked(profile),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("invalid direct action-advantage v2 fit profile: " + ", ".join(failed))


def fit_direct_decision_group_advantage_v2(
    config: DirectDecisionGroupAdvantageV2FitConfig,
) -> dict[str, Any]:
    """Fit and freeze a research candidate without opening held-out roles."""

    pre_registration_path = config.pre_registration_manifest_path.resolve()
    fit_profile_path = config.fit_profile_path.resolve()
    _verify_pin(
        pre_registration_path,
        config.expected_pre_registration_manifest_sha256,
        name="direct action-advantage v2 pre-registration manifest",
    )
    _verify_pin(
        fit_profile_path,
        config.expected_fit_profile_sha256,
        name="direct action-advantage v2 fit profile",
    )
    pre_registration = _load_json(pre_registration_path)
    profile = {
        **_load_json(fit_profile_path),
        "fit_profile_sha256": _sha256_file(fit_profile_path),
    }
    validate_direct_decision_group_advantage_v2_fit_profile(profile)
    lineage = _validate_pre_registration_lineage(
        pre_registration,
        pre_registration_path=pre_registration_path,
        fit_profile=profile,
    )
    role_rows = lineage["role_rows"]
    feature_contract = lineage["feature_contract"]
    feature_columns = tuple(str(value) for value in feature_contract["feature_columns"])

    run_dir = _prepare_run_dir(
        config.output_dir,
        config.run_id,
        overwrite=config.overwrite_existing,
    )
    pre_label_audit = {
        "schema_version": f"{SCHEMA_PREFIX}-pre-label-access-lineage-audit-v1",
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "pre_registration_manifest": _descriptor(pre_registration_path),
        "fit_profile": _descriptor(fit_profile_path),
        "protocol": lineage["protocol_descriptor"],
        "feature_contract": lineage["feature_contract_descriptor"],
        "role_assignment_manifest": lineage["role_manifest_descriptor"],
        "role_assignment_selected_rows": lineage["selected_rows_descriptor"],
        "fit_role": FIT_ROLE,
        "fit_market_count": 90,
        "fit_market_ids_sha256": canonical_json_sha256(
            sorted(str(row["market_id"]) for row in role_rows if row["role"] == FIT_ROLE)
        ),
        "quarantined_roles": list(QUARANTINED_ROLES),
        "quarantined_market_count": 105,
        "role_assignment_metadata_opened": True,
        "feature_files_opened_before_audit": False,
        "label_or_resolution_files_opened_before_audit": False,
        "current_issue189_oof_files_opened": False,
        "validation_confirmatory_or_future_files_opened": False,
        "prediction_attempted_before_audit": False,
        "pre_label_access_validation_passed": True,
        **_diagnostic_safety_fields(),
    }
    pre_label_audit["audit_id"] = canonical_json_sha256(pre_label_audit)
    pre_label_audit_path = run_dir / "pre_label_access_lineage_audit.json"
    _write_json_fsync(pre_label_audit_path, pre_label_audit)
    _write_text(
        run_dir / "pre_label_access_lineage_audit.md",
        _pre_label_markdown(pre_label_audit),
    )

    action_rows_by_role, corpus_audits = _materialize_role_action_rows(
        role_rows,
        feature_columns=feature_columns,
        roles=(FIT_ROLE,),
    )
    train_rows = action_rows_by_role[FIT_ROLE]
    _validate_fit_materialization(train_rows, corpus_audits)
    action_rows_path = run_dir / "direct_advantage_v2_development_train_action_rows.jsonl"
    _write_jsonl(action_rows_path, train_rows)

    cross_fit = _cross_fit_training_predictions(
        train_rows,
        feature_columns=feature_columns,
        model_protocol=dict(profile["cross_fit"]),
    )
    oof_predictions = list(cross_fit.pop("oof_predictions"))
    direct_oof_rows = _attach_direct_estimands(oof_predictions)
    oof_path = run_dir / "direct_advantage_v2_internal_train_oof_predictions.jsonl"
    _write_jsonl(oof_path, direct_oof_rows)
    oof_coverage_path = run_dir / "direct_advantage_v2_internal_oof_coverage_report.json"
    oof_coverage = {
        "schema_version": f"{SCHEMA_PREFIX}-internal-oof-coverage-report-v1",
        "run_id": config.run_id,
        **cross_fit,
        "direct_estimands_attached": list(ESTIMANDS),
        "current_issue189_oof_files_opened": False,
        "development_calibration_confirmatory_or_future_labels_used": False,
        **_diagnostic_safety_fields(),
    }
    oof_coverage["report_id"] = canonical_json_sha256(oof_coverage)
    _write_json(oof_coverage_path, oof_coverage)

    calibration = _build_direct_advantage_calibration(
        direct_oof_rows,
        profile=profile,
        feature_contract_sha256=lineage["feature_contract_descriptor"]["sha256"],
    )
    calibration_path = run_dir / "direct_action_advantage_v2_calibration_artifact.json"
    _write_json(calibration_path, calibration)
    calibration_report_path = run_dir / "direct_action_advantage_v2_calibration_report.json"
    calibration_report = {
        "schema_version": f"{SCHEMA_PREFIX}-calibration-report-v1",
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "source": "new_internal_development_train_oof_predictions_only",
        "oof_predictions": _descriptor(oof_path),
        "calibration_artifact": _descriptor(calibration_path),
        "action_bucket_summary": calibration["actions"],
        "group_count": len(calibration["calibration_groups"]),
        "duplicate_quantile_boundary_merge_count": calibration[
            "duplicate_quantile_boundary_merge_count"
        ],
        "unreachable_empty_bucket_count": calibration["unreachable_empty_bucket_count"],
        "complete_shrunken_estimator_bootstrap_used": True,
        "convex_combination_of_separate_lcbs_used": False,
        "validation_confirmatory_or_future_labels_used": False,
        **_diagnostic_safety_fields(),
    }
    calibration_report["report_id"] = canonical_json_sha256(calibration_report)
    _write_json(calibration_report_path, calibration_report)

    booster = _train_pairwise_ranker(
        train_rows,
        feature_columns=feature_columns,
        model_protocol=_xgb_model_protocol(dict(profile["cross_fit"])),
    )
    model_path = run_dir / MODEL_FILENAME
    booster.save_model(model_path)
    final_predictions = _predict_role_rows(
        train_rows,
        booster=booster,
        feature_columns=feature_columns,
    )
    scored_predictions = _apply_direct_advantage_calibration(
        final_predictions,
        calibration=calibration,
        profile=profile,
    )
    stripped_predictions = [_strip_training_targets(row) for row in scored_predictions]
    _validate_target_stripped_rows(stripped_predictions)
    stripped_path = run_dir / "direct_advantage_v2_target_stripped_predictions.jsonl"
    _write_jsonl(stripped_path, stripped_predictions)
    viability_rows = _outcome_blind_acceptance_replay(
        stripped_predictions,
        entry_threshold=float(
            profile["decision_rule"]["absolute_post_cost_net_return_lcb_minimum"]
        ),
        runner_up_advantage_threshold=0.0,
    )
    viability_path = run_dir / "direct_advantage_v2_outcome_blind_viability_rows.jsonl"
    _write_jsonl(viability_path, viability_rows)
    viability_report = _viability_report(
        run_id=config.run_id,
        viability_rows=viability_rows,
        scored_predictions=stripped_predictions,
    )
    viability_report_path = run_dir / "direct_advantage_v2_outcome_blind_viability_report.json"
    _write_json(viability_report_path, viability_report)

    training_report = {
        "schema_version": f"{SCHEMA_PREFIX}-training-report-v1",
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "fit_market_count": 90,
        "materialized_action_row_count": len(train_rows),
        "decision_group_count": len(train_rows) // len(REQUIRED_ACTIONS),
        "corpus_audit_count": len(corpus_audits),
        "feature_causality_violation_count": sum(
            int(audit["feature_causality_violation_count"]) for audit in corpus_audits
        ),
        "cost_component_violation_count": sum(
            int(audit["cost_component_violation_count"]) for audit in corpus_audits
        ),
        "new_internal_oof_market_count": cross_fit["oof_market_count"],
        "new_internal_oof_prediction_count": cross_fit["oof_prediction_count"],
        "hyperparameter_search_enabled": False,
        "training_targets_include_costs": True,
        "training_outcomes_used_as_targets_only": True,
        "current_oof_validation_or_confirmatory_pnl_used_for_tuning": False,
        "development_calibration_confirmatory_or_future_files_opened": False,
        "model": _descriptor(model_path),
        "calibration_artifact": _descriptor(calibration_path),
        "viability_report": _descriptor(viability_report_path),
        "outcome_blind_viability_passed": viability_report["outcome_blind_viability_passed"],
        "outcome_blind_viability_blocking_reason_codes": viability_report[
            "outcome_blind_viability_blocking_reason_codes"
        ],
        **_diagnostic_safety_fields(),
    }
    training_report["report_id"] = canonical_json_sha256(training_report)
    training_report_path = run_dir / "direct_advantage_v2_training_report.json"
    _write_json(training_report_path, training_report)

    candidate_manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-research-candidate-freeze-manifest-v1",
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "research_candidate_frozen": True,
        "research_candidate_only": True,
        "outcome_blind_viability_passed": viability_report["outcome_blind_viability_passed"],
        "research_candidate_rejected_for_zero_viability": not viability_report[
            "outcome_blind_viability_passed"
        ],
        "candidate_specific_future_evaluation_allowed": False,
        "candidate_specific_future_evaluation_blocking_reason_codes": viability_report[
            "outcome_blind_viability_blocking_reason_codes"
        ],
        "pre_registration_manifest": _descriptor(pre_registration_path),
        "fit_profile": _descriptor(fit_profile_path),
        "protocol": lineage["protocol_descriptor"],
        "feature_contract": lineage["feature_contract_descriptor"],
        "role_assignment_manifest": lineage["role_manifest_descriptor"],
        "pre_label_access_lineage_audit": _descriptor(pre_label_audit_path),
        "development_train_action_rows": _descriptor(action_rows_path),
        "internal_train_oof_predictions": _descriptor(oof_path),
        "internal_oof_coverage_report": _descriptor(oof_coverage_path),
        "model": _descriptor(model_path),
        "direct_action_advantage_calibration_artifact": _descriptor(calibration_path),
        "calibration_report": _descriptor(calibration_report_path),
        "target_stripped_predictions": _descriptor(stripped_path),
        "outcome_blind_viability_rows": _descriptor(viability_path),
        "outcome_blind_viability_report": _descriptor(viability_report_path),
        "training_report": _descriptor(training_report_path),
        "development_calibration_confirmatory_or_future_files_opened": False,
        "future_labels_remain_sealed": True,
        "future_evaluation_started": False,
        "current_issue189_oof_files_opened": False,
        "current_oof_validation_or_confirmatory_pnl_used_for_tuning": False,
        "threshold_guard_cost_or_sizing_mutated": False,
        **_diagnostic_safety_fields(),
    }
    candidate_manifest["research_candidate_hash"] = canonical_json_sha256(candidate_manifest)
    candidate_manifest_path = run_dir / "research_candidate_freeze_manifest.json"
    _write_json(candidate_manifest_path, candidate_manifest)
    return {
        "run_dir": run_dir,
        "pre_label_audit_path": pre_label_audit_path,
        "oof_coverage": oof_coverage,
        "calibration": calibration,
        "viability_report": viability_report,
        "training_report": training_report,
        "candidate_manifest": candidate_manifest,
        "candidate_manifest_path": candidate_manifest_path,
        "candidate_manifest_sha256": _sha256_file(candidate_manifest_path),
    }


def _validate_pre_registration_lineage(
    pre_registration: dict[str, Any],
    *,
    pre_registration_path: Path,
    fit_profile: dict[str, Any],
) -> dict[str, Any]:
    if not (
        pre_registration.get("pre_registration_ready") is True
        and pre_registration.get("label_outcome_or_pnl_files_opened") is False
        and pre_registration.get("fitting_or_prediction_attempted") is False
        and _safety_blocked(pre_registration)
    ):
        raise ValueError("#197 pre-registration manifest is not fit-eligible")
    if fit_profile["parent_pre_registration_manifest_sha256"] != _sha256_file(
        pre_registration_path
    ):
        raise ValueError("fit profile pre-registration lineage mismatch")
    descriptors = dict(pre_registration.get("input_descriptors") or {})
    protocol_descriptor = _verified_descriptor(descriptors.get("protocol"), name="#197 protocol")
    if fit_profile["parent_protocol_sha256"] != protocol_descriptor["sha256"]:
        raise ValueError("fit profile parent protocol mismatch")
    protocol = _load_json(Path(protocol_descriptor["path"]))
    validate_direct_decision_group_advantage_v2_protocol(protocol)
    role_manifest_descriptor = _verified_descriptor(
        descriptors.get("role_assignment_manifest"),
        name="role assignment manifest",
    )
    if fit_profile["role_assignment_manifest_sha256"] != role_manifest_descriptor["sha256"]:
        raise ValueError("fit profile role assignment lineage mismatch")
    role_manifest = _load_json(Path(role_manifest_descriptor["path"]))
    if not (
        role_manifest.get("role_assignment_ready") is True
        and role_manifest.get("labels_or_outcomes_opened_for_role_assignment") is False
        and _safety_blocked(role_manifest)
    ):
        raise ValueError("role assignment is not outcome-blind and ready")
    selected_rows_descriptor = _verified_descriptor(
        role_manifest.get("selected_rows"), name="role assignment selected rows"
    )
    role_rows = _load_jsonl(Path(selected_rows_descriptor["path"]))
    _validate_role_rows(role_rows)
    if _find_fields({"rows": role_rows}, FORBIDDEN_ROLE_METADATA_FIELDS):
        raise ValueError("role assignment metadata contains forbidden target fields")
    feature_contract_descriptor = _verified_descriptor(
        role_manifest.get("feature_contract"), name="frozen feature contract"
    )
    if fit_profile["feature_contract_sha256"] != feature_contract_descriptor["sha256"]:
        raise ValueError("fit profile feature contract lineage mismatch")
    feature_contract = _load_json(Path(feature_contract_descriptor["path"]))
    validate_pairwise_action_advantage_lcb_feature_contract(
        feature_contract,
        expected_parent_protocol_sha256=feature_contract["parent_protocol_sha256"],
    )
    return {
        "protocol": protocol,
        "protocol_descriptor": protocol_descriptor,
        "role_manifest": role_manifest,
        "role_manifest_descriptor": role_manifest_descriptor,
        "role_rows": role_rows,
        "selected_rows_descriptor": selected_rows_descriptor,
        "feature_contract": feature_contract,
        "feature_contract_descriptor": feature_contract_descriptor,
    }


def _validate_fit_materialization(
    train_rows: list[dict[str, Any]],
    corpus_audits: list[dict[str, Any]],
) -> None:
    markets = {str(row["market_id"]) for row in train_rows}
    if len(markets) != 90 or len(corpus_audits) != 90:
        raise ValueError("development_train materialization coverage is incomplete")
    if {str(row["role"]) for row in train_rows} != {FIT_ROLE}:
        raise ValueError("quarantined role rows were materialized")
    if any(audit["blocking_reason_codes"] for audit in corpus_audits):
        raise ValueError("development_train corpus audit failed")


def _attach_direct_estimands(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["market_id"]), int(row["decision_ts"]))].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda value: (value[1], value[0])):
        group = grouped[key]
        if {str(row["action"]) for row in group} != set(REQUIRED_ACTIONS):
            raise ValueError("direct estimands require a complete five-action group")
        returns = {str(row["action"]): float(row["target_net_pnl_per_contract"]) for row in group}
        no_trade_return = returns["NO_TRADE"]
        if not math.isclose(no_trade_return, 0.0, abs_tol=1e-12):
            raise ValueError("NO_TRADE target must be the zero-return anchor")
        for row in group:
            action = str(row["action"])
            absolute = returns[action]
            best_alternative = max(
                value for candidate, value in returns.items() if candidate != action
            )
            updated = {
                **row,
                "training_target_absolute_post_cost_net_return": absolute,
                "training_target_advantage_vs_no_trade": absolute - no_trade_return,
                "training_target_advantage_vs_best_alternative": (absolute - best_alternative),
                "training_targets_include_costs": True,
                "training_targets_used_as_decision_inputs": False,
            }
            updated["direct_estimand_row_sha256"] = canonical_json_sha256(updated)
            output.append(updated)
    return output


def _build_direct_advantage_calibration(
    oof_rows: list[dict[str, Any]],
    *,
    profile: dict[str, Any],
    feature_contract_sha256: str,
) -> dict[str, Any]:
    calibration_profile = dict(profile["calibration"])
    prior = int(calibration_profile["shrinkage_prior_market_count"])
    minimum_group_markets = int(calibration_profile["minimum_unique_markets_per_bucket"])
    actions: dict[str, Any] = {}
    groups: dict[str, Any] = {}
    merged_boundary_count = 0
    unreachable_bucket_count = 0
    for action_index, action in enumerate(REQUIRED_ACTIONS):
        rows = [row for row in oof_rows if row["action"] == action]
        scores = [float(row["pairwise_group_normalized_rank_score"]) for row in rows]
        boundaries, merged = _adaptive_boundaries(
            scores,
            quantiles=list(calibration_profile["candidate_quantiles"]),
        )
        merged_boundary_count += merged
        bucket_names = [f"bucket_{index}" for index in range(len(boundaries) + 1)]
        rows_by_bucket = {
            name: [
                row
                for row in rows
                if _adaptive_bucket(float(row["pairwise_group_normalized_rank_score"]), boundaries)
                == name
            ]
            for name in bucket_names
        }
        empty = [name for name, values in rows_by_bucket.items() if not values]
        if empty:
            unreachable_bucket_count += len(empty)
            raise ValueError(f"adaptive calibration emitted empty buckets: {empty}")
        action_estimators = {
            estimand: _market_bootstrap_shrunken_estimator(
                rows,
                rows,
                target_field=f"training_target_{estimand}",
                prior_market_count=prior,
                minimum_group_markets=minimum_group_markets,
                bootstrap_resample_count=int(calibration_profile["bootstrap_resample_count"]),
                confidence_level=float(calibration_profile["confidence_level"]),
                seed=int(calibration_profile["bootstrap_seed"]) + action_index * 100_000,
                force_action_level=True,
            )
            for estimand in ESTIMANDS
        }
        actions[action] = {
            "oof_row_count": len(rows),
            "oof_unique_market_count": len({str(row["market_id"]) for row in rows}),
            "adaptive_score_boundaries": boundaries,
            "adaptive_bucket_names": bucket_names,
            "duplicate_quantile_boundary_merge_count": merged,
            "action_level_estimators": action_estimators,
        }
        for bucket_index, bucket_name in enumerate(bucket_names):
            bucket_rows = rows_by_bucket[bucket_name]
            estimators = {
                estimand: _market_bootstrap_shrunken_estimator(
                    rows,
                    bucket_rows,
                    target_field=f"training_target_{estimand}",
                    prior_market_count=prior,
                    minimum_group_markets=minimum_group_markets,
                    bootstrap_resample_count=int(calibration_profile["bootstrap_resample_count"]),
                    confidence_level=float(calibration_profile["confidence_level"]),
                    seed=int(calibration_profile["bootstrap_seed"])
                    + action_index * 100_000
                    + bucket_index * 1_000
                    + ESTIMANDS.index(estimand),
                )
                for estimand in ESTIMANDS
            }
            key = f"{action}|{bucket_name}"
            groups[key] = {
                "action": action,
                "bucket_name": bucket_name,
                "row_count": len(bucket_rows),
                "unique_market_count": len({str(row["market_id"]) for row in bucket_rows}),
                "minimum_required_unique_markets": minimum_group_markets,
                "estimators": estimators,
                "all_estimators_finite": all(
                    math.isfinite(float(estimator["point_estimate"]))
                    and math.isfinite(float(estimator["lower_confidence_bound"]))
                    for estimator in estimators.values()
                ),
            }
    artifact = {
        "schema_version": f"{SCHEMA_PREFIX}-calibration-artifact-v1",
        "candidate_name": CANDIDATE_NAME,
        "source": "new_internal_development_train_oof_predictions_only",
        "estimands": list(ESTIMANDS),
        "feature_contract_sha256": feature_contract_sha256,
        "fit_profile_sha256": profile["fit_profile_sha256"],
        "actions": actions,
        "calibration_groups": groups,
        "duplicate_quantile_boundary_merge_count": merged_boundary_count,
        "unreachable_empty_bucket_count": unreachable_bucket_count,
        "complete_shrunken_estimator_market_bootstrap": True,
        "convex_combination_of_separately_estimated_lcbs_used": False,
        "current_issue189_oof_files_opened": False,
        "validation_confirmatory_or_future_labels_used": False,
        **_diagnostic_safety_fields(),
    }
    artifact["calibration_artifact_id"] = canonical_json_sha256(artifact)
    return artifact


def _adaptive_boundaries(
    scores: list[float],
    *,
    quantiles: list[float],
) -> tuple[list[float], int]:
    if not scores or not all(math.isfinite(value) for value in scores):
        raise ValueError("adaptive bucket scores must be finite and non-empty")
    candidates = [float(np.quantile(scores, quantile, method="linear")) for quantile in quantiles]
    maximum_score = max(scores)
    boundaries: list[float] = []
    for value in candidates:
        if maximum_score - value <= 1e-12:
            continue
        if not boundaries or value - boundaries[-1] > 1e-12:
            boundaries.append(value)
    return boundaries, len(candidates) - len(boundaries)


def _adaptive_bucket(score: float, boundaries: list[float]) -> str:
    for index, boundary in enumerate(boundaries):
        if score <= boundary:
            return f"bucket_{index}"
    return f"bucket_{len(boundaries)}"


def _market_bootstrap_shrunken_estimator(
    action_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    *,
    target_field: str,
    prior_market_count: int,
    minimum_group_markets: int,
    bootstrap_resample_count: int,
    confidence_level: float,
    seed: int,
    force_action_level: bool = False,
) -> dict[str, Any]:
    action_by_market = _market_target_means(action_rows, target_field=target_field)
    group_by_market = _market_target_means(group_rows, target_field=target_field)
    market_ids = sorted(action_by_market)
    if not market_ids:
        raise ValueError("market bootstrap requires action-level market support")
    group_support = len(group_by_market)
    support_passed = group_support >= minimum_group_markets
    weight = (
        0.0
        if force_action_level or not support_passed
        else group_support / (group_support + prior_market_count)
    )
    action_mean = float(np.mean(list(action_by_market.values())))
    group_mean = float(np.mean(list(group_by_market.values()))) if group_by_market else action_mean
    point = weight * group_mean + (1.0 - weight) * action_mean
    rng = np.random.default_rng(seed)
    samples = rng.choice(
        np.asarray(market_ids, dtype=object),
        size=(bootstrap_resample_count, len(market_ids)),
        replace=True,
    )
    bootstrap_values = np.empty(bootstrap_resample_count, dtype=np.float64)
    for index, sample in enumerate(samples):
        action_sample = [action_by_market[str(market_id)] for market_id in sample]
        group_sample = [
            group_by_market[str(market_id)]
            for market_id in sample
            if str(market_id) in group_by_market
        ]
        sampled_action_mean = float(np.mean(action_sample))
        sampled_group_mean = float(np.mean(group_sample)) if group_sample else sampled_action_mean
        bootstrap_values[index] = weight * sampled_group_mean + (1.0 - weight) * sampled_action_mean
    lower = float(np.quantile(bootstrap_values, 1.0 - confidence_level, method="lower"))
    if not all(math.isfinite(value) for value in (point, lower)):
        raise ValueError("complete shrunken estimator bootstrap is not finite")
    return {
        "target_field": target_field,
        "point_estimate": point,
        "lower_confidence_bound": lower,
        "action_level_mean": action_mean,
        "group_level_mean": group_mean,
        "group_unique_market_count": group_support,
        "group_support_passed": support_passed,
        "shrinkage_group_weight": weight,
        "estimate_source": (
            "action_level_market_bootstrap"
            if weight == 0.0
            else "complete_shrunken_estimator_market_bootstrap"
        ),
        "bootstrap_unit": "market_id",
        "bootstrap_resample_count": bootstrap_resample_count,
        "bootstrap_seed": seed,
        "confidence_level": confidence_level,
        "convex_combination_of_separate_lcbs_used": False,
    }


def _market_target_means(
    rows: list[dict[str, Any]],
    *,
    target_field: str,
) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[target_field])
        if not math.isfinite(value):
            raise ValueError("calibration targets must be finite")
        values[str(row["market_id"])].append(value)
    return {market_id: float(np.mean(market_values)) for market_id, market_values in values.items()}


def _apply_direct_advantage_calibration(
    predictions: list[dict[str, Any]],
    *,
    calibration: dict[str, Any],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    decision = dict(profile["decision_rule"])
    output: list[dict[str, Any]] = []
    for row in predictions:
        action = str(row["action"])
        boundaries = list(calibration["actions"][action]["adaptive_score_boundaries"])
        bucket = _adaptive_bucket(float(row["pairwise_group_normalized_rank_score"]), boundaries)
        group = calibration["calibration_groups"][f"{action}|{bucket}"]
        estimators = dict(group["estimators"])
        points = {name: float(estimators[name]["point_estimate"]) for name in ESTIMANDS}
        lcbs = {name: float(estimators[name]["lower_confidence_bound"]) for name in ESTIMANDS}
        passes = bool(
            action != "NO_TRADE"
            and lcbs["absolute_post_cost_net_return"]
            >= float(decision["absolute_post_cost_net_return_lcb_minimum"])
            and lcbs["advantage_vs_no_trade"] > float(decision["advantage_vs_no_trade_lcb_minimum"])
            and lcbs["advantage_vs_best_alternative"]
            > float(decision["advantage_vs_best_alternative_lcb_minimum"])
        )
        if action == "NO_TRADE":
            score = 0.0
            source = "frozen_no_trade_zero_anchor"
        elif passes:
            score = lcbs["absolute_post_cost_net_return"]
            source = "all_direct_advantage_lcb_checks_passed"
        else:
            score = min(0.0, *lcbs.values())
            source = "one_or_more_direct_advantage_lcb_checks_failed"
        updated = {
            **row,
            "direct_advantage_bucket": bucket,
            "direct_advantage_point_estimates": points,
            "direct_advantage_lower_confidence_bounds": lcbs,
            "direct_advantage_all_lcb_checks_passed": passes,
            "direct_advantage_score_source": source,
            "direct_action_advantage_lcb_score": score,
            "calibrated_action_expected_net_return": points["absolute_post_cost_net_return"],
            "action_advantage_lcb_net_return": score,
            "action_advantage_lcb_score_bucket": bucket,
            "action_advantage_lcb_estimate_source": source,
            "ranking_score_source": (
                "model_predicted_pairwise_rank_score_with_direct_group_advantage_lcbs"
            ),
            "target_used_as_decision_input": False,
            "outcome_fields_used_as_decision_input": False,
        }
        updated["direct_scored_prediction_sha256"] = canonical_json_sha256(updated)
        output.append(updated)
    return output


def _strip_training_targets(row: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "market_id",
        "condition_id",
        "market_slug",
        "decision_ts",
        "market_close_ts",
        "max_input_ts",
        "role",
        "market_selection_rank",
        "action",
        "side",
        "action_family",
        "decision_time_features",
        "p_up",
        "p_down",
        "selected_side_probability",
        "microstructure_snapshot",
        "reference_price_feature_provenance",
        "p_up_action_disagreement",
        "raw_pairwise_rank_score",
        "pairwise_action_rank",
        "pairwise_rank_percentile",
        "pairwise_group_normalized_rank_score",
        "pairwise_group_score_range",
        "pairwise_normalized_margin_vs_no_trade",
        "pairwise_normalized_margin_vs_best_alternative",
        "pairwise_rank_normalization_scope",
        "raw_rank_score_cross_model_comparison_allowed",
        "direct_advantage_bucket",
        "direct_advantage_point_estimates",
        "direct_advantage_lower_confidence_bounds",
        "direct_advantage_all_lcb_checks_passed",
        "direct_advantage_score_source",
        "direct_action_advantage_lcb_score",
        "calibrated_action_expected_net_return",
        "action_advantage_lcb_net_return",
        "action_advantage_lcb_score_bucket",
        "action_advantage_lcb_estimate_source",
        "ranking_score_source",
        "target_used_as_decision_input",
        "outcome_fields_used_as_decision_input",
        "paper_only",
        "capital_at_risk",
    }
    stripped = {key: value for key, value in row.items() if key in allowed}
    stripped["training_target_fields_stripped"] = True
    stripped["target_or_outcome_fields_used"] = False
    stripped["target_stripped_row_sha256"] = canonical_json_sha256(stripped)
    return stripped


def _validate_target_stripped_rows(rows: list[dict[str, Any]]) -> None:
    forbidden = set(FORBIDDEN_DECISION_FIELDS) | {
        "target_net_pnl_per_contract",
        "training_target_absolute_post_cost_net_return",
        "training_target_advantage_vs_no_trade",
        "training_target_advantage_vs_best_alternative",
        "target_cost_components",
        "target_resolved_outcome",
    }
    found = _find_fields({"rows": rows}, forbidden)
    if found:
        raise ValueError("target-stripped inference rows still contain training targets")


def _viability_report(
    *,
    run_id: str,
    viability_rows: list[dict[str, Any]],
    scored_predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_actions = Counter(row["source_selected_action"] for row in viability_rows)
    terminal_stages = Counter(row["first_terminal_stage"] for row in viability_rows)
    passed_action_rows = [
        row for row in scored_predictions if row["direct_advantage_all_lcb_checks_passed"]
    ]
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-outcome-blind-viability-report-v1",
        "run_id": run_id,
        "candidate_name": CANDIDATE_NAME,
        "decision_group_count": len(viability_rows),
        "target_stripped_action_row_count": len(scored_predictions),
        "direct_lcb_passed_action_row_count": len(passed_action_rows),
        "direct_lcb_passed_action_count_by_action": dict(
            sorted(Counter(row["action"] for row in passed_action_rows).items())
        ),
        "selected_action_distribution": dict(sorted(selected_actions.items())),
        "execution_guard_evaluated_count": sum(
            bool(row["execution_guard_evaluated"]) for row in viability_rows
        ),
        "execution_guard_allowed_count": sum(
            bool(row["execution_guard_order_allowed"]) for row in viability_rows
        ),
        "first_terminal_stage_distribution": dict(sorted(terminal_stages.items())),
        "first_terminal_stage_reconciled": sum(terminal_stages.values()) == len(viability_rows),
        "training_target_fields_present": False,
        "outcome_or_pnl_used_for_viability": False,
        "viability_is_promotion_evidence": False,
        "outcome_blind_viability_passed": bool(
            passed_action_rows and any(row["execution_guard_evaluated"] for row in viability_rows)
        ),
        "outcome_blind_viability_blocking_reason_codes": _viability_blockers(
            passed_action_rows=passed_action_rows,
            viability_rows=viability_rows,
        ),
        "threshold_guard_cost_or_sizing_mutated": False,
        **_diagnostic_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def _viability_blockers(
    *,
    passed_action_rows: list[dict[str, Any]],
    viability_rows: list[dict[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    if not passed_action_rows:
        blockers.append("zero_direct_lcb_passed_action_support")
    if not any(row["execution_guard_evaluated"] for row in viability_rows):
        blockers.append("zero_execution_guard_evaluable_decision_support")
    return blockers


def _safety_blocked(payload: dict[str, Any]) -> bool:
    expected = _blocked_safety_fields()
    return all(payload.get(key) is value for key, value in expected.items())


def _diagnostic_safety_fields() -> dict[str, Any]:
    return {
        **_blocked_safety_fields(),
        "paper_candidate_allowed": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
    }


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value.lower())


def _prepare_run_dir(output_dir: Path, run_id: str, *, overwrite: bool) -> Path:
    run_dir = output_dir.expanduser().resolve() / run_id
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    return run_dir


def _write_json_fsync(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _pre_label_markdown(audit: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Direct Action-Advantage v2 Pre-Label Audit",
            "",
            f"- fit role / markets: `{audit['fit_role']} / {audit['fit_market_count']}`",
            f"- quarantined roles: `{audit['quarantined_roles']}`",
            "- feature files opened before audit: `false`",
            "- label/resolution files opened before audit: `false`",
            "- current #189 OOF files opened: `false`",
            "- validation/confirmatory/future files opened: `false`",
            "- safety unlock: `false`",
            "",
        ]
    )
