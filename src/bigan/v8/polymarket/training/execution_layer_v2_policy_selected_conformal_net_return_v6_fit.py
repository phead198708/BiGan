"""Fit and freeze the #207 policy-selected conformal net-return v6 candidate."""

from __future__ import annotations

import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_guard_compatible_conformal_net_return_v5 import (
    _raw_target_stripped_predictions,
)
from bigan.v8.polymarket.training.execution_layer_v2_guard_compatible_direct_net_return_v4 import (
    _predict_regressor,
    _row_key,
    _row_sort_key,
    _train_regressor,
    _write_json_fsync,
    _write_jsonl_fsync,
    _write_text_fsync,
)
from bigan.v8.polymarket.training.execution_layer_v2_p_up_aligned_action_value_support import (
    build_execution_compatible_action_universe,
)
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb_fit import (
    _load_corpus_action_rows,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    CANDIDATE_NAME,
    DEVELOPMENT_SETTLEMENT_MANIFEST_SCHEMA_VERSION,
    REQUIRED_ACTIONS,
    SIDES,
    TRADE_ACTIONS,
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
    validate_policy_selected_conformal_v6_profile,
)

SCHEMA_PREFIX = "bigan-v8-policy-selected-conformal-net-return-v6"
MODEL_FILENAME = "policy_selected_conformal_net_return_v6.xgb.json"
FIT_ROLE = "point_model_fit"
CALIBRATION_ROLE = "conformal_calibration"
CHECK_ROLE = "calibration_check"
TARGET_FIELDS = frozenset(
    {
        "accepted_bet_net_pnl",
        "final_outcome",
        "label",
        "net_pnl",
        "oracle_action",
        "realized_pnl",
        "resolved_outcome",
        "settlement_outcome",
        "settlement_pnl",
        "target_net_pnl_per_contract",
        "total_net_pnl_per_notional",
        "winning_outcome",
    }
)


@dataclass(frozen=True, slots=True)
class PolicySelectedConformalNetReturnV6FitConfig:
    """Pinned post-settlement inputs for one non-tuned v6 fit."""

    run_id: str
    output_dir: Path | str
    profile_path: Path | str
    expected_profile_sha256: str
    development_settlement_manifest_path: Path | str
    expected_development_settlement_manifest_sha256: str
    feature_contract_path: Path | str
    expected_feature_contract_sha256: str
    implementation_commit: str
    candidate_freeze_created_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for field in (
            "expected_profile_sha256",
            "expected_development_settlement_manifest_sha256",
            "expected_feature_contract_sha256",
        ):
            _require_sha256(str(getattr(self, field)), name=field)
        _require_git_sha(self.implementation_commit)
        if self.candidate_freeze_created_ts <= 0:
            raise ValueError("candidate_freeze_created_ts must be positive")
        for field in (
            "output_dir",
            "profile_path",
            "development_settlement_manifest_path",
            "feature_contract_path",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)))


def fit_policy_selected_conformal_net_return_v6(
    config: PolicySelectedConformalNetReturnV6FitConfig,
) -> dict[str, Any]:
    """Fit on 150, calibrate on 60, and support-check target-free on 50."""

    profile_path = config.profile_path.resolve()
    settlement_path = config.development_settlement_manifest_path.resolve()
    feature_contract_path = config.feature_contract_path.resolve()
    _verify_pin(profile_path, config.expected_profile_sha256, "v6 profile")
    _verify_pin(
        settlement_path,
        config.expected_development_settlement_manifest_sha256,
        "v6 development settlement manifest",
    )
    _verify_pin(
        feature_contract_path,
        config.expected_feature_contract_sha256,
        "feature contract",
    )
    profile = _load_json(profile_path)
    validate_policy_selected_conformal_v6_profile(profile)
    if config.expected_feature_contract_sha256 != profile["frozen_upstream"][
        "feature_contract_sha256"
    ]:
        raise ValueError("feature contract pin does not match v6 profile")
    feature_contract = _load_json(feature_contract_path)
    feature_columns = tuple(str(value) for value in feature_contract["feature_columns"])
    settlement = _load_json(settlement_path)
    _validate_settlement_manifest_for_fit(settlement)
    settled_role_descriptor = _verified_descriptor(
        settlement.get("settled_role_rows"),
        "settled development role rows",
    )
    role_rows = _load_jsonl(Path(settled_role_descriptor["path"]))
    _validate_settled_role_lineage(role_rows)
    window_descriptor = _verified_descriptor(
        settlement.get("development_window_manifest"),
        "development window manifest",
    )
    window = _load_json(Path(window_descriptor["path"]))
    check_action_descriptor = _verified_descriptor(
        window.get("target_free_five_action_rows"),
        "target-free development action rows",
    )
    check_action_rows = [
        row
        for row in _load_jsonl(Path(check_action_descriptor["path"]))
        if row.get("development_role") == CHECK_ROLE
    ]
    _validate_target_free_check_rows(check_action_rows, feature_columns=feature_columns)

    run_dir = Path(config.output_dir).resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    pre_label = {
        "schema_version": f"{SCHEMA_PREFIX}-pre-label-access-audit-v1",
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        "profile": _descriptor(profile_path),
        "development_settlement_manifest": _descriptor(settlement_path),
        "settled_role_rows": settled_role_descriptor,
        "development_window_manifest": window_descriptor,
        "feature_contract": _descriptor(feature_contract_path),
        "target_free_calibration_check_rows": check_action_descriptor,
        "role_market_counts": dict(
            sorted(Counter(str(row["development_role"]) for row in role_rows).items())
        ),
        "roles_frozen_before_label_access": True,
        "fit_or_calibration_label_files_opened_before_audit": False,
        "calibration_check_label_files_opened_by_fit": False,
        "issue204_outcome_settlement_target_or_pnl_files_opened": False,
        "current_oof_validation_or_confirmatory_pnl_opened": False,
        "policy_pnl_computed": False,
        "pre_label_access_validation_passed": True,
        **_blocked_safety_fields(),
    }
    pre_label["audit_id"] = canonical_json_sha256(pre_label)
    pre_label_path = run_dir / "conformal_v6_pre_label_access_audit.json"
    _write_json_fsync(pre_label_path, pre_label)

    labeled_by_role, corpus_audits = _materialize_fit_and_calibration_rows(
        role_rows,
        feature_columns=feature_columns,
    )
    fit_rows = labeled_by_role[FIT_ROLE]
    calibration_rows = labeled_by_role[CALIBRATION_ROLE]
    _validate_labeled_role_rows(fit_rows, expected_market_count=150, role=FIT_ROLE)
    _validate_labeled_role_rows(
        calibration_rows,
        expected_market_count=60,
        role=CALIBRATION_ROLE,
    )
    if max(int(row["decision_ts"]) for row in fit_rows) >= min(
        int(row["decision_ts"]) for row in calibration_rows
    ):
        raise ValueError("point-model fit does not strictly precede conformal calibration")

    model_config = _xgb_model_config(dict(profile["point_model"]))
    booster = _train_regressor(
        fit_rows,
        feature_columns=feature_columns,
        model_config=model_config,
    )
    model_path = run_dir / MODEL_FILENAME
    booster.save_model(model_path)
    fit_predictions = _raw_target_stripped_predictions(
        booster,
        fit_rows,
        feature_columns=feature_columns,
    )
    calibration_predictions = _raw_target_stripped_predictions(
        booster,
        calibration_rows,
        feature_columns=feature_columns,
    )
    check_predictions = _target_free_predictions(
        booster,
        check_action_rows,
        feature_columns=feature_columns,
    )
    fit_predictions = attach_frozen_execution_compatibility(fit_predictions)
    calibration_predictions = attach_frozen_execution_compatibility(calibration_predictions)
    check_predictions = attach_frozen_execution_compatibility(check_predictions)

    calibration_artifact = build_policy_selected_conformal_artifact(
        calibration_predictions,
        target_rows=calibration_rows,
        profile=profile,
        feature_contract_sha256=config.expected_feature_contract_sha256,
    )
    scored_check = apply_policy_selected_conformal_scores(
        check_predictions,
        calibration_artifact=calibration_artifact,
        profile=profile,
    )
    check_selected = select_sequential_policy_rows(
        scored_check,
        score_field="conformal_net_return_lower_bound",
        require_positive=True,
    )
    check_support = _target_free_check_support(check_selected, profile=profile)
    calibration_gate = _calibration_gate(
        calibration_artifact=calibration_artifact,
        check_support=check_support,
        corpus_audits=corpus_audits,
    )

    fit_rows_path = run_dir / "conformal_v6_point_model_fit_action_rows.jsonl"
    calibration_rows_path = run_dir / "conformal_v6_calibration_action_rows.jsonl"
    fit_predictions_path = run_dir / "conformal_v6_target_stripped_fit_predictions.jsonl"
    calibration_predictions_path = (
        run_dir / "conformal_v6_target_stripped_calibration_predictions.jsonl"
    )
    check_predictions_path = run_dir / "conformal_v6_target_free_check_predictions.jsonl"
    check_scored_path = run_dir / "conformal_v6_target_free_check_scored_rows.jsonl"
    check_selected_path = run_dir / "conformal_v6_target_free_check_selected_rows.jsonl"
    artifact_path = run_dir / "conformal_v6_calibration_artifact.json"
    for path, rows in (
        (fit_rows_path, fit_rows),
        (calibration_rows_path, calibration_rows),
        (fit_predictions_path, fit_predictions),
        (calibration_predictions_path, calibration_predictions),
        (check_predictions_path, check_predictions),
        (check_scored_path, scored_check),
        (check_selected_path, check_selected),
    ):
        _write_jsonl_fsync(path, rows)
    _write_json_fsync(artifact_path, calibration_artifact)

    report = {
        "schema_version": f"{SCHEMA_PREFIX}-calibration-report-v1",
        "report_id": None,
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "point_model_fit_market_count": 150,
        "conformal_calibration_market_count": 60,
        "calibration_check_market_count": 50,
        "point_model_fit_action_row_count": len(fit_rows),
        "conformal_calibration_action_row_count": len(calibration_rows),
        "calibration_check_target_free_action_row_count": len(check_action_rows),
        "policy_selected_calibration": calibration_artifact,
        "target_free_calibration_check_support": check_support,
        "calibration_gate_checks": calibration_gate["checks"],
        "calibration_gate_passed": calibration_gate["passed"],
        "calibration_gate_blocking_reason_codes": calibration_gate["reason_codes"],
        "calibration_check_labels_opened_by_fit": False,
        "policy_pnl_computed_on_calibration": False,
        "policy_pnl_computed_on_calibration_check": False,
        "calibration_threshold_search_enabled": False,
        "uses_204_outcomes_for_fitting": False,
        "uses_204_pnl_for_tuning": False,
        "uses_current_oof_validation_or_confirmatory_pnl_for_tuning": False,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "conformal_v6_calibration_report.json"
    _write_json_fsync(report_path, report)
    _write_text_fsync(
        run_dir / "conformal_v6_calibration_report.md",
        _calibration_markdown(report),
    )

    candidate_frozen = calibration_gate["passed"]
    training_report = {
        "schema_version": f"{SCHEMA_PREFIX}-training-report-v1",
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "primary_policy_target": "action_expected_net_return",
        "action_value_model_family": "policy_selected_split_conformal_direct_net_return_model",
        "training_target_includes_costs": True,
        "model": _descriptor(model_path),
        "calibration_artifact": _descriptor(artifact_path),
        "calibration_gate_passed": candidate_frozen,
        "hyperparameter_search_enabled": False,
        "policy_pnl_computed": False,
        "calibration_check_labels_opened_by_fit": False,
        "future_files_opened": False,
        **_blocked_safety_fields(),
    }
    training_report["report_id"] = canonical_json_sha256(training_report)
    training_report_path = run_dir / "conformal_v6_training_report.json"
    _write_json_fsync(training_report_path, training_report)

    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-research-candidate-freeze-manifest-v1",
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        "candidate_freeze_created_ts": config.candidate_freeze_created_ts,
        "research_candidate_frozen": candidate_frozen,
        "research_candidate_only": True,
        "profile": _descriptor(profile_path),
        "pre_label_access_audit": _descriptor(pre_label_path),
        "development_settlement_manifest": _descriptor(settlement_path),
        "feature_contract": _descriptor(feature_contract_path),
        "point_model_fit_action_rows": _descriptor(fit_rows_path),
        "conformal_calibration_action_rows": _descriptor(calibration_rows_path),
        "target_stripped_fit_predictions": _descriptor(fit_predictions_path),
        "target_stripped_calibration_predictions": _descriptor(calibration_predictions_path),
        "target_free_calibration_check_predictions": _descriptor(check_predictions_path),
        "target_free_calibration_check_scored_rows": _descriptor(check_scored_path),
        "target_free_calibration_check_selected_rows": _descriptor(check_selected_path),
        "model": _descriptor(model_path),
        "calibration_artifact": _descriptor(artifact_path),
        "calibration_report": _descriptor(report_path),
        "training_report": _descriptor(training_report_path),
        "model_sha256": _sha256_file(model_path),
        "policy_dataset_hash": _sha256_file(fit_rows_path),
        "split_hash": canonical_json_sha256(
            {
                "settled_role_rows_sha256": settled_role_descriptor["sha256"],
                "roles": {FIT_ROLE: 150, CALIBRATION_ROLE: 60, CHECK_ROLE: 50},
            }
        ),
        "calibration_gate_passed": candidate_frozen,
        "candidate_specific_future_evaluation_allowed": candidate_frozen,
        "candidate_specific_future_evaluation_blocking_reason_codes": calibration_gate[
            "reason_codes"
        ],
        "uses_204_outcomes_for_fitting": False,
        "uses_204_pnl_for_tuning": False,
        "uses_current_oof_validation_or_confirmatory_pnl_for_tuning": False,
        "calibration_check_labels_opened_by_fit": False,
        "policy_pnl_computed": False,
        "future_files_opened": False,
        "result_driven_rerun_or_parameter_change_allowed": False,
        **_blocked_safety_fields(),
    }
    manifest["research_candidate_hash"] = canonical_json_sha256(
        {
            "candidate_name": CANDIDATE_NAME,
            "implementation_commit": config.implementation_commit,
            "model_sha256": manifest["model_sha256"],
            "calibration_artifact_sha256": manifest["calibration_artifact"]["sha256"],
            "policy_dataset_hash": manifest["policy_dataset_hash"],
            "split_hash": manifest["split_hash"],
        }
    )
    manifest_path = run_dir / "conformal_v6_research_candidate_freeze_manifest.json"
    _write_json_fsync(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "calibration_artifact": calibration_artifact,
        "calibration_report": report,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def attach_frozen_execution_compatibility(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach the unchanged p_up and execution-quality compatibility mask."""

    compatibility_rows = build_execution_compatible_action_universe(predictions)
    compatibility = {
        _row_key(row): bool(row["p_up_alignment_passed"] and row["execution_quality_only_passed"])
        for row in compatibility_rows
    }
    output = []
    for row in sorted(predictions, key=_row_sort_key):
        action = str(row["action"])
        compatible = action == "NO_TRADE" or compatibility[_row_key(row)]
        updated = {
            **row,
            "guard_compatible_before_ranking": compatible,
            "guard_compatibility_mask_applied_before_argmax": True,
            "target_used_as_decision_input": False,
            "outcome_fields_used_as_decision_input": False,
            "target_or_outcome_fields_used": False,
            **_blocked_safety_fields(),
        }
        updated["v6_raw_prediction_row_sha256"] = canonical_json_sha256(updated)
        output.append(updated)
    return output


def select_sequential_policy_rows(
    predictions: list[dict[str, Any]],
    *,
    score_field: str,
    require_positive: bool,
) -> list[dict[str, Any]]:
    """Select at most one causal trade per market from chronological decisions."""

    by_market_decision: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in predictions:
        by_market_decision[str(row["market_id"])][int(row["decision_ts"])].append(row)
    selected = []
    for market_id in sorted(by_market_decision):
        for decision_ts in sorted(by_market_decision[market_id]):
            decision_rows = by_market_decision[market_id][decision_ts]
            actions = {str(row["action"]) for row in decision_rows}
            if actions != set(REQUIRED_ACTIONS):
                raise ValueError("sequential policy decision group is not a five-action grid")
            candidates = [
                row
                for row in decision_rows
                if row["action"] in TRADE_ACTIONS and row["guard_compatible_before_ranking"]
            ]
            if not candidates:
                continue
            top = max(candidates, key=lambda row: (float(row[score_field]), str(row["action"])))
            if require_positive and float(top[score_field]) <= 0.0:
                continue
            selected.append(
                {
                    **top,
                    "sequential_policy_selected": True,
                    "later_decision_rows_visible_to_selection": False,
                    "selected_trade_index_within_market": 1,
                }
            )
            break
    selected.sort(key=lambda row: (int(row["decision_ts"]), str(row["market_id"])))
    if len({str(row["market_id"]) for row in selected}) != len(selected):
        raise ValueError("sequential policy selected multiple trades for a market")
    return selected


def build_policy_selected_conformal_artifact(
    calibration_predictions: list[dict[str, Any]],
    *,
    target_rows: list[dict[str, Any]],
    profile: dict[str, Any],
    feature_contract_sha256: str,
) -> dict[str, Any]:
    """Calibrate one-sided residuals on the frozen policy-selected row per market."""

    selected = select_sequential_policy_rows(
        calibration_predictions,
        score_field="raw_direct_predicted_net_return",
        require_positive=True,
    )
    targets = {_row_key(row): float(row["target_net_pnl_per_contract"]) for row in target_rows}
    if any(_row_key(row) not in targets for row in selected):
        raise ValueError("policy-selected calibration target identity missing")
    residual_rows = [
        {
            "market_id": str(row["market_id"]),
            "side": str(row["side"]),
            "action": str(row["action"]),
            "raw_prediction": float(row["raw_direct_predicted_net_return"]),
            "target": targets[_row_key(row)],
            "residual": float(row["raw_direct_predicted_net_return"]) - targets[_row_key(row)],
        }
        for row in selected
    ]
    calibration = profile["policy_selected_conformal_calibration"]
    alpha = float(calibration["one_sided_alpha"])
    global_group = _conformal_group(residual_rows, alpha=alpha, name="all_trade_sides")
    side_groups = {
        side: _conformal_group(
            [row for row in residual_rows if row["side"] == side],
            alpha=alpha,
            name=f"selected_side:{side}",
        )
        for side in SIDES
    }
    minimum_side = int(calibration["minimum_side_calibration_market_count"])
    minimum_global = int(calibration["minimum_global_calibration_market_count"])
    sides = {}
    for side in SIDES:
        if side_groups[side]["market_count"] >= minimum_side:
            group = side_groups[side]
            source = "selected_side"
        elif global_group["market_count"] >= minimum_global:
            group = global_group
            source = "all_trade_sides"
        else:
            group = {**global_group, "quantile": float("nan")}
            source = "insufficient_support_fail_closed"
        sides[side] = {
            "calibration_source": source,
            "calibration_group_name": group["group_name"],
            "calibration_penalty": float(group["quantile"]),
            "calibration_market_count": int(group["market_count"]),
            "quantile_rank": int(group["quantile_rank"]),
            "empirical_one_sided_coverage": float(group["empirical_one_sided_coverage"]),
            "support_passed": source != "insufficient_support_fail_closed",
        }
    finite_penalties = all(
        math.isfinite(float(row["calibration_penalty"]))
        and abs(float(row["calibration_penalty"])) <= 2.0
        for row in sides.values()
    )
    artifact = {
        "schema_version": f"{SCHEMA_PREFIX}-calibration-artifact-v1",
        "candidate_name": CANDIDATE_NAME,
        "source_split": CALIBRATION_ROLE,
        "method": calibration["method"],
        "decision_score_formula": "raw_direct_predicted_net_return - selected_side_penalty",
        "feature_contract_sha256": feature_contract_sha256,
        "selected_calibration_market_count": len(selected),
        "selected_side_distribution": dict(
            sorted(Counter(str(row["side"]) for row in selected).items())
        ),
        "selected_action_distribution": dict(
            sorted(Counter(str(row["action"]) for row in selected).items())
        ),
        "maximum_selected_trade_rows_per_market": 1,
        "later_decision_rows_visible_to_selection": False,
        "global_group": global_group,
        "side_groups": side_groups,
        "sides": sides,
        "finite_bounded_penalties": finite_penalties,
        "policy_pnl_computed_on_calibration": False,
        "calibration_threshold_search_enabled": False,
        "uses_204_outcomes_for_fitting": False,
        "uses_current_oof_validation_or_confirmatory_pnl_for_tuning": False,
        **_blocked_safety_fields(),
    }
    artifact["calibration_artifact_id"] = canonical_json_sha256(artifact)
    return artifact


def apply_policy_selected_conformal_scores(
    predictions: list[dict[str, Any]],
    *,
    calibration_artifact: dict[str, Any],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply the frozen side penalty to target-free decision rows."""

    if _find_nonempty_fields(predictions, TARGET_FIELDS):
        raise ValueError("v6 inference rows contain target fields")
    mask_score = -1_000_000.0
    output = []
    for row in sorted(predictions, key=_row_sort_key):
        action = str(row["action"])
        raw = 0.0 if action == "NO_TRADE" else float(row["raw_direct_predicted_net_return"])
        if action == "NO_TRADE":
            penalty = 0.0
            lower_bound = 0.0
            selection_score = 0.0
            source = "frozen_no_trade_zero_anchor"
        else:
            calibration = calibration_artifact["sides"][str(row["side"])]
            penalty = float(calibration["calibration_penalty"])
            lower_bound = raw - penalty
            if not row["guard_compatible_before_ranking"] or not math.isfinite(lower_bound):
                selection_score = mask_score
                source = "masked_by_frozen_execution_or_invalid_bound"
            else:
                selection_score = lower_bound
                source = "policy_selected_split_conformal_net_return_lcb"
        updated = {
            **row,
            "conformal_calibration_source": (
                "frozen_no_trade_zero_anchor"
                if action == "NO_TRADE"
                else calibration["calibration_source"]
            ),
            "conformal_calibration_penalty": penalty,
            "conformal_net_return_lower_bound": lower_bound,
            "action_selection_score": selection_score,
            "action_advantage_lcb_net_return": selection_score,
            "calibrated_action_expected_net_return": raw,
            "ranking_score_source": source,
            "target_used_as_decision_input": False,
            "outcome_fields_used_as_decision_input": False,
            "target_or_outcome_fields_used": False,
            **_blocked_safety_fields(),
        }
        updated["v6_prediction_row_sha256"] = canonical_json_sha256(updated)
        output.append(updated)
    return output


def _validate_settlement_manifest_for_fit(manifest: dict[str, Any]) -> None:
    blockers = []
    if manifest.get("schema_version") != DEVELOPMENT_SETTLEMENT_MANIFEST_SCHEMA_VERSION:
        blockers.append("development_settlement_manifest_schema_invalid")
    if manifest.get("development_settled_corpus_ready") is not True:
        blockers.append("development_settled_corpus_not_ready")
    if manifest.get("blocking_reason_codes") != []:
        blockers.append("development_settlement_manifest_has_blockers")
    if manifest.get("policy_pnl_computed") is not False:
        blockers.append("development_policy_pnl_was_computed")
    if manifest.get("source_outcome_blind_rounds_mutated") is not False:
        blockers.append("source_outcome_blind_rounds_were_mutated")
    if manifest.get("direct_training_corpus_exported") is not False:
        blockers.append("development_corpus_exported_before_candidate_gate")
    for key, expected in _blocked_safety_fields().items():
        if manifest.get(key) != expected:
            blockers.append(f"development_settlement_safety_invalid:{key}")
    if blockers:
        raise ValueError("development settlement manifest invalid: " + ", ".join(blockers))


def _validate_settled_role_lineage(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 260:
        raise ValueError("settled development role row count must be 260")
    if len({str(row.get("market_id") or "") for row in rows}) != 260:
        raise ValueError("settled development market identities must be unique")
    expected_roles = {FIT_ROLE: 150, CALIBRATION_ROLE: 60, CHECK_ROLE: 50}
    role_counts = Counter(str(row.get("development_role") or "") for row in rows)
    if dict(role_counts) != expected_roles:
        raise ValueError("settled development role counts do not match frozen protocol")
    if [int(row.get("selection_rank") or 0) for row in rows] != list(range(1, 261)):
        raise ValueError("settled development selection ranks are not contiguous")
    sequences = [int(row.get("sequence") or 0) for row in rows]
    if min(sequences) < 237 or sequences != sorted(sequences):
        raise ValueError("settled development rows cross the #204 source boundary")
    role_order = [str(row["development_role"]) for row in rows]
    if role_order != [FIT_ROLE] * 150 + [CALIBRATION_ROLE] * 60 + [CHECK_ROLE] * 50:
        raise ValueError("settled development roles are not chronological")
    for row in rows:
        if row.get("outcomes_used_as_training_targets_only") is not True:
            raise ValueError("settled outcomes are not restricted to training targets")
        if row.get("outcomes_used_as_decision_inputs") is not False:
            raise ValueError("settled outcome marked as decision input")
        if row.get("policy_pnl_computed") is not False:
            raise ValueError("policy PnL was computed in development settlement")
        if row.get("source_outcome_blind_round_mutated") is not False:
            raise ValueError("source outcome-blind round was mutated")
        _verified_descriptor(row.get("corpus_manifest"), "settled corpus manifest")


def _validate_target_free_check_rows(
    rows: list[dict[str, Any]],
    *,
    feature_columns: tuple[str, ...],
) -> None:
    if _find_nonempty_fields(rows, TARGET_FIELDS):
        raise ValueError("calibration-check rows contain target fields")
    market_ids = {str(row.get("market_id") or "") for row in rows}
    if len(market_ids) != 50 or "" in market_ids:
        raise ValueError("calibration-check must contain exactly 50 markets")
    groups: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows:
        if row.get("development_role") != CHECK_ROLE:
            raise ValueError("calibration-check role mismatch")
        decision_ts = int(row.get("decision_ts") or 0)
        if int(row.get("max_input_ts") or 0) > decision_ts or decision_ts <= 0:
            raise ValueError("calibration-check feature causality violation")
        features = dict(row.get("decision_time_features") or {})
        if set(feature_columns) - set(features):
            raise ValueError("calibration-check decision-time feature missing")
        if not all(math.isfinite(float(features[name])) for name in feature_columns):
            raise ValueError("calibration-check feature is not finite")
        groups[(str(row["market_id"]), decision_ts)].add(str(row.get("action") or ""))
    if any(actions != set(REQUIRED_ACTIONS) for actions in groups.values()):
        raise ValueError("calibration-check action grid is incomplete")


def _materialize_fit_and_calibration_rows(
    role_rows: list[dict[str, Any]],
    *,
    feature_columns: tuple[str, ...],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_role: dict[str, list[dict[str, Any]]] = {FIT_ROLE: [], CALIBRATION_ROLE: []}
    audits = []
    for role_row in role_rows:
        role = str(role_row["development_role"])
        if role == CHECK_ROLE:
            continue
        if role not in by_role:
            raise ValueError(f"unexpected development role: {role}")
        corpus_dir = Path(str(role_row["source_corpus_dir"])).resolve()
        action_rows, audit = _load_corpus_action_rows(
            corpus_dir,
            role_row={
                **role_row,
                "role": role,
                "selection_rank": int(role_row["selection_rank"]),
            },
            feature_columns=feature_columns,
        )
        action_rows = [
            {
                **row,
                "development_role": role,
                "development_role_index": int(role_row["development_role_index"]),
            }
            for row in action_rows
        ]
        by_role[role].extend(action_rows)
        audits.append({**audit, "development_role": role})
    for role in by_role:
        by_role[role].sort(key=_row_sort_key)
    return by_role, audits


def _validate_labeled_role_rows(
    rows: list[dict[str, Any]],
    *,
    expected_market_count: int,
    role: str,
) -> None:
    market_ids = {str(row.get("market_id") or "") for row in rows}
    if len(market_ids) != expected_market_count or "" in market_ids:
        raise ValueError(f"{role} market support does not match frozen role")
    groups: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows:
        if row.get("development_role") != role or row.get("role") != role:
            raise ValueError(f"{role} row role mismatch")
        decision_ts = int(row.get("decision_ts") or 0)
        if int(row.get("max_input_ts") or 0) > decision_ts or decision_ts <= 0:
            raise ValueError(f"{role} feature causality violation")
        target = row.get("target_net_pnl_per_contract")
        if not isinstance(target, int | float) or not math.isfinite(float(target)):
            raise ValueError(f"{role} target is missing or invalid")
        groups[(str(row["market_id"]), decision_ts)].add(str(row.get("action") or ""))
    if any(actions != set(REQUIRED_ACTIONS) for actions in groups.values()):
        raise ValueError(f"{role} action grid is incomplete")
    if {str(row["market_id"]) for row in rows} != market_ids:
        raise ValueError(f"{role} market identity invalid")


def _xgb_model_config(point_model: dict[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "objective",
        "eval_metric",
        "target",
        "training_target_includes_costs",
        "decision_time_features_only",
        "num_boost_round",
        "max_depth",
        "eta",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "seed",
        "nthread",
        "verbosity",
        "hyperparameter_search_enabled",
    }
    if set(point_model) != expected_keys:
        raise ValueError("v6 point-model config differs from preregistration")
    if point_model["target"] != "target_net_pnl_per_contract":
        raise ValueError("v6 point-model target is invalid")
    if point_model["training_target_includes_costs"] is not True:
        raise ValueError("v6 point-model target must include costs")
    if point_model["decision_time_features_only"] is not True:
        raise ValueError("v6 point-model inputs must be decision-time only")
    if point_model["hyperparameter_search_enabled"] is not False:
        raise ValueError("v6 hyperparameter search must remain disabled")
    return dict(point_model)


def _target_free_predictions(
    booster: Any,
    rows: list[dict[str, Any]],
    *,
    feature_columns: tuple[str, ...],
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=_row_sort_key)
    values = _predict_regressor(booster, ordered, feature_columns=feature_columns)
    output = []
    for row, value in zip(ordered, values, strict=True):
        action = str(row["action"])
        updated = {
            **row,
            "raw_model_prediction": value,
            "raw_direct_predicted_net_return": 0.0 if action == "NO_TRADE" else value,
            "raw_prediction_source": (
                "frozen_no_trade_zero_anchor"
                if action == "NO_TRADE"
                else "fit_role_only_direct_net_return_model"
            ),
            "target_used_as_decision_input": False,
            "outcome_fields_used_as_decision_input": False,
            "target_or_outcome_fields_used": False,
        }
        updated["v6_raw_prediction_row_sha256"] = canonical_json_sha256(updated)
        output.append(updated)
    if _find_nonempty_fields(output, TARGET_FIELDS):
        raise ValueError("target-free predictions contain target fields")
    return output


def _target_free_check_support(
    selected_rows: list[dict[str, Any]],
    *,
    profile: dict[str, Any],
) -> dict[str, Any]:
    minimum = int(
        profile["policy_selected_conformal_calibration"][
            "minimum_calibration_check_selected_market_count_per_side"
        ]
    )
    side_counts = Counter(str(row["side"]) for row in selected_rows)
    selected_markets = {str(row["market_id"]) for row in selected_rows}
    checks = {
        "target_free_rows": not _find_nonempty_fields(selected_rows, TARGET_FIELDS),
        "maximum_one_selected_trade_per_market": len(selected_markets) == len(selected_rows),
        "minimum_selected_market_count_per_side": all(
            side_counts.get(side, 0) >= minimum for side in SIDES
        ),
    }
    return {
        "selected_market_count": len(selected_markets),
        "selected_side_market_counts": {side: side_counts.get(side, 0) for side in SIDES},
        "minimum_required_per_side": minimum,
        "checks": checks,
        "passed": all(checks.values()),
        "labels_outcomes_or_pnl_opened": False,
        "policy_pnl_computed": False,
    }


def _calibration_gate(
    *,
    calibration_artifact: dict[str, Any],
    check_support: dict[str, Any],
    corpus_audits: list[dict[str, Any]],
) -> dict[str, Any]:
    checks = {
        "selected_calibration_market_support": (
            int(calibration_artifact["selected_calibration_market_count"]) >= 50
        ),
        "selected_side_or_global_calibration_support": all(
            calibration_artifact["sides"][side]["support_passed"] for side in SIDES
        ),
        "finite_bounded_calibration_penalties": calibration_artifact[
            "finite_bounded_penalties"
        ]
        is True,
        "nominal_one_sided_coverage": all(
            float(calibration_artifact["sides"][side]["empirical_one_sided_coverage"])
            >= 0.9
            for side in SIDES
        ),
        "target_free_calibration_check_support": check_support["passed"] is True,
        "corpus_integrity": all(not audit["blocking_reason_codes"] for audit in corpus_audits),
        "no_calibration_or_check_policy_pnl": (
            calibration_artifact["policy_pnl_computed_on_calibration"] is False
            and check_support["policy_pnl_computed"] is False
        ),
        "no_threshold_search": calibration_artifact["calibration_threshold_search_enabled"]
        is False,
        "no_issue204_or_current_pnl_tuning": (
            calibration_artifact["uses_204_outcomes_for_fitting"] is False
            and calibration_artifact[
                "uses_current_oof_validation_or_confirmatory_pnl_for_tuning"
            ]
            is False
        ),
    }
    reason_map = {
        "selected_calibration_market_support": "selected_calibration_market_support_failed",
        "selected_side_or_global_calibration_support": "side_or_global_calibration_support_failed",
        "finite_bounded_calibration_penalties": "calibration_penalty_invalid_or_unbounded",
        "nominal_one_sided_coverage": "nominal_one_sided_coverage_failed",
        "target_free_calibration_check_support": "target_free_calibration_check_support_failed",
        "corpus_integrity": "development_corpus_integrity_failed",
        "no_calibration_or_check_policy_pnl": "development_policy_pnl_was_computed",
        "no_threshold_search": "calibration_threshold_search_enabled",
        "no_issue204_or_current_pnl_tuning": "prohibited_outcome_or_pnl_tuning_detected",
    }
    reasons = [reason_map[name] for name, passed in checks.items() if not passed]
    return {"checks": checks, "passed": not reasons, "reason_codes": reasons}


def _conformal_group(
    rows: list[dict[str, Any]],
    *,
    alpha: float,
    name: str,
) -> dict[str, Any]:
    by_market: dict[str, float] = {}
    for row in rows:
        market_id = str(row["market_id"])
        residual = float(row["residual"])
        if market_id in by_market:
            raise ValueError("policy-selected conformal group contains duplicate market")
        if not math.isfinite(residual):
            raise ValueError("policy-selected conformal residual is not finite")
        by_market[market_id] = residual
    residuals = [by_market[market_id] for market_id in sorted(by_market)]
    market_count = len(residuals)
    if market_count == 0:
        return {
            "group_name": name,
            "market_count": 0,
            "quantile_rank": 0,
            "quantile": float("nan"),
            "empirical_one_sided_coverage": 0.0,
        }
    rank = min(market_count, math.ceil((market_count + 1) * (1.0 - alpha)))
    quantile = float(sorted(residuals)[rank - 1])
    coverage = sum(value <= quantile for value in residuals) / market_count
    return {
        "group_name": name,
        "market_count": market_count,
        "quantile_rank": rank,
        "quantile": quantile,
        "empirical_one_sided_coverage": coverage,
    }


def _calibration_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# #207 policy-selected conformal v6 calibration",
        "",
        f"- calibration gate passed: `{report['calibration_gate_passed']}`",
        "- point-model fit markets: `150`",
        "- policy-selected calibration markets: "
        f"`{report['policy_selected_calibration']['selected_calibration_market_count']}`",
        "- calibration-check labels opened: `false`",
        "- policy PnL computed: `false`",
        "- threshold search enabled: `false`",
        "",
        "| Side | Source | Markets | Penalty | Coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for side in SIDES:
        row = report["policy_selected_calibration"]["sides"][side]
        lines.append(
            "| {side} | {source} | {markets} | {penalty:.8f} | {coverage:.6f} |".format(
                side=side,
                source=row["calibration_source"],
                markets=row["calibration_market_count"],
                penalty=float(row["calibration_penalty"]),
                coverage=float(row["empirical_one_sided_coverage"]),
            )
        )
    lines.append("")
    return "\n".join(lines)
