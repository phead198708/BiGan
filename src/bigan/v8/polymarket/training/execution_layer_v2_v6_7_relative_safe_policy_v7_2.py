"""Frozen-v6.7-relative safe policy improvement for issue #234."""

from __future__ import annotations

import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_abstention_aware_expected_net_pnl_v7_0 import (
    _v7_0_blocked_safety_fields,
)
from bigan.v8.polymarket.training.execution_layer_v2_abstention_aware_expected_net_pnl_v7_0_fit import (
    FEATURE_NAMES,
    HTS_ACTIONS,
    SBC_ACTIONS,
    materialize_v7_0_sbc_rows,
    validate_v7_0_training_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7 import (
    validate_p_up_semantic_compatibility_v6_7_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
    _descriptor,
    _load_json,
    _load_jsonl,
    _require_git_sha,
    _require_sha256,
    _sha256_file,
    _verify_pin,
    _write_json,
    _write_jsonl,
    _write_text,
)

CANDIDATE_NAME = "frozen_v6_7_relative_safe_policy_v7_2"
PROFILE_SCHEMA_VERSION = "bigan-v8-v6-7-relative-safe-policy-v7-2-profile-v1"
MODEL_SCHEMA_VERSION = "bigan-v8-v6-7-relative-safe-policy-v7-2-model-v1"
REPORT_SCHEMA_VERSION = "bigan-v8-v6-7-relative-safe-policy-v7-2-report-v1"
LEAKAGE_SCHEMA_VERSION = "bigan-v8-v6-7-relative-safe-policy-v7-2-leakage-v1"
MANIFEST_SCHEMA_VERSION = "bigan-v8-v6-7-relative-safe-policy-v7-2-manifest-v1"
POLICY_DECISIONS = ("KEEP_V6_7", "SWITCH_SAME_DECISION_SBC", "NO_TRADE")
FORBIDDEN_INFERENCE_FIELDS = {
    "resolved_outcome",
    "winning_outcome",
    "settlement_pnl",
    "realized_trade_pnl",
    "total_polymarket_pnl",
    "runtime_policy_after_cost_net_pnl_per_contract",
    "runtime_policy_after_cost_net_pnl_at_frozen_size",
    "target_after_cost_net_pnl_per_contract",
    "future_return",
    "label",
    "oracle_action",
}
FROZEN_LINEAGE = {
    "runtime_target_rows_sha256": (
        "1565116daeb2f5d4d8c33fefa507276f59251edd5ffb5f4f313041bcf9dbb0ec"
    ),
    "v7_0_training_profile_sha256": (
        "1f66d8699b9727651538cc34a9a2a25ba5eaac5cfded75cf8f4a258b1b5d3f4a"
    ),
    "v6_7_candidate_profile_sha256": (
        "cec55d243acd6bbf60a5e8474545b487086ddcd4d18073682ae7f2d4660d2248"
    ),
    "prior_rejected_candidate_implementation_commit": (
        "24db43f27348f6e0432febb30f2443f98682a223"
    ),
    "prior_rejected_candidate_evidence_commit": (
        "6ca8af575bc5b7c228527e1b5020e0e84a765c22"
    ),
}


@dataclass(frozen=True, slots=True)
class V67RelativeSafePolicyV72Config:
    """Pinned inputs for the one historical v7.2 fit and replay."""

    run_id: str
    output_dir: Path | str
    profile_path: Path | str
    expected_profile_sha256: str
    v7_0_training_profile_path: Path | str
    v6_7_candidate_profile_path: Path | str
    runtime_target_rows_path: Path | str
    implementation_commit: str
    fit_created_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(self.expected_profile_sha256, "expected_profile_sha256")
        _require_git_sha(self.implementation_commit)
        if self.fit_created_ts <= 0:
            raise ValueError("fit_created_ts must be positive")
        for name in (
            "output_dir",
            "profile_path",
            "v7_0_training_profile_path",
            "v6_7_candidate_profile_path",
            "runtime_target_rows_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


def validate_v6_7_relative_safe_policy_v7_2_profile(
    profile: dict[str, Any],
) -> None:
    """Reject any drift from the pre-implementation #234 contract."""

    split = dict(profile.get("historical_split") or {})
    features = dict(profile.get("feature_contract") or {})
    models = dict(profile.get("incremental_advantage_models") or {})
    conformal = dict(profile.get("conformal_contract") or {})
    replay = dict(profile.get("historical_replay_superiority_gate") or {})
    canary = dict(profile.get("target_free_canary") or {})
    checks = {
        "identity": profile.get("schema_version") == PROFILE_SCHEMA_VERSION
        and profile.get("issue_number") == 234
        and profile.get("candidate_name") == CANDIDATE_NAME
        and profile.get("preregistered_before_implementation_and_fit") is True,
        "lineage": profile.get("lineage") == FROZEN_LINEAGE,
        "prior_exclusion": profile.get("prior_result_exclusion")
        == {
            "issue233_result_artifacts_opened_for_fit_or_tuning": False,
            "issue233_oof_rows_used": False,
            "issue233_selected_rows_used": False,
            "issue233_pnl_or_action_loss_examples_used": False,
            "issue229_or_issue231_future_outcomes_used": False,
            "only_structural_prior_used": "preserve_frozen_v6_7_as_default_policy",
        },
        "baseline": profile.get("baseline_contract")
        == {
            "candidate_name": "p_up_semantic_execution_compatibility_v6_7",
            "score_source": "frozen_v6_2_market_clustered_mean_ev_lcb",
            "selection_rule": (
                "highest_positive_score_per_market_then_earliest_decision_ts_"
                "then_action"
            ),
            "no_positive_score_behavior": "NO_TRADE",
            "source_score_mutation_allowed": False,
            "baseline_is_default_action": True,
        },
        "actions": profile.get("action_contract")
        == {
            "enabled_action_family": "SELL_BEFORE_CLOSE",
            "allowed_policy_decisions": list(POLICY_DECISIONS),
            "enabled_trade_actions": list(SBC_ACTIONS),
            "disabled_action_family": "HOLD_TO_SETTLEMENT",
            "disabled_trade_actions": list(HTS_ACTIONS),
            "no_trade_action": "NO_TRADE",
            "full_five_action_interface_required": True,
            "opposite_action_must_share_baseline_decision_group": True,
            "alternative_decision_timestamp_allowed": False,
            "maximum_bets_per_market": 1,
            "side_quota_allowed": False,
            "side_pnl_hard_gate_enabled": False,
        },
        "split": split
        == {
            "source_market_count": 134,
            "market_order": "minimum_decision_ts_then_market_id",
            "initial_training_market_count": 44,
            "forward_fold_count": 5,
            "forward_validation_market_count_per_fold": 18,
            "forward_oof_market_count": 90,
            "fold_local_conformal_calibration_tail_market_count": 10,
            "validation_labels_used_for_fold_model_or_correction": False,
            "all_historical_markets_used_for_final_model_fit": True,
            "final_conformal_source": "fixed_rolling_oof_residuals",
        },
        "features": tuple(features.get("base_feature_names") or ()) == FEATURE_NAMES
        and features.get("switch_head_feature_construction")
        == (
            "baseline_features_plus_opposite_features_plus_opposite_minus_baseline"
        )
        and features.get("abstain_head_feature_construction")
        == "baseline_features_only"
        and features.get("fit_only_standardization") is True
        and set(features.get("forbidden_inference_field_names") or ())
        == FORBIDDEN_INFERENCE_FIELDS,
        "models": models
        == {
            "heads": ["SWITCH_SAME_DECISION_SBC", "NO_TRADE"],
            "model_family": "weighted_l2_ridge_with_unpenalized_intercept",
            "ridge_alpha": 100.0,
            "coefficient_absolute_bound": 8.0,
            "switch_target": (
                "opposite_same_decision_target_minus_frozen_v6_7_target"
            ),
            "no_trade_target": "zero_minus_frozen_v6_7_target",
            "market_weighting": "one_baseline_selected_example_per_market",
            "hyperparameter_search_enabled": False,
            "feature_search_enabled": False,
            "threshold_search_enabled": False,
            "result_selected_rerun_allowed": False,
        },
        "conformal": conformal
        == {
            "coverage_level": 0.8,
            "lower_residual_quantile": 0.2,
            "fold_correction_source": "latest_10_prior_training_markets_only",
            "final_correction_source": "all_fixed_rolling_oof_residuals",
            "upward_correction_allowed": False,
            "applied_correction": "min(weighted_lower_residual_quantile,0)",
            "selection_threshold": 0.0,
            "threshold_operator": "strictly_greater_than",
            "positive_advantage_tie_break": (
                "higher_lcb_then_policy_decision_lexicographic"
            ),
            "minimum_fold_calibration_selected_market_count": 5,
            "minimum_oof_market_count": 90,
        },
        "replay": replay
        == {
            "exact_evaluation_market_count": 90,
            "evaluation_market_source": "fixed_rolling_oof_market_cohort",
            "no_bet_market_pnl": 0.0,
            "common_selected_row_filter_allowed": False,
            "fixed_position_size": 0.2,
            "primary_metric": "total_after_cost_net_pnl_at_frozen_size",
            "candidate_minus_v6_7_total_pnl_minimum_exclusive": 0.0,
            "candidate_minus_v6_7_largest_winner_removed_pnl_minimum_inclusive": 0.0,
            "exact_baseline_market_and_action_identity_reconciliation_required": True,
            "historical_training_targets_used_for_fixed_coefficient_fit": True,
            "historical_oof_or_validation_pnl_used_for_feature_hyperparameter_or_threshold_tuning": False,
            "historical_pnl_used_for_pre_collection_screening_only": True,
            "gate_failure_or_policy_identity_stops_before_collection": True,
            "promotion_or_paper_unlock_allowed": False,
        },
        "canary": canary
        == {
            "historical_replay_superiority_gate_must_pass_before_collection": True,
            "new_strictly_later_outcome_blind_market_count": 12,
            "maximum_attempt_count": 18,
            "minimum_guard_accepted_unique_market_count_to_continue": 1,
            "minimum_policy_difference_market_count_to_continue": 1,
            "outcome_resolution_label_or_pnl_access_allowed": False,
            "full_execution_guard_unchanged": True,
        },
        "safety": profile.get("safety") == _v7_0_blocked_safety_fields(),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("#234 v7.2 profile invalid: " + ", ".join(blockers))


def fit_v6_7_relative_safe_policy_v7_2(
    *,
    rows: list[dict[str, Any]],
    profile: dict[str, Any],
    implementation_commit: str,
    fit_created_ts: int,
) -> dict[str, Any]:
    """Fit fixed rolling relative-advantage heads and run the historical gate."""

    validate_v6_7_relative_safe_policy_v7_2_profile(profile)
    _validate_canonical_rows(rows)
    market_order = _market_order(rows)
    examples = _relative_examples(rows, market_order=market_order)
    example_by_market = {row["market_id"]: row for row in examples}
    split = profile["historical_split"]
    initial = int(split["initial_training_market_count"])
    width = int(split["forward_validation_market_count_per_fold"])
    tail = int(split["fold_local_conformal_calibration_tail_market_count"])
    oof_rows = []
    fold_reports = []
    all_models_valid = True
    all_calibration_support_valid = True
    for fold_index in range(int(split["forward_fold_count"])):
        train_order = market_order[: initial + fold_index * width]
        validation_order = market_order[
            initial + fold_index * width : initial + (fold_index + 1) * width
        ]
        if len(validation_order) != width:
            raise ValueError("#234 rolling validation support invalid")
        fit_order = train_order[:-tail]
        calibration_order = train_order[-tail:]
        if _market_max_ts(rows, fit_order) >= _market_min_ts(rows, calibration_order):
            raise ValueError("#234 fit/calibration chronology invalid")
        if _market_max_ts(rows, calibration_order) >= _market_min_ts(
            rows, validation_order
        ):
            raise ValueError("#234 calibration/validation chronology invalid")
        fit_examples = _active_examples(example_by_market, fit_order)
        calibration_examples = _active_examples(example_by_market, calibration_order)
        switch_model = _fit_ridge_head(
            fit_examples,
            feature_field="switch_features",
            target_field="switch_advantage_target",
            profile=profile,
        )
        no_trade_model = _fit_ridge_head(
            fit_examples,
            feature_field="baseline_features",
            target_field="no_trade_advantage_target",
            profile=profile,
        )
        all_models_valid = all_models_valid and switch_model[
            "coefficients_finite_and_bounded"
        ] and no_trade_model["coefficients_finite_and_bounded"]
        minimum_calibration = int(
            profile["conformal_contract"][
                "minimum_fold_calibration_selected_market_count"
            ]
        )
        calibration_support_valid = len(calibration_examples) >= minimum_calibration
        all_calibration_support_valid = (
            all_calibration_support_valid and calibration_support_valid
        )
        corrections = _head_corrections(
            calibration_examples,
            switch_model=switch_model,
            no_trade_model=no_trade_model,
            profile=profile,
            support_valid=calibration_support_valid,
        )
        for market_id in validation_order:
            oof_rows.append(
                _score_historical_example(
                    example_by_market[market_id],
                    switch_model=switch_model,
                    no_trade_model=no_trade_model,
                    corrections=corrections,
                    fold_index=fold_index,
                )
            )
        fold_reports.append(
            {
                "fold_index": fold_index,
                "head_fit_market_count": len(fit_order),
                "head_fit_baseline_active_market_count": len(fit_examples),
                "conformal_calibration_market_count": len(calibration_order),
                "conformal_calibration_baseline_active_market_count": len(
                    calibration_examples
                ),
                "validation_market_count": len(validation_order),
                "validation_labels_used_for_fold_model_or_correction": False,
                "calibration_support_valid": calibration_support_valid,
                "switch_correction": corrections["switch"],
                "no_trade_correction": corrections["no_trade"],
                "fit_max_decision_ts": _market_max_ts(rows, fit_order),
                "calibration_min_decision_ts": _market_min_ts(
                    rows, calibration_order
                ),
                "calibration_max_decision_ts": _market_max_ts(
                    rows, calibration_order
                ),
                "validation_min_decision_ts": _market_min_ts(
                    rows, validation_order
                ),
            }
        )
    if len({row["market_id"] for row in oof_rows}) != int(
        split["forward_oof_market_count"]
    ):
        raise ValueError("#234 OOF market support invalid")

    active_examples = [row for row in examples if row["baseline_trade_selected"]]
    final_switch_model = _fit_ridge_head(
        active_examples,
        feature_field="switch_features",
        target_field="switch_advantage_target",
        profile=profile,
    )
    final_no_trade_model = _fit_ridge_head(
        active_examples,
        feature_field="baseline_features",
        target_field="no_trade_advantage_target",
        profile=profile,
    )
    final_corrections = _oof_head_corrections(oof_rows, profile=profile)
    replay = _historical_replay(oof_rows, profile=profile)
    candidate_selected_rows = replay.pop("candidate_selected_rows")
    v6_7_baseline_selected_rows = replay.pop("v6_7_baseline_selected_rows")
    policy_difference_count = sum(
        row["selected_policy_decision"] != "KEEP_V6_7" for row in oof_rows
    )
    checks = {
        "models_finite_and_bounded": all_models_valid
        and final_switch_model["coefficients_finite_and_bounded"]
        and final_no_trade_model["coefficients_finite_and_bounded"],
        "fold_local_calibration_support": all_calibration_support_valid,
        "feature_timestamp_causality": all(
            int(row["max_input_ts"]) <= int(row["decision_ts"]) for row in rows
        ),
        "exact_oof_market_support": len({row["market_id"] for row in oof_rows})
        == int(profile["conformal_contract"]["minimum_oof_market_count"]),
        "validation_labels_excluded_from_fold_fit_and_correction": all(
            row["validation_labels_used_for_fold_model_or_correction"] is False
            for row in fold_reports
        ),
        "same_decision_switch_only": all(
            row["opposite_decision_ts"] in {None, row["baseline_decision_ts"]}
            for row in oof_rows
        ),
        "policy_difference_nonzero": policy_difference_count > 0,
        "candidate_total_pnl_strictly_better_than_v6_7": replay[
            "candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size"
        ]
        > float(
            profile["historical_replay_superiority_gate"][
                "candidate_minus_v6_7_total_pnl_minimum_exclusive"
            ]
        ),
        "candidate_largest_winner_removed_not_worse_than_v6_7": replay[
            "candidate_minus_v6_7_largest_winner_removed_after_cost_net_pnl_at_frozen_size"
        ]
        >= float(
            profile["historical_replay_superiority_gate"][
                "candidate_minus_v6_7_largest_winner_removed_pnl_minimum_inclusive"
            ]
        ),
        "baseline_identity_reconciled": replay["baseline_identity_reconciled"],
    }
    reason_map = {
        "models_finite_and_bounded": "relative_advantage_model_invalid",
        "fold_local_calibration_support": "fold_local_conformal_support_insufficient",
        "feature_timestamp_causality": "decision_time_feature_causality_failed",
        "exact_oof_market_support": "historical_oof_market_support_invalid",
        "validation_labels_excluded_from_fold_fit_and_correction": (
            "validation_label_isolation_failed"
        ),
        "same_decision_switch_only": "alternative_decision_timestamp_used",
        "policy_difference_nonzero": "candidate_identical_to_v6_7",
        "candidate_total_pnl_strictly_better_than_v6_7": (
            "historical_same_dataset_candidate_pnl_not_strictly_better_than_v6_7"
        ),
        "candidate_largest_winner_removed_not_worse_than_v6_7": (
            "historical_same_dataset_largest_winner_removed_pnl_worse_than_v6_7"
        ),
        "baseline_identity_reconciled": "frozen_v6_7_baseline_identity_mismatch",
    }
    blockers = [reason_map[name] for name, passed in checks.items() if not passed]
    model = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": implementation_commit,
        "fit_created_ts": fit_created_ts,
        "frozen": not blockers,
        "decision_time_safe": True,
        "baseline_candidate_name": "p_up_semantic_execution_compatibility_v6_7",
        "baseline_is_default_action": True,
        "allowed_policy_decisions": list(POLICY_DECISIONS),
        "enabled_trade_actions": list(SBC_ACTIONS),
        "disabled_trade_actions": list(HTS_ACTIONS),
        "disabled_action_reason_code": (
            "prior_preregistered_hts_prediction_loss_gate_failed"
        ),
        "switch_head": final_switch_model,
        "no_trade_head": final_no_trade_model,
        "final_conformal_corrections": final_corrections,
        "fold_reports": fold_reports,
        "historical_replay_superiority_gate": replay,
        "historical_gate_checks": checks,
        "historical_gate_passed": not blockers,
        "historical_gate_blocking_reason_codes": blockers,
        "historical_policy_difference_market_count": policy_difference_count,
        "issue233_result_artifacts_opened_for_fit_or_tuning": False,
        "issue229_or_issue231_future_outcomes_used": False,
        "historical_training_targets_used_for_fixed_coefficient_fit": True,
        "historical_oof_or_validation_pnl_used_for_feature_hyperparameter_or_threshold_tuning": False,
        "historical_pnl_used_for_pre_collection_screening_only": True,
        "historical_replay_is_promotion_evidence": False,
        "target_free_canary_collection_allowed": not blockers,
        "target_free_canary_started": False,
        "future_confirmatory_authorized": False,
        **_v7_0_blocked_safety_fields(),
    }
    model["model_artifact_id"] = canonical_json_sha256(model)
    return {
        "model_artifact": model,
        "oof_rows": sorted(oof_rows, key=_row_sort_key),
        "candidate_selected_rows": candidate_selected_rows,
        "v6_7_baseline_selected_rows": v6_7_baseline_selected_rows,
    }


def score_v6_7_relative_safe_policy_v7_2_market(
    rows: list[dict[str, Any]], *, model_artifact: dict[str, Any]
) -> dict[str, Any]:
    """Apply the frozen relative policy to one outcome-free market grid."""

    reasons = []
    if model_artifact.get("schema_version") != MODEL_SCHEMA_VERSION:
        reasons.append("v7_2_model_artifact_schema_invalid")
    if model_artifact.get("historical_gate_passed") is not True:
        reasons.append("v7_2_historical_gate_not_passed")
    if not rows or len({str(row.get("market_id") or "") for row in rows}) != 1:
        reasons.append("v7_2_market_identity_invalid")
    if any(FORBIDDEN_INFERENCE_FIELDS.intersection(row) for row in rows):
        reasons.append("v7_2_forbidden_outcome_field_in_inference_row")
    if reasons:
        return _no_trade_inference(rows, reasons=reasons)
    _validate_target_free_market_rows(rows)
    baseline = _select_baseline(rows)
    if baseline is None:
        return _no_trade_inference(
            rows, reasons=["frozen_v6_7_no_positive_baseline_action"]
        )
    opposite = _same_decision_opposite(rows, baseline)
    example = _inference_example(baseline, opposite)
    switch_prediction = _predict_head(
        example["switch_features"], model_artifact["switch_head"]
    )
    no_trade_prediction = _predict_head(
        example["baseline_features"], model_artifact["no_trade_head"]
    )
    corrections = model_artifact["final_conformal_corrections"]
    switch_lcb = switch_prediction + float(corrections["switch"])
    no_trade_lcb = no_trade_prediction + float(corrections["no_trade"])
    policy_decision = _select_policy_decision(switch_lcb, no_trade_lcb)
    selected = (
        opposite
        if policy_decision == "SWITCH_SAME_DECISION_SBC"
        else baseline
        if policy_decision == "KEEP_V6_7"
        else None
    )
    result = {
        "market_id": str(baseline["market_id"]),
        "baseline_action": str(baseline["action"]),
        "baseline_side": str(baseline["side"]),
        "baseline_decision_ts": int(baseline["decision_ts"]),
        "opposite_action": str(opposite["action"]),
        "opposite_decision_ts": int(opposite["decision_ts"]),
        "selected_policy_decision": policy_decision,
        "selected_action": str(selected["action"]) if selected else "NO_TRADE",
        "selected_side": str(selected["side"]) if selected else "NONE",
        "trade_selected": selected is not None,
        "switch_predicted_incremental_advantage": switch_prediction,
        "switch_incremental_advantage_lcb": switch_lcb,
        "no_trade_predicted_incremental_advantage": no_trade_prediction,
        "no_trade_incremental_advantage_lcb": no_trade_lcb,
        "alternative_decision_timestamp_used": False,
        "source_score_mutated": False,
        "outcome_or_pnl_field_used_at_inference": False,
        "full_five_action_interface": _full_action_diagnostics(
            baseline, opposite, selected_action=str(selected["action"]) if selected else "NO_TRADE"
        ),
        "selection_reason_codes": [],
        **_v7_0_blocked_safety_fields(),
    }
    result["decision_id"] = canonical_json_sha256(result)
    return result


def run_v6_7_relative_safe_policy_v7_2_fit(
    config: V67RelativeSafePolicyV72Config,
) -> dict[str, Any]:
    """Verify pins, fit once, replay once, and write immutable evidence."""

    paths = {
        "profile": Path(config.profile_path).resolve(),
        "v7_0_training_profile": Path(config.v7_0_training_profile_path).resolve(),
        "v6_7_candidate_profile": Path(config.v6_7_candidate_profile_path).resolve(),
        "runtime_target_rows": Path(config.runtime_target_rows_path).resolve(),
    }
    _verify_pin(paths["profile"], config.expected_profile_sha256, "#234 profile")
    profile = _load_json(paths["profile"])
    validate_v6_7_relative_safe_policy_v7_2_profile(profile)
    for key in ("v7_0_training_profile", "v6_7_candidate_profile", "runtime_target_rows"):
        _verify_pin(paths[key], profile["lineage"][f"{key}_sha256"], f"#234 {key}")
    v7_profile = _load_json(paths["v7_0_training_profile"])
    validate_v7_0_training_profile(v7_profile)
    validate_p_up_semantic_compatibility_v6_7_profile(
        _load_json(paths["v6_7_candidate_profile"])
    )
    canonical_rows = materialize_v7_0_sbc_rows(
        _load_jsonl(paths["runtime_target_rows"]), v7_profile
    )
    fit = fit_v6_7_relative_safe_policy_v7_2(
        rows=canonical_rows,
        profile=profile,
        implementation_commit=config.implementation_commit,
        fit_created_ts=config.fit_created_ts,
    )
    model = fit["model_artifact"]
    leakage = _leakage_audit(canonical_rows, model=model)
    report = _report(model=model, leakage=leakage)
    run_dir = Path(config.output_dir).resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    outputs = {
        "model": run_dir / "v7_2_relative_safe_policy_model.json",
        "report": run_dir / "v7_2_historical_replay_report.json",
        "report_markdown": run_dir / "v7_2_historical_replay_report.md",
        "leakage_audit": run_dir / "v7_2_fit_leakage_audit.json",
        "oof_rows": run_dir / "v7_2_forward_oof_policy_rows.jsonl",
        "candidate_selected_rows": run_dir / "v7_2_candidate_selected_rows.jsonl",
        "v6_7_baseline_selected_rows": (
            run_dir / "v7_2_v6_7_baseline_selected_rows.jsonl"
        ),
    }
    _write_json(outputs["model"], model)
    _write_json(outputs["report"], report)
    _write_text(outputs["report_markdown"], _report_markdown(report))
    _write_json(outputs["leakage_audit"], leakage)
    _write_jsonl(outputs["oof_rows"], fit["oof_rows"])
    _write_jsonl(outputs["candidate_selected_rows"], fit["candidate_selected_rows"])
    _write_jsonl(
        outputs["v6_7_baseline_selected_rows"], fit["v6_7_baseline_selected_rows"]
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        **{name: _descriptor(path) for name, path in paths.items()},
        **{name: _descriptor(path) for name, path in outputs.items()},
        "historical_gate_passed": model["historical_gate_passed"],
        "historical_gate_blocking_reason_codes": model[
            "historical_gate_blocking_reason_codes"
        ],
        "target_free_canary_collection_allowed": model[
            "target_free_canary_collection_allowed"
        ],
        "target_free_canary_started": False,
        "fit_leakage_audit_passed": leakage["fit_leakage_audit_passed"],
        **_v7_0_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v7_2_historical_fit_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "model": model,
        "report": report,
        "leakage": leakage,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "outputs": outputs,
    }


def _relative_examples(
    rows: list[dict[str, Any]], *, market_order: list[str]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["market_id"])].append(row)
    examples = []
    for market_id in market_order:
        baseline = _select_baseline(grouped[market_id])
        if baseline is None:
            examples.append(
                {
                    "market_id": market_id,
                    "baseline_trade_selected": False,
                    "baseline_action": "NO_TRADE",
                    "baseline_side": "NONE",
                    "baseline_decision_ts": None,
                    "baseline_max_input_ts": None,
                    "opposite_action": None,
                    "opposite_decision_ts": None,
                }
            )
            continue
        opposite = _same_decision_opposite(grouped[market_id], baseline)
        item = _inference_example(baseline, opposite)
        baseline_target = float(baseline["target_after_cost_net_pnl_per_contract"])
        opposite_target = float(opposite["target_after_cost_net_pnl_per_contract"])
        item.update(
            {
                "baseline_trade_selected": True,
                "baseline_target": baseline_target,
                "opposite_target": opposite_target,
                "switch_advantage_target": opposite_target - baseline_target,
                "no_trade_advantage_target": -baseline_target,
                "target_used_as_decision_time_input": False,
            }
        )
        examples.append(item)
    return examples


def _inference_example(
    baseline: dict[str, Any], opposite: dict[str, Any]
) -> dict[str, Any]:
    baseline_features = _feature_vector(baseline)
    opposite_features = _feature_vector(opposite)
    switch_features = [
        *baseline_features,
        *opposite_features,
        *(right - left for left, right in zip(baseline_features, opposite_features, strict=True)),
    ]
    return {
        "market_id": str(baseline["market_id"]),
        "baseline_action": str(baseline["action"]),
        "baseline_side": str(baseline["side"]),
        "baseline_decision_group_id": str(baseline["decision_group_id"]),
        "baseline_decision_ts": int(baseline["decision_ts"]),
        "baseline_max_input_ts": int(baseline["max_input_ts"]),
        "baseline_score": float(baseline["decision_time_features"]["action_score"]),
        "opposite_action": str(opposite["action"]),
        "opposite_side": str(opposite["side"]),
        "opposite_decision_group_id": str(opposite["decision_group_id"]),
        "opposite_decision_ts": int(opposite["decision_ts"]),
        "opposite_max_input_ts": int(opposite["max_input_ts"]),
        "baseline_features": baseline_features,
        "opposite_features": opposite_features,
        "switch_features": switch_features,
    }


def _fit_ridge_head(
    examples: list[dict[str, Any]],
    *,
    feature_field: str,
    target_field: str,
    profile: dict[str, Any],
) -> dict[str, Any]:
    if not examples:
        raise ValueError("#234 ridge fit examples empty")
    x = np.asarray([row[feature_field] for row in examples], dtype=float)
    y = np.asarray([row[target_field] for row in examples], dtype=float)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    floor = float(profile["feature_contract"]["standard_deviation_floor"])
    scale = np.where(std > floor, std, 1.0)
    z = (x - mean) / scale
    design = np.column_stack((np.ones(len(z)), z))
    alpha = float(profile["incremental_advantage_models"]["ridge_alpha"])
    penalty = np.diag([0.0, *([alpha] * x.shape[1])])
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ y,
    )
    bound = float(
        profile["incremental_advantage_models"]["coefficient_absolute_bound"]
    )
    finite_bounded = bool(
        np.all(np.isfinite(coefficients))
        and np.max(np.abs(coefficients[1:]), initial=0.0) <= bound
        and abs(float(coefficients[0])) <= bound
    )
    return {
        "model_family": "weighted_l2_ridge_with_unpenalized_intercept",
        "feature_field": feature_field,
        "target_field": target_field,
        "ridge_alpha": alpha,
        "feature_count": x.shape[1],
        "feature_mean": [float(value) for value in mean],
        "feature_scale": [float(value) for value in scale],
        "intercept": float(coefficients[0]),
        "coefficients": [float(value) for value in coefficients[1:]],
        "coefficient_absolute_bound": bound,
        "coefficients_finite_and_bounded": finite_bounded,
        "fit_selected_market_count": len(examples),
        "fit_market_ids_hash": canonical_json_sha256(
            sorted(row["market_id"] for row in examples)
        ),
    }


def _head_corrections(
    examples: list[dict[str, Any]],
    *,
    switch_model: dict[str, Any],
    no_trade_model: dict[str, Any],
    profile: dict[str, Any],
    support_valid: bool,
) -> dict[str, float | None]:
    if not support_valid:
        return {"switch": None, "no_trade": None}
    quantile = float(profile["conformal_contract"]["lower_residual_quantile"])
    switch_residuals = [
        row["switch_advantage_target"]
        - _predict_head(row["switch_features"], switch_model)
        for row in examples
    ]
    no_trade_residuals = [
        row["no_trade_advantage_target"]
        - _predict_head(row["baseline_features"], no_trade_model)
        for row in examples
    ]
    return {
        "switch": min(_quantile(switch_residuals, quantile), 0.0),
        "no_trade": min(_quantile(no_trade_residuals, quantile), 0.0),
    }


def _oof_head_corrections(
    rows: list[dict[str, Any]], *, profile: dict[str, Any]
) -> dict[str, Any]:
    active = [row for row in rows if row["baseline_trade_selected"]]
    quantile = float(profile["conformal_contract"]["lower_residual_quantile"])
    switch_residuals = [
        float(row["switch_advantage_target"])
        - float(row["switch_predicted_advantage"])
        for row in active
    ]
    no_trade_residuals = [
        float(row["no_trade_advantage_target"])
        - float(row["no_trade_predicted_advantage"])
        for row in active
    ]
    return {
        "source": "fixed_rolling_oof_residuals",
        "source_market_count": len(active),
        "lower_residual_quantile": quantile,
        "switch": min(_quantile(switch_residuals, quantile), 0.0),
        "no_trade": min(_quantile(no_trade_residuals, quantile), 0.0),
        "upward_correction_allowed": False,
        "outcome_used_at_future_inference": False,
    }


def _score_historical_example(
    example: dict[str, Any],
    *,
    switch_model: dict[str, Any],
    no_trade_model: dict[str, Any],
    corrections: dict[str, float | None],
    fold_index: int,
) -> dict[str, Any]:
    if not example["baseline_trade_selected"]:
        return {
            **example,
            "fold_index": fold_index,
            "switch_predicted_advantage": None,
            "switch_advantage_lcb": None,
            "no_trade_predicted_advantage": None,
            "no_trade_advantage_lcb": None,
            "switch_advantage_target": None,
            "no_trade_advantage_target": None,
            "selected_policy_decision": "KEEP_V6_7",
            "selected_action": "NO_TRADE",
            "selected_side": "NONE",
            "selected_target_after_cost_net_pnl_per_contract": 0.0,
            "baseline_target_after_cost_net_pnl_per_contract": 0.0,
            "validation_labels_used_for_fold_model_or_correction": False,
        }
    switch_prediction = _predict_head(example["switch_features"], switch_model)
    no_trade_prediction = _predict_head(example["baseline_features"], no_trade_model)
    switch_lcb = (
        switch_prediction + float(corrections["switch"])
        if corrections["switch"] is not None
        else None
    )
    no_trade_lcb = (
        no_trade_prediction + float(corrections["no_trade"])
        if corrections["no_trade"] is not None
        else None
    )
    policy_decision = _select_policy_decision(switch_lcb, no_trade_lcb)
    selected_action = (
        example["opposite_action"]
        if policy_decision == "SWITCH_SAME_DECISION_SBC"
        else "NO_TRADE"
        if policy_decision == "NO_TRADE"
        else example["baseline_action"]
    )
    selected_side = (
        example["opposite_side"]
        if policy_decision == "SWITCH_SAME_DECISION_SBC"
        else "NONE"
        if policy_decision == "NO_TRADE"
        else example["baseline_side"]
    )
    selected_target = (
        example["opposite_target"]
        if policy_decision == "SWITCH_SAME_DECISION_SBC"
        else 0.0
        if policy_decision == "NO_TRADE"
        else example["baseline_target"]
    )
    return {
        **example,
        "fold_index": fold_index,
        "switch_predicted_advantage": switch_prediction,
        "switch_advantage_lcb": switch_lcb,
        "no_trade_predicted_advantage": no_trade_prediction,
        "no_trade_advantage_lcb": no_trade_lcb,
        "selected_policy_decision": policy_decision,
        "selected_action": selected_action,
        "selected_side": selected_side,
        "selected_target_after_cost_net_pnl_per_contract": selected_target,
        "baseline_target_after_cost_net_pnl_per_contract": example["baseline_target"],
        "validation_labels_used_for_fold_model_or_correction": False,
        "target_used_as_decision_time_input": False,
    }


def _historical_replay(
    rows: list[dict[str, Any]], *, profile: dict[str, Any]
) -> dict[str, Any]:
    market_ids = sorted(row["market_id"] for row in rows)
    if len(market_ids) != int(
        profile["historical_replay_superiority_gate"]["exact_evaluation_market_count"]
    ):
        raise ValueError("#234 historical replay market count invalid")
    size = float(profile["historical_replay_superiority_gate"]["fixed_position_size"])
    candidate_rows = []
    baseline_rows = []
    for row in rows:
        if row["selected_action"] != "NO_TRADE":
            candidate_rows.append(_replay_row(row, candidate=True, size=size))
        if row["baseline_action"] != "NO_TRADE":
            baseline_rows.append(_replay_row(row, candidate=False, size=size))
    candidate = _replay_metrics(candidate_rows, market_ids=market_ids)
    baseline = _replay_metrics(baseline_rows, market_ids=market_ids)
    total_delta = (
        candidate["total_after_cost_net_pnl_at_frozen_size"]
        - baseline["total_after_cost_net_pnl_at_frozen_size"]
    )
    robust_delta = (
        candidate["largest_winner_removed_after_cost_net_pnl_at_frozen_size"]
        - baseline["largest_winner_removed_after_cost_net_pnl_at_frozen_size"]
    )
    summary = {
        "gate_name": "same_dataset_historical_replay_superiority_over_v6_7",
        "gate_mode": "development_screening_only_before_new_collection",
        "evaluation_market_count": len(market_ids),
        "evaluation_market_ids_hash": canonical_json_sha256(market_ids),
        "fixed_position_size": size,
        "same_runtime_aligned_target_and_cost_contract": True,
        "same_position_management_and_guard_contract": True,
        "one_bet_maximum_per_market": True,
        "v6_7_baseline_market_action_identity_hash": canonical_json_sha256(
            [
                {
                    "market_id": row["market_id"],
                    "decision_ts": row["decision_ts"],
                    "action": row["action"],
                }
                for row in baseline_rows
            ]
        ),
        "candidate": candidate,
        "v6_7_baseline": baseline,
        "candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size": total_delta,
        "candidate_minus_v6_7_largest_winner_removed_after_cost_net_pnl_at_frozen_size": (
            robust_delta
        ),
        "baseline_identity_reconciled": all(
            row["baseline_action"] == baseline_row["action"]
            and row["baseline_decision_ts"] == baseline_row["decision_ts"]
            for row, baseline_row in zip(
                [row for row in rows if row["baseline_action"] != "NO_TRADE"],
                baseline_rows,
                strict=True,
            )
        ),
        "policy_decision_distribution": dict(
            sorted(Counter(row["selected_policy_decision"] for row in rows).items())
        ),
        "common_selected_row_filter_applied": False,
        "no_bet_market_pnl": 0.0,
        "historical_training_targets_used_for_fixed_coefficient_fit": True,
        "historical_oof_or_validation_pnl_used_for_feature_hyperparameter_or_threshold_tuning": False,
        "historical_pnl_used_for_pre_collection_screening_only": True,
        "promotion_or_paper_unlock_allowed": False,
        "candidate_selected_rows": candidate_rows,
        "v6_7_baseline_selected_rows": baseline_rows,
    }
    return summary


def _replay_row(row: dict[str, Any], *, candidate: bool, size: float) -> dict[str, Any]:
    target = float(
        row["selected_target_after_cost_net_pnl_per_contract"]
        if candidate
        else row["baseline_target_after_cost_net_pnl_per_contract"]
    )
    action = row["selected_action"] if candidate else row["baseline_action"]
    side = row["selected_side"] if candidate else row["baseline_side"]
    item = {
        "market_id": row["market_id"],
        "decision_ts": row["baseline_decision_ts"],
        "action": action,
        "side": side,
        "policy_decision": row["selected_policy_decision"] if candidate else "V6_7",
        "target_after_cost_net_pnl_per_contract": target,
        "fixed_position_size": size,
        "after_cost_net_pnl_at_frozen_size": target * size,
        "target_used_as_decision_time_input": False,
        "target_used_for_pre_collection_historical_replay_gate": True,
    }
    item["replay_row_id"] = canonical_json_sha256(item)
    return item


def _replay_metrics(
    rows: list[dict[str, Any]], *, market_ids: list[str]
) -> dict[str, Any]:
    pnl = dict.fromkeys(market_ids, 0.0)
    per_contract = dict.fromkeys(market_ids, 0.0)
    for row in rows:
        pnl[row["market_id"]] = float(row["after_cost_net_pnl_at_frozen_size"])
        per_contract[row["market_id"]] = float(
            row["target_after_cost_net_pnl_per_contract"]
        )
    total = sum(pnl.values())
    largest = max(max(pnl.values(), default=0.0), 0.0)
    return {
        "evaluation_market_count": len(market_ids),
        "selected_bet_count": len(rows),
        "no_bet_market_count": len(market_ids) - len(rows),
        "total_after_cost_net_pnl_per_contract": sum(per_contract.values()),
        "total_after_cost_net_pnl_at_frozen_size": total,
        "largest_winning_market_pnl_at_frozen_size": largest,
        "largest_winner_removed_after_cost_net_pnl_at_frozen_size": total - largest,
        "bet_count_by_side": dict(sorted(Counter(row["side"] for row in rows).items())),
        "bet_count_by_action": dict(
            sorted(Counter(row["action"] for row in rows).items())
        ),
        "pnl_by_side": _sum_by(rows, "side", "after_cost_net_pnl_at_frozen_size"),
    }


def _predict_head(features: list[float], model: dict[str, Any]) -> float:
    x = np.asarray(features, dtype=float)
    mean = np.asarray(model["feature_mean"], dtype=float)
    scale = np.asarray(model["feature_scale"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    return float(model["intercept"] + ((x - mean) / scale) @ coefficients)


def _select_policy_decision(
    switch_lcb: float | None, no_trade_lcb: float | None
) -> str:
    choices = []
    if switch_lcb is not None and switch_lcb > 0.0:
        choices.append((switch_lcb, "SWITCH_SAME_DECISION_SBC"))
    if no_trade_lcb is not None and no_trade_lcb > 0.0:
        choices.append((no_trade_lcb, "NO_TRADE"))
    if not choices:
        return "KEEP_V6_7"
    return sorted(choices, key=lambda item: (-item[0], item[1]))[0][1]


def _select_baseline(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [
        row
        for row in rows
        if float(row["decision_time_features"]["action_score_available"]) > 0.0
        and float(row["decision_time_features"]["action_score"]) > 0.0
    ]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda row: (
            -float(row["decision_time_features"]["action_score"]),
            int(row["decision_ts"]),
            str(row["action"]),
        ),
    )[0]


def _same_decision_opposite(
    rows: list[dict[str, Any]], baseline: dict[str, Any]
) -> dict[str, Any]:
    candidates = [
        row
        for row in rows
        if row["decision_group_id"] == baseline["decision_group_id"]
        and row["action"] != baseline["action"]
        and row["action"] in SBC_ACTIONS
    ]
    if len(candidates) != 1:
        raise ValueError("#234 same-decision opposite SBC action missing")
    opposite = candidates[0]
    if int(opposite["decision_ts"]) != int(baseline["decision_ts"]):
        raise ValueError("#234 alternative decision timestamp mismatch")
    return opposite


def _feature_vector(row: dict[str, Any]) -> list[float]:
    features = dict(row["decision_time_features"])
    if tuple(features) != FEATURE_NAMES:
        raise ValueError("#234 decision-time feature contract invalid")
    values = [float(features[name]) for name in FEATURE_NAMES]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("#234 decision-time feature value invalid")
    return values


def _validate_canonical_rows(rows: list[dict[str, Any]]) -> None:
    if len({row["market_id"] for row in rows}) != 134:
        raise ValueError("#234 historical market count invalid")
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        _feature_vector(row)
        if int(row["max_input_ts"]) > int(row["decision_ts"]):
            raise ValueError("#234 feature causality violation")
        if row.get("target_used_as_decision_time_input") is not False:
            raise ValueError("#234 target used as decision input")
        grouped[str(row["decision_group_id"])].add(str(row["action"]))
    if any(actions != set(SBC_ACTIONS) for actions in grouped.values()):
        raise ValueError("#234 same-decision SBC grid incomplete")


def _validate_target_free_market_rows(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        _feature_vector(row)
        if int(row["max_input_ts"]) > int(row["decision_ts"]):
            raise ValueError("#234 inference feature causality violation")
        grouped[str(row["decision_group_id"])].add(str(row["action"]))
    if any(actions != set(SBC_ACTIONS) for actions in grouped.values()):
        raise ValueError("#234 inference same-decision SBC grid incomplete")


def _active_examples(
    by_market: dict[str, dict[str, Any]], market_order: list[str]
) -> list[dict[str, Any]]:
    return [
        by_market[market_id]
        for market_id in market_order
        if by_market[market_id]["baseline_trade_selected"]
    ]


def _market_order(rows: list[dict[str, Any]]) -> list[str]:
    minimum: dict[str, int] = {}
    for row in rows:
        market_id = str(row["market_id"])
        minimum[market_id] = min(
            minimum.get(market_id, int(row["decision_ts"])),
            int(row["decision_ts"]),
        )
    return [key for key, _ in sorted(minimum.items(), key=lambda item: (item[1], item[0]))]


def _market_min_ts(rows: list[dict[str, Any]], market_ids: list[str]) -> int:
    selected = set(market_ids)
    return min(int(row["decision_ts"]) for row in rows if row["market_id"] in selected)


def _market_max_ts(rows: list[dict[str, Any]], market_ids: list[str]) -> int:
    selected = set(market_ids)
    return max(int(row["decision_ts"]) for row in rows if row["market_id"] in selected)


def _quantile(values: list[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise ValueError("#234 quantile inputs invalid")
    ordered = sorted(float(value) for value in values)
    index = min(math.ceil(quantile * len(ordered)) - 1, len(ordered) - 1)
    return ordered[max(index, 0)]


def _sum_by(rows: list[dict[str, Any]], key: str, value: str) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for row in rows:
        totals[str(row[key])] += float(row[value])
    return dict(sorted(totals.items()))


def _leakage_audit(
    rows: list[dict[str, Any]], *, model: dict[str, Any]
) -> dict[str, Any]:
    checks = {
        "all_134_historical_markets_used": len({row["market_id"] for row in rows})
        == 134,
        "feature_timestamp_causality": all(
            int(row["max_input_ts"]) <= int(row["decision_ts"]) for row in rows
        ),
        "issue233_result_artifacts_not_opened": model[
            "issue233_result_artifacts_opened_for_fit_or_tuning"
        ]
        is False,
        "future_outcomes_not_used": model[
            "issue229_or_issue231_future_outcomes_used"
        ]
        is False,
        "validation_labels_isolated_per_fold": all(
            row["validation_labels_used_for_fold_model_or_correction"] is False
            for row in model["fold_reports"]
        ),
        "no_alternative_decision_timestamp": model["historical_gate_checks"][
            "same_decision_switch_only"
        ],
        "no_outcome_driven_tuning": model[
            "historical_oof_or_validation_pnl_used_for_feature_hyperparameter_or_threshold_tuning"
        ]
        is False,
        "safety_blocked": model["paper_candidate_allowed"] is False
        and model["capital_at_risk"] is False
        and model["v8_execution_handoff_allowed"] is False,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    audit = {
        "schema_version": LEAKAGE_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "fit_leakage_checks": checks,
        "fit_leakage_audit_passed": not blockers,
        "fit_leakage_blocking_reason_codes": blockers,
        "issue233_result_artifact_paths_accepted_by_config": False,
        "future_target_accessed": False,
        **_v7_0_blocked_safety_fields(),
    }
    audit["leakage_audit_id"] = canonical_json_sha256(audit)
    return audit


def _report(model: dict[str, Any], leakage: dict[str, Any]) -> dict[str, Any]:
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "historical_gate_passed": model["historical_gate_passed"],
        "historical_gate_blocking_reason_codes": model[
            "historical_gate_blocking_reason_codes"
        ],
        "historical_policy_difference_market_count": model[
            "historical_policy_difference_market_count"
        ],
        "historical_replay_superiority_gate": model[
            "historical_replay_superiority_gate"
        ],
        "fold_reports": model["fold_reports"],
        "final_conformal_corrections": model["final_conformal_corrections"],
        "fit_leakage_audit_passed": leakage["fit_leakage_audit_passed"],
        "issue233_result_artifacts_opened_for_fit_or_tuning": False,
        "target_free_canary_collection_allowed": model[
            "target_free_canary_collection_allowed"
        ],
        "target_free_canary_started": False,
        "future_confirmatory_authorized": False,
        **_v7_0_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def _report_markdown(report: dict[str, Any]) -> str:
    replay = report["historical_replay_superiority_gate"]
    return "\n".join(
        [
            "# v7.2 Frozen-v6.7 Relative Safe Policy Historical Replay",
            "",
            f"- historical gate passed: `{str(report['historical_gate_passed']).lower()}`",
            f"- blockers: `{report['historical_gate_blocking_reason_codes']}`",
            "- candidate frozen-size PnL: "
            f"`{replay['candidate']['total_after_cost_net_pnl_at_frozen_size']}`",
            "- v6.7 frozen-size PnL: "
            f"`{replay['v6_7_baseline']['total_after_cost_net_pnl_at_frozen_size']}`",
            "- candidate-minus-v6.7 frozen-size PnL: "
            f"`{replay['candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size']}`",
            "- policy decision distribution: "
            f"`{replay['policy_decision_distribution']}`",
            f"- policy differences: `{report['historical_policy_difference_market_count']}`",
            f"- leakage audit passed: `{str(report['fit_leakage_audit_passed']).lower()}`",
            "- #233 result artifacts opened for fit/tuning: `false`",
            "- target-free canary started: `false`",
            "- paper/live/write/wallet/capital unlock: `false`",
            "",
        ]
    )


def _full_action_diagnostics(
    baseline: dict[str, Any], opposite: dict[str, Any], *, selected_action: str
) -> list[dict[str, Any]]:
    output = []
    rows_by_action = {baseline["action"]: baseline, opposite["action"]: opposite}
    for action in (*SBC_ACTIONS, *HTS_ACTIONS, "NO_TRADE"):
        output.append(
            {
                "action": action,
                "score_available": action in rows_by_action or action == "NO_TRADE",
                "selected": action == selected_action,
                "reason_codes": (
                    []
                    if action in rows_by_action or action == "NO_TRADE"
                    else ["prior_preregistered_hts_prediction_loss_gate_failed"]
                ),
            }
        )
    return output


def _no_trade_inference(
    rows: list[dict[str, Any]], *, reasons: list[str]
) -> dict[str, Any]:
    market_ids = {str(row.get("market_id") or "") for row in rows}
    result = {
        "market_id": next(iter(market_ids), ""),
        "baseline_action": "NO_TRADE",
        "selected_policy_decision": "NO_TRADE",
        "selected_action": "NO_TRADE",
        "selected_side": "NONE",
        "trade_selected": False,
        "alternative_decision_timestamp_used": False,
        "source_score_mutated": False,
        "outcome_or_pnl_field_used_at_inference": False,
        "selection_reason_codes": reasons,
        **_v7_0_blocked_safety_fields(),
    }
    result["decision_id"] = canonical_json_sha256(result)
    return result


def _row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["baseline_decision_ts"] or 0),
        str(row["market_id"]),
    )
