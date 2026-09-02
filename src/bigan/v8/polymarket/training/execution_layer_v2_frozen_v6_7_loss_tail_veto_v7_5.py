"""Training-only frozen-v6.7 loss-tail veto policy for issue #237."""

from __future__ import annotations

import base64
import math
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_abstention_aware_expected_net_pnl_v7_0 import (
    _v7_0_blocked_safety_fields,
)
from bigan.v8.polymarket.training.execution_layer_v2_abstention_aware_expected_net_pnl_v7_0_fit import (
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
from bigan.v8.polymarket.training.execution_layer_v2_v6_7_relative_safe_policy_v7_2 import (
    FORBIDDEN_INFERENCE_FIELDS,
    _full_action_diagnostics,
    _historical_replay,
    _inference_example,
    _market_order,
    _no_trade_inference,
    _relative_examples,
    _same_decision_opposite,
    _select_baseline,
    _validate_canonical_rows,
    _validate_target_free_market_rows,
)

CANDIDATE_NAME = "frozen_v6_7_loss_tail_veto_v7_5"
PROFILE_SCHEMA_VERSION = "bigan-v8-frozen-v6-7-loss-tail-veto-v7-5-profile-v1"
MODEL_SCHEMA_VERSION = "bigan-v8-frozen-v6-7-loss-tail-veto-v7-5-model-v1"
REPORT_SCHEMA_VERSION = "bigan-v8-frozen-v6-7-loss-tail-veto-v7-5-report-v1"
ATTRIBUTION_SCHEMA_VERSION = (
    "bigan-v8-frozen-v6-7-loss-tail-veto-v7-5-attribution-v1"
)
LEAKAGE_SCHEMA_VERSION = "bigan-v8-frozen-v6-7-loss-tail-veto-v7-5-leakage-v1"
MANIFEST_SCHEMA_VERSION = "bigan-v8-frozen-v6-7-loss-tail-veto-v7-5-manifest-v1"
PROFILE_NAMES = ("KEEP_V6_7", "LOSS_TAIL_Q95", "LOSS_TAIL_Q90", "LOSS_TAIL_Q80")
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
    "v7_2_relative_policy_source_sha256": (
        "bc099273ca3c1db04062d40a5172e82d0ec075e2c63886cc42dd2dde67cc961a"
    ),
    "xgboost_version": "3.2.0",
}
FROZEN_CLASSIFIER = {
    "booster": "gbtree",
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "num_boost_round": 64,
    "max_depth": 2,
    "eta": 0.05,
    "min_child_weight": 1.0,
    "gamma": 0.0,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 10.0,
    "max_bin": 64,
    "tree_method": "hist",
    "seed": 0,
    "nthread": 1,
    "early_stopping_enabled": False,
    "canonical_features_must_be_finite": True,
    "loss_target_definition": (
        "v6_7_baseline_after_cost_net_return_less_than_or_equal_to_zero"
    ),
    "minimum_prior_active_market_count": 20,
    "minimum_prior_loss_market_count": 5,
    "minimum_prior_non_loss_market_count": 5,
}


@dataclass(frozen=True, slots=True)
class FrozenV67LossTailVetoV75Config:
    """Pinned inputs for the single issue #237 historical fit and replay."""

    run_id: str
    output_dir: Path | str
    profile_path: Path | str
    expected_profile_sha256: str
    v7_0_training_profile_path: Path | str
    v6_7_candidate_profile_path: Path | str
    v7_2_relative_policy_source_path: Path | str
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
            "v7_2_relative_policy_source_path",
            "runtime_target_rows_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


def validate_frozen_v6_7_loss_tail_veto_v7_5_profile(
    profile: dict[str, Any],
) -> None:
    """Reject any drift from the preregistered #237 contract."""

    checks = {
        "identity": profile.get("schema_version") == PROFILE_SCHEMA_VERSION
        and profile.get("issue_number") == 237
        and profile.get("candidate_name") == CANDIDATE_NAME
        and profile.get("preregistered_before_implementation_and_fit") is True,
        "lineage": profile.get("lineage") == FROZEN_LINEAGE,
        "design_split": profile.get("design_split")
        == {
            "market_count": 44,
            "baseline_active_market_count": 34,
            "baseline_no_trade_market_count": 10,
            "baseline_side_counts": {"DOWN": 21, "UP": 13},
            "baseline_target_positive_count": 21,
            "baseline_target_non_positive_count": 13,
            "maximum_decision_ts": 1784245320000,
            "sealed_outer_oof_minimum_decision_ts": 1784245560000,
            "sealed_outer_oof_targets_opened_for_design": False,
        },
        "prior_exclusion": profile.get("prior_result_exclusion")
        == {
            "issue233_issue234_issue235_or_issue236_result_artifacts_accepted_as_inputs": False,
            "issue233_issue234_issue235_or_issue236_oof_rows_used": False,
            "issue233_issue234_issue235_or_issue236_pnl_used_for_selection": False,
            "issue229_or_issue231_future_outcomes_used": False,
        },
        "baseline": profile.get("baseline_and_action_contract")
        == {
            "baseline_candidate": "p_up_semantic_execution_compatibility_v6_7",
            "allowed_policy_decisions": ["KEEP_V6_7", "VETO_TO_NO_TRADE"],
            "baseline_no_trade_cannot_be_activated": True,
            "side_switch_allowed": False,
            "action_family_switch_allowed": False,
            "alternative_decision_timestamp_allowed": False,
            "maximum_bets_per_market": 1,
            "side_quota_allowed": False,
            "full_five_action_interface_required": True,
            "execution_guard_unchanged": True,
        },
        "outer": profile.get("outer_split")
        == {
            "source_market_count": 134,
            "initial_training_market_count": 44,
            "forward_fold_count": 5,
            "validation_market_count_per_fold": 18,
            "outer_oof_market_count": 90,
            "market_order": "minimum_decision_ts_then_market_id",
            "outer_validation_targets_used_for_profile_selection_or_fit": False,
        },
        "classifier": profile.get("loss_classifier") == FROZEN_CLASSIFIER,
        "profiles": profile.get("veto_profiles")
        == [
            {"name": "KEEP_V6_7", "veto_fraction": 0.0},
            {"name": "LOSS_TAIL_Q95", "veto_fraction": 0.05},
            {"name": "LOSS_TAIL_Q90", "veto_fraction": 0.1},
            {"name": "LOSS_TAIL_Q80", "veto_fraction": 0.2},
        ],
        "selector": profile.get("nested_selector")
        == {
            "inner_prequential_initial_market_count": 20,
            "minimum_inner_evaluation_market_count": 20,
            "minimum_profile_available_active_market_count": 10,
            "minimum_policy_difference_market_count": 1,
            "inner_no_bet_market_pnl": 0.0,
            "alternative_total_pnl_must_not_be_worse_than_keep": True,
            "alternative_largest_winner_removed_must_not_be_worse_than_keep": True,
            "tie_break": (
                "highest_pnl_then_highest_lwr_then_fewer_vetoes_then_keep"
            ),
            "no_eligible_alternative_behavior": "KEEP_V6_7",
            "outer_oof_results_used_for_selection": False,
            "result_selected_rerun_allowed": False,
        },
        "replay": profile.get("historical_replay_superiority_gate")
        == {
            "exact_evaluation_market_count": 90,
            "fixed_position_size": 0.2,
            "no_bet_market_pnl": 0.0,
            "common_selected_row_filter_allowed": False,
            "candidate_minus_v6_7_total_pnl_minimum_inclusive": 0.0,
            "candidate_minus_v6_7_largest_winner_removed_pnl_minimum_inclusive": 0.0,
            "exact_baseline_identity_reconciliation_required": True,
            "oof_pnl_used_for_profile_feature_hyperparameter_or_threshold_selection": False,
            "oof_pnl_used_for_pre_collection_screening_only": True,
            "failure_stops_before_collection": True,
            "promotion_or_paper_unlock_allowed": False,
        },
        "canary": profile.get("target_free_canary")
        == {
            "historical_noninferiority_gate_must_pass_before_collection": True,
            "historical_policy_difference_market_count_minimum": 1,
            "strictly_later_outcome_blind_market_count": 12,
            "maximum_attempt_count": 18,
            "outcome_label_or_pnl_access_allowed": False,
            "full_execution_guard_unchanged": True,
        },
        "safety": profile.get("safety") == _v7_0_blocked_safety_fields(),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("#237 v7.5 profile invalid: " + ", ".join(blockers))


def fit_frozen_v6_7_loss_tail_veto_v7_5(
    *,
    rows: list[dict[str, Any]],
    profile: dict[str, Any],
    implementation_commit: str,
    fit_created_ts: int,
) -> dict[str, Any]:
    """Fit nested prior-only veto policies and evaluate the sealed exact-90 once."""

    validate_frozen_v6_7_loss_tail_veto_v7_5_profile(profile)
    _validate_canonical_rows(rows)
    order = _market_order(rows)
    examples = _relative_examples(rows, market_order=order)
    by_market = {row["market_id"]: row for row in examples}
    outer = profile["outer_split"]
    initial = int(outer["initial_training_market_count"])
    width = int(outer["validation_market_count_per_fold"])
    cache: dict[tuple[str, ...], dict[str, Any]] = {}
    outer_rows: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    for fold_index in range(int(outer["forward_fold_count"])):
        train_order = order[: initial + fold_index * width]
        validation_order = order[
            initial + fold_index * width : initial + (fold_index + 1) * width
        ]
        if _max_ts(rows, train_order) >= _min_ts(rows, validation_order):
            raise ValueError("#237 outer fold chronology invalid")
        selection = _nested_select_profile(
            train_order, by_market=by_market, profile=profile, cache=cache
        )
        selected_profile = _profile_by_name(
            profile, selection["selected_veto_profile_name"]
        )
        artifact = _fit_loss_classifier(
            train_order, by_market=by_market, profile=profile, cache=cache
        )
        for market_id in validation_order:
            outer_rows.append(
                _score_example(
                    by_market[market_id],
                    veto_profile=selected_profile,
                    classifier_artifact=artifact,
                    fold_index=fold_index,
                )
            )
        fold_reports.append(
            {
                "fold_index": fold_index,
                "outer_train_market_count": len(train_order),
                "outer_validation_market_count": len(validation_order),
                "outer_train_max_decision_ts": _max_ts(rows, train_order),
                "outer_validation_min_decision_ts": _min_ts(rows, validation_order),
                "outer_validation_targets_used_for_profile_selection_or_fit": False,
                "nested_selection": selection,
            }
        )
    final_selection = _nested_select_profile(
        order, by_market=by_market, profile=profile, cache=cache
    )
    final_profile = _profile_by_name(
        profile, final_selection["selected_veto_profile_name"]
    )
    final_artifact = _fit_loss_classifier(
        order, by_market=by_market, profile=profile, cache=cache
    )
    replay = _historical_replay(outer_rows, profile=profile)
    replay["gate_name"] = "same_dataset_historical_noninferiority_to_v6_7"
    replay["gate_mode"] = (
        "development_noninferiority_screening_only_before_new_collection"
    )
    candidate_rows = replay.pop("candidate_selected_rows")
    baseline_rows = replay.pop("v6_7_baseline_selected_rows")
    difference_count = sum(
        row["selected_policy_decision"] != "KEEP_V6_7" for row in outer_rows
    )
    total_delta = float(
        replay["candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size"]
    )
    lwr_delta = float(
        replay[
            "candidate_minus_v6_7_largest_winner_removed_after_cost_net_pnl_at_frozen_size"
        ]
    )
    gate_profile = profile["historical_replay_superiority_gate"]
    checks = {
        "exact_outer_oof_market_support": len(
            {row["market_id"] for row in outer_rows}
        )
        == int(outer["outer_oof_market_count"]),
        "outer_validation_target_isolation": all(
            item["outer_validation_targets_used_for_profile_selection_or_fit"]
            is False
            for item in fold_reports
        ),
        "outer_chronology": all(
            item["outer_train_max_decision_ts"]
            < item["outer_validation_min_decision_ts"]
            for item in fold_reports
        ),
        "nested_selector_did_not_use_outer_oof": all(
            item["nested_selection"]["outer_oof_results_used_for_selection"]
            is False
            for item in fold_reports
        ),
        "final_profile_did_not_use_outer_oof": final_selection[
            "outer_oof_results_used_for_selection"
        ]
        is False,
        "candidate_total_pnl_noninferior_to_v6_7": total_delta
        >= float(gate_profile["candidate_minus_v6_7_total_pnl_minimum_inclusive"]),
        "candidate_lwr_pnl_noninferior_to_v6_7": lwr_delta
        >= float(
            gate_profile[
                "candidate_minus_v6_7_largest_winner_removed_pnl_minimum_inclusive"
            ]
        ),
        "baseline_identity_reconciled": replay["baseline_identity_reconciled"],
        "veto_contract_preserved": all(_veto_contract_preserved(row) for row in outer_rows),
        "final_classifier_available_or_keep": final_profile["name"] == "KEEP_V6_7"
        or final_artifact["available"],
    }
    reason_map = {
        "exact_outer_oof_market_support": "outer_oof_market_support_invalid",
        "outer_validation_target_isolation": "outer_validation_target_isolation_failed",
        "outer_chronology": "outer_fold_chronology_failed",
        "nested_selector_did_not_use_outer_oof": "outer_oof_used_for_profile_selection",
        "final_profile_did_not_use_outer_oof": "outer_oof_used_for_final_profile",
        "candidate_total_pnl_noninferior_to_v6_7": (
            "historical_same_dataset_candidate_pnl_worse_than_v6_7"
        ),
        "candidate_lwr_pnl_noninferior_to_v6_7": (
            "historical_same_dataset_largest_winner_removed_pnl_worse_than_v6_7"
        ),
        "baseline_identity_reconciled": "frozen_v6_7_baseline_identity_mismatch",
        "veto_contract_preserved": "v7_5_veto_contract_violated",
        "final_classifier_available_or_keep": "final_loss_classifier_unavailable",
    }
    blockers = [reason_map[name] for name, passed in checks.items() if not passed]
    gate_passed = not blockers
    minimum_difference = int(
        profile["target_free_canary"][
            "historical_policy_difference_market_count_minimum"
        ]
    )
    actionability_passed = difference_count >= minimum_difference
    canary_allowed = gate_passed and actionability_passed
    model = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": implementation_commit,
        "fit_created_ts": fit_created_ts,
        "frozen": gate_passed,
        "decision_time_safe": True,
        "policy_profile_names": list(PROFILE_NAMES),
        "outer_fold_reports": fold_reports,
        "final_nested_selection": final_selection,
        "final_veto_profile": final_profile,
        "final_loss_classifier_artifact": final_artifact,
        "historical_replay_noninferiority_gate": replay,
        "historical_policy_difference_market_count": difference_count,
        "historical_actionability_passed": actionability_passed,
        "model_improvement_demonstrated": gate_passed and total_delta > 0.0,
        "historical_gate_checks": checks,
        "historical_noninferiority_gate_passed": gate_passed,
        "historical_gate_blocking_reason_codes": blockers,
        "prior_result_artifacts_opened": False,
        "outer_oof_pnl_used_for_profile_feature_hyperparameter_or_threshold_selection": False,
        "historical_training_targets_used_inside_nested_training_only_selection": True,
        "historical_replay_is_promotion_evidence": False,
        "target_free_canary_collection_allowed": canary_allowed,
        "target_free_canary_collection_blocking_reason_codes": (
            []
            if canary_allowed
            else [
                *blockers,
                *(
                    []
                    if actionability_passed
                    else ["historical_policy_difference_support_absent"]
                ),
            ]
        ),
        "target_free_canary_started": False,
        "future_confirmatory_authorized": False,
        **_v7_0_blocked_safety_fields(),
    }
    model["model_artifact_id"] = canonical_json_sha256(model)
    return {
        "model_artifact": model,
        "outer_oof_rows": sorted(outer_rows, key=_row_sort_key),
        "candidate_selected_rows": candidate_rows,
        "v6_7_baseline_selected_rows": baseline_rows,
        "veto_attribution": _veto_attribution(outer_rows, fixed_size=0.2),
    }


def score_frozen_v6_7_loss_tail_veto_v7_5_market(
    rows: list[dict[str, Any]], *, model_artifact: dict[str, Any]
) -> dict[str, Any]:
    """Apply the final frozen v7.5 veto profile to target-free future rows."""

    reasons = []
    if model_artifact.get("schema_version") != MODEL_SCHEMA_VERSION:
        reasons.append("v7_5_model_artifact_schema_invalid")
    if model_artifact.get("historical_noninferiority_gate_passed") is not True:
        reasons.append("v7_5_historical_noninferiority_gate_not_passed")
    if any(FORBIDDEN_INFERENCE_FIELDS.intersection(row) for row in rows):
        reasons.append("v7_5_forbidden_outcome_field_in_inference_row")
    if reasons:
        return _no_trade_inference(rows, reasons=reasons)
    _validate_target_free_market_rows(rows)
    baseline = _select_baseline(rows)
    if baseline is None:
        return _no_trade_inference(
            rows, reasons=["frozen_v6_7_no_positive_baseline_action"]
        )
    opposite = _same_decision_opposite(rows, baseline)
    example = {
        **_inference_example(baseline, opposite),
        "baseline_trade_selected": True,
    }
    decision = _score_example(
        example,
        veto_profile=model_artifact["final_veto_profile"],
        classifier_artifact=model_artifact["final_loss_classifier_artifact"],
        fold_index=None,
        inference=True,
    )
    result = {
        "market_id": decision["market_id"],
        "selected_veto_profile_name": decision["selected_veto_profile_name"],
        "baseline_action": decision["baseline_action"],
        "baseline_side": decision["baseline_side"],
        "baseline_decision_ts": decision["baseline_decision_ts"],
        "selected_policy_decision": decision["selected_policy_decision"],
        "selected_action": decision["selected_action"],
        "selected_side": decision["selected_side"],
        "trade_selected": decision["selected_action"] != "NO_TRADE",
        "predicted_baseline_loss_probability": decision[
            "predicted_baseline_loss_probability"
        ],
        "loss_probability_veto_threshold": decision[
            "loss_probability_veto_threshold"
        ],
        "side_switch_applied": False,
        "action_family_switch_applied": False,
        "alternative_decision_timestamp_used": False,
        "source_score_mutated": False,
        "outcome_or_pnl_field_used_at_inference": False,
        "full_five_action_interface": _full_action_diagnostics(
            baseline, opposite, selected_action=decision["selected_action"]
        ),
        "selection_reason_codes": [],
        **_v7_0_blocked_safety_fields(),
    }
    result["decision_id"] = canonical_json_sha256(result)
    return result


def run_frozen_v6_7_loss_tail_veto_v7_5_fit(
    config: FrozenV67LossTailVetoV75Config,
) -> dict[str, Any]:
    """Verify lineage, fit once, replay once, and write content-addressed evidence."""

    paths = {
        "profile": Path(config.profile_path).resolve(),
        "v7_0_training_profile": Path(config.v7_0_training_profile_path).resolve(),
        "v6_7_candidate_profile": Path(config.v6_7_candidate_profile_path).resolve(),
        "v7_2_relative_policy_source": Path(
            config.v7_2_relative_policy_source_path
        ).resolve(),
        "runtime_target_rows": Path(config.runtime_target_rows_path).resolve(),
    }
    _verify_pin(paths["profile"], config.expected_profile_sha256, "#237 profile")
    profile = _load_json(paths["profile"])
    validate_frozen_v6_7_loss_tail_veto_v7_5_profile(profile)
    for key in (
        "v7_0_training_profile",
        "v6_7_candidate_profile",
        "v7_2_relative_policy_source",
        "runtime_target_rows",
    ):
        _verify_pin(paths[key], profile["lineage"][f"{key}_sha256"], f"#237 {key}")
    if xgb.__version__ != profile["lineage"]["xgboost_version"]:
        raise ValueError("#237 xgboost version mismatch")
    v7_profile = _load_json(paths["v7_0_training_profile"])
    validate_v7_0_training_profile(v7_profile)
    validate_p_up_semantic_compatibility_v6_7_profile(
        _load_json(paths["v6_7_candidate_profile"])
    )
    rows = materialize_v7_0_sbc_rows(
        _load_jsonl(paths["runtime_target_rows"]), v7_profile
    )
    fit = fit_frozen_v6_7_loss_tail_veto_v7_5(
        rows=rows,
        profile=profile,
        implementation_commit=config.implementation_commit,
        fit_created_ts=config.fit_created_ts,
    )
    model = fit["model_artifact"]
    leakage = _leakage_audit(rows, model=model)
    attribution = fit["veto_attribution"]
    report = _report(model=model, leakage=leakage, attribution=attribution)
    run_dir = Path(config.output_dir).resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    outputs = {
        "model": run_dir / "v7_5_loss_tail_veto_model.json",
        "report": run_dir / "v7_5_historical_noninferiority_report.json",
        "report_markdown": run_dir / "v7_5_historical_noninferiority_report.md",
        "veto_attribution": run_dir / "v7_5_loss_tail_veto_attribution_report.json",
        "leakage_audit": run_dir / "v7_5_fit_leakage_audit.json",
        "outer_oof_rows": run_dir / "v7_5_outer_oof_policy_rows.jsonl",
        "candidate_selected_rows": run_dir / "v7_5_candidate_selected_rows.jsonl",
        "v6_7_baseline_selected_rows": (
            run_dir / "v7_5_v6_7_baseline_selected_rows.jsonl"
        ),
    }
    _write_json(outputs["model"], model)
    _write_json(outputs["report"], report)
    _write_text(outputs["report_markdown"], _report_markdown(report))
    _write_json(outputs["veto_attribution"], attribution)
    _write_json(outputs["leakage_audit"], leakage)
    _write_jsonl(outputs["outer_oof_rows"], fit["outer_oof_rows"])
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
        "historical_noninferiority_gate_passed": model[
            "historical_noninferiority_gate_passed"
        ],
        "historical_policy_difference_market_count": model[
            "historical_policy_difference_market_count"
        ],
        "historical_actionability_passed": model["historical_actionability_passed"],
        "model_improvement_demonstrated": model["model_improvement_demonstrated"],
        "final_selected_veto_profile_name": model["final_nested_selection"][
            "selected_veto_profile_name"
        ],
        "target_free_canary_collection_allowed": model[
            "target_free_canary_collection_allowed"
        ],
        "target_free_canary_collection_blocking_reason_codes": model[
            "target_free_canary_collection_blocking_reason_codes"
        ],
        "target_free_canary_started": False,
        "fit_leakage_audit_passed": leakage["fit_leakage_audit_passed"],
        **_v7_0_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v7_5_historical_fit_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "model": model,
        "report": report,
        "leakage": leakage,
        "attribution": attribution,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "outputs": outputs,
    }


def _nested_select_profile(
    train_order: list[str],
    *,
    by_market: dict[str, dict[str, Any]],
    profile: dict[str, Any],
    cache: dict[tuple[str, ...], dict[str, Any]],
) -> dict[str, Any]:
    selector = profile["nested_selector"]
    initial = int(selector["inner_prequential_initial_market_count"])
    evaluation_order = train_order[initial:]
    if len(evaluation_order) < int(selector["minimum_inner_evaluation_market_count"]):
        raise ValueError("#237 inner prequential support invalid")
    rows_by_profile = {name: [] for name in PROFILE_NAMES}
    available_counts = Counter()
    profiles = {item["name"]: item for item in profile["veto_profiles"]}
    for index, market_id in enumerate(evaluation_order, start=initial):
        prior_order = train_order[:index]
        example = by_market[market_id]
        artifact = _fit_loss_classifier(
            prior_order, by_market=by_market, profile=profile, cache=cache
        )
        for name in PROFILE_NAMES:
            if artifact["available"] and example["baseline_trade_selected"]:
                available_counts[name] += 1
            rows_by_profile[name].append(
                _score_example(
                    example,
                    veto_profile=profiles[name],
                    classifier_artifact=artifact,
                    fold_index=None,
                )
            )
    metrics = {
        name: _policy_metrics(rows, fixed_size=0.2)
        for name, rows in rows_by_profile.items()
    }
    keep = metrics["KEEP_V6_7"]
    minimum_available = int(
        selector["minimum_profile_available_active_market_count"]
    )
    minimum_difference = int(selector["minimum_policy_difference_market_count"])
    eligible = []
    for name in PROFILE_NAMES:
        if name == "KEEP_V6_7":
            continue
        item = metrics[name]
        if (
            available_counts[name] >= minimum_available
            and item["policy_difference_market_count"] >= minimum_difference
            and item["total_after_cost_net_pnl_at_frozen_size"]
            >= keep["total_after_cost_net_pnl_at_frozen_size"]
            and item["largest_winner_removed_after_cost_net_pnl_at_frozen_size"]
            >= keep["largest_winner_removed_after_cost_net_pnl_at_frozen_size"]
        ):
            eligible.append(name)
    selected = (
        sorted(
            eligible,
            key=lambda name: (
                -metrics[name]["total_after_cost_net_pnl_at_frozen_size"],
                -metrics[name][
                    "largest_winner_removed_after_cost_net_pnl_at_frozen_size"
                ],
                metrics[name]["policy_difference_market_count"],
                PROFILE_NAMES.index(name),
            ),
        )[0]
        if eligible
        else "KEEP_V6_7"
    )
    result = {
        "selected_veto_profile_name": selected,
        "inner_train_market_count": len(train_order),
        "inner_evaluation_market_count": len(evaluation_order),
        "profile_available_active_market_counts": dict(sorted(available_counts.items())),
        "profile_metrics": metrics,
        "eligible_alternative_veto_profile_names": sorted(eligible),
        "outer_oof_results_used_for_selection": False,
        "selection_uses_strictly_prior_training_markets_only": True,
    }
    result["nested_selection_id"] = canonical_json_sha256(result)
    return result


def _fit_loss_classifier(
    market_order: list[str],
    *,
    by_market: dict[str, dict[str, Any]],
    profile: dict[str, Any],
    cache: dict[tuple[str, ...], dict[str, Any]],
) -> dict[str, Any]:
    key = tuple(market_order)
    if key in cache:
        return cache[key]
    active = [
        by_market[market_id]
        for market_id in market_order
        if by_market[market_id]["baseline_trade_selected"]
    ]
    losses = [row for row in active if float(row["baseline_target"]) <= 0.0]
    non_losses = [row for row in active if float(row["baseline_target"]) > 0.0]
    cfg = profile["loss_classifier"]
    unavailable = []
    if len(active) < int(cfg["minimum_prior_active_market_count"]):
        unavailable.append("prior_active_market_support_insufficient")
    if len(losses) < int(cfg["minimum_prior_loss_market_count"]):
        unavailable.append("prior_loss_market_support_insufficient")
    if len(non_losses) < int(cfg["minimum_prior_non_loss_market_count"]):
        unavailable.append("prior_non_loss_market_support_insufficient")
    if unavailable:
        artifact = {
            "available": False,
            "unavailable_reason_codes": unavailable,
            "fit_market_count": len(market_order),
            "fit_active_market_count": len(active),
            "fit_loss_market_count": len(losses),
            "fit_non_loss_market_count": len(non_losses),
        }
        cache[key] = artifact
        return artifact
    x = np.asarray([row["baseline_features"] for row in active], dtype=float)
    y = np.asarray([float(row["baseline_target"]) <= 0.0 for row in active], dtype=float)
    if x.ndim != 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("#237 non-finite classifier training values")
    params = dict(cfg)
    rounds = int(params.pop("num_boost_round"))
    for key_name in (
        "early_stopping_enabled",
        "canonical_features_must_be_finite",
        "loss_target_definition",
        "minimum_prior_active_market_count",
        "minimum_prior_loss_market_count",
        "minimum_prior_non_loss_market_count",
    ):
        params.pop(key_name)
    booster = xgb.train(
        params,
        xgb.DMatrix(x, label=y, missing=np.nan),
        num_boost_round=rounds,
        verbose_eval=False,
    )
    risk = booster.predict(xgb.DMatrix(x, missing=np.nan))
    raw = bytes(booster.save_raw(raw_format="json"))
    thresholds = {
        item["name"]: (
            None
            if item["name"] == "KEEP_V6_7"
            else float(
                np.quantile(
                    risk,
                    1.0 - float(item["veto_fraction"]),
                    method="higher",
                )
            )
        )
        for item in profile["veto_profiles"]
    }
    artifact = {
        "available": True,
        "fit_market_count": len(market_order),
        "fit_active_market_count": len(active),
        "fit_loss_market_count": len(losses),
        "fit_non_loss_market_count": len(non_losses),
        "feature_count": int(x.shape[1]),
        "booster_json_base64": base64.b64encode(raw).decode("ascii"),
        "booster_sha256": canonical_json_sha256(
            base64.b64encode(raw).decode("ascii")
        ),
        "loss_probability_threshold_by_profile": thresholds,
        "training_risk_minimum": float(np.min(risk)),
        "training_risk_maximum": float(np.max(risk)),
        "training_risk_mean": float(np.mean(risk)),
        "xgboost_parameters": cfg,
        "fit_market_ids_hash": canonical_json_sha256(market_order),
    }
    cache[key] = artifact
    return artifact


def _score_example(
    example: dict[str, Any],
    *,
    veto_profile: dict[str, Any],
    classifier_artifact: dict[str, Any],
    fold_index: int | None,
    inference: bool = False,
) -> dict[str, Any]:
    profile_name = str(veto_profile["name"])
    if not example["baseline_trade_selected"]:
        return {
            **example,
            "fold_index": fold_index,
            "selected_veto_profile_name": profile_name,
            "classifier_available": bool(classifier_artifact["available"]),
            "predicted_baseline_loss_probability": None,
            "loss_probability_veto_threshold": None,
            "selected_policy_decision": "KEEP_V6_7",
            "selected_action": "NO_TRADE",
            "selected_side": "NONE",
            "selected_target_after_cost_net_pnl_per_contract": 0.0,
            "baseline_target_after_cost_net_pnl_per_contract": 0.0,
            "target_used_as_decision_time_input": False,
            "outer_validation_target_used_for_profile_selection_or_fit": False,
        }
    risk = None
    threshold = None
    decision = "KEEP_V6_7"
    if profile_name != "KEEP_V6_7" and classifier_artifact["available"]:
        risk = _predict_loss_probability(
            example["baseline_features"], classifier_artifact
        )
        threshold = float(
            classifier_artifact["loss_probability_threshold_by_profile"][profile_name]
        )
        if risk >= threshold:
            decision = "VETO_TO_NO_TRADE"
    selected_action = (
        "NO_TRADE" if decision == "VETO_TO_NO_TRADE" else example["baseline_action"]
    )
    selected_side = (
        "NONE" if decision == "VETO_TO_NO_TRADE" else example["baseline_side"]
    )
    baseline_target = None if inference else float(example["baseline_target"])
    selected_target = (
        None
        if inference
        else 0.0
        if decision == "VETO_TO_NO_TRADE"
        else baseline_target
    )
    return {
        **example,
        "fold_index": fold_index,
        "selected_veto_profile_name": profile_name,
        "classifier_available": bool(classifier_artifact["available"]),
        "predicted_baseline_loss_probability": risk,
        "loss_probability_veto_threshold": threshold,
        "selected_policy_decision": decision,
        "selected_action": selected_action,
        "selected_side": selected_side,
        "selected_target_after_cost_net_pnl_per_contract": selected_target,
        "baseline_target_after_cost_net_pnl_per_contract": baseline_target,
        "target_used_as_decision_time_input": False,
        "outer_validation_target_used_for_profile_selection_or_fit": False,
    }


def _predict_loss_probability(
    features: list[float], artifact: dict[str, Any]
) -> float:
    values = np.asarray([features], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("#237 non-finite inference feature")
    booster = xgb.Booster()
    booster.load_model(bytearray(base64.b64decode(artifact["booster_json_base64"])))
    prediction = float(booster.predict(xgb.DMatrix(values, missing=np.nan))[0])
    if not math.isfinite(prediction) or not 0.0 <= prediction <= 1.0:
        raise ValueError("#237 invalid loss probability")
    return prediction


def _policy_metrics(rows: list[dict[str, Any]], *, fixed_size: float) -> dict[str, Any]:
    candidate = [
        float(row["selected_target_after_cost_net_pnl_per_contract"]) * fixed_size
        for row in rows
    ]
    baseline = [
        float(row["baseline_target_after_cost_net_pnl_per_contract"]) * fixed_size
        for row in rows
    ]
    total = sum(candidate)
    baseline_total = sum(baseline)
    largest = max(max(candidate, default=0.0), 0.0)
    baseline_largest = max(max(baseline, default=0.0), 0.0)
    return {
        "evaluation_market_count": len(rows),
        "total_after_cost_net_pnl_at_frozen_size": total,
        "v6_7_total_after_cost_net_pnl_at_frozen_size": baseline_total,
        "candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size": (
            total - baseline_total
        ),
        "largest_winner_removed_after_cost_net_pnl_at_frozen_size": total - largest,
        "v6_7_largest_winner_removed_after_cost_net_pnl_at_frozen_size": (
            baseline_total - baseline_largest
        ),
        "policy_difference_market_count": sum(
            row["selected_policy_decision"] != "KEEP_V6_7" for row in rows
        ),
        "policy_decision_distribution": dict(
            sorted(Counter(row["selected_policy_decision"] for row in rows).items())
        ),
    }


def _veto_attribution(
    rows: list[dict[str, Any]], *, fixed_size: float
) -> dict[str, Any]:
    vetoed = [row for row in rows if row["selected_policy_decision"] == "VETO_TO_NO_TRADE"]
    removed = [
        float(row["baseline_target_after_cost_net_pnl_per_contract"]) * fixed_size
        for row in vetoed
    ]
    report = {
        "schema_version": ATTRIBUTION_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "evaluation_market_count": len(rows),
        "veto_market_count": len(vetoed),
        "vetoed_baseline_pnl_at_frozen_size": sum(removed),
        "avoided_negative_pnl_at_frozen_size": -sum(value for value in removed if value < 0),
        "removed_positive_pnl_at_frozen_size": sum(value for value in removed if value > 0),
        "veto_count_by_side": dict(
            sorted(Counter(row["baseline_side"] for row in vetoed).items())
        ),
        "vetoed_markets": [
            {
                "market_id": row["market_id"],
                "baseline_action": row["baseline_action"],
                "baseline_side": row["baseline_side"],
                "baseline_decision_ts": row["baseline_decision_ts"],
                "selected_veto_profile_name": row["selected_veto_profile_name"],
                "predicted_baseline_loss_probability": row[
                    "predicted_baseline_loss_probability"
                ],
                "loss_probability_veto_threshold": row[
                    "loss_probability_veto_threshold"
                ],
                "baseline_pnl_at_frozen_size": (
                    float(row["baseline_target_after_cost_net_pnl_per_contract"])
                    * fixed_size
                ),
            }
            for row in vetoed
        ],
        "outcome_fields_used_at_inference": False,
        **_v7_0_blocked_safety_fields(),
    }
    report["attribution_id"] = canonical_json_sha256(report)
    return report


def _leakage_audit(
    rows: list[dict[str, Any]], *, model: dict[str, Any]
) -> dict[str, Any]:
    checks = {
        "all_134_historical_markets_used": len({row["market_id"] for row in rows})
        == 134,
        "feature_timestamp_causality": all(
            int(row["max_input_ts"]) <= int(row["decision_ts"]) for row in rows
        ),
        "prior_result_artifacts_not_opened": model["prior_result_artifacts_opened"]
        is False,
        "outer_oof_not_used_for_selection": model[
            "outer_oof_pnl_used_for_profile_feature_hyperparameter_or_threshold_selection"
        ]
        is False,
        "outer_validation_target_isolation": model["historical_gate_checks"][
            "outer_validation_target_isolation"
        ],
        "veto_contract_preserved": model["historical_gate_checks"][
            "veto_contract_preserved"
        ],
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
        "prior_result_artifact_paths_accepted_by_config": False,
        "future_target_accessed": False,
        **_v7_0_blocked_safety_fields(),
    }
    audit["leakage_audit_id"] = canonical_json_sha256(audit)
    return audit


def _report(
    *,
    model: dict[str, Any],
    leakage: dict[str, Any],
    attribution: dict[str, Any],
) -> dict[str, Any]:
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "historical_noninferiority_gate_passed": model[
            "historical_noninferiority_gate_passed"
        ],
        "historical_gate_blocking_reason_codes": model[
            "historical_gate_blocking_reason_codes"
        ],
        "outer_fold_reports": model["outer_fold_reports"],
        "final_nested_selection": model["final_nested_selection"],
        "historical_replay_noninferiority_gate": model[
            "historical_replay_noninferiority_gate"
        ],
        "historical_policy_difference_market_count": model[
            "historical_policy_difference_market_count"
        ],
        "historical_actionability_passed": model["historical_actionability_passed"],
        "model_improvement_demonstrated": model["model_improvement_demonstrated"],
        "veto_attribution_summary": {
            key: attribution[key]
            for key in (
                "veto_market_count",
                "vetoed_baseline_pnl_at_frozen_size",
                "avoided_negative_pnl_at_frozen_size",
                "removed_positive_pnl_at_frozen_size",
                "veto_count_by_side",
            )
        },
        "fit_leakage_audit_passed": leakage["fit_leakage_audit_passed"],
        "target_free_canary_collection_allowed": model[
            "target_free_canary_collection_allowed"
        ],
        "target_free_canary_collection_blocking_reason_codes": model[
            "target_free_canary_collection_blocking_reason_codes"
        ],
        "target_free_canary_started": False,
        **_v7_0_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def _report_markdown(report: dict[str, Any]) -> str:
    replay = report["historical_replay_noninferiority_gate"]
    return "\n".join(
        [
            "# v7.5 Frozen-v6.7 Loss-Tail Veto Historical Replay",
            "",
            "- historical non-inferiority gate passed: "
            f"`{str(report['historical_noninferiority_gate_passed']).lower()}`",
            f"- blockers: `{report['historical_gate_blocking_reason_codes']}`",
            "- outer selected profiles: "
            f"`{[row['nested_selection']['selected_veto_profile_name'] for row in report['outer_fold_reports']]}`",
            "- final selected profile: "
            f"`{report['final_nested_selection']['selected_veto_profile_name']}`",
            "- candidate frozen-size PnL: "
            f"`{replay['candidate']['total_after_cost_net_pnl_at_frozen_size']}`",
            "- v6.7 frozen-size PnL: "
            f"`{replay['v6_7_baseline']['total_after_cost_net_pnl_at_frozen_size']}`",
            "- candidate-minus-v6.7 PnL: "
            f"`{replay['candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size']}`",
            "- historical policy differences: "
            f"`{report['historical_policy_difference_market_count']}`",
            "- model improvement demonstrated: "
            f"`{str(report['model_improvement_demonstrated']).lower()}`",
            "- target-free canary collection allowed: "
            f"`{str(report['target_free_canary_collection_allowed']).lower()}`",
            f"- leakage audit passed: `{str(report['fit_leakage_audit_passed']).lower()}`",
            "- paper/live/write/wallet/capital unlock: `false`",
            "",
        ]
    )


def _veto_contract_preserved(row: dict[str, Any]) -> bool:
    if not row["baseline_trade_selected"]:
        return row["selected_action"] == "NO_TRADE"
    if row["selected_policy_decision"] == "KEEP_V6_7":
        return (
            row["selected_action"] == row["baseline_action"]
            and row["selected_side"] == row["baseline_side"]
        )
    return (
        row["selected_policy_decision"] == "VETO_TO_NO_TRADE"
        and row["selected_action"] == "NO_TRADE"
        and row["selected_side"] == "NONE"
    )


def _profile_by_name(profile: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in profile["veto_profiles"] if item["name"] == name)


def _min_ts(rows: list[dict[str, Any]], market_ids: list[str]) -> int:
    selected = set(market_ids)
    return min(int(row["decision_ts"]) for row in rows if row["market_id"] in selected)


def _max_ts(rows: list[dict[str, Any]], market_ids: list[str]) -> int:
    selected = set(market_ids)
    return max(int(row["decision_ts"]) for row in rows if row["market_id"] in selected)


def _row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (int(row["baseline_decision_ts"] or 0), str(row["market_id"]))
