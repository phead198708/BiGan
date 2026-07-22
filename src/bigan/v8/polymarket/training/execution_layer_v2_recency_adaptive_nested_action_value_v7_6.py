"""Recency-adaptive nested action-value policy for issue #239."""

from __future__ import annotations

import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xgboost as xgb

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_abstention_aware_expected_net_pnl_v7_0 import (
    _v7_0_blocked_safety_fields,
)
from bigan.v8.polymarket.training.execution_layer_v2_abstention_aware_expected_net_pnl_v7_0_fit import (
    materialize_v7_0_sbc_rows,
    validate_v7_0_training_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_nested_boosted_action_value_v7_4 import (
    FROZEN_XGB,
    _fit_head,
    _max_ts,
    _min_ts,
    _policy_metrics,
    _score_example,
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

CANDIDATE_NAME = "recency_adaptive_nested_action_value_v7_6"
PROFILE_SCHEMA_VERSION = "bigan-v8-recency-adaptive-nested-action-value-v7-6-profile-v1"
MODEL_SCHEMA_VERSION = "bigan-v8-recency-adaptive-nested-action-value-v7-6-model-v1"
REPORT_SCHEMA_VERSION = "bigan-v8-recency-adaptive-nested-action-value-v7-6-report-v1"
LEAKAGE_SCHEMA_VERSION = "bigan-v8-recency-adaptive-nested-action-value-v7-6-leakage-v1"
MANIFEST_SCHEMA_VERSION = "bigan-v8-recency-adaptive-nested-action-value-v7-6-manifest-v1"
HEAD_NAMES = ("BASELINE_LOSS_VETO", "RELATIVE_SWITCH_VALUE")
TRAINING_WINDOWS = (30, 60, 0)
EDGE_BUFFERS = (0.0, 0.025, 0.05)
POLICY_PROFILE_NAMES = tuple(
    ["KEEP_V6_7"]
    + [
        f"BASELINE_VETO_W{window:03d}_B{int(buffer * 1000):03d}"
        for window in (30, 60)
        for buffer in EDGE_BUFFERS
    ]
    + [f"BASELINE_VETO_WALL_B{int(buffer * 1000):03d}" for buffer in EDGE_BUFFERS]
    + [
        f"RELATIVE_SWITCH_W{window:03d}_B{int(buffer * 1000):03d}"
        for window in (30, 60)
        for buffer in EDGE_BUFFERS
    ]
    + [f"RELATIVE_SWITCH_WALL_B{int(buffer * 1000):03d}" for buffer in EDGE_BUFFERS]
)
FROZEN_LINEAGE = {
    "runtime_target_rows_sha256": "1565116daeb2f5d4d8c33fefa507276f59251edd5ffb5f4f313041bcf9dbb0ec",
    "v7_0_training_profile_sha256": "1f66d8699b9727651538cc34a9a2a25ba5eaac5cfded75cf8f4a258b1b5d3f4a",
    "v6_7_candidate_profile_sha256": "cec55d243acd6bbf60a5e8474545b487086ddcd4d18073682ae7f2d4660d2248",
    "v7_2_relative_policy_source_sha256": "bc099273ca3c1db04062d40a5172e82d0ec075e2c63886cc42dd2dde67cc961a",
    "v7_4_boosted_source_sha256": "686b445307412cbc545b40fa87e41df7357c442fd8ab3d38af4c69b13e29e26a",
    "xgboost_version": "3.2.0",
}


@dataclass(frozen=True, slots=True)
class RecencyAdaptiveNestedActionValueV76Config:
    """Pinned inputs for the one #239 historical fit and replay."""

    run_id: str
    output_dir: Path | str
    profile_path: Path | str
    expected_profile_sha256: str
    v7_0_training_profile_path: Path | str
    v6_7_candidate_profile_path: Path | str
    v7_2_relative_policy_source_path: Path | str
    v7_4_boosted_source_path: Path | str
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
            "v7_4_boosted_source_path",
            "runtime_target_rows_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


def validate_recency_adaptive_nested_action_value_v7_6_profile(
    profile: dict[str, Any],
) -> None:
    """Reject drift in the frozen #239 contract."""

    profiles = list(profile.get("policy_profiles") or [])
    expected_profiles = [
        {"name": "KEEP_V6_7", "head": "KEEP_V6_7", "training_window_market_count": 0, "edge_buffer": 0.0}
    ]
    for head, prefix in (
        ("BASELINE_LOSS_VETO", "BASELINE_VETO"),
        ("RELATIVE_SWITCH_VALUE", "RELATIVE_SWITCH"),
    ):
        for window in TRAINING_WINDOWS:
            window_name = "ALL" if window == 0 else f"{window:03d}"
            for buffer in EDGE_BUFFERS:
                expected_profiles.append(
                    {
                        "name": f"{prefix}_W{window_name}_B{int(buffer * 1000):03d}",
                        "head": head,
                        "training_window_market_count": window,
                        "edge_buffer": buffer,
                    }
                )
    checks = {
        "identity": profile.get("schema_version") == PROFILE_SCHEMA_VERSION
        and profile.get("issue_number") == 239
        and profile.get("candidate_name") == CANDIDATE_NAME
        and profile.get("preregistered_before_implementation_and_fit") is True,
        "lineage": profile.get("lineage") == FROZEN_LINEAGE,
        "design_split": profile.get("design_split")
        == {
            "market_count": 44,
            "maximum_decision_ts": 1784245320000,
            "sealed_outer_oof_minimum_decision_ts": 1784245560000,
            "sealed_outer_oof_targets_opened_for_design": False,
        },
        "prior_result_exclusion": profile.get("prior_result_exclusion")
        == {
            "issue238_manifest_report_or_target_rows_accepted_as_inputs": False,
            "issue238_side_action_or_pnl_diagnostics_used_for_design": False,
            "issue229_or_issue231_future_outcomes_used": False,
            "v7_4_or_v7_5_outer_oof_rows_used_for_profile_selection": False,
        },
        "baseline": profile.get("baseline_and_action_contract")
        == {
            "baseline_candidate": "p_up_semantic_execution_compatibility_v6_7",
            "baseline_is_default_action": True,
            "baseline_no_trade_cannot_be_activated": True,
            "allowed_policy_decisions": [
                "KEEP_V6_7",
                "SWITCH_SAME_DECISION_SBC",
                "VETO_TO_NO_TRADE",
            ],
            "opposite_action_same_decision_group_required": True,
            "alternative_decision_timestamp_allowed": False,
            "maximum_bets_per_market": 1,
            "side_quota_allowed": False,
            "full_five_action_interface_required": True,
            "hts_disabled_fail_closed": True,
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
        "xgboost": profile.get("xgboost") == FROZEN_XGB,
        "heads": profile.get("model_heads")
        == {
            "head_names": list(HEAD_NAMES),
            "baseline_relative_minimum_prior_active_market_count": 15,
            "baseline_veto_target": "v6_7_baseline_post_cost_net_return",
            "relative_switch_target": "opposite_minus_baseline_post_cost_net_return",
            "historical_targets_allowed_only_on_strictly_prior_training_markets": True,
        },
        "windows": profile.get("training_windows")
        == {
            "market_counts": [30, 60, 0],
            "zero_means_all_available_prior_markets": True,
            "window_applied_after_chronological_ordering": True,
            "outer_or_inner_validation_market_in_window_allowed": False,
        },
        "profiles": profiles == expected_profiles
        and tuple(item["name"] for item in profiles) == POLICY_PROFILE_NAMES,
        "selector": profile.get("nested_selector")
        == {
            "inner_prequential_initial_market_count": 20,
            "minimum_inner_evaluation_market_count": 20,
            "minimum_profile_available_active_market_count": 10,
            "minimum_policy_difference_market_count": 3,
            "inner_no_bet_market_pnl": 0.0,
            "alternative_total_pnl_must_be_noninferior_to_keep": True,
            "alternative_largest_winner_removed_must_be_noninferior_to_keep": True,
            "tie_break": "highest_pnl_then_highest_lwr_then_profile_name_lexicographic",
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
            "comparison_operator": "greater_than_or_equal",
            "equality_passes_noninferiority": True,
            "exact_baseline_identity_reconciliation_required": True,
            "oof_pnl_used_for_profile_feature_hyperparameter_window_or_threshold_selection": False,
            "oof_pnl_used_for_pre_collection_screening_only": True,
            "failure_stops_before_collection": True,
            "promotion_or_paper_unlock_allowed": False,
        },
        "canary": profile.get("target_free_canary")
        == {
            "historical_noninferiority_gate_must_pass_before_collection": True,
            "minimum_historical_policy_difference_market_count": 3,
            "final_selected_profile_must_not_be_keep_v6_7": True,
            "strictly_later_outcome_blind_market_count": 12,
            "minimum_guard_accepted_policy_difference_market_count": 1,
            "outcomes_resolution_labels_or_pnl_opened": False,
            "collection_allowed_only_after_historical_actionability": True,
        },
        "safety": profile.get("safety") == _v7_0_blocked_safety_fields(),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("#239 v7.6 profile invalid: " + ", ".join(blockers))


def fit_recency_adaptive_nested_action_value_v7_6(
    *,
    rows: list[dict[str, Any]],
    profile: dict[str, Any],
    implementation_commit: str,
    fit_created_ts: int,
) -> dict[str, Any]:
    """Select recency windows on prior markets and replay one exact outer OOF."""

    validate_recency_adaptive_nested_action_value_v7_6_profile(profile)
    _validate_canonical_rows(rows)
    order = _market_order(rows)
    examples = _relative_examples(rows, market_order=order)
    by_market = {row["market_id"]: row for row in examples}
    outer = profile["outer_split"]
    initial = int(outer["initial_training_market_count"])
    width = int(outer["validation_market_count_per_fold"])
    cache: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    outer_rows: list[dict[str, Any]] = []
    fold_reports = []
    for fold_index in range(int(outer["forward_fold_count"])):
        train_order = order[: initial + fold_index * width]
        validation_order = order[
            initial + fold_index * width : initial + (fold_index + 1) * width
        ]
        if len(validation_order) != width:
            raise ValueError("#239 outer fold support invalid")
        if _max_ts(rows, train_order) >= _min_ts(rows, validation_order):
            raise ValueError("#239 outer fold chronology invalid")
        selection = _nested_select_profile(
            train_order, by_market=by_market, profile=profile, cache=cache
        )
        selected_profile = _profile_by_name(
            profile, selection["selected_policy_profile_name"]
        )
        artifact = _fit_profile_head(
            selected_profile,
            train_order,
            by_market=by_market,
            profile=profile,
            cache=cache,
        )
        for market_id in validation_order:
            outer_rows.append(
                _score_example(
                    by_market[market_id],
                    policy_profile=selected_profile,
                    head_artifact=artifact,
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
        profile, final_selection["selected_policy_profile_name"]
    )
    final_artifact = _fit_profile_head(
        final_profile,
        order,
        by_market=by_market,
        profile=profile,
        cache=cache,
    )
    replay = _historical_replay(outer_rows, profile=profile)
    replay["gate_name"] = "same_dataset_historical_replay_noninferiority_to_v6_7"
    replay["gate_mode"] = "development_noninferiority_screening_only_before_new_collection"
    replay["comparison_operator"] = "greater_than_or_equal"
    replay["equality_passes_noninferiority"] = True
    candidate_rows = replay.pop("candidate_selected_rows")
    baseline_rows = replay.pop("v6_7_baseline_selected_rows")
    policy_difference_count = sum(
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
    gate = profile["historical_replay_superiority_gate"]
    checks = {
        "exact_outer_oof_market_support": len({row["market_id"] for row in outer_rows})
        == int(outer["outer_oof_market_count"]),
        "outer_validation_target_isolation": all(
            item["outer_validation_targets_used_for_profile_selection_or_fit"] is False
            for item in fold_reports
        ),
        "outer_chronology": all(
            item["outer_train_max_decision_ts"]
            < item["outer_validation_min_decision_ts"]
            for item in fold_reports
        ),
        "same_decision_alternatives_only": all(
            row["opposite_decision_ts"] in {None, row["baseline_decision_ts"]}
            for row in outer_rows
        ),
        "nested_selector_did_not_use_outer_oof": all(
            item["nested_selection"]["outer_oof_results_used_for_selection"] is False
            for item in fold_reports
        ),
        "final_profile_did_not_use_outer_oof": final_selection[
            "outer_oof_results_used_for_selection"
        ]
        is False,
        "candidate_total_pnl_noninferior_to_v6_7": total_delta
        >= float(gate["candidate_minus_v6_7_total_pnl_minimum_inclusive"]),
        "candidate_largest_winner_removed_noninferior_to_v6_7": lwr_delta
        >= float(
            gate[
                "candidate_minus_v6_7_largest_winner_removed_pnl_minimum_inclusive"
            ]
        ),
        "baseline_identity_reconciled": replay["baseline_identity_reconciled"],
        "final_selected_head_available": final_artifact["available"],
    }
    reason_map = {
        "exact_outer_oof_market_support": "outer_oof_market_support_invalid",
        "outer_validation_target_isolation": "outer_validation_target_isolation_failed",
        "outer_chronology": "outer_fold_chronology_failed",
        "same_decision_alternatives_only": "alternative_decision_timestamp_used",
        "nested_selector_did_not_use_outer_oof": "outer_oof_used_for_profile_selection",
        "final_profile_did_not_use_outer_oof": "outer_oof_used_for_final_profile",
        "candidate_total_pnl_noninferior_to_v6_7": "historical_same_dataset_candidate_pnl_worse_than_v6_7",
        "candidate_largest_winner_removed_noninferior_to_v6_7": "historical_same_dataset_largest_winner_removed_pnl_worse_than_v6_7",
        "baseline_identity_reconciled": "frozen_v6_7_baseline_identity_mismatch",
        "final_selected_head_available": "final_nested_selected_head_unavailable",
    }
    blockers = [reason_map[name] for name, passed in checks.items() if not passed]
    gate_passed = not blockers
    canary = profile["target_free_canary"]
    actionability_checks = {
        "historical_noninferiority_gate_passed": gate_passed,
        "minimum_historical_policy_difference_met": policy_difference_count
        >= int(canary["minimum_historical_policy_difference_market_count"]),
        "final_selected_profile_is_not_keep_v6_7": final_profile["name"]
        != "KEEP_V6_7",
    }
    actionability_blockers = [
        name for name, passed in actionability_checks.items() if not passed
    ]
    collection_allowed = not actionability_blockers
    model = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": implementation_commit,
        "fit_created_ts": fit_created_ts,
        "frozen": gate_passed,
        "decision_time_safe": True,
        "policy_profile_names": list(POLICY_PROFILE_NAMES),
        "outer_fold_reports": fold_reports,
        "final_nested_selection": final_selection,
        "final_policy_profile": final_profile,
        "final_head_artifact": final_artifact,
        "historical_replay_noninferiority_gate": replay,
        "historical_policy_difference_market_count": policy_difference_count,
        "historical_noninferiority_gate_passed": gate_passed,
        "historical_gate_passed": gate_passed,
        "historical_gate_checks": checks,
        "historical_gate_blocking_reason_codes": blockers,
        "historical_actionability_checks": actionability_checks,
        "historical_actionability_blocking_reason_codes": actionability_blockers,
        "model_improvement_demonstrated": total_delta > 0.0 and lwr_delta >= 0.0,
        "prior_result_artifacts_opened": False,
        "issue238_artifacts_opened": False,
        "outer_oof_pnl_used_for_profile_feature_hyperparameter_window_or_threshold_selection": False,
        "historical_training_targets_used_inside_nested_training_only_selection": True,
        "historical_replay_is_promotion_evidence": False,
        "target_free_canary_collection_allowed": collection_allowed,
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
    }


def score_recency_adaptive_nested_action_value_v7_6_market(
    rows: list[dict[str, Any]], *, model_artifact: dict[str, Any]
) -> dict[str, Any]:
    """Apply a historically actionable v7.6 artifact to target-free rows."""

    reasons = []
    if model_artifact.get("schema_version") != MODEL_SCHEMA_VERSION:
        reasons.append("v7_6_model_artifact_schema_invalid")
    if model_artifact.get("historical_noninferiority_gate_passed") is not True:
        reasons.append("v7_6_historical_noninferiority_gate_not_passed")
    if model_artifact.get("target_free_canary_collection_allowed") is not True:
        reasons.append("v7_6_historical_actionability_not_passed")
    if any(FORBIDDEN_INFERENCE_FIELDS.intersection(row) for row in rows):
        reasons.append("v7_6_forbidden_outcome_field_in_inference_row")
    if reasons:
        return _no_trade_inference(rows, reasons=reasons)
    _validate_target_free_market_rows(rows)
    baseline = _select_baseline(rows)
    if baseline is None:
        return _no_trade_inference(
            rows, reasons=["frozen_v6_7_no_positive_baseline_action"]
        )
    opposite = _same_decision_opposite(rows, baseline)
    decision = _score_example(
        {**_inference_example(baseline, opposite), "baseline_trade_selected": True},
        policy_profile=model_artifact["final_policy_profile"],
        head_artifact=model_artifact["final_head_artifact"],
        fold_index=None,
        inference=True,
    )
    result = {
        "market_id": decision["market_id"],
        "selected_policy_profile_name": decision["selected_policy_profile_name"],
        "baseline_action": decision["baseline_action"],
        "baseline_decision_ts": decision["baseline_decision_ts"],
        "opposite_action": decision["opposite_action"],
        "opposite_decision_ts": decision["opposite_decision_ts"],
        "selected_policy_decision": decision["selected_policy_decision"],
        "selected_action": decision["selected_action"],
        "selected_side": decision["selected_side"],
        "trade_selected": decision["selected_action"] != "NO_TRADE",
        "predicted_baseline_return": decision["predicted_baseline_return"],
        "predicted_opposite_return": decision["predicted_opposite_return"],
        "predicted_switch_advantage": decision["predicted_switch_advantage"],
        "training_window_market_count": model_artifact["final_policy_profile"][
            "training_window_market_count"
        ],
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


def run_recency_adaptive_nested_action_value_v7_6_fit(
    config: RecencyAdaptiveNestedActionValueV76Config,
) -> dict[str, Any]:
    """Verify pinned lineage, fit once, replay once, and write evidence."""

    paths = {
        "profile": Path(config.profile_path).resolve(),
        "v7_0_training_profile": Path(config.v7_0_training_profile_path).resolve(),
        "v6_7_candidate_profile": Path(config.v6_7_candidate_profile_path).resolve(),
        "v7_2_relative_policy_source": Path(
            config.v7_2_relative_policy_source_path
        ).resolve(),
        "v7_4_boosted_source": Path(config.v7_4_boosted_source_path).resolve(),
        "runtime_target_rows": Path(config.runtime_target_rows_path).resolve(),
    }
    _verify_pin(paths["profile"], config.expected_profile_sha256, "#239 profile")
    profile = _load_json(paths["profile"])
    validate_recency_adaptive_nested_action_value_v7_6_profile(profile)
    for key in (
        "v7_0_training_profile",
        "v6_7_candidate_profile",
        "v7_2_relative_policy_source",
        "v7_4_boosted_source",
        "runtime_target_rows",
    ):
        _verify_pin(paths[key], profile["lineage"][f"{key}_sha256"], f"#239 {key}")
    if xgb.__version__ != profile["lineage"]["xgboost_version"]:
        raise ValueError("#239 xgboost version mismatch")
    v7_profile = _load_json(paths["v7_0_training_profile"])
    validate_v7_0_training_profile(v7_profile)
    validate_p_up_semantic_compatibility_v6_7_profile(
        _load_json(paths["v6_7_candidate_profile"])
    )
    rows = materialize_v7_0_sbc_rows(
        _load_jsonl(paths["runtime_target_rows"]), v7_profile
    )
    fit = fit_recency_adaptive_nested_action_value_v7_6(
        rows=rows,
        profile=profile,
        implementation_commit=config.implementation_commit,
        fit_created_ts=config.fit_created_ts,
    )
    model = fit["model_artifact"]
    leakage = _leakage_audit(rows, model=model)
    report = _report(model=model, leakage=leakage)
    run_dir = Path(config.output_dir).resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    outputs = {
        "model": run_dir / "v7_6_recency_adaptive_model.json",
        "report": run_dir / "v7_6_historical_noninferiority_report.json",
        "report_markdown": run_dir / "v7_6_historical_noninferiority_report.md",
        "leakage_audit": run_dir / "v7_6_fit_leakage_audit.json",
        "outer_oof_rows": run_dir / "v7_6_outer_oof_policy_rows.jsonl",
        "candidate_selected_rows": run_dir / "v7_6_candidate_selected_rows.jsonl",
        "v6_7_baseline_selected_rows": run_dir / "v7_6_v6_7_baseline_selected_rows.jsonl",
    }
    _write_json(outputs["model"], model)
    _write_json(outputs["report"], report)
    _write_text(outputs["report_markdown"], _report_markdown(report))
    _write_json(outputs["leakage_audit"], leakage)
    _write_jsonl(outputs["outer_oof_rows"], fit["outer_oof_rows"])
    _write_jsonl(outputs["candidate_selected_rows"], fit["candidate_selected_rows"])
    _write_jsonl(outputs["v6_7_baseline_selected_rows"], fit["v6_7_baseline_selected_rows"])
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
        "historical_gate_blocking_reason_codes": model[
            "historical_gate_blocking_reason_codes"
        ],
        "historical_actionability_blocking_reason_codes": model[
            "historical_actionability_blocking_reason_codes"
        ],
        "model_improvement_demonstrated": model["model_improvement_demonstrated"],
        "final_selected_policy_profile_name": model["final_nested_selection"][
            "selected_policy_profile_name"
        ],
        "target_free_canary_collection_allowed": model[
            "target_free_canary_collection_allowed"
        ],
        "target_free_canary_started": False,
        "fit_leakage_audit_passed": leakage["fit_leakage_audit_passed"],
        **_v7_0_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v7_6_historical_fit_manifest.json"
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


def _nested_select_profile(
    train_order: list[str],
    *,
    by_market: dict[str, dict[str, Any]],
    profile: dict[str, Any],
    cache: dict[tuple[str, tuple[str, ...]], dict[str, Any]],
) -> dict[str, Any]:
    selector = profile["nested_selector"]
    initial = int(selector["inner_prequential_initial_market_count"])
    evaluation_order = train_order[initial:]
    if len(evaluation_order) < int(selector["minimum_inner_evaluation_market_count"]):
        raise ValueError("#239 inner prequential support invalid")
    profiles = {item["name"]: item for item in profile["policy_profiles"]}
    rows_by_profile = {name: [] for name in POLICY_PROFILE_NAMES}
    available_counts = Counter()
    fit_window_hashes: dict[str, list[str]] = {name: [] for name in POLICY_PROFILE_NAMES}
    for index, market_id in enumerate(evaluation_order, start=initial):
        prior_order = train_order[:index]
        example = by_market[market_id]
        for name in POLICY_PROFILE_NAMES:
            policy_profile = profiles[name]
            artifact = _fit_profile_head(
                policy_profile,
                prior_order,
                by_market=by_market,
                profile=profile,
                cache=cache,
            )
            fit_window_hashes[name].append(artifact["fit_market_ids_hash"])
            if artifact["available"] and example["baseline_trade_selected"]:
                available_counts[name] += 1
            rows_by_profile[name].append(
                _score_example(
                    example,
                    policy_profile=policy_profile,
                    head_artifact=artifact,
                    fold_index=None,
                )
            )
    metrics = {
        name: _policy_metrics(rows, fixed_size=0.2)
        for name, rows in rows_by_profile.items()
    }
    keep = metrics["KEEP_V6_7"]
    minimum_available = int(selector["minimum_profile_available_active_market_count"])
    minimum_difference = int(selector["minimum_policy_difference_market_count"])
    eligible = []
    for name in POLICY_PROFILE_NAMES:
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
                name,
            ),
        )[0]
        if eligible
        else "KEEP_V6_7"
    )
    result = {
        "selected_policy_profile_name": selected,
        "inner_train_market_count": len(train_order),
        "inner_evaluation_market_count": len(evaluation_order),
        "profile_available_active_market_counts": dict(sorted(available_counts.items())),
        "profile_metrics": metrics,
        "profile_fit_window_hashes": {
            name: canonical_json_sha256(values)
            for name, values in sorted(fit_window_hashes.items())
        },
        "eligible_alternative_policy_profile_names": sorted(eligible),
        "alternative_noninferiority_operator": "greater_than_or_equal",
        "outer_oof_results_used_for_selection": False,
        "selection_uses_strictly_prior_training_markets_only": True,
    }
    result["nested_selection_id"] = canonical_json_sha256(result)
    return result


def _fit_profile_head(
    policy_profile: dict[str, Any],
    prior_order: list[str],
    *,
    by_market: dict[str, dict[str, Any]],
    profile: dict[str, Any],
    cache: dict[tuple[str, tuple[str, ...]], dict[str, Any]],
) -> dict[str, Any]:
    window = int(policy_profile["training_window_market_count"])
    fit_order = prior_order[-window:] if window else prior_order
    artifact = _fit_head(
        str(policy_profile["head"]),
        fit_order,
        by_market=by_market,
        profile=profile,
        cache=cache,
    )
    return {
        **artifact,
        "configured_training_window_market_count": window,
        "available_prior_market_count": len(prior_order),
        "fit_window_market_count": len(fit_order),
        "fit_window_first_market_id": fit_order[0] if fit_order else None,
        "fit_window_last_market_id": fit_order[-1] if fit_order else None,
        "fit_window_is_suffix_of_strictly_prior_markets": fit_order
        == (prior_order[-window:] if window else prior_order),
    }


def _profile_by_name(profile: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in profile["policy_profiles"] if item["name"] == name)


def _leakage_audit(
    rows: list[dict[str, Any]], *, model: dict[str, Any]
) -> dict[str, Any]:
    checks = {
        "all_134_historical_markets_used": len({row["market_id"] for row in rows}) == 134,
        "feature_timestamp_causality": all(
            int(row["max_input_ts"]) <= int(row["decision_ts"]) for row in rows
        ),
        "issue238_artifacts_not_opened": model["issue238_artifacts_opened"] is False,
        "outer_oof_not_used_for_selection": model[
            "outer_oof_pnl_used_for_profile_feature_hyperparameter_window_or_threshold_selection"
        ]
        is False,
        "outer_validation_target_isolation": model["historical_gate_checks"][
            "outer_validation_target_isolation"
        ],
        "same_decision_only": model["historical_gate_checks"][
            "same_decision_alternatives_only"
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
        "issue238_artifact_paths_accepted_by_config": False,
        "future_target_accessed": False,
        **_v7_0_blocked_safety_fields(),
    }
    audit["leakage_audit_id"] = canonical_json_sha256(audit)
    return audit


def _report(model: dict[str, Any], leakage: dict[str, Any]) -> dict[str, Any]:
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "historical_noninferiority_gate_passed": model[
            "historical_noninferiority_gate_passed"
        ],
        "historical_gate_blocking_reason_codes": model[
            "historical_gate_blocking_reason_codes"
        ],
        "historical_actionability_blocking_reason_codes": model[
            "historical_actionability_blocking_reason_codes"
        ],
        "outer_fold_reports": model["outer_fold_reports"],
        "final_nested_selection": model["final_nested_selection"],
        "historical_replay_noninferiority_gate": model[
            "historical_replay_noninferiority_gate"
        ],
        "historical_policy_difference_market_count": model[
            "historical_policy_difference_market_count"
        ],
        "model_improvement_demonstrated": model["model_improvement_demonstrated"],
        "fit_leakage_audit_passed": leakage["fit_leakage_audit_passed"],
        "target_free_canary_collection_allowed": model[
            "target_free_canary_collection_allowed"
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
            "# v7.6 Recency-Adaptive Historical Non-Inferiority Replay",
            "",
            "- historical non-inferiority gate passed: "
            f"`{str(report['historical_noninferiority_gate_passed']).lower()}`",
            f"- hard-gate blockers: `{report['historical_gate_blocking_reason_codes']}`",
            "- final selected profile: "
            f"`{report['final_nested_selection']['selected_policy_profile_name']}`",
            "- candidate frozen-size PnL: "
            f"`{replay['candidate']['total_after_cost_net_pnl_at_frozen_size']}`",
            "- v6.7 frozen-size PnL: "
            f"`{replay['v6_7_baseline']['total_after_cost_net_pnl_at_frozen_size']}`",
            "- candidate-minus-v6.7 PnL: "
            f"`{replay['candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size']}`",
            "- historical policy differences: "
            f"`{report['historical_policy_difference_market_count']}`",
            "- collection actionability blockers: "
            f"`{report['historical_actionability_blocking_reason_codes']}`",
            "- target-free canary collection allowed: "
            f"`{str(report['target_free_canary_collection_allowed']).lower()}`",
            f"- leakage audit passed: `{str(report['fit_leakage_audit_passed']).lower()}`",
            "- paper/live/write/wallet/capital unlock: `false`",
            "",
        ]
    )


def _row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (int(row["baseline_decision_ts"] or 0), str(row["market_id"]))
