"""Nested training-only expert safe policy for issue #235."""

from __future__ import annotations

import math
import shutil
from collections import Counter
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
    _select_policy_decision,
    _validate_canonical_rows,
    _validate_target_free_market_rows,
)

CANDIDATE_NAME = "nested_training_only_expert_safe_policy_v7_3"
PROFILE_SCHEMA_VERSION = "bigan-v8-nested-expert-safe-policy-v7-3-profile-v1"
MODEL_SCHEMA_VERSION = "bigan-v8-nested-expert-safe-policy-v7-3-model-v1"
REPORT_SCHEMA_VERSION = "bigan-v8-nested-expert-safe-policy-v7-3-report-v1"
LEAKAGE_SCHEMA_VERSION = "bigan-v8-nested-expert-safe-policy-v7-3-leakage-v1"
MANIFEST_SCHEMA_VERSION = "bigan-v8-nested-expert-safe-policy-v7-3-manifest-v1"
EXPERT_NAMES = ("KEEP_V6_7", "KNN_QUANTILE_RELATIVE", "RIDGE_RELATIVE")
POLICY_DECISIONS = ("KEEP_V6_7", "SWITCH_SAME_DECISION_SBC", "NO_TRADE")
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
    "v7_2_implementation_commit": "b6d8dc0",
    "v7_2_evidence_commit": "f34a365",
}


@dataclass(frozen=True, slots=True)
class NestedExpertSafePolicyV73Config:
    """Pinned inputs for the one v7.3 historical fit and replay."""

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


def validate_nested_expert_safe_policy_v7_3_profile(
    profile: dict[str, Any],
) -> None:
    """Reject drift in expert library, nested selector, replay, or safety."""

    baseline = dict(profile.get("baseline_and_action_contract") or {})
    outer = dict(profile.get("outer_split") or {})
    selector = dict(profile.get("nested_selector") or {})
    library = dict(profile.get("expert_library") or {})
    freeze = dict(profile.get("final_freeze") or {})
    replay = dict(profile.get("historical_replay_superiority_gate") or {})
    canary = dict(profile.get("target_free_canary") or {})
    checks = {
        "identity": profile.get("schema_version") == PROFILE_SCHEMA_VERSION
        and profile.get("issue_number") == 235
        and profile.get("candidate_name") == CANDIDATE_NAME
        and profile.get("preregistered_before_implementation_and_fit") is True,
        "lineage": profile.get("lineage") == FROZEN_LINEAGE,
        "prior_exclusion": profile.get("prior_result_exclusion")
        == {
            "issue233_or_issue234_result_artifacts_accepted_as_inputs": False,
            "issue233_or_issue234_oof_rows_used": False,
            "issue233_or_issue234_pnl_used_for_expert_or_parameter_selection": False,
            "issue229_or_issue231_future_outcomes_used": False,
        },
        "baseline": baseline
        == {
            "baseline_candidate": "p_up_semantic_execution_compatibility_v6_7",
            "baseline_selection_rule": (
                "highest_positive_score_per_market_then_earliest_decision_ts_"
                "then_action"
            ),
            "baseline_is_default_action": True,
            "baseline_no_trade_cannot_be_activated": True,
            "allowed_policy_decisions": list(POLICY_DECISIONS),
            "opposite_action_same_decision_group_required": True,
            "alternative_decision_timestamp_allowed": False,
            "maximum_bets_per_market": 1,
            "side_quota_allowed": False,
            "full_five_action_interface_required": True,
            "hts_disabled_fail_closed": True,
        },
        "outer": outer
        == {
            "source_market_count": 134,
            "initial_training_market_count": 44,
            "forward_fold_count": 5,
            "validation_market_count_per_fold": 18,
            "outer_oof_market_count": 90,
            "market_order": "minimum_decision_ts_then_market_id",
            "outer_validation_targets_used_for_expert_selection_or_fit": False,
        },
        "selector": selector
        == {
            "inner_prequential_initial_market_count": 20,
            "inner_evaluation_no_bet_market_pnl": 0.0,
            "minimum_inner_evaluation_market_count": 20,
            "minimum_inner_expert_available_market_count": 10,
            "alternative_expert_total_pnl_must_strictly_exceed_keep": True,
            "alternative_expert_largest_winner_removed_must_not_be_worse": True,
            "selection_metric": "inner_total_after_cost_net_pnl_at_frozen_size",
            "tie_break": "highest_pnl_then_expert_name_lexicographic",
            "no_eligible_alternative_behavior": "KEEP_V6_7",
            "outer_oof_results_used_for_selection": False,
            "result_selected_rerun_allowed": False,
        },
        "library": tuple(library.get("expert_names") or ()) == EXPERT_NAMES
        and tuple(library.get("base_feature_names") or ()) == FEATURE_NAMES
        and library.get("switch_feature_construction")
        == "baseline_plus_opposite_plus_delta"
        and library.get("abstain_feature_construction") == "baseline_only"
        and library.get("ridge_alpha") == 100.0
        and library.get("ridge_coefficient_absolute_bound") == 8.0
        and library.get("ridge_uncertainty")
        == "leave_one_market_out_lower_residual_quantile"
        and library.get("knn_neighbor_count") == 15
        and library.get("knn_minimum_prior_baseline_active_market_count") == 15
        and library.get("knn_distance") == "fit_standardized_euclidean"
        and library.get("conditional_lower_quantile") == 0.2
        and library.get("upward_score_correction_allowed") is False
        and library.get("change_threshold") == 0.0
        and library.get("change_threshold_operator") == "strictly_greater_than"
        and library.get("hyperparameter_or_feature_search_enabled") is False,
        "freeze": freeze
        == {
            "final_expert_selected_by_same_inner_prequential_rule_on_all_134_markets": True,
            "outer_oof_result_used_for_final_expert_selection": False,
            "historical_targets_allowed_as_fixed_model_training_targets": True,
            "future_inference_decision_time_features_only": True,
        },
        "replay": replay
        == {
            "exact_evaluation_market_count": 90,
            "fixed_position_size": 0.2,
            "no_bet_market_pnl": 0.0,
            "common_selected_row_filter_allowed": False,
            "candidate_minus_v6_7_total_pnl_minimum_exclusive": 0.0,
            "candidate_minus_v6_7_largest_winner_removed_pnl_minimum_inclusive": 0.0,
            "minimum_policy_difference_market_count": 1,
            "exact_baseline_identity_reconciliation_required": True,
            "oof_pnl_used_for_feature_expert_hyperparameter_or_threshold_selection": False,
            "oof_pnl_used_for_pre_collection_screening_only": True,
            "failure_stops_before_collection": True,
            "promotion_or_paper_unlock_allowed": False,
        },
        "canary": canary
        == {
            "historical_superiority_gate_must_pass_before_collection": True,
            "strictly_later_outcome_blind_market_count": 12,
            "maximum_attempt_count": 18,
            "minimum_guard_accepted_policy_difference_market_count": 1,
            "outcome_label_or_pnl_access_allowed": False,
            "full_execution_guard_unchanged": True,
        },
        "safety": profile.get("safety") == _v7_0_blocked_safety_fields(),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("#235 v7.3 profile invalid: " + ", ".join(blockers))


def fit_nested_expert_safe_policy_v7_3(
    *,
    rows: list[dict[str, Any]],
    profile: dict[str, Any],
    implementation_commit: str,
    fit_created_ts: int,
) -> dict[str, Any]:
    """Run nested prior-only expert selection and the fixed outer replay."""

    validate_nested_expert_safe_policy_v7_3_profile(profile)
    _validate_canonical_rows(rows)
    market_order = _market_order(rows)
    examples = _relative_examples(rows, market_order=market_order)
    by_market = {row["market_id"]: row for row in examples}
    outer = profile["outer_split"]
    initial = int(outer["initial_training_market_count"])
    width = int(outer["validation_market_count_per_fold"])
    cache: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    outer_rows = []
    outer_reports = []
    for fold_index in range(int(outer["forward_fold_count"])):
        train_order = market_order[: initial + fold_index * width]
        validation_order = market_order[
            initial + fold_index * width : initial + (fold_index + 1) * width
        ]
        if len(validation_order) != width:
            raise ValueError("#235 outer fold support invalid")
        if _maximum_ts(rows, train_order) >= _minimum_ts(rows, validation_order):
            raise ValueError("#235 outer fold chronology invalid")
        selection = _nested_select_expert(
            train_order,
            by_market=by_market,
            profile=profile,
            cache=cache,
        )
        artifact = _fit_expert(
            selection["selected_expert_name"],
            train_order,
            by_market=by_market,
            profile=profile,
            cache=cache,
        )
        for market_id in validation_order:
            outer_rows.append(
                _score_example(
                    by_market[market_id],
                    expert_artifact=artifact,
                    fold_index=fold_index,
                )
            )
        outer_reports.append(
            {
                "fold_index": fold_index,
                "outer_train_market_count": len(train_order),
                "outer_validation_market_count": len(validation_order),
                "outer_train_max_decision_ts": _maximum_ts(rows, train_order),
                "outer_validation_min_decision_ts": _minimum_ts(
                    rows, validation_order
                ),
                "outer_validation_targets_used_for_expert_selection_or_fit": False,
                "nested_selection": selection,
            }
        )
    final_selection = _nested_select_expert(
        market_order,
        by_market=by_market,
        profile=profile,
        cache=cache,
    )
    final_artifact = _fit_expert(
        final_selection["selected_expert_name"],
        market_order,
        by_market=by_market,
        profile=profile,
        cache=cache,
    )
    replay = _historical_replay(outer_rows, profile=profile)
    candidate_rows = replay.pop("candidate_selected_rows")
    baseline_rows = replay.pop("v6_7_baseline_selected_rows")
    policy_difference_count = sum(
        row["selected_policy_decision"] != "KEEP_V6_7" for row in outer_rows
    )
    checks = {
        "exact_outer_oof_market_support": len({row["market_id"] for row in outer_rows})
        == int(outer["outer_oof_market_count"]),
        "outer_validation_target_isolation": all(
            report["outer_validation_targets_used_for_expert_selection_or_fit"]
            is False
            for report in outer_reports
        ),
        "outer_chronology": all(
            report["outer_train_max_decision_ts"]
            < report["outer_validation_min_decision_ts"]
            for report in outer_reports
        ),
        "same_decision_alternatives_only": all(
            row["opposite_decision_ts"] in {None, row["baseline_decision_ts"]}
            for row in outer_rows
        ),
        "nested_selector_did_not_use_outer_oof": all(
            report["nested_selection"]["outer_oof_results_used_for_selection"]
            is False
            for report in outer_reports
        ),
        "final_expert_did_not_use_outer_oof": final_selection[
            "outer_oof_results_used_for_selection"
        ]
        is False,
        "policy_difference_support": policy_difference_count
        >= int(
            profile["historical_replay_superiority_gate"][
                "minimum_policy_difference_market_count"
            ]
        ),
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
        "final_expert_available": final_artifact["available"],
    }
    reason_map = {
        "exact_outer_oof_market_support": "outer_oof_market_support_invalid",
        "outer_validation_target_isolation": "outer_validation_target_isolation_failed",
        "outer_chronology": "outer_fold_chronology_failed",
        "same_decision_alternatives_only": "alternative_decision_timestamp_used",
        "nested_selector_did_not_use_outer_oof": "outer_oof_used_for_expert_selection",
        "final_expert_did_not_use_outer_oof": "outer_oof_used_for_final_expert",
        "policy_difference_support": "candidate_identical_to_v6_7",
        "candidate_total_pnl_strictly_better_than_v6_7": (
            "historical_same_dataset_candidate_pnl_not_strictly_better_than_v6_7"
        ),
        "candidate_largest_winner_removed_not_worse_than_v6_7": (
            "historical_same_dataset_largest_winner_removed_pnl_worse_than_v6_7"
        ),
        "baseline_identity_reconciled": "frozen_v6_7_baseline_identity_mismatch",
        "final_expert_available": "final_nested_selected_expert_unavailable",
    }
    blockers = [reason_map[name] for name, passed in checks.items() if not passed]
    model = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": implementation_commit,
        "fit_created_ts": fit_created_ts,
        "frozen": not blockers,
        "decision_time_safe": True,
        "expert_library": list(EXPERT_NAMES),
        "outer_fold_reports": outer_reports,
        "final_nested_selection": final_selection,
        "final_expert_artifact": final_artifact,
        "historical_replay_superiority_gate": replay,
        "historical_policy_difference_market_count": policy_difference_count,
        "historical_gate_checks": checks,
        "historical_gate_passed": not blockers,
        "historical_gate_blocking_reason_codes": blockers,
        "issue233_or_issue234_result_artifacts_opened": False,
        "outer_oof_pnl_used_for_feature_expert_hyperparameter_or_threshold_selection": False,
        "historical_training_targets_used_inside_nested_training_only_selection": True,
        "historical_replay_is_promotion_evidence": False,
        "target_free_canary_collection_allowed": not blockers,
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


def score_nested_expert_safe_policy_v7_3_market(
    rows: list[dict[str, Any]], *, model_artifact: dict[str, Any]
) -> dict[str, Any]:
    """Outcome-free future consumer for the final nested-selected expert."""

    reasons = []
    if model_artifact.get("schema_version") != MODEL_SCHEMA_VERSION:
        reasons.append("v7_3_model_artifact_schema_invalid")
    if model_artifact.get("historical_gate_passed") is not True:
        reasons.append("v7_3_historical_gate_not_passed")
    if any(FORBIDDEN_INFERENCE_FIELDS.intersection(row) for row in rows):
        reasons.append("v7_3_forbidden_outcome_field_in_inference_row")
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
    decision = _score_example(
        {**example, "baseline_trade_selected": True},
        expert_artifact=model_artifact["final_expert_artifact"],
        fold_index=None,
        inference=True,
    )
    selected_action = decision["selected_action"]
    result = {
        "market_id": decision["market_id"],
        "selected_expert_name": model_artifact["final_nested_selection"][
            "selected_expert_name"
        ],
        "baseline_action": decision["baseline_action"],
        "baseline_decision_ts": decision["baseline_decision_ts"],
        "opposite_action": decision["opposite_action"],
        "opposite_decision_ts": decision["opposite_decision_ts"],
        "selected_policy_decision": decision["selected_policy_decision"],
        "selected_action": selected_action,
        "selected_side": decision["selected_side"],
        "trade_selected": selected_action != "NO_TRADE",
        "switch_advantage_lcb": decision["switch_advantage_lcb"],
        "no_trade_advantage_lcb": decision["no_trade_advantage_lcb"],
        "alternative_decision_timestamp_used": False,
        "source_score_mutated": False,
        "outcome_or_pnl_field_used_at_inference": False,
        "full_five_action_interface": _full_action_diagnostics(
            baseline, opposite, selected_action=selected_action
        ),
        "selection_reason_codes": [],
        **_v7_0_blocked_safety_fields(),
    }
    result["decision_id"] = canonical_json_sha256(result)
    return result


def run_nested_expert_safe_policy_v7_3_fit(
    config: NestedExpertSafePolicyV73Config,
) -> dict[str, Any]:
    """Verify lineage, fit once, replay once, and write hashable evidence."""

    paths = {
        "profile": Path(config.profile_path).resolve(),
        "v7_0_training_profile": Path(config.v7_0_training_profile_path).resolve(),
        "v6_7_candidate_profile": Path(config.v6_7_candidate_profile_path).resolve(),
        "v7_2_relative_policy_source": Path(
            config.v7_2_relative_policy_source_path
        ).resolve(),
        "runtime_target_rows": Path(config.runtime_target_rows_path).resolve(),
    }
    _verify_pin(paths["profile"], config.expected_profile_sha256, "#235 profile")
    profile = _load_json(paths["profile"])
    validate_nested_expert_safe_policy_v7_3_profile(profile)
    for key in (
        "v7_0_training_profile",
        "v6_7_candidate_profile",
        "v7_2_relative_policy_source",
        "runtime_target_rows",
    ):
        _verify_pin(paths[key], profile["lineage"][f"{key}_sha256"], f"#235 {key}")
    v7_profile = _load_json(paths["v7_0_training_profile"])
    validate_v7_0_training_profile(v7_profile)
    validate_p_up_semantic_compatibility_v6_7_profile(
        _load_json(paths["v6_7_candidate_profile"])
    )
    rows = materialize_v7_0_sbc_rows(
        _load_jsonl(paths["runtime_target_rows"]), v7_profile
    )
    fit = fit_nested_expert_safe_policy_v7_3(
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
        "model": run_dir / "v7_3_nested_expert_model.json",
        "report": run_dir / "v7_3_historical_replay_report.json",
        "report_markdown": run_dir / "v7_3_historical_replay_report.md",
        "leakage_audit": run_dir / "v7_3_fit_leakage_audit.json",
        "outer_oof_rows": run_dir / "v7_3_outer_oof_policy_rows.jsonl",
        "candidate_selected_rows": run_dir / "v7_3_candidate_selected_rows.jsonl",
        "v6_7_baseline_selected_rows": (
            run_dir / "v7_3_v6_7_baseline_selected_rows.jsonl"
        ),
    }
    _write_json(outputs["model"], model)
    _write_json(outputs["report"], report)
    _write_text(outputs["report_markdown"], _report_markdown(report))
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
        "historical_gate_passed": model["historical_gate_passed"],
        "historical_gate_blocking_reason_codes": model[
            "historical_gate_blocking_reason_codes"
        ],
        "final_selected_expert_name": model["final_nested_selection"][
            "selected_expert_name"
        ],
        "target_free_canary_collection_allowed": model[
            "target_free_canary_collection_allowed"
        ],
        "target_free_canary_started": False,
        "fit_leakage_audit_passed": leakage["fit_leakage_audit_passed"],
        **_v7_0_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v7_3_historical_fit_manifest.json"
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


def _nested_select_expert(
    train_order: list[str],
    *,
    by_market: dict[str, dict[str, Any]],
    profile: dict[str, Any],
    cache: dict[tuple[str, tuple[str, ...]], dict[str, Any]],
) -> dict[str, Any]:
    initial = int(profile["nested_selector"]["inner_prequential_initial_market_count"])
    evaluation_order = train_order[initial:]
    if len(evaluation_order) < int(
        profile["nested_selector"]["minimum_inner_evaluation_market_count"]
    ):
        raise ValueError("#235 inner prequential support invalid")
    rows_by_expert: dict[str, list[dict[str, Any]]] = {
        name: [] for name in EXPERT_NAMES
    }
    available_counts = Counter()
    for index, market_id in enumerate(evaluation_order, start=initial):
        prior_order = train_order[:index]
        example = by_market[market_id]
        for expert_name in EXPERT_NAMES:
            artifact = _fit_expert(
                expert_name,
                prior_order,
                by_market=by_market,
                profile=profile,
                cache=cache,
            )
            if artifact["available"] and example["baseline_trade_selected"]:
                available_counts[expert_name] += 1
            rows_by_expert[expert_name].append(
                _score_example(
                    example,
                    expert_artifact=artifact,
                    fold_index=None,
                )
            )
    metrics = {
        name: _inner_metrics(rows, fixed_size=0.2)
        for name, rows in rows_by_expert.items()
    }
    keep = metrics["KEEP_V6_7"]
    minimum_available = int(
        profile["nested_selector"]["minimum_inner_expert_available_market_count"]
    )
    eligible = []
    for expert_name in EXPERT_NAMES:
        if expert_name == "KEEP_V6_7":
            continue
        item = metrics[expert_name]
        if (
            available_counts[expert_name] >= minimum_available
            and item["total_after_cost_net_pnl_at_frozen_size"]
            > keep["total_after_cost_net_pnl_at_frozen_size"]
            and item["largest_winner_removed_after_cost_net_pnl_at_frozen_size"]
            >= keep["largest_winner_removed_after_cost_net_pnl_at_frozen_size"]
        ):
            eligible.append(expert_name)
    selected = (
        sorted(
            eligible,
            key=lambda name: (
                -metrics[name]["total_after_cost_net_pnl_at_frozen_size"],
                name,
            ),
        )[0]
        if eligible
        else "KEEP_V6_7"
    )
    result = {
        "selected_expert_name": selected,
        "inner_train_market_count": len(train_order),
        "inner_evaluation_market_count": len(evaluation_order),
        "expert_available_market_counts": dict(sorted(available_counts.items())),
        "expert_metrics": metrics,
        "eligible_alternative_expert_names": sorted(eligible),
        "outer_oof_results_used_for_selection": False,
        "selection_uses_strictly_prior_training_markets_only": True,
    }
    result["nested_selection_id"] = canonical_json_sha256(result)
    return result


def _fit_expert(
    expert_name: str,
    market_order: list[str],
    *,
    by_market: dict[str, dict[str, Any]],
    profile: dict[str, Any],
    cache: dict[tuple[str, tuple[str, ...]], dict[str, Any]],
) -> dict[str, Any]:
    key = (expert_name, tuple(market_order))
    if key in cache:
        return cache[key]
    active = [
        by_market[market_id]
        for market_id in market_order
        if by_market[market_id]["baseline_trade_selected"]
    ]
    if expert_name == "KEEP_V6_7":
        artifact = {
            "expert_name": expert_name,
            "available": True,
            "fit_market_count": len(market_order),
            "fit_baseline_active_market_count": len(active),
        }
    elif expert_name == "KNN_QUANTILE_RELATIVE":
        artifact = _fit_knn(active, profile=profile)
    elif expert_name == "RIDGE_RELATIVE":
        artifact = _fit_ridge_relative(active, profile=profile)
    else:
        raise ValueError(f"#235 unknown expert: {expert_name}")
    artifact["fit_market_ids_hash"] = canonical_json_sha256(market_order)
    cache[key] = artifact
    return artifact


def _fit_knn(examples: list[dict[str, Any]], *, profile: dict[str, Any]) -> dict[str, Any]:
    minimum = int(
        profile["expert_library"]["knn_minimum_prior_baseline_active_market_count"]
    )
    if len(examples) < minimum:
        return {
            "expert_name": "KNN_QUANTILE_RELATIVE",
            "available": False,
            "unavailable_reason_codes": ["knn_prior_baseline_active_support_insufficient"],
            "fit_baseline_active_market_count": len(examples),
        }
    switch_x = np.asarray([row["switch_features"] for row in examples], dtype=float)
    baseline_x = np.asarray([row["baseline_features"] for row in examples], dtype=float)
    return {
        "expert_name": "KNN_QUANTILE_RELATIVE",
        "available": True,
        "neighbor_count": int(profile["expert_library"]["knn_neighbor_count"]),
        "conditional_lower_quantile": float(
            profile["expert_library"]["conditional_lower_quantile"]
        ),
        "switch_feature_mean": switch_x.mean(axis=0).tolist(),
        "switch_feature_scale": _scale(switch_x).tolist(),
        "baseline_feature_mean": baseline_x.mean(axis=0).tolist(),
        "baseline_feature_scale": _scale(baseline_x).tolist(),
        "switch_features": switch_x.tolist(),
        "baseline_features": baseline_x.tolist(),
        "switch_targets": [float(row["switch_advantage_target"]) for row in examples],
        "no_trade_targets": [
            float(row["no_trade_advantage_target"]) for row in examples
        ],
        "training_market_ids": [row["market_id"] for row in examples],
        "fit_baseline_active_market_count": len(examples),
        "historical_targets_are_frozen_model_training_values": True,
    }


def _fit_ridge_relative(
    examples: list[dict[str, Any]], *, profile: dict[str, Any]
) -> dict[str, Any]:
    minimum = int(
        profile["expert_library"]["knn_minimum_prior_baseline_active_market_count"]
    )
    if len(examples) < minimum:
        return {
            "expert_name": "RIDGE_RELATIVE",
            "available": False,
            "unavailable_reason_codes": ["ridge_lomo_support_insufficient"],
            "fit_baseline_active_market_count": len(examples),
        }
    switch_residuals = []
    no_trade_residuals = []
    for index, held_out in enumerate(examples):
        train = examples[:index] + examples[index + 1 :]
        switch_model = _fit_ridge(
            train,
            feature_field="switch_features",
            target_field="switch_advantage_target",
            profile=profile,
        )
        no_trade_model = _fit_ridge(
            train,
            feature_field="baseline_features",
            target_field="no_trade_advantage_target",
            profile=profile,
        )
        switch_residuals.append(
            float(held_out["switch_advantage_target"])
            - _predict_ridge(held_out["switch_features"], switch_model)
        )
        no_trade_residuals.append(
            float(held_out["no_trade_advantage_target"])
            - _predict_ridge(held_out["baseline_features"], no_trade_model)
        )
    quantile = float(profile["expert_library"]["conditional_lower_quantile"])
    switch_model = _fit_ridge(
        examples,
        feature_field="switch_features",
        target_field="switch_advantage_target",
        profile=profile,
    )
    no_trade_model = _fit_ridge(
        examples,
        feature_field="baseline_features",
        target_field="no_trade_advantage_target",
        profile=profile,
    )
    return {
        "expert_name": "RIDGE_RELATIVE",
        "available": switch_model["valid"] and no_trade_model["valid"],
        "switch_model": switch_model,
        "no_trade_model": no_trade_model,
        "switch_lower_residual_correction": min(
            _quantile(switch_residuals, quantile), 0.0
        ),
        "no_trade_lower_residual_correction": min(
            _quantile(no_trade_residuals, quantile), 0.0
        ),
        "lomo_residual_market_count": len(examples),
        "fit_baseline_active_market_count": len(examples),
        "upward_correction_allowed": False,
    }


def _fit_ridge(
    examples: list[dict[str, Any]],
    *,
    feature_field: str,
    target_field: str,
    profile: dict[str, Any],
) -> dict[str, Any]:
    x = np.asarray([row[feature_field] for row in examples], dtype=float)
    y = np.asarray([row[target_field] for row in examples], dtype=float)
    mean = x.mean(axis=0)
    scale = _scale(x)
    z = (x - mean) / scale
    design = np.column_stack((np.ones(len(z)), z))
    alpha = float(profile["expert_library"]["ridge_alpha"])
    penalty = np.diag([0.0, *([alpha] * x.shape[1])])
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ y,
    )
    bound = float(profile["expert_library"]["ridge_coefficient_absolute_bound"])
    valid = bool(
        np.all(np.isfinite(coefficients))
        and np.max(np.abs(coefficients), initial=0.0) <= bound
    )
    return {
        "feature_field": feature_field,
        "target_field": target_field,
        "ridge_alpha": alpha,
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "intercept": float(coefficients[0]),
        "coefficients": coefficients[1:].tolist(),
        "coefficient_absolute_bound": bound,
        "valid": valid,
    }


def _score_example(
    example: dict[str, Any],
    *,
    expert_artifact: dict[str, Any],
    fold_index: int | None,
    inference: bool = False,
) -> dict[str, Any]:
    if not example["baseline_trade_selected"]:
        return _scored_no_baseline(example, expert_artifact, fold_index=fold_index)
    expert_name = expert_artifact["expert_name"]
    switch_lcb = None
    no_trade_lcb = None
    expert_available = bool(expert_artifact["available"])
    if expert_name == "RIDGE_RELATIVE" and expert_available:
        switch_lcb = _predict_ridge(
            example["switch_features"], expert_artifact["switch_model"]
        ) + float(expert_artifact["switch_lower_residual_correction"])
        no_trade_lcb = _predict_ridge(
            example["baseline_features"], expert_artifact["no_trade_model"]
        ) + float(expert_artifact["no_trade_lower_residual_correction"])
    elif expert_name == "KNN_QUANTILE_RELATIVE" and expert_available:
        switch_lcb = _knn_quantile(
            example["switch_features"],
            training_features=expert_artifact["switch_features"],
            targets=expert_artifact["switch_targets"],
            mean=expert_artifact["switch_feature_mean"],
            scale=expert_artifact["switch_feature_scale"],
            neighbor_count=int(expert_artifact["neighbor_count"]),
            quantile=float(expert_artifact["conditional_lower_quantile"]),
        )
        no_trade_lcb = _knn_quantile(
            example["baseline_features"],
            training_features=expert_artifact["baseline_features"],
            targets=expert_artifact["no_trade_targets"],
            mean=expert_artifact["baseline_feature_mean"],
            scale=expert_artifact["baseline_feature_scale"],
            neighbor_count=int(expert_artifact["neighbor_count"]),
            quantile=float(expert_artifact["conditional_lower_quantile"]),
        )
    policy_decision = (
        _select_policy_decision(switch_lcb, no_trade_lcb)
        if expert_available and expert_name != "KEEP_V6_7"
        else "KEEP_V6_7"
    )
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
    selected_target = None
    if not inference:
        selected_target = (
            float(example["opposite_target"])
            if policy_decision == "SWITCH_SAME_DECISION_SBC"
            else 0.0
            if policy_decision == "NO_TRADE"
            else float(example["baseline_target"])
        )
    row = {
        **example,
        "fold_index": fold_index,
        "selected_expert_name": expert_name,
        "expert_available": expert_available,
        "switch_advantage_lcb": switch_lcb,
        "no_trade_advantage_lcb": no_trade_lcb,
        "selected_policy_decision": policy_decision,
        "selected_action": selected_action,
        "selected_side": selected_side,
        "selected_target_after_cost_net_pnl_per_contract": selected_target,
        "baseline_target_after_cost_net_pnl_per_contract": (
            None if inference else float(example["baseline_target"])
        ),
        "target_used_as_decision_time_input": False,
        "outer_validation_target_used_for_expert_selection_or_fit": False,
    }
    return row


def _scored_no_baseline(
    example: dict[str, Any],
    expert_artifact: dict[str, Any],
    *,
    fold_index: int | None,
) -> dict[str, Any]:
    return {
        **example,
        "fold_index": fold_index,
        "selected_expert_name": expert_artifact["expert_name"],
        "expert_available": bool(expert_artifact["available"]),
        "switch_advantage_lcb": None,
        "no_trade_advantage_lcb": None,
        "selected_policy_decision": "KEEP_V6_7",
        "selected_action": "NO_TRADE",
        "selected_side": "NONE",
        "selected_target_after_cost_net_pnl_per_contract": 0.0,
        "baseline_target_after_cost_net_pnl_per_contract": 0.0,
        "target_used_as_decision_time_input": False,
        "outer_validation_target_used_for_expert_selection_or_fit": False,
    }


def _inner_metrics(rows: list[dict[str, Any]], *, fixed_size: float) -> dict[str, Any]:
    candidate_pnl = [
        float(row["selected_target_after_cost_net_pnl_per_contract"]) * fixed_size
        for row in rows
    ]
    baseline_pnl = [
        float(row["baseline_target_after_cost_net_pnl_per_contract"]) * fixed_size
        for row in rows
    ]
    total = sum(candidate_pnl)
    baseline_total = sum(baseline_pnl)
    largest = max(max(candidate_pnl, default=0.0), 0.0)
    baseline_largest = max(max(baseline_pnl, default=0.0), 0.0)
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
    }


def _knn_quantile(
    features: list[float],
    *,
    training_features: list[list[float]],
    targets: list[float],
    mean: list[float],
    scale: list[float],
    neighbor_count: int,
    quantile: float,
) -> float:
    x = (np.asarray(features) - np.asarray(mean)) / np.asarray(scale)
    training = (np.asarray(training_features) - np.asarray(mean)) / np.asarray(scale)
    distances = np.mean((training - x) ** 2, axis=1)
    order = np.argsort(distances, kind="stable")[:neighbor_count]
    return _quantile([float(targets[index]) for index in order], quantile)


def _predict_ridge(features: list[float], model: dict[str, Any]) -> float:
    x = np.asarray(features, dtype=float)
    mean = np.asarray(model["feature_mean"], dtype=float)
    scale = np.asarray(model["feature_scale"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    return float(model["intercept"] + ((x - mean) / scale) @ coefficients)


def _scale(values: np.ndarray) -> np.ndarray:
    std = values.std(axis=0)
    return np.where(std > 1e-12, std, 1.0)


def _quantile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("#235 quantile values empty")
    ordered = sorted(float(value) for value in values)
    index = max(min(math.ceil(quantile * len(ordered)) - 1, len(ordered) - 1), 0)
    return ordered[index]


def _minimum_ts(rows: list[dict[str, Any]], market_ids: list[str]) -> int:
    selected = set(market_ids)
    return min(int(row["decision_ts"]) for row in rows if row["market_id"] in selected)


def _maximum_ts(rows: list[dict[str, Any]], market_ids: list[str]) -> int:
    selected = set(market_ids)
    return max(int(row["decision_ts"]) for row in rows if row["market_id"] in selected)


def _leakage_audit(
    rows: list[dict[str, Any]], *, model: dict[str, Any]
) -> dict[str, Any]:
    checks = {
        "all_134_historical_markets_used": len({row["market_id"] for row in rows})
        == 134,
        "feature_timestamp_causality": all(
            int(row["max_input_ts"]) <= int(row["decision_ts"]) for row in rows
        ),
        "prior_result_artifacts_not_opened": model[
            "issue233_or_issue234_result_artifacts_opened"
        ]
        is False,
        "outer_oof_not_used_for_selection": model[
            "outer_oof_pnl_used_for_feature_expert_hyperparameter_or_threshold_selection"
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
        "issue233_or_issue234_result_artifact_paths_accepted_by_config": False,
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
        "outer_fold_reports": model["outer_fold_reports"],
        "final_nested_selection": model["final_nested_selection"],
        "historical_replay_superiority_gate": model[
            "historical_replay_superiority_gate"
        ],
        "historical_policy_difference_market_count": model[
            "historical_policy_difference_market_count"
        ],
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
    replay = report["historical_replay_superiority_gate"]
    return "\n".join(
        [
            "# v7.3 Nested Training-Only Expert Historical Replay",
            "",
            f"- historical gate passed: `{str(report['historical_gate_passed']).lower()}`",
            f"- blockers: `{report['historical_gate_blocking_reason_codes']}`",
            "- outer selected experts: "
            f"`{[row['nested_selection']['selected_expert_name'] for row in report['outer_fold_reports']]}`",
            "- final selected expert: "
            f"`{report['final_nested_selection']['selected_expert_name']}`",
            "- candidate frozen-size PnL: "
            f"`{replay['candidate']['total_after_cost_net_pnl_at_frozen_size']}`",
            "- v6.7 frozen-size PnL: "
            f"`{replay['v6_7_baseline']['total_after_cost_net_pnl_at_frozen_size']}`",
            "- candidate-minus-v6.7 PnL: "
            f"`{replay['candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size']}`",
            f"- leakage audit passed: `{str(report['fit_leakage_audit_passed']).lower()}`",
            "- target-free canary started: `false`",
            "- paper/live/write/wallet/capital unlock: `false`",
            "",
        ]
    )


def _row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (int(row["baseline_decision_ts"] or 0), str(row["market_id"]))
