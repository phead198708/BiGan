"""Rolling-origin outcome-free score-rank abstention policy for issue #243."""

from __future__ import annotations

import math
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xgboost as xgb

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training import (
    execution_layer_v2_rolling_origin_conformal_positive_ev_v7_8 as v78,
)
from bigan.v8.polymarket.training import (
    execution_layer_v2_rolling_origin_drift_adaptive_action_value_v7_7 as v77,
)
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
from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7_post_freeze import (
    _runtime_targets_for_decisions,
)
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
    _descriptor,
    _load_json,
    _load_jsonl,
    _sha256_file,
    _verify_pin,
    _write_json,
    _write_jsonl,
    _write_text,
    validate_runtime_aligned_sbc_net_return_v6_4_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_v6_7_relative_safe_policy_v7_2 import (
    FORBIDDEN_INFERENCE_FIELDS,
    _historical_replay,
    _market_order,
    _no_trade_inference,
    _validate_canonical_rows,
)

CANDIDATE_NAME = "rolling_origin_score_rank_abstention_v7_9"
PROFILE_SCHEMA_VERSION = "bigan-v8-rolling-origin-score-rank-abstention-v7-9-profile-v1"
MODEL_SCHEMA_VERSION = "bigan-v8-rolling-origin-score-rank-abstention-v7-9-model-v1"
REPORT_SCHEMA_VERSION = "bigan-v8-rolling-origin-score-rank-abstention-v7-9-report-v1"
LEAKAGE_SCHEMA_VERSION = "bigan-v8-rolling-origin-score-rank-abstention-v7-9-leakage-v1"
MANIFEST_SCHEMA_VERSION = "bigan-v8-rolling-origin-score-rank-abstention-v7-9-manifest-v1"
SBC_ACTIONS = v77.SBC_ACTIONS
FROZEN_LINEAGE = dict(v78.FROZEN_LINEAGE)


@dataclass(frozen=True, slots=True)
class RollingOriginScoreRankAbstentionV79Config(
    v78.RollingOriginConformalPositiveEVV78Config
):
    """Pinned inputs for the single preregistered #243 historical replay."""


def validate_rolling_origin_score_rank_abstention_v7_9_profile(
    profile: dict[str, Any],
) -> None:
    """Reject drift from the preregistered #243 profile."""

    checks = {
        "identity": profile.get("schema_version") == PROFILE_SCHEMA_VERSION
        and profile.get("issue_number") == 243
        and profile.get("candidate_name") == CANDIDATE_NAME
        and profile.get("preregistered_before_implementation_and_fit") is True,
        "lineage": profile.get("lineage") == FROZEN_LINEAGE,
        "stream": profile.get("historical_stream")
        == {
            "seed_market_count": 134,
            "seed_initial_training_market_count": 40,
            "prequential_market_count": 120,
            "market_order": "minimum_selected_decision_ts_then_market_id",
            "current_market_target_available_before_prediction": False,
            "prior_stream_target_available_only_after_prediction_freeze_and_market_close": True,
            "prior_market_close_must_precede_current_decision": True,
            "stream_is_consumed_historical_development_not_promotion_evidence": True,
        },
        "exclusion": profile.get("prior_result_exclusion")
        == {
            "issue241_settlement_outcome_pnl_or_gate_result_accepted_as_input": False,
            "issue242_result_artifacts_or_current_stream_pnl_used_for_threshold_selection": False,
            "issue239_outer_oof_rows_or_metrics_accepted_as_inputs": False,
            "issue238_side_action_or_pnl_attribution_used_for_design_or_tuning": False,
            "result_selected_rerun_allowed": False,
        },
        "model": profile.get("model_contract")
        == {
            "model_family": "single_weighted_xgboost_joint_sbc_action_value",
            "training_actions": list(SBC_ACTIONS),
            "target": "runtime_aligned_after_cost_net_pnl_per_contract",
            "exponential_decay_half_life_markets": 60.0,
            "minimum_training_market_count": 40,
            "fixed_edge_buffer": 0.025,
            "profile_or_hyperparameter_search_allowed": False,
            "side_specific_model_or_quota_allowed": False,
            "point_policy_is_v7_7_unchanged": True,
            "v6_7_no_trade_activation_allowed": False,
            "same_decision_timestamp_required": True,
            "maximum_bets_per_market": 1,
            "hts_disabled_fail_closed": True,
            "full_execution_guard_applied_after_selection": True,
        },
        "rank": profile.get("rank_abstention_contract")
        == {
            "score_source": "v7_7_point_selected_predicted_after_cost_return",
            "score_history_target_or_outcome_free": True,
            "eligible_prior_score_window": 60,
            "minimum_prior_eligible_score_count": 60,
            "rolling_score_quantile": 0.6,
            "finite_sample_rank": "ceil(n_times_quantile)_capped_at_n",
            "current_market_score_available_before_threshold": False,
            "point_selected_score_must_be_strictly_positive": True,
            "point_selected_score_must_be_at_or_above_prior_q60": True,
            "point_veto_remains_no_trade": True,
            "otherwise_abstain_to_no_trade": True,
            "quantile_window_threshold_or_profile_search_allowed": False,
        },
        "xgboost": profile.get("xgboost") == v77.FROZEN_XGB,
        "gate": profile.get("historical_prequential_hard_gate")
        == {
            "exact_evaluation_market_count": 120,
            "fixed_position_size": 0.2,
            "no_bet_market_pnl": 0.0,
            "minimum_candidate_guard_accepted_unique_market_count": 40,
            "candidate_minus_v6_7_total_pnl_minimum_inclusive": 0.0,
            "candidate_minus_v6_7_largest_winner_removed_pnl_minimum_inclusive": 0.0,
            "minimum_policy_difference_market_count_for_collection": 3,
            "exact_baseline_identity_reconciliation_required": True,
            "same_runtime_target_cost_size_guard_and_position_management_required": True,
            "failure_stops_before_collection": True,
            "historical_replay_is_promotion_evidence": False,
        },
        "canary": profile.get("target_free_canary")
        == {
            "strictly_later_outcome_blind_market_count": 12,
            "scan_cap": 18,
            "minimum_market_start_ts_exclusive": 1784806800000,
            "historical_hard_gate_required": True,
            "minimum_guard_accepted_policy_difference_market_count": 1,
            "minimum_guard_accepted_market_count": 4,
            "outcomes_resolution_labels_or_pnl_opened": False,
        },
        "future": profile.get("future_unseen_holdout")
        == {
            "exact_quality_valid_market_count": 120,
            "scan_cap": 180,
            "minimum_candidate_guard_accepted_unique_market_count": 40,
            "candidate_total_after_cost_pnl_minimum_exclusive": 0.0,
            "candidate_minus_v6_7_total_pnl_minimum_inclusive": 0.0,
            "candidate_minus_v6_7_largest_winner_removed_pnl_minimum_inclusive": 0.0,
            "single_use_gate": True,
            "passing_only_allows_promotion_discussion": True,
        },
        "safety": profile.get("safety") == _v7_0_blocked_safety_fields(),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("#243 v7.9 profile invalid: " + ", ".join(blockers))


def finite_sample_score_quantile(values: list[float], quantile: float) -> float:
    """Return the fixed q60 order statistic used by the rank policy."""

    if not values or not 0.0 < quantile < 1.0:
        raise ValueError("#243 rank quantile input invalid")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("#243 rank score is non-finite")
    rank = min(len(ordered), math.ceil(len(ordered) * quantile))
    return ordered[rank - 1]


def fit_rolling_origin_score_rank_abstention_v7_9(
    *,
    seed_rows: list[dict[str, Any]],
    stream_markets: list[dict[str, Any]],
    target_loader: Any,
    profile: dict[str, Any],
    v6_7_profile: dict[str, Any],
    implementation_commit: str,
    fit_created_ts: int,
    _profile_validator: Callable[[dict[str, Any]], None] = (
        validate_rolling_origin_score_rank_abstention_v7_9_profile
    ),
    _rank_controller: Callable[[tuple[bool, ...]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the one strictly-prior historical score-rank gate."""

    _profile_validator(profile)
    _validate_canonical_rows(seed_rows)
    seed_order = _market_order(seed_rows)
    expected_seed = int(profile["historical_stream"]["seed_market_count"])
    expected_stream = int(profile["historical_stream"]["prequential_market_count"])
    if len(seed_order) != expected_seed or len(stream_markets) != expected_stream:
        raise ValueError("#243 historical market support invalid")
    stream_markets = sorted(
        stream_markets,
        key=lambda row: (int(row["decision_ts"]), str(row["market_id"])),
    )
    if len({str(row["market_id"]) for row in stream_markets}) != expected_stream:
        raise ValueError("#243 duplicate prequential market")
    seed_max_close = max(int(row["market_close_ts"]) for row in seed_rows)
    if seed_max_close >= int(stream_markets[0]["decision_ts"]):
        raise ValueError("#243 seed/stream chronology invalid")
    seed_state = _seed_prediction_score_state(seed_rows, profile=profile)
    training_rows = list(seed_state["training_rows"])
    training_order = list(seed_state["training_market_order"])
    eligible_scores = list(seed_state["eligible_prediction_scores"])
    rank_lineage_rows = list(seed_state["rank_lineage_rows"])
    rank_contract = profile["rank_abstention_contract"]
    window_size = int(rank_contract["eligible_prior_score_window"])
    minimum_scores = int(rank_contract["minimum_prior_eligible_score_count"])
    quantile = float(rank_contract["rolling_score_quantile"])
    if len(eligible_scores) < minimum_scores:
        raise ValueError("#243 seed eligible prediction-score support insufficient")

    prequential_rows: list[dict[str, Any]] = []
    guard_replay_rows: list[dict[str, Any]] = []
    loaded_target_rows: list[dict[str, Any]] = []
    controller_guard_acceptance_history: list[bool] = []
    prior_close_ts = seed_max_close
    for stream_index, market in enumerate(stream_markets):
        if prior_close_ts >= int(market["decision_ts"]):
            raise ValueError("#243 prior target unavailable before current decision")
        prior_window = eligible_scores[-window_size:]
        if len(prior_window) != window_size:
            raise ValueError("#243 rolling score window incomplete")
        controller_decision = (
            _rank_controller(tuple(controller_guard_acceptance_history))
            if _rank_controller is not None
            else {
                "selected_quantile": quantile,
                "controller_enabled": False,
            }
        )
        current_quantile = float(controller_decision["selected_quantile"])
        threshold = finite_sample_score_quantile(prior_window, current_quantile)
        artifact = v77._fit_weighted_model(
            training_rows,
            market_order=training_order,
            profile=profile,
        )
        prediction = _score_rank_market(
            market,
            artifact=artifact,
            prior_score_threshold=threshold,
        )
        prediction.update(
            {
                "stream_index": stream_index,
                "prediction_frozen_before_current_target_access": True,
                "training_market_count": len(training_order),
                "training_max_decision_ts": max(
                    int(row["decision_ts"]) for row in training_rows
                ),
                "prior_rank_score_count": len(eligible_scores),
                "prior_rank_window_score_count": len(prior_window),
                "prior_rank_window_market_ids_hash": canonical_json_sha256(
                    [
                        row["market_id"]
                        for row in rank_lineage_rows
                        if row["eligible_prediction_score"]
                    ][-window_size:]
                ),
                "rank_state_max_decision_ts": max(
                    int(row["decision_ts"])
                    for row in rank_lineage_rows
                    if row["eligible_prediction_score"]
                ),
                "current_market_score_used_for_threshold": False,
                "current_market_target_used_for_prediction": False,
                "rank_controller_decision": controller_decision,
                "prior_controller_observation_count": len(
                    controller_guard_acceptance_history
                ),
                "prior_controller_history_hash": canonical_json_sha256(
                    controller_guard_acceptance_history
                ),
                "current_guard_result_used_for_own_controller_decision": False,
            }
        )
        guard = v78._execution_guard_result(
            prediction,
            market=market,
            v6_7_profile=v6_7_profile,
        )
        prediction.update(guard)
        controller_guard_acceptance_history.append(
            prediction["candidate_execution_guard_order_allowed"] is True
        )
        prediction["current_guard_result_added_after_decision_freeze"] = True
        current_rank_row = _rank_lineage_row(
            prediction,
            source_role="consumed_historical_prequential",
        )
        rank_lineage_rows.append(current_rank_row)
        if current_rank_row["eligible_prediction_score"]:
            eligible_scores.append(float(current_rank_row["point_selected_predicted_return"]))
        baseline_target, opposite_target = target_loader(
            market["baseline_row"], market["opposite_row"]
        )
        v78._validate_loaded_targets(
            market,
            baseline_target=baseline_target,
            opposite_target=opposite_target,
        )
        prediction.update(
            v77._attach_targets(
                prediction,
                baseline_target=baseline_target,
                opposite_target=opposite_target,
            )
        )
        prediction["current_market_target_accessed_only_after_prediction_freeze"] = True
        prequential_rows.append(prediction)
        guard_replay_rows.append(v78._guard_replay_row(prediction))
        loaded_target_rows.extend((baseline_target, opposite_target))
        training_rows.extend(
            (
                v77._canonical_stream_training_row(
                    market["baseline_row"],
                    baseline_target,
                    role="consumed_prequential",
                ),
                v77._canonical_stream_training_row(
                    market["opposite_row"],
                    opposite_target,
                    role="consumed_prequential",
                ),
            )
        )
        training_order.append(str(market["market_id"]))
        prior_close_ts = int(market["market_close_ts"])

    final_artifact = v77._fit_weighted_model(
        training_rows,
        market_order=training_order,
        profile=profile,
    )
    final_rank_state = _rank_state(
        eligible_scores,
        rank_lineage_rows=rank_lineage_rows,
        window_size=window_size,
        quantile=(
            float(
                _rank_controller(tuple(controller_guard_acceptance_history))[
                    "selected_quantile"
                ]
            )
            if _rank_controller is not None
            else quantile
        ),
    )
    replay_profile = {
        "historical_replay_superiority_gate": {
            "exact_evaluation_market_count": expected_stream,
            "fixed_position_size": profile["historical_prequential_hard_gate"][
                "fixed_position_size"
            ],
        }
    }
    replay = _historical_replay(guard_replay_rows, profile=replay_profile)
    replay["gate_name"] = "same_stream_score_rank_abstention_noninferiority_gate"
    replay["comparison_operator"] = "greater_than_or_equal"
    replay["equality_passes_noninferiority"] = True
    replay["full_execution_guard_applied_to_candidate_and_baseline"] = True
    candidate_rows = replay.pop("candidate_selected_rows")
    baseline_rows = replay.pop("v6_7_baseline_selected_rows")
    total_delta = float(
        replay["candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size"]
    )
    lwr_delta = float(
        replay[
            "candidate_minus_v6_7_largest_winner_removed_after_cost_net_pnl_at_frozen_size"
        ]
    )
    accepted_count = len(
        {
            str(row["market_id"])
            for row in guard_replay_rows
            if row["selected_action"] != "NO_TRADE"
        }
    )
    difference_count = sum(
        row["selected_action"] != row["baseline_action"] for row in guard_replay_rows
    )
    gate = profile["historical_prequential_hard_gate"]
    checks = {
        "exact_prequential_market_support": len(prequential_rows) == expected_stream,
        "prediction_before_current_target_access": all(
            row["prediction_frozen_before_current_target_access"] is True
            and row["current_market_target_used_for_prediction"] is False
            for row in prequential_rows
        ),
        "strict_prior_target_chronology": all(
            row["training_max_decision_ts"] < row["baseline_decision_ts"]
            for row in prequential_rows
        ),
        "strict_prior_rank_state": all(
            row["rank_state_max_decision_ts"] < row["baseline_decision_ts"]
            and row["current_market_score_used_for_threshold"] is False
            for row in prequential_rows
        ),
        "same_decision_alternatives_only": all(
            row["baseline_decision_ts"] == row["opposite_decision_ts"]
            for row in prequential_rows
        ),
        "feature_timestamp_causality": all(
            row["baseline_max_input_ts"] <= row["baseline_decision_ts"]
            and row["opposite_max_input_ts"] <= row["opposite_decision_ts"]
            for row in prequential_rows
        ),
        "baseline_identity_reconciled": replay["baseline_identity_reconciled"],
        "minimum_candidate_guard_accepted_support": accepted_count
        >= int(gate["minimum_candidate_guard_accepted_unique_market_count"]),
        "candidate_total_pnl_noninferior_to_v6_7": total_delta
        >= float(gate["candidate_minus_v6_7_total_pnl_minimum_inclusive"]),
        "candidate_largest_winner_removed_noninferior_to_v6_7": lwr_delta
        >= float(
            gate[
                "candidate_minus_v6_7_largest_winner_removed_pnl_minimum_inclusive"
            ]
        ),
        "minimum_policy_difference_market_count_met": difference_count
        >= int(gate["minimum_policy_difference_market_count_for_collection"]),
        "final_model_available": final_artifact["available"],
        "rank_state_finite_and_supported": final_rank_state["available"],
        "issue241_and_issue242_result_artifacts_excluded": True,
    }
    reason_map = {
        "exact_prequential_market_support": "prequential_market_support_invalid",
        "prediction_before_current_target_access": (
            "current_market_target_opened_before_prediction"
        ),
        "strict_prior_target_chronology": "non_prior_target_used_for_prediction",
        "strict_prior_rank_state": "current_or_future_score_used_for_rank_threshold",
        "same_decision_alternatives_only": "alternative_decision_timestamp_used",
        "feature_timestamp_causality": "feature_timestamp_causality_violation",
        "baseline_identity_reconciled": "frozen_v6_7_baseline_identity_mismatch",
        "minimum_candidate_guard_accepted_support": (
            "historical_candidate_guard_accepted_support_insufficient"
        ),
        "candidate_total_pnl_noninferior_to_v6_7": (
            "historical_candidate_pnl_worse_than_v6_7"
        ),
        "candidate_largest_winner_removed_noninferior_to_v6_7": (
            "historical_candidate_lwr_pnl_worse_than_v6_7"
        ),
        "minimum_policy_difference_market_count_met": (
            "historical_policy_difference_support_insufficient"
        ),
        "final_model_available": "final_weighted_model_unavailable",
        "rank_state_finite_and_supported": "final_rank_state_unavailable",
        "issue241_and_issue242_result_artifacts_excluded": (
            "forbidden_prior_result_artifact_accessed"
        ),
    }
    blockers = [reason_map[name] for name, passed in checks.items() if not passed]
    gate_passed = not blockers
    model = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": implementation_commit,
        "fit_created_ts": fit_created_ts,
        "frozen": gate_passed,
        "decision_time_safe": True,
        "final_weighted_model": final_artifact,
        "final_rank_state": final_rank_state,
        "historical_prequential_hard_gate": replay,
        "historical_candidate_guard_accepted_unique_market_count": accepted_count,
        "historical_policy_difference_market_count": difference_count,
        "historical_hard_gate_passed": gate_passed,
        "historical_gate_checks": checks,
        "historical_gate_blocking_reason_codes": blockers,
        "issue241_settlement_outcome_pnl_or_gate_result_opened": False,
        "issue242_result_artifacts_or_pnl_opened": False,
        "issue239_outer_oof_rows_or_metrics_opened": False,
        "issue238_side_action_or_pnl_attribution_used_for_tuning": False,
        "profile_hyperparameter_rank_window_or_threshold_search_performed": False,
        "consumed_stream_is_historical_development_not_promotion_evidence": True,
        "target_free_canary_collection_allowed": gate_passed,
        "target_free_canary_started": False,
        **_v7_0_blocked_safety_fields(),
    }
    model["model_artifact_id"] = canonical_json_sha256(model)
    return {
        "model_artifact": model,
        "prequential_rows": prequential_rows,
        "guard_replay_rows": guard_replay_rows,
        "rank_lineage_rows": rank_lineage_rows,
        "loaded_runtime_target_rows": loaded_target_rows,
        "candidate_selected_rows": candidate_rows,
        "v6_7_baseline_selected_rows": baseline_rows,
        "controller_guard_acceptance_history": controller_guard_acceptance_history,
    }


def run_rolling_origin_score_rank_abstention_v7_9_fit(
    config: RollingOriginScoreRankAbstentionV79Config,
) -> dict[str, Any]:
    """Verify pins and execute the one #243 historical gate."""

    paths = {
        "profile": Path(config.profile_path).resolve(),
        "v7_7_profile": Path(config.v7_7_profile_path).resolve(),
        "v7_0_training_profile": Path(config.v7_0_training_profile_path).resolve(),
        "v6_7_candidate_profile": Path(config.v6_7_candidate_profile_path).resolve(),
        "runtime_policy_profile": Path(config.runtime_policy_profile_path).resolve(),
        "seed_runtime_target_rows": Path(config.seed_runtime_target_rows_path).resolve(),
        "consumed_stream_five_action_rows": Path(
            config.consumed_stream_five_action_rows_path
        ).resolve(),
        "consumed_stream_v6_7_candidate_rows": Path(
            config.consumed_stream_v6_7_candidate_rows_path
        ).resolve(),
        "consumed_stream_v6_7_baseline_rows": Path(
            config.consumed_stream_v6_7_baseline_rows_path
        ).resolve(),
        "consumed_stream_settled_index": Path(
            config.consumed_stream_settled_index_path
        ).resolve(),
        "consumed_stream_target_free_freeze_manifest": Path(
            config.consumed_stream_target_free_freeze_manifest_path
        ).resolve(),
    }
    _verify_pin(paths["profile"], config.expected_profile_sha256, "#243 profile")
    profile = _load_json(paths["profile"])
    validate_rolling_origin_score_rank_abstention_v7_9_profile(profile)
    for key, path in paths.items():
        if key != "profile":
            _verify_pin(path, profile["lineage"][f"{key}_sha256"], f"#243 {key}")
    if xgb.__version__ != profile["lineage"]["xgboost_version"]:
        raise ValueError("#243 xgboost version mismatch")
    v7_7_profile = _load_json(paths["v7_7_profile"])
    v77.validate_rolling_origin_drift_adaptive_v7_7_profile(v7_7_profile)
    if (
        profile["xgboost"] != v7_7_profile["xgboost"]
        or profile["model_contract"]["fixed_edge_buffer"]
        != v7_7_profile["model_contract"]["fixed_edge_buffer"]
    ):
        raise ValueError("#243 v7.7 model or point policy changed")
    training_profile = _load_json(paths["v7_0_training_profile"])
    validate_v7_0_training_profile(training_profile)
    v6_7_profile = _load_json(paths["v6_7_candidate_profile"])
    validate_p_up_semantic_compatibility_v6_7_profile(v6_7_profile)
    runtime_profile = _load_json(paths["runtime_policy_profile"])
    validate_runtime_aligned_sbc_net_return_v6_4_profile(runtime_profile)
    raw_seed_rows = _load_jsonl(paths["seed_runtime_target_rows"])
    seed_rows = materialize_v7_0_sbc_rows(raw_seed_rows, training_profile)
    v78._attach_seed_close_timestamps(seed_rows, raw_seed_rows=raw_seed_rows)
    settled_index = _load_json(paths["consumed_stream_settled_index"])
    freeze_manifest = _load_json(paths["consumed_stream_target_free_freeze_manifest"])
    v77._validate_consumed_lineage(
        settled_index,
        freeze_manifest=freeze_manifest,
        freeze_manifest_path=paths["consumed_stream_target_free_freeze_manifest"],
        paths=paths,
    )
    stream_markets = v78._stream_markets_with_guard_sources(
        five_action_rows=_load_jsonl(paths["consumed_stream_five_action_rows"]),
        candidate_rows=_load_jsonl(paths["consumed_stream_v6_7_candidate_rows"]),
        baseline_rows=_load_jsonl(paths["consumed_stream_v6_7_baseline_rows"]),
        training_profile=training_profile,
    )
    settled_entries = list(settled_index["entries"])

    def target_loader(
        baseline_row: dict[str, Any], opposite_row: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return (
            _runtime_targets_for_decisions(
                [v77._target_decision(baseline_row)],
                settled_entries=settled_entries,
                runtime_profile=runtime_profile,
                run_id=f"{config.run_id}-baseline",
                role="consumed_historical_prequential",
            )[0],
            _runtime_targets_for_decisions(
                [v77._target_decision(opposite_row)],
                settled_entries=settled_entries,
                runtime_profile=runtime_profile,
                run_id=f"{config.run_id}-opposite",
                role="consumed_historical_prequential",
            )[0],
        )

    fit = fit_rolling_origin_score_rank_abstention_v7_9(
        seed_rows=seed_rows,
        stream_markets=stream_markets,
        target_loader=target_loader,
        profile=profile,
        v6_7_profile=v6_7_profile,
        implementation_commit=config.implementation_commit,
        fit_created_ts=config.fit_created_ts,
    )
    model = fit["model_artifact"]
    leakage = _leakage_audit(seed_rows, fit=fit, model=model)
    report = _report(model, leakage=leakage)
    run_dir = Path(config.output_dir).resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    outputs = {
        "model": run_dir / "v7_9_rolling_origin_score_rank_model.json",
        "report": run_dir / "v7_9_historical_prequential_hard_gate_report.json",
        "report_markdown": (
            run_dir / "v7_9_historical_prequential_hard_gate_report.md"
        ),
        "leakage_audit": run_dir / "v7_9_fit_leakage_audit.json",
        "prequential_rows": run_dir / "v7_9_prequential_policy_rows.jsonl",
        "guard_replay_rows": run_dir / "v7_9_historical_guard_replay_rows.jsonl",
        "rank_lineage_rows": (
            run_dir / "v7_9_strictly_prior_prediction_rank_lineage.jsonl"
        ),
        "runtime_target_rows": run_dir / "v7_9_consumed_stream_runtime_targets.jsonl",
        "candidate_selected_rows": run_dir / "v7_9_candidate_selected_rows.jsonl",
        "v6_7_baseline_selected_rows": (
            run_dir / "v7_9_v6_7_baseline_selected_rows.jsonl"
        ),
    }
    _write_json(outputs["model"], model)
    _write_json(outputs["report"], report)
    _write_text(outputs["report_markdown"], _report_markdown(report))
    _write_json(outputs["leakage_audit"], leakage)
    for key in (
        "prequential_rows",
        "guard_replay_rows",
        "rank_lineage_rows",
        "runtime_target_rows",
        "candidate_selected_rows",
        "v6_7_baseline_selected_rows",
    ):
        source_key = (
            "loaded_runtime_target_rows" if key == "runtime_target_rows" else key
        )
        _write_jsonl(outputs[key], fit[source_key])
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        **{name: _descriptor(path) for name, path in paths.items()},
        **{name: _descriptor(path) for name, path in outputs.items()},
        "historical_hard_gate_passed": model["historical_hard_gate_passed"],
        "historical_gate_blocking_reason_codes": model[
            "historical_gate_blocking_reason_codes"
        ],
        "historical_candidate_guard_accepted_unique_market_count": model[
            "historical_candidate_guard_accepted_unique_market_count"
        ],
        "target_free_canary_collection_allowed": model[
            "target_free_canary_collection_allowed"
        ],
        "target_free_canary_started": False,
        "fit_leakage_audit_passed": leakage["fit_leakage_audit_passed"],
        **_v7_0_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v7_9_historical_fit_manifest.json"
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


def score_rolling_origin_score_rank_abstention_v7_9_market(
    market: dict[str, Any],
    *,
    model_artifact: dict[str, Any],
    prior_rank_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one outcome-free market and return the deterministic next rank state."""

    rows = [market.get("baseline_row") or {}, market.get("opposite_row") or {}]
    reasons = []
    if model_artifact.get("schema_version") != MODEL_SCHEMA_VERSION:
        reasons.append("v7_9_model_artifact_schema_invalid")
    if model_artifact.get("historical_hard_gate_passed") is not True:
        reasons.append("v7_9_historical_hard_gate_not_passed")
    if model_artifact.get("target_free_canary_collection_allowed") is not True:
        reasons.append("v7_9_target_free_canary_not_authorized")
    if any(
        FORBIDDEN_INFERENCE_FIELDS.intersection(row)
        or FORBIDDEN_INFERENCE_FIELDS.intersection(
            dict(row.get("decision_time_features") or {})
        )
        for row in rows
    ):
        reasons.append("v7_9_forbidden_outcome_field_in_inference_row")
    if reasons:
        return _no_trade_inference(rows, reasons=reasons)
    if (
        not market.get("market_id")
        or {row.get("action") for row in rows} != set(SBC_ACTIONS)
        or len({row.get("decision_ts") for row in rows}) != 1
        or any(
            tuple((row.get("decision_time_features") or {}).keys()) != FEATURE_NAMES
            or int(row["max_input_ts"]) > int(row["decision_ts"])
            for row in rows
        )
    ):
        return _no_trade_inference(
            rows, reasons=["v7_9_target_free_feature_contract_invalid"]
        )
    state = dict(prior_rank_state or model_artifact["final_rank_state"])
    values = [float(value) for value in state["eligible_prediction_scores"]]
    threshold = finite_sample_score_quantile(
        values[-int(state["eligible_prior_score_window"]) :],
        float(state["rolling_score_quantile"]),
    )
    decision = _score_rank_market(
        market,
        artifact=model_artifact["final_weighted_model"],
        prior_score_threshold=threshold,
    )
    rank_row = _rank_lineage_row(decision, source_role="target_free_inference")
    if rank_row["eligible_prediction_score"]:
        values.append(float(rank_row["point_selected_predicted_return"]))
    next_state = _rank_state(
        values,
        rank_lineage_rows=[],
        window_size=int(state["eligible_prior_score_window"]),
        quantile=float(state["rolling_score_quantile"]),
    )
    result = {
        **decision,
        "trade_selected": decision["selected_action"] != "NO_TRADE",
        "next_rank_state": next_state,
        "alternative_decision_timestamp_used": False,
        "outcome_or_pnl_field_used_at_inference": False,
        "selection_reason_codes": [],
        **_v7_0_blocked_safety_fields(),
    }
    result["decision_id"] = canonical_json_sha256(result)
    return result


def _seed_prediction_score_state(
    seed_rows: list[dict[str, Any]], *, profile: dict[str, Any]
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in seed_rows:
        grouped.setdefault(str(row["market_id"]), []).append(row)
    order = _market_order(seed_rows)
    initial = int(profile["historical_stream"]["seed_initial_training_market_count"])
    training_order = list(order[:initial])
    training_rows = [row for market_id in training_order for row in grouped[market_id]]
    eligible_scores: list[float] = []
    rank_lineage_rows: list[dict[str, Any]] = []
    prior_close_ts = max(int(row["market_close_ts"]) for row in training_rows)
    for seed_index, market_id in enumerate(order[initial:], start=initial):
        rows = sorted(grouped[market_id], key=lambda row: str(row["action"]))
        baseline, opposite = v78._seed_baseline_and_opposite(rows)
        if prior_close_ts >= int(baseline["decision_ts"]):
            raise ValueError("#243 seed target chronology invalid")
        artifact = v77._fit_weighted_model(
            training_rows,
            market_order=training_order,
            profile=profile,
        )
        point = v77._score_stream_market(
            {
                "market_id": market_id,
                "market_close_ts": int(baseline["market_close_ts"]),
                "baseline_row": baseline,
                "opposite_row": opposite,
            },
            artifact=artifact,
            profile=profile,
        )
        point.update(
            {
                "seed_index": seed_index,
                "prediction_frozen_before_current_target_access": True,
                "current_market_target_used_for_prediction": False,
            }
        )
        rank_row = _rank_lineage_row(point, source_role="seed_rolling_origin")
        rank_lineage_rows.append(rank_row)
        if rank_row["eligible_prediction_score"]:
            eligible_scores.append(float(rank_row["point_selected_predicted_return"]))
        training_rows.extend(rows)
        training_order.append(market_id)
        prior_close_ts = int(baseline["market_close_ts"])
    return {
        "training_rows": training_rows,
        "training_market_order": training_order,
        "eligible_prediction_scores": eligible_scores,
        "rank_lineage_rows": rank_lineage_rows,
    }


def _point_selected_predicted_return(point: dict[str, Any]) -> float | None:
    decision = str(point["selected_policy_decision"])
    if decision == "VETO_TO_NO_TRADE":
        return None
    if decision == "SWITCH_SAME_DECISION_SBC":
        return float(point["predicted_opposite_return"])
    return float(point["predicted_baseline_return"])


def _score_rank_market(
    market: dict[str, Any],
    *,
    artifact: dict[str, Any],
    prior_score_threshold: float,
) -> dict[str, Any]:
    point = v77._score_stream_market(market, artifact=artifact, profile={
        "model_contract": {"fixed_edge_buffer": v77.FROZEN_EDGE_BUFFER}
    })
    point_score = _point_selected_predicted_return(point)
    rank_passed = (
        point_score is not None
        and point_score > 0.0
        and point_score >= prior_score_threshold
    )
    if rank_passed:
        selected_decision = point["selected_policy_decision"]
        selected_action = point["selected_action"]
        selected_side = point["selected_side"]
    else:
        selected_decision = "VETO_TO_NO_TRADE"
        selected_action = "NO_TRADE"
        selected_side = "NONE"
    return {
        **point,
        "point_selected_policy_decision": point["selected_policy_decision"],
        "point_selected_action": point["selected_action"],
        "point_selected_side": point["selected_side"],
        "point_selected_predicted_return": point_score,
        "prior_rolling_score_q60": prior_score_threshold,
        "point_selected_score_strictly_positive": (
            point_score is not None and point_score > 0.0
        ),
        "point_selected_score_at_or_above_prior_q60": (
            point_score is not None and point_score >= prior_score_threshold
        ),
        "rank_abstention_passed": rank_passed,
        "selected_policy_decision": selected_decision,
        "selected_action": selected_action,
        "selected_side": selected_side,
        "target_used_as_decision_time_input": False,
        "source_score_mutated": False,
    }


def _rank_lineage_row(
    point: dict[str, Any], *, source_role: str
) -> dict[str, Any]:
    point_score = _point_selected_predicted_return(point)
    row = {
        "market_id": str(point["market_id"]),
        "decision_ts": int(point["baseline_decision_ts"]),
        "source_role": source_role,
        "point_selected_policy_decision": point["selected_policy_decision"],
        "point_selected_action": point["selected_action"],
        "point_selected_side": point["selected_side"],
        "point_selected_predicted_return": point_score,
        "eligible_prediction_score": point_score is not None,
        "score_added_only_after_current_decision_freeze": True,
        "score_uses_target_outcome_or_pnl": False,
        "current_market_score_used_for_own_threshold": False,
    }
    row["rank_lineage_row_id"] = canonical_json_sha256(row)
    return row


def _rank_state(
    scores: list[float],
    *,
    rank_lineage_rows: list[dict[str, Any]],
    window_size: int,
    quantile: float,
) -> dict[str, Any]:
    window = [float(value) for value in scores[-window_size:]]
    threshold = finite_sample_score_quantile(window, quantile)
    available = (
        len(window) == window_size
        and all(math.isfinite(value) for value in window)
        and math.isfinite(threshold)
    )
    state = {
        "schema_version": "bigan-v8-rolling-origin-score-rank-state-v7-9-v1",
        "available": available,
        "frozen": available,
        "decision_time_safe": True,
        "eligible_prior_score_window": window_size,
        "rolling_score_quantile": quantile,
        "finite_sample_rank": "ceil(n_times_quantile)_capped_at_n",
        "eligible_prediction_score_count": len(scores),
        "eligible_prediction_scores": window,
        "eligible_prediction_scores_hash": canonical_json_sha256(window),
        "rolling_score_q60": threshold,
        "rank_lineage_hash": canonical_json_sha256(rank_lineage_rows),
        "rank_state_uses_target_outcome_or_pnl": False,
        "current_market_score_used_for_own_threshold": False,
        "quantile_window_threshold_or_profile_search_performed": False,
    }
    state["rank_state_id"] = canonical_json_sha256(state)
    return state


def _leakage_audit(
    seed_rows: list[dict[str, Any]],
    *,
    fit: dict[str, Any],
    model: dict[str, Any],
) -> dict[str, Any]:
    prequential = fit["prequential_rows"]
    rank_rows = fit["rank_lineage_rows"]
    checks = {
        "seed_market_count_134": len({row["market_id"] for row in seed_rows}) == 134,
        "prequential_market_count_120": len(prequential) == 120,
        "seed_feature_causality": all(
            int(row["max_input_ts"]) <= int(row["decision_ts"]) for row in seed_rows
        ),
        "stream_feature_causality": all(
            int(row["baseline_max_input_ts"]) <= int(row["baseline_decision_ts"])
            and int(row["opposite_max_input_ts"]) <= int(row["opposite_decision_ts"])
            for row in prequential
        ),
        "prediction_frozen_before_target": all(
            row["prediction_frozen_before_current_target_access"] is True
            for row in prequential
        ),
        "rank_state_strictly_prior": all(
            row["rank_state_max_decision_ts"] < row["baseline_decision_ts"]
            and row["current_market_score_used_for_threshold"] is False
            for row in prequential
        ),
        "rank_scores_outcome_free": all(
            row["score_uses_target_outcome_or_pnl"] is False for row in rank_rows
        ),
        "issue241_not_opened": model[
            "issue241_settlement_outcome_pnl_or_gate_result_opened"
        ]
        is False,
        "issue242_artifacts_not_opened": model[
            "issue242_result_artifacts_or_pnl_opened"
        ]
        is False,
        "no_result_selected_search": model[
            "profile_hyperparameter_rank_window_or_threshold_search_performed"
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
        "current_market_target_used_as_decision_time_input": False,
        "current_market_score_used_for_own_rank_threshold": False,
        "issue241_or_issue242_result_artifacts_accepted": False,
        "consumed_stream_replay_is_promotion_evidence": False,
        **_v7_0_blocked_safety_fields(),
    }
    audit["leakage_audit_id"] = canonical_json_sha256(audit)
    return audit


def _report(model: dict[str, Any], *, leakage: dict[str, Any]) -> dict[str, Any]:
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "historical_hard_gate_passed": model["historical_hard_gate_passed"],
        "historical_gate_blocking_reason_codes": model[
            "historical_gate_blocking_reason_codes"
        ],
        "historical_prequential_hard_gate": model["historical_prequential_hard_gate"],
        "historical_candidate_guard_accepted_unique_market_count": model[
            "historical_candidate_guard_accepted_unique_market_count"
        ],
        "historical_policy_difference_market_count": model[
            "historical_policy_difference_market_count"
        ],
        "final_rank_state": model["final_rank_state"],
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
    replay = report["historical_prequential_hard_gate"]
    state = report["final_rank_state"]
    return "\n".join(
        [
            "# v7.9 Rolling-Origin Score-Rank Historical Gate",
            "",
            "- historical hard gate passed: "
            f"`{str(report['historical_hard_gate_passed']).lower()}`",
            f"- blockers: `{report['historical_gate_blocking_reason_codes']}`",
            "- candidate guard-accepted markets: "
            f"`{report['historical_candidate_guard_accepted_unique_market_count']}`",
            "- candidate PnL: "
            f"`{replay['candidate']['total_after_cost_net_pnl_at_frozen_size']}`",
            "- v6.7 PnL: "
            f"`{replay['v6_7_baseline']['total_after_cost_net_pnl_at_frozen_size']}`",
            "- candidate-minus-v6.7 PnL: "
            f"`{replay['candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size']}`",
            "- candidate-minus-v6.7 largest-winner-removed PnL: "
            f"`{replay['candidate_minus_v6_7_largest_winner_removed_after_cost_net_pnl_at_frozen_size']}`",
            "- policy difference markets: "
            f"`{report['historical_policy_difference_market_count']}`",
            f"- final rolling q60: `{state['rolling_score_q60']}`",
            "- rank score history uses target/outcome/PnL: `false`",
            "- target-free collection allowed: "
            f"`{str(report['target_free_canary_collection_allowed']).lower()}`",
            "- #241/#242 result artifacts used: `false`",
            "- historical replay is promotion evidence: `false`",
            "- paper/live/write/wallet/capital unlock: `false`",
            "",
        ]
    )
