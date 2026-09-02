"""Rolling-origin conformal positive-EV SBC policy for issue #242."""

from __future__ import annotations

import math
import shutil
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xgboost as xgb

from bigan.v8.polymarket.contracts import canonical_json_sha256
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
    _microstructure_blocking_reasons,
    validate_p_up_semantic_compatibility_v6_7_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7_post_freeze import (
    _runtime_targets_for_decisions,
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
    validate_runtime_aligned_sbc_net_return_v6_4_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_v6_7_relative_safe_policy_v7_2 import (
    FORBIDDEN_INFERENCE_FIELDS,
    _historical_replay,
    _market_order,
    _no_trade_inference,
    _validate_canonical_rows,
)

CANDIDATE_NAME = "rolling_origin_conformal_positive_ev_v7_8"
PROFILE_SCHEMA_VERSION = "bigan-v8-rolling-origin-conformal-positive-ev-v7-8-profile-v1"
MODEL_SCHEMA_VERSION = "bigan-v8-rolling-origin-conformal-positive-ev-v7-8-model-v1"
REPORT_SCHEMA_VERSION = "bigan-v8-rolling-origin-conformal-positive-ev-v7-8-report-v1"
LEAKAGE_SCHEMA_VERSION = "bigan-v8-rolling-origin-conformal-positive-ev-v7-8-leakage-v1"
MANIFEST_SCHEMA_VERSION = "bigan-v8-rolling-origin-conformal-positive-ev-v7-8-manifest-v1"
SBC_ACTIONS = v77.SBC_ACTIONS
FROZEN_LINEAGE = {
    **v77.FROZEN_LINEAGE,
    "v7_7_profile_sha256": (
        "133a78cf5af33cef1bfb4e6d6f5de1771e153c2450d9226282648f4232bf006a"
    ),
}


@dataclass(frozen=True, slots=True)
class RollingOriginConformalPositiveEVV78Config:
    """Pinned inputs for the single preregistered #242 historical replay."""

    run_id: str
    output_dir: Path | str
    profile_path: Path | str
    expected_profile_sha256: str
    v7_7_profile_path: Path | str
    v7_0_training_profile_path: Path | str
    v6_7_candidate_profile_path: Path | str
    runtime_policy_profile_path: Path | str
    seed_runtime_target_rows_path: Path | str
    consumed_stream_five_action_rows_path: Path | str
    consumed_stream_v6_7_candidate_rows_path: Path | str
    consumed_stream_v6_7_baseline_rows_path: Path | str
    consumed_stream_settled_index_path: Path | str
    consumed_stream_target_free_freeze_manifest_path: Path | str
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
            "v7_7_profile_path",
            "v7_0_training_profile_path",
            "v6_7_candidate_profile_path",
            "runtime_policy_profile_path",
            "seed_runtime_target_rows_path",
            "consumed_stream_five_action_rows_path",
            "consumed_stream_v6_7_candidate_rows_path",
            "consumed_stream_v6_7_baseline_rows_path",
            "consumed_stream_settled_index_path",
            "consumed_stream_target_free_freeze_manifest_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


def validate_rolling_origin_conformal_positive_ev_v7_8_profile(
    profile: dict[str, Any],
) -> None:
    """Reject any drift from the preregistered #242 contract."""

    checks = {
        "identity": profile.get("schema_version") == PROFILE_SCHEMA_VERSION
        and profile.get("issue_number") == 242
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
            "profile_or_hyperparameter_search_allowed": False,
            "side_specific_model_or_quota_allowed": False,
            "allowed_policy_decisions": [
                "KEEP_V6_7",
                "SWITCH_SAME_DECISION_SBC",
                "VETO_TO_NO_TRADE",
            ],
            "v6_7_no_trade_activation_allowed": False,
            "same_decision_timestamp_required": True,
            "maximum_bets_per_market": 1,
            "hts_disabled_fail_closed": True,
            "full_execution_guard_applied_after_selection": True,
        },
        "conformal": profile.get("conformal_contract")
        == {
            "action_residual": "predicted_return_minus_realized_after_cost_return",
            "paired_delta_residual": (
                "predicted_opposite_minus_baseline_minus_realized_opposite_minus_baseline"
            ),
            "residual_pooling": "pooled_up_down_no_side_weight_or_quota",
            "minimum_prior_calibration_market_count": 60,
            "one_sided_quantile": 0.9,
            "finite_sample_rank": "ceil((n_plus_1)_times_quantile)_capped_at_n",
            "quantile_profile_search_allowed": False,
            "current_market_residual_available_before_decision": False,
            "switch_requires_opposite_action_lcb_positive": True,
            "switch_requires_delta_lcb_positive": True,
            "keep_requires_baseline_action_lcb_positive": True,
            "otherwise_veto_to_no_trade": True,
        },
        "xgboost": profile.get("xgboost") == v77.FROZEN_XGB,
        "gate": profile.get("historical_prequential_hard_gate")
        == {
            "exact_evaluation_market_count": 120,
            "fixed_position_size": 0.2,
            "no_bet_market_pnl": 0.0,
            "minimum_candidate_guard_accepted_unique_market_count": 40,
            "candidate_total_after_cost_pnl_minimum_exclusive": 0.0,
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
        raise ValueError("#242 v7.8 profile invalid: " + ", ".join(blockers))


def finite_sample_upper_quantile(values: list[float], quantile: float) -> float:
    """Return the fixed finite-sample one-sided upper quantile."""

    if not values or not 0.0 < quantile < 1.0:
        raise ValueError("#242 conformal quantile input invalid")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("#242 conformal residual is non-finite")
    rank = min(len(ordered), math.ceil((len(ordered) + 1) * quantile))
    return ordered[rank - 1]


def fit_rolling_origin_conformal_positive_ev_v7_8(
    *,
    seed_rows: list[dict[str, Any]],
    stream_markets: list[dict[str, Any]],
    target_loader: Callable[
        [dict[str, Any], dict[str, Any]],
        tuple[dict[str, Any], dict[str, Any]],
    ],
    profile: dict[str, Any],
    v6_7_profile: dict[str, Any],
    implementation_commit: str,
    fit_created_ts: int,
) -> dict[str, Any]:
    """Fit and evaluate v7.8 with every residual strictly prior to its decision."""

    validate_rolling_origin_conformal_positive_ev_v7_8_profile(profile)
    _validate_canonical_rows(seed_rows)
    seed_order = _market_order(seed_rows)
    expected_seed = int(profile["historical_stream"]["seed_market_count"])
    expected_stream = int(profile["historical_stream"]["prequential_market_count"])
    if len(seed_order) != expected_seed or len(stream_markets) != expected_stream:
        raise ValueError("#242 historical market support invalid")
    stream_markets = sorted(
        stream_markets,
        key=lambda row: (int(row["decision_ts"]), str(row["market_id"])),
    )
    if len({str(row["market_id"]) for row in stream_markets}) != expected_stream:
        raise ValueError("#242 duplicate prequential market")
    seed_max_close = max(int(row["market_close_ts"]) for row in seed_rows)
    if seed_max_close >= int(stream_markets[0]["decision_ts"]):
        raise ValueError("#242 seed/stream chronology invalid")

    seed_state = _seed_prequential_residual_state(seed_rows, profile=profile)
    training_rows = list(seed_state["training_rows"])
    training_order = list(seed_state["training_market_order"])
    action_residuals = list(seed_state["action_residuals"])
    delta_residuals = list(seed_state["paired_delta_residuals"])
    calibration_market_ids = list(seed_state["calibration_market_ids"])
    residual_lineage_rows = list(seed_state["residual_lineage_rows"])
    prequential_rows: list[dict[str, Any]] = []
    guard_replay_rows: list[dict[str, Any]] = []
    loaded_target_rows: list[dict[str, Any]] = []
    prior_close_ts = seed_max_close
    conformal = profile["conformal_contract"]
    minimum_calibration = int(conformal["minimum_prior_calibration_market_count"])
    quantile = float(conformal["one_sided_quantile"])

    for stream_index, market in enumerate(stream_markets):
        if prior_close_ts >= int(market["decision_ts"]):
            raise ValueError("#242 prior target unavailable before current decision")
        calibration_count = len(calibration_market_ids)
        if calibration_count < minimum_calibration:
            raise ValueError("#242 prior conformal market support insufficient")
        action_q = finite_sample_upper_quantile(action_residuals, quantile)
        delta_q = finite_sample_upper_quantile(delta_residuals, quantile)
        artifact = v77._fit_weighted_model(
            training_rows,
            market_order=training_order,
            profile=profile,
        )
        prediction = _score_conformal_market(
            market,
            artifact=artifact,
            action_residual_q=action_q,
            paired_delta_residual_q=delta_q,
        )
        prediction.update(
            {
                "stream_index": stream_index,
                "prediction_frozen_before_current_target_access": True,
                "training_market_count": len(training_order),
                "training_max_decision_ts": max(
                    int(row["decision_ts"]) for row in training_rows
                ),
                "calibration_market_count_before_decision": calibration_count,
                "calibration_max_target_available_ts": prior_close_ts,
                "current_market_target_used_for_prediction": False,
                "current_market_residual_used_for_prediction": False,
            }
        )
        guard = _execution_guard_result(
            prediction,
            market=market,
            v6_7_profile=v6_7_profile,
        )
        prediction.update(guard)
        baseline_target, opposite_target = target_loader(
            market["baseline_row"], market["opposite_row"]
        )
        _validate_loaded_targets(
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
        guard_replay_rows.append(_guard_replay_row(prediction))
        loaded_target_rows.extend((baseline_target, opposite_target))
        current_residuals = _residual_rows(
            prediction,
            target_available_ts=int(market["market_close_ts"]),
            source_role="consumed_historical_prequential",
        )
        residual_lineage_rows.extend(current_residuals)
        action_residuals.extend(
            float(row["residual"])
            for row in current_residuals
            if row["residual_type"] == "action_overprediction"
        )
        delta_residuals.extend(
            float(row["residual"])
            for row in current_residuals
            if row["residual_type"] == "paired_delta_overprediction"
        )
        calibration_market_ids.append(str(market["market_id"]))
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
    final_conformal_state = _conformal_state(
        action_residuals,
        delta_residuals,
        calibration_market_ids=calibration_market_ids,
        quantile=quantile,
        residual_lineage_rows=residual_lineage_rows,
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
    replay["gate_name"] = "same_stream_conformal_positive_ev_hard_gate"
    replay["comparison_operator"] = "candidate_positive_and_noninferior_to_v6_7"
    replay["full_execution_guard_applied_to_candidate_and_baseline"] = True
    candidate_rows = replay.pop("candidate_selected_rows")
    baseline_rows = replay.pop("v6_7_baseline_selected_rows")
    candidate_total = float(replay["candidate"]["total_after_cost_net_pnl_at_frozen_size"])
    total_delta = float(
        replay["candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size"]
    )
    lwr_delta = float(
        replay[
            "candidate_minus_v6_7_largest_winner_removed_after_cost_net_pnl_at_frozen_size"
        ]
    )
    accepted_market_count = len(
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
            and row["current_market_residual_used_for_prediction"] is False
            for row in prequential_rows
        ),
        "strict_prior_target_and_residual_chronology": all(
            row["training_max_decision_ts"] < row["baseline_decision_ts"]
            and row["calibration_max_target_available_ts"]
            < row["baseline_decision_ts"]
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
        "minimum_candidate_guard_accepted_support": accepted_market_count
        >= int(gate["minimum_candidate_guard_accepted_unique_market_count"]),
        "candidate_total_after_cost_pnl_positive": candidate_total
        > float(gate["candidate_total_after_cost_pnl_minimum_exclusive"]),
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
        "conformal_state_finite_and_supported": final_conformal_state["available"],
        "issue241_inputs_excluded": True,
    }
    reason_map = {
        "exact_prequential_market_support": "prequential_market_support_invalid",
        "prediction_before_current_target_access": (
            "current_market_target_or_residual_opened_before_prediction"
        ),
        "strict_prior_target_and_residual_chronology": (
            "non_prior_target_or_residual_used_for_prediction"
        ),
        "same_decision_alternatives_only": "alternative_decision_timestamp_used",
        "feature_timestamp_causality": "feature_timestamp_causality_violation",
        "baseline_identity_reconciled": "frozen_v6_7_baseline_identity_mismatch",
        "minimum_candidate_guard_accepted_support": (
            "historical_candidate_guard_accepted_support_insufficient"
        ),
        "candidate_total_after_cost_pnl_positive": (
            "historical_candidate_total_after_cost_pnl_not_positive"
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
        "conformal_state_finite_and_supported": "final_conformal_state_unavailable",
        "issue241_inputs_excluded": "issue241_result_accessed",
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
        "final_conformal_state": final_conformal_state,
        "historical_prequential_hard_gate": replay,
        "historical_candidate_guard_accepted_unique_market_count": accepted_market_count,
        "historical_policy_difference_market_count": difference_count,
        "historical_hard_gate_passed": gate_passed,
        "historical_gate_checks": checks,
        "historical_gate_blocking_reason_codes": blockers,
        "issue241_settlement_outcome_pnl_or_gate_result_opened": False,
        "issue239_outer_oof_rows_or_metrics_opened": False,
        "issue238_side_action_or_pnl_attribution_used_for_tuning": False,
        "profile_hyperparameter_or_quantile_search_performed": False,
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
        "residual_lineage_rows": residual_lineage_rows,
        "loaded_runtime_target_rows": loaded_target_rows,
        "candidate_selected_rows": candidate_rows,
        "v6_7_baseline_selected_rows": baseline_rows,
    }


def run_rolling_origin_conformal_positive_ev_v7_8_fit(
    config: RollingOriginConformalPositiveEVV78Config,
) -> dict[str, Any]:
    """Verify all pins, execute once, and write immutable #242 evidence."""

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
    _verify_pin(paths["profile"], config.expected_profile_sha256, "#242 profile")
    profile = _load_json(paths["profile"])
    validate_rolling_origin_conformal_positive_ev_v7_8_profile(profile)
    for key, path in paths.items():
        if key == "profile":
            continue
        _verify_pin(path, profile["lineage"][f"{key}_sha256"], f"#242 {key}")
    if xgb.__version__ != profile["lineage"]["xgboost_version"]:
        raise ValueError("#242 xgboost version mismatch")
    v7_7_profile = _load_json(paths["v7_7_profile"])
    v77.validate_rolling_origin_drift_adaptive_v7_7_profile(v7_7_profile)
    if profile["xgboost"] != v7_7_profile["xgboost"]:
        raise ValueError("#242 v7.7 model parameters changed")
    training_profile = _load_json(paths["v7_0_training_profile"])
    validate_v7_0_training_profile(training_profile)
    v6_7_profile = _load_json(paths["v6_7_candidate_profile"])
    validate_p_up_semantic_compatibility_v6_7_profile(v6_7_profile)
    runtime_profile = _load_json(paths["runtime_policy_profile"])
    validate_runtime_aligned_sbc_net_return_v6_4_profile(runtime_profile)
    raw_seed_rows = _load_jsonl(paths["seed_runtime_target_rows"])
    seed_rows = materialize_v7_0_sbc_rows(raw_seed_rows, training_profile)
    _attach_seed_close_timestamps(seed_rows, raw_seed_rows=raw_seed_rows)
    settled_index = _load_json(paths["consumed_stream_settled_index"])
    freeze_manifest = _load_json(paths["consumed_stream_target_free_freeze_manifest"])
    v77._validate_consumed_lineage(
        settled_index,
        freeze_manifest=freeze_manifest,
        freeze_manifest_path=paths["consumed_stream_target_free_freeze_manifest"],
        paths=paths,
    )
    five_action_rows = _load_jsonl(paths["consumed_stream_five_action_rows"])
    stream_markets = _stream_markets_with_guard_sources(
        five_action_rows=five_action_rows,
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

    fit = fit_rolling_origin_conformal_positive_ev_v7_8(
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
        "model": run_dir / "v7_8_rolling_origin_conformal_model.json",
        "report": run_dir / "v7_8_historical_prequential_hard_gate_report.json",
        "report_markdown": (
            run_dir / "v7_8_historical_prequential_hard_gate_report.md"
        ),
        "leakage_audit": run_dir / "v7_8_fit_leakage_audit.json",
        "prequential_rows": run_dir / "v7_8_prequential_policy_rows.jsonl",
        "guard_replay_rows": run_dir / "v7_8_historical_guard_replay_rows.jsonl",
        "residual_lineage_rows": (
            run_dir / "v7_8_strictly_prior_conformal_residual_lineage.jsonl"
        ),
        "runtime_target_rows": run_dir / "v7_8_consumed_stream_runtime_targets.jsonl",
        "candidate_selected_rows": run_dir / "v7_8_candidate_selected_rows.jsonl",
        "v6_7_baseline_selected_rows": (
            run_dir / "v7_8_v6_7_baseline_selected_rows.jsonl"
        ),
    }
    _write_json(outputs["model"], model)
    _write_json(outputs["report"], report)
    _write_text(outputs["report_markdown"], _report_markdown(report))
    _write_json(outputs["leakage_audit"], leakage)
    for key in (
        "prequential_rows",
        "guard_replay_rows",
        "residual_lineage_rows",
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
    manifest_path = run_dir / "v7_8_historical_fit_manifest.json"
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


def score_rolling_origin_conformal_positive_ev_v7_8_market(
    market: dict[str, Any], *, model_artifact: dict[str, Any]
) -> dict[str, Any]:
    """Apply the frozen outcome-free v7.8 selector before the unchanged guard."""

    rows = [market.get("baseline_row") or {}, market.get("opposite_row") or {}]
    reasons = []
    if model_artifact.get("schema_version") != MODEL_SCHEMA_VERSION:
        reasons.append("v7_8_model_artifact_schema_invalid")
    if model_artifact.get("historical_hard_gate_passed") is not True:
        reasons.append("v7_8_historical_hard_gate_not_passed")
    if model_artifact.get("target_free_canary_collection_allowed") is not True:
        reasons.append("v7_8_target_free_canary_not_authorized")
    if any(
        FORBIDDEN_INFERENCE_FIELDS.intersection(row)
        or FORBIDDEN_INFERENCE_FIELDS.intersection(
            dict(row.get("decision_time_features") or {})
        )
        for row in rows
    ):
        reasons.append("v7_8_forbidden_outcome_field_in_inference_row")
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
            rows, reasons=["v7_8_target_free_feature_contract_invalid"]
        )
    state = model_artifact["final_conformal_state"]
    decision = _score_conformal_market(
        market,
        artifact=model_artifact["final_weighted_model"],
        action_residual_q=float(state["action_overprediction_q90"]),
        paired_delta_residual_q=float(
            state["paired_delta_overprediction_q90"]
        ),
    )
    result = {
        **decision,
        "trade_selected": decision["selected_action"] != "NO_TRADE",
        "alternative_decision_timestamp_used": False,
        "outcome_or_pnl_field_used_at_inference": False,
        "selection_reason_codes": [],
        **_v7_0_blocked_safety_fields(),
    }
    result["decision_id"] = canonical_json_sha256(result)
    return result


def _seed_prequential_residual_state(
    seed_rows: list[dict[str, Any]], *, profile: dict[str, Any]
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[str(row["market_id"])].append(row)
    order = _market_order(seed_rows)
    initial = int(profile["historical_stream"]["seed_initial_training_market_count"])
    training_order = list(order[:initial])
    training_rows = [row for market_id in training_order for row in grouped[market_id]]
    action_residuals: list[float] = []
    delta_residuals: list[float] = []
    calibration_market_ids: list[str] = []
    residual_lineage_rows: list[dict[str, Any]] = []
    prior_close_ts = max(
        int(row["market_close_ts"]) for row in training_rows
    )
    for seed_index, market_id in enumerate(order[initial:], start=initial):
        rows = sorted(grouped[market_id], key=lambda row: str(row["action"]))
        if {str(row["action"]) for row in rows} != set(SBC_ACTIONS):
            raise ValueError("#242 seed same-market SBC pair invalid")
        decision_ts = min(int(row["decision_ts"]) for row in rows)
        if prior_close_ts >= decision_ts:
            raise ValueError("#242 seed residual target chronology invalid")
        artifact = v77._fit_weighted_model(
            training_rows,
            market_order=training_order,
            profile=profile,
        )
        baseline, opposite = _seed_baseline_and_opposite(rows)
        prediction = _score_raw_pair(
            market_id=market_id,
            market_close_ts=int(baseline["market_close_ts"]),
            baseline=baseline,
            opposite=opposite,
            artifact=artifact,
        )
        prediction.update(
            {
                "seed_index": seed_index,
                "prediction_frozen_before_current_target_access": True,
                "training_market_count": len(training_order),
                "training_max_decision_ts": max(
                    int(row["decision_ts"]) for row in training_rows
                ),
                "current_market_target_used_for_prediction": False,
            }
        )
        prediction.update(
            {
                "baseline_target_after_cost_net_pnl_per_contract": float(
                    baseline["target_after_cost_net_pnl_per_contract"]
                ),
                "opposite_target_after_cost_net_pnl_per_contract": float(
                    opposite["target_after_cost_net_pnl_per_contract"]
                ),
            }
        )
        current = _residual_rows(
            prediction,
            target_available_ts=int(baseline["market_close_ts"]),
            source_role="seed_rolling_origin_calibration",
        )
        residual_lineage_rows.extend(current)
        action_residuals.extend(
            float(row["residual"])
            for row in current
            if row["residual_type"] == "action_overprediction"
        )
        delta_residuals.extend(
            float(row["residual"])
            for row in current
            if row["residual_type"] == "paired_delta_overprediction"
        )
        calibration_market_ids.append(market_id)
        training_rows.extend(rows)
        training_order.append(market_id)
        prior_close_ts = int(baseline["market_close_ts"])
    return {
        "training_rows": training_rows,
        "training_market_order": training_order,
        "action_residuals": action_residuals,
        "paired_delta_residuals": delta_residuals,
        "calibration_market_ids": calibration_market_ids,
        "residual_lineage_rows": residual_lineage_rows,
    }


def _seed_baseline_and_opposite(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = sorted(
        rows,
        key=lambda row: (
            -float(row["decision_time_features"]["action_score"]),
            str(row["action"]),
        ),
    )[0]
    opposite = next(row for row in rows if row["action"] != baseline["action"])
    return baseline, opposite


def _score_raw_pair(
    *,
    market_id: str,
    market_close_ts: int,
    baseline: dict[str, Any],
    opposite: dict[str, Any],
    artifact: dict[str, Any],
) -> dict[str, Any]:
    return {
        "market_id": market_id,
        "market_close_ts": market_close_ts,
        "baseline_action": str(baseline["action"]),
        "baseline_side": str(baseline["side"]),
        "baseline_decision_ts": int(baseline["decision_ts"]),
        "baseline_max_input_ts": int(baseline["max_input_ts"]),
        "opposite_action": str(opposite["action"]),
        "opposite_side": str(opposite["side"]),
        "opposite_decision_ts": int(opposite["decision_ts"]),
        "opposite_max_input_ts": int(opposite["max_input_ts"]),
        "predicted_baseline_return": v77._predict(baseline, artifact),
        "predicted_opposite_return": v77._predict(opposite, artifact),
        "target_used_as_decision_time_input": False,
        "source_score_mutated": False,
    }


def _score_conformal_market(
    market: dict[str, Any],
    *,
    artifact: dict[str, Any],
    action_residual_q: float,
    paired_delta_residual_q: float,
) -> dict[str, Any]:
    raw = _score_raw_pair(
        market_id=str(market["market_id"]),
        market_close_ts=int(market["market_close_ts"]),
        baseline=market["baseline_row"],
        opposite=market["opposite_row"],
        artifact=artifact,
    )
    baseline_lcb = raw["predicted_baseline_return"] - action_residual_q
    opposite_lcb = raw["predicted_opposite_return"] - action_residual_q
    predicted_delta = (
        raw["predicted_opposite_return"] - raw["predicted_baseline_return"]
    )
    switch_delta_lcb = predicted_delta - paired_delta_residual_q
    if opposite_lcb > 0.0 and switch_delta_lcb > 0.0:
        decision = "SWITCH_SAME_DECISION_SBC"
        selected_action = raw["opposite_action"]
        selected_side = raw["opposite_side"]
    elif baseline_lcb > 0.0:
        decision = "KEEP_V6_7"
        selected_action = raw["baseline_action"]
        selected_side = raw["baseline_side"]
    else:
        decision = "VETO_TO_NO_TRADE"
        selected_action = "NO_TRADE"
        selected_side = "NONE"
    return {
        **raw,
        "action_overprediction_q90": action_residual_q,
        "paired_delta_overprediction_q90": paired_delta_residual_q,
        "baseline_action_lcb": baseline_lcb,
        "opposite_action_lcb": opposite_lcb,
        "predicted_opposite_minus_baseline": predicted_delta,
        "switch_delta_lcb": switch_delta_lcb,
        "selected_policy_decision": decision,
        "selected_action": selected_action,
        "selected_side": selected_side,
    }


def _residual_rows(
    prediction: dict[str, Any],
    *,
    target_available_ts: int,
    source_role: str,
) -> list[dict[str, Any]]:
    predicted_baseline = float(prediction["predicted_baseline_return"])
    predicted_opposite = float(prediction["predicted_opposite_return"])
    target_baseline = float(
        prediction["baseline_target_after_cost_net_pnl_per_contract"]
    )
    target_opposite = float(
        prediction["opposite_target_after_cost_net_pnl_per_contract"]
    )
    rows = [
        {
            "residual_type": "action_overprediction",
            "action": prediction["baseline_action"],
            "side": prediction["baseline_side"],
            "prediction": predicted_baseline,
            "target": target_baseline,
            "residual": predicted_baseline - target_baseline,
        },
        {
            "residual_type": "action_overprediction",
            "action": prediction["opposite_action"],
            "side": prediction["opposite_side"],
            "prediction": predicted_opposite,
            "target": target_opposite,
            "residual": predicted_opposite - target_opposite,
        },
        {
            "residual_type": "paired_delta_overprediction",
            "action": f"{prediction['opposite_action']}-minus-{prediction['baseline_action']}",
            "side": f"{prediction['opposite_side']}-minus-{prediction['baseline_side']}",
            "prediction": predicted_opposite - predicted_baseline,
            "target": target_opposite - target_baseline,
            "residual": (
                (predicted_opposite - predicted_baseline)
                - (target_opposite - target_baseline)
            ),
        },
    ]
    output = []
    for row in rows:
        item = {
            "market_id": prediction["market_id"],
            "decision_ts": prediction["baseline_decision_ts"],
            "target_available_ts": target_available_ts,
            "source_role": source_role,
            **row,
            "prediction_frozen_before_target_access": True,
            "target_used_as_decision_time_input": False,
            "residual_available_to_same_market_decision": False,
        }
        if (
            not math.isfinite(float(item["residual"]))
            or int(item["target_available_ts"]) <= int(item["decision_ts"])
        ):
            raise ValueError("#242 residual lineage invalid")
        item["residual_row_id"] = canonical_json_sha256(item)
        output.append(item)
    return output


def _conformal_state(
    action_residuals: list[float],
    delta_residuals: list[float],
    *,
    calibration_market_ids: list[str],
    quantile: float,
    residual_lineage_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    action_q = finite_sample_upper_quantile(action_residuals, quantile)
    delta_q = finite_sample_upper_quantile(delta_residuals, quantile)
    available = (
        len(set(calibration_market_ids)) >= 60
        and math.isfinite(action_q)
        and math.isfinite(delta_q)
    )
    state = {
        "schema_version": (
            "bigan-v8-rolling-origin-conformal-positive-ev-v7-8-state-v1"
        ),
        "available": available,
        "frozen": available,
        "decision_time_safe": True,
        "one_sided_quantile": quantile,
        "finite_sample_rank": "ceil((n_plus_1)_times_quantile)_capped_at_n",
        "calibration_market_count": len(set(calibration_market_ids)),
        "calibration_market_ids_hash": canonical_json_sha256(calibration_market_ids),
        "action_residual_count": len(action_residuals),
        "paired_delta_residual_count": len(delta_residuals),
        "action_overprediction_q90": action_q,
        "paired_delta_overprediction_q90": delta_q,
        "residual_lineage_hash": canonical_json_sha256(residual_lineage_rows),
        "current_market_residual_used_for_prediction": False,
        "side_specific_residual_pool_or_quota_used": False,
        "quantile_profile_search_performed": False,
    }
    state["conformal_state_id"] = canonical_json_sha256(state)
    return state


def _execution_guard_result(
    prediction: dict[str, Any],
    *,
    market: dict[str, Any],
    v6_7_profile: dict[str, Any],
) -> dict[str, Any]:
    guard = v6_7_profile["hard_execution_safety"]
    selected_action = str(prediction["selected_action"])
    if selected_action == "NO_TRADE":
        candidate_reasons = ["policy_selected_no_trade"]
        candidate_allowed = False
    else:
        source = market["guard_source_by_action"].get(selected_action)
        candidate_reasons = (
            ["selected_action_source_row_missing"]
            if source is None
            else _microstructure_blocking_reasons(source, guard=guard)
        )
        candidate_allowed = source is not None and not candidate_reasons
    baseline_action = str(prediction["baseline_action"])
    baseline_source = market["guard_source_by_action"].get(baseline_action)
    baseline_reasons = (
        ["baseline_action_source_row_missing"]
        if baseline_source is None
        else _microstructure_blocking_reasons(baseline_source, guard=guard)
    )
    return {
        "pre_guard_selected_action": selected_action,
        "pre_guard_selected_side": prediction["selected_side"],
        "candidate_execution_guard_order_allowed": candidate_allowed,
        "candidate_execution_blocking_reason_codes": candidate_reasons,
        "baseline_execution_guard_order_allowed": (
            baseline_source is not None and not baseline_reasons
        ),
        "baseline_execution_blocking_reason_codes": baseline_reasons,
        "full_execution_guard_unchanged": True,
        "guard_used_outcome_or_pnl_fields": False,
    }


def _guard_replay_row(prediction: dict[str, Any]) -> dict[str, Any]:
    candidate_allowed = prediction["candidate_execution_guard_order_allowed"]
    baseline_allowed = prediction["baseline_execution_guard_order_allowed"]
    row = dict(prediction)
    row["selected_action"] = (
        prediction["pre_guard_selected_action"] if candidate_allowed else "NO_TRADE"
    )
    row["selected_side"] = (
        prediction["pre_guard_selected_side"] if candidate_allowed else "NONE"
    )
    row["selected_target_after_cost_net_pnl_per_contract"] = (
        prediction["selected_target_after_cost_net_pnl_per_contract"]
        if candidate_allowed
        else 0.0
    )
    row["baseline_action"] = prediction["baseline_action"] if baseline_allowed else "NO_TRADE"
    row["baseline_side"] = prediction["baseline_side"] if baseline_allowed else "NONE"
    row["baseline_target_after_cost_net_pnl_per_contract"] = (
        prediction["baseline_target_after_cost_net_pnl_per_contract"]
        if baseline_allowed
        else 0.0
    )
    row["guard_replay_row_id"] = canonical_json_sha256(row)
    return row


def _validate_loaded_targets(
    market: dict[str, Any],
    *,
    baseline_target: dict[str, Any],
    opposite_target: dict[str, Any],
) -> None:
    expected = str(market["market_id"])
    if (
        str(baseline_target.get("market_id")) != expected
        or str(opposite_target.get("market_id")) != expected
        or str(baseline_target.get("action")) != str(market["baseline_row"]["action"])
        or str(opposite_target.get("action")) != str(market["opposite_row"]["action"])
    ):
        raise ValueError("#242 target-loader identity mismatch")


def _attach_seed_close_timestamps(
    seed_rows: list[dict[str, Any]], *, raw_seed_rows: list[dict[str, Any]]
) -> None:
    close_by_key = {
        (str(row["market_id"]), int(row["decision_ts"]), str(row["action"])): int(
            row["market_close_ts"]
        )
        for row in raw_seed_rows
    }
    for row in seed_rows:
        key = (str(row["market_id"]), int(row["decision_ts"]), str(row["action"]))
        close_ts = close_by_key.get(key)
        if close_ts is None or close_ts <= int(row["decision_ts"]):
            raise ValueError("#242 seed market-close lineage missing")
        row["market_close_ts"] = close_ts


def _stream_markets_with_guard_sources(
    *,
    five_action_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    training_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    markets = v77._stream_markets(
        five_action_rows=five_action_rows,
        candidate_rows=candidate_rows,
        baseline_rows=baseline_rows,
        training_profile=training_profile,
    )
    source_by_key = {
        (str(row["market_id"]), int(row["decision_ts"]), str(row["action"])): row
        for row in five_action_rows
        if row.get("action") in SBC_ACTIONS
    }
    for market in markets:
        market["guard_source_by_action"] = {
            action: source_by_key.get(
                (str(market["market_id"]), int(market["decision_ts"]), action)
            )
            for action in SBC_ACTIONS
        }
        if any(value is None for value in market["guard_source_by_action"].values()):
            raise ValueError("#242 same-decision guard source missing")
    return markets


def _leakage_audit(
    seed_rows: list[dict[str, Any]],
    *,
    fit: dict[str, Any],
    model: dict[str, Any],
) -> dict[str, Any]:
    prequential = fit["prequential_rows"]
    residuals = fit["residual_lineage_rows"]
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
        "residual_available_only_after_market_close": all(
            int(row["target_available_ts"]) > int(row["decision_ts"])
            and row["residual_available_to_same_market_decision"] is False
            for row in residuals
        ),
        "strictly_prior_residual_state": all(
            row["calibration_max_target_available_ts"] < row["baseline_decision_ts"]
            and row["current_market_residual_used_for_prediction"] is False
            for row in prequential
        ),
        "issue241_not_opened": model[
            "issue241_settlement_outcome_pnl_or_gate_result_opened"
        ]
        is False,
        "no_result_selected_search": model[
            "profile_hyperparameter_or_quantile_search_performed"
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
        "current_market_target_or_residual_used_as_decision_time_input": False,
        "issue241_inputs_accepted": False,
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
        "final_conformal_state": model["final_conformal_state"],
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
    state = report["final_conformal_state"]
    return "\n".join(
        [
            "# v7.8 Rolling-Origin Conformal Positive-EV Historical Gate",
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
            f"- action overprediction q90: `{state['action_overprediction_q90']}`",
            "- paired delta overprediction q90: "
            f"`{state['paired_delta_overprediction_q90']}`",
            "- target-free collection allowed: "
            f"`{str(report['target_free_canary_collection_allowed']).lower()}`",
            "- #241 inputs used: `false`",
            "- historical replay is promotion evidence: `false`",
            "- paper/live/write/wallet/capital unlock: `false`",
            "",
        ]
    )
