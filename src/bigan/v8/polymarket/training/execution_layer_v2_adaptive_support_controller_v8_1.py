"""Outcome-free adaptive support controller for issue #245."""

from __future__ import annotations

import copy
import shutil
from collections import Counter
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
from bigan.v8.polymarket.training import (
    execution_layer_v2_rolling_origin_score_rank_abstention_v7_9 as v79,
)
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

CANDIDATE_NAME = "adaptive_support_controller_v8_1"
PROFILE_SCHEMA_VERSION = (
    "bigan-v8-adaptive-support-controller-v8-1-profile-v1"
)
MODEL_SCHEMA_VERSION = (
    "bigan-v8-adaptive-support-controller-v8-1-model-v1"
)
REPORT_SCHEMA_VERSION = (
    "bigan-v8-adaptive-support-controller-v8-1-report-v1"
)
LEAKAGE_SCHEMA_VERSION = (
    "bigan-v8-adaptive-support-controller-v8-1-leakage-v1"
)
MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-adaptive-support-controller-v8-1-manifest-v1"
)
FROZEN_LINEAGE = dict(v79.FROZEN_LINEAGE)


@dataclass(frozen=True, slots=True)
class AdaptiveSupportControllerV81Config(
    v79.RollingOriginScoreRankAbstentionV79Config
):
    """Pinned inputs for the single preregistered #245 historical replay."""


def validate_adaptive_support_controller_v8_1_profile(
    profile: dict[str, Any],
) -> None:
    """Validate the fixed online controller and unchanged v7.9 point policy."""

    checks = {
        "identity": profile.get("schema_version") == PROFILE_SCHEMA_VERSION
        and profile.get("issue_number") == 245
        and profile.get("candidate_name") == CANDIDATE_NAME
        and profile.get("preregistered_before_implementation_and_fit") is True,
        "lineage": profile.get("lineage") == FROZEN_LINEAGE,
        "exclusion": profile.get("prior_result_exclusion")
        == {
            "issue241_settlement_outcome_pnl_or_gate_result_accepted_as_input": False,
            "issue242_result_artifacts_or_current_stream_pnl_used_for_threshold_selection": False,
            "issue243_pnl_targets_winners_or_losers_used_for_threshold_selection": False,
            "issue244_pnl_targets_winners_or_losers_used_for_controller_design": False,
            "issue239_outer_oof_rows_or_metrics_accepted_as_inputs": False,
            "issue238_side_action_or_pnl_attribution_used_for_design_or_tuning": False,
            "result_selected_rerun_allowed": False,
        },
        "rank": profile.get("rank_abstention_contract")
        == {
            "score_source": "v7_7_point_selected_predicted_after_cost_return",
            "score_history_target_or_outcome_free": True,
            "eligible_prior_score_window": 60,
            "minimum_prior_eligible_score_count": 60,
            "rolling_score_quantile": "selected_by_controller",
            "controller_observation_window": 20,
            "controller_initial_quantile": 0.4,
            "controller_low_support_quantile": 0.25,
            "controller_balanced_support_quantile": 0.4,
            "controller_high_support_quantile": 0.5,
            "controller_low_support_rate_boundary_exclusive": 1.0 / 3.0,
            "controller_high_support_rate_boundary_exclusive": 0.5,
            "controller_source": "strictly_prior_full_guard_acceptance_only",
            "controller_target_outcome_pnl_free": True,
            "finite_sample_rank": "ceil(n_times_quantile)_capped_at_n",
            "current_market_score_available_before_threshold": False,
            "current_market_guard_result_available_before_threshold": False,
            "point_selected_score_must_be_strictly_positive": True,
            "point_selected_score_must_be_at_or_above_controller_quantile": True,
            "point_veto_remains_no_trade": True,
            "otherwise_abstain_to_no_trade": True,
            "quantile_window_threshold_or_profile_search_allowed": False,
        },
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("#245 v8.1 profile invalid: " + ", ".join(blockers))
    normalized = copy.deepcopy(profile)
    normalized.update(
        {
            "schema_version": v79.PROFILE_SCHEMA_VERSION,
            "issue_number": 243,
            "candidate_name": v79.CANDIDATE_NAME,
        }
    )
    normalized["prior_result_exclusion"] = {
        "issue241_settlement_outcome_pnl_or_gate_result_accepted_as_input": False,
        "issue242_result_artifacts_or_current_stream_pnl_used_for_threshold_selection": False,
        "issue239_outer_oof_rows_or_metrics_accepted_as_inputs": False,
        "issue238_side_action_or_pnl_attribution_used_for_design_or_tuning": False,
        "result_selected_rerun_allowed": False,
    }
    normalized["rank_abstention_contract"] = {
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
    }
    v79.validate_rolling_origin_score_rank_abstention_v7_9_profile(normalized)


def fit_adaptive_support_controller_v8_1(
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the v7.9-tested engine with the frozen outcome-free controller."""

    fit = v79.fit_rolling_origin_score_rank_abstention_v7_9(
        **kwargs,
        _profile_validator=(
            validate_adaptive_support_controller_v8_1_profile
        ),
        _rank_controller=_support_controller_decision,
    )
    model = fit["model_artifact"]
    model.update(
        {
            "schema_version": MODEL_SCHEMA_VERSION,
            "candidate_name": CANDIDATE_NAME,
            "issue243_pnl_targets_winners_or_losers_opened_for_threshold_selection": False,
            "issue244_pnl_targets_winners_or_losers_opened_for_controller_design": False,
        }
    )
    _rename_rank_state(
        model["final_rank_state"],
        controller_history=fit["controller_guard_acceptance_history"],
    )
    for key in ("prequential_rows", "guard_replay_rows"):
        for row in fit[key]:
            _rename_rank_fields(row)
    model["historical_controller_band_distribution"] = dict(
        sorted(
            Counter(
                row["rank_controller_decision"]["controller_band"]
                for row in fit["prequential_rows"]
            ).items()
        )
    )
    model["model_artifact_id"] = canonical_json_sha256(model)
    return fit


def _support_controller_decision(
    prior_guard_acceptance_history: tuple[bool, ...],
) -> dict[str, Any]:
    window = prior_guard_acceptance_history[-20:]
    if len(window) < 20:
        selected_quantile = 0.4
        controller_band = "initial_q40"
        acceptance_rate = None
    else:
        acceptance_rate = sum(window) / len(window)
        if acceptance_rate < 1.0 / 3.0:
            selected_quantile = 0.25
            controller_band = "low_support_q25"
        elif acceptance_rate > 0.5:
            selected_quantile = 0.5
            controller_band = "high_support_q50"
        else:
            selected_quantile = 0.4
            controller_band = "balanced_support_q40"
    decision = {
        "controller_enabled": True,
        "controller_source": "strictly_prior_full_guard_acceptance_only",
        "controller_observation_window": 20,
        "prior_observation_count": len(prior_guard_acceptance_history),
        "prior_window_observation_count": len(window),
        "prior_window_guard_acceptance_rate": acceptance_rate,
        "prior_window_hash": canonical_json_sha256(list(window)),
        "selected_quantile": selected_quantile,
        "controller_band": controller_band,
        "current_market_score_used": False,
        "current_market_guard_result_used": False,
        "target_outcome_label_or_pnl_used": False,
    }
    decision["controller_decision_id"] = canonical_json_sha256(decision)
    return decision


def run_adaptive_support_controller_v8_1_fit(
    config: AdaptiveSupportControllerV81Config,
) -> dict[str, Any]:
    """Verify pins and execute the one #245 historical gate."""

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
    _verify_pin(paths["profile"], config.expected_profile_sha256, "#245 profile")
    profile = _load_json(paths["profile"])
    validate_adaptive_support_controller_v8_1_profile(profile)
    for key, path in paths.items():
        if key != "profile":
            _verify_pin(path, profile["lineage"][f"{key}_sha256"], f"#245 {key}")
    if xgb.__version__ != profile["lineage"]["xgboost_version"]:
        raise ValueError("#245 xgboost version mismatch")
    v7_7_profile = _load_json(paths["v7_7_profile"])
    v77.validate_rolling_origin_drift_adaptive_v7_7_profile(v7_7_profile)
    if (
        profile["xgboost"] != v7_7_profile["xgboost"]
        or profile["model_contract"]["fixed_edge_buffer"]
        != v7_7_profile["model_contract"]["fixed_edge_buffer"]
    ):
        raise ValueError("#245 v7.7 model or point policy changed")
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

    fit = fit_adaptive_support_controller_v8_1(
        seed_rows=seed_rows,
        stream_markets=stream_markets,
        target_loader=target_loader,
        profile=profile,
        v6_7_profile=v6_7_profile,
        implementation_commit=config.implementation_commit,
        fit_created_ts=config.fit_created_ts,
    )
    model = fit["model_artifact"]
    leakage = v79._leakage_audit(seed_rows, fit=fit, model=model)
    leakage.update(
        {
            "schema_version": LEAKAGE_SCHEMA_VERSION,
            "candidate_name": CANDIDATE_NAME,
            "issue243_pnl_targets_winners_or_losers_used_for_threshold_selection": False,
            "issue244_pnl_targets_winners_or_losers_used_for_controller_design": False,
            "controller_strictly_prior_guard_results_only": all(
                row["current_guard_result_used_for_own_controller_decision"] is False
                and row["current_guard_result_added_after_decision_freeze"] is True
                and row["rank_controller_decision"][
                    "target_outcome_label_or_pnl_used"
                ]
                is False
                for row in fit["prequential_rows"]
            ),
        }
    )
    if not leakage["controller_strictly_prior_guard_results_only"]:
        leakage["fit_leakage_audit_passed"] = False
        leakage["fit_leakage_blocking_reason_codes"] = sorted(
            {
                *leakage["fit_leakage_blocking_reason_codes"],
                "controller_state_not_strictly_prior",
            }
        )
    leakage["leakage_audit_id"] = canonical_json_sha256(leakage)
    report = _report(model, leakage=leakage)
    run_dir = Path(config.output_dir).resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    outputs = {
        "model": run_dir / "v8_1_adaptive_support_controller_model.json",
        "report": run_dir / "v8_1_historical_prequential_hard_gate_report.json",
        "report_markdown": (
            run_dir / "v8_1_historical_prequential_hard_gate_report.md"
        ),
        "leakage_audit": run_dir / "v8_1_fit_leakage_audit.json",
        "prequential_rows": run_dir / "v8_1_prequential_policy_rows.jsonl",
        "guard_replay_rows": run_dir / "v8_1_historical_guard_replay_rows.jsonl",
        "rank_lineage_rows": (
            run_dir / "v8_1_strictly_prior_prediction_rank_lineage.jsonl"
        ),
        "runtime_target_rows": run_dir / "v8_1_consumed_stream_runtime_targets.jsonl",
        "candidate_selected_rows": run_dir / "v8_1_candidate_selected_rows.jsonl",
        "v6_7_baseline_selected_rows": (
            run_dir / "v8_1_v6_7_baseline_selected_rows.jsonl"
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
    manifest_path = run_dir / "v8_1_historical_fit_manifest.json"
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


def score_adaptive_support_controller_v8_1_market(
    market: dict[str, Any],
    *,
    model_artifact: dict[str, Any],
    prior_rank_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply one frozen adaptive-controller decision without outcomes."""

    compatibility_model = copy.deepcopy(model_artifact)
    compatibility_model["schema_version"] = v79.MODEL_SCHEMA_VERSION
    compatibility_model["candidate_name"] = v79.CANDIDATE_NAME
    state = copy.deepcopy(
        prior_rank_state or compatibility_model["final_rank_state"]
    )
    controller = _support_controller_decision(
        tuple(bool(value) for value in state["controller_guard_acceptance_history"])
    )
    _restore_v7_9_rank_state(state, controller=controller)
    compatibility_model["final_rank_state"] = state
    result = v79.score_rolling_origin_score_rank_abstention_v7_9_market(
        market,
        model_artifact=compatibility_model,
        prior_rank_state=state,
    )
    _rename_rank_fields(result)
    if "next_rank_state" in result:
        _rename_rank_state(
            result["next_rank_state"],
            controller_history=list(
                prior_rank_state.get("controller_guard_acceptance_history", [])
                if prior_rank_state is not None
                else model_artifact["final_rank_state"][
                    "controller_guard_acceptance_history"
                ]
            ),
        )
        result["next_rank_state"][
            "pending_current_guard_observation_requires_post_guard_advance"
        ] = True
        result["next_rank_state"]["rank_state_id"] = canonical_json_sha256(
            result["next_rank_state"]
        )
    result["rank_controller_decision"] = controller
    result["current_guard_result_used_for_own_controller_decision"] = False
    result["candidate_name"] = CANDIDATE_NAME
    result["decision_id"] = canonical_json_sha256(result)
    return result


def advance_adaptive_support_controller_v8_1_state(
    prior_state: dict[str, Any], *, current_guard_accepted: bool
) -> dict[str, Any]:
    """Append one guard result after its decision has already been frozen."""

    state = copy.deepcopy(prior_state)
    history = [
        *state.get("controller_guard_acceptance_history", []),
        bool(current_guard_accepted),
    ][-20:]
    state["controller_guard_acceptance_history"] = history
    state["controller_guard_acceptance_history_hash"] = canonical_json_sha256(
        history
    )
    state["next_controller_decision"] = _support_controller_decision(tuple(history))
    state["pending_current_guard_observation_requires_post_guard_advance"] = False
    state["current_guard_result_added_after_decision_freeze"] = True
    state["rank_state_id"] = canonical_json_sha256(state)
    return state


def _rename_rank_fields(row: dict[str, Any]) -> None:
    if "prior_rolling_score_q60" in row:
        row["prior_rolling_score_controller_threshold"] = row.pop(
            "prior_rolling_score_q60"
        )
    if "point_selected_score_at_or_above_prior_q60" in row:
        row["point_selected_score_at_or_above_controller_threshold"] = row.pop(
            "point_selected_score_at_or_above_prior_q60"
        )


def _rename_rank_state(
    state: dict[str, Any], *, controller_history: list[bool]
) -> None:
    state["schema_version"] = "bigan-v8-adaptive-support-controller-state-v8-1-v1"
    if "rolling_score_q60" in state:
        state["rolling_score_controller_threshold"] = state.pop(
            "rolling_score_q60"
        )
    history = [bool(value) for value in controller_history[-20:]]
    state["controller_guard_acceptance_history"] = history
    state["controller_guard_acceptance_history_hash"] = canonical_json_sha256(
        history
    )
    state["next_controller_decision"] = _support_controller_decision(tuple(history))
    state["controller_target_outcome_label_or_pnl_free"] = True
    state.update(_v7_0_blocked_safety_fields())
    state["rank_state_id"] = canonical_json_sha256(state)


def _restore_v7_9_rank_state(
    state: dict[str, Any], *, controller: dict[str, Any]
) -> None:
    state["schema_version"] = "bigan-v8-rolling-origin-score-rank-state-v7-9-v1"
    state["rolling_score_quantile"] = controller["selected_quantile"]
    if "rolling_score_controller_threshold" in state:
        state["rolling_score_q60"] = state.pop(
            "rolling_score_controller_threshold"
        )
    for key in (
        "controller_guard_acceptance_history",
        "controller_guard_acceptance_history_hash",
        "next_controller_decision",
        "controller_target_outcome_label_or_pnl_free",
        "pending_current_guard_observation_requires_post_guard_advance",
        "current_guard_result_added_after_decision_freeze",
    ):
        state.pop(key, None)
    state["rank_state_id"] = canonical_json_sha256(state)


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
        "historical_controller_band_distribution": model[
            "historical_controller_band_distribution"
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
            "# v8.1 Adaptive Support Controller Historical Gate",
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
            "- final controller threshold: "
            f"`{state['rolling_score_controller_threshold']}`",
            "- next controller band: "
            f"`{state['next_controller_decision']['controller_band']}`",
            "- historical controller bands: "
            f"`{report['historical_controller_band_distribution']}`",
            "- controller uses strictly-prior guard acceptance only: `true`",
            "- target-free collection allowed: "
            f"`{str(report['target_free_canary_collection_allowed']).lower()}`",
            "- forbidden prior result artifacts used: `false`",
            "- historical replay is promotion evidence: `false`",
            "- paper/live/write/wallet/capital unlock: `false`",
            "",
        ]
    )
