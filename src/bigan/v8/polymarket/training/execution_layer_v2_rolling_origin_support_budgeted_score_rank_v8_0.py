"""Support-budgeted rolling score-rank policy for issue #244."""

from __future__ import annotations

import copy
import shutil
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

CANDIDATE_NAME = "rolling_origin_support_budgeted_score_rank_v8_0"
PROFILE_SCHEMA_VERSION = (
    "bigan-v8-rolling-origin-support-budgeted-score-rank-v8-0-profile-v1"
)
MODEL_SCHEMA_VERSION = (
    "bigan-v8-rolling-origin-support-budgeted-score-rank-v8-0-model-v1"
)
REPORT_SCHEMA_VERSION = (
    "bigan-v8-rolling-origin-support-budgeted-score-rank-v8-0-report-v1"
)
LEAKAGE_SCHEMA_VERSION = (
    "bigan-v8-rolling-origin-support-budgeted-score-rank-v8-0-leakage-v1"
)
MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-rolling-origin-support-budgeted-score-rank-v8-0-manifest-v1"
)
FROZEN_LINEAGE = dict(v79.FROZEN_LINEAGE)


@dataclass(frozen=True, slots=True)
class RollingOriginSupportBudgetedScoreRankV80Config(
    v79.RollingOriginScoreRankAbstentionV79Config
):
    """Pinned inputs for the single preregistered #244 historical replay."""


def validate_rolling_origin_support_budgeted_score_rank_v8_0_profile(
    profile: dict[str, Any],
) -> None:
    """Validate q40 while reusing the complete frozen v7.9 contract."""

    checks = {
        "identity": profile.get("schema_version") == PROFILE_SCHEMA_VERSION
        and profile.get("issue_number") == 244
        and profile.get("candidate_name") == CANDIDATE_NAME
        and profile.get("preregistered_before_implementation_and_fit") is True,
        "lineage": profile.get("lineage") == FROZEN_LINEAGE,
        "exclusion": profile.get("prior_result_exclusion")
        == {
            "issue241_settlement_outcome_pnl_or_gate_result_accepted_as_input": False,
            "issue242_result_artifacts_or_current_stream_pnl_used_for_threshold_selection": False,
            "issue243_pnl_targets_winners_or_losers_used_for_threshold_selection": False,
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
            "rolling_score_quantile": 0.4,
            "quantile_selected_from_outcome_free_support_budget_only": True,
            "finite_sample_rank": "ceil(n_times_quantile)_capped_at_n",
            "current_market_score_available_before_threshold": False,
            "point_selected_score_must_be_strictly_positive": True,
            "point_selected_score_must_be_at_or_above_prior_q40": True,
            "point_veto_remains_no_trade": True,
            "otherwise_abstain_to_no_trade": True,
            "quantile_window_threshold_or_profile_search_allowed": False,
        },
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("#244 v8.0 profile invalid: " + ", ".join(blockers))
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


def fit_rolling_origin_support_budgeted_score_rank_v8_0(
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the v7.9-tested engine under the separately frozen q40 contract."""

    fit = v79.fit_rolling_origin_score_rank_abstention_v7_9(
        **kwargs,
        _profile_validator=(
            validate_rolling_origin_support_budgeted_score_rank_v8_0_profile
        ),
    )
    model = fit["model_artifact"]
    model.update(
        {
            "schema_version": MODEL_SCHEMA_VERSION,
            "candidate_name": CANDIDATE_NAME,
            "issue243_pnl_targets_winners_or_losers_opened_for_threshold_selection": False,
        }
    )
    _rename_rank_state(model["final_rank_state"])
    for key in ("prequential_rows", "guard_replay_rows"):
        for row in fit[key]:
            _rename_rank_fields(row)
    model["model_artifact_id"] = canonical_json_sha256(model)
    return fit


def run_rolling_origin_support_budgeted_score_rank_v8_0_fit(
    config: RollingOriginSupportBudgetedScoreRankV80Config,
) -> dict[str, Any]:
    """Verify pins and execute the one #244 historical gate."""

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
    _verify_pin(paths["profile"], config.expected_profile_sha256, "#244 profile")
    profile = _load_json(paths["profile"])
    validate_rolling_origin_support_budgeted_score_rank_v8_0_profile(profile)
    for key, path in paths.items():
        if key != "profile":
            _verify_pin(path, profile["lineage"][f"{key}_sha256"], f"#244 {key}")
    if xgb.__version__ != profile["lineage"]["xgboost_version"]:
        raise ValueError("#244 xgboost version mismatch")
    v7_7_profile = _load_json(paths["v7_7_profile"])
    v77.validate_rolling_origin_drift_adaptive_v7_7_profile(v7_7_profile)
    if (
        profile["xgboost"] != v7_7_profile["xgboost"]
        or profile["model_contract"]["fixed_edge_buffer"]
        != v7_7_profile["model_contract"]["fixed_edge_buffer"]
    ):
        raise ValueError("#244 v7.7 model or point policy changed")
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

    fit = fit_rolling_origin_support_budgeted_score_rank_v8_0(
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
        "model": run_dir / "v8_0_support_budgeted_score_rank_model.json",
        "report": run_dir / "v8_0_historical_prequential_hard_gate_report.json",
        "report_markdown": (
            run_dir / "v8_0_historical_prequential_hard_gate_report.md"
        ),
        "leakage_audit": run_dir / "v8_0_fit_leakage_audit.json",
        "prequential_rows": run_dir / "v8_0_prequential_policy_rows.jsonl",
        "guard_replay_rows": run_dir / "v8_0_historical_guard_replay_rows.jsonl",
        "rank_lineage_rows": (
            run_dir / "v8_0_strictly_prior_prediction_rank_lineage.jsonl"
        ),
        "runtime_target_rows": run_dir / "v8_0_consumed_stream_runtime_targets.jsonl",
        "candidate_selected_rows": run_dir / "v8_0_candidate_selected_rows.jsonl",
        "v6_7_baseline_selected_rows": (
            run_dir / "v8_0_v6_7_baseline_selected_rows.jsonl"
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
    manifest_path = run_dir / "v8_0_historical_fit_manifest.json"
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


def score_rolling_origin_support_budgeted_score_rank_v8_0_market(
    market: dict[str, Any],
    *,
    model_artifact: dict[str, Any],
    prior_rank_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the frozen q40 rank state without reading any outcome."""

    compatibility_model = copy.deepcopy(model_artifact)
    compatibility_model["schema_version"] = v79.MODEL_SCHEMA_VERSION
    state = compatibility_model["final_rank_state"]
    _restore_v7_9_rank_state(state)
    result = v79.score_rolling_origin_score_rank_abstention_v7_9_market(
        market,
        model_artifact=compatibility_model,
        prior_rank_state=prior_rank_state,
    )
    _rename_rank_fields(result)
    if "next_rank_state" in result:
        _rename_rank_state(result["next_rank_state"])
    result["candidate_name"] = CANDIDATE_NAME
    result["decision_id"] = canonical_json_sha256(result)
    return result


def _rename_rank_fields(row: dict[str, Any]) -> None:
    if "prior_rolling_score_q60" in row:
        row["prior_rolling_score_q40"] = row.pop("prior_rolling_score_q60")
    if "point_selected_score_at_or_above_prior_q60" in row:
        row["point_selected_score_at_or_above_prior_q40"] = row.pop(
            "point_selected_score_at_or_above_prior_q60"
        )


def _rename_rank_state(state: dict[str, Any]) -> None:
    state["schema_version"] = "bigan-v8-rolling-origin-score-rank-state-v8-0-v1"
    if "rolling_score_q60" in state:
        state["rolling_score_q40"] = state.pop("rolling_score_q60")
    state["quantile_selected_from_outcome_free_support_budget_only"] = True
    state["rank_state_id"] = canonical_json_sha256(state)


def _restore_v7_9_rank_state(state: dict[str, Any]) -> None:
    state["schema_version"] = "bigan-v8-rolling-origin-score-rank-state-v7-9-v1"
    if "rolling_score_q40" in state:
        state["rolling_score_q60"] = state.pop("rolling_score_q40")
    state.pop("quantile_selected_from_outcome_free_support_budget_only", None)
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
            "# v8.0 Support-Budgeted Score-Rank Historical Gate",
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
            f"- final rolling q40: `{state['rolling_score_q40']}`",
            "- q40 selected from outcome-free support budget only: `true`",
            "- target-free collection allowed: "
            f"`{str(report['target_free_canary_collection_allowed']).lower()}`",
            "- forbidden prior result artifacts used: `false`",
            "- historical replay is promotion evidence: `false`",
            "- paper/live/write/wallet/capital unlock: `false`",
            "",
        ]
    )
