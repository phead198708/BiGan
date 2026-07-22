"""Rolling-origin drift-adaptive action-value policy for issue #240."""

from __future__ import annotations

import base64
import math
import shutil
from collections import defaultdict
from collections.abc import Callable
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
    FEATURE_NAMES,
    _canonical_training_row,
    _common_feature_values,
    _side_anchor,
    materialize_v7_0_sbc_rows,
    validate_v7_0_training_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_nested_boosted_action_value_v7_4 import (
    FROZEN_XGB,
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
    _historical_replay,
    _market_order,
    _validate_canonical_rows,
)

CANDIDATE_NAME = "rolling_origin_drift_adaptive_action_value_v7_7"
PROFILE_SCHEMA_VERSION = "bigan-v8-rolling-origin-drift-adaptive-action-value-v7-7-profile-v1"
MODEL_SCHEMA_VERSION = "bigan-v8-rolling-origin-drift-adaptive-action-value-v7-7-model-v1"
REPORT_SCHEMA_VERSION = "bigan-v8-rolling-origin-drift-adaptive-action-value-v7-7-report-v1"
LEAKAGE_SCHEMA_VERSION = "bigan-v8-rolling-origin-drift-adaptive-action-value-v7-7-leakage-v1"
MANIFEST_SCHEMA_VERSION = "bigan-v8-rolling-origin-drift-adaptive-action-value-v7-7-manifest-v1"
SBC_ACTIONS = ("BUY_UP_SELL_BEFORE_CLOSE", "BUY_DOWN_SELL_BEFORE_CLOSE")
FROZEN_LINEAGE = {
    "seed_runtime_target_rows_sha256": "1565116daeb2f5d4d8c33fefa507276f59251edd5ffb5f4f313041bcf9dbb0ec",
    "v7_0_training_profile_sha256": "1f66d8699b9727651538cc34a9a2a25ba5eaac5cfded75cf8f4a258b1b5d3f4a",
    "v6_7_candidate_profile_sha256": "cec55d243acd6bbf60a5e8474545b487086ddcd4d18073682ae7f2d4660d2248",
    "runtime_policy_profile_sha256": "1306f6b6f7a6c1216b23413352ff66f4061ec62a9751b0de51eded256ca51264",
    "consumed_stream_five_action_rows_sha256": "929357e4ec57746dae04f608fcbe7740375e40a95137813048080a10ffb06bc5",
    "consumed_stream_v6_7_candidate_rows_sha256": "52acf2b99e855a145122ee595566f300f2fe11d5d9fdeb392e7d63c0cf5638f0",
    "consumed_stream_v6_7_baseline_rows_sha256": "ef2e66a0e7577e1230f7871d7a713dbdbff1c315dec29536409d93d583d00cb1",
    "consumed_stream_settled_index_sha256": "d635de9d03eb8df12410ebdc533cb5cae279df1892393183e02aeb96b024c060",
    "consumed_stream_target_free_freeze_manifest_sha256": "cda467291db0fa03a8b7f6810fb9afb1be886092713da99d37b341e60d327c11",
    "xgboost_version": "3.2.0",
}


@dataclass(frozen=True, slots=True)
class RollingOriginDriftAdaptiveV77Config:
    """Pinned inputs for the one #240 rolling-origin replay."""

    run_id: str
    output_dir: Path | str
    profile_path: Path | str
    expected_profile_sha256: str
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


def validate_rolling_origin_drift_adaptive_v7_7_profile(
    profile: dict[str, Any],
) -> None:
    """Reject drift from the preregistered #240 contract."""

    checks = {
        "identity": profile.get("schema_version") == PROFILE_SCHEMA_VERSION
        and profile.get("issue_number") == 240
        and profile.get("candidate_name") == CANDIDATE_NAME
        and profile.get("preregistered_before_implementation_and_fit") is True,
        "lineage": profile.get("lineage") == FROZEN_LINEAGE,
        "stream": profile.get("historical_stream")
        == {
            "seed_market_count": 134,
            "prequential_market_count": 120,
            "market_order": "minimum_selected_decision_ts_then_market_id",
            "current_market_target_available_before_prediction": False,
            "prior_stream_target_available_only_after_prediction_freeze": True,
            "prior_market_close_must_precede_current_decision": True,
            "stream_is_consumed_historical_development_not_promotion_evidence": True,
        },
        "exclusion": profile.get("prior_result_exclusion")
        == {
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
            "allowed_policy_decisions": [
                "KEEP_V6_7",
                "SWITCH_SAME_DECISION_SBC",
                "VETO_TO_NO_TRADE",
            ],
            "switch_requires_opposite_positive": True,
            "veto_requires_baseline_below_negative_buffer": True,
            "v6_7_no_trade_activation_allowed": False,
            "same_decision_timestamp_required": True,
            "maximum_bets_per_market": 1,
            "hts_disabled_fail_closed": True,
        },
        "xgboost": profile.get("xgboost") == FROZEN_XGB,
        "gate": profile.get("historical_prequential_noninferiority_gate")
        == {
            "exact_evaluation_market_count": 120,
            "fixed_position_size": 0.2,
            "no_bet_market_pnl": 0.0,
            "candidate_minus_v6_7_total_pnl_minimum_inclusive": 0.0,
            "candidate_minus_v6_7_largest_winner_removed_pnl_minimum_inclusive": 0.0,
            "comparison_operator": "greater_than_or_equal",
            "equality_passes_noninferiority": True,
            "minimum_policy_difference_market_count_for_collection": 3,
            "exact_baseline_identity_reconciliation_required": True,
            "same_runtime_target_cost_size_guard_and_position_management_required": True,
            "failure_stops_before_collection": True,
            "historical_replay_is_promotion_evidence": False,
        },
        "canary": profile.get("target_free_canary")
        == {
            "strictly_later_outcome_blind_market_count": 12,
            "historical_noninferiority_and_actionability_required": True,
            "minimum_guard_accepted_policy_difference_market_count": 1,
            "outcomes_resolution_labels_or_pnl_opened": False,
        },
        "safety": profile.get("safety") == _v7_0_blocked_safety_fields(),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("#240 v7.7 profile invalid: " + ", ".join(blockers))


def fit_rolling_origin_drift_adaptive_v7_7(
    *,
    seed_rows: list[dict[str, Any]],
    stream_markets: list[dict[str, Any]],
    target_loader: Callable[[dict[str, Any], dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]],
    profile: dict[str, Any],
    implementation_commit: str,
    fit_created_ts: int,
) -> dict[str, Any]:
    """Freeze each prediction before loading that market's historical targets."""

    validate_rolling_origin_drift_adaptive_v7_7_profile(profile)
    _validate_canonical_rows(seed_rows)
    seed_order = _market_order(seed_rows)
    expected_seed = int(profile["historical_stream"]["seed_market_count"])
    expected_stream = int(profile["historical_stream"]["prequential_market_count"])
    if len(seed_order) != expected_seed or len(stream_markets) != expected_stream:
        raise ValueError("#240 historical market support invalid")
    stream_markets = sorted(
        stream_markets,
        key=lambda row: (int(row["decision_ts"]), str(row["market_id"])),
    )
    if len({row["market_id"] for row in stream_markets}) != expected_stream:
        raise ValueError("#240 duplicate prequential market")
    if max(int(row["decision_ts"]) for row in seed_rows) >= int(
        stream_markets[0]["decision_ts"]
    ):
        raise ValueError("#240 seed/stream chronology invalid")
    training_rows = list(seed_rows)
    training_order = list(seed_order)
    prequential_rows = []
    loaded_target_rows = []
    prior_close_ts: int | None = None
    for stream_index, market in enumerate(stream_markets):
        if prior_close_ts is not None and prior_close_ts >= int(market["decision_ts"]):
            raise ValueError("#240 prior target unavailable before current decision")
        artifact = _fit_weighted_model(
            training_rows,
            market_order=training_order,
            profile=profile,
        )
        prediction = _score_stream_market(market, artifact=artifact, profile=profile)
        prediction["stream_index"] = stream_index
        prediction["prediction_frozen_before_current_target_access"] = True
        prediction["training_market_count"] = len(training_order)
        prediction["training_max_decision_ts"] = max(
            int(row["decision_ts"]) for row in training_rows
        )
        prediction["current_market_target_used_for_prediction"] = False
        baseline_target, opposite_target = target_loader(
            market["baseline_row"], market["opposite_row"]
        )
        if baseline_target["market_id"] != market["market_id"] or opposite_target[
            "market_id"
        ] != market["market_id"]:
            raise ValueError("#240 target-loader market identity mismatch")
        prediction.update(
            _attach_targets(
                prediction,
                baseline_target=baseline_target,
                opposite_target=opposite_target,
            )
        )
        prequential_rows.append(prediction)
        loaded_target_rows.extend((baseline_target, opposite_target))
        training_rows.extend(
            (
                _canonical_stream_training_row(
                    market["baseline_row"], baseline_target, role="consumed_prequential"
                ),
                _canonical_stream_training_row(
                    market["opposite_row"], opposite_target, role="consumed_prequential"
                ),
            )
        )
        training_order.append(str(market["market_id"]))
        prior_close_ts = int(market["market_close_ts"])
    final_artifact = _fit_weighted_model(
        training_rows,
        market_order=training_order,
        profile=profile,
    )
    replay_profile = {
        "historical_replay_superiority_gate": {
            "exact_evaluation_market_count": expected_stream,
            "fixed_position_size": profile[
                "historical_prequential_noninferiority_gate"
            ]["fixed_position_size"],
        }
    }
    replay = _historical_replay(prequential_rows, profile=replay_profile)
    replay["gate_name"] = "same_stream_prequential_noninferiority_to_v6_7"
    replay["comparison_operator"] = "greater_than_or_equal"
    replay["equality_passes_noninferiority"] = True
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
    difference_count = sum(
        row["selected_policy_decision"] != "KEEP_V6_7"
        for row in prequential_rows
    )
    gate_contract = profile["historical_prequential_noninferiority_gate"]
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
        "same_decision_alternatives_only": all(
            row["baseline_decision_ts"] == row["opposite_decision_ts"]
            for row in prequential_rows
        ),
        "baseline_identity_reconciled": replay["baseline_identity_reconciled"],
        "candidate_total_pnl_noninferior_to_v6_7": total_delta
        >= float(gate_contract["candidate_minus_v6_7_total_pnl_minimum_inclusive"]),
        "candidate_largest_winner_removed_noninferior_to_v6_7": lwr_delta
        >= float(
            gate_contract[
                "candidate_minus_v6_7_largest_winner_removed_pnl_minimum_inclusive"
            ]
        ),
        "final_model_available": final_artifact["available"],
    }
    reason_map = {
        "exact_prequential_market_support": "prequential_market_support_invalid",
        "prediction_before_current_target_access": "current_market_target_opened_before_prediction",
        "strict_prior_target_chronology": "non_prior_target_used_for_prediction",
        "same_decision_alternatives_only": "alternative_decision_timestamp_used",
        "baseline_identity_reconciled": "frozen_v6_7_baseline_identity_mismatch",
        "candidate_total_pnl_noninferior_to_v6_7": "historical_same_stream_candidate_pnl_worse_than_v6_7",
        "candidate_largest_winner_removed_noninferior_to_v6_7": "historical_same_stream_lwr_pnl_worse_than_v6_7",
        "final_model_available": "final_weighted_model_unavailable",
    }
    blockers = [reason_map[name] for name, passed in checks.items() if not passed]
    gate_passed = not blockers
    minimum_differences = int(
        gate_contract["minimum_policy_difference_market_count_for_collection"]
    )
    actionability_checks = {
        "historical_noninferiority_gate_passed": gate_passed,
        "minimum_policy_difference_market_count_met": difference_count
        >= minimum_differences,
    }
    actionability_blockers = [
        name for name, passed in actionability_checks.items() if not passed
    ]
    model = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": implementation_commit,
        "fit_created_ts": fit_created_ts,
        "frozen": gate_passed,
        "decision_time_safe": True,
        "final_weighted_model": final_artifact,
        "historical_prequential_noninferiority_gate": replay,
        "historical_policy_difference_market_count": difference_count,
        "historical_noninferiority_gate_passed": gate_passed,
        "historical_gate_checks": checks,
        "historical_gate_blocking_reason_codes": blockers,
        "historical_actionability_checks": actionability_checks,
        "historical_actionability_blocking_reason_codes": actionability_blockers,
        "model_improvement_demonstrated": total_delta > 0.0 and lwr_delta >= 0.0,
        "issue239_outer_oof_rows_or_metrics_opened": False,
        "issue238_side_action_or_pnl_attribution_used_for_tuning": False,
        "profile_or_hyperparameter_search_performed": False,
        "consumed_stream_is_historical_development_not_promotion_evidence": True,
        "target_free_canary_collection_allowed": not actionability_blockers,
        "target_free_canary_started": False,
        **_v7_0_blocked_safety_fields(),
    }
    model["model_artifact_id"] = canonical_json_sha256(model)
    return {
        "model_artifact": model,
        "prequential_rows": prequential_rows,
        "loaded_runtime_target_rows": loaded_target_rows,
        "candidate_selected_rows": candidate_rows,
        "v6_7_baseline_selected_rows": baseline_rows,
    }


def run_rolling_origin_drift_adaptive_v7_7_fit(
    config: RollingOriginDriftAdaptiveV77Config,
) -> dict[str, Any]:
    """Verify lineage and run the single historical prequential evaluation."""

    paths = {
        "profile": Path(config.profile_path).resolve(),
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
    _verify_pin(paths["profile"], config.expected_profile_sha256, "#240 profile")
    profile = _load_json(paths["profile"])
    validate_rolling_origin_drift_adaptive_v7_7_profile(profile)
    for key in paths:
        if key == "profile":
            continue
        _verify_pin(paths[key], profile["lineage"][f"{key}_sha256"], f"#240 {key}")
    if xgb.__version__ != profile["lineage"]["xgboost_version"]:
        raise ValueError("#240 xgboost version mismatch")
    training_profile = _load_json(paths["v7_0_training_profile"])
    validate_v7_0_training_profile(training_profile)
    validate_p_up_semantic_compatibility_v6_7_profile(
        _load_json(paths["v6_7_candidate_profile"])
    )
    runtime_profile = _load_json(paths["runtime_policy_profile"])
    validate_runtime_aligned_sbc_net_return_v6_4_profile(runtime_profile)
    seed_rows = materialize_v7_0_sbc_rows(
        _load_jsonl(paths["seed_runtime_target_rows"]), training_profile
    )
    settled_index = _load_json(paths["consumed_stream_settled_index"])
    freeze_manifest = _load_json(paths["consumed_stream_target_free_freeze_manifest"])
    _validate_consumed_lineage(
        settled_index,
        freeze_manifest=freeze_manifest,
        freeze_manifest_path=paths["consumed_stream_target_free_freeze_manifest"],
        paths=paths,
    )
    stream_markets = _stream_markets(
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
                [_target_decision(baseline_row)],
                settled_entries=settled_entries,
                runtime_profile=runtime_profile,
                run_id=f"{config.run_id}-baseline",
                role="consumed_historical_prequential",
            )[0],
            _runtime_targets_for_decisions(
                [_target_decision(opposite_row)],
                settled_entries=settled_entries,
                runtime_profile=runtime_profile,
                run_id=f"{config.run_id}-opposite",
                role="consumed_historical_prequential",
            )[0],
        )

    fit = fit_rolling_origin_drift_adaptive_v7_7(
        seed_rows=seed_rows,
        stream_markets=stream_markets,
        target_loader=target_loader,
        profile=profile,
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
        "model": run_dir / "v7_7_rolling_origin_model.json",
        "report": run_dir / "v7_7_historical_prequential_noninferiority_report.json",
        "report_markdown": run_dir / "v7_7_historical_prequential_noninferiority_report.md",
        "leakage_audit": run_dir / "v7_7_fit_leakage_audit.json",
        "prequential_rows": run_dir / "v7_7_prequential_policy_rows.jsonl",
        "runtime_target_rows": run_dir / "v7_7_consumed_stream_runtime_targets.jsonl",
        "candidate_selected_rows": run_dir / "v7_7_candidate_selected_rows.jsonl",
        "v6_7_baseline_selected_rows": run_dir / "v7_7_v6_7_baseline_selected_rows.jsonl",
    }
    _write_json(outputs["model"], model)
    _write_json(outputs["report"], report)
    _write_text(outputs["report_markdown"], _report_markdown(report))
    _write_json(outputs["leakage_audit"], leakage)
    _write_jsonl(outputs["prequential_rows"], fit["prequential_rows"])
    _write_jsonl(outputs["runtime_target_rows"], fit["loaded_runtime_target_rows"])
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
        "target_free_canary_collection_allowed": model[
            "target_free_canary_collection_allowed"
        ],
        "target_free_canary_started": False,
        "fit_leakage_audit_passed": leakage["fit_leakage_audit_passed"],
        **_v7_0_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v7_7_historical_fit_manifest.json"
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


def _fit_weighted_model(
    rows: list[dict[str, Any]],
    *,
    market_order: list[str],
    profile: dict[str, Any],
) -> dict[str, Any]:
    minimum = int(profile["model_contract"]["minimum_training_market_count"])
    if len(market_order) < minimum:
        return {"available": False, "reason_codes": ["training_market_support_insufficient"]}
    position = {market_id: index for index, market_id in enumerate(market_order)}
    half_life = float(profile["model_contract"]["exponential_decay_half_life_markets"])
    features = np.asarray(
        [
            [float(row["decision_time_features"][name]) for name in FEATURE_NAMES]
            for row in rows
        ],
        dtype=float,
    )
    targets = np.asarray(
        [float(row["target_after_cost_net_pnl_per_contract"]) for row in rows],
        dtype=float,
    )
    latest = len(market_order) - 1
    weights = np.asarray(
        [2.0 ** (-(latest - position[str(row["market_id"])]) / half_life) for row in rows],
        dtype=float,
    )
    if not (
        np.all(np.isfinite(features))
        and np.all(np.isfinite(targets))
        and np.all(np.isfinite(weights))
        and np.all(weights > 0.0)
    ):
        raise ValueError("#240 weighted training values invalid")
    config = dict(profile["xgboost"])
    rounds = int(config.pop("num_boost_round"))
    config.pop("early_stopping_enabled")
    config.pop("canonical_features_must_be_finite")
    booster = xgb.train(
        config,
        xgb.DMatrix(features, label=targets, weight=weights, missing=np.nan),
        num_boost_round=rounds,
        verbose_eval=False,
    )
    raw = bytes(booster.save_raw(raw_format="json"))
    return {
        "available": True,
        "model_family": profile["model_contract"]["model_family"],
        "training_market_count": len(market_order),
        "training_row_count": len(rows),
        "training_market_ids_hash": canonical_json_sha256(market_order),
        "exponential_decay_half_life_markets": half_life,
        "minimum_row_weight": float(np.min(weights)),
        "maximum_row_weight": float(np.max(weights)),
        "feature_names": list(FEATURE_NAMES),
        "booster_json_base64": base64.b64encode(raw).decode("ascii"),
        "booster_sha256": canonical_json_sha256(base64.b64encode(raw).decode("ascii")),
        "xgboost_parameters": profile["xgboost"],
    }


def _predict(row: dict[str, Any], artifact: dict[str, Any]) -> float:
    values = np.asarray(
        [[float(row["decision_time_features"][name]) for name in FEATURE_NAMES]],
        dtype=float,
    )
    booster = xgb.Booster()
    booster.load_model(bytearray(base64.b64decode(artifact["booster_json_base64"])))
    value = float(booster.predict(xgb.DMatrix(values, missing=np.nan))[0])
    if not math.isfinite(value):
        raise ValueError("#240 model prediction is non-finite")
    return value


def _score_stream_market(
    market: dict[str, Any], *, artifact: dict[str, Any], profile: dict[str, Any]
) -> dict[str, Any]:
    baseline = market["baseline_row"]
    opposite = market["opposite_row"]
    predicted_baseline = _predict(baseline, artifact)
    predicted_opposite = _predict(opposite, artifact)
    buffer = float(profile["model_contract"]["fixed_edge_buffer"])
    if predicted_opposite > 0.0 and predicted_opposite - predicted_baseline >= buffer:
        decision = "SWITCH_SAME_DECISION_SBC"
    elif predicted_baseline < -buffer:
        decision = "VETO_TO_NO_TRADE"
    else:
        decision = "KEEP_V6_7"
    selected_action = (
        opposite["action"]
        if decision == "SWITCH_SAME_DECISION_SBC"
        else "NO_TRADE"
        if decision == "VETO_TO_NO_TRADE"
        else baseline["action"]
    )
    selected_side = (
        opposite["side"]
        if decision == "SWITCH_SAME_DECISION_SBC"
        else "NONE"
        if decision == "VETO_TO_NO_TRADE"
        else baseline["side"]
    )
    return {
        "market_id": market["market_id"],
        "market_close_ts": market["market_close_ts"],
        "baseline_action": baseline["action"],
        "baseline_side": baseline["side"],
        "baseline_decision_ts": baseline["decision_ts"],
        "baseline_max_input_ts": baseline["max_input_ts"],
        "opposite_action": opposite["action"],
        "opposite_side": opposite["side"],
        "opposite_decision_ts": opposite["decision_ts"],
        "opposite_max_input_ts": opposite["max_input_ts"],
        "predicted_baseline_return": predicted_baseline,
        "predicted_opposite_return": predicted_opposite,
        "fixed_edge_buffer": buffer,
        "selected_policy_decision": decision,
        "selected_action": selected_action,
        "selected_side": selected_side,
        "target_used_as_decision_time_input": False,
        "source_score_mutated": False,
    }


def _attach_targets(
    prediction: dict[str, Any],
    *,
    baseline_target: dict[str, Any],
    opposite_target: dict[str, Any],
) -> dict[str, Any]:
    baseline_value = float(
        baseline_target["runtime_policy_after_cost_net_pnl_per_contract"]
    )
    opposite_value = float(
        opposite_target["runtime_policy_after_cost_net_pnl_per_contract"]
    )
    decision = prediction["selected_policy_decision"]
    selected_value = (
        opposite_value
        if decision == "SWITCH_SAME_DECISION_SBC"
        else 0.0
        if decision == "VETO_TO_NO_TRADE"
        else baseline_value
    )
    return {
        "baseline_target_after_cost_net_pnl_per_contract": baseline_value,
        "opposite_target_after_cost_net_pnl_per_contract": opposite_value,
        "selected_target_after_cost_net_pnl_per_contract": selected_value,
        "current_market_target_accessed_only_after_prediction_freeze": True,
        "outer_validation_target_used_for_profile_selection_or_fit": False,
    }


def _stream_markets(
    *,
    five_action_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    training_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    action_by_key = {
        (str(row["market_id"]), int(row["decision_ts"]), str(row["action"])): row
        for row in five_action_rows
        if row.get("action") in SBC_ACTIONS
    }
    candidate_by_key = {
        (str(row["market_id"]), int(row["decision_ts"]), str(row["action"])): row
        for row in candidate_rows
    }
    candidates_by_group: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        candidates_by_group[(str(row["market_id"]), int(row["decision_ts"]))].append(row)
    output = []
    for baseline in baseline_rows:
        market_id = str(baseline["market_id"])
        decision_ts = int(baseline["decision_ts"])
        action = str(baseline["action"])
        opposite_action = (
            "BUY_DOWN_SELL_BEFORE_CLOSE"
            if action == "BUY_UP_SELL_BEFORE_CLOSE"
            else "BUY_UP_SELL_BEFORE_CLOSE"
        )
        baseline_source = action_by_key.get((market_id, decision_ts, action))
        opposite_source = action_by_key.get((market_id, decision_ts, opposite_action))
        if baseline_source is None or opposite_source is None:
            raise ValueError("#240 same-decision SBC action pair missing")
        group = candidates_by_group[(market_id, decision_ts)]
        baseline_canonical = _canonical_stream_feature_row(
            baseline_source,
            score_row=candidate_by_key.get((market_id, decision_ts, action)),
            group_candidates=group,
            training_profile=training_profile,
        )
        opposite_canonical = _canonical_stream_feature_row(
            opposite_source,
            score_row=candidate_by_key.get((market_id, decision_ts, opposite_action)),
            group_candidates=group,
            training_profile=training_profile,
        )
        output.append(
            {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "market_close_ts": int(baseline["market_close_ts"]),
                "baseline_row": baseline_canonical,
                "opposite_row": opposite_canonical,
            }
        )
    return output


def _canonical_stream_feature_row(
    row: dict[str, Any],
    *,
    score_row: dict[str, Any] | None,
    group_candidates: list[dict[str, Any]],
    training_profile: dict[str, Any],
) -> dict[str, Any]:
    features = dict(row["decision_time_features"])
    side = str(row["side"])
    score = (
        float(score_row["calibrated_action_expected_net_return"])
        if score_row is not None
        else 0.0
    )
    other_scores = [
        float(item["calibrated_action_expected_net_return"])
        for item in group_candidates
        if str(item.get("action")) != str(row["action"])
    ]
    margin = score - max(other_scores) if score_row is not None and other_scores else 0.0
    values = _common_feature_values(
        action_score_available=float(score_row is not None),
        action_score=score,
        action_score_margin=margin,
        btc_anchor_direction=_side_anchor(
            side,
            [
                features.get("btc_return_30s"),
                features.get("btc_return_1m"),
                features.get("reference_price_to_beat_distance_at_decision"),
            ],
        ),
        selected_side_probability=features.get("selected_side_probability"),
        execution_price=features.get("execution_price"),
        spread_bps=features.get("selected_side_spread_bps"),
        queue_fill=features.get("selected_side_queue_fill_probability_proxy"),
        book_staleness_ms=features.get("selected_side_book_staleness_ms"),
        time_to_close_seconds=features.get("time_to_close_seconds"),
        pre_entry_market_exposure=0.0,
        same_side_prior_entry=0.0,
        side_flip_prior_entry=0.0,
        side=side,
        profile=training_profile,
    )
    return {
        "source": "consumed_238_target_free_action_grid",
        "market_id": str(row["market_id"]),
        "decision_group_id": f"{row['market_id']}|{row['decision_ts']}",
        "decision_ts": int(row["decision_ts"]),
        "max_input_ts": int(row["max_input_ts"]),
        "role": "consumed_historical_prequential",
        "action_family": "SELL_BEFORE_CLOSE",
        "action": str(row["action"]),
        "side": side,
        "microstructure_snapshot": dict(row["microstructure_snapshot"]),
        "decision_time_features": values,
        "target_used_as_decision_time_input": False,
    }


def _canonical_stream_training_row(
    feature_row: dict[str, Any], target_row: dict[str, Any], *, role: str
) -> dict[str, Any]:
    return _canonical_training_row(
        source="consumed_238_runtime_aligned_target",
        market_id=str(feature_row["market_id"]),
        decision_ts=int(feature_row["decision_ts"]),
        max_input_ts=int(feature_row["max_input_ts"]),
        role=role,
        family="SELL_BEFORE_CLOSE",
        action=str(feature_row["action"]),
        side=str(feature_row["side"]),
        values=dict(feature_row["decision_time_features"]),
        target=float(target_row["runtime_policy_after_cost_net_pnl_per_contract"]),
    )


def _target_decision(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "market_id": row["market_id"],
        "decision_ts": row["decision_ts"],
        "max_input_ts": row["max_input_ts"],
        "side": row["side"],
        "action": row["action"],
        "microstructure_snapshot": dict(row["microstructure_snapshot"]),
    }


def _validate_consumed_lineage(
    settled_index: dict[str, Any],
    *,
    freeze_manifest: dict[str, Any],
    freeze_manifest_path: Path,
    paths: dict[str, Path],
) -> None:
    if (
        int(settled_index.get("entry_count") or 0) != 120
        or len(settled_index.get("entries") or []) != 120
        or settled_index.get("outcomes_used_for_decision_selection_or_tuning") is not False
        or settled_index.get("source_outcome_blind_rounds_mutated") is not False
        or settled_index.get("target_free_freeze_manifest") != _descriptor(freeze_manifest_path)
        or freeze_manifest.get("target_free_freeze_passed") is not True
        or freeze_manifest.get("labels_outcomes_resolution_or_pnl_opened") is not False
        or freeze_manifest["target_free_five_action_rows"]
        != _descriptor(paths["consumed_stream_five_action_rows"])
        or freeze_manifest["v6_7_candidate_rows"]
        != _descriptor(paths["consumed_stream_v6_7_candidate_rows"])
        or freeze_manifest["baseline_v6_7_decisions"]
        != _descriptor(paths["consumed_stream_v6_7_baseline_rows"])
    ):
        raise ValueError("#240 consumed #238 lineage invalid")


def _leakage_audit(
    seed_rows: list[dict[str, Any]],
    *,
    fit: dict[str, Any],
    model: dict[str, Any],
) -> dict[str, Any]:
    prequential = fit["prequential_rows"]
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
        "issue239_oof_not_opened": model[
            "issue239_outer_oof_rows_or_metrics_opened"
        ]
        is False,
        "no_result_selected_search": model["profile_or_hyperparameter_search_performed"]
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
        "consumed_stream_replay_is_promotion_evidence": False,
        **_v7_0_blocked_safety_fields(),
    }
    audit["leakage_audit_id"] = canonical_json_sha256(audit)
    return audit


def _report(model: dict[str, Any], *, leakage: dict[str, Any]) -> dict[str, Any]:
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
        "historical_prequential_noninferiority_gate": model[
            "historical_prequential_noninferiority_gate"
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
    replay = report["historical_prequential_noninferiority_gate"]
    return "\n".join(
        [
            "# v7.7 Rolling-Origin Historical Non-Inferiority Replay",
            "",
            "- historical non-inferiority gate passed: "
            f"`{str(report['historical_noninferiority_gate_passed']).lower()}`",
            f"- hard-gate blockers: `{report['historical_gate_blocking_reason_codes']}`",
            "- candidate PnL: "
            f"`{replay['candidate']['total_after_cost_net_pnl_at_frozen_size']}`",
            "- v6.7 PnL: "
            f"`{replay['v6_7_baseline']['total_after_cost_net_pnl_at_frozen_size']}`",
            "- candidate-minus-v6.7 PnL: "
            f"`{replay['candidate_minus_v6_7_total_after_cost_net_pnl_at_frozen_size']}`",
            "- policy difference markets: "
            f"`{report['historical_policy_difference_market_count']}`",
            "- actionability blockers: "
            f"`{report['historical_actionability_blocking_reason_codes']}`",
            "- target-free collection allowed: "
            f"`{str(report['target_free_canary_collection_allowed']).lower()}`",
            "- consumed stream is promotion evidence: `false`",
            "- paper/live/write/wallet/capital unlock: `false`",
            "",
        ]
    )
