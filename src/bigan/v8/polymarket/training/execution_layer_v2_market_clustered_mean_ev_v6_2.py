"""Run the #211 market-clustered mean-EV v6.2 actionability gate."""

from __future__ import annotations

import math
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_guard_compatible_conformal_net_return_v5 import (
    _raw_target_stripped_predictions,
)
from bigan.v8.polymarket.training.execution_layer_v2_guard_compatible_direct_net_return_v4 import (
    _row_key,
)
from bigan.v8.polymarket.training.execution_layer_v2_outcome_blind_acceptance_viability import (
    _outcome_blind_acceptance_replay,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    SIDES,
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
    _write_json,
    _write_jsonl,
    _write_text,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6_fit import (
    CALIBRATION_ROLE,
    TARGET_FIELDS,
    attach_frozen_execution_compatibility,
    select_sequential_policy_rows,
)
from bigan.v8.polymarket.training.execution_layer_v2_v6_on_v5_target_free_diagnostic import (
    _normalize_v5_labeled_rows,
)

PROFILE_SCHEMA_VERSION = "bigan-v8-market-clustered-mean-ev-v6-2-profile-v1"
REPORT_SCHEMA_VERSION = "bigan-v8-market-clustered-mean-ev-v6-2-report-v1"
MANIFEST_SCHEMA_VERSION = "bigan-v8-market-clustered-mean-ev-v6-2-manifest-v1"
CALIBRATION_SCHEMA_VERSION = "bigan-v8-market-clustered-mean-ev-v6-2-calibration-v1"
CANDIDATE_NAME = "market_clustered_mean_ev_v6_2"


@dataclass(frozen=True, slots=True)
class MarketClusteredMeanEVV62Config:
    """Pinned inputs for one #211 target-free actionability run."""

    run_id: str
    output_dir: Path | str
    profile_path: Path | str
    expected_profile_sha256: str
    issue209_manifest_path: Path | str
    v5_freeze_manifest_path: Path | str
    feature_contract_path: Path | str
    collector_pause_attestation_path: Path | str
    implementation_commit: str
    candidate_freeze_created_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(self.expected_profile_sha256, name="expected_profile_sha256")
        _require_git_sha(self.implementation_commit)
        if self.candidate_freeze_created_ts <= 0:
            raise ValueError("candidate_freeze_created_ts must be positive")
        for name in (
            "output_dir",
            "profile_path",
            "issue209_manifest_path",
            "v5_freeze_manifest_path",
            "feature_contract_path",
            "collector_pause_attestation_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


def validate_market_clustered_mean_ev_v6_2_profile(profile: dict[str, Any]) -> None:
    """Reject drift from the pre-calibration #211 design."""

    lineage = dict(profile.get("source_lineage") or {})
    model = dict(profile.get("point_model") or {})
    calibration = dict(profile.get("mean_risk_calibration") or {})
    decision = dict(profile.get("decision_rule") or {})
    check = dict(profile.get("target_free_check") or {})
    collection = dict(profile.get("collection_policy") or {})
    expected_safety = _blocked_safety_fields() | {"paper_candidate_allowed": False}
    checks = {
        "schema": profile.get("schema_version") == PROFILE_SCHEMA_VERSION,
        "candidate": profile.get("candidate_name") == CANDIDATE_NAME,
        "issue": profile.get("issue_number") == 211,
        "frozen": profile.get("frozen") is True,
        "lineage": bool(lineage) and all(_is_sha256(str(value)) for value in lineage.values()),
        "model": model.get("source") == "exact_frozen_v5_direct_net_return_model"
        and model.get("model_sha256") == lineage.get("v5_model_sha256")
        and model.get("fit_market_count") == 135
        and model.get("model_parameters_or_features_mutated") is False
        and model.get("model_score_mutation_allowed") is False,
        "mean_estimand": calibration.get("selection")
        == "earliest_positive_raw_ev_guard_compatible_trade_per_market"
        and calibration.get("grouping") == "selected_side_buy_up_buy_down_only"
        and calibration.get("confidence_bound")
        == "market_bootstrap_upper_confidence_bound_of_mean_residual"
        and calibration.get("confidence_level") == 0.95
        and calibration.get("bootstrap_resample_count") == 5000
        and calibration.get("bootstrap_seed") == 21060720
        and calibration.get("minimum_selected_market_count") == 40
        and calibration.get("minimum_selected_market_count_per_side") == 20
        and calibration.get("individual_outcome_quantile_subtraction_enabled") is False
        and calibration.get("calibration_policy_pnl_computed") is False
        and calibration.get("threshold_search_enabled") is False
        and calibration.get("candidate_comparison_enabled") is False,
        "decision": decision.get("score_formula")
        == ("raw_direct_predicted_net_return - selected_side_mean_residual_upper_confidence_bound")
        and decision.get("minimum_selected_score_exclusive") == 0.0
        and decision.get("guard_compatibility_mask_before_argmax") is True
        and decision.get("full_execution_guard_unchanged") is True
        and decision.get("cost_model_unchanged") is True
        and decision.get("sizing_and_exposure_unchanged") is True,
        "target_free": check.get("selected_market_count") == 50
        and check.get("selected_sequence_start") == 237
        and check.get("selected_sequence_end") == 286
        and check.get("minimum_positive_mean_ev_lcb_unique_market_count") == 10
        and check.get("minimum_positive_mean_ev_lcb_unique_market_count_per_side") == 5
        and check.get("minimum_full_guard_accepted_unique_market_count") == 10
        and check.get("minimum_full_guard_accepted_unique_market_count_per_side") == 5
        and check.get("labels_outcomes_settlement_targets_or_pnl_opened") is False
        and check.get("result_dependent_extension_allowed") is False,
        "collection": collection.get("persistent_collector_must_be_paused_before_run") is True
        and collection.get(
            "resume_only_if_all_target_free_action_and_full_guard_support_gates_pass"
        )
        is True
        and collection.get("candidate_freeze_must_precede_resumed_collection") is True
        and collection.get("new_strictly_later_future_holdout_required_after_candidate_freeze")
        is True,
        "prohibited": profile.get("prohibited_inputs")
        == {
            "uses_204_outcomes_for_fitting": False,
            "uses_204_pnl_for_tuning": False,
            "uses_target_free_check_labels_for_tuning": False,
            "uses_current_oof_validation_or_confirmatory_pnl_for_tuning": False,
            "result_driven_model_threshold_penalty_or_gate_change_allowed": False,
        },
        "safety": profile.get("safety") == expected_safety,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("#211 profile invalid: " + ", ".join(blockers))


def run_market_clustered_mean_ev_v6_2(
    config: MarketClusteredMeanEVV62Config,
) -> dict[str, Any]:
    """Calibrate mean residual risk and score the sealed #209 target-free rows."""

    profile_path = config.profile_path.resolve()
    issue209_path = config.issue209_manifest_path.resolve()
    v5_path = config.v5_freeze_manifest_path.resolve()
    feature_contract_path = config.feature_contract_path.resolve()
    pause_path = config.collector_pause_attestation_path.resolve()
    _verify_pin(profile_path, config.expected_profile_sha256, "#211 profile")
    profile = _load_json(profile_path)
    validate_market_clustered_mean_ev_v6_2_profile(profile)
    lineage = dict(profile["source_lineage"])
    _verify_pin(issue209_path, lineage["issue209_manifest_sha256"], "#209 manifest")
    _verify_pin(v5_path, lineage["v5_freeze_manifest_sha256"], "v5 freeze manifest")
    _verify_pin(feature_contract_path, lineage["feature_contract_sha256"], "feature contract")
    pause_attestation = _load_json(pause_path)
    validate_collector_pause_attestation(pause_attestation)

    issue209 = _load_json(issue209_path)
    _validate_issue209_source(issue209, lineage=lineage)
    v5 = _load_json(v5_path)
    model_descriptor = _verified_descriptor(issue209.get("model"), "#209 v5 model")
    target_free_descriptor = _verified_descriptor(
        issue209.get("target_free_five_action_rows"), "#209 target-free action rows"
    )
    selected_descriptor = _verified_descriptor(
        issue209.get("selected_target_free_rows"), "#209 selected target-free rows"
    )
    conformal_descriptor = _verified_descriptor(
        v5.get("development_calibration_action_rows"), "v5 conformal action rows"
    )
    expected_descriptors = {
        "v5_model_sha256": model_descriptor["sha256"],
        "v5_conformal_action_rows_sha256": conformal_descriptor["sha256"],
        "target_free_five_action_rows_sha256": target_free_descriptor["sha256"],
        "target_free_selected_rows_sha256": selected_descriptor["sha256"],
    }
    for name, observed in expected_descriptors.items():
        if lineage[name] != observed:
            raise ValueError(f"#211 source lineage mismatch: {name}")
    feature_contract = _load_json(feature_contract_path)
    feature_columns = tuple(str(value) for value in feature_contract["feature_columns"])
    target_free_rows = _load_jsonl(Path(target_free_descriptor["path"]))
    selected_rows = _load_jsonl(Path(selected_descriptor["path"]))
    _validate_target_free_rows(target_free_rows, selected_rows, profile=profile)

    run_dir = config.output_dir.resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    audit = {
        "schema_version": "bigan-v8-market-clustered-mean-ev-v6-2-pre-target-audit-v1",
        "run_id": config.run_id,
        "implementation_commit": config.implementation_commit,
        "profile": _descriptor(profile_path),
        "issue209_manifest": _descriptor(issue209_path),
        "v5_freeze_manifest": _descriptor(v5_path),
        "feature_contract": _descriptor(feature_contract_path),
        "collector_pause_attestation": _descriptor(pause_path),
        "source_model": model_descriptor,
        "target_free_five_action_rows": target_free_descriptor,
        "target_free_selected_rows": selected_descriptor,
        "target_free_market_count": len(selected_rows),
        "target_free_action_row_count": len(target_free_rows),
        "target_free_feature_causality_violation_count": sum(
            int(row["max_input_ts"]) > int(row["decision_ts"]) for row in target_free_rows
        ),
        "target_free_labels_outcomes_settlement_targets_or_pnl_opened": False,
        "pre_target_access_validation_passed": True,
        **_blocked_safety_fields(),
        "paper_candidate_allowed": False,
    }
    audit["audit_id"] = canonical_json_sha256(audit)
    audit_path = run_dir / "v6_2_pre_target_access_audit.json"
    _write_json(audit_path, audit)

    conformal_rows = _normalize_v5_labeled_rows(
        _load_jsonl(Path(conformal_descriptor["path"])),
        role=CALIBRATION_ROLE,
        expected_source_roles={"confirmatory_validation"},
        feature_columns=feature_columns,
    )
    if len({str(row["market_id"]) for row in conformal_rows}) != 60:
        raise ValueError("#211 calibration role must contain exactly 60 markets")
    booster = xgb.Booster()
    booster.load_model(model_descriptor["path"])
    calibration_predictions = attach_frozen_execution_compatibility(
        _raw_target_stripped_predictions(booster, conformal_rows, feature_columns=feature_columns)
    )
    mean_risk = build_market_clustered_mean_risk_calibration(
        calibration_predictions,
        target_rows=conformal_rows,
        profile=profile,
        model_sha256=model_descriptor["sha256"],
    )
    mean_risk_path = run_dir / "v6_2_market_clustered_mean_risk_calibration.json"
    _write_json(mean_risk_path, mean_risk)
    _write_text(mean_risk_path.with_suffix(".md"), _calibration_markdown(mean_risk))

    target_free_predictions = attach_frozen_execution_compatibility(
        _raw_target_stripped_predictions(booster, target_free_rows, feature_columns=feature_columns)
    )
    scored = apply_market_clustered_mean_ev_scores(
        target_free_predictions,
        calibration_artifact=mean_risk,
    )
    static_selected = select_sequential_policy_rows(
        scored,
        score_field="mean_ev_lower_confidence_bound",
        require_positive=True,
    )
    replay = _outcome_blind_acceptance_replay(
        scored,
        entry_threshold=0.0,
        runner_up_advantage_threshold=0.0,
    )
    accepted = [row for row in replay if row["execution_guard_order_allowed"]]
    gate = build_target_free_actionability_gate(
        calibration_artifact=mean_risk,
        static_selected=static_selected,
        accepted=accepted,
        profile=profile,
    )

    calibration_predictions_path = run_dir / "v6_2_calibration_target_stripped_predictions.jsonl"
    target_free_predictions_path = run_dir / "v6_2_target_free_raw_predictions.jsonl"
    scored_path = run_dir / "v6_2_target_free_mean_ev_scored_rows.jsonl"
    selected_path = run_dir / "v6_2_target_free_positive_mean_ev_lcb_rows.jsonl"
    replay_path = run_dir / "v6_2_target_free_full_guard_replay.jsonl"
    accepted_path = run_dir / "v6_2_target_free_guard_accepted_bets.jsonl"
    _write_jsonl(calibration_predictions_path, calibration_predictions)
    _write_jsonl(target_free_predictions_path, target_free_predictions)
    _write_jsonl(scored_path, scored)
    _write_jsonl(selected_path, static_selected)
    _write_jsonl(replay_path, replay)
    _write_jsonl(accepted_path, accepted)

    report = _build_report(
        config=config,
        target_free_rows=target_free_rows,
        selected_rows=selected_rows,
        mean_risk=mean_risk,
        scored=scored,
        static_selected=static_selected,
        replay=replay,
        accepted=accepted,
        gate=gate,
    )
    report_path = run_dir / "v6_2_target_free_actionability_report.json"
    _write_json(report_path, report)
    _write_text(report_path.with_suffix(".md"), _report_markdown(report))
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        "candidate_freeze_created_ts": config.candidate_freeze_created_ts,
        "profile": _descriptor(profile_path),
        "pre_target_access_audit": _descriptor(audit_path),
        "collector_pause_attestation": _descriptor(pause_path),
        "source_issue209_manifest": _descriptor(issue209_path),
        "source_model": model_descriptor,
        "source_v5_conformal_action_rows": conformal_descriptor,
        "source_target_free_five_action_rows": target_free_descriptor,
        "source_target_free_selected_rows": selected_descriptor,
        "market_clustered_mean_risk_calibration": _descriptor(mean_risk_path),
        "calibration_target_stripped_predictions": _descriptor(calibration_predictions_path),
        "target_free_raw_predictions": _descriptor(target_free_predictions_path),
        "target_free_mean_ev_scored_rows": _descriptor(scored_path),
        "target_free_positive_mean_ev_lcb_rows": _descriptor(selected_path),
        "target_free_full_guard_replay": _descriptor(replay_path),
        "target_free_guard_accepted_bets": _descriptor(accepted_path),
        "target_free_actionability_report": _descriptor(report_path),
        "target_free_actionability_gate_passed": gate["passed"],
        "target_free_actionability_blocking_reason_codes": gate["reason_codes"],
        "research_actionability_candidate_frozen": gate["passed"],
        "collector_resume_allowed": gate["passed"],
        "future_collection_minimum_created_ts_exclusive": (
            config.candidate_freeze_created_ts if gate["passed"] else None
        ),
        "new_strictly_later_future_holdout_required": True,
        "promotion_evidence": False,
        "target_free_labels_outcomes_settlement_targets_or_pnl_opened": False,
        **_blocked_safety_fields(),
        "paper_candidate_allowed": False,
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v6_2_actionability_candidate_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "report": report,
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def build_market_clustered_mean_risk_calibration(
    calibration_predictions: list[dict[str, Any]],
    *,
    target_rows: list[dict[str, Any]],
    profile: dict[str, Any],
    model_sha256: str,
) -> dict[str, Any]:
    """Estimate side-specific uncertainty in mean residual, not individual outcomes."""

    selected = select_sequential_policy_rows(
        calibration_predictions,
        score_field="raw_direct_predicted_net_return",
        require_positive=True,
    )
    targets = {_row_key(row): float(row["target_net_pnl_per_contract"]) for row in target_rows}
    residual_rows = []
    for row in selected:
        key = _row_key(row)
        if key not in targets:
            raise ValueError("#211 selected calibration target identity is missing")
        residual_rows.append(
            {
                "market_id": str(row["market_id"]),
                "decision_ts": int(row["decision_ts"]),
                "side": str(row["side"]),
                "action": str(row["action"]),
                "action_family": str(row["action_family"]),
                "raw_prediction": float(row["raw_direct_predicted_net_return"]),
                "residual": float(row["raw_direct_predicted_net_return"]) - targets[key],
            }
        )
    calibration = dict(profile["mean_risk_calibration"])
    sides = {
        side: _mean_residual_bootstrap(
            [row["residual"] for row in residual_rows if row["side"] == side],
            confidence_level=float(calibration["confidence_level"]),
            resample_count=int(calibration["bootstrap_resample_count"]),
            seed=int(calibration["bootstrap_seed"]) + index,
            side=side,
        )
        for index, side in enumerate(SIDES)
    }
    selected_side_counts = Counter(str(row["side"]) for row in residual_rows)
    selected_action_counts = Counter(str(row["action"]) for row in residual_rows)
    minimum_total = int(calibration["minimum_selected_market_count"])
    minimum_side = int(calibration["minimum_selected_market_count_per_side"])
    checks = {
        "selected_total_support": len(residual_rows) >= minimum_total,
        "selected_side_support": all(selected_side_counts[side] >= minimum_side for side in SIDES),
        "finite_mean_residual_bounds": all(
            math.isfinite(float(sides[side]["mean_residual_upper_confidence_bound"]))
            for side in SIDES
        ),
        "maximum_one_selected_trade_per_market": len(residual_rows)
        == len({row["market_id"] for row in residual_rows}),
        "individual_outcome_quantile_subtraction_disabled": calibration[
            "individual_outcome_quantile_subtraction_enabled"
        ]
        is False,
        "no_policy_pnl_or_threshold_search": calibration["calibration_policy_pnl_computed"] is False
        and calibration["threshold_search_enabled"] is False
        and calibration["candidate_comparison_enabled"] is False,
    }
    reason_map = {
        "selected_total_support": "mean_risk_selected_calibration_total_support_failed",
        "selected_side_support": "mean_risk_selected_calibration_side_support_failed",
        "finite_mean_residual_bounds": "mean_residual_confidence_bound_invalid",
        "maximum_one_selected_trade_per_market": "mean_risk_duplicate_selected_market",
        "individual_outcome_quantile_subtraction_disabled": (
            "individual_outcome_quantile_subtraction_enabled"
        ),
        "no_policy_pnl_or_threshold_search": "calibration_policy_pnl_or_threshold_search_enabled",
    }
    reasons = [reason_map[name] for name, passed in checks.items() if not passed]
    artifact = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "frozen": True,
        "decision_time_safe_at_inference": True,
        "source_model_sha256": model_sha256,
        "estimand": "market_clustered_conditional_mean_net_return",
        "score_formula": profile["decision_rule"]["score_formula"],
        "selected_calibration_market_count": len(residual_rows),
        "selected_side_distribution": {side: selected_side_counts[side] for side in SIDES},
        "selected_action_distribution": dict(sorted(selected_action_counts.items())),
        "sides": sides,
        "calibration_gate_checks": checks,
        "calibration_gate_passed": not reasons,
        "calibration_gate_blocking_reason_codes": reasons,
        "individual_outcome_quantile_subtraction_enabled": False,
        "calibration_policy_pnl_computed": False,
        "threshold_search_enabled": False,
        "uses_204_outcomes_for_fitting": False,
        "uses_target_free_check_labels_for_tuning": False,
        **_blocked_safety_fields(),
        "paper_candidate_allowed": False,
    }
    artifact["artifact_id"] = canonical_json_sha256(artifact)
    return artifact


def apply_market_clustered_mean_ev_scores(
    predictions: list[dict[str, Any]],
    *,
    calibration_artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply frozen mean-residual uncertainty bounds to target-free rows."""

    if _find_nonempty_fields(predictions, TARGET_FIELDS):
        raise ValueError("#211 inference rows contain target fields")
    output = []
    for row in predictions:
        action = str(row["action"])
        raw = 0.0 if action == "NO_TRADE" else float(row["raw_direct_predicted_net_return"])
        if action == "NO_TRADE":
            mean_residual = upper_bound = point_ev = lower_bound = selection_score = 0.0
            source = "frozen_no_trade_zero_anchor"
        else:
            side_calibration = calibration_artifact["sides"][str(row["side"])]
            mean_residual = float(side_calibration["mean_residual"])
            upper_bound = float(side_calibration["mean_residual_upper_confidence_bound"])
            point_ev = raw - mean_residual
            lower_bound = raw - upper_bound
            selection_score = (
                lower_bound if row["guard_compatible_before_ranking"] else -1_000_000.0
            )
            source = (
                "market_clustered_mean_ev_lower_confidence_bound"
                if row["guard_compatible_before_ranking"]
                else "masked_by_unchanged_execution_compatibility"
            )
        updated = {
            **row,
            "selected_side_mean_residual": mean_residual,
            "selected_side_mean_residual_upper_confidence_bound": upper_bound,
            "bias_corrected_action_expected_net_return": point_ev,
            "mean_ev_lower_confidence_bound": lower_bound,
            "conformal_net_return_lower_bound": lower_bound,
            "action_selection_score": selection_score,
            "action_advantage_lcb_net_return": selection_score,
            "calibrated_action_expected_net_return": point_ev,
            "ranking_score_source": source,
            "raw_pairwise_rank_score": raw,
            "pairwise_group_normalized_rank_score": raw,
            "action_advantage_lcb_score_bucket": "market_clustered_mean_ev_v6_2",
            "action_advantage_lcb_estimate_source": source,
            "individual_outcome_quantile_subtraction_enabled": False,
            "target_used_as_decision_input": False,
            "outcome_fields_used_as_decision_input": False,
            "target_or_outcome_fields_used": False,
            **_blocked_safety_fields(),
            "paper_candidate_allowed": False,
        }
        updated["v6_2_prediction_row_sha256"] = canonical_json_sha256(updated)
        output.append(updated)
    return output


def build_target_free_actionability_gate(
    *,
    calibration_artifact: dict[str, Any],
    static_selected: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Require two-sided score and full-guard support before collection may resume."""

    check = dict(profile["target_free_check"])
    selected_markets = {str(row["market_id"]) for row in static_selected}
    selected_side_markets = {
        side: {
            str(row["market_id"]) for row in static_selected if str(row.get("side") or "") == side
        }
        for side in SIDES
    }
    accepted_markets = {str(row["market_id"]) for row in accepted}
    accepted_side_markets = {
        side: {
            str(row["market_id"]) for row in accepted if str(row.get("selected_side") or "") == side
        }
        for side in SIDES
    }
    checks = {
        "mean_risk_calibration_gate": calibration_artifact["calibration_gate_passed"] is True,
        "positive_mean_ev_lcb_total_support": len(selected_markets)
        >= int(check["minimum_positive_mean_ev_lcb_unique_market_count"]),
        "positive_mean_ev_lcb_side_support": all(
            len(selected_side_markets[side])
            >= int(check["minimum_positive_mean_ev_lcb_unique_market_count_per_side"])
            for side in SIDES
        ),
        "full_guard_total_support": len(accepted_markets)
        >= int(check["minimum_full_guard_accepted_unique_market_count"]),
        "full_guard_side_support": all(
            len(accepted_side_markets[side])
            >= int(check["minimum_full_guard_accepted_unique_market_count_per_side"])
            for side in SIDES
        ),
        "target_free_targets_remain_sealed": check[
            "labels_outcomes_settlement_targets_or_pnl_opened"
        ]
        is False,
    }
    reason_map = {
        "mean_risk_calibration_gate": "mean_risk_calibration_gate_failed",
        "positive_mean_ev_lcb_total_support": "positive_mean_ev_lcb_total_support_failed",
        "positive_mean_ev_lcb_side_support": "positive_mean_ev_lcb_side_support_failed",
        "full_guard_total_support": "target_free_full_guard_total_support_failed",
        "full_guard_side_support": "target_free_full_guard_side_support_failed",
        "target_free_targets_remain_sealed": "target_free_target_sealing_failed",
    }
    reasons = list(calibration_artifact["calibration_gate_blocking_reason_codes"])
    reasons.extend(reason_map[name] for name, passed in checks.items() if not passed)
    return {
        "passed": not reasons,
        "checks": checks,
        "reason_codes": sorted(set(reasons)),
        "positive_mean_ev_lcb_unique_market_count": len(selected_markets),
        "positive_mean_ev_lcb_side_market_count": {
            side: len(selected_side_markets[side]) for side in SIDES
        },
        "full_guard_accepted_unique_market_count": len(accepted_markets),
        "full_guard_accepted_side_market_count": {
            side: len(accepted_side_markets[side]) for side in SIDES
        },
    }


def validate_collector_pause_attestation(attestation: dict[str, Any]) -> None:
    """Require a completed canary boundary and unloaded persistent collector."""

    checks = {
        "schema": attestation.get("schema_version")
        == "bigan-v8-persistent-collector-pause-attestation-v1",
        "paused": attestation.get("collector_paused") is True,
        "service_unloaded": attestation.get("launchd_service_loaded") is False,
        "complete_boundary": int(attestation.get("last_completed_batch_sequence") or 0) >= 26,
        "canary": attestation.get("last_batch_canary_passed") is True,
        "sealed": attestation.get("labels_outcomes_or_pnl_opened") is False,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("collector pause attestation invalid: " + ", ".join(blockers))


def _mean_residual_bootstrap(
    residuals: list[float],
    *,
    confidence_level: float,
    resample_count: int,
    seed: int,
    side: str,
) -> dict[str, Any]:
    values = np.asarray(residuals, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return {
            "side": side,
            "market_count": 0,
            "mean_residual": float("nan"),
            "residual_standard_deviation": float("nan"),
            "mean_residual_upper_confidence_bound": float("nan"),
            "bootstrap_resample_count": resample_count,
            "bootstrap_seed": seed,
        }
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resample_count, values.size))
    bootstrap_means = values[indices].mean(axis=1)
    upper = float(np.quantile(bootstrap_means, confidence_level, method="higher"))
    return {
        "side": side,
        "market_count": int(values.size),
        "mean_residual": float(values.mean()),
        "residual_standard_deviation": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "mean_residual_upper_confidence_bound": upper,
        "confidence_level": confidence_level,
        "bootstrap_resample_count": resample_count,
        "bootstrap_seed": seed,
        "bootstrap_unit": "market_id",
    }


def _validate_issue209_source(issue209: dict[str, Any], *, lineage: dict[str, str]) -> None:
    if issue209.get("diagnostic_viability_passed") is not False:
        raise ValueError("#209 source state must remain terminal fail-closed")
    if issue209.get("promotion_evidence") is not False:
        raise ValueError("#209 source cannot be promotion evidence")
    report_descriptor = _verified_descriptor(issue209.get("viability_report"), "#209 report")
    report = _load_json(Path(report_descriptor["path"]))
    if (
        report.get("target_free_check_labels_outcomes_settlement_targets_or_pnl_opened")
        is not False
    ):
        raise ValueError("#209 target-free target sealing is invalid")
    model = _verified_descriptor(issue209.get("model"), "#209 model")
    if model["sha256"] != lineage["v5_model_sha256"]:
        raise ValueError("#209 model hash differs from #211 profile")


def _validate_target_free_rows(
    action_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    *,
    profile: dict[str, Any],
) -> None:
    check = dict(profile["target_free_check"])
    if len(selected_rows) != int(check["selected_market_count"]):
        raise ValueError("#211 target-free selected market count mismatch")
    if int(selected_rows[0]["sequence"]) != int(check["selected_sequence_start"]):
        raise ValueError("#211 target-free sequence start mismatch")
    if int(selected_rows[-1]["sequence"]) != int(check["selected_sequence_end"]):
        raise ValueError("#211 target-free sequence end mismatch")
    if len(action_rows) != len(selected_rows) * 4 * 5:
        raise ValueError("#211 target-free five-action coverage mismatch")
    if _find_nonempty_fields(action_rows, TARGET_FIELDS):
        raise ValueError("#211 target-free rows contain target fields")
    if any(int(row["max_input_ts"]) > int(row["decision_ts"]) for row in action_rows):
        raise ValueError("#211 target-free rows violate feature causality")


def _build_report(
    *,
    config: MarketClusteredMeanEVV62Config,
    target_free_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    mean_risk: dict[str, Any],
    scored: list[dict[str, Any]],
    static_selected: list[dict[str, Any]],
    replay: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    raw_positive = [
        row
        for row in scored
        if row["action"] != "NO_TRADE"
        and row["guard_compatible_before_ranking"]
        and float(row["raw_direct_predicted_net_return"]) > 0.0
    ]
    selected_actions = Counter(str(row["action"]) for row in static_selected)
    accepted_actions = Counter(str(row["executed_action"]) for row in accepted)
    blockers = Counter(
        str(reason) for row in replay for reason in row["execution_blocking_reason_codes"]
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_id": None,
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        "source_model_sha256": mean_risk["source_model_sha256"],
        "source_model_parameters_or_features_mutated": False,
        "calibration_market_count": 60,
        "selected_calibration_market_count": mean_risk["selected_calibration_market_count"],
        "selected_calibration_side_distribution": mean_risk["selected_side_distribution"],
        "mean_residual_by_side": {
            side: mean_risk["sides"][side]["mean_residual"] for side in SIDES
        },
        "mean_residual_upper_confidence_bound_by_side": {
            side: mean_risk["sides"][side]["mean_residual_upper_confidence_bound"] for side in SIDES
        },
        "individual_outcome_quantile_subtraction_enabled": False,
        "calibration_policy_pnl_computed": False,
        "target_free_market_count": len(selected_rows),
        "target_free_action_row_count": len(target_free_rows),
        "target_free_feature_causality_violation_count": sum(
            int(row["max_input_ts"]) > int(row["decision_ts"]) for row in target_free_rows
        ),
        "raw_positive_guard_compatible_trade_row_count": len(raw_positive),
        "raw_positive_guard_compatible_unique_market_count": len(
            {str(row["market_id"]) for row in raw_positive}
        ),
        "positive_mean_ev_lcb_selected_unique_market_count": gate[
            "positive_mean_ev_lcb_unique_market_count"
        ],
        "positive_mean_ev_lcb_selected_side_market_count": gate[
            "positive_mean_ev_lcb_side_market_count"
        ],
        "positive_mean_ev_lcb_selected_action_distribution": dict(sorted(selected_actions.items())),
        "full_guard_accepted_bet_count": len(accepted),
        "full_guard_accepted_unique_market_count": gate["full_guard_accepted_unique_market_count"],
        "full_guard_accepted_side_market_count": gate["full_guard_accepted_side_market_count"],
        "full_guard_accepted_action_distribution": dict(sorted(accepted_actions.items())),
        "full_guard_blocking_reason_distribution": dict(sorted(blockers.items())),
        "target_free_actionability_gate_passed": gate["passed"],
        "target_free_actionability_gate_checks": gate["checks"],
        "target_free_actionability_blocking_reason_codes": gate["reason_codes"],
        "collector_resume_allowed": gate["passed"],
        "target_free_labels_outcomes_settlement_targets_or_pnl_opened": False,
        "threshold_search_enabled": False,
        "execution_guard_cost_sizing_or_exposure_mutated": False,
        "new_strictly_later_future_holdout_required": True,
        "promotion_evidence": False,
        **_blocked_safety_fields(),
        "paper_candidate_allowed": False,
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def _calibration_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# #211 market-clustered mean-risk calibration",
            "",
            f"- gate passed: `{report['calibration_gate_passed']}`",
            f"- selected markets: `{report['selected_calibration_market_count']}`",
            f"- selected sides: `{report['selected_side_distribution']}`",
            "- individual outcome quantile subtraction: `false`",
            "- calibration policy PnL computed: `false`",
            "",
        ]
    )


def _report_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# #211 v6.2 target-free actionability",
            "",
            f"- gate passed: `{report['target_free_actionability_gate_passed']}`",
            f"- blockers: `{report['target_free_actionability_blocking_reason_codes']}`",
            "- positive mean-EV-LCB sides: "
            f"`{report['positive_mean_ev_lcb_selected_side_market_count']}`",
            f"- full-guard sides: `{report['full_guard_accepted_side_market_count']}`",
            f"- collector resume allowed: `{report['collector_resume_allowed']}`",
            "- target-free outcomes/labels/PnL opened: `false`",
            "- promotion evidence: `false`",
            "",
        ]
    )


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
