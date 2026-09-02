"""Fit, calibrate, and evaluate the preregistered #223 v6.3 candidate."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from bigan.v8.polymarket.contracts import canonical_json_sha256

FIT_PROFILE_SCHEMA_VERSION = "bigan-v8-sbc-exit-reliability-v6-3-fit-profile-v1"
MODEL_SCHEMA_VERSION = "bigan-v8-sbc-exit-reliability-v6-3-model-v1"
MODEL_REPORT_SCHEMA_VERSION = "bigan-v8-sbc-exit-reliability-v6-3-model-report-v1"
CALIBRATION_SCHEMA_VERSION = "bigan-v8-sbc-exit-reliability-v6-3-calibration-v1"
THRESHOLD_FREEZE_SCHEMA_VERSION = "bigan-v8-sbc-exit-reliability-v6-3-threshold-freeze-v1"
OOF_GATE_SCHEMA_VERSION = "bigan-v8-sbc-exit-reliability-v6-3-oof-gate-v1"
CANDIDATE_MANIFEST_SCHEMA_VERSION = "bigan-v8-sbc-exit-reliability-v6-3-candidate-v1"
CANDIDATE_NAME = "sbc_exit_reliability_v6_3"
SIDES = ("UP", "DOWN")
SBC_ACTIONS = {
    "UP": "BUY_UP_SELL_BEFORE_CLOSE",
    "DOWN": "BUY_DOWN_SELL_BEFORE_CLOSE",
}
TARGET_TOKENS = (
    "outcome",
    "settlement",
    "resolution",
    "target",
    "realized",
    "pnl",
    "oracle",
    "future_return",
)
ALLOWED_FALSE_SAFETY_DECLARATIONS = {
    "target_or_outcome_fields_used",
    "target_or_outcome_used_for_decision",
    "target_used_for_decision",
}


@dataclass(frozen=True, slots=True)
class SBCExitReliabilityV63FitConfig:
    """Pinned inputs for fit/calibration or OOF evaluation."""

    stage: Literal["fit_calibrate", "evaluate_oof"]
    run_id: str
    output_dir: Path | str
    fit_profile_path: Path | str
    expected_fit_profile_sha256: str
    audit_manifest_path: Path | str
    v6_2_historical_manifest_path: Path | str
    implementation_commit: str
    threshold_freeze_manifest_path: Path | str | None = None
    expected_threshold_freeze_manifest_sha256: str | None = None
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if self.stage not in {"fit_calibrate", "evaluate_oof"}:
            raise ValueError("unsupported v6.3 fit stage")
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(self.expected_fit_profile_sha256, "expected_fit_profile_sha256")
        _require_git_sha(self.implementation_commit)
        for name in (
            "output_dir",
            "fit_profile_path",
            "audit_manifest_path",
            "v6_2_historical_manifest_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if self.threshold_freeze_manifest_path is not None:
            object.__setattr__(
                self,
                "threshold_freeze_manifest_path",
                Path(self.threshold_freeze_manifest_path),
            )
        if self.stage == "evaluate_oof":
            if self.threshold_freeze_manifest_path is None:
                raise ValueError("threshold freeze manifest is required")
            _require_sha256(
                str(self.expected_threshold_freeze_manifest_sha256 or ""),
                "expected_threshold_freeze_manifest_sha256",
            )


def validate_sbc_exit_reliability_v6_3_fit_profile(profile: dict[str, Any]) -> None:
    """Validate the immutable model, calibration, and OOF gate contract."""

    roles = dict(profile.get("roles") or {})
    model = dict(profile.get("model") or {})
    stability = dict(profile.get("coefficient_stability") or {})
    calibration = dict(profile.get("calibration") or {})
    gate = dict(profile.get("historical_oof_side_only_gate") or {})
    access = dict(profile.get("access_sequence") or {})
    lineage = dict(profile.get("source_lineage") or {})
    checks = {
        "schema": profile.get("schema_version") == FIT_PROFILE_SCHEMA_VERSION,
        "issue": profile.get("issue_number") == 223,
        "candidate": profile.get("candidate_name") == CANDIDATE_NAME,
        "frozen": profile.get("frozen") is True,
        "lineage": len(lineage) == 6
        and all(_is_sha256(str(value)) for value in lineage.values()),
        "roles": roles
        == {
            "model_fit": "development_train",
            "model_fit_market_count": 89,
            "threshold_calibration": "development_calibration",
            "threshold_calibration_market_count": 45,
            "historical_oof_evaluation": "confirmatory_validation",
            "historical_oof_market_count": 60,
            "market_disjoint": True,
            "chronological": True,
        },
        "model": model.get("family") == "l2_regularized_binary_logistic_regression"
        and len(model.get("feature_columns") or []) == 24
        and float(model.get("l2_penalty") or 0.0) > 0.0
        and int(model.get("gradient_descent_iterations") or 0) >= 1000
        and model.get("hyperparameter_search_enabled") is False
        and model.get("validation_labels_used_for_model_fit") is False
        and model.get("oof_labels_used_for_model_fit") is False,
        "stability": stability.get("method")
        == "market_grouped_bootstrap_fixed_standardization"
        and int(stability.get("bootstrap_resample_count") or 0) >= 20
        and 0.5
        < float(stability.get("minimum_median_sign_agreement") or 0.0)
        <= 1.0,
        "calibration": calibration.get("threshold_search_uses_calibration_labels_only")
        is True
        and calibration.get("threshold_search_uses_oof_labels") is False
        and calibration.get("threshold_search_uses_pnl") is False
        and int(calibration.get("minimum_selected_unique_market_count_per_side") or 0)
        > 0
        and float(calibration.get("minimum_market_bootstrap_precision_lcb") or 0.0)
        > 0.5,
        "gate": gate.get("aggregation") == "buy_up_buy_down_side_only"
        and gate.get("action_and_family_metrics_diagnostic_only") is True
        and gate.get("result_dependent_rerun_allowed") is False
        and int(gate.get("minimum_guard_accepted_unique_market_count_per_side") or 0)
        > 0,
        "access": access
        == {
            "fit_stage_loads_train_and_calibration_labels_only": True,
            "threshold_freeze_manifest_required_before_oof_evaluation": True,
            "oof_support_audit_pre_access_acknowledged": True,
            "oof_support_audit_used_for_model_or_threshold_tuning": False,
            "fully_blinded_historical_oof_claimed": False,
            "strictly_future_unseen_holdout_required_for_promotion": True,
        },
        "safety": profile.get("safety") == _blocked_safety_fields(),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("#223 fit profile invalid: " + ", ".join(blockers))


def run_sbc_exit_reliability_v6_3_fit(
    config: SBCExitReliabilityV63FitConfig,
) -> dict[str, Any]:
    """Run one target-isolated fit/calibration or frozen OOF evaluation stage."""

    profile_path = config.fit_profile_path.resolve()
    audit_path = config.audit_manifest_path.resolve()
    historical_path = config.v6_2_historical_manifest_path.resolve()
    _verify_pin(profile_path, config.expected_fit_profile_sha256, "#223 fit profile")
    profile = _load_json(profile_path)
    validate_sbc_exit_reliability_v6_3_fit_profile(profile)
    lineage = profile["source_lineage"]
    _verify_pin(audit_path, lineage["audit_manifest_sha256"], "#223 audit manifest")
    _verify_pin(
        historical_path,
        lineage["v6_2_historical_manifest_sha256"],
        "v6.2 historical manifest",
    )
    audit = _load_json(audit_path)
    historical = _load_json(historical_path)
    source = _validate_sources(audit, historical=historical, profile=profile)
    run_dir = config.output_dir.resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    if config.stage == "fit_calibrate":
        return _run_fit_calibrate(
            config=config,
            profile=profile,
            profile_path=profile_path,
            audit_path=audit_path,
            historical_path=historical_path,
            source=source,
            run_dir=run_dir,
        )
    return _run_oof_evaluation(
        config=config,
        profile=profile,
        profile_path=profile_path,
        audit_path=audit_path,
        historical_path=historical_path,
        source=source,
        run_dir=run_dir,
    )


def fit_regularized_logistic_exit_model(
    rows: list[dict[str, Any]], *, profile: dict[str, Any]
) -> dict[str, Any]:
    """Fit deterministic standardized L2 logistic regression."""

    model_config = profile["model"]
    columns = tuple(str(value) for value in model_config["feature_columns"])
    matrix = np.asarray(
        [[float(row["features"][name]) for name in columns] for row in rows],
        dtype=np.float64,
    )
    targets = np.asarray([int(row["target"]) for row in rows], dtype=np.float64)
    means = np.mean(matrix, axis=0)
    scales = np.std(matrix, axis=0)
    scales = np.where(scales < 1e-12, 1.0, scales)
    standardized = _standardize(matrix, means=means, scales=scales, profile=profile)
    weights, intercept = _fit_logistic_weights(
        standardized,
        targets,
        l2=float(model_config["l2_penalty"]),
        iterations=int(model_config["gradient_descent_iterations"]),
        learning_rate=float(model_config["learning_rate"]),
        coefficient_bound=float(model_config["coefficient_absolute_bound"]),
    )
    probabilities = _sigmoid(standardized @ weights + intercept)
    artifact = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "model_family": model_config["family"],
        "target": model_config["target"],
        "feature_columns": list(columns),
        "feature_means": means.tolist(),
        "feature_scales": scales.tolist(),
        "coefficients": weights.tolist(),
        "intercept": float(intercept),
        "l2_penalty": float(model_config["l2_penalty"]),
        "gradient_descent_iterations": int(model_config["gradient_descent_iterations"]),
        "learning_rate": float(model_config["learning_rate"]),
        "coefficient_absolute_bound": float(model_config["coefficient_absolute_bound"]),
        "fit_row_count": len(rows),
        "fit_market_count": len({str(row["market_id"]) for row in rows}),
        "fit_target_prevalence": float(np.mean(targets)),
        "fit_log_loss": _log_loss(targets, probabilities),
        "fit_brier_score": _brier(targets, probabilities),
        "coefficients_finite": bool(np.all(np.isfinite(weights)) and math.isfinite(intercept)),
        "coefficients_within_bound": bool(
            np.max(np.abs(weights)) <= float(model_config["coefficient_absolute_bound"])
            and abs(intercept) <= float(model_config["coefficient_absolute_bound"])
        ),
        "decision_time_inputs_only": True,
        "outcome_settlement_pnl_or_future_fields_in_model_inputs": False,
        **_blocked_safety_fields(),
    }
    artifact["model_artifact_id"] = canonical_json_sha256(artifact)
    return artifact


def apply_exit_reliability_model(
    rows: list[dict[str, Any]], *, model: dict[str, Any], profile: dict[str, Any]
) -> list[dict[str, Any]]:
    """Score rows without exposing labels to inference."""

    columns = tuple(str(value) for value in model["feature_columns"])
    matrix = np.asarray(
        [[float(row["features"][name]) for name in columns] for row in rows],
        dtype=np.float64,
    )
    standardized = _standardize(
        matrix,
        means=np.asarray(model["feature_means"], dtype=np.float64),
        scales=np.asarray(model["feature_scales"], dtype=np.float64),
        profile=profile,
    )
    probabilities = _sigmoid(
        standardized @ np.asarray(model["coefficients"], dtype=np.float64)
        + float(model["intercept"])
    )
    return [
        {
            **_strip_target_fields(row),
            "exit_reliability_probability": float(probability),
            "exit_reliability_score_source": "frozen_v6_3_regularized_logistic_model",
            "target_used_for_inference": False,
            **_blocked_safety_fields(),
        }
        for row, probability in zip(rows, probabilities, strict=True)
    ]


def build_exit_reliability_calibration(
    calibration_rows: list[dict[str, Any]],
    scored_rows: list[dict[str, Any]],
    *,
    model: dict[str, Any],
    profile: dict[str, Any],
    stability: dict[str, Any],
) -> dict[str, Any]:
    """Select a reliability threshold from calibration labels only."""

    if len(calibration_rows) != len(scored_rows):
        raise ValueError("calibration score/target row count mismatch")
    targets = np.asarray([int(row["target"]) for row in calibration_rows], dtype=np.float64)
    probabilities = np.asarray(
        [float(row["exit_reliability_probability"]) for row in scored_rows],
        dtype=np.float64,
    )
    constant = np.full_like(targets, float(model["fit_target_prevalence"]))
    config = profile["calibration"]
    threshold_rows = [
        _threshold_metrics(
            threshold=float(threshold),
            calibration_rows=calibration_rows,
            probabilities=probabilities,
            profile=profile,
        )
        for threshold in config["threshold_candidates"]
    ]
    passing = [row for row in threshold_rows if row["threshold_gate_passed"]]
    ranked = sorted(
        passing or threshold_rows,
        key=lambda row: (
            float(row["market_bootstrap_precision_lcb"]),
            int(row["selected_row_count"]),
            float(row["threshold"]),
        ),
        reverse=True,
    )
    selected = ranked[0]
    brier = _brier(targets, probabilities)
    baseline_brier = _brier(targets, constant)
    auc = _roc_auc(targets, probabilities)
    checks = {
        "brier_improvement": baseline_brier - brier
        > float(config["brier_improvement_over_constant_minimum_exclusive"]),
        "roc_auc": auc >= float(config["minimum_roc_auc"]),
        "threshold_available": bool(passing),
        "coefficient_stability": stability["coefficient_stability_gate_passed"],
        "model_coefficients": model["coefficients_finite"]
        and model["coefficients_within_bound"],
    }
    reasons = [f"{name}_gate_failed" for name, passed in checks.items() if not passed]
    artifact = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "model_artifact_id": model["model_artifact_id"],
        "calibration_row_count": len(calibration_rows),
        "calibration_market_count": len(
            {str(row["market_id"]) for row in calibration_rows}
        ),
        "calibration_target_prevalence": float(np.mean(targets)),
        "calibration_brier_score": brier,
        "constant_baseline_brier_score": baseline_brier,
        "brier_improvement_over_constant": baseline_brier - brier,
        "calibration_roc_auc": auc,
        "threshold_candidates": threshold_rows,
        "selected_threshold": float(selected["threshold"]),
        "selected_threshold_metrics": selected,
        "calibrated_exit_availability_precision_lcb": float(
            selected["market_bootstrap_precision_lcb"]
        ),
        "calibration_gate_checks": checks,
        "calibration_gate_passed": all(checks.values()),
        "calibration_gate_reason_codes": reasons,
        "threshold_search_uses_calibration_labels_only": True,
        "threshold_search_uses_oof_labels": False,
        "threshold_search_uses_pnl": False,
        "oof_labels_loaded_by_fit_stage": False,
        "oof_support_audit_pre_access_acknowledged": True,
        "fully_blinded_historical_oof_claimed": False,
        "coefficient_stability": stability,
        **_blocked_safety_fields(),
    }
    artifact["calibration_artifact_id"] = canonical_json_sha256(artifact)
    return artifact


def build_v6_3_side_only_oof_gate(
    evaluation_rows: list[dict[str, Any]], *, profile: dict[str, Any]
) -> dict[str, Any]:
    """Build the one-shot BUY_UP/BUY_DOWN post-cost OOF gate."""

    config = profile["historical_oof_side_only_gate"]
    accepted = [row for row in evaluation_rows if row["v6_3_guard_order_allowed"]]
    baseline = [row for row in evaluation_rows if row["v6_2_guard_order_allowed"]]
    pnl_by_side = {
        side: sum(
            float(row["v6_3_accepted_bet_net_pnl"])
            for row in accepted
            if row["selected_side"] == side
        )
        for side in SIDES
    }
    accepted_market_ids = {str(row["market_id"]) for row in accepted}
    accepted_markets_by_side = {
        side: len(
            {
                str(row["market_id"])
                for row in accepted
                if row["selected_side"] == side
            }
        )
        for side in SIDES
    }
    total_pnl = sum(float(row["v6_3_accepted_bet_net_pnl"]) for row in accepted)
    baseline_pnl = sum(float(row["v6_2_accepted_bet_net_pnl"]) for row in baseline)
    market_pnls = defaultdict(float)
    for row in accepted:
        market_pnls[str(row["market_id"])] += float(row["v6_3_accepted_bet_net_pnl"])
    largest_winner = max(market_pnls.values(), default=0.0)
    bootstrap = _market_pnl_bootstrap(
        evaluation_rows,
        samples=int(config["bootstrap_resample_count"]),
        confidence_level=float(config["bootstrap_confidence_level"]),
        seed=int(config["bootstrap_seed"]),
    )
    checks = {
        "total_market_support": len(accepted_market_ids)
        >= int(config["minimum_guard_accepted_unique_market_count"]),
        "side_market_support": all(
            accepted_markets_by_side[side]
            >= int(config["minimum_guard_accepted_unique_market_count_per_side"])
            for side in SIDES
        ),
        "candidate_total_pnl": total_pnl
        > float(config["candidate_total_post_cost_pnl_minimum_exclusive"]),
        "candidate_each_side_pnl": all(
            pnl_by_side[side]
            > float(config["candidate_pnl_each_side_minimum_exclusive"])
            for side in SIDES
        ),
        "candidate_minus_v6_2_pnl": total_pnl - baseline_pnl
        > float(config["candidate_minus_v6_2_total_pnl_minimum_exclusive"]),
        "market_bootstrap_lcb": bootstrap["lower"]
        > float(config["market_bootstrap_total_pnl_lcb_minimum_exclusive"]),
        "largest_winner_removed": total_pnl - max(0.0, largest_winner)
        > float(config["largest_winner_removed_pnl_minimum_exclusive"]),
    }
    reasons = [f"{name}_gate_failed" for name, passed in checks.items() if not passed]
    report = {
        "schema_version": OOF_GATE_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "historical_oof_market_count": len(
            {str(row["market_id"]) for row in evaluation_rows}
        ),
        "v6_2_guard_accepted_bet_count": len(baseline),
        "v6_3_guard_accepted_bet_count": len(accepted),
        "v6_3_guard_accepted_unique_market_count": len(accepted_market_ids),
        "v6_3_guard_accepted_unique_market_count_by_side": accepted_markets_by_side,
        "v6_2_side_only_post_cost_pnl": baseline_pnl,
        "v6_3_side_only_post_cost_pnl": total_pnl,
        "v6_3_minus_v6_2_side_only_post_cost_pnl": total_pnl - baseline_pnl,
        "v6_3_post_cost_pnl_by_side": pnl_by_side,
        "market_bootstrap_total_pnl_interval": bootstrap,
        "largest_winner_market_pnl": largest_winner,
        "largest_winner_removed_pnl": total_pnl - max(0.0, largest_winner),
        "side_only_gate_checks": checks,
        "historical_side_only_oof_gate_passed": all(checks.values()),
        "historical_side_only_oof_gate_reason_codes": reasons,
        "primary_pnl_aggregation": "buy_up_buy_down_side_only",
        "action_and_family_metrics_diagnostic_only": True,
        "oof_support_audit_pre_access_acknowledged": True,
        "fully_blinded_historical_oof_claimed": False,
        "future_unseen_holdout_required": True,
        **_blocked_safety_fields(),
    }
    report["oof_gate_report_id"] = canonical_json_sha256(report)
    return report


def _run_fit_calibrate(
    *,
    config: SBCExitReliabilityV63FitConfig,
    profile: dict[str, Any],
    profile_path: Path,
    audit_path: Path,
    historical_path: Path,
    source: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    role_rows = source["role_rows"]
    replay_rows = source["target_free_replay_rows"]
    train_rows = _materialize_rows(
        role_rows,
        replay_rows=replay_rows,
        roles={"development_train"},
        include_targets=True,
    )
    calibration_rows = _materialize_rows(
        role_rows,
        replay_rows=replay_rows,
        roles={"development_calibration"},
        include_targets=True,
    )
    if {str(row["market_id"]) for row in train_rows} & {
        str(row["market_id"]) for row in calibration_rows
    }:
        raise ValueError("fit and calibration markets overlap")
    model = fit_regularized_logistic_exit_model(train_rows, profile=profile)
    stability = _coefficient_stability(train_rows, model=model, profile=profile)
    calibration_scored = apply_exit_reliability_model(
        calibration_rows, model=model, profile=profile
    )
    calibration = build_exit_reliability_calibration(
        calibration_rows,
        calibration_scored,
        model=model,
        profile=profile,
        stability=stability,
    )
    model_path = run_dir / "sbc_exit_reliability_v6_3_model.json"
    model_report_path = run_dir / "sbc_exit_reliability_model_report.json"
    calibration_path = run_dir / "sbc_exit_reliability_calibration_report.json"
    _write_json(model_path, model)
    model_report = {
        "schema_version": MODEL_REPORT_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "model": _descriptor(model_path),
        "fit_market_count": model["fit_market_count"],
        "fit_row_count": model["fit_row_count"],
        "fit_target_prevalence": model["fit_target_prevalence"],
        "fit_log_loss": model["fit_log_loss"],
        "fit_brier_score": model["fit_brier_score"],
        "coefficients_finite": model["coefficients_finite"],
        "coefficients_within_bound": model["coefficients_within_bound"],
        "coefficient_stability": stability,
        "validation_or_oof_labels_used_for_fit": False,
        "issue_221_rows_used": False,
        **_blocked_safety_fields(),
    }
    model_report["model_report_id"] = canonical_json_sha256(model_report)
    _write_json(model_report_path, model_report)
    _write_json(calibration_path, calibration)
    _write_text(
        run_dir / "sbc_exit_reliability_model_report.md",
        _model_markdown(model_report),
    )
    _write_text(
        run_dir / "sbc_exit_reliability_calibration_report.md",
        _calibration_markdown(calibration),
    )
    freeze = {
        "schema_version": THRESHOLD_FREEZE_SCHEMA_VERSION,
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        "fit_profile": _descriptor(profile_path),
        "audit_manifest": _descriptor(audit_path),
        "v6_2_historical_manifest": _descriptor(historical_path),
        "model": _descriptor(model_path),
        "model_report": _descriptor(model_report_path),
        "calibration_report": _descriptor(calibration_path),
        "selected_exit_reliability_threshold": calibration["selected_threshold"],
        "calibrated_exit_availability_precision_lcb": calibration[
            "calibrated_exit_availability_precision_lcb"
        ],
        "calibration_gate_passed": calibration["calibration_gate_passed"],
        "calibration_gate_reason_codes": calibration["calibration_gate_reason_codes"],
        "threshold_frozen_before_oof_evaluation": True,
        "oof_labels_loaded_by_fit_stage": False,
        "oof_support_audit_pre_access_acknowledged": True,
        "fully_blinded_historical_oof_claimed": False,
        "future_unseen_holdout_required": True,
        "oof_evaluation_allowed": calibration["calibration_gate_passed"],
        **_blocked_safety_fields(),
    }
    freeze["threshold_freeze_id"] = canonical_json_sha256(freeze)
    freeze_path = run_dir / "v6_3_exit_reliability_threshold_freeze_manifest.json"
    _write_json(freeze_path, freeze)
    return {
        "run_dir": run_dir,
        "threshold_freeze_manifest_path": freeze_path,
        "threshold_freeze_manifest_sha256": _sha256_file(freeze_path),
        "model_report": model_report,
        "calibration": calibration,
        "freeze": freeze,
    }


def _run_oof_evaluation(
    *,
    config: SBCExitReliabilityV63FitConfig,
    profile: dict[str, Any],
    profile_path: Path,
    audit_path: Path,
    historical_path: Path,
    source: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    freeze_path = Path(config.threshold_freeze_manifest_path).resolve()  # type: ignore[arg-type]
    _verify_pin(
        freeze_path,
        str(config.expected_threshold_freeze_manifest_sha256),
        "v6.3 threshold freeze manifest",
    )
    freeze = _load_json(freeze_path)
    _validate_threshold_freeze(
        freeze,
        profile_path=profile_path,
        audit_path=audit_path,
        historical_path=historical_path,
    )
    model = _load_json(Path(_verified_descriptor(freeze["model"], "v6.3 model")["path"]))
    calibration = _load_json(
        Path(_verified_descriptor(freeze["calibration_report"], "calibration report")["path"])
    )
    oof_rows = _materialize_rows(
        source["role_rows"],
        replay_rows=source["target_free_replay_rows"],
        roles={"confirmatory_validation"},
        include_targets=True,
    )
    scored = apply_exit_reliability_model(oof_rows, model=model, profile=profile)
    score_map = {
        (str(row["market_id"]), int(row["decision_ts"]), str(row["action"])): row
        for row in scored
    }
    decision_rows = []
    evaluation_rows = []
    threshold = float(calibration["selected_threshold"])
    precision_lcb = float(calibration["calibrated_exit_availability_precision_lcb"])
    oof_market_ids = {
        str(row["market_id"])
        for row in source["role_rows"]
        if str(row["role"]) == "confirmatory_validation"
    }
    eval_by_key = {
        (str(row["market_id"]), int(row["decision_ts"])): row
        for row in source["historical_evaluation_rows"]
        if str(row["market_id"]) in oof_market_ids
    }
    oof_replay_rows = select_replay_rows_for_role(
        source["target_free_replay_rows"],
        role_rows=source["role_rows"],
        role="confirmatory_validation",
    )
    for replay in oof_replay_rows:
        key = (str(replay["market_id"]), int(replay["decision_ts"]))
        action = str(replay["executed_action"])
        baseline_allowed = replay.get("execution_guard_order_allowed") is True
        score = score_map.get((*key, action)) if action in SBC_ACTIONS.values() else None
        probability = float(score["exit_reliability_probability"]) if score else 0.0
        reliability_passed = bool(score is not None and probability >= threshold)
        candidate_allowed = baseline_allowed and reliability_passed
        decision = {
            **_strip_target_fields(replay),
            "original_v6_2_action": action,
            "v6_2_guard_order_allowed": baseline_allowed,
            "exit_reliability_probability": probability if score else None,
            "exit_reliability_threshold": threshold,
            "calibrated_exit_availability_precision_lcb": precision_lcb,
            "exit_reliability_gate_passed": reliability_passed,
            "v6_3_executed_action": action if candidate_allowed else "NO_TRADE",
            "v6_3_guard_order_allowed": candidate_allowed,
            "v6_2_source_score_mutated": False,
            "execution_guard_mutated": False,
            "target_used_for_decision": False,
            **_blocked_safety_fields(),
        }
        decision_rows.append(decision)
        historical_eval = eval_by_key.get(key)
        if historical_eval is None:
            raise ValueError("OOF historical evaluation row missing")
        baseline_pnl = float(historical_eval["accepted_bet_net_pnl"])
        evaluation_rows.append(
            {
                **decision,
                "selected_side": str(historical_eval["selected_side"]),
                "target_net_pnl_per_contract": float(
                    historical_eval["target_net_pnl_per_contract"]
                ),
                "v6_2_accepted_bet_net_pnl": baseline_pnl if baseline_allowed else 0.0,
                "v6_3_accepted_bet_net_pnl": baseline_pnl if candidate_allowed else 0.0,
                "target_joined_after_v6_3_decision_freeze": True,
                "target_used_for_decision": False,
            }
        )
    report = build_v6_3_side_only_oof_gate(evaluation_rows, profile=profile)
    decision_path = run_dir / "v6_3_target_free_oof_decision_rows.jsonl"
    evaluation_path = run_dir / "v6_3_oof_evaluation_rows.jsonl"
    report_path = run_dir / "v6_3_side_only_oof_pnl_gate_report.json"
    _write_jsonl(decision_path, decision_rows)
    _write_jsonl(evaluation_path, evaluation_rows)
    _write_json(report_path, report)
    _write_text(
        run_dir / "v6_3_side_only_oof_pnl_gate_report.md", _oof_markdown(report)
    )
    manifest = {
        "schema_version": CANDIDATE_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        "fit_profile": _descriptor(profile_path),
        "audit_manifest": _descriptor(audit_path),
        "v6_2_historical_manifest": _descriptor(historical_path),
        "threshold_freeze_manifest": _descriptor(freeze_path),
        "target_free_oof_decision_rows": _descriptor(decision_path),
        "oof_evaluation_rows": _descriptor(evaluation_path),
        "side_only_oof_gate_report": _descriptor(report_path),
        "historical_development_gate_passed": report[
            "historical_side_only_oof_gate_passed"
        ],
        "historical_development_gate_reason_codes": report[
            "historical_side_only_oof_gate_reason_codes"
        ],
        "future_candidate_freeze_step_allowed": report[
            "historical_side_only_oof_gate_passed"
        ],
        "future_unseen_holdout_required": True,
        "historical_oof_promotion_evidence": False,
        "oof_support_audit_pre_access_acknowledged": True,
        "fully_blinded_historical_oof_claimed": False,
        **_blocked_safety_fields(),
    }
    manifest["candidate_manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v6_3_candidate_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "candidate_manifest_path": manifest_path,
        "candidate_manifest_sha256": _sha256_file(manifest_path),
        "report": report,
        "manifest": manifest,
    }


def select_replay_rows_for_role(
    replay_rows: list[dict[str, Any]],
    *,
    role_rows: list[dict[str, Any]],
    role: str,
) -> list[dict[str, Any]]:
    """Select target-free replay rows by the independently frozen market lineage."""

    market_ids = {
        str(row["market_id"]) for row in role_rows if str(row["role"]) == role
    }
    if not market_ids:
        raise ValueError(f"no markets found for frozen lineage role: {role}")
    selected = [
        row for row in replay_rows if str(row.get("market_id")) in market_ids
    ]
    observed_market_ids = {str(row["market_id"]) for row in selected}
    if observed_market_ids != market_ids:
        missing = sorted(market_ids - observed_market_ids)
        raise ValueError(
            f"target-free replay missing frozen lineage markets for {role}: {missing}"
        )
    return sorted(
        selected,
        key=lambda row: (
            int(row["decision_ts"]),
            str(row["market_id"]),
            int(row.get("decision_index") or 0),
        ),
    )


def _validate_sources(
    audit: dict[str, Any], *, historical: dict[str, Any], profile: dict[str, Any]
) -> dict[str, Any]:
    if audit.get("audit_gate_passed") is not True or audit.get("fit_allowed") is not True:
        raise ValueError("#223 audit did not authorize fit")
    lineage_descriptor = _verified_descriptor(
        audit.get("pre_target_access_lineage_manifest"), "pre-target lineage manifest"
    )
    if lineage_descriptor["sha256"] != profile["source_lineage"][
        "pre_target_lineage_manifest_sha256"
    ]:
        raise ValueError("pre-target lineage hash mismatch")
    lineage = _load_json(Path(lineage_descriptor["path"]))
    role_rows_descriptor = _verified_descriptor(lineage.get("lineage_rows"), "lineage rows")
    role_rows = _load_jsonl(Path(role_rows_descriptor["path"]))
    eligible = [row for row in role_rows if row.get("eligible_for_exit_reliability") is True]
    expected_counts = profile["roles"]
    observed = Counter(str(row["role"]) for row in eligible)
    for role, field in (
        ("development_train", "model_fit_market_count"),
        ("development_calibration", "threshold_calibration_market_count"),
        ("confirmatory_validation", "historical_oof_market_count"),
    ):
        if observed[role] != int(expected_counts[field]):
            raise ValueError(f"#223 role count mismatch: {role}")
    replay_descriptor = _verified_descriptor(
        historical.get("candidate_target_free_guard_replay"),
        "v6.2 target-free guard replay",
    )
    evaluation_descriptor = _verified_descriptor(
        historical.get("candidate_historical_evaluation_rows"),
        "v6.2 historical evaluation rows",
    )
    if replay_descriptor["sha256"] != profile["source_lineage"][
        "v6_2_target_free_guard_replay_sha256"
    ]:
        raise ValueError("v6.2 target-free replay hash mismatch")
    if evaluation_descriptor["sha256"] != profile["source_lineage"][
        "v6_2_historical_evaluation_rows_sha256"
    ]:
        raise ValueError("v6.2 historical evaluation hash mismatch")
    replay_rows = _load_jsonl(Path(replay_descriptor["path"]))
    if _find_forbidden_fields(replay_rows):
        raise ValueError("target/outcome field found in v6.2 target-free replay")
    return {
        "lineage": lineage,
        "role_rows": eligible,
        "target_free_replay_rows": replay_rows,
        "historical_evaluation_rows": _load_jsonl(Path(evaluation_descriptor["path"])),
    }


def _materialize_rows(
    role_rows: list[dict[str, Any]],
    *,
    replay_rows: list[dict[str, Any]],
    roles: set[str],
    include_targets: bool,
) -> list[dict[str, Any]]:
    replay_map = {
        (str(row["market_id"]), int(row["decision_ts"])): row for row in replay_rows
    }
    exposure = _pre_entry_exposure(replay_rows)
    output = []
    for source in role_rows:
        if str(source["role"]) not in roles:
            continue
        features = {
            (str(row["market_id"]), int(row["decision_ts"])): row
            for row in _load_jsonl(Path(_verified_descriptor(source["feature_rows"], "features")["path"]))
        }
        labels = _load_jsonl(Path(_verified_descriptor(source["label_rows"], "labels")["path"]))
        for label in labels:
            action = str(label.get("action") or "")
            if action not in SBC_ACTIONS.values():
                continue
            key = (str(label["market_id"]), int(label["decision_ts"]))
            feature = features.get(key)
            replay = replay_map.get(key)
            if feature is None or replay is None:
                raise ValueError("feature/replay identity missing for exit reliability row")
            side = "UP" if action == SBC_ACTIONS["UP"] else "DOWN"
            ranking = {
                str(row["action"]): row for row in replay["full_five_action_ranking"]
            }
            action_rank = ranking[action]
            other_scores = [
                float(row["action_advantage_lcb_net_return"])
                for name, row in ranking.items()
                if name != action
            ]
            state = exposure[key]
            values = _feature_values(
                feature=dict(feature["features"]),
                replay=replay,
                action_rank=action_rank,
                action_score_margin=float(action_rank["action_advantage_lcb_net_return"])
                - max(other_scores),
                side=side,
                state=state,
            )
            row = {
                "market_id": key[0],
                "decision_ts": key[1],
                "role": str(source["role"]),
                "side": side,
                "action": action,
                "features": values,
                "max_input_ts": int(feature["max_input_ts"]),
                "target": int(
                    label.get("sell_before_close_execution_class")
                    == "realizable_sell_before_close"
                    and label.get("label_uses_executable_exit_path") is True
                ),
            }
            if int(row["max_input_ts"]) > int(row["decision_ts"]):
                raise ValueError("exit reliability feature causality violation")
            if not include_targets:
                row.pop("target")
            output.append(row)
    return sorted(output, key=lambda row: (row["decision_ts"], row["market_id"], row["action"]))


def _feature_values(
    *,
    feature: dict[str, Any],
    replay: dict[str, Any],
    action_rank: dict[str, Any],
    action_score_margin: float,
    side: str,
    state: dict[str, Any],
) -> dict[str, float]:
    prefix = side.lower()
    execution_price = float(feature[f"{prefix}_ask"])
    selected_probability = float(replay["p_up"] if side == "UP" else replay["p_down"])
    return {
        "side_is_up": float(side == "UP"),
        "execution_price": execution_price,
        "current_bid": float(feature[f"{prefix}_bid"]),
        "spread_bps": float(feature[f"{prefix}_spread_bps"]),
        "book_staleness_ms": float(feature[f"{prefix}_book_staleness_ms"]),
        "queue_fill_probability_proxy": float(
            feature[f"{prefix}_queue_fill_probability_proxy"]
        ),
        "liquidity_depth_log1p": math.log1p(
            max(0.0, float(feature[f"{prefix}_liquidity_depth"]))
        ),
        "executable_ask_notional_log1p": math.log1p(
            max(0.0, float(feature[f"{prefix}_executable_ask_notional"]))
        ),
        "executable_bid_notional_log1p": math.log1p(
            max(0.0, float(feature[f"{prefix}_executable_bid_notional"]))
        ),
        "time_to_close_seconds": float(feature["time_to_close_seconds"]),
        "recent_book_update_count_1m": float(
            feature[f"{prefix}_recent_book_update_count_1m"]
        ),
        "recent_bid_depth_volatility_1m": float(
            feature[f"{prefix}_recent_bid_depth_volatility_1m"]
        ),
        "recent_spread_stability_1m": float(
            feature[f"{prefix}_recent_spread_stability_1m"]
        ),
        "combined_spread_bps": float(feature["combined_spread_bps"]),
        "liquidity_imbalance": float(feature["liquidity_imbalance"]),
        "btc_return_30s": float(feature["btc_return_30s"]),
        "btc_return_1m": float(feature["btc_return_1m"]),
        "reference_price_to_beat_distance_at_decision": float(
            feature["reference_price_to_beat_distance_at_decision"]
        ),
        "canonical_v6_2_score": float(action_rank["action_advantage_lcb_net_return"]),
        "action_score_margin": action_score_margin,
        "selected_side_probability": selected_probability,
        "pre_entry_market_exposure": float(state["exposure"]),
        "same_side_prior_entry": float(state["last_side"] == side),
        "side_flip_prior_entry": float(
            state["last_side"] not in {"NONE", side}
        ),
    }


def _pre_entry_exposure(replay_rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    state_by_market: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"exposure": 0.0, "last_side": "NONE"}
    )
    output = {}
    for row in sorted(replay_rows, key=lambda item: (int(item["decision_ts"]), str(item["market_id"]))):
        market_id = str(row["market_id"])
        state = state_by_market[market_id]
        output[(market_id, int(row["decision_ts"]))] = dict(state)
        if row.get("execution_guard_order_allowed") is True:
            state["exposure"] += float(row.get("proposed_order_size") or 0.0)
            state["last_side"] = str(row.get("selected_side") or "NONE")
    return output


def _coefficient_stability(
    rows: list[dict[str, Any]], *, model: dict[str, Any], profile: dict[str, Any]
) -> dict[str, Any]:
    config = profile["coefficient_stability"]
    columns = tuple(model["feature_columns"])
    matrix = np.asarray(
        [[float(row["features"][name]) for name in columns] for row in rows],
        dtype=np.float64,
    )
    targets = np.asarray([int(row["target"]) for row in rows], dtype=np.float64)
    standardized = _standardize(
        matrix,
        means=np.asarray(model["feature_means"]),
        scales=np.asarray(model["feature_scales"]),
        profile=profile,
    )
    market_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        market_to_indices[str(row["market_id"])].append(index)
    markets = sorted(market_to_indices)
    rng = np.random.default_rng(int(config["bootstrap_seed"]))
    coefficients = []
    for _ in range(int(config["bootstrap_resample_count"])):
        sampled = rng.choice(markets, size=len(markets), replace=True)
        indices = [index for market in sampled for index in market_to_indices[str(market)]]
        weights, _ = _fit_logistic_weights(
            standardized[indices],
            targets[indices],
            l2=float(profile["model"]["l2_penalty"]),
            iterations=int(config["bootstrap_fit_iterations"]),
            learning_rate=float(profile["model"]["learning_rate"]),
            coefficient_bound=float(profile["model"]["coefficient_absolute_bound"]),
        )
        coefficients.append(weights)
    samples = np.asarray(coefficients)
    full = np.asarray(model["coefficients"])
    important = np.abs(full) >= float(config["important_coefficient_absolute_minimum"])
    sign_agreements = np.mean(np.sign(samples) == np.sign(full), axis=0)
    important_agreements = sign_agreements[important]
    median_agreement = (
        float(np.median(important_agreements)) if important_agreements.size else 0.0
    )
    gate_passed = bool(
        important_agreements.size
        and median_agreement >= float(config["minimum_median_sign_agreement"])
        and np.all(np.isfinite(samples))
    )
    return {
        "method": config["method"],
        "bootstrap_resample_count": int(config["bootstrap_resample_count"]),
        "important_coefficient_count": int(np.sum(important)),
        "median_important_coefficient_sign_agreement": median_agreement,
        "minimum_required_median_sign_agreement": float(
            config["minimum_median_sign_agreement"]
        ),
        "coefficient_sign_agreement_by_feature": {
            name: float(value) for name, value in zip(columns, sign_agreements, strict=True)
        },
        "coefficient_stability_gate_passed": gate_passed,
    }


def _threshold_metrics(
    *,
    threshold: float,
    calibration_rows: list[dict[str, Any]],
    probabilities: np.ndarray,
    profile: dict[str, Any],
) -> dict[str, Any]:
    config = profile["calibration"]
    selected = probabilities >= threshold
    targets = np.asarray([int(row["target"]) for row in calibration_rows])
    selected_count = int(np.sum(selected))
    precision = float(np.mean(targets[selected])) if selected_count else 0.0
    market_ids = np.asarray([str(row["market_id"]) for row in calibration_rows])
    unique_markets = sorted(set(market_ids[selected]))
    side_markets = {
        side: len(
            {
                str(row["market_id"])
                for row, keep in zip(calibration_rows, selected, strict=True)
                if keep and row["side"] == side
            }
        )
        for side in SIDES
    }
    recall_by_side = {}
    for side in SIDES:
        side_mask = np.asarray([row["side"] == side for row in calibration_rows])
        positive = side_mask & (targets == 1)
        recall_by_side[side] = (
            float(np.sum(selected & positive) / np.sum(positive)) if np.sum(positive) else 0.0
        )
    lcb = _market_precision_bootstrap_lcb(
        calibration_rows,
        selected=selected,
        targets=targets,
        samples=int(config["bootstrap_resample_count"]),
        confidence_level=float(config["bootstrap_confidence_level"]),
        seed=int(config["bootstrap_seed"]) + int(round(threshold * 100)),
    )
    checks = {
        "row_support": selected_count >= int(config["minimum_selected_row_count"]),
        "market_support": len(unique_markets)
        >= int(config["minimum_selected_unique_market_count"]),
        "side_market_support": all(
            side_markets[side]
            >= int(config["minimum_selected_unique_market_count_per_side"])
            for side in SIDES
        ),
        "precision": precision >= float(config["minimum_precision"]),
        "precision_lcb": lcb
        >= float(config["minimum_market_bootstrap_precision_lcb"]),
        "side_recall": all(
            recall_by_side[side] >= float(config["minimum_recall_per_side"])
            for side in SIDES
        ),
    }
    return {
        "threshold": threshold,
        "selected_row_count": selected_count,
        "selected_unique_market_count": len(unique_markets),
        "selected_unique_market_count_by_side": side_markets,
        "precision": precision,
        "recall_by_side": recall_by_side,
        "market_bootstrap_precision_lcb": lcb,
        "threshold_gate_checks": checks,
        "threshold_gate_passed": all(checks.values()),
    }


def _fit_logistic_weights(
    matrix: np.ndarray,
    targets: np.ndarray,
    *,
    l2: float,
    iterations: int,
    learning_rate: float,
    coefficient_bound: float,
) -> tuple[np.ndarray, float]:
    weights = np.zeros(matrix.shape[1], dtype=np.float64)
    prevalence = float(np.clip(np.mean(targets), 1e-6, 1.0 - 1e-6))
    intercept = math.log(prevalence / (1.0 - prevalence))
    for iteration in range(iterations):
        probabilities = _sigmoid(matrix @ weights + intercept)
        residual = probabilities - targets
        rate = learning_rate / math.sqrt(1.0 + iteration / 250.0)
        weights -= rate * ((matrix.T @ residual) / len(targets) + l2 * weights)
        intercept -= rate * float(np.mean(residual))
        weights = np.clip(weights, -coefficient_bound, coefficient_bound)
        intercept = float(np.clip(intercept, -coefficient_bound, coefficient_bound))
    return weights, intercept


def _standardize(
    matrix: np.ndarray,
    *,
    means: np.ndarray,
    scales: np.ndarray,
    profile: dict[str, Any],
) -> np.ndarray:
    clip = float(profile["model"]["standardized_feature_clip"])
    return np.clip((matrix - means) / scales, -clip, clip)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _brier(targets: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean((probabilities - targets) ** 2))


def _log_loss(targets: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
    return float(
        -np.mean(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped))
    )


def _roc_auc(targets: np.ndarray, probabilities: np.ndarray) -> float:
    positives = probabilities[targets == 1]
    negatives = probabilities[targets == 0]
    if not len(positives) or not len(negatives):
        return 0.5
    comparisons = positives[:, None] - negatives[None, :]
    return float(np.mean(comparisons > 0.0) + 0.5 * np.mean(comparisons == 0.0))


def _market_precision_bootstrap_lcb(
    rows: list[dict[str, Any]],
    *,
    selected: np.ndarray,
    targets: np.ndarray,
    samples: int,
    confidence_level: float,
    seed: int,
) -> float:
    market_counts: dict[str, tuple[int, int]] = {}
    for market_id in sorted({str(row["market_id"]) for row in rows}):
        indices = np.asarray(
            [index for index, row in enumerate(rows) if str(row["market_id"]) == market_id]
        )
        mask = selected[indices]
        market_counts[market_id] = (
            int(np.sum(targets[indices][mask])),
            int(np.sum(mask)),
        )
    markets = sorted(market_counts)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        chosen = rng.choice(markets, size=len(markets), replace=True)
        positive = sum(market_counts[str(market)][0] for market in chosen)
        count = sum(market_counts[str(market)][1] for market in chosen)
        values.append(positive / count if count else 0.0)
    return float(np.quantile(values, 1.0 - confidence_level, method="lower"))


def _market_pnl_bootstrap(
    rows: list[dict[str, Any]], *, samples: int, confidence_level: float, seed: int
) -> dict[str, Any]:
    market_pnl = defaultdict(float)
    for row in rows:
        market_pnl[str(row["market_id"])] += float(row["v6_3_accepted_bet_net_pnl"])
    markets = sorted(market_pnl)
    rng = np.random.default_rng(seed)
    values = np.asarray(
        [
            sum(market_pnl[str(market)] for market in rng.choice(markets, len(markets), replace=True))
            for _ in range(samples)
        ],
        dtype=np.float64,
    )
    alpha = 1.0 - confidence_level
    return {
        "bootstrap_unit": "market_id",
        "bootstrap_resample_count": samples,
        "confidence_level": confidence_level,
        "lower": float(np.quantile(values, alpha, method="lower")),
        "median": float(np.median(values)),
        "upper": float(np.quantile(values, confidence_level, method="higher")),
    }


def _validate_threshold_freeze(
    freeze: dict[str, Any], *, profile_path: Path, audit_path: Path, historical_path: Path
) -> None:
    checks = {
        "schema": freeze.get("schema_version") == THRESHOLD_FREEZE_SCHEMA_VERSION,
        "candidate": freeze.get("candidate_name") == CANDIDATE_NAME,
        "profile": freeze.get("fit_profile") == _descriptor(profile_path),
        "audit": freeze.get("audit_manifest") == _descriptor(audit_path),
        "historical": freeze.get("v6_2_historical_manifest")
        == _descriptor(historical_path),
        "calibration": freeze.get("calibration_gate_passed") is True,
        "threshold_frozen": freeze.get("threshold_frozen_before_oof_evaluation") is True,
        "oof_closed": freeze.get("oof_labels_loaded_by_fit_stage") is False,
        "safety": all(freeze.get(key) == value for key, value in _blocked_safety_fields().items()),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("v6.3 threshold freeze invalid: " + ", ".join(blockers))


def _find_forbidden_fields(rows: list[dict[str, Any]]) -> list[str]:
    found = set()
    for row in rows:
        for name, value in _flatten_items(row):
            lower = name.lower()
            if any(token in lower for token in TARGET_TOKENS):
                leaf_name = name.rsplit(".", 1)[-1]
                if (
                    leaf_name in ALLOWED_FALSE_SAFETY_DECLARATIONS
                    and value is False
                ):
                    continue
                found.add(name)
    return sorted(found)


def _strip_target_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if not any(token in key.lower() for token in TARGET_TOKENS)
    }


def _flatten_items(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        output = []
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            output.append((name, child))
            output.extend(_flatten_items(child, name))
        return output
    if isinstance(value, list):
        return [item for child in value for item in _flatten_items(child, prefix)]
    return []


def _model_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6.3 SBC Exit-Reliability Model",
            "",
            f"- fit_market_count: `{report['fit_market_count']}`",
            f"- fit_row_count: `{report['fit_row_count']}`",
            f"- fit_brier_score: `{report['fit_brier_score']}`",
            f"- coefficients_finite: `{str(report['coefficients_finite']).lower()}`",
            f"- coefficient_stability_gate_passed: `{str(report['coefficient_stability']['coefficient_stability_gate_passed']).lower()}`",
            "- validation_or_oof_labels_used_for_fit: `false`",
            "",
        ]
    )


def _calibration_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6.3 SBC Exit-Reliability Calibration",
            "",
            f"- calibration_market_count: `{report['calibration_market_count']}`",
            f"- selected_threshold: `{report['selected_threshold']}`",
            f"- precision_lcb: `{report['calibrated_exit_availability_precision_lcb']}`",
            f"- calibration_gate_passed: `{str(report['calibration_gate_passed']).lower()}`",
            f"- reason_codes: `{json.dumps(report['calibration_gate_reason_codes'])}`",
            "- threshold_search_uses_oof_labels: `false`",
            "- threshold_search_uses_pnl: `false`",
            "",
        ]
    )


def _oof_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6.3 Side-Only Historical OOF PnL Gate",
            "",
            f"- accepted_market_count: `{report['v6_3_guard_accepted_unique_market_count']}`",
            f"- accepted_markets_by_side: `{json.dumps(report['v6_3_guard_accepted_unique_market_count_by_side'], sort_keys=True)}`",
            f"- v6_2_post_cost_pnl: `{report['v6_2_side_only_post_cost_pnl']}`",
            f"- v6_3_post_cost_pnl: `{report['v6_3_side_only_post_cost_pnl']}`",
            f"- candidate_minus_v6_2_pnl: `{report['v6_3_minus_v6_2_side_only_post_cost_pnl']}`",
            f"- pnl_by_side: `{json.dumps(report['v6_3_post_cost_pnl_by_side'], sort_keys=True)}`",
            f"- historical_side_only_oof_gate_passed: `{str(report['historical_side_only_oof_gate_passed']).lower()}`",
            f"- reason_codes: `{json.dumps(report['historical_side_only_oof_gate_reason_codes'])}`",
            "- fully_blinded_historical_oof_claimed: `false`",
            "- future_unseen_holdout_required: `true`",
            "",
        ]
    )


def _blocked_safety_fields() -> dict[str, Any]:
    return {
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }


def _verified_descriptor(value: Any, name: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} descriptor missing")
    path = Path(str(value.get("path") or "")).resolve()
    expected = str(value.get("sha256") or "")
    _require_sha256(expected, f"{name} sha256")
    _verify_pin(path, expected, name)
    return {"path": str(path), "sha256": expected}


def _descriptor(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": _sha256_file(resolved)}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_pin(path: Path, expected: str, name: str) -> None:
    if not path.is_file() or _sha256_file(path) != expected:
        raise ValueError(f"{name} sha256 mismatch")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _require_sha256(value: str, name: str) -> None:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be a SHA-256 hex digest")


def _require_git_sha(value: str) -> None:
    if len(value) != 40 or not all(char in "0123456789abcdef" for char in value):
        raise ValueError("implementation_commit must be a Git SHA-1")
