"""Freeze the #226 policy-selected, guard-accepted runtime-PnL point model."""

from __future__ import annotations

import hashlib
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
    _blocked_safety_fields,
    _descriptor,
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
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4_fit import (
    _ridge_fit,
)

PROFILE_SCHEMA_VERSION = "bigan-v8-policy-selected-runtime-pnl-v6-6-profile-v1"
MODEL_SCHEMA_VERSION = "bigan-v8-policy-selected-runtime-pnl-v6-6-point-model-v1"
REPORT_SCHEMA_VERSION = "bigan-v8-policy-selected-runtime-pnl-v6-6-fit-report-v1"
MANIFEST_SCHEMA_VERSION = "bigan-v8-policy-selected-runtime-pnl-v6-6-point-freeze-manifest-v1"
CANDIDATE_NAME = "policy_selected_runtime_pnl_v6_6"
SIDES = ("UP", "DOWN")


@dataclass(frozen=True, slots=True)
class PolicySelectedRuntimePNLV66Config:
    """Pinned inputs for one #226 train-only point-model freeze."""

    run_id: str
    output_dir: Path | str
    profile_path: Path | str
    expected_profile_sha256: str
    v6_5_train_manifest_path: Path | str
    expected_v6_5_train_manifest_sha256: str
    v6_2_target_free_guard_replay_path: Path | str
    expected_v6_2_target_free_guard_replay_sha256: str
    external_selected_train_corpus_dir: Path | str
    implementation_commit: str
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for name, value in (
            ("expected_profile_sha256", self.expected_profile_sha256),
            ("expected_v6_5_train_manifest_sha256", self.expected_v6_5_train_manifest_sha256),
            (
                "expected_v6_2_target_free_guard_replay_sha256",
                self.expected_v6_2_target_free_guard_replay_sha256,
            ),
        ):
            _require_sha256(value, name)
        _require_git_sha(self.implementation_commit)
        for name in (
            "output_dir",
            "profile_path",
            "v6_5_train_manifest_path",
            "v6_2_target_free_guard_replay_path",
            "external_selected_train_corpus_dir",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


def validate_policy_selected_runtime_pnl_v6_6_profile(profile: dict[str, Any]) -> None:
    """Reject drift from the #226 selected-population protocol."""

    lineage = dict(profile.get("source_lineage") or {})
    population = dict(profile.get("fit_population") or {})
    model = dict(profile.get("model") or {})
    cross_fit = dict(profile.get("train_only_cross_fit_gate") or {})
    collection = dict(profile.get("fresh_calibration_collection") or {})
    calibration = dict(profile.get("fresh_calibration_gate") or {})
    prohibited = dict(profile.get("prohibited") or {})
    checks = {
        "schema": profile.get("schema_version") == PROFILE_SCHEMA_VERSION,
        "issue": profile.get("issue_number") == 226,
        "candidate": profile.get("candidate_name") == CANDIDATE_NAME,
        "frozen": profile.get("frozen") is True,
        "lineage": len(lineage) == 4
        and all(_is_sha256(str(value)) for value in lineage.values()),
        "population": population
        == {
            "role": "development_train",
            "execution_guard_order_allowed": True,
            "selected_action_family": "SELL_BEFORE_CLOSE",
            "target_action_must_equal_executed_action": True,
            "target_side_must_equal_selected_side": True,
            "expected_row_count": 65,
            "expected_unique_market_count": 65,
            "expected_side_count": {"UP": 22, "DOWN": 43},
            "one_row_per_market": True,
            "counterfactual_rows_included": False,
        },
        "model": model.get("family")
        == "deterministic_l2_ridge_policy_selected_runtime_net_pnl"
        and model.get("target")
        == "runtime_policy_after_cost_net_pnl_per_contract"
        and model.get("feature_columns")
        == [
            "side_is_up",
            "execution_price",
            "current_bid",
            "spread_bps",
            "queue_fill_probability_proxy",
            "time_to_close_seconds",
            "selected_side_probability",
            "canonical_v6_2_score",
        ]
        and float(model.get("ridge_alpha")) == 1.0
        and model.get("intercept_penalized") is False
        and float(model.get("standardized_feature_clip")) == 8.0
        and float(model.get("coefficient_absolute_bound")) == 8.0
        and all(
            model.get(name) is False
            for name in (
                "hyperparameter_search_enabled",
                "feature_set_search_enabled",
                "threshold_search_enabled",
                "validation_labels_used_for_model_fit",
                "oof_or_future_labels_used_for_model_fit",
            )
        ),
        "cross_fit": cross_fit.get("method")
        == "five_fold_sha256_market_grouped_cross_fit"
        and int(cross_fit.get("fold_count")) == 5
        and int(cross_fit.get("minimum_held_row_count")) == 8
        and cross_fit.get("require_both_sides_across_all_folds") is True
        and float(
            cross_fit.get("minimum_relative_mae_improvement_over_fold_train_mean_exclusive")
        )
        == 0.0
        and float(
            cross_fit.get("minimum_relative_mse_improvement_over_fold_train_mean_exclusive")
        )
        == 0.0
        and float(cross_fit.get("important_coefficient_absolute_minimum")) == 0.02
        and float(
            cross_fit.get("minimum_median_important_coefficient_sign_agreement")
        )
        == 0.65
        and cross_fit.get("result_selected_rerun_allowed") is False,
        "collection": int(collection.get("collector_index_boundary_sequence")) == 528
        and collection.get("collector_index_boundary_sha256")
        == "d5a4eb8c70b79b323c3783ebca722b365d434cca84d3e17cc1f59d8767af71d1"
        and collection.get("collector_last_entry_sha256")
        == "7af8b5f66f5115c1970d12a17161b0bf250635efb6931cf0516b1e6217cf3472"
        and int(collection.get("minimum_market_start_ts_exclusive")) == 1784541000000
        and int(collection.get("target_quality_valid_market_count")) == 60
        and int(collection.get("maximum_attempted_market_count")) == 90
        and int(collection.get("batch_market_count")) == 12
        and collection.get("labels_outcomes_resolution_or_pnl_opened_during_collection")
        is False
        and collection.get("candidate_scoring_during_collection") is False,
        "calibration": calibration.get("method")
        == "side_specific_market_bootstrap_upper_confidence_bound_of_mean_residual"
        and float(calibration.get("confidence_level")) == 0.95
        and int(calibration.get("bootstrap_resample_count")) == 5000
        and int(calibration.get("bootstrap_seed")) == 2262026
        and int(calibration.get("minimum_positive_lcb_unique_market_count_per_side"))
        == 20
        and calibration.get("threshold_search_enabled") is False
        and calibration.get("result_selected_rerun_allowed") is False,
        "prohibited": prohibited
        == {
            "v6_4_consumed_calibration_labels_used": False,
            "historical_oof_labels_or_pnl_used": False,
            "prior_future_or_paper_labels_or_pnl_used": False,
            "fresh_calibration_outcomes_opened_before_window_freeze": False,
            "outcome_settlement_pnl_or_future_fields_used_as_model_inputs": False,
            "v6_2_source_score_mutation_allowed": False,
            "execution_guard_cost_sizing_or_position_manager_mutation_allowed": False,
        },
        "safety": profile.get("safety") == _blocked_safety_fields(),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("#226 profile invalid: " + ", ".join(blockers))


def run_policy_selected_runtime_pnl_v6_6(
    config: PolicySelectedRuntimePNLV66Config,
) -> dict[str, Any]:
    """Select the frozen v6.2 deployment population and freeze ridge or block."""

    profile_path = Path(config.profile_path).resolve()
    train_manifest_path = Path(config.v6_5_train_manifest_path).resolve()
    replay_path = Path(config.v6_2_target_free_guard_replay_path).resolve()
    _verify_pin(profile_path, config.expected_profile_sha256, "#226 profile")
    _verify_pin(
        train_manifest_path,
        config.expected_v6_5_train_manifest_sha256,
        "#225 train manifest",
    )
    _verify_pin(
        replay_path,
        config.expected_v6_2_target_free_guard_replay_sha256,
        "v6.2 target-free replay",
    )
    profile = _load_json(profile_path)
    validate_policy_selected_runtime_pnl_v6_6_profile(profile)
    lineage = profile["source_lineage"]
    if _sha256_file(train_manifest_path) != lineage["v6_5_train_manifest_sha256"]:
        raise ValueError("#226 train manifest lineage mismatch")
    if _sha256_file(replay_path) != lineage["v6_2_target_free_guard_replay_sha256"]:
        raise ValueError("#226 replay lineage mismatch")
    train_manifest = _load_json(train_manifest_path)
    train_rows_descriptor = _verified_descriptor(train_manifest.get("train_rows"), "train rows")
    if train_rows_descriptor["sha256"] != lineage["v6_5_train_rows_sha256"]:
        raise ValueError("#226 train rows lineage mismatch")
    train_rows = _load_jsonl(Path(train_rows_descriptor["path"]))
    replay_rows = _load_jsonl(replay_path)
    selected_rows, support_audit = _select_policy_population(
        train_rows, replay_rows=replay_rows, profile=profile
    )

    run_dir = Path(config.output_dir).resolve() / config.run_id
    external_dir = Path(config.external_selected_train_corpus_dir).resolve()
    if not external_dir.is_relative_to(Path("/Volumes/PHILIPS/v8").resolve()):
        raise ValueError("#226 direct training corpus must live under /Volumes/PHILIPS/v8")
    for path in (run_dir, external_dir):
        if path.exists():
            if not config.overwrite_existing:
                raise FileExistsError(f"run path exists: {path}")
            shutil.rmtree(path)
        path.mkdir(parents=True)
    selected_path = external_dir / "policy_selected_runtime_pnl_v6_6_train_rows.jsonl"
    _write_jsonl(selected_path, selected_rows)
    model = _fit_model(selected_rows, profile=profile)
    cross_fit = _cross_fit(selected_rows, profile=profile)
    model.update(
        {
            "implementation_commit": config.implementation_commit,
            "profile_sha256": config.expected_profile_sha256,
            "policy_dataset_hash": _sha256_file(selected_path),
            "cross_fit_gate_passed": cross_fit["cross_fit_gate_passed"],
        }
    )
    model["model_artifact_id"] = canonical_json_sha256(model)
    model_path = run_dir / "v6_6_policy_selected_runtime_pnl_point_model.json"
    _write_json(model_path, model)
    checks = {
        "population_support": support_audit["population_support_gate_passed"],
        "feature_causality": all(
            int(row["max_input_ts"]) <= int(row["decision_ts"])
            for row in selected_rows
        ),
        "coefficients_finite_and_bounded": model["coefficients_finite"] is True
        and model["coefficients_bounded"] is True,
        "train_only_cross_fit": cross_fit["cross_fit_gate_passed"] is True,
        "consumed_calibration_excluded": True,
        "prior_oof_future_paper_excluded": True,
    }
    reasons = list(cross_fit["cross_fit_gate_blocking_reason_codes"])
    reasons.extend(
        f"{name}_gate_failed" for name, passed in checks.items() if not passed
    )
    reasons = sorted(set(reasons))
    split_hash = canonical_json_sha256(
        {"development_train_policy_selected": sorted(row["market_id"] for row in selected_rows)}
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        "profile": _descriptor(profile_path),
        "v6_5_train_manifest": _descriptor(train_manifest_path),
        "v6_2_target_free_guard_replay": _descriptor(replay_path),
        "selected_train_rows": _descriptor(selected_path),
        "point_model": _descriptor(model_path),
        "model_sha256": _sha256_file(model_path),
        "policy_dataset_hash": _sha256_file(selected_path),
        "split_hash": split_hash,
        "target_free_population_support_audit": support_audit,
        "cross_fit": cross_fit,
        "point_model_freeze_gate_checks": checks,
        "point_model_freeze_gate_passed": not reasons,
        "point_model_freeze_blocking_reason_codes": reasons,
        "fresh_calibration_boundary": {
            key: profile["fresh_calibration_collection"][key]
            for key in (
                "collector_index_boundary_sequence",
                "collector_index_boundary_sha256",
                "collector_last_entry_sha256",
                "minimum_market_start_ts_exclusive",
                "target_quality_valid_market_count",
                "maximum_attempted_market_count",
                "batch_market_count",
            )
        },
        "consumed_v6_4_calibration_labels_opened": False,
        "historical_oof_or_prior_future_paper_labels_opened": False,
        "fresh_calibration_outcomes_opened": False,
        "candidate_scoring_frozen": False,
        **_blocked_safety_fields(),
    }
    report_path = run_dir / "v6_6_policy_selected_runtime_pnl_fit_report.json"
    _write_json(report_path, report)
    _write_text(report_path.with_suffix(".md"), _report_markdown(report))
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        "profile": _descriptor(profile_path),
        "selected_train_rows": _descriptor(selected_path),
        "point_model": _descriptor(model_path),
        "fit_report": _descriptor(report_path),
        "model_sha256": _sha256_file(model_path),
        "policy_dataset_hash": _sha256_file(selected_path),
        "split_hash": split_hash,
        "point_model_frozen": report["point_model_freeze_gate_passed"],
        "fresh_calibration_collection_allowed": report[
            "point_model_freeze_gate_passed"
        ],
        "fresh_calibration_boundary": report["fresh_calibration_boundary"],
        "consumed_v6_4_calibration_labels_opened": False,
        "historical_oof_or_prior_future_paper_labels_opened": False,
        "fresh_calibration_outcomes_opened": False,
        "candidate_scoring_frozen": False,
        **_blocked_safety_fields(),
    }
    manifest["point_freeze_manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v6_6_policy_selected_runtime_pnl_point_freeze_manifest.json"
    _write_json(manifest_path, manifest)
    external_manifest = {
        "schema_version": "bigan-v8-policy-selected-runtime-pnl-v6-6-train-manifest-v1",
        "run_id": config.run_id,
        "selected_train_rows": _descriptor(selected_path),
        "source_point_freeze_manifest": _descriptor(manifest_path),
        "direct_training_corpus_only": True,
        **_blocked_safety_fields(),
    }
    external_manifest["manifest_id"] = canonical_json_sha256(external_manifest)
    _write_json(
        external_dir / "policy_selected_runtime_pnl_v6_6_train_manifest.json",
        external_manifest,
    )
    return {
        "run_dir": run_dir,
        "external_selected_train_corpus_dir": external_dir,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "report": report,
        "manifest": manifest,
    }


def score_policy_selected_runtime_pnl_rows(
    rows: list[dict[str, Any]], *, model: dict[str, Any]
) -> list[dict[str, Any]]:
    """Score target-free selected rows with the frozen compact feature contract."""

    matrix = np.asarray(
        [
            [float(row["features"][name]) for name in model["feature_columns"]]
            for row in rows
        ],
        dtype=np.float64,
    )
    standardized = np.clip(
        (matrix - np.asarray(model["feature_means"]))
        / np.asarray(model["feature_scales"]),
        -float(model["standardized_feature_clip"]),
        float(model["standardized_feature_clip"]),
    )
    predictions = standardized @ np.asarray(model["coefficients"]) + float(
        model["intercept"]
    )
    return [
        {
            **row,
            "runtime_expected_net_pnl_point": float(prediction),
            "runtime_expected_net_pnl_source": "frozen_policy_selected_train_only_ridge",
            "target_fields_used_for_prediction": False,
        }
        for row, prediction in zip(rows, predictions, strict=True)
    ]


def _select_policy_population(
    train_rows: list[dict[str, Any]],
    *,
    replay_rows: list[dict[str, Any]],
    profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    replay = {
        (str(row["market_id"]), int(row["decision_ts"])): row for row in replay_rows
    }
    selected = []
    reasons: Counter[str] = Counter()
    for row in train_rows:
        source = replay.get((str(row["market_id"]), int(row["decision_ts"])))
        if source is None:
            reasons["target_free_replay_row_missing"] += 1
            continue
        checks = {
            "role": row.get("role") == "development_train",
            "guard": source.get("execution_guard_order_allowed") is True,
            "family": source.get("selected_action_family") == "SELL_BEFORE_CLOSE",
            "action": source.get("executed_action") == row.get("action"),
            "side": source.get("selected_side") == row.get("side"),
        }
        if all(checks.values()):
            selected.append(row)
        else:
            reasons.update(name for name, passed in checks.items() if not passed)
    selected.sort(key=lambda row: (int(row["decision_ts"]), str(row["market_id"])))
    side_count = Counter(str(row["side"]) for row in selected)
    market_count = len({str(row["market_id"]) for row in selected})
    one_per_market = len(selected) == market_count
    expected = profile["fit_population"]
    checks = {
        "row_count": len(selected) == int(expected["expected_row_count"]),
        "market_count": market_count == int(expected["expected_unique_market_count"]),
        "side_count": dict(side_count) == expected["expected_side_count"],
        "one_row_per_market": one_per_market,
        "feature_causality": all(
            int(row["max_input_ts"]) <= int(row["decision_ts"]) for row in selected
        ),
    }
    return selected, {
        "source_train_row_count": len(train_rows),
        "selected_row_count": len(selected),
        "selected_unique_market_count": market_count,
        "selected_side_count": dict(sorted(side_count.items())),
        "excluded_reason_distribution": dict(sorted(reasons.items())),
        "population_support_checks": checks,
        "population_support_gate_passed": all(checks.values()),
        "outcome_settlement_target_or_pnl_fields_used_for_selection": False,
    }


def _fit_model(rows: list[dict[str, Any]], *, profile: dict[str, Any]) -> dict[str, Any]:
    config = profile["model"]
    columns = list(config["feature_columns"])
    matrix = np.asarray(
        [[float(row["features"][name]) for name in columns] for row in rows],
        dtype=np.float64,
    )
    targets = np.asarray([float(row[config["target"]]) for row in rows])
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    standardized = np.clip(
        (matrix - means) / scales,
        -float(config["standardized_feature_clip"]),
        float(config["standardized_feature_clip"]),
    )
    coefficients, intercept = _ridge_fit(
        standardized, targets, alpha=float(config["ridge_alpha"])
    )
    parameters = np.append(coefficients, intercept)
    bound = float(config["coefficient_absolute_bound"])
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "model_family": config["family"],
        "target": config["target"],
        "fit_population": "frozen_v6_2_policy_selected_guard_accepted_sbc",
        "fit_market_count": len({str(row["market_id"]) for row in rows}),
        "fit_row_count": len(rows),
        "feature_columns": columns,
        "feature_means": means.tolist(),
        "feature_scales": scales.tolist(),
        "standardized_feature_clip": float(config["standardized_feature_clip"]),
        "ridge_alpha": float(config["ridge_alpha"]),
        "coefficients": coefficients.tolist(),
        "intercept": float(intercept),
        "coefficient_absolute_bound": bound,
        "coefficients_finite": bool(np.all(np.isfinite(parameters))),
        "coefficients_bounded": bool(np.max(np.abs(parameters)) <= bound),
        "hyperparameter_search_enabled": False,
        "feature_set_search_enabled": False,
        "threshold_search_enabled": False,
        "validation_labels_used_for_model_fit": False,
        "oof_or_future_labels_used_for_model_fit": False,
        "target_fields_used_as_model_inputs": False,
    }


def _cross_fit(rows: list[dict[str, Any]], *, profile: dict[str, Any]) -> dict[str, Any]:
    config = profile["train_only_cross_fit_gate"]
    fold_count = int(config["fold_count"])
    fold_by_market = {
        str(row["market_id"]): int(
            hashlib.sha256(str(row["market_id"]).encode()).hexdigest(), 16
        )
        % fold_count
        for row in rows
    }
    predictions = []
    fold_reports = []
    coefficients = []
    for fold in range(fold_count):
        fit_rows = [row for row in rows if fold_by_market[str(row["market_id"])] != fold]
        held_rows = [row for row in rows if fold_by_market[str(row["market_id"])] == fold]
        model = _fit_model(fit_rows, profile=profile)
        scored = score_policy_selected_runtime_pnl_rows(held_rows, model=model)
        baseline = float(np.mean([float(row[profile["model"]["target"]]) for row in fit_rows]))
        predictions.extend(
            {**row, "cross_fit_fold": fold, "fold_train_mean_baseline": baseline}
            for row in scored
        )
        coefficients.append([*model["coefficients"], model["intercept"]])
        side_count = Counter(str(row["side"]) for row in held_rows)
        fold_reports.append(
            {
                "fold": fold,
                "fit_row_count": len(fit_rows),
                "held_row_count": len(held_rows),
                "held_side_count": dict(sorted(side_count.items())),
                "support_passed": len(held_rows)
                >= int(config["minimum_held_row_count"])
                and all(side_count[side] > 0 for side in SIDES),
            }
        )
    targets = np.asarray([float(row[profile["model"]["target"]]) for row in predictions])
    point = np.asarray([float(row["runtime_expected_net_pnl_point"]) for row in predictions])
    baseline = np.asarray([float(row["fold_train_mean_baseline"]) for row in predictions])
    point_metrics = _error_metrics(targets, point)
    baseline_metrics = _error_metrics(targets, baseline)
    relative_mae = _relative_improvement(point_metrics["mae"], baseline_metrics["mae"])
    relative_mse = _relative_improvement(point_metrics["mse"], baseline_metrics["mse"])
    full_model = _fit_model(rows, profile=profile)
    full = np.asarray([*full_model["coefficients"], full_model["intercept"]])
    samples = np.asarray(coefficients)
    important = np.abs(full) >= float(config["important_coefficient_absolute_minimum"])
    sign_agreement = np.mean(np.sign(samples) == np.sign(full), axis=0)
    median_sign = float(np.median(sign_agreement[important])) if np.any(important) else 1.0
    checks = {
        "fold_support": all(report["support_passed"] for report in fold_reports),
        "relative_mae_improvement": relative_mae
        > float(config["minimum_relative_mae_improvement_over_fold_train_mean_exclusive"]),
        "relative_mse_improvement": relative_mse
        > float(config["minimum_relative_mse_improvement_over_fold_train_mean_exclusive"]),
        "coefficient_stability": median_sign
        >= float(config["minimum_median_important_coefficient_sign_agreement"]),
        "coefficients_finite_and_bounded": full_model["coefficients_finite"] is True
        and full_model["coefficients_bounded"] is True,
        "no_result_selected_rerun": config["result_selected_rerun_allowed"] is False,
    }
    reasons = [f"{name}_gate_failed" for name, passed in checks.items() if not passed]
    return {
        "method": config["method"],
        "fold_count": fold_count,
        "fold_reports": fold_reports,
        "point_model_metrics": point_metrics,
        "fold_train_mean_constant_metrics": baseline_metrics,
        "relative_mae_improvement_over_fold_train_mean": relative_mae,
        "relative_mse_improvement_over_fold_train_mean": relative_mse,
        "important_coefficient_count": int(np.sum(important)),
        "median_important_coefficient_sign_agreement": median_sign,
        "cross_fit_gate_checks": checks,
        "cross_fit_gate_passed": not reasons,
        "cross_fit_gate_blocking_reason_codes": reasons,
        "validation_oof_or_future_labels_used": False,
    }


def _error_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    errors = predictions - targets
    return {
        "mae": float(np.mean(np.abs(errors))),
        "mse": float(np.mean(errors**2)),
        "mean_error": float(np.mean(errors)),
    }


def _relative_improvement(candidate: float, baseline: float) -> float:
    return (baseline - candidate) / baseline if baseline > 0.0 else 0.0


def _report_markdown(report: dict[str, Any]) -> str:
    audit = report["target_free_population_support_audit"]
    cross_fit = report["cross_fit"]
    return "\n".join(
        [
            "# #226 v6.6 policy-selected runtime-PnL point freeze",
            "",
            f"- run id: `{report['run_id']}`",
            f"- selected markets/rows: `{audit['selected_unique_market_count']}/"
            f"{audit['selected_row_count']}`",
            f"- selected side count: `{audit['selected_side_count']}`",
            f"- model SHA-256: `{report['model_sha256']}`",
            f"- policy dataset hash: `{report['policy_dataset_hash']}`",
            f"- split hash: `{report['split_hash']}`",
            f"- cross-fit relative MAE improvement: "
            f"`{cross_fit['relative_mae_improvement_over_fold_train_mean']}`",
            f"- cross-fit relative MSE improvement: "
            f"`{cross_fit['relative_mse_improvement_over_fold_train_mean']}`",
            f"- cross-fit gate passed: `{cross_fit['cross_fit_gate_passed']}`",
            f"- point-model freeze passed: `{report['point_model_freeze_gate_passed']}`",
            f"- blockers: `{report['point_model_freeze_blocking_reason_codes']}`",
            "- outcome/settlement/target/PnL used for population selection: `false`",
            "- consumed calibration/OFF/future/paper labels opened: `false`",
            "- paper/live/write/wallet/capital/promotion unlock: `false`",
        ]
    ) + "\n"


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
