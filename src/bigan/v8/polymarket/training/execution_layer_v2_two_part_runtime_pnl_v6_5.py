"""Build and freeze the train-only #225 two-part runtime-PnL point model."""

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
    _market_runtime_target_rows,
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
    _feature_matrix,
    _ridge_fit,
)
from bigan.v8.polymarket.training.execution_layer_v2_sbc_exit_reliability_v6_3_fit import (
    _fit_logistic_weights,
    _materialize_rows,
    _roc_auc,
    _sigmoid,
)

PROFILE_SCHEMA_VERSION = "bigan-v8-two-part-runtime-pnl-v6-5-profile-v1"
MODEL_SCHEMA_VERSION = "bigan-v8-two-part-runtime-pnl-v6-5-point-model-v1"
REPORT_SCHEMA_VERSION = "bigan-v8-two-part-runtime-pnl-v6-5-fit-report-v1"
MANIFEST_SCHEMA_VERSION = "bigan-v8-two-part-runtime-pnl-v6-5-point-freeze-manifest-v1"
CANDIDATE_NAME = "two_part_runtime_pnl_v6_5"
SIDES = ("UP", "DOWN")


@dataclass(frozen=True, slots=True)
class TwoPartRuntimePNLV65Config:
    """Pinned inputs for the train-only #225 point-model freeze."""

    run_id: str
    output_dir: Path | str
    profile_path: Path | str
    expected_profile_sha256: str
    v6_4_lineage_freeze_manifest_path: Path | str
    expected_v6_4_lineage_freeze_manifest_sha256: str
    external_train_corpus_dir: Path | str
    implementation_commit: str
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(self.expected_profile_sha256, "expected_profile_sha256")
        _require_sha256(
            self.expected_v6_4_lineage_freeze_manifest_sha256,
            "expected_v6_4_lineage_freeze_manifest_sha256",
        )
        _require_git_sha(self.implementation_commit)
        for name in (
            "output_dir",
            "profile_path",
            "v6_4_lineage_freeze_manifest_path",
            "external_train_corpus_dir",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


def validate_two_part_runtime_pnl_v6_5_profile(profile: dict[str, Any]) -> None:
    """Validate the frozen #225 architecture and fresh calibration boundary."""

    lineage = dict(profile.get("source_lineage") or {})
    roles = dict(profile.get("roles") or {})
    model = dict(profile.get("model") or {})
    sentinel = dict(model.get("sentinel_feature_policy") or {})
    exit_model = dict(model.get("exit_probability_model") or {})
    closed_model = dict(model.get("closed_path_model") or {})
    residual_model = dict(model.get("residual_path_model") or {})
    cross_fit = dict(profile.get("train_only_cross_fit_gate") or {})
    collection = dict(profile.get("fresh_calibration_collection") or {})
    calibration = dict(profile.get("fresh_calibration_gate") or {})
    prohibited = dict(profile.get("prohibited") or {})
    checks = {
        "schema": profile.get("schema_version") == PROFILE_SCHEMA_VERSION,
        "issue": profile.get("issue_number") == 225,
        "candidate": profile.get("candidate_name") == CANDIDATE_NAME,
        "frozen": profile.get("frozen") is True,
        "lineage": len(lineage) == 4
        and all(_is_sha256(str(value)) for value in lineage.values()),
        "roles": roles
        == {
            "model_fit": "development_train",
            "model_fit_market_count": 89,
            "consumed_v6_4_calibration_market_count_excluded": 45,
            "historical_oof_market_count_excluded": 60,
            "fresh_calibration_market_count": 60,
            "market_disjoint": True,
            "chronological": True,
        },
        "model": model.get("family")
        == "fixed_two_part_runtime_policy_net_pnl_mixture"
        and model.get("target")
        == "runtime_policy_after_cost_net_pnl_per_contract"
        and model.get("lifecycle_target") == "position_lifecycle_class"
        and len(model.get("raw_feature_columns") or []) == 24
        and len(set(model.get("raw_feature_columns") or [])) == 24
        and sentinel.get("fields")
        == ["canonical_v6_2_score", "action_score_margin"]
        and float(sentinel.get("sentinel_maximum")) == -100000.0
        and float(sentinel.get("replacement_value")) == 0.0
        and sentinel.get("availability_indicator_suffix") == "_available"
        and exit_model
        == {
            "family": "deterministic_l2_logistic_regression",
            "l2_penalty": 0.05,
            "gradient_descent_iterations": 3000,
            "learning_rate": 0.05,
            "probability_clip": 1e-06,
        }
        and closed_model
        == {"family": "deterministic_l2_ridge_regression", "ridge_alpha": 1.0}
        and residual_model
        == {"family": "deterministic_l2_ridge_regression", "ridge_alpha": 1.0}
        and float(model.get("standardized_feature_clip")) == 8.0
        and float(model.get("coefficient_absolute_bound")) == 8.0
        and model.get("hyperparameter_search_enabled") is False
        and model.get("feature_set_search_enabled") is False
        and model.get("validation_labels_used_for_model_fit") is False
        and model.get("oof_or_future_labels_used_for_model_fit") is False,
        "cross_fit": cross_fit.get("method")
        == "five_fold_market_hash_grouped_cross_fit"
        and int(cross_fit.get("fold_count")) == 5
        and cross_fit.get("market_hash") == "sha256_market_id_mod_fold_count"
        and int(cross_fit.get("minimum_closed_rows_per_fold")) == 40
        and int(cross_fit.get("minimum_residual_rows_per_fold")) == 20
        and float(
            cross_fit.get("minimum_relative_mae_improvement_over_fold_train_mean_exclusive")
        )
        == 0.0
        and float(
            cross_fit.get("minimum_relative_mse_improvement_over_fold_train_mean_exclusive")
        )
        == 0.0
        and float(cross_fit.get("minimum_exit_probability_roc_auc")) == 0.55
        and int(cross_fit.get("minimum_market_count_per_side")) == 40
        and float(cross_fit.get("important_coefficient_absolute_minimum")) == 0.02
        and float(
            cross_fit.get("minimum_median_important_coefficient_sign_agreement")
        )
        == 0.65
        and cross_fit.get("result_selected_rerun_allowed") is False,
        "collection_boundary": int(collection.get("collector_index_boundary_sequence"))
        == 528
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
        and collection.get("candidate_scoring_during_collection") is False
        and collection.get("per_round_github_comments") is False,
        "calibration": calibration.get("method")
        == "side_specific_market_bootstrap_upper_confidence_bound_of_mean_residual"
        and float(calibration.get("confidence_level")) == 0.95
        and int(calibration.get("bootstrap_resample_count")) == 5000
        and int(calibration.get("bootstrap_seed")) == 2252026
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
        raise ValueError("#225 profile invalid: " + ", ".join(blockers))


def run_two_part_runtime_pnl_v6_5(
    config: TwoPartRuntimePNLV65Config,
) -> dict[str, Any]:
    """Rebuild train-only targets, cross-fit, and freeze the point model or block."""

    profile_path = Path(config.profile_path).resolve()
    lineage_path = Path(config.v6_4_lineage_freeze_manifest_path).resolve()
    _verify_pin(profile_path, config.expected_profile_sha256, "#225 profile")
    _verify_pin(
        lineage_path,
        config.expected_v6_4_lineage_freeze_manifest_sha256,
        "#224 lineage freeze",
    )
    profile = _load_json(profile_path)
    validate_two_part_runtime_pnl_v6_5_profile(profile)
    if _sha256_file(lineage_path) != profile["source_lineage"][
        "v6_4_lineage_freeze_manifest_sha256"
    ]:
        raise ValueError("#225 lineage profile pin mismatch")
    lineage = _load_json(lineage_path)
    target_profile_descriptor = _verified_descriptor(lineage.get("profile"), "target profile")
    if target_profile_descriptor["sha256"] != profile["source_lineage"][
        "v6_4_target_profile_sha256"
    ]:
        raise ValueError("#225 target profile pin mismatch")
    target_profile = _load_json(Path(target_profile_descriptor["path"]))
    historical_descriptor = _verified_descriptor(
        lineage.get("v6_2_historical_manifest"), "v6.2 historical manifest"
    )
    if historical_descriptor["sha256"] != profile["source_lineage"][
        "v6_2_historical_manifest_sha256"
    ]:
        raise ValueError("#225 v6.2 historical pin mismatch")
    historical = _load_json(Path(historical_descriptor["path"]))
    replay_descriptor = _verified_descriptor(
        historical.get("candidate_target_free_guard_replay"), "v6.2 target-free replay"
    )
    replay_rows = _load_jsonl(Path(replay_descriptor["path"]))
    source_rows = _load_jsonl(
        Path(_verified_descriptor(lineage.get("lineage_rows"), "lineage rows")["path"])
    )
    train_sources = [row for row in source_rows if row.get("role") == "development_train"]
    if len(train_sources) != 89:
        raise ValueError("#225 train-only source count mismatch")

    run_dir = Path(config.output_dir).resolve() / config.run_id
    external_dir = Path(config.external_train_corpus_dir).resolve()
    if not external_dir.is_relative_to(Path("/Volumes/PHILIPS/v8").resolve()):
        raise ValueError("#225 direct training corpus must live under /Volumes/PHILIPS/v8")
    for path in (run_dir, external_dir):
        if path.exists():
            if not config.overwrite_existing:
                raise FileExistsError(f"run path exists: {path}")
            shutil.rmtree(path)
        path.mkdir(parents=True)

    rows: list[dict[str, Any]] = []
    lifecycle_reasons: Counter[str] = Counter()
    for source in sorted(train_sources, key=lambda row: int(row["minimum_decision_ts"])):
        feature_rows = _load_jsonl(
            Path(_verified_descriptor(source["feature_rows"], "feature rows")["path"])
        )
        label_rows = _load_jsonl(
            Path(_verified_descriptor(source["label_rows"], "label rows")["path"])
        )
        decision_rows = _materialize_rows(
            [source],
            replay_rows=replay_rows,
            roles={"development_train"},
            include_targets=False,
        )
        market_rows, reasons = _market_runtime_target_rows(
            source=source,
            feature_rows=feature_rows,
            label_rows=label_rows,
            decision_rows=decision_rows,
            profile=target_profile,
            run_id=config.run_id,
        )
        rows.extend(market_rows)
        lifecycle_reasons.update(reasons)
    rows.sort(key=lambda row: (int(row["decision_ts"]), str(row["market_id"]), str(row["action"])))
    if len(rows) != 712 or len({str(row["market_id"]) for row in rows}) != 89:
        raise ValueError("#225 train-only target corpus support mismatch")
    if any(row.get("role") != "development_train" for row in rows):
        raise ValueError("#225 consumed calibration role entered train corpus")
    corpus_path = external_dir / "two_part_runtime_pnl_v6_5_train_rows.jsonl"
    _write_jsonl(corpus_path, rows)

    cross_fit = _market_grouped_cross_fit(rows, profile=profile)
    model = _fit_two_part_model(rows, profile=profile)
    model.update(
        {
            "implementation_commit": config.implementation_commit,
            "profile_sha256": config.expected_profile_sha256,
            "train_corpus_sha256": _sha256_file(corpus_path),
            "cross_fit_gate_passed": cross_fit["cross_fit_gate_passed"],
        }
    )
    model["model_artifact_id"] = canonical_json_sha256(model)
    model_path = run_dir / "v6_5_two_part_runtime_pnl_point_model.json"
    _write_json(model_path, model)
    report = _fit_report(
        config=config,
        profile_path=profile_path,
        lineage_path=lineage_path,
        corpus_path=corpus_path,
        model_path=model_path,
        rows=rows,
        lifecycle_reasons=lifecycle_reasons,
        model=model,
        cross_fit=cross_fit,
    )
    report_path = run_dir / "v6_5_two_part_runtime_pnl_fit_report.json"
    _write_json(report_path, report)
    _write_text(report_path.with_suffix(".md"), _report_markdown(report))
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        "profile": _descriptor(profile_path),
        "v6_4_lineage_freeze_manifest": _descriptor(lineage_path),
        "train_rows": _descriptor(corpus_path),
        "point_model": _descriptor(model_path),
        "fit_report": _descriptor(report_path),
        "model_sha256": _sha256_file(model_path),
        "policy_dataset_hash": _sha256_file(corpus_path),
        "split_hash": report["split_hash"],
        "point_model_frozen": report["point_model_freeze_gate_passed"],
        "fresh_calibration_collection_allowed": report[
            "point_model_freeze_gate_passed"
        ],
        "fresh_calibration_boundary": report["fresh_calibration_boundary"],
        "fresh_calibration_outcomes_opened": False,
        "consumed_v6_4_calibration_labels_opened": False,
        "historical_oof_or_future_labels_opened": False,
        "candidate_scoring_frozen": False,
        "future_side_only_pnl_gate_required": True,
        **_blocked_safety_fields(),
    }
    manifest["point_freeze_manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v6_5_two_part_runtime_pnl_point_freeze_manifest.json"
    _write_json(manifest_path, manifest)
    external_manifest = {
        "schema_version": "bigan-v8-two-part-runtime-pnl-v6-5-train-corpus-manifest-v1",
        "run_id": config.run_id,
        "train_rows": _descriptor(corpus_path),
        "source_point_freeze_manifest": _descriptor(manifest_path),
        "direct_training_corpus_only": True,
        "consumed_v6_4_calibration_labels_opened": False,
        **_blocked_safety_fields(),
    }
    external_manifest["manifest_id"] = canonical_json_sha256(external_manifest)
    external_manifest_path = external_dir / "two_part_runtime_pnl_v6_5_train_manifest.json"
    _write_json(external_manifest_path, external_manifest)
    return {
        "run_dir": run_dir,
        "external_train_corpus_dir": external_dir,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "report": report,
        "manifest": manifest,
    }


def score_two_part_runtime_pnl_rows(
    rows: list[dict[str, Any]], *, model: dict[str, Any]
) -> list[dict[str, Any]]:
    """Apply the frozen three-component point estimand without target access."""

    matrix, names = _feature_matrix(
        rows, model["sentinel_feature_policy"], model["raw_feature_columns"]
    )
    if names != list(model["model_feature_columns"]):
        raise ValueError("#225 model feature contract mismatch")
    standardized = np.clip(
        (matrix - np.asarray(model["feature_means"]))
        / np.asarray(model["feature_scales"]),
        -float(model["standardized_feature_clip"]),
        float(model["standardized_feature_clip"]),
    )
    exit_component = model["exit_probability_component"]
    closed_component = model["closed_path_component"]
    residual_component = model["residual_path_component"]
    p_exit = _sigmoid(
        standardized @ np.asarray(exit_component["coefficients"])
        + float(exit_component["intercept"])
    )
    p_exit = np.clip(p_exit, float(exit_component["probability_clip"]), 1.0 - float(exit_component["probability_clip"]))
    closed = standardized @ np.asarray(closed_component["coefficients"]) + float(
        closed_component["intercept"]
    )
    residual = standardized @ np.asarray(residual_component["coefficients"]) + float(
        residual_component["intercept"]
    )
    combined = p_exit * closed + (1.0 - p_exit) * residual
    return [
        {
            **row,
            "runtime_exit_probability": float(probability),
            "runtime_closed_path_expected_net_pnl": float(closed_value),
            "runtime_residual_path_expected_net_pnl": float(residual_value),
            "runtime_expected_net_pnl_point": float(combined_value),
            "runtime_expected_net_pnl_source": "frozen_train_only_two_part_mixture",
            "target_fields_used_for_prediction": False,
        }
        for row, probability, closed_value, residual_value, combined_value in zip(
            rows, p_exit, closed, residual, combined, strict=True
        )
    ]


def _fit_two_part_model(rows: list[dict[str, Any]], *, profile: dict[str, Any]) -> dict[str, Any]:
    config = profile["model"]
    matrix, names = _feature_matrix(
        rows, config["sentinel_feature_policy"], config["raw_feature_columns"]
    )
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    standardized = np.clip(
        (matrix - means) / scales,
        -float(config["standardized_feature_clip"]),
        float(config["standardized_feature_clip"]),
    )
    lifecycle = np.asarray(
        [float(row["position_lifecycle_class"] == "closed_before_settlement") for row in rows]
    )
    targets = np.asarray([float(row[config["target"]]) for row in rows])
    exit_config = config["exit_probability_model"]
    exit_coefficients, exit_intercept = _fit_logistic_weights(
        standardized,
        lifecycle,
        l2=float(exit_config["l2_penalty"]),
        iterations=int(exit_config["gradient_descent_iterations"]),
        learning_rate=float(exit_config["learning_rate"]),
        coefficient_bound=float(config["coefficient_absolute_bound"]),
    )
    closed_mask = lifecycle == 1.0
    residual_mask = ~closed_mask
    closed_coefficients, closed_intercept = _ridge_fit(
        standardized[closed_mask], targets[closed_mask], alpha=1.0
    )
    residual_coefficients, residual_intercept = _ridge_fit(
        standardized[residual_mask], targets[residual_mask], alpha=1.0
    )
    all_parameters = np.concatenate(
        (
            exit_coefficients,
            [exit_intercept],
            closed_coefficients,
            [closed_intercept],
            residual_coefficients,
            [residual_intercept],
        )
    )
    bound = float(config["coefficient_absolute_bound"])
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "model_family": config["family"],
        "combined_estimand": config["combined_estimand"],
        "target": config["target"],
        "fit_role": "development_train",
        "fit_market_count": len({str(row["market_id"]) for row in rows}),
        "fit_row_count": len(rows),
        "raw_feature_columns": list(config["raw_feature_columns"]),
        "model_feature_columns": names,
        "sentinel_feature_policy": config["sentinel_feature_policy"],
        "feature_means": means.tolist(),
        "feature_scales": scales.tolist(),
        "standardized_feature_clip": float(config["standardized_feature_clip"]),
        "coefficient_absolute_bound": bound,
        "exit_probability_component": {
            "family": exit_config["family"],
            "l2_penalty": float(exit_config["l2_penalty"]),
            "probability_clip": float(exit_config["probability_clip"]),
            "coefficients": exit_coefficients.tolist(),
            "intercept": float(exit_intercept),
        },
        "closed_path_component": {
            "family": config["closed_path_model"]["family"],
            "ridge_alpha": 1.0,
            "fit_row_count": int(np.sum(closed_mask)),
            "coefficients": closed_coefficients.tolist(),
            "intercept": float(closed_intercept),
        },
        "residual_path_component": {
            "family": config["residual_path_model"]["family"],
            "ridge_alpha": 1.0,
            "fit_row_count": int(np.sum(residual_mask)),
            "coefficients": residual_coefficients.tolist(),
            "intercept": float(residual_intercept),
        },
        "coefficients_finite": bool(np.all(np.isfinite(all_parameters))),
        "coefficients_bounded": bool(np.max(np.abs(all_parameters)) <= bound),
        "hyperparameter_search_enabled": False,
        "feature_set_search_enabled": False,
        "validation_labels_used_for_model_fit": False,
        "oof_or_future_labels_used_for_model_fit": False,
        "target_fields_used_as_model_inputs": False,
    }


def _market_grouped_cross_fit(
    rows: list[dict[str, Any]], *, profile: dict[str, Any]
) -> dict[str, Any]:
    config = profile["train_only_cross_fit_gate"]
    fold_count = int(config["fold_count"])
    markets = sorted({str(row["market_id"]) for row in rows})
    fold_by_market = {
        market: int(hashlib.sha256(market.encode()).hexdigest(), 16) % fold_count
        for market in markets
    }
    predictions = []
    fold_reports = []
    coefficient_vectors = []
    for fold in range(fold_count):
        fit_rows = [row for row in rows if fold_by_market[str(row["market_id"])] != fold]
        held_rows = [row for row in rows if fold_by_market[str(row["market_id"])] == fold]
        if not held_rows:
            raise ValueError("#225 market hash cross-fit produced empty fold")
        model = _fit_two_part_model(fit_rows, profile=profile)
        scored = score_two_part_runtime_pnl_rows(held_rows, model=model)
        baseline = float(np.mean([float(row[profile["model"]["target"]]) for row in fit_rows]))
        for row in scored:
            predictions.append({**row, "cross_fit_fold": fold, "fold_train_mean_baseline": baseline})
        coefficient_vectors.append(_flatten_component_coefficients(model))
        held_closed = sum(
            row["position_lifecycle_class"] == "closed_before_settlement" for row in held_rows
        )
        held_residual = len(held_rows) - held_closed
        fold_reports.append(
            {
                "fold": fold,
                "fit_market_count": len({str(row["market_id"]) for row in fit_rows}),
                "held_market_count": len({str(row["market_id"]) for row in held_rows}),
                "held_row_count": len(held_rows),
                "held_closed_row_count": held_closed,
                "held_residual_row_count": held_residual,
                "support_passed": held_closed >= int(config["minimum_closed_rows_per_fold"])
                and held_residual >= int(config["minimum_residual_rows_per_fold"]),
            }
        )
    predictions.sort(key=lambda row: (int(row["decision_ts"]), str(row["market_id"]), str(row["action"])))
    targets = np.asarray([float(row[profile["model"]["target"]]) for row in predictions])
    point = np.asarray([float(row["runtime_expected_net_pnl_point"]) for row in predictions])
    baseline = np.asarray([float(row["fold_train_mean_baseline"]) for row in predictions])
    point_metrics = _error_metrics(targets, point)
    baseline_metrics = _error_metrics(targets, baseline)
    relative_mae = _relative_improvement(point_metrics["mae"], baseline_metrics["mae"])
    relative_mse = _relative_improvement(point_metrics["mse"], baseline_metrics["mse"])
    lifecycle = np.asarray(
        [float(row["position_lifecycle_class"] == "closed_before_settlement") for row in predictions]
    )
    exit_probabilities = np.asarray([float(row["runtime_exit_probability"]) for row in predictions])
    auc = _roc_auc(lifecycle, exit_probabilities)
    full_model = _fit_two_part_model(rows, profile=profile)
    full_coefficients = _flatten_component_coefficients(full_model)
    samples = np.asarray(coefficient_vectors)
    important = np.abs(full_coefficients) >= float(config["important_coefficient_absolute_minimum"])
    sign_agreement = np.mean(np.sign(samples) == np.sign(full_coefficients), axis=0)
    median_sign = float(np.median(sign_agreement[important])) if np.any(important) else 1.0
    side_markets = {
        side: len({str(row["market_id"]) for row in rows if row["side"] == side})
        for side in SIDES
    }
    checks = {
        "fold_support": all(report["support_passed"] for report in fold_reports),
        "relative_mae_improvement": relative_mae
        > float(config["minimum_relative_mae_improvement_over_fold_train_mean_exclusive"]),
        "relative_mse_improvement": relative_mse
        > float(config["minimum_relative_mse_improvement_over_fold_train_mean_exclusive"]),
        "exit_probability_roc_auc": auc >= float(config["minimum_exit_probability_roc_auc"]),
        "side_market_support": all(
            side_markets[side] >= int(config["minimum_market_count_per_side"])
            for side in SIDES
        ),
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
        "market_hash": config["market_hash"],
        "fold_reports": fold_reports,
        "point_model_metrics": point_metrics,
        "fold_train_mean_constant_metrics": baseline_metrics,
        "relative_mae_improvement_over_fold_train_mean": relative_mae,
        "relative_mse_improvement_over_fold_train_mean": relative_mse,
        "exit_probability_roc_auc": auc,
        "side_market_count": side_markets,
        "important_coefficient_count": int(np.sum(important)),
        "median_important_coefficient_sign_agreement": median_sign,
        "cross_fit_gate_checks": checks,
        "cross_fit_gate_passed": not reasons,
        "cross_fit_gate_blocking_reason_codes": reasons,
        "validation_oof_or_future_labels_used": False,
        "result_selected_rerun_allowed": False,
    }


def _flatten_component_coefficients(model: dict[str, Any]) -> np.ndarray:
    values = []
    for name in (
        "exit_probability_component",
        "closed_path_component",
        "residual_path_component",
    ):
        component = model[name]
        values.extend(float(value) for value in component["coefficients"])
        values.append(float(component["intercept"]))
    return np.asarray(values, dtype=np.float64)


def _error_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    errors = predictions - targets
    return {
        "mae": float(np.mean(np.abs(errors))),
        "mse": float(np.mean(errors**2)),
        "mean_error": float(np.mean(errors)),
    }


def _relative_improvement(candidate: float, baseline: float) -> float:
    return (baseline - candidate) / baseline if baseline > 0.0 else 0.0


def _fit_report(
    *,
    config: TwoPartRuntimePNLV65Config,
    profile_path: Path,
    lineage_path: Path,
    corpus_path: Path,
    model_path: Path,
    rows: list[dict[str, Any]],
    lifecycle_reasons: Counter[str],
    model: dict[str, Any],
    cross_fit: dict[str, Any],
) -> dict[str, Any]:
    markets = sorted({str(row["market_id"]) for row in rows})
    split_hash = canonical_json_sha256({"development_train": markets})
    checks = {
        "train_market_support": len(markets) == 89,
        "train_row_support": len(rows) == 712,
        "feature_causality": all(int(row["max_input_ts"]) <= int(row["decision_ts"]) for row in rows),
        "model_finite_and_bounded": model["coefficients_finite"] is True
        and model["coefficients_bounded"] is True,
        "train_only_cross_fit": cross_fit["cross_fit_gate_passed"] is True,
        "consumed_calibration_excluded": True,
        "prior_oof_future_paper_excluded": True,
    }
    reasons = list(cross_fit["cross_fit_gate_blocking_reason_codes"])
    reasons.extend(f"{name}_gate_failed" for name, passed in checks.items() if not passed)
    reasons = sorted(set(reasons))
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        "profile": _descriptor(profile_path),
        "v6_4_lineage_freeze_manifest": _descriptor(lineage_path),
        "train_rows": _descriptor(corpus_path),
        "point_model": _descriptor(model_path),
        "model_sha256": _sha256_file(model_path),
        "policy_dataset_hash": _sha256_file(corpus_path),
        "split_hash": split_hash,
        "train_market_count": len(markets),
        "train_row_count": len(rows),
        "position_lifecycle_class_distribution": dict(
            sorted(Counter(str(row["position_lifecycle_class"]) for row in rows).items())
        ),
        "runtime_policy_residual_reason_distribution": dict(sorted(lifecycle_reasons.items())),
        "cross_fit": cross_fit,
        "point_model_freeze_gate_checks": checks,
        "point_model_freeze_gate_passed": not reasons,
        "point_model_freeze_blocking_reason_codes": reasons,
        "fresh_calibration_boundary": {
            "collector_index_boundary_sequence": 528,
            "collector_index_boundary_sha256": "d5a4eb8c70b79b323c3783ebca722b365d434cca84d3e17cc1f59d8767af71d1",
            "collector_last_entry_sha256": "7af8b5f66f5115c1970d12a17161b0bf250635efb6931cf0516b1e6217cf3472",
            "minimum_market_start_ts_exclusive": 1784541000000,
            "target_quality_valid_market_count": 60,
            "maximum_attempted_market_count": 90,
        },
        "consumed_v6_4_calibration_labels_opened": False,
        "historical_oof_or_prior_future_paper_labels_opened": False,
        "fresh_calibration_outcomes_opened": False,
        "candidate_scoring_frozen": False,
        **_blocked_safety_fields(),
    }


def _report_markdown(report: dict[str, Any]) -> str:
    cross_fit = report["cross_fit"]
    return "\n".join(
        [
            "# #225 v6.5 two-part runtime-PnL point-model freeze",
            "",
            f"- run id: `{report['run_id']}`",
            f"- model SHA-256: `{report['model_sha256']}`",
            f"- policy dataset hash: `{report['policy_dataset_hash']}`",
            f"- split hash: `{report['split_hash']}`",
            f"- train markets/rows: `{report['train_market_count']}/{report['train_row_count']}`",
            f"- cross-fit relative MAE improvement: "
            f"`{cross_fit['relative_mae_improvement_over_fold_train_mean']}`",
            f"- cross-fit relative MSE improvement: "
            f"`{cross_fit['relative_mse_improvement_over_fold_train_mean']}`",
            f"- cross-fit exit AUC: `{cross_fit['exit_probability_roc_auc']}`",
            f"- cross-fit gate passed: `{cross_fit['cross_fit_gate_passed']}`",
            f"- point-model freeze gate passed: `{report['point_model_freeze_gate_passed']}`",
            f"- blockers: `{report['point_model_freeze_blocking_reason_codes']}`",
            "- consumed #224 calibration labels opened: `false`",
            "- historical OOF/prior future/paper labels opened: `false`",
            "- fresh calibration outcomes opened: `false`",
            "- paper/live/write/wallet/capital/promotion unlock: `false`",
        ]
    ) + "\n"


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
