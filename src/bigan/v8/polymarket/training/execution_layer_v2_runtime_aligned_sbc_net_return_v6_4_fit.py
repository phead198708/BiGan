"""Fit and freeze the preregistered #224 runtime-aligned v6.4 overlay."""

from __future__ import annotations

import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
    CANDIDATE_NAME,
    DEVELOPMENT_ROLES,
    FORBIDDEN_FEATURE_TOKENS,
    SIDES,
    TARGET_MANIFEST_SCHEMA_VERSION,
    _blocked_safety_fields,
    _descriptor,
    _flatten_keys,
    _is_sha256,
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

FIT_PROFILE_SCHEMA_VERSION = (
    "bigan-v8-runtime-aligned-sbc-net-return-v6-4-fit-profile-v1"
)
MODEL_SCHEMA_VERSION = "bigan-v8-runtime-aligned-sbc-net-return-v6-4-model-v1"
CALIBRATION_SCHEMA_VERSION = (
    "bigan-v8-runtime-aligned-sbc-net-return-v6-4-conformal-calibration-v1"
)
FIT_REPORT_SCHEMA_VERSION = (
    "bigan-v8-runtime-aligned-sbc-net-return-v6-4-fit-report-v1"
)
CANDIDATE_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-runtime-aligned-sbc-net-return-v6-4-candidate-manifest-v1"
)


@dataclass(frozen=True, slots=True)
class RuntimeAlignedSBCNetReturnV64FitConfig:
    """Pinned inputs for the one fixed #224 fit/calibration run."""

    run_id: str
    output_dir: Path | str
    fit_profile_path: Path | str
    expected_fit_profile_sha256: str
    target_manifest_path: Path | str
    expected_target_manifest_sha256: str
    implementation_commit: str
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(self.expected_fit_profile_sha256, "expected_fit_profile_sha256")
        _require_sha256(self.expected_target_manifest_sha256, "expected_target_manifest_sha256")
        _require_git_sha(self.implementation_commit)
        for name in ("output_dir", "fit_profile_path", "target_manifest_path"):
            object.__setattr__(self, name, Path(getattr(self, name)))


def validate_runtime_aligned_v6_4_fit_profile(profile: dict[str, Any]) -> None:
    """Reject any drift from the no-search #224 fit/calibration contract."""

    lineage = dict(profile.get("source_lineage") or {})
    roles = dict(profile.get("roles") or {})
    model = dict(profile.get("model") or {})
    sentinel = dict(model.get("sentinel_feature_policy") or {})
    stability = dict(profile.get("coefficient_stability") or {})
    calibration = dict(profile.get("calibration") or {})
    freeze = dict(profile.get("candidate_freeze_gate") or {})
    access = dict(profile.get("access_policy") or {})
    checks = {
        "schema": profile.get("schema_version") == FIT_PROFILE_SCHEMA_VERSION,
        "issue": profile.get("issue_number") == 224,
        "candidate": profile.get("candidate_name") == CANDIDATE_NAME,
        "frozen": profile.get("frozen") is True,
        "lineage": len(lineage) == 5
        and all(_is_sha256(str(value)) for value in lineage.values()),
        "roles": roles
        == {
            "model_fit": "development_train",
            "model_fit_market_count": 89,
            "conformal_calibration": "development_calibration",
            "conformal_calibration_market_count": 45,
            "historical_oof_market_count_included": 0,
            "market_disjoint": True,
            "chronological": True,
        },
        "model": model.get("family")
        == "deterministic_l2_ridge_runtime_policy_net_pnl"
        and model.get("target")
        == "runtime_policy_after_cost_net_pnl_per_contract"
        and len(model.get("raw_feature_columns") or []) == 24
        and len(set(model.get("raw_feature_columns") or [])) == 24
        and sentinel.get("fields")
        == ["canonical_v6_2_score", "action_score_margin"]
        and float(sentinel.get("sentinel_maximum")) == -100000.0
        and float(sentinel.get("replacement_value")) == 0.0
        and sentinel.get("availability_indicator_suffix") == "_available"
        and float(model.get("ridge_alpha")) == 1.0
        and model.get("intercept_penalized") is False
        and float(model.get("standardized_feature_clip")) == 8.0
        and float(model.get("coefficient_absolute_bound")) == 8.0
        and model.get("hyperparameter_search_enabled") is False
        and model.get("validation_labels_used_for_model_fit") is False
        and model.get("oof_or_future_labels_used_for_model_fit") is False,
        "stability": stability.get("method")
        == "leave_one_market_out_fixed_train_standardization"
        and float(stability.get("important_coefficient_absolute_minimum")) == 0.02
        and float(
            stability.get("minimum_median_important_coefficient_sign_agreement")
        )
        == 0.65
        and float(stability.get("maximum_median_absolute_coefficient_deviation"))
        == 0.5,
        "calibration": calibration.get("method")
        == "side_specific_market_max_overprediction_split_conformal"
        and calibration.get("cluster_unit") == "market_id"
        and calibration.get("within_market_aggregation")
        == "maximum_overprediction_residual"
        and float(calibration.get("confidence_level")) == 0.9
        and calibration.get("quantile_method")
        == "higher_finite_sample_ceil_n_plus_one"
        and int(calibration.get("minimum_row_count_per_side")) == 160
        and int(calibration.get("minimum_market_count_per_side")) == 40
        and float(calibration.get("minimum_empirical_row_coverage_per_side"))
        == 0.85
        and float(
            calibration.get("minimum_empirical_simultaneous_market_coverage_per_side")
        )
        == 0.85
        and calibration.get("constant_baseline")
        == "development_train_target_mean"
        and float(
            calibration.get("minimum_relative_mae_improvement_over_constant_exclusive")
        )
        == 0.0
        and float(
            calibration.get("minimum_relative_mse_improvement_over_constant_exclusive")
        )
        == 0.0
        and calibration.get("threshold_search_enabled") is False
        and calibration.get("calibration_policy_pnl_computed") is False
        and calibration.get("oof_or_future_labels_used") is False,
        "freeze_gate": freeze.get("decision_rule")
        == "retain_frozen_v6_2_sbc_only_if_runtime_policy_net_pnl_lcb_strictly_positive"
        and float(freeze.get("minimum_runtime_policy_net_pnl_lcb_exclusive")) == 0.0
        and int(freeze.get("minimum_positive_lcb_row_count")) == 20
        and int(freeze.get("minimum_positive_lcb_unique_market_count")) == 10
        and int(freeze.get("minimum_positive_lcb_unique_market_count_per_side")) == 3
        and freeze.get("full_execution_guard_unchanged") is True
        and freeze.get("cost_model_unchanged") is True
        and freeze.get("sizing_and_position_manager_unchanged") is True
        and freeze.get("result_selected_threshold_allowed") is False,
        "access": access.get("fit_stage_roles_only") == list(DEVELOPMENT_ROLES)
        and all(
            access.get(name) is False
            for name in (
                "issue_223_oof_opened",
                "issue_212_future_outcomes_opened",
                "issue_221_paper_outcomes_opened",
                "issue_192_prefreeze_rows_opened",
                "new_future_holdout_outcomes_opened",
            )
        ),
        "safety": profile.get("safety") == _blocked_safety_fields(),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("#224 fit profile invalid: " + ", ".join(blockers))


def run_runtime_aligned_sbc_net_return_v6_4_fit(
    config: RuntimeAlignedSBCNetReturnV64FitConfig,
) -> dict[str, Any]:
    """Fit train-only ridge, calibrate market-clustered LCB, and freeze or block."""

    profile_path = Path(config.fit_profile_path).resolve()
    target_manifest_path = Path(config.target_manifest_path).resolve()
    _verify_pin(profile_path, config.expected_fit_profile_sha256, "#224 fit profile")
    _verify_pin(
        target_manifest_path,
        config.expected_target_manifest_sha256,
        "#224 target manifest",
    )
    profile = _load_json(profile_path)
    validate_runtime_aligned_v6_4_fit_profile(profile)
    target_manifest = _load_json(target_manifest_path)
    rows_path = _validate_target_lineage(
        target_manifest,
        target_manifest_path=target_manifest_path,
        profile=profile,
    )
    rows = _load_jsonl(rows_path)
    split = _validate_rows(rows, profile=profile)

    run_dir = Path(config.output_dir).resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    train_rows = [row for row in rows if row["role"] == "development_train"]
    calibration_rows = [
        row for row in rows if row["role"] == "development_calibration"
    ]
    model = _fit_model(train_rows, profile=profile)
    model["implementation_commit"] = config.implementation_commit
    model["profile_sha256"] = config.expected_fit_profile_sha256
    model["policy_dataset_hash"] = _sha256_file(rows_path)
    model["split_hash"] = split["split_hash"]
    model["model_artifact_id"] = canonical_json_sha256(model)
    model_path = run_dir / "v6_4_runtime_policy_net_pnl_model.json"
    _write_json(model_path, model)

    train_predictions = score_runtime_aligned_sbc_rows(train_rows, model=model)
    calibration_predictions = score_runtime_aligned_sbc_rows(
        calibration_rows, model=model
    )
    stability = _coefficient_stability(train_rows, model=model, profile=profile)
    calibration = _build_calibration(
        train_rows=train_rows,
        train_predictions=train_predictions,
        calibration_rows=calibration_rows,
        calibration_predictions=calibration_predictions,
        model=model,
        stability=stability,
        profile=profile,
    )
    calibration_path = run_dir / "v6_4_runtime_policy_market_conformal_calibration.json"
    _write_json(calibration_path, calibration)
    _write_text(
        calibration_path.with_suffix(".md"), _calibration_markdown(calibration)
    )

    scored_calibration = apply_runtime_policy_lcb(
        calibration_predictions,
        calibration=calibration,
    )
    scored_path = run_dir / "v6_4_calibration_runtime_policy_lcb_rows.jsonl"
    _write_jsonl(scored_path, scored_calibration)
    freeze_gate = _candidate_freeze_gate(
        scored_calibration,
        model=model,
        calibration=calibration,
        stability=stability,
        profile=profile,
    )
    report = _fit_report(
        config=config,
        profile=profile,
        profile_path=profile_path,
        target_manifest_path=target_manifest_path,
        rows_path=rows_path,
        split=split,
        model_path=model_path,
        model=model,
        calibration_path=calibration_path,
        calibration=calibration,
        stability=stability,
        scored_path=scored_path,
        freeze_gate=freeze_gate,
    )
    report_path = run_dir / "v6_4_runtime_policy_fit_report.json"
    _write_json(report_path, report)
    _write_text(report_path.with_suffix(".md"), _report_markdown(report))

    manifest = {
        "schema_version": CANDIDATE_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        "fit_profile": _descriptor(profile_path),
        "target_manifest": _descriptor(target_manifest_path),
        "runtime_aligned_rows": _descriptor(rows_path),
        "model": _descriptor(model_path),
        "calibration": _descriptor(calibration_path),
        "calibration_lcb_rows": _descriptor(scored_path),
        "fit_report": _descriptor(report_path),
        "model_sha256": _sha256_file(model_path),
        "policy_dataset_hash": _sha256_file(rows_path),
        "split_hash": split["split_hash"],
        "candidate_scoring_frozen": freeze_gate["candidate_freeze_gate_passed"],
        "candidate_freeze_gate_passed": freeze_gate["candidate_freeze_gate_passed"],
        "candidate_freeze_blocking_reason_codes": freeze_gate[
            "candidate_freeze_blocking_reason_codes"
        ],
        "outcome_blind_future_collection_resume_allowed": freeze_gate[
            "candidate_freeze_gate_passed"
        ],
        "future_unseen_side_only_pnl_gate_required": True,
        "historical_oof_opened": False,
        "issue_212_future_outcomes_opened": False,
        "issue_221_paper_outcomes_opened": False,
        "result_selected_threshold_used": False,
        **_blocked_safety_fields(),
    }
    manifest["candidate_manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v6_4_runtime_policy_candidate_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "candidate_manifest_path": manifest_path,
        "candidate_manifest_sha256": _sha256_file(manifest_path),
        "report": report,
        "manifest": manifest,
    }


def score_runtime_aligned_sbc_rows(
    rows: list[dict[str, Any]], *, model: dict[str, Any]
) -> list[dict[str, Any]]:
    """Apply the frozen decision-time transform and ridge point model."""

    columns = list(model["raw_feature_columns"])
    transformed_names = list(model["model_feature_columns"])
    matrix, observed_names = _feature_matrix(rows, model["sentinel_feature_policy"], columns)
    if observed_names != transformed_names:
        raise ValueError("v6.4 model feature contract mismatch")
    means = np.asarray(model["feature_means"], dtype=np.float64)
    scales = np.asarray(model["feature_scales"], dtype=np.float64)
    standardized = np.clip(
        (matrix - means) / scales,
        -float(model["standardized_feature_clip"]),
        float(model["standardized_feature_clip"]),
    )
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    predictions = float(model["intercept"]) + standardized @ coefficients
    return [
        {
            **row,
            "runtime_policy_point_predicted_net_pnl_per_contract": float(value),
            "runtime_policy_prediction_source": (
                "frozen_train_only_deterministic_l2_ridge"
            ),
            "target_fields_used_for_prediction": False,
        }
        for row, value in zip(rows, predictions, strict=True)
    ]


def apply_runtime_policy_lcb(
    rows: list[dict[str, Any]], *, calibration: dict[str, Any]
) -> list[dict[str, Any]]:
    """Apply the frozen side-specific market-clustered conformal correction."""

    output = []
    for row in rows:
        side = str(row["side"])
        correction = float(calibration["sides"][side]["overprediction_correction"])
        point = float(row["runtime_policy_point_predicted_net_pnl_per_contract"])
        lcb = point - correction
        source_score = float(row["features"]["canonical_v6_2_score"])
        precondition = source_score > 0.0
        output.append(
            {
                **row,
                "runtime_policy_overprediction_correction": correction,
                "runtime_policy_net_pnl_lcb": lcb,
                "frozen_v6_2_sbc_precondition_passed": precondition,
                "runtime_aligned_candidate_retained": precondition and lcb > 0.0,
            }
        )
    return output


def _validate_target_lineage(
    target_manifest: dict[str, Any],
    *,
    target_manifest_path: Path,
    profile: dict[str, Any],
) -> Path:
    if target_manifest.get("schema_version") != TARGET_MANIFEST_SCHEMA_VERSION:
        raise ValueError("#224 target manifest schema mismatch")
    if target_manifest.get("target_corpus_gate_passed") is not True:
        raise ValueError("#224 target corpus gate did not pass")
    lineage = profile["source_lineage"]
    if _sha256_file(target_manifest_path) != lineage["target_manifest_sha256"]:
        raise ValueError("#224 target manifest profile lineage mismatch")
    rows = _verified_descriptor(
        target_manifest.get("runtime_aligned_rows"), "runtime-aligned rows"
    )
    if rows["sha256"] != lineage["target_rows_sha256"]:
        raise ValueError("#224 target rows profile lineage mismatch")
    target_profile = _verified_descriptor(target_manifest.get("profile"), "target profile")
    if target_profile["sha256"] != lineage["target_profile_sha256"]:
        raise ValueError("#224 target profile lineage mismatch")
    freeze = _verified_descriptor(
        target_manifest.get("lineage_freeze_manifest"), "lineage freeze"
    )
    if freeze["sha256"] != lineage["lineage_freeze_manifest_sha256"]:
        raise ValueError("#224 lineage freeze profile mismatch")
    return Path(rows["path"])


def _validate_rows(rows: list[dict[str, Any]], *, profile: dict[str, Any]) -> dict[str, Any]:
    if len(rows) != 1072:
        raise ValueError("#224 target row count mismatch")
    roles = Counter(str(row.get("role")) for row in rows)
    if roles != Counter({"development_train": 712, "development_calibration": 360}):
        raise ValueError("#224 target split row count mismatch")
    market_roles: dict[str, str] = {}
    forbidden = set()
    for row in rows:
        market_id = str(row.get("market_id") or "")
        role = str(row.get("role") or "")
        side = str(row.get("side") or "")
        if not market_id or role not in DEVELOPMENT_ROLES or side not in SIDES:
            raise ValueError("#224 target identity invalid")
        if market_id in market_roles and market_roles[market_id] != role:
            raise ValueError("#224 target market split overlap")
        market_roles[market_id] = role
        if int(row["max_input_ts"]) > int(row["decision_ts"]):
            raise ValueError("#224 fit feature causality violation")
        target = float(row[profile["model"]["target"]])
        if not math.isfinite(target):
            raise ValueError("#224 fit target is not finite")
        features = dict(row.get("features") or {})
        for name in profile["model"]["raw_feature_columns"]:
            value = float(features[name])
            if not math.isfinite(value):
                raise ValueError(f"#224 fit feature is not finite: {name}")
        forbidden.update(
            key
            for key in _flatten_keys(features)
            if any(token in key.lower() for token in FORBIDDEN_FEATURE_TOKENS)
        )
    if forbidden:
        raise ValueError("#224 fit forbidden decision-time feature fields")
    train_markets = sorted(
        market for market, role in market_roles.items() if role == "development_train"
    )
    calibration_markets = sorted(
        market
        for market, role in market_roles.items()
        if role == "development_calibration"
    )
    if len(train_markets) != 89 or len(calibration_markets) != 45:
        raise ValueError("#224 fit market support mismatch")
    train_max = max(
        int(row["decision_ts"]) for row in rows if row["role"] == "development_train"
    )
    calibration_min = min(
        int(row["decision_ts"])
        for row in rows
        if row["role"] == "development_calibration"
    )
    if train_max >= calibration_min:
        raise ValueError("#224 fit split is not chronological")
    split_payload = {
        "development_train": train_markets,
        "development_calibration": calibration_markets,
    }
    return {
        "split_hash": canonical_json_sha256(split_payload),
        "train_market_count": len(train_markets),
        "calibration_market_count": len(calibration_markets),
        "train_max_decision_ts": train_max,
        "calibration_min_decision_ts": calibration_min,
        "market_disjoint": set(train_markets).isdisjoint(calibration_markets),
        "chronological": train_max < calibration_min,
    }


def _fit_model(rows: list[dict[str, Any]], *, profile: dict[str, Any]) -> dict[str, Any]:
    config = profile["model"]
    raw_columns = list(config["raw_feature_columns"])
    matrix, model_columns = _feature_matrix(
        rows, config["sentinel_feature_policy"], raw_columns
    )
    targets = np.asarray(
        [float(row[config["target"]]) for row in rows], dtype=np.float64
    )
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
    bound = float(config["coefficient_absolute_bound"])
    finite = bool(
        np.all(np.isfinite(coefficients)) and math.isfinite(float(intercept))
    )
    bounded = bool(np.max(np.abs(coefficients)) <= bound and abs(intercept) <= bound)
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "model_family": config["family"],
        "target": config["target"],
        "fit_role": "development_train",
        "fit_market_count": len({str(row["market_id"]) for row in rows}),
        "fit_row_count": len(rows),
        "raw_feature_columns": raw_columns,
        "model_feature_columns": model_columns,
        "sentinel_feature_policy": config["sentinel_feature_policy"],
        "feature_means": means.tolist(),
        "feature_scales": scales.tolist(),
        "standardized_feature_clip": float(config["standardized_feature_clip"]),
        "ridge_alpha": float(config["ridge_alpha"]),
        "intercept_penalized": False,
        "coefficients": coefficients.tolist(),
        "intercept": float(intercept),
        "coefficient_absolute_bound": bound,
        "coefficients_finite": finite,
        "coefficients_bounded": bounded,
        "hyperparameter_search_enabled": False,
        "validation_labels_used_for_model_fit": False,
        "oof_or_future_labels_used_for_model_fit": False,
        "target_fields_used_as_model_inputs": False,
    }


def _feature_matrix(
    rows: list[dict[str, Any]],
    sentinel_policy: dict[str, Any],
    raw_columns: list[str],
) -> tuple[np.ndarray, list[str]]:
    sentinel_fields = list(sentinel_policy["fields"])
    threshold = float(sentinel_policy["sentinel_maximum"])
    replacement = float(sentinel_policy["replacement_value"])
    suffix = str(sentinel_policy["availability_indicator_suffix"])
    names = [*raw_columns, *(f"{name}{suffix}" for name in sentinel_fields)]
    output = []
    for row in rows:
        features = dict(row["features"])
        values = []
        indicators = []
        for name in raw_columns:
            value = float(features[name])
            if name in sentinel_fields:
                available = value > threshold
                indicators.append(float(available))
                value = value if available else replacement
            values.append(value)
        output.append([*values, *indicators])
    return np.asarray(output, dtype=np.float64), names


def _ridge_fit(
    matrix: np.ndarray, targets: np.ndarray, *, alpha: float
) -> tuple[np.ndarray, float]:
    design = np.column_stack((np.ones(len(matrix), dtype=np.float64), matrix))
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    parameters = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return parameters[1:], float(parameters[0])


def _coefficient_stability(
    rows: list[dict[str, Any]], *, model: dict[str, Any], profile: dict[str, Any]
) -> dict[str, Any]:
    config = profile["coefficient_stability"]
    matrix, _ = _feature_matrix(
        rows, model["sentinel_feature_policy"], model["raw_feature_columns"]
    )
    means = np.asarray(model["feature_means"], dtype=np.float64)
    scales = np.asarray(model["feature_scales"], dtype=np.float64)
    standardized = np.clip(
        (matrix - means) / scales,
        -float(model["standardized_feature_clip"]),
        float(model["standardized_feature_clip"]),
    )
    targets = np.asarray([float(row[model["target"]]) for row in rows])
    markets: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        markets[str(row["market_id"])].append(index)
    full = np.asarray(model["coefficients"], dtype=np.float64)
    replicates = []
    for market in sorted(markets):
        excluded = set(markets[market])
        indices = [index for index in range(len(rows)) if index not in excluded]
        coefficients, _ = _ridge_fit(
            standardized[indices],
            targets[indices],
            alpha=float(model["ridge_alpha"]),
        )
        replicates.append(coefficients)
    samples = np.asarray(replicates)
    important = np.abs(full) >= float(config["important_coefficient_absolute_minimum"])
    sign_agreement = np.mean(np.sign(samples) == np.sign(full), axis=0)
    deviations = np.median(np.abs(samples - full), axis=0)
    important_present = bool(np.any(important))
    median_sign = (
        float(np.median(sign_agreement[important])) if important_present else 1.0
    )
    median_deviation = (
        float(np.median(deviations[important]))
        if important_present
        else float(np.max(deviations))
    )
    passed = bool(
        np.all(np.isfinite(samples))
        and median_sign
        >= float(config["minimum_median_important_coefficient_sign_agreement"])
        and median_deviation
        <= float(config["maximum_median_absolute_coefficient_deviation"])
    )
    return {
        "method": config["method"],
        "replicate_count": len(replicates),
        "important_coefficient_count": int(np.sum(important)),
        "no_important_coefficients_treated_as_low_complexity": not important_present,
        "median_important_coefficient_sign_agreement": median_sign,
        "minimum_required_sign_agreement": float(
            config["minimum_median_important_coefficient_sign_agreement"]
        ),
        "median_important_absolute_coefficient_deviation": median_deviation,
        "maximum_allowed_median_absolute_coefficient_deviation": float(
            config["maximum_median_absolute_coefficient_deviation"]
        ),
        "coefficient_stability_gate_passed": passed,
    }


def _build_calibration(
    *,
    train_rows: list[dict[str, Any]],
    train_predictions: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    calibration_predictions: list[dict[str, Any]],
    model: dict[str, Any],
    stability: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    config = profile["calibration"]
    target_field = model["target"]
    baseline = float(np.mean([float(row[target_field]) for row in train_rows]))
    prediction_values = np.asarray(
        [row["runtime_policy_point_predicted_net_pnl_per_contract"] for row in calibration_predictions]
    )
    targets = np.asarray([float(row[target_field]) for row in calibration_rows])
    baseline_values = np.full(len(targets), baseline)
    point_metrics = _error_metrics(targets, prediction_values)
    baseline_metrics = _error_metrics(targets, baseline_values)
    relative_mae = _relative_improvement(point_metrics["mae"], baseline_metrics["mae"])
    relative_mse = _relative_improvement(point_metrics["mse"], baseline_metrics["mse"])
    sides: dict[str, Any] = {}
    side_checks = {}
    for side in SIDES:
        side_rows = [row for row in calibration_predictions if row["side"] == side]
        by_market: dict[str, list[float]] = defaultdict(list)
        for row in side_rows:
            residual = float(row["runtime_policy_point_predicted_net_pnl_per_contract"]) - float(
                row[target_field]
            )
            by_market[str(row["market_id"])].append(residual)
        market_max = {market: max(values) for market, values in by_market.items()}
        correction = _finite_sample_higher_quantile(
            list(market_max.values()), float(config["confidence_level"])
        )
        covered_rows = [
            float(row[target_field])
            >= float(row["runtime_policy_point_predicted_net_pnl_per_contract"]) - correction
            for row in side_rows
        ]
        covered_markets = [
            all(
                float(row[target_field])
                >= float(row["runtime_policy_point_predicted_net_pnl_per_contract"])
                - correction
                for row in side_rows
                if str(row["market_id"]) == market
            )
            for market in sorted(by_market)
        ]
        row_coverage = float(np.mean(covered_rows)) if covered_rows else 0.0
        market_coverage = float(np.mean(covered_markets)) if covered_markets else 0.0
        checks = {
            "row_support": len(side_rows) >= int(config["minimum_row_count_per_side"]),
            "market_support": len(by_market)
            >= int(config["minimum_market_count_per_side"]),
            "row_coverage": row_coverage
            >= float(config["minimum_empirical_row_coverage_per_side"]),
            "simultaneous_market_coverage": market_coverage
            >= float(
                config["minimum_empirical_simultaneous_market_coverage_per_side"]
            ),
        }
        side_checks[side] = checks
        sides[side] = {
            "row_count": len(side_rows),
            "market_count": len(by_market),
            "market_max_overprediction_residuals": market_max,
            "overprediction_correction": correction,
            "empirical_row_coverage": row_coverage,
            "empirical_simultaneous_market_coverage": market_coverage,
            "calibration_checks": checks,
            "calibration_gate_passed": all(checks.values()),
        }
    checks = {
        "model_coefficients_finite": model["coefficients_finite"] is True,
        "model_coefficients_bounded": model["coefficients_bounded"] is True,
        "coefficient_stability": stability["coefficient_stability_gate_passed"] is True,
        "relative_mae_improvement": relative_mae
        > float(config["minimum_relative_mae_improvement_over_constant_exclusive"]),
        "relative_mse_improvement": relative_mse
        > float(config["minimum_relative_mse_improvement_over_constant_exclusive"]),
        "side_conformal_support_and_coverage": all(
            all(values.values()) for values in side_checks.values()
        ),
        "no_threshold_search": config["threshold_search_enabled"] is False,
        "calibration_policy_pnl_not_computed": config["calibration_policy_pnl_computed"]
        is False,
        "oof_or_future_labels_not_used": config["oof_or_future_labels_used"] is False,
    }
    reasons = [
        f"{name}_gate_failed" for name, passed in checks.items() if not passed
    ]
    artifact = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "method": config["method"],
        "fit_role": "development_train",
        "calibration_role": "development_calibration",
        "confidence_level": float(config["confidence_level"]),
        "cluster_unit": "market_id",
        "within_market_aggregation": "maximum_overprediction_residual",
        "quantile_method": config["quantile_method"],
        "constant_baseline_value": baseline,
        "point_model_metrics": point_metrics,
        "constant_baseline_metrics": baseline_metrics,
        "relative_mae_improvement_over_constant": relative_mae,
        "relative_mse_improvement_over_constant": relative_mse,
        "sides": sides,
        "coefficient_stability": stability,
        "calibration_gate_checks": checks,
        "calibration_gate_passed": not reasons,
        "calibration_gate_blocking_reason_codes": reasons,
        "threshold_search_enabled": False,
        "calibration_policy_pnl_computed": False,
        "oof_or_future_labels_used": False,
        "validation_labels_used_for_model_fit": False,
    }
    artifact["calibration_artifact_id"] = canonical_json_sha256(artifact)
    return artifact


def _finite_sample_higher_quantile(values: list[float], confidence: float) -> float:
    if not values:
        raise ValueError("split conformal requires market residuals")
    ordered = sorted(float(value) for value in values)
    rank = min(len(ordered), math.ceil((len(ordered) + 1) * confidence))
    return ordered[rank - 1]


def _error_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    errors = predictions - targets
    return {
        "mae": float(np.mean(np.abs(errors))),
        "mse": float(np.mean(errors**2)),
        "mean_error": float(np.mean(errors)),
    }


def _relative_improvement(candidate: float, baseline: float) -> float:
    return (baseline - candidate) / baseline if baseline > 0.0 else 0.0


def _candidate_freeze_gate(
    rows: list[dict[str, Any]],
    *,
    model: dict[str, Any],
    calibration: dict[str, Any],
    stability: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    config = profile["candidate_freeze_gate"]
    selected = [row for row in rows if row["runtime_aligned_candidate_retained"]]
    markets = {str(row["market_id"]) for row in selected}
    side_markets = {
        side: len({str(row["market_id"]) for row in selected if row["side"] == side})
        for side in SIDES
    }
    checks = {
        "model_finite_and_bounded": model["coefficients_finite"] is True
        and model["coefficients_bounded"] is True,
        "coefficient_stability": stability["coefficient_stability_gate_passed"] is True,
        "calibration": calibration["calibration_gate_passed"] is True,
        "positive_lcb_row_support": len(selected)
        >= int(config["minimum_positive_lcb_row_count"]),
        "positive_lcb_market_support": len(markets)
        >= int(config["minimum_positive_lcb_unique_market_count"]),
        "positive_lcb_side_market_support": all(
            side_markets[side]
            >= int(config["minimum_positive_lcb_unique_market_count_per_side"])
            for side in SIDES
        ),
        "decision_boundary_frozen_at_zero": float(
            config["minimum_runtime_policy_net_pnl_lcb_exclusive"]
        )
        == 0.0,
        "execution_contract_unchanged": all(
            config[name] is True
            for name in (
                "full_execution_guard_unchanged",
                "cost_model_unchanged",
                "sizing_and_position_manager_unchanged",
            )
        ),
        "no_result_selected_threshold": config["result_selected_threshold_allowed"]
        is False,
    }
    reasons = list(calibration["calibration_gate_blocking_reason_codes"])
    reasons.extend(
        f"{name}_gate_failed" for name, passed in checks.items() if not passed
    )
    reasons = sorted(set(reasons))
    return {
        "candidate_freeze_gate_checks": checks,
        "positive_lcb_row_count": len(selected),
        "positive_lcb_unique_market_count": len(markets),
        "positive_lcb_unique_market_count_by_side": side_markets,
        "positive_lcb_action_distribution": dict(
            sorted(Counter(str(row["action"]) for row in selected).items())
        ),
        "candidate_freeze_gate_passed": not reasons,
        "candidate_freeze_blocking_reason_codes": reasons,
    }


def _fit_report(
    *,
    config: RuntimeAlignedSBCNetReturnV64FitConfig,
    profile: dict[str, Any],
    profile_path: Path,
    target_manifest_path: Path,
    rows_path: Path,
    split: dict[str, Any],
    model_path: Path,
    model: dict[str, Any],
    calibration_path: Path,
    calibration: dict[str, Any],
    stability: dict[str, Any],
    scored_path: Path,
    freeze_gate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": FIT_REPORT_SCHEMA_VERSION,
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        "fit_profile": _descriptor(profile_path),
        "target_manifest": _descriptor(target_manifest_path),
        "runtime_aligned_rows": _descriptor(rows_path),
        "model": _descriptor(model_path),
        "calibration": _descriptor(calibration_path),
        "calibration_lcb_rows": _descriptor(scored_path),
        "model_sha256": _sha256_file(model_path),
        "policy_dataset_hash": _sha256_file(rows_path),
        "split_hash": split["split_hash"],
        "split_validation": split,
        "model_family": model["model_family"],
        "target": model["target"],
        "fit_market_count": model["fit_market_count"],
        "fit_row_count": model["fit_row_count"],
        "calibration_market_count": split["calibration_market_count"],
        "calibration_row_count": 360,
        "sentinel_feature_policy": model["sentinel_feature_policy"],
        "coefficients_finite": model["coefficients_finite"],
        "coefficients_bounded": model["coefficients_bounded"],
        "coefficient_stability": stability,
        "calibration_gate_passed": calibration["calibration_gate_passed"],
        "calibration_gate_blocking_reason_codes": calibration[
            "calibration_gate_blocking_reason_codes"
        ],
        "point_model_metrics": calibration["point_model_metrics"],
        "constant_baseline_metrics": calibration["constant_baseline_metrics"],
        "relative_mae_improvement_over_constant": calibration[
            "relative_mae_improvement_over_constant"
        ],
        "relative_mse_improvement_over_constant": calibration[
            "relative_mse_improvement_over_constant"
        ],
        "conformal_sides": calibration["sides"],
        **freeze_gate,
        "candidate_scoring_frozen": freeze_gate["candidate_freeze_gate_passed"],
        "future_unseen_side_only_pnl_gate_required": True,
        "historical_oof_opened": False,
        "issue_212_future_outcomes_opened": False,
        "issue_221_paper_outcomes_opened": False,
        "threshold_search_enabled": False,
        "calibration_policy_pnl_computed": False,
        "action_and_family_metrics_diagnostic_only": True,
        **_blocked_safety_fields(),
    }


def _calibration_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# #224 v6.4 market-clustered conformal calibration",
        "",
        f"- calibration gate passed: `{report['calibration_gate_passed']}`",
        f"- blockers: `{report['calibration_gate_blocking_reason_codes']}`",
        f"- point metrics: `{report['point_model_metrics']}`",
        f"- constant baseline metrics: `{report['constant_baseline_metrics']}`",
        "- threshold search enabled: `false`",
        "- calibration policy PnL computed: `false`",
        "- OOF/future labels used: `false`",
        "",
        "## Side corrections",
    ]
    for side in SIDES:
        value = report["sides"][side]
        lines.extend(
            [
                f"- {side}: correction `{value['overprediction_correction']}`, "
                f"row coverage `{value['empirical_row_coverage']}`, market coverage "
                f"`{value['empirical_simultaneous_market_coverage']}`",
            ]
        )
    return "\n".join(lines) + "\n"


def _report_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# #224 runtime-aligned SBC net-PnL v6.4 fit",
            "",
            f"- run id: `{report['run_id']}`",
            f"- model SHA-256: `{report['model_sha256']}`",
            f"- policy dataset hash: `{report['policy_dataset_hash']}`",
            f"- split hash: `{report['split_hash']}`",
            f"- fit/calibration markets: `{report['fit_market_count']}/"
            f"{report['calibration_market_count']}`",
            f"- relative MAE improvement: "
            f"`{report['relative_mae_improvement_over_constant']}`",
            f"- relative MSE improvement: "
            f"`{report['relative_mse_improvement_over_constant']}`",
            f"- calibration gate passed: `{report['calibration_gate_passed']}`",
            f"- positive-LCB rows/markets: `{report['positive_lcb_row_count']}/"
            f"{report['positive_lcb_unique_market_count']}`",
            f"- positive-LCB markets by side: "
            f"`{report['positive_lcb_unique_market_count_by_side']}`",
            f"- candidate freeze gate passed: "
            f"`{report['candidate_freeze_gate_passed']}`",
            f"- blockers: `{report['candidate_freeze_blocking_reason_codes']}`",
            "- historical OOF/future/paper outcomes opened: `false`",
            "- paper/live/write/wallet/capital/promotion unlock: `false`",
        ]
    ) + "\n"
