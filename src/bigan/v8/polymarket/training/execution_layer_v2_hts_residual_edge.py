"""Development-only HTS residual-edge and market-level power diagnostics.

The workflow consumes an exhausted probability-estimand goal.  Every previously
inspected row is treated as development evidence and is permanently excluded
from future confirmatory validation.  Market probability remains a fixed logit
offset; candidate models can only learn a regularized residual correction.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any

from bigan.v8.polymarket.training.execution_layer_v2_estimand_reformulation import (
    SUPPORTED_FAMILY,
    _normalize_development_rows,
    safety_fields,
    sha256_file,
)

SCHEMA_PREFIX = "bigan-v8-hts-residual-edge"
MODEL_FAMILY = "market_probability_logit_offset_regularized_residual"
DEFAULT_CONFIRMATORY_CONFIDENCE_LEVEL = 1.0 - (0.05 / 3.0)
DEFAULT_MINIMUM_RELATIVE_IMPROVEMENT = 0.03


@dataclass(frozen=True, slots=True)
class HTSResidualEdgePowerConfig:
    run_id: str
    output_dir: Path | str
    repository_root: Path | str
    source_estimand_goal_dir: Path | str
    created_at: str
    bootstrap_samples: int = 2_000
    bootstrap_confidence_level: float = DEFAULT_CONFIRMATORY_CONFIDENCE_LEVEL
    minimum_training_runs: int = 3
    minimum_relative_brier_improvement: float = (
        DEFAULT_MINIMUM_RELATIVE_IMPROVEMENT
    )
    minimum_relative_log_loss_improvement: float = (
        DEFAULT_MINIMUM_RELATIVE_IMPROVEMENT
    )
    minimum_prospective_market_count: int = 100
    target_power_levels: tuple[float, ...] = (0.80, 0.90)
    random_seed: int = 17041

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        for field in (
            "output_dir",
            "repository_root",
            "source_estimand_goal_dir",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)).resolve())
        if self.bootstrap_samples < 20:
            raise ValueError("bootstrap_samples must be at least 20")
        if not 0.5 < self.bootstrap_confidence_level < 1.0:
            raise ValueError("bootstrap_confidence_level must be between 0.5 and 1")
        if self.minimum_training_runs < 2:
            raise ValueError("minimum_training_runs must be at least 2")
        if self.minimum_prospective_market_count < 25:
            raise ValueError("minimum_prospective_market_count must be at least 25")
        if not self.target_power_levels or any(
            not 0.5 < value < 1.0 for value in self.target_power_levels
        ):
            raise ValueError("target_power_levels must be between 0.5 and 1")

    @property
    def analysis_dir(self) -> Path:
        return Path(self.output_dir) / self.run_id / "hts_residual_edge_power_analysis"

    def payload(self) -> dict[str, Any]:
        payload = asdict(self)
        for field in (
            "output_dir",
            "repository_root",
            "source_estimand_goal_dir",
        ):
            payload[field] = str(payload[field])
        payload["target_power_levels"] = list(self.target_power_levels)
        payload.update(
            {
                "schema_version": f"{SCHEMA_PREFIX}-config-v1",
                "analysis_mode": "post_validation_development_only",
                "new_confirmatory_validation_started": False,
                "prospective_collection_started": False,
                "market_probability_offset_coefficient": 1.0,
                "market_probability_offset_trainable": False,
                "residual_coefficients_shrinkage_target": 0.0,
                **safety_fields(),
            }
        )
        return payload


def run_hts_residual_edge_power_analysis(
    config: HTSResidualEdgePowerConfig,
) -> dict[str, Any]:
    output_dir = config.analysis_dir
    if output_dir.exists():
        raise FileExistsError(f"analysis directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    source_goal = Path(config.source_estimand_goal_dir)
    source_report_path = source_goal / "pre_promotion_readiness_report.json"
    source_manifest_path = source_goal / "pre_promotion_readiness_manifest.json"
    source_report = _load_json(source_report_path)
    _verify_sha_descriptor(source_manifest_path)
    _verify_source_report_lineage(source_report_path, source_manifest_path)
    _validate_exhausted_source_goal(source_report)

    config_payload = config.payload()
    config_payload.update(
        {
            "source_goal_report": _descriptor(source_report_path),
            "source_goal_manifest": _descriptor(source_manifest_path),
            "source_goal_final_state": source_report["final_state"],
            "source_goal_blocking_reason_codes": source_report[
                "blocking_reason_codes"
            ],
            "source_tree_commit": _git_head(Path(config.repository_root)),
        }
    )
    _write_hashed_json(output_dir, "hts_residual_edge_analysis_configuration", config_payload)

    rows, lineage = _load_post_validation_development_rows(source_goal)
    if not rows:
        raise ValueError("source goal produced no HTS development rows")
    combined_rows_path = output_dir / "hts_post_validation_development_rows.jsonl"
    _write_jsonl(combined_rows_path, rows)
    _write_sha_descriptor(combined_rows_path)

    development_manifest = _build_development_manifest(
        rows=rows,
        lineage=lineage,
        source_report_path=source_report_path,
        source_manifest_path=source_manifest_path,
        combined_rows_path=combined_rows_path,
    )
    _write_json(
        output_dir / "hts_incremental_edge_development_manifest.json",
        development_manifest,
    )

    candidate_specs = _candidate_specs()
    protocol = _candidate_protocol(config, candidate_specs, development_manifest)
    protocol_path, protocol_hash_path = _write_hashed_json(
        output_dir,
        "hts_residual_offset_candidate_protocol",
        protocol,
    )

    candidate_reports = [
        _evaluate_candidate_oof(
            rows,
            spec,
            minimum_training_runs=config.minimum_training_runs,
        )
        for spec in candidate_specs
    ]
    candidate_reports.sort(key=_candidate_ranking_key)
    selected = candidate_reports[0]
    selected_spec = next(
        spec
        for spec in candidate_specs
        if spec["candidate_name"] == selected["candidate_name"]
    )
    selected_contract = fit_residual_offset_contract(rows, selected_spec)
    development_gate = _development_candidate_gate(selected, config)

    diagnostic_report = _diagnostic_report(
        rows=rows,
        candidate_reports=candidate_reports,
        selected=selected,
        selected_contract=selected_contract,
        development_gate=development_gate,
        protocol_path=protocol_path,
        combined_rows_path=combined_rows_path,
    )
    diagnostic_json = output_dir / "hts_incremental_edge_diagnostic_report.json"
    diagnostic_md = output_dir / "hts_incremental_edge_diagnostic_report.md"
    _write_json(diagnostic_json, diagnostic_report)
    _write_text(diagnostic_md, _diagnostic_markdown(diagnostic_report))
    selected_contract_payload = {
        **selected_contract,
        "contract_scope": "development_diagnostic_only",
        "selected_from_post_validation_development_only": True,
        "selected_candidate_development_gate": development_gate,
        "prospective_collection_candidate_eligible": development_gate["passed"],
        "fresh_confirmatory_validation_start_allowed": False,
        "future_market_disjoint_confirmatory_validation_required": True,
        "candidate_protocol_sha256": sha256_file(protocol_path),
        "development_rows_sha256": sha256_file(combined_rows_path),
    }
    selected_contract_path, _ = _write_hashed_json(
        output_dir,
        "hts_residual_offset_selected_candidate_contract",
        selected_contract_payload,
    )

    power_report = _power_report(
        selected=selected,
        config=config,
        development_gate=development_gate,
    )
    power_json = output_dir / "hts_market_level_power_report.json"
    power_md = output_dir / "hts_market_level_power_report.md"
    _write_json(power_json, power_report)
    _write_text(power_md, _power_markdown(power_report))

    artifacts = [
        {
            "relative_path": str(path.relative_to(output_dir)),
            "sha256": sha256_file(path),
        }
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
        and path.name
        not in {
            "hts_residual_edge_manifest.json",
            "hts_residual_edge_manifest.sha256",
        }
    ]
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-manifest-v1",
        "run_id": config.run_id,
        "analysis_status": "DEVELOPMENT_DIAGNOSTIC_COMPLETE",
        "selected_candidate_name": selected["candidate_name"],
        "development_candidate_gate_passed": development_gate["passed"],
        "incremental_signal_status": power_report["incremental_signal_status"],
        "recommended_minimum_fresh_confirmatory_markets": power_report[
            "recommended_minimum_fresh_confirmatory_markets"
        ],
        "candidate_protocol": _descriptor(protocol_path),
        "selected_candidate_contract": _descriptor(selected_contract_path),
        "candidate_protocol_sha256_descriptor": _descriptor(protocol_hash_path),
        "development_manifest": _descriptor(
            output_dir / "hts_incremental_edge_development_manifest.json"
        ),
        "diagnostic_report": _descriptor(diagnostic_json),
        "power_report": _descriptor(power_json),
        "new_confirmatory_validation_started": False,
        "prospective_collection_started": False,
        "fresh_confirmatory_validation_start_allowed": False,
        "future_unseen_confirmatory_validation_required": True,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "manifest_self_hash_embedded": False,
        "manifest_hash_descriptor_external": True,
        **safety_fields(),
    }
    manifest_path = output_dir / "hts_residual_edge_manifest.json"
    _write_json(manifest_path, manifest)
    manifest_hash_path = _write_sha_descriptor(manifest_path)
    return {
        "analysis_dir": output_dir,
        "manifest_path": manifest_path,
        "manifest_sha256_path": manifest_hash_path,
        "manifest_sha256": sha256_file(manifest_path),
        "selected_candidate_name": selected["candidate_name"],
        "development_candidate_gate_passed": development_gate["passed"],
        "incremental_signal_status": power_report["incremental_signal_status"],
        "recommended_minimum_fresh_confirmatory_markets": power_report[
            "recommended_minimum_fresh_confirmatory_markets"
        ],
    }


def fit_residual_offset_contract(
    rows: list[dict[str, Any]],
    spec: dict[str, Any],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot fit residual offset without rows")
    feature_names = list(spec["feature_names"])
    weights = _market_weights(rows)
    scaling = _fit_feature_scaling(rows, feature_names, weights)
    matrix = [
        [1.0, *[_scaled_feature(row, name, scaling) for name in feature_names]]
        for row in rows
    ]
    offsets = [_logit(_selected_probability(row)) for row in rows]
    targets = [float(row["selected_side_win_target"]) for row in rows]
    parameters = _fit_offset_logistic(
        matrix=matrix,
        offsets=offsets,
        targets=targets,
        weights=weights,
        regularization=float(spec["regularization"]),
    )
    bound = float(spec["maximum_absolute_residual_coefficient"])
    parameters = [min(max(value, -bound), bound) for value in parameters]
    return {
        "schema_version": f"{SCHEMA_PREFIX}-model-contract-v1",
        "candidate_name": spec["candidate_name"],
        "model_family": MODEL_FAMILY,
        "model_output_semantics": "selected_side_win_probability",
        "market_probability_offset_field": "selected_side_probability",
        "market_probability_offset_transform": "logit",
        "market_probability_offset_coefficient": 1.0,
        "market_probability_offset_trainable": False,
        "residual_parameters": parameters,
        "residual_parameter_names": ["intercept_delta", *feature_names],
        "residual_coefficients_shrinkage_target": 0.0,
        "regularization": spec["regularization"],
        "feature_names": feature_names,
        "feature_scaling": scaling,
        "probability_bounds": spec["probability_bounds"],
        "finite_and_bounded": all(
            math.isfinite(value) and abs(value) <= bound for value in parameters
        ),
        "decision_time_features_only": True,
        "selected_side_probability_used_as_fixed_market_baseline": True,
        "selected_side_probability_used_as_regime_vote": False,
        "settlement_outcome_used_as_training_target_only": True,
        "settlement_outcome_used_as_input": False,
        "derived_ev_semantics": (
            "predicted_probability_minus_execution_price_minus_decision_time_cost"
        ),
        "execution_cost_subtracted_exactly_once": True,
        **safety_fields(),
    }


def predict_residual_offset_probability(
    row: dict[str, Any], contract: dict[str, Any]
) -> float:
    offset = _logit(_selected_probability(row))
    vector = [
        1.0,
        *[
            _scaled_feature(row, name, contract["feature_scaling"])
            for name in contract["feature_names"]
        ],
    ]
    residual = sum(
        parameter * value
        for parameter, value in zip(
            contract["residual_parameters"], vector, strict=True
        )
    )
    low, high = contract["probability_bounds"]
    return min(max(_sigmoid(offset + residual), float(low)), float(high))


def _validate_exhausted_source_goal(report: dict[str, Any]) -> None:
    reasons = set(report.get("blocking_reason_codes", []))
    rounds = report.get("validation_round_history", [])
    required = {
        "all_predeclared_candidates_exhausted",
        "all_three_validation_rounds_failed",
        "no_validation_round_passed_all_frozen_gates",
    }
    if report.get("final_state") != "PRE_PROMOTION_BLOCKED":
        raise ValueError("source estimand goal is not PRE_PROMOTION_BLOCKED")
    if not required.issubset(reasons):
        raise ValueError("source estimand goal is not an exhausted three-round goal")
    if len(rounds) != 3 or any(
        row.get("all_confirmatory_gates_passed") is not False for row in rounds
    ):
        raise ValueError("source validation round history is not three failed rounds")
    for field in (
        "source_model_candidate_eligible",
        "freeze_ready",
        "promotion_evidence_eligible",
    ):
        if report.get(field) is not False:
            raise ValueError(f"source safety field is not fail-closed: {field}")


def _verify_source_report_lineage(
    source_report_path: Path, source_manifest_path: Path
) -> None:
    manifest = _load_json(source_manifest_path)
    expected_relative = source_report_path.name
    expected_hash = sha256_file(source_report_path)
    matches = [
        row
        for row in manifest.get("artifacts", [])
        if row.get("relative_path") == expected_relative
    ]
    if len(matches) != 1 or matches[0].get("sha256") != expected_hash:
        raise ValueError("source readiness report is not verified by source manifest")


def _load_post_validation_development_rows(
    source_goal: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths = [source_goal / "immutable_development_rows.jsonl"]
    paths.extend(sorted(source_goal.glob("round*/round_*_unseen_validation_rows.jsonl")))
    by_identity: dict[str, dict[str, Any]] = {}
    identity_source: dict[str, str] = {}
    lineage: list[dict[str, Any]] = []
    for path in paths:
        _verify_sha_descriptor(path)
        source_rows = _load_jsonl(path)
        normalized, rejected = _normalize_development_rows(source_rows)
        if rejected:
            reasons = Counter(
                reason
                for row in rejected
                for reason in row.get("reason_codes", [])
            )
            raise ValueError(f"invalid source rows in {path}: {dict(reasons)}")
        included = 0
        duplicates = 0
        for row in normalized:
            if row["action_family"] != SUPPORTED_FAMILY:
                continue
            identity = str(row["row_identity"])
            content_hash = row["row_content_sha256"]
            if identity in by_identity:
                if by_identity[identity]["row_content_sha256"] != content_hash:
                    raise ValueError(f"row identity content mismatch: {identity}")
                duplicates += 1
                continue
            row.update(
                {
                    "lineage": "post_validation_development_only",
                    "development_evidence_only": True,
                    "unseen_validation_eligible": False,
                    "future_shadow_eligible": False,
                    "promotion_evidence_eligible": False,
                    "future_confirmatory_validation_eligible": False,
                    "post_validation_source_path": str(path),
                }
            )
            by_identity[identity] = row
            identity_source[identity] = str(path)
            included += 1
        lineage.append(
            {
                "source": _descriptor(path),
                "source_row_count": len(source_rows),
                "included_unique_hts_row_count": included,
                "duplicate_hts_row_count": duplicates,
                "lineage": "post_validation_development_only",
                "future_confirmatory_validation_eligible": False,
            }
        )
    rows = sorted(
        by_identity.values(),
        key=lambda row: (
            float(row["decision_ts"]),
            str(row["market_id"]),
            str(row["row_identity"]),
        ),
    )
    if any(float(row["max_input_ts"]) > float(row["decision_ts"]) for row in rows):
        raise ValueError("post-validation rows contain decision-time causality violations")
    return rows, lineage


def _build_development_manifest(
    *,
    rows: list[dict[str, Any]],
    lineage: list[dict[str, Any]],
    source_report_path: Path,
    source_manifest_path: Path,
    combined_rows_path: Path,
) -> dict[str, Any]:
    outcomes_by_market = {
        str(row["market_id"]): str(row["target_provenance"]["resolved_outcome"])
        for row in rows
    }
    return {
        "schema_version": f"{SCHEMA_PREFIX}-development-manifest-v1",
        "lineage": "post_validation_development_only",
        "all_previously_inspected_rows_excluded_from_future_confirmatory": True,
        "future_confirmatory_validation_eligible": False,
        "source_goal_report": _descriptor(source_report_path),
        "source_goal_manifest": _descriptor(source_manifest_path),
        "source_row_artifacts": lineage,
        "combined_rows": _descriptor(combined_rows_path),
        "combined_row_count": len(rows),
        "combined_market_count": len({row["market_id"] for row in rows}),
        "combined_condition_count": len({row["condition_id"] for row in rows}),
        "combined_source_run_count": len({row["source_run_id"] for row in rows}),
        "excluded_row_identities": sorted(str(row["row_identity"]) for row in rows),
        "excluded_market_ids": sorted(str(row["market_id"]) for row in rows),
        "excluded_condition_ids": sorted(str(row["condition_id"]) for row in rows),
        "excluded_source_run_ids": sorted(str(row["source_run_id"]) for row in rows),
        "side_counts": dict(Counter(str(row["selected_side"]) for row in rows)),
        "resolved_outcome_market_counts": dict(Counter(outcomes_by_market.values())),
        "causality_violation_count": 0,
        "all_row_content_hashes_present": all(
            bool(row.get("row_content_sha256")) for row in rows
        ),
        "new_confirmatory_validation_started": False,
        **safety_fields(),
    }


def _candidate_specs() -> list[dict[str, Any]]:
    common = {
        "model_family": MODEL_FAMILY,
        "market_probability_offset_field": "selected_side_probability",
        "market_probability_offset_transform": "logit",
        "market_probability_offset_coefficient": 1.0,
        "market_probability_offset_trainable": False,
        "residual_coefficients_shrinkage_target": 0.0,
        "probability_bounds": [0.01, 0.99],
        "maximum_absolute_residual_coefficient": 3.0,
        "decision_time_features_only": True,
        "settlement_outcome_used_as_target_only": True,
        "settlement_outcome_used_as_input": False,
    }
    return [
        {
            **common,
            "candidate_name": "hts_residual_rank_only_offset",
            "feature_names": ["canonical_o_action_score", "action_score_margin"],
            "regularization": 25.0,
        },
        {
            **common,
            "candidate_name": "hts_residual_rank_anchor_offset",
            "feature_names": [
                "canonical_o_action_score",
                "action_score_margin",
                "combined_btc_anchor_alignment",
            ],
            "regularization": 35.0,
        },
        {
            **common,
            "candidate_name": "hts_residual_independent_groups_offset",
            "feature_names": [
                "canonical_o_action_score",
                "action_score_margin",
                "combined_btc_anchor_alignment",
                "selected_side_probability_minus_execution_price",
                "spread_bps",
                "queue_fill_proxy",
                "book_staleness_ms",
                "log_time_to_close_seconds",
                "cumulative_market_exposure_before_entry",
                "same_side_reentry",
                "side_flip",
            ],
            "regularization": 60.0,
        },
    ]


def _candidate_protocol(
    config: HTSResidualEdgePowerConfig,
    specs: list[dict[str, Any]],
    development_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA_PREFIX}-candidate-protocol-v1",
        "protocol_frozen_before_any_new_prospective_data": True,
        "protocol_created_at": config.created_at,
        "candidate_count": len(specs),
        "candidate_specifications": specs,
        "baseline": {
            "candidate_name": "raw_selected_side_market_probability",
            "model_family": "identity_probability",
            "confirmatory_baseline_only": True,
        },
        "feature_group_contract": {
            "rank_signal": ["canonical_o_action_score", "action_score_margin"],
            "btc_anchor_direction": {
                "combined_feature": "combined_btc_anchor_alignment",
                "source_fields": [
                    "btc_momentum",
                    "reference_price_to_beat_distance_at_decision",
                ],
                "single_coefficient_group": True,
                "double_counting_allowed": False,
            },
            "market_price_value": [
                "selected_side_probability_minus_execution_price"
            ],
            "execution_quality": [
                "spread_bps",
                "queue_fill_proxy",
                "book_staleness_ms",
                "log_time_to_close_seconds",
            ],
            "pre_entry_exposure": [
                "cumulative_market_exposure_before_entry",
                "same_side_reentry",
                "side_flip",
            ],
        },
        "candidate_ranking_metrics": [
            "forward_oof_market_weighted_brier_score",
            "forward_oof_market_weighted_log_loss",
            "worst_fold_brier_score",
            "parameter_count",
            "candidate_name",
        ],
        "development_gate": {
            "minimum_relative_brier_improvement_vs_raw": config.minimum_relative_brier_improvement,
            "minimum_relative_log_loss_improvement_vs_raw": config.minimum_relative_log_loss_improvement,
            "finite_bounded_coefficients_required": True,
            "positive_market_mean_brier_improvement_required": True,
        },
        "future_confirmatory_gate_unchanged": True,
        "future_confirmatory_confidence_level": config.bootstrap_confidence_level,
        "future_confirmatory_market_disjointness_required": True,
        "future_confirmatory_strict_chronology_required": True,
        "future_confirmatory_exactly_once_required": True,
        "prior_seen_row_count": development_manifest["combined_row_count"],
        "prior_seen_market_count": development_manifest["combined_market_count"],
        "prior_seen_rows_development_only": True,
        "future_confirmatory_validation_start_allowed": False,
        "new_confirmatory_validation_started": False,
        **safety_fields(),
    }


def _evaluate_candidate_oof(
    rows: list[dict[str, Any]],
    spec: dict[str, Any],
    *,
    minimum_training_runs: int,
    include_prediction_rows: bool = False,
) -> dict[str, Any]:
    runs = sorted(
        {str(row["source_run_id"]) for row in rows},
        key=lambda run: min(
            float(row["decision_ts"])
            for row in rows
            if str(row["source_run_id"]) == run
        ),
    )
    predictions: dict[str, float] = {}
    fold_reports: list[dict[str, Any]] = []
    parameter_sets: list[list[float]] = []
    for index in range(minimum_training_runs, len(runs)):
        train_runs = set(runs[:index])
        validation_run = runs[index]
        train = [row for row in rows if row["source_run_id"] in train_runs]
        validation = [row for row in rows if row["source_run_id"] == validation_run]
        if not train or not validation:
            continue
        if min(row["decision_ts"] for row in validation) <= max(
            row["decision_ts"] for row in train
        ):
            raise ValueError("forward OOF fold chronology violation")
        contract = fit_residual_offset_contract(train, spec)
        fold_predictions = [
            predict_residual_offset_probability(row, contract) for row in validation
        ]
        for row, prediction in zip(validation, fold_predictions, strict=True):
            predictions[str(row["row_identity"])] = prediction
        candidate_metrics = _probability_metrics(validation, fold_predictions)
        raw_metrics = _probability_metrics(
            validation, [_selected_probability(row) for row in validation]
        )
        parameter_sets.append(list(contract["residual_parameters"]))
        fold_reports.append(
            {
                "fold_number": len(fold_reports) + 1,
                "training_run_count": len(train_runs),
                "training_row_count": len(train),
                "training_market_count": len({row["market_id"] for row in train}),
                "validation_run_id": validation_run,
                "validation_row_count": len(validation),
                "validation_market_count": len(
                    {row["market_id"] for row in validation}
                ),
                "strict_chronology_passed": True,
                "candidate_metrics": candidate_metrics,
                "raw_baseline_metrics": raw_metrics,
            }
        )
    oof_rows = [row for row in rows if str(row["row_identity"]) in predictions]
    oof_predictions = [predictions[str(row["row_identity"])] for row in oof_rows]
    if not oof_rows:
        raise ValueError("no forward OOF rows available")
    raw_predictions = [_selected_probability(row) for row in oof_rows]
    candidate_metrics = _probability_metrics(oof_rows, oof_predictions)
    raw_metrics = _probability_metrics(oof_rows, raw_predictions)
    relative_brier = _relative_improvement(
        raw_metrics["market_weighted_brier_score"],
        candidate_metrics["market_weighted_brier_score"],
    )
    relative_log_loss = _relative_improvement(
        raw_metrics["market_weighted_log_loss"],
        candidate_metrics["market_weighted_log_loss"],
    )
    market_deltas = _market_error_deltas(oof_rows, oof_predictions, raw_predictions)
    attribution = {
        "by_side": _group_error_attribution(
            oof_rows,
            oof_predictions,
            raw_predictions,
            lambda row: str(row["selected_side"]),
        ),
        "by_horizon": _group_error_attribution(
            oof_rows, oof_predictions, raw_predictions, _horizon_bucket
        ),
        "by_selected_probability_bucket": _group_error_attribution(
            oof_rows,
            oof_predictions,
            raw_predictions,
            lambda row: _probability_bucket(_selected_probability(row)),
        ),
        "by_execution_price_bucket": _group_error_attribution(
            oof_rows,
            oof_predictions,
            raw_predictions,
            lambda row: _probability_bucket(
                float(row["decision_time_features"]["execution_price"])
            ),
        ),
        "by_time_to_close_bucket": _group_error_attribution(
            oof_rows, oof_predictions, raw_predictions, _time_to_close_bucket
        ),
        "by_btc_anchor_alignment": _group_error_attribution(
            oof_rows, oof_predictions, raw_predictions, _btc_anchor_bucket
        ),
        "by_same_side_reentry": _group_error_attribution(
            oof_rows,
            oof_predictions,
            raw_predictions,
            lambda row: str(
                int(row["decision_time_features"]["same_side_reentry"])
            ),
        ),
        "by_side_flip": _group_error_attribution(
            oof_rows,
            oof_predictions,
            raw_predictions,
            lambda row: str(int(row["decision_time_features"]["side_flip"])),
        ),
    }
    stability = _parameter_stability(parameter_sets)
    report = {
        "candidate_name": spec["candidate_name"],
        "candidate_specification": spec,
        "forward_oof_fold_count": len(fold_reports),
        "forward_oof_row_count": len(oof_rows),
        "forward_oof_market_count": len({row["market_id"] for row in oof_rows}),
        "fold_reports": fold_reports,
        "candidate_metrics": candidate_metrics,
        "raw_baseline_metrics": raw_metrics,
        "relative_brier_improvement_vs_raw": relative_brier,
        "relative_log_loss_improvement_vs_raw": relative_log_loss,
        "market_level_error_deltas": market_deltas,
        "residual_error_attribution": attribution,
        "largest_negative_market_deltas": sorted(
            market_deltas["by_market"], key=lambda row: row["brier_improvement"]
        )[:10],
        "largest_positive_market_deltas": sorted(
            market_deltas["by_market"],
            key=lambda row: row["brier_improvement"],
            reverse=True,
        )[:10],
        "learning_curve_by_forward_fold": [
            {
                "fold_number": fold["fold_number"],
                "training_market_count": fold["training_market_count"],
                "validation_market_count": fold["validation_market_count"],
                "candidate_brier": fold["candidate_metrics"][
                    "market_weighted_brier_score"
                ],
                "raw_baseline_brier": fold["raw_baseline_metrics"][
                    "market_weighted_brier_score"
                ],
                "brier_improvement": fold["raw_baseline_metrics"][
                    "market_weighted_brier_score"
                ]
                - fold["candidate_metrics"]["market_weighted_brier_score"],
            }
            for fold in fold_reports
        ],
        "parameter_stability": stability,
        "parameter_count": len(spec["feature_names"]) + 1,
        "uses_fresh_future_confirmatory_labels_for_tuning": False,
        "uses_post_validation_development_labels": True,
        "prior_failed_validation_labels_reclassified_as_development": True,
        "confirmatory_evidence": False,
    }
    if include_prediction_rows:
        report["_forward_oof_prediction_rows"] = [
            {
                "candidate_name": spec["candidate_name"],
                "row_identity": str(row["row_identity"]),
                "market_id": str(row["market_id"]),
                "source_run_id": str(row["source_run_id"]),
                "decision_ts": int(row["decision_ts"]),
                "market_close_ts": int(row["market_close_ts"]),
                "selected_action": str(row["selected_action"]),
                "selected_side": str(row["selected_side"]),
                "execution_price": float(
                    row["decision_time_features"]["execution_price"]
                ),
                "raw_baseline_probability": _selected_probability(row),
                "candidate_probability": prediction,
                "decision_time_expected_execution_cost_per_unit": float(
                    row["decision_time_expected_execution_cost_per_unit"]
                ),
                "target_net_return_after_cost": float(
                    row["target_net_return_after_cost"]
                ),
                "evaluation_target_net_return_after_cost_by_action": dict(
                    row.get("evaluation_target_net_return_after_cost_by_action") or {}
                ),
                "evaluation_target_net_pnl_per_contract_by_action": dict(
                    row.get("evaluation_target_net_pnl_per_contract_by_action")
                    or {}
                ),
                "evaluation_target_pnl_components_by_action": dict(
                    row.get("evaluation_target_pnl_components_by_action") or {}
                ),
                "execution_handoff_context": dict(
                    row.get("execution_handoff_context") or {}
                ),
                "selected_side_win_target": int(row["selected_side_win_target"]),
                "target_used_for_selection": False,
                "outcome_aware_evaluation_only": True,
            }
            for row, prediction in zip(oof_rows, oof_predictions, strict=True)
        ]
    return report


def _development_candidate_gate(
    selected: dict[str, Any], config: HTSResidualEdgePowerConfig
) -> dict[str, Any]:
    checks = {
        "relative_brier_improvement_vs_raw_passed": selected[
            "relative_brier_improvement_vs_raw"
        ]
        >= config.minimum_relative_brier_improvement,
        "relative_log_loss_improvement_vs_raw_passed": selected[
            "relative_log_loss_improvement_vs_raw"
        ]
        >= config.minimum_relative_log_loss_improvement,
        "positive_market_mean_brier_improvement_passed": selected[
            "market_level_error_deltas"
        ]["mean_brier_improvement"]
        > 0.0,
        "parameter_stability_diagnostic_passed": selected["parameter_stability"][
            "finite"
        ],
    }
    reasons = [
        name.removesuffix("_passed") + "_failed"
        for name, passed in checks.items()
        if not passed
    ]
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "blocking_reason_codes": reasons,
        "development_only": True,
        "does_not_authorize_confirmatory_validation": True,
    }


def _diagnostic_report(
    *,
    rows: list[dict[str, Any]],
    candidate_reports: list[dict[str, Any]],
    selected: dict[str, Any],
    selected_contract: dict[str, Any],
    development_gate: dict[str, Any],
    protocol_path: Path,
    combined_rows_path: Path,
) -> dict[str, Any]:
    coverage = {
        feature: sum(
            _feature_available(row, feature)
            for row in rows
        )
        for feature in _all_source_features()
    }
    return {
        "schema_version": f"{SCHEMA_PREFIX}-diagnostic-report-v1",
        "analysis_scope": "HTS_ONLY",
        "development_lineage": "post_validation_development_only",
        "row_count": len(rows),
        "market_count": len({row["market_id"] for row in rows}),
        "source_run_count": len({row["source_run_id"] for row in rows}),
        "side_counts": dict(Counter(row["selected_side"] for row in rows)),
        "feature_coverage_counts": coverage,
        "feature_coverage_complete": all(value == len(rows) for value in coverage.values()),
        "causality_violation_count": sum(
            row["max_input_ts"] > row["decision_ts"] for row in rows
        ),
        "candidate_reports": candidate_reports,
        "selected_candidate_name": selected["candidate_name"],
        "selected_candidate_contract": selected_contract,
        "selected_candidate_development_gate": development_gate,
        "market_probability_fixed_offset_verified": (
            selected_contract["market_probability_offset_coefficient"] == 1.0
            and selected_contract["market_probability_offset_trainable"] is False
        ),
        "no_incremental_signal_behavior": "residual_delta_shrinks_to_zero_market_baseline",
        "candidate_protocol": _descriptor(protocol_path),
        "development_rows": _descriptor(combined_rows_path),
        "new_confirmatory_validation_started": False,
        "prospective_collection_started": False,
        "future_confirmatory_validation_start_allowed": False,
        "sell_before_close_scope": "blocked_separate_estimand_required",
        **safety_fields(),
    }


def _power_report(
    *,
    selected: dict[str, Any],
    config: HTSResidualEdgePowerConfig,
    development_gate: dict[str, Any],
) -> dict[str, Any]:
    deltas = [
        float(row["brier_improvement"])
        for row in selected["market_level_error_deltas"]["by_market"]
    ]
    mean_effect = statistics.mean(deltas)
    std_effect = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    bootstrap_values = _bootstrap_market_mean(
        deltas,
        samples=config.bootstrap_samples,
        seed=config.random_seed,
    )
    alpha = (1.0 - config.bootstrap_confidence_level) / 2.0
    interval = {
        "confidence_level": config.bootstrap_confidence_level,
        "lower": _quantile(bootstrap_values, alpha),
        "upper": _quantile(bootstrap_values, 1.0 - alpha),
    }
    raw_brier = float(
        selected["raw_baseline_metrics"]["market_weighted_brier_score"]
    )
    minimum_gate_effect = raw_brier * config.minimum_relative_brier_improvement
    estimates = {
        "observed_effect": _sample_size_estimates(
            effect=mean_effect,
            standard_deviation=std_effect,
            confidence_level=config.bootstrap_confidence_level,
            power_levels=config.target_power_levels,
        ),
        "minimum_gate_effect": _sample_size_estimates(
            effect=minimum_gate_effect,
            standard_deviation=std_effect,
            confidence_level=config.bootstrap_confidence_level,
            power_levels=config.target_power_levels,
        ),
    }
    observed_n90 = estimates["observed_effect"].get("power_0_90_market_count")
    gate_n90 = estimates["minimum_gate_effect"].get("power_0_90_market_count")
    finite_targets = [
        value for value in (observed_n90, gate_n90) if isinstance(value, int)
    ]
    planning_count = max([config.minimum_prospective_market_count, *finite_targets])
    recommended = planning_count if development_gate["passed"] else None
    if mean_effect <= 0.0:
        status = "no_positive_incremental_signal"
    elif interval["lower"] > 0.0:
        status = "development_signal_positive_not_confirmatory"
    else:
        status = "weak_inconclusive_incremental_signal"
    blockers = []
    if not development_gate["passed"]:
        blockers.append("development_candidate_gate_failed")
    if interval["lower"] <= 0.0:
        blockers.append("development_market_bootstrap_interval_crosses_zero")
    blockers.extend(
        [
            "all_prior_evidence_is_development_only",
            "fresh_market_disjoint_confirmatory_evidence_not_collected",
        ]
    )
    return {
        "schema_version": f"{SCHEMA_PREFIX}-power-report-v1",
        "selected_candidate_name": selected["candidate_name"],
        "resampling_unit": "market_id",
        "development_market_count": len(deltas),
        "observed_market_mean_brier_improvement": mean_effect,
        "observed_market_median_brier_improvement": statistics.median(deltas),
        "observed_market_brier_improvement_standard_deviation": std_effect,
        "market_improvement_win_rate": sum(value > 0.0 for value in deltas) / len(deltas),
        "bootstrap_samples": config.bootstrap_samples,
        "bootstrap_mean_improvement_interval": interval,
        "raw_baseline_market_weighted_brier_score": raw_brier,
        "minimum_relative_brier_improvement": config.minimum_relative_brier_improvement,
        "minimum_absolute_gate_effect": minimum_gate_effect,
        "market_count_estimates": estimates,
        "recommended_minimum_fresh_confirmatory_markets": recommended,
        "planning_market_count_if_candidate_development_gate_later_passes": planning_count,
        "current_candidate_collection_authorized": False,
        "recommended_market_count_is_planning_estimate_not_gate_relaxation": True,
        "incremental_signal_status": status,
        "prospective_collection_plan_ready": development_gate["passed"],
        "prospective_collection_started": False,
        "fresh_confirmatory_validation_start_allowed": False,
        "future_confirmatory_validation_exactly_once_required": True,
        "future_confirmatory_validation_market_disjoint_required": True,
        "future_confirmatory_validation_strictly_later_required": True,
        "blocking_reason_codes": sorted(set(blockers)),
        **safety_fields(),
    }


def _probability_metrics(
    rows: list[dict[str, Any]], predictions: list[float]
) -> dict[str, float]:
    weights = _market_weights(rows)
    total = sum(weights)
    targets = [float(row["selected_side_win_target"]) for row in rows]
    return {
        "market_weighted_brier_score": sum(
            weight * (target - prediction) ** 2
            for weight, target, prediction in zip(
                weights, targets, predictions, strict=True
            )
        )
        / total,
        "market_weighted_log_loss": -sum(
            weight
            * (
                target * math.log(_clip_probability(prediction))
                + (1.0 - target)
                * math.log(1.0 - _clip_probability(prediction))
            )
            for weight, target, prediction in zip(
                weights, targets, predictions, strict=True
            )
        )
        / total,
    }


def _market_error_deltas(
    rows: list[dict[str, Any]],
    candidate: list[float],
    raw: list[float],
) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["market_id"])].append(index)
    result = []
    for market_id, indexes in sorted(grouped.items()):
        candidate_error = statistics.mean(
            (float(rows[index]["selected_side_win_target"]) - candidate[index]) ** 2
            for index in indexes
        )
        raw_error = statistics.mean(
            (float(rows[index]["selected_side_win_target"]) - raw[index]) ** 2
            for index in indexes
        )
        result.append(
            {
                "market_id": market_id,
                "row_count": len(indexes),
                "candidate_brier": candidate_error,
                "raw_baseline_brier": raw_error,
                "brier_improvement": raw_error - candidate_error,
            }
        )
    values = [row["brier_improvement"] for row in result]
    return {
        "market_count": len(result),
        "mean_brier_improvement": statistics.mean(values),
        "median_brier_improvement": statistics.median(values),
        "positive_market_count": sum(value > 0.0 for value in values),
        "negative_market_count": sum(value < 0.0 for value in values),
        "zero_market_count": sum(value == 0.0 for value in values),
        "by_market": result,
    }


def _group_error_attribution(
    rows: list[dict[str, Any]],
    candidate: list[float],
    raw: list[float],
    grouper: Any,
) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(grouper(row))].append(index)
    result: dict[str, Any] = {}
    for group, indexes in sorted(grouped.items()):
        group_rows = [rows[index] for index in indexes]
        candidate_metrics = _probability_metrics(
            group_rows, [candidate[index] for index in indexes]
        )
        raw_metrics = _probability_metrics(
            group_rows, [raw[index] for index in indexes]
        )
        result[group] = {
            "row_count": len(indexes),
            "market_count": len({row["market_id"] for row in group_rows}),
            "candidate_brier": candidate_metrics["market_weighted_brier_score"],
            "raw_baseline_brier": raw_metrics["market_weighted_brier_score"],
            "brier_improvement": raw_metrics["market_weighted_brier_score"]
            - candidate_metrics["market_weighted_brier_score"],
            "candidate_log_loss": candidate_metrics["market_weighted_log_loss"],
            "raw_baseline_log_loss": raw_metrics["market_weighted_log_loss"],
        }
    return result


def _horizon_bucket(row: dict[str, Any]) -> str:
    slug = str(row.get("market_slug", ""))
    return "15m" if "15m" in slug else "5m"


def _probability_bucket(value: float) -> str:
    bucket = min(int(value * 10.0), 9)
    return f"{bucket / 10.0:.1f}-{(bucket + 1) / 10.0:.1f}"


def _time_to_close_bucket(row: dict[str, Any]) -> str:
    value = float(row["decision_time_features"]["time_to_close_seconds"])
    if value < 120.0:
        return "lt_120s"
    if value < 300.0:
        return "120_300s"
    if value < 600.0:
        return "300_600s"
    return "gte_600s"


def _btc_anchor_bucket(row: dict[str, Any]) -> str:
    value = _raw_feature(row, "combined_btc_anchor_alignment")
    if value < -1e-12:
        return "opposes_selected_side"
    if value > 1e-12:
        return "supports_selected_side"
    return "neutral"


def _fit_feature_scaling(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    weights: list[float],
) -> dict[str, dict[str, float]]:
    total = sum(weights)
    result = {}
    for name in feature_names:
        values = [_raw_feature(row, name) for row in rows]
        mean = sum(weight * value for weight, value in zip(weights, values, strict=True)) / total
        variance = sum(
            weight * (value - mean) ** 2
            for weight, value in zip(weights, values, strict=True)
        ) / total
        result[name] = {"mean": mean, "scale": max(math.sqrt(variance), 1e-8)}
    return result


def _scaled_feature(
    row: dict[str, Any], name: str, scaling: dict[str, dict[str, float]]
) -> float:
    contract = scaling[name]
    return (_raw_feature(row, name) - contract["mean"]) / contract["scale"]


def _raw_feature(row: dict[str, Any], name: str) -> float:
    features = row["decision_time_features"]
    if name == "combined_btc_anchor_alignment":
        side_sign = 1.0 if row["selected_side"] == "UP" else -1.0
        return side_sign * (
            float(features["btc_momentum"])
            + float(features["reference_price_to_beat_distance_at_decision"])
        )
    if name == "chainlink_anchor_alignment":
        side_sign = 1.0 if row["selected_side"] == "UP" else -1.0
        values = sorted(
            float(features[field])
            for field in (
                "chainlink_momentum_30s",
                "chainlink_momentum_60s",
                "chainlink_momentum_120s",
                "reference_price_to_beat_distance_at_decision",
            )
        )
        return side_sign * statistics.median(values)
    if name == "chainlink_anchor_overextension_abs":
        return abs(float(features["reference_price_to_beat_distance_at_decision"]))
    if name == "log1p_spread_bps":
        return math.log1p(max(float(features["spread_bps"]), 0.0))
    if name == "queue_fill_shortfall":
        return 1.0 - min(max(float(features["queue_fill_proxy"]), 0.0), 1.0)
    if name == "log1p_book_staleness_ms":
        return math.log1p(max(float(features["book_staleness_ms"]), 0.0))
    if name == "late_window_pressure":
        return max(0.0, 120.0 - float(features["time_to_close_seconds"])) / 120.0
    if name == "anchor_alignment_x_price_value":
        return _raw_feature(row, "chainlink_anchor_alignment") * float(
            features["selected_side_probability_minus_execution_price"]
        )
    if name == "spread_x_staleness_quality_penalty":
        return _raw_feature(row, "log1p_spread_bps") * _raw_feature(
            row, "log1p_book_staleness_ms"
        )
    if name == "log_time_to_close_seconds":
        return math.log1p(max(float(features["time_to_close_seconds"]), 0.0))
    return float(features[name])


def _fit_offset_logistic(
    *,
    matrix: list[list[float]],
    offsets: list[float],
    targets: list[float],
    weights: list[float],
    regularization: float,
) -> list[float]:
    size = len(matrix[0])
    parameters = [0.0] * size
    for _ in range(80):
        gradient = [0.0] * size
        hessian = [[0.0] * size for _ in range(size)]
        for vector, offset, target, weight in zip(
            matrix, offsets, targets, weights, strict=True
        ):
            probability = _sigmoid(
                offset
                + sum(
                    parameter * value
                    for parameter, value in zip(parameters, vector, strict=True)
                )
            )
            variance = max(probability * (1.0 - probability), 1e-6)
            for left in range(size):
                gradient[left] += weight * (probability - target) * vector[left]
                for right in range(size):
                    hessian[left][right] += (
                        weight * variance * vector[left] * vector[right]
                    )
        for index in range(size):
            gradient[index] += regularization * parameters[index]
            hessian[index][index] += regularization
        step = _solve_linear(hessian, gradient)
        parameters = [
            value - delta for value, delta in zip(parameters, step, strict=True)
        ]
        if max(abs(delta) for delta in step) < 1e-8:
            break
    return parameters


def _solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector, strict=True)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            augmented[pivot][column] = 1e-12
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    augmented[row], augmented[column], strict=True
                )
            ]
    return [augmented[row][-1] for row in range(size)]


def _parameter_stability(parameter_sets: list[list[float]]) -> dict[str, Any]:
    if not parameter_sets:
        return {
            "fold_count": 0,
            "finite": False,
            "maximum_fold_parameter_range": None,
        }
    width = min(len(values) for values in parameter_sets)
    ranges = [
        max(values[index] for values in parameter_sets)
        - min(values[index] for values in parameter_sets)
        for index in range(width)
    ]
    return {
        "fold_count": len(parameter_sets),
        "finite": all(
            math.isfinite(value) for values in parameter_sets for value in values
        ),
        "maximum_fold_parameter_range": max(ranges, default=0.0),
    }


def _bootstrap_market_mean(
    values: list[float], *, samples: int, seed: int
) -> list[float]:
    rng = random.Random(seed)
    return [
        statistics.mean(rng.choice(values) for _ in values) for _ in range(samples)
    ]


def _sample_size_estimates(
    *,
    effect: float,
    standard_deviation: float,
    confidence_level: float,
    power_levels: tuple[float, ...],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "effect": effect,
        "market_level_standard_deviation": standard_deviation,
        "confidence_level": confidence_level,
        "method": "normal_approximation_paired_market_mean",
    }
    if effect <= 0.0:
        result["estimable"] = False
        result["reason_code"] = "non_positive_observed_effect"
        for power in power_levels:
            result[_power_key(power)] = None
        return result
    result["estimable"] = True
    z_confidence = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    for power in power_levels:
        z_power = NormalDist().inv_cdf(power)
        count = math.ceil(
            ((z_confidence + z_power) * standard_deviation / effect) ** 2
        )
        result[_power_key(power)] = max(count, 2)
    return result


def _power_key(power: float) -> str:
    return f"power_{power:.2f}".replace(".", "_") + "_market_count"


def _candidate_ranking_key(report: dict[str, Any]) -> tuple[Any, ...]:
    return (
        report["candidate_metrics"]["market_weighted_brier_score"],
        report["candidate_metrics"]["market_weighted_log_loss"],
        max(
            fold["candidate_metrics"]["market_weighted_brier_score"]
            for fold in report["fold_reports"]
        ),
        report["parameter_count"],
        report["candidate_name"],
    )


def _all_source_features() -> list[str]:
    return [
        "canonical_o_action_score",
        "action_score_margin",
        "btc_momentum",
        "reference_price_to_beat_distance_at_decision",
        "selected_side_probability",
        "execution_price",
        "selected_side_probability_minus_execution_price",
        "spread_bps",
        "queue_fill_proxy",
        "book_staleness_ms",
        "time_to_close_seconds",
        "cumulative_market_exposure_before_entry",
        "same_side_reentry",
        "side_flip",
    ]


def _feature_available(row: dict[str, Any], name: str) -> bool:
    value = row.get("decision_time_features", {}).get(name)
    return isinstance(value, int | float) and math.isfinite(float(value))


def _selected_probability(row: dict[str, Any]) -> float:
    return _clip_probability(
        float(row["decision_time_features"]["selected_side_probability"])
    )


def _market_weights(rows: list[dict[str, Any]]) -> list[float]:
    counts = Counter(str(row["market_id"]) for row in rows)
    return [1.0 / counts[str(row["market_id"])] for row in rows]


def _relative_improvement(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / baseline if baseline > 0.0 else 0.0


def _clip_probability(value: float) -> float:
    return min(max(value, 1e-6), 1.0 - 1e-6)


def _logit(value: float) -> float:
    clipped = _clip_probability(value)
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _diagnostic_markdown(report: dict[str, Any]) -> str:
    selected = report["selected_candidate_contract"]
    gate = report["selected_candidate_development_gate"]
    selected_report = next(
        item
        for item in report["candidate_reports"]
        if item["candidate_name"] == report["selected_candidate_name"]
    )
    lines = [
        "# HTS Incremental Edge Diagnostic",
        "",
        f"- lineage: `{report['development_lineage']}`",
        f"- rows / markets: `{report['row_count']}` / `{report['market_count']}`",
        f"- selected candidate: `{report['selected_candidate_name']}`",
        f"- market probability fixed offset: `{str(report['market_probability_fixed_offset_verified']).lower()}`",
        f"- relative Brier improvement vs raw: `{selected_report['relative_brier_improvement_vs_raw']:.6f}`",
        f"- relative log-loss improvement vs raw: `{selected_report['relative_log_loss_improvement_vs_raw']:.6f}`",
        f"- development candidate gate passed: `{str(gate['passed']).lower()}`",
        f"- blocking reason codes: `{gate['blocking_reason_codes']}`",
        f"- residual parameter count: `{len(selected['residual_parameters'])}`",
        "- new confirmatory validation started: `false`",
        "- paper/live/promotion unlock: `false`",
        "",
        "## Candidate Comparison",
        "",
        "| candidate | OOF rows | OOF markets | Brier | log loss | rel Brier vs raw | rel log loss vs raw |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in report["candidate_reports"]:
        metrics = candidate["candidate_metrics"]
        lines.append(
            "| {name} | {rows} | {markets} | {brier:.6f} | {log_loss:.6f} | {rel_brier:.6f} | {rel_log:.6f} |".format(
                name=candidate["candidate_name"],
                rows=candidate["forward_oof_row_count"],
                markets=candidate["forward_oof_market_count"],
                brier=metrics["market_weighted_brier_score"],
                log_loss=metrics["market_weighted_log_loss"],
                rel_brier=candidate["relative_brier_improvement_vs_raw"],
                rel_log=candidate["relative_log_loss_improvement_vs_raw"],
            )
        )
    lines.extend(
        [
            "",
            "All rows in this report have already been inspected and are development-only. They cannot be reused as future confirmatory evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def _power_markdown(report: dict[str, Any]) -> str:
    interval = report["bootstrap_mean_improvement_interval"]
    return "\n".join(
        [
            "# HTS Market-Level Power Analysis",
            "",
            f"- selected candidate: `{report['selected_candidate_name']}`",
            f"- development markets: `{report['development_market_count']}`",
            f"- incremental signal status: `{report['incremental_signal_status']}`",
            f"- mean Brier improvement: `{report['observed_market_mean_brier_improvement']:.8f}`",
            f"- bootstrap interval: `[{interval['lower']:.8f}, {interval['upper']:.8f}]`",
            f"- recommended minimum fresh confirmatory markets: `{report['recommended_minimum_fresh_confirmatory_markets']}`",
            f"- planning markets if a future candidate passes development gate: `{report['planning_market_count_if_candidate_development_gate_later_passes']}`",
            f"- prospective collection plan ready: `{str(report['prospective_collection_plan_ready']).lower()}`",
            f"- blocking reason codes: `{report['blocking_reason_codes']}`",
            "- confirmatory validation start allowed: `false`",
            "- paper/live/promotion unlock: `false`",
            "",
            "The market count is a planning estimate from paired market-level error variance. It is not a relaxed gate and does not authorize data evaluation.",
            "",
        ]
    )


def _descriptor(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _verify_sha_descriptor(path: Path) -> None:
    descriptor = path.with_suffix(".sha256")
    if not descriptor.exists():
        raise FileNotFoundError(f"sha256 descriptor missing: {descriptor}")
    expected = descriptor.read_text(encoding="utf-8").split()[0]
    actual = sha256_file(path)
    if expected != actual:
        raise ValueError(f"sha256 mismatch for {path}")


def _write_hashed_json(
    directory: Path, stem: str, payload: dict[str, Any]
) -> tuple[Path, Path]:
    path = directory / f"{stem}.json"
    _write_json(path, payload)
    return path, _write_sha_descriptor(path)


def _write_sha_descriptor(path: Path) -> Path:
    descriptor = path.with_suffix(".sha256")
    descriptor.write_text(
        f"{sha256_file(path)}  {path.name}\n", encoding="utf-8"
    )
    return descriptor


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _git_head(repository_root: Path) -> str:
    head = repository_root / ".git"
    if not head.exists():
        return "unknown"
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
