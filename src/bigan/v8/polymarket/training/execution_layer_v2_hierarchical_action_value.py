"""Historical-fit-only hierarchical action-value research candidate."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.contracts import compact_safety_fields
from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_action_value import (
    REQUIRED_ACTIONS,
    build_pnl_aligned_action_conditioned_rows,
)
from bigan.v8.polymarket.training.post_freeze_o_replay_aligned_source_ranking import (
    _v8_apply_simulated_order_to_state,
    _v8_execution_guard_config,
    _v8_execution_guard_decision,
    _v8_initial_runtime_state,
)

SCHEMA_PREFIX = "bigan-v8-execution-layer-v2-hierarchical-action-value"
PROTOCOL_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-protocol-v1"
TRADE_FAMILIES = ("HOLD_TO_SETTLEMENT", "SELL_BEFORE_CLOSE")
FAMILY_MODEL_FILENAMES = {
    "HOLD_TO_SETTLEMENT": "hierarchical_action_value_hts_model.xgb.json",
    "SELL_BEFORE_CLOSE": "hierarchical_action_value_sbc_model.xgb.json",
}
FORBIDDEN_FUTURE_REGISTRY_FIELDS = {
    "accepted_bet_net_pnl",
    "evaluation_target_net_pnl_per_contract_by_action",
    "evaluation_target_net_return_after_cost_by_action",
    "future_return",
    "gross_pnl",
    "net_pnl",
    "oracle_action",
    "realized_pnl",
    "resolved_outcome",
    "settlement_pnl",
    "settlement_return",
    "target_net_return_after_cost",
    "total_net_pnl_per_notional",
}


@dataclass(frozen=True, slots=True)
class HierarchicalActionValueFitConfig:
    """Inputs for one immutable historical train/calibration/validation run."""

    run_id: str
    output_dir: Path | str
    protocol_path: Path | str
    expected_protocol_sha256: str
    source_action_protocol_path: Path | str
    expected_source_action_protocol_sha256: str
    historical_corpus_manifest_path: Path | str
    excluded_future_decision_rows_path: Path | str
    expected_excluded_future_decision_rows_sha256: str
    excluded_future_artifact_pins: tuple[tuple[Path | str, str], ...]

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for name, digest in (
            ("expected_protocol_sha256", self.expected_protocol_sha256),
            (
                "expected_source_action_protocol_sha256",
                self.expected_source_action_protocol_sha256,
            ),
            (
                "expected_excluded_future_decision_rows_sha256",
                self.expected_excluded_future_decision_rows_sha256,
            ),
        ):
            _require_sha256(digest, name=name)
        pins: list[tuple[Path, str]] = []
        for path, digest in self.excluded_future_artifact_pins:
            _require_sha256(digest, name="excluded future artifact SHA-256")
            pins.append((Path(path), digest.lower()))
        if not pins:
            raise ValueError("at least one excluded future artifact pin is required")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "protocol_path", Path(self.protocol_path))
        object.__setattr__(
            self, "source_action_protocol_path", Path(self.source_action_protocol_path)
        )
        object.__setattr__(
            self,
            "historical_corpus_manifest_path",
            Path(self.historical_corpus_manifest_path),
        )
        object.__setattr__(
            self,
            "excluded_future_decision_rows_path",
            Path(self.excluded_future_decision_rows_path),
        )
        object.__setattr__(self, "excluded_future_artifact_pins", tuple(pins))


def validate_hierarchical_action_value_protocol(protocol: dict[str, Any]) -> None:
    """Reject semantic drift in the predeclared hierarchical protocol."""

    split = dict(protocol.get("split_protocol") or {})
    model = dict(protocol.get("family_model_protocol") or {})
    calibration = dict(protocol.get("calibration_protocol") or {})
    validation = dict(protocol.get("historical_validation_gates") or {})
    future = dict(protocol.get("future_evidence_gates") or {})
    safety = dict(protocol.get("safety") or {})
    fractions = [
        float(split.get("train_market_fraction") or 0.0),
        float(split.get("calibration_market_fraction") or 0.0),
        float(split.get("validation_market_fraction") or 0.0),
    ]
    feature_columns = list(protocol.get("feature_columns") or [])
    checks = {
        "schema_version": protocol.get("schema_version") == PROTOCOL_SCHEMA_VERSION,
        "candidate_name": protocol.get("candidate_name")
        == "historical_fit_only_hierarchical_action_value_v2",
        "frozen": protocol.get("frozen") is True,
        "decision_time_safe": protocol.get("decision_time_safe") is True,
        "historical_fit_only": protocol.get("historical_fit_only") is True,
        "no_future_fit": protocol.get("uses_future_holdout_labels_for_fitting")
        is False,
        "no_validation_tuning": protocol.get(
            "uses_historical_validation_labels_for_tuning"
        )
        is False,
        "no_excluded_future_tuning": protocol.get(
            "uses_excluded_future_evidence_for_tuning"
        )
        is False,
        "complete_action_grid": tuple(protocol.get("actions") or ())
        == REQUIRED_ACTIONS,
        "trade_families": tuple(protocol.get("trade_families") or ())
        == TRADE_FAMILIES,
        "target": protocol.get("primary_target")
        == "execution_realizable_cost_aware_net_return_per_notional",
        "features_unique": bool(feature_columns)
        and len(feature_columns) == len(set(feature_columns)),
        "split_method": split.get("method")
        == "chronological_market_grouped_contiguous_v1",
        "split_fractions": math.isclose(sum(fractions), 1.0, abs_tol=1e-12)
        and all(value > 0.0 for value in fractions),
        "split_overlap_forbidden": split.get("market_overlap_allowed") is False
        and split.get("chronological_overlap_allowed") is False,
        "fixed_objective": model.get("objective") == "reg:squarederror",
        "fixed_seed": isinstance(model.get("seed"), int),
        "deterministic_threads": model.get("nthread") == 1,
        "calibration_split_only": calibration.get("source_split")
        == "historical_calibration_only",
        "calibration_method": calibration.get("method")
        == "ridge_affine_by_action_family_v1",
        "validation_support": int(
            validation.get("minimum_validation_accepted_bet_count") or 0
        )
        >= 10,
        "future_support": int(future.get("minimum_unique_market_count") or 0) >= 30
        and int(future.get("minimum_accepted_bet_count") or 0) >= 30
        and int(future.get("minimum_accepted_bet_count_per_side") or 0) >= 10,
        "safety": safety.get("paper_only") is True
        and safety.get("capital_at_risk") is False
        and safety.get("polymarket_write_enabled") is False
        and safety.get("wallet_signing_enabled") is False
        and safety.get("v8_execution_handoff_allowed") is False
        and safety.get("source_model_candidate_eligible") is False
        and safety.get("freeze_ready") is False
        and safety.get("promotion_evidence_eligible") is False
        and safety.get("#134_resume_allowed") is False
        and safety.get("#146_start_allowed") is False,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("invalid hierarchical action-value protocol: " + ", ".join(failed))


def fit_historical_hierarchical_action_value(
    config: HierarchicalActionValueFitConfig,
) -> dict[str, Any]:
    """Fit fixed family heads and evaluate once on untouched historical validation."""

    protocol_path = config.protocol_path.resolve()
    _verify_file_pin(
        protocol_path,
        config.expected_protocol_sha256,
        name="hierarchical protocol",
    )
    protocol = _load_json(protocol_path)
    validate_hierarchical_action_value_protocol(protocol)

    source_protocol_path = config.source_action_protocol_path.resolve()
    _verify_file_pin(
        source_protocol_path,
        config.expected_source_action_protocol_sha256,
        name="source action-grid protocol",
    )
    source_protocol = _load_json(source_protocol_path)

    corpus_manifest_path = config.historical_corpus_manifest_path.resolve()
    corpus_manifest = _load_json(corpus_manifest_path)
    source_rows_descriptor = dict(corpus_manifest.get("development_rows") or {})
    source_rows_path = Path(str(source_rows_descriptor.get("path") or "")).resolve()
    if not source_rows_path.is_file():
        raise ValueError("historical development rows are missing")
    _verify_file_pin(
        source_rows_path,
        str(source_rows_descriptor.get("sha256") or ""),
        name="historical development rows",
    )
    source_rows = _load_jsonl(source_rows_path)
    if not source_rows:
        raise ValueError("historical development rows are empty")

    future_rows_path = config.excluded_future_decision_rows_path.resolve()
    _verify_file_pin(
        future_rows_path,
        config.expected_excluded_future_decision_rows_sha256,
        name="excluded future decision rows",
    )
    future_decision_rows = _load_jsonl(future_rows_path)
    future_forbidden = sorted(
        {
            field
            for row in future_decision_rows
            for field in _find_fields(row, FORBIDDEN_FUTURE_REGISTRY_FIELDS)
        }
    )
    if future_forbidden:
        raise ValueError(
            "excluded future market registry source contains outcome fields: "
            + ", ".join(future_forbidden)
        )
    future_market_ids = sorted(
        {str(row.get("market_id") or "") for row in future_decision_rows}
    )
    if not future_market_ids or "" in future_market_ids:
        raise ValueError("excluded future market registry source is incomplete")
    future_decision_timestamps = [
        int(row.get("decision_ts") or 0) for row in future_decision_rows
    ]
    if any(value <= 0 for value in future_decision_timestamps):
        raise ValueError("excluded future decision timestamp is invalid")

    artifact_descriptors = []
    for path, expected_sha256 in config.excluded_future_artifact_pins:
        resolved = path.resolve()
        _verify_file_pin(resolved, expected_sha256, name="excluded future artifact")
        artifact_descriptors.append(_descriptor(resolved))

    historical_market_ids = sorted({str(row["market_id"]) for row in source_rows})
    market_overlap = sorted(set(historical_market_ids) & set(future_market_ids))
    max_historical_decision_ts = max(int(row["decision_ts"]) for row in source_rows)
    min_excluded_future_decision_ts = min(future_decision_timestamps)
    if market_overlap:
        raise ValueError("historical corpus overlaps excluded future markets")
    if max_historical_decision_ts >= min_excluded_future_decision_ts:
        raise ValueError("historical corpus is not strictly earlier than excluded future rows")

    run_dir = config.output_dir / config.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    exclusion_registry = {
        "schema_version": f"{SCHEMA_PREFIX}-exclusion-registry-v1",
        "run_id": config.run_id,
        "excluded_future_decision_rows": _descriptor(future_rows_path),
        "excluded_future_artifacts": artifact_descriptors,
        "excluded_future_market_count": len(future_market_ids),
        "excluded_future_market_ids": future_market_ids,
        "excluded_future_market_ids_sha256": canonical_json_sha256(future_market_ids),
        "minimum_excluded_future_decision_ts": min_excluded_future_decision_ts,
        "historical_market_count": len(historical_market_ids),
        "maximum_historical_decision_ts": max_historical_decision_ts,
        "historical_future_market_overlap_count": 0,
        "historical_strictly_precedes_excluded_future": True,
        "excluded_future_artifacts_opened_for_hashing_only": True,
        "excluded_future_outcome_values_loaded": False,
        "excluded_future_pnl_values_loaded": False,
        "excluded_future_evidence_used_for_fitting_or_tuning": False,
        "paper_only": True,
        "capital_at_risk": False,
    }
    exclusion_registry["exclusion_registry_id"] = canonical_json_sha256(
        exclusion_registry
    )
    exclusion_path = run_dir / "historical_action_value_exclusion_registry.json"
    _write_json(exclusion_path, exclusion_registry)

    action_rows, source_audit = build_pnl_aligned_action_conditioned_rows(
        source_rows,
        protocol=source_protocol,
        require_targets=True,
    )
    if source_audit["blocking_reason_codes"]:
        raise ValueError(
            "source action-grid audit failed: "
            + ", ".join(source_audit["blocking_reason_codes"])
        )
    split_rows, split_summary = _chronological_market_split(
        action_rows,
        protocol=protocol,
    )
    split_paths: dict[str, Path] = {}
    for split_name, rows in split_rows.items():
        path = run_dir / f"hierarchical_action_value_{split_name}_rows.jsonl"
        _write_jsonl(path, rows)
        split_paths[split_name] = path

    split_manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-split-manifest-v1",
        "run_id": config.run_id,
        "protocol": _descriptor(protocol_path),
        "source_action_grid_protocol": _descriptor(source_protocol_path),
        "historical_corpus_manifest": _descriptor(corpus_manifest_path),
        "historical_development_rows": _descriptor(source_rows_path),
        "exclusion_registry": _descriptor(exclusion_path),
        "split_method": protocol["split_protocol"]["method"],
        "split_summary": split_summary,
        "split_rows": {
            name: _descriptor(path) for name, path in split_paths.items()
        },
        "market_overlap_count": 0,
        "chronology_validation_passed": True,
        "excluded_future_market_overlap_count": 0,
        "uses_historical_validation_labels_for_tuning": False,
        "uses_excluded_future_evidence_for_tuning": False,
        **_blocked_safety_fields(),
    }
    split_manifest["split_hash"] = canonical_json_sha256(
        {
            "protocol": split_manifest["protocol"]["sha256"],
            "exclusion_registry": split_manifest["exclusion_registry"]["sha256"],
            "summary": split_summary,
            "rows": {
                name: descriptor["sha256"]
                for name, descriptor in split_manifest["split_rows"].items()
            },
        }
    )
    split_manifest_path = run_dir / "hierarchical_action_value_split_manifest.json"
    _write_json(split_manifest_path, split_manifest)

    feature_columns = tuple(str(name) for name in protocol["feature_columns"])
    model_paths: dict[str, Path] = {}
    boosters: dict[str, xgb.Booster] = {}
    family_training_metrics: dict[str, Any] = {}
    raw_predictions_by_split: dict[str, dict[str, dict[str, float]]] = {
        split_name: {} for split_name in split_rows
    }
    for family in TRADE_FAMILIES:
        family_train_rows = [
            row
            for row in split_rows["historical_train"]
            if row["action_family"] == family
        ]
        booster = _train_family_booster(
            family_train_rows,
            feature_columns=feature_columns,
            model_protocol=dict(protocol["family_model_protocol"]),
        )
        model_path = run_dir / FAMILY_MODEL_FILENAMES[family]
        booster.save_model(model_path)
        model_paths[family] = model_path
        boosters[family] = booster
        for split_name, rows in split_rows.items():
            family_rows = [row for row in rows if row["action_family"] == family]
            values = _predict_booster(
                booster,
                family_rows,
                feature_columns=feature_columns,
            )
            raw_predictions_by_split[split_name][family] = {
                str(row["action_row_sha256"]): value
                for row, value in zip(family_rows, values, strict=True)
            }
        train_targets = [
            float(row["target_net_pnl_per_contract"]) for row in family_train_rows
        ]
        train_predictions = [
            raw_predictions_by_split["historical_train"][family][
                str(row["action_row_sha256"])
            ]
            for row in family_train_rows
        ]
        family_training_metrics[family] = _regression_metrics(
            train_targets,
            train_predictions,
        )

    calibration_parameters: dict[str, dict[str, float]] = {}
    calibration_metrics: dict[str, Any] = {}
    for family in TRADE_FAMILIES:
        calibration_rows = [
            row
            for row in split_rows["historical_calibration"]
            if row["action_family"] == family
        ]
        raw_values = [
            raw_predictions_by_split["historical_calibration"][family][
                str(row["action_row_sha256"])
            ]
            for row in calibration_rows
        ]
        targets = [float(row["target_net_pnl_per_contract"]) for row in calibration_rows]
        parameters = _fit_ridge_affine_calibration(
            raw_values,
            targets,
            dict(protocol["calibration_protocol"]),
        )
        calibrated = [_calibrate(value, parameters) for value in raw_values]
        calibration_parameters[family] = parameters
        calibration_metrics[family] = {
            "raw": _regression_metrics(targets, raw_values),
            "calibrated": _regression_metrics(targets, calibrated),
            "row_count": len(targets),
            "source_split": "historical_calibration_only",
        }
    calibration_artifact = {
        "schema_version": f"{SCHEMA_PREFIX}-calibration-artifact-v1",
        "candidate_name": protocol["candidate_name"],
        "method": protocol["calibration_protocol"]["method"],
        "source_split": "historical_calibration_only",
        "parameters_by_action_family": calibration_parameters,
        "uses_historical_validation_labels_for_tuning": False,
        "uses_excluded_future_evidence_for_tuning": False,
        "calibration_config_hash": canonical_json_sha256(
            protocol["calibration_protocol"]
        ),
        **_blocked_safety_fields(),
    }
    calibration_artifact["calibration_artifact_id"] = canonical_json_sha256(
        calibration_artifact
    )
    calibration_path = run_dir / "hierarchical_action_value_calibration_artifact.json"
    _write_json(calibration_path, calibration_artifact)

    validation_predictions = _materialize_split_predictions(
        split_rows["historical_validation"],
        raw_predictions=raw_predictions_by_split["historical_validation"],
        calibration_parameters=calibration_parameters,
    )
    validation_predictions_path = (
        run_dir / "hierarchical_action_value_historical_validation_predictions.jsonl"
    )
    _write_jsonl(validation_predictions_path, validation_predictions)
    candidate_replay = _run_historical_policy_replay(
        validation_predictions,
        entry_threshold=float(
            protocol["frozen_execution_contract"]["entry_edge_threshold"]
        ),
        policy_name=str(protocol["candidate_name"]),
        use_raw_probability_baseline=False,
    )
    baseline_replay = _run_historical_policy_replay(
        validation_predictions,
        entry_threshold=float(
            protocol["frozen_execution_contract"]["entry_edge_threshold"]
        ),
        policy_name="raw_market_probability_selected_o_action_baseline",
        use_raw_probability_baseline=True,
    )
    candidate_metrics = _accepted_bet_metrics(candidate_replay)
    baseline_metrics = _accepted_bet_metrics(baseline_replay)
    validation_calibration_metrics = _validation_calibration_metrics(
        validation_predictions
    )
    diagnostics = _market_robustness(candidate_replay, baseline_replay)
    gate = _historical_validation_gate(
        protocol=protocol,
        split_summary=split_summary,
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
        calibration_metrics=validation_calibration_metrics,
        diagnostics=diagnostics,
    )

    candidate_replay_path = (
        run_dir / "hierarchical_action_value_historical_validation_replay.jsonl"
    )
    baseline_replay_path = (
        run_dir / "hierarchical_action_value_historical_validation_baseline_replay.jsonl"
    )
    _write_jsonl(candidate_replay_path, candidate_replay)
    _write_jsonl(baseline_replay_path, baseline_replay)

    leakage_audit = {
        "schema_version": f"{SCHEMA_PREFIX}-leakage-audit-v1",
        "source_action_grid_audit": source_audit,
        "feature_max_input_ts_violation_count": source_audit[
            "feature_max_input_ts_violation_count"
        ],
        "forbidden_decision_field_violation_count": source_audit[
            "forbidden_decision_field_violation_count"
        ],
        "historical_future_market_overlap_count": 0,
        "split_market_overlap_count": 0,
        "split_chronology_validation_passed": True,
        "excluded_future_outcome_values_loaded": False,
        "excluded_future_pnl_values_loaded": False,
        "historical_validation_labels_used_for_tuning": False,
        "historical_validation_labels_used_for_report_only": True,
        "decision_time_inputs_only": True,
        "passed": source_audit["passed"] is True,
        **_blocked_safety_fields(),
    }
    leakage_path = run_dir / "hierarchical_action_value_leakage_audit.json"
    _write_json(leakage_path, leakage_audit)
    leakage_md_path = run_dir / "hierarchical_action_value_leakage_audit.md"
    _write_text(leakage_md_path, _leakage_markdown(leakage_audit))

    training_report = {
        "schema_version": f"{SCHEMA_PREFIX}-training-report-v1",
        "run_id": config.run_id,
        "candidate_name": protocol["candidate_name"],
        "model_family": protocol["family_model_protocol"]["model_family"],
        "models": {
            family: _descriptor(path) for family, path in model_paths.items()
        },
        "feature_columns": list(feature_columns),
        "training_metrics_by_family": family_training_metrics,
        "training_split_only": True,
        "uses_historical_validation_labels_for_tuning": False,
        "uses_excluded_future_evidence_for_tuning": False,
        **_blocked_safety_fields(),
    }
    training_report_path = run_dir / "hierarchical_action_value_training_report.json"
    _write_json(training_report_path, training_report)
    training_md_path = run_dir / "hierarchical_action_value_training_report.md"
    _write_text(training_md_path, _training_markdown(training_report))

    calibration_report = {
        "schema_version": f"{SCHEMA_PREFIX}-calibration-report-v1",
        "run_id": config.run_id,
        "calibration_artifact": _descriptor(calibration_path),
        "calibration_metrics_by_family": calibration_metrics,
        "source_split": "historical_calibration_only",
        "uses_historical_validation_labels_for_tuning": False,
        "uses_excluded_future_evidence_for_tuning": False,
        **_blocked_safety_fields(),
    }
    calibration_report_path = run_dir / "hierarchical_action_value_calibration_report.json"
    _write_json(calibration_report_path, calibration_report)
    calibration_md_path = run_dir / "hierarchical_action_value_calibration_report.md"
    _write_text(calibration_md_path, _calibration_markdown(calibration_report))

    validation_report = {
        "schema_version": f"{SCHEMA_PREFIX}-historical-validation-report-v1",
        "run_id": config.run_id,
        "candidate_name": protocol["candidate_name"],
        "candidate_metrics": candidate_metrics,
        "baseline_metrics": baseline_metrics,
        "candidate_minus_baseline_net_pnl": candidate_metrics["net_pnl_sum"]
        - baseline_metrics["net_pnl_sum"],
        "validation_calibration_metrics": validation_calibration_metrics,
        "market_robustness_diagnostics": diagnostics,
        "historical_validation_gate_checks": gate["checks"],
        "historical_validation_gate_passed": gate["passed"],
        "historical_validation_gate_blocking_reason_codes": gate["reason_codes"],
        "historical_validation_labels_used_for_report_only": True,
        "historical_validation_labels_used_for_tuning": False,
        "candidate_frozen_for_future_evaluation": gate["passed"],
        "future_collection_allowed": gate["passed"],
        **_blocked_safety_fields(),
    }
    validation_report_path = (
        run_dir / "hierarchical_action_value_historical_validation_report.json"
    )
    _write_json(validation_report_path, validation_report)
    validation_md_path = (
        run_dir / "hierarchical_action_value_historical_validation_report.md"
    )
    _write_text(validation_md_path, _validation_markdown(validation_report))

    freeze_manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-freeze-manifest-v1",
        "run_id": config.run_id,
        "candidate_name": protocol["candidate_name"],
        "protocol": _descriptor(protocol_path),
        "source_action_grid_protocol": _descriptor(source_protocol_path),
        "historical_corpus_manifest": _descriptor(corpus_manifest_path),
        "exclusion_registry": _descriptor(exclusion_path),
        "split_manifest": _descriptor(split_manifest_path),
        "models": {
            family: _descriptor(path) for family, path in model_paths.items()
        },
        "calibration_artifact": _descriptor(calibration_path),
        "training_report": _descriptor(training_report_path),
        "calibration_report": _descriptor(calibration_report_path),
        "historical_validation_report": _descriptor(validation_report_path),
        "leakage_audit": _descriptor(leakage_path),
        "candidate_frozen_for_future_evaluation": gate["passed"],
        "future_collection_allowed": gate["passed"],
        "future_unseen_evaluation_required": True,
        "historical_validation_gate_passed": gate["passed"],
        "historical_validation_gate_blocking_reason_codes": gate["reason_codes"],
        "uses_historical_validation_labels_for_tuning": False,
        "uses_excluded_future_evidence_for_tuning": False,
        **_blocked_safety_fields(),
    }
    freeze_manifest["research_candidate_hash"] = canonical_json_sha256(
        {
            "protocol": freeze_manifest["protocol"]["sha256"],
            "exclusion_registry": freeze_manifest["exclusion_registry"]["sha256"],
            "split_manifest": freeze_manifest["split_manifest"]["sha256"],
            "models": {
                family: descriptor["sha256"]
                for family, descriptor in freeze_manifest["models"].items()
            },
            "calibration": freeze_manifest["calibration_artifact"]["sha256"],
        }
    )
    freeze_manifest_path = run_dir / "hierarchical_action_value_freeze_manifest.json"
    _write_json(freeze_manifest_path, freeze_manifest)

    return {
        "run_dir": run_dir,
        "exclusion_registry_path": exclusion_path,
        "split_manifest_path": split_manifest_path,
        "training_report_path": training_report_path,
        "calibration_report_path": calibration_report_path,
        "validation_report_path": validation_report_path,
        "leakage_audit_path": leakage_path,
        "freeze_manifest_path": freeze_manifest_path,
        "freeze_manifest_sha256": _sha256_file(freeze_manifest_path),
        "validation_report": validation_report,
        "freeze_manifest": freeze_manifest,
    }


def predict_frozen_hierarchical_action_values(
    *,
    model_dir: Path | str,
    decision_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run outcome-blind inference only for a validation-approved frozen candidate."""

    model_dir = Path(model_dir).resolve()
    manifest_path = model_dir / "hierarchical_action_value_freeze_manifest.json"
    manifest = _load_json(manifest_path)
    if not (
        manifest.get("candidate_frozen_for_future_evaluation") is True
        and manifest.get("future_collection_allowed") is True
        and manifest.get("historical_validation_gate_passed") is True
    ):
        return [], {
            "status": "BLOCKED_FAIL_CLOSED_BEFORE_PREDICTION",
            "prediction_attempted": False,
            "blocking_reason_codes": list(
                manifest.get("historical_validation_gate_blocking_reason_codes") or []
            ),
            **_blocked_safety_fields(),
        }
    for descriptor in [
        manifest["protocol"],
        manifest["source_action_grid_protocol"],
        manifest["calibration_artifact"],
        *manifest["models"].values(),
    ]:
        _verify_file_pin(
            Path(descriptor["path"]),
            str(descriptor["sha256"]),
            name="frozen hierarchical artifact",
        )
    protocol = _load_json(Path(manifest["protocol"]["path"]))
    validate_hierarchical_action_value_protocol(protocol)
    source_protocol = _load_json(Path(manifest["source_action_grid_protocol"]["path"]))
    action_rows, audit = build_pnl_aligned_action_conditioned_rows(
        decision_rows,
        protocol=source_protocol,
        require_targets=False,
    )
    if not audit["passed"]:
        return [], {
            "status": "BLOCKED_FAIL_CLOSED_BEFORE_PREDICTION",
            "prediction_attempted": False,
            "blocking_reason_codes": audit["blocking_reason_codes"],
            **_blocked_safety_fields(),
        }
    calibration = _load_json(Path(manifest["calibration_artifact"]["path"]))
    feature_columns = tuple(protocol["feature_columns"])
    raw_predictions: dict[str, dict[str, float]] = {}
    for family in TRADE_FAMILIES:
        booster = xgb.Booster()
        booster.load_model(Path(manifest["models"][family]["path"]))
        rows = [row for row in action_rows if row["action_family"] == family]
        values = _predict_booster(booster, rows, feature_columns=feature_columns)
        raw_predictions[family] = {
            str(row["action_row_sha256"]): value
            for row, value in zip(rows, values, strict=True)
        }
    predictions = _materialize_split_predictions(
        action_rows,
        raw_predictions=raw_predictions,
        calibration_parameters=dict(calibration["parameters_by_action_family"]),
        include_targets=False,
    )
    return predictions, {
        "status": "OUTCOME_BLIND_HIERARCHICAL_PREDICTION_COMPLETE",
        "prediction_attempted": True,
        "prediction_count": len(predictions),
        "decision_count": len(predictions) // len(REQUIRED_ACTIONS),
        "complete_5_action_grid": len(predictions) == len(decision_rows) * 5,
        "outcome_fields_used": False,
        "realized_pnl_used": False,
        "source_o_score_mutated": False,
        "source_ranking_mutated": False,
        **_blocked_safety_fields(),
    }


def _chronological_market_split(
    rows: list[dict[str, Any]],
    *,
    protocol: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    market_first_ts: dict[str, int] = {}
    for row in rows:
        market_id = str(row["market_id"])
        market_first_ts[market_id] = min(
            market_first_ts.get(market_id, int(row["decision_ts"])),
            int(row["decision_ts"]),
        )
    ordered_markets = sorted(market_first_ts, key=lambda value: (market_first_ts[value], value))
    split_protocol = dict(protocol["split_protocol"])
    total = len(ordered_markets)
    if total < int(split_protocol["minimum_total_market_count"]):
        raise ValueError("insufficient historical market support")
    train_count = int(math.floor(total * float(split_protocol["train_market_fraction"])))
    calibration_count = int(
        math.floor(total * float(split_protocol["calibration_market_fraction"]))
    )
    validation_count = total - train_count - calibration_count
    counts = {
        "historical_train": train_count,
        "historical_calibration": calibration_count,
        "historical_validation": validation_count,
    }
    for split_name, minimum_field in (
        ("historical_train", "minimum_train_market_count"),
        ("historical_calibration", "minimum_calibration_market_count"),
        ("historical_validation", "minimum_validation_market_count"),
    ):
        if counts[split_name] < int(split_protocol[minimum_field]):
            raise ValueError(f"insufficient {split_name} market support")
    market_splits = {
        "historical_train": set(ordered_markets[:train_count]),
        "historical_calibration": set(
            ordered_markets[train_count : train_count + calibration_count]
        ),
        "historical_validation": set(ordered_markets[train_count + calibration_count :]),
    }
    output = {
        name: sorted(
            [row for row in rows if str(row["market_id"]) in market_ids],
            key=lambda row: (
                int(row["decision_ts"]),
                str(row["market_id"]),
                str(row["action"]),
            ),
        )
        for name, market_ids in market_splits.items()
    }
    boundaries = {}
    previous_max: int | None = None
    for name in ("historical_train", "historical_calibration", "historical_validation"):
        decision_ts = [int(row["decision_ts"]) for row in output[name]]
        minimum = min(decision_ts)
        maximum = max(decision_ts)
        if previous_max is not None and minimum <= previous_max:
            raise ValueError("historical market split chronology overlaps")
        previous_max = maximum
        boundaries[name] = {
            "minimum_decision_ts": minimum,
            "maximum_decision_ts": maximum,
        }
    overlap = (
        (market_splits["historical_train"] & market_splits["historical_calibration"])
        | (market_splits["historical_train"] & market_splits["historical_validation"])
        | (
            market_splits["historical_calibration"]
            & market_splits["historical_validation"]
        )
    )
    if overlap:
        raise ValueError("historical market split overlap")
    return output, {
        "total_market_count": total,
        "total_action_row_count": len(rows),
        "splits": {
            name: {
                "market_count": len(market_splits[name]),
                "market_ids": sorted(market_splits[name]),
                "market_ids_sha256": canonical_json_sha256(sorted(market_splits[name])),
                "action_row_count": len(output[name]),
                "decision_count": len(output[name]) // len(REQUIRED_ACTIONS),
                "support_by_family_side": _support_by_family_side(output[name]),
                **boundaries[name],
            }
            for name in output
        },
        "market_overlap_count": 0,
        "chronology_validation_passed": True,
    }


def _train_family_booster(
    rows: list[dict[str, Any]],
    *,
    feature_columns: tuple[str, ...],
    model_protocol: dict[str, Any],
) -> xgb.Booster:
    if not rows:
        raise ValueError("family training rows are empty")
    matrix = _feature_matrix(rows, feature_columns)
    labels = np.asarray(
        [float(row["target_net_pnl_per_contract"]) for row in rows],
        dtype=np.float64,
    )
    params = dict(model_protocol)
    params.pop("model_family", None)
    num_boost_round = int(params.pop("num_boost_round"))
    return xgb.train(
        params=params,
        dtrain=xgb.DMatrix(matrix, label=labels, feature_names=list(feature_columns)),
        num_boost_round=num_boost_round,
    )


def _predict_booster(
    booster: xgb.Booster,
    rows: list[dict[str, Any]],
    *,
    feature_columns: tuple[str, ...],
) -> list[float]:
    if not rows:
        return []
    values = booster.predict(
        xgb.DMatrix(_feature_matrix(rows, feature_columns), feature_names=list(feature_columns))
    )
    return [float(value) for value in values]


def _feature_matrix(
    rows: list[dict[str, Any]],
    feature_columns: tuple[str, ...],
) -> np.ndarray:
    matrix = np.asarray(
        [
            [float(row["decision_time_features"][name]) for name in feature_columns]
            for row in rows
        ],
        dtype=np.float64,
    )
    if not np.isfinite(matrix).all():
        raise ValueError("hierarchical action-value features must be finite")
    return matrix


def _fit_ridge_affine_calibration(
    predictions: list[float],
    targets: list[float],
    protocol: dict[str, Any],
) -> dict[str, float]:
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("calibration rows are incomplete")
    x = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.diag([0.0, float(protocol["ridge_lambda"])])
    intercept, slope = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    slope = float(
        np.clip(
            slope,
            float(protocol["minimum_slope"]),
            float(protocol["maximum_slope"]),
        )
    )
    intercept = float(
        np.clip(
            intercept,
            float(protocol["minimum_intercept"]),
            float(protocol["maximum_intercept"]),
        )
    )
    if not (math.isfinite(intercept) and math.isfinite(slope)):
        raise ValueError("calibration parameters must be finite")
    return {"intercept": intercept, "slope": slope}


def _calibrate(value: float, parameters: dict[str, float]) -> float:
    result = float(parameters["intercept"]) + float(parameters["slope"]) * value
    if not math.isfinite(result):
        raise ValueError("calibrated action value must be finite")
    return result


def _materialize_split_predictions(
    rows: list[dict[str, Any]],
    *,
    raw_predictions: dict[str, dict[str, float]],
    calibration_parameters: dict[str, dict[str, float]],
    include_targets: bool = True,
) -> list[dict[str, Any]]:
    predictions = []
    for row in rows:
        family = str(row["action_family"])
        if family == "NO_TRADE":
            raw_value = 0.0
            calibrated_value = 0.0
        else:
            raw_value = float(raw_predictions[family][str(row["action_row_sha256"])])
            calibrated_value = _calibrate(raw_value, calibration_parameters[family])
        prediction = {
            key: row[key]
            for key in (
                "market_id",
                "decision_ts",
                "market_close_ts",
                "max_input_ts",
                "source_run_id",
                "source_row_identity",
                "action",
                "side",
                "action_family",
            )
        }
        prediction.update(
            {
                "raw_family_model_expected_net_return": raw_value,
                "calibrated_expected_net_return": calibrated_value,
                "ranking_score_source": "calibrated_family_specific_expected_net_return",
                "decision_time_features": row["decision_time_features"],
                "execution_handoff_context": row["execution_handoff_context"],
                "target_used_as_decision_input": False,
                "outcome_fields_used_as_decision_input": False,
                "source_o_score_mutated": False,
                "source_ranking_mutated": False,
                "paper_only": True,
                "capital_at_risk": False,
            }
        )
        if include_targets:
            prediction["evaluation_target_net_pnl_per_contract"] = float(
                row["target_net_pnl_per_contract"]
            )
            prediction["evaluation_target_used_for_report_only"] = True
        prediction["prediction_sha256"] = canonical_json_sha256(prediction)
        predictions.append(prediction)
    return predictions


def _run_historical_policy_replay(
    predictions: list[dict[str, Any]],
    *,
    entry_threshold: float,
    policy_name: str,
    use_raw_probability_baseline: bool,
) -> list[dict[str, Any]]:
    by_decision: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_decision[(str(row["market_id"]), int(row["decision_ts"]))].append(row)
    guard_config = _v8_execution_guard_config()
    state = _v8_initial_runtime_state(guard_config)
    closes: dict[str, int] = {}
    replay: list[dict[str, Any]] = []
    for index, ((market_id, decision_ts), action_rows) in enumerate(
        sorted(by_decision.items(), key=lambda item: (item[0][1], item[0][0])),
        start=1,
    ):
        _release_closed_positions(
            state=state,
            market_close_by_open_position=closes,
            decision_ts=decision_ts,
        )
        if {row["action"] for row in action_rows} != set(REQUIRED_ACTIONS):
            raise ValueError("historical validation action grid is incomplete")
        if use_raw_probability_baseline:
            selected = max(
                action_rows,
                key=lambda row: (
                    float(row["decision_time_features"]["canonical_o_action_score"]),
                    str(row["action"]),
                ),
            )
            features = dict(selected["decision_time_features"])
            score = (
                float(features["selected_side_probability"])
                - float(features["execution_price"])
                - _decision_time_execution_cost(features)
            )
        else:
            selected = max(
                action_rows,
                key=lambda row: (
                    float(row["calibrated_expected_net_return"]),
                    str(row["action"]),
                ),
            )
            score = float(selected["calibrated_expected_net_return"])
        action = str(selected["action"])
        blockers: list[str] = []
        guard_row: dict[str, Any] | None = None
        if action == "NO_TRADE":
            blockers.append("policy_selected_no_trade")
        elif score < entry_threshold:
            blockers.append("expected_net_return_below_frozen_entry_threshold")
        else:
            guard_row = _v8_execution_guard_decision(
                dict(selected["execution_handoff_context"]),
                guard_config=guard_config,
                runtime_state=state,
                runtime_mode="simulated_runtime_state",
            )
            blockers.extend(guard_row["execution_blocking_reason_codes"])
        accepted = bool(guard_row and guard_row["order_allowed"])
        size = float(guard_row["proposed_order_size"]) if accepted else 0.0
        if accepted:
            order_id = f"{policy_name}-validation-{index:06d}"
            _v8_apply_simulated_order_to_state(
                state=state,
                decision=guard_row,
                simulated_order_id=order_id,
            )
            closes[market_id] = int(selected["market_close_ts"])
        target = float(selected["evaluation_target_net_pnl_per_contract"])
        execution_price = float(selected["decision_time_features"]["execution_price"])
        row = {
            "policy_name": policy_name,
            "source_row_identity": str(selected["source_row_identity"]),
            "market_id": market_id,
            "decision_ts": decision_ts,
            "selected_action": action,
            "selected_side": str(selected["side"]),
            "selected_action_family": str(selected["action_family"]),
            "decision_score": score,
            "frozen_entry_threshold": entry_threshold,
            "execution_guard_order_allowed": accepted,
            "proposed_order_size": size,
            "accepted_bet_cost_basis": execution_price * size,
            "accepted_bet_net_pnl": target * size if accepted else 0.0,
            "evaluation_target_used_after_selection_for_report_only": True,
            "execution_blocking_reason_codes": sorted(set(blockers)),
            "paper_only": True,
            "capital_at_risk": False,
        }
        row["replay_row_sha256"] = canonical_json_sha256(row)
        replay.append(row)
    return replay


def _accepted_bet_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row["execution_guard_order_allowed"]]
    pnl = sum(float(row["accepted_bet_net_pnl"]) for row in accepted)
    cost_basis = sum(float(row["accepted_bet_cost_basis"]) for row in accepted)
    return {
        "accepted_bet_count": len(accepted),
        "accepted_unique_market_count": len({row["market_id"] for row in accepted}),
        "accepted_bet_count_by_side": dict(
            sorted(Counter(row["selected_side"] for row in accepted).items())
        ),
        "accepted_bet_count_by_family": dict(
            sorted(Counter(row["selected_action_family"] for row in accepted).items())
        ),
        "accepted_bet_count_by_action": dict(
            sorted(Counter(row["selected_action"] for row in accepted).items())
        ),
        "cost_basis_sum": cost_basis,
        "net_pnl_sum": pnl,
        "roi": pnl / cost_basis if cost_basis > 0.0 else 0.0,
        "win_rate": (
            sum(float(row["accepted_bet_net_pnl"]) > 0.0 for row in accepted)
            / len(accepted)
            if accepted
            else 0.0
        ),
    }


def _validation_calibration_metrics(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    by_family: dict[str, Any] = {}
    all_targets: list[float] = []
    all_raw: list[float] = []
    all_calibrated: list[float] = []
    for family in TRADE_FAMILIES:
        rows = [row for row in predictions if row["action_family"] == family]
        targets = [float(row["evaluation_target_net_pnl_per_contract"]) for row in rows]
        raw = [float(row["raw_family_model_expected_net_return"]) for row in rows]
        calibrated = [float(row["calibrated_expected_net_return"]) for row in rows]
        by_family[family] = {
            "raw": _regression_metrics(targets, raw),
            "calibrated": _regression_metrics(targets, calibrated),
        }
        all_targets.extend(targets)
        all_raw.extend(raw)
        all_calibrated.extend(calibrated)
    return {
        "by_family": by_family,
        "overall_raw": _regression_metrics(all_targets, all_raw),
        "overall_calibrated": _regression_metrics(all_targets, all_calibrated),
        "historical_validation_labels_used_for_report_only": True,
        "historical_validation_labels_used_for_tuning": False,
    }


def _market_robustness(
    candidate_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate = _pnl_by_market(candidate_rows)
    baseline = _pnl_by_market(baseline_rows)
    markets = sorted(set(candidate) | set(baseline))
    deltas = {market: candidate.get(market, 0.0) - baseline.get(market, 0.0) for market in markets}
    full = sum(deltas.values())
    leave_one_out = [full - deltas[market] for market in markets]
    largest_winner = max(candidate.items(), key=lambda item: item[1], default=(None, 0.0))
    interval = _market_bootstrap_interval(deltas)
    return {
        "market_count": len(markets),
        "candidate_minus_baseline_net_pnl": full,
        "market_bootstrap_interval_95": interval,
        "leave_one_market_out": {
            "reported": True,
            "minimum_delta": min(leave_one_out) if leave_one_out else 0.0,
            "maximum_delta": max(leave_one_out) if leave_one_out else 0.0,
            "all_scenarios_positive": bool(leave_one_out)
            and all(value > 0.0 for value in leave_one_out),
        },
        "largest_winner_removal": {
            "reported": True,
            "largest_winning_market_id": largest_winner[0],
            "largest_winning_market_pnl": largest_winner[1],
            "candidate_net_pnl_after_removal": sum(candidate.values()) - largest_winner[1],
        },
    }


def _historical_validation_gate(
    *,
    protocol: dict[str, Any],
    split_summary: dict[str, Any],
    candidate_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    calibration_metrics: dict[str, Any],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    gates = dict(protocol["historical_validation_gates"])
    side_counts = dict(candidate_metrics["accepted_bet_count_by_side"])
    family_counts = dict(candidate_metrics["accepted_bet_count_by_family"])
    validation_support = dict(
        split_summary["splits"]["historical_validation"]["support_by_family_side"]
    )
    minimum_rows = int(gates["minimum_action_rows_per_family_side"])
    checks = {
        "validation_accepted_bet_support": candidate_metrics["accepted_bet_count"]
        >= int(gates["minimum_validation_accepted_bet_count"]),
        "validation_side_support": all(
            int(side_counts.get(side, 0))
            >= int(gates["minimum_validation_accepted_bet_count_per_side"])
            for side in ("UP", "DOWN")
        ),
        "validation_family_support": all(
            int(family_counts.get(family, 0))
            >= int(gates["minimum_validation_accepted_bet_count_per_family"])
            for family in TRADE_FAMILIES
        ),
        "validation_action_row_support": all(
            int(validation_support.get(f"{family}|{side}", 0)) >= minimum_rows
            for family in TRADE_FAMILIES
            for side in ("UP", "DOWN")
        ),
        "candidate_net_pnl_positive": candidate_metrics["net_pnl_sum"] > 0.0,
        "candidate_roi_positive": candidate_metrics["roi"] > 0.0,
        "candidate_better_than_raw_baseline": candidate_metrics["net_pnl_sum"]
        > baseline_metrics["net_pnl_sum"],
        "calibrated_mae_not_worse": calibration_metrics["overall_calibrated"]["mae"]
        <= calibration_metrics["overall_raw"]["mae"],
        "market_bootstrap_reported": bool(
            diagnostics["market_bootstrap_interval_95"].get("reported")
        ),
        "leave_one_market_out_reported": diagnostics["leave_one_market_out"][
            "reported"
        ]
        is True,
        "largest_winner_removal_reported": diagnostics["largest_winner_removal"][
            "reported"
        ]
        is True,
    }
    reason_map = {
        "validation_accepted_bet_support": "insufficient_historical_validation_accepted_bet_support",
        "validation_side_support": "insufficient_historical_validation_side_support",
        "validation_family_support": "insufficient_historical_validation_family_support",
        "validation_action_row_support": "insufficient_historical_validation_action_row_support",
        "candidate_net_pnl_positive": "historical_validation_candidate_net_pnl_not_positive",
        "candidate_roi_positive": "historical_validation_candidate_roi_not_positive",
        "candidate_better_than_raw_baseline": "historical_validation_candidate_not_better_than_baseline",
        "calibrated_mae_not_worse": "historical_validation_calibration_mae_worse_than_raw",
        "market_bootstrap_reported": "historical_validation_market_bootstrap_missing",
        "leave_one_market_out_reported": "historical_validation_leave_one_market_out_missing",
        "largest_winner_removal_reported": "historical_validation_largest_winner_removal_missing",
    }
    reasons = [reason_map[name] for name, passed in checks.items() if not passed]
    return {"passed": not reasons, "checks": checks, "reason_codes": reasons}


def _market_bootstrap_interval(deltas: dict[str, float]) -> dict[str, Any]:
    values = np.asarray([deltas[key] for key in sorted(deltas)], dtype=np.float64)
    if values.size == 0:
        return {"reported": False, "lower": None, "upper": None, "samples": 0}
    rng = np.random.default_rng(20260715)
    samples = np.asarray(
        [float(rng.choice(values, size=len(values), replace=True).sum()) for _ in range(2000)]
    )
    return {
        "reported": True,
        "sampling_unit": "market_id",
        "samples": 2000,
        "lower": float(np.quantile(samples, 0.025)),
        "upper": float(np.quantile(samples, 0.975)),
    }


def _pnl_by_market(rows: list[dict[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = defaultdict(float)
    for row in rows:
        if row["execution_guard_order_allowed"]:
            result[str(row["market_id"])] += float(row["accepted_bet_net_pnl"])
    return dict(result)


def _decision_time_execution_cost(features: dict[str, Any]) -> float:
    spread = max(float(features.get("spread_bps") or 0.0), 0.0) / 20000.0
    queue = min(max(float(features.get("queue_fill_proxy") or 0.0), 0.0), 1.0)
    staleness = max(float(features.get("book_staleness_ms") or 0.0), 0.0)
    return min(
        0.05,
        0.001
        + spread
        + (1.0 - queue) * 0.002
        + min(staleness / 1000.0, 1.0) * 0.001,
    )


def _release_closed_positions(
    *,
    state: dict[str, Any],
    market_close_by_open_position: dict[str, int],
    decision_ts: int,
) -> None:
    closed = sorted(
        market_id
        for market_id, close_ts in market_close_by_open_position.items()
        if close_ts <= decision_ts
    )
    for market_id in closed:
        position = state["open_position_by_market_id"].pop(market_id, None)
        market_close_by_open_position.pop(market_id, None)
        if not isinstance(position, dict):
            continue
        side = str(position.get("side") or "NONE")
        size = float(position.get("notional") or 0.0)
        state["open_position_by_market_side"].pop(f"{market_id}|{side}", None)
        state["current_market_exposure_by_market_id"].pop(market_id, None)
        state["current_side_exposure_by_side"][side] = max(
            0.0,
            float(state["current_side_exposure_by_side"].get(side) or 0.0) - size,
        )
        state["current_total_exposure"] = max(
            0.0,
            float(state.get("current_total_exposure") or 0.0) - size,
        )


def _support_by_family_side(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                f"{row['action_family']}|{row['side']}"
                for row in rows
                if row["action_family"] in TRADE_FAMILIES
            ).items()
        )
    )


def _regression_metrics(targets: list[float], predictions: list[float]) -> dict[str, Any]:
    if len(targets) != len(predictions) or not targets:
        raise ValueError("regression metrics require aligned non-empty rows")
    errors = [prediction - target for target, prediction in zip(targets, predictions, strict=True)]
    return {
        "row_count": len(targets),
        "mae": sum(abs(value) for value in errors) / len(errors),
        "mse": sum(value * value for value in errors) / len(errors),
        "target_mean": sum(targets) / len(targets),
        "prediction_mean": sum(predictions) / len(predictions),
    }


def _find_fields(payload: Any, forbidden: set[str], prefix: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key) in forbidden:
                found.add(path)
            found.update(_find_fields(value, forbidden, path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.update(_find_fields(value, forbidden, f"{prefix}[{index}]"))
    return found


def _blocked_safety_fields() -> dict[str, Any]:
    return {
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }


def _training_markdown(report: dict[str, Any]) -> str:
    lines = ["# Hierarchical Action-Value Training", ""]
    lines.append(f"- candidate: `{report['candidate_name']}`")
    lines.append("- fitting source: `historical_train_only`")
    for family, metrics in report["training_metrics_by_family"].items():
        lines.append(f"- {family} train rows: `{metrics['row_count']}`")
        lines.append(f"- {family} train MAE: `{metrics['mae']}`")
    lines.extend(["- validation labels used for tuning: `false`", "- paper/live unlock: `false`", ""])
    return "\n".join(lines)


def _calibration_markdown(report: dict[str, Any]) -> str:
    lines = ["# Hierarchical Action-Value Calibration", "", "- source split: `historical_calibration_only`"]
    for family, metrics in report["calibration_metrics_by_family"].items():
        lines.append(f"- {family} raw MAE: `{metrics['raw']['mae']}`")
        lines.append(f"- {family} calibrated MAE: `{metrics['calibrated']['mae']}`")
    lines.extend(["- validation labels used for tuning: `false`", "- paper/live unlock: `false`", ""])
    return "\n".join(lines)


def _validation_markdown(report: dict[str, Any]) -> str:
    candidate = report["candidate_metrics"]
    baseline = report["baseline_metrics"]
    return "\n".join(
        [
            "# Hierarchical Action-Value Historical Validation",
            "",
            f"- gate passed: `{str(report['historical_validation_gate_passed']).lower()}`",
            f"- candidate accepted bets: `{candidate['accepted_bet_count']}`",
            f"- candidate net PnL: `{candidate['net_pnl_sum']}`",
            f"- candidate ROI: `{candidate['roi']}`",
            f"- baseline net PnL: `{baseline['net_pnl_sum']}`",
            f"- blockers: `{report['historical_validation_gate_blocking_reason_codes']}`",
            "- validation labels used for tuning: `false`",
            "- future unseen evaluation still required: `true`",
            "- source/freeze/promotion/paper/live unlock: `false`",
            "",
        ]
    )


def _leakage_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Hierarchical Action-Value Leakage Audit",
            "",
            f"- passed: `{str(report['passed']).lower()}`",
            f"- timestamp violations: `{report['feature_max_input_ts_violation_count']}`",
            f"- forbidden input violations: `{report['forbidden_decision_field_violation_count']}`",
            "- excluded future outcomes/PnL loaded: `false`",
            "- historical validation labels used for tuning: `false`",
            "",
        ]
    )


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def _verify_file_pin(path: Path, expected_sha256: str, *, name: str) -> None:
    if not path.is_file():
        raise ValueError(f"{name} is missing: {path}")
    _require_sha256(expected_sha256, name=f"{name} SHA-256")
    if _sha256_file(path) != expected_sha256.lower():
        raise ValueError(f"{name} SHA-256 mismatch")


def _require_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise ValueError(f"{name} must be a SHA-256 digest")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"expected JSON object row: {path}")
                rows.append(payload)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")
