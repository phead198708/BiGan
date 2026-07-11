"""Statistically valid v2 calibration protocol for regime-conditioned EV.

Settled net returns are accepted only as historical targets.  The fitted
artifact contains decision-time transforms and coefficients, while validation
and future-shadow evidence remain separate and fail closed.
"""

from __future__ import annotations

import json
import math
import random
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2 import (
    EXECUTION_LAYER_V2_FORBIDDEN_OUTCOME_FIELDS,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_replay import (
    _load_forward_shadow_rows,
    _lookup_value,
    _recursive_forbidden_field_paths,
    _safety_report_fields,
    _sha256_file,
    _write_json,
    _write_text,
)
from bigan.v8.polymarket.training.execution_layer_v2_regime_conditioned_ev import (
    CURRENT_75_ROW_REPLAY_RUN_ID,
    FROZEN_REGIME_CONDITIONED_EV_V2_ARTIFACT_NAME,
    FROZEN_REGIME_CONDITIONED_EV_V2_SCHEMA_VERSION,
    LATEST_ONE_HOUR_RECONCILED_RUN_ID,
    REGIME_CONDITIONED_EV_V2_FEATURE_GROUPS,
    ExecutionLayerV2RegimeConditionedEVForwardShadowConfig,
    run_execution_layer_v2_regime_conditioned_ev_forward_shadow,
    validate_frozen_regime_conditioned_ev_artifact,
)

REGIME_CONDITIONED_EV_V2_CALIBRATION_REPORT_SCHEMA_VERSION = (
    "bigan-v8-execution-layer-v2-regime-conditioned-ev-v2-calibration-report-v1"
)
REGIME_CONDITIONED_EV_V2_SPLIT_REPORT_SCHEMA_VERSION = (
    "bigan-v8-execution-layer-v2-regime-conditioned-ev-v2-split-report-v1"
)
REGIME_CONDITIONED_EV_V2_PROTOCOL_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-execution-layer-v2-regime-conditioned-ev-v2-protocol-manifest-v1"
)
V2_REQUIRED_FEATURES = tuple(
    feature
    for features in REGIME_CONDITIONED_EV_V2_FEATURE_GROUPS.values()
    for feature in features
)
V2_REQUIRED_EXCLUDED_RUN_IDS = (
    CURRENT_75_ROW_REPLAY_RUN_ID,
    LATEST_ONE_HOUR_RECONCILED_RUN_ID,
)
V2_CALIBRATION_ROW_SCHEMA_PATH = (
    Path(__file__).resolve().parents[5]
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_regime_conditioned_ev_calibration_row_v2.schema.json"
)
V2_APPROVED_TARGET_PROVENANCE_SOURCES = (
    "polymarket_clob_read_only_settlement",
    "polymarket_gamma_read_only_settlement",
    "paper_ledger_read_only_settlement_reconciliation",
)
V2_ACTION_CONTRACT = {
    "BUY_UP_HOLD_TO_SETTLEMENT": ("UP", "HOLD_TO_SETTLEMENT"),
    "BUY_DOWN_HOLD_TO_SETTLEMENT": ("DOWN", "HOLD_TO_SETTLEMENT"),
    "BUY_UP_SELL_BEFORE_CLOSE": ("UP", "SELL_BEFORE_CLOSE"),
    "BUY_DOWN_SELL_BEFORE_CLOSE": ("DOWN", "SELL_BEFORE_CLOSE"),
}
V2_REQUIRED_VALIDATION_SIDES = ("UP", "DOWN")
V2_REQUIRED_VALIDATION_ACTION_FAMILIES = (
    "HOLD_TO_SETTLEMENT",
    "SELL_BEFORE_CLOSE",
)
V2_REQUIRED_VALIDATION_RESOLVED_OUTCOMES = ("UP", "DOWN")
_CALIBRATION_ROW_FIELDS = {
    "source_run_id",
    "source_intent_id",
    "source_fill_id",
    "row_identity",
    "source_lineage",
    "market_id",
    "decision_ts",
    "max_input_ts",
    "market_close_ts",
    "selected_side",
    "selected_action",
    "action_family",
    "decision_time_features",
    "target_net_return_after_cost",
    "target_provenance",
}
_GROUP_WEIGHTS: dict[str, dict[str, float]] = {
    "canonical_o_score_and_action_margin": {
        "canonical_o_action_score": 0.5,
        "action_score_margin": 0.5,
    },
    "btc_anchor_direction": {
        "btc_momentum": 0.5,
        "reference_price_to_beat_distance_at_decision": 0.5,
    },
    "market_price_value": {
        "selected_side_probability": 0.25,
        "execution_price": -0.25,
        "selected_side_probability_minus_execution_price": 0.5,
    },
    "execution_quality": {
        "spread_bps": -0.25,
        "book_staleness_ms": -0.25,
        "queue_fill_proxy": 0.25,
        "time_to_close_seconds": 0.25,
    },
    "pre_entry_exposure_state": {
        "entry_index_within_market": -0.25,
        "cumulative_market_exposure_before_entry": -0.25,
        "same_side_reentry": 0.25,
        "side_flip": -0.25,
    },
}


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2RegimeConditionedEVCalibrationConfig:
    """Configuration for a bounded historical-fit/validation protocol."""

    run_id: str
    input_path: Path | str
    output_dir: Path | str
    future_shadow_input_path: Path | str | None = None
    validation_fraction: float = 0.25
    ridge_alpha: float = 1.0
    entry_ev_threshold: float = 0.02
    min_fit_rows: int = 100
    min_validation_rows: int = 30
    min_fit_markets: int = 20
    min_validation_markets: int = 10
    max_abs_coefficient: float = 2.0
    probability_price_tolerance: float = 1e-9
    min_relative_mae_improvement: float = 0.05
    min_relative_mse_improvement: float = 0.05
    bootstrap_samples: int = 1_000
    bootstrap_confidence_level: float = 0.95
    min_bootstrap_improvement_lower_bound: float = 0.0
    max_lomo_coefficient_absolute_deviation: float = 0.50
    min_lomo_coefficient_sign_agreement: float = 0.75
    min_validation_rows_per_side: int = 5
    min_validation_rows_per_action_family: int = 5
    min_validation_rows_per_resolved_outcome: int = 5
    min_validation_markets_per_category: int = 2
    statistical_random_seed: int = 17_029
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between zero and 0.5")
        for name in (
            "ridge_alpha",
            "entry_ev_threshold",
            "max_abs_coefficient",
            "probability_price_tolerance",
            "min_relative_mae_improvement",
            "min_relative_mse_improvement",
            "min_bootstrap_improvement_lower_bound",
            "max_lomo_coefficient_absolute_deviation",
            "min_lomo_coefficient_sign_agreement",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "min_fit_rows",
            "min_validation_rows",
            "min_fit_markets",
            "min_validation_markets",
            "bootstrap_samples",
            "min_validation_rows_per_side",
            "min_validation_rows_per_action_family",
            "min_validation_rows_per_resolved_outcome",
            "min_validation_markets_per_category",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.bootstrap_confidence_level < 1.0:
            raise ValueError("bootstrap_confidence_level must be in (0, 1)")
        for name in (
            "min_relative_mae_improvement",
            "min_relative_mse_improvement",
        ):
            if float(getattr(self, name)) > 1.0:
                raise ValueError(f"{name} must be <= 1")
        if not 0.0 <= self.min_lomo_coefficient_sign_agreement <= 1.0:
            raise ValueError(
                "min_lomo_coefficient_sign_agreement must be in [0, 1]"
            )
        object.__setattr__(self, "input_path", Path(self.input_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.future_shadow_input_path is not None:
            object.__setattr__(
                self,
                "future_shadow_input_path",
                Path(self.future_shadow_input_path),
            )

    @property
    def run_dir(self) -> Path:
        return self.output_dir.expanduser().resolve() / self.run_id

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_path"] = str(self.input_path)
        payload["output_dir"] = str(self.output_dir)
        if self.future_shadow_input_path is not None:
            payload["future_shadow_input_path"] = str(
                self.future_shadow_input_path
            )
        return payload


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2RegimeConditionedEVCalibrationResult:
    output_dir: Path
    artifact_paths: dict[str, Path]
    artifact_hashes: dict[str, str]
    split_report: dict[str, Any]
    calibration_report: dict[str, Any]
    manifest: dict[str, Any]


def regime_conditioned_ev_v2_calibration_row_identity(
    *,
    source_run_id: str,
    market_id: str,
    decision_ts: float | int,
    selected_action: str,
    source_intent_id: str,
    source_fill_id: str,
) -> str:
    """Return the stable economic decision/fill identity used by #167."""

    return canonical_json_sha256(
        {
            "source_run_id": source_run_id,
            "market_id": market_id,
            "decision_ts": decision_ts,
            "selected_action": selected_action,
            "source_intent_id": source_intent_id,
            "source_fill_id": source_fill_id,
        }
    )


def validate_regime_conditioned_ev_v2_calibration_rows(
    rows: list[dict[str, Any]],
    *,
    source_root: Path | str,
    probability_price_tolerance: float = 1e-9,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Reuse the exact v2 schema/runtime validator for corpus construction."""

    return _normalize_calibration_rows(
        rows,
        source_root=Path(source_root),
        probability_price_tolerance=probability_price_tolerance,
    )


def run_execution_layer_v2_regime_conditioned_ev_calibration(
    config: ExecutionLayerV2RegimeConditionedEVCalibrationConfig,
) -> ExecutionLayerV2RegimeConditionedEVCalibrationResult:
    """Fit historical rows, evaluate once on validation, then shadow if valid."""

    if not config.input_path.exists():
        raise FileNotFoundError(f"calibration input not found: {config.input_path}")
    run_dir = config.run_dir
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"calibration output exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    input_file = _resolve_input_file(config.input_path)
    raw_rows = _load_rows(input_file)
    normalized_rows, invalid_rows, excluded_rows = (
        validate_regime_conditioned_ev_v2_calibration_rows(
        raw_rows,
        source_root=input_file.parent,
        probability_price_tolerance=config.probability_price_tolerance,
        )
    )
    fit_rows, validation_rows, split_reasons = _chronological_market_split(
        normalized_rows,
        validation_fraction=config.validation_fraction,
        min_fit_rows=config.min_fit_rows,
        min_validation_rows=config.min_validation_rows,
        min_fit_markets=config.min_fit_markets,
        min_validation_markets=config.min_validation_markets,
    )
    split_report = _build_split_report(
        config,
        raw_rows=raw_rows,
        normalized_rows=normalized_rows,
        invalid_rows=invalid_rows,
        excluded_rows=excluded_rows,
        fit_rows=fit_rows,
        validation_rows=validation_rows,
        split_reasons=split_reasons,
    )

    fit_result: dict[str, Any] | None = None
    artifact_payload: dict[str, Any] | None = None
    eligibility_reasons = list(split_report["blocking_reason_codes"])
    if not eligibility_reasons:
        fit_result = _fit_and_evaluate(
            fit_rows,
            validation_rows,
            ridge_alpha=config.ridge_alpha,
            max_abs_coefficient=config.max_abs_coefficient,
            min_relative_mae_improvement=config.min_relative_mae_improvement,
            min_relative_mse_improvement=config.min_relative_mse_improvement,
            bootstrap_samples=config.bootstrap_samples,
            bootstrap_confidence_level=config.bootstrap_confidence_level,
            min_bootstrap_improvement_lower_bound=(
                config.min_bootstrap_improvement_lower_bound
            ),
            max_lomo_coefficient_absolute_deviation=(
                config.max_lomo_coefficient_absolute_deviation
            ),
            min_lomo_coefficient_sign_agreement=(
                config.min_lomo_coefficient_sign_agreement
            ),
            min_validation_rows_per_side=config.min_validation_rows_per_side,
            min_validation_rows_per_action_family=(
                config.min_validation_rows_per_action_family
            ),
            min_validation_rows_per_resolved_outcome=(
                config.min_validation_rows_per_resolved_outcome
            ),
            min_validation_markets_per_category=(
                config.min_validation_markets_per_category
            ),
            statistical_random_seed=config.statistical_random_seed,
        )
        eligibility_reasons.extend(fit_result["blocking_reason_codes"])
        if not eligibility_reasons:
            artifact_payload = _build_frozen_artifact(
                config,
                fit_rows=fit_rows,
                validation_rows=validation_rows,
                fit_result=fit_result,
            )

    artifact_paths = {
        "split_report": run_dir
        / "execution_layer_v2_regime_conditioned_ev_v2_split_report.json",
        "split_summary": run_dir
        / "execution_layer_v2_regime_conditioned_ev_v2_split_report.md",
        "calibration_report": run_dir
        / "execution_layer_v2_regime_conditioned_ev_v2_calibration_report.json",
        "calibration_summary": run_dir
        / "execution_layer_v2_regime_conditioned_ev_v2_calibration_report.md",
        "protocol_manifest": run_dir
        / "execution_layer_v2_regime_conditioned_ev_v2_protocol_manifest.json",
    }
    if artifact_payload is not None:
        artifact_paths["frozen_artifact"] = (
            run_dir / "execution_layer_v2_frozen_regime_conditioned_ev_v2.json"
        )
        _write_json(artifact_paths["frozen_artifact"], artifact_payload)
        validation = validate_frozen_regime_conditioned_ev_artifact(
            artifact_paths["frozen_artifact"]
        )
        if not validation["valid"]:
            eligibility_reasons.extend(validation["blocking_reason_codes"])
            artifact_paths["frozen_artifact"].unlink()
            artifact_paths.pop("frozen_artifact")
            artifact_payload = None

    future_shadow_summary = _run_future_shadow_if_ready(
        config,
        run_dir=run_dir,
        artifact_path=artifact_paths.get("frozen_artifact"),
        fit_rows=fit_rows,
        validation_rows=validation_rows,
    )
    calibration_report = _build_calibration_report(
        config,
        split_report=split_report,
        fit_result=fit_result,
        artifact_path=artifact_paths.get("frozen_artifact"),
        eligibility_reasons=sorted(set(eligibility_reasons)),
        future_shadow_summary=future_shadow_summary,
    )
    _write_json(artifact_paths["split_report"], split_report)
    _write_text(artifact_paths["split_summary"], _split_report_markdown(split_report))
    _write_json(artifact_paths["calibration_report"], calibration_report)
    _write_text(
        artifact_paths["calibration_summary"],
        _calibration_report_markdown(calibration_report),
    )

    artifact_hashes = {
        name: _sha256_file(path)
        for name, path in artifact_paths.items()
        if name != "protocol_manifest"
    }
    manifest = {
        "schema_version": REGIME_CONDITIONED_EV_V2_PROTOCOL_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "input_path": str(input_file),
        "input_sha256": _sha256_file(input_file),
        "split_report_id": split_report["report_id"],
        "calibration_report_id": calibration_report["report_id"],
        "artifact_created": artifact_payload is not None,
        "artifact_path": (
            str(artifact_paths["frozen_artifact"])
            if "frozen_artifact" in artifact_paths
            else None
        ),
        "artifact_sha256": artifact_hashes.get("frozen_artifact"),
        "fit_dataset_hash": split_report["fit_dataset_hash"],
        "validation_dataset_hash": split_report["validation_dataset_hash"],
        "split_hash": split_report["split_hash"],
        "calibration_row_schema_sha256": split_report[
            "calibration_row_schema_sha256"
        ],
        "target_observation_time_contract": split_report[
            "target_observation_time_contract"
        ],
        "leakage_checks_passed": split_report["leakage_checks_passed"],
        "validation_improved_over_constant_and_legacy": calibration_report[
            "validation_improved_over_constant_and_legacy"
        ],
        "statistical_eligibility_passed": calibration_report[
            "statistical_eligibility_passed"
        ],
        "statistical_eligibility_config_hash": calibration_report[
            "statistical_eligibility_config_hash"
        ],
        "statistical_eligibility_summary_hash": calibration_report[
            "statistical_eligibility_summary_hash"
        ],
        "schema_runtime_validation_agreement_passed": split_report[
            "schema_runtime_validation_agreement_passed"
        ],
        "invalid_row_reason_distribution": split_report[
            "invalid_row_reason_distribution"
        ],
        "final_artifact_eligibility_reason_codes": calibration_report[
            "blocking_reason_codes"
        ],
        "future_shadow": future_shadow_summary,
        "outcomes_reconciled_after_shadow_window": False,
        "refit_from_future_shadow_result_allowed": False,
        "production_gate_implemented": False,
        "source_scores_mutated": False,
        "o_score_mutated": False,
        **_safety_report_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    _write_json(artifact_paths["protocol_manifest"], manifest)
    artifact_hashes["protocol_manifest"] = _sha256_file(
        artifact_paths["protocol_manifest"]
    )
    return ExecutionLayerV2RegimeConditionedEVCalibrationResult(
        output_dir=run_dir,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        split_report=split_report,
        calibration_report=calibration_report,
        manifest=manifest,
    )


def _resolve_input_file(path: Path) -> Path:
    if path.is_dir():
        candidates = sorted(path.glob("*.jsonl")) + sorted(path.glob("*.json"))
        if len(candidates) != 1:
            raise ValueError("calibration directory must contain one JSON/JSONL input")
        return candidates[0]
    return path


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("calibration input must contain object rows")
    return rows


def _normalize_calibration_rows(
    rows: list[dict[str, Any]],
    *,
    source_root: Path,
    probability_price_tolerance: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    normalized: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    schema_validator = _calibration_row_schema_validator()
    for index, row in enumerate(rows):
        source_run_id = str(row.get("source_run_id") or "")
        schema_reason_codes = _schema_validation_reason_codes(
            schema_validator, row
        )
        runtime_reason_codes: list[str] = []
        unknown_fields = sorted(set(row) - _CALIBRATION_ROW_FIELDS)
        runtime_reason_codes.extend(
            f"calibration_row_unknown_top_level_field:{field}"
            for field in unknown_fields
        )
        features = row.get("decision_time_features")
        if not isinstance(features, dict):
            runtime_reason_codes.append("decision_time_features_missing")
            features = {}
        forbidden = _recursive_forbidden_field_paths(
            features, set(EXECUTION_LAYER_V2_FORBIDDEN_OUTCOME_FIELDS)
        )
        if forbidden:
            runtime_reason_codes.append(
                "forbidden_outcome_field_present_in_decision_time_inputs"
            )
        decision_ts = _finite_float(row.get("decision_ts"))
        max_input_ts = _finite_float(row.get("max_input_ts"))
        market_close_ts = _finite_float(row.get("market_close_ts"))
        target = _finite_float(row.get("target_net_return_after_cost"))
        target_provenance = row.get("target_provenance")
        source_intent_id = str(row.get("source_intent_id") or "")
        source_fill_id = str(row.get("source_fill_id") or "")
        source_lineage = row.get("source_lineage")
        if not str(row.get("market_id") or ""):
            runtime_reason_codes.append("market_id_missing")
        if not source_run_id:
            runtime_reason_codes.append("source_run_id_missing")
        if not source_intent_id:
            runtime_reason_codes.append("source_intent_id_missing")
        if not source_fill_id:
            runtime_reason_codes.append("source_fill_id_missing")
        if decision_ts is None:
            runtime_reason_codes.append("decision_ts_invalid")
        if max_input_ts is None:
            runtime_reason_codes.append("max_input_ts_invalid")
        elif decision_ts is not None and max_input_ts > decision_ts:
            runtime_reason_codes.append("feature_max_input_ts_after_decision_ts")
        if market_close_ts is None:
            runtime_reason_codes.append("market_close_ts_invalid")
        elif decision_ts is not None and market_close_ts <= decision_ts:
            runtime_reason_codes.append("market_close_ts_not_after_decision_ts")
        if target is None:
            runtime_reason_codes.append("target_net_return_after_cost_invalid")
        if not isinstance(target_provenance, dict):
            runtime_reason_codes.append("target_provenance_missing")
            target_provenance = {}
        else:
            source_type = str(target_provenance.get("source_type") or "")
            if source_type not in V2_APPROVED_TARGET_PROVENANCE_SOURCES:
                runtime_reason_codes.append(
                    "target_provenance_source_not_approved_read_only_settlement"
                )
            if not str(target_provenance.get("source_artifact_path") or ""):
                runtime_reason_codes.append(
                    "target_provenance_source_artifact_path_missing"
                )
            if not _is_sha256(target_provenance.get("source_artifact_sha256")):
                runtime_reason_codes.append(
                    "target_provenance_source_artifact_sha256_invalid"
                )
            source_path_text = str(
                target_provenance.get("source_artifact_path") or ""
            )
            source_path = Path(source_path_text).expanduser()
            if source_path_text and not source_path.is_absolute():
                source_path = source_root / source_path
            if source_path_text and not source_path.is_file():
                runtime_reason_codes.append(
                    "target_provenance_source_artifact_not_found"
                )
            elif source_path.is_file() and _sha256_file(source_path) != str(
                target_provenance.get("source_artifact_sha256")
            ):
                runtime_reason_codes.append(
                    "target_provenance_source_artifact_sha256_mismatch"
                )
        resolution_status = str(
            target_provenance.get("resolution_status") or ""
        ).lower()
        resolved_outcome = str(
            target_provenance.get("resolved_outcome") or ""
        ).upper()
        outcome_observed_at_ts = _finite_float(
            target_provenance.get("outcome_observed_at_ts")
        )
        observation_time_source = str(
            target_provenance.get("outcome_observation_time_source") or ""
        )
        if resolution_status != "resolved":
            runtime_reason_codes.append("target_resolution_status_not_resolved")
        if resolved_outcome not in V2_REQUIRED_VALIDATION_RESOLVED_OUTCOMES:
            runtime_reason_codes.append("resolved_outcome_invalid")
        if target_provenance.get("outcome_observed_after_market_close") is not True:
            runtime_reason_codes.append(
                "outcome_observed_after_market_close_not_verified"
            )
        if observation_time_source not in {
            "provider_response_clock",
            "artifact_recorded",
            "not_recorded_historical",
        }:
            runtime_reason_codes.append("outcome_observation_time_source_invalid")
        if outcome_observed_at_ts is not None:
            if market_close_ts is not None and outcome_observed_at_ts < market_close_ts:
                runtime_reason_codes.append("outcome_observed_before_market_close")
            if observation_time_source == "not_recorded_historical":
                runtime_reason_codes.append(
                    "outcome_observation_timestamp_source_inconsistent"
                )
        elif observation_time_source != "not_recorded_historical":
            runtime_reason_codes.append(
                "outcome_observation_timestamp_missing_for_recorded_source"
            )

        missing = [
            feature
            for feature in V2_REQUIRED_FEATURES
            if _finite_float(features.get(feature)) is None
        ]
        runtime_reason_codes.extend(
            f"required_feature_missing:{feature}" for feature in missing
        )
        selected_side = str(row.get("selected_side") or "").upper()
        selected_action = str(row.get("selected_action") or "")
        action_family = str(row.get("action_family") or "")
        if selected_side not in V2_REQUIRED_VALIDATION_SIDES:
            runtime_reason_codes.append("selected_side_invalid")
        expected_action_contract = V2_ACTION_CONTRACT.get(selected_action)
        if expected_action_contract is None:
            runtime_reason_codes.append("selected_action_invalid")
        elif (selected_side, action_family) != expected_action_contract:
            runtime_reason_codes.append("selected_action_side_family_mismatch")
        if action_family not in V2_REQUIRED_VALIDATION_ACTION_FAMILIES:
            runtime_reason_codes.append("action_family_invalid")
        expected_row_identity = regime_conditioned_ev_v2_calibration_row_identity(
            source_run_id=source_run_id,
            market_id=str(row.get("market_id") or ""),
            decision_ts=row.get("decision_ts"),
            selected_action=selected_action,
            source_intent_id=source_intent_id,
            source_fill_id=source_fill_id,
        )
        if row.get("row_identity") != expected_row_identity:
            runtime_reason_codes.append("calibration_row_identity_mismatch")
        runtime_reason_codes.extend(
            _source_lineage_reason_codes(source_lineage, source_root=source_root)
        )
        if isinstance(source_lineage, dict) and isinstance(target_provenance, dict):
            if target_provenance.get("source_artifact_path") != source_lineage.get(
                "settlement_artifact_path"
            ):
                runtime_reason_codes.append(
                    "target_and_lineage_settlement_artifact_path_mismatch"
                )
            if target_provenance.get(
                "source_artifact_sha256"
            ) != source_lineage.get("settlement_artifact_sha256"):
                runtime_reason_codes.append(
                    "target_and_lineage_settlement_artifact_hash_mismatch"
                )

        probability = _finite_float(features.get("selected_side_probability"))
        execution_price = _finite_float(features.get("execution_price"))
        relative_value = _finite_float(
            features.get("selected_side_probability_minus_execution_price")
        )
        for field_name, value in (
            ("selected_side_probability", probability),
            ("execution_price", execution_price),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                runtime_reason_codes.append(f"{field_name}_outside_unit_interval")
        if (
            probability is not None
            and execution_price is not None
            and relative_value is not None
            and not math.isclose(
                relative_value,
                probability - execution_price,
                rel_tol=0.0,
                abs_tol=probability_price_tolerance,
            )
        ):
            runtime_reason_codes.append(
                "selected_side_probability_minus_execution_price_mismatch"
            )
        for feature in (
            "action_score_margin",
            "spread_bps",
            "book_staleness_ms",
            "time_to_close_seconds",
            "cumulative_market_exposure_before_entry",
        ):
            value = _finite_float(features.get(feature))
            if value is not None and value < 0.0:
                runtime_reason_codes.append(f"negative_feature_not_allowed:{feature}")
        entry_index = _finite_float(features.get("entry_index_within_market"))
        if entry_index is not None and entry_index < 1.0:
            runtime_reason_codes.append("entry_index_within_market_below_one")
        queue_fill = _finite_float(features.get("queue_fill_proxy"))
        if queue_fill is not None and not 0.0 <= queue_fill <= 1.0:
            runtime_reason_codes.append("queue_fill_proxy_outside_unit_interval")
        for feature in ("same_side_reentry", "side_flip"):
            value = features.get(feature)
            if isinstance(value, bool) or value not in {0, 1}:
                runtime_reason_codes.append(f"binary_feature_invalid:{feature}")

        schema_valid = not schema_reason_codes
        runtime_reason_codes = sorted(set(runtime_reason_codes))
        runtime_valid = not runtime_reason_codes
        exclusion_reason = _fit_source_exclusion_reason(source_run_id)
        if not schema_valid or not runtime_valid:
            invalid.append(
                {
                    "row_index": index,
                    "market_id": row.get("market_id"),
                    "source_run_id": source_run_id,
                    "schema_valid": schema_valid,
                    "runtime_valid": runtime_valid,
                    "schema_runtime_validation_agreement": (
                        schema_valid == runtime_valid
                    ),
                    "schema_reason_codes": schema_reason_codes,
                    "runtime_reason_codes": runtime_reason_codes,
                    "reason_codes": sorted(
                        set(schema_reason_codes + runtime_reason_codes)
                    ),
                    "forbidden_field_paths": forbidden,
                }
            )
            continue
        if exclusion_reason:
            excluded.append(
                {
                    "row_index": index,
                    "source_run_id": source_run_id,
                    "market_id": row.get("market_id"),
                    "reason_code": exclusion_reason,
                    "schema_valid": True,
                    "runtime_valid": True,
                }
            )
            continue
        normalized.append(
            {
                "row_index": index,
                "source_run_id": source_run_id,
                "source_intent_id": source_intent_id,
                "source_fill_id": source_fill_id,
                "row_identity": str(row["row_identity"]),
                "source_lineage": source_lineage,
                "market_id": str(row["market_id"]),
                "decision_ts": float(decision_ts),
                "max_input_ts": float(max_input_ts),
                "market_close_ts": float(market_close_ts),
                "selected_side": selected_side,
                "selected_action": selected_action,
                "action_family": action_family,
                "decision_time_features": {
                    feature: float(features[feature]) for feature in V2_REQUIRED_FEATURES
                },
                "target_net_return_after_cost": float(target),
                "target_provenance": target_provenance,
                "resolved_outcome": resolved_outcome,
            }
        )
    normalized.sort(key=lambda row: (row["decision_ts"], row["market_id"], row["row_index"]))
    return normalized, invalid, excluded


def _source_lineage_reason_codes(
    payload: Any, *, source_root: Path
) -> list[str]:
    if not isinstance(payload, dict):
        return ["source_lineage_missing"]
    reasons: list[str] = []
    lineage_artifacts = (
        ("source_manifest", "source_manifest_path", "source_manifest_sha256"),
        ("trace_manifest", "trace_manifest_path", "trace_manifest_sha256"),
        ("trace", "trace_artifact_path", "trace_artifact_sha256"),
        ("intent", "intent_artifact_path", "intent_artifact_sha256"),
        ("fill", "fill_artifact_path", "fill_artifact_sha256"),
        (
            "settlement",
            "settlement_artifact_path",
            "settlement_artifact_sha256",
        ),
    )
    for artifact_name, path_field, hash_field in lineage_artifacts:
        path_text = str(payload.get(path_field) or "")
        expected_hash = str(payload.get(hash_field) or "")
        if not path_text:
            reasons.append(f"source_lineage_path_missing:{artifact_name}")
            continue
        path = Path(path_text).expanduser()
        if not path.is_absolute():
            path = source_root / path
        if not path.is_file():
            reasons.append(f"source_lineage_artifact_not_found:{artifact_name}")
        elif not _is_sha256(expected_hash):
            reasons.append(f"source_lineage_artifact_hash_invalid:{artifact_name}")
        elif _sha256_file(path) != expected_hash:
            reasons.append(f"source_lineage_artifact_hash_mismatch:{artifact_name}")
    for field_name in ("trace_row_id", "settlement_row_id"):
        if not str(payload.get(field_name) or ""):
            reasons.append(f"source_lineage_identifier_missing:{field_name}")
    return reasons


@lru_cache(maxsize=1)
def _calibration_row_schema_validator() -> Draft202012Validator:
    schema = json.loads(V2_CALIBRATION_ROW_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _schema_validation_reason_codes(
    validator: Draft202012Validator,
    row: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    errors = sorted(
        validator.iter_errors(row),
        key=lambda error: (list(error.absolute_path), error.validator or ""),
    )
    for error in errors:
        path = ".".join(str(part) for part in error.absolute_path) or "$"
        reasons.append(f"json_schema_validation_failed:{path}:{error.validator}")
    return sorted(set(reasons))


def _fit_source_exclusion_reason(source_run_id: str) -> str | None:
    if source_run_id in V2_REQUIRED_EXCLUDED_RUN_IDS:
        return "required_run_excluded_from_coefficient_fitting"
    lowered = source_run_id.lower()
    if "forward-shadow" in lowered or "forward_shadow" in lowered:
        return "future_unseen_forward_shadow_run_excluded_from_fitting"
    return None


def _chronological_market_split(
    rows: list[dict[str, Any]],
    *,
    validation_fraction: float,
    min_fit_rows: int,
    min_validation_rows: int,
    min_fit_markets: int,
    min_validation_markets: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    by_market: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_market.setdefault(row["market_id"], []).append(row)
    markets = sorted(
        by_market,
        key=lambda market: (
            min(row["decision_ts"] for row in by_market[market]),
            market,
        ),
    )
    candidates: list[tuple[float, list[dict[str, Any]], list[dict[str, Any]]]] = []
    target_validation_rows = len(rows) * validation_fraction
    for boundary in range(1, len(markets)):
        fit = [row for market in markets[:boundary] for row in by_market[market]]
        validation = [row for market in markets[boundary:] for row in by_market[market]]
        if max(row["decision_ts"] for row in fit) >= min(
            row["decision_ts"] for row in validation
        ):
            continue
        if len(fit) < min_fit_rows or len(validation) < min_validation_rows:
            continue
        if boundary < min_fit_markets or len(markets) - boundary < min_validation_markets:
            continue
        candidates.append((abs(len(validation) - target_validation_rows), fit, validation))
    if not candidates:
        return [], [], ["no_valid_chronological_market_disjoint_split"]
    _, fit_rows, validation_rows = min(candidates, key=lambda item: item[0])
    return fit_rows, validation_rows, []


def _build_split_report(
    config: ExecutionLayerV2RegimeConditionedEVCalibrationConfig,
    *,
    raw_rows: list[dict[str, Any]],
    normalized_rows: list[dict[str, Any]],
    invalid_rows: list[dict[str, Any]],
    excluded_rows: list[dict[str, Any]],
    fit_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    split_reasons: list[str],
) -> dict[str, Any]:
    fit_markets = {row["market_id"] for row in fit_rows}
    validation_markets = {row["market_id"] for row in validation_rows}
    chronological = bool(
        fit_rows
        and validation_rows
        and max(row["decision_ts"] for row in fit_rows)
        < min(row["decision_ts"] for row in validation_rows)
    )
    market_disjoint = not bool(fit_markets & validation_markets)
    timestamp_violations = sum(
        row["max_input_ts"] > row["decision_ts"] for row in normalized_rows
    )
    fitted_source_run_ids = sorted({row["source_run_id"] for row in fit_rows})
    prohibited_fit_ids = [
        run_id
        for run_id in fitted_source_run_ids
        if _fit_source_exclusion_reason(run_id)
    ]
    reasons = list(split_reasons)
    if invalid_rows:
        reasons.append("invalid_calibration_rows_present")
    if not chronological:
        reasons.append("chronological_split_not_verified")
    if not market_disjoint:
        reasons.append("market_id_disjointness_not_verified")
    if timestamp_violations:
        reasons.append("feature_timestamp_causality_violation")
    if prohibited_fit_ids:
        reasons.append("prohibited_source_run_present_in_fit_split")
    invalid_reason_distribution: Counter[str] = Counter()
    for row in invalid_rows:
        invalid_reason_distribution.update(row["reason_codes"])
    schema_valid_count = len(normalized_rows) + len(excluded_rows) + sum(
        bool(row["schema_valid"]) for row in invalid_rows
    )
    runtime_valid_count = len(normalized_rows) + len(excluded_rows) + sum(
        bool(row["runtime_valid"]) for row in invalid_rows
    )
    schema_runtime_disagreements = [
        row
        for row in invalid_rows
        if not row["schema_runtime_validation_agreement"]
    ]
    fit_hash = canonical_json_sha256(fit_rows)
    validation_hash = canonical_json_sha256(validation_rows)
    split_identity = {
        "fit_row_ids": [row["row_index"] for row in fit_rows],
        "validation_row_ids": [row["row_index"] for row in validation_rows],
        "fit_dataset_hash": fit_hash,
        "validation_dataset_hash": validation_hash,
    }
    report = {
        "schema_version": REGIME_CONDITIONED_EV_V2_SPLIT_REPORT_SCHEMA_VERSION,
        "run_id": config.run_id,
        "split_order": [
            "historical_fit",
            "validation",
            "future_unseen_shadow_holdout",
        ],
        "raw_row_count": len(raw_rows),
        "eligible_historical_row_count": len(normalized_rows),
        "invalid_row_count": len(invalid_rows),
        "excluded_from_fit_row_count": len(excluded_rows),
        "fit_row_count": len(fit_rows),
        "validation_row_count": len(validation_rows),
        "fit_market_count": len(fit_markets),
        "validation_market_count": len(validation_markets),
        "fit_source_run_ids": fitted_source_run_ids,
        "validation_source_run_ids": sorted(
            {row["source_run_id"] for row in validation_rows}
        ),
        "required_excluded_run_ids": list(V2_REQUIRED_EXCLUDED_RUN_IDS),
        "excluded_rows": excluded_rows,
        "invalid_rows": invalid_rows,
        "invalid_row_reason_distribution": dict(
            sorted(invalid_reason_distribution.items())
        ),
        "calibration_row_schema_path": str(V2_CALIBRATION_ROW_SCHEMA_PATH),
        "calibration_row_schema_sha256": _sha256_file(
            V2_CALIBRATION_ROW_SCHEMA_PATH
        ),
        "target_observation_time_contract": {
            "exact_settlement_timestamp_required": False,
            "resolved_official_outcome_required": True,
            "historical_missing_outcome_observation_timestamp_allowed": True,
            "recorded_outcome_observation_timestamp_must_follow_market_close": True,
        },
        "approved_target_provenance_sources": list(
            V2_APPROVED_TARGET_PROVENANCE_SOURCES
        ),
        "schema_validation_row_count": len(raw_rows),
        "schema_valid_row_count": schema_valid_count,
        "schema_invalid_row_count": len(raw_rows) - schema_valid_count,
        "runtime_valid_row_count": runtime_valid_count,
        "runtime_invalid_row_count": len(raw_rows) - runtime_valid_count,
        "schema_runtime_validation_agreement_count": (
            len(raw_rows) - len(schema_runtime_disagreements)
        ),
        "schema_runtime_validation_disagreement_count": len(
            schema_runtime_disagreements
        ),
        "schema_runtime_validation_agreement_passed": not bool(
            schema_runtime_disagreements
        ),
        "schema_runtime_validation_disagreement_rows": (
            schema_runtime_disagreements
        ),
        "chronological_split_passed": chronological,
        "market_id_disjointness_passed": market_disjoint,
        "market_id_overlap": sorted(fit_markets & validation_markets),
        "fit_max_decision_ts": max(
            (row["decision_ts"] for row in fit_rows), default=None
        ),
        "validation_min_decision_ts": min(
            (row["decision_ts"] for row in validation_rows), default=None
        ),
        "feature_max_input_ts_violation_count": timestamp_violations,
        "uses_validation_labels_for_fitting": False,
        "uses_validation_labels_for_threshold_selection": False,
        "uses_holdout_labels_for_fitting": False,
        "future_unseen_shadow_outcome_free": True,
        "fit_dataset_hash": fit_hash,
        "validation_dataset_hash": validation_hash,
        "split_hash": canonical_json_sha256(split_identity),
        "leakage_checks_passed": not reasons,
        "blocking_reason_codes": sorted(set(reasons)),
        "diagnostic_only": True,
        **_safety_report_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def _fit_and_evaluate(
    fit_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    *,
    ridge_alpha: float,
    max_abs_coefficient: float,
    min_relative_mae_improvement: float,
    min_relative_mse_improvement: float,
    bootstrap_samples: int,
    bootstrap_confidence_level: float,
    min_bootstrap_improvement_lower_bound: float,
    max_lomo_coefficient_absolute_deviation: float,
    min_lomo_coefficient_sign_agreement: float,
    min_validation_rows_per_side: int,
    min_validation_rows_per_action_family: int,
    min_validation_rows_per_resolved_outcome: int,
    min_validation_markets_per_category: int,
    statistical_random_seed: int,
) -> dict[str, Any]:
    transforms = _fit_feature_transforms(fit_rows)
    fit_x = [_group_scores(row, transforms) for row in fit_rows]
    validation_x = [_group_scores(row, transforms) for row in validation_rows]
    fit_y = [row["target_net_return_after_cost"] for row in fit_rows]
    validation_y = [row["target_net_return_after_cost"] for row in validation_rows]
    coefficients = _ridge_fit(fit_x, fit_y, ridge_alpha)
    fit_predictions = _predict_matrix(fit_x, coefficients)
    validation_predictions = _predict_matrix(validation_x, coefficients)
    constant_predictions = [sum(fit_y) / len(fit_y)] * len(validation_y)
    legacy_x = [
        [row["decision_time_features"]["canonical_o_action_score"]]
        for row in fit_rows
    ]
    legacy_validation_x = [
        [row["decision_time_features"]["canonical_o_action_score"]]
        for row in validation_rows
    ]
    legacy_coefficients = _ridge_fit(legacy_x, fit_y, ridge_alpha)
    legacy_predictions = _predict_matrix(legacy_validation_x, legacy_coefficients)
    metrics = {
        "historical_fit": _regression_metrics(fit_y, fit_predictions),
        "validation_candidate": _regression_metrics(
            validation_y, validation_predictions
        ),
        "validation_constant_baseline": _regression_metrics(
            validation_y, constant_predictions
        ),
        "validation_legacy_o_score_baseline": _regression_metrics(
            validation_y, legacy_predictions
        ),
    }
    market_metrics = {
        "validation_candidate": _market_level_metrics(
            validation_rows, validation_y, validation_predictions
        ),
        "validation_constant_baseline": _market_level_metrics(
            validation_rows, validation_y, constant_predictions
        ),
        "validation_legacy_o_score_baseline": _market_level_metrics(
            validation_rows, validation_y, legacy_predictions
        ),
    }
    relative_improvements = _relative_baseline_improvements(
        metrics,
        market_metrics,
        min_relative_mae_improvement=min_relative_mae_improvement,
        min_relative_mse_improvement=min_relative_mse_improvement,
    )
    bootstrap = _market_bootstrap_improvement_intervals(
        validation_rows,
        validation_y,
        candidate_predictions=validation_predictions,
        baseline_predictions={
            "constant_baseline": constant_predictions,
            "legacy_o_score_baseline": legacy_predictions,
        },
        samples=bootstrap_samples,
        confidence_level=bootstrap_confidence_level,
        minimum_lower_bound=min_bootstrap_improvement_lower_bound,
        random_seed=statistical_random_seed,
    )
    coefficient_stability = _leave_one_market_out_coefficient_stability(
        fit_rows,
        fit_x,
        fit_y,
        full_coefficients=coefficients,
        ridge_alpha=ridge_alpha,
        max_absolute_deviation_allowed=(
            max_lomo_coefficient_absolute_deviation
        ),
        min_sign_agreement_required=min_lomo_coefficient_sign_agreement,
    )
    validation_coverage = _validation_coverage_gate(
        validation_rows,
        min_rows_per_side=min_validation_rows_per_side,
        min_rows_per_action_family=min_validation_rows_per_action_family,
        min_rows_per_resolved_outcome=min_validation_rows_per_resolved_outcome,
        min_markets_per_category=min_validation_markets_per_category,
    )
    finite_bounded = all(
        math.isfinite(value) and abs(value) <= max_abs_coefficient
        for value in coefficients
    )
    reasons: list[str] = []
    if not relative_improvements["row_level_gate_passed"]:
        reasons.append("row_level_relative_improvement_gate_failed")
    if not relative_improvements["market_level_gate_passed"]:
        reasons.append("market_level_relative_improvement_gate_failed")
    if not bootstrap["confidence_gate_passed"]:
        reasons.append("market_bootstrap_confidence_gate_failed")
    if not coefficient_stability["stability_gate_passed"]:
        reasons.append("coefficient_stability_gate_failed")
    reasons.extend(validation_coverage["blocking_reason_codes"])
    if not finite_bounded:
        reasons.append("coefficients_not_finite_and_bounded")
    statistical_eligibility_config = {
        "min_relative_mae_improvement": min_relative_mae_improvement,
        "min_relative_mse_improvement": min_relative_mse_improvement,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_confidence_level": bootstrap_confidence_level,
        "min_bootstrap_improvement_lower_bound": (
            min_bootstrap_improvement_lower_bound
        ),
        "max_lomo_coefficient_absolute_deviation": (
            max_lomo_coefficient_absolute_deviation
        ),
        "min_lomo_coefficient_sign_agreement": (
            min_lomo_coefficient_sign_agreement
        ),
        "min_validation_rows_per_side": min_validation_rows_per_side,
        "min_validation_rows_per_action_family": (
            min_validation_rows_per_action_family
        ),
        "min_validation_rows_per_resolved_outcome": (
            min_validation_rows_per_resolved_outcome
        ),
        "min_validation_markets_per_category": (
            min_validation_markets_per_category
        ),
        "statistical_random_seed": statistical_random_seed,
    }
    statistical_summary = {
        "relative_improvements": relative_improvements,
        "market_bootstrap_confidence_intervals": bootstrap,
        "coefficient_stability": coefficient_stability,
        "validation_coverage": validation_coverage,
    }
    return {
        "feature_transforms": transforms,
        "group_weights": _GROUP_WEIGHTS,
        "coefficients": coefficients,
        "legacy_coefficients": legacy_coefficients,
        "metrics": metrics,
        "market_level_metrics": market_metrics,
        "relative_baseline_improvements": relative_improvements,
        "market_bootstrap_confidence_intervals": bootstrap,
        "coefficient_stability_metrics": coefficient_stability,
        "validation_coverage": validation_coverage,
        "statistical_eligibility_config": statistical_eligibility_config,
        "statistical_eligibility_config_hash": canonical_json_sha256(
            statistical_eligibility_config
        ),
        "statistical_eligibility_summary_hash": canonical_json_sha256(
            statistical_summary
        ),
        "validation_improved_over_constant_and_legacy": bool(
            relative_improvements["row_level_gate_passed"]
            and relative_improvements["market_level_gate_passed"]
        ),
        "statistical_eligibility_passed": not reasons,
        "coefficients_finite_and_bounded": finite_bounded,
        "fit_coefficients_hash": canonical_json_sha256(coefficients),
        "blocking_reason_codes": reasons,
    }


def _market_level_metrics(
    rows: list[dict[str, Any]],
    actual: list[float],
    predicted: list[float],
) -> dict[str, Any]:
    by_market: dict[str, list[float]] = {}
    for row, target, prediction in zip(rows, actual, predicted, strict=True):
        by_market.setdefault(row["market_id"], []).append(prediction - target)
    market_rows = {
        market_id: {
            "row_count": len(errors),
            "mae": sum(abs(error) for error in errors) / len(errors),
            "mse": sum(error * error for error in errors) / len(errors),
        }
        for market_id, errors in sorted(by_market.items())
    }
    return {
        "market_count": len(market_rows),
        "mae": sum(row["mae"] for row in market_rows.values())
        / len(market_rows),
        "mse": sum(row["mse"] for row in market_rows.values())
        / len(market_rows),
        "by_market": market_rows,
    }


def _relative_baseline_improvements(
    row_metrics: dict[str, dict[str, float]],
    market_metrics: dict[str, dict[str, Any]],
    *,
    min_relative_mae_improvement: float,
    min_relative_mse_improvement: float,
) -> dict[str, Any]:
    thresholds = {
        "mae": min_relative_mae_improvement,
        "mse": min_relative_mse_improvement,
    }
    levels = {
        "row_level": row_metrics,
        "market_level": market_metrics,
    }
    result: dict[str, Any] = {"thresholds": thresholds}
    for level_name, values in levels.items():
        candidate = values["validation_candidate"]
        comparisons: dict[str, Any] = {}
        for baseline_name in (
            "validation_constant_baseline",
            "validation_legacy_o_score_baseline",
        ):
            baseline = values[baseline_name]
            comparison = {
                metric: _relative_error_improvement(
                    float(candidate[metric]), float(baseline[metric])
                )
                for metric in ("mae", "mse")
            }
            comparison["passed"] = all(
                comparison[metric] >= thresholds[metric]
                for metric in ("mae", "mse")
            )
            comparisons[baseline_name] = comparison
        result[level_name] = comparisons
        result[f"{level_name}_gate_passed"] = all(
            comparison["passed"] for comparison in comparisons.values()
        )
    return result


def _relative_error_improvement(candidate: float, baseline: float) -> float:
    if baseline <= 1e-15:
        return 1.0 if candidate < baseline else 0.0
    return (baseline - candidate) / baseline


def _market_bootstrap_improvement_intervals(
    rows: list[dict[str, Any]],
    actual: list[float],
    *,
    candidate_predictions: list[float],
    baseline_predictions: dict[str, list[float]],
    samples: int,
    confidence_level: float,
    minimum_lower_bound: float,
    random_seed: int,
) -> dict[str, Any]:
    by_market: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_market.setdefault(row["market_id"], []).append(index)
    market_ids = sorted(by_market)
    rng = random.Random(random_seed)
    alpha = (1.0 - confidence_level) / 2.0
    comparisons: dict[str, Any] = {}
    for baseline_name, baseline_values in baseline_predictions.items():
        metric_results: dict[str, Any] = {}
        per_market = {
            market_id: {
                metric: _market_error_improvement(
                    by_market[market_id],
                    actual,
                    candidate_predictions,
                    baseline_values,
                    metric=metric,
                )
                for metric in ("mae", "mse")
            }
            for market_id in market_ids
        }
        for metric in ("mae", "mse"):
            values = [per_market[market_id][metric] for market_id in market_ids]
            bootstrap_means = []
            for _ in range(samples):
                sample = [rng.choice(values) for _ in values]
                bootstrap_means.append(sum(sample) / len(sample))
            bootstrap_means.sort()
            lower = _quantile(bootstrap_means, alpha)
            upper = _quantile(bootstrap_means, 1.0 - alpha)
            metric_results[metric] = {
                "baseline_minus_candidate_point_estimate": sum(values)
                / len(values),
                "confidence_interval_lower": lower,
                "confidence_interval_upper": upper,
                "minimum_lower_bound_required": minimum_lower_bound,
                "passed": lower > minimum_lower_bound,
            }
        metric_results["passed"] = all(
            metric_results[metric]["passed"] for metric in ("mae", "mse")
        )
        comparisons[baseline_name] = metric_results
    return {
        "resampling_unit": "market_id",
        "market_count": len(market_ids),
        "bootstrap_samples": samples,
        "confidence_level": confidence_level,
        "random_seed": random_seed,
        "comparisons": comparisons,
        "confidence_gate_passed": all(
            comparison["passed"] for comparison in comparisons.values()
        ),
    }


def _market_error_improvement(
    indices: list[int],
    actual: list[float],
    candidate: list[float],
    baseline: list[float],
    *,
    metric: str,
) -> float:
    exponent = 1 if metric == "mae" else 2
    candidate_error = sum(
        abs(candidate[index] - actual[index]) ** exponent for index in indices
    ) / len(indices)
    baseline_error = sum(
        abs(baseline[index] - actual[index]) ** exponent for index in indices
    ) / len(indices)
    return baseline_error - candidate_error


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return float("nan")
    position = probability * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _leave_one_market_out_coefficient_stability(
    fit_rows: list[dict[str, Any]],
    fit_x: list[list[float]],
    fit_y: list[float],
    *,
    full_coefficients: list[float],
    ridge_alpha: float,
    max_absolute_deviation_allowed: float,
    min_sign_agreement_required: float,
) -> dict[str, Any]:
    markets = sorted({row["market_id"] for row in fit_rows})
    replicate_coefficients: list[list[float]] = []
    by_omitted_market: dict[str, Any] = {}
    for omitted_market in markets:
        kept = [
            index
            for index, row in enumerate(fit_rows)
            if row["market_id"] != omitted_market
        ]
        coefficients = _ridge_fit(
            [fit_x[index] for index in kept],
            [fit_y[index] for index in kept],
            ridge_alpha,
        )
        replicate_coefficients.append(coefficients)
        deviations = [
            abs(value - full)
            for value, full in zip(
                coefficients, full_coefficients, strict=True
            )
        ]
        by_omitted_market[omitted_market] = {
            "coefficient_hash": canonical_json_sha256(coefficients),
            "max_absolute_deviation": max(deviations),
        }
    coefficient_stddev = []
    for index in range(len(full_coefficients)):
        values = [coefficients[index] for coefficients in replicate_coefficients]
        mean = sum(values) / len(values)
        coefficient_stddev.append(
            math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
        )
    max_deviation = max(
        row["max_absolute_deviation"] for row in by_omitted_market.values()
    )
    sign_checks = []
    for coefficients in replicate_coefficients:
        for value, full in zip(coefficients[1:], full_coefficients[1:], strict=True):
            if abs(full) <= 1e-12:
                sign_checks.append(abs(value) <= max_absolute_deviation_allowed)
            else:
                sign_checks.append((value > 0.0) == (full > 0.0))
    sign_agreement = sum(sign_checks) / len(sign_checks) if sign_checks else 0.0
    passed = bool(
        max_deviation <= max_absolute_deviation_allowed
        and sign_agreement >= min_sign_agreement_required
    )
    return {
        "method": "leave_one_market_out_fixed_fit_transforms",
        "fit_market_count": len(markets),
        "replicate_count": len(replicate_coefficients),
        "coefficient_stddev": coefficient_stddev,
        "max_coefficient_stddev": max(coefficient_stddev),
        "max_absolute_deviation": max_deviation,
        "max_absolute_deviation_allowed": max_absolute_deviation_allowed,
        "coefficient_sign_agreement_rate": sign_agreement,
        "min_sign_agreement_required": min_sign_agreement_required,
        "by_omitted_market": by_omitted_market,
        "stability_gate_passed": passed,
    }


def _validation_coverage_gate(
    validation_rows: list[dict[str, Any]],
    *,
    min_rows_per_side: int,
    min_rows_per_action_family: int,
    min_rows_per_resolved_outcome: int,
    min_markets_per_category: int,
) -> dict[str, Any]:
    side_counts = Counter(row["selected_side"] for row in validation_rows)
    family_counts = Counter(row["action_family"] for row in validation_rows)
    outcome_counts = Counter(row["resolved_outcome"] for row in validation_rows)
    side_market_counts = _unique_market_counts(validation_rows, "selected_side")
    family_market_counts = _unique_market_counts(validation_rows, "action_family")
    outcome_market_counts = _unique_market_counts(validation_rows, "resolved_outcome")
    side_passed = all(
        side_counts[value] >= min_rows_per_side
        for value in V2_REQUIRED_VALIDATION_SIDES
    )
    family_passed = all(
        family_counts[value] >= min_rows_per_action_family
        for value in V2_REQUIRED_VALIDATION_ACTION_FAMILIES
    )
    outcome_passed = all(
        outcome_counts[value] >= min_rows_per_resolved_outcome
        for value in V2_REQUIRED_VALIDATION_RESOLVED_OUTCOMES
    )
    side_market_passed = all(
        side_market_counts[value] >= min_markets_per_category
        for value in V2_REQUIRED_VALIDATION_SIDES
    )
    family_market_passed = all(
        family_market_counts[value] >= min_markets_per_category
        for value in V2_REQUIRED_VALIDATION_ACTION_FAMILIES
    )
    outcome_market_passed = all(
        outcome_market_counts[value] >= min_markets_per_category
        for value in V2_REQUIRED_VALIDATION_RESOLVED_OUTCOMES
    )
    reasons = []
    if not side_passed:
        reasons.append("validation_side_coverage_gate_failed")
    if not family_passed:
        reasons.append("validation_action_family_coverage_gate_failed")
    if not outcome_passed:
        reasons.append("validation_resolved_outcome_coverage_gate_failed")
    if not side_market_passed:
        reasons.append("validation_side_market_coverage_gate_failed")
    if not family_market_passed:
        reasons.append("validation_action_family_market_coverage_gate_failed")
    if not outcome_market_passed:
        reasons.append("validation_resolved_outcome_market_coverage_gate_failed")
    return {
        "side_counts": dict(sorted(side_counts.items())),
        "action_family_counts": dict(sorted(family_counts.items())),
        "resolved_outcome_counts": dict(sorted(outcome_counts.items())),
        "side_unique_market_counts": side_market_counts,
        "action_family_unique_market_counts": family_market_counts,
        "resolved_outcome_unique_market_counts": outcome_market_counts,
        "minimum_rows_per_side": min_rows_per_side,
        "minimum_rows_per_action_family": min_rows_per_action_family,
        "minimum_rows_per_resolved_outcome": min_rows_per_resolved_outcome,
        "minimum_markets_per_category": min_markets_per_category,
        "side_coverage_passed": side_passed and side_market_passed,
        "action_family_coverage_passed": family_passed and family_market_passed,
        "resolved_outcome_coverage_passed": outcome_passed and outcome_market_passed,
        "coverage_gate_passed": bool(
            side_passed
            and family_passed
            and outcome_passed
            and side_market_passed
            and family_market_passed
            and outcome_market_passed
        ),
        "blocking_reason_codes": reasons,
    }


def _unique_market_counts(
    rows: list[dict[str, Any]], field_name: str
) -> dict[str, int]:
    markets: dict[str, set[str]] = {}
    for row in rows:
        markets.setdefault(str(row[field_name]), set()).add(row["market_id"])
    return {
        value: len(market_ids) for value, market_ids in sorted(markets.items())
    }


def _fit_feature_transforms(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    transforms: dict[str, dict[str, float]] = {}
    for feature in V2_REQUIRED_FEATURES:
        values = [_feature_value_for_model(row, feature) for row in rows]
        center = sum(values) / len(values)
        variance = sum((value - center) ** 2 for value in values) / len(values)
        scale = max(math.sqrt(variance), 1e-9)
        transforms[feature] = {
            "center": center,
            "scale": scale,
            "clip_min": -3.0,
            "clip_max": 3.0,
        }
    return transforms


def _feature_value_for_model(row: dict[str, Any], feature: str) -> float:
    value = float(row["decision_time_features"][feature])
    if feature in REGIME_CONDITIONED_EV_V2_FEATURE_GROUPS["btc_anchor_direction"]:
        return -value if row["selected_side"] == "DOWN" else value
    return value


def _group_scores(
    row: dict[str, Any], transforms: dict[str, dict[str, float]]
) -> list[float]:
    scores: list[float] = []
    for group_name, features in REGIME_CONDITIONED_EV_V2_FEATURE_GROUPS.items():
        score = 0.0
        for feature in features:
            transform = transforms[feature]
            normalized = (
                _feature_value_for_model(row, feature) - transform["center"]
            ) / transform["scale"]
            normalized = max(
                transform["clip_min"], min(transform["clip_max"], normalized)
            )
            score += _GROUP_WEIGHTS[group_name][feature] * normalized
        scores.append(max(-1.0, min(1.0, score)))
    return scores


def _ridge_fit(x: list[list[float]], y: list[float], alpha: float) -> list[float]:
    design = [[1.0, *row] for row in x]
    width = len(design[0])
    matrix = [[0.0] * width for _ in range(width)]
    vector = [0.0] * width
    for row, target in zip(design, y, strict=True):
        for i in range(width):
            vector[i] += row[i] * target
            for j in range(width):
                matrix[i][j] += row[i] * row[j]
    for i in range(1, width):
        matrix[i][i] += alpha
    return _solve_linear_system(matrix, vector)


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [matrix[index][:] + [vector[index]] for index in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("ridge system is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                current - factor * pivot_value
                for current, pivot_value in zip(
                    augmented[row], augmented[column], strict=True
                )
            ]
    return [augmented[index][-1] for index in range(size)]


def _predict_matrix(x: list[list[float]], coefficients: list[float]) -> list[float]:
    return [
        coefficients[0]
        + sum(
            coefficient * value
            for coefficient, value in zip(coefficients[1:], row, strict=True)
        )
        for row in x
    ]


def _regression_metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
    errors = [prediction - target for target, prediction in zip(actual, predicted, strict=True)]
    return {
        "row_count": len(actual),
        "mse": sum(error * error for error in errors) / len(errors),
        "mae": sum(abs(error) for error in errors) / len(errors),
        "mean_prediction": sum(predicted) / len(predicted),
        "mean_target": sum(actual) / len(actual),
    }


def _build_frozen_artifact(
    config: ExecutionLayerV2RegimeConditionedEVCalibrationConfig,
    *,
    fit_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    fit_result: dict[str, Any],
) -> dict[str, Any]:
    coefficient_values = fit_result["coefficients"]
    coefficient_groups: dict[str, Any] = {}
    for index, (group_name, features) in enumerate(
        REGIME_CONDITIONED_EV_V2_FEATURE_GROUPS.items(), start=1
    ):
        coefficient_groups[group_name] = {
            "group_coefficient": coefficient_values[index],
            "maximum_absolute_contribution": config.max_abs_coefficient,
            "feature_weights": _GROUP_WEIGHTS[group_name],
            "feature_transforms": {
                feature: fit_result["feature_transforms"][feature]
                for feature in features
            },
        }
    split_identity = {
        "fit": fit_rows,
        "validation": validation_rows,
    }
    calibration_config = {
        "validation_fraction": config.validation_fraction,
        "ridge_alpha": config.ridge_alpha,
        "entry_ev_threshold": config.entry_ev_threshold,
        "min_fit_rows": config.min_fit_rows,
        "min_validation_rows": config.min_validation_rows,
        "min_fit_markets": config.min_fit_markets,
        "min_validation_markets": config.min_validation_markets,
        "max_abs_coefficient": config.max_abs_coefficient,
        "probability_price_tolerance": config.probability_price_tolerance,
        "statistical_eligibility_config": fit_result[
            "statistical_eligibility_config"
        ],
        "calibration_row_schema_sha256": _sha256_file(
            V2_CALIBRATION_ROW_SCHEMA_PATH
        ),
        "group_weights": _GROUP_WEIGHTS,
    }
    feature_groups = {
        group_name: {"features": list(features)}
        for group_name, features in REGIME_CONDITIONED_EV_V2_FEATURE_GROUPS.items()
    }
    return {
        "schema_version": FROZEN_REGIME_CONDITIONED_EV_V2_SCHEMA_VERSION,
        "artifact_name": FROZEN_REGIME_CONDITIONED_EV_V2_ARTIFACT_NAME,
        "diagnostic_only": True,
        "frozen": True,
        "decision_time_safe": True,
        "uses_validation_labels_for_tuning": False,
        "market_implied_probability_used_as_direct_fair_value_ev": False,
        "market_implied_probability_used_as_conditioning_feature": True,
        "market_implied_probability_used_as_regime_direction_vote": False,
        "no_outcome_field_usage": True,
        "no_oracle_field_usage": True,
        "no_future_return_field_usage": True,
        "source_score_mutation_enabled": False,
        "o_score_mutation_enabled": False,
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
        "fit_provenance": {
            "coefficients_source": "historical_fit_split_only",
            "settled_outcomes_or_pnl_used_as_training_targets": True,
            "settled_outcomes_or_pnl_used_as_decision_time_inputs": False,
            "uses_validation_labels_for_fitting": False,
            "uses_validation_labels_for_threshold_selection": False,
            "uses_holdout_labels_for_fitting": False,
            "uses_holdout_labels_for_threshold_selection": False,
            "uses_oracle_actions_for_fitting": False,
            "uses_future_returns_for_fitting": False,
            "future_unseen_run_pattern_excluded": True,
            "fitted_from_run_ids": sorted(
                {row["source_run_id"] for row in fit_rows}
            ),
            "excluded_run_ids": list(V2_REQUIRED_EXCLUDED_RUN_IDS),
            "fit_dataset_hash": canonical_json_sha256(fit_rows),
            "validation_dataset_hash": canonical_json_sha256(validation_rows),
            "split_hash": canonical_json_sha256(split_identity),
            "calibration_config_hash": canonical_json_sha256(calibration_config),
            "fit_coefficients_hash": fit_result["fit_coefficients_hash"],
            "calibration_row_schema_sha256": _sha256_file(
                V2_CALIBRATION_ROW_SCHEMA_PATH
            ),
            "statistical_eligibility_config_hash": fit_result[
                "statistical_eligibility_config_hash"
            ],
            "statistical_eligibility_summary_hash": fit_result[
                "statistical_eligibility_summary_hash"
            ],
            "statistical_eligibility_passed": True,
        },
        "calibration_protocol": {
            "split_order": [
                "historical_fit",
                "validation",
                "future_unseen_shadow_holdout",
            ],
            "chronological_split_required": True,
            "market_id_disjointness_required_where_possible": True,
            "feature_max_input_ts_must_not_exceed_decision_ts": True,
            "historical_fit_target": "settled_net_return_after_cost",
            "validation_labels_used_for_evaluation_only": True,
            "future_shadow_outcome_free_at_inference": True,
            "threshold_selection_source": "fixed_pre_validation_config",
            "refit_from_future_shadow_result_allowed": False,
            "statistical_eligibility_required": True,
            "market_level_evaluation_required": True,
            "market_bootstrap_confidence_required": True,
            "coefficient_stability_required": True,
            "validation_coverage_required": True,
        },
        "independence_constraints": {
            "selected_side_probability_single_group": "market_price_value",
            "p_up_p_down_used_only_to_derive_selected_side_probability": True,
            "btc_anchor_fields_single_group": "btc_anchor_direction",
            "btc_anchor_maximum_signal_vote_weight": 1.0,
            "correlated_momentum_reference_counted_as_independent_votes": False,
        },
        "feature_groups": feature_groups,
        "coefficients": {
            "intercept": coefficient_values[0],
            "groups": coefficient_groups,
            "side_offsets": {},
            "family_offsets": {},
            "subtract_execution_cost": False,
        },
    }


def _run_future_shadow_if_ready(
    config: ExecutionLayerV2RegimeConditionedEVCalibrationConfig,
    *,
    run_dir: Path,
    artifact_path: Path | None,
    fit_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if config.future_shadow_input_path is None:
        return {
            "status": "not_run_future_shadow_input_not_provided",
            "outcome_free": True,
            "refit_performed": False,
        }
    if artifact_path is None:
        return {
            "status": "blocked_no_valid_frozen_artifact",
            "outcome_free": True,
            "refit_performed": False,
        }
    future_rows = _load_forward_shadow_rows(config.future_shadow_input_path)
    prior_markets = {
        row["market_id"] for row in [*fit_rows, *validation_rows]
    }
    future_markets = {
        str(_lookup_value(row, "market_id") or "") for row in future_rows
    }
    overlap = sorted((prior_markets & future_markets) - {""})
    future_decision_ts = [
        value
        for row in future_rows
        if (value := _finite_float(_lookup_value(row, "decision_ts"))) is not None
    ]
    validation_max_ts = max(
        (row["decision_ts"] for row in validation_rows), default=None
    )
    future_time_valid = bool(
        future_decision_ts
        and validation_max_ts is not None
        and min(future_decision_ts) > validation_max_ts
    )
    if overlap or not future_time_valid:
        reasons: list[str] = []
        if overlap:
            reasons.append("future_shadow_market_overlap_with_fit_or_validation")
        if not future_time_valid:
            reasons.append("future_shadow_not_strictly_later_than_validation")
        return {
            "status": "blocked_fail_closed_future_shadow_provenance",
            "blocking_reason_codes": reasons,
            "market_id_overlap": overlap,
            "future_shadow_min_decision_ts": min(future_decision_ts, default=None),
            "validation_max_decision_ts": validation_max_ts,
            "outcome_free": True,
            "refit_performed": False,
        }
    result = run_execution_layer_v2_regime_conditioned_ev_forward_shadow(
        ExecutionLayerV2RegimeConditionedEVForwardShadowConfig(
            run_id=f"{config.run_id}-future-unseen-shadow",
            input_path=config.future_shadow_input_path,
            output_dir=run_dir / "future_shadow",
            frozen_regime_conditioned_ev_artifact=artifact_path,
            entry_ev_threshold=config.entry_ev_threshold,
        )
    )
    report = result.forward_shadow_report
    return {
        "status": report["forward_shadow_status"],
        "run_id": report["run_id"],
        "report_path": str(
            result.artifact_paths[
                "execution_layer_v2_regime_conditioned_ev_forward_shadow_report"
            ]
        ),
        "report_sha256": result.artifact_hashes[
            "execution_layer_v2_regime_conditioned_ev_forward_shadow_report"
        ],
        "regime_conditioned_ev_produced_count": report[
            "regime_conditioned_ev_produced_count"
        ],
        "regime_conditioned_ev_missing_count": report[
            "regime_conditioned_ev_missing_count"
        ],
        "candidate_count": report["candidate_count"],
        "full_guard_passed_count": report["full_guard_passed_count"],
        "executable_shadow_count": report["executable_shadow_count"],
        "outcome_free": True,
        "market_id_disjoint_from_fit_and_validation": True,
        "strictly_later_than_validation": True,
        "outcomes_reconciled": False,
        "refit_performed": False,
    }


def _build_calibration_report(
    config: ExecutionLayerV2RegimeConditionedEVCalibrationConfig,
    *,
    split_report: dict[str, Any],
    fit_result: dict[str, Any] | None,
    artifact_path: Path | None,
    eligibility_reasons: list[str],
    future_shadow_summary: dict[str, Any],
) -> dict[str, Any]:
    report = {
        "schema_version": REGIME_CONDITIONED_EV_V2_CALIBRATION_REPORT_SCHEMA_VERSION,
        "run_id": config.run_id,
        "protocol_status": (
            "frozen_diagnostic_artifact_created"
            if artifact_path is not None
            else "blocked_fail_closed_no_artifact"
        ),
        "artifact_created": artifact_path is not None,
        "artifact_path": str(artifact_path) if artifact_path is not None else None,
        "artifact_sha256": _sha256_file(artifact_path) if artifact_path else None,
        "fit_metrics": fit_result["metrics"] if fit_result else {},
        "market_level_metrics": (
            fit_result["market_level_metrics"] if fit_result else {}
        ),
        "relative_baseline_improvements": (
            fit_result["relative_baseline_improvements"] if fit_result else {}
        ),
        "market_bootstrap_confidence_intervals": (
            fit_result["market_bootstrap_confidence_intervals"]
            if fit_result
            else {}
        ),
        "coefficient_stability_metrics": (
            fit_result["coefficient_stability_metrics"] if fit_result else {}
        ),
        "validation_coverage": (
            fit_result["validation_coverage"] if fit_result else {}
        ),
        "statistical_eligibility_passed": bool(
            fit_result and fit_result["statistical_eligibility_passed"]
        ),
        "statistical_eligibility_config_hash": (
            fit_result["statistical_eligibility_config_hash"]
            if fit_result
            else None
        ),
        "statistical_eligibility_config": (
            fit_result["statistical_eligibility_config"] if fit_result else {}
        ),
        "statistical_eligibility_summary_hash": (
            fit_result["statistical_eligibility_summary_hash"]
            if fit_result
            else None
        ),
        "fit_coefficients_hash": (
            fit_result["fit_coefficients_hash"] if fit_result else None
        ),
        "validation_improved_over_constant_and_legacy": bool(
            fit_result
            and fit_result["validation_improved_over_constant_and_legacy"]
        ),
        "coefficients_finite_and_bounded": bool(
            fit_result and fit_result["coefficients_finite_and_bounded"]
        ),
        "entry_ev_threshold": config.entry_ev_threshold,
        "threshold_selection_source": "fixed_pre_validation_config",
        "uses_validation_labels_for_fitting": False,
        "uses_validation_labels_for_threshold_selection": False,
        "uses_holdout_labels_for_fitting": False,
        "settled_outcomes_or_pnl_used_as_historical_training_targets": True,
        "settled_outcomes_or_pnl_used_as_decision_time_inputs": False,
        "leakage_checks_passed": split_report["leakage_checks_passed"],
        "schema_runtime_validation_agreement_passed": split_report[
            "schema_runtime_validation_agreement_passed"
        ],
        "invalid_row_reason_distribution": split_report[
            "invalid_row_reason_distribution"
        ],
        "blocking_reason_codes": eligibility_reasons,
        "final_artifact_eligibility_reason_codes": eligibility_reasons,
        "future_shadow": future_shadow_summary,
        "future_shadow_outcomes_may_be_reconciled_only_after_window_close": True,
        "refit_from_future_shadow_result_allowed": False,
        "diagnostic_only": True,
        "production_gate_implemented": False,
        "source_scores_mutated": False,
        "o_score_mutated": False,
        **_safety_report_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def _split_report_markdown(report: dict[str, Any]) -> str:
    reasons = report["blocking_reason_codes"] or ["none"]
    return "\n".join(
        [
            "# Regime-Conditioned EV v2 Split Report",
            "",
            f"- fit rows: `{report['fit_row_count']}`",
            f"- validation rows: `{report['validation_row_count']}`",
            f"- fit markets: `{report['fit_market_count']}`",
            f"- validation markets: `{report['validation_market_count']}`",
            f"- chronological: `{report['chronological_split_passed']}`",
            f"- market disjoint: `{report['market_id_disjointness_passed']}`",
            f"- leakage checks passed: `{report['leakage_checks_passed']}`",
            "- schema/runtime validation agreement: "
            f"`{report['schema_runtime_validation_agreement_passed']}`",
            f"- invalid rows: `{report['invalid_row_count']}`",
            f"- invalid reason distribution: `{report['invalid_row_reason_distribution']}`",
            "",
            "## Blocking Reasons",
            "",
            *[f"- `{reason}`" for reason in reasons],
            "",
        ]
    )


def _calibration_report_markdown(report: dict[str, Any]) -> str:
    reasons = report["blocking_reason_codes"] or ["none"]
    return "\n".join(
        [
            "# Regime-Conditioned EV v2 Calibration Report",
            "",
            f"- status: `{report['protocol_status']}`",
            f"- artifact created: `{report['artifact_created']}`",
            "- statistical eligibility passed: "
            f"`{report['statistical_eligibility_passed']}`",
            "- schema/runtime validation agreement: "
            f"`{report['schema_runtime_validation_agreement_passed']}`",
            "- row-level metrics: "
            f"`{report['fit_metrics']}`",
            "- market-level metrics: "
            f"`{report['market_level_metrics']}`",
            "- relative baseline improvements: "
            f"`{report['relative_baseline_improvements']}`",
            "- market bootstrap confidence: "
            f"`{report['market_bootstrap_confidence_intervals']}`",
            "- coefficient stability: "
            f"`{report['coefficient_stability_metrics']}`",
            f"- validation coverage: `{report['validation_coverage']}`",
            "- validation labels used for fitting: `false`",
            "- validation labels used for threshold selection: `false`",
            "- future shadow refit allowed: `false`",
            f"- future shadow status: `{report['future_shadow']['status']}`",
            "- paper/live/handoff unlock: `false`",
            "",
            "## Blocking Reasons",
            "",
            *[f"- `{reason}`" for reason in reasons],
            "",
        ]
    )


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)
