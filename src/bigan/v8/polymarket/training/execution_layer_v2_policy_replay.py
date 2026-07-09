"""Settlement-CSV policy replay diagnostics for v8 Execution Layer v2.

The replay in this module is outcome-aware by construction because it reads a
settlement PnL CSV.  It is therefore diagnostic-only and never promotion,
paper/live, or execution-handoff evidence.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.contracts import compact_safety_fields
from bigan.v8.polymarket.training.execution_layer_v2 import (
    EXECUTION_LAYER_V2_FORBIDDEN_OUTCOME_FIELDS,
)

EXECUTION_LAYER_V2_POLICY_REPLAY_SCHEMA_VERSION = (
    "bigan-v8-polymarket-execution-layer-v2-policy-replay-v1"
)
EXECUTION_LAYER_V2_POLICY_REPLAY_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-polymarket-execution-layer-v2-policy-replay-manifest-v1"
)
EXECUTION_LAYER_V2_CALIBRATED_EV_MAPPING_SCHEMA_VERSION = (
    "bigan-v8-polymarket-execution-layer-v2-calibrated-ev-mapping-v1"
)
EXECUTION_LAYER_V2_CALIBRATED_EV_SOURCE_SCHEMA_VERSION = (
    "bigan-v8-polymarket-execution-layer-v2-calibrated-ev-source-v1"
)
EXECUTION_LAYER_V2_FORWARD_SHADOW_SCHEMA_VERSION = (
    "bigan-v8-polymarket-execution-layer-v2-forward-shadow-policy-v1"
)
EXECUTION_LAYER_V2_FORWARD_SHADOW_GUARD_INTERSECTION_SCHEMA_VERSION = (
    "bigan-v8-polymarket-execution-layer-v2-forward-shadow-guard-intersection-v1"
)
EXECUTION_LAYER_V2_HTS_TIME_WINDOW_REMAP_SCHEMA_VERSION = (
    "bigan-v8-polymarket-execution-layer-v2-hts-time-window-remap-v1"
)
EXECUTION_LAYER_V2_HTS_REGIME_RISK_REPLAY_SCHEMA_VERSION = (
    "bigan-v8-polymarket-execution-layer-v2-hts-regime-risk-replay-v1"
)
EXECUTION_LAYER_V2_HTS_REGIME_RISK_REPLAY_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-polymarket-execution-layer-v2-hts-regime-risk-replay-manifest-v1"
)
EXECUTION_LAYER_V2_FORWARD_SHADOW_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-polymarket-execution-layer-v2-forward-shadow-manifest-v1"
)
EXECUTION_LAYER_V2_RECOMMENDED_EXECUTION_POLICY_NAME = (
    "bucket_aware_execution_policy_v1_diagnostic"
)

POLICY_REPLAY_VARIANTS: tuple[str, ...] = (
    "all_executed_baseline",
    "price_070_090_only",
    "exclude_buy_up_hts",
    "sell_before_close_only",
    "buy_down_hts_only",
    "five_min_only",
    "fifteen_min_only",
    "bucket_aware_v1_conservative",
    "bucket_aware_v1_plus_sbc",
)

FORWARD_SHADOW_POLICY_VARIANTS: tuple[str, ...] = (
    "baseline_current_guard",
    "bucket_aware_v1_conservative",
    "bucket_aware_v1_plus_sbc",
    "calibrated_ev_v2",
    "calibrated_ev_plus_bucket_v2",
)
HTS_REGIME_RISK_POLICY_VARIANTS: tuple[str, ...] = (
    "baseline_all",
    "side_blind_hts",
    "regime_aware_up_down_hts",
    "up_hts_only_when_up_regime_confirmed",
    "down_hts_only_when_down_regime_confirmed",
    "hts_allowed_only_when_regime_and_price_bucket_agree",
    "hts_to_sbc_when_late_or_uncertain",
)
HTS_REGIME_CANONICAL_FEATURE_FIELDS: tuple[str, ...] = (
    "btc_momentum",
    "reference_price_to_beat_distance_at_decision",
    "time_since_market_start_seconds",
    "action_score_margin",
    "side_specific_action_score_margin",
)

PRICE_BUCKET_EDGES: tuple[tuple[str, float, float | None], ...] = (
    ("lt_0_60", -math.inf, 0.60),
    ("0_60_0_70", 0.60, 0.70),
    ("0_70_0_90", 0.70, 0.90),
    ("gt_0_90", 0.90, None),
)
MISSING_SORT_NUMBER = 10**30
DEFAULT_FORWARD_SHADOW_EXECUTION_COST = 0.001


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2PolicyReplayConfig:
    """Configuration for a settlement-CSV policy replay bundle."""

    run_id: str
    input_csv: Path | str
    output_dir: Path | str
    overwrite_existing: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False
    v8_execution_handoff_allowed: bool = False
    source_model_candidate_eligible: bool = False
    freeze_ready: bool = False
    promotion_evidence_eligible: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        object.__setattr__(self, "input_csv", Path(self.input_csv))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        _validate_safety_flags(self)

    @property
    def run_dir(self) -> Path:
        return self.output_dir.expanduser().resolve() / self.run_id

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_csv"] = str(self.input_csv)
        payload["output_dir"] = str(self.output_dir)
        return payload


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2PolicyReplayResult:
    """Written settlement-CSV replay bundle."""

    output_dir: Path
    artifact_paths: dict[str, Path]
    artifact_hashes: dict[str, str]
    report: dict[str, Any]
    manifest: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2HTSRegimeRiskReplayConfig:
    """Configuration for outcome-aware HTS regime risk diagnostics."""

    run_id: str
    input_path: Path | str
    output_dir: Path | str
    overwrite_existing: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False
    v8_execution_handoff_allowed: bool = False
    source_model_candidate_eligible: bool = False
    freeze_ready: bool = False
    promotion_evidence_eligible: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        object.__setattr__(self, "input_path", Path(self.input_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        _validate_safety_flags(self)

    @property
    def run_dir(self) -> Path:
        return self.output_dir.expanduser().resolve() / self.run_id

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_path"] = str(self.input_path)
        payload["output_dir"] = str(self.output_dir)
        return payload


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2HTSRegimeRiskReplayResult:
    """Written HTS regime risk diagnostic bundle."""

    output_dir: Path
    artifact_paths: dict[str, Path]
    artifact_hashes: dict[str, str]
    report: dict[str, Any]
    manifest: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2ForwardShadowConfig:
    """Configuration for outcome-free execution-layer v2 forward shadow replay."""

    run_id: str
    input_path: Path | str
    output_dir: Path | str
    overwrite_existing: bool = False
    max_rows: int | None = None
    entry_ev_threshold: float = 0.02
    default_execution_cost: float = DEFAULT_FORWARD_SHADOW_EXECUTION_COST
    frozen_ev_calibration_artifact: Path | str | None = None
    paper_only: bool = True
    capital_at_risk: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False
    v8_execution_handoff_allowed: bool = False
    source_model_candidate_eligible: bool = False
    freeze_ready: bool = False
    promotion_evidence_eligible: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if self.max_rows is not None and self.max_rows <= 0:
            raise ValueError("max_rows must be positive when provided")
        if not math.isfinite(self.entry_ev_threshold) or self.entry_ev_threshold < 0.0:
            raise ValueError("entry_ev_threshold must be finite and non-negative")
        if not math.isfinite(self.default_execution_cost) or self.default_execution_cost < 0.0:
            raise ValueError("default_execution_cost must be finite and non-negative")
        object.__setattr__(self, "input_path", Path(self.input_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.frozen_ev_calibration_artifact is not None:
            object.__setattr__(
                self,
                "frozen_ev_calibration_artifact",
                Path(self.frozen_ev_calibration_artifact),
            )
        _validate_safety_flags(self)

    @property
    def run_dir(self) -> Path:
        return self.output_dir.expanduser().resolve() / self.run_id

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_path"] = str(self.input_path)
        payload["output_dir"] = str(self.output_dir)
        if self.frozen_ev_calibration_artifact is not None:
            payload["frozen_ev_calibration_artifact"] = str(
                self.frozen_ev_calibration_artifact
            )
        return payload


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2ForwardShadowResult:
    """Written calibrated-EV mapping and forward-shadow artifact bundle."""

    output_dir: Path
    artifact_paths: dict[str, Path]
    artifact_hashes: dict[str, str]
    calibrated_ev_source_report: dict[str, Any]
    ev_mapping_report: dict[str, Any]
    forward_shadow_report: dict[str, Any]
    guard_intersection_report: dict[str, Any]
    hts_time_window_remap_report: dict[str, Any]
    manifest: dict[str, Any]


def run_execution_layer_v2_policy_replay_from_settlement_csv(
    config: ExecutionLayerV2PolicyReplayConfig,
) -> ExecutionLayerV2PolicyReplayResult:
    """Run and write diagnostic policy replay artifacts."""

    if not config.input_csv.exists():
        raise FileNotFoundError(f"settlement CSV not found: {config.input_csv}")
    run_dir = config.run_dir
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"policy replay output exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    raw_rows = _load_csv_rows(config.input_csv)
    normalized_rows = [_normalize_settlement_row(row, index) for index, row in enumerate(raw_rows)]
    report = build_execution_layer_v2_policy_replay_report(
        normalized_rows,
        run_id=config.run_id,
        input_csv=str(config.input_csv),
    )

    artifact_paths = {
        "execution_layer_v2_policy_replay_report": run_dir
        / "execution_layer_v2_policy_replay_report.json",
        "execution_layer_v2_policy_replay_summary": run_dir
        / "execution_layer_v2_policy_replay_report.md",
        "execution_layer_v2_policy_replay_manifest": run_dir
        / "execution_layer_v2_policy_replay_manifest.json",
    }
    _write_json(artifact_paths["execution_layer_v2_policy_replay_report"], report)
    _write_text(
        artifact_paths["execution_layer_v2_policy_replay_summary"],
        execution_layer_v2_policy_replay_report_to_markdown(report),
    )
    artifact_hashes = {
        "execution_layer_v2_policy_replay_report": _sha256_file(
            artifact_paths["execution_layer_v2_policy_replay_report"]
        ),
        "execution_layer_v2_policy_replay_summary": _sha256_file(
            artifact_paths["execution_layer_v2_policy_replay_summary"]
        ),
    }
    manifest = {
        "schema_version": EXECUTION_LAYER_V2_POLICY_REPLAY_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "input_csv": str(config.input_csv),
        "artifact_paths": {name: str(path) for name, path in artifact_paths.items()},
        "artifact_hashes": dict(artifact_hashes),
        "report_id": report["execution_layer_v2_policy_replay_report_id"],
        "row_count": report["row_count"],
        "policy_variant_names": list(POLICY_REPLAY_VARIANTS),
        "max_drawdown_ordering": report["max_drawdown_ordering"],
        "chronological_sort_fields": list(report["chronological_sort_fields"]),
        "recommended_execution_policy": report["recommended_execution_policy_v1"][
            "policy_name"
        ],
        "ev_mapping_status": report["signal_to_ev_diagnostic"]["ev_mapping_status"],
        "diagnostic_only": True,
        "uses_settlement_pnl_csv_for_evaluation": True,
        "uses_settlement_pnl_csv_for_tuning": False,
        "source_scores_mutated": False,
        "paper_live_unlock_changed": False,
        **_safety_report_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    _write_json(artifact_paths["execution_layer_v2_policy_replay_manifest"], manifest)
    artifact_hashes["execution_layer_v2_policy_replay_manifest"] = _sha256_file(
        artifact_paths["execution_layer_v2_policy_replay_manifest"]
    )
    return ExecutionLayerV2PolicyReplayResult(
        output_dir=run_dir,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        report=report,
        manifest=manifest,
    )


def run_execution_layer_v2_hts_regime_risk_replay(
    config: ExecutionLayerV2HTSRegimeRiskReplayConfig,
) -> ExecutionLayerV2HTSRegimeRiskReplayResult:
    """Run and write diagnostic-only HTS regime risk replay artifacts."""

    if not config.input_path.exists():
        raise FileNotFoundError(f"HTS regime replay input not found: {config.input_path}")
    run_dir = config.run_dir
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"HTS regime replay output exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    rows = _load_hts_regime_replay_rows(config.input_path)
    report = build_execution_layer_v2_hts_regime_risk_replay_report(
        rows,
        run_id=config.run_id,
        input_path=str(config.input_path),
    )
    artifact_paths = {
        "execution_layer_v2_hts_regime_risk_replay_report": run_dir
        / "execution_layer_v2_hts_regime_risk_replay_report.json",
        "execution_layer_v2_hts_regime_risk_replay_summary": run_dir
        / "execution_layer_v2_hts_regime_risk_replay_report.md",
        "execution_layer_v2_hts_regime_risk_replay_manifest": run_dir
        / "execution_layer_v2_hts_regime_risk_replay_manifest.json",
    }
    _write_json(
        artifact_paths["execution_layer_v2_hts_regime_risk_replay_report"],
        report,
    )
    _write_text(
        artifact_paths["execution_layer_v2_hts_regime_risk_replay_summary"],
        execution_layer_v2_hts_regime_risk_replay_report_to_markdown(report),
    )
    artifact_hashes = {
        "execution_layer_v2_hts_regime_risk_replay_report": _sha256_file(
            artifact_paths["execution_layer_v2_hts_regime_risk_replay_report"]
        ),
        "execution_layer_v2_hts_regime_risk_replay_summary": _sha256_file(
            artifact_paths["execution_layer_v2_hts_regime_risk_replay_summary"]
        ),
    }
    manifest = {
        "schema_version": (
            EXECUTION_LAYER_V2_HTS_REGIME_RISK_REPLAY_MANIFEST_SCHEMA_VERSION
        ),
        "run_id": config.run_id,
        "input_path": str(config.input_path),
        "artifact_paths": {name: str(path) for name, path in artifact_paths.items()},
        "artifact_hashes": dict(artifact_hashes),
        "report_id": report["execution_layer_v2_hts_regime_risk_replay_report_id"],
        "fill_count": report["fill_count"],
        "hts_fill_count": report["hts_fill_count"],
        "settled_pnl": report["policy_variants"]["baseline_all"]["settled_pnl"],
        "policy_variant_names": list(HTS_REGIME_RISK_POLICY_VARIANTS),
        "recommended_guard_signal_count": len(
            report["recommended_decision_time_guard_signals"]
        ),
        "diagnostic_only": True,
        "outcome_aware_offline_replay": True,
        "uses_outcome_for_policy_selection": False,
        "uses_outcome_for_offline_evaluation": True,
        "thresholds_tuned": False,
        "source_scores_mutated": False,
        "o_score_mutated": False,
        "paper_live_unlock_changed": False,
        **_safety_report_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    _write_json(
        artifact_paths["execution_layer_v2_hts_regime_risk_replay_manifest"],
        manifest,
    )
    artifact_hashes["execution_layer_v2_hts_regime_risk_replay_manifest"] = (
        _sha256_file(
            artifact_paths["execution_layer_v2_hts_regime_risk_replay_manifest"]
        )
    )
    return ExecutionLayerV2HTSRegimeRiskReplayResult(
        output_dir=run_dir,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        report=report,
        manifest=manifest,
    )


def run_execution_layer_v2_forward_shadow_policy(
    config: ExecutionLayerV2ForwardShadowConfig,
) -> ExecutionLayerV2ForwardShadowResult:
    """Run outcome-free calibrated EV mapping and policy shadow diagnostics."""

    if not config.input_path.exists():
        raise FileNotFoundError(f"forward shadow input not found: {config.input_path}")
    run_dir = config.run_dir
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"forward shadow output exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    raw_rows = _load_forward_shadow_rows(config.input_path)
    if config.max_rows is not None:
        raw_rows = raw_rows[: config.max_rows]
    forbidden = _forbidden_fields_by_row(raw_rows)
    ev_calibration_artifact = _load_frozen_ev_calibration_artifact(
        config.frozen_ev_calibration_artifact
    )
    normalized_rows = (
        []
        if forbidden
        else [
            _normalize_forward_shadow_row(
                row,
                index,
                default_execution_cost=config.default_execution_cost,
                ev_calibration_artifact=ev_calibration_artifact,
            )
            for index, row in enumerate(raw_rows)
        ]
    )
    ev_mapping_report = build_execution_layer_v2_calibrated_ev_mapping_report(
        normalized_rows,
        run_id=config.run_id,
        input_path=str(config.input_path),
        raw_row_count=len(raw_rows),
        forbidden_outcome_fields_by_row=forbidden,
    )
    forward_shadow_report = build_execution_layer_v2_forward_shadow_policy_report(
        normalized_rows,
        run_id=config.run_id,
        input_path=str(config.input_path),
        raw_row_count=len(raw_rows),
        forbidden_outcome_fields_by_row=forbidden,
        entry_ev_threshold=config.entry_ev_threshold,
    )
    guard_intersection_report = (
        build_execution_layer_v2_forward_shadow_guard_intersection_report(
            normalized_rows,
            run_id=config.run_id,
            input_path=str(config.input_path),
            raw_row_count=len(raw_rows),
            forbidden_outcome_fields_by_row=forbidden,
            entry_ev_threshold=config.entry_ev_threshold,
        )
    )
    hts_time_window_remap_report = (
        build_execution_layer_v2_hts_time_window_remap_report(
            normalized_rows,
            run_id=config.run_id,
            input_path=str(config.input_path),
            raw_row_count=len(raw_rows),
            forbidden_outcome_fields_by_row=forbidden,
            entry_ev_threshold=config.entry_ev_threshold,
        )
    )
    calibrated_ev_source_report = build_execution_layer_v2_calibrated_ev_source_report(
        normalized_rows,
        run_id=config.run_id,
        input_path=str(config.input_path),
        raw_row_count=len(raw_rows),
        forbidden_outcome_fields_by_row=forbidden,
        ev_calibration_artifact=ev_calibration_artifact,
        forward_shadow_report=forward_shadow_report,
        guard_intersection_report=guard_intersection_report,
    )
    artifact_paths = {
        "execution_layer_v2_calibrated_ev_source_report": run_dir
        / "execution_layer_v2_calibrated_ev_source_report.json",
        "execution_layer_v2_calibrated_ev_source_summary": run_dir
        / "execution_layer_v2_calibrated_ev_source_report.md",
        "execution_layer_v2_calibrated_ev_mapping_report": run_dir
        / "execution_layer_v2_calibrated_ev_mapping_report.json",
        "execution_layer_v2_calibrated_ev_mapping_summary": run_dir
        / "execution_layer_v2_calibrated_ev_mapping_report.md",
        "execution_layer_v2_forward_shadow_policy_report": run_dir
        / "execution_layer_v2_forward_shadow_policy_report.json",
        "execution_layer_v2_forward_shadow_policy_summary": run_dir
        / "execution_layer_v2_forward_shadow_policy_report.md",
        "execution_layer_v2_forward_shadow_guard_intersection_report": run_dir
        / "execution_layer_v2_forward_shadow_guard_intersection_report.json",
        "execution_layer_v2_forward_shadow_guard_intersection_summary": run_dir
        / "execution_layer_v2_forward_shadow_guard_intersection_report.md",
        "execution_layer_v2_hts_time_window_remap_report": run_dir
        / "execution_layer_v2_hts_time_window_remap_report.json",
        "execution_layer_v2_hts_time_window_remap_summary": run_dir
        / "execution_layer_v2_hts_time_window_remap_report.md",
        "execution_layer_v2_forward_shadow_manifest": run_dir
        / "execution_layer_v2_forward_shadow_manifest.json",
    }
    _write_json(
        artifact_paths["execution_layer_v2_calibrated_ev_source_report"],
        calibrated_ev_source_report,
    )
    _write_text(
        artifact_paths["execution_layer_v2_calibrated_ev_source_summary"],
        execution_layer_v2_calibrated_ev_source_report_to_markdown(
            calibrated_ev_source_report
        ),
    )
    _write_json(
        artifact_paths["execution_layer_v2_calibrated_ev_mapping_report"],
        ev_mapping_report,
    )
    _write_text(
        artifact_paths["execution_layer_v2_calibrated_ev_mapping_summary"],
        execution_layer_v2_calibrated_ev_mapping_report_to_markdown(ev_mapping_report),
    )
    _write_json(
        artifact_paths["execution_layer_v2_forward_shadow_policy_report"],
        forward_shadow_report,
    )
    _write_text(
        artifact_paths["execution_layer_v2_forward_shadow_policy_summary"],
        execution_layer_v2_forward_shadow_policy_report_to_markdown(
            forward_shadow_report
        ),
    )
    _write_json(
        artifact_paths["execution_layer_v2_forward_shadow_guard_intersection_report"],
        guard_intersection_report,
    )
    _write_text(
        artifact_paths["execution_layer_v2_forward_shadow_guard_intersection_summary"],
        execution_layer_v2_forward_shadow_guard_intersection_report_to_markdown(
            guard_intersection_report
        ),
    )
    _write_json(
        artifact_paths["execution_layer_v2_hts_time_window_remap_report"],
        hts_time_window_remap_report,
    )
    _write_text(
        artifact_paths["execution_layer_v2_hts_time_window_remap_summary"],
        execution_layer_v2_hts_time_window_remap_report_to_markdown(
            hts_time_window_remap_report
        ),
    )
    artifact_hashes = {
        name: _sha256_file(path)
        for name, path in artifact_paths.items()
        if name != "execution_layer_v2_forward_shadow_manifest"
    }
    manifest = {
        "schema_version": EXECUTION_LAYER_V2_FORWARD_SHADOW_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "input_path": str(config.input_path),
        "frozen_ev_calibration_artifact": (
            str(config.frozen_ev_calibration_artifact)
            if config.frozen_ev_calibration_artifact is not None
            else None
        ),
        "artifact_paths": {name: str(path) for name, path in artifact_paths.items()},
        "artifact_hashes": dict(artifact_hashes),
        "calibrated_ev_source_report_id": calibrated_ev_source_report[
            "execution_layer_v2_calibrated_ev_source_report_id"
        ],
        "ev_mapping_report_id": ev_mapping_report[
            "execution_layer_v2_calibrated_ev_mapping_report_id"
        ],
        "forward_shadow_report_id": forward_shadow_report[
            "execution_layer_v2_forward_shadow_policy_report_id"
        ],
        "forward_shadow_guard_intersection_report_id": guard_intersection_report[
            "execution_layer_v2_forward_shadow_guard_intersection_report_id"
        ],
        "hts_time_window_remap_report_id": hts_time_window_remap_report[
            "execution_layer_v2_hts_time_window_remap_report_id"
        ],
        "raw_row_count": len(raw_rows),
        "accepted_signal_row_count": len(normalized_rows),
        "forbidden_outcome_fields_present": bool(forbidden),
        "ev_mapping_status": ev_mapping_report["ev_mapping_status"],
        "calibrated_ev_source_status": calibrated_ev_source_report[
            "calibrated_ev_source_status"
        ],
        "calibrated_ev_available": ev_mapping_report["calibrated_ev_available"],
        "calibrated_ev_produced_count": calibrated_ev_source_report[
            "calibrated_ev_produced_count"
        ],
        "calibrated_ev_missing_count": calibrated_ev_source_report[
            "calibrated_ev_missing_count"
        ],
        "frozen_ev_calibration_artifact_hash": calibrated_ev_source_report[
            "calibration_artifact_hash"
        ],
        "market_implied_probability_used_for_ev": ev_mapping_report[
            "market_implied_probability_used_for_ev"
        ],
        "forward_shadow_policy_variant_names": list(FORWARD_SHADOW_POLICY_VARIANTS),
        "guard_intersection_policy_variant_names": list(FORWARD_SHADOW_POLICY_VARIANTS),
        "guard_intersection_summary": {
            name: {
                "policy_candidate_count": metrics["policy_candidate_count"],
                "guard_passed_candidate_count": metrics[
                    "guard_passed_candidate_count"
                ],
                "guard_unknown_candidate_count": metrics[
                    "guard_unknown_candidate_count"
                ],
                "executable_shadow_count": metrics["executable_shadow_count"],
            }
            for name, metrics in guard_intersection_report[
                "policy_variant_guard_intersections"
            ].items()
        },
        "hts_time_window_remap_summary": {
            "hts_time_window_blocked_count": hts_time_window_remap_report[
                "hts_time_window_blocked_count"
            ],
            "same_side_sbc_alternative_available_count": hts_time_window_remap_report[
                "same_side_sbc_alternative_available_count"
            ],
            "same_side_sbc_calibrated_ev_available_count": (
                hts_time_window_remap_report[
                    "same_side_sbc_calibrated_ev_available_count"
                ]
            ),
            "same_side_sbc_guard_passed_count": hts_time_window_remap_report[
                "same_side_sbc_guard_passed_count"
            ],
            "remap_candidate_count": hts_time_window_remap_report[
                "remap_candidate_count"
            ],
            "remap_guard_passed_count": hts_time_window_remap_report[
                "remap_guard_passed_count"
            ],
            "reason_distribution": hts_time_window_remap_report[
                "remap_reason_distribution"
            ],
        },
        "diagnostic_only": True,
        "uses_settlement_pnl_or_outcome_labels": False,
        "uses_oracle_actions_or_future_returns": False,
        "uses_settlement_labels_for_threshold_tuning": False,
        "source_scores_mutated": False,
        "o_score_mutated": False,
        "paper_live_unlock_changed": False,
        **_safety_report_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    _write_json(artifact_paths["execution_layer_v2_forward_shadow_manifest"], manifest)
    artifact_hashes["execution_layer_v2_forward_shadow_manifest"] = _sha256_file(
        artifact_paths["execution_layer_v2_forward_shadow_manifest"]
    )
    return ExecutionLayerV2ForwardShadowResult(
        output_dir=run_dir,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        calibrated_ev_source_report=calibrated_ev_source_report,
        ev_mapping_report=ev_mapping_report,
        forward_shadow_report=forward_shadow_report,
        guard_intersection_report=guard_intersection_report,
        hts_time_window_remap_report=hts_time_window_remap_report,
        manifest=manifest,
    )


def build_execution_layer_v2_policy_replay_report(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    input_csv: str,
) -> dict[str, Any]:
    """Build a diagnostic replay report from normalized settlement rows."""

    variant_reports = {
        name: _policy_variant_metrics(rows, name) for name in POLICY_REPLAY_VARIANTS
    }
    report = {
        "schema_version": EXECUTION_LAYER_V2_POLICY_REPLAY_SCHEMA_VERSION,
        "run_id": run_id,
        "input_csv": input_csv,
        "row_count": len(rows),
        "diagnostic_only": True,
        "uses_settlement_pnl_csv_for_evaluation": True,
        "uses_settlement_pnl_csv_for_tuning": False,
        "thresholds_tuned": False,
        "source_scores_mutated": False,
        "o_score_mutated": False,
        "source_ranking_score_mutated": False,
        "paper_live_unlock_changed": False,
        "max_drawdown_ordering": "chronological",
        "chronological_sort_fields": [
            "numeric_iteration",
            "decision_ts_numeric",
            "intent_id",
            "row_index",
        ],
        "policy_variant_names": list(POLICY_REPLAY_VARIANTS),
        "policy_variant_definitions": _policy_variant_definitions(),
        "policy_variants": variant_reports,
        "price_bucket_summary": _price_bucket_summary(rows),
        "action_family_summary": _family_summary(rows),
        "signal_to_ev_diagnostic": _signal_to_ev_diagnostic(rows),
        "recommended_execution_policy_v1": _recommended_execution_policy(rows, variant_reports),
        "small_sample_warnings": _small_sample_warnings(variant_reports),
        **_safety_report_fields(),
    }
    report["execution_layer_v2_policy_replay_report_id"] = canonical_json_sha256(report)
    return report


def build_execution_layer_v2_hts_regime_risk_replay_report(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    input_path: str,
) -> dict[str, Any]:
    """Build an outcome-aware, decision-time-feature HTS risk diagnostic report."""

    variant_reports = {
        name: _hts_regime_policy_metrics(rows, name)
        for name in HTS_REGIME_RISK_POLICY_VARIANTS
    }
    hts_rows = [row for row in rows if row["family"] == "HOLD_TO_SETTLEMENT"]
    up_hts_rows = [
        row for row in hts_rows if row["action"] == "BUY_UP_HOLD_TO_SETTLEMENT"
    ]
    report = {
        "schema_version": EXECUTION_LAYER_V2_HTS_REGIME_RISK_REPLAY_SCHEMA_VERSION,
        "run_id": run_id,
        "input_path": input_path,
        "fill_count": len(rows),
        "hts_fill_count": len(hts_rows),
        "up_hts_fill_count": len(up_hts_rows),
        "diagnostic_only": True,
        "outcome_aware_offline_replay": True,
        "uses_outcome_for_policy_selection": False,
        "uses_outcome_for_offline_evaluation": True,
        "uses_settlement_pnl_for_decision_time_logic": False,
        "uses_oracle_actions_or_future_returns": False,
        "uses_validation_labels_for_threshold_tuning": False,
        "thresholds_tuned": False,
        "global_up_hts_disable_recommended": False,
        "policy_variant_names": list(HTS_REGIME_RISK_POLICY_VARIANTS),
        "policy_variant_definitions": _hts_regime_policy_definitions(),
        "decision_time_regime_feature_fields": _hts_regime_decision_time_fields(),
        "evaluation_only_fields": [
            "resolved_outcome",
            "settlement_pnl",
            "settlement_status",
        ],
        "feature_coverage_before": _hts_regime_feature_coverage(
            rows,
            canonical_only=True,
        ),
        "feature_coverage_after": _hts_regime_feature_coverage(rows),
        "feature_coverage": _hts_regime_feature_coverage(rows),
        "policy_variants": variant_reports,
        "pnl_by_side": _pnl_distribution(rows, "side"),
        "pnl_by_action": _pnl_distribution(rows, "action"),
        "pnl_by_regime": _pnl_distribution(rows, "market_regime"),
        "pnl_by_price_bucket": _pnl_distribution(rows, "price_bucket"),
        "pnl_by_time_window": _pnl_distribution(rows, "time_window_bucket"),
        "false_positive_up_hts_examples": _false_positive_up_hts_examples(rows),
        "missed_opportunity_up_hts_examples": _missed_opportunity_up_hts_examples(rows),
        "up_hts_win_examples": _up_hts_win_examples(rows),
        "up_hts_loss_cluster_diagnostics": _up_hts_loss_cluster_diagnostics(rows),
        "recommended_decision_time_guard_signals": (
            _recommended_hts_regime_guard_signals(rows, variant_reports)
        ),
        "recommendation_summary": (
            "Do not disable UP HTS globally. Use regime-aware, decision-time-only "
            "guard signals and keep UP HTS available when UP regime evidence and "
            "entry quality are strong enough."
        ),
        "source_scores_mutated": False,
        "o_score_mutated": False,
        "paper_live_unlock_changed": False,
        **_safety_report_fields(),
    }
    report["execution_layer_v2_hts_regime_risk_replay_report_id"] = (
        canonical_json_sha256(report)
    )
    return report


def execution_layer_v2_policy_replay_report_to_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown summary for #166 review."""

    ev = report["signal_to_ev_diagnostic"]
    policy = report["recommended_execution_policy_v1"]
    lines = [
        "# v8 Execution Layer v2 Policy Replay",
        "",
        f"- run_id: `{report['run_id']}`",
        f"- input_csv: `{report['input_csv']}`",
        f"- row_count: `{report['row_count']}`",
        f"- diagnostic_only: `{report['diagnostic_only']}`",
        f"- ev_mapping_status: `{ev['ev_mapping_status']}`",
        f"- recommended_ev_source: `{ev['recommended_ev_source']}`",
        f"- recommended_policy: `{policy['policy_name']}`",
        f"- max_drawdown_ordering: `{report['max_drawdown_ordering']}`",
        f"- small_sample_warnings: `{report['small_sample_warnings']}`",
        f"- sell_before_close_positive_in_csv: `{policy['sell_before_close_positive_in_csv']}`",
        f"- paper_only: `{report['paper_only']}`",
        f"- capital_at_risk: `{report['capital_at_risk']}`",
        f"- v8_execution_handoff_allowed: `{report['v8_execution_handoff_allowed']}`",
        "",
        "## Policy Variants",
        "",
        "| variant | rows | cost_basis | settlement_pnl | roi | win_rate | max_drawdown |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in POLICY_REPLAY_VARIANTS:
        metrics = report["policy_variants"][name]
        lines.append(
            f"| `{name}` | {metrics['row_count']} | "
            f"{metrics['cost_basis']:.6f} | {metrics['settlement_pnl']:.6f} | "
            f"{metrics['roi']:.6f} | {metrics['win_rate']:.6f} | "
            f"{metrics['max_drawdown']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## EV Mapping Diagnostic",
            "",
            f"- p_market_implied_source_fields: `{ev['p_market_implied_source_fields']}`",
            f"- p_model_fair_value_source_fields_present: `{ev['p_model_fair_value_source_fields_present']}`",
            f"- ev_mapping_blocking_reason_codes: `{ev['ev_mapping_blocking_reason_codes']}`",
            "",
            "## Recommended Policy Rules",
            "",
        ]
    )
    for rule in policy["rules"]:
        lines.append(f"- {rule}")
    lines.append("")
    return "\n".join(lines)


def execution_layer_v2_hts_regime_risk_replay_report_to_markdown(
    report: dict[str, Any],
) -> str:
    """Render the HTS regime risk diagnostic report."""

    lines = [
        "# v8 Execution Layer v2 HTS Regime Risk Replay",
        "",
        f"- run_id: `{report['run_id']}`",
        f"- input_path: `{report['input_path']}`",
        f"- fill_count: `{report['fill_count']}`",
        f"- hts_fill_count: `{report['hts_fill_count']}`",
        f"- up_hts_fill_count: `{report['up_hts_fill_count']}`",
        f"- diagnostic_only: `{report['diagnostic_only']}`",
        "- uses_outcome_for_policy_selection: "
        f"`{report['uses_outcome_for_policy_selection']}`",
        "- uses_outcome_for_offline_evaluation: "
        f"`{report['uses_outcome_for_offline_evaluation']}`",
        "- global_up_hts_disable_recommended: "
        f"`{report['global_up_hts_disable_recommended']}`",
        f"- v8_execution_handoff_allowed: `{report['v8_execution_handoff_allowed']}`",
        "",
        "## Policy Variants",
        "",
        "| variant | fills | pnl | roi | win_rate | max_drawdown |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in HTS_REGIME_RISK_POLICY_VARIANTS:
        metrics = report["policy_variants"][name]
        lines.append(
            f"| `{name}` | {metrics['fill_count']} | "
            f"{metrics['settled_pnl']:.6f} | {metrics['roi']:.6f} | "
            f"{metrics['win_rate']:.6f} | {metrics['max_drawdown']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## PnL Breakdown",
            "",
            f"- pnl_by_side: `{report['pnl_by_side']}`",
            f"- pnl_by_action: `{report['pnl_by_action']}`",
            f"- pnl_by_regime: `{report['pnl_by_regime']}`",
            f"- pnl_by_price_bucket: `{report['pnl_by_price_bucket']}`",
            f"- pnl_by_time_window: `{report['pnl_by_time_window']}`",
            "",
            "## Recommended Decision-Time Guard Signals",
            "",
        ]
    )
    for signal in report["recommended_decision_time_guard_signals"]:
        lines.append(f"- {signal}")
    lines.extend(
        [
            "",
            "## False-Positive UP HTS Examples",
            "",
            "| market | action | regime | price_bucket | time_window | pnl | reasons |",
            "| --- | --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for row in report["false_positive_up_hts_examples"][:10]:
        lines.append(
            f"| `{_short_id(row['market_id'])}` | `{row['action']}` | "
            f"`{row['market_regime']}` | `{row['price_bucket']}` | "
            f"`{row['time_window_bucket']}` | {row['settlement_pnl']:.6f} | "
            f"`{row['diagnostic_reason_codes']}` |"
        )
    lines.extend(
        [
            "",
            "## Missed-Opportunity UP HTS Examples",
            "",
            "| market | action | regime | price_bucket | time_window | pnl | reasons |",
            "| --- | --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for row in report["missed_opportunity_up_hts_examples"][:10]:
        lines.append(
            f"| `{_short_id(row['market_id'])}` | `{row['action']}` | "
            f"`{row['market_regime']}` | `{row['price_bucket']}` | "
            f"`{row['time_window_bucket']}` | {row['settlement_pnl']:.6f} | "
            f"`{row['diagnostic_reason_codes']}` |"
        )
    lines.extend(
        [
            "",
            "## Feature Coverage",
            "",
            f"- before: `{report['feature_coverage_before']}`",
            f"- after: `{report['feature_coverage_after']}`",
            "",
            "## UP HTS Loss Clusters",
            "",
            f"`{report['up_hts_loss_cluster_diagnostics']}`",
        ]
    )
    lines.append("")
    return "\n".join(lines)


def build_execution_layer_v2_calibrated_ev_source_report(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    input_path: str,
    raw_row_count: int | None = None,
    forbidden_outcome_fields_by_row: list[dict[str, Any]] | None = None,
    ev_calibration_artifact: dict[str, Any],
    forward_shadow_report: dict[str, Any],
    guard_intersection_report: dict[str, Any],
) -> dict[str, Any]:
    """Build calibrated EV source production diagnostics."""

    forbidden = forbidden_outcome_fields_by_row or []
    missing_rows = [
        row
        for row in rows
        if not row["calibrated_action_expected_net_return_available"]
    ]
    produced_rows = [
        row for row in rows if row["calibrated_ev_source"] == "frozen_ev_calibration_artifact"
    ]
    input_rows = [
        row
        for row in rows
        if row["calibrated_ev_source"] == "input_calibrated_action_expected_net_return"
    ]
    reason_codes = sorted(
        {
            reason
            for row in missing_rows
            for reason in row["calibrated_ev_blocking_reason_codes"]
        }
    )
    if forbidden:
        status = "blocked_forbidden_outcome_fields_present"
        reason_codes = ["forbidden_outcome_fields_present"]
    elif not rows:
        status = "blocked_no_forward_shadow_rows"
        reason_codes = ["no_forward_shadow_rows"]
    elif missing_rows:
        status = "blocked_missing_calibrated_ev_source"
    else:
        status = "calibrated_ev_source_available"

    calibrated_ev_v2 = forward_shadow_report["policy_variants"]["calibrated_ev_v2"]
    calibrated_ev_plus_bucket = forward_shadow_report["policy_variants"][
        "calibrated_ev_plus_bucket_v2"
    ]
    guard_intersections = guard_intersection_report["policy_variant_guard_intersections"]
    report = {
        "schema_version": EXECUTION_LAYER_V2_CALIBRATED_EV_SOURCE_SCHEMA_VERSION,
        "run_id": run_id,
        "input_path": input_path,
        "raw_row_count": raw_row_count if raw_row_count is not None else len(rows),
        "accepted_signal_row_count": len(rows),
        "diagnostic_only": True,
        "uses_settlement_pnl_or_outcome_labels": False,
        "uses_oracle_actions_or_future_returns": False,
        "uses_settlement_labels_for_threshold_tuning": False,
        "forbidden_outcome_fields_present": bool(forbidden),
        "forbidden_outcome_fields_by_row": forbidden,
        "calibrated_ev_source_status": status,
        "calibrated_ev_source_blocking_reason_codes": reason_codes,
        "calibrated_ev_source": ev_calibration_artifact["source"],
        "calibration_artifact_path": ev_calibration_artifact["path"],
        "calibration_artifact_hash": ev_calibration_artifact["sha256"],
        "calibration_artifact_valid": ev_calibration_artifact["valid"],
        "calibration_artifact_status": ev_calibration_artifact["status"],
        "calibration_artifact_blocking_reason_codes": ev_calibration_artifact[
            "blocking_reason_codes"
        ],
        "calibrated_ev_produced_count": len(produced_rows),
        "input_calibrated_ev_count": len(input_rows),
        "calibrated_ev_missing_count": len(missing_rows),
        "calibrated_ev_available_count": len(rows) - len(missing_rows),
        "source_fields_used": sorted(
            {
                field
                for row in produced_rows
                for field in row["calibrated_ev_source_provenance"][
                    "source_fields_used"
                ]
                if field
            }
        ),
        "source_rows": [_calibrated_ev_source_row(row) for row in rows],
        "calibrated_ev_v2_candidate_count": calibrated_ev_v2[
            "allowed_decision_count"
        ],
        "calibrated_ev_v2_guard_passed_count": guard_intersections[
            "calibrated_ev_v2"
        ]["guard_passed_candidate_count"],
        "calibrated_ev_plus_bucket_v2_candidate_count": calibrated_ev_plus_bucket[
            "allowed_decision_count"
        ],
        "calibrated_ev_plus_bucket_v2_guard_passed_count": guard_intersections[
            "calibrated_ev_plus_bucket_v2"
        ]["guard_passed_candidate_count"],
        "market_implied_probability_used_for_ev": False,
        "source_scores_mutated": False,
        "o_score_mutated": False,
        "paper_live_unlock_changed": False,
        **_safety_report_fields(),
    }
    report["execution_layer_v2_calibrated_ev_source_report_id"] = (
        canonical_json_sha256(report)
    )
    return report


def execution_layer_v2_calibrated_ev_source_report_to_markdown(
    report: dict[str, Any],
) -> str:
    """Render calibrated EV source production diagnostics."""

    return "\n".join(
        [
            "# v8 Execution Layer v2 Calibrated EV Source",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- raw_row_count: `{report['raw_row_count']}`",
            f"- accepted_signal_row_count: `{report['accepted_signal_row_count']}`",
            f"- calibrated_ev_source_status: `{report['calibrated_ev_source_status']}`",
            f"- calibrated_ev_source: `{report['calibrated_ev_source']}`",
            f"- calibration_artifact_path: `{report['calibration_artifact_path']}`",
            f"- calibration_artifact_hash: `{report['calibration_artifact_hash']}`",
            f"- calibration_artifact_valid: `{report['calibration_artifact_valid']}`",
            f"- calibrated_ev_produced_count: `{report['calibrated_ev_produced_count']}`",
            f"- input_calibrated_ev_count: `{report['input_calibrated_ev_count']}`",
            f"- calibrated_ev_missing_count: `{report['calibrated_ev_missing_count']}`",
            f"- calibrated_ev_v2_candidate_count: `{report['calibrated_ev_v2_candidate_count']}`",
            f"- calibrated_ev_v2_guard_passed_count: `{report['calibrated_ev_v2_guard_passed_count']}`",
            f"- calibrated_ev_plus_bucket_v2_candidate_count: `{report['calibrated_ev_plus_bucket_v2_candidate_count']}`",
            f"- calibrated_ev_plus_bucket_v2_guard_passed_count: `{report['calibrated_ev_plus_bucket_v2_guard_passed_count']}`",
            f"- market_implied_probability_used_for_ev: `{report['market_implied_probability_used_for_ev']}`",
            f"- v8_execution_handoff_allowed: `{report['v8_execution_handoff_allowed']}`",
            "",
            "## Blocking Reasons",
            "",
            *_markdown_list(report["calibrated_ev_source_blocking_reason_codes"]),
            "",
        ]
    )


def build_execution_layer_v2_calibrated_ev_mapping_report(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    input_path: str,
    raw_row_count: int | None = None,
    forbidden_outcome_fields_by_row: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the decision-time EV source contract report."""

    forbidden = forbidden_outcome_fields_by_row or []
    row_count = len(rows)
    missing_rows = [row for row in rows if not row["calibrated_ev_available"]]
    blocking_reasons = sorted(
        {
            reason
            for row in missing_rows
            for reason in row["ev_mapping_blocking_reason_codes"]
        }
    )
    if forbidden:
        ev_status = "blocked_forbidden_outcome_fields_present"
        blocking_reasons = ["forbidden_outcome_fields_present"]
    elif row_count == 0:
        ev_status = "blocked_no_forward_shadow_rows"
        blocking_reasons = ["no_forward_shadow_rows"]
    elif missing_rows:
        ev_status = "blocked_missing_calibrated_ev_source"
    else:
        ev_status = "calibrated_ev_available"

    report = {
        "schema_version": EXECUTION_LAYER_V2_CALIBRATED_EV_MAPPING_SCHEMA_VERSION,
        "run_id": run_id,
        "input_path": input_path,
        "raw_row_count": raw_row_count if raw_row_count is not None else row_count,
        "accepted_signal_row_count": row_count,
        "diagnostic_only": True,
        "uses_settlement_pnl_or_outcome_labels": False,
        "uses_oracle_actions_or_future_returns": False,
        "uses_settlement_labels_for_threshold_tuning": False,
        "forbidden_outcome_fields_present": bool(forbidden),
        "forbidden_outcome_fields_by_row": forbidden,
        "ev_mapping_contract": {
            "required_sources": [
                "p_market_implied",
                "p_model_fair_value",
                "calibrated_action_expected_net_return",
                "canonical_o_action_score",
                "execution_price",
            ],
            "allowed_ev_sources": [
                "calibrated_action_expected_net_return",
                "p_model_fair_value_minus_execution_price_minus_cost",
            ],
            "forbidden_ev_sources": [
                "p_market_implied",
                "execution_price_as_probability",
                "settlement_pnl",
                "oracle_action",
                "future_return",
            ],
            "market_implied_probability_used_for_ev": False,
        },
        "ev_mapping_status": ev_status,
        "ev_mapping_blocking_reason_codes": blocking_reasons,
        "calibrated_ev_available": bool(row_count and not missing_rows and not forbidden),
        "calibrated_ev_available_count": row_count - len(missing_rows),
        "calibrated_ev_missing_count": len(missing_rows),
        "market_implied_probability_used_for_ev": False,
        "probability_source_summary": {
            "p_market_implied_count": sum(
                1 for row in rows if row["p_market_implied"] is not None
            ),
            "p_model_fair_value_count": sum(
                1 for row in rows if row["p_model_fair_value"] is not None
            ),
            "calibrated_action_expected_net_return_count": sum(
                1
                for row in rows
                if row["calibrated_action_expected_net_return"] is not None
            ),
            "canonical_o_action_score_count": sum(
                1 for row in rows if row["canonical_o_action_score"] is not None
            ),
            "execution_price_count": sum(
                1 for row in rows if row["execution_price"] is not None
            ),
        },
        "row_ev_mapping_contracts": [
            _ev_mapping_contract_row(row) for row in rows
        ],
        "source_scores_mutated": False,
        "o_score_mutated": False,
        "paper_live_unlock_changed": False,
        **_safety_report_fields(),
    }
    report["execution_layer_v2_calibrated_ev_mapping_report_id"] = (
        canonical_json_sha256(report)
    )
    return report


def build_execution_layer_v2_forward_shadow_policy_report(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    input_path: str,
    raw_row_count: int | None = None,
    forbidden_outcome_fields_by_row: list[dict[str, Any]] | None = None,
    entry_ev_threshold: float = 0.02,
) -> dict[str, Any]:
    """Build an outcome-free forward shadow comparison of v2 policy variants."""

    forbidden = forbidden_outcome_fields_by_row or []
    if forbidden:
        variant_reports = {
            name: _blocked_forward_shadow_variant_metrics(
                name,
                rows,
                ["forbidden_outcome_fields_present"],
            )
            for name in FORWARD_SHADOW_POLICY_VARIANTS
        }
        policy_status = "blocked_fail_closed"
        blocking_reasons = ["forbidden_outcome_fields_present"]
    else:
        variant_reports = {
            name: _forward_shadow_variant_metrics(
                rows,
                name,
                entry_ev_threshold=entry_ev_threshold,
            )
            for name in FORWARD_SHADOW_POLICY_VARIANTS
        }
        policy_status = "diagnostic_only_fail_closed"
        blocking_reasons = []
        if not rows:
            policy_status = "blocked_fail_closed"
            blocking_reasons.append("no_forward_shadow_rows")

    report = {
        "schema_version": EXECUTION_LAYER_V2_FORWARD_SHADOW_SCHEMA_VERSION,
        "run_id": run_id,
        "input_path": input_path,
        "raw_row_count": raw_row_count if raw_row_count is not None else len(rows),
        "accepted_signal_row_count": len(rows),
        "forward_shadow_policy_status": policy_status,
        "forward_shadow_policy_blocking_reason_codes": blocking_reasons,
        "diagnostic_only": True,
        "uses_settlement_pnl_or_outcome_labels": False,
        "uses_oracle_actions_or_future_returns": False,
        "uses_settlement_labels_for_threshold_tuning": False,
        "entry_ev_threshold": entry_ev_threshold,
        "policy_variant_names": list(FORWARD_SHADOW_POLICY_VARIANTS),
        "policy_variant_definitions": _forward_shadow_variant_definitions(),
        "policy_variants": variant_reports,
        "forbidden_outcome_fields_present": bool(forbidden),
        "forbidden_outcome_fields_by_row": forbidden,
        "time_window_behavior": _forward_shadow_time_window_behavior(rows),
        "market_implied_probability_used_as_calibrated_ev_source": False,
        "market_implied_probability_diagnostic_only": True,
        "source_scores_mutated": False,
        "o_score_mutated": False,
        "paper_live_unlock_changed": False,
        **_safety_report_fields(),
    }
    report["execution_layer_v2_forward_shadow_policy_report_id"] = (
        canonical_json_sha256(report)
    )
    return report


def build_execution_layer_v2_forward_shadow_guard_intersection_report(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    input_path: str,
    raw_row_count: int | None = None,
    forbidden_outcome_fields_by_row: list[dict[str, Any]] | None = None,
    entry_ev_threshold: float = 0.02,
) -> dict[str, Any]:
    """Build policy-candidate by execution-guard intersection diagnostics."""

    forbidden = forbidden_outcome_fields_by_row or []
    if forbidden:
        variant_reports = {
            name: _blocked_guard_intersection_variant_metrics(
                name,
                rows,
                ["forbidden_outcome_fields_present"],
            )
            for name in FORWARD_SHADOW_POLICY_VARIANTS
        }
        status = "blocked_fail_closed"
        blocking_reasons = ["forbidden_outcome_fields_present"]
    else:
        variant_reports = {
            name: _guard_intersection_variant_metrics(
                rows,
                name,
                entry_ev_threshold=entry_ev_threshold,
            )
            for name in FORWARD_SHADOW_POLICY_VARIANTS
        }
        status = "diagnostic_only_fail_closed"
        blocking_reasons = []
        if not rows:
            status = "blocked_fail_closed"
            blocking_reasons.append("no_forward_shadow_rows")

    report = {
        "schema_version": (
            EXECUTION_LAYER_V2_FORWARD_SHADOW_GUARD_INTERSECTION_SCHEMA_VERSION
        ),
        "run_id": run_id,
        "input_path": input_path,
        "raw_row_count": raw_row_count if raw_row_count is not None else len(rows),
        "accepted_signal_row_count": len(rows),
        "guard_intersection_status": status,
        "guard_intersection_blocking_reason_codes": blocking_reasons,
        "diagnostic_only": True,
        "uses_settlement_pnl_or_outcome_labels": False,
        "uses_oracle_actions_or_future_returns": False,
        "uses_settlement_labels_for_threshold_tuning": False,
        "entry_ev_threshold": entry_ev_threshold,
        "policy_variant_names": list(FORWARD_SHADOW_POLICY_VARIANTS),
        "policy_variant_guard_intersections": variant_reports,
        "guard_pass_definition": {
            "order_allowed_must_be_true": True,
            "blocking_reason_codes_must_be_empty": True,
            "selected_action_must_not_be_no_trade": True,
            "safety_flags_must_remain_blocked": True,
            "missing_guard_fields_are_executable": False,
        },
        "missing_guard_fields_reason_code": "missing_execution_guard_decision_fields",
        "forbidden_outcome_fields_present": bool(forbidden),
        "forbidden_outcome_fields_by_row": forbidden,
        "market_implied_probability_used_as_calibrated_ev_source": False,
        "source_scores_mutated": False,
        "o_score_mutated": False,
        "paper_live_unlock_changed": False,
        **_safety_report_fields(),
    }
    report["execution_layer_v2_forward_shadow_guard_intersection_report_id"] = (
        canonical_json_sha256(report)
    )
    return report


def build_execution_layer_v2_hts_time_window_remap_report(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    input_path: str,
    raw_row_count: int | None = None,
    forbidden_outcome_fields_by_row: list[dict[str, Any]] | None = None,
    entry_ev_threshold: float = 0.02,
) -> dict[str, Any]:
    """Build diagnostic-only HTS-to-same-side-SBC time-window remap evidence."""

    forbidden = forbidden_outcome_fields_by_row or []
    if forbidden:
        remap_rows: list[dict[str, Any]] = []
        status = "blocked_fail_closed"
        blocking_reasons = ["forbidden_outcome_fields_present"]
    else:
        remap_rows = [
            _hts_time_window_remap_row(
                row,
                entry_ev_threshold=entry_ev_threshold,
            )
            for row in rows
            if _is_hts_time_window_blocked_row(row)
        ]
        status = "diagnostic_only_fail_closed"
        blocking_reasons = []
        if not rows:
            status = "blocked_fail_closed"
            blocking_reasons.append("no_forward_shadow_rows")

    candidate_rows = [row for row in remap_rows if row["diagnostic_remap_candidate"]]
    guard_passed_rows = [
        row for row in remap_rows if row["diagnostic_remap_guard_passed"]
    ]
    reason_counter: Counter[str] = Counter()
    original_reason_counter: Counter[str] = Counter()
    for row in remap_rows:
        reason_counter.update(row["remap_reason_codes"])
        original_reason_counter.update(row["original_guard_reason_codes"])

    examples = sorted(
        remap_rows,
        key=lambda row: (not row["diagnostic_remap_guard_passed"], row["row_index"]),
    )[:20]
    report = {
        "schema_version": EXECUTION_LAYER_V2_HTS_TIME_WINDOW_REMAP_SCHEMA_VERSION,
        "run_id": run_id,
        "input_path": input_path,
        "raw_row_count": raw_row_count if raw_row_count is not None else len(rows),
        "accepted_signal_row_count": len(rows),
        "hts_time_window_remap_status": status,
        "hts_time_window_remap_blocking_reason_codes": blocking_reasons,
        "diagnostic_only": True,
        "remap_policy": (
            "if BUY_UP/DOWN_HOLD_TO_SETTLEMENT is blocked in "
            "sell_before_close_only_window, evaluate same-side "
            "BUY_UP/DOWN_SELL_BEFORE_CLOSE as a diagnostic-only alternative"
        ),
        "remap_score_source": (
            "existing decision-time calibrated EV from the selected row; "
            "O/source ranking scores are not mutated"
        ),
        "entry_ev_threshold": entry_ev_threshold,
        "hts_time_window_blocked_count": len(remap_rows),
        "same_side_sbc_alternative_available_count": sum(
            1 for row in remap_rows if row["same_side_sbc_alternative_available"]
        ),
        "same_side_sbc_calibrated_ev_available_count": sum(
            1 for row in remap_rows if row["same_side_sbc_calibrated_ev_available"]
        ),
        "same_side_sbc_guard_passed_count": len(guard_passed_rows),
        "remap_candidate_count": len(candidate_rows),
        "remap_guard_passed_count": len(guard_passed_rows),
        "remap_reason_distribution": dict(sorted(reason_counter.items())),
        "original_guard_reason_distribution": dict(
            sorted(original_reason_counter.items())
        ),
        "time_window_distribution": _count_distribution(
            remap_rows,
            "time_window_bucket",
        ),
        "proposed_action_distribution": _count_distribution(
            remap_rows,
            "proposed_same_side_sbc_action",
        ),
        "remap_rows": remap_rows,
        "examples": examples,
        "forbidden_outcome_fields_present": bool(forbidden),
        "forbidden_outcome_fields_by_row": forbidden,
        "uses_settlement_pnl_or_outcome_labels": False,
        "uses_oracle_actions_or_future_returns": False,
        "uses_settlement_labels_for_threshold_tuning": False,
        "market_implied_probability_used_as_calibrated_ev_source": False,
        "source_scores_mutated": False,
        "o_score_mutated": False,
        "paper_live_unlock_changed": False,
        **_safety_report_fields(),
    }
    report["execution_layer_v2_hts_time_window_remap_report_id"] = (
        canonical_json_sha256(report)
    )
    return report


def execution_layer_v2_calibrated_ev_mapping_report_to_markdown(
    report: dict[str, Any],
) -> str:
    """Render the calibrated EV mapping contract report."""

    return "\n".join(
        [
            "# v8 Execution Layer v2 Calibrated EV Mapping",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- raw_row_count: `{report['raw_row_count']}`",
            f"- accepted_signal_row_count: `{report['accepted_signal_row_count']}`",
            f"- ev_mapping_status: `{report['ev_mapping_status']}`",
            f"- calibrated_ev_available: `{report['calibrated_ev_available']}`",
            f"- calibrated_ev_available_count: `{report['calibrated_ev_available_count']}`",
            f"- calibrated_ev_missing_count: `{report['calibrated_ev_missing_count']}`",
            f"- market_implied_probability_used_for_ev: `{report['market_implied_probability_used_for_ev']}`",
            f"- forbidden_outcome_fields_present: `{report['forbidden_outcome_fields_present']}`",
            f"- v8_execution_handoff_allowed: `{report['v8_execution_handoff_allowed']}`",
            "",
            "## Blocking Reasons",
            "",
            *_markdown_list(report["ev_mapping_blocking_reason_codes"]),
            "",
            "## Probability Source Summary",
            "",
            *[
                f"- {key}: `{value}`"
                for key, value in report["probability_source_summary"].items()
            ],
            "",
        ]
    )


def execution_layer_v2_forward_shadow_policy_report_to_markdown(
    report: dict[str, Any],
) -> str:
    """Render the forward shadow policy comparison report."""

    lines = [
        "# v8 Execution Layer v2 Forward Shadow Policy",
        "",
        f"- run_id: `{report['run_id']}`",
        f"- raw_row_count: `{report['raw_row_count']}`",
        f"- accepted_signal_row_count: `{report['accepted_signal_row_count']}`",
        f"- forward_shadow_policy_status: `{report['forward_shadow_policy_status']}`",
        f"- diagnostic_only: `{report['diagnostic_only']}`",
        f"- uses_settlement_pnl_or_outcome_labels: `{report['uses_settlement_pnl_or_outcome_labels']}`",
        f"- market_implied_probability_used_as_calibrated_ev_source: `{report['market_implied_probability_used_as_calibrated_ev_source']}`",
        f"- v8_execution_handoff_allowed: `{report['v8_execution_handoff_allowed']}`",
        "",
        "## Policy Variants",
        "",
        "| variant | allowed | entries | exits | holds | no_trade | rejected |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in FORWARD_SHADOW_POLICY_VARIANTS:
        metrics = report["policy_variants"][name]
        lines.append(
            f"| `{name}` | {metrics['allowed_decision_count']} | "
            f"{metrics['entry_count']} | {metrics['exit_count']} | "
            f"{metrics['hold_count']} | {metrics['no_trade_count']} | "
            f"{metrics['rejected_decision_count']} |"
        )
    lines.extend(
        [
            "",
            "## Blocking Reasons",
            "",
            *_markdown_list(report["forward_shadow_policy_blocking_reason_codes"]),
            "",
        ]
    )
    return "\n".join(lines)


def execution_layer_v2_forward_shadow_guard_intersection_report_to_markdown(
    report: dict[str, Any],
) -> str:
    """Render the forward shadow guard intersection report."""

    lines = [
        "# v8 Execution Layer v2 Forward Shadow Guard Intersection",
        "",
        f"- run_id: `{report['run_id']}`",
        f"- raw_row_count: `{report['raw_row_count']}`",
        f"- accepted_signal_row_count: `{report['accepted_signal_row_count']}`",
        f"- guard_intersection_status: `{report['guard_intersection_status']}`",
        f"- diagnostic_only: `{report['diagnostic_only']}`",
        f"- v8_execution_handoff_allowed: `{report['v8_execution_handoff_allowed']}`",
        "",
        "## Policy Candidate x Guard Intersection",
        "",
        "| variant | policy_candidate_count | guard_passed_candidate_count | "
        "guard_blocked_candidate_count | guard_unknown_candidate_count | "
        "executable_shadow_count |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in FORWARD_SHADOW_POLICY_VARIANTS:
        metrics = report["policy_variant_guard_intersections"][name]
        lines.append(
            f"| `{name}` | {metrics['policy_candidate_count']} | "
            f"{metrics['guard_passed_candidate_count']} | "
            f"{metrics['guard_blocked_candidate_count']} | "
            f"{metrics['guard_unknown_candidate_count']} | "
            f"{metrics['executable_shadow_count']} |"
        )
    lines.extend(
        [
            "",
            "## Blocking Reasons",
            "",
            *_markdown_list(report["guard_intersection_blocking_reason_codes"]),
            "",
        ]
    )
    return "\n".join(lines)


def execution_layer_v2_hts_time_window_remap_report_to_markdown(
    report: dict[str, Any],
) -> str:
    """Render the diagnostic HTS-to-same-side-SBC remap report."""

    lines = [
        "# v8 Execution Layer v2 HTS Time-Window Remap",
        "",
        f"- run_id: `{report['run_id']}`",
        f"- raw_row_count: `{report['raw_row_count']}`",
        f"- accepted_signal_row_count: `{report['accepted_signal_row_count']}`",
        f"- hts_time_window_remap_status: `{report['hts_time_window_remap_status']}`",
        f"- hts_time_window_blocked_count: `{report['hts_time_window_blocked_count']}`",
        "- same_side_sbc_alternative_available_count: "
        f"`{report['same_side_sbc_alternative_available_count']}`",
        "- same_side_sbc_calibrated_ev_available_count: "
        f"`{report['same_side_sbc_calibrated_ev_available_count']}`",
        f"- same_side_sbc_guard_passed_count: `{report['same_side_sbc_guard_passed_count']}`",
        f"- remap_candidate_count: `{report['remap_candidate_count']}`",
        f"- remap_guard_passed_count: `{report['remap_guard_passed_count']}`",
        f"- diagnostic_only: `{report['diagnostic_only']}`",
        f"- source_scores_mutated: `{report['source_scores_mutated']}`",
        f"- o_score_mutated: `{report['o_score_mutated']}`",
        f"- v8_execution_handoff_allowed: `{report['v8_execution_handoff_allowed']}`",
        "",
        "## Remap Reason Distribution",
        "",
        *[
            f"- `{reason}`: `{count}`"
            for reason, count in report["remap_reason_distribution"].items()
        ],
        "",
        "## Original Guard Reason Distribution",
        "",
        *[
            f"- `{reason}`: `{count}`"
            for reason, count in report["original_guard_reason_distribution"].items()
        ],
        "",
        "## Blocking Reasons",
        "",
        *_markdown_list(report["hts_time_window_remap_blocking_reason_codes"]),
        "",
    ]
    return "\n".join(lines)


def _policy_variant_metrics(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    rejected_reasons: Counter[str] = Counter()
    for row in rows:
        allowed, reasons = _variant_allows_row(row, variant)
        if allowed:
            selected.append(row)
        else:
            rejected_reasons.update(reasons)
    pnl_values = [float(row["settlement_pnl"]) for row in selected]
    chronological_rows = sorted(
        selected,
        key=lambda row: tuple(row["chronological_sort_key"]),
    )
    chronological_pnl_values = [float(row["settlement_pnl"]) for row in chronological_rows]
    cost_basis = sum(float(row["cost_basis"]) for row in selected)
    settlement_pnl = sum(pnl_values)
    roi = settlement_pnl / cost_basis if cost_basis else 0.0
    win_rate = (
        sum(1 for value in pnl_values if value > 0.0) / len(pnl_values)
        if pnl_values
        else 0.0
    )
    return {
        "variant_name": variant,
        "row_count": len(selected),
        "cost_basis": cost_basis,
        "settlement_pnl": settlement_pnl,
        "roi": roi,
        "win_rate": win_rate,
        "action_distribution": _count_distribution(selected, "action"),
        "family_distribution": _count_distribution(selected, "family"),
        "horizon_distribution": _count_distribution(selected, "horizon"),
        "price_bucket_distribution": _count_distribution(selected, "price_bucket"),
        "max_drawdown": _max_drawdown(chronological_pnl_values),
        "max_drawdown_ordering": "chronological",
        "chronological_sort_fields": [
            "numeric_iteration",
            "decision_ts_numeric",
            "intent_id",
            "row_index",
        ],
        "rejected_reason_counts": dict(sorted(rejected_reasons.items())),
        "diagnostic_only": True,
    }


def _policy_variant_definitions() -> dict[str, str]:
    return {
        "all_executed_baseline": "all rows from the settlement CSV",
        "price_070_090_only": "only rows with entry price in [0.70, 0.90]",
        "exclude_buy_up_hts": "all rows except BUY_UP_HOLD_TO_SETTLEMENT",
        "sell_before_close_only": "only SELL_BEFORE_CLOSE family rows",
        "buy_down_hts_only": "only BUY_DOWN_HOLD_TO_SETTLEMENT rows",
        "five_min_only": "only 5m horizon rows",
        "fifteen_min_only": "only 15m horizon rows",
        "bucket_aware_v1_conservative": (
            "price 0.70-0.90, exclude BUY_UP_HOLD_TO_SETTLEMENT, "
            "allow BUY_DOWN_HOLD_TO_SETTLEMENT, allow SELL_BEFORE_CLOSE only "
            "if it also passes the 0.70-0.90 price bucket"
        ),
        "bucket_aware_v1_plus_sbc": (
            "allow BUY_DOWN_HOLD_TO_SETTLEMENT only in price 0.70-0.90, "
            "allow SELL_BEFORE_CLOSE regardless of price bucket, exclude "
            "BUY_UP_HOLD_TO_SETTLEMENT"
        ),
    }


def _hts_regime_policy_definitions() -> dict[str, str]:
    return {
        "baseline_all": "all settled paper fills, including HTS and SBC rows",
        "side_blind_hts": "all HOLD_TO_SETTLEMENT entries regardless of side/regime",
        "regime_aware_up_down_hts": (
            "HTS entries only when the selected side is confirmed by "
            "decision-time regime signals"
        ),
        "up_hts_only_when_up_regime_confirmed": (
            "BUY_UP_HOLD_TO_SETTLEMENT only when decision-time UP regime is confirmed"
        ),
        "down_hts_only_when_down_regime_confirmed": (
            "BUY_DOWN_HOLD_TO_SETTLEMENT only when decision-time DOWN regime is confirmed"
        ),
        "hts_allowed_only_when_regime_and_price_bucket_agree": (
            "HTS entries only when side regime is confirmed and entry price is "
            "inside the fixed diagnostic quality bucket"
        ),
        "hts_to_sbc_when_late_or_uncertain": (
            "Observed SBC rows remain candidates; HTS rows are kept only when "
            "regime is confirmed and the time window is HTS-safe. Late or "
            "uncertain HTS rows are reported as SBC-remap candidates without "
            "fabricating counterfactual PnL."
        ),
    }


def _hts_regime_policy_metrics(
    rows: list[dict[str, Any]],
    variant: str,
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    rejected_reasons: Counter[str] = Counter()
    decision_rows: list[dict[str, Any]] = []
    for row in rows:
        allowed, reasons = _hts_regime_variant_allows_row(row, variant)
        if allowed:
            selected.append(row)
        else:
            rejected_reasons.update(reasons)
        decision_rows.append(
            {
                "row_index": row["row_index"],
                "market_id": row["market_id"],
                "decision_ts": row["decision_ts"],
                "action": row["action"],
                "side": row["side"],
                "family": row["family"],
                "market_regime": row["market_regime"],
                "regime_feature_vote_summary": row["regime_feature_vote_summary"],
                "p_up_down_balance": row["p_up_down_balance"],
                "btc_momentum_regime": row["btc_momentum_regime"],
                "reference_distance_bucket": row["reference_distance_bucket"],
                "action_score_margin_bucket": row["action_score_margin_bucket"],
                "price_bucket": row["price_bucket"],
                "time_window_bucket": row["time_window_bucket"],
                "same_market_entry_index": row["same_market_entry_index"],
                "side_context": row["side_context"],
                "policy_selected": allowed,
                "policy_reason_codes": reasons,
                "settlement_pnl": row["settlement_pnl"],
                "resolved_outcome": row["resolved_outcome"],
                "diagnostic_only": True,
            }
        )
    return {
        "variant_name": variant,
        "fill_count": len(selected),
        "settled_pnl": sum(float(row["settlement_pnl"]) for row in selected),
        "cost_basis": sum(float(row["cost_basis"]) for row in selected),
        "roi": _safe_roi(selected),
        "win_rate": _win_rate(selected),
        "pnl_by_side": _pnl_distribution(selected, "side"),
        "pnl_by_action": _pnl_distribution(selected, "action"),
        "pnl_by_regime": _pnl_distribution(selected, "market_regime"),
        "pnl_by_price_bucket": _pnl_distribution(selected, "price_bucket"),
        "pnl_by_time_window": _pnl_distribution(selected, "time_window_bucket"),
        "action_distribution": _count_distribution(selected, "action"),
        "family_distribution": _count_distribution(selected, "family"),
        "side_distribution": _count_distribution(selected, "side"),
        "regime_distribution": _count_distribution(selected, "market_regime"),
        "price_bucket_distribution": _count_distribution(selected, "price_bucket"),
        "time_window_distribution": _count_distribution(selected, "time_window_bucket"),
        "max_drawdown": _max_drawdown(
            [
                float(row["settlement_pnl"])
                for row in sorted(
                    selected,
                    key=lambda row: tuple(row["chronological_sort_key"]),
                )
            ]
        ),
        "rejected_reason_counts": dict(sorted(rejected_reasons.items())),
        "decision_rows": decision_rows,
        "diagnostic_only": True,
        "uses_outcome_for_policy_selection": False,
        "uses_outcome_for_offline_evaluation": True,
    }


def _hts_regime_variant_allows_row(
    row: Mapping[str, Any],
    variant: str,
) -> tuple[bool, list[str]]:
    action = str(row["action"])
    family = str(row["family"])
    if variant == "baseline_all":
        return True, []
    if variant == "side_blind_hts":
        if family == "HOLD_TO_SETTLEMENT":
            return True, []
        return False, ["not_hold_to_settlement"]
    if variant == "regime_aware_up_down_hts":
        return _allow_hts_when_side_regime_confirmed(row)
    if variant == "up_hts_only_when_up_regime_confirmed":
        if action != "BUY_UP_HOLD_TO_SETTLEMENT":
            return False, ["not_buy_up_hold_to_settlement"]
        return _allow_hts_when_side_regime_confirmed(row)
    if variant == "down_hts_only_when_down_regime_confirmed":
        if action != "BUY_DOWN_HOLD_TO_SETTLEMENT":
            return False, ["not_buy_down_hold_to_settlement"]
        return _allow_hts_when_side_regime_confirmed(row)
    if variant == "hts_allowed_only_when_regime_and_price_bucket_agree":
        allowed, reasons = _allow_hts_when_side_regime_confirmed(row)
        if not _hts_entry_price_quality_passed(row):
            reasons.append("hts_price_bucket_not_quality_candidate")
        return allowed and "hts_price_bucket_not_quality_candidate" not in reasons, reasons
    if variant == "hts_to_sbc_when_late_or_uncertain":
        if family == "SELL_BEFORE_CLOSE":
            return True, []
        allowed, reasons = _allow_hts_when_side_regime_confirmed(row)
        if row.get("time_window_bucket") in {
            "sell_before_close_only_window",
            "final_no_trade_window",
        }:
            reasons.append("hts_time_window_late_prefer_sbc")
        if not allowed:
            reasons.append("hts_regime_uncertain_prefer_sbc")
        return allowed and not reasons, reasons
    raise ValueError(f"unsupported HTS regime risk variant: {variant}")


def _allow_hts_when_side_regime_confirmed(
    row: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    if row.get("family") != "HOLD_TO_SETTLEMENT":
        return False, ["not_hold_to_settlement"]
    side = str(row.get("side") or "")
    if side == "UP" and bool(row.get("up_regime_confirmed")):
        return True, []
    if side == "DOWN" and bool(row.get("down_regime_confirmed")):
        return True, []
    return False, [f"{side.lower() or 'unknown'}_hts_regime_not_confirmed"]


def _hts_entry_price_quality_passed(row: Mapping[str, Any]) -> bool:
    price = row.get("entry_price")
    if price is None:
        return False
    return 0.50 <= float(price) <= 0.90


def _variant_allows_row(row: dict[str, Any], variant: str) -> tuple[bool, list[str]]:
    action = str(row["action"])
    family = str(row["family"])
    horizon = str(row["horizon"])
    price = row["entry_price"]
    if variant == "all_executed_baseline":
        return True, []
    if variant == "price_070_090_only":
        if _price_in_070_090(price):
            return True, []
        return False, [_price_rejection_reason(price)]
    if variant == "exclude_buy_up_hts":
        if action == "BUY_UP_HOLD_TO_SETTLEMENT":
            return False, ["excluded_buy_up_hold_to_settlement"]
        return True, []
    if variant == "sell_before_close_only":
        if family == "SELL_BEFORE_CLOSE":
            return True, []
        return False, ["not_sell_before_close"]
    if variant == "buy_down_hts_only":
        if action == "BUY_DOWN_HOLD_TO_SETTLEMENT":
            return True, []
        return False, ["not_buy_down_hold_to_settlement"]
    if variant == "five_min_only":
        if horizon == "5m":
            return True, []
        return False, ["not_5m_horizon"]
    if variant == "fifteen_min_only":
        if horizon == "15m":
            return True, []
        return False, ["not_15m_horizon"]
    if variant == "bucket_aware_v1_conservative":
        reasons = []
        if not _price_in_070_090(price):
            reasons.append("bucket_aware_conservative_price_not_070_090")
        if action == "BUY_UP_HOLD_TO_SETTLEMENT":
            reasons.append("bucket_aware_conservative_excluded_buy_up_hts")
        if family != "SELL_BEFORE_CLOSE" and action != "BUY_DOWN_HOLD_TO_SETTLEMENT":
            reasons.append("bucket_aware_conservative_action_not_candidate")
        return not reasons, reasons
    if variant == "bucket_aware_v1_plus_sbc":
        reasons = []
        if action == "BUY_UP_HOLD_TO_SETTLEMENT":
            reasons.append("bucket_aware_plus_sbc_excluded_buy_up_hts")
        if family == "SELL_BEFORE_CLOSE":
            return not reasons, reasons
        if action != "BUY_DOWN_HOLD_TO_SETTLEMENT":
            reasons.append("bucket_aware_plus_sbc_action_not_candidate")
        if not _price_in_070_090(price):
            reasons.append("bucket_aware_plus_sbc_price_not_070_090_for_hts")
        return not reasons, reasons
    raise ValueError(f"unsupported policy replay variant: {variant}")


def _forward_shadow_variant_metrics(
    rows: list[dict[str, Any]],
    variant: str,
    *,
    entry_ev_threshold: float,
) -> dict[str, Any]:
    allowed_rows: list[dict[str, Any]] = []
    rejected_reasons: Counter[str] = Counter()
    decision_rows: list[dict[str, Any]] = []
    for row in rows:
        allowed, reasons = _forward_shadow_variant_allows_row(
            row,
            variant,
            entry_ev_threshold=entry_ev_threshold,
        )
        if allowed:
            allowed_rows.append(row)
        else:
            rejected_reasons.update(reasons)
        decision_rows.append(_forward_shadow_decision_row(row, variant, allowed, reasons))

    entry_count = sum(1 for row in allowed_rows if _is_entry_action(row["action"]))
    exit_count = sum(1 for row in allowed_rows if _is_exit_action(row["action"]))
    hold_count = sum(1 for row in allowed_rows if _is_hold_action(row["action"]))
    no_trade_count = sum(1 for row in allowed_rows if row["action"] == "NO_TRADE")
    return {
        "variant_name": variant,
        "allowed_decision_count": len(allowed_rows),
        "rejected_decision_count": len(rows) - len(allowed_rows),
        "entry_count": entry_count,
        "exit_count": exit_count,
        "hold_count": hold_count,
        "no_trade_count": no_trade_count,
        "action_distribution": _count_distribution(allowed_rows, "action"),
        "family_distribution": _count_distribution(allowed_rows, "family"),
        "side_distribution": _count_distribution(allowed_rows, "side"),
        "time_window_distribution": _count_distribution(rows, "time_window_bucket"),
        "allowed_time_window_distribution": _count_distribution(
            allowed_rows,
            "time_window_bucket",
        ),
        "rejected_reason_counts": dict(sorted(rejected_reasons.items())),
        "decision_rows": decision_rows,
        "uses_settlement_pnl_or_outcome_labels": False,
        "uses_oracle_actions_or_future_returns": False,
        "market_implied_probability_used_as_calibrated_ev_source": False,
        "diagnostic_only": True,
    }


def _blocked_forward_shadow_variant_metrics(
    variant: str,
    rows: list[dict[str, Any]],
    reason_codes: list[str],
) -> dict[str, Any]:
    return {
        "variant_name": variant,
        "allowed_decision_count": 0,
        "rejected_decision_count": len(rows),
        "entry_count": 0,
        "exit_count": 0,
        "hold_count": 0,
        "no_trade_count": 0,
        "action_distribution": {},
        "family_distribution": {},
        "side_distribution": {},
        "time_window_distribution": {},
        "allowed_time_window_distribution": {},
        "rejected_reason_counts": dict(Counter(reason_codes)),
        "decision_rows": [],
        "uses_settlement_pnl_or_outcome_labels": False,
        "uses_oracle_actions_or_future_returns": False,
        "market_implied_probability_used_as_calibrated_ev_source": False,
        "diagnostic_only": True,
    }


def _guard_intersection_variant_metrics(
    rows: list[dict[str, Any]],
    variant: str,
    *,
    entry_ev_threshold: float,
) -> dict[str, Any]:
    candidate_rows: list[dict[str, Any]] = []
    guard_passed_rows: list[dict[str, Any]] = []
    guard_blocked_rows: list[dict[str, Any]] = []
    guard_unknown_rows: list[dict[str, Any]] = []
    guard_reasons: Counter[str] = Counter()
    intersection_rows: list[dict[str, Any]] = []
    for row in rows:
        policy_allowed, policy_reasons = _forward_shadow_variant_allows_row(
            row,
            variant,
            entry_ev_threshold=entry_ev_threshold,
        )
        if not policy_allowed:
            continue
        candidate_rows.append(row)
        guard_status, guard_status_reasons = _execution_guard_status(row)
        guard_reasons.update(guard_status_reasons)
        if guard_status == "guard_passed":
            guard_passed_rows.append(row)
        elif guard_status == "guard_unknown":
            guard_unknown_rows.append(row)
        else:
            guard_blocked_rows.append(row)
        intersection_rows.append(
            _guard_intersection_decision_row(
                row,
                variant,
                policy_reasons=policy_reasons,
                guard_status=guard_status,
                guard_reason_codes=guard_status_reasons,
            )
        )
    executable_count = len(guard_passed_rows)
    return {
        "variant_name": variant,
        "policy_candidate_count": len(candidate_rows),
        "guard_passed_candidate_count": len(guard_passed_rows),
        "guard_blocked_candidate_count": len(guard_blocked_rows),
        "guard_unknown_candidate_count": len(guard_unknown_rows),
        "candidate_but_not_executable_count": len(guard_blocked_rows)
        + len(guard_unknown_rows),
        "executable_shadow_count": executable_count,
        "executable_shadow_entry_count": sum(
            1 for row in guard_passed_rows if _is_entry_action(row["action"])
        ),
        "executable_shadow_exit_count": sum(
            1 for row in guard_passed_rows if _is_exit_action(row["action"])
        ),
        "executable_shadow_hold_count": sum(
            1 for row in guard_passed_rows if _is_hold_action(row["action"])
        ),
        "executable_shadow_no_trade_count": sum(
            1 for row in guard_passed_rows if row["action"] == "NO_TRADE"
        ),
        "guard_blocking_reason_distribution": dict(sorted(guard_reasons.items())),
        "time_window_distribution_for_guard_passed": _count_distribution(
            guard_passed_rows,
            "time_window_bucket",
        ),
        "time_window_distribution_for_guard_blocked": _count_distribution(
            guard_blocked_rows,
            "time_window_bucket",
        ),
        "time_window_distribution_for_guard_unknown": _count_distribution(
            guard_unknown_rows,
            "time_window_bucket",
        ),
        "intersection_rows": intersection_rows,
        "diagnostic_only": True,
        "uses_settlement_pnl_or_outcome_labels": False,
        "uses_oracle_actions_or_future_returns": False,
        "market_implied_probability_used_as_calibrated_ev_source": False,
    }


def _blocked_guard_intersection_variant_metrics(
    variant: str,
    rows: list[dict[str, Any]],
    reason_codes: list[str],
) -> dict[str, Any]:
    return {
        "variant_name": variant,
        "policy_candidate_count": 0,
        "guard_passed_candidate_count": 0,
        "guard_blocked_candidate_count": 0,
        "guard_unknown_candidate_count": 0,
        "candidate_but_not_executable_count": 0,
        "executable_shadow_count": 0,
        "executable_shadow_entry_count": 0,
        "executable_shadow_exit_count": 0,
        "executable_shadow_hold_count": 0,
        "executable_shadow_no_trade_count": 0,
        "guard_blocking_reason_distribution": dict(Counter(reason_codes)),
        "time_window_distribution_for_guard_passed": {},
        "time_window_distribution_for_guard_blocked": {},
        "time_window_distribution_for_guard_unknown": {},
        "intersection_rows": [],
        "diagnostic_only": True,
        "uses_settlement_pnl_or_outcome_labels": False,
        "uses_oracle_actions_or_future_returns": False,
        "market_implied_probability_used_as_calibrated_ev_source": False,
    }


def _execution_guard_status(row: dict[str, Any]) -> tuple[str, list[str]]:
    if row["action"] == "NO_TRADE":
        return "guard_blocked", ["selected_action_no_trade"]
    missing_fields = list(row["missing_execution_guard_decision_fields"])
    if missing_fields:
        return (
            "guard_unknown",
            [
                "missing_execution_guard_decision_fields",
                *[f"missing_guard_field:{field}" for field in missing_fields],
            ],
        )
    if not row["execution_guard_safety_flags_blocked"]:
        return "guard_blocked", ["execution_guard_safety_flags_not_blocked"]
    blocking_codes = list(row["execution_guard_blocking_reason_codes"])
    if row["order_allowed"] is not True:
        return (
        "guard_blocked",
        blocking_codes or ["execution_guard_order_not_allowed"],
    )
    if blocking_codes:
        return "guard_blocked", blocking_codes
    return "guard_passed", []


_HTS_TIME_WINDOW_REMAP_IGNORABLE_GUARD_CODES = {
    "execution_hts_downgraded_to_same_side_sbc",
    "execution_hts_guard_failed",
    "execution_time_to_close_unsafe",
}


def _is_hts_time_window_blocked_row(row: dict[str, Any]) -> bool:
    if row["action"] not in {
        "BUY_UP_HOLD_TO_SETTLEMENT",
        "BUY_DOWN_HOLD_TO_SETTLEMENT",
    }:
        return False
    if row["time_window_bucket"] != "sell_before_close_only_window":
        return False
    guard_status, guard_reasons = _execution_guard_status(row)
    if guard_status == "guard_passed":
        return False
    reason_set = set(guard_reasons)
    return bool(
        reason_set
        & {
            "execution_time_to_close_unsafe",
            "execution_hts_downgraded_to_same_side_sbc",
            "execution_hts_guard_failed",
        }
    )


def _hts_time_window_remap_row(
    row: dict[str, Any],
    *,
    entry_ev_threshold: float,
) -> dict[str, Any]:
    proposed_action = _same_side_sbc_action(row)
    original_guard_status, original_guard_reasons = _execution_guard_status(row)
    reason_codes: list[str] = []
    if proposed_action is None or not _same_side_sbc_action_available(
        row,
        proposed_action,
    ):
        reason_codes.append("same_side_sbc_alternative_missing")
    if not row["calibrated_ev_available"]:
        reason_codes.append("same_side_sbc_calibrated_ev_missing")
        reason_codes.extend(row["ev_mapping_blocking_reason_codes"])
    elif float(row["calibrated_ev"]) < entry_ev_threshold:
        reason_codes.append("same_side_sbc_calibrated_ev_below_threshold")
    if row["missing_execution_guard_decision_fields"]:
        reason_codes.append("missing_execution_guard_decision_fields")
        reason_codes.extend(
            f"missing_guard_field:{field}"
            for field in row["missing_execution_guard_decision_fields"]
        )
    if not row["execution_guard_safety_flags_blocked"]:
        reason_codes.append("execution_guard_safety_flags_not_blocked")
    if row["time_window_bucket"] == "final_no_trade_window":
        reason_codes.append("same_side_sbc_time_to_close_final_no_trade_window")
    elif row["time_window_bucket"] != "sell_before_close_only_window":
        reason_codes.append("same_side_sbc_time_window_not_sbc_only")

    non_remappable_guard_codes = [
        code
        for code in original_guard_reasons
        if code not in _HTS_TIME_WINDOW_REMAP_IGNORABLE_GUARD_CODES
    ]
    reason_codes.extend(non_remappable_guard_codes)
    reason_codes = sorted(set(reason_codes))
    candidate = (
        proposed_action is not None
        and "same_side_sbc_alternative_missing" not in reason_codes
        and row["calibrated_ev_available"]
        and float(row["calibrated_ev"]) >= entry_ev_threshold
    )
    remap_guard_passed = candidate and not reason_codes
    if remap_guard_passed:
        reason_codes = ["diagnostic_remap_guard_passed"]
    elif candidate and not reason_codes:
        reason_codes = ["diagnostic_remap_guard_blocked_unknown"]
    return {
        "row_index": row["row_index"],
        "market_id": row["market_id"],
        "decision_ts": row["decision_ts"],
        "selected_action": row["action"],
        "selected_side": row["side"],
        "selected_family": row["family"],
        "time_window_bucket": row["time_window_bucket"],
        "time_to_close_seconds": row["time_to_close_seconds"],
        "required_min_time_to_close_seconds": row[
            "required_min_time_to_close_seconds"
        ],
        "proposed_same_side_sbc_action": proposed_action,
        "proposed_same_side_sbc_side": row["side"],
        "proposed_same_side_sbc_family": (
            "SELL_BEFORE_CLOSE" if proposed_action is not None else None
        ),
        "same_side_sbc_alternative_available": (
            proposed_action is not None
            and _same_side_sbc_action_available(row, proposed_action)
        ),
        "same_side_sbc_calibrated_ev_available": row["calibrated_ev_available"],
        "calibrated_ev": row["calibrated_ev"],
        "entry_ev_threshold": entry_ev_threshold,
        "diagnostic_remap_candidate": candidate,
        "diagnostic_remap_guard_passed": remap_guard_passed,
        "original_guard_status": original_guard_status,
        "original_guard_reason_codes": original_guard_reasons,
        "remap_reason_codes": reason_codes,
        "non_remappable_guard_reason_codes": sorted(set(non_remappable_guard_codes)),
        "available_actions": row["available_actions"],
        "execution_price": row["execution_price"],
        "spread_bps": row["spread_bps"],
        "book_staleness_ms": row["book_staleness_ms"],
        "queue_fill_proxy": row["queue_fill_proxy"],
        "execution_guarded_action": row["execution_guarded_action"],
        "execution_guarded_side": row["execution_guarded_side"],
        "source_scores_mutated": False,
        "o_score_mutated": False,
        "paper_live_unlock_changed": False,
        "diagnostic_only": True,
    }


def _same_side_sbc_action(row: Mapping[str, Any]) -> str | None:
    if row.get("side") == "UP" and row.get("action") == "BUY_UP_HOLD_TO_SETTLEMENT":
        return "BUY_UP_SELL_BEFORE_CLOSE"
    if row.get("side") == "DOWN" and row.get("action") == "BUY_DOWN_HOLD_TO_SETTLEMENT":
        return "BUY_DOWN_SELL_BEFORE_CLOSE"
    return None


def _same_side_sbc_action_available(row: Mapping[str, Any], action: str) -> bool:
    available_actions = row.get("available_actions")
    if available_actions is None:
        return True
    return action in set(available_actions)


def _guard_intersection_decision_row(
    row: dict[str, Any],
    variant: str,
    *,
    policy_reasons: list[str],
    guard_status: str,
    guard_reason_codes: list[str],
) -> dict[str, Any]:
    return {
        "variant_name": variant,
        "row_index": row["row_index"],
        "market_id": row["market_id"],
        "decision_ts": row["decision_ts"],
        "selected_action": row["action"],
        "selected_side": row["side"],
        "family": row["family"],
        "time_window_bucket": row["time_window_bucket"],
        "policy_candidate": True,
        "policy_reason_codes": policy_reasons,
        "guard_status": guard_status,
        "guard_reason_codes": guard_reason_codes,
        "order_allowed": row["order_allowed"],
        "execution_guarded_action": row["execution_guarded_action"],
        "execution_guarded_side": row["execution_guarded_side"],
        "execution_blocking_reason_codes": row[
            "execution_guard_blocking_reason_codes"
        ],
        "time_to_close_gate_passed": row["time_to_close_gate_passed"],
        "time_to_close_seconds": row["time_to_close_seconds"],
        "required_min_time_to_close_seconds": row[
            "required_min_time_to_close_seconds"
        ],
        "spread_bps": row["spread_bps"],
        "book_staleness_ms": row["book_staleness_ms"],
        "queue_fill_proxy": row["queue_fill_proxy"],
        "paper_intent_id": row["paper_intent_id"],
        "paper_fill_id": row["paper_fill_id"],
        "executable_shadow": guard_status == "guard_passed",
        "diagnostic_only": True,
    }


def _forward_shadow_variant_allows_row(
    row: dict[str, Any],
    variant: str,
    *,
    entry_ev_threshold: float,
) -> tuple[bool, list[str]]:
    if row["action"] == "NO_TRADE":
        return False, ["selected_action_no_trade"]
    if variant == "baseline_current_guard":
        return _calibrated_ev_threshold_decision(
            row,
            entry_ev_threshold=entry_ev_threshold,
            missing_reason="baseline_current_guard_missing_calibrated_ev_source",
            threshold_reason="baseline_current_guard_ev_threshold_not_met",
        )
    if variant == "bucket_aware_v1_conservative":
        return _variant_allows_row(row, "bucket_aware_v1_conservative")
    if variant == "bucket_aware_v1_plus_sbc":
        return _variant_allows_row(row, "bucket_aware_v1_plus_sbc")
    if variant == "calibrated_ev_v2":
        return _calibrated_ev_threshold_decision(
            row,
            entry_ev_threshold=entry_ev_threshold,
            missing_reason="calibrated_ev_source_missing",
            threshold_reason="calibrated_ev_threshold_not_met",
        )
    if variant == "calibrated_ev_plus_bucket_v2":
        allowed, reasons = _calibrated_ev_threshold_decision(
            row,
            entry_ev_threshold=entry_ev_threshold,
            missing_reason="calibrated_ev_source_missing",
            threshold_reason="calibrated_ev_threshold_not_met",
        )
        bucket_allowed, bucket_reasons = _variant_allows_row(
            row,
            "bucket_aware_v1_plus_sbc",
        )
        reasons.extend(bucket_reasons)
        return allowed and bucket_allowed, reasons
    raise ValueError(f"unsupported forward shadow variant: {variant}")


def _calibrated_ev_threshold_decision(
    row: dict[str, Any],
    *,
    entry_ev_threshold: float,
    missing_reason: str,
    threshold_reason: str,
) -> tuple[bool, list[str]]:
    if not row["calibrated_ev_available"]:
        return False, [missing_reason, *row["ev_mapping_blocking_reason_codes"]]
    if float(row["calibrated_ev"]) < entry_ev_threshold:
        return False, [threshold_reason]
    return True, []


def _forward_shadow_decision_row(
    row: dict[str, Any],
    variant: str,
    allowed: bool,
    reason_codes: list[str],
) -> dict[str, Any]:
    return {
        "variant_name": variant,
        "row_index": row["row_index"],
        "market_id": row["market_id"],
        "decision_ts": row["decision_ts"],
        "selected_action": row["action"],
        "selected_side": row["side"],
        "family": row["family"],
        "time_window_bucket": row["time_window_bucket"],
        "execution_price": row["execution_price"],
        "p_market_implied": row["p_market_implied"],
        "p_model_fair_value": row["p_model_fair_value"],
        "calibrated_action_expected_net_return": row[
            "calibrated_action_expected_net_return"
        ],
        "canonical_o_action_score": row["canonical_o_action_score"],
        "calibrated_ev": row["calibrated_ev"],
        "ev_source": row["ev_source"],
        "ev_mapping_status": row["ev_mapping_status"],
        "shadow_decision_allowed": allowed,
        "shadow_decision_type": _decision_type(row["action"]) if allowed else "REJECTED",
        "rejection_reason_codes": reason_codes,
        "market_implied_probability_used_as_calibrated_ev_source": False,
        "diagnostic_only": True,
    }


def _forward_shadow_variant_definitions() -> dict[str, str]:
    return {
        "baseline_current_guard": (
            "current v2 entry EV threshold contract, fail-closed unless a calibrated "
            "EV source is available"
        ),
        "bucket_aware_v1_conservative": (
            "diagnostic bucket policy from #166: price 0.70-0.90, exclude "
            "BUY_UP_HOLD_TO_SETTLEMENT, and require SELL_BEFORE_CLOSE to pass "
            "the same price bucket"
        ),
        "bucket_aware_v1_plus_sbc": (
            "diagnostic bucket policy from #166: allow SELL_BEFORE_CLOSE regardless "
            "of price bucket, allow BUY_DOWN_HOLD_TO_SETTLEMENT only in 0.70-0.90, "
            "exclude BUY_UP_HOLD_TO_SETTLEMENT"
        ),
        "calibrated_ev_v2": (
            "allow rows only when calibrated_action_expected_net_return or "
            "calibrated fair value minus executable price minus cost clears the "
            "entry EV threshold"
        ),
        "calibrated_ev_plus_bucket_v2": (
            "calibrated_ev_v2 plus bucket_aware_v1_plus_sbc action/price filters"
        ),
    }


def _forward_shadow_time_window_behavior(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "time_window_distribution": _count_distribution(rows, "time_window_bucket"),
        "entry_time_window_distribution": _count_distribution(
            [row for row in rows if _is_entry_action(row["action"])],
            "time_window_bucket",
        ),
        "exit_time_window_distribution": _count_distribution(
            [row for row in rows if _is_exit_action(row["action"])],
            "time_window_bucket",
        ),
        "hold_time_window_distribution": _count_distribution(
            [row for row in rows if _is_hold_action(row["action"])],
            "time_window_bucket",
        ),
    }


def _ev_mapping_contract_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_index": row["row_index"],
        "market_id": row["market_id"],
        "decision_ts": row["decision_ts"],
        "selected_action": row["action"],
        "selected_side": row["side"],
        "p_market_implied": row["p_market_implied"],
        "p_model_fair_value": row["p_model_fair_value"],
        "calibrated_action_expected_net_return": row[
            "calibrated_action_expected_net_return"
        ],
        "canonical_o_action_score": row["canonical_o_action_score"],
        "execution_price": row["execution_price"],
        "execution_cost": row["execution_cost"],
        "ev_source": row["ev_source"],
        "ev_source_provenance": row["ev_source_provenance"],
        "ev_mapping_status": row["ev_mapping_status"],
        "ev_mapping_blocking_reason_codes": row["ev_mapping_blocking_reason_codes"],
        "calibrated_ev_available": row["calibrated_ev_available"],
        "calibrated_ev": row["calibrated_ev"],
        "market_implied_probability_used_for_ev": False,
    }


def _calibrated_ev_source_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_index": row["row_index"],
        "market_id": row["market_id"],
        "decision_ts": row["decision_ts"],
        "selected_action": row["action"],
        "selected_side": row["side"],
        "family": row["family"],
        "canonical_o_action_score": row["canonical_o_action_score"],
        "canonical_o_raw_score": row["canonical_o_raw_score"],
        "execution_price": row["execution_price"],
        "executable_exit_bid_proxy": row["executable_exit_bid_proxy"],
        "spread_bps": row["spread_bps"],
        "queue_fill_proxy": row["queue_fill_proxy"],
        "book_staleness_ms": row["book_staleness_ms"],
        "time_to_close_seconds": row["time_to_close_seconds"],
        "calibrated_action_expected_net_return": row[
            "calibrated_action_expected_net_return"
        ],
        "calibrated_ev_source": row["calibrated_ev_source"],
        "calibrated_ev_source_provenance": row["calibrated_ev_source_provenance"],
        "calibrated_ev_available": row[
            "calibrated_action_expected_net_return_available"
        ],
        "calibrated_ev_blocking_reason_codes": row[
            "calibrated_ev_blocking_reason_codes"
        ],
        "market_implied_probability_used_for_ev": False,
    }


def _signal_to_ev_diagnostic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    model_fields = sorted(
        {
            field
            for row in rows
            for field in row["raw_fields"]
            if field.lower()
            in {
                "p_model_fair_value",
                "model_p_up",
                "calibrated_p_up",
                "model_probability",
                "predicted_probability",
                "fair_value_probability",
                "action_expected_net_return",
                "calibrated_action_score",
            }
        }
    )
    implied_fields = sorted(
        {
            row["entry_price_source_field"]
            for row in rows
            if row.get("entry_price_source_field")
        }
    )
    reason_codes = [
        "p_up_probability_provenance_not_confirmed_calibrated_model_fair_value",
        "market_implied_probability_collapses_ev_to_spread_minus_cost",
        "settlement_csv_is_outcome_evaluation_not_decision_time_model_training_input",
    ]
    return {
        "current_ev_formula": "entry_ev = p_side - ask - cost",
        "p_market_implied_source_fields": implied_fields,
        "p_model_fair_value_source_fields_present": bool(model_fields),
        "p_model_fair_value_candidate_fields": model_fields,
        "current_p_up_should_not_be_used_as_ev_fair_value_without_provenance": True,
        "ev_mapping_status": "blocked_requires_calibrated_model_fair_value",
        "ev_mapping_blocking_reason_codes": reason_codes,
        "recommended_ev_source": (
            "calibrated_model_fair_value_probability_or_action_expected_net_return"
        ),
        "code_bug_indicated": False,
        "design_issue_indicated": True,
        "diagnosis": (
            "The no-action behavior is expected when p_side is market-implied "
            "rather than calibrated fair value; EV becomes approximately "
            "negative spread plus cost."
        ),
    }


def _recommended_execution_policy(
    rows: list[dict[str, Any]],
    variant_reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    sbc_metrics = variant_reports["sell_before_close_only"]
    sbc_count = sbc_metrics["row_count"]
    sbc_positive = float(sbc_metrics["settlement_pnl"]) > 0.0
    return {
        "policy_name": EXECUTION_LAYER_V2_RECOMMENDED_EXECUTION_POLICY_NAME,
        "derived_from_settlement_csv_diagnostics_only": True,
        "uses_validation_labels_for_threshold_tuning": False,
        "do_not_relax_execution_guard_thresholds": True,
        "candidate_variant_name": "bucket_aware_v1_plus_sbc",
        "candidate_variant_metrics": variant_reports["bucket_aware_v1_plus_sbc"],
        "comparison_variant_metrics": {
            "bucket_aware_v1_conservative": variant_reports[
                "bucket_aware_v1_conservative"
            ],
            "bucket_aware_v1_plus_sbc": variant_reports["bucket_aware_v1_plus_sbc"],
        },
        "small_sample_warnings": (
            ["sell_before_close_small_sample"] if 0 < sbc_count < 30 else []
        ),
        "sell_before_close_summary": sbc_metrics,
        "sell_before_close_positive_in_csv": sbc_positive,
        "sell_before_close_diagnostic_interpretation": (
            "SELL_BEFORE_CLOSE is positive in this CSV but remains small-sample."
            if sbc_positive and 0 < sbc_count < 30
            else "SELL_BEFORE_CLOSE support is not sufficient for promotion evidence."
        ),
        "rules": [
            "Do not use market-implied p_up as EV fair value without calibrated provenance.",
            "Avoid BUY_UP_HOLD_TO_SETTLEMENT unless strong calibrated edge exists.",
            "Prefer entry price bucket 0.70-0.90 for BUY_DOWN_HOLD_TO_SETTLEMENT when calibrated edge exists.",
            "Keep SELL_BEFORE_CLOSE as a candidate even below 0.70, but mark small-sample until support grows.",
            "Keep BUY_DOWN_HOLD_TO_SETTLEMENT as a candidate.",
            "Avoid price >0.90 unless calibrated edge is strong.",
            "Avoid price 0.60-0.70 by default.",
            "Do not relax execution guard thresholds.",
        ],
        "expected_follow_up": (
            "Use calibrated O action expected return or calibrated fair-value "
            "probability for EV mapping, then replay against future holdout."
        ),
        "row_count": len(rows),
    }


def _small_sample_warnings(
    variant_reports: dict[str, dict[str, Any]],
) -> list[str]:
    sbc_count = variant_reports["sell_before_close_only"]["row_count"]
    if 0 < sbc_count < 30:
        return ["sell_before_close_small_sample"]
    return []


def _replay_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return bool(isinstance(value, dict) and not value)


def _merge_replay_sources_preserve_non_null(
    *sources: Mapping[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for source in sources:
        for key, value in dict(source).items():
            if _replay_missing_value(value):
                continue
            merged[key] = value
    return merged


def _hts_regime_canonical_feature_presence(
    row: Mapping[str, Any],
) -> dict[str, bool]:
    return {
        field: row.get(field) is not None
        for field in HTS_REGIME_CANONICAL_FEATURE_FIELDS
    }


def _load_hts_regime_replay_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        settlement_path = path / "settlement_pnl_rows.jsonl"
        fills_path = path / "one_hour_paper_fill_log.jsonl"
        intents_path = path / "one_hour_paper_intent_log.jsonl"
        if not settlement_path.exists():
            raise ValueError(
                "HTS regime replay directory must include settlement_pnl_rows.jsonl"
            )
        settlement_rows = _read_jsonl_dicts(settlement_path)
        fill_by_intent = {
            str(row.get("paper_fresh_order_intent_id")): row
            for row in _read_jsonl_dicts(fills_path)
            if row.get("paper_fresh_order_intent_id")
        }
        intent_by_id = {
            str(row.get("paper_fresh_order_intent_id")): row
            for row in _read_jsonl_dicts(intents_path)
            if row.get("paper_fresh_order_intent_id")
        }
        trace_by_intent = _load_hts_regime_trace_rows_by_intent(path)
        merged_rows = []
        for row in settlement_rows:
            intent_id = str(row.get("paper_fresh_order_intent_id") or "")
            pre_trace_merged = _merge_replay_sources_preserve_non_null(
                intent_by_id.get(intent_id, {}),
                fill_by_intent.get(intent_id, {}),
                row,
            )
            trace_row = trace_by_intent.get(intent_id, {})
            merged = _merge_replay_sources_preserve_non_null(
                trace_row,
                intent_by_id.get(intent_id, {}),
                fill_by_intent.get(intent_id, {}),
                row,
            )
            merged["_pre_trace_merge_regime_feature_presence"] = (
                _hts_regime_canonical_feature_presence(pre_trace_merged)
            )
            merged["_trace_regime_feature_presence"] = (
                _hts_regime_canonical_feature_presence(trace_row)
            )
            merged_rows.append(merged)
        return _attach_hts_regime_sequence_context(
            [
                _normalize_hts_regime_replay_row(row, index)
                for index, row in enumerate(merged_rows)
            ]
        )
    if path.suffix.lower() == ".csv":
        rows = _load_csv_rows(path)
        return _attach_hts_regime_sequence_context(
            [
                _normalize_hts_regime_replay_row(row, index)
                for index, row in enumerate(rows)
            ]
        )
    if path.suffix.lower() == ".jsonl":
        rows = _read_jsonl_dicts(path)
        return _attach_hts_regime_sequence_context(
            [
                _normalize_hts_regime_replay_row(row, index)
                for index, row in enumerate(rows)
            ]
        )
    raise ValueError("HTS regime replay input must be a run directory, CSV, or JSONL")


def _load_hts_regime_trace_rows_by_intent(path: Path) -> dict[str, dict[str, Any]]:
    trace_paths = [
        path / "o_v8_paper_fresh_signal_trace.json",
        path / "incremental_fresh_loop" / "o_v8_paper_fresh_signal_trace.json",
    ]
    trace_paths.extend(
        sorted(
            path.glob(
                "incremental_fresh_loop_cycles/*/o_v8_paper_fresh_signal_trace.json"
            )
        )
    )
    rows_by_intent: dict[str, dict[str, Any]] = {}
    for trace_path in trace_paths:
        if not trace_path.exists():
            continue
        try:
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, Mapping):
            continue
        for row in payload.get("trace_rows") or []:
            if not isinstance(row, Mapping):
                continue
            intent_id = str(row.get("paper_intent_id") or "")
            if not intent_id:
                continue
            rows_by_intent.setdefault(intent_id, dict(row))
    return rows_by_intent


def _normalize_hts_regime_replay_row(
    row: Mapping[str, Any],
    index: int,
) -> dict[str, Any]:
    action = _canonical_action(
        _first_text(
            row,
            (
                "execution_guarded_action",
                "action",
                "selected_action",
                "source_selected_action",
            ),
            default="UNKNOWN",
        )
    )
    side = _infer_side(dict(row), action)
    family = _infer_family(dict(row), action)
    entry_price, entry_price_field = _first_float_with_field(
        row,
        (
            "paper_fill_price",
            "entry_price",
            "execution_price",
            "paper_limit_price",
            "entry_ask",
            "ask",
        ),
    )
    filled_size = _first_float(row, ("filled_size", "size", "shares"), default=0.0)
    execution_cost = _first_float(row, ("total_execution_cost", "execution_cost"), default=0.0)
    cost_basis = _first_float(
        row,
        ("cost_basis", "paper_notional", "notional"),
        default=0.0,
    )
    if cost_basis <= 0.0 and entry_price is not None and filled_size is not None:
        cost_basis = (entry_price * filled_size) + (execution_cost or 0.0)
    decision_ts_raw = _first_text(row, ("decision_ts", "ts", "timestamp"), default=str(index))
    decision_ts_numeric = _parse_sort_number(decision_ts_raw)
    numeric_iteration = _first_sort_number(
        row,
        ("iteration", "round_iteration", "loop_iteration", "cycle_index"),
        default=MISSING_SORT_NUMBER,
    )
    intent_id = _first_text(
        row,
        ("paper_fresh_order_intent_id", "intent_id", "order_intent_id", "signal_id"),
        default="",
    )
    p_up = _first_float(row, ("p_up", "p_market_implied_up", "probability_up"), default=None)
    p_down = _first_float(
        row,
        ("p_down", "p_market_implied_down", "probability_down"),
        default=None,
    )
    btc_momentum = _first_float(
        row,
        (
            "btc_momentum",
            "recent_btc_momentum_120s",
            "recent_btc_momentum_60s",
            "recent_btc_momentum_30s",
            "recent_reference_price_momentum_120s",
            "recent_reference_price_momentum_60s",
            "recent_reference_price_momentum_30s",
        ),
        default=None,
    )
    reference_distance = _first_float(
        row,
        (
            "reference_price_to_beat_distance_at_decision",
            "price_to_beat_distance",
            "reference_distance",
        ),
        default=None,
    )
    reference_price_to_beat = _first_float(
        row,
        (
            "reference_price_to_beat_at_decision",
            "reference_price_to_beat",
            "price_to_beat",
        ),
        default=None,
    )
    time_to_close = _first_float(row, ("time_to_close_seconds",), default=None)
    time_since_start = _first_float(
        row,
        (
            "time_since_market_start_seconds",
            "time_since_start_seconds",
            "elapsed_since_market_start_seconds",
        ),
        default=None,
    )
    spread_bps = _first_float(row, ("spread_bps",), default=None)
    book_staleness_ms = _first_float(row, ("book_staleness_ms",), default=None)
    queue_fill_proxy = _first_float(row, ("queue_fill_proxy",), default=None)
    score_margin = _first_float(
        row,
        (
            "best_action_margin",
            "action_score_margin",
            "top_action_margin",
            "score_margin",
        ),
        default=None,
    )
    side_score_margin = _first_float(
        row,
        (
            "side_specific_action_score_margin",
            "selected_side_action_score_margin",
            "side_score_margin",
        ),
        default=None,
    )
    action_score = _first_float(
        row,
        ("source_model_score", "execution_guarded_score", "canonical_o_action_score"),
        default=None,
    )
    settlement_pnl = _first_float(
        row,
        (
            "settlement_pnl",
            "settlement_pnl_usdc",
            "total_polymarket_pnl",
            "pnl",
            "realized_pnl",
            "net_pnl",
        ),
        default=0.0,
    )
    resolved_outcome = _first_text(
        row,
        ("resolved_outcome", "winning_outcome"),
        default="UNKNOWN",
    ).upper()
    p_balance = (p_up - p_down) if p_up is not None and p_down is not None else None
    p_balance_regime = _p_up_down_balance_regime(p_balance)
    btc_regime = _btc_momentum_regime(btc_momentum)
    reference_bucket = _reference_distance_bucket(reference_distance)
    score_bucket = _score_margin_bucket(score_margin)
    regime_vote_summary = _hts_regime_vote_summary(
        p_balance_regime=p_balance_regime,
        btc_regime=btc_regime,
        reference_distance_bucket=reference_bucket,
        action_score_margin_bucket=score_bucket,
    )
    regime_provenance = dict(row.get("decision_time_regime_feature_provenance") or {})
    return {
        "row_index": index,
        "market_id": _first_text(row, ("market_id", "condition_id", "slug"), default=""),
        "decision_ts": decision_ts_raw,
        "decision_ts_numeric": decision_ts_numeric,
        "numeric_iteration": numeric_iteration,
        "intent_id": intent_id,
        "chronological_sort_key": [
            numeric_iteration,
            decision_ts_numeric,
            intent_id,
            index,
        ],
        "action": action,
        "family": family,
        "side": side,
        "horizon": _infer_horizon(dict(row)),
        "entry_price": entry_price,
        "entry_price_source_field": entry_price_field,
        "price_bucket": _price_bucket(entry_price),
        "cost_basis": cost_basis or 0.0,
        "settlement_pnl": settlement_pnl or 0.0,
        "settlement_status": _first_text(row, ("settlement_status",), default="settled"),
        "resolved_outcome": resolved_outcome,
        "p_up": p_up,
        "p_down": p_down,
        "p_up_down_balance": p_balance,
        "p_up_down_balance_regime": p_balance_regime,
        "btc_momentum": btc_momentum,
        "btc_momentum_regime": btc_regime,
        "reference_price_to_beat_at_decision": reference_price_to_beat,
        "reference_price_to_beat_distance_at_decision": reference_distance,
        "reference_distance_bucket": reference_bucket,
        "time_since_market_start_seconds": time_since_start,
        "time_since_market_start_bucket": _time_since_start_bucket(time_since_start),
        "time_to_close_seconds": time_to_close,
        "time_window_bucket": _time_window_bucket(time_to_close),
        "spread_bps": spread_bps,
        "spread_bucket": _spread_bucket(spread_bps),
        "book_staleness_ms": book_staleness_ms,
        "staleness_bucket": _staleness_bucket(book_staleness_ms),
        "queue_fill_proxy": queue_fill_proxy,
        "queue_bucket": _queue_bucket(queue_fill_proxy),
        "action_score_margin": score_margin,
        "action_score_margin_bucket": score_bucket,
        "side_specific_action_score_margin": side_score_margin,
        "side_specific_action_score_margin_bucket": _score_margin_bucket(
            side_score_margin
        ),
        "action_score": action_score,
        "market_regime": regime_vote_summary["market_regime"],
        "regime_feature_vote_summary": regime_vote_summary,
        "decision_time_regime_feature_provenance": regime_provenance,
        "decision_time_regime_feature_provenance_valid": bool(
            regime_provenance.get("provenance_valid")
        ),
        "decision_time_regime_feature_max_input_ts": _first_float(
            row,
            ("decision_time_regime_feature_max_input_ts",),
            default=None,
        ),
        "canonical_regime_feature_presence": dict(
            row.get("_pre_trace_merge_regime_feature_presence")
            or _hts_regime_canonical_feature_presence(row)
        ),
        "trace_regime_feature_presence": dict(
            row.get("_trace_regime_feature_presence") or {}
        ),
        "up_regime_confirmed": regime_vote_summary["market_regime"]
        == "up_regime_confirmed",
        "down_regime_confirmed": regime_vote_summary["market_regime"]
        == "down_regime_confirmed",
        "raw_fields": sorted(str(key) for key in row),
    }


def _attach_hts_regime_sequence_context(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    previous_side_by_market: dict[str, str] = {}
    count_by_market: Counter[str] = Counter()
    enriched = []
    for row in sorted(rows, key=lambda item: tuple(item["chronological_sort_key"])):
        copied = dict(row)
        market_id = str(copied.get("market_id") or "")
        previous_side = previous_side_by_market.get(market_id)
        copied["same_market_prior_entry_count"] = count_by_market[market_id]
        copied["same_market_entry_index"] = count_by_market[market_id] + 1
        if previous_side is None:
            copied["side_context"] = "first_entry"
        elif previous_side == copied.get("side"):
            copied["side_context"] = "trend_continuation_same_side"
        else:
            copied["side_context"] = "side_flip"
        count_by_market[market_id] += 1
        if copied.get("side") in {"UP", "DOWN"}:
            previous_side_by_market[market_id] = str(copied["side"])
        enriched.append(copied)
    return sorted(enriched, key=lambda item: int(item["row_index"]))


def _p_up_down_balance_regime(balance: float | None) -> str:
    if balance is None:
        return "missing_p_up_down_balance"
    if balance >= 0.10:
        return "p_up_regime"
    if balance <= -0.10:
        return "p_down_regime"
    return "p_up_down_uncertain"


def _btc_momentum_regime(momentum: float | None) -> str:
    if momentum is None:
        return "missing_btc_momentum"
    if momentum > 0.0:
        return "btc_up_momentum"
    if momentum < 0.0:
        return "btc_down_momentum"
    return "btc_flat_momentum"


def _hts_regime_vote_summary(
    *,
    p_balance_regime: str,
    btc_regime: str,
    reference_distance_bucket: str,
    action_score_margin_bucket: str,
) -> dict[str, Any]:
    votes: list[dict[str, str]] = []
    p_side = {
        "p_up_regime": "UP",
        "p_down_regime": "DOWN",
    }.get(p_balance_regime)
    if p_side is not None:
        votes.append({"feature": "p_up_down_balance", "side": p_side})
    btc_side = {
        "btc_up_momentum": "UP",
        "btc_down_momentum": "DOWN",
    }.get(btc_regime)
    if btc_side is not None:
        votes.append({"feature": "btc_momentum", "side": btc_side})
    reference_side = {
        "above_reference_price": "UP",
        "below_reference_price": "DOWN",
    }.get(reference_distance_bucket)
    if reference_side is not None:
        votes.append(
            {
                "feature": "reference_price_to_beat_distance_at_decision",
                "side": reference_side,
            }
        )
    up_votes = sum(1 for vote in votes if vote["side"] == "UP")
    down_votes = sum(1 for vote in votes if vote["side"] == "DOWN")
    if up_votes > down_votes:
        market_regime = "up_regime_confirmed"
    elif down_votes > up_votes:
        market_regime = "down_regime_confirmed"
    elif votes:
        market_regime = "conflicting_regime"
    else:
        market_regime = "uncertain_or_missing_regime"
    return {
        "market_regime": market_regime,
        "vote_count": len(votes),
        "up_vote_count": up_votes,
        "down_vote_count": down_votes,
        "votes": votes,
        "p_up_down_balance_regime": p_balance_regime,
        "btc_momentum_regime": btc_regime,
        "reference_distance_bucket": reference_distance_bucket,
        "action_score_margin_bucket": action_score_margin_bucket,
        "directional_features_used": [
            "p_up_down_balance",
            "btc_momentum",
            "reference_price_to_beat_distance_at_decision",
        ],
        "quality_features_used": ["action_score_margin"],
    }


def _combined_market_regime(p_balance_regime: str, btc_regime: str) -> str:
    p_side = {
        "p_up_regime": "UP",
        "p_down_regime": "DOWN",
    }.get(p_balance_regime)
    btc_side = {
        "btc_up_momentum": "UP",
        "btc_down_momentum": "DOWN",
    }.get(btc_regime)
    if p_side is not None and btc_side is not None and p_side != btc_side:
        return "conflicting_regime"
    side = p_side or btc_side
    if side == "UP":
        return "up_regime_confirmed"
    if side == "DOWN":
        return "down_regime_confirmed"
    return "uncertain_or_missing_regime"


def _reference_distance_bucket(value: float | None) -> str:
    if value is None:
        return "missing_reference_distance"
    if value > 0.0:
        return "above_reference_price"
    if value < 0.0:
        return "below_reference_price"
    return "at_reference_price"


def _time_since_start_bucket(value: float | None) -> str:
    if value is None:
        return "missing_time_since_start"
    if value < 60.0:
        return "first_minute"
    if value < 180.0:
        return "early_round"
    if value < 420.0:
        return "mid_round"
    return "late_round"


def _spread_bucket(value: float | None) -> str:
    if value is None:
        return "missing_spread"
    if value <= 300.0:
        return "tight_spread"
    if value <= 800.0:
        return "medium_spread"
    return "wide_spread"


def _staleness_bucket(value: float | None) -> str:
    if value is None:
        return "missing_staleness"
    if value <= 2_000.0:
        return "fresh_book"
    if value <= 10_000.0:
        return "aging_book"
    return "stale_book"


def _queue_bucket(value: float | None) -> str:
    if value is None:
        return "missing_queue"
    if value >= 0.75:
        return "high_queue_fill"
    if value >= 0.50:
        return "medium_queue_fill"
    return "low_queue_fill"


def _score_margin_bucket(value: float | None) -> str:
    if value is None:
        return "missing_score_margin"
    if value >= 0.05:
        return "strong_margin"
    if value >= 0.02:
        return "medium_margin"
    return "weak_margin"


def _pnl_distribution(rows: list[dict[str, Any]], field_name: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for row in rows:
        key = str(row.get(field_name) or "unknown")
        totals[key] = totals.get(key, 0.0) + float(row.get("settlement_pnl") or 0.0)
    return dict(sorted(totals.items()))


def _safe_roi(rows: list[dict[str, Any]]) -> float:
    cost = sum(float(row.get("cost_basis") or 0.0) for row in rows)
    if cost == 0.0:
        return 0.0
    pnl = sum(float(row.get("settlement_pnl") or 0.0) for row in rows)
    return pnl / cost


def _win_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if float(row.get("settlement_pnl") or 0.0) > 0.0) / len(
        rows
    )


def _false_positive_up_hts_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples = [
        _hts_regime_example_row(
            row,
            ["up_hts_lost", *_regime_example_reason_codes(row)],
        )
        for row in rows
        if row["action"] == "BUY_UP_HOLD_TO_SETTLEMENT"
        and float(row["settlement_pnl"]) < 0.0
    ]
    return sorted(examples, key=lambda row: float(row["settlement_pnl"]))[:20]


def _missed_opportunity_up_hts_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        if row["action"] != "BUY_UP_HOLD_TO_SETTLEMENT":
            continue
        if float(row["settlement_pnl"]) <= 0.0:
            continue
        allowed, reasons = _hts_regime_variant_allows_row(
            row,
            "up_hts_only_when_up_regime_confirmed",
        )
        if allowed:
            continue
        examples.append(
            _hts_regime_example_row(
                row,
                ["profitable_up_hts_would_be_blocked", *reasons],
            )
        )
    return sorted(examples, key=lambda row: -float(row["settlement_pnl"]))[:20]


def _up_hts_win_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples = [
        _hts_regime_example_row(
            row,
            ["profitable_up_hts", *_regime_example_reason_codes(row)],
        )
        for row in rows
        if row["action"] == "BUY_UP_HOLD_TO_SETTLEMENT"
        and float(row["settlement_pnl"]) > 0.0
    ]
    return sorted(examples, key=lambda row: -float(row["settlement_pnl"]))[:20]


def _hts_regime_example_row(
    row: Mapping[str, Any],
    reason_codes: list[str],
) -> dict[str, Any]:
    return {
        "row_index": row["row_index"],
        "market_id": row["market_id"],
        "decision_ts": row["decision_ts"],
        "intent_id": row["intent_id"],
        "action": row["action"],
        "side": row["side"],
        "entry_price": row["entry_price"],
        "price_bucket": row["price_bucket"],
        "resolved_outcome": row["resolved_outcome"],
        "settlement_pnl": row["settlement_pnl"],
        "market_regime": row["market_regime"],
        "p_up": row["p_up"],
        "p_down": row["p_down"],
        "p_up_down_balance": row["p_up_down_balance"],
        "btc_momentum": row["btc_momentum"],
        "btc_momentum_regime": row["btc_momentum_regime"],
        "reference_price_to_beat_at_decision": row[
            "reference_price_to_beat_at_decision"
        ],
        "reference_price_to_beat_distance_at_decision": row[
            "reference_price_to_beat_distance_at_decision"
        ],
        "reference_distance_bucket": row["reference_distance_bucket"],
        "time_since_market_start_seconds": row["time_since_market_start_seconds"],
        "time_since_market_start_bucket": row["time_since_market_start_bucket"],
        "time_window_bucket": row["time_window_bucket"],
        "spread_bucket": row["spread_bucket"],
        "staleness_bucket": row["staleness_bucket"],
        "queue_bucket": row["queue_bucket"],
        "action_score_margin": row["action_score_margin"],
        "action_score_margin_bucket": row["action_score_margin_bucket"],
        "side_specific_action_score_margin": row[
            "side_specific_action_score_margin"
        ],
        "side_specific_action_score_margin_bucket": row[
            "side_specific_action_score_margin_bucket"
        ],
        "regime_feature_vote_summary": row["regime_feature_vote_summary"],
        "decision_time_regime_feature_provenance_valid": row[
            "decision_time_regime_feature_provenance_valid"
        ],
        "same_market_entry_index": row["same_market_entry_index"],
        "side_context": row["side_context"],
        "diagnostic_reason_codes": sorted(set(reason_codes)),
    }


def _regime_example_reason_codes(row: Mapping[str, Any]) -> list[str]:
    reasons = []
    if row.get("market_regime") != "up_regime_confirmed":
        reasons.append("up_regime_not_confirmed")
    if row.get("time_window_bucket") in {
        "sell_before_close_only_window",
        "final_no_trade_window",
    }:
        reasons.append("late_or_sbc_only_time_window")
    if row.get("price_bucket") == "gt_0_90":
        reasons.append("entry_price_above_090")
    if row.get("side_context") == "side_flip":
        reasons.append("same_market_side_flip")
    if int(row.get("same_market_prior_entry_count") or 0) > 0:
        reasons.append("repeated_same_market_entry")
    if row.get("staleness_bucket") == "stale_book":
        reasons.append("stale_book")
    return reasons


def _recommended_hts_regime_guard_signals(
    rows: list[dict[str, Any]],
    variant_reports: dict[str, dict[str, Any]],
) -> list[str]:
    up_hts = [
        row for row in rows if row["action"] == "BUY_UP_HOLD_TO_SETTLEMENT"
    ]
    up_hts_pnl = sum(float(row["settlement_pnl"]) for row in up_hts)
    missed_up_count = len(_missed_opportunity_up_hts_examples(rows))
    recommendations = [
        "Do not disable BUY_UP_HOLD_TO_SETTLEMENT globally; keep it available for confirmed UP regimes.",
        "Require side-specific regime confirmation before HTS entries when coverage is sufficient.",
        "Treat p_up/p_down balance, BTC/reference momentum, and reference-price distance as decision-time guard inputs.",
        "Track repeated same-market entries and side flips before adding more HTS exposure.",
        "Keep spread, queue-fill, and book-staleness execution guards intact.",
        "Route late or uncertain HTS candidates toward SBC evaluation rather than forcing HTS.",
        "Do not tune these diagnostic thresholds from settlement outcomes; freeze them before forward holdout.",
    ]
    if up_hts_pnl < 0.0 and missed_up_count:
        recommendations.append(
            "UP HTS is negative here but has profitable missed-opportunity examples, so a regime gate is preferred over a side ban."
        )
    regime_variant = variant_reports["regime_aware_up_down_hts"]
    if regime_variant["fill_count"] == 0:
        recommendations.append(
            "Current artifacts do not have enough decision-time regime feature coverage for a deployable HTS regime gate."
        )
    return recommendations


def _up_hts_loss_cluster_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    losses = [
        row
        for row in rows
        if row["action"] == "BUY_UP_HOLD_TO_SETTLEMENT"
        and float(row["settlement_pnl"]) < 0.0
    ]
    cluster_fields = [
        "market_regime",
        "btc_momentum_regime",
        "reference_distance_bucket",
        "action_score_margin_bucket",
        "side_specific_action_score_margin_bucket",
        "time_since_market_start_bucket",
        "time_window_bucket",
        "price_bucket",
        "spread_bucket",
        "staleness_bucket",
        "queue_bucket",
        "side_context",
    ]
    clusters = {
        field: _pnl_distribution(losses, field)
        for field in cluster_fields
    }
    strongest_cluster_by_field = {
        field: _largest_loss_cluster(distribution)
        for field, distribution in clusters.items()
    }
    return {
        "loss_count": len(losses),
        "loss_pnl_sum": sum(float(row["settlement_pnl"]) for row in losses),
        "cluster_fields": cluster_fields,
        "pnl_by_cluster": clusters,
        "strongest_loss_cluster_by_field": strongest_cluster_by_field,
        "diagnostic_conclusion": _up_hts_loss_cluster_conclusion(
            strongest_cluster_by_field
        ),
    }


def _largest_loss_cluster(distribution: dict[str, float]) -> dict[str, Any]:
    if not distribution:
        return {"bucket": None, "pnl": 0.0}
    bucket, pnl = min(distribution.items(), key=lambda item: item[1])
    return {"bucket": bucket, "pnl": pnl}


def _up_hts_loss_cluster_conclusion(
    strongest_cluster_by_field: dict[str, dict[str, Any]],
) -> list[str]:
    conclusions = []
    bucket_by_field = {
        field: str(value.get("bucket"))
        for field, value in strongest_cluster_by_field.items()
    }
    if bucket_by_field.get("btc_momentum_regime") in {
        "btc_down_momentum",
        "missing_btc_momentum",
    }:
        conclusions.append("up_hts_losses_cluster_around_weak_or_missing_btc_momentum")
    if bucket_by_field.get("reference_distance_bucket") in {
        "below_reference_price",
        "missing_reference_distance",
    }:
        conclusions.append(
            "up_hts_losses_cluster_around_poor_or_missing_reference_distance"
        )
    if bucket_by_field.get("action_score_margin_bucket") in {
        "weak_margin",
        "missing_score_margin",
    }:
        conclusions.append("up_hts_losses_cluster_around_low_or_missing_score_margin")
    if bucket_by_field.get("time_window_bucket") in {
        "sell_before_close_only_window",
        "final_no_trade_window",
    }:
        conclusions.append("up_hts_losses_cluster_around_late_entry")
    if bucket_by_field.get("price_bucket") == "gt_0_90":
        conclusions.append("up_hts_losses_cluster_around_high_entry_price")
    if not conclusions:
        conclusions.append("up_hts_loss_cluster_not_explained_by_single_bucket")
    return conclusions


def _hts_regime_feature_coverage(
    rows: list[dict[str, Any]],
    *,
    canonical_only: bool = False,
) -> dict[str, Any]:
    fields = _hts_regime_decision_time_fields()
    tracked_fields = {
        "p_up",
        "p_down",
        "btc_momentum",
        "reference_price_to_beat_distance_at_decision",
        "time_since_market_start_seconds",
        "time_to_close_seconds",
        "spread_bps",
        "book_staleness_ms",
        "queue_fill_proxy",
        "action_score_margin",
        "side_specific_action_score_margin",
    }
    coverage = {}
    for field in fields:
        if field not in tracked_fields:
            continue
        if canonical_only:
            canonical_presence_rows = [
                row
                for row in rows
                if field in (row.get("canonical_regime_feature_presence") or {})
            ]
            available_count = sum(
                1
                for row in canonical_presence_rows
                if bool(
                    (row.get("canonical_regime_feature_presence") or {}).get(field)
                )
            )
            if not canonical_presence_rows:
                available_count = sum(1 for row in rows if row.get(field) is not None)
        else:
            available_count = sum(1 for row in rows if row.get(field) is not None)
        coverage[field] = {
            "available_count": available_count,
            "row_count": len(rows),
        }
    return coverage


def _hts_regime_decision_time_fields() -> list[str]:
    return [
        "p_up",
        "p_down",
        "p_up_down_balance",
        "btc_momentum",
        "reference_price_to_beat_distance_at_decision",
        "time_since_market_start_seconds",
        "time_to_close_seconds",
        "entry_price",
        "spread_bps",
        "book_staleness_ms",
        "queue_fill_proxy",
        "action_score_margin",
        "side_specific_action_score_margin",
        "action_score",
        "same_market_prior_entry_count",
        "side_context",
    ]


def _normalize_settlement_row(row: dict[str, str], index: int) -> dict[str, Any]:
    action = _first_text(
        row,
        (
            "action",
            "selected_action",
            "policy_action",
            "entry_policy_action",
            "signal_action",
        ),
        default="UNKNOWN",
    )
    action = _canonical_action(action)
    entry_price, entry_price_field = _first_float_with_field(
        row,
        (
            "entry_price",
            "execution_price",
            "fill_price",
            "avg_price",
            "price",
            "entry_ask",
            "ask",
        ),
    )
    cost_basis = _first_float(
        row,
        (
            "cost_basis",
            "cost_basis_usdc",
            "entry_cost",
            "paper_notional",
            "notional",
            "size_usdc",
            "fill_notional",
            "entry_notional",
        ),
        default=0.0,
    )
    if cost_basis <= 0.0:
        shares = _first_float(row, ("shares", "quantity", "size"), default=0.0)
        if shares > 0.0 and entry_price is not None:
            cost_basis = shares * entry_price
    settlement_pnl = _first_float(
        row,
        (
            "settlement_pnl",
            "settlement_pnl_usdc",
            "total_polymarket_pnl",
            "pnl",
            "realized_pnl",
            "net_pnl",
        ),
        default=0.0,
    )
    horizon = _infer_horizon(row)
    family = _infer_family(row, action)
    decision_ts_raw = _first_text(row, ("decision_ts", "ts", "timestamp"), default=str(index))
    decision_ts_numeric = _parse_sort_number(decision_ts_raw)
    numeric_iteration = _first_sort_number(
        row,
        ("iteration", "round_iteration", "loop_iteration", "cycle_index"),
        default=MISSING_SORT_NUMBER,
    )
    intent_id = _first_text(
        row,
        ("intent_id", "paper_intent_id", "order_intent_id", "signal_id"),
        default="",
    )
    return {
        "row_index": index,
        "market_id": _first_text(row, ("market_id", "condition_id", "slug"), default=""),
        "decision_ts": decision_ts_raw,
        "decision_ts_numeric": decision_ts_numeric,
        "numeric_iteration": numeric_iteration,
        "intent_id": intent_id,
        "chronological_sort_key": [
            numeric_iteration,
            decision_ts_numeric,
            intent_id,
            index,
        ],
        "action": action,
        "family": family,
        "horizon": horizon,
        "entry_price": entry_price,
        "price_bucket": _price_bucket(entry_price),
        "cost_basis": cost_basis,
        "settlement_pnl": settlement_pnl,
        "entry_price_source_field": entry_price_field,
        "raw_fields": sorted(row.keys()),
    }


def _load_forward_shadow_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        rows: list[dict[str, Any]] = []
        for trace_path in sorted(path.rglob("o_v8_paper_fresh_signal_trace.json")):
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
            for row in _extract_forward_shadow_rows_from_object(payload):
                copied = dict(row)
                copied.setdefault("source_signal_trace_path", str(trace_path))
                rows.append(copied)
        if not rows:
            raise ValueError(
                "forward shadow input directory contains no "
                "o_v8_paper_fresh_signal_trace.json trace rows"
            )
        return rows
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise ValueError("forward shadow JSONL rows must be objects")
                    rows.append(payload)
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = _extract_forward_shadow_rows_from_object(payload)
    else:
        raise ValueError("forward shadow input must be a JSON object, array, or JSONL")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("forward shadow input rows must be objects")
    return list(rows)


def _extract_forward_shadow_rows_from_object(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for field_name in (
        "signal_trace_rows",
        "trace_rows",
        "holdout_decision_rows",
        "decision_rows",
        "rows",
        "feature_rows",
    ):
        rows = payload.get(field_name)
        if isinstance(rows, list):
            return [_copy_row_with_report_context(row, payload) for row in rows]
    raise ValueError(
        "forward shadow JSON object must include one of: signal_trace_rows, "
        "trace_rows, holdout_decision_rows, decision_rows, rows, feature_rows"
    )


def _copy_row_with_report_context(
    row: Any,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(row, dict):
        return row
    copied = dict(row)
    for field_name in (
        "paper_only",
        "capital_at_risk",
        "polymarket_write_enabled",
        "wallet_signing_enabled",
        "v8_execution_handoff_allowed",
        "source_model_candidate_eligible",
        "freeze_ready",
        "promotion_evidence_eligible",
        "#134_resume_allowed",
        "#146_start_allowed",
    ):
        if field_name in payload and field_name not in copied:
            copied[field_name] = payload[field_name]
    return copied


def _load_frozen_ev_calibration_artifact(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "path": None,
            "sha256": None,
            "payload": {},
            "valid": False,
            "status": "missing_frozen_ev_calibration_artifact",
            "source": "missing_frozen_ev_calibration_artifact",
            "blocking_reason_codes": ["missing_frozen_ev_calibration_artifact"],
        }
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return {
            "path": str(path),
            "sha256": None,
            "payload": {},
            "valid": False,
            "status": "frozen_ev_calibration_artifact_not_found",
            "source": "missing_frozen_ev_calibration_artifact",
            "blocking_reason_codes": ["frozen_ev_calibration_artifact_not_found"],
        }
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            "path": str(resolved),
            "sha256": _sha256_file(resolved),
            "payload": {},
            "valid": False,
            "status": "frozen_ev_calibration_artifact_invalid_json",
            "source": "invalid_frozen_ev_calibration_artifact",
            "blocking_reason_codes": ["frozen_ev_calibration_artifact_invalid_json"],
        }
    forbidden_paths = _recursive_forbidden_field_paths(
        payload,
        set(EXECUTION_LAYER_V2_FORBIDDEN_OUTCOME_FIELDS),
    )
    blocking_reasons: list[str] = []
    if not bool(payload.get("frozen", False)):
        blocking_reasons.append("frozen_ev_calibration_artifact_not_frozen")
    if not bool(payload.get("decision_time_safe", False)):
        blocking_reasons.append("frozen_ev_calibration_artifact_not_decision_time_safe")
    if bool(payload.get("uses_validation_labels_for_tuning", False)):
        blocking_reasons.append("frozen_ev_calibration_artifact_uses_validation_labels")
    if bool(payload.get("market_implied_probability_used_for_ev", False)):
        blocking_reasons.append("frozen_ev_calibration_artifact_uses_market_implied_ev")
    if forbidden_paths:
        blocking_reasons.append("frozen_ev_calibration_artifact_forbidden_fields_present")
    valid = not blocking_reasons
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "payload": payload,
        "valid": valid,
        "status": "frozen_ev_calibration_artifact_valid"
        if valid
        else "frozen_ev_calibration_artifact_invalid",
        "source": "frozen_ev_calibration_artifact"
        if valid
        else "invalid_frozen_ev_calibration_artifact",
        "blocking_reason_codes": blocking_reasons,
        "forbidden_field_paths": forbidden_paths,
    }


def _normalize_forward_shadow_row(
    row: dict[str, Any],
    index: int,
    *,
    default_execution_cost: float,
    ev_calibration_artifact: dict[str, Any],
) -> dict[str, Any]:
    action = _canonical_action(
        _first_text(
            row,
            (
                "selected_action",
                "canonical_selected_action",
                "execution_guarded_action",
                "canonical_o_selected_action",
                "canonical_action",
                "action",
                "policy_action",
                "signal_action",
            ),
            default="NO_TRADE",
        )
    )
    side = _infer_side(row, action)
    execution_price, execution_price_field = _execution_price_for_side(row, side)
    p_market, p_market_field = _probability_source_for_side(
        row,
        side,
        (
            "p_market_implied",
            "market_implied_probability",
            "p_market_implied_{side}",
            "market_implied_probability_{side}",
            "p_{side}",
        ),
    )
    if p_market is None and execution_price is not None:
        p_market = execution_price
        p_market_field = execution_price_field
    p_model, p_model_field = _probability_source_for_side(
        row,
        side,
        (
            "p_model_fair_value",
            "model_fair_value_probability",
            "fair_value_probability",
            "p_model_fair_value_{side}",
            "model_fair_value_{side}",
            "calibrated_p_{side}",
            "model_p_{side}",
            "predicted_probability_{side}",
        ),
    )
    expected_return, expected_return_field = _first_float_with_field(
        row,
        (
            "calibrated_action_expected_net_return",
            "action_expected_net_return",
            "expected_net_return",
            "selected_action_expected_net_return",
            "calibrated_ev",
        ),
    )
    canonical_score, canonical_score_field = _first_float_with_field(
        row,
        (
            "canonical_o_action_score",
            "canonical_corrected_score",
            "execution_guarded_score",
            "canonical_action_score",
            "raw_calibrated_action_score",
            "selected_score",
            "action_score",
            "model_score",
        ),
    )
    canonical_raw_score, canonical_raw_score_field = _first_float_with_field(
        row,
        (
            "canonical_o_raw_score",
            "canonical_raw_score",
            "raw_calibrated_action_score",
        ),
    )
    execution_cost = _first_float(
        row,
        ("execution_cost", "cost", "estimated_cost", "fee_plus_spread_cost"),
        default=default_execution_cost,
    )
    decision_ts = _first_text(row, ("decision_ts", "ts", "timestamp"), default=str(index))
    decision_ts_numeric = _parse_sort_number(decision_ts)
    market_id = _first_text(row, ("market_id", "condition_id", "slug"), default="")
    family = _infer_family(row, action)
    time_to_close = _first_float(
        row,
        (
            "time_to_close_seconds",
            "time_to_expiry_seconds",
            "microstructure_snapshot.time_to_close_seconds",
            "features.time_to_close_seconds",
        ),
        default=None,
    )
    guard_blocking_codes, guard_blocking_source_fields = _reason_codes_from_fields(
        row,
        (
            "execution_blocking_reason_codes",
            "execution_guard_blocking_reason_codes",
            "blocking_reason_codes",
            "reason_codes",
            "execution_guard_reason_codes",
        ),
    )
    order_allowed, order_allowed_field = _first_bool_with_field(
        row,
        ("order_allowed", "execution_order_allowed", "guard_order_allowed"),
    )
    execution_guarded_action = _canonical_action(
        _first_text(
            row,
            (
                "execution_guarded_action",
                "guarded_action",
                "order_action",
            ),
            default="",
        )
    )
    execution_guarded_side = _first_text(
        row,
        ("execution_guarded_side", "guarded_side", "order_side"),
        default="",
    ).upper()
    time_to_close_gate_passed, _ = _first_bool_with_field(
        row,
        ("time_to_close_gate_passed", "execution_time_to_close_gate_passed"),
    )
    required_min_time_to_close = _first_float(
        row,
        ("required_min_time_to_close_seconds", "min_time_to_close_seconds"),
        default=None,
    )
    spread_bps = _first_float(
        row,
        ("spread_bps", "microstructure_snapshot.spread_bps", "features.spread_bps"),
        default=None,
    )
    book_staleness_ms = _first_float(
        row,
        (
            "book_staleness_ms",
            "microstructure_snapshot.book_staleness_ms",
            "features.book_staleness_ms",
        ),
        default=None,
    )
    queue_fill_proxy = _first_float(
        row,
        (
            "queue_fill_proxy",
            "queue_fill_probability_estimate",
            "microstructure_snapshot.queue_fill_proxy",
            "features.queue_fill_proxy",
        ),
        default=None,
    )
    executable_exit_bid_proxy = _first_float(
        row,
        (
            "executable_exit_bid_proxy",
            "terminal_bid",
            "best_candidate_bid",
            "microstructure_snapshot.executable_exit_bid_proxy",
            "features.executable_exit_bid_proxy",
        ),
        default=None,
    )
    (
        expected_return,
        expected_return_field,
        calibrated_ev_source,
        calibrated_ev_source_provenance,
        calibrated_ev_blocking_reasons,
    ) = _calibrated_expected_return_source(
        input_expected_return=expected_return,
        input_expected_return_field=expected_return_field,
        canonical_score=canonical_score,
        canonical_score_field=canonical_score_field,
        canonical_raw_score=canonical_raw_score,
        canonical_raw_score_field=canonical_raw_score_field,
        execution_price=execution_price,
        execution_price_field=execution_price_field,
        executable_exit_bid_proxy=executable_exit_bid_proxy,
        spread_bps=spread_bps,
        queue_fill_proxy=queue_fill_proxy,
        book_staleness_ms=book_staleness_ms,
        time_to_close=time_to_close,
        family=family,
        side=side,
        execution_cost=execution_cost,
        ev_calibration_artifact=ev_calibration_artifact,
    )
    missing_guard_fields = _missing_guard_decision_fields(
        order_allowed_field=order_allowed_field,
        execution_guarded_action=execution_guarded_action,
        execution_guarded_side=execution_guarded_side,
        guard_blocking_source_fields=guard_blocking_source_fields,
    )
    calibrated_ev, ev_source, ev_provenance, ev_status, ev_reasons = (
        _derive_calibrated_ev_contract(
            expected_return=expected_return,
            expected_return_field=expected_return_field,
            p_model=p_model,
            p_model_field=p_model_field,
            execution_price=execution_price,
            execution_price_field=execution_price_field,
            execution_cost=execution_cost,
        )
    )
    return {
        "row_index": index,
        "market_id": market_id,
        "decision_ts": decision_ts,
        "decision_ts_numeric": decision_ts_numeric,
        "action": action,
        "family": family,
        "side": side,
        "horizon": _infer_horizon(row),
        "entry_price": execution_price,
        "execution_price": execution_price,
        "execution_price_source_field": execution_price_field,
        "price_bucket": _price_bucket(execution_price),
        "p_market_implied": p_market,
        "p_market_implied_source_field": p_market_field,
        "p_model_fair_value": p_model,
        "p_model_fair_value_source_field": p_model_field,
        "calibrated_action_expected_net_return": expected_return,
        "calibrated_action_expected_net_return_source_field": expected_return_field,
        "calibrated_action_expected_net_return_available": expected_return is not None,
        "calibrated_ev_source": calibrated_ev_source,
        "calibrated_ev_source_provenance": calibrated_ev_source_provenance,
        "calibrated_ev_blocking_reason_codes": calibrated_ev_blocking_reasons,
        "canonical_o_action_score": canonical_score,
        "canonical_o_action_score_source_field": canonical_score_field,
        "canonical_o_raw_score": canonical_raw_score,
        "canonical_o_raw_score_source_field": canonical_raw_score_field,
        "execution_cost": execution_cost,
        "time_to_close_seconds": time_to_close,
        "time_window_bucket": _time_window_bucket(time_to_close),
        "order_allowed": order_allowed,
        "execution_guarded_action": execution_guarded_action or None,
        "execution_guarded_side": execution_guarded_side or None,
        "execution_guard_blocking_reason_codes": guard_blocking_codes,
        "execution_guard_blocking_reason_source_fields": guard_blocking_source_fields,
        "missing_execution_guard_decision_fields": missing_guard_fields,
        "available_actions": _available_actions(row),
        "time_to_close_gate_passed": time_to_close_gate_passed,
        "required_min_time_to_close_seconds": required_min_time_to_close,
        "spread_bps": spread_bps,
        "book_staleness_ms": book_staleness_ms,
        "queue_fill_proxy": queue_fill_proxy,
        "executable_exit_bid_proxy": executable_exit_bid_proxy,
        "paper_intent_id": _first_text(row, ("paper_intent_id",), default="") or None,
        "paper_fill_id": _first_text(row, ("paper_fill_id",), default="") or None,
        "execution_guard_safety_flags_blocked": _execution_guard_safety_flags_blocked(
            row
        ),
        "calibrated_ev": calibrated_ev,
        "ev_source": ev_source,
        "ev_source_provenance": ev_provenance,
        "ev_mapping_status": ev_status,
        "ev_mapping_blocking_reason_codes": ev_reasons,
        "calibrated_ev_available": calibrated_ev is not None,
        "market_implied_probability_used_for_ev": False,
        "raw_fields": sorted(row.keys()),
    }


def _calibrated_expected_return_source(
    *,
    input_expected_return: float | None,
    input_expected_return_field: str | None,
    canonical_score: float | None,
    canonical_score_field: str | None,
    canonical_raw_score: float | None,
    canonical_raw_score_field: str | None,
    execution_price: float | None,
    execution_price_field: str | None,
    executable_exit_bid_proxy: float | None,
    spread_bps: float | None,
    queue_fill_proxy: float | None,
    book_staleness_ms: float | None,
    time_to_close: float | None,
    family: str,
    side: str,
    execution_cost: float | None,
    ev_calibration_artifact: dict[str, Any],
) -> tuple[float | None, str | None, str, dict[str, Any], list[str]]:
    if input_expected_return is not None:
        return (
            input_expected_return,
            input_expected_return_field,
            "input_calibrated_action_expected_net_return",
            {
                "source_fields_used": [input_expected_return_field],
                "formula": "input_calibrated_action_expected_net_return",
                "market_implied_probability_used_for_ev": False,
            },
            [],
        )
    if not ev_calibration_artifact["valid"]:
        return (
            None,
            None,
            ev_calibration_artifact["source"],
            {
                "source_fields_used": [],
                "formula": None,
                "calibration_artifact_path": ev_calibration_artifact["path"],
                "calibration_artifact_hash": ev_calibration_artifact["sha256"],
                "market_implied_probability_used_for_ev": False,
            },
            list(ev_calibration_artifact["blocking_reason_codes"]),
        )
    payload = ev_calibration_artifact["payload"]
    score_config = payload.get("score_to_expected_net_return") or payload.get(
        "score_to_ev",
        {},
    )
    if not isinstance(score_config, Mapping):
        return (
            None,
            None,
            "invalid_frozen_ev_calibration_artifact",
            {
                "source_fields_used": [],
                "formula": None,
                "calibration_artifact_path": ev_calibration_artifact["path"],
                "calibration_artifact_hash": ev_calibration_artifact["sha256"],
                "market_implied_probability_used_for_ev": False,
            },
            ["frozen_ev_calibration_artifact_missing_score_to_expected_net_return"],
        )
    features = {
        "canonical_o_action_score": canonical_score,
        "canonical_o_raw_score": canonical_raw_score,
        "execution_price": execution_price,
        "executable_exit_bid_proxy": executable_exit_bid_proxy,
        "spread_bps": spread_bps,
        "queue_fill_proxy": queue_fill_proxy,
        "book_staleness_ms": book_staleness_ms,
        "time_to_close_seconds": time_to_close,
    }
    field_sources = {
        "canonical_o_action_score": canonical_score_field,
        "canonical_o_raw_score": canonical_raw_score_field,
        "execution_price": execution_price_field,
        "executable_exit_bid_proxy": "executable_exit_bid_proxy",
        "spread_bps": "spread_bps",
        "queue_fill_proxy": "queue_fill_proxy",
        "book_staleness_ms": "book_staleness_ms",
        "time_to_close_seconds": "time_to_close_seconds",
    }
    value = _float_from_mapping(score_config, "intercept", default=0.0)
    source_fields_used: list[str] = ["intercept"]
    missing_features: list[str] = []
    for feature_name, feature_value in features.items():
        weight = _feature_weight(score_config, feature_name)
        if weight == 0.0:
            continue
        if feature_value is None:
            missing_features.append(feature_name)
            continue
        value += weight * float(feature_value)
        source_fields_used.append(field_sources[feature_name] or feature_name)
    if missing_features:
        return (
            None,
            None,
            "frozen_ev_calibration_artifact",
            {
                "source_fields_used": sorted(set(source_fields_used)),
                "formula": "linear_score_to_expected_net_return",
                "missing_required_feature_fields": sorted(missing_features),
                "calibration_artifact_path": ev_calibration_artifact["path"],
                "calibration_artifact_hash": ev_calibration_artifact["sha256"],
                "market_implied_probability_used_for_ev": False,
            },
            [
                f"missing_decision_time_ev_feature:{feature}"
                for feature in sorted(missing_features)
            ],
        )
    value += _offset_from_mapping(payload.get("family_offsets"), family)
    value += _offset_from_mapping(payload.get("side_offsets"), side)
    source_fields_used.extend(["selected_action", "selected_side", "action_family"])
    if bool(payload.get("subtract_execution_cost", True)):
        value -= 0.0 if execution_cost is None else float(execution_cost)
        source_fields_used.append("execution_cost")
    return (
        value,
        "frozen_ev_calibration_artifact",
        "frozen_ev_calibration_artifact",
        {
            "source_fields_used": sorted(set(source_fields_used)),
            "formula": "linear_score_to_expected_net_return_minus_cost",
            "calibration_artifact_path": ev_calibration_artifact["path"],
            "calibration_artifact_hash": ev_calibration_artifact["sha256"],
            "score_to_expected_net_return_config": dict(score_config),
            "family": family,
            "side": side,
            "market_implied_probability_used_for_ev": False,
        },
        [],
    )


def _derive_calibrated_ev_contract(
    *,
    expected_return: float | None,
    expected_return_field: str | None,
    p_model: float | None,
    p_model_field: str | None,
    execution_price: float | None,
    execution_price_field: str | None,
    execution_cost: float | None,
) -> tuple[float | None, str, dict[str, Any], str, list[str]]:
    if expected_return is not None:
        return (
            expected_return,
            "calibrated_action_expected_net_return",
            {
                "source_fields_used": [expected_return_field],
                "formula": "calibrated_action_expected_net_return",
                "market_implied_probability_used_for_ev": False,
            },
            "calibrated_ev_available",
            [],
        )
    if p_model is not None and execution_price is not None:
        cost = 0.0 if execution_cost is None else execution_cost
        return (
            p_model - execution_price - cost,
            "p_model_fair_value_minus_execution_price_minus_cost",
            {
                "source_fields_used": [p_model_field, execution_price_field, "execution_cost"],
                "formula": "p_model_fair_value - execution_price - cost",
                "market_implied_probability_used_for_ev": False,
            },
            "calibrated_ev_available",
            [],
        )
    reasons = []
    if p_model is None and expected_return is None:
        reasons.append("missing_calibrated_model_fair_value_or_action_expected_return")
    if execution_price is None and expected_return is None:
        reasons.append("missing_execution_price_for_fair_value_ev")
    return (
        None,
        "missing_calibrated_ev_source",
        {
            "source_fields_used": [],
            "formula": None,
            "market_implied_probability_used_for_ev": False,
        },
        "blocked_missing_calibrated_ev_source",
        reasons,
    )


def _reason_codes_from_fields(
    row: Mapping[str, Any],
    field_names: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    values: list[str] = []
    source_fields: list[str] = []
    for field_name in field_names:
        raw_value = _lookup_value(row, field_name)
        if raw_value is None:
            continue
        source_fields.append(field_name)
        values.extend(_coerce_reason_codes(raw_value))
    return sorted(set(values)), source_fields


def _coerce_reason_codes(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        if not raw_value.strip():
            return []
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            parsed = raw_value
        else:
            return _coerce_reason_codes(parsed)
        return [part.strip() for part in parsed.split(",") if part.strip()]
    if isinstance(raw_value, list | tuple | set):
        values: list[str] = []
        for item in raw_value:
            values.extend(_coerce_reason_codes(item))
        return values
    return [str(raw_value)]


def _available_actions(row: Mapping[str, Any]) -> list[str] | None:
    for field_name in (
        "available_actions",
        "candidate_actions",
        "action_candidates",
    ):
        raw_value = _lookup_value(row, field_name)
        if raw_value is None:
            continue
        if isinstance(raw_value, str):
            try:
                parsed = json.loads(raw_value)
            except json.JSONDecodeError:
                parsed = [part.strip() for part in raw_value.split(",")]
            raw_value = parsed
        if not isinstance(raw_value, list | tuple | set):
            continue
        actions = sorted(
            {
                _canonical_action(str(item))
                for item in raw_value
                if str(item).strip()
            }
        )
        return actions
    return None


def _missing_guard_decision_fields(
    *,
    order_allowed_field: str | None,
    execution_guarded_action: str,
    execution_guarded_side: str,
    guard_blocking_source_fields: list[str],
) -> list[str]:
    missing = []
    if order_allowed_field is None:
        missing.append("order_allowed")
    if not execution_guarded_action:
        missing.append("execution_guarded_action")
    if not execution_guarded_side:
        missing.append("execution_guarded_side")
    if not guard_blocking_source_fields:
        missing.append("execution_blocking_reason_codes")
    return missing


def _execution_guard_safety_flags_blocked(row: Mapping[str, Any]) -> bool:
    return (
        _optional_bool(row, "capital_at_risk", default=False) is False
        and _optional_bool(row, "polymarket_write_enabled", default=False) is False
        and _optional_bool(row, "wallet_signing_enabled", default=False) is False
        and _optional_bool(row, "v8_execution_handoff_allowed", default=False) is False
    )


def _execution_price_for_side(
    row: dict[str, Any],
    side: str,
) -> tuple[float | None, str | None]:
    side_lower = side.lower()
    side_fields = (
        f"execution_price_{side_lower}",
        f"entry_ask_{side_lower}",
        f"ask_{side_lower}",
        f"{side_lower}_ask",
        f"microstructure_snapshot.{side_lower}_ask",
        f"features.{side_lower}_ask",
    )
    generic_fields = (
        "execution_price",
        "entry_price",
        "entry_ask",
        "ask",
        "fill_price",
        "microstructure_snapshot.entry_ask",
        "features.entry_ask",
    )
    return _first_float_with_field(row, (*side_fields, *generic_fields))


def _probability_source_for_side(
    row: dict[str, Any],
    side: str,
    field_templates: tuple[str, ...],
) -> tuple[float | None, str | None]:
    side_lower = side.lower()
    side_upper = side.upper()
    fields = tuple(
        template.format(side=side_lower, SIDE=side_upper)
        for template in field_templates
    )
    value, field = _first_float_with_field(row, fields)
    if value is not None:
        return value, field
    return None, None


def _infer_side(row: dict[str, Any], action: str) -> str:
    explicit = _first_text(
        row,
        (
            "selected_side",
            "canonical_selected_side",
            "execution_guarded_side",
            "side",
            "target_side",
            "outcome",
        ),
        default="",
    ).upper()
    if explicit in {"UP", "DOWN", "NONE"}:
        return explicit
    if "_UP_" in f"_{action}_" or action.startswith("BUY_UP"):
        return "UP"
    if "_DOWN_" in f"_{action}_" or action.startswith("BUY_DOWN"):
        return "DOWN"
    return "NONE"


def _infer_family(row: dict[str, Any], action: str) -> str:
    family = _first_text(
        row,
        (
            "family",
            "action_family",
            "selected_action_family",
            "canonical_selected_family",
            "execution_guarded_family",
            "exit_policy",
        ),
        default="",
    ).upper()
    if "SELL_BEFORE_CLOSE" in family or "SELL_BEFORE_CLOSE" in action:
        return "SELL_BEFORE_CLOSE"
    if "HOLD_TO_SETTLEMENT" in family or "HOLD_TO_SETTLEMENT" in action:
        return "HOLD_TO_SETTLEMENT"
    if action == "NO_TRADE":
        return "NO_TRADE"
    return family or "UNKNOWN"


def _infer_horizon(row: dict[str, Any]) -> str:
    explicit = _first_text(
        row,
        ("horizon", "market_horizon", "market_family", "slug"),
        default="",
    ).lower()
    horizon_ms = _first_float(row, ("horizon_ms",), default=None)
    if horizon_ms is not None:
        if int(horizon_ms) == 300_000:
            return "5m"
        if int(horizon_ms) == 900_000:
            return "15m"
    if "15m" in explicit or "15-min" in explicit or "15_min" in explicit:
        return "15m"
    if "5m" in explicit or "5-min" in explicit or "5_min" in explicit:
        return "5m"
    return explicit or "unknown"


def _price_in_070_090(price: float | None) -> bool:
    return price is not None and 0.70 <= price <= 0.90


def _price_rejection_reason(price: float | None) -> str:
    if price is None:
        return "missing_entry_price"
    if price < 0.70:
        return "price_below_070"
    return "price_above_090"


def _price_bucket(price: float | None) -> str:
    if price is None:
        return "missing"
    for name, low, high in PRICE_BUCKET_EDGES:
        if price >= low and (high is None or price < high):
            return name
    return "unknown"


def _price_bucket_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary = {}
    for bucket in sorted({row["price_bucket"] for row in rows}):
        bucket_rows = [row for row in rows if row["price_bucket"] == bucket]
        summary[bucket] = _metric_subset(bucket_rows)
    return summary


def _family_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary = {}
    for family in sorted({row["family"] for row in rows}):
        family_rows = [row for row in rows if row["family"] == family]
        summary[family] = _metric_subset(family_rows)
    return summary


def _metric_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cost = sum(float(row["cost_basis"]) for row in rows)
    pnl = sum(float(row["settlement_pnl"]) for row in rows)
    return {
        "row_count": len(rows),
        "cost_basis": cost,
        "settlement_pnl": pnl,
        "roi": pnl / cost if cost else 0.0,
        "win_rate": (
            sum(1 for row in rows if float(row["settlement_pnl"]) > 0.0) / len(rows)
            if rows
            else 0.0
        ),
    }


def _max_drawdown(pnl_values: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for pnl in pnl_values:
        equity += pnl
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity - peak)
    return max_drawdown


def _count_distribution(rows: list[dict[str, Any]], field_name: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field_name, "unknown")) for row in rows).items()))


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row must be an object: {path}")
            rows.append(payload)
    return rows


def _canonical_action(action: str) -> str:
    return action.strip().upper().replace(" ", "_")


def _first_text(
    row: Mapping[str, Any],
    field_names: tuple[str, ...],
    *,
    default: str,
) -> str:
    for field_name in field_names:
        value = _lookup_value(row, field_name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _first_float(
    row: Mapping[str, Any],
    field_names: tuple[str, ...],
    *,
    default: float | None,
) -> float | None:
    value, _ = _first_float_with_field(row, field_names)
    return default if value is None else value


def _first_float_with_field(
    row: Mapping[str, Any],
    field_names: tuple[str, ...],
) -> tuple[float | None, str | None]:
    for field_name in field_names:
        raw_value = _lookup_value(row, field_name)
        if raw_value is None or str(raw_value).strip() == "":
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value, field_name
    return None, None


def _first_sort_number(
    row: Mapping[str, Any],
    field_names: tuple[str, ...],
    *,
    default: float,
) -> float:
    for field_name in field_names:
        raw_value = _lookup_value(row, field_name)
        if raw_value is None or str(raw_value).strip() == "":
            continue
        parsed = _parse_sort_number(str(raw_value))
        if math.isfinite(parsed):
            return parsed
    return float(default)


def _first_bool_with_field(
    row: Mapping[str, Any],
    field_names: tuple[str, ...],
) -> tuple[bool | None, str | None]:
    for field_name in field_names:
        raw_value = _lookup_value(row, field_name)
        if raw_value is None:
            continue
        parsed = _coerce_bool(raw_value)
        if parsed is not None:
            return parsed, field_name
    return None, None


def _optional_bool(row: Mapping[str, Any], field_name: str, *, default: bool) -> bool:
    raw_value = _lookup_value(row, field_name)
    parsed = _coerce_bool(raw_value)
    return default if parsed is None else parsed


def _coerce_bool(raw_value: Any) -> bool | None:
    if isinstance(raw_value, bool):
        return raw_value
    if raw_value is None:
        return None
    if isinstance(raw_value, int | float) and math.isfinite(float(raw_value)):
        return bool(raw_value)
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return None


def _float_from_mapping(
    payload: Mapping[str, Any],
    field_name: str,
    *,
    default: float,
) -> float:
    value = payload.get(field_name, default)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _feature_weight(payload: Mapping[str, Any], feature_name: str) -> float:
    for key in (
        f"{feature_name}_weight",
        feature_name,
    ):
        if key in payload:
            return _float_from_mapping(payload, key, default=0.0)
    return 0.0


def _offset_from_mapping(payload: Any, key: str) -> float:
    if not isinstance(payload, Mapping):
        return 0.0
    value = payload.get(key) or payload.get(str(key).upper()) or payload.get(str(key).lower())
    if value is None:
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _lookup_value(row: Mapping[str, Any], field_name: str) -> Any:
    if field_name in row:
        return row.get(field_name)
    current: Any = row
    for part in field_name.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return None
    return current


def _parse_sort_number(raw_value: str) -> float:
    try:
        value = float(raw_value)
    except ValueError:
        return float(MISSING_SORT_NUMBER)
    return value if math.isfinite(value) else float(MISSING_SORT_NUMBER)


def _time_window_bucket(time_to_close: float | None) -> str:
    if time_to_close is None:
        return "missing"
    if time_to_close <= 60.0:
        return "final_no_trade_window"
    if time_to_close <= 120.0:
        return "sell_before_close_only_window"
    if time_to_close <= 300.0:
        return "hts_allowed_window"
    return "early_window"


def _is_entry_action(action: str) -> bool:
    return action.startswith("BUY_") or action in {"ENTER_POSITION", "ROTATE_POSITION"}


def _is_exit_action(action: str) -> bool:
    return action in {"EXIT_POSITION", "SELL_POSITION"} or action.startswith("SELL_")


def _is_hold_action(action: str) -> bool:
    return action in {"HOLD", "HOLD_POSITION"}


def _decision_type(action: str) -> str:
    if _is_entry_action(action):
        return "ENTRY"
    if _is_exit_action(action):
        return "EXIT"
    if _is_hold_action(action):
        return "HOLD"
    if action == "NO_TRADE":
        return "NO_TRADE"
    return "OTHER"


def _forbidden_fields_by_row(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    forbidden_set = set(EXECUTION_LAYER_V2_FORBIDDEN_OUTCOME_FIELDS)
    violations = []
    for index, row in enumerate(rows):
        present = sorted(_recursive_forbidden_field_paths(row, forbidden_set))
        if present:
            violations.append(
                {
                    "row_index": index,
                    "market_id": str(row.get("market_id", "")),
                    "decision_ts": str(row.get("decision_ts", "")),
                    "forbidden_fields": present,
                }
            )
    return violations


def _recursive_forbidden_field_paths(
    payload: Any,
    forbidden_set: set[str],
    *,
    prefix: str = "",
) -> list[str]:
    paths: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if key_text in forbidden_set:
                paths.append(path)
            paths.extend(
                _recursive_forbidden_field_paths(
                    value,
                    forbidden_set,
                    prefix=path,
                )
            )
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            paths.extend(
                _recursive_forbidden_field_paths(
                    value,
                    forbidden_set,
                    prefix=path,
                )
            )
    return paths


def _markdown_list(items: list[str]) -> list[str]:
    if not items:
        return ["- none"]
    return [f"- `{item}`" for item in items]


def _short_id(value: Any) -> str:
    text = str(value or "")
    if len(text) <= 18:
        return text
    return f"{text[:10]}...{text[-6:]}"


def _safety_report_fields() -> dict[str, Any]:
    return {
        **compact_safety_fields(),
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }


def _validate_safety_flags(obj: Any) -> None:
    for field_name, expected in _safety_report_fields().items():
        if field_name.startswith("#"):
            continue
        if hasattr(obj, field_name) and getattr(obj, field_name) is not expected:
            raise ValueError(f"{field_name} must be {expected}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
