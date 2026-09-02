"""Fail-closed orchestration evidence for the v8 pre-promotion boundary."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_regime_conditioned_ev import (
    CURRENT_75_ROW_REPLAY_RUN_ID,
    LATEST_ONE_HOUR_RECONCILED_RUN_ID,
)

PRE_PROMOTION_GOAL_CONFIG_SCHEMA_VERSION = (
    "bigan-v8-execution-layer-v2-pre-promotion-goal-config-v1"
)
PRE_PROMOTION_EXCLUSION_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-execution-layer-v2-pre-promotion-exclusion-manifest-v1"
)
PRE_PROMOTION_GOAL_STATE_SCHEMA_VERSION = (
    "bigan-v8-execution-layer-v2-pre-promotion-goal-state-v1"
)
PRE_PROMOTION_READINESS_REPORT_SCHEMA_VERSION = (
    "bigan-v8-execution-layer-v2-pre-promotion-readiness-report-v1"
)
PRE_PROMOTION_READINESS_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-execution-layer-v2-pre-promotion-readiness-manifest-v1"
)

EXPLICIT_EXCLUDED_RUN_IDS = (
    CURRENT_75_ROW_REPLAY_RUN_ID,
    LATEST_ONE_HOUR_RECONCILED_RUN_ID,
    LATEST_ONE_HOUR_RECONCILED_RUN_ID.removesuffix(
        "-clob-settlement-reconciled"
    ),
    "execution-layer-v2-one-hour-stage-aware-paper-20260711T073751Z",
    "execution-layer-v2-one-hour-stage-aware-outcome-reconciliation-final-20260711T085600Z",
)
EXCLUDED_RUN_NAME_FRAGMENTS = (
    "future-shadow",
    "future_holdout",
    "post-freeze",
    "schedule-debug",
    "settlement-debug",
)


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2PrePromotionGoalConfig:
    run_id: str
    output_dir: Path | str
    evidence_root: Path | str
    created_at: str
    starting_commit: str
    maximum_wall_clock_seconds: int = 18_000
    historical_collection_window_seconds: int = 3_600
    maximum_historical_collection_windows: int = 4
    collection_poll_interval_seconds: float = 60.0
    settlement_poll_max_wait_seconds: float = 600.0
    settlement_poll_interval_seconds: float = 15.0
    minimum_total_calibration_rows: int = 130
    minimum_total_calibration_markets: int = 30
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
    required_future_shadow_window_count: int = 2
    future_shadow_collection_window_seconds: int = 1_800
    minimum_future_shadow_rows: int = 30
    minimum_future_shadow_markets: int = 10
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if not self.starting_commit.strip():
            raise ValueError("starting_commit is required")
        datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        positive_fields = (
            "maximum_wall_clock_seconds",
            "historical_collection_window_seconds",
            "maximum_historical_collection_windows",
            "collection_poll_interval_seconds",
            "settlement_poll_max_wait_seconds",
            "settlement_poll_interval_seconds",
            "minimum_total_calibration_rows",
            "minimum_total_calibration_markets",
            "ridge_alpha",
            "entry_ev_threshold",
            "min_fit_rows",
            "min_validation_rows",
            "min_fit_markets",
            "min_validation_markets",
            "max_abs_coefficient",
            "bootstrap_samples",
            "required_future_shadow_window_count",
            "future_shadow_collection_window_seconds",
            "minimum_future_shadow_rows",
            "minimum_future_shadow_markets",
        )
        if any(float(getattr(self, field)) <= 0.0 for field in positive_fields):
            raise ValueError("goal budget and gate values must be positive")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between zero and 0.5")
        object.__setattr__(self, "output_dir", Path(self.output_dir).resolve())
        object.__setattr__(self, "evidence_root", Path(self.evidence_root).resolve())

    @property
    def goal_dir(self) -> Path:
        return Path(self.output_dir) / self.run_id / "pre_promotion_readiness"

    def frozen_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("overwrite_existing", None)
        payload["output_dir"] = str(self.output_dir)
        payload["evidence_root"] = str(self.evidence_root)
        payload.update(
            {
                "schema_version": PRE_PROMOTION_GOAL_CONFIG_SCHEMA_VERSION,
                "calibration_split_order": [
                    "historical_fit",
                    "validation",
                    "future_unseen_shadow",
                ],
                "market_condition_disjoint_required": True,
                "chronological_split_required": True,
                "no_validation_or_shadow_tuning": True,
                "subtract_execution_cost": False,
                "target_semantics": "settled_net_return_after_execution_cost",
                "stop_conditions": [
                    "PRE_PROMOTION_READY",
                    "configured_wall_clock_budget_reached",
                    "configured_data_window_budget_reached",
                    "non_sample_size_hard_gate_failed",
                    "public_provider_fail_closed",
                ],
                **_safety_fields(),
            }
        )
        return payload


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2PrePromotionInitializationResult:
    goal_dir: Path
    goal_configuration_path: Path
    goal_configuration_sha256_path: Path
    excluded_evidence_manifest_path: Path
    goal_state_path: Path
    goal_configuration_sha256: str


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2PrePromotionFinalizationConfig:
    goal_dir: Path | str
    historical_collection_dirs: tuple[Path | str, ...]
    outcome_reconciliation_dirs: tuple[Path | str, ...]
    calibration_corpus_dir: Path | str | None = None
    calibration_run_dir: Path | str | None = None
    stop_reason_codes: tuple[str, ...] = ()
    resumable_next_command: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_dir", Path(self.goal_dir).resolve())
        object.__setattr__(
            self,
            "historical_collection_dirs",
            tuple(Path(path).resolve() for path in self.historical_collection_dirs),
        )
        object.__setattr__(
            self,
            "outcome_reconciliation_dirs",
            tuple(Path(path).resolve() for path in self.outcome_reconciliation_dirs),
        )
        if self.calibration_corpus_dir is not None:
            object.__setattr__(
                self,
                "calibration_corpus_dir",
                Path(self.calibration_corpus_dir).resolve(),
            )
        if self.calibration_run_dir is not None:
            object.__setattr__(
                self,
                "calibration_run_dir",
                Path(self.calibration_run_dir).resolve(),
            )


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2PrePromotionFinalizationResult:
    goal_dir: Path
    readiness_report_path: Path
    readiness_manifest_path: Path
    readiness_manifest_sha256_path: Path
    final_state: str
    pre_promotion_readiness_complete: bool


def initialize_pre_promotion_readiness_goal(
    config: ExecutionLayerV2PrePromotionGoalConfig,
) -> ExecutionLayerV2PrePromotionInitializationResult:
    goal_dir = config.goal_dir
    if goal_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"pre-promotion goal already exists: {goal_dir}")
        shutil.rmtree(goal_dir)
    goal_dir.mkdir(parents=True)

    configuration_path = goal_dir / "goal_configuration.json"
    configuration_sha_path = goal_dir / "goal_configuration.sha256"
    exclusion_path = goal_dir / "excluded_evidence_manifest.json"
    state_path = goal_dir / "pre_promotion_goal_state.json"
    configuration = config.frozen_payload()
    _write_json(configuration_path, configuration)
    configuration_sha256 = _sha256_file(configuration_path)
    configuration_sha_path.write_text(configuration_sha256 + "\n", encoding="utf-8")

    exclusion_manifest = _excluded_evidence_manifest(
        evidence_root=Path(config.evidence_root),
        goal_configuration_sha256=configuration_sha256,
    )
    _write_json(exclusion_path, exclusion_manifest)
    state = {
        "schema_version": PRE_PROMOTION_GOAL_STATE_SCHEMA_VERSION,
        "run_id": config.run_id,
        "goal_configuration_sha256": configuration_sha256,
        "current_phase": "phase_0_audit_and_goal_configuration_complete",
        "completed_phases": ["phase_0_audit_and_goal_configuration"],
        "next_phase": "phase_1_collect_strict_causal_historical_corpus",
        "resumable": True,
        "goal_status": "IN_PROGRESS",
        "excluded_evidence_manifest_sha256": _sha256_file(exclusion_path),
        **_safety_fields(),
    }
    state["state_id"] = canonical_json_sha256(state)
    _write_json(state_path, state)
    return ExecutionLayerV2PrePromotionInitializationResult(
        goal_dir=goal_dir,
        goal_configuration_path=configuration_path,
        goal_configuration_sha256_path=configuration_sha_path,
        excluded_evidence_manifest_path=exclusion_path,
        goal_state_path=state_path,
        goal_configuration_sha256=configuration_sha256,
    )


def finalize_pre_promotion_readiness_goal(
    config: ExecutionLayerV2PrePromotionFinalizationConfig,
) -> ExecutionLayerV2PrePromotionFinalizationResult:
    """Seal a fail-closed pre-promotion bundle from immutable phase artifacts."""

    goal_dir = Path(config.goal_dir)
    goal_configuration_path = goal_dir / "goal_configuration.json"
    goal_configuration_sha_path = goal_dir / "goal_configuration.sha256"
    exclusion_path = goal_dir / "excluded_evidence_manifest.json"
    if not goal_configuration_path.is_file() or not goal_configuration_sha_path.is_file():
        raise FileNotFoundError("frozen goal configuration is missing")
    expected_config_hash = goal_configuration_sha_path.read_text(encoding="utf-8").strip()
    actual_config_hash = _sha256_file(goal_configuration_path)
    if actual_config_hash != expected_config_hash:
        raise ValueError("frozen goal configuration hash mismatch")
    if not exclusion_path.is_file():
        raise FileNotFoundError("excluded evidence manifest is missing")
    goal_configuration = _load_json(goal_configuration_path)

    collection_rows = [
        _phase_directory_evidence(path, "one_hour_remap_paper_goal_report.json")
        for path in config.historical_collection_dirs
    ]
    reconciliation_rows = [
        _phase_directory_evidence(path, "clob_settlement_reconciliation_report.json")
        for path in config.outcome_reconciliation_dirs
    ]
    historical_manifest = {
        "schema_version": "bigan-v8-pre-promotion-historical-collection-manifest-v1",
        "goal_configuration_sha256": expected_config_hash,
        "collection_window_count": len(collection_rows),
        "collection_windows": collection_rows,
        "complete_round_count": sum(
            int(row.get("report", {}).get("complete_round_count", 0))
            for row in collection_rows
        ),
        "paper_fill_count": sum(
            int(row.get("report", {}).get("paper_fill_count", 0))
            for row in collection_rows
        ),
        **_safety_fields(),
    }
    historical_manifest["manifest_id"] = canonical_json_sha256(historical_manifest)
    historical_path = goal_dir / "historical_collection_manifest.json"
    _write_json(historical_path, historical_manifest)

    unresolved_condition_count = sum(
        int(row.get("report", {}).get("unresolved_fill_count_after", 0))
        for row in reconciliation_rows
    )
    reconciliation_manifest = {
        "schema_version": "bigan-v8-pre-promotion-outcome-reconciliation-manifest-v1",
        "goal_configuration_sha256": expected_config_hash,
        "reconciliation_run_count": len(reconciliation_rows),
        "reconciliation_runs": reconciliation_rows,
        "unresolved_fill_count_after": unresolved_condition_count,
        "all_original_source_artifacts_immutable": all(
            not bool(row.get("report", {}).get("original_source_artifacts_mutated", True))
            for row in reconciliation_rows
        ),
        **_safety_fields(),
    }
    reconciliation_manifest["manifest_id"] = canonical_json_sha256(
        reconciliation_manifest
    )
    reconciliation_path = goal_dir / "outcome_reconciliation_manifest.json"
    _write_json(reconciliation_path, reconciliation_manifest)

    corpus_source = _optional_named_file(
        config.calibration_corpus_dir,
        "execution_layer_v2_regime_conditioned_ev_v2_corpus_manifest.json",
    )
    corpus_report_source = _optional_named_file(
        config.calibration_corpus_dir,
        "execution_layer_v2_regime_conditioned_ev_v2_corpus_quality_report.json",
    )
    corpus_payload = _load_json(corpus_source) if corpus_source else {}
    corpus_report = _load_json(corpus_report_source) if corpus_report_source else {}
    goal_corpus_gate_blockers = _goal_corpus_gate_blockers(
        goal_configuration, corpus_report
    )
    corpus_bundle_payload = {
        "schema_version": "bigan-v8-pre-promotion-calibration-corpus-manifest-v1",
        "goal_configuration_sha256": expected_config_hash,
        "source_manifest": _file_descriptor(corpus_source),
        "source_quality_report": _file_descriptor(corpus_report_source),
        "eligible_row_count": int(corpus_report.get("eligible_row_count", 0)),
        "rejected_row_count": int(corpus_report.get("excluded_row_count", 0)),
        "unique_market_count": int(corpus_report.get("unique_market_count", 0)),
        "corpus_sha256": corpus_payload.get("corpus_sha256"),
        "builder_advisory_readiness_reason_codes": corpus_report.get(
            "readiness_blocking_reason_codes", ["calibration_corpus_not_available"]
        ),
        "goal_corpus_gate_blocking_reason_codes": goal_corpus_gate_blockers,
        **_safety_fields(),
    }
    corpus_bundle_payload["manifest_id"] = canonical_json_sha256(
        corpus_bundle_payload
    )
    corpus_path = goal_dir / "calibration_corpus_manifest.json"
    _write_json(corpus_path, corpus_bundle_payload)

    calibration_dir = (
        Path(config.calibration_run_dir)
        if config.calibration_run_dir is not None
        else None
    )
    split_source = _optional_named_file(
        calibration_dir,
        "execution_layer_v2_regime_conditioned_ev_v2_split_report.json",
    )
    calibration_source = _optional_named_file(
        calibration_dir,
        "execution_layer_v2_regime_conditioned_ev_v2_calibration_report.json",
    )
    artifact_source = _optional_named_file(
        calibration_dir,
        "execution_layer_v2_frozen_regime_conditioned_ev_v2.json",
    )
    split_payload = _load_json(split_source) if split_source else {}
    calibration_payload = _load_json(calibration_source) if calibration_source else {}

    split_path = goal_dir / "split_manifest.json"
    _write_json(
        split_path,
        _bundle_report_payload(
            "bigan-v8-pre-promotion-split-manifest-v1",
            expected_config_hash,
            split_source,
            split_payload,
            "split_not_run",
        ),
    )
    fit_path = goal_dir / "fit_report.json"
    fit_summary = {
        "fit_metrics": calibration_payload.get("fit_metrics"),
        "fit_coefficients_hash": calibration_payload.get("fit_coefficients_hash"),
        "coefficients_finite_and_bounded": calibration_payload.get(
            "coefficients_finite_and_bounded"
        ),
        "coefficient_stability_metrics": calibration_payload.get(
            "coefficient_stability_metrics"
        ),
        "threshold_selection_source": calibration_payload.get(
            "threshold_selection_source"
        ),
        "uses_validation_labels_for_fitting": calibration_payload.get(
            "uses_validation_labels_for_fitting"
        ),
        "uses_validation_labels_for_threshold_selection": calibration_payload.get(
            "uses_validation_labels_for_threshold_selection"
        ),
    }
    _write_json(
        fit_path,
        _bundle_report_payload(
            "bigan-v8-pre-promotion-fit-report-v1",
            expected_config_hash,
            calibration_source,
            fit_summary,
            "fit_not_run",
        ),
    )
    validation_path = goal_dir / "validation_report.json"
    _write_json(
        validation_path,
        _bundle_report_payload(
            "bigan-v8-pre-promotion-validation-report-v1",
            expected_config_hash,
            calibration_source,
            calibration_payload,
            "validation_not_run",
        ),
    )

    artifact_created = bool(
        artifact_source
        and calibration_payload.get("artifact_created") is True
        and calibration_payload.get("statistical_eligibility_passed") is True
    )
    if artifact_created:
        frozen_path = goal_dir / "frozen_diagnostic_artifact.json"
        shutil.copyfile(artifact_source, frozen_path)
        (goal_dir / "frozen_diagnostic_artifact.sha256").write_text(
            _sha256_file(frozen_path) + "\n", encoding="utf-8"
        )

    future_shadow = calibration_payload.get("future_shadow", {})
    future_shadow_status = str(future_shadow.get("status", "not_started"))
    future_shadow_path = goal_dir / "future_shadow_manifest.json"
    _write_json(
        future_shadow_path,
        {
            "schema_version": "bigan-v8-pre-promotion-future-shadow-manifest-v1",
            "goal_configuration_sha256": expected_config_hash,
            "status": future_shadow_status,
            "frozen_artifact_created": artifact_created,
            "future_shadow": future_shadow,
            "blocking_reason_codes": (
                [] if future_shadow_status == "completed" else [
                    "future_shadow_not_completed"
                ]
            ),
            **_safety_fields(),
        },
    )
    future_shadow_evaluation_path = goal_dir / "future_shadow_evaluation_report.json"
    _write_json(
        future_shadow_evaluation_path,
        {
            "schema_version": (
                "bigan-v8-pre-promotion-future-shadow-evaluation-report-v1"
            ),
            "goal_configuration_sha256": expected_config_hash,
            "status": future_shadow_status,
            "evaluation": future_shadow,
            "promotion_evidence": False,
            **_safety_fields(),
        },
    )

    blockers = set(config.stop_reason_codes)
    blockers.update(goal_corpus_gate_blockers)
    blockers.update(calibration_payload.get("blocking_reason_codes", []))
    if unresolved_condition_count:
        blockers.add("unresolved_official_settlements_remaining")
    if not artifact_created:
        blockers.add("valid_frozen_diagnostic_artifact_not_created")
    shadow_complete = future_shadow_status == "completed"
    if artifact_created and not shadow_complete:
        blockers.add("required_future_unseen_shadow_not_completed")
    readiness_complete = artifact_created and shadow_complete and not blockers
    final_state = "PRE_PROMOTION_READY" if readiness_complete else "PRE_PROMOTION_BLOCKED"

    readiness_report = {
        "schema_version": PRE_PROMOTION_READINESS_REPORT_SCHEMA_VERSION,
        "goal_configuration_sha256": expected_config_hash,
        "final_state": final_state,
        "pre_promotion_readiness_complete": readiness_complete,
        "historical_collection_summary": {
            "collection_window_count": len(collection_rows),
            "complete_round_count": historical_manifest["complete_round_count"],
            "paper_fill_count": historical_manifest["paper_fill_count"],
        },
        "outcome_reconciliation_summary": {
            "reconciliation_run_count": len(reconciliation_rows),
            "unresolved_fill_count_after": unresolved_condition_count,
        },
        "calibration_corpus_summary": {
            "accepted_calibration_row_count": corpus_bundle_payload[
                "eligible_row_count"
            ],
            "rejected_calibration_row_count": corpus_bundle_payload[
                "rejected_row_count"
            ],
            "unique_market_count": corpus_bundle_payload["unique_market_count"],
            "corpus_sha256": corpus_bundle_payload["corpus_sha256"],
        },
        "split_summary": split_payload,
        "fit_validation_summary": calibration_payload,
        "frozen_diagnostic_artifact_created": artifact_created,
        "frozen_diagnostic_artifact_sha256": (
            _sha256_file(goal_dir / "frozen_diagnostic_artifact.json")
            if artifact_created
            else None
        ),
        "future_shadow_status": future_shadow_status,
        "passed_gate_names": _passed_gate_names(
            corpus_report, split_payload, calibration_payload
        ),
        "blocking_reason_codes": sorted(blockers),
        "residual_risks": [
            "pre_promotion_result_is_not_promotion_evidence",
            "future_live_and_capital_paths_remain_disabled",
        ],
        "resumable_next_command": config.resumable_next_command,
        **_safety_fields(),
    }
    readiness_report["report_id"] = canonical_json_sha256(readiness_report)
    readiness_report_path = goal_dir / "pre_promotion_readiness_report.json"
    _write_json(readiness_report_path, readiness_report)
    readiness_markdown_path = goal_dir / "pre_promotion_readiness_report.md"
    readiness_markdown_path.write_text(
        _readiness_markdown(readiness_report), encoding="utf-8"
    )

    state = {
        "schema_version": PRE_PROMOTION_GOAL_STATE_SCHEMA_VERSION,
        "goal_configuration_sha256": expected_config_hash,
        "current_phase": "phase_9_pre_promotion_readiness_bundle_complete",
        "completed_phases": [
            "phase_0_audit_and_goal_configuration",
            "phase_1_historical_collection",
            "phase_2_official_outcome_reconciliation",
            "phase_3_calibration_corpus",
            "phase_4_split",
            "phase_5_fit_validation",
            "phase_9_pre_promotion_readiness_bundle",
        ],
        "next_phase": (
            "promotion_evidence_not_started" if readiness_complete else "resume_blocker"
        ),
        "resumable": not readiness_complete,
        "goal_status": final_state,
        "blocking_reason_codes": sorted(blockers),
        **_safety_fields(),
    }
    state["state_id"] = canonical_json_sha256(state)
    _write_json(goal_dir / "pre_promotion_goal_state.json", state)

    artifact_paths = sorted(
        path
        for path in goal_dir.iterdir()
        if path.is_file()
        and path.name
        not in {
            "pre_promotion_readiness_manifest.json",
            "pre_promotion_readiness_manifest.sha256",
        }
    )
    manifest = {
        "schema_version": PRE_PROMOTION_READINESS_MANIFEST_SCHEMA_VERSION,
        "goal_configuration_sha256": expected_config_hash,
        "final_state": final_state,
        "pre_promotion_readiness_complete": readiness_complete,
        "blocking_reason_codes": sorted(blockers),
        "artifact_count": len(artifact_paths),
        "artifacts": [
            {
                "name": path.name,
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
            }
            for path in artifact_paths
        ],
        "manifest_self_hash_embedded": False,
        "manifest_hash_descriptor_external": True,
        **_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    readiness_manifest_path = goal_dir / "pre_promotion_readiness_manifest.json"
    _write_json(readiness_manifest_path, manifest)
    readiness_manifest_sha_path = (
        goal_dir / "pre_promotion_readiness_manifest.sha256"
    )
    readiness_manifest_sha_path.write_text(
        _sha256_file(readiness_manifest_path) + "\n", encoding="utf-8"
    )

    return ExecutionLayerV2PrePromotionFinalizationResult(
        goal_dir=goal_dir,
        readiness_report_path=readiness_report_path,
        readiness_manifest_path=readiness_manifest_path,
        readiness_manifest_sha256_path=readiness_manifest_sha_path,
        final_state=final_state,
        pre_promotion_readiness_complete=readiness_complete,
    )


def _excluded_evidence_manifest(
    *,
    evidence_root: Path,
    goal_configuration_sha256: str,
) -> dict[str, Any]:
    run_rows: list[dict[str, Any]] = []
    if evidence_root.exists():
        for run_dir in sorted(path for path in evidence_root.iterdir() if path.is_dir()):
            reason_codes = _exclusion_reasons(run_dir.name)
            if not reason_codes:
                continue
            files = sorted(path for path in run_dir.rglob("*") if path.is_file())
            artifact_rows = [
                {
                    "path": str(path.resolve()),
                    "relative_path": str(path.relative_to(run_dir)),
                    "sha256": _sha256_file(path),
                }
                for path in files
                if "manifest" in path.name or path.suffix in {".json", ".jsonl"}
            ]
            run_rows.append(
                {
                    "run_id": run_dir.name,
                    "run_dir": str(run_dir.resolve()),
                    "exclusion_reason_codes": reason_codes,
                    "artifact_count": len(files),
                    "audited_artifact_count": len(artifact_rows),
                    "audited_artifacts": artifact_rows,
                    "run_tree_sha256": canonical_json_sha256(
                        [
                            {
                                "relative_path": str(path.relative_to(run_dir)),
                                "sha256": _sha256_file(path),
                            }
                            for path in files
                        ]
                    ),
                }
            )
    payload = {
        "schema_version": PRE_PROMOTION_EXCLUSION_MANIFEST_SCHEMA_VERSION,
        "goal_configuration_sha256": goal_configuration_sha256,
        "explicit_excluded_run_ids": list(EXPLICIT_EXCLUDED_RUN_IDS),
        "excluded_run_name_fragments": list(EXCLUDED_RUN_NAME_FRAGMENTS),
        "excluded_run_count": len(run_rows),
        "excluded_runs": run_rows,
        "exclusion_applies_to": [
            "coefficient_fitting",
            "validation",
            "threshold_selection",
            "model_selection",
            "future_unseen_shadow_scoring",
        ],
        "diagnostic_uses_only": [
            "regression_testing",
            "pipeline_compatibility_checks",
            "immutable_historical_audit",
            "plumbing_numerical_equivalence",
        ],
        **_safety_fields(),
    }
    payload["manifest_id"] = canonical_json_sha256(payload)
    return payload


def _exclusion_reasons(run_id: str) -> list[str]:
    reasons = []
    if run_id in EXPLICIT_EXCLUDED_RUN_IDS:
        reasons.append("explicitly_excluded_diagnostic_or_debug_run")
    if any(fragment in run_id for fragment in EXCLUDED_RUN_NAME_FRAGMENTS):
        reasons.append("previously_inspected_shadow_holdout_or_debug_lineage")
    return reasons


def _safety_fields() -> dict[str, Any]:
    return {
        "diagnostic_only": True,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "promotion_evidence_stage_started": False,
        "live_evidence_stage_started": False,
        "live_evidence_allowed": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }


def _phase_directory_evidence(path: Path, report_name: str) -> dict[str, Any]:
    report_path = path / report_name
    if not report_path.is_file():
        matches = sorted(path.rglob(report_name)) if path.is_dir() else []
        report_path = matches[0] if matches else report_path
    report = _load_json(report_path) if report_path.is_file() else {}
    files = sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else []
    return {
        "run_dir": str(path),
        "run_dir_exists": path.is_dir(),
        "report": report,
        "report_artifact": _file_descriptor(report_path if report_path.is_file() else None),
        "manifest_artifacts": [
            _file_descriptor(item) for item in files if "manifest" in item.name
        ],
        "run_tree_sha256": canonical_json_sha256(
            [
                {
                    "relative_path": str(item.relative_to(path)),
                    "sha256": _sha256_file(item),
                }
                for item in files
            ]
        ),
    }


def _optional_named_file(root: Path | str | None, name: str) -> Path | None:
    if root is None:
        return None
    candidate = Path(root) / name
    return candidate if candidate.is_file() else None


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _file_descriptor(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def _bundle_report_payload(
    schema_version: str,
    goal_configuration_sha256: str,
    source_path: Path | None,
    source_payload: dict[str, Any],
    missing_reason_code: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": schema_version,
        "goal_configuration_sha256": goal_configuration_sha256,
        "source_artifact": _file_descriptor(source_path),
        "status": "available" if source_path is not None else "not_run_fail_closed",
        "blocking_reason_codes": [] if source_path is not None else [missing_reason_code],
        "report": source_payload,
        **_safety_fields(),
    }
    payload["report_id"] = canonical_json_sha256(payload)
    return payload


def _goal_corpus_gate_blockers(
    goal_configuration: dict[str, Any],
    corpus_report: dict[str, Any],
) -> list[str]:
    blockers = []
    if not corpus_report:
        return ["calibration_corpus_not_available"]
    if int(corpus_report.get("eligible_row_count", 0)) < int(
        goal_configuration["minimum_total_calibration_rows"]
    ):
        blockers.append("minimum_goal_calibration_row_support_not_met")
    if int(corpus_report.get("unique_market_count", 0)) < int(
        goal_configuration["minimum_total_calibration_markets"]
    ):
        blockers.append("minimum_goal_calibration_market_support_not_met")
    provenance = corpus_report.get("provenance_coverage", {})
    if int(provenance.get("violation_count", 0)):
        blockers.append("calibration_corpus_provenance_violation")
    deduplication = corpus_report.get("deduplication", {})
    if int(deduplication.get("conflicting_identity_count", 0)):
        blockers.append("calibration_corpus_conflicting_identity")
    if corpus_report.get("incremental_full_rebuild_hash_match") is not True:
        blockers.append("calibration_corpus_rebuild_hash_mismatch")
    return blockers


def _passed_gate_names(*payloads: dict[str, Any]) -> list[str]:
    return sorted(
        {
            key
            for payload in payloads
            for key, value in payload.items()
            if isinstance(value, bool) and value and ("pass" in key or "verified" in key)
        }
    )


def _readiness_markdown(report: dict[str, Any]) -> str:
    corpus = report["calibration_corpus_summary"]
    reconciliation = report["outcome_reconciliation_summary"]
    blockers = report["blocking_reason_codes"]
    lines = [
        "# v8 Execution Layer v2 Pre-Promotion Readiness",
        "",
        f"- Final state: `{report['final_state']}`",
        "- Promotion evidence stage started: `false`",
        "- Promotion evidence eligible: `false`",
        "- Live evidence allowed: `false`",
        f"- Accepted calibration rows: `{corpus['accepted_calibration_row_count']}`",
        f"- Unique calibration markets: `{corpus['unique_market_count']}`",
        f"- Unresolved official settlements: `{reconciliation['unresolved_fill_count_after']}`",
        f"- Frozen diagnostic artifact created: `{str(report['frozen_diagnostic_artifact_created']).lower()}`",
        f"- Future shadow status: `{report['future_shadow_status']}`",
        "",
        "## Blocking Reasons",
        "",
    ]
    lines.extend(f"- `{reason}`" for reason in blockers)
    if not blockers:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Safety Boundary",
            "",
            "This bundle stops before promotion evidence. It does not authorize paper, live, wallet, write, capital, freeze, promotion, or execution handoff paths.",
            "",
            "## Resumable Command",
            "",
            f"`{report.get('resumable_next_command') or 'none'}`",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "ExecutionLayerV2PrePromotionFinalizationConfig",
    "ExecutionLayerV2PrePromotionFinalizationResult",
    "ExecutionLayerV2PrePromotionGoalConfig",
    "ExecutionLayerV2PrePromotionInitializationResult",
    "finalize_pre_promotion_readiness_goal",
    "initialize_pre_promotion_readiness_goal",
    "utc_now_iso",
]
