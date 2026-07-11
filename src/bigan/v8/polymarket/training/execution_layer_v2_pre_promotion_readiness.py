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
    "ExecutionLayerV2PrePromotionGoalConfig",
    "ExecutionLayerV2PrePromotionInitializationResult",
    "initialize_pre_promotion_readiness_goal",
    "utc_now_iso",
]
